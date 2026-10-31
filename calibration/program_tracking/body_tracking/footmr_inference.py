"""Source-clock WholeBody extraction and official local FootMR inference.

No SLAM, world translation, IK, foot locking or ground snapping is imported.
Raw model observations remain separate from previously admitted source pixels.
"""
from pathlib import Path
from rinkspace.storage import AssetStore
import argparse,hashlib,importlib.metadata,json,time,sys
import numpy as np
from .rebuild_prepare import digest,load
from .footmr_contract import COCO23,check_input,export_candidate


def wholebody(root,model,output):
    import av,cv2,onnxruntime as ort
    root,model,output=map(Path,[root,model,output]);model=AssetStore(Path(__file__).resolve().parents[3]).source(model.resolve());output.mkdir(parents=True,exist_ok=True);p=load(root/'input.npz');final=output/'wholebody_coco23.npz'
    if final.exists():raise FileExistsError(final)
    config=dict(source_input_sha256=digest(root/'input.npz'),model_sha256=digest(model),model_path=str(model.resolve()),backend='RTMW-l 384x288 ONNX; RTMLib',onnxruntime=importlib.metadata.version('onnxruntime'),rtmlib=importlib.metadata.version('rtmlib'),device='CPU',intra_op_threads=2,source_pts_unchanged=True,original_observations_overwritten=False,anatomical_points_on_boots_are_network_candidates=True)
    (output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf8');original_session=ort.InferenceSession
    def constrained_session(*a,**kw):
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1;options.log_severity_level=3;kw['sess_options']=options;return original_session(*a,**kw)
    ort.InferenceSession=constrained_session
    from rtmlib import RTMPose
    cv2.setNumThreads(2);pose=RTMPose(str(model),model_input_size=(288,384),backend='onnxruntime',device='cpu',to_openpose=False);ort.InferenceSession=original_session
    actual=pose.session.get_inputs()[0].shape
    if actual[-2:] not in [[384,288],['height','width']]:raise ValueError('RTMW input shape disagrees with selected model '+str(actual))
    raw=np.zeros((len(p['pts']),133,3),np.float32);seen=np.zeros(len(p['pts']),bool);boxes=p['source_bbox'].copy();valid=np.isfinite(boxes).all(1)&np.all(boxes[:,2:]>boxes[:,:2],1)
    for j in range(4):boxes[~valid,j]=np.interp(p['pts'][~valid],p['pts'][valid],boxes[valid,j])
    start=time.perf_counter()
    with (output/'inference_progress.jsonl').open('w',encoding='utf8',buffering=1) as log, av.open(str(root/'offline_review/source_s01.mp4')) as video:
        log.write(json.dumps(dict(event='start',**config,onnx_input_shape=actual))+'\n');stream=video.streams.video[0];stream.codec_context.thread_count=2
        for frame in video.decode(stream):
            i=int(np.searchsorted(p['pts'],frame.pts))
            if i>=len(seen) or p['pts'][i]!=frame.pts:continue
            uv,score=pose(frame.to_ndarray(format='bgr24'),bboxes=[boxes[i].tolist()]);assert uv.shape==(1,133,2) and score.shape==(1,133)
            raw[i,:,:2]=uv[0];raw[i,:,2]=score[0];seen[i]=True
            if seen.sum()%25==0:
                log.write(json.dumps(dict(event='progress',source_pts=int(frame.pts),frames=int(seen.sum()),seconds=time.perf_counter()-start))+'\n');np.savez_compressed(output/'partial.pending.npz',pts=p['pts'],raw_wholebody133=raw,processed=seen);(output/'partial.pending.npz').replace(output/'partial.npz')
        if not seen.all():raise ValueError('Source exposures missing during WholeBody extraction')
        usable=raw[:,:23].copy();inside=(usable[:,:,0]>=0)&(usable[:,:,0]<1920)&(usable[:,:,1]>=0)&(usable[:,:,1]<1080);usable[:,:,2]*=inside;usable[p['lower_body_inferred'],15:,2]=0
        np.savez_compressed(final,pts=p['pts'],source_time_base=np.array([1,12800]),pixel_space=np.array('original_full_image'),keypoint_names=np.asarray(COCO23),kp2d_coco23=usable,raw_wholebody133=raw,source_bbox_model=boxes,source_bbox_inferred=~valid,source_lower_body_inferred=p['lower_body_inferred'],human_reviewed=np.zeros(len(seen),bool),model_sha256=np.array(config['model_sha256']))
        report=dict(**config,model_executed=True,source_exposures=len(seen),seconds=time.perf_counter()-start,original_pts_preserved=True,foot_points_above_0p5=(usable[:,17:23,2]>.5).sum(0).tolist(),lower_body_inferred_frames=int(p['lower_body_inferred'].sum()),foot_points_are_directly_visible_anatomy=False,contact_allowed=False)
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8');log.write(json.dumps(dict(event='complete',**report))+'\n')
    print(json.dumps(report),flush=True);return final


def refine(contract,checkpoint,code_root,gvhmr_root,output):
    import torch
    contract,checkpoint,code_root,gvhmr_root,output=map(Path,[contract,checkpoint,code_root,gvhmr_root,output]);output.mkdir(parents=True,exist_ok=True)
    project=Path(__file__).resolve().parents[3]
    checkpoint,code_root,gvhmr_root=[AssetStore(project).source(p.resolve()) for p in (checkpoint,code_root,gvhmr_root)]
    checked=check_input(contract);data=load(contract)
    if not checked['any_new_toe_heel_evidence']:raise ValueError('FootMR input does not contain supported six-foot-point candidates')
    if (output/'prediction.npz').exists():raise FileExistsError(output/'prediction.npz')
    sys.path[:0]=[str(code_root.resolve()),str(gvhmr_root.resolve())];torch.set_num_threads(2)
    from hmr4d.network.footmr.foot_transformer import FootEncoderRoPE
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True);state=saved.get('state_dict',saved);prefix='pipeline.foot_motion_refiner.';weights={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
    if not weights:raise ValueError('Official FootMR state prefix missing')
    model=FootEncoderRoPE().cpu().eval();model.load_state_dict(weights,strict=True);start=time.perf_counter()
    config=dict(model_commit='530da95641f906b42ba6792df8e55c1dd667f98d',checkpoint_sha256=digest(checkpoint),input_contract_sha256=digest(contract),device='CPU',torch=torch.__version__,strict_state_tensors=len(weights),model_executed=False,model_grid_fps=float(data['model_fps']),output_already_contains_initial_ankle_rotation=True)
    with (output/'inference_progress.jsonl').open('w',encoding='utf8',buffering=1) as log:
        log.write(json.dumps(dict(event='loaded',**config))+'\n')
        obs=torch.tensor(data['foot_obs_model'][None],dtype=torch.float32);cliff=torch.tensor(data['f_cliffcam_model'][None],dtype=torch.float32);rotation=torch.tensor(data['global_rot6d_model'][None],dtype=torch.float32);length=torch.tensor([obs.shape[1]],dtype=torch.long)
        with torch.inference_mode():prediction=model(length,obs,cliff,rotation)[0].numpy().reshape(-1,2,6)
        np.savez_compressed(output/'prediction.npz',model_relative_seconds=data['model_relative_seconds'],global_ankle_rot6d=prediction,input_contract_sha256=np.array(digest(contract)),model_executed=np.array(True))
        result=export_candidate(contract,output/'prediction.npz',output/'candidate_50hz.npz');config.update(model_executed=True,model_frames=len(prediction),seconds=time.perf_counter()-start,candidate=result,root_updated=False,camera_updated=False,contact_feedback=False)
        (output/'report.json').write_text(json.dumps(config,indent=2),encoding='utf8');log.write(json.dumps(dict(event='complete',**config))+'\n')
    print(json.dumps(config),flush=True);return output/'candidate_50hz.npz'


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='action',required=True);q=s.add_parser('wholebody');q.add_argument('--root',required=True);q.add_argument('--model',required=True);q.add_argument('--output',required=True);q=s.add_parser('refine');q.add_argument('--contract',required=True);q.add_argument('--checkpoint',required=True);q.add_argument('--code-root',required=True);q.add_argument('--gvhmr-root',required=True);q.add_argument('--output',required=True);a=p.parse_args()
    if a.action=='wholebody':wholebody(a.root,a.model,a.output)
    else:refine(a.contract,a.checkpoint,a.code_root,a.gvhmr_root,a.output)
