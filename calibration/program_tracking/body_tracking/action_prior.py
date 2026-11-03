"""Real GVHMR camera-pose inference for explicitly uniform source clips."""
from pathlib import Path
import argparse,json,time
import numpy as np
from .rebuild_prepare import load,digest


def infer(prepared,output):
    from .body_prior import (_upstream_runtime,GVHMR_CHECKPOINT,HMR2_CHECKPOINT,
        SMPLX_MODEL,IMAGE_MEAN,IMAGE_STD)
    prepared=Path(prepared).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    data=load(prepared);pts=data['source_pts'];n=len(pts)
    if n<2 or pts.dtype.kind not in 'iu' or not np.all(np.diff(pts)==512):
        raise ValueError('Require one continuous native 25Hz sequence of real 1/12800 source PTS')
    expected={'rgb_crop256':(n,256,256,3),'rect_coco17':(n,17,3),
        'bbox_rect_xys':(n,3),'K_rect':(n,3,3),'R_w2c':(n,3,3)}
    for name,shape in expected.items():
        if data[name].shape!=shape or not np.isfinite(data[name]).all():raise ValueError('Invalid prepared '+name)
    log=output/'inference_progress.jsonl';start=time.perf_counter()
    def event(name,**fields):
        row=dict(event=name,seconds=time.perf_counter()-start,**fields)
        with log.open('a',encoding='utf-8')as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    torch=_upstream_runtime()
    from hmr4d.utils.preproc.vitfeat_extractor import Extractor
    from hmr4d.utils.geo.hmr_cam import normalize_kp2d
    from hmr4d.utils.geo_transform import compute_cam_angvel
    from hydra import compose,initialize_config_module
    from hydra.utils import instantiate
    import hmr4d.model.gvhmr.gvhmr_pl_demo
    import hmr4d.model.gvhmr.utils.endecoder
    import hmr4d.network.gvhmr.relative_transformer
    torch.set_grad_enabled(False)
    event('real_inference_started',prepared=str(prepared),prepared_sha256=digest(prepared),
        frames=n,source_pts=[int(pts[0]),int(pts[-1])],postproc=False,static_cam=False)
    extractor=Extractor(tqdm_leave=False);features=np.empty((n,1024),np.float32)
    for first in range(0,n,32):
        stop=min(first+32,n);rgb=data['rgb_crop256'][first:stop].astype(np.float32)
        crops=torch.from_numpy(((rgb/255.-IMAGE_MEAN)/IMAGE_STD).transpose(0,3,1,2))
        box=torch.from_numpy(data['bbox_rect_xys'][first:stop].copy()).float()
        with torch.inference_mode():features[first:stop]=extractor.extract_video_features(crops,box).cpu().numpy()
        event('image_features',completed=stop,total=n)
    del extractor;torch.cuda.empty_cache()
    with initialize_config_module(version_base='1.3',config_module='hmr4d.configs'):
        cfg=compose(config_name='demo',overrides=['video_name=rinkspace_action_clip'])
    unit=instantiate(cfg.model,_recursive_=False);unit.load_pretrained_model(GVHMR_CHECKPOINT)
    unit.eval().requires_grad_(False).cuda()
    rotations=torch.from_numpy(data['R_w2c']).float()
    boxes=torch.from_numpy(data['bbox_rect_xys']).float()
    joints=torch.from_numpy(data['rect_coco17']).float()
    batch=dict(length=torch.tensor([n],dtype=torch.long),obs=normalize_kp2d(joints,boxes)[None],
        bbx_xys=boxes[None],K_fullimg=torch.from_numpy(data['K_rect']).float()[None],
        cam_angvel=compute_cam_angvel(rotations)[None],f_imgseq=torch.from_numpy(features)[None])
    batch={k:v.cuda()for k,v in batch.items()}
    with torch.inference_mode():result=unit.pipeline.forward(batch,train=False,postproc=False,static_cam=False)
    incam={k:v[0].detach().cpu().numpy()for k,v in result['pred_smpl_params_incam'].items()}
    for name,values in incam.items():
        if not np.isfinite(values).all():raise ValueError('Nonfinite GVHMR '+name)
    np.savez_compressed(output/'camera_prior.npz',pts=pts,source_pts=pts,R_w2c=data['R_w2c'],K_rect=data['K_rect'],
        **incam)
    record=dict(schema='rinkspace.gvhmr.action-camera-prior.v1',prepared=str(prepared),prepared_sha256=digest(prepared),
        output=str(output/'camera_prior.npz'),output_sha256=digest(output/'camera_prior.npz'),
        source_time_base=[1,12800],pts_step=512,model_slots=n,one_continuous_model_call=True,
        checkpoint_sha256=digest(GVHMR_CHECKPOINT),hmr2_checkpoint_sha256=digest(HMR2_CHECKPOINT),
        smplx_model_sha256=digest(SMPLX_MODEL),postproc=False,static_cam=False,
        world_output_used=False,network_translation_used_for_world=False,
        model_time_contract='uniform 25Hz real source sampling; native model frame-index features, no fabricated timestamps',
        source_time_compatibility='25Hz adaptation is explicit; no claim of retrained/calibrated temporal model',
        roi_proposal_slots=np.flatnonzero(data.get('bbox_inferred',np.zeros(n,bool))).tolist(),
        seconds=time.perf_counter()-start)
    (output/'camera_prior.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    event('real_inference_complete',output=record['output'],frames=n)
    return record


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(argv);infer(a.prepared,a.output)


if __name__=='__main__':main()
