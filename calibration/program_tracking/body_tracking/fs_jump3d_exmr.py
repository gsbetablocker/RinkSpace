"""Frozen ExMR camera-frame inference for one complete real FS trial.

This module consumes one camera only. It never loads QTM reference points or
other camera videos, and does not execute ground/static/world postprocessing.
"""
from pathlib import Path
import argparse,json,time,gc
import numpy as np
from . import exmr_prior as prior
from .fs_jump3d_validation import read,write,sha,camera,clocks

def source_detector(data,output):
    """Compatibility for the already-local trusted Ultralytics checkpoint only."""
    import torch
    from rinkspace.vision_instances import SourcePersonBoxes
    weight=(Path(data)/'runtime_tools/body/GVHMR/inputs/checkpoints/yolo/yolov8x.pt').resolve();original=torch.load
    def known_load(file,*args,**kwargs):
        if isinstance(file,(str,Path))and Path(file).resolve()==weight:kwargs.setdefault('weights_only',False)
        return original(file,*args,**kwargs)
    try:
        torch.load=known_load
        return SourcePersonBoxes(weight,runtime_directory=Path(output)/'yolo_runtime',device='cuda')
    finally:torch.load=original

def prepare(data,video,calibration,camera_id,output):
    torch=prior.runtime(data)
    import cv2
    from rinkspace.vision_instances import SourcePersonBoxes
    output=Path(output);frames=output/'frames';frames.mkdir(parents=True,exist_ok=True)
    if any(frames.glob('*.png')):raise FileExistsError('Preserve already prepared source crops')
    log=prior.Progress(output,'fs_prepare');stream,pts=clocks(video);base=[int(v)for v in stream['time_base'].split('/')]
    cam=camera(read(calibration)[f'fs_camera_{camera_id}']);K0=cam['K'];size=1024;source_hash=sha(video)
    detector=source_detector(data,output)
    cap=cv2.VideoCapture(str(video));boxes=[];raw_boxes=[];affines=[];Ks=[];scores=[];frame_hashes=[];previous=None
    # Rectification uses frozen released K/distortion, not fitted body points.
    for i,pt in enumerate(pts):
        ok,bgr=cap.read()
        if not ok:raise RuntimeError('Missing original video exposure '+str(i))
        frame_hashes.append(__import__('hashlib').sha256(bgr.tobytes()).hexdigest())
        binding=dict(source_sha256=source_hash,source_pts=int(pt),time_base=stream['time_base'],shot_id='FS_Skater_A_Axel_1_camera'+str(camera_id),image_domain='native_brown_pixels')
        candidates=detector.read(bgr[:,:,::-1],binding)
        if candidates:
            bb=np.asarray([x['box_xyxy']for x in candidates]);area=np.prod(bb[:,2:]-bb[:,:2],axis=1)
            if previous is None:chosen=int(np.argmax(area))
            else:
                centre=(bb[:,:2]+bb[:,2:])/2;last=(previous[:2]+previous[2:])/2
                chosen=int(np.argmin(np.linalg.norm(centre-last,axis=1)/np.sqrt(np.maximum(area,1.))))
            raw=bb[chosen];score=candidates[chosen]['provenance']['confidence'];previous=raw.copy()
        elif previous is not None:raw=previous.copy();score=0.
        else:raise RuntimeError('No source-only initial person box; inspect input view, never use reference bbox')
        x0,y0,x1,y1=raw;t=np.linspace(0,1,9);border=np.concatenate([np.c_[x0+(x1-x0)*t,np.full(9,y0)],np.c_[x0+(x1-x0)*t,np.full(9,y1)],np.c_[np.full(9,x0),y0+(y1-y0)*t],np.c_[np.full(9,x1),y0+(y1-y0)*t]])
        rect=cv2.undistortPoints(border[:,None,:],K0,cam['distortion_opencv'],P=K0)[:,0,:];lo,hi=rect.min(0),rect.max(0);center=(lo+hi)/2;wh=hi-lo;model_side=1.2*max(wh[1],wh[0]/.75);extent=max(1.5*max(wh),model_side);scale=(size-1)/extent
        A=np.array([[scale,0,(size-1)/2-scale*center[0]],[0,scale,(size-1)/2-scale*center[1]],[0,0,1.]])
        K=A@K0;mx,my=cv2.initUndistortRectifyMap(K0,cam['distortion_opencv'],np.eye(3),K,(size,size),cv2.CV_32FC1);image=cv2.remap(bgr,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        if not cv2.imwrite(str(frames/f'{i:06d}.png'),image):raise RuntimeError('Cannot save inference crop')
        raw_boxes.append(np.r_[prior.transform_pixels(lo,A),prior.transform_pixels(hi,A)]);boxes.append(np.r_[prior.transform_pixels(center,A),model_side*scale]);affines.append(A);Ks.append(K);scores.append(score)
        if i%30==0 or i==len(pts)-1:log('source_only_input',completed=i+1,total=len(pts),detector_confidence=float(score))
    cap.release();u,_,vt=np.linalg.svd(cam['R_w2c']);R=u@vt
    np.savez_compressed(output/'prepared.npz',source_pts=pts,source_time_base=np.asarray(base),bbox_xys=np.asarray(boxes,np.float32),bbox_xyxy=np.asarray(raw_boxes,np.float32),K_input=np.asarray(Ks,np.float32),rect_to_input_affine=np.asarray(affines),R_w2c=np.repeat(R[None],len(pts),0),K_rect=np.repeat(K0[None],len(pts),0),coco17=np.zeros((len(pts),17,3),np.float32),source_person_detection_score=np.asarray(scores),source_frame_bgr_sha256=np.asarray(frame_hashes),actual_source_frequency_hz=np.array(60.))
    write(output/'input_identity.json',dict(video=str(video),video_sha256=source_hash,camera=int(camera_id),frames=len(pts),source_time_base=base,first_last_pts=pts[[0,-1]].tolist(),detector_sha256=detector.weights_sha256,source_only=True,reference_points_used=False,holdout_videos_used=False,rotation_serialization_SO3_projection_max=float(np.max(abs(R-cam['R_w2c']))),rotation_note='released six-decimal R retained raw in calibration; nearest proper SO3 only for network camera angular-velocity input; no camera fit',crops='native original pixels -> released Brown inverse -> per-frame known affine, original actual frame retained by hash',network_fps=60.,missing_box_count=int((np.asarray(scores)==0).sum())))
    log('complete',frames=len(pts),input_camera=int(camera_id))

def infer(data,output):
    torch=prior.runtime(data)
    from exmr.model.exmr import ExMR
    from exmr.preprocessing.keypoints import VISIBILITY_THRESHOLD
    from exmr.geometry.rotation_conversions import matrix_to_rotation_6d
    output=Path(output);d=prior.load(output/'prepared.npz');k=prior.load(output/'keypoints.npz');b=prior.load(output/'body_features.npz');h=prior.load(output/'hand_features.npz');pts=d['source_pts'];log=prior.Progress(output,'fs_exmr')
    for other in (k,b,h):np.testing.assert_array_equal(pts,other['source_pts'])
    checkpoint=Path(data)/'cache/pretrained/exmr/exmr/exmr.safetensors';smpl=Path(data)/'runtime_tools/body/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz';model=ExMR.from_pretrained(checkpoint,smpl,device='cuda')
    R=torch.from_numpy(d['R_w2c']).float();ang=matrix_to_rotation_6d(R[1:]@R[:-1].mT);ang=torch.cat([ang,ang[-1:]],0);points=k['coco63_native_network'][:,:23].copy();points[:,:,2]=(points[:,:,2]>VISIBILITY_THRESHOLD['rtmw']).astype(np.float32)
    inputs={name:torch.from_numpy(value).float()[None].cuda()for name,value in dict(kp2d=points,bbx_xys=d['bbox_xys'],K_fullimg=d['K_input'],cam_angvel=ang.numpy(),body_feat=b['body_feat'],hand_feat=h['hand_feat']).items()}
    log('inference_started',frames=len(pts),fps=60.,camera_only=True,training=False)
    with torch.inference_mode():result=model(**inputs,postprocess=False,snap_to_ground=False,smooth_incam=False,smooth_global=False,camera_only=True,fps=60.)
    for suffix,key in [('prefoot','smplx_incam_prefoot'),('postfoot','smplx_incam_postfoot'),('final','smplx_incam')]:
        cpu={name:tensor[0].cpu().numpy()for name,tensor in result[key].items()};path=output/('exmr_'+suffix+'.npz')
        if path.exists():raise FileExistsError(path)
        np.savez_compressed(path,source_pts=pts,source_time_base=d['source_time_base'],**cpu)
    # Current S05 uses HandMR but no FootMR. Construct that exact frozen-network
    # ablation from pre-foot ankle values plus final hand/elbow/wrist output.
    final={k:v[0].clone()for k,v in result['smplx_incam'].items()};final['body_pose'][:,18:24]=result['smplx_incam_prefoot']['body_pose'][0,:,18:24]
    np.savez_compressed(output/'exmr_hand_nofoot.npz',source_pts=pts,source_time_base=d['source_time_base'],**{k:v.cpu().numpy()for k,v in final.items()})
    np.savez_compressed(output/'static_network_diagnostic.npz',source_pts=pts,static_conf_logits=result['static_conf_logits'][0].cpu().numpy(),used_for_contact_truth=np.array(False))
    joint_packets={}
    for name,params in [('prefoot',result['smplx_incam_prefoot']),('postfoot',result['smplx_incam_postfoot']),('final',result['smplx_incam']),('hand_nofoot',{k:v[None]for k,v in final.items()})]:
        pieces=[]
        for first in range(0,len(pts),32):
            with torch.inference_mode():joints=model.body_model.joints(**{k:v[0,first:first+32]for k,v in params.items()})
            pieces.append(joints.cpu().numpy())
        joint_packets[name+'_joints_camera_m']=np.concatenate(pieces)
    np.savez_compressed(output/'all_camera_joints.npz',source_pts=pts,source_time_base=d['source_time_base'],joint_names=np.asarray(model.body_model.joint_names),**joint_packets)
    write(output/'inference.json',dict(model='ExMR',commit=prior.EXMR_COMMIT,checkpoint_sha256=sha(checkpoint),SMPLX_sha256=sha(smpl),frames=len(pts),fps=60.,source_time_base=d['source_time_base'].tolist(),one_continuous_model_call=True,camera_only=True,world_branch=False,postprocess=False,snap_to_ground=False,smooth_incam=False,smooth_global=False,orientation_normalization=False,trained=False,reference_or_holdout_used=False,network_static_probabilities_are_contact_truth=False,candidates=['prefoot','postfoot','final','hand_nofoot'],temporal_limit='60Hz actual source cadence; no sports fine-tuning or cadence adapter',peak_cuda_bytes=int(torch.cuda.max_memory_allocated())))
    log('complete',frames=len(pts),candidates=4)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=['prepare','keypoints','sam','hands','infer']);ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--video',type=Path);ap.add_argument('--calibration',type=Path);ap.add_argument('--camera',type=int,default=7);ap.add_argument('--batch-size',type=int,default=8);a=ap.parse_args()
    if a.stage=='prepare':prepare(a.data_root,a.video,a.calibration,a.camera,a.output)
    elif a.stage=='infer':infer(a.data_root,a.output)
    else:{'keypoints':prior.keypoints,'sam':prior.sam_features,'hands':prior.hand_features}[a.stage](a.data_root,a.output,a.batch_size)

if __name__=='__main__':main()
