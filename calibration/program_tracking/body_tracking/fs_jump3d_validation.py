"""Real full-trial FS acquisition checks and single-camera frozen-model validation.

Original all-marker QTM data and clocks are preserved. Reference data are used
only on the evaluation side; no per-frame alignment or holdout fitting occurs.
"""
from __future__ import annotations
import argparse,json,time,hashlib,subprocess
from pathlib import Path
import numpy as np

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        while b:=f.read(4<<20):h.update(b)
    return h.hexdigest()

def qtm_reference(path):
    data=read(path);start=data['Timebase']['Range']['Start'];end=data['Timebase']['Range']['End'];n=end-start+1
    names=[m['Name']for m in data['Markers']];world=np.full((n,len(names),3),np.nan);fourth=np.full((n,len(names)),np.nan);valid=np.zeros((n,len(names)),bool)
    for i,marker in enumerate(data['Markers']):
        for part in marker['Parts']:
            values=np.asarray(part['Values'],float);a=part['Range']['Start']-start;b=part['Range']['End']-start+1
            if values.shape!=(b-a,4)or a<0 or b>n:raise ValueError('Original QTM part/range mismatch '+marker['Name'])
            if valid[a:b,i].any():raise ValueError('Overlapping original QTM parts')
            world[a:b,i]=values[:,:3];fourth[a:b,i]=values[:,3];valid[a:b,i]=np.isfinite(values[:,:3]).all(1)&(values[:,3]>=0)
    return dict(world_points_mm=world,original_fourth_channel=fourth,valid_mask=valid,names=np.asarray(names),source_frame=np.arange(start,end+1,dtype=np.int64),reference_source_pts=np.arange(n,dtype=np.int64),reference_time_base=np.array([1,round(data['Timebase']['Frequency'])]),source_metadata_json=np.array(json.dumps({k:v for k,v in data.items()if k!='Markers'})),marker_metadata_json=np.array(json.dumps([{k:v for k,v in m.items()if k!='Parts'}for m in data['Markers']])))

def camera(raw):
    R=np.diag([1.,-1.,-1.])@np.asarray(raw['extrinsic_matrix']);C=np.asarray(raw['xyz'])/1000.;K=np.asarray(raw['affine_intrinsics_matrix']);d=np.asarray(raw['distortion'])
    # AP3D gendb.py: first3 radial, final2 tangential. OpenCV order differs.
    return dict(R_w2c=R,C_world_m=C,t_w2c_m=-R@C,K=K,distortion_opencv=d[[0,1,3,4,2]],distortion_original=d)

def project(world_m,cam):
    X=(np.asarray(world_m)-cam['C_world_m'])@cam['R_w2c'].T
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        xy=X[...,:2]/X[...,2:];x,y=xy[...,0],xy[...,1];r2=x*x+y*y;k1,k2,p1,p2,k3=cam['distortion_opencv'];rad=1+k1*r2+k2*r2*r2+k3*r2*r2*r2
        xx=x*rad+2*p1*x*y+p2*(r2+2*x*x);yy=y*rad+p1*(r2+2*y*y)+2*p2*x*y
        uv=np.stack([xx*cam['K'][0,0]+cam['K'][0,2],yy*cam['K'][1,1]+cam['K'][1,2]],-1)
    return uv,np.isfinite(uv).all(-1)&(X[...,2]>0)

def clocks(video):
    command=['ffprobe','-v','error','-select_streams','v:0','-show_streams','-show_frames','-show_entries','stream=width,height,time_base,r_frame_rate,avg_frame_rate,nb_frames,start_pts,duration_ts:frame=best_effort_timestamp,pkt_duration','-of','json',str(video)]
    data=json.loads(subprocess.check_output(command));return data['streams'][0],np.asarray([f['best_effort_timestamp']for f in data['frames']],np.int64)

def prepare_reference(original,calibration,root):
    import cv2
    original,root=Path(original),Path(root);source=original/'json/Skater_A/Axel/Axel_1.json';reference=qtm_reference(source);path=root/'reference_original142.npz'
    if path.exists():raise FileExistsError(path)
    np.savez_compressed(path,**reference);names=reference['names'].tolist();world=reference['world_points_mm']/1000.;valid=reference['valid_mask'];raw=read(calibration);cams={};coverage=[]
    ids=[names.index(k)for k in ['HEAD','LSHO','RSHO','LHIP','RHIP','LKNE','RKNE','LANK','RANK']];panels=[];clock_records={}
    links=[('HEAD','LSHO'),('HEAD','RSHO'),('LSHO','RSHO'),('LSHO','LELB'),('LELB','LWRT'),('RSHO','RELB'),('RELB','RWRT'),('LSHO','LHIP'),('RSHO','RHIP'),('LHIP','RHIP'),('LHIP','LKNE'),('LKNE','LANK'),('RHIP','RKNE'),('RKNE','RANK'),('LANK','LHEL'),('LHEL','L_Toe'),('RANK','RHEL'),('RHEL','R_Toe')]
    for n in range(1,13):
        cam=camera(raw[f'fs_camera_{n}']);cams[str(n)]=cam;video=original/f'skater_A/cam_{n}/Axel_1.mp4';stream,pts=clocks(video);height,width=stream['height'],stream['width'];uv,front=project(world,cam)
        if len(pts)!=len(world)or not np.array_equal(pts,np.arange(300)*256):raise ValueError('Real video/source sampling differs')
        inside=front&valid&(uv[...,0]>=0)&(uv[...,0]<width)&(uv[...,1]>=0)&(uv[...,1]<height)
        coverage.append(dict(camera=n,all_major_points_visible=int(inside[:,ids].all(1).sum()),some_major_points_visible=int(inside[:,ids].any(1).sum()),per_frame_major_count=inside[:,ids].sum(1).tolist()))
        clock_records[str(n)]=dict(path=str(video),sha256=sha(video),stream=stream,actual_pts=pts.tolist(),clock_mapping='reference frame1..300 -> video frame0..299; based on released hardware-synchronized full5s exports; content check follows; no offset fitted',timing_uncertainty='sampling interval1/60s; exposure/readout and subframe sync error not released')
        cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,140);ok,image=cap.read();cap.release()
        if not ok:raise RuntimeError(video)
        for a,b in links:
            ia,ib=names.index(a),names.index(b)
            if inside[140,ia]and inside[140,ib]:cv2.line(image,tuple(np.rint(uv[140,ia]).astype(int)),tuple(np.rint(uv[140,ib]).astype(int)),(0,255,0),2)
        tile=cv2.resize(image,(480,272));cv2.putText(tile,f'cam{n} original142 reference/frame141',(7,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1);panels.append(tile)
    cv2.imwrite(str(root/'source_checks/reference_all_cameras_frame141.jpg'),np.concatenate([np.concatenate(panels[i:i+4],axis=1)for i in range(0,12,4)],axis=0))
    np.savez_compressed(root/'cameras.npz',**{f'cam{n}_{k}':v for n,c in cams.items()for k,v in c.items()})
    write(root/'source_checks/capture_coverage.json',dict(source=str(source),all_original_names=len(names),frames=len(world),rows=coverage,source_clock_uncertainty_retained=True))
    write(root/'video_clocks.json',clock_records);write(root/'reference_identity.json',dict(source=str(source),sha256=sha(source),all_original_names=names,reference_npz=str(path),reference_sha256=sha(path),points=world.shape[1],frames=len(world),unit='mm',original_world_unchanged=True,missing_point_slots=int((~valid).sum()),calibration_source=str(calibration),calibration_sha256=sha(calibration),camera_convention='official diag(1,-1,-1)@R and camera centre xyz; OpenCV coefficient order=[original0,1,3,4,2]',world_transform='identity; no fitted alignment'))
    print(json.dumps(dict(event='real_trial_reference_ready',frames=len(world),all_original_names=len(names),coverage=[{k:v for k,v in row.items()if k!='per_frame_major_count'}for row in coverage])),flush=True)

BODY_PAIRS=[('left_shoulder','LSHO'),('right_shoulder','RSHO'),('left_elbow','LELB'),('right_elbow','RELB'),('left_wrist','LWRT'),('right_wrist','RWRT'),('left_hip','LHIP'),('right_hip','RHIP'),('left_knee','LKNE'),('right_knee','RKNE'),('left_ankle','LANK'),('right_ankle','RANK')]

def distribution(values):
    values=np.asarray(values);values=values[np.isfinite(values)]
    return dict(count=len(values),median=float(np.median(values)),p95=float(np.percentile(values,95)),maximum=float(values.max()),mean=float(values.mean()))if len(values)else dict(count=0,median=None,p95=None,maximum=None,mean=None)

def evaluate(root,original,calibration,input_camera=7,output=None):
    import cv2
    root,original=Path(root),Path(original);out=Path(output)if output is not None else root/'evaluation';out.mkdir(exist_ok=False)
    with np.load(root/'reference_original142.npz',allow_pickle=False)as z:ref={k:z[k]for k in z.files}
    with np.load(root/f'input_cam{input_camera}/all_camera_joints.npz',allow_pickle=False)as z:network={k:z[k]for k in z.files}
    names=ref['names'].tolist();jn=network['joint_names'].tolist();GT=ref['world_points_mm']/1000.;mask=ref['valid_mask'];raw=read(calibration);input_cam=camera(raw[f'fs_camera_{input_camera}']);Rinv=np.linalg.inv(input_cam['R_w2c'])
    pred_ids=[jn.index(a)for a,b in BODY_PAIRS];ref_ids=[names.index(b)for a,b in BODY_PAIRS];reference=GT[:,ref_ids];valid=mask[:,ref_ids];results={};arrays={};t=np.arange(len(GT))/60.
    for variant in ['prefoot','postfoot','final','hand_nofoot']:
        X=network[variant+'_joints_camera_m']@Rinv.T+input_cam['C_world_m'];arrays[variant+'_joints_world_m']=X
        errors=np.linalg.norm(X[:,pred_ids]-reference,axis=-1);errors[~valid]=np.nan
        root_error=np.linalg.norm(X[:,jn.index('pelvis')]-GT[:,names.index('PELVIS')],axis=-1)
        centred=(X[:,pred_ids]-X[:,[jn.index('pelvis')]])-(reference-GT[:,[names.index('PELVIS')]])
        relative=np.linalg.norm(centred,axis=-1);relative[~valid]=np.nan
        foot={}
        for side,prefix in [('left','L'),('right','R')]:
            heel=X[:,jn.index(side+'_heel')];toe=(X[:,jn.index(side+'_big_toe')]+X[:,jn.index(side+'_small_toe')])/2
            true_heel=GT[:,names.index(prefix+'HEL')];true_toe=GT[:,names.index(prefix+'_Toe')];a=toe-heel;b=true_toe-true_heel;angle=np.degrees(np.arccos(np.clip(np.sum(a*b,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)),-1,1)))
            foot[side]=dict(longitudinal_direction_degrees=distribution(angle),heel_position_m=distribution(np.linalg.norm(heel-true_heel,axis=1)),toe_position_m=distribution(np.linalg.norm(toe-true_toe,axis=1)),semantics='SMPL-X mesh heel/mean toes vs Theia anatomical proxies; not blade, boot shell, inner/outer edge or measured ice contact')
        view_errors=[]
        for n in range(1,13):
            cam=camera(raw[f'fs_camera_{n}']);pu,pfront=project(X[:,pred_ids],cam);gu,gfront=project(reference,cam);source_visible=gfront&valid&(gu[...,0]>=0)&(gu[...,0]<1920)&(gu[...,1]>=0)&(gu[...,1]<1088);distance=np.linalg.norm(pu-gu,axis=-1);distance[~(source_visible&pfront)]=np.nan
            view_errors.append(dict(camera=n,role='input'if n==input_camera else 'holdout',joint_exposures_possible=300*len(BODY_PAIRS),reference_visible_count=int(source_visible.sum()),prediction_behind_camera_count=int((source_visible&~pfront).sum()),native_pixel_error=distribution(distance)))
        per_frame=np.nanmean(errors,axis=1);worst=np.argsort(per_frame)[-8:][::-1]
        results[variant]=dict(absolute_body_3D_m=distribution(errors),root_relative_body_3D_m=distribution(relative),pelvis_proxy_position_m=distribution(root_error),per_joint=[dict(network=a,reference=b,error_m=distribution(errors[:,i]))for i,(a,b)in enumerate(BODY_PAIRS)],foot=foot,views=view_errors,worst_frames=[dict(reference_frame=int(i+1),video_pts=int(network['source_pts'][i]),source_seconds=float(t[i]),mean_body_error_m=float(per_frame[i]))for i in worst])
        arrays[variant+'_body_absolute_error_m']=errors;arrays[variant+'_body_root_relative_error_m']=relative;arrays[variant+'_pelvis_error_m']=root_error
    np.savez_compressed(out/'world_predictions_and_errors.npz',source_pts=network['source_pts'],source_time_base=network['source_time_base'],joint_names=network['joint_names'],reference_world_points_mm=ref['world_points_mm'],reference_names=ref['names'],reference_valid_mask=mask,**arrays)
    report=dict(schema='real_FS_single_view_ExMR_validation_v1',trial='FS-Jump3D/Skater_A/Axel/Axel_1',complete_frames=300,input_camera=input_camera,holdout_cameras=[n for n in range(1,13)if n!=input_camera],reference_all_names=142,reference_missing_slots=int((~mask).sum()),network_to_reference_body_pairs=BODY_PAIRS,no_world_alignment_or_scale_fit=True,no_per_frame_alignment=True,training=False,world_transform='identity; prediction inverted through only the released input camera',camera_parameters_fitted=False,holdout_used_for_model_or_weight_selection=False,reference_role='Theia multiview reference, not exact physical/ice truth',variants=results,accuracy_pass_declared=False,system_scope='frozen network camera-coordinate initializer, not full S05 human-shoe-ice solve',temporal_limit='native60Hz without cadence adapter; no world smooth/ground/static-joint postprocess')
    write(out/'metrics.json',report)
    # Actual pixel comparisons in all 12 released source views; no alignment.
    links=[('left_shoulder','right_shoulder'),('left_shoulder','left_elbow'),('left_elbow','left_wrist'),('right_shoulder','right_elbow'),('right_elbow','right_wrist'),('left_shoulder','left_hip'),('right_shoulder','right_hip'),('left_hip','right_hip'),('left_hip','left_knee'),('left_knee','left_ankle'),('right_hip','right_knee'),('right_knee','right_ankle')];pair_index={a:i for i,(a,b)in enumerate(BODY_PAIRS)}
    for frame in [0,59,110,140,170,230,299]:
        tiles=[]
        for n in range(1,13):
            cap=cv2.VideoCapture(str(original/f'skater_A/cam_{n}/Axel_1.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read();cap.release()
            if not ok:raise RuntimeError('Missing actual comparison exposure')
            cam=camera(raw[f'fs_camera_{n}']);gu,gf=project(reference[frame],cam);pu,pf=project(arrays['hand_nofoot_joints_world_m'][frame,pred_ids],cam)
            for points,front,color in [(gu,gf&valid[frame],(0,255,0)),(pu,pf,(255,0,255))]:
                for a,b in links:
                    ia,ib=pair_index[a],pair_index[b]
                    if front[ia]and front[ib]and np.max(abs(points[[ia,ib]]))<20000:cv2.line(image,tuple(np.rint(points[ia]).astype(int)),tuple(np.rint(points[ib]).astype(int)),color,2)
            tile=cv2.resize(image,(640,363));cv2.putText(tile,f'cam{n} {"INPUT"if n==input_camera else "CHECK"} frame{frame+1}',(7,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),1);tiles.append(tile)
        cv2.imwrite(str(out/f'all_views_frame_{frame+1:03d}.jpg'),np.concatenate([np.concatenate(tiles[i:i+4],axis=1)for i in range(0,12,4)],axis=0))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(13,10));A=arrays['hand_nofoot_joints_world_m'][:,jn.index('pelvis')];B=GT[:,names.index('PELVIS')]
    ax[0,0].plot(B[:,0],B[:,1],label='Theia original world');ax[0,0].plot(A[:,0],A[:,1],label='ExMR camera7 -> same world');ax[0,0].set_aspect('equal');ax[0,0].set_xlabel('world X / m');ax[0,0].set_ylabel('world Y / m');ax[0,0].legend()
    ax[0,1].plot(t,B[:,2],label='Theia pelvis proxy');ax[0,1].plot(t,A[:,2],label='ExMR pelvis');ax[0,1].set_ylabel('raw world height / m');ax[0,1].legend()
    for variant in results:ax[1,0].plot(t,arrays[variant+'_pelvis_error_m'],label=variant)
    ax[1,0].set_ylabel('absolute pelvis proxy error / m');ax[1,0].legend()
    for variant in results:ax[1,1].plot(t,np.nanmean(arrays[variant+'_body_root_relative_error_m'],axis=1),label=variant)
    ax[1,1].set_ylabel('root-relative body discrepancy / m')
    for a in ax[1]:a.set_xlabel('original video seconds');a.grid(True)
    fig.suptitle('Full 300 exposures; fixed released cameras/world/scale; no per-frame alignment');fig.tight_layout();fig.savefig(out/'full_trial_trajectory_and_errors.png',dpi=150);plt.close(fig)
    print(json.dumps({v:{k:r[k]for k in ['absolute_body_3D_m','root_relative_body_3D_m','pelvis_proxy_position_m']}for v,r in results.items()}),flush=True)

def image_clock_audit(root,original,calibration,data):
    """Evaluation-only RTMW reads actual pixels using reference-assisted ROIs.

    This checks released camera/clock consistency, not blind person localization.
    It is isolated from the already saved single-camera ExMR predictions.
    """
    from .exmr_prior import runtime,Progress
    torch=runtime(data)
    import cv2,onnxruntime as ort
    from exmr.preprocessing.keypoints import rtmw,VISIBILITY_THRESHOLD
    root,original=Path(root),Path(original);log=Progress(root,'image_clock_check')
    with np.load(root/'reference_original142.npz',allow_pickle=False)as z:world=z['world_points_mm']/1000.;mask=z['valid_mask'];names=z['names'].tolist()
    raw=read(calibration);ids=[names.index(b)for a,b in BODY_PAIRS];coco_ids=np.arange(5,17);model=Path(data)/'cache/pretrained/exmr/rtmw/rtmw-x_384x288_fp16.onnx';options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(model),sess_options=options,providers=[('CUDAExecutionProvider',{'device_id':0}),'CPUExecutionProvider'])
    if session.get_providers()[0]!='CUDAExecutionProvider':raise RuntimeError('Image audit RTMW CUDA unavailable')
    inp=session.get_inputs()[0];outputs=[x.name for x in session.get_outputs()];all_kp=np.full((12,300,17,3),np.nan,np.float32);rois=np.full((12,300,4),np.nan);result=[];limit=VISIBILITY_THRESHOLD['rtmw']
    for n in range(1,13):
        cam=camera(raw[f'fs_camera_{n}']);uv,front=project(world[:,ids],cam);inside=front&mask[:,ids]&(uv[...,0]>=0)&(uv[...,0]<1920)&(uv[...,1]>=0)&(uv[...,1]<1088);cap=cv2.VideoCapture(str(original/f'skater_A/cam_{n}/Axel_1.mp4'));batch=[];centres=[];scales=[];frames=[]
        def flush():
            if not batch:return
            tensor=np.stack(batch);count=len(tensor);tensor=np.pad(tensor,[(0,8-count),(0,0),(0,0),(0,0)],mode='edge');sx,sy=session.run(outputs,{inp.name:tensor});xy,score=rtmw._decode(sx[:count],sy[:count],np.array(centres),np.array(scales));all_kp[n-1,frames]=np.concatenate([xy[:,:17],score[:,:17,None]],axis=-1);batch.clear();centres.clear();scales.clear();frames.clear()
        for i in range(300):
            ok,image=cap.read()
            if not ok:raise RuntimeError('Image audit missing exposure')
            if inside[i].sum()<4:continue
            points=uv[i,inside[i]];lo=points.min(0)-25;hi=points.max(0)+25;box=np.r_[lo,hi];rois[n-1,i]=box;centre,scale=rtmw._bbox_xyxy2cs(box,rtmw.BOX_PADDING);crop,scale=rtmw._top_down_affine(rtmw.INPUT_SIZE,scale,centre,image);normalized=(crop-np.array(rtmw.MEAN))/np.array(rtmw.STD);batch.append(np.ascontiguousarray(normalized.transpose(2,0,1),dtype=np.float32));centres.append(centre);scales.append(scale);frames.append(i)
            if len(batch)==8:flush()
        flush();cap.release();observed=all_kp[n-1][:,coco_ids]
        offsets=[]
        for shift in range(-3,4):
            a=np.arange(max(0,-shift),min(300,300-shift));b=a+shift;use=inside[b]&np.isfinite(observed[a,:,0])&(observed[a,:,2]>limit);dist=np.linalg.norm(observed[a,:,:2]-uv[b],axis=-1);offsets.append(dict(reference_shift_frames=shift,pixel_discrepancy=distribution(dist[use])))
        nominal=next(x for x in offsets if x['reference_shift_frames']==0);best=min([x for x in offsets if x['pixel_discrepancy']['count']],key=lambda x:x['pixel_discrepancy']['median'])
        row=dict(camera=n,total_video_exposures=300,ROI_read_exposures=int(np.isfinite(rois[n-1,:,0]).sum()),source_reference_visible_joint_exposures=int(inside.sum()),nominal_zero_shift=nominal,best_diagnostic_shift=best,all_tested_shifts=offsets,shift_applied=False);result.append(row);log('camera_complete',camera=n,ROI_frames=row['ROI_read_exposures'],zero_median_px=nominal['pixel_discrepancy']['median'],diagnostic_best_shift=best['reference_shift_frames'])
    np.savez_compressed(root/'source_checks/actual_image_clock_observations.npz',coco17_native_image_network=all_kp,reference_assisted_evaluation_roi=rois,actual_video_pts=np.arange(300)*256,time_base=np.array([1,15360]))
    write(root/'source_checks/image_clock_audit.json',dict(schema='actual_pixels_FS_camera_clock_check_v1',all_frames_per_camera=300,cameras=12,model_sha256=sha(model),network_score_threshold=float(limit),body_pairs=BODY_PAIRS,GT_assisted_evaluation_ROI=True,independent_person_localization_test=False,no_fit_or_offset_applied=True,ExMR_input_or_weights_changed=False,source_clock_basis='published hardware synchronization, matching full300 original QTM samples and actual videoPTS; shifts diagnostic only',subframe_hardware_error_bound_known=False,limitations=['Theia virtual joints and RTMW anatomical pixels are different proxy definitions; nonzero best shift can reflect reference/tracker lag, not just video clock.','Spatial calibration/ROI and clock checks are acquisition checks; frozen ExMR predictions were saved before reading holdout images.','No camera, scale, distortion or time shift was fitted.'],rows=result))
    log('complete',cameras=12,actual_video_exposures=3600)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['reference','evaluate','clock'],required=True);ap.add_argument('--original',type=Path,required=True);ap.add_argument('--calibration',type=Path,required=True);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--data-root',type=Path);ap.add_argument('--evaluation-output',type=Path);a=ap.parse_args()
    if a.stage=='reference':prepare_reference(a.original,a.calibration,a.root)
    elif a.stage=='evaluate':evaluate(a.root,a.original,a.calibration,output=a.evaluation_output)
    else:image_clock_audit(a.root,a.original,a.calibration,a.data_root)

if __name__=='__main__':main()
