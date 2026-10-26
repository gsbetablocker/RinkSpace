"""Inspect current ice residuals on exact source frames; never edits fit input."""
from pathlib import Path
import hashlib
import io
import json
import cv2
import numpy as np
import prepare_native_s01 as prep
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.session.gpu_geometry import NativeSceneGPU


def arrays(path):
    with np.load(io.BytesIO(path.read_bytes()),allow_pickle=False) as archive:
        return {key:archive[key] for key in archive.files}


def run():
    out=prep.RUN/'ice_residual_audit';out.mkdir(exist_ok=True)
    snapshot=prep.RUN/'accepted_pixels_latest.npz';state_sha=prep.digest(snapshot)
    state=arrays(snapshot);points=arrays(prep.RUN/'point_observations.npz')
    evaluation=arrays(prep.RUN/'evaluation_pixels.npz');mapping=prep.read(prep.RUN/'mapping.json')
    assert str(state['point_input_sha256'])==prep.digest(prep.RUN/'point_observations.npz')
    assert np.array_equal(state['source_rows'],points['source_rows'])
    assert not np.any(evaluation['status'])
    paths=np.asarray(mapping['path_ids'])[points['path_indices']]
    errors=np.linalg.norm(evaluation['values'][:,9:11],axis=1)
    source=prep.DATA/'runs/native_ad_replacement_s01_20261026/visibility.json'
    frames={row['pts']:row for row in prep.read(source)['frames']}
    camera=prep.camera_for_snapshot(prep.read(prep.CAMERA),state)
    core=NativeCore(prep.LIBRARY)
    model=SceneParameterModel(prep.read(prep.SCENE),prep.SCENE,optimize_scene=True)
    native=NativeSceneGPU(prep.SCENE,core,parameter_state=ParameterState(model.initial_state.layout,state['scene_values']))
    labels={p['id']:p.get('label',p['id']) for p in prep.read(prep.SCENE)['paths']}
    statistics=[]
    for path in np.unique(paths):
        if native.paths[path]['layer']!='ice':continue
        rows=np.flatnonzero(paths==path);temporal=[]
        for pts in np.unique(points['pts'][rows]):
            group=rows[points['pts'][rows]==pts]
            temporal.append(dict(pts=int(pts),rows=len(group),median=float(np.median(errors[group])),
                maximum=float(errors[group].max()),cost=float(.5*np.sum(errors[group]**2))))
        statistics.append(dict(path=path,label=labels.get(path,path),rows=len(rows),frames=len(temporal),
            median_p95_max=np.percentile(errors[rows],[50,95,100]).tolist(),time_series=temporal))
    prep.write(out/'ice_path_residuals.json',dict(state_sha256=state_sha,paths=statistics))
    groups=[('F3','d78_material_152/path0',[491264,491520,491776,492032,492288],[1570,800,1920,1000]),
            ('C0','d78_material_144/path0',[373248,373504,373760,374016,374272],[120,770,800,1080])]
    records=[]
    for label,path,pts_list,crop in groups:
        panels=[];lo=np.asarray(crop[:2]);hi=np.asarray(crop[2:]);width,height=hi-lo
        for pts in pts_list:
            frame=frames[pts];image=cv2.imread(frame['image'])
            if hashlib.sha256(image.tobytes()).hexdigest()!=frame['source_BGR_sha256']:
                raise ValueError('Source image identity changed')
            stations=np.linspace(0,1,4097)
            with native.resident(camera={**camera,'rig':state['rig'].tolist()},controls=state['controls'],
                    path_ids=[path]*len(stations),stations=stations,pts=np.full(len(stations),pts),
                    observed=np.zeros((len(stations),2))) as resident:
                evaluated=resident.evaluate('cuda')
            good=evaluated['status']==0;uv=evaluated['values'][:,3:5]
            visible=uv[good];tip=np.flatnonzero(good)[np.argmin(visible[:,0])]
            time=(pts-camera['reference_pts'])/12800
            moving=core.project_motion(state['rig'],state['controls'],camera['first_control_time'],camera['control_dt'],
                camera['image_height'],time+np.array([-.01,0,.01]),np.tile(evaluated['values'][tip,:3],(3,1)))
            motion=core.evaluate_motion_kinematics(state['controls'],camera['first_control_time'],camera['control_dt'],[time])
            rows=np.flatnonzero((paths==path)&(points['pts']==pts));overlay=image.copy()
            transitions=np.flatnonzero(np.diff(np.r_[False,good,False])).reshape(-1,2)
            for first,last in transitions:
                if last-first>1:cv2.polylines(overlay,[np.rint(uv[first:last]).astype(np.int32)],False,(0,255,255),1,cv2.LINE_AA)
            for row in rows:
                measured=tuple(np.rint(points['uv'][row]).astype(int));fitted=tuple(np.rint(evaluation['values'][row,3:5]).astype(int))
                cv2.line(overlay,measured,fitted,(40,120,255),1,cv2.LINE_AA)
                cv2.circle(overlay,measured,3,(0,0,255),1,cv2.LINE_AA)
                cv2.drawMarker(overlay,fitted,(0,255,0),cv2.MARKER_CROSS,5,1)
            top=image[lo[1]:hi[1],lo[0]:hi[0]];bottom=overlay[lo[1]:hi[1],lo[0]:hi[0]]
            panel=np.zeros((height*2+36,width,3),np.uint8);panel[36:36+height]=top;panel[36+height:]=bottom
            cv2.putText(panel,f'{label} {pts/12800:.3f}s PTS {pts}',(6,23),cv2.FONT_HERSHEY_SIMPLEX,.50,(255,255,255),1,cv2.LINE_AA)
            panels.append(panel)
            record=dict(label=label,path=path,pts=pts,source_image=frame['image'],source_BGR_sha256=frame['source_BGR_sha256'],
                native_left_tip=uv[tip].tolist(),fixed_tip_station=float(stations[tip]),
                fixed_tip_uv_at_minus10_zero_plus10ms=moving['uv'].tolist(),tip_shift_status=moving['status'].tolist(),
                pan_tilt_rate_deg_s=np.degrees(motion['velocity'][0,:2]).tolist(),
                observations=[dict(id=str(points['observation_ids'][i]),source_row=int(points['source_rows'][i]),
                    uv=points['uv'][i].tolist(),fitted_uv=evaluation['values'][i,3:5].tolist(),error_px=float(errors[i])) for i in rows])
            records.append(record)
            print(json.dumps({k:record[k] for k in ['label','pts','native_left_tip','fixed_tip_uv_at_minus10_zero_plus10ms','pan_tilt_rate_deg_s']},ensure_ascii=True),flush=True)
        cv2.imwrite(str(out/f'{label}_source_and_native.png'),np.concatenate(panels,axis=1))
    prep.write(out/'source_review.json',dict(state_sha256=state_sha,
        point_input_sha256=prep.digest(prep.RUN/'point_observations.npz'),scene_revision=native.scene_revision,
        source_index_sha256=prep.digest(source),records=records,
        time_offsets_are_motion_diagnostics_not_exposure_calibration=True,
        observations_changed=False,parameters_changed=False))
    if prep.digest(snapshot)!=state_sha:raise ValueError('Accepted state changed during audit')


def maximum_neighborhood(radius_frames=25,source_range_pts=None):
    """Show every source exposure around the maximum; no state or point edits."""
    import av
    from fractions import Fraction
    out=prep.RUN/'maximum_neighborhood';out.mkdir(exist_ok=True)
    snapshot=prep.RUN/'accepted_pixels_latest.npz';state_sha=prep.digest(snapshot)
    state=arrays(snapshot);points=arrays(prep.RUN/'point_observations.npz')
    point_sha=prep.digest(prep.RUN/'point_observations.npz')
    assert str(state['point_input_sha256'])==point_sha
    assert np.array_equal(state['source_rows'],points['source_rows'])
    mapping=prep.read(prep.RUN/'mapping.json');paths=np.asarray(mapping['path_ids'])[points['path_indices']]
    camera=prep.camera_for_snapshot(prep.read(prep.CAMERA),state);camera={**camera,'rig':state['rig'].tolist()}
    core=NativeCore(prep.LIBRARY);model=SceneParameterModel(prep.read(prep.SCENE),prep.SCENE,optimize_scene=True)
    native=NativeSceneGPU(prep.SCENE,core,parameter_state=ParameterState(model.initial_state.layout,state['scene_values']))
    with native.resident(camera=camera,controls=state['controls'],path_ids=paths,stations=state['stations'],
                         pts=points['pts'],observed=points['uv']) as resident:
        evaluation=resident.evaluate('cuda')
    assert not np.any(evaluation['status'])
    errors=np.linalg.norm(evaluation['values'][:,9:11],axis=1);worst=int(np.argmax(errors))
    path=str(paths[worst]);peak=int(points['pts'][worst]);all_pts=np.unique(points['pts'])
    first,last=(peak-256*radius_frames,peak+256*radius_frames) if source_range_pts is None else source_range_pts
    selected=all_pts[(all_pts>=first)&(all_pts<=last)]
    if not len(selected) or peak not in selected:raise ValueError('Review interval must include the maximum and source exposures')
    source_index=prep.DATA/'runs/native_ad_replacement_s01_20261026/visibility.json'
    frames={int(row['pts']):row for row in prep.read(source_index)['frames']}
    samples=4097;end=core.board_perimeter(native.scene['board']) if native.paths[path]['kind']==7 else 1.
    stations=np.tile(np.linspace(0,end,samples),len(selected))
    with native.resident(camera=camera,controls=state['controls'],path_ids=[path]*len(stations),
                         stations=stations,pts=np.repeat(selected,samples),observed=np.zeros((len(stations),2))) as resident:
        prediction=resident.evaluate('cuda')
    projected=prediction['values'][:,3:5].reshape(len(selected),samples,2)
    good=(prediction['status']==0).reshape(len(selected),samples)
    visible=good&(projected[:,:,0]>=0)&(projected[:,:,0]<1920)&(projected[:,:,1]>=0)&(projected[:,:,1]<1080)
    coordinates=projected[visible]
    observed=points['uv'][(paths==path)&np.isin(points['pts'],selected)]
    coordinates=np.vstack((coordinates,observed))
    lo=np.maximum(0,np.floor(coordinates.min(0)-42)).astype(int)
    hi=np.minimum([1920,1080],np.ceil(coordinates.max(0)+42)).astype(int)
    near=np.abs(selected-peak)<=1280
    near_coordinates=np.vstack((projected[near][visible[near]],points['uv'][(paths==path)&np.isin(points['pts'],selected[near])]))
    near_lo=np.maximum(0,np.floor(near_coordinates.min(0)-30)).astype(int)
    near_hi=np.minimum([1920,1080],np.ceil(near_coordinates.max(0)+30)).astype(int)
    # One fixed crop for the whole sequence: no tracking that could hide jumps.
    width,height=hi-lo;video=out/'maximum_source_native_50fps.mp4'
    vw=int(2*width);vh=int(height+72);vw+=vw%2;vh+=vh%2
    output=av.open(str(video),'w',options={'movflags':'+faststart','video_track_timescale':'12800'})
    stream=output.add_stream('libx264',rate=50);stream.width=vw;stream.height=vh;stream.pix_fmt='yuv420p'
    stream.time_base=Fraction(1,12800);stream.codec_context.time_base=Fraction(1,12800)
    stream.options={'crf':'17','preset':'fast'};panels=[];near_panels=[];records=[]
    try:
        for i,pts in enumerate(selected):
            source=frames[int(pts)];raw=cv2.imread(source['image'])
            if raw is None or hashlib.sha256(raw.tobytes()).hexdigest()!=source['source_BGR_sha256']:
                raise ValueError('Source frame identity changed')
            overlay=raw.copy();uv=projected[i];valid=good[i]
            for first,last in np.flatnonzero(np.diff(np.r_[False,valid,False])).reshape(-1,2):
                if last-first>1:cv2.polylines(overlay,[np.rint(np.clip(uv[first:last],-1e7,1e7)).astype(np.int32)],False,(0,255,255),1,cv2.LINE_AA)
            rows=np.flatnonzero((paths==path)&(points['pts']==pts));frame_rows=np.flatnonzero(points['pts']==pts)
            for row in rows:
                measured=tuple(np.rint(points['uv'][row]).astype(int));fitted=tuple(np.rint(evaluation['values'][row,3:5]).astype(int))
                cv2.line(overlay,measured,fitted,(40,120,255),1,cv2.LINE_AA)
                cv2.circle(overlay,measured,3,(0,0,255),1,cv2.LINE_AA)
                cv2.drawMarker(overlay,fitted,(0,255,0),cv2.MARKER_CROSS,6,1)
            a=raw[lo[1]:hi[1],lo[0]:hi[0]];b=overlay[lo[1]:hi[1],lo[0]:hi[0]]
            maximum=float(errors[rows].max()) if len(rows) else None
            label=f'{pts/12800:.3f}s | PTS {pts} | n={len(rows)} | max '+(f'{maximum:.2f}px' if maximum is not None else 'NO POINTS')
            canvas=np.full((vh,vw,3),18,np.uint8);canvas[72:72+height,:width]=a;canvas[72:72+height,width:2*width]=b
            cv2.putText(canvas,label,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(canvas,'SOURCE | NATIVE: yellow curve, red observations, green predictions',(8,54),cv2.FONT_HERSHEY_SIMPLEX,.49,(255,255,255),1,cv2.LINE_AA)
            frame=av.VideoFrame.from_ndarray(canvas,format='bgr24');frame.pts=int(pts-selected[0]);frame.time_base=Fraction(1,12800)
            for packet in stream.encode(frame):output.mux(packet)
            def panel(a,b):
                panel_width=410;panel_height=int(round(a.shape[0]*panel_width/a.shape[1]))
                result=np.full((52+2*panel_height,panel_width,3),18,np.uint8)
                cv2.putText(result,f'{pts/12800:.3f}s | PTS {pts}',(7,19),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(result,f'n={len(rows)} | max '+(f'{maximum:.2f}px' if maximum is not None else 'NO POINTS'),(7,41),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
                result[52:52+panel_height]=cv2.resize(a,(panel_width,panel_height),interpolation=cv2.INTER_AREA)
                result[52+panel_height:]=cv2.resize(b,(panel_width,panel_height),interpolation=cv2.INTER_AREA)
                return result
            panels.append(panel(a,b))
            if near[i]:near_panels.append(panel(raw[near_lo[1]:near_hi[1],near_lo[0]:near_hi[0]],overlay[near_lo[1]:near_hi[1],near_lo[0]:near_hi[0]]))
            tip=np.flatnonzero(valid)[np.argmin(uv[valid,0])]
            records.append(dict(pts=int(pts),source_seconds=float(pts/12800),source_image=source['image'],source_BGR_sha256=source['source_BGR_sha256'],
                path_observations=len(rows),path_max_px=maximum,path_median_px=float(np.median(errors[rows])) if len(rows) else None,
                frame_max_px=float(errors[frame_rows].max()),frame_above_20px=int(np.count_nonzero(errors[frame_rows]>20)),
                native_left_tip=uv[tip].tolist(),observations=[dict(source_row=int(points['source_rows'][r]),id=str(points['observation_ids'][r]),
                    uv=points['uv'][r].tolist(),fitted_uv=evaluation['values'][r,3:5].tolist(),error_px=float(errors[r])) for r in rows]))
        for packet in stream.encode():output.mux(packet)
    finally:output.close()
    def contact(indices,name,images=panels):
        indices=list(indices);ph,pw=images[0].shape[:2];grid=np.full((((len(indices)+3)//4)*ph,4*pw,3),18,np.uint8)
        for k,i in enumerate(indices):grid[(k//4)*ph:(k//4+1)*ph,(k%4)*pw:(k%4+1)*pw]=images[i]
        cv2.imwrite(str(out/name),grid)
    contact(range(len(near_panels)),'every_frame_near_peak.png',near_panels)
    contact(sorted(set(list(range(0,len(selected),5))+[int(np.flatnonzero(selected==peak)[0]),len(selected)-1])),'one_second_context.png')
    with av.open(str(video)) as container:
        decoded=list(container.decode(video=0));actual=[float(f.pts*f.time_base) for f in decoded]
    assert np.allclose(actual,(selected-selected[0])/12800,rtol=0,atol=1e-8)
    assert prep.digest(snapshot)==state_sha and prep.digest(prep.RUN/'point_observations.npz')==point_sha
    report=dict(snapshot=str(snapshot),snapshot_sha256=state_sha,point_input_sha256=point_sha,
        native_library=str(prep.LIBRARY),native_library_sha256=prep.digest(prep.LIBRARY),path=path,
        peak_pts=peak,maximum_px=float(errors[worst]),source_first_seconds=float(selected[0]/12800),source_last_seconds=float(selected[-1]/12800),
        frames=len(selected),crop_xyxy=np.r_[lo,hi].tolist(),near_peak_crop_xyxy=np.r_[near_lo,near_hi].tolist(),crop_fixed_across_time=True,source_index_sha256=prep.digest(source_index),
        video=str(video),video_pts_checked=True,observations_changed=False,parameters_changed=False,records=records)
    prep.write(out/'review.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=True),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--maximum-neighborhood',action='store_true');parser.add_argument('--radius-frames',type=int,default=25)
    parser.add_argument('--source-range-pts',type=int,nargs=2);args=parser.parse_args()
    if args.run_dir:prep.RUN=args.run_dir.resolve(strict=True)
    if args.maximum_neighborhood:maximum_neighborhood(args.radius_frames,args.source_range_pts)
    else:run()
