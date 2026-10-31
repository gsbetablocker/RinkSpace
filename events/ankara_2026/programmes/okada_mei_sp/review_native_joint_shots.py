"""Selected exact-source frames from a frozen joint native candidate; no fitting."""
from pathlib import Path
import argparse,collections,io,json
import numpy as np,cv2
import prepare_native_joint_shots as p
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.review.scene import sample_native_review_scene
from rinkspace.review.render import source_frames
from rinkspace.review.draw import COLORS

def main(args):
    root=getattr(args,'run_root',None) or p.RUN;m=p.read(root/'joint_manifest.json');mp=p.read(root/'mapping.json');z=p.npz(root/'point_observations.npz')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    blob=args.snapshot.read_bytes();(out/'inspected_pixels.npz').write_bytes(blob);v=dict(np.load(io.BytesIO(blob),allow_pickle=False))
    scene=Path(m['scene_path']);core=p.NativeCore(m['native_library']);raw=p.read(scene);model=p.SceneParameterModel(raw,scene,optimize_scene=True)
    multi=v['rig'].ndim==2
    if multi:
        import prepare_native_multicamera as prepared
        prepared.validate_snapshot(v,model.initial_state.layout,mp,m,z)
    else:p.validate_joint_snapshot(v,model.initial_state.layout,mp,m,z)
    state=p.ParameterState(model.initial_state.layout,v['scene_values']);native=p.NativeSceneGPU(scene,core,parameter_state=state)
    tl=ShotMotionTimeline(m['shots'],m['control_dt']);camera={**p.read(m['cameras'][0]['camera_definition'] if multi else m['shared_camera_definition']),'rig':(v['rig'][0] if multi else v['rig']).tolist(),'first_control_time':tl.first,'control_dt':tl.dt}
    camera_ids=np.zeros(len(z['pts']),np.int32)
    for s in m['shots']:camera_ids[s['row_begin']:s['row_stop']]=s.get('camera_index',0)
    paths=np.array(mp['path_ids'])[z['path_indices']];selection=np.asarray(args.pts,dtype=np.int64)
    rows=np.flatnonzero(np.isin(z['pts'],selection));times=tl.row_times(z['pts'])
    with native.resident(camera=camera,rigs=v['rig'] if multi else None,camera_ids=camera_ids[rows] if multi else None,controls=v['controls'],path_ids=paths[rows].tolist(),stations=v['stations'][rows],pts=z['pts'][rows],observed=z['uv'][rows],motion_times=times[rows]) as rr:
        rr.configure_projection_time_support(half_seconds=m['max_readout']/2);pred=rr.evaluate('cuda')
    samples=sample_native_review_scene(raw,core,scene,parameter_state=state,observed_path_ids=set(paths),native=native);world=samples['xyz']
    source_scope=p.read(p.HERE/'rebuild_scope.json');source={**source_scope['source'],**source_scope['source_clock']};records=[]
    for pt in sorted(selection):
        shot=next(s for s in m['shots'] if s['first_pts']<=pt<=s['last_pts']);t=(pt-shot['reference_pts'])/12800+tl.offset(shot)
        _,im=next(source_frames(source['registered_path'],source,[int(pt)]));raw_image=im.copy()
        rig=v['rig'][shot['camera_index']] if multi else v['rig']
        ev=core.project_motion(rig,v['controls'],tl.first,tl.dt,1080,t,world);good=(ev['status']==0)&np.isfinite(ev['uv']).all(1)
        pixels=np.rint(np.clip(np.nan_to_num(ev['uv']),-1e6,1e6)).astype(np.int32)
        for cut in samples['cuts']:
            a,b=cut['begin'],cut['end'];color=COLORS[cut['layer']]
            for lo,hi in np.flatnonzero(np.diff(np.r_[False,good[a:b],False])).reshape(-1,2):
                if hi-lo>1:cv2.polylines(im,[pixels[a+lo:a+hi]],bool(cut.get('closed') and lo==0 and hi==b-a),color,1,cv2.LINE_AA)
        jj=np.flatnonzero(z['pts'][rows]==pt)
        for j in jj:
            i=rows[j];uv=tuple(np.rint(z['uv'][i]).astype(int));cv2.circle(im,uv,2,(0,0,255),-1,cv2.LINE_AA)
            if pred['status'][j]==0:
                at=tuple(np.rint(pred['values'][j,3:5]).astype(int));cv2.line(im,uv,at,(30,130,240),1,cv2.LINE_AA);cv2.drawMarker(im,at,(60,255,70),cv2.MARKER_CROSS,5,1,cv2.LINE_AA)
        errors=np.linalg.norm(pred['values'][jj,9:11],axis=1)
        valid=(pred['status'][jj]==0)&np.isfinite(errors);maximum=float(errors[valid].max()) if valid.any() else None;invalid=int(np.count_nonzero(~valid))
        text=f"{shot['id'].upper()} | {pt/12800:.3f}s PTS {pt} | raw UV red / native fitted green | max {maximum}px >20px {np.count_nonzero(errors[valid]>20)} invalid {invalid}"
        cv2.rectangle(im,(0,0),(1920,32),(0,0,0),-1);cv2.putText(im,text,(9,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1,cv2.LINE_AA)
        cv2.imwrite(str(out/f'{shot["id"]}_{pt}.png'),im);cv2.imwrite(str(out/f'{shot["id"]}_{pt}_source.png'),raw_image)
        records.append(dict(shot=shot['id'],pts=int(pt),rows=len(jj),max_px=maximum,above_20px=int(np.count_nonzero(errors[valid]>20)),invalid=invalid))
    p.write(out/'review.json',dict(snapshot=str(args.snapshot),snapshot_sha256=p.digest(out/'inspected_pixels.npz'),input_sha256=m['point_input_sha256'],native_library=m['native_library'],selected_frames_only=True,records=records))
    print(json.dumps(records),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run-root',type=Path);a.add_argument('--snapshot',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--pts',nargs='+',type=int,required=True);main(a.parse_args())
