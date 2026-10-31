"""Image-only navigation for newly registered shots; never emits fit observations."""
from pathlib import Path
import argparse, hashlib, json, time
import av, cv2, numpy as np
from scipy.optimize import least_squares
from rinkspace.core.native import NativeCore
from rinkspace.session.source_registration import RegistrationOptions, frame_features, track_pair
from rinkspace.geometry import project


def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def frames(index, shot):
    w=shot['processing']; expected=w['first_pts']
    with av.open(index['source']['resolved_path']) as container:
        stream=container.streams.video[0]; container.seek(expected,stream=stream,backward=True)
        for frame in container.decode(stream):
            if frame.pts<expected: continue
            if frame.pts>w['last_pts']: break
            if frame.pts!=expected: raise ValueError('Actual native source clock discontinuity')
            if str(frame.time_base)!='1/12800': raise ValueError('Source time base changed')
            yield int(frame.pts),frame.to_ndarray(format='bgr24')
            expected+=256
    if expected!=w['last_pts']+256: raise ValueError('Incomplete original decode')


def source_mask(root, sid, pts):
    mask=np.ones((1080,1920),bool)
    mask[:290,:570]=False; mask[940:,1580:]=False
    path=root/'person_masks'/sid/f'{pts}.png'
    if path.exists():
        person=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if person.shape!=mask.shape: raise ValueError('Person-mask dimensions differ')
        mask &= ~cv2.dilate((person>0).astype('uint8'),np.ones((15,15),'uint8')).astype(bool)
    else:
        # Conservative central subject search exclusion, not a segmentation claim.
        mask[160:,600:1320]=False
    return mask


def image_graph(root,index,shot):
    sid=shot['shot_id']; out=root/sid/'navigation';out.mkdir(exist_ok=True)
    cache=out/'source_graph.npz'
    if cache.exists(): return dict(np.load(cache,allow_pickle=False))
    options=RegistrationOptions(max_features=1000)
    previous=None; key=None; hprev=np.eye(3); rows=[]; matrices=[]; ptss=[]; anchor_to_key=np.eye(3)
    started=time.monotonic()
    with (out/'source_graph.jsonl').open('w',encoding='utf8',buffering=1) as log:
        for i,(pts,image) in enumerate(frames(index,shot)):
            current=frame_features(image,{'background':source_mask(root,sid,pts)},options)
            if key is None:
                key=current; previous=current; keyindex=i; hprev=np.eye(3)
                record=dict(status='anchor',inliers=0,check_p90_px=0.); h=np.eye(3)
            else:
                record,a,b,keep=track_pair(key,current,'background',options,initial_h=hprev)
                h=np.asarray(record.get('H_current_from_reference',np.eye(3)),float)
                if record['status']!='navigation':
                    retry,aa,bb,kk=track_pair(previous,current,'background',options)
                    if retry['status']=='navigation':
                        h=np.asarray(retry['H_current_from_reference'])@hprev
                        record=dict(retry,recovered_by_adjacent_pair=True)
                    elif 'H_current_from_reference' not in record or record.get('inliers',0)<12:
                        h=hprev.copy();record=dict(record,navigation_hold=True)
                step=h@np.linalg.inv(hprev);step/=step[2,2]
                if not np.isfinite(step).all(): raise ValueError('Nonfinite source navigation')
                matrices.append(step)
            ptss.append(pts);record.update(pts=pts,frame=i,keyframe=keyindex,fit_observations_created=0)
            rows.append(record);log.write(json.dumps(record)+'\n')
            hprev=h;previous=current
            # Retain a clear source image as the reference for ten exposures.
            if i-keyindex>=10 and record['status']=='navigation':
                key=current;keyindex=i;hprev=np.eye(3)
            if i%100==0:print(json.dumps(dict(stage='source_flow',shot=sid,frames=i+1,seconds=time.monotonic()-started)),flush=True)
    graph=dict(pts=np.asarray(ptss,np.int64),adjacent_h=np.asarray(matrices),supported=np.array([r['status'] in ('navigation','anchor') for r in rows]))
    np.savez_compressed(cache,**graph)
    write(out/'source_graph_summary.json',dict(frames=len(ptss),supported=int(graph['supported'].sum()),held=sum(bool(r.get('navigation_hold'))for r in rows),source_sha256=index['source']['actual_sha256'],source_clock='1/12800',navigation_only=True,fit_observations_created=0,old_new_shot_points_or_motion_loaded=False))
    return graph


def solve_navigation(root,index,shot,anchor,core,world,cuts,rig):
    sid=shot['shot_id']; out=root/sid/'navigation';graph=image_graph(root,index,shot)
    candidates=np.concatenate([world[np.linspace(c['begin'],c['end']-1,min(8,c['end']-c['begin'])).round().astype(int)] for c in cuts if c['layer'] in ('ice','board','advertisement')])
    poses=np.full((len(graph['pts']),3),np.nan); origin=int(np.flatnonzero(graph['pts']==anchor['pts'])[0]);poses[origin]=anchor['initial_pose']
    def projection(theta,xyz):return core.project_motion(rig,np.tile(theta,(6,1)),-2.,1.,1080,0.,xyz)
    reports=[];started=time.monotonic()
    for direction in (1,-1):
        indices=range(origin+1,len(poses)) if direction==1 else range(origin-1,-1,-1)
        for i in indices:
            prev=i-direction;theta=poses[prev]
            result=projection(theta,candidates);uv=result['uv'];valid=(result['status']==0)&np.isfinite(uv).all(1)&(uv>[-150,-100]).all(1)&(uv<[2070,1180]).all(1)
            ids=np.flatnonzero(valid)
            if len(ids)<6:raise ValueError(f'Insufficient visible native search geometry {sid} {i}')
            # Spatially balanced search geometry; it is not a point weight scheme.
            cells=np.floor(uv[ids]/[240,180]).astype(int);_,sel=np.unique(cells,axis=0,return_index=True);ids=ids[sel]
            xyz=candidates[ids];old_uv=uv[ids]
            h=graph['adjacent_h'][i-1] if direction==1 else np.linalg.inv(graph['adjacent_h'][i])
            target=project(h,old_uv)
            def residual(value):
                q=projection(value,xyz);r=q['uv']-target
                if np.any(q['status']) or not np.isfinite(r).all():return np.full(2*len(xyz),1e5)
                return r.ravel()
            fit=least_squares(residual,theta,loss='soft_l1',f_scale=2.,diff_step=1e-5,x_scale=[.2,.1,1.],max_nfev=12,ftol=1e-6,xtol=1e-7,gtol=1e-5)
            poses[i]=fit.x
            reports.append(dict(pts=int(graph['pts'][i]),direction=direction,navigation_projection_rms=float(np.sqrt(np.mean(residual(fit.x)**2))),source_flow_supported=bool(graph['supported'][i]),success=bool(fit.success)))
            if len(reports)%200==0:print(json.dumps(dict(stage='native_navigation',shot=sid,frames=len(reports),seconds=time.monotonic()-started)),flush=True)
    np.savez_compressed(out/'poses.npz',pts=graph['pts'],poses=poses,source_flow_supported=graph['supported'])
    write(out/'poses.json',dict(anchor=anchor,frames=reports,navigation_only=True,fit_observations_created=0,rig_inherited_from_baseline=True,new_shot_motion_initialization='independent source image correspondences from known-rig anchor',old_new_shot_points_or_motion_loaded=False))
    print(json.dumps(dict(stage='navigation_done',shot=sid,frames=len(poses),pan_range=poses[:,0].min().item(),pan_max=poses[:,0].max().item(),focal_range=np.exp(poses[:,2]).min().item(),focal_max=np.exp(poses[:,2]).max().item())),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--shots',nargs='+',default=['s07','s08','s09']);args=p.parse_args()
    root=args.root;index=json.loads((root/'index.json').read_text('utf8'));base=json.loads((root.parent/'baseline.json').read_text('utf8'))
    if hashlib.file_digest(Path(index['source']['resolved_path']).open('rb'),'sha256').hexdigest()!=index['source']['actual_sha256']:raise ValueError('Original source hash changed')
    anchors=json.loads((root/'independent_view_seeds.json').read_text('utf8'))['anchors'];data=dict(np.load(root/'native_world.npz',allow_pickle=False));cuts=json.loads((root/'native_world.json').read_text('utf8'))['cuts'];core=NativeCore(base['inputs']['library']['path']);cv2.setNumThreads(2)
    for sid in args.shots:
        shot=next(s for s in index['shots'] if s['shot_id']==sid);anchor=next(s for s in anchors if s['shot']==sid)
        solve_navigation(root,index,shot,anchor,core,data['xyz'],cuts,data['rig'][anchor['camera_index']])
if __name__=='__main__':main()
