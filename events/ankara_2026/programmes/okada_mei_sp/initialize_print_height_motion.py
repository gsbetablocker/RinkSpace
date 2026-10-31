"""Recover a motion seed from measured print edges, without inventing source UV.

Static native pose fits are initialization only. All pixels, rig, scene and
other shots remain unchanged; the production solve still uses rolling shutter
and the original independent curve observations.
"""
from pathlib import Path
import argparse, collections, hashlib, json
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
import prepare_native_multicamera as prepared
from build_independent_shot_packets import fit_pose_controls
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.review.scene import sample_scene
from rinkspace.path_policy import guard_project_write


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    m=read(a.prepared/'joint_manifest.json');mp=read(a.prepared/'mapping.json')
    v=prepared.npz(a.resume);z=prepared.npz(a.prepared/'point_observations.npz')
    if sha(a.prepared/'point_observations.npz')!=m['point_input_sha256']:raise ValueError('Current source identity differs')
    scene=read(m['scene_path']);model=SceneParameterModel(scene,Path(m['scene_path']),optimize_scene=True)
    prepared.validate_snapshot(v,model.initial_state.layout,mp,m,z)
    core=NativeCore(m['native_library']);timeline=ShotMotionTimeline(m['shots'],m['control_dt'],m['first_control_time'])
    shot=next(s for s in m['shots'] if s['id']==a.shot);rig=v['rig'][shot['camera_index']]
    prefix='print_instance/'+a.instance+'::'+a.instance+'.'
    names=[prefix+s for s in ('top_edge','bottom_edge','left_vertical','right_vertical')]
    sampled=sample_scene(scene,core,Path(m['scene_path']),parameter_state=ParameterState(model.initial_state.layout,v['scene_values']),observed_path_ids=names)
    curves={c['id']:sampled['xyz'][c['begin']:c['end']] for c in sampled['cuts'] if c['id'] in names}
    if set(curves)!=set(names):raise ValueError('Missing native rectangular print edges')
    # Keep ordered geometry samples; source q50 coordinates are never changed.
    curves={n:x[np.unique(np.r_[np.arange(0,len(x),max(1,len(x)//64)),len(x)-1])] for n,x in curves.items()}
    groups=collections.defaultdict(dict);observed=set()
    for pt in np.unique(z['pts'][shot['row_begin']:shot['row_stop']]):observed.add(int(pt))
    for i,n in enumerate(mp['path_ids']):
        if n not in names:continue
        ii=np.flatnonzero((z['path_indices']==i)&(z['pts']>=shot['first_pts'])&(z['pts']<=shot['last_pts']))
        for pt in np.unique(z['pts'][ii]):groups[int(pt)][n]=z['uv'][ii[z['pts'][ii]==pt]]
    packets=[]
    for packet in a.packets:
        pp=prepared.npz(packet/'point_observations.npz');mm=read(packet/'mapping.json')
        for i,n in enumerate(mm['path_ids']):
            if n not in names:continue
            ii=np.flatnonzero(pp['path_indices']==i)
            for pt in np.unique(pp['pts'][ii]):groups[int(pt)][n]=pp['uv'][ii[pp['pts'][ii]==pt]]
        packets.append(dict(path=str(packet),point_sha256=sha(packet/'point_observations.npz')))
    full=np.arange(shot['first_pts'],shot['last_pts']+256,256,dtype=np.int64)
    times=(full-shot['reference_pts'])/12800+timeline.offset(shot)
    poses=core.evaluate_motion_kinematics(v['controls'],timeline.first,timeline.dt,times)['position']
    camera=core.evaluate_motion_camera(rig,v['controls'],timeline.first,timeline.dt,times)
    top,bottom=names[:2];vertical=curves[names[2]]
    height=float(np.ptp(vertical[:,2]));centre=np.mean(np.vstack(list(curves.values())),axis=0)
    if height<=0:raise ValueError('Native print has no vertical extent')
    out=guard_project_write(a.output.resolve())
    if out.exists():raise FileExistsError(out)
    fitted={};notes=[];previous=None
    with out.with_suffix('.jsonl').open('x',encoding='utf8',buffering=1) as log:
        for pt,g in sorted(groups.items()):
            if top not in g or bottom not in g:continue
            index=int((pt-full[0])//256)
            if index<0 or index>=len(full):raise ValueError('New source outside shot')
            edges=list(g);xyz=np.vstack([curves[n] for n in edges]);lengths=[len(curves[n]) for n in edges]
            bounds=np.r_[0,np.cumsum(lengths)]
            def residual(theta):
                projected=core.project_motion(rig,np.tile(theta,(6,1)),-2.,1.,1080,0.,xyz)
                if np.any(projected['status']) or not np.isfinite(projected['uv']).all():return np.full(sum(2*len(g[n]) for n in edges),1e5)
                values=[]
                for j,n in enumerate(edges):
                    uv=projected['uv'][bounds[j]:bounds[j+1]];left=uv[:-1];delta=np.diff(uv,axis=0)
                    d=g[n][:,None,:]-left[None,:,:];den=np.sum(delta*delta,axis=1)
                    alpha=np.clip(np.einsum('ijk,jk->ij',d,delta)/np.maximum(den,1e-300),0,1)
                    error=d-alpha[:,:,None]*delta;nearest=np.argmin(np.sum(error*error,axis=2),axis=1)
                    values.append(error[np.arange(len(error)),nearest].ravel())
                return np.concatenate(values)
            mid=float(np.median(np.r_[g[top][:,0],g[bottom][:,0]]))
            heights=[float(np.polyval(np.polyfit(g[n][:,0],g[n][:,1],1),mid)) for n in (top,bottom)]
            depth=float(camera['rotation'][index,2]@(centre-camera['centre'][index]))
            guess=poses[index].copy();guess[2]=np.log(abs(heights[1]-heights[0])*max(depth,1e-6)/height)
            candidates=[poses[index],guess]+([] if previous is None else [previous])
            initial=min(candidates,key=lambda x:float(residual(x)@residual(x)))
            fit=least_squares(residual,initial,x_scale=[.2,.1,1.],diff_step=1e-5,ftol=1e-8,xtol=1e-8,gtol=1e-6)
            errors=residual(fit.x)
            if not np.isfinite(fit.x).all() or np.max(abs(errors))>=1e4:raise ValueError('No finite source print pose at '+str(pt))
            fitted[pt]=fit.x;previous=fit.x
            note=dict(pts=pt,pose=fit.x.tolist(),actual_source_rows=sum(map(len,g.values())),paths=edges,initialization_rms=float(np.sqrt(np.mean(errors**2))),nfev=fit.nfev)
            notes.append(note);log.write(json.dumps(note)+'\n')
    # Interpolate only within an Ad3 observation component. Exposures carrying
    # other entities and no Ad3 divide the components; no cross-shot transfer.
    mask=np.array([int(pt) in groups or int(pt) not in observed for pt in full])
    initialized=poses.copy();segments=[]
    starts=np.flatnonzero(mask&~np.r_[False,mask[:-1]]);ends=np.flatnonzero(mask&~np.r_[mask[1:],False])
    for start,end in zip(starts,ends):
        anchors=np.array([pt for pt in fitted if full[start]<=pt<=full[end]],np.int64)
        if len(anchors)<2:continue
        sel=(full>=anchors[0])&(full<=anchors[-1])
        initialized[sel]=PchipInterpolator(anchors-anchors[0],np.array([fitted[int(p)] for p in anchors]),axis=0)(full[sel]-anchors[0])
        segments.append(dict(first_pts=int(anchors[0]),last_pts=int(anchors[-1]),source_pose_frames=len(anchors)))
    controls,chart=fit_pose_controls(full,initialized)
    if controls.shape!=v['controls'][shot['control_begin']:shot['control_stop']].shape or chart['knot_stride']!=shot['knot_stride']:raise ValueError('Motion chart changed')
    v['controls']=v['controls'].copy();v['controls'][shot['control_begin']:shot['control_stop']]=controls
    np.savez(out,**v)
    out.with_suffix('.json').write_text(json.dumps(dict(source_state=str(a.resume),source_state_sha256=sha(a.resume),output_sha256=sha(out),packets=packets,shot=a.shot,instance=a.instance,source_pose_frames=len(fitted),segments=segments,chart=chart,source_UV_changed=False,rig_scene_and_other_shots_changed=False,initialization_only=True,final_trajectory_postsmoothed=False),indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(output=str(out),source_pose_frames=len(fitted),segments=segments)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','resume','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--packets',nargs='+',type=Path,required=True);p.add_argument('--shot',required=True);p.add_argument('--instance',required=True)
    main(p.parse_args())
