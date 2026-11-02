"""Initialize existing endpoint motion from raw-source consecutive displacements.

Unknown board texture is conditional initializer geometry, never fixed fit data.
The returned zoom shape may be explicitly retained in a later deficient interval.
"""
from pathlib import Path
import argparse,json,time,cv2,numpy as np
from scipy.linalg import null_space
from scipy.optimize import least_squares
import prepare_native_multicamera as p
from audit_native_source_motion import board_intersections
from rinkspace.core.native import NativeCore
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline,continue_exposure_halos
from rinkspace.domain.motion_chart import UniformMotionChart,cubic_samples
from rinkspace.domain.parameters import SceneParameterModel
from rinkspace.path_policy import guard_project_write


def main(a):
    started=time.monotonic();out=guard_project_write(a.output.resolve())
    if out.exists() or out.with_suffix('.json').exists():raise FileExistsError(out)
    m=p.read(a.prepared/'joint_manifest.json');mapping=p.read(a.prepared/'mapping.json');z=p.npz(a.prepared/'point_observations.npz');v=p.npz(a.state);model=SceneParameterModel(p.read(m['scene_path']),Path(m['scene_path']),optimize_scene=True);p.validate_snapshot(v,model.initial_state.layout,mapping,m,z)
    report=p.read(a.source_flow);arrays=p.npz(a.source_flow.with_suffix('.npz'))
    if report['identities'][str(a.state.resolve())]!=p.digest(a.state) or report['identities'][str((a.prepared/'point_observations.npz').resolve())]!=m['point_input_sha256']:raise ValueError('Flow diagnostic is not tied to this state and current source input')
    sid,start,stop,anchor=a.window.split(':');start,stop,anchor=map(int,(start,stop,anchor));shot=next(s for s in m['shots'] if s['id']==sid);core=NativeCore(m['native_library']);tl=ShotMotionTimeline(m['shots'],m['control_dt'],m['first_control_time']);rig=v['rig'][shot['camera_index']];xx=[];tt=[];next_tt=[];target=[];counts=[]
    for rec in report['records']:
        if rec['shot']!=sid or not start<=rec['pts']<stop or rec['status']!='compared':continue
        pts=rec['pts'];old=arrays[f'{sid}_{pts}_source_uv'];new=arrays[f'{sid}_{pts}_next_source_uv'];H,inliers=cv2.findHomography(old,new,cv2.RANSAC,2.,maxIters=10000,confidence=.999)
        if H is None:raise ValueError('No coherent source displacement')
        keep=inliers[:,0].astype(bool);old,new=old[keep],new[keep];t=tl.exposure_time(shot,pts);good,xyz=board_intersections(core,rig,v['controls'],tl.first,tl.dt,t,old,v['board']);old,new=old[good],new[good]
        if len(old)<12:raise ValueError('Insufficient source board displacements')
        xx.extend(xyz);tt.extend([t]*len(xyz));next_tt.extend([tl.exposure_time(shot,pts+256)]*len(xyz));target.extend(new-old);counts.append(dict(pts=pts,source_tracks=rec['tracks'],coherent_board_tracks=len(old)))
    if not counts:raise ValueError('No source-supported consecutive pairs')
    xx,tt,next_tt,target=np.asarray(xx),np.asarray(tt),np.asarray(next_tt),np.asarray(target);sl=slice(shot['control_begin'],shot['control_stop']);local,labels=UniformMotionChart.create(shot['control_stop']-shot['control_begin'],shot['first_control_time'],tl.dt,shot['knot_stride'],shot['motion_reference_time']).directions(reference_angle_gauge=False);full=tl.local_exposure_time(shot,np.array([shot['first_pts'],shot['last_pts']],np.int64));local=continue_exposure_halos(local,shot['first_control_time'],tl.dt,full);hi=tl.local_exposure_time(shot,stop);selected=[]
    for j in range(local.shape[2]):
        support=np.flatnonzero(np.any(local[:,:,j]!=0,axis=1))
        if len(support) and shot['first_control_time']+(support.max()+2)*tl.dt<hi:selected.append(j)
    raw=local[:,:,selected];indices,weights=cubic_samples(len(raw),shot['first_control_time'],tl.dt,np.array([tl.local_exposure_time(shot,anchor)]));A=np.einsum('ij,ijkm->km',weights,raw[indices]);N=null_space(A);direction=np.einsum('ijk,km->ijm',raw,N)*.001;base=v['controls'].copy();calls=0
    def residual(x):
        nonlocal calls
        controls=base.copy();controls[sl]+=np.einsum('ijk,k->ij',direction,x);at=[core.project_motion(rig,controls,tl.first,tl.dt,1080,frame_times,xx) for frame_times in (tt,next_tt)];calls+=1
        if any(np.any(q['status']) for q in at):return np.full(target.size,1e6)
        r=(at[1]['uv']-at[0]['uv']-target).ravel()
        if calls%100==1:print(json.dumps(dict(projection_calls=calls,pair_displacement_rms=float(np.sqrt(np.mean(r*r))))),flush=True)
        return r
    initial=np.zeros(direction.shape[2]);before=residual(initial);opt=least_squares(residual,initial,loss='soft_l1',f_scale=1.,ftol=1e-9,xtol=1e-9,gtol=1e-8);after=residual(opt.x);result={k:x.copy() for k,x in v.items()};result['controls'][sl]+=np.einsum('ijk,k->ij',direction,opt.x)
    if np.linalg.norm(after)>=np.linalg.norm(before):raise ValueError('Source displacement initializer did not improve')
    for key in ('rig','scene_values','stations','source_rows'):
        if not np.array_equal(result[key],v[key]):raise ValueError('Initializer changed '+key)
    np.savez(out,**result);p.write(out.with_suffix('.json'),dict(schema='rinkspace.source_flow_motion_initialization.v1',parent_state=str(a.state),parent_state_sha256=p.digest(a.state),source_flow=str(a.source_flow),source_flow_sha256=p.digest(a.source_flow),source_pairs_sha256=p.digest(a.source_flow.with_suffix('.npz')),window=a.window,counts=counts,active_initialization_columns=direction.shape[2],existing_knot_stride=shot['knot_stride'],conditional_board_texture_not_added_as_observations=True,source_uv_pts_unchanged=True,rig_scene_stations_unchanged=True,anchor_motion_values_preserved=True,before_pair_rms=float(np.sqrt(np.mean(before**2))),after_pair_rms=float(np.sqrt(np.mean(after**2))),projection_calls=calls,solver_message=str(opt.message),output_sha256=p.digest(out),elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(output=str(out),before_rms=float(np.sqrt(np.mean(before**2))),after_rms=float(np.sqrt(np.mean(after**2))),columns=direction.shape[2],elapsed_s=time.monotonic()-started)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','state','source-flow','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--window',required=True);main(parser.parse_args())
