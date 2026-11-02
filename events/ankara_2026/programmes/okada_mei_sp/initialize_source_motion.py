"""Seed endpoint motion from actual stationary-board texture in the source.

Tracked unknown texture is used only to obtain a nearby motion seed. It is not
added as fixed measured world points to the production joint fit.
"""
from pathlib import Path
import argparse,json,time
import av,cv2,numpy as np
from scipy.optimize import least_squares
import prepare_native_multicamera as prepared
from audit_native_source_motion import board_intersections,stats
from rinkspace.core.native import NativeCore
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline,continue_exposure_halos
from rinkspace.domain.motion_chart import UniformMotionChart
from rinkspace.domain.parameters import SceneParameterModel
from rinkspace.path_policy import guard_project_write


def strip_mask(points,names,pts,person_mask):
    mask=np.zeros((1080,1920),np.uint8);present=set(names[points['pts']==pts])
    pairs=[('board/continuous/top','board/continuous/bottom')]
    for path in sorted(present):
        if path.startswith('print_instance/') and path.endswith('.top_edge/path0'):
            other=path.replace('.top_edge/path0','.bottom_edge/path0')
            if other in present:pairs.append((path,other))
    for pair in pairs:
        edges=[]
        for path,offset in zip(pair,(8,-12)):
            uv=points['uv'][(points['pts']==pts)&(names==path)]
            if len(uv)<2:break
            uv=uv[np.argsort(uv[:,0])];edges.append(np.interp(np.arange(1920),uv[:,0],uv[:,1],left=np.nan,right=np.nan)+offset)
        if len(edges)!=2:continue
        for x in range(16,1904):
            if np.isfinite([edges[0][x],edges[1][x]]).all():
                lo=max(0,int(edges[0][x]));hi=min(1080,int(edges[1][x]))
                if hi>lo:mask[lo:hi,x]=255
    if person_mask.shape!=mask.shape:raise ValueError('Source person mask dimensions differ')
    mask[person_mask>0]=0
    return mask


def main(a):
    started=time.monotonic();out=guard_project_write(a.output.resolve())
    if out.exists() or out.with_suffix('.json').exists():raise FileExistsError(out)
    m=prepared.read(a.prepared/'joint_manifest.json');mapping=prepared.read(a.prepared/'mapping.json')
    z=prepared.npz(a.prepared/'point_observations.npz');v=prepared.npz(a.state)
    model=SceneParameterModel(prepared.read(m['scene_path']),Path(m['scene_path']),optimize_scene=True)
    prepared.validate_snapshot(v,model.initial_state.layout,mapping,m,z)
    if prepared.digest(a.prepared/'point_observations.npz')!=m['point_input_sha256']:raise ValueError('Current input changed')
    core=NativeCore(m['native_library']);timeline=ShotMotionTimeline(m['shots'],m['control_dt'],m['first_control_time'])
    names=np.asarray(mapping['path_ids'])[z['path_indices']];result={k:x.copy() for k,x in v.items()};notes=[];tracked={}
    for specification in a.window:
        sid,start,stop,anchor=specification.split(':');start,stop,anchor=map(int,(start,stop,anchor));shot=next(s for s in m['shots'] if s['id']==sid)
        if start<shot['first_pts'] or stop>shot['last_pts'] or not start<=anchor<=stop or (stop-start)%256 or (anchor-start)%256:raise ValueError('Endpoint source window required')
        frames={}
        with av.open(str(a.source)) as container:
            stream=container.streams.video[0]
            if str(stream.time_base)!='1/12800':raise ValueError('Wrong source clock')
            container.seek(start,stream=stream,backward=True)
            for frame in container.decode(stream):
                if frame.pts>stop:break
                if frame.pts>=start:frames[int(frame.pts)]=frame.to_ndarray(format='gray')
        if sorted(frames)!=list(range(start,stop+1,256)):raise ValueError('Missing exact source exposure')
        person_masks={}
        for pts in frames:
            path=a.person_masks/sid/(str(pts)+'.png')
            if sid in a.central_exclusion_shot:
                mask=np.zeros((1080,1920),np.uint8);mask[:,500:1240]=255
            else:
                mask=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
                if mask is None:raise ValueError('Missing same-exposure person mask: '+str(path))
            person_masks[pts]=mask
        mask=strip_mask(z,names,anchor,person_masks[anchor]);features=cv2.goodFeaturesToTrack(frames[anchor],0,.005,8,mask=mask)
        if features is None or len(features)<12:raise ValueError('Insufficient stationary source texture')
        tracks={anchor:features[:,0].astype(float)};valid={anchor:np.ones(len(features),bool)}
        orders=(range(anchor-256,start-1,-256),range(anchor+256,stop+1,256))
        print(json.dumps(dict(shot=sid,anchor=anchor,features=len(features),anchor_mask_pixels=int(np.count_nonzero(mask)))),flush=True)
        for order in orders:
            previous=anchor
            for pts in order:
                p=tracks[previous].astype(np.float32).reshape(-1,1,2)
                nxt,ok,_=cv2.calcOpticalFlowPyrLK(frames[previous],frames[pts],p,None,winSize=(31,31),maxLevel=5,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,60,.001))
                back,bok,_=cv2.calcOpticalFlowPyrLK(frames[pts],frames[previous],nxt,None,winSize=(31,31),maxLevel=5,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,60,.001))
                good=valid[previous]&ok[:,0].astype(bool)&bok[:,0].astype(bool)&(np.linalg.norm(back[:,0]-p[:,0],axis=1)<1.)
                q=nxt[:,0];inside=(q[:,0]>=0)&(q[:,0]<1919)&(q[:,1]>=0)&(q[:,1]<1079);good&=inside
                indices=np.flatnonzero(inside);good[indices]&=person_masks[pts][np.rint(q[indices,1]).astype(int),np.rint(q[indices,0]).astype(int)]==0
                tracks[pts]=q.astype(float);valid[pts]=good;previous=pts
                print(json.dumps(dict(shot=sid,tracked_pts=pts,valid_tracks=int(good.sum()),forward_backward_valid=int((ok[:,0].astype(bool)&bok[:,0].astype(bool)&(np.linalg.norm(back[:,0]-p[:,0],axis=1)<1.)).sum()))),flush=True)
        rig=v['rig'][shot['camera_index']];anchor_time=(anchor-shot['reference_pts'])/12800+timeline.offset(shot)
        good,xyz=board_intersections(core,rig,v['controls'],timeline.first,timeline.dt,anchor_time,tracks[anchor],v['board'])
        print(json.dumps(dict(shot=sid,anchor_board_intersections=int(good.sum()))),flush=True)
        ids=np.flatnonzero(good);xyz_by_id={i:p for i,p in zip(ids,xyz,strict=True)}
        tt=[];xx=[];yy=[];slices={};begin=0
        for pts in sorted(frames):
            chosen=np.flatnonzero(valid[pts]&good)
            if len(chosen)<12:raise ValueError('Insufficient consistent source tracks at '+str(pts))
            count=len(chosen);tt.extend([(pts-shot['reference_pts'])/12800+timeline.offset(shot)]*count);xx.extend(xyz_by_id[i] for i in chosen);yy.extend(tracks[pts][chosen]);slices[pts]=(begin,begin+count);begin+=count
        tt=np.asarray(tt);xx=np.asarray(xx);yy=np.asarray(yy)
        initial_projection=core.project_motion(rig,result['controls'],timeline.first,timeline.dt,1080,tt,xx)
        admitted=initial_projection['status']==0
        excluded_native_rows=int(np.count_nonzero(~admitted));new_slices={};offset=0
        for pts,(begin,end) in slices.items():
            count=int(np.count_nonzero(admitted[begin:end]))
            print(json.dumps(dict(shot=sid,initial_native_pts=pts,valid_projection_rows=count,invalid_projection_rows=end-begin-count)),flush=True)
            if count<12:raise ValueError('Insufficient valid native seed projections at '+str(pts))
            new_slices[pts]=(offset,offset+count);offset+=count
        tt,xx,yy=tt[admitted],xx[admitted],yy[admitted];slices=new_slices
        chart=UniformMotionChart.create(shot['control_stop']-shot['control_begin'],shot['first_control_time'],timeline.dt,a.initializer_stride,shot['motion_reference_time'])
        local,labels=chart.directions(reference_angle_gauge=False);chosen=[]
        full_interval=[(shot[k]-shot['reference_pts'])/12800 for k in ('first_pts','last_pts')]
        local=continue_exposure_halos(local,shot['first_control_time'],timeline.dt,full_interval)
        rel_anchor=(anchor-shot['reference_pts'])/12800;half=abs(rig[37])/2
        window_lo=(start-shot['reference_pts'])/12800;window_hi=(stop-shot['reference_pts'])/12800
        for j,label in enumerate(labels):
            support=np.flatnonzero(np.any(local[:,:,j]!=0,axis=1))
            if not len(support):continue
            lo=shot['first_control_time']+(support.min()-2)*timeline.dt;hi=shot['first_control_time']+(support.max()+2)*timeline.dt
            if (start==shot['first_pts'] or lo>window_lo+half) and (stop==shot['last_pts'] or hi<window_hi-half):chosen.append(j)
        # Restrict to columns that actually touch a requested exposure. No
        # global rig/scene coordinate is changed by this initializer.
        sample_lo=(start-shot['reference_pts'])/12800-half;sample_hi=(stop-shot['reference_pts'])/12800+half
        chosen=[j for j in chosen if shot['first_control_time']+(np.flatnonzero(np.any(local[:,:,j]!=0,axis=1)).max()+2)*timeline.dt>sample_lo and shot['first_control_time']+(np.flatnonzero(np.any(local[:,:,j]!=0,axis=1)).min()-2)*timeline.dt<sample_hi]
        direction=local[:,:,chosen]*.01;base=result['controls'].copy();sl=slice(shot['control_begin'],shot['control_stop']);calls=0;invalid_trials=0
        def project(x):
            nonlocal invalid_trials
            controls=base.copy();controls[sl]+=np.einsum('ijk,k->ij',direction,x)
            p=core.project_motion(rig,controls,timeline.first,timeline.dt,1080,tt,xx)
            if np.any(p['status']):
                invalid_trials+=1
                # A finite rejecting oracle lets the trust region shrink.
                # These are auxiliary initializer trials, never fit UV.
                return yy+1e6
            return p['uv']
        def residual(x):
            nonlocal calls
            r=(project(x)-yy).ravel();calls+=1
            if calls%100==1:print(json.dumps(dict(shot=sid,projection_calls=calls,source_rms=float(np.sqrt(np.mean(r*r))))),flush=True)
            return r
        initial=np.zeros(len(chosen));before=project(initial);solve=least_squares(residual,initial,x_scale=1.,loss='soft_l1',f_scale=1.,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        after=project(solve.x);result['controls'][sl]+=np.einsum('ijk,k->ij',direction,solve.x)
        if np.linalg.norm(after-yy)>=np.linalg.norm(before-yy):raise RuntimeError('Endpoint source-motion initialization did not improve')
        anchor_rows=slice(*slices[anchor])
        if anchor in (start,stop) and np.max(abs(after[anchor_rows]-before[anchor_rows]))>1e-8:raise RuntimeError('Boundary anchor pose changed')
        per_frame={str(pts):dict(tracks=end-begin,before=stats(np.linalg.norm(before[begin:end]-yy[begin:end],axis=1)),after=stats(np.linalg.norm(after[begin:end]-yy[begin:end],axis=1))) for pts,(begin,end) in slices.items()}
        notes.append(dict(shot=sid,initializer_stride=a.initializer_stride,endpoint_cubic_continuation_used_in_update_basis=True,anchor_is_interior=anchor not in (start,stop),anchor_projection_change=stats(np.linalg.norm(after[anchor_rows]-before[anchor_rows],axis=1)),production_knot_stride_unchanged=shot['knot_stride'],source_mask_mode='conservative_middle_strip' if sid in a.central_exclusion_shot else 'existing_same_exposure_person_mask',start_pts=start,stop_pts=stop,anchor_pts=anchor,active_local_motion_columns=len(chosen),initial_native_invalid_tracking_rows=excluded_native_rows,rejected_invalid_projection_trials=invalid_trials,projection_calls=calls,solver_success=bool(solve.success),solver_message=str(solve.message),before=stats(np.linalg.norm(before-yy,axis=1)),after=stats(np.linalg.norm(after-yy,axis=1)),frames=per_frame))
        tracked[sid+'_frame_times']=tt;tracked[sid+'_source_uv']=yy;tracked[sid+'_anchor_derived_xyz']=xx;tracked[sid+'_before_uv']=before;tracked[sid+'_after_uv']=after
        print(json.dumps(notes[-1]),flush=True)
        prepared.validate_snapshot(result,model.initial_state.layout,mapping,m,z)
        np.savez(out.with_suffix('.'+sid+'.npz'),**result)
        prepared.write(out.with_suffix('.'+sid+'.json'),dict(previous_state=str(a.state),previous_state_sha256=prepared.digest(a.state),notes=[notes[-1]],rig_scene_stations_unchanged=True,source_UV_PTS_unchanged=True,tracks_added_to_production_fit=False))
    for key in ('rig','scene_values','stations','source_rows'):
        if result[key].tobytes()!=v[key].tobytes():raise RuntimeError('Motion initializer changed '+key)
    prepared.validate_snapshot(result,model.initial_state.layout,mapping,m,z)
    np.savez(out,**result);np.savez_compressed(out.with_suffix('.tracks.npz'),**tracked)
    receipt=dict(schema='rinkspace.source_motion_initialization.v1',previous_state=str(a.state.resolve()),previous_state_sha256=prepared.digest(a.state),source=str(a.source.resolve()),source_sha256=prepared.digest(a.source),point_input_sha256=m['point_input_sha256'],notes=notes,rig_scene_stations_unchanged=True,source_UV_PTS_unchanged=True,tracks_added_to_production_fit=False,unknown_texture_world_points_are_conditional_initialization_only=True,elapsed_seconds=time.monotonic()-started)
    prepared.write(out.with_suffix('.json'),receipt)
    print(json.dumps(dict(output=str(out),elapsed_seconds=receipt['elapsed_seconds'])),flush=True)



def finalize_seed(prepared_root,state_path,output,windows):
    from copy import deepcopy
    from prepare_spatial_refinement import _initialize_touched
    from rinkspace.domain.multi_shot_motion import continue_exposure_halos
    out=guard_project_write(Path(output).resolve())
    if out.exists():raise FileExistsError(out)
    root=Path(prepared_root);m=prepared.read(root/'joint_manifest.json');mapping=prepared.read(root/'mapping.json')
    z=prepared.npz(root/'point_observations.npz');v=prepared.npz(state_path);result={k:x.copy() for k,x in v.items()}
    model=SceneParameterModel(prepared.read(m['scene_path']),Path(m['scene_path']),optimize_scene=True)
    prepared.validate_snapshot(v,model.initial_state.layout,mapping,m,z)
    touched=np.zeros(len(z['pts']),bool);closure=[]
    for sid,aa,bb in windows:
        shot=next(s for s in m['shots'] if s['id']==sid);sl=slice(shot['control_begin'],shot['control_stop'])
        timeline=ShotMotionTimeline(m['shots'],m['control_dt'],m['first_control_time']);interval=timeline.local_exposure_time(shot,np.array([shot['first_pts'],shot['last_pts']],np.int64))
        before=result['controls'][sl].copy();result['controls'][sl]=continue_exposure_halos(before,shot['first_control_time'],m['control_dt'],interval)
        closure.append(dict(shot=sid,changed_halo_coefficients=int(np.count_nonzero(before!=result['controls'][sl]))))
        touched[shot['row_begin']:shot['row_stop']]=(z['pts'][shot['row_begin']:shot['row_stop']]>=aa)&(z['pts'][shot['row_begin']:shot['row_stop']]<=bb)
    notes=_initialize_touched(z,result['stations'],m['shots'],touched,result,m,mapping,model)
    if not np.array_equal(result['stations'][~touched],v['stations'][~touched]):raise RuntimeError('Untouched stations changed')
    if not np.array_equal(result['stations'][z['kind']==0],v['stations'][z['kind']==0]):raise RuntimeError('Fixed material point changed')
    prepared.validate_snapshot(result,model.initial_state.layout,mapping,m,z)
    np.savez(out,**result)
    receipt=dict(schema='rinkspace.endpoint_seed_finalization.v1',parent_state=str(state_path),parent_state_sha256=prepared.digest(state_path),source_UV_PTS_identity_unchanged=True,rig_scene_unchanged=True,station_initialization=notes,endpoint_cubic_closure=closure,source_windows=[list(w) for w in windows],output=str(out),output_sha256=prepared.digest(out))
    prepared.write(out.with_suffix('.json'),receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','state','source','output','person-masks'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--window',action='append',required=True,help='shot:start_integer_PTS:stop_integer_PTS:anchor_integer_PTS')
    p.add_argument('--initializer-stride',type=int,default=1)
    p.add_argument('--central-exclusion-shot',action='append',default=[])
    main(p.parse_args())
