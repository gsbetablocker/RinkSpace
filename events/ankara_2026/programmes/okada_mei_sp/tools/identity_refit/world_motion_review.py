"""Compare source-clock world trajectories and independently observed camera motion."""
import argparse
import json
from pathlib import Path

import numpy as np

from compare import bind
from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference, curve_from_packet
from calibration.program_tracking.body_tracking.continuous_physics import ContinuousMassGeometry
from calibration.program_tracking.body_tracking.skate_physics import batch_features, fk_state
from joint_source_objective import material_points
from source_review import digest
from source_camera_rebind import write


def stats(values):
    x=np.asarray(values);x=x[np.isfinite(x)]
    return dict(n=len(x),**({k:float(v) for k,v in zip(('median','p95','p99','max'),np.percentile(x,[50,95,99,100]))} if len(x) else {}))


def paths(ref,curve,t):
    ref=absolute_reference(ref,curve);mass=ContinuousMassGeometry(ref)
    materials=material_points(ref.binding)[[3,4]]
    rows=[]
    for a in range(0,len(t),240):
        sample=t[a:a+240]
        s=fk_state(ref,ref.at(sample),np.zeros((len(sample),69)));c,_=mass.from_pose(s)
        shoe=(s['O'][:,:,None]+np.einsum('nfij,kj->nfki',s['R'],materials))*ref.binding.ratio
        feat=batch_features(ref.binding,s['R'],s['O']);mask=np.isin(feat['semantics'],['runner','toe_pick'])
        rows.append((mass.com(c),shoe,feat['height'][:,:,mask].min(-1)))
    return tuple(np.concatenate(x) for x in zip(*rows))


def physics(com,blade,height,pts,shot,episodes):
    seconds=pts/12800;dt=np.diff(seconds);assert np.allclose(dt,dt[0]);dt=float(dt[0])
    a=np.diff(com,n=2,axis=0)/dt**2
    result={}
    for key,values in [('com',com),('ground_com',com[:,:2]),('blade_fixed',blade)]:
        for k,units in [(1,'step_m'),(2,'acc_mps2'),(3,'jerk_mps3')]:
            dx=np.diff(values,n=k,axis=0)/(dt**k if k>1 else 1.)
            norm=np.linalg.norm(dx,axis=-1);imax=np.unravel_index(np.argmax(norm),norm.shape)
            result[key+'_'+units]=dict(**stats(norm),source_seconds=seconds[imax[0]+(1 if k==2 else k)],component=list(map(int,imax[1:])))
    cut=np.flatnonzero(shot[:-1]!=shot[1:])+1
    result['cuts']=[dict(source_seconds=seconds[i],com_step_m=float(np.linalg.norm(com[i]-com[i-1])),
                         max_blade_step_m=float(np.linalg.norm(blade[i]-blade[i-1],axis=-1).max())) for i in cut]
    result['flight']={};result['support']={}
    for ep in episodes:
        inside=(pts>=ep['start_pts'])&(pts<=ep['end_pts_inclusive'])
        k=ep['episode_id']
        if ep['kind']=='flight_visible_candidate':
            active=inside[:-2]&inside[1:-1]&inside[2:]
            result['flight'][k]=stats(np.linalg.norm(a[active]-[0,0,-9.81],axis=-1))
        if ep['kind']=='support_candidate' and ep.get('foot') in (0,1):
            foot=ep['foot']
            result['support'][k]=stats(np.abs(height[inside,foot]))
    result['max_penetration_m']=float(max(0,-height.min()))
    result['over_3cm_penetration_exposures']=int((height.min(1)<-.03).sum())
    return result


def camera_audit(programme,ref,cam,baseline_com):
    reg=json.loads((programme/'s14_boards_1118.json').read_text(encoding='utf8'))
    def checked(record):
        path=Path(record['path'])
        if digest(path)!=record['sha256']:raise ValueError(path)
        return path
    pixel=json.loads(checked(reg['pixel_review']).read_text(encoding='utf8'))
    receipt=json.loads(checked(pixel['full_check']).read_text(encoding='utf8'))
    audit=checked(receipt['prior_evidence']['motion']);manifest=json.loads(audit.read_text(encoding='utf8'))
    pts=ref.p['pts'];n=len(pts);acc=np.diff(baseline_com,n=2,axis=0)/.02**2
    centre=cam.midpoint['centre']*cam.scale*ref.binding.ratio
    # Confirm which coordinate convention the native midpoint actually uses.
    np.testing.assert_allclose(cam.midpoint['centre'],cam.midpoint['values'][:,13:16])
    line=baseline_com[1:-1]-centre[1:-1];line/=np.linalg.norm(line,axis=1)[:,None]
    parallel=(line*acc).sum(1)
    p95=np.percentile(np.linalg.norm(acc,axis=1),95)
    high=np.r_[False,np.linalg.norm(acc,axis=1)>=p95,False]
    near_cut=np.zeros(n,bool)
    for i in np.flatnonzero(cam.shot_ids[1:]!=cam.shot_ids[:-1])+1:near_cut[max(0,i-2):i+3]=True
    shot_records={};merged={}
    for name in reg['shots']:
        if not receipt['shots'][name]['camera_projection_identical']:raise ValueError('Camera audit needs reprojection')
        state='lane4_side_1118/verified'
        if state not in manifest['shots'][name]:raise ValueError('Unbound camera')
        mask=cam.shot_ids==name.split('_')[-1]
        indices=np.flatnonzero(mask)
        cam_values={}
        with np.load(audit.parent/'frames'/(name+'.npz')) as z:
            np.testing.assert_array_equal(pts[mask],z['pts'])
            for key in ('board_adjacent','ice_adjacent','board_jitter','ice_jitter'):
                v=z['1_'+key];cam_values[key]=stats(v)
                merged.setdefault(key,np.full(n,np.nan))[mask]=v
        with np.load(checked(pixel['shots'][name])) as z,np.load(checked(reg['fit_observations'][name])) as obs:
            if len(z['error'])!=len(obs['pts']):raise ValueError('Pixel row ledger mismatch')
            bad_frames=pts[mask&high]
            for family in ('ice','board'):
                classes=np.asarray(obs['family']).astype(str)
                board=np.char.startswith(classes,'board.')|np.char.startswith(classes,'Ad')
                # Ankara's remaining registered families are rink paint; exclude
                # unrecognized classes from a physical-surface claim.
                import re
                known=[s for s in np.unique(classes) if re.fullmatch(r'(?:[BRFC][0-9]*|lane[0-9]+|house_connection_.+)',s)]
                keep=board if family=='board' else np.isin(classes,known)
                cam_values[family+'_geometry_residual_px']=stats(z['error'][keep])
            cam_values['static_geometry_on_body_spikes_px']=stats(z['error'][np.isin(obs['pts'],bad_frames)])
        shot_records[name]=cam_values
    velocity=np.linalg.norm(np.diff(baseline_com,axis=0),axis=-1)
    cases=set([24.02,182.82,float(pts[np.argmax(velocity)+1]/12800),111.44,53.5,113.92])
    for name in reg['shots']:
        ix=np.flatnonzero(cam.shot_ids[1:-1]==name.split('_')[-1]);cases.add(pts[ix[np.linalg.norm(acc[ix],axis=1).argmax()]+1]/12800)
    cases=sorted(cases)
    def value(v):return float(v) if np.isfinite(v) else None
    return dict(version=reg['version_id'],field=reg['geometry_binding']['version_id'],camera_sha256=reg['camera']['sha256'],
        static_motion_source=str(audit),static_motion_source_sha256=digest(audit),unchanged_cameras=reg['shots'],
        high_acceleration_frames=int(high.sum()),within_two_exposures_of_cut=int((high&near_cut).sum()),
        previous_fraction_acceleration_energy_along_view=float(parallel@parallel/np.sum(acc**2)),
        per_shot=shot_records,pooled={k:dict(all=stats(v),body_spikes=stats(v[high])) for k,v in merged.items()},
        specific_frames=[dict(source_seconds=t,**{k:value(v[np.searchsorted(pts,round(t*12800))]) for k,v in merged.items()}) for t in cases],
        limitations='Material flow is measured on the original broadcast independently of the body. It was used in camera development. Missing textured patches and uncertain depth are retained.')


def phase_details(packet,episodes):
    from calibration.program_tracking.body_tracking.centroidal_physics_diagnostics import flight_projection
    pts=packet['source_pts'];com=packet['com_world_mid'];H=packet['com_specific_angular_momentum_world_m2ps']
    rot=packet['skate_R_world_mid'];origin=packet['skate_origin_world_mid'];witness=packet['skate_witness_world_mid']
    ans={}
    for ep in episodes:
        ix=np.flatnonzero((pts>=ep['start_pts'])&(pts<=ep['end_pts_inclusive']))
        if len(ix)<3:continue
        label=ep['episode_id'];t=pts[ix]/12800
        if ep['kind']=='flight_visible_candidate':
            arc=flight_projection(t-t[0],com[ix],[0,0,-9.81])
            moment=H[ix[1:-1]];average=moment.mean(0)
            ans[label]=dict(kind='reviewed_flight_interior',seconds=t[[0,-1]].tolist(),
                ballistic_rms_m=arc['position_error_rms_m'],ballistic_max_m=arc['position_error_max_m'],
                specific_angular_momentum_m2ps=average.tolist(),
                angular_momentum_spread_m2ps=stats(np.linalg.norm(moment-average,axis=1)))
        if ep['kind']=='support_candidate' and ep.get('foot') in (0,1):
            f=ep['foot'];i=ix[1:-1]
            # Evaluate a fixed steel point through adjacent exposures even if
            # the lowest-point witness selected by the next frame changes.
            local=(rot[i,f].swapaxes(1,2)@(witness[i,f]-origin[i,f])[:,:,None])
            after=origin[i+1,f]+(rot[i+1,f]@local)[:,:,0]
            before=origin[i-1,f]+(rot[i-1,f]@local)[:,:,0]
            v=(after-before)/((pts[i+1]-pts[i-1])/12800)[:,None]
            axis=rot[i,f,:2,0];side=np.stack([-axis[:,1],axis[:,0]],1)
            side/=np.maximum(np.linalg.norm(side,axis=1,keepdims=True),1e-8)
            ans[label]=dict(kind='source_support_candidate',foot=f,seconds=t[[0,-1]].tolist(),
                blade_lateral_speed_mps=stats(abs((v[:,:2]*side).sum(1))))
    return ans


def run(args):
    ref,cam,_,_,_=bind(args.prepared,args.parent/'source.json',args.parent/'candidate.npz')
    ep=ref.source_document['source_episode_review']['episodes'];pts=ref.p['pts']
    rr={};n=len(pts);materials=material_points(ref.binding)[[3,4]]
    for label,path,heights in [('early',args.early/'fitted/result.npz',args.shared/'early_source_measurements.npz'),
            ('calibration',args.calibration/'result.npz',args.shared/'before_source_measurements.npz'),
            ('previous',args.parent/'result.npz',args.parent/'joint_source_measurements.npz')]:
        with np.load(path) as z,np.load(heights) as h:
            np.testing.assert_array_equal(z['source_pts'],pts)
            com=z['com_world_mid'];blade=z['skate_origin_world_mid'][:,:,None]+np.einsum('nfij,kj->nfki',z['skate_R_world_mid'],materials)*ref.binding.ratio
            rr[label]=physics(com,blade,h['height'],pts,cam.shot_ids,ep)
            if label=='previous':
                camera=camera_audit(args.programme,ref,cam,com)
                proj=z['com_projection_world_mid'].copy();np.testing.assert_array_equal(proj[:,:2],com[:,:2]);assert not proj[:,2].any()
    with np.load(args.output/'candidate.npz') as z:curve=curve_from_packet(z)
    # Evaluate three additional instants per exposure directly on the spline.
    t=np.linspace(ref.times[0],ref.times[-1],4*n-3);com,blade,height=paths(ref,curve,t)
    rr['current']=physics(com[::4],blade[::4],height[::4],pts,cam.shot_ids,ep)
    rr['current_200hz']=physics(com,blade,height,pts[0]+np.arange(len(t))*64,cam.shot_ids[np.arange(len(t))//4],ep)
    rr['per_shot']={}
    phases={}
    for key,path in [('early',args.early/'fitted/result.npz'),('calibration',args.calibration/'result.npz'),
            ('previous',args.parent/'result.npz'),('current',args.output/'result.npz')]:
        if path.is_file():
            with np.load(path) as z:phases[key]=phase_details(z,ep)
    with np.load(args.parent/'result.npz') as z,np.load(args.parent/'joint_source_measurements.npz') as h:
        oldc=z['com_world_mid'];oldb=z['skate_origin_world_mid'][:,:,None]+np.einsum('nfij,kj->nfki',z['skate_R_world_mid'],materials)*ref.binding.ratio
        for shot in dict.fromkeys(cam.shot_ids):
            ix=cam.shot_ids==shot
            rr['per_shot'][str(shot)]={key:physics(c[ix],b[ix],hh[ix],pts[ix],cam.shot_ids[ix],[]) for key,c,b,hh in
                [('previous',oldc,oldb,h['height']),('current',com[::4],blade[::4],height[::4])]}
    write(args.output/'world_review.json',dict(candidate=digest(args.output/'candidate.npz'),physics=rr,camera=camera,phases=phases))
    print('Validated',rr['previous']['com_acc_mps2'],rr['current']['com_acc_mps2'],rr['current']['blade_fixed_acc_mps2'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('prepared','parent','shared','early','calibration','programme','output'):
        p.add_argument(k,type=lambda s:Path(s).resolve())
    run(p.parse_args())
