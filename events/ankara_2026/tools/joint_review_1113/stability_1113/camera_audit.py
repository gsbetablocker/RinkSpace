"""Evaluate published trajectories on the registered exposure clock."""
from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joint_inputs as ji
import solve_joint as sj
import verify_joint as vj
import numpy as np
from scipy.spatial.transform import Rotation

HERE = ji.OUT/'stability_1113'
LENS = ('fx','fy','cx','cy','k1','k2','p1','p2')
PREP = HERE.parent/'prepared_guarded'


def setup(path=HERE.parent/'final/review.npz'):
    meta = ji.read(PREP/'setup.json')
    core = sj.NativeCore(meta['core'])
    history = vj.load_checkpoint(PREP/'baseline.npz')
    state = vj.load_checkpoint(path)
    return meta, core, history, state


def native_scene(meta, core, state):
    return sj.NativeSceneGPU(meta['scene_path'], core,
        parameter_state=sj.ParameterState(meta['scene_layout'],state['scene_values']))


def at_exposures(core, shot, cs, rigs, indices, times):
    unique, ix = np.unique(rigs[indices], axis=0, return_inverse=True)
    values = np.empty((len(indices),24))
    status = np.empty(len(indices),int)
    order = np.argsort(ix,kind='stable')
    groups = np.split(order,np.flatnonzero(np.diff(ix[order]))+1)
    for group in groups:
        r = core.evaluate_motion_camera(unique[ix[group[0]]],cs,shot.first,shot.dt,times[group])
        values[group] = r['values']
        status[group] = r['status']
    return values, status


def sensor_bound(values):
    # Bound the whole optical disk until the lower Jacobian eigenvalue fails.
    # Bounds below zero require a more specific check; they are not proof of a fold.
    fx,fy,cx,cy,k1,k2,p1,p2 = values[:,16:24].T
    radius = np.hypot(np.maximum(abs(cx),abs(1919-cx))/fx,
                      np.maximum(abs(cy),abs(1079-cy))/fy)
    tangent = np.hypot(p1,p2)
    r = np.linspace(0.,1.5,451)[None,:]
    k1,k2,tangent = (x[:,None] for x in (k1,k2,tangent))
    radial = 1.+k1*r**2+k2*r**4
    dr = 1.+3*k1*r**2+5*k2*r**4
    lower = np.minimum(radial,dr)-6*tangent*r
    valid = np.minimum.accumulate(lower > .025,axis=1)
    mapped = r*(radial-3*tangent*r)
    return np.max(np.where(valid,mapped,-np.inf),axis=1)-radius


def extrema(core, shot, cs, rig):
    span = [shot.times[0]-max(abs(rig[:,37]))/2,shot.times[-1]+max(abs(rig[:,37]))/2]
    z = core.motion_derivative_extrema(cs,shot.first,shot.dt,span)
    value = z['value'].copy()
    value[:2] *= 180/np.pi
    return dict(by_channel_pan_deg_tilt_deg_log_f=value.tolist(),
                source_seconds=(z['time']-shot.times[0]+shot.pts[0]/12800.).tolist())


def top_values(pts, data, i, n=5):
    scores=np.nan_to_num(data,nan=-np.inf)
    return [[round(float(pts[k])/12800.,3),round(float(data[k]),6)] for k in np.argsort(scores)[-n:][::-1]
        if np.isfinite(scores[k]) and i(k)]


def audit(core, shot, cs, rig):
    n=len(shot.pts)
    ii=np.arange(n)
    row=np.array([-.5,-.25,0,.25,.5])
    k=np.repeat(ii,len(row))
    times=shot.times[k]+rig[k,37]*np.tile(row,n)
    sample, status=at_exposures(core,shot,cs,rig,k,times)
    middle=sample.reshape(n,len(row),-1)[:,2,:]
    if np.any(status!=0):
        raise ValueError((shot.name,np.unique(status,return_counts=True)))
    margin=np.min(sensor_bound(sample).reshape(n,len(row)),axis=1)
    # Counterfactual change at an identical native time. Ordinary PTZ cancels.
    mid=(shot.times[:-1]+shot.times[1:])/2
    prior, sa=at_exposures(core,shot,cs,rig,ii[:-1],mid)
    later, sb=at_exposures(core,shot,cs,rig,ii[1:],mid)
    if np.any(sa|sb):
        raise ValueError('Counterfactual sample outside support')
    angle=Rotation.from_matrix((later[:,4:13].reshape(-1,3,3) @
                       prior[:,4:13].reshape(-1,3,3).transpose(0,2,1))).magnitude()
    look_jump=angle*(prior[:,16]+later[:,16])*.5
    pp_jump=np.linalg.norm(later[:,18:20]-prior[:,18:20],axis=1)
    pivot_jump=np.linalg.norm(np.diff(rig[:,15:18],axis=0),axis=1)
    pos_jump=np.linalg.norm(later[:,13:16]-prior[:,13:16],axis=1)
    rot=Rotation.from_matrix(middle[:,4:13].reshape(-1,3,3))
    rv=(rot[1:]*rot[:-1].inv()).as_rotvec()
    dv=np.diff(rv,axis=0)
    accel=np.linalg.norm(dv,axis=1)*middle[1:-1,16]
    zoom=np.abs(np.diff(middle[:,2],n=2))*500.
    def pack(vals,pts):
        return dict(p50=float(np.median(vals)),p95=float(np.quantile(vals,.95)),max=float(np.max(vals)),
            worst=top_values(pts,vals,lambda _:True))
    stats=dict(frames=n, camera_index=shot.camera_index,
        source_seconds=(shot.pts[[0,-1]]/12800.).tolist(), exact_derivatives=extrema(core,shot,cs,rig),
        sampled_exposure_rows=len(sample), principal_px_range=np.stack([sample[:,18:20].min(0),sample[:,18:20].max(0)]).tolist(),
        focal_px_range=[float(sample[:,16].min()),float(sample[:,16].max())],
        nonpositive_focal_rows=int(np.count_nonzero(sample[:,16:18]<=0)),
        centre_height_range=[float(sample[:,15].min()),float(sample[:,15].max())],
        max_abs_orthogonality_error=float(np.max(np.abs(rot.as_matrix()-middle[:,4:13].reshape(-1,3,3)))),
        readout_ms=np.unique(rig[:,37]*1000).tolist(),
        brown_sensor_margin_min=float(np.min(margin)), brown_unproven_frames=int((margin<=0).sum()),
        brown_worst=top_values(shot.pts,-margin,lambda _:True,3),
        varying_rig_rows=int(np.count_nonzero(np.any(np.diff(rig,axis=0)!=0,axis=1))),
        rig_look_jump_equiv_px=pack(look_jump,shot.pts[1:]),
        rig_principal_jump_px=pack(pp_jump,shot.pts[1:]),
        rig_pivot_jump_rink_units=pack(pivot_jump,shot.pts[1:]),
        rig_centre_jump_rink_units=pack(pos_jump,shot.pts[1:]),
        pan_tilt_second_difference_equiv_px=pack(accel,shot.pts[1:-1]),
        zoom_second_difference_at_500px=pack(zoom,shot.pts[1:-1]))
    return stats, dict(pts=shot.pts, snapshot=middle, sensor_margin=margin, rig_jump=look_jump,
                       principal_jump=pp_jump, pivot_jump=pivot_jump, angle_second_diff=accel, zoom_second_diff=zoom)


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,default=HERE.parent/'final/review.npz')
    p.add_argument('--out',default='camera_before')
    p.add_argument('--jin-only',action='store_true')
    a=p.parse_args()
    m,c,base,z=setup(a.candidate)
    reports={}
    arrays={}
    for name in m['shots']:
        if a.jin_only and name.startswith('okada'):continue
        shot=vj.shot_from_inputs(name,m,base)
        use=base if shot.program=='okada' else z
        s,x=audit(c,shot,use[name+'_controls'],use[name+'_rig'])
        reports[name]=s
        arrays.update({name+'_'+k:u for k,u in x.items()})
        print(name,'focal',np.round(s['focal_px_range']).tolist(),'Brown',round(s['brown_sensor_margin_min'],3),
            s['brown_unproven_frames'],'rig jump',round(s['rig_look_jump_equiv_px']['max'],2),
            'angular and zoom p95',round(s['pan_tilt_second_difference_equiv_px']['p95'],2),
            round(s['zoom_second_difference_at_500px']['p95'],2),flush=True)
    sj.write_json(HERE/(a.out+'.json'),reports)
    sj.atomic_arrays(HERE/(a.out+'.npz'),**arrays)


if __name__=='__main__':
    main()
