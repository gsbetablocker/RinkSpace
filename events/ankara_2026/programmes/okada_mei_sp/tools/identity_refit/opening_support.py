"""Saved programme double support, evaluated on the actual articulated skate in metric units."""
import numpy as np
from scipy.sparse import diags, eye, hstack, vstack
from scipy.sparse.linalg import lsqr

from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference, evaluate_curve
from calibration.program_tracking.body_tracking.absolute_motion_curve import AbsoluteMotionCurve, _exp
from calibration.program_tracking.body_tracking.skate_physics import fk_state, batch_features
from calibration.program_tracking.body_tracking.body_clock import time_base


def samples(ref, policy):
    pts = np.asarray(policy['pts'], dtype=np.int64)
    numerator, denominator = time_base(ref.p)
    if list(policy['time_base']) != [numerator, denominator]:
        raise ValueError('Opening support clock differs from the programme')
    lo, hi = (pts - ref.p['pts'][0]) * numerator / denominator
    return np.linspace(lo, hi, round((hi-lo)*policy['grid_hz'])+1)


def height(ref, curve, times, delta=None):
    model = absolute_reference(ref, curve)
    pose = fk_state(model, model.at(times), np.zeros((len(times), 69)) if delta is None else delta)
    features = batch_features(model.binding, pose['R'], pose['O'], pose)
    blade = np.isin(features['semantics'], ['runner', 'toe_pick'])
    z = features['height']
    return z[..., blade].min(-1), z.min(-1)


def verify(ref, curve, policy):
    times = samples(ref, policy)
    blade, solid = height(ref, curve, times)
    worst = np.max(np.abs(blade), axis=0)
    lo, hi = policy['pts']
    exposure_times = ref.times[(ref.p['pts'] >= lo) & (ref.p['pts'] <= hi)]
    indices = np.round((exposure_times-times[0])*policy['grid_hz']).astype(int)
    if np.allclose(times[indices], exposure_times, atol=1e-9, rtol=0):
        exposed = blade[indices]
    else:
        exposed, _ = height(ref, curve, exposure_times)
    result = dict(pts=policy['pts'], samples=len(times), max_abs_blade_m=worst.tolist(),
                  min_any_part_m=solid.min(0).tolist(), max_exposure_abs_m=np.abs(exposed).max(0).tolist())
    tolerance = policy['tolerance_m']
    if (not np.isfinite(blade).all() or not np.isfinite(solid).all() or not np.isfinite(exposed).all()
            or max(worst)>tolerance or np.abs(exposed).max()>tolerance or solid.min() < -tolerance):
        raise ValueError(('Opening double support violated',result))
    return result


def enforce(ref, curve, policy, emit):
    times = samples(ref, policy)
    B = curve.basis_sparse(times)
    controls = np.unique(B.indices)
    B = B[:, controls]
    channels = [(joint,axis) for joint in (1,2,4,5,7,8) for axis in range(3)]
    for iteration in range(12):
        blade, solid = height(ref, curve, times)
        err = np.abs(blade).max()
        emit('opening_contact', iteration=iteration, max_abs_left_right_m=np.abs(blade).max(0).tolist(),
             min_solid_m=solid.min(0).tolist())
        if err < policy['tolerance_m']*.8 and solid.min() >= -policy['tolerance_m']:
            return curve
        jac = [vstack((B,B))*ref.binding.ratio]
        for joint,axis in channels:
            delta = np.zeros((len(times),69)); delta[:,3+3*joint+axis] = 1e-5
            plus,_ = height(ref,curve,times,delta)
            slope = (plus-blade)/1e-5
            jac.append(vstack([diags(slope[:,f])@B for f in range(2)]))
        design = hstack(jac).tocsr()
        n=len(controls)
        scales=np.r_[np.ones(n),np.full(n*len(channels),.18)]
        smooth=diags([1.,-2.,1.],[0,1,2],shape=(n-2,n))
        from scipy.sparse import block_diag
        sys=vstack([design,diags(scales*.006),block_diag([smooth]*len(jac))@diags(scales*.2)])
        step=lsqr(sys,np.r_[-blade.T.ravel(),np.zeros(sys.shape[0]-design.shape[0])],atol=1e-10,btol=1e-10)[0].reshape(-1,n)
        pos=curve.position_controls.copy();pos[controls,2]+=step[0]
        rot=curve.rotation_controls.copy()
        for k,j in enumerate((1,2,4,5,7,8)):
            rot[controls,j]=rot[controls,j]@_exp(step[1+k*3:4+k*3].T)
        curve=AbsoluteMotionCurve(curve.knots,pos,rot,curve.edge_reference_quaternion)
    verify(ref,curve,policy)
    return curve
