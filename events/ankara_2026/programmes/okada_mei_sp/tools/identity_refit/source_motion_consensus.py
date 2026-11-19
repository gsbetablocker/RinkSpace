"""Remove isolated joint-rotation reversals while retaining the source-fitted motion."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.absolute_motion_curve import (
    AbsoluteMotionCurve, greville_times, _exp, _log)
from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet, evaluate_curve
from compare import bind
from opening_support import samples
from source_joint_refit import save, POLICY
from source_review import digest


def stabilize(curve, preceding, early, ref):
    g = greville_times(curve.knots)
    early_at = evaluate_curve(early, g)['rotations']
    rotations = curve.rotation_controls.copy()
    protected = np.unique(curve.basis_sparse(samples(ref, POLICY)).indices)
    grid = np.arange(g[0], g[-1]+.005, .005)
    def speed(r):
        a = np.rad2deg(np.linalg.norm(_log(r[:-1].swapaxes(-1,-2)@r[1:]), axis=-1))
        return np.vstack([a[0],a])
    actual = speed(evaluate_curve(curve,ref.times)['local_R'])
    prior = speed(evaluate_curve(early,ref.times)['local_R'])
    previous_speed = speed(evaluate_curve(preceding,ref.times)['local_R'])
    def localized(excess):
        v=np.clip(np.interp(grid,ref.times,excess)/6.,0.,1.)
        return np.interp(g,grid,gaussian_filter1d(maximum_filter1d(v,size=39),sigma=5.))
    # The preceding whole-body pass already fits the shoe. Blend its entire
    # kinematic chain together when the later knee fit introduced a sharp
    # excursion. Filtering hip, knee and ankle independently displaces the shoe.
    for chain in ([1,4,7],[2,5,8]):
        ix=np.asarray(chain)-1
        excessive=actual[:,ix]-np.maximum(12.,np.minimum(16.,previous_speed[:,ix]+5.))
        alpha=localized(excessive.max(1))
        alpha[protected]=0.
        for j in chain:
            direction=_log(rotations[:,j].swapaxes(-1,-2)@preceding.rotation_controls[:,j])
            rotations[:,j]=rotations[:,j]@_exp(alpha[:,None]*direction)
    for j in range(1,22):
        # Fit the offset from the early motion, so a genuine rapid turn in
        # that sequence survives. Only atypically fast local joints are edited.
        excess = previous_speed[:,j-1] - np.maximum(16.,np.minimum(23.,prior[:,j-1]+6.))
        alpha = localized(excess)
        offset = early_at[:,j].swapaxes(-1,-2)@rotations[:,j]
        q = Rotation.from_matrix(offset).as_quat()
        # Consistent signs across the actual, nonuniform control times.
        signs = np.where((q[1:]*q[:-1]).sum(1)<0,-1.,1.)
        q[1:] *= np.cumprod(signs)[:,None]
        uniform = np.column_stack([np.interp(grid,g,v) for v in q.T])
        mean = gaussian_filter1d(uniform,sigma=.063/.005,axis=0,mode='nearest')
        qbar = np.column_stack([np.interp(g,grid,v) for v in mean.T])
        if np.min(np.linalg.norm(qbar,axis=1))<.1:
            raise ValueError('Ambiguous temporal average for local joint '+str(j))
        proposal = early_at[:,j]@Rotation.from_quat(qbar).as_matrix()
        if j in (1,2,4,5,7,8):
            alpha[protected] = 0.
        idx = np.flatnonzero(alpha>1e-8)
        if not idx.size:
            continue
        correction = _log(rotations[idx,j].swapaxes(-1,-2)@proposal[idx])
        rotations[idx,j] = rotations[idx,j]@_exp(alpha[idx,None]*correction)
    result = AbsoluteMotionCurve(curve.knots,curve.position_controls,rotations,curve.edge_reference_quaternion)
    result.check_continuous_edge_update(curve)
    return result


def material_anchors(curve, donor, ref, path):
    # Right boot portions separately visible on the broadcast. At these source
    # exposures the articulated donor has the closer measured heel/blade arcs.
    seconds=np.array([163.62,163.78,164.92,165.00,172.12])
    source=json.loads(path.read_text(encoding='utf8'))
    stamps=np.round(seconds*12800).astype(int)
    annotated={int(r['source_pts']) for r in source['frames'] if r['arcs']}
    if not set(stamps)<=annotated:
        raise ValueError('Missing reviewed source arcs')
    t=(stamps-ref.p['pts'][0])/12800.
    group=[2,5,8]
    now=evaluate_curve(curve,t)['rotations'][:,group]
    goal=evaluate_curve(donor,t)['rotations'][:,group]
    target=_log(now.swapaxes(-1,-2)@goal)
    g=greville_times(curve.knots)
    # Fit a smooth correction of the entire right leg on the actual spline
    # basis. A single-exposure joint twist would otherwise satisfy these pixels.
    width=.075
    kernels=np.exp(-.5*((g[:,None]-t)/width)**2)
    kernels[np.abs(g[:,None]-t)>5*width]=0.
    fields=curve.basis_sparse(t)@kernels
    weights=np.linalg.solve(fields+.018*np.eye(len(t)),target.reshape(len(t),-1))
    correction=(kernels@weights).reshape(len(g),len(group),3)
    rot=curve.rotation_controls.copy()
    active=np.flatnonzero(kernels.any(1))
    rot[np.ix_(active,group)]=rot[np.ix_(active,group)]@_exp(correction[active])
    result=AbsoluteMotionCurve(curve.knots,curve.position_controls,rot,curve.edge_reference_quaternion)
    result.check_continuous_edge_update(curve)
    return result,stamps.tolist()


def run(prepared, parent, previous_checkpoint, early_checkpoint, arc_source, output):
    output.mkdir(exist_ok=True)
    shutil.copyfile(parent/'source.json',output/'source.json')
    ref,_,_,_,pkt=bind(prepared,output/'source.json',parent/'candidate.npz')
    if json.loads(str(pkt['programme_double_support_json'])) != POLICY:
        raise ValueError('Source and parent opening support differ')
    with np.load(early_checkpoint) as z:
        early=curve_from_packet(z)
    with np.load(previous_checkpoint) as z:
        preceding=curve_from_packet(z)
    np.testing.assert_array_equal(ref.absolute_curve.knots,preceding.knots)
    curve=stabilize(ref.absolute_curve,preceding,early,ref)
    curve,arc_pts=material_anchors(curve,ref.absolute_curve,ref,arc_source)
    save(output,'candidate.npz',pkt,curve,ref,dict(stage='source_temporal_review',
         parent=str(parent/'candidate.npz'),parent_sha256=digest(parent/'candidate.npz'),
         early=str(early_checkpoint),early_sha256=digest(early_checkpoint),
         preceding=str(previous_checkpoint),preceding_sha256=digest(previous_checkpoint),
         right_material_arc_pts=arc_pts,material_source_sha256=digest(arc_source),
         full_joint_fit=json.loads(str(pkt['joint_source_fit_json'])),
         scope='Adaptive local rotation offsets on the native source clock. Root and opening support chain are frozen.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for arg in ('prepared','parent','previous_checkpoint','early_checkpoint','arc_source','output'):
        parser.add_argument(arg,type=lambda v:Path(v).resolve())
    run(**vars(parser.parse_args()))
