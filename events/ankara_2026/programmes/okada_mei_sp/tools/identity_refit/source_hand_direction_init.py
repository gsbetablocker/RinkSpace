"""Initialize continuous wrist orientation from visible native hand chords."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

from compare import bind
from source_review import digest
from calibration.program_tracking.body_tracking.absolute_motion_curve import AbsoluteMotionCurve, _log
from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference, curve_from_packet
from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState
from calibration.program_tracking.body_tracking.source_motion_refit import NativeSurfacePoints


def lift(curve):
    v=curve._check_log_chart();a=np.linalg.norm(v,axis=-1)
    scale=np.divide(np.sin(a/2),a,out=np.full_like(a,.5),where=a>1e-12)
    return np.concatenate((v*scale[...,None],np.cos(a/2)[...,None]),axis=-1)


def run(args):
    from types import SimpleNamespace
    r,c,_,_,packet=bind(args.prepared,args.source,args.parameter)
    parent=curve_from_packet(packet);baseline=curve_from_packet(dict(np.load(args.reference)))
    previous=curve_from_packet(dict(np.load(args.opening_reference)))
    positions=parent.position_controls.copy();rotations=parent.rotation_controls.copy()
    points=json.loads(args.points.read_text(encoding='utf-8'))
    stamps=np.array([v['pts'] for v in points['surface_samples']],dtype=np.int64)
    hand_times=r.times[np.searchsorted(r.p['pts'],stamps)]
    opening=np.unique(parent.basis_sparse(np.linspace(hand_times.min(),hand_times.max()+.12,25)).indices)
    # Discard the rejected absolute-hand trial's common arm displacement.
    positions[opening]=previous.position_controls[opening]
    rotations[opening]=previous.rotation_controls[opening]
    provisional=AbsoluteMotionCurve(parent.knots,positions,rotations,parent.edge_reference_quaternion)
    q=lift(provisional);old_q=lift(baseline)
    changed=np.sum(q*old_q,axis=-1)<0
    # Transport the old curve's SU(2) lift to the updated control matrices.
    # This is an explicit proposal; it does not silently rewrite a checkpoint.
    q=np.where(changed[...,None],-q,q)
    curve=AbsoluteMotionCurve(parent.knots,positions,rotations,q)
    ref=absolute_reference(r,curve);state=SequenceState(ref,packet['seconds'])
    problem=SimpleNamespace(reference=ref,camera=c,binding=ref.binding,
        body=SimpleNamespace(state=state),size=17+state.size)
    factor=NativeSurfacePoints(problem,points)
    controls=np.unique(state.basis(factor.times).indices)
    columns=(controls[:,None]*69+np.arange(63,66)).ravel()
    zero=np.zeros(state.size);initial=factor.evaluate(zero)
    # Existing local wrist curve, with finite acceleration and pose terms.
    times=r.times[(r.times>=hand_times.min())&(r.times<=hand_times.max()+.6)]
    base_R=ref.at(times)['local_R'][:,19]
    h=np.diff(times)
    last=[];progress_clock=[time.perf_counter()];log=None
    def residual(value):
        # The visible opening holds one palm gesture. Fit its common local
        # turn, retaining the existing continuous changes between controls.
        full=zero.copy();full[columns]=np.tile(value,len(controls))
        current=state.current_curve(full)
        current.check_continuous_edge_update(curve)
        pixels=factor.evaluate(full)['residual']
        from calibration.program_tracking.body_tracking.absolute_motion_state import evaluate_curve
        R=evaluate_curve(current,times)['local_R'][:,19]
        velocity=_log(R[1:]@R[:-1].swapaxes(-1,-2))/h[:,None]
        alpha=2*np.diff(velocity,axis=0)/(h[1:]+h[:-1])[:,None]
        acceleration=alpha*np.sqrt((h[1:]+h[:-1])*.5*.16)[:,None]/40.
        pose=_log(base_R.swapaxes(-1,-2)@R)*.03
        result=np.r_[pixels,acceleration.ravel(),pose.ravel()]
        last[:]=[float(result@result/2)]
        if log is not None and time.perf_counter()-progress_clock[0]>3.:
            log.write(json.dumps(dict(event='hand_direction_evaluation',cost=last[0],
                source_chord_norm=float(np.linalg.norm(pixels))))+'\n')
            progress_clock[0]=time.perf_counter()
        return result
    with args.log.open('a',encoding='utf-8',buffering=1) as log:
        log.write(json.dumps(dict(event='source_hand_direction_initialize',
            parent=str(args.parameter),parent_sha256=digest(args.parameter),
            active_variables=3,continuous_wrist_control_columns=len(columns),source_pts=stamps.tolist(),
            transported_edge_lifts=np.argwhere(changed).tolist(),
            opening_controls_restored=opening.tolist()))+'\n')
        solution=least_squares(residual,np.zeros(3),x_scale='jac',
            ftol=1e-5,xtol=1e-6,gtol=1e-5)
        full=zero.copy();full[columns]=np.tile(solution.x,len(controls));final=state.current_curve(full)
        values=factor.evaluate(full)
        result=dict(packet)
        result.update(x=np.r_[packet['x'][:17],np.zeros(state.size)],
            absolute_position_controls=final.position_controls,
            absolute_rotation_controls=final.rotation_controls,
            absolute_seed_position_controls=final.position_controls,
            absolute_seed_rotation_controls=final.rotation_controls,
            absolute_rotation_edge_reference_quaternion=lift(final),
            relative_skate_seed_rotation_controls=packet['relative_skate_rotation_controls'])
        receipt=dict(event='source_hand_direction_initialized',parameter=str(args.output),
            model='same body and finger geometry; only continuous left-wrist controls fitted',
            before=initial['metrics'],after=values['metrics'],cost=float(solution.cost),
            optimality=float(solution.optimality),evaluations=int(solution.nfev),
            transported_edge_lifts=np.argwhere(changed).tolist(),user_accepted=False)
        result['source_hand_initialization_json']=np.array(json.dumps(receipt,sort_keys=True))
        with args.output.open('xb') as out:np.savez_compressed(out,**result)
        receipt['sha256']=digest(args.output);log.write(json.dumps(receipt)+'\n')
        print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('prepared','source','parameter','reference','opening-reference','points','output','log'):
        parser.add_argument('--'+key,type=Path,required=True)
    run(parser.parse_args())
