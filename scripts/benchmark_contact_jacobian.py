"""Bounded same-input comparison of the contact Jacobian CPU kernel.

No fit, model download or output candidate is produced. Both versions see
the same nonzero pose, selected witnesses, FP64 data and sparse spline basis.
The default physical sample and active-column counts match the 3d6f51f
support/ExMR logs; geometry in those two cases is synthetic. --saved-state
additionally uses an existing exported
midpoint state and does not claim to reconstruct its full exposure grid.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import platform
import sys
import time

import numpy as np
from scipy.interpolate import BSpline
from threadpoolctl import threadpool_limits


def timed(function,args):
    start=time.perf_counter();value=function(*args)
    return value,time.perf_counter()-start


def run_case(name,args,old,new,repeats):
    expected=old(*args);actual=new(*args)
    difference=actual-expected
    error=float(np.max(np.abs(difference.data),initial=0.))
    relative=float(np.linalg.norm(difference.data)/max(np.linalg.norm(expected.data),1e-30))
    if error>2e-9 or relative>2e-11:raise RuntimeError('Contact Jacobian mismatch: '+name)
    old_seconds=[];new_seconds=[]
    for repetition in range(repeats):
        order=((old,old_seconds),(new,new_seconds)) if repetition%2==0 else ((new,new_seconds),(old,old_seconds))
        for function,values in order:
            _,seconds=timed(function,args);values.append(seconds)
    om=float(np.median(old_seconds));nm=float(np.median(new_seconds))
    evaluation=args[-1];rows=len(evaluation['rows']);columns=len(args[-3])
    result=dict(name=name,physics_samples=len(args[2]),constraint_rows=rows,active_local_columns=columns,
                body_control_columns=actual.shape[1],nnz=actual.nnz,max_abs_difference=error,
                relative_frobenius_difference=relative,baseline_seconds=old_seconds,optimized_seconds=new_seconds,
                baseline_median_seconds=om,optimized_median_seconds=nm,kernel_speedup=om/nm,
                baseline_default_list_reference_slots=rows*rows*columns)
    print(json.dumps(result),flush=True)
    return result


def saved_state_case(path):
    with np.load(path,allow_pickle=False) as packet:
        n=len(packet['pts']);Q=packet['Q_world_mid'].copy();local=packet['local_R_mid'].copy()
        ratio=float(json.loads(str(packet['camera_binding_json']))['length_export_ratio'])
        R=packet['skate_R_world_mid'].copy();O=packet['skate_origin_world_mid']/ratio
        witness=packet['skate_witness_world_mid']/ratio
        state=dict(P=packet['P_world_mid']/ratio,Q=Q,native_world=packet['native22_world_mid']/ratio,R=R,O=O)
        times=(packet['pts']-packet['pts'][0])/12800.
    parents=np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19])
    frames=np.broadcast_to(np.eye(3),(n,22,3,3)).copy()
    for joint in range(1,22):frames[:,joint]=frames[:,parents[joint]]@local[:,joint-1]
    state['frames']=frames
    q=np.einsum('nfji,nfj->nfi',R,witness-O)
    rows=[(frame,foot,q[frame,foot].copy(),1.) for frame in range(n) for foot in (0,1)]
    delta=np.zeros((n,69));active=np.arange(69)
    knots=np.r_[[times[0]]*4,np.arange(times[0]+.4,times[-1]-1e-9,.4),[times[-1]]*4]
    B=BSpline(knots,np.eye(len(knots)-4),3)(times)
    reference=SimpleNamespace(binding=SimpleNamespace(ratio=ratio))
    evaluation=dict(rows=rows,row_upper_vertex=[-1]*len(rows),times=times)
    return reference,state,delta,active,B,evaluation


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-project',type=Path,required=True)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--saved-state',type=Path)
    args=parser.parse_args(argv)
    if args.repeats<1:parser.error('--repeats must be positive')
    if args.output.exists():parser.error('Output already exists; choose a new evidence file')
    sys.path.insert(0,str(args.project/'tests'))
    from test_skate_physics_performance import load_kernel,make_fixture
    old=load_kernel(args.baseline_project).body_constraint_jacobian
    new=load_kernel(args.project).body_constraint_jacobian
    rows=[]
    with threadpool_limits(limits=1,user_api='blas'):
        for name,n,active,upper,controls in [
            ('synthetic_support_log_sample_and_column_counts',1505,np.r_[np.arange(6),9,10,11,18,19,20,27,28,29],False,23),
            ('synthetic_exmr_log_sample_and_column_counts',5705,np.arange(69),False,51),
            ('synthetic_deforming_upper_nonzero_pose',128,np.arange(69),True,20)]:
            _,reference,_,state,delta,active,B,evaluation=make_fixture(n,active=active,upper=upper,controls=controls)
            rows.append(run_case(name,(reference,state,delta,active,B,evaluation),old,new,args.repeats))
        if args.saved_state:
            rows.append(run_case('saved_source_midpoint_rigid_witnesses',saved_state_case(args.saved_state),old,new,args.repeats))
    relative='calibration/program_tracking/body_tracking/skate_physics.py'
    record=dict(schema='rinkspace.contact-jacobian-kernel-benchmark.v1',scope='CPU operator only; no full fit and no target-machine claim',
                python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,blas_threads=1,
                baseline_sha256=hashlib.sha256((args.baseline_project/relative).read_bytes()).hexdigest(),
                optimized_sha256=hashlib.sha256((args.project/relative).read_bytes()).hexdigest(),
                saved_state_sha256=None if args.saved_state is None else hashlib.sha256(args.saved_state.read_bytes()).hexdigest(),
                results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
