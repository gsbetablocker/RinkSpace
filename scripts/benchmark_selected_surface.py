"""Bounded FP64 selected-skin +/- replay benchmark; no fit or real model.

The fixture has dense pose blend shapes and weights on all 55 joints. Both
paths evaluate all 63 local pose columns. Timings include the same exact
kinematic pairs and, on the optimized path, one fresh surface preparation.
They exclude foot/collar residual geometry and sparse solve time. This is a
synthetic CPU operator measurement, not a target-machine/full-fit speedup.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time

import numpy as np
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits


def rotations(value):
    return Rotation.from_rotvec(value.reshape(-1,3)).as_matrix().reshape(*value.shape[:-1],3,3)


def run_case(samples,vertices,repeats,model,weight_kind):
    from calibration.program_tracking.body_tracking.compact_body_derivatives import CompactBodyLinearization
    from calibration.program_tracking.body_tracking.selected_surface_linearization import SelectedSurfaceLinearization
    from rinkspace.viewer3d.prepare import FullSurface
    rng=np.random.default_rng(6651104)
    surface=FullSurface.__new__(FullSurface);surface.body=model;surface._sample_double_cache=None
    surface.shaped=rng.normal(0.,.2,(vertices,3)).astype(np.float32)
    surface.posedirs=rng.normal(0.,.001,(486,vertices*3)).astype(np.float32)
    weights=rng.uniform(.01,1.,(vertices,55));weights/=weights.sum(axis=1,keepdims=True)
    if weight_kind=='localized':
        weights[:]=0.
        weights[:vertices//2,[1,3,7,15]]=.25
        weights[vertices//2:,[2,5,11,23]]=.25
    surface.weights=weights.astype(np.float32)
    reference=rotations(rng.normal(0.,.6,(samples,21,3)))
    delta=rng.normal(0.,.25,(samples,21,3))
    kinematics=CompactBodyLinearization(model,reference,delta,include_vertices=False)
    ids=np.arange(vertices)
    # Selection tensors are model data shared by both paths. A pose cache is
    # rebuilt for each optimized timing, just as for a new linearization.
    surface.sample_vertices(kinematics.native[:1],kinematics.frames[:1],ids,double_precision=True)
    context=SelectedSurfaceLinearization(surface,kinematics.native,kinematics.frames,ids)
    errors=[];difference_errors=[]
    for column in (6,24,68):
        pair=kinematics.kinematic_pair(column)
        got=context.pair(column,*pair)
        expected=tuple(surface.sample_vertices(s['native'],s['frames'],ids,double_precision=True) for s in pair)
        errors.extend(float(np.max(np.abs(x-y),initial=0.)) for x,y in zip(got,expected))
        difference_errors.append(float(np.max(np.abs((got[0]-got[1]-expected[0]+expected[1])/2e-5),initial=0.)))
    if max(errors)>1e-11 or max(difference_errors)>1e-7:raise RuntimeError('Selected-surface replay mismatch')
    memory=context.record();del context,got,expected,pair
    baseline=[];optimized=[];checksums={'baseline':[],'optimized':[]}
    def evaluate(fast):
        start=time.perf_counter();check=0.
        linearization=SelectedSurfaceLinearization(surface,kinematics.native,kinematics.frames,ids) if fast else None
        for column in range(6,69):
            pair=kinematics.kinematic_pair(column)
            values=linearization.pair(column,*pair) if fast else tuple(
                surface.sample_vertices(s['native'],s['frames'],ids,double_precision=True) for s in pair)
            check+=float(values[0].sum()+values[1].sum())
        return time.perf_counter()-start,check
    for repetition in range(repeats):
        for fast in ((False,True) if repetition%2==0 else (True,False)):
            elapsed,check=evaluate(fast)
            (optimized if fast else baseline).append(elapsed)
            checksums['optimized' if fast else 'baseline'].append(check)
    if not np.allclose(checksums['baseline'],checksums['optimized'],rtol=3e-13,atol=3e-9):
        raise RuntimeError('Selected-surface output checksum mismatch')
    old=float(np.median(baseline));new=float(np.median(optimized))
    result=dict(samples=samples,selected_vertices=vertices,weight_kind=weight_kind,local_columns=63,perturbed_skins=126,
                baseline_seconds=baseline,optimized_seconds=optimized,baseline_median_seconds=old,
                optimized_median_seconds=new,speedup=old/new,max_sample_error=max(errors),
                max_central_difference_error=max(difference_errors),memory_allocation_estimates=memory,
                includes='same exact FK pairs; optimized includes one new surface context; 126 complete selected skins',
                excludes='foot/collar residual evaluations, sparse assembly, linear solve, models/inference')
    print(json.dumps(result),flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,nargs='+',default=[128,951])
    parser.add_argument('--vertices',type=int,default=2048)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--weights',choices=('dense','localized'),default='dense',
                        help='Synthetic skinning support; both keep dense pose blend shapes and all 63 columns')
    args=parser.parse_args(argv)
    if min(args.samples+[args.vertices,args.repeats])<1:parser.error('All numeric values must be positive')
    if args.output.exists():parser.error('Output already exists; select a new evidence file')
    project=Path(__file__).resolve().parents[1];sys.path.insert(0,str(project/'src'));sys.path.insert(0,str(project));sys.path.insert(0,str(project/'tests'))
    from test_compact_body_performance import synthetic_model
    names=['calibration/program_tracking/body_tracking/selected_surface_linearization.py',
           'calibration/program_tracking/body_tracking/compact_body_derivatives.py','src/rinkspace/viewer3d/prepare.py']
    identities={name:hashlib.sha256((project/name).read_bytes()).hexdigest() for name in names}
    with tempfile.TemporaryDirectory() as directory,threadpool_limits(limits=1,user_api='blas'):
        model=synthetic_model(Path(directory)/'synthetic.npz')
        results=[run_case(n,args.vertices,args.repeats,model,args.weights) for n in args.samples]
    record=dict(schema='rinkspace.selected-surface-kernel-benchmark.v1',scope='Synthetic FP64 CPU operator; no target hardware or fit speed claim',
                python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,blas_threads=1,
                source_sha256=identities,results=results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
