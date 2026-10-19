"""Run the isolated complete v12 source solver with verified native kernels.

The production checkout, its checkpoints and its active processes are never
modified. All outputs belong to this task. Input q and profile arrays are read
from an explicitly selected checkpoint, or the frozen benchmark checkpoint.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--pattern-workers',type=int,default=4)
    parser.add_argument('--single-process',action='store_true')
    parser.add_argument('--fresh-start-checks',action='store_true',
        help='Repeat original initial finite differences; accepted checkpoints normally continue the existing solve loop')
    parser.add_argument('--benchmark-original',action='store_true',
        help='Measurement only: run all original v12 math and serial reviews on this isolated branch')
    parser.add_argument('--geometry-fd-step',type=float,default=1e-5,
        help='Explicit numerical geometry residual FD step in metres; original default is retained')
    parser.add_argument('--cost-fd-step',type=float,default=1e-5)
    parser.add_argument('--full-shared-fd',action='store_true',
        help='Fresh checks for every declared shared parameter, including all geometry and print patterns')
    parser.add_argument('--verification-iterations',type=int,
        help='Only for an isolated finite entry verification; default follows solver convergence')
    args = parser.parse_args()
    if args.threads < 1 or args.workers < 1 or args.pattern_workers<0 or (args.verification_iterations is not None and args.verification_iterations<1):
        parser.error('threads and workers must be positive')
    if not all(math.isfinite(s)and s>0 for s in(args.geometry_fd_step,args.cost_fd_step)):
        parser.error('Finite positive numerical FD steps required')
    if args.full_shared_fd and not args.fresh_start_checks:
        parser.error('Full shared FD requires explicit --fresh-start-checks')
    os.environ['RINKSPACE_MOTION_BACKEND'] = 'native'
    os.environ['RINKSPACE_MOTION_THREADS'] = str(args.threads)
    os.environ['RINKSPACE_SCATTER_BACKEND'] = 'native'
    os.environ['RINKSPACE_SCATTER_THREADS'] = str(args.threads)
    os.environ['RINKSPACE_PROJECTION_BACKEND'] = 'native'
    os.environ['RINKSPACE_PROJECTION_VALUE_BACKEND'] = 'opencv'
    os.environ['RINKSPACE_SOLVER_ACCELERATION'] = 'auto'
    os.environ['RINKSPACE_GEOMETRY_SPARSE_FD'] = '1'
    os.environ['RINKSPACE_GEOMETRY_FD_STEP_M'] = str(args.geometry_fd_step)
    os.environ['RINKSPACE_PROFILE_MOTION_ONLY'] = '1'
    os.environ['RINKSPACE_PATTERN_WORKERS'] = str(args.pattern_workers)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    if args.benchmark_original:
        for key in('RINKSPACE_MOTION_BACKEND','RINKSPACE_SCATTER_BACKEND','RINKSPACE_PROJECTION_BACKEND'):
            os.environ[key]='numpy'
        os.environ['RINKSPACE_GEOMETRY_SPARSE_FD']='0'
        os.environ['RINKSPACE_PROFILE_MOTION_ONLY']='0'
        os.environ['RINKSPACE_PATTERN_WORKERS']='0'
    from isolated_solver_bootstrap import install
    base = install()
    output = (args.output or base/('fast_solve_'+str(time.time_ns()))).resolve()
    if not output.is_relative_to(base.resolve()) or output == base.resolve():
        parser.error('output must be a new directory inside this isolated task folder')
    if output.exists():
        parser.error('output already exists; select a new directory')
    seed_source = (args.seed or base/'frozen_current_v12_seed.npz').resolve()
    if not seed_source.is_file():
        parser.error('checkpoint does not exist: '+str(seed_source))
    seed_bytes=seed_source.read_bytes()
    seed_sha=hashlib.sha256(seed_bytes).hexdigest()
    frozen_meta=json.loads((base/'frozen_seed_manifest.json').read_text(encoding='utf8'))
    seed=seed_source
    if args.seed is not None:
        seed=base/('selected_checkpoint_'+str(time.time_ns())+'.npz')
        seed.write_bytes(seed_bytes)
    from z190_motion_native import metadata as motion_metadata
    from z190_csr_scatter_native import metadata as scatter_metadata
    from z190_projection_guard_native import metadata as guard_metadata
    backend = dict(motion=motion_metadata(),scatter=scatter_metadata(),projection_guard=guard_metadata())
    for name,details in backend.items():
        if not details['available']:
            raise RuntimeError('Matching native '+name+' backend unavailable: '+str(details))
    import benchmark_harness as harness
    solve_args = harness.arguments(seed)
    solve_args.output = output
    solve_args.source_workers = args.workers
    solve_args.source_processes = not (args.single_process or args.benchmark_original)
    solve_args.source_actor_threads = args.threads
    solve_args.cache_native_source_replay = not args.benchmark_original
    solve_args.cache_complete_source_review = not args.benchmark_original
    solve_args.compact_source_review = not args.benchmark_original
    solve_args.source_stage_timing = True
    solve_args.initial_source_cost_fd_step = args.cost_fd_step
    print(json.dumps(dict(stage='isolated_fast_solver', output=str(output), seed=str(seed),
        source_workers=args.workers, source_processes=solve_args.source_processes,
        pattern_workers=args.pattern_workers,backend=backend, production_processes_modified=False),
        ensure_ascii=False), flush=True)
    import z190_joint_intercamera_fixed_xyz_source as solver
    import numpy as np
    with np.load(seed,allow_pickle=False) as selected:
        if str(selected['schema'])!='rinkspace.joint-source-LM-checkpoint.v1':
            raise ValueError('Unsupported original solver checkpoint schema')
        from calibration.solver_runtime.checkpoint_history import read_checkpoint_history
        history=read_checkpoint_history(selected,seed)
        if not history or history[-1].get('accepted') is not True:
            raise ValueError('Continuation requires an optimizer-accepted checkpoint')
        last=history[-1]
        if not all(np.isfinite(float(last[key]))for key in('cost','next_cost','damping')) or not float(last['next_cost'])<float(last['cost']):
            raise ValueError('Original accepted checkpoint cost history invalid')
        if selected['q'].ndim!=1 or selected['initial_q'].shape!=selected['q'].shape or not np.isfinite(selected['q']).all() or not np.isfinite(selected['initial_q']).all():
            raise ValueError('Finite complete original checkpoint q required')
        next_damping=float(selected['damping'])
        if not np.isfinite(next_damping)or next_damping<=0 or next_damping!=float(last['damping']):
            raise ValueError('Original accepted damping history changed')
        identity=json.loads(str(selected['identity_json']))
        authority=identity.get('input_manifest',{})
        manifest_path=Path(authority.get('path','')).resolve()
        if not manifest_path.is_file()or hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=authority.get('sha256'):
            raise ValueError('Original checkpoint input manifest identity changed')
        if seed_sha!=frozen_meta['sha256'] and manifest_path!=seed_source.parent.parent/'input_manifest.json':
            raise ValueError('Selected checkpoint must belong to its own solver/input manifest directory')
        source_fd=manifest_path.parent/'actual_global_source_cost_direction_checks.json'
        if not source_fd.is_file():
            prior=json.loads(manifest_path.read_text(encoding='utf8')).get('accepted_checkpoint_continuation',{})
            source_fd=Path(prior.get('original_initial_FD_evidence','')).resolve()
        source_checks=json.loads(source_fd.read_text(encoding='utf8'))
        required=set(solve_args.initial_source_fd_columns)
        passing={int(c['column'])for c in source_checks if float(c['relative_error'])<=.005}
        if not required<=passing or any(float(c['relative_error'])>.005 for c in source_checks):
            raise ValueError('Original fresh-start source FD evidence failed or incomplete')
        continuation=dict(seed_source=str(seed_source),seed_sha256=seed_sha,
            accepted_history_iteration=int(last['iteration']),accepted_source_cost=float(last['next_cost']),
            original_input_manifest=str(manifest_path),original_input_manifest_sha256=authority['sha256'],
            original_initial_FD_evidence=str(source_fd),original_initial_FD_evidence_sha256=hashlib.sha256(source_fd.read_bytes()).hexdigest(),
            fresh_start_FD_failure_diagnosis=str(base/'initial_fd_main7_row_summary.json'),
            optimizer_acceptance_is_not_source_alignment_accuracy=True,
            continuation_of_original_accepted_loop=True,source_objective_and_J_unchanged=True)
    if not args.fresh_start_checks:
        solve_args.accepted_checkpoint_continuation=continuation
    if args.full_shared_fd:solve_args.initial_source_fd_columns=None
    original_solve=solver.solve_joint_source
    def solve_with_saved_damping(*positional,**keywords):
        keywords.setdefault('initial_damping',next_damping)
        if args.verification_iterations is not None:
            original_review=keywords.get('checkpoint_callback')
            accepted=0
            def finite_verification(q,event):
                nonlocal accepted
                review=original_review(q,event)if original_review is not None else None
                if event['checkpoint_kind']=='accepted':accepted+=1
                if accepted>=args.verification_iterations:
                    review=dict(review or {},stop_branch=True,
                        reason='Explicit isolated entry verification completed; production unchanged')
                return review
            keywords['checkpoint_callback']=finite_verification
        began=time.perf_counter()
        result=original_solve(*positional,**keywords)
        timing=dict(wall_seconds=time.perf_counter()-began,original_serial_baseline=args.benchmark_original,
            complete_initial_and_accepted_source_reviews_included=True,source_rows=len(positional[2]),
            parameters=len(positional[1]),accepted_steps=sum(bool(x['accepted'])for x in result.history),
            seed_sha256=seed_sha)
        (output/'actual_complete_solver_wall.json').write_text(json.dumps(timing,ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps(dict(stage='actual_complete_solver_wall',**timing)),flush=True)
        return result
    solver.solve_joint_source=solve_with_saved_damping
    cache=base/'prepared_full8503_state.pickle'
    if cache.is_file():
        # Retain the exact admitted graph. Validate a selected warm state
        # against every saved source mask/scale and binding-array identity.
        cap=harness.capture(solve_args,{})
        state=cap['state']
        captured_actors=cap.get('source_actors')
        if captured_actors is not None:captured_actors.close()
        state.source_actors=None
        del cap
        import z190_joint_source_solver as source_solver
        frozen=base/'frozen_current_v12_seed.npz'
        with np.load(frozen,allow_pickle=False) as reference,np.load(seed,allow_pickle=False) as saved:
            for key in reference.files:
                if key.startswith('aux__') or key in ('quadratic','scales'):
                    if key not in saved or reference[key].dtype!=saved[key].dtype or not np.array_equal(reference[key],saved[key]):
                        raise ValueError('Prepared graph source/binding contract changed: '+key)
            profile_keys={k for k in reference.files if k.startswith('profile__')}
            if profile_keys!={k for k in saved.files if k.startswith('profile__')}:
                raise ValueError('Accepted checkpoint profile inventory changed')
            for i,block in enumerate(state.blocks):
                for field in('present','robust'):
                    value=saved['profile__'+field+'_'+str(i)]
                    if value.shape!=()or value.dtype!=np.dtype(bool):
                        raise ValueError('Original profile flags must be scalar boolean')
            if saved['q'].shape!=state.q.shape:
                raise ValueError('Prepared graph parameter inventory changed')
            state.q=saved['q'].copy()
            state.seed_contract=(saved['quadratic'].copy(),saved['scales'].copy())
            source_solver.make_profile_update(state.blocks).restore_checkpoint(
                {key[9:]:saved[key].copy() for key in saved.files if key.startswith('profile__')})
            for block in state.blocks:
                weights=block.bundle.profile_weights
                if weights is not None and (weights.shape!=(block.bundle.motion_rows//2,2)or not np.isfinite(weights).all()or np.any(weights<=0)or np.any(weights>1)):
                    raise ValueError('Complete finite positive accepted profile weights required: '+block.name)
        state.output=output
        original_prepare=solver.prepare
        def prepared(unused):
            output.mkdir(parents=True,exist_ok=False)
            return state
        solver.prepare=prepared
        try:
            solver.run(solve_args)
        finally:
            actors=getattr(state,'source_actors',None)
            if actors is not None:actors.close()
            from z190_parallel_pattern_evaluation import close_pattern_pools
            close_pattern_pools()
            solver.prepare=original_prepare
        return
    solver.run(solve_args)


if __name__ == '__main__':
    main()
