"""Isolated real v12 complete-source timing and equality harness.

Everything written by this harness lives beside it. The v12 checkpoint is copied
once before preparation. Data source files and production processes are read only.
The driver's run closure is recovered before its expensive reporting/FD section;
the original source factory, physical guard and solver are retained unchanged.
"""
from __future__ import annotations
from calibration.solver_runtime.config import data_root as _runtime_data_root, data_module_dir as _runtime_data_module_dir

import argparse
import cProfile
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import pstats
import shutil
import sys
import time
from types import SimpleNamespace

BASE=_runtime_data_root()
from calibration.solver_runtime.config import PROGRAM_SOURCE_ROOT
SNAPSHOT = PROGRAM_SOURCE_ROOT
SOURCE_ROOT = Path('D:/OneDrive/codex_projs/RinkSpace')
N = Path('D:/CodexData/.codex/visualizations/2026/10/02/private-thread-00008/RinkSpace_tmp')
CURRENT = N / 'joint_fullprogramme_shared_lanes_Ad1_14_qualified_outer_viewLM_streaming_v12/solver/latest_accepted_state.npz'
from isolated_solver_bootstrap import install
install()
from prepared_cache_helpers import install_mappingproxy_pickle,install_geometry_unpickle
install_mappingproxy_pickle()
install_geometry_unpickle()
sys.path[:0] = [str(SNAPSHOT), str(SOURCE_ROOT), str(SOURCE_ROOT / 'src')]
for candidate in (SOURCE_ROOT / '.venv/Lib/site-packages', SOURCE_ROOT.parent / 'MotionTrail_Codex_Starter_v1/.venv-gpu/Lib/site-packages'):
    if candidate.is_dir(): sys.path.append(str(candidate))

import numpy as np


def write(path, payload):
    diagnostic_dir = os.environ.get('RINKSPACE_BENCHMARK_DIAGNOSTIC_DIR')
    if diagnostic_dir and path.parent.resolve() == BASE.resolve():
        path = Path(diagnostic_dir) / path.name
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=lambda x: str(x) if isinstance(x, Path) else x.tolist() if isinstance(x, np.ndarray) else x.item() if isinstance(x, np.generic) else str(x)), encoding='utf8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measured(label, operation, report, profiling=False):
    print(json.dumps(dict(stage=label, status='begin')), flush=True)
    wall, cpu = time.perf_counter(), time.process_time()
    profiler = cProfile.Profile() if profiling else None
    if profiler: profiler.enable()
    result = operation()
    if profiler:
        profiler.disable()
        profiler.dump_stats(str(BASE / f'{label}.prof'))
        with (BASE / f'{label}_profile.txt').open('w', encoding='utf8') as out:
            pstats.Stats(profiler, stream=out).sort_stats('cumulative').print_stats(80)
    timing = dict(wall_seconds=time.perf_counter()-wall, process_cpu_seconds=time.process_time()-cpu, profiled=profiling)
    report[label] = timing
    print(json.dumps(dict(stage=label, status='end', **timing)), flush=True)
    timing_path = Path(os.environ.get('RINKSPACE_BENCHMARK_TIMINGS_PATH', str(BASE / 'benchmark_timings_latest.json')))
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    write(timing_path, report)
    return result


def freeze_seed():
    target = BASE / 'frozen_current_v12_seed.npz'
    if not target.exists():
        for attempt in range(5):
            before = CURRENT.stat()
            shutil.copyfile(CURRENT, target)
            after = CURRENT.stat()
            if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns): break
        else: raise RuntimeError('Live checkpoint changed during all copy attempts')
        with np.load(target, allow_pickle=False) as seed:
            from calibration.solver_runtime.checkpoint_history import read_checkpoint_history
            meta = dict(source=str(CURRENT), copied_size=target.stat().st_size, copied_source_mtime_ns=after.st_mtime_ns,
                sha256=sha(target), q_size=len(seed['q']), source_rows=len(seed['scales']), damping=float(seed['damping']),
                source_history_iterations=len(read_checkpoint_history(seed,target)),
                canonical_exposures=len(seed['aux__canonical_pts']), profile_keys=[k for k in seed.files if k.startswith('profile__')])
        write(BASE / 'frozen_seed_manifest.json', meta)
    return target


def arguments(seed):
    import z190_joint_intercamera_fixed_xyz_source as c
    import z190_secondary_qualified_core_source
    z190_secondary_qualified_core_source.TEMP_ROOT=BASE
    output=BASE/'prepared_full8503'
    if output.exists() and not (BASE/'prepared_full8503_state.pickle').exists():
        output=BASE/('prepared_full8503_'+str(time.time_ns()))
    return SimpleNamespace(output=output, warm=c.WARM, low_temporal_source=c.DENSE,
        source_packets=[N/'ice_alignment_review/long_side_real_camera_source_v1/low5_to_main_B_whole_pan_tilt_head_support_QUALIFIED_FROZEN_v2.json', N/'source_solver_audit/low_camera_mutual_native_source_v1/secondary6_to_low_whole_pan_tilt_head_support_QUALIFIED_FROZEN_v3.json'],
        main4_source_admission_cache=N/'joint_fullprogramme_intercamera_fixed_XYZ_dense_v1/fresh_main4_original_source/original_motion_source_admission.npz',
        physical_volume_loss='quadratic', linear_strategy='global-schur', schur_backtracking=True,
        schur_view_lm_ratio=1., schur_backtracking_aware_damping=True, pivot_border_policy='outside-soft', pivot_board_soft_sigma_m=.30,
        centrosymmetric_ads=True, ad_corner_source=N/'low5_repair/secondary6_Ad14_native_source_v1/secondary6_Ad14_native_corners_QUALIFIED_FROZEN.json',
        external_main_initial=N/'ice_alignment_review/external_main_low_conditional_initialization_v1/external_actual38_conditional_q_UNACCEPTED.npz',
        joint_state_seed=seed,
        secondary_original_ring_source_qualification=N/'ice_alignment_review/secondary_old_R2_R3_source_provenance_v1/SOURCE_QUALIFIED_candidate_v3/R2_R3_original31_outer_source_QUALIFIED_CANDIDATE_FROZEN.json',
        shared_board_patterns=True, shared_lane_geometry=True, shared_ad1_14_geometry=True,
        initial_source_fd_columns=[1,2,3,4,8,9,10],
        initial_source_fd_evidence=N/'joint_fullprogramme_shared_lanes_Ad1_14_patterns_viewLM_streaming_v11/actual_global_source_cost_direction_checks.json',
        streaming_source=True, solve=True, source_stage_timing=True)


def objects(root):
    pending=[root]; seen=set()
    while pending:
        obj=pending.pop()
        if id(obj) in seen: continue
        seen.add(id(obj))
        if isinstance(obj,dict): pending.extend(obj.values())
        elif isinstance(obj,(list,tuple)): pending.extend(obj)
        elif hasattr(obj,'__dict__') and not inspect.ismodule(obj) and not callable(obj):
            yield obj
            pending.extend(vars(obj).values())


def restore_local_priors(state):
    from z190_moving_pivot_source_bundle import MovingPivotSourceBundle
    blocks=[*state.blocks,*state.original.blocks]
    for obj in {id(block.bundle):block.bundle for block in blocks}.values():
        if getattr(obj,'_isolated_cache_heightless_prior',False):
            obj.priors=lambda q,cache,bundle=obj:tuple(x[1:]for x in MovingPivotSourceBundle.priors(bundle,q,cache))


def serialize_prepared(state):
    removed=[]
    blocks=[*state.blocks,*state.original.blocks]
    for obj in {id(block.bundle):block.bundle for block in blocks}.values():
        value=vars(obj).get('priors')
        if callable(value) and value.__name__=='<lambda>':
            removed.append((obj,value))
            obj._isolated_cache_heightless_prior=True
            del vars(obj)['priors']
    try: return pickle.dumps(state,protocol=5)
    finally:
        for obj,value in removed: obj.priors=value


def capture(args, timing):
    from calibration.solver_runtime.runtime import verify_legacy_dependencies
    verify_legacy_dependencies()
    import z190_joint_intercamera_fixed_xyz_source as c
    import z190_joint_shared_F_height_source as shared
    import z190_joint_rink_midpoint_source as midpoint
    old_module_at=shared.module_at
    def registered_module_at(path):
        mod=old_module_at(path)
        sys.modules[mod.__name__]=mod
        return mod
    shared.module_at=registered_module_at
    midpoint.module_at=registered_module_at
    registered_module_at(N/'ice_alignment_review/main9_full_source_factory.py')
    cache = BASE / 'prepared_full8503_state.pickle'
    if cache.exists():
        state = measured('prepare_cache_load', lambda: pickle.loads(cache.read_bytes()), timing)
        restore_local_priors(state)
    else:
        state = measured('prepare_actual8503', lambda: c.prepare(args), timing)
        try:
            measured('prepare_cache_save', lambda: cache.write_bytes(serialize_prepared(state)), timing)
        except (pickle.PickleError, TypeError, AttributeError) as exc:
            write(BASE / 'prepare_pickle_limitation.json', dict(type=type(exc).__name__, error=str(exc)))
    # This only compiles an in-process function from the copied driver. It does
    # not modify the snapshot or skip computations inside any source evaluator.
    source = inspect.getsource(c.run)
    marker = "    write(state.output/'input_manifest.json',manifest);guard(state.q);source_review(state.q,dict(stage='initial_unaccepted'))"
    if source.count(marker) != 1: raise RuntimeError('Driver capture boundary changed')
    source = source[:source.index(marker)] + '    return locals()\n'
    source = source.replace('def run(args):', 'def isolated_capture(args):', 1)
    original_prepare = c.prepare
    c.prepare = lambda unused: state
    try:
        namespace = dict(c.__dict__)
        exec(compile(source, str(BASE/'driver_closure_capture'), 'exec'), namespace)
        captured = namespace['isolated_capture'](args)
    finally: c.prepare = original_prepare
    write(BASE / 'captured_source_inventory.json', dict(q_size=len(state.q), actual_exposures=sum(len(b.bundle.stamps) for b in state.blocks),
        source_rows=len(captured['scales']), block_names=[b.name for b in state.blocks], block_exposures=[len(b.bundle.stamps) for b in state.blocks],
        shared_indices=captured['shared_schur_indices'], source_seed=sha(args.joint_state_seed), solver_file=str(Path(c.solve_joint_source.__code__.co_filename)),
        program_blocks_untouched=True))
    return captured


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode', choices=('prepare','profile-source','fixed','iteration','absolute-fixture','suite','pattern-parity','initial-fd','initial-normal'), default='prepare')
    p.add_argument('--label', default='baseline')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--cache-native-replay', action='store_true')
    p.add_argument('--enable-fast', action='store_true')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--profiles-from',type=Path)
    p.add_argument('--pattern-workers',type=int,default=0)
    p.add_argument('--source-processes',action='store_true')
    p.add_argument('--actor-threads',type=int,default=3)
    p.add_argument('--accepted-iterations',type=int,default=1)
    p.add_argument('--start-from',type=Path)
    p.add_argument('--warm-source',action='store_true')
    p.add_argument('--fd-column',type=int,default=1)
    p.add_argument('--fd-width',type=float,default=1e-5)
    args=p.parse_args()
    if not args.enable_fast:
        os.environ['RINKSPACE_MOTION_BACKEND']='numpy'
        os.environ['RINKSPACE_GEOMETRY_SPARSE_FD']='0'
        os.environ['RINKSPACE_PROFILE_MOTION_ONLY']='0'
        os.environ['RINKSPACE_PATTERN_WORKERS']='0'
    else:
        os.environ['RINKSPACE_MOTION_BACKEND']='native'
        os.environ['RINKSPACE_MOTION_THREADS']=str(args.threads)
        os.environ['RINKSPACE_SCATTER_THREADS']=str(args.threads)
        os.environ['RINKSPACE_GEOMETRY_SPARSE_FD']='1'
        os.environ['RINKSPACE_PROFILE_MOTION_ONLY']='1'
        os.environ['RINKSPACE_PATTERN_WORKERS']=str(args.pattern_workers)
    timing={}
    seed=freeze_seed()
    driver_args=arguments(seed)
    driver_args.source_workers=args.workers
    driver_args.source_processes=args.source_processes
    driver_args.source_actor_threads=args.actor_threads
    cap=capture(driver_args,timing)
    actors=cap.get('source_actors')
    if actors is not None:
        measured(args.label+'_actor_startup',actors.start,timing)
    if args.mode=='prepare': return
    import z190_joint_source_solver as solver
    state=cap['state']; q=state.q.copy()
    if args.mode=='pattern-parity':
        os.environ['RINKSPACE_PATTERN_WORKERS']='0'
        original=measured('pattern_sequential',lambda:state.patterns.evaluate(q,True),timing)
        os.environ['RINKSPACE_PATTERN_WORKERS']=str(args.pattern_workers or 4)
        parallel=measured('pattern_parallel_cold',lambda:state.patterns.evaluate(q,True),timing)
        warm=measured('pattern_parallel_warm',lambda:state.patterns.evaluate(q,True),timing)
        delta=(original[1]-parallel[1]).tocoo()
        write(BASE/'pattern_parallel_equivalence.json',dict(residual_exact=np.array_equal(original[0],parallel[0]),jacobian_sparse_values_exact=not np.any(delta.data),residual_max_error=float(np.max(np.abs(original[0]-parallel[0]),initial=0.)),jacobian_max_error=float(np.max(np.abs(delta.data),initial=0.)),warm_residual_exact=np.array_equal(parallel[0],warm[0]),timing=timing))
        return
    profiles=cap['profile_update']
    if args.mode=='initial-fd':
        target=BASE/(args.label+'_initial_fd');target.mkdir(exist_ok=True)
        entries=[]
        def loss_rows(r,mask):
            a=np.abs(r)
            return np.where(mask,.5*r*r,np.where(a<=2.,.5*r*r,2.*a-2.))
        def take_jac():
            for block in cap['iter_source_blocks'](q,True):
                family=block.row_ledger['family'];r=np.asarray(block.r).copy();quad=np.asarray(block.quadratic,bool).copy()
                weights=np.minimum(1.,2./np.maximum(np.abs(r),1e-12));weights[quad]=1.
                column=np.asarray(block.J[:,args.fd_column].toarray()).ravel()
                entries.append(dict(family=family,r0=r,quadratic=quad,column=column,gradient=float(column@(weights*r)),initial_cost=solver.source_cost(r,quad)))
        measured(args.label+'_initial_jac',take_jac,timing)
        for sign,key in ((1,'plus'),(-1,'minus')):
            trial=q.copy();trial[args.fd_column]+=sign*args.fd_width
            def take_cost():
                for entry,block in zip(entries,cap['iter_source_blocks'](trial,False)):
                    if entry['family']!=block.row_ledger['family']:raise ValueError('FD source order changed')
                    entry[key]=np.asarray(block.r).copy();entry[key+'_cost']=solver.source_cost(block.r,block.quadratic)
            measured(args.label+'_initial_'+key,take_cost,timing)
        report=[]
        for i,entry in enumerate(entries):
            fd=(entry['plus_cost']-entry['minus_cost'])/(2*args.fd_width)
            rows_g=entry['column']*(np.minimum(1.,2./np.maximum(abs(entry['r0']),1e-12))*entry['r0']);rows_g[entry['quadratic']]=entry['column'][entry['quadratic']]*entry['r0'][entry['quadratic']]
            rows_fd=(loss_rows(entry['plus'],entry['quadratic'])-loss_rows(entry['minus'],entry['quadratic']))/(2*args.fd_width)
            top=np.argsort(np.abs(rows_g-rows_fd))[-12:][::-1]
            report.append(dict(family=entry['family'],gradient=entry['gradient'],source_fd=fd,delta=entry['gradient']-fd,initial_cost=entry['initial_cost'],plus_cost=entry['plus_cost'],minus_cost=entry['minus_cost'],top_rows=[dict(row=int(row),r0=float(entry['r0'][row]),rp=float(entry['plus'][row]),rm=float(entry['minus'][row]),jac=float(entry['column'][row]),g=float(rows_g[row]),fd=float(rows_fd[row]),difference=float(rows_g[row]-rows_fd[row]),quadratic=bool(entry['quadratic'][row]))for row in top]))
            np.savez(target/('block_%02d'%i),r0=entry['r0'],rp=entry['plus'],rm=entry['minus'],column=entry['column'],quadratic=entry['quadratic'])
        write(target/'report.json',dict(column=args.fd_column,width=args.fd_width,blocks=report,total_gradient=sum(row['gradient']for row in report),total_fd=sum(row['source_fd']for row in report),profile_not_updated=True,timing=timing))
        if actors is not None:actors.close()
        return
    if args.start_from:
        with np.load(args.start_from,allow_pickle=False)as continued:
            if continued['q'].shape!=q.shape or not np.array_equal(continued['quadratic'],cap['quadratic'])or not np.array_equal(continued['scales'],cap['scales']):
                raise ValueError('Continuation checkpoint q/mask/scales inventory changed')
            q=continued['q'].copy()
            profiles.restore_checkpoint({key[9:]:continued[key]for key in continued.files if key.startswith('profile__')})
        state.q=q.copy()
    if args.warm_source:
        from z190_streaming_source_normal import accumulate_source_normal
        token=solver._frozen_profile_token(profiles)
        warmed=measured(args.label+'_startup_full_source_rJ',lambda:accumulate_source_normal(lambda:cap['iter_source_blocks'](q,True),len(q),expected_row_count=len(cap['scales']),frozen_profile_token=token),timing)
        write(BASE/(args.label+'_startup_full_source_report.json'),warmed.report)
        del warmed
    if args.mode=='absolute-fixture':
        fixture=[]
        for block in state.blocks:
            kinds=set()
            for pts,rows in block.groups:
                for row in rows:
                    kind=(row.get('curve_observation_kind'),row.get('kind'),row.get('source_world_feature_id'),bool('normal_interval'in row))
                    if kind in kinds:continue
                    kinds.add(kind)
                    rebound=block.geometry.rebind(row,q[block.geometry_parameter_indices])
                    cam=block.bundle.camera(block.local(q),pts)
                    r,j=block.absolute(cam,[rebound],True)
                    fixture.append(dict(block=block.name,pts=int(pts),kind=kind,camera25=cam,rows=[rebound],absolute_evaluator_module=block.absolute.__module__,absolute_evaluator_name=block.absolute.__name__,baseline_r=r,baseline_j=j))
                    if len(kinds)>=12:break
                if len(kinds)>=12:break
        target=BASE/'real_absolute_fixture.pkl'
        target.write_bytes(pickle.dumps(fixture,protocol=5))
        write(target.with_suffix('.json'),dict(entries=len(fixture),bytes=target.stat().st_size,kinds=[entry['kind']for entry in fixture]))
        return
    import z190_profiled_motion_schur as motion
    original_motion=motion.SchurMotionKernel.__call__
    def capture_motion(self,**k):
        fixture=BASE/('real_motion_fixture_jac.npz'if k['jacobian']else'real_motion_fixture_residual.npz')
        if not fixture.exists():
            np.savez(fixture,**{key:np.asarray(value) for key,value in k.items()if value is not None})
            write(fixture.with_suffix('.json'),dict(none_keys=[key for key,value in k.items()if value is None]))
        return original_motion(self,**k)
    motion.SchurMotionKernel.__call__=capture_motion
    if args.mode in ('profile-source','fixed','suite','initial-normal'):
        if args.mode!='initial-normal':
            measured(args.label+'_profile_update', lambda:profiles(q), timing, args.profile)
            if actors is not None:write(BASE/(args.label+'_actor_profile_stats.json'),actors.last_stats)
        if args.profiles_from:
            with np.load(args.profiles_from,allow_pickle=False)as frozen:
                exact={key[9:]:frozen[key]for key in frozen.files if key.startswith('profile__')}
                current=profiles.checkpoint_arrays()
                write(BASE/(args.label+'_profile_update_comparison.json'),dict(max_absolute_weights_error=max(float(np.max(np.abs(np.asarray(current[key],float)-np.asarray(value,float)),initial=0.))for key,value in exact.items()),frozen_profile_reference=str(args.profiles_from)))
                profiles.restore_checkpoint(exact)
        if args.mode!='initial-normal':
            measured(args.label+'_source_residual', lambda: [(len(block.r)) for block in cap['iter_source_blocks'](q,False)],timing,args.profile)
            if actors is not None:write(BASE/(args.label+'_actor_residual_stats.json'),actors.last_stats)
        from z190_streaming_source_normal import accumulate_source_normal
        token=solver._frozen_profile_token(profiles)
        normal=measured(args.label+'_source_normal', lambda:accumulate_source_normal(lambda:cap['iter_source_blocks'](q,True),len(q),expected_row_count=len(cap['scales']),frozen_profile_token=token),timing,args.profile)
        if actors is not None:write(BASE/(args.label+'_actor_normal_stats.json'),actors.last_stats)
        from scipy.sparse import save_npz
        target=BASE/args.label
        target.mkdir(exist_ok=True)
        save_npz(target/'normal.npz',normal.normal)
        np.savez(target/'fixed_outputs.npz',q=q,gradient=normal.gradient,cost=normal.cost,diagonal=np.maximum(normal.normal.diagonal(),1.),global_indices=cap['shared_schur_indices'],quadratic=cap['quadratic'],scales=cap['scales'],**{'profile__'+k:v for k,v in profiles.checkpoint_arrays().items()})
        write(target/'normal_report.json',normal.report)
        write(target/'timings.json',timing)
        if args.mode!='suite':
            if actors is not None:actors.close()
            return
    fixed=[] if state.external is None else [state.external_start+4*state.external.index[p]+c for p in state.external.held_pts for c in range(4)]
    out=BASE/(args.label+'_iteration')
    with np.load(args.start_from or seed,allow_pickle=False) as saved: damping=float(saved['damping'])
    accepted_count=0
    def finish(q,event):
        nonlocal accepted_count
        if event['checkpoint_kind']=='accepted':
            accepted_count+=1
            if accepted_count>=args.accepted_iterations:return dict(stop_branch=True,reason='Requested complete real-source iterations measured by isolated benchmark harness')
    result=measured(args.label+'_iteration',lambda:solver.solve_joint_source(cap['evaluate'],q,cap['quadratic'],cap['scales'],out,cap['guard'],
        update_profiles=profiles,checkpoint_callback=finish,initial_damping=damping,
        global_schur_count=len(cap['shared_schur_indices']),global_schur_indices=cap['shared_schur_indices'],source_block_factory=cap['iter_source_blocks'],
        fixed_parameter_indices=fixed,schur_backtracking=True,schur_view_lm_ratio=1.,schur_backtracking_aware_damping=True,
        cache_native_source_replay=args.cache_native_replay,source_stage_timing=True),timing,args.profile)
    write(BASE/(args.label+'_iteration_summary.json'),dict(timing=timing,history=result.history,status=result.status,q_sha256=hashlib.sha256(result.q.tobytes()).hexdigest(),explicit_held_states_unchanged=np.array_equal(q[fixed],result.q[fixed]),start_from=None if args.start_from is None else str(args.start_from),start_state_sha256=hashlib.sha256(q.tobytes()).hexdigest(),requested_accepted_iterations=args.accepted_iterations))
    if actors is not None:
        write(BASE/(args.label+'_actor_final_stats.json'),actors.last_stats)
        actors.close()


if __name__=='__main__': main()
