"""Complete source branch with three secondary installation lens increments.

The complete original source graph and accepted state are retained. This is
an explicitly identified model candidate, with all 32 shared source FD gates.
"""
from calibration.solver_runtime.config import data_root as _runtime_data_root, data_module_dir as _runtime_data_module_dir
from pathlib import Path
import argparse,hashlib,json,os,time
import sys
BASE=_runtime_data_root()

def load_graph(seed,output):
    import benchmark_harness as harness
    import numpy as np
    from z190_joint_source_solver import make_profile_update
    args=harness.arguments(seed)
    args.source_processes=False;args.source_workers=1
    cap=harness.capture(args,{})
    state=cap['state'];del cap
    reference=BASE/'frozen_current_v12_seed.npz'
    with np.load(seed,allow_pickle=False)as z,np.load(reference,allow_pickle=False)as old:
        if str(z['schema'])!='rinkspace.joint-source-LM-checkpoint.v1':raise ValueError('Original accepted state schema required')
        from calibration.solver_runtime.checkpoint_history import read_checkpoint_history
        history=read_checkpoint_history(z,seed)
        if not history:raise ValueError('Original optimizer state history required')
        last=history[-1];accepted=[event for event in history if event.get('accepted')is True]
        if not accepted:raise ValueError('Seed has no optimizer-accepted state')
        previous=accepted[-1]
        if not float(previous['next_cost'])<float(previous['cost']):raise ValueError('Accepted history did not reduce its actual source cost')
        seed_sha=hashlib.sha256(seed.read_bytes()).hexdigest()
        terminal=last.get('accepted')is False
        if terminal:
            # The original solver writes its final rejected-trial diagnostic
            # into latest_accepted_state as well. It keeps the accepted q and
            # profile; distinguish that terminal metadata from an accepted step.
            known='02c2309cbcd1d46352bfe45729efc136d851145a6025066ae7bd6f1eb1822a40'
            if (seed_sha!=known or history[-2]is not previous or
                    float(last['cost'])!=float(last['next_cost'])or
                    float(last['cost'])!=float(previous['next_cost'])):
                raise ValueError('Only the audited unchanged terminal q/profile checkpoint may follow a failed trial')
        if z['q'].shape!=state.q.shape or not np.isfinite(z['q']).all():raise ValueError('Complete original finite q required')
        for key in old.files:
            if key.startswith('aux__')or key in('quadratic','scales'):
                if key not in z or z[key].dtype!=old[key].dtype or not np.array_equal(z[key],old[key]):
                    raise ValueError('Original complete source binding changed: '+key)
        expected={k for k in old.files if k.startswith('profile__')}
        if expected!={k for k in z.files if k.startswith('profile__')}:raise ValueError('Original profile inventory changed')
        identity=json.loads(str(z['identity_json']))
        authority=identity['input_manifest'];manifest=Path(authority['path'])
        if hashlib.sha256(manifest.read_bytes()).hexdigest()!=authority['sha256']:raise ValueError('Seed source manifest hash changed')
        state.q=z['q'].copy();state.seed_contract=(z['quadratic'].copy(),z['scales'].copy())
        make_profile_update(state.blocks).restore_checkpoint({k[9:]:z[k].copy()for k in expected})
        damping=float(z['damping'])
        if not np.isfinite(damping)or damping<=0:raise ValueError('Finite accepted damping required')
        provenance=dict(state_seed=str(seed),state_seed_sha256=seed_sha,
            active_initial_q='exact original terminal q/profile followed by three zero lens increments'if terminal else'exact accepted seed followed by three zero lens increments',
            terminal_failed_trial_did_not_change_q_or_profile=terminal,
            original_seed_manifest=authority,last_accepted_iteration=int(previous['iteration']),
            original_source_rows=len(z['scales']),original_parameter_count=len(z['q']),
            original_cost=float(last['next_cost']),prepared_graph_cache=str(BASE/'prepared_full8503_state.pickle'),
            prepared_pattern_initialization=state.patterns.manifest.get('initialization'),
            prepared_pattern_initialization_metadata_is_historical_cache_origin=True,
            original_source_UV_sigma_fit_held_and_aliases_preserved=True)
    state.output=output
    args.output=output;args.joint_state_seed=seed
    args.initial_source_fd_columns=None;args.initial_source_cost_fd_step=1e-6
    args.source_processes=True;args.source_actor_threads=3;args.source_workers=1
    args.cache_native_source_replay=True;args.cache_complete_source_review=True
    args.compact_source_review=True;args.source_stage_timing=True
    return state,args,damping,provenance

def capture_driver(driver,args,state):
    import inspect
    text=inspect.getsource(driver.run)
    marker="    write(state.output/'input_manifest.json',manifest);guard(state.q);source_review(state.q,dict(stage='initial_unaccepted'))"
    if text.count(marker)!=1:raise RuntimeError('Candidate driver capture boundary changed')
    text=text[:text.index(marker)]+'    return locals()\n'
    text=text.replace('def run(args):','def isolated_capture(args):',1)
    namespace=dict(driver.__dict__);namespace['prepare']=lambda unused:state
    exec(compile(text,str(BASE/'secondary_optics_candidate/driver_capture'),'exec'),namespace)
    return namespace['isolated_capture'](args)

def checked_initial_state_evidence(directory,state,args,provenance):
    """Reuse the complete 32 gate only for its identical q/profile/source state."""
    import math
    directory=directory.resolve()
    if not directory.is_relative_to(BASE.resolve()):raise ValueError('Own-task complete FD evidence required')
    manifest_path=directory/'input_manifest.json'
    checks_path=directory/'actual_global_source_cost_direction_checks.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    checks=json.loads(checks_path.read_text(encoding='utf8'))
    expected=set(range(14))|set(range(state.patterns.start,state.patterns.start+9))|set(map(int,state.geometry_parameter_indices[5:]))|set(map(int,state.optics_parameter_indices))
    if (len(checks)!=32 or {int(c['column'])for c in checks}!=expected or
            any(not math.isfinite(float(c['relative_error']))or float(c['relative_error'])>.005 for c in checks)):
        raise ValueError('Original complete 32 shared direction gate did not pass')
    seed=manifest['initial_joint_source_state']
    if seed['sha256']!=provenance['state_seed_sha256']:
        raise ValueError('Checked initial q/profile checkpoint bytes differ')
    if (float(manifest['geometry_residual_fd_step_m'])!=float(os.environ['RINKSPACE_GEOMETRY_FD_STEP_M'])or
            float(manifest['initial_source_cost_fd_step'])!=float(args.initial_source_cost_fd_step)):
        raise ValueError('Checked source geometry/cost derivative definition changed')
    for key in('base_calibration','parameter_indices','physical_column_scales','physical_derivative_steps','original_parameter_count'):
        if manifest['secondary_shared_optics'][key]!=state.optics_contract[key]:
            raise ValueError('Checked installation lens contract changed: '+key)
    for check in checks:
        col=int(check['column']);step=args.initial_source_cost_fd_steps.get(col,args.initial_source_cost_fd_step)
        if float(check['step'])!=float(step):raise ValueError('Checked full-source numerical direction step changed')
    # All three extensions and the complete driver were written before this
    # gate and remain unchanged; numeric damping does not enter r/J/profile.
    math_files=[BASE/'secondary_optics_candidate'/name for name in(
        'secondary_optics_graph.py','z190_motion_optics_extension.py',
        'z190_joint_intercamera_fixed_xyz_optics_source.py')]
    if any(path.stat().st_mtime_ns>checks_path.stat().st_mtime_ns for path in math_files):
        raise ValueError('Candidate mathematical source was edited after the complete gate')
    return dict(seed_source=provenance['state_seed'],seed_sha256=provenance['state_seed_sha256'],
        accepted_source_cost=provenance['original_cost'],
        original_input_manifest=str(manifest_path),original_input_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        original_initial_FD_evidence=str(checks_path),original_initial_FD_evidence_sha256=hashlib.sha256(checks_path.read_bytes()).hexdigest(),
        verified_identical_initial_q_profiles_and_complete32_source_definition=True,
        initial_state_gate_reused_after_numerical_damping_change=True,
        current_math_file_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in math_files},
        optimizer_acceptance_is_not_source_alignment_accuracy=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=Path,default=BASE/'main_f514_stable_branch_full29/solver/latest_accepted_state.npz')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--geometry-fd-step',type=float,default=1e-7)
    parser.add_argument('--cost-fd-step',type=float,default=1e-7)
    parser.add_argument('--initial-damping',type=float)
    parser.add_argument('--initial-fd-evidence',type=Path)
    parser.add_argument('--verification-iterations',type=int)
    args=parser.parse_args()
    out=args.output.resolve()
    if not out.is_relative_to(BASE.resolve())or out==BASE.resolve()or out.exists():parser.error('New own-task output directory required')
    if args.verification_iterations is not None and args.verification_iterations<1:parser.error('Positive verification iterations required')
    if not all(float(s)>0 and __import__('math').isfinite(float(s))for s in(args.geometry_fd_step,args.cost_fd_step)):
        parser.error('Finite positive source FD widths required')
    if args.initial_damping is not None and not (args.initial_damping>0 and __import__('math').isfinite(args.initial_damping)):
        parser.error('Finite positive numerical damping required')
    for name in('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    for name in('RINKSPACE_MOTION_BACKEND','RINKSPACE_SCATTER_BACKEND','RINKSPACE_PROJECTION_BACKEND'):os.environ[name]='native'
    os.environ['RINKSPACE_MOTION_THREADS']='3';os.environ['RINKSPACE_SCATTER_THREADS']='3'
    os.environ['RINKSPACE_PROJECTION_VALUE_BACKEND']='opencv';os.environ['RINKSPACE_SOLVER_ACCELERATION']='auto'
    os.environ['RINKSPACE_GEOMETRY_SPARSE_FD']='1';os.environ['RINKSPACE_GEOMETRY_FD_STEP_M']=str(args.geometry_fd_step)
    os.environ['RINKSPACE_PROFILE_MOTION_ONLY']='1';os.environ['RINKSPACE_PATTERN_WORKERS']='4'
    from isolated_solver_bootstrap import install
    install()
    state,solve_args,damping,provenance=load_graph(args.seed.resolve(),out)
    solve_args.initial_source_cost_fd_step=args.cost_fd_step
    from secondary_optics_candidate.secondary_optics_graph import install_secondary_optics
    from secondary_optics_candidate import z190_joint_intercamera_fixed_xyz_optics_source as driver
    contract=install_secondary_optics(state)
    contract['seed_provenance']=provenance
    solve_args.initial_source_cost_fd_steps={int(c):float(s)for c,s in zip(state.optics_parameter_indices,(1e-4,1e-4,1e-3))}
    if args.initial_fd_evidence is not None:
        solve_args.accepted_checkpoint_continuation=checked_initial_state_evidence(args.initial_fd_evidence,state,solve_args,provenance)
    if args.initial_damping is not None:
        contract['numerical_damping_initialization']=dict(saved=damping,selected=args.initial_damping,
            source_objective_J_profiles_and_priors_unchanged=True)
        damping=args.initial_damping
    solve_args.solve=not args.check_only
    def prepare(unused):
        out.mkdir(parents=True,exist_ok=False)
        (out/'extension_initialization.json').write_text(json.dumps(dict(contract=contract,provenance=provenance),ensure_ascii=False,indent=2),encoding='utf8')
        return state
    driver.prepare=prepare
    original_solve=driver.solve_joint_source
    def solve(*positional,**keywords):
        keywords['initial_damping']=damping
        if args.verification_iterations is not None:
            callback=keywords['checkpoint_callback'];accepted=0
            def review(q,event):
                nonlocal accepted
                result=callback(q,event)
                if event['checkpoint_kind']=='accepted':accepted+=1
                if accepted>=args.verification_iterations:
                    return dict(result or {},stop_branch=True,reason='Explicit candidate entry verification completed')
                return result
            keywords['checkpoint_callback']=review
        began=time.perf_counter();result=original_solve(*positional,**keywords)
        timing=dict(wall_seconds=time.perf_counter()-began,complete_source_and_initial_and_accepted_reviews=True,
            source_rows=len(positional[2]),parameter_count=len(positional[1]),
            shared_count=32,accepted_steps=sum(bool(x['accepted'])for x in result.history))
        (out/'actual_complete_solver_wall.json').write_text(json.dumps(timing,indent=2),encoding='utf8')
        print(json.dumps(dict(stage='actual_complete_optics_solver_wall',**timing)),flush=True)
        return result
    driver.solve_joint_source=solve
    try:driver.run(solve_args)
    finally:
        actors=getattr(state,'source_actors',None)
        if actors is not None:actors.close()
        from z190_parallel_pattern_evaluation import close_pattern_pools
        close_pattern_pools()

if __name__=='__main__':main()
