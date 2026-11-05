"""Same saved S05 state/objective execution check; never exports a user candidate."""
from pathlib import Path
import argparse, contextlib, hashlib, importlib.util, json, os, sys, time

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--base-root',type=Path,required=True)
    ap.add_argument('--frozen-run',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--before',type=Path)
    ap.add_argument('--accepted',type=int,default=2)
    a=ap.parse_args();p=a.project.resolve();b=a.base_root.resolve();f=a.frozen_run.resolve();out=a.output.resolve()
    if out.exists():raise FileExistsError('Preserve previous execution check: '+str(out))
    if a.accepted<0:raise ValueError('accepted must be nonnegative')
    # Load only the preserved numerical modules before production consumers.
    # All other code, inputs, camera, objective, and runtime are identical.
    if a.before:
        for name in ['body_execution','fullbody_fit','absolute_motion_state','body_resident_torch','continuous_motion_factors']:
            full='calibration.program_tracking.body_tracking.'+name
            path=a.before/'calibration/program_tracking/body_tracking'/(name+'.py')
            spec=importlib.util.spec_from_file_location(full,path);module=importlib.util.module_from_spec(spec)
            sys.modules[full]=module;spec.loader.exec_module(module)
    import numpy as np
    import torch
    from scipy.sparse import save_npz
    from . import spin_continuous_refit as run
    from .body_execution import BodyExecution,BodyExecutionConfig
    from .absolute_motion_curve import AbsoluteMotionCurve
    from .stage_quality import StageQualityTrace
    from .support_refinement import measure
    from .skate_run_policy import RunPolicy
    from scripts.profile_contact_solver import PhaseTrace
    from .rebuild_prepare import load,digest
    os.environ['RINKSPACE_BODY_NATIVE_LIBRARY']=str(b.parent/'body_perf_1103/native_cpu/rinkspace_core.dll')
    out.mkdir(parents=True);common=out/'common/exmr_hand_nofoot';common.mkdir(parents=True)
    for name in ['input.npz','observations.npz','compact_smplx.npz','export_source.npz','initial.npz']:
        os.link(f/'common/exmr_hand_nofoot'/name,common/name)
    for name in ['exmr_body_hand_nofoot.npz','palm_grip_geometry.json','grasp_relations.json']:
        os.link(f/name,out/name)
    document=json.loads((f/'source_evidence.json').read_text(encoding='utf-8'))
    initial=load(common/'initial.npz');curve=AbsoluteMotionCurve(initial['absolute_knots'],initial['absolute_seed_position_controls'],initial['absolute_seed_rotation_controls'])
    execution=BodyExecution(BodyExecutionConfig(workers=12,blas_threads=1,local_derivatives='analytic',compatibility_backend='cuda_resident'))
    started=time.perf_counter()
    try:
        with execution.limits(), (out/'progress.jsonl').open('w',encoding='utf-8',buffering=1) as log, PhaseTrace() as trace:
            parent,camera,info,_=run.restore_parent(p,b,b/'spin_entry_gravity_1104',out/'parent_geometry',p/run.PROGRAMME/'spin_entry_gravity_1104.npz')
            problem,review,_=run.make_problem(p,b,out,'joint',parent,camera,info,curve,document,execution,True,log)
            if a.before:problem.linear_solver_backend='lsmr'
            state=p/run.PROGRAMME/'spin_continuous_source_1105.npz';x=load(state)['x'].copy()
            baseline=json.loads((f/'source_stage/cross_stage_quality.json').read_text(encoding='utf-8'))['original_A']
            problem.stage_quality=StageQualityTrace(out/'cross_stage_quality.json',dict(same_frozen_source=True),baseline,lambda xx,v:measure(problem,review,xx,v))
            trace.install(problem,solve=True)
            resident=execution._compatibility_backend;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            report=dict(source_state_sha256=digest(state),observations_sha256=digest(common/'observations.npz'),source_evidence_sha256=digest(f/'source_evidence.json'),setup_seconds=time.perf_counter()-started,source_frames=problem.body.n,body_variables=problem.body.size,finite_flight=True,numerical_modules={})
            for name in ['body_execution','fullbody_fit','absolute_motion_state','body_resident_torch','continuous_motion_factors']:
                module=sys.modules['calibration.program_tracking.body_tracking.'+name];path=Path(module.__file__)
                report['numerical_modules'][name]=dict(path=str(path),sha256=digest(path))
            (out/'started.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps(dict(event='actual_problem_ready',output=str(out),before=bool(a.before))),flush=True)
            tick=time.perf_counter();cpu=time.process_time();v=problem.evaluate(x,True);torch.cuda.synchronize();elapsed=time.perf_counter()-tick
            report['complete_jacobian']=dict(seconds=elapsed,cpu_core_seconds=time.process_time()-cpu,rows=len(v['residual']),nnz=v['jac'].nnz,parts=v['parts'],phases=v.get('evaluation_phases'),gpu=resident.measurement())
            np.savez(out/'snapshot.npz',residual=v['residual'],x=x);save_npz(out/'jacobian.npz',v['jac'],compressed=False)
            print(json.dumps(dict(event='complete_jacobian',seconds=elapsed)),flush=True)
            if a.accepted:
                trace.reset();resident.reset_measurement();torch.cuda.reset_peak_memory_stats();problem.jacobian_seconds_hint=elapsed
                tick=time.perf_counter();cpu=time.process_time()
                result,record=problem.solve(x,RunPolicy(wall_seconds=3600,max_accepted=a.accepted,max_outer=1,stagnation_window=6,allow_penalty_growth=False))
                torch.cuda.synchronize();wall=time.perf_counter()-tick;value=problem.evaluate(result,False)
                np.savez(out/'final.npz',x=result,residual=value['residual'])
                report['solve']=dict(seconds=wall,cpu_core_seconds=time.process_time()-cpu,accepted=problem.accepted,retained=record['selection']['retained_step'],parts=value['parts'],cost=float(value['residual']@value['residual']/2),stop=record['stop'],trace=trace.snapshot(),gpu=resident.measurement())
            (out/'numerical_result.json').write_text(json.dumps(run.serial(report),indent=2),encoding='utf-8')
            from scripts.profile_body_solver import process_peak_rss_bytes
            report.update(linear_solver_statistics=getattr(problem,'linear_solver_statistics',[]),execution=execution.record(),peak_rss_bytes=process_peak_rss_bytes(),gpu_peak_allocated=int(torch.cuda.max_memory_allocated()),gpu_peak_reserved=int(torch.cuda.max_memory_reserved()),no_user_candidate_or_input_modified=True,actual_inner_linear_backend=getattr(problem,'linear_solver_backend','lsmr'))
            (out/'result.json').write_text(json.dumps(run.serial(report),indent=2),encoding='utf-8')
            print(json.dumps(dict(event='finished',jacobian=report['complete_jacobian']['seconds'],solve=report.get('solve',{}).get('seconds'),accepted=problem.accepted)),flush=True)
    finally:execution.close()

if __name__=='__main__':main()
