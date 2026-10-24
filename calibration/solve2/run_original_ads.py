"""New advertisement solve on the exactly recovered original first shot.

prepare: affected current-input rebinding/scope only; preflight: no Ceres solve;
solve: consumes current model/prepared/config and an optional complete state.
No control server, old packet producer, archived code or retired solver loads.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time,traceback,uuid
from dataclasses import replace,asdict
from pathlib import Path
import numpy as np
from rinksolve.model import load_model_bundle,save_model_bundle
from rinksolve.preparation import load_prepared
from rinksolve.original_ads_solver import OriginalAdsOptimizer,prepare_original_ad_scope
from rinksolve.original_reference import OriginalReferenceGuard
from rinksolve.native_backend import NativeBackend,SolveOptions
from rinksolve.constrained import ALState,ALOptions
from rinksolve.atomic_io import atomic_replace
from rinksolve.store import atomic_json,now_iso
from rinksolve.numerical_stagnation import stagnation_event,dominant_augmented_gradient
from rinksolve.runtime_state import CachedModelSnapshotWriter,RuntimeModelStore


def options_from(config):
    numeric=config['numerics'];inner=SolveOptions(**numeric['inner_solver'])
    return ALOptions(**numeric['augmented_lagrangian'],inner_options=inner,
        inner_parameter_metric=numeric['inner_parameter_metric'],continuous_inner=True)


def bind_current_policy(state,policy,config):
    current=asdict(policy)
    if state.policy==current:return state
    reason=config.get('numerical_policy_update_reason',config.get('continuation_policy_update_reason'))
    if not isinstance(reason,str) or not reason.strip():raise ValueError('changed current numerical policy needs an explicit reason in the run config')
    event={'kind':'explicit_numerical_policy_warm_restart','reason':reason,'before':state.policy,'after':current,
        'physical_parameters_source_E_W_Query_lambda_rho_unchanged':True,'native_private_TR_history_restored':False,'Ceres_accepted_step':False}
    return replace(state,policy=current,history=state.history+(event,))


def engine_from(config,*,initialize_loci):
    data=Path(config['data_root']).resolve();model=load_model_bundle(config['model_package'])
    reference=load_model_bundle(config['original_reference'])
    guard=OriginalReferenceGuard(reference);guard.assert_model(model)
    prepared=load_prepared(config['prepared'],data_root=data)
    if config.get('subproblem_ad_id'):
        from rinksolve.original_ad_scope import scope_one_ad
        prepared=scope_one_ad(prepared,model,config['subproblem_ad_id'])
    numeric=config['numerics']
    backend=NativeBackend(config['native_manifest'])
    engine=OriginalAdsOptimizer(model,prepared,backend,constraints_path=config['constraints_path'],
        whitening_manifest=config['whitening_manifest'],num_threads=numeric['num_threads'],geometry_difference_step=numeric['geometry_difference_step'],
        axis_difference_step=numeric['axis_difference_step'],row_equality_scale_px=numeric['row_equality_scale_px'],
        query_equality_scale_px=numeric['query_equality_scale_px'],phase=config.get('phase','feasibility'),
        free_mode='ad_layouts',initialize_loci=initialize_loci)
    guard.assert_free_blocks(model,engine.params.selected_blocks)
    engine.original_reference_guard=guard
    engine.assert_current_inputs()
    if config.get('resident_gpu_pipeline',False):
        from rinksolve.original_ads_gpu import attach_original_ads_gpu
        attach_original_ads_gpu(engine)
    if config.get('finite_corner_working_set',False):
        from rinksolve.original_ads_workset import attach_corner_working_set
        attach_corner_working_set(engine)
    if config.get('resume_unfinished_inner_metric'):
        spec=config['resume_unfinished_inner_metric'];values=np.load(spec['parameter_scales_file'],allow_pickle=False)
        if spec['problem_digest']!=engine.problem.digest:raise ValueError('unfinished inner metric belongs to another current graph')
        if values.shape!=engine.params.initial.shape or not np.isfinite(values).all() or np.any(values<=0):raise ValueError('unfinished inner metric dimensions/values')
        if hashlib.sha256(values.tobytes()).hexdigest()!=spec['record']['parameter_scales_sha256']:raise ValueError('unfinished inner metric array digest differs')
        if hashlib.sha256(engine.problem.parameter_scales.tobytes()).hexdigest()!=spec['record']['declared_scales_sha256']:raise ValueError('unfinished inner declared units differ')
        use_once=[True]
        def restored_metric(x):
            if not use_once[0]:return engine.full_uv_metric_scales(x)
            use_once[0]=False;record=dict(spec['record'],restored_unfinished_inner_chart=True,source_chart_recomputed=False)
            extra=engine.metric_event_callback(record,values) if callable(engine.metric_event_callback) else None
            if extra:record.update(extra)
            engine.last_inner_metric=record;return np.array(values,copy=True)
        engine.al.inner_scale_callback=restored_metric
    return engine,guard,data


def preflight(config,output,resume=None):
    engine,guard,data=engine_from(config,initialize_loci=resume is None)
    try:
        policy=options_from(config);p=engine.params;x=p.initial.copy()
        if resume:
            restored=bind_current_policy(ALState.load(resume,data_root=data),policy,config)
            if restored.problem_digest!=engine.problem.digest or restored.identity!=engine.problem.identity:raise ValueError('restored complete state differs from current formal solve entry')
            x=restored.parameters.copy()
            if restored.multipliers.shape!=(engine.problem.constraints.layout.residual_count,) or restored.penalties.shape!=(len(engine.problem.constraints.families),):raise ValueError('restored complete AL dimensions differ')
        model=p.model_for(x);preservation=guard.assert_model(model)
        cache=engine._compute(x,True)
        gpu=cache['forward'];cpu=cache['operator'].project_at(cache['xyz'],cache['times'])
        if not cpu['valid'].all():raise RuntimeError('CPU operator rejects the prepared initial ad state')
        max_difference=float(np.max(np.abs(gpu['uv']-cpu['uv'])))
        if max_difference>1e-7:raise RuntimeError('current CPU/GPU ad forward differs by '+str(max_difference))
        state=restored if resume else engine.al.initialize(x,policy)
        original_ice=[c.id for c in engine.model.scene.curves if c.kind in ('ice_circle','ice_segment')]
        report={'schema':'rinksolve.original_ads_preflight.v1','current_inputs_only':True,'uses_legacy_solver':False,'Ceres_started':False,'formal_entry_current_inputs_assertion_passed':True,'complete_ready_state_restore_checked':bool(resume),
            'model_package':config['model_package'],'prepared':config['prepared'],'native_build':config['native_manifest'],
            'preservation':preservation,'parameter_layout':p.inventory(),'CPU_GPU_UV_max_difference_px':max_difference,
            'projection_valid_rows':int(gpu['valid'].sum()),'current_ad_observations':p.count,
            'original_ice_curve_count':len(original_ice),'diagnostics':engine.current_diagnostics(x),
            'original_W_factors':len(engine.prepared.manifest['statistical_factors']),
            'original_W_rows':engine.stat_layout.residual_count,'genuine_Query_sources':len(engine.profile_rows),
            'continuous_inner_single_ceres_solve':policy.continuous_inner,'original_units_KKT_unchanged':True,
            'next_shot_allowed':False,'visual_acceptance':False}
        out=Path(output).resolve();out.relative_to(data);out.mkdir(parents=True,exist_ok=True)
        p.save_layout(out);state.save(out/'ready_state',data_root=data)
        atomic_json(out/'preflight.json',report)
        return report
    finally:engine.close()



def publication_base_revision(engine,state,output):
    """Continue the legal same-model publication counter across warm restarts."""
    base=engine.model.state.revision;package=Path(output)/'current_model'
    if not (package/'current.json').exists():return base
    current=RuntimeModelStore(package,engine.prepared.manifest['data_root']).load()
    expected=engine.params.model_for(state.parameters)
    if (current.id,current.revision)!=(engine.model.id,engine.model.revision):raise ValueError('existing output pointer belongs to another current model')
    if current.state.digest()!=expected.state.digest():raise ValueError('existing output accepted physical state differs from resumed complete checkpoint')
    return max(base,current.state.revision)


def solve(config,main_log,resume=None):
    data=Path(config['data_root']).resolve();log=Path(main_log).resolve();log.relative_to(data)
    log.parent.mkdir(parents=True,exist_ok=True)
    if log.exists():raise FileExistsError('each actual start needs its own unique main log')
    output=Path(config['output_root']).resolve();output.relative_to(data);output.mkdir(parents=True,exist_ok=True)
    stream=log.open('x',encoding='utf8',buffering=1)
    def emit(kind,**payload):
        stream.write(json.dumps({'time':now_iso(),'kind':kind,**payload},ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
    engine=None
    try:
        emit('starting',pid=os.getpid(),config=config,main_log=str(log),uses_legacy_solver=False,
            resume_checkpoint=str(resume) if resume else None,resume_semantics='warm restart from complete x/lambda/rho; native private trust-region rejection history is not serialized')
        engine,guard,data=engine_from(config,initialize_loci=resume is None);policy=options_from(config);p=engine.params
        snapshot_writer=CachedModelSnapshotWriter(engine.model,output/'snapshot_template')
        engine.domain_event_callback=lambda event,vector:emit('ad_domain_rejection',**event)
        metric_values={};stop_record={};working_set_events=[]
        def save_metric(record,scales):
            directory=output/'numerical_metrics';directory.mkdir(parents=True,exist_ok=True)
            path=directory/('S_'+record['source_native_parameters_sha256']+'_'+uuid.uuid4().hex[:8]+'.npy')
            values=np.ascontiguousarray(scales,dtype=np.float64);metric_values['scales']=values
            digest=hashlib.sha256(values.tobytes()).hexdigest()
            if digest!=record['parameter_scales_sha256']:raise ValueError('actual inner numerical metric digest differs')
            temporary=path.with_suffix('.tmp')
            with temporary.open('xb') as file:np.save(file,values,allow_pickle=False);file.flush();os.fsync(file.fileno())
            atomic_replace(temporary,path)
            details={'parameter_scales_file':str(path),'parameter_scales_array_sha256':digest,
                'parameter_scales_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'process_restart_is_warm_restart':True,'native_private_LM_history_serialized':False}
            emit('inner_metric',**record,**details);return details
        engine.metric_event_callback=save_metric
        if resume:
            state=bind_current_policy(ALState.load(resume,data_root=data),policy,config)
            if state.problem_digest!=engine.problem.digest or state.identity!=engine.problem.identity:raise ValueError('resume state/current model and scope differ')
        else:state=engine.al.initialize(p.initial,policy)
        guard.assert_free_blocks(engine.model,p.selected_blocks);p.save_layout(output/'layout')
        if hasattr(engine,'corner_working_set_solver'):
            def working_set_event(event,vector):
                working_set_events.append(event)
                current=replace(state,parameters=vector,phase='verified_corner_working_set_transition',history=state.history+tuple(working_set_events))
                guard.assert_model(p.model_for(vector));current.save(output/'checkpoint',data_root=data)
                emit('finite_corner_working_set_event',event=event,complete_x_lambda_rho_saved=True,
                    original_source_E_W_Query_geometry_unchanged=True)
            engine.corner_working_set_solver.event_callback=working_set_event
        state.save(output/'checkpoint',data_root=data)
        accepted=0;serial=0;publication_base=publication_base_revision(engine,state,output);emit('publication_counter',base_parameter_revision=publication_base);previous_accepted_x=[np.array(state.parameters,copy=True)]
        def publish_model(vector,event):
            nonlocal serial
            serial+=1
            model=p.model_for(vector,revision=publication_base+serial,state_id='original_ads@'+str(serial))
            preservation=guard.assert_model(model)
            package=output/'current_model';generation='generations/g%08d_'%model.state.revision+uuid.uuid4().hex[:8]
            snapshot_writer.write(model,package/generation)
            atomic_json(package/'current.json',{'schema':'rinksolve.current_model_pointer.v1','generation':generation,
                'model_id':model.id,'model_revision':model.revision,'state_id':model.state.id,
                'parameter_revision':model.state.revision,'sha256':model.state.digest(),'updated_at':now_iso()})
            return preservation
        def checkpoint(current,event):
            guard.assert_model(p.model_for(current.parameters));current.save(output/'checkpoint',data_root=data)
            preservation=publish_model(current.parameters,event)
            emit('checkpoint',event=event,outer_iteration=current.outer_iteration,phase=current.phase,diagnostics=current.diagnostics,preservation=preservation)
        def iteration(event,parameters):
            nonlocal accepted
            callback_times={};clock=time.perf_counter();guard.assert_model(p.model_for(parameters));callback_times['guard_seconds']=time.perf_counter()-clock
            if event.get('step_successful') and not event.get('initial_state'):
                event=dict(event,actual_inner_chart_step_norm=float(np.linalg.norm((parameters-previous_accepted_x[0])/metric_values.get('scales',p.scales))),ceres_reported_step_norm_units='ambient')
                accepted+=1
                clock=time.perf_counter();actual_diagnostics,_=engine.al._diagnostics(parameters,state.multipliers,state.penalties,jacobian=False)
                callback_times['actual_current_AL_diagnostics_seconds']=time.perf_counter()-clock
                actual_diagnostics.update(current_inner_iteration=event,visual_acceptance=False,
                    diagnostics_current_at_saved_parameters=True,diagnostic_native_x_sha256=hashlib.sha256(np.ascontiguousarray(parameters).tobytes()).hexdigest(),
                    normalized_augmented_stationarity_inf=event.get('normalized_augmented_stationarity_inf'))
                current=replace(state,parameters=parameters,phase='inner_progress',last_inner={'iteration':event,'accepted_steps_this_process':accepted},
                    diagnostics=actual_diagnostics)
                clock=time.perf_counter();current.save(output/'checkpoint',data_root=data);callback_times['complete_checkpoint_seconds']=time.perf_counter()-clock
                clock=time.perf_counter();preservation=publish_model(parameters,event);callback_times['atomic_model_generation_seconds']=time.perf_counter()-clock
                clock=time.perf_counter();ad_diagnostics=engine.current_diagnostics(parameters);callback_times['current_ad_diagnostics_seconds']=time.perf_counter()-clock
                clock=time.perf_counter();emit('accepted',accepted_steps_this_process=accepted,event=event,ad_diagnostics=ad_diagnostics,preservation=preservation)
                callback_times['main_log_emit_seconds']=time.perf_counter()-clock;emit('accepted_callback_timing',accepted_steps_this_process=accepted,timings=callback_times)
                stalled=stagnation_event(event,parameters,metric_values.get('scales',p.scales),policy.inner_stationarity_tolerance,previous_parameters=previous_accepted_x[0])
                previous_accepted_x[0]=np.array(parameters,copy=True)
                if stalled:
                    stop_record.update(stalled);emit('numerical_stagnation_detected',**{k:v for k,v in stalled.items() if k!='kind'});return 1
            elif event.get('initial_state'):
                previous_accepted_x[0]=np.array(parameters,copy=True);emit('inner_start',event=event,outer_iteration=state.outer_iteration)
            else:emit('trial',event=event)
            return 0
        emit('current_input_ready',parameter_layout=p.inventory(),initial_ad_diagnostics=engine.current_diagnostics(state.parameters),
            original_W_factors=len(engine.prepared.manifest['statistical_factors']),continuous_inner=True,
            finite_corner_working_set_enabled=bool(config.get('finite_corner_working_set',False)),
            native_context_change_rule='only verified finite-corner active-set changes' if config.get('finite_corner_working_set') else 'one Ceres solve per fixed AL inner',
            recovered_non_ad_preservation=guard.assert_model(p.model_for(state.parameters)))
        while True:
            engine.assert_current_inputs()
            result=engine.al.run(state,policy,outer_iterations=1,on_checkpoint=checkpoint,on_inner_iteration=iteration)
            state=result.state
            if working_set_events:
                state=replace(state,history=state.history+tuple(working_set_events));working_set_events.clear()
                state.save(output/'checkpoint',data_root=data)
            if stop_record:
                state=replace(state,phase='numerical_stagnation_requires_repair',history=state.history+(dict(stop_record),))
                state.save(output/'checkpoint',data_root=data)
                diagnostic=dominant_augmented_gradient(engine,state)
                atomic_json(output/'numerical_stagnation.json',{'event':stop_record,'gradient':diagnostic,'current_ad_sources':engine.current_diagnostics(state.parameters),
                    'complete_checkpoint':str(output/'checkpoint/state.json'),'restart_requires_a_concrete_numerical_or_model_repair':True})
                emit('numerical_stagnation_requires_repair',diagnostic=str(output/'numerical_stagnation.json'),mathematical_convergence=False,complete_state_preserved=True)
                engine.write_conflict_report(state.parameters,output/'conflicts.json');return 'numerical_stagnation_requires_repair'
            emit('outer_result',status=result.status,outer_iteration=state.outer_iteration,phase=state.phase,diagnostics=state.diagnostics)
            if result.status=='outer_chunk_exhausted':continue
            if result.numerically_converged and engine.phase=='feasibility' and state.parameters[p.budget_offset]<=engine.target_budget_sq+policy.feasibility_tolerance:
                x=engine.set_phase('statistics',state.parameters);state=engine.al.initialize(x,policy)
                emit('phase_transition',from_phase='feasibility',to_phase='statistics',same_physical_parameters=True,original_complete_W_statistics=True)
                checkpoint(state,{'kind':'explicit_statistics_phase_after_current_E_feasibility'})
                continue
            emit('requires_review',numerical_status=result.status,numerical_stage=engine.phase,
                visual_acceptance=False,next_shot_allowed=False,accepted_steps_this_process=accepted)
            engine.write_conflict_report(state.parameters,output/'conflicts.json');break
        return result.status
    except BaseException as exc:
        emit('failure',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        raise
    finally:
        if engine is not None:engine.close()
        stream.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--source',required=True);prep.add_argument('--model',required=True);prep.add_argument('--output',required=True);prep.add_argument('--data-root',required=True)
    pre=sub.add_parser('preflight');pre.add_argument('--config',required=True);pre.add_argument('--output',required=True);pre.add_argument('--resume')
    run=sub.add_parser('solve');run.add_argument('--config',required=True);run.add_argument('--main-log',required=True);run.add_argument('--resume')
    args=parser.parse_args(argv)
    if args.command=='prepare':
        source=load_prepared(args.source,data_root=args.data_root);model=load_model_bundle(args.model)
        result=prepare_original_ad_scope(source,model,args.output,data_root=args.data_root)
        print(json.dumps(result.manifest['original_ad_scope'],ensure_ascii=False));return 0
    config=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    if config.get('schema')!='rinksolve.original_ads_run_policy.v1':raise ValueError('explicit original ad solve policy required')
    if args.command=='preflight':
        report=preflight(config,args.output,args.resume);print(json.dumps({k:report[k] for k in ('Ceres_started','current_ad_observations','CPU_GPU_UV_max_difference_px','original_W_factors','original_W_rows','genuine_Query_sources')},ensure_ascii=False));return 0
    print(solve(config,args.main_log,args.resume));return 0


if __name__=='__main__':raise SystemExit(main())
