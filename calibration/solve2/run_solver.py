"""Run the current, independently prepared S01 problem through new Ceres/PHR.

This entry point does not qualify historical data or import a retired solver.
The parent task supplies one explicit JSON numerical policy and main log path.
Numerical convergence always hands off to actual full-shot visual acceptance.
"""
from __future__ import annotations
import argparse
from dataclasses import replace,asdict
import hashlib,json,os,sys,time,traceback,uuid
from pathlib import Path
import numpy as np
from rinksolve.model import load_model_bundle,save_model_bundle
from rinksolve.preparation import load_prepared,refresh_current_target_snapshot
from rinksolve.native_backend import NativeBackend,SolveOptions
from rinksolve.constrained import ALOptions,ALState,CheckpointCallbackFailure
from rinksolve.optimizer import S01Optimizer,IMPLEMENTED_CONSTRAINTS
from rinksolve.runtime_state import model_store
from rinksolve.observation_edits import ObservationSelection,SelectionSnapshot,apply_selection
from rinksolve.store import atomic_json,checked_path,now_iso

BINDING_IMPLEMENTATIONS=IMPLEMENTED_CONSTRAINTS
TARGET_IDS={'numerical.intersection_budget','numerical.ad7_fixed_corner_budget','numerical.shared_budget_schedule'}


def semantic_bindings(ledger,stage):
    return [dict({k:item.get(k) for k in ('id','enabled','approval','expression','scope')},implementation=BINDING_IMPLEMENTATIONS[item['id']])
        for item in ledger['constraints'] if item['id'] in BINDING_IMPLEMENTATIONS and not (stage=='warm_geometry' and item['id'] in ('ad.object_model','numerical.ad7_fixed_corner_budget'))]


def semantic_hash(bindings):
    return hashlib.sha256(json.dumps([{k:v.get(k) for k in ('id','enabled','approval','expression','scope')} for v in bindings],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf8')).hexdigest()


STAGES={'warm_geometry':('ice_board','camera_scene'),'warm_artwork':('all','ad_layouts'),'joint':('all','all')}


def scope_prepared(prepared,model,stage):
    if stage not in STAGES:raise ValueError('explicit S01 numerical stage required')
    scope,mode=STAGES[stage]
    if scope=='all':return prepared
    active=prepared.selection_current_active_rows();curve_ids=prepared.bound_curve_ids(active)
    curves={c.id:c for c in model.scene.curves}
    admitted=np.array([curves[str(key)].label.startswith(('ice.','board.')) for key in curve_ids],dtype=bool)
    excluded=active[~admitted];excluded_set=set(map(int,excluded));groups=[]
    for index,group in enumerate(prepared.manifest['groups']):
        lo,hi=prepared.arrays['group_member_offsets'][index:index+2]
        members=prepared.arrays['group_member_rows'][lo:hi]
        if group.get('active_statistical') and any(int(row) in excluded_set for row in members):groups.append(group['id'])
    supplements=tuple(v['id'] for v in prepared.manifest.get('ad7_fixed_observations',[]))
    snapshot=SelectionSnapshot(prepared.manifest.get('selection_revision',0),excluded,excluded,supplements,tuple(groups),{})
    selected=apply_selection(prepared,snapshot)
    selected.manifest['numerical_data_scope']=scope
    selected.manifest['numerical_stage_id']=stage
    selected.manifest['source_admission_changed_by_stage']=False
    selected.manifest['stage_excluded_original_rows']=len(excluded)
    bits=np.array(prepared.arrays['exclusion_bits'],copy=True);bits[excluded]|=32768;bits.setflags(write=False)
    selected.arrays['exclusion_bits']=bits
    selected.manifest['reason_bits']['32768']='temporary numerical warm-geometry scope; source admission unchanged'
    return selected


def auxiliary_initialization(config,data):
    record=config.get('auxiliary_initialization')
    if not record:return None
    state=ALState.load(record['checkpoint'],data_root=data)
    path=Path(record['layout']).resolve()
    if not path.is_relative_to(data):raise ValueError('current auxiliary initialization layout must stay inside Data')
    with np.load(path,allow_pickle=False) as arrays:
        storage=arrays['native_block_ids_utf8'].tobytes();offsets=arrays['native_block_ids_offsets'];parameters=arrays['parameter_offsets']
        if parameters[-1]!=len(state.parameters):raise ValueError('warm-stage complete state/layout disagree')
        result={}
        for b in range(len(offsets)-1):
            identity=storage[offsets[b]:offsets[b+1]].decode('utf8')
            result[identity]=state.parameters[parameters[b]:parameters[b+1]].copy()
    return result


def rebind_numerical_strategy(state,engine,prior_runtime,*,reason,layout_path):
    """A declared numerical-policy update never changes the mathematical graph."""
    if not isinstance(reason,str) or not reason.strip():raise ValueError('explicit numerical strategy rebind reason required')
    prior=prior_runtime['run']['constraint_bindings'];current=engine.constraint_bindings
    keys=('id','enabled','approval','expression','scope')
    old={v['id']:{k:v.get(k) for k in keys} for v in prior};new={v['id']:{k:v.get(k) for k in keys} for v in current}
    changed={key for key in set(old)|set(new) if old.get(key)!=new.get(key)}
    if changed!={'numerical.s01_block_warm_start'}:raise ValueError('numerical strategy rebind may only change the declared warm-start auxiliary policy')
    a=json.loads(json.dumps(old['numerical.s01_block_warm_start']));b=json.loads(json.dumps(new['numerical.s01_block_warm_start']))
    a['expression'].pop('auxiliary_initialization',None);b['expression'].pop('auxiliary_initialization',None)
    if a!=b:raise ValueError('the declared update changes more than the current auxiliary numerical initialization')
    prior_identity=dict(state.identity);current_identity=dict(engine.problem.identity)
    prior_identity.pop('constraint_semantic_sha256',None);current_identity.pop('constraint_semantic_sha256',None)
    if prior_identity!=current_identity:raise ValueError('current source/model/scope identity differs; not a numerical strategy-only update')
    with np.load(layout_path,allow_pickle=False) as layout:
        p=engine.params
        for name in ('block_sizes','parameter_offsets','raw_observation_rows','local_block','station_column','row_column','alpha_column','scales'):
            if not np.array_equal(layout[name],np.asarray(getattr(p,name))):raise ValueError('changed current native layout/bounds: '+name)
        old_lower=layout['lower'];old_upper=layout['upper']
        differences=np.flatnonzero((old_lower!=p.lower)|(old_upper!=p.upper))
        bounds_repairs=[]
        if len(differences):
            declared=engine.ledger_by_id['numerical.s01_block_warm_start']['expression'].get('auxiliary_initialization',{}).get('block_coordinate_restoration',{}).get('native_finite_station_bounds_equivalence',{})
            if not declared.get('native_missing_bounds_repaired'):raise ValueError('changed bounds lack the explicit typed-domain equivalence migration')
            repairs={v['parameter_offset']:v for v in p.shared_station_bounds}
            for offset in differences:
                record=repairs.get(int(offset))
                if record is None or old_lower[offset]!=-np.inf or old_upper[offset]!=np.inf:
                    raise ValueError('only missing native bounds of existing shared finite stations may be repaired')
                if (p.lower[offset],p.upper[offset])!=(record['lower'],record['upper']) or not p.lower[offset]<=state.parameters[offset]<=p.upper[offset]:
                    raise ValueError('typed shared station domain or restored value differs')
                bounds_repairs.append(dict(record,before_native_lower=None,before_native_upper=None,
                    projector_domain_unchanged=True,parameters_changed_by_migration=False))
        old_ids=layout['native_block_ids_utf8'].tobytes();offsets=layout['native_block_ids_offsets']
        decoded=tuple(old_ids[offsets[i]:offsets[i+1]].decode('utf8') for i in range(len(offsets)-1))
        if decoded!=p.block_ids:raise ValueError('current native stable parameter IDs changed')
    if len(state.multipliers)!=engine.problem.constraints.layout.residual_count or len(state.parameters)!=len(engine.params.initial):raise ValueError('current mathematical dimensions changed')
    old_families=state.diagnostics.get('families',[]);families=engine.problem.constraints.families
    if len(old_families)!=len(families):raise ValueError('constraint families changed')
    for prior,family in zip(old_families,families):
        if (prior['id'],prior['sense'],prior['unit'])!=(family.id,family.sense,family.unit):raise ValueError('constraint family semantics/units changed')
        scale=np.asarray(family.normalization_scale)
        if prior['normalization_scale_min']!=float(scale.min()) or prior['normalization_scale_max']!=float(scale.max()):raise ValueError('constraint normalization changed')
    event={'kind':'numerical_strategy_metadata_rebind','reason':reason,'changed_constraint_ids':sorted(changed),'source_model_variable_ownership_objective_and_constraint_equations_unchanged':True,
        'parameters_multipliers_penalties_phase_preserved':True,'before_problem_digest':state.problem_digest,'after_problem_digest':engine.problem.digest,
        'existing_typed_station_domain_exposed_as_native_bounds':bounds_repairs}
    return replace(state,problem_digest=engine.problem.digest,identity=engine.problem.identity,history=state.history+(event,)),event


def migrate_statistical_level_set(state,engine,options,transition,*,source_checkpoint,layout_path):
    """Explicit new-objective state; retain the exact old primal/dual state."""
    if transition.get('from')!='feasibility' or transition.get('to')!='statistical_level_set' or state.identity.get('stage')!='feasibility' or engine.phase!='statistical_level_set':
        raise ValueError('unsupported explicit objective transition')
    reason=transition.get('reason')
    if not isinstance(reason,str) or not reason.strip():raise ValueError('objective transition needs an explicit reason')
    budget=float(transition['fixed_budget_squared'])
    if budget!=float(state.parameters[engine.params.budget_offset]) or budget!=engine.level_set_budget_sq:raise ValueError('level-set b must exactly retain the identified source checkpoint budget')
    if Path(transition['source_checkpoint']).resolve()!=Path(source_checkpoint).resolve():raise ValueError('objective transition source checkpoint differs from the actual complete state')
    old=dict(state.identity);new=dict(engine.problem.identity)
    for identity in (old,new):identity.pop('stage',None);identity.pop('constraint_semantic_sha256',None);identity.pop('fixed_budget_level_set_squared',None)
    if old!=new:raise ValueError('objective migration cannot change source/model/current data scope or variable ownership')
    with np.load(layout_path,allow_pickle=False) as layout:
        p=engine.params
        for name in ('block_sizes','parameter_offsets','raw_observation_rows','local_block','station_column','row_column','alpha_column','lower','upper','scales'):
            if not np.array_equal(layout[name],np.asarray(getattr(p,name))):raise ValueError('objective migration changed native layout/bounds/units: '+name)
        storage=layout['native_block_ids_utf8'].tobytes();offsets=layout['native_block_ids_offsets']
        if tuple(storage[offsets[i]:offsets[i+1]].decode('utf8') for i in range(len(offsets)-1))!=p.block_ids:raise ValueError('objective migration changed parameter stable IDs')
    families=engine.problem.constraints.families;old_families=state.diagnostics.get('families',[])
    if len(old_families)!=len(families) or state.multipliers.shape!=(engine.problem.constraints.layout.residual_count,):raise ValueError('objective migration changed constraint dimensions')
    for prior,family in zip(old_families,families):
        scale=np.asarray(family.normalization_scale)
        if (prior['id'],prior['sense'],prior['unit'],prior['normalization_scale_min'],prior['normalization_scale_max'])!=(family.id,family.sense,family.unit,float(scale.min()),float(scale.max())):raise ValueError('objective migration changed constraint family semantics/scale')
    diagnostics,_=engine.al._diagnostics(state.parameters,state.multipliers,state.penalties)
    event={'kind':'explicit_objective_transition','from':'feasibility','to':'statistical_level_set','reason':reason,'source_checkpoint':str(Path(source_checkpoint).resolve()),'fixed_budget_squared':budget,'b_is_not_proven_optimal':True,'old_problem_digest':state.problem_digest,'new_problem_digest':engine.problem.digest,'old_complete_state_retained':True,'parameters_multipliers_penalties_unchanged':True,'Ceres_accepted_step':False,'old_objective':'min b','new_objective':'complete original W and independent sigma statistics','constraints_source_model_and_all_DOF_unchanged':True,'requires_same_model_visual_geometry_review':True}
    return replace(state,problem_digest=engine.problem.digest,identity=engine.problem.identity,outer_iteration=0,phase='ready',policy=asdict(options),diagnostics=diagnostics,last_inner=None,history=state.history+(event,)),event


def apply_auxiliary_initialization(engine,state,*,row_tolerance_px,strategy='existing_locus_row'):
    if strategy=='feasibility_block_coordinate_v1':
        from rinksolve.auxiliary_initialization import feasibility_auxiliary_candidate
        vector,diagnostics,report=feasibility_auxiliary_candidate(engine,state,root_tolerance_px=row_tolerance_px)
        if not report['accepted_detached_candidate']:raise RuntimeError('auxiliary initialization retained the original complete state: '+report['error'])
        event={'kind':'explicit_feasibility_block_coordinate_initialization',
            'source_native_x_sha256':report['source_native_x_sha256'],'candidate_native_x_sha256':report['candidate_native_x_sha256'],
            'shared_FEATURE_unknown_physical_positions_may_change':True,
            'camera_scene_rig_curves_source_W_sigma_E_Query_unchanged':True,
            'lambda_rho_phase_and_accepted_count_preserved':True,'Ceres_accepted_step':False}
        return replace(state,parameters=vector,diagnostics=diagnostics,history=state.history+(event,)),report
    if strategy!='existing_locus_row':raise ValueError('unknown explicit auxiliary initialization strategy')
    before=state.parameters;cache=engine._compute(before,False);physics=cache['model'].state.digest();before_d=engine.current_diagnostics(before)
    special='raw:12:s01_original:281856:ice.R4:outer/segment/1/vertex/2'
    records={}
    if special in engine.params.observation_ids:
        index=engine.params.observation_ids.index(special)
        records['before']={'station':float(cache['stations'][index]),'uv':cache['forward']['uv'][index].tolist(),'raw_error_px':cache['error'][index].tolist()}
    vector,locus=engine._reseed_current_loci(before)
    after_loci=engine._compute(vector,False)
    if after_loci['model'].state.digest()!=physics:raise RuntimeError('locus reinitialization changed physical state')
    if records:records['after_locus']={'station':float(after_loci['stations'][index]),'uv':after_loci['forward']['uv'][index].tolist(),'raw_error_px':after_loci['error'][index].tolist()}
    vector,row_report=engine.precondition_formal_rows(vector,tolerance_px=row_tolerance_px)
    vector,joint_report=engine.precondition_joint_rows(vector,state.multipliers,state.penalties,tolerance_px=row_tolerance_px)
    diagnostics,_=engine.al._diagnostics(vector,state.multipliers,state.penalties)
    after_d=engine.current_diagnostics(vector)
    if after_d['state_sha256']!=physics:raise RuntimeError('auxiliary reinitialization changed physical state')
    event={'kind':'same_equation_auxiliary_initialization','physical_state_sha256':physics,'Ceres_accepted_step':False,'source_pixels_or_entity_associations_changed':False}
    updated=replace(state,parameters=vector,diagnostics=diagnostics,phase=state.phase,history=state.history+(event,))
    if not np.array_equal(updated.multipliers,state.multipliers) or not np.array_equal(updated.penalties,state.penalties):raise RuntimeError('auxiliary initialization altered AL dual/penalty state')
    report={'schema':'rinksolve.auxiliary_initialization.v1','physical_state_sha256_before':physics,'physical_state_sha256_after':after_d['state_sha256'],
        'multipliers_penalties_and_budget_unchanged':bool(np.array_equal(updated.multipliers,state.multipliers) and np.array_equal(updated.penalties,state.penalties) and vector[engine.params.budget_offset]==before[engine.params.budget_offset]),
        'source_model_objective_constraint_families_unchanged':True,'locus_reinitialization':locus,'formal_row_roots':row_report,'joint_station_row_roots':joint_report,
        'R4_exact_source_replay':records,'diagnostics_before':before_d,'diagnostics_after':after_d}
    return updated,report


def make_engine(config,model,prepared,backend,phase,**initial):
    numeric=config['numerics'];stage=config['numerical_stage'];prepared=scope_prepared(prepared,model,stage)
    return S01Optimizer(model,prepared,backend,constraints_path=config['constraints_path'],whitening_manifest=config['whitening_manifest'],
        num_threads=numeric['num_threads'],geometry_difference_step=numeric['geometry_difference_step'],axis_difference_step=numeric['axis_difference_step'],
        row_equality_scale_px=numeric['row_equality_scale_px'],query_equality_scale_px=numeric['query_equality_scale_px'],phase=phase,free_mode=STAGES[stage][1],level_set_budget_sq=(config.get('objective_transition') or {}).get('fixed_budget_squared'),**initial)


def publish_monitor_snapshot(path,payload,emit):
    """Retryable monitoring publication never invalidates a numerical step.

    This handles only the disposable current-run display. Model publications
    and complete primal/dual checkpoints keep their normal failure contract.
    """
    info=payload['run'];previous=info.get('monitor_publication',{})
    if previous.get('pending_retry'):info['monitor_publication']={**previous,'pending_retry':False,'recovered_at':now_iso()}
    try:atomic_json(path,payload)
    except OSError as exc:
        if getattr(exc,'winerror',None) not in (5,32,33):raise
        info['monitor_publication']={'pending_retry':True,'failures':previous.get('failures',0)+1,'last_error':str(exc),'destination':str(path),'last_failure_at':now_iso(),'complete_solver_checkpoint_errors_are_not_ignored':True}
        if not previous.get('pending_retry'):emit('monitor_publication_deferred',**info['monitor_publication'])
        return False
    if previous.get('pending_retry'):emit('monitor_publication_recovered',failures=previous.get('failures',0))
    return True


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--main-log',required=True)
    parser.add_argument('--resume',help='one complete new-solver AL state.json')
    parser.add_argument('--preflight',action='store_true',help='real r/J evaluation only; never mark a run running')
    args=parser.parse_args(argv)
    config=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    if config.get('schema')!='rinksolve.s01_run_policy.v1':raise ValueError('explicit current run-policy schema required')
    data=Path(config['data_root']).resolve();out=checked_path(config['output_root']);out.mkdir(parents=True,exist_ok=True)
    for path in (out,Path(args.main_log).resolve(),Path(config['current_run']).resolve()):
        if not path.is_relative_to(data):raise ValueError('run outputs and main log must stay inside Data')
    log=checked_path(args.main_log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf8',buffering=1) as stream:
        return run(args,config,data,out,stream)


def run(args,config,data,out,log):
    def emit(kind,**values):
        record={'time':now_iso(),'kind':kind,**values};line=json.dumps(record,ensure_ascii=False,allow_nan=False)
        print(line,flush=True);log.write(line+'\n');log.flush()
    work_stage=config['numerical_stage']
    if work_stage not in STAGES:raise ValueError('explicit current S01 stage selection required')
    backend=NativeBackend(config['native_manifest']);policy=config['numerics']
    inner=SolveOptions(**policy['inner_solver'])
    if inner.num_threads!=policy['num_threads']:raise ValueError('CPU thread policies disagree')
    options=ALOptions(**policy['augmented_lagrangian'],inner_options=inner,inner_parameter_metric=policy.get('inner_parameter_metric','ceres_default'))
    if options.inner_parameter_metric!='ceres_default' and not backend.has_unit_chart:raise ValueError('explicit unit metric requires compatible current native build')
    chunk=policy['outer_iterations_per_chunk']
    if type(chunk)is not int or chunk<=0:raise ValueError('explicit positive resumable chunk length required')
    ledger_path=Path(config['constraints_path']);read_ledger=lambda:json.loads(ledger_path.read_text(encoding='utf-8-sig'))
    prepared_base=load_prepared(config['prepared'],data_root=data)
    target_prepared=refresh_current_target_snapshot(prepared_base,ledger_path)
    selection=ObservationSelection(target_prepared,config['selection_path'],data_root=data)
    store=model_store(config['model_package'],data);current=store.load()
    prior_runtime=json.loads((out/'runtime.json').read_text(encoding='utf8')) if args.resume and (out/'runtime.json').exists() else None
    loaded=ALState.load(args.resume,data_root=data) if args.resume else None
    phase=loaded.identity['stage'] if loaded else 'feasibility'
    objective_transition=config.get('objective_transition')
    migrate_objective=bool(loaded and objective_transition and phase==objective_transition.get('from') and objective_transition.get('to')!=phase)
    if migrate_objective:phase=objective_transition['to']
    model=current
    if loaded:
        # Resume consumes this run's independently saved current input bundle,
        # never a retired solver package or historical qualification source.
        original=Path(config.get('resume_model_package',out/'input_model'))
        model=load_model_bundle(original)
        if (current.id,current.revision)!=(model.id,model.revision):raise ValueError('resume structure differs from current visible model')
    else:save_model_bundle(model,checked_path(out/'input_model'))
    snapshot=selection.snapshot();prepared=apply_selection(target_prepared,snapshot)
    bindings=semantic_bindings(read_ledger(),work_stage);binding_hash=semantic_hash(bindings)
    run_id=config.get('run_id') or 's01_'+uuid.uuid4().hex[:12]
    info={'id':run_id,'name':config.get('name','S01 current Ceres/PHR'),'status':'preparing','started_at':(prior_runtime or {}).get('run',{}).get('started_at',now_iso()),'updated_at':now_iso(),
        'iteration':(prior_runtime or {}).get('run',{}).get('iteration',0) if loaded else 0,'accepted_steps':(prior_runtime or {}).get('run',{}).get('accepted_steps',0) if loaded else 0,'cost':None,'main_log':str(Path(args.main_log).resolve()),'pid':os.getpid(),
        'model_id':model.id,'model_revision':model.revision,'parameter_revision':current.state.revision,'parameter_sha256':current.state.digest(),
        'stage':phase,'scope':work_stage,'numerical_stage':work_stage,'constraint_revision':read_ledger()['revision'],'constraint_bindings':json.loads(json.dumps(bindings)),
        'bound_constraint_ids':[v['id'] for v in bindings if v['enabled']], 'semantic_snapshot_hash':binding_hash,
        'selection_revision':snapshot.revision,'prepared':str(prepared_base.directory),'diagnostics':{},'visual_acceptance':False,'next_shot_allowed':False}
    info['inner_continuation']={'implementation':'single_ceres_solve' if options.continuous_inner else 'scheduled_chunks',
        'chart_lambda_rho_fixed_during_inner':True,'scheduling_restarts_native_solver':False if options.continuous_inner else True,
        'termination_units':'original_declared_parameter_scales','inner_stationarity_tolerance':options.inner_stationarity_tolerance,
        'process_recovery':'warm_restart; private Ceres LM rejection history is not serialized'}
    state=None;engine=None;expected_revision=current.state.revision;pending_change=None;transition=False
    current_status=Path(config['current_run'])
    def status(value,message,**more):
        info.update(status=value,updated_at=now_iso(),stage=phase,scope=work_stage,**more)
        return publish_monitor_snapshot(current_status,{'schema':'rinksolve.current_run.v1','run':info,'phase':phase,'message':message,
            'engine':{'available':True,'legacy_solver_enabled':False,'ceres_version':backend.ceres_version,'cuda_projection':True}},emit)
    domain_attempt=uuid.uuid4().hex[:12]
    def domain_rejected(event,point):
        folder=checked_path(out/'domain_trials'/domain_attempt);folder.mkdir(parents=True,exist_ok=True)
        trial=checked_path(folder/('trial_%08d.npy'%event['trial_index']))
        with trial.open('xb') as stream:np.save(stream,np.asarray(point,dtype=np.float64),allow_pickle=False)
        record={**event,'native_trial_file':str(trial),'native_layout':str(out/'layout/parameter_layout.json')}
        with checked_path(out/'domain_rejections.jsonl').open('a',encoding='utf8') as stream:
            stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
        info['last_domain_failure']=record;info['domain_rejection_log']=str(out/'domain_rejections.jsonl')
        emit('domain_rejected',**record)
    def metric_prepared(record,scales):
        folder=checked_path(out/'numerical_metrics');folder.mkdir(parents=True,exist_ok=True)
        path=checked_path(folder/('unit_'+record['source_native_parameters_sha256'][:12]+'_'+record['parameter_scales_sha256'][:12]+'.npy'))
        if not path.exists():
            with path.open('xb') as stream:np.save(stream,scales,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
        details={**record,'parameter_scales_file':str(path),'native_manifest':str(backend.manifest_path),'native_unit_chart_abi':1,'declared_units_file':str(out/'layout/parameter_layout.npz'),'declared_units_key':'scales'}
        atomic_json(path.with_suffix('.json'),details);info['numerical_metric']=details
        emit('inner_numerical_metric_prepared',**details)
        return {'parameter_scales_file':str(path)}
    def detect_changes():
        nonlocal pending_change
        live=store.load()
        if (live.id,live.revision)!=(model.id,model.revision):pending_change='visible_model_structure_changed';return True
        if live.state.revision!=expected_revision:pending_change='visible_parameters_edited';return True
        if selection.snapshot().revision!=snapshot.revision:pending_change='observation_selection_edited';return True
        now=semantic_bindings(read_ledger(),work_stage)
        if semantic_hash(now)!=binding_hash:
            prior={v['id']:{k:v.get(k) for k in ('enabled','approval','expression','scope')} for v in bindings};current_map={v['id']:{k:v.get(k) for k in ('enabled','approval','expression','scope')} for v in now}
            changed={identifier for identifier in set(prior)|set(current_map) if prior.get(identifier)!=current_map.get(identifier)}
            pending_change='engineering_target_edited' if changed<={'numerical.intersection_budget','numerical.ad7_fixed_corner_budget'} else 'constraint_semantics_edited'
            return True
        return False
    def publish_vector(vector,*,initialization=False):
        nonlocal expected_revision
        visible=engine.params.model_for(vector,revision=expected_revision+1,state_id='state/'+run_id+'/'+uuid.uuid4().hex)
        if visible.state.digest()==store.load().state.digest():return
        published=store.publish_solver_state(visible,expected_revision=expected_revision,diagnostics={'run_id':run_id,'stage':work_stage+'/'+phase,'accepted_steps':info['accepted_steps'],'initialization_repairs':engine.initialization_repairs if initialization else None},initialization=initialization)
        expected_revision=published['revision'];info.update(parameter_revision=expected_revision,parameter_sha256=published['sha256'],visible_parameter_sha256=published['sha256'],working_initialization_sha256=published['sha256'])
    def checkpoint(updated,event):
        nonlocal state
        state=updated;manifest=state.save(out/'checkpoint'/phase,data_root=data)
        diagnostics={**state.diagnostics,**engine.current_diagnostics(state.parameters)}
        info['visible_parameter_sha256']=store.load().state.digest();info['working_initialization_sha256']=diagnostics['state_sha256']
        info['diagnostics']=diagnostics;info['checkpoint']=str(manifest);info['outer_iteration']=state.outer_iteration
        conflict=out/'entity_conflicts.json';engine.write_conflict_report(state.parameters,conflict);info['conflict_report']=str(conflict)
        atomic_json(out/'runtime.json',{'schema':'rinksolve.runner_resume.v1','run':info,'expected_visible_revision':expected_revision,'input_model':str(out/'input_model')})
        status('running' if not args.preflight else 'preflight',event['kind']);emit('checkpoint',event=event,stage=phase,budget_sq=float(state.parameters[engine.params.budget_offset]),diagnostics=state.diagnostics)
    def on_iteration(event,vector):
        nonlocal transition
        info.update(iteration=info['iteration']+1,cost=float(event['cost']),last_inner_iteration=event)
        if detect_changes():status('rebinding',pending_change);return 1
        if event['step_successful'] and not event['initial_state']:
            info['accepted_steps']+=1;publish_vector(vector)
            diagnostic=engine.current_diagnostics(vector)
            constraint_metrics,_=engine.al._diagnostics(vector,state.multipliers,state.penalties,jacobian=False)
            constraint_metrics['normalized_projected_stationarity_inf']=None
            if event.get('normalized_augmented_stationarity_inf') is not None:
                constraint_metrics['normalized_augmented_stationarity_inf']=event['normalized_augmented_stationarity_inf']
                constraint_metrics['inner_stationarity_tolerance']=options.inner_stationarity_tolerance
            info['diagnostics']={**diagnostic,**constraint_metrics}
            accepted_state=replace(state,parameters=vector,phase='inner_incomplete',diagnostics=info['diagnostics'],
                last_inner={'partial_inner_iteration':event,'actual_accepted_steps':info['accepted_steps'],'multipliers_not_updated':True,
                    'inner_continuation':info['inner_continuation'],'fixed_inner_numerical_metric':info.get('numerical_metric')})
            info['checkpoint']=str(accepted_state.save(out/'checkpoint'/phase,data_root=data))
            emit('accepted_step',iteration=info['iteration'],cost=event['cost'],stage=phase,budget_sq=float(vector[engine.params.budget_offset]),parameter_revision=expected_revision)
            if phase=='feasibility' and vector[engine.params.budget_offset]<=engine.target_budget_sq:
                c=np.empty(engine.problem.constraints.layout.residual_count)
                if engine.evaluate_constraints(vector,c,None,False)is not False:
                    q=c/engine.problem.constraints.scales;equality=engine.problem.constraints.equality
                    feasibility=max(np.max(np.abs(q[equality]),initial=0),np.max(np.maximum(q[~equality],0),initial=0))
                    if feasibility<=options.feasibility_tolerance:transition=True
        status('running','accepted trust-region state' if event['step_successful'] else 'trust-region trial rejected')
        return 1 if transition else 0
    try:
        emit('preparation_started',model_id=model.id,model_revision=model.revision,prepared=str(prepared_base.directory),selection_revision=snapshot.revision,main_log=info['main_log'])
        if not args.preflight:status('preparing','assembling current model and sparse factors')
        engine=make_engine(config,model,prepared,backend,phase,domain_event_callback=domain_rejected,metric_event_callback=metric_prepared,initial_vector=loaded.parameters if loaded else None,initialize_loci=loaded is None,initial_values_by_block=auxiliary_initialization(config,data))
        families={'ice':0,'board':0,'advertisement':0,'shared_ice_intersection':0,'fixed_artwork_corner':0}
        for row,curve_id in enumerate(engine.params.curve_ids):
            kind=int(engine.params.kinds[row]);curve=engine.params.curve_lookup[str(curve_id)]
            family='shared_ice_intersection' if kind==3 else 'fixed_artwork_corner' if kind==4 else 'ice' if curve.label.startswith('ice.') else 'board' if curve.label.startswith('board.') else 'advertisement'
            families[family]+=1
        info['active_source_families']=families
        info['numerical_metric']={'implementation':options.inner_parameter_metric,'native_manifest':str(backend.manifest_path),'native_unit_chart_abi':1 if backend.has_unit_chart else None,'ambient_bounds':True,'reduced_degrees_of_freedom':0,'jacobi_scaling':False if options.inner_parameter_metric!='ceres_default' else True,'min_lm_diagonal':1 if options.inner_parameter_metric!='ceres_default' else None,'max_lm_diagonal':1 if options.inner_parameter_metric!='ceres_default' else None,'parameter_scales_file':str(out/'layout/parameter_layout.npz'),'parameter_scales_key':'scales','parameter_scales_sha256':hashlib.sha256(engine.problem.parameter_scales.tobytes()).hexdigest(),'scale_source':'current declared native numerical units, no observation sigma or physical prior'}
        info['active_source_counts']={'original':engine.params.original_count,'supplemental':engine.params.count-engine.params.original_count,'total':engine.params.count,
            'statistical_factors':len(engine.stat_layout.factor_ids),'statistical_residuals':engine.stat_layout.residual_count,'engineering_budget':len(engine.budget_rows),'query_equalities':2*len(engine.profile_rows)}
        for binding in info['constraint_bindings']:
            if binding['id']=='numerical.s01_block_warm_start':binding['implementation']+='; current stage '+work_stage
        if loaded is not None:state=loaded
        else:
            vector=engine.params.initial
            if 'row_preconditioner_tolerance_px' in policy:
                vector,row_report=engine.precondition_formal_rows(vector,tolerance_px=policy['row_preconditioner_tolerance_px'])
                if phase=='feasibility':vector[engine.params.budget_offset]=max(engine.target_budget_sq,float(engine._budget_quantities(engine._compute(vector,False))[0].max()))
                atomic_json(out/'initial_row_preconditioner.json',row_report)
            state=engine.al.initialize(vector,options)
        if loaded is not None and engine.params.model_for(loaded.parameters).state.digest()!=current.state.digest():
            prior_visible=(prior_runtime or {}).get('run',{})
            expected=(prior_runtime or {}).get('expected_visible_revision')
            if not config.get('continuation_policy_update_reason') or expected!=current.state.revision or prior_visible.get('visible_parameter_sha256',prior_visible.get('parameter_sha256'))!=current.state.digest():
                raise ValueError('current physical parameters changed outside the complete resume checkpoint; preserve user edit')
            info['continuation_reconciles_unpublished_last_accepted_state']=True
        if migrate_objective:
            before=state.save(out/'objective_history'/'feasibility_before_statistical_level_set',data_root=data)
            state,objective_event=migrate_statistical_level_set(state,engine,options,objective_transition,source_checkpoint=args.resume,layout_path=out/'layout/parameter_layout.npz')
            objective_event['old_complete_checkpoint']=str(before)
            atomic_json(out/'objective_transition.json',objective_event);info['objective_transition']=objective_event;info['objective_kind']='complete_original_statistics_at_fixed_budget_level_set'
            checkpoint(state,{'kind':'explicit_objective_transition_ready','old_complete_checkpoint':str(before),'fixed_budget_squared':engine.level_set_budget_sq})
            emit('explicit_objective_transition',event=objective_event)
        elif state.problem_digest!=engine.problem.digest:
            prior_numeric_checkpoint=state.save(out/'numerical_policy_history'/state.problem_digest[:12],data_root=data)
            state,rebind_event=rebind_numerical_strategy(state,engine,prior_runtime,reason=config.get('continuation_policy_update_reason'),layout_path=out/'layout/parameter_layout.npz')
            rebind_event['prior_complete_checkpoint']=str(prior_numeric_checkpoint)
            emit('numerical_strategy_metadata_rebind',event=rebind_event)
        engine.params.save_layout(out/'layout')
        if loaded is not None and config.get('apply_auxiliary_initialization_on_resume'):
            state.save(out/'before_auxiliary_update'/phase,data_root=data)
            state,initialization_report=apply_auxiliary_initialization(engine,state,row_tolerance_px=policy['row_preconditioner_tolerance_px'],strategy=config.get('auxiliary_initialization_strategy','existing_locus_row'))
            report_path=out/'auxiliary_initialization.json';atomic_json(report_path,initialization_report);info['auxiliary_initialization_report']=str(report_path)
            checkpoint(state,{'kind':'same_equation_auxiliary_initialization_ready'})
        if state.policy!=asdict(options):
            reason=config.get('continuation_policy_update_reason')
            if not reason:raise ValueError('current numerical policy update requires its explicit reason')
            event={'kind':'numerical_policy_changed','reason':reason,'before':state.policy,'after':asdict(options)}
            state=replace(state,policy=asdict(options),history=state.history+(event,))
            checkpoint(state,event)
        info['diagnostics']={**state.diagnostics,**engine.current_diagnostics(state.parameters)}
        if phase=='statistical_level_set':
            info['objective_kind']='complete_original_statistics_at_fixed_budget_level_set';info['fixed_budget_level_set_squared']=engine.level_set_budget_sq;info['is_proven_optimal_budget']=False
        if args.preflight:
            if options.inner_parameter_metric=='full_uv_unit_tangent_identity_lm':engine.full_uv_metric_scales(state.parameters)
            checkpoint(state,{'kind':'real_current_r_J_preflight'});status('preflight_ready','current residuals and sparse Jacobians evaluated; no solver started')
            emit('preflight_ready',active_observations=engine.params.count,budget_sq=float(state.parameters[engine.params.budget_offset]),checkpoint=info['checkpoint']);return 0
        publish_vector(state.parameters,initialization=True)
        if args.resume:
            event={'kind':'ceres_warm_restart_from_complete_checkpoint','source_checkpoint':str(Path(args.resume).resolve()),
                'parameters_lambda_rho_phase_preserved':True,'Ceres_private_LM_state_restored':False,
                'continuation_within_new_inner':'single_ceres_solve' if options.continuous_inner else 'scheduled_chunks'}
            state=replace(state,history=state.history+(event,));emit('ceres_warm_restart',event=event)
        status('running','current Ceres/PHR solve started');emit('solver_started',stage=phase)
        while True:
            result=engine.al.run(state,options,outer_iterations=chunk,on_checkpoint=checkpoint,on_inner_iteration=on_iteration,policy_change_reason=config.get('continuation_policy_update_reason'))
            state=result.state
            if pending_change:
                checkpoint(state,{'kind':'input_change_saved','reason':pending_change})
                if pending_change in ('visible_model_structure_changed','constraint_semantics_edited'):
                    status('current_input_update_required','saved complete state; rebuild only changed model/constraint families before continuation')
                    emit('current_input_update_required',reason=pending_change);return 4
                warm={identifier:state.parameters[engine.params.parameter_offsets[b]:engine.params.parameter_offsets[b+1]].copy()
                    for b,identifier in enumerate(engine.params.block_ids) if engine.params.block_owners[b]['kind']=='current_source_nuisance_group'}
                engine.close();engine=None;model=store.load();expected_revision=model.state.revision
                target_prepared=refresh_current_target_snapshot(prepared_base,ledger_path)
                selection=ObservationSelection(target_prepared,config['selection_path'],data_root=data);snapshot=selection.snapshot()
                bindings=semantic_bindings(read_ledger(),work_stage);binding_hash=semantic_hash(bindings)
                info.update(constraint_bindings=json.loads(json.dumps(bindings)),bound_constraint_ids=[v['id'] for v in bindings if v['enabled']],semantic_snapshot_hash=binding_hash,constraint_revision=read_ledger()['revision'])
                prepared=apply_selection(target_prepared,snapshot);info['selection_revision']=snapshot.revision
                # Physical edits use the user's current state. Auxiliary owners
                # with unchanged IDs retain their last accepted values.
                engine=make_engine(config,model,prepared,backend,phase,domain_event_callback=domain_rejected,metric_event_callback=metric_prepared,initial_values_by_block=warm,initialize_loci=False)
                save_model_bundle(model,out/'input_model');engine.params.save_layout(out/'layout')
                state=engine.al.initialize(engine.params.initial,options);pending_change=None
                checkpoint(state,{'kind':'current_graph_rebuilt','source_requalification_replayed':False});continue
            if transition or (phase=='feasibility' and float(state.parameters[engine.params.budget_offset])<=engine.target_budget_sq and state.diagnostics['normalized_feasibility_inf']<=options.feasibility_tolerance):
                before=state.parameters;phase='statistics';vector=engine.set_phase(phase,before);state=engine.al.initialize(vector,options);transition=False
                checkpoint(state,{'kind':'shared_budget_phase_transition','b_fixed_to':engine.target_budget_sq,'new_multiplier_state':'zero for new fixed-b objective; previous complete phase checkpoint retained'});continue
            if result.numerically_converged:
                if phase=='feasibility':
                    status('entity_review_required','minimum shared budget remains above one; inspect entity/source diagnostics')
                    emit('feasibility_target_unmet',budget_sq=float(state.parameters[engine.params.budget_offset]));return 3
                publish_vector(state.parameters)
                waiting={'warm_geometry':'awaiting_geometry_review','warm_artwork':'awaiting_artwork_review','joint':'awaiting_visual_acceptance'}[work_stage]
                status(waiting,'stage numerical feasibility/KKT met; root must inspect the same model on actual S01 exposures before advancing')
                emit('numeric_stage_candidate_ready_visual_review_required',numerical_stage=work_stage);return 0
            if result.status in ('outer_chunk_exhausted','inner_incomplete'):
                accepted=state.last_inner.get('actual_accepted_steps',0) if state.last_inner else 0
                if result.status=='inner_incomplete' and not accepted:
                    status('numerical_repair_required','inner stationarity incomplete with no accepted step; complete state and conflicts retained')
                    emit('numerical_stagnation_requires_repair',last_inner=state.last_inner);return 5
                emit('continuing_resumable_chunk',status=result.status,outer_iteration=state.outer_iteration);continue
            status('entity_review_required' if result.status=='penalty_limit_requires_review' else 'recoverable_failure',result.status)
            emit('recoverable_solver_state',status=result.status,last_inner=state.last_inner);return 3
    except BaseException as exc:
        if isinstance(exc,CheckpointCallbackFailure):state=exc.state
        if state is not None:
            try:state.save(out/'checkpoint'/phase,data_root=data)
            except BaseException as checkpoint_error:emit('checkpoint_save_error',error=str(checkpoint_error))
        info['failure']={'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()}
        status('interrupted_recoverable' if isinstance(exc,KeyboardInterrupt) else 'recoverable_failure',str(exc))
        emit('runner_failure',**info['failure']);return 130 if isinstance(exc,KeyboardInterrupt) else 2
    finally:
        if engine is not None:engine.close()


if __name__=='__main__':raise SystemExit(main())
