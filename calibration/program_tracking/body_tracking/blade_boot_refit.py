"""Bounded boot-image repair on one current continuous S05 body state.

Consumes an independent current case; preserves frozen shoes/camera, finite
ice and existing action factors. No prepare/inference/historical pose replay.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,copy,json,os,sys,time
import numpy as np
from scipy.sparse import csr_matrix,vstack
from rinkspace.path_policy import guard_project_write
from .rebuild_prepare import load,digest
from .fullbody_round import write_json
from .blade_current_reference import restore_current_reference
from .absolute_motion_state import absolute_reference
from .body_execution import BodyExecution,BodyExecutionConfig
from .spin_continuous_refit import ContinuousJoint
from .spin_continuous_continue import _configure_local_controls
from .spin_grasp_factors import SpinGraspFactors
from .continuous_motion_factors import ActualMotionFactors
from .continuous_physics import ContinuousMassGeometry,ConfirmedFlightFactors
from .boot_axis_factors import BootAxisFactors
from .support_image import SupportImageFactors
from .skate_integration import source_evidence
from .contact_feasibility import ContactAL
from .skate_run_policy import RunPolicy
from .exmr_integration import CASE_FIELD,case_identity,save_packet,PROGRAMME


def _json(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def _link(src,dst):
    dst=_checked_path(dst)
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():raise FileExistsError(dst)
    os.link(src,dst)



def _checked_path(path,project=None):
    return guard_project_write(path,project_root=project)


def _preserve_driver_before_runner(path):
    path=_checked_path(path)
    payload=Path(__file__).read_bytes()
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with path.open('xb') as stream:stream.write(payload)
    except FileExistsError:
        if not path.is_file() or path.read_bytes()!=payload:
            raise FileExistsError('Preserve inconsistent existing driver backup: '+str(path))
    return dict(path=str(path),sha256=digest(path))


def _preflight_targets(project,stage,parameter,document,output,parameter_stem,driver_backup_path=None):
    if not parameter_stem.replace('_','').isalnum():raise ValueError('Expected a plain parameter basename')
    archive=_checked_path(project/PROGRAMME/(parameter_stem+'.npz'),project)
    output=_checked_path(output,project)
    configured=driver_backup_path or os.environ.get('RINKSPACE_BODY_DRIVER_BACKUP')
    backup=_checked_path(configured or output.parent/'implementation_before/driver.py',project)
    if output.exists():raise FileExistsError('One bounded run; preserve existing output: '+str(output))
    if archive.exists():raise FileExistsError('Preserve final parameter path: '+str(archive))
    if not stage.is_dir() or not parameter.is_file() or not document.is_file():
        raise FileNotFoundError('Current stage, parameter and source document must exist before creating a run')
    if output.is_relative_to(stage) or stage.is_relative_to(output):
        raise ValueError('New output must be separate from the parent run')
    protected={Path(__file__).resolve(),parameter,document,archive}
    if backup in protected or backup==output or backup.is_relative_to(output) or backup.is_relative_to(stage):
        raise ValueError('Driver backup must not cover a source, candidate or run target: '+str(backup))
    if backup.exists() and (not backup.is_file() or backup.read_bytes()!=Path(__file__).read_bytes()):
        raise FileExistsError('Preserve inconsistent existing driver backup: '+str(backup))
    return archive,backup


def _seconds(value,name):
    value=np.asarray(value,float)
    if value.shape!=(2,) or not np.isfinite(value).all() or value[0]>=value[1]:
        raise ValueError('Ordered finite '+name+' required')
    return tuple(map(float,value))


def _snapshot_sources(project,output,camera_info,driver_backup):
    # Exact local code, never model weights or environment/package copies.
    source_root=_checked_path(output/'source_snapshot',project);source_root.mkdir()
    paths=set(Path(__file__).resolve().parent.glob('*.py'))
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            path=Path(name).resolve()
            if path.suffix=='.py' and path.is_relative_to(project) and path.is_file():paths.add(path)
    entries=[]
    for src in sorted(paths):
        relative=src.relative_to(project);dst=_checked_path(source_root/relative,project)
        dst.parent.mkdir(parents=True,exist_ok=True)
        payload=src.read_bytes()
        with dst.open('xb') as stream:stream.write(payload)
        entries.append(dict(project_relative_path=relative.as_posix(),snapshot_relative_path=dst.relative_to(output).as_posix(),sha256=digest(dst),bytes=len(payload)))
    from .body_foot_cuda_bridge import default_library
    native_paths={Path(p).resolve() for p in (os.environ.get('RINKSPACE_BODY_NATIVE_LIBRARY'),camera_info.get('native_library'),str(default_library())) if p}
    native=[dict(path=str(p),sha256=digest(p) if p.is_file() else None,present=p.is_file(),binary_copied=False) for p in sorted(native_paths)]
    manifest=source_root/'manifest.json'
    write_json(manifest,dict(scope='body_tracking/*.py plus currently imported project Python source; native library identities only',before_first_factor_or_joint_evaluate=True,sources=entries,native_libraries=native,driver_before_runner=driver_backup,models_copied=False,environment_copied=False))
    return dict(path=str(manifest),sha256=digest(manifest),source_files=len(entries),native_libraries=native,driver_before_runner=driver_backup)


def _select_body_columns(problem,local,body_columns):
    indices=np.asarray(problem.active_parameter_indices,dtype=int)
    if np.any(indices<17) or np.any(indices>=problem.size):
        raise ValueError('Bounded body controls must not activate shared parameters')
    columns=list(range(69)) if body_columns is None else list(body_columns)
    if not columns or any(isinstance(c,(bool,np.bool_)) or not isinstance(c,(int,np.integer)) or not 0<=c<69 for c in columns):
        raise ValueError('body_columns must contain body control column integers in [0,68]')
    columns=sorted(set(map(int,columns)))
    if body_columns is not None:
        indices=indices[np.isin((indices-17)%69,columns)]
        if not len(indices):raise ValueError('No requested body columns in the existing active control selection')
        problem.active_parameter_indices=indices
    problem.checkpoint_extra.update(local_active_parameter_indices=indices.copy(),local_body_control_columns=np.asarray(columns,dtype=int))
    local=dict(local or {},active_variables=len(indices),body_control_columns=columns,body_state_columns_per_control=69,shared_active_variables=0)
    return local


def _directional_metrics(numeric,exact):
    numeric=np.asarray(numeric);exact=np.asarray(exact);difference=numeric-exact
    if not np.isfinite(numeric).all() or not np.isfinite(exact).all():
        raise ValueError('Nonfinite local factor directional derivative')
    numeric_norm=float(np.linalg.norm(numeric));exact_norm=float(np.linalg.norm(exact))
    exercised=bool(len(numeric) and (numeric_norm>0. or exact_norm>0.))
    return dict(status='checked' if exercised else 'unexercised',exercised=exercised,
        relative_directional_error=float(np.linalg.norm(difference)/max(numeric_norm,1e-12)) if exercised else None,
        maximum_absolute=float(np.max(abs(difference),initial=0.)),residual_rows=len(numeric),
        numeric_directional_norm=numeric_norm,exact_directional_norm=exact_norm)


def _factor_directional_check(factor,x,d,required_groups=()):
    value=factor.evaluate(x[17:],True);h=1e-6
    numeric=(factor.evaluate(x[17:]+h*d[17:])['residual']-factor.evaluate(x[17:]-h*d[17:])['residual'])/(2*h)
    exact=value['jac']@d;report=_directional_metrics(numeric,exact)
    blocks=getattr(factor,'block_indices',{})
    if any(name not in blocks for name in required_groups):raise ValueError('Missing runner residual block identities')
    report['groups']={}
    for name,indices in blocks.items():
        indices=np.asarray(indices,dtype=int)
        if indices.ndim!=1 or np.any(indices<0) or np.any(indices>=len(numeric)):
            raise ValueError('Invalid residual block indices: '+name)
        report['groups'][name]=_directional_metrics(numeric[indices],exact[indices])
    if hasattr(factor,'last'):factor.last=value
    return value,report


def _derivative_failed(report,gate=3e-3):
    return any(row['exercised'] and row['relative_directional_error']>gate
        for row in [report,*report.get('groups',{}).values()])


def _source_quality_groups(metrics,core_seconds,context_seconds):
    result={}
    for scope,seconds in [('core',core_seconds),('context',context_seconds)]:
        groups={}
        for row in metrics.get('observations',[]):
            if not seconds[0]-1e-9<=row['pts']/12800.<=seconds[1]+1e-9:continue
            key=(row['foot'],row['kind'],row.get('source_semantic',row['kind']),row['role'])
            groups.setdefault(key,[]).append(row)
        result[scope]=[]
        for (foot,kind,semantic,role),rows in sorted(groups.items()):
            item=dict(foot=foot,kind=kind,source_semantic=semantic,role=role,rows=len(rows),pts=[r['pts'] for r in rows])
            for name in ['centre_error_px','outside_region_px','actual_geometry_centre_distance_px','actual_geometry_outside_region_px']:
                values=np.asarray([r[name] for r in rows],float)
                item[name]=dict(median=float(np.median(values)),p95=float(np.quantile(values,.95)),maximum=float(values.max()))
            result[scope].append(item)
    return result


def _action_quality(actions):
    parts=dict(actions.last['parts']);motion=dict(actions.motion.last['parts'])
    runner=actions.runner_support
    return dict(action_parts=parts,pose_prior_cost=motion['finite_network_pose_prior'],motion_parts=motion,
        pose_prior_scope='full native source grid; finite archived-initial orientation prior, not a source-image measurement',
        pose_quality_monitor_enabled=False,pose_prior_report_only=True,
        runner_support_mode_costs=None if runner is None else copy.deepcopy(runner.last['metrics'].get('runner_support_mode_costs')),
        runner_support_metrics=None if runner is None else copy.deepcopy(runner.last['metrics']))


def _runner_switch_check(actions,x,on_value,full_value):
    # One same-state factor switch; no full-model finite differences.
    off=actions.evaluate(x[17:],runner_support_enabled=False)
    off_cost=float(off['residual']@off['residual']/2);on_cost=float(on_value['residual']@on_value['residual']/2)
    added_rows=len(on_value['residual'])-len(off['residual'])
    return dict(same_state=True,scope='existing action factors with runner support toggled',disabled_residual_rows=len(off['residual']),enabled_residual_rows=len(on_value['residual']),added_residual_rows=added_rows,disabled_cost=off_cost,enabled_cost=on_cost,added_cost=on_cost-off_cost,full_joint_enabled_residual_rows=len(full_value['residual']),full_joint_disabled_residual_rows=len(full_value['residual'])-added_rows,full_joint_enabled_cost=float(full_value['residual']@full_value['residual']/2),full_joint_disabled_cost=float(full_value['residual']@full_value['residual']/2)-(on_cost-off_cost))


def _prior(initial,reference):
    # The original action prior consumes only the exact source midpoint grid.
    # Recover its existing P/Q/local_R from the current archived initial packet.
    times=reference.times.copy();values={k:initial[k].copy() for k in ('P','Q','local_R')}
    def at(t):
        t=np.asarray(t);indices=np.searchsorted(times,t)
        if np.any(indices>=len(times)) or not np.allclose(times[indices],t,rtol=0,atol=1e-10):
            raise ValueError('Original motion prior requested outside its archived exact PTS')
        return {k:v[indices] for k,v in values.items()}
    return SimpleNamespace(at=at)


def _mass(reference,stage):
    path=stage/'mass_geometry.npz'
    if not path.exists():path=stage/'centroidal_physics.npz'
    with np.load(path,allow_pickle=False) as z:proxy=json.loads(str(z['model_identity_json']))
    mass=object.__new__(ContinuousMassGeometry);mass.ref=reference;mass.binding=reference.binding;mass.proxy=proxy
    mass.count=int(proxy['body_segment_count']);mass.ids=np.asarray(proxy['segment_native_joint_ids'],int)[:mass.count]
    mass.offsets=np.asarray(proxy['segment_com_rest_local_m']);mass.mass=np.asarray(proxy['segment_mass_fractions']);mass.inertia=np.asarray(proxy['segment_unit_mass_inertia_local_m2'])
    return mass,path


class ExistingActionsWithBootAxes:
    def __init__(self,problem,prior,mass,runtime,runner_support_policy=None):
        self.grasp=SpinGraspFactors(problem,problem.reference.grasp_relations)
        self.motion=ActualMotionFactors(problem,prior)
        self.flight=ConfirmedFlightFactors(problem,mass)
        self.axes=BootAxisFactors(problem,runtime,runtime['boot_axis_observations'])
        self.mass=mass;self.runner_support=None;self.last=None
        if runner_support_policy is not None:
            from .runner_support_factors import RunnerSupportFactors
            self.runner_support=RunnerSupportFactors(problem,runner_support_policy)
    def evaluate(self,x,jacobian=False,*,runner_support_enabled=True):
        factors=[self.grasp,self.motion,self.flight,self.axes]
        if self.runner_support is not None and runner_support_enabled:factors.append(self.runner_support)
        outputs=[f.evaluate(x,jacobian) for f in factors]
        result=dict(residual=np.concatenate([v['residual'] for v in outputs]),jac=vstack([v['jac'] for v in outputs],format='csr') if jacobian else None,parts={},metrics={})
        for v in outputs:result['parts'].update(v['parts']);result['metrics'].update(v['metrics'])
        result['factor_row_counts']=[(type(f).__name__,len(v['residual'])) for f,v in zip(factors,outputs)]
        self.last=result
        return result


def run(project,stage,parameter,document,output,core_seconds,context_seconds,*,max_accepted,wall_seconds,parameter_stem='spin_boot_axis_entry_1105',runner_support_policy=None,body_columns=None,driver_backup_path=None,active_control_seconds=None,image_safe_trials=False):
    project,stage,parameter,document,output=map(lambda p:Path(p).resolve(),(project,stage,parameter,document,output))
    core_seconds=_seconds(core_seconds,'core seconds');context_seconds=_seconds(context_seconds,'context seconds')
    active_control_seconds=_seconds(core_seconds if active_control_seconds is None else active_control_seconds,'active control seconds')
    if any(s[0]<context_seconds[0] or s[1]>context_seconds[1] for s in (core_seconds,active_control_seconds)):
        raise ValueError('Context must contain both the review core and active control support')
    if not np.isfinite(wall_seconds) or not 0.<wall_seconds<=600.:raise ValueError('Explicit candidate wall budget must be in (0,600] seconds')
    if isinstance(max_accepted,bool) or not isinstance(max_accepted,(int,np.integer)) or max_accepted<0:
        raise ValueError('Explicit nonnegative accepted-step budget required')
    archive,backup_path=_preflight_targets(project,stage,parameter,document,output,parameter_stem,driver_backup_path)
    if runner_support_policy is not None and not Path(runner_support_policy).is_file():raise FileNotFoundError(runner_support_policy)
    start=time.perf_counter();driver_backup=_preserve_driver_before_runner(backup_path)
    output.mkdir(parents=True);log_path=output/'fit_progress.jsonl'
    runner_policy=None;runner_policy_provenance=None
    if runner_support_policy is not None:
        runner_support_policy=Path(runner_support_policy).resolve();policy_copy=_checked_path(output/'runner_support_policy.json')
        policy_copy.write_bytes(runner_support_policy.read_bytes());runner_policy=_json(policy_copy)
        runner_policy_provenance=dict(path=str(runner_support_policy),sha256=digest(policy_copy),copied_path=str(policy_copy),copied_sha256=digest(policy_copy))
    print(json.dumps(dict(main_log=str(log_path),max_accepted=max_accepted,budget_seconds=wall_seconds)),flush=True)
    case=output/'common/exmr_hand_nofoot';case.mkdir(parents=True)
    oldcase=stage/'common/exmr_hand_nofoot'
    for name in ['input.npz','compact_smplx.npz','initial.npz','observations.npz','export_source.npz']:_link(oldcase/name,case/name)
    for name in ['grasp_relations.json','palm_grip_geometry.json','exmr_body_hand_nofoot.npz']:_link(stage/name,output/name)
    source=_json(document);write_json(output/'support_and_free_shoe_regions.json',source)
    write_json(output/'source_evidence.json',source)
    ref,camera,info,saved=restore_current_reference(project,stage,parameter,seed=True)
    snapshot=_snapshot_sources(project,output,info,driver_backup)
    ref.root=case;ref.export_source_path=case/'export_source.npz'
    identity=case_identity(project,output.parents[1],output,'exmr_hand_nofoot',info,initial=case/'initial.npz')
    ref.fixed_checkpoint_identity[CASE_FIELD]=np.array(json.dumps(identity,sort_keys=True))
    ref.footwear_observations=dict(visible_rows=[],blade_rows=[],landmark_rows=[],provenance=dict(source_document_sha256=digest(output/'support_and_free_shoe_regions.json')))
    execution=BodyExecution(BodyExecutionConfig(workers=12,blas_threads=1,local_derivatives='analytic',compatibility_backend='cuda_resident'))
    execution.attach(ref);mass,mass_path=_mass(ref,stage);prior=_prior(load(case/'initial.npz'),ref)
    np.savez_compressed(output/'mass_geometry.npz',model_identity_json=np.array(json.dumps(mass.proxy,sort_keys=True)),source_model_file_sha256=np.array(digest(mass_path)),rest_geometry_only=np.array(True))
    with log_path.open('a',encoding='utf-8',buffering=1) as log:
        def emit(event,**kw):log.write(json.dumps(dict(event=event,elapsed=time.perf_counter()-start,**kw))+'\n')
        with execution.limits():
            supports=source_evidence(case,shot_id='s05',source_pts=ref.p['pts'],records=source['supports'])
            problem=ContinuousJoint(ref,camera,case/'observations.npz',tuple(context_seconds),supports,output/'solve',log,'absolute_motion_v1',optimize_shared=False,ice_policy=ref.ice_policy,record_acceptance_before_jacobian=True)
            problem.image_safe_trials=bool(image_safe_trials)
            class NoOpening:
                def evaluate(self,x,jacobian=False):return dict(residual=np.empty(0),jac=csr_matrix((0,problem.body.size)) if jacobian else None,parts={},metrics={})
            problem.source_motion=NoOpening()
            # Exact stable AL identities are restricted to the new source context.
            origin=ref.p['pts']/12800.;lo,hi=np.asarray(saved['seconds'],float)
            oldpts=ref.p['pts'][(origin>=lo-1e-9)&(origin<=hi+1e-9)]
            labels=[('neutral',name) for name in problem.cavity_labels]+[(int(p)*78125,name) for p in oldpts for name in problem.cavity_labels]
            if len(labels)!=len(saved['multiplier']):raise ValueError('Current AL source-row identity cannot be recovered')
            oldal=ContactAL(saved['multiplier'],float(saved['penalty']))
            problem.set_physics_times(problem.physics_times,old=(oldal,labels))
            write_json(output/'al_identity_mapping.json',dict(old_rows=len(labels),new_rows=len(problem.row_labels),all_target_labels_present=all(k in set(labels) for k in problem.row_labels),mapped_multiplier_rows=sum(k in set(labels) for k in problem.row_labels),new_stage_rows_initialized_to_zero=sum(k not in set(labels) for k in problem.row_labels),new_stage_zero_is_not_a_contact_or_identity_claim=True,ice_multiplier_rows=0,penalty=float(saved['penalty'])))
            runtime=copy.deepcopy(source);mult=ref.grasp_relations['factor_parameters']['shoe_region_weight_multiplier']
            for row in runtime['observations']:row['objective_weight']*=mult
            for row in runtime['boot_axis_observations']:row['objective_weight']*=mult
            problem.support_image=SupportImageFactors(problem,runtime,arc_ids={})
            review=SupportImageFactors(problem,runtime,include_review=True,arc_ids={})
            problem.action_factors=ExistingActionsWithBootAxes(problem,prior,mass,runtime,runner_policy)
            axis_review=BootAxisFactors(problem,runtime,runtime['boot_axis_observations'],include_review=True)
            problem.checkpoint_extra={k:np.asarray(saved[k]).copy() for k in ['initial_reference','initial_reference_sha256']}
            problem.checkpoint_extra.update(parent_parameter_sha256=np.array(digest(parameter)),absolute_parent_case_root=np.array(str(stage)),absolute_case_root=np.array(str(output)),evidence_document_sha256=np.array(digest(output/'support_and_free_shoe_regions.json')),actual_observations_sha256=np.array(digest(case/'observations.npz')),finite_flight_enabled=np.array(True),boot_axis_document_sha256=np.array(digest(document)))
            local=_select_body_columns(problem,_configure_local_controls(problem,active_control_seconds),body_columns);x=saved['x'].copy()
            problem.checkpoint_extra.update(source_snapshot_manifest=np.array(snapshot['path']),source_snapshot_manifest_sha256=np.array(snapshot['sha256']),driver_before_runner_sha256=np.array(driver_backup['sha256']),image_safe_trials=np.array(bool(image_safe_trials)))
            if runner_policy_provenance is not None:
                problem.checkpoint_extra.update(runner_support_policy_path=np.array(str(runner_support_policy)),runner_support_policy_sha256=np.array(runner_policy_provenance['sha256']),runner_support_policy_copy=np.array(runner_policy_provenance['copied_path']))
            # Local factor directional checks; no full-model finite differences.
            d=np.zeros_like(x);rng=np.random.default_rng(1105816);d[problem.active_parameter_indices]=rng.normal(size=len(problem.active_parameter_indices));d/=np.linalg.norm(d)
            fv,check=_factor_directional_check(problem.action_factors.axes,x,d)
            check.update(axis_rows=len(fv['residual']),body_controls=problem.body.state.m,source_rows=len(problem.support_image.rows),local_selection=local)
            write_json(output/'new_axis_derivative_check.json',check)
            if _derivative_failed(check):raise ValueError('New source-axis Jacobian mismatch')
            runner_check=None;start_runner=None;runner_switch=None
            runner=problem.action_factors.runner_support
            if runner is not None:
                runner_start,runner_check=_factor_directional_check(runner,x,d,('normal_fan','gap','u','edge'))
                start_runner=runner_start['metrics'].get('runner_support_mode')
                runner_check.update(body_controls=problem.body.state.m,local_selection=local,relative_error_gate=3e-3)
                write_json(output/'runner_support_derivative_check.json',runner_check)
                if _derivative_failed(runner_check):raise ValueError('Runner-support Jacobian mismatch')
            start_source=review.evaluate(x[17:])['metrics'];start_axes=axis_review.evaluate(x[17:])['metrics']
            emit('bounded_entry_setup',new_axis_check=check,runner_support_check=runner_check,runner_support_policy=runner_policy_provenance,source_snapshot=snapshot,context_seconds=list(context_seconds),core_seconds=list(core_seconds),active_control_seconds=list(active_control_seconds),local_selection=local,mass_proxy_reused=str(mass_path),no_new_shoe_state=True)
            t=time.perf_counter();first=problem.evaluate(x,True);jsec=time.perf_counter()-t
            if not np.isfinite(first['residual']).all() or not np.isfinite(first['jac'].data).all():raise ValueError('Nonfinite first full residual/J')
            start_action=_action_quality(problem.action_factors)
            if runner is not None:
                runner_switch=_runner_switch_check(problem.action_factors,x,problem.action_factors.last,first)
                write_json(output/'runner_support_factor_switch.json',runner_switch)
            quality_scopes=dict(body_image_scope='context body frames only; other source-image factors retain their existing full-document scope',body_image_context_seconds=list(context_seconds),body_image_context_frames=problem.body.n,full_source_body_image_cost=None,full_source_body_image_cost_status='not evaluated by this bounded context problem',pose_prior_scope='full native source grid',pose_prior_frames=len(ref.p['pts']))
            start_groups=_source_quality_groups(start_source,core_seconds,context_seconds)
            write_json(output/'start_quality.json',dict(parent_final_parameter=str(parameter),same_current_source_document=True,source_regions=start_source,source_quality_groups=start_groups,axes=start_axes,joint_parts=first['parts'],joint_cost=float(first['residual']@first['residual']/2),image_cost=first['image_cost'],body_image_cost=.02*first['raw_data_parts'].get('body_pixels',0.),quality_scopes=quality_scopes,raw_nominal_ice_diagnostics=first['ice']['parts'],new_axis_in_real_objective=start_action['action_parts']['boot_axis'],runner_support_before=start_runner,runner_support_in_real_objective=start_action['action_parts'].get('runner_support_mode'),action_quality=start_action,runner_support_policy=runner_policy_provenance,runner_support_factor_switch=runner_switch,source_snapshot=snapshot,first_J_seconds=jsec))
            problem.jacobian_seconds_hint=jsec
            reserve=max(30.,2*jsec);remaining=max(0.,wall_seconds-(time.perf_counter()-start)-reserve)
            policy=RunPolicy(wall_seconds=remaining,max_accepted=max_accepted,max_outer=1,allow_penalty_growth=False)
            emit('bounded_solve_start',first_J_seconds=jsec,setup_seconds=time.perf_counter()-start,reserve_seconds=reserve,policy=vars(policy),body_source_frames=problem.body.n,first_J_reused=True,image_safe_trials=bool(image_safe_trials))
            result,report=problem.solve(x,policy)
            final_value=problem.evaluate(result,False);end_action=_action_quality(problem.action_factors)
            frozen=np.ones(problem.size,bool);frozen[problem.active_parameter_indices]=False
            np.testing.assert_array_equal(result[frozen],saved['x'][frozen])
            end_source=review.evaluate(result[17:])['metrics'];end_axes=axis_review.evaluate(result[17:])['metrics']
            end_runner=runner.evaluate(result[17:])['metrics'].get('runner_support_mode') if runner is not None else None
            final=absolute_reference(ref,problem.body.state.current_curve(result[17:]),root=case)
            candidate=load(problem.output/'candidate.npz')
            with archive.open('xb') as stream:np.savez_compressed(stream,**candidate)
            # Bake the authoritative full951 state without extra optimization.
            shell=copy.copy(problem);shell.reference=final
            packet=save_packet(shell,info,supports,source,case,status='joint_fit',report=report)
            original=load(oldcase/'fitted/result.npz')
            from .skate_physics import fk_state
            s=fk_state(final,final.at(final.times),np.zeros((len(final.times),69)));C,R=mass.evaluate(final.at(final.times),np.zeros((len(final.times),69)))
            p=ref.binding.packet()['parameters'];from .skate_template import profile_height
            xx=np.array([-p.runner_rear_extension_m,.5*(-p.runner_rear_extension_m+p.sole_length_m-p.runner_front_inset_m),p.sole_length_m-p.runner_front_inset_m]);q=np.c_[xx,np.zeros(3),profile_height(xx,p)]
            packet.update(com_world_mid=mass.com(C),blade_material_world_mid=(s['O'][:,:,None]+np.einsum('nfij,kj->nfki',s['R'],q))*ref.binding.ratio,absolute_parameter_path=np.array(str(archive)),**problem.body.state.packet(result[17:]))
            outside=(ref.p['pts']/12800.<active_control_seconds[0])|(ref.p['pts']/12800.>active_control_seconds[1])
            equality={}
            for k in ['P_world_mid','Q_world_mid','local_R_mid','native22_world_mid','skate_R_world_mid','skate_origin_world_mid']:
                equality[k]=float(np.max(abs(packet[k][outside]-original[k][outside])))
                if equality[k]>2e-10:raise ValueError('Local body state changed outside selected support: '+k)
            np.savez_compressed(case/'fitted/result.npz',**packet)
            elapsed=time.perf_counter()-start
            summary=dict(accepted_steps=problem.accepted,retained_step=report['selection']['retained_step'],stop=report['stop'],parameter=str(archive),result=str(case/'fitted/result.npz'),wall_seconds=elapsed,wall_budget_seconds=wall_seconds,wall_budget_overrun_seconds=max(0.,elapsed-wall_seconds),first_J_seconds=jsec,source_frames=len(ref.p['pts']),fit_context_frames=problem.body.n,core_seconds=list(core_seconds),context_seconds=list(context_seconds),active_control_seconds=list(active_control_seconds),source_regions_before=start_source,source_regions_after=end_source,source_quality_groups_before=start_groups,source_quality_groups_after=_source_quality_groups(end_source,core_seconds,context_seconds),boot_axes_before=start_axes,boot_axes_after=end_axes,initial_parts=first['parts'],final_parts=report['parts'],joint_cost_before=float(first['residual']@first['residual']/2),joint_cost_after=float(final_value['residual']@final_value['residual']/2),image_cost_before=first['image_cost'],image_cost_after=final_value['image_cost'],body_image_cost_before=.02*first['raw_data_parts'].get('body_pixels',0.),body_image_cost_after=.02*final_value['raw_data_parts'].get('body_pixels',0.),quality_scopes=quality_scopes,action_quality_before=start_action,action_quality_after=end_action,new_axis_derivative=check,runner_support_before=start_runner,runner_support_after=end_runner,runner_support_policy=runner_policy_provenance,runner_support_derivative=runner_check,runner_support_factor_switch=runner_switch,runner_support_added_residual_rows=0 if runner_switch is None else runner_switch['added_residual_rows'],source_snapshot=snapshot,local_selection=local,outside_state_check_seconds=list(active_control_seconds),outside_state_max_errors=equality,shared_camera_and_shoes_unchanged=True,ice_policy_unchanged=True,accuracy_accepted=False,user_accepted=False,numerical_convergence=False,not_default=True)
            summary['image_safe_trials']=bool(image_safe_trials)
            write_json(output/'run_summary.json',summary);emit('bounded_solve_complete',accepted_steps=problem.accepted,parameter=str(archive),result=summary['result'],stop=report['stop'])
    execution.close()
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','stage','parameter','document','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--core-seconds',nargs=2,type=float,required=True);p.add_argument('--context-seconds',nargs=2,type=float,required=True)
    p.add_argument('--parameter-stem',default='spin_boot_axis_entry_1105');p.add_argument('--max-accepted',type=int,required=True);p.add_argument('--wall-seconds',type=float,required=True)
    p.add_argument('--runner-support-policy',type=Path);p.add_argument('--body-columns',nargs='+',type=int)
    p.add_argument('--driver-backup-path',type=Path);p.add_argument('--active-control-seconds',nargs=2,type=float)
    p.add_argument('--image-safe-trials',action='store_true')
    a=p.parse_args();r=run(a.project,a.stage,a.parameter,a.document,a.output,a.core_seconds,a.context_seconds,max_accepted=a.max_accepted,wall_seconds=a.wall_seconds,parameter_stem=a.parameter_stem,runner_support_policy=a.runner_support_policy,body_columns=a.body_columns,driver_backup_path=a.driver_backup_path,active_control_seconds=a.active_control_seconds,image_safe_trials=a.image_safe_trials)
    print(json.dumps({k:r[k] for k in ['accepted_steps','retained_step','stop','parameter','result','wall_seconds']}),flush=True)
if __name__=='__main__':main()
