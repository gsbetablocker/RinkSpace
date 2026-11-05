"""Bounded boot-image repair on one current continuous S05 body state.

Consumes an independent current case; preserves frozen shoes/camera, finite
ice and existing action factors. No prepare/inference/historical pose replay.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,copy,json,os,sys,time
import numpy as np
from scipy.sparse import csr_matrix,vstack
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
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():raise FileExistsError(dst)
    os.link(src,dst)



_DRIVER_BACKUP=Path(r'D:\CodexData\RinkSpace_Data\runs\body_s01_20261029\blade_support_profile_1105\implementation_before_runner\blade_boot_refit.py')


def _checked_path(path):
    path=Path(path).resolve()
    if len(str(path).encode('utf-16-le'))//2>=260 or any(len(p.encode('utf-16-le'))//2>255 for p in path.parts):
        raise ValueError('Windows UTF-16 path limit: '+str(path))
    return path


def _preserve_driver_before_runner():
    path=_checked_path(_DRIVER_BACKUP)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        with path.open('xb') as stream:stream.write(Path(__file__).read_bytes())
    return dict(path=str(path),sha256=digest(path))


def _snapshot_sources(project,output,camera_info,driver_backup):
    # Exact local code, never model weights or environment/package copies.
    source_root=_checked_path(output/'source_snapshot');source_root.mkdir()
    paths=set(Path(__file__).resolve().parent.glob('*.py'))
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            path=Path(name).resolve()
            if path.suffix=='.py' and path.is_relative_to(project) and path.is_file():paths.add(path)
    entries=[]
    for src in sorted(paths):
        relative=src.relative_to(project);dst=_checked_path(source_root/relative)
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


def _factor_directional_check(factor,x,d):
    value=factor.evaluate(x[17:],True);h=1e-6
    numeric=(factor.evaluate(x[17:]+h*d[17:])['residual']-factor.evaluate(x[17:]-h*d[17:])['residual'])/(2*h)
    exact=value['jac']@d;difference=numeric-exact
    return value,dict(relative_directional_error=float(np.linalg.norm(difference)/max(np.linalg.norm(numeric),1e-12)),maximum_absolute=float(np.max(abs(difference),initial=0.)),residual_rows=len(value['residual']))


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
        self.last=result
        return result


def run(project,stage,parameter,document,output,core_seconds,context_seconds,*,max_accepted=8,wall_seconds=600.,parameter_stem='spin_boot_axis_entry_1105',runner_support_policy=None,body_columns=None):
    project,stage,parameter,document,output=map(lambda p:Path(p).resolve(),(project,stage,parameter,document,output))
    if output.exists():raise FileExistsError('One bounded run; preserve existing output: '+str(output))
    driver_backup=_preserve_driver_before_runner()
    output.mkdir(parents=True);start=time.perf_counter();log_path=output/'fit_progress.jsonl'
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
            local=_select_body_columns(problem,_configure_local_controls(problem,core_seconds),body_columns);x=saved['x'].copy()
            snapshot=_snapshot_sources(project,output,info,driver_backup)
            problem.checkpoint_extra.update(source_snapshot_manifest=np.array(snapshot['path']),source_snapshot_manifest_sha256=np.array(snapshot['sha256']),driver_before_runner_sha256=np.array(driver_backup['sha256']))
            if runner_policy_provenance is not None:
                problem.checkpoint_extra.update(runner_support_policy_path=np.array(str(runner_support_policy)),runner_support_policy_sha256=np.array(runner_policy_provenance['sha256']),runner_support_policy_copy=np.array(runner_policy_provenance['copied_path']))
            # Local factor directional checks; no full-model finite differences.
            d=np.zeros_like(x);rng=np.random.default_rng(1105816);d[problem.active_parameter_indices]=rng.normal(size=len(problem.active_parameter_indices));d/=np.linalg.norm(d)
            fv,check=_factor_directional_check(problem.action_factors.axes,x,d)
            check.update(axis_rows=len(fv['residual']),body_controls=problem.body.state.m,source_rows=len(problem.support_image.rows),local_selection=local)
            write_json(output/'new_axis_derivative_check.json',check)
            if check['relative_directional_error']>3e-3:raise ValueError('New source-axis Jacobian mismatch')
            runner_check=None;start_runner=None;runner_switch=None
            runner=problem.action_factors.runner_support
            if runner is not None:
                runner_start,runner_check=_factor_directional_check(runner,x,d)
                start_runner=runner_start['metrics'].get('runner_support_mode')
                runner_check.update(body_controls=problem.body.state.m,local_selection=local,relative_error_gate=3e-3)
                write_json(output/'runner_support_derivative_check.json',runner_check)
                if runner_check['relative_directional_error']>3e-3:raise ValueError('Runner-support Jacobian mismatch')
            start_source=review.evaluate(x[17:])['metrics'];start_axes=axis_review.evaluate(x[17:])['metrics']
            emit('bounded_entry_setup',new_axis_check=check,runner_support_check=runner_check,runner_support_policy=runner_policy_provenance,source_snapshot=snapshot,context_seconds=list(context_seconds),core_seconds=list(core_seconds),mass_proxy_reused=str(mass_path),no_new_shoe_state=True)
            t=time.perf_counter();first=problem.evaluate(x,True);jsec=time.perf_counter()-t
            if not np.isfinite(first['residual']).all() or not np.isfinite(first['jac'].data).all():raise ValueError('Nonfinite first full residual/J')
            if runner is not None:
                runner_switch=_runner_switch_check(problem.action_factors,x,problem.action_factors.last,first)
                write_json(output/'runner_support_factor_switch.json',runner_switch)
            write_json(output/'start_quality.json',dict(parent_final_parameter=str(parameter),same_current_source_document=True,source_regions=start_source,axes=start_axes,joint_parts=first['parts'],image_cost=first['image_cost'],raw_nominal_ice_diagnostics=first['ice']['parts'],new_axis_in_real_objective=first['parts'].get('boot_axis'),runner_support_before=start_runner,runner_support_in_real_objective=first['parts'].get('runner_support_mode'),runner_support_policy=runner_policy_provenance,runner_support_factor_switch=runner_switch,source_snapshot=snapshot,first_J_seconds=jsec))
            problem.jacobian_seconds_hint=jsec
            reserve=max(30.,2*jsec);remaining=max(0.,wall_seconds-(time.perf_counter()-start)-reserve)
            policy=RunPolicy(wall_seconds=remaining,max_accepted=max_accepted,max_outer=1,allow_penalty_growth=False)
            emit('bounded_solve_start',first_J_seconds=jsec,setup_seconds=time.perf_counter()-start,reserve_seconds=reserve,policy=vars(policy),body_source_frames=problem.body.n,first_J_reused=True)
            result,report=problem.solve(x,policy)
            frozen=np.ones(problem.size,bool);frozen[problem.active_parameter_indices]=False
            np.testing.assert_array_equal(result[frozen],saved['x'][frozen])
            end_source=review.evaluate(result[17:])['metrics'];end_axes=axis_review.evaluate(result[17:])['metrics']
            end_runner=runner.evaluate(result[17:])['metrics'].get('runner_support_mode') if runner is not None else None
            final=absolute_reference(ref,problem.body.state.current_curve(result[17:]),root=case)
            if not parameter_stem.replace('_','').isalnum():raise ValueError('Expected a plain parameter basename')
            archive=project/PROGRAMME/(parameter_stem+'.npz')
            if len(str(archive).encode('utf-16-le'))//2>=260 or archive.exists():raise FileExistsError('Preserve final parameter path: '+str(archive))
            candidate=load(problem.output/'candidate.npz');np.savez_compressed(archive,**candidate)
            # Bake the authoritative full951 state without extra optimization.
            shell=copy.copy(problem);shell.reference=final
            packet=save_packet(shell,info,supports,source,case,status='joint_fit',report=report)
            original=load(oldcase/'fitted/result.npz')
            from .skate_physics import fk_state
            s=fk_state(final,final.at(final.times),np.zeros((len(final.times),69)));C,R=mass.evaluate(final.at(final.times),np.zeros((len(final.times),69)))
            p=ref.binding.packet()['parameters'];from .skate_template import profile_height
            xx=np.array([-p.runner_rear_extension_m,.5*(-p.runner_rear_extension_m+p.sole_length_m-p.runner_front_inset_m),p.sole_length_m-p.runner_front_inset_m]);q=np.c_[xx,np.zeros(3),profile_height(xx,p)]
            packet.update(com_world_mid=mass.com(C),blade_material_world_mid=(s['O'][:,:,None]+np.einsum('nfij,kj->nfki',s['R'],q))*ref.binding.ratio,absolute_parameter_path=np.array(str(archive)),**problem.body.state.packet(result[17:]))
            outside=(ref.p['pts']/12800.<core_seconds[0])|(ref.p['pts']/12800.>core_seconds[1])
            equality={}
            for k in ['P_world_mid','Q_world_mid','local_R_mid','native22_world_mid','skate_R_world_mid','skate_origin_world_mid']:
                equality[k]=float(np.max(abs(packet[k][outside]-original[k][outside])))
                if equality[k]>2e-10:raise ValueError('Local body state changed outside selected support: '+k)
            np.savez_compressed(case/'fitted/result.npz',**packet)
            summary=dict(accepted_steps=problem.accepted,retained_step=report['selection']['retained_step'],stop=report['stop'],parameter=str(archive),result=str(case/'fitted/result.npz'),wall_seconds=time.perf_counter()-start,wall_budget_seconds=wall_seconds,first_J_seconds=jsec,source_frames=951,fit_context_frames=problem.body.n,core_seconds=list(core_seconds),context_seconds=list(context_seconds),source_regions_before=start_source,source_regions_after=end_source,boot_axes_before=start_axes,boot_axes_after=end_axes,initial_parts=first['parts'],final_parts=report['parts'],new_axis_derivative=check,runner_support_before=start_runner,runner_support_after=end_runner,runner_support_policy=runner_policy_provenance,runner_support_derivative=runner_check,runner_support_factor_switch=runner_switch,runner_support_added_residual_rows=0 if runner_switch is None else runner_switch['added_residual_rows'],source_snapshot=snapshot,local_selection=local,outside_state_max_errors=equality,shared_camera_and_shoes_unchanged=True,ice_policy_unchanged=True,accuracy_accepted=False,user_accepted=False,numerical_convergence=False,not_default=True)
            write_json(output/'run_summary.json',summary);emit('bounded_solve_complete',accepted_steps=problem.accepted,parameter=str(archive),result=summary['result'],stop=report['stop'])
    execution.close()
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','stage','parameter','document','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--core-seconds',nargs=2,type=float,required=True);p.add_argument('--context-seconds',nargs=2,type=float,required=True)
    p.add_argument('--parameter-stem',default='spin_boot_axis_entry_1105');p.add_argument('--max-accepted',type=int,default=8);p.add_argument('--wall-seconds',type=float,default=600.)
    p.add_argument('--runner-support-policy',type=Path);p.add_argument('--body-columns',nargs='+',type=int)
    a=p.parse_args();r=run(a.project,a.stage,a.parameter,a.document,a.output,a.core_seconds,a.context_seconds,max_accepted=a.max_accepted,wall_seconds=a.wall_seconds,parameter_stem=a.parameter_stem,runner_support_policy=a.runner_support_policy,body_columns=a.body_columns)
    print(json.dumps({k:r[k] for k in ['accepted_steps','retained_step','stop','parameter','result','wall_seconds']}),flush=True)
if __name__=='__main__':main()
