"""One restored actual-source owner with fixed-head native pan/tilt and optics.

The complete source snapshot was transformed and losslessly round-tripped
before launch. Previous source r/J are provenance only. All current projection,
physical guards and the first full linearization run on the new physical model.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import json
import os
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[4]
DATA=Path(r'D:\CodexData\RinkSpace_Data')
Z=DATA/'events/ankara_2026/programmes/okada_mei_sp/results/shot1_physical_fit_20261014/zoom_poly_72ac'
sys.path[:0]=[str(ROOT),str(Z),str(Z.parent/'head_axis_audit'),str(Path(__file__).parent)]
from calibration.solver_runtime import install
install()
import numpy as np
from scipy.sparse import hstack,csr_matrix
import fit_venue_material as base
from calibration.program_tracking.first_shot_source_transition import reference,read,_digest
from calibration.program_tracking.first_shot_joint_initializer import read_complete_parent,_same
from calibration.program_tracking.first_shot_joint_stage_runtime import JointStageRunner,export_effective_review,write,sha
from calibration.program_tracking.first_shot_source_snapshot import load_source_snapshot
from calibration.program_tracking.first_shot_restored_owner import restore_programme_owner
from calibration.program_tracking.first_shot_fixed_head_transition import camera_initializer,expand_derived_chart,install as install_camera
from calibration.program_tracking.first_shot_derived_inactive import DerivedInactiveChart,DERIVED40
from calibration.program_tracking.first_shot_derived_venue_stage import DerivedVenueStage
from calibration.program_tracking.first_shot_static_venue_stage import StaticVenueStage
from calibration.program_tracking.static_venue_source import AdmittedStaticPacket,StaticVenueFactor,StaticCameraPolicy
from calibration.program_tracking.static_venue_admission import VerifiedStaticMaterialPacket
from native_transverse_bundle import load_model

SCHEMA='rinkspace.restored-reviewed-source-fixed-head-optics-stage.v1'
SAFE_INDICES=np.array([200,384,588,841,1238,1247])


def metadata_preflight(owner,output):
    """Serialize the actual complete runtime identities before any r/J/pool."""
    from calibration.program_tracking.first_shot_safe_frame_export import _encode
    for name in ('bundle','start','state','shared','fixed','scales','quadratic','formal_offsets',
            'source_producers','recovery_references','worker_receipts','evaluation_receipts',
            'transverse_world_cache','guard_cache','raw_retention_receipt','seed_receipt'):
        if name not in owner.programme.__dict__:raise ValueError('Incomplete restored owner field: '+name)
    identity=owner.checkpoint_identity();json.dumps(identity,ensure_ascii=False,allow_nan=False)
    aux=owner.checkpoint_arrays()
    for name,value in aux.items():
        if np.asarray(value).dtype.hasobject:raise ValueError('Unsafe checkpoint auxiliary array: '+name)
    arrays={};payload=owner.complete_state_payload(owner.start)
    encoded=_encode(payload,arrays);json.dumps(encoded,ensure_ascii=False,allow_nan=False)
    kwargs=owner.solver_kwargs()
    if (kwargs['global_schur_count']!=len(owner.shared)
            or not np.array_equal(kwargs['global_schur_indices'],owner.shared)
            or not np.array_equal(kwargs['fixed_parameter_indices'],owner.fixed)):
        raise ValueError('Complete actual solver metadata does not match expanded chart')
    write(output,dict(schema='rinkspace.actual-complete-owner-metadata-preflight.v1',passed=True,
        checkpoint_identity_actually_JSON_serialized=True,complete_safe_payload_actually_encoded_and_JSON_serialized=True,
        auxiliary_count=len(aux),all_auxiliary_arrays_non_object=True,safe_payload_array_count=len(arrays),
        full_parameter_count=len(owner.start),shared_count=len(owner.shared),fixed_count=len(owner.fixed),
        source_residual_or_J_evaluated=False,source_or_geometry_pool_opened=False))
    return reference(output)


def verify_actual_fixed_head(owner,parent,manifest,output):
    """Verify real revised source and camera chains without a duplicate full J."""
    from calibration.program_tracking.first_shot_row_source import evaluate_row_timed_frame
    from calibration.program_tracking.first_shot_fit import FrozenWhitening
    from native_transverse_bundle import source_chunk
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    programme=owner.programme;q=owner.start;qe,E=owner.effective_q_and_chain(q);n=owner.programme_count
    if n!=7687 or len(q)!=7700:raise ValueError('Complete R2/shared-optics/wide width changed')
    world_columns=programme.state.world_parameter_indices
    retained_world=world_columns[world_columns!=97]
    _same(q[retained_world],parent['q'][retained_world],'canonical retained170 world')
    _same(q[n:],parent['q'][7674:],'accepted independent wide13')
    for key,value in parent['aux'].items():
        _same(owner.checkpoint_arrays()['historical_parent__'+key],value,'all15 historical aux '+key)
    owner.guard(q)
    worlds,_=programme.world(q[:n],False);effective,_=programme.world(qe[:n],False)
    if tuple(worlds)!=tuple(effective):raise ValueError('Derived chart changed target inventory')
    for key in worlds:_same(worlds[key],effective[key],'all current active physical world '+key)
    blocks=list(owner.source_blocks(q,False));r=np.concatenate([block.r for block in blocks])
    raw=owner.raw_programme(q)
    if not programme.raw_retention_receipt['verified']or programme.raw_retention_receipt['verified_native_points']!=len(raw):
        raise ValueError('New full source trial/raw shadow did not verify')
    with np.load(manifest['authenticated_parent_source_arrays']['path'],allow_pickle=False)as saved:
        old_packet=read(manifest['previous_static_admission'])
        packet=AdmittedStaticPacket(old_packet['source_packet']['path'],expected_sha256=old_packet['source_packet']['sha256'],
            admitted_control_ids=old_packet['admitted_control_ids'])
        retained,_=base.compare_admitted_rows(packet,owner.factor.packet)
        before_endpoint=qe.copy();before_endpoint[97]=parent['q'][97]
        _same(owner.factor.block(before_endpoint,False).r[retained],saved['wide_r'],
            'all old37 source before the explicit original endpoint initializer')
        wide_delta=owner.factor.block(qe,False).r[retained]-saved['wide_r']
    _same(owner.factor.block(q,False).r,owner.factor.block(qe,False).r,'all44 canonical/effective source')
    trial=q.copy();trial[124]+=.01
    changed=list(owner.source_blocks(trial,False));returned=list(owner.source_blocks(q,False))
    for a,b in zip(blocks,returned,strict=True):
        _same(a.r,b.r,'all new source parent-return');_same(a.scales,b.scales,'all new source retained sigma')
        if _digest(a.row_ledger)!=_digest(b.row_ledger):raise ValueError('Returned row ledger changed')
    if all(np.array_equal(a.r,b.r)for a,b in zip(blocks,changed,strict=True)):
        raise ValueError('Shared PP trial silently replayed')
    # The actual unchanged world arrays are supplied to the real native source
    # workers. Geometry derivatives are explicitly zero only for this camera-
    # chain check. The formal first J computes every real geometry derivative.
    camera_D={key:np.zeros(value.shape+(len(world_columns),))for key,value in effective.items()}
    frames=list(map(int,SAFE_INDICES));task=(qe[:n],effective,camera_D,frames,sha(qe[:n]))
    jr,J,receipt=next(programme._pool().map(source_chunk,[task],chunksize=1))
    if J.shape!=(len(jr),n)or J[:,programme.state.fixed_camera_parameter_indices].nnz:
        raise ValueError('Fixed camera slots received actual native derivatives')
    pulled=(hstack((J,csr_matrix((len(jr),13))),format='csr')@E).tocsr()
    columns=[int(programme.state.quadratic_index),*map(int,programme.state.head_axes_indices),
        *map(int,programme.state.view_columns(1248)[:2])]
    checks=[]
    for column in columns:
        step=1e-6 if column in programme.state.head_axes_indices or column in programme.state.view_columns(1248)[:2] else 1e-5
        values=[]
        for sign in(-1.,1.):
            sample=q.copy();sample[column]+=sign*step;sphysical,_=owner.effective_q_and_chain(sample)
            local=[]
            for i in SAFE_INDICES:
                frame=programme.bundle.frames[int(i)];w=programme.bundle.whitening[int(i)]
                result=evaluate_row_timed_frame(programme.state,sphysical[:n],frame,w,effective,None)
                local.append(result[0])
            values.append(np.concatenate(local))
        numeric=(values[1]-values[0])/(2*step);predicted=pulled[:,column].toarray().ravel()
        relative=float(np.linalg.norm(numeric-predicted)/max(np.linalg.norm(numeric),1e-30))
        checks.append(dict(column=column,step=step,relative_L2=relative,
            numeric_L2=float(np.linalg.norm(numeric)),maximum_absolute=float(np.max(np.abs(numeric-predicted)))))
        if relative>1e-4 and np.max(np.abs(numeric-predicted))>1e-5:
            raise ValueError('Actual revised-source fixed-head FD failed: '+str(checks[-1]))
    proof=dict(schema='rinkspace.actual-reviewed-source-fixed-head-camera-QA.v1',passed=True,
        canonical_q_sha256=sha(q),physical_q_sha256=sha(qe),parameter_count=len(q),programme_parameter_count=n,
        source_points=len(raw),programme_formal_rows=len(programme.scales),complete_rows=len(owner.scales),
        all_current_active_world_canonical_effective_byte_exact=True,all15_parent_aux_history_byte_exact=True,
        canonical_world170_and_wide13_parent_values_byte_exact=True,
        q97_only_original_unmeasured_source_coverage_initializer=True,
        current_old37_wide_after_q97_changed_components=int(np.count_nonzero(wide_delta)),
        current_old37_wide_after_q97_maximum_normalized_difference=float(np.max(np.abs(wide_delta))),
        all_old37_source_before_q97_initialization_byte_exact=True,new_full_source_parent_trial_parent_exact=True,
        actual_new_model_trial_raw_full_shadow_verified=True,
        inactive_PP2496_and_third_pose1248_actual_J_columns_zero=True,
        actual_six_frame_source_camera_J_and_chain_FD=checks,spawn_worker=receipt,
        geometry_J_computed_for_this_camera_only_QA=False,
        first_formal_J_requires_all_current_world171_derivatives=True,old_source_r_J_used_for_new_objective=False,
        initializer_not_accepted_fit=True,source_accuracy_not_claimed_by_numeric_QA=True)
    write(output/'proof.json',proof);return reference(output/'proof.json')


def phase_runner(previous,owner,output,manifest_reference):
    """Keep the same immutable media inventory, with a fresh phase J writer."""
    runner=JointStageRunner.__new__(JointStageRunner)
    runner.owner=owner;runner.output=Path(output);runner.output.mkdir(parents=True,exist_ok=False)
    runner.manifest_reference=manifest_reference;runner.generation=0;runner.records=[];runner.started=time.time()
    runner.cache=runner.output/'source_J_cache';runner.inventory=previous.inventory;runner.media=previous.media
    runner.review_export=lambda q,folder:export_effective_review(owner,q,folder,physical_export=base.previous.export_review)
    # The public driver references an inventory file beside its phase output.
    # Reuse its exact bytes, without another source/media export or frame copy.
    (runner.output/'immutable_programme_source_inventory.npz').write_bytes(
        (previous.output/'immutable_programme_source_inventory.npz').read_bytes())
    return runner


def run_camera_then_joint(owner,initial_runner,manifest,manifest_reference,out):
    """Retain this source pool across two explicit public-solver active sets."""
    import importlib
    write(out/'status.json',dict(stage='initial_safe_packet_ready_configuring_camera_initialization',pid=os.getpid(),fit_started=False))
    control_path=out/'camera_initializer_configuration.json'
    while not control_path.exists():time.sleep(2.)
    control_ref=reference(control_path);control=read(control_ref)
    if control.get('action')!='configure_camera_only':raise ValueError('Explicit camera-only phase configuration required')
    for key in ('producer','QA'):
        if reference(control[key]['path'])!=control[key]:raise ValueError('Camera-only phase producer/QA changed')
    if read(control['QA']).get('passed')is not True:raise ValueError('Passed camera-only protocol QA required')
    module=importlib.import_module('calibration.program_tracking.first_shot_camera_initialization')
    if reference(module.__file__)!=control['producer']:raise ValueError('Camera-only module import differs')
    camera=module.CameraInitializationStage(owner,entry_reference=reference(__file__),initializer_reference=control_ref)
    camera_runner=phase_runner(initial_runner,camera,out/'camera_initializer',control_ref)
    metadata_preflight(camera,camera_runner.output/'metadata_preflight.json')
    camera_runner.prepare(initial_actual_source_QA=reference(out/'append_QA.json'),
        complete_geometry_and_wide_fixed=True,geometry_derivative_cache_used=False,
        source_rows_operators_and_actual_RS_neighbour_links_preserved=True)
    write(out/'status.json',dict(stage='camera_only_full1248_formal_running',pid=os.getpid(),fit_started=True,
        fixed_world171_and_wide13=True,camera_shared=28,native_pan_tilt_zoom=3744))
    result=camera_runner.run(wait_for_control=False)
    owner.guard(result.q)
    latest=read(reference(camera_runner.output/'current.json'))
    checkpoint=latest['canonical_checkpoint']
    with np.load(checkpoint['path'],allow_pickle=False)as saved:
        _same(saved['q'],result.q,'camera-only final complete accepted state')
        damping=float(saved['damping']);fraction=float(saved['next_line_search_fraction'])
    programme=owner.programme
    programme.seed_receipt=deepcopy(programme.seed_receipt)
    programme.seed_receipt.update(initial_damping=damping,initial_line_search_fraction=fraction,
        accepted_camera_initialization_parent=checkpoint)
    transition=out/'joint_phase_transition.json'
    write(transition,dict(schema='rinkspace.same-owner-camera-initialization-to-full-joint.v1',
        camera_report=reference(camera_runner.output/'current.json'),camera_complete_checkpoint=checkpoint,
        q_sha256=sha(result.q),complete_q_and_aux_retained=True,
        old_camera_restricted_source_J_and_zero_worldD_not_reused=True,
        source_pool_reused_without_source_or_model_change=True,fresh_full_world171_and_complete_source_J_required=True,
        initial_damping=damping,initial_line_search_fraction=fraction))
    joint=DerivedVenueStage(owner.physical_stage,owner.chart,canonical_start=result.q.copy(),
        inherited_aux=owner.checkpoint_arrays(),entry_reference=reference(__file__),
        transition_reference=reference(transition),producer_references=(control_ref,))
    joint_runner=phase_runner(initial_runner,joint,out/'joint',reference(transition))
    metadata_preflight(joint,joint_runner.output/'metadata_preflight.json')
    joint_runner.prepare(camera_initialization_parent=checkpoint,all171world_and_wide13_released_to_joint=True,
        camera_J_or_zero_worldD_cache_loaded=False)
    write(out/'status.json',dict(stage='joint_full_world_fresh_J_formal_running',pid=os.getpid(),fit_started=True))
    joint_runner.run(wait_for_control=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage-manifest',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--geometry-workers',type=int,default=20)
    parser.add_argument('--source-workers',type=int,default=6);parser.add_argument('--serve',action='store_true')
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    ref=reference(args.stage_manifest);manifest=read(ref)
    if manifest.get('schema')!=SCHEMA or manifest.get('entry')!=reference(__file__):
        raise ValueError('Explicit restored-source fixed-head entry/manifest required')
    for value in manifest.values():
        if isinstance(value,dict)and {'path','sha256'}<=set(value):
            if reference(value['path'])!={k:value[k]for k in('path','sha256')}:
                raise ValueError('Explicit source/model producer changed: '+value['path'])
    from native_pp_continuation import configure_native_runtime
    configure_native_runtime(stage='restored-reviewed-source-fixed-head-optics',native_threads=1)
    (out/'run_wrapper_executed_snapshot.py').write_bytes(Path(__file__).read_bytes())
    programme=owner=None;started=time.time()
    try:
        write(out/'status.json',dict(stage='restoring_complete_actual_source_metadata',pid=os.getpid(),fit_started=False))
        parent=read_complete_parent(manifest,pool_transition=Z/'geometry_pool_reuse')
        if sha(parent['q'])!=manifest['accepted_parent_q_sha256']:raise ValueError('Accepted full parent changed')
        from calibration.program_tracking import first_shot_value_execution as value,first_shot_raw_execution as raw,first_shot_fused_execution as fused,row_scene_projection_fast
        parent_installation=value.install();row_scene_projection_fast.install()
        programme,restored=restore_programme_owner(manifest['complete_source_snapshot'],parent,
            model_reference=manifest['previous_model'],association_reference=manifest['association'],
            geometry_workers=args.geometry_workers,source_workers=args.source_workers,entry_reference=reference(__file__))
        write(out/'source_restore.json',restored)
        previous_chart=DerivedInactiveChart.from_recipe(programme.state,manifest['derived_recipe']['path'],
            parameter_count=len(parent['q']),release_rows=manifest['release_rows'],admission_dependencies=manifest['admission_dependencies'])
        effective_parent,_=previous_chart.effective_q_and_chain(parent['q'])
        model=load_model(manifest['fixed_head_model']);state=model.TransverseNativePrincipalState(programme.state)
        with np.load(manifest['physical_camera_initializer']['path'],allow_pickle=False)as saved:seed=saved['q'].copy()
        with np.load(manifest['feasible_complete_initializer']['path'],allow_pickle=False)as saved:
            start=saved['original_initializer_q'].copy()
            feasible=saved['q'].copy();saved_physical=saved['physical_q'].copy()
        initial_proof=read(manifest['feasible_initializer_proof'])
        if (initial_proof.get('passed')is not True or initial_proof['model']!=manifest['fixed_head_model']
                or initial_proof['source_snapshot']!=manifest['complete_source_snapshot']):
            raise ValueError('Complete feasible initializer proof belongs to another model/source')
        original_proof=read(initial_proof['parent_initial_coverage'])
        mapping=deepcopy(original_proof['initializer_mapping'])
        state._q(start[:state.count])
        _same(start[state.world_parameter_indices],parent['q'][state.world_parameter_indices],'original canonical171 world initializer')
        _same(start[state.count:],parent['q'][7674:],'original accepted independent wide13')
        pose=np.concatenate([state.view_columns(i)[:3]for i in range(1,state.frame_count+1)])
        camera=np.setdiff1d(np.arange(7674),np.r_[state.world_parameter_indices,pose])
        _same(start[camera],seed[camera],'physical shared camera and all1248 original focal initializer')
        if np.any(start[state.optics_indices]!=0.):raise ValueError('Unexpected shared optics initializer change')
        changed=np.flatnonzero(feasible!=start)
        if not np.array_equal(changed,np.array([97]))or feasible[97]<start[97]:
            raise ValueError('Explicit feasible initializer changed parameters outside original q97 monotonic extent')
        start=feasible
        mapping.update(original_q97_value=float(parent['q'][97]),feasible_q97_value=float(start[97]),
            original_public_source_coverage_initializer=manifest['feasible_initializer_proof'],
            q97_source_supported_lower_bound_not_measured_endpoint=True,other_world170_and_camera_and_wide_parameters_byte_exact=True)
        from calibration.program_tracking.first_shot_transverse_execution import install_geometry_context
        context=install_geometry_context(programme)
        raw.install(programme)
        execution=fused.install_worker_execution(programme,parent_installation=parent_installation,entry_reference=reference(__file__),
            validation_references=(reference(Z/'value_real_state_QA/report.json'),reference(Z/'pivot_step_QA/report.json')))
        execution.update(full_same_seed_source_r_J_proof_required=False,
            cross_source_basis_old_new_r_J_equality_not_claimed=True,new_source_initial_full_r_J_required=True,
            unused_programme_only_conditional_metric_not_constructed=True,
            complete_source_public_normal_includes_static_cross_terms=True)
        camera=install_camera(programme,model_reference=manifest['fixed_head_model'],start=start[:state.count],
            full_parent_aux=parent['aux'],entry_reference=reference(__file__),transition_reference=ref)
        write(out/'camera_transition.json',dict(camera,parameter_mapping=mapping))
        admission=read(manifest['static_material_admission']);packet_ref=admission['source_packet']
        packet=VerifiedStaticMaterialPacket(packet_ref['path'],expected_sha256=packet_ref['sha256'],
            admitted_control_ids=admission['admitted_control_ids'],reviewed_admission=manifest['static_material_admission'])
        template=parent['identity']['persistent_static_venue_stage']['static_venue_factor']['camera']['template_camera25']
        factor=StaticVenueFactor(packet,StaticCameraPolicy('brown13',template),**base.factor_options(programme))
        physical=StaticVenueStage(programme,factor,parent_reference=manifest['accepted_parent_report'],entry_reference=reference(__file__),
            source_references=[v for v in manifest.values()if isinstance(v,dict)and {'path','sha256'}<=set(v)])
        chart=expand_derived_chart(previous_chart,programme.state,parameter_count=len(start),camera_start=programme.start)
        projected,_=chart.effective_q_and_chain(start)
        _same(projected,saved_physical,'complete canonical-effective feasible initializer')
        dependencies=sorted({int(i)for row in packet.ledger['reviewed_material_extension']['bindings']for i in row['geometry_dependencies']})
        chart.assert_source_admission(dependencies)
        aux=physical.checkpoint_arrays();aux['static_venue_initial_camera_state']=parent['aux']['static_venue_initial_camera_state'].copy()
        owner=DerivedVenueStage(physical,chart,canonical_start=start,inherited_aux=aux,entry_reference=reference(__file__),
            transition_reference=ref,producer_references=(reference(out/'source_restore.json'),reference(out/'camera_transition.json')))
        counts=dict(programme_source_points=sum(f['source_point_count']for f in programme.bundle.frames),
            programme_formal_rows=len(programme.scales),static_group_count=len(packet.rows),static_formal_rows=len(factor.scales),
            complete_formal_rows=len(owner.scales),parameter_count=len(owner.start),global_shared_count=len(owner.shared),fixed_count=len(owner.fixed))
        for key,v in counts.items():
            if v!=manifest['expected_'+key]:raise ValueError('Actual restored-source/model contract changed: '+key)
        metadata=metadata_preflight(owner,out/'metadata_preflight.json')
        write(out/'status.json',dict(stage='actual_fixed_head_source_guard_and_camera_J_QA',pid=os.getpid(),elapsed_seconds=time.time()-started,counts=counts))
        proof=verify_actual_fixed_head(owner,parent,manifest,out/'actual_source_QA')
        runner=JointStageRunner(owner,output=out,manifest_reference=ref,
            review_export=lambda q,folder:export_effective_review(owner,q,folder,physical_export=base.previous.export_review))
        runner.prepare(actual_new_source_camera_QA=proof,source_restore=reference(out/'source_restore.json'),
            camera_transition=reference(out/'camera_transition.json'),complete_source_contract=counts,geometry_context=context,
            actual_metadata_preflight=metadata,new_fixed_head_source_r_J_identity=True,initial_state_not_accepted_fit=True)
        print(json.dumps(dict(stage='actual_fixed_head_source_QA_passed',path=str(out/'append_QA.json'),pid=os.getpid(),elapsed_seconds=time.time()-started)),flush=True)
        if args.serve:run_camera_then_joint(owner,runner,manifest,ref,out)
    except BaseException as error:
        from calibration.program_tracking.first_shot_safe_frame_export import _encode
        error_arrays={};error_report=_encode(getattr(error,'report',None),error_arrays)
        if error_arrays:np.savez_compressed(out/'run_failure_arrays.npz',**error_arrays)
        write(out/'run_failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            actual_domain_or_source_report=error_report,
            previous_failures_and_accepted_parent_not_overwritten=True))
        raise
    finally:
        if owner is not None:owner.close()
        elif programme is not None:programme.close()


if __name__=='__main__':main()
