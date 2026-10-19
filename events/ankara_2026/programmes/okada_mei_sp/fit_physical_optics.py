"""One explicit material-identity transition after a fully replayed parent.

No live producer is edited. The old accepted source is authenticated first;
its workers close before the reviewed whole-W/ordinary corrections are applied.
The new source/model receives fresh caches and a fresh formal Jacobian.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import json
import os
import sys
import traceback

ROOT=Path(__file__).resolve().parents[4]
DATA=Path(r'D:\CodexData\RinkSpace_Data')
Z=DATA/'events/ankara_2026/programmes/okada_mei_sp/results/shot1_physical_fit_20261014/zoom_poly_72ac'
sys.path[:0]=[str(ROOT),str(Z),str(Path(__file__).parent)]
import numpy as np
from scipy.sparse import csr_matrix,hstack
import fit_venue_material as base
from calibration.program_tracking.first_shot_source_transition import reference,read,_digest
from calibration.program_tracking.first_shot_joint_initializer import (
    read_complete_parent,write_programme_prefix_initializer,authenticate_prefix_parameters,
    authenticate_old_stage_source,_same)
from calibration.program_tracking.first_shot_joint_stage_runtime import (
    JointStageRunner,export_effective_review,preflight_review_media,write,sha)
from calibration.program_tracking.first_shot_safe_frame_export import load_source_frames
from calibration.program_tracking.first_shot_material_source_revision import (
    preflight_material_source,parent_source_without_raw_retention,close_source_revision_pools,
    transform_material_source,revision_comparison_maps,install_revised_source)
from calibration.program_tracking.static_venue_source import AdmittedStaticPacket,StaticCameraPolicy,StaticVenueFactor
from calibration.program_tracking.static_venue_admission import VerifiedStaticMaterialPacket
from calibration.program_tracking.first_shot_static_venue_stage import StaticVenueStage
from calibration.program_tracking.first_shot_derived_inactive import DerivedInactiveChart,DERIVED40
from calibration.program_tracking.first_shot_derived_venue_stage import DerivedVenueStage


from calibration.program_tracking.first_shot_shared_optics_transition import camera_initializer,expand_derived_chart,install as install_shared_optics
from native_transverse_bundle import load_model

SCHEMA='rinkspace.explicit-reviewed-source-and-physical-shared-optics-stage.v1'


def offline_preflight(manifest,output):
    """Read existing safe metadata only, before a source owner is constructed."""
    ref=manifest['material_source_metadata'];payload,receipt=load_source_frames(ref['path'])
    if reference(ref['path'])!=ref or receipt.get('canonical_q_sha256')!=manifest['accepted_parent_q_sha256']:
        raise ValueError('Material source metadata must bind the exact complete parent')
    quarantine=read(manifest['material_group_quarantine'])
    for key,expected in(('parent_binding',manifest['binding']),('parent_association',manifest['association']),
            ('parent_inventory',manifest['accepted_parent_inventory']),
            ('accepted_complete_parameter_parent',manifest['accepted_parent_checkpoint']),('actual_safe_metadata',ref)):
        if quarantine[key]!=expected:raise ValueError('Reviewed material group parent differs: '+key)
    inventory_ref=manifest['accepted_parent_inventory']
    if reference(inventory_ref['path'])!=inventory_ref:raise ValueError('Accepted source inventory bytes changed')
    with np.load(inventory_ref['path'],allow_pickle=False)as saved:
        inventory={key:saved[key].copy()for key in saved.files}
    with np.load(manifest['accepted_parent_checkpoint']['path'],allow_pickle=False)as saved:
        old_static_rows=read(manifest['previous_static_admission'])['control_scalar_rows']
        if len(saved['scales'])-old_static_rows!=manifest['parent_programme_formal_rows']:
            raise ValueError('Preflight parent formal count differs from complete accepted checkpoint')
    result=preflight_material_source(inventory,binding=read(manifest['binding']),
        group_manifest_reference=manifest['material_group_quarantine'],
        ordinary_revision_reference=manifest['material_identity_revision'],
        selected_frames=[item['frame']for item in payload['frames']],
        old_formal_rows=manifest['parent_programme_formal_rows'])
    if (result['source_points']!=manifest['expected_programme_source_points']
            or result['formal_rows']!=manifest['expected_programme_formal_rows']):
        raise ValueError('Reviewed material preflight counts differ from explicit stage contract')
    result.update(actual_source_metadata=ref,immutable_inventory=inventory_ref,
        full_descriptor_and_remaining_W_checks_still_required_on_actual_owner=True)
    write(output,result)
    return reference(output)


def verify_revised_effective_source(owner,old_factor,parent,old_replay,maps,output):
    """New physical camera uses fresh source values; no old camera-r equality."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    programme=owner.programme;q=owner.start;qe,E=owner.effective_q_and_chain(q);n=owner.programme_count
    if n!=7683 or len(q)!=7696:raise ValueError('Explicit expanded camera+wide chart required')
    world_columns=programme.state.world_parameter_indices
    _same(q[world_columns],parent['q'][world_columns],'canonical old171 world coordinates')
    _same(q[n:],parent['q'][7674:],'accepted independent wide13 parameters')
    for key,value in parent['aux'].items():
        _same(owner.checkpoint_arrays()['historical_parent__'+key],value,'historical complete aux '+key)
    owner.guard(q)
    worlds,_=programme.world(q[:n],False);physical,_=programme.world(qe[:n],False)
    if tuple(worlds)!=tuple(physical):raise ValueError('Derived chart changed current material target order')
    for key,value in worlds.items():_same(value,physical[key],'current active canonical/effective world '+key)
    source=list(programme.source_blocks(qe[:n],False));current_r=np.concatenate([block.r for block in source])
    physical_raw=programme.raw(qe[:n]);canonical_raw=programme.raw(q[:n])
    _same(canonical_raw,physical_raw,'all current camera/world canonical-effective raw')
    if not programme.raw_retention_receipt['verified']or programme.raw_retention_receipt['verified_native_points']!=len(physical_raw):
        raise ValueError('New physical source raw did not pass its original full shadow')
    with np.load(output.parent/'old37_to_current44_scalar_map.npz',allow_pickle=False)as saved_map:
        old37_rows=saved_map['retained_static_scalar_indices'].copy()
    with np.load(old_replay['actual_source_arrays']['path'],allow_pickle=False)as saved:
        _same(owner.factor.block(qe,False).r[old37_rows],saved['wide_r'],
            'all old37 source components under unchanged geometry and widecamera')
    _same(owner.factor.block(q,False).r,owner.factor.block(qe,False).r,'all44 canonical/effective wide source')
    first=list(owner.source_blocks(q,False));trial=q.copy();trial[124]+=.01
    middle=list(owner.source_blocks(trial,False));last=list(owner.source_blocks(q,False))
    for a,b in zip(first,last,strict=True):
        _same(a.r,b.r,'new full identity parent-return residual');_same(a.scales,b.scales,'new full identity sigma')
        if _digest(a.row_ledger)!=_digest(b.row_ledger):raise ValueError('New source parent-return ledger changed')
    if all(np.array_equal(a.r,b.r)for a,b in zip(first,middle,strict=True)):
        raise ValueError('Actual shared PP camera trial was silently replayed')
    current_world,D=programme.world(qe[:n],True)
    positions=np.array([int(np.flatnonzero(world_columns==i)[0])for i in owner.chart.fixed_canonical_rows])
    for key,value in D.items():
        if np.any(value[:,:,positions]!=0.):raise ValueError('Derived inactive row drives current target: '+key)
    selected=np.asarray([200,384,588,841,1238,1247]);direction=[];new_column=int(programme.state.optics_indices[0])
    for block in programme.source_blocks(qe[:n],True,indices=selected):
        if block.J[:,programme.state.reduced_camera_fixed_columns].nnz:
            raise ValueError('Inactive2496PP received actual source derivatives')
        J=hstack((block.J,csr_matrix((len(block.r),13))),format='csr');pulled=(J@E).tocsr()
        diff=pulled-J;diff.eliminate_zeros()
        if diff.nnz:raise ValueError('Derived chart changed actual admitted native source J')
        direction.append(pulled[:,new_column].toarray().ravel())
    predicted=np.concatenate(direction);values=[];step=1e-5
    for sign in(-1.,1.):
        sample=q.copy();sample[new_column]+=sign*step;physical_sample,_=owner.effective_q_and_chain(sample)
        values.append(np.concatenate([b.r for b in programme.source_blocks(physical_sample[:n],False,indices=selected)]))
    numeric=(values[1]-values[0])/(2*step)
    fd=dict(column=new_column,step=step,relative_L2=float(np.linalg.norm(predicted-numeric)/max(np.linalg.norm(numeric),1e-30)))
    if fd['relative_L2']>1e-4:raise ValueError('Actual new physical camera native source direction failed: '+str(fd))
    token=np.asarray(qe[:n][world_columns],float).tobytes();programme.transverse_world_cache[token]=(current_world,D)
    np.savez(output/'retained_component_maps.npz',**maps)
    proof=dict(schema='rinkspace.real-reviewed-source-and-physical-optics-stage-QA.v1',passed=True,
        canonical_q_sha256=sha(q),physical_q_sha256=sha(qe),world_targets=len(worlds),
        programme_parameter_count=n,complete_parameter_count=len(q),
        all_current_active_world_and_raw_canonical_effective_exact=True,
        canonical_world171_and_latest_wide13_preserved=True,all15_parent_aux_history_byte_exact=True,
        new_physical_camera_not_claimed_equivalent_to_old_accepted_camera=True,
        old_programme_raw_or_cost_not_reused_or_compared_as_same_camera=True,
        current_programme_source_points=len(physical_raw),current_programme_formal_rows=len(current_r),
        all_old37_wide_source_and_current44_chart_source_exact=True,
        all_current_world_D_derived_columns_zero=True,actual_native_PP2496_J_columns_zero=True,
        six_frame_actual_J_chain_exact=True,new_shared_axial_direction_FD=fd,
        actual_current_physical_camera_raw_full_shadow_verified=True,
        full_current_parent_trial_parent_exact=True,new_stage_world_D_retained_for_first_formal_J=True,
        source_maps=reference(output/'retained_component_maps.npz'),formal_optimizer_called=False)
    write(output/'proof.json',proof);return reference(output/'proof.json')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage-manifest',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--geometry-workers',type=int,default=20)
    parser.add_argument('--source-workers',type=int,default=6);parser.add_argument('--serve',action='store_true')
    parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    ref=reference(args.stage_manifest);manifest=read(ref)
    if manifest.get('schema')!=SCHEMA or manifest.get('entry')!=reference(__file__):
        raise ValueError('Explicit reviewed material entry/manifest identity required')
    for value in manifest.values():
        if isinstance(value,dict)and {'path','sha256'}<=set(value):
            if reference(value['path'])!={key:value[key]for key in('path','sha256')}:
                raise ValueError('Reviewed source/model transition reference changed')
    preflight=offline_preflight(manifest,out/'material_preflight.json')
    if args.preflight_only:return
    from native_pp_continuation import configure_native_runtime
    configure_native_runtime(stage='reviewed-material-corrected44-and-derived-model',native_threads=1)
    (out/'run_wrapper_executed_snapshot.py').write_bytes(Path(__file__).read_bytes())
    programme=None;owner=None
    write(out/'status.json',dict(stage='authenticating_one_complete_old_source_owner',pid=os.getpid(),formal_fit_started=False))
    try:
        media_preflight=preflight_review_media(manifest['binding'],out/'media_preflight.json')
        parent=read_complete_parent(manifest,pool_transition=Z/'geometry_pool_reuse')
        if sha(parent['q'])!=manifest['accepted_parent_q_sha256']:raise ValueError('Full accepted parent q changed')
        prefix_ref=write_programme_prefix_initializer(parent,out/'programme_prefix_initializer.npz')
        bootstrap=deepcopy(manifest);bootstrap['accepted_parent_checkpoint']=prefix_ref
        bootstrap['accepted_parent_inventory']=read(manifest['uncertain_group_manifest'])['parent_inventory']
        original=base.previous.authenticate_parent
        base.previous.authenticate_parent=lambda actual,_:authenticate_prefix_parameters(actual,parent)
        try:programme,context=base.previous.build_owner(bootstrap,out,(args.geometry_workers,args.source_workers))
        finally:base.previous.authenticate_parent=original
        admission=read(manifest['previous_static_admission']);packet_ref=admission['source_packet']
        old_packet=AdmittedStaticPacket(packet_ref['path'],expected_sha256=packet_ref['sha256'],
            admitted_control_ids=admission['admitted_control_ids'])
        template=parent['identity']['persistent_static_venue_stage']['static_venue_factor']['camera']['template_camera25']
        policy=StaticCameraPolicy('brown13',template);options=base.factor_options(programme)
        old_factor=StaticVenueFactor(old_packet,policy,**options)
        import native_transverse_bundle as native
        with parent_source_without_raw_retention(programme,native.CompleteTransverseBundle.source_blocks):
            old_replay=authenticate_old_stage_source(programme,old_factor,parent,
                inventory_reference=manifest['accepted_parent_inventory'],output=out/'accepted_parent_source_QA')
        pool_boundary=close_source_revision_pools(programme)
        old_frames=programme.bundle.frames
        frames,formal_indices,revision_proof=transform_material_source(old_frames,
            group_manifest_reference=manifest['material_group_quarantine'],
            ordinary_revision_reference=manifest['material_identity_revision'],binding=read(manifest['binding']),policy=None)
        maps=revision_comparison_maps(old_frames,frames,formal_indices,manifest['material_identity_revision'])
        correction=install_revised_source(programme,frames,formal_indices,revision_reference=ref,proof=revision_proof)
        current=read(manifest['static_material_admission'])
        packet=VerifiedStaticMaterialPacket(packet_ref['path'],expected_sha256=packet_ref['sha256'],
            admitted_control_ids=current['admitted_control_ids'],reviewed_admission=manifest['static_material_admission'])
        dependencies=sorted({int(i)for row in packet.ledger['reviewed_material_extension']['bindings']for i in row['geometry_dependencies']})
        required=sorted(set(dependencies)&set(DERIVED40))
        if required!=sorted(manifest['release_rows'])or required!=sorted(manifest['admission_dependencies']):
            raise ValueError('Actual admitted material dependencies require different derived release')
        retained,delta=base.compare_admitted_rows(old_packet,packet)
        np.savez(out/'old37_to_current44_scalar_map.npz',retained_static_scalar_indices=retained)
        write(out/'source_transition.json',dict(schema='rinkspace.complete-reviewed-material-source-transition.v1',
            old_parent_source_replay=reference(out/'accepted_parent_source_QA/proof.json'),
            reviewed_material_correction=correction,preformal_execution_pool_boundary=pool_boundary,
            static_admission=manifest['static_material_admission'],static_old37_extension=delta,
            offline_metadata_preflight=preflight,complete_parent_world_wide_aux_history_damping_fraction_preserved=True,
            old_source_r_J_and_raw_cache_not_reused_for_new_objective=True))
        # One explicit model/parameter transition after old-source authentication.
        model=load_model(manifest['shared_optics_model'])
        candidate_state=model.TransverseNativePrincipalState(programme.state)
        previous_chart=DerivedInactiveChart.from_recipe(programme.state,manifest['derived_recipe']['path'],
            parameter_count=len(parent['q']),release_rows=manifest['release_rows'],admission_dependencies=manifest['admission_dependencies'])
        effective_parent,_=previous_chart.effective_q_and_chain(parent['q'])
        with np.load(manifest['physical_camera_initializer']['path'],allow_pickle=False)as saved:physical_seed=saved['q'].copy()
        new_start,camera_map=camera_initializer(parent['q'],physical_seed,candidate_state,effective_parent_q=effective_parent)
        camera_transition=install_shared_optics(programme,model_reference=manifest['shared_optics_model'],
            start=new_start[:candidate_state.count],full_parent_aux=parent['aux'],entry_reference=reference(__file__),transition_reference=ref)
        write(out/'camera_transition.json',dict(camera_transition,parameter_mapping=camera_map,
            physical_initializer=manifest['physical_camera_initializer'],accepted_parent_q_sha256=sha(parent['q']),
            new_complete_initializer_q_sha256=sha(new_start),new_programme_initializer_q_sha256=sha(programme.start)))
        options=base.factor_options(programme)
        factor=StaticVenueFactor(packet,policy,**options)
        source_refs=[manifest[key]for key in('static_material_admission','previous_static_admission','derived_recipe',
            'derived_producer','derived_wrapper','material_policy_producer','initializer_producer','runtime_producer',
            'base_static_producer','static_camera_kernel','static_world_kernel','static_projection_kernel',
            'material_group_quarantine','material_identity_revision','material_source_revision_producer',
            'shared_optics_model','shared_optics_transition_producer','physical_camera_initializer','physical_mount_facts')]
        physical=StaticVenueStage(programme,factor,parent_reference=manifest['accepted_parent_report'],
            entry_reference=reference(__file__),source_references=source_refs)
        chart=expand_derived_chart(previous_chart,programme.state,parameter_count=len(new_start),camera_start=programme.start)

        chart.assert_source_admission(dependencies)
        new_aux=physical.checkpoint_arrays()
        new_aux['static_venue_initial_camera_state']=parent['aux']['static_venue_initial_camera_state'].copy()
        owner=DerivedVenueStage(physical,chart,canonical_start=new_start,inherited_aux=new_aux,
            entry_reference=reference(__file__),transition_reference=ref,producer_references=(reference(out/'source_transition.json'),))
        counts=dict(programme_source_points=sum(f['source_point_count']for f in programme.bundle.frames),
            programme_formal_rows=len(programme.scales),static_group_count=len(packet.rows),static_UV_count=len(factor.scales)//2,
            static_formal_rows=len(factor.scales),complete_formal_rows=len(owner.scales),parameter_count=len(owner.start),
            inherited_auxiliary_count=len(parent['aux']),global_shared_count=len(owner.shared),fixed_count=len(owner.fixed))
        for key,value in counts.items():
            if manifest['expected_'+key]!=value:raise ValueError('Explicit corrected-source contract changed: '+key)
        proof=verify_revised_effective_source(owner,old_factor,parent,old_replay,maps,out/'effective_source_QA')
        runner=JointStageRunner(owner,output=out,manifest_reference=ref,
            review_export=lambda q,folder:export_effective_review(owner,q,folder,physical_export=base.previous.export_review))
        runner.prepare(complete_parent_source_QA=reference(out/'accepted_parent_source_QA/proof.json'),effective_source_QA=proof,
            material_source_preflight=preflight,geometry_context=context,authenticated_source_and_parameter_contract=counts,
            complete_media_preflight=media_preflight,canonical_complete_parent_world171_and_wide13_bits_preserved=True,
            all_parent_auxiliary_values_preserved_in_explicit_history=True,static_camera_reconditioned_or_replaced=False,
            reviewed_source_and_explicit_physical_camera_model_transition=True,
            all_native_PP2496_fixed_zero_and_shared_optics9_joint=True)
        print(json.dumps(dict(stage='physical_optics_corrected_source_QA_passed_waiting_control',path=str(out/'append_QA.json'),pid=os.getpid())),flush=True)
        if args.serve:runner.run(wait_for_control=True)
    except BaseException as error:
        write(out/'run_failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            accepted_parent_and_previous_live_source_not_overwritten=True))
        raise
    finally:
        if owner is not None:owner.close()
        elif programme is not None:programme.close()


if __name__=='__main__':main()
