"""Explicit full accepted37 -> reviewed44 and derived-reference model stage."""
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
from calibration.solver_runtime import install
install()
import numpy as np
from scipy.sparse import csr_matrix,hstack
import fit_venue_joint as previous
from calibration.program_tracking.first_shot_source_transition import reference,read,_digest
from calibration.program_tracking.first_shot_joint_initializer import (
    read_complete_parent,write_programme_prefix_initializer,authenticate_prefix_parameters,
    authenticate_old_stage_source,_same)
from calibration.program_tracking.first_shot_joint_stage_runtime import (
    JointStageRunner,export_effective_review,preflight_review_media,write,sha)
from calibration.program_tracking.static_venue_source import AdmittedStaticPacket,StaticCameraPolicy,StaticVenueFactor
from calibration.program_tracking.static_venue_admission import VerifiedStaticMaterialPacket
from calibration.program_tracking.first_shot_static_venue_stage import StaticVenueStage
from calibration.program_tracking.first_shot_derived_inactive import DerivedInactiveChart,DERIVED40
from calibration.program_tracking.first_shot_derived_venue_stage import DerivedVenueStage


def factor_options(programme):
    def guard(q):
        programme.state._activate(q)
        return programme.state.original.guard_geometry(q[:programme.state.original.count])
    return dict(programme_count=len(programme.start),geometry_columns=programme.state.world_parameter_indices,
        scene_at=programme.state.scene,geometry_guard=guard,
        geometry_fd_step=programme.state.original.provider.fd_step)


def compare_admitted_rows(old,new):
    by_id={row['id']:row for row in new.rows};cursor=0;positions={}
    for row in new.rows:
        positions[row['id']]=np.arange(cursor,cursor+2*len(row['pixels_uv']));cursor+=2*len(row['pixels_uv'])
    retained=[]
    for row in old.rows:
        if row['id']not in by_id or _digest(row)!=_digest(by_id[row['id']]):
            raise ValueError('Material extension changed an original base37 row')
        retained.extend(positions[row['id']])
    retained=np.asarray(retained,np.int64)
    _same(new.scales[retained],old.scales,'retained37 sigma/scalar order')
    _same(new.quadratic[retained],old.quadratic,'retained37 masks')
    _same(new.loss_weights.integration_weights[retained],old.loss_weights.integration_weights,'retained37 literal weights')
    if old.frame!=new.frame or old.contract!=new.contract:
        raise ValueError('Independent static source clock/contract changed')
    added=[row['id']for row in new.rows if row['id']not in {r['id']for r in old.rows}]
    return retained,dict(old_groups=len(old.rows),new_groups=len(new.rows),added_group_ids=added,
        old_rows=len(old.scales),new_rows=len(new.scales),added_rows=len(new.scales)-len(old.scales),
        retained_row_map_sha256=sha(retained),all_old_UV_sigma_target_weight_and_PTS_exact=True)


def verify_effective_source(owner,old_factor,parent,old_replay,output):
    """Real source/world invariance and a native q3 derivative on six frames."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    programme=owner.programme;q=owner.start;qe,E=owner.effective_q_and_chain(q)
    _same(q,parent['q'],'canonical full start, including current wide13')
    for key,value in parent['aux'].items():_same(owner.checkpoint_arrays()[key],value,'inherited aux '+key)
    owner.guard(q)
    worlds,_=programme.world(q[:7674],False);physical,_=programme.world(qe[:7674],False)
    if tuple(worlds)!=tuple(physical):raise ValueError('Effective model changed original source world keys')
    for key,value in worlds.items():_same(value,physical[key],'effective source world '+key)
    raw=programme.raw(qe[:7674])
    with np.load(old_replay['actual_source_arrays']['path'],allow_pickle=False)as saved:
        _same(raw,saved['programme_raw'],'all programme raw under effective model')
        _same(old_factor.block(qe,False).r,saved['wide_r'],'all old37 raw under effective model')
    a=owner.factor.block(q,False);b=owner.factor.block(qe,False)
    _same(a.r,b.r,'all new44 raw under effective model')
    # Evaluate a full parent -> trial -> parent on the new source/model.
    first=list(owner.source_blocks(q,False));trial=q.copy();trial[3]+=programme.state.original.provider.fd_step
    middle=list(owner.source_blocks(trial,False));last=list(owner.source_blocks(q,False))
    for initial,returned in zip(first,last,strict=True):
        _same(initial.r,returned.r,'new source cache parent-return residual')
        _same(initial.scales,returned.scales,'new source cache parent-return sigma')
        if _digest(initial.row_ledger)!=_digest(returned.row_ledger):raise ValueError('Parent-return source ledger changed')
    if all(np.array_equal(x.r,y.r)for x,y in zip(first,middle,strict=True)):
        raise ValueError('Actual changed q3 source trial was silently replayed')
    # All current programme world derivatives are produced once in this new
    # identity. Keep that same-q cache for the formal first full source J.
    current_world,D=programme.world(qe[:7674],True)
    columns=np.asarray(programme.state.world_parameter_indices,int)
    derived_positions=np.array([int(np.flatnonzero(columns==i)[0])for i in owner.chart.fixed_canonical_rows])
    for key,value in D.items():
        if np.any(value[:,:,derived_positions]!=0.):
            raise ValueError('A derived physical coordinate still drives programme world: '+key)
    selected=np.asarray([200,384,588,841,1238,1247])
    native=list(programme.source_blocks(qe[:7674],True,indices=selected))
    native_j=[]
    for block in native:
        J=hstack((block.J,csr_matrix((len(block.r),13))),format='csr')
        pulled=(J@E).tocsr();difference=pulled-J;difference.eliminate_zeros()
        if difference.nnz:raise ValueError('Inactive chart changed an admitted programme native Jacobian')
        native_j.append(pulled[:,3].toarray().ravel())
    predicted=np.concatenate(native_j);directions=[]
    for step in (programme.state.original.provider.fd_step,programme.state.original.provider.fd_step/2):
        values=[]
        for sign in(-1.,1.):
            canonical=q.copy();canonical[3]+=sign*step
            physical_sample,_=owner.effective_q_and_chain(canonical)
            values.append(np.concatenate([block.r for block in programme.source_blocks(
                physical_sample[:7674],False,indices=selected)]))
        actual=(values[1]-values[0])/(2*step)
        relative=float(np.linalg.norm(actual-predicted)/max(np.linalg.norm(actual),1e-30))
        cosine=float(actual@predicted/max(np.linalg.norm(actual)*np.linalg.norm(predicted),1e-30))
        directions.append(dict(column=3,step=step,relative_L2=relative,cosine=cosine))
    if max(r['relative_L2']for r in directions)>1e-4:
        raise ValueError('Actual six-frame native J@E direction failed: '+str(directions))
    token=np.asarray(qe[:7674][columns],float).tobytes()
    programme.transverse_world_cache[token]=(current_world,D)
    programme.transverse_world_cache.move_to_end(token)
    proof=dict(schema='rinkspace.real-complete-canonical-effective-source-QA.v1',passed=True,
        canonical_q_sha256=sha(q),physical_q_sha256=sha(qe),world_targets=len(worlds),
        programme_source_points=len(raw),old37_and_new44_all_raw_byte_exact=True,
        all_programme_world_and_raw_byte_exact=True,full_parent_trial_parent_exact=True,
        all_programme_world_D_derived_columns_exact_zero=True,
        six_frames_actual_J_chain_exact=True,six_frame_native_directional_FD=directions,
        same_new_stage_q_geometry_cache_retained=True,older_identity_or_source_J_loaded=False,
        derived_canonical_rows=owner.chart.fixed_canonical_rows.tolist(),formal_optimizer_called=False)
    write(output/'proof.json',proof)
    return reference(output/'proof.json')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage-manifest',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--geometry-workers',type=int,default=20)
    parser.add_argument('--source-workers',type=int,default=6);parser.add_argument('--serve',action='store_true')
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    ref=reference(args.stage_manifest);manifest=read(ref)
    if (manifest.get('schema')!='rinkspace.explicit-complete-programme-static-material-transition.v2'
            or manifest.get('entry')!=reference(__file__)):
        raise ValueError('Explicit complete new-source/model entry identity required')
    for value in manifest.values():
        if isinstance(value,dict)and {'path','sha256'}<=set(value):
            if reference(value['path'])!={key:value[key]for key in('path','sha256')}:
                raise ValueError('New source/model transition reference changed')
    from native_pp_continuation import configure_native_runtime
    configure_native_runtime(stage='reviewed44-and-derived-reference-model',native_threads=1)
    (out/'run_wrapper_executed_snapshot.py').write_bytes(Path(__file__).read_bytes())
    programme=None;owner=None
    write(out/'status.json',dict(stage='authenticating_one_full_parent_owner',pid=os.getpid(),formal_fit_started=False))
    try:
        media_preflight=preflight_review_media(manifest['binding'],out/'media_preflight.json')
        parent=read_complete_parent(manifest,pool_transition=Z/'geometry_pool_reuse')
        if sha(parent['q'])!=manifest['accepted_parent_q_sha256']:
            raise ValueError('Explicit full accepted parent parameter hash changed')
        prefix_ref=write_programme_prefix_initializer(parent,out/'programme_prefix_initializer.npz')
        bootstrap=deepcopy(manifest);bootstrap['accepted_parent_checkpoint']=prefix_ref
        bootstrap['accepted_parent_inventory']=read(manifest['uncertain_group_manifest'])['parent_inventory']
        original=previous.authenticate_parent
        previous.authenticate_parent=lambda actual,_:authenticate_prefix_parameters(actual,parent)
        try:programme,context=previous.build_owner(bootstrap,out,(args.geometry_workers,args.source_workers))
        finally:previous.authenticate_parent=original
        admission=read(manifest['previous_static_admission']);packet_ref=admission['source_packet']
        old_packet=AdmittedStaticPacket(packet_ref['path'],expected_sha256=packet_ref['sha256'],
            admitted_control_ids=admission['admitted_control_ids'])
        template=parent['identity']['persistent_static_venue_stage']['static_venue_factor']['camera']['template_camera25']
        policy=StaticCameraPolicy('brown13',template);options=factor_options(programme)
        old_factor=StaticVenueFactor(old_packet,policy,**options)
        old_replay=authenticate_old_stage_source(programme,old_factor,parent,
            inventory_reference=manifest['accepted_parent_inventory'],output=out/'accepted_parent_source_QA')
        current=read(manifest['static_material_admission'])
        new_packet=VerifiedStaticMaterialPacket(packet_ref['path'],expected_sha256=packet_ref['sha256'],
            admitted_control_ids=current['admitted_control_ids'],reviewed_admission=manifest['static_material_admission'])
        actual_dependencies=sorted({int(i)for row in new_packet.ledger['reviewed_material_extension']['bindings']
            for i in row['geometry_dependencies']})
        required_release=sorted(set(actual_dependencies)&set(DERIVED40))
        if (required_release!=sorted(manifest['release_rows'])
                or required_release!=sorted(manifest['admission_dependencies'])):
            raise ValueError('Actual admitted material dependencies require a different derived-row release')
        retained,delta=compare_admitted_rows(old_packet,new_packet)
        np.savez(out/'old37_to_new44_scalar_map.npz',retained_static_scalar_indices=retained)
        write(out/'source_transition.json',dict(schema='rinkspace.explicit-admitted-material-source-transition.v1',
            previous_admission=manifest['previous_static_admission'],new_admission=manifest['static_material_admission'],
            delta=delta,row_map=reference(out/'old37_to_new44_scalar_map.npz'),
            old_source_replay=reference(out/'accepted_parent_source_QA/proof.json'),
            actual_new_material_geometry_dependencies=actual_dependencies,
            released_reference_rows=required_release,
            complete_parent_q_and_aux_preserved=True,old_new_costs_different_objectives=True,
            old_source_J_or_value_cache_borrowed=False))
        factor=StaticVenueFactor(new_packet,policy,**options)
        source_refs=[manifest[key]for key in('static_material_admission','previous_static_admission',
            'derived_recipe','derived_producer','derived_wrapper','material_policy_producer','initializer_producer',
            'runtime_producer','base_static_producer',
            'static_camera_kernel','static_world_kernel','static_projection_kernel')]
        physical=StaticVenueStage(programme,factor,parent_reference=manifest['accepted_parent_report'],
            entry_reference=reference(__file__),source_references=source_refs)
        chart=DerivedInactiveChart.from_recipe(programme.state,manifest['derived_recipe']['path'],
            parameter_count=len(parent['q']),release_rows=manifest['release_rows'],
            admission_dependencies=manifest['admission_dependencies'])
        chart.assert_source_admission(actual_dependencies)
        owner=DerivedVenueStage(physical,chart,canonical_start=parent['q'],inherited_aux=parent['aux'],
            entry_reference=reference(__file__),transition_reference=ref,
            producer_references=(reference(out/'source_transition.json'),))
        actual_counts=dict(programme_source_points=sum(f['source_point_count']for f in programme.bundle.frames),
            programme_formal_rows=len(programme.scales),static_group_count=len(new_packet.rows),
            static_UV_count=len(factor.scales)//2,static_formal_rows=len(factor.scales),
            complete_formal_rows=len(owner.scales),parameter_count=len(owner.start),
            inherited_auxiliary_count=len(parent['aux']),global_shared_count=len(owner.shared),fixed_count=len(owner.fixed))
        for key,value in actual_counts.items():
            if manifest['expected_'+key]!=value:
                raise ValueError('Explicit source/model row contract changed: '+key)
        proof=verify_effective_source(owner,old_factor,parent,old_replay,out/'effective_source_QA')
        for key,value in parent['aux'].items():_same(owner.checkpoint_arrays()[key],value,'final inherited aux '+key)
        runner=JointStageRunner(owner,output=out,manifest_reference=ref,
            review_export=lambda q,folder:export_effective_review(owner,q,folder,physical_export=previous.export_review))
        runner.prepare(complete_parent_source_QA=reference(out/'accepted_parent_source_QA/proof.json'),
            effective_source_QA=proof,geometry_context=context,
            authenticated_source_and_parameter_contract=actual_counts,
            complete_media_preflight=media_preflight,
            canonical_complete_parent_parameter_bits_preserved=True,
            all_parent_auxiliary_values_preserved=True,static_camera_reconditioned_or_replaced=False)
        print(json.dumps(dict(stage='reviewed44_model_QA_passed_waiting_control',
            path=str(out/'append_QA.json'),pid=os.getpid())),flush=True)
        if args.serve:runner.run(wait_for_control=True)
    except BaseException as error:
        write(out/'run_failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            old_accepted_complete_state_not_overwritten=True))
        raise
    finally:
        if owner is not None:owner.close()
        elif programme is not None:programme.close()


if __name__=='__main__':main()
