"""Ankara shot1: actual reviewed source prefix plus admitted material motion.

Run from the project checkout. Outputs, complete checkpoints and diagnostics stay
in Data. Preflight never constructs a full new J; --fit continues through the
unchanged public solver after that same owner's preflight passes.
"""
from pathlib import Path
from copy import deepcopy
from types import MethodType, SimpleNamespace
import argparse
import importlib.util
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
AUDIT=Path(r'D:/CodexData/RinkSpace_Data/audit_1017')
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
from calibration.program_tracking.first_shot_source_transition import reference,read
from calibration.program_tracking.first_shot_joint_stage_runtime import write,sha
from calibration.program_tracking.measured_board_motion import BoardMotionPacket,MeasuredBoardMotionObjective


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def selected_complete_point(previous, packet=None):
    """Restore one closed parent's authenticated complete numerical boundary."""
    closed_reference=reference(previous/'owner_closed.json');closed=read(closed_reference)
    if closed.get('passed')is not True or closed.get('normal_close_returned')is not True:
        raise ValueError('The selected previous owner must complete its normal close')
    current_reference=reference(previous/'current.json');current=read(current_reference)
    payload=current['snapshot'];canonical=current['canonical_checkpoint']
    if reference(payload['path'])!=payload or reference(canonical['path'])!=canonical:
        raise ValueError('The selected complete or canonical checkpoint changed')
    with np.load(payload['path'],allow_pickle=False)as saved:
        q=saved['q_augmented'].copy();physical=saved['physical_q_augmented'].copy()
        expanded=saved['native_time_expanded_base_canonical_q'].copy()
        auxiliary={k[5:]:saved[k].copy()for k in saved.files if k.startswith('aux__')}
        complete_damping=float(saved['damping'])
        complete_fraction=float(saved['next_line_search_fraction'])
    with np.load(canonical['path'],allow_pickle=False)as saved:
        if (q.shape!=saved['q'].shape or q.dtype!=saved['q'].dtype or q.tobytes()!=saved['q'].tobytes()
                or {k[5:]for k in saved.files if k.startswith('aux__')}!=set(auxiliary)):
            raise ValueError('Selected complete and canonical inventories differ')
        for key,value in auxiliary.items():
            actual=saved['aux__'+key]
            if actual.shape!=value.shape or actual.dtype!=value.dtype or actual.tobytes()!=value.tobytes():
                raise ValueError('Selected complete and canonical auxiliary differs: '+key)
        damping=float(saved['damping']);fraction=float(saved['next_line_search_fraction'])
    if (not np.isfinite(q).all() or not np.isfinite(physical).all() or not np.isfinite(expanded).all()
            or not np.isfinite(damping) or damping<=0. or not np.isfinite(fraction) or not 0.<fraction<=1.
            or damping!=complete_damping or fraction!=complete_fraction
            or damping!=float(current['damping']) or fraction!=float(current['next_line_search_fraction'])):
        raise ValueError('Selected complete/canonical/report numerical state differs')
    if reference(previous/'current.json')!=current_reference:
        raise ValueError('The selected previous current boundary moved during restoration')
    archive='ad7_source_paths_222c3d0cb80c9a54_json'
    motion_aux={}if packet is None else packet.checkpoint_arrays()
    restored_motion={key:auxiliary[key]for key in motion_aux if key in auxiliary}
    if restored_motion:
        if set(restored_motion)!=set(motion_aux):
            raise ValueError('Selected joint parent has an incomplete motion inventory')
        for key,expected in motion_aux.items():
            actual=restored_motion[key]
            if (actual.dtype!=expected.dtype or actual.shape!=expected.shape
                    or actual.tobytes()!=expected.tobytes()):
                raise ValueError('Selected joint parent motion packet differs: '+key)
    base_auxiliary={key:value for key,value in auxiliary.items()if key not in restored_motion}
    if sha(q)!=current['q_sha256']or len(base_auxiliary)!=406 or archive not in base_auxiliary:
        raise ValueError('The exact 406 base auxiliaries plus a complete authenticated motion inventory are required')
    if len(auxiliary)not in (406,406+len(motion_aux)):
        raise ValueError('Selected complete parent contains unknown auxiliaries')
    probe_direction=None;probe_reference=None
    index_path=previous/'accepted_state_index.json'
    if index_path.exists():
        index_reference=reference(index_path)
        for item in reversed(read(index_reference)['records']):
            if item['q_sha256']==current['q_sha256']:continue
            record=read(item['report']);prior=record['snapshot']
            if reference(prior['path'])!=prior:raise ValueError('Prior accepted probe state changed')
            with np.load(prior['path'],allow_pickle=False)as stored:
                candidate=stored['q_augmented'].copy()
                candidate_aux={k[5:]:stored[k]for k in stored.files if k.startswith('aux__')}
                if set(candidate_aux)!=set(auxiliary):continue
                if any(candidate_aux[k].dtype!=v.dtype or candidate_aux[k].shape!=v.shape
                       or candidate_aux[k].tobytes()!=v.tobytes()for k,v in auxiliary.items()):continue
            if sha(candidate)!=item['q_sha256']or candidate.shape!=q.shape:
                raise ValueError('Prior accepted probe identity differs')
            probe_direction=candidate-q
            probe_reference=dict(index=index_reference,report=item['report'],snapshot=prior,
                q_sha256=item['q_sha256'],role='derivative_probe_direction_only_no_optimizer_seed_change')
            break
    return dict(q=q,physical_q=physical,expanded_q=expanded,
        probe_reference_direction=probe_direction,probe_reference=probe_reference,
        aux={k:v for k,v in base_auxiliary.items()if k!=archive},identity_aux={archive:base_auxiliary[archive]},
        complete_aux=base_auxiliary,complete_parent_aux=auxiliary,restored_motion_aux=restored_motion,
        damping=damping,fraction=fraction,payload=payload,
        parent_current=current_reference,parent_closed=closed_reference,
        parent_canonical=canonical,parent_source_identity=current.get('source_identity',{}),
        checkpoint_is_accuracy_accepted=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion-packet',required=True,type=Path)
    parser.add_argument('--output',type=Path,default=AUDIT/'motion_joint_body')
    parser.add_argument('--seed',choices=('latest-main','motion-v5-current'),default='latest-main')
    parser.add_argument('--previous',type=Path,help='Closed complete parent in audit_1017; restores its exact current state including motion evidence')
    parser.add_argument('--reviewed-print-paths',type=Path,help='Explicit original-RGB membership review in existing print paths')
    parser.add_argument('--fresh-source-jacobian',action='store_true',help='Conservative full same-state derivative refresh for all material branches')
    parser.add_argument('--check-complete-source-J',action='store_true',help='Before the first real direction, compare fresh full source J against guarded actual source values')
    parser.add_argument('--finite-ad-coverage',action='store_true',help='Restore authenticated finite advertisement support with the existing phase-I/truth gates')
    parser.add_argument('--same-state-source-J',type=Path,help='Explicit passed complete source J at exactly this q and all 414 auxiliaries; no changed-state reuse')
    parser.add_argument('--fit',action='store_true')
    parser.add_argument('--contract-only',action='store_true')
    parser.add_argument('--manifest-only',action='store_true',help='Authenticate new producer transition and complete parent without constructing an owner')
    args=parser.parse_args();out=args.output.resolve()
    if out.parent!=AUDIT or out.exists():
        raise ValueError('A new direct child of audit_1017 is required; existing runs are preserved')
    # Authenticate material/body/clock and all long-chain evidence before any
    # expensive historical owner construction or source worker starts.
    packet=BoardMotionPacket(reference(args.motion_packet))
    out.mkdir();write(out/'motion_packet_QA.json',dict(passed=True,packet=packet.receipt,
        producer=reference(__file__),owner_started=False,full_source_J_constructed=False))
    if args.contract_only:return
    previous=(args.previous.resolve()if args.previous is not None else AUDIT/('solve6'if args.seed=='latest-main'else'motion_joint_v5'))
    if previous.parent!=AUDIT or not previous.is_dir():
        raise ValueError('An existing complete parent directly inside audit_1017 is required')
    point=selected_complete_point(previous,packet);payload=point['payload'];auxiliary=point['complete_aux']
    if point['parent_source_identity'].get('finite_ad_support_policy') is not None:
        args.finite_ad_coverage=True
    if point['parent_source_identity'].get('source_J_execution_policy',{}).get('fresh_current_source_J_every_new_state'):
        args.fresh_source_jacobian=True
    inherited_review=point['parent_source_identity'].get('reviewed_print_path_membership')
    if args.reviewed_print_paths is None and inherited_review is not None:
        inherited_mapping=inherited_review['mapping']
        if reference(inherited_mapping['path'])!=inherited_mapping:raise ValueError('Inherited print review changed')
        args.reviewed_print_paths=Path(inherited_mapping['path'])
    reviewed_paths_ref=None if args.reviewed_print_paths is None else reference(args.reviewed_print_paths)
    if reviewed_paths_ref is not None:
        from calibration.program_tracking import first_shot_reviewed_print_paths as membership
        membership.validate_manifest(reviewed_paths_ref)
    write(out/'parent_restoration.json',dict(passed=True,parent_current=point['parent_current'],
        parent_canonical=point['parent_canonical'],q_sha256=sha(point['q']),
        base_aux_count=len(auxiliary),complete_parent_aux_count=len(point['complete_parent_aux']),
        restored_motion_aux_count=len(point['restored_motion_aux']),
        damping=point['damping'],next_line_search_fraction=point['fraction'],
        exact_current_parent_restored=True,old_J_or_stop_control_restored=False,
        derivative_probe_reference=point['probe_reference']))
    # Reuse the existing reviewed full-shot construction and C0 inventory
    # exporter. No source/J/world values or old trial cache become the new target.
    legacy=load('measured_motion_historical_factory',AUDIT/'motion_joint_v5/run.py')
    factory=legacy.factory;factory.OUT=out;factory.qa.OUT=out;legacy.HERE=out
    factory.qa.base.main=legacy._original_construction_main
    for filename in ('combined_c0.json','source_successors.json'):
        source=AUDIT/'motion_joint_v5'/filename
        (out/filename).write_bytes(source.read_bytes())
    class CompleteContinuation(legacy.OriginalContinuation):
        def __init__(self,base,selected,entry):
            super().__init__(base,selected,entry)
            self.continuation.update(selected_parent=point['parent_current'],
                selected_parent_closed=point['parent_closed'],selected_canonical_parent=point['parent_canonical'],
                selected_seed_role=('explicit-complete-parent'if args.previous is not None else args.seed),
                current_aux_count=406,complete_parent_aux_count=len(point['complete_parent_aux']),
                accepted_accuracy_claimed=False,motion_evidence_is_new_objective=True)
            actual=self.checkpoint_arrays()
            for key,value in auxiliary.items():factory.qa.same(actual[key],value,'all selected inherited auxiliary '+key)
        def checkpoint_arrays(self):
            result=super().checkpoint_arrays();result.update(self.point['identity_aux']);return result
    CompleteContinuation.__module__=factory.__name__
    factory.CompleteContinuation=CompleteContinuation;factory.CurrentContinuation=CompleteContinuation
    factory.policy_entry.load_current_point=lambda:point
    # Explicit producer successor for this new source/derivative experiment.
    # The frozen manifest and every historical run remain byte unchanged.
    def prepare_producer_successor():
        original_ref=reference(Path(__file__).resolve().parent/'source_datasets/s1_wide/fixed_head_time4b2.json')
        config=deepcopy(read(original_ref));changes=[]
        declared={
            'first_shot_prediction_refresh.py':('d99f679c74cc0a18460395a886f3f8ce9114cb7da41e786d2e528a575285eff6',None,'Previously certified original rejected-Schur retry'),
            'first_shot_jacobian_reuse.py':('e73d1f2c346c5a501429bd46e0693a5725dbb310a25d6ef4c7e57e84d91d9aac',AUDIT/'first_shot_jacobian_reuse_before_fresh_option.py','Explicit complete-current-J option; original reuse mode retained'),
            'first_shot_native_circle.py':('9787eb6cfc4bbb9332a1ba7515ee0005b3cd7c8f36a40a69d93b010e02276043',AUDIT/'first_shot_native_circle_before_C0_1017.py','C0 native midpoint reader extension; existing F2 and projection code retained; C0 not installed in FIT')}
        expected={str((ROOT/'calibration/program_tracking'/name).resolve()):(old,backup,reason)for name,(old,backup,reason)in declared.items()}
        originals={}
        for path,(old,backup,reason)in expected.items():
            if backup is not None:
                oldref=reference(backup)
                if oldref['sha256']!=old:raise ValueError('Preserved historical producer does not match the declared transition: '+path)
                originals[path]=oldref
        def revise(value,key=''):
            if isinstance(value,dict):
                if {'path','sha256'}<=set(value):
                    actual=reference(value['path']);old={k:value[k]for k in ('path','sha256')}
                    if actual!=old:
                        path=str(Path(value['path']).resolve());declaration=expected.get(path)
                        if declaration is None or value['sha256']!=declaration[0]:
                            raise ValueError('Undeclared source/producer change: '+repr(value))
                        changes.append(dict(key=key,source_path=path,historical_sha256=value['sha256'],
                            current_sha256=actual['sha256'],preserved_historical_copy=originals.get(path),reason=declaration[2]))
                        value.update(actual)
                else:
                    for k,v in value.items():revise(v,key+'.'+k)
            elif isinstance(value,list):
                for i,v in enumerate(value):revise(v,key+'['+str(i)+']')
        revise(config)
        counts={name:sum(Path(c['source_path']).name==name for c in changes)for name in declared}
        if counts!={'first_shot_prediction_refresh.py':2,'first_shot_jacobian_reuse.py':2,'first_shot_native_circle.py':1}:
            raise ValueError('Exact declared producer transition inventory differs: '+str(counts))
        config['actual20_read_only_execution']=dict(entry=reference(__file__),historical_manifest=original_ref,
            explained_reference_updates=changes,source_math_changed=False,
            policy_scope='Explicit fresh-J execution and diagnostic-only C0 reader successor; original source/camera/world constructor unchanged',
            old_manifest_preserved=True,original_source_and_physics_references_authenticate=True,
            actual_parent_source_replay_required=True,formal_fit_or_acceptance_forbidden=True)
        config['explicit_producer_successor']=dict(entry=reference(__file__),changes=changes,
            old_source_r_J_or_H_reused=False,unchanged_references_strictly_authenticated=True,
            later_source_alias_membership_version_is_explicit=True)
        from calibration.program_tracking.first_shot_time_successor import authenticate_references
        authenticate_references(config)
        target=out/'execution_manifest2.json';write(target,config)
        write(out/'identity_transition2.json',dict(passed=True,original=original_ref,current=reference(target),
            changed=changes,no_SHA_ignored=True,all_other_references_and_initial_state_still_verified=True))
        return reference(target)
    factory.qa.prepare_manifest=prepare_producer_successor
    original_version_install=factory.install_version_transition
    def install_producer_successors():
        original_version_install()
        from calibration.program_tracking.first_shot_resume_cache import CompleteSourceReplay
        inherited=CompleteSourceReplay.verify_reference
        declarations={}
        for name,old,backup,reason in (
            ('first_shot_jacobian_reuse.py','e73d1f2c346c5a501429bd46e0693a5725dbb310a25d6ef4c7e57e84d91d9aac','first_shot_jacobian_reuse_before_fresh_option.py','Explicit full-fresh-J option, with the original default reuse math retained'),
            ('first_shot_native_circle.py','9787eb6cfc4bbb9332a1ba7515ee0005b3cd7c8f36a40a69d93b010e02276043','first_shot_native_circle_before_C0_1017.py','Diagnostic C0 reader only; it is not admitted into this original parent FIT')):
            historical=reference(AUDIT/backup)
            if historical['sha256']!=old:raise ValueError('Producer successor lost its exact historical bytes')
            current=reference(ROOT/'calibration/program_tracking'/name)
            declarations[(str(Path(current['path']).resolve()),old)]=(current,historical,reason)
        def verify(verifier,ref):
            key=(str(Path(ref['path']).resolve()),ref['sha256'])
            if key not in declarations:return inherited(verifier,ref)
            current,historical,reason=declarations[key]
            if reference(current['path'])!=current or reference(historical['path'])!=historical:
                raise ValueError('Frozen explicit producer transition changed')
            certificate=dict(schema='rinkspace.explicit-source-execution-producer-successor.v1',
                historical_source_path=ref['path'],historical_sha256=ref['sha256'],
                preserved_historical_producer=historical,current_producer=current,entry=reference(__file__),reason=reason,
                historical_state_and_native_observations_not_rewritten=True,
                original_parent_raw_source_replay_required=True,new_target_old_J_H_reuse_forbidden=True,
                other_references_strict=True)
            verifier.transitions[key]=certificate;verifier.verified[key]=current
        CompleteSourceReplay.verify_reference=verify
    factory.install_version_transition=install_producer_successors
    def prepare_manifest():
        ref=legacy.original_prepare();config=read(ref)
        config['actual_full_shot_execution'].update(measured_motion_entry=reference(__file__),
            selected_complete_parent=payload,motion_packet=packet.reference,
            reviewed_existing_print_paths=reviewed_paths_ref,
            source_objective_changed=True,all_original_parameters_and_physical_domain_retained=True,
            old_source_J_H_and_stop_control_not_inherited=True,
            fresh_source_J_every_state=bool(args.fresh_source_jacobian),
            finite_ad_support_restoration=bool(args.finite_ad_coverage),
            original_material_side_evidence=reference(Path(__file__).resolve().parent/'s1_arc_dir_1017.json'),
            ordinary_source_identity_quarantine=reference(Path(__file__).resolve().parent/'s1_arc_q_1017.json'),
            purpose='Same complete original source prefix plus strictly body-admitted actual measured motion')
        write(out/'execution_manifest.json',config);return reference(out/'execution_manifest.json')
    factory.prepare_manifest=prepare_manifest
    if args.manifest_only:
        prepare_manifest();return
    def run_owner(base_owner,parent,manifest):
        base=CompleteContinuation(base_owner,point,reference(__file__));base.guard(base.start)
        factory.export_packet(base,base.start,(397,1125,1180),out/'current_packet')
        # Complete the original Ad7/C0/ordered-source installation first. Its
        # exports and full-W checks refer solely to the unchanged source prefix.
        original_runner_factory=factory.make_runner
        hook=load('measured_motion_original_source_hook',AUDIT/'motion_joint_v5/arm.py')
        hook.HERE=out;hook.C0_PATH=out/'combined_c0.json';hook.revised_values.OUT=out
        from calibration.program_tracking.first_shot_ordered_successor import reviewed_ordered_successor
        ordered_transition=reference(Path(__file__).resolve().parent/'s1_ordered_1017c.json')
        original_identity_revision=hook.source_revision
        ordinary_quarantine=reference(Path(__file__).resolve().parent/'s1_arc_q_1017.json')
        material_side=reference(Path(__file__).resolve().parent/'s1_arc_dir_1017.json')
        def source_with_ordinary_identity(current,*,entry_reference):
            original=original_identity_revision(current,entry_reference=entry_reference)
            from calibration.program_tracking.first_shot_ordinary_quarantine import install_ordinary_quarantine
            from calibration.program_tracking.first_shot_safe_frame_export import _encode
            receipt,proof=install_ordinary_quarantine(current,ordinary_quarantine)
            arrays={};encoded=_encode(proof['archive'],arrays)
            np.savez_compressed(out/'inactive_ordinary_rows.npz',**arrays)
            write(out/'inactive_ordinary_rows.json',dict(payload=encoded,arrays=reference(out/'inactive_ordinary_rows.npz'),
                original_mixed_identity_rows_UV_ID_sigma_and_user_history_preserved=True))
            np.savez_compressed(out/'ordinary_retained_mappings.npz',formal_indices=proof['retained_formal_indices'],
                source_point_indices=proof['retained_source_point_indices'])
            receipt.update(archived_original_rows=reference(out/'inactive_ordinary_rows.json'),
                retained_mapping=reference(out/'ordinary_retained_mappings.npz'))
            write(out/'ordinary_identity_QA.json',dict(passed=True,**receipt))
            for obj in hook.objects(current):
                previous=vars(obj).get('receipt',{})
                if previous.get('schema')=='rinkspace.programme-source-wide-reference-only.v1':
                    obj.receipt=dict(previous,previous_source_scope=deepcopy(previous),programme_formal_rows=len(current.programme.scales),
                        ordinary_identity_quarantine=deepcopy(receipt))
            combined=dict(original,ordinary_identity_quarantine=receipt)
            current.source_identity_revision=deepcopy(combined)
            return combined
        hook.source_revision=source_with_ordinary_identity
        original_source_install=hook.install_owner
        def source_with_reviewed_order(current,*,entry_reference):
            from calibration.program_tracking.first_shot_ordered_contours import bind_alias_lineage
            lineage=bind_alias_lineage(current.programme.bundle.frames,current.programme.bundle.state.binding_reference,
                allowed_targets=((5,'component017_inner'),))
            write(out/'ordered_point_lineage.json',lineage)
            from calibration.program_tracking.first_shot_arc_polarity import bind_material_side_evidence
            polarity=bind_material_side_evidence(current.programme.bundle.frames,material_side)
            write(out/'original_material_side_evidence.json',polarity)
            physical,_=current.effective_q_and_chain(current.start)
            with reviewed_ordered_successor(ordered_transition,physical_q_sha256=sha(np.asarray(physical)[:current.programme_count])) as certificates:
                result=original_source_install(current,entry_reference=entry_reference)
            write(out/'ordered_policy_transition.json',dict(passed=True,certificates=certificates))
            for name in ('source_producers','recovery_references','_references'):
                if hasattr(current,name):
                    for ref in (ordered_transition,material_side,ordinary_quarantine,reference(ROOT/'calibration/program_tracking/first_shot_arc_polarity.py'),reference(ROOT/'calibration/program_tracking/first_shot_ordinary_quarantine.py'),reference(ROOT/'calibration/program_tracking/first_shot_ordered_successor.py')):
                        if ref not in getattr(current,name):getattr(current,name).append(deepcopy(ref))
            return dict(result,explicit_ordered_policy_successor=certificates)
        hook.install_owner=source_with_reviewed_order
        if reviewed_paths_ref is None:
            receipt=hook.install_owner(base,entry_reference=reference(__file__))
        else:
            # The existing source hook installs all historical successors before
            # any pool. Append only the explicit same-bank membership correction
            # at that boundary; its normal world/source workers need no new math.
            from calibration.program_tracking import first_shot_ad7_source_arcs as arcs
            original_successor=arcs.install_successor
            def with_membership(current,*,mapping_reference):
                original=original_successor(current,mapping_reference=mapping_reference)
                additional=membership.install_successor(current,mapping_reference=reviewed_paths_ref)
                write(out/'reviewed_print_path_membership.json',additional)
                return dict(original,additional_existing_path_membership=additional)
            arcs.install_successor=with_membership
            try:receipt=hook.install_owner(base,entry_reference=reference(__file__))
            finally:arcs.install_successor=original_successor
        owner=MeasuredBoardMotionObjective(base,packet,entry_reference=reference(__file__))
        restored_arrays=owner.checkpoint_arrays()
        for key,value in point['complete_parent_aux'].items():
            factory.qa.same(restored_arrays[key],value,'complete selected joint auxiliary '+key)
        if point['restored_motion_aux']and set(restored_arrays)!=set(point['complete_parent_aux']):
            raise ValueError('Complete restored joint auxiliary inventory changed')
        write(out/'status.json',dict(stage='checking_unchanged_original_source_prefix',pid=os.getpid(),utc=time.time()))
        from calibration.solver_runtime.source_loss import loss_weight_arrays,weighted_source_cost
        blocks=list(base.source_blocks(base.start,False));prefix=np.concatenate([b.r for b in blocks])
        ww=[];nn=[]
        for block in blocks:
            w,n=loss_weight_arrays(block.loss_weights,len(block.r))
            ww.append(np.ones(len(block.r))if w is None else w)
            nn.append(np.ones(len(block.r))if n is None else n)
        with np.load(out/'before_source_ordered/source_arrays.npz',allow_pickle=False)as before:
            for actual,key in ((prefix,'r'),(base.scales,'scales'),(base.quadratic,'quadratic'),
                    (np.concatenate(ww),'source_loss_integration_weights'),
                    (np.concatenate(nn),'source_loss_working_normalization_multiplier')):
                factory.qa.same(actual,before[key],'unchanged actual source prefix '+key)
        write(out/'status.json',dict(stage='checking_all_admitted_motion_domains',pid=os.getpid(),utc=time.time()))
        owner.guard(owner.start)
        write(out/'status.json',dict(stage='checking_real_motion_directional_derivative',pid=os.getpid(),utc=time.time()))
        rng=np.random.default_rng(1017);d=np.zeros(len(owner.start))
        free=np.setdiff1d(np.arange(len(d)),owner.fixed);d[free]=rng.normal(size=len(free));d/=np.linalg.norm(d)
        probe_epsilon=1e-6
        if point['probe_reference_direction'] is not None:
            d=np.asarray(point['probe_reference_direction']).copy();probe_epsilon=1e-4
        np.savez(out/'directional_probe.npz',q=owner.start,direction=d,epsilon=np.asarray(probe_epsilon))
        try:
            derivative=owner.directional_check(owner.start,direction=d,epsilon=probe_epsilon,
                fallback_direction=point['probe_reference_direction'])
        finally:
            used=getattr(owner,'last_directional_direction',None)
            np.savez(out/'directional_probe.npz',q=owner.start,direction=d,
                used_direction=np.asarray(used)if used is not None else np.empty(0),
                epsilon=np.asarray(probe_epsilon))
            probe=getattr(owner,'last_directional_probe',None)
            if probe is not None:write(out/'directional_probe_attempts.json',probe)
        passed=derivative['max_absolute']<1e-3 and derivative['relative_norm']<1e-4
        motion=owner._motion_block(owner.start,False)
        prefix_cost=sum(weighted_source_cost(b.r,b.quadratic,b.loss_weights)for b in blocks)
        motion_cost=weighted_source_cost(motion.r,motion.quadratic,motion.loss_weights)
        write(out/'joint_preflight.json',dict(passed=passed,original_prefix_rows=len(prefix),
            motion_rows=len(motion.r),complete_rows=len(owner.scales),
            original_source_r_W_masks_scales_byte_exact=True,derivative=derivative,
            equality_reference='same-policy source hook export at restored q; historical scores not directly comparable',
            explicit_reviewed_print_path_policy=reviewed_paths_ref,
            prefix_cost=prefix_cost,motion_cost=motion_cost,combined_cost=prefix_cost+motion_cost,
            complete_auxiliary_count=len(owner.checkpoint_arrays()),inherited406_auxiliary_unchanged=True,
            original_full_guard_and_all_motion_first_hit_domains_passed=True,
            new_full_source_J_constructed=False,source_hook=receipt,motion_packet=packet.reference,
            accuracy_accepted=False))
        np.savez_compressed(out/'preflight_complete_state.npz',**owner.complete_state_payload(owner.start),
            damping=np.asarray(point['damping']),next_line_search_fraction=np.asarray(point['fraction']))
        if not passed:raise ValueError('Complete-chart actual admitted-motion directional derivative check failed')
        if not args.fit:return
        # Joint wrapper precedes runner, reuse policy, prediction policy and the
        # complete physical-domain adapter. Both true costs contain motion.
        prefix_runner=factory.make_runner(base,manifest)
        # The historical hook intentionally memoizes its prefix runner. Build
        # the joint runner with the original factory, reusing authenticated
        # immutable media metadata instead of returning that prefix-only owner.
        from calibration.program_tracking import first_shot_joint_stage_runtime as runner_runtime
        original_inputs=runner_runtime.export_complete_review_inputs
        def verified_inputs(programme,output):
            if programme is not base.programme or Path(output)!=out:
                raise ValueError('Only the exact already-exported programme/media may be reused')
            if reference(prefix_runner.media['path'])!=prefix_runner.media:
                raise ValueError('Authenticated original source media changed')
            return prefix_runner.inventory,prefix_runner.media
        runner_runtime.export_complete_review_inputs=verified_inputs
        try:runner=original_runner_factory(owner,manifest)
        finally:runner_runtime.export_complete_review_inputs=original_inputs
        if runner.owner is not owner:raise ValueError('Joint runner is still bound to the prefix-only owner')
        factory.make_runner=lambda current,manifest:runner
        def scope(this):return dict(programme_source_objective=base.receipt,
            measured_board_motion=owner.motion_receipt,original_source_prefix_rows=len(base.scales),
            motion_rows=packet.rows,wide_source_role='historical/reference only',
            objective_is_complete_official50_programme_plus_admitted_motion=True)
        runner._scope=MethodType(scope,runner)
        import z190_joint_source_solver as solver
        from calibration.program_tracking.first_shot_jacobian_reuse import install_source_jacobian_reuse
        from calibration.program_tracking.first_shot_prediction_refresh import install_prediction_refresh
        from calibration.program_tracking.first_shot_full_physical import install
        reuse=install_source_jacobian_reuse(owner,cache_directory=runner.cache,entry_reference=reference(__file__),
            reuse_source_jacobian=not args.fresh_source_jacobian)
        driver=SimpleNamespace(solve_joint_source=solver.solve_joint_source,solver_runtime=solver)
        if args.fresh_source_jacobian:
            from calibration.program_tracking.first_shot_prediction_refresh import compile_quality_solver
            from calibration.solver_runtime import schur_rejection_retry
            driver.solve_joint_source=compile_quality_solver(solver.solve_joint_source,retry_rejected_schur=True)
            prediction=dict(schema='rinkspace.fresh-source-original-LM-retry.v1',
                fresh_complete_source_J_each_state=True,lagged_prediction_refresh_not_installed=True,
                rejected_exact_Schur_direction_retries_existing_LM=True,
                saved_q_aux_damping_and_fraction_unchanged=True,
                producers=[reference(ROOT/'calibration/program_tracking/first_shot_prediction_refresh.py'),reference(schur_rejection_retry.__file__),reference(__file__)])
            for name in ('source_producers','recovery_references','_references'):
                if hasattr(owner,name):
                    for ref in prediction['producers']:
                        if ref not in getattr(owner,name):getattr(owner,name).append(deepcopy(ref))
        else:
            prediction=install_prediction_refresh(owner,driver=driver,entry_reference=reference(__file__),
                minimum_prediction_ratio=.25,reset_initial_fresh_source_fraction=False)
        coverage=None;coverage_receipt=None
        if args.finite_ad_coverage:
            from calibration.program_tracking.first_shot_finite_ad_coverage import FiniteAdCoveragePolicy
            coverage=FiniteAdCoveragePolicy(owner);coverage_receipt=coverage.install(reference(__file__))
            finite_direction=point['probe_reference_direction']
            if finite_direction is None:raise ValueError('Finite support QA requires the actual prior camera/world direction')
            finite_QA=coverage.directional_check(owner.start,finite_direction)
            write(out/'finite_ad_directional_QA.json',finite_QA)
            if not finite_QA['passed']:raise ValueError('Actual full finite-ad support camera/world derivative failed')
            initial_coverage=coverage.evaluate(owner.start)
            write(out/'finite_ad_initial.json',dict(policy=coverage_receipt,
                maximum_uncovered_px=float(initial_coverage['group_violations'].max(initial=0.)),
                group_violations_px=initial_coverage['group_violations'].tolist(),records=initial_coverage['records']))
        physical_policy=install(owner,driver=driver,entry_reference=reference(__file__),
            domain_builder=None if coverage is None else coverage.domain)
        kwargs=owner.solver_kwargs();kwargs['source_block_factory']=runner.source_blocks
        anchor_receipt=None;anchor_validations=[]
        def save_full_source_QA(report):
            write(out/'full_source_directional_QA.json',report)
            if anchor_receipt is not None:
                write(out/'same_source_anchor_actual_QA.json',dict(passed=bool(report['passed'] and anchor_validations),
                    setup=anchor_receipt,actual_block_checks=anchor_validations,
                    full_source_directional_QA=reference(out/'full_source_directional_QA.json')))
        if args.check_complete_source_J:
            from calibration.program_tracking.first_shot_source_directional_QA import checked_source_factory
            direction=point['probe_reference_direction']
            if direction is None:raise ValueError('Complete source QA requires the actual prior accepted joint direction')
            kwargs['source_block_factory']=checked_source_factory(owner,runner.source_blocks,
                direction=direction,epsilon=2e-5,confirmation_epsilon=1e-5,
                write_report=save_full_source_QA)
            qa_ref=reference(ROOT/'calibration/program_tracking/first_shot_source_directional_QA.py')
            owner.source_producers.append(qa_ref);owner.recovery_references.append(qa_ref)
            kwargs['checkpoint_identity']['full_source_directional_QA']=dict(producer=qa_ref,
                required_before_first_trial=True,all_current_formal_rows_and_complete_W_retained=True,
                primary_epsilon=2e-5,confirmation_epsilon=1e-5,
                frozen_branch_crossing_diagnosis=reference(AUDIT/'source10_branch697.json'))
        if args.same_state_source_J is not None:
            if not args.check_complete_source_J:raise ValueError('Exact source anchor requires both full-source derivative stencils before trial')
            from calibration.program_tracking.first_shot_exact_source_anchor import install_exact_source_anchor
            transitions=[]
            for path,backup,reason in (
                (Path(__file__),AUDIT/'fit_measured_motion_body12.py','Constraint world-interface correction, its QA and exact-state CSR loader only; source R/J producers unchanged'),
                (ROOT/'calibration/program_tracking/first_shot_finite_ad_coverage.py',AUDIT/'finite_ad_before_world_fix.py','Use the actual programme world171 for support constraints; original source likelihood and its J unchanged')):
                transitions.append(dict(historical_path=str(path.resolve()),historical_producer=reference(backup),
                    current_producer=reference(path),reason=reason))
            anchor_receipt,anchor_validations=install_exact_source_anchor(owner,reference(args.same_state_source_J),transitions=transitions,source_prefix_owner=base)
            write(out/'same_source_anchor_install.json',dict(authenticated_setup=True,actual_block_checks_pending=True,**anchor_receipt))
            kwargs['checkpoint_identity'].update(exact_source_anchor=anchor_receipt,initial_exact_complete_current_state_full_J_reuse=True)
        kwargs['checkpoint_identity'].update(execution_manifest=manifest,measured_motion_entry=reference(__file__))
        public=out/'public_solver';public.mkdir()
        write(public/'control.json',dict(stop_branch=False,reason='New admitted-motion objective; old stop state not inherited'))
        write(out/'run_identity.json',kwargs['checkpoint_identity'])
        write(out/'policy_install.json',dict(reuse=reuse,prediction=prediction,physical=physical_policy,
            motion=owner.motion_receipt,source_prefix=receipt,finite_ad_support=coverage_receipt,fresh_combined_J_required=True))
        write(out/'status.json',dict(stage='admitted_motion_joint_optimizer_running',pid=os.getpid()))
        result=driver.solve_joint_source(owner.evaluate,owner.start,owner.quadratic,owner.scales,public,owner.guard,
            checkpoint_callback=runner.checkpoint,source_stage_timing=True,verbose=True,**kwargs)
        # The public final save may update damping after a rejected direction;
        # retain its exact final q, complete auxiliary inventory and search seed.
        final_canonical=reference(result.checkpoint)
        with np.load(final_canonical['path'],allow_pickle=False)as saved:
            factory.qa.same(saved['q'],result.q,'final public numerical q')
            final_aux={k[5:]for k in saved.files if k.startswith('aux__')}
            expected_aux=owner.checkpoint_arrays()
            if final_aux!=set(expected_aux):raise ValueError('Final complete auxiliary inventory differs')
            for key,value in expected_aux.items():factory.qa.same(saved['aux__'+key],value,'final joint auxiliary '+key)
            final_damping=float(saved['damping']);final_fraction=float(saved['next_line_search_fraction'])
        np.savez_compressed(out/'final_complete_state.npz',**owner.complete_state_payload(result.q),
            damping=np.asarray(final_damping),next_line_search_fraction=np.asarray(final_fraction))
        # The historical runner owns final same-model geometry, full native row
        # cameras and ordered/nearest original-source captures. Its owner is the
        # prefix adapter, so appended motion never enters W, point inventories or
        # original rendering. Complete joint state is saved separately above.
        if prefix_runner.owner is not base:raise ValueError('Final prefix exporter owner changed')
        prefix_runner.export_review(result.q,'final')
        if coverage is not None:
            final_coverage=coverage.evaluate(result.q)
            write(out/'finite_ad_final.json',dict(policy=coverage_receipt,
                maximum_uncovered_px=float(final_coverage['group_violations'].max(initial=0.)),
                group_violations_px=final_coverage['group_violations'].tolist(),records=final_coverage['records']))
        final_motion=owner._motion_block(result.q,False)
        np.savez_compressed(out/'final_motion.npz',r=final_motion.r,q=result.q,
            packet_identity_json=np.asarray(json.dumps(packet.receipt,sort_keys=True)))
        write(out/'result.json',dict(status=result.status,stop_reason=result.stop_reason,q_sha256=sha(result.q),
            accepted_iterations=sum(r['kind']=='accepted'for r in runner.records),accuracy_accepted=False,
            video_generated=False,motion_packet=packet.reference,
            complete_state=reference(out/'final_complete_state.npz'),canonical_checkpoint=final_canonical,
            damping=final_damping,next_line_search_fraction=final_fraction,
            original_source_exports=reference(out/'final_source_exports.json'),
            same_model_geometry=reference(out/'actual_geometry_final/completion.json')))
    factory.run_owner=run_owner
    sys.argv=[str(Path(factory.__file__)),'--serve'];factory.main()


if __name__=='__main__':main()
