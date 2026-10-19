"""Compose authenticated Board185/86-truth continuation; no graph or solver."""
from copy import deepcopy
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from calibration.solver_runtime import install
install()
from structured_programme_source_candidate.human_stroke_truth import human_support_operator_contract
from structured_programme_source_candidate.reviewed_manual_append_restore import _checked_truth_operator_revision
from types import SimpleNamespace
SYS=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
BASE=SYS.parents[2]/'calibration/runs/solver_acceleration_20261003'
PROGRAMME=ROOT/'events/ankara_2026/programmes/okada_mei_sp'
NEXT=PROGRAMME/'full_source_review86_same757_board185_constant_axes_20261011.json'
PREFIX='q757_truth86_board185_exact_constant_axes_20261011'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def checked(v):
    assert sha(v['path'])==v['sha256'];return Path(v['path'])
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-ice-pixel-correction-source',type=Path)
    parser.add_argument('--original-board-curve-integration-policy',type=Path)
    parser.add_argument('--original-source-input-contract-audit',type=Path)
    opt=parser.parse_args()
    if (opt.original_ice_pixel_correction_source is None)!=(opt.original_board_curve_integration_policy is None):
        parser.error('A reviewed original pixel revision and its actual curve integration policy must be supplied together')
    assembly=read(SYS/'native_board185_same757_constant86_full_source_config_assembly_and_diff_20261011.json')
    config=read(checked(assembly['skeleton']))
    template=read(checked(assembly['template']))
    oldscope=read(template['source_options']['--truth-restoration-reviewed-manual-append-audit'])
    audit=deepcopy(read(checked(oldscope['same_q_public_audit'])))
    PROOF=checked(assembly['new_constant86_actual_proof']);proof=read(PROOF)
    APPEND=checked(assembly['actual185_public_R_J_weights_and_protected_proof']);append=read(APPEND)
    PP=SYS/'actual_same_q757_PP6_existing_engineering_prior_inheritance_20261011.json'
    contract=human_support_operator_contract()
    assert proof['passed'] and proof['operator_contract']==contract and proof['group_count']==86
    assert append['passed'] and append['passed_R_J'] and append['complete_q_sha256']==proof['complete_q_sha256']
    assert read(PP)['passed'] and read(PP)['complete_q_sha256']==proof['complete_q_sha256']
    parent=read(checked(proof['original_restoration_caps']));old=read(checked(proof['original_runtime_manifest']))
    assert old['source_rows']==4501970
    options=config['source_options'];board=checked(assembly['board']);policy=checked(assembly['policy137'])
    soft_revision=None
    if opt.original_ice_pixel_correction_source is not None:
        corrected=opt.original_ice_pixel_correction_source.resolve();integrated=opt.original_board_curve_integration_policy.resolve()
        # Actual source qualification/UV/budget proofs remain in these derived
        # packets and their public installers. This transaction records the
        # source target change; it does not fabricate a complete actor audit.
        correction=read(corrected)
        if correction.get('schema')!='rinkspace.original-native-ice-pixel-correction.v1':
            raise ValueError('The existing authenticated original pixel correction packet is required')
        read(integrated)
        if opt.original_source_input_contract_audit is None:
            raise ValueError('The actual revised original RGB input installer receipt is required')
        input_proof=read(opt.original_source_input_contract_audit)
        if(input_proof.get('schema')!='rinkspace.actual-original-TOP-revised-RGB-material-contract-audit.v1'
            or input_proof.get('passed')is not True
            or input_proof.get('complete_q_sha256')!=proof['complete_q_sha256']
            or checked(input_proof['correction'])!=corrected):
            raise ValueError('Revised original source installer proof belongs to another source or q')
        soft_revision=dict(original_pixel_correction_before=ref(options['--original-ice-pixel-correction-source']),
            original_pixel_correction_after=ref(corrected),
            original_board_curve_integration_before=ref(options['--original-board-curve-integration-policy']),
            original_board_curve_integration_after=ref(integrated),
            actual_original_RGB_input_contract_proof=ref(opt.original_source_input_contract_audit),
            source_only_original_RGB_pixel_revision=True,complete_source_scalar_slots_expected_unchanged=True,
            current_full_original_source_installers_and_all86_startup_guards_still_required=True,
            all86_protected_source_packets_and_original_parent_caps_unchanged=True)
        options['--original-ice-pixel-correction-source']=str(corrected)
        options['--original-board-curve-integration-policy']=str(integrated)
    assert Path(options['--additional-board-source']).resolve()==board.resolve()
    assert Path(options['--native-base-curve-integration-policy']).resolve()==policy.resolve()
    transaction=SYS/(PREFIX+'_transaction.json')
    write(transaction,dict(schema='rinkspace.same-q-protected-truth-semantic-and-measured-native-source-append-transaction.v1',
        passed=True,actual_all86_semantic_proof=ref(PROOF),actual_board185_source_and_guard_proof=ref(APPEND),
        PP6_inheritance_proof=ref(PP),parent_truth_state=proof['original_restoration_caps'],
        parent_source_manifest=proof['original_runtime_manifest'],complete_state_reference=proof['complete_state_reference'],
        source_rows_before=4501970,source_rows_after=4502414,new_scalar_rows=444,new_human_groups=0,
        new_direct_manual_centre_rows=0,original86_source_supports_uncertainties_and_roles_unchanged=True,
        cap_policy='minimum_parent_cap_and_actual_revised_same_q_violation',operator_contract=contract,
        measured_board185_source=ref(board),native_base137_curve_integration_policy=ref(policy),
        reviewed_soft_original_RGB_source_revision=soft_revision,
        fresh_source_J_and_profiles_required=True,old_query_caps_not_claimed_semantically_equivalent=True,
        source_objective_changed_explicitly=True,complete_new_actor_must_validate_source_count_and_truth_values=True,
        solver_started=False,accuracy_accepted=False))
    families={family:dict(group_ids=row['group_ids'],old_count=len(row['group_ids']),
        violations_px=row['corrected_actual_same_q_violation_px'],limits=row['proposed_ratchet_only_corrected_semantic_caps_px'],
        uncertainties=row['original_uncertainties_exact'])for family,row in proof['families'].items()}
    manual=deepcopy(audit['new_source_options'])
    # Preserve authenticated original RAW ice ownership; actual qualified2db
    # is already explicitly admitted through source_qualified_native_packet_sources.
    for key in ('--additional-board-source','--direct-human-truth-authority','--direct-human-full-stroke-guard',
                '--direct-human-stroke-cap-support','--practical-human-location-envelope','--user-confirmed-source-truth-authority'):
        manual[key]=ref(options[key])
    audit.update(complete_q_sha256=proof['complete_q_sha256'],complete_state_reference=proof['complete_state_reference'],
        old_caps_reference=proof['original_restoration_caps'],parent_manifest_reference=proof['original_runtime_manifest'],
        total_added_source_scalars=444,added_direct_manual_centre_scalars=0,new_complete_source_scalar_count=4502414,
        parent_group_count=86,new_group_count=86,families=families,new_source_options=manual,
        current_source_target_transaction=ref(transaction),actual_same_q86_invariance_proof=ref(PROOF),
        actual_same_q_protected_native_source_append=ref(APPEND),engineering_prior_append_reference=ref(PP),
        existing86_caps_uncertainties_group_ids_exact=False,existing86_source_support_uncertainties_group_ids_exact=True,
        finite_truth_semantic_revision_not_numerical_accuracy_improvement=True,
        complete_new_actor_still_must_recompute_actual_same_q86_and_source_count=True)
    audit_path=SYS/(PREFIX+'_startup_audit.json');write(audit_path,audit)
    values_path=SYS/(PREFIX+'_caps_values.json')
    write(values_path,dict(schema='rinkspace.exact-parent-plus-current-source-cap-values.v1',
        complete_q_sha256=proof['complete_q_sha256'],audit=ref(audit_path),families=families))
    scope=deepcopy(oldscope);scope.update(same_q_public_audit=ref(audit_path),exact_caps_values=ref(values_path),
        current_source_target_transaction=ref(transaction),
        admitted_engineering_prior_append=dict(actual_same_q_public_audit=ref(PP),rows=6,new_rows_in_this_revision=0),
        admitted_truth_support_operator_revision=dict(actual_same_q_public_audit=ref(PROOF),contract=contract,
            cap_policy='minimum_parent_cap_and_actual_revised_same_q_violation'),
        admitted_truth_preserving_native_source_append=dict(actual_same_q_public_audit=ref(APPEND),source_scalar_delta=444),
        all86_existing_human_groups_and_caps_preserved=False,all86_source_supports_and_group_ids_preserved=True)
    scope['admitted_operator_sources']['native_base_curve_integration']['reference']=ref(policy)
    if soft_revision is not None:
        scope['admitted_operator_sources']['original_ice_pixel_correction']['reference']=soft_revision['original_pixel_correction_after']
        scope['admitted_operator_sources']['original_board_curve_integration']['reference']=soft_revision['original_board_curve_integration_after']
    binding=SYS/(PREFIX+'_startup_binding.json');write(binding,scope)
    with np.load(checked(proof['complete_state_reference']),allow_pickle=False)as z:
        q=z['q'].copy();damping=float(z['damping']);fraction=float(z['next_line_search_fraction']);fields=len(z.files)
    assert hashlib.sha256(q.tobytes()).hexdigest()==proof['complete_q_sha256']
    options.update({'--camera-seed':proof['complete_state_reference']['path'],
        '--truth-restoration-state':proof['original_restoration_caps']['path'],
        '--truth-restoration-reviewed-manual-append-audit':str(binding),'--initial-damping':damping})
    # One existing initial-cache export preserves the actual H/g and their
    # original frozen r/J identity for follow-up diagnosis without recalculating
    # the complete programme. The driver streams its already generated arrays.
    options['--save-initial-linearization-cache']=True
    options.pop('--accepted-steps',None)
    config['source_manifest_sha256']=sha(ROOT/'calibration/solver_runtime/source_manifest.json')
    config.pop('startup_binding_pending_actual_same_q_proofs',None)
    config.pop('production_startup_authorized_by_this_skeleton',None)
    config['preparation_status']=dict(final_programme_configuration=True,ready_to_start=True,
        actual_saved_public_receipt_combination_gate_passed=True,
        actual_complete_startup_source_count_and_all86_validation_still_required=True,
        historical_skeleton_status_not_used_as_execution_authority=True)
    config['continuation']=dict(complete_accepted_state=proof['complete_state_reference']['path'],
        checkpoint_file_sha256=proof['complete_state_reference']['sha256'],truth_restoration_state=proof['original_restoration_caps']['path'],
        truth_restoration_state_sha256=proof['original_restoration_caps']['sha256'],
        state=dict(q_sha256=proof['complete_q_sha256'],parameters=len(q),fields=fields,damping=damping,next_line_search_fraction=fraction),
        previous_input_configuration=assembly['template'],parent_source_input_manifest=proof['original_runtime_manifest'],
        expected_complete_active_source_scalars=4502414,current_truth_group_count=86,
        current_source_target_transaction=ref(transaction),reviewed_source_target_startup_binding=ref(binding),
        actual_same_q86_proof=ref(PROOF),actual_same_q_board185_and_protected_truth_proof=ref(APPEND),
        PP6_same_q_existing_prior_proof=ref(PP),next_start_requires_fresh_source_J=True,
        no_cached_J_H_g_or_prefix_imported=True,original_supports_uncertainties_and_world_unchanged=True,
        cap_semantic_revision_is_not_geometric_accuracy_acceptance=True,solver_run_started=False,accuracy_accepted=False)
    # Replay the bounded combination gate with real saved public receipts.
    # This carrier is NOT claimed to be a newly assembled complete source actor.
    manifest=deepcopy(old);manifest.update(human_truth_support_operator=contract)
    manifest['source_contract']['human_truth_support_operator']=contract
    manifest['additional_real_native_board_edges'].update(packet_path=str(board),packet_sha256=sha(board),scalar_rows=1310)
    factors={f:SimpleNamespace(contract=dict(human_truth_support_operator=contract))for f in ('native','confirmed')}
    _checked_truth_operator_revision(scope,audit,parent,old,manifest,factors)
    for key,entry in scope['admitted_operator_sources'].items():checked(entry['reference'])
    for value in manual.values():checked(value)
    assert read(values_path)['audit']==ref(audit_path) and read(binding)['same_q_public_audit']==ref(audit_path)
    write(NEXT,config)
    print(json.dumps(dict(config=ref(NEXT),startup_binding=ref(binding),transaction=ref(transaction),
        actual_receipt_combination_gate_passed=True,source_rows_expected=4502414,
        actual_complete_startup_source_count_and_all86_validation_still_required=True,
        solver_started=False),ensure_ascii=False))
if __name__=='__main__':main()
