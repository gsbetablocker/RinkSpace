"""Replay real unchanged-support proofs for the combined source revision."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from calibration.solver_runtime import install
install()
from structured_programme_source_candidate.reviewed_manual_append_restore import _checked_truth_operator_revision
from structured_programme_source_candidate.human_stroke_truth import human_support_operator_contract
from scripts.diagnostics.rebind_board185_protected_sources import SYS,ref,read,save

PROOF=SYS/'same_q757_original86_rectangle_exact_constant_axes_actual_audit_20261011.json'
APPEND=SYS/'native_board185_q757_actual_R_J_integration_and_protected_guards_20261011.json'
HANDOFF=SYS/'native_board185_protected_source_options_Handoff_20261011.json'
pytestmark=pytest.mark.skipif(not APPEND.exists()or not PROOF.exists(),reason='Actual combined public-factor receipts unavailable')

def fixture(tmp_path):
    proof=read(PROOF);current=human_support_operator_contract()
    actual=read(SYS/'q757_all86_constant_axes_A_precontraction_actual_20261011.json')
    assert actual['source_arrays_before_after_public_J_bitexact'] and actual['operator_contract']==current
    assert proof['operator_contract']==current
    p=PROOF
    parent=read(proof['original_restoration_caps']['path']);old=read(proof['original_runtime_manifest']['path'])
    options=read(HANDOFF)['source_options'];append=read(APPEND)
    audit=dict(complete_q_sha256=proof['complete_q_sha256'],old_caps_reference=proof['original_restoration_caps'],
        parent_manifest_reference=proof['original_runtime_manifest'],complete_state_reference=proof['complete_state_reference'],
        total_added_source_scalars=444,added_direct_manual_centre_scalars=0,
        new_source_options={k:ref(options[k])for k in ('--additional-board-source','--direct-human-truth-authority',
            '--direct-human-full-stroke-guard','--direct-human-stroke-cap-support','--practical-human-location-envelope',
            '--user-confirmed-source-truth-authority')})
    manifest=deepcopy(old);manifest['human_truth_support_operator']=current;manifest['source_contract']['human_truth_support_operator']=current
    manifest['additional_real_native_board_edges'].update(packet_path=options['--additional-board-source'],
        packet_sha256=append['candidate']['sha256'],scalar_rows=1310)
    scope=dict(admitted_truth_support_operator_revision=dict(actual_same_q_public_audit=ref(p),contract=current,
        cap_policy='minimum_parent_cap_and_actual_revised_same_q_violation'),
        admitted_truth_preserving_native_source_append=dict(actual_same_q_public_audit=ref(APPEND),source_scalar_delta=444))
    factors={f:SimpleNamespace(contract=dict(human_truth_support_operator=current))for f in ('native','confirmed')}
    return scope,audit,parent,old,manifest,factors

def test_actual_board185_append_preserves_same86_hard_semantic_caps(tmp_path):
    values=fixture(tmp_path)
    path,proof=_checked_truth_operator_revision(*values)
    assert path.exists() and proof['group_count']==86
    for family,row in proof['families'].items():
        assert row['group_ids']==values[2]['families'][family]['group_ids']
        assert all(new<=old for new,old in zip(row['proposed_ratchet_only_corrected_semantic_caps_px'],row['original_caps_px']))

@pytest.mark.parametrize('change',('delta','ice','centre','missing_proof','different_q','changed_guard_receipt'))
def test_combination_rejects_an_unproved_source_or_protection_change(tmp_path,change):
    values=list(fixture(tmp_path));scope,audit,parent,old,manifest,factors=values
    if change=='delta':audit['total_added_source_scalars']+=2
    elif change=='ice':manifest['additional_real_native_ice_curves']['new_scalar_rows']+=2
    elif change=='centre':audit['added_direct_manual_centre_scalars']=2
    elif change=='missing_proof':del scope['admitted_truth_preserving_native_source_append']
    elif change=='different_q':audit['complete_q_sha256']='0'*64
    else:
        proof=read(APPEND);proof['original54_human_primary_full_all_queries_bounds_carriers_bitexact']=False
        p=tmp_path/'invalid_protected_receipt.json';save(p,proof)
        scope['admitted_truth_preserving_native_source_append']['actual_same_q_public_audit']=ref(p)
    with pytest.raises(ValueError):_checked_truth_operator_revision(*values)
