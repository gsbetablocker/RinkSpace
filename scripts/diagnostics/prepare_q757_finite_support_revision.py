"""Bind the actual saved q757 semantic audit; no graph or numerical solve."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from calibration.solver_runtime import install
install()
from structured_programme_source_candidate.reviewed_manual_append_restore import prepare_reviewed_manual_append_state
from structured_programme_source_candidate.truth_source_role_migration import role_target,target
from structured_programme_source_candidate.human_stroke_truth import human_support_operator_contract

SYS=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
PROOF=SYS/'same_q757_original86_finite_rectangle_semantics_endpoint_face_final_actual_audit_20261011.json'
PP=SYS/'actual_same_q757_PP6_existing_engineering_prior_inheritance_20261011.json'
CONFIG=ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_same971b_joint_curve_integration_20261010.json'
NEXT=CONFIG.with_name('full_source_review86_same757_original_support_rectangle_20261011.json')
PREFIX='q757_truth86_original_support_rectangle_semantic_revision_20261011'

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def actual_receipt_carriers(directory):
    scope=read(SYS/(PREFIX+'_startup_binding.json'))
    audit=read(scope['same_q_public_audit']['path']);values=read(scope['exact_caps_values']['path'])
    parent=read(audit['old_caps_reference']['path']);manifest=deepcopy(read(audit['parent_manifest_reference']['path']))
    contract=human_support_operator_contract()
    manifest['human_truth_support_operator']=contract
    manifest['source_contract']['human_truth_support_operator']=contract
    manifest['physical_domain_initial_q_sha256']=audit['complete_q_sha256']
    manifest['highest_direct_human_hard_truth']['native']['human_truth_support_operator']=contract
    manifest['highest_user_confirmed_native_sources']['human_truth_support_operator']=contract
    factors={}
    for family,row in values['families'].items():
        if family=='native':bound=deepcopy(manifest['highest_direct_human_hard_truth']['native'])
        elif family=='confirmed':bound=deepcopy(manifest['highest_user_confirmed_native_sources'])
        else:bound=dict(authority=parent['families'][family]['authority'])
        factors[family]=SimpleNamespace(contract=bound,ids=row['group_ids'],
            uncertainties=np.asarray(row['uncertainties'],float),violations=np.asarray(row['violations_px'],float))
    with np.load(audit['complete_state_reference']['path'],allow_pickle=False)as z:q=z['q'].copy()
    path=Path(directory)/'SAVED_RECEIPT_CARRIER_NOT_PRODUCTION_MANIFEST.json';write(path,manifest)
    return SimpleNamespace(scope=scope,audit=audit,parent=parent,manifest=manifest,manifest_path=path,q=q,factors=factors,
        binding=SYS/(PREFIX+'_startup_binding.json'),parent_path=Path(audit['old_caps_reference']['path']))

def invoke(f):
    captured={}
    output=prepare_reviewed_manual_append_state(f.binding,f.parent_path,f.manifest_path,f.manifest,f.q,f.factors,
        group_ids=lambda factor,family:factor.ids,
        violation_buffer=lambda factor,q,family:factor.violations,
        uncertainties=lambda factor,family:factor.uncertainties,source_target=target,role_target=role_target,
        write=lambda path,packet:captured.update(packet=packet))
    return captured['packet'],output

def main():
    proof=read(PROOF);pp=read(PP);contract=human_support_operator_contract()
    assert proof['passed']and proof['group_count']==86 and proof['operator_contract']==contract
    assert pp['passed']and pp['complete_q_sha256']==proof['complete_q_sha256']and pp['rows']==6
    config=read(CONFIG);oldscope=read(config['source_options']['--truth-restoration-reviewed-manual-append-audit'])
    previous_audit=read(oldscope['same_q_public_audit']['path'])
    parent=read(proof['original_restoration_caps']['path']);old=read(proof['original_runtime_manifest']['path'])
    assert old['source_rows']==4501970
    transaction_path=SYS/(PREFIX+'_transaction.json')
    write(transaction_path,dict(schema='rinkspace.actual-same-q-finite-original-support-semantic-revision-transaction.v1',
        passed=True,actual_all86_proof=ref(PROOF),PP6_inheritance_proof=ref(PP),
        parent_truth_state=proof['original_restoration_caps'],parent_source_manifest=proof['original_runtime_manifest'],
        complete_state_reference=proof['complete_state_reference'],original_source_arrays_and_layout_unchanged=True,
        source_rows_before=old['source_rows'],source_rows_after=old['source_rows'],new_scalar_rows=0,
        old_operator_caps_not_claimed_equivalent=True,cap_policy='minimum_parent_cap_and_actual_revised_same_q_violation',
        operator_contract=contract,fresh_source_J_and_profiles_required=True,solver_started=False,accuracy_accepted=False))
    families={}
    for family,row in proof['families'].items():
        families[family]=dict(group_ids=row['group_ids'],old_count=len(row['group_ids']),
            violations_px=row['corrected_actual_same_q_violation_px'],limits=row['proposed_ratchet_only_corrected_semantic_caps_px'],
            uncertainties=row['original_uncertainties_exact'])
    audit=deepcopy(previous_audit)
    audit.update(complete_q_sha256=proof['complete_q_sha256'],complete_state_reference=proof['complete_state_reference'],
        old_caps_reference=proof['original_restoration_caps'],parent_manifest_reference=proof['original_runtime_manifest'],
        total_added_source_scalars=0,added_direct_manual_centre_scalars=0,new_complete_source_scalar_count=old['source_rows'],
        parent_group_count=86,new_group_count=86,families=families,current_source_target_transaction=ref(transaction_path),
        existing86_caps_uncertainties_group_ids_exact=False,existing86_source_support_uncertainties_group_ids_exact=True,
        actual_same_q86_invariance_proof=ref(PROOF),engineering_prior_append_reference=ref(PP),
        finite_truth_semantic_revision_not_numerical_accuracy_improvement=True)
    audit_path=SYS/(PREFIX+'_startup_audit.json');write(audit_path,audit)
    values_path=SYS/(PREFIX+'_caps_values.json')
    write(values_path,dict(schema='rinkspace.exact-parent-plus-current-source-cap-values.v1',
        complete_q_sha256=proof['complete_q_sha256'],audit=ref(audit_path),families=families))
    scope=deepcopy(oldscope);scope.update(same_q_public_audit=ref(audit_path),exact_caps_values=ref(values_path),
        current_source_target_transaction=ref(transaction_path),
        admitted_engineering_prior_append=dict(actual_same_q_public_audit=ref(PP),rows=6,new_rows_in_this_revision=0),
        admitted_truth_support_operator_revision=dict(actual_same_q_public_audit=ref(PROOF),contract=contract,
            cap_policy='minimum_parent_cap_and_actual_revised_same_q_violation'),
        all86_existing_human_groups_and_caps_preserved=False,all86_source_supports_and_group_ids_preserved=True)
    binding=SYS/(PREFIX+'_startup_binding.json');write(binding,scope)
    with np.load(proof['complete_state_reference']['path'],allow_pickle=False)as z:
        q=z['q'];damping=float(z['damping']);fraction=float(z['next_line_search_fraction']);fields=len(z.files)
    assert hashlib.sha256(q.tobytes()).hexdigest()==proof['complete_q_sha256']
    options=config['source_options'];options.update({'--camera-seed':proof['complete_state_reference']['path'],
        '--truth-restoration-state':proof['original_restoration_caps']['path'],
        '--truth-restoration-reviewed-manual-append-audit':str(binding),'--initial-damping':damping})
    options.pop('--accepted-steps',None)
    config['source_manifest_sha256']=sha(ROOT/'calibration/solver_runtime/source_manifest.json')
    config['continuation']=dict(complete_accepted_state=proof['complete_state_reference']['path'],
        checkpoint_file_sha256=proof['complete_state_reference']['sha256'],truth_restoration_state=proof['original_restoration_caps']['path'],
        truth_restoration_state_sha256=proof['original_restoration_caps']['sha256'],
        state=dict(q_sha256=proof['complete_q_sha256'],parameters=len(q),fields=fields,damping=damping,next_line_search_fraction=fraction),
        previous_input_configuration=str(CONFIG),parent_source_input_manifest=proof['original_runtime_manifest'],
        expected_complete_active_source_scalars=old['source_rows'],current_truth_group_count=86,
        current_source_target_transaction=ref(transaction_path),reviewed_source_target_startup_binding=ref(binding),
        actual_same_q86_proof=ref(PROOF),PP6_same_q_existing_prior_proof=ref(PP),
        next_start_requires_fresh_source_J=True,no_cached_J_H_g_or_prefix_imported=True,
        original_support_rectangles_UV_normal_sigma_and_world_unchanged=True,
        operator_revision_is_not_geometric_accuracy_acceptance=True,solver_run_started=False,accuracy_accepted=False)
    write(NEXT,config)
    with tempfile.TemporaryDirectory(prefix='q757_semantic_binding_',dir=SYS)as directory:
        f=actual_receipt_carriers(directory);packet,_=invoke(f)
        assert sum(len(v['limits'])for v in packet['families'].values())==86
        for family,row in families.items():
            assert packet['families'][family]['limits']==row['limits']
            assert packet['families'][family]['uncertainties']==parent['families'][family]['uncertainties']
    print(json.dumps(dict(config=ref(NEXT),startup_binding=ref(binding),
        receipt_gate_passed=True,actual_startup_must_recompute_all86=True,source_rows=old['source_rows'],
        q_sha256=proof['complete_q_sha256'],damping=damping,solver_started=False)))

if __name__=='__main__':main()
