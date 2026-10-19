"""Use the existing exact role-only migration for a complete stopped seed."""
from copy import deepcopy
from pathlib import Path
import argparse,hashlib,json,sys,time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from calibration.solver_runtime.checkpoint_history import iter_solver_history

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def ref(path):
    p=Path(path).resolve()
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--input-config',type=Path,required=True)
    cli.add_argument('--state',type=Path,required=True)
    cli.add_argument('--roles',type=Path,required=True)
    cli.add_argument('--source-proof',type=Path,required=True)
    cli.add_argument('--thin-RJ-proof',type=Path,required=True)
    cli.add_argument('--output-config',type=Path,required=True)
    cli.add_argument('--output-directory',type=Path,required=True)
    cli.add_argument('--expected-added-role-count',type=int,default=672)
    args=cli.parse_args();began=time.perf_counter()
    from calibration.solver_runtime import install
    install()
    from structured_programme_source_candidate import truth_source_role_migration as api
    from structured_programme_source_candidate import original_motion_far_board_roles as role_api
    import numpy as np
    state=args.state.resolve();out=args.output_directory.resolve();out.mkdir(exist_ok=True)
    truth=state/'truth_restoration_state.json'
    packet,original,caps=api.authenticate_state(truth)
    with np.load(state/'complete_state_UNACCEPTED.npz',allow_pickle=False)as z:
        q=z['q'].copy();fields=len(z.files)
        profiles=[name for name in z.files if name.endswith('_scales')or name.endswith('_sigma_weights')]
    identity=read(state/'state_identity.json')
    assert packet['complete_q_sha256']==identity['q_sha256']
    assert len(q)==34340 and original['source_rows']==4502662
    assert sum(len(r['group_ids'])for r in packet['families'].values())==86
    proof=api.prepare_proof(truth,args.roles,role_api.OriginalMotionFarBoardRoleBinding,Path(role_api.__file__))
    count=args.expected_added_role_count
    assert count>0 and len(proof['exact_added_point_inventory'])==count
    # An independently constructed public packet binding must match every
    # implementation-produced contract field, not just count a whitelist.
    binding=role_api.OriginalMotionFarBoardRoleBinding(args.roles)
    expected=proof['expected_actual_role_contract']
    for key,value in binding.contract.items():
        assert api.canonical(value)==api.canonical(expected[key]),key
    staged=deepcopy(original);staged[api.ROLE_KEY]=expected
    proofpath=out/f'same_complete_q_original86_caps_actual{count}_role_migration_proof.json';save(proofpath,proof)
    audit=api.authenticate_role_correction(proofpath,truth,q,staged,
        role_api.OriginalMotionFarBoardRoleBinding,Path(role_api.__file__))
    assert audit['group_count']==86 and audit['exact_changed_original_point_count']==count
    badq=q.copy();badq[0]=np.nextafter(badq[0],np.inf)
    try:api.authenticate_state(truth,badq)
    except ValueError:wrongq_rejected=True
    else:raise AssertionError('Different complete q must be rejected')
    changed=deepcopy(staged);changed['source_rows']+=2
    try:api.authenticate_role_correction(proofpath,truth,q,changed,
        role_api.OriginalMotionFarBoardRoleBinding,Path(role_api.__file__))
    except ValueError:changed_target_rejected=True
    else:raise AssertionError('Role migration must not authorize other source changes')
    config=read(args.input_config);options=config['source_options']
    for key in ('--reuse-complete-linearization','--reuse-source-blocks-from','--reuse-wide-linearization',
        '--verify-source-blocks-from','--verified-source-prefix-report','--truth-restoration-reviewed-manual-append-audit',
        '--truth-restoration-source-append-proof','--truth-restoration-source-revision-proof','--accepted-steps'):
        options.pop(key,None)
    options.update({'--camera-seed':str(state/'complete_state_UNACCEPTED.npz'),
        '--truth-restoration-state':str(truth),'--truth-restoration-source-role-proof':str(proofpath),
        '--original-motion-far-board-role-source':str(args.roles.resolve()),
        '--save-initial-linearization-cache':True,'--save-complete-source-directions':True})
    oldhistory=iter_solver_history(state.parent.parent/'solver/iterations.json')
    last=[row for row in oldhistory if row.get('accepted')][-1]
    options['--initial-damping']=last['damping']
    config['source_manifest_sha256']=ref(ROOT/'calibration/solver_runtime/source_manifest.json')['sha256']
    config['continuation']=dict(complete_accepted_state=str(state/'complete_state_UNACCEPTED.npz'),
        checkpoint_file_sha256=ref(state/'complete_state_UNACCEPTED.npz')['sha256'],
        truth_restoration_state=str(truth),truth_restoration_state_sha256=ref(truth)['sha256'],
        state=dict(q_sha256=identity['q_sha256'],parameters=len(q),fields=fields,
            frozen_profile_token=identity['frozen_profile_token'],damping=last['damping']),
        previous_input_configuration=ref(args.input_config),current_truth_group_count=86,
        expected_complete_active_source_scalars=4502662,next_start_requires_fresh_source_J=True,
        original_manual_group_cap_and_uncertainty_identity=caps,
        explicit_role_only_migration=ref(proofpath),
        old_source_J_H_normals_not_reused=True,source_clock_UV_sigma_and_existing_compaction_maps_retained=True,
        no_new_source_scalar_or_parameter=True,no_step_or_runtime_limit_added=True,accuracy_accepted=False)
    config['preparation_status']='Existing exact source-role proof prepared; production startup must independently authenticate the actual loaded new role contract and all original source/manual identities.'
    config['numerical_actual_SOC_progress']=dict(implementation=ref(ROOT/'calibration/solver_runtime/optional_soc_runtime.py'),
        prediction_disagreement_only_holds_lambda=True,
        actual_maximum_and_total_merit_nonworsening_and_resolvable_progress_required=True,
        minimum_actual_to_predicted_merit_reduction_fraction=.25,
        sufficient_progress_distinct_from_symmetric_prediction_error_trust=True,
        original_complete_KKT_and_all_nonlinear_physical_source_truth_acceptance_gates_retained=True)
    save(args.output_config,config)
    receipt=dict(schema='rinkspace.actual-existing-source-role-only-continuation-preparation.v1',passed=True,
        checkpoint=ref(state/'complete_state_UNACCEPTED.npz'),original_truth_state=ref(truth),
        new_role_packet=ref(args.roles),source_qualification_proof=ref(args.source_proof),
        thin_public_motion_R_J_proof=ref(args.thin_RJ_proof),config=ref(args.output_config),
        exact_role_migration_proof=ref(proofpath),public_role_binding_contract_matches_expected=True,
        prepared_migration_authentication=audit,different_q_rejected=wrongq_rejected,
        other_source_target_change_rejected=changed_target_rejected,
        complete_actual_programme_source_not_rebuilt_in_this_preparation=True,
        original_complete86_caps_bytes_preserved=True,source_scalar_count=4502662,
        new_source_scalars=0,added_original_roles=count,seconds=time.perf_counter()-began)
    receiptpath=out/f'actual_complete_seed_original86_caps_and_{count}_role_continuation_preparation.json';save(receiptpath,receipt)
    print(json.dumps(dict(passed=True,seconds=receipt['seconds'],config=ref(args.output_config),receipt=ref(receiptpath))))

if __name__=='__main__':main()
