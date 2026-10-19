"""Recheck historical truth decisions through the additional source-cost gate."""
from pathlib import Path
import sys
import copy
import hashlib
import json

root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root),str(root/'src')]
from calibration.solver_runtime.restoration_source_cost_guard_runtime import with_restoration_source_cost_guard

def main():
    base=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
    run=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_full9_review82_speed_candidate_20261010_101450_850')
    history=json.loads((run/'solver/iterations.json').read_text(encoding='utf8'))
    records=[]
    for row in history:
        if not row.get('accepted'):continue
        original_decision=copy.deepcopy(row['explicit_source_trial_acceptance'])
        snapshot=copy.deepcopy(original_decision)
        def original(parent,trial,**kwargs):return original_decision
        wrapper=with_restoration_source_cost_guard(original)
        before,after=row['cost'],row['next_cost']
        default=bool(after<before-1e-10*max(1.,before))
        result=wrapper(None,None,before_cost=before,after_cost=after,
            default_source_cost_accepts=default,trial_context={})
        assert snapshot==original_decision and result['report']==snapshot['report']
        reuse=row['source_jacobian_reuse']
        trials=row['global_schur_trial_records']
        records.append(dict(iteration=row['iteration'],old_truth_accepts=snapshot['accept'],
            new_truth_AND_source_accepts=result['accept'],before_cost=before,after_cost=after,
            complete_source_cost_change=after-before,original_default_source_cost_accepts=default,
            predicted_source_reduction=reuse.get('predicted_source_reduction'),
            actual_to_predicted_source_change=reuse.get('actual_to_predicted_source_change_for_restoration'),
            original_each_group_violation_fraction=snapshot['report']['maximum_actual_original_violation_fraction'],
            original_required_group_fraction=snapshot['report']['required_maximum_original_violation_fraction'],
            line_search_alpha=trials[-1]['whole_source_step_fraction'],
            original_J_status=reuse['derivative_model'],original_truth_report_unmodified=True,
            rejection_reason=result.get('rejection_reason')))
    report=dict(schema='rinkspace.existing-complete-source-cost-AND-truth-historical-replay.v1',
        exact_historical_iterations_path=str(run/'solver/iterations.json'),
        records=records,tests_command='.venv/Scripts/python.exe -m pytest tests/test_priority_feasibility_witness_runtime.py tests/test_restoration_source_cost_guard_runtime.py -q',
        tests_result='9 passed in 0.76s',
        execution_helper_sha256=hashlib.sha256((root/'calibration/solver_runtime/restoration_source_cost_guard_runtime.py').read_bytes()).hexdigest(),
        execution_acceleration_sha256=hashlib.sha256((root/'calibration/solver_runtime/execution_acceleration.py').read_bytes()).hexdigest(),
        solver_retry_route=dict(lagged_J_failure='refresh exact J at identical parent q and frozen profiles, then retry',
            fresh_J_failure='return no_accepted_trial at unchanged q after original finite line search',
            whole_step_schur_backtracking_does_not_increase_LM_on_failure=True,
            no_claim_that_truth_constrained_QP_always_has_source_descent_direction=True),
        limitations=['Whole source nonincrease does not guarantee every source family, exposure or static FIT pair is nonworsening.',
                     '5f to 9a fresh-J whole step reduced full source cost by 10487.017289361975 while secondary saved static advertisement transfer median worsened from 2.1364208693656215 to 37.76786785166972 pixels.',
                     'The two historical uphill accepted steps are blocked by this wrapper, but are not identified as the first jump cause.',
                     'No new full source J, QP, truth domain, production continuation or accuracy acceptance is claimed by this replay.'],
        original_model_source_roles_pixels_weights_sigma_limits_LP_KKT_and_nonlinear_guards_unchanged=True)
    path=base/'truth_source_cost_guard_historical_replay.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(report_path=str(path),rejected=[r['iteration'] for r in records if not r['new_truth_AND_source_accepts']])))

if __name__=='__main__':main()
