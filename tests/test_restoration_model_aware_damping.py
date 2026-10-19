import json
from pathlib import Path
import pytest

from calibration.solver_runtime import install
install()
from z190_joint_source_solver import accepted_schur_damping,restoration_source_model_agreement


@pytest.mark.parametrize('predicted,actual,trusted',[(10.,10.,True),(-10.,-9.,True),
    (10.,-1.,False),(10.,2.,False),(-10.,-30.,False),(0.,0.,False),(float('nan'),1.,False)])
def test_signed_prediction_controls_only_numerical_lambda_not_truth_acceptance(predicted,actual,trusted):
    report=restoration_source_model_agreement(predicted,actual)
    assert report['reliable_for_restoration_lambda_decrease']is trusted
    assert report['complete_source_cost_nonincrease_is_not_a_truth_acceptance_requirement']
    assert accepted_schur_damping(1.,1.,backtracking_aware=True,truth_restoration=True,
        restoration_model_agreement=report)==(.3 if trusted else 1.)
    assert accepted_schur_damping(1.,.5,backtracking_aware=True,truth_restoration=True,
        restoration_model_agreement=report)==1.


def test_no_report_and_ordinary_source_objective_keep_historical_arithmetic():
    assert accepted_schur_damping(1.,1.,backtracking_aware=True,truth_restoration=True)==.3
    assert accepted_schur_damping(1.,.5,backtracking_aware=True,truth_restoration=True)==1.
    assert accepted_schur_damping(1.,.5,backtracking_aware=True)==2.


ACTUAL=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/current2109_fresh_model_reversal_and_truth_lambda_replay_20261010.json')


@pytest.mark.skipif(not ACTUAL.exists(),reason='Actual complete event cycle receipts unavailable')
def test_real_fresh_reversed_complete_steps_no_longer_reduce_lambda():
    rows=json.loads(ACTUAL.read_text(encoding='utf8'))['actual_complete_steps']
    actual_fresh_reversed=[]
    for row in rows:
        report=restoration_source_model_agreement(row['predicted_source_reduction'],row['actual_source_reduction'])
        revised=accepted_schur_damping(row['old_lambda'],row['alpha'],backtracking_aware=True,
            truth_restoration=True,restoration_model_agreement=report)
        if row['exact_current_J']and row['model_reversed']:
            actual_fresh_reversed.append(row['iteration'])
            assert revised==row['old_lambda']>row['committed_lambda']
    assert actual_fresh_reversed==[2,3,4,5]
