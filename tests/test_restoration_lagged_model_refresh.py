import json
from pathlib import Path
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.solver_runtime import install
install()
from structured_programme_source_candidate.lagged_jacobian_source import LaggedJacobianSource


@pytest.mark.parametrize('predicted,actual,refresh',[(-10.,-16.,True),(-10.,-9.,False),
    (10.,16.,True),(10.,9.,False),(10.,-1.,True),(0.,0.,True)])
def test_truth_signed_growth_uses_symmetric_model_agreement_without_cost_veto(predicted,actual,refresh):
    model=LaggedJacobianSource(.25);model.force_fresh=False
    model.jacobians=(csr_matrix([[1.]]),)
    model.accepted(100.,100.-actual,np.ones(1),csr_matrix((1,1)),np.array([-predicted]),
        source_objective_acceptance=False)
    assert model.force_fresh is refresh
    assert model.current_report['restoration_source_model_agreement']['complete_source_cost_nonincrease_is_not_a_truth_acceptance_requirement']
    assert model.begin(np.array([1.]),'updated_profile') is not refresh


def test_ordinary_accepted_cost_decrease_keeps_original_ratio_rule():
    model=LaggedJacobianSource(.25);model.force_fresh=False
    model.accepted(100.,84.,np.ones(1),csr_matrix((1,1)),np.array([-10.]))
    assert not model.force_fresh
    assert 'restoration_source_model_agreement'not in model.current_report


RUN=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_review86_TOP_raster_PP6_20261010_2211')


@pytest.mark.skipif(not(RUN/'solver/iterations.json').exists(),reason='Saved actual complete restoration cycles unavailable')
def test_actual_2211_shortened_growth_cycles_request_current_fresh_J():
    rows=json.loads((RUN/'solver/iterations.json').read_text(encoding='utf8'))
    tested=[]
    for row in rows:
        update=row.get('numerical_schur_backtracking_aware_damping',{}).get('accepted_update',{})
        agreement=update.get('restoration_source_model_agreement')
        if not row.get('accepted')or agreement is None or update['accepted_whole_source_step_fraction']>=1.:continue
        model=LaggedJacobianSource(.25);model.force_fresh=False
        p=agreement['predicted_source_reduction'];a=agreement['actual_source_reduction']
        model.accepted(100.,100.-a,np.ones(1),csr_matrix((1,1)),np.array([-p]),source_objective_acceptance=False)
        assert model.force_fresh is (not agreement['reliable_for_restoration_lambda_decrease'])
        if p<0. and a<0. and not agreement['reliable_for_restoration_lambda_decrease']:
            assert a/p>.25
            assert model.force_fresh
            tested.append(row['iteration'])
    assert {1,2}.issubset(tested)
