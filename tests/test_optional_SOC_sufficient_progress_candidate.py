"""An optional correction needs sufficient progress, while retaining its best."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
CANDIDATE=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/optional_SOC_sufficient_actual_predicted_progress_EXECUTION_CANDIDATE_20261011.py')


@pytest.fixture
def replay(monkeypatch):
    from calibration.solver_runtime import install
    install()
    import z190_joint_source_solver as joint
    from structured_programme_source_candidate import source_step_curvature_tightening as source
    spec=importlib.util.spec_from_file_location('calibration.solver_runtime._SOC_sufficient_progress_test',CANDIDATE)
    control=importlib.util.module_from_spec(spec);sys.modules[spec.name]=control;spec.loader.exec_module(control)
    spec2=importlib.util.spec_from_file_location('_SOC_progress_fixture',ROOT/'tests/test_optional_soc_nonlinear_progress.py')
    proof=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(proof);proof.runtime=control
    original=source.measured_outward_step_tightening
    while hasattr(original,'__wrapped__'):original=original.__wrapped__
    monkeypatch.setattr(source,'measured_outward_step_tightening',original)
    damp=joint.accepted_schur_damping
    while hasattr(damp,'__wrapped__')and getattr(damp,'_retained_SOC_face_model_trust',False):damp=damp.__wrapped__
    monkeypatch.setattr(joint,'accepted_schur_damping',damp)
    return control,proof.replay_actual_QP_and_nonlinear_source,joint


def agreement(max_before,max_after,sum_before,sum_after):
    return {key:dict(predicted_source_reduction=before,actual_source_reduction=before-after)
            for key,before,after in [('maximum',max_before,max_after),('total',sum_before,sum_after)]}


def test_real_0758_merits_retain_correction_but_do_not_request_slow_crawl(replay):
    control,_,_=replay
    result=control._sufficient_SOC_feasibility_reduction(agreement(
        .0008195538668838154,.000817225892876472,.004354977977583642,.0012506939889762592),1e-15,1e-15)
    assert not result['sufficient']
    assert not result['merits']['maximum']['sufficient']
    assert result['merits']['total']['sufficient']
    assert result['merits']['maximum']['actual_to_predicted_reduction']<.003


def test_0725_large_real_merit_improvements_can_continue_despite_lambda_distrust(replay):
    control,_,_=replay
    result=control._sufficient_SOC_feasibility_reduction(agreement(1.,.489,1.,.256),1e-15,1e-15)
    assert result['sufficient']
    # Existing symmetric prediction error .489 is still above the trust .25.
    assert result['merits']['maximum']['actual_to_predicted_reduction']==pytest.approx(.511)


def test_actual_QP_and_nonlinear_plateau_exit_retains_current_better_full_KKT(replay):
    control,run,joint=replay
    initial,final,attempts,b,options=run(curvature=.1,original_target=2.8)
    outer=final.report['sequential_measured_shared_source_curvature_tightening']
    rows=outer['actual_original_nonlinear_feasibility_progress']['measurements']
    assert len(attempts)==3 and final is attempts[-1]
    assert rows[1]['sufficient_actual_to_predicted_feasibility_reduction']['sufficient']
    assert not rows[-1]['sufficient_actual_to_predicted_feasibility_reduction']['sufficient']
    assert rows[-1]['current_complete_KKT_direction_retained']
    assert not rows[-1]['current_candidate_discarded']
    assert outer['stop_reason']=='optional_actual_original_nonlinear_feasibility_insufficient_model_reduction'
    assert control._previous_full_KKT_verified(final)
    assert rows[-1]['actual_original_face_feasibility']['maximum_violation_px']<rows[-2]['actual_original_face_feasibility']['maximum_violation_px']
    assert options['global_step_constraint_lower'][0]>b[0]
    assert final.step[0]>=options['global_step_constraint_lower'][0]-1e-12
    assert -final.step[0]>=options['global_step_constraint_lower'][1]
    control._install_retained_SOC_damping_trust(joint)
    trust=joint.restoration_source_model_agreement(10.,10.)
    assert joint.accepted_schur_damping(.01,1.,truth_restoration=True,restoration_model_agreement=trust)==.01


def test_true_growth_or_missing_full_KKT_still_returns_previous_witness(replay,monkeypatch):
    control,run,_=replay
    initial,final,attempts,b,options=run(curvature=np.array([.23,.1,.1]),original_target=[2.1,2.,2.])
    assert final is initial and len(attempts)==2
    np.testing.assert_array_equal(options['global_step_constraint_lower'],b)
    original=control._previous_full_KKT_verified;calls=0
    def verified(linear):
        nonlocal calls
        calls+=1
        return False if calls==4 else original(linear)
    monkeypatch.setattr(control,'_previous_full_KKT_verified',verified)
    initial,final,attempts,b,options=run(curvature=.1,original_target=2.8)
    assert final is initial
    np.testing.assert_array_equal(options['global_step_constraint_lower'],b)


def test_both_merits_required_and_zero_predicted_roundoff_not_divided(replay):
    control,_,_=replay
    result=control._sufficient_SOC_feasibility_reduction(agreement(1.,.1,1.,.999),1e-15,1e-15)
    assert not result['sufficient']
    result=control._sufficient_SOC_feasibility_reduction(agreement(0.,0.,1.,.5),1e-15,1e-15)
    assert result['sufficient'] and result['merits']['maximum']['actual_to_predicted_reduction']is None
