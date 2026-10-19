"""Uninstalled candidate uses real QPs and actual nonlinear carriers."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
CANDIDATE=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/optional_SOC_continue_on_actual_progress_EXECUTION_CANDIDATE_20261011.py')

@pytest.fixture
def replay(monkeypatch):
    from calibration.solver_runtime import install
    install()
    import z190_joint_source_solver as joint
    from structured_programme_source_candidate import source_step_curvature_tightening as source
    spec=importlib.util.spec_from_file_location('calibration.solver_runtime._SOC_actual_progress_candidate_test',CANDIDATE)
    control=importlib.util.module_from_spec(spec);sys.modules[spec.name]=control;spec.loader.exec_module(control)
    reference_spec=importlib.util.spec_from_file_location('_SOC_actual_nonlinear_replay_fixture',ROOT/'tests/test_optional_soc_nonlinear_progress.py')
    proof=importlib.util.module_from_spec(reference_spec);reference_spec.loader.exec_module(proof)
    proof.runtime=control
    original=source.measured_outward_step_tightening
    while hasattr(original,'__wrapped__'):original=original.__wrapped__
    monkeypatch.setattr(source,'measured_outward_step_tightening',original)
    damp=joint.accepted_schur_damping
    while hasattr(damp,'__wrapped__')and getattr(damp,'_retained_SOC_face_model_trust',False):damp=damp.__wrapped__
    monkeypatch.setattr(joint,'accepted_schur_damping',damp)
    return control,proof.replay_actual_QP_and_nonlinear_source,joint


def test_prediction_disagreement_can_continue_real_improving_corrections_and_hold_lambda(replay):
    control,run,joint=replay
    initial,final,attempts,b,options=run(curvature=.1,original_target=2.8)
    assert len(attempts)>2
    report=final.report['sequential_measured_shared_source_curvature_tightening']
    entries=report['actual_original_nonlinear_feasibility_progress']['measurements']
    assert not entries[1]['original_face_reduction_model_is_reliable']
    assert entries[1]['prediction_disagreement_only_holds_damping']
    assert entries[2]['complete_original_KKT_verified']
    # This nonlinear source has max value 2.5, below its requested target2.8.
    # The candidate exits on genuine merit nonprogress, not a prediction rule.
    best=min(entries,key=lambda e:e['actual_original_face_feasibility']['maximum_violation_px'])
    assert report['retained_complete_KKT_direction_sha256']==best['direction_sha256']
    assert report['stop_reason']=='optional_actual_original_nonlinear_feasibility_nonprogress'
    assert report['actual_original_nonlinear_feasibility_progress']['no_iteration_or_time_limit']
    assert control._previous_full_KKT_verified(final)
    control._install_retained_SOC_damping_trust(joint)
    trust=joint.restoration_source_model_agreement(10.,10.)
    assert joint.accepted_schur_damping(.01,1.,truth_restoration=True,restoration_model_agreement=trust)==.01
    assert not trust['reliable_for_restoration_lambda_decrease']


def test_either_merit_worsening_returns_prior_verified_direction(replay):
    control,run,_=replay
    initial,final,attempts,b,options=run(curvature=np.array([.23,.1,.1]),original_target=[2.1,2.,2.])
    assert len(attempts)==2 and final is initial
    report=final.report['sequential_measured_shared_source_curvature_tightening']
    before,after=[r['actual_original_face_feasibility']for r in report['actual_original_nonlinear_feasibility_progress']['measurements']]
    assert after['maximum_violation_px']>before['maximum_violation_px']
    assert after['total_violation_px']<before['total_violation_px']
    assert control._previous_full_KKT_verified(final)
    np.testing.assert_array_equal(options['global_step_constraint_lower'],b)


def test_unverified_correction_cannot_replace_last_valid_direction(replay,monkeypatch):
    control,run,_=replay
    original=control._previous_full_KKT_verified;calls=0
    def verified(linear):
        nonlocal calls
        calls+=1
        # QP constructor validates at calls1 and3; the second actual-measurement
        # consumer deliberately receives a missing original KKT certificate.
        return False if calls==4 else original(linear)
    monkeypatch.setattr(control,'_previous_full_KKT_verified',verified)
    initial,final,attempts,b,options=run(curvature=.1,original_target=2.8)
    assert len(attempts)==2 and final is initial
    report=final.report['sequential_measured_shared_source_curvature_tightening']
    assert report['stop_reason']=='optional_actual_original_nonlinear_feasibility_unverified_KKT'
    np.testing.assert_array_equal(options['global_step_constraint_lower'],b)


def test_true_source_plateau_stops_without_resolvable_actual_progress(replay,monkeypatch):
    control,run,_=replay
    from structured_programme_source_candidate import source_step_curvature_tightening as source
    original=source.measured_outward_step_tightening
    def plateau(matrix,lower,step,responses,actual):
        # Independent actual saturated source, while the current tangent is
        # stale: another linear cut cannot establish true query progress.
        actual={name:np.minimum(value,2.)for name,value in actual.items()}
        return original(matrix,lower,step,responses,actual)
    monkeypatch.setattr(source,'measured_outward_step_tightening',plateau)
    observer=control.original_nonlinear_step_face_feasibility
    def actual_plateau(matrix,lower,step,responses,actual):
        return observer(matrix,lower,step,responses,{name:np.minimum(value,2.)for name,value in actual.items()})
    monkeypatch.setattr(control,'original_nonlinear_step_face_feasibility',actual_plateau)
    initial,final,attempts,b,options=run(curvature=.01,original_target=2.8)
    assert len(attempts)==2 and final is initial
    report=final.report['sequential_measured_shared_source_curvature_tightening']
    entries=report['actual_original_nonlinear_feasibility_progress']['measurements']
    assert not entries[-1]['improves_best_observed_actual_feasibility']
    assert entries[-1]['maximum_and_total_actual_feasibility_nonworsening']
    assert report['stop_reason']=='optional_actual_original_nonlinear_feasibility_nonprogress'
    assert control._previous_full_KKT_verified(final)
