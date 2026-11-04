"""Old hard-ice thresholds remain readable without deciding finite-soft quality."""
import json
from types import SimpleNamespace

import numpy as np

from calibration.program_tracking.body_tracking import skate_diagnostics as diagnostics
from calibration.program_tracking.body_tracking import skate_delivery as delivery
from calibration.program_tracking.body_tracking import skate_review as review


def test_delivery_soft_status_does_not_depend_on_legacy_gate_or_missing_gate():
    for old in ({}, dict(dense_gate=dict(feasible=False, max_violation_m=.12))):
        report = dict(feasible=True, **old)
        saved = dict(ice_mode=np.array('finite_soft_objective'))
        result = delivery._delivery_ice_status(report, saved)
        assert result['candidate_core_feasible'] is None
        assert not result['ice_is_delivery_gate']
        legacy = result['legacy_hard_ice_diagnostics']
        assert legacy['legacy_only'] and not legacy['used_for_current_quality']
        if old:assert legacy['dense_gate'] == old['dense_gate']
    legacy = delivery._delivery_ice_status(dict(feasible=True, dense_gate=dict(feasible=False)))
    assert legacy['candidate_core_feasible'] is False


def test_soft_checkpoint_keeps_old_raw_numbers_archives_report_and_skips_hard_counterfactual(tmp_path):
    out = tmp_path/'out'
    out.mkdir()
    source = tmp_path/'saved.npz'
    np.savez_compressed(source, x=np.zeros(18))
    (out/'candidate.npz').write_bytes(source.read_bytes())
    old_bytes = b'{"stop":"old budget","dense_gate":{"feasible":false},"untouched":7}\n'
    (out/'candidate.json').write_bytes(old_bytes)
    old_diagnosis = b'original historical hard-ice diagnosis'
    (out/'start_3520_diagnosis.json').write_bytes(old_diagnosis)

    class Problem:
        def __init__(self):
            self.physics_times=np.array([0., .01])
            self.body=SimpleNamespace(state=SimpleNamespace(bounds_time=(0., .01), seconds=(88., 88.01)))
            self.al=SimpleNamespace(multiplier=np.zeros(1), penalty=10.)
            self.calls=[]

        def evaluate(self, x, jacobian):
            return dict(parts={'ice_penetration': 1.}, gate=dict(feasible=True, ice_is_delivery_gate=False))

        def contact(self, x, times):
            self.calls.append(np.array(times))
            assert np.all((times >= 0.) & (times <= .01))
            return dict(margin_m=np.array([-.12]), labels=[(88000000000, 'left_nonpenetration')])

    problem=Problem()
    report=diagnostics.inspect_checkpoint(problem, dict(x=np.zeros(18),
        ice_mode=np.array('finite_soft_objective')), source, out)
    assert report['feasible']
    assert report['ice_mode'] == 'finite_soft_objective'
    assert 'dense_gate' not in report and 'physical_failure' not in report
    assert not report['legacy_hard_ice_diagnostics']['dense_gate']['feasible']
    assert len(problem.calls) == 1
    assert list((out/'history').glob('*/candidate.json'))[0].read_bytes() == old_bytes
    assert (out/'start_3520_diagnosis.json').read_bytes() == old_diagnosis


def test_review_keeps_geometric_heights_without_current_ice_failure(monkeypatch):
    pts=np.array([256], np.int64)
    raw=dict(clearance=np.array([[-.12, .02]]),
        features=dict(height=np.array([[[-.122], [.018]]]), semantics=['runner'], ids=['runner']))
    monkeypatch.setattr(review, 'fk_state', lambda *args: {})
    monkeypatch.setattr(review, 'constraints', lambda *args: raw)
    ref=SimpleNamespace(times=np.array([0.]), at=lambda times: {}, binding=object())
    _, result=review._physics(ref, dict(ice_mode=np.array('finite_soft_objective')), pts, [])
    assert result['signed_minimum_clearance_m'] == -.12
    assert 'physical_failure' not in result
    assert result['current_ice_quality_verdict'] is None
    assert result['legacy_hard_ice_diagnostics']['physical_failure']
    assert not result['legacy_hard_ice_diagnostics']['used_for_current_quality']
