from collections import OrderedDict
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.solver_runtime import install
install()

from calibration.solver_runtime.staged_source_guard import (
    GuardProbe, StagedSourceGuard, compare_recorded_guard_outcome, observe_guard,
    preserve_programme_camera_bindings, restoration_progress_precheck,
)


def qtoken(q):
    return hashlib.sha256(np.asarray(q, float).tobytes()).hexdigest()


def test_initial_limits_cannot_be_created_by_early_probes():
    calls = []
    def complete(q):
        calls.append("original_initialization")
        return True
    def forbidden(q):
        raise AssertionError("Probe may not initialize limits")
    guard = StagedSourceGuard(complete, [GuardProbe("truth", forbidden, lambda e: True)],
                             ready=lambda: False, allow_rejection_priority_change=True)
    assert guard(np.array([1.])) is True
    assert calls == ["original_initialization"]
    assert guard.last_report["complete_original_guard_executed"]


def test_restoration_precheck_reports_exact_failed_groups_from_existing_two_queries():
    from calibration.solver_runtime.trial_precheck_runtime import restoration_precheck_from_callback
    parent=np.array([0.]);trial=np.array([1.]);calls=[]
    def complete_truth_violations(q):
        calls.append(float(q[0]))
        return (np.array([2.,0.,3.])if q[0]==0. else np.array([2.1,1e-7,2.])),['worsened','protected','progressed']
    priority_restoration_domain=dict(q_sha256=qtoken(parent),common_fraction=.5)
    def source_trial_acceptance(*args,**kwargs):
        return complete_truth_violations,priority_restoration_domain
    probe=restoration_precheck_from_callback(source_trial_acceptance,blocks=[])
    result=probe(parent,trial,trial_context=dict(line_search_fraction=.1))
    assert result['action']=='reject'and result['acceptance_authority']is False
    assert calls==[0.,1.]
    rows=result['report']['failed_original_truth_groups']
    assert [r['group_id']for r in rows]==['worsened','protected']
    assert rows[0]['each_nonworsening_failed']and rows[1]['protected_zero_group']
    assert rows[1]['trial_violation_px']==1e-7


def test_failed_then_passing_trial_restores_camera_binding_and_runs_full_guard():
    bundle = SimpleNamespace(calibration="parent")
    block = SimpleNamespace(bundle=bundle, camera_local=lambda q: q,
                            _native_camera_valid_worlds=OrderedDict([(b"parent", True)]))
    cache = block._native_camera_valid_worlds
    calls = []
    def truth(q):
        bundle.calibration = "trial"
        block._native_camera_valid_worlds[b"trial"] = True
        if q[0] < 0:
            raise ValueError("original truth support rejected")
        return True
    def complete(q):
        assert bundle.calibration == "parent"
        assert block._native_camera_valid_worlds is cache
        assert list(cache) == [b"parent"]
        calls.append("complete")
        truth(q)
        return True
    guard = StagedSourceGuard(complete, [GuardProbe("truth", truth,
        lambda e: str(e) == "original truth support rejected")], ready=lambda: True,
        probe_context=lambda: preserve_programme_camera_bindings([block]),
        allow_rejection_priority_change=True)
    with pytest.raises(ValueError, match="original truth support rejected"):
        guard(np.array([-1.]))
    assert calls == []
    assert bundle.calibration == "parent"
    assert guard(np.array([1.])) is True
    assert calls == ["complete"]
    assert guard.early_rejections == 1


def test_unknown_probe_failure_and_default_priority_preserve_original_error():
    def complete(q):
        raise RuntimeError("earlier physical domain")
    def truth(q):
        raise ValueError("later truth support")
    for recognized, priority in [(False, True), (True, False)]:
        guard = StagedSourceGuard(complete, [GuardProbe("truth", truth, lambda e: recognized)],
                                 ready=lambda: True, allow_rejection_priority_change=priority)
        with pytest.raises(RuntimeError, match="earlier physical domain"):
            guard(np.array([1.]))
        assert guard.original_calls == 1


def test_probe_cannot_mutate_complete_state_or_skip_original_return():
    q = np.array([1.])
    def mutate(value):
        value[0] = 999.
    guard = StagedSourceGuard(lambda value: False, [GuardProbe("broken", mutate, lambda e: False)],
                             ready=lambda: True, allow_rejection_priority_change=True)
    assert guard(q) is False
    assert np.array_equal(q, [1.])


def test_original_restoration_predicate_with_costs_removed_is_rejection_only():
    # Use the real original function, without preparing any source graph.
    from calibration.solver_runtime import install
    install()
    from structured_programme_source_candidate.feasibility_priority_restoration import actual_restoration_trial_decision
    old = np.array([2., 0., 3.])
    for new in (np.array([1., 0., 2.]), np.array([2.1, 0., 2.]),
                np.array([1., 1e-7, 2.]), np.array([2., 0., 3.])):
        early = restoration_progress_precheck(actual_restoration_trial_decision, old, new,
                                              common_fraction=.5, step_fraction=.1)
        for before, after in [(123., 122.), (123., 456.)]:
            original = actual_restoration_trial_decision(old, new, common_fraction=.5,
                step_fraction=.1, physical_guard_passed=True,
                previous_source_cost=before, trial_source_cost=after)
            assert early["reject"] == (not original["accepted"])
            assert early["reason"] == original["reason"]
        assert "accepted" not in early
        assert all("cost" not in key or key.endswith("pending") or "placeholder" in key
                   for key in early["report"])


def test_recorded_endpoint_event_uses_existing_pixel_comparison_precision():
    from calibration.solver_runtime import install
    install()
    from structured_programme_source_candidate.feasibility_priority_restoration import actual_restoration_trial_decision
    # RUN0520 fe074 trial2: the physical endpoint passed; only this Ad24
    # comparison rejected the complete trial. These are actual recorded V.
    old = np.array([0., 0.01665837477151655])
    new = np.array([0., 0.016658374773152573])
    before = old.copy(), new.copy()
    common = .9902812389103697
    alpha = 1.2851208255103187e-7
    result = actual_restoration_trial_decision(old, new, common_fraction=common,
        step_fraction=alpha, physical_guard_passed=True,
        previous_source_cost=10., trial_source_cost=10.)
    assert result['accepted']
    assert not result['strictly_nonworsening_without_numerical_tolerance']
    assert result['maximum_positive_group_growth_px'] == new[1] - old[1]
    assert result['restoration_comparison_tolerance_px'] == 1e-8
    assert result['maximum_actual_original_violation_fraction'] > 1.
    np.testing.assert_array_equal(old, before[0])
    np.testing.assert_array_equal(new, before[1])
    # Early rejection uses precisely the same convention; it cannot grant
    # acceptance or bypass the complete physical/cost/truth guard.
    early = restoration_progress_precheck(actual_restoration_trial_decision,
        old, new, common_fraction=common, step_fraction=alpha)
    assert not early['reject']
    assert early['report']['original_complete_acceptance_still_required']
    for changed in (np.array([1.1e-8, new[1]]),
                    np.array([0., old[1] + 1e-7])):
        rejected = actual_restoration_trial_decision(old, changed,
            common_fraction=common, step_fraction=alpha,
            physical_guard_passed=True, previous_source_cost=10., trial_source_cost=1.)
        assert not rejected['accepted']
    # The real native35 backtrack increase is still rejected.
    rejected = actual_restoration_trial_decision([.07855888335817696],
        [.07982750124949067], common_fraction=common, step_fraction=1/2048,
        physical_guard_passed=True, previous_source_cost=10., trial_source_cost=1.)
    assert not rejected['accepted']


def test_endpoint_numerical_progress_precheck_does_not_report_false_failed_groups():
    from calibration.solver_runtime.trial_precheck_runtime import restoration_precheck_from_callback
    previous = np.array([0., .01665837477151655])
    actual = np.array([0., .016658374773152573])
    priority_restoration_domain = dict(q_sha256=qtoken([0.]),
                                      common_fraction=.9902812389103697)
    def complete_truth_violations(q):
        return (previous if q[0] == 0 else actual), [('native', 0), ('confirmed', 20)]
    def source_trial_acceptance(*args, **kwargs):
        return complete_truth_violations, priority_restoration_domain
    probe = restoration_precheck_from_callback(source_trial_acceptance, blocks=[])
    result = probe(np.array([0.]), np.array([1.]),
                   trial_context=dict(line_search_fraction=1.2851208255103187e-7))
    assert result['action'] == 'continue'
    assert result['acceptance_authority'] is False
    assert result['report']['failed_original_truth_groups'] == []
    assert result['report']['maximum_positive_group_growth_px'] > 0.


def test_saved_outcome_compare_refuses_different_complete_state():
    recorded = observe_guard(lambda q: True, np.array([1.]))
    same = observe_guard(lambda q: True, np.array([1.]))
    assert compare_recorded_guard_outcome(recorded, same)["passed"]
    with pytest.raises(ValueError, match="Same original complete q"):
        compare_recorded_guard_outcome(recorded, observe_guard(lambda q: True, np.array([2.])))


def test_no_restoration_violation_returns_to_original_source_cost_acceptance():
    def forbidden(*args, **kwargs):
        raise AssertionError("Original driver would choose its source-objective branch")
    result = restoration_progress_precheck(forbidden, [0., 1e-8], [0., 0.],
                                          common_fraction=.5, step_fraction=.1)
    assert result["reject"] is False
    assert result["reason"] == "ordinary_source_objective_pending"


def make_callback_fixture(mode="normal"):
    bundle = SimpleNamespace(calibration="original")
    block = SimpleNamespace(bundle=bundle, camera_local=lambda q: q)
    human_truth = SimpleNamespace(state=SimpleNamespace(blocks=[block]))
    priority_restoration_domain = dict(q_sha256=qtoken([0.]), common_fraction=.5)
    calls = []
    def complete_truth_violations(q):
        calls.append(float(q[0]))
        human_truth.state.blocks[0].bundle.calibration = "query"
        if mode == "error" and q[0] > 0:
            raise ValueError("Original physical query domain")
        labels = [("native", 0), ("native", 1)]
        if mode == "labels" and q[0] > 0:
            labels.reverse()
        return (np.array([2., 0.]) if q[0] == 0 else np.array([float(q[0]), 0.])), labels
    def source_trial_acceptance(parent, trial, **kwargs):
        return complete_truth_violations(parent), priority_restoration_domain
    return source_trial_acceptance, bundle, calls, priority_restoration_domain


def test_callback_precheck_uses_same_domain_restores_binding_and_never_accepts():
    from calibration.solver_runtime.trial_precheck_runtime import restoration_precheck_from_callback
    callback, bundle, calls, domain = make_callback_fixture()
    precheck = restoration_precheck_from_callback(callback)
    for trial, action in [(np.array([2.1]), "reject"), (np.array([1.]), "continue")]:
        result = precheck(np.array([0.]), trial, trial_context=dict(line_search_fraction=.1))
        assert result["action"] == action
        assert result["acceptance_authority"] is False
        assert result["trial_q_sha256"] == qtoken(trial)
        assert result["parent_q_sha256"] == domain["q_sha256"]
        assert bundle.calibration == "original"
    assert calls == [0., 2.1, 0., 1.]


def test_callback_precheck_without_ready_domain_does_not_query_or_init_limits():
    from calibration.solver_runtime.trial_precheck_runtime import restoration_precheck_from_callback
    callback, bundle, calls, domain = make_callback_fixture()
    precheck = restoration_precheck_from_callback(callback, ready=lambda: False)
    assert precheck(np.array([0.]), np.array([2.1]),
                    trial_context=dict(line_search_fraction=.1))["action"] == "continue"
    assert calls == []
    assert domain == dict(q_sha256=qtoken([0.]), common_fraction=.5)
    precheck = restoration_precheck_from_callback(callback)
    assert precheck(np.array([-1.]), np.array([2.1]),
                    trial_context=dict(line_search_fraction=.1))["action"] == "continue"
    assert calls == []


@pytest.mark.parametrize("mode", ["error", "labels"])
def test_callback_precheck_domain_or_label_failure_defers_to_original_order(mode):
    from calibration.solver_runtime.trial_precheck_runtime import restoration_precheck_from_callback
    callback, bundle, calls, domain = make_callback_fixture(mode)
    result = restoration_precheck_from_callback(callback)(np.array([0.]), np.array([2.1]),
        trial_context=dict(line_search_fraction=.1))
    assert result["action"] == "continue"
    assert result["acceptance_authority"] is False
    assert "deferred_error" in result["report"]
    assert bundle.calibration == "original"


def test_wide_runtime_installer_retains_class_calls_original_first_and_is_idempotent(monkeypatch):
    from calibration.solver_runtime import wide_top_only_candidate as candidate
    from structured_programme_source_candidate.human_stroke_truth import WideStrokeTruth
    calls = []
    original_query = WideStrokeTruth._queries_for_records
    def original(self, required, *, supplied=None):
        calls.append(("original", required, supplied))
        self.initialized = True
    def install(instance):
        assert type(instance) is WideStrokeTruth and instance.initialized
        calls.append(("install",))
    monkeypatch.setattr(WideStrokeTruth, '__init__', original)
    monkeypatch.setattr(WideStrokeTruth, '_rinkspace_wide_top_only_execution_receipt', None, raising=False)
    monkeypatch.setattr(candidate, 'install_wide_top_only', install)
    receipt = candidate.install_wide_top_only_runtime()
    instance = WideStrokeTruth(123, supplied="unchanged")
    assert type(instance) is WideStrokeTruth
    assert calls == [("original", 123, "unchanged"), ("install",)]
    assert WideStrokeTruth._queries_for_records is original_query
    assert candidate.install_wide_top_only_runtime() is receipt
    assert len(receipt['original_constructor_sha256']) == 64
    assert len(receipt['helper_sha256']) == 64
