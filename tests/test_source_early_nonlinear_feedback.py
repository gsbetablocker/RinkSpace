"""Early nonlinear feedback on actual small coupled source proposals."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.program_tracking.body_tracking.programme_source_step import (
    CompleteNonlinearGuardChecks,
    SourceMargins,
    source_conditioned_direction,
)


ACTIVE = np.array([3, 8, 11, 15])
WIDTH = 17
LIMITS = np.array([.8, 1., 1., 1.])
PREFERRED = np.array([.8, .2, .95, .7])
HESSIAN = np.array([[2., 1., 0., 0.], [1., 2., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
VALUE = {"residual": np.array([2.])}
READY_CAP = .24


class CoupledNormal:
    scale_inv = np.ones(4)

    def solve_scaled(self, rhs, scale, *, damp):
        np.testing.assert_array_equal(rhs, -VALUE["residual"])
        np.testing.assert_array_equal(scale, self.scale_inv)
        assert damp == .07
        return PREFERRED.copy(), {}

    def scaled_normal_objective(self, rhs, scale, *, damp):
        return self.objective

    @staticmethod
    def objective(point):
        offset = point - PREFERRED
        gradient = HESSIAN @ offset
        return float(.5 * (offset @ gradient - PREFERRED @ HESSIAN @ PREFERRED)), gradient


def domains(ready_kind):
    dimension = 3 if ready_kind == "flight" else 2
    ready = {
        "flight": "__flight__/episode/jump_2a/flight_specific_momentum",
        "body_point": "__body__/0/10",
        "halfspace": "__joint__/1",
    }[ready_kind]
    unfinished = "__body__/0/11" if ready_kind == "body_point" else "__flight__/episode/jump_3f/flight_integrated_com"
    linear = np.zeros((2, dimension, WIDTH))
    linear[0, 0, [1, ACTIVE[0]]] = [1., .25]  # A dependency on a frozen column must stay frozen.
    linear[1, 1, ACTIVE[2]] = 1.
    offsets = np.zeros((2, dimension))
    offsets[1, 0] = .6
    radii = np.array([READY_CAP, 1.])
    jacobians = {key: csr_matrix(rows) for key, rows in zip((ready, unfinished), linear, strict=True)}

    def ball(key, row):
        return dict(source_id=key, model_foot=None, unit="px" if ready_kind == "body_point" else "proxy residual",
                    labels=[key], offset=offsets[row:row+1], radius=radii[row:row+1], jacobian=jacobians[key])

    halfspaces, slack, labels = csr_matrix((0, WIDTH)), np.empty(0), []
    if ready_kind == "body_point":
        balls = [dict(source_id="__body_group__/0", model_foot=None, labels=[ready, unfinished],
                      offset=offsets, radius=radii, jacobian=csr_matrix(linear.reshape(-1, WIDTH)))]
    else:
        balls = [ball(unfinished, 1)]
        if ready_kind == "halfspace":
            halfspaces, slack, labels = jacobians[ready][:1], radii[:1], [ready]
        else:
            balls.insert(0, ball(ready, 0))
    margins = SourceMargins(halfspaces, slack, labels, balls)

    def dependency(key):
        return dict(jacobian=jacobians[key], model_foot=None)

    margins.guard_dependency = dependency
    return margins, ready, unfinished


def solve(ready_kind, *, early=None, ready_curvature=2., unfinished_cap=1.02, retain=False):
    margins, ready, unfinished = domains(ready_kind)
    trust, events, checked = {}, [], []

    def check(point):
        x, _, z, _ = point
        # At the origin the derivatives agree with the affine domains. These
        # curvature terms make their nonlinear decisions differ away from it.
        values = ((ready, .25*x + ready_curvature*x*x, READY_CAP),
                  (unfinished, np.hypot(.6, z + .01*z*z), unfinished_cap))
        outcome = {
            name: dict(rejected=bool(after > cap), before=cap, after=float(after), model_foot=None)
            for name, after, cap in values
        }
        checked.append((point.copy(), outcome))
        return outcome

    options = {} if early is None else {"early_nonlinear_ready_feedback": early}
    step, report = source_conditioned_direction(
        CoupledNormal(), VALUE, (-1., 1.), margins, ACTIVE, control_step_limit=LIMITS,
        nonlinear_source_check=check, trust_state=trust, retain_affine_cuts=retain,
        progress=lambda **event: events.append(event), **options,
    )
    np.testing.assert_array_equal(step, checked[-1][0])
    assert not any(row["rejected"] for row in checked[-1][1].values())
    assert report["local_source_feedback"][-1]["status"] == "passed"
    np.testing.assert_array_equal(LIMITS, [.8, 1., 1., 1.])
    assert np.all(np.abs(step) <= trust["control_step_limit"])
    assert np.hypot(.6, step[2]) <= 1. + 1e-8
    cost, _ = CoupledNormal.objective(step)
    assert cost < 0.
    assert report["bounded_quadratic"]["after_source_projection_quadratic_change"] == pytest.approx(cost)
    return step, report, trust["control_step_limit"], events, checked, ready, unfinished


@pytest.mark.parametrize("ready_kind", ["flight", "body_point", "halfspace"])
def test_ready_failure_contracts_only_its_sparse_dependency_before_other_ball_converges(ready_kind):
    step, report, limits, events, checked, ready, unfinished = solve(ready_kind, early=True, retain=True)
    first_point, first_checks = checked[0]
    np.testing.assert_allclose(first_point, PREFERRED, atol=1e-9)
    assert .25*first_point[0] <= READY_CAP
    assert np.hypot(.6, first_point[2]) > 1.
    assert first_checks[ready]["rejected"] and first_checks[unfinished]["rejected"]

    early_feedback, final_feedback = report["local_source_feedback"]
    assert early_feedback["affine_incomplete"]
    assert early_feedback["failed_source_ids"] == [ready]
    assert early_feedback["changed_global_columns"] == [ACTIVE[0]]
    assert final_feedback["status"] == "passed"
    np.testing.assert_array_equal(limits, [.2, 1., 1., 1.])
    np.testing.assert_allclose(step, [.2, .5, .8, .7], atol=4e-8, rtol=0.)
    clipped = step.copy()
    clipped[1] = first_point[1]
    assert CoupledNormal.objective(step)[0] < CoupledNormal.objective(clipped)[0]

    starts = [row for row in events if row["phase"] == "source_projection_start"]
    assert [row["local_trust_round"] for row in starts].count(0) == 1
    first_projection = events.index(starts[0])
    first_feedback = next(i for i, row in enumerate(events) if row["phase"] == "source_early_nonlinear_ready_feedback")
    assert any("norm_cut" in row["phase"] for row in events[first_projection+1:first_feedback])
    assert events[first_feedback]["failed_source_ids"] == [ready]
    second_solve = next(i for i, row in enumerate(events) if row["phase"] == "bounded_normal_start" and row["local_trust_round"] == 1)
    assert first_projection < first_feedback < second_solve < events.index(starts[1])


def test_unfinished_ball_is_deferred_but_still_enforces_original_nonlinear_guard():
    step, report, limits, events, checked, ready, unfinished = solve(
        "body_point", early=True, ready_curvature=0., unfinished_cap=.93,
    )
    first_point, first_checks = checked[0]
    assert np.hypot(.6, first_point[2]) > 1.
    assert first_checks[unfinished]["rejected"] and not first_checks[ready]["rejected"]
    early = [row for row in events if row["phase"] == "source_early_nonlinear_ready_feedback"]
    assert early and all(row["status"] == "passed" and not row["failed_source_ids"] for row in early)

    contraction, final = report["local_source_feedback"]
    assert not contraction.get("affine_incomplete")
    assert contraction["failed_source_ids"] == [unfinished]
    assert contraction["changed_global_columns"] == [ACTIVE[2]]
    assert final["status"] == "passed"
    np.testing.assert_array_equal(limits, [.8, 1., .5, 1.])
    np.testing.assert_allclose(step, [.8, .2, .5, .7], atol=4e-8, rtol=0.)


@pytest.mark.parametrize("early", [None, False], ids=["omitted", "explicitly-disabled"])
def test_default_waits_for_every_affine_domain_before_nonlinear_feedback(early):
    step, report, limits, events, checked, ready, unfinished = solve("flight", early=early)
    assert report["early_nonlinear_ready_feedback_checks"] == 0
    assert not any(row["phase"] == "source_early_nonlinear_ready_feedback" for row in events)
    first_point, first_checks = checked[0]
    assert np.hypot(.6, first_point[2]) <= 1. + 1e-8
    assert first_checks[ready]["rejected"] and not first_checks[unfinished]["rejected"]
    contraction, _ = report["local_source_feedback"]
    assert not contraction.get("affine_incomplete")
    assert contraction["failed_source_ids"] == [ready]
    assert [row["local_trust_round"] for row in events if row["phase"] == "source_projection_start"].count(0) > 1
    np.testing.assert_array_equal(limits, [.2, 1., 1., 1.])
    np.testing.assert_allclose(step, [.2, .5, .8, .7], atol=4e-8, rtol=0.)


def test_fraction_first_keeps_a_complete_guarded_descent():
    margins, ready, unfinished = domains("flight")
    events = []

    def check(point):
        x, _, z, _ = point
        result = CompleteNonlinearGuardChecks({
            ready: dict(rejected=.25*x + 2.*x*x > READY_CAP),
            unfinished: dict(rejected=np.hypot(.6,z) > 1.02),
        })
        result.objective_cost = 2. + CoupledNormal.objective(point)[0]
        return result

    step, report = source_conditioned_direction(
        CoupledNormal(),VALUE,(-1.,1.),margins,ACTIVE,control_step_limit=LIMITS,
        nonlinear_source_check=check,early_nonlinear_ready_feedback=True,
        early_complete_nonlinear_proposal=True,check_fraction_before_refinement=True,
        progress=lambda **event:events.append(event),
    )
    np.testing.assert_allclose(step,.25*PREFERRED,atol=1.e-9)
    assert report["checked_fraction_before_refinement"] == .25
    assert report["nonlinear_completed_early"]
    probes=[event for event in events if event["phase"]=="source_fraction_before_refinement"]
    assert [event["guards_pass"] for event in probes] == [False,False,True]


def test_full_actual_failure_contracts_its_dependency_before_affine_completion():
    margins, ready, unfinished = domains('flight')
    events = []

    def check(point):
        actual = CompleteNonlinearGuardChecks({
            ready: dict(rejected=False),
            unfinished: dict(rejected=np.hypot(.6,point[2])>.93),
        })
        actual.objective_cost = 2. + CoupledNormal.objective(point)[0]
        return actual

    step, _ = source_conditioned_direction(CoupledNormal(),VALUE,(-1.,1.),margins,ACTIVE,
        control_step_limit=LIMITS,nonlinear_source_check=check,early_nonlinear_ready_feedback=True,
        early_complete_nonlinear_proposal=True,refine_failed_full_checks=True,
        progress=lambda **event:events.append(event))
    first = next(event for event in events if event['phase']=='source_local_trust_feedback')
    assert first['affine_incomplete'] and not first['only_affine_ready_failures_used']
    assert first['failed_source_ids']==[unfinished]
    assert first['changed_global_columns']==[ACTIVE[2]]
    assert not any(row['rejected'] for row in check(step).values())
