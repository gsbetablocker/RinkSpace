"""Early completion needs a full actual guard and full quadratic descent."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.program_tracking.body_tracking import programme_source_step as proposal
from test_source_early_nonlinear_feedback import (
    ACTIVE,
    LIMITS,
    VALUE,
    WIDTH,
    CoupledNormal,
    domains,
)


EARLY_PHASE = "source_early_nonlinear_ready_feedback"
COMPLETE_PHASE = "source_complete_nonlinear_proposal"
PROJECTION_PHASE = "source_projection_start"
AFFINE_RADIUS = 1.
complete = proposal.CompleteNonlinearGuardChecks


def affine_norm(point):
    return np.hypot(.6, point[2])


def guard_row(value, cap):
    return dict(rejected=bool(value > cap), before=cap, after=float(value), model_foot=None)


def solve(margins, check, events, *, solver=None, **options):
    return proposal.source_conditioned_direction(
        CoupledNormal() if solver is None else solver, VALUE, (-1., 1.), margins, ACTIVE,
        control_step_limit=LIMITS, nonlinear_source_check=check,
        early_nonlinear_ready_feedback=True, retain_affine_cuts=True,
        progress=lambda **event: events.append(event), **options,
    )


def test_shortcut_returns_only_projected_rows_and_runs_final_full_check(monkeypatch):
    margins, ready, unfinished = domains("halfspace")
    events, checked, projected = [], [], []
    original_projector = proposal.project_box_halfspaces_dual

    def check(point):
        z = point[2]
        # The first projection fails the true unfinished guard. The first
        # tangent moves into the true domain while the inner affine ball
        # still has positive excess. Its next tangent must stay unapplied.
        result = {
            ready: guard_row(.25 * point[0], .24),
            unfinished: guard_row(np.hypot(.6, z + .01 * z*z), 1.04),
            "unrelated_full_guard": guard_row(point[3]**2, .55),
        }
        checked.append((point.copy(), result))
        return complete(result)

    def remember_projection(point, lower, upper, matrix, slack, **options):
        result, receipt = original_projector(point, lower, upper, matrix, slack, **options)
        projected.append((matrix.copy(), slack.copy(), result.copy()))
        return result, receipt

    monkeypatch.setattr(proposal, "project_box_halfspaces_dual", remember_projection)
    step, receipt = solve(margins, check, events, early_complete_nonlinear_proposal=True)

    assert checked[0][1][unfinished]["rejected"]
    assert all(not row["rejected"] for _, rowset in checked[-2:] for row in rowset.values())
    for final_point, _ in checked[-2:]:
        np.testing.assert_array_equal(final_point, step)
    assert affine_norm(step) > AFFINE_RADIUS + 1e-8
    quadratic_change = CoupledNormal.objective(step)[0]
    assert quadratic_change < 0.
    assert receipt["bounded_quadratic"]["after_source_projection_quadratic_change"] == pytest.approx(quadratic_change)

    completed, = [event for event in events if event["phase"] == COMPLETE_PHASE]
    assert completed["checked_domains"] == len(checked[-1][1])
    assert completed["unneeded_affine_cut_rows"] > 0
    assert completed["outer_nonlinear_cost_and_guard_required"]
    assert receipt["original_nonlinear_guard_required"]
    assert receipt["local_source_feedback"][-1]["status"] == "passed"
    assert receipt["nonlinear_completed_early"]
    linear = receipt["source_projection"]
    assert linear["completed_by_actual_nonlinear_guards"] and not linear["affine_proxy_complete"]
    assert linear["max_affine_norm_excess_by_unit"]["proxy residual"] > 0.

    final_matrix, final_slack, projected_step = projected[-1]
    assert final_matrix.shape[0] > margins[0].shape[0]
    np.testing.assert_array_equal(step, projected_step)
    np.testing.assert_array_equal(receipt["inner_margin_px"], final_slack)
    np.testing.assert_allclose(receipt["adjusted_linear_change_px"], final_matrix @ step, atol=1e-14, rtol=0)
    assert len(receipt["constraint_rows"]) == final_matrix.shape[0]
    assert receipt["constraint_rows"][0] == ready
    actual_cuts = receipt["constraint_rows"][1:]
    assert set(linear["affine_norm_cut_units"]) == set(actual_cuts)
    assert all(name.startswith(unfinished + "/affine_norm_cut_") for name in actual_cuts)
    assert linear["affine_norm_cut_rounds"] == len(projected) - 1


@pytest.mark.parametrize("blocked_by", [
    "unfinished_failure", "other_full_failure", "domain_unavailable", "empty_result", "partial_dict",
])
def test_ready_subset_pass_or_incomplete_actual_result_cannot_skip_proxy(blocked_by):
    margins, ready, unfinished = domains("halfspace")
    events, checked = [], []

    def check(point):
        # A nonlinear scalar residual with the same crossing of its true
        # boundary as the affine norm. The ready halfspace is always satisfied.
        z = point[2]
        over = affine_norm(point) > AFFINE_RADIUS + 1e-8
        ready_pass = guard_row(.25 * point[0], .24)
        if over and blocked_by == "domain_unavailable":
            result = None
        elif over and blocked_by == "empty_result":
            result = complete()
        elif over and blocked_by == "partial_dict":
            result = {ready: ready_pass}
        else:
            result = complete({ready: ready_pass, unfinished: guard_row(z + .01*z*z, .8064 + 2e-8)})
            if blocked_by == "other_full_failure":
                result[unfinished] = guard_row(z, 1.)
                result["untracked_nonlinear_guard"] = guard_row(z + .01*z*z, .8064 + 2e-8)
        checked.append((point.copy(), result))
        return result

    step, receipt = solve(margins, check, events, early_complete_nonlinear_proposal=True)

    assert affine_norm(checked[0][0]) > AFFINE_RADIUS + 1e-8
    first_early = next(event for event in events if event["phase"] == EARLY_PHASE)
    if blocked_by in ("unfinished_failure", "other_full_failure"):
        failed = unfinished if blocked_by == "unfinished_failure" else "untracked_nonlinear_guard"
        assert first_early["status"] == "passed" and first_early["failed_source_ids"] == []
        assert first_early["full_nonlinear_failed_source_ids"] == [failed]
    if blocked_by == "domain_unavailable":
        assert first_early["status"] == "domain_unattributed"
        assert not first_early["full_nonlinear_check_available"]
    if blocked_by == "empty_result":
        assert checked[0][1] == {}
    if blocked_by == "partial_dict":
        assert type(checked[0][1]) is dict and list(checked[0][1]) == [ready]
        assert first_early["status"] == "passed"
        assert first_early["full_nonlinear_failed_source_ids"] == []
    assert not any(event["phase"] == COMPLETE_PHASE for event in events)
    assert not receipt["nonlinear_completed_early"]
    assert receipt["source_projection"]["affine_proxy_complete"]
    assert affine_norm(step) <= AFFINE_RADIUS + 1e-8
    np.testing.assert_array_equal(step, checked[-1][0])
    assert not any(row["rejected"] for row in checked[-1][1].values())


class AnisotropicNormal(CoupledNormal):
    preferred = np.array([0., .8, .95, .7])
    curvature = np.array([100., 1., 1., 1.])

    def solve_scaled(self, rhs, scale, *, damp):
        return self.preferred.copy(), {}

    @classmethod
    def objective(cls, point):
        delta = point - cls.preferred
        gradient = cls.curvature * delta
        change = .5 * (delta @ gradient - cls.preferred @ (cls.curvature * cls.preferred))
        return float(change), gradient


def test_non_descent_cannot_take_shortcut_even_when_actual_guards_pass():
    fixture, ready, unfinished = domains("halfspace")
    # Orthogonal projection trades a cheap move in column 8 for an expensive
    # move in column 3. It is geometrically feasible but raises the actual
    # anisotropic quadratic relative to the origin.
    matrix = csr_matrix(([-1., 1.], ([0, 0], ACTIVE[:2])), shape=(1, WIDTH))
    margins = proposal.SourceMargins(matrix, np.zeros(1), [ready], fixture.image_balls)
    events, checked = [], []

    def check(point):
        checked.append(point.copy())
        return complete({ready: guard_row(point[1] - point[0], 1e-8), unfinished: guard_row(point[2], 1.)})

    with pytest.raises(ArithmeticError, match="no resolved descent in the full quadratic"):
        solve(margins, check, events, solver=AnisotropicNormal(), early_complete_nonlinear_proposal=True)

    assert affine_norm(checked[0]) > AFFINE_RADIUS + 1e-8
    assert AnisotropicNormal.objective(checked[0])[0] > 0.
    assert next(event for event in events if event["phase"] == EARLY_PHASE)["full_nonlinear_failed_source_ids"] == []
    assert len([event for event in events if event["phase"] == PROJECTION_PHASE]) > 1
    assert not any(event["phase"] == COMPLETE_PHASE for event in events)


@pytest.mark.parametrize("options", [{}, {"early_complete_nonlinear_proposal": False}], ids=["omitted", "disabled"])
def test_legacy_early_feedback_still_finishes_affine_norm_projection(options):
    margins, ready, unfinished = domains("halfspace")
    events, checked = [], []

    def check(point):
        checked.append(point.copy())
        return complete({ready: guard_row(.25 * point[0], .24), unfinished: guard_row(point[2], 1.)})

    step, receipt = solve(margins, check, events, **options)

    assert affine_norm(checked[0]) > AFFINE_RADIUS + 1e-8
    assert next(event for event in events if event["phase"] == EARLY_PHASE)["full_nonlinear_failed_source_ids"] == []
    assert not any(event["phase"] == COMPLETE_PHASE for event in events)
    assert receipt["source_projection"]["affine_proxy_complete"]
    assert not receipt["source_projection"]["completed_by_actual_nonlinear_guards"]
    np.testing.assert_allclose(step, [.8, .2, .8, .7], atol=4e-8, rtol=0)


@pytest.mark.parametrize("recheck", ["domain_unavailable", "missing_marker", "empty_marker", "unattributed_failure"])
def test_final_check_must_reconfirm_actual_completion(recheck):
    margins, ready, unfinished = domains("halfspace")
    # This case intentionally has no registered dependency for a new guard.
    if recheck == "unattributed_failure":
        margins.guard_dependency = None
    events, checked = [], []

    def check(point):
        checked.append(point.copy())
        passed = {ready: guard_row(.25 * point[0], .24), unfinished: guard_row(point[2], 1.)}
        if len(checked) == 1:
            return complete(passed)
        if recheck == "domain_unavailable":
            return None
        if recheck == "missing_marker":
            return passed
        if recheck == "empty_marker":
            return complete()
        return complete({**passed, "guard_without_dependency": guard_row(2., 1.)})

    step, receipt = solve(margins, check, events, early_complete_nonlinear_proposal=True)

    assert any(event["phase"] == COMPLETE_PHASE for event in events)
    assert len(checked) == 2 and affine_norm(step) > AFFINE_RADIUS + 1e-8
    for attempted in checked:
        np.testing.assert_array_equal(attempted, step)
    assert receipt["local_source_feedback"][-1]["status"] == "domain_unattributed"
    assert receipt["original_nonlinear_guard_required"]
    assert not receipt["nonlinear_completed_early"]
    linear = receipt["source_projection"]
    assert not linear["completed_by_actual_nonlinear_guards"]
    assert not linear["affine_proxy_complete"]
    assert linear["early_complete_check_not_reconfirmed"] == "domain_unattributed"


def test_final_guard_contraction_reprojects_without_reusing_completion_receipt():
    margins, ready, unfinished = domains("halfspace")
    events = []

    def check(point):
        # The first actual check is allowed. Recheck catches the failure, and
        # the original nonlinear trust callback can trace it to the z column.
        rechecking = any(event["phase"] == COMPLETE_PHASE for event in events)
        cap = .6 if rechecking else 1.
        return complete({ready: guard_row(.25 * point[0], .24), unfinished: guard_row(point[2], cap)})

    step, receipt = solve(margins, check, events, early_complete_nonlinear_proposal=True)

    assert len([event for event in events if event["phase"] == COMPLETE_PHASE]) == 1
    assert not receipt["nonlinear_completed_early"]
    assert receipt["source_projection"]["affine_proxy_complete"]
    assert not receipt["source_projection"]["completed_by_actual_nonlinear_guards"]
    assert "early_complete_check_not_reconfirmed" not in receipt["source_projection"]
    contracted, passed = receipt["local_source_feedback"]
    assert contracted["status"] == "contracted" and contracted["failed_source_ids"] == [unfinished]
    assert contracted["changed_global_columns"] == [int(ACTIVE[2])]
    assert passed["status"] == "passed"
    np.testing.assert_allclose(step, [.8, .2, .5, .7], atol=4e-8, rtol=0)


def test_shortcut_configuration_requires_actual_early_feedback():
    margins, _, _ = domains("halfspace")
    with pytest.raises(ValueError, match="requires full early nonlinear feedback"):
        proposal.source_conditioned_direction(
            CoupledNormal(), VALUE, (-1., 1.), margins, ACTIVE, control_step_limit=LIMITS,
            nonlinear_source_check=lambda point: {}, early_complete_nonlinear_proposal=True,
        )
