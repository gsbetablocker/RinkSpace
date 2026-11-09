"""Regression coverage for affine source cuts across local trust solves."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix, vstack

from calibration.program_tracking.body_tracking import programme_source_step as source_step
from calibration.program_tracking.body_tracking.weighted_blocks import block


SELECTION = np.array([2, 5, 9])
BAND_RADIUS = 2.
ORIGINAL_P95_LIMIT = .5
SOURCE_ID = "runner"
VALUE = {"residual": np.array([2.])}


class CoupledNormal:
    scale_inv = np.array([2., .5, 1.25])
    preferred = np.array([1.8, .7, .4])
    hessian = np.array([[3., 1.1, 0.], [1.1, 2.5, 0.], [0., 0., 1.7]])

    def solve_scaled(self, rhs, scale, *, damp):
        np.testing.assert_array_equal(rhs, -VALUE["residual"])
        np.testing.assert_array_equal(scale, 1 / self.scale_inv)
        assert damp == .07
        return self.preferred.copy(), {}

    def scaled_normal_objective(self, rhs, scale, *, damp):
        def objective(point):
            offset = point - self.preferred
            gradient = self.hessian @ offset
            value = .5 * (offset @ gradient - self.preferred @ self.hessian @ self.preferred)
            return float(value), gradient

        return objective


def source_margins(offset, pixel_chain):
    matrix = np.zeros((2, 11))
    matrix[:, SELECTION] = pixel_chain
    contour = SimpleNamespace(
        rows=[{"id": SOURCE_ID, "foot": 1}], point_rows=np.array([0]),
        source_uv=np.array([[150., 50.]]),
        proposal_pixel_prediction=np.array([[150., 50.]]) + offset,
        proposal_pixel_jacobian=csr_matrix(matrix),
    )
    # The source prediction starts inside the zero-cost uncertainty band.
    source_rows = [{"id": SOURCE_ID, "outside_visible_band_px": [0.]}]
    value = {
        "action_metrics": {"visible_runner_contours": source_rows},
        "weighted_blocks": [block("full_source_skate_structure", np.zeros(2), csr_matrix((2, 11)))],
    }
    margins = source_step.contour_margins(value, contour, {SOURCE_ID: (0., BAND_RADIUS)})
    ball, = margins.image_balls
    assert ball["guard_margin_px"] == ORIGINAL_P95_LIMIT
    np.testing.assert_allclose(ball["radius"], [BAND_RADIUS + .8 * ORIGINAL_P95_LIMIT])
    return margins


def check_pixel(move, offset, pixel_chain, *, curved=False):
    pixel = offset + pixel_chain @ move
    if curved:
        x, y = move[:2]
        pixel = pixel + np.array([.8 * (x*x + y*y), .15*x*y])
    outside = max(np.linalg.norm(pixel) - BAND_RADIUS, 0.)
    return {SOURCE_ID: dict(model_foot=1, rejected=outside > ORIGINAL_P95_LIMIT,
                            before=0., after=outside, original_radius_px=BAND_RADIUS)}


@pytest.mark.parametrize("reuse", [False, True], ids=["default", "reuse-cuts-and-reserve"])
def test_affine_cuts_remain_valid_when_nonlinear_guard_contracts_local_box(monkeypatch, reuse):
    offset = np.array([1.7, .1])
    pixel_chain = np.array([[.35, .05, 0.], [.3, 1.4, 0.]])
    source = source_margins(offset, pixel_chain)
    guard = np.zeros((1, 11))
    guard[0, SELECTION] = [1., .2, 0.]
    margins = source_step.SourceMargins(
        vstack((source[0], csr_matrix(guard)), format="csr"), np.r_[source[1], .95],
        [*source[2], "coupled_original_halfspace"], source.image_balls,
    )
    original_rows = len(margins[1])
    trust = np.array([1., 1.4, .7])
    lower, upper = np.array([-.9, -1., -.5]), np.array([.95, 1.35, .5])
    checked_moves, starts, warm_starts = [], [], []
    original_project = source_step.project_box_halfspaces_dual

    def progress(**event):
        if event["phase"] == "source_projection_start":
            starts.append(event)

    def project(*args, state=None, **kwargs):
        initial = state.get("dual") if state is not None else None
        warm_starts.append(None if initial is None else initial.copy())
        return original_project(*args, state=state, **kwargs)

    monkeypatch.setattr(source_step, "project_box_halfspaces_dual", project)

    def original_check(move):
        outcome = check_pixel(move, offset, pixel_chain, curved=True)
        checked_moves.append((move.copy(), outcome[SOURCE_ID]))
        return outcome

    tolerance_fraction = .02 if reuse else 0.
    options = dict(retain_affine_cuts=True, affine_guard_reserve_fraction=tolerance_fraction) if reuse else {}
    trust_state = {}
    step, report = source_step.source_conditioned_direction(
        CoupledNormal(), VALUE, (lower, upper), margins, SELECTION, control_step_limit=trust,
        nonlinear_source_check=original_check, trust_state=trust_state, progress=progress, **options,
    )
    assert checked_moves[0][1]["rejected"]
    assert not checked_moves[-1][1]["rejected"]
    assert checked_moves[-1][1]["after"] <= ORIGINAL_P95_LIMIT
    np.testing.assert_array_equal(step, checked_moves[-1][0])
    feedback = report["local_source_feedback"]
    assert any(row["status"] == "contracted" and row["changed_global_columns"] == [2, 5]
               for row in feedback)
    assert feedback[-1]["status"] == "passed"
    assert all(row["original_nonlinear_limits_unchanged"] for row in feedback)

    final_limits = trust_state["control_step_limit"]
    assert np.all(final_limits[:2] < trust[:2])
    assert final_limits[2] == trust[2]
    assert step[2] == pytest.approx(CoupledNormal.preferred[2] / CoupledNormal.scale_inv[2])
    assert np.all(step >= lower) and np.all(step <= upper)
    assert np.all(np.abs(step) <= final_limits)
    assert np.max(margins[0][:, SELECTION] @ step - margins[1]) <= 1e-10
    assert np.max(np.subtract(report["adjusted_linear_change_px"], report["inner_margin_px"])) <= 1e-10

    projection = report["source_projection"]
    assert projection["retained_affine_cuts_across_local_trust"] is reuse
    assert projection["affine_proposal_tolerance_fraction_of_original_guard_margin"] == tolerance_fraction
    proposal_radius = margins.image_balls[0]["radius"][0]
    allowed_excess = max(1e-8, tolerance_fraction * ORIGINAL_P95_LIMIT)
    for move, _ in checked_moves:
        assert np.linalg.norm(offset + pixel_chain @ move) <= proposal_radius + allowed_excess + 1e-12

    assert starts[0]["rows"] == original_rows
    assert max(event["rows"] for event in starts if event["local_trust_round"] == 0) > original_rows
    later = [i for i, event in enumerate(starts) if event["local_trust_round"] != 0]
    assert later
    if reuse:
        assert all(first["rows"] <= second["rows"] for first, second in zip(starts, starts[1:]))
        assert warm_starts[later[0]] is not None and np.any(warm_starts[later[0]] > 0)
    else:
        assert starts[later[0]]["rows"] == original_rows
        assert warm_starts[later[0]] is None


def test_affine_proposal_reserve_leaves_original_pixel_guard_unchanged():
    offset = np.array([1.9, 0.])
    proposal_radius = BAND_RADIUS + .8 * ORIGINAL_P95_LIMIT
    # The unconstrained point exceeds the proposal radius by .005 px,
    # through a tangential displacement with zero initial radial derivative.
    desired_y = np.sqrt((proposal_radius + .005)**2 - offset[0]**2)
    pixel_chain = np.array([[0., 0., 0.], [0., desired_y / 1.4, 0.]])
    margins = source_margins(offset, pixel_chain)

    def check(move):
        return check_pixel(move, offset, pixel_chain)

    def run(fraction):
        return source_step.source_conditioned_direction(
            CoupledNormal(), VALUE, (-2., 2.), margins, SELECTION, control_step_limit=[1., 1.5, .7],
            nonlinear_source_check=check, affine_guard_reserve_fraction=fraction,
        )

    strict, _ = run(0.)
    relaxed, report = run(.02)
    assert np.linalg.norm(offset + pixel_chain @ strict) <= proposal_radius + 1e-8
    relaxed_norm = np.linalg.norm(offset + pixel_chain @ relaxed)
    assert relaxed_norm == pytest.approx(proposal_radius + .005, abs=1e-9)
    assert report["source_projection"]["max_affine_pixel_excess_px"] == pytest.approx(.005, abs=1e-9)
    for move in (strict, relaxed):
        measurement = check(move)[SOURCE_ID]
        assert not measurement["rejected"]
        assert ORIGINAL_P95_LIMIT - measurement["after"] >= .18 * ORIGINAL_P95_LIMIT - 1e-12


@pytest.mark.parametrize("fraction", [-.01, .2, .3, np.nan])
def test_affine_guard_reserve_fraction_rejects_invalid_interval(fraction):
    with pytest.raises(ValueError, match="inside the original unused guard reserve"):
        source_step.source_conditioned_direction(
            CoupledNormal(), VALUE, (-1., 1.), None, SELECTION,
            affine_guard_reserve_fraction=fraction,
        )


def test_positive_affine_guard_reserve_requires_original_nonlinear_check():
    with pytest.raises(ValueError, match="requires original nonlinear feedback"):
        source_step.source_conditioned_direction(
            CoupledNormal(), VALUE, (-1., 1.), None, SELECTION,
            affine_guard_reserve_fraction=.02,
        )
