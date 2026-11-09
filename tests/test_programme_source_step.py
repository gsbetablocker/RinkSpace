"""Small scaled-step tests. The nonlinear acceptance path remains in the solver."""

from itertools import islice
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye, vstack

from calibration.program_tracking.body_tracking.fullbody_fit import FullBodyProblem
from calibration.program_tracking.body_tracking.programme_action_factors import (
    turn_episode_image_checks,
    turn_episode_image_metrics,
)
from calibration.program_tracking.body_tracking.programme_source_step import (
    SourceMargins,
    all_source_margins,
    backtracking_fractions,
    contour_margins,
    project_box_halfspaces,
    project_box_halfspaces_dual,
    source_conditioned_direction,
    solve_bounded_quadratic,
)
from calibration.program_tracking.body_tracking.spin_grasp_factors import region_terms
from calibration.program_tracking.body_tracking.weighted_blocks import block, pack_weighted


class ScaledProposal:
    def __init__(self, step, scale_inv):
        self.step = np.asarray(step, float)
        self.scale_inv = np.asarray(scale_inv, float)

    def solve_scaled(self, rhs, scale, *, damp):
        np.testing.assert_array_equal(rhs, [-2.0])
        np.testing.assert_array_equal(scale, 1 / self.scale_inv)
        assert damp == .07
        return self.step.copy(), {"mock_scaled_solve": True}

    def scaled_normal_objective(self, rhs, scale, *, damp):
        return lambda z: (float(.5*z@z-z@self.step), z-self.step)


def proposal(step, scale_inv, *, trust=None, bounds=(-100., 100.), margin=None):
    n = len(step)
    # Model non-contiguous active columns in the larger saved state.
    selection = np.arange(1, 2 * n, 2)
    raw = np.zeros((1, 2 * n))
    if margin is None:
        slack = 1.
    else:
        gradient, slack = margin
        raw[0, selection] = gradient
    return source_conditioned_direction(
        ScaledProposal(step, scale_inv), {"residual": np.array([2.])}, bounds,
        (csr_matrix(raw), np.array([slack]), ["clear_source"]), selection,
        control_step_limit=trust,
    )


def test_unconfigured_step_preserves_legacy_scaled_unit_norm():
    move, report = proposal([3., 4.], [2., .5])
    np.testing.assert_allclose(move, [.3, 1.6])
    assert report["raw_scaled_step_norm"] == 5.
    assert report["initial_step_shrink"] == 5.
    assert report["dimension_dependent_unit_norm_cap"]


def test_component_box_preserves_each_increment_when_more_controls_repeat():
    step = [3., -4.]
    scale_inv = [2., .5]
    radius = [.10, .20]
    small, one = proposal(step, scale_inv, trust=radius)
    np.testing.assert_allclose(small, [.10, -.20])
    repeats = 128
    extended, many = proposal(np.tile(step, repeats), np.tile(scale_inv, repeats),
                              trust=np.tile(radius, repeats))
    np.testing.assert_allclose(extended.reshape(-1, 2), np.tile(small, (repeats, 1)))
    assert many["raw_scaled_step_norm"] > one["raw_scaled_step_norm"]
    assert many["initial_step_shrink"] == one["initial_step_shrink"] == 1.
    assert many["previous_global_box_shrink"] == one["previous_global_box_shrink"] == 40.
    assert not many["dimension_dependent_unit_norm_cap"]


def test_active_source_halfspace_projects_the_scaled_step():
    move, report = proposal([1., 0.], [2., .5], trust=[.1, .1], margin=([1., 1.], .02))
    assert report["initial_linear_change_px"] == pytest.approx([.1])
    assert move.sum() == pytest.approx(.02)
    assert report["adjusted_linear_change_px"] == pytest.approx([move.sum()])
    assert report["original_nonlinear_guard_required"]


@pytest.mark.parametrize(
    "step,bounds,gradient,slack,expected",
    [
        ([1., 0.], ([-1., -.01], [1., 1.]), [1., 1.], 0., [.01, -.01]),
        ([2., 2.], (-1., 1.), [1., -1.], -.1, [0., .10]),
    ],
)
def test_source_and_control_box_projection_preserves_their_intersection(
    step, bounds, gradient, slack, expected,
):
    radius = [.1, .1]
    move, report = proposal(step, [1., 1.], trust=radius, bounds=bounds, margin=(gradient, slack))
    np.testing.assert_allclose(move, expected, rtol=0., atol=2e-10)
    assert np.all(np.abs(move) <= radius)
    assert np.all(move >= bounds[0]) and np.all(move <= bounds[1])
    actual = np.asarray(gradient) @ move
    assert report["adjusted_linear_change_px"] == pytest.approx([actual])
    assert actual <= slack + 2e-10
    assert report["original_nonlinear_guard_required"]


@pytest.mark.parametrize(
    "local_point,local_planes,local_expected",
    [
        pytest.param([1., 2.], [[1., 0.], [1., 1.]], [-.5, .5], id="release-correction"),
        pytest.param([2., 2.], [[1., 0.], [-1., 1.]], [0., 0.], id="new-violation"),
    ],
)
def test_sparse_projection_matches_intersection_with_many_inactive_planes(
    local_point, local_planes, local_expected,
):
    point = np.linspace(-.2, .2, 128)
    columns = [2, 71]
    point[columns] = local_point
    expected = point.copy()
    expected[columns] = local_expected
    local = np.zeros((2, len(point)))
    local[:, columns] = local_planes
    planes = vstack([csr_matrix(local), eye(len(point), format="csr")], format="csr")
    slack = np.r_[0., 0., np.full(len(point), 3.)]
    upper = np.full_like(point, 3.)
    projected, report = project_box_halfspaces(point, -upper, upper, planes, slack)
    np.testing.assert_allclose(projected, expected, rtol=0., atol=2e-10)
    assert report["projection_cycles"] > 1
    assert report["linear_source_max_excess"] <= 1e-10


@pytest.mark.parametrize(
    "point,box,rows,source_limits,expected",
    [
        # Both nonorthogonal planes and the second coordinate's lower bound
        # are active. Their outward normals with positive weights explain p-x.
        pytest.param(
            [.08, 0., .08, .017], ([-.1, -.01, -.1, -.05], [.1, .1, .1, .05]),
            [[4., 4., 0., 0.], [8., 0., 8., 0.], [0., -.5, 1., 0.], [0., 0., 0., 0.]],
            [0., .4, .1, 0.], [.01, -.01, .04, .017], id="active-box-and-scaled-planes",
        ),
        # Opposing nearly parallel normals make cyclic correction slow.
        pytest.param(
            [0., .3], ([-.4, -.5], [.4, .4]), [[1., .12], [-1., .12]],
            [0., 0.], [0., 0.], id="shallow-intersection",
        ),
    ],
)
def test_dual_projection_preserves_box_and_matches_dykstra_with_gap_certificate(
    point, box, rows, source_limits, expected,
):
    point = np.asarray(point)
    lower, upper = map(np.asarray, box)
    planes = csr_matrix(rows)
    slack = np.asarray(source_limits)
    reference, old_report = project_box_halfspaces(point, lower, upper, planes, slack)
    projected, report = project_box_halfspaces_dual(point, lower, upper, planes, slack)

    assert old_report["projection_cycles"] > 1
    np.testing.assert_allclose(projected, reference, rtol=0., atol=1e-8)
    np.testing.assert_allclose(projected, expected, rtol=0., atol=1e-8)
    np.testing.assert_array_equal(lower, box[0])
    np.testing.assert_array_equal(upper, box[1])
    np.testing.assert_array_equal(slack, source_limits)
    assert np.all(lower <= projected) and np.all(projected <= upper)
    maximum_excess = np.max(planes @ projected - slack, initial=0.)
    assert maximum_excess <= 1e-10
    assert report["linear_source_max_excess"] == pytest.approx(maximum_excess, abs=1e-14)

    primal = .5 * np.sum((projected - point)**2)
    old_primal = .5 * np.sum((reference - point)**2)
    optimum = .5 * np.sum((expected - point)**2)
    assert primal == pytest.approx(old_primal, rel=0., abs=2e-10)
    assert report["projection_squared_distance"] == pytest.approx(2 * primal, abs=1e-14)
    gap = report["duality_gap"]
    assert 0. <= gap <= 1e-8 + 1e-7 * max(1., primal)
    # Reported dual lower bound must also bound the known analytic optimum.
    assert primal - gap <= optimum + 1e-12
    assert report["relative_duality_gap"] == pytest.approx(gap / max(1., primal), abs=1e-15)
    assert report["original_nonlinear_limits_unchanged"]


def test_dual_working_set_adds_an_initially_satisfied_halfspace_after_projection():
    point = np.array([2., 0.])
    planes = csr_matrix([[1., 1.], [0., -1.], [1., 0.]])
    slack = np.array([0., .5, 2.5])
    np.testing.assert_array_equal(planes @ point > slack, [True, False, False])
    # Projecting onto x+y <= 0 alone gives (1, -1), which violates -y <= .5.
    assert (planes @ [1., -1.] - slack)[1] > 0.

    projected, report = project_box_halfspaces_dual(point, [-3., -3.], [3., 3.], planes, slack)
    np.testing.assert_allclose(projected, [.5, -.5], rtol=0., atol=1e-9)
    assert np.max(planes @ projected - slack) <= 1e-10
    assert report["linear_source_max_excess"] <= 1e-10
    assert report["working_updates"] > 0
    assert report["working_rows"] == 2
    assert report["original_constraint_rows"] == 3


@pytest.mark.parametrize("source_limits", [[], [0., .5]], ids=["no-rows", "redundant-zero-rows"])
def test_dual_projection_with_only_zero_rows_keeps_exact_box_projection(source_limits):
    planes = csr_matrix((len(source_limits), 3))
    point = np.array([-1., .25, 2.])
    lower = np.array([-.1, -.2, -.3])
    upper = np.array([.2, .3, .4])
    projected, report = project_box_halfspaces_dual(point, lower, upper, planes, source_limits)
    np.testing.assert_array_equal(projected, [-.1, .25, .4])
    assert report["linear_source_max_excess"] == report["duality_gap"] == 0.


def test_dual_projection_rejects_negative_slack_for_a_zero_row():
    planes = csr_matrix([[1., 1.], [0., 0.]])
    with pytest.raises(ValueError, match="An empty source row excludes every step"):
        project_box_halfspaces_dual([.1, .2], [-.5, -.5], [.5, .5], planes, [.5, -.1])


@pytest.mark.parametrize("radius", ([.1], [.1, 0.], [.1, np.nan], [.1, np.inf]))
def test_invalid_component_limits_are_rejected(radius):
    with pytest.raises(ValueError, match="Whole-state trust limits"):
        proposal([1., 1.], [1., 1.], trust=radius)


def test_one_remote_bad_column_does_not_throttle_an_independent_trajectory():
    move, report = proposal([1000., .08], [1., 1.], trust=[.1, .1])
    np.testing.assert_allclose(move, [.1, .08], atol=1e-12)
    assert report['previous_global_box_shrink'] == 10000.
    assert report['initial_step_shrink'] == 1.


def test_backtracking_halves_past_four_trials_to_float64_resolution():
    precision = np.finfo(np.float64)
    # Bound the test itself so a nonterminating generator fails promptly.
    cap = 2 * precision.nmant
    fractions = np.asarray(list(islice(backtracking_fractions(), cap)))
    assert 6 <= len(fractions) < cap
    np.testing.assert_array_equal(fractions[:6], [1., .5, .25, .125, .0625, .03125])
    np.testing.assert_array_equal(fractions[1:], .5 * fractions[:-1])
    assert fractions[-1] > precision.eps
    assert .5 * fractions[-1] == precision.eps


def test_coupled_bounded_quadratic_keeps_cross_terms():
    H=np.array([[2.,1.],[1.,2.]])
    b=np.array([1.,.05])
    objective=lambda z:(float(.5*z@H@z-b@z),H@z-b)
    step, record=solve_bounded_quadratic(objective,np.linalg.solve(H,b),np.full(2,-.3),np.full(2,.3))
    np.testing.assert_allclose(step,[.3,-.125],atol=1e-8)
    assert record['projected_gradient_inf']<1e-7


def _contour(name, uv, chain, radius, weight, *, raw_proposal=False):
    region = dict(centre_uv=np.zeros(2), region_shape_px2=np.eye(2)*radius**2,
                  objective_weight=weight, reliability=.72)
    terms = [region_terms(point, region) for point in uv]
    residual = np.concatenate([value for value, _, _ in terms])
    jacobian = np.concatenate([derivative@pixel_chain for (_, derivative, _), pixel_chain in zip(terms, chain)])
    metrics = [dict(id=name, outside_visible_band_px=_pixel_distances(uv, radius))]
    contour = SimpleNamespace(rows=[dict(id=name)], point_rows=np.zeros(len(uv), dtype=int),
                              last=dict(residual=residual, metrics=dict(visible_runner_contours=metrics)))
    if raw_proposal:
        contour.source_uv = np.broadcast_to([160., 90.], uv.shape)
        contour.proposal_pixel_prediction = contour.source_uv + uv
        contour.proposal_pixel_jacobian = csr_matrix(chain.reshape(uv.size, -1))
    return contour, residual, csr_matrix(jacobian), metrics


def _pixel_distances(uv, radius=0.):
    return np.maximum(np.linalg.norm(uv, axis=1)-radius, 0.)


def _p95_gradient(uv, chain, radius=0.):
    h = 1e-5
    return np.array([
        (np.percentile(_pixel_distances(uv+h*direction, radius), 95)
         - np.percentile(_pixel_distances(uv-h*direction, radius), 95))/(2*h)
        for direction in np.moveaxis(chain, -1, 0)
    ])


def test_contour_margin_recovers_raw_pixel_p95_with_room_inside_source_limit():
    radius = 2.75
    uv = np.array([[2.4, 1.8], [-8., 6.], [5., 12.]])
    chain = np.array([[[1., 3.], [4., .5]], [[1.5, -2.], [.75, .2]], [[3., .6], [-.1, 2.]]])
    contour, residual, jacobian, rows = _contour('assigned', uv, chain, radius, weight=.29)
    terms = block('full_source_skate_structure', residual, jacobian)
    # Point-wise solver weights and the region sigma both cancel out of the pixel gradient.
    _, _, weighted = pack_weighted([terms], {terms.name: np.repeat([.13, 2.7, .43], 2)}, jacobian=True)
    old = 120.
    value = dict(action_metrics=dict(visible_runner_contours=rows), weighted_blocks=weighted)
    gradient, slack, labels = contour_margins(value, contour, {'assigned': (old, radius)})
    assert isinstance(gradient, csr_matrix)
    assert labels == ['assigned']
    np.testing.assert_allclose(gradient.toarray()[0], _p95_gradient(uv, chain, radius), atol=2e-9)
    assert slack[0] == pytest.approx(.8*(old+.25*radius-np.percentile(_pixel_distances(uv, radius), 95)))


def test_contour_point_halfspaces_protect_p95_when_an_inside_band_point_changes_rank():
    radius = 4.
    uv = np.array([[1., 0.], [9., 0.], [10., 0.]])
    chain = np.zeros((3, 2, 3))
    chain[:, 0, :] = np.eye(3)
    contour, residual, jacobian, rows = _contour("assigned", uv, chain, radius, .29, raw_proposal=True)
    terms = block("full_source_skate_structure", residual, jacobian)
    _, _, weighted = pack_weighted([terms], {terms.name: np.repeat([.13, 2.7, .43], 2)}, jacobian=True)
    value = dict(action_metrics=dict(visible_runner_contours=rows), weighted_blocks=weighted)
    limits = {"assigned": (5.5, radius)}
    gradient, slack, labels = contour_margins(value, contour, limits)

    np.testing.assert_array_equal(weighted[0].residual[:2], [0., 0.])
    np.testing.assert_array_equal(weighted[0].jacobian[:2].toarray(), np.zeros((2, 3)))
    assert labels == ["assigned/point_0", "assigned/point_1", "assigned/point_2", "assigned"]
    np.testing.assert_allclose(gradient.toarray(), np.vstack([np.eye(3), [0., .1, .9]]), atol=1e-12)
    np.testing.assert_allclose(slack, [3.48, .48, .48, .48], rtol=0., atol=1e-12)

    raw_step = np.array([12., -1., -1.])
    upper = np.full_like(raw_step, 12.)
    assert (gradient[-1] @ raw_step).item() == pytest.approx(-1.)
    old_step, _ = project_box_halfspaces(raw_step, -upper, upper, gradient[-1:], slack[-1:])
    np.testing.assert_array_equal(old_step, raw_step)
    guarded_step, _ = project_box_halfspaces(raw_step, -upper, upper, gradient, slack)
    np.testing.assert_allclose(guarded_step, [3.48, -1., -1.], rtol=0., atol=1e-10)

    assert limits == {"assigned": (5.5, 4.)}
    original_p95_limit = 5.5 + .25 * radius
    old_trial = _pixel_distances(uv + chain @ old_step, radius)
    assert np.argmax(old_trial) == 0
    assert np.percentile(old_trial, 95) == pytest.approx(8.6)
    assert np.percentile(old_trial, 95) > original_p95_limit
    assert np.percentile(_pixel_distances(uv + chain @ guarded_step, radius), 95) <= original_p95_limit


@pytest.mark.parametrize(
    "offset_x,expected_radius",
    [pytest.param(1., 1.1, id="pure-tangent"), pytest.param(0., .6, id="zero-offset")],
)
def test_contour_affine_pixel_ball_protects_tangent_and_zero_offset(offset_x, expected_radius):
    uv = np.array([[offset_x, 0.]])
    radius = .5
    old_p95 = max(0., offset_x - radius)
    limits = {"assigned": (old_p95, radius)}
    # Only the y pixel component moves on selected controls. The second
    # selected control is independent, and scaling differs across controls.
    chain = np.array([[[4., 0., 3., 0.], [0., 1., 0., 0.]]])
    selection = np.array([1, 3])
    raw, residual, jacobian, rows = _contour("assigned", uv, chain, radius, 1., raw_proposal=True)
    legacy, _, _, _ = _contour("assigned", uv, chain, radius, 1.)
    value = dict(residual=np.array([2.]), action_metrics=dict(visible_runner_contours=rows),
                 weighted_blocks=[block("full_source_skate_structure", residual, jacobian)])
    margins = contour_margins(value, raw, limits)
    gradient, slack, labels = margins
    assert len(margins) == 3 and len(labels) == len(slack)
    assert len(margins.image_balls) == 1
    ball, = margins.image_balls
    np.testing.assert_array_equal(ball["offset"], uv)
    assert ball["radius"] == pytest.approx([expected_radius])

    proposal = ScaledProposal([2., .1], [2., .5])
    legacy_margins = contour_margins(value, legacy, limits)
    if legacy_margins is not None:
        assert legacy_margins.image_balls == []
    before, old_report = source_conditioned_direction(
        proposal, value, (-2., 2.), legacy_margins, selection, control_step_limit=[1.1, .3],
    )
    np.testing.assert_allclose(before, [1., .2], rtol=0., atol=1e-12)
    assert old_report["source_projection"]["affine_pixel_domains"] == 0
    np.testing.assert_array_equal(gradient[:, selection] @ before, np.zeros(len(slack)))
    assert np.linalg.norm(uv + chain[..., selection] @ before) > expected_radius

    step, report = source_conditioned_direction(
        proposal, value, (-2., 2.), margins, selection, control_step_limit=[1.1, .3],
    )
    expected_y = np.sqrt(expected_radius**2 - offset_x**2)
    np.testing.assert_allclose(step, [expected_y, .2], rtol=0., atol=3e-8)
    trial_uv = uv + chain[..., selection] @ step
    assert np.linalg.norm(trial_uv) <= expected_radius + 1e-8
    assert _pixel_distances(trial_uv, radius)[0] <= old_p95 + .25 * radius
    assert limits == {"assigned": (old_p95, radius)}
    projection = report["source_projection"]
    assert projection["affine_pixel_domains"] == 1
    assert projection["affine_pixel_cut_rounds"] > 0
    assert projection["max_affine_pixel_excess_px"] <= 1e-8
    assert report["original_nonlinear_guard_required"]


@pytest.mark.parametrize("dimension,declared_unit", [(2, None), (3, "m/s²"), (17, None)])
@pytest.mark.parametrize("sign", [-1., 1.])
def test_affine_norm_ball_cuts_both_tangents_on_scaled_active_columns(dimension, declared_unit, sign):
    offset = np.zeros((2, dimension))
    offset[:, 0] = [.6, .4]
    radius = np.array([1., .5])
    selection = np.array([1, 4, 7])
    chain = np.zeros((2, dimension, 8))
    chain[:, :, 0] = 9.  # Frozen control, excluded before applying the norm cuts.
    chain[0, -1, selection[0]] = 2.
    chain[1, -1, selection[1]] = -.75
    ball = dict(offset=offset, radius=radius, labels=["first", "second"],
                jacobian=csr_matrix(chain.reshape(offset.size, -1)))
    if declared_unit is not None:
        ball["unit"] = declared_unit
    balls = [ball]
    if dimension != 2:
        # A feasible legacy pixel ball still counts in the pixel receipt.
        balls.append(dict(offset=np.zeros((1, 2)), radius=np.ones(1),
                          labels=["pixel"], jacobian=csr_matrix((2, 8))))
    margins = SourceMargins(csr_matrix((0, 8)), np.empty(0), [], balls)
    events = []
    proposal = ScaledProposal(sign * np.array([2., 2., .12]), [2., .5, 1.])
    step, receipt = source_conditioned_direction(
        proposal, dict(residual=np.array([2.])), (-2., 2.), margins, selection,
        control_step_limit=[2., 2., .5], progress=lambda **event: events.append(event),
    )
    np.testing.assert_allclose(step, sign * np.array([.4, .4, .12]), rtol=0, atol=4e-8)
    trial = offset + chain[..., selection] @ step
    assert np.all(np.linalg.norm(trial, axis=1) <= radius + 1e-8)
    assert {name.split("/")[0] for name in receipt["constraint_rows"]} == {"first", "second"}
    report = receipt["source_projection"]
    if dimension == 2:
        assert report["affine_pixel_domains"] == 2
        assert report["affine_pixel_cut_rounds"] > 0
        assert "affine_norm_domains_by_unit" not in report
        assert any(event["phase"] == "source_affine_pixel_norm_cut" for event in events)
    else:
        unit = declared_unit or "unspecified"
        assert report["affine_pixel_domains"] == 1
        assert report["affine_pixel_cut_rounds"] == 0
        assert report["max_affine_pixel_excess_px"] == 0.
        assert report["affine_norm_domains_by_unit"] == {unit: 2, "px": 1}
        assert report["max_affine_norm_excess_by_unit"][unit] <= 1e-8
        assert report["affine_norm_cut_units"] == dict.fromkeys(receipt["constraint_rows"], unit)
        assert any(event["phase"] == "source_affine_norm_cut" for event in events)


@pytest.mark.parametrize("bad_fields", [
    dict(offset=np.zeros(3)),
    dict(offset=np.zeros((1, 0))),
    dict(jacobian=csr_matrix((2, 1))),
    dict(radius=np.array([[1.]])),
    dict(labels=[]),
])
def test_affine_norm_ball_rejects_invalid_offsets_and_row_shapes(bad_fields):
    ball = dict(offset=np.zeros((1, 3)), radius=np.ones(1), jacobian=csr_matrix((3, 1)), labels=["residual"])
    ball.update(bad_fields)
    margins = SourceMargins(csr_matrix((0, 1)), np.empty(0), [], [ball])
    with pytest.raises(ValueError, match="Affine norm ball requires offset"):
        source_conditioned_direction(
            ScaledProposal([1.], [1.]), dict(residual=np.array([2.])), (-2., 2.), margins, np.array([0]),
            control_step_limit=[1.],
        )


def test_contour_margin_rejects_raw_geometry_that_differs_from_guarded_baseline():
    radius = 2.
    uv = np.array([[1., 0.], [3., 4.]])
    chain = np.broadcast_to(np.eye(2), (len(uv), 2, 2))
    contour, residual, jacobian, rows = _contour("assigned", uv, chain, radius, 1., raw_proposal=True)
    contour.proposal_pixel_prediction[1] += [.5, 0.]
    value = dict(action_metrics=dict(visible_runner_contours=rows),
                 weighted_blocks=[block("full_source_skate_structure", residual, jacobian)])
    with pytest.raises(AssertionError, match="Raw source proposal geometry must match the guarded baseline"):
        contour_margins(value, contour, {"assigned": (4., radius)})


@pytest.mark.parametrize("with_photo", [False, True], ids=["contour-only", "trailing-photo-rows"])
def test_unassigned_episode_pairs_selected_branch_metrics_with_its_named_weighted_block(with_photo):
    radius = 1.3
    uv = np.array([[4., 3.], [-5., 12.]])
    chain = np.array([[[.3, -.2], [1., 2.]], [[2., .4], [.1, -1.5]]])
    chosen, residual, jacobian, rows = _contour('selected', uv, chain, radius, weight=1.8, raw_proposal=True)
    decoy, other_residual, other_jacobian, _ = _contour('decoy', uv[:, ::-1], -chain, radius, weight=.23)
    if with_photo:
        # The selected episode combines 2N contour rows with N trailing scalar profile rows.
        residual = np.r_[residual, [120., 35.]]
        jacobian = vstack([jacobian, csr_matrix([[40., -25.], [-11., 7.]])], format="csr")
        chosen.last["residual"] = residual
    episode = SimpleNamespace(episode={'episode_id': 'unknown_turn'}, episode_choice=1,
                              episode_factors=[decoy, chosen])
    actions = SimpleNamespace(contour=None, unassigned=None, extra_unassigned_episodes=[('unknown_turn', episode)],
                              existing=SimpleNamespace(turn_factors=[]))
    # Empty body block isolates the selected episode from the unrelated decoy block.
    body = SimpleNamespace(
        frames=np.empty(0, dtype=int), of=np.empty(0, dtype=int), oj=np.empty(0, dtype=int),
        layout=SimpleNamespace(body_landmarks=np.arange(17)),
        cache={'uv': np.empty((0, 17, 2))}, obs={'uv': np.empty((0, 17, 2))},
    )
    terms = [
        block('unassigned_source_skate_sets', other_residual, other_jacobian),
        block('body_pixels', np.empty(0), csr_matrix((0, 2))),
        block('additional_unassigned_source_unknown_turn', residual, jacobian),
    ]
    _, _, weighted = pack_weighted(terms, {terms[-1].name: 2.7}, jacobian=True)
    old = 17.
    margins = all_source_margins(
        SimpleNamespace(action_factors=actions, body=body),
        dict(weighted_blocks=weighted, action_metrics=dict(visible_unassigned_skate_contours=rows)),
        {'selected': (old, radius), 'decoy': (99., radius)}, {}, {'pixel_p95': 0.}, None,
    )
    gradient, slack, labels = margins
    assert labels == ['selected/point_0', 'selected/point_1', 'selected']
    point_gradients = [_p95_gradient(uv[i:i+1], chain[i:i+1], radius) for i in range(len(uv))]
    np.testing.assert_allclose(gradient.toarray(), [*point_gradients, _p95_gradient(uv, chain, radius)],
                               atol=2e-9)
    margin = .8 * (old + .25 * radius - np.percentile(_pixel_distances(uv, radius), 95))
    np.testing.assert_allclose(slack, np.full(3, margin), rtol=0., atol=1e-12)
    ball, = margins.image_balls
    np.testing.assert_array_equal(ball["offset"], uv)
    np.testing.assert_array_equal(ball["jacobian"].toarray(), chain.reshape(uv.size, -1))
    np.testing.assert_allclose(ball["radius"], np.linalg.norm(uv, axis=1) + margin, rtol=0., atol=1e-12)


def _weighted_body_pixels(body, uv_chain, weights):
    uv = body.cache["uv"]
    residual, pixel_jacobian, _, _ = FullBodyProblem.body_factors(body, uv)
    jacobian = pixel_jacobian @ csr_matrix(uv_chain.reshape(uv.size, -1))
    terms = block("body_pixels", residual, jacobian)
    _, _, weighted = pack_weighted([terms], {terms.name: np.repeat(weights, 2)}, jacobian=True)
    return weighted


def test_body_margins_use_original_per_turn_masks_and_nonlinear_p95_limits():
    frames = np.array([41, 72, 96])
    of = np.repeat(np.arange(3), [1, 5, 6])
    oj = np.array([5, 5, 6, 11, 13, 14, 5, 6, 13, 14, 15, 7])
    model_oj = (oj+3) % 17
    distances = np.array([25, 4, 8, 200, 7, 13, 18, 26, 11, 22, 250, 350])
    angles = np.linspace(.3, 1.7, len(oj))
    offsets = distances[:, None]*np.c_[np.cos(angles), np.sin(angles)]
    target = np.array([30., -12.])
    uv = np.zeros((len(frames), 17, 2))
    uv[of, model_oj] = target+offsets
    observed = np.zeros((100, 17, 2))
    observed[frames[of], oj] = target
    sigma = np.ones((100, 17))
    sigma[frames[of], oj] = np.linspace(.6, 4.2, len(oj))
    observation_mask = np.ones(sigma.shape, dtype=bool)
    observation_mask[frames[of[[3, 10]]], oj[[3, 10]]] = False
    body = SimpleNamespace(
        frames=frames, of=of, oj=oj, model_oj=model_oj, cache={'uv': uv},
        layout=SimpleNamespace(body_landmarks=np.arange(17), point_count=17),
        obs={'uv': observed, 'sigma_px': sigma},
    )
    turn_a = SimpleNamespace(episode={'episode_id': 'turn_a'}, rows=[72])
    turn_b = SimpleNamespace(episode={'episode_id': 'turn_b'}, rows=[96])
    actions = SimpleNamespace(contour=None, unassigned=None, extra_unassigned_episodes=[],
                              existing=SimpleNamespace(turn_factors=[turn_a, turn_b]))
    problem = SimpleNamespace(body=body, action_factors=actions)
    uv_chain = np.zeros((*uv.shape, len(oj)))
    uv_chain[of, model_oj, :, np.arange(len(oj))] = np.c_[np.linspace(.4, 2.1, len(oj)), np.full(len(oj), -.8)]
    weighted = _weighted_body_pixels(body, uv_chain, np.linspace(.14, 2.3, len(oj)))
    baseline = {
        'turn_a/torso': dict(observations=2, pixel_p95=9.),
        'turn_a/lower_limbs': dict(observations=2, pixel_p95=15.),
        'turn_b/torso': dict(observations=2, pixel_p95=40.),
        'turn_b/lower_limbs': dict(observations=2, pixel_p95=25.),
    }
    torso_baseline = {'pixel_p95': 28.}
    gradient, slack, labels = all_source_margins(
        problem, dict(weighted_blocks=weighted, action_metrics={}), {}, baseline, torso_baseline, observation_mask,
    )
    assert isinstance(gradient, csr_matrix)
    current = turn_episode_image_metrics(problem, [turn_a, turn_b], observation_mask)
    checks = turn_episode_image_checks(baseline, current)
    masks = {
        'torso': [0, 1, 2, 6, 7],
        'turn_image/turn_a/torso': [1, 2],
        'turn_image/turn_a/lower_limbs': [4, 5],
        'turn_image/turn_b/torso': [6, 7],
        'turn_image/turn_b/lower_limbs': [8, 9],
    }
    expected_labels = []
    for name, rows in masks.items():
        i = len(expected_labels)
        expected_labels.append(name)
        expected_labels.extend(f"{name}/observation_{frames[of[k]]}_{oj[k]}" for k in rows)
        expected_p95 = np.percentile(distances[rows], 95)
        if name == 'torso':
            limit = 30.8
        else:
            check = checks[name]
            assert check['after'] == pytest.approx(expected_p95)
            assert check['original_observations'] == len(rows)
            limit = check['limit_px']
        expected_gradient = _p95_gradient(offsets[rows], uv_chain[of[rows], model_oj[rows]])
        np.testing.assert_allclose(gradient.getrow(i).toarray()[0], expected_gradient, atol=2e-9)
        margin = .8 * (limit - expected_p95)
        assert slack[i] == pytest.approx(margin)
        for point_row, k in enumerate(rows, start=i + 1):
            chain = uv_chain[of[k], model_oj[k]][None]
            expected_radial = _p95_gradient(offsets[k:k+1], chain)
            np.testing.assert_allclose(gradient.getrow(point_row).toarray()[0], expected_radial, atol=2e-9)
            assert slack[point_row] == pytest.approx(margin)
    assert labels == expected_labels


def test_body_observation_halfspaces_protect_p95_when_rank_changes():
    joints = np.array([13, 14, 15])
    uv = np.zeros((1, 17, 2))
    uv[0, joints, 0] = [8., 9., 10.]
    uv_chain = np.zeros((*uv.shape, len(joints)))
    uv_chain[0, joints, 0, np.arange(len(joints))] = 1.
    body = SimpleNamespace(
        frames=np.array([0]), of=np.zeros(len(joints), dtype=int), oj=joints, cache={"uv": uv},
        layout=SimpleNamespace(body_landmarks=np.arange(17), point_count=17),
        obs={"uv": np.zeros_like(uv), "sigma_px": np.ones(uv.shape[:2])},
    )
    turn = SimpleNamespace(episode={"episode_id": "turn"}, rows=[0])
    actions = SimpleNamespace(contour=None, unassigned=None, extra_unassigned_episodes=[],
                              existing=SimpleNamespace(turn_factors=[turn]))
    problem = SimpleNamespace(body=body, action_factors=actions)
    baseline = turn_episode_image_metrics(problem, [turn])
    weighted = _weighted_body_pixels(body, uv_chain, [0.4, 1.5, 2.])
    gradient, slack, labels = all_source_margins(
        problem, dict(weighted_blocks=weighted, action_metrics={}), {}, baseline, {"pixel_p95": 0.}, None,
    )
    name = "turn_image/turn/lower_limbs"
    assert labels == [name, *(f"{name}/observation_0_{joint}" for joint in joints)]
    raw = np.array([6., -1., -1.])
    upper = np.full_like(raw, 8.)
    # The old top two observations both improve, but the smallest becomes the worst.
    assert (gradient[0] @ raw).item() == pytest.approx(-1.)
    old_step, _ = project_box_halfspaces(raw, -upper, upper, gradient[:1], slack[:1])
    np.testing.assert_array_equal(old_step, raw)
    assert np.argmax(uv[0, joints, 0]) == 2
    assert np.argmax(uv[0, joints, 0] + old_step) == 0
    guarded_step, _ = project_box_halfspaces(raw, -upper, upper, gradient, slack)
    np.testing.assert_allclose(guarded_step, [1.6, -1., -1.], rtol=0., atol=1e-10)

    for step, rejected in ((old_step, True), (guarded_step, False)):
        body.cache["uv"] = uv + uv_chain @ step
        trial = turn_episode_image_metrics(problem, [turn])
        check = turn_episode_image_checks(baseline, trial)[name]
        assert check["limit_px"] == pytest.approx(11.9)
        assert check["rejected"] is rejected
        assert check["original_observations"] == len(joints)
        if rejected:
            assert check["after"] == pytest.approx(13.5)



def test_descent_receipt_is_json_safe_and_rejects_unresolved_reduction():
    import json
    from calibration.program_tracking.body_tracking.programme_source_step import resolved_descent
    for trial, expected in ((np.float64(9.), True), (None, False), (np.nan, False), (np.nextafter(10.,0.), False)):
        keep=resolved_descent(trial,np.float64(10.))
        assert type(keep) is bool and keep is expected
        assert json.loads(json.dumps({"accepted":keep}))["accepted"] is expected
