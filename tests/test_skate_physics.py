"""Reference equivalence and real nonzero-pose derivative checks; no fit."""
from pathlib import Path
import io
import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from test_skate_binding import synthetic_binding
from calibration.program_tracking.body_tracking.skate_contact import (
    SupportBand, feature_minimum, frame_residuals, template_world_features,
    trapezoid_time_weights,
)
from calibration.program_tracking.body_tracking.contact_feasibility import contact_inequalities
from calibration.program_tracking.body_tracking.skate_physics import (
    ALLOWANCE, NORMALIZATION, SUPPORT_BAND, batch_features, constraints,
    fk_state, body_constraint_jacobian,
)
from calibration.program_tracking.body_tracking.fullbody_state import WindowState


def metric_features(binding, rotation, origin):
    result = template_world_features(binding.packet()["template"], rotation, origin)
    for feature in result:
        if feature["kind"] == "arc":
            feature["center"] *= binding.ratio
            feature["radius_m"] *= binding.ratio
        else:
            feature["points"] *= binding.ratio
    return result


def test_batch_features_matches_exact_reference_for_random_proper_poses(synthetic_binding):
    b = synthetic_binding
    rng = np.random.default_rng(31103)
    R = Rotation.random(24, random_state=rng).as_matrix().reshape(12, 2, 3, 3)
    O = rng.normal(0., .2, (12, 2, 3))
    batched = batch_features(b, R, O)
    for i in range(12):
        for foot in range(2):
            for j, feature in enumerate(metric_features(b, R[i, foot], O[i, foot])):
                witness = feature_minimum(feature)
                np.testing.assert_allclose(batched["height"][i, foot, j], witness["height_m"], atol=2e-14)
                world = (O[i, foot]+R[i, foot]@batched["local_point"][i, foot, j])*b.ratio
                np.testing.assert_allclose(world, witness["point"], atol=2e-12)


def test_constraints_match_reference_bounds_and_expanded_time_weights(synthetic_binding):
    b = synthetic_binding
    rng = np.random.default_rng(41103)
    times = np.array([0., .02, .06, .10, .16])
    origin = 35.2
    R = Rotation.random(10, random_state=rng).as_matrix().reshape(5, 2, 3, 3)
    O = rng.normal(0., .15, (5, 2, 3))
    evidence = [
        dict(id="test_left_glide", pts=round((origin+times[1])*12800), feet=[0], mode="glide"),
        dict(id="test_right_pick", pts=round((origin+times[2])*12800), feet=[1], mode="pick"),
        dict(id="test_uncertain_foot", pts=round((origin+times[4])*12800), feet=[0, 1], mode="glide"),
    ]
    actual = constraints(b, dict(R=R, O=O), times, origin, evidence)
    expected, weights = [], []
    time_weights = trapezoid_time_weights(times)
    for i, t in enumerate(times):
        skates = {side: metric_features(b, R[i, foot], O[i, foot])
                  for foot, side in enumerate(("left", "right"))}
        bands = []
        for row in evidence:
            if abs(t+origin-row["pts"]/12800.) < 1e-8:
                ids = b.packet()["template"]["support_feature_ids"][row["mode"]]
                members = tuple((("left", "right")[foot], name)
                                for foot in row["feet"] for name in ids)
                bands.append(SupportBand(members, *SUPPORT_BAND, .005, row["id"]))
        frame = frame_residuals(skates, nonpenetration_sigma_m=.005,
                                allowed_penetration_m=ALLOWANCE, supports=bands)
        adapted = contact_inequalities(frame, normalization_m=NORMALIZATION)
        expected.extend(adapted["margin_m"])
        weights.extend([time_weights[i]]*len(adapted["g"]))
    np.testing.assert_allclose(actual["margin_m"], expected, atol=2e-14)
    np.testing.assert_allclose(actual["g"], np.asarray(expected)/NORMALIZATION, atol=3e-12)
    np.testing.assert_array_equal(actual["weights"], weights)
    assert len(set(actual["labels"])) == len(actual["labels"])


@pytest.fixture(scope="module")
def source_skate_fixture():
    selected = os.environ.get("RINKSPACE_S01_BODY_ROOT")
    if not selected:
        pytest.skip("requires selected local source-bound skate fixture")
    root = Path(selected)
    project = Path(__file__).resolve().parents[1]
    library = os.environ.get("RINKSPACE_BODY_NATIVE_LIBRARY")
    if not library:
        pytest.skip("requires selected native performance library")
    from rinkspace.core.native import NativeCore
    NativeCore(library)
    from calibration.program_tracking.body_tracking.skate_integration import build
    out = root/"skate_kernel_checks_1103"
    out.mkdir(exist_ok=True)
    ref, camera, info, _ = build(project, root, out)
    return ref, camera, root, out


@pytest.fixture
def nonzero_source_state(source_skate_fixture):
    ref, _, _, _ = source_skate_fixture
    saved = ref.binding.x.copy()
    ref.binding.x = np.r_[
        [.004, -.003, .002, .17, -.12, .09, -.002, .005, .003, -.13, .08, .16],
        np.log([1.04, .95, 1.08, .94, 1.06])]
    origin = ref.p["pts"][0]/12800.
    times = np.array([35., 35.2, 35.42, 35.68, 36.02, 36.6, 36.82, 37.])-origin
    raw = ref.at(times)
    window = WindowState(ref, (34.8, 37.2), "full", spacing=.4, taper=.2)
    rng = np.random.default_rng(1103)
    delta = rng.normal(0., .10, (len(times), 69))
    delta[:, :3] *= .15
    evidence = [dict(id="test_support", pts=round(36.02*12800), feet=[0, 1], mode="glide")]
    try:
        yield ref, raw, times, origin, delta, window, evidence
    finally:
        ref.binding.x = saved


def test_zero_budget_preserves_real_source_candidate_without_optimization(source_skate_fixture,tmp_path):
    from calibration.program_tracking.body_tracking.skate_integration import JointContactProblem
    from calibration.program_tracking.body_tracking.skate_run_policy import RunPolicy
    ref,camera,root,_=source_skate_fixture
    original=ref.binding.x.copy()
    try:
        problem=JointContactProblem(ref,camera,root/'boot_integration_1102/observations.npz',
            (35.5,35.7),[],tmp_path,io.StringIO(),'continuous_shot')
        initial=np.r_[original,np.zeros(problem.body.size)]
        returned,report=problem.solve(initial,RunPolicy(wall_seconds=0))
        np.testing.assert_array_equal(returned,initial)
        with np.load(tmp_path/'candidate.npz',allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['x'],initial)
            np.testing.assert_array_equal(saved['active'],np.arange(69))
            assert str(saved['state_kind'])=='continuous_shot'
        assert problem.accepted==0
        assert not report['numerical_convergence']
        assert any(row['code']=='wall_budget' for row in report['run_policy']['reasons'])
    finally:
        ref.binding.x=original


def test_real_fk_matches_full_body_native_frames_and_binding(nonzero_source_state):
    ref, raw, _, _, delta, _, _ = nonzero_source_state
    reduced = fk_state(ref, raw, delta)
    full = ref.world(raw, delta)
    for actual, expected in [
        (reduced["native_world"], full["native"]), (reduced["frames"], full["frames"]),
        (reduced["R"], full["boot_R"]), (reduced["O"], full["boot_O"]),
    ]:
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_real_analytic_body_constraint_jacobian_at_nonzero_rotvec_and_shape(nonzero_source_state):
    ref, raw, times, origin, delta, window, evidence = nonzero_source_state
    B, active = window.basis(times), window.active
    result = constraints(ref.binding, fk_state(ref, raw, delta), times, origin, evidence)
    J = body_constraint_jacobian(ref, fk_state(ref, raw, delta), delta, active, B, result)
    rng = np.random.default_rng(1103)
    leg_columns = [6+(j-1)*3+a for j in (1, 2, 4, 5, 7, 8) for a in range(3)]
    for select in [active<3, (active>=3)&(active<6), np.isin(active, leg_columns),
                   np.ones(len(active), bool)]:
        direction = np.zeros((window.m, len(active)))
        direction[:, select] = rng.normal(size=(window.m, int(select.sum())))
        direction /= np.linalg.norm(direction)
        physical = np.zeros_like(delta)
        physical[:, active] = B@direction
        h = 1e-6
        plus = constraints(ref.binding, fk_state(ref, raw, delta+h*physical), times, origin, evidence)
        minus = constraints(ref.binding, fk_state(ref, raw, delta-h*physical), times, origin, evidence)
        assert plus["labels"] == minus["labels"] == result["labels"]
        numeric = (plus["g"]-minus["g"])/(2*h)
        np.testing.assert_allclose(J@direction.ravel(), numeric, atol=3e-6, rtol=2e-6)


@pytest.mark.parametrize('state_kind',['halo_1103','continuous_shot'])
@pytest.mark.parametrize('collar',[False,True])
def test_real_joint_replay_and_shared_body_jacobian(source_skate_fixture,state_kind,collar):
    from calibration.program_tracking.body_tracking.skate_integration import JointContactProblem
    ref, camera, root, out = source_skate_fixture
    saved = ref.binding.x.copy()
    was_enabled=ref.binding.collar_constraints_enabled
    if collar:ref.binding.enable_collar_constraints()
    problem = JointContactProblem(ref, camera, root/"boot_integration_1102/observations.npz",
                                  (35.5, 35.7), [], out/"joint", io.StringIO(),state_kind)
    rng = np.random.default_rng(11103)
    x = np.r_[ref.binding.x, rng.normal(0., .01, problem.body.size)]
    x[:12] = [.004, -.003, .002, .17, -.12, .09, -.002, .005, .003, -.13, .08, .16]
    x[12:17] = np.log([1.04, .95, 1.08, .94, 1.06])
    try:
        problem.binding.x = x[:17].copy()
        base = problem.body.evaluate(x[17:])
        state = base["world"]
        local = np.einsum("nji,nkj->nki", state["Q"], state["native"]-state["P"][:, None])
        mid = np.arange(problem.body.n)*5+2
        problem._replay_samples = problem.binding.surface.sample_vertices(
            local[mid], state["frames"][mid], problem.binding.sample_ids, double_precision=True)
        replay = problem.shared_replay(x[:17], x[17:], base)
        assert len(problem.body.gear.blade_rows) > 0
        np.testing.assert_array_equal(replay, base["residual"])
        value = problem.evaluate(x, True)
        for block in ("shared", "body", "joint"):
            direction = np.zeros_like(x)
            if block != "body":
                direction[:17] = rng.normal(size=17)
            if block != "shared":
                direction[17:] = rng.normal(size=problem.body.size)
            direction /= np.linalg.norm(direction)
            h = 1e-6
            numeric = (problem.evaluate(x+h*direction)["residual"]-
                       problem.evaluate(x-h*direction)["residual"])/(2*h)
            np.testing.assert_allclose(value["jac"]@direction, numeric, atol=2e-5, rtol=2e-4,err_msg=block)
    finally:
        ref.binding.x = saved
        ref.binding.collar_constraints_enabled=was_enabled
