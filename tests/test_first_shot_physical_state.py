"""Actual accepted native-50 chart against the complete frozen parent state."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibration.program_tracking.first_shot_physical_state import FirstShotPhysicalState
from calibration.program_tracking.complete_official_fit_dataset import camera_fit_state


@pytest.fixture(scope="module")
def state():
    return FirstShotPhysicalState.from_binding()


def test_complete_parent_roundtrip_and_permutation(state):
    assert state.count == 5119
    assert state.geometry_count == 118 and state.shared_count == 127
    assert state.frame_count == 1248
    np.testing.assert_array_equal(state.parent_projection(state.initial_q), state.parent_q)
    evaluation = state.layout.evaluation_q(state.parent_q)
    np.testing.assert_array_equal(evaluation[state.evaluation_shared_slots], state.initial_q[:127])
    changed = state.initial_q.copy()
    changed[:127] += np.linspace(0., 1e-4, 127)
    projected = state.parent_projection(changed)
    np.testing.assert_array_equal(projected[state.parent_shared_slots], changed[:127])
    keep = np.ones(34340, bool); keep[state.parent_shared_slots] = False
    np.testing.assert_array_equal(projected[keep], state.parent_q[keep])
    assert not state.parent_q.flags.writeable


def test_exact_native_source_clock_without_legacy_aliases(state):
    assert state.source_clock == {"time_base": "1/12800", "fps": "50/1"}
    assert state.pts.tolist() == list(range(256256, 575489, 256))
    assert state.proof["historical_pixel_factors_connected_to_active_FIT"] is False
    assert state.proof["historical_full_programme_graph_loaded"] is False
    assert state.proof["shared_geometry_constructor_legacy_initialization_helpers_reused"] is True
    assert state.proof["temporary_legacy_initialization_factor"]["included_in_current_FIT"] is False
    assert state.proof["historical_profiles_loaded"] is False
    assert not hasattr(state, "aliases")
    assert state.proof["seed_mapping_is_observation_or_exposure_proof"] is False


@pytest.mark.parametrize("frame", [1, 624, 1248])
def test_actual_physical_analytic_camera_chain(state, frame):
    q = state.initial_q.copy()
    camera, columns, chain = state.camera15_mapping(q, frame)
    np.testing.assert_array_equal(camera, state.camera25(q, frame))
    assert chain.shape == (15, 13)
    assert columns.tolist() == list(range(118, 127)) + state.view_columns(frame).tolist()
    fd = np.empty_like(chain)
    for j, column in enumerate(columns):
        step = 1e-7 if column >= state.view_start and j < 12 else 1e-5
        plus, minus = q.copy(), q.copy(); plus[column] += step; minus[column] -= step
        fd[:, j] = (camera_fit_state(state.camera25(plus, frame)) - camera_fit_state(state.camera25(minus, frame))) / (2 * step)
    np.testing.assert_allclose(chain, fd, rtol=2e-7, atol=2e-8)


def test_no_per_frame_C_aspect_PP_or_Brown(state):
    q = state.initial_q.copy()
    before = state.camera25(q, 1)
    q[state.view_columns(1)[0]] += .005
    after = state.camera25(q, 1)
    assert not np.array_equal(before[:3], after[:3])  # C follows rigid rotation.
    np.testing.assert_array_equal(before[8:], after[8:])
    assert before[6] - before[7] == after[6] - after[7]
    np.testing.assert_array_equal(state.camera25(q, 2), state.camera25(state.initial_q, 2))
    q = state.initial_q.copy(); q[124] += 1.25
    for frame in (1, 1248):
        assert abs(state.camera25(q, frame)[8] - state.camera25(state.initial_q, frame)[8] - 1.25) < 1e-10


def test_initial_physical_domains_and_approved_topology(state):
    assert state.guard(state.initial_q)["passed"]
    scene = state.scene(state.initial_q)
    assert "curling_centerline_row3" not in {f["id"] for f in scene["ice_features"]}
    assert scene["superseded_ice_feature_aliases"]["curling_centerline_row3"] == "house_connection_F3_R3"
    assert scene["physical_topology_corrections"] == state.approved_scene["physical_topology_corrections"]
    invalid = state.initial_q.copy()
    # Put the actual tripod pivot at the interior board centroid.
    initial = state.calibration(invalid)
    invalid[118:120] -= np.asarray(initial.pivot_xyz[:2]) * state.scene_unit_working_m
    with pytest.raises(ValueError, match="inside"):
        state.guard(invalid)


def test_existing_v4_bounds_are_exposed_without_silent_parent_reinitialization(state):
    assert state.saved_mount_bounds_active is False
    assert state.active_shared_parameter_bounds == {}
    assert state.shared_parameter_bounds == {}
    assert set(state.historical_mount_parameter_bounds) == {121, 122, 123}
    status = state.mount_bounds_status(state.initial_q)
    assert status["valid"] is False
    assert status["lower_violation_working_m"][1] > 0
    assert status["upper_violation_working_m"][2] > 0
    with pytest.raises(ValueError, match="v4"):
        state.guard(state.initial_q, check_mount_bounds=True)
    candidate = state.initial_q.copy()
    for column, (low, high) in state.historical_mount_parameter_bounds.items():
        candidate[column] = (low + high) / 2
    assert state.mount_bounds_status(candidate)["valid"]
    # This is an explicit test candidate, never the stored initialization.
    np.testing.assert_array_equal(state.parent_projection(state.initial_q), state.parent_q)


def test_dynamic_prints_and_circles_are_existing_parent_parameters(state):
    initial = state.scene(state.initial_q)
    q = state.initial_q.copy(); q[100] += 1e-3; q[109] += 1e-3
    scene = state.scene(q)
    assert scene["shared_board_patterns"]["shared_parameters_m"][0] != initial["shared_board_patterns"]["shared_parameters_m"][0]
    assert scene["additional_board_circle_patterns"]["current_circle6"][0] != initial["additional_board_circle_patterns"]["current_circle6"][0]
    assert q[100] == state.parent_projection(q)[34180]
    assert q[109] == state.parent_projection(q)[34296]


def test_initial_registered_targets_preserve_approved_world(state):
    proof = state.validate_initial_targets(list(state.registry["targets"].values()))
    assert proof["passed"] and proof["target_count"] == len(state.registry["targets"])
    assert proof["maximum_absolute_difference"] <= 1e-10


def test_reject_missing_exposure_and_incomplete_state(state):
    with pytest.raises(ValueError):
        state.camera25(state.initial_q[:-1], 1)
    with pytest.raises(ValueError):
        state.camera25(state.initial_q, 1249)
    with pytest.raises(ValueError):
        state.camera25(state.initial_q, 0)
    with pytest.raises(ValueError):
        state.camera25(state.initial_q, 1.5)
    with pytest.raises(ValueError):
        state.scene(state.initial_q[:100])
