"""Ice-paint validity follows the actual same-state ice, not an offset cap."""
from pathlib import Path
from types import MethodType

import numpy as np
import pytest

from calibration.program_tracking.first_shot_physical_state import FirstShotPhysicalState, GeometryDomainError


def synthetic_material_state():
    state = FirstShotPhysicalState.__new__(FirstShotPhysicalState)
    state.count = 118
    state.scene_unit_working_m = .3
    state.registry = {"targets": {"ice.current_paint": {"kind": "ice_curve", "curve_id": "current_paint"}}}
    def scene(self, q):
        half = q[0]
        return {"ice_boundary": {"points_xyz": [[-half, -5., 0.], [half, -5., 0.],
                                                [half, 5., 0.], [-half, 5., 0.]]}}
    def paths(self, q, descriptors):
        return (np.array([[0., 0., 0.], [q[95], 0., 0.]]),)
    state.scene = MethodType(scene, state)
    state.world_paths = MethodType(paths, state)
    return state


def test_actual_ice_boundary_rejects_outside_endpoint_without_a_parameter_cap():
    state = synthetic_material_state()
    q = np.zeros(118); q[0] = 10.; q[95] = 20.
    status = state.ice_material_containment_status(q)
    assert not status["valid"]
    violation = status["violations"][0]
    assert violation["target_id"] == "ice.current_paint"
    assert violation["worst_vertex_is_endpoint"]
    assert violation["minimum_signed_clearance_scene_units"] == pytest.approx(-10.)
    assert violation["minimum_signed_clearance_working_m"] == pytest.approx(-3.)
    # The same extension is admissible in a genuinely larger current ice.
    # An arbitrary hard maximum on the extension would incorrectly reject it.
    q[0] = 25.
    assert state.ice_material_containment_status(q)["valid"]


def test_actual_ice_boundary_accepts_material_on_the_current_footprint():
    state = synthetic_material_state()
    q = np.zeros(118); q[0] = q[95] = 10.
    status = state.ice_material_containment_status(q)
    assert status["valid"] and not status["violations"]
    assert status["arbitrary_parameter_cap_or_new_measurement_added"] is False


def test_geometry_only_guard_does_not_swallow_a_broken_world_interface():
    state = synthetic_material_state()
    state.provider = type('Provider', (), {'validate_parameters': lambda self, q: q})()
    def broken_paths(self, q, descriptors):
        raise ValueError('A real target lookup interface is missing')
    state.world_paths = MethodType(broken_paths, state)
    q = np.zeros(118); q[0] = 10.
    with pytest.raises(ValueError, match='real target lookup') as failure:
        state.guard_geometry(q)
    assert not isinstance(failure.value, GeometryDomainError)


def test_geometry_only_guard_exposes_only_a_physical_domain_failure():
    state = synthetic_material_state()
    state.provider = type('Provider', (), {'validate_parameters': lambda self, q: q})()
    q = np.zeros(118); q[0] = 10.; q[95] = 20.
    with pytest.raises(GeometryDomainError, match='Current ice paint leaves'):
        state.guard_geometry(q)
    q[0] = 25.
    result = state.guard_geometry(q)
    assert result['passed'] and result['installation_parameters_checked'] is False


@pytest.fixture(scope="module")
def actual_initial_and_failed_material_state():
    folder = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/shot1_physical_fit_20261014/solver_valid/raw_alignment')
    initial = folder/'initial_raw_errors.npz'
    failed = folder/'raw_000027_d6445eb8670f.npz'
    if not initial.is_file() or not failed.is_file():
        pytest.skip("The preserved exact 8a/failed-fit diagnostic states are unavailable")
    state = FirstShotPhysicalState()
    with np.load(initial, allow_pickle=False) as saved:
        q0 = saved['q'].copy()
    with np.load(failed, allow_pickle=False) as saved:
        q = saved['q'].copy()
    np.testing.assert_array_equal(q0, state.initial_q)
    return state, q0, q


def test_exact_8a_initial_material_geometry_remains_admissible(actual_initial_and_failed_material_state):
    state, q0, q = actual_initial_and_failed_material_state
    result = state.guard(q0)
    assert result['passed'] and result['ice_material_containment']['valid']


def test_preserved_d644_material_runaway_is_rejected_as_outside_ice(actual_initial_and_failed_material_state):
    state, q0, q = actual_initial_and_failed_material_state
    before = q.copy()
    status = state.ice_material_containment_status(q)
    assert not status['valid']
    lane = next(r for r in status['violations'] if r['curve_id'] == 'house_connection_F2_R2:segment0')
    assert lane['minimum_signed_clearance_working_m'] < -600.
    assert lane['worst_vertex_is_endpoint']
    with pytest.raises(ValueError, match='Current ice paint leaves actual ice footprint'):
        state.guard(q)
    np.testing.assert_array_equal(q, before)
