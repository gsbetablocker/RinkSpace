"""A changed spline grid must not silently reuse stale video exposures."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.program_tracking.body_tracking.absolute_motion_curve import clamped_cubic_knots
from calibration.program_tracking.body_tracking.skate_checkpoint import IDENTITY_FIELDS


@pytest.fixture
def changed_exposures(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / (
        'events/ankara_2026/programmes/okada_mei_sp/tools/identity_refit')
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('source_video_clock', directory / 'source_motion_update_video.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.changed_exposures


def packet(times):
    knots = clamped_cubic_knots(times)
    n = len(knots) - 4
    result = {key: np.array('{}' if key.endswith('_json') else 'fixed') for key in IDENTITY_FIELDS}
    result.update(R_FB_initial=np.eye(3), t_FB_initial_solver=np.zeros(3),
                  x=np.zeros(17), relative_skate_knots=np.array([0., 1.]),
                  relative_skate_rotation_controls=np.eye(3)[None],
                  state_schema_version=np.array('rinkspace.absolute_motion_v1'),
                  absolute_knots=knots, absolute_position_controls=np.zeros((n, 3)),
                  absolute_rotation_controls=np.broadcast_to(np.eye(3), (n, 22, 3, 3)).copy())
    return result


def test_refined_grid_can_reuse_only_exactly_identical_readout_states(changed_exposures):
    old, new = packet([0., 1.]), packet([0., .5, 1.])
    reference = SimpleNamespace(times=np.array([.2, .5, .8]))
    camera = SimpleNamespace(readout_times=lambda frames: np.full(len(frames), .1))
    np.testing.assert_array_equal(changed_exposures(old, new, reference, camera), [False]*3)
    new['absolute_rotation_controls'][2, 7] = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    np.testing.assert_array_equal(changed_exposures(old, new, reference, camera), [True]*3)


def test_refined_grid_checks_readout_beyond_the_native_centre(changed_exposures):
    old, new = packet([0., .5, 1.]), packet([0., 1.])
    # The cubic Bernstein controls represent (t - .5)**2. The native
    # centre is unchanged; the nonzero readout trajectory still needs rendering.
    new['absolute_position_controls'][:, 0] = [.25, -1/12, -1/12, .25]
    reference = SimpleNamespace(times=np.array([.5]))
    camera = SimpleNamespace(readout_times=lambda frames: np.full(len(frames), .2))
    np.testing.assert_array_equal(changed_exposures(old, new, reference, camera), [True])


def test_identical_control_grid_keeps_existing_exact_support_reuse(changed_exposures):
    old, new = packet([0., .5, 1.]), packet([0., .5, 1.])
    reference = SimpleNamespace(times=np.array([.2, .5, .8]))
    camera = SimpleNamespace(readout_times=lambda frames: np.zeros(len(frames)))
    np.testing.assert_array_equal(changed_exposures(old, new, reference, camera), [False]*3)


@pytest.mark.parametrize('refined', [False, True])
@pytest.mark.parametrize('changed_field', ['rig', 'controls'])
def test_changed_camera_invalidates_its_shot_even_when_motion_is_identical(changed_exposures, refined, changed_field):
    old = packet([0., 1.])
    new = packet([0., .5, 1.] if refined else [0., 1.])
    old['camera_binding_json'] = np.array(json.dumps({'state_sha256': 'old'}))
    new['camera_binding_json'] = np.array(json.dumps({'state_sha256': 'new'}))
    def camera(identity):
        shots = {}
        for shot in ('s01', 's02'):
            shots[shot] = SimpleNamespace(core=SimpleNamespace(library_path='same.dll'), p=dict(
                rig=np.zeros(43), controls=np.zeros((4, 3)), motion_times=np.array([0., .02]),
                first_control_time=np.array(0.), control_dt=np.array(.02)))
        return SimpleNamespace(cameras=shots, shot_ids=np.array(['s01','s02','s02']),
            info={'state_sha256': identity}, readout_times=lambda frames: np.zeros(len(frames)))
    previous, current = camera('old'), camera('new')
    current.cameras['s02'].p[changed_field].flat[0] = .001
    reference = SimpleNamespace(times=np.array([.2, .5, .8]))
    np.testing.assert_array_equal(changed_exposures(old,new,reference,current,previous),[False,True,True])
    # A declared camera transition does not authorize another mount/model.
    new['t_FB_initial_solver'] = np.ones(3)
    with pytest.raises(ValueError):
        changed_exposures(old,new,reference,current,previous)
