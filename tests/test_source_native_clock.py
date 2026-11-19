"""Native-clock refinement preserves the saved path before image fitting."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.absolute_motion_curve import (
    AbsoluteMotionCurve, clamped_cubic_knots, greville_times, _exp, _log,
)
from calibration.program_tracking.body_tracking.absolute_motion_state import evaluate_curve


@pytest.fixture
def refine(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / (
        'events/ankara_2026/programmes/okada_mei_sp/tools/identity_refit')
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('native_clock_refinement', directory / 'source_continuous_refine.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.refine_native_rotation_clock


def test_commuting_rotation_and_cubic_translation_are_preserved(refine):
    knots = clamped_cubic_knots(np.linspace(0, 2, 9))
    clock = greville_times(knots)
    vector = np.zeros((len(clock), 22, 3))
    vector[:, :, 2] = .8 * clock[:, None]
    before = AbsoluteMotionCurve(knots, np.c_[clock, clock**2, clock**3], _exp(vector))
    native = np.arange(101) / 50
    after, record = refine(before, native)
    sample = np.linspace(0, 2, 401)
    a, b = evaluate_curve(before, sample), evaluate_curve(after, sample)
    np.testing.assert_allclose(a['P'], b['P'], atol=2e-12, rtol=0)
    np.testing.assert_allclose(a['rotations'], b['rotations'], atol=1e-10, rtol=0)
    assert record['new_controls'] > record['old_controls']
    assert all(np.min(abs(after.knots - t)) < 1e-9 for t in native)


def test_noncommuting_rotation_path_is_retained(refine):
    knots = clamped_cubic_knots(np.linspace(0, 2, 9))
    clock = greville_times(knots)
    vector = np.zeros((len(clock), 22, 3))
    vector[:, :, 0] = .4 * np.sin(3 * clock[:, None])
    vector[:, :, 2] = .8 * clock[:, None]
    before = AbsoluteMotionCurve(knots, np.zeros((len(clock), 3)), _exp(vector))
    after, record = refine(before, np.arange(101) / 50)
    sample = np.linspace(0, 2, 401)
    a, b = evaluate_curve(before, sample), evaluate_curve(after, sample)
    error = np.linalg.norm(_log(a['rotations'].swapaxes(-1, -2) @ b['rotations']), axis=-1)
    assert np.rad2deg(error.max()) < 1e-5
    assert record['greville_error_radians'][-1] < 1e-8


def test_lifted_long_rotation_edges_do_not_lose_turns(refine):
    knots = clamped_cubic_knots([0., 2.])
    vector = np.zeros((4, 22, 3))
    vector[:, :, 2] = np.arange(4)[:, None] * 1.1 * np.pi
    edge_vectors = np.zeros((3, 22, 3))
    edge_vectors[:, :, 2] = 1.1 * np.pi
    edges = Rotation.from_rotvec(edge_vectors.reshape(-1, 3)).as_quat().reshape(3, 22, 4)
    before = AbsoluteMotionCurve(knots, np.zeros((4, 3)), _exp(vector), edges)
    after, _ = refine(before, np.arange(101) / 50)
    sample = np.linspace(0, 2, 401)
    a, b = evaluate_curve(before, sample), evaluate_curve(after, sample)
    np.testing.assert_allclose(a['rotations'], b['rotations'], atol=1e-9, rtol=0)
    rotations = b['rotations'][:, 0]
    travelled = np.linalg.norm(_log(rotations[:-1].swapaxes(-1, -2) @ rotations[1:]), axis=-1).sum()
    assert travelled == pytest.approx(3.3 * np.pi, abs=1e-9)
