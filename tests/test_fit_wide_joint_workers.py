"""Spawned numerical-Jacobian workers must evaluate the saved real problem identically."""
import copy
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import fit_wide_joint as wide_fit


REAL_FIT = ROOT / 'outputs/calibration/ankara_2026_wide_joint_20260923/equal_radius_lane_hard_v2'
MIXED_CAMERA_FIT = ROOT / 'outputs/calibration/ankara_2026_wide_joint_20260923/equal_radius_lane_crossview_v1'


def test_older_scipy_uses_clear_parallel_compatibility_path(monkeypatch):
    monkeypatch.setattr(wide_fit, '_default_worker_count', lambda: 4)
    assert wide_fit._resolve_worker_count(None, False) == (1, 'auto_serial_fallback_older_scipy')
    assert wide_fit._resolve_worker_count(4, True) == (4, 'explicit')
    with pytest.raises(ValueError, match='requires SciPy'):
        wide_fit._resolve_worker_count(4, False)


@pytest.mark.skipif(not (REAL_FIT / 'problem.json').exists(), reason='local real-video calibration absent')
def test_spawned_worker_matches_parent_on_real_wide_problem_and_visibility_penalty(tmp_path):
    problem_data = json.loads((REAL_FIT / 'problem.json').read_text(encoding='utf8'))
    previous = json.loads((REAL_FIT / 'fit.json').read_text(encoding='utf8'))
    for outside in (None, dict(minimum_margin_scene_units=2., hinge_weight=100.)):
        data = copy.deepcopy(problem_data)
        if outside is not None:
            data['outside_right_margin_experiment'] = outside
        path = tmp_path / ('problem_outside.json' if outside else 'problem.json')
        path.write_text(json.dumps(data), encoding='utf8')
        parent = wide_fit._build_joint_problem(data)
        wide_fit._ACTIVE_PROBLEM = parent
        values = dict(zip(previous['parameter_names'], previous['parameter_values']))
        x = np.asarray([values.get(name, initial) for name, initial in
                        zip(parent.parameter_names, parent.x0)], float)
        serial = wide_fit._worker_objective(x)
        with multiprocessing.get_context('spawn').Pool(
                2, initializer=wide_fit._initialize_worker, initargs=(str(path),)) as pool:
            parallel = pool.map(wide_fit._worker_objective, [x, x.copy()])
        assert all(np.array_equal(result, serial) for result in parallel)
        if outside is not None:
            assert len(serial) == len(parent.residual(x)) + 1


@pytest.mark.skipif(not (MIXED_CAMERA_FIT / 'problem.json').exists(),
                    reason='local same-event multiview calibration absent')
def test_spawned_worker_matches_real_mixed_camera_problem():
    problem_path = MIXED_CAMERA_FIT / 'problem.json'
    data = json.loads(problem_path.read_text(encoding='utf8'))
    previous = json.loads((MIXED_CAMERA_FIT / 'fit.json').read_text(encoding='utf8'))
    parent = wide_fit._build_joint_problem(data)
    wide_fit._ACTIVE_PROBLEM = parent
    values = dict(zip(previous['parameter_names'], previous['parameter_values']))
    x = np.asarray([values.get(name, initial) for name, initial in
                    zip(parent.parameter_names, parent.x0)], float)
    serial = wide_fit._worker_objective(x)
    with multiprocessing.get_context('spawn').Pool(
            2, initializer=wide_fit._initialize_worker, initargs=(str(problem_path),)) as pool:
        parallel = pool.map(wide_fit._worker_objective, [x])
    np.testing.assert_array_equal(parallel[0], serial)
