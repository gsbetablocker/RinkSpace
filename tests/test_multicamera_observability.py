"""Independent scaling, sparse-column and station projection checks."""
import importlib.util
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parents[1]/'events/ankara_2026/programmes/okada_mei_sp'
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('multicamera_information', HERE/'audit_multicamera_observability.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_active_camera_columns_do_not_assume_full_31_chart():
    active = tuple(range(15))+tuple(range(20, 31))
    cameras = [dict(id=name) for name in ('main_long_side', 'secondary_long_side', 'end')]
    records = audit.camera_records(2151, cameras, [active]*3, audit.chart_steps())
    by_name = {r['name']: r for r in records}
    assert len(records) == 78
    assert records[-1]['column'] == 2228
    assert by_name['secondary_long_side/pivot_z']['column'] == 2185
    assert by_name['secondary_long_side/lever_y']['column'] == 2187
    assert by_name['secondary_long_side/pivot_z']['chart_step'] == .001
    assert by_name['secondary_long_side/lever_y']['chart_step'] == .00025
    assert 'secondary_long_side/cx' not in by_name
    assert sum(r['group'] == 'mechanical' for r in records) == 45


def test_existing_profile_is_invariant_under_solver_chart_rescaling():
    rng = np.random.default_rng(528)
    j = rng.normal(size=(30, 7))
    j[:, 1] = .7*j[:, 0]+.02*j[:, 1]
    h = j.T@j
    target = np.array([0, 1])
    first = audit.profile_camera(h, target, ['pivot_z', 'lever_y'])
    change = np.array([.001, .00025, 3., 10., .003, .1, .05])
    second = audit.profile_camera(h*change[:, None]*change[None, :], target, ['pivot_z', 'lever_y'])
    np.testing.assert_allclose(first['profiled'], second['profiled'], atol=4e-14)
    np.testing.assert_allclose(first['eigenvalues'], second['eigenvalues'], atol=4e-14)
    delta = np.array([-.1636, -.17])
    a = first['scale'][target]*delta
    b = second['scale'][target]*(delta/change[target])
    np.testing.assert_allclose(a, b, atol=1e-14)
    np.testing.assert_allclose(a@first['profiled']@a, b@second['profiled']@b, atol=1e-14)


def test_zero_column_pruning_and_free_station_projection_match_full_oracle():
    raw = np.zeros((3, 2, 6))
    raw[:, :, [0, 3]] = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[2., 3.], [4., 5.]]])
    raw[:, :, -1] = [[2., 1.], [1., -1.], [0., 0.]]
    columns, small, zero = audit.project_station_tangents(raw, [1, 0, 1])
    np.testing.assert_array_equal(columns, [0, 3])
    oracle = raw[:, :, :5].copy()
    tangent = raw[0, :, -1]
    oracle[0] = (np.eye(2)-np.outer(tangent, tangent)/(tangent@tangent))@oracle[0]
    np.testing.assert_allclose(small, oracle[:, :, columns], atol=1e-14)
    assert zero == 1
    assembled = np.zeros((5, 5))
    assembled[np.ix_(columns, columns)] = small.reshape(-1, 2).T@small.reshape(-1, 2)
    np.testing.assert_allclose(assembled, oracle.reshape(-1, 5).T@oracle.reshape(-1, 5), atol=1e-13)
