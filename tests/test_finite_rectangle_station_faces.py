"""Finite-station elimination against an independent convex 1D oracle."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

from calibration.solver_runtime.finite_rectangle_station_faces import finite_rectangle_station_halfspaces


def oracle(foot, jacobian, vector, station, domain, uv, normal, nb, tb, step):
    tangent = np.array([-normal[1], normal[0]])
    axes = np.array([normal, tangent])
    origin = axes @ (foot + jacobian @ step - uv)
    slopes = axes @ vector
    lines = [(0., 0.)]
    for axis, bounds in enumerate((nb, tb)):
        lines.extend([(-slopes[axis], bounds[0] - origin[axis]),
                      (slopes[axis], origin[axis] - bounds[1])])
    low, high = np.asarray(domain) - station
    candidates = [low, high]
    for i, (mi, ci) in enumerate(lines):
        for mj, cj in lines[i + 1:]:
            if mi != mj:
                value = (cj - ci) / (mi - mj)
                if low <= value <= high:
                    candidates.append(value)
    return min(max(m * value + c for m, c in lines) for value in candidates)


@pytest.mark.parametrize("station", [0., .3, 1.])
@pytest.mark.parametrize("vector", [(1., 2.), (0., -2.), (-3., 0.)])
def test_elimination_matches_original_rectangle_oracle_and_releases_endpoint(station, vector):
    foot = np.array([[3., 5.]])
    jacobian = np.array([[[.3, -.2], [-.4, .7]]])
    vector = np.array([vector])
    source = np.array([[.5, 1.]])
    normals = np.array([[1., 0.]])
    normal_bounds = np.array([[-1., 2.]])
    tangent_bounds = np.array([[-3., 4.]])
    domain = np.array([[0., 1.]])
    inputs = [foot, jacobian, vector, source, normals, normal_bounds, tangent_bounds, domain]
    unchanged = [value.copy() for value in inputs]
    result = finite_rectangle_station_halfspaces(
        foot, jacobian, vector, [station], domain, source, normals,
        normal_bounds, tangent_bounds, [0.])
    for step in (np.array([-1., .7]), np.zeros(2), np.array([.2, -1.]), np.array([1., 1.])):
        expected = oracle(foot[0], jacobian[0], vector[0], station, domain[0], source[0],
                          normals[0], normal_bounds[0], tangent_bounds[0], step)
        actual = max(result['lower'] - result['matrix'] @ step)
        assert actual == pytest.approx(expected, abs=1e-12)
    assert result['contract']['endpoint_release_allowed']
    assert all(np.array_equal(before, after) for before, after in zip(unchanged, inputs))
    assert len(result['face_provenance']) == len(result['lower'])


def test_actual_limit_is_not_widened_and_zero_weak_faces_are_kept():
    result = finite_rectangle_station_halfspaces(
        [[0., 2.]], [[[0., 1e-15], [0., 0.]]], [[1., 0.]], [0.], [[0., 1.]],
        [[0., 0.]], [[1., 0.]], [[-1., 1.]], [[-1., 1.]], [.1])
    assert np.any(result['matrix'] != 0.)
    assert np.any(~np.any(result['matrix'] != 0., axis=1))
    assert max(result['lower']) == pytest.approx(.9, abs=1e-15)
    assert result['contract']['zero_and_weak_faces_not_threshold_filtered']
    assert not np.all(result['matrix'] @ np.zeros(2) >= result['lower'])


@pytest.mark.parametrize('station,cone,inward,outward',[
    (0.,[0.,np.inf],-1.,1.),(1.,[-np.inf,0.],1.,-1.)])
def test_true_endpoint_tangent_cone_releases_only_inward(station,cone,inward,outward):
    result=finite_rectangle_station_halfspaces([[0.,0.]],[[[1.],[0.]]],[[1.,0.]],
        [station],[[0.,1.]],[[0.,0.]],[[1.,0.]],[[-.1,.1]],[[-1.,1.]],[0.],
        station_tangent_cone=[cone])
    value=lambda step:max(result['lower']-result['matrix']@np.array([step]))
    assert value(inward)<=0.
    assert value(outward)==pytest.approx(.9)
    assert value(0.)==0.
    assert result['contract']['true_clamped_physical_endpoint_allows_only_inward_station_release']


def test_interior_tangent_cone_does_not_use_finite_distant_caps_and_supports_value_only():
    result=finite_rectangle_station_halfspaces([[0.,0.]],np.zeros((1,2,0)),[[1.,0.]],
        [.5],[[0.,1.]],[[0.,0.]],[[1.,0.]],[[-.1,.1]],[[-1.,1.]],[0.],
        station_tangent_cone=[[-np.inf,np.inf]])
    assert result['matrix'].shape==(len(result['lower']),0)
    assert all('physical_'not in row['kind']for row in result['face_provenance'])
    with pytest.raises(ValueError,match='true global physical endpoint'):
        finite_rectangle_station_halfspaces([[0.,0.]],np.zeros((1,2,0)),[[1.,0.]],
            [.5],[[0.,1.]],[[0.,0.]],[[1.,0.]],[[-.1,.1]],[[-1.,1.]],[0.],
            station_tangent_cone=[[0.,np.inf]])


def test_drawing_samples_do_not_supply_station_domain_or_hidden_tolerance():
    with pytest.raises(ValueError, match="Nominal foot station"):
        finite_rectangle_station_halfspaces(
            [[0., 0.]], np.zeros((1, 2, 1)), [[1., 0.]], [-1e-15], [[0., 1.]],
            [[0., 0.]], [[1., 0.]], [[-1., 1.]], [[-1., 1.]], [0.])


@pytest.mark.parametrize("station,foot,direction", [(0., -.5, -1.), (1., .5, 1.)])
def test_clamped_physical_endpoint_can_release_without_zero_support_growth(station, foot, direction):
    result = finite_rectangle_station_halfspaces(
        [[foot, 0.]], [[[direction], [0.]]], [[1., 0.]], [station], [[0., 1.]],
        [[0., 0.]], [[1., 0.]], [[-.5, .5]], [[-.5, .5]], [0.])
    assert max(result['lower'] - result['matrix'][:, 0]) == pytest.approx(0., abs=1e-15)
    assert abs(foot + direction) - .5 == 1.  # A permanently fixed foot would veto.


ACTUAL_PROOF = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/native35_four_states_finite_physical_station_local_epigraph_ACTUAL_PROOF_20261011.json')


@pytest.mark.skipif(not ACTUAL_PROOF.is_file(), reason="Actual event diagnostic arrays are not installed on this host")
def test_four_saved_complete_directions_detect_kink_and_allow_fresh_descent():
    proof = json.loads(ACTUAL_PROOF.read_text(encoding='utf8'))
    parent_path = Path(proof['actual_parent']['path'])
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == proof['actual_parent']['sha256']
    parent = json.loads(parent_path.read_text(encoding='utf8'))
    arrays_ref = parent['array_receipt']
    assert hashlib.sha256(Path(arrays_ref['path']).read_bytes()).hexdigest() == arrays_ref['sha256']
    with np.load(arrays_ref['path'], allow_pickle=False) as arrays:
        uv = arrays['source_uv'][:1].copy()
        normal = arrays['source_normals'][:1].copy()
        nb = arrays['source_bounds'][:1].copy()
        tb = arrays['source_tangent_bounds'][:1].copy()
    for row in proof['rows']:
        foot = np.array(row['nominal_position'])[None]
        jacobian = np.array(row['fixed_nominal_station_projection_directional_derivative'])[None, :, None]
        vector = np.array(row['local_global_station_screen_tangent'])[None]
        station = row['nominal_physical_station']
        result = finite_rectangle_station_halfspaces(
            foot, jacobian, vector, [station], [[0., 1.]], uv, normal, nb, tb, [0.])
        for trial in row['trials']:
            alpha = trial['alpha']
            prediction = max(result['lower'] - result['matrix'][:, 0] * alpha)
            assert prediction == pytest.approx(trial['finite_station_local_model'], abs=1e-12)
        atzero = next(x for x in row['trials'] if x['alpha'] == 0.)
        if row['q'].startswith('4a2c4'):
            descent = next(x for x in row['trials'] if x['alpha'] == 1 / 32)
            assert descent['finite_station_local_model'] < atzero['finite_station_local_model']
            assert descent['model_optimal_physical_station'] > 0.
            assert abs(descent['model_minus_actual']) < 7e-9
        else:
            bad = next(x for x in row['trials'] if x['alpha'] == 1 / 2048)
            assert bad['finite_station_local_model'] > atzero['finite_station_local_model']
            assert bad['unbounded_interior_local_model'] < atzero['finite_station_local_model']
            assert abs(bad['model_minus_actual']) < 8e-10
        assert all(item['physical_station_bounds'] == [0., 1.] for item in result['face_provenance'])
