"""Contracts for real image evidence in conditional landmark identity ranking."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

from rinkspace.io import sha256
from rinkspace.landmark_graph import contact_category_conflicts, edge_residual, resolve_frame
from rinkspace.multisurface_camera import Camera, ground_points, project_camera


def _camera(*, shallow=False, rotation_delta=0., centre_delta=0.):
    angle = np.deg2rad(10 if shallow else 34)
    rotation = np.array([[1., 0., 0.], [0., -np.sin(angle), -np.cos(angle)],
                         [0., np.cos(angle), -np.sin(angle)]])
    rotation = cv2.Rodrigues(np.array([0., rotation_delta, 0.]))[0] @ rotation
    return Camera(rotation, np.array([centre_delta, -12., 2. if shallow else 8.]),
                  700., np.array([480., 270.]))


def _hypothesis(camera, identity, score):
    return {'camera': camera.to_dict(), 'score': score,
            'groups': [{'physical_id': identity, 'observation_indices': [0],
                        'residual_summary_sample_px': {'median_px': score, 'p95_px': score}}]}


def _fixture(*, surface='ice_texture', long_edge=False):
    reference, good, wrong = _camera(), _camera(), _camera(rotation_delta=.15)
    world = np.c_[np.linspace(-4, 4, 12), np.linspace(0, 6, 12), np.zeros(12)]
    previous, valid = project_camera(reference, world)
    current, valid2 = project_camera(good, world)
    assert valid.all() and valid2.all()
    points = current * 2
    frame = {'source_id': 'fixture', 'source_sha256': 'source-sha', 'pts': 13056,
             'time_base': '1/12800', 'source_frame_index': 51, 'shot_id': 'shot0',
             'source_size': [1920, 1080], 'sample_size': [960, 540],
             'landmark_observations': [{'observation_id': 'anonymous_arc', 'colour_class': 'gray',
                'points_source_px': points.tolist(), 'association_status': 'detected_unidentified',
                'selected_model_id': None, 'used_by_solver': False}],
             # The visually wrong identity has the better contour-only score.
             'association': {'candidate_hypotheses': [_hypothesis(wrong, 'F3', 0.),
                                                     _hypothesis(good, 'F2', 1.)]},
             'landmark_camera_candidate': wrong.to_dict(), 'landmark_initialization_accepted': False}
    baseline = {12800: {'pts': 12800, 'time_base': '1/12800', 'source_frame_index': 50,
                        'shot_id': 'shot0', 'camera': reference.to_dict()},
                13056: {'pts': 13056, 'time_base': '1/12800', 'source_frame_index': 51,
                        'shot_id': 'shot0', 'camera': None}}
    if long_edge:
        link = {'npz_prefix': 'persist', 'reference_source_pts': 12800, 'reference_sample_index': 0}
        channel = {'long_edge': link}
        arrays = {'persist_reference_xy': previous*2, 'persist_current_xy': current*2}
    else:
        channel = {'npz_prefix': 'adjacent', 'previous_source_pts': 12800, 'previous_sample_index': 0}
        arrays = {'adjacent_previous_xy': previous*2, 'adjacent_current_xy': current*2}
    scene = {13056: {'pts': 13056, 'time_base': '1/12800', 'source_frame_index': 51,
                     'source_sha256': 'source-sha', 'shot_id': 'shot0', 'channels': {surface: channel}}}
    return frame, baseline, scene, arrays


@pytest.mark.parametrize('long_edge', [False, True])
def test_measured_ice_pairs_can_resolve_repeated_circle_identity(long_edge):
    frame, baseline, scene, arrays = _fixture(long_edge=long_edge)
    original = deepcopy(frame)
    result = resolve_frame(frame, baseline, scene, arrays)
    assert result['landmark_observations'][0]['selected_model_id'] == 'F2'
    assert result['landmark_observations'][0]['used_by_solver'] is False
    assert result['landmark_initialization_accepted'] is False  # only one physical group
    assert result['graph_identity_resolution']['accuracy_validated'] is False
    score = next(s for s in result['graph_identity_resolution']['hypotheses'] if s['hypothesis_index'] == 1)
    assert score['image_edge_groups'][0]['point_count'] == 12
    assert score['image_edge_groups'][0]['median_sample_px'] < 1e-9
    assert frame == original


def test_board_points_above_optical_centre_use_rays_not_ground():
    reference, current = _camera(shallow=True), _camera(shallow=True, rotation_delta=.03)
    # Elevated structure visible above the horizon: intersecting these rays
    # with Z=0 would put them behind the camera and destroy valid evidence.
    xyz = np.c_[np.linspace(-3, 3, 12), np.linspace(10, 18, 12), np.full(12, 3.)]
    previous, valid = project_camera(reference, xyz)
    observed, valid2 = project_camera(current, xyz)
    assert valid.all() and valid2.all()
    assert not ground_points(reference, previous)[1].any()
    board = edge_residual(reference, current, previous, observed, 'board_surface')
    assert board['point_count'] == 12 and board['median_sample_px'] < 1e-9
    assert edge_residual(reference, current, previous, observed, 'ice_texture') is None
    moved = _camera(shallow=True, rotation_delta=.03, centre_delta=.02)
    assert edge_residual(reference, moved, previous, observed, 'board_surface') is None


def test_blue_centre_circle_is_checked_by_actual_motion_and_axes_stay_unselected():
    frame, baseline, scene, arrays = _fixture()
    observation = frame['landmark_observations'][0]
    observation.update(colour_class='blue', selected_model_id='centre_circle')
    # Two geometrically different candidates for the same physical circle.
    # The image edge must select its camera rather than bypassing blue items.
    for hypothesis in frame['association']['candidate_hypotheses']:
        hypothesis['groups'][0]['physical_id'] = 'centre_circle'
    withheld = dict(observation, observation_id='heldout_blue_axis',
                    selected_model_id=None, rejection_reason='physical_axis_held_out')
    frame['landmark_observations'].append(withheld)
    result = resolve_frame(frame, baseline, scene, arrays)
    assert result['landmark_observations'][0]['selected_model_id'] == 'centre_circle'
    assert result['landmark_observations'][0]['identity_basis'][-1] == 'conditional_same_shot_image_edge_ranking'
    assert result['landmark_observations'][1]['selected_model_id'] is None
    assert result['graph_identity_resolution']['selected_hypothesis_index'] == 1


def test_hypothesis_without_a_valid_board_ray_edge_cannot_win_by_missing_zero_cost():
    frame, baseline, scene, arrays = _fixture(surface='board_surface')
    frame['association']['candidate_hypotheses'] = [
        _hypothesis(_camera(centre_delta=4.), 'F3', 0.),
        _hypothesis(_camera(), 'F2', 2.5)]
    result = resolve_frame(frame, baseline, scene, arrays)
    assert result['landmark_observations'][0]['selected_model_id'] == 'F2'
    chosen = result['graph_identity_resolution']['selected_hypothesis_index']
    assert chosen == 1


@pytest.mark.parametrize('remove_pose, cross_cut', [(True, False), (False, True)])
def test_missing_reference_pose_or_cut_cannot_inject_unresolved_best_camera(remove_pose, cross_cut):
    frame, baseline, scene, arrays = _fixture()
    if remove_pose:
        baseline[12800]['camera'] = None
    if cross_cut:
        baseline[12800]['shot_id'] = 'previous_shot'
    result = resolve_frame(frame, baseline, scene, arrays)
    assert result['graph_identity_resolution']['status'] == 'no_anchored_same_shot_image_edge'
    assert result['landmark_camera_candidate'] is None
    assert result['landmark_initialization_accepted'] is False
    assert result['landmark_observations'][0]['selected_model_id'] is None


def _contact_case(boundary, points, flags=None):
    hypothesis = {'groups': [{'physical_id': 'F2', 'observation_indices': [0]}]}
    observations = [{'points_source_px': np.asarray(points, float).tolist()}]
    contact = {'observed_xy': np.asarray(boundary, float).tolist(),
               'inlier_flags': [True]*len(boundary) if flags is None else flags,
               'uncertainty_source_px': 6.}
    scene = {'channels': {'ice_board_contact': {'observation': contact}}}
    return hypothesis, observations, scene


def test_actual_contact_category_can_reject_false_ring_without_a_camera():
    boundary = np.c_[np.arange(0, 401, 40), np.full(11, 500.)]
    points = np.c_[np.linspace(80, 320, 16), np.full(16, 503.)]
    case = _contact_case(boundary, points)
    conflicts = contact_category_conflicts(*case)
    assert len(conflicts) == 1 and conflicts[0]['physical_id'] == 'F2'
    case[0]['groups'][0]['physical_id'] = 'midline'
    assert contact_category_conflicts(*case) == []


def test_contact_outliers_and_a_masked_gap_do_not_create_observed_support():
    x = np.r_[np.arange(0, 121, 40), np.arange(360, 481, 40)]
    boundary = np.c_[x, np.full(len(x), 500.)]
    # The entire candidate is inside an unmeasured 240px occlusion gap, near
    # its left end. A nearest-x<80 rule plus connected polyline is insufficient.
    points = np.c_[np.linspace(145, 185, 12), np.full(12, 500.)]
    assert contact_category_conflicts(*_contact_case(boundary, points)) == []
    # Apparent points at y=700 were measured but explicitly rejected by the
    # contact estimator; they cannot become a boundary after resorting.
    real = np.c_[np.arange(0, 401, 40), np.full(11, 500.)]
    false = np.c_[np.arange(0, 401, 40), np.full(11, 700.)]
    candidate = np.c_[np.linspace(80, 320, 16), np.full(16, 700.)]
    assert contact_category_conflicts(*_contact_case(np.r_[real, false], candidate, [True]*11+[False]*11)) == []


def test_contact_flags_must_bind_every_measured_point():
    boundary = np.c_[np.arange(0, 201, 40), np.full(6, 500.)]
    with pytest.raises(ValueError, match='flags'):
        contact_category_conflicts(*_contact_case(boundary, boundary, [True]))


def test_unanchored_contact_misclassification_does_not_keep_a_bootstrap_pose():
    """A direct contour decision still must respect its measured surface type."""
    frame, baseline, scene, arrays = _fixture()
    baseline[12800]['camera'] = None
    contact = np.c_[np.arange(0, 401, 40), np.full(11, 500.)]
    frame['landmark_observations'][0].update(
        points_source_px=np.c_[np.linspace(80, 320, 16), np.full(16, 503.)].tolist(),
        association_status='associated', selected_model_id='F2')
    frame['landmark_initialization_accepted'] = True
    scene[13056]['channels']['ice_board_contact'] = {
        'observation': {'observed_xy': contact.tolist(), 'inlier_flags': [True]*len(contact),
                        'uncertainty_source_px': 6.}}
    result = resolve_frame(frame, baseline, scene, arrays)
    rejected = result['landmark_observations'][0]
    assert rejected['selected_model_id'] is None
    assert rejected['association_status'] == 'conflict'
    assert rejected['rejection_stage'] == 'geometric_category'
    assert result['landmark_camera_candidate'] is None
    assert result['landmark_initialization_accepted'] is False


@pytest.mark.parametrize('corruption', ['source_sha256', 'time_base', 'source_frame_index', 'npz_sha256'])
def test_cli_rejects_scene_provenance_mismatch_before_ranking(tmp_path, monkeypatch, corruption):
    """Same numeric PTS do not make another source or another clock equivalent."""
    script = Path(__file__).resolve().parents[1] / 'scripts/resolve_landmark_graph_v10.py'
    spec = importlib.util.spec_from_file_location('graph_cli_under_test', script)
    entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
    monkeypatch.setattr(entry, 'ROOT', tmp_path)
    implementation = tmp_path / 'src/rinkspace/landmark_graph.py'
    implementation.parent.mkdir(parents=True)
    implementation.write_bytes((script.parents[1]/'src/rinkspace/landmark_graph.py').read_bytes())
    frame = {'pts': 12800, 'time_base': '1/12800', 'source_frame_index': 50, 'shot_id': 'shot0',
             'source_sha256': 'source-A', 'landmark_observations': [], 'camera': None}
    data = {'source_sha256': 'source-A', 'model_sha256': 'model-A', 'frames': [frame]}
    baseline = deepcopy(data)
    for name, value in [('association', data), ('baseline', baseline)]:
        (tmp_path/f'{name}.json').write_text(json.dumps(value))
    np.savez(tmp_path/'scene.npz', empty=np.empty((0, 2)))
    scene = {'source_sha256': 'source-A', 'input_localization_sha256': sha256(tmp_path/'baseline.json'),
             'npz_file': 'scene.npz', 'npz_sha256': sha256(tmp_path/'scene.npz'),
             'frames': [deepcopy(frame)]}
    if corruption == 'source_sha256':
        scene['source_sha256'] = 'source-B'
    elif corruption == 'npz_sha256':
        scene['npz_sha256'] = 'not-the-actual-array-hash'
    elif corruption == 'time_base':
        scene['frames'][0]['time_base'] = '1/50'
    else:
        scene['frames'][0]['source_frame_index'] = 51
    (tmp_path/'scene.json').write_text(json.dumps(scene))
    monkeypatch.setattr(sys, 'argv', [str(script), '--association', 'association.json',
        '--baseline', 'baseline.json', '--scene', 'scene.json', '--out', 'resolved.json'])
    with pytest.raises(ValueError):
        entry.main()
    assert not (tmp_path/'resolved.json').exists()
