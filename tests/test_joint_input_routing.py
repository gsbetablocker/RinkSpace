"""Synthetic input routing: identities and held-out pixels must not be invented."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rinkspace.calibration_companion import Camera, look_at_rvec
from rinkspace.calibration_companion.advertisement_tracks import assemble_tracks
from rinkspace.calibration_companion.camera_identity import (
    CAMERA_IDS,
    installation_records,
    resolve_frame_camera,
)


SIDES = ['upper_long_side', 'lower_long_side', 'left_end', 'right_end']
CORNERS = ['upper_left', 'upper_right', 'lower_left', 'lower_right']
PARAMETERS = {'bounds': [-5., -3., 5., 3.], 'corner_radius': .8,
              'corner_radii': {'upper_left': .7, 'upper_right': 1.1,
                               'lower_left': .9, 'lower_right': 1.3},
              'board_height': .6}


def frame(name='source_frame', camera='secondary_long_side', **extra):
    return {'id': name, 'source_id': 'oxXu1avhrvs', 'camera_id': camera,
            'media_track_id': 'synthetic:source', 'source_pts': 153600,
            'time_base': '1/15360', **extra}


@pytest.fixture
def runner(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('synthetic_input_routing_runner',
                                                 scripts / 'fit_joint_scene_video.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_four_physical_cameras_keep_separate_installations_and_free_guesses():
    assert set(CAMERA_IDS) == {'camera_main_long_side', 'camera_secondary_long_side',
                               'camera_low_end', 'camera_high_oblique'}
    frames = [frame(str(i), camera) for i, camera in enumerate(CAMERA_IDS)]
    resolved = [resolve_frame_camera(row) for row in frames]
    assert len({row['installation_id'] for row in resolved}) == 4
    records = installation_records(frames, legacy_main_center=[-3., -99., 18.])
    assert len({tuple(row['initial_center_xyz']) for row in records}) == 4
    for row in records:
        assert row['fit_center'] is True
        assert row['initialization_status'] == 'conditional_free_initial_guess_NOT_measured_position'
        if row['physical_camera_id'] != 'camera_main_long_side':
            assert row['initial_center_xyz'] != [-3., -99., 18.]


@pytest.mark.parametrize('identity', [
    {}, {'camera_id': 'unknown'}, {'camera_id': 'nonwide_unknown'},
    {'installation_id': 'prefix_camera_main_long_side_observed_keyframes'},
    {'installation_id': 'a_new_camera_that_is_not_high'},
])
def test_missing_or_unresolved_identity_cannot_fall_back_to_main(identity):
    with pytest.raises(ValueError):
        resolve_frame_camera({'id': 'unresolved', 'source_id': 'oxXu1avhrvs', **identity})


@pytest.mark.parametrize('claims', [
    {'camera_id': 'low_end', 'physical_camera_id': 'main_long_side'},
    {'camera_id': 'secondary_long_side', 'installation_id': 'women_sp_main'},
    {'camera_id': 'high_oblique_overview', 'installation_id': 'women_sp_low'},
])
def test_conflicting_camera_fields_are_rejected(claims):
    with pytest.raises(ValueError, match='conflicting'):
        resolve_frame_camera({'id': 'conflict', 'source_id': 'oxXu1avhrvs', **claims})


def test_exact_legacy_alias_is_retained_but_custom_epoch_and_supplied_pose_survive():
    legacy = resolve_frame_camera({'id': 'legacy',
        'installation_id': 'oxXu1avhrvs_camera_main_long_side_observed_keyframes'})
    assert legacy['installation_id'] == 'women_sp_main'
    assert legacy['physical_camera_id'] == 'camera_main_long_side'
    supplied = [{'id': 'custom_secondary_after_move',
                 'physical_camera_id': 'camera_secondary_long_side',
                 'installation_epoch': 'second_setup', 'fit_center': False,
                 'initial_center_xyz': [22., -130., 8.],
                 'center_bounds_xyz': [[20., -135., 6.], [25., -125., 10.]]}]
    original = copy.deepcopy(supplied)
    source = frame('new_epoch', 'secondary_long_side',
                   installation_id='custom_secondary_after_move',
                   installation_epoch='second_setup')
    resolved = resolve_frame_camera(source, supplied)
    assert resolved['installation_id'] == 'custom_secondary_after_move'
    assert resolved['installation_epoch'] == 'second_setup'
    records = installation_records([source], supplied, legacy_main_center=[0., -80., 8.], start=1)
    assert records == original
    assert supplied == original
    # An explicit epoch is not folded into the default women's-short group.
    next_epoch = frame('third_epoch', 'secondary_long_side', installation_id='third_setup')
    separate = installation_records([source, next_epoch], supplied)
    assert {row['id'] for row in separate} == {'custom_secondary_after_move', 'third_setup'}
    assert separate[1]['fit_center'] is True
    separate[1]['initial_center_xyz'][0] += 100
    assert supplied == original


def test_registry_conflict_and_two_cameras_sharing_one_epoch_are_rejected():
    supplied = [{'id': 'explicit_epoch', 'physical_camera_id': 'camera_low_end'}]
    with pytest.raises(ValueError, match='conflicting'):
        resolve_frame_camera(frame(installation_id='explicit_epoch'), supplied)
    with pytest.raises(ValueError, match='multiple physical cameras'):
        installation_records([frame('a', 'low_end', installation_id='shared_epoch'),
                              frame('b', 'secondary_long_side', installation_id='shared_epoch')])


def test_supplied_legacy_installation_centre_is_not_lost_when_its_id_is_normalized():
    legacy_id = 'oxXu1avhrvs_camera_main_long_side_observed_keyframes'
    supplied = [{'id': legacy_id, 'physical_camera_id': 'camera_main_long_side',
                 'initial_center_xyz': [12., -77., 9.], 'fit_center': False,
                 'center_bounds_xyz': [[10., -79., 7.], [14., -75., 11.]]}]
    records = installation_records([{'id': 'old_frame', 'installation_id': legacy_id}], supplied,
                                   legacy_main_center=[-3., -99., 18.])
    assert records[0]['id'] == 'women_sp_main'
    assert records[0]['initial_center_xyz'] == [12., -77., 9.]
    assert records[0]['fit_center'] is False


def test_generic_camera_names_do_not_merge_installations_across_sources_without_registry():
    frames = [{'id': source, 'source_id': source, 'installation_id': 'main_long_side'}
              for source in ['source_A', 'source_B']]
    separate = [resolve_frame_camera(row) for row in frames]
    assert separate[0]['installation_id'] != separate[1]['installation_id']
    assert len(installation_records(frames)) == 2
    supplied = [{'id': 'main_long_side', 'physical_camera_id': 'camera_main_long_side',
                 'initial_center_xyz': [12., -77., 9.], 'fit_center': False}]
    shared = [resolve_frame_camera(row, supplied) for row in frames]
    assert {row['installation_id'] for row in shared} == {'main_long_side'}
    assert len(installation_records(frames, supplied)) == 1


def legacy_board_observation(side, use='fit', edge='contact'):
    return {'id': side+'_'+use, 'frame_id': 'source_frame', 'physical_group_id': side+'_'+edge,
            'type': 'curve', 'target': {'kind': 'board_curve', 'height': .6 if edge == 'top' else 0.,
                                       's_range': [3., 4.]},
            'use': use, 'pixels_uv': [[10., 20.], [15., 25.]], 'sigma_px': 1.}


def test_normalization_preserves_holdout_roles_and_all_eight_surface_identities(runner):
    observations = [legacy_board_observation(side, 'fit' if i%2 else 'check',
                                               'contact' if i%2 else 'top')
                    for i, side in enumerate(SIDES+CORNERS)]
    source = {'frames': [frame()], 'observations': observations}
    before = copy.deepcopy(source)
    normalized = runner.normalize_observations(source)
    assert source == before
    for actual, original in zip(normalized['observations'], observations):
        assert actual['use'] == original['use']
        assert actual['pixels_uv'] == original['pixels_uv']
        assert actual['target'] == {'kind': 'board_side',
                                     'side': original['id'].removesuffix('_'+original['use']),
                                     'edge': 'top' if original['target']['height'] else 'contact'}
    assert normalized['frames'][0]['physical_camera_id'] == 'camera_secondary_long_side'


@pytest.mark.parametrize('mutation', ['unknown_side', 'two_sides', 'side_conflict',
                                     'edge_conflict', 'group_edge_conflict', 'missing_use'])
def test_ambiguous_or_conflicting_legacy_board_fields_cannot_be_silently_rewritten(runner, mutation):
    observation = legacy_board_observation('left_end')
    if mutation == 'unknown_side':
        observation['physical_group_id'] = 'some_board_contact'
    elif mutation == 'two_sides':
        observation['physical_group_id'] = 'left_end_right_end_contact'
    elif mutation == 'side_conflict':
        observation['target']['side'] = 'right_end'
    elif mutation == 'edge_conflict':
        observation['target']['edge'] = 'top'
    elif mutation == 'group_edge_conflict':
        observation['physical_group_id'] = 'left_end_top'
    else:
        del observation['use']
    with pytest.raises(ValueError):
        runner.normalize_observations({'frames': [frame()], 'observations': [observation]})


def test_identified_top_edge_does_not_require_a_measured_height(runner):
    observation = legacy_board_observation('lower_long_side', edge='top')
    observation['target'].update(height=None, edge='top')
    normalized = runner.normalize_observations({'frames': [frame()], 'observations': [observation]})
    assert normalized['observations'][0]['target'] == {
        'kind': 'board_side', 'side': 'lower_long_side', 'edge': 'top'}


def track(track_id, instance, side, *, observations=None, **extra):
    return {'track_id': track_id, 'advertisement_instance_id': instance,
            'label': 'SAME BRAND', 'board_side': side, **extra,
            'observations': observations or [{'frame_id': 'source_frame',
                'pixels_uv': [400., 300.], 'sigma_px': 1., 'use': 'fit'}]}


def test_repeated_brand_instances_remain_separate_without_any_high_camera():
    source = {'frames': [frame()], 'landmarks': [
        track('left_letter', 'left_copy', 'left_end'),
        track('right_letter', 'right_copy', 'right_end')]}
    before = copy.deepcopy(source)
    result = assemble_tracks(source, PARAMETERS, identity_records=[
        {'advertisement_instance_id': 'left_copy', 'user_number': 9},
        {'advertisement_instance_id': 'right_copy', 'user_number': None}])
    assert source == before
    assert len(result['board_landmarks']) == 2
    assert {row['advertisement_instance_id'] for row in result['board_landmarks']} == {'left_copy', 'right_copy'}
    assert len({o['correlation_group_id'] for o in result['observations']}) == 2
    assert result['frames'][0]['physical_camera_id'] == 'camera_secondary_long_side'
    for landmark in result['board_landmarks']:
        assert landmark['fit'] and len(landmark['evidence_observation_ids']) == 1
        assert not landmark['provenance']['initialization']['original_user_schematic_xyz_used_as_measurement']


def test_only_fit_pixels_can_initialize_points_and_check_only_tracks_require_frozen_estimates():
    xyz = [1., -3., .3]
    centre = np.array([0., 0., 2.])
    camera = Camera(centre, look_at_rvec(centre, xyz),
                    [[500., 0., 400.], [0., 500., 300.], [0., 0., 1.]], [])
    uv = camera.project([xyz])[0][0].tolist()

    class ForbiddenCheckCamera:
        @property
        def K_px(self):
            raise AssertionError('held-out camera used to initialize a training point')

    rows = [{'frame_id': 'check_frame', 'pixels_uv': [20., 40.], 'use': 'check', 'sigma_px': 1.},
            {'frame_id': 'source_frame', 'pixels_uv': uv, 'use': 'fit', 'sigma_px': 1.}]
    source = {'frames': [frame(), frame('check_frame', source_pts=153856)],
              'landmarks': [track('shared_detail', 'instance_a', 'lower_long_side', observations=rows)]}
    cameras = {'source_frame': camera, 'check_frame': ForbiddenCheckCamera()}
    result = assemble_tracks(source, PARAMETERS, cameras=cameras)
    shared = result['board_landmarks'][0]
    assert shared['initial_along'] == pytest.approx(1., abs=1e-9)
    assert shared['initial_z_fraction'] == pytest.approx(.5, abs=1e-9)
    assert shared['evidence_observation_ids'] == ['source_frame_shared_detail']
    assert result['fit_observation_count'] == 1 and result['check_observation_count'] == 1
    changed = copy.deepcopy(source)
    for detail in changed['landmarks']:
        for observation in detail['observations']:
            if observation['use'] == 'check':
                observation['pixels_uv'] = [99999., -33333.]
    again = assemble_tracks(changed, PARAMETERS, cameras=cameras)
    assert again['board_landmarks'] == result['board_landmarks']
    check_only = {'frames': source['frames'], 'landmarks': [
        track('only_check', 'instance_b', 'right_end', observations=rows[:1])]}
    with pytest.raises(ValueError, match='frozen'):
        assemble_tracks(check_only, PARAMETERS, cameras=cameras)
    initializer = {'id': 'only_check', 'advertisement_instance_id': 'instance_b', 'side': 'right_end',
                   'initial_along': 0., 'initial_z_fraction': .5,
                   'bounds_along': [-2., 2.], 'bounds_z_fraction': [0., 1.]}
    with pytest.raises(ValueError, match='frozen'):
        assemble_tracks(check_only, PARAMETERS, cameras=cameras, initial_landmarks=[initializer])
    frozen = {**initializer, 'along': .7, 'z_fraction': .6}
    checked = assemble_tracks(check_only, PARAMETERS, cameras=cameras, initial_landmarks=[frozen])
    heldout = checked['board_landmarks'][0]
    assert heldout['fit'] is False and heldout['evidence_observation_ids'] == []
    assert heldout['initial_along'] == .7 and heldout['initial_z_fraction'] == .6
    assert heldout['provenance']['initialization']['method'] == 'provided_frozen_conditional_estimate_for_check_only'


def test_reused_initializer_cannot_cross_an_advert_instance_or_surface():
    source = {'frames': [frame()], 'landmarks': [track('detail', 'copy_a', 'right_end')]}
    old = {'id': 'detail', 'advertisement_instance_id': 'copy_b', 'side': 'right_end'}
    with pytest.raises(ValueError, match='another instance/surface'):
        assemble_tracks(source, PARAMETERS, initial_landmarks=[old])
    old.update(advertisement_instance_id='copy_a', side='left_end')
    with pytest.raises(ValueError, match='another instance/surface'):
        assemble_tracks(source, PARAMETERS, initial_landmarks=[old])


def test_reused_initializer_cannot_keep_permission_for_the_opposite_corner():
    source = {'frames': [frame()], 'landmarks': [
        track('detail', 'copy_a', 'upper_long_side', allow_corner='upper_right')]}
    old = {'id': 'detail', 'advertisement_instance_id': 'copy_a', 'side': 'upper_long_side',
           'allow_corner': 'upper_left', 'initial_along': -4.6,
           'bounds_along': [-4.99, 4.99], 'initial_z_fraction': .4,
           'bounds_z_fraction': [.05, .95]}
    with pytest.raises(ValueError):
        assemble_tracks(source, PARAMETERS, initial_landmarks=[old])


def test_tiny_negative_angle_at_upper_right_endpoint_does_not_jump_to_other_arc_end():
    radius = PARAMETERS['corner_radii']['upper_right']
    # Roundoff just below angle zero is accepted at the endpoint by the finite
    # surface intersection; normalizing 360 degrees again must not turn it into 90.
    xyz = np.array([5., 3.-radius-radius*1e-10, .3])
    centre = np.array([0., 0., 2.])
    camera = Camera(centre, look_at_rvec(centre, xyz),
                    [[500., 0., 400.], [0., 500., 300.], [0., 0., 1.]], [])
    uv = camera.project([xyz])[0][0].tolist()
    source = {'frames': [frame()], 'landmarks': [track('corner', 'corner_copy', 'upper_right',
        observations=[{'frame_id': 'source_frame', 'pixels_uv': uv, 'use': 'fit', 'sigma_px': 1.}])]}
    result = assemble_tracks(source, PARAMETERS, cameras={'source_frame': camera})
    point = result['board_landmarks'][0]
    intersection = point['provenance']['initialization']['ray_intersection']
    assert intersection['valid']
    assert intersection['along'] == pytest.approx(0., abs=1e-8)
    assert point['initial_along'] == pytest.approx(0., abs=1e-8)


SURFACE_CASES = [(side, None) for side in SIDES+CORNERS] + [
    ('upper_long_side', 'upper_left'), ('upper_long_side', 'upper_right'),
    ('lower_long_side', 'lower_left'), ('lower_long_side', 'lower_right'),
    ('left_end', 'lower_left'), ('left_end', 'upper_left'),
    ('right_end', 'lower_right'), ('right_end', 'upper_right'),
]


@pytest.mark.parametrize('side,corner', SURFACE_CASES)
def test_each_surface_routes_to_its_own_geometry_parameters_not_an_upper_side_fallback(runner, side, corner):
    source = {'frames': [frame()], 'landmarks': [track('detail', 'physical_copy', side,
        **({'allow_corner': corner} if corner else {}))]}
    assembled = assemble_tracks(source, PARAMETERS)
    point = assembled['board_landmarks'][0]
    assert point['side'] == side
    if side in CORNERS:
        assert point['along_parameterization'] == 'corner_fraction'
        assert point['bounds_along'] == [0., 1.]
    if corner:
        assert point['allow_corner'] == corner
    check = copy.deepcopy(assembled['observations'][0])
    check.update(id='excluded_check_evidence', use='check')
    specs, _ = runner.scene_parameterization(SimpleNamespace(features={}),
        assembled['observations'] + [check], assembled['board_landmarks'])
    by_kind = {row['kind']: row for row in specs if row['kind'] != 'boundary_corner_radius'}
    expected_axes = {0, 1} if side in CORNERS or corner else (
        {1} if side in SIDES[:2] else {0})
    fit_id = assembled['observations'][0]['id']
    for axis, kind in [(0, 'boundary_length'), (1, 'boundary_width')]:
        assert by_kind[kind]['evidence_observation_ids'] == ([fit_id] if axis in expected_axes else [])
    assert by_kind['board_height']['evidence_observation_ids'] == [fit_id]
    expected_corner = side if side in CORNERS else corner
    for spec in specs:
        assert 'excluded_check_evidence' not in spec['evidence_observation_ids']
        if spec['kind'] == 'boundary_corner_radius':
            assert spec['evidence_observation_ids'] == ([fit_id] if spec['corner'] == expected_corner else [])
