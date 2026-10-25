"""Current field + installed print integration; synthetic pixels are not fit evidence."""
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from rinkspace.core.native import NativeCore
from rinkspace.domain.print_assets import resolve_instance
from rinkspace.domain.scene_assets import snapshot_scene, scene_content_identity
from rinkspace.review.scene import sample_review_scene, common_print_stations
from rinkspace.review.render import load_state, render, verified_source_paths
from rinkspace.session.dataset import digest, write_array, write_json
from rinkspace.storage import AssetStore
from test_print_source import make_source

ROOT = Path(__file__).resolve().parents[1]
FIELD = ROOT/'events/ankara_2026/scene/field/model.json'


def expected_current_inventory():
    """Read declared geometry directly, independently of the review sampler."""
    scene = json.loads(FIELD.read_text(encoding='utf-8'))
    paths = {path['id'] for path in scene['paths']}
    registered, localized, unlocalized = set(), set(), set()
    for ref in scene['advertisement_assets']:
        assets = json.loads((FIELD.parent/ref['path']).read_text(encoding='utf-8'))
        instances = {item['id']: item for item in assets['instances']}
        templates = {item['id']: item for item in assets['templates']}
        blocks = {item['id']: item for item in assets['parameter_blocks']}
        for identity in ref['instance_ids']:
            registered.add(identity)
            instance = instances[identity]
            if (blocks[instance['placement_block_id']]['values'] is None
                    or blocks[instance['size_block_id']]['values'] is None):
                unlocalized.add(identity)
            else:
                localized.add(identity)
                paths.update(identity+'::'+path['id']
                             for path in templates[instance['template_id']]['paths'])
    return {'path_ids': paths, 'registered': registered, 'localized': localized,
            'unlocalized': unlocalized,
            'missing_count': len(scene.get('missing_entities', []))+len(unlocalized)}


@pytest.fixture
def core():
    library = os.environ.get('RINKSPACE_CORE_LIBRARY')
    if not library:
        pytest.skip('explicit native library required')
    return NativeCore(library)


def test_current_review_keeps_all_registered_field_and_located_print_paths(core):
    scene = json.loads(FIELD.read_text(encoding='utf-8'))
    expected = expected_current_inventory()
    data = sample_review_scene(scene, core, FIELD)
    assert len(data['cuts']) == data['inventory']['path_count'] == len(expected['path_ids'])
    assert {cut['id'] for cut in data['cuts']} == expected['path_ids']
    assert set(data['inventory']['path_ids']) == expected['path_ids']
    # Keep the original three-pair coverage as an exact subset of the full scene.
    assert len(scene['paths']) == 66
    baseline = {'print_instance/Ad1', 'print_instance/Ad14',
                'print_instance/Ad2', 'print_instance/Ad3'}
    assert sum(c.get('instance_id') in baseline for c in data['cuts']) == 92
    assert data['inventory']['derived_display_paths'] == []
    assert sum(c['layer'] == 'board' for c in data['cuts']) == 2
    assert set(data['inventory']['registered_print_instance_ids']) == expected['registered']
    assert set(data['inventory']['sampled_print_instance_ids']) == expected['localized']
    assert baseline <= expected['localized']
    assert set(data['inventory']['unlocalized_print_instance_ids']) == expected['unlocalized']
    assert expected['unlocalized'] == {'print_instance/Ad15', 'print_instance/Ad16'}
    assert all(not entry['geometry_fabricated'] for entry in data['missing_entities'])
    assert np.isfinite(data['xyz']).all()


def test_changed_print_size_uses_common_material_stations_for_comparison(core, tmp_path):
    scene, _ = snapshot_scene(FIELD, tmp_path/'candidate')
    scene_path = tmp_path/'candidate/scene.json'
    before = sample_review_scene(scene, core, scene_path)
    ref = next(r for r in scene['advertisement_assets'] if r['advertisement_number'] == 3)
    path = scene_path.parent/ref['path']
    assets = json.loads(path.read_text(encoding='utf-8'))
    _, _, size, _ = resolve_instance(assets, 'print_instance/Ad3')
    size['values'][0] += np.log(1.4)
    write_json(path, assets)
    ref['sha256'] = digest(path)
    write_json(scene_path, scene)
    after = sample_review_scene(scene, core, scene_path)
    assert len(before['xyz']) != len(after['xyz'])
    grids = common_print_stations(before, after)
    old_scene = json.loads(FIELD.read_text(encoding='utf-8'))
    aligned_before = sample_review_scene(old_scene, core, FIELD, print_stations=grids)
    aligned_after = sample_review_scene(scene, core, scene_path, print_stations=grids)
    assert aligned_before['xyz'].shape == aligned_after['xyz'].shape
    assert [(p['id'], p['begin'], p['end']) for p in aligned_before['cuts']] == [
        (p['id'], p['begin'], p['end']) for p in aligned_after['cuts']]
    assert not np.array_equal(aligned_before['xyz'], aligned_after['xyz'])


def test_review_rejects_wrong_video_bytes_and_hashes_shared_source_only_once(tmp_path):
    video, _, scope, _ = make_source(tmp_path)
    manifest = {'sources': [dict(scope['source'], asset_path=str(video))],
                'shots': [{'id': sid, 'source_id': 'synthetic_only'} for sid in ('s01', 's02')]}
    store = AssetStore(tmp_path)
    with patch('rinkspace.review.render.digest', wraps=digest) as read_hash:
        result = verified_source_paths(manifest, ['s01', 's02'], {}, store)
        assert result == {'synthetic_only': video}
        assert read_hash.call_count == 1
    video.write_bytes(video.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='source video SHA256'):
        verified_source_paths(manifest, ['s01'], {}, store)


def test_complete_render_accepts_current_scene_snapshot_and_preserves_every_native_pts(core, tmp_path):
    video, _, scope, _ = make_source(tmp_path)
    prepared = tmp_path/'prepared'
    scene, _ = snapshot_scene(FIELD, prepared)
    source = dict(scope['source'], asset_path=str(video), time_base_num=1, time_base_den=12800,
                  pts_ref=write_array(prepared, 'source_pts', [0, 256, 512], '<i8'))
    arrays = {key: write_array(prepared, key, value) for key, value in {
        'uv': np.empty((0, 2), dtype='<f8'), 'pts': np.empty(0, dtype='<i8'),
        'active': np.empty(0, dtype='u1'), 'shot_index': np.empty(0, dtype='<i4')}.items()}
    shot = {'id': 's01', 'source_id': source['id'], 'processing': {
        'first_pts': 0, 'last_pts': 512, 'stop_pts_exclusive': 768, 'native_frame_count': 3}}
    write_json(prepared/'manifest.json', {'schema': 'rinkspace.prepared.v1', 'revision': 'synthetic',
        'scene_ref': 'scene.json', 'arrays': arrays, 'sources': [source], 'shots': [shot],
        'required_shot_ids': ['s01'], 'selected_shot_ids': ['s01']})
    rig = [1.,0.,0.,0.,1.,0.,0.,0.,1., 0.,0.,1., 1.,0.,0., 0.,0.,-2., 0.,0.,0.,
           0.,0.,0., 20.,1., 16.,12.,0., 0.,0., 0.,0.,0.,0.,0.,0.,0.]
    np.save(prepared/'controls.npy', np.tile([0., 0., np.log(20.)], (8, 1)), allow_pickle=False)
    camera = {'schema': 'rinkspace.camera.motion.v1', 'rig': rig, 'controls_ref': 'controls.npy',
        'mount_family': 'orthogonal_pan_tilt_pivot', 'lens_family': 'brown_zoom_linear',
        'pupil_model': 'zoom_quadratic_camera_z', 'readout_mode': 'predicted_row_implicit_unique',
        'first_control_time': -2., 'control_dt': 1., 'reference_pts': 0,
        'first_pts': 0, 'stop_pts_exclusive': 768, 'source_id': source['id'],
        'time_base_num': 1, 'time_base_den': 12800, 'image_width': 32, 'image_height': 24}
    write_json(prepared/'camera.json', camera)
    from rinkspace.domain.parameters import compile_scene_parameters, write_parameter_state
    parameters = compile_scene_parameters(scene, prepared/'scene.json')
    write_parameter_state(prepared, parameters, prepared_revision='synthetic',
        shots=[{'id':'s01','camera_ref':'camera.json'}], required_shot_ids=['s01'], selected_shot_ids=['s01'])
    write_json(tmp_path/'assets.json', {'project_root': str(tmp_path), 'sources': {source['id']: str(video)}})
    _, loaded, samples, _ = load_state(prepared/'state.json', core, ['s01'])
    assert scene_content_identity(loaded, prepared/'scene.json') == scene_content_identity(
        json.loads(FIELD.read_text(encoding='utf-8')), FIELD)
    expected = expected_current_inventory()
    assert samples['inventory']['path_count'] == len(expected['path_ids'])
    assert {cut['id'] for cut in samples['cuts']} == expected['path_ids']
    receipt = render(prepared, prepared/'state.json', tmp_path/'assets.json', 'all',
                     os.environ['RINKSPACE_CORE_LIBRARY'], tmp_path/'review',
                     compare_path=prepared/'state.json')
    result = receipt.review
    assert result['covered_pts']['s01']['count'] == 3
    assert result['artifacts'][0]['verification']['native_PTS_equal'] is True
    assert result['comparison']['maximum_uv_difference_px'] == 0.
    assert result['comparison']['status_mismatches'] == 0
    assert result['fit_executed'] is result['cuda_executed'] is False
    assert result['entity_inventory']['path_count'] == len(expected['path_ids'])
    assert set(result['entity_inventory']['path_ids']) == expected['path_ids']
    assert set(result['entity_inventory']['sampled_print_instance_ids']) == expected['localized']
    assert set(result['entity_inventory']['unlocalized_print_instance_ids']) == expected['unlocalized']
    assert len(result['missing_entities']) == expected['missing_count']
