"""Independent audit regressions for complete camera and prepared input identity."""
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from rinkspace.core.native import NativeCore
from rinkspace.domain.runtime_rigs import _camera_data, snapshot_runtime
from rinkspace.session.camera_exposure import FAMILY, snapshot_exposure
from rinkspace.session.camera_window import snapshot_window
from rinkspace.session.gpu_geometry import complete_camera_rig


@pytest.fixture
def core():
    library = os.environ.get('RINKSPACE_CORE_LIBRARY')
    if not library:
        pytest.skip('explicit native CPU library required')
    return NativeCore(library)


def camera_definition():
    rig = np.r_[np.eye(3).ravel(), [0., 0., 1.], [1., 0., 0.],
                [0., 0., 0.], [.1, .2, .3], [0., 0., 0.], [1000., 2.],
                [960., 540., 0., 0., 0., 0., 0., 0., 0., 0., 0., .02]]
    controls = np.tile([0., 0., np.log(1000.)], (12, 1)).astype('<f8')
    camera = {**FAMILY, 'rig': rig.tolist(), 'controls_ref': 'controls.npy',
              'source_id': 'native-source', 'shot_id': 's01', 'rig_id': 'main',
              'reference_pts': 2**54, 'first_pts': 2**54,
              'stop_pts_exclusive': 2**54+2048,
              'time_base_num': 1, 'time_base_den': 12800,
              'image_width': 1920, 'image_height': 1080,
              'first_control_time': -.04, 'control_dt': .02,
              'brown_wide_extension': {
                  'schema': 'rinkfit.source_supported_wide_brown_hinge.v1', 'power': 2,
                  'activation_log_f_native': float(np.log(1000.)+1),
                  'coefficients': [.2, 0., 0., 0.]}}
    return camera, controls


def save_camera(root, camera, controls):
    root.mkdir(parents=True, exist_ok=True)
    path = root/'camera.json'
    path.write_text(json.dumps(camera), encoding='utf-8')
    np.save(root/'controls.npy', controls, allow_pickle=False)
    return path


@pytest.mark.parametrize('representation', ['explicit_extension', 'saved_inactive_proof', 'packed43'])
def test_every_native_replay_keeps_active_wide_optics(core, representation):
    from rinkspace.compat.d78_state import project_native_pts
    from rinkspace.review.render import project
    from rinkspace.session.print_installation import _projection
    camera, controls = camera_definition()
    expected_pack = complete_camera_rig(camera)
    if representation == 'saved_inactive_proof':
        definition = camera.pop('brown_wide_extension')
        camera['inactive_extension_proofs'] = [{'kind': 'wide_brown_hinge',
            'original_definition': definition, 'inactive_on_original_controls': True}]
    elif representation == 'packed43':
        camera['rig'] = expected_pack.tolist()
        camera.pop('brown_wide_extension')
    point = np.array([[4., 2., 10.]])
    expected = core.project_motion(expected_pack, controls, -.04, .02, 1080, 0., point)
    omitted = core.project_motion(expected_pack[:38], controls, -.04, .02, 1080, 0., point)
    assert np.all(expected['status'] == 0)
    assert np.linalg.norm(expected['uv']-omitted['uv']) > 17.0
    before = deepcopy(camera)
    replay_camera = {**camera, 'controls': controls}
    for actual in (project(core, replay_camera, point, camera['first_pts']),
                   _projection(core, replay_camera, point, camera['first_pts']),
                   project_native_pts(core, camera, controls,
                                      np.array([camera['first_pts']], dtype=np.int64), point)):
        np.testing.assert_array_equal(actual['status'], expected['status'])
        np.testing.assert_allclose(actual['uv'], expected['uv'], atol=1e-10, rtol=0.)
    assert camera == before


@pytest.mark.parametrize('mode', ['exposure', 'window'])
def test_camera_snapshots_report_the_same_complete_native_optics(core, tmp_path, mode):
    camera, controls = camera_definition()
    path = save_camera(tmp_path/'input', camera, controls)
    if mode == 'exposure':
        result = snapshot_exposure(path, camera['first_pts'], core, tmp_path/'out')
        assert all(abs(row['k1']-.2) < 1e-12 for row in result['row_states'])
    else:
        result = snapshot_window(path, [camera['first_pts'], camera['first_pts']+256], core, tmp_path/'out')
        # The snapshot evaluator must receive the full pack even when no row
        # states are exposed in its report. Compare the saved evaluator below.
        assert result['checked_row_states'] == 6
    local, _, _, _, kept = _camera_data(tmp_path/'out/camera.json')
    expected = core.evaluate_motion_camera(complete_camera_rig(camera), controls, -.04, .02, [0.])
    actual = core.evaluate_motion_camera(complete_camera_rig(local), kept,
        local['first_control_time'], .02, [0.])
    np.testing.assert_allclose(actual['values'], expected['values'], atol=1e-10, rtol=0.)


def test_runtime_snapshot_accepts_exported_complete_43_value_camera(tmp_path):
    camera, controls = camera_definition()
    camera['rig'] = complete_camera_rig(camera).tolist()
    path = save_camera(tmp_path/'input', camera, controls)
    for name, document in (
        ('rigs', {'rigs': [{'rig_id': 'main', 'camera_ref': 'camera.json'}]}),
        ('shots', {'shots': [{'id': 's01', 'rig_id': 'main', 'camera_ref': 'camera.json'}]})):
        (path.parent/(name+'.json')).write_text(json.dumps(document), encoding='utf-8')
    rigs, shots, _ = snapshot_runtime(path.parent/'rigs.json', path.parent/'shots.json', tmp_path/'out')
    relocated = tmp_path/'out'/shots['shots'][0]['camera_ref']
    copied, _, _, _, copied_controls = _camera_data(relocated)
    np.testing.assert_array_equal(copied['rig'], camera['rig'])
    np.testing.assert_array_equal(copied_controls, controls)
    assert rigs['rigs'][0]['camera_ref'] == shots['shots'][0]['camera_ref']


@pytest.mark.parametrize('defect', ['wide_nonfinite', 'unknown_extension', 'base_nonfinite'])
def test_invalid_effective_optics_rejected_before_snapshot_mutation(tmp_path, defect):
    camera, controls = camera_definition()
    if defect == 'wide_nonfinite':
        camera['brown_wide_extension']['coefficients'][0] = float('inf')
    elif defect == 'unknown_extension':
        camera['brown_wide_extension']['power'] = 3
    else:
        camera['rig'][26] = float('nan')
    path = save_camera(tmp_path/'input', camera, controls)
    with pytest.raises(ValueError):
        _camera_data(path)


def prepared_camera_input(tmp_path):
    # Reuse the existing two-source/two-shot fixture to ensure this exercises
    # prepare and reload, not a hand-built manifest that could omit dependencies.
    from test_prepare_source_binding import PrepareSourceBinding
    from rinkspace.session.dataset import read_json, write_json
    root = tmp_path/'input'
    PrepareSourceBinding().make_input(root, [dict(pts=100), dict(pts=150)])
    camera, controls = camera_definition()
    camera.update(source_id='official', reference_pts=100, first_pts=100, stop_pts_exclusive=200)
    save_camera(root, camera, controls)
    shots = read_json(root/'shots.json')
    shots['shots'][0].update(rig_id='main', camera_ref='camera.json')
    write_json(root/'shots.json', shots)
    write_json(root/'rigs.json', {'rigs': [{'rig_id': 'main', 'camera_ref': 'camera.json'}]})
    return root


@pytest.mark.parametrize('changed', ['camera', 'controls', 'scope', 'manifest_source_clock',
                                     'manifest_feature', 'required_shots', 'scope_status'])
def test_prepared_reload_rejects_changed_current_dependencies(tmp_path, changed):
    from rinkspace.session.dataset import DatasetError, load_prepared, read_json, write_json
    from rinkspace.session.prepare import prepare
    root = prepared_camera_input(tmp_path)
    out = tmp_path/'prepared'
    prepare(root/'input.json', root/'scope.json', 's01', out)
    original, arrays = load_prepared(out)
    camera_path = out/original['shots'][0]['camera_ref']
    if changed == 'controls':
        path = camera_path.parent/read_json(camera_path)['controls_ref']
        controls = np.load(path, allow_pickle=False).copy()
        controls[:, 0] += .01
        np.save(path, controls, allow_pickle=False)
    elif changed == 'camera':
        camera = read_json(camera_path)
        camera['rig'][26] += 10.
        write_json(camera_path, camera)
    elif changed == 'scope':
        path = out/original['scope_ref']
        scope = read_json(path)
        scope['shots'][0]['processing']['stop_pts_exclusive'] += 1
        write_json(path, scope)
    else:
        if changed == 'manifest_source_clock':
            original['sources'][0]['time_base_den'] *= 2
        elif changed == 'manifest_feature':
            original['features'][0]['label'] = 'ice.B1'
        elif changed == 'required_shots':
            original['required_shot_ids'] = ['s01']
        else:
            original['scope_status'] = 'complete'
        write_json(out/'manifest.json', original)
    for purpose in ('fit', 'review', 'camera_navigation'):
        with pytest.raises(DatasetError, match='revision|snapshot'):
            load_prepared(out, purpose=purpose)


@pytest.mark.parametrize('defect', ['source', 'clock', 'image', 'processing', 'spline', 'readout', 'reference_pts'])
def test_prepare_checks_actual_camera_not_only_domain_claim(tmp_path, defect):
    from rinkspace.session.dataset import read_json, write_json
    from rinkspace.session.prepare import prepare
    root = prepared_camera_input(tmp_path)
    path = root/'camera.json'
    camera = read_json(path)
    if defect == 'source':
        camera['source_id'] = 'other'
    elif defect == 'clock':
        camera['time_base_den'] = 60000
    elif defect == 'image':
        camera['image_width'] = 1280
    elif defect == 'processing':
        camera['stop_pts_exclusive'] = 150
    elif defect == 'spline':
        camera['first_control_time'] = 1.
    elif defect == 'readout':
        camera['rig'][37] = 1.
    else:
        camera['reference_pts'] = 100.25
    write_json(path, camera)
    with pytest.raises(ValueError, match='Camera|camera'):
        prepare(root/'input.json', root/'scope.json', 's01', tmp_path/'prepared')
    assert not (tmp_path/'prepared').exists()


def navigation_geometry(core, monkeypatch, xyz):
    from types import SimpleNamespace
    from rinkspace.session import source_navigation
    monkeypatch.setattr(source_navigation, 'scene_content_identity', lambda scene, path: 'scene-revision')
    return SimpleNamespace(core=core, scene={}, scene_path=Path('scene.json'),
        samples={'xyz': np.asarray(xyz), 'cuts': [{'id': 'path', 'label': 'ice.C', 'layer': 'ice',
            'begin': 0, 'end': len(xyz), 'support': [0., 1.]}]},
        current_paths={'path': {'id': 'path', 'label': 'ice.C'}},
        ice_inventory=lambda: [], refs=[{'sha256': 'scene-hash'}])


def test_source_search_proposals_use_the_complete_optical_state(core, monkeypatch):
    from rinkspace.session.source_navigation import project_navigation
    camera, controls = camera_definition()
    points = np.array([[4., 2., 10.], [4.5, 2., 10.]])
    geometry = navigation_geometry(core, monkeypatch, points)
    result = project_navigation(geometry, {'camera': camera, 'controls': controls},
                                camera['first_pts'], {'search_radius_px': 18.})
    expected = core.project_motion(complete_camera_rig(camera), controls, -.04, .02, 1080, 0., points)
    assert len(result) == 1
    np.testing.assert_allclose(result[0]['points_uv'], expected['uv'], atol=1e-10, rtol=0.)
    assert result[0]['hint_is_source_measurement'] is False


def test_source_search_proposals_respect_the_registered_native_width(core, monkeypatch):
    from rinkspace.session.source_navigation import project_navigation
    camera, controls = camera_definition()
    camera['image_width'] = 1280
    camera['rig'][26] = 640.
    geometry = navigation_geometry(core, monkeypatch, [[8., 2., 10.], [9., 2., 10.]])
    result = project_navigation(geometry, {'camera': camera, 'controls': controls},
                                camera['first_pts'], {'search_radius_px': 18.})
    assert result == []


@pytest.mark.parametrize('changed', ['rig', 'wide', 'time_origin', 'clock', 'control_dt'])
def test_source_search_revision_binds_optics_and_time_not_only_controls(core, monkeypatch, changed):
    from rinkspace.session.source_navigation import project_navigation
    camera, controls = camera_definition()
    geometry = navigation_geometry(core, monkeypatch, [[4., 2., 10.], [4.5, 2., 10.]])
    def revision():
        result = project_navigation(geometry, {'camera': camera, 'controls': controls},
            camera['first_pts'], {'search_radius_px': 18.})
        return result[0]['model_search_projection']['camera_revision']
    before = revision()
    if changed == 'rig':
        camera['rig'][26] += 10.
    elif changed == 'wide':
        camera['brown_wide_extension']['coefficients'][0] = .1
    elif changed == 'time_origin':
        camera['reference_pts'] += 1
    elif changed == 'clock':
        camera['time_base_den'] *= 2
    else:
        camera['control_dt'] += .001
    assert revision() != before


@pytest.mark.parametrize('formatting', ['compact', 'crlf', 'bom'])
@pytest.mark.parametrize('legacy_scope_hash', [False, True])
def test_prepared_scope_serialization_preserves_semantic_identity(
        tmp_path, monkeypatch, formatting, legacy_scope_hash):
    import importlib
    import shutil
    from rinkspace.session.dataset import DatasetError, digest, load_prepared, read_json, write_json
    prepare_module = importlib.import_module('rinkspace.session.prepare')
    root = prepared_camera_input(tmp_path)
    scope_path = root/'scope.json'
    scope = read_json(scope_path)
    if formatting == 'compact':
        raw = json.dumps(scope, separators=(',', ':')).encode('utf-8')
    elif formatting == 'crlf':
        raw = (json.dumps(scope, indent=2)+'\n').replace('\n', '\r\n').encode('utf-8')
    else:
        raw = b'\xef\xbb\xbf'+(json.dumps(scope, indent=2)+'\n').encode('utf-8')
    scope_path.write_bytes(raw)
    source_hash = digest(scope_path)
    out = tmp_path/'prepared'
    if legacy_scope_hash:
        # Recreate the prior writer's recorded source-byte hash and revision,
        # while still exercising the real prepare/snapshot/load path.
        monkeypatch.setattr(prepare_module, 'digest', lambda path:
            source_hash if Path(path) == out/'scope.json' else digest(path))
    prepare_module.prepare(root/'input.json', scope_path, 's01', out)
    manifest = read_json(out/'manifest.json')
    snapshot_hash = digest(out/'scope.json')
    assert snapshot_hash != source_hash
    assert manifest['dependency_revision']['scope'] == (source_hash if legacy_scope_hash else snapshot_hash)
    shutil.rmtree(root)  # A saved current snapshot must not need its producer.
    for purpose in ('fit', 'review', 'camera_navigation'):
        loaded, _ = load_prepared(out, purpose=purpose)
        assert loaded['revision'] == manifest['revision']
    scope['shots'][0]['processing']['stop_pts_exclusive'] += 1
    write_json(out/'scope.json', scope)
    for purpose in ('fit', 'review', 'camera_navigation'):
        with pytest.raises(DatasetError, match='revision|snapshot'):
            load_prepared(out, purpose=purpose)
