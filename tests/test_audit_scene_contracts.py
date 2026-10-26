"""Regression evidence for current-input identities and scene sampling."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from rinkspace.compat.d78_entities import sample_entities
from rinkspace.calibration_companion.scene_camera import Camera
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel
from rinkspace.domain.scene_assets import snapshot_scene
from rinkspace.review.render import load_state
from rinkspace.session.dataset import DatasetError, load_prepared, read_json, write_array, write_json
from rinkspace.session.prepare import prepare
import test_prepare_source_binding as source_binding_fixture
import test_d78_local_state_identity as local_identity_fixture


ROOT = Path(__file__).resolve().parents[1]


def current_input(tmp_path):
    source = tmp_path/'input'
    source_binding_fixture.PrepareSourceBinding().make_input(source, [dict(pts=100), dict(pts=150)])
    return source


def child_load(path, cwd):
    # A new interpreter at an unrelated cwd must use only the relocated
    # prepared asset and its declared current dependencies.
    environment = dict(os.environ, PYTHONPATH=str(ROOT/'src'))
    return subprocess.run([sys.executable, '-c',
        'import sys; from rinkspace.session.dataset import load_prepared; '
        'm,a=load_prepared(sys.argv[1]); print(m["revision"]); print(a["uv"].tolist())',
        str(path)], cwd=cwd, env=environment, text=True, capture_output=True)


def test_relocated_prepared_load_rejects_changed_current_bytes_in_fresh_process(tmp_path):
    source = current_input(tmp_path)
    original = tmp_path/'prepared'
    receipt = prepare(source/'input.json', source/'scope.json', 's01', original)
    moved = tmp_path/'moved'
    original.rename(moved)
    shutil.rmtree(source)
    successful = child_load(moved, tmp_path)
    assert successful.returncode == 0, successful.stderr
    assert receipt.revision in successful.stdout
    manifest = read_json(moved/'manifest.json')
    pixels = moved/manifest['arrays']['uv']['path']
    changed = np.load(pixels, allow_pickle=False)
    changed[0, 0] += 13.
    np.save(pixels, changed, allow_pickle=False)
    rejected = child_load(moved, tmp_path)
    assert rejected.returncode != 0
    assert 'Array content differs from current revision' in rejected.stderr
    # The read-only replay path must not publish altered source pixels under
    # the same prepared identity either.
    with pytest.raises(DatasetError, match='content differs'):
        load_prepared(moved, purpose='review')


def test_prepared_source_index_is_bound_to_its_actual_saved_bytes(tmp_path):
    source = current_input(tmp_path)
    destination = tmp_path/'prepared'
    prepare(source/'input.json', source/'scope.json', 's01', destination)
    manifest = read_json(destination/'manifest.json')
    clock = destination/manifest['sources'][0]['pts_ref']['path']
    changed = np.load(clock, allow_pickle=False)
    changed[0] += 1
    np.save(clock, changed, allow_pickle=False)
    with pytest.raises(DatasetError, match='content differs'):
        load_prepared(destination)


def test_unreachable_distorted_pixel_does_not_become_valid_nan_ice_position():
    camera = Camera([0., 0., 10.], [np.pi, 0., 0.], np.diag([100., 100., 1.]),
                    [-1., 0., 0., 0., 0.])
    result = camera.pixel_to_ice([100., 0.])
    assert result['valid'] is False
    assert result['xyz'] is None
    # A reachable source pixel still gives the same conditional ray/ice
    # intersection; this is not a test of body-centre vertical projection.
    xyz = np.array([.5, .2, 0.])
    uv, _ = camera.project([xyz])
    result = camera.pixel_to_ice(uv[0])
    assert result['valid'] is True
    np.testing.assert_allclose(result['xyz'], xyz, atol=1e-10)


@pytest.mark.parametrize('mismatch', ['model', 'source'])
def test_complete_local_state_must_belong_to_selected_model(tmp_path, mismatch):
    helper = local_identity_fixture.D78LocalStateIdentity()
    arguments = helper.make_input(tmp_path, np.array([.125], dtype='<f8').tobytes())
    model = read_json(arguments['model_path'])
    model['source_identity'] = 'a'*64
    write_json(arguments['model_path'], model)
    complete = read_json(arguments['complete_state_path'])
    complete.update(model={'sha256':hashlib.sha256(arguments['model_path'].read_bytes()).hexdigest()},
                    source_identity=model['source_identity'])
    if mismatch == 'model':
        complete['model']['sha256'] = 'b'*64
    else:
        complete['source_identity'] = 'b'*64
    write_json(arguments['complete_state_path'], complete)
    with pytest.raises(ValueError, match='different .*identity'):
        helper.import_state(arguments)
    assert not arguments['out_dir'].exists()


@pytest.mark.parametrize('problem,expected', [
    ('missing_local', 'nonnegative station'),
    ('split', 'closure crosses data split'),
    ('support', 'incompatible finite supports'),
    ('feature', 'different feature paths'),
])
def test_prepare_rejects_invalid_shared_material_identity(tmp_path, problem, expected):
    source = current_input(tmp_path)
    observations = read_json(source/'obs.json')
    updates = {'local_index': np.array([0, 0], dtype='<i8')}
    if problem == 'missing_local':
        updates['local_index'][:] = -1
    elif problem == 'split':
        updates['split'] = np.array([0, 2], dtype='u1')
    elif problem == 'support':
        updates['support'] = np.array([[0., .2], [.8, 1.]], dtype='<f8')
    elif problem == 'feature':
        document = read_json(source/'features.json')
        document['features'].append({**document['features'][0], 'id':'ice.B1', 'label':'ice.B1'})
        write_json(source/'features.json', document)
        updates['feature_index'] = np.array([0, 1], dtype='<i4')
    for name, values in updates.items():
        observations['arrays'][name] = write_array(source, name, values)
    write_json(source/'obs.json', observations)
    with pytest.raises(DatasetError, match=expected):
        prepare(source/'input.json', source/'scope.json', 's01', tmp_path/'prepared')
    assert not (tmp_path/'prepared').exists()


def test_shared_material_identity_with_same_split_survives_roundtrip(tmp_path):
    source = current_input(tmp_path)
    observations = read_json(source/'obs.json')
    observations['arrays']['local_index'] = write_array(source, 'local_index', [7, 7], '<i8')
    write_json(source/'obs.json', observations)
    prepare(source/'input.json', source/'scope.json', 's01', tmp_path/'prepared')
    _, arrays = load_prepared(tmp_path/'prepared')
    np.testing.assert_array_equal(arrays['local_index'], [7, 7])
    np.testing.assert_array_equal(arrays['uv'], [[10., 20.], [10., 20.]])


def test_scene_sampling_respects_partial_material_support():
    library = os.environ.get('RINKSPACE_CORE_LIBRARY')
    if not library:
        pytest.skip('explicit RINKSPACE_CORE_LIBRARY required')
    core = NativeCore(library)
    scene = {'board':[.5, .25, .1, .1, 2., .02], 'paths':[
        dict(id='line', entity_id='line', label='line', layer='ice', kind='ice_segment',
             parameters=[0., 0., 0., 1., 0., 0.], finite_support=[.25, .75]),
        dict(id='arc', entity_id='arc', label='arc', layer='ice', kind='ice_circle',
             parameters=[0., 0., 0., .1, 0., 2*np.pi], support=[.25, .75], closed=True),
    ]}
    result = sample_entities(scene, core, samples=5)
    np.testing.assert_allclose(result['xyz'][:5, 0], np.linspace(.25, .75, 5))
    np.testing.assert_allclose(result['xyz'][5:, :2],
        .1*np.column_stack([np.cos(np.linspace(np.pi/2, 3*np.pi/2, 5)),
                           np.sin(np.linspace(np.pi/2, 3*np.pi/2, 5))]), atol=1e-15)
    assert not result['cuts'][1]['closed']


@pytest.mark.parametrize('mismatch', ['shot', 'rig', 'duplicate'])
def test_saved_review_camera_cannot_be_rebound_to_another_declared_shot(tmp_path, mismatch):
    library = os.environ.get('RINKSPACE_CORE_LIBRARY')
    if not library:
        pytest.skip('explicit RINKSPACE_CORE_LIBRARY required')
    saved = tmp_path/'original'
    saved.mkdir()
    scene = {'schema':'rinkspace.scene.replay.v1', 'board':[.5, .25, .1, .1, 2., .02], 'paths':[
        dict(id='line', entity_id='line', label='line', layer='ice', kind='ice_segment',
             parameters=[0., 0., 0., 1., 0., 0.])]}
    write_json(saved/'scene.json', scene)
    camera = {'schema':'rinkspace.camera.motion.v1', 'mount_family':'orthogonal_pan_tilt_pivot',
        'lens_family':'brown_zoom_linear', 'pupil_model':'zoom_quadratic_camera_z',
        'readout_mode':'predicted_row_implicit_unique', 'shot_id':'s01', 'rig_id':'main',
        'controls_ref':'controls.npy',
        'rig':[1.,0.,0.,0.,1.,0.,0.,0.,1., 0.,0.,1., 1.,0.,0., 0.,0.,-2., 0.,0.,0.,
               0.,0.,0., 20.,1.,16.,12.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]}
    state = {'schema':'rinkspace.state.v1', 'scene_ref':'scene.json',
        'shots':[{'id':'s01', 'rig_id':'main', 'camera_ref':'camera.json'}]}
    if mismatch == 'shot':
        camera['shot_id'] = 's02'
    elif mismatch == 'rig':
        camera['rig_id'] = 'low'
    else:
        state['shots'].append(deepcopy(state['shots'][0]))
    write_json(saved/'camera.json', camera)
    write_json(saved/'state.json', state)
    np.save(saved/'controls.npy', np.zeros((4, 3), dtype='<f8'), allow_pickle=False)
    moved = tmp_path/'relocated'
    saved.rename(moved)
    with pytest.raises(ValueError, match='identity|identities'):
        load_state(moved/'state.json', NativeCore(library), ['s01'])


@pytest.mark.parametrize('single_id_only', [False, True])
def test_common_state_accepts_same_print_asset_selected_by_disjoint_references(tmp_path, single_id_only):
    scene, _ = snapshot_scene(ROOT/'events/ankara_2026/scene/field/model.json', tmp_path/'scene')
    original = deepcopy(scene['advertisement_assets'][0])
    identities = original['instance_ids']
    references = []
    for identity in identities:
        ref = {**deepcopy(original), 'instance_id':identity, 'instance_ids':[identity]}
        if single_id_only:
            ref.pop('instance_ids')
        references.append(ref)
    scene['advertisement_assets'] = references
    model = SceneParameterModel(scene, tmp_path/'scene/scene.json')
    assert {identity.split('::')[0] for identity in model.paths} == set(identities)
    placements = [b for b in model.initial_state.blocks.values() if b['kind'] == 'print_placement']
    assert all(b['active'] == b['available'] for b in placements)
    assert len(model.initial_state.blocks) == len(set(model.initial_state.blocks))
