"""Prepared rigs retain only explicit current camera and motion dependencies."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from rinkspace.domain.runtime_rigs import snapshot_rigs, snapshot_runtime


ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ROOT/'events/ankara_2026/programmes/okada_mei_sp/ad1_cameras'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


@pytest.fixture
def local_rigs(tmp_path):
    root = tmp_path/'source'
    rows, originals = [], []
    for shot in ('s03', 's06'):
        path = CAMERAS/f'{shot}.json'
        camera = json.loads(path.read_text(encoding='utf-8'))
        controls_raw = (path.parent/camera['controls_ref']).read_bytes()
        # Relative controls belong to the camera, including valid ../ siblings.
        camera['controls_ref'] = f'../motion/{shot}.npy'
        target = root/'models'/path.name
        write(target, camera)
        (root/'motion').mkdir(exist_ok=True)
        (root/'motion'/f'{shot}.npy').write_bytes(controls_raw)
        rows.append({'rig_id': camera['rig_id'], 'camera_ref': f'../models/{path.name}'})
        originals.append((camera, controls_raw))
    rows.extend([{'rig_id': 'missing', 'camera_ref': None, 'state_status': 'missing_original_parameters'},
                 {'id': 'legacy', 'status': 'frozen_d78_replay'}])
    rigs = {'schema': 'rinkspace.rigs.v1', 'rigs': rows,
            'origin': {'unavailable_history': 'do-not-open/original.json'}}
    source = root/'registry'/'rigs.json'
    write(source, rigs)
    return source, rigs, originals


def test_current_cameras_remain_readable_without_original_directory(local_rigs, tmp_path):
    source, rigs, originals = local_rigs
    out = tmp_path/'prepared'; out.mkdir()
    (out/'scene.json').write_bytes(b'untouched scene')
    snapshot, summary = snapshot_rigs(source, out)
    restored = deepcopy(snapshot)
    for old, new in zip(rigs['rigs'], restored['rigs']):
        if old.get('camera_ref') is not None:
            new['camera_ref'] = old['camera_ref']
    assert restored == rigs
    shutil.rmtree(source.parent.parent)
    assert (out/'scene.json').read_bytes() == b'untouched scene'
    assert json.loads((out/'rigs.json').read_text()) == snapshot
    assert len(summary) == 4
    for row, (original, controls_raw) in zip(snapshot['rigs'], originals):
        path = out/row['camera_ref']
        camera = json.loads(path.read_text())
        controls_path = path.parent/camera['controls_ref']
        assert controls_path.read_bytes() == controls_raw
        controls = np.load(controls_path, allow_pickle=False)
        assert controls.ndim == 2 and controls.shape[1] == 3 and len(controls) >= 4
        camera['controls_ref'] = original['controls_ref']
        assert camera == original  # All numerical values and provenance unchanged.
    for dependency in summary:
        raw = (out/dependency['snapshot_path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == dependency['sha256']
        assert len(raw) == dependency['size_bytes']


def test_old_rigs_without_cameras_need_no_extra_files(tmp_path):
    source = tmp_path/'input.json'
    rigs = {'rigs': [{'id': 'main', 'status': 'frozen_d78_replay'}, {'rig_id': 'other', 'camera_ref': None}]}
    write(source, rigs)
    snapshot, summary = snapshot_rigs(source, tmp_path/'prepared')
    assert snapshot == rigs and summary == []
    assert sorted(p.name for p in (tmp_path/'prepared').iterdir()) == ['rigs.json']


@pytest.mark.parametrize('defect', ['rig_shape', 'control_shape', 'nonfinite', 'wrong_dtype', 'missing', 'identity'])
def test_all_current_dependencies_are_validated_before_output_mutation(local_rigs, tmp_path, defect):
    source, rigs, _ = local_rigs
    camera_path = (source.parent/rigs['rigs'][1]['camera_ref']).resolve()
    camera = json.loads(camera_path.read_text())
    controls_path = (camera_path.parent/camera['controls_ref']).resolve()
    if defect == 'rig_shape':
        camera['rig'].pop()
        write(camera_path, camera)
        expected = '38 finite'
    elif defect == 'identity':
        camera['rig_id'] = 'wrong'
        write(camera_path, camera)
        expected = 'rig_id differ'
    elif defect == 'missing':
        controls_path.unlink()
        expected = None
    else:
        controls = np.zeros((4, 3), dtype='<f8')
        if defect == 'control_shape':
            controls = np.zeros((4, 2), dtype='<f8')
        elif defect == 'nonfinite':
            controls[0, 0] = np.nan
        else:
            controls = controls.astype('<f4')
        np.save(controls_path, controls, allow_pickle=False)
        expected = r'float64\[N,3\]'
    out = tmp_path/'prepared'; out.mkdir()
    (out/'scene.json').write_bytes(b'preserved')
    with pytest.raises(FileNotFoundError if defect == 'missing' else ValueError, match=expected):
        snapshot_rigs(source, out)
    assert [(p.name, p.read_bytes()) for p in out.iterdir()] == [('scene.json', b'preserved')]


@pytest.mark.parametrize('occupied', ['rigs.json', 'cameras'])
def test_existing_destinations_are_never_overwritten(local_rigs, tmp_path, occupied):
    source, _, _ = local_rigs
    out = tmp_path/'prepared'; out.mkdir()
    target = out/occupied
    target.write_bytes(b'keep')
    with pytest.raises(FileExistsError):
        snapshot_rigs(source, out)
    assert target.read_bytes() == b'keep'
    assert [p.name for p in out.iterdir()] == [occupied]


def test_invalid_camera_path_type_is_rejected(local_rigs, tmp_path):
    source, rigs, _ = local_rigs
    rigs['rigs'][0]['camera_ref'] = {'path': 'camera.json'}
    write(source, rigs)
    with pytest.raises(ValueError, match='camera_ref needs an explicit file path'):
        snapshot_rigs(source, tmp_path/'prepared')
    assert not (tmp_path/'prepared').exists()


def test_shots_share_camera_files_and_retain_shot_only_current_camera(local_rigs, tmp_path):
    source, rigs, originals = local_rigs
    # Current s03 is shared between its rig and shot; s06 is declared only by
    # its shot. Neither can depend on the old camera.json location after copy.
    rigs['rigs'][1]['camera_ref'] = None
    write(source, rigs)
    shots = {'schema': 'rinkspace.shots.v1', 'required_shot_ids': ['s03', 's06', 's09'],
             'shots': [{'id': shot, 'rig_id': camera['rig_id'],
                        'camera_ref': f'../../models/{shot}.json', 'source_id': camera['source_id']}
                       for shot, (camera, _) in zip(('s03', 's06'), originals)] +
                      [{'id': 's09', 'camera_ref': None, 'state_status': 'not_imported_no_invented_state'}]}
    shots_path = source.parent/'nested'/'shots.json'
    write(shots_path, shots)
    out = tmp_path/'prepared'
    copied_rigs, copied_shots, summary = snapshot_runtime(source, shots_path, out)
    assert len(summary) == 4  # Two cameras, not three copies or one rig-derived state.
    assert copied_rigs['rigs'][0]['camera_ref'] == copied_shots['shots'][0]['camera_ref']
    assert copied_rigs['rigs'][1]['camera_ref'] is None
    assert copied_shots['shots'][0]['camera_ref'] != copied_shots['shots'][1]['camera_ref']
    restored = deepcopy(copied_shots)
    for old, new in zip(shots['shots'], restored['shots']):
        new['camera_ref'] = old['camera_ref']
    assert restored == shots
    shutil.rmtree(source.parent.parent)
    assert json.loads((out/'shots.json').read_text()) == copied_shots
    for row, (original, controls_raw) in zip(copied_shots['shots'], originals):
        path = out/row['camera_ref']
        camera = json.loads(path.read_text())
        assert (path.parent/camera['controls_ref']).read_bytes() == controls_raw
        camera['controls_ref'] = original['controls_ref']
        assert camera == original
    for item in summary:
        assert hashlib.sha256((out/item['snapshot_path']).read_bytes()).hexdigest() == item['sha256']


def test_same_rig_can_have_independent_shot_motion_controls(local_rigs, tmp_path):
    source, rigs, originals = local_rigs
    write(source, {'rigs': [rigs['rigs'][0]]})
    second = deepcopy(originals[0][0])
    second['shot_id'] = 'synthetic_second_shot'
    second['controls_ref'] = '../motion/second.npy'
    second['origin'] = {'kind': 'synthetic_unit_test'}
    second_path = source.parent.parent/'models'/'second.json'
    write(second_path, second)
    first_controls = source.parent.parent/'motion'/'s03.npy'
    second_controls = np.load(first_controls, allow_pickle=False).copy()
    second_controls[:, 0] += .001
    controls_path = second_path.parent/second['controls_ref']
    np.save(controls_path, second_controls, allow_pickle=False)
    expected_bytes = controls_path.read_bytes()
    shots_path = source.parent/'shots.json'
    write(shots_path, {'shots': [
        {'id': 's03', 'rig_id': second['rig_id'], 'camera_ref': '../models/s03.json'},
        {'id': second['shot_id'], 'rig_id': second['rig_id'], 'camera_ref': '../models/second.json'}]})
    out = tmp_path/'prepared'
    copied_rigs, copied_shots, summary = snapshot_runtime(source, shots_path, out)
    assert len(summary) == 4
    rows = copied_shots['shots']
    assert rows[0]['rig_id'] == rows[1]['rig_id']
    assert rows[0]['camera_ref'] == copied_rigs['rigs'][0]['camera_ref']
    assert rows[0]['camera_ref'] != rows[1]['camera_ref']
    copied_camera_path = out/rows[1]['camera_ref']
    copied_camera = json.loads(copied_camera_path.read_text())
    assert (copied_camera_path.parent/copied_camera['controls_ref']).read_bytes() == expected_bytes
    assert expected_bytes != originals[0][1]


@pytest.mark.parametrize('defect', ['missing_camera', 'wrong_shot'])
def test_shot_dependency_failure_precedes_all_snapshot_writes(local_rigs, tmp_path, defect):
    source, _, originals = local_rigs
    shots_path = source.parent/'shots.json'
    row = {'id': 's03', 'rig_id': originals[0][0]['rig_id'], 'camera_ref': '../models/s03.json'}
    if defect == 'missing_camera':
        row['camera_ref'] = '../models/absent.json'
    else:
        row['id'] = 'wrong_shot'
    write(shots_path, {'shots': [row]})
    out = tmp_path/'prepared'; out.mkdir()
    (out/'scene.json').write_bytes(b'preserved')
    error = FileNotFoundError if defect == 'missing_camera' else ValueError
    with pytest.raises(error):
        snapshot_runtime(source, shots_path, out)
    assert [(p.name, p.read_bytes()) for p in out.iterdir()] == [('scene.json', b'preserved')]


def test_existing_shots_registry_is_not_overwritten(local_rigs, tmp_path):
    source, _, _ = local_rigs
    shots_path = source.parent/'shots.json'
    write(shots_path, {'shots': []})
    out = tmp_path/'prepared'; out.mkdir()
    (out/'shots.json').write_bytes(b'keep')
    with pytest.raises(FileExistsError):
        snapshot_runtime(source, shots_path, out)
    assert [(p.name, p.read_bytes()) for p in out.iterdir()] == [('shots.json', b'keep')]
