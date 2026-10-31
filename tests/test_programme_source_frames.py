from copy import deepcopy
import hashlib
import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/
        'events/ankara_2026/programmes/okada_mei_sp'))
    return importlib.import_module('programme_source_frames')


def scope(sid):
    return dict(source=dict(sha256='video'), source_clock=dict(time_base_num=1,
        time_base_den=12800, native_pts_step=256), shots=[dict(id=sid,
        processing=dict(first_pts=256, last_pts=512, stop_pts_exclusive=768, native_frame_count=2),
        physical=dict(first_pts=0, last_pts=512, stop_pts_exclusive=768, native_frame_count=3))])


def test_s01_trims_physical_cache_before_selection_and_never_requires_board(adapter, monkeypatch):
    run = adapter.programme
    visibility = dict(schema='rinkspace.s01_existing_source_sam2_exclusion.v1', frames=[
        dict(pts=pt, mask_path=f'{pt}.png', mask_sha256=f'{pt}.png', source_video_sha256='video',
             authority=f'{pt}.json', authority_sha256=f'{pt}.json') for pt in (0, 256, 512)])
    reads = []
    def read(path):
        reads.append(Path(path).name)
        return deepcopy(scope('s01') if Path(path).name == 'rebuild_scope.json' else visibility)
    hashes = []
    def digest(path):
        hashes.append(str(path))
        return str(path)
    monkeypatch.setattr(run, 'read', read)
    monkeypatch.setattr(run, 'digest', digest)
    vis, frames = run.source_frames('s01', selected_pts={512})
    assert reads[0] == 'visibility.json'
    assert vis['source_video_sha256'] == 'video'
    assert [f['pts'] for f in frames] == [512]
    assert frames[0]['_surface_roles_available'] == ['unoccluded_source']
    assert hashes == ['512.png', '512.json']
    visibility['frames'].pop(0)
    with pytest.raises(ValueError, match='full physical S01'):
        run.source_frames('s01', selected_pts={512})


def test_s01_surfaces_explicitly_leave_board_and_ice_unavailable(adapter, monkeypatch):
    from rinkspace import vision_near_board_qualified
    image = np.zeros((2, 3, 3), np.uint8)
    mask = np.array([[0, 255, 0], [0, 0, 0]], np.uint8)
    graphics = np.array([[False, False, True], [False, False, False]])
    monkeypatch.setattr(adapter.programme.cv2, 'imread', lambda *a: mask)
    monkeypatch.setattr(vision_near_board_qualified, 'current_graphics', lambda image: (graphics, {}))
    frame = dict(mask_path='mask.png', _surface_roles_available=['unoccluded_source'],
        _surface_dependencies=[dict(path='mask.png', sha256='mask')])
    result, dependencies = adapter.programme.surfaces(frame, image)
    np.testing.assert_array_equal(result['unoccluded_source'], (mask == 0) & ~graphics)
    assert result['ice'] is result['board'] is None
    assert result['surface_roles_available'] == ['unoccluded_source']
    assert dependencies == frame['_surface_dependencies']


def test_cached_adapter_returns_original_BGR_and_actual_exclusion(adapter, monkeypatch, tmp_path):
    image = np.zeros((1080, 1920, 3), np.uint8)
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    frame = dict(pts=512, image='original.png', source_BGR_sha256=digest,
        source_video_sha256='video', mask_path='mask.png')
    selected = []
    def cached(sid, *, selected_pts):
        selected.append((sid, selected_pts))
        return dict(source_video_sha256='video'), [frame]
    monkeypatch.setattr(adapter.programme, 'read', lambda p: scope('s02'))
    monkeypatch.setattr(adapter.programme, 'source_frames', cached)
    monkeypatch.setattr(adapter.cv2, 'imread', lambda *a: image)
    visible = np.ones(image.shape[:2], bool); visible[0, 0] = False
    monkeypatch.setattr(adapter.programme, 'surfaces', lambda *a: (
        dict(unoccluded_source=visible), [dict(path='mask.png', sha256='mask')]))
    monkeypatch.setattr(adapter, '_digest', lambda p: 'file')
    result, = adapter.iter_registration_source_frames('s02', bootstrap_root=tmp_path, selected_pts={512})
    assert selected == [('s02', {512})]
    assert result['image'] is image and result['pts'] == 512
    np.testing.assert_array_equal(result['source_exclusion_mask'], ~visible)
    assert result['source_exclusion_sha256'] == hashlib.sha256((~visible).tobytes()).hexdigest()
    frame['source_BGR_sha256'] = 'changed'
    with pytest.raises(ValueError, match='BGR pixels changed'):
        list(adapter.iter_registration_source_frames('s02', bootstrap_root=tmp_path, selected_pts={512}))


@pytest.mark.parametrize('failure', [None, 'rgb', 'scope', 'short_decode'])
def test_bootstrap_adapter_authenticates_same_RGB_mask_and_full_scope(
        adapter, monkeypatch, tmp_path, failure):
    import capture_independent_ice as ice
    import capture_independent_materials as materials
    image = np.zeros((1080, 1920, 3), np.uint8)
    rgb_sha = hashlib.sha256(image.tobytes()).hexdigest()
    current = scope('s07')
    source = dict(sha256='video', actual_sha256='video', width=1920, height=1080,
        resolved_path=str(tmp_path/'original.mp4'))
    index = dict(source=source, source_clock=current['source_clock'],
        shots=[dict(shot_id='s07', processing=deepcopy(current['shots'][0]['processing']))])
    if failure == 'scope': index['shots'][0]['processing']['native_frame_count'] = 1
    monkeypatch.setattr(adapter.programme, 'read', lambda p:
        current if Path(p).name == 'rebuild_scope.json' else index)
    calls = []
    def capture(index, shot, selected):
        calls.append(selected)
        if failure != 'short_decode': yield 1, (512, image)
    monkeypatch.setattr(ice, 'capture_frames', capture)
    mask_path = tmp_path/'person_masks/s07/512.png'
    exclusion = np.zeros(image.shape[:2], bool)
    monkeypatch.setattr(materials, 'source_exclusion', lambda *a: (exclusion, mask_path))
    record = dict(source_binding=dict(source_sha256='video', source_pts=512, shot_id='s07',
        time_base='1/12800', native_hw=[1080, 1920]), mask_path=str(mask_path), mask_sha256='mask',
        decoded_native_RGB_sha256='0'*64 if failure == 'rgb' else rgb_sha)
    monkeypatch.setattr(materials, '_source_mask_document', lambda p: ({('s07', 512): record}, 'ledger'))
    monkeypatch.setattr(adapter, '_digest', lambda p:
        'video' if Path(p).suffix == '.mp4' else 'mask')
    if failure:
        with pytest.raises(ValueError, match={'rgb': 'exact original RGB/clock',
                'scope': 'complete registered shot', 'short_decode': 'Incomplete original source decode'}[failure]):
            list(adapter.iter_registration_source_frames('s07', bootstrap_root=tmp_path, selected_pts={512}))
    else:
        result, = adapter.iter_registration_source_frames('s07', bootstrap_root=tmp_path, selected_pts={512})
        assert result['image'] is image and result['source_RGB_sha256'] == rgb_sha
        assert calls == [{512}]
        assert result['mask_sources'][0]['role'] == 'same_exposure_mask_ledger'
