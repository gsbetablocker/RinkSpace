"""Native original RGB and same-exposure exclusions for print registration.

This adapter only authenticates existing source frames and masks. It does not
project geometry, identify material, create observations, or initialize a GPU.
"""
from pathlib import Path
import hashlib

import cv2
import numpy as np

import run_native_programme as programme


_FILE_HASHES = {}


def _digest(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    cached = _FILE_HASHES.get(path)
    if cached is None or cached[0] != signature:
        cached = (signature, programme.digest(path))
        _FILE_HASHES[path] = cached
    return cached[1]


def _reference(path, role):
    return dict(path=str(Path(path).resolve()), sha256=_digest(path), role=role)


def _image_hash(image):
    if image is None or image.shape != (1080, 1920, 3) or image.dtype != np.uint8:
        raise ValueError('Original source must be full native 1920x1080 uint8 BGR')
    return hashlib.sha256(image.tobytes()).hexdigest()


def _result(shot_id, pts, image, bgr_sha, source_sha, exclusion, mask_path, sources, **extra):
    exclusion = np.asarray(exclusion)
    if exclusion.dtype != np.bool_ or exclusion.shape != image.shape[:2]:
        raise ValueError('Same-exposure exclusion must be a native boolean mask')
    return dict(shot_id=shot_id, pts=int(pts), image=image,
        source_BGR_sha256=bgr_sha, source_video_sha256=source_sha, time_base='1/12800',
        pixel_domain='native_distorted', source_exclusion_mask=exclusion,
        source_exclusion_sha256=hashlib.sha256(exclusion.tobytes()).hexdigest(),
        mask_path=str(mask_path), mask_sha256=_digest(mask_path), mask_sources=sources,
        surface_roles_available=['unoccluded_source'], **extra)


def iter_registration_source_frames(shot_id, *, bootstrap_root, selected_pts=None):
    """Yield full BGR arrays, native PTS/hash bindings and actual source masks.

    ``source_exclusion_mask`` is boolean with True meaning excluded. Its hash
    describes the applied mask, while ``mask_sha256`` describes the referenced
    original PNG. Full clock scope is checked before selecting a pilot subset;
    source dependencies outside that subset are not read or hashed.
    """
    scope = programme.read(programme.HERE/'rebuild_scope.json')
    shot_scope = next((s for s in scope['shots'] if s['id'] == shot_id), None)
    if shot_scope is None:
        raise ValueError('Unknown registered programme shot: '+shot_id)
    clock = scope['source_clock']
    if (clock['time_base_num'], clock['time_base_den'], clock['native_pts_step']) != (1, 12800, 256):
        raise ValueError('Registered programme source clock changed')
    source_sha = scope['source']['sha256']
    window = shot_scope['processing']
    expected = list(range(window['first_pts'], window['stop_pts_exclusive'], 256))
    if len(expected) != window['native_frame_count'] or expected[-1] != window['last_pts']:
        raise ValueError('Registered programme shot scope is incomplete')
    selected = None if selected_pts is None else set(selected_pts)
    if selected is not None:
        inside = {pt for pt in selected if window['first_pts'] <= pt <= window['last_pts']}
        if not inside <= set(expected):
            raise ValueError('Selected PTS must be actual native source exposures')
        expected = [pt for pt in expected if pt in inside]

    if shot_id in ('s01', 's02', 's03', 's04', 's05', 's06'):
        visibility, records = programme.source_frames(shot_id, selected_pts=selected)
        if [f['pts'] for f in records] != expected:
            raise ValueError('Cached source frames differ from the registered selection')
        if visibility.get('source_video_sha256', source_sha) != source_sha:
            raise ValueError('Cached source belongs to another original video')
        adapter = _reference(programme.__file__, 'existing_source_surface_adapter')
        for frame in records:
            if frame.get('source_video_sha256', source_sha) != source_sha:
                raise ValueError('Cached frame belongs to another original video')
            image = cv2.imread(frame['image'], cv2.IMREAD_COLOR)
            bgr_sha = _image_hash(image)
            if bgr_sha != frame['source_BGR_sha256']:
                raise ValueError('Cached original BGR pixels changed')
            surfaces, dependencies = programme.surfaces(frame, image)
            available = surfaces.get('unoccluded_source')
            if available is None:
                raise ValueError('Current source exclusion surface is unavailable')
            yield _result(shot_id, frame['pts'], image, bgr_sha, source_sha,
                ~available, frame['mask_path'], [*dependencies, adapter], image_path=frame['image'])
        return

    from capture_independent_ice import capture_frames
    from capture_independent_materials import (source_exclusion, _source_mask_document,
        _source_mask_record)

    root = Path(bootstrap_root)
    index = programme.read(root/'index.json')
    source = index['source']
    index_clock = index['source_clock']
    if (source.get('sha256'), source.get('actual_sha256')) != (source_sha, source_sha):
        raise ValueError('Bootstrap source belongs to another original video')
    if (source['width'], source['height']) != (1920, 1080):
        raise ValueError('Bootstrap source dimensions changed')
    if (index_clock['time_base_num'], index_clock['time_base_den'], index_clock['native_pts_step']) != (1, 12800, 256):
        raise ValueError('Bootstrap source clock changed')
    shot = next(s for s in index['shots'] if s['shot_id'] == shot_id)
    if any(shot['processing'][key] != window[key] for key in
           ('first_pts', 'last_pts', 'stop_pts_exclusive', 'native_frame_count')):
        raise ValueError('Bootstrap source scope differs from the complete registered shot')
    if not expected:
        return
    if _digest(source['resolved_path']) != source_sha:
        raise ValueError('Original source video bytes changed')
    adapter = _reference(source_exclusion.__code__.co_filename, 'existing_same_exposure_mask_adapter')
    count = 0
    for _, (pts, image) in capture_frames(index, shot, selected):
        if count >= len(expected) or pts != expected[count]:
            raise ValueError('Decoded source frames differ from the registered selection')
        bgr_sha = _image_hash(image)
        exclusion, mask_path = source_exclusion(root, shot_id, pts)
        refined = root/'person_masks_refined/full'/shot_id/f'{pts}.png'
        ledger = root/('person_masks_refined/full/frames.jsonl' if refined.exists()
                      else 'person_masks/frames.jsonl')
        records, ledger_sha = _source_mask_document(ledger)
        record = _source_mask_record(records, shot_id, pts)
        binding = record['source_binding']
        rgb_sha = hashlib.sha256(np.ascontiguousarray(image[:, :, ::-1]).tobytes()).hexdigest()
        if (binding['source_sha256'] != source_sha or binding['time_base'] != '1/12800'
                or record['decoded_native_RGB_sha256'] != rgb_sha):
            raise ValueError('Source mask does not belong to this exact original RGB/clock')
        raw_mask = Path(record['mask_path'])
        if not raw_mask.is_absolute(): raw_mask = ledger.parent/raw_mask
        if _digest(raw_mask) != record['mask_sha256']:
            raise ValueError('Original same-exposure source mask bytes changed')
        sources = [dict(path=str(ledger.resolve()), sha256=ledger_sha, role='same_exposure_mask_ledger'),
            _reference(raw_mask, 'original_same_exposure_person_mask'), adapter]
        if Path(mask_path).resolve() != raw_mask.resolve():
            guard_binding_path = root/'person_mask_guard_binding.json'
            guard_binding, _ = _source_mask_document(guard_binding_path)
            sources.append(_reference(guard_binding_path, 'same_RGB_guard_binding'))
            for name in ('base_ledger', 'refined_ledger', 'guard_ledger'):
                path = Path(guard_binding[name]['path'])
                if not path.is_absolute(): path = root/path
                sources.append(_reference(path, name))
            sources.append(_reference(mask_path, 'actual_same_RGB_guard_PNG'))
        yield _result(shot_id, pts, image, bgr_sha, source_sha, exclusion,
            mask_path, sources, source_RGB_sha256=rgb_sha, image_path=None)
        count += 1
    if count != len(expected):
        raise ValueError('Incomplete original source decode')
