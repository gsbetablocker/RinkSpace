"""Audit the complete Ankara review videos, source clocks and calibration files."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

import av
import numpy as np

from shared import (EVENT, OUT, PROGRAMMES, NativeCore, NativeSceneGPU, ParameterState,
                    SceneParameterModel, arrays, numeric_sha, programme_state_sha, read, save, sha, write)


PARENT = OUT / 'lane4_side_1118/verified'
UPDATE = OUT / 's14_boards_1118'
REPORT = UPDATE / 'four_check'
PARENT_NAME = PARENT.relative_to(OUT).as_posix()
TIME_BASE = Fraction(1, 12800)
EXPOSURE_STEP = 256
CAMERA_KEYS = ('pts', 'times', 'rigs', 'controls', 'active_folds')
SOURCE_COLUMNS = ('idx', 'pts', 'uv', 'path', 'source_row', 'source_packet', 'station_fixed')


def ref(path):
    return dict(path=str(path), sha256=sha(path))


@lru_cache(maxsize=None)
def digest(path):
    return sha(path)


def verified(record):
    path = Path(record['path']).resolve(strict=True)
    assert digest(path) == record['sha256'], ('Registered file hash changed', path)
    return path


def equal(a, b):
    return a.dtype == b.dtype and np.array_equal(a, b)


def packet_path(prepared, name, shot):
    local = prepared / (name + '.npz')
    return (local if local.is_file() else prepared / shot['observations']).resolve(strict=True)


def packet_rows(path, state, shot, name):
    pts = state[name + '_pts']
    with np.load(path, allow_pickle=False) as packet:
        idx, fold, uv = (packet[key] for key in ('idx', 'fold', 'uv'))
        assert ((idx >= 0) & (idx < len(pts))).all(), ('Observation outside shot', name, path)
        np.testing.assert_array_equal(packet['pts'], pts[idx], err_msg=name)
        if 'source_sha256' in packet:
            assert packet['source_sha256'].item() == shot['source_sha256'], (name, path)
        active = np.isin(fold, state[name + '_active_folds'])
        assert np.isfinite(uv[active]).all(), ('Non-finite active source point', name, path)
        selected = np.flatnonzero(active)
        selected = selected[np.argsort(idx[selected], kind='stable')]
        return np.bincount(idx[active], minlength=len(pts)), idx[selected], uv[selected], len(fold)


def capture_scope(programme, names, setup, state):
    path = EVENT / 'programmes' / PROGRAMMES[programme] / 'rebuild_scope.json'
    scope = read(path)
    assert names == [f'{programme}_{shot}' for shot in scope['required_shot_ids']], programme
    for shot in scope['shots']:
        name = programme + '_' + shot['id']
        if name not in names:
            continue
        definition = shot['processing']
        np.testing.assert_array_equal(state[name + '_pts'],
            np.arange(definition['first_pts'], definition['stop_pts_exclusive'], EXPOSURE_STEP), err_msg=name)
        assert setup[name]['source_sha256'] == scope['source']['sha256'], name
    window = scope['processing_scope']
    assert (state[names[0] + '_pts'][0], state[names[-1] + '_pts'][-1]) == (window['first_pts'], window['last_pts'])
    return ref(path)


def compare_export(path, names, setup, state):
    camera = arrays(path)
    for key in ('scene_values_sha256', 'layout_revision'):
        np.testing.assert_array_equal(state[key], camera[key], err_msg=key)
    for name in names:
        shot = setup[name]
        sid = shot['shot']
        if 'shot_ids' in camera:
            index = camera['shot_ids'].tolist().index(sid)
            lo, hi = camera['source_offsets'][index:index + 2]
            u, v = camera['control_offsets'][index:index + 2]
            data = [camera['source_pts'][lo:hi], camera['local_exposure_seconds'][lo:hi],
                    camera['rig_by_source_pts'][lo:hi], camera['controls'][u:v]]
            assert camera['camera_indices'][index] == shot['camera_index'], name
        else:
            pts = camera[sid + '_pts']
            times = camera.get(sid + '_exposure_times', (pts - pts[0]) / 12800.)
            data = [pts, times, np.broadcast_to(camera[sid + '_rig'], state[name + '_rigs'].shape),
                    camera[sid + '_controls']]
        for key, value in zip(CAMERA_KEYS, data + [camera['active_observation_folds']], strict=True):
            assert equal(state[f'{name}_{key}'], value), ('Exported camera differs', name, key)


def luminance(frame, header):
    if frame.format.name not in ('yuv420p', 'yuvj420p', 'nv12'):
        raise ValueError(('Unsupported decoded luma format', frame.format.name))
    plane = frame.planes[0]
    luma = np.ndarray((frame.height, plane.line_size), dtype=np.uint8, buffer=plane)
    return luma[header + 8:header + 135:4, 625:1575:4].astype(np.int16)


def decode_timeline(source, video, pts, target):
    """Match every decoded output frame with the exact original exposure."""
    differences = np.empty((len(pts), 2))
    with av.open(str(source)) as original, av.open(str(video)) as encoded:
        source_stream = original.streams.video[0]
        review_stream = encoded.streams.video[0]
        for stream, size in ((source_stream, (1920, 1080)), (review_stream, (2560, 1144))):
            if stream.time_base != TIME_BASE or (stream.width, stream.height) != size:
                raise ValueError(('Unexpected stream clock or dimensions', source, video, size))
            stream.thread_count = 2
            stream.thread_type = 'AUTO'
        if review_stream.average_rate != 50:
            raise ValueError(('Unexpected review frame rate', video, review_stream.average_rate))
        first = int(pts[0])
        original.seek(first, stream=source_stream, backward=True)
        originals = (frame for frame in original.decode(source_stream) if frame.pts >= first)
        reviews = encoded.decode(review_stream)
        for index, pt in enumerate(pts):
            raw = next(originals, None)
            projected = next(reviews, None)
            if raw is None or projected is None or raw.pts != pt or projected.pts != pt - first:
                raise ValueError(('Exposure PTS mismatch', video, int(pt),
                                  getattr(raw, 'pts', None), getattr(projected, 'pts', None)))
            delta = np.abs(luminance(raw, 0) - luminance(projected, 64))
            differences[index] = np.percentile(delta, [50, 90])
        if next(reviews, None) is not None:
            raise ValueError(('Extra encoded exposure', video))
    save(target.with_suffix('.npz'), source_pts=pts, median_luma_difference=differences[:, 0],
         p90_luma_difference=differences[:, 1])
    report = dict(source=ref(source), video=ref(video), frames=len(pts),
                  native_time_base=[1, 12800], source_first_pts=int(pts[0]),
                  source_last_pts=int(pts[-1]), region=dict(source_xyxy=[625, 8, 1575, 135],
                  stride=4, review_y_offset=64, units='8-bit decoded luma'),
                  quantiles=[50, 90, 99, 100],
                  median_luma=np.percentile(differences[:, 0], [50, 90, 99, 100]).tolist(),
                  p90_luma=np.percentile(differences[:, 1], [50, 90, 99, 100]).tolist(),
                  arrays=ref(target.with_suffix('.npz')))
    write(target.with_suffix('.json'), report)
    print('Checked decoded source and review timeline', video, len(pts), flush=True)
    return report


def baseline_timelines(programmes):
    setup = read(PARENT / 'setup.json')['shots']
    state = arrays(PARENT / 'state.npz')
    jobs = []
    for programme in programmes:
        names = [name for name, shot in setup.items() if shot['program'] == programme]
        pts = np.concatenate([state[name + '_pts'] for name in names])
        np.testing.assert_array_equal(np.diff(pts), EXPOSURE_STEP)
        coverage = read(PARENT / 'review' / programme / 'coverage.json')
        source = setup[names[0]]['source']
        jobs.append((source, coverage['video'], pts, REPORT / 'prior_decoded' / programme))
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        return list(executor.map(lambda args: decode_timeline(*args), jobs))


def verify_ledger(directory, names, state, fit_counts, coverage, world_vertices, previous=None):
    ledger = directory / 'source_frames.jsonl'
    report, rows = {}, []
    old_rows = iter(previous) if previous is not None else None
    with ledger.open(encoding='utf-8') as stream:
        for name in names:
            pts = state[name + '_pts']
            statuses = Counter()
            visible, misses = [], 0
            for pt, count in zip(pts, fit_counts[name], strict=True):
                row = json.loads(next(stream))
                assert (row['shot'], row['pts'], row['fit_points']) == (name, pt, count), (ledger, name, int(pt))
                status = row['projection_status']
                vis = row['visible_samples']
                assert 0 < vis <= status.get('0', 0) and sum(status.values()) == world_vertices, (name, int(pt))
                statuses.update(status)
                visible.append(vis)
                misses += not row['foreground_matte']
                rows.append(row)
                if old_rows is not None:
                    previous_row = next(old_rows)
                    if previous_row is not None:
                        for key in ('shot', 'pts', 'visible_samples', 'foreground_matte', 'projection_status'):
                            assert row[key] == previous_row[key], ('Frozen projection changed', name, int(pt), key)
            rendered = coverage['shots'][name]
            expected = dict(frames=len(pts), first_pts=int(pts[0]), last_pts=int(pts[-1]),
                            visible_samples_min=min(visible), projected_vertex_status=dict(statuses),
                            native_camera_status={'0': len(pts)},
                            frames_without_fit_points=int(np.count_nonzero(fit_counts[name] == 0)),
                            frames_without_foreground_matte=misses)
            assert all(rendered[k] == v for k, v in expected.items()), ('Coverage ledger mismatch', name, ledger)
            checks = rendered['cpu_gpu_checks_px']
            assert set(checks) == set(map(str, pts[[0, len(pts) // 2, -1]])), (name, list(checks))
            assert all(p['maximum_on_sensor_uv_error_px'] <= 1e-6 for p in checks.values()), name
            report[name] = dict(**expected, cpu_gpu_checks_px=checks)
        assert next(stream, None) is None, ('Extra source ledger rows', ledger)
    return dict(file=ref(ledger), shots=report), rows


def verify_review(directory, programme, names, state, packet_hashes, fit_counts, field, world, source, state_hash,
                  previous_rows=None, equivalent_render_packets=None):
    binding_file = directory / 'binding.json'
    coverage_file = directory / 'coverage.json'
    binding, coverage = read(binding_file), read(coverage_file)
    pts = np.concatenate([state[name + '_pts'] for name in names])
    np.testing.assert_array_equal(np.diff(pts), EXPOSURE_STEP)
    assert binding['state_sha256'] == state_hash and binding['programme'] == programme, binding_file
    assert digest(Path(binding['state']).resolve(strict=True)) == state_hash, binding_file
    assert binding['mode'] == 'complete' and binding['all_paths_drawn_without_error_filter'], binding_file
    assert binding['programme_state_sha256'] == programme_state_sha(state, names), binding_file
    assert binding['source_sha256'] == source['sha256'] and Path(binding['source']).samefile(source['path']), binding_file
    assert binding['scene_sha256'] == field['structure']['sha256'] and binding['inventory'] == world['inventory'], binding_file
    actual_packets = binding['observation_packets']
    assert set(actual_packets) == set(names), binding_file
    equivalent_render_packets = equivalent_render_packets or {}
    assert all(actual_packets[n] == packet_hashes[n] or
               (n in equivalent_render_packets and actual_packets[n] == equivalent_render_packets[n])
               for n in names), binding_file
    assert list(coverage['shots']) == names and coverage['validation']['native_PTS_equal'], coverage_file
    assert [binding['source_first_pts'], binding['source_last_pts']] == [pts[0], pts[-1]], binding_file
    assert coverage['source_seconds'] == (pts[[0, -1]] / 12800).tolist(), coverage_file
    assert len(pts) == binding['native_frames'] == coverage['frames'] == coverage['validation']['decoded_frames'], coverage_file
    ledger_directory = Path(binding.get('source_ledger_directory', directory))
    ledger, rows = verify_ledger(ledger_directory, names, state, fit_counts, coverage, len(world['xyz']), previous_rows)
    return dict(binding=ref(binding_file), coverage=ref(coverage_file), ledger=ledger, render_packet_sha256=actual_packets,
                frames=len(pts), source=source, source_seconds=coverage['source_seconds']), coverage, pts, rows


def check_old_decode(programme, registered, video, source, pts):
    prior_path = verified(registered)
    prior = read(prior_path)
    fresh_path = REPORT / 'prior_decoded' / (programme + '.json')
    if not fresh_path.is_file():
        decode_timeline(source['path'], video, pts, fresh_path)
    fresh = read(fresh_path)
    for record in (prior, fresh):
        assert record['frames'] == len(pts), programme
        assert record['source']['sha256'] == source['sha256'], programme
        assert record['video']['sha256'] == digest(Path(video).resolve(strict=True)), programme
    old = arrays(prior_path.with_suffix('.npz'))
    new = arrays(verified(fresh['arrays']))
    for key in ('source_pts', 'median_luma_difference', 'p90_luma_difference'):
        np.testing.assert_array_equal(old[key], new[key], err_msg=programme)
    np.testing.assert_array_equal(new['source_pts'], pts)
    return dict(registered=registered, rerun=ref(fresh_path), p90_luma=fresh['p90_luma'], exact_array_reproduction=True)


def verify_pan_revision(admission, prepared, updated, shot):
    from review import source_frames

    audit_file = Path(admission['pan_ice_role_changes'])
    audit = arrays(audit_file)
    assert audit['source_packet_sha256'].item() == digest(prepared)
    red_ref = admission['pan_red']
    red_file = Path(red_ref['packet'])
    assert red_file.samefile(shot['source_packets'][str(red_ref['code'])]['path'])
    red = arrays(red_file)
    rows = audit['row']
    promoted = (audit['previous'] == 6) & (audit['current'] == 0)
    retired = (audit['previous'] < 9) & (audit['current'] == 9)
    assert int(promoted.sum()) == admission['newly_developed_arc_rows']
    assert int(retired.sum()) == admission['prior_automatic_red_removed']
    assert red['source_sha256'].item() == shot['source_sha256']
    with np.load(prepared) as before, np.load(updated) as after:
        size = len(before['fold'])
        for key in SOURCE_COLUMNS:
            np.testing.assert_array_equal(before[key], after[key][:size], err_msg=key)
        roles_changed = before['fold'] != after['fold'][:size]
        types_changed = before['material'] != after['material'][:size]
        np.testing.assert_array_equal(np.flatnonzero(roles_changed | types_changed), rows)
        np.testing.assert_array_equal(before['fold'][rows], audit['previous'])
        np.testing.assert_array_equal(after['fold'][rows], audit['current'])
        for key in ('pts', 'uv'):
            np.testing.assert_array_equal(before[key][rows], audit[key])
        physical_rows = audit['physical_paint_rows']
        np.testing.assert_array_equal(np.flatnonzero(types_changed), physical_rows)
        for code in np.unique(after['source_packet'][physical_rows]):
            source_rows = physical_rows[after['source_packet'][physical_rows] == code]
            indices = after['source_row'][source_rows]
            original = arrays(shot['source_packets'][str(code)]['path'])
            np.testing.assert_array_equal(after['uv'][source_rows], original['uv'][indices])
            np.testing.assert_array_equal(after['path'][source_rows], original['keys'][original['path'][indices]])
        count = len(after['fold']) - size
        assert (after['source_packet'][size:] == red_ref['code']).all()
        ids = after['source_row'][size:]
        for key in ('idx', 'pts', 'uv', 'path', 'fold'):
            np.testing.assert_array_equal(after[key][size:], red[key][ids], err_msg=key)
        valid_red_pts = red['pts'][ids]
        red_roles = red['fold'][ids]
        assert red_ref['fit'] == int((red_roles < 5).sum()) and red_ref['check'] == int((red_roles >= 5).sum())
    for pts, raw in source_frames(shot['source'], np.unique(valid_red_pts)):
        expected = red['source_frame_BGR_sha256'][ids[valid_red_pts == pts]]
        assert np.all(expected == sha256(raw.tobytes()).hexdigest()), ('Red-paint source frame differs', int(pts))
    return dict(retired=int(retired.sum()), promoted_to_development=int(promoted.sum()),
                typed_rows=len(physical_rows), appended=count, audit=ref(audit_file),
                red_source=ref(red_file), red_source_exposures=len(np.unique(valid_red_pts)),
                red_checked_exposures=[int(pts) for pts in np.unique(valid_red_pts[red_roles >= 5])])


def circle_verdicts(parent, packet):
    """Resolve the C0 decisions back to every legacy held row in its source packet."""
    evidence = UPDATE / 'motion_check/circle_c0_source'
    registry = read(evidence / 'audit.json')
    assert parent.samefile(verified(registry['input_packet']))
    verified(registry['source_stripes'])
    table = evidence / 'held_rows.csv'
    with table.open(newline='', encoding='utf-8') as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: int(row['packet_row_zero_based']))
    index = np.asarray([int(row['packet_row_zero_based']) for row in rows])
    retained = np.asarray([row['C0_check_use'] == 'retain_C0_check' for row in rows])
    assert all(row['C0_check_use'].startswith('exclude_') for row, keep in zip(rows, retained, strict=True) if not keep)
    np.testing.assert_array_equal(retained, [row['classification'] == 'C0_verified_pair' for row in rows])
    fold = packet['fold']
    np.testing.assert_array_equal(index, np.flatnonzero((fold == 6) & (packet['family'] == 'C0')))
    np.testing.assert_array_equal(packet['source_row'][index], [int(row['source_stripe_row_zero_based']) for row in rows])
    np.testing.assert_array_equal(packet['pts'][index], [int(row['source_pts']) for row in rows])
    np.testing.assert_array_equal(packet['idx'][index], [int(row['idx']) for row in rows])
    np.testing.assert_array_equal(packet['uv'][index], [[float(row['source_u']), float(row['source_v'])] for row in rows])
    paired_source = evidence / 'c0_paired_source.csv'
    with paired_source.open(newline='', encoding='utf-8') as stream:
        separate_readings = sum(1 for _ in csv.DictReader(stream))
    return index[~retained], index[retained], dict(
        table=ref(table), source_audit=ref(evidence / 'audit.json'), evidence_checkpoint=registry['reviewed'],
        separate_source_readings=dict(rows=separate_readings, file=ref(paired_source)))


def verify_circle_revision(admission, parent, updated):
    audit_file = Path(admission['circle_roles'])
    audit = arrays(audit_file)
    assert audit['source_packet_sha256'].item() == digest(parent)
    reason_keys = ('admission_reason', 'restored_admission_reason')
    with np.load(parent) as before, np.load(updated) as after:
        excluded, retained, evidence = circle_verdicts(parent, before)
        assert set(before.files) == set(after.files), 'C0 changed packet schema'
        np.testing.assert_array_equal(audit['row'], excluded)
        np.testing.assert_array_equal(audit['pts'], before['pts'][excluded])
        np.testing.assert_array_equal(audit['uv'], before['uv'][excluded])
        fold = before['fold']
        expected = fold.copy()
        expected[excluded] = 9
        np.testing.assert_array_equal(audit['previous'], fold[excluded])
        np.testing.assert_array_equal(audit['current'], expected[excluded])
        assert after['fold'].dtype == fold.dtype
        np.testing.assert_array_equal(after['fold'], expected)
        reason_changed = np.zeros(len(expected), dtype=bool)
        for key in before.files:
            if key == 'fold':
                continue
            old_values, new_values = before[key], after[key]
            if key in reason_keys:
                assert old_values.shape == new_values.shape and new_values.dtype.kind == old_values.dtype.kind, key
                changed = old_values != new_values
                assert np.isin(np.flatnonzero(changed), excluded).all() and (new_values[changed] != '').all(), key
                reason_changed |= changed
            else:
                assert old_values.dtype == new_values.dtype, key
                np.testing.assert_array_equal(old_values, new_values, err_msg=key)
        np.testing.assert_array_equal(np.flatnonzero(reason_changed), excluded)
    return dict(parent=ref(parent), audit=ref(audit_file), rejected_rows=len(excluded),
                retained_fold6_rows=len(retained), from_fold=6, to_fold=9, **evidence)


def verify_role_revision(admission, old_hash, old_packet, new_packet, old_length, new_length, shot):
    assert admission['parent_sha256'] == old_hash == digest(Path(admission['parent']).resolve(strict=True))
    assert admission['source_sha256'] == shot['source_sha256']
    revision = arrays(admission['roles'])
    assert revision['parent_sha256'].item() == old_hash and len(revision['row']) == admission['retired']
    corrections = None
    pan_revision = None
    circle_revision = None
    pan_packet = new_packet
    if 'circle_roles' in admission:
        pan_packet = Path(admission['circle_parent']).resolve(strict=True)
        circle_revision = verify_circle_revision(admission, pan_packet, new_packet)
    revised_packet = pan_packet
    if 'corrected_roles' in admission:
        revised_packet = Path(admission['parent_packet']).resolve(strict=True)
        amended_file = Path(admission['corrected_roles'])
        amended = arrays(amended_file)
        assert amended['parent_packet_sha256'].item() == digest(revised_packet)
        corrected_packet = pan_packet
        if 'pan_ice_role_changes' in admission:
            corrected_packet = (amended_file.parent / new_packet.name).resolve(strict=True)
            pan_revision = verify_pan_revision(admission, corrected_packet, pan_packet, shot)
        with np.load(revised_packet) as before, np.load(corrected_packet) as after:
            amended_rows = amended['row']
            size = len(before['fold'])
            np.testing.assert_array_equal(np.flatnonzero(before['fold'] != after['fold'][:size]), np.sort(amended_rows))
            np.testing.assert_array_equal(before['fold'][amended_rows], amended['previous'])
            assert (after['fold'][amended_rows] == 9).all()
            for key in SOURCE_COLUMNS:
                np.testing.assert_array_equal(before[key], after[key][:size], err_msg=key)
            for key in ('pts', 'uv'):
                np.testing.assert_array_equal(before[key][amended_rows], amended[key], err_msg=key)
            corrections = dict(retired=len(amended_rows), appended=len(after['fold'])-size, audit=ref(amended_file))
    with np.load(old_packet) as before, np.load(revised_packet) as after:
        rows = np.flatnonzero(before['fold'] != after['fold'][:old_length])
        np.testing.assert_array_equal(rows, revision['row'])
        np.testing.assert_array_equal(before['fold'][rows], revision['before'])
        np.testing.assert_array_equal(after['fold'][rows], revision['after'])
        for key in SOURCE_COLUMNS:
            np.testing.assert_array_equal(before[key], after[key][:old_length], err_msg=key)
        for key in ('pts', 'uv'):
            np.testing.assert_array_equal(before[key][rows], revision[key], err_msg=key)
        assert (after['fold'][rows] == 9).all()
    return dict(retired=len(rows), total_appended=new_length-old_length, audit=ref(Path(admission['roles'])),
                corrections=corrections, pan=pan_revision, circle=circle_revision)


def new_optics(native, shot, name, state):
    # Use the release's nine sensor samples and per-row rolling-shutter clock.
    from flow_depths import rays

    samples = np.array([[x, y] for y in (0, 539.5, 1079) for x in (0, 959.5, 1919)])
    times = state[name + '_times']
    controls = state[name + '_controls']
    rigs, indices = np.unique(state[name + '_rigs'], axis=0, return_inverse=True)
    eigenvalues, distortion, roundtrip = [], [], []
    for i, rig in enumerate(rigs):
        exposure = times[indices == i]
        pixels = np.tile(samples, (len(exposure), 1))
        row_time = np.repeat(exposure, len(samples)) + rig[37] * (pixels[:, 1] / 1079. - .5)
        camera = native.core.evaluate_motion_camera(rig, controls, shot['first'], shot['dt'], row_time)
        ray = np.einsum('nij,nj->ni', camera['rotation'], rays(pixels, camera))
        x, y = (ray[:, :2] / ray[:, 2:]).T
        r2 = x*x+y*y
        k1, k2, p, q = (camera[key] for key in ('k1', 'k2', 'p1', 'p2'))
        slope = k1+2*k2*r2
        radial = 1+r2*(k1+r2*k2)
        a = radial+2*x*x*slope+2*p*y+6*q*x
        b = 2*x*y*slope+2*p*x+2*q*y
        c = radial+2*y*y*slope+6*p*y+2*q*x
        eig = (a+c-np.hypot(a-c, 2*b))/2
        origin = np.c_[camera['cx'], camera['cy']]
        focal = np.c_[camera['fx'], camera['fy']]
        undistorted = origin+focal*np.c_[x, y]
        reprojection = origin+focal*np.c_[x*radial+2*p*x*y+q*(r2+2*x*x), y*radial+p*(r2+2*y*y)+2*q*x*y]
        assert np.all((camera['status'] == 0) & np.isfinite(eig) & (eig > 0)), ('Invalid sensor Jacobian', name)
        assert np.isfinite(reprojection).all(), ('Invalid sensor projection', name)
        eigenvalues.append(float(eig.min()))
        distortion.append(float(np.linalg.norm(undistorted-pixels, axis=1).max()))
        roundtrip.append(float(np.linalg.norm(reprojection-pixels, axis=1).max()))
    return dict(checked=len(times)*len(samples), invalid=0, minimum_sampled_jacobian=min(eigenvalues),
                maximum_image_distortion_px=max(distortion), maximum_sensor_roundtrip_px=max(roundtrip))


def audit(prepared, candidate=None, require_media=False):
    from motion_delivery import camera_audit, media_check
    from motion_audit_1118 import trajectory
    from review import sample

    registry_path = EVENT / 'scene/lane4_side_1118.json'
    registry = read(registry_path)
    field = registry['geometry_binding']
    old_state_path = verified(registry['checkpoint'])
    previous = arrays(old_state_path)
    old_setup_file = verified(registry['fit_setup'])
    old_setup = read(old_setup_file)['shots']
    current_setup = read(prepared / 'setup.json')['shots']
    old_source = read(verified(registry['source_bindings']))
    prior_motion_file = verified(registry['motion'])
    prior_motion = read(prior_motion_file)
    prior_optics = read(verified(registry['optics']))
    prior_photo = read(verified(registry['original_photo_review']))
    verified(registry['original_photo_index'])
    dimensions = read(verified(registry['coordinates']))
    assert set(old_setup) == set(current_setup) == set(old_source), 'Incomplete candidate shot inventory'
    prior_hash = registry['checkpoint']['sha256']
    assert prior_motion['candidates'][PARENT_NAME]['sha256'] == prior_hash == prior_photo['state_sha256']
    old_mounts_file = PARENT / 'physical_mounts.json'
    old_mounts = read(old_mounts_file)
    assert old_mounts['state_sha256'] == prior_hash

    state_path = candidate / 'state.npz' if candidate else old_state_path
    state_hash = digest(state_path.resolve(strict=True))
    present = arrays(state_path) if candidate else previous
    assert set(previous) == set(present), 'Incomplete candidate state array inventory'
    for key in ('scene_values', 'scene_path', 'scene_values_sha256', 'layout_revision'):
        assert equal(present[key], previous[key]), ('Changed shared model', key)
    assert numeric_sha(present['scene_values']) == field['value_sha256'] == present['scene_values_sha256'].item()
    assert field['layout_revision'] == present['layout_revision'].item()
    scene_path = verified(field['structure'])
    assert Path(present['scene_path'].item()).samefile(scene_path)
    snapshot = arrays(verified(field['snapshot']))
    for key in ('scene_values', 'scene_values_sha256', 'layout_revision'):
        np.testing.assert_array_equal(present[key], snapshot[key], err_msg=key)
    model = SceneParameterModel(read(scene_path), scene_path, optimize_scene=True)
    native = NativeSceneGPU(scene_path, NativeCore(verified(field['native_library'])),
                            parameter_state=ParameterState(model.initial_state.layout, present['scene_values']))
    world = sample(native)
    inventory = world['inventory']
    assert field['path_count'] == len(inventory['path_ids']) == len(native.paths)
    assert field['advertisement_count'] == len(inventory['sampled_print_instance_ids']) and not native.unlocalized
    np.testing.assert_array_equal(native.scene['board'], dimensions['board'])
    baseline_file = (old_setup_file.parent / 'baseline.npz').resolve(strict=True)
    assert digest((prepared / 'baseline.npz').resolve(strict=True)) == digest(baseline_file)
    baseline = arrays(baseline_file)
    role_audit_file = prepared / 'source_audit.json'
    role_audit = read(role_audit_file)
    certified_media = read(verified(registry['media_verification']))
    record = dict(audit_script=ref(Path(__file__).resolve()), prior_registry=ref(registry_path),
                  prior_checkpoint=registry['checkpoint'], current_checkpoint=ref(state_path),
                  current_setup=ref(prepared / 'setup.json'), current_source_audit=ref(role_audit_file),
                  shared_field=dict(geometry=field, coordinates=registry['coordinates'], board=dimensions['board'],
                                    coordinate_interpretation=dimensions['coordinates'], sampled_vertices=len(world['xyz'])),
                  prior_evidence=dict(appearance=registry['original_photo_review'], optics=registry['optics'],
                                      motion=registry['motion'], mounts=ref(old_mounts_file), media=registry['media_verification']),
                  programmes={}, shots={})
    old_counts, new_counts, old_hashes, new_hashes = {}, {}, {}, {}
    equivalent_render_packets = {}
    for name, shot in current_setup.items():
        before = old_setup[name]
        previous_binding = old_source[name]
        for key in ('source', 'source_sha256', 'program', 'shot', 'first', 'dt', 'camera_index'):
            assert shot[key] == before[key], ('Changed source or camera clock', name, key)
        assert (previous_binding['source_sha256'], previous_binding['source']) == (before['source_sha256'], before['source'])
        old_packet = verified(previous_binding['packet'])
        assert packet_path(old_setup_file.parent, name, before) == old_packet, (name, old_packet)
        new_packet = packet_path(prepared, name, shot)
        old_hashes[name] = previous_binding['packet']['sha256']
        new_hashes[name] = digest(new_packet)
        camera_same = {key: equal(previous[f'{name}_{key}'], present[f'{name}_{key}']) for key in CAMERA_KEYS}
        pts = present[name + '_pts']
        assert camera_same['pts'] and camera_same['times'], ('Changed source clock', name)
        for key in ('pts', 'times'):
            assert equal(present[f'{name}_{key}'], baseline[f'{name}_{key}']), (name, key)
        old_counts[name], old_idx, old_uv, old_n = packet_rows(old_packet, previous, before, name)
        same_packet = new_hashes[name] == old_hashes[name]
        if same_packet and camera_same['active_folds']:
            new_counts[name], new_idx, new_uv, new_n = old_counts[name], old_idx, old_uv, old_n
        else:
            new_counts[name], new_idx, new_uv, new_n = packet_rows(new_packet, present, shot, name)
        changed_rows = None if same_packet else verify_role_revision(role_audit[name], old_hashes[name],
            old_packet, new_packet, old_n, new_n, shot)
        if changed_rows and changed_rows['circle'] is not None:
            circle = changed_rows['circle']
            active_folds = present[name + '_active_folds']
            assert not np.isin([circle['from_fold'], circle['to_fold']], active_folds).any(), name
            evidence_checkpoint = verified(circle['evidence_checkpoint'])
            if candidate is not None and circle['evidence_checkpoint']['sha256'] != state_hash:
                evidence_state = arrays(evidence_checkpoint)
                unchanged = ('scene_values', 'scene_path', 'layout_revision',
                             name + '_pts', name + '_times', name + '_active_folds')
                for key in unchanged:
                    assert equal(evidence_state[key], present[key]), ('C0 source binding differs', key)
            equivalent_render_packets[name] = circle['parent']['sha256']
        optics = prior_optics[name]['after']
        assert optics['checked'] == len(pts)*9 and optics['invalid'] == 0 and optics['minimum_sampled_jacobian'] > 0, name
        motion_shot = prior_motion['shots'][name]
        prior_trajectory = motion_shot[PARENT_NAME]
        assert motion_shot['source_sha256'] == before['source_sha256'], name
        motion_frames_path = prior_motion_file.parent / 'frames' / (name + '.npz')
        old_motion = arrays(motion_frames_path)
        np.testing.assert_array_equal(old_motion['pts'], pts, err_msg=name)
        np.testing.assert_array_equal(old_motion['1_local_exposure_seconds'], previous[name + '_times'], err_msg=name)
        np.testing.assert_array_equal(old_motion['1_fit_rows'], old_counts[name], err_msg=name)
        current_motion = trajectory(native, shot, name, present, pts)
        assert current_motion['valid_camera'].all() and (current_motion['camera_status'] == 0).all(), name
        mounts = camera_audit(native, present, name, shot)
        assert mounts['valid'] == len(pts) and mounts['pivot_range_L'] == [0., 0., 0.], name
        projection_same = all(camera_same[key] for key in CAMERA_KEYS if key != 'active_folds')
        trajectory_file = None
        current_optics = optics
        if projection_same:
            for key, values in current_motion.items():
                np.testing.assert_allclose(values, old_motion['1_' + key], rtol=1e-10, atol=1e-8, err_msg=f'{name} {key}')
        else:
            current_optics = new_optics(native, shot, name, present)
            path = REPORT / candidate.name / 'trajectories' / (name + '.npz')
            save(path, pts=pts, **current_motion)
            trajectory_file = ref(path)
        prior_pixel_file = verified(previous_binding['matched_errors'])
        pixel_stamp = prior_pixel_file.with_suffix('.json')
        pixel = read(pixel_stamp)
        assert pixel['state'] == programme_state_sha(previous, [name]), name
        assert pixel['packet_sha256'] == old_hashes[name] and pixel['frames'] == len(pts), name
        assert prior_photo['programmes'][shot['program']]['shots'][name]['matched_packet_sha256'] == old_hashes[name], name
        assert prior_trajectory['coverage']['camera_valid'] == len(pts), name
        record['shots'][name] = dict(
            frames=len(pts), source_pts=[int(pts[0]), int(pts[-1])], array_identical=camera_same,
            camera_projection_identical=projection_same, active_overlay_points_identical=equal(old_idx, new_idx) and equal(old_uv, new_uv),
            old_packet=previous_binding['packet'], current_packet=dict(path=str(new_packet), sha256=new_hashes[name]),
            packet_identical=same_packet, changed_rows=changed_rows, old_pixel_review=ref(pixel_stamp),
            prior_observation_errors=previous_binding['matched_errors'], prior_motion_arrays=ref(motion_frames_path),
            motion_reusable=projection_same, prior_motion_coverage={k: v for k, v in prior_trajectory['coverage'].items() if not k.endswith('_ranges')},
            local_exposure_clock=prior_trajectory['source_clock'],
            current_missing_overlay_frames=int(np.count_nonzero(new_counts[name] == 0)),
            current_mount_and_trajectory=mounts, current_trajectory_arrays=trajectory_file,
            prior_optics=optics, current_optics=current_optics, prior_optics_reusable=projection_same,
            legacy_mount_reference_unchanged=mounts['pivot'] == old_mounts['shots'][name]['pivot'])
    assert {n for n, row in record['shots'].items() if not row['packet_identical']} == set(role_audit)

    for programme in PROGRAMMES:
        names = [n for n, shot in current_setup.items() if shot['program'] == programme]
        first = current_setup[names[0]]
        source = dict(path=first['source'], sha256=first['source_sha256'])
        verified(source)
        scope = capture_scope(programme, names, current_setup, present)
        registration = registry['programmes'][programme]
        programme_registry = read(verified(registration))
        registered = programme_registry['full_review']
        compare_export(verified(programme_registry['camera']), names, old_setup, previous)
        assert programme_registry['fit_observations'] == {n: old_source[n]['packet'] for n in names}, programme
        verified(registered['coverage'])
        verified(registered['render_state'])
        video = verified(registered['video'])
        old_review, old_coverage, pts, rows = verify_review(PARENT / 'review' / programme, programme, names,
            previous, old_hashes, old_counts, field, world, source, prior_hash)
        assert Path(old_coverage['video']).samefile(video), programme
        old_review['video'] = registered['video']
        old_review['decode'] = check_old_decode(programme, certified_media[programme]['all_decoded_source_pictures'], video, source, pts)
        old_review['audio'] = media_check(video, old_coverage)
        models_equal = all(record['shots'][n]['camera_projection_identical'] for n in names)
        overlays_equal = all(record['shots'][n]['active_overlay_points_identical'] for n in names)
        state_equal = programme_state_sha(present, names) == programme_state_sha(previous, names)
        reusable = models_equal and state_equal and overlays_equal and all(record['shots'][n]['packet_identical'] for n in names)
        outcome = dict(scope=scope, programme_registration=registration, baseline_review=old_review,
                       camera_projection_identical=models_equal, active_overlay_points_identical=overlays_equal,
                       programme_state_identical=state_equal, reusable_prior_review=reusable)
        directory = candidate / 'review' / programme if candidate is not None else None
        if candidate is not None and reusable:
            outcome['current_review'] = dict(reused=True, video=registered['video'], frames=len(pts),
                                             source_seconds=old_coverage['source_seconds'])
            if (directory / 'coverage.json').is_file():
                rebound, coverage, _, _ = verify_review(directory, programme, names, present, new_hashes,
                    new_counts, field, world, source, state_hash, rows)
                assert digest(Path(coverage['video']).resolve(strict=True)) == registered['video']['sha256'], programme
                outcome['current_review']['registered_reuse'] = rebound
        elif candidate is not None:
            if (directory / 'coverage.json').is_file():
                compatible = [row if record['shots'][row['shot']]['camera_projection_identical'] else None for row in rows]
                report, coverage, pts, _ = verify_review(directory, programme, names, present, new_hashes,
                    new_counts, field, world, source, state_hash, compatible, equivalent_render_packets)
                assert coverage['source_audio'] == old_coverage['source_audio'], programme
                report['decode'] = decode_timeline(source['path'], coverage['video'], pts, REPORT / candidate.name / 'decoded' / programme)
                assert report['decode']['source']['sha256'] == source['sha256'], programme
                report['audio'] = media_check(coverage['video'], coverage)
                report['video'] = report['decode']['video']
                outcome['current_review'] = report
            elif require_media:
                raise FileNotFoundError(directory / 'coverage.json')
            else:
                outcome['current_review'] = dict(pending_review=str(directory))
        if candidate is not None:
            movies = sorted((candidate / 'videos').glob(programme + '_*.mp4'))
            current_review = outcome['current_review']
            if movies and 'video' in current_review:
                assert len(movies) == 1, ('Ambiguous review movie', movies)
                delivered = movies[0].resolve(strict=True)
                checked = current_review['video']
                if not delivered.samefile(checked['path']):
                    assert digest(delivered) == checked['sha256'], ('Different local review movie', delivered)
                outcome['delivery_video'] = dict(path=str(delivered), sha256=checked['sha256'])
        record['programmes'][programme] = outcome
        print(programme, len(pts), 'source exposures; reuse:', reusable, flush=True)
    records = list(record['shots'].values())
    record['totals'] = dict(shots=len(records), exposures=sum(r['frames'] for r in records),
                            changed_camera_shots=[n for n, r in record['shots'].items() if not r['camera_projection_identical']],
                            changed_source_packets=[n for n, r in record['shots'].items() if not r['packet_identical']],
                            lens_samples=sum(r['current_optics']['checked'] for r in records),
                            minimum_sampled_lens_jacobian=min(r['current_optics']['minimum_sampled_jacobian'] for r in records))
    assert sha(state_path) == state_hash, ('State changed during audit', state_path)
    file = REPORT / ((candidate.name + '.json') if candidate is not None else 'prior_and_inputs.json')
    write(file, record)
    print('Recorded', file, flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--prior-timelines', nargs='*', choices=PROGRAMMES)
    parser.add_argument('--prepared', type=Path, help='prepared directory; defaults to the candidate input record')
    parser.add_argument('--candidate', type=Path, help='candidate directory containing state.npz and review outputs')
    parser.add_argument('--require-media', action='store_true', help='require a complete review for changed programmes')
    args = parser.parse_args()
    if args.prior_timelines is not None:
        baseline_timelines(args.prior_timelines or list(PROGRAMMES))
    else:
        prepared = args.prepared
        if prepared is None:
            prepared = Path(read(args.candidate / 'inputs.json')['prepared']) if args.candidate else UPDATE / 'inputs'
        if args.require_media and not args.candidate:
            parser.error('--require-media also needs --candidate')
        audit(prepared, args.candidate, args.require_media)
