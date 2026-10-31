"""Pack qualified captures for existing prepared shots without camera initialization.

These packets are consumed by prepare_native_source_update, not the new-shot
bootstrap interface. Source eligibility must already have been decided.
"""
from pathlib import Path
import argparse
import json

import numpy as np

from build_independent_shot_packets import capture_rows, digest, json_line, write
from prepare_native_extension import lossless_cast, validate_points
from prepare_native_source_update import authenticate_packet, _source_hash
from rinkspace.path_policy import guard_project_write

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
SCOPE_PATH = HERE / 'rebuild_scope.json'


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=digest(path))


def selected_shots(manifest, requested):
    ids = [sid.strip() for sid in requested.split(',')] if isinstance(requested, str) else list(requested)
    if not ids or any(not isinstance(sid, str) or not sid for sid in ids) or len(set(ids)) != len(ids):
        raise ValueError('Shot IDs must be nonempty and distinct')
    available = [shot['id'] for shot in manifest['shots']]
    if len(set(available)) != len(available):
        raise ValueError('Current prepared manifest has duplicate shot IDs')
    missing = [sid for sid in ids if sid not in available]
    if missing:
        raise ValueError('Shots missing from current prepared manifest: ' + ', '.join(missing))
    return [shot for shot in manifest['shots'] if shot['id'] in ids]


def expected_pts(manifest, shot, scope):
    clock = scope['source_clock']
    step = clock['native_pts_step']
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError('Current scope needs a positive integer native PTS step')
    for key in ('time_base_num', 'time_base_den'):
        if shot[key] != clock[key]:
            raise ValueError('Prepared shot and current scope source time bases differ')
    if any(manifest.get('source_clock', {}).get(key, value) != value for key, value in clock.items()):
        raise ValueError('Prepared manifest and current scope source clocks differ')
    window = next(item for item in scope['shots'] if item['id'] == shot['id'])['processing']
    for key in ('first_pts', 'last_pts', 'stop_pts_exclusive'):
        if shot[key] != window[key]:
            raise ValueError('Prepared shot and current scope source ranges differ')
    pts = np.arange(shot['first_pts'], shot['stop_pts_exclusive'], step, dtype=np.int64)
    if not len(pts) or int(pts[-1]) != shot['last_pts'] or len(pts) != window['native_frame_count']:
        raise ValueError('Prepared shot has an inconsistent complete source clock')
    return pts


def source_frames(path):
    """Retain every frame location and reject implicit row eligibility."""
    frames = []
    with path.open('rb') as stream:
        for number, raw in enumerate(stream, 1):
            frame = json.loads(raw)
            if frame.get('fit_eligible') is True:
                if any(row.get('fit_eligible') is not True for row in frame['rows']):
                    raise ValueError(f'{path}:{number}: every qualified source row needs fit_eligible=true')
            frames.append(dict(pts=frame['pts'], source_line=number,
                source_BGR_sha256=frame.get('source_BGR_sha256'), fit_eligible=frame.get('fit_eligible') is True,
                measurement_rows=len(frame['rows']) if isinstance(frame.get('rows'), list) else None))
    return frames


def build(prepared, capture, output, shots):
    prepared, capture, output = (Path(path).resolve() for path in (prepared, capture, output))
    output = guard_project_write(output)
    if output.exists():
        raise FileExistsError('Preserve existing packet output: ' + str(output))
    if output.is_relative_to(PROJECT) or output.is_relative_to(prepared):
        raise ValueError('Point packets belong outside the source tree and current prepared directory')
    manifest_path, mapping_path = prepared / 'joint_manifest.json', prepared / 'mapping.json'
    point_path = prepared / 'point_observations.npz'
    manifest, current_mapping, scope = read(manifest_path), read(mapping_path), read(SCOPE_PATH)
    selected = selected_shots(manifest, shots)
    protected = {str(path): digest(path) for path in (manifest_path, mapping_path, point_path, SCOPE_PATH.resolve())}
    if protected[str(point_path)] != manifest['point_input_sha256']:
        raise ValueError('Current prepared point input digest differs')
    if current_mapping.get('point_input_sha256', manifest['point_input_sha256']) != manifest['point_input_sha256']:
        raise ValueError('Current mapping point input digest differs')
    with np.load(point_path, allow_pickle=False) as source:
        current = dict(source)
    validate_points(current, current_mapping)
    staged, offset = [], 0
    for shot in selected:
        sid = shot['id']
        path = capture / sid / 'source_rows.jsonl'
        if not path.is_file():
            raise FileNotFoundError(f'{sid}: missing source capture: {path}')
        pts = expected_pts(manifest, shot, scope)
        source_hash = _source_hash(manifest, current_mapping, sid)
        if source_hash != scope['source']['sha256']:
            raise ValueError('Current prepared and scope source video identities differ')
        rows = capture_rows(path, sid, pts, offset)
        if rows['source_frame_records'] != len(pts):
            raise ValueError(f'{sid}: incomplete source PTS records; every native exposure must be retained')
        frames = source_frames(path)
        paths = list(dict.fromkeys(point['path_id'] for point in rows['points']))
        indices = {name: i for i, name in enumerate(paths)}
        count = len(rows['points'])
        arrays = dict(pts=np.array([p['pts'] for p in rows['points']], dtype=np.int64),
            observation_ids=np.array([p['observation_id'] for p in rows['points']]),
            uv=np.array([p['uv'] for p in rows['points']], dtype=np.float64),
            kind=np.ones(count, dtype=np.int64), stations=np.full(count, .5),
            source_rows=np.array([p['source_row'] for p in rows['points']], dtype=np.int64),
            path_indices=np.array([indices[p['path_id']] for p in rows['points']], dtype=np.int64))
        arrays = {key: np.ascontiguousarray(lossless_cast(value, current[key].dtype, key))
                  for key, value in arrays.items()}
        protected[str(path)] = rows['source_sha256']
        replacement_path, pending_path = path.parent / 'source_replacement.json', path.parent / 'pending.jsonl'
        replacement = read(replacement_path) if replacement_path.is_file() else None
        pending = pending_path.read_bytes() if pending_path.is_file() else None
        for optional in (replacement_path, pending_path):
            if optional.is_file():
                protected[str(optional)] = digest(optional)
        staged.append((sid, arrays, paths, rows, frames, source_hash, replacement, pending, len(pts)))
        offset += count
    output.mkdir(parents=True)
    results = []
    for sid, arrays, paths, rows, frames, source_hash, replacement, pending, expected in staged:
        folder = output / sid
        folder.mkdir()
        np.savez(folder / 'point_observations.npz', **arrays)
        (folder / 'source_row_manifest.jsonl').write_text(''.join(map(json_line, rows['locations'])), encoding='utf8')
        (folder / 'pending_source_rows.jsonl').write_bytes(rows['pending_bytes'])
        write(folder / 'source_frames.json', frames)
        if pending is not None:
            (folder / 'pending.jsonl').write_bytes(pending)
        observed = len(np.unique(arrays['pts']))
        mapping = dict(shot_id=sid, path_ids=paths, source_video_sha256=source_hash,
            pixel_domain='native_distorted', image_size=[1920, 1080], independent_source_capture=True,
            fit_eligible=True, source_UV_synthesized=False, source_measurements_and_pts_generated=False,
            point_input_sha256=digest(folder / 'point_observations.npz'),
            source_row_manifest='source_row_manifest.jsonl', source_row_manifest_sha256=digest(folder / 'source_row_manifest.jsonl'),
            source_capture_jsonl=rows['source_file'], source_capture_sha256=rows['source_sha256'],
            expected_exposures=expected, observed_exposures=observed, source_frame_records=rows['source_frame_records'],
            partial_source_coverage=observed != expected, existing_joint_camera_and_controls_must_be_preserved=True,
            kind_policy='Actual measured source curves are kind=1; station=0.5 is only the nuisance initializer.')
        if replacement is not None:
            mapping['source_replacement'] = replacement
        write(folder / 'mapping.json', mapping)
        authenticated = authenticate_packet(folder, manifest, current_mapping, current)
        results.append(dict(shot=sid, packet=str(folder), rows=authenticated['authenticated_rows'],
            mapping=ref(folder / 'mapping.json'), source_UV_exact=True, source_frame_records_complete=True,
            expected_exposures=expected, observed_exposures=observed,
            pending_frame_records=rows['pending_frames'], pending_source_rows_preserved=pending is not None,
            authenticated_packet=True, authenticated_partial_span_groups=len(authenticated['replacement_spans'])))
        print(json_line(dict(shot=sid, rows=authenticated['authenticated_rows'], authenticated=True)), end='', flush=True)
    for path, sha in protected.items():
        if digest(path) != sha:
            raise RuntimeError('Current prepared/capture input changed while packaging: ' + path)
    validation = dict(schema='rinkspace.registered_source_packets.v1', prepared=str(prepared), capture=str(capture),
        output=str(output), shots=results, rows=sum(item['rows'] for item in results), input_sha256=protected,
        seven_unweighted_fields=True, point_dtypes={key: value.dtype.str for key, value in current.items()},
        source_rows_and_replacement_spans_authenticated=True, prepared_modified=False,
        source_measurements_and_pts_generated=False, source_eligibility_decisions_performed=False,
        camera_pose_or_controls_initialized=False, prepare_executed=False, pixel_fit_executed=False)
    write(output / 'validation.json', validation)
    print(json_line(dict(output=str(output), rows=validation['rows'], authenticated=True)), end='', flush=True)
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'capture', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--shots', required=True, help='Comma-separated IDs already present in current prepared.')
    args = parser.parse_args()
    build(args.prepared, args.capture, args.output, args.shots)


if __name__ == '__main__':
    main()
