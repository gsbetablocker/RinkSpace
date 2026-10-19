"""Preserve a source-bound fitted programme trajectory as compact model data.

The original fitted NPY bytes are retained. Observations and diagnostics stay
in their Data archive; no fit, source decoding, filtering or contact inference
runs here. This uses the existing body_model_parameters_UNACCEPTED.npz format.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np


PARAMETER_KEYS = (
    'actual_pts', 'source_native_pts', 'P_world_m', 'Q_world', 'camera25',
    'valid', 'inferred', 'connect_from_previous', 'G_world_m', 'G_native_uv',
    'body_scale', 'metres_per_scene_unit', 'native_pts',
    'native_exposure_representative_pts', 'native_actual_row',
    'native_P_world_m', 'native_Q_world', 'native_G_world_m',
    'native_G_native_uv', 'native_valid', 'native_inferred', 'shot_index',
    'fit_time_seconds', 'edge_dt_seconds', 'edge_time_origin',
    'body_root_position_source_mode', 'local_pose_origin', 'source_hip_status',
    'source_hip_proxy_valid', 'body_local_pose_inferred',
    'body_geometry_source_fit_enabled', 'source_original_truncated',
    'unique_world_G_known', 'unique_world_G_m',
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size)


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _ranges(mask, pts, shot_index):
    """Describe every contiguous display interval without changing its flags."""
    mask, pts, shots = np.asarray(mask, bool), np.asarray(pts), np.asarray(shot_index)
    records, start = [], None
    for i in range(len(mask) + 1):
        split = (i == len(mask) or not mask[i]
                 or (start is not None and shots[i] != shots[i - 1]))
        if start is not None and split:
            records.append(dict(first_source_pts=int(pts[start]),
                last_source_pts_inclusive=int(pts[i - 1]),
                native_presentations=i - start, shot_index=int(shots[start])))
            start = None
        if i < len(mask) and mask[i] and start is None:
            start = i
    return records


def export_parameters(body_directory, state_identity_path, source_shots_path, output):
    directory, identity_path, shots_path, output = map(Path,
        (body_directory, state_identity_path, source_shots_path, output))
    identity, shots = read(identity_path), read(shots_path)
    report = read(directory / 'report.json')
    trace_path = directory / 'actual_body_root_fit.npz'
    camera_path = Path(identity['camera_path'])
    scene_path = identity_path.parent / 'scene_UNACCEPTED.json'
    if digest(trace_path) != report['output']['sha256']:
        raise ValueError('Original fitted trajectory changed')
    if (digest(camera_path) != identity['camera_sha256']
            or digest(scene_path) != identity['scene_sha256']):
        raise ValueError('Camera/scene differs from the complete model identity')
    body_binding = report['pelvis_source_root_refinement']
    for key, path in (('camera', camera_path), ('scene', scene_path)):
        if (Path(body_binding[key]['path']).resolve() != path.resolve()
                or body_binding[key]['sha256'] != digest(path)):
            raise ValueError('Trajectory was fitted against a different model')
    with np.load(trace_path, allow_pickle=False) as trace, np.load(camera_path, allow_pickle=False) as camera:
        data = {key: trace[key].copy() for key in PARAMETER_KEYS}
        meta = json.loads(str(camera['metadata_json']))
        native, actual, rows = data['native_pts'], data['actual_pts'], data['native_actual_row']
        scope = shots['analysis_scope']
        if (meta['source_sha256'] != shots['source_sha256_from_timeline']
                or meta['source_time_base'] != shots['source_time_base']):
            raise ValueError('Body/model/source-shot identity or clock differs')
        expected_native = np.arange(scope['start_pts'], scope['end_pts_exclusive'],
            scope['frame_step_pts'], dtype=np.int64)
        if not same(native, expected_native) or not same(native, camera['pts']):
            raise ValueError('Complete authorised original PTS coverage is required')
        if (np.any(np.diff(actual) <= 0) or np.any(rows < 0)
                or np.any(rows >= len(actual))
                or not same(actual[rows], data['native_exposure_representative_pts'])
                or not same(data['native_exposure_representative_pts'], camera['exposure_representative_pts'])
                or not same(data['camera25'][rows], camera['camera25'])):
            raise ValueError('Actual/native exposure aliases or fitted camera differ')
        for key in ('P_world_m', 'Q_world', 'G_world_m', 'G_native_uv', 'valid', 'inferred'):
            if not same(data[key][rows], data['native_' + key]):
                raise ValueError('Native output differs from its canonical body row: ' + key)
        if (not same(data['P_world_m'][:, :2], data['G_world_m'][:, :2])
                or np.any(data['G_world_m'][:, 2] != 0)
                or not same(trace['native22_world_m'][:, 0], data['P_world_m'])):
            raise ValueError('P joint0 / G world vertical projection differs')
        times = data['fit_time_seconds']
        if (not np.isfinite(times).all() or np.any(np.diff(times) < 0)
                or not np.allclose(np.diff(times), data['edge_dt_seconds'], rtol=0, atol=1e-12)):
            raise ValueError('Fitted exposure time basis differs from its saved edge ledger')
    target, record_path = output / 'body_model_parameters_UNACCEPTED.npz', output / 'binding.json'
    final_camera = output / camera_path.name
    final_scene = output / 'scene_UNACCEPTED.json'
    if any(path.exists() for path in (target, record_path, final_camera, final_scene)):
        raise FileExistsError('Preserve existing final body/camera/scene parameters and bindings')
    output.mkdir(parents=True, exist_ok=True)
    members = []
    with zipfile.ZipFile(trace_path) as source, zipfile.ZipFile(target, 'x',
            compression=zipfile.ZIP_DEFLATED) as destination:
        for key in PARAMETER_KEYS:
            payload = source.read(key + '.npy')
            destination.writestr(key + '.npy', payload)
            members.append(dict(name=key, bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest()))
    with np.load(target, allow_pickle=False) as copied:
        if any(not same(copied[key], value) for key, value in data.items()):
            raise ValueError('Saved compact parameter export changed an original fitted array')
    # These are final model parameters, not the large solver checkpoint or
    # source cache. Keep exact model bytes alongside the final body parameter
    # subset so downstream callers need not depend on an internal Data run.
    shutil.copyfile(camera_path, final_camera)
    shutil.copyfile(scene_path, final_scene)
    if (digest(final_camera) != identity['camera_sha256']
            or digest(final_scene) != identity['scene_sha256']):
        raise ValueError('Saved final camera/scene bytes differ from the authenticated model')
    unit = float(data['metres_per_scene_unit'])
    native_shots = data['shot_index'][rows]
    time_base = Fraction(shots['source_time_base'])
    record = dict(schema='rinkspace.programme-body-model-parameter-subset.v1',
        status='UNACCEPTED', accuracy_accepted=False,
        output=binding(target), original_fitted_body=binding(trace_path),
        original_body_report=binding(directory / 'report.json'),
        original_source_observations=binding(directory / 'fullprogramme_body_source_input.npz'),
        camera=binding(camera_path), scene=binding(scene_path),
        final_camera_parameters=binding(final_camera), final_scene=binding(final_scene),
        complete_state_identity=binding(identity_path), complete_q_sha256=identity['q_sha256'],
        source_shots=binding(shots_path), source_video_sha256=meta['source_sha256'],
        source_time_base=shots['source_time_base'],
        source_pts_first=int(native[0]), source_pts_last_inclusive=int(native[-1]),
        source_pts_end_exclusive=int(scope['end_pts_exclusive']),
        native_presentations=len(native), canonical_camera_exposures=len(actual),
        distinct_body_fit_times=len(np.unique(times)), shots=len(np.unique(native_shots)),
        source_presentation_elapsed_seconds=float((native[-1] - native[0]) * time_base),
        body_fit_elapsed_seconds=float(times[-1] - times[0]),
        time_semantics=dict(native_pts='Original video presentation PTS; retain repeated presentations',
            source_native_pts='Original pixel observation PTS of each canonical body row',
            native_actual_row='Native presentation to canonical camera/body exposure alias',
            fit_time_seconds='Retained fit basis from verified official exposures and explicit presentation approximations; distinct from original presentation elapsed time',
            edge_time_origin='Per-edge provenance of that retained fit basis',
            actual_speed_or_original_clock_derivative_accepted=False),
        P_semantics='Estimated SMPL-X native joint0 in fitted 3D body; conditional body geometry and calibration',
        G_semantics='G=(P.x,P.y,0), world vertical projection of the estimated P; not a bone-pixel ray intersection with ice or blade contact',
        source_hip_pixels_and_local_pose_are_in_original_source_observations=True,
        scale=dict(metres_per_scene_unit=unit,
            status='declared_uniform_working_scale_not_measured_calibration',
            scale_source=body_binding['scale'], coordinate_domain='conditional_euclidean_rink'),
        native_numeric_valid=int(data['native_valid'].sum()),
        canonical_inferred=int(data['inferred'].sum()),
        native_inferred=int(data['native_inferred'].sum()),
        inferred_native_intervals=_ranges(data['native_inferred'], native, native_shots),
        numeric_validity_is_not_world_accuracy=True,
        blade_contact=dict(left='not_estimated', right='not_estimated',
            airborne_vs_contact_state='unknown', missing_or_inferred_frames_are_not_contact=True),
        original_NPY_member_bytes_retained=True, members=members,
        refit_filter_pose_inference_source_decode_or_video_generation_run=False,
        code=binding(__file__))
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--body-directory', type=Path, required=True)
    parser.add_argument('--state-identity', type=Path, required=True)
    parser.add_argument('--source-shots', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record = export_parameters(args.body_directory, args.state_identity, args.source_shots, args.output)
    print(json.dumps({key: record[key] for key in ('output', 'complete_q_sha256',
        'native_presentations', 'canonical_inferred', 'status')}), flush=True)


if __name__ == '__main__':
    main()
