"""Build current seven-field packets from reviewed independent source captures.

Dense source navigation poses initialize a 0.1-second cubic update space; they
are samples, never native spline coefficients. This module does not fit pixels.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re

import numpy as np

from rinkspace.domain.motion_chart import UniformMotionChart, cubic_samples
from rinkspace.domain.multi_shot_motion import continue_exposure_halos
from rinkspace.path_policy import guard_project_write

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
DT = .02
KNOT_STRIDE = 5
HALO = .4
DEFAULT_SHOT_IDS = ('s07', 's08', 's09')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def json_line(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'


def native_pts(window):
    expected = np.arange(window['first_pts'], window['stop_pts_exclusive'], 256, dtype=np.int64)
    if (not len(expected) or int(expected[-1]) != window['last_pts']
            or len(expected) != window['native_frame_count']):
        raise ValueError('Source scope has an inconsistent native exposure clock')
    return expected


def fit_pose_controls(pts, poses):
    """Least squares in the existing coarse cubic basis, followed by halo closure."""
    pts, poses = np.asarray(pts), np.asarray(poses)
    if pts.ndim != 1 or pts.dtype.kind not in 'iu' or len(pts) < 2:
        raise ValueError('Pose source clock must contain native integer exposures')
    if not np.all(np.diff(pts) == 256) or poses.shape != (len(pts), 3) or not np.isfinite(poses).all():
        raise ValueError('Invalid full-shot pan/tilt/logf samples')
    times = (pts.astype(np.int64)-int(pts[0]))/12800
    halo_count = round(HALO/DT)
    count = len(pts)+2*halo_count
    first = -halo_count*DT
    chart = UniformMotionChart.create(count, first, DT, KNOT_STRIDE, 0.)
    indices, weights = cubic_samples(count, first, DT, times)
    sample_matrix = np.einsum('ij,ijk->ik', weights, chart.refinement[indices])
    # Outside-halo coarse columns have exactly zero observations. Their values
    # are supplied by the explicit endpoint cubic continuation below.
    used = np.any(sample_matrix != 0., axis=0)
    coefficients = np.zeros((chart.refinement.shape[1], 3), dtype=np.float64)
    solution, _, rank, singular = np.linalg.lstsq(sample_matrix[:, used], poses, rcond=None)
    coefficients[used] = solution
    before = chart.refinement@coefficients
    controls = continue_exposure_halos(before, first, DT, [float(times[0]), float(times[-1])])
    evaluated = np.einsum('ij,ijk->ik', weights, controls[indices])
    unchanged = np.einsum('ij,ijk->ik', weights, before[indices])
    closure_delta = float(np.max(np.abs(evaluated-unchanged)))
    if closure_delta > 1e-10:
        raise RuntimeError('Halo continuation changed an observed source pose')
    closed_again = continue_exposure_halos(controls, first, DT, [float(times[0]), float(times[-1])])
    error = evaluated-poses
    return controls, dict(first_control_time=first, control_dt=DT, knot_stride=KNOT_STRIDE,
        motion_reference_time=0., reference_pts=int(pts[0]), halo_seconds=HALO,
        pose_samples=len(pts), native_controls=len(controls), coarse_coefficients=len(coefficients),
        observed_coarse_columns=int(used.sum()), least_squares_rank=int(rank),
        smallest_singular_value=float(singular[-1]) if len(singular) else None,
        channels=['pan_radians', 'tilt_radians', 'log_focal'],
        rms_by_channel=np.sqrt(np.mean(error**2, axis=0)).tolist(),
        max_abs_by_channel=np.max(np.abs(error), axis=0).tolist(),
        abs_error_p50_p95_p100_by_channel=np.percentile(np.abs(error), [50,95,100], axis=0).tolist(),
        frame_pose_delta_from_halo_closure=closure_delta,
        repeated_halo_closure_max_delta=float(np.max(np.abs(closed_again-controls))),
        dense_pose_samples_treated_as_controls=False,
        initialization_only=True, pixel_fit_executed=False)


def capture_rows(path, sid, expected_pts, source_row_offset=0):
    """Reject malformed qualified rows; preserve unqualified frame lines verbatim."""
    path = Path(path)
    raw = path.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    allowed_pts = set(map(int, expected_pts))
    points, locations, pending, frames, seen = [], [], [], [], set()
    for line_number, line in enumerate(raw.splitlines(keepends=True), 1):
        if not line.strip():
            raise ValueError(f'{path}:{line_number}: empty capture record')
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError(f'{path}:{line_number}: invalid capture JSON') from error
        if not isinstance(record, dict):
            raise ValueError(f'{path}:{line_number}: capture frame must be an object')
        pts = record.get('pts')
        if isinstance(pts, bool) or not isinstance(pts, int) or pts not in allowed_pts or pts in seen:
            raise ValueError(f'{path}:{line_number}: duplicate or non-native source PTS')
        seen.add(pts)
        if record.get('fit_eligible') is not True:
            pending.append(line)
            continue
        bgr_hash = record.get('source_BGR_sha256')
        if not isinstance(bgr_hash, str) or re.fullmatch(r'[0-9a-f]{64}', bgr_hash) is None:
            raise ValueError(f'{path}:{line_number}: qualified frame needs its source BGR SHA256')
        rows = record.get('rows')
        if not isinstance(rows, list):
            raise ValueError(f'{path}:{line_number}: qualified frame needs actual measurement rows')
        frames.append(dict(pts=pts, source_line=line_number, source_BGR_sha256=bgr_hash,
                           fit_eligible=True, measurement_rows=len(rows)))
        line_hash = hashlib.sha256(line).hexdigest()
        for row_index, row in enumerate(rows):
            context = f'{path}:{line_number}:row[{row_index}]'
            if not isinstance(row, dict) or not isinstance(row.get('path_id'), str) or not row['path_id']:
                raise ValueError(context+': qualified measurement needs a native path_id')
            uv = np.asarray(row.get('uv'), dtype=float)
            if (uv.shape != (2,) or not np.isfinite(uv).all()
                    or np.any(uv < 0) or np.any(uv > [1919,1079])):
                raise ValueError(context+': qualified source UV must be finite and inside the native image')
            if 'measurement_operator' not in row:
                raise ValueError(context+': qualified measurement needs its measurement_operator')
            # The operator and actual RGB-boundary diagnostics remain in the
            # immutable capture row. No navigation residual becomes a gate.
            identity = f'independent:{sid}:p{pts}:r{row_index}:{line_hash[:16]}'
            if len(identity) > 96:
                raise ValueError(context+': generated observation ID exceeds baseline U96')
            source_row = source_row_offset+len(points)
            points.append(dict(pts=pts, observation_id=identity, source_row=source_row,
                               path_id=row['path_id'], uv=uv.tolist()))
            locations.append(dict(observation_id=identity, source_row=source_row,
                packet_row=len(points)-1, pts=pts, path_id=row['path_id'], uv=uv.tolist(),
                source_BGR_sha256=bgr_hash, source_jsonl=str(path.resolve()),
                source_jsonl_sha256=source_hash, source_line=line_number,
                source_line_sha256=line_hash, row_index=row_index,
                original_measurement_record_preserved_at_source=True))
    if not points:
        raise ValueError(f'{sid}: no qualified actual source measurements; pending records remain in {path}')
    return dict(points=points, locations=locations, pending_bytes=b''.join(pending), frames=frames,
                source_file=str(path.resolve()), source_sha256=source_hash,
                source_frame_records=len(seen), qualified_frames=len(frames), pending_frames=len(pending))


def selected_shots(index, shot_ids=None):
    """Select registered bootstrap shots before opening any capture input."""
    requested = DEFAULT_SHOT_IDS if shot_ids is None else (
        [sid.strip() for sid in shot_ids.split(',')] if isinstance(shot_ids, str) else list(shot_ids))
    if (not requested or any(not isinstance(sid, str) or not sid for sid in requested)
            or len(set(requested)) != len(requested)):
        raise ValueError('Shot IDs must be nonempty and distinct')
    available = [shot['shot_id'] for shot in index['shots']]
    if len(set(available)) != len(available):
        raise ValueError('Bootstrap index has duplicate shot IDs')
    missing = [sid for sid in requested if sid not in available]
    if missing:
        raise ValueError('Selected shots missing from bootstrap index: '+', '.join(missing))
    return [shot for shot in index['shots'] if shot['shot_id'] in requested]


def capture_inputs(capture, shots):
    inputs = []
    for shot in shots:
        sid = shot['shot_id']; folder = capture/sid
        rows_path, poses_path = folder/'source_rows.jsonl', folder/'poses.npz'
        for path in (rows_path, poses_path):
            if not path.is_file():
                raise FileNotFoundError(sid+': missing selected-shot capture input: '+str(path))
        inputs.append((shot, rows_path, poses_path))
    return inputs


def metadata(root, shot_ids=None):
    index_path = root/'index.json'
    baseline_path = root.parent/'baseline.json'
    index = read(index_path)
    index = {**index, 'shots': selected_shots(index, shot_ids)}
    baseline = read(baseline_path)
    manifest_path = Path(baseline['inputs']['manifest']['path']).resolve(strict=True)
    if digest(manifest_path) != baseline['inputs']['manifest']['sha256']:
        raise ValueError('Frozen baseline camera assignment manifest changed')
    manifest = read(manifest_path)
    scope_path = Path(baseline['source_scope'])
    scope_path = (scope_path if scope_path.is_absolute() else PROJECT/scope_path).resolve(strict=True)
    scope = read(scope_path)
    source_hash = index['source']['actual_sha256']
    if source_hash != index['source']['sha256'] or source_hash != scope['source']['sha256']:
        raise ValueError('Source bootstrap and registered original video identities differ')
    for clock in (scope['source_clock'], index['source_clock']):
        if (clock['time_base_num'],clock['time_base_den'],clock['native_pts_step']) != (1,12800,256):
            raise ValueError('Independent capture native source clock differs')
    scope_shots = {s['id']:s for s in scope['shots']}
    camera_indices = {}
    for shot in manifest['shots']:
        rig_id = scope_shots[shot['id']]['rig_id']
        if rig_id in camera_indices and camera_indices[rig_id] != shot['camera_index']:
            raise ValueError('Frozen baseline rig assignment is inconsistent')
        camera_indices[rig_id] = shot['camera_index']
    for shot in index['shots']:
        registered = scope_shots[shot['shot_id']]
        if shot['processing'] != registered['processing'] or shot['known_rig_id'] != registered['rig_id']:
            raise ValueError('Bootstrap shot identity/range differs from current scope')
    return index, camera_indices, {str(p):digest(p) for p in
        (index_path,baseline_path,manifest_path,scope_path)}


def build(root, capture, output, shot_ids=None):
    root, capture, output = [Path(p).resolve() for p in (root,capture,output)]
    output = guard_project_write(output)
    if output.exists():
        raise FileExistsError('Preserve existing packet output: '+str(output))
    if output.is_relative_to(PROJECT.resolve()):
        raise ValueError('Captured point packets belong outside the OneDrive source tree')
    index, camera_indices, protected = metadata(root, shot_ids)
    selected_inputs = capture_inputs(capture, index['shots'])
    staged, source_row_offset = [], 0
    all_ids = set()
    for shot, rows_path, poses_path in selected_inputs:
        sid = shot['shot_id']
        pts_expected = native_pts(shot['processing'])
        rows = capture_rows(rows_path,sid,pts_expected,source_row_offset)
        source_row_offset += len(rows['points'])
        with np.load(poses_path,allow_pickle=False) as pose_file:
            pts, poses = pose_file['pts'], pose_file['poses']
        if pts.dtype.kind not in 'iu' or not np.array_equal(pts,pts_expected):
            raise ValueError(sid+': pose samples must cover every actual native source PTS')
        controls, pose_note = fit_pose_controls(pts,poses)
        paths = list(dict.fromkeys(p['path_id'] for p in rows['points']))
        lookup = {p:i for i,p in enumerate(paths)}
        z = dict(pts=np.array([p['pts'] for p in rows['points']],np.int64),
            observation_ids=np.array([p['observation_id'] for p in rows['points']],dtype='U96'),
            uv=np.array([p['uv'] for p in rows['points']],dtype=np.float64),
            kind=np.ones(len(rows['points']),np.int64), stations=np.full(len(rows['points']),.5),
            source_rows=np.array([p['source_row'] for p in rows['points']],np.int64),
            path_indices=np.array([lookup[p['path_id']] for p in rows['points']],np.int32))
        if any(p in all_ids for p in z['observation_ids']):
            raise ValueError('Independent capture observation identities collide globally')
        all_ids.update(z['observation_ids'])
        seed = dict(id=sid,camera_index=camera_indices[shot['known_rig_id']],
            reference_pts=int(pts[0]),time_base_num=1,time_base_den=12800,
            first_control_time=pose_note['first_control_time'],control_dt=DT,knot_stride=KNOT_STRIDE,
            motion_reference_time=0.,controls_ref='controls.npy',independent_source_capture=True,
            source_video_sha256=index['source']['actual_sha256'])
        protected[str(rows_path)] = rows['source_sha256']; protected[str(poses_path)] = digest(poses_path)
        staged.append(dict(sid=sid,points=z,paths=paths,rows=rows,seed=seed,
                           controls=controls,pose_note=pose_note,poses_path=str(poses_path)))
    output.mkdir(parents=True,exist_ok=False)
    summaries = []
    for packet in staged:
        sid=packet['sid']; folder=output/sid; folder.mkdir()
        z, rows = packet['points'],packet['rows']
        np.savez(folder/'point_observations.npz',**z)
        np.save(folder/'controls.npy',packet['controls'])
        write(folder/'shot_seed.json',packet['seed'])
        (folder/'source_row_manifest.jsonl').write_text(''.join(map(json_line,rows['locations'])),encoding='utf-8')
        (folder/'pending_source_rows.jsonl').write_bytes(rows['pending_bytes'])
        write(folder/'source_frames.json',rows['frames'])
        write(folder/'pose_initialization.json',packet['pose_note'])
        observed=len(np.unique(z['pts'])); expected=packet['pose_note']['pose_samples']
        mapping=dict(path_ids=packet['paths'],source_video_sha256=index['source']['actual_sha256'],
            pixel_domain='native_distorted',image_size=[1920,1080],
            point_input_sha256=digest(folder/'point_observations.npz'),
            source_row_manifest='source_row_manifest.jsonl', source_row_manifest_sha256=digest(folder/'source_row_manifest.jsonl'),
            source_capture_jsonl=rows['source_file'],source_capture_sha256=rows['source_sha256'],
            source_eligibility='Only frame records explicitly fit_eligible=true; every measurement UV strictly validated.',
            kind_policy='All measured source contours are kind=1; station=0.5 is an initializer for current native projection.',
            independent_source_capture=True,expected_exposures=expected,observed_exposures=observed,
            partial_source_coverage=observed!=expected)
        write(folder/'mapping.json',mapping)
        summaries.append(dict(shot_id=sid,packet=str(folder),rows=len(z['pts']),
            expected_exposures=expected,observed_exposures=observed,partial_source_coverage=observed!=expected,
            qualified_frames=rows['qualified_frames'],pending_frames=rows['pending_frames'],
            pending_path=str(folder/'pending_source_rows.jsonl'),
            pose_input=packet['poses_path'],pose_input_sha256=protected[packet['poses_path']],
            pose_initialization=packet['pose_note']))
    for path,sha in protected.items():
        if digest(path)!=sha:
            raise RuntimeError('Capture/bootstrap dependency changed during packet creation: '+path)
    receipt=dict(schema='rinkspace.independent_source_packets.v1',root=str(root),capture=str(capture),
        output=str(output),selected_shot_ids=[s['shot_id'] for s in index['shots']],
        shots=summaries,input_sha256=protected,source_video_sha256=index['source']['actual_sha256'],
        seven_unweighted_fields=True,all_kind1=True,source_measurements_and_pts_generated=False,
        unqualified_source_records_preserved_pending=True,qualified_invalid_rows_silently_skipped=False,
        legacy_new_shot_points_or_motion_loaded=False,prepare_executed=False,pixel_fit_executed=False)
    write(output/'packets_manifest.json',receipt)
    print(json_line(dict(output=str(output),shots=[p['sid'] for p in staged],rows=len(all_ids),
        partial_source_coverage=any(s['partial_source_coverage'] for s in summaries),pixel_fit_executed=False)),end='',flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','capture','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--shot-ids', help='Comma-separated bootstrap shot IDs; default: s07,s08,s09')
    args=parser.parse_args()
    build(args.root,args.capture,args.output,shot_ids=args.shot_ids)


if __name__=='__main__':
    main()
