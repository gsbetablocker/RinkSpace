"""Plan or export a complete saved-model review through the existing public CLIs.

The programme profile owns source/shot facts. Body fitting happens once, then
one two-worker renderer runs at a time. --single-pass decodes all shots once.
No video is rendered without --execute.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('D:/CodexData/RinkSpace_Data').resolve()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def same_input(row, expected):
    return Path(row['path']).resolve() == Path(expected['path']).resolve() and row['sha256'] == expected['sha256']


def prepare(state_directory, output, profile_path, previous_body=None, current_source_points=None,
            single_pass=False):
    state_directory, output = state_directory.resolve(), output.resolve()
    if not output.is_relative_to(DATA):
        raise ValueError('Review output must be inside the existing RinkSpace Data root')
    profile = read(profile_path)
    identity = read(state_directory / 'state_identity.json')
    camera = Path(identity['camera_path']).resolve()
    scene = state_directory / 'scene_UNACCEPTED.json'
    if sha(camera) != identity['camera_sha256'] or sha(scene) != identity['scene_sha256']:
        raise ValueError('Saved camera/scene differs from this complete state identity')
    from calibration.program_tracking.current_source_points import CurrentSourcePoints
    point_reference = identity.get('current_source_points')
    point_path = Path(current_source_points or (point_reference or {}).get('path',
                      state_directory / 'current_active_source_points.npz')).resolve()
    if not point_path.is_file():
        raise FileNotFoundError('Complete review requires current active source points for this saved model: ' + str(point_path))
    if point_reference is not None and current_source_points is None and sha(point_path) != point_reference['sha256']:
        raise ValueError('Saved current source points changed')
    point_view = CurrentSourcePoints(point_path, camera, q_sha256=identity['q_sha256'])
    if point_view.metadata['input_manifest']['sha256'] != identity['input_manifest_sha256']:
        raise ValueError('Current source points belong to different fitting inputs than the saved model')
    with np.load(state_directory / 'complete_state_UNACCEPTED.npz', allow_pickle=False) as archive:
        if hashlib.sha256(archive['q'].tobytes()).hexdigest() != identity['q_sha256']:
            raise ValueError('Complete q identity differs')
    with np.load(camera, allow_pickle=False) as archive:
        native = archive['pts'].astype(np.int64)
        actual = archive['exposure_representative_pts'].astype(np.int64)
        metadata = json.loads(str(archive['metadata_json']))
    source = Path(profile['source']).resolve()
    shots_path = Path(profile['source_shots']['path']).resolve()
    if not source.is_file() or sha(shots_path) != profile['source_shots']['sha256']:
        raise ValueError('Registered source or authenticated source-shot packet missing/changed')
    shots = read(shots_path)
    if not (shots['source_sha256_from_timeline'] == metadata['source_sha256'] == profile['source_video_sha256']
            and shots['source_time_base'] == metadata['source_time_base'] == profile['source_time_base']):
        raise ValueError('Programme shots and cameras must use the same original source clock')
    if (len(native) != profile['native_presentations'] or len(np.unique(actual)) != profile['actual_exposures']
            or np.any(np.diff(native) <= 0)):
        raise ValueError('Complete original programme presentation/exposure coverage changed')
    previous = Path(previous_body or profile['previous_body_directory']).resolve()
    for name in ('report.json', 'camera_rebind_prepare.json', 'fullprogramme_body_source_input.npz', 'actual_body_root_fit.npz'):
        if not (previous / name).is_file():
            raise FileNotFoundError(previous / name)
    body_directory = output / 'body'
    body = body_directory / 'actual_body_root_fit.npz'
    body_command = [sys.executable, '-B', '-X', 'utf8', '-m',
                    'calibration.program_tracking.body_tracking.pelvis_source_root_refit',
                    '--previous-body-directory', str(previous), '--camera', str(camera),
                    '--scene', str(scene), '--output', str(body_directory)]
    for key, value in profile.get('body_options', {}).items():
        body_command.extend([key, str(value)])
    render_base = [sys.executable, '-B', '-X', 'utf8', '-m',
                   'calibration.program_tracking.render_programme_root_trace',
                   '--camera', str(camera), '--body', str(body), '--scene', str(scene),
                   '--current-source-points', str(point_path),
                   '--source', str(source), '--no-display-camera-smoothing', '--video',
                   '--draw-workers', '2', '--draw-slots', '8', '--draw-chunk-size', '2']
    segments, joined = [], []
    for index, shot in enumerate(shots['shots'], 1):
        selected = native[(native >= shot['start_pts']) & (native < shot['end_pts'])]
        if not len(selected):
            continue
        directory = output / f'shot_{index:02d}'
        # Source shots use an exclusive end. The public renderer is inclusive.
        first, last = int(selected[0]), int(selected[-1])
        command = render_base + ['--output', str(directory), '--start-pts', str(first), '--end-pts', str(last)]
        segments.append(dict(shot_index=index, directory=str(directory), native_frames=len(selected),
            source_start_pts=first, source_last_pts_inclusive=last,
            source_shot_end_pts_exclusive=int(shot['end_pts']),
            native_pts_sha256=hashlib.sha256(selected.tobytes()).hexdigest(), command=command))
        joined.append(selected)
    if len(segments) != profile['shots'] or not np.array_equal(np.concatenate(joined), native):
        raise ValueError('Exact complete camera PTS must be covered once in source-shot order')
    point_coverage = None
    if single_pass:
        rows = np.searchsorted(point_view.pts, actual)
        markers = np.diff(point_view.offsets)[rows]
        triples = np.column_stack((native, actual, markers)).astype(np.int64)
        point_coverage = dict(native_presentations=len(native),
            actual_exposures=len(np.unique(actual)),
            native_presentations_with_points=int((markers > 0).sum()),
            native_presentations_without_points=int((markers == 0).sum()),
            first_native_pts=int(native[0]), first_actual_pts=int(actual[0]),
            first_frame_points=int(markers[0]), last_native_pts=int(native[-1]),
            last_actual_pts=int(actual[-1]), last_frame_points=int(markers[-1]),
            maximum_frame_points=int(markers.max()),
            maximum_frame_points_native_pts=int(native[np.argmax(markers)]),
            native_actual_marker_count_rows_sha256=hashlib.sha256(triples.tobytes()).hexdigest())
    return dict(schema='rinkspace.complete-programme-sequential-review-export.v1',
        state_directory=str(state_directory), q_sha256=identity['q_sha256'],
        state_identity=reference(state_directory / 'state_identity.json'),
        camera=reference(camera), scene=reference(scene), profile=reference(profile_path),
        current_source_points=point_view.reference,
        source=str(source), source_video_sha256=metadata['source_sha256'],
        source_shots=reference(shots_path), previous_body_directory=str(previous),
        body_directory=str(body_directory), body_command=body_command, output=str(output),
        native_presentations=len(native), actual_exposures=len(np.unique(actual)),
        source_pts_first=int(native[0]), source_pts_last=int(native[-1]),
        native_pts_sha256=hashlib.sha256(native.tobytes()).hexdigest(), segments=segments,
        expected_geometry=profile['expected_geometry'], display_camera_smoothing=False,
        trajectory_post_filter=False, draw_workers=2, draw_slots=8, draw_chunk_size=2,
        whole_programme_accuracy_accepted=False,
        **({'single_pass': True, 'current_source_point_coverage_expected': point_coverage}
           if single_pass else {}))


def run_command(command, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / f'{name}.command.json', command)
    environment = os.environ.copy()
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        environment[key] = '1'
    with (directory / f'{name}.stdout.log').open('w', encoding='utf-8') as stdout, \
            (directory / f'{name}.stderr.log').open('w', encoding='utf-8') as stderr:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
    if result.returncode:
        raise RuntimeError(f'{name} failed ({result.returncode}); retained logs: {directory}')


def check_body(plan):
    body_directory = Path(plan['body_directory'])
    report = read(body_directory / 'report.json')
    preparation = read(body_directory / 'camera_rebind_prepare.json')
    if not same_input(preparation['replacement_camera'], plan['camera']):
        raise ValueError('Body preparation uses a different saved camera')
    if not same_input(report['pelvis_source_root_refinement']['camera'], plan['camera']):
        raise ValueError('Body refinement uses a different saved camera')
    if not same_input(report['pelvis_source_root_refinement']['scene'], plan['scene']):
        raise ValueError('Body refinement uses a different complete scene')
    if not all(report['pelvis_source_root_refinement']['preserved_source_local_pose_clock_fields'].values()):
        raise ValueError('Body source/local pose/clock fields were not preserved')
    body_path = body_directory / 'actual_body_root_fit.npz'
    if sha(body_path) != report['output']['sha256']:
        raise ValueError('Finished body output identity changed')
    with np.load(plan['camera']['path'], allow_pickle=False) as camera, np.load(body_path, allow_pickle=False) as body:
        actual, representatives = body['actual_pts'], camera['exposure_representative_pts']
        rows = np.searchsorted(actual, representatives)
        if (np.any(rows >= len(actual)) or not np.array_equal(actual[rows], representatives)
                or not np.array_equal(body['camera25'][rows], camera['camera25'])
                or not np.isfinite(body['P_world_m']).all()):
            raise ValueError('Body must retain every actual exposure in the exact saved camera')
    return reference(body_path)


def check_segment(plan, segment, body):
    directory = Path(segment['directory'])
    receipt = read(directory / 'render.json')
    production = receipt['video_production']
    expected = [plan['camera'], body, plan['scene'], plan['current_source_points']]
    if len(receipt['inputs']) != len(expected) or not all(same_input(a, b) for a, b in zip(receipt['inputs'], expected)):
        raise ValueError('Segment used a different camera/body/scene')
    if receipt.get('current_source_points') != plan['current_source_points']:
        raise ValueError('Complete video omitted or changed the current active source points')
    if plan.get('single_pass'):
        coverage = production.get('current_source_point_draw_coverage', {})
        expected_coverage = plan['current_source_point_coverage_expected']
        if (any(coverage.get(k) != v for k, v in expected_coverage.items())
                or coverage.get('all_native_frames_points_draw_invoked') is not True
                or coverage.get('only_current_exposure_points') is not True
                or coverage.get('missing_exposures_not_filled') is not True):
            raise ValueError('All source frames, including frame zero and cut returns, must draw their own current points')
    if (receipt['native_video_frames'] != segment['native_frames']
            or production['source_pts_first'] != segment['source_start_pts']
            or production['source_pts_last'] != segment['source_last_pts_inclusive']
            or production['encoded_input_native_pts_sha256'] != segment['native_pts_sha256']
            or not production['encoded_input_exact_source_order']):
        raise ValueError('Segment lost/duplicated an exact source presentation')
    drawn = receipt['source_and_plan_all_geometry']
    required = plan['expected_geometry']['source_and_plan_all_geometry']
    # Grid fragments change with the saved rink extent and metric scale. Keep
    # every physical layer exact and require the grid to be present, rather
    # than requiring a new scene to have the old scene's fragment count.
    complete_geometry = (drawn.keys() == required.keys() and drawn.get('grid', 0) > 0
                         and all(drawn[k] == v for k, v in required.items() if k != 'grid'))
    if (receipt['display_camera_smoothing']['enabled'] or receipt['trajectory_filtered_in_renderer']
            or not receipt['camera_and_body_exposures_equal']
            or not complete_geometry
            or receipt['advertisement_ids'] != plan['expected_geometry']['advertisement_ids']):
        raise ValueError('Complete same-model layers/body or raw-camera display contract changed')
    if not (directory / 'grid_and_trajectory_WORKING.mp4').is_file():
        raise FileNotFoundError(directory / 'grid_and_trajectory_WORKING.mp4')
    return receipt


def check_encoded_endpoints(path, count):
    """Read only the first and final encoded GOPs; never render/export samples."""
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        first = next(container.decode(stream))
        first_time = first.pts * stream.time_base
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        expected_last = Fraction(count - 1, 60)
        container.seek(int(expected_last / stream.time_base), stream=stream, backward=True)
        last = None
        for frame in container.decode(stream):
            last = frame
        if last is None:
            raise ValueError('Final encoded source frame cannot be decoded')
        last_time = last.pts * stream.time_base
    if first_time != 0 or last_time != expected_last:
        raise ValueError('Decoded first/final frame cadence differs from exact presentation count')
    return dict(first_and_last_encoded_frames_decoded=True, first_encoded_frame_index=0,
        last_encoded_frame_index=count - 1, first_encoded_time=str(first_time),
        last_encoded_time=str(last_time), sample_images_or_videos_written=False)


def complete_programme_segment(plan):
    """Select every ordered native presentation in one public renderer call."""
    command = list(plan['segments'][0]['command'])
    for flag, value in (('--output', plan['output']),
                        ('--start-pts', plan['source_pts_first']),
                        ('--end-pts', plan['source_pts_last'])):
        command[command.index(flag) + 1] = str(value)
    return dict(directory=plan['output'], command=command,
        native_frames=plan['native_presentations'],
        source_start_pts=plan['source_pts_first'],
        source_last_pts_inclusive=plan['source_pts_last'],
        native_pts_sha256=plan['native_pts_sha256'])


def execute_single_pass(plan, resume):
    """Refit body once, then decode/draw/encode the complete programme once."""
    output = Path(plan['output'])
    started = time.perf_counter()
    status = dict(q_sha256=plan['q_sha256'], status='running', single_pass=True)

    def stage(value):
        status['stage'] = value
        write(output / 'export_status.json', status)
        print(json.dumps(status, ensure_ascii=False), flush=True)

    try:
        stage('source_identity')
        if sha(plan['source']) != plan['source_video_sha256']:
            raise ValueError('Registered original source video bytes changed')
        stage('body_source_refit')
        body_directory = Path(plan['body_directory'])
        if body_directory.exists():
            if not resume:
                raise FileExistsError('Existing body output requires --resume and complete identity checks')
        else:
            run_command(plan['body_command'], output, 'body_refit')
        body = check_body(plan)
        body_check = body_directory / 'same_current_camera_body_source_check.json'
        write(body_check, dict(q_sha256=plan['q_sha256'], body=body,
            camera=plan['camera'], scene=plan['scene'],
            camera_body_all_native_presentations_bitexact=True,
            source_local_pose_clock_preserved=True))
        stage('whole_programme_render')
        segment = complete_programme_segment(plan)
        target = output / 'grid_and_trajectory_WORKING.mp4'
        receipt_path = output / 'render.json'
        if receipt_path.exists():
            if not resume:
                raise FileExistsError(receipt_path)
        else:
            if target.exists():
                if not resume:
                    raise FileExistsError(target)
                if target.resolve().parent != output.resolve() or not output.resolve().is_relative_to(DATA):
                    raise ValueError('Incomplete video preservation path escaped its Data output')
                target.rename(output / f'grid_and_trajectory_FAILED_{time.time_ns()}.mp4')
            run_command(segment['command'], output, 'render')
        receipt = check_segment(plan, segment, body)
        stage('stream_and_endpoint_check')
        ffprobe = shutil.which('ffprobe')
        if not ffprobe:
            raise FileNotFoundError('Existing ffprobe required for final stream count')
        command = [ffprobe, '-v', 'error', '-show_entries',
            'stream=codec_name,codec_type,width,height,avg_frame_rate,time_base,nb_frames,duration:format=duration,size',
            '-of', 'json', str(target)]
        write(output / 'ffprobe.command.json', command)
        probe = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
        video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        if (int(video['nb_frames']) != plan['native_presentations']
                or video['avg_frame_rate'] != '60/1'
                or [video['width'], video['height']] != [2580, 1160]
                or abs(float(probe['format']['duration']) - plan['native_presentations'] / 60.) > .03):
            raise ValueError('Complete stream count/dimensions/cadence differ')
        write(output / 'ffprobe_complete_video.json', probe)
        endpoints = check_encoded_endpoints(target, plan['native_presentations'])
        endpoints.update(source_pts_first=plan['source_pts_first'], source_pts_last=plan['source_pts_last'])
        write(output / 'encoded_endpoint_decode_check.json', endpoints)
        receipt.update(complete_q_sha256=plan['q_sha256'],
            body_source_validation=reference(body_check),
            complete_source_shots=plan['source_shots'],
            single_pass_complete_programme=True,
            encoded_endpoint_decode_check=reference(output / 'encoded_endpoint_decode_check.json'),
            wall_seconds_including_body_decode_draw_encode=time.perf_counter()-started,
            whole_programme_accuracy_accepted=False)
        write(receipt_path, receipt)
        status.update(status='complete', video=str(target),
            native_presentations=plan['native_presentations'], actual_exposures=plan['actual_exposures'])
        stage('complete')
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(output / 'export_status.json', status)
        raise


def execute(plan, resume):
    if plan.get('single_pass'):
        return execute_single_pass(plan, resume)
    output = Path(plan['output'])
    started = time.perf_counter()
    status = dict(q_sha256=plan['q_sha256'], status='running', completed_shots=[])
    def stage(value):
        status['stage'] = value
        write(output / 'export_status.json', status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
    try:
        stage('source_identity')
        # Hash the registered original once per invocation, never once per shot.
        if sha(plan['source']) != plan['source_video_sha256']:
            raise ValueError('Registered original source video bytes changed')
        stage('body_source_refit')
        body_directory = Path(plan['body_directory'])
        if body_directory.exists():
            if not resume:
                raise FileExistsError('Existing body output requires --resume and complete identity checks')
        else:
            run_command(plan['body_command'], output, 'body_refit')
        body = check_body(plan)
        write(body_directory / 'same_current_camera_body_source_check.json',
            dict(q_sha256=plan['q_sha256'], body=body, camera=plan['camera'], scene=plan['scene'],
                 camera_body_all_native_presentations_bitexact=True, source_local_pose_clock_preserved=True))
        receipts = []
        for segment in plan['segments']:
            stage(f"shot_{segment['shot_index']:02d}")
            directory = Path(segment['directory'])
            receipt_path = directory / 'render.json'
            target = directory / 'grid_and_trajectory_WORKING.mp4'
            if receipt_path.exists():
                if not resume:
                    raise FileExistsError(receipt_path)
            else:
                if target.exists():
                    if not resume:
                        raise FileExistsError(target)
                    # Preserve only this owned incomplete file, in the same output folder.
                    if target.resolve().parent != directory.resolve() or not directory.resolve().is_relative_to(output):
                        raise ValueError('Incomplete-segment preservation path escaped this output')
                    target.rename(directory / f'grid_and_trajectory_FAILED_{time.time_ns()}.mp4')
                run_command(segment['command'], directory, 'render')
            receipt = check_segment(plan, segment, body)
            receipts.append(receipt)
            status['completed_shots'].append(segment['shot_index'])
        if len({r['video_production']['selected_encoder_name'] for r in receipts}) != 1:
            raise ValueError('Segments selected different encoder configurations; inspect retained evidence')
        stage('concat_and_stream_check')
        import imageio_ffmpeg
        concat = output / 'nine_source_shot_files.ffconcat'
        files = [str((Path(s['directory']) / 'grid_and_trajectory_WORKING.mp4').resolve()).replace('\\', '/')
                 for s in plan['segments']]
        concat.write_text('ffconcat version 1.0\n' + ''.join("file '" + p.replace("'", "'\\''") + "'\n" for p in files), encoding='utf-8')
        target = output / 'grid_and_trajectory_WORKING.mp4'
        if target.exists():
            if not resume or not (output / 'render.json').exists():
                raise FileExistsError('Existing complete/partial concat retained; use a new output for an incomplete concat')
        else:
            command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                '-f', 'concat', '-safe', '0', '-i', str(concat), '-c', 'copy', '-an',
                '-movflags', '+faststart', str(target)]
            run_command(command, output, 'concat')
        ffprobe = shutil.which('ffprobe')
        if not ffprobe:
            raise FileNotFoundError('Existing ffprobe required for final stream count')
        command = [ffprobe, '-v', 'error', '-show_entries',
            'stream=codec_name,codec_type,width,height,avg_frame_rate,time_base,nb_frames,duration:format=duration,size',
            '-of', 'json', str(target)]
        write(output / 'ffprobe.command.json', command)
        probe = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
        video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        if (int(video['nb_frames']) != plan['native_presentations'] or video['avg_frame_rate'] != '60/1'
                or [video['width'], video['height']] != [2580, 1160]
                or abs(float(probe['format']['duration']) - plan['native_presentations'] / 60.) > .03):
            raise ValueError('Final concatenated stream count/dimensions/cadence differ')
        write(output / 'ffprobe_complete_video.json', probe)
        endpoints = check_encoded_endpoints(target, plan['native_presentations'])
        endpoints.update(source_pts_first=plan['source_pts_first'], source_pts_last=plan['source_pts_last'])
        write(output / 'encoded_endpoint_decode_check.json', endpoints)
        final = dict(receipts[0])
        final.update(files=[str(target)], native_video_frames=plan['native_presentations'],
            complete_q_sha256=plan['q_sha256'], body_source_validation=reference(body_directory / 'same_current_camera_body_source_check.json'),
            video_production=dict(mode='public_body_refit_then_nine_exact_source_shots_stream_concat',
                selected_encoder_name=receipts[0]['video_production']['selected_encoder_name'],
                source_pts_first=plan['source_pts_first'], source_pts_last=plan['source_pts_last'],
                encoded_input_native_pts_sha256=plan['native_pts_sha256'], encoded_input_exact_source_order=True,
                wall_seconds_including_body_decode_draw_encode=time.perf_counter()-started,
                drawing_pipeline=dict(workers=2, slots=8, chunk_size=2, sequential_segments=True),
                segments=[dict(shot_index=s['shot_index'], native_frames=s['native_frames'],
                    first_source_pts=s['source_start_pts'], last_source_pts=s['source_last_pts_inclusive'],
                    receipt=reference(Path(s['directory']) / 'render.json'), video=str(Path(s['directory']) / 'grid_and_trajectory_WORKING.mp4'))
                    for s in plan['segments']], source_clock=plan['source_shots']),
            encoded_endpoint_decode_check=reference(output / 'encoded_endpoint_decode_check.json'),
            whole_programme_accuracy_accepted=False)
        write(output / 'render.json', final)
        status.update(status='complete', video=str(target), native_presentations=plan['native_presentations'])
        stage('complete')
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(output / 'export_status.json', status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--previous-body-directory', type=Path)
    parser.add_argument('--current-source-points', type=Path,
                        help='Complete current active UV archive bound to the saved model and camera')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--single-pass', action='store_true',
                        help='Refit body once and render all complete shots in one source decode')
    args = parser.parse_args()
    plan = prepare(args.state_directory, args.output, args.profile, args.previous_body_directory,
                   args.current_source_points, args.single_pass)
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=True)
    previous_plan = output / 'export_plan.json'
    if previous_plan.exists() and read(previous_plan) != plan:
        raise ValueError('Existing output belongs to a different export plan; preserve it and choose a new output')
    write(previous_plan, plan)
    if args.execute:
        execute(plan, args.resume)
    else:
        print(json.dumps(dict(status='prepared_no_body_fit_or_render_started', plan=str(previous_plan),
            native_presentations=plan['native_presentations'], shots=len(plan['segments']), q_sha256=plan['q_sha256']), ensure_ascii=False))


if __name__ == '__main__':
    main()
