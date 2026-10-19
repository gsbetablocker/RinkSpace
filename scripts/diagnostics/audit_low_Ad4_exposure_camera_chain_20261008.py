"""Numeric PTS/view/export audit over the existing half-second source interval."""
from pathlib import Path
import argparse, hashlib, json, sys
import cv2
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project(camera, xyz):
    R = cv2.Rodrigues(camera[3:6])[0]
    K = np.array([[np.exp(camera[6]), camera[10], camera[8]],
                  [0., np.exp(camera[7]), camera[9]], [0., 0., 1.]])
    return cv2.projectPoints(np.asarray(xyz, float), camera[3:6], -R @ camera[:3], K, camera[11:25])[0].reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser()
    for key in ('state', 'initial', 'clock', 'connectivity', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    ident = read(args.state / 'state_identity.json')
    initial_ident = read(args.initial / 'state_identity.json')
    state_path = args.state / 'complete_state_UNACCEPTED.npz'
    assert sha(state_path) == ident['state_npz_sha256']
    assert sha(ident['camera_path']) == ident['camera_sha256']
    assert sha(initial_ident['camera_path']) == initial_ident['camera_sha256']
    with np.load(state_path, allow_pickle=False) as archive:
        q = archive['q'].copy()
        lengths = archive['aux__block_lengths'] // 4
        blocks = archive['aux__block_names'].tolist()
        offsets = np.r_[0, np.cumsum(lengths)]
        bi = blocks.index('low')
        stamps = archive['aux__canonical_pts'][offsets[bi]:offsets[bi + 1]].copy()
        start = int(archive['aux__block_start'][bi])
        layout = json.loads(str(archive['aux__current_complete_parameter_layout_json'].item()))
    with np.load(ident['camera_path'], allow_pickle=False) as archive:
        cameras = {key: archive[key].copy() for key in archive.files}
    with np.load(initial_ident['camera_path'], allow_pickle=False) as archive:
        initial_cameras = {key: archive[key].copy() for key in ('pts', 'camera25')}
    lookup = {int(p): i for i, p in enumerate(cameras['pts'])}
    old_lookup = {int(p): i for i, p in enumerate(initial_cameras['pts'])}
    aliases = dict(zip(map(int, cameras['pts']), map(int, cameras['exposure_representative_pts'])))
    metadata = json.loads(str(cameras['metadata_json'].item()))
    low_id = str(cameras['installation_id'][lookup[229953536]])
    cal = metadata['calibrations'][low_id]
    assert cal['focal_domain'] == 'final_effective' and cal['A_source_to_final'] is None
    assert cal['aspect_fx_over_fy_in_domain'] == 1. and cal['axial_coefficients'] == [0.]
    p0, p1 = 229953536, 229961216
    times = cameras['pts'][(cameras['pts'] >= p0) & (cameras['pts'] <= p1)]
    assert len(times) == 31 and np.array_equal(np.diff(times), np.full(30, 256))
    clock = {}
    for line in args.clock.open(encoding='utf8'):
        row = json.loads(line)
        if p0 <= int(row['full_pts']) <= p1:
            clock[int(row['full_pts'])] = row
    assert len(clock) == len(times)
    scene = read(args.state / 'scene_UNACCEPTED.json')
    initial_scene = read(args.initial / 'scene_UNACCEPTED.json')
    ad4 = next(row for row in scene['advertisements'] if row['user_number'] == 4)
    old_ad4 = next(row for row in initial_scene['advertisements'] if row['user_number'] == 4)
    world = np.asarray(ad4['fitted_boundary_curves_xyz']['left_vertical_xyz'], float)
    old_world = np.asarray(old_ad4['fitted_boundary_curves_xyz']['left_vertical_xyz'], float)
    rows, previous = [], None
    max_export_error = max_view_error = max_alias_error = max_focal_error = 0.
    for native in map(int, times):
        i = lookup[native]
        actual = aliases[native]
        j = int(np.flatnonzero(stamps == actual)[0])
        columns = list(range(start + 4 * j, start + 4 * j + 4))
        assert columns[-1] < layout['original_physical_core_parameter_count']
        stored = cameras['view_state4'][i]
        view = q[columns].copy()
        view[3] -= np.log(1920.)
        cam = cameras['camera25'][i]
        canonical = cameras['camera25'][lookup[actual]]
        assert clock[native]['match_status'] == 'unique_pixel_match'
        assert clock[native]['accepted_official_pts'] == clock[actual]['accepted_official_pts']
        R = cv2.Rodrigues(stored[:3])[0]
        centre = np.asarray(cal['pivot_xyz']) + R.T @ np.asarray(cal['lever_xyz'])
        pp = [1920. * cal['principal_x_fraction_coefficients'][0], 1080. * cal['principal_y_fraction_coefficients'][0]]
        log_f = np.log(1920. * np.exp(stored[3]))
        replay = np.r_[centre, stored[:3], log_f, log_f, pp, 0., cal['k1_coefficients'][0], np.zeros(13)]
        max_export_error = max(max_export_error, float(abs(cam - replay).max()))
        max_view_error = max(max_view_error, float(abs(stored - view).max()))
        max_alias_error = max(max_alias_error, float(abs(cam - canonical).max()))
        native_f = np.exp(q[columns[-1]])
        max_focal_error = max(max_focal_error, float(abs(native_f - np.exp(cam[6])) / native_f))
        rotation_degrees = 0. if previous is None else float(np.degrees(np.linalg.norm(cv2.Rodrigues(R @ previous.T)[0])))
        old_cam = initial_cameras['camera25'][old_lookup[native]]
        uv = project(cam, world)
        old_camera_uv = project(old_cam, world)
        geometry_only_uv = project(old_cam, old_world)
        rows.append(dict(native_pts=native, actual_pts=actual, official_pts=clock[native]['accepted_official_pts'],
            source_clock_match=clock[native]['match_status'], low_view_index=j, canonical_q_RF4_indices=columns,
            source_independent_RF4=True, view_RF4=stored.tolist(), focal_px=float(np.exp(cam[6])),
            previous_native_rotation_change_degrees=rotation_degrees, optical_centre_scene=cam[:3].tolist(),
            same_actual_exposure_camera_is_exact_copy=bool(np.array_equal(cam, canonical)),
            projected_current_Ad4_left_midpoint_native_pixels=uv[len(uv) // 2].tolist(),
            same_current_world_camera_update_midpoint_pixels=(uv - old_camera_uv)[len(uv) // 2].tolist(),
            same_initial_camera_geometry_update_midpoint_pixels=(old_camera_uv - geometry_only_uv)[len(uv) // 2].tolist()))
        previous = R
    assert max_export_error < 1e-11 and max_view_error == 0 and max_alias_error == 0 and max_focal_error < 1e-14
    unique = sorted(set(row['actual_pts'] for row in rows))
    assert len(unique) == 26
    indices = [int(np.flatnonzero(stamps == p)[0]) for p in unique]
    assert indices == list(range(indices[0], indices[-1] + 1))
    event = read(args.state / 'source_review_UNACCEPTED.json')['event']
    fixed = event.get('explicit_unobserved_held_state_indices_fixed', [])
    assert not set(fixed) & {k for row in rows for k in row['canonical_q_RF4_indices']}
    report = dict(schema='rinkspace.original-low-halfsecond-source-clock-and-physical-camera-chain-audit.v1',
        complete_q_sha256=ident['q_sha256'], camera_path=ident['camera_path'], camera_sha256=ident['camera_sha256'],
        source_interval_native_pts=[p0, p1], source_time_base=metadata['source_time_base'], source_interval_seconds=(p1 - p0) / 15360.,
        native_presentations=len(times), actual_independent_exposures=len(unique), repeated_presentations=5,
        endpoint_aliases_self=bool(aliases[p0] == p0 and aliases[p1] == p1),
        all_source_clock_matches_unique_and_aliases_same_exposure=True,
        all26_RF4_are_distinct_contiguous_free_q_blocks=True, actual_fixed_indices=fixed,
        any_interval_RF4_fixed=False, actual_view_q_replay_max_error=max_view_error,
        actual_physical_camera_formula_replay_max_error=max_export_error, repeated_camera_copy_max_error=max_alias_error,
        native_motion_vs_export_focal_relative_max_error=max_focal_error,
        no_interpolated_or_neighbour_filled_camera_on_active_source_export=True,
        original_motion_connectivity=dict(path=str(args.connectivity), sha256=sha(args.connectivity),
            note='Original source-only pair ledger; connectiveness does not establish finite-material observability'),
        maximum_adjacent_native_rotation_change_degrees=max(row['previous_native_rotation_change_degrees'] for row in rows),
        maximum_relative_native_focal_change=float(np.max(abs(np.diff([row['focal_px'] for row in rows])) / np.asarray([row['focal_px'] for row in rows[:-1]]))),
        time_or_camera_chart_bug_found=False, geometry_or_source_accuracy_accepted=False,
        source_UV_sigma_roles_profiles_or_parameters_modified=False, programme_graph_or_global_J_constructed=False,
        source_camera_route_files=[str(Path('calibration/solver_runtime/legacy/snapshot/three_frame_refit/program_tracking') / 'z190_joint_fixed_xyz_rink_block.py'),
            str(Path('calibration/solver_runtime/legacy/snapshot/three_frame_refit/program_tracking') / 'z190_moving_pivot_source_bundle.py'),
            str(Path('calibration/solver_runtime/legacy/snapshot/three_frame_refit/program_tracking') / 'z190_physical_camera.py')], rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps(dict(output=str(args.output), sha256=sha(args.output), actual_exposures=len(unique),
        max_camera_replay=max_export_error, max_view_replay=max_view_error, max_alias_copy=max_alias_error,
        endpoint_camera_update_midpoint_pixels=[rows[0]['same_current_world_camera_update_midpoint_pixels'], rows[-1]['same_current_world_camera_update_midpoint_pixels']],
        max_relative_focal_change=report['maximum_relative_native_focal_change'], clock_or_camera_chart_bug_found=False)))


if __name__ == '__main__':
    main()
