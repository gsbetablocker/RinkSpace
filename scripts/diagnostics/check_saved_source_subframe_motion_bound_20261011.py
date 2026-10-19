"""Local saved-camera time-offset screen; no readout fit, graph, J or files.

Official-source pixel matches set neighboring distinct exposure times.  This
is a conditional finite-difference screen, not a bound on unknown motion
inside an exposure or evidence that rolling shutter was present on site.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
from calibration.solver_runtime import install, configure_native_runtime
install(); configure_native_runtime(stage='readonly_subframe_source_motion_screen', native_threads=1)
from rinkspace.vision_camera_probe import FrozenCameraSourceProbe
from structured_programme_source_candidate.native_ice_curve_source import _curve_error_on_association, _scanline_value
from structured_programme_source_candidate.current_ice_material_scene import install_source_ice_material_view
from wide_anchor_candidate.full_brown_projection import project_central_valid_full_brown
import low_end_confirmed_exposure_aliases as evidence
from continuous_exposure_clock import OFFICIAL_TICKS_PER_SECOND, OFFICIAL_FRAME_TICKS


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def emit(stage, **value):
    print(json.dumps(dict(stage=stage, **value), allow_nan=False,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--child', default='f9d83084051c0c5498ed01646f2fd4b8ef64dd15e46f8ec471eb5fba5ec97496')
    args = parser.parse_args()
    options = read(Path(str(args.run) + '.bootstrap/input_config.json'))['source_options']
    directory = args.run / 'accepted_states' / args.child
    identity = read(directory / 'state_identity.json')
    state_path = directory / 'complete_state_UNACCEPTED.npz'
    if sha(state_path) != identity['state_npz_sha256'] or sha(identity['camera_path']) != identity['camera_sha256']:
        raise ValueError('Actual saved state/camera file identity changed')
    probe = FrozenCameraSourceProbe(state_path, identity['camera_path'], options['--world-init'], options)
    if probe.provenance['q_sha256'] != args.child:
        raise ValueError('Actual q identity changed')
    install_source_ice_material_view(probe.binder)
    path = evidence.CADENCE / 'native60_to_official50.jsonl'
    if sha(path) != evidence.MAPPING_SHA256:
        raise ValueError('Existing official exposure mapping changed')
    clock = {}
    with path.open(encoding='utf8') as stream:
        for line in stream:
            row = json.loads(line)
            if int(row['full_pts']) in probe.actual_index:
                clock[int(row['full_pts'])] = row
    frame_seconds = OFFICIAL_FRAME_TICKS / OFFICIAL_TICKS_PER_SECOND
    emit('authenticated_scope', q=args.child, mapping_path=str(path), mapping_sha256=evidence.MAPPING_SHA256,
        public_camera_bitexact=True, official_exposure_interval_seconds=frame_seconds,
        display_frame_interval_not_used=True, actual_intra_exposure_readout_and_motion_not_measured=True)
    times = (229937408, 229940480, 229943552, 230506752, 230507776)
    for family, factor in probe.native.items():
        for role, rows in [('FIT', factor.rows), ('HELD', factor.held)]:
            for row in rows:
                curve = row['curve']; at = int(row['actual_pts'])
                selected = (at in times and (
                    (curve.get('kind') == 'advertisement_boundary' and curve.get('advertisement_number') == 4)
                    or (curve.get('kind') == 'current_ice_segment' and curve.get('feature_id') == 'B1')
                    or (curve.get('kind') == 'house_ring' and curve.get('side') == 'F' and curve.get('row') == 3)))
                if not selected:
                    continue
                c = clock.get(at)
                if c is None or c['match_status'] != 'unique_pixel_match' or c['accepted_official_pts'] is None:
                    emit('unavailable_row', id=row['id'], reason='No unique official source time'); continue
                official = int(c['accepted_official_pts']); shot = c['shot_index']
                neighbors = {}
                for sign in (-1, 1):
                    candidates = [pt for pt, rec in clock.items() if rec['shot_index'] == shot
                        and rec['match_status'] == 'unique_pixel_match'
                        and rec['accepted_official_pts'] == official + sign * OFFICIAL_FRAME_TICKS]
                    if len(candidates) != 1:
                        neighbors = {}; break
                    neighbors[sign] = candidates[0]
                if not neighbors:
                    emit('unavailable_row', id=row['id'], reason='Neighbor official exposure absent or conservatively ambiguous'); continue
                cameras = {sign: probe.camera['camera25'][probe.actual_index[pt]] for sign, pt in neighbors.items()}
                nominal_camera = probe.camera['camera25'][probe.actual_index[at]]
                iid = probe.camera['installation_id'][probe.actual_index[at]]
                if any(probe.camera['installation_id'][probe.actual_index[pt]] != iid for pt in neighbors.values()):
                    raise ValueError('A neighboring exposure crosses physical installations')
                world = factor._world(probe.q, curve)
                is_scanline = row.get('curve_observation_kind') == 'axis_scanline_branch'
                if is_scanline:
                    raw, assoc = _scanline_value(nominal_camera, world, row, with_association=True)
                    index, station = assoc['segment_indices'], assoc['nominal_station']
                else:
                    raw, (index, station) = _curve_error_on_association(nominal_camera, world, row['uv'])
                    station = np.clip(station, 0., 1.)
                raw = np.asarray(raw).reshape(-1, 2)
                # The finite segment's image station is converted to a world
                # station using endpoint depths; Brown chord interpolation
                # discrepancy is measured explicitly below, never hidden.
                uv, depth, valid = project_central_valid_full_brown(nominal_camera, world)
                if not np.all(valid[index] & valid[index+1]):
                    raise ValueError('Nominal finite source station leaves central domain')
                t_world = station * depth[index] / ((1.-station)*depth[index+1] + station*depth[index])
                material = world[index] + t_world[:, None] * (world[index+1] - world[index])
                def project(camera):
                    value, _, central = project_central_valid_full_brown(camera, material)
                    if not np.all(central):
                        raise ValueError('Fixed source material station leaves central domain')
                    return value
                point0 = project(nominal_camera)
                point = {sign: project(camera) for sign, camera in cameras.items()}
                velocity = (point[1]-point[-1])/(2.*frame_seconds)
                def value(camera):
                    return (np.asarray(_scanline_value(camera, world, row)).reshape(-1,2) if is_scanline else
                        _curve_error_on_association(camera, world, row['uv'])[0].reshape(-1,2))
                actual = {sign: value(camera) for sign, camera in cameras.items()}
                if is_scanline:
                    normals = np.tile([1., 0.], (len(raw), 1))
                    errors = raw[:, 0]
                    expected_foot = uv[index] + station[:,None]*(uv[index+1]-uv[index])
                else:
                    errors = np.linalg.norm(raw, axis=1)
                    normals = np.divide(raw, errors[:,None], out=np.zeros_like(raw), where=errors[:,None]>0.)
                    expected_foot = row['uv'] + raw
                normal_velocity = np.einsum('ni,ni->n', normals, (actual[1]-actual[-1])/(2.*frame_seconds))
                one_frame_axis_displacement = np.maximum(
                    abs(np.einsum('ni,ni->n', normals, actual[-1]-raw)),
                    abs(np.einsum('ni,ni->n', normals, actual[1]-raw)))
                required_seconds = np.divide(-errors, normal_velocity,
                    out=np.full(len(errors), np.nan), where=normal_velocity != 0.)
                emit('row', id=row['id'], role=role, pts=at, official_pts=official,
                    neighboring_actual_pts=[neighbors[-1], neighbors[1]], installation=str(iid),
                    source_uv=row['uv'], actual_raw_residual_px=raw,
                    fixed_material_velocity_xy_px_per_s=velocity,
                    previous_next_point_displacement_px=[point[-1]-point0, point[1]-point0],
                    actual_operator_residual_previous_current_next_px=[actual[-1],raw,actual[1]],
                    measured_axis_velocity_px_per_s=normal_velocity,
                    conservative_one_official_frame_measured_axis_displacement_px=one_frame_axis_displacement,
                    required_local_time_offset_seconds=[None if not np.isfinite(t) else float(t) for t in required_seconds],
                    original_same_row_source_clock_preserved=True,
                    world_material_chord_projection_error_max_px=float(np.linalg.norm(point0-expected_foot,axis=1).max()),
                    finite_difference_is_conditional_saved_model_not_unmeasured_readout_certificate=True)


if __name__ == '__main__':
    main()
