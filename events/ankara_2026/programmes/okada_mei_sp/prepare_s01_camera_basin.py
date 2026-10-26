"""Explicit alternate camera initialisation; never changes the production run.

Native projections of the accepted scene provide temporary pose targets only.
They are not observations, priors, weights, or a second production objective.
The resulting state is consumed by the unchanged full native joint solver.
"""
from pathlib import Path
import argparse
import io
import json
import os
import shutil
import time

import numpy as np
from scipy.optimize import least_squares

import prepare_native_s01 as prep
from rinkspace.core.native import NativeCore
from rinkspace.domain.motion_chart import UniformMotionChart, cubic_samples


def snapshot(path):
    with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def make_projection_feasible(out):
    """Back out initial zoom only until every real row has a native RS root."""
    state = snapshot(out / 'seed_state.npz')
    if (out / 'feasible_seed_state.npz').exists():
        raise ValueError('Existing feasible seed must not be overwritten')
    observations, _, _ = prep.load_pixel_input(out)
    if str(state['point_input_sha256']) != prep.digest(out/'point_observations.npz'):
        raise ValueError('Candidate input changed')
    camera = prep.camera_for_snapshot(prep.read(prep.CAMERA), state)
    core = NativeCore(prep.LIBRARY)
    xyz = snapshot(prep.RUN / 'evaluation_pixels.npz')['values'][:, :3]
    times = (observations['pts']-camera['reference_pts'])/12800.
    attempts = []
    old_controls = state['controls'].copy()
    while True:
        projected = core.project_motion(state['rig'], state['controls'],
            camera['first_control_time'], camera['control_dt'], camera['image_height'], times, xyz)
        codes, counts = np.unique(projected['status'], return_counts=True)
        attempts.append(dict(log_f_shift=float(state['controls'][0,2]-old_controls[0,2]),
                             status_counts=dict(zip(map(str,codes),map(int,counts)))))
        if not np.any(projected['status']):
            break
        if any(code not in (0,4) for code in codes):
            raise ValueError('Initial camera has a domain failure other than off-image rows')
        state['controls'][:, 2] -= np.log(1.05)
        if not np.isfinite(np.exp(state['controls'][:,2])).all() or np.min(np.exp(state['controls'][:,2])) == 0:
            raise ValueError('No finite projection-feasible initial zoom found')
    np.savez(out / 'feasible_seed_state.npz', **state)
    report = dict(attempts=attempts, observations_changed=False,
                  adjusted_initial_zoom_only=True, no_new_fitted_bound=True,
                  all_rows_valid=True, native_rows=len(observations['pts']))
    prep.write(out/'seed_projection_feasibility.json', report)
    print(json.dumps(report, ensure_ascii=True), flush=True)


def prepare(out, fraction, interior_mount=False):
    if not 0 < fraction <= 1:
        raise ValueError('Optical initialisation fraction must be in (0, 1]')
    out = out.resolve()
    if not out.is_relative_to(prep.RUN.resolve()) or out == prep.RUN.resolve():
        raise ValueError('Candidate must be isolated below the current run')
    if (out / 'seed_state.npz').exists() or (out / 'solve.jsonl').exists():
        raise ValueError('An existing numerical candidate must not be overwritten')
    out.mkdir(parents=True, exist_ok=True)
    log = (out / 'initialization.jsonl').open('a', encoding='utf-8', buffering=1)
    started = time.monotonic()

    def emit(event, **values):
        record = dict(event=event, elapsed_seconds=time.monotonic()-started, **values)
        line = json.dumps(record, ensure_ascii=True, allow_nan=False)
        log.write(line+'\n')
        print(line, flush=True)

    source = prep.RUN / 'accepted_pixels_latest.npz'
    source_sha = prep.digest(source)
    state = snapshot(source)
    observations, mapping, _ = prep.load_pixel_input(prep.RUN)
    input_sha = prep.digest(prep.RUN / 'point_observations.npz')
    if str(state['point_input_sha256']) != input_sha:
        raise ValueError('Accepted state and current observations differ')
    for name in ('point_observations.npz', 'initialized_points.npz'):
        if not (out / name).exists():
            os.link(prep.RUN / name, out / name)
        elif prep.digest(out / name) != prep.digest(prep.RUN / name):
            raise ValueError('Existing candidate input differs: '+name)
    for source_file, destination in [(prep.RUN/name, out/name) for name in ('mapping.json', 'point_input.json')]+[(source, out/'before_state.npz')]:
        if not destination.exists():
            shutil.copyfile(source_file, destination)
        elif prep.digest(destination) != prep.digest(source_file):
            raise ValueError('Existing candidate baseline differs: '+str(destination))
    evaluated = snapshot(prep.RUN / 'evaluation_pixels.npz')
    if np.any(evaluated['status']) or len(evaluated['values']) != len(observations['pts']):
        raise ValueError('Accepted scene evaluation is incomplete')
    camera = prep.camera_for_snapshot(prep.read(prep.CAMERA), state)
    core = NativeCore(prep.LIBRARY)
    rig = state['rig'].copy()
    original_rig = rig.copy()
    target = rig.copy()
    target[26:31] = [959.5, 539.5, 0., 0., 0.]
    target[31:37] = 0.
    target[39:43] = 0.
    rig += fraction*(target-rig)
    # The new focal trajectory must not inherit an endpoint-saturated pupil
    # polynomial as an accidental initialisation constraint. Both coefficients
    # remain present and active in the subsequent 31-coordinate camera chart.
    rig[22:24] = 0.
    metres = prep.read(prep.ROOT / 'events/ankara_2026/model_constraints/ankara_2026_metric_scale_assumed_v1.json')['converted_span_xy_m'][0]
    if interior_mount:
        mount = prep.read(prep.ROOT / 'events/ankara_2026/model_constraints/camera_hardware/z190_mount_working_ranges_20261026.json')['working_ranges']
        rig[17] = mount['single_pan_reference_height_above_ice_m']['initial']/metres
        rig[18:21] = np.asarray(mount['complete_tilt_axis_to_projection_centre_camera_m']['initial_reference'])/metres
        rig[21] = mount['axis_spacing_m']['initial']/metres
    controls = state['controls']
    first, dt = camera['first_control_time'], camera['control_dt']
    chart = UniformMotionChart.create(len(controls), first, dt,
                                     int(state['motion_knot_stride']),
                                     float(state['motion_reference_time']))
    pts = observations['pts']
    unique_pts = np.unique(pts)
    frame_times = (unique_pts-camera['reference_pts'])/12800.
    # Sample twice per fitted cubic interval plus both native support endpoints.
    # This selects initial coefficients only; every real frame enters the fit.
    sample_times = np.unique(np.r_[first+dt,
        np.arange(frame_times[0], frame_times[-1], dt*chart.stride/2),
        frame_times[-1], first+(len(controls)-2)*dt, chart.reference_time])
    old_pose = core.evaluate_motion_camera(original_rig, controls, first, dt, sample_times)
    if np.any(old_pose['status']):
        raise ValueError('Original trajectory does not cover initialisation times')
    original_q = np.c_[old_pose['pan'], old_pose['tilt'], old_pose['log_f']]
    initialized = []
    pose_errors = []
    row_groups = {int(p): np.flatnonzero(pts == p) for p in unique_pts}

    def project(physical, q, xyz):
        instant = core.evaluate_motion_camera(physical, np.tile(q, (4, 1)), -1.5, 1., [0.])
        if np.any(instant['status']):
            raise ValueError('Alternate instantaneous physical camera is invalid')
        # Static native primitive deliberately permits temporary off-image UV.
        # No row-exposure root is needed for these constant-pose seed targets.
        static = np.r_[instant['rotation'][0].ravel(), instant['centre'][0],
            [instant[key][0] for key in ('fx','fy','cx','cy')], 0.,
            [instant[key][0] for key in ('k1','k2','p1','p2')], 0.]
        pixels, status = core.project(static, xyz)
        pixels = np.asarray(pixels)
        if np.any(status) or not np.isfinite(pixels).all():
            raise ValueError('Alternate instantaneous camera projection is invalid')
        return pixels

    emit('begin', baseline_sha256=source_sha, point_input_sha256=input_sha,
         optical_fraction=fraction, interior_mount=interior_mount,
         target_status='unmeasured initialization hypothesis, not a fixed optical model',
         temporary_projection_targets_are_observations=False)
    for index, (sample_time, q) in enumerate(zip(sample_times, original_q)):
        frame = int(np.argmin(np.abs(frame_times-sample_time)))
        rows = row_groups[int(unique_pts[frame])]
        xyz = evaluated['values'][rows, :3]
        uv = project(original_rig, q, xyz)
        result = least_squares(lambda candidate: (project(rig, candidate, xyz)-uv).ravel(),
                               q, method='lm', ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if not result.success:
            raise RuntimeError('Pose initialisation failed: '+result.message)
        initialized.append(result.x)
        pose_errors.append(float(np.sqrt(np.mean(result.fun**2))))
        if index % 40 == 0:
            emit('pose_initialization', samples_done=index+1, samples_total=len(sample_times),
                 virtual_target_rms_px=pose_errors[-1])
    indices, weights = cubic_samples(len(controls), first, dt, sample_times)
    basis = np.einsum('ij,ijk->ik', weights, chart.refinement[indices])
    coefficients, _, rank, _ = np.linalg.lstsq(basis, initialized, rcond=None)
    if rank != basis.shape[1]:
        raise ValueError('Initial motion coefficient sampling is rank deficient')
    new_controls = chart.refinement @ coefficients
    before_reference = core.evaluate_motion_camera(original_rig, controls, first, dt, [chart.reference_time])
    after_reference = core.evaluate_motion_camera(rig, new_controls, first, dt, [chart.reference_time])
    delta = np.array([after_reference[key][0]-before_reference[key][0] for key in ('pan','tilt')])
    # Exact angle-gauge transport: Rb(b-db) [Rb(db) R0 Rn(da)] Rn(a-da).
    # The actual rotations, axes and projection centres do not change here.
    transported = core.evaluate_motion_camera(rig, np.tile([*delta, np.log(rig[24])], (4,1)),
                                             -1.5, 1., [0.])
    if np.any(transported['status']):
        raise ValueError('Reference chart transport failed')
    pre_transport = core.evaluate_motion_camera(rig, new_controls, first, dt, sample_times)
    rig[:9] = transported['rotation'][0].ravel()
    new_controls[:, :2] -= delta
    post_transport = core.evaluate_motion_camera(rig, new_controls, first, dt, sample_times)
    gauge_difference = float(np.max(np.abs(pre_transport['values'][:,4:]-post_transport['values'][:,4:])))
    if np.any(post_transport['status']) or gauge_difference > 1e-8:
        raise ValueError('Reference chart transport changed the physical camera')
    state.update(rig=rig, controls=new_controls)
    np.savez(out / 'seed_state.npz', **state)
    report = dict(schema='rinkspace.s01_camera_basin_seed.v1',
        baseline=str(source), baseline_sha256=source_sha,
        point_input_sha256=input_sha, scene_revision=mapping['scene_revision'],
        rig_before=original_rig.tolist(), rig_seed=rig.tolist(),
        optical_fraction=fraction, interior_mount=interior_mount,
        temporary_projection_samples=len(sample_times),
        virtual_target_rms_min_median_max_px=np.percentile(pose_errors,[0,50,100]).tolist(),
        coarse_interpolation_max_angle_logf_change=np.max(np.abs(basis@coefficients-np.asarray(initialized)),axis=0).tolist(),
        reference_gauge_transport_max=gauge_difference,
        all_31_camera_parameters_remain_active=True,
        observation_fields_unchanged=True, production_state_changed=False,
        hypothetical_optical_seed_is_not_calibration=True,
        initializer_is_not_a_second_production_solver=True)
    prep.write(out / 'seed_report.json', report)
    if prep.digest(source) != source_sha or prep.digest(prep.RUN / 'point_observations.npz') != input_sha:
        raise RuntimeError('Authoritative inputs changed while preparing isolated seed')
    emit('complete', seed=str(out/'seed_state.npz'),
         virtual_target_rms_min_median_max_px=report['virtual_target_rms_min_median_max_px'],
         reference_gauge_transport_max=gauge_difference)
    log.close()
    make_projection_feasible(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--fraction', type=float, default=1.)
    parser.add_argument('--interior-mount', action='store_true')
    parser.add_argument('--projection-feasibility-only', action='store_true')
    args = parser.parse_args()
    if args.projection_feasibility_only:
        make_projection_feasible(args.out)
    else:
        prepare(args.out, args.fraction, args.interior_mount)
