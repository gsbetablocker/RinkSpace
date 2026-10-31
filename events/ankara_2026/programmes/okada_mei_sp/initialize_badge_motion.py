"""Initialize a previously unconstrained motion interval from source badge search.

Homography-warped catalogue samples are navigation only. Actual observations
remain the independently measured source exterior; the subsequent native solve
uses those original UV values and free curve stations.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

from build_independent_shot_packets import fit_pose_controls
from rinkspace.core.native import NativeCore
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.geometry import project
from rinkspace.path_policy import guard_project_write


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(a):
    manifest = json.loads((a.prepared/'joint_manifest.json').read_text())
    state = dict(np.load(a.resume, allow_pickle=False))
    library = Path(manifest['native_library'])
    if str(state['native_library_provenance_sha256']) != sha(library):
        raise ValueError('Input state/native library differs')
    core = NativeCore(library)
    timeline = ShotMotionTimeline(manifest['shots'], manifest['control_dt'], manifest['first_control_time'])
    shot = next(s for s in manifest['shots'] if s['id'] == 's07')
    cam = shot['camera_index']
    full = np.arange(shot['first_pts'], shot['last_pts']+256, 256)
    times = (full-shot['reference_pts'])/12800+timeline.offset(shot)
    camera = core.evaluate_motion_camera(state['rig'][cam], state['controls'], timeline.first, timeline.dt, times)
    poses = np.c_[camera['pan'], camera['tilt'], np.log(camera['fx'])]
    geometry = dict(np.load(a.reference/'reference_geometry.npz', allow_pickle=False))
    xyz = geometry['world_xyz']
    # The older independent reference provides a visible-pose seed only.
    seed_index = int(np.flatnonzero(full == a.seed_pts)[0])
    seed = poses[seed_index].copy()
    original = dict(np.load(a.reference/'frozen_state.npz', allow_pickle=False))
    seed_camera = core.evaluate_motion_camera(original['rig'][cam], original['controls'],
        timeline.first, timeline.dt, np.array([times[seed_index]]))
    alternative = np.array([seed_camera['pan'][0], seed_camera['tilt'][0], np.log(seed_camera['fx'][0])])
    registrations = {r['pts']: r for r in map(json.loads, (a.capture/'registration.jsonl').read_text().splitlines())}
    qualified = [r for r in map(json.loads, (a.qualified/'s07/source_rows.jsonl').read_text().splitlines()) if r['fit_eligible']]
    def projection(theta):
        return core.project_motion(state['rig'][cam], np.tile(theta, (6, 1)), -2., 1., 1080, 0., xyz)
    source_poses = {}
    reports = []
    output = guard_project_write(a.output.resolve())
    if output.exists():
        raise FileExistsError(output)
    log = output.with_suffix('.jsonl')
    with log.open('w', encoding='utf8', buffering=1) as stream:
        for record in qualified:
            pts = record['pts']
            note = registrations[pts]
            if not note['source_pattern_check']['supported']:
                raise ValueError('Badge initialization lacks actual pattern identity')
            target = project(np.array(note['H_current_from_reference']), geometry['reference_search_uv'])
            def residual(theta):
                result = projection(theta)
                value = result['uv']-target
                if np.any(result['status']) or not np.isfinite(value).all():
                    return np.full(2*len(xyz), 1e5)
                return value.ravel()
            current = poses[int(np.flatnonzero(full == pts)[0])]
            initial = min((seed, alternative, current), key=lambda x: np.linalg.norm(residual(x)))
            fit = least_squares(residual, initial, x_scale=[.2, .1, 1.], diff_step=1e-5,
                max_nfev=80, ftol=1e-8, xtol=1e-8, gtol=1e-6)
            value = residual(fit.x)
            if not np.isfinite(fit.x).all() or np.max(abs(value)) >= 1e4:
                raise ValueError(f'No projectable physical badge navigation pose at {pts}')
            seed = fit.x.copy()
            source_poses[pts] = seed
            report = dict(pts=pts, pose=seed.tolist(), navigation_rms_px=float(np.sqrt(np.mean(value**2))),
                navigation_max_px=float(abs(value).max()), initialization_only=True, observation_UV_created=0)
            reports.append(report)
            stream.write(json.dumps(report)+'\n')
    with np.load(a.prepared/'point_observations.npz', allow_pickle=False) as data:
        observed = np.unique(data['pts'][shot['row_begin']:shot['row_stop']])
    anchors = {int(p): poses[int(np.flatnonzero(full == p)[0])] for p in observed}
    anchors.update(source_poses)
    xp = np.array(sorted(anchors))
    values = np.array([anchors[p] for p in xp])
    initialized = PchipInterpolator(xp-xp[0], values, axis=0)(full-xp[0])
    controls, chart_note = fit_pose_controls(full, initialized)
    lo, hi = shot['control_begin'], shot['control_stop']
    before = state['controls'].copy()
    if controls.shape != state['controls'][lo:hi].shape:
        raise ValueError('S07 motion chart changed')
    state['controls'][lo:hi] = controls
    if not np.array_equal(before[:lo], state['controls'][:lo]) or not np.array_equal(before[hi:], state['controls'][hi:]):
        raise ValueError('Other shot controls changed')
    np.savez(output, **state)
    output.with_suffix('.json').write_text(json.dumps(dict(
        source_state=str(a.resume), source_state_sha256=sha(a.resume), output=str(output), output_sha256=sha(output),
        source_pattern_capture=str(a.capture), source_registration_sha256=sha(a.capture/'registration.jsonl'),
        qualified_badge_source=str(a.qualified), source_pose_frames=len(source_poses),
        existing_observed_anchor_frames=len(observed), source_query_pixels_are_navigation_only=True,
        actual_source_UV_unchanged=True, native_scene_and_rigs_unchanged=True, other_shot_controls_unchanged=True,
        initialization_only=True, final_trajectory_postsmoothed=False,
        native_motion_space_unchanged=True, chart_fit=chart_note, source_navigation=reports), indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(output=str(output), source_pose_frames=len(source_poses), chart=chart_note)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'resume', 'reference', 'capture', 'qualified', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--seed-pts', type=int, default=1433856)
    main(p.parse_args())
