"""Read-only two-rig F2 nested-factor and local Jacobian-rank probe.

This does not solve, export a camera or alter a live solver module.  The main
14 original source curves and low-end historical source pixels are examined
with the same candidate scene and one F2 scene key.  A local Schur rank here is
not the rank of the complete 169-second graph with all source constraints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from runtime.bootstrap import add_geometry_dependencies

add_geometry_dependencies()
import numpy as np
from scipy.linalg import qr, svdvals

from anchors.geometry import OLD_SCENE, OBSERVATIONS, old_camera_vector, world_curve
from audit_quadratic_result import reconstruct
from continuous_camera_model import FrameBinding, ShotSpline
from continuous_reference_calibration import (
    StaticReferenceBasis, _measurement_callback, seed_reference_initial)
from continuous_rig_camera import ContinuousRigCameraModel
from continuous_quadratic_rig_optin import QuadraticRigCameraModel
from continuous_time_principal_optin import TimePrincipalRigCameraModel
from continuous_brown_adapter import adapt_factor
from robust_observations import pointwise_huber
from secondary_nearboard_adaptive_curve import (
    AdaptiveProjectedPolyline, adaptive_curve_callback)
from shared_f2_nested_factor import F2SourceCurve, NestedF2Factor, F2_SCENE_KEY


ROOT = Path(__file__).resolve().parents[2]
TMP = Path('D:/CodexData/RinkSpace_program_tracking_tmp')
SCENE = TMP/'joint_programme_atlas_provenance_full_v13/candidate_scene.json'
MAIN = TMP/'continuous_camera_model_v13_source_identity/main_long_side_curve_fix'
LOW = TMP/'continuous_camera_model_v11_time_principal/low_end'
MAIN_EXPECTED_SHA = '9b64abccc5f42c494307a2bb2274c762782f1d70277aefd6382d107bfb7e7f6d'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _packet(path: Path):
    with np.load(path, allow_pickle=False) as z:
        keys, values = z['keys'].copy(), z['states'].copy()
    if (values.shape != (len(keys), 11) or len(set(map(int, keys))) != len(keys)
            or not np.isfinite(values).all()):
        raise ValueError('Saved graph state packet is invalid')
    return {int(key): value for key, value in zip(keys, values)}


def low_time_model():
    """Reconstruct the actual v11 time-principal programme rig, read-only."""
    phase = json.loads((LOW/'phase_00.json').read_text(encoding='utf8'))
    states = _packet(LOW/'states.npz')
    prior = _packet(LOW/'prior_initializer_states.npz')
    if set(states) != set(prior):
        raise ValueError('Low-end state/prior graph keys differ')
    splines = {int(shot): ShotSpline(int(shot), tuple(map(int, row['keys'])),
                                    tuple(map(float, row['pts'])))
               for shot, row in phase['shot_knots'].items()}
    with np.load(LOW/'shot_cameras.npz', allow_pickle=False) as packet:
        frames = [FrameBinding(
            int(pts), int(rep), int(shot), 'low_end', camera.copy(),
            float(phase['frame_clock_virtual_pts'][str(int(pts))]))
            for pts, rep, shot, camera in zip(
                packet['pts'], packet['exposure_representative_pts'],
                packet['shot_index'], packet['camera25'])]
    base = ContinuousRigCameraModel(
        splines, frames, phase['lens_base_keys'], phase['lens_slope_keys'],
        phase['zoom_reference'], mount_keys=phase['mount_keys'],
        offset_keys=phase['offset_keys'],
        offset_slope_keys=phase['offset_slope_keys'],
        brown_keys=phase['brown_keys'],
        brown_slope_keys=phase['brown_slope_keys'])
    model = TimePrincipalRigCameraModel.from_rig(
        base, phase['time_principal'][
            'programme_principal_control_keys_by_shot'])
    if not {key for pts in model.frames for key in model.dependencies(pts)} <= set(states):
        raise ValueError('Low-end time-principal rig dependency is missing')
    return model, states, prior, phase


def _scene():
    scene = json.loads(SCENE.read_text(encoding='utf8'))
    feature = next(f for f in scene['ice_features'] if f['id'] == 'F2')
    outer = next(r for r in feature['rings'] if r['role'] == 'outer')
    return scene, np.asarray(outer['center_xyz'][:2], float), float(outer['radius'])


def main_sources(scene, center, radius):
    path = OBSERVATIONS['main_long_side']
    if sha(path) != MAIN_EXPECTED_SHA:
        raise ValueError('Original main source observation packet changed')
    raw = json.loads(path.read_text(encoding='utf-8-sig'))
    frames = {f['id']: f for f in raw['frames']}
    rows = [r for r in raw['observations']
            if r.get('target') == {'kind': 'circle_outline', 'feature_id': 'F2'}]
    if (len(rows) != 14 or sum(len(r['pixels_uv']) for r in rows) != 435
            or len({r['frame_id'] for r in rows}) != 14):
        raise ValueError('Main F2 actual-source 14/435 inventory changed')
    sources, companion = {}, {}
    for row in rows:
        if (row['use'] != 'fit' or row['type'] != 'curve'
                or row.get('synthetic_or_model_projected') is not False
                or row.get('measurement_origin') !=
                'actual_image_remeasurement_from_traceable_seed_identity'):
            raise ValueError('Main F2 source measurement identity changed')
        pts = int(frames[row['frame_id']]['source_pts'])
        sources[pts] = F2SourceCurve(
            pts, row['id'], np.asarray(row['pixels_uv'], float),
            world_curve(scene, row['target']), center, radius,
            float(row['sigma_px']))
        others = []
        for candidate in raw['observations']:
            if (candidate['frame_id'] != row['frame_id'] or candidate is row
                    or candidate.get('type') != 'curve'
                    or candidate.get('use') != 'fit'
                    or candidate.get('synthetic_or_model_projected') is not False
                    or not candidate.get('measurement_origin', '').startswith('actual_image')):
                continue
            xyz = world_curve(scene, candidate['target'])
            if xyz is not None:
                others.append((candidate, xyz))
        companion[pts] = others
    if 71147520 not in sources or 80547840 not in sources:
        raise ValueError('Reviewed main cross-rig source timestamps absent')
    return sources, companion, frames


def add_main_reference_rig(base_model, states, sources, frames):
    """Only new C/R reference nodes; all main shared P/K/Brown keys are reused."""
    old = json.loads(OLD_SCENE.read_text(encoding='utf-8-sig'))
    old_camera = {int(f['source_pts']): old_camera_vector(f)
                  for f in old['frame_cameras']
                  if f['camera_id'] == 'camera_main_long_side'
                  and int(f['source_pts']) in sources}
    if set(old_camera) != set(sources):
        raise ValueError('One of 14 actual-source main frames lacks old pose initializer')
    start = max(states) + 1
    splines, bindings, seed_rows = {}, [], []
    for ordinal, pts in enumerate(sorted(sources)):
        frame_key, control_key = start+2*ordinal, start+2*ordinal+1
        shot = 30_000+ordinal
        splines[shot] = StaticReferenceBasis(shot, pts, (control_key,))
        bindings.append(FrameBinding(pts, frame_key, shot, 'main_long_side',
                                     old_camera[pts].copy()))
        states[frame_key] = np.zeros(11)
        states[control_key] = np.zeros(11)
        seed_rows.append(dict(source_pts=pts, frame_key=frame_key,
                              control_key=control_key,
                              camera25_initializer=old_camera[pts].copy()))
    base = ContinuousRigCameraModel(
        splines, bindings, base_model.lens_base_keys,
        base_model.lens_slope_keys, base_model.lens_zoom_reference,
        mount_keys=base_model.mount_keys,
        offset_keys=base_model.offset_keys,
        offset_slope_keys=base_model.offset_slope_keys,
        brown_keys=base_model.brown_keys,
        brown_slope_keys=base_model.brown_slope_keys)
    model = QuadraticRigCameraModel.from_rig(
        base, lens_quadratic_keys=base_model.lens_quadratic_keys,
        brown_quadratic_keys=base_model.brown_quadratic_keys)
    seed_reference_initial(model, states, seed_rows)
    return model, seed_rows


def low_sources(scene, center, radius, *, directory=None, pts_set=None):
    result = {}
    directory = directory or TMP/'low_f2_outer_full_arc/measured_v4'
    pts_set = pts_set or (230353920, 230361600, 230368256)
    for pts in pts_set:
        path = directory/f'{pts}.npz'
        with np.load(path, allow_pickle=False) as packet:
            if (int(packet['pts']) != pts or
                    str(packet['source_sha256']) !=
                    'f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891'
                    or set(map(str, packet['physical_id'])) != {'F2_outer'}):
                raise ValueError('Low F2 source packet identity changed')
            uv = packet['observed_uv'].astype(float)
            sigma = packet['point_sigma_px'].astype(float)
        # These source packets used heterogeneous point sigmas.  The scalar
        # group median below only permits a local rank probe; it is not the
        # original low-end captured factor and no old-cost equivalence is claimed.
        result[pts] = F2SourceCurve(
            pts, f'low_measured_v4_{pts}', uv,
            world_curve(scene, {'kind': 'circle_outline', 'feature_id': 'F2'}),
            center, radius, float(np.median(sigma)))
    return result


def independent_fixed_factor(model, source, active, states, *, mode='sampled'):
    template = model.frames[source.pts].template25
    if mode == 'adaptive':
        measurements = [dict(x_interval_px=(float(x), float(x)),
                             y=float(y)) for x, y in source.uv]
        curve = AdaptiveProjectedPolyline(
            source.world0, measurements, tolerance_px=.01, closed=True,
            factor_family='original_reference_curve',
            no_central_reason='reference_curve_no_central_segment')
        raw, jac, _prime = adaptive_curve_callback(
            curve, template, (source.pts,), with_prime=True)
    else:
        raw, jac = _measurement_callback('curve', source.world0, source.uv,
                                         template, (source.pts,))
    robust, robust_jac = pointwise_huber(
        raw, jac, group_size=2, delta=2.*source.sigma_px)
    return adapt_factor((source.pts,), robust, robust_jac,
                        dict(residual_dim=2*len(source.uv),
                             sigma=source.factor_sigma), active, {}, model)


def verify_zero(model, states, sources, *, mode='sampled'):
    active = set(states)
    result = []
    for pts, source in sorted(sources.items()):
        nested = NestedF2Factor(model, source, active, {}, curve_mode=mode)
        fixed = independent_fixed_factor(model, source, active, states, mode=mode)
        if fixed is None or tuple(fixed[0]) != nested.camera_keys or fixed[3] != nested.options:
            raise AssertionError('Zero-release factor keys or source sigma changed')
        camera_values = [states[key] for key in nested.camera_keys]
        zero = np.zeros(11)
        before = fixed[1](camera_values)
        after = nested.residual([*camera_values, zero])
        old_jac = fixed[2](camera_values)
        new_jac = nested.jacobian([*camera_values, zero])
        delta_r = float(np.max(np.abs(before-after)))
        delta_j = max(float(np.max(np.abs(a-b)))
                      for a, b in zip(old_jac, new_jac[:-1]))
        if delta_r > 1e-12 or delta_j > 1e-9:
            raise AssertionError(f'Zero-world factor changed at {pts}: {delta_r}, {delta_j}')
        result.append(dict(pts=pts, source_uv=len(source.uv),
                           numerical_curve_mode=mode,
                           camera_keys=len(nested.camera_keys),
                           residual_max_abs=delta_r,
                           camera_jacobian_max_abs=delta_j,
                           world_jacobian_nonzero_columns=[int(k) for k in
                               np.flatnonzero(np.max(np.abs(new_jac[-1]), axis=0))]))
    return result


def local_rank(model, states, sources, companion=None):
    """Profile local camera columns with source F2 and optional same-frame marks.

    No camera prior, temporal factor, board ray or other world release is in
    this rank.  It must not be used to claim the complete graph is deficient.
    """
    companion = companion or {}
    pieces = []
    for pts, source in sorted(sources.items()):
        nested = NestedF2Factor(model, source, set(states), {},
                                curve_mode='sampled')
        values = [states[key] for key in nested.camera_keys]
        blocks = nested.jacobian([*values, np.zeros(11)])
        pieces.append((2*len(source.uv),
                       {k: b/nested.options['sigma'] for k, b in
                        zip(nested.camera_keys, blocks[:-1])},
                       blocks[-1][:, :3]/nested.options['sigma']))
        for row, xyz in companion.get(pts, []):
            uv = np.asarray(row['pixels_uv'], float)
            sigma = float(row['sigma_px'])
            raw, jac = _measurement_callback(
                'curve', xyz, uv, model.frames[pts].template25, (pts,))
            robust, robust_jac = pointwise_huber(
                raw, jac, group_size=2, delta=2.*sigma)
            adapted = adapt_factor(
                (pts,), robust, robust_jac,
                dict(residual_dim=2*len(uv), sigma=sigma*np.sqrt(len(uv))),
                set(states), {}, model)
            keys, _, derivative, options = adapted
            blocks = derivative([states[k] for k in keys])
            pieces.append((2*len(uv),
                           {k: b/options['sigma'] for k, b in zip(keys, blocks)},
                           np.zeros((2*len(uv), 3))))
    keys = sorted({key for _, jac, _ in pieces for key in jac})
    offsets = {key: i*11 for i, key in enumerate(keys)}
    total = sum(size for size, _, _ in pieces)
    camera = np.zeros((total, len(keys)*11))
    world = np.zeros((total, 3))
    cursor = 0
    for size, blocks, common in pieces:
        world[cursor:cursor+size] = common
        for key, block in blocks.items():
            col = offsets[key]
            camera[cursor:cursor+size, col:col+11] += block
        cursor += size
    norm = np.linalg.norm(camera, axis=0)
    camera = camera[:, norm > 0]/norm[norm > 0]
    if camera.shape[1]:
        q, r, piv = qr(camera, mode='economic', pivoting=True)
        diagonal = np.abs(np.diag(r))
        rank_tolerance = diagonal[0]*max(camera.shape)*np.finfo(float).eps*100
        camera_rank = int(np.sum(diagonal > rank_tolerance))
        reduced = world-q[:, :camera_rank]@(q[:, :camera_rank].T@world)
    else:
        camera_rank, reduced, rank_tolerance = 0, world, 0.
    singular = svdvals(reduced)
    raw_singular = svdvals(world)
    # Scale the numerical threshold to the *unprojected* world columns.
    # Scaling to the tiny projected values would label roundoff as rank 3.
    world_tolerance = (raw_singular[0]*max(reduced.shape)*
                       np.finfo(float).eps*100 if len(raw_singular) else 0.)
    return dict(rows=total, factors=len(pieces), camera_nodes=len(keys),
                active_camera_columns=int(camera.shape[1]),
                camera_numerical_rank=camera_rank,
                camera_rank_tolerance=float(rank_tolerance),
                raw_world_singular_values=raw_singular.tolist(),
                profiled_world_singular_values=singular.tolist(),
                profiled_over_raw_singular_ratio=(singular/raw_singular).tolist(),
                profiled_world_rank=int(np.sum(singular > world_tolerance)),
                world_rank_tolerance=float(world_tolerance),
                world_rank_tolerance_basis='raw-world-column scale times machine epsilon',
                world_column_labels=['F2_center_x_scene_per_100',
                                     'F2_center_y_scene_per_100',
                                     'F2_outer_radius_scene_per_100'])


def main_gauge_probe(model, states, sources, seed_rows, *, step=1e-5,
                     curve_mode='sampled'):
    """Check exact F2-only translation/scale gauges in values and Jacobians."""
    controls = {int(row['source_pts']): int(row['control_key']) for row in seed_rows}
    rows = []
    for pts, source in sorted(sources.items()):
        factor = NestedF2Factor(model, source, set(states), {},
                                curve_mode=curve_mode)
        camera_values = [states[k].copy() for k in factor.camera_keys]
        control_key = controls[pts]
        control_index = factor.camera_keys.index(control_key)
        scene0 = np.zeros(11)
        baseline = factor.residual([*camera_values, scene0])
        jac = factor.jacobian([*camera_values, scene0])
        camera_center = model.camera25(pts, states)[:3]
        world_center = np.r_[source.center_xy, source.world0[0, 2]]
        directions = [
            ('translation_x', np.r_[1., 0., 0.], np.r_[1., 0., 0.]),
            ('translation_y', np.r_[0., 1., 0.], np.r_[0., 1., 0.]),
            ('circle_scale', (camera_center-world_center)/source.outer_radius,
             np.r_[0., 0., 1.])]
        for name, control_direction, scene_direction in directions:
            changed = [v.copy() for v in camera_values]
            changed[control_index][:3] += step*control_direction
            scene = np.zeros(11)
            scene[:3] += step*scene_direction
            measured = factor.residual([*changed, scene])
            camera_only = factor.residual([*changed, scene0])
            world_only = factor.residual([*camera_values, scene])
            finite_gauge = np.linalg.norm((measured-baseline)/step)
            finite_camera = (camera_only-baseline)/step
            finite_world = (world_only-baseline)/step
            analytic_camera = jac[control_index][:, :3]@control_direction
            analytic_world = jac[-1][:, :3]@scene_direction
            j_gauge = (jac[control_index][:, :3]@control_direction
                       + jac[-1][:, :3]@scene_direction)
            rows.append(dict(pts=pts, gauge=name, curve_mode=curve_mode,
                             finite_residual_derivative_l2=float(finite_gauge),
                             analytic_jacobian_gauge_l2=float(np.linalg.norm(j_gauge)),
                             camera_jacobian_vs_finite_l2=float(np.linalg.norm(
                                 analytic_camera-finite_camera)),
                             world_jacobian_vs_finite_l2=float(np.linalg.norm(
                                 analytic_world-finite_world)),
                             finite_camera_plus_world_l2=float(np.linalg.norm(
                                 finite_camera+finite_world)),
                             world_jacobian_l2=float(np.linalg.norm(
                                 jac[-1][:, :3]@scene_direction)),
                             finite_residual_max_abs=float(np.max(np.abs(measured-baseline)))))
    return rows


def run(output: Path):
    scene, center, radius = _scene()
    main_sources_by_pts, companion, frames = main_sources(scene, center, radius)
    main_phase, main_programme, _, main_saved, main_meta = reconstruct(
        MAIN, state_name='accepted_states.npz', require_quadratic=True)
    low_programme, low_saved, low_prior, low_phase = low_time_model()
    if (main_phase['model_sha256'] != sha(SCENE)
            or low_phase['model_sha256'] != sha(SCENE)
            or main_phase['source_sha256'] != low_phase['source_sha256']):
        raise ValueError('Main and low rig do not share the exact source/scene')
    main_prior = MAIN/'prior_initializer_states.npz'
    low_prior_path = LOW/'prior_initializer_states.npz'
    prior_hashes = {str(main_prior): sha(main_prior),
                    str(low_prior_path): sha(low_prior_path)}
    main_states = {key: value.copy() for key, value in main_saved.items()}
    main_model, main_rows = add_main_reference_rig(
        main_programme, main_states, main_sources_by_pts, frames)
    low_curves = low_sources(scene, center, radius)
    confirmed_low = low_sources(
        scene, center, radius,
        directory=TMP/'low_f2_outer_full_arc/final_holdout_v4',
        pts_set=(230355712,))
    main_equivalence = verify_zero(main_model, main_states, main_sources_by_pts)
    low_equivalence = verify_zero(low_programme, low_saved, low_curves)
    main_adaptive_equivalence = verify_zero(
        main_model, main_states,
        {pts: main_sources_by_pts[pts] for pts in (71147520, 80547840)},
        mode='adaptive')
    confirmed_pair_adaptive_equivalence = verify_zero(
        low_programme, low_saved, confirmed_low, mode='adaptive')
    f2_only_main = local_rank(main_model, main_states, main_sources_by_pts)
    f2_plus_companion_main = local_rank(
        main_model, main_states, main_sources_by_pts, companion)
    f2_only_low = local_rank(low_programme, low_saved, low_curves)
    gauge = main_gauge_probe(main_model, main_states,
                             main_sources_by_pts, main_rows)
    adaptive_gauge = main_gauge_probe(
        main_model, main_states,
        {pts: main_sources_by_pts[pts] for pts in (71147520, 80547840)},
        main_rows, curve_mode='adaptive')
    if any(sha(Path(path)) != digest for path, digest in prior_hashes.items()):
        raise AssertionError('Existing camera prior target file changed')
    report = dict(
        schema='rinkspace.shared-f2-nested-local-probe.v1',
        scope='read-only factor and local rank; no nonlinear solve or full graph',
        source_sha256=main_phase['source_sha256'], scene_sha256=sha(SCENE),
        main_actual_source_packet_sha256=sha(OBSERVATIONS['main_long_side']),
        main_accepted_state_sha256=main_meta['provenance']['state_sha256'],
        low_saved_state_sha256=sha(LOW/'states.npz'),
        unchanged_prior_target_sha256=prior_hashes,
        scene_key=F2_SCENE_KEY, common_geometry_variables=[
            'F2 center x/100', 'F2 center y/100', 'F2 outer radius delta/100'],
        fixed_scene_geometry=dict(center_xy=center.tolist(), outer_radius=radius),
        main_new_reference_nodes=len(main_rows)*2,
        main_f2_factor_count=len(main_sources_by_pts),
        main_f2_original_uv=sum(len(s.uv) for s in main_sources_by_pts.values()),
        low_existing_source_factor_count=len(low_curves),
        low_source_uv=sum(len(s.uv) for s in low_curves.values()),
        user_confirmed_same_circle_pair=dict(
            main_source_pts=71147520, main_source_time='01:17:12.000',
            low_source_pts=230355712, low_source_time='04:09:57.117',
            low_historical_source_uv=len(confirmed_low[230355712].uv),
            low_packet_sha256=sha(TMP/'low_f2_outer_full_arc/final_holdout_v4/230355712.npz'),
            role='identity bridge and zero-release check; excluded from local training rank'),
        main_zero_release_equivalence=main_equivalence,
        main_adaptive_zero_release_equivalence=main_adaptive_equivalence,
        confirmed_pair_low_adaptive_zero_release_equivalence=(
            confirmed_pair_adaptive_equivalence),
        low_zero_release_equivalence_under_reference_curve_callback=low_equivalence,
        local_rank=dict(main_f2_only=f2_only_main,
                        main_f2_plus_other_bound_source_curves=f2_plus_companion_main,
                        low_f2_only=f2_only_low),
        main_f2_only_gauge=gauge,
        main_adaptive_f2_only_gauge=adaptive_gauge,
        important_limitations=[
            'Low source scalar group sigma is the median of heterogeneous original point sigmas; no equivalence to the captured low-end original objective is claimed.',
            'The 14 additional main cameras use old camera25 only to initialize new pose/centre nodes; no old camera component is a measurement or prior target.',
            'Local rank omits complete programme board, advertisement, ice-line, temporal and camera-prior factors and releases no other world geometry.',
            'Current scene units are conditional and are not verified metres.',
            'These actual source curves and existing low historical packets are development evidence, not fresh blind validation.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                 allow_nan=False), encoding='utf8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(dict(report=str(args.output.resolve()),
                          main_f2_uv=result['main_f2_original_uv'],
                          main_zero_max=max(r['residual_max_abs'] for r in
                                            result['main_zero_release_equivalence']),
                          local_rank=result['local_rank']), ensure_ascii=False))
