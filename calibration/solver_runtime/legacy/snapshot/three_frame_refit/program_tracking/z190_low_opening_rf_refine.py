"""Refine actual low-cut source states within the standing Z190 model.

Default mode freezes completed C/PP/k1. Explicit user-position mode freezes
the annotated Cxy and fits shared PP/k1; optional standing height uses one
soft working prior. New source edges bind to the actual finite AdGeometry
render inventory. No warp, interpolation, old hardware or geometry fit.
"""
from __future__ import annotations
from calibration.solver_runtime.config import data_module_dir as _runtime_data_module_dir

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import sys

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from main7_source_bundle_candidate import Bundle, huber_cost
from main7_source_camera_candidate import sha
from render_result import advertisement_boundary_paths
from z190_continuous_source_bundle import StandingSourceBundle, SCENE_UNIT_M, HARD_SCENE_SHA, SOURCE_SHA
from z190_physical_camera import PhysicalCameraModel, CanonicalView, ViewState
from z190_source_camera_diagnostic import source_errors, jsonable, absolute_value_jac15
from z190_low_source_authority import admit_low_source_rows, split_rows, PENDING


def convert_row(row):
    result = dict(row)
    for key in ('uv', 'world', 'sigma', 'normal', 'normal_interval'):
        if key in result:
            result[key] = np.asarray(result[key], float)
    return result


def opening_rows(packet_path, scene, aliases):
    """Semantic manual UV observations; physical geometry is always external."""
    packet = json.loads(packet_path.read_text(encoding='utf8'))
    if packet['scene_sha256'] != HARD_SCENE_SHA or packet['source_sha256'] != SOURCE_SHA:
        raise ValueError('Opening source packet is not bound to unchanged hard scene')
    inventory = {(p['number'], p['role']): p['xyz'] for p in advertisement_boundary_paths(scene)}
    rows = []
    for original in packet['rows']:
        row = dict(original)
        p = int(row.get('source_native_pts', row['pts']))
        if p not in aliases:
            raise ValueError('Unknown opening source exposure PTS')
        if row.get('point_correspondence') or 'world' in row:
            raise ValueError('Opening edges require semantic finite-curve binding; supplied world/corners refused')
        if row['physical_identity']['kind'] == 'physical_advertisement_external_border':
            identity = row['physical_identity']
            number, role = int(identity['ad_id']), str(identity['geometry_role'])
            row['world'] = inventory[number, role].copy()
            row['id'] = f'ad{number}:{role}'
            locator = f'advertisement_boundary_paths(scene): number={number}, role={role}'
        elif row['physical_identity']['kind'] == 'physical_board_upper_edge':
            row['world'] = np.asarray(scene['board_surface']['top_edge_xyz'], float)
            if row['physical_identity']['board_side'] != 'upper_long_side':
                raise ValueError('Only actual identified upper long board side supported')
            # Keep the finite vertices of the identified long side. No full
            # perimeter nearest-branch ambiguity or generated corner bridge.
            ymax = float(row['world'][:, 1].max())
            row['world'] = row['world'][np.isclose(row['world'][:, 1], ymax, rtol=0., atol=1e-9)]
            row['world'] = row['world'][np.argsort(row['world'][:, 0])]
            if len(row['world']) < 2:
                raise ValueError('No finite upper long board segment in hard scene')
            row['id'] = 'board_top'
            locator = 'board_surface.top_edge_xyz actual finite upper_long_side vertices at y=max'
        else:
            raise ValueError('Unmodeled opening source identity')
        row['source_native_pts'], row['pts'] = p, aliases[p]
        row['uv'] = np.asarray(row['uv'], float)
        row['sigma'] = np.broadcast_to(np.asarray(row['sigma_px_assumed'], float), (len(row['uv']),)).copy()
        row['original_source_split'] = row['split']
        if row['split'] == 'new_direct_source_fit':
            row['split'] = 'fit'
        elif row['split'] == 'new_direct_source_held_neighbour_time':
            row['split'] = 'held_neighbour_time'
        else:
            raise ValueError('Unrecognized manually specified opening split')
        if row['uv'].ndim != 2 or row['uv'].shape[1] != 2 or not np.isfinite(row['uv']).all():
            raise ValueError('Opening UV must be finite actual source samples')
        row['world_scene_binding'] = dict(scene_sha256=HARD_SCENE_SHA, geometry_locator=locator,
            actual_solved_corners_same_as_renderer=True, historical_fitted_boundary_curves_used=False,
            source_uv_changed=False)
        row['origin'] = str(packet_path)
        rows.append(convert_row(row))
    return rows, packet


def solve_frozen_pivot(bundle, q0, output, *, fit_shared_optics=False, fit_height=False):
    """Same source LM; Cxy is immutable, optionally fit Cz and shared PP/k1.

    Callers supply the actual completed source states. In user-pivot mode the
    separate evidence binding must authorize the fixed XY change beforehand.
    """
    q = q0.copy()
    free = np.arange(2 if fit_height else 3 if fit_shared_optics else 6, len(q))
    n = bundle.source_residual_rows
    history, damping = [], 1e-3
    started = time.perf_counter()
    def cost(r, robust):
        return huber_cost(r[:n], robust)+.5*float(r[n:]@r[n:])
    for robust in (False, True):
        for iteration in range(45):
            r, fullj = bundle.evaluate(q)
            j = fullj[:, free]
            before = cost(r, robust)
            weight = np.ones(len(r))
            if robust:
                weight[:n] = np.minimum(1., 2./np.maximum(np.abs(r[:n]), 1e-12))
            jw = j.multiply(np.sqrt(weight)[:, None])
            normal, gradient = (jw.T@jw).tocsc(), j.T@(weight*r)
            diagonal = np.maximum(normal.diagonal(), 1.)
            accepted, reason = False, ''
            for trial in range(12):
                step = spsolve(normal+diags(damping*diagonal), -gradient, permc_spec='MMD_AT_PLUS_A')
                trialq = q.copy(); trialq[free] += step
                try:
                    if (not np.isfinite(trialq).all() or
                            np.any((trialq[9::4] < np.log(100)) | (trialq[9::4] > np.log(50000)))):
                        raise ValueError('Numerical focal domain')
                    after = cost(bundle.evaluate(trialq, False), robust)
                    if after < before:
                        q, accepted = trialq, True
                        damping = max(damping*.3, 1e-12)
                        break
                    reason = 'objective_not_lower'
                except (ValueError, RuntimeError) as exc:
                    reason = str(exc)[:160]
                damping *= 10.
            mapped = j@step
            source_step = float(np.max(np.abs(mapped[:n]*bundle.native_residual_sigma)))
            prior_step = float(np.max(np.abs(mapped[n:])))
            history.append(dict(phase='huber' if robust else 'linear', iteration=iteration,
                cost=before, next_cost=after if accepted else before, accepted=accepted,
                damping=damping, trials=trial+1, failure_reason=None if accepted else reason,
                maximum_linearized_source_step_px=source_step, maximum_prior_step_sigma=prior_step))
            (output/'bundle_iterations.json').write_text(json.dumps(history, indent=2), encoding='utf8')
            print(json.dumps(history[-1]), flush=True)
            if not accepted or (source_step < .05 and prior_step < .001):
                break
    fixed = 2 if fit_height else 3 if fit_shared_optics else 6
    if not np.array_equal(q[:fixed], q0[:fixed]):
        raise ValueError('Frozen installation changed')
    return q, history, time.perf_counter()-started


def solve_rf_only(bundle, q0, output):
    return solve_frozen_pivot(bundle, q0, output, fit_shared_optics=False)


def finalize_saved_user_binding(output: Path, evidence_path: Path):
    """Refresh provenance only after unrelated user-record amendments.

    The low pivot/source/scene must be exactly unchanged. Archive numeric
    inputs replay exactly; the solve-start evidence hash stays traceable.
    """
    evidence_path = evidence_path.resolve()
    evidence = json.loads(evidence_path.read_text(encoding='utf8'))
    oldpath = output/'physical_dense_low_rf_candidate_unaccepted.npz'
    model, native, aliases, views = PhysicalCameraModel.load_npz(oldpath)
    identifier, calibration = next(iter(model.calibrations.items()))
    low = evidence['low_end']
    if (evidence['scene_sha256'] != HARD_SCENE_SHA or evidence['source_sha256'] != SOURCE_SHA or
            not np.array_equal(np.asarray(calibration.pivot_xyz[:2]), low['nominal_pivot_xy_scene_units']) or
            evidence['metres_per_scene_unit'] != SCENE_UNIT_M or low['installation_id'] != 2):
        raise ValueError('Final evidence changed a solved low installation/source/scene assumption')
    png = Path(evidence['annotated_image']['path'])
    if sha(png) != evidence['annotated_image']['sha256']:
        raise ValueError('User annotated PNG changed')
    settings = dict(calibration.field_settings)
    previous = settings['installation_position_binding']
    current = dict(user_json_path=str(evidence_path), user_json_sha256=sha(evidence_path),
        user_png_path=str(png.resolve()), user_png_sha256=sha(png),
        nominal_pivot_xy_scene_units=low['nominal_pivot_xy_scene_units'],
        pivot_policy='fixed_user_approximate_nominal_outside_board', position_accuracy=low['position_accuracy'])
    settings['installation_position_binding'] = current
    updated = PhysicalCameraModel({identifier:replace(calibration, field_settings=settings)})
    destination = output/'physical_dense_low_user_pivot_candidate_unaccepted.npz'
    updated.save_npz(destination, native, aliases, views, source_sha256=SOURCE_SHA, time_base='1/15360')
    if not np.array_equal(model.exposure_arrays(native, aliases, views)['camera25'],
                          updated.exposure_arrays(native, aliases, views)['camera25']):
        raise ValueError('Provenance finalization changed actual numerical cameras')
    record = dict(schema='rinkspace.z190-low-user-pivot-final-binding.v1',
        user_record_binding_at_solve_start=previous, final_user_record_binding=current,
        metadata_amendment_reason='Final user record revalidated; low nominal XY/source/scene unchanged; numerical state unchanged on metadata export',
        low_pivot_xy_unchanged=True, camera_numeric_arrays_unchanged=True, complete_source_solve_not_rerun=True,
        latest_physical_archive=str(destination), latest_physical_archive_sha256=sha(destination),
        original_solved_physical_archive=str(oldpath), original_solved_physical_archive_sha256=sha(oldpath))
    (output/'final_user_position_binding.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf8')
    manifest = json.loads((output/'input_manifest.json').read_text(encoding='utf8'))
    manifest['user_installation_constraint']['path'] = str(evidence_path)
    manifest['user_installation_constraint']['sha256'] = current['user_json_sha256']
    manifest['final_user_position_binding'] = record
    (output/'handoff_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    return record


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    freeze_calibration = bool(getattr(args, 'freeze_completed_calibration', False))
    if freeze_calibration and (args.soft_height or not args.user_tripod_positions):
        raise ValueError('Frozen completed low calibration requires retained user-position evidence and no height fit')
    if args.soft_height and (not args.user_tripod_positions or set(args.cuts) != {3,5,8}):
        raise ValueError('Soft standing height requires fixed-user XY and complete low3/5/8 source graph')
    if sha(args.scene) != HARD_SCENE_SHA:
        raise ValueError('Hard scene changed')
    cv2.setNumThreads(1)
    scene = json.loads(args.scene.read_text(encoding='utf8'))
    model, native, aliases, fullviews = PhysicalCameraModel.load_npz(args.warm_standing)
    if len(model.calibrations) != 1:
        raise ValueError('Use completed low installation archive only')
    identifier, calibration = next(iter(model.calibrations.items()))
    userconstraint = None
    if args.user_tripod_positions:
        args.user_tripod_positions = args.user_tripod_positions.resolve()
        evidence = json.loads(args.user_tripod_positions.read_text(encoding='utf8'))
        if (evidence['authority'] != 'current_user_direct_annotation' or
                evidence['scene_sha256'] != HARD_SCENE_SHA or evidence['source_sha256'] != SOURCE_SHA or
                evidence['low_end']['installation_id'] != 2 or evidence['metres_per_scene_unit'] != SCENE_UNIT_M):
            raise ValueError('User installation evidence does not match this low model/source/scale')
        xy = evidence['low_end']['nominal_pivot_xy_scene_units']
        if freeze_calibration and not np.array_equal(np.asarray(xy), np.asarray(calibration.pivot_xyz[:2])):
            raise ValueError('Frozen completed calibration does not match the retained user nominal XY')
        position_binding = dict(user_json_path=str(args.user_tripod_positions), user_json_sha256=sha(args.user_tripod_positions),
            user_png_path=evidence['annotated_image']['path'], user_png_sha256=evidence['annotated_image']['sha256'],
            nominal_pivot_xy_scene_units=list(map(float, xy)),
            pivot_policy='fixed_user_approximate_nominal_outside_board',
            position_accuracy=evidence['low_end']['position_accuracy'])
        settings = dict(calibration.field_settings)
        settings['installation_position_binding'] = position_binding
        calibration = replace(calibration, pivot_xyz=(*map(float, xy), float(calibration.pivot_xyz[2])), field_settings=settings)
        userconstraint = dict(path=str(args.user_tripod_positions), sha256=sha(args.user_tripod_positions),
            annotated_image=evidence['annotated_image'], original_user_statement=evidence['user_text'],
            nominal_pivot_xyz_scene_units=calibration.pivot_xyz,
            nominal_position_status=evidence['low_end']['position_accuracy'],
            no_user_tolerance_inferred=True, pivot_xy_variables_optimized=0,
            height_fit_policy='fixed_saved_physical_height' if freeze_calibration else 'soft_1_6m_working_prior' if args.soft_height else 'fixed_1_6m_working_height',
            optical_lever_and_axial_fixed_zero=True)
    timeline = json.loads(_runtime_data_module_dir(__file__) / 'source_shots.json'.read_text(encoding='utf8'))
    cut_for = {int(p): cut for cut in args.cuts for p in native
               if timeline['shots'][cut]['start_pts'] <= p < timeline['shots'][cut]['end_pts']}
    stamps = sorted({aliases[p] for p in cut_for})
    observations = {p: [] for p in stamps}
    original = json.loads(args.original_rows.read_text(encoding='utf8'))
    original_rows = [convert_row(r) for r in original['fit_rows']+original['held_rows']+original.get('identity_pending_rows', []) if int(r['pts']) in observations]
    original_rows, admission = admit_low_source_rows(original_rows, scene,
        confirmations_path=getattr(args, 'source_confirmations', None),
        input_binding=dict(path=str(args.original_rows.resolve()), sha256=sha(args.original_rows)),
        raw_source_base=getattr(args, 'raw_source_base', None), timeline=timeline)
    admission_path = args.output/'source_authority_admission.json'
    admission_path.write_text(json.dumps(jsonable(admission), ensure_ascii=False, indent=2), encoding='utf8')
    settings = dict(calibration.field_settings)
    settings['source_authority_admission'] = dict(path=str(admission_path), sha256=sha(admission_path),
        unconfirmed_ice_proposals_excluded_from_fit=True, pending_ice_rows=admission['pending_ice_rows'],
        source_confirmation_manifest=admission['confirmation_manifest'])
    calibration = replace(calibration, field_settings=settings)
    added, source_packet = opening_rows(args.opening_source, scene, aliases)
    outside_window = [dict(native_pts=r['source_native_pts'], id=r['id'], split=r['split'])
                      for r in added if r['pts'] not in observations]
    added = [r for r in added if r['pts'] in observations]
    for row in original_rows+added:
        if row['pts'] not in observations:
            raise ValueError('Opening source lies outside requested complete cuts')
        if row['split'] == 'fit':
            observations[row['pts']].append(row)
    pairs, source_manifests = [], []
    for cut in args.cuts:
        path = args.motion_audit/f'low{cut}_original_row_person_status.npz'
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k].copy() for k in z.files}
        rawpath = Path(f'D:/CodexData/RinkSpace_program_tracking_tmp/motion/motion_shot_{cut:02d}.npz')
        with np.load(rawpath, allow_pickle=False) as z:
            if (not np.array_equal(data['native_pair_pts'], z['pair_pts']) or
                not np.array_equal(data['xy0'], z['xy0'][data['original_global_row_indices']]) or
                not np.array_equal(data['xy1'], z['xy1'][data['original_global_row_indices']])):
                raise ValueError('Audited original pixel/row identity changed')
        selected, counts = [], []
        for i, (a, b) in enumerate(data['native_pair_pts']):
            lo, hi = map(int, data['offsets'][i:i+2])
            take = np.arange(lo, hi)
            if args.motion_profile == 'source-person-mask-excluded':
                take = take[data['source_person_status'][take] == 0]
            if len(take):
                pairs.append((aliases[int(a)], aliases[int(b)], data['xy0'][take].astype(float), data['xy1'][take].astype(float)))
            counts.append(len(take)); selected.append(data['original_global_row_indices'][take])
        rowpath = args.output/f'low{cut}_retained_original_source_rows.npz'
        np.savez_compressed(rowpath, native_pair_pts=data['native_pair_pts'], offsets=np.r_[0, np.cumsum(counts)],
            original_global_row_indices=np.concatenate(selected), quality_policy=args.motion_profile,
            selection_uses_camera_or_residual=False)
        source_manifests.append(dict(cut=cut, rawpath=str(rawpath), rawsha256=sha(rawpath),
            selected_rows=str(rowpath), selected_sha256=sha(rowpath), matches=sum(counts),
            excluded_person_rows=int(np.sum(data['source_person_status'] == 1)) if args.motion_profile != 'all-lk' else 0,
            unmasked_static_identity='unknown, no geometric surface identity inferred'))
    base = Bundle(stamps, observations, pairs)
    wide = min(np.exp(v.view.state4()[3])*1920. for v in fullviews.values())
    bundle = StandingSourceBundle(base, calibration, scene_unit_m=SCENE_UNIT_M, prior_wide_focal_px=wide,
                                  fixed_height=not args.soft_height,
                                  fixed_height_scene_units=calibration.pivot_xyz[2] if freeze_calibration else None)
    q0 = np.r_[bundle.initial_shared(), np.concatenate([np.r_[fullviews[p].view.state4()[:3],
                fullviews[p].view.state4()[3]+np.log(1920.)] for p in stamps])]
    if args.soft_height:
        q0[2] = calibration.pivot_xyz[2]/100.
    # Never trust an implicit recovered shape: completed physical archive is
    # the sole numeric warm start and must roundtrip to exactly the same K/C/D.
    for p in stamps:
        checkstart = 3 if userconstraint else 0
        if not np.allclose(bundle.camera(q0, p)[checkstart:], model.camera25(fullviews[p])[checkstart:], atol=1e-12, rtol=0.):
            raise ValueError('Warm completed Z190 state changed on binding')
    bound = split_rows(original_rows+added)
    (args.output/'absolute_source_rows.json').write_text(json.dumps(jsonable(bound), ensure_ascii=False, indent=2), encoding='utf8')
    manifest = dict(schema='rinkspace.z190-low-opening-rf-input.v1', camera_model='Sony PXW-Z190',
        warm_completed_physical_archive=str(args.warm_standing), warm_sha256=sha(args.warm_standing),
        scene_sha256=HARD_SCENE_SHA, metres_per_scene_unit=SCENE_UNIT_M, scene_variables_optimized=0,
        frozen_installation_parameters=['Cx','Cy','Cz','PPx','PPy','k1'] if freeze_calibration else (['Cx','Cy'] if args.soft_height else ['Cx','Cy','Cz']) if userconstraint else ['Cx','Cy','Cz','PPx','PPy','k1'], initial_state6=q0[:6],
        optimized_parameters=(['Cz'] if args.soft_height else [])+(['PPx','PPy','k1'] if userconstraint and not freeze_calibration else [])+['R3','effective_logf_px'], complete_cuts=args.cuts,
        height_prior=(dict(target_m=1.6,sigma_m=.18,applied_once=True,
            meaning='unmeasured standing optical-height/floor-to-ice working prior, not user survey') if args.soft_height else None),
        user_installation_constraint=userconstraint, run_command_argv=sys.argv,
        source_exposures=len(stamps), matches=len(base.a), source_residual_rows=bundle.source_residual_rows,
        source_motion=source_manifests, opening_source=str(args.opening_source), opening_source_sha256=sha(args.opening_source),
        original_source_rows=str(args.original_rows), original_source_rows_sha256=sha(args.original_rows),
        source_authority_admission=admission, frozen_completed_calibration=freeze_calibration,
        pending_identity_rows_are_not_validation_holdouts=True,
        opening_source_rows_outside_requested_complete_cuts_not_used=outside_window,
        historical_fitted_boundary_curves_used=False, no_camera_warp=True, no_missing_pose_interpolation=True,
        source_uv_unchanged=True, accuracy_accepted=False, complete_cut_accepted=False)
    (args.output/'input_manifest.json').write_text(json.dumps(jsonable(manifest), ensure_ascii=False, indent=2), encoding='utf8')
    checks = []
    for p in sorted({row['pts'] for row in added if row['split'] == 'fit'}):
        rows = [row for row in added if row['pts'] == p and row['split'] == 'fit']
        _, raw = absolute_value_jac15(bundle.camera(q0, p), rows)
        expected = np.c_[raw[:, 3:6], raw[:, 6]+raw[:, 7]]
        off = 6+4*bundle.index[p]
        for k in range(4):
            plus, minus = q0.copy(), q0.copy()
            plus[off+k] += 1e-7; minus[off+k] -= 1e-7
            numeric = (absolute_value_jac15(bundle.camera(plus, p), rows, False)-
                       absolute_value_jac15(bundle.camera(minus, p), rows, False))/2e-7
            checks.append(dict(pts=p, view_column=k, actual_source_pixels=sum(len(row['uv']) for row in rows),
                relative_l2_difference=float(np.linalg.norm(numeric-expected[:, k])/max(1., np.linalg.norm(numeric)))))
        if args.soft_height:
            plus, minus = q0.copy(), q0.copy()
            plus[2] += 1e-7; minus[2] -= 1e-7
            numeric = (absolute_value_jac15(bundle.camera(plus, p), rows, False)-
                       absolute_value_jac15(bundle.camera(minus, p), rows, False))/2e-7
            checks.append(dict(pts=p, shared_column=2, meaning='Cz/100', actual_source_pixels=sum(len(row['uv']) for row in rows),
                relative_l2_difference=float(np.linalg.norm(numeric-raw[:,2])/max(1., np.linalg.norm(numeric)))))
    if max(row['relative_l2_difference'] for row in checks) > 2e-4:
        raise ValueError('Actual opening source RF derivative mismatch: '+json.dumps(checks))
    (args.output/'actual_opening_rf_derivatives.json').write_text(json.dumps(checks, indent=2), encoding='utf8')
    print(json.dumps(jsonable(dict(stage='prepared_rf_only', exposures=len(stamps), source_matches=len(base.a),
        new_source_rows=len(added), frozen_state6=q0[:6]))), flush=True)
    if not args.run:
        return manifest
    q, history, seconds = solve_frozen_pivot(bundle, q0, args.output, fit_shared_optics=bool(userconstraint) and not freeze_calibration,
                                            fit_height=args.soft_height)
    np.savez_compressed(args.output/'optimized_state6rf.npz', q=q)
    views = dict(fullviews)
    for p in stamps:
        off = 6+4*bundle.index[p]
        views[p] = CanonicalView(identifier, ViewState(tuple(q[off:off+3]), float(q[off+3]-np.log(1920.))))
    calibration = bundle.current_calibration(q)
    model = PhysicalCameraModel({identifier:calibration})
    model.save_npz(args.output/'physical_dense_low_rf_candidate_unaccepted.npz', native, aliases, views,
                   source_sha256=SOURCE_SHA, time_base='1/15360')
    expanded = model.exposure_arrays(native, aliases, views)
    np.savez_compressed(args.output/'candidate_cameras.npz', **expanded, model_sha256=HARD_SCENE_SHA,
        source_sha256=SOURCE_SHA, status=np.full(len(native), 'z190_source_rf_candidate_unvalidated'))
    report = dict(schema='rinkspace.z190-low-opening-rf-report.v1', seconds=seconds, history=history,
        C_scene=calibration.pivot_xyz, PP_px=[q[3]*1920., q[4]*1080.], k1=q[5],
        fixed_user_pivot_xy_unchanged=np.array_equal(q[:2], q0[:2]), fixed_pivot_unchanged=np.array_equal(q[:3], q0[:3]), shared_state_unchanged=np.array_equal(q[:6], q0[:6]),
        height_assumed_m=float(q[2]*100.*SCENE_UNIT_M), height_prior_standardized_residual=float((q[2]*100.*SCENE_UNIT_M-1.6)/.18) if args.soft_height else None,
        user_installation_constraint=userconstraint, accuracy_accepted=False, complete_cut_accepted=False,
        source_errors=[dict(native_pts=r.get('source_native_pts', r['pts']), id=r['id'], split=r['split'],
            start=source_errors(bundle.camera(q0, r['pts']), [r]), result=source_errors(bundle.camera(q, r['pts']), [r]))
            for r in original_rows+added])
    (args.output/'report.json').write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(stage='finished_frozen_pivot_source', seconds=seconds, fixed_user_xy_unchanged=True,
        shared_optics_fitted=bool(userconstraint) and not freeze_calibration, soft_height_fitted=args.soft_height,
        height_assumed_m=report['height_assumed_m'], PP_px=report['PP_px'], k1=q[5])), flush=True)
    if userconstraint:
        finalize_saved_user_binding(args.output, args.user_tripod_positions)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('scene','warm-standing','original-rows','motion-audit','opening-source','output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--cuts', type=int, nargs='+', default=[5,8], choices=[3,5,8])
    p.add_argument('--motion-profile', choices=['all-lk','source-person-mask-excluded'], default='all-lk')
    p.add_argument('--user-tripod-positions', type=Path, help='Freeze the user nominal low XYZ and solve shared PP/k1')
    p.add_argument('--soft-height', action='store_true', help='Fix user XY; fit Cz with one 1.6m, sigma .18m working prior and shared PP/k1')
    p.add_argument('--freeze-completed-calibration', action='store_true', help='Keep saved per-installation Cxyz/PP/k1 exactly; source-authority repair changes only actual-exposure R/f')
    p.add_argument('--source-confirmations', type=Path, help='Exact-row direct user ice-identity records; absent ice identities remain unfitted_identity_pending')
    p.add_argument('--raw-source-base', type=Path, help='Optional original low shotN_actual_source library for bit-exact raw proposal tracing')
    p.add_argument('--run', action='store_true')
    run(p.parse_args())
