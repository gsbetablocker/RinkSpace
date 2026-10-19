"""Fit all low-end exposures with the fixed user tripod and nonzero lever.

Uses the exact previously admitted 96 fit / 7 held / 10 pending rows and
435008 retained original motion observations. Unknown material stays an
unknown positive-depth nuisance; no inferred ice/board assignment is made.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np
from main7_source_bundle_candidate import Bundle
from z190_physical_camera import PhysicalCameraModel, CanonicalView
from z190_continuous_source_bundle import HARD_SCENE_SHA, SOURCE_SHA, SCENE_UNIT_M
from z190_low_opening_rf_refine import convert_row
from z190_source_camera_diagnostic import source_errors
from z190_moving_pivot_source_bundle import MovingPivotSourceBundle, solve_moving_bundle
from z190_main_nonzero_mount_full_source import moving_motion_wrapper, derivative_checks, save_depth_trace
from z190_main_zoom_k1_source_information import read, write, ref, CURRENT_PHYSICAL, CURRENT
from continuous_central_brown4_optin import enable_projection_backend

IDENTIFIER = 'okada_sp_installation_2_conditional_z190'
SOURCE_DIR = CURRENT/'z190_low_source_authority_rf_v1'
SEED = CURRENT/'z190_full_programme_nonzero_mount_initialization_v1/installation_2/physical_nonzero_mount_initialization.npz'


def numeric_identity(row):
    fields = ('pts', 'id', 'split', 'point_correspondence')
    header = tuple(row.get(k) for k in fields)
    arrays = tuple(hashlib.sha256(np.asarray(row[k], np.float64).tobytes()).hexdigest()
                   if k in row else None for k in ('uv', 'world', 'sigma', 'normal', 'normal_interval'))
    return header+arrays


def source_changes(rows, baseline_model, baseline_views, fitted_model, fitted_views, aliases):
    result = []
    for split, original in rows:
        row = convert_row(original)
        p = aliases[int(row['pts'])]
        result.append(dict(pts=int(row.get('source_native_pts', row['pts'])), canonical_pts=p,
            id=row['id'], split=split, source_split=row['split'], origin=row.get('origin'),
            before=source_errors(baseline_model.camera25(baseline_views[p]), [row])[0],
            candidate=source_errors(fitted_model.camera25(fitted_views[p]), [row])[0]))
    return result


def prepare(output):
    manifest_path = SOURCE_DIR/'input_manifest.json'
    manifest = read(manifest_path)
    scene_path = Path(manifest['run_command_argv'][manifest['run_command_argv'].index('--scene')+1])
    if ref(scene_path)['sha256'] != HARD_SCENE_SHA:
        raise ValueError('Low hard scene changed')
    baseline, all_native, all_aliases, all_views = PhysicalCameraModel.load_npz(CURRENT_PHYSICAL)
    seed, native, aliases, views = PhysicalCameraModel.load_npz(SEED)
    cal = seed.calibrations[IDENTIFIER]
    if not np.array_equal(np.asarray(cal.lever_xyz)*SCENE_UNIT_M, [0., -.2, .1]):
        raise ValueError('Latest nonzero user initialization changed')
    if aliases != {p: a for p, a in all_aliases.items() if all_views[a].installation_id == IDENTIFIER}:
        raise ValueError('Low actual exposure identity changed')
    stamps = sorted(views)
    if (len(native), len(stamps)) != (2219, 1851):
        raise ValueError('Incomplete low programme source scope')
    rows_path = CURRENT/'z190_low_source_authority_v1/source_authority_rows.json'
    original_path = SOURCE_DIR/'absolute_source_rows.json'
    packet, exact = read(rows_path), read(original_path)
    from collections import Counter
    roles = ('fit_rows', 'held_rows', 'identity_pending_rows')
    if [len(packet[k]) for k in roles] != [96, 7, 10]:
        raise ValueError('Latest low admission changed')
    for role in roles:
        if Counter(map(numeric_identity, packet[role])) != Counter(map(numeric_identity, exact[role])):
            raise ValueError('Low role/source geometry differs from the completed current source graph')
    rows = [(role, r) for role in roles for r in packet[role]]
    observed = {p: [] for p in stamps}
    for r in packet['fit_rows']:
        if r['split'] != 'fit':
            raise ValueError('Nonfit low identity admitted')
        observed[aliases[int(r['pts'])]].append(convert_row(r))
    pairs, bindings = [], []
    pair_cuts = []
    for item in manifest['source_motion']:
        rawpath, indexpath = Path(item['rawpath']), Path(item['selected_rows'])
        if ref(rawpath)['sha256'] != item['rawsha256'] or ref(indexpath)['sha256'] != item['selected_sha256']:
            raise ValueError('Original low motion source binding changed')
        with np.load(rawpath, allow_pickle=False) as raw, np.load(indexpath, allow_pickle=False) as idx:
            if not np.array_equal(raw['pair_pts'], idx['native_pair_pts']) or str(idx['quality_policy']) != 'all-lk':
                raise ValueError('Low original pair order/selection changed')
            selected = idx['original_global_row_indices']
            aa, bb = raw['xy0'][selected].astype(float), raw['xy1'][selected].astype(float)
            offsets = idx['offsets']
            if len(aa) != item['matches'] or int(offsets[-1]) != len(aa):
                raise ValueError('Retained low motion count changed')
            for i, (pa, pb) in enumerate(idx['native_pair_pts']):
                lo, hi = map(int, offsets[i:i+2])
                if hi > lo:
                    pairs.append((aliases[int(pa)], aliases[int(pb)], aa[lo:hi], bb[lo:hi]))
                    pair_cuts.append(dict(cut=item['cut'], original_pair_index=i,
                        native_pair_pts=[int(pa), int(pb)], canonical_pair_pts=[aliases[int(pa)], aliases[int(pb)]],
                        retained_row_start=lo, retained_row_end_exclusive=hi))
        bindings.append(item)
    base = Bundle(stamps, observed, pairs)
    if len(base.a) != 435008 or sum(len(r) for _, r in base.abs_groups) != 96:
        raise ValueError('Complete low source graph count changed')
    settings = dict(cal.field_settings)
    settings['active_mount_parameters'] = dict(settings['active_mount_parameters'],
        stage='full_low_source_fit', source_rows=ref(rows_path), historical_full_source=ref(manifest_path))
    settings['reference_optical_height_prior'] = dict(reference_exposure_pts=stamps[0],
        target_above_ice_working_m=1.6, sigma_m=.18, operator_ground_minus_ice_unknown=True,
        pivot_height_is_not_lens_height=True, counted_once_per_installation=True)
    cal = replace(cal, field_settings=settings)
    bundle = MovingPivotSourceBundle(base, cal, moving_motion_wrapper, reference_pts=stamps[0],
        initial_reference_optical_height_scene=seed.camera25(views[stamps[0]])[2], scene=read(scene_path))
    q = np.r_[bundle.initial_shared(), np.concatenate([
        np.r_[views[p].view.rvec_world_to_camera, views[p].view.log_f_ratio+np.log(1920.)] for p in stamps])]
    del base, pairs
    gc.collect()
    binding = dict(schema='rinkspace.low-full-source-moving-pivot.v1',
        source_sha256=SOURCE_SHA, scene=ref(scene_path), metres_per_scene_unit=SCENE_UNIT_M,
        baseline=ref(CURRENT_PHYSICAL), mechanical_initialization=ref(SEED),
        source_authority_rows=ref(rows_path), exact_current_absolute_rows=ref(original_path),
        previous_input_manifest=ref(manifest_path), source_motion=bindings,
        native_frames=len(native), actual_exposures=len(stamps), original_motion_points=len(bundle.a),
        pair_count=len(pair_cuts), fit_rows=96, held_rows=7, pending_rows=10,
        finite_material_motion_rows=0, unknown_positive_depth_rows=len(bundle.a),
        arbitrary_surface_assignment=False, original_roles_and_pixels_retained=True,
        fixed_pivot_xy=list(cal.pivot_xyz[:2]), fixed_PP_k1=True, independent_Pz_and_shared_lever3=True,
        first_actual_reference_pts=stamps[0], reference_lens_height_prior=settings['reference_optical_height_prior'],
        constant_offset_across_zoom=True, per_frame_unrestricted_C=False,
        source_entrypoint=ref(__file__), moving_bundle=ref(Path(__file__).with_name('z190_moving_pivot_source_bundle.py')),
        motion_kernel=ref(Path(__file__).with_name('z190_moving_center_motion.py')),
        accuracy_accepted=False, whole_cut_accepted=False)
    write(output/'input_manifest.json', binding)
    write(output/'retained_original_pair_bindings.json', pair_cuts)
    np.savez_compressed(output/'initial_state4rf.npz', q=q, canonical_pts=stamps)
    return bundle, q, baseline, all_views, native, aliases, views, rows, binding


def run(args):
    output = args.output.resolve()
    if not output.is_relative_to(Path('D:/CodexData').resolve()):
        raise ValueError('Numeric products must stay outside OneDrive')
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    enable_projection_backend()
    bundle, q, baseline, oldviews, native, aliases, views, rows, binding = prepare(output)
    checks = derivative_checks(bundle, q)
    write(output/'actual_low_derivative_checks.json', checks)
    print(json.dumps(dict(stage='low_nonzero_prepared', native=len(native), actual=len(views),
        motion_points=len(bundle.a), pending_not_fit=10, derivative_max_relative=max(x['relative_error'] for x in checks))), flush=True)
    if not args.solve:
        return
    fitted, history, seconds = solve_moving_bundle(bundle, q, output)
    cal = bundle.current_calibration(fitted)
    settings = dict(cal.field_settings)
    settings['active_mount_parameters'] = dict(settings['active_mount_parameters'],
        fitted_lever_xyz_m=(np.asarray(cal.lever_xyz)*SCENE_UNIT_M).tolist(),
        fitted_pivot_xyz_scene_units=list(cal.pivot_xyz))
    cal = replace(cal, field_settings=settings)
    model = PhysicalCameraModel({IDENTIFIER: cal})
    newviews = {p: CanonicalView(IDENTIFIER, bundle.view(fitted, p)) for p in bundle.stamps}
    camera_path = output/'physical_low_nonzero_mount_candidate_unaccepted.npz'
    model.save_npz(camera_path, native, aliases, newviews, source_sha256=SOURCE_SHA, time_base='1/15360')
    loaded, pts2, aliases2, views2 = PhysicalCameraModel.load_npz(camera_path)
    if not np.array_equal(pts2, native) or aliases2 != aliases:
        raise ValueError('Low source exposure replay changed')
    changes = source_changes(rows, baseline, oldviews, loaded, views2, aliases)
    write(output/'all_absolute_source_changes.json', changes)
    depth_trace = save_depth_trace(bundle, fitted, output)
    centres = np.asarray([model.camera25(newviews[p])[:3] for p in bundle.stamps])*SCENE_UNIT_M
    report = dict(binding, solve_seconds=seconds, history=history, candidate=ref(camera_path),
        fitted_pivot_xyz_m=(np.asarray(cal.pivot_xyz)*SCENE_UNIT_M).tolist(),
        fitted_lever_xyz_m=(np.asarray(cal.lever_xyz)*SCENE_UNIT_M).tolist(),
        optical_height_range_m=[float(centres[:,2].min()), float(centres[:,2].max())],
        all_source_changes=ref(output/'all_absolute_source_changes.json'), latent_depth_trace=depth_trace,
        disposition='unaccepted_real_source_review_pending')
    write(output/'report.json', report)
    print(json.dumps(dict(stage='low_nonzero_exported', candidate=str(camera_path), seconds=seconds)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=CURRENT/'z190_low_nonzero_mount_full_source_v1')
    parser.add_argument('--solve', action='store_true')
    run(parser.parse_args())
