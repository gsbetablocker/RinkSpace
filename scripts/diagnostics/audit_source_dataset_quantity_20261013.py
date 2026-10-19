"""Read-only final source quantity and source-clock audit without a solver.

Missing working support, explicit HELD and source roles stay separate. Numeric
integrity does not certify pixel accuracy, geometry, or complete solver speed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from calibration.program_tracking.compact_source_header_inventory import read_minimal_compact_source_headers
from calibration.program_tracking.strict_model_scope import authenticate_source_reference, file_sha


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def audit(source_ref, coverage_ref=None, input_audit_ref=None):
    started = time.perf_counter()
    authenticate_source_reference(source_ref)
    rows, offsets = read_minimal_compact_source_headers(source_ref['path'])
    with np.load(source_ref['path'], allow_pickle=False) as z:
        metadata = json.loads(str(z['metadata_json'].item()))
        uv, sigma = z['uv'], z['sigma_px']
        frame_values = tuple(z[n] for n in ('source_frame_actual_pts', 'source_frame_native_pts', 'source_frame_BGR_sha256'))
        custom_weight_array_names = [n for n in z.files if 'integration_weight' in n or 'loss_weight' in n]
    finite = np.isfinite(sigma) & (sigma > 0)
    if (offsets[-1] != len(uv) or len(uv) != len(sigma) or not np.isfinite(uv).all()
            or np.isinf(sigma).any() or np.any(sigma[np.isfinite(sigma)] <= 0)):
        raise ValueError('Final source offsets or actual UV/working sigma are invalid')
    frames = {int(a):(int(n), bytes(h).decode('ascii')) for a,n,h in zip(*frame_values)}
    if len(frames) != len(frame_values[0]) or sorted(frames) != metadata['processed_actual_pts']:
        raise ValueError('Final source processing clock and frame binding disagree')
    expected = metadata['expected_actual_pts']
    if set(frames) != set(expected) or not metadata['all_exposures_processed'] or not metadata['all_exposures_exact_native_BGR_bound']:
        raise ValueError('Final canonical source coverage is incomplete')
    coverage = None
    if coverage_ref:
        authenticate_source_reference(coverage_ref)
        with np.load(coverage_ref['path'], allow_pickle=False) as z:
            canonical = np.flatnonzero(z['representative'])
            original = {int(z['actual_pts'][i]):(int(z['native_pts'][i]), bytes(z['BGR_sha256'][i]).decode('ascii')) for i in canonical}
            coverage = dict(reference=coverage_ref, native_frames=len(z['native_pts']),
                canonical_exposures=len(canonical), native_alias_frames=len(z['native_pts'])-len(canonical),
                old_original_points_retired=len(z['old_pixel_disposition']),
                all_native_original_RGB_processed=bool(z['native_material_measurement_attempted'].all()))
        if frames != original or not coverage['all_native_original_RGB_processed']:
            raise ValueError('Final pack lost or changed an original canonical exposure/RGB binding')
        coverage['all_actual_native_RGB_bindings_bit_exact'] = True
    points_by_split = Counter()
    finite_by_split = Counter()
    roles, families = Counter(), Counter()
    materials = {}
    group_ids, choices = {}, {}
    duplicates = split_conflicts = 0
    reconstructed_rows = reconstructed_points = reconstructed_finite_points = 0
    for ri, row in enumerate(rows):
        if row['split'] not in ('fit', 'held') or row['scope'] not in ('ice', 'board'):
            raise ValueError('Final source split/scope invalid')
        actual = row['actual_pts']
        if frames[actual] != (row['native_pts'], row['decoded_BGR_sha256']):
            raise ValueError('Final row lost its own exact original source binding')
        lo, hi = map(int, offsets[ri:ri+2])
        n = hi-lo
        points_by_split[row['split']] += n
        finite_by_split[row['split']] += int(finite[lo:hi].sum())
        roles[row.get('vision_training_role', 'unspecified')] += n
        families[row.get('family', 'unspecified')] += n
        material = row['physical_material_id']
        stats = materials.setdefault(material, dict(points=0, rows=0, finite_working_sigma_points=0,
            exposures=set(), fit_points=0, held_points=0))
        stats['points'] += n
        stats['rows'] += 1
        stats['finite_working_sigma_points'] += int(finite[lo:hi].sum())
        stats['exposures'].add(actual)
        stats[row['split']+'_points'] += n
        if row.get('source_working_support_reconstructed_from_saved_actual_readout') is True:
            reconstructed_rows += 1
            reconstructed_points += n
            reconstructed_finite_points += int(finite[lo:hi].sum())
        physical = (actual, row['scope'], material,
            json.dumps(row['model_target'], sort_keys=True, separators=(',', ':')), row['observation_semantics'])
        gi = group_ids.setdefault(physical, len(group_ids))
        for point in uv[lo:hi]:
            key = (gi, np.asarray(point, np.float64).tobytes())
            if key in choices:
                duplicates += 1
                split_conflicts += choices[key] != row['split']
            else:
                choices[key] = row['split']
    if duplicates or split_conflicts:
        raise ValueError('Final source pool still contains exact pixel/target duplicates')
    for stats in materials.values():
        stats['actual_exposures'] = len(stats.pop('exposures'))
        stats['missing_working_sigma_points'] = stats['points']-stats['finite_working_sigma_points']
    ready = points_by_split['fit'] > 0 and finite_by_split['fit'] == points_by_split['fit']
    if metadata['source_uncertainty_ready_for_fit'] != ready:
        raise ValueError('Final FIT readiness disagrees with its literal working support')
    source_audit = None
    if input_audit_ref:
        authenticate_source_reference(input_audit_ref)
        prior = json.loads(Path(input_audit_ref['path']).read_text(encoding='utf8'))
        source_audit = dict(reference=input_audit_ref, source_input_points=prior['points'],
            source_unique_exact_pixel_target_count=prior['unique_exact_pixel_target_count'],
            input_exact_duplicate_occurrences=prior['counts'].get('exact_duplicate_occurrences', 0),
            input_operator_flag_conflicting_occurrences=prior['counts'].get('operator_flag_conflicting_occurrences', 0),
            literal_same_input_unique_point_quantity=prior['unique_exact_pixel_target_count'] == len(uv),
            newly_finite_support_points_relative_to_input_unique_pixels=int(finite.sum())-
                prior['counts']['unique_finite_working_sigma_points_before_saved_readout_reconstruction'])
    for ref in metadata['original_references']:
        authenticate_source_reference(ref)
    authenticate_source_reference(source_ref)
    return dict(schema='rinkspace.final-source-dataset-quantity-and-clock-audit.v1', source_reference=source_ref,
        source_points=len(uv), source_rows=len(rows), scalar_residual_rows_for_all_source_points=2*len(uv),
        scalar_residual_rows_for_finite_fit_points=2*finite_by_split['fit'],
        points_by_split=dict(points_by_split), finite_working_sigma_points_by_split=dict(finite_by_split),
        missing_working_sigma_points_by_split={s:n-finite_by_split[s] for s,n in points_by_split.items()},
        finite_working_sigma_points=int(finite.sum()), missing_working_sigma_points=int((~finite).sum()),
        working_sigma_min_median_max=None if not finite.any() else np.quantile(sigma[finite], [0,.5,1]).tolist(),
        working_sigma_is_actual_readout_support_not_verified_Gaussian_covariance=True,
        source_metadata_working_support_ready_for_fit=ready, current_declared_fit_is_not_solver_admission=True,
        exact_duplicate_pixel_target_occurrences=duplicates, conflicting_fit_held_duplicate_occurrences=split_conflicts,
        physical_target_exposure_groups=len(group_ids), source_points_by_training_role=dict(roles),
        source_points_by_family=dict(families), material_summary=materials,
        saved_actual_readout_support_reconstructed_rows=reconstructed_rows,
        saved_actual_readout_support_reconstructed_points=reconstructed_points,
        saved_actual_readout_support_reconstructed_finite_points=reconstructed_finite_points,
        custom_loss_or_integration_weight_array_names=custom_weight_array_names,
        actual_exposures=len(frames), all_expected_actual_exposures_present=True,
        original_native_RGB_coverage=coverage, source_input_comparison=source_audit,
        original_source_and_authority_bytes_authenticated=True, original_source_arrays_changed=False,
        weights_installed=False, complete_world100_and_shared_optics_qualified=False,
        complete_iteration_speed_verified=False, whole_shot_visual_accuracy_accepted=False,
        solver_started=False, elapsed_seconds=time.perf_counter()-started,
        implementation_reference=reference(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--coverage', type=Path)
    parser.add_argument('--input-audit', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve a previous source quantity audit')
    result = audit(dict(path=str(args.pack.resolve()), sha256=args.expected_sha256),
        reference(args.coverage) if args.coverage else None,
        reference(args.input_audit) if args.input_audit else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, separators=(',', ':'))
    print(json.dumps({k:result[k] for k in ('source_points', 'source_rows', 'finite_working_sigma_points',
        'missing_working_sigma_points', 'actual_exposures', 'saved_actual_readout_support_reconstructed_points')},
        separators=(',', ':')), flush=True)
