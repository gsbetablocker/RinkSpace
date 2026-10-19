"""Audit original wide45 support without decoding, detecting, or fitting.

The returned records are inert inputs for the public sampling compiler. They
retain exact v5 UV and sigma. Board target regions stay separate until an
explicit physical-material equivalence decision exists. Original profile
paths are recorded as provenance only. All compiler records use the existing
isolated authenticated-anchor policy without claiming visible paint between
source stations.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.interpolate import PchipInterpolator
from rinkspace.source_sampling import compile_curve_sampling, source_support_masses
from rinkspace.storage import AssetStore
from rinkspace.path_policy import guard_project_write
from calibration.solver_runtime.legacy.wide_anchor_candidate.native_wide_source import (
    FILES, SOURCE_IMAGE_SHA256, SOURCE_PTS, SOURCE_VIDEO_SHA256,
    load_native_wide_source_packet)

H = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/'
         'current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
OUT = H / 'current_wide45_original_curve_support_audit_20261011.json'
POLICY = ROOT / ('events/ankara_2026/observations/source_packets/diagnostics/'
    'current56_shared_active_set_filter_20261009/systematic_continuation_20261010/'
    'original_WIDE45_current_physical_curve_isolated_anchor_POLICY_OPTIONAL_20261011.json')
TRACE = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/evidence/'
             'user_contours_20261004/edge1_2e2bd471877d/edge1_source_uv.json')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024): value.update(chunk)
    return value.hexdigest()


def ref(path, pointer=None):
    value = dict(path=str(Path(path).resolve()), sha256=digest(path))
    if pointer is not None: value['json_pointer'] = pointer
    return value


def numeric_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype='<f8').tobytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def fragments(stations):
    """Existing points remain ordered; omitted source stations break a path."""
    stations = np.asarray(stations, int)
    if len(set(stations.tolist())) != len(stations) or np.any(np.diff(stations) <= 0):
        raise ValueError('Original profile station order is not uniquely increasing')
    boundaries = np.r_[0, np.flatnonzero(np.diff(stations) != 1) + 1, len(stations)]
    return [list(range(int(a), int(b))) for a, b in zip(boundaries[:-1], boundaries[1:])]


def guided_stations(row):
    """Recover indices from the original profile grid, never image extrema.

    Each retained rounded point must match exactly one original guide station
    and one original normal offset.  This validates provenance, not visibility.
    """
    guide = np.asarray(row['search_guide_image_px'], float)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(guide, axis=0), axis=1))]
    q = PchipInterpolator(arc, guide, axis=0)(np.linspace(0., arc[-1], max(8, int(arc[-1] / 7))))
    tangent = np.gradient(q, axis=0)
    normal = np.c_[-tangent[:, 1], tangent[:, 0]]
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    radius = float(row['search_radius_px'])
    offsets = np.arange(-radius, radius + .01, .25)
    original_grid = np.round(q[:, None] + normal[:, None] * offsets[None, :, None], 3)
    points = row['pixels_uv']
    if 'per_point_remeasurement' in row:
        changes = row['per_point_remeasurement']
        if [p['remeasured_uv'] for p in changes] != points:
            raise ValueError('Explicit remeasurement lineage changed')
        points = [p['old_uv'] for p in changes]
    result = []
    for point in points:
        hits = np.argwhere(np.all(original_grid == np.asarray(point), axis=2))
        if len(hits) != 1:
            raise ValueError(f'Original profile point does not have a unique exact station: {point}')
        result.append(int(hits[0, 0]))
    return result, len(q)


def support(row, audit_dir):
    uv = np.asarray(row['pixels_uv'], float)
    singleton = [[i] for i in range(len(uv))]
    old = row.get('original_observation', {})
    if old.get('geometry') == 'closed_circle_contour':
        source = np.asarray(old['points_px'], float)
        indices = []
        for point in uv:
            matches = np.flatnonzero(np.all(source == point, axis=1))
            if len(matches) != 1: raise ValueError('Current outer contour is not an exact original source subset')
            indices.append(int(matches[0]))
        if np.any(np.diff(indices) <= 0): raise ValueError('Original closed contour order changed')
        paths = [list(range(len(uv))) + [0]]
        return paths, dict(kind='original_declared_closed_source_contour_retained_exact_subset',
            original_source_points=len(source), original_indices=indices,
            original_source_uv_sha256=numeric_sha(source),
            current_points_are_original_measured_anchors=True,
            continuous_support_authority='original_observation.geometry=closed_circle_contour and original source measurement method',
            full_source_path_used_as_new_FIT_pixels=False, arclength_is_existing_retained_source_chord_quadrature=True)
    if row.get('method', '').startswith('image-only manual guide, normal scan, local paint-profile maximum'):
        stations, total = guided_stations(row)
        return fragments(stations), dict(kind='original_profile_grid_adjacent_retained_station_fragments',
            source_station_indices=stations, original_grid_station_count=total,
            raw_rejected_search_samples=row.get('rejected_search_samples'),
            source_reference=ref(audit_dir / 'extract_additions.py'),
            current_remeasurement_reference=ref(audit_dir / 'revise_lane_qualifications.py') if 'per_point_remeasurement' in row else None,
            source_gap_inference=False, consecutive_grid_stations_only=True,
            missed_stations_not_bridged=True, source_pixels_not_remeasured=True,
            continuous_support_authority='original image-only guided trace and exact original profile grid adjacency',
            station_adjacency_is_not_independent_visibility_validation=True)
    if row.get('method', '').startswith('raw image row-wise first blue chroma onset'):
        start = 700 if 'lower_long' in row['id'] else 842
        end = 1069 if 'lower_long' in row['id'] else 1077
        ys = np.arange(start, end, 8)
        stations = []
        for point in uv:
            matches = np.flatnonzero(ys == point[1])
            if len(matches) != 1: raise ValueError('Original near-rail row grid changed')
            stations.append(int(matches[0]))
        return fragments(stations), dict(kind='original_declared_regular_source_y_grid',
            source_station_indices=stations, original_grid_station_count=len(ys),
            source_reference=ref(audit_dir / 'near_rail.py'),
            missed_stations_not_bridged=True,
            continuous_support_authority='original raw RGB scan grid; declared skipped/failed rows retain gaps',
            station_adjacency_is_not_independent_visibility_validation=True)
    if 'signed brightness transition maxima' in row.get('method', '') and len(uv) == 96:
        return [list(range(96)) + [0]], dict(kind='original_all_96_angular_profile_stations_retained',
            source_reference=ref(audit_dir / 'extract_inner_rings.py'),
            continuous_support_authority='original n=96 ordered angular profiles with no rejected station',
            inferred_ellipse_pixels_used=False, source_ellipse_is_search_initializer_only=True,
            station_adjacency_is_not_independent_visibility_validation=True)
    return singleton, dict(kind='isolated_original_measured_anchor_support_only',
        reason='No exact complete retained profile-station or visible-path authority recovered; do not bridge source gaps',
        actual_physical_isolation_proven=False, continuous_support_claim=False,
        original_source_description=old.get('geometry'), sparse_source_path_association_is_not_imputed=True)


def identity(row):
    target = row['target']; kind = target['kind']
    if kind == 'circle_outline':
        material = ['circle', target['feature_id'], target.get('ring_role', 'outer')]
        status = 'explicit_current_named_circle_and_exact_ring_layer'
    elif kind == 'polygon_supporting_line':
        material = ['lane_paint_supporting_line', target['feature_id'], int(target['index'])]
        status = 'explicit_current_named_paint_edge_no_observed_endpoint'
    elif kind == 'line_segment':
        material = ['named_line_paint', target['feature_id'], int(target.get('segment', 0))]
        status = 'explicit_current_named_line_paint'
    elif kind == 'polygon_outline':
        material = ['polygon_outline', target['feature_id'], row.get('edge_definition')]
        status = 'explicit_current_named_polygon_boundary'
    elif kind == 'board_side':
        # Target-region overlap alone is not a material-equivalence certificate.
        material = ['board_source_region_pending_cross_region_equivalence', target,
                    row.get('edge_definition') or row.get('original_observation', {}).get('surface_type')]
        status = 'same_current_course_height_role_known_cross_region_material_equivalence_not_certified'
    else: raise ValueError('Unrecognized original current wide target')
    return json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(',', ':')), status


def build_records(root=ROOT):
    """Return exact original records plus a small traceability audit.

    No caller has installed the returned loss plan.  Literal near4/global4
    remain separate engineering cost factors and are not encoded as sigma.
    """
    store = AssetStore(Path(root))
    packet = load_native_wide_source_packet(root, verify_source_image=False)
    problem = store.resolve(FILES['v5_problem'][0])
    audit_dir = problem.parent.parent / 'observations_audit'
    records = []; rows = []
    for i, row in enumerate(packet.observations):
        uv = np.ascontiguousarray(row['pixels_uv'], dtype='<f8')
        recovered_paths, recovered_proof = support(row, audit_dir)
        paths = [[j] for j in range(len(uv))]
        masses = source_support_masses(uv, paths)
        proof = dict(kind='isolated_authenticated_original_anchor_support',
            source_path_visible_support_independently_certified=False,
            continuous_support_claim=False, actual_physical_isolation_proven=False,
            recovered_profile_or_contour_path_not_used_for_integration=True,
            source_support_mass_per_unique_anchor=1.)
        physical, status = identity(row)
        sigma = np.broadcast_to(np.asarray(row['sigma_px'], float), (len(uv),)).copy()
        record = dict(id=row['id'], split='fit', source_video_sha256=SOURCE_VIDEO_SHA256,
            actual_pts=SOURCE_PTS, physical_curve_id=physical,
            operator_id='wide_full_Brown_native_xy_nearest_finite_curve:' + json.dumps(row['target'], sort_keys=True, separators=(',', ':')),
            source_anchor_identity_authority=ref(problem),
            uv=uv, stored_sigma_px=sigma.copy(), working_sigma_px=sigma.copy(),
            prior_sampling_multiplier=np.ones(len(uv)), point_integration_mass=masses,
            sampling_kind=['measured'] * len(uv), literal_source_weight=float(row.get('weight', 1.)),
            global_engineering_multiplier=4., source_path_indices=paths,
            current_source_row_reference=ref(problem, f'/observations/{i}'),
            physical_identity_status=status, source_support_proof=proof)
        records.append(record)
        distance = np.linalg.norm(np.diff(uv, axis=0), axis=1)
        rows.append(dict(id=row['id'], original_v5_index=i, target=row['target'], physical_curve_id=physical,
            physical_identity_status=status, points=len(uv), native_uv_sha256=numeric_sha(uv),
            source_row_reference=record['current_source_row_reference'], source_frame=row['source_frame'],
            source_material_definition=row.get('edge_definition'), original_method=row.get('original_observation', {}).get('method'),
            current_method=row.get('method'), original_measurement_origin=row['measurement_origin'],
            original_working_sigma_px=row['sigma_px'], working_sigma_unchanged=True,
            measured_statistical_covariance_claim=False, prior_sampling_multiplier=1.,
            literal_source_weight=record['literal_source_weight'], global_engineering_multiplier=4.,
            source_support_paths=paths, source_support_proof=proof,
            reconstructed_original_profile_path_indices_for_audit_only=recovered_paths,
            reconstructed_original_profile_path_provenance=recovered_proof,
            source_support_mass=float(masses.sum()), singleton_fragments=sum(len(p) == 1 for p in paths),
            adjacent_original_uv_distance_min_median_max_px=[float(distance.min()), float(np.median(distance)), float(distance.max())],
            no_extra_FIT_points=True, duplicate_original_UV_count=len(uv) - len(np.unique(uv, axis=0)),
            current_outer_weighted_quadratic_per_axis_coefficient=float(np.sum(4. * record['literal_source_weight'] / sigma ** 2)),
            source_partition_of_observation_id=row.get('partition_of_observation_id'),
            source_uncertainty_note=row.get('uncertainty_note'),
            material_relation_to_other_board_regions_requires_source_confirmation=row['target']['kind'] == 'board_side'))
    plan = compile_curve_sampling(records, duplicate_anchor_policy=dict(
        kind='same_source_actual_physical_curve_operator_UV_share_budget_preserve_pointwise_sigma',
        source_anchor_identity_authority=ref(problem)))
    for row in rows:
        entry = plan['rows'][row['id']]
        weights = entry['point_integration_weights']
        row['inert_compiler_preview'] = dict(point_integration_weight_sum=float(weights.sum()),
            weight_min=float(weights.min()), weight_max=float(weights.max()),
            working_normalization_multiplier_is_one=bool(np.all(entry['working_normalization_multiplier'] == 1.)),
            original_literal_and_global_multipliers_not_applied_or_changed_by_compiler=True)
    report = dict(schema='rinkspace.current-wide45-exact-original-support-audit.v1',
        source_packet_contract=packet.contract, original_current_observations=45,
        original_current_UV=sum(len(r['uv']) for r in records), source_rows=rows,
        source_support_kinds=dict(Counter(r['source_support_proof']['kind'] for r in records)),
        original_profile_path_provenance_kinds=dict(Counter(r['reconstructed_original_profile_path_provenance']['kind'] for r in rows)),
        inert_compiler_preview=dict(groups=plan['groups'], policy=plan['policy'],
            not_installed_in_active_source_or_optimizer=True,
            board_source_regions_not_merged_without_material_equivalence=True,
            rings_and_lane_edges_not_combined_into_broad_feature_budget=True,
            exact_original_UV_order_sigma_roles_target_and_multipliers_preserved=True),
        public_compiler_reference=ref(ROOT / 'src/rinkspace/source_sampling.py'),
        actual_current_producer_references=[ref(ROOT / p) for p in (
            'calibration/solver_runtime/legacy/structured_programme_source_candidate/current_wide_source.py',
            'calibration/solver_runtime/legacy/wide_anchor_candidate/native_wide_source.py',
            'calibration/solver_runtime/legacy/complete_ice_template_candidate/full_free_boundary_geometry.py')],
        scope='Original wide45 only; original corners/entity and two separate user strokes are not newly regrouped',
        no_images_decoded_or_written=True, no_residual_or_J_or_fit=True, no_source_resampling=True,
        source_support_is_engineering_integration_not_independent_covariance=True)
    return records, report


def user_top_overlap(records):
    trace = json.loads(TRACE.read_text(encoding='utf-8'))
    result = []
    for contour in trace['contours']:
        user_uv = np.asarray(contour['observed_centres_source_px'], float)
        columns = {int(v[0]): v[1:3] for v in contour['column_support_x_ymin_ymax_alpha_sum']}
        if np.any(np.diff(user_uv[:, 0]) != 1): raise ValueError('Original user alpha column support is not contiguous')
        matches = []
        for rec in records:
            if not rec['physical_curve_id'].startswith('["board_source_region'): continue
            uv = rec['uv']; xrange = (uv[:, 0] >= user_uv[0, 0] - .5) & (uv[:, 0] <= user_uv[-1, 0] + .5)
            if not xrange.any(): continue
            band_indices = []
            for i, point in enumerate(uv):
                col = columns.get(int(np.floor(point[0] + .5)))
                if col is not None and col[0] <= point[1] <= col[1]: band_indices.append(i)
            exact = set(map(tuple, uv)) & set(map(tuple, user_uv))
            matches.append(dict(original_id=rec['id'], original_target_region=rec['physical_curve_id'],
                original_points_in_same_user_x_span=int(xrange.sum()),
                original_point_indices_inside_original_alpha_column_band=band_indices,
                exact_duplicate_source_centre_UV_count=len(exact),
                image_annotation_overlap_is_not_a_material_equivalence_or_source_covariance_certificate=True))
        result.append(dict(id=contour['id'], source_reference=ref(TRACE), original_alpha_support_columns=len(user_uv),
            original_user_UV_sha256=numeric_sha(user_uv), physical_kind=contour['physical_kind'],
            existing_objective='per-stroke average squared native source distance / engineering sigma2px squared; outer4 retained',
            existing_source_support='original full alpha column band; unquantified user drawing/source blur',
            overlaps=matches, source_pixels_or_original_user_support_not_modified=True,
            unresolved_relation='Same-image TOP annotation overlap must not acquire an invented independent measurement budget'))
    return result


def frozen_policy(records, report, audit_path):
    """Small authenticated policy; production consumers reconstruct original UV."""
    plan = compile_curve_sampling(records, duplicate_anchor_policy=dict(
        kind='same_source_actual_physical_curve_operator_UV_share_budget_preserve_pointwise_sigma',
        source_anchor_identity_authority=report['source_packet_contract']['source_files']['v5_problem']))
    rows = []
    for index, (record, summary) in enumerate(zip(records, report['source_rows'])):
        rows.append(dict(original_v5_index=index, id=record['id'], points=len(record['uv']),
            target=summary['target'], original_point_indices=list(range(len(record['uv']))),
            source_UV_float64_sha256=numeric_sha(record['uv']),
            source_sigma_float64_sha256=numeric_sha(record['stored_sigma_px']),
            original_sigma_px=summary['original_working_sigma_px'],
            stored_sigma_px=record['stored_sigma_px'], working_sigma_px=record['working_sigma_px'],
            prior_sampling_multiplier=record['prior_sampling_multiplier'],
            point_integration_mass=record['point_integration_mass'], sampling_kind=record['sampling_kind'],
            actual_pts=record['actual_pts'], source_video_sha256=record['source_video_sha256'],
            physical_curve_id=record['physical_curve_id'], operator_id=record['operator_id'],
            physical_identity_status=record['physical_identity_status'],
            source_support_basis='isolated_authenticated_source_anchors',
            source_row_reference=record['current_source_row_reference'],
            literal_source_weight=record['literal_source_weight'],
            global_engineering_multiplier=record['global_engineering_multiplier'],
            point_integration_weights=plan['rows'][record['id']]['point_integration_weights']))
    return dict(schema='rinkspace.original-wide45-curve-source-integration.v1',
        policy_role='optional_authenticated_counterfactual_source_budget_not_accuracy_acceptance',
        curve_budget=1., support_basis='isolated_authenticated_source_anchors',
        source_video_sha256=SOURCE_VIDEO_SHA256, source_pts=SOURCE_PTS, native_pts=SOURCE_PTS, actual_pts=SOURCE_PTS,
        time_base='1/15360', native_image=report['source_packet_contract']['native_image'],
        source_packet_reference=report['source_packet_contract']['source_files']['v5_problem'],
        source_files=report['source_packet_contract']['source_files'],
        source_support_audit_reference=ref(audit_path),
        original_observation_indices=list(range(45)), original_UV_count=2144,
        rows=rows, groups=plan['groups'], duplicate_anchor_policy=plan['duplicate_anchor_policy'],
        original_source_order_UV_sigma_roles_targets_and_point_indices_unchanged=True,
        all_original_UV_have_positive_quality=True, new_source_UV_or_resampling=False,
        enable_point_compression=False, working_sigma_unchanged=True,
        normalization=None, sigma_decomposition='stored=working; no original sampling factor declared; prior=1',
        global_engineering_multiplier=4., preserve_global_engineering_multiplier=True,
        preserve_literal_source_weights=True,
        literal_source_weight_policy='Retain original literal row weight outside Huber; near-right4 remains4',
        global_engineering_multiplier_policy='Retain original outer family multiplier4 independently',
        loss='point_integration_weight * original_literal_weight * original_global4 * original_component_Huber2(native_error / original_working_sigma)',
        original_26_advertisement_corners_2_entity_scalars_and_2_user_TOP_strokes_unchanged=True,
        user_TOP_alpha_supports_and_correlated_objectives_not_regrouped=True,
        user_TOP_overlap_reference=dict(path=str(Path(audit_path).resolve()),
            sha256=digest(audit_path), json_pointer='/two_existing_user_TOP_original_alpha_overlap'),
        board_source_regions_retained_separately=True,
        board_same_physical_edge_across_regions_unknown_not_guessed=True,
        rings_and_lane_paint_edges_separate=True, same_named_circle_fragments_share_budget=True,
        support_sampling_count_is_not_independent_precision=True,
        engineering_integration_policy_not_measured_covariance=True,
        active_configuration_modified=False, production_implementation_modified=False,
        fitted_state_modified=False, accuracy_accepted=False,
        public_compiler_reference=report['public_compiler_reference'],
        builder_reference=report['script_reference'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--write-policy', action='store_true')
    parser.add_argument('--policy-output', type=Path, default=POLICY)
    args = parser.parse_args()
    args.output = guard_project_write(args.output, project_root=ROOT)
    if args.write_policy:
        args.policy_output = guard_project_write(AssetStore(ROOT).resolve(args.policy_output), project_root=ROOT)
    records, report = build_records()
    image = Path(report['source_packet_contract']['native_image']['path'])
    actual_sha = digest(image)
    if actual_sha != SOURCE_IMAGE_SHA256: raise ValueError('Original native image bytes changed')
    report['original_native_image_stream_sha_authentication'] = dict(path=str(image), sha256=actual_sha,
        bytes=image.stat().st_size, image_decoded=False)
    report['two_existing_user_TOP_original_alpha_overlap'] = user_top_overlap(records)
    report['script_reference'] = ref(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=plain, allow_nan=False) + '\n', encoding='utf-8')
    result = dict(path=str(args.output), sha256=digest(args.output), bytes=args.output.stat().st_size,
        rows=len(records), UV=report['original_current_UV'],
        groups=len(report['inert_compiler_preview']['groups']), source_support_kinds=report['source_support_kinds'])
    if args.write_policy:
        policy = frozen_policy(records, report, args.output)
        args.policy_output.parent.mkdir(parents=True, exist_ok=True)
        args.policy_output.write_text(json.dumps(policy, ensure_ascii=False, indent=2, default=plain, allow_nan=False) + '\n', encoding='utf-8')
        result['policy'] = ref(args.policy_output)
        result['policy']['bytes'] = args.policy_output.stat().st_size
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
