"""Compile immutable full-source semantics and the original grouped likelihood.

Source UV and primary keys come exclusively from Observations. A historical
safe source snapshot supplies read-only metadata and whole frozen W groups.
Matching is by (native_pts, source_point_id), never by an identifier alone.
No legacy geometry, owner, fit residual, outlier logic or W update is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .data import FIELDS, Observations, SeedCatalog, sha256_file
from .runtime import atomic_json
from .semantic_snapshot import SnapshotArrays, read_reference, stream_snapshot_frames

SCHEMA = 'rinkfit.semantic_packet.v1'
FINITE_CURVE, ENTITY_CORNER, TRUE_PROFILE = 0, 1, 2


def _reference(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256_file(path)}


def _write_array(path, value, dtype):
    value = np.ascontiguousarray(value, dtype=dtype)
    value.tofile(path)
    return {'file': path.name, 'dtype': value.dtype.str, 'shape': list(value.shape),
            'layout': 'C', 'sha256': sha256_file(path)}


def _json_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def validate_group_membership(point_indices, point_ids, global_rows, operator):
    """Reject partial or overlapping statistical groups instead of slicing W."""
    indices = np.asarray(point_indices, dtype=int)
    if (indices.ndim != 1 or not len(indices) or len(set(indices.tolist())) != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(point_ids))):
        raise ValueError('Invalid original statistical group membership')
    if tuple(point_ids[i] for i in indices) != tuple(operator['source_point_ids']):
        raise ValueError('Frozen W membership does not match its original IDs')
    rows = np.asarray([global_rows[i] for i in indices], dtype=np.int64)
    selected = rows >= 0
    if selected.any() and not selected.all():
        raise ValueError('Current source cuts an original correlated W group; whole-source evidence required')
    matrix = np.asarray(operator['whitening_matrix'], dtype='<f8')
    if (matrix.ndim != 2 or matrix.shape[1] != 2 * len(indices)
            or not matrix.shape[0] or not np.isfinite(matrix).all()):
        raise ValueError('Invalid complete frozen W matrix')
    if np.any(np.linalg.norm(matrix.reshape(len(matrix), len(indices), 2), axis=(0, 2)) == 0):
        raise ValueError('A source member would not enter its frozen W')
    return rows if selected.all() else None, matrix


def profile_semantics(row, source_packet):
    """Recover the exact original query and finite interval, with no default."""
    saved = row['native_line_queries']
    query = np.asarray(saved['query_uv'], dtype=float)
    normal = np.asarray(saved['profile_normal'], dtype=float)
    if query.shape != normal.shape or query.shape != np.asarray(row['uv']).shape:
        raise ValueError('Profile query shape differs from source membership')
    if not np.allclose(np.linalg.norm(normal, axis=1), 1., rtol=0, atol=1e-10):
        raise ValueError('Original profile normal is not unit length')
    ref = source_packet.get('ice_source_readout_packet')
    if ref is None:
        raise ValueError('True profile has no original native readout packet')
    packet = read_reference(ref)
    feature_id = row['source_feature_id']
    candidates = [f for f in packet['features'] if f['id'] == feature_id]
    if len(candidates) != 1:
        raise ValueError('Profile original native material is not unique')
    readouts = candidates[0]['source_line_fit']['direct_readouts']
    by_query = {int(r['source_evidence']['query_index']): r for r in readouts}
    bounds, evidence = [], []
    for j, pid in enumerate(row['source_point_ids']):
        index = int(str(pid).rsplit('/query/', 1)[1])
        raw = by_query[index]
        origin = raw['source_evidence']
        for actual, expected, name in ((raw['uv'], row['uv'][j], 'UV'),
                (origin['query_uv'], query[j], 'query'), (origin['profile_normal'], normal[j], 'normal')):
            if np.asarray(actual, dtype='<f8').tobytes() != np.asarray(expected, dtype='<f8').tobytes():
                raise ValueError('True profile changed original ' + name)
        interval = np.asarray(origin['complete_band_native_interval_px'], dtype=float)
        if interval.shape != (2,) or not np.isfinite(interval).all() or interval[0] >= interval[1]:
            raise ValueError('True profile requires its exact finite recorded alpha interval')
        bounds.append(interval)
        evidence.append({'native_readout_packet': ref, 'query_index': index,
            'source_evidence': origin, 'sigma_px': raw['sigma_px'],
            'query_parameter_units': origin.get('scalar_profile_units', 'recorded_native_profile_px')})
    return query, normal, np.asarray(bounds), evidence


def _variant_rows(obs, catalog, branch_ref):
    ids = sorted(catalog.variants)
    index = {name: i for i, name in enumerate(ids)}
    result = np.full(len(obs.source_uv), np.iinfo(np.uint32).max, dtype='<u4')
    branch = read_reference(branch_ref)
    if (branch.get('review_complete') is not True or branch.get('model_argmin_is_identity_authority') is not False
            or branch.get('source_UV_ID_sigma_W_clock_unchanged') is not True):
        raise ValueError('Explicit source-reviewed finite branch provenance required')
    source_arcs = {(int(r['native_pts']), r['source_point_id']): r for r in branch['records']}
    used = set()
    for label, material in catalog.materials.items():
        rows = np.flatnonzero(obs.source_target_labels == label)
        variants = material['variants']
        if len(variants) == 1:
            result[rows] = index[variants[0]]
            continue
        arc = [v for v in variants if catalog.variants[v]['descriptor'].get('source_reviewed_material_arc')]
        union = [v for v in variants if v not in arc]
        if len(arc) != 1 or len(union) != 1:
            raise ValueError('Ambiguous material variant needs original branch evidence: ' + label)
        for row in rows:
            key = int(obs.native_pts[row]), str(obs.source_point_ids[row])
            record = source_arcs.get(key)
            if record is None:
                result[row] = index[union[0]]
            else:
                if (record['source_boundary_side'] != 'exterior_bottom_arc'
                        or np.asarray(record['source_uv'], dtype='<f8').tobytes() != obs.source_uv[row].tobytes()):
                    raise ValueError('Reviewed material branch changed source UV')
                result[row] = index[arc[0]]
                used.add(key)
    if used != set(source_arcs):
        raise ValueError('Source-reviewed material branch inventory does not match all original obligations')
    return ids, result


def _ordered_source_runs(entries, packet, frame, source_ref, output):
    """Record source-proven run order; never infer a world direction from UV order."""
    count = points = 0
    by_segment = defaultdict(list)
    features = {f['id']: f for f in packet.get('features', [])}
    for entry in entries:
        row, vertex, global_row, variant = entry
        if row.get('native_line_queries') or row['target'].get('kind') == 'ad_corner':
            continue
        feature = features.get(row.get('source_feature_id'))
        if (feature is None or feature.get('world_correspondence_status') != 'named_same_physical_print_component'
                or feature.get('current_source_only_readout') is not True
                or feature.get('model_or_template_generated_observation') is not False):
            continue
        segment_id = int(row.get('source_segment_index', -1))
        if not 0 <= segment_id < len(feature.get('segments', [])):
            continue
        segment = feature['segments'][segment_id]
        if (segment.get('measurement_origin') != 'current_native_material_normal_half_transition'
                or segment.get('pixels_measured_in_this_frame') is not True):
            continue
        original_vertex = int(row['original_vertex_indices'][vertex])
        if not np.array_equal(np.asarray(segment['uv'][original_vertex]), row['uv'][vertex]):
            raise ValueError('Source run metadata changed original UV')
        by_segment[(feature['id'], segment_id, variant)].append((original_vertex, global_row, segment))
    for (feature_id, segment_id, variant), values in by_segment.items():
        # Sorting is by the actual saved extraction vertex address, not by the
        # observation-array order. Missing vertices split independent runs.
        values.sort(key=lambda x: x[0])
        runs = []
        for value in values:
            if runs and value[0] == runs[-1][-1][0] + 1:
                runs[-1].append(value)
            else:
                runs.append([value])
        for run in runs:
            if len(run) < 2:
                continue
            vertices = [v[0] for v in run]
            segment = run[0][2]
            closed = bool(segment['closed']) and vertices == list(range(len(segment['uv'])))
            _json_line(output, {'frame': frame, 'source_packet': source_ref,
                'source_feature_id': feature_id, 'source_segment_index': segment_id,
                'source_vertex_indices': vertices, 'source_rows': [v[1] for v in run],
                'variant_index': variant, 'source_closed': closed, 'source_order_verified': True,
                'world_direction': None, 'world_direction_inferred_from_observation_order': False,
                'provenance': 'actual saved current-native material extraction segment; no occlusion gap joined'})
            count += 1; points += len(run)
    return count, points


def export_semantic_packet(data_dir, output_dir, binding_path, *, progress=None):
    start = time.monotonic()
    data_dir, output = Path(data_dir), Path(output_dir)
    binding = json.loads(Path(binding_path).read_text(encoding='utf-8'))
    obs, catalog = Observations.load(data_dir), SeedCatalog(data_dir)
    catalog.validate_source(obs)
    source = binding['source_snapshot']
    if sha256_file(source['path']) != source['sha256']:
        raise ValueError('Complete original source metadata snapshot changed')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        return check_semantic_packet(output / 'manifest.json', obs=obs)
    variant_ids, variant_index = _variant_rows(obs, catalog, binding['finite_branch_review'])
    n = len(obs.source_uv)
    kind = np.zeros(n, dtype='u1')
    query_uv = np.zeros((n, 2), dtype='<f8')
    normal = np.zeros((n, 2), dtype='<f8')
    alpha_bounds = np.zeros((n, 2), dtype='<f8')
    sigma = np.full(n, np.nan, dtype='<f8')
    seen = np.zeros(n, bool); statistics_seen = np.zeros(n, dtype='u1')
    lookup = {(int(t), str(pid)): i for i, (t, pid) in enumerate(zip(obs.native_pts, obs.source_point_ids))}
    grouped_count = group_members = weight_count = formal_rows = ordinary_count = 0
    source_run_count = source_run_points = profile_count = snapshot_count = 0
    max_profile_transverse = 0.
    files = ['groups.jsonl', 'profiles.jsonl', 'source_runs.jsonl', 'source_rows.jsonl']
    handles = {name: (output/name).open('w', encoding='utf-8') for name in files}
    weights = (output/'weights.f64').open('wb')
    membership = (output/'group_rows.u64').open('wb')
    try:
        with SnapshotArrays(binding['source_arrays']['path'], binding['source_arrays']['sha256']) as saved:
            for encoded in stream_snapshot_frames(source['path']):
                frame = int(encoded['index']); pts = int(encoded['native_pts'])
                if frame != snapshot_count + 1:
                    raise ValueError('Complete original snapshot frame sequence changed')
                snapshot_count += 1
                rows = []
                for original in encoded['rows']:
                    # Do not decode archived estimator/noise proof arrays.
                    # Their complete immutable address is retained below.
                    keys = ('target', 'source_point_ids', 'uv', 'sigma', 'native_line_queries',
                        'source_feature_id', 'source_segment_index', 'original_vertex_indices')
                    rows.append(saved.decode({k: original[k] for k in keys if k in original}))
                point_ids, global_rows, entries = [], [], []
                source_ref = encoded['source_packet']
                packet = None
                for row_number, row in enumerate(rows):
                    local = []
                    for j, pid in enumerate(row['source_point_ids']):
                        global_row = lookup.get((pts, str(pid)), -1)
                        point_ids.append(str(pid)); global_rows.append(global_row)
                        if global_row < 0:
                            continue
                        if seen[global_row] or obs.frame_indices[global_row] != frame:
                            raise ValueError('Duplicate or foreign source primary key')
                        if np.asarray(row['uv'][j], dtype='<f8').tobytes() != obs.source_uv[global_row].tobytes():
                            raise ValueError('Frozen snapshot source UV differs from immutable row ' + str(global_row))
                        seen[global_row] = True; sigma[global_row] = row['sigma'][j]
                        variant = catalog.variants[variant_ids[int(variant_index[global_row])]]
                        if variant['observation_semantic'] == 'physical_point':
                            kind[global_row] = ENTITY_CORNER
                        entries.append((row, j, global_row, int(variant_index[global_row])))
                        local.append((j, global_row))
                        _json_line(handles['source_rows.jsonl'], {'source_row': global_row,
                            'source_packet': source_ref, 'snapshot_frame': frame, 'snapshot_row': row_number,
                            'snapshot_vertex': j, 'source_feature_id': row.get('source_feature_id'),
                            'source_segment_index': row.get('source_segment_index'),
                            'original_vertex_index': (row['original_vertex_indices'][j]
                                if len(row.get('original_vertex_indices', [])) == len(row['uv']) else None),
                            'native_query_id': str(pid) if row.get('native_line_queries') else None})
                    if local and row.get('native_line_queries'):
                        if packet is None:
                            packet = read_reference(source_ref)
                        q, normals, bounds, provenance = profile_semantics(row, packet)
                        for j, global_row in local:
                            kind[global_row] = TRUE_PROFILE
                            query_uv[global_row], normal[global_row], alpha_bounds[global_row] = q[j], normals[j], bounds[j]
                            transverse = np.array([normals[j, 1], -normals[j, 0]])
                            max_profile_transverse = max(max_profile_transverse,
                                abs(float(transverse @ (obs.source_uv[global_row]-q[j]))))
                            _json_line(handles['profiles.jsonl'], {'source_row': global_row, **provenance[j]})
                            profile_count += 1
                for group_number, raw_group in enumerate(encoded['whitening_groups']):
                    operator = saved.decode(raw_group['operator'])
                    indices = saved.decode(raw_group['point_indices'])
                    selected, matrix = validate_group_membership(indices, point_ids, global_rows, operator)
                    if selected is None:
                        continue
                    if np.any(statistics_seen[selected]):
                        raise ValueError('Source point enters more than one original statistical group')
                    statistics_seen[selected] = 1
                    matrix.tofile(weights); np.asarray(selected, dtype='<u8').tofile(membership)
                    _json_line(handles['groups.jsonl'], {'group_id': operator['group_id'], 'frame': frame,
                        'native_pts': pts, 'member_start': group_members, 'member_count': len(selected),
                        'weight_start': weight_count, 'rank': len(matrix), 'kind': 'original_dense_W',
                        'covariance_origin': operator['covariance_origin'],
                        'matrix_shape': list(matrix.shape), 'matrix_sha256': hashlib.sha256(matrix.tobytes()).hexdigest(),
                        'snapshot_json_pointer': '/payload/frames/' + str(frame-1) + '/whitening_groups/' + str(group_number),
                        'complete_design_and_noise_provenance': source,
                        'numerical_rank_evidence': operator.get('numerical_rank_evidence'),
                        'one_statistical_measurement_per_original_group': True})
                    grouped_count += 1; group_members += len(selected)
                    weight_count += matrix.size; formal_rows += len(matrix)
                if packet is None and entries:
                    packet = read_reference(source_ref)
                if packet is not None:
                    count, points = _ordered_source_runs(entries, packet, frame, source_ref, handles['source_runs.jsonl'])
                    source_run_count += count; source_run_points += points
                if progress is not None and frame % 50 == 0:
                    progress({'stage': 'semantic_export', 'frames': frame, 'source_rows': int(seen.sum()),
                        'full_W_groups': grouped_count, 'profiles': profile_count,
                        'elapsed_seconds': time.monotonic()-start})
        if snapshot_count != obs.manifest['frames'] or not seen.all():
            raise ValueError('Original metadata does not account for all immutable source rows: ' + str(int(seen.sum())))
        if not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError('Original source scalar localization scale is missing')
        # Ungrouped rows retain exactly their original independent x/y sigma.
        # This diagonal applies only outside all original correlated W groups.
        ordinary = np.flatnonzero(statistics_seen == 0)
        statistics_seen[ordinary] = 1; ordinary_count = len(ordinary)
        np.asarray(ordinary, dtype='<u8').tofile(output/'ordinary_rows.u64')
        formal_rows += 2*ordinary_count
    finally:
        weights.close(); membership.close()
        for handle in handles.values():
            handle.close()
    arrays = {
        'kind': _write_array(output/'kind.u8', kind, 'u1'),
        'variant_index': _write_array(output/'variant_index.u32', variant_index, '<u4'),
        'query_uv': _write_array(output/'query_uv.f64', query_uv, '<f8'),
        'profile_normal': _write_array(output/'normal.f64', normal, '<f8'),
        'alpha_bounds': _write_array(output/'alpha_bounds.f64', alpha_bounds, '<f8'),
        'source_sigma': _write_array(output/'sigma.f64', sigma, '<f8')}
    for name, filename, dtype, count in [('weights', 'weights.f64', '<f8', weight_count),
            ('group_rows', 'group_rows.u64', '<u8', group_members),
            ('ordinary_rows', 'ordinary_rows.u64', '<u8', ordinary_count)]:
        arrays[name] = {'file': filename, 'dtype': dtype, 'shape': [count],
            'layout': 'C', 'sha256': sha256_file(output/filename)}
    atomic_json(output/'materials.json', {'variant_ids': variant_ids, 'materials': catalog.catalog['materials'],
        'variants': {v: {'descriptor': catalog.variants[v]['descriptor'],
            'geometry': catalog.variants[v]['geometry']} for v in variant_ids},
        'finite_branch_review': binding['finite_branch_review'],
        'path_gaps_are_not_connectors': True, 'model_geometry_is_conditional_seed': True})
    assets = {name: {'file': name, 'sha256': sha256_file(output/name)} for name in files + ['materials.json']}
    manifest = {'schema': SCHEMA, 'shot_id': obs.manifest['shot_id'], 'observations': n,
        'frames': snapshot_count, 'material_labels': obs.manifest['material_labels'],
        'primary_key': ['shot_id', 'native_pts', 'source_point_id'],
        'immutable_source_arrays': obs.manifest['arrays'], 'source_snapshot': source,
        'source_arrays': binding['source_arrays'], 'binding': _reference(binding_path),
        'arrays': arrays, 'assets': assets, 'variant_ids': variant_ids,
        'semantic_kinds': {'finite_curve': FINITE_CURVE, 'entity_corner': ENTITY_CORNER, 'true_profile': TRUE_PROFILE},
        'semantic_counts': {name: int(np.count_nonzero(kind == value)) for name, value in
            [('finite_curve', FINITE_CURVE), ('entity_corner', ENTITY_CORNER), ('true_profile', TRUE_PROFILE)]},
        'statistics': {'full_grouped_W_included': True, 'original_groups': grouped_count,
            'grouped_members': group_members, 'ordinary_independent_rows': ordinary_count,
            'formal_residual_rows': formal_rows, 'every_source_counted_once': bool(np.all(statistics_seen == 1)),
            'group_W_sliced_or_reestimated': False, 'scalar_sigma_replaces_group_W': False,
            'common_modes': 'already marginalized by original full W; original design/noise/priors retained by exact snapshot JSON pointers',
            'objective': 'frozen sum of squared full W residuals plus original independent scalar rows'},
        'source_runs': {'proven_runs': source_run_count, 'points_in_runs': source_run_points,
            'world_direction_inferred': False, 'unproven_rows_remain_independent_finite_stations': True},
        'profiles': {'count': profile_count, 'query_center_is_not_measured_UV': True,
            'query_equation': 'cross(profile_normal,predicted_uv-query_uv)=0',
            'alpha': 'dot(profile_normal,predicted_uv-query_uv)',
            'normal_error': 'dot(profile_normal,predicted_uv-original_source_uv)',
            'finite_interval_not_relaxed_with_budget': True,
            'measured_UV_transverse_to_recorded_query_max_px': max_profile_transverse,
            'original_physical_width_midpoint_conversion_preserved_as_evidence': True},
        'all_original_rows_retained': bool(seen.all()), 'all_five_source_arrays_byte_verified': True,
        'old_geometry_or_fit_residuals_loaded': False, 'legacy_owner_constructed': False,
        'elapsed_seconds': time.monotonic()-start}
    atomic_json(output/'manifest.json', manifest)
    return check_semantic_packet(output/'manifest.json', obs=obs)


def check_semantic_packet(manifest_path, *, obs=None):
    path = Path(manifest_path); root = path.parent
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest['schema'] != SCHEMA:
        raise ValueError('Unknown semantic packet schema')
    for spec in list(manifest['arrays'].values()) + list(manifest['assets'].values()):
        file = root/spec['file']
        if sha256_file(file) != spec['sha256']:
            raise ValueError('Semantic packet bytes changed: ' + str(file))
        if 'dtype' in spec and file.stat().st_size != int(np.prod(spec['shape']))*np.dtype(spec['dtype']).itemsize:
            raise ValueError('Semantic array size mismatch')
    if obs is not None and manifest['immutable_source_arrays'] != obs.manifest['arrays']:
        raise ValueError('Semantic packet immutable source identity differs')
    n = manifest['observations']
    rows = np.fromfile(root/manifest['arrays']['group_rows']['file'], dtype='<u8')
    ordinary = np.fromfile(root/manifest['arrays']['ordinary_rows']['file'], dtype='<u8')
    if len(rows)+len(ordinary) != n or not np.array_equal(np.sort(np.r_[rows, ordinary]), np.arange(n)):
        raise ValueError('Full source statistical coverage is not an exact partition')
    return {'manifest': str(path), 'sha256': sha256_file(path), 'observations': n,
        'frames': manifest['frames'], 'material_labels': manifest['material_labels'],
        'statistics': manifest['statistics'], 'semantic_counts': manifest['semantic_counts'],
        'profiles': manifest['profiles'], 'all_original_rows_retained': True}


def apply_frozen_statistics(manifest_path, raw_pixel_residual, *, verify=True):
    """Apply full original W and ungrouped sigma to one complete raw state.

    The returned ordering is groups.jsonl order followed by ordinary source
    rows. This ordering differs from the historical per-frame concatenation;
    its squared norm is the same frozen statistical objective.
    """
    path = Path(manifest_path); root = path.parent
    if verify:
        check_semantic_packet(path)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    raw = np.asarray(raw_pixel_residual, dtype=float)
    if raw.shape != (manifest['observations'], 2) or not np.isfinite(raw).all():
        raise ValueError('Complete finite original raw native residual table required')
    weights = np.fromfile(root/manifest['arrays']['weights']['file'], dtype='<f8')
    members = np.fromfile(root/manifest['arrays']['group_rows']['file'], dtype='<u8')
    ordinary = np.fromfile(root/manifest['arrays']['ordinary_rows']['file'], dtype='<u8')
    sigma = np.fromfile(root/manifest['arrays']['source_sigma']['file'], dtype='<f8')
    result = []
    with (root/'groups.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            group = json.loads(line)
            begin, count = group['member_start'], group['member_count']
            start, rank = group['weight_start'], group['rank']
            matrix = weights[start:start+rank*2*count].reshape(rank, 2*count)
            result.append(matrix @ raw[members[begin:begin+count]].ravel())
    result.append((raw[ordinary]/sigma[ordinary, None]).ravel())
    return np.concatenate(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True)
    args = parser.parse_args()
    result = export_semantic_packet(args.data_dir, args.output_dir, args.binding,
        progress=lambda value: print(json.dumps(value, ensure_ascii=False), flush=True))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
