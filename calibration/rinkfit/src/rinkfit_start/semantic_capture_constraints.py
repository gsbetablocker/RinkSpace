"""Transport chosen run direction/seam seeds without changing source facts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .data import sha256_file
from .runtime import atomic_json


def export_run_constraint_seed(model_path, capture_dir):
    model_path, capture = Path(model_path), Path(capture_dir)
    model = json.loads(model_path.read_text(encoding='utf-8'))
    audit_path = capture/'source_order_after.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if not audit['passed'] or audit['failed_runs']:
        raise ValueError('A complete source-run legal seed is required')
    if (capture/'run_constraint_seed.json').exists():
        raise ValueError('Run constraint seed already frozen')
    comparisons_path = capture/'run_capture_comparisons.json'
    if comparisons_path.exists():
        comparisons = json.loads(comparisons_path.read_text(encoding='utf-8'))
    elif audit.get('active_hard_source_runs') == 0 and not audit['records']:
        comparisons = []
    else:
        raise ValueError('A nonempty source-run registry requires capture comparisons')
    selected = {item['run_id']: item['selected']['direction_seed'] for item in comparisons if not item['unresolved']}
    station = np.fromfile(capture/'station.f64', '<f8'); branch = np.fromfile(capture/'path.u32', '<u4')
    material_paths = json.dumps(model['materials'], ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    path_hash = hashlib.sha256(material_paths).hexdigest()
    records = []
    by_label = {material['label']: material for material in model['materials']}
    for item in audit['records']:
        rows = np.asarray(item['source_rows'], dtype=np.int64)
        direction = selected.get(item['run_id'], item['legal_direction_candidates'][0])
        if direction not in item['legal_direction_candidates']:
            raise ValueError('Chosen capture direction does not reproduce independent order legality')
        if item.get('coordinate_system') == 'same continuous board component arclength':
            raw_coordinate = np.asarray(item['board_arclength_L'])/item['board_component_length_L']
            method = 'current_shared_board_component_arclength'
        else:
            raw_coordinate = station[rows]
            method = 'finite_path_station'
        alternatives = []
        for sign in item['allowed_direction_candidates']:
            candidate = item['direction_candidates'][str(sign)]
            unwrapped = np.asarray(candidate['unwrapped_station'], dtype=float)
            lifts = np.rint(unwrapped-raw_coordinate).astype(np.int64)
            if np.max(abs(unwrapped-raw_coordinate-lifts), initial=0.) > 1e-8:
                raise ValueError('Source-run seam witness is not an integer lift of the same finite path')
            alternatives.append({'direction': sign, 'integer_lifts': lifts.tolist(),
                'unwrapped_seed_coordinate': unwrapped.tolist(), 'seed_valid': candidate['valid'],
                'seed_span': candidate['span_normalized']})
        chosen = next(value for value in alternatives if value['direction'] == direction)
        material = by_label[item['material_labels'][0]]
        records.append({'run_id': item['run_id'], 'frame': item['frame'], 'source_rows': rows.tolist(),
            'source_feature_id': item['source_feature_id'], 'source_segment_index': item['source_segment_index'],
            'variant_id': item['variant_id'], 'branch_indices': branch[rows].tolist(),
            'branch_path_ids': [material['paths'][int(value)]['id'] for value in branch[rows]],
            'coordinate_method': method, 'world_path_closed': item['world_closed'],
            'source_closed': item['source_closed'], 'source_world_direction_fact': item['world_direction'],
            'chosen_seed_direction': direction, 'chosen_integer_lifts': chosen['integer_lifts'],
            'orientation_alternatives': alternatives,
            'board_segments_in_cyclic_order': item.get('board_segments_in_cyclic_order'),
            'monotonicity_constraint': 'direction*((coordinate[j+1]+lift[j+1])-(coordinate[j]+lift[j])) >= 0',
            'closed_span_constraint': 'direction*((coordinate[last]+lift[last])-(coordinate[first]+lift[first])) <= 1' if item['world_closed'] else None,
            'current_board_arclength_must_follow_shared_geometry': method == 'current_shared_board_component_arclength',
            'minimum_circle_coverage_imposed': False, 'direction_choice_is_material_identity_fact': False,
            'material_paths_sha256': path_hash})
    result = {'schema': 'rinkfit.source-run-constraint-initialization-seed.v1',
        'runs': len(records), 'all_source_run_obligations_retained': True,
        'complete_original_path_topology_sha256': path_hash,
        'path_topology_hash_definition': 'UTF-8 json(model.materials, ensure_ascii=False, sort_keys=True, separators=comma-colon, allow_nan=False)',
        'model': {'path': str(model_path), 'sha256': sha256_file(model_path)},
        'semantic_manifest_sha256': audit['semantic_manifest_sha256'],
        'source_runs_sha256': audit['source_runs_sha256'],
        'station_sha256': sha256_file(capture/'station.f64'), 'branch_sha256': sha256_file(capture/'path.u32'),
        'independent_order_audit': {'path': str(audit_path), 'sha256': sha256_file(audit_path)},
        'source_direction_not_declared_remains_two_hypotheses': True,
        'chosen_discrete_hypothesis_role': 'initialization; a solver using this fixed orientation/lift must declare and hash the candidate, other hypotheses remain available',
        'non_strict_station_order': True, 'equal_station_spacing_constraint_added': False,
        'source_UV_ID_material_variant_W_geometry_changed': False,
        'records': records}
    if 'source_order_overlay_sha256' in audit:
        if audit.get('active_hard_source_runs') != len(records):
            raise ValueError('Active hard source-run seed count changed')
        result.update(active_hard_source_runs=len(records),
            audit_source_runs_retained=audit['audit_source_runs_retained'],
            navigation_query_sequence_only_runs=audit['navigation_query_sequence_only_runs'],
            navigation_query_sequence_only_rows=audit['navigation_query_sequence_only_rows'],
            navigation_audit_registry=audit['navigation_audit_registry'],
            navigation_classifications=audit['navigation_classifications'],
            source_order_overlay_sha256=audit['source_order_overlay_sha256'],
            original_navigation_order_verified=False,
            all_original_navigation_runs_are_real_traces=False)
    atomic_json(capture/'run_constraint_seed.json', result)
    return {k: v for k, v in result.items() if k != 'records'}
