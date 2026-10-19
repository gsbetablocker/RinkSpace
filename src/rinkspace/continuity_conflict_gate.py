"""Conservative publication gate for already fitted unresolved camera graphs.

This does not refit or replace any camera pose. Known conflicts cut the observed
image graph in time. A surviving pose needs an actual anchor in its remaining
connected segment, and its primary provenance must refer to that segment.
Losing a competing hypothesis to an error budget is not conflict resolution.
"""
from __future__ import annotations

import copy
from fractions import Fraction

from .continuity_graph_observed import _actual_anchor_pairs, _graph

ALGORITHM_VERSION = 'unresolved-camera-conflict-publication-gate-1'
DRAWABLE = {'model_reobserved', 'image_tracked', 'candidate', 'predicted'}


def _time(row):
    return int(row['pts']) * Fraction(row['time_base'])


def _actual_anchor(row):
    if row.get('status') not in DRAWABLE or row.get('actual_anchor_validation_rejected'):
        return False
    candidate = row.get('model_anchor_candidate')
    if not candidate or not candidate.get('projection_chain'):
        return False
    try:
        return _actual_anchor_pairs(row, candidate) is not None
    except (ValueError, KeyError, TypeError):
        return False


def gate_unresolved_components(frames, graph):
    """Return gated deep copies; complete local point pairs must be loaded first."""
    unresolved = {c['id'] for c in graph.get('components', [])
                  if c.get('reason') == 'unresolved_anchor_hypotheses_retained'}
    result = copy.deepcopy(frames)
    groups = {}
    for row in result:
        if row.get('offline_camera_graph_id') in unresolved:
            identity = (row.get('source_sha256'), row.get('shot_id'),
                        row.get('camera_component_id'), row['offline_camera_graph_id'])
            groups.setdefault(identity, []).append(row)
    diagnostics = {'algorithm': ALGORITHM_VERSION, 'numeric_fit_changed': False,
        'known_conflicts_do_not_expire_with_hypothesis_budget': True,
        'time_barriers_apply_even_to_long_keyframe_edges': True,
        'real_anchor_and_primary_pose_lineage_required_in_same_surviving_segment': True,
        'independent_geometry_validation': False, 'gated_frames': [], 'segments': [],
        'removed_relative_edges': [], 'invalid_relative_edges': []}
    for identity, rows in groups.items():
        rows.sort(key=_time)
        matrices, edges, _, _, rejected = _graph(rows)
        diagnostics['invalid_relative_edges'].extend(rejected)
        # Ambiguous nodes are observations of unresolved contradictions. A
        # later locally equivalent/single surviving hypothesis cannot erase them.
        conflict = {i for i, row in enumerate(rows) if row.get('status') == 'ambiguous'}
        epochs, epoch = {}, 0
        for i in range(len(rows)):
            if i in conflict:
                epoch += 1
            else:
                epochs[i] = epoch
        nodes = set(matrices) - conflict
        adjacency = {i: [] for i in nodes}
        for edge in edges:
            a, b = edge['current'], edge['parent']
            if a not in nodes or b not in nodes or epochs[a] != epochs[b]:
                diagnostics['removed_relative_edges'].append({
                    'source_sha256': identity[0], 'shot_id': identity[1],
                    'current_sample_index': rows[a]['sample_index'],
                    'parent_sample_index': rows[b]['sample_index'],
                    'reason': 'conflicting_node_or_edge_crosses_known_temporal_conflict'})
                continue
            adjacency[a].append(b); adjacency[b].append(a)
        remaining, segment_for, valid_anchors_for = set(nodes), {}, {}
        while remaining:
            todo, found = [min(remaining)], set()
            while todo:
                i = todo.pop()
                if i in found:
                    continue
                found.add(i)
                todo.extend(j for j in adjacency[i] if j not in found)
            remaining -= found
            anchors = {rows[i]['sample_index'] for i in found if _actual_anchor(rows[i])}
            segment_id = f'conflict_gate_segment_{len(diagnostics["segments"]):04d}'
            diagnostics['segments'].append({'id': segment_id, 'source_sha256': identity[0],
                'shot_id': identity[1], 'offline_camera_graph_id': identity[3],
                'start_s': min(rows[i]['source_time_s'] for i in found),
                'end_s': max(rows[i]['source_time_s'] for i in found),
                'observed_node_count': len(found), 'actual_anchor_sample_indexes': sorted(anchors),
                'has_real_anchor_after_conflict_partition': bool(anchors)})
            for i in found:
                segment_for[i], valid_anchors_for[i] = segment_id, anchors
        for i, row in enumerate(rows):
            if row.get('status') not in DRAWABLE:
                continue
            anchors = valid_anchors_for.get(i, set())
            lineage = row.get('anchor_lineage') or []
            primary = lineage[0].get('sample_index') if lineage else (
                row['sample_index'] if _actual_anchor(row) and row.get('status') in {'model_reobserved', 'candidate'} else None)
            allowed = bool(anchors and primary in anchors)
            evidence = {'policy': ALGORITHM_VERSION, 'unresolved_graph_id': identity[3],
                'surviving_segment_id': segment_for.get(i), 'segment_real_anchor_samples': sorted(anchors),
                'published_primary_anchor_sample': primary, 'passed': allowed,
                'not_independent_accuracy_validation': True}
            row['unresolved_conflict_gate'] = evidence
            if allowed:
                continue
            reason = ('known_anchor_conflict_no_real_anchor_in_surviving_observed_segment' if not anchors
                      else 'published_pose_lineage_crosses_known_anchor_conflict')
            before = {key: copy.deepcopy(row.get(key)) for key in (
                'status', 'reason', 'projection_chain', 'H_image_from_model', 'H_model_from_image',
                'anchor_id', 'anchor_age_s', 'model_bridge', 'anchor_lineage', 'model_anchor_hypotheses')}
            row['conflict_gate_previous_output'] = before
            row.update(status='ambiguous', reason=reason, projection_chain=None,
                H_image_from_model=None, H_model_from_image=None, anchor_id=None, model_bridge=None,
                independent_validation=False, map_usable_for_metric_output=False)
            diagnostics['gated_frames'].append({'source_sha256': row['source_sha256'],
                'source_time_s': row['source_time_s'], 'pts': row['pts'], 'time_base': row['time_base'],
                'sample_index': row['sample_index'], 'previous_status': before['status'],
                'reason': reason, 'gate': evidence})
    diagnostics['gated_frame_count'] = len(diagnostics['gated_frames'])
    return {'frames': result, 'diagnostics': diagnostics}
