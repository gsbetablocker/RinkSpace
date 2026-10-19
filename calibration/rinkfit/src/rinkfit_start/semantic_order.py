"""Independent source-proven run order audit, without inventing orientation.

The saved source run determines membership and sequence. With no authenticated
world direction, both non-strict orientations are retained as alternatives.
Adjacent continuous board segments share one arclength chart; disconnected
print contours never acquire an invented connector. No minimum circle coverage
or equal station spacing is imposed by this audit.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from .data import sha256_file


def _kind(path):
    kind = path['kind']
    return {'ice_segment': 0, 'ice_circle': 1, 'board_path': 2, 'board_boundary': 3}.get(kind, kind)


def _board_lengths(model):
    width, rx, ry, power = np.asarray(model['blocks'][0]['values'], float)[:4]
    if not (0 < rx < .5 and 0 < ry < width/2 and power > 1):
        raise ValueError('Source order requires a legal current continuous board')
    def speed(angle):
        c, s = max(np.cos(angle), 0.), max(np.sin(angle), 0.)
        denominator = c**power+s**power
        scale = denominator**(-1/power)
        derivative = (c**(power-1)*s-s**(power-1)*c)/denominator
        return np.hypot(rx*scale*(-s+c*derivative), ry*scale*(c+s*derivative))
    arc = quad(speed, 0., np.pi/2, epsabs=1e-12, epsrel=1e-11)[0]
    return np.asarray([1-2*rx, arc, width-2*ry, arc]*2)


def _board_components(paths, variant):
    segments = {int(p['boundary_segment']) for p in paths if _kind(p) == 3
        and p.get('variant_id') == variant and int(p.get('boundary_segment', -1)) >= 0}
    if not segments:
        return []
    if segments == set(range(8)):
        return [list(range(8))]
    starts = sorted(s for s in segments if (s-1) % 8 not in segments)
    components = []
    for start in starts:
        component = []; segment = start
        while segment in segments:
            component.append(segment); segment = (segment+1) % 8
        components.append(component)
    return components


def _direction_candidate(coordinate, direction, closed, tolerance):
    difference = direction*np.diff(coordinate)
    if closed:
        # A single directed circuit may cross its seam. Each reversal by more
        # than floating tolerance needs one wrap; >one circuit remains invalid.
        increments = np.where(difference < -tolerance, difference+1., difference)
        span = float(increments.sum())
        valid = bool(np.all(increments >= -tolerance) and span <= 1.+tolerance)
        unwrapped = np.r_[coordinate[0], coordinate[0]+direction*np.cumsum(increments)]
        return {'valid': valid, 'span_normalized': span,
                'unwrapped_station': unwrapped.tolist(), 'seam_crossings': int(np.count_nonzero(difference < -tolerance))}
    return {'valid': bool(np.all(difference >= -tolerance)),
            'span_normalized': float(direction*(coordinate[-1]-coordinate[0])),
            'minimum_directed_increment': float(np.min(difference, initial=0.)),
            'unwrapped_station': coordinate.tolist(), 'seam_crossings': 0}


def audit_source_runs(manifest, model, stations, branches, *, tolerance=1e-10):
    """Audit every proven source run against one complete current state.

    ``manifest`` is its semantic manifest path, and ``model`` is the immutable
    model dictionary with block values updated to the current physical state.
    Only the continuous board lengths depend on those block values. Caller
    supplied station/branch vectors must retain all original source rows.
    """
    manifest_path = Path(manifest)
    semantic = json.loads(manifest_path.read_text(encoding='utf-8'))
    order_overlay = None
    if 'source_order_parent_manifest' in semantic or 'source_order_overlay' in semantic:
        from .semantic_resolution import load_semantics
        semantic, _, _, info = load_semantics(manifest_path,semantic['observations'])
        order_overlay = info['source_order_overlay']
    source_spec = semantic['assets']['source_runs.jsonl']
    source_path = manifest_path.parent/source_spec['file']
    if sha256_file(source_path) != source_spec['sha256']:
        raise ValueError('Source run provenance changed')
    stations = np.asarray(stations)
    raw_branches = np.asarray(branches)
    n = semantic['observations']
    if stations.shape != (n,) or raw_branches.shape != (n,):
        raise ValueError('Source order needs complete station and branch vectors')
    if not np.issubdtype(raw_branches.dtype, np.integer):
        if not np.isfinite(raw_branches).all() or np.any(raw_branches != np.floor(raw_branches)):
            raise ValueError('Fractional/nonfinite material branch is not an identity')
    branches = raw_branches.astype(np.int64)
    variant_materials = defaultdict(list)
    for material in model['materials']:
        for variant in {p.get('variant_id') for p in material['paths']}:
            variant_materials[variant].append(material)
    board_lengths = None
    records = []; failed = []; reasons = Counter(); by_material = defaultdict(Counter)
    with source_path.open(encoding='utf-8') as handle:
        for ordinal, line in enumerate(handle):
            source = json.loads(line)
            run_id = f'run/{ordinal:06d}'
            rows = np.asarray(source['source_rows'], dtype=np.int64)
            if (not source.get('source_order_verified') or rows.ndim != 1 or len(rows) < 2
                    or np.any(rows < 0) or np.any(rows >= n) or len(set(rows.tolist())) != len(rows)):
                raise ValueError('Invalid explicit source run membership/order')
            variant = semantic['variant_ids'][int(source['variant_index'])]
            materials = variant_materials.get(variant, [])
            if not materials:
                raise ValueError('Source run variant missing from physical model')
            # Multiple source board labels may bind the same model target.
            # Verify identical branch topology before using the shared chart.
            signature = lambda m: [(_kind(p), p.get('variant_id'), p.get('boundary_segment'), p.get('id'), bool(p.get('closed'))) for p in m['paths']]
            if any(signature(m) != signature(materials[0]) for m in materials[1:]):
                raise ValueError('Shared source variant has ambiguous path topology')
            material = materials[0]; paths = material['paths']; b = branches[rows]; s = stations[rows]
            record = {'run_id': run_id, 'frame': source['frame'], 'source_feature_id': source['source_feature_id'],
                'source_segment_index': source['source_segment_index'], 'source_rows': rows.tolist(),
                'variant_id': variant, 'material_labels': [m['label'] for m in materials],
                'source_closed': bool(source['source_closed']), 'world_direction': source.get('world_direction'),
                'source_provenance': {'file': str(source_path), 'line': ordinal+1, 'sha256': source_spec['sha256']}}
            reason = None; coordinate = s.copy(); world_closed = False
            if (not np.isfinite(s).all() or np.any(s < -tolerance) or np.any(s > 1.+tolerance)):
                reason = 'invalid_finite_station'
            elif np.any(b < 0) or np.any(b >= len(paths)):
                reason = 'invalid_material_branch'
            elif any(paths[int(branch)].get('variant_id') != variant for branch in b):
                reason = 'source_variant_branch_mismatch'
            else:
                kinds = {_kind(paths[int(branch)]) for branch in b}
                if kinds == {3}:
                    if board_lengths is None:
                        board_lengths = _board_lengths(model)
                    segments = [int(paths[int(branch)].get('boundary_segment', -1)) for branch in b]
                    if all(segment == -1 for segment in segments):
                        world_closed = True
                    elif any(segment < 0 for segment in segments):
                        reason = 'mixed_full_perimeter_and_segment_charts'
                    else:
                        components = _board_components(paths, variant)
                        common = [component for component in components if set(segments) <= set(component)]
                        if len(common) != 1:
                            reason = 'run_crosses_disconnected_board_support'
                        else:
                            component = common[0]; length = float(board_lengths[component].sum())
                            prefix = {}; cursor = 0.
                            for segment in component:
                                prefix[segment] = cursor; cursor += board_lengths[segment]
                            coordinate = np.asarray([prefix[segment]+station*board_lengths[segment]
                                for segment, station in zip(segments, s)])/length
                            world_closed = len(component) == 8
                            record.update({'coordinate_system': 'same continuous board component arclength',
                                'board_segments_in_cyclic_order': component, 'board_component_length_L': length,
                                'board_arclength_L': (coordinate*length).tolist()})
                elif len(set(b.tolist())) != 1:
                    reason = 'run_crosses_disconnected_material_paths'
                elif len(kinds) != 1:
                    reason = 'run_changes_physical_primitive'
                else:
                    world_closed = bool(paths[int(b[0])].get('closed', False))
                    record['coordinate_system'] = 'same finite physical path normalized parameter'
            record['world_closed'] = world_closed
            directions = source.get('world_direction')
            if directions is None:
                allowed = [1, -1]
            elif directions in (1, 'forward', '+'):
                allowed = [1]
            elif directions in (-1, 'backward', '-'):
                allowed = [-1]
            else:
                raise ValueError('Unsupported authenticated source/world direction')
            record['allowed_direction_candidates'] = allowed
            record['direction_was_inferred_from_source_UV'] = False
            if reason is None:
                candidates = {str(direction): _direction_candidate(coordinate, direction, world_closed, tolerance)
                    for direction in allowed}
                record['direction_candidates'] = candidates
                record['legal_direction_candidates'] = [int(k) for k, v in candidates.items() if v['valid']]
                record['valid'] = bool(record['legal_direction_candidates'])
                if not record['valid']:
                    reason = 'no_legal_single_direction_monotone_run'
            else:
                record.update(direction_candidates={}, legal_direction_candidates=[], valid=False)
            record['failure_reason'] = reason
            records.append(record)
            by_material[material['label']]['runs'] += 1
            if not record['valid']:
                failed.append(run_id); reasons[reason] += 1
                by_material[material['label']]['failed_runs'] += 1
    result = {'schema': 'rinkfit.independent_source_run_order_audit.v1',
        'semantic_manifest_sha256': sha256_file(manifest_path), 'source_runs_sha256': source_spec['sha256'],
        'observations': n, 'proven_source_runs': len(records), 'passed_runs': len(records)-len(failed),
        'failed_runs': len(failed), 'failed_run_ids': failed, 'all_run_obligations_checked': True,
        'passed': not failed, 'failure_reasons': dict(reasons),
        'by_material': {label: dict(count) for label, count in by_material.items()},
        'station_monotonicity_tolerance': tolerance, 'strict_station_spacing_imposed': False,
        'minimum_world_circle_coverage_imposed': False,
        'direction_not_declared_by_source_remains_two_candidates': True,
        'records': records}
    if order_overlay is not None:
        result.update(audit_source_runs_retained=order_overlay['audit_run_count'],
            navigation_query_sequence_only_runs=order_overlay['audit_run_count'],
            navigation_query_sequence_only_rows=order_overlay['audit_row_count'],
            navigation_audit_registry=order_overlay['audit_source_runs'],
            navigation_classifications=order_overlay['classifications'],
            source_order_overlay_sha256=order_overlay['overlay_sha256'],
            active_hard_source_runs=len(records),
            original_navigation_order_verified=False,
            all_original_navigation_runs_are_real_traces=False,
            order_result='no_independent_trace_order_obligations' if not records else 'active_trace_obligations_checked')
    return result
