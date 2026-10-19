"""Offline camera constraints from actual same-shot ice correspondences.

The shared rink geometry is immutable. A local registrar's arbitrary gauge is
never a map anchor. Real model reobservations constrain camera corrections and
relative image pairs constrain their variation. Similarity parameters (not the
nine homography coefficients) are optimized; a bounded plane correction is a
separately scored alternative. All error ranges are engineering budgets, not
calibrated statistical confidence intervals or independent rink validation.
"""
from __future__ import annotations

import copy
import heapq
from dataclasses import asdict, dataclass

import numpy as np

from .geometry import project

ALGORITHM_VERSION = "same-shot-observed-camera-graph-1"


@dataclass(frozen=True)
class ContinuityGraphConfig:
    max_engineering_range_px_at_960: float = 12.
    anchor_sigma_px_at_960: float = 2.
    max_anchor_p95_px_at_960: float = 4.
    max_edge_holdout_p95_px_at_960: float = 3.2
    max_edge_holdout_increase_px_at_960: float = 1.2
    conflict_p95_px_at_960: float = 12.
    max_points_per_edge: int = 32
    max_points_per_anchor: int = 48
    max_nfev: int = 24
    compare_constrained_plane: bool = True

    def __post_init__(self):
        numeric = [v for v in asdict(self).values() if not isinstance(v, bool)]
        if any(v <= 0 for v in numeric):
            raise ValueError("Graph budgets and counts must be positive")
        if self.max_points_per_edge < 8 or self.max_points_per_anchor < 8:
            raise ValueError("Camera graph needs spatially distributed samples")


def _matrix(value):
    result = np.asarray(value, float)
    if result.shape != (3, 3) or not np.isfinite(result).all() or abs(np.linalg.det(result)) < 1e-12:
        raise ValueError("Expected a finite nonsingular 3x3 image transform")
    return result / np.linalg.norm(result[2])


def _pick(values, limit):
    return np.asarray(values)[np.linspace(0, len(values)-1, min(len(values), limit)).astype(int)]


def _chain_xy(chain, points):
    """Project model points to sample pixels, applying reference lens once."""
    h = np.c_[points, np.ones(len(points))] @ _matrix(chain['H_reference_undistorted_from_model']).T
    good = chain.get('reference_visible_branch_sign', 1)*h[:, 2] > 1e-10
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        xy = h[:, :2]/h[:, 2:]
        distortion = chain.get('reference_radial_distortion')
        if distortion is not None:
            centre = np.asarray(distortion['centre_image'], float)
            delta = xy-centre
            radius2 = np.sum((delta/float(distortion['scale_image']))**2, axis=1)
            k1 = float(distortion['k1'])
            good &= 1+3*k1*radius2 > 0
            xy = centre+delta*(1+k1*radius2[:, None])
        raw = np.c_[xy, np.ones(len(xy))] @ _matrix(chain['H_current_sample_from_reference_raw']).T
        good &= chain.get('raw_transfer_visible_branch_sign', 1)*raw[:, 2] > 1e-10
        xy = raw[:, :2]/raw[:, 2:]
    good &= np.isfinite(xy).all(axis=1)
    return xy, good


def _transport(chain, target_local, anchor_local, correction=None):
    result = copy.deepcopy(chain)
    transfer = np.linalg.inv(target_local) @ anchor_local @ _matrix(chain['H_current_sample_from_reference_raw'])
    if correction is not None:
        transfer = correction @ transfer
    result['H_current_sample_from_reference_raw'] = transfer.tolist()
    probe = result.get('reference_branch_probe_raw')
    if probe is not None:
        result['raw_transfer_visible_branch_sign'] = 1 if transfer[2] @ np.r_[probe, 1.] > 0 else -1
    return result


def _anchor(row):
    candidate = row.get('model_anchor_candidate')
    if candidate and candidate.get('projection_chain'):
        return {**candidate, 'sample_index': row['sample_index'],
                'source_time_s': row['source_time_s'], 'pts': row['pts'],
                'time_base': row['time_base'],
                'reference_id': candidate.get('reference_id', row.get('anchor_id')),
                'fit': candidate.get('fit', row.get('ice_reobservation', {}))}
    if row.get('status') in {'candidate', 'model_reobserved'} and row.get('projection_chain'):
        return {'projection_chain': row['projection_chain'], 'reference_id': row.get('anchor_id'),
                'sample_index': row['sample_index'], 'source_time_s': row['source_time_s'],
                'pts': row['pts'], 'time_base': row['time_base'], 'fit': row.get('ice_reobservation', {}),
                'model_reference': row.get('model_reference')}
    return None


def _group(row):
    local = row.get('local_registration', {})
    return (row.get('source_sha256'), row.get('shot_id'),
            row.get('camera_component_id', (row.get('propagation_segment_id'), local.get('graph_component_id'))))


def _edge_variance(local):
    """A residual/support-dependent path budget; keyframes do not reset it."""
    residual = float(local.get('inlier_p95_reference_px') or 1.)
    count = max(float(local.get('train_inliers') or 6.), 6.)
    support = max(float(local.get('support_fraction') or .025), .025)
    # The floor acknowledges correlated image errors. More matched pixels do
    # not make successive tracking steps into independent perfect measurements.
    step = max(.18, residual/np.sqrt(min(count, 128)/8)) / np.sqrt(min(1., support/.15))
    return step*step


def _graph(rows):
    by_sample = {r['sample_index']: i for i, r in enumerate(rows)}
    if len(by_sample) != len(rows):
        raise ValueError("sample_index must be unique within camera component")
    matrices, edges, rejected = {}, [], []
    for i, row in enumerate(rows):
        local = row.get('local_registration', {})
        if local.get('status') != 'candidate' or local.get('H_reference_from_current') is None:
            continue
        try:
            matrices[i] = _matrix(local['H_reference_from_current'])
        except ValueError:
            rejected.append({'sample_index': row['sample_index'], 'reason': 'invalid_local_matrix'})
    for i in matrices:
        row, local = rows[i], rows[i]['local_registration']
        parent = by_sample.get(local.get('edge_keyframe_sample_index'))
        if local.get('reference_frame_only') or parent is None or parent == i or parent not in matrices:
            continue
        x = np.asarray(local.get('fit_current_xy', []), float)
        y = np.asarray(local.get('fit_keyframe_xy', []), float)
        if x.ndim != 2 or x.shape != y.shape or x.shape[1:] != (2,) or len(x) < 6 or not np.isfinite([x, y]).all():
            rejected.append({'sample_index': row['sample_index'], 'reason': 'missing_actual_relative_point_pairs'})
            continue
        # Do not trust a shared component label if the archived matrices and
        # actual keyframe correspondences no longer describe the same edge.
        transfer = np.linalg.inv(matrices[parent]) @ matrices[i]
        width, height = row.get('sample_size', [960, 540])
        normalization = np.diag([1/width, 1/height, 1.])
        condition = float(np.linalg.cond(normalization @ transfer @ np.linalg.inv(normalization)))
        if not np.isfinite(condition) or condition >= 50:
            rejected.append({'sample_index': row['sample_index'], 'reason': 'relative_edge_extrapolation_condition',
                             'normalized_edge_condition': condition})
            continue
        error = np.linalg.norm(project(transfer, x)-y, axis=1)
        if not np.isfinite(error).all() or np.percentile(error, 95) > max(4., 2*float(local.get('inlier_p95_reference_px') or 2.)):
            rejected.append({'sample_index': row['sample_index'], 'reason': 'matrix_edge_point_pair_disagreement'})
            continue
        edges.append({'current': i, 'parent': parent, 'current_xy': x, 'parent_xy': y,
                      'variance': _edge_variance(local), 'baseline_p95': float(np.percentile(error, 95))})
    adjacency = {i: [] for i in matrices}
    for edge in edges:
        a, b = edge['current'], edge['parent']
        adjacency[a].append((b, edge['variance']))
        adjacency[b].append((a, edge['variance']))
    components, remaining = [], set(matrices)
    while remaining:
        todo, found = [min(remaining)], set()
        while todo:
            i = todo.pop()
            if i in found:
                continue
            found.add(i)
            todo.extend(j for j, _ in adjacency[i] if j not in found)
        remaining -= found
        components.append(sorted(found))
    return matrices, edges, adjacency, components, rejected


def _distances(adjacency, origin):
    distance, steps, queue = {origin: 0.}, {origin: 0}, [(0., origin)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distance[node]:
            continue
        for other, variance in adjacency[node]:
            proposed = cost+variance
            if proposed < distance.get(other, np.inf):
                distance[other], steps[other] = proposed, steps[node]+1
                heapq.heappush(queue, (proposed, other))
    return distance, steps


def _normalizer(size):
    w, h = size
    return np.array([[1/w, 0., -.5], [0., 1/w, -h/(2*w)], [0., 0., 1.]])


def _correction(parameters, normalizer):
    tx, ty, logscale, angle = parameters[:4]
    c, s = np.cos(angle), np.sin(angle)
    if len(parameters) == 4:
        stretch, perspective = np.eye(2)*np.exp(logscale), [0., 0.]
    else:
        anisotropy, shear, px, py = parameters[4:]
        stretch = np.array([[np.exp(logscale+anisotropy), shear], [0., np.exp(logscale-anisotropy)]])
        perspective = [px, py]
    matrix = np.eye(3)
    matrix[:2, :2] = np.array([[c, -s], [s, c]]) @ stretch
    matrix[:2, 2] = [tx, ty]
    matrix[2, :2] = perspective
    return np.linalg.inv(normalizer) @ matrix @ normalizer


def _visible(points, valid, size):
    w, h = size
    return valid & (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)


def _fit_component(rows, ids, matrices, edges, anchors, probes, config, *, plane=False):
    """Optimize camera corrections while measuring true relative pixel pairs."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    base_id = min(anchors)
    base = anchors[base_id]
    subset = set(ids)
    edges = [edge for edge in edges if edge['current'] in subset and edge['parent'] in subset]
    # A non-keyframe camera can be obtained from its observed edge after the
    # optimization. All anchors and every edge parent remain explicit nodes.
    nodes = sorted(set(anchors) | {e['parent'] for e in edges} | {ids[0]})
    ordinal = {value: index for index, value in enumerate(nodes)}
    selected_edges = [e for e in edges if e['current'] in ordinal]
    d = 8 if plane else 4
    size = rows[base_id].get('sample_size', [960, 540])
    scale = size[0]/960
    normalizer = _normalizer(size)
    anchor_data = []
    for i, anchor in anchors.items():
        transported = _transport(base['projection_chain'], matrices[i], matrices[base_id])
        predicted, vp = _chain_xy(transported, probes)
        target, vt = _chain_xy(anchor['projection_chain'], probes)
        valid = vp & _visible(target, vt, size) & (np.max(np.abs(predicted), axis=1) < 3*size[0])
        if np.count_nonzero(valid) < 8:
            return {'accepted': False, 'method': 'constrained_plane' if plane else 'similarity',
                    'reason': 'insufficient_common_visible_model_probes', 'anchor_sample_index': rows[i]['sample_index']}
        pairs = _pick(np.c_[predicted[valid], target[valid]], config.max_points_per_anchor)
        anchor_data.append((i, pairs[:, :2], pairs[:, 2:]))
    fitting_edges, withheld_edges = [], []
    for edge in selected_edges:
        pairs = _pick(np.c_[edge['current_xy'], edge['parent_xy']], config.max_points_per_edge)
        test = np.arange(len(pairs)) % 4 == 0
        for where, mask in ((fitting_edges, ~test), (withheld_edges, test)):
            where.append((edge, pairs[mask, :2], pairs[mask, 2:]))
    # Fixed dimensions permit sparse finite differences. Only the two actual
    # endpoint cameras participate in an edge; no temporal interpolation term.
    blocks = [(2*len(x), [ordinal[e['current']], ordinal[e['parent']]]) for e, x, _ in fitting_edges]
    blocks += [(2*len(x), [ordinal[i]]) for i, x, _ in anchor_data]
    blocks += [(d, [i]) for i in range(len(nodes))]
    sparsity = lil_matrix((sum(length for length, _ in blocks), len(nodes)*d), dtype=int)
    cursor = 0
    for length, members in blocks:
        for member in members:
            sparsity[cursor:cursor+length, member*d:(member+1)*d] = 1
        cursor += length
    transfers = {e['current']: np.linalg.inv(matrices[e['current']]) @ matrices[e['parent']] for e in edges}

    def unpack(value):
        return {i: _correction(p, normalizer) for i, p in zip(nodes, value.reshape(-1, d))}

    def edge_errors(corrections, observations):
        answer = []
        for edge, current, parent in observations:
            i, j = edge['current'], edge['parent']
            transform = corrections[i] @ transfers[i] @ np.linalg.inv(corrections[j])
            answer.append((project(transform, parent)-current))
        return answer

    def objective(value):
        corrections = unpack(value)
        residual = [error.ravel()/max(.5*scale, np.sqrt(edge['variance']))
                    for (edge, _, _), error in zip(fitting_edges, edge_errors(corrections, fitting_edges))]
        residual += [(project(corrections[i], x)-y).ravel()/(config.anchor_sigma_px_at_960*scale)
                     for i, x, y in anchor_data]
        # A proper engineering restriction on extra perspective flexibility,
        # not a claim that a common calibrated camera/tripod has been measured.
        prior = np.array([3., 3., 1., 1.] + ([.025, .025, .015, .015] if plane else []))
        residual += [(p/prior).ravel() for p in value.reshape(-1, d)]
        return np.nan_to_num(np.concatenate(residual), nan=1e6, posinf=1e6, neginf=-1e6)

    bounds = np.array([.35, .35, .2, .2] + ([.04, .04, .04, .04] if plane else []))
    solved = least_squares(objective, np.zeros(len(nodes)*d), jac_sparsity=sparsity.tocsr(),
        bounds=(-np.tile(bounds, len(nodes)), np.tile(bounds, len(nodes))),
        max_nfev=config.max_nfev, loss='soft_l1', f_scale=1., xtol=1e-5, ftol=1e-5, gtol=1e-5)
    corrections = unpack(solved.x)
    anchor_errors = [np.linalg.norm(project(corrections[i], x)-y, axis=1) for i, x, y in anchor_data]
    heldout = [np.linalg.norm(error, axis=1) for error in edge_errors(corrections, withheld_edges)]
    identity = {i: np.eye(3) for i in nodes}
    baseline = [np.linalg.norm(error, axis=1) for error in edge_errors(identity, withheld_edges)]
    # Gate the worst anchor/edge, not only a pooled percentile in which a short
    # bad transition disappears under thousands of good points.
    anchor_p95 = max(float(np.percentile(x, 95)) for x in anchor_errors)
    heldout_p95 = max((float(np.percentile(x, 95)) for x in heldout), default=0.)
    baseline_p95 = max((float(np.percentile(x, 95)) for x in baseline), default=0.)
    increase = max((float(np.percentile(a, 95)-np.percentile(b, 95)) for a, b in zip(heldout, baseline)), default=0.)
    accepted = (anchor_p95 <= config.max_anchor_p95_px_at_960*scale and
                heldout_p95 <= config.max_edge_holdout_p95_px_at_960*scale and
                increase <= config.max_edge_holdout_increase_px_at_960*scale)
    # Non-keyframes inherit a correction ONLY through their measured edge.
    for edge in edges:
        i, j = edge['current'], edge['parent']
        if i not in corrections and j in corrections:
            t = transfers[i]
            corrections[i] = t @ corrections[j] @ np.linalg.inv(t)
    score = float(np.median(np.concatenate(heldout))) if heldout else 0.
    return {'accepted': bool(accepted), 'method': 'constrained_plane' if plane else 'similarity',
            'reason': 'joint_image_edges_and_model_anchors_consistent' if accepted else 'joint_constraints_disagree',
            'camera_degrees_of_freedom': d, 'optimized_camera_nodes': len(nodes),
            'actual_relative_edges_optimized': len(selected_edges), 'optimizer_success': bool(solved.success),
            'optimizer_nfev': int(solved.nfev), 'anchor_worst_p95_sample_px': anchor_p95,
            'edge_holdout_worst_p95_sample_px': heldout_p95, 'edge_baseline_worst_p95_sample_px': baseline_p95,
            'edge_holdout_worst_increase_sample_px': increase, 'edge_holdout_median_sample_px': score,
            'withheld_points_are_independent_geometry_validation': False,
            'camera_nodes_sample_index': [rows[i]['sample_index'] for i in nodes],
            'base_anchor_sample_index': rows[base_id]['sample_index'],
            'corrections': corrections, 'base_id': base_id}


def _summary(fit):
    return {k: v for k, v in fit.items() if k not in {'corrections', 'base_id'}}


def _lineage(anchor, row, distance, steps, sigma):
    return {'reference_id': anchor.get('reference_id'), 'sample_index': anchor['sample_index'],
            'pts': anchor['pts'], 'time_base': anchor['time_base'], 'source_time_s': anchor['source_time_s'],
            'future_anchor': anchor['source_time_s'] > row['source_time_s'], 'observed_edge_count': steps,
            'engineering_range_sample_px': sigma+float(np.sqrt(distance)),
            'range_is_calibrated_confidence_interval': False}


def _publish(row, chain, anchor, lineage, *, reason, engineering_range, joint=False):
    row['status'] = 'model_reobserved' if _anchor(row) else 'image_tracked'
    row['reason'], row['projection_chain'] = reason, chain
    row['anchor_id'] = anchor.get('reference_id')
    row['anchor_age_s'] = abs(row['source_time_s']-anchor['source_time_s'])
    row['anchor_lineage'] = lineage
    row['future_anchor_used'] = any(item['future_anchor'] for item in lineage)
    future = [item for item in lineage if item['future_anchor']]
    if future:
        row['future_anchor_observation'] = min(future, key=lambda item: item['source_time_s'])
    if row.get('local_registration', {}).get('reference_frame_only'):
        row['last_image_observation_age_s'] = None
        row['image_observation_support'] = 'offline_connected_future_relative_edges; local_identity_is_not_an_observation'
    else:
        row['last_image_observation_age_s'] = 0.
        row['image_observation_support'] = 'actual_current_to_observed_keyframe_point_pairs'
    if anchor.get('model_reference'):
        row['model_reference'] = copy.deepcopy(anchor['model_reference'])
    row['camera_uncertainty'] = {'engineering_range_sample_px': float(engineering_range),
        'calibrated_confidence_interval': False, 'independent_accuracy_measurement': False,
        'method': 'actual_edge_path_budget_plus_joint_anchor_and_withheld_edge_residual' if joint else 'actual_edge_path_budget',
        'keyframe_replacement_resets_budget': False}
    row['uncertainty'] = {'kind': 'engineering_not_calibrated_confidence',
        'estimated_error_px_at_960': float(engineering_range)*960/row.get('sample_size', [960])[0],
        'absolute_model_error_included': False, 'accepted': True,
        'reference': row['camera_uncertainty']['method']}
    row['independent_validation'] = False
    row['map_usable_for_metric_output'] = False
    distortion = chain.get('reference_radial_distortion')
    if not distortion or not distortion.get('k1'):
        h = (_matrix(chain['H_source_image_from_sample']) @ _matrix(chain['H_current_sample_from_reference_raw'])
             @ _matrix(chain['H_reference_undistorted_from_model']))
        row['H_image_from_model'], row['H_model_from_image'] = h.tolist(), np.linalg.inv(h).tolist()
    else:
        row['H_image_from_model'], row['H_model_from_image'] = None, None


def optimize_continuity(frames, model_probe_xy, config=None, *, progress=None):
    """Return JSON-serializable ``{frames, diagnostics}`` without input mutation.

    Required frame fields are source identity, shot/component identity, exact
    PTS, sample size and complete ``local_registration`` observations. A genuine
    model observation is read from ``model_anchor_candidate`` first, otherwise
    from a ``model_reobserved``/legacy ``candidate`` projection chain. Missing
    edge pairs, cuts, source changes and disconnected components are barriers.
    ``progress`` may be a callback accepting a compact JSON-serializable event.
    """
    config = config or ContinuityGraphConfig()
    probes = np.asarray(model_probe_xy, float)
    if probes.ndim != 2 or probes.shape[1] != 2 or len(probes) < 8 or not np.isfinite(probes).all():
        raise ValueError("Require at least eight finite model probes; these are not measurements")
    # These numerical domain probes are not fitted image features. Retaining a
    # deterministic bounded subset avoids multiplying dense display polylines
    # by every candidate anchor on every video frame.
    probes = _pick(probes, 2048)
    result = copy.deepcopy(frames)
    groups = {}
    for ordinal, row in enumerate(result):
        groups.setdefault(_group(row), []).append((ordinal, row))
    diagnostics = {'algorithm': ALGORITHM_VERSION, 'config': asdict(config), 'components': [],
        'model_geometry_modified': False, 'inferred_frames_become_anchors': False,
        'engineering_ranges_are_statistical_confidence_intervals': False,
        'source_and_shot_barriers_enforced': True, 'homography_coefficient_interpolation_used': False,
        'independent_geometry_validation': False}
    for group_number, (_, members) in enumerate(groups.items()):
        rows = [row for _, row in members]
        matrices, edges, adjacency, components, rejected = _graph(rows)
        for component_number, ids in enumerate(components):
            anchors = {i: a for i in ids if (a := _anchor(rows[i])) is not None}
            component = {'id': f'graph_{group_number:03d}_{component_number:03d}',
                'source_sha256': rows[ids[0]].get('source_sha256'), 'shot_id': rows[ids[0]].get('shot_id'),
                'frames': len(ids), 'start_s': min(rows[i]['source_time_s'] for i in ids),
                'end_s': max(rows[i]['source_time_s'] for i in ids),
                'real_anchor_count': len(anchors), 'rejected_relative_edges': rejected,
                'anchor_samples': [rows[i]['sample_index'] for i in anchors], 'camera_hypothesis_comparisons': []}
            diagnostics['components'].append(component)
            if progress is not None:
                progress({'phase': 'component_started', 'id': component['id'], 'frames': len(ids), 'anchors': len(anchors)})
            for i in ids:
                rows[i]['offline_camera_graph_id'] = component['id']
            if not anchors:
                component['reason'] = 'connected_image_component_without_model_anchor'
                for i in ids:
                    rows[i].update(status='unavailable', reason='local_component_without_model_anchor',
                        projection_chain=None, H_image_from_model=None, H_model_from_image=None, anchor_id=None)
                if progress is not None:
                    progress({'phase': 'component_finished', 'id': component['id'], 'reason': component['reason']})
                continue
            scale = rows[ids[0]].get('sample_size', [960, 540])[0]/960
            sigma = config.anchor_sigma_px_at_960*scale
            distance_maps = {i: _distances(adjacency, i) for i in anchors}
            similarity = _fit_component(rows, ids, matrices, edges, anchors, probes, config)
            component['camera_hypothesis_comparisons'].append(_summary(similarity))
            selected = similarity if similarity['accepted'] else None
            if config.compare_constrained_plane and len(anchors) > 1:
                plane = _fit_component(rows, ids, matrices, edges, anchors, probes, config, plane=True)
                component['camera_hypothesis_comparisons'].append(_summary(plane))
                # Extra freedom is retained only if it improves withheld image
                # pairs, or resolves a rejected similarity while passing every
                # anchor/held-out-edge guard. No in-sample score promotion.
                if plane['accepted'] and (selected is None or plane['edge_holdout_median_sample_px'] < .9*selected['edge_holdout_median_sample_px']):
                    selected = plane
            component['selected_camera_strategy'] = selected['method'] if selected else None
            component['reason'] = 'joint_constraints_consistent' if selected else 'unresolved_anchor_hypotheses_retained'
            component['lineage_retained_at_keyframe_refresh'] = True
            for i in ids:
                row = rows[i]
                candidates = []
                for a, anchor in anchors.items():
                    distances, steps = distance_maps[a]
                    if i not in distances:
                        continue
                    lineage = _lineage(anchor, row, distances[i], steps[i], sigma)
                    if lineage['engineering_range_sample_px'] <= config.max_engineering_range_px_at_960*scale:
                        candidates.append((a, anchor, lineage))
                candidates.sort(key=lambda item: item[2]['engineering_range_sample_px'])
                if not candidates:
                    row.update(status='unavailable', reason='accumulated_image_path_uncertainty_exceeded',
                        projection_chain=None, H_image_from_model=None, H_model_from_image=None, anchor_id=None)
                    continue
                a, anchor, closest = candidates[0]
                if selected is not None and i in selected['corrections']:
                    base_id = selected['base_id']
                    chain = _transport(anchors[base_id]['projection_chain'], matrices[i], matrices[base_id], selected['corrections'][i])
                    # All jointly used anchors remain in provenance even if a
                    # distant one exceeds this frame's propagation budget.
                    lineage = [_lineage(item, row, distance_maps[j][0][i], distance_maps[j][1][i], sigma)
                               for j, item in anchors.items()]
                    uncertainty = closest['engineering_range_sample_px'] + selected['anchor_worst_p95_sample_px'] + selected['edge_holdout_worst_increase_sample_px']
                    if uncertainty > config.max_engineering_range_px_at_960*scale:
                        row.update(status='unavailable', reason='joint_camera_uncertainty_exceeded', projection_chain=None,
                            H_image_from_model=None, H_model_from_image=None, anchor_id=None)
                        continue
                    _publish(row, chain, anchors[base_id], lineage, reason='offline_joint_actual_image_edges_and_model_anchors',
                             engineering_range=uncertainty, joint=True)
                    row['joint_camera_strategy'] = selected['method']
                    if row.get('model_anchor_hypotheses') or row.get('model_anchor_consistency', {}).get('conflict'):
                        row['prior_anchor_conflict_resolution'] = 'camera_graph_fit_passed_anchor_and_withheld_relative_edge_guards_not_independent_validation'
                    continue
                # Incompatible anchors are never averaged. Their complete
                # branches survive until actual connectivity/uncertainty rules
                # eliminate one; a new keyframe is not a reason to forget one.
                hypotheses = []
                hypothesis_projections = []
                for a, anchor, lineage in candidates:
                    chain = _transport(anchor['projection_chain'], matrices[i], matrices[a])
                    xy, valid = _chain_xy(chain, probes)
                    equivalent = None
                    for number, (previous_xy, previous_valid) in enumerate(hypothesis_projections):
                        common = valid & previous_valid & (_visible(xy, valid, row['sample_size']) |
                                                           _visible(previous_xy, previous_valid, row['sample_size']))
                        if np.count_nonzero(common) >= 8:
                            error = np.linalg.norm(xy[common]-previous_xy[common], axis=1)
                            if np.percentile(error, 95) <= config.conflict_p95_px_at_960*scale:
                                equivalent = number
                                break
                    if equivalent is not None:
                        hypotheses[equivalent]['supporting_anchor_observations'].append(lineage)
                        continue
                    hypotheses.append({'role': 'real_anchor_propagated_through_actual_image_edges',
                        'reference_id': anchor.get('reference_id'), 'anchor_observation': lineage, 'projection_chain': chain,
                        'supporting_anchor_observations': [lineage], 'representative_is_not_average': True})
                    hypothesis_projections.append((xy, valid))
                first_xy, first_valid = hypothesis_projections[0]
                disagreement = []
                for other_xy, other_valid in hypothesis_projections[1:]:
                    mask = first_valid & other_valid & (_visible(first_xy, first_valid, row['sample_size']) |
                                                        _visible(other_xy, other_valid, row['sample_size']))
                    error = np.linalg.norm(first_xy[mask]-other_xy[mask], axis=1)
                    disagreement.append(float(np.percentile(error, 95)) if len(error) >= 8 else None)
                conflict = any(value is None or value > config.conflict_p95_px_at_960*scale for value in disagreement)
                row['graph_anchor_hypothesis_disagreement'] = {'p95_sample_px': disagreement,
                    'engineering_check_only': True, 'independent_accuracy_measurement': False}
                if conflict:
                    row.update(status='ambiguous', reason='unresolved_model_anchor_conflict_over_observed_component',
                        model_anchor_hypotheses=hypotheses, projection_chain=None, H_image_from_model=None,
                        H_model_from_image=None, anchor_id=None,
                        anchor_lineage=[item[2] for item in candidates], future_anchor_used=any(item[2]['future_anchor'] for item in candidates))
                else:
                    _publish(row, hypotheses[0]['projection_chain'], candidates[0][1], [item[2] for item in candidates],
                        reason='connected_actual_image_observations_from_real_model_anchor',
                        engineering_range=candidates[0][2]['engineering_range_sample_px'])
            if progress is not None:
                progress({'phase': 'component_finished', 'id': component['id'], 'reason': component['reason'],
                          'selected_camera_strategy': component.get('selected_camera_strategy')})
    diagnostics['states'] = {state: sum(row.get('status') == state for row in result) for state in
        ('model_reobserved', 'image_tracked', 'motion_predicted', 'ambiguous', 'unavailable')}
    return {'frames': result, 'diagnostics': diagnostics}
