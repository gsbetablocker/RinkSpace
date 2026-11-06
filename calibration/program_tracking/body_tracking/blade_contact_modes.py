"""Model blade axes and source-derived episode mode priors.

Angles describe the saved model, not a source measurement or contact truth.
The mode schedule never reads a rotation, geometric minimum or feature ID.
"""
from collections import defaultdict

import numpy as np


MODES = ('inside_edge', 'outside_edge', 'flat_or_unknown', 'toe_pick',
         'heel', 'airborne', 'unknown')
_UNKNOWN = ('flat_or_unknown', 'unknown')


def blade_axes_diagnostics(R, ice_normal_world=(0., 0., 1.)):
    """Return vector-defined model diagnostics with the batch shape of R.

    Local x is heel-to-toe, y is lateral and z is boot-up. The supplied ice
    normal is normalized, then broadcast to ``R.shape[:-2] + (3,)``. Positive
    fore-aft pitch means the forward axis rises; positive bank means local
    positive y rises. Plane-normal angles are distinct from bank and pitch.
    """
    R = np.asarray(R, dtype=float)
    if R.ndim < 2 or R.shape[-2:] != (3, 3) or not np.isfinite(R).all():
        raise ValueError('Expected finite rotations with shape (...,3,3)')
    n = np.asarray(ice_normal_world, dtype=float)
    if not n.ndim or n.shape[-1] != 3 or not np.isfinite(n).all():
        raise ValueError('Expected finite ice normals with shape (...,3)')
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    if np.any(norm <= 0):
        raise ValueError('Ice normals must be nonzero')
    n = np.broadcast_to(n/norm, R.shape[:-2]+(3,)).copy()
    F, L, U = (R[..., :, j].copy() for j in range(3))
    fn = np.clip(np.sum(F*n, axis=-1), -1., 1.)
    ln = np.clip(np.sum(L*n, axis=-1), -1., 1.)
    un = np.clip(np.sum(U*n, axis=-1), -1., 1.)
    return dict(forward_world=F, lateral_world=L, up_world=U,
                ice_normal_world=n,
                fore_aft_pitch_degrees=np.asarray(np.degrees(np.arcsin(fn))),
                edge_bank_degrees=np.asarray(np.degrees(np.arctan2(ln, un))),
                transverse_inclination_degrees=np.asarray(np.degrees(np.arcsin(ln))),
                total_lean_degrees=np.asarray(np.degrees(np.arccos(un))),
                blade_plane_normal_world=L.copy(),
                plane_normal_to_ice_normal_degrees=np.asarray(np.degrees(np.arccos(np.abs(ln)))),
                oriented_plane_normal_to_ice_normal_degrees=np.asarray(np.degrees(np.arccos(ln))))


def inner_edge_y_sign(reference, binding):
    """Conditional inside-side signs from neutral anatomy and final R_FB.

    Each neutral native ankle's medial vector points toward the other ankle.
    Its dot product with the final registered local-positive-y axis determines
    +1/-1; missing or numerically degenerate mappings return 0. This static
    model mapping does not establish the source foot identity or active edge.
    """
    result = np.zeros(2, dtype=np.int8)
    try:
        joints = np.asarray(reference.body.J, dtype=float)
        rotations = np.asarray(binding.parameters()[0], dtype=float)
        if joints.ndim != 2 or joints.shape[0] < 9 or joints.shape[1] != 3:
            return result
        if rotations.shape != (2, 3, 3):
            return result
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        return result
    eps = 64*np.finfo(float).eps
    for foot, (ankle, other) in enumerate(((7, 8), (8, 7))):
        medial = joints[other]-joints[ankle]
        lateral = rotations[foot, :, 1]
        if not np.isfinite(medial).all() or not np.isfinite(lateral).all():
            continue
        scale = np.linalg.norm(medial)*np.linalg.norm(lateral)
        if scale <= 0:
            continue
        alignment = float(medial@lateral/scale)
        if abs(alignment) > eps:
            result[foot] = 1 if alignment > 0 else -1
    return result


def _source_cost_row(row, source_id, native_pts, foot, interval):
    """Admit only explicitly identified source rows; no model-derived costs."""
    if not source_id or row.get('source_id') != source_id:
        return None, 'missing_or_mismatched_source_id'
    if row.get('role') != 'fit' or row.get('visibility') != 'visible':
        return None, 'not_visible_fit_source_evidence'
    if row.get('semantic') != 'source_edge_mode' or not row.get('id'):
        return None, 'missing_source_edge_semantic_or_id'
    if row.get('foot') != foot or row.get('pts') not in native_pts:
        return None, 'not_an_exact_sampled_foot_pts'
    if interval.get('state') not in ('supported', 'airborne'):
        return None, 'unknown_episode_state'
    costs = row.get('mode_costs')
    if not isinstance(costs, dict) or not costs:
        return None, 'missing_explicit_mode_costs'
    clean = {}
    for mode, value in costs.items():
        if mode not in MODES or isinstance(value, (bool, np.bool_)):
            return None, 'invalid_mode_cost'
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None, 'invalid_mode_cost'
        if not np.isfinite(value):
            return None, 'nonfinite_mode_cost'
        # An explicit airborne interval breaks supported episodes. It cannot
        # acquire a contact edge because an unrelated cost row mentions one.
        if interval.get('state') == 'airborne' and mode not in (*_UNKNOWN, 'airborne'):
            return None, 'contact_mode_in_airborne_episode'
        clean[mode] = value
    return clean, None


def _episode_path(events, switch_cost):
    """Small Viterbi DP on actual evidence PTS only; absent costs are inf."""
    states = [mode for mode in MODES
              if mode in _UNKNOWN or any(mode in costs for _, _, costs in events)]
    cost = np.full((len(events), len(states)), np.inf)
    for i, (_, _, explicit) in enumerate(events):
        for j, mode in enumerate(states):
            if mode in explicit:
                cost[i, j] = explicit[mode]
    if not np.isfinite(cost).any(axis=1).all():
        raise ValueError('Source cost rows have no shared admitted mode at one PTS')
    dp = cost[0].copy()
    back = np.zeros_like(cost, dtype=int)
    for i in range(1, len(events)):
        transition = dp[:, None]+switch_cost*(np.arange(len(states))[:, None] != np.arange(len(states))[None, :])
        back[i] = np.argmin(transition, axis=0)
        dp = cost[i]+transition[back[i], np.arange(len(states))]
    j = int(np.argmin(dp))
    path = [j]
    for i in range(len(events)-1, 0, -1):
        j = int(back[i, j])
        path.append(j)
    return [states[j] for j in path[::-1]]


def source_edge_schedule(pts, feet, policy):
    """Choose source-derived episode priors, never new per-frame observations.

    ``pts`` contains real integer samples and ``feet`` is scalar or (N,).
    Intervals have explicit foot/start_pts/end_pts/state/role, and must not
    overlap on the queried foot samples. Each interval is solved separately,
    including across airborne/unknown boundaries. Uncovered or unadmitted
    episodes stay ``flat_or_unknown``.

    Optional ``edge_evidence`` rows must have an exact queried foot/PTS,
    matching policy source_id, id, fit role, visible visibility, semantic
    ``source_edge_mode`` and explicit finite ``mode_costs``. Missing mode
    costs mean unavailable, not an inferred zero cost. If any row is admitted,
    ``edge_switch_cost`` must be explicitly finite and nonnegative. Costs at
    the same foot/PTS sum; a state must have an explicit cost in every row.

    The selected first evidence mode supplies the initial *episode prior*,
    including its prefix before that evidence. Later transitions happen only
    at actual evidence PTS and pay the switch cost. No source image points or
    per-frame source truth are created by holding that prior between rows.
    """
    pts = np.asarray(pts)
    if pts.ndim != 1 or not np.issubdtype(pts.dtype, np.number) or not np.isfinite(pts).all():
        raise ValueError('Expected finite source PTS (N,)')
    if np.any(pts != pts.astype(np.int64)):
        raise ValueError('Source PTS must be integers')
    pts = pts.astype(np.int64)
    feet = np.broadcast_to(np.asarray(feet), pts.shape)
    if not np.isin(feet, (0, 1)).all():
        raise ValueError('Expected physical feet 0/1')
    modes = np.full(len(pts), 'flat_or_unknown', dtype=f'<U{max(map(len, MODES))}')
    result = dict(modes=modes, switches=[], transition_cost=0., episodes=[],
                  admitted_evidence_ids=[], rejected_evidence=[],
                  physical_contact_truth='unknown',
                  mode_semantics='source-derived episode prior; not per-frame source observations')
    penalty = policy.get('edge_switch_cost')
    if penalty is not None:
        if isinstance(penalty, (bool, np.bool_)) or not np.isfinite(float(penalty)) or float(penalty) < 0:
            raise ValueError('edge_switch_cost must be finite and nonnegative')
        penalty = float(penalty)
    claimed = np.zeros(len(pts), dtype=bool)
    used_ids = set()
    for number, interval in enumerate(policy.get('intervals', [])):
        foot = interval.get('foot')
        if foot not in (0, 1):
            raise ValueError('Each mode interval needs a fixed physical foot')
        start, end = interval['start_pts'], interval['end_pts']
        if not np.isfinite([start, end]).all() or int(start) != start or int(end) != end or end < start:
            raise ValueError('Expected ordered integer source interval bounds')
        selected = (feet == foot) & (pts >= start) & (pts <= end)
        if np.any(claimed & selected):
            raise ValueError('Overlapping mode intervals duplicate a foot sample')
        claimed |= selected
        if not selected.any() or interval.get('role') != 'fit' or not interval.get('optimization_enabled', True):
            continue
        native_pts = set(pts[selected].tolist())
        evidence = defaultdict(list)
        for row in policy.get('edge_evidence', []):
            if row.get('foot') != foot or not start <= row.get('pts', -np.inf) <= end:
                continue
            costs, reason = _source_cost_row(row, policy.get('source_id'), native_pts, foot, interval)
            if reason:
                result['rejected_evidence'].append(dict(id=row.get('id'), pts=row.get('pts'), reason=reason))
                continue
            if row['id'] in used_ids:
                raise ValueError('Duplicate source mode evidence id')
            used_ids.add(row['id'])
            evidence[int(row['pts'])].append((row['id'], costs))
        if not evidence:
            continue
        if penalty is None:
            raise ValueError('Explicit edge_switch_cost required with admitted source evidence')
        events = []
        for point, rows in sorted(evidence.items()):
            common = set.intersection(*(set(costs) for _, costs in rows))
            combined = {mode:sum(costs[mode] for _, costs in rows) for mode in common}
            events.append((point, [name for name, _ in rows], combined))
        path = _episode_path(events, penalty)
        modes[selected] = path[0]
        episode_id = interval.get('id', f'interval_{number}')
        result['episodes'].append(dict(interval_id=episode_id, foot=int(foot),
                                       first_mode=path[0], first_evidence_pts=events[0][0],
                                       first_evidence_ids=events[0][1],
                                       evidence_pts=[event[0] for event in events],
                                       selected_evidence_modes=path))
        result['admitted_evidence_ids'].extend(name for _, names, _ in events for name in names)
        for i in range(1, len(events)):
            if path[i] == path[i-1]:
                continue
            point, names, costs = events[i]
            modes[selected & (pts >= point)] = path[i]
            result['switches'].append(dict(pts=point, foot=int(foot), interval_id=episode_id,
                                           from_mode=path[i-1], to_mode=path[i],
                                           source_evidence_ids=names,
                                           previous_source_evidence_ids=events[i-1][1],
                                           explicit_mode_costs=costs, transition_cost=penalty))
            result['transition_cost'] += penalty
    result['switches'].sort(key=lambda row: (row['pts'], row['foot']))
    return result
