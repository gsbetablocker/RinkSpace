"""User-declared rink relations as exact geometry, independent of video fitting."""
from __future__ import annotations

import copy
import numpy as np

HARD_GEOMETRY_PROFILE = 'user_20260923_alignment_symmetry'
F_COLUMN = tuple(f'F{i}' for i in range(1, 5))
R_COLUMN = tuple(f'R{i}' for i in range(1, 5))
EIGHT_CIRCLES = F_COLUMN + R_COLUMN
A_PAIR = ('A1', 'A2')
LANES = tuple(f'lane{i}' for i in range(1, 5))
CORNERS = ('upper_left', 'upper_right', 'lower_left', 'lower_right')
LANE_FLAGS = ('lane_centerline_through_FR_rows', 'lane_left_open',
              'lane_equal_width', 'lane_right_collinear')
HARD_FLAGS = (*LANE_FLAGS, 'equal_corner_radii', 'A1_A2_mirror_equal_radius')
LINE_PAIRS = (('B1', 'B2'), ('goal_line_left', 'goal_line_right'))
REQUIRED_FEATURES = EIGHT_CIRCLES + ('C', 'C0', 'B1', 'B2', 'goal_line_left',
    'goal_line_right', 'goal_crease_left', 'goal_crease_right')
HARD_PARAMETER_KINDS = ('circle_shared_center_x', 'circle_row_origin_y',
    'circle_row_gap', 'circle_shared_radius', 'symmetric_line_distance',
    'A_pair_shared_center_x', 'A_pair_mirror_y_distance', 'A_pair_shared_radius')


def normalize_hard_geometry_constraints(scene_data, problem=None):
    """Inherit scene relations; an empty/disabled problem field cannot erase them."""
    values = [scene_data.get('hard_geometry_constraints'),
              (problem or {}).get('hard_geometry_constraints')]
    active = None
    for value in values:
        if value is None or value is False or value == {}:
            continue
        if not isinstance(value, dict):
            raise ValueError('hard_geometry_constraints must name a supported profile')
        profile = value.get('profile')
        if profile is None and active is not None:
            if not any(key in value for key in HARD_FLAGS):
                continue
            profile = HARD_GEOMETRY_PROFILE
        if profile != HARD_GEOMETRY_PROFILE:
            raise ValueError(f'unsupported hard geometry profile: {profile}')
        previous = active or {}
        for key in HARD_FLAGS:
            if key in value and not isinstance(value[key], bool):
                raise ValueError(f'{key} must be an explicit boolean declaration')
        active = {**previous, **copy.deepcopy(value)}
        for key in HARD_FLAGS:
            if previous.get(key) is True:
                active[key] = True  # a problem override cannot erase a scene relation
    if active is not None:
        active.update(profile=HARD_GEOMETRY_PROFILE, enforcement='exact_parameterization', enabled=True)
    return active


def _move_center(feature, axis, value):
    delta = float(value) - feature['center_xyz'][axis]
    feature['center_xyz'][axis] = float(value)
    for ring in feature.get('rings', []):
        ring['center_xyz'][axis] += delta


def structural_state(features, parameters, constraints=None):
    """Project only relations: column/row/radius and paired-distance means."""
    missing = sorted(set(REQUIRED_FEATURES) - set(features))
    if missing:
        raise ValueError(f'hard geometry profile missing required features: {missing}')
    lane_flags = {key: bool((constraints or {}).get(key, False)) for key in LANE_FLAGS}
    if any(lane_flags.values()):
        missing_lanes = sorted(set(LANES) - set(features))
        if missing_lanes:
            raise ValueError(f'declared lane relations missing required features: {missing_lanes}')
    bounds = np.asarray(parameters['bounds'], float)
    center = (bounds[:2] + bounds[2:]) / 2
    rows = np.array([np.mean([features[f'{col}{i}']['center_xyz'][1] for col in 'FR'])
                     for i in range(1, 5)])
    gaps = np.diff(rows)
    if np.any(gaps <= 0):
        raise ValueError('hard geometry profile requires row 1..4 order; mean projection cannot relabel rows')
    column_x = {col: float(np.mean([features[f'{col}{i}']['center_xyz'][0] for i in range(1, 5)]))
                for col in 'FR'}
    if not column_x['F'] < center[0] < column_x['R']:
        raise ValueError('hard geometry profile requires F and R columns on their respective sides of C')
    distances = {}
    for left, right in LINE_PAIRS:
        lx = float(np.mean(np.asarray(features[left]['segments_xyz'])[..., 0]))
        rx = float(np.mean(np.asarray(features[right]['segments_xyz'])[..., 0]))
        distances[left] = (abs(lx - center[0]) + abs(rx - center[0])) / 2.
    local_crease = np.asarray(features['goal_crease_left']['polygon_xyz'], float).copy()
    # The source may already have a moved line and a stale, detached crease.
    # Keep the left shape, anchoring its flat back edge to the paired goal line.
    local_crease[:, 0] -= float(local_crease[:, 0].min())
    state = {'column_x': column_x, 'row_origin_y': float(rows[0]), 'row_gaps': gaps.tolist(),
            'radius': float(np.mean([features[fid]['radius'] for fid in EIGHT_CIRCLES])),
            'line_distances': distances, 'left_crease_local_xyz': local_crease.tolist(),
            **lane_flags}
    if (constraints or {}).get('A1_A2_mirror_equal_radius'):
        missing_pair = sorted(set(A_PAIR) - set(features))
        if missing_pair:
            raise ValueError(f'declared A1/A2 mirror relation missing features: {missing_pair}')
        a1, a2 = (features[fid] for fid in A_PAIR)
        if any(f.get('kind') != 'circle' for f in (a1, a2)):
            raise ValueError('A1/A2 mirror relation requires two circle features')
        xy = np.asarray([a1['center_xyz'][:2], a2['center_xyz'][:2]], float)
        radii = np.asarray([a1['radius'], a2['radius']], float)
        if not np.isfinite(xy).all() or not np.isfinite(radii).all() or np.any(radii <= 0):
            raise ValueError('A1/A2 mirror relation requires finite centers and positive radii')
        if not xy[0, 1] < center[1] < xy[1, 1]:
            raise ValueError('A1 and A2 must lie on opposite sides of the rink long-axis centerline')
        state['A_pair'] = {'shared_center_x': float(np.mean(xy[:, 0])),
                           'mirror_y_distance': float((xy[1, 1] - xy[0, 1]) / 2.),
                           'shared_radius': float(np.mean(radii))}
    if (constraints or {}).get('equal_corner_radii'):
        source_radii = parameters.get('corner_radii', {})
        state['corner_radius'] = float(np.mean([
            source_radii.get(corner, parameters['corner_radius']) for corner in CORNERS]))
    if lane_flags['lane_equal_width']:
        state['lane_half_width'] = float(np.mean([
            np.ptp(np.asarray(features[fid]['polygon_xyz'], float)[:, 1]) / 2. for fid in LANES]))
    if lane_flags['lane_right_collinear']:
        state['lane_right_x'] = float(np.mean([
            np.max(np.asarray(features[fid]['polygon_xyz'], float)[:, 0]) for fid in LANES]))
    return state


def validate_hard_parameter(spec, parameters, constraints=None):
    """Reject alternative variables that could break a declared relation."""
    kind = spec['kind']; ids = spec.get('feature_ids', [])
    center_x = (parameters['bounds'][0] + parameters['bounds'][2]) / 2
    A_kinds = ('A_pair_shared_center_x', 'A_pair_mirror_y_distance', 'A_pair_shared_radius')
    if kind in A_kinds:
        if not (constraints or {}).get('A1_A2_mirror_equal_radius') or len(ids) != 2 or set(ids) != set(A_PAIR) or spec.get('feature_id') is not None:
            raise ValueError(f'{kind} requires declared A1/A2 mirror equality and feature_ids=[A1,A2]')
        if kind == 'A_pair_shared_radius' and spec.get('ring_role', 'outer') != 'outer':
            raise ValueError('A1/A2 shared radius applies to the outer circle only')
        if kind != 'A_pair_shared_center_x' and 'bounds' in spec and spec['bounds'][0] <= 0:
            raise ValueError('A1/A2 mirror distance and shared radius bounds must remain positive')
    elif (constraints or {}).get('A1_A2_mirror_equal_radius') and kind.startswith('circle_') and (
            spec.get('feature_id') in A_PAIR or set(ids) & set(A_PAIR)):
        raise ValueError('independent A1/A2 circle variables would break mirror/equal-radius relation; use A_pair shared parameters')
    if (((constraints or {}).get('lane_centerline_through_FR_rows') or
         (constraints or {}).get('lane_equal_width')) and
            kind in ('lane_bound_y_min', 'lane_bound_y_max') and spec.get('feature_id') in LANES):
        raise ValueError('independent lane Y bound would break the declared lane centerline or equal width; use shared_lane_half_width')
    if (constraints or {}).get('equal_corner_radii') and kind == 'boundary_corner_radius':
        raise ValueError('independent corner radius would break equal_corner_radii; use shared_corner_radius')
    if kind == 'shared_corner_radius':
        if not (constraints or {}).get('equal_corner_radii') or spec.get('corner') is not None:
            raise ValueError('shared_corner_radius requires the declared four-corner equality')
        if 'bounds' in spec and spec['bounds'][0] <= 0:
            raise ValueError('shared corner radius bounds must remain positive')
    if (constraints or {}).get('lane_equal_width') and kind == 'lane_half_width':
        raise ValueError('independent lane width would break lane_equal_width; use shared_lane_half_width')
    if (constraints or {}).get('lane_right_collinear') and kind == 'lane_bound_x_max' and spec.get('feature_id') in LANES:
        raise ValueError('independent lane right X would break lane_right_collinear; use shared_lane_right_x')
    if kind == 'shared_lane_right_x':
        ids = spec.get('feature_ids', [])
        if not (constraints or {}).get('lane_right_collinear') or len(ids) != len(LANES) or set(ids) != set(LANES):
            raise ValueError('shared_lane_right_x requires all four lanes and declared right-edge collinearity')
    if kind == 'lane_left_endpoint_x':
        if spec.get('feature_id') not in LANES or spec.get('edge') not in ('upper', 'lower'):
            raise ValueError('lane_left_endpoint_x requires one lane and upper or lower edge')
    if kind in ('lane_half_width', 'shared_lane_half_width'):
        lane_ids = [spec.get('feature_id')] if kind == 'lane_half_width' else spec.get('feature_ids', [])
        if not lane_ids or len(set(lane_ids)) != len(lane_ids) or not set(lane_ids) <= set(LANES):
            raise ValueError('lane half width must name unique lane1..lane4 features')
        if kind == 'shared_lane_half_width' and (constraints or {}).get('lane_equal_width') and set(lane_ids) != set(LANES):
            raise ValueError('declared equal lane width requires all four lanes')
        if 'bounds' in spec and spec['bounds'][0] <= 0:
            raise ValueError('lane half width bounds must remain strictly positive')
    if kind == 'circle_shared_center_x':
        if set(ids) not in (set(F_COLUMN), set(R_COLUMN)) or len(ids) != 4:
            raise ValueError('hard profile shared X must name one complete F or R column')
        if 'bounds' in spec:
            lo, hi = spec['bounds']
            if (set(ids) == set(F_COLUMN) and hi >= center_x) or (set(ids) == set(R_COLUMN) and lo <= center_x):
                raise ValueError('hard profile column X bounds must remain strictly on their side of C')
    elif kind in ('circle_shared_radius', 'circle_row_origin_y'):
        if set(ids) != set(EIGHT_CIRCLES) or len(ids) != 8:
            raise ValueError(f'hard profile {kind} must name all eight F/R circles')
        if kind == 'circle_shared_radius' and spec.get('ring_role', 'outer') != 'outer':
            raise ValueError('hard profile shared radius is the eight outer circles')
    elif kind == 'circle_row_gap':
        row = spec.get('after_row')
        expected = {f'{col}{i}' for col in 'FR' for i in range(row + 1, 5)} if row in (1, 2, 3) else set()
        if not expected or set(ids) != expected or len(ids) != len(expected):
            raise ValueError('hard profile gap requires after_row 1/2/3 and exactly its downstream row pairs')
        if 'bounds' in spec and spec['bounds'][0] <= 0:
            raise ValueError('hard profile row gap bounds must remain positive')
    elif kind == 'symmetric_line_distance':
        if tuple(ids) not in LINE_PAIRS:
            raise ValueError('hard profile line distance requires ordered [B1,B2] or [goal_line_left,goal_line_right]')
        if 'bounds' in spec and spec['bounds'][0] <= 0:
            raise ValueError('hard profile paired line distance bounds must remain positive')
    elif kind.startswith('circle_'):
        if spec.get('feature_id') in EIGHT_CIRCLES or spec.get('feature_id') == 'C0' and kind in ('circle_center_x', 'circle_center_y') or set(ids) & set(EIGHT_CIRCLES) or 'C0' in ids and kind in ('circle_shared_center_x', 'circle_shared_center_y'):
            raise ValueError(f'{kind} would break the hard profile; use its shared/origin/gap parameterization')
    elif kind == 'line_offset' and spec.get('feature_id') in ('C', 'B1', 'B2', 'goal_line_left', 'goal_line_right', 'goal_crease_left', 'goal_crease_right'):
        raise ValueError('independent line_offset would break hard alignment/symmetry')


def apply_structural_parameter(state, spec, value):
    kind = spec['kind']
    if kind in ('A_pair_shared_center_x', 'A_pair_mirror_y_distance', 'A_pair_shared_radius') and 'A_pair' in state:
        key = {'A_pair_shared_center_x': 'shared_center_x',
               'A_pair_mirror_y_distance': 'mirror_y_distance',
               'A_pair_shared_radius': 'shared_radius'}[kind]
        state['A_pair'][key] = float(value)
    elif kind == 'circle_shared_center_x':
        state['column_x'][spec['feature_ids'][0][0]] = float(value)
    elif kind == 'circle_shared_radius':
        state['radius'] = float(value)
    elif kind == 'circle_row_origin_y':
        state['row_origin_y'] = float(value)
    elif kind == 'circle_row_gap':
        if value <= 0:
            raise ValueError('hard row gap must be positive')
        state['row_gaps'][spec['after_row'] - 1] = float(value)
    elif kind == 'symmetric_line_distance':
        if value <= 0:
            raise ValueError('hard symmetric line distance must be positive')
        state['line_distances'][spec['feature_ids'][0]] = float(value)
    elif kind == 'shared_corner_radius' and 'corner_radius' in state:
        state['corner_radius'] = float(value)
    elif kind == 'shared_lane_half_width' and state.get('lane_equal_width'):
        state['lane_half_width'] = float(value)
    elif kind == 'shared_lane_right_x' and state.get('lane_right_collinear'):
        state['lane_right_x'] = float(value)
    else:
        return False
    return True


def vertical_boundary_interval(x, bounds, radii, *, grid_spacing=None, arc_steps=24):
    """Cross-section of the shared perimeter, with exporter-identical arc facets.

    Using the same facets avoids restoring slightly longer analytic segments
    when a saved mesh-backed scene is loaded again. Without meshing parameters
    this helper uses the exact quarter-circle cross-section.
    """
    x0, y0, x1, y1 = bounds
    if not x0 <= x <= x1:
        return ((y0 + y1) / 2, (y0 + y1) / 2)
    low, high = y0, y1
    for side in ('left', 'right'):
        for vertical in ('upper', 'lower'):
            radius = radii[f'{vertical}_{side}']
            cx = x0 + radius if side == 'left' else x1 - radius
            dx = x - cx
            if (side == 'left' and dx < 0) or (side == 'right' and dx > 0):
                q = np.sqrt(max(radius**2 - dx**2, 0.))
                if grid_spacing is not None:
                    count = max(arc_steps, int(np.ceil(np.pi * radius / (2 * grid_spacing))))
                    theta = np.linspace(0., np.pi / 2., count + 1)
                    q = np.interp(abs(dx), (radius*np.cos(theta))[::-1], (radius*np.sin(theta))[::-1])
                if vertical == 'upper':
                    high = min(high, y1 - radius + q)
                else:
                    low = max(low, y0 + radius - q)
    return float(low), float(high)


def apply_structural_state(features, parameters, corner_radii, state):
    bounds = parameters['bounds']; cx, cy = (np.asarray(bounds[:2]) + bounds[2:]) / 2
    if 'corner_radius' in state:
        radius = state['corner_radius']
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError('shared corner radius must be finite and positive')
        corner_radii.update({corner: radius for corner in CORNERS})
        parameters['corner_radius'] = radius
        parameters['corner_radii'] = {corner: radius for corner in CORNERS}
    rows = state['row_origin_y'] + np.r_[0., np.cumsum(state['row_gaps'])]
    if not state['column_x']['F'] < cx < state['column_x']['R']:
        raise ValueError('hard F/R columns cannot cross C')
    for col in 'FR':
        for i, y in enumerate(rows, 1):
            f = features[f'{col}{i}']; _move_center(f, 0, state['column_x'][col]); _move_center(f, 1, y)
            f['radius'] = state['radius']
            for ring in f.get('rings', []):
                if ring['role'] == 'outer':
                    ring['radius'] = state['radius']
    if 'A_pair' in state:
        pair = state['A_pair']; x = pair['shared_center_x']
        distance = pair['mirror_y_distance']; radius = pair['shared_radius']
        if not np.isfinite([x, distance, radius]).all() or distance <= 0 or radius <= 0:
            raise ValueError('A1/A2 shared X must be finite; mirror distance and radius must be positive')
        for fid, y in (('A1', cy - distance), ('A2', cy + distance)):
            feature = features[fid]
            _move_center(feature, 0, x); _move_center(feature, 1, y)
            feature['radius'] = radius
            for ring in feature.get('rings', []):
                if ring['role'] == 'outer':
                    ring['radius'] = radius
    _move_center(features['C0'], 0, cx); _move_center(features['C0'], 1, cy)
    meshing = {'grid_spacing': parameters.get('grid_spacing'), 'arc_steps': parameters.get('arc_steps_per_quarter', 24)}
    lo, hi = vertical_boundary_interval(cx, bounds, corner_radii, **meshing)
    features['C']['segments_xyz'] = [[[float(cx), hi, 0.], [float(cx), lo, 0.]]]
    for left, right in LINE_PAIRS:
        distance = state['line_distances'][left]
        lx, rx = cx - distance, cx + distance
        ll, lh = vertical_boundary_interval(lx, bounds, corner_radii, **meshing)
        rl, rh = vertical_boundary_interval(rx, bounds, corner_radii, **meshing)
        low, high = max(ll, rl), min(lh, rh)
        for fid, x in ((left, lx), (right, rx)):
            features[fid]['segments_xyz'] = [[[float(x), high, 0.], [float(x), low, 0.]]]
            features[fid]['hard_boundary_rule'] = 'paired lines use the common Y interval inside both independent rounded boundaries; exact X mirror'
            # Old per-side clipping records are historical, not current geometry.
            features[fid].pop('joint_unclipped_segments_xyz', None)
            features[fid].pop('joint_export_boundary_clipped', None)
    left = np.asarray(state['left_crease_local_xyz'], float).copy()
    left[:, 0] += cx - state['line_distances']['goal_line_left']
    right = left.copy(); right[:, 0] = 2 * cx - right[:, 0]
    features['goal_crease_left']['polygon_xyz'] = left.tolist()
    features['goal_crease_right']['polygon_xyz'] = right.tolist()
    if any(state.get(key, False) for key in LANE_FLAGS):
        for i, fid in enumerate(LANES, 1):
            feature = features[fid]
            polygon = np.asarray(feature['polygon_xyz'], float)
            if polygon.shape not in ((4, 3), (5, 3)) or not np.isfinite(polygon).all():
                raise ValueError(f'{fid}: declared lane relations require a finite rectangular polygon')
            if len(polygon) == 5 and not np.allclose(polygon[0], polygon[-1], atol=1e-9, rtol=0):
                raise ValueError(f'{fid}: fifth lane vertex must only repeat its first vertex')
            lo, hi = polygon.min(axis=0), polygon.max(axis=0)
            left_upper_x, left_lower_x = map(float, polygon[[0, 3], 0])
            if feature.get('left_endpoint_x_intervals_scene_units') is not None:
                goal_x = float(np.mean(np.asarray(features['goal_line_left']['segments_xyz'], float)[..., 0]))
                F_leftmost_x = float(features[f'F{i}']['center_xyz'][0] - features[f'F{i}']['radius'])
                # These bounds are relations to other current scene features,
                # not fixed coordinates inherited from the versioned source.
                feature['left_endpoint_x_intervals_scene_units'] = {
                    edge: [goal_x, F_leftmost_x] for edge in ('upper', 'lower')}
                bound_sources = feature.get('left_endpoint_x_interval_bound_sources')
                if bound_sources is not None:
                    if (not isinstance(bound_sources, dict) or
                            not isinstance(bound_sources.get('lower_x'), dict) or
                            not isinstance(bound_sources.get('upper_x'), dict) or
                            bound_sources['lower_x'].get('feature_id') != 'goal_line_left' or
                            bound_sources['upper_x'].get('feature_id') != f'F{i}'):
                        raise ValueError(f'{fid}: invalid left endpoint interval bound sources')
                    bound_sources['lower_x']['x_scene_units'] = goal_x
                    bound_sources['upper_x']['x_scene_units'] = F_leftmost_x
                roles = feature.setdefault('left_endpoint_coordinate_roles', {})
                for edge in ('upper', 'lower'):
                    role = roles.setdefault(edge, 'guaranteed_core_proxy' if
                                            feature.get('left_endpoint_status') == 'unresolved'
                                            else 'image_fit_candidate_unvalidated')
                    if role not in ('guaranteed_core_proxy', 'image_fit_candidate_unvalidated'):
                        raise ValueError(f'{fid} {edge}: unknown left endpoint coordinate role')
                    if role == 'guaranteed_core_proxy':
                        if edge == 'upper':
                            left_upper_x = F_leftmost_x
                        else:
                            left_lower_x = F_leftmost_x
            half_width = state['lane_half_width'] if state.get('lane_equal_width') else (hi[1] - lo[1]) / 2.
            right_x = state['lane_right_x'] if state.get('lane_right_collinear') else hi[0]
            if right_x <= max(left_upper_x, left_lower_x) or half_width <= 0:
                raise ValueError(f'{fid}: declared lane must have positive length and half width')
            # Width comes from the CURRENT features after all free width
            # parameters, so exact row alignment never discards a fitted width.
            center_y = float(rows[i - 1]) if state.get('lane_centerline_through_FR_rows') else (lo[1] + hi[1]) / 2.
            y0, y1 = center_y - half_width, center_y + half_width
            feature['polygon_xyz'] = [[left_upper_x, y1, 0.], [float(right_x), y1, 0.],
                                      [float(right_x), y0, 0.], [left_lower_x, y0, 0.]]
            if state.get('lane_equal_width'):
                feature['lane_half_width_scene_units'] = float(half_width)
                feature.pop('candidate_shared_lane_half_width', None)
            if state.get('lane_right_collinear'):
                feature['lane_shared_right_x_scene_units'] = float(right_x)
            feature['lane_edge_order'] = ['upper_long', 'right_short', 'lower_long',
                                         'left_open' if state.get('lane_left_open') else 'left_short']
            # Any old clipped display geometry belongs to the previous lane
            # placement. The exporter rebuilds it from these updated sides.
            for key in ('paint_segments_xyz', 'paint_segment_source_edge_indices',
                        'paint_segments_authoritative_for_boundary_clipped_display'):
                feature.pop(key, None)
            if state.get('lane_centerline_through_FR_rows'):
                # The centerline is a construction aid, not an observed left
                # endpoint. Start only where both painted long edges exist.
                feature['centerline_xyz'] = [[max(left_upper_x, left_lower_x), center_y, 0.],
                                             [float(right_x), center_y, 0.]]
                feature['lane_centerline_through_FR_row'] = i
                feature['centerline_source_features'] = [f'F{i}', f'R{i}']
            if state.get('lane_left_open'):
                feature['open_edges'] = [3]
                feature['closed'] = False
                feature['paint_topology'] = 'three_sides_open_at_xmin'
            feature['hard_geometry_profile'] = HARD_GEOMETRY_PROFILE
    for fid in REQUIRED_FEATURES:
        features[fid]['hard_geometry_profile'] = HARD_GEOMETRY_PROFILE
    if 'A_pair' in state:
        for fid in A_PAIR:
            features[fid]['hard_geometry_profile'] = HARD_GEOMETRY_PROFILE


def hard_geometry_checks(features, parameters, corner_radii, constraints=None):
    """Check rendered semantic values, including effective mirrored endpoints."""
    cx, cy = (np.asarray(parameters['bounds'][:2]) + parameters['bounds'][2:]) / 2
    same = lambda a, b: bool(np.allclose(a, b, atol=1e-9, rtol=0))
    checks = {}
    for col in 'FR':
        centers = np.asarray([features[f'{col}{i}']['center_xyz'] for i in range(1, 5)])
        checks[f'{col}_same_x'] = same(centers[:, 0], centers[0, 0])
        checks[f'{col}_row_order'] = bool(np.all(np.diff(centers[:, 1]) > 0))
    checks['row_pairs_same_y'] = all(same(features[f'F{i}']['center_xyz'][1], features[f'R{i}']['center_xyz'][1]) for i in range(1, 5))
    checks['eight_equal_outer_radii'] = same([features[fid]['radius'] for fid in EIGHT_CIRCLES], features['F1']['radius'])
    if (constraints or {}).get('A1_A2_mirror_equal_radius'):
        a1, a2 = (features.get(fid, {}) for fid in A_PAIR)
        p1 = np.asarray(a1.get('center_xyz', [np.nan] * 3), float)
        p2 = np.asarray(a2.get('center_xyz', [np.nan] * 3), float)
        checks['A1_A2_same_center_x'] = same(p1[0], p2[0])
        checks['A1_A2_centers_mirror_across_long_axis'] = same(p1[1] + p2[1], 2 * cy)
        checks['A1_A2_opposite_sides_of_long_axis'] = bool(p1[1] < cy < p2[1])
        checks['A1_A2_equal_outer_radii'] = same(a1.get('radius', np.nan), a2.get('radius', np.nan))
    if (constraints or {}).get('equal_corner_radii'):
        checks['four_equal_corner_radii'] = same([corner_radii[c] for c in CORNERS], corner_radii[CORNERS[0]])
        checks['corner_radius_parameter_consistent'] = (same(parameters['corner_radius'], corner_radii[CORNERS[0]]) and
            same([parameters.get('corner_radii', {}).get(c) for c in CORNERS], corner_radii[CORNERS[0]]))
    checks['C_fixed_center_x_parallel_short_side'] = same(np.asarray(features['C']['segments_xyz'])[..., 0], cx)
    checks['C0_fixed_rink_center'] = same(features['C0']['center_xyz'], [cx, cy, 0.])
    checks['columns_on_respective_sides'] = bool(features['F1']['center_xyz'][0] < cx < features['R1']['center_xyz'][0])
    for left, right in LINE_PAIRS:
        a = np.asarray(features[left]['segments_xyz'], float); b = np.asarray(features[right]['segments_xyz'], float)
        reflected = a.copy(); reflected[..., 0] = 2 * cx - reflected[..., 0]
        checks[f'{left}_{right}_effective_segments_mirrored'] = same(reflected, b)
        checks[f'{left}_{right}_parallel_short_side'] = same(a[..., 0], a[0, 0, 0]) and same(b[..., 0], b[0, 0, 0])
        checks[f'{left}_{right}_nonzero_length'] = bool(np.linalg.norm(a[0, 1] - a[0, 0]) > 1e-9)
    left = np.asarray(features['goal_crease_left']['polygon_xyz']); right = np.asarray(features['goal_crease_right']['polygon_xyz'])
    reflected = left.copy(); reflected[:, 0] = 2 * cx - reflected[:, 0]
    checks['goal_creases_mirrored'] = same(reflected, right)
    checks['goal_crease_left_attached_to_line'] = same(left[:, 0].min(), np.asarray(features['goal_line_left']['segments_xyz'])[0, 0, 0])
    checks['goal_crease_right_attached_to_line'] = same(right[:, 0].max(), np.asarray(features['goal_line_right']['segments_xyz'])[0, 0, 0])
    if any((constraints or {}).get(key, False) for key in LANE_FLAGS):
        for i, fid in enumerate(LANES, 1):
            feature = features.get(fid, {})
            polygon = np.asarray(feature.get('polygon_xyz', []), float)
            valid = polygon.shape == (4, 3) and np.isfinite(polygon).all()
            checks[f'{fid}_rectangular_at_z0'] = bool(valid and same(polygon[:, 2], 0.) and
                same(polygon[[1, 2], 0], polygon[1, 0]) and
                same(polygon[[0, 1], 1], polygon[0, 1]) and same(polygon[[2, 3], 1], polygon[2, 1]) and
                max(polygon[0, 0], polygon[3, 0]) < polygon[1, 0] and polygon[2, 1] < polygon[0, 1])
            intervals = feature.get('left_endpoint_x_intervals_scene_units')
            if intervals is not None:
                goal_x = float(np.mean(np.asarray(features['goal_line_left']['segments_xyz'], float)[..., 0]))
                F_leftmost_x = float(features[f'F{i}']['center_xyz'][0] - features[f'F{i}']['radius'])
                bound_sources = feature.get('left_endpoint_x_interval_bound_sources')
                if bound_sources is not None:
                    lower = bound_sources.get('lower_x', {}) if isinstance(bound_sources, dict) else {}
                    upper = bound_sources.get('upper_x', {}) if isinstance(bound_sources, dict) else {}
                    lower_value = lower.get('x_scene_units') if isinstance(lower, dict) else None
                    upper_value = upper.get('x_scene_units') if isinstance(upper, dict) else None
                    checks[f'{fid}_left_interval_bound_sources_follow_current_geometry'] = bool(
                        isinstance(lower, dict) and isinstance(upper, dict) and
                        lower.get('feature_id') == 'goal_line_left' and
                        upper.get('feature_id') == f'F{i}' and
                        isinstance(lower_value, (int, float, np.number)) and
                        isinstance(upper_value, (int, float, np.number)) and
                        np.isfinite([lower_value, upper_value]).all() and
                        same(lower_value, goal_x) and same(upper_value, F_leftmost_x))
                roles = feature.get('left_endpoint_coordinate_roles', {})
                for edge, vertex in (('upper', 0), ('lower', 3)):
                    interval = intervals.get(edge) if isinstance(intervals, dict) else None
                    role = roles.get(edge, 'guaranteed_core_proxy')
                    prefix = 'core_proxy' if role == 'guaranteed_core_proxy' else 'candidate'
                    checks[f'{fid}_{edge}_{prefix}_inside_interval'] = bool(
                        valid and isinstance(interval, (list, tuple)) and len(interval) == 2 and
                        np.isfinite(interval).all() and interval[0] < interval[1] and
                        interval[0] - 1e-9 <= polygon[vertex, 0] <= interval[1] + 1e-9)
                    checks[f'{fid}_{edge}_{prefix}_between_goal_and_F'] = bool(
                        valid and goal_x < F_leftmost_x and
                        goal_x - 1e-9 <= polygon[vertex, 0] <= F_leftmost_x + 1e-9)
                    checks[f'{fid}_{edge}_interval_follows_current_goal_and_F'] = bool(
                        isinstance(interval, (list, tuple)) and len(interval) == 2 and
                        np.isfinite(interval).all() and
                        same(interval, [goal_x, F_leftmost_x]))
                    if role == 'guaranteed_core_proxy':
                        checks[f'{fid}_{edge}_core_proxy_at_F_leftmost'] = bool(
                            valid and same(polygon[vertex, 0], F_leftmost_x))
            if (constraints or {}).get('lane_centerline_through_FR_rows'):
                mid_y = (polygon[0, 1] + polygon[2, 1]) / 2. if valid else np.nan
                checks[f'{fid}_midline_through_F{i}_R{i}'] = bool(valid and
                    same(mid_y, features[f'F{i}']['center_xyz'][1]) and same(mid_y, features[f'R{i}']['center_xyz'][1]))
            if (constraints or {}).get('lane_left_open'):
                checks[f'{fid}_left_open_exactly_three_edges'] = bool(valid and feature.get('open_edges') == [3] and feature.get('closed') is False)
        if (constraints or {}).get('lane_equal_width'):
            widths = [np.ptp(np.asarray(features[fid]['polygon_xyz'], float)[:, 1]) for fid in LANES]
            checks['four_lanes_equal_width'] = same(widths, widths[0])
        if (constraints or {}).get('lane_right_collinear'):
            right_xs = [np.max(np.asarray(features[fid]['polygon_xyz'], float)[:, 0]) for fid in LANES]
            checks['four_lane_right_edges_collinear'] = same(right_xs, right_xs[0])
    unresolved_lane_ends = any(features.get(fid, {}).get('left_endpoint_status') == 'unresolved'
                               for fid in LANES)
    interpretation = 'Exact user-declared relations; neither video refitting nor physical dimension verification.'
    if unresolved_lane_ends:
        interpretation += (' Lane polygon left vertices marked guaranteed_core_proxy are display proxies; '
                           'vertices marked image_fit_candidate_unvalidated are estimates. Neither is a '
                           'measured physical endpoint; interval checks do not verify real endpoint positions.')
    return {'profile': HARD_GEOMETRY_PROFILE, 'all_constraints_satisfied': bool(all(checks.values())), 'checks': checks,
            'interpretation': interpretation}


def project_scene_to_hard_constraints(scene_data, config=None):
    """Return a versionable semantic scene after deterministic relation projection."""
    data = copy.deepcopy(scene_data)
    active = normalize_hard_geometry_constraints(data, {'hard_geometry_constraints': config} if config is not None else None)
    if active is None:
        raise ValueError('relation projection requires an explicit hard geometry profile')
    features = {f['id']: f for f in data['ice_features']}; pars = data['parameters']
    corners = {c: pars.get('corner_radii', {}).get(c, pars['corner_radius']) for c in ('upper_left', 'upper_right', 'lower_left', 'lower_right')}
    state = structural_state(features, pars, active); apply_structural_state(features, pars, corners, state)
    for fid in (*REQUIRED_FEATURES, *(LANES if any(active.get(k, False) for k in LANE_FLAGS) else ()),
                *(A_PAIR if active.get('A1_A2_mirror_equal_radius') else ())):
        features[fid]['geometry_status'] = (
            'user_constrained_endpoint_intervals_no_point_fit'
            if fid in LANES and features[fid].get('left_endpoint_status') == 'unresolved'
            else 'user_hard_relation_projection_not_video_refit')
    data['hard_geometry_constraints'] = active
    data['hard_geometry_projection'] = {'method': 'column X means; paired row Y means; eight radius mean; optional A1/A2 shared X, mirror Y distance and radius means; paired line distance means; left crease template mirrored and attached',
        'video_fitting_run': False, 'checks': hard_geometry_checks(features, pars, corners, active)}
    return data
