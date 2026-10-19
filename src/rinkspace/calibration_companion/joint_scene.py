"""Joint semantic rink geometry and installation/frame camera adjustment.

This module consumes identified pixel observations; it does not identify objects.
Every frame sees one shared parametric scene. Unknown fields remain schematic.
Boundary dimensions may vary; model coordinates are not measured metres.
Checks never enter the objective or initializer. An unseen-clip check requires a
separate problem with no scene parameters and its own pose-fit/check split.
"""
from __future__ import annotations

import copy
from collections import Counter
from fractions import Fraction
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .scene_camera import Camera, Scene, nearest_polyline_vectors
from .keyframe_solver import _finite, _object_ids, _stats
from .structural_constraints import (HARD_PARAMETER_KINDS, EIGHT_CIRCLES,
    normalize_hard_geometry_constraints, structural_state, validate_hard_parameter,
    apply_structural_parameter, apply_structural_state, hard_geometry_checks)


CORNERS = ('upper_right', 'upper_left', 'lower_left', 'lower_right')
STRAIGHT_SIDES = ('upper_long_side', 'lower_long_side', 'left_end', 'right_end')
# Neighbours ordered by increasing world X (long sides) or Y (end sides).
SIDE_CORNERS = {'upper_long_side': ('upper_left', 'upper_right'),
                'lower_long_side': ('lower_left', 'lower_right'),
                'left_end': ('lower_left', 'upper_left'),
                'right_end': ('lower_right', 'upper_right')}
CORNER_START_DEGREES = {'upper_right': 0., 'upper_left': 90.,
                        'lower_left': 180., 'lower_right': 270.}
BOUNDARY_KINDS = ('boundary_length', 'boundary_width', 'boundary_corner_radius', 'shared_corner_radius')
GEOMETRY_EPSILON = 1e-6
INNER_RING_ROLES = ('middle', 'inner', 'center')
LANE_FEATURES = ('lane1', 'lane2', 'lane3', 'lane4')


def _lane_half_width_ids(spec):
    if spec['kind'] == 'lane_half_width':
        ids = [spec.get('feature_id')]
        if ids[0] not in LANE_FEATURES:
            raise ValueError('lane_half_width requires one identified lane1..lane4')
    else:
        ids = spec.get('feature_ids', [])
        if len(ids) != 4 or set(ids) != set(LANE_FEATURES):
            raise ValueError('shared_lane_half_width requires lane1..lane4 exactly once')
    if 'bounds' in spec and spec['bounds'][0] <= 0:
        raise ValueError('lane half-width bounds must remain positive')
    return ids


def _validate_shared_ring_spec(spec):
    ids = spec.get('feature_ids', [])
    if len(ids) != len(EIGHT_CIRCLES) or set(ids) != set(EIGHT_CIRCLES):
        raise ValueError('ring_shared_radius requires all eight F/R circles exactly once')
    if spec.get('ring_role') not in INNER_RING_ROLES:
        raise ValueError('ring_shared_radius requires middle, inner or center role; outer uses the hard shared radius')
    if 'bounds' in spec and spec['bounds'][0] <= 0:
        raise ValueError('shared ring radius bounds must remain positive')


def landmark_boundary_axes(point):
    """Boundary dimension axes affecting a landmark's projected position.

    A straight-side world-along coordinate does not scale with that side's
    length. Its neighbouring corner radii still affect its legal surface domain.
    """
    side = point['side']
    if side in CORNERS or point.get('allow_corner') is not None:
        return (0, 1)
    if side in ('upper_long_side', 'lower_long_side'):
        return (1,)
    if side in ('left_end', 'right_end'):
        return (0,)
    raise ValueError(f'unknown board landmark surface: {side}')


def landmark_corner_dependencies(point):
    """Corner radii affecting coordinates, excluding domain-only constraints."""
    if point['side'] in CORNERS:
        return (point['side'],)
    if point['side'] not in STRAIGHT_SIDES:
        raise ValueError(f'unknown board landmark surface: {point["side"]}')
    return (point['allow_corner'],) if point.get('allow_corner') is not None else ()


class SharedGeometry:
    """Small semantic representation; no mesh vertices are optimization variables."""

    def __init__(self, scene: Scene, specifications: list[dict], values: np.ndarray,
                 board_landmarks: dict[str, dict] | None = None,
                 hard_geometry_constraints: dict | None = None):
        self.source = scene
        self.features = copy.deepcopy(scene.features)
        self.parameters = copy.deepcopy(scene.data['parameters'])
        self.board_landmarks = board_landmarks or {}
        radius = self.parameters['corner_radius']
        self.corner_radii = {c: self.parameters.get('corner_radii', {}).get(c, radius)
                             for c in CORNERS}
        self.hard_geometry_constraints = normalize_hard_geometry_constraints(scene.data,
            {'hard_geometry_constraints': hard_geometry_constraints})
        hard_state = structural_state(self.features, self.parameters, self.hard_geometry_constraints) if self.hard_geometry_constraints else None
        for spec, value in zip(specifications, values):
            kind = spec['kind']
            value = float(value)
            if hard_state is not None:
                validate_hard_parameter(spec, self.parameters, self.hard_geometry_constraints)
                if apply_structural_parameter(hard_state, spec, value):
                    continue
            if kind in ('boundary_length', 'boundary_width'):
                axis = 0 if kind == 'boundary_length' else 1
                bounds = self.parameters['bounds']
                center = (bounds[axis] + bounds[axis + 2]) / 2.
                bounds[axis], bounds[axis + 2] = center - value / 2., center + value / 2.
            elif kind in ('circle_shared_center_x', 'circle_shared_center_y'):
                axis = 0 if kind.endswith('_x') else 1
                for fid in spec['feature_ids']:
                    feature = self.features[fid]
                    delta = value - feature['center_xyz'][axis]
                    feature['center_xyz'][axis] = value
                    for ring in feature.get('rings', []):
                        ring['center_xyz'][axis] += delta
            elif kind == 'circle_shared_radius':
                for fid in spec['feature_ids']:
                    feature = self.features[fid]
                    feature['radius'] = value
                    for ring in feature.get('rings', []):
                        if ring['role'] == 'outer':
                            ring['radius'] = value
            elif kind == 'ring_shared_radius':
                _validate_shared_ring_spec(spec)
                if not np.isfinite(value) or value <= 0:
                    raise ValueError('shared ring radius must remain finite positive')
                for fid in spec['feature_ids']:
                    feature = self.features[fid]
                    rings = [r for r in feature.get('rings', []) if r['role'] == spec['ring_role']]
                    if len(rings) != 1:
                        raise ValueError(f'{fid}: exactly one {spec["ring_role"]} ring required')
                    rings[0]['radius'] = value
                    rings[0]['center_xyz'] = list(feature['center_xyz'])
            elif kind.startswith('circle_'):
                feature = self.features[spec['feature_id']]
                if kind in ('circle_center_x', 'circle_center_y'):
                    axis = 0 if kind.endswith('_x') else 1
                    delta = value - feature['center_xyz'][axis]
                    feature['center_xyz'][axis] = value
                    for ring in feature.get('rings', []):
                        ring['center_xyz'][axis] += delta
                elif kind == 'circle_radius':
                    role = spec.get('ring_role', 'outer')
                    if role == 'outer':
                        feature['radius'] = value
                    for ring in feature.get('rings', []):
                        if ring['role'] == role:
                            ring['radius'] = value
            elif kind == 'line_offset':
                feature = self.features[spec['feature_id']]
                segments = np.asarray(feature['segments_xyz'], float)
                direction = segments[0, 1, :2] - segments[0, 0, :2]
                normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
                segments[:, :, :2] += value * normal
                feature['segments_xyz'] = segments.tolist()
            elif kind == 'board_height':
                self.parameters['board_height'] = value
            elif kind == 'boundary_corner_radius':
                self.corner_radii[spec['corner']] = value
            elif kind in ('shared_crease_depth', 'shared_crease_half_width'):
                if hard_state is None or value <= 0:
                    raise ValueError('shared crease size requires positive value and hard mirror profile')
                local = np.asarray(hard_state['left_crease_local_xyz'], float)
                axis = 0 if kind == 'shared_crease_depth' else 1
                old = np.max(local[:, 0]) if axis == 0 else np.max(np.abs(local[:, 1]))
                if not np.isfinite(old) or old <= 0:
                    raise ValueError('shared crease template must have positive finite extent')
                local[:, axis] *= value / old
                hard_state['left_crease_local_xyz'] = local.tolist()
            elif kind in ('lane_half_width', 'shared_lane_half_width'):
                if not np.isfinite(value) or value <= 0:
                    raise ValueError('lane half-width must remain finite positive')
                for fid in _lane_half_width_ids(spec):
                    feature = self.features[fid]
                    polygon = np.asarray(feature['polygon_xyz'], float)
                    low, high = polygon[:, 1].min(), polygon[:, 1].max()
                    center = (low + high) / 2.
                    if not high > low or not np.all(np.isclose(polygon[:, 1], low) | np.isclose(polygon[:, 1], high)):
                        raise ValueError('lane half-width requires an axis-aligned rectangular template')
                    polygon[:, 1] = center + np.where(np.isclose(polygon[:, 1], low), -value, value)
                    feature['polygon_xyz'] = polygon.tolist()
                    feature['lane_half_width_scene_units'] = value
                    if kind == 'shared_lane_half_width':
                        feature['candidate_shared_lane_half_width'] = {
                            'parameter_id': spec['id'], 'feature_ids': list(spec['feature_ids']),
                            'user_declared_equal_widths': False,
                            'status': 'optional_joint_fit_hypothesis_not_user_hard_constraint'}
            elif kind.startswith('lane_bound_'):
                if spec['feature_id'] not in ('lane1', 'lane2', 'lane3', 'lane4'):
                    raise ValueError('lane bounds may only alter rectangular lane features')
                feature = self.features[spec['feature_id']]
                if kind == 'lane_bound_x_min' and feature.get('left_endpoint_x_intervals_scene_units') is not None:
                    raise ValueError('uncertain lane left endpoints require separate upper/lower parameters')
                polygon = np.asarray(feature['polygon_xyz'], float)
                axis = 0 if '_x_' in kind else 1
                old = polygon[:, axis].min() if kind.endswith('_min') else polygon[:, axis].max()
                polygon[np.isclose(polygon[:, axis], old), axis] = value
                feature['polygon_xyz'] = polygon.tolist()
            elif kind == 'lane_left_endpoint_x':
                if spec.get('feature_id') not in LANE_FEATURES or spec.get('edge') not in ('upper', 'lower'):
                    raise ValueError('lane_left_endpoint_x requires one lane and upper or lower edge')
                feature = self.features[spec['feature_id']]
                polygon = np.asarray(feature['polygon_xyz'], float)
                vertex = 0 if spec['edge'] == 'upper' else 3
                polygon[vertex, 0] = value
                feature['polygon_xyz'] = polygon.tolist()
                feature.setdefault('left_endpoint_coordinate_roles', {})[spec['edge']] = (
                    'image_fit_candidate_unvalidated')
            else:
                raise ValueError(f'unsupported scene parameter kind: {kind}')
        if hard_state is not None:
            apply_structural_state(self.features, self.parameters, self.corner_radii, hard_state)
        # Assign after all shared row/column moves as well, so concentricity is
        # exact in the serialized coordinates, not just equal within roundoff.
        for spec in specifications:
            if spec['kind'] == 'ring_shared_radius':
                for fid in spec['feature_ids']:
                    feature = self.features[fid]
                    next(r for r in feature['rings'] if r['role'] == spec['ring_role'])['center_xyz'] = list(feature['center_xyz'])

    def hard_geometry_checks(self):
        return (hard_geometry_checks(self.features, self.parameters, self.corner_radii, self.hard_geometry_constraints)
                if self.hard_geometry_constraints else None)

    def boundary_validity_margins(self):
        """Positive margins describe a non-self-overlapping rounded rectangle.

        Adjacent corner radii constrain each straight segment. An individual
        corner need not be smaller than half the rink width when its neighbour
        is smaller; the previous half-width bound needlessly froze this freedom.
        """
        x0, y0, x1, y1 = self.parameters['bounds']
        length, width = x1 - x0, y1 - y0
        r = self.corner_radii
        return {'length': length, 'width': width,
                **{f'{corner}_radius': r[corner] for corner in CORNERS},
                'upper_straight_length': length - r['upper_left'] - r['upper_right'],
                'lower_straight_length': length - r['lower_left'] - r['lower_right'],
                'left_straight_length': width - r['upper_left'] - r['lower_left'],
                'right_straight_length': width - r['upper_right'] - r['lower_right']}

    def boundary_clearance(self, point_xy):
        """Exact signed distance to straight segments and quarter-circle arcs.

        A positive distance means inside. This is a semantic validity check,
        not a pixel observation or an extra fitting residual.
        """
        point = np.asarray(point_xy, float)
        x0, y0, x1, y1 = self.parameters['bounds']
        r = self.corner_radii
        inside = x0 <= point[0] <= x1 and y0 <= point[1] <= y1
        distances = []
        for side in ('upper_long_side', 'lower_long_side', 'left_end', 'right_end'):
            endpoints = self.board_side({'side': side}, 2)[:, :2]
            delta = endpoints[1] - endpoints[0]
            fraction = np.clip(np.dot(point - endpoints[0], delta) / max(np.dot(delta, delta), 1e-30), 0., 1.)
            distances.append(float(np.linalg.norm(point - endpoints[0] - fraction * delta)))
        for corner in CORNERS:
            radius = r[corner]
            center = np.array([x1 - radius if 'right' in corner else x0 + radius,
                               y1 - radius if 'upper' in corner else y0 + radius])
            delta = point - center
            norm = float(np.linalg.norm(delta))
            in_corner_quadrant = ((delta[0] >= 0 if 'right' in corner else delta[0] <= 0) and
                                  (delta[1] >= 0 if 'upper' in corner else delta[1] <= 0))
            if in_corner_quadrant:
                distances.append(abs(norm - radius))
                if norm > radius:
                    inside = False
            # Arc endpoints cover nearest points when the radial projection
            # falls outside this corner's quarter-circle.
            endpoints = self.board_side({'side': corner}, 2)[:, :2]
            distances.extend(np.linalg.norm(endpoints - point, axis=1).tolist())
        nearest = min(distances)
        return nearest if inside else -nearest

    def circle_validity_checks(self):
        """Check every semantic circle, including unobserved retained circles.

        Unobserved schematic circles do not constrain the objective. They still
        prevent a whole-scene export from silently containing invalid geometry.
        """
        rows = []
        for fid, feature in self.features.items():
            if 'center_xyz' not in feature or 'radius' not in feature:
                continue
            radius = float(feature['radius'])
            center = np.asarray(feature['center_xyz'][:2], float)
            violations = []
            if not np.isfinite(radius) or radius <= 0 or not np.isfinite(center).all():
                rows.append({'feature_id': fid, 'valid': False,
                             'boundary_clearance_scene_units': None,
                             'containment_margin_scene_units': None,
                             'inner_radius_margin_scene_units': None,
                             'violations': ['nonfinite_or_nonpositive_circle_geometry']})
                continue
            clearance = self.boundary_clearance(center)
            containment = clearance - radius
            # Preserve existing inner geometry. Check its actual extent in case
            # a source explicitly contains a nonconcentric component ring.
            inner_margins = []
            for ring in feature.get('rings', []):
                if ring.get('role') == 'outer':
                    continue
                inner_radius = float(ring['radius'])
                inner_center = np.asarray(ring.get('center_xyz', feature['center_xyz'])[:2], float)
                if not np.isfinite(inner_radius) or inner_radius < 0 or not np.isfinite(inner_center).all():
                    violations.append('nonfinite_or_negative_inner_ring_geometry')
                    continue
                inner_margins.append(radius - inner_radius - float(np.linalg.norm(inner_center - center)))
            inner_margin = min(inner_margins) if inner_margins else None
            if containment < -1e-8:
                violations.append('outer_circle_outside_rounded_boundary')
            if inner_margin is not None and inner_margin < -1e-8:
                violations.append('inner_ring_exceeds_outer_circle')
            rows.append({'feature_id': fid, 'valid': not violations,
                         'boundary_clearance_scene_units': clearance,
                         'containment_margin_scene_units': containment,
                         'inner_radius_margin_scene_units': inner_margin,
                         'violations': violations})
        return rows

    def board_landmark(self, landmark_id):
        point = self.board_landmarks[landmark_id]
        x0, y0, x1, y1 = self.parameters['bounds']
        side, along = point['side'], point['along']
        z = point['z_fraction'] * self.parameters['board_height']
        if side in CORNERS:
            if point.get('along_parameterization') != 'corner_fraction':
                raise ValueError('direct corner landmarks require along_parameterization=corner_fraction')
            radius = self.corner_radii[side]
            center = np.array([x1 - radius if 'right' in side else x0 + radius,
                               y1 - radius if 'upper' in side else y0 + radius])
            theta = np.deg2rad(CORNER_START_DEGREES[side] + 90. * along)
            return np.r_[center + radius * np.array([np.cos(theta), np.sin(theta)]), z]
        if side not in STRAIGHT_SIDES:
            raise ValueError(f'unknown board landmark surface: {side}')
        is_long = side in ('upper_long_side', 'lower_long_side')
        x, y = ((along, y1 if side == 'upper_long_side' else y0) if is_long
                else (x0 if side == 'left_end' else x1, along))
        corner = point.get('allow_corner')
        if corner is not None:
            if corner not in SIDE_CORNERS[side]:
                raise ValueError('allow_corner must be adjacent to the identified straight side')
            radius = self.corner_radii[corner]
            cx = x1 - radius if 'right' in corner else x0 + radius
            cy = y1 - radius if 'upper' in corner else y0 + radius
            delta = along - (cx if is_long else cy)
            beyond_tangent = delta < 0 if corner == SIDE_CORNERS[side][0] else delta > 0
            if beyond_tangent:
                radial_component = np.sqrt(max(radius * radius - delta * delta, 0.))
                if is_long:
                    y = cy + (radial_component if 'upper' in corner else -radial_component)
                else:
                    x = cx + (radial_component if 'right' in corner else -radial_component)
        return np.array([x, y, z])

    def board_landmark_margins(self, landmark_id):
        point = self.board_landmarks[landmark_id]
        x0, y0, x1, y1 = self.parameters['bounds']
        side = point['side']
        if side in CORNERS:
            return np.array([point['along'], 1. - point['along']])
        if side not in STRAIGHT_SIDES:
            raise ValueError(f'unknown board landmark surface: {side}')
        low_corner, high_corner = SIDE_CORNERS[side]
        if side in ('upper_long_side', 'lower_long_side'):
            extent_lo, extent_hi = x0, x1
        else:
            extent_lo, extent_hi = y0, y1
        lo, hi = extent_lo + self.corner_radii[low_corner], extent_hi - self.corner_radii[high_corner]
        # World-along continuation is singular at the remote arc endpoint.
        # Direct corner fractions can represent either endpoint without this
        # limitation, and are preferable for details wholly on a corner.
        if point.get('allow_corner') == low_corner:
            lo = extent_lo + GEOMETRY_EPSILON
        elif point.get('allow_corner') == high_corner:
            hi = extent_hi - GEOMETRY_EPSILON
        return np.array([point['along'] - lo, hi - point['along']])

    def board_landmark_segment(self, landmark_id):
        point = self.board_landmarks[landmark_id]
        side = point['side']
        if side in CORNERS:
            return side + '_corner'
        corner = point.get('allow_corner')
        if corner is not None:
            if corner not in SIDE_CORNERS[side]:
                raise ValueError('allow_corner must be adjacent to the identified straight side')
            axis = 0 if side in ('upper_long_side', 'lower_long_side') else 1
            bounds = self.parameters['bounds']
            if corner == SIDE_CORNERS[side][0]:
                on_corner = point['along'] < bounds[axis] + self.corner_radii[corner]
            else:
                on_corner = point['along'] > bounds[axis + 2] - self.corner_radii[corner]
            if on_corner:
                return corner + '_corner'
        return side

    def board_side(self, target: dict, samples: int) -> np.ndarray:
        x0, y0, x1, y1 = self.parameters['bounds']
        r = self.corner_radii
        side = target['side']
        if side == 'upper_long_side':
            xy = np.linspace([x0 + r['upper_left'], y1], [x1 - r['upper_right'], y1], samples)
        elif side == 'lower_long_side':
            xy = np.linspace([x0 + r['lower_left'], y0], [x1 - r['lower_right'], y0], samples)
        elif side == 'left_end':
            xy = np.linspace([x0, y0 + r['lower_left']], [x0, y1 - r['upper_left']], samples)
        elif side == 'right_end':
            xy = np.linspace([x1, y0 + r['lower_right']], [x1, y1 - r['upper_right']], samples)
        elif side in CORNERS:
            radius = r[side]
            cx = x1 - radius if 'right' in side else x0 + radius
            cy = y1 - radius if 'upper' in side else y0 + radius
            start = {'upper_right': 0, 'upper_left': 90,
                     'lower_left': 180, 'lower_right': 270}[side]
            theta = np.deg2rad(np.linspace(start, start + 90, samples))
            xy = [cx, cy] + radius * np.column_stack([np.cos(theta), np.sin(theta)])
        else:
            raise ValueError(f'unknown board side: {side}')
        edge = target.get('edge', 'contact')
        if edge not in ('contact', 'top'):
            raise ValueError('board_side edge must be contact or top')
        height = self.parameters['board_height'] if edge == 'top' else 0.
        points = np.column_stack([xy, np.full(len(xy), height)])
        if target.get('allow_adjacent_corners', False):
            # This is a candidate surface domain, not a claim that either
            # corner is visible. Each observed pixel may match any point on
            # this continuous portion of the physical boundary.
            if side != 'right_end' or target.get('continue_corner') is not None:
                raise ValueError('adjacent-corner candidate domain requires right_end without continue_corner')
            lower = self.board_side({'side': 'lower_right', 'edge': edge}, samples)
            upper = self.board_side({'side': 'upper_right', 'edge': edge}, samples)
            if not (np.allclose(lower[-1], points[0], atol=1e-7) and
                    np.allclose(points[-1], upper[0], atol=1e-7)):
                raise ValueError('right-end candidate domain is not continuous')
            return np.vstack([lower[:-1], points, upper[1:]])
        continuation = target.get('continue_corner')
        if continuation is not None:
            if side not in STRAIGHT_SIDES or continuation not in SIDE_CORNERS[side]:
                raise ValueError('board continuation must be one adjacent corner')
            arc = self.board_side({'side': continuation, 'edge': edge}, samples)
            # Orient the two actual boundary pieces at their shared endpoint.
            # Never bridge across ice or through a different, unobserved side.
            distances = np.array([[np.linalg.norm(a - b) for b in (arc[0], arc[-1])]
                                  for a in (points[0], points[-1])])
            a, b = np.unravel_index(np.argmin(distances), distances.shape)
            if distances[a, b] > 1e-7:
                raise ValueError('board continuation has no common endpoint')
            if a == 0:
                points = points[::-1]
            if b == 1:
                arc = arc[::-1]
            points = np.vstack([points, arc[1:]])
        return points

    def points(self, target: dict, observation_type: str, samples: int) -> np.ndarray:
        kind = target['kind']
        if kind == 'polygon_supporting_line':
            fid = target['feature_id']
            if observation_type != 'curve' or fid not in ('lane1', 'lane2', 'lane3', 'lane4'):
                raise ValueError('polygon_supporting_line requires an identified rectangular lane curve')
            polygon = np.asarray(self.features[fid]['polygon_xyz'], float)
            index = int(target['index'])
            if not 0 <= index < len(polygon):
                raise ValueError('polygon_supporting_line edge index out of range')
            if index in self.features[target['feature_id']].get('open_edges', []):
                raise ValueError('cannot observe an explicitly open polygon edge')
            a, b = polygon[index].copy(), polygon[(index+1) % len(polygon)].copy()
            delta = np.abs(b[:2]-a[:2])
            if (delta > 1e-9).sum() != 1 or not np.allclose([a[2], b[2]], 0.):
                raise ValueError('polygon_supporting_line requires a nondegenerate axis-aligned ice edge')
            direction = int(np.argmax(delta))
            a[direction], b[direction] = self.parameters['bounds'][direction], self.parameters['bounds'][direction+2]
            # Only the observed infinite-line position is measured. Template
            # endpoints do not truncate the fit; the exported lane stays finite.
            return np.linspace(a, b, max(8, samples))
        if kind == 'polygon_outline':
            feature = self.features[target['feature_id']]
            polygon = np.asarray(feature['polygon_xyz'], float)
            if observation_type != 'curve' or len(polygon) < 3:
                raise ValueError('polygon_outline requires a polygon curve')
            # Brown distortion bends projected straight edges. Projecting only
            # polygon vertices would incorrectly replace each edge by a chord.
            # Keep all original vertices and a fixed count per edge while fitting.
            open_edges = set(feature.get('open_edges', []))
            if feature.get('closed') is False and not open_edges:
                open_edges.add(len(polygon) - 1)
            indices = [i for i in range(len(polygon)) if i not in open_edges]
            if not indices or any(b != a + 1 for a, b in zip(indices[:-1], indices[1:])):
                raise ValueError('polygon_outline requires one continuous chain; use separate edge observations for disjoint pieces')
            count = max(2, int(np.ceil(samples / len(indices))))
            pieces = [np.linspace(polygon[i], polygon[(i+1) % len(polygon)], count, endpoint=False) for i in indices]
            return np.vstack([*pieces, polygon[[(indices[-1] + 1) % len(polygon)]]])
        if kind == 'polygon_edge' and int(target['index']) in self.features[target['feature_id']].get('open_edges', []):
            raise ValueError('cannot observe an explicitly open polygon edge')
        if kind == 'board_landmark':
            if observation_type != 'point':
                raise ValueError('board_landmark must be an identified point observation')
            return self.board_landmark(target['landmark_id']).reshape(1, 3)
        if kind == 'board_corner':
            target = {**target, 'kind': 'board_side', 'side': target['corner']}
            kind = 'board_side'
        if kind == 'board_side':
            if observation_type != 'curve':
                raise ValueError('board_side is a curve; do not invent a point correspondence')
            return self.board_side(target, samples)
        # Reuse the established target semantics without copying large meshes.
        proxy = copy.copy(self.source)
        proxy.features = self.features
        if kind in ('board_curve', 'board_point') and target.get('edge') == 'top':
            target = {**target, 'height': self.parameters['board_height']}
        return (proxy.curve(target, samples) if observation_type == 'curve'
                else proxy.point(target).reshape(1, 3))


class JointSceneProblem:
    """Same camera/observation input as KeyframeProblem, plus scene_parameters.

    scene_parameters entries require id, kind, bounds, and fit observation IDs in
    evidence_observation_ids. Optional prior={mean,sigma} is reported explicitly.
    An unsupported parameter is retained at its input value, rather than being
    silently optimized on a prior. Parameter values are absolute except for
    line_offset, which is a translation along the original line's left normal.
    boundary_length/width change centered bounds without scaling ice features.
    circle_shared_radius sets the outer radius of every member of feature_ids;
    inner radii remain unchanged. Its initial_value defaults to the mean of the
    input outer radii, whose individual values are retained in the change log.

    geometric_constraints.circle_order contains feature_ids, axis (0 or 1), and
    optional min_gap. Only ordering is imposed, not alignment or equal spacing.
    Constraint violations are penalized, reported, and rejected at scene export.

    hard_geometry_constraints.profile=user_20260923_alignment_symmetry instead
    makes the user-declared F/R alignment, shared radii, center line/circle and
    paired B/goal/crease geometry exact. Four row ordinates use one origin plus
    three positive gaps. Empty problem metadata cannot erase a scene profile.

    installations[].radial_distortion accepts initial_k1_image, scale_image,
    fit, bounds, and optional prior. k1_opencv=k1_image*(focal/scale_image)**2.
    This conditional image-radius lens approximation is shared per installation.

    board_landmarks entries identify latent points on any identified board side:
    id, advertisement_instance_id, side, initial_along, bounds_along,
    initial_z_fraction, bounds_z_fraction, fit, evidence_observation_ids.
    z=z_fraction*board_height; along is world X on either long side and world Y
    on either end. These are numerical initial guesses, never measured PPT XYZ.
    Observations use target={kind:board_landmark, landmark_id:...}, type=point.
    Every straight side may allow_corner naming either adjacent quarter-circle,
    continuing onto that arc while keeping along as world X or Y. The remote
    arc endpoint is excluded to avoid the world-along tangent singularity.
    Direct corner landmarks use side equal to a CORNERS name and explicitly set
    along_parameterization='corner_fraction', with along bounds within [0,1].
    The fraction runs counterclockwise through the quarter-circle, starting at
    0, 90, 180, 270 degrees for upper-right, upper-left, lower-left, lower-right.

    split_mode is whole_object (default) or frame_object. frame_object permits
    testing the same physical feature in another frame and explicitly cannot be
    represented as independent validation of that feature's geometry.
    """

    def __init__(self, scene: Scene, problem: dict[str, Any]):
        self.scene, self.problem = scene, copy.deepcopy(problem)
        self.hard_geometry_constraints = normalize_hard_geometry_constraints(scene.data, problem)
        if self.hard_geometry_constraints:
            self.problem['hard_geometry_constraints'] = copy.deepcopy(self.hard_geometry_constraints)
        problem = self.problem
        self.initial_hard_state = (structural_state(scene.features, scene.data['parameters'], self.hard_geometry_constraints)
                                   if self.hard_geometry_constraints else None)
        if problem.get('scene_sha256') != scene.sha256:
            raise ValueError('scene_sha256 mismatch')
        if problem.get('geometry_mode') != 'shared_scene_joint':
            raise ValueError('geometry_mode must be shared_scene_joint')
        self.installations = {i['id']: i for i in problem['installations']}
        self.frames = {f['id']: f for f in problem['frames']}
        if len(self.installations) != len(problem['installations']):
            raise ValueError('duplicate installation IDs')
        if not self.frames or len(self.frames) != len(problem['frames']):
            raise ValueError('missing or duplicate frames')
        self.samples = int(problem.get('curve_samples', 128))
        if self.samples < 8:
            raise ValueError('curve_samples must be at least 8')
        self.circle_orders = copy.deepcopy(problem.get('geometric_constraints', {}).get('circle_order', []))
        order_keys = set()
        for order in self.circle_orders:
            ids = order.get('feature_ids', [])
            axis, gap = order.get('axis'), order.get('min_gap', 0.)
            if len(ids) < 2 or len(set(ids)) != len(ids) or axis not in (0, 1):
                raise ValueError('circle_order requires distinct feature_ids and axis 0 or 1')
            if any(fid not in scene.features or 'center_xyz' not in scene.features[fid] for fid in ids):
                raise ValueError('circle_order requires identified circle centers')
            if not np.isfinite(gap) or gap < 0:
                raise ValueError('circle_order min_gap must be finite and nonnegative')
            key = (tuple(ids), axis)
            if key in order_keys:
                raise ValueError('duplicate circle_order constraint')
            order_keys.add(key)
            order['min_gap'] = float(gap)
        split_mode = problem.get('split_mode', 'whole_object')
        if split_mode not in ('whole_object', 'frame_object'):
            raise ValueError('unknown split_mode')
        self.landmark_specs = {item['id']: copy.deepcopy(item) for item in problem.get('board_landmarks', [])}
        if len(self.landmark_specs) != len(problem.get('board_landmarks', [])):
            raise ValueError('duplicate board landmark IDs')
        self.observations, seen, object_uses = [], set(), {}
        for original in problem['observations']:
            o = copy.deepcopy(original)
            if o['id'] in seen:
                raise ValueError('duplicate observation ID')
            seen.add(o['id'])
            if o['frame_id'] not in self.frames or o['use'] not in ('fit', 'check'):
                raise ValueError('invalid observation frame/use')
            if o['type'] not in ('curve', 'point'):
                raise ValueError('observation type must be curve or point')
            uv = np.asarray(o['pixels_uv'], float).reshape(-1, 2)
            if not len(uv) or not np.isfinite(uv).all():
                raise ValueError('empty/nonfinite observation pixels')
            if o['type'] == 'point' and len(uv) != 1:
                raise ValueError('point observation requires exactly one point')
            if not np.isfinite([o['sigma_px'], o.get('weight', 1.)]).all() or min(o['sigma_px'], o.get('weight', 1.)) <= 0:
                raise ValueError('sigma and weight must be finite positive numbers')
            identities = _object_ids(o)
            if o['target']['kind'] == 'board_landmark':
                lid = o['target']['landmark_id']
                if lid not in self.landmark_specs or o['type'] != 'point':
                    raise ValueError('unknown board landmark or non-point observation')
                identities.add(('board_landmark', lid))
                # Different visible details on the same physical advert are not
                # independent whole-advert fit/check evidence in one frame.
                identities.add(('advertisement_instance', self.landmark_specs[lid]['advertisement_instance_id']))
            for identity in identities:
                key = identity if split_mode == 'whole_object' else (o['frame_id'], *identity)
                if object_uses.setdefault(key, o['use']) != o['use']:
                    raise ValueError('whole-object split violation')
            self.observations.append((o, uv))
        # An unresolved lane's polygon left vertices mark only the rightmost
        # guaranteed painted core. They are not observations of its physical
        # endpoints. Finite edge/outline targets must therefore carry genuine
        # endpoint evidence and separate parameters; support-line targets do
        # not depend on the unknown edge lengths.
        endpoint_specs = {}
        for spec in problem.get('scene_parameters', []):
            if spec.get('kind') == 'lane_left_endpoint_x':
                endpoint_specs[(spec.get('feature_id'), spec.get('edge'))] = set(
                    spec.get('evidence_observation_ids', []))
            elif (spec.get('kind') == 'lane_bound_x_min' and
                  scene.features.get(spec.get('feature_id'), {}).get('left_endpoint_status') == 'unresolved'):
                raise ValueError('unresolved lane left ends require separate upper/lower parameters')
        for observation, _ in self.observations:
            target = observation['target']
            fid = target.get('feature_id')
            if scene.features.get(fid, {}).get('left_endpoint_status') != 'unresolved':
                continue
            kind = target['kind']
            index = int(target['index']) if kind in ('polygon_edge', 'polygon_vertex') else None
            if kind == 'polygon_outline':
                required_edges = ('upper', 'lower')
            elif kind == 'polygon_edge' and index in (0, 2):
                required_edges = ('upper',) if index == 0 else ('lower',)
            elif kind == 'polygon_vertex' and index in (0, 3):
                required_edges = ('upper',) if index == 0 else ('lower',)
            else:
                continue
            for edge in required_edges:
                evidence_ids = endpoint_specs.get((fid, edge), set())
                if not evidence_ids or (observation['use'] == 'fit' and observation['id'] not in evidence_ids):
                    raise ValueError(
                        f'{observation["id"]}: unresolved {fid} {edge} left endpoint cannot use '
                        'a finite polygon target without its own fitted endpoint parameter '
                        'and fit observation; use polygon_supporting_line for line-only evidence')
        fit_by_id = {o['id']: o for o, _ in self.observations if o['use'] == 'fit'}
        self.scene_specs, self.retained_parameters = [], []
        values, lower, upper = [], [], []
        self.parameter_names, self.parameter_priors = [], []

        def add(name, value, bounds, prior=None):
            lo, hi = map(float, bounds)
            if not np.isfinite(value) or not lo <= value <= hi or not lo < hi:
                raise ValueError(f'invalid bounds/initial value for {name}')
            index = len(values)
            values.append(float(value)); lower.append(lo); upper.append(hi)
            self.parameter_names.append(name)
            if prior is not None:
                if not np.isfinite([prior['mean'], prior['sigma']]).all() or prior['sigma'] <= 0:
                    raise ValueError('prior mean/sigma must be finite and sigma positive')
                self.parameter_priors.append((index, prior))
            return index

        names, parameter_keys = set(), set()
        for original in problem.get('scene_parameters', []):
            spec = copy.deepcopy(original)
            if self.hard_geometry_constraints:
                validate_hard_parameter(spec, scene.data['parameters'], self.hard_geometry_constraints)
            if spec['id'] in names:
                raise ValueError('duplicate scene parameter ID')
            names.add(spec['id'])
            value = self._initial_scene_value(spec)
            if spec['kind'] == 'circle_shared_radius':
                keys = {('circle_radius', fid, 'outer', None) for fid in spec['feature_ids']}
                spec['original_feature_values'] = {fid: float(scene.features[fid]['radius'])
                                                   for fid in spec['feature_ids']}
            elif spec['kind'] == 'A_pair_shared_radius':
                keys = {('circle_radius', fid, 'outer', None) for fid in spec['feature_ids']}
                spec['original_feature_values'] = {fid: float(scene.features[fid]['radius'])
                                                   for fid in spec['feature_ids']}
            elif spec['kind'] in ('A_pair_shared_center_x', 'A_pair_mirror_y_distance'):
                axis_kind = 'circle_center_x' if spec['kind'].endswith('_x') else 'circle_center_y'
                keys = {(axis_kind, fid, 'outer', None) for fid in spec['feature_ids']}
            elif spec['kind'] == 'ring_shared_radius':
                keys = {('circle_radius', fid, spec['ring_role'], None) for fid in spec['feature_ids']}
                spec['original_feature_values'] = {
                    fid: float(next(r['radius'] for r in scene.features[fid]['rings'] if r['role'] == spec['ring_role']))
                    for fid in spec['feature_ids']}
            elif spec['kind'] in ('circle_shared_center_x', 'circle_shared_center_y'):
                kind = 'circle_center_x' if spec['kind'].endswith('_x') else 'circle_center_y'
                keys = {(kind, fid, 'outer', None) for fid in spec['feature_ids']}
            elif spec['kind'] in ('circle_row_origin_y', 'circle_row_gap'):
                keys = {(spec['kind'], spec.get('after_row'))}
            elif spec['kind'] == 'symmetric_line_distance':
                keys = {(spec['kind'], tuple(spec['feature_ids']))}
            elif spec['kind'] in ('lane_half_width', 'shared_lane_half_width'):
                keys = {('lane_half_width', fid) for fid in _lane_half_width_ids(spec)}
            elif spec['kind'] == 'shared_lane_right_x':
                keys = {('lane_bound_x_max', fid) for fid in spec['feature_ids']}
            elif spec['kind'] == 'lane_left_endpoint_x':
                keys = {('lane_left_endpoint_x', spec['feature_id'], spec['edge'])}
            elif spec['kind'] == 'shared_corner_radius':
                keys = {('boundary_corner_radius', corner) for corner in CORNERS}
            elif spec['kind'] == 'boundary_corner_radius':
                keys = {('boundary_corner_radius', spec['corner'])}
            else:
                keys = {(spec['kind'], spec.get('feature_id'), spec.get('ring_role', 'outer'), spec.get('corner'))}
            if keys & parameter_keys:
                raise ValueError('duplicate semantic scene parameter')
            parameter_keys.update(keys)
            evidence = spec.get('evidence_observation_ids', [])
            if any(oid not in fit_by_id for oid in evidence):
                raise ValueError('scene parameter evidence must refer only to fit observations')
            if not evidence:
                self.retained_parameters.append({**spec, 'value': value,
                    'status': 'unobserved_retained_schematic_not_estimated'})
                continue
            if any(not self._affects(spec, fit_by_id[oid]['target']) for oid in evidence):
                raise ValueError('scene parameter evidence refers to an unrelated target')
            if spec['kind'] in ('circle_radius', 'circle_shared_radius', 'ring_shared_radius', 'circle_row_gap', 'symmetric_line_distance', 'shared_crease_depth', 'shared_crease_half_width', 'lane_half_width', 'shared_lane_half_width', 'A_pair_mirror_y_distance', 'A_pair_shared_radius', 'board_height', *BOUNDARY_KINDS) and spec['bounds'][0] <= 0:
                raise ValueError('radius, height and dimension bounds must remain positive')
            spec['index'] = add('scene:' + spec['id'], value, spec['bounds'], spec.get('prior'))
            spec['original_value'] = value
            self.scene_specs.append(spec)
        self.scene_count = len(self.scene_specs)
        self.variable_boundary = any(s['kind'] in BOUNDARY_KINDS for s in self.scene_specs)
        variable_dimensions = {0 if s['kind'] == 'boundary_length' else 1 for s in self.scene_specs
                               if s['kind'] in ('boundary_length', 'boundary_width')}
        self.landmark_slices = {}
        self.active_board_landmark_ids = []
        old_height = scene.data['parameters']['board_height']
        for lid, spec in self.landmark_specs.items():
            if not isinstance(spec.get('advertisement_instance_id'), str) or not spec['advertisement_instance_id']:
                raise ValueError('board landmark needs a physical advertisement_instance_id')
            side = spec['side']
            if side not in (*STRAIGHT_SIDES, *CORNERS):
                raise ValueError('board landmark requires an identified straight side or individual corner')
            if side in CORNERS:
                if spec.get('along_parameterization') != 'corner_fraction':
                    raise ValueError('direct corner landmarks require along_parameterization=corner_fraction')
                if not 0 <= spec['bounds_along'][0] < spec['bounds_along'][1] <= 1:
                    raise ValueError('direct corner fraction bounds must remain within [0,1]')
            else:
                axis = 0 if side in ('upper_long_side', 'lower_long_side') else 1
                representation = 'world_x' if axis == 0 else 'world_y'
                if spec.get('along_parameterization', representation) != representation:
                    raise ValueError('straight-side along parameterization must preserve world X or Y')
                spec['along_parameterization'] = representation
            if spec.get('allow_corner') is not None:
                if side not in STRAIGHT_SIDES or spec['allow_corner'] not in SIDE_CORNERS[side]:
                    raise ValueError('allow_corner must be adjacent to the identified straight side')
                bounds = scene.data['parameters']['bounds']
                if axis not in variable_dimensions:
                    if spec['allow_corner'] == SIDE_CORNERS[side][0] and spec['bounds_along'][0] <= bounds[axis]:
                        name = 'xmin' if axis == 0 else 'ymin'
                        raise ValueError(f'corner landmark lower along bound must remain strictly above {name}')
                    if spec['allow_corner'] == SIDE_CORNERS[side][1] and spec['bounds_along'][1] >= bounds[axis + 2]:
                        name = 'xmax' if axis == 0 else 'ymax'
                        raise ValueError(f'corner landmark upper along bound must remain strictly below {name}')
            if 'initial_z_fraction' not in spec:
                spec['initial_z_fraction'] = float(spec['initial_z']) / old_height
                spec['bounds_z_fraction'] = (np.asarray(spec['bounds_z'], float) / old_height).tolist()
                spec['input_height_representation'] = 'absolute initial guess converted to fraction of base schematic board height'
            else:
                spec['input_height_representation'] = 'fraction_of_joint_board_height'
            zlo, zhi = spec['bounds_z_fraction']
            if not 0 <= zlo < zhi <= 1 or not zlo <= spec['initial_z_fraction'] <= zhi:
                raise ValueError('board landmark height fractions must remain within [0,1]')
            alo, ahi = spec['bounds_along']
            if not np.isfinite([alo, ahi, spec['initial_along']]).all() or not alo < ahi or not alo <= spec['initial_along'] <= ahi:
                raise ValueError('invalid board landmark along bounds/initial guess')
            fit_observations = [o for o in fit_by_id.values() if o['target'].get('landmark_id') == lid]
            evidence = spec.get('evidence_observation_ids', [])
            if any(oid not in fit_by_id or fit_by_id[oid]['target'].get('landmark_id') != lid for oid in evidence):
                raise ValueError('board landmark evidence must refer to its own fit observations only')
            if fit_observations:
                self.active_board_landmark_ids.append(lid)
            if spec.get('fit', True) and evidence:
                start = len(values)
                add(f'board_landmark:{lid}:along', spec['initial_along'], spec['bounds_along'])
                add(f'board_landmark:{lid}:z_fraction', spec['initial_z_fraction'], spec['bounds_z_fraction'])
                self.landmark_slices[lid] = slice(start, len(values))
                spec['parameter_status'] = 'jointly_estimated_latent_board_point_NOT_measured'
            else:
                if spec.get('fit', True) and fit_observations:
                    raise ValueError('a variable board landmark with fit observations requires explicit evidence IDs')
                spec['parameter_status'] = ('fixed_conditional_board_point' if not spec.get('fit', True)
                                            else 'unobserved_initial_guess_retained_NOT_estimated')
        if self.variable_boundary:
            if any(o['target']['kind'] in ('board_curve', 'board_point', 'advert_anchor') for o, _ in self.observations):
                raise ValueError('variable boundary requires semantic board_side targets, not stale arc lengths')
        self.center_slices, self.radial_indices, self.frame_slices = {}, {}, {}
        for iid, install in self.installations.items():
            center = _finite(install['initial_center_xyz'], 3, 'initial_center_xyz')
            if install.get('fit_center', True):
                lo, hi = install['center_bounds_xyz']
                start = len(values)
                for axis in range(3):
                    prior = install.get('center_prior')
                    prior = None if prior is None else {'mean': prior['mean'][axis], 'sigma': prior['sigma'][axis]}
                    add(f'installation:{iid}:center_{axis}', center[axis], [lo[axis], hi[axis]], prior)
                self.center_slices[iid] = slice(start, len(values))
            radial = install.get('radial_distortion')
            if radial is not None:
                if not np.isfinite(radial['scale_image']) or radial['scale_image'] <= 0:
                    raise ValueError('positive radial scale required')
                if not np.isfinite(radial['initial_k1_image']):
                    raise ValueError('finite radial initial coefficient required')
                if radial.get('fit', False):
                    self.radial_indices[iid] = add(f'installation:{iid}:k1_image', radial['initial_k1_image'], radial['bounds'], radial.get('prior'))
        for fid, frame in self.frames.items():
            iid = frame['installation_id']
            if iid not in self.installations:
                raise ValueError('unknown installation')
            wh = _finite(frame['image_size_wh'], 2, 'image_size_wh')
            if np.any(wh <= 0):
                raise ValueError('positive image size required')
            _finite(frame['principal_point_px'], 2, 'principal_point_px')
            rot = _finite(frame['initial_rvec_world_to_camera'], 3, 'initial_rvec')
            if frame.get('fy_over_fx', 1.) <= 0:
                raise ValueError('positive camera aspect required')
            if self.installations[iid].get('radial_distortion') and frame.get('fy_over_fx', 1.) != 1.:
                raise ValueError('image-radius radial model requires square pixels')
            if not problem.get('synthetic', False):
                for key in ('source_id', 'media_track_id', 'source_pts', 'time_base', 'image_path', 'image_sha256', 'pixel_frame_id'):
                    if frame.get(key) is None:
                        raise ValueError(f'{fid}: missing frame provenance {key}')
                Fraction(frame['time_base'])
            if frame.get('fit_pose', True):
                if not any(o['frame_id'] == fid and o['use'] == 'fit' for o, _ in self.observations):
                    raise ValueError('each variable frame needs fit observations')
                start = len(values)
                for axis in range(3):
                    add(f'frame:{fid}:rvec_{axis}', rot[axis], [-np.inf, np.inf])
                focal = float(frame['initial_focal_px'])
                bounds = frame['focal_bounds_px']
                if min(focal, *bounds) <= 0:
                    raise ValueError('positive focal length required')
                add(f'frame:{fid}:log_focal', np.log(focal), np.log(bounds))
                self.frame_slices[fid] = slice(start, len(values))
        self.x0 = np.asarray(values, float)
        self.bounds = (np.asarray(lower, float), np.asarray(upper, float))
        self.track_counts = Counter(o.get('correlation_group_id', o['id']) for o, _ in self.observations if o['use'] == 'fit')
        self._sparsity = self._make_sparsity()
        if len(self.residual(self.x0)) < len(self.x0):
            raise ValueError('fewer residuals than unknowns')
        geometry = self.geometry(self.x0)
        for o, _ in self.observations:
            if o['use'] == 'fit':
                xyz = geometry.points(o['target'], o['type'], self.samples)
                if np.any(self.camera(self.x0, o['frame_id']).project(xyz)[1] <= 0):
                    raise ValueError('initial fit geometry behind camera')

    def _initial_scene_value(self, spec):
        kind = spec['kind']
        if kind in ('A_pair_shared_center_x', 'A_pair_mirror_y_distance', 'A_pair_shared_radius'):
            if self.initial_hard_state is None or 'A_pair' not in self.initial_hard_state:
                raise ValueError(f'{kind} requires declared A1/A2 mirror equality')
            key = {'A_pair_shared_center_x': 'shared_center_x',
                   'A_pair_mirror_y_distance': 'mirror_y_distance',
                   'A_pair_shared_radius': 'shared_radius'}[kind]
            return float(spec.get('initial_value', self.initial_hard_state['A_pair'][key]))
        if kind == 'shared_corner_radius':
            if self.initial_hard_state is None or 'corner_radius' not in self.initial_hard_state:
                raise ValueError('shared_corner_radius requires declared four-corner equality')
            return float(spec.get('initial_value', self.initial_hard_state['corner_radius']))
        if kind == 'shared_lane_right_x':
            if self.initial_hard_state is None or 'lane_right_x' not in self.initial_hard_state:
                raise ValueError('shared_lane_right_x requires declared lane right-edge collinearity')
            return float(spec.get('initial_value', self.initial_hard_state['lane_right_x']))
        if kind in ('lane_half_width', 'shared_lane_half_width'):
            widths = []
            for fid in _lane_half_width_ids(spec):
                polygon = np.asarray(self.scene.features[fid]['polygon_xyz'], float)
                widths.append(float(np.ptp(polygon[:, 1]) / 2.))
            return float(spec.get('initial_value', np.mean(widths)))
        if kind == 'ring_shared_radius':
            _validate_shared_ring_spec(spec)
            radii = []
            for fid in spec['feature_ids']:
                rings = [r for r in self.scene.features[fid].get('rings', []) if r['role'] == spec['ring_role']]
                if len(rings) != 1:
                    raise ValueError(f'{fid}: exactly one {spec["ring_role"]} ring required')
                radii.append(float(rings[0]['radius']))
            return float(np.mean(radii))
        if kind in ('shared_crease_depth', 'shared_crease_half_width'):
            if self.initial_hard_state is None:
                raise ValueError('shared crease size requires hard mirror profile')
            local = np.asarray(self.initial_hard_state['left_crease_local_xyz'], float)
            return float(np.max(local[:, 0]) if kind == 'shared_crease_depth' else np.max(np.abs(local[:, 1])))
        if kind.startswith('lane_bound_'):
            if spec['feature_id'] not in ('lane1', 'lane2', 'lane3', 'lane4'):
                raise ValueError('lane bounds may only alter rectangular lane features')
            if (kind == 'lane_bound_x_min' and
                    self.scene.features[spec['feature_id']].get('left_endpoint_status') == 'unresolved'):
                raise ValueError('unresolved lane left ends require separate upper/lower parameters')
            polygon = np.asarray(self.scene.features[spec['feature_id']]['polygon_xyz'], float)
            axis = 0 if '_x_' in kind else 1
            return float(polygon[:, axis].min() if kind.endswith('_min') else polygon[:, axis].max())
        if kind == 'lane_left_endpoint_x':
            feature = self.scene.features[spec['feature_id']]
            vertex = 0 if spec['edge'] == 'upper' else 3
            interval = feature.get('left_endpoint_x_intervals_scene_units', {}).get(spec['edge'])
            if interval is not None and (len(interval) != 2 or
                    spec['bounds'][0] < interval[0] or spec['bounds'][1] > interval[1]):
                raise ValueError('lane left endpoint bounds must remain inside the declared interval')
            return float(spec.get('initial_value', feature['polygon_xyz'][vertex][0]))
        if self.initial_hard_state is not None and kind in HARD_PARAMETER_KINDS:
            state = self.initial_hard_state
            if kind == 'circle_shared_center_x':
                value = state['column_x'][spec['feature_ids'][0][0]]
            elif kind == 'circle_shared_radius':
                value = state['radius']
            elif kind == 'circle_row_origin_y':
                value = state['row_origin_y']
            elif kind == 'circle_row_gap':
                value = state['row_gaps'][spec['after_row'] - 1]
            else:
                value = state['line_distances'][spec['feature_ids'][0]]
            return float(spec.get('initial_value', value))
        if kind in ('circle_row_origin_y', 'circle_row_gap', 'symmetric_line_distance'):
            raise ValueError(f'{kind} requires the hard alignment/symmetry profile')
        if kind in ('circle_shared_center_x', 'circle_shared_center_y'):
            ids = spec.get('feature_ids', [])
            if len(ids) < 2 or len(set(ids)) != len(ids) or any(fid not in self.scene.features or 'center_xyz' not in self.scene.features[fid] for fid in ids):
                raise ValueError('shared circle center requires distinct identified feature_ids')
            axis = 0 if kind.endswith('_x') else 1
            return float(spec.get('initial_value', np.mean([self.scene.features[fid]['center_xyz'][axis] for fid in ids])))
        if kind in ('boundary_length', 'boundary_width'):
            bounds = self.scene.data['parameters']['bounds']
            axis = 0 if kind == 'boundary_length' else 1
            return float(bounds[axis + 2] - bounds[axis])
        if kind == 'circle_shared_radius':
            ids = spec.get('feature_ids', [])
            if len(ids) < 2 or len(set(ids)) != len(ids):
                raise ValueError('shared circle radius requires at least two distinct feature_ids')
            if spec.get('ring_role', 'outer') != 'outer':
                raise ValueError('shared circle radius currently supports the outer ring only')
            if any(fid not in self.scene.features or 'radius' not in self.scene.features[fid] for fid in ids):
                raise ValueError('shared circle radius requires identified circular features')
            return float(spec.get('initial_value', np.mean([self.scene.features[fid]['radius'] for fid in ids])))
        if kind.startswith('circle_'):
            feature = self.scene.features[spec['feature_id']]
            if kind in ('circle_center_x', 'circle_center_y'):
                return feature['center_xyz'][0 if kind.endswith('_x') else 1]
            if kind == 'circle_radius':
                role = spec.get('ring_role', 'outer')
                if role == 'outer':
                    return feature['radius']
                return next(r['radius'] for r in feature['rings'] if r['role'] == role)
        elif kind == 'line_offset':
            segments = np.asarray(self.scene.features[spec['feature_id']]['segments_xyz'], float)
            if np.linalg.norm(segments[0, 1, :2] - segments[0, 0, :2]) <= 1e-12:
                raise ValueError('degenerate line')
            return 0.
        elif kind == 'board_height':
            return self.scene.data['parameters']['board_height']
        elif kind == 'boundary_corner_radius':
            if spec.get('corner') not in CORNERS:
                raise ValueError('corner radius requires an individual corner identity')
            pars = self.scene.data['parameters']
            return pars.get('corner_radii', {}).get(spec['corner'], pars['corner_radius'])
        raise ValueError(f'unsupported scene parameter kind: {kind}')

    def _affects(self, spec, target):
        if spec['kind'] in ('A_pair_shared_center_x', 'A_pair_mirror_y_distance'):
            return target.get('feature_id') in spec['feature_ids'] and target['kind'] in ('circle_outline', 'circle_center')
        if spec['kind'] == 'A_pair_shared_radius':
            return (target.get('feature_id') in spec['feature_ids'] and target['kind'] == 'circle_outline'
                    and target.get('ring_role', 'outer') == 'outer')
        if spec['kind'] == 'shared_lane_right_x':
            if target.get('feature_id') not in spec['feature_ids']:
                return False
            if target['kind'] == 'polygon_supporting_line':
                polygon = np.asarray(self.scene.features[target['feature_id']]['polygon_xyz'], float)
                index = int(target['index'])
                return np.isclose(polygon[index, 0], polygon[:, 0].max()) and np.isclose(
                    polygon[(index + 1) % len(polygon), 0], polygon[:, 0].max())
            return True
        if spec['kind'] in ('lane_half_width', 'shared_lane_half_width'):
            if target.get('feature_id') not in _lane_half_width_ids(spec):
                return False
            if target['kind'] == 'polygon_supporting_line':
                polygon = np.asarray(self.scene.features[target['feature_id']]['polygon_xyz'], float)
                index = int(target['index'])
                # Infinite vertical support does not depend on its Y endpoints.
                return abs(polygon[(index + 1) % len(polygon), 0] - polygon[index, 0]) > 1e-9
            return True
        if spec['kind'] == 'ring_shared_radius':
            return (target.get('feature_id') in spec['feature_ids'] and target['kind'] == 'circle_outline'
                    and target.get('ring_role', 'outer') == spec['ring_role'])
        if spec['kind'] in ('shared_crease_depth', 'shared_crease_half_width'):
            return target.get('feature_id') in ('goal_crease_left', 'goal_crease_right')
        if spec['kind'] in ('circle_shared_center_x', 'circle_shared_center_y', 'circle_row_origin_y', 'circle_row_gap'):
            if (self.hard_geometry_constraints and self.hard_geometry_constraints.get('lane_centerline_through_FR_rows')
                    and spec['kind'] in ('circle_row_origin_y', 'circle_row_gap')
                    and target.get('feature_id') in LANE_FEATURES):
                row = int(target['feature_id'][-1])
                downstream = spec['kind'] == 'circle_row_origin_y' or row > spec['after_row']
                if target['kind'] == 'polygon_supporting_line':
                    polygon = np.asarray(self.scene.features[target['feature_id']]['polygon_xyz'], float)
                    index = int(target['index'])
                    return downstream and abs(polygon[(index + 1) % len(polygon), 0] - polygon[index, 0]) > 1e-9
                return downstream
            return target.get('feature_id') in spec['feature_ids']
        if spec['kind'] == 'symmetric_line_distance':
            ids = list(spec['feature_ids'])
            if ids == ['goal_line_left', 'goal_line_right']:
                ids += ['goal_crease_left', 'goal_crease_right']
            return target.get('feature_id') in ids
        if self.hard_geometry_constraints and spec['kind'] in BOUNDARY_KINDS and target.get('feature_id') in ('C', 'B1', 'B2', 'goal_line_left', 'goal_line_right'):
            # Effective paired line endpoints are the common cross-section of
            # two independently rounded boundaries, so either corner can bind.
            return True
        if spec['kind'] in ('boundary_length', 'boundary_width'):
            if target['kind'] == 'polygon_supporting_line':
                polygon = np.asarray(self.scene.features[target['feature_id']]['polygon_xyz'], float)
                index = int(target['index'])
                direction = int(np.argmax(np.abs(polygon[(index+1) % len(polygon), :2]-polygon[index, :2])))
                return spec['kind'] == ('boundary_length' if direction == 0 else 'boundary_width')
            if target['kind'] == 'board_landmark':
                landmark = self.landmark_specs[target['landmark_id']]
                axis = 0 if spec['kind'] == 'boundary_length' else 1
                return axis in landmark_boundary_axes(landmark)
            return target['kind'] in ('board_side', 'board_corner')
        if spec['kind'] == 'circle_shared_radius':
            return (target.get('feature_id') in spec['feature_ids'] and target['kind'] == 'circle_outline'
                    and target.get('ring_role', 'outer') == 'outer')
        if spec['kind'] == 'board_height':
            if target['kind'] == 'board_landmark':
                return True
            return target['kind'] in ('board_side', 'board_corner', 'board_curve', 'board_point') and target.get('edge') == 'top'
        if spec['kind'] == 'boundary_corner_radius':
            if target['kind'] == 'board_landmark':
                return spec['corner'] in landmark_corner_dependencies(self.landmark_specs[target['landmark_id']])
            side = target.get('side', target.get('corner', ''))
            corner = spec['corner']
            return target['kind'] in ('board_side', 'board_corner') and (side == corner or
                (side == 'upper_long_side' and corner.startswith('upper')) or
                (side == 'lower_long_side' and corner.startswith('lower')) or
                (side == 'left_end' and corner.endswith('left')) or
                (side == 'right_end' and corner.endswith('right')))
        if spec['kind'] == 'shared_corner_radius':
            return target['kind'] in ('board_side', 'board_corner', 'board_landmark', 'board_curve', 'board_point')
        if target.get('feature_id') != spec.get('feature_id'):
            return False
        if spec['kind'].startswith('lane_bound_') and target['kind'] == 'polygon_supporting_line':
            polygon = np.asarray(self.scene.features[target['feature_id']]['polygon_xyz'], float)
            index = int(target['index'])
            direction = int(np.argmax(np.abs(polygon[(index+1) % len(polygon), :2]-polygon[index, :2])))
            normal = 1-direction
            edge = 'min' if np.isclose(polygon[index, normal], polygon[:, normal].min()) else 'max'
            return spec['kind'] == f'lane_bound_{"x" if normal == 0 else "y"}_{edge}'
        if spec['kind'] == 'lane_left_endpoint_x':
            vertex = 0 if spec['edge'] == 'upper' else 3
            return ((target['kind'] == 'polygon_vertex' and int(target['index']) == vertex) or
                    (target['kind'] == 'polygon_edge' and int(target['index']) == (0 if vertex == 0 else 2)) or
                    target['kind'] == 'polygon_outline')
        if spec['kind'] == 'circle_radius':
            return target['kind'] == 'circle_outline' and target.get('ring_role', 'outer') == spec.get('ring_role', 'outer')
        return True

    def circle_order_margins(self, geometry):
        result = {}
        for index, order in enumerate(self.circle_orders):
            for first, second in zip(order['feature_ids'], order['feature_ids'][1:]):
                key = f'{index}:{first}<{second}:axis{order["axis"]}'
                result[key] = (geometry.features[second]['center_xyz'][order['axis']]
                               - geometry.features[first]['center_xyz'][order['axis']] - order['min_gap'])
        return result

    def geometry(self, x):
        landmarks = {}
        for lid, spec in self.landmark_specs.items():
            along, z_fraction = (x[self.landmark_slices[lid]] if lid in self.landmark_slices
                                 else (spec['initial_along'], spec['initial_z_fraction']))
            landmarks[lid] = {'side': spec['side'], 'along': float(along), 'z_fraction': float(z_fraction),
                             'allow_corner': spec.get('allow_corner'),
                             'along_parameterization': spec['along_parameterization']}
        return SharedGeometry(self.scene, self.scene_specs, np.asarray(x)[:self.scene_count], landmarks,
                              self.hard_geometry_constraints)

    def camera(self, x, fid):
        frame = self.frames[fid]; iid = frame['installation_id']; install = self.installations[iid]
        center = x[self.center_slices[iid]] if iid in self.center_slices else install['initial_center_xyz']
        if fid in self.frame_slices:
            pars = x[self.frame_slices[fid]]; rot, focal = pars[:3], np.exp(pars[3])
        else:
            rot, focal = frame['initial_rvec_world_to_camera'], frame['initial_focal_px']
        cx, cy = frame['principal_point_px']
        K = np.array([[focal, 0., cx], [0., focal * frame.get('fy_over_fx', 1.), cy], [0., 0., 1.]])
        radial = install.get('radial_distortion')
        distortion = np.asarray(frame.get('distortion_opencv', []), float)
        if radial is not None:
            if len(distortion) and np.any(distortion):
                raise ValueError('do not mix fixed OpenCV distortion with image-radius installation distortion')
            k = x[self.radial_indices[iid]] if iid in self.radial_indices else radial['initial_k1_image']
            distortion = np.array([k * (focal / radial['scale_image'])**2, 0., 0., 0., 0.])
        return Camera(center, rot, K, distortion)

    def _error(self, x, o, uv, geometry, cameras):
        xyz = geometry.points(o['target'], o['type'], self.samples)
        cam = cameras[o['frame_id']]
        pred, depth = cam.project(xyz)
        if not np.isfinite(pred).all():
            raise ValueError('nonfinite projection')
        delta = pred - uv if o['type'] == 'point' else nearest_polyline_vectors(uv, pred)
        fold = 1.
        if self.installations[self.frames[o['frame_id']]['installation_id']].get('radial_distortion'):
            q = (xyz - cam.center_xyz) @ cam.R.T
            rr = np.sum((q[:, :2] / np.maximum(q[:, 2:3], 1e-8))**2, axis=1)
            fold = float(np.min(1 + 3 * cam.distortion_opencv[0] * rr))
        return delta, depth, fold

    def residual(self, x):
        geometry = self.geometry(x)
        cameras = {fid: self.camera(x, fid) for fid in self.frames}
        parts = []
        for o, uv in self.observations:
            if o['use'] != 'fit':
                continue
            delta, depth, fold = self._error(x, o, uv, geometry, cameras)
            repeats = self.track_counts[o.get('correlation_group_id', o['id'])] if self.problem.get('balance_track_groups', True) else 1
            weight = np.sqrt(o.get('weight', 1.) / len(uv) / repeats) / o['sigma_px']
            parts.extend([(delta * weight).ravel(),
                          np.array([min(float(depth.min()) - 1e-6, 0.) * 1e3,
                                    min(fold - 1e-5, 0.) * 1e3])])
        parts.extend(np.array([(x[index] - prior['mean']) / prior['sigma']]) for index, prior in self.parameter_priors)
        # A board detail cannot migrate outside its identified surface domain.
        # Only points with fit observations enter this geometric domain guard.
        parts.extend(np.minimum(geometry.board_landmark_margins(lid), 0.) * 1e3
                     for lid in self.active_board_landmark_ids)
        if self.variable_boundary:
            parts.append(np.minimum(np.asarray(list(geometry.boundary_validity_margins().values()))
                                    - GEOMETRY_EPSILON, 0.) * 1e3)
        if self.circle_orders:
            parts.append(np.minimum(np.asarray(list(self.circle_order_margins(geometry).values())), 0.) * 1e3)
        return np.concatenate(parts) if parts else np.empty(0)

    def _make_sparsity(self):
        rows = sum(2 * len(uv) + 2 for o, uv in self.observations if o['use'] == 'fit') + len(self.parameter_priors) + 2 * len(self.active_board_landmark_ids)
        rows += (10 if self.variable_boundary else 0) + sum(len(order['feature_ids']) - 1 for order in self.circle_orders)
        matrix = lil_matrix((rows, len(self.x0)), dtype=np.int8)
        cursor = 0
        for o, uv in self.observations:
            if o['use'] != 'fit':
                continue
            fid = o['frame_id']; iid = self.frames[fid]['installation_id']; end = cursor + 2 * len(uv) + 2
            for spec in self.scene_specs:
                if self._affects(spec, o['target']):
                    matrix[cursor:end, spec['index']] = 1
            lid = o['target'].get('landmark_id')
            if lid in self.landmark_slices:
                matrix[cursor:end, self.landmark_slices[lid]] = 1
            if iid in self.center_slices:
                matrix[cursor:end, self.center_slices[iid]] = 1
            if iid in self.radial_indices:
                matrix[cursor:end, self.radial_indices[iid]] = 1
            if fid in self.frame_slices:
                matrix[cursor:end, self.frame_slices[fid]] = 1
            cursor = end
        for index, _ in self.parameter_priors:
            matrix[cursor, index] = 1; cursor += 1
        for lid in self.active_board_landmark_ids:
            if lid in self.landmark_slices:
                matrix[cursor:cursor+2, self.landmark_slices[lid].start] = 1
            landmark = self.landmark_specs[lid]
            # Direct corner fractions have a fixed [0,1] domain; physical
            # dimensions only affect their projections, handled above.
            if landmark['side'] in STRAIGHT_SIDES:
                corners = SIDE_CORNERS[landmark['side']]
                axis_kind = 'boundary_length' if landmark['side'] in ('upper_long_side', 'lower_long_side') else 'boundary_width'
                for spec in self.scene_specs:
                    if spec['kind'] == axis_kind or (
                            spec['kind'] == 'boundary_corner_radius' and spec['corner'] in corners) or (
                            spec['kind'] == 'shared_corner_radius'):
                        matrix[cursor:cursor+2, spec['index']] = 1
            cursor += 2
        if self.variable_boundary:
            for spec in self.scene_specs:
                if spec['kind'] in BOUNDARY_KINDS:
                    matrix[cursor:cursor+10, spec['index']] = 1
            cursor += 10
        for order in self.circle_orders:
            for first, second in zip(order['feature_ids'], order['feature_ids'][1:]):
                for spec in self.scene_specs:
                    if any(self._affects(spec, {'kind': 'circle_center', 'feature_id': fid}) for fid in (first, second)):
                        matrix[cursor, spec['index']] = 1
                cursor += 1
        return matrix.tocsr()

    @property
    def jac_sparsity(self):
        return self._sparsity.copy()

    def board_surface_diagnostics(self, x):
        """Report surface identity/domain without claiming observed coverage.

        A supported side is a solver capability. It is not evidence that any
        advertisement there has actually been matched or used in fitting.
        """
        geometry = self.geometry(x)
        rows = []
        for lid, point in geometry.board_landmarks.items():
            margins = geometry.board_landmark_margins(lid)
            rows.append({'landmark_id': lid,
                         'advertisement_instance_id': self.landmark_specs[lid]['advertisement_instance_id'],
                         'identified_side': point['side'], 'allowed_corner': point.get('allow_corner'),
                         'surface_segment': geometry.board_landmark_segment(lid),
                         'along_parameterization': point['along_parameterization'],
                         'along_domain_margins': margins.tolist(),
                         'margin_units': 'quarter_circle_fraction' if point['side'] in CORNERS else 'scene_units',
                         'inside_allowed_surface': bool(np.all(margins >= -1e-8)),
                         'used_in_fit': lid in self.active_board_landmark_ids})
        return {'supported_straight_sides': list(STRAIGHT_SIDES),
                'supported_corners': list(CORNERS),
                'supported_adjacent_corner_continuations': {side: list(corners) for side, corners in SIDE_CORNERS.items()},
                'direct_corner_parameterization': 'corner_fraction; counterclockwise from CORNER_START_DEGREES; [0,1]',
                'direct_corner_start_degrees': dict(CORNER_START_DEGREES),
                'height_parameterization': 'z_fraction_times_shared_board_height',
                'landmarks': rows,
                'note': 'Surface support is implementation coverage, not a claim that observations exist on the entire perimeter.'}

    def summarize(self, x):
        geometry = self.geometry(x); cameras = {fid: self.camera(x, fid) for fid in self.frames}
        hard_checks = geometry.hard_geometry_checks()
        hard_valid = hard_checks is None or hard_checks['all_constraints_satisfied']
        boundary_margins = geometry.boundary_validity_margins()
        invalid_boundary = [name for name, margin in boundary_margins.items() if margin < GEOMETRY_EPSILON - 1e-9]
        order_margins = self.circle_order_margins(geometry)
        invalid_order = [name for name, margin in order_margins.items() if margin < -1e-8]
        circle_checks = geometry.circle_validity_checks()
        invalid_circles = [row['feature_id'] for row in circle_checks if not row['valid']]
        rows, groups = [], {'fit': [], 'check': []}
        for o, uv in self.observations:
            delta, depth, fold = self._error(x, o, uv, geometry, cameras)
            errors = np.linalg.norm(delta, axis=1).tolist()
            rows.append({'id': o['id'], 'frame_id': o['frame_id'], 'use': o['use'],
                         'target': o['target'], 'min_depth_scene_units': float(depth.min()),
                         'minimum_radial_derivative': fold, **_stats(errors)})
            groups[o['use']].extend(errors)
        return {'observations': rows, 'overall': {use: _stats(errors) for use, errors in groups.items()},
                'hard_geometry_checks': hard_checks,
                'boundary_geometry_valid': not invalid_boundary,
                'boundary_invalid_constraints': invalid_boundary,
                'boundary_validity_margins_scene_units': boundary_margins,
                'circle_order_valid': not invalid_order,
                'circle_order_invalid_constraints': invalid_order,
                'circle_order_margins_scene_units': order_margins,
                'circle_geometry_valid': not invalid_circles,
                'circle_invalid_feature_ids': invalid_circles,
                'circle_geometry_checks': circle_checks,
                'circle_geometry_check_use': 'acceptance_and_export_only; unobserved_circles_do_not_add_fitting_residuals',
                'fit_geometry_valid': hard_valid and not invalid_boundary and not invalid_order and not invalid_circles and all(r['min_depth_scene_units'] > 0 and r['minimum_radial_derivative'] > 0
                                          for r in rows if r['use'] == 'fit') and all(
                                              np.all(geometry.board_landmark_margins(lid) >= -1e-8) for lid in self.active_board_landmark_ids),
                'board_landmarks_outside_allowed_surface': [lid for lid in self.active_board_landmark_ids
                    if np.any(geometry.board_landmark_margins(lid) < -1e-8)],
                'board_surface_diagnostics': self.board_surface_diagnostics(x),
                'note': 'Pixel counts are correlated samples, not independent evidence counts.'}

    def export_scene_data(self, x, revision='joint_scene_candidate'):
        """Return updated semantic scene; caller must rebuild changed boundary meshes.

        Height-only updates include existing board mesh/panel top vertices. Radius
        and dimension edits invalidate boundary-derived meshes; do not ship stale
        meshes as if they had been rebuilt.
        """
        data = copy.deepcopy(self.scene.data); geometry = self.geometry(x)
        invalid_boundary = [name for name, margin in geometry.boundary_validity_margins().items()
                            if margin < GEOMETRY_EPSILON - 1e-9]
        invalid_order = [name for name, margin in self.circle_order_margins(geometry).items() if margin < -1e-8]
        invalid_circles = [row['feature_id'] for row in geometry.circle_validity_checks() if not row['valid']]
        invalid_landmarks = [lid for lid in self.active_board_landmark_ids
                             if np.any(geometry.board_landmark_margins(lid) < -1e-8)]
        hard_checks = geometry.hard_geometry_checks()
        if hard_checks is not None and not hard_checks['all_constraints_satisfied']:
            raise ValueError(f'cannot export violated hard geometry: {hard_checks}')
        if invalid_boundary or invalid_order or invalid_circles or invalid_landmarks:
            raise ValueError(f'cannot export invalid geometry: boundary={invalid_boundary}; circle_order={invalid_order}; circles={invalid_circles}; board_landmarks={invalid_landmarks}')
        data['ice_features'] = list(geometry.features.values())
        data['parameters'] = geometry.parameters
        data['board_landmarks'] = self.landmark_estimates(x)
        if self.hard_geometry_constraints:
            data['hard_geometry_constraints'] = copy.deepcopy(self.hard_geometry_constraints)
            data['hard_geometry_checks'] = hard_checks
        if self.variable_boundary:
            data['parameters']['corner_radii'] = geometry.corner_radii
        data['revision'] = revision
        data['status'] = 'video_joint_candidate_not_physically_validated'
        data.setdefault('uncertainty', {})['physical_accuracy_verified'] = False
        data['joint_refinement'] = {'base_scene_sha256': self.scene.sha256,
            'coordinate_gauge': self.coordinate_gauge_description(),
            'parameter_changes': self.parameter_changes(x),
            'retained_unobserved_parameters': self.retained_parameters,
            'geometric_constraints': copy.deepcopy(self.problem.get('geometric_constraints', {})),
            'hard_geometry_constraints': copy.deepcopy(self.hard_geometry_constraints),
            'hard_geometry_checks': hard_checks,
            'board_surface_diagnostics': self.board_surface_diagnostics(x),
            'circle_geometry_check_use': 'acceptance_and_export_only; unobserved_circles_do_not_add_fitting_residuals',
            'boundary_mesh_rebuild_required': self.variable_boundary,
            'physical_accuracy_verified': False}
        shared_lane_specs = [s for s in self.scene_specs if s['kind'] == 'shared_lane_half_width']
        if shared_lane_specs:
            records = [{
                'kind': 'shared_lane_half_width', 'parameter_id': s['id'],
                'feature_ids': list(s['feature_ids']), 'half_width_scene_units': float(x[s['index']]),
                'user_hard_constraint': bool(self.hard_geometry_constraints and self.hard_geometry_constraints.get('lane_equal_width')),
                'interpretation': ('Declared equal-width lane geometry.' if self.hard_geometry_constraints and
                    self.hard_geometry_constraints.get('lane_equal_width') else
                    'Optional equality hypothesis for this candidate.')
            } for s in shared_lane_specs]
            if self.hard_geometry_constraints and self.hard_geometry_constraints.get('lane_equal_width'):
                data['joint_refinement']['hard_shared_lane_width'] = records[0]
            else:
                data['joint_refinement']['candidate_geometry_assumptions'] = records
        height = geometry.parameters['board_height']; old_height = self.scene.data['parameters']['board_height']
        if height != old_height:
            for mesh in data['meshes']:
                if mesh['name'] == 'rink_boards':
                    for point in mesh['positions']:
                        if abs(point[2] - old_height) < 1e-9:
                            point[2] = height
            for panel in data['board_surface']['panels']:
                for key in ('height_range', 'z_range'):
                    if key in panel:
                        panel[key][1] = height
        changed_features = {s['feature_id'] for s in self.scene_specs if 'feature_id' in s}
        changed_features.update(fid for s in self.scene_specs for fid in s.get('feature_ids', []))
        if any(s['kind'] in ('shared_crease_depth', 'shared_crease_half_width') or
               s['kind'] == 'symmetric_line_distance' and s['feature_ids'] == ['goal_line_left', 'goal_line_right']
               for s in self.scene_specs):
            changed_features.update(('goal_crease_left', 'goal_crease_right'))
        for feature in data['ice_features']:
            if feature['id'] in changed_features:
                feature['geometry_status'] = 'joint_video_fit_candidate_not_independent_validation'
        return data

    def coordinate_gauge_description(self):
        dimensions = [s['kind'] for s in self.scene_specs if s['kind'] in ('boundary_length', 'boundary_width')]
        if len(dimensions) == 2:
            return ('Original boundary center and axis directions retained; length and width are independent variables, '
                    'so aspect ratio is not fixed. Model-unit scale remains conditional, not measured metres.')
        if dimensions:
            return (f'Original boundary center and axis directions retained; {dimensions[0]} varies and the other dimension '
                    'remains fixed. Model-unit scale and the unvaried dimension remain conditional.')
        return ('Original boundary extents, origin and axes retained; this fixes aspect ratio as well as a model-unit '
                'scale. The aspect ratio is a conditional geometry assumption, not merely a coordinate gauge.')

    def landmark_estimates(self, x):
        geometry = self.geometry(x); rows = []
        def source_frame(fid):
            frame = self.frames[fid]
            row = {key: frame.get(key) for key in ('id', 'installation_id', 'source_id', 'media_track_id',
                   'source_pts', 'time_base', 'image_path', 'image_sha256', 'pixel_frame_id')}
            if frame.get('source_pts') is not None and frame.get('time_base') is not None:
                row['source_seconds'] = float(int(frame['source_pts']) * Fraction(frame['time_base']))
            return row

        for lid, spec in self.landmark_specs.items():
            observations = [o for o, _ in self.observations if o['target'].get('landmark_id') == lid]
            fit = [o for o in observations if o['use'] == 'fit']
            frames = sorted({o['frame_id'] for o in fit})
            installations = sorted({self.frames[fid]['installation_id'] for fid in frames})
            margins = geometry.board_landmark_margins(lid)
            warnings = []
            if len(frames) < 2:
                warnings.append('fewer_than_two_fit_views; latent_point_and_camera_jointly_confounded')
            if len(installations) == 1:
                warnings.append('single_installation_only; pan_zoom_is_not_independent_translation_baseline')
            if lid in self.landmark_slices and any(s['kind'] == 'board_height' for s in self.scene_specs) and not any(
                    o['use'] == 'fit' and o['target'].get('edge') == 'top' for o, _ in self.observations):
                warnings.append('height_fraction_and_free_board_height_need_an_independent_vertical_constraint')
            if np.any(margins < -1e-6):
                warnings.append('outside_current_allowed_board_surface')
            rows.append({**{k: v for k, v in spec.items() if k != 'index'},
                'along': geometry.board_landmarks[lid]['along'],
                'z_fraction': geometry.board_landmarks[lid]['z_fraction'],
                'estimated_xyz': geometry.board_landmark(lid).tolist(),
                'surface_segment': geometry.board_landmark_segment(lid),
                'along_parameterization': spec['along_parameterization'],
                'surface_domain_margins': margins.tolist(),
                'surface_domain_margin_units': 'quarter_circle_fraction' if spec['side'] in CORNERS else 'scene_units',
                'inside_allowed_surface': bool(np.all(margins >= -1e-8)),
                'fit_observation_ids': [o['id'] for o in fit],
                'check_observation_ids': [o['id'] for o in observations if o['use'] == 'check'],
                'fit_frame_ids': frames, 'fit_installation_ids': installations,
                'source_frames': [source_frame(fid) for fid in frames],
                'check_source_frames': [source_frame(fid) for fid in sorted({o['frame_id'] for o in observations if o['use'] == 'check'})],
                'degeneracy_warnings': warnings, 'physical_xyz_verified': False,
                'interpretation': 'shared latent board-surface detail; image correspondence is observed, XYZ is estimated jointly and remains conditional'})
        return rows

    def parameter_changes(self, x):
        return [{k: v for k, v in spec.items() if k != 'index'} | {
                'old_value': spec['original_value'], 'new_value': float(x[spec['index']]),
                'delta': float(x[spec['index']] - spec['original_value']),
                'status': 'conditional_joint_estimate_not_measurement'} for spec in self.scene_specs]

    def solve(self, max_nfev=None, *, verbose=0, callback=None, x0=None):
        """Solve, optionally warm starting and forwarding SciPy's callback.

        A callback with signature ``callback(intermediate_result)`` receives
        SciPy's OptimizeResult containing x, fun, cost and iteration metadata.
        Callers can checkpoint every desired number of iterations; checkpointing
        is not an evidence limit. Resume with x0=checkpoint['x'], after checking
        the saved parameter_names and problem/scene hashes against this problem.
        """
        if not len(self.x0):
            raise ValueError('no variable parameters; use summarize for a fixed scene/camera check')
        initial = self.x0 if x0 is None else np.asarray(x0, float)
        if initial.shape != self.x0.shape or not np.isfinite(initial).all():
            raise ValueError('warm start must match parameter vector size and be finite')
        if np.any(initial < self.bounds[0]) or np.any(initial > self.bounds[1]):
            raise ValueError('warm start outside parameter bounds')
        result = least_squares(self.residual, initial, bounds=self.bounds,
            jac_sparsity=self._sparsity, method='trf', tr_solver='lsmr',
            x_scale='jac', loss='soft_l1', f_scale=1., max_nfev=max_nfev, verbose=verbose,
            callback=callback)
        self.result = result
        # The Gram matrix keeps the SVD proportional to parameter count, not
        # observation count. This remains only a local numerical diagnostic.
        jac = result.jac
        if hasattr(jac, 'power'):
            norms = np.sqrt(np.asarray(jac.power(2).sum(axis=0)).ravel())
            normalized = jac.multiply(1. / np.maximum(norms, 1e-15))
            gram = (normalized.T @ normalized).toarray()
        else:
            norms = np.linalg.norm(jac, axis=0)
            normalized = jac / np.maximum(norms, 1e-15)
            gram = normalized.T @ normalized
        eigen = np.linalg.eigvalsh(gram)
        singular = np.sqrt(np.maximum(eigen[::-1], 0.))
        threshold = singular[0] * 1e-6 if len(singular) else 0.
        installations = []
        for iid, install in self.installations.items():
            related = next((fid for fid, frame in self.frames.items() if frame['installation_id'] == iid), None)
            if related is None:
                continue
            radial = install.get('radial_distortion')
            installations.append({'id': iid, 'physical_camera_id': install.get('physical_camera_id'),
                'center_xyz': self.camera(result.x, related).center_xyz.tolist(),
                'position_status': 'conditional_joint_candidate_NOT_measured_installation',
                'center_bounds_xyz': install.get('center_bounds_xyz'),
                'radial_k1_image': None if radial is None else float(result.x[self.radial_indices[iid]] if iid in self.radial_indices else radial['initial_k1_image']),
                'radial_scale_image': None if radial is None else radial['scale_image']})
        after = self.summarize(result.x)
        return {'schema_version': 'rinkspace.joint-scene-fit.v1',
            'synthetic': self.problem.get('synthetic', False), 'scene_sha256': self.scene.sha256,
            'scene_geometry_held_fixed': not bool(self.scene_specs or self.landmark_slices), 'physical_accuracy_verified': False,
            'rink_geometry_held_fixed': not bool(self.scene_specs),
            'board_detail_geometry_held_fixed': not bool(self.landmark_slices),
            'accepted_geometry_candidate': bool(after['fit_geometry_valid']),
            'coordinate_gauge': self.coordinate_gauge_description(),
            'geometric_constraints': copy.deepcopy(self.problem.get('geometric_constraints', {})),
            'optimizer': {'success': bool(result.success), 'message': result.message,
                'nfev': int(result.nfev), 'cost': float(result.cost),
                'warm_started': x0 is not None,
                'near_bounds_parameters': [self.parameter_names[i] for i in np.flatnonzero(result.active_mask)]},
            'parameter_names': self.parameter_names, 'parameter_values': result.x.tolist(),
            'parameter_changes': self.parameter_changes(result.x), 'retained_parameters': self.retained_parameters,
            'board_landmarks': self.landmark_estimates(result.x),
            'before': self.summarize(self.x0), 'after': after,
            'installations': installations,
            'frames': [{**{k: v for k, v in frame.items() if not k.startswith('initial_')},
                        'camera': self.camera(result.x, fid).to_dict()} for fid, frame in self.frames.items()],
            'local_numerical_identifiability': {'column_normalized_singular_values': singular.tolist(),
                'rank_at_relative_1e-6': int(np.sum(singular > threshold)), 'parameter_count': len(result.x),
                'zero_sensitivity_parameters': [self.parameter_names[i] for i in np.flatnonzero(norms < 1e-12)],
                'note': 'Conditional local numerical diagnostic, not physical uncertainty or global uniqueness.'},
            'limitations': [self.coordinate_gauge_description(),
                'Optical assumptions remain conditional.',
                'Constraint penalties can leave an invalid numerical candidate; acceptance diagnostics and scene-export rejection must be respected.',
                'Only explicitly listed semantic parameters change; all other dimensions remain schematic.',
                'Shared optical centre is an approximation, not a surveyed tripod location.',
                'Check observations do not choose the solution; this problem alone does not establish unseen-clip validation.',
                'A feature used for geometry fitting is not independently geometry-validated by checking it in another frame.']}
