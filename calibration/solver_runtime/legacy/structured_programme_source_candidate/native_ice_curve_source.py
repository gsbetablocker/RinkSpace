"""Append explicit real native-image ice curves to the preserved full source.

New source pixels are separate from old row inventories and geometry priors.
Held curves are reviewed, never silently fitted. World curves use the current
named marking family and the same installation camera, without full-scene
reconstruction for every small derivative column.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import copy, deepcopy
import hashlib
import json
from pathlib import Path
import types

import numpy as np
from .native_camera_local import native_camera_local
from scipy.sparse import csr_matrix

from z190_streaming_source_normal import SourceRowBlock
from wide_anchor_candidate.full_brown_projection import curve_vector_error_full_brown
from z190_source_camera_diagnostic import horizontal_extent_value_jac15
from complete_ice_template_candidate.complete_ice_numeric_geometry import _boundary_hit
from .paired_programme_chart import CORE_COUNT, WORLD_INDICES
from calibration.solver_runtime.source_scanline_curve import (
    horizontal_scanline_curve_value, horizontal_scanline_curve_chart_response,
)


def _current_source_parameter_layout(state, parameter_count):
    """Read the owned evaluation chart; physical camera prefix stays intact."""
    chart = getattr(state, 'structured_programme_coordinate_chart', None)
    core_count = int(getattr(chart, 'core_count', CORE_COUNT))
    world_indices = np.asarray(
        getattr(state, 'geometry_parameter_indices', WORLD_INDICES), dtype=int).copy()
    world_count = int(getattr(state.geometry, 'parameter_count', len(world_indices)))
    if (world_indices.ndim != 1 or len(world_indices) != world_count
            or world_count < 50 or len(np.unique(world_indices)) != world_count
            or np.any(world_indices < 0) or np.any(world_indices >= core_count)
            or core_count > int(parameter_count)):
        raise ValueError('Owned complete evaluation core and logical current world columns required')
    return core_count, world_indices
from .current_ice_material_scene import CurrentIceMaterialView
from .fixed_pivot_positive_domain_fd import (
    positive_pivot_height_columns, positive_pivot_coordinate_derivative,
)

SOURCE_SHA = 'f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891'
CURRENT_SEGMENT_POLYLINES = frozenset((
    'B1', 'B2', 'C', 'goal_line_left', 'goal_line_right',
    'goal_trapezoid_left_lower_outer', 'goal_trapezoid_left_upper_outer',
    'goal_trapezoid_right_lower_outer', 'goal_trapezoid_right_upper_outer'))
CURRENT_SEGMENT_LANES = frozenset(('lane1', 'lane2', 'lane3', 'lane4'))
CURRENT_GOAL_CREASES = frozenset(('goal_crease_left', 'goal_crease_right'))


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _native_frame_reference(row):
    """Normalize explicit provenance aliases on the derived row only."""
    paths = [row[k] for k in ('source_raw_frame', 'source_path') if k in row]
    hashes = [row[k] for k in ('source_file_sha256', 'source_frame_sha256') if k in row]
    if not paths or not hashes:
        raise ValueError('Native source frame path and SHA256 required')
    resolved = [Path(p).resolve() for p in paths]
    if any(p != resolved[0] for p in resolved[1:]) or any(h != hashes[0] for h in hashes[1:]):
        raise ValueError('Native source frame provenance aliases disagree')
    row['source_raw_frame'] = str(resolved[0])
    row['source_file_sha256'] = hashes[0]
    return resolved[0], hashes[0]


def _axes(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c, s, 0.]), np.array([-s, c, 0.])


def _source_optical_support(row):
    """Use only explicitly admitted native optical support, before raw sigma.

    Existing rows keep their original objective, including rows whose normal
    intervals were saved solely as diagnostic metadata. Support never moves
    their observed UV or removes tangential finite-endpoint errors.
    """
    if row.get('source_optical_normal_support_authorized') is not True:
        return None
    from source_edge_intervals import SourceNormalIntervals
    uv = np.asarray(row['uv'], float)
    normal = np.asarray(row['normal'], float)
    if normal.shape != uv.shape or not np.isfinite(normal).all():
        raise ValueError('Explicit native source normals must match every UV')
    lengths = np.linalg.norm(normal, axis=1)
    if np.any(np.abs(lengths - 1.) > 1e-6):
        raise ValueError('Native optical support needs original unit normals')
    # Only float32 storage roundoff is removed in the derived operator.
    return SourceNormalIntervals(normal / lengths[:, None],
        row['normal_interval_px'], np.ones(len(uv), bool))


def _native_error_with_optical_support(row, raw):
    raw = np.asarray(raw, float).reshape(-1)
    support = row.get('_source_optical_support')
    return raw if support is None else support.apply(raw)


def _axis_scanline(row):
    return row.get('curve_observation_kind') == 'axis_scanline_branch'


def _scanline_value(camera, world, row, *, with_association=False):
    return horizontal_scanline_curve_value(camera, world, row['uv'],
        side='right', with_association=with_association)


def _curve_error_on_association(camera, world, observed, association=None):
    """Replay the original operator on one linearization's finite segments.

    Coordinate FD must differentiate the nominal nearest-segment branch. A
    larger coordinate perturbation can otherwise select an adjacent polyline
    segment, even though the actual coupled direction stays on the first one.
    Only the derivative replay fixes the association; source values and trial
    residuals continue to use the original nearest-segment search.
    """
    selected = []
    original = curve_vector_error_full_brown.__globals__['nearest_segments']

    def nearest(observed_uv, start, direction, length2, valid):
        if association is None:
            result = original(observed_uv, start, direction, length2, valid)
            selected.append((np.asarray(result[0], int).copy(),
                             np.asarray(result[1], float).copy()))
            return result
        index, nominal_t = association
        index = np.asarray(index, int)
        if (index.shape != (len(observed_uv),) or np.any(index < 0)
                or np.any(index >= len(start)) or not np.all(valid[index])):
            raise ValueError('Nominal native-ice derivative segment left its valid domain')
        delta = np.asarray(observed_uv) - start[index]
        tangent = direction[index]
        t = np.einsum('ni,ni->n', delta, tangent) / length2[index]
        # Freeze the nominal clamp branch too. Interior t differentiates its
        # unconstrained projection; a nominal clamped endpoint keeps t=0/1.
        # This matches the original analytic finite-segment Jacobian even if
        # an individual coordinate FD crosses a clamp or adjacent-segment cut.
        interior = (nominal_t > 0.) & (nominal_t < 1.)
        clipped = np.where(interior, t, np.clip(nominal_t, 0., 1.))
        return index, t, clipped, start[index] + clipped[:, None] * tangent

    globals_ = dict(curve_vector_error_full_brown.__globals__)
    globals_['nearest_segments'] = nearest
    replay = types.FunctionType(curve_vector_error_full_brown.__code__, globals_,
        '_native_ice_fixed_association_error', curve_vector_error_full_brown.__defaults__,
        curve_vector_error_full_brown.__closure__)
    raw = replay(camera, world, observed, with_jacobian=False)
    return np.asarray(raw).reshape(-1), selected[0] if association is None else association


class NativeIceCurveFactor:
    def __init__(self, state, packet_path, parameter_count):
        self.state = state
        self.parameter_count = int(parameter_count)
        self.packet_path = Path(packet_path).resolve()
        packet = json.loads(self.packet_path.read_text(encoding='utf8'))
        if (packet.get('schema') != 'rinkspace.additional-native-ice-curves.v1'
                or packet.get('source_video_sha256') != SOURCE_SHA
                or packet.get('time_base') != '1/15360'):
            raise ValueError('Explicit same-video native-image ice-curve packet required')
        from calibration.solver_runtime.source_point_qualification import load_source_point_qualification
        qualification_packet = packet
        if 'source_scanline_operator_revision' in packet:
            from calibration.solver_runtime.truth_source_scanline_revision import original_qualified_packet
            qualification_packet = original_qualified_packet(self.packet_path, packet)
        self.source_point_qualification = load_source_point_qualification(
            self.packet_path, qualification_packet, source_family='additional_native_ice_curves')
        self.geometry = state.geometry.current
        self.unit = float(self.geometry.scene_unit_m)
        self.core_count, self.world_indices = _current_source_parameter_layout(state, self.parameter_count)
        self.world_parameter_count = len(self.world_indices)
        self._ice_material_view = None
        self.rows, self.held = [], []
        ids = set()
        for record in packet['rows']:
            row = deepcopy(record)
            if row['id'] in ids:
                raise ValueError('Additional observation IDs must be unique')
            ids.add(row['id'])
            native = int(row['pts'])
            if native not in state.original.aliases:
                raise ValueError('Source PTS has no authenticated exposure alias')
            actual = int(state.original.aliases[native])
            if 'actual_pts' in row and int(row['actual_pts']) != actual:
                raise ValueError('Declared additional observation exposure disagrees with authenticated alias')
            if actual not in state.bypts or not 229114112 <= native < 231705600:
                raise ValueError('Additional observation lies outside the authorized programme')
            frame_path, frame_sha = _native_frame_reference(row)
            if _sha(frame_path) != frame_sha:
                raise ValueError('Native source frame hash changed')
            uv = np.asarray(row['uv'], float)
            sigma = np.broadcast_to(np.asarray(row['sigma_px'], float), (len(uv),)).copy()
            horizontal_extent = row.get('curve_observation_kind') == 'horizontal_extent'
            if _axis_scanline(row):
                receipt = row.get('scanline_location_measurement')
                if (row.get('scanline_fixed_axis') != 'y'
                        or row.get('scanline_branch') != 'max_x'
                        or row.get('curve', {}).get('kind') != 'house_ring'
                        or row.get('source_optical_normal_support_authorized') is True
                        or row.get('point_correspondence') is True
                        or row.get('endpoint_observation') is True
                        or not isinstance(receipt, dict)
                        or not receipt.get('path') or not receipt.get('sha256')
                        or _sha(receipt['path']) != receipt['sha256']):
                    raise ValueError('Authenticated fixed-y right-branch RGB location and original horizontal sigma required')
            if horizontal_extent and (row.get('curve',{}).get('kind')!='house_ring'
                    or row.get('extent_side')not in('left','right') or len(uv)!=1
                    or not np.array_equal(np.asarray(row.get('normal'),float),np.array([[1.,0.]]))
                    or row.get('point_correspondence')is not False
                    or row.get('endpoint_observation')is not False
                    or row.get('source_optical_normal_support_authorized')is not True):
                raise ValueError('Explicit whole house-ring horizontal source extent, one horizontal-normal support and no point/endpoint correspondence required')
            minimum_uv = 1 if horizontal_extent or _axis_scanline(row) or row.get('curve',{}).get('kind') == 'current_ice_segment' else 2
            if (uv.ndim != 2 or uv.shape[1:] != (2,) or len(uv) < minimum_uv
                    or not np.isfinite(uv).all() or not np.isfinite(sigma).all()
                    or np.any(sigma <= 0) or np.any(uv < 0)
                    or np.any(uv >= np.array([1920., 1080.]))):
                raise ValueError('Finite original 1920x1080 source UV and positive sigma required')
            row.update(actual_pts=actual, uv=uv, sigma=sigma)
            row['_source_optical_support'] = _source_optical_support(row)
            self._validate_curve(row['curve'])
            if row['curve']['kind'] in ('current_ice_segment', 'current_goal_crease'):
                if (row.get('feature_id') != row['curve']['feature_id']
                        or row.get('world') is not None):
                    raise ValueError('Current paint identity must agree; fixed world observations are prohibited')
                if self._ice_material_view is None:
                    self._ice_material_view = CurrentIceMaterialView(self.geometry)
            if row['split'] == 'fit':
                self.rows.append(row)
            elif row['split'] == 'holdout':
                self.held.append(row)
            else:
                raise ValueError('Explicit new fit/holdout admission required')
        if not self.rows:
            raise ValueError('No real additional fit observations')
        # The packet declares per-UV working engineering sigma. It does not
        # measure a covariance or an effective sample count. Preserve those
        # supplied scales directly instead of imposing curve-level mean loss.
        # Original programme source scales/covariance/profile remain delegated.
        self.scales = np.concatenate([np.repeat(r['sigma'], 2)
                                      for r in self.rows])
        if self.source_point_qualification is not None:
            self.scales = self.scales[self.source_point_qualification.scalar_mask(self.rows)]
        self.quadratic = np.ones(len(self.scales), bool)
        self.row_count = len(self.scales)
        self.blocks = {r['actual_pts']: state.bypts[r['actual_pts']]
                       for r in self.rows + self.held}
        camera_columns = []
        self.camera_steps = {}
        for actual, block in self.blocks.items():
            slots = np.r_[block.xyz_slots, block.optics_chart.indices,
                          block.optics_chart.mount_indices,
                          block.start + 4 * block.bundle.index[actual] + np.arange(4)].astype(int)
            camera_columns.extend(slots)
            for col in slots:
                self.camera_steps[int(col)] = 1e-6
            for col in block.optics_chart.indices:
                self.camera_steps[int(col)] = 1e-4
            view = block.start + 4 * block.bundle.index[actual]
            for col in range(view, view + 3):
                self.camera_steps[col] = 1e-7
        self.camera_columns = np.unique(camera_columns)
        if np.intersect1d(self.camera_columns, self.world_indices).size:
            raise ValueError('Camera and named marking columns overlap')
        self._cache = OrderedDict()
        self._active_fit_view_cache = None
        self.contract = dict(
            schema='rinkspace.additional-real-native-ice-curve-factor.v2',
            packet_path=str(self.packet_path), packet_sha256=_sha(self.packet_path),
            source_video_sha256=SOURCE_SHA, time_base='1/15360',
            new_fit_observation_ids=[r['id'] for r in self.rows],
            held_observation_ids=[r['id'] for r in self.held],
            new_scalar_rows=self.row_count, parameter_count=self.parameter_count,
            evaluation_core_parameter_count=self.core_count,
            logical_world_parameter_count=self.world_parameter_count,
            logical_world_parameter_indices=self.world_indices.tolist(),
            original_source_rows_UV_sigma_roles_profiles_and_clock_unchanged=True,
            new_target_objective_explicit=True,
            observation_operator='native full-Brown vector distance to current world curve',
            trace_objective='sum of per-UV squared native-pixel vector distance normalized by declared raw sigma',
            objective_revision='v2 removes unmeasured sqrt(group_UV_count) normalization from v1 only',
            group_sqrt_N_multiplier_applied=False,
            within_curve_statistical_independence_calibrated=False,
            within_curve_covariance_measured=False,
            raw_sigma_px_preserved=True,
            sigma_is_declared_working_weight_not_new_measured_uncertainty=True,
            no_plan_pixels_or_old_WORLD_as_image_observations=True,
            carrier_support_is_not_observed_paint_endpoint=True,
            held_rows_used_in_fit=False, accuracy_accepted=False)
        self.contract['explicit_optical_normal_support_ids'] = [
            r['id'] for r in self.rows + self.held
            if r['_source_optical_support'] is not None]
        self.contract['other_native_ice_objectives_unchanged'] = True
        if any(_axis_scanline(r) for r in self.rows + self.held):
            from calibration.solver_runtime.truth_source_scanline_revision import expected_scanline_contract
            self.contract.update(
                observation_operator='Native full-Brown distance with explicitly qualified fixed-axis scanline intersections',
                trace_objective='Original curve-vector distances and qualified measured-axis scanline distances normalized by original declared sigma',
                axis_scanline_location_objective=expected_scanline_contract(self.rows + self.held))
        if self.source_point_qualification is not None:
            self.contract['source_only_review_qualification'] = deepcopy(self.source_point_qualification.contract)
            self.contract['active_FIT_compaction'] = dict(
                schema='rinkspace.native-source-qualified-active-view.v1',
                original_scalar_rows=sum(2 * len(r['uv']) for r in self.rows),
                active_scalar_rows=self.row_count,
                removed_scalar_rows=sum(2 * len(r['uv']) for r in self.rows) - self.row_count,
                original_rows_retained_for_truth_and_review=True,
                qualified_rows_removed_before_residual_and_J=True,
                retained_UV_sigma_and_optical_support_unchanged=True,
                fresh_source_profiles_and_J_required=True)

    def _qualified_active_fit_view(self):
        """Keep immutable source rows for truth, compute only admitted FIT UV.

        A typed shallow factor view retains the existing parallel FD callback
        and private-owner path. Human-centre/suffix views may replace ``rows``;
        their identity invalidates this derived cache instead of inheriting a
        stale full-packet selection. No missing point becomes a zero residual.
        """
        qualification = getattr(self, 'source_point_qualification', None)
        if qualification is None:
            return None
        key = tuple(id(row) for row in self.rows)
        cached = getattr(self, '_active_fit_view_cache', None)
        if cached is not None and cached[0] == key:
            return cached[1]
        masks = [qualification.point_mask(row) for row in self.rows]
        if all(mask.all() for mask in masks):
            self._active_fit_view_cache = (key, None)
            return None
        active_rows, mapping = [], []
        for row, mask in zip(self.rows, masks):
            indices = np.flatnonzero(mask)
            mapping.append(dict(id=row['id'], original_point_count=len(mask),
                retained_original_point_indices=indices.tolist()))
            if not len(indices):
                continue
            if mask.all():
                active_rows.append(row)
                continue
            derived = dict(row)
            for field in ('uv', 'sigma', 'normal', 'normal_interval_px',
                          'source_blur_y_intervals', 'source_integration_weights'):
                if field in row:
                    values = np.asarray(row[field])
                    if values.ndim and len(values) == len(mask):
                        derived[field] = values[indices].copy()
            # sigma_px is provenance, sigma is the already-expanded numerical
            # operator. Never reinterpret inherited weights after exclusion.
            derived['_source_optical_support'] = _source_optical_support(derived)
            active_rows.append(derived)
        if not active_rows:
            raise ValueError('Source qualification leaves no active native ice FIT observations')
        view = copy(self)
        view.rows = active_rows
        view.source_point_qualification = None
        view._active_fit_view_cache = None
        view._cache = OrderedDict()
        view.scales = np.concatenate([np.repeat(row['sigma'], 2) for row in active_rows])
        view.quadratic = np.ones(len(view.scales), bool)
        view.row_count = len(view.scales)
        view._qualified_original_index_map = mapping
        self._active_fit_view_cache = (key, view)
        return view

    @staticmethod
    def _validate_curve(curve):
        kind = curve.get('kind')
        if kind in ('curling_centreline', 'house_ring'):
            if curve.get('row') not in (1, 2, 3, 4):
                raise ValueError('Explicit shared curling row required')
            if kind == 'house_ring' and (curve.get('side') not in ('F', 'R')
                    or curve.get('ring') not in ('outer', 'middle', 'inner', 'center')):
                raise ValueError('Explicit actual house and ring identity required')
        elif kind == 'hockey_circle':
            if curve.get('feature_id') not in ('C0', 'A1', 'A2', 'A3', 'A4'):
                raise ValueError('Explicit shared hockey-circle identity required')
        elif kind == 'current_goal_crease':
            if curve.get('feature_id') not in CURRENT_GOAL_CREASES:
                raise ValueError('Explicit existing left/right goal-crease polygon identity required')
        elif kind == 'current_ice_segment':
            identity, index = curve.get('feature_id'), curve.get('segment_index')
            if type(index) is not int:
                raise ValueError('Explicit integer current paint segment index required')
            if identity in CURRENT_SEGMENT_LANES:
                if index not in (0, 2):
                    raise ValueError('Only current Lane long painted sides 0/2 are admitted; left_open and short sides are excluded')
            elif identity in CURRENT_SEGMENT_POLYLINES:
                if index != 0:
                    raise ValueError('Only the explicitly modeled current polyline segment 0 is admitted')
            else:
                raise ValueError('Current ice segment feature is not whitelisted')
        else:
            raise ValueError('Unsupported native ice-curve identity')

    def _current_segment(self, p, curve):
        # This is the same current scene producer used by the complete export.
        # Source UV never supplies an old XYZ target or a painted endpoint.
        if self._ice_material_view is None:
            self._ice_material_view = CurrentIceMaterialView(self.geometry)
        features = self._ice_material_view.features(p)
        identity, index = curve['feature_id'], curve['segment_index']
        if identity not in features:
            raise ValueError('Whitelisted segment is absent from the actual current ice model')
        feature = features[identity]
        field = 'paint_segments_xyz' if identity in CURRENT_SEGMENT_LANES else 'segments_xyz'
        if field not in feature or not 0 <= index < len(feature[field]):
            raise ValueError('Requested painted segment is not modeled in the current producer')
        segment = np.asarray(feature[field][index], float)
        if (segment.shape != (2, 3) or not np.isfinite(segment).all()
                or np.linalg.norm(segment[1] - segment[0]) <= 0.):
            raise ValueError('Finite nondegenerate current straight paint segment required')
        return np.linspace(segment[0], segment[1], 1025)

    def _world(self, q, curve):
        p = self.geometry.effective_world(np.asarray(q, float)[self.world_indices])
        if curve['kind'] == 'current_goal_crease':
            if self._ice_material_view is None:
                self._ice_material_view = CurrentIceMaterialView(self.geometry)
            feature = self._ice_material_view.features(p)[curve['feature_id']]
            world = np.asarray(feature.get('polygon_xyz'), float)
            if (feature.get('closed') is not True or world.ndim != 2
                    or world.shape[1:] != (3,) or len(world) < 3
                    or not np.isfinite(world).all()):
                raise ValueError('Existing complete closed goal-crease paint polygon required')
            return world if np.array_equal(world[0], world[-1]) else np.vstack((world, world[0]))
        if curve['kind'] == 'current_ice_segment':
            return self._current_segment(p, curve)
        if curve['kind'] in ('curling_centreline', 'house_ring'):
            axis, across = _axes(p[14])
            center = np.r_[p[12:14], 0.] + across * ((curve['row'] - 2.5) * p[16])
            if curve['kind'] == 'curling_centreline':
                # The current ice-footprint intersections bound numerical
                # carrier coverage. They are not claimed paint endpoints,
                # sheet-length constraints or regulation hack/back dimensions.
                course, cq, _ = self.geometry.print_course_binding(p)
                bottom = course.sample_perimeter(cq, with_jacobian=False)[0]
                first = _boundary_hit(center / self.unit, -axis, np.asarray(bottom))
                last = _boundary_hit(center / self.unit, axis, np.asarray(bottom))
                return np.linspace(first, last, 1025)
            center += axis * p[15] / 2. * (-1. if curve['side'] == 'F' else 1.)
            ratio = {'outer': 1., 'middle': p[18], 'inner': p[19], 'center': p[20]}[curve['ring']]
            radius = p[17] * ratio
        else:
            axis, across = _axes(p[2])
            center = np.r_[p[:2], 0.]
            identity = curve['feature_id']
            if identity != 'C0':
                course, cq, _ = self.geometry.print_course_binding(p)
                bottom = course.sample_perimeter(cq, with_jacobian=False)[0]
                side = 'right' if identity in ('A1', 'A2') else 'left'
                inward = -axis if side == 'right' else axis
                wall = _boundary_hit(center / self.unit, -inward, np.asarray(bottom)) * self.unit
                goal = wall + inward * p[8 if side == 'right' else 7]
                sign = -1. if identity in ('A1', 'A4') else 1.
                center = goal + inward * p[4] + across * sign * p[5]
            radius = p[3]
        angle = np.linspace(0., 2. * np.pi, 361)
        return (center + radius * np.c_[np.cos(angle), np.sin(angle), np.zeros(len(angle))]) / self.unit

    def _cameras(self, q):
        core = np.asarray(q, float)[:self.core_count]
        return {actual: block.bundle.camera(native_camera_local(block,core), actual)
                for actual, block in self.blocks.items()}

    def _residual(self, q, rows, *, cameras=None):
        if cameras is None:
            cameras = self._cameras(q)
        values = []
        for row in rows:
            if _axis_scanline(row):
                raw = _scanline_value(cameras[row['actual_pts']], self._world(q, row['curve']), row)
            elif row.get('curve_observation_kind')=='horizontal_extent':
                raw=horizontal_extent_value_jac15(cameras[row['actual_pts']],
                    dict(row,world=self._world(q,row['curve'])),False)
            else:
                raw = curve_vector_error_full_brown(cameras[row['actual_pts']],
                    self._world(q, row['curve']), row['uv'], with_jacobian=False)
            # The original routine returns residual directly for value-only.
            if isinstance(raw, tuple):
                raw = raw[0]
            values.append(_native_error_with_optical_support(row, raw) / np.repeat(row['sigma'], 2))
        residual = np.concatenate(values) if values else np.empty(0)
        if not np.isfinite(residual).all():
            raise ValueError('Additional source leaves the valid optical domain')
        return residual

    def _derivative_residual(self, q, *, cameras, associations, apply_optical_support=True, row_indices=None):
        values = []
        groups = np.arange(len(self.rows)) if row_indices is None else np.asarray(row_indices, int)
        for position in groups:
            row, indices = self.rows[position], associations[position]
            if _axis_scanline(row):
                raw = horizontal_scanline_curve_chart_response(cameras[row['actual_pts']],
                    self._world(q, row['curve']), row['uv'], indices)
            elif row.get('curve_observation_kind')=='horizontal_extent':
                raw=horizontal_extent_value_jac15(cameras[row['actual_pts']],
                    dict(row,world=self._world(q,row['curve'])),False)
            else:
                raw, _ = _curve_error_on_association(cameras[row['actual_pts']],
                    self._world(q, row['curve']), row['uv'], indices)
            values.append(_native_error_with_optical_support(row, raw) if apply_optical_support else raw)
        # The declared factor scales also retain the diagnostic v1 sqrt(N)
        # replay's normalization without changing raw source sigma or roles.
        offsets = np.r_[0, np.cumsum([2 * len(row['uv']) for row in self.rows])]
        scalar_indices = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in groups])
        return np.concatenate(values) / self.scales[scalar_indices]

    def block(self, q, jacobian=True, *, original_start=0):
        view = self._qualified_active_fit_view()
        if view is not None:
            result = view.block(q, jacobian, original_start=original_start)
            result.row_ledger.update(
                source_only_review_qualification=deepcopy(self.source_point_qualification.contract),
                active_FIT_original_point_map=deepcopy(view._qualified_original_index_map),
                original_scalar_rows=sum(2 * len(row['uv']) for row in self.rows),
                removed_scalar_rows=sum(2 * len(row['uv']) for row in self.rows) - view.row_count,
                excluded_points_computed_as_zero_slots=False,
                retained_source_sigma_unchanged=True)
            if hasattr(view, '_source_dependency_fd_report'):
                self._source_dependency_fd_report = view._source_dependency_fd_report
            return result
        q = np.asarray(q, float)
        if q.shape != (self.parameter_count,) or not np.isfinite(q).all():
            raise ValueError('Explicit full current physical state required')
        active_world = np.unique(np.concatenate([
            self.world_indices[np.r_[np.arange(28), np.arange(44, 50)]]
                if r['curve']['kind'] == 'current_ice_segment'
                else self.world_indices[np.r_[[0, 1, 2, 7, 8, 9, 44, 45, 46, 47, 48, 49],list(getattr(self.geometry,'goal_crease_transverse_indices',{}).values())].astype(int)]
                if r['curve']['kind'] == 'current_goal_crease'
                else self.world_indices[[12, 13, 14, 15, 16, 17, 18, 19, 20, 44, 45, 46, 47, 48, 49]]
                if r['curve']['kind'] != 'hockey_circle'
                else self.world_indices[[0, 1, 2, 3, 4, 5, 7, 8, 44, 45, 46, 47, 48, 49]]
            for r in self.rows]))
        if getattr(self.geometry, 'shared_hockey_curling_axis', False):
            # Hockey's historical yaw slot is derived from the common curling
            # axis. Sparse row/suffix factors must differentiate the actual
            # owner, including when their own rows contain no curling feature.
            active_world = np.unique(np.where(active_world == self.world_indices[2],
                                               self.world_indices[14], active_world))
        fit_cameras = np.unique(np.concatenate([
            np.r_[self.blocks[r['actual_pts']].xyz_slots,
                  self.blocks[r['actual_pts']].optics_chart.indices,
                  self.blocks[r['actual_pts']].optics_chart.mount_indices,
                  self.blocks[r['actual_pts']].start +
                    4 * self.blocks[r['actual_pts']].bundle.index[r['actual_pts']] + np.arange(4)]
            for r in self.rows])).astype(int)
        active = np.unique(np.r_[active_world, fit_cameras])
        key = q[active].tobytes()
        cached = self._cache.get(key)
        if cached is not None and (not jacobian or cached[1] is not None):
            residual, matrix = cached
        else:
            cameras = self._cameras(q)
            residual = self._residual(q, self.rows, cameras=cameras)
            matrix = None
            if jacobian:
                associations=[];nominal_native=[]
                for row in self.rows:
                    if _axis_scanline(row):
                        raw, association = _scanline_value(cameras[row['actual_pts']],
                            self._world(q, row['curve']), row, with_association=True)
                    elif row.get('curve_observation_kind')=='horizontal_extent':
                        raw=horizontal_extent_value_jac15(cameras[row['actual_pts']],
                            dict(row,world=self._world(q,row['curve'])),False);association=None
                    else:
                        raw,association=_curve_error_on_association(cameras[row['actual_pts']],
                            self._world(q,row['curve']),row['uv'])
                    associations.append(association);nominal_native.append(raw)
                raw_native=np.concatenate(nominal_native)
                raw_residual=raw_native/self.scales
                # Differentiate the original raw finite-curve branch first.
                # The existing interval operator then maps J on its NOMINAL
                # interior/exterior branch; coordinate FD must not average
                # different support branches near an unchanged source bound.
                compact = np.zeros((len(residual), len(active)))
                height_columns = positive_pivot_height_columns(self.blocks.values())
                try:
                    from calibration.solver_runtime.native_factor_dependency_fd import native_factor_dependency_jacobian
                    compact = native_factor_dependency_jacobian(self, q, self.rows, active,
                        self.world_indices, raw_residual,
                        lambda value, groups: self._derivative_residual(value,
                            cameras=cameras if np.array_equal(value[fit_cameras], q[fit_cameras]) else self._cameras(value),
                            associations=associations, apply_optical_support=False, row_indices=groups), height_columns)
                finally:
                    self._cameras(q)
                cursor=0
                for row in self.rows:
                    stop=cursor+2*len(row['uv']);support=row.get('_source_optical_support')
                    if support is not None:
                        raw=raw_native[cursor:stop]
                        # Each UV has the original same sigma for x/y, so the
                        # native normal projection commutes with normalization.
                        compact[cursor:stop]=support.apply(raw,(compact[cursor:stop],))[1][0]
                    cursor=stop
                if cursor!=len(residual):raise ValueError('Original native ice support derivative row span changed')
                rr, cc = np.nonzero(compact)
                matrix = csr_matrix((compact[rr, cc], (rr, active[cc])),
                    shape=(len(residual), self.parameter_count))
            self._cache[key] = (residual.copy(), matrix)
            while len(self._cache) > 4:
                self._cache.popitem(last=False)
        ledger = dict(family='additional_real_native_ice_curves',
            original_start=int(original_start), original_stop=int(original_start) + self.row_count,
            packet_sha256=self.contract['packet_sha256'],
            objective_schema=self.contract['schema'],
            group_sqrt_N_multiplier_applied=False,
            native_pts=[r['pts'] for r in self.rows],
            actual_pts=[r['actual_pts'] for r in self.rows],
            feature_ids=[r['feature_id'] for r in self.rows],
            fit_observation_ids=[r['id'] for r in self.rows],
            held_observation_ids=[r['id'] for r in self.held],
            plan_pixels_used=False, paint_endpoints_inferred=False)
        qualification = getattr(self, 'source_point_qualification', None)
        if qualification is not None:
            residual, matrix, inactive = qualification.mask_values_and_jacobian(self.rows, residual, matrix)
            ledger.update(source_only_review_qualification=deepcopy(qualification.contract),
                inactive_scalar_slots_excluded_from_source_accuracy_statistics=(inactive + int(original_start)).tolist(),
                inactive_zero_residual_is_accuracy_measurement=False)
        return SourceRowBlock(residual, matrix, self.quadratic, self.scales, ledger)

    def review(self, q):
        cameras = self._cameras(q)
        result = []
        for row in self.rows + self.held:
            normalized = self._residual(q, [row], cameras=cameras)
            error = normalized.reshape(-1, 2) * row['sigma'][:, None]
            distance = np.linalg.norm(error, axis=1)
            scanline_distance = distance.copy() if _axis_scanline(row) else None
            if scanline_distance is not None:
                # Preserve the existing raw nearest-curve review metric for
                # before/after comparisons. The objective's horizontal error
                # has a different meaning and is reported separately.
                raw = curve_vector_error_full_brown(cameras[row['actual_pts']],
                    self._world(q, row['curve']), row['uv'], with_jacobian=False)
                if isinstance(raw, tuple): raw = raw[0]
                distance = np.linalg.norm(np.asarray(raw).reshape(-1, 2), axis=1)
            result.append(dict(id=row['id'], pts=row['pts'], actual_pts=row['actual_pts'],
                split=row['split'], feature_id=row['feature_id'], source_uv_count=len(distance),
                mean_native_px=float(distance.mean()), p90_native_px=float(np.percentile(distance, 90)),
                max_native_px=float(distance.max()), accuracy_accepted=False))
            if scanline_distance is not None:
                result[-1]['axis_scanline_location_objective'] = dict(
                    fixed_axis='y', branch='max_x',
                    mean_horizontal_error_px=float(scanline_distance.mean()),
                    p90_horizontal_error_px=float(np.percentile(scanline_distance, 90)),
                    signed_horizontal_error_px=error[:, 0].tolist(),
                    squared_normalized_horizontal_residual=float(normalized @ normalized),
                    original_horizontal_sigma_preserved=True,
                    legacy_nearest_curve_review_metric_preserved=True)
            qualification = getattr(self, 'source_point_qualification', None)
            qualified = qualification is not None and row['id'] in qualification.records
            if qualified:
                result[-1].update(qualification.review_metrics(row, distance),
                    source_only_review_record=deepcopy(qualification.records[row['id']]))
            if row['_source_optical_support'] is not None:
                if row.get('curve_observation_kind')=='horizontal_extent':
                    raw=horizontal_extent_value_jac15(cameras[row['actual_pts']],
                        dict(row,world=self._world(q,row['curve'])),False)
                else:
                    raw = curve_vector_error_full_brown(cameras[row['actual_pts']],
                        self._world(q, row['curve']), row['uv'], with_jacobian=False)
                if isinstance(raw, tuple): raw = raw[0]
                raw_distance = np.linalg.norm(np.asarray(raw).reshape(-1, 2), axis=1)
                if qualified:
                    raw_distance = raw_distance[qualification.point_mask(row)]
                result[-1].update(native_metric_is_outside_optical_support=True,
                    mean_raw_native_px=float(raw_distance.mean()) if len(raw_distance) else None,
                    max_raw_native_px=float(raw_distance.max()) if len(raw_distance) else None,
                    optical_support_is_not_physical_paint_width=True)
        return dict(contract=self.contract, rows=result, accuracy_accepted=False)


class SourceWithAdditionalNativeIceCurves:
    def __init__(self, base, factor):
        self.base, self.native_ice_factor = base, factor
        self.count, self.updater = int(base.count), base.updater
        self.quadratic = np.r_[base.quadratic, factor.quadratic]
        self.scales = np.r_[base.scales, factor.scales]
        self.contract = dict(schema='rinkspace.complete-source-plus-real-native-ice-curves.v1',
            original_source_contract=deepcopy(base.contract),
            additional_real_ice_curve_contract=deepcopy(factor.contract),
            original_source_rows=len(base.scales), additional_scalar_rows=factor.row_count,
            total_rows=len(self.scales), original_blocks_and_profiles_preserved=True,
            accuracy_accepted=False)

    def __getattr__(self, name):
        base = self.__dict__.get('base')
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def guard(self, q):
        self.base.guard(q)
        self.native_ice_factor.block(q, False)

    def blocks(self, q, jacobian=True):
        yield from self.base.blocks(q, jacobian)
        yield self.native_ice_factor.block(q, jacobian, original_start=len(self.base.scales))


def wrap_additional_native_ice_curves(base, state, packet_path):
    return SourceWithAdditionalNativeIceCurves(base,
        NativeIceCurveFactor(state, packet_path, base.count))
