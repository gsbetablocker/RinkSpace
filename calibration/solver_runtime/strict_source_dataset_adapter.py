"""Finite reviewed dataset on the existing complete physical parameter chart.

This factor replaces absolute pixels; it is deliberately not an append wrapper
and does not authorize any old motion/absolute family or relax a startup gate.
All source support selection is performed by the immutable dataset consumer.
"""
from __future__ import annotations

from collections import OrderedDict, Counter
from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csr_matrix, vstack

from calibration.program_tracking.strict_model_scope import authenticate_source_reference, file_sha
from calibration.program_tracking.strict_source_fit_dataset import SCHEMA
from calibration.program_tracking.strict_source_fit_dataset_stream import iter_dataset_rows


def _key(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def compact_dataset_rows(rows, *, split):
    """Only share a finite primitive call, retaining each original point locator.

    Concatenating observations does not concatenate source paths or create a
    chord. Nearest-segment association is always against the model primitive.
    The deterministic group permutation is explicit in the source locators.
    """
    if split not in ('fit', 'held'):
        raise ValueError('Separate original FIT and HELD required')
    groups = OrderedDict()
    for row in rows:
        if row['split'] != split or row['scope'] not in ('ice', 'board'):
            raise ValueError('Unconfirmed role/surface cannot enter strict factor')
        uv = np.asarray(row['uv'])
        sigma = np.asarray(row['sigma_px'])
        if (uv.ndim != 2 or uv.shape[1:] != (2,) or sigma.shape != (len(uv),)
                or not len(uv) or not np.isfinite(uv).all()
                or np.isinf(sigma).any() or np.any(sigma[np.isfinite(sigma)] <= 0)):
            raise ValueError('Original native finite UV and finite-positive-or-missing sigma required')
        if split == 'fit' and not np.isfinite(sigma).all():
            raise ValueError('Missing support cannot enter FIT')
        if any(k in row for k in ('quadratic', 'loss_weights', 'loss_weight', 'loss',
                'loss_function', 'robust_loss', 'huber_delta', 'trace_objective',
                'source_integration_mass_rule', 'source_integration_weights')):
            raise ValueError('Explicit dataset loss policy requires a qualified original loss adapter')
        # v6 has ordinary finite curves only. A later operator revision must
        # add an explicit implementation; no support metadata changes loss.
        if (row.get('curve_observation_kind') not in (None, 'measured_transition_location')
                or row.get('must_preserve_upper_branch')
                or any(row.get(k) is True for k in ('source_optical_normal_support_authorized',
                    'normal_interval_operator_authorized', 'source_point_bounds_authorized'))
                or any(k in row for k in ('normal_interval', 'normal_interval_px',
                    'point_support_bounds_xy_px', 'source_tangent_interval_px',
                    'clip_boundary_x', 'extent_side', 'scanline_branch', 'source_operator_array_fields'))):
            raise ValueError('Dataset declares an unimplemented typed source operator')
        point = row.get('point_correspondence') is True
        if point and split == 'fit':
            raise ValueError('This strict FIT revision has no qualified point-correspondence route')
        identity = _key(dict(actual_pts=int(row['actual_pts']), target=row['model_target'],
            scope=row['scope'], semantics=row['observation_semantics'], point_correspondence=point,
            physical_material_id=row['physical_material_id']))
        group = groups.setdefault(identity, dict(actual_pts=int(row['actual_pts']),
            model_target=dict(row['model_target']), scope=row['scope'], split=split,
            observation_semantics=row['observation_semantics'], physical_material_id=row['physical_material_id'],
            point_correspondence=point, uv_parts=[], sigma_parts=[], source_locators=[], points=0))
        lo = group['points']; hi = lo + len(uv)
        locator = dict(row['source_dataset_locator'])
        locator['group_point_range'] = [lo, hi]
        locator['original_native_pts'] = int(row['native_pts'])
        locator['decoded_BGR_sha256'] = row['decoded_BGR_sha256']
        locator['source_visible_paths'] = row.get('source_visible_paths')
        group['source_locators'].append(locator)
        group['uv_parts'].append(uv.copy()); group['sigma_parts'].append(sigma.copy())
        group['points'] = hi
    result = []
    for group in groups.values():
        group['uv'] = np.concatenate(group.pop('uv_parts'))
        group['sigma'] = np.concatenate(group.pop('sigma_parts'))
        group['uv'].flags.writeable = False; group['sigma'].flags.writeable = False
        result.append(group)
    return result


def partition_dataset_rows_by_family(rows, state):
    """Stable original programme-family order, without changing any point row."""
    groups = OrderedDict((block.name, []) for block in state.blocks)
    if len(groups) != len(state.blocks):
        raise ValueError('Distinct original programme families required')
    for row in rows:
        block = state.bypts[row['actual_pts']]
        if block.name not in groups:
            raise ValueError('Strict source exposure has no original programme family')
        groups[block.name].append(row)
    return OrderedDict((name, values) for name, values in groups.items() if values)


class StrictDatasetFactor:
    """Complete raw-sigma absolute factor, with bounded local CSR construction."""
    def __init__(self, state, dataset_reference, parameter_count, *, circle_factor,
                 material_factor, pattern_source=None):
        started = time.perf_counter()
        if isinstance(dataset_reference, (str, Path)):
            path = Path(dataset_reference).resolve()
            dataset_reference = dict(path=str(path), sha256=file_sha(path))
        authenticate_source_reference(dataset_reference)
        manifest = json.loads(Path(dataset_reference['path']).read_text(encoding='utf8'))
        if manifest['schema'] != SCHEMA:
            raise ValueError('Immutable finite reviewed dataset required')
        authenticate_source_reference(manifest['frozen_model_reference'])
        identity = json.loads(Path(manifest['frozen_model_reference']['path']).read_text(encoding='utf8'))
        self.state = state; self.parameter_count = int(parameter_count)
        from structured_programme_source_candidate.native_ice_curve_source import _current_source_parameter_layout
        from structured_programme_source_candidate.current_ice_material_scene import CurrentIceMaterialView
        self.core_count, self.world_indices = _current_source_parameter_layout(state, parameter_count)
        self.geometry = state.geometry.current
        self.ice = CurrentIceMaterialView(self.geometry)
        self.circle_factor, self.material_factor = circle_factor, material_factor
        self.pattern_source = pattern_source
        from .strict_dataset_read_runtime import load_strict_dataset_rows_once
        self.rows, self.held, self.read_receipt = load_strict_dataset_rows_once(dataset_reference)
        from .strict_dataset_target_bindings import apply_confirmed_v6_target_bindings
        self.target_binding_receipt = apply_confirmed_v6_target_bindings(self.rows, self.held, dataset_reference)
        fit_ready_seconds = time.perf_counter() - started
        print(json.dumps(dict(stage='strict_dataset_all_FIT_consumed',
            seconds=fit_ready_seconds, groups=len(self.rows),
            points=sum(len(r['uv']) for r in self.rows))), flush=True)
        self.family_groups = partition_dataset_rows_by_family(self.rows, state)
        self.rows = [row for values in self.family_groups.values() for row in values]
        counts = {role: sum(len(row['uv']) for row in values)
            for role, values in (('fit', self.rows), ('held', self.held))}
        if counts != dict(fit=manifest['fit_points'], held=manifest['held_points']):
            raise ValueError('Strict source graph lost original FIT/HELD points')
        if set(manifest['expected_actual_pts']) != set(state.bypts):
            raise ValueError('Strict source programme exposure clock differs from complete chart')
        aliases = state.original.aliases
        for row in self.rows + self.held:
            for locator in row['source_locators']:
                if aliases.get(locator['original_native_pts']) != row['actual_pts']:
                    raise ValueError('Dataset native/actual PTS differs from existing physical camera chart')
            target = row['model_target']
            if 'scene_q_sha256' in target and target['scene_q_sha256'] != identity['q_sha256']:
                raise ValueError('Print contour material initialization has changed identity')
        self.scales = np.concatenate([np.repeat(r['sigma'], 2) for r in self.rows])
        # v6 carries source support and geometric operators, but its rows and
        # manifest do not declare a statistical loss. Explicitly reuse the
        # complete native absolute-source component Huber2 convention. This
        # is a solver policy for a new input graph, not a source measurement.
        self.quadratic = np.zeros(len(self.scales), bool)
        self.row_count = len(self.scales)
        self.contract = dict(schema='rinkspace.strict-reviewed-finite-dataset-factor.v1',
            dataset_reference=dict(dataset_reference), original_fit_points=counts['fit'],
            held_points=counts['held'], scalar_rows=self.row_count,
            compact_fit_primitive_groups=len(self.rows), compact_held_primitive_groups=len(self.held),
            fit_source_blocks=len(self.family_groups),
            original_programme_family_order=list(self.family_groups),
            family_fit_points={name:sum(len(row['uv']) for row in rows)
                for name,rows in self.family_groups.items()},
            parameter_count=self.parameter_count, evaluation_core_parameter_count=self.core_count,
            logical_world_parameter_indices=self.world_indices.tolist(),
            source_video=manifest['source'], raw_per_point_sigma_preserved=True,
            pixels_or_time_subsampled=False, source_paths_joined=False,
            same_exposure_target_primitive_calls_shared=True, source_group_permutation_explicit_in_locators=True,
            authenticated_FIT_consumer_seconds=fit_ready_seconds,
            strict_source_read_receipt=self.read_receipt,
            confirmed_original_target_label_bindings=self.target_binding_receipt,
            strict_source_read_runtime_sha256=file_sha(Path(__file__).with_name('strict_dataset_read_runtime.py')),
            dataset_factor_runtime_sha256=file_sha(Path(__file__)),
            complete_factor_preparation_seconds=time.perf_counter()-started,
            original_source_proof_arrays_retained_in_authenticated_packs=True,
            missing_support_and_named_pending_not_fit=True, held_used_in_fit=False,
            operator='Existing full-Brown finite-segment vector distance divided by original per-point sigma',
            source_loss_policy=dict(schema='rinkspace.strict-new-source-explicit-loss-policy.v1',
                dataset_declares_robust_loss=False,
                chosen_solver_policy='existing_native_complete_absolute_source_component_Huber2',
                normalized_component_huber_delta=2., quadratic_scalar_rows=0,
                additional_integration_weights=None,
                loss_choice_is_source_measurement=False,
                raw_source_per_point_sigma_unchanged=True,
                prior_branch_cost_comparable=False,
                new_target_requires_fresh_profiles_J_and_same_graph_cost_comparisons=True),
            statistical_independence_or_sigma_accuracy_calibrated=False, accuracy_accepted=False)
        self._world_cache_q = None; self._world_cache = {}; self._derivative_cache = {}
        self._camera_cache_q = None; self._camera_cache = {}
        self._evaluation_q = None
        self.last_timing = None

    def _pattern_context(self, q):
        method = getattr(self.pattern_source, 'pattern_placement_context', None)
        return nullcontext() if method is None else method(q)

    def _check_q(self, q):
        value = np.asarray(q, float)
        if value.shape != (self.parameter_count,) or not np.isfinite(value).all():
            raise ValueError('Same complete finite evaluation state required')
        return value

    def _prepare_evaluation(self, q):
        """Authenticate full numeric identity once, before streaming groups."""
        token = q.tobytes()
        if self._world_cache_q != token:
            self._world_cache_q = token; self._world_cache = {}; self._derivative_cache = {}
        if self._camera_cache_q != token:
            self._camera_cache_q = token; self._camera_cache = {}
        self._evaluation_q = q

    def _ice_world(self, q, target):
        features = self.ice.features(q[self.world_indices])
        identity, role = target['feature_id'], target['world_curve_role']
        if target['curve_id'] != identity + ':' + role or identity not in features:
            raise ValueError('Exact current modeled ice material curve missing')
        feature = features[identity]
        # These are precisely the knot definitions of ice_curve_inventory,
        # the source dataset's navigation producer. Expand only the requested
        # material, preserving its original endpoints and closure convention.
        if feature['kind'] == 'circle':
            rings = feature.get('rings') or [dict(role='outer',
                center_xyz=feature['center_xyz'], radius=feature['radius'])]
            selected = [ring for ring in rings if ring['role'] == role]
            if len(selected) != 1:
                raise ValueError('Exact original source house layer absent')
            ring = selected[0]; theta = np.linspace(0., 2 * np.pi, 257)
            directions = np.c_[np.cos(theta), np.sin(theta), np.zeros(len(theta))]
            return np.asarray(ring['center_xyz']) + float(ring['radius']) * directions
        if feature['kind'] == 'polyline' and role.startswith('segment'):
            points = np.asarray(feature['segments_xyz'][int(role[7:])])
            return np.concatenate([np.linspace(a, b, 129)[:-1]
                for a, b in zip(points[:-1], points[1:])] + [points[-1:]])
        if feature['kind'] == 'polygon' and identity.startswith('lane') and role.startswith('edge'):
            points = np.asarray(feature['paint_segments_xyz'][int(role[4:])])
            return np.linspace(points[0], points[-1], 257)
        if feature['kind'] == 'polygon' and role == 'boundary':
            points = np.asarray(feature['polygon_xyz'], float)
            return np.vstack([np.linspace(a, b, 65)[:-1]
                for a, b in zip(points, np.roll(points, -1, axis=0))])
        raise ValueError('Unsupported current source ice material role')

    def _produce_world(self, q, target):
        kind = target['kind']; p = q[self.world_indices]
        if kind == 'current_scene_ice_curve':
            return self._ice_world(q, target), None
        if kind == 'advertisement_boundary':
            number = target['advertisement_number']
            if self.geometry._ad(number).get('extent_scope') != target['extent_scope']:
                raise ValueError('Source advertisement boundary does not match modeled material scope')
            edge = {'left_vertical': 'left_vertical_xyz', 'right_vertical': 'right_vertical_xyz',
                'top_edge': 'top_edge_xyz', 'bottom_edge': 'bottom_edge_xyz'}[target['edge']]
            u, v = self.geometry._materials[number, edge]
            world, jacobian, _ = self.geometry.material(number, u, v, p)
            return np.asarray(world), (self.world_indices, np.asarray(jacobian))
        if kind == 'current_scene_board_print_contour':
            number = target['advertisement_id']
            record = self.geometry.source_print_templates[self.geometry.source_template_instance[number]]
            contours = [c for c in record['external_contours'] if c['id'] == target['contour_id']]
            if len(contours) != 1:
                raise ValueError('Source-identified finite print contour absent from current material producer')
            material = np.asarray(contours[0]['initialization']['material_coordinates_normalized'], float)
            world, jacobian, _ = self.geometry.source_template_material(number, *material.T, p)
            return np.asarray(world), (self.world_indices, np.asarray(jacobian))
        if kind == 'board_circle':
            if self.circle_factor is None:
                raise ValueError('Original independently placed Ad11/24 circle producer required')
            return self.circle_factor.world(q, target['advertisement_number']), None
        if kind in ('shared_board_pattern_circle_curve', 'shared_board_pattern_material_points'):
            if self.material_factor is None:
                raise ValueError('Original current-course shared pattern material producer required')
            curve = dict(target)
            row = dict(curve=curve, material_roles=target.get('material_roles', []))
            with self._pattern_context(q):
                return self.material_factor._world(q, row), None
        raise ValueError('Unsupported exact source model target: ' + kind)

    def _world_columns(self, target):
        kind = target['kind']
        if kind == 'current_scene_ice_curve':
            # Actual producer controls extra Lane and crease columns. Keep all
            # current world columns; exact zero columns are removed afterwards.
            return self.world_indices
        if kind == 'board_circle':
            return np.unique(np.r_[self.circle_factor.active_world,
                self.circle_factor.active_circle_indices]).astype(int)
        if kind in ('shared_board_pattern_circle_curve', 'shared_board_pattern_material_points'):
            columns = np.r_[self.world_indices, np.arange(self.material_factor.pattern_start,
                self.material_factor.pattern_start + 9)]
            placements = getattr(self.pattern_source, 'placement_indices', [])
            return np.unique(np.r_[columns, placements]).astype(int)
        return self.world_indices

    def _world(self, q, target, jacobian):
        if self._evaluation_q is not q:
            self._prepare_evaluation(q)
        key = _key(target)
        if key not in self._world_cache:
            try:
                self._world_cache[key] = self._produce_world(q, target)
            except (ValueError, KeyError, IndexError) as error:
                raise ValueError('Strict current model target rejected: ' +
                    _key(target) + '; ' + str(error)) from error
        world, analytical = self._world_cache[key]
        if not jacobian:
            return world, None, None
        if key not in self._derivative_cache:
            if analytical is not None:
                columns, derivative = analytical
            else:
                columns = self._world_columns(target)
                derivative = np.empty((*world.shape, len(columns)))
                for i, column in enumerate(columns):
                    h = 1e-6
                    plus = q.copy(); minus = q.copy()
                    plus[column] += h; minus[column] -= h
                    wp = np.asarray(self._produce_world(plus, target)[0])
                    wm = np.asarray(self._produce_world(minus, target)[0])
                    if wp.shape != world.shape or wm.shape != world.shape:
                        raise ValueError('Current material derivative changed finite curve knots')
                    derivative[:, :, i] = (wp - wm) / (2 * h)
            nonzero = np.any(derivative != 0, axis=(0, 1))
            columns, derivative = np.asarray(columns, int)[nonzero], derivative[:, :, nonzero]
            self._derivative_cache[key] = (columns, derivative)
        return world, *self._derivative_cache[key]

    def _camera(self, q, actual, jacobian):
        from structured_programme_source_candidate.native_camera_local import native_camera_local
        from structured_programme_source_candidate.fixed_pivot_positive_domain_fd import (
            positive_pivot_height_columns, positive_pivot_coordinate_derivative)
        if self._evaluation_q is not q:
            self._prepare_evaluation(q)
        key = (actual, bool(jacobian))
        if key in self._camera_cache:
            return self._camera_cache[key]
        block = self.state.bypts[actual]
        def camera(value):
            return np.asarray(block.bundle.camera(native_camera_local(block, value[:self.core_count]), actual), float)
        nominal = camera(q)
        if not jacobian:
            result = (nominal, None, None)
            self._camera_cache[key] = result
            return result
        columns = np.unique(np.r_[block.xyz_slots, block.optics_chart.indices,
            block.optics_chart.mount_indices, block.start + 4 * block.bundle.index[actual] + np.arange(4)]).astype(int)
        inverse_condition = np.r_[[.01] * 3, [1.] * 5, [.001] * 3, [1.] * 4]
        chain = np.empty((15, len(columns)))
        height = positive_pivot_height_columns([block])
        view = block.start + 4 * block.bundle.index[actual]
        try:
            for i, column in enumerate(columns):
                h = (1e-4 if column in block.optics_chart.indices else
                     1e-7 if view <= column < view + 3 else 1e-6)
                full = positive_pivot_coordinate_derivative(camera, q, int(column), h, height, base=nominal)
                if np.any(np.abs(full[15:]) > 1e-12):
                    raise ValueError('Active camera parameter outside original camera15 derivative route')
                chain[:, i] = full[:15] * inverse_condition
        finally:
            camera(q)
        value = (nominal, columns, chain)
        self._camera_cache[key] = value
        return value

    def _evaluate_row(self, q, row, jacobian):
        from z190_source_camera_diagnostic import absolute_value_jac15
        from structured_programme_source_candidate.prepared_finite_curve_world_columns import curve_current_world_columns
        world, world_cols, world_jac = self._world(q, row['model_target'], jacobian)
        camera, camera_cols, camera_chain = self._camera(q, row['actual_pts'], jacobian)
        evaluated = dict(world=world, uv=row['uv'], sigma=row['sigma'],
            point_correspondence=False, curve_observation_kind='measured_transition_location')
        matrix = None
        if jacobian:
            residual, camera_jac = absolute_value_jac15(camera, [evaluated], True)
            native, world_matrix = curve_current_world_columns(camera, world, row['uv'], world_jac)
            if not np.allclose(native / np.repeat(row['sigma'], 2), residual, rtol=0, atol=1e-9):
                raise ValueError('Prepared world columns changed original native finite source residual')
            local = np.c_[world_matrix / np.repeat(row['sigma'], 2)[:, None], camera_jac @ camera_chain]
            columns = np.r_[world_cols, camera_cols]
            rr, cc = np.nonzero(local)
            matrix = csr_matrix((local[rr, cc], (rr, columns[cc])), shape=(len(residual), self.parameter_count))
        else:
            residual = absolute_value_jac15(camera, [evaluated], False)
        if not np.isfinite(residual).all():
            raise ValueError('Strict source left original finite optical chart')
        capture=getattr(self,'_preflight_probe_records',None)
        if capture is not None:
            from .source_direction_probe import capture_source_direction_record
            capture[self._preflight_probe_label]=capture_source_direction_record(
                camera,world,row['uv'],row['sigma'],residual,world_columns=world_cols,
                world_jacobian=world_jac,camera_columns=camera_cols,camera_chain=camera_chain,
                local_camera_jacobian=camera_jac if jacobian else None)
        return residual, matrix

    def iter_blocks(self, q, jacobian=True):
        """Yield existing programme families after stable local CSR stacking.

        Primitive sharing stays inside each family. Six source blocks keep the
        original logging and Gram interfaces from doing per-fragment work; no
        rows-by-global-parameter dense matrix is allocated.
        """
        from z190_streaming_source_normal import SourceRowBlock
        q = self._check_q(q); self._prepare_evaluation(q)
        started = time.perf_counter(); cursor = 0
        family_groups = getattr(self, 'family_groups', None)
        if family_groups is None:
            family_groups = partition_dataset_rows_by_family(self.rows, self.state)
        ordinal = 0
        for family, rows in family_groups.items():
            residuals, matrices, row_ledger = [], [], []
            lo = cursor
            for row in rows:
                residual, matrix = self._evaluate_row(q, row, jacobian)
                residuals.append(residual)
                if jacobian: matrices.append(matrix)
                stop = cursor + len(residual)
                row_ledger.append(dict(strict_group_index=ordinal, actual_pts=row['actual_pts'],
                    scalar_range=[cursor, stop], model_target=row['model_target'],
                    original_source_locators=row['source_locators']))
                cursor = stop; ordinal += 1
            ledger = dict(family='strict_reviewed_finite_dataset:' + family,
                programme_family=family, original_start=lo, original_stop=cursor,
                compact_primitive_groups=len(rows), original_primitive_row_ledger=row_ledger,
                stable_original_programme_family_CSR_concatenation=True,
                source_UV_sigma_FIT_HELD_and_operator_preserved=True)
            residual = np.concatenate(residuals)
            matrix = vstack(matrices, format='csr') if jacobian else None
            # Release primitive matrices before yielding the single family
            # CSR to the original normal accumulator and its replay cache.
            del residuals, matrices
            yield SourceRowBlock(residual, matrix, self.quadratic[lo:cursor], self.scales[lo:cursor], ledger)
        if cursor != self.row_count:
            raise ValueError('Strict complete FIT scalar inventory changed')
        self.last_timing = dict(jacobian=bool(jacobian), seconds=time.perf_counter()-started,
            scalar_rows=cursor, groups=len(self.rows), prepared_materials=len(self._world_cache),
            prepared_cameras=len(self._camera_cache), source_blocks=len(family_groups))

    def preflight(self, q):
        """Actual material-route coverage and complete-column nonlinear FD gate.

        Pixel subsets here only qualify implementation, never fit or accuracy.
        All declared targets are inspected; the executable factor still carries
        every admitted observation. Failure remains explicit before optimization.
        """
        q = self._check_q(q); self._prepare_evaluation(q)
        started = time.perf_counter()
        self.geometry.validate_parameters(q[self.world_indices])
        targets = OrderedDict((_key(row['model_target']), row['model_target'])
            for row in self.rows + self.held)
        material_records = []
        for target in targets.values():
            world = self._world(q, target, False)[0]
            if (world.ndim != 2 or world.shape[1:] != (3,) or not np.isfinite(world).all()
                    or len(world) < (1 if target['kind'] == 'shared_board_pattern_material_points' else 2)):
                raise ValueError('Strict target has no complete finite current material producer')
            material_records.append(dict(model_target=target, knots=len(world), finite_current_world=True))
        representatives = OrderedDict()
        for row in self.rows:
            block = self.state.bypts[row['actual_pts']]
            installation = block.bundle.calibration.installation_id
            representatives.setdefault((row['model_target']['kind'], installation), row)
        checks = []
        try:
            for (kind, installation), original in representatives.items():
                take = np.unique(np.linspace(0, len(original['uv']) - 1,
                    min(16, len(original['uv']))).astype(int))
                row = dict(original, uv=original['uv'][take], sigma=original['sigma'][take])
                self._prepare_evaluation(q)
                self._preflight_probe_records={};self._preflight_probe_label='nominal'
                residual, matrix = self._evaluate_row(q, row, True)
                columns = np.unique(matrix.indices)
                if not len(columns):
                    raise ValueError('Representative current strict source is unobservable in complete chart')
                direction = np.zeros(len(q)); direction[columns] = np.random.default_rng(70213).normal(size=len(columns))
                predicted = matrix @ direction
                # A small physical displacement preserves the current finite
                # association. Step selection never consults measured FD error.
                native = predicted * np.repeat(row['sigma'], 2)
                h = min(1e-6, .001 / max(1., float(np.abs(native).max())))
                plus = q + h * direction; minus = q - h * direction
                self._preflight_probe_label='plus'
                self._prepare_evaluation(plus); rp = self._evaluate_row(plus, row, False)[0]
                self._preflight_probe_label='minus'
                self._prepare_evaluation(minus); rm = self._evaluate_row(minus, row, False)[0]
                measured = (rp - rm) / (2 * h)
                from .source_direction_probe import qualify_source_direction_records
                qualification=qualify_source_direction_records(self._preflight_probe_records,matrix,direction,h,
                    original_group_point_indices=take)
                maximum=qualification['original_nominal_segment_and_clamp_fixed_FD_maximum_error']
                relative=qualification['original_nominal_segment_and_clamp_fixed_FD_relative_error']
                if not qualification['passed']:
                    from .source_direction_probe import save_source_direction_probe
                    probe=save_source_direction_probe(q,direction,h,matrix,self._preflight_probe_records,
                        dict(kind=kind,installation=installation,actual_pts=row['actual_pts'],
                            model_target=row['model_target'],selected_original_group_point_indices=take.tolist(),
                            source_locators=row['source_locators'],maximum=maximum,relative=relative,
                            qualification=qualification))
                    raise ValueError('Strict current complete source derivative failed: ' +
                        _key(dict(kind=kind, installation=installation,relative=relative,maximum=maximum,probe=probe)))
                checks.append(dict(kind=kind, installation=installation, actual_pts=row['actual_pts'],
                    points=len(take), active_global_columns=columns.tolist(), symmetric_step=h,
                    maximum_normalized_derivative_error=maximum, relative_maximum_derivative_error=relative,
                    original_branch_and_unrestricted_stable_trials_match_complete_J=True,
                    directional_branch_qualification=qualification))
        finally:
            self._preflight_probe_records=None
            self._prepare_evaluation(q)
        return dict(schema='rinkspace.strict-dataset-actual-complete-chart-preflight.v1',
            seconds=time.perf_counter()-started, full_material_target_records=material_records,
            actual_full_chart_directional_checks=checks, fit_pixels_removed=0,
            partial_interface_validation_is_accuracy_acceptance=False,
            original_fit_points=sum(len(row['uv']) for row in self.rows), passed=True)

    def review(self, q):
        """HELD raw native distances retain missing sigma and never enter cost."""
        from z190_source_camera_diagnostic import absolute_value_jac15
        q = self._check_q(q); self._prepare_evaluation(q); result = []
        for row in self.held:
            world = self._world(q, row['model_target'], False)[0]
            camera = self._camera(q, row['actual_pts'], False)[0]
            # Unit denominator merely exposes raw native-pixel distances in
            # this review. It does not fill the stored missing source support.
            raw = absolute_value_jac15(camera, [dict(world=world, uv=row['uv'], sigma=np.ones(len(row['uv'])),
                point_correspondence=row['point_correspondence'])], False)
            distance = np.linalg.norm(raw.reshape(-1, 2), axis=1)
            result.append(dict(actual_pts=row['actual_pts'], model_target=row['model_target'],
                source_locators=row['source_locators'], points=len(distance),
                raw_native_distance_px=distance.tolist(), missing_original_sigma_points=int(np.isnan(row['sigma']).sum()),
                mean_px=float(distance.mean()), max_px=float(distance.max()), held_used_in_fit=False))
        return dict(schema='rinkspace.strict-dataset-original-HELD-review.v1', rows=result,
            points=sum(r['points'] for r in result), held_used_in_fit=False, accuracy_accepted=False)
