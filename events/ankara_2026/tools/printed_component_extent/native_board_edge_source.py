"""Additional source-qualified finite print and far-board edges on one graph.

The observation operator is the original native finite-curve/normal-interval
operator. Source scanlines do not become fixed-Z or tangential point matches.
"""
from collections import OrderedDict
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import numpy as np
from .native_camera_local import native_camera_local
from scipy.sparse import csr_matrix
from z190_streaming_source_normal import SourceRowBlock
from z190_source_camera_diagnostic import absolute_value_jac15
from .native_ice_curve_source import (SOURCE_SHA, _sha, _native_frame_reference,
    _current_source_parameter_layout)
from .fixed_pivot_positive_domain_fd import (
    positive_pivot_height_columns, positive_pivot_coordinate_derivative,
)
from .paired_programme_chart import CORE_COUNT, WORLD_INDICES

LEGACY_AD5_INTERNAL_SOURCE_SHA = 'fc3b4eb3de768d040ef7820a8b95c422408a17f4f343b80bd72a2e16f815530a'


class NativeBoardEdgeFactor:
    def __init__(self, state, packet_path, parameter_count):
        self.state, self.parameter_count = state, int(parameter_count)
        self.packet_path = Path(packet_path).resolve()
        packet = json.loads(self.packet_path.read_text(encoding='utf8'))
        if (packet.get('schema') != 'rinkspace.additional-native-board-boundary-source.v1'
                or packet.get('source_video_sha256') != SOURCE_SHA
                or packet.get('time_base') != '1/15360'
                or packet.get('status') != 'explicit_new_fit_target_authorized'):
            raise ValueError('Explicit admitted same-source native board-edge packet required')
        self._physical_course_scope_overrides={};self.physical_course_scope_authority=None
        scope_path=getattr(state,'current_user_physical_course_scope_source',None)
        if scope_path is not None:
            scope_path=Path(scope_path).resolve();scope=json.loads(scope_path.read_text(encoding='utf8'))
            if (scope.get('schema')!='rinkspace.direct-human-existing-course-scope-authority.v1'
                or scope.get('record_count')!=2 or len(scope.get('records',[]))!=2
                or scope.get('all_original_source_UV_sigma_band_127_primary_5760_guard_vertices_unchanged')is not True
                or scope.get('no_lower_left_or_shortside_admitted')is not True
                or scope.get('new_geometry_or_camera_parameter_count')!=0):
                raise ValueError('Exact two-source existing connected course scope authority required')
            for record in scope['records']:
                ref=record['original_packet'];original=packet['rows'][record['original_row_index']]
                parent_path=Path(ref['path']).resolve()
                if _sha(parent_path)!=ref['sha256']:
                    raise ValueError('Original physical course scope packet bytes changed')
                parent_packet=json.loads(parent_path.read_text(encoding='utf8'))
                if original!=parent_packet['rows'][record['original_row_index']]:
                    raise ValueError('Appended packet changed the exact historical physical course source row')
                raw=json.dumps(original,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
                if (original['id']!=record['id'] or original['curve']!=record['original_curve']
                    or hashlib.sha256(raw).hexdigest()!=record['source_row_canonical_json_sha256']
                    or original['pts']!=record['pts'] or original['actual_pts']!=record['actual_pts']
                    or record['effective_connected_course_primitives']!=['lower_long_side','lower_right']
                    or record['existing_course_original_vertex_indices_inclusive']!=[336,456]
                    or record['human_UV_normals_support_or_history_changed']is not False
                    or record['physical_corner_or_paint_endpoint_observation_added']is not False
                    or record['unbounded_line_or_other_course_facet_added']is not False):
                    raise ValueError('Scope correction must bind exact original human source row and connected existing course')
                for key in ('human_authority_reference','source_annotation_registration','source_annotation_attachment','actual_solver_source_geometry_diagnostic'):
                    reference=record[key]
                    if _sha(reference['path'])!=reference['sha256']:raise ValueError('Scope source/RGB/diagnostic evidence changed')
                if _sha(record['source_raw_frame'])!=record['source_file_sha256']:raise ValueError('Scope original source RGB changed')
                key=json.dumps(original['curve'],sort_keys=True,separators=(',',':'))
                if key in self._physical_course_scope_overrides:raise ValueError('Distinct TOP/contact scope identities required')
                self._physical_course_scope_overrides[key]=record['effective_connected_course_primitives']
            self.physical_course_scope_authority=dict(path=str(scope_path),sha256=_sha(scope_path),records=scope['records'])
        self.geometry = state.geometry.current
        self.core_count, self.world_indices = _current_source_parameter_layout(state, self.parameter_count)
        self.world_parameter_count = len(self.world_indices)
        self.rows, self.held = [], []
        self._printed_contour_materials = {}
        legacy_ref = packet.get('legacy_internal_scope_alias_authority',
                                packet.get('source_measurement_candidate'))
        legacy_rows = {}
        if isinstance(legacy_ref, dict) and legacy_ref.get('sha256') == LEGACY_AD5_INTERNAL_SOURCE_SHA:
            authority = Path(legacy_ref['path']).resolve()
            if _sha(authority) != LEGACY_AD5_INTERNAL_SOURCE_SHA:
                raise ValueError('Original Ad5 internal-graphic source authority changed')
            original = json.loads(authority.read_text(encoding='utf8'))
            legacy_rows = {r['id']: r for r in original['rows']
                if r['curve'].get('advertisement_number') == 5
                and r.get('geometry_role') == 'retained_internal_blue_white_green_MEIWA_graphic_boundary'}
        scope_aliases = []
        ids = set()
        for record in packet['rows']:
            row = deepcopy(record)
            if row['id'] in ids:
                raise ValueError('Additional board source IDs must be unique')
            ids.add(row['id'])
            curve = row['curve']
            number = curve.get('advertisement_number')
            edge_key = {'left_vertical': 'left_vertical_xyz',
                        'right_vertical': 'right_vertical_xyz',
                        'top_edge': 'top_edge_xyz',
                        'bottom_edge': 'bottom_edge_xyz'}.get(curve.get('edge'))
            far_board = (curve.get('kind') == 'board_top'
                         and curve.get('source_structure') == 'far_visible_board')
            far_contact = (curve.get('kind') == 'ice_board_contact'
                and curve.get('source_structure') == 'far_visible_board'
                and curve.get('physical_boundary') == 'bottom_edge_of_yellow_kick_strip_at_ice_contact')
            physical_facet = (curve.get('kind') in ('board_top', 'ice_board_contact')
                and curve.get('source_structure') == 'physical_course_facet'
                and (curve['kind'] != 'ice_board_contact'
                     or curve.get('physical_boundary') == 'bottom_edge_of_yellow_kick_strip_at_ice_contact'))
            printed_contour = curve.get('kind') == 'printed_component_contour'
            if printed_contour:
                self._printed_contour_material(curve)
                if (row.get('curve_observation_kind') != 'horizontal_extent'
                        or row.get('extent_side') not in ('left', 'right')
                        or len(row.get('uv', [])) != 1
                        or not np.array_equal(np.asarray(row.get('normal'), float), [[1., 0.]])
                        or any(key in row or key in curve for key in
                               ('world', 'world_xyz', 'world_xyz_scene_units',
                                'material_coordinates_normalized'))):
                    raise ValueError('One source-only x extent of the exact frozen printed component required')
            scope = curve.get('extent_scope')
            actual_scope = (self.geometry._ad(number).get('extent_scope')
                            if isinstance(number, int) and not isinstance(number, bool)
                            and edge_key is not None and (number, edge_key) in self.geometry._materials
                            else None)
            legacy_internal = (number == 5 and scope == 'external_border'
                and actual_scope == 'internal_graphic_only'
                and record.get('geometry_role') == 'retained_internal_blue_white_green_MEIWA_graphic_boundary'
                and record == legacy_rows.get(record['id']))
            print_edge = (curve.get('kind') == 'advertisement_boundary'
                    and scope in ('external_border', 'internal_graphic_only')
                    and isinstance(number, int) and not isinstance(number, bool)
                    and edge_key is not None and (number, edge_key) in self.geometry._materials
                    and ((actual_scope == scope
                          and (scope != 'internal_graphic_only'
                               or row.get('geometry_role') == 'internal_graphic_only_boundary'))
                         or legacy_internal))
            if (not (far_board or far_contact or physical_facet or print_edge or printed_contour)
                    or row.get('point_correspondence') is not False
                    or (row.get('curve_observation_kind') != 'normal_interval' and not printed_contour)):
                raise ValueError('Exact modeled print scope or source-identified far board top/ice contact required')
            row['material_edge_key'] = edge_key
            if print_edge:
                row['actual_modeled_extent_scope'] = actual_scope
            if legacy_internal:
                scope_aliases.append(dict(id=row['id'], authority=legacy_ref,
                    stored_scope=scope, actual_modeled_scope=actual_scope,
                    original_row_all_metadata_and_UV_sigma_support_exact=True,
                    existing_internal_material_producer_unchanged=True,
                    scope_alias_changes_observation_operator=False))
            native = int(row['pts'])
            if native not in state.original.aliases:
                raise ValueError('Source PTS has no original authenticated camera alias')
            actual = int(state.original.aliases[native])
            if ('actual_pts' in row and int(row['actual_pts']) != actual) or actual not in state.bypts:
                raise ValueError('Additional board exposure alias disagrees with original source')
            path, expected = _native_frame_reference(row)
            if _sha(path) != expected:
                raise ValueError('Additional source raw frame bytes changed')
            uv = np.asarray(row['uv'], float)
            sigma = np.broadcast_to(np.asarray(row['sigma_px'], float), (len(uv),)).copy()
            normal, intervals = np.asarray(row['normal'], float), np.asarray(row['normal_interval_px'], float)
            # A single independently read pixel is a valid query against the
            # current finite material edge. The edge geometry comes from the
            # model, so a second source pixel is not needed to define a line.
            if (uv.ndim != 2 or uv.shape[1:] != (2,) or len(uv) < 1
                    or sigma.shape != (len(uv),) or np.any(sigma <= 0)
                    or normal.shape != uv.shape or intervals.shape != uv.shape
                    or not all(np.isfinite(a).all() for a in (uv, sigma, normal, intervals))
                    or np.any(intervals[:, 0] > intervals[:, 1])
                    or np.any(uv < 0) or np.any(uv >= [1920., 1080.])):
                raise ValueError('Finite unchanged source UV, support and positive raw sigma required')
            row.update(actual_pts=actual, uv=uv, sigma=sigma, normal=normal,
                       normal_interval=intervals)
            if far_board or far_contact or physical_facet:
                # Only native float32 unit-vector roundoff is normalized in
                # the derived operator; stored UV/support/normal stay intact.
                from rinkspace.vision_camera_initialization import unit_source_normals
                row['normal'], row['source_normal_unit_roundoff'] = unit_source_normals(normal, len(uv))
            if row['split'] == 'fit':
                self.rows.append(row)
            elif row['split'] == 'holdout':
                self.held.append(row)
            else:
                raise ValueError('Explicit new FIT or HELD role required')
        if not self.rows:
            raise ValueError('At least one explicitly admitted source edge required')
        self.blocks = {r['actual_pts']: state.bypts[r['actual_pts']] for r in self.rows + self.held}
        self.camera_steps, camera_cols = {}, []
        for actual, block in self.blocks.items():
            view = block.start + 4 * block.bundle.index[actual]
            cols = np.r_[block.xyz_slots, block.optics_chart.indices,
                         block.optics_chart.mount_indices, view + np.arange(4)].astype(int)
            camera_cols.extend(cols)
            for col in cols: self.camera_steps[int(col)] = 1e-6
            for col in block.optics_chart.indices: self.camera_steps[int(col)] = 1e-4
            for col in range(view, view + 3): self.camera_steps[col] = 1e-7
        self.camera_columns = np.unique(camera_cols)
        if np.intersect1d(self.camera_columns, self.world_indices).size:
            raise ValueError('Camera and material columns overlap')
        all_rows = self.rows + self.held
        numbers = sorted({r['curve']['advertisement_number'] for r in all_rows
                          if r['curve']['kind'] in ('advertisement_boundary', 'printed_component_contour')})
        world_columns = [
            self.world_indices[self.geometry.material_parameter_indices(number)]
            for number in numbers]
        far_board_ids = [r['id'] for r in all_rows if r['curve']['kind'] == 'board_top'
                         and r['curve'].get('source_structure') == 'far_visible_board']
        far_contact_ids = [r['id'] for r in all_rows if r['curve']['kind'] == 'ice_board_contact'
                           and r['curve'].get('source_structure') == 'far_visible_board']
        if any(r['curve']['kind'] in ('board_top', 'ice_board_contact') for r in all_rows):
            # The reviewed current scene producer owns course and top-height
            # dependence. Keep the complete logical world, no frozen WORLD.
            world_columns.append(self.world_indices)
        self.active_world = np.unique(np.concatenate(world_columns))
        self.active = np.unique(np.r_[self.camera_columns, self.active_world]).astype(int)
        self.scales = np.concatenate([np.repeat(r['sigma'], 2) for r in self.rows])
        self.quadratic = np.ones(len(self.scales), bool)
        self.row_count, self._cache = len(self.scales), OrderedDict()
        self.contract = dict(schema='rinkspace.additional-real-native-board-edge-factor.v4',
            packet_path=str(self.packet_path), packet_sha256=_sha(self.packet_path),
            original_operator='z190_source_camera_diagnostic.absolute_value_jac15 with SourceNormalIntervals',
            current_material='Current named66 actual modeled finite print edges from original material producer',
            advertisement_numbers=numbers,
            evaluation_core_parameter_count=self.core_count,
            logical_world_parameter_count=self.world_parameter_count,
            logical_world_parameter_indices=self.world_indices.tolist(),
            far_visible_board_top_ids=far_board_ids,
            far_visible_ice_board_contact_ids=far_contact_ids,
            physical_course_facet_ids={r['id']: r['curve']['course_primitive'] for r in all_rows
                                      if r['curve'].get('source_structure') == 'physical_course_facet'},
            physical_course_facet_operator='Complete continuous current perimeter; source direction names are hints, never a face or tangent domain',
            course_direction_labels_do_not_restrict_material_region=True,
            print_extent_scopes={r['id']: r['actual_modeled_extent_scope'] for r in all_rows
                                if r['curve']['kind'] == 'advertisement_boundary'},
            historical_scope_aliases=scope_aliases,
            physical_course_scope_authority=self.physical_course_scope_authority,
            prior_connected_scope_retained_as_history_not_runtime_region=True,
            print_scope_matches_actual_modeled_material=True,
            far_board_operator='nearest finite projected foot on complete current perimeter, with unchanged SourceNormalIntervals',
            nearest_finite_complete_course_foot=True,
            source_y_or_direction_label_does_not_bound_a_course_region=True,
            fit_ids=[r['id'] for r in self.rows], held_ids=[r['id'] for r in self.held],
            scalar_rows=self.row_count, source_UV_sigma_normal_intervals_preserved=True,
            raw_sigma_used_without_additional_sqrt_N=True, source_y_is_context_not_fixed_Z=True,
            optical_support_J='FD unchanged raw finite-curve geometry; exact existing SourceNormalIntervals nominal derivative',
            optical_support_coordinate_FD_cannot_average_across_support_boundary=True,
            original27_Ad7_FIT_HELD_and_all_original_source_unchanged=True,
            new_target_objective_explicit=True, geometry_or_camera_DOF_added=False,
            accuracy_accepted=False)

    def _printed_contour_material(self, curve):
        """Resolve an existing certified source component, never supplied XYZ/UV."""
        key = json.dumps(curve, sort_keys=True, separators=(',', ':'))
        if key in self._printed_contour_materials:
            return self._printed_contour_materials[key]
        reference = curve.get('frozen_print_source_packet')
        if (not isinstance(reference, dict)
                or reference.get('sha256') != 'dc5af1b4b485f763b8b1c95a1f19af64d097c9542a8f7df3ba6685d6c4fae81c'):
            raise ValueError('Authenticated admitted native frozen-print parent required')
        path = Path(reference['path']).resolve()
        if _sha(path) != reference['sha256']:
            raise ValueError('Frozen printed-component source packet bytes changed')
        source = json.loads(path.read_text(encoding='utf8'))
        if (source.get('schema') != 'rinkspace.native-static-board-tracks.v1'
                or source.get('source_sha256') != SOURCE_SHA
                or source.get('time_base') != '1/15360'
                or source.get('model_projection_used_to_generate_or_select_uv') is not False):
            raise ValueError('Original native frozen print authority required')
        number, identifier = curve.get('advertisement_number'), curve.get('template_id')
        if (not isinstance(number, int) or isinstance(number, bool)
                or getattr(self.geometry, 'source_template_instance', {}).get(number) != identifier):
            raise ValueError('Actual printed instance must own the named existing frozen template')
        records = [r for r in source.get('frozen_source_print_templates', []) if r['id'] == identifier]
        if len(records) != 1:
            raise ValueError('One authentic frozen template identity required')
        record = records[0]
        owner = self.geometry.source_print_templates[identifier]
        for field in ('initialization', 'external_contours', 'canonical_advertisement_number',
                      'counterpart_advertisement_number', 'shared_dimensions_authority'):
            if record[field] != owner[field]:
                raise ValueError('Frozen contour and actual current material producer differ: ' + field)
        if number not in (record['canonical_advertisement_number'], record['counterpart_advertisement_number']):
            raise ValueError('Printed component instance differs from its certified pair')
        authority = record['shared_dimensions_authority']
        if _sha(authority['path']) != authority['sha256']:
            raise ValueError('Printed paired-dimensions authority changed')
        contours = [c for c in record['external_contours'] if c['id'] == curve.get('component_id')]
        if len(contours) != 1:
            raise ValueError('One exact source-identified printed component required')
        contour = contours[0]
        material = np.asarray(contour['initialization']['material_coordinates_normalized'], float)
        if (contour.get('role') != 'actual_printed_component_external_contour'
                or contour.get('closed') is not True
                or contour['initialization'].get('derived_not_source_measurement') is not True
                or material.shape != (len(contour['uv']), 2) or len(material) < 4
                or not np.isfinite(material).all()):
            raise ValueError('Complete finite existing normalized external print contour required')
        material = material.copy(); material.flags.writeable = False
        self._printed_contour_materials[key] = material
        return material

    def _cameras(self, q):
        core = np.asarray(q, float)[:self.core_count]
        return {actual: block.bundle.camera(native_camera_local(block,core), actual)
                for actual, block in self.blocks.items()}

    def effective_curve(self,curve):
        """Direction labels are context; the immutable physical identity stays."""
        if curve.get('kind')in('board_top','ice_board_contact'):
            return dict(curve,continuous_course_perimeter=True)
        key=json.dumps(curve,sort_keys=True,separators=(',',':'))
        primitives=self._physical_course_scope_overrides.get(key)
        return curve if primitives is None else dict(curve,connected_course_primitives=primitives)

    def _world(self, q, curve=None):
        if curve is not None:curve=self.effective_curve(curve)
        p = np.asarray(q, float)[self.world_indices]
        if curve is not None and curve.get('kind') == 'printed_component_contour':
            material = self._printed_contour_material(curve)
            return self.geometry.source_template_material(
                curve['advertisement_number'], *material.T, p)[0]
        if curve is not None and curve.get('kind') in ('board_top', 'ice_board_contact'):
            structure = curve.get('source_structure')
            if structure not in ('far_visible_board', 'physical_course_facet'):
                raise ValueError('An explicit original far-board or physical facet source structure is required')
            # Both exported physical board edges are this same current course.
            # Preserve the full physical guard but do not rebuild every print
            # scene for every camera/world finite-difference evaluation.
            self.geometry.validate_parameters(p)
            course, course_q, _ = self.geometry.print_course_binding(p)
            level = 1 if curve['kind'] == 'board_top' else 0
            perimeter = course.sample_perimeter(course_q, with_jacobian=False)
            world = np.asarray(perimeter[level], float)
            if not np.array_equal(world[0],world[-1]):world=np.vstack((world,world[0]))
            if world.ndim != 2 or world.shape[1] != 3 or len(world) < 3 or not np.isfinite(world).all():
                raise ValueError('Current complete finite physical board top/ice contact required')
            return world
        if curve is None:
            number, edge_key = 7, 'left_vertical_xyz'
        else:
            number = curve['advertisement_number']
            edge_key = {'left_vertical': 'left_vertical_xyz',
                        'right_vertical': 'right_vertical_xyz',
                        'top_edge': 'top_edge_xyz',
                        'bottom_edge': 'bottom_edge_xyz'}[curve['edge']]
        # Exact material coordinates of the actual exported finite edge,
        # never old WORLD, a missing edge, or inverse-projected source UV.
        u, v = self.geometry._materials[number, edge_key]
        return self.geometry.material(number, u, v, p)[0]

    def _source_rows(self, q, row, camera):
        derived = dict(row, world=self._world(q, row['curve']), sigma=row['sigma'])
        return [derived]

    def _residual(self, q, rows, *, cameras=None, apply_optical_support=True):
        cameras = self._cameras(q) if cameras is None else cameras
        values = []
        for row in rows:
            camera = cameras[row['actual_pts']]
            derived = self._source_rows(q, row, camera)
            if not apply_optical_support:
                derived = [{k: v for k, v in part.items() if k != 'normal_interval'}
                           for part in derived]
            values.append(absolute_value_jac15(camera, derived, False))
        result = np.concatenate(values) if values else np.empty(0)
        if not np.isfinite(result).all():
            raise ValueError('Additional board source has invalid projection')
        return result

    def _support_branches(self, q):
        """Read the actual fixed support branch without changing the objective."""
        from source_edge_intervals import SourceNormalIntervals
        raw = self._residual(q, self.rows, apply_optical_support=False)
        branches, signed, bounds, points = [], [], [], []
        offset = 0
        for row in self.rows:
            stop = offset + 2*len(row['uv'])
            if 'normal_interval' in row:
                support = SourceNormalIntervals(row['normal'], row['normal_interval'],
                                                np.ones(len(row['uv']), bool))
                pixels = (raw[offset:stop]*np.repeat(row['sigma'], 2)).reshape(-1, 2)
                normal_value = np.einsum('ni,ni->n', pixels, support.normals)
                # SourceNormalIntervals uses a strict interior and the exterior
                # derivative at either boundary. Keep those exact comparisons.
                branch = np.where(normal_value <= support.bounds[:, 0], -1,
                                  np.where(normal_value >= support.bounds[:, 1], 1, 0))
                branches.extend(branch.tolist())
                signed.extend(normal_value.tolist())
                bounds.extend(support.bounds.tolist())
                points.extend((row['id'], i) for i in range(len(row['uv'])))
            offset = stop
        return np.asarray(branches, dtype=np.int8), np.asarray(signed), bounds, points

    def local_support_directional_step(self, q, direction, initial_h):
        """Select a validation step only when the finite step crosses support.

        No derivative error is consulted. A same-branch step is returned for
        the existing FD gate to judge with its unchanged direction and tolerance.
        """
        q, direction = np.asarray(q, float), np.asarray(direction, float)
        if (q.shape != (self.parameter_count,) or direction.shape != q.shape
                or not np.isfinite(q).all() or not np.isfinite(direction).all()
                or not np.isfinite(initial_h) or initial_h <= 0):
            raise ValueError('Finite complete state, direction and positive FD step required')
        nominal, nominal_signed, bounds, points = self._support_branches(q)
        h, trials, found = float(initial_h), [], False
        labels = {-1: 'below_or_on_lower', 0: 'strictly_inside', 1: 'above_or_on_upper'}
        try:
            while True:
                plus, minus = q+h*direction, q-h*direction
                if not np.any(plus != q) or not np.any(minus != q):
                    trials.append(dict(h=h, perturbation_representable=False))
                    break
                positive, positive_signed, _, _ = self._support_branches(plus)
                negative, negative_signed, _, _ = self._support_branches(minus)
                changed = np.flatnonzero((positive != nominal) | (negative != nominal))
                changes = [dict(id=points[i][0], point_index=points[i][1],
                    normal_interval_px=bounds[i], nominal_signed_px=float(nominal_signed[i]),
                    positive_signed_px=float(positive_signed[i]), negative_signed_px=float(negative_signed[i]),
                    nominal_branch=labels[int(nominal[i])], positive_branch=labels[int(positive[i])],
                    negative_branch=labels[int(negative[i])]) for i in changed]
                trials.append(dict(h=h, perturbation_representable=True,
                                   changed_support_points=len(changed), changes=changes))
                if not len(changed):
                    found = True
                    break
                h *= .1
        finally:
            self._cameras(q)
        return h, dict(initial_h=float(initial_h), final_h=h, same_support_branch_step_found=found,
            step_selection='shrink only when actual plus/minus SourceNormalIntervals branch differs from nominal',
            direction_and_nominal_J_unchanged=True, source_support_and_objective_unchanged=True, trials=trials)

    def block(self, q, jacobian=True, *, original_start=0):
        q = np.asarray(q, float)
        if q.shape != (self.parameter_count,) or not np.isfinite(q).all():
            raise ValueError('Same full physical named66 state required')
        key = q[self.active].tobytes()
        cached = self._cache.get(key)
        if cached is not None and (not jacobian or cached[1] is not None):
            residual, matrix = cached
        else:
            cameras = self._cameras(q)
            residual, matrix = self._residual(q, self.rows, cameras=cameras), None
            if jacobian:
                compact = np.empty((len(residual), len(self.active)))
                raw = self._residual(q, self.rows, cameras=cameras, apply_optical_support=False)
                height_columns = positive_pivot_height_columns(self.blocks.values())
                try:
                    for i, col in enumerate(self.active):
                        h = self.camera_steps.get(int(col), 1e-6)
                        same_camera = cameras if col in self.active_world else None
                        compact[:, i] = positive_pivot_coordinate_derivative(
                            lambda value: self._residual(value, self.rows, cameras=same_camera,
                                                         apply_optical_support=False),
                            q, col, h, height_columns, base=raw)
                finally:
                    self._cameras(q)
                # The source's support operator is piecewise linear. A legal
                # geometry FD step can cross a support boundary even while
                # the unchanged nominal observation lies strictly inside it.
                # Differentiate the raw pixel geometry first, then apply the
                # existing exact nominal support derivative. Raw sigma is
                # per point, so its normalization commutes with this mapping.
                from source_edge_intervals import SourceNormalIntervals
                offset = 0
                for row in self.rows:
                    stop = offset + 2*len(row['uv'])
                    if 'normal_interval' in row:
                        support = SourceNormalIntervals(row['normal'], row['normal_interval'],
                                                        np.ones(len(row['uv']), bool))
                        physical_raw = raw[offset:stop]*np.repeat(row['sigma'], 2)
                        _, mapped = support.apply(physical_raw, [compact[offset:stop]])
                        compact[offset:stop] = mapped[0]
                    offset = stop
                rr, cc = np.nonzero(compact)
                matrix = csr_matrix((compact[rr, cc], (rr, self.active[cc])),
                                    shape=(len(residual), self.parameter_count))
            self._cache[key] = (residual.copy(), matrix)
            while len(self._cache) > 4: self._cache.popitem(last=False)
        ledger = dict(family='additional_real_native_board_edges', original_start=int(original_start),
                      original_stop=int(original_start)+self.row_count,
                      packet_sha256=self.contract['packet_sha256'],
                      fit_observation_ids=self.contract['fit_ids'], held_observation_ids=self.contract['held_ids'],
                      native_pts=[r['pts'] for r in self.rows], actual_pts=[r['actual_pts'] for r in self.rows],
                      source_UV_sigma_support_roles_unchanged=True, fixed_Z_matches_added=False)
        return SourceRowBlock(residual, matrix, self.quadratic, self.scales, ledger)

    def review(self, q):
        cameras = self._cameras(q)
        results = []
        from z190_source_camera_diagnostic import source_errors
        for row in self.rows + self.held:
            camera = cameras[row['actual_pts']]
            derived = self._source_rows(q, row, camera)
            residual = absolute_value_jac15(camera, derived, False)
            d = np.linalg.norm((residual * np.repeat(row['sigma'], 2)).reshape(-1, 2), axis=1)
            if (row['curve']['kind'] in ('board_top', 'ice_board_contact')
                    and row['curve'].get('source_structure') == 'far_visible_board'):
                raw_rows = [{k: v for k, v in part.items() if k != 'normal_interval'} for part in derived]
                raw = absolute_value_jac15(camera, raw_rows, False)*np.repeat(row['sigma'], 2)
                distances = np.linalg.norm(raw.reshape(-1, 2), axis=1)
                display = dict(pts=row['pts'], id=row['id'], split=row['split'], points=len(row['uv']),
                    observation_kind='source_identified_continuous_board_curve_normal_interval',
                    physical_board_edge=row['curve']['kind'],
                    median_px=float(np.median(distances)), p95_px=float(np.quantile(distances, .95)),
                    maximum_px=float(distances.max()),
                    outside_source_support_median_px=float(np.median(d)),
                    source_y_not_used_to_select_perimeter_branch=True,
                    disjoint_central_forward_components_not_bridged=True)
            else:
                display = source_errors(camera, derived)[0]
            results.append(dict(id=row['id'], pts=row['pts'], actual_pts=row['actual_pts'], split=row['split'],
                source_uv_count=len(row['uv']), mean_outside_source_support_px=float(d.mean()),
                max_outside_source_support_px=float(d.max()), raw_source_pixel_error=display))
        return dict(contract=self.contract, rows=results, accuracy_accepted=False)


class SourceWithAdditionalNativeBoardEdges:
    def __init__(self, base, factor):
        self.base, self.native_board_factor = base, factor
        self.count, self.updater = int(base.count), base.updater
        self.quadratic = np.r_[base.quadratic, factor.quadratic]
        self.scales = np.r_[base.scales, factor.scales]
        self.contract = dict(schema='rinkspace.complete-source-plus-real-native-board-edges.v1',
            original_source_contract=deepcopy(base.contract), additional_board_contract=deepcopy(factor.contract),
            original_source_rows=len(base.scales), additional_scalar_rows=factor.row_count,
            total_rows=len(self.scales), original_blocks_profiles_and_geometry_DOF_preserved=True,
            accuracy_accepted=False)

    def __getattr__(self, name):
        base = self.__dict__.get('base')
        if base is None: raise AttributeError(name)
        return getattr(base, name)

    def guard(self, q):
        self.base.guard(q)
        self.native_board_factor.block(q, False)

    def blocks(self, q, jacobian=True):
        yield from self.base.blocks(q, jacobian)
        yield self.native_board_factor.block(q, jacobian, original_start=len(self.base.scales))


def wrap_additional_native_board_edges(base, state, packet_path):
    return SourceWithAdditionalNativeBoardEdges(base, NativeBoardEdgeFactor(state, packet_path, base.count))
