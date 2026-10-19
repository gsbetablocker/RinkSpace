"""Shared print materials with separately estimated physical placements.

This target adds only Ad25 station/Z placement. The shared q9 material shape,
circle radius, original source operators, UV, sigma and complete old chart
remain intact. The geometry provider accepts an Ad26 offset too, but a source
without an Ad26 observation keeps that unestimated initializer at zero rather
than adding unsupported solver columns.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import MethodType
import hashlib
import json
import numpy as np
from scipy.sparse import csr_matrix, hstack

from z190_streaming_source_normal import SourceRowBlock
from z190_shared_board_pattern_source import _q9
from shared_board_course_candidate.course_pattern_adapter import _factor_geometry
from shared_board_course_candidate.course_pattern_geometry import CourseBoardPatternGeometry
from .paired_programme_chart import CORE_COUNT, WORLD_INDICES
from .native_board_circle_source import PrefixProfileUpdate

PLACEMENT_BINDING = 'same_template_independent_station_z.v1'
PARAMETER_NAMES = ('Ad25_station_offset_from_Ad12_template_m',
                   'Ad25_z_offset_from_Ad12_template_m')
AD26_PARAMETER_NAMES = ('Ad26_station_offset_from_Ad13_template_m',
                        'Ad26_z_offset_from_Ad13_template_m')
AUTHORITY_SHA256 = '89f3dcf0580148e5be2eeeec9033c86ac73a12b2d7eb07ae86a0eb82b15020a0'


def _offsets(value):
    result = np.asarray(value, float)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ValueError('Finite Ad25 and Ad26 station/Z placement offsets required')
    return result.copy()


class IndependentlyPlacedCoursePatterns(CourseBoardPatternGeometry):
    """Both instances use the same intrinsic station/Z printed material."""
    def __init__(self, original, course, q_course, placement_offsets):
        super().__init__(original, course, q_course)
        self.independent_pair_station_z_offsets_m = _offsets(placement_offsets)

    def instance_q9(self, q9, ad_id):
        q = _q9(q9).copy()
        if ad_id == 25:
            q[3:].reshape(3, 2)[:] += self.independent_pair_station_z_offsets_m[:2]
        elif ad_id == 26:
            q[:2] += self.independent_pair_station_z_offsets_m[2:]
        elif ad_id not in (12, 13):
            raise ValueError('Only original Ad12/25 and Ad13/26 print materials supported')
        return q

    def guard(self, q9):
        # Guard both complete finite instances. No size, pose or equality prior
        # is introduced; the old material guard retains its original meaning.
        super().guard(q9)
        super().guard(self.instance_q9(q9, 25))
        super().guard(self.instance_q9(q9, 26))
        return True

    def circle(self, q9, theta, *, ad_id=13):
        return super().circle(self.instance_q9(q9, ad_id), theta, ad_id=ad_id)

    def shield(self, q9, *, ad_id=12):
        return super().shield(self.instance_q9(q9, ad_id), ad_id=ad_id)

    def circle_course_jacobian(self, q9, theta, *, ad_id=13):
        return super().circle_course_jacobian(self.instance_q9(q9, ad_id), theta, ad_id=ad_id)

    def shield_course_jacobian(self, q9, *, ad_id=12):
        return super().shield_course_jacobian(self.instance_q9(q9, ad_id), ad_id=ad_id)

    def descriptor(self, q9, *, engineering_circle_samples=256):
        result = super().descriptor(q9, engineering_circle_samples=engineering_circle_samples)
        q = _q9(q9)
        result.update(placement_binding=PLACEMENT_BINDING,
            instance_station_z_offsets_m={
                '25': self.independent_pair_station_z_offsets_m[:2].tolist(),
                '26': self.independent_pair_station_z_offsets_m[2:].tolist()},
            instance_material_coordinates_m={
                'shield12': q[3:].reshape(3, 2).tolist(),
                'shield25': self.instance_q9(q, 25)[3:].reshape(3, 2).tolist(),
                'circle13': q[:3].tolist(),
                'circle26': self.instance_q9(q, 26)[:3].tolist()},
            print_shape_shared=True, print_radius_shared=True,
            paired_instance_positions_forced_halfturn=False,
            placement_initialization_is_not_source_measurement=True)
        result.pop('mirror_rule', None)
        return result


@contextmanager
def _bind_independent_current_geometry(owner, core):
    q = np.asarray(core, float)
    if q.shape != (owner.state.structured_programme_coordinate_chart.core_count,) or not np.isfinite(q).all():
        raise ValueError('Complete named66 current pattern core required')
    old = owner.geometry, owner.factor, owner.external_factor
    course = owner.course_world_geometry.source_course.course
    cq = owner.course_world_geometry.current.current_course6(q[owner.state.geometry_parameter_indices])
    geometry = IndependentlyPlacedCoursePatterns(owner._original_course_pattern_geometry,
        course, cq, owner._independent_pair_offsets)
    owner.geometry = geometry
    owner.factor = _factor_geometry(owner._original_course_pattern_factor, geometry)
    owner.external_factor = _factor_geometry(owner._original_course_external_pattern_factor, geometry)
    try:
        yield geometry
    finally:
        owner.geometry, owner.factor, owner.external_factor = old


class SourceWithIndependentPatternPlacements:
    """Complete old source and profiles with two appended placement columns."""
    def __init__(self, base, state, authority_path):
        self.base, self.state = base, state
        self.core_count=int(state.structured_programme_coordinate_chart.core_count)
        self.prefix_count = int(base.count)
        self.quadratic, self.scales = base.quadratic, base.scales
        self.updater = PrefixProfileUpdate(base.updater, self.prefix_count)
        path = Path(authority_path).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != AUTHORITY_SHA256:
            raise ValueError('Exact current direct-user shared-template/independent-location authority required')
        authority = json.loads(path.read_text(encoding='utf8'))
        constraint = authority.get('constraint', {})
        if (authority.get('event_id') != 'ankara_2026'
                or constraint.get('same_printed_template') is not True
                or constraint.get('shared_printed_dimensions') is not True
                or constraint.get('physical_locations_independent') is not True
                or constraint.get('no_added_equality_observation_or_reweighted_source') is not True):
            raise ValueError('Explicit shared size and independent physical placement required')
        owner = self.patterns = getattr(state, 'patterns', None)
        if (owner is None or not callable(getattr(owner, 'bind_current_geometry', None))
                or not hasattr(owner, '_original_course_pattern_geometry')
                or vars(owner).get('_independent_pair_placement_source') is not None):
            raise ValueError('Single existing current-course pattern owner required')
        self.material_factor = getattr(base, 'native_board_material_point_factor', None)
        if self.material_factor is None or self.material_factor.patterns is not owner:
            raise ValueError('Actual admitted Ad12/25 material factor must share the original pattern owner')
        if self.material_factor.parameter_count > self.prefix_count:
            raise ValueError('Material factor prefix is outside the complete old state')
        original_frames = list(owner.fit) + list(owner.held)
        has26 = any(int(f['physical_ad_id']) == 26 for f in original_frames) or any(
            int(r['curve']['advertisement_id']) == 26 for r in self.material_factor.rows+self.material_factor.held)
        fit26 = any(int(f['physical_ad_id']) == 26 for f in owner.fit) or any(
            int(r['curve']['advertisement_id']) == 26 for r in self.material_factor.rows)
        if has26 and not fit26:
            raise ValueError('Ad26 placement columns require explicitly admitted actual FIT source, not HELD-only evidence')
        self.placement_count = 4 if has26 else 2
        self.count = self.prefix_count + self.placement_count
        self.placement_indices = np.arange(self.prefix_count, self.count)
        if not any(r['curve']['advertisement_id'] == 25 for r in self.material_factor.rows):
            raise ValueError('Ad25 independent placement requires existing actual FIT source points')
        owner._independent_pair_offsets = np.zeros(4)
        owner._independent_pair_placement_source = self
        owner.bind_current_geometry = MethodType(_bind_independent_current_geometry, owner)
        self.placement_contract = dict(
            schema='rinkspace.shared-print-template-independent-physical-placements.v1',
            placement_binding=PLACEMENT_BINDING,
            user_authority=dict(path=str(path), sha256=digest),
            original_parameter_count=self.prefix_count, parameter_count=self.count,
            parameter_names=list(PARAMETER_NAMES+(AD26_PARAMETER_NAMES if has26 else ())), placement_parameter_indices=self.placement_indices.tolist(),
            independently_estimated_advertisement_numbers=[25,26] if has26 else [25],
            Ad26_placement_status='actual_FIT_source; independent station/Z offsets' if has26 else 'no_active_or_HELD_source; unchanged unestimated initializer only',
            Ad26_future_provider_support=True,
            existing_q9_start=int(owner.start), existing_q9_shape_and_radius_slots_unchanged=True,
            old_complete_parameter_prefix_and_camera_indices_unchanged=True,
            original_source_scalar_rows=len(self.scales), source_scalar_rows=len(self.scales),
            new_observation_or_prior_rows=0, equality_observation_or_sigma_added=False,
            source_UV_sigma_roles_support_FIT_HELD_and_clock_preserved=True,
            original_typed_pattern_factor_and_native_material_point_factor_share_one_provider=True,
            placement_J_method='original actual residual central difference, step 1e-6 working metres',
            zero_placement_initializer_preserves_original_world=True, accuracy_accepted=False)
        self.contract = dict(schema='rinkspace.complete-source-with-independent-pattern-placements.v1',
            original_source_contract=deepcopy(base.contract), independent_pattern_placements=deepcopy(self.placement_contract),
            parameter_count=self.count, source_rows=len(self.scales), accuracy_accepted=False)

    def __getattr__(self, name):
        base = self.__dict__.get('base')
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def _value(self, q):
        value = np.asarray(q, float)
        if value.shape != (self.count,) or not np.isfinite(value).all():
            raise ValueError('Complete original prefix plus actual-source-qualified independent pattern placements required')
        return value

    @contextmanager
    def _offset_context(self, offset):
        offset = _offsets(offset)
        owner = self.patterns
        old = owner._independent_pair_offsets.copy()
        changed = not np.array_equal(old, offset)
        if changed:
            self.material_factor._cache.clear()
        owner._independent_pair_offsets = offset
        try:
            yield
        finally:
            owner._independent_pair_offsets = old
            if changed:
                self.material_factor._cache.clear()

    @contextmanager
    def pattern_placement_context(self, q):
        value = self._value(q)
        offsets = value[self.placement_indices]
        if self.placement_count == 2:offsets=np.r_[offsets,0.,0.]
        with self._offset_context(offsets):
            yield

    def extend_initial(self, prefix, suffix=None):
        q = np.asarray(prefix, float)
        if q.shape == (self.count,) and suffix is None:
            return self._value(q).copy()
        if q.shape != (self.prefix_count,) or not np.isfinite(q).all():
            raise ValueError('Complete unchanged previous prefix required')
        tail = np.zeros(self.placement_count) if suffix is None else np.asarray(suffix, float)
        if tail.shape == (2,) and self.placement_count == 4:tail=np.r_[tail,0.,0.]
        if tail.shape != (self.placement_count,) or not np.isfinite(tail).all():
            raise ValueError('Finite explicit actual-source-qualified independent pattern placement initializer required')
        result = np.r_[q, tail]
        with self.pattern_placement_context(result), self.patterns.bind_current_geometry(result[:self.core_count]) as geometry:
            geometry.guard(result[self.patterns.start:self.patterns.start+9])
        return result

    def unpack(self, q):
        return self.base.unpack(self._value(q)[:self.prefix_count])

    def initial(self, *args, **kwargs):
        return self.extend_initial(self.base.initial(*args, **kwargs))

    def guard(self, q):
        value = self._value(q)
        with self.pattern_placement_context(value):
            return self.base.guard(value[:self.prefix_count])

    def _placement_columns(self, q, family):
        value = self._value(q)
        prefix, offsets = value[:self.prefix_count], value[self.placement_indices]
        if self.placement_count == 2:offsets=np.r_[offsets,0.,0.]
        if family == 'shared_board_patterns':
            if not any(int(f['physical_ad_id']) in (25,26) for f in self.patterns.fit):
                return None
            evaluate = lambda: self.patterns.evaluate(prefix[:self.core_count], False)
        elif family == 'additional_real_native_board_material_points':
            cameras = self.material_factor._cameras(prefix)
            evaluate = lambda: self.material_factor._residual(prefix, self.material_factor.rows, cameras=cameras)
        else:
            return None
        columns = []
        for k in range(self.placement_count):
            plus, minus = offsets.copy(), offsets.copy()
            plus[k] += 1e-6
            minus[k] -= 1e-6
            with self._offset_context(plus):
                rp = np.asarray(evaluate(), float)
            with self._offset_context(minus):
                rm = np.asarray(evaluate(), float)
            if rp.shape != rm.shape:
                raise ValueError('Independent placement changed actual source row inventory')
            columns.append((rp-rm)/(2e-6))
        return np.column_stack(columns)

    def blocks(self, q, jacobian=True):
        value = self._value(q)
        with self.pattern_placement_context(value):
            for block in self.base.blocks(value[:self.prefix_count], jacobian):
                matrix = None
                if jacobian:
                    old = block.J.tocsr(copy=False)
                    if old.shape[1] != self.prefix_count:
                        raise ValueError('Independent placement altered original source column prefix')
                    extra = self._placement_columns(value, block.row_ledger['family'])
                    if extra is None:
                        matrix = csr_matrix((old.data, old.indices, old.indptr), shape=(old.shape[0], self.count), copy=False)
                    else:
                        if extra.shape != (len(block.r), self.placement_count):
                            raise ValueError('Actual placement derivative source rows changed')
                        matrix = hstack((old, csr_matrix(extra)), format='csr')
                # Static source identity only. Numerical placements/residuals
                # are state, and must never enter inventory hashes.
                ledger = dict(block.row_ledger, independent_pattern_placement_binding=PLACEMENT_BINDING,
                    independent_pattern_placement_parameter_indices=self.placement_indices.tolist())
                yield SourceRowBlock(block.r, matrix, block.quadratic, block.scales, ledger,getattr(block,'loss_weights',None))

    def export_pattern_scene(self, scene, q):
        value = self._value(q)
        result = deepcopy(scene)
        with self.pattern_placement_context(value):
            descriptor = self.patterns.descriptor(value[:self.core_count])
        descriptor['independent_placement_contract'] = deepcopy(self.placement_contract)
        result['shared_board_patterns'] = descriptor
        return result

    def export_additional_circle_scene(self, scene, q):
        value = self._value(q)
        return self.base.export_additional_circle_scene(scene, value[:self.prefix_count])

    def pattern_source_review(self, q):
        value = self._value(q)
        with self.pattern_placement_context(value):
            result = self.patterns.source_report(value[:self.core_count])
        result['independent_placement_contract'] = deepcopy(self.placement_contract)
        return result

    def material_point_source_review(self, q):
        value = self._value(q)
        with self.pattern_placement_context(value):
            result = self.material_factor.review(value[:self.material_factor.parameter_count])
        result['independent_placement_contract'] = deepcopy(self.placement_contract)
        return result


def wrap_pattern_independent_placements(base, state, authority_path):
    return SourceWithIndependentPatternPlacements(base, state, authority_path)
