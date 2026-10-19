"""Explicit reference coordinates for currently unobserved venue materials.

This module is not installed in a solver or renderer. A future stage must use
the returned physical q for *all* source, guard, scene and constraint calls,
and pull physical derivatives back through E. Reference transport is not a
measurement and does not establish a common-scale gauge.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, eye, issparse


DERIVED40 = (4, 5, 7, 8, 9, 10, 11, 28, 29, 30, 31, 62, 63, 64, 65,
             66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 80, 81, 86, 87,
             92, 93, 100, 101, 102, 112, 113, 114, 115, 116, 117)
_A = (4, 5, 7, 8)
_LENGTH = (9, 10, 11, 29, 30, 31, 101, 102, 113, 117)
_STATION = (28, 62, 64, 66, 68, 70, 72, 74, 80, 86, 92)
_Z = (63, 65, 67, 69, 71, 73, 75, 81, 87, 93)


def _reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class PhysicalCourseChart:
    """Actual finite-course values and analytic tangents; no resampling chart."""
    def __init__(self, state, anchor_q):
        self.state = state
        self.base = state.original
        self.core = self.base.provider.current
        self.count = int(state.count)
        self.unit = float(self.core.scene_unit_m)
        # This is a geometry-only state supplied by the caller. The tangent
        # index is an immutable material binding, not a nearest-point search.
        state.scene(np.asarray(anchor_q)[:self.count])
        self.pattern_origin_index = int(
            self.base._base_pattern_geometry.course_source_boundary_indices[0])

    def __call__(self, q):
        from shared_unmeasured_advertisement_geometry_candidate.shared_unmeasured_advertisement_geometry import current_course_binding
        core = self.core
        course, cq, chain = current_course_binding(
            core.parent, np.asarray(q)[:core.parent_parameter_count])
        bottom, _, jb, _, parts = course.sample_perimeter(cq, with_jacobian=True)
        edges = np.roll(bottom, -1, axis=0)[:, :2] - bottom[:, :2]
        jedges = np.roll(jb, -1, axis=0)[:, :2] - jb[:, :2]
        lengths = np.linalg.norm(edges, axis=1) * self.unit
        if not np.all(lengths > 0):
            raise ValueError('Actual finite course has a zero-length edge')
        jl = np.einsum('ni,nik->nk', edges, jedges)
        jl *= (self.unit ** 2 / lengths)[:, None]
        cumulative = np.r_[0., np.cumsum(lengths)]
        jcumulative = np.vstack((np.zeros((1, 6)), np.cumsum(jl, axis=0)))
        right = next(p for p in parts if p['id'] == 'right_end')
        right_origin = int(right['first_vertex'])
        selected = (len(lengths), self.pattern_origin_index, right_origin)
        values = np.r_[cumulative[list(selected)], 2 * cq[:2] / self.unit]
        local_j = np.vstack((jcumulative[list(selected)],
                             np.eye(6)[:2] * (2 / self.unit))) @ chain
        jac = np.zeros((5, len(q)))
        jac[:, :core.parent_parameter_count] = local_j
        return values, jac


class DerivedInactiveChart:
    """Physical q(qcanonical) and sparse E=dqphysical/dqcanonical.

    The physical anchor is the explicitly saved rescue_all40 reference. The
    appended camera tail, programme camera, print48 and every unmapped value
    pass through exactly. Released rows become ordinary independent canonical
    coordinates. Admission must declare dependencies and is rejected whenever
    it touches a row that remains derived.
    """
    def __init__(self, state, physical_anchor_q, *, recipe_reference,
                 parameter_count=None, release_rows=(), admission_dependencies=(),
                 metrics=None, dependency_records=()):
        anchor = np.asarray(physical_anchor_q)
        self.base_count = int(state.count)
        self.count = self.base_count if parameter_count is None else int(parameter_count)
        if (anchor.dtype != np.float64 or anchor.shape != (self.base_count,)
                or self.count < self.base_count or not np.isfinite(anchor).all()):
            raise ValueError('Complete finite physical float64 anchor required')
        released = set(map(int, release_rows))
        if not released <= set(DERIVED40):
            raise ValueError('Release rows must be explicitly inside the reference chart')
        self.anchor = anchor.copy()
        self.recipe_reference = deepcopy(recipe_reference)
        self.released_rows = tuple(sorted(released))
        self.derived_rows = tuple(i for i in DERIVED40 if i not in released)
        self.fixed_canonical_rows = np.asarray(self.derived_rows, dtype=np.int64)
        self.dependency_records = deepcopy(tuple(dependency_records))
        self.metrics = PhysicalCourseChart(state, anchor) if metrics is None else metrics
        self.anchor_metrics, _ = self._metrics(anchor)
        self.anchor_scale = self._scale(*self.anchor_metrics[3:])[0]
        if not anchor[3] > 0:
            raise ValueError('Positive authorized common circle radius required')
        core = state.original.provider.current
        self.datums = {}
        for number in range(14, 21):
            indices = np.asarray(core.location_indices[number], int)
            datum = (np.asarray(core._provisional_locations[number], float)
                     if number in core.location_offset_indices else np.zeros(2))
            self.datums[int(indices[0])] = float(datum[0])
            self.datums[int(indices[1])] = float(datum[1])
        for number in (21, 22, 23):
            record = core.source_print_templates[core.source_template_instance[number]]
            indices = record['instance_location_offset_indices'][number]
            datum = record['initialization']['instance_datum_station_zmin_m'][str(number)]
            self.datums.update({int(i): float(v) for i, v in zip(indices, datum, strict=True)})
        self.assert_source_admission(admission_dependencies)

    @classmethod
    def from_recipe(cls, state, recipe_path, **kwargs):
        recipe_ref = _reference(recipe_path)
        recipe = json.loads(Path(recipe_path).read_text(encoding='utf8'))
        if (recipe.get('schema') != 'rinkspace.all-inactive-explicit-reference-transport.v1'
                or recipe.get('not_new_measurement') is not True
                or tuple(sorted(recipe['changed_parameter_indices'])) != DERIVED40):
            raise ValueError('Explicit complete inactive40 reference recipe required')
        for name in ('candidate', 'parent', 'reference', 'retained_A_rescue7'):
            if _reference(recipe[name]['path']) != recipe[name]:
                raise ValueError('Reference recipe artifact changed: ' + name)
        with np.load(recipe['candidate']['path'], allow_pickle=False) as saved:
            anchor = saved['q'].copy()
        if _sha(anchor) != recipe['q_sha256']:
            raise ValueError('Physical anchor q identity changed')
        rows = {int(row['index']) for row in recipe['changes']}
        if rows != set(DERIVED40) - set(_A):
            raise ValueError('Expected exact 36 inactive recipe rows')
        names = state.original.world_parameter_names
        if any(names[int(row['index'])] != row['name'] for row in recipe['changes']):
            raise ValueError('Current physical parameter semantics differ from the recipe')
        return cls(state, anchor, recipe_reference=recipe_ref,
                   dependency_records=recipe['changes'], **kwargs)

    def _metrics(self, q):
        values, jac = self.metrics(q)
        values, jac = np.asarray(values, float), np.asarray(jac, float)
        if (values.shape != (5,) or jac.shape != (5, len(q))
                or not np.isfinite(values).all() or not np.isfinite(jac).all()
                or min(values[0], values[3], values[4]) <= 0):
            raise ValueError('Actual P/origins/L/W and complete analytic tangent required')
        return values, jac

    @staticmethod
    def _scale(length, width):
        n, d = 60 * length + 30 * width, length * length + width * width
        return n / d, np.array([(60 * d - 2 * length * n) / d ** 2,
                                (30 * d - 2 * width * n) / d ** 2])

    def assert_source_admission(self, dependency_indices):
        conflict = sorted(set(map(int, dependency_indices)) & set(self.derived_rows))
        if conflict:
            raise ValueError('Source admission requires explicit release of derived rows: '
                             + ','.join(map(str, conflict)))
        return True

    def effective_q_and_chain(self, q):
        q = np.asarray(q)
        if q.dtype != np.float64 or q.shape != (self.count,) or not np.isfinite(q).all():
            raise ValueError('Complete finite canonical float64 q required')
        values, jac = self._metrics(q)
        pa, oa, o24a, _, _ = self.anchor_metrics
        p, o, o24, length, width = values
        dp, do, do24, dl, dw = jac
        scale, ds = self._scale(length, width)
        u = self.anchor_scale / scale
        du = -(u / scale) * (ds[0] * dl + ds[1] * dw)
        a = self.anchor
        effective = q.copy()
        chain = eye(self.count, format='lil', dtype=np.float64)

        def set_row(i, value, derivative):
            if i in self.derived_rows:
                effective[i] = value
                nonzero = np.flatnonzero(derivative)
                chain.rows[i] = nonzero.tolist()
                chain.data[i] = derivative[nonzero].tolist()

        radius_d = np.zeros(self.count); radius_d[3] = 1.
        for i in _A:
            set_row(i, a[i] * (q[3] / a[3]), (a[i] / a[3]) * radius_d)
        for i in _LENGTH:
            set_row(i, a[i] * u, a[i] * du)
        for i in _STATION:
            datum = self.datums.get(i, 0.)
            fraction = (a[i] + datum) / pa
            set_row(i, a[i] + fraction * (p - pa), fraction * dp)
        for i in _Z:
            datum = self.datums.get(i, 0.)
            set_row(i, a[i] + (a[i] + datum) * (u - 1.), (a[i] + datum) * du)
        f13 = (oa + a[100]) / pa
        set_row(100, a[100] + f13 * (p - pa) - (o - oa), f13 * dp - do)
        # Ad24's actual right-end origin has positive station direction.
        f24 = (o24a + a[112]) / pa
        set_row(112, a[112] + f24 * (p - pa) - (o24 - o24a), f24 * dp - do24)
        s12, z12 = np.array([103, 105, 107]), np.array([104, 106, 108])
        s12a, z12a = a[s12].mean(), a[z12].mean()
        f25 = (oa + pa / 2 + s12a + a[114]) / pa
        d25s = (f25 - .5) * dp - do
        d25s[s12] -= 1 / 3
        set_row(114, a[114] + (f25 - .5) * (p - pa) - (o - oa)
                - (q[s12].mean() - s12a), d25s)
        d25z = (z12a + a[115]) * du
        d25z[z12] -= 1 / 3
        set_row(115, a[115] + (z12a + a[115]) * (u - 1.)
                - (q[z12].mean() - z12a), d25z)
        f26 = a[116] / pa
        set_row(116, a[116] + f26 * (p - pa), f26 * dp)
        result = chain.tocsr()
        result.eliminate_zeros()
        if not np.isfinite(result.data).all():
            raise ValueError('Nonfinite derived coordinate tangent')
        return effective, result

    def pullback_constraints(self, original, q, physical_columns, *, canonical_columns=None,
                             fixed_columns=()):
        """Preserve original current lower/margins and apply M @ E exactly.

        The caller must first evaluate original constraints at effective_q.
        Columns omitted from the returned chart must be explicitly fixed.
        """
        physical = np.asarray(physical_columns, int)
        canonical = physical if canonical_columns is None else np.asarray(canonical_columns, int)
        matrix = original['matrix']
        if (matrix.shape[1] != len(physical) or len(set(physical)) != len(physical)
                or len(set(canonical)) != len(canonical)
                or np.any(physical < 0) or np.any(physical >= self.count)
                or np.any(canonical < 0) or np.any(canonical >= self.count)):
            raise ValueError('Exact original and selected constraint columns required')
        _, chain = self.effective_q_and_chain(q)
        full = csr_matrix(matrix) @ chain[physical]
        full.eliminate_zeros()
        omitted = np.setdiff1d(np.arange(self.count), np.r_[canonical, list(fixed_columns)])
        if full[:, omitted].nnz:
            raise ValueError('Constraint chain has undeclared omitted canonical dependencies')
        pulled = full[:, canonical].tocsr()
        report = dict(original.get('report', {}),
            derived_inactive_chain_applied=True, original_rows_current_lower_and_margins_preserved=True,
            physical_columns=physical.tolist(), canonical_columns=canonical.tolist(),
            explicitly_fixed_omitted_columns=list(map(int, fixed_columns)))
        return dict(original, matrix=pulled if issparse(matrix) else pulled.toarray(), report=report)

    def identity(self):
        return dict(schema='rinkspace.explicit-derived-inactive-chart.v1',
            recipe=deepcopy(self.recipe_reference), physical_anchor_q_sha256=_sha(self.anchor),
            parameter_count=self.count, derived_rows=list(self.derived_rows),
            released_rows=list(self.released_rows), reference_not_measurement=True,
            raw_derived_rows_require_true_solver_fixed_elimination=True,
            A_four_authorized_q3_ratio=True, station_unwrapped_before_projection=True,
            closest60x30_single_assumed_metric_scale=True,
            pure_common_scale_gauge_proved=False, installed_in_running_model=False,
            unobserved_expected_proxy_rows_outside_this_recipe_not_covered=True,
            current_painted_lane_end_coordinates_not_derived=True,
            all_source_admission_dependencies_must_be_declared=True,
            source_scene_guard_and_constraints_must_share_effective_q=True)
