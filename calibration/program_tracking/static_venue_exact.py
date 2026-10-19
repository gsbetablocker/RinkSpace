"""Exact finite-difference reuse for unchanged static material curves.

Every original world column, guard and FD side remains. A sample curve whose
float64 world bytes equal the current base curve has exactly the same source
operator and camera, so its base residual is reused without projection.
"""
import hashlib
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .first_shot_physical_state import GeometryDomainError
from .static_venue_source import StaticVenueFactor, current_material_curve
from .z190_source_camera_diagnostic import absolute_value_jac15
from wide_anchor_candidate.full_brown_projection import curve_vector_error_full_brown
from z190_streaming_source_normal import SourceRowBlock


def _same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


class ExactStaticVenueFactor(StaticVenueFactor):
    """Same factor and full FD inventory; no approximate zero-column test."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.identity = dict(self.identity, exact_static_reuse=dict(
            producer=dict(path=str(Path(__file__).resolve()),
                sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
            schema='rinkspace.exact-current-world-FD-source-reuse.v1',
            comparison='float64 dtype shape and every world byte',
            original_all_geometry_columns_and_stencils_retained=True,
            camera_and_original_UV_sigma_operator_fixed_within_one_J=True,
            tolerance_or_small_J_selection=False,
            cache_lifetime='one exact block Jacobian evaluation'))
        self.last_exact_reuse = None

    def _sample_values(self, scene, camera, originals, baseline_worlds, baseline_r):
        values = baseline_r.copy()
        changed_rows, changed_slices = [], []
        cursor = 0
        for original, baseline in zip(originals, baseline_worlds, strict=True):
            n = 2 * len(original['pixels_uv'])
            world = current_material_curve(scene, original['target'], samples=self.samples)
            if not _same(np.asarray(world), baseline):
                changed_rows.append(dict(world=world, uv=np.asarray(original['pixels_uv'], float),
                    sigma=np.broadcast_to(np.asarray(original['sigma_px'], float), (n // 2,))))
                changed_slices.append(slice(cursor, cursor+n))
            cursor += n
        if cursor != len(values):
            raise ValueError('Original static source row layout changed')
        if changed_rows:
            if self.camera_policy.kind == 'brown13':
                updated = absolute_value_jac15(camera, changed_rows, False)
            else:
                updated = np.concatenate([curve_vector_error_full_brown(
                    camera, row['world'], row['uv'], False) / np.repeat(row['sigma'], 2)
                    for row in changed_rows])
            offset = 0
            for span in changed_slices:
                n = span.stop-span.start
                values[span] = updated[offset:offset+n]
                offset += n
            if offset != len(updated):
                raise ValueError('Exact changed-curve source values changed row count')
        return values, len(originals)-len(changed_rows)

    def block(self, q, jacobian):
        if not jacobian:
            return super().block(q, False)
        programme, camera = self.split(q)
        self.geometry_guard(programme)
        scene = self.scene_at(programme)
        residual, camera_jac = self._evaluate_scene(scene, camera, True)
        originals = self.packet.rows
        baseline_worlds = tuple(np.asarray(current_material_curve(scene, row['target'],
            samples=self.samples)).copy() for row in originals)
        data, indices, reuse = [], [], []
        self.last_geometry_stencils = []
        for column in self.geometry_columns:
            candidates, failures, unchanged = [], [], []
            for sign in (-1., 1.):
                sample = programme.copy(); sample[column] += sign*self.fd_step
                try:
                    self.geometry_guard(sample)
                except GeometryDomainError as error:
                    candidates.append(None); failures.append(str(error)); unchanged.append(None)
                else:
                    value, count = self._sample_values(self.scene_at(sample), camera,
                        originals, baseline_worlds, residual)
                    candidates.append(value); failures.append(None); unchanged.append(count)
            minus, plus = candidates
            if minus is None and plus is None:
                raise GeometryDomainError('Both original static-venue world FD sides invalid: '+str(column))
            if minus is not None and plus is not None:
                derivative = (plus-minus)/(2*self.fd_step); method = 'central_original_step'
            elif plus is not None:
                derivative = (plus-residual)/self.fd_step; method = 'forward_original_step'
            else:
                derivative = (residual-minus)/self.fd_step; method = 'backward_original_step'
            if derivative.shape != residual.shape:
                raise ValueError('Static venue geometry FD changed source rows')
            data.append(derivative); indices.append(int(column))
            self.last_geometry_stencils.append(dict(column=int(column), method=method,
                step=self.fd_step, minus_geometry_domain_failure=failures[0],
                plus_geometry_domain_failure=failures[1]))
            reuse.append(dict(column=int(column), unchanged_material_groups_minus_plus=unchanged))
        matrix = np.column_stack([*data, *camera_jac.T])
        columns = np.r_[indices, self.camera_columns]
        row_ids = np.repeat(np.arange(len(residual)), len(columns))
        col_ids = np.tile(columns, len(residual))
        jac = csr_matrix((matrix.ravel(), (row_ids, col_ids)),
            shape=(len(residual), self.parameter_count))
        jac.eliminate_zeros()
        self.last_exact_reuse = dict(material_groups=len(originals), columns=reuse,
            unchanged_group_projections_reused=sum(n for r in reuse
                for n in r['unchanged_material_groups_minus_plus'] if n is not None),
            source_rows_and_every_geometry_column_retained=True)
        return SourceRowBlock(residual, jac, self.quadratic, self.scales,
            self.packet.ledger, self.packet.loss_weights)
