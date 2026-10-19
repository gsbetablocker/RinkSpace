"""Replace archived pixel factors with the authenticated finite dataset.

The archived graph remains available for complete-state and scene export.
Its pixel evaluators and profile updates are never called by this source.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import csr_matrix

from .strict_model_scope_inputs import load_strict_programme_motion_inputs


class FrozenArchivedProfileUpdate:
    """Keep unused historical auxiliary profiles without evaluating their UVs."""

    def __init__(self, archived, parameter_count):
        self.archived = archived
        self.parameter_count = int(parameter_count)

    def __call__(self, q):
        q = np.asarray(q, float)
        if q.shape != (self.parameter_count,) or not np.isfinite(q).all():
            raise ValueError('Complete finite strict-source parameter vector required')

    def checkpoint_arrays(self):
        return self.archived.checkpoint_arrays()

    def iter_checkpoint_arrays(self):
        return self.archived.iter_checkpoint_arrays()

    def restore_checkpoint(self, values):
        return self.archived.restore_checkpoint(values)


def _full_width(matrix, count):
    matrix = matrix.tocsr(copy=False)
    if matrix.shape[1] > count:
        raise ValueError('Engineering derivative exceeds complete parameter chart')
    return csr_matrix((matrix.data, matrix.indices, matrix.indptr),
                      shape=(matrix.shape[0], count), copy=False)


class StrictReviewedProgrammeSource:
    """Finite source pixels and the existing explicit engineering priors only.

    Construct after the historical parameter suffixes and before canonical
    permutation. ``base`` owns the archived export interfaces, ``raw`` is the
    original CurrentProgrammeSource, and ``bias_source`` is ExplicitBiasSource.
    Current motion partitions have zero admitted rows. A nonempty future
    partition needs a finite operator adapter and is rejected here.
    """

    def __init__(self, base, raw, bias_source, dataset_factor, motion_reference):
        from calibration.program_tracking.strict_model_scope import file_sha

        self.base = base
        self.raw = raw
        self.state = raw.state
        self.bias_source = bias_source
        self.strict_dataset_factor = dataset_factor
        self.count = int(base.count)
        self.core_count = int(bias_source.core_count)
        if int(dataset_factor.parameter_count) != self.count:
            raise ValueError('Strict dataset and archived complete chart differ')
        motion, binding = load_strict_programme_motion_inputs(
            motion_reference['path'], expected_sha256=motion_reference['sha256'])
        if set(motion) != {b.name for b in self.state.blocks}:
            raise ValueError('Strict motion binding must cover every programme family')
        motion_counts = []
        for block in self.state.blocks:
            values = motion[block.name]
            bundle = block.bundle
            if not np.array_equal(values['stamps'], np.asarray(bundle.stamps)):
                raise ValueError('Strict motion exposure clock changed: ' + block.name)
            previous = np.asarray(getattr(bundle, 'original_motion_compaction_active_to_old',
                                          np.arange(len(bundle.a))), np.int64)
            if not np.array_equal(values['previous_active_to_original'], previous):
                raise ValueError('Strict motion archive index identity changed: ' + block.name)
            if len(values['a']):
                raise ValueError('Nonempty strict motion needs its authenticated finite operator; '
                                 'UNKNOWN_DEPTH is never admitted')
            motion_counts.append(dict(block=block.name, archived_points=len(bundle.a),
                                      admitted_points=0))
        self.reference_factor = getattr(base, 'reference_factor', None)
        if self.reference_factor is None:
            raise ValueError('Existing explicit reference18 factor required')
        self.external_roll_positions = np.asarray(bias_source.external_active_positions, int)
        self.sigma = float(bias_source.sigma)
        self.bias_slots = np.asarray(bias_source.slots, int)
        self.soft_pivot = getattr(raw.args, 'pivot_border_policy', 'outside-soft') == 'outside-soft'
        self.prior_count = (sum(len(b.bundle.stamps) for b in self.state.blocks)
                            + len(self.external_roll_positions) + len(self.state.ids)
                            + 1 + (len(self.state.ids) if self.soft_pivot else 0) + 3
                            + len(self.reference_factor.scales))
        self.quadratic = np.r_[np.ones(self.prior_count, bool), dataset_factor.quadratic]
        self.scales = np.r_[np.ones(self.prior_count - len(self.reference_factor.scales)),
                            self.reference_factor.scales, dataset_factor.scales]
        self.updater = FrozenArchivedProfileUpdate(base.updater, self.count)
        from .strict_dataset_execution_runtime import install_strict_dataset_execution
        execution_receipt = install_strict_dataset_execution(dataset_factor)
        self.contract = dict(
            schema='rinkspace.strict-reviewed-finite-programme-source.v1',
            parameter_count=self.count, source_rows=len(self.scales),
            archived_source_rows=len(base.scales),
            engineering_prior_rows=self.prior_count,
            admitted_motion_points=0, archived_motion_blocks=motion_counts,
            strict_motion_reference=deepcopy(motion_reference),
            strict_motion_review_reference=deepcopy(binding['review']),
            source_identity=deepcopy(binding['source']),
            finite_dataset_contract=deepcopy(dataset_factor.contract),
            strict_dataset_execution=deepcopy(execution_receipt),
            full_parameter_chart_and_source_clock_preserved=True,
            archived_pixel_factors_in_FIT=False,
            archived_pixel_guards_evaluated=False,
            archived_motion_profiles_evaluated=False,
            historical_profile_arrays_retained_as_unused_auxiliary=True,
            allowed_static_surfaces=['ice', 'board'],
            unknown_or_unreviewed_default='excluded',
            all_pixel_families_routed_to_strict_input=True,
            source_scope_runtime_adapter_path=str(Path(__file__).resolve()),
            source_scope_runtime_adapter_sha256=file_sha(Path(__file__).resolve()),
            original_engineering_sigma_and_nominal_values_preserved=True,
            fresh_initial_source_J_required=True, accuracy_accepted=False)

    def __getattr__(self, name):
        base = self.__dict__.get('base')
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def _value(self, q):
        q = np.asarray(q, float)
        if q.shape != (self.count,) or not np.isfinite(q).all():
            raise ValueError('Complete finite evaluation-chart q required')
        return q

    def _placement_context(self, q):
        method = getattr(self.base, 'pattern_placement_context', None)
        return nullcontext() if method is None else method(q)

    def guard(self, q):
        """Physical domains only; admitted pixels are checked by their factor."""
        from installation_mount_candidate.mount_graph import validate_mounts
        from structured_programme_source_candidate.native_camera_local import native_camera_local

        q = self._value(q)
        core = q[:self.core_count]
        state = self.state
        state.geometry.validate_parameters(core[state.geometry_parameter_indices])
        validate_mounts(state, core)
        for name in state.ids:
            if core[state.slots[name]][2] <= 0:
                raise ValueError('Physical panhead height must be positive')
        for block in state.blocks:
            local = native_camera_local(block, core)
            focal = local[7::4]
            if np.any((focal < np.log(100.)) | (focal > np.log(50000.))):
                raise ValueError('Physical focal domain')
        if state.patterns is not None:
            with self._placement_context(q), state.patterns.bind_current_geometry(core) as geometry:
                geometry.guard(core[state.patterns.start:state.patterns.start+9])
        circle = getattr(self.base, 'circle_factor', None)
        if circle is not None:
            circle.guard(q[:int(circle.parameter_count)])

    def _roll_block(self, q, positions, installation, family, jacobian):
        from correlated_roll_prior_candidate.explicit_bias_source import theta_and_gradient
        from z190_streaming_source_normal import SourceRowBlock

        positions = np.asarray(positions, int)
        rotations = np.asarray([q[p:p+3] for p in positions])
        theta, gradient = theta_and_gradient(rotations)
        residual = (theta - q[self.bias_slots[installation]]) / self.sigma
        matrix = None
        if jacobian:
            rows = np.repeat(np.arange(len(positions)), 4)
            columns = np.c_[positions[:, None] + np.arange(3),
                            np.full(len(positions), self.bias_slots[installation])].ravel()
            values = np.c_[gradient / self.sigma,
                           np.full(len(positions), -1. / self.sigma)].ravel()
            matrix = csr_matrix((values, (rows, columns)), shape=(len(residual), self.count))
        return SourceRowBlock(residual, matrix, np.ones(len(residual), bool),
                              np.ones(len(residual)), dict(family=family,
                              source_role='explicit_engineering_roll_prior',
                              statistical_roll_bias_inst=installation))

    def _engineering_blocks(self, q, jacobian):
        from z190_streaming_source_normal import SourceRowBlock
        from z190_continuous_source_bundle import SCENE_UNIT_M
        from z190_fixed_pivot_board_side_support import fixed_pivot_board_side_support
        from structured_programme_source_candidate.native_camera_local import native_camera_local

        state = self.state
        core = q[:self.core_count]
        order = tuple(state.ids)
        for block in state.blocks:
            positions = block.start + 4 * np.arange(len(block.bundle.stamps))
            installation = order.index(block.bundle.calibration.installation_id)
            yield self._roll_block(q, positions, installation,
                                   'programme_engineering_roll:' + block.name, jacobian)
        if len(self.external_roll_positions):
            yield self._roll_block(q, self.external_roll_positions, 0,
                                   'external_engineering_roll', jacobian)
        for name in state.ids:
            block = self.raw.refs[name]
            local = native_camera_local(block, core)
            calibration = block.bundle.current_calibration(local)
            camera, _, view_j = calibration.camera25_and_jacobians(
                block.bundle.view(local, block.bundle.reference_pts))
            residual = np.array([(camera[2] * SCENE_UNIT_M - 1.6) / .18])
            matrix = None
            if jacobian:
                values = np.zeros((1, self.count))
                values[0, state.slots[name][2]] = 1. / .18
                position = block.start + 4 * block.bundle.index[block.bundle.reference_pts]
                values[0, position:position+4] = view_j[2] * SCENE_UNIT_M / .18
                values[0, block.optics_chart.mount_indices] = (
                    cv2.Rodrigues(np.asarray(camera[3:6]))[0].T[2] / .18)
                matrix = csr_matrix(values)
            yield SourceRowBlock(residual, matrix, np.ones(1, bool), np.ones(1),
                                 dict(family='optical_height:' + name,
                                      source_role='existing_engineering_height_prior'))
        yield SourceRowBlock(np.zeros(1), csr_matrix((1, self.count)) if jacobian else None,
                             np.ones(1, bool), np.ones(1),
                             dict(family='common_row_midpoint', source_role='structural_zero'))
        if self.soft_pivot:
            indices = np.asarray(state.geometry_parameter_indices, int)
            world = core[indices]
            geometry = state.geometry.current
            course, cq, _ = geometry.print_course_binding(world)
            polygon = course.sample_perimeter(cq)[0][:, :2] * SCENE_UNIT_M
            sigma = self.raw.args.pivot_board_soft_sigma_m
            for name in state.ids:
                residual, pivot_j, _ = fixed_pivot_board_side_support(
                    polygon, core[state.slots[name]], sigma)
                matrix = None
                if jacobian:
                    values = np.zeros((1, self.count))
                    values[:, state.slots[name]] = pivot_j
                    for column in range(44, 50):
                        plus, minus = world.copy(), world.copy()
                        plus[column] += 1e-6
                        minus[column] -= 1e-6
                        pp = course.sample_perimeter(geometry.current_course6(plus))[0][:, :2] * SCENE_UNIT_M
                        pm = course.sample_perimeter(geometry.current_course6(minus))[0][:, :2] * SCENE_UNIT_M
                        rp = fixed_pivot_board_side_support(pp, core[state.slots[name]], sigma)[0]
                        rm = fixed_pivot_board_side_support(pm, core[state.slots[name]], sigma)[0]
                        values[:, indices[column]] = (rp - rm) / (2e-6)
                    matrix = csr_matrix(values)
                yield SourceRowBlock(residual, matrix, np.ones(1, bool), np.ones(1),
                                     dict(family='fixed_pivot_board_side:' + name,
                                          source_role='existing_engineering_pivot_prior'))
        bias = q[self.bias_slots]
        matrix = (csr_matrix((np.full(3, 1. / self.sigma),
                              (np.arange(3), self.bias_slots)), shape=(3, self.count))
                  if jacobian else None)
        yield SourceRowBlock(bias / self.sigma, matrix, np.ones(3, bool), np.ones(3),
                             dict(family='installation_correlated_roll_hardware_bias',
                                  source_role='explicit_statistical_hardware_hypothesis_candidate'))
        reference = self.reference_factor.block(
            q[:int(self.reference_factor.parameter_count)], jacobian)
        yield SourceRowBlock(reference.r, _full_width(reference.J, self.count) if jacobian else None,
                             reference.quadratic, reference.scales, reference.row_ledger)

    def blocks(self, q, jacobian=True):
        from z190_streaming_source_normal import SourceRowBlock

        q = self._value(q)
        cursor = 0
        for block in self._engineering_blocks(q, jacobian):
            stop = cursor + len(block.r)
            ledger = dict(block.row_ledger, original_start=cursor, original_stop=stop,
                          strict_source_consumer=True)
            yield SourceRowBlock(block.r, block.J, block.quadratic, block.scales, ledger,
                                 getattr(block, 'loss_weights', None))
            cursor = stop
        if cursor != self.prior_count:
            raise ValueError('Strict engineering prior inventory changed')
        for block in self.strict_dataset_factor.iter_blocks(q, jacobian):
            stop = cursor + len(block.r)
            ledger = dict(block.row_ledger, original_start=cursor, original_stop=stop,
                          strict_source_consumer=True)
            yield SourceRowBlock(block.r, block.J, block.quadratic, block.scales, ledger,
                                 getattr(block, 'loss_weights', None))
            cursor = stop
        if cursor != len(self.scales):
            raise ValueError('Strict source omitted active rows')

    def strict_source_review(self, q):
        return self.strict_dataset_factor.review(self._value(q))

