"""Opt-in six shared principal-point engineering prior rows.

Reuse the PP6 target/operator described by MappedPP6WorkingPriorBlock, while
binding the current complete canonical chart rather than its historical fixed
34238 layout. This is an identifiability candidate, not source evidence or a
repair of a prior promised by the present source contract.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

TARGET_NATIVE_PX = np.array([960.0, 540.0])
ENGINEERING_SIGMA_NATIVE_PX = 32.0
SCHEMA = 'rinkspace.current-shared-PP6-engineering-identifiability-candidate.v1'


def implementation_sources():
    """Authenticate the opt-in operator and the reused historical definition."""
    paths = (Path(__file__).resolve(),
             Path(__file__).resolve().parent / 'legacy' / 'shared_board_course_candidate'
             / 'course_source_chart.py')
    return [dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for path in paths]


class CurrentPP6WorkingPriorBlock:
    rows = 6
    mode = 'principal_point_only6'

    def __init__(self, state, source):
        self.parameter_count = int(source.count)
        self._evaluation_core = source.evaluation_core
        self._ids = tuple(state.ids)
        if len(self._ids) != 3 or len(set(self._ids)) != 3:
            raise ValueError('Exactly three original physical installations required')
        self._bindings = []
        columns = []
        for installation in self._ids:
            blocks = [b for b in state.blocks
                      if b.bundle.calibration.installation_id == installation]
            if not blocks:
                raise ValueError('Missing actual physical installation producer')
            block = blocks[0]
            chart = state.installation_optics[installation]
            base = block.optics_base_calibration
            if (tuple(base.output_mode.domain_wh) != (1920, 1080)
                    or tuple(base.output_mode.final_wh) != (1920, 1080)
                    or not np.array_equal(base.output_mode.A, np.eye(3))
                    or len(base.principal_x_fraction) != 1
                    or len(base.principal_y_fraction) != 1):
                raise ValueError('Original native identity domain and constant principal point required')
            for other in blocks:
                if (other.optics_chart.indices != chart.indices
                        or other.optics_base_calibration.principal_x_fraction != base.principal_x_fraction
                        or other.optics_base_calibration.principal_y_fraction != base.principal_y_fraction):
                    raise ValueError('Physical installation principal-point source producers disagree')
            slots = source.parameter_layout.canonical_indices(chart.indices[:2])
            columns.extend(np.asarray(slots, int).tolist())
            self._bindings.append((installation, chart, base))
        self.columns = np.asarray(columns, int)
        if (len(set(columns)) != 6 or np.any(self.columns < 0)
                or np.any(self.columns >= self.parameter_count)):
            raise ValueError('Six distinct existing canonical principal-point columns required')
        self.quadratic = np.ones(6, bool)
        self.scales = np.full(6, ENGINEERING_SIGMA_NATIVE_PX)
        self._J = csr_matrix((np.full(6, 1.0 / ENGINEERING_SIGMA_NATIVE_PX),
                             (np.arange(6), self.columns)),
                            shape=(6, self.parameter_count))
        self.contract = dict(schema=SCHEMA, rows=6, parameter_count=self.parameter_count,
            implementation_sources=implementation_sources(),
            installation_order=list(self._ids), principal_parameter_indices=self.columns.tolist(),
            current_native_principal_target_px=TARGET_NATIVE_PX.tolist(),
            engineering_sigma_native_px=ENGINEERING_SIGMA_NATIVE_PX,
            reference='existing MappedPP6WorkingPriorBlock [960,540]/32 native-pixel operator',
            historical_reference_module='shared_board_course_candidate.course_source_chart',
            original_target_not_recentered_on_current_or_seed_fit=True,
            prior_counted_once_per_physical_installation=True,
            physical_intrinsics_are_existing_shared_parameters=True,
            radial_rows_admitted=False, new_geometry_or_camera_parameters=0,
            source_UV_sigma_roles_and_human_truth_unchanged=True,
            engineering_prior_not_user_hard_truth=True,
            manufacturer_or_survey_measurement_claim=False,
            opt_in_identifiability_candidate_not_claimed_missing_contract_repair=True,
            accuracy_accepted=False)

    def native_principal_points(self, q):
        q = np.asarray(q, float)
        if q.shape != (self.parameter_count,) or not np.isfinite(q).all():
            raise ValueError('Exact finite complete canonical q required')
        core = self._evaluation_core(q)
        result = []
        for _, chart, base in self._bindings:
            current = chart.calibration(base, core)
            result.append([1920.0 * current.principal_x_fraction[0],
                           1080.0 * current.principal_y_fraction[0]])
        return np.asarray(result)

    def evaluate(self, q, jacobian=True):
        values = ((self.native_principal_points(q) - TARGET_NATIVE_PX)
                  / ENGINEERING_SIGMA_NATIVE_PX).ravel()
        return (values, self._J) if jacobian else values

    def source_block(self, q, jacobian=True, *, original_start):
        from z190_streaming_source_normal import SourceRowBlock
        result = self.evaluate(q, jacobian)
        residual, matrix = result if jacobian else (result, None)
        return SourceRowBlock(residual, matrix, self.quadratic, self.scales,
            dict(family='explicit_optics_engineering_working_prior',
                 original_start=int(original_start), original_stop=int(original_start)+6,
                 source_role='engineering_working_hypothesis_prior',
                 optical_prior_contract=self.contract))


class SourceWithCurrentPP6WorkingPrior:
    def __init__(self, base, factor):
        self.base = base
        self.principal_point_working_prior = factor
        self.count = int(base.count)
        if factor.parameter_count != self.count:
            raise ValueError('Prior must share the complete current parameter chart')
        self.prior_start = len(base.scales)
        self.quadratic = np.r_[base.quadratic, factor.quadratic]
        self.scales = np.r_[base.scales, factor.scales]
        self.contract = dict(base.contract,
            principal_point_engineering_prior=factor.contract,
            source_rows=len(self.scales),
            original_source_scalar_prefix_preserved=True,
            original_source_scalars_before_PP6=self.prior_start)

    def __getattr__(self, name):
        base = self.__dict__.get('base')
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def blocks(self, q, jacobian=True):
        yield from self.base.blocks(q, jacobian)
        yield self.principal_point_working_prior.source_block(
            q, jacobian, original_start=self.prior_start)


def wrap_current_principal_point_working_prior(source, state):
    if hasattr(source, 'principal_point_working_prior'):
        raise ValueError('PP6 engineering prior already installed')
    return SourceWithCurrentPP6WorkingPrior(
        source, CurrentPP6WorkingPriorBlock(state, source))
