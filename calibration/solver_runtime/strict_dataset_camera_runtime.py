"""Prepared physical camera mapping for the strict dataset, without an installer.

The existing camera chart still supplies the nominal local state and qualified
calibration. Only its 13 coordinate derivatives use the original calibration's
analytical shared/view mapping instead of repeatedly rebuilding the full chart.
Callers must qualify this helper against the factor's original FD before use.
"""
from __future__ import annotations

import time
import hashlib
from pathlib import Path
from types import MethodType

import numpy as np


def strict_dataset_camera_mapping(block, q, actual_pts, *, core_count):
    """Return the same camera25, global columns and conditioned camera15 chain."""
    from structured_programme_source_candidate.native_camera_local import native_camera_local
    from z190_continuous_source_bundle import SCENE_UNIT_M
    from z190_fixed_lever_xyz_source_bundle import FixedLeverXYZSourceBundle
    from z190_physical_camera import SharedCalibration

    if not isinstance(block.bundle, FixedLeverXYZSourceBundle):
        raise TypeError('Original positive-pivot fixed XYZ camera bundle required')
    value = np.asarray(q, float)
    if value.ndim != 1 or not np.isfinite(value).all() or not 0 < core_count <= len(value):
        raise ValueError('Complete unchanged finite evaluation camera chart required')
    actual_pts = int(actual_pts)
    local = native_camera_local(block, value[:core_count])
    calibration = block.bundle.current_calibration(local)
    if not isinstance(calibration, SharedCalibration):
        raise TypeError('Original reviewed analytical physical camera mapping required')
    camera, shared, view = calibration.camera25_and_jacobians(block.bundle.view(local, actual_pts))
    if any(len(getattr(calibration, name)) != 1 for name in (
            'principal_x_fraction', 'principal_y_fraction', 'k1')):
        raise ValueError('Original constant installation lens increment chart required')
    xyz = np.asarray(block.xyz_slots, int)
    optics = np.asarray(block.optics_chart.indices, int)
    mount = np.asarray(block.optics_chart.mount_indices, int)
    view_columns = block.start + 4 * block.bundle.index[actual_pts] + np.arange(4)
    ordered = np.r_[xyz, optics, mount, view_columns]
    if (xyz.shape != (3,) or optics.shape != (3,) or mount.shape != (3,)
            or len(np.unique(ordered)) != 13 or np.any(ordered < 0)
            or np.any(ordered >= core_count)):
        raise ValueError('Distinct existing XYZ3/lens3/mount3/view4 global columns required')
    names = calibration.shared_state_names()
    pivots = [names.index('pivot_' + axis + '_scene_div100') for axis in 'xyz']
    levers = [names.index('lever_' + axis + '_scene_div100') for axis in 'xyz']
    lens = [names.index(name + '_degree_0') for name in (
        'principal_x_fraction', 'principal_y_fraction', 'k1')]
    # Keep the nominal chart's exact division/multiplication order above.
    # Shared maps use scene/100, while global pivot/mount coordinates use m.
    chain25 = np.c_[shared[:, pivots] / (100. * SCENE_UNIT_M),
        shared[:, lens] * np.array([1. / 1920., 1. / 1080., .001]),
        shared[:, levers] / (100. * SCENE_UNIT_M), view]
    if np.any(np.abs(chain25[15:]) > 1e-12):
        raise ValueError('Active camera parameter outside original camera15 derivative route')
    columns = np.sort(ordered)
    permutation = np.searchsorted(ordered, columns) if np.all(np.diff(ordered) > 0) else np.argsort(ordered)
    inverse_condition = np.r_[[.01] * 3, [1.] * 5, [.001] * 3, [1.] * 4]
    chain = chain25[:15, permutation] * inverse_condition[:, None]
    if not np.isfinite(camera).all() or not np.isfinite(chain).all():
        raise ValueError('Finite unchanged camera and complete analytical camera15 chain required')
    return camera, columns, chain


def compare_strict_dataset_camera_mapping(factor, q, actual_pts):
    """Qualify all 13 existing FD columns on caller-selected current exposures.

    This helper never installs or mutates the factor's evaluation method. Its
    single-call timing is diagnostic; it is not a full solver throughput claim.
    """
    value = factor._check_q(q)
    factor._prepare_evaluation(value)
    checks = []
    for actual in actual_pts:
        actual = int(actual)
        block = factor.state.bypts[actual]
        # The baseline cache must not replace real FD work in this comparison.
        factor._camera_cache.pop((actual, True), None)
        started = time.perf_counter()
        baseline = factor._camera(value, actual, True)
        baseline_seconds = time.perf_counter() - started
        started = time.perf_counter()
        prepared = strict_dataset_camera_mapping(block, value, actual, core_count=factor.core_count)
        prepared_seconds = time.perf_counter() - started
        if (baseline[0].dtype != prepared[0].dtype or baseline[0].tobytes() != prepared[0].tobytes()
                or not np.array_equal(baseline[1], prepared[1])):
            raise ValueError('Prepared camera changed actual nominal camera bytes or complete FD columns')
        error = np.abs(baseline[2] - prepared[2])
        scale = np.maximum(1., np.abs(baseline[2]))
        if np.any(error > 2e-7 * scale):
            raise ValueError('Prepared analytical camera mapping failed original complete-column FD qualification')
        checks.append(dict(actual_pts=actual, programme_family=block.name,
            nominal_camera25_bytes_identical=True, global_columns=prepared[1].tolist(),
            absolute_errors_by_global_column=error.max(axis=0).tolist(),
            maximum_absolute_error=float(error.max()), original_FD_seconds=baseline_seconds,
            prepared_analytical_seconds=prepared_seconds, all_existing_13_columns_qualified=True,
            nominal_Pz_m=float(value[block.xyz_slots[2]])))
    if not checks:
        raise ValueError('Actual current camera exposures required for independent mapping qualification')
    return dict(schema='rinkspace.strict-dataset-original-camera-mapping-qualification.v1',
        checks=checks, passed=True, factor_method_installed=False,
        original_parameter_layout_source_pixels_and_objective_unchanged=True,
        timing_is_single_mapping_call_not_solver_throughput=True)


def install_strict_dataset_camera_execution(factor):
    """Qualify all families with the original FD before replacing this factor.

    Qualification is deferred until the driver supplies the real complete q.
    Every later evaluation still obtains the nominal camera from the original
    chart and requires its bytes to match the original analytical producer.
    """
    if hasattr(factor, 'strict_camera_execution_receipt'):
        return factor.strict_camera_execution_receipt
    original_camera = factor._camera
    original_preflight = factor.preflight
    path = Path(__file__).resolve()
    receipt = dict(schema='rinkspace.strict-dataset-qualified-camera-execution.v1',
        helper_path=str(path), helper_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        original_FD_retained_for_qualification=True,
        qualification_sampling='First, middle and last FIT exposure of every original programme family',
        complete_13_global_columns_and_nominal_camera_bytes_required=True,
        original_nominal_camera_called_and_byte_checked_at_every_new_q=True,
        source_residual_sigma_world_producers_and_parameter_chart_unchanged=True,
        prepared_mapping_installed=False, qualification=None)
    factor.strict_camera_execution_receipt = receipt

    def prepared_camera(self, q, actual, jacobian):
        if self._evaluation_q is not q:
            self._prepare_evaluation(q)
        key = (actual, bool(jacobian))
        if key in self._camera_cache:
            return self._camera_cache[key]
        if not jacobian:
            return original_camera(q, actual, False)
        nominal = original_camera(q, actual, False)[0]
        mapped, columns, chain = strict_dataset_camera_mapping(
            self.state.bypts[actual], q, actual, core_count=self.core_count)
        if nominal.dtype != mapped.dtype or nominal.shape != mapped.shape or nominal.tobytes() != mapped.tobytes():
            raise ValueError('Qualified analytical camera changed original current nominal camera bytes')
        result = (nominal, columns, chain)
        self._camera_cache[key] = result
        return result

    def qualified_preflight(self, q):
        if not receipt['prepared_mapping_installed']:
            actuals = []
            family_groups = self.family_groups
            for family, rows in family_groups.items():
                exposures = sorted({int(row['actual_pts']) for row in rows})
                if not exposures:
                    raise ValueError('Camera qualification requires every FIT family: ' + family)
                actuals.extend(exposures[i] for i in sorted({0, len(exposures)//2, len(exposures)-1}))
            qualification = compare_strict_dataset_camera_mapping(self, q, actuals)
            # No method or receipt is switched before every original column
            # and nominal camera across all families has passed its FD gate.
            self._camera_cache.clear()
            self._camera = MethodType(prepared_camera, self)
            receipt['qualification'] = qualification
            receipt['prepared_mapping_installed'] = True
        report = original_preflight(q)
        report['strict_camera_execution'] = receipt
        return report

    factor.preflight = MethodType(qualified_preflight, factor)
    return receipt
