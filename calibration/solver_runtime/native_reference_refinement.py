"""Independent same-equation correction for the failed undamped native gate.

The caller retains the original pixel threshold and owns the full source
replay. This changes neither the constrained LM equation nor trial acceptance.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numpy as np
from scipy.sparse import diags

from . import normal_equation_refinement_candidate as mathematics


def _reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def refine_undamped_native_reference(normal, gradient, diagonal, gn, reference_step,
        shared_indices, fixed_indices, *, native_replay, initial_native_error):
    """Refine each existing solution independently, then replay all original J.

    Reuse the original float64 A=H+1e-8*diag and b=-g. Each route keeps its own
    factor and correction solve; the sparse and Schur steps are never copied
    from one another. No success threshold is introduced in this helper.
    """
    started = time.perf_counter()
    gradient = np.asarray(gradient, float)
    diagonal = np.asarray(diagonal, float)
    gn = np.asarray(gn, float)
    reference_step = np.asarray(reference_step, float)
    fixed = np.asarray(fixed_indices, int)
    n = len(gradient)
    if (normal.shape != (n, n) or diagonal.shape != (n,) or gn.shape != (n,)
            or reference_step.shape != (n,)
            or not np.array_equal(diagonal, np.maximum(normal.diagonal(), 1.))):
        raise ValueError('Original complete undamped normal equation required')
    if fixed.ndim != 1 or np.any(fixed < 0) or np.any(fixed >= n):
        raise ValueError('Original fixed parameter identities required')
    equation = normal + diags(1e-8 * diagonal)
    rhs = -gradient
    full_factor = mathematics.SameEquationLU(equation, equilibrate=False)
    schur_factor = mathematics.SameEquationSchur(equation, shared_indices, equilibrate=False)
    corrected_gn, full_report = mathematics.refine_same_equation(
        equation, rhs, gn, full_factor.solve)
    corrected_reference, schur_report = mathematics.refine_same_equation(
        equation, rhs, reference_step, schur_factor.solve)
    if (full_report['equation_identity'] != schur_report['equation_identity']
            or full_report['rhs_identity'] != schur_report['rhs_identity']):
        raise ValueError('Independent reference correction changed equation identity')
    if (not np.isfinite(corrected_gn).all() or not np.isfinite(corrected_reference).all()
            or np.any(corrected_gn[fixed] != 0.) or np.any(corrected_reference[fixed] != 0.)):
        raise ValueError('Refined undamped reference must retain finite and exact fixed-zero steps')
    maximum_gn, native_error = np.asarray(native_replay(
        [corrected_gn, corrected_reference - corrected_gn]), float)
    if not np.isfinite([maximum_gn, native_error]).all() or min(maximum_gn, native_error) < 0:
        raise ValueError('Complete original native replay must be finite')
    report = dict(schema='rinkspace.same-equation-undamped-native-reference-refinement.v1',
        original_normal_identity=mathematics.system_identity(normal),
        equation_identity=full_report['equation_identity'], rhs_identity=full_report['rhs_identity'],
        original_native_reference_error_px=float(initial_native_error),
        refined_native_reference_error_px=float(native_error),
        refined_maximum_GN_native_step_px=float(maximum_gn),
        independent_full_sparse_LU=full_report, independent_global_Schur=schur_report,
        original_coordinate_system=True, reversible_equilibration_used=False,
        original_ridge_factor=1e-8, original_full_source_replay_required=True,
        original_fixed_step_values_exact_zero=True,
        constrained_QP_normal_gradient_and_diagonal_unchanged=True,
        real_source_cost_and_acceptance_unchanged=True,
        implementation_sources=[_reference(__file__), _reference(mathematics.__file__)],
        seconds=time.perf_counter() - started, visual_accuracy_accepted=False)
    return corrected_gn, corrected_reference, float(maximum_gn), float(native_error), report
