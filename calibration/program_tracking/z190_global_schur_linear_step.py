"""Opt-in elimination of complete R/f views from a frozen source normal system.

This helper does not evaluate sources, update profiles, apply a camera step, or
perform a line search. The caller owns all source rows, masks, physical guards
and acceptance. Numerical ridge is explicit linear algebra, never survey data.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from scipy.sparse import csc_matrix,diags
from scipy.sparse.linalg import splu


@dataclass
class GlobalSchurStep:
    step:np.ndarray
    global_step:np.ndarray
    view_step:np.ndarray
    reduced_normal:np.ndarray
    reduced_gradient:np.ndarray
    global_damping_diagonal:np.ndarray
    report:dict
    parameter_permutation:np.ndarray
    inverse_parameter_permutation:np.ndarray


def solve_global_schur_step(normal,gradient,*,global_count=14,global_parameter_indices=None,global_damping=0.,
                            numerical_ridge_diagonal=None,global_parameter_scales=None,
                            global_damping_diagonal_floor=0.):
    """Eliminate views exactly, damp the reduced global system, and backsolve.

    For H=[A B;B.T D], S=A-B D^-1 B.T and gs=gg-B D^-1 gv.
    The optional numerical ridge is applied once to H before this elimination.
    Global LM adds lambda*max(diag(S), declared_floor) to S only. View rows get
    no additional LM penalty. This equals solving that same regularized full H
    with the corresponding diagonal addition on its global A block.

    Optional global_parameter_scales express dg=scale*du for the small dense
    solve. The damping floor is declared in original parameter coordinates and
    transforms with scale squared, so a change of coordinates changes no step.
    No default ridge resolves a gauge silently: a singular D/S raises instead.
    Frozen profile/objective line-search and physical guard remain the caller's
    responsibility. A positive numerical ridge does not identify an XYZ gauge.

    Explicit ``global_parameter_indices`` selects any ordered, distinct original
    columns, overriding ``global_count``. Remaining original columns retain their
    ascending order. The source normal/gradient are permuted only for this linear
    calculation; ``step`` is returned in the original parameter order. Global
    scales/floors and ``global_step`` follow the declared selected-column order.
    ``parameter_permutation`` maps permuted positions to original indices;
    ``inverse_parameter_permutation`` maps original indices back to positions.
    """
    started=time.perf_counter();h=csc_matrix(normal,dtype=float);h.sum_duplicates()
    g=np.asarray(gradient,float);n=h.shape[0]
    if h.shape!=(n,n)or g.shape!=(n,):
        raise ValueError('Square normal and complete gradient required')
    if global_parameter_indices is None:
        ng=int(global_count);selected=np.arange(ng,dtype=np.int64)
    else:
        raw_indices=np.asarray(global_parameter_indices)
        if raw_indices.ndim!=1 or not np.issubdtype(raw_indices.dtype,np.integer):
            raise ValueError('Explicit shared parameter indices must be a one-dimensional integer array')
        selected=raw_indices.astype(np.int64,copy=True);ng=len(selected)
    if not 0<ng<n or np.any(selected<0)or np.any(selected>=n)or len(np.unique(selected))!=ng:
        raise ValueError('Distinct in-range shared parameter indices and a nonempty remaining block required')
    remaining_mask=np.ones(n,bool);remaining_mask[selected]=False
    remaining=np.flatnonzero(remaining_mask)
    permutation=np.r_[selected,remaining]
    inverse_permutation=np.empty(n,dtype=np.int64);inverse_permutation[permutation]=np.arange(n)
    if not np.isfinite(h.data).all()or not np.isfinite(g).all():raise ValueError('Finite frozen normal and gradient required')
    difference=h-h.T;maximum=float(np.max(abs(h.data)))if h.nnz else 0.
    asymmetry=float(np.max(abs(difference.data)))if difference.nnz else 0.
    if asymmetry>1e-10*max(1.,maximum):raise ValueError('Source normal must be symmetric before Schur elimination')
    damping=float(global_damping)
    if not np.isfinite(damping)or damping<0:raise ValueError('Finite nonnegative global LM damping required')
    ridge=np.zeros(n)if numerical_ridge_diagonal is None else np.asarray(numerical_ridge_diagonal,float)
    if ridge.shape!=(n,)or not np.isfinite(ridge).all()or np.any(ridge<0):raise ValueError('Explicit nonnegative full normal ridge diagonal required')
    scales=np.ones(ng)if global_parameter_scales is None else np.asarray(global_parameter_scales,float)
    if scales.shape!=(ng,)or not np.isfinite(scales).all()or np.any(scales<=0):raise ValueError('Positive global coordinate scales required')
    floor=np.broadcast_to(np.asarray(global_damping_diagonal_floor,float),(ng,)).copy()
    if not np.isfinite(floor).all()or np.any(floor<0):raise ValueError('Nonnegative declared damping floor required')
    regularized_original=h+diags(ridge,format='csc')
    regularized=regularized_original[permutation,:][:,permutation]
    permuted_gradient=g[permutation]
    a=regularized[:ng,:ng].toarray();b=regularized[:ng,ng:];d=regularized[ng:,ng:]
    try:factor=splu(d,permc_spec='MMD_AT_PLUS_A')
    except RuntimeError as exc:raise RuntimeError('View normal is singular; caller must declare the numerical/gauge treatment')from exc
    solved=factor.solve(np.column_stack((permuted_gradient[ng:],b.T.toarray())))
    dv0=solved[:,0];db=solved[:,1:]
    raw=a-b@db;reduced_asymmetry=float(np.max(abs(raw-raw.T)))
    s=.5*(raw+raw.T);gs=permuted_gradient[:ng]-b@dv0
    eigenvalues=np.linalg.eigvalsh(s)
    # A Gram normal can have tiny cancellation roundoff in a gauge direction.
    # The eigenvalue is reported, never presented as measured position support.
    diagonal=np.maximum(np.diag(s),floor)
    if np.any(diagonal<0):raise ValueError('Negative reduced diagonal; source Gram/gauge treatment must be audited')
    su=s*scales[:,None]*scales[None,:]
    gu=gs*scales;du_diagonal=diagonal*scales**2
    damped=su+np.diag(damping*du_diagonal)
    try:du=np.linalg.solve(damped,-gu)
    except np.linalg.LinAlgError as exc:raise RuntimeError('Reduced global normal is singular; no hidden XYZ prior was added')from exc
    dg=scales*du;dv=-dv0-db@dg
    step=np.empty(n);step[permutation]=np.r_[dg,dv]
    if not np.isfinite(step).all():raise RuntimeError('Nonfinite Schur/backsolve step')
    actual_linear_residual=regularized_original@step+g
    actual_linear_residual[selected]+=damping*diagonal*dg
    denominator=max(1.,float(np.linalg.norm(g)))
    report=dict(parameters=n,global_count=ng,complete_view_parameters=n-ng,
        global_parameter_indices=selected.tolist(),
        explicit_global_parameter_indices=global_parameter_indices is not None,
        inverse_selected_mapping=[dict(original_parameter_index=int(col),permuted_parameter_index=i)
            for i,col in enumerate(selected)],
        linear_permutation='selected global indices in declared order, then remaining original indices ascending',
        step_return_order='original parameter order',
        full_permutation_and_inverse_available_on_result=True,
        source_normal_unchanged=True,additional_global_damping=damping,
        numerical_ridge_explicit=numerical_ridge_diagonal is not None,
        numerical_ridge_is_not_a_source_prior_or_XYZ_measurement=True,
        normal_maximum_asymmetry=asymmetry,reduced_maximum_asymmetry=reduced_asymmetry,
        reduced_eigenvalues=eigenvalues.tolist(),global_parameter_scales=scales.tolist(),
        reduced_scaled_eigenvalues=np.linalg.eigvalsh(su).tolist(),
        full_equivalent_linear_residual_norm=float(np.linalg.norm(actual_linear_residual)),
        full_equivalent_linear_residual_relative_norm=float(np.linalg.norm(actual_linear_residual))/denominator,
        view_factorization='SuperLU MMD_AT_PLUS_A; no independent SPD claim',
        zero_additional_view_LM=True,source_or_profile_evaluated=False,
        line_search_or_camera_step_applied=False,seconds=time.perf_counter()-started)
    return GlobalSchurStep(step,dg,dv,s,gs,diagonal,report,permutation,inverse_permutation)
