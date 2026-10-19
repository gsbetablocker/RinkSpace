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


def _refine_restricted_identity_kkt(rows, targets, gradient, value, dual):
    """Polish an IPM candidate without discarding any restricted inequality.

    Positive IPM duals propose a small equality basis. QR avoids squaring its
    condition number. Every original inequality and the caller's complete
    original-unit KKT certificate still decide whether the result is usable.
    Failure returns the original candidate for the established fallback.
    """
    from scipy.linalg import qr,solve_triangular
    B=np.asarray(rows,float);b=np.asarray(targets,float);q=np.asarray(gradient,float)
    y=np.asarray(value,float);z=np.asarray(dual,float)
    working=np.flatnonzero(z>np.sqrt(np.finfo(float).eps)*max(1.,float(z.max())))
    history=[];seen=set();started=time.perf_counter()
    try:
        while len(working):
            identity=tuple(working.tolist())
            if identity in seen:raise RuntimeError('Restricted KKT equality basis cycled')
            seen.add(identity)
            if len(working)>len(q):raise RuntimeError('Restricted equality basis is not independent')
            W=B[working];Q,R=qr(W.T,mode='economic')
            v=solve_triangular(R.T,b[working]+W@q,lower=True)
            candidate=-q+Q@v
            previous=np.inf
            while True:
                residual=b[working]-W@candidate
                error=float(np.max(abs(residual)))
                if error==0. or error>=previous:break
                previous=error
                correction=Q@solve_triangular(R.T,residual,lower=True)
                updated=candidate+correction
                if np.array_equal(updated,candidate):break
                candidate=updated
            multiplier=solve_triangular(R,Q.T@(candidate+q),lower=False)
            if float(multiplier.min()) < -1e-11:
                index=int(np.argmin(multiplier))
                history.append(dict(removed_basis_row=int(working[index]),multiplier=float(multiplier[index])))
                working=np.delete(working,index);continue
            slack=B@candidate-b
            arithmetic=64*np.finfo(float).eps*(np.abs(B)@np.abs(candidate)+np.abs(b)+1.)
            violated=np.flatnonzero(slack < -arithmetic)
            if len(violated):
                available=np.setdiff1d(violated,working)
                if not len(available):raise RuntimeError('Restricted equality residual did not refine')
                index=int(available[np.argmin(slack[available])])
                history.append(dict(added_basis_row=index,slack=float(slack[index])))
                working=np.union1d(working,[index]);continue
            result_dual=np.zeros(len(b));result_dual[working]=multiplier
            if not np.isfinite(candidate).all() or not np.isfinite(result_dual).all():
                raise RuntimeError('Restricted equality refinement became nonfinite')
            return candidate,result_dual,dict(refined=True,equality_basis_rows=int(len(working)),
                original_restricted_rows=int(len(b)),basis_history=history,
                maximum_equality_residual=float(np.max(abs(slack[working]))),
                maximum_identity_stationarity=float(np.max(abs(candidate+q-B.T@result_dual))),
                all_inequalities_retained_for_original_unit_gate=True,
                equality_basis_does_not_retire_near_parallel_faces=True,
                arithmetic='float64 QR and residual refinement; no extended precision claim',
                seconds=time.perf_counter()-started)
        candidate=-q
        return candidate,np.zeros(len(b)),dict(refined=True,equality_basis_rows=0,
            original_restricted_rows=int(len(b)),basis_history=history,seconds=time.perf_counter()-started)
    except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
        return y,z,dict(refined=False,error_type=type(error).__name__,error=str(error),
            basis_history=history,seconds=time.perf_counter()-started)


def _linearly_constrained_shared_step(matrix, gradient, lower, upper,
                                     constraint_matrix, constraint_lower):
    """Solve all original half-spaces using reversible coordinate scaling."""
    constant=np.all(constraint_matrix==0.,axis=1)
    if np.any(constant&(constraint_lower>0.)):
        raise ValueError('An exact constant original QP half-space is infeasible')
    if np.any(constant):
        retained=np.flatnonzero(~constant)
        constrained,held,report=_bounded_shared_step(matrix,gradient,lower,upper,
            constraint_matrix[retained],constraint_lower[retained])
        from calibration.solver_runtime.original_halfspace_primal import original_halfspace_primal
        eye=np.eye(len(gradient));finite_lower=np.flatnonzero(np.isfinite(lower));finite_upper=np.flatnonzero(np.isfinite(upper))
        rows=np.vstack((constraint_matrix,eye[finite_lower],-eye[finite_upper]))
        targets=np.r_[constraint_lower,lower[finite_lower],-upper[finite_upper]]
        _,_,_,primal=original_halfspace_primal(rows,targets,constrained)
        if not primal['accepted']:raise RuntimeError('QP constant-face reduction failed complete original primal verification')
        report=dict(report,complete_original_primal=primal,linear_constraint_count=len(constraint_lower),
            exact_constant_original_halfspaces=np.flatnonzero(constant).tolist(),
            mathematical_constant_faces_retained_with_zero_dual=True)
        if 'active_linear_constraint_rows' in report:
            report['active_linear_constraint_rows']=retained[report['active_linear_constraint_rows']].tolist()
        return constrained,held,report
    value=np.linalg.solve(matrix,-gradient)
    if (np.all(value>=lower)and np.all(value<=upper)
            and np.all(constraint_matrix@value>=constraint_lower)):
        return value,[],dict(linear_constraints_inactive=True)
    import hashlib
    from importlib.metadata import version
    from pathlib import Path
    import clarabel
    from scipy.sparse import csc_matrix,triu
    from importlib.util import find_spec
    from scipy.linalg import solve_triangular
    finite_lower=np.flatnonzero(np.isfinite(lower))
    finite_upper=np.flatnonzero(np.isfinite(upper))
    eye=np.eye(len(value))
    rows=np.vstack((constraint_matrix,eye[finite_lower],-eye[finite_upper]))
    targets=np.r_[constraint_lower,lower[finite_lower],-upper[finite_upper]]
    # Uniform positive objective scaling preserves the exact QP minimizer.
    # Use its smallest diagonal magnitude to avoid erasing the source g at
    # machine precision when large LM produces a very wide diagonal range.
    objective_scale=max(1.,float(np.min(np.diag(matrix))))
    coordinate_scale=np.sqrt(objective_scale)/np.sqrt(np.diag(matrix))
    scaled_matrix=matrix*coordinate_scale[:,None]*coordinate_scale[None,:]/objective_scale
    scaled_gradient=gradient*coordinate_scale/objective_scale
    scaled_rows=rows*coordinate_scale[None,:]
    solver_row_scale=np.linalg.norm(scaled_rows,axis=1)
    factor=np.linalg.cholesky(scaled_matrix)
    whitened=solve_triangular(factor,scaled_rows.T,lower=True,check_finite=False).T
    metric=np.einsum('ij,ij->i',whitened,whitened)
    if (np.any(metric<=0.)or not np.isfinite(metric).all()
            or np.any(solver_row_scale<=0.)or not np.isfinite(solver_row_scale).all()):
        raise ValueError('Every declared linear step constraint needs a finite nonzero normal')
    row_scale=np.sqrt(metric)/np.sqrt(objective_scale)
    normalized_rows=rows/row_scale[:,None]
    # A zero affine center avoids subtracting a large unconstrained solution.
    # The translated gradient retains exactly the same original objective.
    stable_value=np.zeros_like(gradient)
    rhs=(targets-rows@stable_value)/row_scale
    tolerance=1e-11
    from calibration.solver_runtime.original_halfspace_primal import original_halfspace_primal
    # Same original QP, with only reversible coordinate/positive row
    # scaling. IPM avoids native GI cycling on near-dependent SOC faces.
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
    settings.static_regularization_enable=False;settings.dynamic_regularization_enable=False
    precision=min(tolerance*.1,np.finfo(float).eps**.875)
    for name in ('tol_feas','tol_gap_abs','tol_gap_rel','tol_infeas_abs','tol_infeas_rel'):
        setattr(settings,name,min(getattr(settings,name),precision))
    # Constraint separation keeps the complete mathematical problem.  A
    # restricted QP is returned only after its extended nonnegative dual
    # satisfies the original full-row KKT certificate.  Every failed route
    # retains the existing complete IPM as a numerical fallback.
    separation_report={}
    # This is one in-process scheduling hint, never a stored solution or
    # evidence cache.  Scope and shape changes invalidate the previous seed.
    import os
    seed_scope=os.environ.get('RINKSPACE_SOLVER_DATA_ROOT')
    seed_state=getattr(_linearly_constrained_shared_step,'_last_verified_separation_faces',None)
    warm_rows=np.empty(0,dtype=int)
    if (isinstance(seed_state,dict)and seed_state.get('shape')==rows.shape
            and seed_state.get('scope')==seed_scope):
        try:
            saved=np.asarray(seed_state.get('rows',()))
            if saved.ndim==1 and saved.dtype.kind in 'iu':
                warm_rows=np.unique(saved[(saved>=0)&(saved<len(rows))].astype(int))
        except (TypeError,ValueError,OverflowError):
            pass
    def separated_affine_step():
        begun=time.perf_counter()
        scale=np.sqrt(metric)
        translated_gradient=solve_triangular(factor,
            coordinate_scale*(matrix@stable_value+gradient)/objective_scale,
            lower=True,check_finite=False)
        affine_rows=whitened/scale[:,None]
        affine_targets=(targets-rows@stable_value)/scale
        # The zero affine center may be feasible while the already computed
        # unconstrained optimum violates original faces. Seed separation from
        # those faces; retain the zero center and unchanged translated QP.
        unconstrained_slack=rows@value-targets
        violated=np.flatnonzero(unconstrained_slack<0.)
        initial_count=min(len(value),len(violated))
        if initial_count:
            initial=violated[np.argsort(-unconstrained_slack[violated]/scale[violated])[-initial_count:]]
        else:initial=np.empty(0,dtype=int)
        selected=np.union1d(np.union1d(initial,warm_rows),
            np.arange(len(constraint_lower),len(rows)))
        history=[]
        while len(selected):
            round_start=time.perf_counter()
            restricted=clarabel.DefaultSolver(csc_matrix(np.eye(len(value))),translated_gradient,
                csc_matrix(-affine_rows[selected]),-affine_targets[selected],
                [clarabel.NonnegativeConeT(len(selected))],settings)
            result=restricted.solve()
            entry=dict(selected_rows=int(len(selected)),status=str(result.status),
                iterations=int(result.iterations),seconds=time.perf_counter()-round_start)
            history.append(entry)
            if str(result.status)not in ('Solved','AlmostSolved'):
                separation_report.update(accepted=False,reason='restricted_numerical_status',
                    rounds=history,elapsed_seconds=time.perf_counter()-begun)
                return None
            candidate=stable_value+coordinate_scale*solve_triangular(factor.T,
                np.asarray(result.x,float),lower=False,check_finite=False)
            candidate_dual=np.zeros(len(rows))
            candidate_dual[selected]=np.asarray(result.z,float)*np.sqrt(objective_scale)
            candidate_slack=rows@candidate-targets
            candidate_slack,primal_tolerance,primal_bad,primal_report=original_halfspace_primal(
                rows,targets,candidate,slack=candidate_slack)
            normalized_candidate_slack=candidate_slack/row_scale
            unselected=np.ones(len(rows),dtype=bool);unselected[selected]=False
            violated=np.flatnonzero(unselected&primal_bad)
            entry.update(all_original_rows_checked=True,
                minimum_normalized_slack=float(np.min(normalized_candidate_slack)),
                unselected_violations=int(len(violated)),
                complete_original_primal=primal_report)
            if np.any(primal_bad[selected]):
                before_primal=primal_report
                refined_value,refined_dual,refinement=_refine_restricted_identity_kkt(
                    affine_rows[selected],affine_targets[selected],translated_gradient,
                    np.asarray(result.x,float),np.asarray(result.z,float))
                candidate=stable_value+coordinate_scale*solve_triangular(factor.T,
                    refined_value,lower=False,check_finite=False)
                candidate_dual=np.zeros(len(rows))
                candidate_dual[selected]=refined_dual*np.sqrt(objective_scale)
                candidate_slack,primal_tolerance,primal_bad,primal_report=original_halfspace_primal(
                    rows,targets,candidate)
                normalized_candidate_slack=candidate_slack/row_scale
                violated=np.flatnonzero(unselected&primal_bad)
                entry.update(restricted_original_KKT_refinement=refinement,
                    original_primal_before_refinement=before_primal,
                    complete_original_primal=primal_report,
                    minimum_normalized_slack=float(np.min(normalized_candidate_slack)),
                    unselected_violations=int(len(violated)))
                if np.any(primal_bad[selected]):
                    separation_report.update(accepted=False,reason='restricted_original_selected_primal_failure',
                        rounds=history,elapsed_seconds=time.perf_counter()-begun)
                    return None
            if len(violated):
                # Batch size is a work scheduling choice, not a constraint cap.
                # Each unsuccessful round strictly adds original violated rows.
                batch=min(max(1,len(value)//2),len(violated))
                added=violated[np.argsort(normalized_candidate_slack[violated])[:batch]]
                selected=np.union1d(selected,added)
                continue
            candidate_stationarity=matrix@candidate+gradient-normalized_rows.T@candidate_dual
            candidate_kkt_tolerance=1e-9*max(1.,float(np.max(abs(gradient))),
                float(np.max(abs(matrix@candidate))))
            candidate_original_dual=candidate_dual/row_scale
            candidate_complementarity=candidate_slack*candidate_original_dual
            candidate_complementarity_tolerance=1e-9*max(1.,abs(float(candidate@matrix@candidate)),
                abs(float(gradient@candidate)),abs(float(targets@candidate_original_dual)))
            accepted=(np.isfinite(candidate).all()and np.isfinite(candidate_dual).all()
                and primal_report['accepted']
                and np.max(abs(candidate_stationarity))<=candidate_kkt_tolerance
                and np.min(candidate_dual)>=-tolerance
                and np.sum(abs(candidate_complementarity))<=candidate_complementarity_tolerance)
            separation_report.update(accepted=bool(accepted),rounds=history,
                elapsed_seconds=time.perf_counter()-begun,
                selected_row_count=int(len(selected)),complete_original_row_count=int(len(rows)),
                all_original_rows_checked_each_round=True,
                previous_verified_face_seed_count=int(len(warm_rows)),
                unselected_rows_have_exact_zero_dual_certificate=True,
                original_full_primal_dual_KKT_gate_retained=True)
            if accepted:return result,candidate,candidate_dual
            separation_report['reason']='original_full_KKT_failure'
            return None
        separation_report.update(accepted=False,reason='empty_restricted_problem',
            rounds=history,elapsed_seconds=time.perf_counter()-begun)
        return None
    try:
        accelerated=separated_affine_step()
    except (ValueError,RuntimeError)as error:
        # A restricted numerical backend failure is not a conclusion about
        # the complete problem.  Preserve the established full IPM route.
        separation_report.update(accepted=False,reason='restricted_numerical_exception',
            error_type=type(error).__name__,error=str(error))
        accelerated=None
    numerical_fallback={}
    if accelerated is None:
        solver=clarabel.DefaultSolver(triu(csc_matrix(scaled_matrix),format='csc'),scaled_gradient,
            csc_matrix(-scaled_rows/solver_row_scale[:,None]),-targets/solver_row_scale,
            [clarabel.NonnegativeConeT(len(targets))],settings)
        solved=solver.solve()
    else:
        solved,constrained,multipliers=accelerated
        numerical_fallback.update(trigger=dict(kind='verified_full_constraint_separation'),
            exact_affine_Cholesky_coordinates=True,original_H_g_all_rows_box_unchanged=True,
            elapsed_seconds=separation_report['elapsed_seconds'],
            status=str(solved.status),iterations=int(solved.iterations))
    def affine_whitened_retry(reason):
        # Exact affine Cholesky coordinates solve the same original H/g and
        # every original half-space/box. The floating center's translated
        # gradient is retained; identity H is a reversible variable change.
        begun=time.perf_counter()
        translated_gradient=solve_triangular(factor,
            coordinate_scale*(matrix@stable_value+gradient)/objective_scale,
            lower=True,check_finite=False)
        scale=np.sqrt(metric)
        fallback=clarabel.DefaultSolver(csc_matrix(np.eye(len(value))),translated_gradient,
            csc_matrix(-whitened/scale[:,None]),-(targets-rows@stable_value)/scale,
            [clarabel.NonnegativeConeT(len(targets))],settings)
        result=fallback.solve()
        numerical_fallback.update(trigger=reason,
            exact_affine_Cholesky_coordinates=True,
            original_H_g_all_rows_box_unchanged=True,
            elapsed_seconds=time.perf_counter()-begun,
            status=str(result.status),iterations=int(result.iterations))
        if str(result.status)not in ('Solved','AlmostSolved'):
            raise RuntimeError('Clarabel shared half-space QP did not return a solved numerical state: '+str(result.status))
        step=stable_value+coordinate_scale*solve_triangular(factor.T,
            np.asarray(result.x,float),lower=False,check_finite=False)
        dual=np.asarray(result.z,float)*np.sqrt(objective_scale)
        return result,step,dual
    if accelerated is not None:
        pass
    elif str(solved.status)not in ('Solved','AlmostSolved'):
        solved,constrained,multipliers=affine_whitened_retry(
            dict(kind='original_diagonal_numerical_status',status=str(solved.status)))
    else:
        constrained=coordinate_scale*np.asarray(solved.x,float)
        multipliers=np.asarray(solved.z,float)*objective_scale*row_scale/solver_row_scale
    # Keep the actual nonnegative primal-dual certificate. A nearly active
    # face can carry an essential dual even outside a narrow primal slack
    # threshold; setting it to zero can destroy original-unit stationarity.
    active_slack=(rows@constrained-targets)/row_scale
    _,primal_tolerance,primal_bad,primal_report=original_halfspace_primal(rows,targets,constrained)
    working=np.flatnonzero(abs(active_slack*row_scale)<=primal_tolerance).tolist()
    if (any(i<0 or i>=len(rows)for i in working)
            or len(set(working))!=len(working)
            or not np.isfinite(constrained).all()or not np.isfinite(multipliers).all()):
        raise RuntimeError('Official shared half-space QP returned an invalid finite KKT state')
    slack=rows@constrained-targets
    stationarity=matrix@constrained+gradient-normalized_rows.T@multipliers
    kkt_tolerance=1e-9*max(1.,float(np.max(abs(gradient))),
                         float(np.max(abs(matrix@constrained))))
    slack,primal_tolerance,primal_bad,primal_report=original_halfspace_primal(
        rows,targets,constrained,slack=slack)
    normalized_slack=slack/row_scale
    original_duals=multipliers/row_scale
    complementarity=slack*original_duals
    complementarity_tolerance=1e-9*max(1.,abs(float(constrained@matrix@constrained)),
        abs(float(gradient@constrained)),abs(float(targets@original_duals)))
    if not numerical_fallback and (not primal_report['accepted']
            or np.max(abs(stationarity))>kkt_tolerance
            or np.min(multipliers)<-tolerance
            or np.sum(abs(complementarity))>complementarity_tolerance
            or (working and np.any(abs(slack[working])>primal_tolerance[working]))):
        reason=dict(kind='original_diagonal_original_KKT_failure',
            minimum_normalized_slack=float(np.min(normalized_slack)),
            complete_original_primal=primal_report,
            maximum_stationarity=float(np.max(abs(stationarity))),
            stationarity_threshold=kkt_tolerance,
            minimum_multiplier=float(np.min(multipliers)),
            dual_threshold=tolerance,
            complementarity_absolute_sum=float(np.sum(abs(complementarity))),
            complementarity_threshold=complementarity_tolerance)
        solved,constrained,multipliers=affine_whitened_retry(reason)
        if not np.isfinite(constrained).all()or not np.isfinite(multipliers).all():
            raise RuntimeError('Official shared half-space QP returned an invalid finite KKT state')
        slack=rows@constrained-targets
        slack,primal_tolerance,primal_bad,primal_report=original_halfspace_primal(
            rows,targets,constrained,slack=slack)
        normalized_slack=slack/row_scale
        working=np.flatnonzero(abs(slack)<=primal_tolerance).tolist()
        stationarity=matrix@constrained+gradient-normalized_rows.T@multipliers
        kkt_tolerance=1e-9*max(1.,float(np.max(abs(gradient))),
                             float(np.max(abs(matrix@constrained))))
        original_duals=multipliers/row_scale
        complementarity=slack*original_duals
        complementarity_tolerance=1e-9*max(1.,abs(float(constrained@matrix@constrained)),
            abs(float(gradient@constrained)),abs(float(targets@original_duals)))
    if (not primal_report['accepted']
            or np.max(abs(stationarity))>kkt_tolerance
            or np.min(multipliers)<-tolerance
            or np.sum(abs(complementarity))>complementarity_tolerance
            or (working and np.any(abs(slack[working])>primal_tolerance[working]))):
        raise RuntimeError('Shared half-space quadratic solution failed its KKT check')
    if accelerated is not None:
        _linearly_constrained_shared_step._last_verified_separation_faces=dict(
            shape=rows.shape,scope=seed_scope,rows=tuple(working))
    count=len(constraint_lower)
    lower_rows=[i-count for i in working if count<=i<count+len(finite_lower)]
    upper_rows=[i-count-len(finite_lower)for i in working if i>=count+len(finite_lower)]
    held=np.r_[finite_lower[lower_rows],finite_upper[upper_rows]].astype(int)
    binary=Path(find_spec('clarabel.clarabel').origin).resolve()
    refined_dual=any(entry.get('restricted_original_KKT_refinement',{}).get('refined')
        for entry in separation_report.get('rounds',()))
    return constrained,held.tolist(),dict(
        constrained_quadratic_method=('official Clarabel constraint separation with original full-row primal-dual KKT'
            if accelerated is not None else 'official Clarabel primal-dual IPM on unchanged reduced LM quadratic'),
        verified_full_constraint_separation=separation_report,
        dual_Gram_route='primal-dual IPM; positive row and reversible diagonal coordinate scaling',
        complete_constraint_rows_retained=True,square_full_dual_Gram_allocated=False,
        linear_constraint_count=count,
        active_linear_constraint_rows=[i for i in working if i<count],
        active_bound_sides=[-1]*len(lower_rows)+[1]*len(upper_rows),
        constraint_minimum_normalized_slack=float(np.min(normalized_slack)),
        constraint_minimum_original_slack=float(np.min(slack)),
        complete_original_primal=primal_report,
        affine_center_kind='zero original step; exact translated source gradient',
        free_stationarity_max_absolute=float(np.max(abs(stationarity))),
        minimum_active_multiplier=float(np.min(multipliers[working]))if working else 0.,
        kkt_tolerance=kkt_tolerance,numerical_inner_iterations=int(solved.iterations),
        numerical_solver_status=str(solved.status),
        IPM_inactive_barrier_duals_zeroed_before_original_KKT=bool(refined_dual),
        restricted_equality_KKT_dual_refinement_applied=bool(refined_dual),
        returned_dual_certificate_kind=('restricted equality KKT refinement with all original inequalities checked'
            if refined_dual else 'original nonnegative backend IPM duals'),
        positive_uniform_objective_scale=objective_scale,
        original_unit_complementarity_absolute_sum=float(np.sum(abs(complementarity))),
        original_unit_complementarity_tolerance=complementarity_tolerance,
        complete_original_nonnegative_IPM_dual_certificate_checked=not refined_dual,
        complete_original_nonnegative_dual_certificate_checked=True,
        internal_static_and_dynamic_KKT_regularization_enabled=False,
        backend=dict(name='Clarabel',version=version('clarabel'),binary_path=str(binary),
                     binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()),
        original_H_gradient_constraints_bounds_and_KKT_gate_retained=True,
        internal_scaling=('positive uniform objective scale; exact x = center + D L^-T y; original translated gradient and all original-unit full dual KKT/rows'
            if numerical_fallback else 'positive uniform objective scale; x = sqrt(scale)*u/sqrt(diag(original H)); original-unit full dual KKT and all original rows'),
        **(dict(verified_same_original_QP_numerical_fallback=numerical_fallback)if numerical_fallback else {}))


def _bounded_shared_step(matrix, gradient, lower, upper,
                         constraint_matrix=None,constraint_lower=None):
    """Solve a convex shared LM step in the caller's explicit parameter box.

    Cholesky expresses the same quadratic as a small bounded least-squares
    system. BVLS re-solves each active face; no clipped camera step is returned.
    Unconstrained feasible steps keep the original solve and exact bytes.
    """
    if constraint_matrix is not None and len(constraint_lower):
        return _linearly_constrained_shared_step(matrix,gradient,lower,upper,
                                                 constraint_matrix,constraint_lower)
    value=np.linalg.solve(matrix,-gradient)
    if np.all(value>=lower)and np.all(value<=upper):return value,[],{}
    from scipy.optimize import lsq_linear
    factor=np.linalg.cholesky(matrix)
    bounded=lsq_linear(factor.T,-np.linalg.solve(factor,gradient),
        bounds=(lower,upper),method='bvls',tol=1e-12)
    value=bounded.x
    if not bounded.success or np.any(value<lower)or np.any(value>upper):
        raise RuntimeError('Declared bounded shared LM solution is not feasible')
    held=np.flatnonzero(bounded.active_mask)
    free=bounded.active_mask==0
    stationarity=matrix@value+gradient
    signed=stationarity[held]*(-bounded.active_mask[held])
    tolerance=1e-9*max(1.,float(np.max(abs(gradient))),float(np.max(abs(matrix@value))))
    free_error=float(np.max(abs(stationarity[free])))if np.any(free)else 0.
    multiplier_min=float(np.min(signed))if len(held)else 0.
    if free_error>tolerance or multiplier_min<-tolerance:
        raise RuntimeError('Declared bounded shared LM solution failed its KKT check')
    return value,held.tolist(),dict(free_stationarity_max_absolute=free_error,
        minimum_active_multiplier=multiplier_min,kkt_tolerance=tolerance,
        bounded_least_squares_method='BVLS on Cholesky of the unchanged reduced LM matrix',
        active_bound_sides=bounded.active_mask[held].astype(int).tolist(),
        numerical_inner_iterations=int(bounded.nit))


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
                            global_damping_diagonal_floor=0.,nonnegative_global_step_indices=(),
                            global_step_lower_bounds=None,global_step_upper_bounds=None,
                            global_step_constraint_matrix=None,global_step_constraint_lower=None):
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

    ``nonnegative_global_step_indices`` names original shared coordinates that
    already lie at the caller's positive-domain numerical boundary. Their
    current value is retained if the constrained step is zero; outward steps
    remain free. The remaining globals and every view are solved jointly on
    that face, rather than clipping an unconstrained camera update.
    Optional lower/upper step arrays follow the selected-global order and
    express the caller's existing parameter intervals relative to current q.
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
    raw_bounds=np.asarray(nonnegative_global_step_indices)
    if raw_bounds.size and (raw_bounds.ndim!=1 or not np.issubdtype(raw_bounds.dtype,np.integer)):
        raise ValueError('Explicit positive-bound step indices must be integers')
    bounds=raw_bounds.astype(np.int64).reshape(-1)
    if len(np.unique(bounds))!=len(bounds)or not np.all(np.isin(bounds,selected)):
        raise ValueError('Positive-bound step coordinates must be distinct selected globals')
    bound_positions=np.flatnonzero(np.isin(selected,bounds))
    lower=np.full(ng,-np.inf)if global_step_lower_bounds is None else np.asarray(global_step_lower_bounds,float).copy()
    upper=np.full(ng,np.inf)if global_step_upper_bounds is None else np.asarray(global_step_upper_bounds,float).copy()
    if lower.shape!=(ng,)or upper.shape!=(ng,)or np.isnan(lower).any()or np.isnan(upper).any():
        raise ValueError('Explicit ordered shared-step lower/upper bound arrays required')
    lower[bound_positions]=np.maximum(lower[bound_positions],0.)
    if np.any(lower>=upper):raise ValueError('Shared-step bound intervals must be nondegenerate')
    constraints=np.empty((0,ng))if global_step_constraint_matrix is None else np.asarray(global_step_constraint_matrix,float)
    constraint_lower=np.empty(0)if global_step_constraint_lower is None else np.asarray(global_step_constraint_lower,float)
    if (constraints.ndim!=2 or constraints.shape[1]!=ng or constraint_lower.shape!=(len(constraints),)
            or not np.isfinite(constraints).all()or not np.isfinite(constraint_lower).all()):
        raise ValueError('Ordered finite shared-step half-spaces A @ step >= lower required')
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
    try:du,held_positions,bound_check=_bounded_shared_step(damped,gu,lower/scales,upper/scales,
        constraints*scales[None,:],constraint_lower)
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
    bounded_positions=np.flatnonzero(np.isfinite(lower)|np.isfinite(upper))
    if len(bounded_positions):
        report['shared_parameter_bound_step']=dict(
            candidate_original_parameter_indices=selected[bounded_positions].tolist(),
            positive_domain_tangent_original_parameter_indices=bounds.tolist(),
            active_original_parameter_indices=selected[held_positions].tolist(),
            explicit_step_intervals=[dict(original_parameter_index=int(selected[i]),
                lower=None if not np.isfinite(lower[i])else float(lower[i]),
                upper=None if not np.isfinite(upper[i])else float(upper[i]))for i in bounded_positions],
            current_parameter_values_unchanged_by_linear_solver=True,
            remaining_globals_and_views_jointly_backsolved=True,
            source_H_g_and_undamped_reference_unchanged=True,**bound_check)
    if len(constraints):
        report['shared_linear_domain_step']=dict(
            inequality='declared current-domain A @ shared_step >= lower',
            source_prior_rows_added=False,current_parameter_values_unchanged=True,
            all_remaining_globals_and_views_jointly_backsolved=True,
            source_H_g_and_undamped_reference_unchanged=True,**bound_check)
    return GlobalSchurStep(step,dg,dv,s,gs,diagonal,report,permutation,inverse_permutation)
