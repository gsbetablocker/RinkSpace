"""Data-only SQP feasibility priority on unchanged native support faces.

The caller supplies the already produced source Schur M/g, physical step
bounds, original support targets and a per-face source-group index.  A group
index of -1 means an immutable physical/protected face.  The same maximum
normal AND finite-coverage violation used by the nonlinear guard supplies
the epigraph coefficients.  No source row, sigma, support or model DOF is
created.  Auxiliary t exists only in this numerical subproblem.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import linprog


def _complete_original_phase_I_primal_witness(solution,B,expansion,rhs,norm):
    """Inspect every unchanged normalized face and the exact auxiliary box.

    Phase I needs a feasible common-contraction hint, not LP optimality. A
    backend success flag cannot certify that hint. All exact constant rows
    were separately verified by the caller before these nonconstant rows.
    """
    try:x=np.asarray(solution.x,float)
    except (AttributeError,TypeError,ValueError):x=np.empty(0)
    finite=bool(x.shape==(B.shape[1]+1,)and np.isfinite(x).all())
    box=bool(finite and 0.<=x[-1]<=1.)
    slack=(B@x[:-1]+expansion*x[-1]-rhs)/norm if finite else np.array([-np.inf])
    minimum=float(np.min(slack))
    passed=bool(getattr(solution,'success',False)and box and np.isfinite(slack).all()and minimum>=-1e-9)
    return dict(backend_success=bool(getattr(solution,'success',False)),
        complete_finite_primal_candidate=finite,original_auxiliary_box_feasible=box,
        original_all_face_normalized_minimum_slack=minimum if np.isfinite(minimum)else None,
        original_primal_tolerance=1e-9,LP_optimality_or_dual_certificate_required=False,passed=passed)


def contract_original_support_bounds(lower, upper,
        constraint_matrix, original_constraint_lower, row_group_index,
        current_group_violations, *, matrix=None):
    """Find a common feasible contraction for the existing driver domain.

    group violations include all source groups, including current zeros.
    Original targets must include every existing complete-stroke/endpoint
    face; current restoration limits must NOT already be subtracted here.
    The optional M only supplies an internal invertible preconditioner.
    No M/g is needed by the public driver; its existing source QP is kept.
    """
    started=time.perf_counter()
    lo=np.asarray(lower,float);hi=np.asarray(upper,float)
    A=np.asarray(constraint_matrix,float);b=np.asarray(original_constraint_lower,float)
    ids=np.asarray(row_group_index,int);v=np.asarray(current_group_violations,float)
    n=len(lo)
    if (A.shape!=(len(b),n) or ids.shape!=b.shape
            or lo.shape!=(n,) or hi.shape!=(n,) or np.any(ids < -1)
            or np.any(ids>=len(v)) or np.any(v<0.) or not np.isfinite(v).all()):
        raise ValueError('Authenticated complete native face/group domain required')
    positive=v>1e-8
    coeff=np.zeros(len(b));owned=ids>=0;coeff[owned]=np.where(positive[ids[owned]],v[ids[owned]],0.)
    constant=(np.all(A==0.,axis=1)&(coeff==0.))
    if np.any(constant&(b>0.)):
        raise ValueError('An exact constant original phase-I face is infeasible')
    if not positive.any():
        return dict(matrix=A,lower=b,restoration_required=False,
            report=dict(schema='rinkspace.original-source-SQP-feasibility-priority.v1',
                restoration_required=False))
    finite_lo=np.flatnonzero(np.isfinite(lo));finite_hi=np.flatnonzero(np.isfinite(hi))
    eye=np.eye(n)
    rows=np.vstack((A,eye[finite_lo],-eye[finite_hi]))
    rhs=np.r_[b,lo[finite_lo],-hi[finite_hi]]
    expansion=np.r_[coeff,np.zeros(len(finite_lo)+len(finite_hi))]
    if matrix is not None:
        M=np.asarray(matrix,float)
        if M.shape!=(n,n):raise ValueError('Optional unchanged source metric must match the chart')
        D=1./np.sqrt(np.diag(M));K=M*D[:,None]*D[None,:];L=np.linalg.cholesky(K)
        B=solve_triangular(L,(rows*D[None,:]).T,lower=True,check_finite=False).T
        preconditioner='reversible original-source diagonal and Cholesky whitening'
    else:
        column_norm=np.max(np.abs(rows),axis=0)
        D=np.divide(1.,column_norm,out=np.ones(n),where=column_norm>0.)
        B=rows*D[None,:]
        preconditioner='reversible original-domain column equilibration'
    norm=np.sqrt(np.einsum('ij,ij->i',B,B)+expansion*expansion)
    if not np.isfinite(norm).all():raise ValueError('Every original source/physical face must have finite coefficients')
    active=norm!=0.
    if np.any(~active&(rhs>0.)):
        raise ValueError('An exact constant original phase-I face is infeasible')
    full_rows,full_rhs,full_expansion=rows,rhs,expansion
    exact_constant_rows=np.flatnonzero(~active)
    B,expansion,rhs,norm=B[active],expansion[active],rhs[active],norm[active]
    objective=np.r_[np.zeros(n),1.]
    lp=linprog(objective,A_ub=-np.c_[B,expansion]/norm[:,None],
        b_ub=-rhs/norm,bounds=[(None,None)]*n+[(0.,1.)],method='highs-ipm',
        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    phase_I_backend='scipy HiGHS IPM';fallback_report=None
    def mapped_primal(candidate, fraction):
        coordinate=np.asarray(candidate,float)[:-1]
        direction=D*(solve_triangular(L.T,coordinate,lower=False,check_finite=False)
            if matrix is not None else coordinate)
        slack=full_rows@direction+full_expansion*fraction-full_rhs
        passed=bool(np.isfinite(direction).all()and np.isfinite(slack).all()
            and np.min(slack[active]/norm)>=-1e-9 and np.all(slack[~active]>=0.))
        return direction,slack,passed
    primary_primal=_complete_original_phase_I_primal_witness(lp,B,expansion,rhs,norm)
    if primary_primal['passed']:
        _,primary_slack,mapped_ok=mapped_primal(lp.x,float(lp.x[-1]))
        primary_primal.update(original_unit_refilled_complete_primal_passed=mapped_ok,
            original_unit_all_face_normalized_minimum_slack=float(np.min(primary_slack[active]/norm)))
        primary_primal['passed']=mapped_ok
    if not primary_primal['passed']:
        lp=_verified_same_original_conic_phaseI(B,expansion,rhs,norm,
            original_failure=str(lp.message)+'; original complete primal witness: '+str(primary_primal))
        phase_I_backend='official Clarabel same original LP after HiGHS failure or uncertified primal'
        fallback_report=lp.original_phaseI_report
        # Map back through precisely the original reversible preconditioner.
        # This witness is only a feasible contraction hint. The source QP and
        # complete nonlinear guards retain their independent original gates.
        _,slack,mapped_ok=mapped_primal(lp.x,float(lp.x[-1]))
        if not mapped_ok:
            raise ValueError('Conic phase I did not verify every original support/physical face')
        fallback_report.update(original_unit_all_face_minimum_slack=float(np.min(slack)),
            original_unit_support_face_minimum_slack=float(np.min(slack[:len(b)])),
            original_unit_box_face_minimum_slack=float(np.min(slack[len(b):]))if len(slack)>len(b)else None,
            original_unit_all_face_normalized_minimum_slack=float(np.min(slack[active]/norm)),
            original_unit_direction_refilled_without_clipping=True)
    if not lp.success or not np.isfinite(lp.x).all():
        raise ValueError('No verified contraction of the existing source-support domain: '+str(lp.message))
    minimum=float(lp.x[-1])
    if not 0.<=minimum<=1. or minimum>=1.-64*np.finfo(float).eps:
        raise ValueError('This complete linearized chart provides no common strict truth restoration direction')
    # An interior midpoint avoids accepting an approximate LP optimum as a
    # production KKT solution.  The subsequent original-unit native QP and
    # its existing complete KKT gate are authoritative.
    target=.5*(1.+minimum)
    _,midpoint_slack,midpoint_ok=mapped_primal(lp.x,target)
    if not midpoint_ok:
        raise ValueError('Phase I midpoint did not verify every original support/physical face')
    targets=b-target*coeff
    predicted=np.maximum(0.,v.copy())
    predicted[positive]=target*v[positive]
    report=dict(schema='rinkspace.original-source-SQP-feasibility-priority.v1',
        restoration_required=True,phase_I_backend=phase_I_backend,
        phase_I_reversible_preconditioner=preconditioner,
        phase_I_minimum_common_original_violation_fraction=minimum,
        phase_I_feasible_common_fraction_witness=minimum,
        demanded_common_original_violation_fraction=target,
        current_original_group_violations_px=v.tolist(),demanded_group_violations_px=predicted.tolist(),
        phase_I_failed_backend_conic_fallback=fallback_report,
        phase_I_primary_backend_complete_primal_check=primary_primal,
        phase_I_optimality_certificate_claimed=False,
        phase_I_fraction_is_a_complete_primal_witness_not_a_proven_minimum=True,
        original_midpoint_all_faces_and_physical_boxes_primal_verified=True,
        original_midpoint_all_face_normalized_minimum_slack=float(np.min(midpoint_slack[active]/norm)),
        original_midpoint_all_face_original_unit_minimum_slack=float(np.min(midpoint_slack)),
        phase_I_iterations=int(lp.nit),phase_I_normalized_minimum_slack=float(np.min((B@lp.x[:-1]+expansion*minimum-rhs)/norm)),
        exact_constant_original_faces_omitted_from_numerical_LP=exact_constant_rows.tolist(),
        exact_constant_original_faces_retained_and_mathematically_verified=True,
        all_original_source_H_g_sigma_UV_roles_supports_retained=True,
        protected_zero_groups_and_physical_bounds_unchanged=True,
        source_cost_is_secondary_within_the_contracted_truth_domain=True,
        auxiliary_t_is_not_a_model_parameter=True,accuracy_accepted=False,
        elapsed_seconds=time.perf_counter()-started)
    return dict(matrix=A,lower=targets,restoration_required=True,
        demanded_group_violations=predicted,common_fraction=target,report=report)


def contract_original_support_domain(matrix, gradient, lower, upper,
        constraint_matrix, original_constraint_lower, row_group_index,
        current_group_violations, *, qp_solver):
    """Offline verifier; live driver only needs contract_original_support_bounds."""
    result=contract_original_support_bounds(lower,upper,constraint_matrix,
        original_constraint_lower,row_group_index,current_group_violations,matrix=matrix)
    step,held,report=qp_solver(matrix,gradient,lower,upper,result['matrix'],result['lower'])
    result.update(step=step,held=held)
    result['report'].update(native_QP_original_units_allrows_KKT=report,
        original_source_quadratic_model_change=float(gradient@step+.5*step@matrix@step))
    return result


def actual_restoration_trial_decision(previous_group_violations,
        trial_group_violations, *, common_fraction, step_fraction,
        physical_guard_passed, previous_source_cost, trial_source_cost):
    """Per-group nonlinear progress before source-cost-only optimization resumes.

    Violations must be recomputed from ALL unchanged primary/full guard
    carriers including finite coverage, not residual cost or trace means.
    The standard Armijo fraction concerns numerical progress only. Use the
    existing 1e-8 pixel zero-group comparison precision for positive groups
    too: a ratio test alone demands unresolvable progress at an endpoint
    event. This is a numerical acceptance convention, not a certified bound
    on the full camera projection. Original supports and caps do not change.
    """
    old=np.asarray(previous_group_violations,float);new=np.asarray(trial_group_violations,float)
    if (old.shape!=new.shape or np.any(old<0.) or np.any(new<0.)
            or not np.isfinite(old).all() or not np.isfinite(new).all()
            or not np.isfinite([previous_source_cost,trial_source_cost]).all()):
        raise ValueError('All authenticated nonlinear truth-group violations required')
    fraction=float(common_fraction);alpha=float(step_fraction)
    if not 0.<=fraction<1. or not 0.<alpha<=1.:
        raise ValueError('Explicit strict restoration contraction and actual trial fraction required')
    tolerance=1e-8
    positive=old>tolerance;protected=~positive
    required=1.-.1*alpha*(1.-fraction)
    ratios=np.divide(new,old,out=np.zeros_like(new),where=positive)
    growth=new-old
    progress_miss=new-required*old
    protected_ok=bool(np.all(new[protected]<=tolerance))
    each_nonworsening=bool(np.all(growth[positive]<=tolerance))
    common_progress=bool(positive.any() and np.all(progress_miss[positive]<=tolerance))
    accepted=bool(physical_guard_passed and protected_ok and each_nonworsening and common_progress)
    return dict(accepted=accepted,reason='verified_each_group_truth_restoration'if accepted else'nonlinear_truth_restoration_progress_or_physical_guard_failed',
        protected_zero_groups_inside_original_support=protected_ok,
        every_previously_violated_group_nonworsening=each_nonworsening,
        maximum_actual_original_violation_fraction=float(np.max(ratios[positive]))if positive.any()else 0.,
        required_maximum_original_violation_fraction=required,
        restoration_comparison_tolerance_px=tolerance,
        comparison_tolerance_basis='same pre-existing original zero-group numerical comparison precision; not a projection error certificate',
        maximum_positive_group_growth_px=float(np.max(growth[positive]))if positive.any()else 0.,
        maximum_positive_group_progress_miss_px=float(np.max(progress_miss[positive]))if positive.any()else 0.,
        strictly_nonworsening_without_numerical_tolerance=bool(np.all(growth[positive]<=0.)),
        resolvably_improved_positive_groups=int(np.count_nonzero(positive&(growth < -tolerance))),
        measured_violations_not_clipped_or_replaced=True,
        previous_source_cost=float(previous_source_cost),trial_source_cost=float(trial_source_cost),
        full_source_cost_change=float(trial_source_cost-previous_source_cost),
        complete_source_cost_increase_is_recorded_not_accuracy=bool(trial_source_cost>previous_source_cost),
        all_source_observations_and_final_supports_retained=True,accuracy_accepted=False)


def _verified_same_original_conic_phaseI(B,expansion,rhs,norm,*,original_failure):
    """Solve the exact same normalized LP, never change final supports or caps.

    The 0-Hessian LP has free unused source coordinates. Only columns with
    identically zero constraint AND objective coefficients are eliminated.
    Clarabel's internal KKT regularization is a numerical factorization aid;
    P remains exactly zero and no model objective or physical prior is added.
    All normalized original faces, auxiliary bounds and dual evidence are
    checked/recorded independently of the backend's success label.
    """
    import clarabel,hashlib
    from importlib.util import find_spec
    from importlib.metadata import version
    from pathlib import Path
    from types import SimpleNamespace
    from scipy.sparse import csc_matrix,vstack
    n=B.shape[1];c=np.r_[np.zeros(n),1.]
    A=-np.c_[B,expansion]/norm[:,None];b=-rhs/norm
    active=np.flatnonzero(np.any(A!=0.,axis=0)|(c!=0.))
    dropped=np.setdiff1d(np.arange(n+1),active)
    if np.any(A[:,dropped]!=0.)or np.any(c[dropped]!=0.):
        raise ValueError('Only exact zero-coefficient free LP coordinates may be omitted')
    tposition=int(np.flatnonzero(active==n)[0]);bound=np.zeros((2,len(active)))
    bound[0,tposition]=-1.;bound[1,tposition]=1.
    conic_A=vstack((csc_matrix(A[:,active]),csc_matrix(bound)),format='csc')
    conic_b=np.r_[b,0.,1.]
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
    for key in ('tol_feas','tol_gap_abs','tol_gap_rel','tol_infeas_abs','tol_infeas_rel'):
        setattr(settings,key,1e-10)
    solved=clarabel.DefaultSolver(csc_matrix((len(active),len(active))),c[active],conic_A,conic_b,
        [clarabel.NonnegativeConeT(len(conic_b))],settings).solve()
    if str(solved.status)!='Solved':
        raise ValueError('No verified contraction of unchanged support domain; HiGHS: '+original_failure+'; Clarabel: '+str(solved.status))
    candidate=np.asarray(solved.x,float);dual=np.asarray(solved.z,float)
    if candidate.shape!=(len(active),)or dual.shape!=(len(conic_b),):
        raise ValueError('Conic phase I candidate must contain every original active coordinate and dual row')
    x=np.zeros(n+1);x[active]=candidate
    # A rounding projection of the numerical auxiliary t respects its exact
    # original [0,1] box; it never clips a source/model coordinate.
    original_t=float(x[-1]);x[-1]=np.clip(x[-1],0.,1.)
    slack=np.r_[b-A@x,x[-1],1.-x[-1]]
    if (not np.isfinite(x).all()or not np.isfinite(dual).all()
            or np.min(slack)<-1e-9 or np.min(dual)<-1e-10):
        raise ValueError('Conic phase I candidate failed original primal feasibility or retained dual diagnostic sign gate')
    full_A=vstack((csc_matrix(A),csc_matrix(np.r_[np.zeros(n),-1.][None,:]),
        csc_matrix(np.r_[np.zeros(n),1.][None,:])),format='csc')
    stationarity=c+full_A.T@dual
    gap=float(c@x+conic_b@dual)
    binary=Path(find_spec('clarabel.clarabel').origin).resolve()
    report=dict(backend='official Clarabel',version=version('clarabel'),binary_path=str(binary),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),original_HiGHS_failure=original_failure,
        exact_zero_free_coordinate_indices=dropped.tolist(),dropped_all_original_A_coefficients_and_objective_exact_zero=True,
        original_normalized_A_b_objective_and_bounds_unchanged=True,LP_P_exact_zero=True,
        internal_static_KKT_regularization_enabled=bool(settings.static_regularization_enable),
        internal_dynamic_KKT_regularization_enabled=bool(settings.dynamic_regularization_enable),
        auxiliary_t_before_roundoff_projection=original_t,auxiliary_t_after_projection=float(x[-1]),
        normalized_all_face_and_box_minimum_slack=float(np.min(slack)),minimum_dual=float(np.min(dual)),
        stationarity_maximum_absolute=float(np.max(abs(stationarity))),
        complementarity_maximum_absolute=float(np.max(abs(slack*dual))),primal_dual_gap=gap,
        original_all_faces_and_auxiliary_bounds_verified=True,phase_I_only_selects_a_feasible_contraction_hint=True,
        original_source_QP_full_KKT_and_nonlinear_truth_guard_remain_authoritative=True,
        source_q_UV_sigma_supports_caps_and_objective_not_changed=True)
    return SimpleNamespace(success=True,x=x,nit=int(solved.iterations),message=str(solved.status),original_phaseI_report=report)
