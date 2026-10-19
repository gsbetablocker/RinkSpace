"""Complete free-coordinate physical QP using the existing verified kernel.

The solve function is migrated without mathematical changes from the validated
full_metric_mode_qp Data prototype. It adds no source, bound or mode target.
Original-unit KKT verification never replaces the original nonlinear guards.
"""
import time
import numpy as np
from scipy.linalg import cholesky,solve_triangular,qr
from scipy.sparse import csr_matrix,csc_matrix
import clarabel

def solve(Q,g,A,b,*,E=None,f=None,exact_zero_coordinates=()):
    start=time.time();Q=csc_matrix(Q);g=np.asarray(g);A=csr_matrix(A);b=np.asarray(b)
    n=len(g);E=np.empty((0,n))if E is None else np.asarray(E);f=np.empty(0)if f is None else np.asarray(f)
    zero=np.asarray(exact_zero_coordinates,dtype=int);keep=np.setdiff1d(np.arange(n),zero)
    H=Q[keep,:][:,keep].toarray();gradient=g[keep];ineq=A[:,keep].toarray();equal=E[:,keep]
    scale=1/np.sqrt(np.diag(H));scaled=H*scale[:,None]*scale[None,:]
    L=cholesky(scaled,lower=True,check_finite=False)
    a=solve_triangular(L,gradient*scale,lower=True,check_finite=False)
    C=solve_triangular(L,(ineq*scale[None,:]).T,lower=True,check_finite=False).T
    F=solve_triangular(L,(equal*scale[None,:]).T,lower=True,check_finite=False).T
    k=len(f)
    if k:
        mode_scale=np.linalg.norm(F,axis=1)
        assert np.all(mode_scale>0)
        U,R=qr((F/mode_scale[:,None]).T,mode='full',check_finite=False)
        rank=np.linalg.matrix_rank(R[:k,:])
        if rank!=k:raise ValueError('Numerical mode equalities are not independent in source metric')
        ypart=U[:,:k]@solve_triangular(R[:k,:].T,f/mode_scale,lower=True,check_finite=False)
        Z=U[:,k:]
    else:
        mode_scale=np.empty(0);U=np.eye(len(keep));R=np.empty((0,0));ypart=np.zeros(len(keep));Z=U
    rows=-(C@Z);rhs=-b+C@ypart;linear=Z.T@(a+ypart)
    rownorm=np.linalg.norm(rows,axis=1);constant=rownorm==0.
    if np.any(constant&(rhs<0.)):
        # Exact constant mathematical rows retain the original numerical
        # precision rule; they are never relaxed or silently discarded.
        from calibration.solver_runtime.original_halfspace_primal import original_halfspace_primal
        center=np.zeros(n);center[keep]=scale*solve_triangular(L.T,ypart,lower=False,check_finite=False)
        _,_,bad,check=original_halfspace_primal(A.toarray(),b,center)
        if np.any(constant&bad):raise ValueError('Original constant halfspace infeasible after mode equality elimination')
    retained=np.flatnonzero(~constant);settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1;settings.static_regularization_enable=False;settings.dynamic_regularization_enable=False
    for key in('tol_feas','tol_gap_abs','tol_gap_rel','tol_infeas_abs','tol_infeas_rel'):setattr(settings,key,1e-12)
    result=clarabel.DefaultSolver(csc_matrix(np.eye(Z.shape[1])),linear,
        csc_matrix(rows[retained]/rownorm[retained,None]),rhs[retained]/rownorm[retained],
        [clarabel.NonnegativeConeT(len(retained))],settings).solve()
    y=ypart+Z@np.asarray(result.x);value=np.zeros(n);value[keep]=scale*solve_triangular(L.T,y,lower=False,check_finite=False)
    multiplier=np.zeros(len(b));multiplier[retained]=np.asarray(result.z)/rownorm[retained]
    if k:
        rem=y+a-C.T@multiplier
        nu=solve_triangular(R[:k,:],-U[:,:k].T@rem,lower=False,check_finite=False)/mode_scale
    else:nu=np.empty(0)
    stationarity=Q@value+g-A.T@multiplier+E.T@nu
    slack=A@value-b;mode_error=E@value-f
    from calibration.solver_runtime.original_halfspace_primal import original_halfspace_primal
    from scipy.sparse import vstack
    allA=vstack((A,csr_matrix(E),csr_matrix(-E)),format='csr');allb=np.r_[b,f,-f]
    _,_,_,primal=original_halfspace_primal(allA.toarray(),allb,value)
    tolerance=1e-9*max(1.,float(np.max(abs(g))),float(np.max(abs(Q@value))))
    complementarity=float(abs(slack)@multiplier)+float(abs(mode_error)@abs(nu))
    comp_tolerance=1e-9*max(1.,abs(float(value@(Q@value))),abs(float(g@value)),abs(float(b@multiplier)),abs(float(f@nu)))
    report=dict(schema='rinkspace.full-free-source-metric-mode-QP.v1',status=str(result.status),iterations=int(result.iterations),parameters=n,numerical_parameters=len(keep),mode_equalities=k,original_inequalities=len(b),analytically_zero_coordinates=zero.tolist(),original_full_primal=primal,mode_equality_max_error=float(np.max(abs(mode_error),initial=0.)),free_stationarity_max_absolute=float(np.max(abs(stationarity))),kkt_tolerance=tolerance,dual_minimum=float(np.min(multiplier,initial=0.)),original_unit_complementarity_absolute_sum=complementarity,original_unit_complementarity_tolerance=comp_tolerance,all_original_rows_and_complete_free_Hessian_checked=True,full_metric_cholesky_and_QR_are_reversible_coordinates=True,numerical_original_KKT_passed=bool(primal['accepted']and np.max(abs(stationarity))<=tolerance and complementarity<=comp_tolerance and np.min(multiplier,initial=0.)>=0.),seconds=time.time()-start)
    return value,multiplier,nu,report


def original_trial_matrix(normal, gradient, source_diagonal, *, shared_indices,
                          fixed_indices=(), damping, schur_view_lm_ratio=0.):
    """Form exactly the existing Schur LM system in complete free columns."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import splu
    normal=csc_matrix(normal);gradient=np.asarray(gradient);diagonal=np.asarray(source_diagonal)
    n=normal.shape[0];shared=np.asarray(shared_indices,dtype=int);fixed=np.asarray(fixed_indices,dtype=int)
    damping=float(damping);ratio=float(schur_view_lm_ratio)
    if (normal.shape!=(n,n) or gradient.shape!=(n,) or diagonal.shape!=(n,)
            or not np.isfinite(gradient).all() or not np.isfinite(normal.data).all()
            or not np.isfinite(diagonal).all() or np.any(diagonal<=0)
            or not np.isfinite(damping) or damping<=0 or ratio not in (0.,1.)):
        raise ValueError('Finite original source normal and positive existing LM strategy required')
    if (len(np.unique(shared))!=len(shared) or len(np.unique(fixed))!=len(fixed)
            or np.any(shared<0) or np.any(shared>=n) or np.any(fixed<0) or np.any(fixed>=n)
            or np.intersect1d(shared,fixed).size):
        raise ValueError('Distinct disjoint original shared and fixed coordinates required')
    free=np.setdiff1d(np.arange(n),fixed);positions=np.searchsorted(free,shared)
    local=np.flatnonzero(~np.isin(free,shared));H=normal[free,:][:,free];g=gradient[free]
    ridge=1e-8*diagonal[free];ridge[local]+=damping*ratio*diagonal[free[local]]
    base=(H+diags(ridge)).tocsc()
    reduced=base[positions,:][:,positions].toarray()
    if len(local):
        cross=base[positions,:][:,local]
        factor=splu(base[local,:][:,local].tocsc(),permc_spec='MMD_AT_PLUS_A')
        reduced-=cross@factor.solve(cross.T.toarray())
    ridge[positions]+=damping*np.maximum(np.diag(reduced),1.)
    return dict(free=free,source_normal=H,gradient=g,
        trial_normal=(H+diags(ridge)).tocsc(),numerical_regularization_diagonal=ridge)


def exact_separable_null_coordinates(normal, gradient, matrix, lower):
    """Retain weak columns; eliminate only independent exact source-null components."""
    normal=csr_matrix(normal);matrix=csr_matrix(matrix);gradient=np.asarray(gradient);lower=np.asarray(lower)
    candidates={j for j in range(len(gradient)) if gradient[j]==0.
        and np.max(abs(normal.getrow(j).data),initial=0.)==0.}
    while True:
        removed=set()
        for j in candidates:
            for row in matrix.getcol(j).nonzero()[0]:
                if lower[row]>0. or not set(matrix.getrow(row).indices).issubset(candidates):
                    removed.add(j);break
        if not removed:break
        candidates-=removed
    return np.asarray(sorted(candidates),dtype=int)


def physical_trial_step(normal, gradient, source_diagonal, *, shared_indices,
                        fixed_indices=(), damping, schur_view_lm_ratio=0.,
                        matrix, lower):
    """Solve the existing trial system with complete canonical physical rows."""
    from types import SimpleNamespace
    system=original_trial_matrix(normal,gradient,source_diagonal,shared_indices=shared_indices,
        fixed_indices=fixed_indices,damping=damping,schur_view_lm_ratio=schur_view_lm_ratio)
    free=system['free'];matrix=csr_matrix(matrix);lower=np.asarray(lower)
    if matrix.shape!=(len(lower),normal.shape[0]):
        raise ValueError('Every physical row must use the complete canonical column order')
    A=matrix[:,free]
    null=exact_separable_null_coordinates(system['source_normal'],system['gradient'],A,lower)
    value,dual,_,report=solve(system['trial_normal'],system['gradient'],A,lower,exact_zero_coordinates=null)
    if not report['numerical_original_KKT_passed']:
        raise ValueError('Complete original-unit physical QP KKT verification failed')
    step=np.zeros(normal.shape[0]);step[free]=value
    report.update(complete_free_physical_trial=True,free_parameter_indices=free.tolist(),
        shared_parameter_indices=np.asarray(shared_indices,dtype=int).tolist(),
        original_source_H_g_and_schur_LM_rule_unchanged=True,
        no_development_mode_targets_added=True,physical_dual=dual.tolist(),
        nonlinear_source_acceptance_and_all_original_guards_still_required=True)
    return SimpleNamespace(step=step,report=report)
