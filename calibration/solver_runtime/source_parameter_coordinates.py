"""Reversible source-conditional coordinates; no physical quantity is fixed.

A source normal block determines a numerical combination c=p+K*l, with
K=H_pp^-1 H_pl.  The remaining l coordinates retain all their information.
The optional LM experiment changes only the local trial metric; the source
objective, nonlinear model and all physical inequalities stay canonical.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csc_matrix, csr_matrix, eye, vstack


@dataclass(frozen=True)
class ConditionalCoordinates:
    matrix: csr_matrix
    inverse: csr_matrix
    anchor_indices: np.ndarray
    coupled_indices: np.ndarray
    coefficients: np.ndarray
    report: dict

    def push_step(self, step):
        return np.asarray(self.matrix @ np.asarray(step)).ravel()

    def pull_step(self, step):
        return np.asarray(self.inverse @ np.asarray(step)).ravel()

    def pull_jacobian(self, jacobian):
        return csr_matrix(jacobian) @ self.matrix

    def pull_normal(self, normal):
        return (self.matrix.T @ csc_matrix(normal) @ self.matrix).tocsc()

    def encode(self, q, reference_q):
        return self.pull_step(np.asarray(q)-np.asarray(reference_q))

    def decode(self, coordinates, reference_q):
        return np.asarray(reference_q)+self.push_step(coordinates)


def conditional_coordinates(normal, *, anchor_indices, coupled_indices):
    """Build a square invertible shear, without rank truncation or a new prior."""
    normal=csc_matrix(normal);n=normal.shape[0]
    p=np.asarray(anchor_indices,dtype=int);l=np.asarray(coupled_indices,dtype=int)
    if (normal.shape!=(n,n) or not np.isfinite(normal.data).all() or not len(p) or not len(l)
            or len(np.unique(np.r_[p,l]))!=len(p)+len(l)
            or np.any(np.r_[p,l]<0) or np.any(np.r_[p,l]>=n)):
        raise ValueError('Finite square normal and distinct anchor/coupled coordinates required')
    block=normal[p,:][:,p].toarray();cross=normal[p,:][:,l].toarray()
    if not np.allclose(block,block.T,rtol=32*np.finfo(float).eps,atol=0.):
        raise ValueError('Symmetric source anchor normal required')
    # A singular anchor is rejected; weak coupled columns are never removed.
    factor=cho_factor((block+block.T)*.5,lower=True,check_finite=True)
    coefficient=cho_solve(factor,cross,check_finite=True)
    matrix=eye(n,format='lil');inverse=eye(n,format='lil')
    for i,row in enumerate(p):
        for j,column in enumerate(l):
            matrix[row,column]=-coefficient[i,j];inverse[row,column]=coefficient[i,j]
    matrix=matrix.tocsr();inverse=inverse.tocsr()
    transformed=(matrix.T@normal@matrix).tocsc()
    rem=transformed[p,:][:,l].toarray()
    coupled_before=normal[l,:][:,l].diagonal()
    coupled_after=transformed[l,:][:,l].diagonal()
    report=dict(schema='rinkspace.source-conditional-reversible-coordinates.v1',
        parameters=n,anchor_indices=p.tolist(),coupled_indices=l.tolist(),
        canonical_step_formula='delta_p = delta_c - K delta_l; delta_l unchanged',
        source_conditional_combination_K=coefficient.tolist(),
        anchor_coupled_cross_max_absolute=float(np.max(abs(rem),initial=0.)),
        source_conditional_coupled_diagonal=coupled_after.tolist(),
        original_coupled_diagonal=coupled_before.tolist(),
        coupled_information_retained_fraction=np.divide(coupled_after,coupled_before,
            out=np.zeros(len(l)),where=coupled_before!=0.).tolist(),
        all_coordinates_and_all_weak_directions_retained=True,
        physical_identifiability_or_hardware_measurement_not_claimed=True,
        no_gauge_inferred_from_eigenvalue_threshold=True)
    return ConditionalCoordinates(matrix,inverse,p.copy(),l.copy(),coefficient,report)


def conditional_physical_trial_step(normal, gradient, source_diagonal, *, shared_indices,
        fixed_indices=(), damping, schur_view_lm_ratio=0., matrix, lower,
        anchor_indices, coupled_indices, metric='preserve'):
    """Pull the complete trial to a reversible chart and verify canonical KKT.

    preserve: T' Q T, an equivalent solve of the existing trial.
    rebuild_lm: apply the existing Schur LM rule to T' H T and max(diag,1).
    This is an explicit numerical trust metric experiment, never new evidence.
    """
    from .full_free_physical import original_trial_matrix,exact_separable_null_coordinates,solve
    from .original_halfspace_primal import original_halfspace_primal
    normal=csc_matrix(normal);gradient=np.asarray(gradient);source_diagonal=np.asarray(source_diagonal)
    n=normal.shape[0];fixed=np.asarray(fixed_indices,dtype=int);shared=np.asarray(shared_indices,dtype=int)
    selected=np.r_[anchor_indices,coupled_indices]
    if np.intersect1d(selected,fixed).size or not np.all(np.isin(selected,shared)):
        raise ValueError('Conditional coordinates must remain in the original free shared block')
    if metric not in ('preserve','rebuild_lm'):
        raise ValueError('Explicit preserve or rebuild_lm numerical metric required')
    coordinates=conditional_coordinates(normal,anchor_indices=anchor_indices,coupled_indices=coupled_indices)
    transformed=coordinates.pull_normal(normal);g=np.asarray(coordinates.matrix.T@gradient).ravel()
    A=csr_matrix(matrix);b=np.asarray(lower)
    if A.shape!=(len(b),n):raise ValueError('All original physical rows and canonical columns required')
    newA=(A@coordinates.matrix).tocsr()
    before=original_trial_matrix(normal,gradient,source_diagonal,shared_indices=shared,
        fixed_indices=fixed,damping=damping,schur_view_lm_ratio=schur_view_lm_ratio)
    free=before['free'];T=coordinates.matrix[free,:][:,free];Ti=coordinates.inverse[free,:][:,free]
    if metric=='preserve':
        Q=(T.T@before['trial_normal']@T).tocsc()
        ridge=before['numerical_regularization_diagonal']
    else:
        diagonal=np.maximum(transformed.diagonal(),1.)
        after=original_trial_matrix(transformed,g,diagonal,shared_indices=shared,
            fixed_indices=fixed,damping=damping,schur_view_lm_ratio=schur_view_lm_ratio)
        if not np.array_equal(after['free'],free):raise ValueError('Free physical coordinate order changed')
        Q=after['trial_normal'];ridge=after['numerical_regularization_diagonal']
    source=transformed[free,:][:,free];constraint=newA[:,free]
    null=exact_separable_null_coordinates(source,g[free],constraint,b)
    value,dual,_,inner=solve(Q,g[free],constraint,b,exact_zero_coordinates=null)
    solver_step=np.zeros(n);solver_step[free]=value;step=coordinates.push_step(solver_step)
    if np.any(step[fixed]!=0.):raise ValueError('Original fixed physical coordinates changed')
    canonical_Q=(Ti.T@Q@Ti).tocsc()
    canonical_gradient=gradient[free];canonical_A=A[:,free]
    physical_step=step[free];qvalue=canonical_Q@physical_step
    stationarity=qvalue+canonical_gradient-canonical_A.T@dual
    _,_,_,primal=original_halfspace_primal(canonical_A.toarray(),b,physical_step)
    tolerance=1e-9*max(1.,float(np.max(abs(canonical_gradient))),float(np.max(abs(qvalue))))
    slack=canonical_A@physical_step-b
    complementarity=float(abs(slack)@dual)
    comp_tolerance=1e-9*max(1.,abs(float(physical_step@qvalue)),
        abs(float(canonical_gradient@physical_step)),abs(float(b@dual)))
    passed=bool(inner['numerical_original_KKT_passed']and primal['accepted']
        and np.max(abs(stationarity),initial=0.)<=tolerance and complementarity<=comp_tolerance)
    report=dict(schema='rinkspace.source-conditional-physical-trial.v1',coordinates=coordinates.report,
        numerical_metric=metric,metric_is_numerical_trust_strategy_not_physical_prior=True,
        original_source_objective_model_domain_and_q_width_unchanged=True,
        source_null_free_coordinates_exactly_eliminated=null.tolist(),
        canonical_original_primal=primal,canonical_stationarity_max_absolute=float(np.max(abs(stationarity),initial=0.)),
        canonical_stationarity_tolerance=tolerance,canonical_complementarity=complementarity,
        canonical_complementarity_tolerance=comp_tolerance,canonical_KKT_passed=passed,
        source_g_dot=float(gradient@step),source_GN_change=float(gradient@step+.5*step@(normal@step)),
        numerical_regularization_diagonal=ridge.tolist(),inner=inner,
        nonlinear_source_acceptance_and_all_original_guards_still_required=True)
    if not passed:raise ValueError('Conditional-coordinate canonical KKT verification failed: '+repr(report))
    return SimpleNamespace(step=step,report=report)

