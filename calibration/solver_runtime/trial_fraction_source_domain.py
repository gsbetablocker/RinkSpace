"""Restore the same source supports at the actual numerical trial fraction."""
from __future__ import annotations

import numpy as np


def attach_trial_fraction_domain(domain, *, original_constraint_lower,
        row_group_index, previous_group_violations, common_fraction):
    """Bind a pure short-step domain without changing the parent's truth caps.

    The full-step target is rho * v. At alpha it becomes
    (1 - alpha * (1 - rho)) * v, using the same parent violations and row
    ownership as phase I. Physical and protected rows have zero expansion.
    Each call starts from these original faces, never accumulated SOC cuts.
    """
    matrix=np.asarray(domain['matrix'],float)
    full_lower=np.asarray(domain['lower'],float).copy()
    original=np.asarray(original_constraint_lower,float).copy()
    owners=np.asarray(row_group_index)
    if owners.dtype.kind not in 'iu':
        raise ValueError('Original source row group indices must be integers')
    owners=owners.copy()
    previous=np.asarray(previous_group_violations,float).copy()
    rho=float(common_fraction)
    if (matrix.ndim!=2 or original.shape!=(matrix.shape[0],)
            or full_lower.shape!=original.shape or owners.shape!=original.shape
            or previous.ndim!=1 or np.any(owners < -1)
            or np.any(owners>=len(previous)) or np.any(previous<0.)
            or not np.isfinite(original).all() or not np.isfinite(full_lower).all()
            or not np.isfinite(previous).all() or not np.isfinite(matrix).all()
            or not 0.<=rho<1.):
        raise ValueError('Complete original parent support domain required')
    positive=previous>1e-8
    coefficient=np.zeros(len(original))
    owned=owners>=0
    coefficient[owned]=np.where(positive[owners[owned]],previous[owners[owned]],0.)
    if not np.array_equal(full_lower,original-rho*coefficient):
        raise ValueError('Full-step faces must match the same original phase-I contraction')
    responses={key:dict(value) for key,value in domain['curvature_responses'].items()}
    spans={};offset=0
    covered=np.zeros(len(original),bool)
    full_targets=previous.copy();full_targets[positive]*=rho
    for family,response in responses.items():
        targets=np.asarray(response['dynamic_group_limits'],float).copy()
        stop=offset+len(targets)
        if targets.ndim!=1 or not np.array_equal(targets,full_targets[offset:stop]):
            raise ValueError('Original source group target order changed')
        begin,end=response['row_start'],response['row_stop']
        local=np.asarray(response['dynamic_row_group_index'])
        if (not isinstance(begin,(int,np.integer)) or not isinstance(end,(int,np.integer))
                or not 0<=begin<=end<=len(original) or np.any(covered[begin:end])
                or local.dtype.kind not in 'iu' or local.shape!=(end-begin,)
                or np.any(local < -1) or np.any(local>=len(targets))):
            raise ValueError('Exact original source family row group indices required')
        expected=np.full(len(local),-1,dtype=int)
        owned_local=local>=0
        expected[owned_local]=offset+local[owned_local]
        if not np.array_equal(owners[begin:end],expected):
            raise ValueError('Original source family/global row ownership differs')
        covered[begin:end]=True
        spans[family]=(offset,stop)
        response['dynamic_group_limits']=targets
        response['dynamic_row_group_index']=local.copy()
        offset=stop
    if offset!=len(previous):
        raise ValueError('Every original parent truth group must belong to a family')
    if np.any((owners>=0)&~covered):
        raise ValueError('Every owned original source row must belong to a family')
    base_report=dict(domain.get('report',{}))
    matrix_reference=domain['matrix']

    def at_trial_fraction(alpha):
        alpha=float(alpha)
        if not np.isfinite(alpha) or not 0.<alpha<=1.:
            raise ValueError('Trial fraction must lie in (0, 1]')
        fraction=rho if alpha==1. else 1.-alpha*(1.-rho)
        targets=previous.copy();targets[positive]*=fraction
        trial_responses={}
        for family,response in responses.items():
            begin,end=spans[family]
            trial_responses[family]=dict(response,
                dynamic_group_limits=targets[begin:end].copy(),
                dynamic_group_target_basis='same_parent_priority_demand_at_actual_trial_fraction')
        return dict(matrix=matrix_reference,
            lower=full_lower.copy() if alpha==1. else original-fraction*coefficient,
            curvature_responses=trial_responses,
            report=dict(base_report,trial_fraction_domain=dict(
                alpha=alpha,parent_common_fraction=rho,
                demanded_common_original_violation_fraction=fraction,
                source_supports_and_parent_caps_unchanged=True,
                original_physical_and_zero_group_faces_unchanged=True,
                accumulated_full_step_SOC_cuts_reused=False)))

    return dict(domain,at_trial_fraction=at_trial_fraction)
