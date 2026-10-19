"""Exact public circle/material family J without enumerating programme blocks.

Only the complete admitted FIT family is refreshed. The original factor,
placement derivative and canonical permutation produce r/J; other families
remain the explicitly declared modified-Newton derivative model.
"""
from __future__ import annotations
import copy
from contextlib import nullcontext
import numpy as np
from scipy.sparse import csr_matrix,hstack
from z190_streaming_source_normal import SourceRowBlock,_checked_block
from structured_programme_source_candidate.paired_programme_chart import CanonicalCompleteProgrammeSource
from structured_programme_source_candidate.course_patterns_independent_placements import (
    SourceWithIndependentPatternPlacements,PLACEMENT_BINDING)

from wide_anchor_candidate.complete_wide_free_optics_source import CompleteWideSource
from correlated_roll_prior_candidate.explicit_bias_source import ExplicitBiasSource
from structured_programme_source_candidate.current_full_source import CurrentProgrammeSource
from z190_parallel_source_evaluation import ordered_block_evaluations
from z190_continuous_source_bundle import SOURCE_SHA

FAMILIES={
    'programme:main1':None,
    'programme:main7':None,
    'programme:secondary':None,
    'programme:low':None,
    'programme:main9':None,
    'programme:main4':None,
    'strong_high_wide_source':None,
    'shared_board_patterns':None,
    'external_main_to_low':None,
    'additional_real_native_ice_curves':'native_ice_factor',
    'additional_real_native_board_edges':'native_board_factor',
    'additional_real_native_persistent_ice_long_baseline':'persistent_ice_factor',
    'additional_native_static_board_tracks':'native_board_track_factor',
    'additional_real_native_board_circles':'circle_factor',
    'additional_real_native_board_material_points':'native_board_material_point_factor',
}

def _owners(source):
    """Inspect real wrapper links, never forwarded attribute aliases."""
    seen=set()
    while source is not None and id(source)not in seen:
        seen.add(id(source));yield source
        values=vars(source)
        source=values.get('parent',values.get('base'))

def _pad_prefix(matrix,count):
    matrix=matrix.tocsr(copy=False)
    if matrix.shape[1]>count:raise ValueError('Small source factor exceeds original parameter prefix')
    if matrix.shape[1]==count:return matrix
    return csr_matrix((matrix.data,matrix.indices,matrix.indptr),shape=(matrix.shape[0],count),copy=False)

def _fresh_complete_programme_family(chain,value,ledger):
    """One original programme owner and exact public bias/retirement operators.

    The full source iterator is never enumerated. The cloned bias wrapper is
    supplied exactly this complete family and its first public transformed
    block is consumed; nuisance/tail families are outside this callback.
    """
    bias=next((owner for owner in chain if isinstance(owner,ExplicitBiasSource)),None)
    if bias is None:return None
    programme=getattr(bias.cap.get('iter_source_blocks'),'__self__',None)
    if not isinstance(programme,CurrentProgrammeSource)or programme.actors is not None:return None
    family=ledger['family'];name=family[10:]
    owner=bias.programme.get(name)
    if owner is None or sum(b is owner for b in programme.state.blocks)!=1:return None
    core,_=bias.unpack(value[:bias.count])
    _,result=next(ordered_block_evaluations((owner,),core,True,1))
    callback=getattr(programme,'after_programme_jacobian',None)
    if callback is not None:callback()
    residual,matrix=result
    start=int(ledger['original_start']);stop=start+len(residual)
    if stop!=int(ledger['original_stop']):raise ValueError('Complete programme source family row count changed')
    raw=SourceRowBlock(residual,matrix,programme.quadratic[start:stop],programme.scales[start:stop],
        dict(family=family,original_start=start,original_stop=stop,source_sha256=SOURCE_SHA))
    isolated=copy.copy(bias);isolated.cap=dict(bias.cap)
    isolated.validated_programme=set(bias.validated_programme)
    isolated.cap['iter_source_blocks']=lambda core,jacobian:iter((raw,))
    iterator=isolated.blocks(value[:bias.count],True)
    try:result=next(iterator)
    finally:iterator.close()
    # Preserve every source-qualified role operator in its original inner to
    # outer order. Remaining wrappers only append exact zero suffix columns.
    for wrapper in reversed(chain):
        retirement=vars(wrapper).get('original_role_retirement')
        if retirement is not None and family=='programme:secondary':result=retirement.block(result,True)
        retirement=vars(wrapper).get('static_motion_role_retirement')
        if retirement is not None and retirement.block_binding['family']==family:
            result=retirement.block(result,True)
        if 'absolute_input_compaction' in vars(wrapper):result=wrapper.apply_block(result)
        if 'original_board_curve_integration' in vars(wrapper):result=wrapper.apply_block(result)
        if 'original_ice_curve_integration' in vars(wrapper):result=wrapper.apply_block(result)
    return result


def _fresh_complete_tail_family(chain,value,ledger):
    """One original pattern/external owner and original public bias operator.

    No programme, cross-target, or other tail family is evaluated.
    The surrounding original placement context is owned by the caller.
    """
    bias=next((owner for owner in chain if isinstance(owner,ExplicitBiasSource)),None)
    if bias is None:return None
    programme=getattr(bias.cap.get('iter_source_blocks'),'__self__',None)
    if not isinstance(programme,CurrentProgrammeSource):return None
    family=ledger['family']
    core,_=bias.unpack(value[:bias.count])
    if family=='shared_board_patterns':
        patterns=programme.state.patterns
        if patterns is None:return None
        residual,matrix=patterns.evaluate(core,True)
    elif family=='external_main_to_low':
        if programme.state.external is None:return None
        result=programme.external.evaluate(core,jacobian=True)
        if [r['id']for r in result.source_row_ledger]!=programme.state.external_contract[2]:
            raise ValueError('Original external rows changed')
        residual,matrix=result.residual,result.global_jacobian
    elif family.startswith('physical_target:'):
        matches=[row for row in programme.state.cross_fit if 'physical_target:'+row['id']==family]
        if len(matches)!=1:return None
        result=programme.cross(core,matches[0],True)
        residual,matrix=result[:2]
    elif family.startswith('fixed_pivot_board_side:'):
        name=family.split(':',1)[1]
        if name not in programme.state.ids:return None
        # Reuse its literal public tail operator; drop unrelated expensive
        # families. The one optical-height companion is cheap and unchanged.
        restricted=copy.copy(programme);restricted.state=copy.copy(programme.state)
        restricted.state.ids=(name,);restricted.state.cross_fit=[]
        restricted.state.external=None;restricted.state.patterns=None
        match=next((item for item in restricted.tail_evaluations(core,True)if item[0]==family),None)
        if match is None:return None
        _,residual,matrix=match
    else:return None
    start=int(ledger['original_start']);stop=start+len(residual)
    if stop!=int(ledger['original_stop']):raise ValueError('Complete pattern source family row count changed')
    raw=SourceRowBlock(residual,matrix,programme.quadratic[start:stop],programme.scales[start:stop],
        dict(family=family,original_start=start,original_stop=stop,source_sha256=SOURCE_SHA))
    isolated=copy.copy(bias);isolated.cap=dict(bias.cap)
    isolated.cap['iter_source_blocks']=lambda core,jacobian:iter((raw,))
    iterator=isolated.blocks(value[:bias.count],True)
    try:return next(iterator)
    finally:iterator.close()


def _fresh_complete_wide_family(chain,value):
    """Original wide wrapper, original integer quadrature, empty base stream."""
    wide=next((owner for owner in chain if isinstance(owner,CompleteWideSource)),None)
    if wide is None:return None
    isolated=copy.copy(wide);isolated.base=copy.copy(wide.base)
    isolated.base.blocks=lambda q,jacobian:iter(())
    iterator=isolated.blocks(value[:wide.count],True)
    try:return next(iterator)
    finally:iterator.close()


def fresh_activated_small_source_block(source,q,supplied):
    """Return exact complete family SourceRowBlock, or None for other families.

    q is the canonical solver state. Source ledger, current residual, native
    scales and quadratic roles are checked exactly before replacing any J.
    A callback must not receive evaluation coordinates in place of q.
    """
    block=SourceRowBlock(**supplied)if isinstance(supplied,dict)else supplied
    if not isinstance(block,SourceRowBlock):raise TypeError('Original SourceRowBlock required')
    family=block.row_ledger.get('family')if isinstance(block.row_ledger,dict)else None
    if not isinstance(family,str):return None
    if family not in FAMILIES and not family.startswith(('physical_target:','fixed_pivot_board_side:')):return None
    chain=tuple(_owners(source))
    canonical=next((o for o in chain if isinstance(o,CanonicalCompleteProgrammeSource)),None)
    placements=next((o for o in chain if isinstance(o,SourceWithIndependentPatternPlacements)),None)
    if canonical is None or placements is None:return None
    layout=canonical.parameter_layout
    value=layout.evaluation_q(q)
    is_programme=family.startswith('programme:')
    is_pattern=family=='shared_board_patterns'
    is_tail=is_pattern or family=='external_main_to_low'or family.startswith(('physical_target:','fixed_pivot_board_side:'))
    is_wide=family=='strong_high_wide_source'
    factor=None if is_programme or is_tail or is_wide else getattr(source,FAMILIES[family],None)
    if factor is None and not (is_programme or is_tail or is_wide):return None
    # Native appendages outside the placement wrapper already own its full
    # parameter prefix. Their original ledger must not acquire inner fields.
    factor_owner=next((owner for owner in chain if factor is not None
        and vars(owner).get(FAMILIES[family])is factor),None)
    if factor is not None and factor_owner is None:return None
    outside_placements=factor is not None and chain.index(factor_owner)<chain.index(placements)
    start=int(block.row_ledger['original_start'])
    with (nullcontext()if outside_placements else placements.pattern_placement_context(value)):
        if is_programme:fresh=_fresh_complete_programme_family(chain,value,block.row_ledger)
        elif is_tail:fresh=_fresh_complete_tail_family(chain,value,block.row_ledger)
        elif is_wide:fresh=_fresh_complete_wide_family(chain,value)
        else:fresh=factor.block(value[:factor.parameter_count],True,original_start=start)
        if fresh is None:return None
        if outside_placements:
            matrix=_pad_prefix(fresh.J,canonical.count)
            ledger=fresh.row_ledger
        else:
            matrix=_pad_prefix(fresh.J,placements.prefix_count)
            extra=placements._placement_columns(value,family)
            if extra is None:
                matrix=_pad_prefix(matrix,placements.count)
            else:
                if extra.shape!=(len(fresh.r),placements.placement_count):
                    raise ValueError('Actual placement derivative source rows changed')
                matrix=hstack((matrix,csr_matrix(extra)),format='csr')
            ledger=dict(fresh.row_ledger,
                independent_pattern_placement_binding=PLACEMENT_BINDING,
                independent_pattern_placement_parameter_indices=placements.placement_indices.tolist())
            if is_pattern:
                # Preserve the tail release after outer placement FD columns.
                bias=next(owner for owner in chain if isinstance(owner,ExplicitBiasSource))
                programme=bias.cap['iter_source_blocks'].__self__
                callback=getattr(programme,'after_shared_pattern_evaluation',None)
                if callback is not None:callback()
    matrix=_pad_prefix(matrix,canonical.count)
    if layout.additional_world_parameter_count!=0:
        matrix=matrix[:,layout.canonical_to_evaluation].tocsr()
    if not matrix.has_sorted_indices:matrix.sort_indices()
    result=SourceRowBlock(fresh.r,matrix,fresh.quadratic,fresh.scales,ledger,getattr(fresh,'loss_weights',None))
    for wrapper in reversed(chain):
        if 'native_base_curve_integration' in vars(wrapper):result=wrapper.apply_block(result)
    r,jac,mask,scales,signature=_checked_block(result,canonical.count,True)
    previous_r,_,previous_mask,previous_scales,previous_signature=_checked_block(block,canonical.count,False)
    if (signature.rows!=previous_signature.rows or signature.inventory_sha256!=previous_signature.inventory_sha256
            or not np.array_equal(r,previous_r)or not np.array_equal(mask,previous_mask)
            or not np.array_equal(scales,previous_scales)):
        raise ValueError('Exact activated family changed original FIT ledger/current residual/mask/native scales')
    return result

def make_activated_small_source_block_refresh(source):
    """Bind existing source once for solve_joint_source optional callback."""
    callback=lambda q,supplied:fresh_activated_small_source_block(source,q,supplied)
    callback.exact_row_subset_refresh=lambda q,supplied,rows:fresh_activated_small_source_rows(source,q,supplied,rows)
    return callback


"""Optional same-q full-family audit using an already-owned source/cache.

This file builds nothing and launches nothing. Call only where the existing
source owner and its exact frozen-profile replay cache are already available.
"""
import numpy as np
from z190_joint_source_solver import _frozen_profile_token
from z190_streaming_source_normal import FrozenNativeReplayCache,SourceRowBlock

def verify_cached_programme_family_refresh(source,q,cache,refresh_callback,
        families=('programme:secondary','programme:main1')):
    if not isinstance(cache,FrozenNativeReplayCache):raise TypeError('Existing exact native replay cache required')
    if not np.array_equal(np.asarray(q,float),cache.source_state):raise ValueError('Exact same source q required')
    if _frozen_profile_token(source.updater)!=cache.contract.frozen_profile_token:
        raise ValueError('Original source profiles must remain the cached frozen profile')
    wanted=set(families);reports=[]
    for old in cache.blocks:
        family=old.row_ledger.get('family')
        if family not in wanted:continue
        supplied=SourceRowBlock(old.r,None,old.quadratic,old.scales,old.row_ledger,getattr(old,'loss_weights',None))
        new=refresh_callback(np.asarray(q,float).copy(),supplied)
        if new is None:raise ValueError('Requested existing programme source owner is unsupported')
        assert new.row_ledger==old.row_ledger and new.J.shape==old.J.shape
        assert np.array_equal(new.r,old.r)and np.array_equal(new.quadratic,old.quadratic)and np.array_equal(new.scales,old.scales)
        assert np.array_equal(new.J.data,old.J.data)and np.array_equal(new.J.indices,old.J.indices)and np.array_equal(new.J.indptr,old.J.indptr)
        reports.append(dict(family=family,all_current_original_rows=len(new.r),complete_columns=new.J.shape[1],
            original_public_sameq_full_family_r_J_CSR_roles_scales_ledger_bitexact=True))
    if set(r['family']for r in reports)!=wanted:raise ValueError('Requested complete original family missing from exact cache')
    if _frozen_profile_token(source.updater)!=cache.contract.frozen_profile_token:
        raise ValueError('Public derivative refresh changed frozen original profiles')
    return dict(all_passed=True,families=reports,only_requested_original_programme_owners_recomputed=all(name.startswith('programme:')for name in wanted),
        only_requested_original_family_owners_recomputed=True,
        no_full_source_iterator_evaluation=True,no_programme_graph_constructed=True)


def fresh_activated_small_source_rows(source,q,supplied,activated_rows):
    """Exact original programme absolute records, or complete-family fallback."""
    from .activated_source_row_subset import (programme_absolute_rows,
        patch_result,sparse_full_row_view)
    block=SourceRowBlock(**supplied)if isinstance(supplied,dict)else supplied
    if not isinstance(block,SourceRowBlock):raise TypeError('Original current source block required')
    family=block.row_ledger.get('family')
    chain=tuple(_owners(source))
    canonical=next((o for o in chain if isinstance(o,CanonicalCompleteProgrammeSource)),None)
    placements=next((o for o in chain if isinstance(o,SourceWithIndependentPatternPlacements)),None)
    if canonical is None or placements is None:return None
    layout=canonical.parameter_layout;value=layout.evaluation_q(q)
    if family.startswith('programme:'):
        bias=next((owner for owner in chain if isinstance(owner,ExplicitBiasSource)),None)
        if bias is None:return None
        programme=getattr(bias.cap.get('iter_source_blocks'),'__self__',None)
        if not isinstance(programme,CurrentProgrammeSource)or programme.actors is not None:return None
        owner=bias.programme.get(family[10:])
        if owner is None or sum(b is owner for b in programme.state.blocks)!=1:return None
        # Motion exact-batch prototype is retained for diagnosis only. This
        # first production candidate uses verified absolute/record subsets;
        # any motion request keeps the previous complete-family fallback.
        if np.any(np.asarray(activated_rows)<owner.bundle.motion_rows):return None
        core,_=bias.unpack(value[:bias.count])
        with placements.pattern_placement_context(value):
            result=programme_absolute_rows(owner,core,activated_rows)
            if result is None:return None
            rows,residual,matrix,detail=result
            # Public bias/retirement transforms see the original complete
            # scalar row identity. Only the selected derivative payload exists.
            start=int(block.row_ledger['original_start']);stop=start+len(block.r)
            raw_r=np.asarray(block.r).copy();raw_r[rows]=residual
            # Correlated roll-bias transforms only original prior rows;
            # requested absolute rows do not acquire that derivative. Retain
            # the supplied current priors and exact zero suffix columns.
            fresh=SourceRowBlock(raw_r,_pad_prefix(sparse_full_row_view(rows,matrix,len(raw_r)),bias.count),
                block.quadratic,block.scales,block.row_ledger,getattr(block,'loss_weights',None))
            for wrapper in reversed(chain):
                retirement=vars(wrapper).get('original_role_retirement')
                if retirement is not None and family=='programme:secondary':fresh=retirement.block(fresh,True)
                retirement=vars(wrapper).get('static_motion_role_retirement')
                if retirement is not None and retirement.block_binding['family']==family:fresh=retirement.block(fresh,True)
            residual=np.asarray(fresh.r)[rows];matrix=_pad_prefix(fresh.J[rows],placements.count)
    else:return None
    matrix=_pad_prefix(matrix,canonical.count)
    if layout.additional_world_parameter_count!=0:matrix=matrix[:,layout.canonical_to_evaluation].tocsr()
    if not matrix.has_sorted_indices:matrix.sort_indices()
    return patch_result(block,rows,residual,matrix,detail,canonical.count)

def make_activated_small_source_row_refresh(source):
    return lambda q,supplied,activated_rows:fresh_activated_small_source_rows(source,q,supplied,activated_rows)
