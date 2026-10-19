"""Guarded joint source LM with recoverable, reviewable accepted states.

The caller owns geometry, source admission, normalized residuals and profiles.
This module preserves the existing Huber2/LM loop; numerical convergence is
not source accuracy. A source review may stop a candidate branch explicitly.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json,time
from pathlib import Path
from typing import Callable,Mapping
import numpy as np
from scipy.sparse import csr_matrix,diags
from scipy.sparse.linalg import spsolve

SCHEMA='rinkspace.joint-source-LM-checkpoint.v1'


def _json_default(value):
    if isinstance(value,Path):return str(value)
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f'Not JSON serializable: {type(value).__name__}')


def _json_text(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,default=_json_default)


def _write_json(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(_json_text(value),encoding='utf8');temporary.replace(path)


def source_cost(residual,quadratic):
    """Exactly the existing normalized Huber2 and quadratic source cost."""
    r=np.asarray(residual,float);mask=np.asarray(quadratic,bool);a=np.abs(r[~mask])
    return float(np.where(a<=2.,.5*a*a,2.*a-2.).sum())+.5*float(r[mask]@r[mask])


def _control_stop(output):
    path=output/'control.json'
    if not path.exists():return None
    try:control=json.loads(path.read_text(encoding='utf8'))
    except (json.JSONDecodeError,UnicodeDecodeError):
        # A writer may be replacing the control file. Only explicit, valid
        # instructions stop a branch; a partial write is not authorization.
        return None
    if not isinstance(control,dict)or control.get('stop_branch')is not True:return None
    return dict(origin='control.json',reason=str(control.get('reason','Explicit source-review stop_branch')),
        control=control,goal_paused=False)


def _check_physical(guard,q):
    if not np.isfinite(q).all():raise ValueError('Physical joint parameter domain')
    answer=guard(q)
    if answer is not None:
        if not isinstance(answer,(bool,np.bool_)):raise TypeError('Physical guard returns None/bool or raises')
        if not bool(answer):raise ValueError('Physical joint parameter domain')


class _BlockProfileUpdate:
    """The old joint loop's ordered update, including frozen profile state."""
    def __init__(self,blocks):self.blocks=tuple(blocks)
    def __call__(self,q):
        for block in self.blocks:
            block.bundle.robust_phase=True
            preliminary=block.evaluate(q,False)
            block.bundle.profile_weights=np.minimum(1.,2./np.maximum(
                np.abs(preliminary[:block.bundle.motion_rows]),1e-12)).reshape(-1,2)
    def iter_checkpoint_arrays(self):
        """Read existing arrays for hashing without copying all motion weights."""
        for i,block in enumerate(self.blocks):
            value=block.bundle.profile_weights
            yield f'weights_{i}',np.empty((0,2))if value is None else value
            yield f'present_{i}',np.array(value is not None)
            yield f'robust_{i}',np.array(bool(block.bundle.robust_phase))
    def checkpoint_arrays(self):
        return {key:value.copy()for key,value in self.iter_checkpoint_arrays()}
    def restore_checkpoint(self,arrays):
        expected={key for i in range(len(self.blocks))for key in (f'weights_{i}',f'present_{i}',f'robust_{i}')}
        if set(arrays)!=expected:raise ValueError('Profile checkpoint block inventory changed')
        for i,block in enumerate(self.blocks):
            value=arrays[f'weights_{i}']
            if value.shape not in ((0,2),(block.bundle.motion_rows//2,2)):
                raise ValueError('Profile checkpoint original motion row order/shape changed')
            if not np.isfinite(value).all()or np.any((value<0)|(value>1)):
                raise ValueError('Invalid frozen profile weights')
            block.bundle.profile_weights=value.copy()if bool(arrays[f'present_{i}'])else None
            block.bundle.robust_phase=bool(arrays[f'robust_{i}'])


def make_profile_update(blocks):
    """Return the unchanged profile callback with exact snapshot/restore hooks."""
    return _BlockProfileUpdate(blocks)


def _frozen_profile_token(update_profiles):
    """Hash frozen original arrays directly, never take a bulk weight snapshot."""
    digest=hashlib.sha256(b'rinkspace.frozen-profile-no-copy.v1')
    if update_profiles is None:
        digest.update(b'no-profile-callback');return digest.hexdigest()
    iterator=getattr(update_profiles,'iter_checkpoint_arrays',None)
    if iterator is None:
        if getattr(update_profiles,'stateless',False)is True:
            digest.update(b'explicit-stateless-callback');return digest.hexdigest()
        raise ValueError('Streaming stateful profile callback needs noncopying iter_checkpoint_arrays()')
    names=set()
    for key,array in iterator():
        key=str(key)
        if key in names:raise ValueError('Duplicate frozen profile array identity')
        names.add(key);value=np.asarray(array)
        if value.dtype.hasobject:raise ValueError('Frozen profiles must not contain objects')
        digest.update(key.encode('utf8'));digest.update(value.dtype.str.encode('ascii'))
        digest.update(str(value.shape).encode('ascii'))
        if not value.size:continue
        if value.flags.c_contiguous:
            digest.update(memoryview(value).cast('B'))
        else:
            # Bound conversion of any strided custom profile to a tiny chunk.
            # Existing motion weight arrays are contiguous and use no copies.
            for chunk in np.nditer(value,flags=['external_loop','buffered'],
                    op_flags=['readonly'],order='C',buffersize=8192):
                digest.update(memoryview(np.ascontiguousarray(chunk)).cast('B'))
    return digest.hexdigest()


@dataclass
class JointSourceResult:
    q:np.ndarray
    history:list
    status:str
    stop_reason:dict|None
    damping:float
    seconds:float
    checkpoint:Path

    def __iter__(self):
        # Existing joint callers can keep q, history, seconds = solve(...).
        yield self.q;yield self.history;yield self.seconds


def schur_trial_linear_step(normal,gradient,source_diagonal,*,global_count,
        global_parameter_indices=None,damping,schur_view_lm_ratio=0.):
    """Solve one numerical trial system, without changing its source H/g.

    Ratio zero calls the original helper with the original matrix.  Ratio one
    adds lambda*max(diag(source H),1) on every remaining original column before
    eliminating that view block.  The helper then applies its existing reduced
    shared LM diagonal.  The same explicit 1e-8 source-diagonal ridge is used in
    both modes; no row, source prior, profile or objective is added.
    """
    from z190_global_schur_linear_step import solve_global_schur_step
    ratio=float(schur_view_lm_ratio)
    if ratio not in (0.,1.):raise ValueError('Schur view LM ratio must be explicitly 0 or 1')
    diagonal=np.asarray(source_diagonal,float)
    if diagonal.shape!=(normal.shape[0],)or not np.isfinite(diagonal).all()or np.any(diagonal<=0):
        raise ValueError('Original positive source normal diagonal required')
    trial_normal=normal
    if ratio:
        selected=(np.arange(global_count,dtype=int)if global_parameter_indices is None
                  else np.asarray(global_parameter_indices,dtype=int))
        if selected.ndim!=1 or len(selected)!=global_count or len(np.unique(selected))!=len(selected)or np.any(selected<0)or np.any(selected>=len(diagonal)):
            raise ValueError('Distinct actual shared indices required for view LM complement')
        view_lm_diagonal=float(damping)*ratio*diagonal.copy()
        view_lm_diagonal[selected]=0.
        trial_normal=normal+diags(view_lm_diagonal,format='csc')
    linear=solve_global_schur_step(trial_normal,gradient,
        global_count=global_count,global_damping=damping,
        global_parameter_indices=global_parameter_indices,
        numerical_ridge_diagonal=1e-8*diagonal,global_damping_diagonal_floor=1.)
    if ratio:
        linear.report.update(
            source_normal_unchanged=True,
            helper_input_normal='original_source_H_plus_view_LM_diagonal',
            helper_adds_view_LM_internally=False,
            zero_additional_view_LM=False,
            schur_view_lm_ratio=ratio,
            view_LM_damping=float(damping)*ratio,
            view_LM_diagonal_rule='maximum(original_source_H_diagonal, 1)',
            view_LM_original_parameter_selection='complement of declared shared indices; remaining original indices ascending',
            view_LM_parameter_count=len(diagonal)-global_count,
            view_LM_maximum_diagonal=float(np.max(view_lm_diagonal)),
            numerical_damping_is_not_an_added_source_prior=True,
            undamped_GN_reference_uses_original_source_H_and_original_1e_minus_8_ridge=True)
    return linear


def accepted_schur_damping(damping,whole_step_fraction,*,backtracking_aware=False):
    """Update a numerical LM value after accepting the unchanged source cost.

    The default is the original update, including its floor.  The opt-in keeps
    that update for a whole step; an accepted shortened step increases the next
    damping by 1/alpha instead.  This is an update to the next linear system,
    not an objective row, source weight, profile, prior or parameter constraint.
    """
    damping=float(damping);alpha=float(whole_step_fraction)
    if not np.isfinite(damping)or damping<=0:raise ValueError('Positive finite accepted LM damping required')
    if not np.isfinite(alpha)or not 0<alpha<=1:raise ValueError('Accepted whole-step fraction must lie in (0, 1]')
    updated=max(damping/alpha if backtracking_aware and alpha<1 else damping*.3,1e-12)
    if not np.isfinite(updated)or updated<=0:raise ValueError('Positive finite updated LM damping required')
    return updated


def solve_joint_source(evaluate:Callable,q,quadratic,scales,output,physicalguard:Callable,*,
        update_profiles=None,checkpoint_callback=None,checkpoint_identity=None,
        checkpoint_arrays:Mapping|None=None,resume=False,diagnostic_values=None,
        initial_damping=1e-3,verbose=True,global_schur_count=None,global_schur_indices=None,
        fixed_parameter_indices=(),schur_backtracking=False,source_block_factory=None,
        schur_view_lm_ratio=0.,schur_backtracking_aware_damping=False,
        cache_native_source_replay=False,source_stage_timing=False):
    """Solve already-admitted sources without modifying their sigma or roles.

    ``evaluate(q, True)`` returns normalized residuals and a sparse Jacobian;
    ``evaluate(q, False)`` returns the same residuals. ``scales`` convert each
    normalized row's step back to native pixels (priors use 1), unchanged.
    ``physicalguard(q)`` raises ValueError/RuntimeError for an invalid trial,
    or returns False. It must not mutate q or silently project geometry.

    ``update_profiles(q)`` runs once before each linearization. For faithful
    continuation it must expose checkpoint_arrays()/restore_checkpoint(),
    as make_profile_update(blocks) does. A custom stateless callback may set
    ``stateless=True`` explicitly instead of exposing hooks.
    ``checkpoint_identity`` identifies the unchanged source graph,
    geometry schema and binding manifest; it is required for resume.

    ``checkpoint_callback(q_readonly, event)`` runs on the initial/resumed
    state and after every accepted state has been atomically saved. It may
    return JSON diagnostics, including {"stop_branch":true,"reason":...}.
    This stops this candidate branch only. The same explicit instruction in
    output/control.json is checked between iterations/trials. No automatic
    time, capacity or source-error cutoff is introduced.

    ``global_schur_count`` opts into exact elimination of the complete view
    block before damping the shared parameters. ``global_schur_indices``
    explicitly selects noncontiguous shared material states without changing
    their checkpoint layout. Source residuals,
    profiles, guards and objective acceptance are unchanged. The explicit
    numerical ridge is the same one used for the undamped GN evidence.
    ``schur_view_lm_ratio=1`` adds view-only LM to each trial normal before
    shared Schur elimination. Its default zero preserves the old numerical
    path. This damping is not a source residual/prior; GN evidence still uses
    the original source normal and the same 1e-8 numerical ridge.
    ``schur_backtracking_aware_damping=True`` changes only the next numerical
    damping after acceptance: a whole step retains lambda*.3, while a shortened
    whole-source step uses lambda/alpha.  The default retains the old update.

    Optional ``source_block_factory(q, jacobian)`` yields original ordered
    SourceRowBlocks without a whole-source J. Every block's mask/scales must
    equal its original contiguous checkpoint rows. H/g/cost accumulate per
    block; GN/Schur native-step evidence replays the same frozen r/J. Trial
    costs replay unchanged profiles and source row contracts. Stateful profile
    callbacks additionally expose noncopying iter_checkpoint_arrays(); the
    standard make_profile_update callback already does. The legacy evaluate
    path remains unchanged when this opt-in factory is absent.
    ``cache_native_source_replay`` retains only this linearization's generated
    normalized r/CSR blocks until the native GN/reference replay. Exact q,
    frozen profile, block contract and r/J hashes must match; the cache is
    released before trial source costs. No prepared graph is copied or saved.
    Optional ``source_stage_timing`` adds perf_counter records without changing
    any source computations or acceptance rule. Both options default false.
    """
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    q=np.asarray(q,float).copy();quadratic=np.asarray(quadratic,bool).copy()
    scales=np.asarray(scales,float).copy();initial_q=q.copy()
    schur_view_lm_ratio=float(schur_view_lm_ratio)
    if schur_view_lm_ratio not in (0.,1.):raise ValueError('Schur view LM ratio must be explicitly 0 or 1')
    if schur_view_lm_ratio and global_schur_count is None:
        raise ValueError('View LM opt-in requires the shared Schur strategy')
    if schur_backtracking_aware_damping and(global_schur_count is None or not schur_backtracking):
        raise ValueError('Backtracking-aware damping requires shared Schur whole-step backtracking')
    if cache_native_source_replay and source_block_factory is None:
        raise ValueError('Same-state native replay retention requires the streaming source strategy')
    if q.ndim!=1 or not q.size or not np.isfinite(q).all():raise ValueError('Finite one-dimensional current q required')
    fixed=np.asarray(fixed_parameter_indices,dtype=int).reshape(-1)
    if len(set(fixed.tolist()))!=len(fixed)or np.any(fixed<0)or np.any(fixed>=len(q)):
        raise ValueError('Explicit fixed source-state indices must be distinct and in range')
    active=np.ones(len(q),bool);active[fixed]=False
    shared_indices=None
    if global_schur_indices is not None:
        shared_indices=np.asarray(global_schur_indices,dtype=int).reshape(-1)
        if (global_schur_count is None or len(shared_indices)!=global_schur_count or
                len(set(shared_indices.tolist()))!=len(shared_indices)or
                np.any(shared_indices<0)or np.any(shared_indices>=len(q))):
            raise ValueError('Explicit shared Schur indices must match the distinct global parameter count')
    if quadratic.ndim!=1 or scales.shape!=quadratic.shape or not len(scales):raise ValueError('Residual row masks/scales differ')
    if not np.isfinite(scales).all()or np.any(scales<=0):raise ValueError('Unchanged positive native residual scales required')
    _check_physical(physicalguard,q)
    if update_profiles is not None:
        snapshot=getattr(update_profiles,'checkpoint_arrays',None)
        restore=getattr(update_profiles,'restore_checkpoint',None)
        if not (callable(snapshot)and callable(restore))and getattr(update_profiles,'stateless',False)is not True:
            raise ValueError('Stateful profile callback needs checkpoint_arrays/restore_checkpoint for faithful continuation')
    if source_block_factory is not None:
        if not callable(source_block_factory):raise TypeError('Source block factory must be callable')
        # Validate the no-copy protocol now, before any solve/large evaluate.
        _frozen_profile_token(update_profiles)
        from z190_streaming_source_normal import (SourceRowBlock,accumulate_source_normal,
            replay_native_step_maxima,replay_source_cost,sparse_gn_step)
    extras={str(k):np.asarray(v).copy()for k,v in (checkpoint_arrays or {}).items()}
    if any(value.dtype.hasobject for value in extras.values()):raise ValueError('Checkpoint auxiliary arrays must not contain objects')
    identity=_json_text(checkpoint_identity);history=[];reviews=[];damping=float(initial_damping)
    if not np.isfinite(damping)or damping<=0:raise ValueError('Positive finite LM damping required')
    checkpoint=output/'latest_accepted_state.npz';elapsed_offset=0.
    if checkpoint.exists()and not resume:raise ValueError('Existing accepted checkpoint requires resume or a new branch output')
    if resume:
        if checkpoint_identity is None:raise ValueError('Resume needs the unchanged source/model checkpoint_identity')
        with np.load(checkpoint,allow_pickle=False)as saved:
            if str(saved['schema'].item())!=SCHEMA or str(saved['identity_json'].item())!=identity:
                raise ValueError('Checkpoint source/model identity changed')
            if not np.array_equal(saved['quadratic'],quadratic)or not np.array_equal(saved['scales'],scales):
                raise ValueError('Checkpoint source row mask or native sigma/scale changed')
            if saved['q'].shape!=q.shape:raise ValueError('Checkpoint parameter/block inventory changed')
            oldextras={k[5:]:saved[k]for k in saved.files if k.startswith('aux__')}
            if set(oldextras)!=set(extras)or any(oldextras[k].dtype!=extras[k].dtype or not np.array_equal(oldextras[k],extras[k])for k in extras):
                raise ValueError('Checkpoint fixed source/block arrays changed')
            profiles={k[9:]:saved[k].copy()for k in saved.files if k.startswith('profile__')}
            if profiles:
                restore=getattr(update_profiles,'restore_checkpoint',None)
                if restore is None:raise ValueError('Frozen profiles need their restore_checkpoint callback')
                restore(profiles)
            q=saved['q'].copy();initial_q=saved['initial_q'].copy();damping=float(saved['damping'])
            history=json.loads(str(saved['history_json'].item()));reviews=json.loads(str(saved['reviews_json'].item()))
            elapsed_offset=float(saved['elapsed_seconds'])
        _check_physical(physicalguard,q)
    started=time.perf_counter()-elapsed_offset
    def elapsed():return time.perf_counter()-started
    def save():
        arrays=dict(schema=np.array(SCHEMA),identity_json=np.array(identity),q=q,initial_q=initial_q,
            quadratic=quadratic,scales=scales,damping=np.array(damping),history_json=np.array(_json_text(history)),
            reviews_json=np.array(_json_text(reviews)),elapsed_seconds=np.array(elapsed()))
        arrays.update({'aux__'+k:v for k,v in extras.items()})
        snapshot=getattr(update_profiles,'checkpoint_arrays',None)
        if snapshot is not None:
            arrays.update({'profile__'+str(k):np.asarray(v)for k,v in snapshot().items()})
        temporary=checkpoint.with_name(checkpoint.name+'.tmp')
        with temporary.open('wb')as stream:np.savez_compressed(stream,**arrays)
        temporary.replace(checkpoint)
        _write_json(output/'iterations.json',history);_write_json(output/'source_reviews.json',reviews)
    def review(kind,record):
        if checkpoint_callback is None:return None
        readonly=q.copy();readonly.setflags(write=False)
        event=dict(record or {},checkpoint_kind=kind,checkpoint=str(checkpoint),
            elapsed_seconds=elapsed(),numerical_convergence_is_not_accuracy=True)
        response=checkpoint_callback(readonly,event)
        if response is not None:
            # Persist the source result with the same recoverable state.
            reviews.append(dict(event=event,result=response));save()
        if isinstance(response,dict)and response.get('stop_branch')is True:
            return dict(origin='checkpoint_source_review',reason=str(response.get('reason','Explicit source-review stop_branch')),
                review=response,goal_paused=False)
        return None
    def checked_evaluate(state,jacobian):
        result=evaluate(state,jacobian)
        residual=np.asarray(result[0]if jacobian else result,float)
        if residual.shape!=scales.shape or not np.isfinite(residual).all():raise ValueError('Source residual row inventory/finite values changed')
        if not jacobian:return residual
        jac=csr_matrix(result[1])
        if jac.shape!=(len(scales),len(q))or not np.isfinite(jac.data).all():raise ValueError('Source Jacobian row/block inventory changed')
        return residual,jac
    def checked_source_blocks(state,jacobian):
        cursor=0
        for supplied in source_block_factory(state,jacobian):
            block=SourceRowBlock(**supplied)if isinstance(supplied,Mapping)else supplied
            if not isinstance(block,SourceRowBlock):raise TypeError('Original SourceRowBlock required')
            r=np.asarray(block.r,float)
            if r.ndim!=1:raise ValueError('Original source block residual must be one-dimensional')
            stop=cursor+len(r)
            if stop>len(scales)or not np.array_equal(np.asarray(block.quadratic,bool),quadratic[cursor:stop]):
                raise ValueError('Original ordered source quadratic mask changed')
            if not np.array_equal(np.asarray(block.scales,float),scales[cursor:stop]):
                raise ValueError('Original ordered source native sigma/scale changed')
            yield block
            cursor=stop
            del supplied,block,r
        if cursor!=len(scales):raise ValueError('Streaming original source row inventory changed')
    def check_frozen_profiles(token):
        if _frozen_profile_token(update_profiles)!=token:
            raise RuntimeError('Original frozen profile arrays changed during source replay')
    save();stop=_control_stop(output)
    if stop is None:stop=review('resumed'if resume else'initial',None)
    status='stopped_by_control'if stop is not None and stop['origin']=='control.json'else'stopped_by_source_review'
    while stop is None:
        stop=_control_stop(output)
        if stop is not None:status='stopped_by_control';break
        stage_times={}if source_stage_timing else None
        stage_begin=time.perf_counter()if source_stage_timing else None
        if update_profiles is not None:update_profiles(q)
        if stage_times is not None:stage_times['profile_update_seconds']=time.perf_counter()-stage_begin
        stop=_control_stop(output)
        if stop is not None:status='stopped_by_control';save();break
        streaming_report=None
        stage_begin=time.perf_counter()if source_stage_timing else None
        if source_block_factory is None:
            residual,jac=checked_evaluate(q,True);before=source_cost(residual,quadratic)
        else:
            profile_token=_frozen_profile_token(update_profiles)
            accumulated=accumulate_source_normal(lambda:checked_source_blocks(q,True),len(q),
                expected_row_count=len(scales),frozen_profile_token=profile_token,
                fixed_parameter_indices=fixed,retain_native_replay=cache_native_source_replay,
                source_state=q if cache_native_source_replay else None)
            check_frozen_profiles(profile_token)
            before=accumulated.cost
            streaming_report=dict(accumulated.report,frozen_profile_token=profile_token,
                peak_process_memory_measured_by_solver=False,
                input_CSR_bytes_is_not_total_process_peak=True,
                source_alignment_accuracy_not_implied=True)
        stop=_control_stop(output)
        if stop is not None:
            if source_block_factory is not None:accumulated.native_replay_cache=None
            status='stopped_by_control';save();break
        if source_block_factory is None:
            weights=np.minimum(1.,2./np.maximum(np.abs(residual),1e-12));weights[quadratic]=1.
            jw=jac.multiply(np.sqrt(weights)[:,None]);normal=(jw.T@jw).tocsc();gradient=np.asarray(jac.T@(weights*residual)).ravel()
        else:
            normal,gradient=accumulated.normal,accumulated.gradient
        if stage_times is not None:stage_times['source_linearization_and_normal_seconds']=time.perf_counter()-stage_begin
        if len(fixed)and source_block_factory is None:
            # Algebraic elimination of explicitly unobserved held view
            # states. The identity block enforces a zero step; it adds no
            # residual, prior, or source information to the objective.
            free=diags(active.astype(float))
            normal=(free@normal@free+diags((~active).astype(float))).tocsc()
            gradient[fixed]=0.
        diagonal=np.maximum(normal.diagonal(),1.)
        stage_begin=time.perf_counter()if source_stage_timing else None
        gn=spsolve(normal+diags(1e-8*diagonal),-gradient,permc_spec='MMD_AT_PLUS_A')
        if stage_times is not None:stage_times['undamped_GN_solve_seconds']=time.perf_counter()-stage_begin
        if not np.isfinite(gn).all():raise RuntimeError('Nonfinite undamped GN evidence; accepted checkpoint preserved')
        gn_step=None if source_block_factory is not None else float(np.max(np.abs((jac@gn)*scales)))
        accepted=False;reason=None;after=before;trials=0;linear_report=None;accepted_damping_update=None
        if global_schur_count is not None:
            from z190_global_schur_linear_step import solve_global_schur_step
            stage_begin=time.perf_counter()if source_stage_timing else None
            reference=solve_global_schur_step(normal,gradient,
                global_count=global_schur_count,global_damping=0.,
                global_parameter_indices=shared_indices,
                numerical_ridge_diagonal=1e-8*diagonal,
                global_damping_diagonal_floor=1.)
            if stage_times is not None:stage_times['undamped_global_reference_solve_seconds']=time.perf_counter()-stage_begin
            stage_begin=time.perf_counter()if source_stage_timing else None
            if source_block_factory is None:
                reference_error=float(np.max(abs((jac@(reference.step-gn))*scales)))
            else:
                check_frozen_profiles(profile_token)
                try:
                    gn_step,reference_error=replay_native_step_maxima(lambda:checked_source_blocks(q,True),
                        [gn,reference.step-gn],accumulated.contract,frozen_profile_token=profile_token,
                        native_replay_cache=accumulated.native_replay_cache,source_state=q)
                finally:accumulated.native_replay_cache=None
                check_frozen_profiles(profile_token)
            if stage_times is not None:stage_times['native_step_replay_seconds']=time.perf_counter()-stage_begin
            if reference_error>1e-3:
                raise RuntimeError('Actual full source Schur and sparse GN steps disagree')
        elif source_block_factory is not None:
            check_frozen_profiles(profile_token)
            stage_begin=time.perf_counter()if source_stage_timing else None
            try:
                gn_step=replay_native_step_maxima(lambda:checked_source_blocks(q,True),gn,
                    accumulated.contract,frozen_profile_token=profile_token,
                    native_replay_cache=accumulated.native_replay_cache,source_state=q)
            finally:accumulated.native_replay_cache=None
            check_frozen_profiles(profile_token)
            if stage_times is not None:stage_times['native_step_replay_seconds']=time.perf_counter()-stage_begin
        trial_records=[];schur_full_step=None
        trial_cost_total_seconds=0.
        for trial in range(12):
            stop=_control_stop(output)
            if stop is not None:status='stopped_by_control';break
            trials=trial+1
            if global_schur_count is None:
                step=spsolve(normal+diags(damping*diagonal),-gradient,permc_spec='MMD_AT_PLUS_A')
            else:
                if schur_full_step is None or not schur_backtracking:
                    linear=schur_trial_linear_step(normal,gradient,diagonal,
                        global_count=global_schur_count,damping=damping,
                        global_parameter_indices=shared_indices,
                        schur_view_lm_ratio=schur_view_lm_ratio)
                    schur_full_step=linear.step;linear_report=linear.report
                alpha=2.**(-trial)if schur_backtracking else 1.
                step=alpha*schur_full_step
            nextq=q+step
            trial_cost_seconds=0.
            try:
                _check_physical(physicalguard,nextq)
                trial_cost_begin=time.perf_counter()if source_stage_timing else None
                try:
                    if source_block_factory is None:
                        after=source_cost(checked_evaluate(nextq,False),quadratic)
                    else:
                        check_frozen_profiles(profile_token)
                        after=replay_source_cost(lambda:checked_source_blocks(nextq,False),
                            accumulated.contract,frozen_profile_token=profile_token)
                        check_frozen_profiles(profile_token)
                finally:
                    if source_stage_timing:
                        # Include rejected source/domain evaluations as well;
                        # timing never changes their original exception path.
                        trial_cost_seconds=time.perf_counter()-trial_cost_begin
                        trial_cost_total_seconds+=trial_cost_seconds
                if after<before-1e-10*max(1.,before):
                    if schur_backtracking_aware_damping:
                        # Validate the next numerical value before committing
                        # an accepted state or its accepted trial record.
                        next_damping=accepted_schur_damping(damping,alpha,backtracking_aware=True)
                    if global_schur_count is not None:
                        trial_records.append(dict(trial=trials,damping=damping,accepted=True,cost=after,
                            whole_source_step_fraction=alpha,shared_step=step[:global_schur_count].tolist()if shared_indices is None else step[shared_indices].tolist()))
                        if source_stage_timing:trial_records[-1]['source_cost_seconds']=trial_cost_seconds
                    q=nextq;accepted=True
                    if schur_backtracking_aware_damping:
                        previous_damping=damping
                        damping=next_damping
                        accepted_damping_update=dict(previous_damping=previous_damping,
                            accepted_whole_source_step_fraction=alpha,next_damping=damping,
                            update_rule='lambda_divided_by_alpha'if alpha<1 else'original_lambda_times_0.3',
                            reason='A shortened accepted source step retains evidence that the current numerical linear step needed damping.',
                            source_H_g_GN_reference_objective_priors_profiles_unchanged=True)
                    else:
                        # Keep the exact original arithmetic in the default path.
                        damping=max(damping*.3,1e-12)
                    break
                reason='objective_not_lower_at_profile_precision'
            except (ValueError,RuntimeError)as exc:reason=str(exc)[:180]
            if global_schur_count is not None:
                trial_records.append(dict(trial=trials,damping=damping,accepted=False,reason=reason,
                    evaluated_trial_cost=after if reason=='objective_not_lower_at_profile_precision'else None,
                    whole_source_step_fraction=alpha,shared_step=step[:global_schur_count].tolist()if shared_indices is None else step[shared_indices].tolist()))
                if source_stage_timing:trial_records[-1]['source_cost_seconds']=trial_cost_seconds
            if not(global_schur_count is not None and schur_backtracking):damping*=10.
        if stop is not None:save();break
        record=dict(iteration=len(history),accepted=accepted,cost=before,next_cost=after if accepted else before,
            trial=trials,damping=damping,failure_reason=None if accepted else reason,
            maximum_undamped_GN_native_step_px=gn_step,
            scaled_gradient_maximum=float(np.max(np.abs(gradient)/np.sqrt(diagonal))),elapsed_seconds=elapsed())
        if diagnostic_values is not None:record['source_state_diagnostics']=diagnostic_values(q.copy())
        if len(fixed):record['explicit_unobserved_held_state_indices_fixed']=fixed.tolist()
        if streaming_report is not None:record['streaming_source_linearization']=streaming_report
        if cache_native_source_replay:
            record['same_state_native_replay_cache']=dict(enabled=True,
                provenance=streaming_report['same_state_native_replay_retention'],
                released_before_trial_cost=True,source_cost_and_profile_update_not_cached=True,
                prepared_source_graph_copied=False,initial_normal_reuse=False)
        if stage_times is not None:
            stage_times['trial_source_cost_total_seconds']=trial_cost_total_seconds
            record['source_stage_timing_seconds']=stage_times
        if global_schur_count is not None:
            record['global_schur_linear_report']=linear_report
            record['actual_full_GN_reference_difference_native_px']=reference_error
            record['global_schur_trial_records']=trial_records
            if schur_view_lm_ratio:
                record['numerical_schur_view_LM']=dict(ratio=schur_view_lm_ratio,
                    original_source_H_g_GN_reference_unchanged=True,
                    source_rows_sigma_priors_profiles_unchanged=True,
                    whole_step_backtracking_and_guard_unchanged=True)
            if schur_backtracking_aware_damping:
                record['numerical_schur_backtracking_aware_damping']=dict(enabled=True,
                    accepted_update=accepted_damping_update,
                    zero_additional_source_rows_or_priors=True,
                    original_source_trialcost_and_frozen_profile_unchanged=True,
                    undamped_GN_reference_uses_original_source_H_and_original_1e_minus_8_ridge=True)
        history.append(record);save()
        if verbose:print(_json_text(dict(stage='joint_full_source_iteration',**record)),flush=True)
        if accepted:
            stop=review('accepted',record)
            if stop is not None:status='stopped_by_source_review';break
            stop=_control_stop(output)
            if stop is not None:status='stopped_by_control';break
        if not accepted:status='no_accepted_trial';break
        if gn_step<.05:status='numerical_converged';break
    save();result=JointSourceResult(q.copy(),history,status,stop,damping,elapsed(),checkpoint)
    _write_json(output/'solver_result.json',dict(status=status,stop_reason=stop,damping=damping,seconds=result.seconds,
        accepted_iterations=sum(r['accepted']for r in history),checkpoint=checkpoint,accuracy_accepted=False,
        goal_paused=False,source_sigma_roles_and_geometry_owned_by_caller=True))
    return result
