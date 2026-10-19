"""Guarded joint source LM with recoverable, reviewable accepted states.

The caller owns geometry, source admission, normalized residuals and profiles.
This module preserves the existing Huber2/LM loop; numerical convergence is
not source accuracy. A source review may stop a candidate branch explicitly.
"""
from __future__ import annotations
from dataclasses import dataclass
import copy,hashlib,json,time
from pathlib import Path
from typing import Callable,Mapping
import numpy as np
from calibration.solver_runtime.checkpoint_history import (HistorySegmentWriter,
    checkpoint_history_reference, read_checkpoint_history, read_checkpoint_reviews)
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
    temporary.write_text(_json_text(value),encoding='utf8')
    for attempt in range(64):
        try:
            temporary.replace(path)
            return
        except OSError as error:
            # Windows readers can briefly hold the destination against replace.
            # Leave both files recoverable if the lock persists or another error occurs.
            if getattr(error,'winerror',None)not in(5,32)or attempt==63:
                raise
            time.sleep(min(.05*2**attempt,.5))


def source_cost(residual,quadratic,loss_weights=None):
    """Exactly the existing normalized Huber2 and quadratic source cost."""
    if loss_weights is not None:
        from calibration.solver_runtime.source_loss import weighted_source_cost
        return weighted_source_cost(residual,quadratic,loss_weights)
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


def _uphill_lagged_source_trials(trials,before):
    """Detect locally uphill real costs before spending more stale-J trials.

    This only requests a fresh derivative. It never accepts a state or changes
    source cost, row weighting, physical guards or the fresh-J trial search.
    """
    if len(trials)<2:return None
    pair=trials[-2:]
    if any(row.get('reason')!='objective_not_lower_at_profile_precision'for row in pair):return None
    alpha=np.asarray([row['whole_source_step_fraction']for row in pair],float)
    values=np.asarray([row['evaluated_trial_cost']for row in pair],float)
    if not np.isfinite(values).all()or alpha[1]>1./32 or not np.isclose(alpha[0],2*alpha[1],rtol=0,atol=0):return None
    delta=values-float(before)
    if np.any(delta<=1e-10*max(1.,abs(float(before)))):return None
    slopes=delta/alpha
    if abs(slopes[0]-slopes[1])>.25*max(abs(slopes[0]),abs(slopes[1])):return None
    return dict(whole_source_fractions=alpha.tolist(),actual_source_costs=values.tolist(),
        positive_directional_cost_slopes=slopes.tolist(),
        reason='Small lagged-J trials have consistent positive real-cost slopes; refresh before more stale backtracks',
        source_cost_or_physical_guard_not_relaxed=True)


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
        global_parameter_indices=None,damping,schur_view_lm_ratio=0.,
        nonnegative_global_step_indices=(),global_step_lower_bounds=None,
        global_step_upper_bounds=None,global_step_constraint_matrix=None,
        global_step_constraint_lower=None):
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
        numerical_ridge_diagonal=1e-8*diagonal,global_damping_diagonal_floor=1.,
        nonnegative_global_step_indices=nonnegative_global_step_indices,
        global_step_lower_bounds=global_step_lower_bounds,global_step_upper_bounds=global_step_upper_bounds,
        global_step_constraint_matrix=global_step_constraint_matrix,
        global_step_constraint_lower=global_step_constraint_lower)
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


def restoration_source_model_agreement(predicted_reduction,actual_reduction):
    """Numerical trust evidence, never a veto of actual source truth progress."""
    predicted=float(predicted_reduction);actual=float(actual_reduction)
    finite=bool(np.isfinite([predicted,actual]).all())
    useful=bool(finite and max(abs(predicted),abs(actual))>1e-12)
    same_sign=bool(useful and ((predicted>0 and actual>0)or(predicted<0 and actual<0)))
    scale=max(abs(predicted),abs(actual))if useful else None
    relative_error=(float(abs(predicted/scale-actual/scale))if useful else None)
    reliable=bool(same_sign and relative_error<=.25)
    return dict(predicted_source_reduction=predicted if np.isfinite(predicted)else None,
        actual_source_reduction=actual if np.isfinite(actual)else None,
        finite_useful_prediction=useful,predicted_and_actual_same_sign=same_sign,
        symmetric_relative_source_model_error=relative_error,maximum_relative_model_error_for_lambda_decrease=.25,
        reliable_for_restoration_lambda_decrease=reliable,
        modelled_source_increase_is_not_rejected=bool(finite and predicted<0 and actual<0),
        complete_source_cost_nonincrease_is_not_a_truth_acceptance_requirement=True)


def accepted_schur_damping(damping,whole_step_fraction,*,backtracking_aware=False,truth_restoration=False,
        restoration_model_agreement=None):
    """Update a numerical LM value after accepting the unchanged source cost.

    The default is the original update, including its floor.  The opt-in keeps
    that update for a whole step; an accepted shortened step increases the next
    damping by 1/alpha instead.  This is an update to the next linear system,
    not an objective row, source weight, profile, prior or parameter constraint.
    In constrained truth restoration the active support domain can force a
    finite step regardless of LM damping. A feasibility backtrack therefore
    must not multiply damping and freeze the unconstrained source views.
    Keep lambda for a shortened accepted step; an unshortened accepted step
    restores the original numerical decrease only if its optional current
    source model agrees with the actual signed change. Without that optional
    report the historical arithmetic is unchanged. Source rows and truth
    acceptance remain unchanged; necessary modelled cost growth is allowed.
    """
    damping=float(damping);alpha=float(whole_step_fraction)
    if not np.isfinite(damping)or damping<=0:raise ValueError('Positive finite accepted LM damping required')
    if not np.isfinite(alpha)or not 0<alpha<=1:raise ValueError('Accepted whole-step fraction must lie in (0, 1]')
    if truth_restoration:
        trusted=(restoration_model_agreement is None or
            restoration_model_agreement.get('reliable_for_restoration_lambda_decrease')is True)
        updated=max(damping if alpha<1 or not trusted else damping*.3,1e-12)
    else:
        updated=max(damping/alpha if backtracking_aware and alpha<1 else damping*.3,1e-12)
    if not np.isfinite(updated)or updated<=0:raise ValueError('Positive finite updated LM damping required')
    return updated


def _sequential_measured_shared_source_curvature(q, linear, selected, domain,
        linear_options, normal, gradient, diagonal, tightening, solve_linear_step):
    """Accumulate only measured outward cuts on this unchanged source normal.

    Every measurement is relative to the original physical/source domain,
    never the previous numerical interior target.  A failed tighter QP does
    not replace the last solved direction; original backtracking and actual
    nonlinear guards remain authoritative.
    """
    original_lower=np.asarray(domain['lower'],float)
    current_lower=np.asarray(linear_options['global_step_constraint_lower'],float).copy()
    if current_lower.shape!=original_lower.shape or np.any(current_lower<original_lower):
        raise ValueError('Initial SOC targets must preserve every original step face')
    records=[];attempts=0;last_measured_report=None;stop_reason=None
    while True:
        correction=tightening(q.copy(),linear.step.copy(),selected.copy(),domain)
        if correction is None:
            stop_reason='no_measured_outward_source_face_correction';break
        proposed=np.asarray(correction['lower'],float)
        if (proposed.shape!=original_lower.shape or not np.isfinite(proposed).all()
                or np.any(proposed<original_lower)):
            raise ValueError('Curvature correction may only tighten unchanged original step faces')
        accumulated=np.maximum(current_lower,proposed)
        changed=np.flatnonzero(accumulated>current_lower)
        if not len(changed):
            stop_reason='no_strict_accumulated_target_progress';break
        options=dict(linear_options,global_step_constraint_lower=accumulated)
        attempts+=1
        try:
            corrected=solve_linear_step(normal,gradient,diagonal,**options)
        except (ValueError,RuntimeError)as exc:
            records.append(dict(attempt=attempts,accepted_numerical_direction=False,
                changed_source_step_faces=changed.tolist(),failure=str(exc),
                last_solved_direction_retained_for_original_backtracking=True))
            stop_reason='tightened_QP_failed_keep_last_verified_direction';break
        increase=accumulated-current_lower
        records.append(dict(attempt=attempts,accepted_numerical_direction=True,
            changed_source_step_faces=changed.tolist(),maximum_step_lower_increase_px=float(increase.max()),
            measured_outward_correction=correction.get('report',{})))
        linear=corrected;current_lower=accumulated
        linear_options['global_step_constraint_lower']=current_lower
        last_measured_report=correction.get('report',{})
    if last_measured_report is not None:
        linear.report['measured_shared_source_curvature_tightening']=last_measured_report
    linear.report['sequential_measured_shared_source_curvature_tightening']=dict(
        schema='rinkspace.sequential-measured-outward-source-face-SOC.v1',
        stop_reason=stop_reason,correction_QP_attempts=attempts,
        solved_corrections=sum(r['accepted_numerical_direction']for r in records),
        steps=records,original_source_H_g_J_profiles_weights_sigmas_bounds_and_nonlinear_guards_unchanged=True,
        only_measured_outward_source_faces_accumulated_by_max=True,
        original_measurement_domain_not_replaced_by_numerical_interior_targets=True,
        accuracy_accepted=False)
    return linear


def _guard_identity_scaled_source_trial(parent, proposed, fixed, recovery, guard, run_stage):
    """Probe the original scaled candidate, without solving or approving cost.

    A recovery must preserve the exact proposed state identity. A changed
    recovery is left to the original independent-QP/recovery/guard path.
    """
    proposed_token=hashlib.sha256(np.ascontiguousarray(proposed).tobytes()).hexdigest()
    identity=dict(recovery_called=recovery is not None,proposed_q_sha256=proposed_token,
        recovered_q_sha256=proposed_token,changed_complete_state_parameter_indices=[],
        recovery_changed_complete_state=False,complete_parameter_count=len(parent))
    feasibility=dict(complete_state_recovery_identity=identity)
    diagnostic=dict(passed=False,independent_fraction_QP_skipped=False,
        new_fraction_QP_KKT_certificate_claimed=False,original_full_guard_passed=False)
    value=np.asarray(proposed,float).copy()
    try:
        if recovery is not None:
            recovered=run_stage('trial_feasibility_recovery_total',recovery,parent.copy(),proposed.copy())
            if not isinstance(recovered,dict)or 'q'not in recovered:
                raise ValueError('Explicit trial recovery must return q and optional report')
            value=np.asarray(recovered['q'],float)
            original=recovered.get('report')
            feasibility=dict(original)if isinstance(original,dict)else dict(original_recovery_report=original)
            valid=value.shape==parent.shape and np.isfinite(value).all()
            changed=np.flatnonzero(value!=proposed).tolist()if valid else None
            identity.update(recovered_q_sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()if valid else None,
                changed_complete_state_parameter_indices=changed,
                recovery_changed_complete_state=bool(changed)if valid else None,
                recovered_state_shape_and_finite_values_valid=bool(valid),
                declared_fixed_values_preserved=bool(valid and np.array_equal(value[fixed],parent[fixed])))
            feasibility['complete_state_recovery_identity']=identity
            if not valid or not np.array_equal(value[fixed],parent[fixed]):
                raise ValueError('Trial feasibility recovery changed the state domain or a fixed source coordinate')
            if identity['recovered_q_sha256']!=proposed_token:
                raise ValueError('Scaled candidate recovery changed the exact proposed state identity')
        run_stage('trial_physical_guard_total',_check_physical,guard,value)
    except (ValueError,RuntimeError)as exc:
        diagnostic.update(reason=str(exc),failure_type=type(exc).__name__)
        return False,None,feasibility,diagnostic
    diagnostic.update(passed=True,independent_fraction_QP_skipped=True,
        original_full_guard_passed=True,reason='Original scaled candidate retained exact recovery identity and passed original full guard')
    return True,value,feasibility,diagnostic


def _needs_trial_fraction_SOC(linear,domain):
    """A failed full-step correction does not settle shorter trial scales."""
    if not callable(domain.get('at_trial_fraction')):return False
    report=linear.report.get('sequential_measured_shared_source_curvature_tightening',{})
    if report.get('stop_reason')=='tightened_QP_failed_keep_last_verified_direction':return True
    measurements=report.get('actual_original_nonlinear_feasibility_progress',{}).get('measurements',[])
    return bool(measurements and any(
        entry.get('actual_original_face_feasibility',{}).get('maximum_violation_px',0.)>0.
        for entry in measurements if entry.get('current_complete_KKT_direction_retained')))


def _trial_fraction_source_step(q,alpha,selected,domain,linear_options,
        normal,gradient,diagonal,tightening,solve_linear_step):
    """Solve and remeasure this scale on fresh cuts of the same parent faces.

    alpha scales only the numerical predictor gradient. It never scales the
    returned displacement again, the source objective used for acceptance,
    H/ridge/LM, or any physical box. The caller's pure domain callback supplies
    this scale's original restoration demand, including every zero group.
    """
    alpha=float(alpha)
    if not np.isfinite(alpha)or not 0.<alpha<1.:
        raise ValueError('Independent source correction needs a shortened trial fraction')
    fraction_domain=domain['at_trial_fraction'](alpha)
    original_matrix=np.asarray(domain['matrix'],float)
    A=np.asarray(fraction_domain['matrix'],float);b=np.asarray(fraction_domain['lower'],float)
    if (A.shape!=original_matrix.shape or not np.array_equal(A,original_matrix)
            or b.shape!=np.asarray(domain['lower']).shape or not np.isfinite(b).all()
            or set(fraction_domain.get('curvature_responses',{}))!=set(domain.get('curvature_responses',{}))):
        raise ValueError('Trial fraction must retain all ordered original faces and source families')
    options=dict(linear_options,global_step_constraint_matrix=fraction_domain['matrix'],
        global_step_constraint_lower=b.copy())
    numerical_gradient=alpha*np.asarray(gradient,float)
    initial=solve_linear_step(normal,numerical_gradient,diagonal,**options)
    from calibration.solver_runtime.original_halfspace_primal import original_full_qp_kkt_verified
    if not original_full_qp_kkt_verified(initial.report.get('shared_linear_domain_step',{}),initial.step):
        raise ValueError('Short-scale source QP lacks a complete original-row primal/dual KKT direction')
    corrected=_sequential_measured_shared_source_curvature(q,initial,selected,fraction_domain,
        options,normal,numerical_gradient,diagonal,tightening,solve_linear_step)
    if not original_full_qp_kkt_verified(corrected.report.get('shared_linear_domain_step',{}),corrected.step):
        raise ValueError('Short-scale SOC lacks a complete original-row primal/dual KKT direction')
    corrected.report['independent_trial_fraction_source_step']=dict(
        schema='rinkspace.independent-trial-fraction-source-SOC.v1',alpha=alpha,
        numerical_QP_gradient='alpha times unchanged parent source gradient',
        numerical_QP_gradient_sha256=hashlib.sha256(np.ascontiguousarray(numerical_gradient).tobytes()).hexdigest(),
        actual_source_model_decrease=float(-gradient@corrected.step-.5*corrected.step@(normal@corrected.step)),
        QP_KKT_is_for_numerical_fraction_gradient=True,
        actual_cost_prediction_and_acceptance_use_original_parent_gradient=True,
        returned_step_is_actual_displacement_not_scaled_again=True,
        accumulated_full_step_SOC_cuts_reused=False,
        source_H_ridge_LM_profiles_and_physical_boxes_unchanged=True,
        parent_supports_caps_and_true_trial_acceptance_unchanged=True,
        complete_original_QP_primal_dual_KKT_verified=True,
        fraction_domain=fraction_domain.get('report',{}).get('trial_fraction_domain'))
    return corrected


def solve_joint_source(evaluate:Callable,q,quadratic,scales,output,physicalguard:Callable,*,
        update_profiles=None,checkpoint_callback=None,checkpoint_identity=None,
        checkpoint_arrays:Mapping|None=None,resume=False,diagnostic_values=None,
        initial_damping=1e-3,verbose=True,global_schur_count=None,global_schur_indices=None,
        fixed_parameter_indices=(),schur_backtracking=False,source_block_factory=None,
        schur_view_lm_ratio=0.,schur_backtracking_aware_damping=False,
        cache_native_source_replay=False,source_stage_timing=False,
        reuse_source_jacobian=False,adaptive_source_backtracking=False,source_jacobian_refresh_ratio=0.,
        initial_line_search_fraction=1.,positive_parameter_indices=(),
        positive_parameter_resolution=0.,shared_parameter_bounds=None,
        shared_linear_constraints=None,shared_step_curvature_tightening=None,trial_feasibility_recovery=None,
        source_physical_endpoint_event=None,
        numerical_convergence_guard=None,save_current_linearization_cache=None,
        conditional_view_backtracking=False,conditional_view_protected_parameter_indices=(),
        source_trial_acceptance=None,refresh_activated_source_block=None,refresh_activated_source_rows=None,
        bounded_motion_model_factory=None):
    """Solve already-admitted sources without modifying their sigma or roles.

    ``evaluate(q, True)`` returns normalized residuals and a sparse Jacobian;
    ``evaluate(q, False)`` returns the same residuals. ``scales`` convert each
    normalized row's step back to native pixels (priors use 1), unchanged.
    ``physicalguard(q)`` raises ValueError/RuntimeError for an invalid trial,
    or returns False. It must not mutate q or silently project geometry.
    ``trial_feasibility_recovery(parent, proposed)`` is an optional explicit
    numerical step recovery, returning dict(q=..., report=...). It runs before
    the unchanged nonlinear guard and actual source cost. Every correction is
    recorded; the actual corrected step drives reduction predictions.

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
    ``adaptive_source_backtracking`` starts whole-source Schur backtracking at
    the learned fraction rather than repeating larger rejected real-cost trials.
    It changes only the numerical trial seed, persists that scalar in its
    checkpoint, and leaves the original true source cost and guards unchanged.
    """
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if refresh_activated_source_rows is None:
        refresh_activated_source_rows=getattr(refresh_activated_source_block,'exact_row_subset_refresh',None)
    q=np.asarray(q,float).copy();quadratic=np.asarray(quadratic,bool).copy()
    scales=np.asarray(scales,float).copy();initial_q=q.copy()
    schur_view_lm_ratio=float(schur_view_lm_ratio)
    if schur_view_lm_ratio not in (0.,1.):raise ValueError('Schur view LM ratio must be explicitly 0 or 1')
    if schur_view_lm_ratio and global_schur_count is None:
        raise ValueError('View LM opt-in requires the shared Schur strategy')
    if schur_backtracking_aware_damping and(global_schur_count is None or not schur_backtracking):
        raise ValueError('Backtracking-aware damping requires shared Schur whole-step backtracking')
    if adaptive_source_backtracking and(global_schur_count is None or not schur_backtracking):
        raise ValueError('Adaptive source backtracking requires shared Schur whole-step backtracking')
    next_line_search_fraction=float(initial_line_search_fraction)
    if not np.isfinite(next_line_search_fraction)or not 0<next_line_search_fraction<=1:
        raise ValueError('Initial line-search fraction must lie in (0, 1]')
    if cache_native_source_replay and source_block_factory is None:
        raise ValueError('Same-state native replay retention requires the streaming source strategy')
    if reuse_source_jacobian and not(source_block_factory is not None and cache_native_source_replay):
        raise ValueError('Modified-Newton reuse needs streaming source blocks and the one-state native replay cache')
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
    positive=np.asarray(positive_parameter_indices,dtype=int).reshape(-1)
    resolution=float(positive_parameter_resolution)
    if len(positive):
        selected=np.arange(global_schur_count)if shared_indices is None and global_schur_count is not None else shared_indices
        if (selected is None or len(np.unique(positive))!=len(positive)or
                not np.all(np.isin(positive,selected))or np.any(q[positive]<=0.)or
                not np.isfinite(resolution)or resolution<=0.):
            raise ValueError('Explicit positive shared parameters and their numerical resolution required')
    protected_view=np.asarray(conditional_view_protected_parameter_indices,dtype=int).reshape(-1)
    if bounded_motion_model_factory is not None and (
            not callable(bounded_motion_model_factory) or global_schur_count is None
            or conditional_view_backtracking):
        raise ValueError('Bounded motion nuisances require the complete Schur QP without conditional view backtracking')
    if conditional_view_backtracking:
        if global_schur_count is None or not schur_backtracking:
            raise ValueError('Conditional views require the existing explicit shared Schur backtracking strategy')
        protected_shared=np.arange(global_schur_count)if shared_indices is None else shared_indices
        if (len(np.unique(protected_view))!=len(protected_view)or np.any(protected_view<0)
                or np.any(protected_view>=len(q))or not np.isin(protected_view,protected_shared).all()):
            raise ValueError('Every original truth dependency must remain in the explicit shared chart')
    elif len(protected_view):
        raise ValueError('Protected conditional-view columns require the explicit opt-in strategy')
    physical_bounds={int(k):tuple(map(float,v))for k,v in (shared_parameter_bounds or {}).items()}
    selected=np.arange(global_schur_count)if shared_indices is None and global_schur_count is not None else shared_indices
    if physical_bounds and (selected is None or not np.all(np.isin(list(physical_bounds),selected))):
        raise ValueError('Explicit working bounds must name selected shared parameters')
    if shared_linear_constraints is not None and (selected is None or not callable(shared_linear_constraints)):
        raise ValueError('Declared shared-domain half-spaces need a callable and explicit shared columns')
    if source_physical_endpoint_event is not None and(not callable(source_physical_endpoint_event)
            or global_schur_count is None or not schur_backtracking):
        raise ValueError('Physical endpoint events require an explicit callback and complete Schur backtracking')
    if numerical_convergence_guard is not None and not callable(numerical_convergence_guard):
        raise ValueError('Numerical convergence guard must be an explicit caller-owned predicate')
    if save_current_linearization_cache is not None and not callable(save_current_linearization_cache):
        raise ValueError('Current linearization export must use the owning source writer')
    for col,interval in physical_bounds.items():
        if len(interval)!=2 or not np.isfinite(interval).all()or not interval[0]<interval[1]:
            raise ValueError('Explicit finite nondegenerate shared working intervals required')
    fixed_shared_elimination=None
    if selected is not None and np.any(np.isin(selected,fixed)):
        requested=np.asarray(selected,int).copy()
        removed=requested[np.isin(requested,fixed)]
        # Fixed source coordinates already have isolated identity H rows and
        # zero g through eliminate_fixed_parameters. They must also be absent
        # from the LP/QP/curvature variable chart, so constraints cannot reopen
        # them through numerical tolerance. Complete state slots remain intact.
        for column in removed:
            if int(column) in physical_bounds:
                lower,upper=physical_bounds[int(column)]
                if not lower<=q[column]<=upper:
                    raise ValueError('Declared fixed source value is outside its original physical bounds')
                del physical_bounds[int(column)]
        selected=requested[~np.isin(requested,fixed)]
        if not len(selected):raise ValueError('Shared source chart has no free coordinate after declared fixed elimination')
        shared_indices=selected.copy();global_schur_count=len(selected)
        positive=positive[~np.isin(positive,fixed)]
        protected_view=protected_view[~np.isin(protected_view,fixed)]
        fixed_shared_elimination=dict(requested_shared_parameter_indices=requested.tolist(),
            effective_shared_parameter_indices=selected.tolist(),
            algebraically_eliminated_fixed_shared_indices=removed.tolist(),
            complete_parameter_count=len(q),complete_state_slots_preserved=True,
            original_fixed_normal_identity_and_zero_gradient_protocol_reused=True,
            source_halfspace_callbacks_rebuilt_on_effective_chart=True,
            no_approximate_fixed_box_or_trial_clipping=True)
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
    restored_history_reference=None;restored_reviews_reference=None
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
            history=read_checkpoint_history(saved,checkpoint);reviews=read_checkpoint_reviews(saved,checkpoint)
            restored_history_reference=checkpoint_history_reference(saved,checkpoint)
            restored_reviews_reference=checkpoint_history_reference(saved,checkpoint,field='reviews_json')
            if 'next_line_search_fraction' in saved:
                next_line_search_fraction=float(saved['next_line_search_fraction'])
                if not np.isfinite(next_line_search_fraction)or not 0<next_line_search_fraction<=1:
                    raise ValueError('Checkpoint line-search fraction must lie in (0, 1]')
            elapsed_offset=float(saved['elapsed_seconds'])
        _check_physical(physicalguard,q)
    history_writer=HistorySegmentWriter(output/'history_segments',reference=restored_history_reference,history=history)
    reviews_writer=HistorySegmentWriter(output/'review_segments',reference=restored_reviews_reference,history=reviews)
    started=time.perf_counter()-elapsed_offset
    stage_times=None
    bounded_motion_model=None
    def elapsed():return time.perf_counter()-started
    def run_stage(name, callback, *args, **kwargs):
        # Account for every invocation, including rejected trials and failed
        # SOC solves. The former last-QP report hid most of the real latency.
        if stage_times is None:return callback(*args, **kwargs)
        begin=time.perf_counter()
        if verbose:print(_json_text(dict(stage='source_stage_begin',operation=name,iteration=len(history))),flush=True)
        succeeded=False
        try:
            value=callback(*args, **kwargs);succeeded=True
            return value
        finally:
            seconds=time.perf_counter()-begin
            stage_times[name+'_seconds']=stage_times.get(name+'_seconds',0.)+seconds
            stage_times[name+'_calls']=stage_times.get(name+'_calls',0)+1
            if verbose:print(_json_text(dict(stage='source_stage_end',operation=name,iteration=len(history),
                seconds=seconds,succeeded=succeeded)),flush=True)
    def timed_linear_step(*args, **kwargs):
        if bounded_motion_model is not None:
            return run_stage('constrained_bounded_motion_schur_solve_total',
                bounded_motion_model.solve,*args,schur_trial_linear_step,**kwargs)
        return run_stage('constrained_schur_solve_total',schur_trial_linear_step,*args,**kwargs)
    def source_model_reduction(step):
        if bounded_motion_model is not None:
            return bounded_motion_model.model_reduction(normal,gradient,step)
        return float(-gradient@step-.5*step@(normal@step))
    def timed_curvature(*args, **kwargs):
        return run_stage('shared_curvature_measurement_total',shared_step_curvature_tightening,*args,**kwargs)
    def save():
        history_reference=history_writer.commit(history)
        reviews_reference=reviews_writer.commit(reviews)
        arrays=dict(schema=np.array(SCHEMA),identity_json=np.array(identity),q=q,initial_q=initial_q,
            quadratic=quadratic,scales=scales,damping=np.array(damping),history_json=np.array(_json_text(history_reference)),
            reviews_json=np.array(_json_text(reviews_reference)),elapsed_seconds=np.array(elapsed()))
        if adaptive_source_backtracking:
            arrays['next_line_search_fraction']=np.array(next_line_search_fraction)
        arrays.update({'aux__'+k:v for k,v in extras.items()})
        snapshot=getattr(update_profiles,'checkpoint_arrays',None)
        if snapshot is not None:
            arrays.update({'profile__'+str(k):np.asarray(v)for k,v in snapshot().items()})
        temporary=checkpoint.with_name(checkpoint.name+'.tmp')
        with temporary.open('wb')as stream:np.savez_compressed(stream,**arrays)
        temporary.replace(checkpoint)
        _write_json(output/'iterations.json',history_reference);_write_json(output/'source_reviews.json',reviews_reference)
    def review(kind,record):
        if checkpoint_callback is None:return None
        readonly=q.copy();readonly.setflags(write=False)
        event=dict(record or {},checkpoint_kind=kind,checkpoint=str(checkpoint),
            elapsed_seconds=elapsed(),numerical_convergence_is_not_accuracy=True)
        response=checkpoint_callback(readonly,event)
        if response is not None:
            # Persist the source result with the same recoverable state.
            reviews.append(copy.deepcopy(dict(event=event,result=response)));save()
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
    derivative_model=None
    if reuse_source_jacobian:
        from structured_programme_source_candidate.lagged_jacobian_source import LaggedJacobianSource
        derivative_model=LaggedJacobianSource(refresh_reduction_ratio=source_jacobian_refresh_ratio)
    repeat_frozen_profiles=False
    def early_control_stop():
        stop=_control_stop(output)
        if stop is not None and stop['control'].get('save_current_linearization_cache')is True:
            if source_block_factory is None or not cache_native_source_replay or not callable(save_current_linearization_cache):
                raise ValueError('Explicit current-cache stop requires retained source blocks and the owning writer')
            # No new model trial is taken. Reach one fresh derivative at the
            # current q/profile, then export at the existing normal boundary.
            return None
        return stop
    save();stop=early_control_stop()
    if stop is None:stop=review('resumed'if resume else'initial',None)
    status='stopped_by_control'if stop is not None and stop['origin']=='control.json'else'stopped_by_source_review'
    while stop is None:
        stop=early_control_stop()
        if stop is not None:status='stopped_by_control';break
        stage_times={}if source_stage_timing else None
        stage_begin=time.perf_counter()if source_stage_timing else None
        if update_profiles is not None and not repeat_frozen_profiles:update_profiles(q)
        repeat_frozen_profiles=False
        if stage_times is not None:stage_times['profile_update_seconds']=time.perf_counter()-stage_begin
        stop=early_control_stop()
        if stop is not None:status='stopped_by_control';save();break
        streaming_report=None
        stage_begin=time.perf_counter()if source_stage_timing else None
        if source_block_factory is None:
            residual,jac=checked_evaluate(q,True);before=source_cost(residual,quadratic)
        else:
            profile_token=_frozen_profile_token(update_profiles)
            use_lagged=derivative_model.begin(q,profile_token)if derivative_model is not None else False
            if use_lagged and bounded_motion_model_factory is not None:
                derivative_model.refresh('Bounded nuisance row replacement requires the same-state source Jacobian')
                use_lagged=derivative_model.begin(q,profile_token)
            cache_stop=_control_stop(output)
            if cache_stop is not None and cache_stop['control'].get('save_current_linearization_cache')is True:
                use_lagged=False
            linearization_blocks=(lambda:derivative_model.blocks(lambda:checked_source_blocks(q,False),
                fresh_activated_block=None if refresh_activated_source_block is None else
                    lambda supplied:refresh_activated_source_block(q.copy(),supplied),
                fresh_activated_rows=None if refresh_activated_source_rows is None else
                    lambda supplied,rows:refresh_activated_source_rows(q.copy(),supplied,rows)))if use_lagged else(lambda:checked_source_blocks(q,True))
            from structured_programme_source_candidate.lagged_jacobian_source import LaggedZeroJacobianActivation
            try:
                accumulated=accumulate_source_normal(linearization_blocks,len(q),
                    expected_row_count=len(scales),frozen_profile_token=profile_token,
                    fixed_parameter_indices=fixed,retain_native_replay=cache_native_source_replay,
                    source_state=q if cache_native_source_replay else None)
            except LaggedZeroJacobianActivation as activation:
                if derivative_model is None or not use_lagged:raise
                activation_receipt=dict(activation.report)
                # Discard partially accumulated lagged normal and its trace before
                # generating the existing exact source J at unchanged q/profile.
                activation.__traceback__=None
                derivative_model.refresh('original zero-residual numeric-zero-J source row became active')
                use_lagged=derivative_model.begin(q,profile_token)
                if use_lagged:raise RuntimeError('Activated source support requires an exact linearization')
                accumulated=accumulate_source_normal(lambda:checked_source_blocks(q,True),len(q),
                    expected_row_count=len(scales),frozen_profile_token=profile_token,
                    fixed_parameter_indices=fixed,retain_native_replay=cache_native_source_replay,
                    source_state=q if cache_native_source_replay else None)
                derivative_model.current_report['zero_J_support_activation_refresh']=activation_receipt
            if derivative_model is not None and not use_lagged:
                derivative_model.capture(accumulated.native_replay_cache)
            check_frozen_profiles(profile_token)
            before=accumulated.cost
            streaming_report=dict(accumulated.report,frozen_profile_token=profile_token,
                peak_process_memory_measured_by_solver=False,
                input_CSR_bytes_is_not_total_process_peak=True,
                source_alignment_accuracy_not_implied=True)
        stop=_control_stop(output)
        if stop is not None:
            if source_block_factory is not None:
                if stop['control'].get('save_current_linearization_cache')is True:
                    if use_lagged:
                        # A request may arrive during a mixed-J pass. It is not
                        # a current full-source cache; refresh at this same q
                        # and frozen profile before the owning writer seals it.
                        accumulated.native_replay_cache=None
                        derivative_model.refresh('explicit current-cache handoff requires fresh full source J')
                        repeat_frozen_profiles=True
                        stop=None
                        continue
                    if not callable(save_current_linearization_cache):
                        raise ValueError('Explicit current-linearization export requested without the owning source writer')
                    stop['current_linearization_cache']=save_current_linearization_cache(q,accumulated)
                accumulated.native_replay_cache=None
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
        bounded_motion_model=None
        if bounded_motion_model_factory is not None:
            from calibration.solver_runtime.bounded_motion_schur import BoundedMotionSchurModel
            bounded_motion_model=run_stage('bounded_motion_model_prepare_total',
                bounded_motion_model_factory,q.copy())
            if (not isinstance(bounded_motion_model,BoundedMotionSchurModel)
                    or bounded_motion_model.parameter_count!=len(q)):
                raise ValueError('Complete original bounded nuisance moment model required')
            bounded_motion_model.eliminate_fixed(fixed)
            if source_block_factory is not None:check_frozen_profiles(profile_token)
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
                    refinement_report=None
                    if reference_error>1e-3:
                        from calibration.solver_runtime.native_reference_refinement import refine_undamped_native_reference
                        original_reference_error=reference_error
                        try:
                            def original_native_replay(steps):
                                return replay_native_step_maxima(lambda:checked_source_blocks(q,True),
                                    steps,accumulated.contract,frozen_profile_token=profile_token,
                                    native_replay_cache=accumulated.native_replay_cache,source_state=q)
                            corrected_gn,corrected_reference,corrected_gn_step,corrected_error,refinement_report=refine_undamped_native_reference(
                                normal,gradient,diagonal,gn,reference.step,
                                np.arange(global_schur_count)if shared_indices is None else shared_indices,fixed,
                                native_replay=original_native_replay,initial_native_error=original_reference_error)
                            refinement_report.update(original_threshold_native_px=1e-3,
                                original_complete_native_gate_passed=bool(corrected_error<=1e-3),
                                iteration=len(history),q_sha256=hashlib.sha256(np.asarray(q).tobytes()).hexdigest(),
                                derivative_model='lagged_J'if use_lagged else'fresh_source_J')
                            if corrected_error<=1e-3:
                                gn,gn_step,reference_error=corrected_gn,corrected_gn_step,corrected_error
                        except Exception as refinement_error:
                            refinement_report=dict(original_native_reference_error_px=original_reference_error,
                                original_complete_native_gate_passed=False,
                                error_type=type(refinement_error).__name__,error=str(refinement_error))
                        _write_json(output/('native_reference_refinement_%06d.json'%len(history)),refinement_report)
                        print(json.dumps(dict(stage='same_equation_native_reference_refinement',
                            original_reference_error_native_px=original_reference_error,
                            final_reference_error_native_px=reference_error,
                            original_complete_native_gate_passed=bool(reference_error<=1e-3))),flush=True)
                    if reference_error>1e-3:
                        from calibration.solver_runtime.failed_native_reference_capture import capture_failed_native_reference
                        try:
                            captured=capture_failed_native_reference(output,q=q,normal=normal,gradient=gradient,
                                diagonal=diagonal,gn=gn,reference_step=reference.step,
                                shared_indices=np.arange(global_schur_count)if shared_indices is None else shared_indices,
                                fixed_indices=fixed,cache=accumulated.native_replay_cache,
                                metadata=dict(iteration=len(history),reference_error_native_px=reference_error,
                                    maximum_GN_native_step_px=gn_step,original_threshold_native_px=1e-3,
                                    derivative_model='lagged_J'if use_lagged else'fresh_source_J',
                                    lagged_J=bool(use_lagged),global_schur_count=global_schur_count,
                                    derivative_model_report=None if derivative_model is None else derivative_model.current_report,
                                    failed_same_equation_refinement=refinement_report,
                                    original_sparse_permutation='MMD_AT_PLUS_A'))
                            print(json.dumps(dict(stage='failed_complete_native_reference_captured',**captured)),flush=True)
                        except Exception as capture_error:
                            print(json.dumps(dict(stage='failed_complete_native_reference_capture_error',
                                error_type=type(capture_error).__name__,error=str(capture_error))),flush=True)
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
        trial_records=[];schur_full_step=None;uphill_lagged_refresh=None;fraction_SOC_needed=False
        conditional_step=None;accepted_search_mode=None;accepted_trial_policy=None
        physical_endpoint_event=None;accepted_physical_endpoint_event=False
        line_search_seed=next_line_search_fraction if adaptive_source_backtracking else 1.
        accepted_source_reduction_ratio=None
        trial_cost_total_seconds=0.
        trial_schedule=(tuple(('conditional_views',i)for i in range(12))+tuple(('original_whole_step',i)for i in range(12))
                if conditional_view_backtracking else tuple(('original_whole_step',i)for i in range(12)))
        if source_physical_endpoint_event is not None:
            position=trial_schedule.index(('original_whole_step',0))+1
            trial_schedule=trial_schedule[:position]+(('physical_endpoint_event',0),)+trial_schedule[position:]
        for trial_ordinal,(search_mode,trial)in enumerate(trial_schedule):
            if search_mode=='physical_endpoint_event'and physical_endpoint_event is None:continue
            stop=early_control_stop()
            if stop is not None:status='stopped_by_control';break
            trials+=1
            if global_schur_count is None:
                step=spsolve(normal+diags(damping*diagonal),-gradient,permc_spec='MMD_AT_PLUS_A')
            else:
                if schur_full_step is None or not schur_backtracking:
                    # At the existing source-FD resolution, preserve the
                    # nominal positive value on an active lower-bound face.
                    # Recompute eligibility and the constrained reduced solve
                    # every state: outward motion releases the face. This is
                    # not a persisted fixed parameter or a new height prior.
                    boundary=positive[q[positive]<=resolution]
                    step_lower=np.full(global_schur_count,-np.inf)
                    step_upper=np.full(global_schur_count,np.inf)
                    for col,(lower,upper)in physical_bounds.items():
                        pos=int(np.flatnonzero(selected==col)[0])
                        step_lower[pos]=lower-q[col];step_upper[pos]=upper-q[col]
                    domain=(run_stage('shared_constraint_build_total',shared_linear_constraints,q,selected)
                            if shared_linear_constraints is not None else None)
                    if domain is not None and (not isinstance(domain,dict)
                            or 'matrix' not in domain or 'lower' not in domain):
                        raise ValueError('Shared-domain callback must return matrix/lower and optional report')
                    linear_options=dict(global_count=global_schur_count,damping=damping,
                        global_parameter_indices=shared_indices,schur_view_lm_ratio=schur_view_lm_ratio,
                        nonnegative_global_step_indices=boundary,
                        global_step_lower_bounds=step_lower,global_step_upper_bounds=step_upper,
                        global_step_constraint_matrix=None if domain is None else domain['matrix'],
                        global_step_constraint_lower=None if domain is None else domain['lower'])
                    linear=timed_linear_step(normal,gradient,diagonal,**linear_options)
                    if shared_step_curvature_tightening is not None and domain is not None:
                        linear=_sequential_measured_shared_source_curvature(q,linear,selected,domain,
                            linear_options,normal,gradient,diagonal,timed_curvature,
                            timed_linear_step)
                        fraction_SOC_needed=_needs_trial_fraction_SOC(linear,domain)
                    if domain is not None:
                        linear.report['shared_linear_domain_provenance']=domain.get('report',{})
                    if fixed_shared_elimination is not None:
                        if np.any(linear.step[fixed]!=0.):
                            raise RuntimeError('Declared fixed source elimination did not return an exact zero step')
                        linear.report['declared_fixed_shared_elimination']=dict(fixed_shared_elimination,
                            all_declared_fixed_step_values_exact_zero=True)
                    schur_full_step=linear.step;linear_report=linear.report
                    if source_physical_endpoint_event is not None:
                        event_begin=time.perf_counter()
                        try:
                            physical_endpoint_event=source_physical_endpoint_event(q.copy(),schur_full_step.copy(),line_search_seed)
                            if physical_endpoint_event is not None:
                                fraction=float(physical_endpoint_event['fraction'])
                                if not np.isfinite(fraction)or not 0.<fraction<=line_search_seed:
                                    raise ValueError('Physical endpoint event fraction is outside the original trial direction')
                                linear_report['actual_physical_endpoint_event']=physical_endpoint_event['report']
                        except ValueError as exc:
                            physical_endpoint_event=None
                            linear_report['actual_physical_endpoint_event_unavailable']=str(exc)
                        if stage_times is not None:
                            stage_times['physical_endpoint_event_seconds']=time.perf_counter()-event_begin
                    if conditional_view_backtracking:
                        from structured_programme_source_candidate.conditional_view_backtracking import ConditionalViewBacktracking
                        conditional_step=ConditionalViewBacktracking(normal,gradient,diagonal,schur_full_step,
                            shared_parameter_indices=selected,damping=damping,schur_view_lm_ratio=schur_view_lm_ratio,
                            protected_parameter_indices=protected_view,fixed_parameter_indices=fixed)
                alpha=(float(physical_endpoint_event['fraction'])if search_mode=='physical_endpoint_event'else
                    line_search_seed*2.**(-trial)if schur_backtracking else 1.)
                step=(conditional_step.step(alpha)if search_mode=='conditional_views'else alpha*schur_full_step)
            nextq=q+step
            trial_cost_seconds=0.
            proposed_q=nextq.copy()
            proposed_token=hashlib.sha256(memoryview(np.ascontiguousarray(proposed_q)).cast('B')).hexdigest()
            feasibility_report=dict(complete_state_recovery_identity=dict(
                recovery_called=False,proposed_q_sha256=proposed_token,recovered_q_sha256=proposed_token,
                changed_complete_state_parameter_indices=[],recovery_changed_complete_state=False,
                complete_parameter_count=len(q)))
            trial_curvature_report=None;independent_fraction_step=None
            scaled_candidate_guard_passed=False
            trial_acceptance_decision=None;trial_cost_evaluated=False
            fraction_info={}if global_schur_count is None else(dict(whole_source_step_fraction=None,shared_source_step_fraction=alpha,
                line_search_fraction=alpha,line_search_mode=search_mode)if search_mode=='conditional_views'else
                dict(whole_source_step_fraction=alpha,line_search_fraction=alpha if global_schur_count is not None else None,line_search_mode=search_mode))
            try:
                if (fraction_SOC_needed and search_mode=='original_whole_step'and 0.<alpha<1.
                        and domain.get('prefer_scaled_candidate_before_fraction_soc') is True):
                    scaled_candidate_guard_passed,guarded_q,scaled_recovery,scaled_probe=(
                        _guard_identity_scaled_source_trial(q,nextq,fixed,trial_feasibility_recovery,
                                                            physicalguard,run_stage))
                    scaled_probe['alpha']=alpha
                    scaled_probe['trial']=trials
                    scaled_probe['trial_feasibility_recovery']=scaled_recovery
                    linear_report.setdefault('scaled_candidates_before_fraction_SOC',[]).append(scaled_probe)
                    if scaled_candidate_guard_passed:
                        nextq=guarded_q
                        if trial_feasibility_recovery is not None:step=nextq-q
                        feasibility_report=scaled_recovery
                        fraction_info['actual_step_basis']='original_scaled_full_step_guarded_before_fraction_SOC'
                if (fraction_SOC_needed and search_mode=='original_whole_step'and 0.<alpha<1.
                        and not scaled_candidate_guard_passed):
                    # Each scale begins at the unchanged parent's own faces.
                    # A failed full-step cut is never shrunk or accumulated
                    # into this independent numerical predictor/correction.
                    fraction_entry=dict(alpha=alpha,accepted_numerical_direction=False)
                    linear_report.setdefault('independent_trial_fraction_source_steps',[]).append(fraction_entry)
                    try:
                        independent_fraction_step=run_stage('trial_fraction_source_QP_and_SOC_total',
                            _trial_fraction_source_step,q.copy(),alpha,selected.copy(),domain,linear_options,
                            normal,gradient,diagonal,timed_curvature,timed_linear_step)
                    except (ValueError,RuntimeError)as exc:
                        fraction_entry['failure']=str(exc)
                        fraction_entry['original_scaled_verified_direction_retained_for_actual_trial']=True
                    else:
                        step=independent_fraction_step.step.copy();nextq=q+step
                        if np.any(step[fixed]!=0.):
                            raise RuntimeError('Short-scale source elimination changed a declared fixed source coordinate')
                        fraction_entry.update(accepted_numerical_direction=True,
                            direction_sha256=hashlib.sha256(np.ascontiguousarray(step).tobytes()).hexdigest(),
                            QP_and_SOC=independent_fraction_step.report)
                        independent_fraction_step.report['independent_trial_fraction_source_step'][
                            'actual_source_model_decrease']=source_model_reduction(step)
                        fraction_info.update(whole_source_step_fraction=None,
                            numerical_source_gradient_fraction=alpha,
                            actual_step_basis='independently_solved_trial_fraction_source_model')
                        proposed_q=nextq.copy()
                        proposed_token=hashlib.sha256(np.ascontiguousarray(proposed_q).tobytes()).hexdigest()
                        feasibility_report['complete_state_recovery_identity'].update(
                            proposed_q_sha256=proposed_token,recovered_q_sha256=proposed_token)
                if trial_feasibility_recovery is not None and not scaled_candidate_guard_passed:
                    recovered=run_stage('trial_feasibility_recovery_total',trial_feasibility_recovery,q.copy(),nextq.copy())
                    if not isinstance(recovered,dict)or 'q'not in recovered:
                        raise ValueError('Explicit trial recovery must return q and optional report')
                    value=np.asarray(recovered['q'],float)
                    original_recovery_report=recovered.get('report')
                    feasibility_report=(dict(original_recovery_report)if isinstance(original_recovery_report,dict)
                        else dict(original_recovery_report=original_recovery_report))
                    valid=value.shape==q.shape and np.isfinite(value).all()
                    changed=np.flatnonzero(value!=proposed_q).tolist()if valid else None
                    feasibility_report['complete_state_recovery_identity']=dict(
                        recovery_called=True,proposed_q_sha256=proposed_token,
                        recovered_q_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast('B')).hexdigest()if valid else None,
                        changed_complete_state_parameter_indices=changed,
                        recovery_changed_complete_state=bool(changed)if valid else None,
                        complete_parameter_count=len(q),recovered_state_shape_and_finite_values_valid=bool(valid),
                        declared_fixed_values_preserved=bool(valid and np.array_equal(value[fixed],q[fixed])))
                    if value.shape!=q.shape or not np.isfinite(value).all()or not np.array_equal(value[fixed],q[fixed]):
                        raise ValueError('Trial feasibility recovery changed the state domain or a fixed source coordinate')
                    nextq=value;step=nextq-q
                if search_mode=='conditional_views'and shared_step_curvature_tightening is not None and domain is not None:
                    # Actual candidate replay, including its full conditional view step.
                    # The callback can only tighten original faces; it cannot approve
                    # this state. The unchanged nonlinear guard still rejects it.
                    trial_correction=timed_curvature(q.copy(),step.copy(),selected.copy(),domain)
                    if trial_correction is not None:
                        trial_lower=np.asarray(trial_correction['lower'],float)
                        if (trial_lower.shape!=np.asarray(domain['lower']).shape or not np.isfinite(trial_lower).all()
                                or np.any(trial_lower<domain['lower'])):
                            raise ValueError('Conditional candidate curvature may only tighten original source faces')
                        trial_curvature_report=trial_correction.get('report',{})
                if not scaled_candidate_guard_passed:
                    run_stage('trial_physical_guard_total',_check_physical,physicalguard,nextq)
                if trial_curvature_report is not None:
                    raise ValueError('Conditional candidate remains outside an original source support face; retain original whole-step fallback')
                trial_cost_begin=time.perf_counter()if source_stage_timing else None
                try:
                    if source_block_factory is None:
                        after=source_cost(checked_evaluate(nextq,False),quadratic)
                    else:
                        check_frozen_profiles(profile_token)
                        after=replay_source_cost(lambda:checked_source_blocks(nextq,False),
                            accumulated.contract,frozen_profile_token=profile_token)
                        check_frozen_profiles(profile_token)
                    trial_cost_evaluated=True
                finally:
                    if source_stage_timing:
                        trial_cost_seconds=time.perf_counter()-trial_cost_begin
                        trial_cost_total_seconds+=trial_cost_seconds
                source_cost_accepts=after<before-1e-10*max(1.,before)
                trial_accepts=source_cost_accepts
                if source_trial_acceptance is not None:
                    parent_readonly=q.copy();parent_readonly.flags.writeable=False
                    trial_readonly=nextq.copy();trial_readonly.flags.writeable=False
                    step_readonly=step.copy();step_readonly.flags.writeable=False
                    trial_acceptance_decision=run_stage('trial_acceptance_total',source_trial_acceptance,parent_readonly,trial_readonly,
                        before_cost=float(before),after_cost=float(after),default_source_cost_accepts=bool(source_cost_accepts),
                        trial_context=dict(iteration=len(history),trial=trials,
                            line_search_fraction=None if global_schur_count is None else alpha,
                            alpha=None if global_schur_count is None else alpha,
                            line_search_mode=search_mode,actual_step=step_readonly,
                            predicted_source_reduction=source_model_reduction(step),
                            physical_and_complete_source_guard_passed=True))
                    if (not isinstance(trial_acceptance_decision,dict)
                            or type(trial_acceptance_decision.get('accept'))is not bool
                            or trial_acceptance_decision.get('mode','source_objective')not in('source_objective','truth_restoration')):
                        raise ValueError('Source trial acceptance must return an explicit boolean and supported acceptance mode')
                    trial_accepts=trial_acceptance_decision['accept']
                    if trial_acceptance_decision.get('mode','source_objective')=='source_objective'and trial_accepts and not source_cost_accepts:
                        raise ValueError('Ordinary source-objective acceptance cannot approve a source-cost increase')
                restoring=(trial_acceptance_decision is not None
                    and trial_acceptance_decision.get('mode','source_objective')=='truth_restoration')
                if trial_accepts:
                    endpoint_transition=(independent_fraction_step is None and physical_endpoint_event is not None and search_mode!='conditional_views'
                        and alpha>=physical_endpoint_event['fraction'])
                    restoration_trust_report=(restoration_source_model_agreement(
                        source_model_reduction(step),before-after)if restoring else None)
                    if restoring:
                        next_damping=damping;learned_fraction=line_search_seed
                    if schur_backtracking_aware_damping:
                        # Validate the next numerical value before committing
                        # an accepted state or its accepted trial record.
                        damping_fraction=alpha/line_search_seed if adaptive_source_backtracking else alpha
                        next_damping=accepted_schur_damping(damping,damping_fraction,backtracking_aware=True,truth_restoration=restoring,
                            restoration_model_agreement=restoration_trust_report)
                    if adaptive_source_backtracking and not restoring:
                        predicted=source_model_reduction(step)
                        ratio=float((before-after)/predicted)if np.isfinite(predicted)and predicted>0 else None
                        accepted_source_reduction_ratio=ratio if ratio is not None and np.isfinite(ratio)else None
                        learned_fraction=min(1.,2*alpha)if accepted_source_reduction_ratio is not None and accepted_source_reduction_ratio>=.75 else alpha
                    if endpoint_transition:
                        # A physical association event calls for a new model,
                        # not an alpha-derived huge LM penalty or a tiny next
                        # seed. The original true acceptance gates ran above.
                        next_damping=damping;learned_fraction=1.
                    if global_schur_count is not None:
                        trial_records.append(dict(trial=trials,damping=damping,accepted=True,cost=after,
                            **fraction_info,shared_step=step[:global_schur_count].tolist()if shared_indices is None else step[shared_indices].tolist()))
                        if source_stage_timing:trial_records[-1]['source_cost_seconds']=trial_cost_seconds
                        if feasibility_report is not None:trial_records[-1]['trial_feasibility_recovery']=feasibility_report
                    if global_schur_count is not None and trial_acceptance_decision is not None:
                        trial_records[-1]['explicit_source_trial_acceptance']=trial_acceptance_decision
                    q=nextq;accepted=True;accepted_search_mode=search_mode
                    accepted_physical_endpoint_event=endpoint_transition
                    accepted_trial_policy=trial_acceptance_decision
                    if trial_curvature_report is not None:trial_records[-1]['actual_conditional_candidate_curvature']=trial_curvature_report
                    if adaptive_source_backtracking:next_line_search_fraction=learned_fraction
                    if restoring:
                        previous_damping=damping
                        damping=next_damping
                        accepted_damping_update=dict(previous_damping=previous_damping,next_damping=damping,
                            accepted_whole_source_step_fraction=(alpha if global_schur_count is not None and independent_fraction_step is None else None),
                            numerical_source_gradient_fraction=alpha if independent_fraction_step is not None else None,
                            update_rule=('truth_restoration_retains_lambda_for_feasibility_backtrack'if damping_fraction<1 else
                                'truth_restoration_lambda_times_0.3_for_reliable_unshortened_source_model'if restoration_trust_report['reliable_for_restoration_lambda_decrease']else
                                'truth_restoration_retains_lambda_for_unreliable_source_model')if schur_backtracking_aware_damping else'truth_restoration_retains_lambda',
                            source_descent_rho_used=False,
                            restoration_source_model_agreement=restoration_trust_report,
                            source_H_g_objective_and_guards_unchanged=True)
                        if adaptive_source_backtracking:
                            accepted_damping_update['initial_line_search_fraction']=line_search_seed
                            accepted_damping_update['fraction_relative_to_initial_seed']=alpha/line_search_seed
                    elif schur_backtracking_aware_damping:
                        previous_damping=damping
                        damping=next_damping
                        accepted_damping_update=dict(previous_damping=previous_damping,
                            accepted_whole_source_step_fraction=alpha,next_damping=damping,
                            update_rule=('lambda_times_seed_divided_by_alpha'if alpha<line_search_seed else'original_lambda_times_0.3')if adaptive_source_backtracking else('lambda_divided_by_alpha'if alpha<1 else'original_lambda_times_0.3'),
                            reason='An additional backtrack below the learned seed changes the next numerical damping.'if adaptive_source_backtracking else'A shortened accepted source step retains evidence that the current numerical linear step needed damping.',
                            source_H_g_GN_reference_objective_priors_profiles_unchanged=True)
                        if adaptive_source_backtracking:
                            accepted_damping_update['initial_line_search_fraction']=line_search_seed
                            accepted_damping_update['fraction_relative_to_initial_seed']=alpha/line_search_seed
                    else:
                        # Keep the exact original arithmetic in the default path.
                        damping=max(damping*.3,1e-12)
                    if endpoint_transition:
                        damping=next_damping
                        accepted_damping_update=dict(previous_damping=damping,next_damping=damping,
                            update_rule='actual_physical_endpoint_event_retains_lambda_and_relinearizes',
                            accepted_whole_source_step_fraction=alpha,
                            next_line_search_fraction=next_line_search_fraction,
                            source_support_objective_weights_and_true_acceptance_unchanged=True)
                    break
                reason=('objective_not_lower_at_profile_precision'if trial_acceptance_decision is None else
                    str(trial_acceptance_decision.get('rejection_reason','explicit_source_trial_acceptance_rejected'))[:180])
            except (ValueError,RuntimeError)as exc:reason=str(exc)[:180]
            if global_schur_count is not None:
                trial_records.append(dict(trial=trials,damping=damping,accepted=False,reason=reason,
                    evaluated_trial_cost=after if trial_cost_evaluated else None,
                    **fraction_info,shared_step=step[:global_schur_count].tolist()if shared_indices is None else step[shared_indices].tolist()))
                if source_stage_timing:trial_records[-1]['source_cost_seconds']=trial_cost_seconds
                if trial_acceptance_decision is not None:trial_records[-1]['explicit_source_trial_acceptance']=trial_acceptance_decision
                if feasibility_report is not None:trial_records[-1]['trial_feasibility_recovery']=feasibility_report
                if trial_curvature_report is not None:trial_records[-1]['actual_conditional_candidate_curvature']=trial_curvature_report
                if derivative_model is not None and use_lagged and search_mode=='original_whole_step'and independent_fraction_step is None:
                    # Conditional alpha->0 contains a nonzero view-only step, so
                    # delta_cost/alpha is not the old whole-step direction slope.
                    original_records=[row for row in trial_records if row.get('line_search_mode')=='original_whole_step'
                        and row.get('numerical_source_gradient_fraction')is None]
                    uphill_lagged_refresh=_uphill_lagged_source_trials(original_records,before)
                    if uphill_lagged_refresh is not None:break
            if not(global_schur_count is not None and schur_backtracking):damping*=10.
        if stop is not None:save();break
        if derivative_model is not None:
            if accepted:derivative_model.accepted(before,after,step,normal,gradient,
                source_objective_acceptance=not(accepted_trial_policy is not None and accepted_trial_policy.get('mode')=='truth_restoration'))
            elif use_lagged:
                derivative_model.refresh(uphill_lagged_refresh['reason']if uphill_lagged_refresh is not None else
                    'lagged derivative model found no accepted real-source trial; retry exact J at the same state and profiles')
                if adaptive_source_backtracking:next_line_search_fraction=1.
            if use_lagged and gn_step<.05:
                derivative_model.refresh('only a fresh source J may establish numerical convergence')
            if accepted_physical_endpoint_event:
                derivative_model.refresh('actual open physical endpoint branch changed; fresh complete source model required')
        record=dict(iteration=len(history),accepted=accepted,cost=before,next_cost=after if accepted else before,
            trial=trials,damping=damping,failure_reason=None if accepted else reason,
            maximum_undamped_GN_native_step_px=gn_step,
            scaled_gradient_maximum=float(np.max(np.abs(gradient)/np.sqrt(diagonal))),elapsed_seconds=elapsed())
        if accepted_trial_policy is not None:
            record['explicit_source_trial_acceptance']=accepted_trial_policy
        if accepted_physical_endpoint_event:
            record['actual_physical_endpoint_transition']=dict(physical_endpoint_event['report'],
                accepted_with_original_truth_physical_and_complete_source_cost_gates=True,
                complete_source_J_refresh_required_before_next_direction=True)
        if adaptive_source_backtracking:
            record['adaptive_source_backtracking']=dict(initial_line_search_fraction=line_search_seed,
                next_line_search_fraction=next_line_search_fraction,
                actual_to_predicted_reduction=accepted_source_reduction_ratio,
                next_seed_rule='double_accepted_fraction_up_to_one_if_rho_at_least_0.75_else_accepted_fraction',
                real_source_cost_guard_and_trial_count_unchanged=True)
        if diagnostic_values is not None:record['source_state_diagnostics']=diagnostic_values(q.copy())
        if len(fixed):record['explicit_unobserved_held_state_indices_fixed']=fixed.tolist()
        if streaming_report is not None:record['streaming_source_linearization']=streaming_report
        if derivative_model is not None:
            record['source_jacobian_reuse']=derivative_model.report()
            if uphill_lagged_refresh is not None:record['actual_uphill_lagged_refresh']=uphill_lagged_refresh
        if conditional_view_backtracking:
            record['conditional_view_backtracking']=dict(enabled=True,
                accepted_line_search_mode=accepted_search_mode,
                conditional_view_solve=None if conditional_step is None else conditional_step.report,
                rejected_conditional_search_falls_back_to_original_whole_step=True,
                actual_trial_step_used_for_prediction_and_rho=True,
                all_original_physical_and_truth_guards_and_full_source_cost_required=True,
                original_QP_bounds_and_curvature_tightening_retained=True,
                new_actual_trial_curvature_callback_retained=True,
                unchanged_profiles_source_rows_sigma_masks_parameters_and_GN_reference=True)
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
        history.append(record);run_stage('checkpoint_write_total',save)
        if verbose:print(_json_text(dict(stage='joint_full_source_iteration',**record)),flush=True)
        if not accepted:
            history[-1]=copy.deepcopy(record)
            _write_json(output/'iterations.json',history_writer.commit(history))
        if accepted:
            stop=run_stage('accepted_source_review_total',review,'accepted',record)
            if stage_times is not None:
                record['complete_cycle_elapsed_seconds']=elapsed()
            # Detach all nested reports before the next iteration can reuse
            # runtime report objects. Only the current tail receives timings.
            history[-1]=copy.deepcopy(record)
            _write_json(output/'iterations.json',history_writer.commit(history))
            if stop is not None:status='stopped_by_source_review';break
            stop=early_control_stop()
            if stop is not None:status='stopped_by_control';break
        if not accepted:
            if derivative_model is not None and use_lagged:
                repeat_frozen_profiles=True
                continue
            status='no_accepted_trial';break
        if accepted_trial_policy is not None and accepted_trial_policy.get('mode')=='truth_restoration':
            continue
        if gn_step<.05:
            if derivative_model is not None and use_lagged:
                continue
            if numerical_convergence_guard is not None and not numerical_convergence_guard(q):
                continue
            status='numerical_converged';break
    save();result=JointSourceResult(q.copy(),history,status,stop,damping,elapsed(),checkpoint)
    _write_json(output/'solver_result.json',dict(status=status,stop_reason=stop,damping=damping,seconds=result.seconds,
        accepted_iterations=sum(r['accepted']for r in history),checkpoint=checkpoint,accuracy_accepted=False,
        goal_paused=False,source_sigma_roles_and_geometry_owned_by_caller=True))
    return result
