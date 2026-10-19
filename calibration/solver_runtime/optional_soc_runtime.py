"""Stop an optional SOC numerical failure while retaining a verified direction.

This does not certify mathematical infeasibility.  Initial QPs and corrections
without a previously verified full-row KKT direction keep every existing
fallback.  Only a measured, strictly tighter SOC may stop after its restricted
route fails to supply a step passing the original complete KKT checks.  The
original line search, guards, costs and acceptance still decide whether to use
the retained direction.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from contextvars import ContextVar
import functools
import hashlib
import inspect
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np


_optional_soc = ContextVar("rinkspace_optional_soc_numerical_context", default=None)
_outer_soc_measurement = ContextVar("rinkspace_optional_soc_actual_face_measurement", default=None)
_retained_soc_face_model = ContextVar("rinkspace_retained_SOC_face_model", default=None)
_QP_calls = itertools.count(1)


class OptionalSOCNumericalFailure(RuntimeError):
    def __init__(self, report):
        self.optional_soc_report = report
        super().__init__(
            "Optional measured SOC restricted route failed: "
            + report["restricted_failure_reason"]
            + (" (" + report["status"] + ")" if report["status"] is not None else "")
            + "; retain the previous full-KKT verified direction "
            "for original backtracking (mathematical infeasibility not certified)")


def _digest(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def _previous_full_KKT_verified(linear):
    report = linear.report.get("shared_linear_domain_step", {})
    from .original_halfspace_primal import original_full_qp_kkt_verified
    return original_full_qp_kkt_verified(report,linear.step)


def original_nonlinear_step_face_feasibility(matrix, lower, shared_step,
                                            responses, actual_carriers,*,actual_group_violations=None):
    """Measure every unchanged step face using the already evaluated carriers.

    Native/wide/confirmed rows use actual nonlinear queries. The other rows
    retain their original affine response; the original physical trial guard
    is still required later. This adds no source evaluation or support margin.
    """
    if actual_group_violations is not None:
        from structured_programme_source_candidate.source_step_curvature_tightening import dynamic_source_group_merits
        return dynamic_source_group_merits(matrix,lower,shared_step,responses,actual_group_violations)
    lower=np.asarray(lower,float);step=np.asarray(shared_step,float)
    if (lower.ndim!=1 or matrix.shape!=(len(lower),len(step))
            or not np.isfinite(lower).all() or not np.isfinite(step).all()):
        raise ValueError('Complete original step faces and finite shared direction required')
    linear_slack=matrix@step-lower
    slack=linear_slack.copy();measured=np.zeros(len(lower),bool);families={}
    from structured_programme_source_candidate.source_step_curvature_tightening import actual_carrier_step_response
    for family,metadata in responses.items():
        begin,end=metadata['row_start'],metadata['row_stop']
        base=np.asarray(metadata['nominal_carrier'],float)
        actual=np.asarray(actual_carriers[family],float)
        if (not 0<=begin<=end<=len(lower) or np.any(measured[begin:end])
                or actual.shape!=base.shape
                or not np.isfinite(actual).all()):
            raise ValueError('Actual complete source carrier and original row identity required')
        response=actual_carrier_step_response(metadata,actual)
        if response.shape!=(end-begin,):
            raise ValueError('Actual source face map changed')
        slack[begin:end]=response-lower[begin:end];measured[begin:end]=True
        excess=np.maximum(-slack[begin:end],0.)
        families[family]=dict(rows=end-begin,maximum_violation_px=float(excess.max(initial=0.)),
            total_violation_px=float(excess.sum()),violated_rows=int(np.count_nonzero(excess)))
    affine=np.flatnonzero(~measured)
    if not np.isfinite(slack).all():
        raise ValueError('Actual original step-face responses must remain finite')
    excess=np.maximum(-slack,0.)
    predicted_excess=np.maximum(-linear_slack,0.)
    return dict(maximum_violation_px=float(excess.max(initial=0.)),
        total_violation_px=float(excess.sum()),violated_rows=int(np.count_nonzero(excess)),
        predicted_maximum_violation_px=float(predicted_excess.max(initial=0.)),
        predicted_total_violation_px=float(predicted_excess.sum()),
        complete_original_step_rows=len(slack),actual_nonlinear_source_rows=int(measured.sum()),
        original_affine_rows=int(len(affine)),families=families,
        original_targets_not_accumulated_SOC_targets=True,
        actual_physical_trial_guard_still_required=True,extra_source_evaluations=0)


def _install_original_nonlinear_face_observer():
    from structured_programme_source_candidate import source_step_curvature_tightening as module
    original=module.measured_outward_step_tightening
    if getattr(original,'_observes_original_nonlinear_faces',False):return
    @functools.wraps(original)
    def measured(matrix,lower,shared_step,responses,actual_carriers,*,actual_group_violations=None):
        result=(original(matrix,lower,shared_step,responses,actual_carriers)if actual_group_violations is None else
            original(matrix,lower,shared_step,responses,actual_carriers,actual_group_violations=actual_group_violations))
        context=_outer_soc_measurement.get()
        if context is not None:
            context['measurement']=original_nonlinear_step_face_feasibility(
                matrix,lower,shared_step,responses,actual_carriers,actual_group_violations=actual_group_violations)
        return result
    measured._observes_original_nonlinear_faces=True
    module.measured_outward_step_tightening=measured


def observe_original_physical_SOC_measurement(measurement):
    """Record the same proposal's physical merit in its separate units."""
    context = _outer_soc_measurement.get()
    if context is not None:
        context['physical_measurement'] = measurement


def _physical_first_SOC_progress(current, previous):
    """Physical feasibility precedes comparing source-pixel SOC merits.

    An inside-course proposal cannot replace a physical candidate because it
    has smaller pixel residuals.  Conversely reaching the original physical
    domain can temporarily increase a source merit; subsequent SOC must then
    correct the source faces before any original final guard accepts it.
    """
    if current is None:
        return None
    if current.get('certified',True) is not True:
        return dict(progress=False, nonworsening=False, physical_measurement_unavailable=True)
    if previous is None:
        return None if current['valid'] else dict(progress=True, nonworsening=True)
    if previous['valid']:
        return None if current['valid'] else dict(progress=False, nonworsening=False)
    if current['valid']:
        return dict(progress=True, nonworsening=True)
    before = np.array([previous['maximum_violation_working_m'], previous['total_violation_working_m']])
    after = np.array([current['maximum_violation_working_m'], current['total_violation_working_m']])
    roundoff = 8 * np.finfo(float).eps * np.maximum(1., np.maximum(before, after))
    nonworsening = bool(np.all(after <= before + roundoff))
    progress = bool(nonworsening and np.any(after < before - roundoff))
    return dict(progress=progress, nonworsening=nonworsening,
        physical_merit_comparison_roundoff_working_m=roundoff.tolist())


def _install_physical_fraction_SOC(joint):
    original = joint._needs_trial_fraction_SOC
    if getattr(original, '_coupled_original_physical_SOC', False):
        return
    @functools.wraps(original)
    def needed(linear, domain):
        if not callable(domain.get('at_trial_fraction')):
            return original(linear, domain)
        records = linear.report.get('sequential_measured_shared_source_curvature_tightening', {}).get(
            'actual_original_nonlinear_feasibility_progress', {}).get('measurements', [])
        if any((record.get('current_complete_KKT_direction_retained')
            and record.get('actual_original_physical_domain_feasibility', {}).get('valid') is False)
            or (record.get('complete_original_KKT_verified')
                and record.get('original_nonlinear_measurement_unavailable'))
            for record in records):
            return True
        return original(linear, domain)
    needed._coupled_original_physical_SOC = True
    joint._needs_trial_fraction_SOC = needed


def _sufficient_SOC_feasibility_reduction(agreement, maximum_roundoff, total_roundoff):
    """Require a useful fraction of each promised actual feasibility reduction.

    The one-quarter sufficient-progress fraction is an algorithmic condition
    for another optional correction. It is distinct from the existing one-
    quarter symmetric prediction-error rule that controls damping reduction.
    A better verified direction is retained before this check; all final
    physical, truth and objective gates remain authoritative.
    """
    fraction=.25
    if agreement is None:
        return dict(sufficient=True, first_measurement=True,
                    minimum_actual_to_predicted_reduction_fraction=fraction, merits={})
    merits={};promises=0
    for key,roundoff in [('maximum',maximum_roundoff),('total',total_roundoff)]:
        roundoff=float(roundoff)
        entry=agreement[key]
        predicted=float(entry['predicted_source_reduction'])
        actual=float(entry['actual_source_reduction'])
        if not np.isfinite([predicted,actual,roundoff]).all() or roundoff<0.:
            raise ValueError('Finite same-merit SOC predicted/actual reductions required')
        promised=predicted>roundoff
        promises+=int(promised)
        sufficient=(not promised or actual+roundoff>=fraction*predicted)
        merits[key]=dict(predicted_reduction_px=predicted,actual_reduction_px=actual,
            numerical_merit_roundoff_px=roundoff,predicted_reduction_resolvable=promised,
            actual_to_predicted_reduction=None if not promised else actual/predicted,
            sufficient=bool(sufficient))
    return dict(sufficient=bool(promises and all(v['sufficient']for v in merits.values())),
        first_measurement=False,minimum_actual_to_predicted_reduction_fraction=fraction,
        compared_quantities='Same original-face maximum and total violations before/after current candidate',
        no_source_support_or_acceptance_tolerance_change=True,merits=merits)


def _with_outer_nonlinear_progress(original_SOC,q,linear,selected,domain,linear_options,
                                   normal,gradient,diagonal,tightening,solve_linear_step):
    """Keep a better verified candidate even when further cuts are unreliable.

    Increasing the accumulated numerical lower targets is not nonlinear
    progress. Both actual feasibility merits must be nonworsening and one
    must decrease beyond roundoff. The existing .25 model trust rule controls
    lambda reduction. Another correction requires sufficient actual
    reduction of both promised feasibility merits.
    This only abandons further optional SOC;
    all original full trial guards, source cost and acceptance remain in use.
    """
    _retained_soc_face_model.set(None)
    current=linear;retained=None;retained_lower=None;best=None;previous=None;records=[];stalled=None;unreliable_retained=None
    best_physical=None
    def measured_tightening(value,step,columns,original_domain):
        nonlocal retained,retained_lower,best,previous,stalled,unreliable_retained,best_physical
        context={};token=_outer_soc_measurement.set(context)
        try:correction=tightening(value,step,columns,original_domain)
        finally:_outer_soc_measurement.reset(token)
        measurement=context.get('measurement')
        physical=context.get('physical_measurement')
        if 'original_signed_margin_rows' in original_domain and physical is None:
            # The original callback can decline an out-of-domain full-step
            # measurement.  Keep its QP direction only for authoritative
            # trial guards/backtracking; do not turn missing data into a
            # nonlinear certificate or abort before shorter scales.
            physical=dict(valid=False,certified=False,measurement_unavailable=True,
                original_physical_domain_not_certified=True)
        prediction=float(-np.asarray(gradient)@step-.5*step@(normal@step))
        entry=dict(measurement=len(records),direction_sha256=_digest(step),
            original_quadratic_model_decrease=prediction,
            complete_original_KKT_verified=_previous_full_KKT_verified(current),
            actual_original_face_feasibility=measurement,
            another_measured_correction_requested=correction is not None)
        if physical is not None:
            entry['actual_original_physical_domain_feasibility']=physical
        if ('original_signed_margin_rows' in original_domain
                and (measurement is None or physical.get('certified') is not True)):
            entry.update(original_nonlinear_measurement_unavailable=True,
                current_complete_KKT_direction_retained=entry['complete_original_KKT_verified'],
                direction_retained_only_for_original_final_guard_and_backtracking=True)
        # Unknown third-party callbacks retain the historical route; no
        # nonlinear certificate is claimed without complete actual carriers.
        if measurement is not None:
            maximum=measurement['maximum_violation_px'];total=measurement['total_violation_px']
            maximum_roundoff=8*np.finfo(float).eps*max(1.,maximum,0. if best is None else best[0])
            total_roundoff=8*np.finfo(float).eps*max(1.,total,0. if best is None else best[1])
            nonworsening=(best is None or (maximum<=best[0]+maximum_roundoff
                and total<=best[1]+total_roundoff))
            progress=(best is None or (nonworsening and
                (maximum<best[0]-maximum_roundoff or total<best[1]-total_roundoff)))
            agreement=None;reliable=True
            if previous is not None:
                import z190_joint_source_solver as joint
                agreement={}
                for key in ('maximum','total'):
                    actual_before=previous[key+'_violation_px']
                    agreement[key]=joint.restoration_source_model_agreement(
                        actual_before-measurement['predicted_'+key+'_violation_px'],
                        actual_before-measurement[key+'_violation_px'])
                reliable=any(r['reliable_for_restoration_lambda_decrease']
                    and r['predicted_source_reduction']>0. and r['actual_source_reduction']>0.
                    for r in agreement.values())
            sufficient=_sufficient_SOC_feasibility_reduction(agreement,maximum_roundoff,total_roundoff)
            source_progress=progress;source_nonworsening=nonworsening;source_sufficient=sufficient
            physical_progress=_physical_first_SOC_progress(physical,best_physical)
            if physical_progress is not None:
                nonworsening=physical_progress['nonworsening']
                progress=physical_progress['progress']
                sufficient=dict(sufficient=bool(progress),
                    original_physical_domain_feasibility_precedes_source_merit=True,
                    physical_and_source_merits_not_added_or_rescaled=True)
                # A physical-domain transition is not a trustworthy source
                # reduction prediction for decreasing numerical damping.
                reliable=False
                entry['physical_first_SOC_progress']=physical_progress
            # A source correction can have useful pixel progress yet reveal
            # another physical curvature defect.  It is not retained, but
            # may receive the measured correction of those same physical
            # rows before falling back to the last physical candidate.
            pending_physical_repair=bool(physical is not None and best_physical is not None
                and best_physical['valid'] and not physical['valid']
                and correction is not None and source_progress and source_nonworsening
                and source_sufficient['sufficient'])
            entry.update(sufficient_actual_to_predicted_feasibility_reduction=sufficient,
                improves_best_observed_actual_feasibility=bool(progress),
                maximum_and_total_actual_feasibility_nonworsening=bool(nonworsening),
                prior_best_maximum_violation_px=None if best is None else best[0],
                prior_best_total_violation_px=None if best is None else best[1],
                maximum_merit_roundoff=maximum_roundoff,total_merit_roundoff=total_roundoff,
                original_face_reduction_model_agreement=agreement,
                original_face_reduction_model_is_reliable=bool(reliable))
            verified=entry['complete_original_KKT_verified']
            pending_physical_repair &= verified
            entry['temporary_unphysical_candidate_requires_existing_row_SOC']=pending_physical_repair
            if verified and progress:
                retained=current
                retained_lower=np.asarray(linear_options['global_step_constraint_lower'],float).copy()
                best=(maximum,total)
                previous=measurement
                best_physical=physical
                entry['current_complete_KKT_direction_retained']=True
                if not reliable:unreliable_retained=entry
                entry['prediction_disagreement_only_holds_damping']=not reliable
            else:
                entry['current_complete_KKT_direction_retained']=False
            if retained is not None and (not progress or not verified or not sufficient['sufficient'])and not pending_physical_repair:
                stalled=entry
                entry['further_optional_correction_abandoned']=correction is not None
                entry['current_candidate_discarded']=current is not retained
                correction=None
        records.append(entry)
        print(json.dumps(dict(stage='optional_SOC_actual_original_face_progress',
            wall_time_ns=time.time_ns(),**entry)),flush=True)
        return correction
    def solve_correction(*args,**options):
        nonlocal current
        current=solve_linear_step(*args,**options)
        return current
    result=original_SOC(q,linear,selected,domain,linear_options,normal,gradient,diagonal,
                        measured_tightening,solve_correction)
    outer_report=result.report['sequential_measured_shared_source_curvature_tightening']
    if (stalled is None and retained is not None and records
            and records[-1].get('temporary_unphysical_candidate_requires_existing_row_SOC')):
        stalled=records[-1]
        stalled['physical_SOC_repair_unresolved']=True
        stalled['current_candidate_discarded']=current is not retained
    if stalled is not None:
        result=retained
        linear_options['global_step_constraint_lower']=retained_lower
        outer_report['stop_reason']=('optional_original_physical_SOC_unresolved_keep_verified_candidate'
            if stalled.get('physical_SOC_repair_unresolved')else
            'optional_actual_original_nonlinear_feasibility_unverified_KKT'
            if not stalled['complete_original_KKT_verified']else
            ('optional_actual_original_nonlinear_feasibility_insufficient_model_reduction'
             if stalled['improves_best_observed_actual_feasibility']else
             'optional_actual_original_nonlinear_feasibility_nonprogress'))
        outer_report['stopping_candidate_sha256']=stalled['direction_sha256']
        outer_report['stopping_candidate_retained']=not stalled['current_candidate_discarded']
        outer_report['retained_complete_KKT_direction_sha256']=_digest(result.step)
        result.report['sequential_measured_shared_source_curvature_tightening']=outer_report
    outer_report['actual_original_nonlinear_feasibility_progress']=dict(
        schema='rinkspace.optional-SOC-actual-original-nonlinear-face-progress.v4',
        measurements=records,optional_correction_abandoned=stalled is not None,
        no_iteration_or_time_limit=True,no_original_support_or_target_relaxation=True,
        minimum_actual_to_predicted_feasibility_reduction_fraction=.25,
        sufficient_progress_is_distinct_from_symmetric_prediction_error_trust=True,
        initial_required_QP_unchanged=True,mathematical_infeasibility_certified=False,
        original_physical_full_source_cost_truth_and_backtracking_still_required=True)
    if unreliable_retained is not None:
        _retained_soc_face_model.set(dict(
            retained_complete_KKT_direction_sha256=_digest(result.step),
            original_q_sha256=_digest(q),
            original_face_reduction_model_agreement=unreliable_retained['original_face_reduction_model_agreement'],
            reliable_for_restoration_lambda_decrease=False,
            current_candidate_retained=True,
            later_improved_candidates_may_have_been_retained=True,
            actual_full_trial_acceptance_still_required=True))
    return result


def _install_retained_SOC_damping_trust(joint):
    """Use face model reliability only for the existing numerical LM update."""
    original=joint.accepted_schur_damping
    if getattr(original,'_retained_SOC_face_model_trust',False):return
    @functools.wraps(original)
    def accepted(damping,whole_step_fraction,*,backtracking_aware=False,
                 truth_restoration=False,restoration_model_agreement=None):
        face_model=_retained_soc_face_model.get()
        _retained_soc_face_model.set(None)
        if truth_restoration and face_model is not None:
            if restoration_model_agreement is None:
                restoration_model_agreement={}
            restoration_model_agreement['source_model_reliable_before_SOC_face_check']=(
                restoration_model_agreement.get('reliable_for_restoration_lambda_decrease'))
            restoration_model_agreement['retained_SOC_original_face_model']=face_model
            restoration_model_agreement['reliable_for_restoration_lambda_decrease']=False
        return original(damping,whole_step_fraction,backtracking_aware=backtracking_aware,
            truth_restoration=truth_restoration,
            restoration_model_agreement=restoration_model_agreement)
    accepted._retained_SOC_face_model_trust=True
    joint.accepted_schur_damping=accepted


def restricted_original_KKT_report(matrix, gradient, rows, targets, row_scale,
                                   normalized_rows, selected, candidate,
                                   candidate_dual, candidate_slack, tolerance):
    """Check the solved restricted QP before using it to separate more faces.

    Unselected rows deliberately have zero duals.  Their primal feasibility is
    still checked by the existing separation loop, not required by this gate.
    All thresholds and original-unit equations are those of its final gate.
    """
    chosen = np.asarray(selected, dtype=int)
    dual = np.asarray(candidate_dual, float)
    original_dual = dual / row_scale
    stationarity = matrix @ candidate + gradient - normalized_rows.T @ dual
    normalized_slack = candidate_slack[chosen] / row_scale[chosen]
    complementarity = candidate_slack[chosen] * original_dual[chosen]
    stationarity_tolerance = 1e-9 * max(
        1., float(np.max(abs(gradient))), float(np.max(abs(matrix @ candidate))))
    complementarity_tolerance = 1e-9 * max(
        1., abs(float(candidate @ matrix @ candidate)),
        abs(float(gradient @ candidate)), abs(float(targets @ original_dual)))
    minimum_slack = float(np.min(normalized_slack))
    maximum_stationarity = float(np.max(abs(stationarity)))
    minimum_dual = float(np.min(dual[chosen]))
    absolute_complementarity = float(np.sum(abs(complementarity)))
    finite = bool(np.isfinite(candidate).all() and np.isfinite(dual).all()
                  and np.isfinite(candidate_slack[chosen]).all()
                  and np.isfinite(stationarity).all())
    from .original_halfspace_primal import original_halfspace_primal
    primal={}
    if finite:
        _,_,_,primal=original_halfspace_primal(rows[chosen],targets[chosen],candidate,
                                              slack=candidate_slack[chosen])
    return dict(
        accepted=bool(finite and primal.get('accepted') is True
                      and maximum_stationarity <= stationarity_tolerance
                      and minimum_dual >= -tolerance
                      and absolute_complementarity <= complementarity_tolerance),
        selected_row_count=int(len(chosen)), original_selected_rows_checked=True,
        minimum_selected_normalized_slack=minimum_slack,
        selected_original_primal=primal,
        maximum_original_stationarity=maximum_stationarity,
        original_stationarity_tolerance=stationarity_tolerance,
        minimum_normalized_dual=minimum_dual, dual_tolerance=tolerance,
        original_complementarity_absolute_sum=absolute_complementarity,
        original_complementarity_tolerance=complementarity_tolerance,
        unselected_rows_still_checked_by_original_separation=True,
        arbitrary_iteration_or_time_limit_used=False)


def _optional_restricted_KKT_report(*args):
    if _optional_soc.get() is None:
        return None
    return restricted_original_KKT_report(*args)


def _optional_strictly_tightened_face_seed(row_count):
    context=_optional_soc.get()
    if context is None:return np.empty(0,dtype=int)
    rows=np.asarray(context.get('strictly_tightened_original_faces',()),dtype=int)
    if rows.ndim!=1 or np.any(rows<0)or np.any(rows>=row_count):
        raise ValueError('Optional SOC tightened faces must use the current original row map')
    return rows


def _optional_complete_feasibility_progress(normalized_slack, tolerance):
    """Abandon only an optional correction when both full-domain merits stall.

    Constraint separation need not improve these merits monotonically. This
    is a safe optional performance retreat, never an infeasibility certificate
    or an early successful solve. The caller retains its verified full-domain
    direction and still runs all original nonlinear backtracking checks.
    """
    context = _optional_soc.get()
    if context is None:
        return None
    slack = np.asarray(normalized_slack, float)
    allowed=np.asarray(tolerance,float)
    # Legacy callers pass a scalar dual precision; the current QP supplies
    # one original-row feasibility threshold per normalized row.
    allowed=10*allowed if allowed.ndim==0 else allowed
    if allowed.ndim and allowed.shape!=slack.shape:
        raise ValueError('Optional progress needs the complete per-row primal thresholds')
    excess = np.maximum(-slack - allowed, 0.)
    current = dict(maximum_full_primal_excess=float(excess.max()),
                   total_full_primal_excess=float(excess.sum()))
    previous = context.get('previous_full_primal_merit')
    maximum_roundoff = 8*np.finfo(float).eps*max(1.,current['maximum_full_primal_excess'],
        0. if previous is None else previous['maximum_full_primal_excess'])
    total_roundoff = 8*np.finfo(float).eps*max(1.,current['total_full_primal_excess'],
        0. if previous is None else previous['total_full_primal_excess'])
    progressing = (previous is None or current['maximum_full_primal_excess'] == 0.
        or current['maximum_full_primal_excess'] < previous['maximum_full_primal_excess'] - maximum_roundoff
        or current['total_full_primal_excess'] < previous['total_full_primal_excess'] - total_roundoff)
    report = dict(continue_optional_separation=bool(progressing), current=current,
        previous=previous, merit_maximum_roundoff=maximum_roundoff,
        merit_sum_roundoff=total_roundoff,
        per_original_row_primal_thresholds_used=bool(allowed.ndim==1),
        all_original_rows_checked=int(len(slack)),
        mathematical_infeasibility_certified=False,
        initial_required_QP_unchanged=True,
        previous_complete_KKT_direction_and_original_backtracking_retained=True,
        arbitrary_iteration_or_time_limit_used=False)
    context['previous_full_primal_merit'] = current
    return report


def _capture_optional_QP_problem(matrix, gradient, lower, upper,
                                 constraint_matrix, constraint_lower):
    """Explicit diagnostic only: retain one small reduced problem, never J."""
    context = _optional_soc.get()
    directory = os.environ.get('RINKSPACE_OPTIONAL_SOC_CAPTURE_DIR')
    if context is None or not directory:
        return
    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stem = destination / ('optional_SOC_reduced_' + context['original_q_sha256'])
    if stem.with_suffix('.json').exists():
        return
    arrays = dict(matrix=matrix, gradient=gradient, lower=lower, upper=upper,
                  constraint_matrix=constraint_matrix, constraint_lower=constraint_lower)
    array_path = stem.with_suffix('.npz')
    if array_path.exists():
        raise FileExistsError('Incomplete prior optional SOC diagnostic capture is preserved')
    np.savez(array_path, **arrays)
    import z190_global_schur_linear_step as qp
    warm = getattr(qp._linearly_constrained_shared_step, '_last_verified_separation_faces', None)
    receipt = dict(schema='rinkspace.original-optional-SOC-reduced-QP-capture.v1',
        original_q_sha256=context['original_q_sha256'],
        previous_direction_sha256=context['previous_direction_sha256'],
        previous_direction_full_original_KKT_verified=True,
        strictly_tighter_measured_SOC_faces=context['strictly_tighter_measured_SOC_faces'],
        array_path=str(array_path), array_file_sha256=hashlib.sha256(array_path.read_bytes()).hexdigest(),
        arrays={key:dict(shape=list(np.shape(value)), dtype=str(np.asarray(value).dtype),
                        sha256=_digest(value)) for key,value in arrays.items()},
        warm_faces=warm, complete_source_J_or_normal_saved=False,
        reduced_QP_only=True, diagnostic_only=True)
    stem.with_suffix('.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    print(json.dumps(dict(stage='optional_SOC_reduced_QP_captured', receipt=str(stem.with_suffix('.json')))), flush=True)


def run_logged_QP_backend(solver, phase, constraint_rows, parameter_count, selected_faces=None):
    """Log each original backend call without changing settings or callbacks."""
    call = next(_QP_calls)
    context = _optional_soc.get()
    if context is not None and phase == 'restricted_affine' and selected_faces is not None:
        context['last_restricted_face_rows'] = tuple(int(i) for i in selected_faces)
    report = dict(QP_call=call, QP_phase=phase, constraint_rows=int(constraint_rows),
                  parameter_count=int(parameter_count), pid=os.getpid(),
                  optional_measured_SOC=context is not None)
    if context is not None:
        report["original_q_sha256"] = context["original_q_sha256"]
    linear_solver = solver.get_info().linsolver
    report.update(linear_solver_backend=linear_solver.name,
                  linear_solver_threads=int(linear_solver.threads))
    print(json.dumps(dict(stage="QP_backend_begin", wall_time_ns=time.time_ns(), **report)), flush=True)
    begun = time.perf_counter()
    result = None
    error = None
    try:
        result = solver.solve()
        return result
    except BaseException as exc:
        error = exc
        raise
    finally:
        finished = dict(stage="QP_backend_end", wall_time_ns=time.time_ns(),
                        seconds=time.perf_counter() - begun, **report)
        if result is not None:
            finished.update(status=str(result.status), iterations=int(result.iterations))
        if error is not None:
            finished.update(error_type=type(error).__name__, error=str(error))
        print(json.dumps(finished), flush=True)


def _optional_restricted_infeasibility(result,selected,rows,targets,scale,
        affine_rows,affine_targets,row_scale,tolerance,settings):
    """Classify a failed optional subproblem without another backend or dump."""
    if _optional_soc.get() is None:return None
    from .qp_infeasibility_certificate import verify_restricted_infeasibility
    report,dual=verify_restricted_infeasibility(result,selected,rows,targets,scale,
        affine_rows,affine_targets,row_scale,tolerance,settings)
    report['only_current_tightened_linear_subproblem_not_original_nonlinear_model']=True
    report['complete_IPM_or_new_source_evaluation_performed']=False
    if dual is not None:
        indices=np.asarray(selected,int)
        report['original_selected_row_multipliers']=[dict(original_row=int(row),
            normalized_original_multiplier=float(dual[row]),original_lower=float(targets[row]))
            for row in indices if dual[row]>0.]
    return report


def stop_failed_optional_SOC(separation_report, rows):
    """Called only when the original restricted route returned no valid step."""
    context = _optional_soc.get()
    if context is None or separation_report.get("accepted") is not False:
        return
    rounds = separation_report.get("rounds", [])
    last = rounds[-1] if rounds else {}
    status = last.get("status")
    ray=last.get('original_restricted_infeasibility_ray')
    report = dict(
        schema="rinkspace.optional-measured-SOC-numerical-failure.v2",
        status=None if status is None else str(status),
        restricted_failure_reason=separation_report.get("reason", "unreported_restricted_failure"),
        original_restricted_failure_report=deepcopy(separation_report),
        restricted_selected_row_count=last.get("selected_rows"),
        complete_original_row_count=len(rows), restricted_iterations=last.get("iterations"),
        previous_direction_full_original_KKT_verified=True,
        previous_direction_sha256=context["previous_direction_sha256"],
        original_q_sha256=context["original_q_sha256"],
        strictly_tighter_measured_SOC_faces=context["strictly_tighter_measured_SOC_faces"],
        previous_verified_direction_retained_for_original_backtracking=True,
        mathematical_infeasibility_certified=False,
        numerical_tightened_linear_infeasibility_ray_verified=bool(ray and ray.get('accepted')),
        original_restricted_infeasibility_ray=ray,
        full_IPM_retry_skipped_only_for_optional_SOC=True,
        initial_QP_and_no_verified_direction_keep_complete_IPM=True,
        original_H_g_source_rows_constraints_bounds_and_final_KKT_gate_unchanged=True,
        original_line_search_physical_guard_cost_and_acceptance_retained=True,
        wall_time_ns=time.time_ns())
    context["failure_report"] = report
    print(json.dumps(dict(stage="optional_SOC_restricted_numerical_failure", **report)), flush=True)
    raise OptionalSOCNumericalFailure(report)


def install_optional_SOC_numerical_failure_route(*, expand_separation_batches=False, parallel_backend=None):
    """Install process-only candidate wrappers; never rewrite signed sources."""
    from calibration.solver_runtime import install
    install()
    import z190_global_schur_linear_step as qp
    import z190_joint_source_solver as joint
    original_qp = qp._linearly_constrained_shared_step
    if parallel_backend not in (None, 'faer'):
        raise ValueError('Unsupported original QP parallel factorization backend')
    if getattr(original_qp, "_optional_SOC_numerical_failure_route", False):
        if original_qp._optional_SOC_installation_report.get('geometric_violated_face_batches', False) != bool(expand_separation_batches):
            raise RuntimeError('QP separation scheduling was already installed with another mode')
        if original_qp._optional_SOC_installation_report.get('parallel_factorization_backend') != parallel_backend:
            raise RuntimeError('QP factorization backend was already installed with another mode')
        return original_qp._optional_SOC_installation_report
    if getattr(original_qp, "_verified_restricted_infeasibility", False):
        raise RuntimeError("Replace the diagnostic certificate installer before installing the optional SOC route")
    source = inspect.getsource(original_qp)
    tree = ast.parse(source)
    function = tree.body[0]
    function.body.insert(1, ast.parse(
        '_capture_optional_QP_problem(matrix, gradient, lower, upper, constraint_matrix, constraint_lower)').body[0])
    attempts = [node for node in function.body if isinstance(node, ast.Try)
                and len(node.body) == 1 and isinstance(node.body[0], ast.Assign)
                and ast.unparse(node.body[0].value) == "separated_affine_step()"]
    if len(attempts) != 1:
        raise RuntimeError("Optional SOC route expected exactly one original acceleration try")
    complete_fallback_gates = [node for node in function.body if isinstance(node, ast.If)
                              and ast.unparse(node.test) == "accelerated is None"]
    if len(complete_fallback_gates) != 1:
        raise RuntimeError("Optional SOC route expected exactly one original complete-fallback gate")
    complete_fallback_gates[0].body.insert(
        0, ast.parse("_stop_failed_optional_SOC(separation_report, rows)").body[0])
    nested=next(node for node in function.body if isinstance(node,ast.FunctionDef)
        and node.name=='separated_affine_step')
    status_gates=[node for node in ast.walk(nested)if isinstance(node,ast.If)
        and ast.unparse(node.test)=="str(result.status) not in ('Solved', 'AlmostSolved')"]
    if len(status_gates)!=1:raise RuntimeError('Exactly one original restricted numerical status gate required')
    status_gates[0].body[:0]=ast.parse('''
infeasibility_ray = _optional_restricted_infeasibility(result, selected, rows, targets,
    scale, affine_rows, affine_targets, row_scale, tolerance, settings)
if infeasibility_ray is not None:
    entry['original_restricted_infeasibility_ray'] = infeasibility_ray
''').body
    # A root-owned cancellation helper may raise a BaseException outside the
    # original ValueError/RuntimeError handler.  Its exception is never caught
    # or reclassified by this optional correction route.
    class BackendCalls(ast.NodeTransformer):
        phases = {"restricted": ("restricted_affine", "len(selected)"),
                  "solver": ("full_diagonal", "len(targets)"),
                  "fallback": ("full_affine", "len(targets)")}

        def __init__(self):
            self.changed = []

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if (isinstance(node.func, ast.Attribute) and node.func.attr == "solve"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in self.phases):
                name = node.func.value.id
                phase, count = self.phases[name]
                if node.args or node.keywords:
                    raise RuntimeError("QP logger requires the original argument-free solve call")
                self.changed.append(name)
                return ast.copy_location(ast.Call(
                    func=ast.Name(id="_run_logged_QP_backend", ctx=ast.Load()),
                    args=[node.func.value, ast.Constant(value=phase),
                          ast.parse(count, mode="eval").body,
                          ast.parse("len(value)", mode="eval").body,
                          ast.Name(id='selected', ctx=ast.Load()) if name == 'restricted' else ast.Constant(None)],
                    keywords=[]), node)
            return node

    backend_calls = BackendCalls()
    tree = backend_calls.visit(tree)
    if sorted(backend_calls.changed) != ["fallback", "restricted", "solver"]:
        raise RuntimeError("QP logger expected exactly the three original restricted/full solve sites")
    class RestrictedWitnessGate(ast.NodeTransformer):
        def __init__(self):
            self.sites = []

        def visit_If(self, node):
            # Public QR residual refinement must have its opportunity before
            # an optional candidate is judged. This gate still precedes every
            # expansion of the selected set, and never weakens its KKT checks.
            if ast.unparse(node.test) == 'len(violated)':
                self.sites.append(node.lineno)
                gate = ast.parse('''
restricted_KKT = _optional_restricted_KKT_report(matrix, gradient, rows, targets,
    row_scale, normalized_rows, selected, candidate, candidate_dual, candidate_slack, tolerance)
if restricted_KKT is not None:
    entry['original_selected_KKT'] = restricted_KKT
    if not restricted_KKT['accepted']:
        separation_report.update(accepted=False, reason='restricted_original_selected_KKT_failure',
            rounds=history, elapsed_seconds=time.perf_counter()-begun,
            complete_original_row_count=len(rows), all_original_rows_checked_each_round=True,
            original_full_primal_dual_KKT_gate_retained=True)
        return None
    feasibility_progress = _optional_complete_feasibility_progress(normalized_candidate_slack, primal_tolerance/row_scale)
    entry['optional_all_original_primal_progress'] = feasibility_progress
    if not feasibility_progress['continue_optional_separation']:
        separation_report.update(accepted=False, reason='optional_complete_feasibility_progress_lost',
            rounds=history, elapsed_seconds=time.perf_counter()-begun,
            complete_original_row_count=len(rows), all_original_rows_checked_each_round=True,
            original_full_primal_dual_KKT_gate_retained=True,
            mathematical_infeasibility_certified=False)
        return None
''').body
                return [*gate, node]
            return self.generic_visit(node)

    gate = RestrictedWitnessGate()
    tree = gate.visit(tree)
    if len(gate.sites) != 1:
        raise RuntimeError('Exactly one original restricted full-row separation witness check required')
    class TightenedFaceSeed(ast.NodeTransformer):
        def __init__(self):self.sites=[]
        def visit_Assign(self,node):
            if (len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
                    and node.targets[0].id=='selected'
                    and ast.unparse(node.value)==
                    'np.union1d(np.union1d(initial, warm_rows), np.arange(len(constraint_lower), len(rows)))'):
                self.sites.append(node.lineno)
                node.value=ast.Call(func=ast.Attribute(value=ast.Name(id='np',ctx=ast.Load()),attr='union1d',ctx=ast.Load()),
                    args=[node.value,ast.parse('_optional_strictly_tightened_face_seed(len(rows))',mode='eval').body],keywords=[])
            return self.generic_visit(node)
    seed=TightenedFaceSeed();tree=seed.visit(tree)
    if len(seed.sites)!=1:raise RuntimeError('Exactly one original restricted seed union required')
    # Growing the selected set by a fixed n/2 rows repeatedly solves nearly
    # identical increasing QPs. Keep the original worst-violation ordering,
    # but after each failed separation round allow the selected set to double.
    # Every original row is still checked each round and the unchanged full
    # primal/dual KKT certificate remains the only successful return route.
    if expand_separation_batches:
        class ExpandBatches(ast.NodeTransformer):
            def __init__(self):
                self.sites = []

            def visit_Assign(self, node):
                if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id == 'batch'
                        and ast.unparse(node.value) == 'min(max(1, len(value) // 2), len(violated))'):
                    self.sites.append(node.lineno)
                    node.value = ast.parse(
                        'min(max(1, len(value) // 2, len(selected)), len(violated))',
                        mode='eval').body
                return self.generic_visit(node)

        expand = ExpandBatches()
        tree = expand.visit(tree)
        if len(expand.sites) != 1:
            raise RuntimeError('Exactly one original violated-face scheduling batch is required')
    ast.fix_missing_locations(tree)
    namespace = dict(original_qp.__globals__)
    namespace.update(_stop_failed_optional_SOC=stop_failed_optional_SOC,
                     _run_logged_QP_backend=run_logged_QP_backend,
                     _optional_restricted_KKT_report=_optional_restricted_KKT_report,
                     _optional_strictly_tightened_face_seed=_optional_strictly_tightened_face_seed,
                     _optional_complete_feasibility_progress=_optional_complete_feasibility_progress,
                     _optional_restricted_infeasibility=_optional_restricted_infeasibility,
                     _capture_optional_QP_problem=_capture_optional_QP_problem)
    exec(compile(tree, str(Path(__file__).resolve()) + ":QP_candidate", "exec"), namespace)
    replacement = namespace[original_qp.__name__]
    if parallel_backend:
        serial_replacement = replacement
        parallel_tree = deepcopy(tree)
        sites = []
        class ParallelFactorization(ast.NodeTransformer):
            def visit_Assign(self, node):
                if (len(node.targets) == 1 and ast.unparse(node.targets[0]) == 'settings.max_threads'
                        and isinstance(node.value, ast.Constant) and node.value.value == 1):
                    sites.append(node.lineno)
                    return [node, *ast.parse("settings.direct_solve_method = 'faer'\nsettings.max_threads = _rinkspace_QP_threads").body]
                return self.generic_visit(node)
        parallel_tree = ParallelFactorization().visit(parallel_tree)
        if len(sites) != 1:
            raise RuntimeError('Exactly one original QP linear solver thread setting required')
        ast.fix_missing_locations(parallel_tree)
        parallel_namespace = dict(namespace, _rinkspace_QP_threads=max(1, os.cpu_count() or 1))
        exec(compile(parallel_tree, str(Path(__file__).resolve()) + ':parallel_QP_candidate', 'exec'), parallel_namespace)
        parallel_replacement = parallel_namespace[original_qp.__name__]
        @functools.wraps(serial_replacement)
        def replacement(*args, **kwargs):
            try:
                return parallel_replacement(*args, **kwargs)
            except OptionalSOCNumericalFailure:
                raise
            except Exception as error:
                print(json.dumps(dict(stage='parallel_QP_original_serial_fallback',
                    wall_time_ns=time.time_ns(), failed_backend=parallel_backend,
                    error_type=type(error).__name__, reason=str(error))), flush=True)
                return serial_replacement(*args, **kwargs)
    replacement.__dict__.update(original_qp.__dict__)
    original_SOC = joint._sequential_measured_shared_source_curvature
    _install_original_nonlinear_face_observer()
    _install_retained_SOC_damping_trust(joint)
    _install_physical_fraction_SOC(joint)

    def sequential_optional_SOC(q, linear, selected, domain, linear_options,
                                normal, gradient, diagonal, tightening, solve_linear_step):
        prior = linear
        prior_lower = np.asarray(linear_options["global_step_constraint_lower"], float).copy()
        original_lower = np.asarray(domain["lower"], float)
        original_matrix = linear_options.get("global_step_constraint_matrix")
        failure_reports = []

        def solve_optional_correction(*args, **options):
            nonlocal prior, prior_lower
            next_lower = np.asarray(options["global_step_constraint_lower"], float)
            same_matrix = (options.get("global_step_constraint_matrix") is original_matrix
                           or np.array_equal(options.get("global_step_constraint_matrix"), original_matrix))
            eligible = (_previous_full_KKT_verified(prior) and same_matrix
                        and next_lower.shape == prior_lower.shape == original_lower.shape
                        and np.isfinite(next_lower).all()
                        and np.all(next_lower >= prior_lower)
                        and np.all(next_lower >= original_lower)
                        and np.any(next_lower > prior_lower))
            context = None if not eligible else dict(
                previous_direction_sha256=_digest(prior.step), original_q_sha256=_digest(q),
                strictly_tightened_original_faces=np.flatnonzero(next_lower>prior_lower),
                strictly_tighter_measured_SOC_faces=int(np.count_nonzero(next_lower > prior_lower)))
            begun = time.perf_counter()
            print(json.dumps(dict(stage="optional_SOC_QP_begin", wall_time_ns=time.time_ns(),
                                  prior_full_KKT_direction_available=bool(eligible))), flush=True)
            token = _optional_soc.set(context)
            try:
                corrected = solve_linear_step(*args, **options)
            except OptionalSOCNumericalFailure as exc:
                failure_reports.append(exc.optional_soc_report)
                raise
            finally:
                _optional_soc.reset(token)
                print(json.dumps(dict(stage="optional_SOC_QP_end", wall_time_ns=time.time_ns(),
                                      seconds=time.perf_counter() - begun)), flush=True)
            prior = corrected
            prior_lower = next_lower.copy()
            return corrected

        result = _with_outer_nonlinear_progress(original_SOC,q,linear,selected,domain,linear_options,
                              normal,gradient,diagonal,tightening,solve_optional_correction)
        if failure_reports:
            report = result.report["sequential_measured_shared_source_curvature_tightening"]
            report["optional_SOC_restricted_numerical_failure"] = failure_reports[-1]
            report["mathematical_infeasibility_certified"] = False
        return result

    sequential_optional_SOC._original_SOC_producer=original_SOC

    report = dict(mode="optional_measured_SOC_verified_restricted_failure_route.v6",
                  helper_path=str(Path(__file__).resolve()),
                  helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  original_QP_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  original_SOC_function_sha256=hashlib.sha256(inspect.getsource(original_SOC).encode()).hexdigest(),
                  initial_QP_and_unverified_directions_keep_all_complete_fallbacks=True,
                  original_KKT_equations_and_actual_final_guards_unchanged=True,
                  complete_per_original_row_primal_certificate_required=True,
                  optional_SOC_stops_only_after_original_restricted_route_returns_no_KKT_valid_step=True,
                  optional_SOC_checks_original_selected_KKT_after_public_refinement_before_more_separation=True,
                  optional_SOC_first_restricted_problem_includes_every_strictly_tightened_original_face=True,
                  optional_SOC_can_retreat_if_both_complete_primal_merits_stop_improving=True,
                  optional_outer_SOC_requires_actual_original_nonlinear_face_progress=True,
                  coupled_physical_SOC_uses_original_signed_margins_in_separate_working_m=True,
                  coupled_physical_SOC_requires_physical_feasibility_before_source_merit_comparison=True,
                  failed_physical_SOC_can_use_independent_original_short_fraction_domain=True,
                  optional_outer_SOC_prediction_disagreement_only_holds_lambda=True,
                  optional_outer_SOC_requires_quarter_of_each_resolvable_predicted_merit_reduction=True,
                  optional_outer_SOC_retains_previous_complete_KKT_direction_on_nonprogress=True,
                  optional_outer_SOC_keeps_more_feasible_verified_candidate_on_model_disagreement=True,
                  optional_outer_SOC_requires_both_merits_nonworsening=True,
                  optional_outer_SOC_model_disagreement_prevents_restoration_lambda_decrease=True,
                  optional_outer_SOC_records_original_source_quadratic_prediction_each_measurement=True,
                  optional_outer_SOC_feasibility_reduction_uses_existing_restoration_model_agreement=True,
                  extra_source_evaluations_for_nonlinear_progress=0,
                  exact_original_restricted_failure_report_retained=True,
                  original_restricted_full_diagonal_full_affine_calls_logged=True,
                  original_solver_precision_regularization_and_termination_callbacks_unchanged=True,
                  parallel_factorization_backend=parallel_backend,
                  parallel_factorization_threads=None if parallel_backend is None else max(1, os.cpu_count() or 1),
                  original_auto_backend_with_max_threads_one_on_parallel_numerical_failure=bool(parallel_backend),
                  geometric_violated_face_batches=bool(expand_separation_batches),
                  original_worst_violation_order_and_all_row_KKT_checks_unchanged=True,
                  no_constraint_row_or_solver_iteration_cap_added=True,
                  mathematical_infeasibility_certificate_claimed=False,
                  failed_optional_restricted_status_has_same_precision_original_row_ray_check=True,
                  ray_check_does_not_run_another_solver_or_save_matrices=True,
                  actual_matrix_or_failed_ray_dump_enabled=bool(os.environ.get('RINKSPACE_OPTIONAL_SOC_CAPTURE_DIR')),
                  registered_producer_files_or_manifest_rewritten=False)
    replacement._optional_SOC_numerical_failure_route = True
    replacement._optional_SOC_installation_report = report
    qp._linearly_constrained_shared_step = replacement
    joint._sequential_measured_shared_source_curvature = sequential_optional_SOC
    return report
