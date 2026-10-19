"""Only measured outward source-face curvature tightens a numerical QP step."""
import numpy as np


def bracketed_physical_endpoint_event(contexts, query_at_fraction, maximum_fraction=1.):
    """Locate an observed physical endpoint transition, not a polyline knot.

    Contexts contain only nominal interior witnesses on the first/last segment
    of an authenticated open primitive. The predicate uses the actual finite
    query category, so the returned representable fraction is already on the
    endpoint side. No station inequality is added to the source domain. The
    observed bracket is not a proof of the earliest event on an arbitrary
    nonmonotone nonlinear trajectory; true trial guards remain authoritative.
    """
    maximum_fraction=float(maximum_fraction)
    if not np.isfinite(maximum_fraction)or not 0.<maximum_fraction<=1.:
        raise ValueError('Physical endpoint event needs a finite fraction in (0,1]')
    if not contexts:return None
    calls=0
    def endpoints(fraction):
        nonlocal calls
        calls+=1;result=[]
        queries=query_at_fraction(fraction)
        if len(queries)!=len(contexts):raise ValueError('Complete endpoint witness inventory changed')
        for context,query in zip(contexts,queries):
            segment=np.asarray(query['nearest_finite_segment_index'],int)
            category=np.asarray(query['nominal_finite_segment_categories'],int)
            first=np.asarray(context['first'],bool)
            last=np.asarray(context['last'],bool)
            if segment.shape!=first.shape or category.shape!=first.shape or last.shape!=first.shape:
                raise ValueError('Original physical endpoint source identity changed')
            result.append(np.where(first&(segment==0)&(category==1),1,
                np.where(last&(segment==context['last_segment'])&(category==2),2,0)))
        return result
    initial=endpoints(0.)
    if any(np.any(value)for value in initial):
        raise ValueError('Physical endpoint event must start on an interior witness')
    at_upper=endpoints(maximum_fraction)
    if not any(np.any(value)for value in at_upper):return None
    lo=0.;hi=maximum_fraction
    # Stop at adjacent representable fractions, rather than an arbitrary
    # iteration count or distance from the physical boundary.
    while True:
        midpoint=lo+(hi-lo)*.5
        if midpoint==lo or midpoint==hi:break
        if any(np.any(value)for value in endpoints(midpoint)):hi=midpoint
        else:lo=midpoint
    at_upper=endpoints(hi);at_lower=endpoints(lo)
    if any(np.any(value)for value in at_lower)or not any(np.any(value)for value in at_upper):
        raise ValueError('Physical endpoint transition could not be bracketed')
    events=[]
    for context,changed in zip(contexts,at_upper):
        for point in np.flatnonzero(changed):
            events.append(dict(group=context['group'],id=context['id'],actual_pts=context['actual_pts'],
                original_source_vertex=int(context['vertices'][point]),
                physical_endpoint='first'if changed[point]==1 else'last'))
    return dict(fraction=hi,report=dict(schema='rinkspace.actual-open-physical-endpoint-branch-event.v1',
        interior_fraction=lo,endpoint_fraction=hi,actual_finite_endpoint_category_verified=True,
        query_evaluations=calls,events=events,source_support_and_curve_unchanged=True,
        physical_endpoint_station_is_not_a_permanent_guard=True,
        global_earliest_event_on_nonlinear_trajectory_claimed=False,
        next_complete_source_linearization_must_be_fresh=True,
        clamped_witness_may_release_on_the_next_direction=True))


def physical_endpoint_branch_event(truth,q,step,maximum_fraction=1.):
    """Numerical event for original full strokes on real open primitives."""
    from .finite_endpoint_restoration_step_constraints import authenticated_open_curve
    from .human_stroke_truth import finite_stroke_queries
    from .native_camera_local import native_camera_local
    q=np.asarray(q,float);step=np.asarray(step,float)
    if q.shape!=step.shape or q.shape!=(truth.parameter_count,)or not np.isfinite(q).all()or not np.isfinite(step).all():
        raise ValueError('Complete finite q and actual full step required for an endpoint event')
    records=truth.full_stroke_records
    if records is None:raise ValueError('Original full stroke supports required for an endpoint event')
    nominal=truth.queries(q,full_stroke=True);contexts=[]
    for group,(record,result)in enumerate(zip(records,nominal)):
        world=record['factor']._world(q,record['row']['curve'])
        if not authenticated_open_curve(record['row']['curve'],world):continue
        query=result['query'];segment=np.asarray(query['nearest_finite_segment_index'],int)
        category=np.asarray(query['nominal_finite_segment_categories'],int)
        first=(segment==0)&(category==0);last=(segment==len(world)-2)&(category==0)
        vertices=np.flatnonzero(first|last)
        if not len(vertices):continue
        contexts.append(dict(group=group,id=result['id'],actual_pts=int(result['actual_pts']),
            vertices=vertices,first=first[vertices],last=last[vertices],last_segment=len(world)-2,
            record=record))
    def queries(fraction):
        value=q+fraction*step
        if not np.isfinite(value).all()or np.any(value[truth.height_columns]<=0.):
            raise ValueError('Endpoint event retains the original positive pivot domain')
        truth.state.geometry.validate_parameters(value[truth.world_indices])
        core=value[:truth.state.structured_programme_coordinate_chart.core_count]
        cameras={};result=[]
        for context in contexts:
            pts=context['actual_pts'];record=context['record'];indices=context['vertices']
            if pts not in cameras:
                block=truth.blocks[pts]
                cameras[pts]=block.bundle.camera(native_camera_local(block,core),pts)
            world=record['factor']._world(value,record['row']['curve'])
            if len(world)-2!=context['last_segment']or not authenticated_open_curve(record['row']['curve'],world):
                raise ValueError('Original open physical primitive identity changed')
            result.append(finite_stroke_queries(cameras[pts],world,record['uv'][indices],record['normals'][indices],
                record['bounds'][indices],tangent_bounds=record['tangent_bounds'][indices]))
        return result
    return bracketed_physical_endpoint_event(contexts,queries,maximum_fraction)

def constraints_on_declared_free_chart(factor,q,selected,declared_fixed=(),**options):
    """Pull back original support faces onto the caller's exact fixed tangent.

    Original producers retain their complete dependency coverage checks and
    full face order. Only explicitly fixed omitted columns are restored for
    producer construction, then multiplied by their exact zero displacement.
    Unspecified free dependencies must still be in the selected chart.
    """
    selected=np.asarray(selected,int).reshape(-1);fixed=np.asarray(declared_fixed,int).reshape(-1)
    active=np.asarray(factor.active,int).reshape(-1)
    if(len(np.unique(selected))!=len(selected)or len(np.unique(fixed))!=len(fixed)
        or np.any(selected<0)or np.any(selected>=len(q))or np.any(fixed<0)or np.any(fixed>=len(q))):
        raise ValueError('Exact distinct selected and declared fixed source coordinates required')
    omitted=np.unique(active[~np.isin(active,selected)])
    if not np.isin(omitted,fixed).all():
        raise ValueError('A nonfixed original truth dependency is missing from the shared chart')
    if not len(omitted):return factor.constraints(q,selected,**options)
    augmented=np.r_[selected,omitted]
    result=factor.constraints(q,augmented,**options)
    original=np.asarray(result['matrix'],float)
    if original.shape!=(len(result['lower']),len(augmented))or not np.isfinite(original).all():
        raise ValueError('Original truth faces do not match their complete selected chart')
    result=dict(result,matrix=original[:,:len(selected)])
    result['report']=dict(result.get('report',{}),declared_fixed_tangent_pullback=dict(
        effective_selected_parameter_indices=selected.tolist(),
        producer_selected_parameter_indices=augmented.tolist(),
        omitted_explicit_fixed_parameter_indices=omitted.tolist(),
        original_fixed_derivative_nonzero_count=int(np.count_nonzero(original[:,len(selected):])),
        original_ordered_faces_and_targets_preserved=True,
        exact_fixed_displacement=0.,unfixed_dependency_coverage_still_required=True))
    return result

def carrier_step_response(carrier,face_mask):
    carrier=np.asarray(carrier,float);mask=np.asarray(face_mask,bool)
    if carrier.ndim!=1 or mask.shape!=(2*len(carrier),)or not np.isfinite(carrier).all():
        raise ValueError('Complete finite source carrier and exact retained step-face mask required')
    return dict(nominal_carrier=carrier.copy(),retained_face_mask=mask.copy())

def actual_carrier_step_response(metadata,actual):
    """Pull actual unchanged query carriers onto each numerical step face."""
    base=np.asarray(metadata['nominal_carrier'],float);actual=np.asarray(actual,float)
    if base.ndim!=1 or actual.shape!=base.shape or not np.isfinite(base).all()or not np.isfinite(actual).all():
        raise ValueError('Actual finite source carrier identity changed')
    change=actual-base
    if 'row_carrier_projection'in metadata:
        from scipy.sparse import csr_matrix
        projection=metadata['row_carrier_projection']
        if not isinstance(projection,csr_matrix)or projection.shape[1]!=len(base)or not np.isfinite(projection.data).all():
            raise ValueError('Exact sparse numerical face/source-carrier map required')
        result=np.asarray(projection@change).ravel()
    else:
        mask=np.asarray(metadata['retained_face_mask'],bool)
        if mask.shape!=(2*len(base),):raise ValueError('Exact original retained carrier face mask required')
        result=np.r_[change,-change][mask]
    if not np.isfinite(result).all():raise ValueError('Finite actual source-face response required')
    return result


def dynamic_source_group_merits(matrix,lower,shared_step,responses,actual_group_violations):
    """Compare actual and predicted excess in the same original group units.

    Local face counts and witness changes are not nonlinear truth merits.
    Group targets are immutable snapshots of this parent's existing priority
    target (or its original guard cap), never newly enlarged tolerances.
    """
    matrix=np.asarray(matrix,float);lower=np.asarray(lower,float);step=np.asarray(shared_step,float)
    if set(actual_group_violations)!=set(responses):raise ValueError('Every original dynamic source group family required')
    slack=matrix@step-lower;families={};actual_all=[];predicted_all=[]
    for family,metadata in responses.items():
        lo,hi=metadata['row_start'],metadata['row_stop']
        target=np.asarray(metadata['dynamic_group_limits'],float)
        actual=np.asarray(actual_group_violations[family],float)
        groups=np.asarray(metadata['dynamic_row_group_index'],int)
        if(target.ndim!=1 or actual.shape!=target.shape or groups.shape!=(hi-lo,)
            or np.any(groups < -1)or np.any(groups>=len(target))
            or not np.isfinite(target).all()or not np.isfinite(actual).all()
            or np.any(target<0.)or np.any(actual<0.)):
            raise ValueError('Exact original nonnegative group violations, targets and row ownership required')
        excess=np.maximum(actual-target,0.);prediction=np.zeros(len(target))
        owned=groups>=0
        np.maximum.at(prediction,groups[owned],np.maximum(-slack[lo:hi][owned],0.))
        families[family]=dict(groups=len(target),actual_group_violations=actual.tolist(),
            original_group_targets=target.tolist(),actual_group_excess_px=excess.tolist(),
            predicted_group_excess_px=prediction.tolist(),maximum_violation_px=float(excess.max(initial=0.)),
            total_violation_px=float(excess.sum()),violated_groups=int(np.count_nonzero(excess)),
            locally_unowned_original_rows=int(np.count_nonzero(~owned)))
        actual_all.extend(excess);predicted_all.extend(prediction)
    actual=np.asarray(actual_all);predicted=np.asarray(predicted_all)
    return dict(maximum_violation_px=float(actual.max(initial=0.)),total_violation_px=float(actual.sum()),
        violated_rows=int(np.count_nonzero(actual)),predicted_maximum_violation_px=float(predicted.max(initial=0.)),
        predicted_total_violation_px=float(predicted.sum()),complete_original_source_groups=len(actual),families=families,
        merit_basis='original_dynamic_group_excess_above_same_parent_priority_target',
        predicted_and_actual_merits_share_original_group_units=True,
        local_station_face_combinations_not_claimed_as_nonlinear_guard=True,
        actual_physical_trial_guard_still_required=True,source_support_and_targets_unchanged=True)

def measured_outward_step_tightening(matrix,lower,shared_step,responses,actual_carriers,*,actual_group_violations=None):
    matrix=np.asarray(matrix,float);lower=np.asarray(lower,float);shared_step=np.asarray(shared_step,float)
    if actual_group_violations is not None:
        dynamic_source_group_merits(matrix,lower,shared_step,responses,actual_group_violations)
    tightened=lower.copy();details=[]
    for family,metadata in responses.items():
        begin,end=metadata['row_start'],metadata['row_stop']
        actual_response=actual_carrier_step_response(metadata,actual_carriers[family])
        if actual_response.shape!=(end-begin,):raise ValueError('Source step-face row mapping changed')
        linear_response=matrix[begin:end]@shared_step
        actual_slack=actual_response-lower[begin:end]
        eligible=actual_slack < -1e-8
        if actual_group_violations is not None:
            target=np.asarray(metadata['dynamic_group_limits'],float)
            actual=np.asarray(actual_group_violations[family],float)
            groups=np.asarray(metadata['dynamic_row_group_index'],int)
            if actual.shape!=target.shape or groups.shape!=actual_slack.shape:
                raise ValueError('Complete original dynamic source group gate required')
            tolerance=float(metadata['dynamic_guard_comparison_tolerance_px'])
            if not np.isfinite(tolerance)or tolerance<0.:raise ValueError('Original guard comparison precision required')
            owned=groups>=0;gate=np.zeros(len(groups),bool)
            gate[owned]=actual[groups[owned]]>target[groups[owned]]+tolerance
            # The independent actual guard decides whether correction is
            # needed. A frozen witness's excess alone never creates a cut.
            eligible &= gate
        for index in np.flatnonzero(eligible):
            curvature=float(linear_response[index]-actual_response[index])
            # This target is stricter only inside the numerical subproblem.
            # The original source support and nonlinear guards are untouched.
            increment=max(0.,curvature)+1e-4
            tightened[begin+index]+=increment
            detail=dict(family=family,source_step_face=int(index),
                measured_outward_curvature_px=curvature,strict_inside_step_margin_px=1e-4,step_lower_increase=increment)
            if actual_group_violations is None:
                detail['actual_original_violation_px']=float(-actual_slack[index])
            else:
                owner=int(groups[index])
                detail.update(local_chart_face_violation_px=float(-actual_slack[index]),
                    original_source_group_index=owner,
                    actual_original_group_violation_px=float(actual[owner]),
                    group_target_px=float(target[owner]),
                    actual_original_group_excess_px=float(max(actual[owner]-target[owner],0.)))
            details.append(detail)
    if not details:return None
    return dict(lower=tightened,report=dict(schema='rinkspace.measured-outward-source-face-QP-curvature-compensation.v1',
        original_source_H_g_profiles_weights_bounds_and_guards_unchanged=True,
        only_original_step_lower_tightened=True,extra_source_rows_or_priors=0,compensated_faces=details,
        original_dynamic_group_guard_gates_local_chart_cuts=actual_group_violations is not None))
