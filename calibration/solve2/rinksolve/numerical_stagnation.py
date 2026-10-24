"""Numerical-stagnation diagnostics, distinct from mathematical convergence."""
from __future__ import annotations
import numpy as np


def stagnation_event(event,parameters,parameter_scales,strict_inner_tolerance,*,previous_parameters=None):
    if event.get('initial_state') or not event.get('step_successful'):return None
    if event.get('finite_corner_working_set_transition_required'):return None
    if previous_parameters is None:return None
    stationarity=event.get('normalized_augmented_stationarity_inf')
    if stationarity is None or stationarity<=strict_inner_tolerance:return None
    eps=np.finfo(np.float64).eps;resolution=np.sqrt(eps)
    cost_scale=max(1.,abs(float(event['cost'])))
    local_state_scale=max(1.,float(np.linalg.norm(np.asarray(parameters)/np.asarray(parameter_scales))))
    relative_cost_progress=abs(float(event['cost_change']))/cost_scale
    actual_chart_step=float(np.linalg.norm((np.asarray(parameters)-np.asarray(previous_parameters))/np.asarray(parameter_scales)))
    relative_local_step=actual_chart_step/local_state_scale
    if relative_cost_progress>resolution or relative_local_step>resolution:return None
    return {'kind':'numerical_stagnation_requires_repair','numerically_converged':False,
        'strict_original_units_inner_stationarity_tolerance':float(strict_inner_tolerance),
        'actual_original_units_augmented_stationarity_inf':float(stationarity),
        'relative_objective_progress':relative_cost_progress,'relative_actual_inner_chart_step':relative_local_step,
        'actual_inner_chart_step_norm_from_accepted_x':actual_chart_step,'ceres_reported_ambient_step_norm':float(event['step_norm']),
        'float64_first_order_resolution':float(resolution),'resolution_formula':'sqrt(float64 epsilon)',
        'diagnostic_not_a_convergence_tolerance':True,'iteration_or_time_limit_used':False,
        'reason':'both objective progress and actual inner-chart step are at the float64 first-order resolution scale while strict original-unit stationarity fails',
        'required_next_action':'inspect active source constraints, finite-curve corners, scaling and Jacobians; repair before restarting'}


def dominant_augmented_gradient(engine,state,*,count=20):
    p=engine.params;x=state.parameters;c=engine.problem.constraints
    r,j,cv,cj=engine.al._evaluate(x,True);q=cv/c.scales;rho=engine.al._rho_rows(state.penalties)
    multiplier=np.where(c.equality,state.multipliers+rho*q,np.maximum(0,state.multipliers+rho*q))
    gradient=engine.backend.transpose_product(engine.problem._objective,j,r)+engine.backend.transpose_product(c.layout,cj,multiplier/c.scales)
    projected=engine.al._projected_gradient(x,gradient);cache=engine._compute(x,False)
    ranks=np.argsort(np.abs(projected))[::-1][:count];records=[];budget_index={int(row):k for k,row in enumerate(engine.budget_rows)}
    nb=len(engine.budget_rows)
    for offset in ranks:
        b=int(np.searchsorted(p.parameter_offsets,offset,side='right')-1);component=int(offset-p.parameter_offsets[b])
        item={'native_parameter_offset':int(offset),'native_block_id':p.block_ids[b],'component':component,
            'value':float(x[offset]),'declared_scale':float(p.scales[offset]),'raw_gradient':float(gradient[offset]),
            'declared_units_projected_gradient':float(projected[offset]),'owner':p.block_owners[b]}
        rr=np.flatnonzero(p.local_block==b);stations=rr[p.station_column[rr]==component]
        if len(stations):
            i=int(stations[0]);curve=str(p.curve_ids[i]);s=float(cache['stations'][i]);definition=cache['model'].scene.resolved_definition(p.curve_lookup[curve],cache['model'].state)
            n=len(definition.get('unwrapped_station_height',[]));item.update(observation_id=p.observation_ids[i],native_pts=int(p.native_pts[i]),curve_id=curve,
                raw_uv=p.target_uv[i].tolist(),predicted_uv=cache['forward']['uv'][i].tolist(),station=s,kind=int(p.kinds[i]),row_equality_px=float(cache['roweq'][i]))
            if n>1 and i in budget_index:
                vertex=round(s*(n-1))/(n-1);h=np.sqrt(np.finfo(float).eps)*max(1.,abs(vertex));values=np.array([max(0.,vertex-h),vertex,min(1.,vertex+h)])
                xyz=cache['model'].scene.world_path(curve,values,cache['model'].state)
                out=cache['operator'].project_at(xyz,np.full(3,cache['times'][i]));bi=budget_index[i]
                if out['valid'].all():
                    error=out['uv']-p.target_uv[i];qi=np.sum(error*error,axis=1)/(engine.budgets[bi]**2)-x[p.budget_offset]
                    ri=cache['rows'][i]-out['uv'][:,1];rindex=nb+i
                    z=ri/c.scales[rindex]+state.multipliers[rindex]/rho[rindex]
                    merit=.5*rho[bi]*np.maximum(qi+state.multipliers[bi]/rho[bi],0.)**2+.5*rho[rindex]*z*z
                    left=(merit[1]-merit[0])/(values[1]-values[0]) if values[1]>values[0] else None
                    right=(merit[2]-merit[1])/(values[2]-values[1]) if values[2]>values[1] else None
                    item['finite_polyline_corner']={'vertices':n,'nearest_vertex_station':vertex,'distance_from_internal_vertex':abs(s-vertex),
                        'left_directional_derivative_at_vertex':float(left) if left is not None else None,
                        'right_directional_derivative_at_vertex':float(right) if right is not None else None,
                        'corner_is_local_minimum_of_this_existing_source_PHR':bool(left is not None and right is not None and left<=0 and right>=0),
                        'actual_point_is_exact_vertex':bool(s==vertex),'source_identity_or_equations_changed':False}
        records.append(item)
    return {'schema':'rinksolve.augmented_gradient_diagnostic.v1','strict_original_units_stationarity_inf':float(np.max(np.abs(projected))),
        'physical_state_sha256':cache['model'].state.digest(),'dominant_components':records,'no_KKT_tolerance_relaxed':True}
