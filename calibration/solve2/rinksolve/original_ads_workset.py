"""Temporary finite-corner working charts for the unchanged complete ad problem.

A true internal polyline corner has two feasible one-sided tangents.  A corner
with left derivative <= 0 <= right derivative may be held in this Ceres chart;
its row and every residual stay free/present.  This is numerical active-set
bookkeeping, not a change to a material curve or its physical admissible locus.
"""
from __future__ import annotations
from dataclasses import dataclass,replace
import numpy as np
from .native_backend import BatchLayout


class ReducedCeresChart:
    """Embed a temporary scalar subset into the complete existing x/layout."""
    def __init__(self,full_layout,anchor,fixed_offsets=()):
        self.full=full_layout;self.anchor=np.array(anchor,copy=True)
        fixed=np.zeros(full_layout.parameter_count,dtype=bool);fixed[np.asarray(tuple(fixed_offsets),dtype=np.int64)]=True
        masks=[];block_map={};block_ids=[];sizes=[];indices=[]
        for b,name in enumerate(full_layout.block_ids):
            a,z=full_layout.parameter_offsets[b:b+2];mask=np.flatnonzero(~fixed[a:z]);masks.append(mask)
            if len(mask):
                block_map[b]=len(block_ids);block_ids.append(name);sizes.append(len(mask));indices.extend((a+mask).tolist())
        self.full_indices=np.asarray(indices,dtype=np.int64);self.full_blocks=np.asarray(list(block_map),dtype=np.int64)
        dependencies=[];jindices=[]
        for f,blocks in enumerate(full_layout.factor_blocks):
            kept=[];edge=int(full_layout.factor_edge_offsets[f]);rows=full_layout.residual_sizes[f]
            for k,b in enumerate(blocks):
                if b not in block_map:continue
                kept.append(block_map[b]);start=int(full_layout.jacobian_offsets[edge+k]);width=full_layout.block_sizes[b]
                jindices.extend((start+np.arange(rows)[:,None]*width+masks[b][None,:]).ravel().tolist())
            if not kept:raise ValueError('working chart would remove every dependency of original residual '+full_layout.factor_ids[f])
            dependencies.append(tuple(kept))
        self.layout=BatchLayout(tuple(block_ids),tuple(sizes),full_layout.factor_ids,full_layout.residual_sizes,tuple(dependencies))
        self.jacobian_indices=np.asarray(jindices,dtype=np.int64)
        if self.layout.residual_count!=full_layout.residual_count:raise AssertionError('working chart lost residual rows')
    def expand(self,values):
        result=self.anchor.copy();result[self.full_indices]=values;return result
    def pack(self,values):return np.ascontiguousarray(np.asarray(values)[self.full_indices])
    def jacobian(self,values):return np.ascontiguousarray(np.asarray(values)[self.jacobian_indices])
    def block_values(self,values):return None if values is None else np.ascontiguousarray(np.asarray(values)[self.full_blocks])


def phr_evaluate(engine,state,x,need_jacobian=True):
    r,j,cv,cj=engine.al._evaluate(x,need_jacobian);c=engine.problem.constraints;rho=engine.al._rho_rows(state.penalties)
    shift=cv/c.scales+state.multipliers/rho;active=c.equality|(shift>0)
    residual=np.r_[r,np.sqrt(rho)*np.where(c.equality,shift,np.maximum(0,shift))]
    jacobian=np.r_[j,cj*(np.sqrt(rho)*active/c.scales)[c.jacobian_rows]] if need_jacobian else None
    return residual,jacobian


def internal_vertices(engine,x,*,near=False):
    p=engine.params;result=[];cache=engine._compute(x,False)
    for curve,rows in p.world_groups.items():
        d=cache['model'].scene.resolved_definition(p.curve_lookup[curve],cache['model'].state)
        n=len(d.get('unwrapped_station_height',[]))
        if n<3:continue
        rr=rows[p.station_column[rows]>=0]
        if not len(rr):continue
        offsets=p.parameter_offsets[p.local_block[rr]]+p.station_column[rr]
        stations=x[offsets];k=np.rint(stations*(n-1)).astype(np.int64);vertex=k/(n-1)
        close=np.abs(stations-vertex)<=np.sqrt(np.finfo(float).eps) if near else stations==vertex
        for row,offset,v in zip(rr[close&(k>0)&(k<n-1)],offsets[close&(k>0)&(k<n-1)],vertex[close&(k>0)&(k<n-1)]):
            result.append({'source_index':int(row),'parameter_offset':int(offset),'vertex':float(v),'vertices':n,
                'source_id':p.observation_ids[row],'curve_id':curve,'station':float(x[offset])})
    return result


def corner_derivatives(engine,x,records,*,objective_residual,constraint_multipliers):
    """Evaluate exact left/right branch Jacobians against unchanged base cotangents.

    The evaluation coordinate moves a few ulps solely to select the adjacent
    existing segment.  Base residual and lambda/PHR coefficients stay fixed;
    this does not finite-difference or round the real geometry.
    """
    if not records:return []
    left=np.array(x,copy=True);right=left.copy();offsets=np.array([v['parameter_offset'] for v in records],dtype=np.int64)
    for record in records:
        off=record['parameter_offset'];v=record['vertex'];n=record['vertices'];a=z=v
        while int(np.floor(a*(n-1)))>=round(v*(n-1)):a=np.nextafter(a,-np.inf)
        while int(np.floor(z*(n-1)))<round(v*(n-1)):z=np.nextafter(z,np.inf)
        left[off]=a;right[off]=z
    residual=np.array(objective_residual,copy=True);multipliers=np.array(constraint_multipliers,copy=True);c=engine.problem.constraints
    gradients=[]
    for v in (left,right):
        _,j,_,cj=engine.al._evaluate(v,True)
        g=engine.backend.transpose_product(engine.problem._objective,j,residual)
        if c:g+=engine.backend.transpose_product(c.layout,cj,multipliers/c.scales)
        gradients.append(g[offsets])
    return [dict(v,left_derivative=float(a),right_derivative=float(b),
        stationarity_violation=float(max(a,-b,0.)*engine.problem.parameter_scales[v['parameter_offset']]),
        active_corner_condition=bool(a<=0<=b)) for v,a,b in zip(records,*gradients)]


def corner_phr_records(engine,state,x,*,near=False):
    records=internal_vertices(engine,x,near=near)
    if not records:return []
    r,_,cv,_=engine.al._evaluate(x,False);c=engine.problem.constraints;rho=engine.al._rho_rows(state.penalties)
    aug=np.where(c.equality,state.multipliers+rho*cv/c.scales,np.maximum(0,state.multipliers+rho*cv/c.scales))
    return corner_derivatives(engine,x,records,objective_residual=r,constraint_multipliers=aug)


def snap_active_corners(engine,state):
    x=np.array(state.parameters,copy=True);records=corner_phr_records(engine,state,x,near=True)
    active=[v for v in records if v['active_corner_condition']]
    r,_=phr_evaluate(engine,state,x,False);before=.5*float(r@r)
    candidate=x.copy()
    for record in active:candidate[record['parameter_offset']]=record['vertex']
    rr,_=phr_evaluate(engine,state,candidate,False);after=.5*float(rr@rr)
    # The complete true PHR, including correlated W, is the acceptance guard.
    if after>before:
        active=[];candidate=x;after=before
    exact=corner_phr_records(engine,state,candidate,near=False)
    accepted={v['parameter_offset']:v for v in exact if v['active_corner_condition']}
    active=[accepted[v['parameter_offset']] for v in active if v['parameter_offset'] in accepted]
    return candidate,active,{'PHR_before':before,'PHR_after':after,'identified_near_vertices':len(records),
        'active_internal_corners':len(active),'all_original_residual_rows_preserved':True,'physical_geometry_changed':False}


def attach_corner_stationarity(engine):
    """Correct only exact internal-corner first-order feasible directions."""
    def project(x,gradient,objective_residual,constraint_multipliers):
        result=engine.al._projected_gradient(x,gradient)
        records=internal_vertices(engine,x,near=False)
        try:
            sides=corner_derivatives(engine,x,records,objective_residual=objective_residual,constraint_multipliers=constraint_multipliers)
        finally:
            # AL keeps shared residual/J buffers; callers must still see x.
            if records:engine.al._evaluate(x,True)
        for record in sides:result[record['parameter_offset']]=record['stationarity_violation']
        result[engine.al._constant]=0.
        return result
    engine.al.projected_stationarity_callback=project
    return project


class CornerWorkingSetSolver:
    """Existing Ceres on temporary charts; fixed PHR/metric throughout an inner.

    Chart changes occur only at a verified finite-corner active-set transition.
    This class never updates lambda/rho and never declares convergence itself.
    """
    def __init__(self,engine,event_callback=None):
        self.engine=engine;self.backend=engine.backend;self.event_callback=event_callback
    def solve(self,layout,parameters,kernel,*,options,constant_blocks=None,lower_bounds=None,upper_bounds=None,
              ordering_groups=None,on_iteration=None,parameter_scales=None):
        from .native_backend import SolveResult
        e=self.engine;all_events=[];reports=[];errors=[];contexts=0;current=np.array(parameters,copy=True)
        final_result=None;current_options=options
        def evaluate(x,jacobian,allow_rejection=False):
            r=np.empty(layout.residual_count);j=np.empty(layout.jacobian_value_count) if jacobian else None
            if kernel(x,r,j,True) is False:
                if allow_rejection:return None,None
                raise RuntimeError('corner working set current leaves original forward domain')
            return r,j
        def records_at(x,near=False):
            records=internal_vertices(e,x,near=near)
            if not records:return []
            residual,_=evaluate(x,False);left=x.copy();right=x.copy()
            for v in records:
                off=v['parameter_offset'];vertex=v['vertex'];n=v['vertices'];a=z=vertex
                while int(np.floor(a*(n-1)))>=round(vertex*(n-1)):a=np.nextafter(a,-np.inf)
                while int(np.floor(z*(n-1)))<round(vertex*(n-1)):z=np.nextafter(z,np.inf)
                left[off]=a;right[off]=z
            sides=[];offsets=np.array([v['parameter_offset'] for v in records],dtype=np.int64)
            for point in (left,right):
                _,j=evaluate(point,True);sides.append(self.backend.transpose_product(layout,j,residual)[offsets])
            evaluate(x,False)
            return [dict(v,left_derivative=float(a),right_derivative=float(b),active_corner_condition=bool(a<=0<=b)) for v,a,b in zip(records,*sides)]
        def working_set(x):
            records=records_at(x,near=True);candidate=x.copy();active=[v for v in records if v['active_corner_condition']]
            for v in active:candidate[v['parameter_offset']]=v['vertex']
            r,_=evaluate(x,False);before=.5*float(r@r)
            if not np.array_equal(candidate,x):
                rr,_=evaluate(candidate,False);after=.5*float(rr@rr)
                if after>before:candidate=x.copy();after=before;active=[v for v in active if v['station']==v['vertex']]
                else:active=[v for v in records_at(candidate) if v['active_corner_condition']]
            else:after=before
            return candidate,active,{'PHR_before':before,'PHR_after':after,'active_corner_count':len(active)}
        def notify(event,x):
            if callable(self.event_callback):self.event_callback(event,np.array(x,copy=True))
        current,active,snap=working_set(current)
        notify({'kind':'finite_corner_working_set_initialization','Ceres_accepted_step':False,'native_private_TR_history_restored':False,
                'active_station_offsets':[v['parameter_offset'] for v in active],**snap},current)
        while True:
            contexts+=1;chart=ReducedCeresChart(layout,current,[v['parameter_offset'] for v in active]);transition={}
            active_offsets={v['parameter_offset'] for v in active}
            def chart_kernel(z,r,j,new_point):
                x=chart.expand(z);full_r,full_j=evaluate(x,j is not None,allow_rejection=True)
                if full_r is None:return False
                r[:]=full_r
                if j is not None:j[:]=chart.jacobian(full_j)
            def iteration(event,z):
                x=chart.expand(z);event=dict(event,corner_working_set_generation=contexts,
                    inner_continuation='continuous_Ceres_until_verified_finite_corner_working_set_change')
                candidate=None;next_active=None
                if event['step_successful'] and not event['initial_state']:
                    candidate,next_active,progress=working_set(x);next_offsets={v['parameter_offset'] for v in next_active}
                    if next_offsets!=active_offsets or not np.array_equal(candidate,x):
                        event['finite_corner_working_set_transition_required']=True
                        transition.update(parameters=candidate,active=next_active,progress=progress,
                            added=sorted(next_offsets-active_offsets),released=sorted(active_offsets-next_offsets))
                all_events.append(event)
                external=on_iteration(event,x) if callable(on_iteration) else 0
                if external:
                    transition.clear();return external
                return 1 if transition else 0
            result=self.backend.solve(chart.layout,chart.pack(current),chart_kernel,options=current_options,
                constant_blocks=chart.block_values(constant_blocks),lower_bounds=chart.pack(lower_bounds) if lower_bounds is not None else None,
                upper_bounds=chart.pack(upper_bounds) if upper_bounds is not None else None,ordering_groups=chart.block_values(ordering_groups),
                on_iteration=iteration,parameter_scales=chart.pack(parameter_scales) if parameter_scales is not None else None)
            reports.append(result.report);errors.extend(result.callback_errors);final_result=result;current=chart.expand(result.parameters)
            if not transition or not result.usable or result.callback_errors:break
            candidate=transition['parameters'];rr,_=evaluate(current,False);rn,_=evaluate(candidate,False)
            if .5*float(rn@rn)>.5*float(rr@rr):raise RuntimeError('verified corner transition increased unchanged true PHR')
            current=candidate;active=transition['active']
            public_radius=float(next(v['trust_region_radius'] for v in reversed(all_events) if v['step_successful'] and not v['initial_state']))
            if not np.isfinite(public_radius) or not 0<public_radius<=options.max_trust_region_radius:raise ValueError('invalid accepted public trust-region radius')
            current_options=replace(options,initial_trust_region_radius=public_radius)
            notify({'kind':'verified_finite_corner_working_set_transition','completed_native_contexts':contexts,
                'added_station_offsets':transition['added'],'released_station_offsets':transition['released'],
                'original_row_parameters_free':True,'Ceres_accepted_step':False,'lambda_rho_or_metric_changed':False,
                'native_private_TR_history_restored':False,'next_native_initial_public_TR_radius':public_radius,**transition['progress']},current)
        summary=dict(final_result.summary,finite_corner_native_contexts=contexts,
            finite_corner_solver='same complete fixed-PHR/metric; existing Ceres; chart changes only on verified corner active-set changes')
        return SolveResult(np.array(current,copy=True),summary,tuple(all_events),tuple(errors),'\n\n'.join(reports))


def attach_corner_working_set(engine,event_callback=None):
    attach_corner_stationarity(engine)
    driver=CornerWorkingSetSolver(engine,event_callback)
    engine.al.inner_solve_callback=driver.solve
    engine.corner_working_set_solver=driver
    return driver
