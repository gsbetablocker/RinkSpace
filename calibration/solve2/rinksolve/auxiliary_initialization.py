"""Detached numerical initialization on existing S01 equations.

This module never starts Ceres, publishes a runtime state, edits source data or
changes admission. Its caller must supply one frozen complete state at an
explicit initialization boundary. The finite-branch search is a multi-basin
candidate search; it does not certify a global closest-point optimum.
"""
from __future__ import annotations
import hashlib
import time
from dataclasses import asdict, replace
from scipy.optimize import minimize_scalar
import numpy as np


def kind0_formal_rs_candidate(engine, state, *, optimize_budget=True,
                             root_tolerance_px=1e-7, observation_rows=None, on_progress=None):
    """Return an audited candidate, retaining original values on any guard failure.

    Only free kind0 observation-local station/row components can change. All
    shared material positions, fixed landmarks, profile stations and Query
    alpha remain bit-identical. The optional single-b update exactly minimizes
    the current normalized PHR function at fixed multipliers and penalties.
    The returned candidate is an initialization, never a Ceres accepted step.
    """
    e=engine;old=state;p=e.params;c=e.problem.constraints
    if e.phase!='feasibility':raise ValueError('shared-b minimization requires the explicit min-b feasibility phase')
    if old.problem_digest!=e.problem.digest or old.identity!=e.problem.identity:raise ValueError('candidate must bind the exact current problem and complete state')
    if not np.isfinite(root_tolerance_px) or root_tolerance_px<=0:raise ValueError('explicit positive numerical root tolerance required')
    if type(optimize_budget)is not bool:raise TypeError('optimize_budget must be bool')
    if e.problem.constant_blocks[p.budget_block]:raise ValueError('single shared budget must remain free in this min-b stage')
    nb=len(e.budget_rows)
    if np.any(c.equality[:nb]) or not np.all(c.equality[nb:]):raise ValueError('unexpected S01 constraint order/senses')
    if not all(p.budget_block in blocks for blocks in c.layout.factor_blocks[:nb]) or any(p.budget_block in blocks for blocks in c.layout.factor_blocks[nb:]):raise ValueError('all and only actual budget-dependent constraints must participate in waterfill')
    initial=e._compute(old.parameters,False);m=initial['model'];op=initial['operator'];board=m.scene.board(m.state)
    H=p.rig.image_height-1;tau=op.camera_definition['signed_readout_seconds'];tol=float(root_tolerance_px)
    bound=int(np.ceil(np.log(H/tol)/(-np.log(.75))))+4
    start=time.perf_counter();working=old.parameters.copy();sbest=initial['stations'].copy();rbest=initial['rows'].copy()
    coarse_station=sbest.copy();coarse_row=rbest.copy();score=np.full(p.count,np.inf)
    eligible=(p.kinds==0)&(p.station_column>=0)&(e.problem.constant_blocks[p.local_block]==0);details=[]
    if observation_rows is not None:
        selected=np.asarray(observation_rows)
        if selected.ndim!=1 or selected.dtype.kind not in "iu" or np.any(selected<0) or np.any(selected>=p.count):raise ValueError("selected current observation rows must be valid integer indices")
        mask=np.zeros(p.count,dtype=bool);mask[selected]=True;eligible&=mask
    def rs(curve,s,frame,*,derivative=False):
     s=np.asarray(s,dtype=np.float64);frame=np.asarray(frame,dtype=np.float64);n=len(s);capacity=1<<(max(1,n)-1).bit_length();seed=e._seed_project(m,op,curve,s,frame,board,need_station=derivative);camera=e._seed_gpu_pool[capacity][0]
     def project(rows,full=False):
      clock=np.empty(capacity);clock[:n]=frame+tau*(rows/H-.5);clock[n:]=clock[n-1];camera.set_rows(time=clock);v=camera.evaluate() if full else camera.evaluate_uv();return {k:a[:n] for k,a in v.items() if isinstance(a,np.ndarray) and a.shape[:1]==(capacity,)}
     left=np.zeros(n);right=np.full(n,float(H));a=project(left);b=project(right);fa=-a['uv'][:,1];fb=H-b['uv'][:,1];bracket=a['valid']&b['valid']&(fa<=0)&(fb>=0);rows=np.full(n,H/2);done=bracket&((np.abs(fa)<=tol)|(np.abs(fb)<=tol));rows[bracket&(np.abs(fa)<=tol)]=0.;rows[bracket&(np.abs(fb)<=tol)]=H;fl=fa.copy()
     for iteration in range(bound):
      active=bracket&~done
      if not np.any(active):break
      v=project(rows);f=rows-v['uv'][:,1];done|=active&v['valid']&(np.abs(f)<=tol);active&=~done
      same=np.signbit(f)==np.signbit(fl);take=active&same;left[take]=rows[take];fl[take]=f[take];take=active&~same;right[take]=rows[take]
      candidate=v['uv'][:,1];width=right-left;safe=np.isfinite(candidate)&(candidate>left+.25*width)&(candidate<right-.25*width);candidate=np.where(safe,candidate,(left+right)/2);rows[active]=candidate[active]
     v=project(rows,derivative);valid=bracket&done&v['valid']&(np.abs(rows-v['uv'][:,1])<=tol);v.update(valid=valid,row=rows,fa=fa,fb=fb)
     if derivative:
      direct=np.einsum('nij,nj->ni',v['jacobian_world'],seed['world_station_jacobian']);gamma=tau/H;den=1-v['jacobian_time'][:,1]*gamma;vr=direct[:,1]/den;v['station_derivative']=np.column_stack((direct[:,0]+v['jacobian_time'][:,0]*gamma*vr,vr))
     return v
    for curve,allrows in p.world_groups.items():
     rr=allrows[eligible[allrows]]
     if not len(rr):continue
     definition=m.scene.resolved_definition(p.curve_lookup[curve],m.state);minimum=513 if definition['kind'] in ('ice_circle','board_boundary') else 33;grid=np.linspace(0,1,minimum);vertices=np.asarray(definition.get('unwrapped_station_height',[]))
     if len(vertices)>1:grid=np.unique(np.r_[grid,np.linspace(0,1,len(vertices))])
     frames,inverse=np.unique(p.frame_times[rr],return_inverse=True);G=len(grid);F=len(frames);value=rs(curve,np.tile(grid,F),np.repeat(frames,G));uv=value['uv'].reshape(F,G,2);valid=value['valid'].reshape(F,G);rows=value['row'].reshape(F,G)
     for begin in range(0,len(rr),max(1,250000//G)):
      end=min(len(rr),begin+max(1,250000//G));idx=inverse[begin:end];target=p.target_uv[rr[begin:end]];errors=np.sum((uv[idx]-target[:,None,:])**2,axis=2);errors[~valid[idx]]=np.inf;j=np.argmin(errors,axis=1);locations=rr[begin:end];best=errors[np.arange(len(locations)),j];score[locations]=best;sbest[locations]=grid[j];rbest[locations]=rows[idx,j]
     coarse_station[rr]=sbest[rr];coarse_row[rr]=rbest[rr]
     # Include genuine sensor-boundary intersections between coarse stations.
     edge_candidates=[]
     for sensor,key in ((0.,'fa'),(float(H),'fb')):
      values=value[key].reshape(F,G);fi,gi=np.nonzero(np.isfinite(values[:,:-1])&np.isfinite(values[:,1:])&(np.signbit(values[:,:-1])!=np.signbit(values[:,1:])))
      if not len(fi):continue
      lo=grid[gi].copy();hi=grid[gi+1].copy();flo=values[fi,gi].copy();ss=(lo+hi)/2;tt=frames[fi]+tau*(sensor/H-.5);done=np.zeros(len(ss),dtype=bool)
      for iteration in range(bound):
       out=e._seed_project(m,op,curve,ss,tt,board,need_station=True);f=sensor-out['uv'][:,1];done|=out['valid']&(np.abs(f)<=tol)
       if done.all():break
       same=np.signbit(f)==np.signbit(flo);take=~done&same;lo[take]=ss[take];flo[take]=f[take];take=~done&~same;hi[take]=ss[take];der=-np.einsum('nj,nj->n',out['jacobian_world'][:,1],out['world_station_jacobian']);proposal=ss-np.divide(f,der,out=np.zeros_like(f),where=np.abs(der)>np.finfo(float).eps);width=hi-lo;safe=(proposal>lo+.25*width)&(proposal<hi-.25*width)&np.isfinite(proposal);ss[~done]=np.where(safe,proposal,(lo+hi)/2)[~done]
      out=e._seed_project(m,op,curve,ss,tt,board);good=done&out['valid']&(np.abs(sensor-out['uv'][:,1])<=tol)
      for frame in np.unique(fi[good]):
       loc=rr[inverse==frame];edges=np.flatnonzero(good&(fi==frame));err=np.sum((p.target_uv[loc,None,:]-out['uv'][edges][None,:,:])**2,axis=2);which=np.argmin(err,axis=1);best=err[np.arange(len(loc)),which];use=best<score[loc];chosen=loc[use];score[chosen]=best[use];sbest[chosen]=ss[edges[which[use]]];rbest[chosen]=sensor
     # Refine the globally seeded same-branch formal-RS distances with true implicit derivatives.
     refinements=0
     for iteration in range(np.finfo(float).nmant+1):
      out=rs(curve,sbest[rr],p.frame_times[rr],derivative=True);J=out['station_derivative'];error=out['uv']-p.target_uv[rr];step=np.divide(np.sum(J*error,axis=1),np.sum(J*J,axis=1),out=np.zeros(len(rr)),where=np.sum(J*J,axis=1)>0);pending=np.flatnonzero(out['valid']&np.isfinite(step)&(np.abs(step)>16*np.finfo(float).eps));any_improve=False
      for backtrack in range(np.finfo(float).nmant+1):
       if not len(pending):break
       proposed=np.clip(sbest[rr[pending]]-step[pending],0,1);trial=rs(curve,proposed,p.frame_times[rr[pending]]);newscore=np.sum((trial['uv']-p.target_uv[rr[pending]])**2,axis=1);accept=trial['valid']&(score[rr[pending]]-newscore>64*np.finfo(float).eps*np.maximum(1,score[rr[pending]]));take=pending[accept];sbest[rr[take]]=proposed[accept];rbest[rr[take]]=trial['row'][accept];score[rr[take]]=newscore[accept];refinements+=len(take);any_improve|=bool(len(take));remaining=(~accept)&(proposed!=sbest[rr[pending]]);pending=pending[remaining];step[pending]*=.5
      if not any_improve:break
     if on_progress is not None:on_progress({'curve_id':curve,'completed_kind0_rows':int(np.isfinite(score).sum()),'total_kind0_rows':int(eligible.sum())})
     details.append({'curve_id':curve,'kind0_rows':len(rr),'native_frame_count':F,'global_finite_branch_station_grid':G,'formal_RS_refinement_accepts':refinements,'finite_candidates':int(np.isfinite(score[rr]).sum())})
    # Independently confirm final candidates through the public formal true-RS operator.
    rr=np.flatnonzero(eligible&np.isfinite(score));stations=initial['stations'].copy();stations[rr]=sbest[rr];xyz,_=e._world(m,stations,need_station=False);check=op.project_rs(xyz[rr],p.native_pts[rr],root_iterations=64,root_tolerance_px=tol);fallback=~check['numerical_root_valid'];fallback_count=int(fallback.sum())
    if fallback_count:
     failed=rr[fallback];sbest[failed]=coarse_station[failed];rbest[failed]=coarse_row[failed];stations[failed]=sbest[failed];xyz,_=e._world(m,stations,need_station=False);check=op.project_rs(xyz[rr],p.native_pts[rr],root_iterations=64,root_tolerance_px=tol)
    confirmed=check['numerical_root_valid'];candidate_error=np.sum((check['uv']-p.target_uv[rr])**2,axis=1);nb=len(e.budget_rows);budget_position=np.full(p.count,-1,dtype=int);budget_position[e.budget_rows]=np.arange(nb);c=e.problem.constraints;rho=e.al._rho_rows(old.penalties);qold=e._budget_quantities(initial)[0];b=float(old.parameters[p.budget_offset]);accepted=[];rejected=[]
    for local,i in enumerate(rr):
     bi=budget_position[i];ri=nb+i
     if bi<0 or not confirmed[local]:rejected.append({'observation_id':p.observation_ids[i],'reason':'not_independently_confirmed_formal_sensor_RS_root','formal_status_code':int(check['status_code'][local]),'curve_id':str(p.curve_ids[i])});continue
     qnew=candidate_error[local]/e.budgets[bi]**2;h0=initial['roweq'][i]/c.scales[ri];h1=check['root_equality_px'][local]/c.scales[ri];oldcost=.5*rho[bi]*max(0,(qold[bi]-b)/c.scales[bi]+old.multipliers[bi]/rho[bi])**2+.5*rho[ri]*(h0+old.multipliers[ri]/rho[ri])**2;newcost=.5*rho[bi]*max(0,(qnew-b)/c.scales[bi]+old.multipliers[bi]/rho[bi])**2+.5*rho[ri]*(h1+old.multipliers[ri]/rho[ri])**2
     if qnew<=qold[bi]+64*np.finfo(float).eps*max(1,qold[bi]) and newcost<=oldcost+64*np.finfo(float).eps*max(1,oldcost):
      off=p.parameter_offsets[p.local_block[i]];working[off+p.station_column[i]]=sbest[i];working[off+p.row_column[i]]=check['row'][local];accepted.append(int(i))
     else:rejected.append({'observation_id':p.observation_ids[i],'reason':'engineering_q_or_actual_PHR_contribution_would_increase'})
    def merit(x):
     residual,_,raw,_=e.al._evaluate(x,False);shift=raw/c.scales+old.multipliers/rho;return .5*float(residual@residual)+.5*float(np.sum(rho*np.where(c.equality,shift,np.maximum(shift,0))**2))
    oldmerit=merit(old.parameters);auxmerit=merit(working);auxcache=e._compute(working,False);qaux=e._budget_quantities(auxcache)[0]
    # Exact one-dimensional shared-b PHR minimization, using actual normalization.
    rhob=rho[:nb]/(c.scales[:nb]**2);a=qaux+old.multipliers[:nb]*c.scales[:nb]/rho[:nb]
    bexact=exact_shared_budget_phr(qaux,old.multipliers[:nb],rho[:nb],c.scales[:nb])
    combined=working.copy();combined[p.budget_offset]=bexact if optimize_budget else b;combinedmerit=merit(combined);diagnostics,_=e.al._diagnostics(combined,old.multipliers,old.penalties);assert auxmerit<=oldmerit+1e-6 and combinedmerit<=auxmerit+1e-6;assert np.array_equal(old.parameters[:p.model_parameter_count],combined[:p.model_parameter_count]);changed=np.flatnonzero(old.parameters!=working);allowed=[]
    for i in accepted:
     off=p.parameter_offsets[p.local_block[i]];allowed.extend((int(off+p.station_column[i]),int(off+p.row_column[i])))
    assert set(changed)<=set(allowed)

    nonlocal_mask=np.ones(len(old.parameters),dtype=bool);nonlocal_mask[p.budget_offset]=False;nonlocal_mask[np.asarray(allowed,dtype=np.int64)]=False
    assert np.array_equal(old.parameters[nonlocal_mask],combined[nonlocal_mask])
    finalcache=e._compute(combined,False)
    assert np.array_equal(initial['error'][e.profile_rows],finalcache['error'][e.profile_rows])
    assert np.array_equal(initial['query_error'][e.profile_rows],finalcache['query_error'][e.profile_rows])
    report={'schema':'rinksolve.kind0_true_RS_auxiliary_and_waterfill.v1',
        'source_native_x_sha256':hashlib.sha256(old.parameters.tobytes()).hexdigest(),
        'candidate_native_x_sha256':hashlib.sha256(combined.tobytes()).hexdigest(),
        'physical_state_sha256':m.state.digest(),'seconds':time.perf_counter()-start,
        'kind0_eligible':int(eligible.sum()),'kind0_accepted':len(accepted),'kind0_rejected':len(rejected),
        'public_RS_rejected_candidates_tried_coarse_fallback':fallback_count,
        'accepted_formal_rows_without_unique_elimination_certificate':int(np.sum(np.isin(rr,accepted)&~check['valid'])),
        'unique_elimination_certificate_is_required_by_this_rowaux_problem':False,
        'profile_Query_fixed_shared_physics_bit_identical':True,
        'lambda_rho_unchanged':True,'actual_Ceres_accepted_step':False,
        'source_pixels_entity_clock_and_admission_unchanged':True,
        'budget_before':b,'budget_after':float(combined[p.budget_offset]),
        'exact_b_candidate':bexact,'waterfill_applied_to_candidate':optimize_budget,
        'waterfill_derivative_at_exact_candidate':float(1-np.sum(rhob*np.maximum(a-bexact,0))),
        'PHR_before':oldmerit,'PHR_after_kind0_aux_only':auxmerit,'PHR_after_combined':combinedmerit,
        'max_q_before':float(qold.max()),'max_q_after_aux':float(qaux.max()),
        'row_equality_max_before_px':float(np.abs(initial['roweq']).max()),
        'row_equality_max_after_px':float(np.abs(finalcache['roweq']).max()),
        'row_equality_squared_sum_before':float(initial['roweq']@initial['roweq']),
        'row_equality_squared_sum_after':float(finalcache['roweq']@finalcache['roweq']),
        'budget_normalization_scale_min':float(c.scales[:nb].min()),
        'budget_normalization_scale_max':float(c.scales[:nb].max()),
        'curve_searches':details,'rejected_sources':rejected}
    return combined,diagnostics,report


def shared_feature_position_candidate(engine, state, *, optimize_budget=True,
                                      root_tolerance_px=1e-7, on_progress=None):
    """Jointly initialize each existing unknown FEATURE position across its frames.

    Each feature retains one material station and every exposure has only its
    formal sensor row. This changes the feature's unknown physical coordinate;
    camera, scene, curves and all other physical parameters stay unchanged.
    The engineering q sum selects a candidate, without adding any statistical
    residual. Actual normalized PHR and every positive E violation guard it.
    """
    e=engine;old=state
    if e.phase!='feasibility':raise ValueError('shared feature initializer requires the explicit min-b stage')
    if old.problem_digest!=e.problem.digest or old.identity!=e.problem.identity:raise ValueError('candidate must bind the exact complete current problem')
    if type(optimize_budget)is not bool or not np.isfinite(root_tolerance_px) or root_tolerance_px<=0:raise ValueError('explicit budget selection and positive numerical root tolerance required')
    start=time.perf_counter();p=e.params;c=e.problem.constraints;cache=e._compute(old.parameters,False);m=cache['model'];op=cache['operator'];nb=len(e.budget_rows);rho=e.al._rho_rows(old.penalties);bp=np.full(p.count,-1,dtype=int);bp[e.budget_rows]=np.arange(nb);qb=e._budget_quantities(cache)[0];b=float(old.parameters[p.budget_offset]);working=old.parameters.copy();groups={};features={v.position.station_variable_block_id:v for v in m.landmarks if hasattr(v.position,'station_variable_block_id')};reports=[];allowed=[]
    for row,key in p.shared_landmark_blocks.items():groups.setdefault(key,[]).append(int(row))
    def actual_merit(x):
     residual,_,raw,_=e.al._evaluate(x,False);z=raw/c.scales+old.multipliers/rho;return .5*float(residual@residual)+.5*float(np.sum(rho*np.where(c.equality,z,np.maximum(z,0))**2))
    before_merit=actual_merit(old.parameters)
    for key,locations in groups.items():
     rr=np.asarray(locations,dtype=int);feature=features[key];
     if e.problem.constant_blocks[p.semantic_to_native[key]]:continue
     assert np.all(p.kinds[rr]==3) and np.all(p.station_column[rr]==-1)
     interval=feature.position.allowed_interval;lo=float(interval.lower);hi=float(interval.upper);native=p.semantic_to_native[key];off=int(p.parameter_offsets[native]);before_s=float(old.parameters[off]);curve=str(p.curve_ids[rr[0]]);assert np.all(p.curve_ids[rr]==curve);assert p.block_sizes[native]==1;bi=bp[rr];assert np.all(bi>=0);E=e.budgets[bi];grid=np.linspace(lo,hi,1025)
     def evaluate(stations):
      ss=np.asarray(stations,dtype=float).reshape(-1);X=m.scene.world_path(curve,ss,m.state);G=len(ss);out=op.project_rs(np.tile(X,(len(rr),1)),np.repeat(p.native_pts[rr],G),root_iterations=64,root_tolerance_px=root_tolerance_px);uv=out['uv'].reshape(len(rr),G,2);valid=out['numerical_root_valid'].reshape(len(rr),G);q=np.sum((uv-p.target_uv[rr,None,:])**2,axis=2)/(E[:,None]**2);scores=q.sum(axis=0);scores[~valid.all(axis=0)]=np.inf;return scores,out,q
     values,_,_=evaluate(grid);local=np.flatnonzero(np.isfinite(values)&(values<=np.r_[np.inf,values[:-1]])&(values<=np.r_[values[1:],np.inf]));candidates=[before_s,lo,hi]
     for j in local:
      left=grid[max(0,j-1)];right=grid[min(len(grid)-1,j+1)]
      if right>left:
       result=minimize_scalar(lambda s:float(evaluate([s])[0][0]),bounds=(left,right),method='bounded',options={'xatol':64*np.finfo(float).eps});candidates.extend((float(result.x),float(grid[j])))
     values,_,_=evaluate(candidates)
     if not np.isfinite(values).any():
      reports.append({'feature_id':feature.id,'shared_station_block_id':key,'accepted_detached_candidate':False,'reason':'no_jointly_valid_formal_sensor_RS_candidate_for_all_feature_exposures','native_bounds_need_exact_finite_semantic_repair':bool(p.lower[off]!=lo or p.upper[off]!=hi),'observations':[p.observation_ids[i] for i in rr]});continue
     best=int(np.argmin(values));station=float(candidates[best]);_,out,q=evaluate([station]);candidate=working.copy();candidate[off]=station
     for j,i in enumerate(rr):candidate[p.parameter_offsets[p.local_block[i]]+p.row_column[i]]=out['row'][j]
     cm=actual_merit(candidate);wm=actual_merit(working);newcache=e._compute(candidate,False);qnew=e._budget_quantities(newcache)[0];oldpositive=np.maximum(qb[bi]-b,0);newpositive=np.maximum(qnew[bi]-b,0);valid=bool(out['numerical_root_valid'].all());accepted=valid and np.all(newpositive<=oldpositive+1e-10) and cm<=wm+1e-7
     raw_before=np.linalg.norm(cache['error'][rr],axis=1);raw_after=np.linalg.norm(newcache['error'][rr],axis=1);world_before=m.scene.world_path(curve,np.array([before_s]),m.state)[0];world_after=m.scene.world_path(curve,np.array([station]),m.state)[0];obs=[]
     for j,i in enumerate(rr):obs.append({'id':p.observation_ids[i],'native_pts':int(p.native_pts[i]),'raw_uv':p.target_uv[i].tolist(),'raw_error_before_px':float(raw_before[j]),'raw_error_candidate_px':float(raw_after[j]),'q_before':float(qb[bi[j]]),'q_candidate':float(qnew[bi[j]]),'roweq_before_px':float(cache['roweq'][i]),'roweq_candidate_px':float(newcache['roweq'][i]),'formal_RS_status_code':int(out['status_code'][j]),'formal_row_numerical_root_valid':bool(out['numerical_root_valid'][j]),'unique_elimination_certificate_valid':bool(out['valid'][j])})
     reports.append({'feature_id':feature.id,'shared_station_block_id':key,'native_block_id':p.block_ids[native],'native_ambient_bounds_current':[None if not np.isfinite(p.lower[off]) else float(p.lower[off]),None if not np.isfinite(p.upper[off]) else float(p.upper[off])],'typed_allowed_interval':asdict(interval),'native_bounds_need_exact_finite_semantic_repair':bool(p.lower[off]!=lo or p.upper[off]!=hi),'exposure_count':len(rr),'one_shared_station_for_all_exposures':True,'station_before':before_s,'station_candidate':station,'feature_world_before':world_before.tolist(),'feature_world_candidate':world_after.tolist(),'FEATURE_unknown_physical_position_changed':station!=before_s,'all_formal_RS_roots_valid':valid,'accepted_detached_candidate':bool(accepted),'selection_target':'sum of engineering q for initialization, not new statistical residual; guarded by unchanged actual PHR and each original positive E violation','PHR_before_this_feature':wm,'PHR_candidate':cm,'observations':obs})
     if accepted:
      working=candidate;allowed.append(off);allowed.extend(int(p.parameter_offsets[p.local_block[i]]+p.row_column[i]) for i in rr)
    positions_merit=actual_merit(working);positions=e._compute(working,False);qa=e._budget_quantities(positions)[0];rhob=rho[:nb]/c.scales[:nb]**2;a=qa+old.multipliers[:nb]*c.scales[:nb]/rho[:nb];bstar=exact_shared_budget_phr(qa,old.multipliers[:nb],rho[:nb],c.scales[:nb]);combined=working.copy();combined[p.budget_offset]=bstar if optimize_budget else b;final_merit=actual_merit(combined);diag,_=e.al._diagnostics(combined,old.multipliers,old.penalties);after=e._compute(combined,False);allowed.append(p.budget_offset);mask=np.ones(len(combined),dtype=bool);mask[allowed]=False;assert np.array_equal(old.parameters[mask],combined[mask]);assert np.array_equal(cache['query_error'][e.profile_rows],after['query_error'][e.profile_rows]);assert positions_merit<=before_merit+1e-6 and final_merit<=positions_merit+1e-6

    final=after
    report={'schema':'rinksolve.shared_material_feature_update.v1',
        'source_native_x_sha256':hashlib.sha256(old.parameters.tobytes()).hexdigest(),
        'candidate_native_x_sha256':hashlib.sha256(combined.tobytes()).hexdigest(),
        'source_physical_state_sha256':m.state.digest(),
        'candidate_physical_state_sha256':final['model'].state.digest(),
        'seconds':time.perf_counter()-start,'shared_FEATURE_block_count':len(groups),
        'shared_exposure_count':len(p.shared_landmark_blocks),
        'accepted_FEATURE_blocks':sum(v['accepted_detached_candidate'] for v in reports),
        'physics_changed_only_in_listed_unknown_FEATURE_coordinates':True,
        'camera_scene_rig_curve_geometry_and_all_other_parameters_bit_identical':True,
        'lambda_rho_phase_source_clock_pixels_W_sigma_E_unchanged':True,
        'profile_Query_fixed_and_other_local_aux_bit_identical':True,
        'native_shared_station_bounds_missing_count':sum(v.get('native_bounds_need_exact_finite_semantic_repair',False) for v in reports),
        'PHR_before_shared':before_merit,'PHR_after_shared_only':positions_merit,
        'PHR_after_shared_and_exact_b':final_merit,'b_before':b,
        'budget_after':float(combined[p.budget_offset]),'exact_b_candidate':bstar,
        'waterfill_applied_to_candidate':optimize_budget,
        'waterfill_derivative':float(1-np.sum(rhob*np.maximum(a-bstar,0))),
        'max_q_before':float(qb.max()),'max_q_after':float(qa.max()),
        'roweq_max_before_px':float(np.abs(cache['roweq']).max()),
        'roweq_max_after_px':float(np.abs(final['roweq']).max()),
        'features':reports,'actual_Ceres_accepted_step':False}
    return combined,diag,report


def exact_shared_budget_phr(q, multipliers, penalties, normalization_scales):
    """Exact b>=0 minimizer of b plus the existing normalized inequality PHR."""
    q,mu,rho,scale=(np.asarray(v,dtype=np.float64) for v in (q,multipliers,penalties,normalization_scales))
    if q.ndim!=1 or not len(q) or any(v.shape!=q.shape for v in (mu,rho,scale)):
        raise ValueError('budget q, lambda, rho and scales must be matching nonempty vectors')
    if not all(np.isfinite(v).all() for v in (q,mu,rho,scale)) or np.any(q<0) or np.any(mu<0) or np.any(rho<=0) or np.any(scale<=0):
        raise ValueError('actual inequality data must be finite with q/lambda>=0 and rho/scale>0')
    weights=rho/(scale*scale);knots=q+mu*scale/rho
    if np.sum(weights*np.maximum(knots,0))<=1:return 0.
    order=np.argsort(knots)[::-1];a=knots[order];w=weights[order]
    candidates=(np.cumsum(w*a)-1)/np.cumsum(w);next_knots=np.r_[a[1:],-np.inf]
    admissible=np.flatnonzero((candidates<=a)&(candidates>=next_knots))
    if not len(admissible):raise ArithmeticError('finite shared-b PHR active interval was not resolved')
    return max(0.,float(candidates[admissible[0]]))


def feasibility_auxiliary_candidate(engine, state, *, root_tolerance_px=1e-7,
                                    on_progress=None):
    """Prepare the explicit kind0 -> shared FEATURE -> exact single-b chain.

    Every stage keeps the same objective, source, equations, lambda and rho.
    On a failed stage the original complete vector is returned with a failure
    report; the caller must not publish it as successful initialization.
    """
    original=state
    try:
        vector,diag,locus=kind0_formal_rs_candidate(engine,state,optimize_budget=False,root_tolerance_px=root_tolerance_px,on_progress=on_progress)
        intermediate=replace(state,parameters=vector,diagnostics=diag)
        vector,diag,features=shared_feature_position_candidate(engine,intermediate,optimize_budget=False,root_tolerance_px=root_tolerance_px,on_progress=on_progress)
        p=engine.params;c=engine.problem.constraints;nb=len(engine.budget_rows)
        cache=engine._compute(vector,False);q=engine._budget_quantities(cache)[0];rho=engine.al._rho_rows(state.penalties)
        b=exact_shared_budget_phr(q,state.multipliers[:nb],rho[:nb],c.scales[:nb]);candidate=vector.copy();candidate[p.budget_offset]=b
        def merit(x):
            r,_,v,_=engine.al._evaluate(x,False);z=v/c.scales+state.multipliers/rho
            return .5*float(r@r)+.5*float(np.sum(rho*np.where(c.equality,z,np.maximum(z,0))**2))
        before=merit(original.parameters);after=merit(candidate)
        if after>before+64*np.finfo(float).eps*max(1,abs(before)):raise ArithmeticError('combined initialization increased the actual normalized PHR')
        diag,_=engine.al._diagnostics(candidate,state.multipliers,state.penalties)
        report={'schema':'rinksolve.feasibility_auxiliary_chain.v1','accepted_detached_candidate':True,
            'source_native_x_sha256':hashlib.sha256(original.parameters.tobytes()).hexdigest(),
            'candidate_native_x_sha256':hashlib.sha256(candidate.tobytes()).hexdigest(),
            'PHR_before':before,'PHR_after':after,'budget_before':float(original.parameters[p.budget_offset]),
            'budget_after':b,'waterfill_derivative':float(1-np.sum(rho[:nb]/c.scales[:nb]**2*np.maximum(q+state.multipliers[:nb]*c.scales[:nb]/rho[:nb]-b,0))),
            'kind0':locus,'shared_FEATURE_positions':features,'lambda_rho_and_phase_unchanged':True,
            'actual_Ceres_accepted_step':False,'source_pixels_entities_clock_W_sigma_E_Query_unchanged':True}
        return candidate,diag,report
    except Exception as exc:
        return original.parameters.copy(),original.diagnostics,{'schema':'rinksolve.feasibility_auxiliary_chain.v1',
            'accepted_detached_candidate':False,'original_complete_parameters_preserved':True,
            'error':type(exc).__name__+': '+str(exc),'actual_Ceres_accepted_step':False}
