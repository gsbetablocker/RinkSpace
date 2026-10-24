"""Explicit same-physics initializers for the recovered first-shot ad solve.

These are detached candidates, never Ceres accepted steps or entity edits.
A curve/exposure grid is projected once; source points share that work. Only
existing observation-owned kind-0 finite stations/formal rows may change.
"""
from __future__ import annotations
from dataclasses import replace
import hashlib,time
import numpy as np
from .gpu_projector import CudaProjectionBatch


def ad_station_candidate(engine,state,*,root_tolerance_px=1e-7,curve_ids=None):
    if engine.phase!='feasibility':raise ValueError('this explicit source-E restoration is a feasibility initialization only')
    p=engine.params;source=np.array(state.parameters,copy=True);base=engine._compute(source,False)
    model=base['model'];op=base['operator'];physics=model.state.digest();height=p.rig.image_height-1
    tau=op.camera_definition['signed_readout_seconds'];start=time.perf_counter();pool={};proposed=source.copy()
    E=np.full(p.count,np.nan);E[engine.budget_rows]=engine.budgets
    eligible=(p.kinds==0)&(p.station_column>=0)&np.isfinite(E)&(np.linalg.norm(base['error'],axis=1)>E)
    if curve_ids is not None:eligible&=np.isin(p.curve_ids,np.asarray(tuple(curve_ids),dtype=str))
    selected=np.flatnonzero(eligible)
    if not len(selected):return state,{'schema':'rinksolve.original_ads_station_candidate.v1','Ceres_accepted_step':False,'eligible_observations':0,'PHR_guard_accepted':0,'source_and_candidate_parameters_identical':True}
    seeded=np.zeros(p.count,dtype=bool);curve_reports=[]
    def batch(xyz,times,*,full=False):
        n=len(times);capacity=1<<(max(1,n)-1).bit_length()
        if capacity not in pool:pool[capacity]=CudaProjectionBatch(op,np.zeros((capacity,3),dtype=np.float64),np.zeros(capacity,dtype=np.float64),device=engine.gpu.device)
        camera=pool[capacity];x=np.empty((capacity,3));t=np.empty(capacity)
        x[:n]=xyz;x[n:]=xyz[-1];t[:n]=times;t[n:]=times[-1]
        camera.set_rows(xyz=np.ascontiguousarray(x),time=np.ascontiguousarray(t));out=camera.evaluate() if full else camera.evaluate_uv()
        return {k:v[:n] for k,v in out.items() if isinstance(v,np.ndarray) and v.shape[:1]==(capacity,)}
    try:
        board=model.scene.board(model.state)
        for curve,all_rows in p.world_groups.items():
            rr=all_rows[eligible[all_rows]]
            if not len(rr):continue
            definition=model.scene.resolved_definition(p.curve_lookup[curve],model.state)
            nvertices=len(definition.get('unwrapped_station_height',[]))
            grid=np.unique(np.r_[np.linspace(0.,1.,33),np.linspace(0.,1.,max(2,nvertices))])
            xyz=model.scene.world_path(curve,grid,model.state,board=board)
            frames=np.unique(p.native_pts[rr]);chunk=max(1,250000//len(grid));found=0
            for begin in range(0,len(frames),chunk):
                pts=frames[begin:begin+chunk];ng=len(grid);times=p.shot.frame_time(pts)
                out=batch(np.tile(xyz,(len(pts),1)),np.repeat(times,ng))
                uv=out['uv'].reshape(len(pts),ng,2);valid=out['valid'].reshape(len(pts),ng)
                for fi,frame in enumerate(pts):
                    ii=rr[p.native_pts[rr]==frame];segments=valid[fi,:-1]&valid[fi,1:]
                    if not segments.any():continue
                    q0=uv[fi,:-1];delta=uv[fi,1:]-q0;length=np.sum(delta*delta,axis=1)
                    difference=p.target_uv[ii,None,:]-q0[None,:,:]
                    fraction=np.divide(np.sum(difference*delta[None,:,:],axis=2),length[None,:],out=np.zeros((len(ii),len(length))),where=length[None,:]>0)
                    fraction=np.minimum(1.,np.maximum(0.,fraction))
                    closest=q0[None,:,:]+fraction[:,:,None]*delta[None,:,:]
                    score=np.sum((closest-p.target_uv[ii,None,:])**2,axis=2);score[:,~segments]=np.inf
                    index=np.argmin(score,axis=1);s=grid[index]+fraction[np.arange(len(ii)),index]*(grid[index+1]-grid[index])
                    offsets=p.parameter_offsets[p.local_block[ii]]+p.station_column[ii]
                    proposed[offsets]=s;seeded[ii]=True;found+=len(ii)
            curve_reports.append({'curve_id':curve,'eligible_sources':len(rr),'unique_native_exposures':len(frames),'grid_vertices':len(grid),'seeded_sources':found,
                'grid_projections':len(frames)*len(grid),'source_times_grid_avoided':len(rr)*len(grid)-len(frames)*len(grid)})
        ii=np.flatnonzero(seeded)
        if not len(ii):return state,{'schema':'rinksolve.original_ads_station_candidate.v1','Ceres_accepted_step':False,'eligible_observations':int(eligible.sum()),'PHR_guard_accepted':0,'source_and_candidate_parameters_identical':True,'reason':'no valid projected seed segment; all existing sources and auxiliaries preserved'}
        stations,old_rows,alpha=p.auxiliaries(proposed,model)
        xyz=model.scene.world_rows(p.curve_ids[ii],stations[ii],model.state)
        left=np.zeros(len(ii));right=np.full(len(ii),float(height))
        def at(rows):return batch(xyz,p.frame_times[ii]+tau*(rows/height-.5))
        fl=at(left);fr=at(right);fleft=left-fl['uv'][:,1];fright=right-fr['uv'][:,1]
        bracket=fl['valid']&fr['valid']&(fleft*fright<=0)
        work=p.target_uv[ii,1].copy();inside=(work>=0)&(work<=height);work[~inside]=old_rows[ii][~inside]
        converged=np.zeros(len(ii),dtype=bool);failed=~bracket
        limit=int(np.ceil(np.log2(max(float(height)/root_tolerance_px,1.))))+3
        for iteration in range(limit):
            active=bracket&~converged&~failed
            if not active.any():break
            out=at(work);value=work-out['uv'][:,1]
            converged|=active&out['valid']&(np.abs(value)<=root_tolerance_px)
            failed|=active&~out['valid'];active&=~converged&~failed
            same=np.signbit(value)==np.signbit(fleft)
            choose=active&same;left[choose]=work[choose];fleft[choose]=value[choose]
            choose=active&~same;right[choose]=work[choose];fright[choose]=value[choose]
            # Bisection has an explicit finite sensor bracket; no root clipping.
            work[active]=.5*(left[active]+right[active])
        converged&=bracket&~failed
        offsets=p.parameter_offsets[p.local_block[ii]]+p.row_column[ii]
        proposed[offsets[converged]]=work[converged]
        accepted_root=np.zeros(p.count,dtype=bool);accepted_root[ii[converged]]=True
        # Failed finite-sensor roots preserve both original auxiliary values.
        for col in (p.station_column,p.row_column):
            off=p.parameter_offsets[p.local_block[ii]]+col[ii]
            proposed[off[~converged]]=source[off[~converged]]
        engine._last_x=None;trial=engine._compute(proposed,False)
        before_q,_=engine._budget_quantities(base);after_q,_=engine._budget_quantities(trial)
        constraint=engine.problem.constraints;rho=engine.al._rho_rows(state.penalties);nb=len(engine.budget_rows)
        b=float(source[p.budget_offset]);lam=np.asarray(state.multipliers)
        def per_source(cache,q):
            value=np.zeros(p.count)
            shifted=(q-b)/constraint.scales[:nb]+lam[:nb]/rho[:nb]
            value[engine.budget_rows]+=.5*rho[:nb]*np.maximum(shifted,0.)**2
            row=cache['roweq']/constraint.scales[nb:nb+p.count]+lam[nb:nb+p.count]/rho[nb:nb+p.count]
            value+=.5*rho[nb:nb+p.count]*row*row
            return value
        before_merit=per_source(base,before_q);after_merit=per_source(trial,after_q)
        before_norm=np.linalg.norm(base['error'],axis=1);after_norm=np.linalg.norm(trial['error'],axis=1)
        guard=accepted_root&(after_merit<=before_merit+64*np.finfo(float).eps*np.maximum(1.,before_merit))&(after_norm<=before_norm)
        output=source.copy();take=np.flatnonzero(guard)
        for col in (p.station_column,p.row_column):
            off=p.parameter_offsets[p.local_block[take]]+col[take];output[off]=proposed[off]
        engine._last_x=None;final=engine._compute(output,False)
        final_q,_=engine._budget_quantities(final);final_merit=per_source(final,final_q)
        if final['model'].state.digest()!=physics or not np.array_equal(output[:p.model_parameter_count],source[:p.model_parameter_count]):raise RuntimeError('station initializer changed static physical objects')
        if np.sum(final_merit)>np.sum(before_merit)+128*np.finfo(float).eps*max(1.,float(np.sum(before_merit))):raise RuntimeError('complete current PHR merit increased')
        if not np.array_equal(output[p.budget_offset:],source[p.budget_offset:]):raise RuntimeError('station initializer changed shared b')
        records=[]
        for i in selected[np.argsort(before_norm[selected])[::-1][:40]]:
            records.append({'observation_id':p.observation_ids[i],'native_pts':int(p.native_pts[i]),'curve_id':str(p.curve_ids[i]),'source_uv':p.target_uv[i].tolist(),
                'before_error_px':float(before_norm[i]),'after_error_px':float(np.linalg.norm(final['error'][i])),
                'before_station':float(base['stations'][i]),'after_station':float(final['stations'][i]),'accepted_auxiliary_reseed':bool(guard[i]),
                'formal_row_equality_after_px':float(final['roweq'][i])})
        report={'schema':'rinksolve.original_ads_station_candidate.v1','read_only_candidate':True,'Ceres_accepted_step':False,
            'source_native_x_sha256':hashlib.sha256(source.tobytes()).hexdigest(),'candidate_native_x_sha256':hashlib.sha256(output.tobytes()).hexdigest(),
            'physical_sha256_before':physics,'physical_sha256_after':final['model'].state.digest(),'static_ad_camera_ice_board_parameters_changed':False,
            'UV_PTS_source_bindings_E_W_Query_lambda_rho_changed':False,'affected_curve_filter':list(curve_ids) if curve_ids is not None else None,'eligible_observations':int(eligible.sum()),'seeded_observations':int(seeded.sum()),
            'formal_root_converged':int(converged.sum()),'formal_root_unresolved_preserved':int((~converged).sum()),'PHR_guard_accepted':int(guard.sum()),
            'PHR_constraint_merit_before':float(np.sum(before_merit)),'PHR_constraint_merit_after':float(np.sum(final_merit)),
            'source_q_max_before':float(before_q.max()),'source_q_max_after':float(final_q.max()),'shared_b_unchanged':b,
            'roweq_max_before_px':float(np.abs(base['roweq']).max()),'roweq_max_after_px':float(np.abs(final['roweq']).max()),
            'per_curve':curve_reports,'worst_before_sources':records,'seconds':time.perf_counter()-start}
        event={'kind':'explicit_same_physics_ad_station_initialization','source_native_x_sha256':report['source_native_x_sha256'],
            'candidate_native_x_sha256':report['candidate_native_x_sha256'],'Ceres_accepted_step':False,'physical_parameters_changed':False,
            'lambda_rho_budget_unchanged':True}
        return replace(state,parameters=output,phase='ready',last_inner=None,history=state.history+(event,)),report
    finally:
        for camera in pool.values():camera.close()


def migrate_complete_ad_state(source_engine,target_engine,state,*,reason):
    """Warm initialization of a changed current graph, addressed by stable IDs.

    Current inputs are consumed, never historical packets or qualification
    producers. Existing physical/auxiliary and dual values are mapped exactly;
    a new graph identity never pretends the private Ceres state was restored.
    """
    if not isinstance(reason,str) or not reason.strip():raise ValueError('explicit affected-input migration reason required')
    a=source_engine.params;b=target_engine.params
    if state.problem_digest!=source_engine.problem.digest or state.identity!=source_engine.problem.identity:raise ValueError('source complete state/current source graph disagree')
    x=np.array(b.initial,copy=True);old_blocks={key:i for i,key in enumerate(a.block_ids)};physical=[]
    for view in b.views:
        key=view.semantic_block_id
        if key not in a.selected_blocks:raise ValueError('unapproved new physical advertisement block: '+key)
        old=old_blocks[key];new=b.semantic_to_native[key]
        if a.block_sizes[old]!=b.block_sizes[new]:raise ValueError('static advertisement block dimension changed')
        x[b.parameter_offsets[new]:b.parameter_offsets[new+1]]=state.parameters[a.parameter_offsets[old]:a.parameter_offsets[old+1]]
        physical.append(key)
    source_physics=a.model_for(state.parameters)
    for key in set(a.selected_blocks)-set(b.selected_blocks):
        if not np.array_equal(target_engine.model.state.get(key),source_physics.state.get(key)):raise ValueError('another advertisement changed while constructing an independent scope: '+key)
    old_obs={key:i for i,key in enumerate(a.observation_ids)};mapped=0;new_ids=[]
    for i,key in enumerate(b.observation_ids):
        if key not in old_obs:new_ids.append(key);continue
        j=old_obs[key]
        if int(a.kinds[j])!=int(b.kinds[i]) or int(a.native_pts[j])!=int(b.native_pts[i]) or not np.array_equal(a.target_uv[j],b.target_uv[i]):raise ValueError('existing raw observation semantic kind/PTS/UV changed: '+key)
        for name in ('station_column','row_column','alpha_column'):
            ac=int(getattr(a,name)[j]);bc=int(getattr(b,name)[i])
            if (ac>=0)!=(bc>=0):raise ValueError('existing auxiliary semantic component changed: '+key+'/'+name)
            if bc>=0:
                old=int(a.parameter_offsets[a.local_block[j]])+ac;new=int(b.parameter_offsets[b.local_block[i]])+bc
                if a.lower[old]!=b.lower[new] or a.upper[old]!=b.upper[new]:raise ValueError('existing observation auxiliary domain changed: '+key+'/'+name)
                x[new]=state.parameters[old];mapped+=1
    if new_ids:raise ValueError('new observations require explicit initialization authority: '+repr(new_ids[:5]))
    x[b.budget_offset]=state.parameters[a.budget_offset]
    if not np.isfinite(x).all() or np.any(x<b.lower) or np.any(x>b.upper):raise ValueError('mapped current state violates declared native bounds')
    ac=source_engine.problem.constraints;bc=target_engine.problem.constraints
    af={key:i for i,key in enumerate(ac.layout.factor_ids)};lam=np.zeros(bc.layout.residual_count);common=0;new_factors=[]
    for j,key in enumerate(bc.layout.factor_ids):
        if key not in af:new_factors.append(key);continue
        i=af[key];alo,ahi=ac.layout.residual_offsets[i:i+2];blo,bhi=bc.layout.residual_offsets[j:j+2]
        if ahi-alo!=bhi-blo or not np.array_equal(ac.scales[alo:ahi],bc.scales[blo:bhi]):raise ValueError('existing constraint component dimension/unit scale changed: '+key)
        if not np.array_equal(ac.equality[alo:ahi],bc.equality[blo:bhi]):raise ValueError('existing constraint sense changed: '+key)
        lam[blo:bhi]=state.multipliers[alo:ahi];common+=int(ahi-alo)
    if new_factors:raise ValueError('new constraint factors need explicit dual initialization: '+repr(new_factors[:5]))
    families={v.id:i for i,v in enumerate(ac.families)};rho=[]
    for f in bc.families:
        if f.id not in families:raise ValueError('new constraint family requires explicit penalty initialization: '+f.id)
        old=ac.families[families[f.id]]
        if (old.sense,old.unit)!=(f.sense,f.unit):raise ValueError('existing constraint family sense or unit changed')
        rho.append(float(state.penalties[families[f.id]]))
    rho=np.asarray(rho)
    current=target_engine.params.model_for(x);target_engine.original_reference_guard.assert_free_blocks(current,physical)
    diagnostics,q=target_engine.al._diagnostics(x,lam,rho)
    violation=target_engine.al._measure(q,lam,rho)
    event={'kind':'explicit_changed_current_graph_warm_initialization','reason':reason,
        'source_problem_digest':state.problem_digest,'target_problem_digest':target_engine.problem.digest,
        'physical_blocks_preserved':physical,'mapped_observation_auxiliary_components':mapped,'mapped_constraint_dual_components':common,
        'removed_observations':len(a.observation_ids)-len(b.observation_ids),'removed_constraint_factors':len(ac.layout.factor_ids)-len(bc.layout.factor_ids),
        'parameters_lambda_rho_by_stable_semantic_ID':True,'previous_violation_and_diagnostics_recomputed_on_current_equations':True,
        'Ceres_accepted_step':False,'private_Ceres_TR_history_restored':False,'new_graph_is_same_old_numerical_problem':False}
    output=replace(state,problem_digest=target_engine.problem.digest,identity=target_engine.problem.identity,
        parameters=x,multipliers=lam,penalties=rho,previous_violation=violation,outer_iteration=0,phase='ready',
        diagnostics=diagnostics,last_inner=None,history=state.history+(event,))
    return output,event
