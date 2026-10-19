"""Replay a sealed normal with public truth owners, never a programme graph/J."""
from pathlib import Path
import argparse,json,os,sys,time
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RINKSPACE_MOTION_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts/diagnostics')]
from calibration.solver_runtime import install
install()
import audit_current_user_ice_truth as a
import numpy as np
from scipy.sparse import csc_matrix,save_npz,load_npz
from calibration.solver_runtime.finite_rectangle_station_step_constraints import finite_rectangle_station_constraints,actual_station_model_carriers
from structured_programme_source_candidate.source_step_curvature_tightening import constraints_on_declared_free_chart,actual_carrier_step_response,measured_outward_step_tightening
from calibration.solver_runtime.optional_soc_runtime import install_optional_SOC_numerical_failure_route,original_nonlinear_step_face_feasibility

def emit(**value):print(json.dumps(value,ensure_ascii=False),flush=True)

def replay_cached_domain(args, state, canonical_q, q, H,g,profile,mapping,selected,fixed,human,wide,confirmed,parent,step_report,began):
    from calibration.solver_runtime.finite_rectangle_station_step_constraints import _current_boundary_clipped_primitive
    d=args.domain_from;out=args.output
    with np.load(d/'actual_current_original_domain.npz',allow_pickle=False)as z:
        np.testing.assert_array_equal(z['q'],canonical_q);np.testing.assert_array_equal(z['shared_indices'],selected)
        A=z['matrix'].copy();lower=z['lower'].copy();ownership=z['row_group_index'].copy();v=z['current_violations'].copy()
    np.testing.assert_array_equal(v,step_report['report']['truth_feasibility_priority']['current_original_group_violations_px'])
    responses={}
    with np.load(d/'actual_current_original_carrier_maps.npz',allow_pickle=False)as z:
        for family in ('native','wide','confirmed'):
            responses[family]={k[len(family)+1:]:z[k].copy()for k in z.files if k.startswith(family+'_')}
            for key in ('row_start','row_stop'):responses[family][key]=int(responses[family][key])
            p=d/(family+'_row_carrier_projection.npz')
            if p.exists():responses[family]['row_carrier_projection']=load_npz(p)
    offsets=np.r_[0,np.cumsum([2*len(r['uv'])for r in human.full_stroke_records])];contexts=[]
    for group,(record,result)in enumerate(zip(human.full_stroke_records,human.queries(q,full_stroke=True))):
        query=result['query']
        if query.get('finite_source_association')!='original_support_rectangle_Linf':continue
        world=_current_boundary_clipped_primitive(record,q)
        if world is None:continue
        segments=np.asarray(query['nearest_finite_segment_index']);stations=np.asarray(query['nearest_finite_segment_station'])
        for vertex in np.flatnonzero((segments==0)|(segments==len(world)-2)):
            contexts.append(dict(record=record,group=group,source_vertex=int(vertex),segment=int(segments[vertex]),station=float(stations[vertex]),original_carrier_offset=int(offsets[group]+2*vertex)))
    responses['native']['finite_station_contexts']=contexts
    def carriers(value):
        trial=state.evaluation_q(value)
        return dict(native=actual_station_model_carriers(human,trial,responses['native']),wide=wide._carriers(trial),confirmed=confirmed._carriers(trial)[0])
    base=carriers(canonical_q)
    for family,r in responses.items():np.testing.assert_array_equal(base[family],r['nominal_carrier'])
    import z190_joint_source_solver as joint
    install_optional_SOC_numerical_failure_route(parallel_backend='faer')
    lo=np.full(len(selected),-np.inf);hi=np.full(len(selected),np.inf)
    for col,pair in step_report['finite_working_intervals'].items():
        j=int(np.flatnonzero(selected==int(col))[0]);lo[j],hi[j]=pair
    cfg=a.read(args.config)
    opts=dict(global_count=len(selected),global_parameter_indices=selected,damping=cfg['source_options']['--initial-damping'],schur_view_lm_ratio=1.,global_step_lower_bounds=lo,global_step_upper_bounds=hi,global_step_constraint_matrix=A,global_step_constraint_lower=lower)
    diagonal=np.maximum(H.diagonal(),1.);linear=joint.schur_trial_linear_step(H,g,diagonal,**opts);records=[];steps=[]
    def source_response(value,family):return actual_carrier_step_response(responses[family],carriers(value)[family])
    for cycle in range(3):
        np.testing.assert_array_equal(linear.step[fixed],0.);step=linear.step.copy();steps.append(step)
        np.savez(out/('complete_direction_'+str(cycle)+'.npz'),q=canonical_q,step=step,canonical_to_evaluation=mapping,selected=selected,frozen_profile_token=np.asarray(profile))
        actual=carriers(canonical_q+step);merit=original_nonlinear_step_face_feasibility(A,lower,step[selected],responses,actual)
        slack=A@step[selected]-lower
        for family,r in responses.items():slack[r['row_start']:r['row_stop']]=actual_carrier_step_response(r,actual[family])-lower[r['row_start']:r['row_stop']]
        worst=int(np.argmin(slack));family=next(f for f,r in responses.items()if r['row_start']<=worst<r['row_stop']);r=responses[family];local=worst-r['row_start'];gid=int(ownership[worst])
        identity=dict(family=family,group=gid,local_face=local,original_face=worst)
        if family=='native':
            columns=r['row_carrier_projection'].getrow(local).indices
            record=human.full_stroke_records[gid];identity.update(source_id=record['row']['id'],curve=record['row']['curve'],carrier_indices=columns.tolist())
            vertex_columns=[int(c-offsets[gid])for c in columns if offsets[gid]<=c<offsets[gid+1]]
            if vertex_columns:
                vertex=vertex_columns[0]//2;identity.update(source_vertex=vertex,axis='normal'if vertex_columns[0]%2==0 else'tangent',source_uv=record['uv'][vertex].tolist())
        derivatives=[]
        for h in (1e-5,1e-6):
            for anchor,center,direction in [('parent',canonical_q,step),('trial',canonical_q+step,step)]+([]if cycle==0 else[('previous_trial_correction',canonical_q+steps[-2],step-steps[-2])]):
                plus=source_response(center+h*direction,family)[local];minus=source_response(center-h*direction,family)[local]
                derivatives.append(dict(anchor=anchor,h=h,actual_FD=float((plus-minus)/(2*h)),frozen_Jv=float(A[worst]@direction[selected])))
        entry=dict(cycle=cycle,schur_view_lm_ratio=1.,actual_original_face_feasibility=merit,worst_source_identity=identity,source_directional_derivatives=derivatives,full_KKT_report=linear.report['shared_linear_domain_step'])
        records.append(entry);a.save(out/('measured_direction_'+str(cycle)+'.json'),entry)
        emit(stage='same_production_ratio_actual_fullstep_face',cycle=cycle,maximum=merit['maximum_violation_px'],total=merit['total_violation_px'],identity=identity,derivatives=derivatives)
        correction=measured_outward_step_tightening(A,lower,step[selected],responses,actual)
        if correction is None:break
        opts['global_step_constraint_lower']=np.maximum(opts['global_step_constraint_lower'],correction['lower'])
        if cycle<2:linear=joint.schur_trial_linear_step(H,g,diagonal,**opts)
    a.save(out/'actual_same_normal_full_directions_and_SOC_face_replay.json',dict(passed=True,schur_view_lm_ratio=1.,normal=a.ref(args.run/'first_complete_source_normal_numeric.npz'),source_profile_seal=a.ref(args.run/'actual_complete_cache_seal.json'),cached_guard_domain=a.ref(d/'actual_current_original_domain.npz'),nominal_all86_carriers_byteexact=True,full_parameters=len(q),records=records,source_J_or_programme_graph_rebuilt=False,full_source_cost_and_production_acceptance_not_claimed=True,seconds=time.perf_counter()-began))
    emit(stage='saved_normal_original_ratio_full_direction_replay_complete',seconds=time.perf_counter()-began,output=str(out))

def main():
    cli=argparse.ArgumentParser();cli.add_argument('--run',type=Path,required=True)
    cli.add_argument('--config',type=Path,required=True);cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--domain-from',type=Path)
    args=cli.parse_args();began=time.perf_counter();run=args.run;out=args.output;out.mkdir(exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Existing actual replay artifacts are preserved')
    config=a.read(args.config);options=config['source_options'];directory=run/'initial_states'/config['continuation']['state']['q_sha256']
    manifest=a.read(run/'input_manifest.json');parent=a.read(options['--truth-restoration-state'])
    step_report=a.read(run/'actual_initial_shared_step_domain.json');selected=np.asarray(step_report['effective_shared_parameter_indices'],int)
    with np.load(run/'first_complete_source_normal_numeric.npz',allow_pickle=False)as z:
        canonical_q=z['q'].copy();H=csc_matrix((z['normal_data'].copy(),z['normal_indices'].copy(),z['normal_indptr'].copy()),shape=tuple(z['normal_shape']))
        g=z['gradient'].copy();normal_report=json.loads(str(z['report_json']));profile=str(z['frozen_profile_token'])
    state,q,identity,error=a.load_state(directory,Path(options['--world-init']),manifest);assert error==0.
    mapping=state.canonical_parameter_layout.canonical_to_evaluation
    np.testing.assert_array_equal(q,state.evaluation_q(canonical_q));evalcols=mapping[selected]
    fixed=np.asarray(normal_report['fixed_parameter_indices'],int);fixed_eval=mapping[fixed]
    assert not np.isin(selected,fixed).any()
    identity['_accepted_directory']=str(directory)
    board=a.NativeBoardEdgeFactor(state,options['--additional-board-source'],len(q));ice=a.NativeIceCurveFactor(state,options['--additional-ice-source'],len(q))
    human=a.bind_human(state,board,ice,options,len(q));wide=a.wide_factor(state,identity,options,len(q),options['--practical-human-location-envelope'])
    circle,material,placement=a.existing_LOW_source_owners(state,q,directory,options,a.read(directory/'scene_UNACCEPTED.json'))
    confirmed=a.UserConfirmedSourceTruth(state,options['--user-confirmed-source-truth-authority'],board,ice,parameter_count=len(q),circle_factor=circle,material_factor=material,placement_source=placement)
    factors={'native':human,'wide':wide,'confirmed':confirmed}
    actual=a.values(human,wide,confirmed,q)
    v=np.concatenate([actual['buffers'][family]for family in factors])
    priority=step_report['report']['truth_feasibility_priority'];np.testing.assert_array_equal(v,priority['current_original_group_violations_px'])
    target=priority['demanded_common_original_violation_fraction']
    if args.domain_from is not None:
        return replay_cached_domain(args,state,canonical_q,q,H,g,profile,mapping,selected,fixed,human,wide,confirmed,parent,step_report,began)
    # Exact pivot rows are completely recorded by the original public API:
    # original XY normal, every nonzero world derivative and tangent lower.
    pivot=step_report['report']['pivot_domain'];pos={int(c):i for i,c in enumerate(evalcols)}
    P=np.zeros((len(pivot['installations']),len(selected)))
    for i,row in enumerate(pivot['installations']):
        for c,d in zip(row['pivot_parameter_indices'][:2],row['outward_normal_xy']):P[i,pos[c]]=d
    byid={r['installation_id']:i for i,r in enumerate(pivot['installations'])}
    for row in pivot['world_derivative_branches']:P[byid[row['installation_id']],pos[row['column']]]=row['derivative']
    pivot_lower=np.asarray(pivot['step_lower_working_m'])
    print_domain=state.geometry.current.print_vertical_domain_step_constraints(q[state.geometry_parameter_indices])
    np.testing.assert_equal(print_domain['report'],step_report['report']['print_vertical_domain'])
    pm=np.asarray(print_domain['matrix']);worldcols=np.asarray(state.geometry_parameter_indices)
    print_A=np.zeros((len(pm),len(selected)))
    for j,col in enumerate(worldcols):
        if int(col)not in pos:assert not np.any(pm[:,j])
        else:print_A[:,pos[int(col)]]=pm[:,j]
    # Importing the function defines its original API; it does not run main().
    from run_current_named66_full_programme import circle_height_step_constraints
    circles=circle_height_step_constraints(circle,q,evalcols)
    np.testing.assert_equal(circles['report'],step_report['report']['additional_circle_height_domain'])
    matrices=[P,print_A,circles['matrix']];lowers=[pivot_lower,print_domain['lower'],circles['lower']]
    own=[np.full(sum(len(x)for x in lowers),-1,int)];responses={};cursor=len(own[0]);groupoffset=0;face_sources=[]
    for family,factor in factors.items():
        limits=np.asarray(parent['families'][family]['limits'])
        builder=finite_rectangle_station_constraints if family=='native'else constraints_on_declared_free_chart
        kw=dict(group_limits=limits,return_step_carrier_response=True)
        if family=='native':kw['full_stroke']=True
        source=builder(factor,q,evalcols,fixed_eval,**kw)
        matrix=np.asarray(source['matrix']);lower=np.asarray(source['lower']);response=source['step_carrier_response']
        if 'row_carrier_projection'in response:
            ids=np.asarray(response['row_group_index']);expansion=np.asarray(response['row_limit_expansion'])
            projection=response['row_carrier_projection'];face_carriers=[projection.getrow(i).indices.tolist()for i in range(len(lower))]
        else:
            records=factor.full_stroke_records if family=='native'else factor.records
            sizes=[2*len(r['uv'])for r in records]if family!='confirmed'else [r.get('carrier_size',len(r['row']['uv'])*(2 if r['normals']is not None else 1))for r in records]
            group=np.repeat(np.arange(len(sizes)),sizes);exp=np.repeat(limits,sizes);lg=group.copy();le=exp.copy()
            if family=='confirmed':
                j=0
                for record,size in zip(records,sizes):
                    if not record.get('expand_lower_faces',record['normals']is not None):lg[j:j+size]=-1;le[j:j+size]=0.
                    j+=size
            mask=response['retained_face_mask'];ids=np.r_[lg,group][mask];expansion=np.r_[le,exp][mask]
            C=len(response['nominal_carrier']);face_carriers=[[int(i)]for i in np.r_[np.arange(C),np.arange(C)][mask]]
        sizes=([2*len(r['uv'])for r in human.full_stroke_records]if family=='native'else [])
        offsets=np.r_[0,np.cumsum(sizes)]
        for i,(gid,carriers)in enumerate(zip(ids,face_carriers)):
            record=dict(family=family,group=int(gid),family_face=i,carrier_indices=carriers)
            if family=='native'and gid>=0:
                row=human.full_stroke_records[int(gid)];record['source_id']=row['row']['id'];record['curve']=row['row']['curve']
                local=[c-int(offsets[gid])for c in carriers if offsets[gid]<=c<offsets[gid+1]]
                if local:
                    vertex=local[0]//2;record.update(source_vertex=vertex,axis='normal'if local[0]%2==0 else'tangent',source_uv=row['uv'][vertex].tolist())
            face_sources.append(record)
        response=dict(response,row_start=cursor,row_stop=cursor+len(lower));responses[family]=response
        matrices.append(matrix);lowers.append(lower+expansion)
        own.append(np.where(ids>=0,ids+groupoffset,-1));cursor+=len(lower);groupoffset+=len(limits)
    A=np.vstack(matrices);base_lower=np.concatenate(lowers);ownership=np.concatenate(own)
    assert len(base_lower)==step_report['domain_rows'];assert len(selected)==285 and groupoffset==86
    coeff=np.zeros(len(base_lower));mask=ownership>=0;coeff[mask]=np.where(v[ownership[mask]]>1e-8,v[ownership[mask]],0.)
    lower=base_lower-target*coeff
    domain=dict(matrix=A,lower=lower,curvature_responses=responses)
    np.savez(out/'actual_current_original_domain.npz',matrix=A,lower=lower,original_lower=base_lower,row_group_index=ownership,current_violations=v,shared_indices=selected,q=canonical_q)
    response_arrays={}
    for family,r in responses.items():
        for key,value in r.items():
            if key=='row_carrier_projection':save_npz(out/(family+'_row_carrier_projection.npz'),value)
            elif isinstance(value,(np.ndarray,int,float)):response_arrays[family+'_'+key]=value
    np.savez(out/'actual_current_original_carrier_maps.npz',**response_arrays)
    emit(stage='same_sealed_normal_current_public86_domain_ready',seconds=time.perf_counter()-began,rows=len(lower),source_J_or_programme_graph_rebuilt=False)
    lo=np.full(len(selected),-np.inf);hi=np.full(len(selected),np.inf)
    for col,pair in step_report['finite_working_intervals'].items():
        j=int(np.flatnonzero(selected==int(col))[0]);lo[j],hi[j]=pair
    import z190_joint_source_solver as joint
    install_optional_SOC_numerical_failure_route(parallel_backend='faer')
    options=dict(global_count=len(selected),global_parameter_indices=selected,damping=config['source_options']['--initial-damping'],schur_view_lm_ratio=1.,global_step_lower_bounds=lo,global_step_upper_bounds=hi,global_step_constraint_matrix=A,global_step_constraint_lower=lower)
    diagonal=np.maximum(H.diagonal(),1.)
    linear=joint.schur_trial_linear_step(H,g,diagonal,**options)
    records=[]
    for cycle in range(3):
        np.testing.assert_array_equal(linear.step[fixed],0.)
        np.savez(out/('complete_direction_'+str(cycle)+'.npz'),q=canonical_q,step=linear.step,canonical_to_evaluation=mapping,selected=selected,frozen_profile_token=np.asarray(profile))
        trial=state.evaluation_q(canonical_q+linear.step)
        carriers=dict(native=actual_station_model_carriers(human,trial,responses['native']),wide=wide._carriers(trial),confirmed=confirmed._carriers(trial)[0])
        merit=original_nonlinear_step_face_feasibility(A,lower,linear.step[selected],responses,carriers)
        slack=A@linear.step[selected]-lower
        for family,r in responses.items():slack[r['row_start']:r['row_stop']]=actual_carrier_step_response(r,carriers[family])-lower[r['row_start']:r['row_stop']]
        worst=int(np.argmin(slack));physical_count=len(own[0]);identity=None if worst<physical_count else face_sources[worst-physical_count]
        correction=measured_outward_step_tightening(A,lower,linear.step[selected],responses,carriers)
        entry=dict(cycle=cycle,actual_original_face_feasibility=merit,worst_original_face=worst,worst_source_identity=identity,full_KKT_report=linear.report['shared_linear_domain_step'])
        a.save(out/('measured_direction_'+str(cycle)+'.json'),entry);records.append(entry)
        emit(stage='actual_saved_complete_direction_source_face',cycle=cycle,maximum=merit['maximum_violation_px'],total=merit['total_violation_px'],worst_face=worst,identity=identity)
        if correction is None:break
        options['global_step_constraint_lower']=np.maximum(options['global_step_constraint_lower'],correction['lower'])
        # Exactly three recorded diagnostic witnesses, not a production
        # stopping policy or a successful trial acceptance declaration.
        if cycle<2:linear=joint.schur_trial_linear_step(H,g,diagonal,**options)
    a.save(out/'actual_same_normal_full_directions_and_SOC_face_replay.json',dict(passed=True,normal=a.ref(run/'first_complete_source_normal_numeric.npz'),original_source_profile_seal=a.ref(run/'actual_complete_cache_seal.json'),full_parameters=len(q),records=records,source_J_or_programme_graph_rebuilt=False,full_source_cost_and_production_acceptance_not_claimed=True,seconds=time.perf_counter()-began))
    emit(stage='same_normal_actual_SOC_face_replay_saved',seconds=time.perf_counter()-began,output=str(out))

if __name__=='__main__':main()
