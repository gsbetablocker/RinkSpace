"""Read-only LP diagnosis of the declared morning-state step half-spaces.

This does not produce a solver state or change the QP, source, or guards.
It reconstructs only the small physical/material owner and existing cameras.
"""
from pathlib import Path
import os,sys,json,pickle,time,hashlib
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
from calibration.solver_runtime import install
install()
import numpy as np
from scipy.optimize import linprog
from prepared_cache_helpers import install_mappingproxy_pickle,install_geometry_unpickle
from verify_current_native_primitives_20261007 import state_camera_adapter
from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
from structured_programme_source_candidate.human_stroke_truth import HumanStrokeTruth,WideStrokeTruth,UserConfirmedSourceTruth
from structured_programme_source_candidate.current_wide_source import CurrentWideSource

def main():
    started=time.perf_counter()
    r=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003')
    a=r/'okada_joint_morning_c134965_commonaxis_board79_fulltarget_20261008_184808_452'
    c=r/'okada_joint_morning_c134965_cached_fastQP_20261008_192005_957'
    old=r/'okada_joint_compact_fastcamera_c134_20261008_104825_235'
    initial=old/'initial_states/965e58d4c3ac40833090d4d9db639dd2870dbb17558d449893d10b72405f4031'
    auth=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/evidence/user_video_review_20261008/c134_full_programme/registered_user_complete_stroke_source_append_20261008')
    output=c/'actual_shared_halfspace_LP_diagnosis.json'
    if output.exists()and not any(flag in sys.argv for flag in ('--existing-guard-mapping','--compare-accepted')):raise FileExistsError(output)
    manifest=json.loads((a/'input_manifest.json').read_text())
    domain=json.loads((a/'actual_initial_shared_step_domain.json').read_text())
    with np.load(a/'solver/latest_accepted_state.npz',allow_pickle=False)as z:
        arrays={k:z[k]for k in z.files if k=='q'or k.startswith('aux__')}
    identity=json.loads((initial/'state_identity.json').read_text())
    with np.load(identity['camera_path'],allow_pickle=False)as z:cameras={k:z[k]for k in z.files}
    owner=json.loads((c/'actual_absolute_geometry_FD_pool_runtime.json').read_text())['material_owner']
    install_mappingproxy_pickle();install_geometry_unpickle()
    with Path(owner['path']).open('rb')as stream:provider=pickle.load(stream)
    state,camera_error=state_camera_adapter(arrays,cameras,provider)
    layout=state.canonical_parameter_layout
    canonical=arrays['q'];q=state.evaluation_q(canonical)
    selected=np.asarray(manifest['shared_indices'],int)
    columns=layout.canonical_to_evaluation[selected]
    world=q[state.geometry_parameter_indices]
    assert provider.current.shared_hockey_curling_axis is True
    state.current_user_physical_course_scope_source=auth/'user20_existing_lower_long_side_connected_lower_right_scope_authority_v1.json'
    board=NativeBoardEdgeFactor(state,auth/'board_parent77_plus_user_confirmed230501120_top_contact_v1.json',len(q))
    ice=NativeIceCurveFactor(state,auth/'ice_parent51_plus_user19_right_goal_baseline_registered_complete_stroke_FIT_CANDIDATE_v2_HUMAN_LABEL_SUPPORT.json',len(q))
    truth=HumanStrokeTruth(state,auth/'human_registered_hard_truth_authorities_v1.json',board,ice,parameter_count=len(q))
    truth.bind_full_stroke_guard(auth/'human_registered_full_stroke_hard_guard_v1.json')
    truth.bind_stroke_cap_support(auth/'human_registered_stroke_end_2D_source_footprint_support_v2.json')
    envelope=r/'human_hardtruth_native5760_wide733_caps_PhaseI_20261008_142016_806/fixed_practical_manual_location_envelope_v2_ENGINEERING_LOCALIZATION.json'
    truth.bind_practical_location_envelope(envelope)
    scene=json.loads((initial/'scene_UNACCEPTED.json').read_text(encoding='utf8'))
    camera25=[row['camera25']for row in scene['frame_cameras']if row.get('source_pts')==56218880][0]
    wauth=auth/'human_original_wide_top_contours_hard_authority_v1.json'
    wmeta=json.loads(wauth.read_text(encoding='utf8'));trace=json.loads(Path(wmeta['trace_packet_reference']['path']).read_text(encoding='utf8'))
    wide=CurrentWideSource.__new__(CurrentWideSource)
    wide.geometry=provider.current;wide.qualified_scene=provider.qualified_scene
    wide.geometry_parameter_indices=state.geometry_parameter_indices;wide.observations=[];wide.ad_packet={'rows':[]}
    wide.template_camera25=np.asarray(camera25,float);wide.trace_packet=trace
    wide.trace_uv=[np.asarray(row['observed_centres_source_px'],float)for row in trace['contours']]
    wi=layout.canonical_to_evaluation[np.asarray(arrays['aux__current_wide_camera_parameter_indices'],int)]
    wt=WideStrokeTruth(wide,wauth,len(q),wi);wt.bind_practical_location_envelope(envelope)
    confirmed=UserConfirmedSourceTruth(state,auth/'user_confirmed_native_source_priority_authorities_v1.json',board,ice,parameter_count=len(q))
    if '--compare-accepted'in sys.argv:
        path=Path(sys.argv[sys.argv.index('--compare-accepted')+1]);ident=json.loads(path.read_text(encoding='utf8'))
        with np.load(path.parent/'complete_state_UNACCEPTED.npz',allow_pickle=False)as z:nextq=state.evaluation_q(z['q'])
        rows=[]
        for family,factor in [('native',truth),('wide',wt),('confirmed',confirmed)]:
            if family=='confirmed':
                _,_,_,before=factor._carriers(q);_,_,_,after=factor._carriers(nextq)
                for x,y in zip(before,after):rows.append(dict(family=family,id=x['id'],
                    before_max_violation_px=x['maximum_violation_px'],after_max_violation_px=y['maximum_violation_px']))
            else:
                kwargs={'full_stroke':True}if family=='native'else{}
                before=factor.queries(q,**kwargs);after=factor.queries(nextq,**kwargs)
                for x,y in zip(before,after):rows.append(dict(family=family,id=x['id'],
                    before_max_violation_px=float(abs(x['query']['residual']).max()),
                    after_max_violation_px=float(abs(y['query']['residual']).max())))
        nextcams=dict(np.load(ident['camera_path'],allow_pickle=False));at={int(p):i for i,p in enumerate(nextcams['pts'])}
        error=0.
        for stamp in set(record['row']['pts']for record in truth.records):
            actual=state.original.aliases[int(stamp)]
            error=max(error,float(abs(state.bypts[actual].bundle.camera(nextq,actual)-nextcams['camera25'][at[int(stamp)]]).max()))
        assert error<1e-10,error
        result=dict(schema='rinkspace.actual-accepted-full-stroke-group-before-after.v1',
            state_identity=str(path),q_sha256=ident['q_sha256'],same_actual_saved_camera_max_difference=error,
            before_q_sha256=hashlib.sha256(canonical.tobytes()).hexdigest(),groups=rows,
            normal_or_source_J_computed=False,accuracy_accepted=False,seconds=time.perf_counter()-started)
        target=path.parent/'actual_all_human_group_before_after.json'
        if target.exists():raise FileExistsError(target)
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps(result),flush=True);return
    if '--existing-guard-mapping'in sys.argv:
        policy=json.loads(envelope.read_text(encoding='utf8'));reports=[]
        for family,factor in [('native',truth),('wide',wt),('confirmed',confirmed)]:
            if family=='confirmed':
                value,lo,hi,groups=factor._carriers(q)
                violations=np.asarray([row['maximum_violation_px']for row in groups])
                uncertainty=np.asarray([row['radius'].min()for row in factor.records])
                sizes=[len(row['row']['uv'])*(2 if row['normals']is not None else 1)for row in factor.records]
            else:
                queries=factor.queries(q)
                violations=np.asarray([abs(row['query']['residual']).max()for row in queries])
                if family=='native':
                    queries=factor.queries(q,full_stroke=True)
                    violations=np.maximum(violations,[abs(row['query']['residual']).max()for row in queries])
                records=factor.full_stroke_records if family=='native'else factor.records
                value=np.concatenate([np.c_[row['query']['signed_normal_px'],row['query']['coverage_tangent_px']].ravel()for row in queries])
                lo=np.concatenate([np.c_[row['bounds'][:,0],row['tangent_bounds'][:,0]].ravel()for row in records])
                hi=np.concatenate([np.c_[row['bounds'][:,1],row['tangent_bounds'][:,1]].ravel()for row in records])
                uncertainty=np.asarray([row['normal_expansion_px']for row in policy['records']if row['family']==family])
                sizes=[2*len(row['uv'])for row in records]
            limits=violations+uncertainty;expanded=np.repeat(limits,sizes)
            newlo=lo-expanded;newhi=hi+expanded
            if family=='confirmed':
                at=0
                for item,size in zip(factor.records,sizes):
                    if item['normals']is None:newlo[at:at+size]=lo[at:at+size]
                    at+=size
            target=np.r_[newlo-value,value-newhi]
            if family=='native':
                oldtarget=np.r_[lo-value,value-hi]
                raw_group=np.concatenate([np.full(size,i,int)for i,size in enumerate(sizes)])
                raw_group=np.r_[raw_group,raw_group]
                with np.load(c/'actual_shared_halfspace_LP_inputs.npz',allow_pickle=False)as saved:
                    savedtarget=saved['lower'][saved['labels']=='native']
                matched=[];cursor=0
                for item in savedtarget:
                    candidates=np.flatnonzero(oldtarget[cursor:]==item)
                    if not len(candidates):raise ValueError(('Original native halfspace value did not reproduce',len(matched),item))
                    cursor+=int(candidates[0]);matched.append(cursor);cursor+=1
                critical=[]
                for globalrow in (4554,10610,10453,5473):
                    local=globalrow-47;raw=int(matched[local]);carrier=raw%(len(oldtarget)//2)
                    group=int(raw_group[raw]);start=sum(sizes[:group]);vertex=(carrier-start)//2
                    critical.append(dict(original_global_halfspace=globalrow,id=records[group]['row']['id'],
                        source_vertex=int(vertex),carrier_component='normal'if (carrier-start)%2==0 else'finite_coverage_tangent',
                        source_UV=records[group]['uv'][vertex].tolist(),original_lower=float(oldtarget[raw]),
                        carrier_value=float(value[carrier]),side='lower'if raw<len(oldtarget)//2 else'upper'))
                (c/'actual_LP_critical_native_source_rows.json').write_text(json.dumps(dict(
                    schema='rinkspace.readonly-critical-LP-native-source-row-identification.v1',
                    method='Exact monotone matching of saved filtered target to the same original carrier halfspaces; no source deletion or mutation',
                    all_11808_saved_native_target_values_reproduced=True,critical_rows=critical),indent=2)+'\n',encoding='utf8')
            assert np.max(target)<=1e-8,(family,np.max(target))
            # This is a pointwise algebraic equivalence, not an LP policy:
            # abs(value-clip(value,lo,hi)) <= limit iff lo-limit<=value<=hi+limit.
            oldtest=abs(value-np.clip(value,lo,hi))<=expanded
            newtest=(value>=newlo)&(value<=newhi)
            assert np.array_equal(oldtest,newtest)
            reports.append(dict(family=family,actual_group_violation=violations.tolist(),
                existing_guard_uncertainty=uncertainty.tolist(),existing_guard_limit=limits.tolist(),
                all_actual_carrier_guard_decisions_equal=True,
                maximum_new_zero_step_halfspace_lower=float(np.max(target)),
                current_carrier_derivatives_unchanged_by_constant_bound_translation=True,
                ordinary_point_count=len(value),source_UV_sigma_or_final_band_changed=False))
        with np.load(c/'actual_shared_halfspace_LP_inputs.npz',allow_pickle=False)as z:
            physical=np.isin(z['labels'],['pivot','print'])
            assert np.max(z['lower'][physical])<=1e-8
            box_roundoff=max(float(np.maximum(z['box_lower'],0.).max()),float(np.maximum(-z['box_upper'],0.).max()))
            assert box_roundoff<=np.finfo(float).eps
        output=c/'actual_existing_nonlinear_guard_linear_interval_mapping.json'
        report=dict(schema='rinkspace.actual-existing-nonlinear-group-guard-halfspace-equivalence.v1',
            q_sha256=hashlib.sha256(canonical.tobytes()).hexdigest(),source_camera_max_difference=camera_error,
            groups=reports,zero_shared_step_is_feasible_for_mapped_existing_guard=True,
            physical_pivot_print_and_parameter_box_RHS_unchanged=True,new_optimization_or_LP_strategy=False,
            zero_step_existing_box_roundoff_max=box_roundoff,
            same_carrier_and_derivative_operator=True,seconds=time.perf_counter()-started)
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps(report),flush=True);return
    matrices=[];targets=[];labels=[]
    def append(name,result):
        matrices.append(result['matrix']);targets.append(result['lower'])
        labels.extend([name]*len(result['lower']))
        print(json.dumps(dict(stage=name,rows=len(result['lower']),seconds=time.perf_counter()-started)),flush=True)
    pd=provider.current.print_vertical_domain_step_constraints(world)
    reduced=np.zeros((len(pd['lower']),len(columns)));mapping={int(col):i for i,col in enumerate(columns)}
    for i,col in enumerate(state.geometry_parameter_indices):
        if int(col)in mapping:reduced[:,mapping[int(col)]]=pd['matrix'][:,i]
        else:assert not np.any(pd['matrix'][:,i])
    # These saved pivot tangents were produced at the identical canonical q.
    # Their exact derivative values and lower margins are authoritative receipts.
    pivot=domain['report']['pivot_domain'];pm=np.zeros((len(pivot['installations']),len(columns)))
    for i,entry in enumerate(pivot['installations']):
        for j,col in enumerate(entry['pivot_parameter_indices'][:2]):pm[i,mapping[int(col)]]=entry['outward_normal_xy'][j]
        for item in pivot['world_derivative_branches']:
            if item['installation_id']==entry['installation_id']:pm[i,mapping[int(item['column'])]]=item['derivative']
    append('pivot',dict(matrix=pm,lower=np.asarray(pivot['step_lower_working_m'])))
    append('print',dict(matrix=reduced,lower=np.asarray(pd['lower'])))
    append('native',truth.constraints(q,columns,full_stroke=True))
    append('wide',wt.constraints(q,columns))
    append('confirmed',confirmed.constraints(q,columns))
    A=np.vstack(matrices);b=np.concatenate(targets);labels=np.asarray(labels)
    assert A.shape==(domain['domain_rows'],domain['shared_count']),A.shape
    bounds=[(None,None)for _ in selected]
    for col,interval in domain['finite_working_intervals'].items():
        pos=int(np.flatnonzero(selected==int(col))[0]);bounds[pos]=(interval[0]-canonical[int(col)],interval[1]-canonical[int(col)])
    # The solver fixes the historical yaw and unobserved external RF columns.
    for col in manifest['persistent_source_replay_identity']['fixed_parameter_indices']:
        pos=np.flatnonzero(selected==int(col))
        if len(pos):bounds[int(pos[0])]=(0.,0.)
    np.savez(c/'actual_shared_halfspace_LP_inputs.npz',matrix=A,lower=b,selected=selected,
             box_lower=np.asarray([(-np.inf if lo is None else lo)for lo,hi in bounds]),
             box_upper=np.asarray([(np.inf if hi is None else hi)for lo,hi in bounds]),labels=labels,q=canonical)
    scale=np.maximum(np.linalg.norm(A,axis=1),1e-15)
    def feasible(mask):
        result=linprog(np.zeros(len(selected)),A_ub=-A[mask]/scale[mask,None],b_ub=-b[mask]/scale[mask],bounds=bounds,method='highs')
        return dict(success=bool(result.success),status=int(result.status),message=result.message,
                    minimum_original_slack=None if not result.success else float(np.min(A[mask]@result.x-b[mask])),
                    iterations=int(result.nit))
    all_rows=np.ones(len(b),bool);checks={'all':feasible(all_rows)}
    for name in ('native','wide','confirmed','print','pivot'):
        checks['without_'+name]=feasible(labels!=name)
    # Read-only diagnostic of the largest common fraction of positive deficits.
    # This result is not applied to the solver or to any nonlinear guard.
    positive=np.maximum(b,0.);base=np.minimum(b,0.)
    progress=linprog(np.r_[np.zeros(len(selected)),-1.],
        A_ub=np.c_[-A/scale[:,None],positive/scale],b_ub=-base/scale,
        bounds=bounds+[(0.,1.)],method='highs')
    report=dict(schema='rinkspace.readonly-actual-shared-halfspace-LP-diagnosis.v1',
        original_q_sha256=hashlib.sha256(canonical.tobytes()).hexdigest(),
        actual_rows=len(b),actual_shared_columns=len(selected),actual_camera_reproduce_max_difference=camera_error,
        original_QP_constraints_or_nonlinear_guards_changed=False,full_motion_graph_loaded=False,
        LP_feasibility_checks=checks,diagnostic_only_positive_deficit_fraction=dict(success=bool(progress.success),
            status=int(progress.status),fraction=None if not progress.success else float(progress.x[-1]),
            minimum_original_slack=None if not progress.success else float(np.min(A@progress.x[:-1]-b)),
            applied_to_solver=False),seconds=time.perf_counter()-started)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
