"""Rebuild every public truth tangent against the saved real H/g, then replay."""
from pathlib import Path
from copy import deepcopy
import hashlib
import os,sys,time,json
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RINKSPACE_MOTION_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/diagnostics'),str(ROOT),str(ROOT/'src')]
import numpy as np
import audit_current_user_ice_truth as a
import z190_global_schur_linear_step as qp
from calibration.solver_runtime.priority_feasibility_witness_runtime import install_priority_feasibility_witness_runtime
install_priority_feasibility_witness_runtime()
from structured_programme_source_candidate.feasibility_priority_restoration import contract_original_support_bounds,actual_restoration_trial_decision
from structured_programme_source_candidate.human_stroke_truth import human_support_operator_contract
from calibration.solver_runtime.original_halfspace_primal import original_halfspace_primal

HERE=a.HERE
B=a.RUNS/'okada_review86_joint_curve_integration_20261010_2342'
Q='75719b65139fa662366146c4517312319bdad9a88e6f2c25704966f550cb1eee'
S=B/'accepted_states'/Q
CAP=a.RUNS/'required_QP_capture_q757_20261011_0100/required_reduced_QP'

def emit(**values):
    print(json.dumps(values,ensure_ascii=False,allow_nan=False),flush=True)

def face_map(family,factor,response,offset,caps):
    if family=='native':sizes=[2*len(row['uv'])for row in factor.full_stroke_records]
    elif family=='wide':sizes=[2*len(row['uv'])for row in factor.records]
    else:sizes=[item.get('carrier_size',len(item['row']['uv'])*(2 if item['normals'] is not None else 1))for item in factor.records]
    group=np.repeat(np.arange(len(sizes))+offset,sizes);added=np.repeat(caps,sizes)
    lg=group.copy();la=added.copy()
    if family=='confirmed':
        cursor=0
        for item,size in zip(factor.records,sizes):
            if not item.get('expand_lower_faces',item['normals']is not None):lg[cursor:cursor+size]=-1;la[cursor:cursor+size]=0.
            cursor+=size
    mask=np.asarray(response['retained_face_mask'],bool)
    assert mask.shape==(2*len(group),)
    return np.r_[lg,group][mask],np.r_[la,added][mask]

def main():
    started=time.perf_counter()
    config=ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_same757_original_support_rectangle_20261011.json'
    options=a.read(config)['source_options']
    state,q,identity,camera_error=a.load_state(S,Path(options['--world-init']),a.read(B/'input_manifest.json'))
    assert camera_error==0.;identity['_accepted_directory']=str(S)
    board=a.NativeBoardEdgeFactor(state,options['--additional-board-source'],len(q));ice=a.NativeIceCurveFactor(state,options['--additional-ice-source'],len(q))
    human=a.bind_human(state,board,ice,options,len(q));wide=a.wide_factor(state,identity,options,len(q),options['--practical-human-location-envelope'])
    old71=a.read(ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_resume_user_review71_20261009.json')['source_options']
    circle,material,placement=a.existing_LOW_source_owners(state,q,S,old71,a.read(S/'scene_UNACCEPTED.json'))
    confirmed=a.UserConfirmedSourceTruth(state,options['--user-confirmed-source-truth-authority'],board,ice,parameter_count=len(q),circle_factor=circle,material_factor=material,placement_source=placement)
    factors=dict(native=human,wide=wide,confirmed=confirmed)
    proof_path=HERE/'same_q757_original86_finite_rectangle_semantics_endpoint_face_final_actual_audit_20261011.json';proof=a.read(proof_path)
    caps={k:np.asarray(v['proposed_limits'],float)for k,v in proof['families'].items()}
    base=a.values(human,wide,confirmed,q)
    revised=deepcopy(proof);revised['operator_contract']=human_support_operator_contract()
    originalcaps=a.read(S/'truth_restoration_state.json')
    sameq={}
    for fam in caps:
        actual=np.asarray(base['buffers'][fam]);old=np.asarray(proof['families'][fam]['violations_px'])
        assert np.max(abs(actual-old))<=1e-10
        cap=np.minimum(np.asarray(originalcaps['families'][fam]['limits']),np.where(actual<=1e-8,0.,actual))
        assert np.array_equal(cap,caps[fam])
        sameq[fam]=dict(actual_violation_px=actual.tolist(),previous_violation_px=old.tolist(),
            max_guard_roundoff_change_px=float(np.max(abs(actual-old))),caps_px=cap.tolist(),
            uncertainties=originalcaps['families'][fam]['uncertainties'],group_ids=originalcaps['families'][fam]['group_ids'])
        row=revised['families'][fam]
        for key in ('violations_px','corrected_actual_same_q_violation_px'):row[key]=actual.tolist()
        for key in ('proposed_limits','proposed_ratchet_only_corrected_semantic_caps_px'):row[key]=cap.tolist()
        for item,value,c in zip(row['groups'],actual,cap):item.update(corrected_actual_px=float(value),proposed_new_cap_px=float(c))
    revised.update(guard_roundoff_only_after_exact_constant_axis_identity=True,
        all86_previous_caps_and_uncertainties_preserved=True,previous_certificate=a.ref(proof_path),
        actual_guard_values_and_caps_policy_bitwise_unchanged_after_derivative_only_revision=False)
    revisedproof=HERE/'same_q757_original86_rectangle_exact_constant_axes_actual_audit_20261011.json'
    a.save(revisedproof,revised)
    def signatures():
        result=[]
        for family,records in [('native_primary',human.records),('native_full',human.full_stroke_records),('wide',wide.records),('confirmed',confirmed.records)]:
            for ordinal,row in enumerate(records):
                result.append(dict(family=family,index=ordinal,fields={key:dict(shape=list(np.shape(row[key])),dtype=np.asarray(row[key]).dtype.str,
                    sha256=hashlib.sha256(np.ascontiguousarray(row[key]).tobytes()).hexdigest())
                    for key in ('uv','normals','bounds','tangent_bounds','original_raster_normal_bounds','original_source_tangent_bounds','radius')if key in row and row[key] is not None}))
        return result
    before_signatures=signatures()
    report=dict(schema='rinkspace.all-original-truth-faces-rebuilt-captured-source-QP.v1',complete_q_sha256=Q,
        complete_state_reference=a.ref(S/'complete_state_UNACCEPTED.npz'),original_QP_capture_reference=a.ref(CAP/'capture.json'),
        operator_contract=human_support_operator_contract(),current_same_q_old_truth_values_bitexact=all(np.array_equal(base['buffers'][f],np.asarray(proof['families'][f]['violations_px']))for f in factors),
        all_original_UV_support_sigma_roles_caps_unchanged=True,source_H_g_and_bounds_reused=True,
        no_complete_source_graph_or_J_recomputed=True,stages=[])
    history=a.read(a.RUNS/'okada_review86_original_support_rectangle_20261011_0043/solver/iterations.json')[0]
    saved=history['global_schur_linear_report'];cols=np.asarray(saved['global_parameter_indices'],int);scales=np.asarray(saved['global_parameter_scales'],float)
    evalcols=np.asarray(state.canonical_parameter_layout.canonical_to_evaluation[cols],int)
    with np.load(CAP/'original_problem.npz',allow_pickle=False)as z:
        H=z['matrix'];g=z['gradient'];lo=z['lower'];hi=z['upper'];oldA=z['constraint_matrix']
        oldoriginal=z['priority_original_constraint_lower'];oldowned=z['priority_row_group_index'];v=z['priority_current_group_violations']
    assert np.max(abs(np.concatenate([base['buffers'][f]for f in factors])-v))<=1e-10
    v=np.concatenate([base['buffers'][f]for f in factors])
    prefix=int(np.flatnonzero(oldowned>=0)[0]);parts=[oldA[:prefix].copy()];targets=[oldoriginal[:prefix].copy()];owned=[np.full(prefix,-1,int)]
    responses={};offset=0;cursor=prefix
    emit(stage='actual_all86_owner_loaded',seconds=time.perf_counter()-started,physical_prefix_rows=prefix)
    for family,factor in factors.items():
        t=time.perf_counter();kwargs=dict(group_limits=caps[family],return_step_carrier_response=True)
        if family=='native':kwargs.update(full_stroke=True,endpoint_inward_restoration=False)
        result=factor.constraints(q,evalcols,**kwargs);response=result['step_carrier_response']
        own,added=face_map(family,factor,response,offset,caps[family]);offset+=len(caps[family])
        A=result['matrix']*scales;original=result['lower']+added
        parts.append(A);targets.append(original);owned.append(own)
        responses[family]=dict(response,row_start=cursor,row_stop=cursor+len(original));cursor+=len(original)
        report['stages'].append(dict(family=family,rows=len(original),seconds=time.perf_counter()-t))
        emit(stage='actual_family_original_faces_rebuilt',**report['stages'][-1])
    A=np.vstack(parts);original=np.concatenate(targets);ownership=np.concatenate(owned)
    assert signatures()==before_signatures
    report.update(source_arrays_signatures=before_signatures,all86_same_q_actual=sameq,actual_same_q_semantic_certificate=a.ref(revisedproof),
        source_arrays_before_after_public_J_bitexact=True,old_rows=len(oldoriginal),rows=len(original),shared_columns=len(cols),
        raw_face_removal_is_exact_J_structural_zero_not_norm_threshold=True,
        original_source_support_arrays_and_group_roles_unchanged=True)
    del parts,oldA
    pending=CAP/'all_truth_constant_axes_original_domain.npz'
    np.savez(pending,matrix=H,gradient=g,lower=lo,upper=hi,constraint_matrix=A,
        original_constraint_lower=original,row_group_index=ownership,current_group_violations=v,
        global_parameter_indices=cols,global_parameter_scales=scales,canonical_to_evaluation=evalcols)
    preliminary=HERE/'q757_all86_constant_axes_A_precontraction_actual_20261011.json'
    a.save(preliminary,report)
    emit(stage='updated_all86_original_domain_saved',arrays=str(pending),old_rows=len(oldoriginal),new_rows=len(original))
    np.savez(CAP/'all_truth_constant_axes_carrier_maps.npz',**{family+'_'+key:np.asarray(value)
        for family,response in responses.items()for key,value in response.items()})
    np.savez(CAP/'all_truth_constant_axes_nominal_tokens.npz',**{family+'_'+str(index)+'_signature':result['query']['source_support_branch_signature']
        for family,results in [('native',base['full']),('wide',base['wide'])]for index,result in enumerate(results)
        if 'source_support_branch_signature' in result['query']})
    if '--audit-saved-direction' in sys.argv:
        with np.load(CAP/'per_row_zero_center_direction.npz',allow_pickle=False)as z:du=z['step']
        step=np.zeros_like(q);step[evalcols]=du*scales
        checks=[]
        def carriers(value):
            return dict(native=human._carriers(value,full_stroke=True),wide=wide._carriers(value),confirmed=confirmed._carriers(value)[0])
        for h in (1e-6,1e-7):
            plus,minus=carriers(q+h*step),carriers(q-h*step)
            for family,response in responses.items():
                actual=(plus[family]-minus[family])/(2*h)
                actual=np.r_[actual,-actual][response['retained_face_mask']]
                predicted=A[response['row_start']:response['row_stop']]@du
                error=abs(predicted-actual);i=int(np.argmax(error))
                checks.append(dict(family=family,h=h,max_actual_directional_error=float(error[i]),worst_family_face=i,
                    global_face=response['row_start']+i,predicted=float(predicted[i]),actual=float(actual[i]),
                    group=int(ownership[response['row_start']+i])))
        report.update(all86_actual_directional_checks=checks,seconds=time.perf_counter()-started,
            diagnostic_direction_reference=a.ref(CAP/'per_row_zero_center_direction.npz'),QP_solve_performed=False)
        output=HERE/'q757_all86_updated_A_actual_small_h_directional_audit_20261011.json';a.save(output,report)
        emit(stage='all86_original_updated_A_actual_directional_audit',report=a.ref(output),checks=checks,seconds=report['seconds'])
        return
    t=time.perf_counter();contracted=contract_original_support_bounds(lo,hi,A,original,ownership,v)
    report['contraction']=contracted['report'];target=contracted['lower'];fraction=contracted['common_fraction']
    emit(stage='updated_all86_original_LP_contraction',seconds=time.perf_counter()-t,common_fraction=fraction)
    np.savez(CAP/'all_truth_constant_axes_contracted_problem.npz',matrix=H,gradient=g,lower=lo,upper=hi,
        constraint_matrix=A,constraint_lower=target,original_constraint_lower=original,row_group_index=ownership,
        current_group_violations=v,global_parameter_indices=cols,global_parameter_scales=scales,canonical_to_evaluation=evalcols)
    t=time.perf_counter();du,held,linear=qp._linearly_constrained_shared_step(H,g,lo,hi,A,target)
    report['QP']=linear;report['QP_seconds']=time.perf_counter()-t
    finite_lo=np.flatnonzero(np.isfinite(lo));finite_hi=np.flatnonzero(np.isfinite(hi));eye=np.eye(len(du))
    rows=np.vstack((A,eye[finite_lo],-eye[finite_hi]));lower=np.r_[target,lo[finite_lo],-hi[finite_hi]]
    _,tol,bad,primal=original_halfspace_primal(rows,lower,du);assert primal['accepted']
    report.update(complete_original_primal=primal,source_quadratic_model_change=float(g@du+.5*du@H@du),
        scaled_step_norm=float(np.linalg.norm(du)),physical_step_norm=float(np.linalg.norm(du*scales)))
    np.savez(CAP/'all_truth_constant_axes_direction.npz',step=du,original_primal_tolerance=tol,common_fraction=fraction)
    step=np.zeros_like(q);step[evalcols]=du*scales
    def carriers(value):
        return dict(native=human._carriers(value,full_stroke=True),wide=wide._carriers(value),confirmed=confirmed._carriers(value)[0])
    diagnostics=[]
    for h in (1e-6,1e-7):
        plus,minus=carriers(q+h*step),carriers(q-h*step)
        for family,response in responses.items():
            actual=(plus[family]-minus[family])/(2*h)
            actual=np.r_[actual,-actual][response['retained_face_mask']]
            predicted=A[response['row_start']:response['row_stop']]@du
            error=abs(predicted-actual);i=int(np.argmax(error))
            diagnostics.append(dict(family=family,h=h,max_actual_directional_error=float(error[i]),worst_family_face=i,
                global_face=response['row_start']+i,predicted=float(predicted[i]),actual=float(actual[i]),
                group=int(ownership[response['row_start']+i])))
    report['all86_actual_directional_checks']=diagnostics
    emit(stage='updated_all86_required_QP_solved',QP_seconds=report['QP_seconds'],primal=primal,directional_checks=diagnostics)
    trials=[]
    for alpha in (1.,.5,.25,.125,.0625,.03125,.015625,.0078125,.00390625,.001953125,.0009765625,.00048828125,.000244140625):
        value=a.values(human,wide,confirmed,q+alpha*step);actual=np.concatenate([value['buffers'][f]for f in factors])
        decision=actual_restoration_trial_decision(v,actual,common_fraction=fraction,step_fraction=alpha,
            physical_guard_passed=True,previous_source_cost=0.,trial_source_cost=0.)
        summary=dict(alpha=alpha,decision=decision,families={f:dict(maximum_cap_excess_px=float(np.max(value['buffers'][f]-caps[f])),
            failed_group_indices=np.flatnonzero(value['buffers'][f]-caps[f]>1e-8).tolist(),violations_px=value['buffers'][f].tolist())for f in factors})
        trials.append(summary);emit(stage='updated_all86_actual_trial',alpha=alpha,accepted=decision['accepted'],
            families={f:{k:v for k,v in d.items()if k!='violations_px'}for f,d in summary['families'].items()})
        if decision['accepted']:break
    report.update(trials=trials,seconds=time.perf_counter()-started,nonlinear_truth_accepted=any(t['decision']['accepted']for t in trials))
    path=HERE/'q757_all86_constant_axes_A_same_source_Hg_actual_direction_replay_20261011.json';a.save(path,report)
    emit(stage='all86_original_updated_A_replay_complete',report=a.ref(path),seconds=report['seconds'],truth_accepted=report['nonlinear_truth_accepted'])

if __name__=='__main__':main()
