"""Replay the saved real complete normal and original supports; no source J."""
from pathlib import Path
import argparse,os,sys,time,json,hashlib
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RINKSPACE_MOTION_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/diagnostics'),str(ROOT),str(ROOT/'src')]
import audit_current_user_ice_truth as a
import numpy as np
from scipy.sparse import csc_matrix
import z190_joint_source_solver as joint
from structured_programme_source_candidate.source_step_curvature_tightening import measured_outward_step_tightening
from calibration.solver_runtime.optional_soc_runtime import install_optional_SOC_numerical_failure_route

RUN=a.RUNS/'okada_review86_TOP_RGB36_fixed285_20261011_034757'
CAP=a.RUNS/'required_QP_capture_q757_20261011_0100/required_reduced_QP'
OUT=RUN/'saved_normal_complete_SOC_replay_after_public_polish'
Q='75719b65139fa662366146c4517312319bdad9a88e6f2c25704966f550cb1eee'
SEED=a.RUNS/'okada_review86_joint_curve_integration_20261010_2342/accepted_states'/Q
CONFIG=ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_same757_board185_constant_axes_20261011.json'
def emit(**v):print(json.dumps(v,ensure_ascii=False,allow_nan=False),flush=True)
def digest(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--actual-face-progress-only',action='store_true',
        help='Re-evaluate existing complete direction carriers only; no J or QP')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output is not None:OUT=args.output
    started=time.perf_counter();OUT.mkdir(exist_ok=False)
    history=a.read(RUN/'solver/iterations.json')[0];linear_report=history['global_schur_linear_report']
    selected=np.asarray(linear_report['global_parameter_indices'],int)
    with np.load(RUN/'first_complete_source_normal_numeric.npz',allow_pickle=False)as z:
        q=z['q'].copy();H=csc_matrix((z['normal_data'].copy(),z['normal_indices'].copy(),z['normal_indptr'].copy()),shape=tuple(z['normal_shape']))
        g=z['gradient'].copy();source_report=json.loads(str(z['report_json']));profile=str(z['frozen_profile_token'])
    assert digest(q)==Q;fixed=np.asarray(source_report['fixed_parameter_indices'],int)
    assert 2 in fixed and not np.isin(selected,fixed).any()
    with np.load(CAP/'all_truth_constant_axes_original_domain.npz',allow_pickle=False)as z:
        requested=z['global_parameter_indices'];free=np.isin(requested,selected)
        np.testing.assert_array_equal(requested[free],selected)
        assert np.count_nonzero(z['constraint_matrix'][:,~free])==0
        A=z['constraint_matrix'][:,free].copy();b=z['original_constraint_lower'].copy()
        own=z['row_group_index'].copy();v=z['current_group_violations'].copy()
    provenance=a.read(RUN/'actual_initial_shared_step_domain.json')['report']['truth_feasibility_priority']
    target=float(provenance['demanded_common_original_violation_fraction'])
    np.testing.assert_array_equal(v,provenance['current_original_group_violations_px'])
    coeff=np.zeros(len(b));owned=own>=0;coeff[owned]=np.where(v[own[owned]]>1e-8,v[own[owned]],0.)
    lower=b-target*coeff
    lo=np.full(len(selected),-np.inf);hi=np.full(len(selected),np.inf)
    for entry in linear_report['shared_parameter_bound_step']['explicit_step_intervals']:
        i=int(np.flatnonzero(selected==entry['original_parameter_index'])[0])
        if entry['lower']is not None:lo[i]=entry['lower']
        if entry['upper']is not None:hi[i]=entry['upper']
    boundary=linear_report['shared_parameter_bound_step']['positive_domain_tangent_original_parameter_indices']
    responses={}
    with np.load(CAP/'all_truth_constant_axes_carrier_maps.npz',allow_pickle=False)as z:
        for family in ('native','wide','confirmed'):
            responses[family]={k[len(family)+1:]:z[k].copy()for k in z.files if k.startswith(family+'_')}
            for key in ('row_start','row_stop'):responses[family][key]=int(responses[family][key])
    options=a.read(CONFIG)['source_options']
    state,evalq,identity,camera_error=a.load_state(SEED,Path(options['--world-init']),a.read(SEED.parent.parent/'input_manifest.json'))
    assert camera_error==0.;identity['_accepted_directory']=str(SEED)
    np.testing.assert_array_equal(evalq,state.evaluation_q(q))
    board=a.NativeBoardEdgeFactor(state,options['--additional-board-source'],len(q));ice=a.NativeIceCurveFactor(state,options['--additional-ice-source'],len(q))
    human=a.bind_human(state,board,ice,options,len(q));wide=a.wide_factor(state,identity,options,len(q),options['--practical-human-location-envelope'])
    old71=a.read(ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_resume_user_review71_20261009.json')['source_options']
    circle,material,placement=a.existing_LOW_source_owners(state,evalq,SEED,old71,a.read(SEED/'scene_UNACCEPTED.json'))
    confirmed=a.UserConfirmedSourceTruth(state,options['--user-confirmed-source-truth-authority'],board,ice,parameter_count=len(q),circle_factor=circle,material_factor=material,placement_source=placement)
    def carriers(value):
        evaluated=state.evaluation_q(value)
        return dict(native=human._carriers(evaluated,full_stroke=True),wide=wide._carriers(evaluated),confirmed=confirmed._carriers(evaluated)[0])
    base=carriers(q)
    for family,response in responses.items():np.testing.assert_array_equal(base[family],response['nominal_carrier'])
    emit(stage='saved_complete_normal_and_actual86_supports_ready',seconds=time.perf_counter()-started,
        parameters=len(q),shared_columns=len(selected),domain_rows=len(lower),source_J_recomputed=False)
    if args.actual_face_progress_only:
        from calibration.solver_runtime.optional_soc_runtime import original_nonlinear_step_face_feasibility
        original_run=RUN/'saved_normal_complete_SOC_replay_after_public_polish'
        with np.load(original_run/'actual_required_complete_step_and_trial.npz',allow_pickle=False)as z:
            np.testing.assert_array_equal(z['parent_q'],q);step=z['step'].copy()
        actual=carriers(q+step)
        begun=time.perf_counter()
        merit=original_nonlinear_step_face_feasibility(A,lower,step[selected],responses,actual)
        measurement_seconds=time.perf_counter()-begun
        correction=measured_outward_step_tightening(A,lower,step[selected],responses,actual)
        details=[] if correction is None else correction['report']['compensated_faces']
        for family,family_merit in merit['families'].items():
            values=[r['actual_original_violation_px']for r in details if r['family']==family]
            assert family_merit['maximum_violation_px']>=max(values,default=0.)
            if values:assert family_merit['maximum_violation_px']==max(values)
        a.save(OUT/'actual_original_nonlinear_face_progress_saved_fullstep_replay.json',dict(
            schema='rinkspace.actual-original-nonlinear-face-progress-saved-complete-step-replay.v1',
            complete_q_sha256=Q,normal_reference=a.ref(RUN/'first_complete_source_normal_numeric.npz'),
            fullstep_reference=a.ref(original_run/'actual_required_complete_step_and_trial.npz'),
            actual_original_face_feasibility=merit,
            original_quadratic_model_decrease=float(-g@step-.5*step@(H@step)),
            actual_existing_compensated_face_count=len(details),
            original_helper_maximum_violations_match_actual_merit_exactly=True,
            additional_merit_compute_seconds=measurement_seconds,
            source_J_or_QP_recomputed=False,extra_source_evaluations_for_the_merit=0,
            production_full_physical_cost_acceptance_or_iter4_merit_not_claimed=True,
            seconds=time.perf_counter()-started))
        emit(stage='saved_actual_fullstep_face_progress_replay_complete',
            actual_original_face_feasibility=merit,measurement_seconds=measurement_seconds,
            seconds=time.perf_counter()-started)
        return
    os.environ['RINKSPACE_OPTIONAL_SOC_CAPTURE_DIR']=str(OUT)
    install_optional_SOC_numerical_failure_route(parallel_backend='faer')
    import z190_global_schur_linear_step as qp
    original=qp._linearly_constrained_shared_step
    def capture(H,g,lo,hi,A,b):
        if not(OUT/'actual_required_reduced_QP.npz').exists():
            np.savez(OUT/'actual_required_reduced_QP.npz',matrix=H,gradient=g,lower=lo,upper=hi,constraint_matrix=A,constraint_lower=b)
        return original(H,g,lo,hi,A,b)
    qp._linearly_constrained_shared_step=capture
    diagonal=np.maximum(H.diagonal(),1.)
    linear_options=dict(global_count=len(selected),global_parameter_indices=selected,damping=history['damping'],
        schur_view_lm_ratio=linear_report['schur_view_lm_ratio'],nonnegative_global_step_indices=boundary,
        global_step_lower_bounds=lo,global_step_upper_bounds=hi,global_step_constraint_matrix=A,global_step_constraint_lower=lower)
    t=time.perf_counter();linear=joint.schur_trial_linear_step(H,g,diagonal,**linear_options)
    np.testing.assert_array_equal(linear.step[fixed],0.)
    np.savez(OUT/'actual_required_complete_step_and_trial.npz',parent_q=q,step=linear.step,
             full_trial_q=q+linear.step,evaluation_parent_q=evalq,evaluation_trial_q=state.evaluation_q(q+linear.step),
             canonical_to_evaluation=state.canonical_parameter_layout.canonical_to_evaluation,shared_indices=selected)
    a.save(OUT/'actual_required_complete_step_report.json',linear.report)
    original_trial=history['global_schur_trial_records'][0]['trial_feasibility_recovery']['complete_state_recovery_identity']['proposed_q_sha256']
    emit(stage='actual_saved_complete_direction_reconstructed',seconds=time.perf_counter()-t,
        full_trial_q_sha256=digest(q+linear.step),production_full_trial_q_sha256=original_trial,
        full_original_KKT=linear.report['shared_linear_domain_step']['complete_original_primal'])
    domain=dict(matrix=A,lower=lower,curvature_responses=responses)
    def tightening(value,step,columns,domain):
        actual=carriers(value+step)
        try:return measured_outward_step_tightening(A,lower,step[columns],responses,actual)
        finally:carriers(value)
    corrected=joint._sequential_measured_shared_source_curvature(q,linear,selected,domain,linear_options,H,g,diagonal,
        tightening,joint.schur_trial_linear_step)
    np.testing.assert_array_equal(corrected.step[fixed],0.)
    np.savez(OUT/'post_polish_complete_step_and_trial.npz',parent_q=q,step=corrected.step,full_trial_q=q+corrected.step,
        evaluation_parent_q=evalq,evaluation_trial_q=state.evaluation_q(q+corrected.step),
        canonical_to_evaluation=state.canonical_parameter_layout.canonical_to_evaluation,shared_indices=selected)
    a.save(OUT/'post_polish_complete_step_report.json',corrected.report)
    from structured_programme_source_candidate.feasibility_priority_restoration import actual_restoration_trial_decision
    trials=[]
    for alpha in (1.,.5,.25,.125,.0625,.03125,.015625,.0078125,.00390625,.001953125,.0009765625,.00048828125):
        actual=a.values(human,wide,confirmed,state.evaluation_q(q+alpha*corrected.step))
        actualv=np.concatenate([actual['buffers'][f]for f in ('native','wide','confirmed')])
        decision=actual_restoration_trial_decision(v,actualv,common_fraction=target,step_fraction=alpha,
            physical_guard_passed=True,previous_source_cost=0.,trial_source_cost=0.)
        result=dict(alpha=alpha,actual_group_violations_px=actualv.tolist(),actual_truth_predicate=decision,
                    native35_violation_px=float(actual['buffers']['native'][35]))
        trials.append(result);emit(stage='post_polish_actual_complete86_truth_trial',alpha=alpha,
            native35=result['native35_violation_px'],truth_predicate=decision['accepted'])
        if decision['accepted']:break
    report=dict(schema='rinkspace.saved-complete-source-normal-actual-fullstep-SOC-post-polish-replay.v1',
        complete_q_sha256=Q,normal_reference=a.ref(RUN/'first_complete_source_normal_numeric.npz'),
        production_failure_report_reference=a.ref(RUN/'solver/iterations.json'),frozen_profile_token=profile,
        no_source_J_fullgraph_or_cost_recomputed=True,source_supports_and_sigma_unchanged=True,
        physical_recovery_not_reapplied_here_production_recovery_was_identity=True,
        full_parameters=len(q),shared_parameters=len(selected),complete_original_halfspace_rows=len(lower)+18,
        required_fullstep_report=a.ref(OUT/'actual_required_complete_step_report.json'),
        corrected_fullstep_report=a.ref(OUT/'post_polish_complete_step_report.json'),
        actual_complete_86_truth_trials=trials,full_physical_and_source_cost_acceptance_not_claimed=True,
        accuracy_accepted=False,seconds=time.perf_counter()-started)
    a.save(OUT/'actual_fullstep_SOC_post_polish_replay.json',report)
    emit(stage='actual_saved_normal_complete_SOC_replay_finished',report=a.ref(OUT/'actual_fullstep_SOC_post_polish_replay.json'),seconds=report['seconds'])

if __name__=='__main__':main()
