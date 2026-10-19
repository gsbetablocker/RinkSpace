"""Read only: saved complete-target costs, budget semantics and evidence gaps.

No factor evaluation, optimizer, source J, source frame or numeric cache.
"""
from pathlib import Path
import json, hashlib, math
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
RUN=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_review86_q276_vertical_print_station_one_cycle_20261011')
BOOT=Path(str(RUN)+'.bootstrap')
CONFIG=ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_q276_vertical_print_station_one_cycle_20261011.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))

def saved_partial_numeric_evidence(manifest,signatures,families):
    """Only existing small numeric buffers; verify full original wide bytes."""
    from scipy.sparse import csr_matrix,hstack
    direction_path=next((RUN/'actual_complete_source_directions').glob('*.npz'))
    with np.load(direction_path,allow_pickle=False)as z:
        step=z['step'];mapping=z['canonical_to_evaluation'];inverse=np.argsort(mapping)
    pp_path=RUN/'actual_principal_point_working_prior_initial.json';pp=read(pp_path)
    r=np.asarray(pp['actual_normalized_residual']);pp_d=np.asarray(pp['actual_J_values'])*step[np.asarray(pp['actual_J_columns'])]
    pp_before=float(.5*r@r);pp_after=float(.5*(r+.5*pp_d)@(r+.5*pp_d))
    assert abs(pp_before-families[32]['before'])<5e-14 and abs(pp_after-families[32]['after'])<5e-14
    contract=manifest['source_contract']['evaluation_source_contract']
    while 'numerical_quadrature_counts'not in contract:contract=contract['original_source_contract']
    counts=np.asarray(contract['numerical_quadrature_counts']);indices=np.repeat(np.arange(len(counts)),counts)
    wide_path=RUN/'actual_wide_linearization_diagnostic.npz';diag_path=RUN/'actual_wide_directional_diagnostic.json';diag=read(diag_path)
    digest=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    with np.load(wide_path,allow_pickle=False)as z:
        core=csr_matrix((z['core_J_data'],z['core_J_indices'],z['core_J_indptr']),shape=z['core_J_shape'])
        private=z['private_J'];r=z['residual'];private_indices=z['wide_private_indices']
        raw_direction=z['direction'];profile=str(z['frozen_profile_token'])
        assert str(z['source_manifest_sha256'])==sha(RUN/'input_manifest.json')
        np.testing.assert_array_equal(core@raw_direction[inverse[:core.shape[1]]]+private@raw_direction[private_indices],z['predicted_directional'])
    quad=np.r_[np.zeros(4342,bool),np.ones(len(r)-4342,bool)]
    assert digest(r[indices])==signatures[22]['r']['sha256']
    assert digest(quad[indices])==signatures[22]['quadratic']['sha256']
    full=hstack((core,csr_matrix((len(r),3)),csr_matrix(private),csr_matrix((len(r),10))),format='csr')[indices]
    full=full[:,mapping].tocsr();full.sort_indices()
    for key,a in [('J_data',full.data),('J_indices',full.indices),('J_indptr',full.indptr)]:assert digest(a)==signatures[22][key]['sha256']
    core_response=core@step[inverse[:core.shape[1]]];private_response=private@step[private_indices]
    response=core_response+private_response
    rho=lambda a:np.where(quad,.5*a*a,np.where(abs(a)<=2.,.5*a*a,2.*abs(a)-2.))
    gradient=np.where(quad,r,np.clip(r,-2.,2.))
    before=float(np.sum(counts*rho(r)));model_after=float(np.sum(counts*rho(r+.5*response)))
    assert abs(before-families[22]['before'])<5e-13
    parts=[]
    for record in diag['coupled']['per_family']:
        sl=slice(record['start'],record['stop'])
        parts.append(dict(family=record['family'],unique_scalar_start=record['start'],unique_scalar_stop=record['stop'],
            original_cost_before=float(np.sum(counts[sl]*rho(r)[sl])),
            original_cost_directional_derivative=float(np.sum(counts[sl]*gradient[sl]*response[sl])),
            shared_physical_core_cost_directional_derivative=float(np.sum(counts[sl]*gradient[sl]*core_response[sl])),
            private_wide_camera_cost_directional_derivative=float(np.sum(counts[sl]*gradient[sl]*private_response[sl])),
            saved_linear_residual_model_alpha_half_cost_change=float(np.sum(counts[sl]*(rho(r+.5*response)-rho(r))[sl]))))
    roll_path=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/current_all_programme_roll_prior_q276_q8a15_readonly_20261011.json')
    roll=read(roll_path);matches=[]
    for name in ('main1','main7'):
        replay=next(r for r in roll['per_block']if r['block']==name)
        live=next(r for r in read(RUN/('actual_trial_'+name+'_source_row_cost_pass_0001.json'))['actual_source_partition']if r['family']=='physical_roll_prior')
        errors=[abs(replay['prior_cost_before']-live['actual_before_cost']),abs(replay['prior_cost_after']-live['actual_after_cost']),abs(replay['prior_cost_change']-live['actual_cost_delta'])]
        assert max(errors)<5e-14 and replay['actual_exposures']==live['scalar_rows']
        matches.append(dict(block=name,original_scalar_rows=live['scalar_rows'],before_after_delta_absolute_errors=errors,
            exact_original_roll_theta_bias_sigma_operator_reproduced=True))
    return dict(direction=ref(direction_path),
        principal_point_6_rows=dict(receipt=ref(pp_path),original_J_reused=True,directional_response=pp_d.tolist(),
            original_gradient_dot_step=float(np.asarray(pp['actual_normalized_residual'])@pp_d),
            before_cost=pp_before,after_half_step_cost=pp_after,live_actual_after_reproduced=True),
        wide_existing_partial_J=dict(buffer=ref(wide_path),directional_receipt=ref(diag_path),stored_diagnostic_profile_token=profile,
            motion_profile_differed_but_entire_wide_r_quad_and_J_buffers_match_actual_fresh_initial_source_bytes=True,
            full_original_wide_initial_r_quadratic_J_data_indices_indptr_byte_identity_verified=True,
            source_derivatives_not_recomputed=True,numerical_quadrature_are_not_independent_observations=True,
            unique_rows=len(r),original_numeric_rows=len(indices),original_cost_before=before,
            original_gradient_dot_step=float(np.sum(counts*gradient*response)),
            saved_linear_residual_alpha_half_cost=model_after,live_actual_after=families[22]['after'],
            actual_minus_saved_linear_residual_model=float(families[22]['after']-model_after),per_unique_source_family=parts,
            source_code=[ref(ROOT/'calibration/solver_runtime/legacy'/p)for p in (
                'wide_anchor_candidate/complete_wide_free_optics_source.py','wide_anchor_candidate/wide_high_source_factor.py',
                'structured_programme_source_candidate/current_wide_source.py','structured_programme_source_candidate/paired_programme_chart.py')]),
        original_roll_prior_replay=dict(receipt=ref(roll_path),live_partition_checks=matches),
        partial_gradients_do_not_supply_missing_complete_H_g_or_source_QP_dual_forces=True)
def main():
    initial=read(RUN/'latest_initial_J_family_timings.json');trial=read(RUN/'actual_trial_family_cost_pass_0001.json')
    normal=read(RUN/'first_complete_source_normal.json');history=read(RUN/'solver/iterations.json')[0]
    manifest=read(RUN/'input_manifest.json');cfg=read(CONFIG);options=cfg['source_options']
    with np.load(RUN/'actual_initial_normal_state.npz',allow_pickle=False)as z:
        parent=hashlib.sha256(z['q'].tobytes()).hexdigest();profile=str(z['frozen_profile_token'])
        assert set(z.files)=={'q','frozen_profile_token','source_manifest_sha256'}
        assert str(z['source_manifest_sha256'])==sha(RUN/'input_manifest.json')
    assert parent==initial['q_sha256']=='2766454eeb8282632e6bf1226d0305ffcda70b74fbe7d916b2614b8672f8b308'
    assert profile==trial['frozen_profile_token'];assert trial['source_manifest_sha256']==sha(RUN/'input_manifest.json')
    assert initial['complete'] and trial['complete'];assert len(initial['families'])==len(trial['families'])==33
    families=[]
    for ordinal,(a,b)in enumerate(zip(initial['families'],trial['families'])):
        assert a['family']==b['original_row_ledger']['family'] and a['actual_rows']==b['scalar_rows']
        assert (a['original_start'],a['original_stop'])==(b['original_row_ledger']['original_start'],b['original_row_ledger']['original_stop'])
        families.append(dict(ordinal=ordinal,family=a['family'],scalar_rows=a['actual_rows'],
            before=a['actual_normalized_cost'],after=b['normalized_cost'],delta=b['normalized_cost']-a['actual_normalized_cost']))
    before=math.fsum(x['before']for x in families);after=math.fsum(x['after']for x in families)
    assert before==normal['full_cost']==history['cost'];assert after==trial['normalized_cost']==history['next_cost']
    comparison=read(BOOT/'complete_cycle_comparison.json');signatures=comparison['fresh_source_block_signatures']
    parts=[]
    for name,ordinal in [('main1',0),('main7',1)]:
        path=RUN/('actual_trial_'+name+'_source_row_cost_pass_0001.json');row=read(path)
        assert row['initial_complete_q_sha256']==parent and row['complete_q_sha256']==trial['complete_q_sha256']
        assert row['frozen_profile_token']==profile
        assert row['actual_baseline_r_sha256']==signatures[ordinal]['r']['sha256']
        assert row['actual_before_cost']==families[ordinal]['before'] and row['actual_after_cost']==families[ordinal]['after']
        parts.append(dict(family='programme:'+name,receipt=ref(path),
            partitions=row['actual_source_partition'],absolute_increase_examples=row['largest_actual_absolute_source_cost_increases']))
    domain=read(RUN/'actual_initial_shared_step_domain.json')
    kkt=history['global_schur_linear_report']['shared_linear_domain_step']
    bounded=history['global_schur_linear_report']['shared_parameter_bound_step']
    existing_normals=[]
    base=RUN.parent
    paths=list(base.glob('*/first_complete_source_normal_numeric.npz'))+list(base.glob('*/current_complete_linearization_cache/first_complete_source_normal_numeric.npz'))
    for path in paths:
        with np.load(path,allow_pickle=False)as z:
            qhash=hashlib.sha256(z['q'].tobytes()).hexdigest()
            existing_normals.append(dict(path=str(path),q_sha256=qhash,frozen_profile_token=str(z['frozen_profile_token']),
                same_q_as_this_parent=qhash==parent,H_and_g_not_loaded=True))
    assert not any(r['same_q_as_this_parent']for r in existing_normals)
    policy=read(options['--native-base-curve-integration-policy'])
    policy_by_id={r['source_row_id']:r for r in policy['records']}
    manual=[]
    wanted={'user_native230024192_B2_20261008','user_complete_yellow_F2_outer_native229989888',
        'user_20261010_F2_outer_230390528_branch0','user_20261010_F2_outer_230390528_branch1',
        'user_native230024192_ad20_right_vertical_20261008','user_complete_position_Ad4_right_red_native229766912'}
    for name in ['--additional-ice-source','--additional-board-source']:
        packet=read(options[name])
        for row in packet['rows']:
            if row['id']not in wanted:continue
            n=len(row['uv']);sigma=np.broadcast_to(row['sigma_px'],n)
            count_scale=n if row.get('practical_manual_location_uncertainty_native_px',0.)>0. else 1.
            registered=policy_by_id.get(row['id'],{})
            manual.append(dict(id=row['id'],actual_pts=row.get('actual_pts',row['pts']),UV_count=n,
                packet=ref(options[name]),centre_uses_original_exact_same_UV_as_native_base=True,
                stored_sigma_px_min=float(sigma.min()),stored_sigma_px_max=float(sigma.max()),
                extra_centre_residual_normalization=math.sqrt(count_scale),
                centre_nominal_quadratic_sum_inverse_sigma_squared_per_axis=float(np.sum(1/(sigma*sigma*count_scale))),
                base_declared_budget=registered.get('new_base_budget'),centre_declared_budget=registered.get('new_additional_centre_budget'),
                centre_removes_native_normal_optical_support=True,
                distinct_base_outside_support_vs_centre_raw_location_loss=True,
                repeated_pixels_are_not_an_additional_independent_measurement=True,
                source_sampling_declaration=row.get('source_sampling'),sigma_semantics=row.get('sigma_semantics')))
    b2_groups=[g for g in manifest['original_ice_curve_integration']['groups']if g['actual_pts']==230571776 and g['physical_curve_id']=='["line","B2"]']
    b2_rows=[g for g in manifest['original_ice_curve_integration']['source_rows']if g['actual_pts']==230571776 and g['id']=='B2']
    assert len(b2_groups)==len(b2_rows)==1
    sources=[ROOT/'calibration/solver_runtime'/p for p in ['source_loss.py','curve_source_grouping.py','original_board_curve_integration.py','original_ice_curve_integration.py']]
    sources+=[ROOT/'src/rinkspace/source_sampling.py',ROOT/'calibration/solver_runtime/legacy/structured_programme_source_candidate/human_stroke_truth.py',
        ROOT/'calibration/solver_runtime/legacy/snapshot/three_frame_refit/program_tracking/main7_source_bundle_candidate.py',
        ROOT/'calibration/solver_runtime/legacy/run_current_named66_full_programme.py']
    report=dict(schema='rinkspace.saved-target-cost-and-budget-causal-evidence.v1',script=ref(__file__),config=ref(CONFIG),
        no_solver_or_full_J_or_source_value_evaluation=True,new_numeric_cache_or_source_frames_written=0,
        parent_q_sha256=parent,trial_q_sha256=trial['complete_q_sha256'],frozen_profile_token=profile,
        cost_sources=[ref(RUN/'latest_initial_J_family_timings.json'),ref(RUN/'actual_trial_family_cost_pass_0001.json')],
        all_33_live_costs_authenticated_on_same_parent_trial_profile_and_source_inventory=True,
        full_source_before=before,full_source_after=after,full_source_decrease=before-after,actual_families=families,
        programme_decrease_share=-(math.fsum(x['delta']for x in families[:6]))/(before-after),
        low_block_decrease_share=-families[3]['delta']/(before-after),
        main1_main7_live_actual_partitions=parts,
        actual_cost_changes_are_not_per_family_J_predictions_or_gradient_forces=True,
        normalization_semantics=dict(code=[ref(p)for p in sources],
            objective='sum integration_weight * original component Huber2 or quadratic((raw_source_error/stored_sigma)*explicit_prior_sampling_multiplier)',
            robust_and_prior_weights_are_engineering_model_choices_not_measured_covariance=True,
            original_motion='sigma=max(2,sqrt(original pair N)/3); per-axis nominal quadratic N/sigma^2=min(N/4,9)',
            motion_after_compaction='Keep original sigma; removed or masked points add no information and do not renormalize surviving samples back to9',
            motion_after_profile_and_Huber='Frozen native depth/profile and robust coefficients can reduce or redistribute nominal sensitivity; budget9 is not actual measured Fisher information',
            original_absolute_curves='One budget1 per source video, actual exposure and physical curve within its adapter; exact repeated operator+UV anchors share mass',
            material_corners='Original independently registered material-corner rows are not converted to a curve budget by the board/ice curve adapters',
            repeated_curve_sampling='Measured quadrature and same-anchor copy sharing prevent same-curve sample-count inflation; different exposures, physical curves and operators are retained',
            statistical_independence_claimed=False,cross_pair_temporal_source_covariance_implemented=False,
            base_plus_manual_centre='Same user UV contributes its native outside-support term and deliberate raw-centre secondary term; centre remains separately mean normalized, with no new measurement precision claim',
            implemented_double_budget_error_demonstrated=False,weight_change_justified_by_cost_share=False,
            original_B2_230571776=dict(group=b2_groups[0],row=b2_rows[0],nominal_effective_per_axis_precision=1/9.,old_unintegrated_23_point_precision=23/9.),
            selected_manual_trace_budget_receipts=manual),
        hard_constraint_evidence=dict(receipt=ref(RUN/'actual_initial_shared_step_domain.json'),
            support_limits_are_not_zero_raw_absolute_centre_errors=True,
            hard_source_maximum_outside_support_is_distinct_from_correlated_centre_soft_objective=True,
            phase_I_minimum_common_fraction=domain['report']['truth_feasibility_priority']['phase_I_minimum_common_original_violation_fraction'],
            demanded_common_fraction=domain['report']['truth_feasibility_priority']['demanded_common_original_violation_fraction'],
            original_guard_or_user_working_interval_relaxed=False,
            active_mount_original_parameter_indices=bounded['active_original_parameter_indices'],
            active_mount_bound_sides=bounded['active_bound_sides'],
            finite_working_intervals=bounded['explicit_step_intervals'],
            active_linear_row_ids=kkt['active_linear_constraint_rows'],
            KKT_certificate_values={k:kkt[k]for k in ('free_stationarity_max_absolute','minimum_active_multiplier','original_unit_complementarity_absolute_sum','numerical_solver_status')},
            actual_original_dual_vector_saved=False,active_constraint_source_force_ranking_not_available=True,
            current_priority_LP_minimum_not_a_global_nonlinear_incompatibility_certificate=True),
        reusable_and_missing=dict(current_complete_states_direction_profile_support_costs_and_source_signatures_reusable=True,
            current_q276_full_J_numeric_saved=False,current_q276_H_g_numeric_saved=False,current_q8a15_J_H_g_saved=False,
            current_full_face_matrix_carrier_map_and_dual_vector_saved=False,
            older_numeric_normal_identities=existing_normals,older_normal_not_reusable_as_q276_derivative=True,
            global_quadratic_decrease_reported_but_not_source_decomposed=True,
            missing_family_predictions_or_active_duals_not_inferred_from_json_certificates=True),
        authenticated_partial_saved_numeric_evidence=saved_partial_numeric_evidence(manifest,signatures,families),
        minimal_executable_analysis_plan=[
            'Use existing actual33 cost delta and main1/main7 partitions as observed target tradeoff; join to entity raw and outside-support direction response audit without claiming all absolute entities worsened.',
            'On exactly q276+savedD and unchanged frozen profile, evaluate original selected source-block values at small positive epsilon and2epsilon; compute directional residual response and per-block cost slopes without constructing J, normal, optimizer or disk arrays.',
            'For existing saved geometry-only and camera-only direction pieces, report entity raw-location response and original physical/86 guard validity; identify reinforcement, cancellation or prohibited component motion.',
            'Combine cost slope signs with raw entity improvement signs: objective-tangent opposition, within-block tradeoff, geometric/camera cancellation and hard working-domain bottleneck are different hypotheses.',
            'Formal active-dual force ranking or same-q constrained counterfactual requires current numericalH/g, exact face matrix and original-unit multipliers; those were not saved, so it is unavailable under the no-new-fullJ/no-solve instruction.'],
        no_weight_change_or_new_hard_guard_requested_or_applied=True,
        timing_field_clarification=dict(source_J_and_normal_seconds=normal['source_J_and_normal_seconds'],
            startup_elapsed_to_normal_seconds=normal['seconds_since_start'],optimizer_cycle_seconds=read(RUN/'solver/solver_result.json')['seconds'],
            older_execution_receipt_cold_preflight_fresh_normal_seconds_was_misnamed=True,
            prior_fixed_receipt=ref(BOOT/'vertical_print_station_one_cycle_execution_evidence.json')))
    dest=BOOT/'saved_complete_target_cost_budget_causal_evidence.json'
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,programme_decrease_share=report['programme_decrease_share'])))
if __name__=='__main__':main()
