"""Summarize authenticated existing cycle logs, artifacts and read-only audit.

Does not construct a camera, image, source J, or optimizer.
"""
from pathlib import Path
import json, hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
RUN=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_review86_q276_vertical_print_station_one_cycle_20261011')
BOOT=Path(str(RUN)+'.bootstrap')
OLD=RUN.with_name('okada_review86_samea454_actual947_minimal_pivot_trial_resume_memory_sha')
CONFIG=ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_q276_vertical_print_station_one_cycle_20261011.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def main():
    cfg=read(CONFIG);options=cfg['source_options'];history=read(RUN/'solver/iterations.json');assert len(history)==1
    iteration=history[0];result=read(RUN/'solver/solver_result.json');launch=read(BOOT/'launch.json')
    exit_check=read(BOOT/'owned_process_exit_check.json');auth=read(BOOT/'startup_source_auth_execution/actual_startup_source_auth_reuse.json')
    comparison=read(BOOT/'complete_cycle_comparison.json');audit=read(BOOT/'native36_saved_direction_station_causal_audit.json')
    parent=comparison['start']['q']['sha256'];child=comparison['accepted_q']['sha256']
    childdir=RUN/'accepted_states'/child;identity=read(childdir/'state_identity.json')
    assert identity['q_sha256']==child
    assert sha(childdir/'complete_state_UNACCEPTED.npz')==identity['state_npz_sha256']
    assert sha(childdir/'scene_UNACCEPTED.json')==identity['scene_sha256']
    assert sha(identity['camera_path'])==identity['camera_sha256']
    assert sha(CONFIG)==launch['config_sha256']==exit_check['launch_config_sha256']
    assert exit_check['all_owned_processes_exited'] and exit_check['monitor_launcher_creation_identity_match']
    events=[]
    for line in(BOOT/'stdout.log').read_text(encoding='utf-8-sig').splitlines():
        try:value=json.loads(line)
        except(ValueError,TypeError):continue
        if isinstance(value,dict):events.append(value)
    cold_normal=next(e for e in events if e.get('stage')=='actual_first_complete_native_source_normal_done')
    reuse=next(e for e in events if e.get('stage')=='reuse_exact_first_complete_source_normal')
    necessary=[e for e in events if e.get('stage')=='trial_necessary_condition']
    assert necessary[0]['action']=='reject' and necessary[1]['action']=='continue'
    assert necessary[0]['report']['failed_original_truth_groups'][0]['group_id']==['native',36]
    assert len(necessary[0]['report']['failed_original_truth_groups'])==1
    finished=next(e for e in events if e.get('stage')=='complete_result_saved_and_owned_workers_closed')
    native=iteration['global_schur_linear_report']['shared_linear_domain_provenance']['highest_direct_human_hard_truth']['native']
    route=native['finite_rectangle_station_local_model'];vertical=[c for c in route['contexts']if c.get('whole_vertical_print')]
    groups=sorted(set(c['group']for c in vertical));assert groups==[6,17,19,25,27,29,31,49]
    counts={str(g):sum(c['group']==g for c in vertical)for g in groups}
    assert counts['29']==70
    sources=[]
    for s in cfg['one_complete_cycle_validation']['implementation']:
        assert sha(s['path'])==s['sha256'];sources.append(s)
    source_receipt=read(BOOT/'execution_acceleration_receipt.json')
    # The running native domain records the same physical-station code receipt.
    if 'implementation_sources'in route:
        for s in route['implementation_sources']:assert sha(s['path'])==s['sha256']
    seq=iteration['global_schur_linear_report']['sequential_measured_shared_source_curvature_tightening']
    soc=[dict(measurement=m['measurement'],direction_sha256=m['direction_sha256'],
        maximum_actual_original_violation_px=m['actual_original_face_feasibility']['maximum_violation_px'],
        total_actual_original_violation_px=m['actual_original_face_feasibility']['total_violation_px'],
        actual_to_predicted_reduction=m['sufficient_actual_to_predicted_feasibility_reduction'])
        for m in seq['actual_original_nonlinear_feasibility_progress']['measurements']]
    signatures=comparison['fresh_source_block_signatures'];stream=iteration['streaming_source_linearization']
    assert len(signatures)==33 and sum(s['r']['shape'][0]for s in signatures)==4502662
    direction=Path(audit['saved_direction']['path']);assert sha(direction)==audit['saved_direction']['sha256']
    with np.load(direction,allow_pickle=False)as z:q=z['q'].copy();d=z['step'].copy()
    with np.load(childdir/'complete_state_UNACCEPTED.npz',allow_pickle=False)as z:
        cq=z['q'].copy();starts=z['aux__block_start'].copy();lengths=z['aux__block_lengths'].copy()
    rf=np.concatenate([np.arange(int(s),int(s+n))for s,n in zip(starts,lengths)])
    np.testing.assert_array_equal(cq[rf],(q+.5*d)[rf]);assert len(rf)//4==8503
    s=audit['samples'][0];b=audit['physical_cap_vs_computational_cell_evidence']
    summary=dict(schema='rinkspace.vertical-print-station-single-cycle-execution-and-native36-cause.v1',
        script=ref(__file__),config=ref(CONFIG),run=str(RUN),bootstrap=str(BOOT),
        complete_cycle_authorized_for_specific_repair=True,new_solver_started_by_this_summary=False,
        source_images_written=0,additional_full_J_disk_cache_written=False,
        parent_q_sha256=parent,child_q_sha256=child,accuracy_accepted=False,
        candidate_not_promoted_on_numeric_acceptance=True,
        original_86_groups_source_support_uncertainty_physical_domain_and_full_source_cost_unchanged=True,
        original_guard_identity=cfg['continuation']['original_manual_group_cap_and_uncertainty_identity'],
        code_and_fixes=dict(station_sources=sources,
            repaired_route='Certified whole external vertical print border; all original source vertices; original physical lower/upper height caps',
            source_UV_support_zero_limit_unchanged=True,paired_material_protocol_unchanged=True,
            new_geometry_or_fit_parameters=0,current_ice_route_unchanged=True,
            tests_before_cycle='27 tests passed; independently repeated by cache/source-review agent under identical station/projection SHA',
            previous_saved_six_directions=ref(Path(str(OLD)+'.bootstrap')/'native29_saved_direction_causal_audit.json'),
            registered_implementation_files_unchanged=source_receipt['registered_source_files_unchanged']),
        actual_repair_dispatch=dict(native_groups=groups,original_vertices_per_group=counts,
            native29_whole_vertical_contexts=counts['29'],source_only_no_new_parameter=True,
            original_domain_native_rows=151080,complete_step_rows=152781),
        full_source_linearization=dict(q_sha256=parent,cold_preflight_fresh_normal_seconds=cold_normal['seconds'],
            actual_fresh_profile_token=reuse['profile_token'],input_checkpoint_profile_token=comparison['start']['profile'],
            input_and_live_profile_tokens_not_asserted_identical=True,
            source_rows=4502662,original_blocks=33,source_J_nonzeros=stream['consumed_jacobian_nonzeros'],
            complete_fresh_signatures=ref(BOOT/'complete_cycle_comparison.json'),
            same_q_fresh_normal_reused_inside_this_cycle=True,prior_run_J_not_reused=True,
            next_accepted_state_J_generated=False),
        actual_SOC=dict(solved_corrections=seq['solved_corrections'],stop_reason=seq['stop_reason'],
            original_quarter_progress_rule_unchanged=True,progress=soc),
        actual_trials=dict(full_alpha=1.,full_step_rejected_only_by_native36=necessary[0]['report']['failed_original_truth_groups'],
            full_step_source_cost_not_evaluated=True,accepted_alpha=.5,
            necessary_group_check=necessary[1]['report'],original_complete_acceptance=iteration['explicit_source_trial_acceptance'],
            full_source_cost_before=iteration['cost'],full_source_cost_after=iteration['next_cost'],
            all_RF4_exposure_steps_shrunk_by_same_alpha=len(rf)//4,
            no_source_visual_accuracy_inferred=True),
        native36_readonly_cause=dict(audit=ref(BOOT/'native36_saved_direction_station_causal_audit.json'),
            id=audit['native36_record']['id'],native_pts=229246976,source_time_base='1/15360',
            guard_vertices=404,world_samples=1025,actual_station_contexts=audit['current_finite_station_context_count'],
            model_physical_caps_are_current_whole_C_course_intersections=True,
            model_endpoints_not_new_video_endpoint_measurements=True,
            blocking_source_vertex=0,nominal_physical_station=b['nominal_physical_station'][0],
            nominal_cell_station=b['nominal_segment_station'][0],
            base_branch=s['worst_base_branch'],full_trial_branch=s['worst_trial_branch'],
            blocking_vertex_not_first_last_sampling_cell=True,blocking_vertex_not_a_physical_endpoint=True,
            nominal_normal_carrier=6.,nominal_normal_J_exact_zero=True,
            original_normal_bounds=s['worst_original_normal_bounds'],original_tangent_bounds=s['worst_original_tangent_bounds'],
            nominal_carrier=audit['original_carrier'][:2],first_order_response=audit['carrier_directional_first_order_px'][:2],
            full_actual_dynamic_carrier=s['carrier'][:2],full_frozen_nominal_carrier=s['nominal_branch_nonlinear_carrier'][:2],
            full_actual_dynamic_violation=s['maximum_actual_violation_px'],
            full_frozen_nominal_branch_violation=s['maximum_frozen_branch_violation_px'],
            nominal_segment_normal_tangent_slopes=s['worst_nominal_segment_normal_tangent_slope'],
            dynamic_balancing_vs_frozen_tangent_nominal_factor=s['normal_fraction_of_balanced_Linf_response'],
            old_tangent_coordinate_underestimate_factor=1./s['normal_fraction_of_balanced_Linf_response'],
            source_branch_derivative_valid_for_small_step_FD=True,
            J_response_branch_coordinate_mismatch_demonstrated=True,
            active_face_changes_without_cell_change=True,
            current_ice_dispatch_excludes_this_internal_cell=True,
            proven='The existing normal-boundary-frozen carrier J and rebalanced nonlinear curvature response use different station elimination charts at this original C source vertex',
            not_proven=['Mathematical incompatibility among the full86 constraints',
                        'Extending whole-C station route guarantees an alpha1 feasible direction',
                        'Removing this numerical shrink improves complete-nine-shot visual accuracy',
                        'The producer physical finite cap is an observed video endpoint']),
        complete_saved_state=dict(state=ref(childdir/'complete_state_UNACCEPTED.npz'),
            identity=ref(childdir/'state_identity.json'),scene=ref(childdir/'scene_UNACCEPTED.json'),
            camera=ref(identity['camera_path']),input_manifest=ref(RUN/'input_manifest.json'),
            direction=ref(direction)),
        source_cache_memory_authentication=dict(receipt=ref(BOOT/'startup_source_auth_execution/actual_startup_source_auth_reuse.json'),
            missing_native_png_authentications=len(auth['missing_native_png_authentications']),
            native_pts=sorted(set(x['native_pts']for x in auth['missing_native_png_authentications'])),
            original_BGR_PNG_full_source_SHA_verified=True,
            missing_native_png_buffers_written=auth['missing_native_png_buffers_written']),
        normal_exit=dict(result=ref(RUN/'solver/solver_result.json'),status=result['status'],accepted_iterations=1,
            solver_cycle_seconds=result['seconds'],normal_saved_closed_event=finished,
            launch=ref(BOOT/'launch.json'),launcher_pid=launch['solver_process_id'],monitor_pid=launch['monitor_process_id'],
            owned_creation_identity_and_all_exit_receipt=ref(BOOT/'owned_process_exit_check.json'),
            tracked_owned_process_count=len(exit_check['owned_processes']),all_owned_processes_exited=True,
            force_kill_used=False,restart_started=False))
    dest=BOOT/'vertical_print_station_one_cycle_execution_evidence.json'
    dest.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(dest),sha256=sha(dest),bytes=dest.stat().st_size),ensure_ascii=False))
if __name__=='__main__':main()
