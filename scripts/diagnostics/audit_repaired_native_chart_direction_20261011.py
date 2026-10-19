"""Read-only saved-direction checks of mixed native charts and true guards.

Only seven small truth carrier Jacobians are built. No source objective/J,
fitting, frame decoding or image is constructed.
"""
from pathlib import Path
import copy
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_repaired_native_station_values_20261011 as values
common, np = values.common, values.np
from calibration.solver_runtime import finite_rectangle_station_step_constraints as station
from calibration.solver_runtime.native_truth_nominal_chart import (
    capture_native_nominal_query_chart, native_nominal_query_carriers)
from structured_programme_source_candidate.human_constraint_dependency_fd import group_dependencies
from structured_programme_source_candidate.source_step_curvature_tightening import (
    actual_carrier_step_response, measured_outward_step_tightening)

OUT = values.OUT.with_name('repaired_native_mixed_chart_saved_direction_audit_20261011.json')


def main():
    began = time.perf_counter()
    human, eq, ed, accepted, caps, refs = values.load_saved()
    original_arrays_sha = values.native_source_array_digest(human)
    results = []
    for group in (29,36,33,34,38,51,52):
        local = copy.copy(human)
        local.records = [human.records[group]]
        local.full_stroke_records = [human.full_stroke_records[group]]
        actual = local.records[0]['row']['actual_pts']
        local.blocks = {actual:human.blocks[actual]}
        local.active = group_dependencies(local,local.full_stroke_records)[0]
        chart = capture_native_nominal_query_chart(local,eq)
        nominal_signature = local.queries(eq,full_stroke=True)[0]['query']['source_support_branch_signature']
        model = station.finite_rectangle_station_constraints(local,eq,local.active,
            full_stroke=True,group_limits=np.array([caps[0][group]]),return_step_carrier_response=True)
        response = model['step_carrier_response']
        matrix, lower = model['matrix'], model['lower']

        def trial_chart(q):
            original = native_nominal_query_carriers(local,q,chart)
            return station.actual_station_model_carriers(local,q,response,original_carriers=original)

        base = trial_chart(eq)
        np.testing.assert_allclose(base,response['nominal_carrier'],rtol=0.,atol=1e-8)
        predicted = matrix @ ed[local.active]
        finite_differences = []
        for h in (1e-3,1e-4):
            p = actual_carrier_step_response(response,trial_chart(eq+h*ed))
            m = actual_carrier_step_response(response,trial_chart(eq-h*ed))
            estimate = (p-m)/(2*h)
            finite_differences.append(dict(h=h,maximum_face_Jv_vs_same_chart_directional_FD_px=float(abs(estimate-predicted).max()),
                worst_face=int(np.argmax(abs(estimate-predicted)))))
        samples = []
        for alpha in (1.,.5,.25):
            q = eq+alpha*ed
            query = local.queries(q,full_stroke=True)[0]['query']
            actual_guard = float(abs(query['residual']).max())
            trial = trial_chart(q)
            face_response = actual_carrier_step_response(response,trial)
            numerical_slack = face_response-lower
            dynamic_target = float(caps[0][group])
            metadata = dict(response,row_start=0,row_stop=len(lower),
                dynamic_group_limits=np.array([dynamic_target]),
                dynamic_row_group_index=np.zeros(len(lower),int),
                dynamic_guard_comparison_tolerance_px=1e-8)
            tightening = measured_outward_step_tightening(matrix,lower,alpha*ed[local.active],
                {'native':metadata},{'native':trial},actual_group_violations={'native':np.array([actual_guard])})
            details = [] if tightening is None else tightening['report']['compensated_faces']
            same_chart_curvature = predicted*alpha-face_response
            dynamic_is_feasible = actual_guard <= dynamic_target+1e-8
            if dynamic_is_feasible and details:
                raise AssertionError('Actual feasible original source group was assigned a fake SOC cut')
            samples.append(dict(alpha=alpha,actual_original_dynamic_guard_maximum_px=actual_guard,
                saved_parent_original_guard_limit_px=dynamic_target,actual_original_guard_accepts=dynamic_is_feasible,
                local_chart_faces_are_original_guard=False,
                numerical_chart_negative_slack_faces=int(np.count_nonzero(numerical_slack < -1e-8)),
                maximum_numerical_chart_negative_slack_px=float(np.maximum(-numerical_slack,0.).max()),
                maximum_same_chart_curvature_px=float(abs(same_chart_curvature).max()),
                generated_SOC_cuts=len(details),actual_guard_gated_SOC_preserves_feasible_witness_change=dynamic_is_feasible and not details,
                original_dynamic_guard_still_rejects_actual_violation=not dynamic_is_feasible,
                source_branch_changed_vertices=int(np.count_nonzero(np.any(
                    query['source_support_branch_signature'] != nominal_signature,axis=1)))))
        entry = dict(group=group,id=local.records[0]['row']['id'],source_guard_vertices=len(local.full_stroke_records[0]['uv']),
            selected_truth_dependency_columns=len(local.active),small_truth_J_only=True,source_J_constructed=False,
            finite_station_contexts=len(response.get('finite_station_contexts',[])),
            numerical_chart=('fixed_nominal_projected_segment_foot' if response.get('finite_station_contexts') else
                             'nominal_segment_category_support_face_station_formula'),
            nominal_response_matches_nominal_carrier_maximum_px=float(abs(base-response['nominal_carrier']).max()),
            directional_FD_checks=finite_differences,samples=samples)
        results.append(entry)
        print(common.json.dumps(dict(stage='repaired_native_mixed_chart',group=group,id=entry['id'],
            h1e4_maximum_error=finite_differences[-1]['maximum_face_Jv_vs_same_chart_directional_FD_px'],
            cuts=[s['generated_SOC_cuts'] for s in samples])),flush=True)
    verified_implementation = values.verify_implementation_unchanged()
    assert values.native_source_array_digest(human) == original_arrays_sha
    report = dict(schema='rinkspace.repaired-native-mixed-chart-saved-direction-audit.v1',script=common.ref(__file__),
        references=refs,previous_branch_counterexamples=common.ref(values.old.OUT),
        implementation=verified_implementation,
        implementation_captured_before_import=values.IMPLEMENTATION_BEFORE_IMPORT,
        implementation_import_to_report_SHA_unchanged=True,
        new_fit=False,source_J_H_g_or_motion_graph_constructed=False,small_truth_J_groups=len(results),
        source_images_written=0,source_frames_decoded=0,original_support_budgets_sampling_unchanged=True,
        original_native_source_arrays_before_after_sha256=original_arrays_sha,
        original_native_source_arrays_before_after_bitexact=True,
        q276_half_direction_reproduces_saved_q8_bit_for_bit=True,groups=results,
        limits=['Seven native groups and one saved complete direction; this does not certify arbitrary future directions or wide/confirmed chart routes.',
            'A trial fixed-chart FM value need not equal actual original guard; actual guard separately controls SOC gating and acceptance.',
            'Existing 1e-8 original driver guard comparison is reused; no new source-pixel tolerance is introduced.'],seconds=time.perf_counter()-began)
    OUT.write_text(common.json.dumps(report,default=common.plain,ensure_ascii=False,indent=2),encoding='utf8')
    print(common.json.dumps(dict(path=str(OUT),sha256=common.sha(OUT),bytes=OUT.stat().st_size,seconds=report['seconds'])))


if __name__ == '__main__':
    main()
