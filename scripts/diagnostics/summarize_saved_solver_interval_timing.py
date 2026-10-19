"""Separate cumulative history clocks, actual acceptance intervals and cold J."""
from pathlib import Path
import argparse,hashlib,json
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from calibration.solver_runtime.checkpoint_history import iter_solver_history

def reference(path):
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    opt=p.parse_args();history_path=opt.run/'solver/iterations.json';history=iter_solver_history(history_path)
    initial_path=opt.run/'first_complete_source_normal.json';initial=json.loads(initial_path.read_text(encoding='utf8'))
    entries=[];last=0.;incomplete=[]
    for item in history:
        if 'complete_cycle_elapsed_seconds'not in item:
            incomplete.append(dict(iteration=item['iteration'],accepted=item.get('accepted'),
                elapsed_seconds=item.get('elapsed_seconds')))
            continue
        cumulative=float(item['complete_cycle_elapsed_seconds']);interval=cumulative-last;last=cumulative
        if interval<0.:raise ValueError('Saved cumulative cycle clock must be monotone')
        trial=item['global_schur_trial_records'][-1]
        soc=item['global_schur_linear_report'].get('sequential_measured_shared_source_curvature_tightening',{})
        measurements=soc.get('actual_original_nonlinear_feasibility_progress',{}).get('measurements',[])
        entries.append(dict(iteration=item['iteration'],accepted=item['accepted'],
            complete_cycle_elapsed_cumulative_seconds=cumulative,accepted_interval_wall_seconds=interval,
            iteration_elapsed_also_cumulative_seconds=item['elapsed_seconds'],
            initial_source_J_seconds_not_in_first_interval=initial['source_J_and_normal_seconds']if item['iteration']==0 else None,
            before_cost=item['cost'],trial_cost=item['next_cost'],alpha=trial['line_search_fraction'],
            q_sha256=trial.get('trial_feasibility_recovery',{}).get('complete_state_recovery_identity',{}).get('recovered_q_sha256'),
            existing_individual_stage_timing_seconds=item['source_stage_timing_seconds'],
            SOC_actual_measurements=[dict(maximum_violation_px=m['actual_original_face_feasibility']['maximum_violation_px'],
                total_violation_px=m['actual_original_face_feasibility']['total_violation_px'],
                complete_original_KKT_verified=m['complete_original_KKT_verified'],
                original_quadratic_model_decrease=m['original_quadratic_model_decrease'],
                face_model_reliable=m['original_face_reduction_model_is_reliable'])for m in measurements]))
    cold=float(initial['source_J_and_normal_seconds']);before=float(initial['seconds_since_start'])-cold
    result=dict(schema='rinkspace.saved-solver-cumulative-clock-to-real-acceptance-intervals.v1',
        run=str(opt.run.resolve()),history=reference(history_path),first_normal=reference(initial_path),
        complete_cycle_elapsed_and_iteration_elapsed_are_cumulative=True,
        acceptance_interval_derived_by_consecutive_cumulative_difference=True,
        subsequent_intervals_include_their_actual_intervening_work_and_are_not_source_J_only=True,
        initial_source_J_and_normal_seconds=cold,pre_first_J_preparation_seconds=before,
        first_cold_J_plus_complete_interval_seconds=cold+entries[0]['accepted_interval_wall_seconds'],
        stage_timing_totals_need_not_cover_profile_capture_or_intervening_following_normal_work=True,
        no_algorithm_or_live_state_change=True,accuracy_accepted=False,intervals=entries,
        history_records_without_complete_interval=incomplete)
    opt.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(output=reference(opt.output),steps=len(entries),latest_interval_seconds=entries[-1]['accepted_interval_wall_seconds'])))

if __name__=='__main__':main()
