"""Compare the two original one-accepted-step runs; never evaluate a source."""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from calibration.solver_runtime import install
install()
from structured_programme_source_candidate.complete_same_state_replay import _array_signature


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def cycle_metrics(root, variant, report):
    boot = root / (variant + '.bootstrap')
    launch = read_json(boot/'launch.json')
    created = datetime.datetime.fromisoformat(launch['solver_creation_time_utc'].replace('Z', '+00:00')).timestamp()
    begins = []
    cumulative_before_guard = None
    for line in (boot/'stdout.log').read_text(encoding='utf-8-sig').splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('stage') == 'complete_cycle_comparison_begin':
            begins.append(event)
        elif event.get('stage') == 'current_source_guard':
            cumulative_before_guard = event['seconds']
    if len(begins) != 1:
        raise ValueError('Exactly one original complete cycle start is required')
    start, stop = begins[0]['wall_time_ns']/1e9, report['wall_time_ns']/1e9
    # These historical A/B monitors could adopt surviving children of a
    # reused parent PID. Their rows did not save the parent identity chain,
    # so the aggregate cannot be retrospectively certified as this run's.
    # Birth-time filtering alone does not establish ownership.
    monitor_invalid_reason = (
        'Historical monitor followed ParentProcessId without requiring child '
        'birth after the current parent birth; PID reuse could adopt unrelated '
        'processes. Saved samples lack the parent identity chain needed to '
        'certify ownership. CPU/RSS aggregates withdrawn; wall times unchanged.'
    )
    normal = read_json(root/variant/'first_complete_source_normal.json')
    return dict(complete_cycle_seconds=report['complete_cycle_seconds'],
        cold_start_to_accepted_seconds=report['wall_time_ns']/1e9-created,
        cold_start_boundary='actual launched solver creation UTC through original accepted review callback completion',
        startup_before_complete_cycle_boundary_seconds=start-created,
        solver_creation_time_utc=launch['solver_creation_time_utc'],
        first_complete_source_J_and_normal_seconds=normal['source_J_and_normal_seconds'],
        startup_cumulative_seconds_before_guard=cumulative_before_guard,
        complete_cycle_CPU_average_effective_cores=None,
        complete_cycle_CPU_average_percent_of_20_logical_CPUs=None,
        complete_cycle_CPU_sample_count=None, complete_cycle_CPU_sample_seconds=None,
        owned_summed_RSS_peak_bytes=None, summed_RSS_may_double_count_shared_pages=True,
        owned_CPU_RSS_metrics_valid=False, owned_CPU_RSS_metrics_invalid_reason=monitor_invalid_reason,
        source_stage_timing_seconds=report['source_stage_timing_seconds'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(r'D:\CodexData\RinkSpace_Data\events\ankara_2026\calibration\runs\solver_acceleration_20261003\complete_cycle_comparison_20261010'))
    parser.add_argument('--candidate-name', default='candidate')
    opt = parser.parse_args()
    root = opt.root.resolve()
    if Path(opt.candidate_name).name != opt.candidate_name or not opt.candidate_name.startswith('candidate'):
        raise ValueError('A candidate directory name inside the comparison root is required')
    names = {'baseline':'baseline', 'candidate':opt.candidate_name}
    reports = {v:read_json(root/(names[v]+'.bootstrap')/'complete_cycle_comparison.json') for v in names}
    baseline, candidate = reports['baseline'], reports['candidate']
    configs = {v:read_json(root/(names[v]+'.bootstrap')/'input_config.json')['source_options'] for v in reports}
    differing_options = [k for k in configs['baseline'] if configs['baseline'][k] != configs['candidate'].get(k)]
    if set(configs['baseline']) != set(configs['candidate']) or differing_options != ['--geometry-fd-workers']:
        raise ValueError('Only the explicit 8-to-20 FD execution setting may differ')
    if (str(configs['baseline']['--geometry-fd-workers']), str(configs['candidate']['--geometry-fd-workers'])) != ('8','20'):
        raise ValueError('Declared baseline8/candidate20 FD configuration changed')
    source_blocks_equal = (baseline['fresh_source_block_signatures'] == candidate['fresh_source_block_signatures'])
    if len(baseline['fresh_source_block_signatures']) != 32 or len(candidate['fresh_source_block_signatures']) != 32:
        raise ValueError('Every actual source block must be present in both complete cycles')
    array_reports = []
    compaction_timing_differences = []
    compaction_facts_equal = False
    with np.load(root/'baseline'/'solver'/'latest_accepted_state.npz', allow_pickle=False) as old, np.load(root/names['candidate']/'solver'/'latest_accepted_state.npz', allow_pickle=False) as new:
        fields_equal = set(old.files) == set(new.files)
        for key in sorted(set(old.files)|set(new.files)):
            left = _array_signature(old[key]) if key in old.files else None
            right = _array_signature(new[key]) if key in new.files else None
            array_reports.append(dict(field=key, baseline=left, candidate=right, byte_exact=left==right))
        profile_count = sum(k.startswith('profile__') for k in old.files)
        field_count = len(old.files)
        accepted_q_reports_match_npz = (_array_signature(old['q']) == baseline['accepted_q'] and _array_signature(new['q']) == candidate['accepted_q'])
        compaction_key = 'aux__current_original_motion_input_compaction_json'
        left_compaction = json.loads(str(old[compaction_key]))
        right_compaction = json.loads(str(new[compaction_key]))
        if len(left_compaction['blocks']) == len(right_compaction['blocks']):
            for index, (left_block, right_block) in enumerate(zip(left_compaction['blocks'], right_compaction['blocks'])):
                timing_key = 'actual_retained_r_J_check_seconds'
                if (timing_key in left_block) != (timing_key in right_block):
                    raise ValueError('Compaction timing receipt structure changed')
                if timing_key in left_block:
                    left_time, right_time = left_block.pop(timing_key), right_block.pop(timing_key)
                    if left_time != right_time:
                        compaction_timing_differences.append(dict(path=f'/blocks/{index}/{timing_key}', baseline=left_time, candidate=right_time))
            compaction_facts_equal = left_compaction == right_compaction
    # Compare every field above. These four carry execution times/output
    # references rather than numerical source/state; differences stay visible.
    metadata_fields = {'elapsed_seconds','history_json','reviews_json','identity_json'}
    state_arrays_equal = compaction_facts_equal and all(r['byte_exact'] for r in array_reports if r['field'] not in metadata_fields and r['field'] != compaction_key)
    same_start = baseline['start'] == candidate['start']
    passed = bool(same_start and source_blocks_equal and fields_equal and field_count==77 and profile_count==18 and state_arrays_equal and accepted_q_reports_match_npz)
    metrics = {v:cycle_metrics(root,names[v],reports[v]) for v in reports}
    result = dict(schema='rinkspace.complete-original-cycle-AB-comparison.v1', passed=passed,
        same_initial_q_profile_damping_and_fraction=same_start,
        all_32_fresh_source_r_mask_scales_and_CSR_arrays_byte_exact=source_blocks_equal,
        complete_checkpoint_field_names_equal=fields_equal, fields=field_count, profile_arrays=profile_count,
        accepted_numerical_and_source_auxiliary_state_arrays_byte_exact=state_arrays_equal,
        accepted_callback_q_matches_saved_npz=accepted_q_reports_match_npz,
        every_checkpoint_array_comparison=array_reports,
        compaction_source_facts_equal_excluding_only_recorded_check_seconds=compaction_facts_equal,
        compaction_check_seconds_differences=compaction_timing_differences,
        timing_output_metadata_fields_compared_but_not_used_as_numerical_state=sorted(metadata_fields),
        baseline_geometry_fd_workers=8, candidate_geometry_fd_workers=20,
        candidate_directory_name=opt.candidate_name,
        metrics=metrics, complete_cycle_speedup=metrics['baseline']['complete_cycle_seconds']/metrics['candidate']['complete_cycle_seconds'],
        cold_start_to_accepted_speedup=metrics['baseline']['cold_start_to_accepted_seconds']/metrics['candidate']['cold_start_to_accepted_seconds'],
        baseline_J_H_g_or_cached_prefix_imported=False, accuracy_accepted=False)
    destination = root/'complete_cycle_AB_comparison.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    (root/(opt.candidate_name+'.bootstrap')/'complete_cycle_AB_comparison.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(passed=passed, output=str(destination), metrics=metrics,
        complete_cycle_speedup=result['complete_cycle_speedup'],
        cold_start_to_accepted_speedup=result['cold_start_to_accepted_speedup']), ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
