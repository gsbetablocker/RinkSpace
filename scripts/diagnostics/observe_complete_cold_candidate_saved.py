"""Summarize saved complete-cycle evidence only; no source import/evaluation."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


def data(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def events(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def signature(value):
    return dict(dtype=value.dtype.str, shape=list(value.shape),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def epoch_ns(value):
    stamp = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
    # Creation ticks are used for exact ownership; timestamp conversion here
    # is only a wall-clock metric at submicrosecond precision.
    return int(stamp.timestamp()*1e9)


def cpu_window(samples, start, stop):
    selected = [item for item in samples
                if epoch_ns(item['timestamp_utc']) <= stop
                and epoch_ns(item['timestamp_utc'])-int(item['interval_seconds']*1e9) >= start]
    wall = sum(item['interval_seconds'] for item in selected)
    cpu = sum(item['cpu_seconds_delta'] for item in selected)
    return dict(measurement_start_ns=start, measurement_end_ns=stop,
        fully_contained_sample_count=len(selected), covered_seconds=wall,
        summed_cpu_seconds=cpu, average_effective_busy_cores=cpu/wall if wall else None,
        average_machine_cpu_percent=100*cpu/wall/selected[0]['logical_cpu_count'] if wall else None,
        peak_summed_working_set_bytes=max((item['summed_working_set_bytes'] for item in selected), default=None),
        working_set_sum_may_double_count_shared_pages=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--prior-run', type=Path)
    parser.add_argument('--require-os-exit', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    opt = parser.parse_args()
    run, baseline = opt.run.resolve(), opt.baseline.resolve()
    boot, oldboot = Path(str(run)+'.bootstrap'), Path(str(baseline)+'.bootstrap')
    current, old = data(boot/'complete_cycle_comparison.json'), data(oldboot/'complete_cycle_comparison.json')
    launch, prior_launch = data(boot/'launch.json'), data(oldboot/'launch.json')
    current_start = int(launch['solver_creation_ticks_utc'])*100-62135596800000000000
    baseline_start = int(prior_launch['solver_creation_ticks_utc'])*100-62135596800000000000
    saved = data(boot/'complete_result_saved.json')
    log = events(boot/'stdout.log')
    begin = [item for item in log if item.get('stage') == 'complete_cycle_comparison_begin']
    assert len(begin) == 1
    references = current['fresh_source_block_signatures'], old['fresh_source_block_signatures']
    assert len(references[0]) == len(references[1]) == 32
    blocks = [dict(ordinal=i, family=item.get('family'), complete_bytes_equal=item==other)
              for i, (item, other) in enumerate(zip(*references))]
    checkpoint = run/'accepted_states'/current['accepted_q']['sha256']/'complete_state_UNACCEPTED.npz'
    baseline_checkpoint = baseline/'accepted_states'/old['accepted_q']['sha256']/'complete_state_UNACCEPTED.npz'
    with np.load(checkpoint, allow_pickle=False) as new, \
            np.load(baseline_checkpoint, allow_pickle=False) as previous:
        fields = []
        for key in sorted(set(new.files)|set(previous.files)):
            left = signature(previous[key]) if key in previous else None
            right = signature(new[key]) if key in new else None
            fields.append(dict(field=key, baseline=left, candidate=right, byte_exact=left==right))
        profile_fields = [item for item in fields if item['field'].startswith('profile__')]
        qsig = signature(new['q'])
        assert qsig == current['accepted_q']
        q_difference = dict(max_absolute=float(np.max(np.abs(new['q']-previous['q']))),
            RMS=float(np.sqrt(np.mean((new['q']-previous['q'])**2))))
        field_count, field_order_equal = len(new.files), new.files == previous.files
    truth_start = data(data(boot/'input_config.json')['source_options']['--truth-restoration-state'])
    truth_after = data(run/'accepted_states'/qsig['sha256']/'truth_restoration_state.json')
    authority_equal, monotonic, groups = True, True, 0
    for family, authority in truth_start['families'].items():
        after = truth_after['families'][family]
        authority_equal &= all(authority[k] == after[k] for k in ('authority','group_ids','uncertainties'))
        monotonic &= bool(np.all(np.asarray(after['limits']) <= np.asarray(authority['limits'])))
        groups += len(authority['group_ids'])
    samples = events(boot/'process_tree_utilization.jsonl')
    exit_proof = data(boot/'owned_identity_natural_exit.json')
    process_exit_path = boot/'complete_process_exit.json'
    if opt.require_os_exit and not process_exit_path.is_file():
        raise ValueError('Real OS process-exit evidence is required for the final cold runtime')
    process_exit = data(process_exit_path) if process_exit_path.is_file() else None
    if process_exit is not None:
        if (process_exit['solver_process_id'] != launch['solver_process_id']
                or int(process_exit['solver_creation_ticks_utc']) != int(launch['solver_creation_ticks_utc'])):
            raise ValueError('Native OS exit handle belongs to a different solver creation identity')
    shared = events(boot/'shared_actor_geometry_execution/shared_actor_geometry_events.jsonl')
    if not shared:
        shared = [item for path in (boot/'shared_actor_geometry_execution').glob('*.jsonl') for item in events(path)]
    roles = [item for path in (boot/'shared_actor_geometry_execution/role_block_binding').glob('*.jsonl')
             for item in events(path) if item.get('stage') == 'owned_actor_role_view_installed']
    gc_events = [item for directory in ('source_skeleton_execution', 'shared_actor_geometry_execution/role_block_binding')
                 for item in events(boot/directory/'source_assembly_gc_events.jsonl')]
    camera15 = [item for path in (boot/'camera15_group_prefetch_execution').glob('*.jsonl') for item in events(path)]
    lp = [item for path in (boot/'full_priority_Clarabel_execution').glob('*.jsonl') for item in events(path)]
    qp = [item for item in log if item.get('stage') == 'QP_backend_end']
    native = [item for path in (boot/'native_factor_columns_execution').glob('*.jsonl') for item in events(path)]
    normal = data(run/'first_complete_source_normal.json')
    acceptance_wall = (current['wall_time_ns']-current_start)/1e9
    baseline_wall = (old['wall_time_ns']-baseline_start)/1e9
    report = dict(schema='rinkspace.readonly-complete-cold-candidate-observation.v1',
        candidate=run.name, historical_baseline=baseline.name, no_historical_run_repeated=True,
        same_initial_conditions=current['start']==old['start'], initial_conditions=current['start'],
        complete_cycle_seconds=current['complete_cycle_seconds'], historical_complete_cycle_seconds=old['complete_cycle_seconds'],
        cold_process_creation_to_acceptance_seconds=acceptance_wall,
        cold_process_creation_to_saved_and_workers_closed_seconds=(saved['wall_time_ns']-current_start)/1e9,
        historical_cold_process_creation_to_acceptance_seconds=baseline_wall,
        cold_to_acceptance_speedup=baseline_wall/acceptance_wall,
        cold_target_under60_achieved=(saved['wall_time_ns']-current_start)/1e9 < 60,
        first_complete_source_J_and_normal_seconds=normal['source_J_and_normal_seconds'],
        fresh_source_block_count=32, all_fresh_source_block_bytes_equal=all(item['complete_bytes_equal'] for item in blocks),
        fresh_source_block_comparison=blocks, accepted_q=qsig, accepted_q_exactly_equal=current['accepted_q']==old['accepted_q'],
        accepted_q_numerical_difference=q_difference, accepted_checkpoint_field_count=field_count,
        accepted_checkpoint_key_order_equal=field_order_equal,
        accepted_checkpoint_paths=dict(candidate=str(checkpoint), baseline=str(baseline_checkpoint)),
        baseline_latest_checkpoint_not_used_for_first_accepted_profile_comparison=True,
        accepted_profile_fields=len(profile_fields), all_accepted_profile_fields_byte_equal=all(item['byte_exact'] for item in profile_fields),
        accepted_checkpoint_all_fields=fields, accepted_checkpoint_different_fields=[item for item in fields if not item['byte_exact']],
        original_full_acceptance=current['acceptance'], accuracy_accepted=False,
        truth=dict(groups=groups, original_authorities_group_ids_uncertainties_unchanged=authority_equal,
                   all_limits_monotonically_preserved=monotonic),
        CPU_first_complete_cycle=cpu_window(samples, begin[0]['wall_time_ns'], current['wall_time_ns']),
        CPU_cold_to_saved_and_workers_closed=cpu_window(samples, current_start, saved['wall_time_ns']),
        QP=dict(backend_call_count=len(qp), backend_total_seconds=sum(item.get('seconds',0) for item in qp), backends=qp),
        shared_actor_events=shared, owned_role_events=roles,
        compaction_binding_reindex_events=[item for path in (boot/'compaction_binding_reindex_execution').glob('*.jsonl') for item in events(path)],
        source_skeleton_events=events(boot/'source_skeleton_execution/named66_source_skeleton_execution.jsonl'),
        source_assembly_GC_events=gc_events,
        camera15_group_prefetch_events=camera15,
        full_priority_Clarabel_LP_events=lp,
        optional_SOC_events=[item for item in log if item.get('stage') == 'optional_SOC_restricted_numerical_failure'],
        native_factor_execution_events=native, source_stage_timing_seconds=current['source_stage_timing_seconds'],
        owned_exact_identity_exit_report=str(boot/'owned_identity_natural_exit.json'),
        owned_exact_identity_count=exit_proof['launch_and_monitor_history_unique_identity_count'],
        all_owned_identities_gone=exit_proof['all_owned_exact_identities_absent'])
    if process_exit is not None:
        report.update(complete_native_process_exit=process_exit,
            complete_native_process_exit_report=str(process_exit_path),
            cold_process_creation_to_true_OS_exit_seconds=process_exit['cold_start_to_native_process_exit_seconds'],
            cold_target_under60_achieved=process_exit['cold_start_to_native_process_exit_seconds'] < 60,
            final_cold_runtime_boundary='retained exact native process StartTime through actual OS ExitTime')
        report['CPU_cold_to_true_OS_exit'] = cpu_window(samples, current_start,
            epoch_ns(process_exit['native_exit_time_utc']))
    if opt.prior_run:
        previous = data(Path(str(opt.prior_run)+'.bootstrap')/'complete_cycle_comparison.json')
        report['previous_candidate'] = dict(name=opt.prior_run.name,
            complete_cycle_seconds=previous['complete_cycle_seconds'],
            all32_fresh_source_block_bytes_equal=references[0]==previous['fresh_source_block_signatures'],
            accepted_q_exactly_equal=current['accepted_q']==previous['accepted_q'])
    opt.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(report=str(opt.output), all32=report['all_fresh_source_block_bytes_equal'],
        field_count=field_count, profile_fields=len(profile_fields), profiles_exact=report['all_accepted_profile_fields_byte_equal'],
        q_exact=report['accepted_q_exactly_equal'], differing_fields=[item['field'] for item in report['accepted_checkpoint_different_fields']],
        cycle_seconds=current['complete_cycle_seconds'], cold_saved_seconds=report['cold_process_creation_to_saved_and_workers_closed_seconds'],
        all_owned_gone=report['all_owned_identities_gone']), ensure_ascii=False))


if __name__ == '__main__':
    main()
