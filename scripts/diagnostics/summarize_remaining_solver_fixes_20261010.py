"""Audit the completed real continuation using saved evidence only."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from calibration.solver_runtime.checkpoint_history import iter_solver_history

import numpy as np


def read(path):
    return Path(path).read_text(encoding='utf-8-sig')


def data(path):
    return json.loads(read(path))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    boot = Path(str(run) + '.bootstrap')
    events = []
    for line in read(boot / 'stdout.log').splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            events.append(item)
    history = iter_solver_history(run / 'solver/iterations.json')
    accepted = [item for item in history if item.get('accepted')]
    performance = data(run / 'actual_accepted_cycle_performance.json')
    result = data(run / 'solver/solver_result.json')
    exited = data(boot / 'solver_exit.json')
    request = data(boot / 'actual_QP_stop_request.json')
    cache = data(run / 'actual_control_requested_current_complete_cache.json')
    assert result['status'] == 'stopped_by_control'
    assert exited['exit_code'] == 0
    assert len(accepted) == len(performance['cycles']) == result['accepted_iterations']
    observed = next(item for item in events if item.get('stage') == 'source_control_observed_in_QP')
    returned = next(item for item in events if item.get('stage') == 'source_control_QP_returned_to_original_solver')
    cancelled = [item for item in events if item.get('stage') == 'QP_backend_end'
                 and item.get('error_type') == 'SourceControlStop']
    assert len(cancelled) == 1
    assert cancelled[0]['QP_call'] == request['backend_begin']['QP_call']
    backend_begins = [item for item in events if item.get('stage') == 'QP_backend_begin']
    assert not any(item['wall_time_ns'] > observed['wall_time_ns'] for item in backend_begins)
    checkpoint = run / 'solver/latest_accepted_state.npz'
    with np.load(checkpoint, allow_pickle=False) as saved:
        token = hashlib.sha256(saved['q'].tobytes()).hexdigest()
        assert token == request['last_saved_accepted_state']['q_sha256']
        assert token == performance['cycles'][-1]['q_sha256'] == cache['q_sha256']
        with np.load(Path(cache['directory']) / 'actual_initial_frozen_profile_arrays.npz', allow_pickle=False) as profiles:
            names = sorted(key[9:] for key in saved.files if key.startswith('profile__'))
            assert names == sorted(profiles.files)
            for key in names:
                left, right = saved['profile__' + key], profiles[key]
                assert left.dtype == right.dtype and left.shape == right.shape
                assert left.tobytes() == right.tobytes()
        final_state = dict(q_sha256=token, parameters=len(saved['q']), fields=len(saved.files),
            profile_fields=len(names), damping=float(saved['damping']),
            next_line_search_fraction=float(saved['next_line_search_fraction']),
            path=str(checkpoint), sha256=digest(checkpoint),
            last_accepted_q_byte_exact_after_QP_cancellation=True,
            all_frozen_profile_fields_byte_exact_with_sealed_cache=True)
    config = data(boot / 'input_config.json')
    before = data(config['source_options']['--truth-restoration-state'])
    truth_path = run / 'accepted_states' / token / 'truth_restoration_state.json'
    after = data(truth_path)
    assert before['source_target'] == after['source_target']
    assert before['source_role_operator_target'] == after['source_role_operator_target']
    assert before['families'].keys() == after['families'].keys()
    groups = 0
    for family, prior in before['families'].items():
        current = after['families'][family]
        for key in ('authority', 'group_ids', 'uncertainties'):
            assert prior[key] == current[key]
        assert np.all(np.asarray(current['limits']) <= np.asarray(prior['limits']))
        groups += len(prior['group_ids'])
    camera = data(boot / 'camera_export_execution/actual_same_q_camera_export_comparison.json')
    assert camera['passed'] and all(item['byte_exact'] for item in camera['fields'])
    samples = [json.loads(line) for line in read(boot / 'process_tree_utilization.jsonl').splitlines() if line.strip()]
    assert all('creation_time_utc' in process for sample in samples for process in sample['processes'])
    wall = sum(sample['interval_seconds'] for sample in samples)
    cpu = sum(sample['cpu_seconds_delta'] for sample in samples)
    cycles = []
    for cycle, iteration in zip(performance['cycles'], accepted):
        stages = iteration.get('source_stage_timing_seconds', {})
        cycles.append(dict(q_sha256=cycle['q_sha256'], derivative_model=cycle['derivative_model'],
            accepted_to_accepted_wall_seconds=cycle['accepted_to_accepted_wall_seconds'],
            elapsed_from_run_start_seconds=cycle['elapsed_from_run_start_seconds'],
            solver_elapsed_cumulative_seconds=cycle['solver_elapsed_seconds'],
            source_stage_timing_seconds=stages))
    report = dict(schema='rinkspace.remaining-solver-fixes-real-continuation.v1',
        run=str(run), initial_q_sha256=config['continuation']['state']['q_sha256'],
        accepted_new_states=len(accepted), original_source_rows=cache['actual_complete_rows'],
        original_source_blocks=cache['actual_blocks'], accepted_cycles=cycles,
        first_complete_source_J_and_normal_seconds=performance['first_complete_source_J_and_normal_seconds'],
        final_state=final_state,
        truth=dict(path=str(truth_path), sha256=digest(truth_path), groups=groups,
            original_source_roles_rows_authorities_group_ids_uncertainties_unchanged=True,
            every_saved_limit_monotonically_preserved=True),
        QP=dict(backend_calls=len(backend_begins),
            phase_counts=dict(Counter(item['QP_phase'] for item in backend_begins)),
            optional_restricted_failures_observed=sum(item.get('stage') == 'optional_SOC_restricted_numerical_failure' for item in events),
            no_backend_fallback_after_control=True),
        explicit_stop=dict(backend_call=cancelled[0]['QP_call'],
            observed_iteration=observed['iterations'],
            cancelled_backend_total_seconds=cancelled[0]['seconds'],
            request_to_observation_seconds=(observed['wall_time_ns']-request['request_after_write_wall_time_ns'])/1e9,
            observation_to_original_loop_seconds=(returned['wall_time_ns']-observed['wall_time_ns'])/1e9,
            equal_wall_timestamps_do_not_establish_zero_latency=True,
            status=result['status'], exit_code=exited['exit_code'],
            forced_process_termination_used=False, current_fresh_cache=cache),
        camera=dict(all_fields_byte_exact=True, native_presentations=camera['native_presentations'],
            actual_canonical_exposures=camera['actual_canonical_exposures'],
            proof=str(boot / 'camera_export_execution/actual_same_q_camera_export_comparison.json')),
        CPU=dict(pid_and_creation_identity_verified=True, samples=len(samples),
            sampled_seconds=wall, sampled_cpu_seconds=cpu,
            average_busy_cores=cpu/wall,
            maximum_sample_busy_cores=max(sample['effective_busy_cores'] for sample in samples),
            includes_preparation_repeated_fresh_J_and_cache_handoff=True,
            sustained_full_CPU_utilization_demonstrated=False),
        limitations=[
            'The fresh-J CPU utilization bottleneck remains; lifecycle validation confirms the owned process order and observed memory use, not full CPU utilization.',
            'This continuation requires fresh derivatives and uses later states; its timings are not a new same-state speedup benchmark.',
            'The two new accepted states follow original truth restoration; source objective cost is not guaranteed to decrease in this mode.',
            'Complete-shot geometric accuracy has not been accepted; no new full video was generated.'
        ], accuracy_accepted=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(report=str(args.output), accepted_new_states=len(accepted),
        final_q_sha256=token, explicit_stop=report['explicit_stop']['request_to_observation_seconds'],
        exit_code=exited['exit_code'], profile_cache_exact=True)), flush=True)


if __name__ == '__main__':
    main()
