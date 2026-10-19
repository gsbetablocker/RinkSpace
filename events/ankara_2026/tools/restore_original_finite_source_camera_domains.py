"""Restore actual qualified source domains using the complete cached driver.

This boundary capture stops before full source linearization. The restored
checkpoint preserves the complete parent arrays and changes RF4 view slots
only. Source rows, physical geometry, mounts and hard guards are unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from calibration.solver_runtime import install
install()
import numpy as np


class DomainInitializationComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    if options.output.exists():
        raise FileExistsError(options.output)
    options.output.with_name(options.output.name + '.bootstrap').mkdir(parents=True, exist_ok=True)
    parent = dict(np.load(options.seed, allow_pickle=False))
    original_q = parent['q'].copy()
    config = json.loads(options.config.read_text(encoding='utf8'))
    started = time.perf_counter()
    import run_current_named66_full_programme as driver
    from structured_programme_source_candidate import absolute_camera_pose_initializer as initializer
    original_initializer = initializer.reinitialize_invalid_qualified_source_views

    def capture(state, q, **kwargs):
        frame = sys._getframe(1)
        context = frame.f_locals
        core_count = int(context['core_count'])
        print(json.dumps(dict(stage='actual_complete_finite_initializer_started',
            blocks=[dict(name=b.name, finite_records=len(b.bundle.finite_records)) for b in state.blocks],
            seconds=time.perf_counter() - started)), flush=True)
        for block in state.blocks:
            bundle = block.bundle
            previous = bundle.validate_finite_extents
            def logged(cache, previous=previous, name=block.name):
                result = previous(cache)
                print(json.dumps(dict(stage='actual_original_finite_block_guard_passed', block=name,
                    seconds=time.perf_counter() - started)), flush=True)
                return result
            bundle.validate_finite_extents = logged
        attempt = 0
        active_initializer = original_initializer
        while True:
            attempt += 1
            try:
                candidate, report = active_initializer(state, q, **kwargs)
                context['raw'].guard(candidate[:core_count])
                break
            except ValueError as error:
                trace = error.__traceback__
                details = None
                while trace is not None:
                    values = trace.tb_frame.f_locals
                    if trace.tb_frame.f_code.co_name == 'reinitialize_invalid_qualified_source_views':
                        details = dict(block=getattr(values.get('block'), 'name', None),
                            actual_pts=values.get('pts'), before=values.get('before_facts'),
                            after=values.get('after_facts'), before_domains=values.get('before'),
                            after_domains=values.get('after'), original_RF4=values.get('initial'),
                            proposed_RF4=values.get('proposed'),
                            optimizer_status=str(getattr(values.get('optimized'), 'message', None)))
                    trace = trace.tb_next
                failure = dict(error=str(error), attempt=attempt, actual_failure=details,
                    complete_actual_state_kept_in_this_process=True,
                    source_rows_not_removed=True, hard_guard_not_relaxed=True)
                def json_default(value):
                    return value.tolist() if isinstance(value, np.ndarray) else value.item() if isinstance(value, np.generic) else str(value)
                target = options.output / f'actual_finite_initializer_failure_{attempt:02}.json'
                target.write_text(json.dumps(failure, ensure_ascii=False, indent=2, default=json_default), encoding='utf8')
                print(json.dumps(dict(stage='actual_finite_initializer_failed_warm_state_retained',
                    attempt=attempt, error=str(error), diagnostic=str(target)), ensure_ascii=False), flush=True)
                retry = options.output / f'retry_qualified_pose_initializer_{attempt:02}.json'
                while not retry.exists():
                    time.sleep(1.)
                request = json.loads(retry.read_text(encoding='utf8'))
                if request.get('action') != 'reload_and_retry':
                    raise ValueError('Explicit warm retry action required')
                from calibration.solver_runtime import runtime
                runtime.status(verify_sources=True)
                runtime._finder.modules = runtime._manifest()['modules']
                active_initializer = importlib.reload(initializer).reinitialize_invalid_qualified_source_views
                print(json.dumps(dict(stage='actual_finite_initializer_warm_retry', attempt=attempt+1)), flush=True)
        changed = np.asarray(report['changed_parameter_indices'], int)
        if np.any(changed >= int(parent['aux__block_start'][-1] + parent['aux__block_lengths'][-1])):
            raise ValueError('Initializer changes must be preserved original RF4 prefix slots only')
        canonical = original_q.copy()
        canonical[changed] = candidate[changed]
        report.update(original_complete_seed=str(options.seed), original_complete_q_count=len(canonical),
            full_original_source_guard_passed=True, all_original_finite_block_guards_passed=True,
            original_auxiliary_and_profile_arrays_preserved=True, initializer_only=True,
            original_FIT_HELD_UV_sigma_and_roles_unchanged=True,
            candidate_complete_q_sha256=hashlib.sha256(canonical.tobytes()).hexdigest(),
            elapsed_seconds=time.perf_counter() - started)
        parent['q'] = canonical
        identity = json.loads(str(parent['identity_json'].item()))
        identity['finite_source_RF4_initializer'] = report
        identity['q_sha256'] = report['candidate_complete_q_sha256']
        parent['identity_json'] = np.asarray(json.dumps(identity, ensure_ascii=False, separators=(',', ':')))
        np.savez(options.output / 'complete_original_finite_source_camera_initializer_UNACCEPTED.npz', **parent)
        (options.output / 'actual_original_finite_source_camera_initializer.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        print(json.dumps(dict(stage='actual_complete_finite_initializer_saved',
            repairs=len(report['repairs']), changed_RF4_coordinates=len(changed),
            q_sha256=report['candidate_complete_q_sha256'], output=str(options.output))), flush=True)
        raise DomainInitializationComplete()

    initializer.reinitialize_invalid_qualified_source_views = capture
    source_options = dict(config['source_options'])
    source_options['--camera-seed'] = str(options.seed)
    source_options['--reinitialize-invalid-qualified-source-poses'] = True
    # Execution-only choices avoid unused actors before the captured boundary.
    source_options['--residual-source-workers'] = '0'
    source_options['--serial-current-patterns'] = True
    argv = ['restore_original_finite_source_camera_domains', '--output', str(options.output)]
    for key, value in source_options.items():
        if isinstance(value, bool):
            if value:
                argv.append(key)
        elif value is not None:
            argv.extend([key, str(value)])
    sys.argv = argv
    try:
        driver.main()
    except DomainInitializationComplete:
        return
    raise RuntimeError('The actual finite-source initialization boundary was not reached')


if __name__ == '__main__':
    main()
