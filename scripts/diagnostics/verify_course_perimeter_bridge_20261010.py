"""New direct-perimeter bridge on real saved course; no old station benchmark."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from calibration.solver_runtime import configure_native_runtime
from calibration.solver_runtime.course_perimeter_bridge_candidate import existing_cached_perimeter


def exact(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-directory', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_native_runtime(stage='existing_cache_direct_perimeter_bridge_check', native_threads=1)
    from rinkspace.vision_camera_initialization import current_complete_source_geometry
    from structured_programme_source_candidate.course_station_cache import install_exact_course_station_cache
    options = json.loads(args.config.read_text(encoding='utf-8-sig'))['source_options']
    state = args.state_directory / 'complete_state_UNACCEPTED.npz'
    provider, world, provenance = current_complete_source_geometry(state, options['--world-init'])
    course, cq, _ = provider.current.print_course_binding(world)
    assert exact(course.sample_perimeter(cq), existing_cached_perimeter(course, cq))
    cache = install_exact_course_station_cache(course)
    samples = [('saved_nominal', cq), ('original_initial', course.initial)]
    for column in range(len(cq)):
        for sign in (-1, 1):
            value = cq.copy(); value[column] += sign * 1e-6
            samples.append((f'course_{column}_{sign}', value))
    records = []
    for label, value in samples:
        for jacobian in (False, True):
            begin = time.perf_counter()
            old = course.sample_perimeter(value, with_jacobian=jacobian)
            original_seconds = time.perf_counter() - begin
            # The original cache generates its own unchanged inputs once.
            cache._inputs(value)
            begin = time.perf_counter()
            new = existing_cached_perimeter(course, value, with_jacobian=jacobian)
            seconds = time.perf_counter() - begin
            if not exact(old, new):
                raise AssertionError('Actual original perimeter arrays/metadata differ: ' + label)
            new[0][0, 0] += 1.
            if not exact(old, existing_cached_perimeter(course, value, with_jacobian=jacobian)):
                raise AssertionError('A caller mutation contaminated the original cache')
            records.append(dict(sample=label, jacobian=jacobian, all_array_bytes_and_metadata_equal=True,
                                independent_output_mutation_control=True,
                                original_direct_perimeter_seconds=original_seconds, candidate_warm_seconds=seconds))
    invalid = cq.copy(); invalid[0] = -1.
    errors = []
    for fn in (lambda: course.sample_perimeter(invalid), lambda: existing_cached_perimeter(course, invalid)):
        try:
            fn()
        except (ValueError, RuntimeError) as exc:
            errors.append((type(exc).__name__, str(exc)))
    assert len(errors) == 2 and errors[0] == errors[1]
    helper = ROOT / 'calibration/solver_runtime/course_perimeter_bridge_candidate.py'
    proof = dict(schema='rinkspace.existing-exact-station-cache-direct-perimeter-bridge.v1', passed=True,
        actual_geometry_provenance=provenance, original_station_cache_reused=True,
        original_cache_budget_and_startup_condition_unchanged=True, uninstalled_cache_original_fallback_equal=True,
        actual_samples=records, physical_invalid_domain_exception_equal=errors[0],
        cache_statistics=cache.statistics(), helper=dict(path=str(helper), sha256=hashlib.sha256(helper.read_bytes()).hexdigest()),
        full_programme_graph_J_normal_not_loaded=True, previous_station_benchmark_not_repeated=True,
        installed_on_running_process=False, full_constraint_stage_speedup_not_proven=True, accuracy_accepted=False)
    proof['small_same_input_direct_perimeter_timing'] = dict(
        original_total_seconds=sum(row['original_direct_perimeter_seconds'] for row in records),
        candidate_warm_total_seconds=sum(row['candidate_warm_seconds'] for row in records),
        original_station_cache_was_installed=True, complete_constraint_call_frequency_not_measured=True)
    path = args.output / 'actual_saved_course_perimeter_bridge_PASS.json'
    path.write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(passed=True, proof=str(path), checks=len(records), cache_hits=cache.hits)), flush=True)


if __name__ == '__main__':
    main()
