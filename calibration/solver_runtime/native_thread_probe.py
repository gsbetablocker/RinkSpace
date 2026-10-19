"""Measure native thread scaling on a preserved real source-array fixture.

This does not construct a programme graph or claim complete iteration speed.
The existing fixture is read without modification; all outputs and native
branch traces must remain byte exact across the requested thread counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[3, 8, 12, 20])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--minimum-source-depth-scene-units", type=float, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or any(value < 1 for value in args.threads):
        parser.error("Positive repeats and thread counts are required")

    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[name] = "1"
    from calibration.solver_runtime import install
    install()
    from structured_programme_source_candidate.runtime_backend_contract import restore_native_backend_contract
    restore_native_backend_contract(stage="real-source-array-native-thread-probe", native_threads=args.threads[0])
    from z190_motion_native import native_motion, metadata
    import numpy as np

    with np.load(args.fixture, allow_pickle=False) as archive:
        arrays = {name: archive[name].item() if archive[name].ndim == 0 else archive[name]
                  for name in archive.files}
    arrays.pop("jacobian", None)
    fixture_info = json.loads(args.fixture.with_suffix(".json").read_text(encoding="utf8"))
    for name in fixture_info["none_keys"]:
        arrays[name] = None
    records = []
    for jacobian, schur in ((False, False), (True, True)):
        baseline = None
        for threads in args.threads:
            os.environ["RINKSPACE_MOTION_THREADS"] = str(threads)
            result, checks, trace = native_motion(**arrays, jacobian=jacobian, schur=schur,
                minimum_source_depth_scene_units=args.minimum_source_depth_scene_units,
                return_trace=True)
            outputs = tuple(result) if jacobian else (result,)
            if baseline is None:
                baseline = (outputs, checks, trace)
            output_exact = all(np.array_equal(left, right) for left, right in zip(outputs, baseline[0]))
            trace_exact = bool(np.array_equal(trace, baseline[2]))
            checks_exact = checks == baseline[1]
            if not (output_exact and trace_exact and checks_exact):
                raise RuntimeError(f"Native output or branch changed with {threads} threads")
            wall, cpu = [], []
            for _ in range(args.repeats):
                begin_wall, begin_cpu = time.perf_counter(), time.process_time()
                measured = native_motion(**arrays, jacobian=jacobian, schur=schur,
                    minimum_source_depth_scene_units=args.minimum_source_depth_scene_units)
                wall.append(time.perf_counter() - begin_wall)
                cpu.append(time.process_time() - begin_cpu)
                del measured
            record = dict(threads=threads, jacobian=jacobian, rows=len(arrays["a"]),
                output_exact=output_exact, trace_exact=trace_exact, checks_exact=checks_exact,
                wall_seconds=wall, cpu_seconds=cpu,
                median_wall_seconds=float(np.median(wall)), total_cpu_to_wall=float(sum(cpu)/sum(wall)))
            records.append(record)
            print(json.dumps(record, separators=(",", ":")), flush=True)
    report = dict(schema="rinkspace.real-source-array-native-thread-scaling.v1", passed=True,
        fixture=str(args.fixture.resolve()), fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        fixture_array_bytes=sum(getattr(value, "nbytes", 0) for value in arrays.values()),
        logical_processors=os.cpu_count(), native_backend=metadata(), records=records,
        current_complete_programme_graph_reconstructed=False, complete_iteration_speedup_claimed=False,
        accuracy_accepted=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(dict(passed=True, report=str(args.output.resolve()))), flush=True)


if __name__ == "__main__":
    main()
