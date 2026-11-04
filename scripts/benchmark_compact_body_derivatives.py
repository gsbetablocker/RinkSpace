"""Bounded synthetic compact-operator check, never an end-to-end fit claim.

Run with the selected project/runtime.  Both paths include all 63 local chart
columns, 55 FK/skin joints, dense pose blend shapes and all 17 regressed points.
The new timing includes constructing its FK/skinning linearization context.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time

import numpy as np
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'src'))
from calibration.program_tracking.body_tracking.rebuild_body_geometry import CompactBody
from calibration.program_tracking.body_tracking.compact_body_derivatives import CompactBodyLinearization


def synthetic_model(path, vertices):
    rng = np.random.default_rng(361104)
    weights = rng.random((vertices, 55)); weights /= weights.sum(1, keepdims=True)
    reg = rng.random((vertices, 17)); reg /= reg.sum(0, keepdims=True)
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
        13, 14, 16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29,
        20, 31, 32, 20, 34, 35, 20, 37, 38, 21, 40, 41, 21, 43, 44,
        21, 46, 47, 21, 49, 50, 21, 52, 53]
    np.savez(path, v_template=rng.normal(scale=.15, size=(vertices, 3)),
        shapedirs=rng.normal(scale=.01, size=(vertices, 3, 10)),
        posedirs=rng.normal(scale=.001, size=(486, vertices, 3)),
        lbs_weights=weights, smplx2coco17_interestd=reg, parents=np.asarray(parents),
        other_default_pose=rng.normal(scale=.3, size=99),
        J_template=rng.normal(scale=.1, size=(55, 3)),
        J_shapedirs=rng.normal(scale=.01, size=(55, 3, 10)),
        rest_patch_centroids=rng.normal(scale=.05, size=(2, 2, 3)),
        rest_patch_height=rng.random((2, 2))*.05)
    return CompactBody(path, rng.normal(scale=.2, size=10), 1.13)


def measure(model, samples, repeats):
    rng = np.random.default_rng(1104+samples)
    reference = Rotation.from_rotvec(rng.normal(scale=.6, size=(samples*21, 3))).as_matrix().reshape(samples, 21, 3, 3)
    delta = rng.normal(scale=.3, size=(samples, 21, 3))
    columns = list(range(6, 69)); step = 1e-5

    def original():
        results = []
        for column in columns:
            sides = []
            for sign in (1., -1.):
                trial = delta.copy(); trial.reshape(samples, 63)[:, column-6] += sign*step
                local = reference @ Rotation.from_rotvec(trial.reshape(-1, 3)).as_matrix().reshape(samples, 21, 3, 3)
                sides.append(model.forward_rotations(local, with_frames=True)[0])
            results.append((sides[0]-sides[1])/(2*step))
        return results

    def analytic():
        linearization = CompactBodyLinearization(model, reference, delta)
        return [linearization.derivative(column, kinematics=False)['coco'] for column in columns]

    expected = original(); actual = analytic()
    maximum = max(float(np.max(np.abs(a-b))) for a, b in zip(actual, expected))
    relative = float(np.linalg.norm(np.asarray(actual)-expected)/np.linalg.norm(expected))
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-8)
    timings = {'original_fd_seconds': [], 'analytic_seconds': []}
    for repeat in range(repeats):
        order = [('original_fd_seconds', original), ('analytic_seconds', analytic)]
        if repeat % 2: order.reverse()
        for key, call in order:
            start = time.perf_counter(); value = call()
            timings[key].append(time.perf_counter()-start)
            del value
    old = statistics.median(timings['original_fd_seconds'])
    new = statistics.median(timings['analytic_seconds'])
    return dict(samples=samples, vertices=len(model.W), columns=len(columns),
        old_median_seconds=old, new_median_seconds=new, speedup=old/new,
        max_abs_difference_from_central_fd=maximum, relative_frobenius_difference=relative,
        original_full_compact_forwards_per_linearization=126,
        analytic_linearization_contexts_per_linearization=1, timings=timings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', nargs='+', type=int, default=[64, 476])
    parser.add_argument('--vertices', type=int, default=132)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(*args.samples, args.vertices, args.repeats) < 1:
        parser.error('Sizes and repeats must be positive')
    if args.output is not None and args.output.exists():
        parser.error('Use a new output path; preserve previous evidence')
    with tempfile.TemporaryDirectory() as temporary, threadpool_limits(limits=1, user_api='blas'):
        model = synthetic_model(Path(temporary)/'synthetic.npz', args.vertices)
        report = dict(schema='rinkspace.synthetic-compact-derivative-benchmark.v1',
            scope='63-column compact COCO operator only, synthetic model, CPU; no camera, footwear, sparse solver or fitting',
            target_hardware_measured=False, gpu_measured=False, dtype='float64',
            blas_threads=1, python=platform.python_version(), platform=platform.platform(),
            cases=[measure(model, n, args.repeats) for n in args.samples])
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text+'\n', encoding='utf8')
    print(text)


if __name__ == '__main__':
    main()
