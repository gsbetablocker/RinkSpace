"""One real original absolute15/FD group, without a source graph or processes."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from calibration.solver_runtime import install
install()
from calibration.solver_runtime import absolute_geometry_fd_pool_candidate as source
from calibration.solver_runtime import absolute_material_value_runtime as material
from calibration.solver_runtime import camera_absolute_reuse_runtime as camera
from calibration.solver_runtime import camera15_group_prefetch_runtime as combined
from calibration.solver_runtime.completion_driven_absolute_fd_runtime import install_completion_driven_absolute_fd_runtime
from z190_source_camera_diagnostic import absolute_value_jac15


class SmallGeometry:
    parameter_count = 1
    def validate_parameters(self, world):
        if np.asarray(world).shape != (1,) or not np.isfinite(world).all():
            raise ValueError('Tiny physical world domain')
    def _key(self, row):
        return row['id']
    def source_parameter_activity(self, row, world):
        return np.ones(1, bool)
    def source_parameter_fields(self, row):
        return ('world',)
    def rebind(self, row, world):
        self.validate_parameters(world)
        points = np.asarray(row['world']).copy()
        points[:, 0] += world[0]
        return dict(row, world=points)


def nominal_failure(camera25, rows, jacobian=True):
    if jacobian:
        raise ValueError('original nominal exception')
    return absolute_value_jac15(camera25, rows, False)


def FD_failure(camera25, rows, jacobian=True):
    if not jacobian:
        raise ValueError('original FD exception')
    return absolute_value_jac15(camera25, rows, True)


@contextmanager
def owner(operator, geometry, snapshot, rows, camera25):
    bundle = SimpleNamespace(absolute_evaluator=operator, camera=lambda local, stamp:camera25.copy())
    pool = source.AbsoluteGeometryGroupPool(snapshot, workers=2, max_inflight=1,
        groups_per_job=16, parent_geometry=geometry)
    # One thread executes the exact native job; no real graph or process starts.
    pool.executor = ThreadPoolExecutor(max_workers=1)
    value = SimpleNamespace(name='tiny_real_group', absolute=operator, bundle=bundle,
        groups=[(101, rows)], _absolute_geometry_fd_pool=pool)
    reuse = dict(owner=value, results=[], arrays={}, position=0, hits=0, fallbacks=0)
    token = camera._SCOPE.set(reuse)
    try:
        yield value, pool
    finally:
        camera._SCOPE.reset(token)
        pool.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    install_completion_driven_absolute_fd_runtime()
    material._install_runtime()
    camera.install_camera_absolute_reuse_runtime()
    receipt = combined.install_camera15_group_prefetch_runtime()
    source.initialize_absolute_geometry_worker()
    geometry = SmallGeometry()
    snapshot = dict(path='tiny-preloaded-private-owner', sha256='tiny-owner-identity')
    source._PRIVATE_GEOMETRIES[(snapshot['path'], snapshot['sha256'])] = (geometry, set())
    rows = [dict(id='tiny-actual-original-point-operator',
        world=np.array([[0., 0., 0.], [1., 1., 0.]]),
        uv=np.array([[960.4, 540.3], [1200.2, 780.1]]),
        sigma=np.array([1.2, .9]), point_correspondence=True)]
    camera25 = np.zeros(25)
    camera25[2] = -5.
    camera25[6:8] = np.log(1200.)
    camera25[8:10] = (.96, .54)
    world = np.array([.12])
    moved = [geometry.rebind(row, world) for row in rows]
    reference_camera = absolute_value_jac15(camera25, moved, True)
    job = dict(material_owner=snapshot, ordinal=0, actual_pts=101,
        rows=rows, source_group_sha256=source._source_group_digest(rows),
        camera25=camera25, world66=world, absolute=absolute_value_jac15,
        step=1e-7, batched_projection=True)
    reference_FD = source._evaluate_one_absolute_geometry_group(job)['geo']
    with owner(absolute_value_jac15, geometry, snapshot, rows, camera25) as (value, pool):
        active = combined._begin(value, np.zeros(4), world, True)
        assert active is not None
        try:
            actual_camera = combined._capture(value.bundle, camera25, moved, True)
            actual_FD = [result['geo'].copy() for result in combined._geometry_spans(
                pool, [(101, rows, camera25)], world, value.absolute, 1e-7,
                batched_projection=True, group_count=1)][0]
            assert all(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
                       for a,b in zip(actual_camera, reference_camera))
            assert actual_FD.dtype == reference_FD.dtype and actual_FD.tobytes() == reference_FD.tobytes()
        finally:
            combined._finish(active, True)
        assert '_executor' not in vars(pool)
    exceptions = []
    for operator, expected, phase in [(nominal_failure, 'original nominal exception', 'nominal'),
                                      (FD_failure, 'original FD exception', 'FD')]:
        with owner(operator, geometry, snapshot, rows, camera25) as (value, pool):
            active = combined._begin(value, np.zeros(4), world, True)
            try:
                combined._capture(value.bundle, camera25, moved, True)
                list(combined._geometry_spans(pool, [(101, rows, camera25)], world,
                    value.absolute, 1e-7, batched_projection=True, group_count=1))
                raise AssertionError('Original error not delivered')
            except ValueError as error:
                assert str(error) == expected
                exceptions.append(phase)
            finally:
                combined._finish(active, False)
            assert '_executor' not in vars(pool)
    report = dict(schema='rinkspace.small-real-original-camera15-and-FD-group-check.v1',
        original_absolute15_and_FD_bytes_equal=True, original_nominal_and_FD_exception_stages_retained=exceptions,
        owned_feeder_futures_drained_and_executor_restored=True, original_caller_AST_installation_passed=True,
        no_full_source_graph_or_process_pool_started=True, helper_receipt=receipt,
        seconds=time.perf_counter()-started)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
