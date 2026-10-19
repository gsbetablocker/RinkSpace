"""Small original exception/replay check, without source graphs or processes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.diagnostics import check_camera15_group_prefetch_small_20261010 as fixture
from reference_board_pair import BoardIntersectionError

np = fixture.np
source, material, camera, combined = fixture.source, fixture.material, fixture.camera, fixture.combined
absolute = fixture.absolute_value_jac15


def board_nominal(camera25, rows, jacobian=True):
    if jacobian:
        raise BoardIntersectionError('original nominal board domain', [2, 5])
    return absolute(camera25, rows, False)


def board_FD(camera25, rows, jacobian=True):
    if not jacobian and rows[0]['id'].endswith('-1'):
        raise BoardIntersectionError('original FD board domain', [1, 4])
    return absolute(camera25, rows, jacobian)


def check_error(error, reason, indices):
    assert type(error) is BoardIntersectionError
    assert error.reason == reason
    assert error.indices.dtype == np.dtype(np.int32)
    assert error.indices.tolist() == indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    fixture.install_completion_driven_absolute_fd_runtime()
    material._install_runtime()
    camera.install_camera_absolute_reuse_runtime()
    receipt = combined.install_camera15_group_prefetch_runtime()
    source.initialize_absolute_geometry_worker()
    geometry = fixture.SmallGeometry()
    snapshot = dict(path='tiny-preloaded-private-owner', sha256='tiny-owner-identity')
    source._PRIVATE_GEOMETRIES[(snapshot['path'], snapshot['sha256'])] = (geometry, set())
    template = dict(id='tiny-original-point', world=np.array([[0., 0., 0.], [1., 1., 0.]]),
        uv=np.array([[960.4, 540.3], [1200.2, 780.1]]), sigma=np.array([1.2, .9]),
        point_correspondence=True)
    groups = [(101+i, [dict(template, id='tiny-original-point-'+str(i))]) for i in range(6)]
    camera25 = np.zeros(25)
    camera25[2] = -5.
    camera25[6:8] = np.log(1200.)
    camera25[8:10] = (.96, .54)
    world = np.array([.12])
    marker = combined._error_marker(BoardIntersectionError('marker original board domain', [2, 5]))
    assert pickle.loads(pickle.dumps(marker, protocol=5)) == marker
    assert set(marker) == {'type', 'message'}
    capture = combined._ORIGINAL_CAPTURE
    capture_calls = []
    def counted_capture(*args, **kwargs):
        capture_calls.append(threading.get_ident())
        return capture(*args, **kwargs)
    combined._ORIGINAL_CAPTURE = counted_capture
    try:
        with fixture.owner(board_nominal, geometry, snapshot, groups[0][1], camera25) as (value, pool):
            active = combined._begin(value, np.zeros(4), world, True)
            try:
                moved = [geometry.rebind(row, world) for row in groups[0][1]]
                combined._capture(value.bundle, camera25, moved, True)
                raise AssertionError('Original nominal board exception omitted')
            except BoardIntersectionError as error:
                check_error(error, 'original nominal board domain', [2, 5])
                assert capture_calls == [threading.get_ident()]
                transport = active[0].get(0)
                assert pickle.loads(pickle.dumps(transport, protocol=5))['camera15_error'] == transport['camera15_error']
            finally:
                combined._finish(active, False)
    finally:
        combined._ORIGINAL_CAPTURE = capture

    published = []
    with fixture.owner(board_FD, geometry, snapshot, groups[0][1], camera25) as (value, pool):
        value.groups = groups
        active = combined._begin(value, np.zeros(4), world, True)
        try:
            for result in combined._geometry_spans(pool,
                    [(stamp, rows, camera25) for stamp, rows in groups], world,
                    board_FD, 1e-7, batched_projection=True, group_count=len(groups)):
                published.append(result['ordinal'])
            raise AssertionError('Original FD board exception omitted')
        except BoardIntersectionError as error:
            check_error(error, 'original FD board domain', [1, 4])
            assert published == []
        finally:
            combined._finish(active, False)

    original_private = source._evaluate_one_absolute_geometry_group
    def worker_only_board_failure(job):
        if job['ordinal'] == 1:
            raise BoardIntersectionError('private-only board failure', [1, 4])
        return original_private(job)
    source._evaluate_one_absolute_geometry_group = worker_only_board_failure
    try:
        with fixture.owner(absolute, geometry, snapshot, groups[0][1], camera25) as (value, pool):
            value.groups = groups
            active = combined._begin(value, np.zeros(4), world, True)
            succeeded = False
            try:
                actual = list(combined._geometry_spans(pool,
                    [(stamp, rows, camera25) for stamp, rows in groups], world,
                    absolute, 1e-7, batched_projection=True, group_count=len(groups)))
                assert [result['ordinal'] for result in actual] == list(range(6))
                # The span iterator deliberately frees consumed arrays. Their
                # authentic original FD shapes/routes remain in original stats.
                assert all(result['stats']['scalar_rows'] == 4 for result in actual)
                assert [result['ordinal'] for result in actual
                    if result['stats']['actual_route'] == 'original_error_parent_replay'] == [1, 2]
                assert active[0].fd_parent_replays == 2
                assert pool.last_stats[1] is actual[1]['stats']
                assert pool.last_stats[2] is actual[2]['stats']
                succeeded = True
            finally:
                combined._finish(active, succeeded)
    finally:
        source._evaluate_one_absolute_geometry_group = original_private
    report = dict(schema='rinkspace.small-original-board-exception-replay-check.v1',
        standard_pickle_marker_and_worker_result_roundtrip=True,
        original_nominal_capture_called_in_parent=True,
        original_nominal_and_FD_BoardIntersectionError_type_reason_indices_preserved=True,
        original_16_group_failed_envelope_published_no_prefix=True,
        private_only_FD_failure_and_skipped_group_filled_by_original_parent_math=[1, 2],
        original_stats_objects_updated=True, no_full_source_graph_or_process_pool_started=True,
        helper_sha256=hashlib.sha256(Path(combined.__file__).read_bytes()).hexdigest(),
        seconds=time.perf_counter()-started)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
