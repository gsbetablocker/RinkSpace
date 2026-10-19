"""Small original-minimal-clone fixture; no source graph, actor pool or solve."""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from calibration.solver_runtime import install
from calibration.solver_runtime import shared_actor_geometry_payload_runtime as helper


def finite_records(geometry, q):
    return geometry._original_motion_far_board_role_binding.records


class FixtureRole:
    def value(self):
        return self.geometry.current_q


class FixtureGeometry:
    pass


class FixtureBlock:
    pass


def digest(value):
    array = np.asarray(value)
    return dict(shape=list(array.shape), dtype=array.dtype.str,
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    install(verify=False)
    import z190_process_source_evaluation as source
    geometry = FixtureGeometry()
    geometry.current_q = np.arange(98, dtype=np.float64)/13
    # Protocol 5 passes large ndarray data to write() as PickleBuffer. Small
    # arrays alone do not exercise the actual production transport interface.
    geometry.large_array = np.arange(65537, dtype=np.float64).reshape(-1, 1)
    geometry.source_sigma = np.array([2., 2.], dtype='<f8')
    role = FixtureRole()
    role.geometry = geometry
    role.records = [dict(source_uv=np.arange(12, dtype='<f8').reshape(6, 2),
                         source_sigma=geometry.source_sigma, block='first')]
    role.bound_value = role.value
    geometry._original_motion_far_board_role_binding = role
    geometry.finite_board_records = partial(finite_records, geometry)
    geometry.self_cycle = [geometry, role]
    block = FixtureBlock()
    block.name = 'first'
    block.geometry = geometry
    block._original_motion_far_board_role_binding = role
    block.geometry_array_alias = geometry.current_q
    block.record_alias = role.records[0]
    block.prepared = object()  # Original _minimal_pickle explicitly drops it.
    block.bundle = SimpleNamespace(motion_kernel=SimpleNamespace(latest_checks=[1, 2]),
        profile_weights=np.array([[1., .5]], dtype=np.float64),
        ia=np.array([0, 1], dtype=np.int32),
        source_uv=role.records[0]['source_uv'], priors=lambda *args: ())
    paths = []
    try:
        original_payload, _ = source._minimal_pickle(block)
        original = pickle.loads(original_payload)
        common_path = args.output/'fixture.common.pickle'
        paths.append(common_path)
        common, memo = helper._write_pickle(common_path, geometry)

        def dump(value, *, protocol):
            assert protocol == 5
            path = args.output/'fixture.private.pickle'
            paths.append(path)
            return helper._write_pickle(path, value, common_memo=memo)

        minimal, original_sha = helper._compile_minimal(source._minimal_pickle, dump)
        private, modules = minimal(block)
        descriptor = dict(common=common, private=private,
                          original_aliases=helper._original_aliases(block))
        one, two = helper._load_actor(descriptor), helper._load_actor(descriptor)
        assert one.geometry is not two.geometry and one.geometry is not geometry
        checks = []
        for loaded in (one, two):
            g = loaded.geometry
            r = loaded._original_motion_far_board_role_binding
            assert r is g._original_motion_far_board_role_binding
            assert r.geometry is g and r.bound_value.__self__ is r
            assert g.self_cycle[0] is g and g.self_cycle[1] is r
            assert g.finite_board_records.args[0] is g
            assert loaded.synchronize_course.args[0] is loaded
            assert loaded.geometry_array_alias is g.current_q
            assert loaded.record_alias is r.records[0]
            assert loaded.bundle.source_uv is r.records[0]['source_uv']
            assert r.records[0]['source_sigma'] is g.source_sigma
            for before, after in ((original.geometry.current_q, g.current_q),
                    (original.geometry.large_array, g.large_array),
                    (original.bundle.profile_weights, loaded.bundle.profile_weights),
                    (original.bundle.source_uv, loaded.bundle.source_uv),
                    (original.bundle.ia, loaded.bundle.ia),
                    (original.geometry.source_sigma, g.source_sigma)):
                assert digest(before) == digest(after)
            assert loaded.prepared is None
            assert loaded.bundle.motion_kernel.latest_checks == []
            assert loaded.bundle._isolated_cache_heightless_prior is True
            assert 'priors' not in vars(loaded.bundle)
            checks.append('complete_numeric_SHA_and_loop_aliases_PASS')
        one.geometry.current_q[0] = 999
        assert digest(two.geometry.current_q) == digest(geometry.current_q)
        # Only load-site AST compilation; no original worker is executed.
        helper._compile_worker(source)
        saved = Path(private['path']).read_bytes()
        Path(private['path']).write_bytes(saved+b'x')
        try:
            helper._load_actor(descriptor)
        except helper.SharedActorGeometryPayloadError:
            corrupt_rejected = True
        else:
            raise AssertionError('Changed private payload SHA accepted')
        Path(private['path']).write_bytes(saved)
        from calibration.solver_runtime.source_actor_payload_runtime import install_source_actor_payload_runtime
        install_source_actor_payload_runtime(report_directory=args.output/'original-file-install')
        installer = helper.install_shared_actor_geometry_payload_runtime(
            report_directory=args.output/'install')
        assert installer['cross_q_or_cross_start_reuse'] is False
        report = dict(schema='rinkspace.shared-actor-current-geometry-small-check.v1',
            original_minimal_clone_sha256=original_sha,
            helper_sha256=hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest(),
            exact_original_minimal_clone_executed=True,
            original_single_actor_bytes=len(original_payload),
            common_bytes=common['bytes'], private_bytes=private['bytes'],
            shared_objects=private['common_object_references'],
            every_loaded_actor_independent_from_other_actor_and_parent=True,
            checks=checks, tampered_complete_payload_rejected=corrupt_rejected,
            original_worker_and_start_AST_sites_compiled=True,
            installed_after_current_original_file_transport=True,
            whole_graph_solver_or_process_pool_started=False)
        (args.output/'shared_actor_geometry_small_check.json').write_text(
            json.dumps(report, sort_keys=True, indent=2)+'\n', encoding='utf8')
        print(json.dumps(report, sort_keys=True))
    finally:
        for path in paths:
            path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
