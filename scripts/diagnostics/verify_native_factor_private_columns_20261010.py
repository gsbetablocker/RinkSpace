"""Tiny original typed cameras/factors/FD with genuine Windows spawn.

The material model is a declared numeric fixture, not an event or a graph.
No production benchmark or programme reconstruction is performed here.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.solver_runtime import install
install()

import numpy as np
from calibration.solver_runtime.absolute_geometry_fd_pool_candidate import AbsoluteGeometryGroupPool
from calibration.solver_runtime.native_factor_columns_runtime import install_native_factor_columns_runtime
from calibration.solver_runtime.constraint_sample_reuse_runtime import install_constraint_sample_reuse_runtime
from calibration.solver_runtime.tail_world_memo_runtime import install_tail_world_memo_runtime
from structured_programme_source_candidate.paired_programme_blocks import StructuredProgrammeBlock
from structured_programme_source_candidate.paired_programme_chart import (
    PairedProgrammeCoordinateChart, PairedProgrammeMountOpticsChart)
from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor, _source_optical_support
from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
from z190_fixed_lever_xyz_source_bundle import FixedLeverXYZSourceBundle
from z190_physical_camera import SharedCalibration
from z190_continuous_source_bundle import SCENE_UNIT_M
from wide_anchor_candidate.full_brown_projection import project_central_valid_full_brown


class NumericCurrent:
    scene_unit_m = SCENE_UNIT_M
    parameter_count = 66
    shared_hockey_curling_axis = False

    def __init__(self):
        self._materials = {(7, 'left_vertical_xyz'): (np.zeros(4), np.linspace(0., 1., 4))}

    def effective_world(self, values):
        return np.asarray(values, float).copy()

    def material(self, number, u, v, world):
        self.validate_parameters(world)
        # Numeric fixture only. The native source and FD operator are real.
        xyz = np.c_[np.full(len(v), world[0]), np.full(len(v), world[1]), np.asarray(v)] / self.scene_unit_m
        return xyz, None

    def validate_parameters(self, values):
        values = np.asarray(values, float)
        if values.shape != (66,) or not np.isfinite(values).all() or values[3] <= 0.:
            raise ValueError('Numeric fixture radius/world outside original declared fixture domain')


class NumericOwner:
    parameter_count = 66

    def __init__(self):
        self.current = NumericCurrent()

    def validate_parameters(self, values):
        return self.current.validate_parameters(values)


def array_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def factors(pool):
    chart = PairedProgrammeCoordinateChart()
    q = np.zeros(chart.core_count)
    xyz = np.array([0, 1, 2])
    world = np.arange(30, 96)
    q[xyz] = [0., 0., 10.]
    q[3:7] = [np.pi, 0., 0., np.log(1200.)]
    q[world] = np.ones(66)
    q[world[:4]] = [1., 2., 0., 3.]
    owner = pool.parent_geometry
    state = SimpleNamespace(geometry=owner, structured_programme_coordinate_chart=chart,
                            geometry_parameter_indices=world)
    block = StructuredProgrammeBlock.__new__(StructuredProgrammeBlock)
    block.xyz_slots, block.start, block.q0 = xyz, 3, np.zeros(8)
    block.optics_chart = PairedProgrammeMountOpticsChart('tiny-original-camera', (20, 21, 22),
        chart.core_count, mount_indices=(23, 24, 25))
    block.optics_base_calibration = SharedCalibration('tiny-original-camera', (0., 0., 10. / SCENE_UNIT_M))
    block.geometry, block.geometry_parameter_indices = owner, world
    block._native_camera_valid_worlds = OrderedDict()
    block._absolute_geometry_fd_pool = pool
    block.bundle = FixedLeverXYZSourceBundle.__new__(FixedLeverXYZSourceBundle)
    block.bundle.calibration, block.bundle.index = block.optics_base_calibration, {10: 0}
    camera_columns = np.r_[xyz, np.arange(3, 7), np.arange(20, 26)]
    result = []
    for cls in (NativeIceCurveFactor, NativeBoardEdgeFactor):
        factor = cls.__new__(cls)
        factor.state, factor.geometry, factor.blocks = state, owner.current, {10: block}
        factor.parameter_count = factor.core_count = len(q)
        factor.world_indices, factor.world_parameter_count, factor.unit = world, 66, SCENE_UNIT_M
        factor.camera_columns = np.unique(camera_columns)
        factor.camera_steps = {int(column): 1e-6 for column in camera_columns}
        factor._ice_material_view = None
        factor._physical_course_scope_overrides = {}
        factor._printed_contour_materials = {}
        factor.packet_path = Path('numeric-fixture-only.json')
        factor._cache = OrderedDict()
        curve = dict(kind='hockey_circle', feature_id='C0') if cls is NativeIceCurveFactor else dict(
            kind='advertisement_edge', advertisement_number=7, edge='left_vertical')
        camera = factor._cameras(q)[10]
        xyz_values = factor._world(q, curve)
        selected = xyz_values[[13, 64, 105]] if cls is NativeIceCurveFactor else xyz_values
        uv = project_central_valid_full_brown(camera, selected)[0] + np.array([.2, -.1])
        row = dict(id=cls.__name__ + ':tiny', pts=10, actual_pts=10, curve=curve,
                   uv=uv, sigma=np.ones(len(uv)), feature_id='C0' if cls is NativeIceCurveFactor else 'Ad7',
                   normal=np.tile([1., 0.], (len(uv), 1)), point_correspondence=False)
        if cls is NativeIceCurveFactor:
            row.update(source_optical_normal_support_authorized=True,
                       normal_interval_px=np.tile([-.05, .05], (len(uv), 1)))
            row['_source_optical_support'] = _source_optical_support(row)
        else:
            row['normal_interval'] = np.tile([-.05, .05], (len(uv), 1))
        factor.rows, factor.held = [row], []
        factor.scales = np.ones(2 * len(uv))
        factor.quadratic = np.ones(len(factor.scales), bool)
        factor.row_count = len(factor.scales)
        factor.active = np.unique(np.r_[camera_columns, world])
        factor.contract = dict(packet_sha256='numeric-fixture', schema='numeric-fixture',
                               fit_ids=[row['id']], held_ids=[])
        result.append(factor)
    return q, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from prepared_cache_helpers import install_mappingproxy_pickle
    install_mappingproxy_pickle()
    install_constraint_sample_reuse_runtime()
    install_tail_world_memo_runtime()
    private = NumericOwner()
    payload = pickle.dumps(private, protocol=5)
    snapshot = args.output / 'native_columns_numeric_fixture_owner.pkl'
    snapshot.write_bytes(payload)
    descriptor = dict(path=str(snapshot), sha256=hashlib.sha256(payload).hexdigest(), serialized_bytes=len(payload))
    pool = AbsoluteGeometryGroupPool(descriptor, workers=2, parent_geometry=private)
    try:
        q, sources = factors(pool)
        original = [factor.block(q) for factor in sources]
        install_native_factor_columns_runtime(report_directory=args.output)
        checks = []
        previous_executor = None
        for factor, expected in zip(sources, original):
            factor._cache.clear()
            actual = factor.block(q)
            assert all(array_equal(getattr(expected, name), getattr(actual, name)) for name in ('r', 'quadratic', 'scales'))
            assert expected.J.shape == actual.J.shape
            assert all(array_equal(getattr(expected.J, name), getattr(actual.J, name)) for name in ('data', 'indices', 'indptr'))
            assert expected.row_ledger == actual.row_ledger
            receipt = pool._rinkspace_native_columns_idle_receipt
            assert receipt['executor_identity'] == id(pool.executor) and receipt['pending_jobs'] == 0
            assert not pool._rinkspace_native_columns_active
            assert pool._rinkspace_native_columns_pending_jobs == 0
            if previous_executor is not None:
                assert previous_executor is pool.executor
            previous_executor = pool.executor
            checks.append(dict(factor=type(factor).__name__, r_J_CSR_scales_roles_and_ledger_byte_equal=True,
                               original_complete_dependency_columns=len(factor._source_dependency_fd_report['contexts']),
                               complete_scalar_rows=len(actual.r), idle_receipt=receipt))
        report = dict(schema='rinkspace.tiny-real-native-spawn-interface-check.v1',
            passed=True, private_process_workers=2, real_typed_camera_factor_projection_and_original_FD=True,
            material_model='explicit numeric fixture; no production/event graph',
            workers_reused_between_native_factors=True, checks=checks,
            original_support_and_fixed_nearest_segment_branches_preserved=True,
            actual_production_equivalence_and_speedup_not_claimed=True)
        (args.output / 'native_columns_small_spawn_check.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        pool.close()


if __name__ == '__main__':
    main()
