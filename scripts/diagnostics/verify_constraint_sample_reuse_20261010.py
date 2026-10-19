"""Small original FD/constraints fixture; no event graph or benchmark."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import pickle
import sys
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.solver_runtime import install
install()

import numpy as np
from structured_programme_source_candidate import human_stroke_truth as human
from structured_programme_source_candidate import human_constraint_dependency_fd as fd
from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
from structured_programme_source_candidate.paired_programme_blocks import StructuredProgrammeBlock


@dataclass(frozen=True)
class Calibration:
    value: float


class Domain:
    def __init__(self):
        self.admissions = []

    def validate_parameters(self, world):
        world = np.asarray(world)
        self.admissions.append(world.tobytes())
        if not np.isfinite(world).all() or world[0] <= 0.:
            raise ValueError('original fixture world domain')


class Chart:
    def __init__(self, optics, mount):
        self.indices = np.array([optics])
        self.mount_indices = np.array([mount])
        self.coordinate_chart = SimpleNamespace(core_q=lambda q: np.asarray(q)[:34])
        self.calls = 0

    def calibration(self, base, q):
        self.calls += 1
        return Calibration(base.value + float(q[self.indices[0]]) + float(q[self.mount_indices[0]]))


class Bundle:
    def __init__(self, stamps):
        self.index = {p: i for i, p in enumerate(stamps)}
        self.calibration = Calibration(2.)
        self.exposures = []

    def camera(self, local, exposure):
        self.exposures.append(exposure)
        start = 4 + 4 * self.index[exposure]
        return np.r_[local[:3], local[start:start + 4], self.calibration.value]


def blocks(domain):
    result = []
    for stamps, xyz, start, optics, mount in (((10, 20, 30), (0, 1, 2), 6, 26, 28),
                                             ((40, 50), (3, 4, 5), 18, 27, 29)):
        block = StructuredProgrammeBlock.__new__(StructuredProgrammeBlock)
        block.bundle = Bundle(stamps)
        block.optics_base_calibration = block.bundle.calibration
        block.optics_chart = Chart(optics, mount)
        block.geometry = domain
        block.geometry_parameter_indices = np.array([30, 31])
        block.xyz_slots = np.array(xyz)
        block.start = start
        block.q0 = np.zeros(4 + 4 * len(stamps))
        result.append(block)
    a, b = result
    return OrderedDict(((10, a), (40, b), (20, a), (50, b), (30, a))), result


def one_record(owner, q, item, cameras, *, factor_cameras=None):
    factor = item['factor']
    camera = factor_cameras[id(factor)][item['row']['actual_pts']]
    world = np.asarray(q)[owner.world_indices]
    value = human._confirmed_pure_query_memo(owner, item, q, camera, world,
                                           lambda: np.array([camera.sum() + world.sum()]))
    return value, np.array([-100.]), np.array([100.]), dict(id=item['row']['id'], original_query=True)


def fixture():
    domain = Domain()
    mapping, distinct = blocks(domain)
    state = SimpleNamespace(geometry=domain, structured_programme_coordinate_chart=SimpleNamespace(core_count=34))
    factor = NativeBoardEdgeFactor.__new__(NativeBoardEdgeFactor)
    factor.blocks = mapping
    factor.core_count = factor.parameter_count = 34
    owner = human.UserConfirmedSourceTruth.__new__(human.UserConfirmedSourceTruth)
    owner.state = state
    owner.blocks = mapping
    owner.parameter_count = 34
    owner.world_indices = np.array([30, 31])
    owner.active = np.array([26, 30])
    owner.height_columns = np.array([2, 5])
    owner.steps = {26: 1e-6, 30: 1e-6}
    owner.contract = dict(authority=dict(original=True))
    owner.records = [dict(row=dict(id=f'row{i}', pts=p, actual_pts=p,
                                   uv=np.array([[1., 2.]]), curve=dict(kind='board_top'),
                                   original_history=dict(trace=list(range(100)))),
                          radius=np.array([1.]), normals=np.array([[1., 0.]]),
                          bounds=np.array([[-1., 1.]]), factor=factor,
                          original_support_operator='fixture_original_query', carrier_size=1)
                     for i, p in enumerate((10, 40, 20))]
    owner._one_record_carrier = MethodType(one_record, owner)
    return owner, distinct, domain


def state_fingerprint(owner, distinct, domain):
    return dict(calibrations=[asdict(b.bundle.calibration) for b in distinct],
                camera_world_cache_keys=[list(b._native_camera_valid_worlds) for b in distinct],
                exposures=[list(b.bundle.exposures) for b in distinct],
                domain_admissions=list(domain.admissions),
                source=pickle.dumps([human._truth_record_input_bytes(dict(row=i['row'],radius=i['radius'],normals=i['normals'],bounds=i['bounds'])) for i in owner.records],protocol=5))


def byte_equal(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and (a.shape, a.dtype.str, a.tobytes()) == (b.shape, b.dtype.str, b.tobytes())
    if isinstance(a, dict):
        return list(a) == list(b) and all(byte_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(byte_equal(x, y) for x, y in zip(a, b))
    return a == b


def query(camera, world, uv, normals, bounds, **kwargs):
    normal = np.full(len(uv), camera.sum() + np.asarray(world).sum())
    tangent = np.full(len(uv), camera[0] - np.asarray(world).sum())
    return dict(signed_normal_px=normal, coverage_tangent_px=tangent,
                residual=np.c_[normal, tangent].ravel())


def human_fixture():
    confirmed, distinct, domain = fixture()
    owner = human.HumanStrokeTruth.__new__(human.HumanStrokeTruth)
    for name in ('state', 'blocks', 'parameter_count', 'world_indices', 'active', 'height_columns', 'steps'):
        setattr(owner, name, getattr(confirmed, name))
    for item in confirmed.records:
        item['uv'] = item['row']['uv']
        item['tangent_bounds'] = item['bounds'].copy()
        item['factor'] = SimpleNamespace(_world=lambda q, curve: np.asarray(q)[owner.world_indices].reshape(1, 2))
    owner.records = owner.full_stroke_records = confirmed.records
    return owner, distinct, domain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    q = np.ones(34)
    q[:6] = [1., 2., 3., 4., 5., 6.]
    q[26:32] = [1., 2., 3., 4., 5., 6.]
    selected = np.array([26, 30])
    source_pickle_calls = [0]
    original_pickle = pickle.dumps
    def counted_pickle(value, *args, **kwargs):
        if isinstance(value, dict) and list(value) == ['row', 'radius', 'normals', 'bounds', 'original_support_operator']:
            source_pickle_calls[0] += 1
        return original_pickle(value, *args, **kwargs)
    pickle.dumps = counted_pickle
    owner, distinct, domain = fixture()
    baseline = owner.constraints(q, selected, return_step_carrier_response=True)
    baseline_source_serializations = source_pickle_calls[0]
    baseline_state = state_fingerprint(owner, distinct, domain)
    baseline_local_calls = sum(b.optics_chart.calls for b in distinct)
    key_reference = [human._confirmed_record_memo_key(owner, q, item) for item in owner.records]
    human.finite_stroke_queries = query
    howner, hdistinct, hdomain = human_fixture()
    hbaseline = fd.dependent_carrier_jacobian(howner, q)
    hbaseline_admissions = list(hdomain.admissions)
    hbaseline_exposures = [list(b.bundle.exposures) for b in hdistinct]

    from calibration.solver_runtime.constraint_sample_reuse_runtime import (
        install_constraint_sample_reuse_runtime, _CAMERA_SAMPLE, _LITERAL_CONSTRUCTION,
        _camera_sample_scope, _original_exposure_cameras)
    receipt = install_constraint_sample_reuse_runtime()
    owner2, distinct2, domain2 = fixture()
    source_pickle_calls[0] = 0
    candidate = owner2.constraints(q, selected, return_step_carrier_response=True)
    candidate_source_serializations = source_pickle_calls[0]
    candidate_state = state_fingerprint(owner2, distinct2, domain2)
    assert byte_equal(baseline, candidate), 'original full constraints/step response changed'
    assert byte_equal(baseline_state, candidate_state), 'camera/world/source final state changed'
    candidate_local_calls = sum(b.optics_chart.calls for b in distinct2)
    assert candidate_local_calls < baseline_local_calls
    assert candidate_source_serializations < baseline_source_serializations
    assert candidate_source_serializations == 2 * len(owner2.records)
    assert all(byte_equal(a, human._confirmed_record_memo_key(owner2, q, item)) for a, item in zip(key_reference, owner2.records))
    assert _CAMERA_SAMPLE.get() is None and _LITERAL_CONSTRUCTION.get() is None
    howner2, hdistinct2, hdomain2 = human_fixture()
    hcandidate = fd.dependent_carrier_jacobian(howner2, q)
    assert byte_equal(hbaseline.data, hcandidate.data) and byte_equal(hbaseline.indices, hcandidate.indices) and byte_equal(hbaseline.indptr, hcandidate.indptr)
    assert hbaseline_admissions == hdomain2.admissions
    assert hbaseline_exposures == [list(b.bundle.exposures) for b in hdistinct2]

    # Source mutation rejects the complete construction before returning rows.
    mutation_owner, _, _ = fixture()
    def mutate(owner, q, item, cameras, **kwargs):
        result = one_record(owner, q, item, cameras, **kwargs)
        if not getattr(owner, '_mutated', False):
            item['row']['original_history']['trace'].append(-1)
            owner._mutated = True
        return result
    mutation_owner._one_record_carrier = MethodType(mutate, mutation_owner)
    try:
        mutation_owner.constraints(q, selected)
        raise AssertionError('mutable source descriptor was accepted')
    except RuntimeError as error:
        assert 'literal source inputs changed' in str(error)
    assert _CAMERA_SAMPLE.get() is None and _LITERAL_CONSTRUCTION.get() is None

    # A control BaseException must cross the wrapper as the identical object.
    from calibration.solver_runtime.solver_control_runtime import SourceControlStop
    stop = SourceControlStop(dict(reason='fixture control'))
    stop_owner, _, _ = fixture()
    def stopped(*args, **kwargs):
        raise stop
    stop_owner._one_record_carrier = stopped
    try:
        stop_owner.constraints(q, selected)
        raise AssertionError('control was swallowed')
    except SourceControlStop as caught:
        assert caught is stop
    assert _CAMERA_SAMPLE.get() is None and _LITERAL_CONSTRUCTION.get() is None

    # Separate q, binding mutation and separate scope cannot reuse mutable local.
    from structured_programme_source_candidate.native_camera_local import native_camera_local
    check_owner, check_blocks, _ = fixture()
    with _camera_sample_scope():
        first = _original_exposure_cameras(check_owner.blocks, q, native_camera_local)
        second = _original_exposure_cameras(check_owner.blocks, q, native_camera_local)
        assert byte_equal(first, second)
        assert sum(b.optics_chart.calls for b in check_blocks) == 2
        changed = q.copy(); changed[26] += .1
        _original_exposure_cameras(check_owner.blocks, changed, native_camera_local)
        assert sum(b.optics_chart.calls for b in check_blocks) == 4
        check_blocks[0].bundle.calibration = Calibration(-123.)
        _original_exposure_cameras(check_owner.blocks, changed, native_camera_local)
        assert sum(b.optics_chart.calls for b in check_blocks) == 5
    with _camera_sample_scope():
        _original_exposure_cameras(check_owner.blocks, q, native_camera_local)
    assert sum(b.optics_chart.calls for b in check_blocks) == 7

    report = dict(passed=True, fixture='original StructuredProgrammeBlock.camera_local/local and original FD/constraints; tiny query/bundle/geometry stand-ins, no event graph',
                  confirmed_constraint_matrix_lower_report_and_step_response_byte_equal=True,
                  human_dependency_FD_CSR_payload_byte_equal=True,
                  every_original_world_admission_and_exposure_camera_call_preserved=True,
                  final_calibration_values_world_cache_and_source_bytes_equal=True,
                  baseline_camera_local_calls=baseline_local_calls, candidate_camera_local_calls=candidate_local_calls,
                  baseline_literal_source_serializations=baseline_source_serializations,
                  candidate_literal_source_serializations_including_exit_authentication=candidate_source_serializations,
                  original_literal_key_bytes_equal=True, source_mutation_rejected=True,
                  identical_control_BaseException_propagates=True,
                  different_q_binding_changes_and_scope_exit_do_not_share_mutable_local=True,
                  helper_receipt=receipt, actual_production_speedup_pending=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k != 'helper_receipt'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
