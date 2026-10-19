"""One original role producer fixture, with the existing two helpers installed."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from calibration.solver_runtime import install
from calibration.solver_runtime import role_block_binding_runtime as helper


class Course:
    def sample_perimeter(self, q, with_jacobian=False):
        a, b, height = q[0], q[1], q[5]
        bottom = np.array([[-a, -b, 0.], [a, -b, 0.], [a, b, 0.], [-a, b, 0.]])
        top = bottom.copy()
        top[:, 2] = height
        if not with_jacobian:
            return bottom, top, None
        jac = np.zeros((4, 3, 6))
        jac[:, 0, 0] = [-1, 1, 1, -1]
        jac[:, 1, 1] = [-1, -1, 1, 1]
        jtop = jac.copy()
        jtop[:, 2, 5] = 1.
        return bottom, top, jac, jtop


class Current:
    def print_course_binding(self, world):
        chain = np.zeros((6, 3))
        chain[[0, 1, 5], [0, 1, 2]] = 1.
        return Course(), np.array([world[0], world[1], 0., 0., 0., world[2]]), chain


def same_arrays(actual, expected):
    return (actual.shape == expected.shape and actual.dtype == expected.dtype
            and actual.tobytes() == expected.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    install(verify=False)
    from calibration.solver_runtime.role_record_index_runtime import install_role_record_index_runtime
    from calibration.solver_runtime.role_binding_value_reuse_runtime import install_role_binding_value_reuse_runtime
    install_role_record_index_runtime()
    install_role_binding_value_reuse_runtime()
    from structured_programme_source_candidate import original_motion_far_board_roles as roles
    cls = roles.OriginalMotionFarBoardRoleBinding
    code_before = {name: getattr(cls, name).__code__ for name in
                   ('current_binding', '_current_source_facets', '_build_current_binding')}
    dispatch_code = roles._role_finite_board_records.__code__
    role = cls.__new__(cls)
    role.binding_id = 'fixture-role-certificate'
    role.contract = dict(original_source_count=6, sigma_unchanged=True)
    role.records = []
    for i, (name, stamp) in enumerate((('first', 10), ('second', 20), ('first', 10),
                                     ('second', 20), ('first', 11), ('second', 21))):
        role.records.append(dict(source_role_block=name, source_pts=stamp, target_pts=stamp+1,
            original_global_bundle_index=i, active_global_bundle_index=100+i,
            source_uv=[960.+i*14., 570.+i*5.], target_uv=[962.+i*14., 572.+i*5.],
            source_sigma_px=[2., 2.], split='fit', physical_material='finite-board-fixture',
            additional_source_role_qualification_packets=[dict(sha256='same-alias-'+str(i))]))
    role.indices = np.array([r['original_global_bundle_index'] for r in role.records], int)
    role.source_uv = np.array([r['source_uv'] for r in role.records], float)
    role.source_row_groups = {stamp: np.array([i for i, r in enumerate(role.records)
                            if r['source_pts'] == stamp], int) for stamp in (10, 11, 20, 21)}
    camera = np.zeros(25)
    camera[:3] = [0., 0., .7]
    camera[3] = np.pi/2
    camera[6:8] = np.log(1000.)
    camera[8:10] = [960., 540.]
    camera[10] = 2.
    camera[11:13] = [1e-4, -1e-6]
    role.source_cameras = {stamp: camera.copy() for stamp in role.source_row_groups}
    role._current_binding_cache = role._facet_course_cache = None
    role.exact_cache_statistics = dict.fromkeys(('binding_hits', 'binding_misses',
        'facet_group_hits', 'facet_group_misses', 'unchanged_facet_binding_hits'), 0)
    geometry = SimpleNamespace(parameter_count=3, scene_unit_m=1., current=Current(),
        _original_motion_far_board_role_binding=role)
    records_before = copy.deepcopy(role.records)
    world = np.array([4., 2., 1.4])
    checks = []
    scopes = []
    for name in ('first', 'second'):
        # Simulate two independently hydrated worker graphs.
        block = copy.deepcopy(SimpleNamespace(name=name, geometry=geometry,
            _original_motion_far_board_role_binding=role))
        scope = helper._ActorScope(block)
        scope.role.current_binding = scope.current_binding
        scopes.append(scope)
        assert scope.view.source_cameras is scope.role.source_cameras
        for value in (world, world+np.array([.2, -.1, .15])):
            expected = cls.current_binding(scope.role, scope.geometry, value)
            selected = np.array([i for i, record in enumerate(expected['records'])
                                 if record['source_role_block'] == name], int)
            actual = scope.role.current_binding(scope.geometry, value)
            for field in ('planes', 'plane_jacobian', 'extent_world',
                          'extent_world_jacobian', 'selected_course_facets'):
                assert same_arrays(actual[field], expected[field][selected]), (name, field)
            for field in ('z_range_scene', 'z_range_world_jacobian'):
                assert same_arrays(actual[field], expected[field]), (name, field)
            assert actual['records'] == [expected['records'][i] for i in selected]
            assert len(actual['records']) == 3 and len(expected['records']) == 6
            checks.append(dict(block=name, world=value.tolist(), all_arrays_and_records_bitexact=True))
        assert scope.role.records == records_before and scope.role.contract == role.contract
    scopes[0].role.source_cameras[10][0] = .25
    assert scopes[1].role.source_cameras[10][0] == 0.
    assert role.source_cameras[10][0] == 0.
    assert roles._role_finite_board_records.__code__ is dispatch_code
    assert all(getattr(cls, name).__code__ is code for name, code in code_before.items())
    invalid = scopes[0]
    del invalid.role.source_cameras[10]
    try:
        invalid.role.current_binding(invalid.geometry, world)
    except ValueError as exc:
        assert 'Actual current source camera must bind' in str(exc)
        domain_error_preserved = True
    else:
        raise AssertionError('Original current source error swallowed')
    mixed = copy.deepcopy(role)
    mixed.records[1]['source_pts'] = 10
    mixed.source_row_groups[10] = np.array([0, 1, 2], int)
    mixed.source_row_groups[20] = np.array([3], int)
    view, reason = helper._make_view(mixed, 'first')
    assert view is None and 'shares a source stamp across blocks' in reason
    report = dict(schema='rinkspace.owned-actor-role-block-original-producer-small-check.v1',
        helper_sha256=hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest(),
        installed_original_index_and_role_value_reuse_helpers=True,
        original_dispatch_class_current_binding_selection_build_code_objects_unchanged=True,
        checks=checks, independent_worker_and_parent_mutable_graphs=True,
        source_camera_or_domain_error_preserved=domain_error_preserved,
        mixed_block_source_stamp_falls_back_to_original_full_binding=True,
        parent_contract_qualification_alias_and_role_records_unchanged=True,
        whole_graph_or_solver_or_process_pool_started=False)
    (args.output/'role_block_binding_small_check.json').write_text(
        json.dumps(report, indent=2, sort_keys=True)+'\n', encoding='utf8')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
