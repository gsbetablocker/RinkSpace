"""Two real original role rows; no programme/source graph is constructed."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from calibration.solver_runtime import runtime
runtime.install()
from structured_programme_source_candidate.original_motion_far_board_roles import (
    OriginalMotionFarBoardRoleBinding, PROGRAMME_REVIEW_SCHEMA)
from calibration.solver_runtime import compaction_binding_reindex_runtime as helper


class TinyCourse:
    def sample_perimeter(self, q, *, with_jacobian=False):
        bottom = np.array([[-2., -2., 0.], [2., -2., 0.], [2., 2., 0.], [-2., 2., 0.]])
        bottom[:, 0] += q[0]
        top = bottom.copy()
        top[:, 2] = 5.
        J = np.zeros((4, 3, 6))
        J[:, 0, 0] = 1.
        return bottom, top, J, J.copy()


class TinyCurrent:
    def __init__(self):
        self.course = TinyCourse()

    def print_course_binding(self, world):
        q = np.r_[world[0], np.zeros(5)]
        chain = np.zeros((6, 1))
        chain[0, 0] = 1.
        return self.course, q, chain


class TinyCalibration:
    def manifest(self):
        return {'source_clock': [11, 12], 'camera_aliases': [[11, 11]]}


def fixture(packet):
    role = OriginalMotionFarBoardRoleBinding.__new__(OriginalMotionFarBoardRoleBinding)
    role.packet_path = packet
    role.packet = {'schema': PROGRAMME_REVIEW_SCHEMA}
    role.records = [dict(original_global_bundle_index=i, source_role_block='main1',
        source_pts=11, target_pts=12, source_uv=[1000., 0.], target_uv=[900., 0.],
        physical_material={'original_source_surface': 'board'}, original_alias_evidence=[7, 8])
        for i in (7, 8)]
    role._finish_certificate()
    camera = np.zeros(25)
    camera[2] = 1.
    camera[6:8] = np.log(1000.)
    role.source_cameras[11] = camera
    geometry = SimpleNamespace(current=TinyCurrent(), parameter_count=1, scene_unit_m=1.)
    state = SimpleNamespace(geometry=geometry, q=np.zeros(1),
        geometry_parameter_indices=np.array([0]), original_motion_far_board_roles=role)
    block = SimpleNamespace(name='main1', bundle=SimpleNamespace(calibration=TinyCalibration()))
    block.local = lambda q: role.current_binding(geometry, q[state.geometry_parameter_indices])
    state.blocks = [block]
    role.current_binding(geometry, state.q)
    return state


def original_fixture_compact(state, *, changed_camera=False, later_failure=False):
    # Only this orchestration is a fixture; binding/finite rays/planes/J below
    # all use the unchanged actual source producer.
    state.original_diagnostic_calls = getattr(state, 'original_diagnostic_calls', 0) + 1
    role = state.original_motion_far_board_roles
    for row in role.records:
        row['active_global_bundle_index'] = 0
    role.records[1]['duplicate_runtime_representative_original_global_bundle_index'] = 7
    if changed_camera:
        role.source_cameras[11][2] += .25
    role._current_binding_cache = None
    if later_failure:
        raise ValueError('original later qualification failure')
    return state.blocks[0].local(state.q)


class CompactionBindingReindexTest(unittest.TestCase):
    def test_actual_original_binding_alias_bytes_and_original_failure(self):
        with TemporaryDirectory() as temp:
            packet = Path(temp) / 'original_packet.json'
            packet.write_text(json.dumps({'schema': PROGRAMME_REVIEW_SCHEMA}), encoding='utf8')
            baseline = fixture(packet)
            candidate = fixture(packet)
            old = candidate.original_motion_far_board_roles._current_binding_cache[1]
            old_records = deepcopy(old['records'])
            wrapped = helper._wrap(original_fixture_compact)
            expected = original_fixture_compact(baseline)
            actual = wrapped(candidate)
            self.assertEqual(candidate.original_diagnostic_calls, 1)
            self.assertEqual(actual['records'], expected['records'])
            self.assertEqual(old['records'], old_records)
            self.assertEqual(len(actual['records']), 2)
            self.assertEqual([row['global_bundle_index'] for row in actual['records']], [0, 0])
            for name in helper._ARRAY_FIELDS:
                self.assertEqual(actual[name].shape, expected[name].shape)
                self.assertEqual(actual[name].dtype, expected[name].dtype)
                self.assertEqual(actual[name].tobytes(), expected[name].tobytes())
                self.assertIs(actual[name], old[name])
            for record, row in zip(actual['records'], candidate.original_motion_far_board_roles.records):
                self.assertIs(record['material_identity'], row['physical_material'])
                self.assertIsNot(record['actual_source_record'], row)
            self.assertEqual(candidate.original_motion_far_board_roles.exact_cache_statistics['binding_misses'], 1)
            self.assertEqual(baseline.original_motion_far_board_roles.exact_cache_statistics['binding_misses'], 2)
            self.assertTrue(helper.snapshot_compaction_binding_reindex_runtime()['events'][-1]['reused_existing_numeric_binding'])

            changed = fixture(packet)
            reference = fixture(packet)
            a = wrapped(changed, changed_camera=True)
            b = original_fixture_compact(reference, changed_camera=True)
            self.assertEqual(a['records'], b['records'])
            for name in helper._ARRAY_FIELDS:
                self.assertEqual(a[name].tobytes(), b[name].tobytes())
            self.assertFalse(helper.snapshot_compaction_binding_reindex_runtime()['events'][-1]['reused_existing_numeric_binding'])

            failed = fixture(packet)
            with self.assertRaisesRegex(ValueError, '^original later qualification failure$'):
                wrapped(failed, later_failure=True)
            self.assertIsNone(failed.original_motion_far_board_roles._current_binding_cache)


if __name__ == '__main__':
    unittest.main()
