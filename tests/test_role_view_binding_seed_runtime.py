from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_compaction_binding_reindex_runtime import fixture
from calibration.solver_runtime.role_block_binding_runtime import _make_view
from calibration.solver_runtime.role_view_binding_seed_runtime import seed_owned_role_view_cache


def two_block_fixture(packet):
    state = fixture(packet)
    role = state.original_motion_far_board_roles
    role.records[1]['source_role_block'] = 'main2'
    role.records[1]['source_pts'] = 13
    cameras = role.source_cameras
    role._finish_certificate()
    role.source_cameras[11] = cameras[11]
    role.source_cameras[13] = cameras[11].copy()
    role.current_binding(state.geometry, state.q)
    return state


class RoleViewBindingSeedTest(unittest.TestCase):
    def test_real_owned_projection_alias_hit_q_miss_and_stale_metadata_rejection(self):
        with TemporaryDirectory() as temp:
            packet = Path(temp) / 'original_packet.json'
            packet.write_text(json.dumps({'schema': 'fixture'}), encoding='utf8')
            state = two_block_fixture(packet)
            role = state.original_motion_far_board_roles
            full_cache = role._current_binding_cache
            old_records = deepcopy(full_cache[1]['records'])
            view, reason = _make_view(role, 'main1')
            self.assertIsNone(reason)
            receipt = seed_owned_role_view_cache(role, view, [0])
            self.assertTrue(receipt['seeded'], receipt)
            self.assertFalse(receipt['full_worker_numeric_cache_released'])
            self.assertIs(role._current_binding_cache, full_cache)
            cached = view._current_binding_cache[1]
            for name in ('planes', 'plane_jacobian', 'extent_world', 'extent_world_jacobian', 'selected_course_facets'):
                expected = full_cache[1][name][[0]]
                self.assertEqual(cached[name].shape, expected.shape)
                self.assertEqual(cached[name].dtype, expected.dtype)
                self.assertEqual(cached[name].tobytes(), expected.tobytes())
                self.assertFalse(np.shares_memory(cached[name], full_cache[1][name]))
            for name in ('z_range_scene', 'z_range_world_jacobian'):
                self.assertEqual(cached[name].tobytes(), full_cache[1][name].tobytes())
                self.assertFalse(np.shares_memory(cached[name], full_cache[1][name]))
            self.assertIs(cached['records'][0], full_cache[1]['records'][0])
            self.assertIs(cached['records'][0]['material_identity'], role.records[0]['physical_material'])
            self.assertEqual(full_cache[1]['records'], old_records)
            hit = view.current_binding(state.geometry, state.q)
            self.assertIs(hit, cached)
            self.assertEqual(view.exact_cache_statistics['binding_hits'], 1)
            self.assertEqual(view.exact_cache_statistics['binding_misses'], 0)
            reference, _ = _make_view(role, 'main1')
            changed = view.current_binding(state.geometry, np.array([.1]))
            expected = reference.current_binding(state.geometry, np.array([.1]))
            self.assertEqual(view.exact_cache_statistics['binding_misses'], 1)
            for name in ('planes', 'plane_jacobian', 'extent_world', 'extent_world_jacobian',
                         'selected_course_facets', 'z_range_scene', 'z_range_world_jacobian'):
                self.assertEqual(changed[name].tobytes(), expected[name].tobytes())
            self.assertEqual(changed['records'], expected['records'])

            worker_view, _ = _make_view(role, 'main2')
            with patch('multiprocessing.current_process', return_value=SimpleNamespace(name='rink-source-fixture')):
                receipt = seed_owned_role_view_cache(role, worker_view, [1])
            self.assertTrue(receipt['seeded'], receipt)
            self.assertTrue(receipt['full_worker_numeric_cache_released'])
            self.assertIsNone(role._current_binding_cache)
            self.assertIsNotNone(worker_view._current_binding_cache)

            stale = two_block_fixture(packet)
            stale_role = stale.original_motion_far_board_roles
            # Existing same-facet shortcut can leave a historical source record
            # in a numeric cache. Seeding must reject even index-only staleness.
            stale_role.records[0]['active_global_bundle_index'] = 99
            stale_view, _ = _make_view(stale_role, 'main1')
            old = stale_role._current_binding_cache
            receipt = seed_owned_role_view_cache(stale_role, stale_view, [0])
            self.assertFalse(receipt['seeded'])
            self.assertIsNone(stale_view._current_binding_cache)
            self.assertIs(stale_role._current_binding_cache, old)


if __name__ == '__main__':
    unittest.main()
