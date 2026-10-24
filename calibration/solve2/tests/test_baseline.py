"""First-principles projection and clock/constraint semantics checks."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rinksolve.baseline import (BaselineInitialization, camera_from_original,
    extract_scene_paths, plane_homography, project_homography,
    project_world_pinhole, rotation_exp, validate_native_table)


class IndependentProjectionTests(unittest.TestCase):
    def test_rodrigues_noncommuting_axis_and_small_angle(self):
        rotation = rotation_exp(np.array([0., 0., np.pi / 2]))
        np.testing.assert_allclose(rotation @ [1, 0, 0], [0, 1, 0], atol=1e-14)
        for vector in ([1e-12, -2e-12, 3e-12], [.3, -.7, .2]):
            rotation = rotation_exp(np.asarray(vector))
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(rotation), 1, places=14)

    def test_analytic_camera_center_sign_rotation_and_plane(self):
        # Camera at Z=10, looking down.  Positive world Y projects upward.
        camera = np.zeros(25)
        camera[:3] = [0, 0, 10]
        camera[3:6] = [np.pi, 0, 0]
        camera[6:8] = np.log([100, 200])
        camera[8:11] = [7, 11, 5]
        center, rotation, intrinsic = camera_from_original(camera)
        xyz = np.array([[1., 2., 0.], [-3., 1., 0.], [1., 2., 12.]])
        uv, valid = project_world_pinhole(center, rotation, intrinsic, xyz)
        np.testing.assert_array_equal(valid, [True, True, False])
        np.testing.assert_allclose(uv[:2], [[16., -29.], [-23.5, -9.]], atol=1e-12)
        h = plane_homography(center, rotation, intrinsic)
        plane, good = project_homography(h, xyz[:2, :2])
        np.testing.assert_array_equal(good, [True, True])
        np.testing.assert_allclose(plane, uv[:2], atol=1e-12)

    def test_original_brown_fields_are_inactive(self):
        camera = np.zeros(25)
        camera[:3] = [0, 0, 10]
        camera[3] = np.pi
        camera[6:8] = np.log([100, 200])
        before = camera_from_original(camera)
        camera[11:] = np.linspace(-.3, .7, 14)
        after = camera_from_original(camera)
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)

    def test_native_pts_requests_never_round_or_interpolate(self):
        package = BaselineInitialization(Path('.'), {}, {'native_pts': np.array([256256, 256512, 256768])}, {}, [])
        self.assertEqual(package.package_index(np.int64(256512)), 1)
        for value in (256257, 20.02, 256256., '256256', True, 0, 575488):
            with self.assertRaises(ValueError):
                package.package_index(value)


class SourceAndSceneTests(unittest.TestCase):
    def native_table(self):
        pts = np.array([256256, 256512, 256768], dtype=np.int64)
        table = {"native_pts": pts, "pts": pts.copy(), "time_seconds": pts / 12800,
                 "camera25": np.zeros((3, 25)), "world_xy_to_native_pixel_H": np.tile(np.eye(3), (3, 1, 1)),
                 "bridge_mode": np.array([0, 1, 0], dtype=np.uint8), "direct_mapping": np.array([True, False, True]),
                 "shot_index": np.ones(3, dtype=np.int16),
                 "anchor_left_official_pts": np.array([256256, 256256, 256768]),
                 "anchor_right_official_pts": np.array([256256, 256768, 256768]),
                 "anchor_left_legacy_pts": np.array([100, 100, 300]),
                 "anchor_right_legacy_pts": np.array([100, 300, 300])}
        clock = {"first_pts": 256256, "last_pts_inclusive": 256768, "pts_step": 256, "frames": 3}
        source = {"native_fps": "50/1", "time_base": "1/12800"}
        return table, clock, source

    def test_native_clock_and_interpolation_provenance(self):
        table, clock, source = self.native_table()
        validate_native_table(table, clock, source)
        for key, value in (("native_pts", np.array([256256, 256513, 256768])),
                           ("time_seconds", np.array([20.02, 20.04, 21.])),
                           ("direct_mapping", np.ones(3, dtype=bool)),
                           ("shot_index", np.array([1, 2, 1])),
                           ("anchor_left_official_pts", np.array([256256, 256768, 256768]))):
            broken = dict(table)
            broken[key] = value
            with self.assertRaises(ValueError, msg=key):
                validate_native_table(broken, clock, source)
        with self.assertRaises(ValueError):
            validate_native_table(table, clock, {"native_fps": "60/1", "time_base": "1/15360"})

    def test_draw_vertices_do_not_become_landmarks_or_close_open_lane(self):
        # Explicit paint segments must override the historical polygon proxy.
        scene = {"ice_boundary": {"id": "boundary", "points_xyz": [[0, 0, 0], [1, 0, 0], [1, 1, 0]]},
                 "board_surface": {"bottom_edge_xyz": [[0, 0, 0], [1, 0, 0]], "top_edge_xyz": [[0, 0, 1], [1, 0, 1]]},
                 "ice_grid": {"cells": []},
                 "ice_features": [{"id": "lane1", "surface_id": "ice_surface", "kind": "polygon", "polygon_xyz": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                                   "closed": False, "open_edges": [3], "paint_segments_xyz": [[[0, 0, 0], [1, 0, 0]], [[1, 0, 0], [1, 1, 0]], [[1, 1, 0], [0, 1, 0]]]}],
                 "unobserved_standard_reference_ice_features": [{"id": "unseen_standard"}]}
        paths, audit = extract_scene_paths(scene)
        lane = [p for p in paths if p['group'] == 'ice']
        self.assertEqual(len(lane), 3)
        self.assertTrue(all(not p['closed'] for p in lane))
        self.assertTrue(all(p['initial_geometry_only'] and p['constraint_semantics_unassigned'] and not p['samples_are_physical_landmarks'] for p in paths))
        self.assertEqual(audit['excluded_unobserved_standard_ids'], ['unseen_standard'])
        self.assertFalse(audit['unparsed'])


if __name__ == '__main__':
    unittest.main()
