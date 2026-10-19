"""Small original course/semantic producers; no event graph or solver load."""
from copy import deepcopy
import unittest

import numpy as np

from calibration.solver_runtime import install
from calibration.solver_runtime.wide_source_scene_runtime import _clone_scene

install()
from shared_board_course_candidate.shared_board_course_numeric_geometry import SharedBoardCourseNumericGeometry
from simple_wide_perspective_reset_candidate.differentiable_material_course_domain import DifferentiableBoardCourseGeometry
from wide_anchor_candidate.native_wide_source import world_curve


class WideNumericSceneTest(unittest.TestCase):
    def test_original_numeric_fields_curves_and_base_isolation(self):
        initial = np.array([30., 15., 4., 3., 2.7, 1.2])
        seed = SharedBoardCourseNumericGeometry.__new__(SharedBoardCourseNumericGeometry)
        seed.initial = initial.copy()
        seed._initial_scene = initial.copy()
        seed.scene_unit_m = 1.
        bottom, top, _, _, parts = seed._perimeter(initial)
        base = dict(
            parameters=dict(bounds=[-30., -15., 30., 15.], corner_radius=4.,
                            board_height=1.2, corner_radii={'left': 4., 'right': 4.},
                            corner_curve=dict(kind='four_congruent_symmetric_superellipse_quarters',
                                              tangent_extent_xy_scene_units=[4., 3.], exponent=2.7)),
            ice_boundary=dict(points_xyz=bottom.tolist(), parts_counterclockwise=parts),
            board_surface=dict(bottom_edge_xyz=bottom.tolist(), top_edge_xyz=top.tolist(),
                               panels=[dict(s_range=[0., 1.], z_range=[0., 1.2]) for _ in range(480)],
                               perimeter_length=1.),
            shared_geometry_checks=dict(checks={}, unknown_check_reasons={
                'four_corners_keep_one_symmetric_profile': 'unconfirmed'}),
            meshes=[dict(name='ice_surface', positions=bottom.tolist()),
                    dict(name='rink_boards', positions=np.vstack((bottom, top)).tolist())],
            ice_features=[dict(id='F1', kind='circle', center_xyz=[-7., 2., 0.], radius=1.2,
                               plane_normal_xyz=[0., 0., 1.]),
                          dict(id='lane1', kind='polygon', polygon_xyz=[[-10., -2., 0.],
                               [10., -2., 0.], [10., 2., 0.], [-10., 2., 0.]], open_edges=[])],
        )
        course = SharedBoardCourseNumericGeometry(base, 1.)
        frozen = deepcopy(course.base)
        candidate = _clone_scene(SharedBoardCourseNumericGeometry.scene)
        targets = [dict(kind='board_side', side='right_end', edge='top', continue_corner='lower_right'),
                   dict(kind='circle_outline', feature_id='F1'),
                   dict(kind='polygon_supporting_line', feature_id='lane1', index=0),
                   dict(kind='polygon_outline', feature_id='lane1')]
        points = [initial, initial + [0.3, 0.2, 0.1, 0.05, 0.2, 0.03]]
        for q in points:
            original = course.scene(q, rebuild_grid=False)
            reduced = candidate(course, q, rebuild_grid=False)
            for key in ('parameters', 'ice_boundary', 'boundary_validity_margins', 'shared_geometry_checks'):
                self.assertEqual(original.get(key), reduced.get(key))
            for field in ('bottom_edge_xyz', 'top_edge_xyz'):
                self.assertEqual(original['board_surface'][field], reduced['board_surface'][field])
            for target in targets:
                a, b = world_curve(original, target), world_curve(reduced, target)
                self.assertEqual((a.shape, a.dtype.str, a.tobytes()), (b.shape, b.dtype.str, b.tobytes()))
            self.assertEqual(course.base, frozen)
        for callable_ in (course.scene, lambda q: candidate(course, q, rebuild_grid=False)):
            with self.assertRaisesRegex(ValueError, 'Positive tangency straights/radii/height and convex p>1'):
                callable_([30., 15., 4., 3., 1., 1.2])
        # The actual registered factory reclasses the original course exactly
        # this way. Inherited scene + contact/top policy produce the same bytes.
        course.__class__ = DifferentiableBoardCourseGeometry
        for q in points:
            original = course.scene(q, rebuild_grid=False)
            reduced = candidate(course, q, rebuild_grid=False)
            for field in ('bottom_edge_xyz', 'top_edge_xyz'):
                a = np.asarray(original['board_surface'][field])
                b = np.asarray(reduced['board_surface'][field])
                self.assertEqual((a.shape, a.dtype.str, a.tobytes()), (b.shape, b.dtype.str, b.tobytes()))
            for target in targets:
                a, b = world_curve(original, target), world_curve(reduced, target)
                self.assertEqual((a.shape, a.dtype.str, a.tobytes()), (b.shape, b.dtype.str, b.tobytes()))
            self.assertEqual(course.base, frozen)


if __name__ == '__main__':
    unittest.main()
