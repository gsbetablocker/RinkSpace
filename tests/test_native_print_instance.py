"""Installed print-chart/native course consistency; synthetic, never fit evidence."""
import ctypes as c
import os
import unittest

import numpy as np

from rinkspace.core.native import NativeCore, NativeError


@unittest.skipUnless(os.environ.get("RINKSPACE_CORE_LIBRARY"), "explicit native library required")
class NativePrintInstance(unittest.TestCase):
    def setUp(self):
        self.core = NativeCore(os.environ["RINKSPACE_CORE_LIBRARY"])
        self.board = [.5, .25, .1, .12, 1.85, .02]

    def assert_same_mapping(self, p, vertices, stations):
        chart = np.asarray(self.core.sample_print_chart(p, vertices, stations))
        mapped = self.core.board_points(self.board, chart[:, 0], chart[:, 1])
        direct = self.core.sample_path(self.board, 5, p, vertices, stations)
        np.testing.assert_array_equal(mapped, direct)
        return chart, np.asarray(direct)

    def test_rotated_scaled_chart_and_explicit_template_orientation(self):
        # Source UV identity is retained by the caller; this is its explicit
        # chart orientation (photo-right/down become decreasing station/z).
        source_uv = [[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0., 0.]]
        vertices = [[1-u, 1-v] for u, v in source_uv]
        p = [.6, .004, .04, .08, .009]
        stations = [0., .125, .25, .375, .5, .75, 1.]
        chart, _ = self.assert_same_mapping(p, vertices, stations)
        np.testing.assert_array_equal(chart[0], chart[-1])
        self.assertGreater(chart[0, 0], chart[2, 0])
        self.assertGreater(chart[2, 1], chart[4, 1])
        self.assertEqual(source_uv[0], [0., 0.])

    def test_all_piece_joins_and_closing_seam_are_continuous(self):
        eps = 1e-7
        breaks = self.core.board_course_chart(self.board)
        for join in breaks:
            with self.subTest(join=join):
                p = [join-.01, .003, 0., .02, .009]
                chart, xyz = self.assert_same_mapping(
                    p, [[0., 1.], [1., 1.]], [.5-eps/.02, .5, .5+eps/.02])
                np.testing.assert_allclose(chart[:, 0], [join-eps, join, join+eps], atol=1e-15)
                self.assertLess(np.linalg.norm(xyz[2]-xyz[0]), 2.01*eps)
        # The installed chart stays unwrapped across the closing seam.
        chart, _ = self.assert_same_mapping(
            [breaks[-1]-.01, .003, 0., .02, .009], [[0., 0.], [1., 0.]], [0., 1.])
        self.assertGreater(chart[-1, 0], breaks[-1])
        self.assertLess(self.core.sample_print_chart([-.01, .003, 0., .02, .009], [[0., 0.]], [0.])[0][0], 0.)

    def test_invalid_dimensions_and_material_input_fail(self):
        for index in (3, 4):
            for value in (0., -.01, float("nan"), float("inf")):
                p = [.1, .003, 0., .02, .009]
                p[index] = value
                with self.subTest(index=index, value=value):
                    with self.assertRaises(NativeError):
                        self.core.sample_print_chart(p, [[0., 0.], [1., 1.]], [0.])
                    with self.assertRaises(NativeError):
                        self.core.sample_path(self.board, 5, p, [[0., 0.], [1., 1.]], [0.])
        for vertices in ([], [[0., float("nan")]]):
            with self.assertRaises(NativeError):
                self.core.sample_print_chart([.1, .003, 0., .02, .009], vertices, [0.])
        with self.assertRaises(ValueError):
            self.core.sample_print_chart([.1, .003, 0., .02, .009], [[0., 0., 0.]], [0.])

    def test_per_point_failures_are_nan_not_clamped(self):
        p = (c.c_double*5)(.1, .003, 0., .02, .009)
        vertices = (c.c_double*4)(0., 0., 1., 1.)
        stations = (c.c_double*4)(0., -.01, 1.01, float("nan"))
        out = (c.c_double*8)()
        status = (c.c_int*4)()
        result = self.core._lib.rs_sample_print_chart(p, 5, vertices, 4, stations, 4, out, 8, status, 4)
        self.assertEqual(result, 0)
        self.assertEqual(list(status), [0, 3, 3, 1])
        np.testing.assert_array_equal(list(out[:2]), [.1, .003])
        self.assertTrue(np.isnan(list(out[2:])).all())
        with self.assertRaisesRegex(NativeError, "row 1"):
            self.core.sample_print_chart(list(p), [[0., 0.], [1., 1.]], list(stations))

    def test_board_height_support_remains_enforced_after_chart_mapping(self):
        p = [.1, .019, 0., .02, .009]
        chart = self.core.sample_print_chart(p, [[0., 1.]], [0.])
        self.assertGreater(chart[0][1], self.board[5])
        with self.assertRaises(NativeError):
            self.core.board_points(self.board, [chart[0][0]], [chart[0][1]])
        with self.assertRaises(NativeError):
            self.core.sample_path(self.board, 5, p, [[0., 1.]], [0.])


class NativePrintCapability(unittest.TestCase):
    def test_missing_additive_capability_has_clear_error(self):
        core = object.__new__(NativeCore)
        core._lib = object()
        with self.assertRaisesRegex(ValueError, "lacks installed print-chart sampling capability"):
            core.sample_print_chart([], [], [])


if __name__ == "__main__":
    unittest.main()
