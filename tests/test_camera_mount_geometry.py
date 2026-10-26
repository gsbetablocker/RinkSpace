"""Independent rigid-motion and gauge checks of the existing native head.

P is the common-normal endpoint on the pan axis, not an extra base bearing.
These tests do not certify a fitted physical installation or its dimensions.
"""
import ctypes as c
import os
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from rinkspace.core.native import NativeCore, D, N


@unittest.skipUnless(os.environ.get("RINKSPACE_G3_LIBRARY"), "explicit native build required")
class CameraMountGeometry(unittest.TestCase):
    def setUp(self):
        self.core = NativeCore(os.environ["RINKSPACE_G3_LIBRARY"])
        world = Rotation.from_rotvec([.021, -.037, .013]).as_matrix()
        level = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        self.rig = np.zeros(38)
        self.rig[:9] = (level @ world.T).ravel()
        self.rig[9:12] = world @ [0., 0., 1.]
        self.rig[12:15] = [1., 0., 0.]
        self.rig[15:18] = world @ [.2, -.4, .8]
        self.rig[18:24] = [.01, -.15, .10, .03, .02, -.005]
        self.rig[24:28] = [800., 2., 960., 540.]
        self.retract_fn = self.core._lib.rs_camera_retract
        self.retract_fn.argtypes = [D, N, D, N, D, N]
        self.retract_fn.restype = c.c_int

    def pose(self, alpha, beta, zoom=0., rig=None):
        rig = self.rig if rig is None else rig
        controls = np.tile([alpha, beta, np.log(rig[24]) + zoom], (8, 1))
        value = self.core.evaluate_motion_camera(rig, controls, -.04, .02, [0.])
        return value["rotation"][0], value["centre"][0]

    def tilt_line(self, rotation):
        n, b, P = self.rig[9:12], self.rig[12:15], self.rig[15:18]
        axis = rotation.T @ b
        h = np.cross(n, axis)
        h /= np.linalg.norm(h)
        return P + self.rig[21] * h, axis

    def test_pan_is_rigid_rotation_about_one_fixed_axis(self):
        alpha0, beta = -.31, .24
        R0, C0 = self.pose(alpha0, beta, .6)
        Q0, q0 = self.tilt_line(R0)
        n, P = self.rig[9:12], self.rig[15:18]
        for alpha in (-.8, 0., .53):
            R, C = self.pose(alpha, beta, .6)
            Q, q = self.tilt_line(R)
            turn = Rotation.from_rotvec(-(alpha-alpha0)*n).as_matrix()
            np.testing.assert_allclose(C, P + turn @ (C0-P), atol=2e-14)
            np.testing.assert_allclose(Q, P + turn @ (Q0-P), atol=2e-14)
            np.testing.assert_allclose(q, turn @ q0, atol=2e-14)
            np.testing.assert_allclose(R, R0 @ turn.T, atol=2e-14)

    def test_tilt_rotates_camera_about_unchanged_tilt_line(self):
        alpha, beta0 = .47, -.18
        R0, C0 = self.pose(alpha, beta0, -.3)
        Q0, q0 = self.tilt_line(R0)
        for beta in (-.42, 0., .37):
            R, C = self.pose(alpha, beta, -.3)
            Q, q = self.tilt_line(R)
            turn = Rotation.from_rotvec(-(beta-beta0)*q0).as_matrix()
            np.testing.assert_allclose(Q, Q0, atol=2e-14)
            np.testing.assert_allclose(q, q0, atol=2e-14)
            np.testing.assert_allclose(C, Q0 + turn @ (C0-Q0), atol=2e-14)
            np.testing.assert_allclose(R, R0 @ turn.T, atol=2e-14)

    def test_P_Q_is_shortest_axis_connector_not_second_height(self):
        n, P = self.rig[9:12], self.rig[15:18]
        for alpha, beta in ((-.7, -.3), (0., 0.), (.6, .4)):
            R, _ = self.pose(alpha, beta)
            Q, q = self.tilt_line(R)
            self.assertLess(abs((Q-P) @ n), 1e-14)
            self.assertLess(abs((Q-P) @ q), 1e-14)
            self.assertAlmostEqual(np.linalg.norm(Q-P), abs(self.rig[21]), places=14)
            # A pan axis within theta of vertical bounds the derived vertical
            # part of the common normal by |a| sin(theta); no height variable.
            tilt = np.arccos(n[2])
            self.assertLessEqual(abs(Q[2]-P[2]), abs(self.rig[21])*np.sin(tilt)+1e-14)

    def test_zoom_changes_pupil_only_along_actual_optical_axis(self):
        R0, C0 = self.pose(.23, .17, 0.)
        for z in (-.8, .4, .9):
            R, C = self.pose(.23, .17, z)
            shift = self.rig[22]*z + self.rig[23]*z*z
            np.testing.assert_allclose(R, R0, atol=2e-14)
            np.testing.assert_allclose(R @ (C-C0), [0., 0., shift], atol=2e-14)
            np.testing.assert_allclose(self.tilt_line(R)[0], self.tilt_line(R0)[0], atol=2e-14)

    def test_independent_head_chart_has_no_extra_base_height_gauge(self):
        samples = [(a, b, z) for a in (-.6, .1, .7)
                   for b in (-.3, .1, .4) for z in (-.8, 0., .7)]
        def values(rig):
            result = []
            for sample in samples:
                R, C = self.pose(*sample, rig=rig)
                result.extend([*R.ravel(), *C])
            return np.asarray(result)
        # Finite differences are only an independent verification oracle.
        jac = []
        eps = 1e-6
        for j in range(15):
            ys = []
            for sign in (-1, 1):
                delta = np.zeros(27); delta[j] = sign*eps
                rig = np.empty(38)
                status = self.retract_fn(self.rig.ctypes.data_as(D), 38,
                    delta.ctypes.data_as(D), 27, rig.ctypes.data_as(D), 38)
                self.assertEqual(status, 0)
                ys.append(values(rig))
            jac.append((ys[1]-ys[0])/(2*eps))
        jac = np.asarray(jac).T
        singular = np.linalg.svd(jac, compute_uv=False)
        self.assertEqual(np.count_nonzero(singular > singular[0]*1e-7), 15)
        # Adding an independent height H along pan would be exactly duplicated
        # by a translation of P. Explicitly demonstrate the forbidden column.
        extra_height = np.tile(np.r_[np.zeros(9), self.rig[9:12]], len(samples))
        np.testing.assert_allclose(extra_height, jac[:, 6:9] @ self.rig[9:12], atol=2e-9)
        redundant = np.linalg.svd(np.column_stack([jac, extra_height]), compute_uv=False)
        self.assertEqual(np.count_nonzero(redundant > redundant[0]*1e-7), 15)


if __name__ == "__main__":
    unittest.main()
