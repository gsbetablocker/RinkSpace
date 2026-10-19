"""Analytic moving-centre board pair Jacobians for physical11/15 cameras.

The residual and all source/target visibility tests are inherited unchanged
from NativeBoardPair. Only the differentiable interior branch is differentiated;
an actual curved panel boundary raises instead of receiving a made-up gradient.
"""
from __future__ import annotations

import cv2
import numpy as np

try:
    from .native_board_pair import NativeBoardPair
    from .projection import from_state
    from .reference_board_pair import BoardIntersectionError
except ImportError:
    from native_board_pair import NativeBoardPair
    from projection import from_state
    from reference_board_pair import BoardIntersectionError


class ContinuousBrownBoardPair(NativeBoardPair):
    """Actual UV board factor with full 15-column source and target Jacobians.

    Physical15 is ``(C/100, rvec, log(fx), log(fy), cx/1000,
    cy/1000, skew/1000, k1, k2, p1, p2)``. Physical11 follows the existing
    NativeBoardPair callback exactly when both ends are 11-dimensional.
    """

    @staticmethod
    def _rotation(camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rotation, jac = cv2.Rodrigues(np.ascontiguousarray(camera[3:6]))
        if jac.shape == (3, 9):
            derivative = jac.reshape(3, 3, 3)
        elif jac.shape == (9, 3):
            derivative = jac.T.reshape(3, 3, 3)
        else:
            raise ValueError(f"Unexpected Rodrigues Jacobian shape {jac.shape}")
        return rotation, derivative

    @staticmethod
    def _opencv_k(camera: np.ndarray) -> np.ndarray:
        fx, fy = np.exp(camera[6:8])
        return np.array([[fx, 0., camera[8]],
                         [0., fy, camera[9]],
                         [0., 0., 1.]], np.float64)

    @classmethod
    def _source_ray_jacobian(cls, camera: np.ndarray, normalized: np.ndarray,
                             rotation: np.ndarray, rotation_jac: np.ndarray
                             ) -> np.ndarray:
        """Differentiate the inverse Brown ray at fixed original source UV.

        OpenCV translation-x/y columns on points (nx, ny, 1) give the exact
        forward-pixel 2x2 derivative. Implicit differentiation solves that
        small matrix against camera-intrinsic and Brown4 pixel derivatives.
        Sensor skew is applied after OpenCV, exactly as in project_fast.
        """
        count = len(normalized)
        local_ray = np.column_stack((normalized, np.ones(count)))
        projected, raw = cv2.projectPoints(
            np.ascontiguousarray(local_ray), np.zeros(3), np.zeros(3),
            cls._opencv_k(camera), np.ascontiguousarray(camera[11:25]))
        uv = projected.reshape(count, 2)
        jac = raw.reshape(count, 2, -1)
        fx, fy = np.exp(camera[6:8])
        skew = camera[10]
        distorted_y = (uv[:, 1]-camera[9])/fy

        pixel_xy = jac[:, :, 3:5].copy()
        pixel_xy[:, 0, :] += (skew/fy)*pixel_xy[:, 1, :]
        determinant = np.linalg.det(pixel_xy)
        stable = (np.isfinite(pixel_xy).all(axis=(1, 2))
                  & np.isfinite(determinant) & (np.abs(determinant) > 1e-9))
        if not np.all(stable):
            raise BoardIntersectionError("noninvertible_brown_pixel_jacobian",
                                         np.flatnonzero(~stable))

        explicit = np.zeros((count, 2, 15), np.float64)
        explicit[:, :, 6] = jac[:, :, 6]*fx
        explicit[:, :, 7] = jac[:, :, 7]*fy
        explicit[:, :, 8] = jac[:, :, 8]*1000.
        explicit[:, :, 9] = jac[:, :, 9]*1000.
        explicit[:, :, 11:15] = jac[:, :, 10:14]
        # u = u_cv + skew * (v_cv-cy)/fy.  The subtractions matter for
        # log(fy) and cy: the distorted normalized y does not depend on K.
        normalized_y_jac = explicit[:, 1, :].copy()
        normalized_y_jac[:, 9] -= 1000.
        normalized_y_jac[:, 7] -= fy*distorted_y
        normalized_y_jac /= fy
        explicit[:, 0, :] += skew*normalized_y_jac
        explicit[:, 0, 10] += 1000.*distorted_y
        inverse = np.linalg.solve(pixel_xy, -explicit[:, :, 6:15])

        ray_jac = np.zeros((count, 3, 15), np.float64)
        for axis in range(3):
            ray_jac[:, :, 3+axis] = local_ray @ rotation_jac[axis]
        ray_jac[:, :, 6:15] = np.einsum(
            "nkp,ki->nip", inverse, rotation[:2, :])
        return ray_jac

    @classmethod
    def _target_brown_jacobian(cls, camera: np.ndarray,
                               xyz: np.ndarray) -> np.ndarray:
        rotation, _ = cls._rotation(camera)
        translation = -rotation @ camera[:3]
        _, raw = cv2.projectPoints(
            np.ascontiguousarray(xyz), np.ascontiguousarray(camera[3:6]),
            translation, cls._opencv_k(camera),
            np.ascontiguousarray(camera[11:25]))
        result = raw[:, 10:14].reshape(len(xyz), 2, 4).copy()
        result[:, 0, :] += camera[10]/np.exp(camera[7])*result[:, 1, :]
        return result.reshape(2*len(xyz), 4)

    def _raw_jacobians(self, states):
        source, target = (np.asarray(value, np.float64) for value in states)
        if source.shape not in ((11,), (15,)) or target.shape not in ((11,), (15,)):
            raise ValueError("Board camera states must have 11 or 15 values")
        if source.shape == target.shape == (11,):
            return super()._raw_jacobians(states)

        # projected() retains NativeBoardPair's source first-hit and target
        # finite/visibility/occlusion checks at the current state.
        _, base_panel, target_core = self.projected([source, target], True)
        hits = self.source_hits(source)
        self._require(hits)
        count = len(self.xy0)
        if not np.array_equal(base_panel, hits.panel):
            raise ValueError("Source board panel changed within one callback")
        fraction = hits.panel_fraction
        smooth = (np.isfinite(fraction)
                  & (fraction > self.seam_fraction_margin)
                  & (fraction < 1.-self.seam_fraction_margin))
        at_start = np.isfinite(fraction) & (fraction <= self.seam_fraction_margin)
        at_end = np.isfinite(fraction) & (fraction >= 1.-self.seam_fraction_margin)
        previous = (hits.panel-1) % len(self._edges)
        following = (hits.panel+1) % len(self._edges)
        smooth |= at_start & self._same_local_plane(hits.panel, previous)
        smooth |= at_end & self._same_local_plane(hits.panel, following)
        if not np.all(smooth):
            raise BoardIntersectionError("nondifferentiable_panel_transition",
                                         np.flatnonzero(~smooth))

        source_camera = from_state(source, self.camera0_template)
        target_camera = from_state(target, self.camera1_template)
        rotation, rotation_jac = self._rotation(source_camera)
        # source_hits intersects camera rays of form [nx, ny, 1] @ R.
        # Reconstruct that normalized ray from its current first board hit;
        # this is the same inverse branch used by the inherited residual.
        camera_delta = (hits.xyz-source_camera[:3]) @ rotation.T
        distance = camera_delta[:, 2]
        stable_distance = np.isfinite(distance) & (distance > 1e-8)
        if not np.all(stable_distance):
            raise BoardIntersectionError("unstable_board_ray_distance",
                                         np.flatnonzero(~stable_distance))
        normalized = camera_delta[:, :2]/distance[:, None]
        rays = np.column_stack((normalized, np.ones(count))) @ rotation
        ray_jac = self._source_ray_jacobian(
            source_camera, normalized, rotation, rotation_jac)

        edge = self._edges[hits.panel]
        normal = np.column_stack((edge[:, 1], -edge[:, 0],
                                  np.zeros(count)))
        denominator = np.einsum("ni,ni->n", normal, rays)
        stable_plane = (np.isfinite(denominator) & (np.abs(denominator) > 1e-7))
        if not np.all(stable_plane):
            raise BoardIntersectionError("unstable_board_plane_derivative",
                                         np.flatnonzero(~stable_plane))
        # For a fixed current panel h=C+lambda*v and n·(h-P)=0:
        # dh=(I-v*n.T/(n·v)) (dC+lambda*dv). The normal and plane are
        # piecewise constant; curved seams were excluded above.
        tangent = (np.eye(3)[None, :, :]
                   - rays[:, :, None]*normal[:, None, :]/denominator[:, None, None])
        centre_jac = np.zeros((count, 3, 15), np.float64)
        centre_jac[:, :, :3] = 100.*np.eye(3)
        world_jac = np.einsum(
            "nij,njp->nip", tangent,
            centre_jac+distance[:, None, None]*ray_jac)
        # Translation of a world point is the negative of camera-centre
        # translation; target_core uses C/100, hence the factor of 1/100.
        image_world = -target_core.reshape(count, 2, 11)[:, :, :3]/100.
        source_full = np.einsum("nai,nip->nap", image_world,
                                world_jac).reshape(2*count, 15)
        target_full = np.column_stack((
            target_core, self._target_brown_jacobian(target_camera, hits.xyz)))
        result = [source_full[:, :len(source)], target_full[:, :len(target)]]
        if any(not np.isfinite(matrix).all() for matrix in result):
            raise ValueError("Continuous Brown board Jacobian is nonfinite")
        self.last_jacobian_counts = {
            "analytic_rows": count, "numeric_rows": 0,
            "brown_inverse_analytic_rows": count,
        }
        return result
