"""Original-pixel static-ray factor conditional on one *exact* optical centre.

This factor transfers the source Brown14 ray direction into the target camera
without intersecting a board, ice plane, or invented world point. The common
centre cancels. It is suitable only after an external exact-fixed-centre rig
has established that both physical Vector11 states have identical centres.
It cannot test whether the real camera centre moved, and its zero centre
Jacobians are not derivatives of a moving-centre camera model.
"""

from __future__ import annotations

try:
    from .runtime.bootstrap import add_geometry_dependencies
except ImportError:
    from runtime.bootstrap import add_geometry_dependencies

add_geometry_dependencies()
import cv2
import numpy as np

try:
    from .projection import from_state, project_fast
except ImportError:
    from projection import from_state, project_fast


class FixedCenterRayPair:
    """Transfer corresponding native pixels under one exactly shared centre.

    xy0 and xy1 are real, independently read Nx2 native source pixels.
    camera*_template are independent 25-vectors with fixed Brown14 tails;
    OpenCV tilt tauX/tauY (the last two coefficients) must be zero. The first
    eleven camera entries are supplied by each conditioned Vector11 state via
    projection.from_state. Residuals are target native pixels, unwhitened.
    """

    def __init__(self, xy0, xy1, camera0_template, camera1_template):
        self.xy0 = np.asarray(xy0, np.float64).copy()
        self.xy1 = np.asarray(xy1, np.float64).copy()
        if (self.xy0.ndim != 2 or self.xy0.shape[1] != 2
                or self.xy1.shape != self.xy0.shape or len(self.xy0) == 0
                or not np.isfinite(self.xy0).all()
                or not np.isfinite(self.xy1).all()
                or np.any(self.xy0 < 0) or np.any(self.xy1 < 0)
                or np.any(self.xy0 >= (1920, 1080))
                or np.any(self.xy1 >= (1920, 1080))):
            raise ValueError("Expected equal nonempty original 1920x1080 native UV pairs")
        self.camera0_template = self._template(camera0_template)
        self.camera1_template = self._template(camera1_template)

    @staticmethod
    def _template(value):
        camera = np.asarray(value, np.float64)
        if (camera.shape != (25,) or not np.isfinite(camera).all()
                or np.any(np.abs(camera[23:25]) > 1e-12)):
            raise ValueError("Independent Brown25 templates require zero fixed tilt")
        return camera.copy()

    @staticmethod
    def _states(states):
        if len(states) != 2:
            raise ValueError("Ray pair requires source and target physical Vector11")
        source, target = (np.asarray(state, np.float64) for state in states)
        if (source.shape != (11,) or target.shape != (11,)
                or not np.isfinite(source).all()
                or not np.isfinite(target).all()):
            raise ValueError("Ray pair requires two finite physical Vector11 states")
        if not np.array_equal(source[:3], target[:3]):
            raise ValueError("Fixed-centre ray transfer requires exactly equal physical centres")
        return source, target

    @staticmethod
    def _world_directions(camera, pixels):
        """Invert the independent Brown14+skew source convention."""
        fx, fy = np.exp(camera[6:8])
        unskewed = np.ascontiguousarray(pixels.copy())
        unskewed[:, 0] -= camera[10] * (pixels[:, 1] - camera[9]) / fy
        k = np.array([[fx, 0., camera[8]],
                      [0., fy, camera[9]],
                      [0., 0., 1.]], np.float64)
        if np.max(np.abs(camera[11:23])) <= 1e-14:
            normalized = np.column_stack(
                ((unskewed[:, 0] - camera[8]) / fx,
                 (unskewed[:, 1] - camera[9]) / fy))
        else:
            normalized = cv2.undistortPoints(
                unskewed.reshape(-1, 1, 2), k,
                np.ascontiguousarray(camera[11:25])).reshape(-1, 2)
        rotation = cv2.Rodrigues(np.ascontiguousarray(camera[3:6]))[0]
        return np.ascontiguousarray(
            np.column_stack((normalized, np.ones(len(pixels)))) @ rotation)

    def _predicted(self, states, *, target_jacobian=False):
        source, target = self._states(states)
        camera0 = from_state(source, self.camera0_template)
        camera1 = from_state(target, self.camera1_template)
        # C is irrelevant to a central ray rotation. Explicit zero centres
        # prevent floating-point cancellation at a large arbitrary world XYZ.
        camera0[:3] = 0.
        camera1[:3] = 0.
        directions = self._world_directions(camera0, self.xy0)
        if not np.isfinite(directions).all():
            raise ValueError("Source Brown14 inverse produced a nonfinite ray")
        source_uv, source_depth = project_fast(camera0, directions)
        if (not np.isfinite(source_uv).all() or np.any(source_depth <= 0)
                or np.max(np.linalg.norm(source_uv - self.xy0, axis=1)) > .05):
            raise ValueError("Source Brown14 inverse failed native-pixel round trip")
        projected = project_fast(camera1, directions,
                                 with_jacobian=target_jacobian)
        uv, depth = projected[:2]
        if not np.isfinite(uv).all() or np.any(depth <= 0):
            raise ValueError("Static ray is not forward-visible in target camera")
        return (uv, projected[2]) if target_jacobian else uv

    def predicted(self, states):
        """Return Nx2 target native UV without making a scene attachment."""
        return self._predicted(states)

    def _raw_residual(self, states):
        return (self._predicted(states) - self.xy1).reshape(-1)

    def _raw_jacobians(self, states):
        source, target = self._states(states)
        _, target_jacobian = self._predicted((source, target),
                                             target_jacobian=True)
        source_jacobian = np.zeros((2 * len(self.xy0), 11), np.float64)
        for axis in range(3, 11):
            step = 1e-6
            high, low = source.copy(), source.copy()
            high[axis] += step
            low[axis] -= step
            hi = self._predicted((high, target))
            lo = self._predicted((low, target))
            source_jacobian[:, axis] = ((hi - lo) / (2 * step)).reshape(-1)
        target_jacobian = target_jacobian.copy()
        target_jacobian[:, :3] = 0.
        if (not np.isfinite(source_jacobian).all()
                or not np.isfinite(target_jacobian).all()):
            raise ValueError("Fixed-centre ray Jacobian is nonfinite")
        return [source_jacobian, target_jacobian]
