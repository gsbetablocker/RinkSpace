"""Analytic source Jacobians for static ice/board plane correspondences.

The ordinary PairConstraint keeps its original numerical derivative route as
the fallback for inverse Brown rays outside a stable differentiable branch.
Both the zero-Brown and physical15 paths retain its measured source pixels,
plane identities, residual and central-domain checks.
"""
from __future__ import annotations

import numpy as np

try:
    from .runtime.bootstrap import add_geometry_dependencies
except ImportError:
    from runtime.bootstrap import add_geometry_dependencies

add_geometry_dependencies()
import cv2

try:
    from .track_sequence import PairConstraint
    from .projection import from_state, project_fast, require_central_projection
    from .continuous_brown_board_pair import ContinuousBrownBoardPair
    from .reference_board_pair import BoardIntersectionError
except ImportError:
    from track_sequence import PairConstraint
    from projection import from_state, project_fast, require_central_projection
    from continuous_brown_board_pair import ContinuousBrownBoardPair
    from reference_board_pair import BoardIntersectionError


class SurfacePairConstraint(PairConstraint):
    """A source-conditioned planar pair whose Jacobian follows the hit point."""

    def jacobians(self, states: list[np.ndarray]) -> list[np.ndarray]:
        source, target = states
        camera = from_state(source, self.camera_template)
        if (np.shape(source) == (15,) and np.shape(target) == (15,)
                and not self.fixed_mount_rays):
            analytic = self._brown15_jacobians(states, camera)
            if analytic is not None:
                return analytic
        if self.fixed_mount_rays or np.max(np.abs(camera[11:25])) > 1e-12:
            return super().jacobians(states)
        target_camera = from_state(target, self.camera_template)
        count = len(self.xy0)
        fx, fy = np.exp(camera[6:8])
        qy = (self.xy0[:, 1] - camera[9]) / fy
        qx = (self.xy0[:, 0] - camera[8] - camera[10] * qy) / fx
        rays_camera = np.column_stack((qx, qy, np.ones(count)))
        rotation, rod_jac = cv2.Rodrigues(np.ascontiguousarray(camera[3:6]))
        if rod_jac.shape == (3, 9):
            drot = rod_jac.reshape(3, 3, 3)
        elif rod_jac.shape == (9, 3):
            drot = rod_jac.T.reshape(3, 3, 3)
        else:
            raise ValueError(f"Unexpected Rodrigues Jacobian shape {rod_jac.shape}")
        rays = rays_camera @ rotation
        centre = camera[:3]
        normal = self.plane_normal
        numerator = np.einsum('ni,ni->n', self.plane_point - centre, normal)
        denominator = np.einsum('ni,ni->n', rays, normal)
        # Near-parallel rays have no stable plane association; keep the
        # original finite-difference behavior for this exceptional case.
        if np.any(np.abs(denominator) < 1e-7):
            return super().jacobians(states)
        raw_distance = numerator / denominator
        if np.any((raw_distance <= .001) | (raw_distance >= 1e5)):
            return super().jacobians(states)

        ray_jac = np.zeros((count, 3, 11), np.float64)
        for axis in range(3):
            ray_jac[:, :, 3 + axis] = rays_camera @ drot[axis]
        camera_ray_jac = np.zeros((count, 3, 5), np.float64)
        camera_ray_jac[:, 0, 0] = -qx
        camera_ray_jac[:, 0, 1] = camera[10] * qy / fx
        camera_ray_jac[:, 1, 1] = -qy
        camera_ray_jac[:, 0, 2] = -1000. / fx
        camera_ray_jac[:, 0, 3] = 1000. * camera[10] / (fx * fy)
        camera_ray_jac[:, 1, 3] = -1000. / fy
        camera_ray_jac[:, 0, 4] = -1000. * qy / fx
        ray_jac[:, :, 6:] = np.einsum('nap,aj->njp', camera_ray_jac, rotation)

        centre_jac = np.zeros((count, 3, 11), np.float64)
        centre_jac[:, :, :3] = 100. * np.eye(3)
        d_numerator = -np.einsum('na,nap->np', normal, centre_jac)
        d_denominator = np.einsum('na,nap->np', normal, ray_jac)
        d_distance = (d_numerator - raw_distance[:, None] * d_denominator) / denominator[:, None]
        world_jac = (centre_jac + raw_distance[:, None, None] * ray_jac
                     + rays[:, :, None] * d_distance[:, None, :])

        xyz = np.ascontiguousarray(centre + raw_distance[:, None] * rays)
        _, _, target_jac = project_fast(target_camera, xyz, with_jacobian=True)
        # The target camera-centre displacement is minus an equal world-point
        # displacement. Its state coordinates use scene units divided by 100.
        image_world_jac = -target_jac[:, :3].reshape(count, 2, 3) / 100.
        source_jac = np.einsum('nai,nij->naj', image_world_jac, world_jac).reshape(-1, 11)
        return [np.nan_to_num(source_jac), np.nan_to_num(target_jac)]

    def _brown15_jacobians(self, states, camera):
        """Differentiate the existing inverse Brown ray and its fixed plane hit.

        ``_world_points`` and ``require_central_projection`` retain the exact
        residual's validity rules. Clipped/ambiguous inverse rays use the
        original numerical derivative path instead of an invented tangent.
        """
        source, target = states
        if not np.isfinite(camera).all():
            return None
        xyz = self._world_points(source)
        target_camera = from_state(target, self.camera_template)
        # This is PairConstraint.projected's target path, including its central
        # Brown domain check; only its analytic columns are augmented below.
        _, _, target_core = require_central_projection(
            target_camera, xyz, stage='target_projection',
            family='surface_pair', source_keys=None, with_jacobian=True)
        rotation, rotation_jac = ContinuousBrownBoardPair._rotation(camera)
        local = (xyz-camera[:3]) @ rotation.T
        distance = local[:, 2]
        # PairConstraint clips the ray parameter to [.001, 1e5]. Leave a
        # derivative-only margin around that piecewise boundary. Near it the
        # numerical derivative is the faithful behavior for a clipped ray.
        if (not np.isfinite(local).all()
                or np.any((distance <= .0011) | (distance >= 1e5-10.))):
            return None
        normal = self.plane_normal
        normalized = local[:, :2]/distance[:, None]
        rays = np.column_stack((normalized, np.ones(len(distance)))) @ rotation
        denominator = np.einsum('ni,ni->n', rays, normal)
        if not np.isfinite(denominator).all() or np.any(np.abs(denominator) < 1e-7):
            return None
        # OpenCV's inverse Brown iteration can stop short of its exact root.
        # In that case the implicit tangent is not the derivative of the
        # actual finite-iteration residual, so retain the numerical route.
        roundtrip = project_fast(camera, camera[:3]+100.*rays)[0]
        if not np.isfinite(roundtrip).all() or np.max(np.abs(roundtrip-self.xy0)) > 1e-7:
            return None
        try:
            ray_jac = ContinuousBrownBoardPair._source_ray_jacobian(
                camera, normalized, rotation, rotation_jac)
        except (BoardIntersectionError, cv2.error, np.linalg.LinAlgError):
            return None
        tangent = (np.eye(3)[None, :, :]
                   - rays[:, :, None]*normal[:, None, :]/denominator[:, None, None])
        centre_jac = np.zeros((len(distance), 3, 15), np.float64)
        centre_jac[:, :, :3] = 100.*np.eye(3)
        world_jac = np.einsum('nij,njp->nip', tangent,
            centre_jac+distance[:, None, None]*ray_jac)
        image_world = -target_core.reshape(len(distance), 2, 11)[:, :, :3]/100.
        source_full = np.einsum('nai,nip->nap', image_world,
                                world_jac).reshape(2*len(distance), 15)
        target_full = np.column_stack((target_core,
            ContinuousBrownBoardPair._target_brown_jacobian(target_camera, xyz)))
        if not np.isfinite(source_full).all() or not np.isfinite(target_full).all():
            return None
        return [np.nan_to_num(source_full), np.nan_to_num(target_full)]
