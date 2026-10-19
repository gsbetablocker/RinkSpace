"""Native static correspondences through the existing Z190 calibration map.

This adapter adds no camera parameters.  A conditional fixed-centre factor
retains shared Brown4 sensitivity; typed finite ice/board factors can instead
reuse ReviewedIceSurfacePair / ReviewedCentralBoardPair.  Source identity and
scene/PTS provenance must be supplied and reviewed by the caller.
"""
from __future__ import annotations

try:
    from .runtime.bootstrap import add_geometry_dependencies
except ImportError:
    from runtime.bootstrap import add_geometry_dependencies

add_geometry_dependencies()
import numpy as np
import cv2

try:
    from .fixed_center_ray_pair import FixedCenterRayPair
    from .continuous_brown_board_pair import ContinuousBrownBoardPair
    from .projection import from_state, project_central_valid
except ImportError:
    from fixed_center_ray_pair import FixedCenterRayPair
    from continuous_brown_board_pair import ContinuousBrownBoardPair
    from projection import from_state, project_central_valid


def physical15(camera):
    """Existing physical15 units, retaining the camera's four fitted Brown terms."""
    values = np.asarray(camera, float)
    if values.shape != (25,) or not np.isfinite(values).all() or np.any(values[15:] != 0.):
        raise ValueError('Z190 pair adapter requires explicit Brown4 and zero other Brown terms')
    return np.r_[values[:3]/100., values[3:8], values[8:11]/1000., values[11:15]]


class BrownFixedCenterRayPair(FixedCenterRayPair):
    """Same-centre native ray transfer with source/target physical15 derivatives.

    Unlike the zero-Brown shared4K kernel, this includes inverse and forward
    Brown4.  Centre columns are zero only under the exact common-C condition;
    nonzero lever/axial or residual-C motion must use finite surface factors.
    """

    @staticmethod
    def _states(states):
        if len(states) != 2:
            raise ValueError('Ray factor requires two physical15 states')
        source, target = (np.asarray(state, float) for state in states)
        if (source.shape != (15,) or target.shape != (15,)
                or not np.isfinite(source).all() or not np.isfinite(target).all()):
            raise ValueError('Ray factor requires finite physical15 states')
        if not np.array_equal(source[:3], target[:3]):
            raise ValueError('Pure-ray factor cannot represent moving optical centres')
        return source, target

    @staticmethod
    def _world_directions(camera, pixels):
        fx, fy = np.exp(camera[6:8])
        unskewed = np.asarray(pixels, float).copy()
        unskewed[:, 0] -= camera[10]*(pixels[:, 1]-camera[9])/fy
        intrinsic = np.array([[fx, 0., camera[8]], [0., fy, camera[9]], [0., 0., 1.]])
        # A converged root is needed for the implicit Brown Jacobian.  Default
        # five-iteration undistortPoints can leave a visible error at k1≈.2.
        normalized = cv2.undistortPointsIter(
            np.ascontiguousarray(unskewed[:, None, :]), intrinsic,
            np.ascontiguousarray(camera[11:25]), None, None,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12))[:, 0, :]
        rotation = cv2.Rodrigues(np.ascontiguousarray(camera[3:6]))[0]
        return np.ascontiguousarray(np.c_[normalized, np.ones(len(pixels))] @ rotation)

    def _predicted(self, states, *, target_jacobian=False):
        source, target = self._states(states)
        camera0, camera1 = (from_state(state, template) for state, template in
                            ((source, self.camera0_template), (target, self.camera1_template)))
        camera0[:3] = camera1[:3] = 0.
        directions = self._world_directions(camera0, self.xy0)
        roundtrip, _, source_valid = project_central_valid(camera0, directions)
        if (not np.all(source_valid) or not np.isfinite(roundtrip).all()
                or np.max(np.abs(roundtrip-self.xy0)) > 1e-7):
            raise ValueError('Source inverse Brown root leaves the central domain or fails roundtrip')
        result = project_central_valid(camera1, directions, with_jacobian=target_jacobian)
        if not np.all(result[-1]):
            raise ValueError('Transferred static rays leave the target central domain')
        return (result[0], result[2]) if target_jacobian else result[0]

    def _raw_jacobians(self, states):
        source, target = self._states(states)
        _, target_core = self._predicted(states, target_jacobian=True)
        camera0 = from_state(source, self.camera0_template)
        camera1 = from_state(target, self.camera1_template)
        camera0[:3] = camera1[:3] = 0.
        rotation, drot = ContinuousBrownBoardPair._rotation(camera0)
        directions = self._world_directions(camera0, self.xy0)
        normalized = (directions @ rotation.T)[:, :2]
        source_ray = ContinuousBrownBoardPair._source_ray_jacobian(camera0, normalized, rotation, drot)
        image_world = -target_core.reshape(len(self.xy0), 2, 11)[:, :, :3]/100.
        source_jac = np.einsum('nai,nip->nap', image_world, source_ray).reshape(-1, 15)
        target_jac = np.c_[target_core,
                           ContinuousBrownBoardPair._target_brown_jacobian(camera1, directions)]
        source_jac[:, :3] = target_jac[:, :3] = 0.
        if not np.isfinite(source_jac).all() or not np.isfinite(target_jac).all():
            raise ValueError('Native Brown ray Jacobian is nonfinite')
        return [source_jac, target_jac]


def z190_pair_value_jacobians(pair, calibration, view0, view1, *, with_jacobian=True,
                            shared_columns=None):
    """Compose an existing static factor with ONE shared Z190 installation map.

    Returns native residual and (shared, source R3/logf, target R3/logf) J.
    Both endpoint shared contributions are summed.  Thus PP, k1 and the
    existing lever/axial are represented once; no extra C/crop node appears.
    The caller chooses fixed-ray or reviewed finite-surface factors explicitly.
    A pure-ray factor needs explicit active shared columns; lever/axial columns
    are refused even at a zero-offset seed because their moving-C tangents
    require finite depth.  Finite-surface factors may expose the full map.
    """
    camera0, shared0, state0 = calibration.camera25_and_jacobians(view0)
    camera1, shared1, state1 = calibration.camera25_and_jacobians(view1)
    states = [physical15(camera0), physical15(camera1)]
    raw_residual = getattr(pair, '_raw_residual', None) or pair.residual
    residual = raw_residual(states)
    if not with_jacobian:
        return residual
    names = calibration.shared_state_names()
    columns = np.arange(len(names)) if shared_columns is None else np.asarray(shared_columns, int)
    if (columns.ndim != 1 or len(set(map(int, columns))) != len(columns)
            or np.any(columns < 0) or np.any(columns >= len(names))):
        raise ValueError('Shared columns must be distinct existing calibration columns')
    if isinstance(pair, BrownFixedCenterRayPair):
        if shared_columns is None or any(names[i].startswith(('lever_', 'axial_')) for i in columns):
            raise ValueError('Pure ray J requires explicit fixed-centre shared columns excluding lever/axial')
    raw_jacobians = getattr(pair, '_raw_jacobians', None) or pair.jacobians
    ja, jb = raw_jacobians(states)
    if ja.shape[1] == 11 or jb.shape[1] == 11:
        # SurfacePairConstraint retains its legacy 11-column numerical route
        # when the old inverse-Brown iteration has not reached a differentiable
        # root. Complete only its missing Brown4 columns on that SAME residual;
        # do not attach an implicit-root tangent to a finite-iteration inverse.
        complete = []
        for endpoint, matrix in enumerate((ja, jb)):
            if matrix.shape[1] == 15:
                complete.append(matrix)
                continue
            if matrix.shape[1] != 11:
                raise ValueError('Existing static factor J must have 11 or 15 physical columns')
            extra = np.empty((len(residual), 4))
            for brown in range(4):
                high, low = [q.copy() for q in states], [q.copy() for q in states]
                high[endpoint][11+brown] += 1e-7
                low[endpoint][11+brown] -= 1e-7
                extra[:, brown] = (raw_residual(high)-raw_residual(low))/2e-7
            complete.append(np.c_[matrix, extra])
        ja, jb = complete
    # Physical15 scaling is the same as projection.from_state.  Brown4 occupy
    # camera25 rows 11:15; all remaining Brown14 terms are explicitly zero.
    scale = np.array([.01]*3+[1.]*5+[.001]*3+[1.]*4)
    shared = (ja @ (scale[:, None]*shared0[:15]) + jb @ (scale[:, None]*shared1[:15]))[:, columns]
    source = ja @ (scale[:, None]*state0[:15])
    target = jb @ (scale[:, None]*state1[:15])
    return residual, shared, source, target
