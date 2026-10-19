"""Shared eccentric pan/tilt mount for the conditional rink camera.

The frame state is the 11-value ``projection.to_state`` vector.  A mount
state has the same width for the iSAM2 backend: ``[P/100, d/100, 0, ..., 0]``.
``P`` is fixed in world coordinates and ``d`` in camera coordinates.  With
the released world-to-camera rotation ``R``, the optical centre is
``C = P + R.T @ d``.  All distances remain conditional scene units.
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
    from .projection import from_state
except ImportError:
    from projection import from_state


STATE_DIM = 11
CENTRE_SCALE = 100.0


def _state(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, np.float64).reshape(-1)
    if result.shape != (STATE_DIM,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain 11 finite values")
    return result


def mount_state(pivot_scene: np.ndarray, offset_camera_scene: np.ndarray) -> np.ndarray:
    """Pack the shared pivot and camera-local lever arm for a Vector11 key."""
    pivot = np.asarray(pivot_scene, np.float64).reshape(-1)
    offset = np.asarray(offset_camera_scene, np.float64).reshape(-1)
    if pivot.shape != (3,) or offset.shape != (3,) or not (
        np.isfinite(pivot).all() and np.isfinite(offset).all()
    ):
        raise ValueError("Pivot and offset must each contain three finite scene coordinates")
    result = np.zeros(STATE_DIM, np.float64)
    result[:3], result[3:6] = pivot / CENTRE_SCALE, offset / CENTRE_SCALE
    return result


def camera_center_scene(frame_state: np.ndarray, mount: np.ndarray) -> np.ndarray:
    """Return the frame's world optical centre in conditional scene units."""
    frame = _state(frame_state, "frame_state")
    shared = _state(mount, "mount_state")
    rotation = cv2.Rodrigues(np.ascontiguousarray(frame[3:6]))[0]
    return CENTRE_SCALE * (shared[:3] + rotation.T @ shared[3:6])


def camera_from_mount(frame_state: np.ndarray, template25: np.ndarray,
                      mount: np.ndarray) -> np.ndarray:
    """Build a Brown25 camera, replacing frame XYZ by the eccentric mount."""
    frame = _state(frame_state, "frame_state")
    camera = from_state(frame, template25)
    camera[:3] = camera_center_scene(frame, mount)
    return camera


def rig_equality_residual(states: list[np.ndarray]) -> np.ndarray:
    """Vector3 factor: frame centre minus the optical centre implied by mount."""
    if len(states) != 2:
        raise ValueError("Rig equality requires [frame_state, mount_state]")
    frame = _state(states[0], "frame_state")
    shared = _state(states[1], "mount_state")
    rotation = cv2.Rodrigues(np.ascontiguousarray(frame[3:6]))[0]
    return frame[:3] - shared[:3] - rotation.T @ shared[3:6]


def rig_equality_jacobians(states: list[np.ndarray]) -> list[np.ndarray]:
    """Analytic 3x11 Jacobians in [frame, mount] order for GTSAM CustomFactor."""
    if len(states) != 2:
        raise ValueError("Rig equality requires [frame_state, mount_state]")
    frame = _state(states[0], "frame_state")
    shared = _state(states[1], "mount_state")
    rotation, derivative = cv2.Rodrigues(np.ascontiguousarray(frame[3:6]))
    if derivative.shape == (3, 9):
        rotation_derivatives = derivative.reshape(3, 3, 3)
    elif derivative.shape == (9, 3):
        rotation_derivatives = derivative.T.reshape(3, 3, 3)
    else:
        raise ValueError(f"Unexpected Rodrigues Jacobian shape {derivative.shape}")
    j_frame = np.zeros((3, STATE_DIM), np.float64)
    j_mount = np.zeros((3, STATE_DIM), np.float64)
    j_frame[:, :3] = np.eye(3)
    for axis in range(3):
        j_frame[:, 3 + axis] = -rotation_derivatives[axis].T @ shared[3:6]
    j_mount[:, :3] = -np.eye(3)
    j_mount[:, 3:6] = -rotation.T
    return [j_frame, j_mount]


def add_rig_equality(graph, frame_key: int, mount_key: int,
                     sigma_scene: float = 0.001) -> None:
    """Register the eccentric mount equality in the conditioned iSAM2 graph."""
    if not np.isfinite(sigma_scene) or sigma_scene <= 0:
        raise ValueError("sigma_scene must be positive and finite")
    graph.add_observation(
        [frame_key, mount_key], rig_equality_residual, residual_dim=3,
        sigma=sigma_scene / CENTRE_SCALE, jac_fn=rig_equality_jacobians,
    )
