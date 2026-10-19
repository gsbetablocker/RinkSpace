"""Shared, bounded scene geometry for *measured* advertisement corners.

The released scene supplies advertisement identity, approximate boundary
position and height.  Each source-frame corner pixel is a separate observation
of one scene-wide ad state; no world point is reverse-projected from an old
camera.  A solved state is a candidate sidecar and never edits the scene.

State coordinates (all scene units divided by 100): left/right displacement
along the existing ice boundary, then bottom/top height displacement.  The
other seven coordinates stay at zero so this fits the camera graph's 11-vector.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

try:
    from .projection import from_state, require_central_projection
except ImportError:  # Direct program_tracking script imports.
    from projection import from_state, require_central_projection


ROLES = ("left_top", "left_bottom", "right_top", "right_bottom")
_SIDE_INDEX = {"left": 0, "right": 1}
_HEIGHT_INDEX = {"bottom": 2, "top": 3}


class AdGeometry:
    """One user-numbered ad, fitted against a shared rink-boundary polyline.

    `ad_id` is the existing `user_number`, never an array position.  The two
    bases are the continuous closest locations on actual boundary *segments*,
    rather than nearest sampled vertices (which can be several scene units
    apart along straight sides in the released model).
    """

    def __init__(self, scene: Mapping, ad_id: int):
        ads = [row for row in scene["advertisements"]
               if int(row["user_number"]) == int(ad_id)]
        if len(ads) != 1:
            raise ValueError(f"Expected one scene advertisement user_number={ad_id}")
        self.user_number = int(ad_id)
        self._source = ads[0]
        corners = ads[0]["corners_xyz_scene_units"]
        self._roles = tuple(role for role in ROLES if role in corners)
        if not self._roles:
            raise ValueError(f"Advertisement {ad_id} has no scene corners")

        xy = np.asarray(scene["ice_boundary"]["points_xyz"], np.float64)[:, :2]
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or not np.isfinite(xy).all():
            raise ValueError("Scene boundary must be a finite XY polyline")
        if np.linalg.norm(xy[-1] - xy[0]) < 1e-9:
            xy = xy[:-1]
        self._starts = xy.copy()
        self._edges = np.roll(xy, -1, axis=0) - xy
        self._lengths = np.linalg.norm(self._edges, axis=1)
        if np.any(self._lengths < 1e-8):
            raise ValueError("Scene boundary contains a zero-length edge")
        self._cum = np.r_[0., np.cumsum(self._lengths)]
        self._perimeter = float(self._cum[-1])

        self._base_s: dict[str, float] = {}
        self._base_height: dict[str, float] = {}
        for side in _SIDE_INDEX:
            role = side + "_bottom" if side + "_bottom" in corners else side + "_top"
            if role in corners:
                point = np.asarray(corners[role], np.float64)
                if point.shape != (3,) or not np.isfinite(point).all():
                    raise ValueError(f"Invalid scene corner {role}")
                self._base_s[side] = self._nearest_arc(point[:2])
        for level in _HEIGHT_INDEX:
            values = [float(corners[side + "_" + level][2])
                      for side in _SIDE_INDEX if side + "_" + level in corners]
            if not values or np.max(values) - np.min(values) > 1e-3:
                raise ValueError(f"Advertisement {ad_id} has inconsistent {level} heights")
            self._base_height[level] = float(np.mean(values))

        self._nominal_width = None
        self._width_sign = None
        if len(self._base_s) == 2:
            left, right = self._base_s["left"], self._base_s["right"]
            delta = (right - left + self._perimeter / 2.) % self._perimeter - self._perimeter / 2.
            if abs(delta) <= .1:
                raise ValueError(f"Advertisement {ad_id} has no physical left/right separation")
            self._width_sign = float(np.sign(delta))
            self._nominal_width = float(abs(delta))
        self._board_top = float(np.max(np.asarray(
            scene["board_surface"]["top_edge_xyz"], np.float64)[:, 2]))
        self._ice_z = float(np.median(np.asarray(
            scene["ice_boundary"]["points_xyz"], np.float64)[:, 2]))

    @property
    def roles(self) -> tuple[str, ...]:
        return self._roles

    @property
    def initial(self) -> np.ndarray:
        return np.zeros(11, np.float64)

    @property
    def prior_sigma(self) -> np.ndarray:
        # Arc shifts up to three scene units and heights up to 0.3 are
        # deliberately allowed by the Gaussian prior.  Unused entries fixed.
        return np.r_[.03, .03, .003, .003, np.full(7, 1e-6)]

    def _nearest_arc(self, point: np.ndarray) -> float:
        t = np.clip(np.sum((point - self._starts) * self._edges, axis=1)
                    / self._lengths**2, 0., 1.)
        closest = self._starts + t[:, None] * self._edges
        edge = int(np.argmin(np.sum((closest - point)**2, axis=1)))
        return float(self._cum[edge] + t[edge] * self._lengths[edge])

    def _point_at_arc(self, arc: float) -> np.ndarray:
        s = float(arc) % self._perimeter
        edge = min(int(np.searchsorted(self._cum, s, side="right") - 1), len(self._starts) - 1)
        t = (s - self._cum[edge]) / self._lengths[edge]
        return self._starts[edge] + t * self._edges[edge]

    @staticmethod
    def _state(state: np.ndarray) -> np.ndarray:
        result = np.asarray(state, np.float64).reshape(-1)
        if result.shape != (11,) or not np.isfinite(result).all():
            raise ValueError("Advertisement state must have 11 finite values")
        return result

    def corner(self, ad_state: np.ndarray, role: str) -> np.ndarray:
        """World XYZ for a *scene-existing* corner at the current shared state."""
        if role not in self._roles:
            raise ValueError(f"Advertisement {self.user_number} has no scene corner {role}")
        state = self._state(ad_state)
        side, level = role.split("_", 1)
        xy = self._point_at_arc(self._base_s[side] + 100. * state[_SIDE_INDEX[side]])
        z = self._base_height[level] + 100. * state[_HEIGHT_INDEX[level]]
        return np.r_[xy, z]

    def _guard(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Only protect ad order and physically possible panel heights."""
        state = self._state(state)
        bottom = self._base_height["bottom"] + 100. * state[2]
        top = self._base_height["top"] + 100. * state[3]
        width_violation = 0.
        gradient = np.zeros((4, 11), np.float64)
        if self._nominal_width is not None:
            width = self._nominal_width + 100. * self._width_sign * (state[1] - state[0])
            min_width = min(.1, self._nominal_width * .1)
            width_violation = max(0., min_width - width)
            if width_violation > 0:
                gradient[0, 0] = 100. * self._width_sign
                gradient[0, 1] = -100. * self._width_sign
        height_violation = max(0., .1 - (top - bottom))
        if height_violation > 0:
            gradient[1, 2:4] = (100., -100.)
        below_ice = max(0., self._ice_z - bottom)
        if below_ice > 0:
            gradient[2, 2] = -100.
        above_board = max(0., top - self._board_top)
        if above_board > 0:
            gradient[3, 3] = 100.
        return np.array((width_violation, height_violation, below_ice, above_board)), gradient

    def add_prior(self, graph, ad_key: int) -> None:
        """Install broad base geometry prior and hinge-only physical guards."""
        graph.add_prior(ad_key, self.initial, self.prior_sigma)
        graph.add_observation(
            [ad_key], lambda states: self._guard(states[0])[0],
            residual_dim=4, sigma=np.full(4, .03),
            jac_fn=lambda states: [self._guard(states[0])[1]],
        )

    def factor(self, graph, frame_key: int, ad_key: int,
               camera_template: np.ndarray, role: str,
               uv: np.ndarray, sigma_px: float | np.ndarray = 3.,
               components: tuple[str, ...] = ("x", "y")) -> None:
        """Connect an actual source-image corner to camera and shared ad state.

        The camera-state Jacobian is the conditioned analytic projection
        derivative.  Only the corner's arc and height components need central
        differences; the other nine ad Jacobian columns remain exactly zero.
        """
        if role not in self._roles:
            raise ValueError(f"Advertisement {self.user_number} has no scene corner {role}")
        if not isinstance(components, (tuple, list)):
            raise ValueError("Advertisement components must be an x/y sequence")
        selected = tuple(components)
        if selected not in (("x", "y"), ("x",), ("y",)):
            raise ValueError("Advertisement factor needs x, y, or both components")
        indices = (0, 1) if selected == ("x", "y") else ((0,) if selected == ("x",) else (1,))
        observed = np.asarray(uv, np.float64).reshape(-1).copy()
        template = np.asarray(camera_template, np.float64).reshape(-1).copy()
        if observed.shape != (2,) or not np.isfinite(observed).all():
            raise ValueError("Source corner must be one finite native-image (u,v) point")
        if template.shape != (25,) or not np.isfinite(template).all():
            raise ValueError("Camera template must contain 25 finite values")
        sigma = np.broadcast_to(np.asarray(sigma_px, float), (2,)).copy()
        if not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError("Pixel sigma must be finite and positive")
        if selected != ("x", "y"):
            sigma = sigma[list(indices)]
        side, level = role.split("_", 1)
        active = (_SIDE_INDEX[side], _HEIGHT_INDEX[level])

        def residual(states: list[np.ndarray]) -> np.ndarray:
            camera = from_state(states[0], template)
            world = self.corner(states[1], role)
            projected, _ = require_central_projection(
                camera, world[None, :], stage="residual",
                family="ad_landmarks_corner", source_keys=(frame_key, ad_key))
            difference = projected[0] - observed
            return difference if selected == ("x", "y") else difference[list(indices)]

        def jacobians(states: list[np.ndarray]) -> list[np.ndarray]:
            camera = from_state(states[0], template)
            _, _, camera_jac = require_central_projection(
                camera, self.corner(states[1], role)[None, :],
                with_jacobian=True, stage="jacobian",
                family="ad_landmarks_corner", source_keys=(frame_key, ad_key))
            ad_jac = np.zeros((2, 11), np.float64)
            for column in active:
                step = 1e-5 * max(1., abs(float(states[1][column])))
                plus, minus = states[1].copy(), states[1].copy()
                plus[column] += step
                minus[column] -= step
                high = require_central_projection(
                    camera, self.corner(plus, role)[None, :],
                    stage="jacobian", family="ad_landmarks_corner",
                    source_keys=(frame_key, ad_key))[0][0]
                low = require_central_projection(
                    camera, self.corner(minus, role)[None, :],
                    stage="jacobian", family="ad_landmarks_corner",
                    source_keys=(frame_key, ad_key))[0][0]
                ad_jac[:, column] = (high - low) / (2. * step)
            if selected == ("x", "y"):
                return [camera_jac, ad_jac]
            return [camera_jac[list(indices)], ad_jac[list(indices)]]

        graph.add_observation(
            [frame_key, ad_key], residual, residual_dim=len(indices),
            sigma=sigma, jac_fn=jacobians, huber_k=3.,
        )

    def solved_geometry(self, state: np.ndarray) -> dict:
        """Serializable scene-unit candidate; scene itself is left untouched."""
        q = self._state(state)
        guard, _ = self._guard(q)
        return {
            "user_number": self.user_number,
            "status": "joint_ad_geometry_candidate_unvalidated",
            "model_roles": list(self._roles),
            "corners_xyz_scene_units": {
                role: self.corner(q, role).tolist() for role in self._roles
            },
            "boundary_arc_shift_scene_units": {
                side: float(100. * q[_SIDE_INDEX[side]])
                for side in self._base_s
            },
            "height_shift_scene_units": {
                level: float(100. * q[_HEIGHT_INDEX[level]])
                for level in _HEIGHT_INDEX
            },
            "physical_guard_violation_scene_units": guard.tolist(),
        }
