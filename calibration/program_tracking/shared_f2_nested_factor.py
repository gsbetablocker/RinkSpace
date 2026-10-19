"""Opt-in F2 outer-ring factor with one shared, movable world circle.

This is a preparation module, not part of the running programme solver.  It
uses exactly the existing reference-curve pixel residual, Huber transform,
source group weight, Brown15 camera chain and camera graph keys.  The only new
graph node is the pre-existing shared-ring Vector11 convention: columns 0:3
are dx/100, dy/100 and outer-radius change/100.  Callers retain the original
camera states and prior targets and add a single scene prior/gauge separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from typing import Mapping

from runtime.bootstrap import add_geometry_dependencies

add_geometry_dependencies()
import numpy as np

from continuous_brown_adapter import adapt_factor
from continuous_reference_calibration import _measurement_callback
from robust_observations import pointwise_huber
from secondary_nearboard_adaptive_curve import (
    AdaptiveProjectedPolyline, adaptive_curve_callback)
from adaptive_curve_parallel import AdaptiveCurveJob
from reference_board_pair import BoardIntersectionError


F2_SCENE_KEY = 2001  # SharedRingScene's sorted C0,F2,R3,R4 key convention.


@dataclass(frozen=True)
class F2SourceCurve:
    pts: int
    source_id: str
    uv: np.ndarray
    world0: np.ndarray
    center_xy: np.ndarray
    outer_radius: float
    sigma_px: float
    same_image_target_count: int = 1

    def __post_init__(self):
        uv = np.asarray(self.uv, np.float64)
        world = np.asarray(self.world0, np.float64)
        center = np.asarray(self.center_xy, np.float64)
        if (uv.ndim != 2 or uv.shape[1] != 2 or len(uv) < 3
                or world.ndim != 2 or world.shape[1] != 3 or len(world) < 3
                or center.shape != (2,) or not np.isfinite(uv).all()
                or not np.isfinite(world).all() or not np.isfinite(center).all()
                or not np.isfinite(self.outer_radius) or self.outer_radius <= 0
                or not np.isfinite(self.sigma_px) or self.sigma_px <= 0
                or self.same_image_target_count < 1):
            raise ValueError('Invalid actual-source F2 outer curve or scene circle')
        if (np.max(np.abs(world[:, 2] - world[0, 2])) > 1e-9
                or np.max(np.abs(np.linalg.norm(world[:, :2]-center, axis=1)
                                  - self.outer_radius)) > 1e-5):
            raise ValueError('World curve does not lie on the named F2 outer circle')
        object.__setattr__(self, 'uv', uv.copy())
        object.__setattr__(self, 'world0', world.copy())
        object.__setattr__(self, 'center_xy', center.copy())

    @property
    def factor_sigma(self) -> float:
        return float(self.sigma_px * np.sqrt(len(self.uv) *
                                             self.same_image_target_count))

    def world(self, scene11: np.ndarray) -> np.ndarray:
        state = np.asarray(scene11, np.float64)
        if state.shape != (11,) or not np.isfinite(state).all():
            raise ValueError('F2 scene node must be finite Vector11')
        if self.outer_radius + 100.*state[2] <= 0:
            failure=BoardIntersectionError(
                'shared_F2_outer_radius_nonpositive',np.arange(len(self.uv)))
            failure.source_pts=int(self.pts)
            failure.source_keys=(int(self.pts),)
            failure.factor_family='shared_F2_outer_reference_curve'
            failure.evaluation_stage='scene_domain'
            raise failure
        world = self.world0.copy()
        world[:, :2] += 100.*state[:2]
        world[:, :2] += (100.*state[2]) * (
            (self.world0[:, :2]-self.center_xy)/self.outer_radius)
        return world


class NestedF2Factor:
    """Append a shared scene node to an otherwise unchanged camera factor.

    ``residual`` and ``jacobian`` take the same list of Vector11 camera nodes
    as the fixed-scene reference factor, followed by the one F2 scene node.
    The returned sigma is the existing per-image target-group sigma.  World
    columns use the circle/camera similarity identity on the *same robust
    physical15 Jacobian*.  This keeps exact translation and circle-scale
    gauge directions instead of creating a false rank from different finite
    difference steps at a nearest-curve tie.  All camera columns use the
    original physical15 callback and rig chain.
    """

    def __init__(self, model, source: F2SourceCurve,
                 active_camera_keys: set[int],
                 fixed_states: Mapping[int, np.ndarray], *,
                 scene_key: int = F2_SCENE_KEY,
                 curve_mode: str = 'adaptive',
                 adaptive_tolerance_px: float = .01):
        if source.pts not in model.frames or scene_key in active_camera_keys:
            raise ValueError('F2 source camera missing or scene key collides')
        if (curve_mode not in ('sampled', 'adaptive')
                or not np.isfinite(adaptive_tolerance_px)
                or adaptive_tolerance_px <= 0):
            raise ValueError('Invalid reference curve numerical mode')
        self.model = model
        self.source = source
        self.scene_key = int(scene_key)
        self.curve_mode = curve_mode
        self.adaptive_tolerance_px = float(adaptive_tolerance_px)
        self.active_camera_keys = set(map(int, active_camera_keys))
        self.fixed_states = fixed_states
        self.template = model.frames[source.pts].template25.copy()
        self._scene_cache = OrderedDict()
        fixed = self._camera_factor(np.zeros(11))
        self.camera_keys = tuple(fixed[0])
        self.keys = (*self.camera_keys, self.scene_key)
        self.options = dict(fixed[3])
        if (self.options['residual_dim'] != 2*len(source.uv)
                or self.options['sigma'] != source.factor_sigma):
            raise ValueError('F2 factor changed original reference weighting')

    def _scene_entry(self, scene11: np.ndarray):
        scene11 = np.asarray(scene11, np.float64)
        token = scene11.tobytes()
        if token in self._scene_cache:
            self._scene_cache.move_to_end(token)
            return self._scene_cache[token]
        world = self.source.world(scene11)
        if self.curve_mode == 'adaptive':
            measurements = [dict(x_interval_px=(float(x), float(x)),
                                 y=float(y)) for x, y in self.source.uv]
            curve = AdaptiveProjectedPolyline(
                world, measurements, tolerance_px=self.adaptive_tolerance_px,
                closed=True, factor_family='original_reference_curve',
                no_central_reason='reference_curve_no_central_segment')
            raw, jac, prime = adaptive_curve_callback(
                curve, self.template, (self.source.pts,), with_prime=True)
        else:
            raw, jac = _measurement_callback(
                'curve', world, self.source.uv,
                self.template, (self.source.pts,))
            curve = prime = None
        robust = pointwise_huber(
            raw, jac, group_size=2, delta=2.*self.source.sigma_px)
        adapted = adapt_factor(
            (self.source.pts,), *robust,
            dict(residual_dim=2*len(self.source.uv),
                 sigma=self.source.factor_sigma),
            self.active_camera_keys, self.fixed_states, self.model)
        if adapted is None:
            raise ValueError('F2 source factor has no active camera columns')
        entry = (adapted, robust[1], curve, prime)
        self._scene_cache[token] = entry
        # A trial optimizer may visit many scene states. Keep the accepted and
        # latest trial callbacks only; worker bundles live in the graph cache.
        while len(self._scene_cache) > 3:
            pinned=getattr(self,'_prepared_scene_token',None)
            victim=next(key for key in self._scene_cache if key!=pinned)
            del self._scene_cache[victim]
        return entry

    def _camera_factor(self, scene11: np.ndarray):
        return self._scene_entry(scene11)[0]

    def adaptive_job(self, scene11: np.ndarray) -> AdaptiveCurveJob:
        """Existing worker job at one exact scene state; caller invalidates its
        cached physical-only token whenever that scene state changes."""
        if self.curve_mode != 'adaptive':
            raise ValueError('Only adaptive F2 factors have worker jobs')
        _, _, curve, prime = self._scene_entry(scene11)
        return AdaptiveCurveJob(curve, self.template, self.model,
                                int(self.source.pts), (int(self.source.pts),),
                                prime)

    def _physical_state(self, camera_values):
        states = dict(self.fixed_states)
        states.update(zip(self.camera_keys, camera_values))
        return self.model.physical_state(self.source.pts, states)

    def fixed_scene_factor(self):
        """Return the unmodified reference factor at zero world release."""
        return self._camera_factor(np.zeros(11))

    def _split(self, values):
        if len(values) != len(self.keys):
            raise ValueError('Wrong graph-node count for nested F2 factor')
        scene = np.asarray(values[-1], np.float64)
        if scene.shape != (11,) or not np.isfinite(scene).all():
            raise ValueError('Invalid F2 scene graph node')
        return values[:-1], scene

    def residual(self, values):
        camera_values, scene = self._split(values)
        try:
            keys, residual, _, options = self._camera_factor(scene)
        except BoardIntersectionError as exc:
            exc.evaluation_stage='residual'
            raise
        if tuple(keys) != self.camera_keys or options != self.options:
            raise RuntimeError('Moving F2 changed camera columns or source weight')
        return residual(camera_values)

    def jacobian(self, values):
        camera_values, scene = self._split(values)
        try:
            keys, residual, derivative, options = self._camera_factor(scene)
        except BoardIntersectionError as exc:
            exc.evaluation_stage='jacobian'
            raise
        if tuple(keys) != self.camera_keys or options != self.options:
            raise RuntimeError('Moving F2 changed camera columns or source weight')
        camera_jac = derivative(camera_values)
        world_jac = np.zeros((2*len(self.source.uv), 11), np.float64)
        physical_jacobian = self._scene_entry(scene)[1]
        physical = self._physical_state(camera_values)
        center = self.source.center_xy + 100.*scene[:2]
        radius = self.source.outer_radius + 100.*scene[2]
        camera_center = physical[:3]*100.
        circle_center = np.r_[center, self.source.world0[0, 2]]
        jac_c = physical_jacobian([physical])[0][:, :3]
        world_jac[:, 0] = -jac_c[:, 0]
        world_jac[:, 1] = -jac_c[:, 1]
        world_jac[:, 2] = -jac_c @ ((camera_center-circle_center)/radius)
        if not np.isfinite(world_jac).all():
            raise ValueError('F2 world Jacobian became nonfinite')
        return [*camera_jac, world_jac]
