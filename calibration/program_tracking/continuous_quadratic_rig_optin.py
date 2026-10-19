"""Optional shared quadratic log-zoom optics for the existing movable-C rig.

Only two new Vector11 nodes per camera are introduced: one holds curvature of
aspect/cx/cy/skew, the other Brown4 curvature.  Their contribution is
``0.5 * (mean_log_f - zref)**2``.  Every original frame, spline, mechanical,
linear lens and Brown node remains adjustable.  This module neither starts a
solve nor changes the production rig or its priors.

The existing round6 native source-edge binary does *not* understand the two
new roles.  A full graph must use the separate quadratic-capable native bridge
and must explicitly verify that bridge before adding source-edge factors.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from continuous_rig_camera import ContinuousRigCameraModel


LENS_COLUMNS = (6, 8, 9, 10)
BROWN_COLUMNS = (0, 1, 2, 3)
# Existing first-order priors are [.1,.2,.2,.1] in Vector11 units per ln(f)
# and [.5,.5,.05,.05] in Brown units per ln(f).  These proposed *soft* scales
# use the same numerical values per ln(f)^2 for coefficient q of 0.5*dz^2*q.
# Principal/skew Vector11 entries are divided by 1000, so their scales mean
# [200,200,100] sensor px per ln(f)^2; no range bound is imposed.
LENS_CURVATURE_SIGMA = np.array([.1, .2, .2, .1], np.float64)
BROWN_CURVATURE_SIGMA = np.array([.5, .5, .05, .05], np.float64)


def _node(states: Mapping[int, np.ndarray], key: int, role: str) -> np.ndarray:
    value = np.asarray(states[int(key)], np.float64)
    if value.shape != (11,) or not np.isfinite(value).all():
        raise ValueError(f'{role} must be a finite Vector11 node')
    return value


class QuadraticRigCameraModel(ContinuousRigCameraModel):
    """The original physical15 model plus shared, zero-able zoom curvature."""

    requires_quadratic_native_edge_roles = True

    def __init__(self, *args, lens_quadratic_keys: Mapping[str, int],
                 brown_quadratic_keys: Mapping[str, int], **kwargs):
        super().__init__(*args, **kwargs)
        cameras = set(self.lens_base_keys)
        self.lens_quadratic_keys = {str(k): int(v)
                                    for k, v in lens_quadratic_keys.items()}
        self.brown_quadratic_keys = {str(k): int(v)
                                     for k, v in brown_quadratic_keys.items()}
        if (set(self.lens_quadratic_keys) != cameras
                or set(self.brown_quadratic_keys) != cameras
                or set(self.brown_keys) != cameras):
            raise ValueError('Quadratic K/Brown4 requires two nodes for every camera')
        extras = list(self.lens_quadratic_keys.values())
        extras.extend(self.brown_quadratic_keys.values())
        old = {key for pts in self.frames
               for key in ContinuousRigCameraModel.dependencies(self, pts)}
        if len(set(extras)) != len(extras) or set(extras) & old:
            raise ValueError('Quadratic nodes collide with existing graph nodes')

    @classmethod
    def from_rig(cls, rig: ContinuousRigCameraModel, *,
                 lens_quadratic_keys: Mapping[str, int],
                 brown_quadratic_keys: Mapping[str, int]):
        return cls(rig.shot_splines, rig.frames, rig.lens_base_keys,
                   rig.lens_slope_keys, rig.lens_zoom_reference,
                   mount_keys=rig.mount_keys, offset_keys=rig.offset_keys,
                   offset_slope_keys=rig.offset_slope_keys,
                   brown_keys=rig.brown_keys,
                   brown_slope_keys=rig.brown_slope_keys,
                   lens_quadratic_keys=lens_quadratic_keys,
                   brown_quadratic_keys=brown_quadratic_keys)

    def dependencies(self, pts: int) -> tuple[int, ...]:
        group = self.frames[int(pts)].camera_id
        return (*super().dependencies(pts),
                self.lens_quadratic_keys[group],
                self.brown_quadratic_keys[group])

    def _physical(self, pts, states, jacobian):
        original = super()._physical(pts, states, jacobian)
        physical, blocks = original if jacobian else (original, None)
        if physical.shape != (15,):
            raise ValueError('Quadratic model needs the original physical15 Brown4 rig')
        binding = self.frames[int(pts)]
        group = binding.camera_id
        frame = _node(states, binding.frame_key, 'frame')
        lens_key = self.lens_quadratic_keys[group]
        brown_key = self.brown_quadratic_keys[group]
        lens = _node(states, lens_key, 'lens curvature')
        brown = _node(states, brown_key, 'Brown curvature')
        dz = float(frame[6] - self.lens_zoom_reference[group])
        half_dz2 = .5 * dz * dz
        ratio_curvature = half_dz2 * lens[6]
        physical[6] += .5 * ratio_curvature
        physical[7] -= .5 * ratio_curvature
        physical[8:11] += half_dz2 * lens[8:11]
        physical[11:15] += half_dz2 * brown[:4]
        if not jacobian:
            return physical
        blocks[binding.frame_key][6, 6] += .5 * dz * lens[6]
        blocks[binding.frame_key][7, 6] -= .5 * dz * lens[6]
        blocks[binding.frame_key][8:11, 6] += dz * lens[8:11]
        blocks[binding.frame_key][11:15, 6] += dz * brown[:4]
        lens_block = np.zeros((15, 11))
        lens_block[6, 6] = .5 * half_dz2
        lens_block[7, 6] = -.5 * half_dz2
        lens_block[8:11, 8:11] = half_dz2 * np.eye(3)
        brown_block = np.zeros((15, 11))
        brown_block[11:15, :4] = half_dz2 * np.eye(4)
        blocks[lens_key] = lens_block
        blocks[brown_key] = brown_block
        return physical, blocks


def prepare_quadratic_rig(rig: ContinuousRigCameraModel,
                          initial: Mapping[int, np.ndarray], *,
                          key_pairs: Mapping[str, tuple[int, int]] | None = None):
    """Create deterministic, zero-curvature nodes without changing old states."""
    if isinstance(rig, QuadraticRigCameraModel):
        raise ValueError('Rig already has quadratic optics')
    cameras = sorted(rig.lens_base_keys)
    if key_pairs is None:
        next_key = max(map(int, initial)) + 1
        key_pairs = {group: (next_key + 2*i, next_key + 2*i + 1)
                     for i, group in enumerate(cameras)}
    if set(key_pairs) != set(cameras):
        raise ValueError('Provide exactly one curvature key pair per camera')
    lens_keys = {group: int(key_pairs[group][0]) for group in cameras}
    brown_keys = {group: int(key_pairs[group][1]) for group in cameras}
    model = QuadraticRigCameraModel.from_rig(
        rig, lens_quadratic_keys=lens_keys, brown_quadratic_keys=brown_keys)
    if any(key in initial for pair in key_pairs.values() for key in pair):
        raise ValueError('New curvature keys must not overwrite original states')
    augmented = {int(key): np.asarray(value, np.float64).copy()
                 for key, value in initial.items()}
    for group in cameras:
        augmented[lens_keys[group]] = np.zeros(11)
        augmented[brown_keys[group]] = np.zeros(11)
    return model, augmented, dict(
        schema='rinkspace.optional-quadratic-logzoom-rig.v1',
        curvature_formula='0.5*(mean_log_f-zref)^2*coefficient',
        lens_quadratic_keys=lens_keys, brown_quadratic_keys=brown_keys,
        old_key_count=len(initial), new_key_count=2*len(cameras),
        original_states_modified=False, original_priors_modified=False,
        native_source_edge_requires_quadratic_bridge=True)


def add_quadratic_priors(graph, model: QuadraticRigCameraModel,
                         initial: Mapping[int, np.ndarray], *,
                         lens_sigma=LENS_CURVATURE_SIGMA,
                         brown_sigma=BROWN_CURVATURE_SIGMA,
                         affine_priors=None) -> dict:
    """Add only soft zero-curvature priors and unused-coordinate dummies."""
    lens_sigma = np.asarray(lens_sigma, np.float64)
    brown_sigma = np.asarray(brown_sigma, np.float64)
    if (lens_sigma.shape != (4,) or brown_sigma.shape != (4,)
            or not np.isfinite(lens_sigma).all()
            or not np.isfinite(brown_sigma).all()
            or np.any(lens_sigma <= 0) or np.any(brown_sigma <= 0)):
        raise ValueError('Curvature priors need four finite positive soft scales')

    def selected_prior(key, columns, sigma):
        select = np.eye(11)[list(columns)]
        target = np.zeros(len(columns))
        if affine_priors is not None:
            affine_priors.add_selected_prior(graph, key, columns, target, sigma)
            affine_priors.add_dummy_factor(graph, key, columns)
        else:
            graph.add_observation([key], lambda q, s=select: s @ q[0],
                                  jac_fn=lambda q, s=select: [s],
                                  residual_dim=len(columns), sigma=sigma)
            unused = tuple(i for i in range(11) if i not in columns)
            dummy = np.eye(11)[list(unused)]
            graph.add_observation([key], lambda q, s=dummy: s @ q[0],
                                  jac_fn=lambda q, s=dummy: [s],
                                  residual_dim=len(unused), sigma=1.)

    for group in sorted(model.lens_base_keys):
        lens_key = model.lens_quadratic_keys[group]
        brown_key = model.brown_quadratic_keys[group]
        _node(initial, lens_key, 'lens curvature')
        _node(initial, brown_key, 'Brown curvature')
        selected_prior(lens_key, LENS_COLUMNS, lens_sigma)
        selected_prior(brown_key, BROWN_COLUMNS, brown_sigma)
    return dict(curvature_formula='0.5*dz^2',
                original_lens_slope_prior_sigma_state=[.1,.2,.2,.1],
                original_brown_slope_prior_sigma=[.5,.5,.05,.05],
                proposed_lens_curvature_sigma_state=lens_sigma.tolist(),
                proposed_brown_curvature_sigma=brown_sigma.tolist(),
                principal_and_skew_state_to_px=1000,
                new_prior_factors=2*len(model.lens_base_keys),
                new_unused_coordinate_factors=2*len(model.lens_base_keys),
                original_prior_targets_modified=False,
                hard_bounds_or_fixed_curvature=False)
