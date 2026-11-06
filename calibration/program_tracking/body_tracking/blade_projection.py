"""Read-only blade queries from one saved absolute-motion candidate.

The current case inputs and authoritative final curve are sufficient; this
adapter does not reconstruct historical initialization or register any DOFs.
Lengths, including ``local_points`` and returned world nodes, stay in solver
units. Multiply by ``ratio`` once when comparing with exported world arrays.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .absolute_motion_state import absolute_reference, curve_from_packet
from .footwear_layout import OBSERVATION_LAYOUT
from .fullbody_state import camera_binding, matrices
from .hand_state import META, hand_curves
from .rebuild_body_geometry import CompactBody
from .rebuild_exposure_body import moving_projection
from .skate_binding import SkateBinding
from .skate_physics import fk_state
from .skate_template import SkateParameters


def _load(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


class SavedBladeProjection:
    """Evaluate S05 saved state and explicit, call-local counterfactuals.

    ``parameter`` is a current saved parameter path or an already loaded
    packet. ``reference`` exposes the saved final ``at(times)`` trajectory;
    no historical parent reference is required. Camera projection uses the
    existing frozen binding and its native rolling-shutter implementation.
    """

    def __init__(self, project, stage, parameter):
        self.project = Path(project).resolve()
        self.stage = Path(stage).resolve()
        case = self.stage / 'common/exmr_hand_nofoot'
        inputs = _load(case / 'input.npz')
        initial = _load(case / 'initial.npz')
        saved = _load(parameter) if isinstance(parameter, (str, os.PathLike)) else parameter
        self.parameter = saved
        self.curve = curve_from_packet(saved)
        self.pts = np.asarray(inputs['pts']).copy()
        self.times = np.asarray(inputs['fit_time_seconds'], dtype=float).copy()
        np.testing.assert_array_equal(initial['pts'], self.pts)
        np.testing.assert_allclose(self.times, (self.pts - self.pts[0]) / 12800.,
                                   rtol=0, atol=1e-12)
        saved_camera = json.loads(str(saved['camera_binding_json']))
        self.ratio = float(saved_camera['length_export_ratio'])
        if not np.isfinite(self.ratio) or self.ratio <= 0:
            raise ValueError('Expected a positive saved length_export_ratio')

        # Rehydrate only the saved static registration and template. The full
        # SkateBinding constructor would recreate historical skin fitting.
        binding = object.__new__(SkateBinding)
        binding.base_parameters = SkateParameters(**json.loads(str(saved['base_skate_parameters_json'])))
        binding.R0 = np.asarray(saved['R_FB_initial'], dtype=float).copy()
        binding.t0 = np.asarray(saved['t_FB_initial_solver'], dtype=float).copy()
        binding.x = np.asarray(saved['x'][:17], dtype=float).copy()
        binding.ratio = self.ratio
        binding.layout = OBSERVATION_LAYOUT
        binding._parameter_key = None
        binding._packet = None
        self.binding = binding

        body = CompactBody(case / 'compact_smplx.npz', inputs['shared_betas'],
                           float(inputs['body_scale']))
        hands = None
        if 'hand_local_R' in initial:
            hand_times = ((initial['hand_prior_pts'] - self.pts[0]) / 12800.
                          if 'hand_prior_pts' in initial else self.times)
            hands = hand_curves(hand_times, initial.get('hand_prior_local_R', initial['hand_local_R']))
        elif any(key in initial for key in META):
            raise ValueError('Hand metadata present without archived hand rotations')
        shared = SimpleNamespace(p=inputs, times=self.times, body=body,
                                 binding=binding, hand_curves=hands,
                                 hand_metadata={key: initial[key] for key in META if key in initial},
                                 layout=OBSERVATION_LAYOUT, root=case)
        self.reference = absolute_reference(shared, self.curve, root=case)
        from .sequence_skate_state import shoe_curve_from_packet
        self.reference.relative_skate_curve=shoe_curve_from_packet(saved)
        native = os.environ.get('RINKSPACE_BODY_NATIVE_LIBRARY') or saved_camera['native_library']
        self.camera, self.camera_info, _, _ = camera_binding(
            self.project, self.reference, record_path=saved_camera['record'],
            native_library=native, shot_id='s05')
        np.testing.assert_allclose(self.camera_info['length_export_ratio'], self.ratio,
                                   rtol=0, atol=1e-12)
        self._frames = {int(pts): index for index, pts in enumerate(self.pts)}

    def project_points(self, pts, foot, local_points, static_delta=None,
                       ankle_delta=None, include_body=False):
        """Return native ``uv`` and five world ``nodes`` for local blade points.

        ``pts`` is (N,), ``foot`` is 0/1 or (N,), and ``local_points`` is
        (K,3) or (N,K,3). ``static_delta`` is one right rotvec on the selected
        saved R_FB, preserving the body and shoe origin. ``ankle_delta`` is
        (N,3), right composed onto the selected anatomical ankle local_R
        (native joint 7/8) at each exposure node, followed by actual FK.
        Both perturbations are temporary and never alter the saved state.

        ``R`` and ``O`` contain the selected foot at every node. With
        ``include_body=True``, ``coco_uv``/``COCO_uv`` are the actual 17-point
        CompactBody vertex-regressor projection, with the same hand state,
        five node times and implicit row solve as the blade projection.
        """
        pts = np.asarray(pts)
        if pts.ndim != 1 or not len(pts) or not np.isfinite(pts).all():
            raise ValueError('Expected finite nonempty pts (N,)')
        if np.any(pts != pts.astype(np.int64)):
            raise ValueError('PTS must be source-clock integers')
        try:
            frames = np.array([self._frames[int(value)] for value in pts], dtype=int)
        except KeyError as exc:
            raise ValueError('Query PTS outside current source exposures') from exc
        n = len(frames)
        feet = np.broadcast_to(np.asarray(foot), (n,))
        if not np.isin(feet, (0, 1)).all():
            raise ValueError('Physical foot must be 0 or 1')
        feet = feet.astype(int)
        local = np.asarray(local_points, dtype=float)
        if local.ndim == 2 and local.shape[-1] == 3:
            local = np.broadcast_to(local, (n, *local.shape))
        if local.ndim != 3 or local.shape[0] != n or local.shape[2] != 3 or not local.shape[1]:
            raise ValueError('Expected local_points (K,3) or (N,K,3)')
        if not np.isfinite(local).all():
            raise ValueError('Nonfinite local point')

        node_times = (self.times[frames, None] + np.linspace(-.5, .5, 5)
                      * float(self.camera.p['rig'][37]))
        ref = self.reference.at(node_times.ravel())
        delta = np.zeros((n * 5, 69))
        if ankle_delta is not None:
            ankle = np.asarray(ankle_delta, dtype=float)
            if ankle.shape != (n, 3) or not np.isfinite(ankle).all():
                raise ValueError('Expected finite ankle_delta (N,3)')
            # Joint 7/8 local rotation occupies body local index 6/7.
            columns = 6 + (6 + np.repeat(feet, 5))[:, None] * 3 + np.arange(3)
            delta[np.arange(n * 5)[:, None], columns] = np.repeat(ankle, 5, axis=0)
        state = fk_state(self.reference, ref, delta)
        pick = np.arange(n * 5)
        foot_nodes = np.repeat(feet, 5)
        rotation = state['R'][pick, foot_nodes].reshape(n, 5, 3, 3)
        origin = state['O'][pick, foot_nodes].reshape(n, 5, 3)
        if static_delta is not None:
            static = np.asarray(static_delta, dtype=float)
            if static.shape != (3,) or not np.isfinite(static).all():
                raise ValueError('Expected finite static_delta rotvec (3,)')
            rotation = rotation @ matrices(static)
        nodes = origin[:, :, None] + np.einsum('nlij,nkj->nlki', rotation, local)
        projected = moving_projection(self.camera, nodes, frames)
        result = dict(projected, nodes=nodes, R=rotation, O=origin,
                      frames=frames, node_times=node_times, foot=feet)
        if include_body:
            kwargs = ({} if 'hand_local_R' not in state
                      else dict(hand_local_R=state['hand_local_R']))
            coco = self.reference.body.forward_rotations(state['local_R'], **kwargs)[0]
            coco_world = state['P'][:, None] + np.einsum('nij,nkj->nki', state['Q'], coco)
            coco_nodes = coco_world.reshape(n, 5, 17, 3)
            coco_projection = moving_projection(self.camera, coco_nodes, frames)
            result.update(coco_uv=coco_projection['uv'], COCO_uv=coco_projection['uv'],
                          coco_nodes=coco_nodes, coco_projection=coco_projection)
        return result
