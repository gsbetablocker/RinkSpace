"""Shared rotating/zooming optical-centre model with explicit residual motion.

C(t) = P + R(t).T @ (d0 + d1 * (log(f(t))-zref)) + s(t).
P is one adjustable mounting position per physical camera, d0/d1 are shared
camera-coordinate offsets, and s(t) is a per-shot world-coordinate spline.
This is a candidate mechanical model, not a measured apparatus calibration.
No mounting position, first exposure or optical centre is hard fixed.
"""
from __future__ import annotations

import cv2
import numpy as np

from continuous_camera_model import ContinuousCameraModel, initialize_from_baseline
from projection import from_state


class ContinuousRigCameraModel(ContinuousCameraModel):
    def __init__(self, *args, mount_keys, offset_keys, offset_slope_keys,
                 brown_keys=None, brown_slope_keys=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mount_keys = dict(mount_keys)
        self.offset_keys = dict(offset_keys)
        self.offset_slope_keys = dict(offset_slope_keys)
        self.brown_keys = dict(brown_keys or {})
        self.brown_slope_keys = dict(brown_slope_keys or {})
        cameras = set(self.lens_base_keys)
        for mapping in (self.mount_keys, self.offset_keys, self.offset_slope_keys):
            if set(mapping) != cameras:
                raise ValueError('Every camera needs its own mount and offset nodes')
        if set(self.brown_keys) not in (set(), cameras):
            raise ValueError('All cameras or none must use shared Brown4')
        if set(self.brown_slope_keys) - set(self.brown_keys):
            raise ValueError('Brown zoom slope needs a shared base node')
        extra = [key for mapping in (self.mount_keys, self.offset_keys,
                 self.offset_slope_keys, self.brown_keys, self.brown_slope_keys)
                 for key in mapping.values()]
        parent = super()
        old = {key for p in self.frames for key in parent.dependencies(p)}
        if len(set(extra)) != len(extra) or old.intersection(extra):
            raise ValueError('Mechanical/Brown graph nodes must be disjoint')

    def dependencies(self, pts):
        camera_id = self.frames[int(pts)].camera_id
        keys = list(super().dependencies(pts))
        for mapping in (self.mount_keys, self.offset_keys, self.offset_slope_keys,
                        self.brown_keys, self.brown_slope_keys):
            if camera_id in mapping:
                keys.append(mapping[camera_id])
        return tuple(keys)

    def _physical(self, pts, states, jacobian):
        result = super()._physical(pts, states, jacobian)
        physical, blocks = result if jacobian else (result, None)
        binding = self.frames[int(pts)]
        group = binding.camera_id
        frame = states[binding.frame_key]
        dz = frame[6]-self.lens_zoom_reference[group]
        mount_key = self.mount_keys[group]
        offset_key = self.offset_keys[group]
        slope_key = self.offset_slope_keys[group]
        offset = states[offset_key][:3]+dz*states[slope_key][:3]
        rotation, derivative = cv2.Rodrigues(np.asarray(frame[3:6], float))
        dr = derivative.reshape(3, 3, 3)
        physical[:3] += states[mount_key][:3]+rotation.T @ offset
        if jacobian:
            blocks[binding.frame_key][:3, 3:6] += np.column_stack(
                [dr[j].T @ offset for j in range(3)])
            blocks[binding.frame_key][:3, 6] += rotation.T @ states[slope_key][:3]
            for key, matrix in ((mount_key, np.eye(3)),
                                (offset_key, rotation.T),
                                (slope_key, dz*rotation.T)):
                block = np.zeros((11, 11))
                block[:3, :3] = matrix
                blocks[key] = block
        if group in self.brown_keys:
            brown_key = self.brown_keys[group]
            brown = states[brown_key][:4].copy()
            brown_slope_key = self.brown_slope_keys.get(group)
            if brown_slope_key is not None:
                brown += dz*states[brown_slope_key][:4]
            physical = np.r_[physical, brown]
            if jacobian:
                blocks = {key: np.vstack((block, np.zeros((4, 11))))
                          for key, block in blocks.items()}
                block = np.zeros((15, 11))
                block[11:15, :4] = np.eye(4)
                blocks[brown_key] = block
                if brown_slope_key is not None:
                    blocks[binding.frame_key][11:15, 6] = states[brown_slope_key][:4]
                    block = np.zeros((15, 11))
                    block[11:15, :4] = dz*np.eye(4)
                    blocks[brown_slope_key] = block
        return (physical, blocks) if jacobian else physical

    def camera25(self, pts, states):
        physical = self.physical_state(pts, states)
        camera = from_state(physical[:11], self.frames[int(pts)].template25)
        if len(physical) == 15:
            camera[11:25] = 0.
            camera[11:15] = physical[11:15]
        return camera


def initialize_rig(model, baseline):
    """Previous cameras supply only initialization, never image observations."""
    plain = ContinuousCameraModel(model.shot_splines, model.frames,
        model.lens_base_keys, model.lens_slope_keys, model.lens_zoom_reference)
    initial, report = initialize_from_baseline(plain, baseline)
    for spline in model.shot_splines.values():
        for key in spline.control_keys:
            initial[key][:3] = 0.
    for group, key in model.mount_keys.items():
        centers = [baseline[p][:3] for p, b in model.frames.items() if b.camera_id == group]
        initial[key] = np.zeros(11)
        initial[key][:3] = np.median(centers, axis=0)/100.
        initial[model.offset_keys[group]] = np.zeros(11)
        initial[model.offset_slope_keys[group]] = np.zeros(11)
    for mapping in (model.brown_keys, model.brown_slope_keys):
        for key in mapping.values():
            initial[key] = np.zeros(11)
    report['mechanical_initialization'] = 'shared median C, zero offsets and residual motion'
    report['brown_initialization'] = 'one shared zero Brown4 candidate, not measured calibration'
    return initial, report
