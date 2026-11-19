"""Replay a single programme-wide body scale without resizing the equipment."""
import numpy as np

from calibration.program_tracking.body_tracking.rebuild_body_geometry import CompactBody


def apply_body_scale(ref, scale):
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid programme body scale')
    surface = ref.binding.surface
    body = CompactBody(ref.root/'compact_smplx.npz', ref.p['shared_betas'], scale,
                       lift=ref.body.lift, rocker=ref.body.rocker)
    ref.body = surface.body = body
    ref.p['body_scale'] = np.array(scale)
    surface.packet['body_scale'] = np.array(scale)
    surface._sample_double_cache = None
    binding = ref.binding
    _, joints, _, frames = body.forward_rotations(np.broadcast_to(np.eye(3),(1,21,3,3)), with_frames=True)
    binding.neutral_native, binding.neutral_frames = joints, frames
    binding.neutral_samples = surface.sample_vertices(joints, frames, binding.sample_ids, double_precision=True)
