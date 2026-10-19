"""Native RGB material-profile sampling on CUDA, without semantic admission.

Coordinates use the existing OpenCV 1/32-pixel interpolation grid. All RGB
calculations stay in full precision; neither autocast nor a trained probability
map changes the source pixels. This moves image sampling and median reduction to
one GPU instead of replicating full images in many CPU workers.
"""
from __future__ import annotations

import numpy as np


def batched_actual_profiles_cuda(frame, query_uv, normals, intervals, signals, *, _debug_query=None):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA profile backend requested but no CUDA device is available')
    count = len(query_uv)
    size = int(np.ceil(np.max(intervals[:, 1]-intervals[:, 0])))+1 if count else 0
    if not count:
        empty = np.empty((0, size))
        return empty, empty.copy(), empty.copy(), empty.astype(bool)
    device = torch.device('cuda')
    with torch.inference_mode():
        points = torch.as_tensor(query_uv, dtype=torch.float64, device=device)
        directions = torch.as_tensor(normals, dtype=torch.float64, device=device)
        limits = torch.as_tensor(intervals, dtype=torch.float64, device=device)
        offsets = limits[:, :1]+torch.arange(size, dtype=torch.float64, device=device)[None]
        valid = offsets <= limits[:, 1:]+.25
        tangent = torch.stack((-directions[:, 1], directions[:, 0]), dim=1)
        xy = (points[:, None, None]+directions[:, None, None]*offsets[:, None, :, None]
            + tangent[:, None, None]*torch.arange(-2, 3, dtype=torch.float64, device=device)[None, :, None, None])
        # CPU cv2.remap receives float32 maps, then rounds to its interpolation
        # table. Preserve that lattice, including tie-to-even conversion.
        xy = torch.round(xy.to(torch.float32).to(torch.float64)*32.)/32.
        height, width = frame.rgb.shape[:2]
        rgb = torch.as_tensor(frame.rgb.copy(), device=device, dtype=torch.float32)
        luminance = torch.as_tensor(frame.luminance.astype(np.float32), device=device)
        fields = torch.stack((luminance, rgb[..., 0]-rgb[..., 1], rgb[..., 2]-rgb[..., 0]))
        channels = torch.as_tensor([{'dark': 0, 'red': 1, 'blue': 2}[str(s)] for s in signals],
            device=device, dtype=torch.long)
        base = torch.floor(xy).to(torch.long)
        fraction = (xy-base).to(torch.float32)
        fx, fy = fraction[..., 0], fraction[..., 1]
        # Explicit source pixels/1/1024 weights avoid normalized-grid rounding.
        # This matches the existing native interpolation table on the device.
        parts = []
        for dx, dy, weight in ((0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                (0, 1, (1-fx)*fy), (1, 1, fx*fy)):
            x, y = base[..., 0]+dx, base[..., 1]+dy
            inside = (x >= 0)&(x < width)&(y >= 0)&(y < height)
            values = fields[channels[:, None, None], y.clamp(0, height-1), x.clamp(0, width-1)]
            parts.append(values*weight*inside)
        selected = (parts[0]+parts[1])+(parts[2]+parts[3])
        profiles = selected.median(dim=1).values.to(torch.float64)
        last = valid.sum(1)-1
        profiles = torch.where(valid, profiles, profiles.gather(1, last[:, None]))
        # Keep the tiny 1D source filter's original arithmetic ordering. CUDA
        # convolution differs at 1e-13 on flat profiles and can create a false
        # extra threshold component. Heavy native-image sampling/median stays
        # on CUDA; this single-thread filter preserves exact candidate choices.
        import cv2
        profiles_native = profiles.cpu().numpy()
        smooth = cv2.GaussianBlur(profiles_native, (5, 1), 1., borderType=cv2.BORDER_REPLICATE)
        result = (offsets.cpu().numpy(), profiles_native, smooth, valid.cpu().numpy())
        if _debug_query is not None:
            qi, si = _debug_query
            return (*result, dict(quantized_xy=xy[qi, :, si].cpu().numpy().tolist(),
                samples=selected[qi, :, si].cpu().numpy().tolist()))
        return result
