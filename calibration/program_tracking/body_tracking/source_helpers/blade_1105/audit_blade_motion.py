#!/usr/bin/env python3
"""Read-only blade-axis continuity audit for RinkSpace saved result NPZ.

This reports motion of the *saved model-derived* blade axis. It is not a
measurement of physical blade orientation and does not start a fit.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    with np.load(a.input, allow_pickle=False) as z:
        required = ('pts', 'blade_material_world_mid')
        missing = [k for k in required if k not in z]
        if missing:
            raise ValueError(f'Missing arrays: {missing}')
        pts = np.asarray(z['pts'], np.int64)
        blade = np.asarray(z['blade_material_world_mid'], float)
        if pts.ndim != 1 or len(pts) < 2 or np.any(np.diff(pts) <= 0):
            raise ValueError('PTS must be strictly increasing')
        if blade.shape != (len(pts), 2, 3, 3) or not np.isfinite(blade).all():
            raise ValueError('Expected finite blade_material_world_mid [N,2,rear-middle-front,xyz]')
        tb = tuple(map(int, z['time_base'])) if 'time_base' in z else (1, 12800)
        seconds = pts * tb[0] / tb[1]
        axis = blade[:, :, 2] - blade[:, :, 0]
        norm = np.linalg.norm(axis, axis=-1)
        if np.any(norm <= 1e-9):
            raise ValueError('Degenerate rear-front material axis')
        axis /= norm[..., None]
        dot = np.sum(axis[1:] * axis[:-1], axis=-1)
        angle = np.arccos(np.clip(dot, -1., 1.))
        dt = np.diff(seconds)
        rate = angle / dt[:, None]
        rows = []
        for foot in (0, 1):
            x = rate[:, foot]
            rows.append(dict(foot=foot, p50_rad_s=float(np.quantile(x, .50)),
                             p95_rad_s=float(np.quantile(x, .95)),
                             maximum_rad_s=float(x.max())))
        peak = np.max(rate, axis=1)
        indices = np.argsort(peak)[-20:][::-1]
        events = [dict(start_pts=int(pts[i]), end_pts=int(pts[i+1]),
                       start_seconds=float(seconds[i]), end_seconds=float(seconds[i+1]),
                       foot0_rad_s=float(rate[i, 0]), foot1_rad_s=float(rate[i, 1]),
                       axis_foot0_start=axis[i, 0].tolist(), axis_foot0_end=axis[i+1, 0].tolist(),
                       axis_foot1_start=axis[i, 1].tolist(), axis_foot1_end=axis[i+1, 1].tolist())
                  for i in indices]
        result = dict(schema='rinkspace.saved-blade-axis-continuity-audit.v1',
                      input=str(a.input), frames=len(pts), time_base=list(tb),
                      first_last_pts=[int(pts[0]), int(pts[-1])],
                      first_last_seconds=[float(seconds[0]), float(seconds[-1])],
                      definition='normalized front-minus-rear saved blade material points; adjacent 3D direction angle divided by actual PTS dt',
                      interpretation='model-derived continuity diagnostic only; not physical ground truth or a speed acceptance threshold',
                      summary=rows, largest_adjacent_direction_changes=events)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=str(a.output), frames=result['frames'], summary=rows), ensure_ascii=False))

if __name__ == '__main__':
    main()
