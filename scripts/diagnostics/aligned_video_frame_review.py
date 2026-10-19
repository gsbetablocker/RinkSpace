"""Render exact source PTS with all existing rink/ad geometry for review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'calibration/program_tracking')]
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np
from render_result import SceneRenderer, decode_selected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--camera', type=Path, required=True)
    ap.add_argument('--scene', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--pts', type=int, nargs='+', required=True)
    ap.add_argument('--label', default='candidate')
    args = ap.parse_args()
    cv2.setNumThreads(1)
    source = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/media/originals/oxXu1avhrvs.mp4')
    with np.load(args.camera, allow_pickle=False) as z:
        cameras = z['camera25'].copy()
        stamps = z['pts'].copy()
        metadata = json.loads(str(z['metadata_json']))
    if metadata['source_sha256'] != 'f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891':
        raise ValueError('Camera belongs to another source')
    index = {int(p): i for i, p in enumerate(stamps)}
    if any(p not in index for p in args.pts):
        raise ValueError('Missing requested native camera PTS')
    renderer = SceneRenderer(json.loads(args.scene.read_text(encoding='utf8')),
                             graphics_regions=((28, 24, 554, 253), (1690, 944, 1865, 1000)))
    args.output.mkdir(parents=True, exist_ok=True)
    for pts, frame in decode_selected(source, sorted(set(args.pts))):
        rendered, metrics = renderer.draw(frame, cameras[index[int(pts)]])
        stem = f'{args.label}_{pts}'
        path = args.output / f'{stem}_all_geometry.jpg'
        if not cv2.imwrite(str(path), rendered, [cv2.IMWRITE_JPEG_QUALITY, 96]):
            raise RuntimeError(f'Could not write {path}')
        raw = args.output / f'native_{pts}.jpg'
        if not raw.exists():
            cv2.imwrite(str(raw), frame, [cv2.IMWRITE_JPEG_QUALITY, 97])
        print(json.dumps(dict(pts=int(pts), overlay=str(path), raw=str(raw),
                              rendered_model_counts=renderer.path_counts), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
