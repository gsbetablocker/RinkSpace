"""Read native frames and detect per-exposure foreground exclusion boxes."""
from __future__ import annotations

from itertools import islice
import os
from time import monotonic

from common import DATA, OUTPUT, scope, save, source_frames, source_pts

import numpy as np


def run():
    out = OUTPUT / "foreground"
    out.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(out / "runtime")
    from ultralytics import YOLO

    detector = YOLO(str(DATA / "cache/pretrained/yolo11n-pose.pt"))
    record = scope()
    path = record["source"]["registered_path"]
    for shot in record["shots"]:
        dest = out / f"{shot['id']}.npz"
        if dest.exists():
            continue
        then = monotonic()
        expected = source_pts(shot)
        stream = source_frames(path, expected)
        boxes, offsets, clock = [], [0], []
        while batch := list(islice(stream, 12)):
            rgb = [frame.to_ndarray(format="bgr24") for _, frame in batch]
            results = detector.predict(rgb, imgsz=640, conf=.21, device=0, half=True,
                                       verbose=False, max_det=12)
            for (pts, _), result in zip(batch, results, strict=True):
                box = result.boxes.xyxy.cpu().numpy()
                conf = result.boxes.conf.cpu().numpy()
                filtered = (box[:, 2]-box[:, 0] >= 18) & (box[:, 3]-box[:, 1] >= 25)
                boxes.append(np.c_[box[filtered], conf[filtered]])
                clock.append(pts)
                offsets.append(offsets[-1] + np.count_nonzero(filtered))
        np.testing.assert_array_equal(clock, expected)
        save(dest, pts=expected, offsets=np.array(offsets, np.int64),
             boxes=np.concatenate(boxes, axis=0), detector="yolo11n-pose.pt", source_sha256=record["source"]["sha256"])
        print(shot["id"], "native frames", len(clock), "frames with personboxes",
              int(np.count_nonzero(np.diff(offsets))), f"{monotonic()-then:.1f}s", flush=True)


if __name__ == "__main__":
    run()
