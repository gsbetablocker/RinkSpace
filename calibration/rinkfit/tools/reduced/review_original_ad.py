"""Compare complete original-format states on exact native video exposures.

Selected stills are evidence for inspection, never whole-shot acceptance.
No observations, geometry, parameters or solver state are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rinkfit_start.native_scene import NativeScene, camera_from_event


def paths_from_model(model):
    scene = NativeScene(model)
    current = model.get("current_input_scope", {}).get("schema") == "rinkfit.current_advertisement_input.v1"
    replaced = {
        p.get("advertisement_block")
        for m in model["materials"] if not m.get("unused_legacy_witness", False)
        for p in m["paths"] if p.get("advertisement_block") is not None
    } if current else set()
    result = []
    for mi, material in enumerate(model["materials"]):
        for bi, path in enumerate(material["paths"]):
            # These old paths only witness fixed parameter domains. The actual
            # current replacement is rendered above, not a duplicate old logo.
            if material.get("unused_legacy_witness", False) and path.get("advertisement_block") in replaced:
                continue
            n = 129
            stations = np.unique(np.r_[np.linspace(0, 1, n), np.linspace(0, 1, max(2, len(path.get("points", []))))])
            result.append(dict(label=material["label"], path_id=path.get("id", f"{mi}/{bi}"),
                               kind=path["kind"], xyz=scene.world(mi, bi, stations)))
    return result


def overlay(model, bgr, pts, title):
    paths = paths_from_model(model)
    world = np.concatenate([p["xyz"] for p in paths])
    camera = camera_from_event(model)
    prediction = camera.project_rs(world, np.full(len(world), (pts - 256256) / 12800.))
    out = bgr.copy()
    offset = 0
    seen = set()
    drawn = []
    for path in paths:
        n = len(path["xyz"])
        uv = prediction["uv"][offset:offset + n]
        valid = prediction["valid"][offset:offset + n]
        offset += n
        label = path["label"]
        ad = re.search(r"Ad[0-9]+", label)
        display_label = ad.group(0) if ad else label.split(":", 1)[0]
        color = (200, 60, 225) if ad else (50, 200, 230)
        first = None
        for j in range(n - 1):
            if not (valid[j] and valid[j + 1]) or np.max(np.abs(uv[j:j + 2])) > 1e6:
                continue
            a, b = np.rint(uv[j:j + 2]).astype(int)
            hit, p, q = cv2.clipLine((0, 0, 1920, 1080), tuple(a), tuple(b))
            if hit:
                cv2.line(out, p, q, color, 1, cv2.LINE_AA)
                first = p if first is None else first
        if first is not None:
            drawn.append(path["path_id"])
            if display_label not in seen:
                cv2.putText(out, display_label, first, cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
                seen.add(display_label)
    cv2.rectangle(out, (0, 0), (1920, 34), (0, 0, 0), -1)
    cv2.putText(out, f"{title} | source {pts / 12800.:.3f}s PTS {pts} | selected still, NOT ACCEPTED",
                (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .54, (255, 255, 255), 1, cv2.LINE_AA)
    return out, dict(enabled_paths=len(paths), visible_paths=drawn,
                     sampled_world_points=len(world), valid_sensor_roots=int(prediction["valid"].sum()))


def run(before, after, source, pts, output):
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    models = [json.loads(Path(p).read_text(encoding="utf8")) for p in (before, after)]
    records = []
    with av.open(str(source)) as video:
        stream = video.streams.video[0]
        if str(stream.time_base) != "1/12800" or stream.width != 1920 or stream.height != 1080:
            raise ValueError("Expected official native 1920x1080 / time base 1/12800")
        for tick in sorted(set(pts)):
            video.seek(tick, stream=stream, backward=True)
            frame = next(f for f in video.decode(stream) if f.pts >= tick)
            if frame.pts != tick:
                raise ValueError("Missing requested native exposure")
            raw = frame.to_ndarray(format="bgr24")
            images, manifests = [], []
            for model, title in zip(models, ("ORIGINAL", "CURRENT")):
                im, meta = overlay(model, raw, tick, title)
                images.append(im)
                manifests.append(meta)
            path = out / f"pts{tick}_original_current.jpg"
            cv2.imwrite(str(path), np.concatenate(images, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 96])
            records.append(dict(native_pts=tick, source_seconds=tick / 12800., image=str(path), models=manifests))
    report = dict(schema="rinkfit.original_same_forward_selected_review.v1",
                  before=dict(path=str(Path(before).resolve()), sha256=hashlib.sha256(Path(before).read_bytes()).hexdigest()),
                  after=dict(path=str(Path(after).resolve()), sha256=hashlib.sha256(Path(after).read_bytes()).hexdigest()),
                  source=str(Path(source).resolve()), all_actual_material_paths_enabled=True,
                  source_uv_or_model_changed=False, whole_shot_acceptance=False, frames=records)
    (out / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--pts", type=int, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.before, args.after, args.source, args.pts, args.output), ensure_ascii=False))
