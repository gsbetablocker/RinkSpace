"""Read-only native whole-ad comparisons using every existing model detail."""
from pathlib import Path
import argparse
import json
import sys

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rinksolve.model import load_model_bundle
from rinksolve.renderer import native_projection_preview


def compare(model_dir, video, targets, output):
    model = load_model_bundle(model_dir)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 23)
    operator = model.operator("S01")
    records = []
    cards = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for ad, pts in targets:
            container.seek(pts, stream=stream, backward=True)
            frame = next(f for f in container.decode(stream) if f.pts >= pts)
            if frame.pts != pts:
                raise ValueError("Requested native exposure is absent")
            raw = frame.to_ndarray(format="bgr24")
            overlay, identity = native_projection_preview(model, pts, raw)
            curves = [c for c in model.scene.curves if c.label.startswith(f"Ad{ad}.")]
            world = np.concatenate([model.scene.world_path(c.id, np.linspace(0, 1, 129), model.state) for c in curves])
            projected = operator.project_rs(world, np.full(len(world), pts, dtype=np.int64))
            uv = projected["uv"]
            good = (projected["status_code"] == 0) & (uv[:, 0] >= 0) & (uv[:, 0] < 1920) & (uv[:, 1] >= 0) & (uv[:, 1] < 1080)
            if not good.any():
                raise ValueError(f"Ad{ad} has no visible model path")
            lo = np.maximum(np.floor(uv[good].min(axis=0)).astype(int) - 24, 0)
            hi = np.minimum(np.ceil(uv[good].max(axis=0)).astype(int) + 24, [1920, 1080])
            box = (int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1]))
            panels = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).crop(box) for im in (raw, overlay)]
            scale = min(1.5, 920 / panels[0].width)
            size = (round(panels[0].width * scale), round(panels[0].height * scale))
            card = Image.new("RGB", (1840, size[1] + 42), "white")
            draw = ImageDraw.Draw(card)
            for j, (panel, title) in enumerate(zip(panels, ("原片", "已有完整字形与边界 / 原版位置"))):
                draw.text((j * 920 + 8, 4), f"Ad{ad} · {pts / 12800.:.2f}秒 · {title}", fill="black", font=font)
                card.paste(panel.resize(size), (j * 920, 42))
            path = out / f"Ad{ad}_pts{pts}.png"
            card.save(path)
            cards.append(card)
            records.append(dict(ad=ad, native_pts=pts, source_seconds=pts / 12800., crop_native_xyxy=box,
                                image=str(path), model_identity=identity, ad_source_curve_count=len(curves)))
    sheet = Image.new("RGB", (1840, sum(c.height for c in cards)), "white")
    y = 0
    for card in cards:
        sheet.paste(card, (0, y))
        y += card.height
    sheet.save(out / "all_advertisements.png")
    report = dict(schema="rinkspace.native_whole_ad_comparison.v1", model=str(Path(model_dir).resolve()),
                  source=str(Path(video).resolve()), parameter_sha256=model.state.digest(),
                  source_or_model_changed=False, all_model_details_enabled=True, whole_shot_acceptance=False, frames=records)
    (out / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--target", nargs=2, type=int, action="append", required=True, metavar=("AD", "PTS"))
    p.add_argument("--output", required=True)
    a = p.parse_args()
    r = compare(a.model, a.source, a.target, a.output)
    print(json.dumps({"images": [f["image"] for f in r["frames"]], "all_model_details_enabled": True}))
