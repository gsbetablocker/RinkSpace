"""Deterministic synthetic geometry demonstration; never presented as rink evidence."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from .geometry import fit_homography, project, heldout_audit, PlaneMap
from .mosaic import ProvenanceMosaic
from .motion import kinematics, window_sensitivity
from .io import dump_json


def synthetic_demo(out: str | Path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    ground_truth = np.array([[.041, .003, -3.0], [.002, -.043, 17.0], [.0002, -.00015, 1.0]])
    train = rng.uniform([45, 50], [800, 490], (40, 2))
    check = np.array([[25, 25], [840, 25], [25, 515], [840, 515], [430, 18], [860, 270]], float)
    target = project(ground_truth, train)
    fitted, inliers = fit_homography(train, target + rng.normal(0, .002, target.shape), .015)
    audit = heldout_audit(fitted, train, [f"train{i}" for i in range(len(train))], check,
                         project(ground_truth, check), [f"check{i}" for i in range(len(check))], 0.0)
    plane = PlaneMap(fitted, "metric_verified", "m", map_id="synthetic_truth_only")
    rows = []
    for t in np.arange(0, 4.0001, .02):
        theta = .35 * t
        rows.append({"t_s": float(t), "x": float(8 * np.cos(theta)), "y": float(8 * np.sin(theta)),
                     "sigma": .005, "contact": "on_ice", "status": "accepted", "video_role": "live",
                     "shot_id": "synthetic", "map_id": plane.map_id, "entity_id": "right_blade",
                     "point_definition": "synthetic_ground_contact", "epoch": "synthetic"})
    curve = kinematics(rows, plane)
    sensitivity = window_sensitivity(rows, plane)
    dump_json(out / "trajectory_input.json", rows)
    dump_json(out / "kinematics.json", curve)
    dump_json(out / "window_sensitivity.json", sensitivity)
    # Two overlapping source tiles. Transparency + pixel-source IDs stay explicit.
    tile = np.full((250, 420, 3), 235, np.uint8)
    for x in range(10, 420, 35): cv2.line(tile, (x, 0), (x, 249), (145, 145, 145), 1)
    for y in range(10, 250, 35): cv2.line(tile, (0, y), (419, y), (145, 145, 145), 1)
    cv2.putText(tile, "SYNTHETIC TILE", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, .8, (30, 30, 30), 2)
    m = ProvenanceMosaic(700, 340, "synthetic")
    mask = np.ones(tile.shape[:2], np.uint8) * 255
    m.add(tile, mask, np.eye(3), 0, .8, "synthetic")
    m.add(tile, mask, np.array([[1,0,240],[0,1,40],[0,0,1]],float), 1, .9, "synthetic")
    cv2.imwrite(str(out / "mosaic.png"), m.rgba())
    np.save(out / "mosaic_source_index.npy", m.source_index)
    canvas = np.full((580, 900, 3), 245, np.uint8)
    for r in rows[::5]:
        x, y = int(130 + r["x"] * 70), int(500 - r["y"] * 55)
        cv2.circle(canvas, (x, y), 3, (40, 40, 40), -1)
    cv2.putText(canvas, "SYNTHETIC: radius=8m, speed=2.8m/s", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, .85, (20, 20, 20), 2)
    cv2.putText(canvas, "NOT measurements from the uploaded videos", (30, 76), cv2.FONT_HERSHEY_SIMPLEX, .65, (20, 20, 20), 1)
    cv2.imwrite(str(out / "trajectory_preview.png"), canvas)
    result = {"dataset": "synthetic_only", "seed": 42, "homography_inliers": int(inliers.sum()), "heldout": audit,
              "known_speed_m_s": 2.8, "fitted_median_speed_m_s": float(np.median([r["speed"] for r in curve])),
              "known_radius_m": 8.0, "fitted_median_radius_m": float(np.median([r["radius"] for r in curve])),
              "real_video_capability_proven": False}
    dump_json(out / "summary.json", result)
    (out / "index.html").write_text("""<!doctype html><meta charset='utf-8'><title>Synthetic test</title>
<style>body{font:18px/1.7 system-ui;max-width:950px;margin:30px auto}img{max-width:100%}</style>
<h1>合成数据：几何与导数链路测试</h1><p>只验证已知几何下的基础代码；不是上传视频的冰面轨迹。</p>
<img src='trajectory_preview.png'><h2>透明全景与来源索引</h2><img src='mosaic.png'>
<p><a href='summary.json'>测试摘要</a> · <a href='kinematics.json'>速度 / 曲率</a> · <a href='window_sensitivity.json'>时间窗口敏感性</a></p>""", encoding="utf-8")
    return result
