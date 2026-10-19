"""Render and inspect a two-second fixed-geometry camera track.

The only geometry source is deliverables/scene.json. Every ice grid line,
marking, board edge and advertisement edge is sampled in world coordinates
and projected with the same central-branch Brown model as the three-frame fit.
No reference-image warp or per-frame rink deformation is used.

Usage (after source.mp4 and poses.json exist in the run directory):
  .venv/Scripts/python.exe -B calibration/shot_tracking/pipeline/review.py \
      --run calibration/shot_tracking/runs/low_end_pm1
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import sys
from urllib.parse import quote

import cv2
import imageio_ffmpeg
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "calibration" / "work"
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))
from physical_projection import safe_project  # noqa: E402


WIDTH, HEIGHT, FPS = 1920, 1080, 60
ANCHOR_PTS = 104693760
FIRST_PTS = ANCHOR_PTS - 15360
LAST_PTS = ANCHOR_PTS + 15360
PTS_STEP = 256
COUNT = 121
LAYER_ORDER = ("grid", "markings", "contact", "board", "ads")
LAYER_COLORS_BGR = {
    "grid": (195, 210, 43),      # thin cyan/yellow grid
    "markings": (207, 72, 228),  # magenta existing painted markings
    "contact": (61, 239, 100),  # green ice/boards contact edge
    "board": (251, 177, 59),    # blue board top
    "ads": (40, 174, 255),      # orange ad borders
}
LAYER_ALPHA = {"grid": .56, "markings": .80, "contact": .90,
               "board": .74, "ads": .70}
LAYER_WIDTH = {"grid": 1, "markings": 2, "contact": 2, "board": 2, "ads": 1}
# Fixed television graphics in this specific two-second source window.
HUD_RECTS = ((31, 28, 550, 103), (79, 105, 400, 251), (1730, 948, 1828, 998))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _as_xyz(value: object, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64).reshape(-1, 3)
    if len(points) < 2 or not np.isfinite(points).all():
        raise ValueError(f"Invalid world path {name}")
    return points


def _densify(points: np.ndarray, max_step: float = .28) -> np.ndarray:
    """Resolve Brown-curved images of physically straight model edges."""
    result = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        count = max(1, math.ceil(float(np.linalg.norm(b - a)) / max_step))
        result.extend(a + (b - a) * (k / count) for k in range(1, count + 1))
    return np.asarray(result, dtype=np.float64)


def _circle(center: object, radius: float) -> np.ndarray:
    c = np.asarray(center, dtype=np.float64)
    n = max(96, math.ceil(2 * math.pi * radius / .22))
    theta = np.linspace(0.0, 2 * math.pi, n + 1)
    return np.column_stack((c[0] + radius * np.cos(theta),
                            c[1] + radius * np.sin(theta),
                            np.full(len(theta), c[2])))


def _grid_from_exported_cells(grid: dict) -> list[tuple[str, np.ndarray]]:
    """Recover every exported 5-unit grid line from clipped cell vertices."""
    cells = grid["cells"]
    vertices = np.asarray([p for cell in cells for p in cell["polygon_xyz"]],
                          dtype=np.float64)
    origin = np.asarray(grid["origin_xyz"], float)
    sx, sy = map(float, grid["spacing_xy"])
    paths: list[tuple[str, np.ndarray]] = []
    for col in range(1, int(grid["columns"])):
        x = origin[0] + col * sx
        ys = vertices[np.abs(vertices[:, 0] - x) < 2e-5, 1]
        if len(ys) > 1 and float(ys.max() - ys.min()) > 1e-5:
            paths.append((f"grid_x_{col}", _densify(np.array(
                [[x, ys.min(), 0.0], [x, ys.max(), 0.0]]))))
    for row in range(1, int(grid["rows"])):
        y = origin[1] + row * sy
        xs = vertices[np.abs(vertices[:, 1] - y) < 2e-5, 0]
        if len(xs) > 1 and float(xs.max() - xs.min()) > 1e-5:
            paths.append((f"grid_y_{row}", _densify(np.array(
                [[xs.min(), y, 0.0], [xs.max(), y, 0.0]]))))
    if not paths:
        raise ValueError("No grid lines recovered from exported ice grid")
    return paths


def world_paths(scene: dict) -> list[tuple[str, str, np.ndarray]]:
    """Return actual scene curves, including true open painted edges."""
    if scene.get("schema_version") != "rinkspace.three_frame_refit.scene.v1":
        raise ValueError("Unexpected scene schema")
    paths: list[tuple[str, str, np.ndarray]] = []

    def add(layer: str, name: str, points: object, *, dense: bool = True) -> None:
        xyz = _as_xyz(points, name)
        paths.append((layer, name, _densify(xyz) if dense else xyz))

    for name, xyz in _grid_from_exported_cells(scene["ice_grid"]):
        add("grid", name, xyz, dense=False)

    boundary = _as_xyz(scene["ice_boundary"]["points_xyz"], "ice contact")
    add("contact", "ice contact", np.vstack((boundary, boundary[0])))
    top = _as_xyz(scene["board_surface"]["top_edge_xyz"], "board top")
    add("board", "board top", np.vstack((top, top[0])))

    for feature in scene["ice_features"]:
        feature_id = feature["id"]
        kind = feature["kind"]
        if kind == "circle":
            rings = feature.get("rings") or [feature]
            for ring in rings:
                name = f"{feature_id}/{ring.get('role', 'outer')}"
                add("markings", name, _circle(ring["center_xyz"], float(ring["radius"])), dense=False)
        elif kind == "polyline":
            for i, segment in enumerate(feature["segments_xyz"]):
                add("markings", f"{feature_id}/{i}", segment)
        elif kind == "polygon":
            vertices = _as_xyz(feature["polygon_xyz"], feature_id)
            closed = bool(feature.get("closed", False))
            count = len(vertices) if closed else len(vertices) - 1
            omitted = set(feature.get("open_edges", []))
            for i in range(count):
                if i not in omitted:
                    add("markings", f"{feature_id}/edge{i}",
                        [vertices[i], vertices[(i + 1) % len(vertices)]])
        else:
            raise ValueError(f"Unsupported fitted feature kind {kind}: {feature_id}")

    for ad in scene["advertisements"]:
        for role, curve in ad["fitted_boundary_curves_xyz"].items():
            add("ads", f"ad{ad['user_number']}/{role}", curve)
    return paths


def _clip(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Liang-Barsky clip in the native 1920x1080 image."""
    d = b - a
    lo, hi = 0.0, 1.0
    for p, q in ((-d[0], a[0]), (d[0], WIDTH - 1.0 - a[0]),
                 (-d[1], a[1]), (d[1], HEIGHT - 1.0 - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None
        elif p < 0:
            lo = max(lo, q / p)
        else:
            hi = min(hi, q / p)
        if lo > hi:
            return None
    return a + lo * d, a + hi * d


def _visible_runs(uv: np.ndarray, valid: np.ndarray) -> list[np.ndarray]:
    runs: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for i in range(len(uv) - 1):
        if not (valid[i] and valid[i + 1]):
            if len(current) > 1:
                runs.append(np.asarray(current))
            current = []
            continue
        if np.linalg.norm(uv[i + 1] - uv[i]) > 1200.0:
            if len(current) > 1:
                runs.append(np.asarray(current))
            current = []
            continue
        clipped = _clip(uv[i], uv[i + 1])
        if clipped is None:
            if len(current) > 1:
                runs.append(np.asarray(current))
            current = []
            continue
        a, b = clipped
        if len(current) and np.linalg.norm(current[-1] - a) > .02:
            if len(current) > 1:
                runs.append(np.asarray(current))
            current = []
        if not len(current):
            current.append(a)
        current.append(b)
    if len(current) > 1:
        runs.append(np.asarray(current))
    return runs


def _svg(runs: list[np.ndarray]) -> str:
    chunks = []
    for run in runs:
        chunks.append("M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in run))
    return "".join(chunks)


class Projector:
    def __init__(self, paths: list[tuple[str, str, np.ndarray]]):
        self.paths = paths
        self.offsets = [0]
        for _, _, world in paths:
            self.offsets.append(self.offsets[-1] + len(world))
        self.world = np.vstack([world for _, _, world in paths])

    def project(self, camera_vector: object) -> tuple[dict[str, list[np.ndarray]], dict[str, str]]:
        vector = np.asarray(camera_vector, float)
        if vector.shape != (25,) or not np.isfinite(vector).all() or np.any(np.abs(vector[23:]) > 1e-12):
            raise ValueError("Expected finite 25-value camera vector with zero Brown tilt")
        uv, _, valid = safe_project(vector, self.world)
        groups: dict[str, list[np.ndarray]] = {layer: [] for layer in LAYER_ORDER}
        for i, (layer, _, _) in enumerate(self.paths):
            a, b = self.offsets[i:i + 2]
            groups[layer].extend(_visible_runs(uv[a:b], valid[a:b]))
        return groups, {layer: _svg(runs) for layer, runs in groups.items()}


def _draw(frame: np.ndarray, paths: dict[str, list[np.ndarray]], mask: np.ndarray | None) -> np.ndarray:
    image = frame.copy()
    for layer in LAYER_ORDER:
        runs = paths[layer]
        if not runs:
            continue
        paint = image.copy()
        for run in runs:
            points = np.rint(run).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(paint, [points], False, LAYER_COLORS_BGR[layer],
                          LAYER_WIDTH[layer], cv2.LINE_AA)
        image = cv2.addWeighted(paint, LAYER_ALPHA[layer], image,
                                1 - LAYER_ALPHA[layer], 0)
    if mask is not None:
        image[mask != 0] = frame[mask != 0]
    for x0, y0, x1, y1 in HUD_RECTS:
        image[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
    return image


def _bbox_for_frame(row: dict) -> list[int] | None:
    box = row.get("athlete_bbox")
    if box is None:
        return None
    if len(box) != 4 or not all(np.isfinite(box)):
        raise ValueError(f"Invalid athlete_bbox at PTS {row['pts']}")
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, x1 = max(0, x0), min(WIDTH, x1)
    y0, y1 = max(0, y0), min(HEIGHT, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Empty athlete_bbox at PTS {row['pts']}")
    return [x0, y0, x1, y1]


def _mask_for_frame(row: dict, run: Path) -> np.ndarray | None:
    mask_path = row.get("foreground_mask_path") or row.get("mask_path")
    if not mask_path:
        box = _bbox_for_frame(row)
        if box is None:
            return None
        mask = np.zeros((HEIGHT, WIDTH), np.uint8)
        x0, y0, x1, y1 = box
        mask[y0:y1, x0:x1] = 255
        return mask
    path = Path(mask_path)
    if not path.is_absolute():
        path = run / path
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != (HEIGHT, WIDTH):
        raise ValueError(f"Missing/invalid full-resolution foreground mask: {path}")
    return mask


def _load_poses(path: Path) -> dict:
    poses = json.loads(path.read_text(encoding="utf-8"))
    if poses.get("time_base") != "1/15360" or int(poses.get("anchor_pts", -1)) != ANCHOR_PTS:
        raise ValueError("Wrong time base or anchor PTS")
    frames = poses.get("frames", [])
    expected = list(range(FIRST_PTS, LAST_PTS + 1, PTS_STEP))
    if len(frames) != COUNT or [int(f["pts"]) for f in frames] != expected:
        raise ValueError("Expected all 121 native PTS from anchor minus to plus one second")
    if not poses.get("source_path"):
        raise ValueError("poses.json needs source_path for source provenance")
    for frame in frames:
        if frame.get("state") not in {"observed", "predicted", "lost"}:
            raise ValueError(f"Invalid frame state at PTS {frame['pts']}")
        if frame["state"] != "lost" and frame.get("camera_vector") is None:
            raise ValueError(f"Missing camera at PTS {frame['pts']}")
    return poses


def _chart_html(ready: bool) -> str:
    inner = ('<p id="endpoints"><img src="independent_endpoints.svg" '
             'alt="首末帧原片复核读数（非独立留出）" style="display:block;width:100%;max-width:1040px"></p>'
             if ready else '')
    return f'<!-- ENDPOINT_CHART_BEGIN -->{inner}<!-- ENDPOINT_CHART_END -->'


def _html(frames: list[dict], scene_hash: str, poses_hash: str,
          source_clip: str, chart_ready: bool, result: dict | None = None) -> str:
    payload = json.dumps({"frames": frames, "scene_sha256": scene_hash,
                          "poses_sha256": poses_hash}, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    return (PAGE.replace("__DATA__", payload)
            .replace("__SOURCE_CLIP__", quote(source_clip, safe="/.-_~"))
            .replace("__ACCEPTANCE__", html.escape((result or {}).get("message_zh",
                "候选尚待画面验收；图像支持不等于定位准确。")))
            .replace("__ENDPOINT_CHART__", _chart_html(chart_ready)))


def build_endpoint_chart(run: Path, scene_path: Path | None = None) -> Path | None:
    """Plot source endpoint readings; overlapping physical curves are not held out."""
    check_path = run / "checks.json"
    if not check_path.is_file():
        return None
    checks = json.loads(check_path.read_text(encoding="utf-8"))
    if checks.get("schema_version") != "rinkspace.shot_tracking.checks.v1":
        return None
    if not (run / "poses.json").is_file() or checks.get("poses_sha256") != _sha(run / "poses.json"):
        return None
    if scene_path is not None and checks.get("scene_sha256") != _sha(scene_path):
        return None
    groups = checks.get("endpoint_source_readings", checks.get("independent_image_readings", {})).get("groups", [])
    if not isinstance(groups, list) or not groups:
        return None
    rows = []
    for pts, label in ((FIRST_PTS, "前 1 秒"), (LAST_PTS, "后 1 秒")):
        matched = [g for g in groups if int(g.get("pts", -1)) == pts and
                   int(g.get("point_count", 0)) > 0]
        if not matched:
            rows.append((label, None, None, None, 0))
            continue
        count = sum(int(g["point_count"]) for g in matched)
        mean = sum(float(g["mean_distance_px"]) * int(g["point_count"])
                   for g in matched) / count
        maximum = max(float(g["max_distance_px"]) for g in matched)
        uncertainty = max(float(g["read_uncertainty_px"]) for g in matched)
        rows.append((label, mean, maximum, uncertainty, count))
    values = [v for _, mean, maximum, uncertainty, _ in rows
              for v in (mean, maximum, uncertainty) if v is not None]
    if not values or not all(math.isfinite(v) and v >= 0 for v in values):
        return None
    limit = max(5.0, math.ceil(max(values) / 5.0) * 5.0)
    left, width = 210.0, 760.0
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1040 280" role="img" '
        'aria-label="首末帧原片复核读数误差（非独立留出）">',
        '<rect width="1040" height="280" fill="#f3f8fb"/>',
        '<text x="30" y="33" font-size="18" font-weight="700" fill="#173548">首末帧原片复核读数</text>',
        '<text x="30" y="54" font-size="12" fill="#526c79">同一物理轮廓也参与拟合；这些读数不构成独立留出验证</text>',
    ]
    for i in range(5):
        x = left + width * i / 4
        parts.append(f'<path d="M{x:.1f},68 V231" stroke="#d3e0e7"/>')
        parts.append(f'<text x="{x:.1f}" y="251" text-anchor="middle" font-size="12" fill="#405e6d">{limit*i/4:g}</text>')
    parts.append('<text x="979" y="251" font-size="12" fill="#405e6d">px</text>')
    for row_number, (label, mean, maximum, uncertainty, count) in enumerate(rows):
        y = 117 + row_number * 92
        parts.append(f'<text x="30" y="{y+4}" font-size="16" font-weight="600" fill="#183b4e">{html.escape(label)}</text>')
        if mean is None:
            parts.append(f'<text x="{left}" y="{y+4}" font-size="14" fill="#a24a52">无有效复核读数</text>')
            continue
        xmean = left + width * mean / limit
        xmax = left + width * maximum / limit
        xuncertainty = left + width * uncertainty / limit
        parts.append(f'<path d="M{left},{y} H{xmean:.1f}" stroke="#268ca4" stroke-width="13" stroke-linecap="round"/>')
        parts.append(f'<path d="M{xuncertainty:.1f},{y-20} V{y+20}" stroke="#777b85" stroke-dasharray="3,3" stroke-width="2"/>')
        parts.append(f'<circle cx="{xmax:.1f}" cy="{y}" r="5.5" fill="#d5545e"/>')
        parts.append(f'<text x="{left}" y="{y+31}" font-size="12" fill="#405e6d">均值 {mean:.1f} · 最大 {maximum:.1f} · 人工读数范围 {uncertainty:.1f} px · {count} 点</text>')
    parts.extend(['<path d="M30,269 H1010" stroke="#d3e0e7"/>', '</svg>'])
    output = run / "independent_endpoints.svg"
    output.write_text("".join(parts), encoding="utf-8")
    return output


def render(run: Path, scene_path: Path, *, source_clip: Path | None = None,
           poses_path: Path | None = None, output_dir: Path | None = None) -> dict:
    run = run.resolve()
    scene_path = scene_path.resolve()
    poses_path = poses_path.resolve() if poses_path else run / "poses.json"
    output_dir = output_dir.resolve() if output_dir else run
    output_dir.mkdir(parents=True, exist_ok=True)
    clip = source_clip.resolve() if source_clip else run / "source.mp4"
    if not poses_path.is_file() or not clip.is_file() or not scene_path.is_file():
        raise FileNotFoundError("Need poses.json, source.mp4, and scene.json")
    poses = _load_poses(poses_path)
    source_index = run / "source.json"
    if source_index.is_file():
        indexed = json.loads(source_index.read_text(encoding="utf-8"))
        if (int(indexed.get("first_pts", -1)) != FIRST_PTS or
                int(indexed.get("last_pts", -1)) != LAST_PTS or
                int(indexed.get("anchor_pts", -1)) != ANCHOR_PTS or
                int(indexed.get("frame_count", -1)) != COUNT or
                int(indexed.get("width", -1)) != WIDTH or
                int(indexed.get("height", -1)) != HEIGHT):
            raise ValueError("source.json timing/resolution disagrees with exact PTS clip")
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    if poses.get("source_model_sha256") != _sha(scene_path):
        raise ValueError("Pose model hash differs from the scene being rendered")
    projector = Projector(world_paths(scene))
    with np.load(run / "observations.npz") as original_frames:
        expected_pts = np.arange(FIRST_PTS, LAST_PTS + PTS_STEP, PTS_STEP)
        if not np.array_equal(original_frames["pts"], expected_pts):
            raise ValueError("Original decoded observations do not match the exact 121 PTS")
        original_gray = original_frames["gray_small"]
        if original_gray.shape != (COUNT, HEIGHT // 2, WIDTH // 2):
            raise ValueError("Original decoded grayscale source has wrong shape")

    reader = cv2.VideoCapture(str(clip))
    if not reader.isOpened():
        raise RuntimeError(f"Cannot decode source clip: {clip}")
    clip_fps = reader.get(cv2.CAP_PROP_FPS)
    if abs(clip_fps - FPS) > .05:
        raise ValueError(f"Expected native 60 fps source clip, got {clip_fps}")
    if int(reader.get(cv2.CAP_PROP_FRAME_COUNT)) != COUNT:
        raise ValueError("Source clip does not contain exactly 121 frames")

    overlay_path = output_dir / "overlay.mp4"
    writer = imageio_ffmpeg.write_frames(
        str(overlay_path), (WIDTH, HEIGHT), fps=FPS, codec="libx264",
        quality=8, pix_fmt_in="bgr24", pix_fmt_out="yuv420p", macro_block_size=1,
        output_params=["-movflags", "+faststart"])
    writer.send(None)
    html_frames = []
    masks_used = 0
    silhouette_masks_used = 0
    visible_path_counts = []
    clip_alignment_mae = []
    contact_frames: list[np.ndarray] = []
    try:
        for index, row in enumerate(poses["frames"]):
            ok, frame = reader.read()
            if not ok or frame.shape[:2] != (HEIGHT, WIDTH):
                raise RuntimeError(f"Missing native-resolution source frame {index}")
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                              (WIDTH // 2, HEIGHT // 2), interpolation=cv2.INTER_AREA).astype(np.int16)
            own_mae = float(np.mean(np.abs(gray - original_gray[index].astype(np.int16))))
            neighbor_mae = min((float(np.mean(np.abs(gray - original_gray[j].astype(np.int16))))
                                for j in (index - 1, index + 1) if 0 <= j < COUNT),
                               default=float("inf"))
            if own_mae > 4.0 or own_mae > neighbor_mae + 1.0:
                raise ValueError(f"Clip frame {index} does not match original source PTS {row['pts']}: "
                                 f"own MAE {own_mae:.2f}, nearest adjacent MAE {neighbor_mae:.2f}")
            clip_alignment_mae.append(own_mae)
            state = row["state"]
            mask_name = row.get("foreground_mask_path") or row.get("mask_path")
            mask = _mask_for_frame(row, run)
            masks_used += int(mask is not None)
            silhouette_masks_used += int(bool(mask_name))
            if state == "lost":
                groups: dict[str, list[np.ndarray]] = {layer: [] for layer in LAYER_ORDER}
                vector_paths = {layer: "" for layer in LAYER_ORDER}
            else:
                groups, vector_paths = projector.project(row["camera_vector"])
            image = _draw(frame, groups, mask)
            if state != "observed":
                color = (0, 170, 255) if state == "predicted" else (35, 35, 230)
                label = "PREDICTED" if state == "predicted" else "LOST: NO GRID"
                cv2.rectangle(image, (8, 8), (300, 52), (20, 20, 20), -1)
                cv2.putText(image, label, (18, 39), cv2.FONT_HERSHEY_SIMPLEX,
                            .8, color, 2, cv2.LINE_AA)
            writer.send(image)
            if index in (0, 60, 120):
                thumb = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
                caption = np.zeros((34, 640, 3), dtype=np.uint8)
                cv2.putText(caption, f"PTS {row['pts']}  {state.upper()}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (230, 240, 240), 1, cv2.LINE_AA)
                contact_frames.append(np.vstack((thumb, caption)))
            visible_path_counts.append(sum(len(v) for v in groups.values()))
            mask_url = None
            if mask_name:
                mask_file = Path(mask_name)
                if not mask_file.is_absolute():
                    mask_file = run / mask_file
                # BGRA masks keep 0/255 in RGB for OpenCV and the same binary
                # alpha for browser erasure; one file serves both consumers.
                mask_rgba = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
                if mask_rgba is not None and mask_rgba.ndim == 3 and mask_rgba.shape[2] == 4:
                    mask_url = quote(os.path.relpath(mask_file, output_dir).replace("\\", "/"), safe="/.-_~")
            html_frames.append({"pts": int(row["pts"]), "state": state,
                                "mask": mask is not None,
                                "mask_url": mask_url,
                                "mask_bbox": _bbox_for_frame(row),
                                "paths": vector_paths})
            if (index + 1) % 20 == 0 or index + 1 == COUNT:
                print(f"rendered {index + 1}/{COUNT}", flush=True)
        more, _ = reader.read()
        if more:
            raise ValueError("Source clip contains extra decoded frames")
    finally:
        reader.release()
        writer.close()

    # Decode the delivered video itself: an encoder must not resize 1080 to
    # a macroblock-aligned 1088 or silently alter the native frame count.
    exported = cv2.VideoCapture(str(overlay_path))
    exported_fps = exported.get(cv2.CAP_PROP_FPS)
    exported_count = 0
    try:
        while True:
            ok, image = exported.read()
            if not ok:
                break
            if image.shape[:2] != (HEIGHT, WIDTH):
                raise ValueError("Encoded overlay changed the source resolution")
            exported_count += 1
    finally:
        exported.release()
    if exported_count != COUNT or abs(exported_fps - FPS) > .01:
        raise ValueError("Encoded overlay changed the frame count or frame rate")

    if len(contact_frames) != 3 or not cv2.imwrite(
            str(output_dir / "contact_sheet.jpg"), np.hstack(contact_frames),
            [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError("Could not save first/anchor/last contact sheet")

    scene_hash, poses_hash = _sha(scene_path), _sha(poses_path)
    result_path = output_dir / "review_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
    if result and (result.get("poses_sha256") != poses_hash or result.get("scene_sha256") != scene_hash):
        raise ValueError("Visual-review result belongs to different poses or scene")
    clip_relative = os.path.relpath(clip, output_dir).replace("\\", "/")
    chart = build_endpoint_chart(output_dir, scene_path)
    (output_dir / "review.html").write_text(_html(html_frames, scene_hash, poses_hash,
                                           clip_relative, chart is not None, result), encoding="utf-8")
    summary = {
        "schema": "rinkspace.shot_tracking.review.v1",
        "source_path": poses["source_path"],
        "source_clip": clip_relative,
        "source_clip_sha256": _sha(clip),
        "source_clip_exact_pts_checked_against_original_decode": True,
        "source_clip_alignment_gray_mae_max": round(max(clip_alignment_mae), 3),
        "scene_sha256": scene_hash,
        "poses_sha256": poses_hash,
        "frame_count": COUNT,
        "fps": FPS,
        "encoded_video_verified": {"size_wh": [WIDTH, HEIGHT],
                                   "decoded_frame_count": exported_count,
                                   "fps": exported_fps},
        "pts_first": FIRST_PTS,
        "pts_anchor": ANCHOR_PTS,
        "pts_last": LAST_PTS,
        "observed_count": sum(f["state"] == "observed" for f in poses["frames"]),
        "predicted_count": sum(f["state"] == "predicted" for f in poses["frames"]),
        "lost_count": sum(f["state"] == "lost" for f in poses["frames"]),
        "foreground_mask_count": silhouette_masks_used,
        "rough_athlete_bbox_count": masks_used - silhouette_masks_used,
        "foreground_note": ("provided silhouettes restore source pixels over foreground; "
                            "review.html can reveal hidden lines for error inspection"
                            if silhouette_masks_used == COUNT else
                            "some frames have only a rough athlete box or no mask; "
                            "grid on skater must not be read as ice-plane fit"),
        "visible_projected_path_count_min": min(visible_path_counts),
        "visible_projected_path_count_max": max(visible_path_counts),
        "world_curve_count": len(projector.paths),
        "world_curve_point_count": len(projector.world),
        "geometry_note": "all frames use one exported scene; grid and markings projected from real 3D paths",
        "acceptance_status": result.get("status") if result else "visual_review_required; rendering alone is not tracking accuracy",
    }
    (output_dir / "review_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(f"{output_dir / 'overlay.mp4'}\n{output_dir / 'review.html'}")
    return summary


def quick_render(camera_vector: object, source_clip: Path, frame_index: int,
                 scene_path: Path, output: Path) -> Path:
    """Render one trial camera against its actual source frame for solver debugging."""
    if not 0 <= frame_index < COUNT:
        raise ValueError("frame_index must lie within the two-second clip")
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    projector = Projector(world_paths(scene))
    reader = cv2.VideoCapture(str(source_clip))
    if not reader.isOpened():
        raise RuntimeError(f"Cannot decode source clip: {source_clip}")
    reader.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = reader.read()
    reader.release()
    if not ok or frame.shape[:2] != (HEIGHT, WIDTH):
        raise RuntimeError(f"Cannot read native frame {frame_index}")
    paths, _ = projector.project(camera_vector)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), _draw(frame, paths, None)):
        raise RuntimeError(f"Cannot save {output}")
    return output


def preview(run: Path, scene_path: Path, *, source_clip: Path | None = None) -> tuple[Path, Path]:
    """Render only first/anchor/last for iterative camera fitting; no video."""
    run = run.resolve()
    clip = source_clip or run / "source.mp4"
    poses = _load_poses(run / "poses.json")
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    projector = Projector(world_paths(scene))
    reader = cv2.VideoCapture(str(clip))
    if not reader.isOpened():
        raise RuntimeError(f"Cannot decode source clip: {clip}")
    masked, unmasked = [], []
    try:
        for index in (0, 60, 120):
            reader.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = reader.read()
            if not ok or frame.shape[:2] != (HEIGHT, WIDTH):
                raise RuntimeError(f"Missing native frame {index}")
            row = poses["frames"][index]
            groups = ({layer: [] for layer in LAYER_ORDER}
                      if row["state"] == "lost" else
                      projector.project(row["camera_vector"])[0])
            for target, mask in ((masked, _mask_for_frame(row, run)), (unmasked, None)):
                image = _draw(frame, groups, mask)
                thumb = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
                caption = np.zeros((34, 640, 3), dtype=np.uint8)
                cv2.putText(caption, f"PTS {row['pts']}  {row['state'].upper()}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (230, 240, 240), 1, cv2.LINE_AA)
                target.append(np.vstack((thumb, caption)))
    finally:
        reader.release()
    out_masked, out_unmasked = run / "contact_sheet.jpg", run / "contact_sheet_unmasked.jpg"
    if not cv2.imwrite(str(out_masked), np.hstack(masked), [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(f"Cannot write {out_masked}")
    if not cv2.imwrite(str(out_unmasked), np.hstack(unmasked), [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(f"Cannot write {out_unmasked}")
    return out_masked, out_unmasked


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>两秒镜头 · 冰面逐帧查看</title>
<style>
body{margin:0;background:#10202b;color:#edf6fa;font:15px system-ui,sans-serif}
main{max-width:1520px;margin:auto;padding:16px}h1{font-size:21px;margin:0 0 12px}
.tools{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:10px 0}
button,label{background:#1b3545;color:#fff;border:1px solid #3e6071;border-radius:6px;padding:7px 10px;cursor:pointer}
label{white-space:nowrap}label input{margin-right:6px}button:hover,label:hover{background:#294b5e}
.stage{position:relative;width:100%;aspect-ratio:16/9;background:#000}
video,canvas{position:absolute;inset:0;width:100%;height:100%}canvas{pointer-events:none}
#scrub{width:min(100%,700px)}small{color:#acc2ce}a{color:#8adff1}
.pill{padding:4px 8px;border-radius:4px;background:#255365}.warning{background:#8a4c24}.lost{background:#8c3139}
</style></head><body><main>
<h1>01:53:35–01:53:37 · 冰面逐帧查看</h1>
<p class="warning" style="padding:12px;border-radius:6px">__ACCEPTANCE__</p>
<p><small>两秒追踪试验。图像支持表示有匹配证据，不代表定位已通过验收。固定光心与共同变焦规律仍是待验证的局部相机模型。</small></p>
<div class="tools"><button id="prev">◀ 前一帧</button><button id="next">后一帧 ▶</button>
<input id="scrub" type="range" min="0" max="120" value="0" step="1">
<span id="time">—</span><span id="status" class="pill">—</span></div>
<div class="tools"><label><input id="overlay" type="checkbox" checked>拟合叠图</label>
<label><input id="mask" type="checkbox" checked>人物遮挡</label>
<label><input id="grid" type="checkbox" checked>等距网格</label>
<label><input id="markings" type="checkbox" checked>冰面标识</label>
<label><input id="contact" type="checkbox" checked>接冰轮廓</label>
<label><input id="board" type="checkbox" checked>挡板顶沿</label>
<label><input id="ads" type="checkbox" checked>广告边界</label></div>
<div class="stage"><video id="video" src="__SOURCE_CLIP__" controls preload="auto" playsinline></video>
<canvas id="canvas" width="1920" height="1080"></canvas></div>
<p><small>按左右方向键逐帧；关掉“拟合叠图”看原片。绿色是接冰轮廓，紫色是冰面标识。网格以场地模型单位等距绘制，米制尚未确认。<a href="overlay.mp4">下载 60 fps 叠图视频</a>。</small></p>
<p><small id="masknote"></small></p>
__ENDPOINT_CHART__
</main><script>
const data=__DATA__;
const video=document.getElementById('video'),canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
const colors={grid:'#2bd2c3',markings:'#e448cf',contact:'#64ef3d',board:'#3bb1fb',ads:'#ffae28'};
const widths={grid:1.25,markings:2,contact:2.3,board:2,ads:1.5};
const layers=Object.keys(colors);let shown=-1;const maskImages=new Map();
function maskImage(i){if(i<0||i>=data.frames.length)return null;const url=data.frames[i].mask_url;if(!url)return null;let im=maskImages.get(url);
if(!im){im=new Image();maskImages.set(url,im);im.onload=()=>{if(shown>=0&&data.frames[shown].mask_url===url)draw(true,shown)};im.src=url;
while(maskImages.size>5){const old=maskImages.keys().next().value;maskImages.get(old).onload=null;maskImages.delete(old);}}
else{maskImages.delete(url);maskImages.set(url,im);}return im;}
function index(){return Math.max(0,Math.min(120,Math.floor(video.currentTime*60+0.0001)));}
function stamp(pts){let sec=pts/15360,h=Math.floor(sec/3600),m=Math.floor(sec%3600/60),s=sec%60;return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${s.toFixed(3).padStart(6,'0')}`;}
function draw(force=false,displayedIndex=null){let i=displayedIndex===null?index():Math.max(0,Math.min(120,displayedIndex));if(i===shown&&!force)return;shown=i;const f=data.frames[i];
document.getElementById('scrub').value=i;document.getElementById('time').textContent=`${stamp(f.pts)} · PTS ${f.pts} · ${i+1}/121`;
const status=document.getElementById('status');status.textContent=f.state==='observed'?'有图像支持 · 候选':f.state==='predicted'?'仅预测':'定位失败';status.className='pill '+(f.state==='observed'?'':f.state==='predicted'?'warning':'lost');
document.getElementById('masknote').textContent=f.mask_url?'本帧使用人物轮廓掩膜，可关闭遮挡复核原图和叠线。':f.mask_bbox?'本帧人物包络是粗遮挡，也遮住了包络内少量冰面；取消人物遮挡可查看被遮住的线。':f.mask?'本帧提供了人物掩膜；视频中的遮挡可与原片对照。':'本帧没有可靠人物遮挡掩膜；人物上的叠线不能用于判断冰面贴合。';
ctx.clearRect(0,0,1920,1080);if(!document.getElementById('overlay').checked)return;
for(const layer of layers){if(!document.getElementById(layer).checked||!f.paths[layer])continue;ctx.beginPath();ctx.strokeStyle=colors[layer];ctx.lineWidth=widths[layer];ctx.globalAlpha=layer==='grid'?.70:.92;ctx.stroke(new Path2D(f.paths[layer]));}ctx.globalAlpha=1;
if(document.getElementById('mask').checked){const im=maskImage(i);for(const d of [-2,-1,1,2])maskImage(i+d);if(im&&im.complete&&im.naturalWidth){ctx.globalCompositeOperation='destination-out';ctx.drawImage(im,0,0,1920,1080);ctx.globalCompositeOperation='source-over';}else if(f.mask_bbox){const [x0,y0,x1,y1]=f.mask_bbox;ctx.clearRect(x0,y0,x1-x0,y1-y0);}}
for(const [x0,y0,x1,y1] of [[31,28,550,103],[79,105,400,251],[1730,948,1828,998]])ctx.clearRect(x0,y0,x1-x0,y1-y0);
}
function seek(i){video.pause();video.currentTime=(Math.max(0,Math.min(120,i))+.45)/60;}
document.getElementById('prev').onclick=()=>seek(index()-1);document.getElementById('next').onclick=()=>seek(index()+1);
document.getElementById('scrub').oninput=e=>seek(Number(e.target.value));
for(const id of ['overlay','mask',...layers])document.getElementById(id).onchange=()=>draw(true);
video.addEventListener('seeked',()=>draw(true));video.addEventListener('loadeddata',()=>draw(true));
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(e.key==='ArrowLeft'){e.preventDefault();seek(index()-1);}if(e.key==='ArrowRight'){e.preventDefault();seek(index()+1);}});
if(video.requestVideoFrameCallback){const tick=(_,meta)=>{draw(true,Math.round(meta.mediaTime*60));video.requestVideoFrameCallback(tick)};video.requestVideoFrameCallback(tick)}
else video.addEventListener('timeupdate',()=>draw());
draw(true);
</script></body></html>'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
                        default=ROOT / "calibration" / "shot_tracking" / "runs" / "low_end_pm1")
    parser.add_argument("--scene", type=Path,
                        default=ROOT / "calibration" / "deliverables" / "scene.json")
    parser.add_argument("--source-clip", type=Path)
    parser.add_argument("--poses", type=Path,
                        help="Candidate poses; source observations remain under --run")
    parser.add_argument("--output-dir", type=Path,
                        help="Candidate review directory; the source clip is linked without copying")
    parser.add_argument("--quick-camera-json", type=Path,
                        help="25-value JSON vector or object with camera_vector; renders one PNG")
    parser.add_argument("--quick-frame-index", type=int, default=60,
                        help="0..120, default anchor frame 60")
    parser.add_argument("--quick-output", type=Path)
    parser.add_argument("--preview-only", action="store_true",
                        help="Render first/anchor/last contact sheets; do not make 121-frame video")
    parser.add_argument("--chart-only", action="store_true",
                        help="Refresh independent endpoint chart after checks.json arrives")
    args = parser.parse_args()
    if args.chart_only:
        chart = build_endpoint_chart(args.run, args.scene)
        page = args.run / "review.html"
        if page.is_file():
            content = page.read_text(encoding="utf-8")
            content = re.sub(r"<!-- ENDPOINT_CHART_BEGIN -->.*?<!-- ENDPOINT_CHART_END -->",
                             _chart_html(chart is not None), content, flags=re.DOTALL)
            page.write_text(content, encoding="utf-8")
        print(chart or "No current independent endpoint readings")
    elif args.preview_only:
        print(*preview(args.run, args.scene, source_clip=args.source_clip), sep="\n")
    elif args.quick_camera_json:
        camera = json.loads(args.quick_camera_json.read_text(encoding="utf-8"))
        vector = camera.get("camera_vector") if isinstance(camera, dict) else camera
        clip = args.source_clip or args.run / "source.mp4"
        output = args.quick_output or args.run / "candidate.png"
        print(quick_render(vector, clip, args.quick_frame_index, args.scene, output))
    else:
        render(args.run, args.scene, source_clip=args.source_clip,
               poses_path=args.poses, output_dir=args.output_dir)
