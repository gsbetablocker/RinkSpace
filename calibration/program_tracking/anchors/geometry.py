"""Traceable physical-curve observations for v0.25 camera references.

The old joint-279 video measurements are retained as image observations only.
Their camera estimates are initial guesses; all world curves come from the
currently released three-frame scene. Old ad-detail coordinates are excluded:
the current scene does not define those exact printed-detail positions.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from rinkspace.storage import resolve_asset

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SCENE = ROOT / "calibration/deliverables/scene.json"
OLD_SCENE = resolve_asset(ROOT, "outputs/rink_model/ankara_2026_shared/v0.16-joint-279/scene.json")
OBSERVATIONS = {
    "main_long_side": resolve_asset(ROOT, "outputs/calibration/ankara_2026_v015_joint/joint_279/main/observations_direct.json"),
    "secondary_long_side": resolve_asset(ROOT, "outputs/calibration/ankara_2026_v015_joint/joint_279/secondary/observations.json"),
}
OLD_CAMERA_ID = {
    "main_long_side": "camera_main_long_side",
    "secondary_long_side": "camera_secondary_long_side",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _circle(center: object, radius: float, n: int = 240) -> np.ndarray:
    c = np.asarray(center, float)
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack((c[0] + radius * np.cos(t),
                            c[1] + radius * np.sin(t),
                            np.full(n, c[2])))


def world_curve(scene: dict, target: dict) -> np.ndarray | None:
    """Map an explicitly identified old measured curve to the new scene.

    No named printed detail is mapped because v0.25 lacks its world position.
    A circle-outline observation names the outer ring in the old fit schema.
    """
    kind = target.get("kind")
    if kind == "board_side":
        side, edge = target.get("side"), target.get("edge")
        if side != "upper_long_side" or edge not in ("top", "contact"):
            return None
        key = "top_edge_xyz" if edge == "top" else "bottom_edge_xyz"
        boundary = np.asarray(scene["board_surface"][key], float)
        # The upper straight is the segment from the end of the right corner
        # to the beginning of the left corner; keep the curved ends separate.
        highest = np.max(boundary[:, 1])
        keep = np.abs(boundary[:, 1] - highest) < 1e-4
        path = boundary[keep]
        if len(path) < 2:
            raise ValueError("No straight upper-board curve in current scene")
        return path[np.argsort(path[:, 0])]
    feature_id = target.get("feature_id")
    feature = next((f for f in scene["ice_features"] if f["id"] == feature_id), None)
    if feature is None:
        return None
    if kind == "circle_outline" and feature["kind"] == "circle":
        ring = next((r for r in feature["rings"] if r["role"] == "outer"), None)
        return _circle(ring["center_xyz"], ring["radius"]) if ring else None
    if kind == "line_segment" and feature["kind"] == "polyline":
        segments = feature["segments_xyz"]
        return np.asarray(segments[0], float) if len(segments) == 1 else None
    return None


def old_camera_vector(record: dict) -> np.ndarray:
    camera = record["camera"]
    k = np.asarray(camera["K_px"], float)
    distortion = np.asarray(camera["distortion_opencv"], float)
    if distortion.size > 14:
        raise ValueError("Unsupported old lens coefficient count")
    result = np.zeros(25, float)
    result[:3] = camera["center_xyz"]
    result[3:6] = camera["rvec_world_to_camera"]
    result[6:8] = np.log([k[0, 0], k[1, 1]])
    result[8:11] = [k[0, 2], k[1, 2], k[0, 1]]
    result[11:11 + len(distortion)] = distortion
    result[23:25] = 0.0
    return result


def project(camera: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """OpenCV Brown14 projection in native 1920x1080 pixel coordinates."""
    v = np.asarray(camera, float)
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    if v.shape != (25,) or np.any(np.abs(v[23:25]) > 1e-12):
        raise ValueError("Expected 25-value camera vector with zero sensor tilt")
    r = cv2.Rodrigues(v[3:6])[0]
    k = np.array([[math.exp(v[6]), 0, v[8]],
                  [0, math.exp(v[7]), v[9]], [0, 0, 1]], float)
    uv = cv2.projectPoints(xyz, v[3:6], -r @ v[:3], k, v[11:25])[0].reshape(-1, 2)
    uv[:, 0] += v[10] * (uv[:, 1] - v[9]) / math.exp(v[7])
    return uv


def reference_observations(camera_id: str, pts: int) -> dict:
    """Return measured image curves with only verified physical-ID mapping.

    The mapping describes identity, not accuracy. Ad-detail pixel points are
    deliberately omitted because their 3-D positions changed with the model.
    """
    if camera_id not in OBSERVATIONS:
        raise ValueError(camera_id)
    scene = read_json(SCENE)
    old = read_json(OLD_SCENE)
    raw = read_json(OBSERVATIONS[camera_id])
    frame = next((f for f in raw["frames"] if f["source_pts"] == pts), None)
    if frame is None:
        raise KeyError(f"No measured reference: {camera_id} PTS {pts}")
    old_frame = next((f for f in old["frame_cameras"]
                      if f["source_pts"] == pts and f["camera_id"] == OLD_CAMERA_ID[camera_id]), None)
    if old_frame is None:
        raise KeyError(f"No old pose initializer: {camera_id} PTS {pts}")
    curves = []
    for row in raw["observations"]:
        if row["frame_id"] != frame["id"] or row.get("synthetic_or_model_projected"):
            continue
        curve = world_curve(scene, row["target"])
        if curve is None:
            continue
        measured = np.asarray(row["pixels_uv"], float).reshape(-1, 2)
        if not np.isfinite(measured).all() or len(measured) < 3:
            continue
        curves.append({"id": row["id"], "target": row["target"],
                       "physical_group_id": row.get("physical_group_id"),
                       "observed_uv": measured,
                       "world_curve_xyz": curve,
                       "sigma_px": float(row["sigma_px"]),
                       "measurement_origin": row.get("measurement_origin")})
    return {"camera_id": camera_id, "source_pts": pts,
            "source_image": str(frame["image_path"]),
            "source_observation_file": str(OBSERVATIONS[camera_id]),
            "scene_path": str(SCENE),
            "old_pose_initial_only": old_camera_vector(old_frame),
            "curves": curves}


def available_references(camera_id: str) -> list[dict]:
    """List old source-backed camera records that have usable v0.25 curves."""
    raw = read_json(OBSERVATIONS[camera_id])
    scene = read_json(SCENE)
    counts = {}
    for item in raw["observations"]:
        if item.get("synthetic_or_model_projected") or world_curve(scene, item["target"]) is None:
            continue
        c = counts.setdefault(item["frame_id"], {"curve_count": 0, "pixel_count": 0, "targets": set()})
        c["curve_count"] += 1
        c["pixel_count"] += len(item["pixels_uv"])
        c["targets"].add(str(item["target"]))
    result = []
    for f in raw["frames"]:
        c = counts.get(f["id"])
        if c:
            result.append({"source_pts": f["source_pts"], "source_image": f["image_path"],
                           "curve_count": c["curve_count"], "pixel_count": c["pixel_count"],
                           "target_count": len(c["targets"])})
    return sorted(result, key=lambda x: (-x["target_count"], -x["pixel_count"]))
