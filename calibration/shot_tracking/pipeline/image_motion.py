"""Bounded raw-pixel image-motion diagnostic for the ±1 s PTZ experiment.

Each adjacent frame is fitted from independently seeded board/ice LK pairs.
The rink's anchor projection is transformed through a robust ICE homography
chain. This is diagnostic image motion, never a physical camera or calibration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1] / "runs/low_end_pm1"
WORK = ROOT / "calibration/work"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(WORK))
from camera import anchor_camera  # noqa: E402
from physical_projection import safe_project  # noqa: E402


def _project(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(-1, 1, 2),
                                    np.asarray(H, np.float64))[:, 0].astype(np.float64)


def _metrics(H: np.ndarray, xy0: np.ndarray, xy1: np.ndarray,
             selection: np.ndarray) -> dict:
    error = np.linalg.norm(_project(H, xy0) - xy1, axis=1)
    selected = error[selection]
    return {"all_median_px": float(np.median(error)),
            "all_p95_px": float(np.percentile(error, 95)),
            "selected_count": int(selection.sum()),
            "selected_median_px": float(np.median(selected)) if len(selected) else None,
            "selected_p95_px": float(np.percentile(selected, 95)) if len(selected) else None}


def _fit_one(xy0: np.ndarray, xy1: np.ndarray) -> tuple[np.ndarray | None, dict]:
    if len(xy0) < 10:
        return None, {"count": int(len(xy0)), "status": "insufficient_points"}
    H, inlier = cv2.findHomography(np.asarray(xy0, np.float32),
                                   np.asarray(xy1, np.float32),
                                   cv2.RANSAC, 3.0, maxIters=2500, confidence=.997)
    if H is None or inlier is None or int(inlier.sum()) < 8:
        return None, {"count": int(len(xy0)), "status": "ransac_failed"}
    H = H / H[2, 2]
    if not np.isfinite(H).all():
        return None, {"count": int(len(xy0)), "status": "nonfinite_homography"}
    selection = inlier[:, 0].astype(bool)
    scores = _metrics(H, xy0, xy1, selection)
    scores.update({"count": int(len(xy0)), "inlier_fraction": float(selection.mean()),
                   "status": "fit"})
    return H, scores


def _edge_fits(pairs: np.lib.npyio.NpzFile, src: int, dst: int) -> tuple[dict, dict]:
    index = ((pairs["from_index"] == src) & (pairs["to_index"] == dst))
    all_results = {}
    matrices = {}
    for label in ("board", "ice", "combined"):
        use = index if label == "combined" else index & (pairs["surface"] == label)
        first, second = pairs["xy_from"][use], pairs["xy_to"][use]
        H, scores = _fit_one(first, second)
        all_results[label] = scores
        if H is not None:
            matrices[label] = H
    if "ice" in matrices and "board" in matrices:
        use = index & (pairs["surface"] == "ice")
        samples = pairs["xy_from"][use]
        diff = np.linalg.norm(_project(matrices["ice"], samples) -
                              _project(matrices["board"], samples), axis=1)
        all_results["board_ice_motion_difference_on_ice_candidates"] = {
            "median_px": float(np.median(diff)),
            "p95_px": float(np.percentile(diff, 95)),
        }
    return matrices, all_results


def _direct_correction(H: np.ndarray, target: int, direct: np.lib.npyio.NpzFile,
                       anchor_xy: np.ndarray) -> tuple[np.ndarray, dict]:
    use = (direct["target_index"] == target) & (direct["surface"] == "ice")
    ids = direct["anchor_id"][use]
    xy = direct["xy_target"][use]
    if len(ids) < 20:
        return H, {"applied": False, "reason": "too_few_direct_ice_matches", "count": int(len(ids))}
    base = anchor_xy[ids]
    if np.ptp(base[:, 0]) < 230 or np.ptp(base[:, 1]) < 70:
        return H, {"applied": False, "reason": "direct_matches_too_local", "count": int(len(ids))}
    predicted = _project(H, base)
    affine, inlier = cv2.estimateAffinePartial2D(predicted.astype(np.float32),
                                                  xy.astype(np.float32),
                                                  method=cv2.RANSAC,
                                                  ransacReprojThreshold=3.0,
                                                  maxIters=2000, confidence=.997)
    if affine is None or inlier is None or int(inlier.sum()) < 15:
        return H, {"applied": False, "reason": "direct_correction_ransac_failed", "count": int(len(ids))}
    delta = np.eye(3)
    delta[:2] = affine
    corrected = delta @ H
    before = np.linalg.norm(predicted - xy, axis=1)
    after = np.linalg.norm(_project(corrected, base) - xy, axis=1)
    movement = np.linalg.norm(_project(corrected, base) - predicted, axis=1)
    allowed = (np.isfinite(corrected).all() and int(inlier.sum()) >= 20
               and float(np.median(after)) < 2.5 and float(np.percentile(after, 95)) < 7.0
               and float(np.median(movement)) < 30.0
               and float(np.median(after)) <= float(np.median(before)) + .05)
    scores = {"applied": bool(allowed), "count": int(len(ids)),
              "inliers": int(inlier.sum()),
              "before_median_px": float(np.median(before)),
              "after_median_px": float(np.median(after)),
              "after_p95_px": float(np.percentile(after, 95)),
              "correction_median_px": float(np.median(movement))}
    if not allowed:
        scores["reason"] = "insufficient_quality_or_unbounded_correction"
    return (corrected / corrected[2, 2] if allowed else H), scores


def _unique_grid_edges(scene: dict) -> list[np.ndarray]:
    edges = {}
    for cell in scene["ice_grid"]["cells"]:
        vertices = np.asarray(cell["polygon_xyz"], float)
        for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
            if np.linalg.norm(a-b) < 1.0:
                continue
            key = tuple(sorted((tuple(np.rint(a[:2] * 1000).astype(int)),
                                tuple(np.rint(b[:2] * 1000).astype(int)))))
            edges[key] = np.linspace(a, b, 8)
    return list(edges.values())


def _feature_world_paths(scene: dict) -> list[np.ndarray]:
    paths = []
    for item in scene["ice_features"]:
        if item["kind"] == "circle":
            for ring in item.get("rings", []):
                centre = np.asarray(ring["center_xyz"], float)
                radius = float(ring["radius"])
                angles = np.linspace(0, 2*np.pi, 145)
                xyz = np.column_stack([centre[0] + radius*np.cos(angles),
                                       centre[1] + radius*np.sin(angles),
                                       np.zeros(len(angles))])
                paths.append(xyz)
        elif item["kind"] == "polygon":
            vertices = np.asarray(item["polygon_xyz"], float)
            vertices = np.vstack([vertices, vertices[0]])
            for a, b in zip(vertices, vertices[1:]):
                paths.append(np.linspace(a, b, 24))
        elif item["kind"] == "polyline":
            for segment in item["segments_xyz"]:
                vertices = np.asarray(segment, float)
                for a, b in zip(vertices, vertices[1:]):
                    paths.append(np.linspace(a, b, 24))
    return paths


def _anchor_paths(scene: dict, cam: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    outputs = []
    for group in (_unique_grid_edges(scene), _feature_world_paths(scene)):
        projected = []
        for path in group:
            uv, _, valid = safe_project(cam, path)
            if valid.all() and np.isfinite(uv).all():
                projected.append(uv)
        outputs.append(projected)
    return outputs[0], outputs[1]


def _draw_paths(frame: np.ndarray, paths: list[np.ndarray], H: np.ndarray,
                color: tuple[int, int, int], thickness: int) -> None:
    for path in paths:
        moved = _project(H, path)
        if not np.isfinite(moved).all() or np.max(np.abs(moved)) > 100000:
            continue
        points = np.rint(moved).astype(np.int32)
        if np.any(np.linalg.norm(np.diff(moved, axis=0), axis=1) > 160):
            continue
        cv2.polylines(frame, [points], False, color, thickness, cv2.LINE_AA)


def _make_images(run_dir: Path, records: list[dict], scene: dict) -> None:
    anchor = anchor_camera(scene, 104693760)
    grid, features = _anchor_paths(scene, anchor)
    wanted = {0, 60, 120}
    with av.open(str(run_dir / "source.mp4")) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index not in wanted:
                continue
            bgr = frame.to_ndarray(format="bgr24")
            H = np.asarray(records[index]["H_image_from_anchor"], float)
            _draw_paths(bgr, grid, H, (230, 213, 44), 1)
            _draw_paths(bgr, features, H, (56, 50, 235), 2)
            label = f"PTS {records[index]['pts']} | raw-image motion diagnostic, not physical camera"
            cv2.rectangle(bgr, (0, 0), (min(1250, len(label)*17), 34), (18, 30, 38), -1)
            cv2.putText(bgr, label, (14, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        .65, (255, 255, 255), 1, cv2.LINE_AA)
            out = run_dir / f"image_motion_{records[index]['pts']}.png"
            if not cv2.imwrite(str(out), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise OSError(out)


def run(run_dir: Path = RUN) -> dict:
    with np.load(run_dir / "observations.npz") as obs, \
         np.load(run_dir / "pair_tracks.npz") as pairs, \
         np.load(run_dir / "anchor_direct.npz") as direct:
        pts = obs["pts"].astype(np.int64)
        if len(pts) != 121 or int(pts[60]) != 104693760:
            raise ValueError("Unexpected source PTS sequence")
        models = {}
        diagnostics = {}
        for src in range(121):
            for dst in (src - 1, src + 1):
                if 0 <= dst < 121:
                    fitted, scores = _edge_fits(pairs, src, dst)
                    models[(src, dst)] = fitted
                    diagnostics[(src, dst)] = scores
                    if "ice" not in fitted:
                        raise ValueError(f"No robust ice fit on {src}->{dst}")
        records = [None] * 121
        records[60] = {"index": 60, "pts": int(pts[60]),
                       "H_image_from_anchor": np.eye(3).tolist(),
                       "anchor_direct_correction": {"applied": False, "reason": "anchor_identity"}}
        for direction in (-1, 1):
            current = 60
            end = 0 if direction < 0 else 120
            while current != end:
                nxt = current + direction
                previous = np.asarray(records[current]["H_image_from_anchor"], float)
                H = models[(current, nxt)]["ice"] @ previous
                H /= H[2, 2]
                correction = {"applied": False, "reason": "not_a_direct_keyframe"}
                if nxt % 5 == 0:
                    H, correction = _direct_correction(H, nxt, direct, obs["anchor_xy"])
                records[nxt] = {"index": nxt, "pts": int(pts[nxt]),
                                "H_image_from_anchor": H.tolist(),
                                "anchor_direct_correction": correction}
                current = nxt
        scene = json.loads((ROOT / "calibration/deliverables/scene.json").read_text(encoding="utf-8"))
        output = {
            "model": "raw_pixel_image_motion_diagnostic_not_physical_camera",
            "coordinate_system": "native 1920x1080 source pixels; H maps distorted anchor image pixels to each distorted image",
            "source": "source.mp4; original PTS in observations.npz",
            "anchor_pts": int(pts[60]),
            "frames": records,
            "edges": [
                {"from_index": src, "to_index": dst,
                 "from_pts": int(pts[src]), "to_pts": int(pts[dst]),
                 "fitted_from": "independently seeded two-way adjacent LK, RANSAC-selected",
                 "diagnostic": diagnostics[(src, dst)]}
                for src in range(121) for dst in (src - 1, src + 1) if 0 <= dst < 121
            ],
            "limits": [
                "Raw distorted-pixel homographies do not encode camera XYZ, pan/tilt/zoom, Brown optics or 3-D board depth.",
                "Adjacent image-pair residuals test internal 2-D consistency, not alignment to rink geometry.",
                "The accumulated warp may drift; direct anchor corrections are applied only where enough original-patch evidence survives.",
                "Grid and markings in three PNGs start from the old fitted anchor projection, not new independent human labels.",
            ],
        }
        (run_dir / "image_motion.json").write_text(json.dumps(output, ensure_ascii=False,
                                                        indent=2), encoding="utf-8")
        _make_images(run_dir, records, scene)
        forward = [diagnostics[(i, i+1)] for i in range(120)]
        print(json.dumps({
            "ice_pair_selected_median_px": float(np.median([e["ice"]["selected_median_px"] for e in forward])),
            "ice_pair_selected_p95_of_p95_px": float(np.percentile([e["ice"]["selected_p95_px"] for e in forward], 95)),
            "board_pair_selected_median_px": float(np.median([e["board"]["selected_median_px"] for e in forward])),
            "direct_corrections": int(sum(f["anchor_direct_correction"]["applied"] for f in records)),
            "images": [f"image_motion_{pts[i]}.png" for i in (0, 60, 120)],
        }))
        return output


if __name__ == "__main__":
    run()
