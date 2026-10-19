"""Measure board edge pixels from real images, without any scene/camera projection.

ROIs only delimit the visually identified physical side and exclude occlusions.
Every reported coordinate comes from chroma profiles and local gradients.  The
colour half-height operation reuses the approach of the v0.13 survey extractor;
this version avoids its hand-written pixel guide, records subpixel refinement,
uncertainty evidence and robust-line diagnostics, and exposes callable functions.
The complete board objects are reserved for checking, never camera fitting.
"""
from __future__ import annotations
from rinkspace.storage import resolve_asset as _resolve_asset

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = _resolve_asset(ROOT, ROOT / "outputs/calibration/ankara_2026_v014_main/board_evidence")
SOURCE = _resolve_asset(ROOT, ROOT / "outputs/rink_model/ankara_2026_shared/v0.13-women-sp/evidence/competition")
SCENE = _resolve_asset(ROOT, ROOT / "outputs/rink_model/ankara_2026_shared/v0.14-scene3d/scene.json")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _crossing(p, peak, threshold, direction):
    j = peak
    while 0 <= j + direction < len(p) and p[j + direction] > threshold:
        j += direction
    k = j + direction
    if not 0 <= k < len(p) or p[j] == p[k]:
        return None
    return float(j + (threshold - p[j]) / (p[k] - p[j]) * direction)


def measure_edge(image, *, roi, edge, exclude_rectangles=(), step=8):
    """Return independent chromatic edge samples in a supplied [x0,y0,x1,y1] ROI.

    edge='blue_rail_upper' means the upper (background-to-blue) colour edge.
    edge='yellow_kickplate_lower' means the yellow-to-ice colour transition;
    it is only a proxy for the physically sharp ice/board contact in blurred video.
    ROI boundaries and x sample limits are not object endpoints.
    """
    x0, y0, x1, y1 = map(int, roi)
    h, w = image.shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(f"ROI outside image: {roi} / {(w, h)}")
    b, g, r = cv2.split(image.astype(np.float32))
    if edge == "blue_rail_upper":
        score, direction, systematic_floor = b - (r + g) / 2, -1, 1.5
    elif edge == "yellow_kickplate_lower":
        score, direction, systematic_floor = (r + g) / 2 - b, 1, 4.0
    else:
        raise ValueError(edge)
    score = cv2.GaussianBlur(score, (5, 5), 0.8)
    rows = []
    for x in range(x0 + 3, x1 - 3, step):
        if any(a <= x <= c and not (d < y0 or b0 > y1)
               for a, b0, c, d in exclude_rectangles):
            continue
        p = np.median(score[y0:y1, x - 2:x + 3], axis=1)
        baseline = float(np.percentile(p, 20))
        peak = int(np.argmax(p))
        prominence = float(p[peak] - baseline)
        if prominence < 5 or not 3 < peak < len(p) - 4:
            continue
        half = _crossing(p, peak, baseline + prominence * .5, direction)
        lo = _crossing(p, peak, baseline + prominence * .2, direction)
        hi = _crossing(p, peak, baseline + prominence * .8, direction)
        if any(v is None for v in (half, lo, hi)):
            continue
        # Local refinement: signed chroma derivative peak within the measured
        # 20%-80% transition, expanded by one pixel; never predicted geometry.
        grad = np.gradient(p) * -direction
        a = max(1, int(np.floor(min(lo, hi))) - 1)
        z = min(len(p) - 2, int(np.ceil(max(lo, hi))) + 1)
        k = a + int(np.argmax(grad[a:z + 1]))
        denom = float(grad[k - 1] - 2 * grad[k] + grad[k + 1])
        correction = (0.5 * float(grad[k - 1] - grad[k + 1]) / denom
                      if abs(denom) > 1e-8 else 0.0)
        correction = float(np.clip(correction, -.5, .5))
        y = float(y0 + k + correction)
        spread = float(abs(lo - hi))
        rows.append({
            "xy_px": [float(x), y],
            "half_chroma_xy_px": [float(x), float(y0 + half)],
            "peak_chroma_above_baseline": prominence,
            "signed_gradient_peak": float(grad[k]),
            "edge_20_80_width_px": spread,
            "uncertainty_px": max(systematic_floor, spread / 2,
                                  abs(y0 + half - y)),
        })
    if len(rows) < 8:
        raise ValueError(f"Insufficient independent visible columns: {len(rows)}")
    xy = np.array([r["xy_px"] for r in rows])
    vx, vy, ox, oy = cv2.fitLine(xy.astype(np.float32), cv2.DIST_HUBER, 0, .001, .001).ravel()
    line = np.array([-vy, vx, vy * ox - vx * oy], dtype=float)
    residual = xy @ line[:2] + line[2]
    mad_sigma = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    threshold = max(2.5, 3 * mad_sigma)
    keep = np.abs(residual) <= threshold
    if keep.sum() < 8:
        raise ValueError("Robust line retains too few visible samples")
    vx, vy, ox, oy = cv2.fitLine(xy[keep].astype(np.float32), cv2.DIST_HUBER, 0, .001, .001).ravel()
    line = np.array([-vy, vx, vy * ox - vx * oy], dtype=float)
    residual = xy @ line[:2] + line[2]
    for row, ok, dist in zip(rows, keep, residual):
        row["accepted_for_check"] = bool(ok)
        row["signed_distance_to_measured_line_px"] = float(dist)
    accepted = xy[keep]
    return {
        "edge_type": edge, "roi_xyxy_px": list(roi),
        "excluded_rectangles_xyxy_px": [list(x) for x in exclude_rectangles],
        "sample_interval_px": step,
        "pixel_points": accepted.tolist(), "all_measurements": rows,
        "measured_line_abc": line.tolist(),
        "sample_extent_is_object_endpoint": False,
        "diagnostic_measured_line_fit": {
            "n_candidates": len(rows), "n_kept": int(keep.sum()),
            "median_abs_distance_px": float(np.median(abs(residual[keep]))),
            "p95_abs_distance_px": float(np.percentile(abs(residual[keep]), 95)),
            "mad_sigma_before_rejection_px": mad_sigma,
            "rejection_threshold_px": threshold,
            "note": "Image-only straight-edge diagnostic; not a calibration error or accuracy claim.",
        },
        "clarity": {
            "median_chroma_contrast": float(np.median([r["peak_chroma_above_baseline"] for r in rows])),
            "median_edge_20_80_width_px": float(np.median([r["edge_20_80_width_px"] for r in rows])),
            "median_uncertainty_px": float(np.median([r["uncertainty_px"] for r in rows])),
            "uncertainty_basis": "Largest of half the measured 20%-80% chroma transition width, gradient-versus-half-chroma displacement, and physical-interface floor; heuristic bound, not a statistical confidence interval.",
            "physical_interface_uncertainty_floor_px": systematic_floor,
        },
    }


def extract_frame(image_path, specs, *, frame_metadata=None, output_dir=DEFAULT_OUT):
    """Measure specified physical board portions and save pixels and review images."""
    image_path = Path(image_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    im = cv2.imread(str(image_path))
    if im is None:
        raise FileNotFoundError(image_path)
    stem = (frame_metadata or {}).get("frame_id", image_path.stem)
    result = {
        "schema_version": "rinkspace.scene3d_real_board_holdout.v1",
        "image_path": str(image_path), "image_sha256": _sha256(image_path),
        "image_size_wh": [im.shape[1], im.shape[0]],
        "scene_path": str(SCENE), "scene_sha256": _sha256(SCENE),
        "frame_metadata": frame_metadata or {},
        "coordinate_convention": "full decoded image, x right, y down, pixel-centre coordinates; no crop/rescale applied",
        "split": "whole_board_objects_holdout_no_fit",
        "method": "Image-only column chroma maxima, measured half-height crossing, local signed-gradient subpixel refinement, Huber line rejection; ROIs delimit identified straight segments and exclude skater/corners.",
        "method_prior": "v0.13 evidence/survey/extract_boundary.py colour profile principle; its manual guide coordinates are not reused",
        "objects": [],
    }
    overlay = im.copy()
    for spec in specs:
        measure = measure_edge(im, roi=spec["roi"], edge=spec["edge"],
                               exclude_rectangles=spec.get("exclude_rectangles", []))
        measure.update({k: v for k, v in spec.items() if k not in ("roi", "edge", "exclude_rectangles")})
        measure["split"] = "holdout"
        result["objects"].append(measure)
        color = (255, 255, 0) if spec["edge"] == "blue_rail_upper" else (255, 0, 255)
        for point in measure["pixel_points"]:
            cv2.circle(overlay, tuple(np.rint(point).astype(int)), 2, color, -1)
        a, b, c, d = spec["roi"]
        cv2.rectangle(overlay, (a, b), (c, d), color, 1)
        cv2.putText(overlay, spec["object_id"], (a + 4, b - 8), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    overlay_path = output_dir / f"{stem}_measured_edges.png"
    cv2.imwrite(str(overlay_path), overlay)
    # Side-by-side original and pixel overlay at native resolution around boards.
    min_y = max(0, min(s["roi"][1] for s in specs) - 30)
    max_y = min(im.shape[0], max(s["roi"][3] for s in specs) + 30)
    detail_path = output_dir / f"{stem}_edge_detail.png"
    cv2.imwrite(str(detail_path), np.vstack([im[min_y:max_y], overlay[min_y:max_y]]))
    result["visuals"] = {"overlay": str(overlay_path.resolve()), "detail": str(detail_path.resolve()),
                          "detail_crop_y_range": [min_y, max_y],
                          "detail_layout": "original above / measured pixels below; native scale"}
    json_path = output_dir / f"{stem}_board_observations.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def default_spec(side, top_roi, bottom_roi, exclusions):
    endpoint_xy = ([[-79.167, 50], [79.167, 50]] if side == "upper_long_side"
                   else [[-100, -29.167], [-100, 29.167]])
    identity = ("Visual board-ad order: advert_ppt_object_115 (Daikichi), 116 (ISU website); x extent stops before upper-right curve."
                if side == "upper_long_side" else
                "Visual left-end board-ad order: advert_ppt_object_114 (ISU APP), 128 (red emblem), 127 (ANKARA shield); left curve and skater excluded.")
    return [{
        "object_id": f"boards_{side}_{suffix}", "edge": edge, "roi": roi,
        "exclude_rectangles": exclusions,
        "identity_status": "visually_associated_with_user_PPT_ad_identity_and_side",
        "identity_evidence": identity,
        "scene_geometry_status": "v0.14_schematic_geometry_condition_not_independent_measurement",
        "scene_object_path": "scene.json:board_surface/ice_boundary",
        "scene_feature_id": "boards_inner_surface" if suffix == "top" else "ice_board_contact",
        "qualitative_side": side,
        "world_line_xyz": [[*p, height] for p in endpoint_xy],
        "world_line_use": "Check point-to-projected-line distance only; observed crop limits have no physical endpoint correspondence.",
        "physical_interface": interface,
        "physical_geometry_caveat": caveat,
    } for suffix, edge, roi, height, interface, caveat in [
        ("top", "blue_rail_upper", top_roi, 4.0,
         "Upper colour edge of blue top cap, background-to-blue transition; not an advertisement edge.",
         "Scene board_height=4 is an unmeasured demonstration value. Blue cap has width/thickness; its upper silhouette may sit on the outer rather than inner board edge; this interface approximation is unverified."),
        ("bottom", "yellow_kickplate_lower", bottom_roi, 0.0,
         "Lower colour transition of the yellow kickplate into ice, locally refined from the actual image.",
         "Blur/reflection make this a contact-line proxy with an explicit >=4 px physical-interface uncertainty; reflection below is excluded from the intended object, but may bias the fading colour transition."),
    ]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_OUT / "measurement_config.json",
                        help="JSON list: image_path/specs/frame_metadata")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    tasks = json.loads(args.config.read_text(encoding="utf-8"))
    report = []
    for task in tasks:
        result = extract_frame(**task, output_dir=args.output_dir)
        report.append({"frame": result["frame_metadata"],
                       "objects": [{"id": o["object_id"], "clarity": o["clarity"],
                                    "line": o["measured_line_abc"],
                                    "diagnostic": o["diagnostic_measured_line_fit"]} for o in result["objects"]]})
    (args.output_dir / "measurement_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
