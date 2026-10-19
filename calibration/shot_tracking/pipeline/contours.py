"""Read actual rink markings in source pixels near model-guided search paths.

The old anchor scene and a raw-image motion chain only define bounded search
windows. Accepted observations are extrema in original decoded source pixels;
ambiguous, occluded, or weak responses are omitted. No old agent readings are
used. Board and ice signals have different photometric meanings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1] / "runs/low_end_pm1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "calibration/work"))
from camera import anchor_camera, from_state, fast_project  # noqa: E402
from physical_projection import safe_project  # noqa: E402
from observations import _BOARD_Y_KEYS  # noqa: E402


def _move(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(uv, np.float32).reshape(-1, 1, 2),
                                    np.asarray(H, np.float64))[:, 0].astype(np.float64)


def _anchor_geometry(scene: dict) -> dict:
    cam = anchor_camera(scene, 104693760)
    geometry = {}
    top = np.asarray(scene["board_surface"]["top_edge_xyz"], float)
    contact = np.asarray(scene["board_surface"]["bottom_edge_xyz"], float)
    top_uv, _, top_ok = safe_project(cam, top)
    contact_uv, _, contact_ok = safe_project(cam, contact)
    indices = np.arange(0, len(top), 4)
    good = top_ok[indices] & contact_ok[indices]
    good &= np.isfinite(top_uv[indices]).all(axis=1) & np.isfinite(contact_uv[indices]).all(axis=1)
    good &= np.max(np.abs(top_uv[indices]), axis=1) < 10000
    good &= np.max(np.abs(contact_uv[indices]), axis=1) < 10000
    indices = indices[good]
    # Retain physical pieces outside the anchor sensor: during the pan they
    # enter the 2-second clip. Per-frame visibility/board-band gates below
    # discard other perimeter arcs rather than dropping them at anchor time.
    geometry["board_top"] = {"anchor_uv": top_uv[indices],
                             "paired_uv": contact_uv[indices],
                             "world_xyz": top[indices]}
    geometry["board_contact"] = {"anchor_uv": contact_uv[indices],
                                 "paired_uv": top_uv[indices],
                                 "world_xyz": contact[indices]}
    for name in ("F3", "F4"):
        item = next(x for x in scene["ice_features"] if x["id"] == name)
        ring = next(x for x in item["rings"] if x["role"] == "outer")
        centre = np.asarray(ring["center_xyz"], float)
        radius = float(ring["radius"])
        angle = np.linspace(0, 2*np.pi, 56, endpoint=False)
        world = np.column_stack((centre[0] + radius*np.cos(angle),
                                 centre[1] + radius*np.sin(angle),
                                 np.zeros(len(angle))))
        uv, _, valid = safe_project(cam, world)
        centre_uv, _, centre_valid = safe_project(cam, centre[None])
        if not bool(centre_valid[0]):
            raise ValueError(f"Anchor ring center not physically projectable: {name}")
        geometry[f"{name}_outer"] = {"anchor_uv": uv[valid],
                                      "world_xyz": world[valid],
                                      "center_world_xyz": centre,
                                      "center_anchor_uv": centre_uv[0]}
    line = next(x for x in scene["ice_features"] if x["id"] == "goal_line_left")
    ends = np.asarray(line["segments_xyz"][0], float)
    world = np.linspace(ends[0], ends[-1], 130)
    uv, _, valid = safe_project(cam, world)
    valid &= np.isfinite(uv).all(axis=1)
    valid &= (uv[:, 0] > -30) & (uv[:, 0] < 1950)
    valid &= (uv[:, 1] > 420) & (uv[:, 1] < 1100)
    geometry["goal_line_left"] = {"anchor_uv": uv[valid][::2],
                                  "world_xyz": world[valid][::2]}
    return geometry


def _normals(identity: str, predicted: np.ndarray, geom: dict, H: np.ndarray,
             center_uv: np.ndarray | None = None) -> np.ndarray:
    if identity.startswith("board"):
        pair = _move(H, geom["paired_uv"])
        normal = pair - predicted if identity == "board_top" else predicted - pair
    else:
        # Contour tangent controls geometric normal; radial sign selects the
        # F3/F4 outer edge's inside-dark to outside-light direction.
        tangent = np.roll(predicted, -1, axis=0) - np.roll(predicted, 1, axis=0)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        if identity.endswith("_outer"):
            centre = (_move(H, geom["center_anchor_uv"][None])[0]
                      if center_uv is None else center_uv)
            radial = predicted - centre
            flip = np.sum(normal * radial, axis=1) < 0
            normal[flip] *= -1
    length = np.linalg.norm(normal, axis=1)
    return np.divide(normal, length[:, None],
                     out=np.full_like(normal, np.nan), where=length[:, None] > 1e-6)


def _signals(bgr: np.ndarray) -> dict[str, np.ndarray]:
    smooth = cv2.GaussianBlur(bgr.astype(np.float32), (5, 5), .9)
    blue = smooth[:, :, 0] - .5*(smooth[:, :, 1] + smooth[:, :, 2])
    yellow = .5*(smooth[:, :, 1] + smooth[:, :, 2]) - smooth[:, :, 0]
    red = smooth[:, :, 2] - .5*(smooth[:, :, 0] + smooth[:, :, 1])
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (5, 5), .9)
    return {"board_top": blue, "board_contact": yellow,
            "F3_outer": gray, "F4_outer": gray, "goal_line_left": red}


def _inside_body(points: np.ndarray, bbox: np.ndarray, margin: float = 12.0) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return ((points[:, 0] >= x0-margin) & (points[:, 0] <= x1+margin)
            & (points[:, 1] >= y0-margin) & (points[:, 1] <= y1+margin))


def _profile_read(identity: str, predicted: np.ndarray, normals: np.ndarray,
                  signal: np.ndarray, bbox: np.ndarray,
                  minimum_ice_y: float) -> tuple[np.ndarray, list[dict]]:
    if len(predicted) < 4:
        return np.empty((0, 2)), []
    radius = 70 if identity in ("F3_outer", "F4_outer", "goal_line_left") else 60
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    xs = predicted[:, 0, None] + normals[:, 0, None]*offsets[None]
    ys = predicted[:, 1, None] + normals[:, 1, None]*offsets[None]
    in_frame = (np.isfinite(xs) & np.isfinite(ys)
                & (xs >= 7) & (xs < 1913) & (ys >= 7) & (ys < 1073))
    not_graphic = ~(((xs < 570) & (ys < 260)) | ((xs > 1660) & (ys > 910)))
    x0, y0, x1, y1 = bbox
    not_body = ~((xs > x0-15) & (xs < x1+15) & (ys > y0-15) & (ys < y1+15))
    usable = in_frame & not_graphic & not_body
    if not identity.startswith("board"):
        # This source-informed visibility gate stops a drifting search window
        # from declaring advertisement letters to be a faint ice circle.
        usable &= ys >= minimum_ice_y
    sampled = cv2.remap(signal, xs.astype(np.float32), ys.astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=-10000).astype(np.float64)
    if identity == "board_top":
        response = (np.roll(sampled, -2, axis=1)-np.roll(sampled, 2, axis=1))/4
        minimum, sign = 2.8, "blue_rise_toward_ice"
    elif identity == "board_contact":
        response = (np.roll(sampled, 2, axis=1)-np.roll(sampled, -2, axis=1))/4
        minimum, sign = 1.15, "yellow_fall_toward_ice"
        before = np.roll(sampled, 5, axis=1)
        after = np.roll(sampled, -5, axis=1)
        # Blue top rail also creates a strong negative yellow derivative. A
        # contact candidate must have actual yellow material immediately
        # above it and less yellow on the ice side.
        usable &= (before >= 2.0) & ((before-after) >= 4.0)
    elif identity in ("F3_outer", "F4_outer"):
        response = (np.roll(sampled, -3, axis=1)-np.roll(sampled, 3, axis=1))/6
        minimum, sign = 1.05, "gray_positive_outward"
    else:
        response = sampled - np.median(np.where(usable, sampled, np.nan), axis=1)[:, None]
        minimum, sign = 4.0, "red_ridge"
    usable[:, :4] = False
    usable[:, -4:] = False
    response[~usable] = -np.inf
    # Board lines and the red L are long coherent curves. Use the robust
    # across-curve peak as a search center, then refine locally. This prevents
    # an advertisement's blue logo or another red marking stealing one sample.
    if identity in ("board_top", "goal_line_left"):
        finite = np.where(np.isfinite(response), np.maximum(response, 0), np.nan)
        with np.errstate(all="ignore"):
            candidate_rows = np.isfinite(finite).sum(axis=1) >= 10
            if candidate_rows.any():
                subset = finite[candidate_rows]
                consensus = np.asarray([
                    float(np.median(column[np.isfinite(column)])) if np.isfinite(column).any() else np.nan
                    for column in subset.T
                ])
            else:
                consensus = np.full(len(offsets), np.nan)
        if np.isfinite(consensus).any():
            global_offset = float(offsets[int(np.nanargmax(consensus - .018*np.abs(offsets)))])
            usable &= np.abs(offsets[None] - global_offset) <= (16 if identity.startswith("board") else 22)
            response[~usable] = -np.inf
    # A weak position preference prevents a distant unrelated edge from
    # winning when responses are close; an actual positive signal is mandatory.
    preference = response - .020*np.abs(offsets[None])
    winner = np.argmax(preference, axis=1)
    chosen = response[np.arange(len(predicted)), winner]
    point = predicted + normals*offsets[winner, None]
    accepted = np.isfinite(chosen) & (chosen >= minimum)
    if identity in ("F3_outer", "F4_outer"):
        # At the actual outer edge, outside ice must be brighter than the
        # darker ring immediately inside. This rejects the inner edge sign.
        outer = np.clip(winner+8, 0, len(offsets)-1)
        inner = np.clip(winner-8, 0, len(offsets)-1)
        contrast = sampled[np.arange(len(predicted)), outer] - sampled[np.arange(len(predicted)), inner]
        accepted &= contrast >= 4.0
    elif identity == "goal_line_left":
        contrast = chosen
        accepted &= sampled[np.arange(len(predicted)), winner] >= 1.0
    else:
        contrast = chosen
    accepted &= ~_inside_body(point, bbox)
    # Isolated spikes in weak ice texture are not a usable contour segment.
    good = accepted.copy()
    for i in np.flatnonzero(accepted):
        nearby = [j for j in range(max(0, i-2), min(len(predicted), i+3))
                  if j != i and accepted[j]]
        if identity.endswith("_outer"):
            nearby += [j % len(predicted) for j in (i-2, i-1, i+1, i+2)
                       if accepted[j % len(predicted)]]
        if len(nearby) < 1:
            good[i] = False
        elif np.median(np.abs(offsets[winner[nearby]]-offsets[winner[i]])) > 19:
            good[i] = False
    if identity == "F4_outer":
        # The far small circle is faint and partly covered in this passage.
        good &= chosen >= 1.4
        good &= np.abs(offsets[winner]) <= 18
    evidence = []
    for i in np.flatnonzero(good):
        k = int(winner[i])
        peak = float(chosen[i])
        neighborhood = np.flatnonzero((response[i] >= peak*.5)
                                      & (np.abs(np.arange(len(offsets))-k) <= 12))
        width = float(np.ptp(neighborhood)) if len(neighborhood) else 8.0
        sigma = float(np.clip(max(1.6, width/2.5), 1.6, 12.0))
        if identity.endswith("_outer"):
            sigma = max(sigma, 3.0)
        evidence.append({"predicted_uv": np.round(predicted[i], 2).tolist(),
                         "observed_uv": np.round(point[i], 2).tolist(),
                         "normal_offset_px": float(offsets[k]),
                         "signed_gradient_or_ridge": round(peak, 3),
                         "local_contrast": round(float(contrast[i]), 3),
                         "uncertainty_px": round(sigma, 2),
                         "signal_definition": sign})
    return point[good], evidence


def _decode_original(source: Path, first_pts: int, last_pts: int):
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        if str(stream.time_base) != "1/15360":
            raise ValueError(f"Unexpected time base {stream.time_base}")
        container.seek(first_pts-2*15360, stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts < first_pts:
                continue
            if frame.pts > last_pts:
                break
            yield int(frame.pts), frame.to_ndarray(format="bgr24")


def _draw_evidence(frame: np.ndarray, observations: list[dict], pts: int, out: Path) -> None:
    colors = {"board_top": (255, 228, 43), "board_contact": (0, 220, 240),
              "F3_outer": (255, 80, 255), "F4_outer": (188, 90, 255),
              "goal_line_left": (60, 120, 255)}
    for row in observations:
        color = colors[row["identity"]]
        for p in row["points_uv"]:
            cv2.circle(frame, tuple(np.rint(p).astype(int)), 3, color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (740, 32), (20, 29, 39), -1)
    cv2.putText(frame, f"SOURCE-PIXEL EDGE READ | PTS {pts}", (12, 23),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(out), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(out)


def run(stride: int = 10, run_dir: Path = RUN) -> dict:
    source_meta = json.loads((run_dir / "source.json").read_text(encoding="utf-8"))
    source = Path(source_meta["original_source"])
    scene = json.loads((ROOT / "calibration/deliverables/scene.json").read_text(encoding="utf-8"))
    image_motion = json.loads((run_dir / "image_motion.json").read_text(encoding="utf-8"))
    geom = _anchor_geometry(scene)
    initial = np.load(run_dir / "joint_initial.npz")
    initial_states = initial["states"]
    initial_profile = initial["profile"]
    anchor_vector = anchor_camera(scene, 104693760)
    with np.load(run_dir / "observations.npz") as source_frames, \
         np.load(run_dir / "pair_tracks.npz") as pair:
        pts = source_frames["pts"].astype(np.int64)
        bboxes = pair["athlete_bbox"]
    selected = set(range(0, len(pts), stride)) | {60, 120}
    output_rows = []
    review = {}
    decoded_count = 0
    for image_pts, bgr in _decode_original(source, int(pts[0]), int(pts[-1])):
        if decoded_count >= len(pts) or image_pts != int(pts[decoded_count]):
            raise ValueError(f"Source PTS mismatch at {decoded_count}: {image_pts}")
        index = decoded_count
        decoded_count += 1
        if index not in selected:
            continue
        H = np.asarray(image_motion["frames"][index]["H_image_from_anchor"], float)
        signals = _signals(bgr)
        frame_rows = []
        board_top_y = float(np.interp(index, _BOARD_Y_KEYS[:, 0], _BOARD_Y_KEYS[:, 1]))
        board_bottom_y = float(np.interp(index, _BOARD_Y_KEYS[:, 0], _BOARD_Y_KEYS[:, 2]))
        minimum_ice_y = float(np.interp(index, _BOARD_Y_KEYS[:, 0], _BOARD_Y_KEYS[:, 2]) - 5)
        for identity, item in geom.items():
            if identity == "F4_outer" and index < 40:
                # In the blurred first third the small far circle has no
                # reliable outer edge. White ice/board texture formerly won
                # the profile search, so emit no fabricated F4 observations.
                continue
            if identity.endswith("_outer"):
                camera = from_state(anchor_vector, initial_states[index],
                                    lens_profile=initial_profile)
                whole_predicted = fast_project(camera, item["world_xyz"])
                center_uv = fast_project(camera, item["center_world_xyz"][None])[0]
                whole_normals = _normals(identity, whole_predicted, item, H,
                                         center_uv=center_uv)
            else:
                whole_predicted = _move(H, item["anchor_uv"])
                whole_normals = _normals(identity, whole_predicted, item, H)
            visible_search = np.isfinite(whole_predicted).all(axis=1) & np.isfinite(whole_normals).all(axis=1)
            visible_search &= (whole_predicted[:, 0] > -65) & (whole_predicted[:, 0] < 1985)
            visible_search &= (whole_predicted[:, 1] > -65) & (whole_predicted[:, 1] < 1145)
            if identity.startswith("board"):
                expected = board_top_y if identity == "board_top" else board_bottom_y
                visible_search &= np.abs(whole_predicted[:, 1] - expected) < 85
            predicted = whole_predicted[visible_search]
            normals = whole_normals[visible_search]
            read, evidence = _profile_read(identity, predicted, normals,
                                           signals[identity], bboxes[index], minimum_ice_y)
            if not len(read):
                continue
            count_floor = 7 if identity.endswith("_outer") else (6 if identity.startswith("board") else 8)
            if len(read) < count_floor:
                continue
            offsets = np.asarray([e["normal_offset_px"] for e in evidence])
            uncertainties = np.asarray([e["uncertainty_px"] for e in evidence])
            gradients = np.asarray([e["signed_gradient_or_ridge"] for e in evidence])
            if identity.endswith("_outer"):
                width_floor = 8.0 if identity == "F4_outer" else (7.0 if index <= 20 else 4.0)
                uncertainties = np.maximum(uncertainties, width_floor)
                for item_evidence, sigma in zip(evidence, uncertainties):
                    item_evidence["uncertainty_px"] = float(sigma)
            row = {"frame_index": index, "pts": image_pts, "identity": identity,
                   "points_uv": np.round(read, 2).tolist(),
                   "quality": {"observed_points": int(len(read)),
                               "search_candidates": int(len(predicted)),
                               "median_signed_gradient_or_ridge": float(np.median(gradients)),
                               "median_normal_offset_px": float(np.median(offsets)),
                               "offset_p95_abs_px": float(np.percentile(np.abs(offsets), 95)),
                               "source": "original video decoded at exact PTS; bounded color/grayscale profile search"},
                   "uncertainty_px": float(np.median(uncertainties)),
                   "point_evidence": evidence}
            if identity == "F4_outer":
                row["quality"]["visibility_class"] = "weak_faint_outer_ring"
            output_rows.append(row)
            frame_rows.append(row)
        if index in (0, 60, 120):
            _draw_evidence(bgr.copy(), frame_rows, image_pts,
                           run_dir / f"contours_{image_pts}.png")
            review[index] = {r["identity"]: len(r["points_uv"]) for r in frame_rows}
    if decoded_count != len(pts):
        raise ValueError(f"Only decoded {decoded_count} of {len(pts)} frames")
    frame_counts = {str(i): {name: len(next((r["points_uv"] for r in output_rows
                                             if r["frame_index"] == i and r["identity"] == name), []))
                             for name in geom}
                    for i in sorted(selected)}
    result = {
        "schema_version": "rinkspace.shot_tracking.source_pixel_contours.v1",
        "source": str(source), "source_time_base": "1/15360",
        "source_frame_pts": pts.tolist(), "anchor_pts": int(pts[60]),
        "model_guidance": "F3/F4: frozen anchor scene projected by preliminary continuous shared-K PTZ states; board/L: anchor projection transformed by raw-pixel image_motion H. All are search guidance, not observations",
        "source_identity_review": "F3 gray ring tracked by visual continuity through the 13 source keyframes; first 40-frame F4 omitted because no reliable distinct outer edge",
        "identities": list(geom), "observations": output_rows,
        "frame_counts": frame_counts,
        "review_images": [f"contours_{int(pts[i])}.png" for i in (0, 60, 120)],
        "limitations": [
            "Gradient/ridge detections are source-image measurements near model-guided windows, not independent 3-D rink geometry.",
            "Weak, blurred, occluded and out-of-frame segments are omitted, never filled by predicted coordinates.",
            "Point evidence reports response and search offset; uncertainty is local pixel edge width, not full model error.",
        ],
    }
    out = run_dir / "contours.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected_frames": len(selected), "rows": len(output_rows),
                      "point_count": sum(len(r["points_uv"]) for r in output_rows),
                      "endpoint_counts": review}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()
    run(args.stride)
