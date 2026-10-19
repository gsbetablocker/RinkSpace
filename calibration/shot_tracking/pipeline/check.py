"""Source-time, optical, and endpoint-source checks for the 121-frame trial.

Usage:
  python calibration/shot_tracking/pipeline/check.py \
    --poses calibration/shot_tracking/runs/low_end_pm1/poses.json

The endpoint readings in agent_readings.json were made from original video
frames before inspecting the trial pose, then audited against those frames.
Their physical board/ice groups also occur in fitted contours, so these
readings are development evidence, not an independent holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "calibration" / "work"
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))

from physical_projection import nearest_valid_vectors, safe_project, verify_camera  # noqa: E402

ANCHOR_PTS = 104693760
FIRST_PTS = ANCHOR_PTS - 15360
LAST_PTS = ANCHOR_PTS + 15360
EXPECTED_PTS = list(range(FIRST_PTS, LAST_PTS + 1, 256))
IMAGE_SIZE = (1920, 1080)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(name: str, from_file: Path) -> Path:
    path = Path(name)
    if path.is_absolute():
        return path
    for candidate in (ROOT / path, from_file.parent / path):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(name)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scene_camera_vector(scene: dict, pts: int) -> np.ndarray:
    """Convert exported physical camera to the solver's 25-component vector."""
    frames = [f for f in scene["frame_cameras"] if int(f["source_pts"]) == pts]
    if len(frames) != 1:
        raise ValueError(f"Scene needs one camera at PTS {pts}")
    c = frames[0]["camera"]
    k = np.asarray(c["K_px"], float)
    d = np.asarray(c["distortion_opencv"], float)
    if k.shape != (3, 3) or len(d) != 14 or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("Invalid exported K/Brown camera")
    if not np.allclose(d[12:14], 0, atol=1e-12):
        raise ValueError("Exported Brown sensor tilt is not zero")
    return np.r_[c["center_xyz"], c["rvec_world_to_camera"],
                 math.log(k[0, 0]), math.log(k[1, 1]),
                 k[0, 2], k[1, 2], k[0, 1], d]


def exact_source_pts(source: Path) -> dict:
    """Independently verify the original 2-second source window by native PTS."""
    found: list[int] = []
    sizes: set[tuple[int, int]] = set()
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        time_base = str(stream.time_base)
        container.seek(FIRST_PTS, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts < FIRST_PTS:
                continue
            if frame.pts > LAST_PTS:
                break
            found.append(int(frame.pts))
            sizes.add((frame.width, frame.height))
    return {
        "source": str(source), "time_base": time_base,
        "native_size_wh": [list(x) for x in sorted(sizes)],
        "frame_count": len(found), "first_pts": found[0] if found else None,
        "last_pts": found[-1] if found else None,
        "all_121_exact_pts": found == EXPECTED_PTS,
        "missing_pts": sorted(set(EXPECTED_PTS) - set(found)),
        "extra_pts": sorted(set(found) - set(EXPECTED_PTS)),
    }


def densify_polyline(points: np.ndarray, *, closed: bool, substeps: int = 5) -> np.ndarray:
    p = np.asarray(points, float).reshape(-1, 3)
    if len(p) < 2:
        raise ValueError("Polyline has fewer than two world points")
    if closed:
        p = np.vstack((p, p[0]))
    fractions = np.linspace(0.0, 1.0, substeps, endpoint=False)
    segments = (p[:-1, None, :] * (1.0 - fractions)[None, :, None]
                + p[1:, None, :] * fractions[None, :, None])
    return np.vstack((segments.reshape(-1, 3), p[-1]))


def curve_world(scene: dict, identity: str) -> np.ndarray:
    if identity == "board_contact":
        return densify_polyline(np.asarray(scene["ice_boundary"]["points_xyz"], float), closed=True)
    if identity == "board_top":
        return densify_polyline(np.asarray(scene["board_surface"]["top_edge_xyz"], float), closed=True)
    if identity == "F3_outer":
        feature = next(f for f in scene["ice_features"] if f["id"] == "F3")
        ring = next(r for r in feature["rings"] if r["role"] == "outer")
        theta = np.linspace(0, 2 * np.pi, 2049)
        center = np.asarray(ring["center_xyz"], float)
        points = np.column_stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))
        return center + float(ring["radius"]) * points
    if identity == "goal_line_left":
        feature = next(f for f in scene["ice_features"] if f["id"] == identity)
        return np.vstack([densify_polyline(np.asarray(segment, float), closed=False, substeps=300)
                          for segment in feature["segments_xyz"]])
    raise ValueError(f"Unknown independent curve identity: {identity}")


def visible_scene_samples(scene: dict) -> np.ndarray:
    """A fixed set of shared world locations for optical checks."""
    chunks = [np.asarray(scene["ice_boundary"]["points_xyz"], float),
              np.asarray(scene["board_surface"]["top_edge_xyz"], float)]
    for cell in scene["ice_grid"]["cells"]:
        polygon = np.asarray(cell["polygon_xyz"], float)
        chunks.append(np.mean(polygon, axis=0, keepdims=True))
    points = np.vstack(chunks)
    if not np.isfinite(points).all():
        raise ValueError("Non-finite shared world geometry")
    return points


def manual_fit_provenance(poses: dict, poses_path: Path, readings: dict,
                          readings_path: Path) -> tuple[set[tuple[int, str]], dict]:
    """Verify declared direct manual fit inputs against the exact source file."""
    declared = poses.get("manual_readings_used_for_fit", [])
    if not isinstance(declared, list):
        raise ValueError("manual_readings_used_for_fit must be a list")
    original = {(int(frame["pts"]), curve["identity"]): curve
                for frame in readings["readings"] for curve in frame["curves"]}
    original_hash = file_hash(readings_path)
    used: set[tuple[int, str]] = set()
    for item in declared:
        key = (int(item["pts"]), str(item["identity"]))
        if item.get("readings_sha256") != original_hash:
            raise ValueError(f"Direct fit reading hash differs from current source: {key}")
        if key not in original or original[key].get("use_for_check") is False:
            raise ValueError(f"Direct fit reading is missing or rejected: {key}")
        if key in used:
            raise ValueError(f"Duplicate direct fit reading: {key}")
        used.add(key)
    extra = poses.get("extra_contours")
    if used and not extra:
        raise ValueError("Direct manual fit readings need extra_contours provenance")
    if not extra:
        return used, {"direct_manual_group_count": 0, "extra_contours": None}
    extra_path = resolve_path(extra["path"], poses_path)
    extra_hash = file_hash(extra_path)
    if extra.get("sha256") != extra_hash:
        raise ValueError("Direct fit extra contour source hash differs from pose metadata")
    extra_data = read_json(extra_path)
    from_extra = {(int(item["pts"]), str(item["identity"]))
                  for item in extra_data.get("manual_readings_used_for_fit", [])}
    if used != from_extra:
        raise ValueError("Direct fit reading declarations differ between pose and extra contours")
    for key in used:
        matches = [o for o in extra_data["observations"]
                   if (int(o["pts"]), str(o["identity"])) == key]
        if len(matches) != 1 or not all(point in matches[0]["points_uv"]
                                        for point in original[key]["points_uv"]):
            raise ValueError(f"Direct fit contour pixels differ from original reading: {key}")
    return used, {"direct_manual_group_count": len(used),
                  "extra_contours": str(extra_path), "extra_contours_sha256": extra_hash}


def check_readings(scene: dict, poses_by_pts: dict[int, np.ndarray], readings: dict,
                   readings_path: Path, fit_contours: dict,
                   directly_fitted: set[tuple[int, str]]) -> dict:
    rows = []
    fitted_rows = []
    rejected = []
    fit_groups: dict[str, dict[int, int]] = {}
    for contour in fit_contours["observations"]:
        identity = str(contour["identity"])
        pts = int(contour["pts"])
        count = len(contour["points_uv"])
        frame_counts = fit_groups.setdefault(identity, {})
        frame_counts[pts] = frame_counts.get(pts, 0) + count
    for frame in readings["readings"]:
        pts = int(frame["pts"])
        if pts not in (FIRST_PTS, LAST_PTS):
            raise ValueError("Manual readings must use exact +/-1 second endpoints")
        path = resolve_path(frame["source_image"], readings_path)
        image_sha = file_hash(path)
        if frame.get("source_image_sha256") != image_sha:
            raise ValueError(f"Original reading image changed at PTS {pts}")
        for curve in frame["curves"]:
            identity = curve["identity"]
            uv = np.asarray(curve["points_uv"], float)
            if uv.ndim != 2 or uv.shape[1] != 2 or len(uv) == 0 or not np.isfinite(uv).all():
                raise ValueError(f"Invalid manual reading coordinates at PTS {pts}: {identity}")
            fit_frame_counts = fit_groups.get(identity, {})
            fit_overlap = {
                "same_physical_identity_used_for_fit": bool(fit_frame_counts),
                "same_pts_contour_used_for_fit": pts in fit_frame_counts,
                "same_manual_coordinates_used_directly_for_fit": (pts, identity) in directly_fitted,
                "fit_contour_points_at_same_pts": fit_frame_counts.get(pts, 0),
                "fit_contour_frame_count_for_identity": len(fit_frame_counts),
            }
            if curve.get("use_for_check") is False:
                if (pts, identity) in directly_fitted:
                    raise ValueError(f"Rejected original reading was used for fit: {(pts, identity)}")
                rejected.append({"pts": pts, "identity_as_originally_read": identity,
                                 "observed_uv": uv.tolist(),
                                 "read_uncertainty_px_as_originally_recorded": curve.get("read_uncertainty_px"),
                                 "fit_evidence_overlap": fit_overlap,
                                 "reason": curve.get("rejected_reason", "Reading rejected")})
                continue
            xyz = curve_world(scene, identity)
            projected, _, valid = safe_project(poses_by_pts[pts], xyz)
            delta, segments = nearest_valid_vectors(uv, projected, valid)
            distance = np.linalg.norm(delta, axis=1)
            sigma = float(curve["read_uncertainty_px"])
            row = {
                "pts": pts, "source_image_sha256": image_sha,
                "identity": identity, "surface": curve["surface"],
                "fit_evidence_overlap": fit_overlap,
                "evidence_role": ("direct_fit_input" if (pts, identity) in directly_fitted
                                  else "development_source_alignment"),
                "point_count": len(uv), "read_uncertainty_px": sigma,
                "observed_uv": uv.tolist(),
                "nearest_projected_curve_uv": np.round(uv - delta, 3).tolist(),
                "closest_curve_distance_px": [round(float(x), 3) for x in distance],
                "mean_distance_px": round(float(np.mean(distance)), 3),
                "max_distance_px": round(float(np.max(distance)), 3),
                "within_read_uncertainty": bool(np.all(distance <= sigma) and np.all(segments >= 0)),
                "valid_nearest_segment_count": int(np.sum(segments >= 0)),
            }
            (fitted_rows if (pts, identity) in directly_fitted else rows).append(row)
    all_distances = np.asarray([d for row in rows for d in row["closest_curve_distance_px"]], float)
    if not len(all_distances):
        raise ValueError("No retained endpoint readings to audit")
    return {
        "method": "historical pre-pose manual original-frame readings, subsequently audited; rejected groups keep their original coordinates and do not enter metrics",
        "scope": "endpoint source-alignment development audit, not a held-out accuracy estimate: the joint solver also uses contours of the same physical board and ice-marking groups; exact manual fit inputs are reported separately and excluded from comparison metrics",
        "fit_evidence_source": "contours.json observations consumed by joint.py",
        "rejected_original_readings": rejected,
        "rejected_group_count": len(rejected),
        "direct_fit_input_groups": fitted_rows,
        "direct_fit_input_group_count": len(fitted_rows),
        "groups": rows, "point_count": len(all_distances),
        "median_distance_px": round(float(np.median(all_distances)), 3),
        "p95_distance_px": round(float(np.percentile(all_distances, 95)), 3),
        "max_distance_px": round(float(np.max(all_distances)), 3),
        "all_retained_groups_within_recorded_read_uncertainty": all(r["within_read_uncertainty"] for r in rows),
    }


def check_poses(scene: dict, poses: dict) -> tuple[dict, dict[int, np.ndarray]]:
    if poses.get("time_base") != "1/15360" or int(poses.get("anchor_pts", -1)) != ANCHOR_PTS:
        raise ValueError("Pose time base or anchor PTS differs from exact source")
    frames = poses["frames"]
    actual_pts = [int(f["pts"]) for f in frames]
    if actual_pts != EXPECTED_PTS:
        raise ValueError("Pose rows are not exactly 121 ordered native frames at 256-PTS intervals")
    values = {}
    states = {}
    optical = []
    shared_world = visible_scene_samples(scene)
    width, height = IMAGE_SIZE
    for frame in frames:
        pts = int(frame["pts"])
        camera = np.asarray(frame["camera_vector"], float)
        if camera.shape != (25,) or not np.isfinite(camera).all():
            raise ValueError(f"Invalid 25-component camera at PTS {pts}")
        if np.any(np.abs(camera[23:25]) > 1e-12):
            raise ValueError(f"Brown sensor tilt released at PTS {pts}")
        values[pts] = camera
        states[frame.get("state", "unspecified")] = states.get(frame.get("state", "unspecified"), 0) + 1
        projected, _, valid = safe_project(camera, shared_world)
        in_frame = (valid & (projected[:, 0] >= 0) & (projected[:, 0] <= width - 1)
                    & (projected[:, 1] >= 0) & (projected[:, 1] <= height - 1))
        visible = shared_world[in_frame]
        # Verify a deterministic subset on every frame, and all currently
        # visible samples at the two endpoints and original anchor.
        if pts not in (FIRST_PTS, ANCHOR_PTS, LAST_PTS) and len(visible) > 160:
            visible = visible[np.linspace(0, len(visible) - 1, 160, dtype=int)]
        optical_check = verify_camera(camera, visible,
                                      image_size_wh=IMAGE_SIZE,
                                      sensor_grid=(31, 17) if pts in (FIRST_PTS, ANCHOR_PTS, LAST_PTS) else (17, 11),
                                      inverse_tolerance_px=0.001)
        curve_check = optical_check["curves"][0]
        curve_inverse = curve_check["inverse"]
        sensor_inverse = optical_check["sensor"]
        # The inverse solver terminates on pixel reprojection distance. In a
        # skewed K, its separately scaled ideal-ray distance may be slightly
        # larger than that stopping tolerance. Solve to 0.001 px, then check
        # both distances independently against the 0.005 px acceptance bound.
        curve_pass = bool(
            len(visible) > 0
            and curve_check["fast_valid_count"] == len(visible)
            and curve_inverse["inverse_valid_count"] == len(visible)
            and curve_inverse["max_reprojection_error_px"] is not None
            and curve_inverse["max_reprojection_error_px"] <= 0.005
            and curve_inverse["max_ray_error_px"] is not None
            and curve_inverse["max_ray_error_px"] <= 0.005)
        sensor_pass = bool(
            sensor_inverse["inverse_valid_count"] == sensor_inverse["count"]
            and sensor_inverse["max_reprojection_error_px"] <= 0.005)
        optical.append({
            "pts": pts, "state": frame.get("state", "unspecified"),
            "shared_world_sample_count": len(shared_world),
            "visible_valid_world_sample_count": int(np.sum(in_frame)),
            "strict_curve_check_count": len(visible),
            "central_curve_inverse_pass": curve_pass,
            "central_sensor_inverse_pass": sensor_pass,
            "curve_fast_valid_count": curve_check["fast_valid_count"],
            "curve_inverse_valid_count": curve_inverse["inverse_valid_count"],
            "curve_max_reprojection_error_px": curve_inverse["max_reprojection_error_px"],
            "curve_max_ray_error_px": curve_inverse["max_ray_error_px"],
            "sensor_checked": optical_check["sensor"]["count"],
            "sensor_inverse_valid_count": sensor_inverse["inverse_valid_count"],
            "sensor_max_reprojection_error_px": sensor_inverse["max_reprojection_error_px"],
            "inverse_solver_tolerance_px": 0.001,
            "inverse_acceptance_tolerance_px": 0.005,
        })
    anchored = scene_camera_vector(scene, ANCHOR_PTS)
    projected_0, _, valid_0 = safe_project(anchored, shared_world)
    projected_1, _, valid_1 = safe_project(values[ANCHOR_PTS], shared_world)
    common = (valid_0 & valid_1 & (projected_0[:, 0] >= 0) & (projected_0[:, 0] < width)
              & (projected_0[:, 1] >= 0) & (projected_0[:, 1] < height))
    shift = np.linalg.norm(projected_1[common] - projected_0[common], axis=1)
    return {
        "frame_count": len(frames), "first_pts": actual_pts[0], "anchor_pts": ANCHOR_PTS,
        "last_pts": actual_pts[-1], "all_121_native_pts": True,
        "states": states, "optical_checks": optical,
        "all_checked_central_inverse_pass": all(
            row["central_curve_inverse_pass"] and row["central_sensor_inverse_pass"] for row in optical),
        "anchor_reproduction": {
            "common_visible_scene_samples": int(np.sum(common)),
            "image_shift_rms_px": round(float(np.sqrt(np.mean(shift**2))), 4) if len(shift) else None,
            "image_shift_max_px": round(float(np.max(shift)), 4) if len(shift) else None,
        },
    }, values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--scene", type=Path,
                        default=ROOT / "calibration" / "deliverables" / "scene.json")
    parser.add_argument("--readings", type=Path,
                        default=ROOT / "calibration" / "shot_tracking" / "runs" / "low_end_pm1" / "agent_readings.json")
    parser.add_argument("--contours", type=Path, default=None,
                        help="Source contour observations consumed by joint.py; defaults beside --readings")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    scene = read_json(args.scene)
    poses = read_json(args.poses)
    readings = read_json(args.readings)
    contours_path = args.contours or args.readings.parent / "contours.json"
    fit_contours = read_json(contours_path)
    directly_fitted, manual_provenance = manual_fit_provenance(poses, args.poses,
                                                               readings, args.readings)
    source = resolve_path(poses["source_path"], args.poses)
    if source.name == "source.mp4":
        # A remuxed 121-frame clip may start its local PTS at zero. Always
        # check exact source PTS against the unmodified original video.
        source = ROOT / "data" / "input" / "oxXu1avhrvs.mp4"
    source_check = exact_source_pts(source)
    pose_check, pose_values = check_poses(scene, poses)
    reading_check = check_readings(scene, pose_values, readings, args.readings,
                                   fit_contours, directly_fitted)
    report = {
        "schema_version": "rinkspace.shot_tracking.checks.v1",
        "scope": "exact 01:53:35.000--01:53:37.000 interval, 121 frames",
        "scene_sha256": file_hash(args.scene), "poses_sha256": file_hash(args.poses),
        "readings_sha256": file_hash(args.readings),
        "fit_contours_sha256": file_hash(contours_path),
        "manual_fit_provenance": manual_provenance,
        "source": source_check, "poses": pose_check,
        "endpoint_source_readings": reading_check,
        "validity": {
            "all_native_pts": bool(source_check["all_121_exact_pts"] and pose_check["all_121_native_pts"]),
            "all_sampled_brown_central_inverse": bool(pose_check["all_checked_central_inverse_pass"]),
            "all_retained_endpoint_readings_within_recorded_uncertainty": bool(reading_check["all_retained_groups_within_recorded_read_uncertainty"]),
            "note": "Optical checks sample world/sensor locations only. Endpoint readings share physical groups with fitted contours and cannot certify the whole ice or all 121 frames; original-video visual review remains necessary.",
        },
    }
    output = args.output or args.poses.parent / "checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "validity": report["validity"],
                      "endpoint_p95_px": reading_check["p95_distance_px"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
