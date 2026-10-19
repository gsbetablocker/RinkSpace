"""Read-only source-pixel inventory for the 14 v0.25 long-side references.

``load_inputs`` returns the original 1920x1080 curve pixels and the two
directly read advertisement corner sets. The other twelve corner sets are
retained in a separate *derived* field: their coordinates were transferred
with a homography, not measured as four corners in each target image.

The returned world curves/corners are conditional v0.25 scene geometry. They
are not observed 3-D points or verified metre-scale positions. The saved
camera25 vectors are exposed only as initialization guesses. No optimizer,
image matcher, model projection, or result camera is run here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from three_frame_refit.program_tracking.anchors.geometry import world_curve


BANK = ROOT / "calibration/program_tracking/anchors/reference_bank.json"
DIRECT_AD = ROOT / "calibration/program_tracking/anchors/source_ad_observations.json"
TRANSFERRED_AD = ROOT / "calibration/program_tracking/anchors/source_ad3_transfers.json"
SCENE = ROOT / "calibration/deliverables/scene.json"
SHOTS = ROOT / "calibration/program_tracking/source_shots.json"
CORNER_ORDER = ("left_top", "right_top", "right_bottom", "left_bottom")
RAW_CAMERA_IDS = {
    "main_long_side": "main_long_side",
    "secondary_long_side": "camera_secondary_long_side",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _uv(value: object, label: str, *, in_image: bool) -> list[list[float]]:
    points = np.asarray(value, dtype=np.float64)
    _require(points.ndim == 2 and points.shape[1] == 2 and len(points) > 0
             and np.isfinite(points).all(), f"Invalid source UV: {label}")
    if in_image:
        _require(bool(np.all((points[:, 0] >= 0) & (points[:, 0] < 1920)
                             & (points[:, 1] >= 0) & (points[:, 1] < 1080))),
                 f"Measured UV outside source image: {label}")
    return points.tolist()


def load_inputs() -> dict:
    """Verify provenance and return source measurements with typed derivations.

    The returned ``references[*].curves[*].source_uv`` and
    ``ad3.direct_source_uv`` are original-image measurements. Curves may be
    dense remeasurements along one image edge, so their points are correlated.
    ``ad3.transferred_uv_derived`` and all scene XYZ are explicitly derived.
    """
    bank, scene, direct, transferred, timeline = map(_json,
        (BANK, SCENE, DIRECT_AD, TRANSFERRED_AD, SHOTS))
    source_sha = timeline["source_sha256_from_timeline"]
    source_path = (ROOT / timeline["source_path"]).resolve()
    _require(timeline["source_sha256_verified_against_file"] is True,
             "Programme source hash has no checked timeline binding")
    _require(len(bank["references"]) == 14, "Expected fourteen released references")
    _require(Path(bank["model_path"]).resolve() == SCENE.resolve(),
             "Reference bank points at a different scene")
    _require(direct["source"] == str(source_path.relative_to(ROOT)).replace("\\", "/")
             and direct["time_base"] == "1/15360"
             and direct["pixel_domain"] == "native distorted 1920x1080",
             "Direct advertisement file has another source or pixel domain")
    _require(scene["coordinate_system"]["unit"] == "conditional_scene_units"
             and scene["coordinate_system"]["verified_metre_scale"] is False,
             "Scene geometry unit status changed")
    scene_sha = _digest(SCENE)
    ad3 = next(a for a in scene["advertisements"] if a["user_number"] == 3)
    ad3_xyz = [ad3["corners_xyz_scene_units"][role] for role in CORNER_ORDER]
    source_files: dict[str, dict] = {}
    references = []
    for ref in bank["references"]:
        camera_id, pts = ref["camera_id"], int(ref["source_pts"])
        _require(camera_id in RAW_CAMERA_IDS and ref["status"] == "accepted",
                 f"Unexpected reference camera/status: {ref['id']}")
        observation_path = Path(ref["source_observation_file"]).resolve()
        if str(observation_path) not in source_files:
            source_files[str(observation_path)] = {
                "path": str(observation_path), "sha256": _digest(observation_path),
                "data": _json(observation_path),
            }
        source_file = source_files[str(observation_path)]
        raw = source_file["data"]
        frames = [f for f in raw["frames"] if int(f["source_pts"]) == pts
                  and Path(f["image_path"]).resolve() == Path(ref["image_path"]).resolve()]
        _require(len(frames) == 1, f"Missing/ambiguous source frame: {ref['id']}")
        frame = frames[0]
        _require(frame["camera_id"] == RAW_CAMERA_IDS[camera_id]
                 and frame["source_sha256"] == source_sha
                 and Path(frame["source_path"]).resolve() == source_path
                 and frame["time_base"] == "1/15360"
                 and frame["image_size_wh"] == [1920, 1080],
                 f"Camera/source/pixel binding differs: {ref['id']}")
        image_path = Path(ref["image_path"]).resolve()
        image_sha = _digest(image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        _require(image is not None and image.shape[:2] == (1080, 1920)
                 and image_sha == frame["image_sha256"],
                 f"Image size/hash differs: {ref['id']}")
        expected_ids = set(ref["fit_observation_ids"])
        _require("ad3_four_source_corners" in expected_ids,
                 f"Reference fit omits its ad #3 corner provenance: {ref['id']}")
        expected_ids.discard("ad3_four_source_corners")
        rows = [o for o in raw["observations"]
                if o["frame_id"] == frame["id"] and o["id"] in expected_ids]
        _require({o["id"] for o in rows} == expected_ids,
                 f"Missing fitted source curves: {ref['id']}")
        eligible_ids = {o["id"] for o in raw["observations"]
                        if o["frame_id"] == frame["id"]
                        and not o.get("synthetic_or_model_projected")
                        and len(o.get("pixels_uv", [])) >= 3
                        and world_curve(scene, o["target"]) is not None}
        _require(expected_ids == eligible_ids,
                 f"Reference curve selection differs from original fit: {ref['id']}")
        curves = []
        for row in rows:
            _require(row["type"] == "curve" and row["use"] == "fit"
                     and row.get("synthetic_or_model_projected") is False
                     and row.get("pure_transfer_points_exported") is False
                     and row.get("measurement_origin", "").startswith("actual_image"),
                     f"Non-source image curve: {row['id']}")
            xyz = world_curve(scene, row["target"])
            _require(xyz is not None and np.isfinite(xyz).all(),
                     f"No conditional scene curve: {row['id']}")
            uv = _uv(row["pixels_uv"], row["id"], in_image=True)
            curves.append({
                "id": row["id"], "source_uv": uv,
                "source_uv_role": "real_image_edge_or_marking_remeasurement",
                "source_uv_correlation": "same_curve_samples_not_independent_pixels",
                "source_pts": pts, "target": row["target"],
                "physical_group_id": row.get("physical_group_id"),
                "identity_status": row.get("identity_status"),
                "identity_evidence": row.get("identity_evidence"),
                "measurement_origin": row["measurement_origin"],
                "measurement_method": row.get("method"),
                "seed_observation_id": row.get("source_seed_observation_id"),
                "sigma_px": float(row["sigma_px"]),
                "world_curve_xyz_scene_units": np.asarray(xyz, float).tolist(),
                "world_curve_role": "derived_conditional_scene_geometry_not_measured_xyz",
                "world_geometry_source": str(SCENE),
                "world_geometry_target": row["target"],
            })
        direct_rows = [o for o in direct["observations"]
                       if o["camera_id"] == camera_id and int(o["source_pts"]) == pts]
        transfer_rows = [o for o in transferred["observations"]
                         if o["camera_id"] == camera_id and int(o["source_pts"]) == pts]
        _require(len(direct_rows) + len(transfer_rows) == 1,
                 f"Ad #3 corner provenance ambiguous: {ref['id']}")
        ad_row = (direct_rows or transfer_rows)[0]
        _require(int(ad_row.get("user_number", 3)) == 3
                 and Path(ad_row["source_image"]).resolve() == image_path,
                 f"Ad #3 identity/image differs: {ref['id']}")
        _require(ref["ad_corner_measurement_method"] == ad_row.get(
            "method", "assistant_source_image_visual_reading")
            and ref.get("ad_transfer_inliers") == ad_row.get("inliers"),
            f"Reference bank and ad #3 pixel origin differ: {ref['id']}")
        corners = _uv(ad_row["corners_uv"], f"{ref['id']}:ad3",
                      in_image=bool(direct_rows))
        _require(len(corners) == 4 and float(ad_row["sigma_px"]) > 0,
                 f"Invalid ad #3 corners: {ref['id']}")
        transferred_corner_indices = ([i for i, (u, v) in enumerate(corners)
                                       if not (0 <= u < 1920 and 0 <= v < 1080)]
                                      if transfer_rows else [])
        ad_info = {
            "user_number": 3,
            "physical_identity_authority": direct["identity_source"],
            "source_pixel_authority": direct["measurement_authority"],
            "corner_order": list(CORNER_ORDER),
            "direct_source_uv": corners if direct_rows else None,
            "transferred_uv_derived": corners if transfer_rows else None,
            "pixel_role": ("assistant_original_image_corner_reading" if direct_rows else
                           "homography_transfer_from_same_ad3_source_image"),
            "pixel_source_file": str(DIRECT_AD if direct_rows else TRANSFERRED_AD),
            "sigma_px": float(ad_row["sigma_px"]),
            "method": ad_row.get("method", "assistant_source_image_visual_reading"),
            "transfer_inliers": ad_row.get("inliers"),
            "transfer_median_match_error_px": ad_row.get("median_match_error_px"),
            "transferred_outside_image_corner_indices": transferred_corner_indices,
            "world_corners_xyz_scene_units": ad3_xyz,
            "world_geometry_role": "derived_conditional_scene_ad3_not_measured_xyz",
            "world_geometry_source": str(SCENE),
        }
        references.append({
            "id": ref["id"], "source_pts": pts, "source_time_s": pts / 15360,
            "camera_id": camera_id, "raw_camera_id": frame["camera_id"],
            "camera_assignment_authority": frame.get("camera_assignment_authority"),
            "installation_id": frame.get("installation_id"),
            "source_shot_id": frame.get("shot_id"),
            "source_video_path": str(source_path), "source_video_sha256": source_sha,
            "image_path": str(image_path), "image_sha256": image_sha,
            "image_size_wh": [1920, 1080],
            "source_observation_file": str(observation_path),
            "source_observation_sha256": source_file["sha256"],
            "curves": curves, "ad3": ad_info,
            "initial_camera25": ref["camera_vector25"],
            "initial_camera25_role": "old_fitted_camera_initializer_only_not_observation",
        })
    return {
        "schema": "rinkspace.reference_calibration_source_inventory.v1",
        "role": "read_only_development_inputs_not_independent_validation",
        "source_video_path": str(source_path), "source_video_sha256": source_sha,
        "source_time_base": "1/15360", "pixel_domain": "native distorted 1920x1080",
        "scene_path": str(SCENE), "scene_sha256": scene_sha,
        "scene_unit": scene["coordinate_system"]["unit"],
        "verified_metre_scale": scene["coordinate_system"]["verified_metre_scale"],
        "reference_bank_path": str(BANK), "reference_bank_sha256": _digest(BANK),
        "direct_ad_source_path": str(DIRECT_AD),
        "direct_ad_source_sha256": _digest(DIRECT_AD),
        "transferred_ad_source_path": str(TRANSFERRED_AD),
        "transferred_ad_source_sha256": _digest(TRANSFERRED_AD),
        "references": references,
    }


def manifest(inventory: dict) -> dict:
    """Compact provenance manifest; ``load_inputs`` retains every original UV."""
    refs = []
    for row in inventory["references"]:
        curves = []
        for curve in row["curves"]:
            curves.append({key: curve.get(key) for key in (
                "id", "target", "physical_group_id", "identity_status",
                "identity_evidence", "measurement_origin", "measurement_method",
                "seed_observation_id", "sigma_px", "source_uv_role",
                "source_uv_correlation", "world_curve_role", "world_geometry_target")}
                | {"source_uv_count": len(curve["source_uv"])})
        ad = row["ad3"].copy()
        ad.pop("world_corners_xyz_scene_units")
        refs.append({key: row[key] for key in (
            "id", "source_pts", "source_time_s", "camera_id", "raw_camera_id",
            "camera_assignment_authority", "installation_id", "source_shot_id",
            "image_path", "image_sha256", "image_size_wh",
            "source_observation_file", "source_observation_sha256",
            "initial_camera25_role")}
            | {"curves": curves, "ad3": ad})
    return {key: inventory[key] for key in inventory if key != "references"} | {
        "references": refs,
        "summary": {
            "reference_frames": len(refs),
            "camera_frames": dict(Counter(r["camera_id"] for r in refs)),
            "source_curve_groups": sum(len(r["curves"]) for r in refs),
            "source_curve_pixels": sum(sum(c["source_uv_count"] for c in r["curves"])
                                       for r in refs),
            "direct_ad3_frames": sum(r["ad3"]["direct_source_uv"] is not None for r in refs),
            "transferred_ad3_frames": sum(r["ad3"]["transferred_uv_derived"] is not None
                                          for r in refs),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Write compact provenance JSON")
    args = parser.parse_args()
    result = manifest(load_inputs())
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print(args.manifest.resolve())
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
