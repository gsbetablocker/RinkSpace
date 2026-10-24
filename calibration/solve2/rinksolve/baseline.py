"""Independent S01 demo-data importer and pinhole initialization.

No legacy solver or renderer is imported.  The historical table describes an
undistorted Z=0 plane warp, even though camera25 has extra optical coefficients.
Those coefficients remain provenance only: this initializer uses C, R and K,
zero Brown distortion and no row-time correction.  Conditional scene units are
preserved.  Projection equality is not material alignment or visual acceptance.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "rinkspace.solve2.s01_baseline_initialization.v1"
CAMERA_CONTRACT = {
    "world_to_camera": "q=R_cw@(X-C_w)",
    "pixel": "[u,v,1]~K@q",
    "plane": "H=K@[R_cw[:,0],R_cw[:,1],-R_cw@C_w]",
    "Brown": "zero; camera25[11:] is preserved but inactive provenance",
    "row_time": "none",
    "coordinate_units": "original conditional scene units; no metric conversion",
    "pixel_units": "official1920x1080 native pixel centers",
    "clock": "ISU official50 native PTS; no resampling or clock conversion",
    "first_shot_display": "frozen first_shot_v2.mp4, not mutable combined outro",
}
BRIDGE_NAMES = {0: "direct", 1: "same_shot_interpolated", 2: "held"}
PROVENANCE_KEYS = (
    "pts", "native_pts", "time_seconds", "camera25", "direct_mapping",
    "bridge_mode", "shot_index", "anchor_left_official_pts",
    "anchor_right_official_pts", "anchor_left_legacy_pts", "anchor_right_legacy_pts",
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def rotation_exp(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues exponential, implemented directly with a stable small-angle branch."""
    vector = np.asarray(rotvec, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("rotation vector must be three finite numbers")
    x, y, z = vector
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    theta2 = float(vector @ vector)
    if theta2 < 1e-12:
        a = 1 - theta2 / 6 + theta2 * theta2 / 120
        b = 0.5 - theta2 / 24 + theta2 * theta2 / 720
    else:
        theta = np.sqrt(theta2)
        a = np.sin(theta) / theta
        b = (1 - np.cos(theta)) / theta2
    return np.eye(3) + a * cross + b * (cross @ cross)


def camera_from_original(camera25: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert the explicitly documented first 11 fields, with no Brown application."""
    camera = np.asarray(camera25, dtype=np.float64)
    if camera.shape != (25,) or not np.isfinite(camera).all():
        raise ValueError("camera25 must have exactly 25 finite fields")
    center = camera[:3].copy()
    rotation = rotation_exp(camera[3:6])
    fx, fy = np.exp(camera[6:8])
    if not np.isfinite([fx, fy]).all() or min(fx, fy) <= 0:
        raise ValueError("invalid positive focal lengths")
    intrinsic = np.array([[fx, camera[10], camera[8]], [0, fy, camera[9]], [0, 0, 1.]])
    return center, rotation, intrinsic


def plane_homography(center: np.ndarray, rotation: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    h = intrinsic @ np.column_stack((rotation[:, 0], rotation[:, 1], -rotation @ center))
    if abs(h[2, 2]) < 1e-12 or not np.isfinite(h).all():
        raise ValueError("plane homography has invalid normalization")
    return h / h[2, 2]


def project_world_pinhole(center: np.ndarray, rotation: np.ndarray, intrinsic: np.ndarray,
                          xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("world points must be a finite Nx3 array")
    camera = (points - center) @ rotation.T
    homogeneous = camera @ intrinsic.T
    valid = np.isfinite(homogeneous).all(axis=1) & (camera[:, 2] > 1e-10)
    uv = np.full((len(points), 2), np.nan)
    uv[valid] = homogeneous[valid, :2] / homogeneous[valid, 2, None]
    return uv, valid


def project_homography(h: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("plane points must be a finite Nx2 array")
    q = np.column_stack((points, np.ones(len(points)))) @ h.T
    valid = np.isfinite(q).all(axis=1) & (np.abs(q[:, 2]) > 1e-10)
    uv = np.full((len(points), 2), np.nan)
    uv[valid] = q[valid, :2] / q[valid, 2, None]
    return uv, valid


def _points(value: Any) -> np.ndarray:
    data = np.asarray(value, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 3 or len(data) < 2 or not np.isfinite(data).all():
        raise ValueError("scene path must contain at least two finite XYZ points")
    return data


def extract_scene_paths(scene: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Interpret actual JSON primitives, preserving IDs/open sides/material curves.

    The 52 unobserved standard references are not venue paint.  Diagnostic grid
    polygons are retained as grid, not as observations. Shield landmark points
    remain points; no unobserved shield outline is invented.
    """
    paths: list[dict[str, Any]] = []
    unknown: list[str] = []

    def add(pid: str, group: str, xyz: Any, *, closed: bool = False,
            label: str | None = None, provenance: str = "", status: str = "", source_id: str | None = None) -> None:
        p = _points(xyz)
        paths.append({"id": pid, "original_id": source_id or pid, "group": group, "label": label or pid,
                      "closed": bool(closed), "xyz": p.tolist(),
                      "provenance": provenance, "geometry_status": status,
                      "is_observation": False, "initial_geometry_only": True,
                      "constraint_semantics_unassigned": True,
                      "samples_are_physical_landmarks": False})

    add(str(scene["ice_boundary"]["id"]), "board", scene["ice_boundary"]["points_xyz"],
        closed=True, provenance="ice_boundary.points_xyz")
    board = scene["board_surface"]
    for key in ("bottom_edge_xyz", "top_edge_xyz"):
        add("boards:" + key, "board", board[key], closed=True, provenance="board_surface." + key)
    # A board mesh vertex is a computational facet, not a physical panel seam.
    for cell in scene["ice_grid"]["cells"]:
        add(cell["id"], "grid", cell["polygon_xyz"], closed=True,
            provenance="ice_grid.cells", status="diagnostic grid; not measured paint")
    for feature in scene.get("ice_features", []):
        fid = str(feature["id"])
        status = feature.get("geometry_status", "")
        if feature.get("surface_id") != "ice_surface":
            raise ValueError("ice feature has unexpected surface: " + fid)
        kind = feature["kind"]
        if kind == "circle":
            rings = feature.get("rings") or [{"role": "outer", "center_xyz": feature["center_xyz"], "radius": feature["radius"]}]
            for ring in rings:
                center = np.asarray(ring["center_xyz"], dtype=np.float64)
                radius = float(ring["radius"])
                if center.shape != (3,) or radius <= 0 or abs(center[2]) > 1e-9:
                    raise ValueError("invalid ice circle: " + fid)
                angles = np.linspace(0, 2 * np.pi, 257)
                xyz = center + radius * np.column_stack((np.cos(angles), np.sin(angles), np.zeros(len(angles))))
                add(fid + ":" + ring["role"], "ice", xyz, closed=True, label=fid,
                    provenance="ice_features.rings or circle parameters", status=status)
        elif "paint_segments_xyz" in feature:
            for i, segment in enumerate(feature["paint_segments_xyz"]):
                add(fid + ":" + str(i), "ice", segment, label=fid,
                    provenance="ice_features.paint_segments_xyz; open edges preserved", status=status)
        elif "segments_xyz" in feature:
            for i, segment in enumerate(feature["segments_xyz"]):
                add(fid + ":" + str(i), "ice", segment, label=fid,
                    provenance="ice_features.segments_xyz", status=status)
        elif "polygon_xyz" in feature:
            vertices = _points(feature["polygon_xyz"])
            open_edges = set(feature.get("open_edges", []))
            if open_edges:
                for i in range(len(vertices)):
                    if i not in open_edges:
                        add(fid + ":" + str(i), "ice", [vertices[i], vertices[(i + 1) % len(vertices)]],
                            label=fid, provenance="ice_features.polygon_xyz with explicit open_edges", status=status)
            else:
                add(fid, "ice", vertices, closed=feature.get("closed", False),
                    provenance="ice_features.polygon_xyz", status=status)
        else:
            unknown.append(fid)
    for ad in scene.get("advertisements", []):
        number = int(ad["user_number"])
        curves = ad.get("fitted_boundary_curves_xyz")
        if not curves:
            patterned_ids = {int(p["advertisement_id"]) for p in scene.get("source_conditioned_board_prints", {}).get("patterns", [])}
            if number not in patterned_ids:
                unknown.append("Ad" + str(number) + ":no_explicit_material_boundary")
            continue
        for role, curve in curves.items():
            add("Ad" + str(number) + ":" + role, "advertisement", curve,
                label="Ad" + str(number), provenance="advertisements.fitted_boundary_curves_xyz",
                status=ad.get("geometry_status", ""))
    for pattern in scene.get("source_conditioned_board_prints", {}).get("patterns", []):
        for contour in pattern.get("external_contours", []):
            add("Ad" + str(pattern["advertisement_id"]) + ":contour:" + contour["id"], "advertisement_print", contour["xyz_scene_units"], source_id=contour["id"],
                closed=contour.get("closed", False), label="Ad" + str(pattern["advertisement_id"]),
                provenance="source_conditioned_board_prints.external_contours; inferred frozen material, not surveyed")
    shared = scene.get("shared_board_patterns", {})
    for aid, curve in shared.get("circle_outer_stroke_paths_xyz_scene_units", {}).items():
        add("Ad" + aid + ":circle", "advertisement_print", curve, closed=True,
            label="Ad" + aid, provenance="shared_board_patterns.circle_outer_stroke_paths_xyz_scene_units")
    for aid, circle in scene.get("additional_board_circle_patterns", {}).get("circles", {}).items():
        add("Ad" + aid + ":circle", "advertisement_print", circle["circle_outer_stroke_xyz_scene_units"],
            closed=True, label="Ad" + aid, provenance="additional_board_circle_patterns.circles")
    ids = [p["id"] for p in paths]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate extracted scene path identity: " + repr(sorted({value for value in ids if ids.count(value) > 1})))
    shield = shared.get("shield_fixed_material_points_xyz_scene_units", {})
    audit = {"path_count": len(paths), "counts": {g: sum(p["group"] == g for p in paths) for g in sorted({p["group"] for p in paths})},
             "unparsed": unknown, "excluded_unobserved_standard_ids": [f["id"] for f in scene.get("unobserved_standard_reference_ice_features", [])],
             "shield_points_preserved_no_inferred_edges": shield,
             "geometry_is_initialization_not_observed_XYZ": True,
             "widthy_ice_straights": "B1/B2/C remain zero-width centerlines, no artificial paint edges"}
    return paths, audit


def validate_native_table(table: dict[str, np.ndarray], clock: dict[str, Any], source: dict[str, Any]) -> None:
    missing = set(PROVENANCE_KEYS + ("world_xy_to_native_pixel_H",)) - table.keys()
    if missing:
        raise ValueError("missing baseline arrays: " + ",".join(sorted(missing)))
    pts = table["native_pts"]
    expected = np.arange(int(clock["first_pts"]), int(clock["last_pts_inclusive"]) + int(clock["pts_step"]), int(clock["pts_step"]), dtype=np.int64)
    if pts.dtype.kind not in "iu" or pts.ndim != 1 or not np.array_equal(pts, expected):
        raise ValueError("baseline native PTS differ from exact S01 exposure clock")
    if len(pts) != int(clock["frames"]) or not np.array_equal(table["pts"], pts):
        raise ValueError("baseline PTS identity/frame count mismatch")
    if Fraction(source["native_fps"]) != 50 or Fraction(source["time_base"]) != Fraction(1, 12800):
        raise ValueError("initializer requires official50 native 1/12800 clock")
    seconds = pts * float(Fraction(source["time_base"]))
    if not np.allclose(table["time_seconds"], seconds, rtol=0, atol=1e-12):
        raise ValueError("baseline seconds disagree with native PTS")
    n = len(pts)
    for key in PROVENANCE_KEYS:
        if len(table[key]) != n:
            raise ValueError("baseline array length mismatch: " + key)
    if table["camera25"].shape != (n, 25) or table["world_xy_to_native_pixel_H"].shape != (n, 3, 3):
        raise ValueError("camera or homography array shape mismatch")
    if not np.isfinite(table["camera25"]).all() or not np.isfinite(table["world_xy_to_native_pixel_H"]).all():
        raise ValueError("camera or homography contains nonfinite numbers")
    if not np.all(table["shot_index"] == 1):
        raise ValueError("baseline contains an exposure outside S01")
    mode = table["bridge_mode"]
    if not np.isin(mode, list(BRIDGE_NAMES)).all():
        raise ValueError("unknown source mapping mode")
    if not np.array_equal(table["direct_mapping"].astype(bool), mode == 0):
        raise ValueError("direct flag disagrees with mapping mode")
    left, right = table["anchor_left_official_pts"], table["anchor_right_official_pts"]
    if ((left < pts[0]) | (right > pts[-1]) | (left > pts) | (right < pts)).any():
        raise ValueError("mapping anchors do not bracket native exposure within S01")
    direct = mode == 0
    if ((left[direct] != pts[direct]) | (right[direct] != pts[direct])).any():
        raise ValueError("direct source mapping has nonidentity official anchors")
    if (left[mode == 1] >= right[mode == 1]).any():
        raise ValueError("interpolated mapping has degenerate anchor bracket")


@dataclass
class BaselineInitialization:
    directory: Path
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    scene: dict[str, Any]
    paths: list[dict[str, Any]]

    @property
    def pts(self) -> np.ndarray:
        return self.arrays["native_pts"]

    def package_index(self, pts: int) -> int:
        if not isinstance(pts, Integral) or isinstance(pts, (bool, np.bool_)):
            raise ValueError("request an integer native PTS, never seconds or frame number")
        value = int(pts)
        i = int(np.searchsorted(self.pts, value))
        if i == len(self.pts) or int(self.pts[i]) != value:
            raise ValueError("native PTS is not an initialized S01 exposure: " + str(value))
        return i

    def camera(self, pts: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        i = self.package_index(pts)
        return self.arrays["C_w"][i], self.arrays["R_cw"][i], self.arrays["K"][i]

    def project_world(self, pts: int, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return project_world_pinhole(*self.camera(pts), xyz)

    def project_ice(self, pts: int, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("ice points must be Nx2 original scene XY")
        return self.project_world(pts, np.column_stack((points, np.zeros(len(points)))))


def load_initialization(directory: str | Path) -> BaselineInitialization:
    """Load only the current self-contained package; no historical qualification replay."""
    directory = Path(directory)
    metadata = _json(directory / "initialization.json")
    if metadata.get("schema") != SCHEMA:
        raise ValueError("unsupported initialization schema")
    with np.load(directory / "initialization.npz", allow_pickle=False) as packed:
        arrays = {key: packed[key] for key in packed.files}
    validate_native_table(arrays, metadata["clock"], metadata["source"])
    n = len(arrays["native_pts"])
    for key, shape in (("C_w", (n, 3)), ("R_cw", (n, 3, 3)), ("K", (n, 3, 3)), ("H", (n, 3, 3))):
        if key not in arrays or arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError("invalid current initialization array: " + key)
    for key, shape in (("brown_zero", (n, 5)), ("row_time_zero", (n,))):
        if key not in arrays or arrays[key].shape != shape or not np.isfinite(arrays[key]).all() or np.any(arrays[key] != 0):
            raise ValueError("pinhole initialization has unexpected active optical/time fields: " + key)
    if not np.allclose(arrays["R_cw"] @ arrays["R_cw"].transpose(0, 2, 1), np.eye(3), atol=1e-10, rtol=0):
        raise ValueError("current camera rotations are not orthogonal")
    if not np.allclose(np.linalg.det(arrays["R_cw"]), 1, atol=1e-10, rtol=0):
        raise ValueError("current camera rotations are not right handed")
    return BaselineInitialization(directory, metadata, arrays, _json(directory / "scene.json"), _json(directory / "paths.json"))


def _plane_test_points(package: BaselineInitialization) -> np.ndarray:
    sampled = [np.asarray(p["xyz"])[:, :2] for p in package.paths if p["group"] in {"grid", "ice"}]
    points = np.unique(np.vstack(sampled), axis=0)
    if len(points) < 4 or np.linalg.matrix_rank(np.column_stack((points, np.ones(len(points))))) != 3:
        raise ValueError("scene does not provide noncollinear plane verification points")
    return points


def verify_projection(package: BaselineInitialization, *, tolerance_px: float = 1e-5) -> dict[str, Any]:
    """Compare new independent XYZ pinhole vs frozen plane-H for every exposure.

    Metrics retain every finite *front-of-camera* ice/grid sample. Native
    viewport errors are separated from out-of-view errors, which can be
    arbitrarily magnified near the horizon. H-only behind-camera points are
    counted, not silently accepted. No actual-source material residual is inferred.
    """
    xy = _plane_test_points(package)
    xyz = np.column_stack((xy, np.zeros(len(xy))))
    width, height = package.metadata["source"]["width"], package.metadata["source"]["height"]
    rows = []
    total_squared = 0.
    total_compared = 0
    maximum = 0.
    all_maximum = 0.
    all_squared = 0.
    all_count = 0
    maximum_h = 0.
    missing_camera_pts = []
    for i, pts in enumerate(package.pts):
        fresh, fresh_valid = package.project_world(int(pts), xyz)
        reference, reference_valid = project_homography(package.arrays["world_xy_to_native_pixel_H"][i], xy)
        both = fresh_valid & reference_valid
        missing = fresh_valid & ~reference_valid
        if not both.any():
            missing_camera_pts.append(int(pts))
        delta = np.linalg.norm(fresh[both] - reference[both], axis=1)
        squared = float(delta @ delta)
        peak = float(delta.max()) if len(delta) else 0.
        visible = both & (reference[:, 0] >= 0) & (reference[:, 0] < width) & (reference[:, 1] >= 0) & (reference[:, 1] < height)
        h = package.arrays["H"][i]
        frozen = package.arrays["world_xy_to_native_pixel_H"][i]
        normalized = frozen / frozen[2, 2]
        visible_delta = np.linalg.norm(fresh[visible] - reference[visible], axis=1)
        visible_squared = float(visible_delta @ visible_delta)
        visible_peak = float(visible_delta.max()) if len(visible_delta) else 0.
        h_error = float(np.max(np.abs(h - normalized)))
        row = {"native_pts": int(pts), "source_seconds": float(package.arrays["time_seconds"][i]),
               "bridge_mode": BRIDGE_NAMES[int(package.arrays["bridge_mode"][i])],
               "compared_points": int(both.sum()), "in_view_points": int(visible.sum()),
               "new_camera_behind_or_singular_points": int((~fresh_valid).sum()),
               "reference_singular_points": int((~reference_valid).sum()),
               "missing_front_camera_reference_points": int(missing.sum()),
               "maximum_px": visible_peak, "rms_euclidean_px": float(np.sqrt(visible_squared / len(visible_delta))) if len(visible_delta) else None,
               "all_front_finite_maximum_px": peak, "all_front_finite_rms_euclidean_px": float(np.sqrt(squared / len(delta))) if len(delta) else None,
               "normalized_H_max_abs": h_error}
        rows.append(row)
        maximum = max(maximum, visible_peak)
        maximum_h = max(maximum_h, h_error)
        total_squared += visible_squared
        total_compared += len(visible_delta)
        all_maximum = max(all_maximum, peak)
        all_squared += squared
        all_count += len(delta)
    passed = (not missing_camera_pts and all(r["missing_front_camera_reference_points"] == 0 and r["in_view_points"] > 0 for r in rows)
              and maximum <= tolerance_px and maximum_h <= 1e-8)
    report = {"schema": "rinkspace.solve2.s01_projection_equivalence.v1", "claim": "independent C/R/K XYZ pinhole equals frozen demo plane-H in native viewport; normalized H equality also checked",
              "metric_domain": "front-camera scene ice/grid samples within native image; all offscreen finite sample errors separately retained",
              "exposures": len(rows), "scene_plane_test_points": len(xy), "total_compared_in_view_points": total_compared,
              "normalized_H_max_abs": maximum_h, "normalized_H_absolute_tolerance": 1e-8,
              "all_front_finite_points": all_count, "all_front_finite_maximum_px": all_maximum,
              "all_front_finite_rms_euclidean_px": float(np.sqrt(all_squared / all_count)) if all_count else None,
              "maximum_px": maximum, "rms_euclidean_px": float(np.sqrt(total_squared / total_compared)) if total_compared else None,
              "missing_camera_native_pts": missing_camera_pts, "missing_front_camera_reference_points": sum(r["missing_front_camera_reference_points"] for r in rows),
              "tolerance_px": tolerance_px, "mathematical_plane_equivalence_pass": passed,
              "old_demo_pixel_display_equivalence_proven": False, "whole_shot_material_alignment_accepted": False,
              "three_dimensional_board_ad_alignment_accepted": False,
              "limitations": ["This is a sampled numerical projection comparison, not a decoded-old-demo or physical-material alignment test.",
                              "Offscreen points close to the projective horizon amplify floating-point cancellation; their raw absolute errors are reported separately and do not certify pixel alignment.",
                              "Old demo fade, masks, particles, heat and raster grid were not reproduced or compared.",
                              "Three-dimensional board/ad pinhole is an explicit new initialization extension; plane equality does not certify it.",
                              "No Brown/rolling-shutter equivalence is claimed; extra original camera25 fields are inactive."],
              "frames": rows}
    return report


def import_baseline(manifest: str | Path, output: str | Path) -> BaselineInitialization:
    """One-time pure-data conversion; never starts an optimizer or imports old code."""
    manifest_path = Path(manifest)
    provenance = _json(manifest_path)
    clock, source = provenance["clock"], provenance["source"]
    if provenance["scope_lock"]["active_shot"] != "S01":
        raise ValueError("only S01 is authorized for this importer")
    old = provenance["old_demo_s01"]
    scene_path = Path(old["scene"]["review_path"])
    table_path = Path(old["native_projection_table"]["path"])
    # These two inputs are converted here. Unchanged historical source chains,
    # optimizer state, complete checkpoints and source packets are not replayed.
    for path, expected in ((scene_path, old["scene"]["sha256"]), (table_path, old["native_projection_table"]["sha256"])):
        if _sha256(path) != expected:
            raise ValueError("changed conversion input SHA256: " + str(path))
    scene = _json(scene_path)
    if scene["coordinate_system"].get("verified_metre_scale"):
        raise ValueError("unexpected metric scene contract; review conversion explicitly")
    with np.load(table_path, allow_pickle=False) as packed:
        table = {key: packed[key] for key in packed.files}
    validate_native_table(table, clock, source)
    cameras = [camera_from_original(camera) for camera in table["camera25"]]
    centers = np.stack([c[0] for c in cameras])
    rotations = np.stack([c[1] for c in cameras])
    intrinsics = np.stack([c[2] for c in cameras])
    homographies = np.stack([plane_homography(*c) for c in cameras])
    paths, scene_audit = extract_scene_paths(scene)
    if scene_audit["unparsed"]:
        raise ValueError("scene has unparsed actual features: " + repr(scene_audit["unparsed"]))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    arrays = dict(table)
    arrays.update(C_w=centers, R_cw=rotations, K=intrinsics, H=homographies,
                  brown_zero=np.zeros((len(centers), 5)), row_time_zero=np.zeros(len(centers)))
    metadata = {"schema": SCHEMA, "shot_id": "S01", "scope_lock": provenance["scope_lock"],
                "source": source, "clock": clock, "camera_contract": CAMERA_CONTRACT,
                "mapping_counts": {BRIDGE_NAMES[mode]: int(np.count_nonzero(table["bridge_mode"] == mode)) for mode in BRIDGE_NAMES},
                "historical_demo": {"frozen_video": old["video"], "original_q_identity": old["original_nine_shot_q_sha256"],
                                    "scene": old["scene"], "projection_table": old["native_projection_table"],
                                    "source_packet_identity_only": old["source_packet"]},
                "one_time_conversion_input": {"manifest": str(manifest_path), "manifest_sha256": _sha256(manifest_path)},
                "initialization_files": {"arrays": "initialization.npz", "scene": "scene.json", "paths": "paths.json", "scene_audit": "scene_audit.json"},
                "initial_geometry_only": True, "constraint_semantics_unassigned": True,
                "constraint_policy": "Neither sampled vertices nor drawing endpoints imply locked landmarks or per-frame sliding stations; new model must explicitly assign curve_support versus landmark and estimated versus locked position.",
                "old_optimizer_state_imported": False, "old_solver_or_renderer_imported": False,
                "Brown_coefficients_from_original_inactive": True,
                "acceptance": {"mathematical_plane_equivalence": "see projection_check.json", "whole_shot_real_source_visual": "not performed", "next_shot_allowed": False}}
    np.savez_compressed(output / "initialization.npz", **arrays)
    _write_json(output / "scene.json", scene)
    _write_json(output / "paths.json", paths)
    _write_json(output / "scene_audit.json", scene_audit)
    _write_json(output / "initialization.json", metadata)
    package = load_initialization(output)
    report = verify_projection(package)
    _write_json(output / "projection_check.json", report)
    if not report["mathematical_plane_equivalence_pass"]:
        raise ValueError("initialization projection differs from frozen plane data; see projection_check.json")
    return package


def render_overlays(package: BaselineInitialization, requested_pts: list[int], output: str | Path | None = None) -> dict[str, Any]:
    """Decode exact native source frames and write a few full-model JPEG diagnostics.

    This deliberately does not render video, infer unseen materials, or claim
    occlusion-aware old-demo visual equivalence. The source and overlay are
    paired in one JPEG to permit direct visual inspection.
    """
    import av
    import cv2

    for pts in requested_pts:
        package.package_index(pts)
    selected = sorted(set(int(pts) for pts in requested_pts))
    if not selected:
        raise ValueError("request at least one exact native PTS")
    directory = Path(output) if output else package.directory / "overlays"
    directory.mkdir(parents=True, exist_ok=True)
    source_path = Path(package.metadata["source"]["path"])
    found: dict[int, str] = {}
    with av.open(str(source_path)) as container:
        stream = container.streams.video[0]
        expected_tb = Fraction(package.metadata["source"]["time_base"])
        if Fraction(stream.time_base) != expected_tb or Fraction(stream.average_rate) != 50:
            raise ValueError("source video native clock differs from initialization")
        container.seek(selected[0], stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts < selected[0]:
                continue
            if frame.pts > selected[-1]:
                break
            pts = int(frame.pts)
            if pts not in selected:
                continue
            image = frame.to_ndarray(format="bgr24")
            height, width = image.shape[:2]
            if (width, height) != (package.metadata["source"]["width"], package.metadata["source"]["height"]):
                raise ValueError("source frame dimensions differ from native pixel contract")
            overlay = image.copy()
            labeled: set[str] = set()
            colors = {"grid": (80, 180, 80), "ice": (20, 30, 250), "board": (0, 240, 255),
                      "advertisement": (255, 100, 0), "advertisement_print": (255, 0, 255)}
            for path in package.paths:
                vertices = np.asarray(path["xyz"])
                projected, valid = package.project_world(pts, vertices)
                sequence = list(range(len(vertices)))
                if path["closed"] and np.linalg.norm(vertices[0] - vertices[-1]) > 1e-8:
                    sequence.append(0)
                for a, b in zip(sequence[:-1], sequence[1:]):
                    if not valid[a] or not valid[b] or np.max(np.abs(projected[[a, b]])) > 1e7:
                        continue
                    p, q = tuple(np.rint(projected[a]).astype(int)), tuple(np.rint(projected[b]).astype(int))
                    hit, start, end = cv2.clipLine((0, 0, width, height), p, q)
                    if hit:
                        cv2.line(overlay, start, end, colors[path["group"]], 1 if path["group"] == "grid" else 2, cv2.LINE_AA)
                        if path["group"] != "grid" and path["label"] not in labeled:
                            cv2.putText(overlay, path["label"], start, cv2.FONT_HERSHEY_SIMPLEX, .45, colors[path["group"]], 1, cv2.LINE_AA)
                            labeled.add(path["label"])
            # Preserve isolated confirmed shield landmarks without fabricating edges.
            audit = _json(package.directory / "scene_audit.json")
            for aid, points in audit["shield_points_preserved_no_inferred_edges"].items():
                for role, xyz in points.items():
                    uv, valid = package.project_world(pts, np.asarray([xyz]))
                    if valid[0] and 0 <= uv[0, 0] < width and 0 <= uv[0, 1] < height:
                        point = tuple(np.rint(uv[0]).astype(int))
                        cv2.drawMarker(overlay, point, (255, 0, 255), cv2.MARKER_CROSS, 8, 1)
                        cv2.putText(overlay, "Ad" + aid, point, cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 0, 255), 1)
            stamp = "S01 nativePTS %d | source %.2fs | pinhole seed; NOT VISUALLY ACCEPTED" % (pts, pts * float(expected_tb))
            cv2.rectangle(overlay, (0, 0), (width, 34), (0, 0, 0), -1)
            cv2.putText(overlay, stamp, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
            # Full native images remain available; compact pair is 1920x540.
            pair = np.hstack((cv2.resize(image, (width // 2, height // 2)), cv2.resize(overlay, (width // 2, height // 2))))
            target = directory / ("s01_pts%d_pair.jpg" % pts)
            if not cv2.imwrite(str(target), pair, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise OSError("failed to write overlay " + str(target))
            found[pts] = str(target)
    report = {"schema": "rinkspace.solve2.s01_source_overlay_diagnostic.v1", "source": str(source_path),
              "requested_native_pts": selected, "frames": [{"native_pts": pts, "source_seconds": pts / 12800, "path": found[pts]} for pts in selected if pts in found],
              "missing_native_pts": [pts for pts in selected if pts not in found],
              "projection": CAMERA_CONTRACT, "whole_shot_visual_acceptance": False,
              "limitations": ["Selected frames only; no continuous shot review performed.", "No 3D board occlusion test or person mask: visible back edges may be diagnostic ghosts.", "All parsed ice, diagnostic grid, boards, ad boundaries/numbers/prints shown; unknown standard paint is excluded.", "Source left; independent pinhole initialization overlay right."]}
    _write_json(directory / "overlay_manifest.json", report)
    if report["missing_native_pts"]:
        raise ValueError("source did not decode requested exact native PTS")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("import", help="convert audited pure S01 demo data into an independent package")
    create.add_argument("--manifest", type=Path, required=True)
    create.add_argument("--out", type=Path, required=True)
    create.add_argument("--overlay-pts", type=int, nargs="*", default=[])
    check = commands.add_parser("check", help="verify the current independent package only")
    check.add_argument("--init-dir", type=Path, required=True)
    overlay = commands.add_parser("overlays", help="write selected exact-native source comparisons")
    overlay.add_argument("--init-dir", type=Path, required=True)
    overlay.add_argument("--pts", type=int, nargs="+", required=True)
    args = parser.parse_args(argv)
    if args.command == "import":
        package = import_baseline(args.manifest, args.out)
        if args.overlay_pts:
            render_overlays(package, args.overlay_pts)
        report = _json(package.directory / "projection_check.json")
    elif args.command == "check":
        package = load_initialization(args.init_dir)
        report = verify_projection(package)
        _write_json(package.directory / "projection_check.json", report)
    else:
        package = load_initialization(args.init_dir)
        report = render_overlays(package, args.pts)
    summary = {key: value for key, value in report.items() if key not in {"frames", "limitations"}}
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report.get("mathematical_plane_equivalence_pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
