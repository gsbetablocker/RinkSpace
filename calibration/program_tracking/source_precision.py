"""Frozen source-pixel precision floors for directly measured graph factors.

This module never changes observed UVs or creates measurements.  A measured
source edge can only increase the existing uncertainty in its observed normal
direction.  Missing and unmeasurable source pixels retain their base sigma.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "rinkspace-source-point-precision-v1"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_hash(input_hashes: dict, path: Path, digest: str) -> None:
    key = str(path.resolve())
    if key in input_hashes and input_hashes[key] != digest:
        raise ValueError(f"Conflicting source precision input digest: {path}")
    input_hashes[key] = digest


def _uv(value) -> np.ndarray:
    result = np.asarray(value, np.float64)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError("Source precision UV must be two finite native pixels")
    return result


def _positive(value, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def curve_normal_stability(uv, *, ratio_threshold: float = .7,
                           angle_threshold_deg: float = 30.) -> dict:
    """Flag ambiguous source-UV normals without reading images or fit results.

    The five-neighbor construction matches ``assess_curve_normals``. A second
    seven-neighbor tangent tests whether that local direction is stable. The
    sign of a PCA axis has no physical meaning, so the comparison uses the
    absolute dot product. This diagnostic does not change solver weighting.
    """
    points = np.asarray(uv, np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 5
            or not np.isfinite(points).all()):
        raise ValueError("Curve stability needs at least five finite source UVs")
    if not (0 < ratio_threshold < 1 and 0 < angle_threshold_deg < 90):
        raise ValueError("Invalid curve normal stability thresholds")
    ratio = np.empty(len(points), np.float64)
    angle = np.empty(len(points), np.float64)
    for index, point in enumerate(points):
        distances = np.linalg.norm(points-point, axis=1)
        order = np.argsort(distances)
        tangents = []
        for count in (5, min(7, len(points))):
            neighbors = points[order[:count]]
            _, singular, vt = np.linalg.svd(
                neighbors-neighbors.mean(axis=0), full_matrices=False)
            tangents.append(vt[0])
            if count == 5:
                ratio[index] = (singular[1]/singular[0]
                                if singular[0] > 0 else np.inf)
        angle[index] = np.degrees(np.arccos(np.clip(
            abs(np.dot(tangents[0], tangents[1])), 0., 1.)))
    return {"s2_s1_k5": ratio, "angle_k5_k7_deg": angle,
            "ambiguous": (ratio > ratio_threshold) |
                         (angle > angle_threshold_deg)}


class SourcePixelPrecision:
    """Match frozen source observations by physical ID and exact native UV."""

    def __init__(self, spec, source_hash: str, input_hashes: dict):
        self.enabled = bool(spec)
        self.provenance = None
        self._ad = {}
        self._curve = {}
        self._counts = {
            "ad_calls": 0, "ad_matched": 0, "ad_missing_key": 0,
            "ad_uv_mismatch": 0, "ad_unknown_components": 0,
            "ad_floorup_components": 0,
            "curve_calls": 0, "curve_points": 0, "curve_matched": 0,
            "curve_missing_group_points": 0, "curve_missing_uv_points": 0,
            "curve_unknown_points": 0, "curve_floorup_points": 0,
        }
        if not spec:
            return
        if not isinstance(spec, dict) or "path" not in spec or "sha256" not in spec:
            raise ValueError("Source precision needs a path and SHA-256")
        path = Path(spec["path"])
        digest = _digest(path)
        if digest != spec["sha256"]:
            raise ValueError("Frozen source precision manifest hash changed")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA:
            raise ValueError("Unknown frozen source precision schema")
        if manifest.get("source_sha256") != source_hash:
            raise ValueError("Source precision belongs to another video")
        if manifest.get("frozen_holdout_used_for_fit") is not False:
            raise ValueError("Source precision must not use frozen holdout pixels")
        if not isinstance(manifest.get("input_sha256"), dict):
            raise ValueError("Source precision input SHA-256 inventory is missing")
        for filename, expected in manifest["input_sha256"].items():
            source = Path(filename)
            actual = _digest(source)
            if actual != expected:
                raise ValueError(f"Source precision input changed: {source}")
            _record_hash(input_hashes, source, actual)
        source_video = Path(manifest["source_path"])
        if source_video.stat().st_size != int(manifest["source_file_size"]):
            raise ValueError("Source precision video size changed")
        _record_hash(input_hashes, path, digest)

        for row in manifest["ad_corners"]:
            if row.get("kind") != "ad_corner":
                raise ValueError("Unexpected source precision ad record")
            key = (int(row["pts"]), int(row["user_number"]), str(row["role"]))
            if key in self._ad:
                raise ValueError(f"Duplicate source precision ad corner: {key}")
            location = _uv(row["uv"])
            quality = row["edge_quality"]
            if not np.array_equal(location, _uv(quality["uv"])):
                raise ValueError(f"Ad quality UV differs from source record: {key}")
            floors = quality.get("sigma_floor_uv")
            if not isinstance(floors, list) or len(floors) != 2:
                raise ValueError(f"Ad component sigma floors are malformed: {key}")
            floors = tuple(None if value is None else _positive(value, "ad floor")
                           for value in floors)
            self._ad[key] = (location, floors)

        for row in manifest["curve_groups"]:
            if row.get("kind") != "curve_normal":
                raise ValueError("Unexpected source precision curve record")
            key = (int(row["pts"]), str(row["physical_id"]))
            if key in self._curve:
                raise ValueError(f"Duplicate source precision curve group: {key}")
            pixels = np.asarray(row["uv"], np.float64).reshape(-1, 2)
            qualities = row["normal_quality"]
            if len(pixels) != len(qualities) or not np.isfinite(pixels).all():
                raise ValueError(f"Malformed source precision curve pixels: {key}")
            points = {}
            for pixel, quality in zip(pixels, qualities):
                if not np.array_equal(pixel, _uv(quality["uv"])):
                    raise ValueError(f"Curve quality UV differs from source record: {key}")
                pixel_key = tuple(map(float, pixel))
                if pixel_key in points:
                    raise ValueError(f"Duplicate source precision curve pixel: {key}")
                if quality.get("status") == "measured":
                    floor = _positive(quality["sigma_floor_normal_px"], "curve floor")
                    normal = _uv(quality["normal_uv"])
                    length = np.linalg.norm(normal)
                    if length < 1e-10:
                        raise ValueError(f"Zero source curve normal: {key}")
                    points[pixel_key] = (floor, normal / length)
                elif quality.get("status") == "unknown":
                    points[pixel_key] = (None, None)
                else:
                    raise ValueError(f"Unknown source curve precision status: {key}")
            self._curve[key] = points
        self.provenance = {
            "path": str(path.resolve()), "sha256": digest,
            "source_sha256": source_hash,
            "input_file_count": len(manifest["input_sha256"]),
            "frozen_holdout_used_for_fit": False,
            "policy": "componentwise_ad_max; curve_normal_only_max; exact_source_uv",
        }

    def ad_sigma(self, pts, number, role, uv, base_sigma) -> np.ndarray:
        """Return existing (u,v) sigmas raised only by measured source floors."""
        base = np.broadcast_to(np.asarray(base_sigma, np.float64), (2,)).copy()
        if not np.isfinite(base).all() or np.any(base <= 0):
            raise ValueError("Ad base sigma needs one or two positive values")
        self._counts["ad_calls"] += 1
        if not self.enabled:
            return base
        key = (int(pts), int(number), str(role))
        entry = self._ad.get(key)
        if entry is None:
            self._counts["ad_missing_key"] += 1
            return base
        location, floors = entry
        if not np.array_equal(_uv(uv), location):
            self._counts["ad_uv_mismatch"] += 1
            return base
        self._counts["ad_matched"] += 1
        for component, floor in enumerate(floors):
            if floor is None:
                self._counts["ad_unknown_components"] += 1
            elif floor > base[component]:
                base[component] = floor
                self._counts["ad_floorup_components"] += 1
        return base

    def curve(self, pts, physical_id, uv, base_sigmas) -> tuple[np.ndarray, np.ndarray]:
        """Return normal sigmas and normals in the caller's exact UV order."""
        pixels = np.asarray(uv, np.float64).reshape(-1, 2)
        base = np.asarray(base_sigmas, np.float64).reshape(-1).copy()
        if len(pixels) != len(base) or not np.isfinite(pixels).all():
            raise ValueError("Curve UV and base sigma lengths differ")
        if not np.isfinite(base).all() or np.any(base <= 0):
            raise ValueError("Curve base sigmas must be finite and positive")
        normal = np.zeros((len(pixels), 2), np.float64)
        self._counts["curve_calls"] += 1
        self._counts["curve_points"] += len(pixels)
        if not self.enabled:
            return base, normal
        group = self._curve.get((int(pts), str(physical_id)))
        if group is None:
            self._counts["curve_missing_group_points"] += len(pixels)
            return base, normal
        for index, pixel in enumerate(pixels):
            entry = group.get(tuple(map(float, pixel)))
            if entry is None:
                self._counts["curve_missing_uv_points"] += 1
                continue
            self._counts["curve_matched"] += 1
            floor, direction = entry
            if floor is None:
                self._counts["curve_unknown_points"] += 1
                continue
            normal[index] = direction
            if floor > base[index]:
                base[index] = floor
                self._counts["curve_floorup_points"] += 1
        return base, normal

    @staticmethod
    def wrap_curve(residual, jacobian, base_sigmas, effective_sigmas, normals):
        """Whiten only the measured normal of baseline-scaled 2N curve rows.

        Inputs are callbacks whose residual and Jacobian are already divided
        by each point's scalar baseline sigma. This constant map is placed
        before any nonlinear robust transform.
        """
        base = np.asarray(base_sigmas, np.float64).reshape(-1)
        effective = np.asarray(effective_sigmas, np.float64).reshape(-1)
        directions = np.asarray(normals, np.float64)
        count = len(base)
        if (effective.shape != base.shape or directions.shape != (count, 2)
                or not np.isfinite(base).all() or not np.isfinite(effective).all()
                or not np.isfinite(directions).all() or np.any(base <= 0)
                or np.any(effective < base)):
            raise ValueError("Invalid curve normal precision arrays")
        lengths = np.linalg.norm(directions, axis=1)
        active = effective > base
        if np.any(active & (lengths < 1e-10)):
            raise ValueError("Inflated curve sigma requires a measured normal")
        unit = np.zeros_like(directions)
        nonzero = lengths > 1e-10
        unit[nonzero] = directions[nonzero] / lengths[nonzero, None]
        coefficient = base / effective - 1.
        transform = (np.eye(2)[None, :, :]
                     + coefficient[:, None, None] *
                     unit[:, :, None] * unit[:, None, :])

        def value(states):
            raw = np.asarray(residual(states), np.float64).reshape(count, 2)
            return np.einsum("nij,nj->ni", transform, raw).reshape(-1)

        def derivative(states):
            result = []
            for matrix in jacobian(states):
                raw = np.asarray(matrix, np.float64).reshape(count, 2, -1)
                result.append(np.einsum("nij,njk->nik", transform, raw).reshape(2 * count, -1))
            return result

        return value, derivative

    def report(self):
        if not self.enabled:
            return None
        return {**self.provenance, "counts": dict(self._counts),
                "manifest_ad_corners": len(self._ad),
                "manifest_curve_groups": len(self._curve)}
