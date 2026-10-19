"""Exploratory local-polynomial kinematics of a consistently defined contact point.

This is NOT a centre-of-mass estimator. Input positions must already be in a
Euclidean plane; projective maps, replays and contact gaps cannot be differentiated.
"""
from __future__ import annotations
from typing import Any
import numpy as np
from .geometry import PlaneMap


def split_segments(rows: list[dict[str, Any]], max_gap_s: float = 0.12) -> list[list[dict[str, Any]]]:
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_key = None
    last_t: float | None = None
    for r in rows:
        t = float(r["t_s"])
        if not np.isfinite(t) or (last_t is not None and t <= last_t):
            raise ValueError("Each analysis track must have strictly increasing source times")
        gap = last_t is not None and t - last_t > max_gap_s
        last_t = t
        usable = (r.get("contact") == "on_ice" and r.get("status") == "accepted"
                  and r.get("video_role") == "live" and r.get("x") is not None and r.get("y") is not None)
        # Persist the same map frame, physical entity and observation definition.
        key = (r.get("shot_id"), r.get("map_id"), r.get("entity_id"), r.get("point_definition"), r.get("epoch"))
        if gap or not usable or key != previous_key:
            if current:
                segments.append(current)
            current = []
        if usable:
            if not all(key[:4]):
                raise ValueError("Usable contacts need shot/map/entity/point_definition ids")
            if not np.isfinite(float(r["x"])) or not np.isfinite(float(r["y"])) or float(r.get("sigma", 1)) <= 0:
                raise ValueError("Invalid coordinate or observation sigma")
            current.append(r)
        previous_key = key if usable else None
    if current:
        segments.append(current)
    return segments


def kinematics(rows: list[dict[str, Any]], plane: PlaneMap, half_window_s: float = 0.24,
               max_gap_s: float = 0.12, min_points: int = 5, speed_floor: float = 0.05) -> list[dict[str, Any]]:
    plane.require_euclidean()
    if half_window_s <= 0 or min_points < 3 or max_gap_s <= 0:
        raise ValueError("Invalid temporal fit parameters")
    output = []
    for segment_index, seg in enumerate(split_segments(rows, max_gap_s)):
        t = np.asarray([r["t_s"] for r in seg], dtype=float)
        p = np.asarray([[r["x"], r["y"]] for r in seg], dtype=float)
        sig = np.asarray([r.get("sigma", 1.0) for r in seg], dtype=float)
        if not np.all(np.isfinite(sig)):
            raise ValueError("Nonfinite uncertainty")
        for i, row in enumerate(seg):
            if row["map_id"] != plane.map_id:
                raise ValueError("Rows do not belong to the supplied plane map")
            if t[i] - t[0] < half_window_s - 1e-9 or t[-1] - t[i] < half_window_s - 1e-9:
                continue  # Do not advertise unstable one-sided endpoint derivatives.
            ok = np.abs(t - t[i]) <= half_window_s + 1e-9
            if np.sum(ok) < min_points:
                continue
            dt = t[ok] - t[i]
            # Local quadratic least squares with precision weight 1/sigma^2.
            a = np.c_[np.ones(len(dt)), dt, dt ** 2]
            aw = a / sig[ok, None]
            if np.linalg.matrix_rank(aw) < 3 or np.linalg.cond(aw) > 1e8:
                continue
            c, *_ = np.linalg.lstsq(aw, p[ok] / sig[ok, None], rcond=None)
            velocity, acc = c[1], 2 * c[2]
            speed = float(np.linalg.norm(velocity))
            curvature = None
            angle = None
            radius = None
            if speed >= speed_floor:
                angle = float(np.degrees(np.arctan2(velocity[1], velocity[0])))
                curvature = float((velocity[0] * acc[1] - velocity[1] * acc[0]) / speed ** 3)
                if abs(curvature) > 1e-7:
                    radius = float(1 / abs(curvature))
            residual = np.sqrt(np.mean(np.sum((a @ c - p[ok]) ** 2, axis=1)))
            output.append({"t_s": float(t[i]), "segment_index": segment_index, "entity_id": row["entity_id"],
                           "point_definition": row["point_definition"], "speed": speed,
                           "speed_unit": plane.unit + "/s", "tangent_deg_xy_y_up": angle,
                           "signed_curvature": curvature, "radius": radius, "length_unit": plane.unit,
                           "geometry_level": plane.level, "half_window_s": half_window_s,
                           "fit_rms": float(residual), "uncertainty_interval": None,
                           "status": "exploratory_derivative_not_validated_measurement"})
    return output


def window_sensitivity(rows: list[dict[str, Any]], plane: PlaneMap,
                       windows: tuple[float, ...] = (0.12, 0.24, 0.40)) -> dict[str, Any]:
    series = [kinematics(rows, plane, half_window_s=w) for w in windows]
    lookup = [{round(r["t_s"], 8): r for r in data} for data in series]
    common = sorted(set.intersection(*(set(d) for d in lookup))) if lookup else []
    results = []
    for t in common:
        v = [d[t]["speed"] for d in lookup]
        k = [d[t]["signed_curvature"] for d in lookup]
        results.append({"t_s": t, "speed_min": min(v), "speed_max": max(v),
                        "speed_spread_over_median": float((max(v) - min(v)) / max(float(np.median(v)), 1e-9)),
                        "curvature_min": min(k) if all(x is not None for x in k) else None,
                        "curvature_max": max(k) if all(x is not None for x in k) else None})
    return {"windows_half_width_s": list(windows), "samples": results,
            "note": "Sensitivity range is not a calibrated confidence interval."}
