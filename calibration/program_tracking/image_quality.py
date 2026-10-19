"""Source-pixel image quality for programme observations.

This module measures *image detail*, independently of camera fits and residuals.
It never edits source observations. A large sigma multiplier means an image
observation should carry less information; ``unknown`` means that the local
pixels do not contain enough texture to diagnose blur, not that they are sharp.

The manifest stores one exact source PTS per programme frame and a small grid
of local measurements. Reference detail is the upper part of the distribution
within the same shot and approximate focal-length bin. This is deliberately a
relative, conservative quality signal, not a calibrated pixel-error model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence
import warnings

import cv2
import numpy as np


SCHEMA = "rinkspace-source-image-quality-v1"
STATUS_UNKNOWN, STATUS_CLEAR, STATUS_SOFT = 0, 1, 2
STATUS_NAMES = np.asarray(("unknown", "clear", "soft"))
SOURCE_WIDTH, SOURCE_HEIGHT = 1920, 1080
GRID_X, GRID_Y = 12, 9
ANALYSIS_WIDTH, ANALYSIS_HEIGHT = 960, 540
PTS_STEP = 256


@dataclass(frozen=True)
class ImageDetail:
    """Unfitted evidence measured in decoded source pixels."""

    score: np.ndarray  # [GRID_Y, GRID_X], larger means finer image detail
    texture: np.ndarray  # [GRID_Y, GRID_X], mid-frequency RMS
    frame_score: float
    textured_tiles: int


def _rms_tiles(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    h, w = values.shape
    if (h, w) != (ANALYSIS_HEIGHT, ANALYSIS_WIDTH):
        raise ValueError("Unexpected image analysis dimensions")
    blocks = values.reshape(GRID_Y, h // GRID_Y, GRID_X, w // GRID_X)
    if weights is None:
        return np.sqrt(np.mean(blocks * blocks, axis=(1, 3)))
    mask = weights.reshape(GRID_Y, h // GRID_Y, GRID_X, w // GRID_X)
    support = np.mean(mask, axis=(1, 3))
    energy = np.mean(blocks * blocks * mask, axis=(1, 3))
    return np.sqrt(energy / np.maximum(support, 1e-6))


def _detail_grid(gray: np.ndarray, roi_mask: np.ndarray | None = None) -> ImageDetail:
    """Compare two spatial-frequency bands only where real texture exists.

    The mid-frequency band supplies the texture gate. The fine/mid ratio and
    normalized edge curvature respond to spread edges while largely cancelling
    exposure/contrast. Blank ice has no valid tile score. All operations use
    decoded image pixels; no projected landmarks or fitted errors enter here.
    """
    if gray.ndim != 2 or gray.shape != (SOURCE_HEIGHT, SOURCE_WIDTH):
        raise ValueError("Expected a native 1920x1080 grayscale source frame")
    half = cv2.resize(gray, (ANALYSIS_WIDTH, ANALYSIS_HEIGHT),
                      interpolation=cv2.INTER_AREA).astype(np.float32)
    base = cv2.GaussianBlur(half, (0, 0), .65)
    middle = cv2.GaussianBlur(base, (0, 0), 1.8)
    coarse = cv2.GaussianBlur(middle, (0, 0), 3.0)
    fine_band, mid_band = base - middle, middle - coarse
    gx = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3, scale=1/8)
    gy = cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3, scale=1/8)
    lap = cv2.Laplacian(base, cv2.CV_32F, ksize=3, scale=1/4)
    weights = None
    if roi_mask is not None:
        if roi_mask.shape != gray.shape:
            raise ValueError("ROI mask must match the native source frame")
        weights = cv2.resize(roi_mask.astype(np.uint8),
                             (ANALYSIS_WIDTH, ANALYSIS_HEIGHT),
                             interpolation=cv2.INTER_AREA).astype(np.float32)/255.
    fine = _rms_tiles(fine_band, weights)
    mid = _rms_tiles(mid_band, weights)
    gradient = np.sqrt(_rms_tiles(gx, weights)**2 + _rms_tiles(gy, weights)**2)
    curvature = _rms_tiles(lap, weights)
    # Each term is contrast-normalized. The offsets prevent tiny/noisy
    # denominators from producing impressive scores on empty ice.
    score = (np.log((fine + .35)/(mid + .35)) +
             .5*np.log((curvature + .35)/(gradient + .35)))
    support = np.ones_like(mid, bool)
    if weights is not None:
        support = np.mean(weights.reshape(GRID_Y, ANALYSIS_HEIGHT//GRID_Y,
                                          GRID_X, ANALYSIS_WIDTH//GRID_X),
                          axis=(1, 3)) >= .45
    textured = (mid >= 1.1) & (gradient >= 1.3) & support
    score = np.where(textured, score, np.nan).astype(np.float32)
    valid = score[np.isfinite(score)]
    frame_score = float(np.percentile(valid, 60)) if len(valid) >= 6 else np.nan
    return ImageDetail(score, mid.astype(np.float32), frame_score, int(len(valid)))


def assess_frame(image: np.ndarray, roi_mask: np.ndarray | None = None) -> ImageDetail:
    """Measure source image detail; low-texture regions remain NaN/unknown."""
    if image.shape[:2] != (SOURCE_HEIGHT, SOURCE_WIDTH):
        raise ValueError("Image must be the original 1920x1080 source frame")
    gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim == 3 else image)
    return _detail_grid(gray, roi_mask)


def assess_observations(image: np.ndarray, uv: np.ndarray,
                        radius_px: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Measure native-pixel patches around observed UV, independent of a fit.

    Returns (detail_score, texture_rms). Noninformative or out-of-frame patches
    have NaN scores. Use a same-feature clear reference before converting these
    raw scores to uncertainty; the saved manifest already does that by shot.
    """
    uv = np.asarray(uv, np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv must have shape (N, 2)")
    if image.shape[:2] != (SOURCE_HEIGHT, SOURCE_WIDTH):
        raise ValueError("Image must be the original source frame")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    radius = int(radius_px)
    if radius < 24:
        raise ValueError("radius_px must be at least 24")
    scores = np.full(len(uv), np.nan, np.float32)
    texture = np.zeros(len(uv), np.float32)
    for i, (x, y) in enumerate(uv):
        if not np.isfinite((x, y)).all():
            continue
        ix, iy = int(round(x)), int(round(y))
        if (ix-radius < 0 or iy-radius < 0 or
                ix+radius >= SOURCE_WIDTH or iy+radius >= SOURCE_HEIGHT):
            continue
        patch = gray[iy-radius:iy+radius, ix-radius:ix+radius]
        base = cv2.GaussianBlur(patch.astype(np.float32), (0, 0), .85)
        middle = cv2.GaussianBlur(base, (0, 0), 2.1)
        coarse = cv2.GaussianBlur(middle, (0, 0), 3.6)
        fine = float(np.sqrt(np.mean((base-middle)**2)))
        mid = float(np.sqrt(np.mean((middle-coarse)**2)))
        gx = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3, scale=1/8)
        gy = cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3, scale=1/8)
        grad = float(np.sqrt(np.mean(gx*gx+gy*gy)))
        lap = float(np.sqrt(np.mean(cv2.Laplacian(
            base, cv2.CV_32F, ksize=3, scale=1/4)**2)))
        texture[i] = mid
        if mid >= 1.1 and grad >= 1.3:
            scores[i] = np.log((fine+.35)/(mid+.35)) + .5*np.log((lap+.35)/(grad+.35))
    return scores, texture


def _step_width(profile: np.ndarray, *, min_contrast: float) -> dict | None:
    """10-90% width of an actual source-pixel color or grayscale step."""
    values = np.asarray(profile, np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) < 25:
        raise ValueError("A source edge profile needs at least 25 native pixels")
    values = np.stack([np.convolve(np.pad(values[:, c], (1, 1),
                                         mode="edge"), [.2, .6, .2], "valid")
                       for c in range(values.shape[1])], axis=1)
    outside = np.median(values[2:7], axis=0)
    inside = np.median(values[-7:-2], axis=0)
    direction = inside-outside
    contrast = float(np.linalg.norm(direction))
    if contrast < min_contrast:
        return None
    response = ((values-outside)@direction)/(contrast*contrast)
    # A printed letter crossing the edge can create a second transition.
    # Require the central step to be reasonably ordered before estimating it.
    middle = len(response)//2
    if (np.median(response[:middle-7]) > .3 or
            np.median(response[middle+7:]) < .7):
        return None
    response = np.maximum.accumulate(response)
    def crossing(level: float) -> float | None:
        indices = np.flatnonzero(response >= level)
        if not len(indices) or indices[0] == 0:
            return None
        j = int(indices[0])
        lo, hi = float(response[j-1]), float(response[j])
        return float(j-1+(level-lo)/max(hi-lo, 1e-6))
    x10, x50, x90 = crossing(.1), crossing(.5), crossing(.9)
    if x10 is None or x50 is None or x90 is None or x90-x10 > 30:
        return None
    return {"width_px": float(x90-x10), "centre_index": x50,
            "contrast_rgb": contrast}


def _corner_side(image: np.ndarray, uv: np.ndarray, role: str,
                 side: str) -> dict:
    vertical = side in ("left", "right")
    normal_axis = 0 if vertical else 1
    normal = float(uv[normal_axis])
    tangent = float(uv[1-normal_axis])
    inside = 1 if side in ("left", "top") else -1
    along = (1 if role.endswith("top") else -1) if vertical else (
        1 if role.startswith("left") else -1)
    n_center = int(round(normal))
    measurements = []
    for offset in (9, 13, 17, 22, 28, 34):
        t_center = int(round(tangent+along*offset))
        axis_values = n_center + inside*np.arange(-20, 21)
        if (axis_values.min() < 0 or
                axis_values.max() >= image.shape[1 if vertical else 0] or
                t_center-1 < 0 or
                t_center+1 >= image.shape[0 if vertical else 1]):
            continue
        if vertical:
            samples = image[t_center-1:t_center+2, axis_values, :]
        else:
            samples = image[axis_values, t_center-1:t_center+2, :]
        profile = np.mean(samples.astype(np.float32), axis=0 if vertical else 1)
        measured = _step_width(profile, min_contrast=18.)
        if measured is None:
            continue
        measured["centre_px"] = n_center + inside*(measured.pop("centre_index")-20)
        measured["along_offset_px"] = offset
        if abs(measured["centre_px"]-normal) <= 12:
            measurements.append(measured)
    if len(measurements) < 3:
        return {"status": "unknown", "support": len(measurements),
                "reason": "no_consistent_source_color_step"}
    widths = np.asarray([item["width_px"] for item in measurements])
    centers = np.asarray([item["centre_px"] for item in measurements])
    contrast = np.asarray([item["contrast_rgb"] for item in measurements])
    width = float(np.median(widths))
    scatter = float(1.4826*np.median(abs(centers-np.median(centers))))
    offset = float(abs(np.median(centers)-normal))
    # This is a conservative localization floor, not an error calibrated from
    # fitted projection residuals. Horizontal and vertical sides remain separate.
    floor = float(np.clip(.4*width+1.5*scatter+.35*offset, 2.5, 20.))
    return {"status": "measured", "support": len(measurements),
            "width_px": round(width, 3), "contrast_rgb": round(float(np.median(contrast)), 3),
            "line_scatter_px": round(scatter, 3),
            "source_uv_offset_px": round(offset, 3),
            "sigma_floor_px": round(floor, 3)}


def assess_corner_edges(image: np.ndarray, uv: Sequence[float], role: str) -> dict:
    """Assess two independent incident printed edges at a measured corner.

    ``sigma_floor_uv`` has separate native-pixel u/v components. An unknown
    edge has ``None`` floor; this cannot be interpreted as a precise 1px edge.
    The exact input UV is returned for provenance, never changed.
    """
    if image.shape != (SOURCE_HEIGHT, SOURCE_WIDTH, 3):
        raise ValueError("Expected original source BGR24 image")
    point = np.asarray(uv, np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("Expected one finite source UV")
    if role not in ("left_top", "right_top", "left_bottom", "right_bottom"):
        raise ValueError("Expected a printed rectangle corner role")
    vertical_side, horizontal_side = role.split("_")
    u_edge = _corner_side(image, point, role, vertical_side)
    v_edge = _corner_side(image, point, role, horizontal_side)
    status = ("measured_both" if u_edge["status"] == v_edge["status"] == "measured"
              else "unknown")
    return {"kind": "ad_corner_source_edges", "uv": point.tolist(),
            "role": role, "status": status,
            "sigma_floor_uv": [u_edge.get("sigma_floor_px"),
                               v_edge.get("sigma_floor_px")],
            "u_from_vertical_edge": u_edge, "v_from_horizontal_edge": v_edge}


def assess_ad_rectangle_corner(image: np.ndarray, corners: Mapping[str, Sequence[float]],
                               role: str) -> dict:
    """Measure a marked ad corner using the *marked* neighboring side directions.

    This is useful for isolated reference images, whose small and slanted ad
    panels can make axis-aligned profiles misleading. Missing neighboring
    corners and sides shorter than 22 native pixels remain unknown. The source
    UVs determine only sampling directions; no camera fit enters this measure.
    """
    if image.shape != (SOURCE_HEIGHT, SOURCE_WIDTH, 3):
        raise ValueError("Expected original source BGR24 image")
    if role not in ("left_top", "right_top", "left_bottom", "right_bottom"):
        raise ValueError("Expected a printed rectangle corner role")
    point = np.asarray(corners[role], np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("Expected finite source UV for selected corner")

    def side(adjacent_role: str, axis: str, inside_sign: int) -> dict:
        if adjacent_role not in corners:
            return {"status": "unknown", "reason": "adjacent_marked_corner_missing"}
        other = np.asarray(corners[adjacent_role], np.float64)
        if other.shape != (2,) or not np.isfinite(other).all():
            return {"status": "unknown", "reason": "adjacent_marked_corner_invalid"}
        direction = other-point
        length = float(np.linalg.norm(direction))
        if length < 22.:
            return {"status": "unknown", "reason": "side_too_short_for_source_profiles",
                    "marked_side_length_px": round(length, 3)}
        tangent = direction/length
        normal = np.array((-tangent[1], tangent[0]))
        if normal[1 if axis == "v" else 0]*inside_sign < 0:
            normal *= -1
        # Use the same 41-pixel native-pixel color-step profile as the
        # programme assay, but follow the actual slanted printed side.
        offsets = sorted(set(float(round(z, 2)) for z in
                             np.clip(np.asarray((9, 13, 17, 22, 28, 34), float),
                                     5., .78*length)))
        measurements = []
        across = np.arange(-20., 21.)
        along = np.asarray((-1., 0., 1.))
        for offset in offsets:
            center = point+offset*tangent
            coords = (center[None, None, :] +
                      along[:, None, None]*tangent[None, None, :] +
                      across[None, :, None]*normal[None, None, :])
            if (np.any(coords[..., 0] < 1) or np.any(coords[..., 0] >= SOURCE_WIDTH-1)
                    or np.any(coords[..., 1] < 1)
                    or np.any(coords[..., 1] >= SOURCE_HEIGHT-1)):
                continue
            colors = cv2.remap(image, coords[..., 0].astype(np.float32),
                               coords[..., 1].astype(np.float32), cv2.INTER_LINEAR)
            measured = _step_width(colors.astype(np.float32).mean(axis=0),
                                   min_contrast=18.)
            if measured is None:
                continue
            measured["centre_px"] = float(measured.pop("centre_index")-20)
            measured["along_offset_px"] = offset
            if abs(measured["centre_px"]) <= 12:
                measurements.append(measured)
        if len(measurements) < 3:
            return {"status": "unknown", "support": len(measurements),
                    "reason": "no_consistent_source_color_step",
                    "marked_side_length_px": round(length, 3),
                    "normal_uv": np.round(normal, 6).tolist()}
        widths = np.asarray([m["width_px"] for m in measurements])
        centers = np.asarray([m["centre_px"] for m in measurements])
        contrasts = np.asarray([m["contrast_rgb"] for m in measurements])
        width = float(np.median(widths))
        scatter = float(1.4826*np.median(abs(centers-np.median(centers))))
        offset = float(abs(np.median(centers)))
        floor = float(np.clip(.4*width+1.5*scatter+.35*offset, 2.5, 20.))
        if offset > 4.5:
            # On the original reference views the dark board cap can be
            # several pixels away from an ad's actual printed top boundary.
            # Its visible color step is not evidence of this marked side's
            # localization precision. Keep the measurement as a diagnostic.
            return {"status": "unknown", "reason": "edge_center_far_from_marked_side",
                    "support": len(measurements),
                    "marked_side_length_px": round(length, 3),
                    "normal_uv": np.round(normal, 6).tolist(),
                    "candidate_width_px": round(width, 3),
                    "candidate_contrast_rgb": round(float(np.median(contrasts)), 3),
                    "candidate_line_scatter_px": round(scatter, 3),
                    "candidate_source_uv_offset_px": round(offset, 3)}
        return {"status": "measured", "support": len(measurements),
                "marked_side_length_px": round(length, 3),
                "normal_uv": np.round(normal, 6).tolist(),
                "width_px": round(width, 3),
                "contrast_rgb": round(float(np.median(contrasts)), 3),
                "line_scatter_px": round(scatter, 3),
                "source_uv_offset_px": round(offset, 3),
                "sigma_floor_px": round(floor, 3)}

    horizontal, vertical = role.split("_")
    u_edge = side(f"{horizontal}_{'bottom' if vertical == 'top' else 'top'}",
                  "u", 1 if horizontal == "left" else -1)
    v_edge = side(f"{'right' if horizontal == 'left' else 'left'}_{vertical}",
                  "v", 1 if vertical == "top" else -1)
    return {"kind": "ad_corner_source_edges", "uv": point.tolist(),
            "role": role,
            "status": ("measured_both" if u_edge["status"] ==
                       v_edge["status"] == "measured" else "unknown"),
            "sigma_floor_uv": [u_edge.get("sigma_floor_px"),
                               v_edge.get("sigma_floor_px")],
            "u_from_vertical_edge": u_edge, "v_from_horizontal_edge": v_edge,
            "method": "marked_side_direction_source_color_step_v1"}


def _existing_corner_edges(uv: Sequence[float], role: str,
                           source_quality: dict) -> dict:
    """Reuse the source extractor's color-specific edge width when available.

    Its chromatic score separates print color from board illumination better
    than the generic RGB step. Re-measuring it generically would inflate an
    already source-derived sigma a second time.
    """
    vertical_side, horizontal_side = role.split("_")
    sides = source_quality.get("incident_sides", {})
    def convert(side: str) -> dict:
        original = sides.get(side, {})
        width = original.get("blur_width_px")
        if width is None or not np.isfinite(width):
            return {"status": "unknown", "reason": "existing_width_unavailable"}
        scatter = float(original.get("line_scatter_px", 0.))
        offset = float(original.get("corner_offset_px", 0.))
        floor = float(np.clip(.4*float(width)+1.5*scatter+.35*offset, 2.5, 20.))
        return {"status": "measured", "support": int(original.get("support", 0)),
                "width_px": float(width), "line_scatter_px": scatter,
                "source_uv_offset_px": offset, "sigma_floor_px": round(floor, 3),
                "method": "existing_color_specific_source_edge_quality"}
    u_edge, v_edge = convert(vertical_side), convert(horizontal_side)
    return {"kind": "ad_corner_source_edges", "uv": list(map(float, uv)),
            "role": role, "status": ("measured_both" if
                u_edge["status"] == v_edge["status"] == "measured" else "unknown"),
            "sigma_floor_uv": [u_edge.get("sigma_floor_px"),
                               v_edge.get("sigma_floor_px")],
            "u_from_vertical_edge": u_edge, "v_from_horizontal_edge": v_edge,
            "method": "reuse_existing_source_edge_quality"}


def assess_curve_normals(image: np.ndarray, uv: np.ndarray) -> list[dict]:
    """Assess real marking-edge spread in each observed curve normal.

    Tangents come only from the source UV polyline. Width is measured in the
    normal direction, so horizontal motion blur need not penalize a horizontal
    board or painted line's normal constraint. Unknown points have no floor.
    """
    points = np.asarray(uv, np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("Curve source UV must be an (N,2) polyline with N>=3")
    if image.shape != (SOURCE_HEIGHT, SOURCE_WIDTH, 3):
        raise ValueError("Expected original source BGR24 image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    result = []
    for i, point in enumerate(points):
        # Solver input de-duplicates source UV with np.unique, which sorts by
        # x. Sorted neighbors need not be neighbors on a curved contour.
        # Local Euclidean neighbors provide an order-independent tangent.
        neighbors = points[np.argsort(np.linalg.norm(points-point, axis=1))[
            :min(5, len(points))]]
        centered = neighbors-neighbors.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        delta = vt[0]
        length = float(np.linalg.norm(delta))
        if length < 1e-6:
            result.append({"kind": "curve_source_normal", "uv": point.tolist(),
                           "status": "unknown", "reason": "degenerate_source_tangent",
                           "sigma_floor_normal_px": None})
            continue
        tangent = delta/length
        normal = np.array((-tangent[1], tangent[0]))
        profiles = []
        axis = np.arange(-20, 21, dtype=np.float32)
        for along in (-5., 0., 5.):
            xy = point[None, :] + along*tangent[None, :] + axis[:, None]*normal[None, :]
            if np.any(xy[:, 0] < 1) or np.any(xy[:, 0] >= SOURCE_WIDTH-1) or np.any(xy[:, 1] < 1) or np.any(xy[:, 1] >= SOURCE_HEIGHT-1):
                continue
            sampled = cv2.remap(gray, xy[:, 0].astype(np.float32)[None, :],
                                xy[:, 1].astype(np.float32)[None, :],
                                cv2.INTER_LINEAR)[0]
            measured = _step_width(sampled, min_contrast=8.)
            if measured is not None and abs(measured["centre_index"]-20) <= 12:
                profiles.append(measured)
        if len(profiles) < 2:
            result.append({"kind": "curve_source_normal", "uv": point.tolist(),
                           "status": "unknown", "reason": "no_consistent_normal_step",
                           "sigma_floor_normal_px": None})
            continue
        width = float(np.median([x["width_px"] for x in profiles]))
        center = float(np.median([x["centre_index"]-20 for x in profiles]))
        contrast = float(np.median([x["contrast_rgb"] for x in profiles]))
        floor = float(np.clip(.4*width+.35*abs(center), 2., 16.))
        result.append({"kind": "curve_source_normal", "uv": point.tolist(),
                       "status": "measured", "normal_uv": normal.tolist(),
                       "support": len(profiles), "width_px": round(width, 3),
                       "contrast_gray": round(contrast, 3),
                       "source_uv_offset_px": round(abs(center), 3),
                       "sigma_floor_normal_px": round(floor, 3)})
    return result


def _multiplier(gap: np.ndarray) -> np.ndarray:
    """Conservative monotone detail deficit -> sigma inflation, never < 1."""
    return np.minimum(4., np.exp(1.5*np.maximum(0., gap-.12)))


def _normalize(raw_score: np.ndarray, raw_frame_score: np.ndarray,
               shot_id: np.ndarray, zoom_log: np.ndarray) -> tuple[np.ndarray, ...]:
    """Within-shot, similar-zoom upper-detail baseline; no residual inputs."""
    n = len(raw_score)
    frame_score = np.full(n, np.nan, np.float32)
    frame_ref = np.full(n, np.nan, np.float32)
    frame_multiplier = np.ones(n, np.float32)
    frame_status = np.zeros(n, np.uint8)
    tile_ref = np.full_like(raw_score, np.nan)
    tile_multiplier = np.ones_like(raw_score, np.float32)
    tile_status = np.zeros(raw_score.shape, np.uint8)
    finite_zoom = np.isfinite(zoom_log)
    zoom_bin = np.zeros(n, np.int16)
    zoom_bin[finite_zoom] = np.rint(zoom_log[finite_zoom]/.12).astype(np.int16)
    for shot in np.unique(shot_id):
        if shot < 0:  # isolated original reference frames have no same-shot baseline
            continue
        own = np.flatnonzero(shot_id == shot)
        for bin_id in np.unique(zoom_bin[own]):
            idx = own[np.abs(zoom_bin[own]-bin_id) <= 1]
            target = own[zoom_bin[own] == bin_id]
            if len(idx) < 24:
                idx = own
            frame_score_raw = raw_frame_score[idx]
            valid_frames = np.isfinite(frame_score_raw)
            if np.count_nonzero(valid_frames) >= 24:
                reference = float(np.percentile(frame_score_raw[valid_frames], 85))
                for i in target:
                    values = raw_score[i][np.isfinite(raw_score[i])]
                    if len(values) < 6:
                        continue
                    score = float(np.percentile(values, 60))
                    frame_score[i] = score
                    frame_ref[i] = reference
                    mul = float(_multiplier(np.asarray(reference-score)))
                    frame_multiplier[i] = mul
                    frame_status[i] = STATUS_SOFT if mul > 1.05 else STATUS_CLEAR
            source = raw_score[idx]
            # Keep tiles with enough textured examples; empty ice cannot set a
            # sharp baseline for observations in that tile.
            support = np.sum(np.isfinite(source), axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                reference_tiles = np.nanpercentile(source, 85, axis=0)
            reference_tiles[support < 20] = np.nan
            for i in target:
                valid = np.isfinite(raw_score[i]) & np.isfinite(reference_tiles)
                tile_ref[i][valid] = reference_tiles[valid]
                mul = _multiplier(reference_tiles[valid]-raw_score[i][valid])
                tile_multiplier[i][valid] = mul
                tile_status[i][valid] = np.where(mul > 1.05,
                                                  STATUS_SOFT, STATUS_CLEAR)
    return (frame_score, frame_ref, frame_multiplier, frame_status,
            tile_ref, tile_multiplier, tile_status)


def _all_programme_pts(shots: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    pts, ids = [], []
    for i, shot in enumerate(shots):
        values = np.arange(int(shot["start_pts"]), int(shot["end_pts"]),
                           PTS_STEP, dtype=np.int64)
        if not len(values) or values[-1] != int(shot["end_pts"])-PTS_STEP:
            raise ValueError(f"Shot {i} has invalid exact-PTS boundaries")
        pts.append(values)
        ids.append(np.full(len(values), i, np.int16))
    all_pts, shot_id = np.concatenate(pts), np.concatenate(ids)
    if len(np.unique(all_pts)) != len(all_pts) or np.any(np.diff(all_pts) != PTS_STEP):
        raise ValueError("Programme PTS must be continuous and unique")
    return all_pts, shot_id


def load_zoom_track(candidate_dir: str | Path, shots: Sequence[dict]) -> dict[int, float]:
    """Interpolate candidate log focal length per shot, solely for zoom bins.

    The zoom estimate is not used as a blur/residual measurement. It only avoids
    comparing image texture at substantially different magnifications.
    """
    root = Path(candidate_dir)
    output: dict[int, float] = {}
    for i, shot in enumerate(shots):
        path = root/f"shot{i:02d}.npz"
        if not path.exists():
            path = root/f"shot{i:02d}_keyframes.npz"
        with np.load(path, allow_pickle=False) as data:
            stamps = np.asarray(data["pts"], np.int64)
            log_focal = np.mean(np.asarray(data["camera25"], np.float64)[:, 6:8], axis=1)
        expected = np.arange(int(shot["start_pts"]), int(shot["end_pts"]),
                             PTS_STEP, dtype=np.int64)
        if not np.array_equal(stamps[:1], expected[:1]) or not np.array_equal(stamps[-1:], expected[-1:]):
            raise ValueError(f"Candidate shot {i} does not span its source cut")
        interpolated = np.interp(expected, stamps, log_focal)
        output.update(zip(map(int, expected), map(float, interpolated)))
    return output


def build_quality_manifest(source: str | Path, shots: Sequence[dict] | str | Path,
                           output: str | Path, *, source_sha256: str,
                           zoom_by_pts: Mapping[int, float] | None = None,
                           extra_pts: Sequence[int] = (),
                           progress_every: int = 600) -> dict:
    """Sequentially decode all programme source frames and save image quality.

    Extra reference PTS are separately sought; this never decodes the hours
    between them and the programme. The supplied source hash is recorded and
    must be independently established by the caller or project checkpoint.
    """
    import av

    source, output = Path(source), Path(output)
    if not source.is_file() or len(source_sha256) != 64:
        raise ValueError("Existing source and its 64-character SHA-256 are required")
    if isinstance(shots, (str, Path)):
        shots = json.loads(Path(shots).read_text(encoding="utf-8"))["shots"]
    pts, shot_id = _all_programme_pts(shots)
    extras = sorted(set(map(int, extra_pts))-set(map(int, pts)))
    combined_pts = np.r_[pts, extras].astype(np.int64)
    combined_shot = np.r_[shot_id, np.full(len(extras), -1, np.int16)]
    zoom = np.asarray([float(zoom_by_pts.get(int(p), np.nan))
                       if zoom_by_pts is not None else np.nan
                       for p in combined_pts], np.float32)
    raw_score = np.full((len(combined_pts), GRID_Y, GRID_X), np.nan, np.float32)
    texture = np.zeros_like(raw_score)
    raw_frame_score = np.full(len(combined_pts), np.nan, np.float32)
    textured_tiles = np.zeros(len(combined_pts), np.int16)

    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        container.seek(max(0, int(pts[0])-2*15360), stream=stream, backward=True)
        next_i = 0
        for frame in container.decode(stream):
            stamp = frame.pts
            if stamp is None or stamp < pts[0]:
                continue
            if next_i == len(pts):
                break
            if stamp != int(pts[next_i]):
                raise ValueError(f"Missing exact source PTS {pts[next_i]}; next decoded {stamp}")
            detail = assess_frame(frame.to_ndarray(format="bgr24"))
            raw_score[next_i], texture[next_i] = detail.score, detail.texture
            raw_frame_score[next_i] = detail.frame_score
            textured_tiles[next_i] = detail.textured_tiles
            next_i += 1
            if progress_every and next_i % progress_every == 0:
                print(json.dumps({"stage": "image_quality", "frames": next_i,
                                  "total": len(pts)}), flush=True)
        if next_i != len(pts):
            raise ValueError(f"Source ended after {next_i}/{len(pts)} programme frames")
        for i, stamp in enumerate(extras, start=len(pts)):
            container.seek(max(0, stamp-2*15360), stream=stream, backward=True)
            found = None
            for frame in container.decode(stream):
                if frame.pts == stamp:
                    found = frame
                    break
                if frame.pts is not None and frame.pts > stamp:
                    break
            if found is None:
                raise ValueError(f"Missing original reference PTS {stamp}")
            detail = assess_frame(found.to_ndarray(format="bgr24"))
            raw_score[i], texture[i] = detail.score, detail.texture
            raw_frame_score[i] = detail.frame_score
            textured_tiles[i] = detail.textured_tiles

    (frame_score, frame_ref, frame_mul, frame_status,
     tile_ref, tile_mul, tile_status) = _normalize(raw_score, raw_frame_score,
                                                 combined_shot, zoom)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output,
        schema=np.asarray(SCHEMA), source_sha256=np.asarray(source_sha256),
        source_path=np.asarray(str(source.resolve())),
        source_file_size=np.asarray(source.stat().st_size, np.int64),
        pts=combined_pts, shot_index=combined_shot, zoom_log=zoom,
        frame_score=frame_score, frame_reference=frame_ref,
        frame_textured_tiles=textured_tiles,
        frame_sigma_multiplier=frame_mul, frame_status=frame_status,
        tile_score=raw_score.astype(np.float16),
        tile_reference=tile_ref.astype(np.float16),
        tile_texture=texture.astype(np.float16),
        tile_sigma_multiplier=tile_mul.astype(np.float16),
        tile_status=tile_status,
        analysis_size=np.asarray((ANALYSIS_WIDTH, ANALYSIS_HEIGHT), np.int16),
        source_size=np.asarray((SOURCE_WIDTH, SOURCE_HEIGHT), np.int16),
        tile_grid=np.asarray((GRID_X, GRID_Y), np.int16))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    summary = {
        "schema": SCHEMA, "path": str(output.resolve()), "sha256": digest,
        "source_sha256": source_sha256, "programme_frames": len(pts),
        "extra_pts": extras, "status_counts": {
            name: int(np.sum(frame_status[:len(pts)] == i))
            for i, name in enumerate(STATUS_NAMES)},
        "sigma_multiplier_quantiles": np.percentile(frame_mul[:len(pts)],
                                            [0, 50, 90, 95, 99, 100]).tolist(),
        "limitations": ("Relative image-detail evidence; not a calibrated pixel-error "
                        "model. Empty or changing texture can be unknown. Original "
                        "reference PTS are recorded without a same-shot clear baseline.")}
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_source_precision_manifest(source: str | Path, config_path: str | Path,
                                    output: str | Path, *, source_sha256: str) -> dict:
    """Measure fitted ad corners and keyframe curves in their exact source frames.

    This is separate from the broad frame-detail manifest. Its directional
    floors should be combined with each original sigma using ``max``, never
    multiplied into a source-derived blur sigma. Withheld QA observations are
    intentionally excluded from this fitted-observation inventory.
    """
    import av

    source, config_path, output = Path(source), Path(config_path), Path(output)
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    ad_rows: dict[tuple[int, int, str], dict] = {}
    inputs: dict[str, str] = {}
    for filename in config.get("ad_corner_observations", []):
        path = Path(filename)
        payload = path.read_bytes()
        inputs[str(path.resolve())] = hashlib.sha256(payload).hexdigest()
        manifest = json.loads(payload)
        if manifest.get("source_sha256") != source_sha256:
            raise ValueError(f"Ad observation source hash differs: {path}")
        for row in manifest["observations"]:
            if row.get("status") == "measured":
                key = (int(row["pts"]), int(row["user_number"]), row["role"])
                ad_rows[key] = {"pts": key[0], "kind": "ad_corner",
                                "user_number": key[1], "role": key[2],
                                "uv": list(map(float, row["uv"])),
                                "source_sigma_px": float(row.get("sigma_px", 3.)),
                                "already_has_source_edge_quality":
                                    "source_edge_quality" in row,
                                "input_file": str(path.resolve()),
                                "_existing_quality": row.get("source_edge_quality")}
    curve_inputs: dict[tuple[int, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    for item in config["shots"]:
        with np.load(item["keyframes"], allow_pickle=False) as data:
            stamps = (data["original_keyframe_pts"] if "original_keyframe_pts" in data.files
                      else data["pts"])
        for spec in item.get("curves", []):
            for stamp in map(int, stamps):
                if "only_pts" in spec and stamp not in spec["only_pts"]:
                    continue
                path = Path(spec.get("overrides", {}).get(
                    str(stamp), str(Path(spec["directory"])/f"{stamp}.npz")))
                if not path.exists():
                    continue
                payload = path.read_bytes()
                inputs[str(path.resolve())] = hashlib.sha256(payload).hexdigest()
                with np.load(path, allow_pickle=False) as data:
                    ids = np.asarray(data["physical_id"])
                    all_uv = np.asarray(data["observed_uv"], np.float64)
                    all_sigma = (np.asarray(data["point_sigma_px"], np.float64)
                                 if "point_sigma_px" in data.files else None)
                    for name in sorted(set(map(str, np.unique(ids))) & set(spec["ids"])):
                        if item.get("source_edges") and name in (
                                "board_top", "ice_board_contact"):
                            continue
                        selected = ids == name
                        uv = all_uv[selected]
                        if len(uv) < 6:
                            continue
                        baseline = float(config.get(
                            "board_sigma_px" if name in ("board_top", "ice_board_contact")
                            else "marking_sigma_px", 2.))
                        sigma = (all_sigma[selected] if all_sigma is not None
                                 else np.full(len(uv), baseline))
                        curve_inputs.setdefault((stamp, name), []).append((uv.copy(), sigma.copy()))
    curves = {}
    for key, packets in curve_inputs.items():
        combined_uv = np.concatenate([p[0] for p in packets])
        combined_sigma = np.concatenate([p[1] for p in packets])
        unique_uv, index = np.unique(combined_uv, axis=0, return_index=True)
        curves[key] = {"pts": key[0], "kind": "curve_normal",
                       "physical_id": key[1], "uv": unique_uv,
                       "source_sigma_px": combined_sigma[index]}
    requested = sorted({key[0] for key in ad_rows} | {key[0] for key in curves})
    if not requested:
        raise ValueError("No fitted ad/curve observation PTS in configuration")
    ads_by_pts: dict[int, list[dict]] = {}
    curves_by_pts: dict[int, list[dict]] = {}
    for row in ad_rows.values():
        ads_by_pts.setdefault(row["pts"], []).append(row)
    for row in curves.values():
        curves_by_pts.setdefault(row["pts"], []).append(row)
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        container.seek(max(0, requested[0]-2*15360), stream=stream, backward=True)
        next_i = 0
        for frame in container.decode(stream):
            stamp = frame.pts
            if stamp is None or stamp < requested[0]:
                continue
            if next_i == len(requested):
                break
            if stamp > requested[next_i]:
                raise ValueError(f"Missing source PTS {requested[next_i]}")
            if stamp != requested[next_i]:
                continue
            image = frame.to_ndarray(format="bgr24")
            for row in ads_by_pts.get(stamp, []):
                existing = row.pop("_existing_quality")
                row["edge_quality"] = (_existing_corner_edges(
                    row["uv"], row["role"], existing) if existing is not None
                    else assess_corner_edges(image, row["uv"], row["role"]))
            for row in curves_by_pts.get(stamp, []):
                measured = assess_curve_normals(image, row["uv"])
                row["normal_quality"] = measured
            next_i += 1
        if next_i != len(requested):
            raise ValueError(f"Decoded only {next_i}/{len(requested)} fitted PTS")
    serialized_curves = []
    for row in curves.values():
        serialized_curves.append({"pts": row["pts"], "kind": row["kind"],
            "physical_id": row["physical_id"], "uv": row["uv"].tolist(),
            "source_sigma_px": row["source_sigma_px"].tolist(),
            "normal_quality": row["normal_quality"]})
    ad_list = list(ad_rows.values())
    ad_measured = sum(r["edge_quality"]["status"] == "measured_both" for r in ad_list)
    curve_list = [p for r in serialized_curves for p in r["normal_quality"]]
    curve_measured = sum(r["status"] == "measured" for r in curve_list)
    summary = {"fitted_ad_corners": len(ad_list), "ad_both_edges_measured": ad_measured,
               "ad_at_least_one_edge_measured": sum(any(x is not None for x in
                   r["edge_quality"]["sigma_floor_uv"]) for r in ad_list),
               "fitted_curve_groups": len(serialized_curves),
               "fitted_curve_points": len(curve_list),
               "curve_normal_measured": curve_measured,
               "source_pts_decoded": len(requested)}
    document = {"schema": "rinkspace-source-point-precision-v1",
        "source_sha256": source_sha256, "source_path": str(source.resolve()),
        "source_file_size": source.stat().st_size,
        "v7_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "input_sha256": inputs,
        "combination_rule": "componentwise max(existing_sigma_px, sigma_floor_uv); unknown has no floor",
        "frozen_holdout_used_for_fit": False,
        "ad_corners": ad_list, "curve_groups": serialized_curves, "summary": summary}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")),
                      encoding="utf-8")
    summary.update(path=str(output.resolve()), sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    return summary


class QualityManifest:
    """Read-only exact-PTS quality lookup for local or paired observations."""

    def __init__(self, path: str | Path, source_sha256: str):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            if str(data["schema"]) != SCHEMA:
                raise ValueError("Unknown image-quality manifest schema")
            if str(data["source_sha256"]) != source_sha256:
                raise ValueError("Image-quality source SHA-256 differs")
            self.pts = np.asarray(data["pts"], np.int64).copy()
            self.frame_multiplier = np.asarray(data["frame_sigma_multiplier"], np.float32).copy()
            self.frame_status = np.asarray(data["frame_status"], np.uint8).copy()
            self.tile_multiplier = np.asarray(data["tile_sigma_multiplier"], np.float32).copy()
            self.tile_status = np.asarray(data["tile_status"], np.uint8).copy()
        if (len(np.unique(self.pts)) != len(self.pts) or
                self.tile_multiplier.shape != (len(self.pts), GRID_Y, GRID_X) or
                np.any(self.frame_multiplier < 1) or np.any(self.tile_multiplier < 1)):
            raise ValueError("Malformed image-quality manifest")
        self._indices = {int(p): i for i, p in enumerate(self.pts)}

    def _lookup(self, pts: int, uv: np.ndarray | None,
                values: np.ndarray, fallback: float | int):
        index = self._indices.get(int(pts))
        if uv is None:
            return values[index] if index is not None else fallback
        xy = np.asarray(uv, np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("uv must have shape (N, 2)")
        output = np.full(len(xy), fallback, dtype=values.dtype)
        if index is None or not len(xy):
            return output
        valid = np.isfinite(xy).all(axis=1)
        valid &= (xy[:, 0] >= 0) & (xy[:, 0] < SOURCE_WIDTH)
        valid &= (xy[:, 1] >= 0) & (xy[:, 1] < SOURCE_HEIGHT)
        tx = np.clip((xy[valid, 0]*GRID_X/SOURCE_WIDTH).astype(int), 0, GRID_X-1)
        ty = np.clip((xy[valid, 1]*GRID_Y/SOURCE_HEIGHT).astype(int), 0, GRID_Y-1)
        output[valid] = values[index, ty, tx]
        return output

    def sigma_multiplier(self, pts: int, uv: np.ndarray | None = None):
        """Return a scalar or N-vector >= 1; unknown has multiplier 1."""
        if uv is None:
            return float(self._lookup(pts, None, self.frame_multiplier, 1.))
        return self._lookup(pts, uv, self.tile_multiplier, 1.).astype(np.float32)

    def status(self, pts: int, uv: np.ndarray | None = None):
        """Return independent clear/soft/unknown status, scalar or N-vector."""
        if uv is None:
            return str(STATUS_NAMES[int(self._lookup(pts, None, self.frame_status, 0))])
        code = self._lookup(pts, uv, self.tile_status, 0)
        return STATUS_NAMES[code]

    def pair_sigma_multiplier(self, a: int, b: int,
                              uv_a: np.ndarray | None = None,
                              uv_b: np.ndarray | None = None):
        """Combine the two endpoint pixel uncertainties in quadrature."""
        if (uv_a is None) != (uv_b is None):
            raise ValueError("Both pair endpoints need UV arrays, or neither")
        ma = self.sigma_multiplier(a, uv_a)
        mb = self.sigma_multiplier(b, uv_b)
        if np.shape(ma) != np.shape(mb):
            raise ValueError("Paired UV arrays must have equal lengths")
        return np.sqrt((np.square(ma)+np.square(mb))/2.)


def load_quality_manifest(path: str | Path, source_sha256: str) -> QualityManifest:
    return QualityManifest(path, source_sha256)


def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Build source-pixel quality manifest")
    parser.add_argument("--source", required=True)
    parser.add_argument("--shots")
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--candidate-dir")
    parser.add_argument("--precision-config")
    parser.add_argument("--extra-pts", type=int, nargs="*", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.precision_config:
        print(json.dumps(build_source_precision_manifest(args.source,
            args.precision_config, args.output,
            source_sha256=args.source_sha256), ensure_ascii=False), flush=True)
        return
    if not args.shots:
        parser.error("--shots is required for a frame quality manifest")
    shots = json.loads(Path(args.shots).read_text(encoding="utf-8"))["shots"]
    zoom = load_zoom_track(args.candidate_dir, shots) if args.candidate_dir else None
    print(json.dumps(build_quality_manifest(args.source, shots, args.output,
        source_sha256=args.source_sha256, zoom_by_pts=zoom,
        extra_pts=args.extra_pts), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    _main()
