"""Reusable, PTS-preserving, candidate local ice-plane registration.

No venue, source filename, time window, skater rectangle or landmark is built in.
The automatic mask is an explicitly heuristic *candidate* segmentation.  Maps
remain local_projective until a separate physical-landmark audit says otherwise.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry import project
from .io import dump_json


ALGORITHM_VERSION = "local-ice-tracks-1"


@dataclass(frozen=True)
class RegistrationConfig:
    max_features: int = 1800
    min_tracks: int = 18
    min_inliers: int = 14
    min_support_fraction: float = 0.025
    min_support_width_fraction: float = 0.18
    min_support_height_fraction: float = 0.12
    ransac_threshold_px: float = 1.7
    max_inlier_p95_px: float = 2.0
    max_pair_displacement_fraction: float = 0.30
    max_sample_gap_s: float = 0.6
    forward_backward_px: float = 0.8
    replenish_below: int = 140
    direct_anchor_every: int = 10
    dense_anchor_every: int = 1
    max_canvas_pixels: int = 1800000

    def __post_init__(self) -> None:
        if any(value <= 0 for value in asdict(self).values()):
            raise ValueError("All registration limits must be positive")
        if self.min_inliers < 4 or self.min_tracks < self.min_inliers:
            raise ValueError("Require min_tracks >= min_inliers >= 4")


def _regions_mask(shape: tuple[int, int], regions: list[dict[str, Any]]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    for region in regions:
        if "polygon" in region:
            polygon = np.asarray(region["polygon"], dtype=float)
            if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
                raise ValueError("Protected polygon needs at least three finite image-pixel vertices")
            cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255)
        elif "xy" in region:
            xy = np.asarray(region["xy"], float)
            radius = float(region.get("radius_px", 20))
            if xy.shape != (2,) or not np.isfinite(xy).all() or not np.isfinite(radius) or radius <= 0:
                raise ValueError("Protected circle requires finite xy and positive radius_px")
            cv2.circle(mask, tuple(np.rint(xy).astype(int)), int(np.ceil(radius)), 255, -1)
        else:
            raise ValueError("Region requires polygon or xy/radius_px")
    return mask


def ice_mask(image: np.ndarray, corrections: dict[str, Any] | None = None,
             protected_regions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build per-frame ice/foreground/protected masks with inspectable provenance.

    Corrections may contain include_polygons and exclude_polygons in decoded
    sample image pixels.  Protection is applied last, so a correction cannot
    silently reintroduce independently reviewed landmarks into fitting.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected uint8 BGR image")
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, value = cv2.split(hsv)
    kernel_size = max(3, int(round(width / 240)) | 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    white = ((sat < 75) & (value > 130)).astype(np.uint8) * 255
    # Fill logos and scratches inside the ice region; their local variation is
    # the useful geometric signal.  Erode the uncertain outer boundary.
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    ice = np.zeros((height, width), np.uint8)
    if count > 1:
        scores = stats[1:, cv2.CC_STAT_AREA].astype(float)
        selected = int(np.argmax(scores)) + 1
        if stats[selected, cv2.CC_STAT_AREA] >= height * width * .12:
            component = (labels == selected).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ice, contours, -1, 255, -1)
            ice = cv2.erode(ice, kernel, iterations=2)

    # A yellow kickplate is a frequent broadcast rink boundary, but is used
    # only when a long, coherent candidate exists.  No assumed rink dimensions.
    yellow = ((hue >= 17) & (hue <= 42) & (sat > 65) & (value > 100)).astype(np.uint8) * 255
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, np.ones((3, max(9, width // 50)), np.uint8))
    contours, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary_candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > width * .30 and h < height * .25 and y < height * .80:
            boundary_candidates.append((w, contour))
    boundary = None
    if boundary_candidates:
        contour = max(boundary_candidates, key=lambda entry: entry[0])[1]
        band = np.zeros_like(ice)
        cv2.drawContours(band, [contour], -1, 255, -1)
        xs = np.flatnonzero(np.any(band, axis=0))
        ys = np.array([np.flatnonzero(band[:, x])[-1] for x in xs])
        slope, intercept = np.polyfit(xs, ys, 1)
        fitted = slope * xs + intercept
        if np.percentile(np.abs(fitted - ys), 90) < max(6, height * .035):
            boundary = np.clip(slope * np.arange(width) + intercept + kernel_size * 2, 0, height)
            ice[np.arange(height)[:, None] < boundary[None, :]] = 0

    # Washed-out broadcasts can make both glass and white boards join the
    # bright ice component.  The bottom edge of a broad populated/sponsor band
    # is a conservative second boundary observation.  A single skater does
    # not occupy the required fraction of a scanline.
    bright_values = value[white > 0]
    ice_brightness = float(np.percentile(bright_values, 70)) if len(bright_values) else 200.
    busy = (value < ice_brightness * .73) | ((sat > 40) & (value < ice_brightness * .98))
    occupancy = np.mean(busy, axis=1).astype(np.float32)
    occupancy = cv2.blur(occupancy[:, None], (1, max(3, height // 90)))[:, 0]
    populated_rows = np.flatnonzero((occupancy > .30) & (np.arange(height) < height * .73))
    populated_boundary = None
    if len(populated_rows):
        populated_boundary = int(populated_rows[-1] + kernel_size * 2)
        ice[:populated_boundary] = 0

    # Foreground candidates are coherent dark/saturated upright objects.
    # Broader moving-object residual checks are handled by the robust tracker.
    colored = (((sat > 55) & (value < 225)) | (value < max(94., ice_brightness * .73))).astype(np.uint8) * 255
    colored = cv2.bitwise_and(colored, ice)
    colored = cv2.morphologyEx(colored, cv2.MORPH_CLOSE, kernel, iterations=2)
    foreground = np.zeros_like(ice)
    count, _, stats, _ = cv2.connectedComponentsWithStats(colored)
    boxes = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        upright = (h > height * .045 and .12 < w / max(h, 1) < 3.8
                   and area > height * width * .00035 and w < width * .55)
        border_text = ((y > height * .86 or y+h < height * .17) and h > height * .012
                       and w > width * .035 and w/h > 1.6 and h < height * .08)
        if upright or border_text:
            pad_x, pad_y = max(5, width // 100, round(w * .15)), max(5, height // 70, round(h * .26))
            boundary_at_object = max(populated_boundary or 0, float(boundary[min(width-1, x+w//2)]) if boundary is not None else 0)
            if upright and y < boundary_at_object + kernel_size * 3:
                # The upper body can lie above the ice boundary and therefore
                # be clipped out of these seeds. Extend the remaining clothing
                # component conservatively to cover pale legs/white skates.
                pad_y = max(pad_y, round(w * .85))
            rect = [max(0, x-pad_x), max(0, y-pad_y), min(width, x+w+pad_x), min(height, y+h+pad_y)]
            foreground[rect[1]:rect[3], rect[0]:rect[2]] = 255
            boxes.append(rect)
    usable = cv2.bitwise_and(ice, cv2.bitwise_not(foreground))
    correction = corrections or {}
    if isinstance(correction, list):
        correction = {"include_polygons": [item["polygon"] for item in correction if item.get("kind") == "include" and "polygon" in item],
                      "exclude_polygons": [item["polygon"] for item in correction if item.get("kind", "exclude") == "exclude" and "polygon" in item]}
    for poly in correction.get("include_polygons", []):
        usable[_regions_mask((height, width), [{"polygon": poly}]) > 0] = 255
    for poly in correction.get("exclude_polygons", []):
        usable[_regions_mask((height, width), [{"polygon": poly}]) > 0] = 0
    protected = _regions_mask((height, width), protected_regions or [])
    # Detector/LK patch support must also stay clear of the protected footprint.
    # 21-pixel LK windows with two pyramid levels have at most a 40-pixel
    # half-footprint, plus filtering/interpolation support.  Keep the original
    # independently observed footprint outside that full support, not merely
    # outside the fitted feature centres.
    protection_margin = 48
    protected = cv2.dilate(protected, np.ones((2*protection_margin+1, 2*protection_margin+1), np.uint8))
    usable[protected > 0] = 0
    border = max(4, width // 160)
    usable[:border] = 0; usable[-border:] = 0
    usable[:, :border] = 0; usable[:, -border:] = 0
    return {"usable": usable, "ice": ice, "foreground": foreground, "protected": protected,
            "foreground_boxes": boxes, "ice_fraction": float(np.mean(ice > 0)),
            "usable_fraction": float(np.mean(usable > 0)), "boundary_detected": boundary is not None or populated_boundary is not None,
            "populated_band_boundary_y": populated_boundary,
            "mask_status": "heuristic_candidate_requires_visual_review",
            "method": "bright-connected-ice+observed-kickplate+foreground-components+review-exclusions"}


def _fit(current_xy: np.ndarray, target_xy: np.ndarray, shape: tuple[int, int],
         config: RegistrationConfig) -> dict[str, Any]:
    height, width = shape
    base: dict[str, Any] = {"status": "unavailable", "matches": len(current_xy)}
    if len(current_xy) < config.min_tracks:
        return {**base, "reason": "insufficient_consistent_ice_tracks"}
    cv2.setRNGSeed(20260919)
    h, flags = cv2.findHomography(current_xy, target_xy, cv2.RANSAC,
                                 config.ransac_threshold_px, maxIters=3000, confidence=.999)
    if h is None or flags is None or not np.isfinite(h).all():
        return {**base, "reason": "homography_fit_failed"}
    inliers = flags.ravel() > 0
    points = current_xy[inliers]
    if len(points) < config.min_inliers or np.mean(inliers) < .45:
        return {**base, "reason": "too_few_ice_plane_inliers", "train_inliers": int(inliers.sum())}
    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)
    area = float(cv2.contourArea(hull) / (height * width))
    span = np.ptp(points, axis=0)
    if (area < config.min_support_fraction or span[0] < width * config.min_support_width_fraction
            or span[1] < height * config.min_support_height_fraction):
        return {**base, "reason": "insufficient_spatial_support", "support_fraction": area}
    try:
        error = np.linalg.norm(project(h, points) - target_xy[inliers], axis=1)
        corners = np.array([[0., 0.], [width, 0.], [width, height], [0., height]])
        projected = project(h, corners)
        denominators = np.c_[corners, np.ones(4)] @ h[2, :]
        if np.min(denominators) * np.max(denominators) <= 0:
            return {**base, "reason": "projective_horizon_inside_image"}
        if not cv2.isContourConvex(projected.astype(np.float32)):
            return {**base, "reason": "folded_frame_projection"}
        oriented_area = float(cv2.contourArea(projected.astype(np.float32), oriented=True))
        if oriented_area <= 0 or not .12 < oriented_area / (width * height) < 8:
            return {**base, "reason": "implausible_frame_area_or_orientation"}
    except (ValueError, np.linalg.LinAlgError):
        return {**base, "reason": "degenerate_projection"}
    if np.percentile(error, 95) > config.max_inlier_p95_px:
        return {**base, "reason": "large_fitting_residual"}
    h = h / h[2, 2]
    return {"status": "candidate", "reason": "independent_landmark_validation_pending",
            "H_reference_from_current": h.tolist(), "train_inliers": int(inliers.sum()),
            "matches": len(current_xy), "support_polygon": hull.tolist(),
            "support_fraction": area, "inlier_median_reference_px": float(np.median(error)),
            "inlier_p95_reference_px": float(np.percentile(error, 95)),
            "fit_current_xy": points.tolist(), "fit_reference_xy": target_xy[inliers].tolist(),
            "inlier_flags": inliers.tolist()}


def _features(gray: np.ndarray, mask: np.ndarray, count: int) -> np.ndarray:
    # A single remaining high-contrast pixel at a skater/board boundary must
    # not set the quality threshold for an otherwise very pale ice sheet.
    # Spatial allocation also prevents the largest logo monopolising fitting.
    height, width = gray.shape
    points = []
    rows, cols = 4, 6
    per_cell = max(4, int(np.ceil(count / (rows*cols))))
    for row in range(rows):
        y0, y1 = row*height//rows, (row+1)*height//rows
        for col in range(cols):
            x0, x1 = col*width//cols, (col+1)*width//cols
            local = cv2.goodFeaturesToTrack(gray[y0:y1, x0:x1], per_cell, .003,
                                           max(6, width // 120), mask=mask[y0:y1, x0:x1], blockSize=7)
            if local is not None:
                points.append(local[:, 0] + [x0, y0])
    return np.concatenate(points).astype(np.float32)[:count] if points else np.empty((0, 2), np.float32)


def _sift_matches(reference: np.ndarray, current: np.ndarray, ref_mask: np.ndarray,
                  cur_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    detector = cv2.SIFT_create(nfeatures=2600, contrastThreshold=.012, edgeThreshold=12)
    kr, dr = detector.detectAndCompute(reference, ref_mask)
    kc, dc = detector.detectAndCompute(current, cur_mask)
    empty = np.empty((0, 2), np.float32)
    if dr is None or dc is None or min(len(dr), len(dc)) < 2:
        return empty, empty
    def isolated(keypoints, descriptors, mask):
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        good = [i for i, keypoint in enumerate(keypoints)
                if distance[int(round(keypoint.pt[1])), int(round(keypoint.pt[0]))] > 6 * keypoint.size]
        return [keypoints[i] for i in good], descriptors[good]
    kr, dr = isolated(kr, dr, ref_mask)
    kc, dc = isolated(kc, dc, cur_mask)
    if min(len(dr), len(dc)) < 2:
        return empty, empty
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward, backward = matcher.knnMatch(dc, dr, k=2), matcher.knnMatch(dr, dc, k=2)
    reverse = {m.queryIdx: m.trainIdx for pair in backward if len(pair) == 2
               for m, n in [pair] if m.distance < .72 * n.distance}
    good = [m for pair in forward if len(pair) == 2 for m, n in [pair]
            if m.distance < .72 * n.distance and reverse.get(m.trainIdx) == m.queryIdx]
    return np.array([kc[m.queryIdx].pt for m in good], np.float32).reshape(-1, 2), \
        np.array([kr[m.trainIdx].pt for m in good], np.float32).reshape(-1, 2)


class LocalRegistrar:
    """Streaming short-shot tracker. Unknown frames never reuse the last H."""

    def __init__(self, config: RegistrationConfig | None = None):
        self.config = config or RegistrationConfig()
        self.component = -1
        self.previous_gray: np.ndarray | None = None
        self.previous_time: float | None = None
        self.previous_h = np.eye(3)
        self.tracks = np.empty((0, 2), np.float32)
        self.reference_tracks = np.empty((0, 2), np.float32)
        self.reference_gray: np.ndarray | None = None
        self.reference_mask: np.ndarray | None = None
        self.reference_sample_index: int | None = None
        self.frame_counter = 0
        self.last_dense_fit_mask: np.ndarray | None = None

    def _start(self, gray: np.ndarray, mask: np.ndarray, sample_index: int) -> bool:
        candidates = _features(gray, mask, self.config.max_features)
        fit = _fit(candidates, candidates, gray.shape, self.config)
        if fit["status"] != "candidate":
            self.tracks = np.empty((0, 2), np.float32)
            return False
        self.component += 1
        self.tracks = candidates
        self.reference_tracks = candidates.copy()
        self.reference_gray, self.reference_mask = gray.copy(), mask.copy()
        self.previous_h = np.eye(3)
        self.reference_sample_index = sample_index
        self.frame_counter = 0
        return True

    def push(self, image: np.ndarray, sample: dict[str, Any], masks: dict[str, Any],
             force_break: bool = False) -> dict[str, Any]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.last_dense_fit_mask = None
        mask = masks["usable"]
        time = float(int(sample["pts"]) * Fraction(sample["time_base"]))
        if abs(time - float(sample["source_time_s"])) > 1e-5:
            raise ValueError("source_time_s disagrees with source PTS/time_base")
        result: dict[str, Any] = {**sample, "status": "unavailable", "H_reference_from_current": None,
            "matrix_direction": "H_reference_from_current", "geometry_level": "local_projective",
            "unit": "reference_px", "independent_validation": False, "map_usable_for_metric_output": False,
            "graph_component_id": None, "support_polygon": [], "fit_current_xy": [],
            "fit_reference_xy": [], "mask_status": masks["mask_status"],
            "ice_fraction": masks["ice_fraction"], "usable_fraction": masks["usable_fraction"]}
        gap = None if self.previous_time is None else time - self.previous_time
        broken = force_break or (gap is not None and (gap <= 0 or gap > self.config.max_sample_gap_s))
        break_reason = "declared_shot_boundary" if force_break else "nonmonotonic_or_large_pts_gap"
        if self.previous_gray is not None and self.previous_gray.shape != gray.shape:
            broken, break_reason = True, "sample_resolution_changed"
        if broken:
            self.tracks = np.empty((0, 2), np.float32)
            result["break_reason"] = break_reason
        if not len(self.tracks):
            if self._start(gray, mask, int(sample["sample_index"])):
                result.update(_fit(self.tracks, self.reference_tracks, gray.shape, self.config))
                result["reason"] = "new_local_reference_requires_independent_validation"
                result["reference_frame_only"] = True
            else:
                result["reason"] = "low_texture_or_insufficient_ice_support"
        else:
            x = self.tracks.reshape(-1, 1, 2)
            y, sf, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, x, None, winSize=(21, 21), maxLevel=2)
            z, sb, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, y, None, winSize=(21, 21), maxLevel=2)
            current = y[:, 0]
            height, width = gray.shape
            finite = np.isfinite(current).all(axis=1) & np.isfinite(z[:, 0]).all(axis=1)
            safe = np.nan_to_num(current, nan=-1000, posinf=-1000, neginf=-1000)
            ij = np.rint(safe).astype(int)
            ij[:, 0] = np.clip(ij[:, 0], 0, width-1); ij[:, 1] = np.clip(ij[:, 1], 0, height-1)
            ok = (finite & (sf[:, 0] > 0) & (sb[:, 0] > 0)
                  & (np.linalg.norm(z[:, 0]-self.tracks, axis=1) < self.config.forward_backward_px)
                  & (current[:, 0] >= 0) & (current[:, 0] < width)
                  & (current[:, 1] >= 0) & (current[:, 1] < height)
                  & (mask[ij[:, 1], ij[:, 0]] > 0))
            current, target = current[ok], self.reference_tracks[ok]
            fit = _fit(current, target, gray.shape, self.config)
            fit["pair_algorithm"] = "persistent_reference_tracks+forward_backward_LK+RANSAC"
            if fit["status"] == "candidate":
                pair_h = np.linalg.inv(self.previous_h) @ np.asarray(fit["H_reference_from_current"])
                corners = np.array([[0., 0.], [width, 0.], [width, height], [0., height]])
                displacement = np.max(np.linalg.norm(project(pair_h, corners)-corners, axis=1))
                if displacement > np.hypot(width, height) * self.config.max_pair_displacement_fraction:
                    fit = {"status": "unavailable", "reason": "camera_cut_or_implausible_interframe_motion"}
            if fit["status"] == "candidate" and self.frame_counter % self.config.direct_anchor_every == 0:
                ac, ar = _sift_matches(self.reference_gray, gray, self.reference_mask, mask)
                direct = _fit(ac, ar, gray.shape, self.config)
                if direct["status"] == "candidate":
                    # A direct match is accepted only with spatial support and
                    # agreement with the short-baseline track estimate.
                    probes = np.asarray(fit["support_polygon"])
                    disagreement = np.median(np.linalg.norm(project(direct["H_reference_from_current"], probes)
                                                            - project(fit["H_reference_from_current"], probes), axis=1))
                    fit["direct_anchor_disagreement_px"] = float(disagreement)
                    if disagreement < 5 and direct["support_fraction"] >= fit["support_fraction"] * .55:
                        fit = {**direct, "pair_algorithm": "direct_mutual_SIFT_to_shot_reference",
                               "direct_anchor_disagreement_px": float(disagreement)}
                        current, target = ac, ar
            if fit["status"] == "candidate":
                if self.frame_counter % self.config.dense_anchor_every == 0:
                    initial = np.asarray(fit["H_reference_from_current"])
                    inverse = np.linalg.inv(initial)
                    reference_visible = cv2.warpPerspective(self.reference_mask, inverse, (width, height), flags=cv2.INTER_NEAREST)
                    dense_mask = cv2.bitwise_and(mask, reference_visible)
                    if np.mean(dense_mask > 0) > .12:
                        try:
                            score, optimized = cv2.findTransformECC(self.reference_gray, gray, inverse.astype(np.float32),
                                cv2.MOTION_HOMOGRAPHY, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-6), dense_mask, 5)
                            optimized = np.asarray(optimized, np.float64)
                            anchored = np.linalg.inv(optimized)
                            probes = np.asarray(fit["support_polygon"])
                            correction = float(np.median(np.linalg.norm(project(anchored, probes) - project(initial, probes), axis=1)))
                            maximum_correction = float(np.max(np.linalg.norm(project(anchored, corners) - project(initial, corners), axis=1)))
                            denominators = np.c_[corners, np.ones(4)] @ anchored[2, :]
                            transformed = project(anchored, corners)
                            plausible = (np.min(denominators)*np.max(denominators) > 0 and
                                         cv2.isContourConvex(transformed.astype(np.float32)) and
                                         cv2.contourArea(transformed.astype(np.float32), oriented=True) > width*height*.12)
                            fit["dense_anchor_correlation"] = float(score)
                            fit["dense_anchor_correction_reference_px"] = correction
                            fit["dense_anchor_max_corner_correction_reference_px"] = maximum_correction
                            if score >= .90 and correction < width * .04 and maximum_correction < 40 and plausible:
                                fit["H_reference_from_current"] = (anchored/anchored[2,2]).tolist()
                                fit["pair_algorithm"] += "+direct_masked_sample_resolution_ECC_to_reference"
                                fit["dense_fit_pixel_count"] = int(np.count_nonzero(dense_mask))
                                fit["dense_anchor_status"] = "candidate_photometric_alignment_not_independent_validation"
                                self.last_dense_fit_mask = dense_mask
                                # Anchor the next tracking step to this dense
                                # image-only solution, never to check landmarks.
                                target = project(anchored, current).astype(np.float32)
                                flags = np.asarray(fit["inlier_flags"], bool)
                                fit["fit_reference_xy"] = target[flags].tolist()
                                fit["sparse_residual_before_dense_refinement"] = True
                            else:
                                fit["dense_anchor_status"] = "rejected_weak_or_excessive_photometric_correction"
                        except (cv2.error, ValueError, np.linalg.LinAlgError):
                            fit["dense_anchor_status"] = "photometric_alignment_did_not_converge"
                result.update(fit)
                flags = np.asarray(fit.pop("inlier_flags", []), bool)
                # Keep only plane-consistent tracks to avoid an occluding
                # athlete becoming a long-lived anchor.
                self.tracks, self.reference_tracks = current[flags], target[flags]
                self.previous_h = np.asarray(fit["H_reference_from_current"])
                if len(self.tracks) < self.config.replenish_below:
                    additional_mask = mask.copy()
                    for xy in self.tracks:
                        cv2.circle(additional_mask, tuple(np.rint(xy).astype(int)), 12, 0, -1)
                    extra = _features(gray, additional_mask, self.config.max_features - len(self.tracks))
                    if len(extra):
                        self.tracks = np.concatenate((self.tracks, extra))
                        self.reference_tracks = np.concatenate((self.reference_tracks,
                                                               project(self.previous_h, extra).astype(np.float32)))
                    result["replenished_features"] = len(extra)
                    result["replenishment_note"] = "new_feature_reference_positions_inherit_candidate_map"
            else:
                result.update(fit)
                result["break_reason"] = fit["reason"]
                self.tracks = np.empty((0, 2), np.float32)
        if result["status"] == "candidate":
            result["graph_component_id"] = f"local_{self.component:03d}"
            result["reference_sample_index"] = self.reference_sample_index
        result.pop("inlier_flags", None)
        self.previous_gray, self.previous_time = gray, time
        self.frame_counter += 1
        return result


def register_sequence(sample_dir: str | Path, out_dir: str | Path,
                      config: RegistrationConfig | dict[str, Any] | None = None,
                      exclusions: dict[str, list[dict[str, Any]]] | None = None,
                      corrections: dict[str, dict[str, Any]] | None = None,
                      shot_breaks: list[int] | None = None,
                      render: bool = True) -> dict[str, Any]:
    """Process a decoded sample manifest with original PTS and source identity.

    exclusions keys are sample_index strings (or '*' for frame-invariant
    regions). Corrections follow the same convention. No interpolation of
    hidden landmarks is performed. Every protected footprint is saved.
    """
    source, output = Path(sample_dir), Path(out_dir)
    record = json.loads((source / "samples.json").read_text(encoding="utf-8"))
    if isinstance(config, dict):
        config = dict(config)
        corrections = config.pop("corrections", corrections)
        exclusions = config.pop("exclusions", exclusions)
        params = RegistrationConfig(**config)
    else:
        params = config or RegistrationConfig()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("masks", "protected", "foreground", "fixed", "overlays", "dense_masks"):
        (output / name).mkdir(exist_ok=True)
    tracker = LocalRegistrar(params)
    frames = []
    dimensions = None
    for sample in record["samples"]:
        image = cv2.imread(str(source / sample["image"]))
        if image is None:
            raise ValueError(f"Cannot decode sample {source / sample['image']}")
        dimensions = [image.shape[1], image.shape[0]]
        key = str(sample["sample_index"])
        protected = (exclusions or {}).get("*", []) + (exclusions or {}).get(key, [])
        correction = (corrections or {}).get(key, (corrections or {}).get("*", {}))
        masks = ice_mask(image, correction, protected)
        row = tracker.push(image, sample, masks, int(sample["sample_index"]) in (shot_breaks or []))
        if tracker.last_dense_fit_mask is not None:
            path = f"dense_masks/{int(sample['sample_index']):06d}.png"
            cv2.imwrite(str(output / path), tracker.last_dense_fit_mask)
            row["dense_fit_mask_path"] = path
        row["protected_regions"] = protected
        row["mask_correction"] = correction
        for name, value in (("masks", masks["usable"]), ("protected", masks["protected"]), ("foreground", masks["foreground"])):
            path = f"{name}/{int(sample['sample_index']):06d}.png"
            cv2.imwrite(str(output / path), value)
            row[{"masks": "mask_path", "protected": "protected_mask_path", "foreground": "foreground_mask_path"}[name]] = path
        overlay = image.copy()
        overlay[masks["usable"] == 0] = (overlay[masks["usable"] == 0] * .36).astype(np.uint8)
        overlay[masks["protected"] > 0] = (overlay[masks["protected"] > 0] * .4 + np.array([130, 0, 130])).clip(0,255).astype(np.uint8)
        if row["support_polygon"]:
            cv2.polylines(overlay, [np.rint(row["support_polygon"]).astype(np.int32)], True, (0, 220, 80), 2)
        cv2.putText(overlay, f"PTS {sample['pts']} | {sample['source_time_s']:.3f}s | {row['status']}",
                    (14, 26), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2, cv2.LINE_AA)
        overlay_path = f"overlays/{int(sample['sample_index']):06d}.jpg"
        cv2.imwrite(str(output / overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 87])
        row["overlay_path"] = overlay_path
        row["mask_image"] = overlay_path
        frames.append(row)
    digest = hashlib.sha256(json.dumps({"config": asdict(params), "exclusions": exclusions,
                                       "corrections": corrections, "algorithm": ALGORITHM_VERSION}, sort_keys=True).encode()).hexdigest()
    result = {"schema_version": "1.0", "algorithm": ALGORITHM_VERSION, "configuration_sha256": digest,
              "algorithm_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_sha256": record.get("source_sha256"), "source_filename": record.get("source_filename"),
              "sample_directory": str(source.resolve()), "decoded_dimensions": dimensions,
              "config": asdict(params), "coordinate_space": "local_projective", "unit": "reference_px",
              "independent_validation": False, "map_usable_for_metric_output": False,
              "quality_note": "RANSAC/flow agreement is a fitting diagnostic, not independent validation.",
              "frames": frames, "summary": {"sample_count": len(frames),
                  "candidate_count": sum(row["status"] == "candidate" for row in frames),
                  "unavailable_count": sum(row["status"] != "candidate" for row in frames),
                  "component_count": len({row["graph_component_id"] for row in frames if row["graph_component_id"] is not None})}}
    if render:
        result["canvases"] = render_fixed_frames(source, output, result, params.max_canvas_pixels)
    dump_json(output / "registration.json", result)
    return result


def render_fixed_frames(sample_dir: Path, out_dir: Path, registration: dict[str, Any],
                        max_pixels: int = 1800000) -> dict[str, Any]:
    """One fixed canvas per disconnected component, with unobserved alpha=0.

    This is a transformed single observed frame, never a temporally filled
    mosaic. Bounds include every image corner in that component, preserving
    skater displacement instead of choosing a per-frame crop.
    """
    width, height = registration["decoded_dimensions"]
    corners = np.array([[0.,0.], [width,0.], [width,height], [0.,height]])
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in registration["frames"]:
        if row["status"] == "candidate":
            groups.setdefault(row["graph_component_id"], []).append(row)
    canvases = {}
    for component, rows in groups.items():
        all_points = np.concatenate([project(row["H_reference_from_current"], corners) for row in rows])
        low, high = np.floor(all_points.min(axis=0))-8, np.ceil(all_points.max(axis=0))+8
        span = high-low
        scale = min(1., np.sqrt(max_pixels/max(float(span.prod()), 1.)))
        size = np.maximum(2, np.ceil(span*scale).astype(int))
        display = np.array([[scale,0,-low[0]*scale], [0,scale,-low[1]*scale], [0,0,1.]])
        canvases[component] = {"H_display_from_reference": display.tolist(), "width": int(size[0]),
                               "height": int(size[1]), "reference_bounds": [low.tolist(), high.tolist()],
                               "policy": "fixed_union_of_all_observed_frame_footprints_in_component"}
        for row in rows:
            image = cv2.imread(str(sample_dir / row["image"]))
            matrix = display @ np.asarray(row["H_reference_from_current"])
            warped = cv2.warpPerspective(image, matrix, tuple(size), flags=cv2.INTER_LINEAR)
            alpha = cv2.warpPerspective(np.full((height,width),255,np.uint8), matrix, tuple(size), flags=cv2.INTER_NEAREST)
            bgra = np.dstack((warped, alpha))
            # PNG preserves exactly which part of this source frame was seen.
            path = f"fixed/{int(row['sample_index']):06d}.png"
            cv2.imwrite(str(out_dir / path), bgra)
            row["fixed_path"] = path
            row["fixed_image"] = path
            row["H_display_from_current"] = matrix.tolist()
    return canvases
