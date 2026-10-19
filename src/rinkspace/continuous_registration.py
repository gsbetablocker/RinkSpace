"""Within-shot registration against observed keyframes, without reference resets.

Every successful edge contains fresh image correspondences. A missing image has
no map. Recovery must reobserve an existing keyframe; it cannot adopt identity.
This candidate backend deliberately has no dense ECC optimization or landmark
input. Original image values outside the fitting mask are removed before LK.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from fractions import Fraction
from typing import Any

import cv2
import numpy as np

from .geometry import project
from .local_registration import RegistrationConfig, _features, _fit


ALGORITHM_VERSION = "observed-keyframe-ice-1"


@dataclass(frozen=True)
class ContinuousConfig:
    keyframe_interval_s: float = .4
    max_keyframes: int = 10
    max_recovery_gap_s: float = .6
    forward_backward_px: float = .7
    patch_radius_px: int = 10
    pyramid_levels: int = 1
    min_patch_correlation: float = .80
    min_patch_std: float = .65
    feature_count: int = 1800


class ContinuousRegistrar:
    """Fresh, mask-isolated LK edges to a bounded set of observed keyframes."""

    def __init__(self, config: ContinuousConfig | None = None,
                 fit_config: RegistrationConfig | None = None):
        self.config = config or ContinuousConfig()
        self.fit_config = fit_config or RegistrationConfig()
        self.keys: list[dict[str, Any]] = []
        self.previous_key = None
        self.component = -1
        self.reference_sample_index = None
        self.last_time = None
        self.last_good_time = None
        self.last_h = None
        self.shape = None
        self.last_dense_fit_mask = None

    def _safe(self, gray, usable):
        # Filtering at all pyramid levels can only see fitting pixels. The
        # conservative erosion also excludes artificial mask edges as anchors.
        radius = (self.config.patch_radius_px + 3) * 2**self.config.pyramid_levels
        mask = cv2.erode(usable, np.ones((radius*2+1, radius*2+1), np.uint8))
        image = cv2.GaussianBlur(np.where(usable > 0, gray, 0).astype(np.uint8), (5,5), 1.0)
        return image, mask

    def _key(self, gray, usable, sample, h):
        isolated, safe = self._safe(gray, usable)
        points = _features(isolated, safe, self.config.feature_count)
        if len(points) < self.fit_config.min_tracks:
            return None
        return {"gray": isolated, "raw": gray, "safe": safe, "points": points,
                "target": project(h, points).astype(np.float32),
                "H_reference_from_keyframe": np.asarray(h).copy(),
                "sample_index": sample["sample_index"], "time": sample["source_time_s"]}

    @staticmethod
    def _inside(points, mask):
        h, w = mask.shape
        xy = np.rint(np.nan_to_num(points, nan=-1000, posinf=-1000, neginf=-1000)).astype(int)
        inside = np.isfinite(points).all(axis=1) & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        good = np.flatnonzero(inside)
        inside[good] &= mask[xy[good, 1], xy[good, 0]] > 0
        return inside

    def _edge(self, key, gray, isolated, safe):
        target = key["target"]
        guess = project(np.linalg.inv(self.last_h), target).astype(np.float32)
        usable = self._inside(guess, safe)
        x, target, guess = key["points"][usable], target[usable], guess[usable]
        if len(x) < self.fit_config.min_tracks:
            return {"status": "unavailable", "reason": "no_visible_keyframe_support"}
        radius = self.config.patch_radius_px
        params = dict(winSize=(radius*2+1, radius*2+1), maxLevel=self.config.pyramid_levels,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .005), minEigThreshold=1e-6)
        y, sf, _ = cv2.calcOpticalFlowPyrLK(key["gray"], isolated, x[:, None], guess[:, None].copy(),
                                         flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **params)
        z, sb, _ = cv2.calcOpticalFlowPyrLK(isolated, key["gray"], y, x[:, None].copy(),
                                         flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **params)
        y, z = y[:, 0], z[:, 0]
        ok = (sf[:, 0] > 0) & (sb[:, 0] > 0) & self._inside(y, safe)
        ok &= np.linalg.norm(z-x, axis=1) < self.config.forward_backward_px
        # The local raw patch check rejects converged flow on featureless ice.
        correlations = np.zeros(len(x), np.float32)
        for i in np.flatnonzero(ok):
            a = cv2.getRectSubPix(key["raw"], (15, 15), tuple(map(float, x[i]))).astype(np.float32)
            b = cv2.getRectSubPix(gray, (15, 15), tuple(map(float, y[i]))).astype(np.float32)
            if min(float(a.std()), float(b.std())) < self.config.min_patch_std:
                ok[i] = False
                continue
            a -= a.mean(); b -= b.mean()
            correlations[i] = np.sum(a*b) / max(float(np.linalg.norm(a)*np.linalg.norm(b)), 1e-8)
        ok &= correlations >= self.config.min_patch_correlation
        # RANSAC uses observed keyframe pixels, whose error scale stays fixed;
        # a changing projective reference must not change the inlier threshold.
        fit = _fit(y[ok], x[ok], gray.shape, self.fit_config)
        fit["edge_keyframe_sample_index"] = key["sample_index"]
        fit["edge_keyframe_time_s"] = key["time"]
        fit["photometric_patch_correlation_median"] = float(np.median(correlations[ok])) if np.any(ok) else None
        fit["correspondence_count_before_ransac"] = int(ok.sum())
        if fit["status"] == "candidate":
            flags = np.asarray(fit["inlier_flags"], bool)
            fit["fit_keyframe_xy"] = x[ok][flags].tolist()
            fit["H_keyframe_from_current"] = fit["H_reference_from_current"]
            combined = key["H_reference_from_keyframe"] @ np.asarray(fit["H_keyframe_from_current"])
            fit["H_reference_from_current"] = (combined/combined[2,2]).tolist()
            fit["fit_reference_xy"] = target[ok][flags].tolist()
            fit["fitting_residual_coordinate_space"] = "observed_keyframe_pixels"
            fit["pair_algorithm"] = "observed_keyframe_mask_isolated_LK+FB+patch_correlation+RANSAC"
        return fit

    def push(self, image, sample, masks, force_break=False):
        time = float(int(sample["pts"]) * Fraction(sample["time_base"]))
        if abs(time - sample["source_time_s"]) > 1e-5:
            raise ValueError("source_time_s disagrees with source PTS/time_base")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gap = None if self.last_time is None else time-self.last_time
        broken = force_break or (gap is not None and (gap <= 0 or gap > self.config.max_recovery_gap_s)) or (self.shape is not None and gray.shape != self.shape)
        if broken:
            self.keys = []; self.previous_key = None; self.last_h = None; self.last_good_time = None
        result = {**sample, "status": "unavailable", "H_reference_from_current": None,
                  "matrix_direction": "H_reference_from_current", "geometry_level": "local_projective",
                  "unit": "reference_px", "independent_validation": False, "map_usable_for_metric_output": False,
                  "graph_component_id": None, "support_polygon": [], "fit_current_xy": [], "fit_reference_xy": [],
                  "mask_status": masks["mask_status"], "ice_fraction": masks["ice_fraction"], "usable_fraction": masks["usable_fraction"],
                  "reason": "insufficient_observed_keyframe_correspondences", "dense_optimization_used": False}
        if broken:
            result["break_reason"] = "declared_shot_boundary" if force_break else "pts_gap_or_resolution_change"
        if not self.keys:
            key = self._key(gray, masks["usable"], sample, np.eye(3))
            fit = _fit(key["points"], key["points"], gray.shape, self.fit_config) if key else {"status": "unavailable"}
            if fit["status"] == "candidate":
                self.component += 1; self.reference_sample_index = sample["sample_index"]
                self.keys = [key]; self.previous_key = key; self.last_h = np.eye(3); self.last_good_time = time
                result.update(fit, reference_frame_only=True, reason="first_observed_reference_only")
        elif time-self.last_good_time <= self.config.max_recovery_gap_s:
            isolated, safe = self._safe(gray, masks["usable"])
            # Oldest surviving anchor first limits accumulated drift. Try the
            # newest as recovery when the camera has left the older footprint.
            attempts = []
            keys = self.keys + ([self.previous_key] if self.previous_key is not None and self.previous_key is not self.keys[-1] else [])
            candidates = []
            for key in keys:
                edge = self._edge(key, gray, isolated, safe)
                attempts.append({"sample_index": key["sample_index"], "status": edge["status"], "reason": edge.get("reason")})
                if edge["status"] == "candidate":
                    candidates.append(edge)
            if candidates:
                best = max(candidates, key=lambda edge: edge['train_inliers'] * edge['support_fraction']**.5)
                # Prefer a longer observed baseline only when it retains most
                # of the fresh edge's spatial support and agrees geometrically.
                probes = np.asarray(best['support_polygon'])
                for edge in candidates:
                    if edge['train_inliers'] < best['train_inliers']*.75 or edge['support_fraction'] < best['support_fraction']*.75:
                        continue
                    difference = np.median(np.linalg.norm(project(edge['H_reference_from_current'],probes)-project(best['H_reference_from_current'],probes),axis=1))
                    if difference < 2 and edge['edge_keyframe_sample_index'] < best['edge_keyframe_sample_index']:
                        best = edge
                result.update(best)
            result["keyframe_attempts"] = attempts
            if result["status"] == "candidate":
                result["recovered_after_missing"] = self.last_time != self.last_good_time
                self.last_h = np.asarray(result["H_reference_from_current"])
                self.last_good_time = time
                self.previous_key = self._key(gray, masks["usable"], sample, self.last_h)
                if time-self.keys[-1]["time"] >= self.config.keyframe_interval_s-1e-6:
                    key = self.previous_key
                    if key is not None:
                        self.keys.append(key)
                        if len(self.keys) > self.config.max_keyframes:
                            self.keys.pop(1)  # retain the original observed reference
        else:
            result["reason"] = "recovery_gap_exceeded_requires_explicit_new_component"
        if result["status"] == "candidate":
            result["graph_component_id"] = f"local_{self.component:03d}"
            result["reference_sample_index"] = self.reference_sample_index
        result.pop("inlier_flags", None)
        result["fitting_patch_policy"] = {"input_mask_isolation": True, "dense_refinement": False,
                                           "center_erosion_px": (self.config.patch_radius_px+3)*2**self.config.pyramid_levels}
        self.last_time = time; self.shape = gray.shape
        return result
