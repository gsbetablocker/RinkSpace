"""Continuous candidate maps from direct, mask-isolated observed-keyframe fits.

Missing frames never get an identity map. A source gap greater than the recovery
limit requires an explicit caller-declared break before a new component starts.
The dense fit is local to the actual observed keyframe's pixels; only afterwards
is its H composed into the shot reference. Photometric samples are not landmarks.
"""
from __future__ import annotations

from fractions import Fraction
from dataclasses import dataclass

import cv2
import numpy as np

from .continuous_registration import ContinuousRegistrar, ContinuousConfig
from .geometry import project
from .masked_photometric import estimate_photometric


ALGORITHM_VERSION = "masked-photometric-observed-keyframes-2"


@dataclass(frozen=True)
class PhotometricConfig(ContinuousConfig):
    motion_model: str = "homography"
    perspective_prior_weight: float = 0.001


class PhotometricRegistrar(ContinuousRegistrar):
    """ContinuousRegistrar-compatible push with at most four dense edge trials."""

    def __init__(self, config=None, fit_config=None):
        super().__init__(config or PhotometricConfig(), fit_config)
        self.last_photometric_audit = None
        self.maximum_fixed_keyframe_trials = 3
        self.display_fit_point_limit = 200
        self.recovery_locked = False

    def _key(self, gray, usable, sample, h):
        key = super()._key(gray, usable, sample, h)
        if key is None:
            # A low-contrast frame may constrain a dense fit without enough
            # sparse corners. The dense helper still checks texture and rank.
            isolated, safe = self._safe(gray, usable)
            key = dict(gray=isolated, safe=safe, points=np.empty((0, 2), np.float32),
                       target=np.empty((0, 2), np.float32),
                       H_reference_from_keyframe=np.asarray(h).copy(),
                       sample_index=sample["sample_index"], time=sample["source_time_s"])
        key["raw"] = gray.copy()
        key["usable"] = usable.copy()
        key["source_pts"] = int(sample["pts"])
        key["source_time_base"] = sample["time_base"]
        return key

    def _photo_edge(self, key, gray, usable, isolated, safe):
        initial = np.linalg.inv(key["H_reference_from_keyframe"]) @ self.last_h
        initializer = "last_successful_pose_expressed_in_observed_keyframe_pixels"
        sparse = dict(status="unavailable", reason="no_sparse_keyframe_features")
        if len(key["points"]) >= self.fit_config.min_tracks:
            sparse = self._edge(key, gray, isolated, safe)
            if sparse["status"] == "candidate":
                initial = np.asarray(sparse["H_keyframe_from_current"])
                initializer = "fresh_mask_isolated_sparse_observed_keyframe_edge"
        initial /= initial[2, 2]
        audit = estimate_photometric(key["raw"], gray, key["usable"], usable, initial,
            motion_model=self.config.motion_model, perspective_prior_weight=self.config.perspective_prior_weight)
        audit.update(edge_keyframe_sample_index=key["sample_index"], edge_keyframe_time_s=key["time"],
                     edge_keyframe_source_pts=key["source_pts"], edge_keyframe_time_base=key["source_time_base"],
                     initializer=initializer, initial_H_keyframe_from_current=initial.tolist(),
                     sparse_initializer_status=sparse["status"], sparse_initializer_reason=sparse.get("reason"),
                     H_reference_from_keyframe=key["H_reference_from_keyframe"].tolist(),
                     coordinate_note="helper reference means observed keyframe; compose only after local fitting")
        self.last_photometric_audit["attempts"].append(audit)
        edge = dict(status=audit["status"], reason=audit["reason"],
                    edge_keyframe_sample_index=key["sample_index"], edge_keyframe_time_s=key["time"],
                    initializer=initializer, photometric_final_correlation=audit.get("final_correlation"),
                    photometric_final_rmse_0_to_1=audit.get("final_rmse_0_to_1"))
        if audit["status"] != "candidate":
            return edge
        combined = key["H_reference_from_keyframe"] @ np.asarray(audit["H_reference_from_current"])
        combined /= combined[2, 2]
        current = np.asarray(audit["fit_current_xy"])
        key_xy = np.asarray(audit["fit_reference_xy"])
        support = np.asarray(audit["support"]["hull_xy"])
        denominator = (np.c_[support,np.ones(len(support))] @ combined.T)[:,2]
        if (not np.all(np.isfinite(combined)) or not np.all(np.isfinite(denominator))
                or np.min(denominator)*np.max(denominator) <= 0):
            edge.update(status="unavailable", reason="composed_reference_horizon_crosses_fitting_support")
            audit["composition_rejection"] = edge["reason"]
            return edge
        indices = np.linspace(0, len(current)-1, min(self.display_fit_point_limit, len(current)), dtype=int)
        edge.update(H_reference_from_current=combined.tolist(),
                    H_keyframe_from_current=audit["H_reference_from_current"],
                    pair_algorithm="observed_keyframe_mask_isolated_analytic_photometric_homography",
                    support_polygon=audit["support"]["hull_xy"], support_fraction=audit["support"]["hull_fraction"],
                    fit_current_xy=current[indices].tolist(), fit_keyframe_xy=key_xy[indices].tolist(),
                    fit_reference_xy=project(key["H_reference_from_keyframe"], key_xy[indices]).tolist(),
                    fit_points_role="photometric_sampling_not_independent_correspondences",
                    fit_points_are_subset=len(indices) < len(current), fit_points_full_count=len(current),
                    fit_points_uniform_indices=indices.tolist(),
                    photometric_safe_support_retained=audit["safe_support_retained"],
                    photometric_gain=audit["gain"], photometric_bias=audit["bias"],
                    photometric_design_rank_ratio=audit["design_rank_ratio"],
                    photometric_converged=audit["converged"],
                    motion_model=audit["motion_model"], model_assumption=audit["model_assumption"],
                    perspective_prior=audit["perspective_prior"],
                    fitting_residual_coordinate_space="normalized_intensity_not_geometric_pixels",
                    dense_optimization_used=True,
                    train_inliers=0, train_inliers_semantics="no_feature_inlier_count_for_dense_photometric_fit")
        return edge

    def push(self, image, sample, masks, force_break=False):
        time = float(int(sample["pts"])*Fraction(sample["time_base"]))
        if abs(time - sample["source_time_s"]) > 1e-5:
            raise ValueError("source_time_s disagrees with source PTS/time_base")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        usable = np.asarray(masks["usable"])
        gap = None if self.last_time is None else time-self.last_time
        if force_break:
            self.keys = []
            self.previous_key = None
            self.last_h = None
            self.last_good_time = None
            self.reference_sample_index = None
            self.recovery_locked = False
        elif ((gap is not None and (gap <= 0 or gap > self.config.max_recovery_gap_s))
              or (self.shape is not None and self.shape != gray.shape)):
            self.recovery_locked = True
        self.last_photometric_audit = dict(schema="rinkspace.photometric-keyframe-audit.v1",
            current_sample=dict(sample), independent_validation=False, attempts=[])
        result = {**sample, "status": "unavailable", "H_reference_from_current": None,
                  "matrix_direction": "H_reference_from_current", "geometry_level": "local_projective",
                  "unit": "reference_px", "independent_validation": False, "map_usable_for_metric_output": False,
                  "graph_component_id": None, "support_polygon": [], "fit_current_xy": [], "fit_reference_xy": [],
                  "mask_status": masks["mask_status"], "ice_fraction": masks["ice_fraction"],
                  "usable_fraction": masks["usable_fraction"], "dense_optimization_used": False,
                  "reason": "insufficient_safe_photometric_support", "keyframe_attempts": [],
                  "fit_points_role": "photometric_sampling_not_independent_correspondences"}
        result["motion_model"] = self.config.motion_model
        if force_break:
            result["break_reason"] = "declared_shot_boundary"
        if self.recovery_locked or (self.last_good_time is not None and
                                   time-self.last_good_time > self.config.max_recovery_gap_s):
            self.recovery_locked = True
            result["reason"] = "recovery_gap_exceeded_requires_explicit_new_component"
        elif not self.keys:
            # Reference initialization is visibly distinct from an estimated map.
            audit = estimate_photometric(gray, gray, usable, usable, np.eye(3),
                motion_model=self.config.motion_model, perspective_prior_weight=self.config.perspective_prior_weight)
            audit["reference_frame_only"] = True
            self.last_photometric_audit["attempts"].append(audit)
            if audit["status"] == "candidate":
                self.component += 1
                self.reference_sample_index = sample["sample_index"]
                self.last_h = np.eye(3)
                self.last_good_time = time
                key = self._key(gray, usable, sample, self.last_h)
                self.keys = [key]
                self.previous_key = key
                result.update(status="candidate", H_reference_from_current=np.eye(3).tolist(),
                              reference_frame_only=True, reason="first_observed_reference_only",
                              support_polygon=audit["support"]["hull_xy"],
                              support_fraction=audit["support"]["hull_fraction"])
            else:
                result["reason"] = audit["reason"]
        else:
            isolated, safe = self._safe(gray, usable)
            # Direct longer-baseline fits reduce pose composition frequency.
            # Only the nearest three fixed observations and previous success
            # are tried; no unbounded search or reference identity fallback.
            candidates = self.keys[-self.maximum_fixed_keyframe_trials:]
            if self.previous_key is not None and all(k["sample_index"] != self.previous_key["sample_index"] for k in candidates):
                candidates = candidates + [self.previous_key]
            for key in candidates:
                edge = self._photo_edge(key, gray, usable, isolated, safe)
                result["keyframe_attempts"].append({k: edge.get(k) for k in (
                    "edge_keyframe_sample_index", "status", "reason", "initializer",
                    "photometric_final_correlation", "photometric_final_rmse_0_to_1")})
                if edge["status"] == "candidate":
                    result.update(edge)
                    break
            if result["status"] == "candidate":
                result["recovered_after_missing"] = self.last_time != self.last_good_time
                self.last_h = np.asarray(result["H_reference_from_current"])
                self.last_good_time = time
                self.previous_key = self._key(gray, usable, sample, self.last_h)
                if time-self.keys[-1]["time"] >= self.config.keyframe_interval_s-1e-6:
                    self.keys.append(self.previous_key)
                    if len(self.keys) > self.config.max_keyframes:
                        self.keys.pop(1)  # Keep the first physical reference available.
        if result["status"] == "candidate":
            result["graph_component_id"] = f"local_{self.component:03d}"
            result["reference_sample_index"] = self.reference_sample_index
        result["fitting_patch_policy"] = dict(input_mask_isolation=True, dense_refinement=True,
            dense_gaussian_radius_px=2, dense_safe_mask_erosion_px=3,
            dense_interpolation="all_four_safe_bilinear_corners",
            full_dense_audit="last_photometric_audit_exported_separately",
            sparse_initializer_center_erosion_px=(self.config.patch_radius_px+3)*2**self.config.pyramid_levels)
        self.last_time = time
        self.shape = gray.shape
        return result
