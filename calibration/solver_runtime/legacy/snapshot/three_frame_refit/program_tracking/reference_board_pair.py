"""Direct original-to-programme pixels on the existing segmented board wall.

Each source Brown ray is intersected afresh with the first visible board
panel in the current camera state.  No atlas XYZ, flat ad plane, homography
prediction, or independently invented world landmark enters the residual.
This candidate factor is intentionally separate from the running solver.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from projection import from_state, project_fast, to_state


def _brown_world_rays(camera: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Invert the independent Brown14 template, including sensor skew."""
    fx, fy = np.exp(camera[6:8])
    native = np.ascontiguousarray(np.asarray(pixels, np.float64).reshape(-1, 2))
    unskewed = native.copy()
    unskewed[:, 0] -= camera[10]*(native[:, 1]-camera[9])/fy
    k = np.array([[fx, 0., camera[8]], [0., fy, camera[9]], [0., 0., 1.]])
    if np.max(np.abs(camera[11:23])) <= 1e-14:
        normalized = np.column_stack(((unskewed[:, 0]-camera[8])/fx,
                                      (unskewed[:, 1]-camera[9])/fy))
    else:
        normalized = cv2.undistortPoints(
            unskewed.reshape(-1, 1, 2), k,
            np.ascontiguousarray(camera[11:25])).reshape(-1, 2)
    rotation = cv2.Rodrigues(np.ascontiguousarray(camera[3:6]))[0]
    return np.ascontiguousarray(np.column_stack((normalized,
                                                np.ones(len(native)))) @ rotation)


class BoardIntersectionError(ValueError):
    """A real pixel ray cannot be assigned to the verified visible print arc."""

    def __init__(self, reason: str, indices):
        self.reason = reason
        self.indices = np.asarray(indices, np.int32).reshape(-1)
        super().__init__(f"{reason} at pair indices {self.indices.tolist()}")


@dataclass(frozen=True)
class BoardHits:
    xyz: np.ndarray
    panel: np.ndarray
    panel_fraction: np.ndarray
    valid: np.ndarray
    reason: tuple[str, ...]


@dataclass
class ReferenceBoardPair:
    """Actual UV pairs linked only through a verified board perimeter arc.

    ``allowed_panels`` identifies the printed physical region, including all
    its short straight and curved panels.  Individual rays may land on
    different adjacent panels; an unrelated near/other-side hit is rejected.
    Invalid or occluded rays raise rather than acquiring a synthetic 3D point.
    """

    xy0: np.ndarray
    xy1: np.ndarray
    scene: dict
    allowed_panels: np.ndarray
    camera0_template: np.ndarray
    camera1_template: np.ndarray

    def __post_init__(self):
        self.xy0 = np.asarray(self.xy0, np.float64).reshape(-1, 2).copy()
        self.xy1 = np.asarray(self.xy1, np.float64).reshape(-1, 2).copy()
        if (self.xy0.shape != self.xy1.shape or not len(self.xy0)
                or not np.isfinite(self.xy0).all() or not np.isfinite(self.xy1).all()
                or np.any(self.xy0 < 0) or np.any(self.xy1 < 0)
                or np.any(self.xy0 >= (1920, 1080))
                or np.any(self.xy1 >= (1920, 1080))):
            raise ValueError("Board pairs need equal, finite native source pixels")
        self.bottom = np.asarray(self.scene["board_surface"]["bottom_edge_xyz"], np.float64)
        self.top = np.asarray(self.scene["board_surface"]["top_edge_xyz"], np.float64)
        if (self.bottom.shape != self.top.shape or self.bottom.ndim != 2
                or self.bottom.shape[1] != 3 or len(self.bottom) < 4
                or not np.isfinite(self.bottom).all() or not np.isfinite(self.top).all()
                or not np.allclose(self.bottom[:, :2], self.top[:, :2], atol=1e-9)
                or np.any(self.top[:, 2] <= self.bottom[:, 2])):
            raise ValueError("Expected the existing vertical segmented board wall")
        if not np.allclose(self.bottom, self.scene["ice_boundary"]["points_xyz"]):
            raise ValueError("Board bottom must be the existing ice boundary")
        panels = np.asarray(self.allowed_panels, np.int32).reshape(-1)
        if (not len(panels) or np.any(panels < 0) or np.any(panels >= len(self.bottom))
                or len(np.unique(panels)) != len(panels)):
            raise ValueError("Verified print region needs unique existing panel IDs")
        self.allowed_panels = panels.copy()
        self._allowed = np.zeros(len(self.bottom), bool)
        self._allowed[panels] = True
        self._starts = self.bottom[:, :2].copy()
        self._edges = np.roll(self._starts, -1, axis=0)-self._starts
        self._polygon = np.ascontiguousarray(self._starts.astype(np.float32))
        for name in ("camera0_template", "camera1_template"):
            camera = np.asarray(getattr(self, name), np.float64)
            if (camera.shape != (25,) or not np.isfinite(camera).all()
                    or np.any(np.abs(camera[23:25]) > 1e-12)):
                raise ValueError("Independent 25-vector Brown templates need zero fixed tilt")
            setattr(self, name, camera.copy())

    def _hits_from_rays(self, centre: np.ndarray, rays: np.ndarray) -> BoardHits:
        """Find global first board hit, then enforce print-arc and ice visibility."""
        count = len(rays)
        rel = self._starts-centre[:2]
        denom = (rays[:, 0, None]*self._edges[None, :, 1]
                 -rays[:, 1, None]*self._edges[None, :, 0])
        numerator = rel[:, 0]*self._edges[:, 1]-rel[:, 1]*self._edges[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = numerator[None, :]/denom
            along = (rel[None, :, 0]*rays[:, None, 1]
                     -rel[None, :, 1]*rays[:, None, 0])/denom
        z = centre[2]+distance*rays[:, 2, None]
        lower = (self.bottom[None, :, 2]*(1-along)
                 +np.roll(self.bottom[:, 2], -1)[None, :]*along)
        upper = (self.top[None, :, 2]*(1-along)
                 +np.roll(self.top[:, 2], -1)[None, :]*along)
        possible = (np.isfinite(distance) & (distance > 1e-8)
                    & (along >= -1e-8) & (along <= 1+1e-8)
                    & (z >= lower-1e-7) & (z <= upper+1e-7))
        candidate = np.where(possible, distance, np.inf)
        first = np.min(candidate, axis=1)
        # Two adjoining panels can meet at one physical seam. In that tie,
        # select the verified side; never skip a nearer unverified wall.
        with np.errstate(invalid="ignore"):
            tied = (np.isfinite(first[:, None])
                    & (np.abs(candidate-first[:, None]) <= 1e-7))
        verified_tied = tied & self._allowed[None, :]
        panel = np.argmin(np.where(verified_tied, candidate, np.inf), axis=1)
        has_verified_first = verified_tied.any(axis=1)
        panel[~has_verified_first] = np.argmin(candidate[~has_verified_first], axis=1)
        fraction = along[np.arange(count), panel]
        xyz = centre[None, :]+first[:, None]*rays
        reason = ["ok"]*count
        for i in range(count):
            if not np.isfinite(rays[i]).all():
                reason[i] = "invalid_brown_ray"
            elif not np.isfinite(first[i]):
                reason[i] = "no_board_hit"
            elif not has_verified_first[i]:
                reason[i] = "first_hit_outside_verified_print_arc"
            else:
                ground_t = (-centre[2]/rays[i, 2]
                            if rays[i, 2] < -1e-10 else np.inf)
                if 0 < ground_t < first[i]-1e-7:
                    ground_xy = centre[:2]+ground_t*rays[i, :2]
                    if cv2.pointPolygonTest(self._polygon,
                                            tuple(map(float, ground_xy)), False) >= 0:
                        reason[i] = "ice_occludes_board_hit"
        valid = np.asarray([r == "ok" for r in reason], bool)
        xyz[~valid] = np.nan
        panel[~valid] = -1
        fraction[~valid] = np.nan
        return BoardHits(xyz, panel, fraction, valid, tuple(reason))

    def source_hits(self, source_state: np.ndarray,
                    source_pixels: np.ndarray | None = None) -> BoardHits:
        camera = from_state(source_state, self.camera0_template)
        native = self.xy0 if source_pixels is None else np.asarray(source_pixels, np.float64)
        rays = _brown_world_rays(camera, native)
        roundtrip, depth = project_fast(camera, camera[:3]+100.*rays)
        good = (np.isfinite(roundtrip).all(axis=1) & (depth > 0)
                & (np.linalg.norm(roundtrip-native, axis=1) <= .05))
        rays[~good] = np.nan
        return self._hits_from_rays(camera[:3], rays)

    @staticmethod
    def _require(hits: BoardHits):
        for reason in sorted(set(hits.reason)-{"ok"}):
            raise BoardIntersectionError(reason,
                [i for i, value in enumerate(hits.reason) if value == reason])

    def _project_pixels(self, states, source_pixels, with_jacobian: bool = False):
        source, target = states
        hits = self.source_hits(source, source_pixels)
        self._require(hits)
        camera = from_state(target, self.camera1_template)
        projected = project_fast(camera, hits.xyz, with_jacobian=with_jacobian)
        uv, depth = projected[:2]
        bad = np.flatnonzero(~np.isfinite(uv).all(axis=1) | (depth <= 0))
        if len(bad):
            raise BoardIntersectionError("target_projection_invalid", bad)
        # Visibility is checked at the dynamically projected source point,
        # not by assigning target measurements a model-derived world point.
        direction = hits.xyz-camera[:3]
        target_hits = self._hits_from_rays(camera[:3], direction)
        self._require(target_hits)
        occluded = np.flatnonzero(np.linalg.norm(target_hits.xyz-hits.xyz, axis=1) > 1e-4)
        if len(occluded):
            raise BoardIntersectionError("target_board_occlusion", occluded)
        return (uv, hits.panel, projected[2]) if with_jacobian else (uv, hits.panel)

    def projected(self, states, with_jacobian: bool = False):
        return self._project_pixels(states, self.xy0, with_jacobian)

    def _raw_residual(self, states):
        uv, _ = self.projected(states)
        return (uv-self.xy1).reshape(-1)

    def _raw_jacobians(self, states):
        source, target = map(lambda item: np.asarray(item, np.float64), states)
        _, base_panel, target_jac = self.projected([source, target], True)
        source_jac = np.empty((2*len(self.xy0), 11), np.float64)
        for axis in range(11):
            step = 1e-6
            high, low = source.copy(), source.copy()
            high[axis] += step; low[axis] -= step
            high_uv, high_panel = self.projected([high, target])
            low_uv, low_panel = self.projected([low, target])
            if np.any(high_panel != base_panel) or np.any(low_panel != base_panel):
                raise BoardIntersectionError("nondifferentiable_panel_transition",
                    np.flatnonzero((high_panel != base_panel) | (low_panel != base_panel)))
            source_jac[:, axis] = ((high_uv-low_uv)/(2*step)).reshape(-1)
        return [source_jac, target_jac]

    def initial_states(self):
        return [to_state(self.camera0_template), to_state(self.camera1_template)]

    @staticmethod
    def _pixel_sigmas(value, count: int, name: str) -> np.ndarray:
        sigmas = np.asarray(value, np.float64)
        if sigmas.ndim == 0:
            sigmas = np.full((count, 2), float(sigmas))
        elif sigmas.shape == (count,):
            sigmas = np.repeat(sigmas[:, None], 2, axis=1)
        elif sigmas.shape != (count, 2):
            raise ValueError(f"{name} must be scalar, N, or Nx2 native pixel sigma")
        if not np.isfinite(sigmas).all() or np.any(sigmas <= 0):
            raise ValueError(f"{name} must be finite and positive")
        return sigmas

    def source_pixel_jacobian(self, states, *, step_px: float = .1) -> np.ndarray:
        """Derivative of target prediction w.r.t. each actual source UV."""
        if not np.isfinite(step_px) or step_px <= 0:
            raise ValueError("Source pixel derivative step must be positive")
        _, base_panel = self.projected(states)
        result = np.empty((len(self.xy0), 2, 2), np.float64)
        for axis in range(2):
            high, low = self.xy0.copy(), self.xy0.copy()
            high[:, axis] += step_px
            low[:, axis] -= step_px
            high_uv, high_panel = self._project_pixels(states, high)
            low_uv, low_panel = self._project_pixels(states, low)
            switched = np.flatnonzero((high_panel != base_panel) |
                                    (low_panel != base_panel))
            if len(switched):
                raise BoardIntersectionError("source_pixel_crosses_board_panel", switched)
            result[:, :, axis] = (high_uv-low_uv)/(2*step_px)
        return result

    def freeze_pixel_uncertainty(self, states, source_sigma_px,
                                 target_sigma_px) -> dict:
        """Combine both image errors once and freeze whitening at the warm start.

        C = J_source Sigma_source J_source.T + Sigma_target per matched
        print point. One two-component factor represents each pixel pair;
        using an extra reverse factor would count the same evidence twice.
        """
        count = len(self.xy0)
        source = self._pixel_sigmas(source_sigma_px, count, "source_sigma_px")
        target = self._pixel_sigmas(target_sigma_px, count, "target_sigma_px")
        pixel_jac = self.source_pixel_jacobian(states)
        covariance = (np.einsum("nij,nj,nkj->nik", pixel_jac, source**2,
                                pixel_jac)
                      + np.einsum("ni,ij->nij", target**2, np.eye(2)))
        chol = np.linalg.cholesky(covariance)
        self._whitener = np.linalg.inv(chol)
        self._source_sigma_px = source
        self._target_sigma_px = target
        source_magnification = np.linalg.svd(pixel_jac, compute_uv=False)
        effective_sigma = np.sqrt(np.linalg.eigvalsh(covariance))
        return {"source_to_target_singular_values": source_magnification.tolist(),
                "combined_pixel_sigma_eigenvalues_sqrt": effective_sigma.tolist(),
                "source_sigma_px": source.tolist(), "target_sigma_px": target.tolist(),
                "covariance_frozen_at_warm_start": True,
                "independent_reverse_factor_added": False}

    def weighted_residual(self, states):
        if not hasattr(self, "_whitener"):
            raise ValueError("Freeze source and target pixel uncertainty first")
        raw = self._raw_residual(states).reshape(-1, 2)
        return np.einsum("nij,nj->ni", self._whitener, raw).reshape(-1)

    def weighted_jacobians(self, states):
        if not hasattr(self, "_whitener"):
            raise ValueError("Freeze source and target pixel uncertainty first")
        result = []
        for matrix in self._raw_jacobians(states):
            raw = np.asarray(matrix, np.float64).reshape(len(self.xy0), 2, -1)
            result.append(np.einsum("nij,njk->nik", self._whitener, raw).reshape(
                2*len(self.xy0), -1))
        return result

    def residual(self, states):
        """Whitened paired residual; apply pointwise Huber only after this."""
        return self.weighted_residual(states)

    def jacobians(self, states):
        """Jacobians of the same frozen-covariance whitened residual."""
        return self.weighted_jacobians(states)


def load_reference_board_pairs(specifications, scene, programme_cameras,
                               reference_cameras, source_hash, scene_hash,
                               input_hashes):
    """Validate archived actual UV pairs; leave graph integration to caller."""
    result = []
    used_source_pixels = set()
    for spec in specifications:
        path = Path(spec["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != spec.get("sha256"):
            raise ValueError("Frozen board reference packet hash changed")
        with np.load(path, allow_pickle=False) as data:
            stamp0, stamp1 = int(data["source_pts"]), int(data["target_pts"])
            if (str(data["schema"]) != "rinkspace.reference_board_uv.v1"
                    or str(data["source_video_sha256"]) != source_hash
                    or str(data["scene_sha256"]) != scene_hash):
                raise ValueError("Board reference packet belongs to another source/scene")
            if stamp0 not in reference_cameras or stamp1 not in programme_cameras:
                raise ValueError("Both exact PTS must be selected graph camera nodes")
            metadata = json.loads(str(data["metadata_json"]))
            if (metadata.get("candidate_only") is not True
                    or metadata.get("actual_video_uv_only") is not True
                    or metadata.get("same_print_visual_review") is not True):
                raise ValueError("Board pair lacks direct same-print source evidence")
            cache = Path(str(data["observation_cache_path"]))
            cache_hash = hashlib.sha256(cache.read_bytes()).hexdigest()
            if cache_hash != str(data["observation_cache_sha256"]):
                raise ValueError("Board pair source observation cache changed")
            atlas = Path(str(data["atlas_feature_path"]))
            atlas_hash = hashlib.sha256(atlas.read_bytes()).hexdigest()
            if atlas_hash != str(data["atlas_feature_sha256"]):
                raise ValueError("Board pair atlas descriptor source changed")
            xy0 = np.asarray(data["source_uv_inliers"], np.float64)
            xy1 = np.asarray(data["target_uv_inliers"], np.float64)
            if len(xy0) < 5:
                raise ValueError("Insufficient real board texture support")
            all0 = np.asarray(data["source_uv_candidates"], np.float64)
            all1 = np.asarray(data["target_uv_candidates"], np.float64)
            accepted = np.asarray(data["accepted_indices"], np.int32)
            if (len(all0) != len(all1) or len(accepted) != len(xy0)
                    or not np.array_equal(all0[accepted], xy0)
                    or not np.array_equal(all1[accepted], xy1)):
                raise ValueError("Board packet accepted pixels differ from archived cache")
            with np.load(cache, allow_pickle=False) as original:
                if int(original["pts"]) != stamp1:
                    raise ValueError("Board cache has another exact native PTS")
                original_pairs = np.column_stack((original["reference_uv"],
                                                  original["uv"]))
                for row in np.column_stack((all0, all1)):
                    if not np.any(np.all(original_pairs == row, axis=1)):
                        raise ValueError("Board packet contains UV absent from real cache")
            qa = Path(metadata["qa_path"])
            if hashlib.sha256(qa.read_bytes()).hexdigest() != metadata["qa_sha256"]:
                raise ValueError("Board same-print visual QA changed")
            source_image = Path(metadata["original_source_image_path"])
            if (hashlib.sha256(source_image.read_bytes()).hexdigest()
                    != metadata["original_source_image_sha256"]):
                raise ValueError("Board original low3 source image changed")
            source_sigma = np.asarray(data["source_sigma_px"], np.float64)
            target_sigma = np.asarray(data["target_sigma_px"], np.float64)
            packet_panels = np.asarray(data["allowed_panel_indices"], np.int32)
        if "allowed_panels" in spec and not np.array_equal(
                np.asarray(spec["allowed_panels"], np.int32), packet_panels):
            raise ValueError("Reviewed print-arc gate differs from frozen packet")
        allowed = packet_panels
        if np.any(source_sigma < 5.) or np.any(target_sigma < 8.):
            raise ValueError("Blurred print needs conservative source and target pixel uncertainty")
        pair = ReferenceBoardPair(xy0, xy1, scene, allowed,
                                  reference_cameras[stamp0], programme_cameras[stamp1])
        source_keys = {(stamp0, *map(float, uv)) for uv in xy0}
        if len(source_keys) != len(xy0) or source_keys & used_source_pixels:
            raise ValueError("Shared source print pixels require correlated cross-frame noise")
        used_source_pixels.update(source_keys)
        noise = pair.freeze_pixel_uncertainty(pair.initial_states(),
                                              source_sigma, target_sigma)
        input_hashes[str(path.resolve())] = digest
        input_hashes[str(cache.resolve())] = cache_hash
        input_hashes[str(atlas.resolve())] = atlas_hash
        input_hashes[str(qa.resolve())] = metadata["qa_sha256"]
        input_hashes[str(source_image.resolve())] = metadata["original_source_image_sha256"]
        result.append((stamp0, stamp1, pair, 1., {**metadata,
            "path": str(path.resolve()), "sha256": digest,
            "actual_pixel_pairs": len(xy0), "static_xyz_used": False,
            "segmented_board_surface": True,
            "factor_residual": "pair.weighted_residual",
            "factor_jacobian": "pair.weighted_jacobians",
            "graph_sigma": 1., "noise": noise}))
    return result
