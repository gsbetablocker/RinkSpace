"""Source-only board-body admission for measured material motion.

The retained source packets contain the original native top/contact readouts.
Their independent visible runs bound conservative convex interior rectangles.
These bounds are admission evidence only: they never add venue fit rows, use a
world projection, bridge source gaps, or certify an unknown source entity.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

SCHEMA = "rinkspace.native-board-body-support.v1"
POLICY = "x_intersection_no_extrapolation_y_extreme_plus_full_transition_iqr"


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def reference(path):
    path = Path(path)
    return dict(path=str(path), sha256=_digest(path.read_bytes()))


def _read(ref):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("Explicit immutable source reference required")
    raw = Path(ref["path"]).read_bytes()
    if _digest(raw) != ref["sha256"]:
        raise ValueError("Native board material evidence changed: " + ref["path"])
    return raw


def _array_digest(value):
    return _digest(np.asarray(value, dtype="<f8").tobytes())


def _boundary_runs(packet, role, *, include_legacy=False):
    prefix = "board_" + role
    output = []
    for feature in packet.get("features", []):
        if (feature.get("physical_surface") != "board"
                or not (feature.get("id") == prefix or str(feature.get("id", "")).startswith(prefix + "."))):
            continue
        for index, segment in enumerate(feature.get("segments", [])):
            legacy_kind = None
            if include_legacy and role == "top" and segment.get("measurement_origin") == "same_frame_joint_straight_edge_fit" and segment.get("source_evidence_measured_in_this_frame") is True and segment.get("hidden_span_filled") is False:
                estimate = feature.get("source_straight_line_estimate", {})
                raw = np.asarray(estimate.get("evidence_uv", []), float)
                widths = np.asarray(estimate.get("evidence_localization_sigma_px", []), float)
                visible = np.asarray(segment.get("uv", []), float)
                if (estimate.get("model_projection_used") is False and raw.ndim == 2 and raw.shape[1:] == (2,)
                        and widths.shape == (len(raw),) and visible.ndim == 2 and visible.shape[1:] == (2,)):
                    keep = (raw[:,0] >= visible[:,0].min()) & (raw[:,0] <= visible[:,0].max())
                    uv, support = raw[keep], widths[keep]
                    support = support + max(0.,float(segment.get("current_curvature_bound_px", 0.)))
                    legacy_kind = "feature_retained_original_colour_profiles_with_visible_segment_support"
            elif include_legacy and role == "contact" and segment.get("pixels_measured_in_this_frame") is True and segment.get("additional_points_measured_in_native_pixels") is True:
                bindings = segment.get("point_bindings", [])
                if bindings and all(b.get("measurement_origin") == "native_colour_half_transition" for b in bindings):
                    uv = np.asarray(segment.get("uv", []),float)
                    widths = list(segment.get("endpoint_evidence_transition_iqr_px", [])) + [b["localization_sigma_px"] for b in bindings if "localization_sigma_px" in b]
                    if widths and np.isfinite(widths).all() and min(widths) > 0:
                        extra = max(0.,float(segment.get("pre_compaction_reconstruction_max_px", 0.)),float(segment.get("current_curvature_bound_px", 0.)))
                        support = np.full(len(uv),max(widths)+extra)
                        legacy_kind = "retained_measured_native_half_transition_with_full_recorded_support_envelope"
            # Fitted representatives are not re-used as source boundary pixels.
            # An unsupported/legacy run stays unavailable.
            if legacy_kind is None and (segment.get("measurement_origin") not in ("current_native_joint_smooth_material_curve_fit",
                        "current_native_joint_straight_material_edge_fit")
                    or segment.get("source_evidence_measured_in_this_frame") is not True):
                continue
            if legacy_kind is None:
                uv = np.asarray(segment.get("source_native_readout_uv", []), float)
                support = np.asarray(segment.get("source_native_transition_iqr_px", []), float)
            if (uv.ndim != 2 or uv.shape[1:] != (2,) or len(uv) < 2
                    or support.shape != (len(uv),) or not np.isfinite(uv).all()
                    or not np.isfinite(support).all() or np.any(support <= 0)):
                continue
            # Each existing run is kept separate, including occlusion gaps.
            if not (np.all(np.diff(uv[:, 0]) > 0) or np.all(np.diff(uv[:, 0]) < 0)):
                continue
            output.append(dict(feature_id=feature["id"], segment_index=index,
                readout_sha256=_array_digest(uv), transition_sha256=_array_digest(support),
                readout_count=len(uv), x_interval=[float(uv[:, 0].min()), float(uv[:, 0].max())],
                supported_y_interval=[float(np.min(uv[:, 1]-support)),
                                      float(np.max(uv[:, 1]+support))]))
    return output


def derive_regions(packet, *, include_legacy=False):
    """Build only source-supported interior boxes; no across-gap interpolation."""
    regions = []
    for upper in _boundary_runs(packet, "top",include_legacy=include_legacy):
        for lower in _boundary_runs(packet, "contact",include_legacy=include_legacy):
            x0 = max(upper["x_interval"][0], lower["x_interval"][0])
            x1 = min(upper["x_interval"][1], lower["x_interval"][1])
            y0 = upper["supported_y_interval"][1]
            y1 = lower["supported_y_interval"][0]
            if x0 >= x1 or y0 >= y1:
                continue
            region_id = "top_" + upper["feature_id"] + "_" + str(upper["segment_index"]) + "__contact_" + lower["feature_id"] + "_" + str(lower["segment_index"])
            regions.append(dict(region_id=region_id, physical_surface="board_body",
                polygon_xy=[[x0,y0],[x1,y0],[x1,y1],[x0,y1]],
                upper=upper, lower=lower, rectangle_policy=POLICY))
    return regions


def frame_evidence(meta, *, include_legacy=False):
    """Freeze an existing original-pixel boundary packet, without admitting flow."""
    packet = json.loads(_read(meta["source_packet"]))
    source = packet["source"]
    index, pts = int(meta["frame"]), int(meta["native_pts"])
    if (source.get("programme_frame_index_1based") != index
            or source.get("native_pts") != pts or source.get("time_base") != "1/12800"
            or source.get("source_video_sha256") != meta["source_video_sha256"]):
        raise ValueError("Native board boundary source clock/identity differs")
    sha = source.get("png_sha256")
    image_auth = "original_PNG_byte_hash"
    if include_legacy and not sha and source.get("decoded_BGR_sha256"):
        raw = Path(source["image_path"]).read_bytes()
        original_bgr = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if original_bgr is None or _digest(original_bgr.tobytes()) != source["decoded_BGR_sha256"]:
            raise ValueError("Legacy original decoded BGR identity changed")
        sha = _digest(raw)
        image_auth = "original_decoded_BGR_hash_plus_new_PNG_byte_hash"
    if (not isinstance(sha, str) or len(sha) != 64
            or source.get("image_path") != meta["image_path"]
            or source.get("mask_path") != meta["mask_path"]
            or source.get("mask_sha256") != meta.get("mask_sha256")):
        raise ValueError("Original RGB and person mask references are incomplete")
    return dict(frame_index_1based=index, native_pts=pts,
        source_video_sha256=meta["source_video_sha256"],
        source_frame=dict(path=source["image_path"],sha256=sha),
        source_packet=dict(meta["source_packet"]),
        person_mask=dict(path=source["mask_path"],sha256=source["mask_sha256"]),
        regions=derive_regions(packet,include_legacy=include_legacy),
        boundary_evidence_mode=("retained-native-boundary-evidence-v2" if include_legacy else "native-run-readouts-v1"),
        source_frame_authentication=image_auth,
        admission_basis="original_rgb_boundary_readouts",
        exclusions=dict(person_mask_positive_pixels=True,person_halo_px=15,
            broadcast_rectangles_xyxy=[[0,0,599,139],[1660,930,1900,1020]],
            image_border_px=20),
        fit_observations_added=False)


class NativeBoardMaterialSupport:
    def __init__(self, evidence):
        if (evidence.get("schema") != SCHEMA or evidence.get("world_projection_used") is not False
                or evidence.get("fit_observations_added") is not False):
            raise ValueError("Explicit original RGB board-body support evidence required")
        self.evidence = evidence
        self.frames = {int(f["frame_index_1based"]):f for f in evidence["frames"]}
        if len(self.frames) != len(evidence["frames"]):
            raise ValueError("Duplicated native board support frame")
        videos = {f["source_video_sha256"] for f in evidence["frames"]}
        if len(videos) != 1:
            raise ValueError("One exact original video identity required")
        self.source_video_sha256 = next(iter(videos))
        self.time_base = "1/12800"
        self.receipt = dict(schema=SCHEMA, source_video_sha256=self.source_video_sha256,
            time_base=self.time_base, world_projection_used=False,
            source_boundary_interiors_are_derived_not_new_per_patch_manual_review=True,
            original_fit_eligibility_not_overridden=True,
            native_boundary_sampling_support_not_a_calibrated_probability_bound=True)
        self._authenticated = {}

    def _frame(self, index):
        if index in self._authenticated:
            return self._authenticated[index]
        f = self.frames.get(index)
        if f is None:
            raise ValueError("No source-confirmed board-body region for frame " + str(index))
        packet = json.loads(_read(f["source_packet"]))
        s = packet["source"]
        if (s.get("native_pts") != f["native_pts"]
                or s.get("programme_frame_index_1based") != index
                or s.get("source_video_sha256") != f["source_video_sha256"]
                or s.get("time_base") != "1/12800"
                or s.get("image_path") != f["source_frame"]["path"]
                or (s.get("png_sha256") != f["source_frame"]["sha256"] and not (f.get("source_frame_authentication") == "original_decoded_BGR_hash_plus_new_PNG_byte_hash" and not s.get("png_sha256") and s.get("decoded_BGR_sha256")))
                or s.get("mask_path") != f["person_mask"]["path"]
                or s.get("mask_sha256") != f["person_mask"]["sha256"]):
            raise ValueError("Material support differs from original native source packet")
        expected = derive_regions(packet,include_legacy=f.get("boundary_evidence_mode") == "retained-native-boundary-evidence-v2")
        if f["regions"] != expected or f.get("admission_basis") != "original_rgb_boundary_readouts":
            raise ValueError("Board-body region is not the exact original-pixel boundary interior")
        exclusions = f["exclusions"]
        if exclusions != dict(person_mask_positive_pixels=True,person_halo_px=15,
                broadcast_rectangles_xyxy=[[0,0,599,139],[1660,930,1900,1020]],image_border_px=20):
            raise ValueError("Material support exclusion policy changed")
        raw_image = _read(f["source_frame"])
        image = cv2.imdecode(np.frombuffer(raw_image,np.uint8),cv2.IMREAD_GRAYSCALE)
        if f.get("source_frame_authentication") == "original_decoded_BGR_hash_plus_new_PNG_byte_hash":
            bgr = cv2.imdecode(np.frombuffer(raw_image,np.uint8),cv2.IMREAD_COLOR)
            if bgr is None or _digest(bgr.tobytes()) != s.get("decoded_BGR_sha256"):
                raise ValueError("Legacy source image differs from original decoded BGR")
        mask = cv2.imdecode(np.frombuffer(_read(f["person_mask"]),np.uint8),cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape != (1080,1920) or mask.shape != image.shape:
            raise ValueError("Original native RGB or person mask has an invalid shape")
        # Rectangular component enclosures are more conservative than the
        # original dilated silhouette and avoid retaining full-frame arrays.
        _, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
        halo = exclusions["person_halo_px"]
        blocked = [[int(a)-halo,int(b)-halo,int(a+c-1)+halo,int(b+d-1)+halo]
                   for a,b,c,d,_ in stats[1:]]
        blocked.extend(exclusions["broadcast_rectangles_xyxy"])
        self._authenticated[index] = f, blocked
        return f, blocked


    def require(self, frame_index, native_pts, uv, support_radius_px):
        """Authenticate the entire actual local patch, not just its center."""
        index = int(frame_index)
        f, blocked = self._frame(index)
        if int(native_pts) != f["native_pts"]:
            raise ValueError("Board-body patch native PTS mismatch")
        point = np.asarray(uv,float)
        radius = float(support_radius_px)
        if point.shape != (2,) or not np.isfinite(point).all() or radius < 17 or not np.isfinite(radius):
            raise ValueError("Full finite 35px-or-larger material support required")
        x0,y0 = np.floor(point-radius).astype(int)
        x1,y1 = np.ceil(point+radius).astype(int)
        if x0 < 0 or y0 < 0 or x1 >= 1920 or y1 >= 1080:
            raise ValueError("Material patch leaves original image")
        border = f["exclusions"]["image_border_px"]
        if x0 < border or y0 < border or x1 >= 1920-border or y1 >= 1080-border:
            raise ValueError("Material patch intersects image-border exclusion")
        if any(x0 <= bx1 and x1 >= bx0 and y0 <= by1 and y1 >= by0 for bx0,by0,bx1,by1 in blocked):
            raise ValueError("Material patch intersects person, overlay, or image-border exclusion")
        for region in f["regions"]:
            r = np.asarray(region["polygon_xy"],float)
            if x0 >= r[:,0].min() and x1 <= r[:,0].max() and y0 >= r[:,1].min() and y1 <= r[:,1].max():
                return region["region_id"]
        raise ValueError("Full material patch is outside source-confirmed board body")


def load_support(ref):
    """Load immutable source evidence; pixels/masks authenticate once per frame."""
    return NativeBoardMaterialSupport(json.loads(_read(ref)))
