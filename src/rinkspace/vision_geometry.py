"""Native-pixel GPU motion and source-bound geometric shape diagnostics.

Flow is image motion, never a semantic label or a measured world trajectory.
Camera priors use the existing Brown camera bridge. A small shape residual is
not evidence of localization accuracy; model priors are never training labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _torch():
    import torch
    return torch


def _cuda_tensor(value, *, dimensions=None):
    torch = _torch()
    if not isinstance(value, torch.Tensor) or value.device.type != "cuda":
        raise ValueError("This operation requires a CUDA tensor")
    if dimensions is not None and value.ndim != dimensions:
        raise ValueError(f"Expected a {dimensions}-dimensional CUDA tensor")
    return value


def _source_binding(binding: Mapping[str, Any]) -> dict:
    result = dict(binding)
    required = ("source_sha256", "source_pts", "shot_id")
    if any(key not in result for key in required):
        raise ValueError("Source binding requires source_sha256, source_pts and shot_id")
    digest = str(result["source_sha256"])
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
        raise ValueError("Source binding needs an actual SHA256 identity")
    stamp = result["source_pts"]
    if isinstance(stamp, bool) or not isinstance(stamp, (int, np.integer)):
        raise ValueError("Source PTS must be an integer")
    time_base = result.get("time_base", result.get("source_time_base"))
    if time_base is None or Fraction(str(time_base)) <= 0 or not result["shot_id"]:
        raise ValueError("A positive source time_base and a shot identity are required")
    result.update(source_sha256=digest.lower(), source_pts=int(stamp),
                  time_base=str(Fraction(str(time_base))))
    return result


def _same_frame(first, second):
    a, b = _source_binding(first), _source_binding(second)
    keys = ("source_sha256", "source_pts", "time_base", "shot_id")
    if any(a[key] != b[key] for key in keys):
        raise ValueError("Camera/target and prediction belong to different source frames")
    for key in ("camera_group_id", "installation_id", "native_hw", "image_domain", "crop_xyxy"):
        if key in a or key in b:
            if a.get(key) != b.get(key):
                raise ValueError(f"Source coordinate binding differs: {key}")
    return a


def source_visibility_matches_frame(binding, evidence):
    """Match native RGB visibility evidence without inventing a shot identity."""
    current = _source_binding(binding)
    if not isinstance(evidence, Mapping):
        return False
    try:
        stamp = evidence["source_pts"]
        clock = Fraction(str(evidence.get("time_base", evidence.get("source_time_base"))))
        digest = evidence.get("source_sha256", evidence.get("source_hash"))
        if (isinstance(stamp, bool) or not isinstance(stamp, (int, np.integer))
                or digest != current["source_sha256"] or stamp != current["source_pts"]
                or str(clock) != current["time_base"]):
            return False
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        return False
    return all(key not in evidence or evidence[key] == current.get(key)
               for key in ("shot_id", "native_hw", "image_domain"))


def warp(source, target_to_source_flow):
    """Sample source at p_target + flow_target_to_source(p_target).

    Returns (sampled image, in-bounds/finite mask). Thus transporting frame 0
    features into frame 1 uses RAFT's *backward* (1 -> 0) flow. Pixel coordinates
    use align_corners=False and half-pixel normalization without changing units.
    """
    torch = _torch()
    import torch.nn.functional as F
    source = _cuda_tensor(source, dimensions=4)
    flow = _cuda_tensor(target_to_source_flow, dimensions=4)
    b, _, h, w = source.shape
    if flow.shape != (b, 2, h, w) or flow.device != source.device:
        raise ValueError("Flow and source must share batch, device and pixel domain")
    y, x = torch.meshgrid(torch.arange(h, device=source.device, dtype=torch.float32),
                          torch.arange(w, device=source.device, dtype=torch.float32), indexing="ij")
    sx, sy = x[None] + flow[:, 0].float(), y[None] + flow[:, 1].float()
    valid = (torch.isfinite(sx) & torch.isfinite(sy) &
             (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1))[:, None]
    grid = torch.stack((2 * (sx + .5) / w - 1, 2 * (sy + .5) / h - 1), dim=-1)
    grid = torch.nan_to_num(grid, nan=2., posinf=2., neginf=-2.)
    # FP32 interpolation also accepts BF16 feature inputs while preserving their gradient.
    sampled = F.grid_sample(source.float(), grid, mode="bilinear",
                            padding_mode="zeros", align_corners=False)
    return sampled.to(source.dtype), valid


def forward_backward_validity(forward, backward, *, absolute_px=1., relative=.05):
    """Forward/backward consistency is a visibility diagnostic, not certainty."""
    torch = _torch()
    if not np.isfinite(absolute_px) or absolute_px < 0 or not np.isfinite(relative) or relative < 0:
        raise ValueError("Flow discrepancy thresholds must be finite and nonnegative")
    opposite, inside = warp(backward, forward)
    error = torch.linalg.vector_norm(forward.float() + opposite.float(), dim=1, keepdim=True)
    magnitude = (torch.linalg.vector_norm(forward.float(), dim=1, keepdim=True) +
                 torch.linalg.vector_norm(opposite.float(), dim=1, keepdim=True))
    valid = (inside & torch.isfinite(forward).all(dim=1, keepdim=True) &
             torch.isfinite(opposite).all(dim=1, keepdim=True) &
             (error <= absolute_px + relative * magnitude))
    return valid, error


@dataclass(frozen=True)
class FlowPair:
    forward: Any  # frame 0 -> frame 1; full native domain, native pixel displacement
    backward: Any  # frame 1 -> frame 0; full native domain, native pixel displacement
    valid_forward: Any
    valid_backward: Any
    fb_error_forward: Any
    fb_error_backward: Any
    metadata: dict


class GPUFlow:
    """Frozen torchvision RAFT, loaded strictly from an existing local weight file."""
    def __init__(self, weights_path=None, *, device="cuda", num_updates=12,
                 fb_absolute_px=1., fb_relative=.05):
        torch = _torch()
        from torchvision.models.optical_flow import raft_large
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("GPUFlow requires an available CUDA device")
        if isinstance(num_updates, bool) or not isinstance(num_updates, int) or num_updates <= 0:
            raise ValueError("RAFT updates must be a positive integer")
        self.weights_path = Path(weights_path or Path(__file__).resolve().parents[2] /
                                 "models/cache/raft_large_C_T_SKHT_V2-ff5fadd5.pth").resolve()
        if not self.weights_path.is_file():
            raise FileNotFoundError(self.weights_path)
        self.weights_sha256 = hashlib.sha256(self.weights_path.read_bytes()).hexdigest()
        self.model = raft_large(weights=None)
        self.model.load_state_dict(torch.load(self.weights_path, map_location="cpu", weights_only=True), strict=True)
        self.model.requires_grad_(False).eval().to(self.device)
        self.num_updates = num_updates
        self.fb_absolute_px, self.fb_relative = fb_absolute_px, fb_relative

    def compute(self, image0, image1, *, analysis_hw=None, crop_xyxy=None,
                source0=None, source1=None) -> FlowPair:
        """RGB float [0,1] -> both flows in full native pixels.

        The same crop is used in both source frames. Padded analysis pixels are
        removed before anisotropic resizing; dx/dy are scaled separately. Outside
        the observed crop flows are zero, masks False and discrepancy infinite.
        Optional source bindings reject cuts and reversed source PTS.
        """
        torch = _torch()
        import torch.nn.functional as F
        image0, image1 = _cuda_tensor(image0, dimensions=4), _cuda_tensor(image1, dimensions=4)
        if image0.shape != image1.shape or image0.shape[1] != 3 or image0.device != image1.device:
            raise ValueError("RAFT requires matching RGB batches in one native pixel domain")
        if image0.device != next(self.model.parameters()).device:
            raise ValueError("Images and the RAFT model must use the same CUDA device")
        for image in (image0, image1):
            if not image.is_floating_point() or not bool(torch.isfinite(image).all()) or bool((image < 0).any()) or bool((image > 1).any()):
                raise ValueError("RAFT images must be finite RGB floats in [0,1]")
        bindings = None
        if source0 is not None or source1 is not None:
            if source0 is None or source1 is None:
                raise ValueError("Provide both source frame bindings")
            a, b = _source_binding(source0), _source_binding(source1)
            if any(a[k] != b[k] for k in ("source_sha256", "time_base", "shot_id")) or a["source_pts"] >= b["source_pts"]:
                raise ValueError("Temporal flow cannot cross cuts, sources or reversed PTS")
            bindings = [a, b]
        batch, _, h, w = image0.shape
        if bindings is not None:
            for binding in bindings:
                if "native_hw" in binding and list(binding["native_hw"]) != [h, w]:
                    raise ValueError("Flow inputs differ from their declared native source resolution")
        crop = tuple((0, 0, w, h) if crop_xyxy is None else crop_xyxy)
        if len(crop) != 4 or any(not isinstance(v, (int, np.integer)) for v in crop):
            raise ValueError("Crop must be integer native (x0,y0,x1,y1)")
        x0, y0, x1, y1 = map(int, crop)
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError("Crop leaves the native image domain")
        ch, cw = y1-y0, x1-x0
        ah, aw = tuple((ch, cw) if analysis_hw is None else analysis_hw)
        if any(not isinstance(v, (int, np.integer)) or v <= 0 for v in (ah, aw)):
            raise ValueError("Analysis size requires positive integer height and width")
        ph, pw = max(128, (ah+7)//8*8)-ah, max(128, (aw+7)//8*8)-aw
        prepared = []
        for image in (image0, image1):
            cropped = image[:, :, y0:y1, x0:x1].float()
            resized = F.interpolate(cropped, size=(ah, aw), mode="bilinear", align_corners=False)
            prepared.append(F.pad(resized * 2 - 1, (0, pw, 0, ph), mode="replicate"))
        with torch.no_grad():
            flows = [self.model(prepared[0], prepared[1], num_flow_updates=self.num_updates)[-1],
                     self.model(prepared[1], prepared[0], num_flow_updates=self.num_updates)[-1]]
            native = []
            for flow in flows:
                restored = F.interpolate(flow[:, :, :ah, :aw], size=(ch, cw), mode="bilinear", align_corners=False)
                restored = restored * restored.new_tensor([cw/aw, ch/ah])[None, :, None, None]
                native.append(restored)
            v0, e0 = forward_backward_validity(native[0], native[1], absolute_px=self.fb_absolute_px, relative=self.fb_relative)
            v1, e1 = forward_backward_validity(native[1], native[0], absolute_px=self.fb_absolute_px, relative=self.fb_relative)
            def embed(value, fill=0):
                full = value.new_full((batch, value.shape[1], h, w), fill)
                full[:, :, y0:y1, x0:x1] = value
                return full
            return FlowPair(embed(native[0]), embed(native[1]), embed(v0), embed(v1),
                            embed(e0, float("inf")), embed(e1, float("inf")), dict(
                native_hw=[h, w], crop_xyxy=list(crop), analysis_hw=[ah, aw],
                analysis_padding_ltrb=[0, 0, pw, ph], displacement_unit="native_pixels",
                forward_direction="frame0_to_frame1", warp0_to1_uses="backward",
                weights_path=str(self.weights_path), weights_sha256=self.weights_sha256,
                source_bindings=bindings, source_alignment_verified=bindings is not None,
                visibility="forward_backward_discrepancy_and_crop_bounds_only",
                flow_is_semantics=False, accuracy_validated=False))


@dataclass(frozen=True)
class ProjectedCurve:
    pixels: np.ndarray
    valid: np.ndarray
    segments: tuple[tuple[int, ...], ...]
    metadata: dict
    inside_native: np.ndarray | None = None


class CameraShapeAdapter:
    """Reuse source-bound native Brown projection; never synthesize a camera."""
    @staticmethod
    def _bridge():
        from calibration.program_tracking.body_tracking import camera_bridge
        return camera_bridge

    def project(self, world_xyz, camera25, *, geometry_id, model_revision,
                source_binding, camera_source_binding, native_hw, segment_ids=None):
        if camera25 is None or not geometry_id or not model_revision:
            raise ValueError("Real camera, geometry identity and model revision are required")
        binding = _same_frame(source_binding, camera_source_binding)
        h, w = tuple(native_hw)
        if any(not isinstance(v, (int, np.integer)) or v <= 0 for v in (h, w)):
            raise ValueError("Native height/width must be positive integers")
        if "native_hw" in binding and list(binding["native_hw"]) != [h, w]:
            raise ValueError("Camera source resolution differs from the requested native domain")
        if binding.get("image_domain", "native_brown_pixels") != "native_brown_pixels":
            raise ValueError("Camera projection requires native Brown source pixels")
        pixels, optical = self._bridge().project_world(world_xyz, camera25)
        valid = optical & np.isfinite(pixels).all(axis=1)
        inside = (valid & (pixels[:, 0] >= 0) & (pixels[:, 0] < w) &
                  (pixels[:, 1] >= 0) & (pixels[:, 1] < h))
        identities = np.asarray(segment_ids if segment_ids is not None else np.zeros(len(valid), int))
        if identities.shape != valid.shape:
            raise ValueError("Every curve sample needs its segment identity")
        segments, current = [], []
        for i, good in enumerate(valid):
            if current and (not good or identities[i] != identities[current[-1]]):
                segments.append(tuple(current)); current = []
            if good:
                current.append(i)
        if current:
            segments.append(tuple(current))
        return ProjectedCurve(pixels, valid, tuple(segments), dict(
            geometry_id=str(geometry_id), model_revision=str(model_revision),
            source_binding=binding, native_hw=[int(h), int(w)],
            image_domain="native_brown_pixels", prior_origin="actual_camera_world_curve",
            camera_domain="native_brown14_with_fixed_zero_sensor_tilt",
            camera25_sha256=hashlib.sha256(np.asarray(camera25, dtype="<f8").tobytes()).hexdigest(),
            world_curve_sha256=hashlib.sha256(np.asarray(world_xyz, dtype="<f8").tobytes()).hexdigest(),
            valid_definition="finite_positive_central_camera_domain",
            offscreen_endpoints_preserved_for_line_clipping=True,
            is_training_label=False, accuracy_validated=False), inside)

    def rectify(self, native_uv, camera25, *, native_hw):
        if camera25 is None:
            raise ValueError("Undistortion requires the actual source camera")
        h, w = native_hw
        if any(not isinstance(v, (int, np.integer)) or v <= 0 for v in (h, w)):
            raise ValueError("Native height/width must be positive integers")
        return self._bridge().native_to_rectified(native_uv, camera25, native_size_wh=(w, h))


def _primitive_design_rank(design, singular):
    """Reuse the fitted SVD with the library's default numerical rank rule.

    Close-to-threshold designs retain matrix_rank's original decomposition.
    This changes diagnostic cost, never the set of source measurement pixels.
    """
    torch = _torch()
    tolerance = singular[0] * max(design.shape) * torch.finfo(design.dtype).eps
    rank = (singular > tolerance).sum()
    margin = ((singular - tolerance).abs() /
              tolerance.clamp_min(torch.finfo(design.dtype).tiny)).min()
    rank_value, relative_margin = torch.stack((rank.to(singular.dtype), margin)).detach().cpu().tolist()
    if not np.isfinite(relative_margin) or relative_margin <= .2:
        return int(torch.linalg.matrix_rank(design)), "original_matrix_rank_near_threshold"
    return int(rank_value), "existing_SVD_default_library_rank_tolerance"


def primitive_diagnostic(points, *, kind="conic", camera25=None, native_hw=None):
    """GPU line/conic Sampson shape residual, explicitly a fitted hypothesis.

    With a camera the existing native-ray inverse supplies rectified pixels.
    Without one this is only a canonical image conic; Brown pixels need not form
    an exact conic. A conic fit cannot establish the identity/radius of a world
    circle, and no fitted pixel is promoted to a calibration observation.
    """
    torch = _torch()
    values = _cuda_tensor(points, dimensions=2)
    if values.shape[1] != 2 or kind not in ("line", "conic", "circle"):
        raise ValueError("Expected Nx2 pixels and a line/conic/circle hypothesis")
    valid = torch.isfinite(values).all(dim=1)
    domain = "canonical_image_pixels_without_camera"
    if camera25 is not None:
        if native_hw is None:
            raise ValueError("Actual native resolution is required for camera rectification")
        rectified, good = CameraShapeAdapter().rectify(values.detach().cpu().numpy(), camera25, native_hw=native_hw)
        values = torch.as_tensor(rectified, device=points.device, dtype=torch.float32)
        valid &= torch.as_tensor(good, device=points.device)
        domain = "actual_camera_rectified_pixels"
    p = values[valid].float()
    minimum = 3 if kind == "line" else 6
    if len(p) < minimum:
        raise ValueError("Insufficient finite source samples for the primitive hypothesis")
    center = p.mean(dim=0)
    scale = torch.linalg.vector_norm(p-center, dim=1).square().mean().sqrt()
    if float(scale) < 1e-6:
        raise ValueError("Primitive samples collapse to one image location")
    xy = (p-center)/scale
    x, y = xy[:, 0], xy[:, 1]
    one = torch.ones_like(x)
    design = (torch.stack((x, y, one), 1) if kind == "line" else
              torch.stack((x*x, x*y, y*y, x, y, one), 1))
    _, singular, vh = torch.linalg.svd(design, full_matrices=False)
    # A conic needs five independent constraints; a line needs two.
    rank, rank_method = _primitive_design_rank(design, singular)
    if rank < design.shape[1]-1:
        raise ValueError("Primitive design is degenerate")
    coefficients = vh[-1]
    algebraic = design @ coefficients
    if kind == "line":
        denominator = torch.linalg.vector_norm(coefficients[:2]).expand_as(x)
        ellipse = None
    else:
        a, b, c, d, e, _ = coefficients
        denominator = torch.sqrt((2*a*x+b*y+d).square() + (b*x+2*c*y+e).square())
        ellipse = bool(b*b-4*a*c < 0)
    residual = algebraic.abs()/denominator.clamp_min(1e-8)*scale
    return dict(residual_px=residual, rms_px=residual.square().mean().sqrt(),
                coefficients_normalized=coefficients, normalization_center=center,
                normalization_scale=scale, singular_values=singular,
                valid_source_indices=torch.nonzero(valid).flatten(), ellipse_hypothesis=ellipse,
                numerical_rank=rank, rank_evaluation=rank_method,
                image_domain=domain, kind=kind, diagnostic_only=True,
                world_circle_verified=False, localization_accuracy_validated=False)


def measure_boundary_profiles(rgb, class_probabilities, points, normals, *,
                              class_index, search_radius=16., sample_step=.5,
                              rgb_source_binding, probability_source_binding,
                              projection_source_binding, segment_ids=None,
                              shadow_probabilities=None, shadow_source_binding=None,
                              minimum_contrast=.04, maximum_edge_width=3.,
                              minimum_typed_probability=.2, ambiguity_ratio=.6):
    """Sample actual native RGB and typed probabilities along projected normals.

    Returns CUDA profiles and separate candidate measurements in native pixels.
    An RGB edge cannot establish paint identity; a learned probability cannot
    establish an accurate boundary. No candidate is a label or an accepted
    localization. Segment IDs are copied, never joined or fitted across gaps;
    absent IDs mean independent point samples with no inferred connectivity.
    """
    torch = _torch()
    import torch.nn.functional as F
    binding = _same_frame(rgb_source_binding, probability_source_binding)
    _same_frame(rgb_source_binding, projection_source_binding)
    rgb = _cuda_tensor(rgb, dimensions=4)
    probabilities = _cuda_tensor(class_probabilities, dimensions=4)
    points, normals = _cuda_tensor(points, dimensions=2), _cuda_tensor(normals, dimensions=2)
    if rgb.shape[0:2] != (1, 3) or min(rgb.shape[-2:]) < 2:
        raise ValueError("Actual RGB must be [1,3,H,W] in its native source domain")
    h, w = rgb.shape[-2:]
    if probabilities.shape[0] != 1 or probabilities.shape[-2:] != (h, w):
        raise ValueError("Probabilities must share the actual RGB native dimensions")
    if (isinstance(class_index, bool) or not isinstance(class_index, int) or
            not 0 <= class_index < probabilities.shape[1]):
        raise ValueError("A valid typed class index is required")
    if points.shape[1:] != (2,) or normals.shape != points.shape or not len(points):
        raise ValueError("Projected points and native-pixel normals must be nonempty Nx2")
    if any(t.device != rgb.device for t in (probabilities, points, normals)):
        raise ValueError("All profiles must share one CUDA device")
    for name, value in (("search_radius", search_radius), ("sample_step", sample_step),
                        ("maximum_edge_width", maximum_edge_width)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, in native pixels")
    if sample_step > search_radius:
        raise ValueError("Sampling step cannot exceed the search radius")
    for name, value in (("minimum_contrast", minimum_contrast),
                        ("minimum_typed_probability", minimum_typed_probability),
                        ("ambiguity_ratio", ambiguity_ratio)):
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must lie in (0,1]")
    if tuple(binding.get("native_hw", (h, w))) != (h, w):
        raise ValueError("Source binding dimensions do not describe the actual RGB")
    if binding.get("image_domain", "native_brown_pixels") != "native_brown_pixels":
        raise ValueError("Profiles require actual native Brown pixels, without rectification")
    if binding.get("crop_xyxy") not in (None, [0, 0, w, h], (0, 0, w, h)):
        raise ValueError("A cropped RGB tensor cannot stand in for the full native source frame")
    for name, tensor in (("RGB", rgb), ("class probabilities", probabilities)):
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must contain finite floating values in [0,1]")
        if bool(((tensor < 0) | (tensor > 1)).any()):
            raise ValueError(f"{name} must lie in [0,1]")
    shadow = None
    if shadow_probabilities is not None:
        if shadow_source_binding is None:
            raise ValueError("Independent shadow probabilities need their own source binding")
        _same_frame(rgb_source_binding, shadow_source_binding)
        shadow = _cuda_tensor(shadow_probabilities, dimensions=4)
        if shadow.shape != (1, 1, h, w) or shadow.device != rgb.device:
            raise ValueError("Shadow probabilities must be [1,1,H,W] on the same CUDA device")
        if not bool(torch.isfinite(shadow).all()) or bool(((shadow < 0) | (shadow > 1)).any()):
            raise ValueError("Shadow probabilities must be finite and lie in [0,1]")
    n = len(points)
    if segment_ids is None:
        segments = torch.arange(n, device=rgb.device, dtype=torch.int64)
        connectivity = "independent_samples_no_connectivity_supplied"
    else:
        segments = _cuda_tensor(segment_ids, dimensions=1)
        if segments.shape != (n,) or segments.device != rgb.device or segments.dtype not in (
                torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("Segment IDs must be one integer CUDA identity per projected point")
        segments = segments.clone()
        connectivity = "caller_geometry_segments_preserved_no_cross_segment_fit"
    points = points.float()
    lengths = torch.linalg.vector_norm(normals.float(), dim=1)
    finite = torch.isfinite(points).all(1) & torch.isfinite(normals).all(1) & (lengths > 1e-8)
    unit_normals = normals.float() / lengths.clamp_min(1e-8)[:, None]
    # Include the exact search endpoints and zero; the returned step is actual.
    half_count = int(np.ceil(search_radius / sample_step))
    offsets = torch.linspace(-search_radius, search_radius, 2*half_count+1, device=rgb.device)
    step = float(search_radius / half_count)
    uv = points[:, None] + unit_normals[:, None] * offsets[None, :, None]

    def sample(tensor, coordinates):
        good = (torch.isfinite(coordinates).all(-1) & (coordinates[..., 0] >= 0) &
                (coordinates[..., 0] <= w-1) & (coordinates[..., 1] >= 0) &
                (coordinates[..., 1] <= h-1))
        grid = torch.stack((2*(coordinates[..., 0]+.5)/w-1,
                            2*(coordinates[..., 1]+.5)/h-1), -1)[None]
        grid = torch.nan_to_num(grid, nan=2., posinf=2., neginf=-2.)
        values = F.grid_sample(tensor.float(), grid, align_corners=False,
                               mode="bilinear", padding_mode="zeros")[0].permute(1, 2, 0)
        return values, good

    rgb_values, inside = sample(rgb, uv)
    valid = inside & finite[:, None]
    typed_values, _ = sample(probabilities[:, class_index:class_index+1], uv)
    typed_values = typed_values[..., 0]
    shadow_values = sample(shadow, uv)[0][..., 0] if shadow is not None else None
    rgb_values = rgb_values.masked_fill(~valid[..., None], float("nan"))
    typed_values = typed_values.masked_fill(~valid, float("nan"))
    if shadow_values is not None:
        shadow_values = shadow_values.masked_fill(~valid, float("nan"))
    pair_valid = valid[:, :-1] & valid[:, 1:]
    rgb_gradient = (rgb_values[:, 1:] - rgb_values[:, :-1]).square().mean(-1).sqrt()/step
    gradient_offsets = (offsets[1:] + offsets[:-1])*.5
    rgb_gradient = rgb_gradient.masked_fill(~pair_valid, float("nan"))

    def peak_width(scores, mask, peak_index, floor):
        # Contiguous FWHM of this peak; another component never extends width.
        peak = scores.gather(1, peak_index[:, None])[:, 0]
        half = mask & (scores >= floor[:, None]+(peak-floor)[:, None]*.5)
        position = torch.arange(scores.shape[1], device=rgb.device)[None]
        left = torch.where((position < peak_index[:, None]) & ~half, position, -1).amax(1)
        right = torch.where((position > peak_index[:, None]) & ~half, position,
                            scores.shape[1]).amin(1)
        return (right-left-1).float()*step

    def peaks(scores, mask, score_floor):
        safe = torch.where(mask, scores, -torch.inf)
        maximum, index = safe.max(1)
        floor = torch.where(mask, scores, torch.inf).min(1).values
        has_values = mask.any(1)
        maximum = torch.where(has_values, maximum, float("nan"))
        floor = torch.where(has_values, floor, 0.)
        previous = F.pad(safe[:, :-1], (1, 0), value=-float("inf"))
        following = F.pad(safe[:, 1:], (0, 1), value=-float("inf"))
        local = mask & (safe >= previous) & (safe > following)
        # Distinct peaks must be more than one native pixel from the best peak.
        positions = torch.arange(scores.shape[1], device=rgb.device)[None]
        competing = (local & ((positions-index[:, None]).abs()*step > 1.) &
                     (scores >= maximum[:, None]*ambiguity_ratio) & (scores >= score_floor))
        width = peak_width(scores, mask, index, floor)
        width = torch.where(has_values, width, float("nan"))
        return maximum, index, floor, width, competing.sum(1), has_values

    edge_peak, edge_index, _, edge_width, edge_competitors, edge_has_values = peaks(
        rgb_gradient, pair_valid, minimum_contrast/max(1., maximum_edge_width))
    typed_peak, typed_index, typed_floor, typed_width, typed_competitors, typed_has_values = peaks(
        typed_values, valid, minimum_typed_probability)
    # Edge width is measured above zero gradient, unlike typed probability baseline.
    edge_width = peak_width(torch.nan_to_num(rgb_gradient), pair_valid, edge_index,
                            torch.zeros(n, device=rgb.device))
    edge_width = torch.where(edge_has_values, edge_width, float("nan"))
    rgb_contrast = (torch.where(valid[..., None], rgb_values, -torch.inf).amax(1) -
                    torch.where(valid[..., None], rgb_values, torch.inf).amin(1)).square().mean(-1).sqrt()
    rgb_contrast = torch.where(valid.any(1), rgb_contrast, float("nan"))
    rgb_offset = gradient_offsets[edge_index]
    typed_offset = offsets[typed_index]
    rgb_peak_uv = points + unit_normals * rgb_offset[:, None]
    typed_peak_uv = points + unit_normals * typed_offset[:, None]
    rgb_peak_uv = rgb_peak_uv.masked_fill(~edge_has_values[:, None], float("nan"))
    typed_peak_uv = typed_peak_uv.masked_fill(~typed_has_values[:, None], float("nan"))
    native_center_inside = finite & (points[:, 0] >= 0) & (points[:, 0] <= w-1) & (
        points[:, 1] >= 0) & (points[:, 1] <= h-1)

    def patch_sharpness(centers):
        delta = centers.new_tensor([[-1., -1.], [0., -1.], [1., -1.], [-1., 0.],
                                    [0., 0.], [1., 0.], [-1., 1.], [0., 1.], [1., 1.]])
        patch, good = sample(rgb, centers[:, None]+delta)
        gray = patch.mean(-1)
        laplacian = (gray[:, 1]+gray[:, 3]+gray[:, 5]+gray[:, 7]-4*gray[:, 4]).abs()
        energy = ((gray[:, 5]-gray[:, 3]).square()+(gray[:, 7]-gray[:, 1]).square()).sqrt()*.5
        complete = good.all(1) & torch.isfinite(centers).all(1)
        return dict(laplacian=laplacian.masked_fill(~complete, float("nan")),
                    gradient=energy.masked_fill(~complete, float("nan")),
                    variance=gray.var(1, correction=0).masked_fill(~complete, float("nan")),
                    patch_complete=complete)

    flags = dict(invalid_point_or_normal=~finite, point_out_of_frame=~native_center_inside,
                 partial_profile=~valid.all(1), no_rgb_contrast=~edge_has_values | (rgb_contrast < minimum_contrast),
                 blur_review=edge_width > maximum_edge_width,
                 multi_edge_ambiguity=edge_competitors > 0,
                 typed_weak=~typed_has_values | (typed_peak < minimum_typed_probability),
                 typed_multimodal=typed_competitors > 0,
                 typed_underdetermined=(typed_peak-typed_floor < .05) | (typed_width > search_radius),
                 peak_at_search_boundary=(edge_index == 0) | (edge_index == len(gradient_offsets)-1))
    edge_shadow = typed_shadow = None
    if shadow is not None:
        edge_shadow = sample(shadow, rgb_peak_uv[:, None])[0][:, 0, 0]
        typed_shadow = sample(shadow, typed_peak_uv[:, None])[0][:, 0, 0]
        edge_shadow = edge_shadow.masked_fill(~edge_has_values, float("nan"))
        typed_shadow = typed_shadow.masked_fill(~typed_has_values, float("nan"))
        flags["shadow_review_at_rgb_peak"] = edge_shadow >= .5
        flags["shadow_review_at_typed_peak"] = typed_shadow >= .5
    rgb_candidate = (native_center_inside & ~flags["partial_profile"] & ~flags["no_rgb_contrast"] &
                     ~flags["blur_review"] & ~flags["multi_edge_ambiguity"] & ~flags["peak_at_search_boundary"])
    typed_candidate = (native_center_inside & ~flags["partial_profile"] & ~flags["typed_weak"] &
                       ~flags["typed_multimodal"] & ~flags["typed_underdetermined"])
    return dict(source_binding=binding, points_native_uv=points, normals_unit=unit_normals,
                segment_ids=segments, offsets_native_px=offsets, sampled_native_uv=uv,
                rgb_profile=rgb_values, typed_probability_profile=typed_values,
                shadow_probability_profile=shadow_values, valid_mask=valid,
                rgb_gradient_profile=rgb_gradient, rgb_gradient_valid_mask=pair_valid,
                rgb_gradient_offsets_native_px=gradient_offsets,
                rgb_peak=dict(uv=rgb_peak_uv, offset_px=rgb_offset, gradient=edge_peak,
                              contrast=rgb_contrast, fwhm_px=edge_width, competing_peaks=edge_competitors,
                              candidate_evidence_usable=rgb_candidate, shadow_probability=edge_shadow),
                typed_peak=dict(uv=typed_peak_uv, offset_px=typed_offset, probability=typed_peak,
                                fwhm_px=typed_width, competing_peaks=typed_competitors,
                                candidate_evidence_usable=typed_candidate, shadow_probability=typed_shadow),
                local_sharpness_at_projection=patch_sharpness(points),
                local_sharpness_at_rgb_peak=patch_sharpness(rgb_peak_uv), flags=flags,
                metadata=dict(native_hw=[h, w], class_index=class_index, sample_step_native_px=step,
                              search_radius_native_px=float(search_radius), segment_policy=connectivity,
                              minimum_contrast=minimum_contrast, maximum_edge_width_native_px=maximum_edge_width,
                              minimum_typed_probability=minimum_typed_probability, ambiguity_ratio=ambiguity_ratio,
                              source_evidence_only=True, identity_verified=False, boundary_accuracy_verified=False,
                              is_training_label=False, camera_projection_is_candidate_prior=True))


def boundary_profile_sanity_checks():
    """Actual CUDA checks for edge units, blur, ambiguity and absent evidence."""
    torch = _torch()
    h, w = 80, 96
    xx = torch.arange(w, device="cuda", dtype=torch.float32)[None, None, None]
    binding = dict(source_sha256="a"*64, source_pts=3727*256, time_base="1/12800",
                   shot_id="synthetic_only", native_hw=[h, w], image_domain="native_brown_pixels")
    probabilities = torch.exp(-.5*((xx-48)/1.5).square()).expand(1, 1, h, w).contiguous()
    point = torch.tensor([[44., 40.]], device="cuda")
    normal = torch.tensor([[2., 0.]], device="cuda")
    arguments = dict(class_index=0, search_radius=12., rgb_source_binding=binding,
                     probability_source_binding=binding, projection_source_binding=binding)

    def measure(image, *, source_points=point, **extra):
        return measure_boundary_profiles(image.expand(1, 3, h, w).contiguous(), probabilities,
                                         source_points, normal.expand_as(source_points), **arguments, **extra)

    sharp = measure((xx >= 48).float())
    if abs(float(sharp["rgb_peak"]["uv"][0, 0])-47.5) > .26 or abs(
            float(sharp["typed_peak"]["uv"][0, 0])-48.) > .01:
        raise AssertionError("RGB/typed peaks were confused or native normal pixel units changed")
    if not bool(sharp["rgb_peak"]["candidate_evidence_usable"][0]) or not float(
            sharp["rgb_peak"]["fwhm_px"][0]) <= 1.5:
        raise AssertionError("A sharp single edge was incorrectly measured as blurred")
    blur = measure(torch.sigmoid((xx-48)/3.))
    if not bool(blur["flags"]["blur_review"][0]) or bool(
            blur["rgb_peak"]["candidate_evidence_usable"][0]):
        raise AssertionError("A broad blurry transition was accepted as a sharp candidate")
    two = measure(((xx >= 44) & (xx < 52)).float())
    if not bool(two["flags"]["multi_edge_ambiguity"][0]) or bool(
            two["rgb_peak"]["candidate_evidence_usable"][0]):
        raise AssertionError("Two equally strong boundaries lost their ambiguity")
    outside = measure((xx >= 48).float(), source_points=torch.tensor([[-40., 40.]], device="cuda"))
    if bool(outside["valid_mask"].any()) or bool(outside["rgb_peak"]["candidate_evidence_usable"].any()):
        raise AssertionError("Out-of-image padding created source evidence")
    uniform = measure(torch.ones_like(xx)*.5)
    if not bool(uniform["flags"]["no_rgb_contrast"][0]) or bool(
            uniform["rgb_peak"]["candidate_evidence_usable"][0]):
        raise AssertionError("Uniform RGB was promoted by a typed probability peak")
    shadow = measure((xx >= 48).float(), shadow_probabilities=torch.ones((1, 1, h, w), device="cuda"),
                     shadow_source_binding=binding)
    if not bool(shadow["flags"]["shadow_review_at_rgb_peak"][0]):
        raise AssertionError("Independent shadow evidence was not retained")
    try:
        measure_boundary_profiles((xx >= 48).float().expand(1, 3, h, w), probabilities, point, normal,
                                  **{**arguments, "projection_source_binding": {**binding, "source_pts": 0}})
    except ValueError:
        pass
    else:
        raise AssertionError("A projection from another source exposure was accepted")
    return dict(sharp_rgb_peak_native_uv=sharp["rgb_peak"]["uv"][0].tolist(),
                sharp_typed_peak_native_uv=sharp["typed_peak"]["uv"][0].tolist(),
                sharp_fwhm_native_px=float(sharp["rgb_peak"]["fwhm_px"][0]),
                blur_fwhm_native_px=float(blur["rgb_peak"]["fwhm_px"][0]),
                two_edge_competitors=int(two["rgb_peak"]["competing_peaks"][0]),
                out_of_image_checked=True, low_contrast_checked=True, shadow_checked=True,
                source_exposure_mismatch_checked=True, device=str(point.device), synthetic_only=True,
                localization_accuracy_verified=False)


def shape_loss(logits, target, valid, *, target_origin, prediction_source_binding,
               target_source_binding, geometry_is_verified=False,
               allow_unverified_model_prior=False):
    """Masked shape BCE. An explicit model prior is only a regularizer.

    Targets must already share the prediction's source pixel domain. This does
    not warp or manufacture labels. Model priors require explicit caller consent
    to using unverified geometry, and remain distinct from source annotations.
    """
    torch = _torch()
    import torch.nn.functional as F
    logits = _cuda_tensor(logits, dimensions=4)
    target, valid = _cuda_tensor(target, dimensions=4), _cuda_tensor(valid, dimensions=4)
    _same_frame(prediction_source_binding, target_source_binding)
    if target_origin not in ("source_annotation", "camera_model_prior"):
        raise ValueError("Declare source_annotation or camera_model_prior")
    if target_origin == "camera_model_prior" and not (geometry_is_verified or allow_unverified_model_prior):
        raise ValueError("An unverified camera prior needs an explicit regularizer flag")
    if logits.shape != target.shape or valid.shape not in (logits.shape, (logits.shape[0], 1, *logits.shape[-2:])):
        raise ValueError("Shape target and validity must share the prediction pixel domain")
    if logits.device != target.device or logits.device != valid.device:
        raise ValueError("Shape tensors must share a CUDA device")
    if valid.dtype != torch.bool:
        raise ValueError("Validity is a boolean source mask, not a confidence score")
    mask = valid.bool().expand_as(logits)
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(target[mask]).all()) or bool(((target[mask] < 0) | (target[mask] > 1)).any()):
        raise ValueError("Shape logits/valid target values must be finite; target lies in [0,1]")
    safe_target = torch.where(mask, target.float(), torch.zeros_like(target, dtype=torch.float32))
    loss = F.binary_cross_entropy_with_logits(logits.float(), safe_target, reduction="none")
    return (loss * mask).sum()/mask.sum().clamp_min(1)


def gpu_sanity_checks(weights_path=None) -> dict:
    """Small reproducible synthetic checks; no optimizer, files or accuracy claim."""
    torch = _torch()
    h, w = 12, 16
    ramp = torch.arange(w, device="cuda", dtype=torch.float32)[None, None, None].expand(1, 1, h, w)
    backward = torch.zeros(1, 2, h, w, device="cuda"); backward[:, 0] = -2
    sampled, valid = warp(ramp, backward)
    if not torch.allclose(sampled[:, :, :, 2:], ramp[:, :, :, :-2], atol=1e-5):
        raise AssertionError("Warp direction is reversed")
    if bool(valid[:, :, :, :2].any()) or not bool(valid[:, :, :, 2:].all()):
        raise AssertionError("Warp edge validity is wrong")
    forward = -backward
    consistency, _ = forward_backward_validity(forward, backward, absolute_px=.01, relative=0)
    if not bool(consistency[:, :, :, :-2].all()) or bool(consistency[:, :, :, -2:].any()):
        raise AssertionError("Forward/backward validity misses the target boundary")
    disagreement, _ = forward_backward_validity(forward, backward*.5, absolute_px=.01, relative=0)
    if bool(disagreement.any()):
        raise AssertionError("Forward/backward disagreement was accepted")
    generator = torch.Generator(device="cuda").manual_seed(13)
    image0 = torch.rand(1, 3, 256, 320, generator=generator, device="cuda")
    image0 = torch.nn.functional.avg_pool2d(image0, 5, stride=1, padding=2)
    image1 = torch.zeros_like(image0); image1[:, :, :, 8:] = image0[:, :, :, :-8]
    engine = GPUFlow(weights_path)
    pair = engine.compute(image0, image1, analysis_hw=(128, 160), crop_xyxy=(16, 24, 304, 232))
    center = pair.forward[:, :, 64:192, 64:256]
    dx, dy = float(center[:, 0].median()), float(center[:, 1].median())
    if pair.forward.shape != (1, 2, 256, 320) or abs(dx-8) > 1 or abs(dy) > 1:
        raise AssertionError("RAFT synthetic translation does not retain native pixel units")
    if bool(pair.valid_forward[:, :, :24].any()):
        raise AssertionError("Flow outside the source crop was accepted")
    # Closed-form flow tests scale restoration independently of learned RAFT accuracy.
    real_model = engine.model
    class ConstantFlow(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros((), device="cuda"), requires_grad=False)
            self.calls = 0
        def forward(self, first, second, num_flow_updates):
            self.calls += 1
            sign = 1 if self.calls == 1 else -1
            value = first.new_zeros((len(first), 2, *first.shape[-2:]))
            value[:, 0], value[:, 1] = 4*sign, 3*sign
            return [value]
    try:
        engine.model = ConstantFlow()
        scaled = engine.compute(image0, image1, analysis_hw=(123, 155), crop_xyxy=(16, 24, 304, 232))
        expected = scaled.forward.new_tensor([4*288/155, 3*208/123])
        if not torch.allclose(scaled.forward[0, :, 100, 100], expected, atol=1e-5):
            raise AssertionError("Anisotropic native flow scaling is wrong")
        if scaled.metadata["analysis_padding_ltrb"] != [0, 0, 5, 5]:
            raise AssertionError("RAFT padding metadata is wrong")
    finally:
        engine.model = real_model
    return dict(warp_direction_checked=True, fb_disagreement_checked=True,
                crop_padding_anisotropic_units_checked=True,
                raft_translation_median_native_px=[dx, dy],
                raft_center_valid_fraction=float(pair.valid_forward[:, :, 64:192, 64:256].float().mean()),
                weights_sha256=engine.weights_sha256, synthetic_only=True,
                real_video_accuracy_validated=False)


def measure_paint_band(
    profiles,
    *,
    signal,
    physical_role,
    source_binding,
    range_source_binding,
    probabilities_conditioned_on_geometry=True,
    minimum_signal_contrast=.04,
    minimum_edge_gradient=.01,
    maximum_edge_width_px=4.,
    minimum_type_probability=.2,
    maximum_shadow_probability=.5,
    ambiguity_ratio=.6,
):
    """Two measured colour transitions for a declared painted-stroke center.

    Input profiles come from native RGB ranges, optionally located by a search
    hint. The strongest integrated *source colour* component is read without
    an offset/distance prior. Its positive and negative transition peaks give
    an optical midpoint. Missing, wide or competing bands stay rejected source
    evidence. Filled F/R disk borders must not enter this center-axis method.
    No feature identity, fitted world point or dense pixel label is produced.
    These gates apply only to this automatic measurement; confirmed human
    source positions/labels take priority and are never rejected by this tool.
    """
    torch = _torch()
    import torch.nn.functional as F
    binding = _same_frame(source_binding, range_source_binding)
    _same_frame(binding, profiles["source_binding"])
    if probabilities_conditioned_on_geometry is not False:
        raise ValueError("Paint-band semantic gates require explicitly declared no-prior probabilities")
    if physical_role != "painted_stroke_center_axis":
        raise ValueError("Pair midpoint requires an explicit source stroke-center role; filled-disc borders are single transitions")
    if signal not in ("red_excess", "blue_excess", "dark_band"):
        raise ValueError("Declare an observed source red, blue or dark band, not a schematic palette")
    for name, value in (("minimum_signal_contrast", minimum_signal_contrast),
                        ("minimum_edge_gradient", minimum_edge_gradient),
                        ("maximum_edge_width_px", maximum_edge_width_px)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive finite")
    for name, value in (("minimum_type_probability", minimum_type_probability),
                        ("maximum_shadow_probability", maximum_shadow_probability),
                        ("ambiguity_ratio", ambiguity_ratio)):
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must lie in (0,1]")
    rgb = _cuda_tensor(profiles["rgb_profile"], dimensions=3)
    valid = _cuda_tensor(profiles["valid_mask"], dimensions=2)
    n, k, channels = rgb.shape
    if channels != 3 or valid.shape != (n, k) or k < 3 or not n or valid.dtype != torch.bool:
        raise ValueError("Use measured RGB [N,K,3] profiles with source boolean validity")
    offsets = profiles["offsets_native_px"]
    step = float(profiles["metadata"]["sample_step_native_px"])
    points, normals = profiles["points_native_uv"], profiles["normals_unit"]
    if signal == "red_excess":
        values = rgb[..., 0]-.5*(rgb[..., 1]+rgb[..., 2])
    elif signal == "blue_excess":
        values = rgb[..., 2]-.5*(rgb[..., 0]+rgb[..., 1])
    else:
        values = 1-rgb.mean(-1)
    values = values.masked_fill(~valid, float("nan"))
    baseline = torch.nanquantile(values, .15, dim=1)
    maximum = torch.where(valid, values, -torch.inf).amax(1)
    contrast = maximum-baseline
    band = valid & (values >= baseline[:, None]+contrast[:, None]*.25)
    start = band & ~F.pad(band[:, :-1], (1, 0), value=False)
    component_ids = start.long().cumsum(1)
    row_base = torch.arange(n, device=rgb.device)[:, None]*(k+1)
    score = torch.nan_to_num(values-baseline[:, None]).clamp_min(0)*band
    component_scores = rgb.new_zeros(n*(k+1))
    component_scores.scatter_add_(0, (row_base+component_ids).reshape(-1), score.reshape(-1))
    component_scores = component_scores.reshape(n, k+1)
    component_scores[:, 0] = 0
    best, component = component_scores.max(1)
    others = component_scores.clone().scatter_(1, component[:, None], 0).amax(1)
    selected = band & (component_ids == component[:, None]) & (component[:, None] > 0)
    position = torch.arange(k, device=rgb.device)[None]
    lo = torch.where(selected, position, k).amin(1)
    hi = torch.where(selected, position, -1).amax(1)
    peak_index = torch.where(selected, values, -torch.inf).argmax(1)
    gradient = (values[:, 1:]-values[:, :-1])/step
    pair_valid = valid[:, :-1] & valid[:, 1:]
    grad_position = torch.arange(k-1, device=rgb.device)[None]
    left_mask = pair_valid & (grad_position >= lo[:, None]-1) & (grad_position < peak_index[:, None])
    right_mask = pair_valid & (grad_position >= peak_index[:, None]) & (grad_position <= hi[:, None])
    left_score, left_index = torch.where(left_mask, gradient, -torch.inf).max(1)
    right_score, right_index = torch.where(right_mask, -gradient, -torch.inf).max(1)
    grad_offsets = (offsets[:-1]+offsets[1:])*.5
    left_offset, right_offset = grad_offsets[left_index], grad_offsets[right_index]
    center_offset = (left_offset+right_offset)*.5

    def edge_width(scores, mask, index, height):
        half = mask & (scores >= height[:, None]*.5)
        p = torch.arange(k-1, device=rgb.device)[None]
        left = torch.where((p < index[:, None]) & ~half, p, -1).amax(1)
        right = torch.where((p > index[:, None]) & ~half, p, k-1).amin(1)
        return (right-left-1).float()*step

    left_width = edge_width(gradient, left_mask, left_index, left_score)
    right_width = edge_width(-gradient, right_mask, right_index, right_score)

    def at_offset(values, location):
        t = ((location-offsets[0])/step).clamp(0, k-1)
        index = torch.nan_to_num(t).floor().long().clamp(0, k-2)
        fraction = (t-index).clamp(0, 1)
        row = torch.arange(n, device=rgb.device)
        return values[row, index]*(1-fraction)+values[row, index+1]*fraction

    typed = at_offset(profiles["typed_probability_profile"], center_offset)
    flags = dict(
        partial_source_range=~valid.all(1),
        weak_source_colour=~torch.isfinite(contrast) | (contrast < minimum_signal_contrast) | (best <= 0),
        competing_colour_components=(others > 0) & (others >= best*ambiguity_ratio),
        missing_one_band_edge=(lo <= 0) | (hi >= k-1) | (hi < lo),
        weak_opposite_gradients=~torch.isfinite(left_score) | ~torch.isfinite(right_score)
                                | (left_score < minimum_edge_gradient) | (right_score < minimum_edge_gradient),
        wide_or_blurred_edges=(left_width > maximum_edge_width_px) | (right_width > maximum_edge_width_px),
        nonpositive_band_width=right_offset <= left_offset,
        weak_type_at_optical_center=~torch.isfinite(typed) | (typed < minimum_type_probability),
    )
    shadow = profiles.get("shadow_probability_profile")
    shadow_value = None
    if shadow is not None:
        shadow_value = torch.stack([at_offset(shadow, q) for q in (left_offset, center_offset, right_offset)], 1).amax(1)
        flags["shadow_on_band_or_border"] = ~torch.isfinite(shadow_value) | (shadow_value >= maximum_shadow_probability)
    source_pair_quality = ~torch.stack(tuple(value for name, value in flags.items()
        if name not in ("weak_type_at_optical_center", "shadow_on_band_or_border"))).any(0)
    usable = ~torch.stack(tuple(flags.values())).any(0)
    has_measurement = ((best > 0) & torch.isfinite(left_score) & torch.isfinite(right_score)
                       & (left_score > 0) & (right_score > 0) & (right_offset > left_offset))
    center_uv = (points+normals*center_offset[:, None]).masked_fill(~has_measurement[:, None], float("nan"))
    left_uv = (points+normals*left_offset[:, None]).masked_fill(~has_measurement[:, None], float("nan"))
    right_uv = (points+normals*right_offset[:, None]).masked_fill(~has_measurement[:, None], float("nan"))
    quality = contrast/(contrast+minimum_signal_contrast)/(1+left_width.square()+right_width.square())
    weights = torch.where(usable, torch.nan_to_num(quality).clamp(0, 1), 0.)
    indices = torch.nonzero(usable & (weights > 0), as_tuple=False).flatten()
    return dict(
        source_binding=binding, accepted_indices=indices, source_uv=center_uv[indices],
        source_normals=normals[indices], weights=weights[indices], candidate_mask=usable,
        source_colour_pair_quality_mask=source_pair_quality,
        source_colour_pair_weights=torch.where(source_pair_quality, torch.nan_to_num(quality).clamp(0, 1), 0.),
        all_center_uv=center_uv, left_border_uv=left_uv, right_border_uv=right_uv,
        source_band_width_native_px=right_offset-left_offset,
        left_transition_fwhm_px=left_width, right_transition_fwhm_px=right_width,
        center_offset_native_px=center_offset,
        optical_center_support_interval_px=torch.stack((center_offset-(left_width+right_width)*.25,
                                                       center_offset+(left_width+right_width)*.25), 1),
        source_colour_contrast=contrast, typed_at_optical_center=typed,
        shadow_at_band=shadow_value, rejection_flags=flags, physical_instance_id=None,
        metadata=dict(signal=signal, physical_role=physical_role,
                      position_rule="midpoint_of_opposite_source_colour_gradient_peaks_in_strongest_complete_band",
                      projection_distance_used_for_peak_selection=False,
                      support_interval_is_calibrated_uncertainty=False,
                      input_range_is_search_scope_not_label=True,
                      probabilities_conditioned_on_geometry=False,
                      human_confirmed_supervision_has_priority=True,
                      model_probability_does_not_veto_human_annotation=True,
                      physical_identity_verified=False, is_pixel_ground_truth=False,
                      is_training_label=False, localization_accuracy_validated=False),
    )


def extract_source_primitives(
    rgb,
    class_probabilities,
    *,
    classes,
    rgb_source_binding,
    probability_source_binding,
    probabilities_conditioned_on_geometry=True,
    minimum_rgb_gradient=.02,
    minimum_type_probability=.2,
    maximum_shadow_probability=.5,
    maximum_edge_width_px=4.,
    minimum_component_pixels=12,
    maximum_shape_rms_px=1.5,
    source_region_constraints=None,
):
    """Native-image edge/straight/conic candidates without a world or camera.

    Sobel colour evidence, nonmaximum suppression and measured edge widths run
    on CUDA. Type probabilities only gate source RGB evidence; all semantic
    predictions and all raw/rejected edges remain available. CPU connected
    components only organize actual source pixel indices. Existing CUDA line
    and image-conic diagnostics characterize those pixels, never replacing
    them with fitted samples or supplying physical marker/advertisement IDs.

    This extractor finds RGB BORDER evidence. A stroke center needs an explicit
    source-named paint-band measurement; an edge is not its center. Filled-disk
    borders and painted-stroke axes must not share a supervision definition.
    Every candidate is unverified and is NOT an automatic training label.
    This automatic extractor does not accept/reject user-confirmed annotations;
    their source positions, semantics and primitives remain primary supervision.
    """
    torch = _torch()
    import torch.nn.functional as F
    import cv2

    binding = _same_frame(rgb_source_binding, probability_source_binding)
    if probabilities_conditioned_on_geometry is not False:
        raise ValueError("Source-only extraction requires explicitly declared no-prior probabilities")
    rgb = _cuda_tensor(rgb, dimensions=4)
    probability = _cuda_tensor(class_probabilities, dimensions=4)
    if rgb.shape[0:2] != (1, 3) or min(rgb.shape[-2:]) < 3:
        raise ValueError("Native RGB must be [1,3,H,W]")
    h, w = rgb.shape[-2:]
    names = tuple(classes)
    if (probability.shape != (1, len(names), h, w) or len(set(names)) != len(names)
            or probability.device != rgb.device):
        raise ValueError("No-prior class probabilities must share native dimensions and unique class names")
    if tuple(binding.get("native_hw", (h, w))) != (h, w):
        raise ValueError("Source binding differs from actual native RGB dimensions")
    if binding.get("crop_xyxy") not in (None, [0, 0, w, h], (0, 0, w, h)):
        raise ValueError("Source-only full-frame extraction does not accept a relabeled crop")
    if binding.get("image_domain", "native_brown_pixels") != "native_brown_pixels":
        raise ValueError("Input must retain actual native pixels, without rectification")
    if rgb.dtype == torch.uint8:
        values = rgb.float()/255.
    elif rgb.dtype in (torch.float32, torch.float64):
        if (not bool(torch.isfinite(rgb).all()) or bool(((rgb < 0) | (rgb > 1)).any())
                or not bool(((rgb*255)-(rgb*255).round()).abs().le(1e-3).all())):
            raise ValueError("Floating RGB must retain original uint8 levels normalized by 255")
        values = rgb.float()
    else:
        raise ValueError("Use original uint8 or lossless normalized FP32/FP64 RGB")
    if (not probability.is_floating_point() or not bool(torch.isfinite(probability).all())
            or bool(((probability < 0) | (probability > 1)).any())
            or not bool(torch.allclose(probability.float().sum(1), torch.ones_like(probability[:, 0]), atol=1e-3, rtol=1e-3))):
        raise ValueError("Class inputs must be finite normalized probabilities")
    for name, value in (("minimum_rgb_gradient", minimum_rgb_gradient),
                        ("maximum_edge_width_px", maximum_edge_width_px),
                        ("maximum_shape_rms_px", maximum_shape_rms_px)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive finite")
    for name, value in (("minimum_type_probability", minimum_type_probability),
                        ("maximum_shadow_probability", maximum_shadow_probability)):
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must lie in (0,1]")
    if (isinstance(minimum_component_pixels, bool) or not isinstance(minimum_component_pixels, int)
            or minimum_component_pixels < 6):
        raise ValueError("At least six actual pixels are needed for component shape characterization")
    kernel_x = values.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])/8
    kernel_y = kernel_x.T.contiguous()
    padded = F.pad(values, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, kernel_x[None, None].expand(3, 1, 3, 3), groups=3)
    gy = F.conv2d(padded, kernel_y[None, None].expand(3, 1, 3, 3), groups=3)
    magnitude = (gx.square()+gy.square()).mean(1, keepdim=True).sqrt()
    strongest_channel = (gx.square()+gy.square()).argmax(1, keepdim=True)
    nx, ny = gx.gather(1, strongest_channel), gy.gather(1, strongest_channel)
    length = (nx.square()+ny.square()).sqrt().clamp_min(1e-12)
    nx, ny = nx/length, ny/length
    y, x = torch.meshgrid(torch.arange(h, device=rgb.device, dtype=torch.float32),
                          torch.arange(w, device=rgb.device, dtype=torch.float32), indexing="ij")

    def sample_magnitude(sx, sy):
        grid = torch.stack((2*(sx+.5)/w-1, 2*(sy+.5)/h-1), -1)
        return F.grid_sample(magnitude, grid, align_corners=False, mode="bilinear", padding_mode="border")

    minus = sample_magnitude(x[None]-nx[:, 0], y[None]-ny[:, 0])
    plus = sample_magnitude(x[None]+nx[:, 0], y[None]+ny[:, 0])
    # Keep plateau evidence too: every output UV still indexes an actual source
    # pixel, and later shape diagnostics do not create artificial curve points.
    raw_edges = ((magnitude >= minus) & (magnitude >= plus)
                 & (magnitude >= minimum_rgb_gradient))
    raw_edges[:, :, (0, -1), :] = False
    raw_edges[:, :, :, (0, -1)] = False
    coordinates_yx = torch.nonzero(raw_edges[0, 0], as_tuple=False)
    edge_width = magnitude.new_zeros((1, 1, h, w))
    measured_normals = torch.stack((nx[0, 0, coordinates_yx[:, 0], coordinates_yx[:, 1]],
                                   ny[0, 0, coordinates_yx[:, 0], coordinates_yx[:, 1]]), 1)
    uv = coordinates_yx[:, [1, 0]].float()
    if len(uv):
        radius = max(4, int(np.ceil(maximum_edge_width_px))+1)
        offsets = torch.arange(-radius, radius+1, device=rgb.device, dtype=torch.float32)
        samples = uv[:, None]+measured_normals[:, None]*offsets[None, :, None]
        profile = sample_magnitude(samples[..., 0][None], samples[..., 1][None])[0, 0]
        center_value = profile[:, radius]
        half = profile >= center_value[:, None]*.5
        position = torch.arange(len(offsets), device=rgb.device)[None]
        left = torch.where((position < radius) & ~half, position, -1).amax(1)
        right = torch.where((position > radius) & ~half, position, len(offsets)).amin(1)
        width = (right-left-1).float()
        edge_width[0, 0, coordinates_yx[:, 0], coordinates_yx[:, 1]] = width
    blur_rejected = raw_edges & (edge_width > maximum_edge_width_px)
    shadow_available = "shadow_or_reflection" in names
    shadow = (probability[:, names.index("shadow_or_reflection"):names.index("shadow_or_reflection")+1]
              if shadow_available else probability.new_zeros((1, 1, h, w)))
    shadow_rejected = raw_edges & (shadow >= maximum_shadow_probability)
    quality_edges = raw_edges & ~blur_rejected & ~shadow_rejected
    type_support = F.max_pool2d(probability.float(), 3, stride=1, padding=1)
    region_masks, region_rejected_masks = {}, {}
    region_metadata = None
    if source_region_constraints is not None:
        if not isinstance(source_region_constraints, Mapping):
            raise ValueError("Source region constraints require explicit source-bound masks")
        _same_frame(binding, source_region_constraints["source_binding"])
        if source_region_constraints.get("world_or_camera_used", False):
            raise ValueError("Object regions must come from this source image, not a projected mask")
        for name, mask in source_region_constraints.get("masks", {}).items():
            if name not in ("ice_contact", "board_top", "ice_paint", "ad_boundary"):
                raise ValueError("Region constraint has an unsupported source role")
            mask = _cuda_tensor(mask, dimensions=4)
            if mask.shape != (1, 1, h, w) or mask.device != rgb.device or mask.dtype != torch.bool:
                raise ValueError("Region search masks must be native-size Boolean CUDA tensors")
            region_masks[name] = mask
        region_metadata = dict(source_region_constraints.get("metadata", {}))
    evidence_masks, components = {}, []
    for name in ("ice_contact", "board_top", "ice_paint", "ad_boundary"):
        if name not in names:
            continue
        evidence = quality_edges & (type_support[:, names.index(name):names.index(name)+1] >= minimum_type_probability)
        if name in region_masks:
            region_rejected_masks[name] = evidence & ~region_masks[name]
            evidence = evidence & region_masks[name]
        evidence_masks[name] = evidence
        cpu_mask = evidence[0, 0].detach().cpu().numpy().astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(cpu_mask, connectivity=8)
        component_slices = []
        for component in range(1, count):
            if int(stats[component, cv2.CC_STAT_AREA]) < minimum_component_pixels:
                continue
            x0, y0, cw, ch = (int(v) for v in stats[component, :4])
            yy, xx = np.nonzero(labels[y0:y0+ch, x0:x0+cw] == component)
            component_slices.append((component, [x0, y0, x0+cw, y0+ch],
                                     np.column_stack((xx+x0, yy+y0))))
        for component, bounds, cpu_uv in component_slices:
            source_pixels = torch.as_tensor(cpu_uv, device=rgb.device, dtype=torch.float32)
            line = conic = None
            try:
                line = primitive_diagnostic(source_pixels, kind="line")
            except ValueError:
                pass
            try:
                conic = primitive_diagnostic(source_pixels, kind="conic")
            except ValueError:
                pass
            nondegenerate_ellipse = False
            if conic is not None and conic["ellipse_hypothesis"]:
                coefficient = conic["coefficients_normalized"].double()
                a, b, c, d, e, f = coefficient
                quadratic = torch.stack((torch.stack((a, b*.5)), torch.stack((b*.5, c))))
                eigenvalues = torch.linalg.eigvalsh(quadratic)
                ratio = float(eigenvalues.abs().min()/eigenvalues.abs().max().clamp_min(1e-30))
                # Two parallel edge rows can yield a nearly degenerate conic
                # with a tiny negative discriminant due to FP32 rounding.
                # That numerical artifact is not an observed ellipse.
                if ratio > 1e-6 and bool((eigenvalues[0]*eigenvalues[1]) > 0):
                    linear = torch.stack((d, e))
                    center = -.5*torch.linalg.solve(quadratic, linear)
                    translated_constant = f+.5*(linear@center)
                    radii_squared = -translated_constant/eigenvalues
                    nondegenerate_ellipse = bool((radii_squared > 0).all())
                conic["nondegenerate_real_ellipse"] = nondegenerate_ellipse
                conic["quadratic_eigenvalue_ratio"] = ratio
            line_good = line is not None and float(line["rms_px"]) <= maximum_shape_rms_px
            conic_good = (conic is not None and nondegenerate_ellipse
                          and float(conic["rms_px"]) <= maximum_shape_rms_px)
            kind = ("straight_edge_fragment" if line_good and not conic_good else
                    "image_conic_candidate" if conic_good and not line_good else
                    "straight_or_conic_ambiguous" if line_good and conic_good else
                    "unclassified_source_edge_component")
            index = source_pixels.long()
            components.append(dict(
                source_component_index=int(component), typed_class_candidate=name,
                physical_instance_id=None, physical_ring_or_band_role=None,
                candidate_kind=kind, source_support_uv=source_pixels,
                source_rgb=values[0, :, index[:, 1], index[:, 0]].T,
                rgb_gradient=magnitude[0, 0, index[:, 1], index[:, 0]],
                edge_width_native_px=edge_width[0, 0, index[:, 1], index[:, 0]],
                source_bbox_xyxy=bounds, line_hypothesis=line, image_conic_hypothesis=conic,
                fitted_points_used_as_observations=False, is_training_label=False,
                identity_verified=False, boundary_accuracy_verified=False,
            ))
    return dict(
        source_binding=binding, native_hw=[h, w], classes=names,
        semantic_probabilities=probability, semantic_argmax=probability.argmax(1),
        raw_rgb_gradient=magnitude, raw_rgb_edge_mask=raw_edges,
        measured_edge_width_native_px=edge_width,
        blur_rejection_mask=blur_rejected, shadow_rejection_mask=shadow_rejected,
        typed_source_edge_masks=evidence_masks, components=components,
        source_region_rejection_masks=region_rejected_masks,
        source_region_constraint_metadata=region_metadata,
        source_red_excess=values[:, 0:1]-.5*(values[:, 1:2]+values[:, 2:3]),
        source_blue_excess=values[:, 2:3]-.5*(values[:, 0:1]+values[:, 1:2]),
        metadata=dict(
            input_domain="unmodified_native_RGB", probabilities_conditioned_on_geometry=False,
            world_camera_projection_input_used=False, GPU_source_gradient_and_width=True,
            topology_backend="CPU_connected_components_of_actual_GPU_selected_source_pixels",
            shape_backend="existing_CUDA_source_pixel_line_and_image_conic_diagnostics",
            minimum_rgb_gradient=float(minimum_rgb_gradient),
            minimum_type_probability=float(minimum_type_probability),
            maximum_shadow_probability=float(maximum_shadow_probability),
            maximum_edge_width_native_px=float(maximum_edge_width_px),
            minimum_component_pixels=int(minimum_component_pixels),
            shadow_filter_available=shadow_available, all_semantic_predictions_retained=True,
            rejected_RGB_evidence_retained=True, physical_ids_inferred=False,
            source_region_search_constraints_used=bool(region_masks),
            source_region_masks_are_pixel_truth=False,
            human_confirmed_supervision_has_priority=True,
            model_probability_does_not_veto_human_annotation=True,
            image_conic_is_verified_world_circle=False, is_pixel_ground_truth=False,
            is_training_label=False, localization_accuracy_validated=False,
        ),
    )


def select_profile_correspondences(
    profiles,
    *,
    source_binding,
    geometry_source_binding,
    point_source_binding,
    probability_is_geometry_conditioned=True,
    independently_verified_point_mask=None,
    minimum_typed_at_rgb_peak=.2,
    maximum_shadow_at_rgb_peak=.5,
    minimum_rgb_gradient=None,
    maximum_edge_width=None,
    source_base_weights=None,
):
    """Filter measured source profiles without a distance-to-projection score.

    Position is the strongest RGB-gradient peak over the entire supplied search
    interval, never a typed peak or the edge nearest its projected center. Type
    probabilities are sampled at that RGB position. Geometry-conditioned type
    probabilities are refused; the caller must explicitly declare no-prior
    inference. Quality weights use source RGB contrast, steepness and width,
    not neural confidence or offset size. Ambiguity/blur/shadow flags are kept.

    Every accepted output is a NORMAL-search hypothesis. It can enter the
    normal-residual mode of refine_physical_view_from_source_points, but only an
    independently verified point mask can authorize its vector-point mode.
    Neither mode nor this selection establishes identity or pixel accuracy.
    """
    torch = _torch()
    binding = _same_frame(source_binding, geometry_source_binding)
    _same_frame(binding, point_source_binding)
    _same_frame(binding, profiles["source_binding"])
    if probability_is_geometry_conditioned is not False:
        raise ValueError("Independent semantic filtering requires explicitly declared no-prior probabilities")
    metadata = profiles["metadata"]
    if not metadata.get("source_evidence_only") or metadata.get("is_training_label"):
        raise ValueError("Use actual source profiles, not a projected target or training label")
    gradient = _cuda_tensor(profiles["rgb_gradient_profile"], dimensions=2)
    gradient_valid = _cuda_tensor(profiles["rgb_gradient_valid_mask"], dimensions=2)
    typed = _cuda_tensor(profiles["typed_probability_profile"], dimensions=2)
    points = _cuda_tensor(profiles["points_native_uv"], dimensions=2)
    normals = _cuda_tensor(profiles["normals_unit"], dimensions=2)
    offsets = _cuda_tensor(profiles["rgb_gradient_offsets_native_px"], dimensions=1)
    n, k = gradient.shape
    if (not n or gradient_valid.shape != (n, k) or typed.shape != (n, k+1)
            or points.shape != (n, 2) or normals.shape != (n, 2) or offsets.shape != (k,)):
        raise ValueError("Measured RGB and typed profiles must share their original sampling axes")
    if any(t.device != gradient.device for t in (gradient_valid, typed, points, normals, offsets)):
        raise ValueError("Profile selection must use one CUDA device")
    if gradient_valid.dtype != torch.bool:
        raise ValueError("RGB profile validity must be a boolean source mask")
    for name, value in (("minimum_typed_at_rgb_peak", minimum_typed_at_rgb_peak),
                        ("maximum_shadow_at_rgb_peak", maximum_shadow_at_rgb_peak)):
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must lie in (0,1]")
    width_limit = float(metadata["maximum_edge_width_native_px"] if maximum_edge_width is None
                        else maximum_edge_width)
    gradient_minimum = float(metadata["minimum_contrast"] / max(1., width_limit)
                             if minimum_rgb_gradient is None else minimum_rgb_gradient)
    if not np.isfinite(width_limit) or width_limit <= 0 or not np.isfinite(gradient_minimum) or gradient_minimum <= 0:
        raise ValueError("Source edge width and gradient limits must be positive finite")
    # No projected-center offset, geometry residual, or distance prior enters argmax.
    safe = torch.where(gradient_valid & torch.isfinite(gradient), gradient, -torch.inf)
    rgb_gradient, peak_index = safe.max(1)
    rgb_offset = offsets[peak_index]
    candidate_uv = points + normals * rgb_offset[:, None]
    row = torch.arange(n, device=gradient.device)
    # RGB gradients are between adjacent half-pixel profile samples.
    typed_at_rgb = (typed[row, peak_index] + typed[row, peak_index+1]) * .5
    peak = profiles["rgb_peak"]
    width = _cuda_tensor(peak["fwhm_px"], dimensions=1)
    contrast = _cuda_tensor(peak["contrast"], dimensions=1)
    sharpness = profiles["local_sharpness_at_rgb_peak"]
    flags = profiles["flags"]
    rejected = {name: _cuda_tensor(flags[name], dimensions=1).bool().clone() for name in (
        "invalid_point_or_normal", "point_out_of_frame", "partial_profile",
        "no_rgb_contrast", "multi_edge_ambiguity", "peak_at_search_boundary", "typed_multimodal")}
    typed_finite = torch.isfinite(typed)
    typed_range = (torch.where(typed_finite, typed, -torch.inf).amax(1)
                   - torch.where(typed_finite, typed, torch.inf).amin(1))
    # A flat type plateau has no competing location modes. It can gate an
    # independently sharp RGB edge but never determine the edge position.
    rejected["typed_multimodal"] &= typed_range >= .05
    rejected.update(
        nonfinite_rgb_peak=~torch.isfinite(rgb_gradient) | ~torch.isfinite(candidate_uv).all(1),
        weak_rgb_gradient=rgb_gradient < gradient_minimum,
        blur_or_wide_edge=~torch.isfinite(width) | (width > width_limit),
        incomplete_sharpness_patch=~sharpness["patch_complete"].bool(),
        weak_type_at_rgb_peak=~torch.isfinite(typed_at_rgb) | (typed_at_rgb < minimum_typed_at_rgb_peak),
    )
    shadow = peak.get("shadow_probability")
    if shadow is not None:
        shadow = _cuda_tensor(shadow, dimensions=1)
        rejected["shadow_at_rgb_peak"] = ~torch.isfinite(shadow) | (shadow >= maximum_shadow_at_rgb_peak)
    if any(value.shape != (n,) or value.device != gradient.device for value in rejected.values()):
        raise ValueError("Source quality flags must share the original profile point axis")
    usable = ~torch.stack(tuple(rejected.values())).any(0)
    verified = torch.zeros(n, dtype=torch.bool, device=gradient.device)
    if independently_verified_point_mask is not None:
        verified = _cuda_tensor(independently_verified_point_mask, dimensions=1)
        if verified.dtype != torch.bool or verified.shape != (n,) or verified.device != gradient.device:
            raise ValueError("An independent point verification mask must be boolean [N] on the profile CUDA device")
    base = gradient.new_ones(n)
    if source_base_weights is not None:
        base = _cuda_tensor(source_base_weights, dimensions=1).float()
        if base.shape != (n,) or base.device != gradient.device or not bool(torch.isfinite(base).all()) or bool((base < 0).any()):
            raise ValueError("Explicit source base weights must be finite nonnegative CUDA [N]")
    # Dimensionless engineering quality, not calibrated inverse covariance.
    quality = (rgb_gradient / (rgb_gradient + gradient_minimum)
               * contrast / (contrast + float(metadata["minimum_contrast"]))
               / width.clamp_min(1.).square())
    rgb_only = ~torch.stack(tuple(value for name, value in rejected.items() if name not in (
        "typed_multimodal", "weak_type_at_rgb_peak", "shadow_at_rgb_peak"))).any(0)
    rgb_only_weights = torch.where(rgb_only, torch.nan_to_num(quality).clamp(0., 1.) * base, 0.)
    all_weights = torch.where(usable, torch.nan_to_num(quality).clamp(0., 1.) * base, 0.)
    indices = torch.nonzero(usable & (all_weights > 0), as_tuple=False).flatten()
    return dict(
        accepted_indices=indices, source_uv=candidate_uv[indices], source_normals=normals[indices],
        weights=all_weights[indices], candidate_mask=usable, all_candidate_uv=candidate_uv,
        all_weights=all_weights, typed_probability_at_rgb_peak=typed_at_rgb,
        source_rgb_quality_mask=rgb_only, source_rgb_quality_weights=rgb_only_weights,
        rgb_gradient_at_peak=rgb_gradient, rgb_offset_native_px=rgb_offset,
        shadow_probability_at_rgb_peak=shadow, rejection_flags=rejected,
        ready_for_point_refinement_mask=usable & verified & (all_weights > 0),
        segment_ids=profiles["segment_ids"][indices], source_binding=binding,
        geometry_source_binding=dict(geometry_source_binding), point_source_binding=dict(point_source_binding),
        metadata=dict(
            position_rule="strongest_source_RGB_gradient_over_entire_search_interval_no_distance_prior",
            semantic_rule="no_prior_type_probability_sampled_at_RGB_peak_is_only_a_gate",
            weight_rule="source_RGB_contrast_gradient_and_edge_width_not_confidence_or_geometry_distance",
            weights_are_calibrated_inverse_covariance=False, search_window_is_geometry_conditioned=True,
            probability_is_geometry_conditioned=False, shadow_filter_available=shadow is not None,
            minimum_typed_at_rgb_peak=float(minimum_typed_at_rgb_peak),
            maximum_shadow_at_rgb_peak=float(maximum_shadow_at_rgb_peak),
            minimum_rgb_gradient=gradient_minimum, maximum_edge_width_native_px=width_limit,
            correspondence_kind="normal_search_candidate_without_tangent_identity",
            source_point_count=n, accepted_candidate_count=len(indices),
            identity_verified_by_this_tool=False, is_pixel_ground_truth=False,
            localization_accuracy_validated=False, is_training_label=False,
        ),
    )


def refine_physical_view_from_source_points(
    calibration,
    initial_view,
    source_world_points,
    source_uv,
    weights,
    *,
    source_normals=None,
    source_binding,
    geometry_source_binding,
    point_source_binding,
    geometry_id,
    model_revision,
    domain_world_points=None,
    max_iterations=10,
    huber_delta_px=3.0,
    initial_damping=1e-3,
    rank_rtol=1e-10,
    device="cuda",
):
    """Refine only the existing physical camera's four view coordinates.

    Brown projection, domain guards and analytic J_view come exclusively from
    SharedCalibration.project_xyz. Installation/shared optics/world geometry
    stay fixed. Source, geometry and extracted source evidence bindings
    must refer to the same original frame and native pixel domain. Optional
    domain_world_points are existing curve samples used ONLY as domain guards.

    The weighted vector-Huber normal system, rank check, solve and true costs
    use CUDA float64. An unobservable/invalid initial view is not optimized;
    invalid trials increase damping without replacing the last legal view.
    Accepted LM trials strictly decrease true cost, never creating source
    observations or asserting correspondence/localization accuracy. Supplying
    source_normals uses only n dot (projection-source_uv) and n dot J_view:
    profile peaks do not create a tangential point correspondence.
    """
    from calibration.program_tracking.z190_physical_camera import SharedCalibration, ViewState

    torch = _torch()
    gpu = torch.device(device)
    if gpu.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Physical view refinement requires CUDA float64")
    if not isinstance(calibration, SharedCalibration) or not isinstance(initial_view, ViewState):
        raise TypeError("Use the existing z190 SharedCalibration and explicit ViewState")
    binding = _same_frame(source_binding, geometry_source_binding)
    _same_frame(binding, point_source_binding)
    if not geometry_id or not model_revision:
        raise ValueError("Source point refinement requires geometry identity and model revision")
    for record in (binding, geometry_source_binding, point_source_binding):
        if record.get("installation_id", calibration.installation_id) != calibration.installation_id:
            raise ValueError("Source points and physical calibration installation differ")
        if record.get("geometry_id", geometry_id) != geometry_id:
            raise ValueError("Source point geometry identity differs")
        if record.get("model_revision", model_revision) != model_revision:
            raise ValueError("Source point model revision differs")
    native_hw = binding.get("native_hw")
    if native_hw is None or tuple(native_hw) != tuple(calibration.output_mode.final_wh)[::-1]:
        raise ValueError("Point pixels must match the calibration's declared final native domain")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer")
    for name, value in (("huber_delta_px", huber_delta_px), ("initial_damping", initial_damping),
                        ("rank_rtol", rank_rtol)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive finite")

    def cpu_array(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64)

    world, pixels, point_weights = map(cpu_array, (source_world_points, source_uv, weights))
    if (world.ndim != 2 or world.shape[1:] != (3,) or pixels.shape != (len(world), 2)
            or point_weights.shape != (len(world),)):
        raise ValueError("Expected world[N,3], independent source_uv[N,2], weights[N]")
    if not all(np.isfinite(value).all() for value in (world, pixels, point_weights)):
        raise ValueError("World, source pixels and weights must be finite")
    if (point_weights < 0).any():
        raise ValueError("Source weights must be nonnegative")
    normals = None if source_normals is None else cpu_array(source_normals)
    if normals is not None:
        if normals.shape != pixels.shape or not np.isfinite(normals).all():
            raise ValueError("Source normals must be finite [N,2] in native pixels")
        normal_lengths = np.linalg.norm(normals, axis=1)
        if (normal_lengths <= 1e-12).any():
            raise ValueError("A normal residual requires nonzero source normals")
        normals = normals / normal_lengths[:, None]
    h, w = native_hw
    if ((pixels[:, 0] < 0) | (pixels[:, 0] >= w) | (pixels[:, 1] < 0) | (pixels[:, 1] >= h)).any():
        raise ValueError("Independently extracted source points must lie in native pixels")
    guard_world = None if domain_world_points is None else cpu_array(domain_world_points)
    if guard_world is not None and (guard_world.ndim != 2 or guard_world.shape[1:] != (3,)
                                   or not np.isfinite(guard_world).all()):
        raise ValueError("Domain curve guard samples must be finite [N,3] original world points")
    target = torch.as_tensor(pixels, dtype=torch.float64, device=gpu)
    base_weight = torch.as_tensor(point_weights, dtype=torch.float64, device=gpu)
    fixed_normals = None if normals is None else torch.as_tensor(normals, dtype=torch.float64, device=gpu)
    initial_state = initial_view.state4().copy()
    offset = initial_view.effective_pp_offset_px
    trials = []
    accepted_steps = rejected_trials = 0
    observed_rank = 0
    singular_values = []
    before = current = None

    def project(view, with_jacobian):
        result = calibration.project_xyz(view, world, with_jacobian=with_jacobian, require_valid=False)
        uv, depth, valid = result[0], result[1], result[-1]
        if (not np.asarray(valid).all() or not np.isfinite(uv).all()
                or not np.isfinite(depth).all()):
            return None
        if guard_world is not None:
            guard_uv, guard_depth, guard_valid = calibration.project_xyz(
                view, guard_world, with_jacobian=False, require_valid=False)
            if (not np.asarray(guard_valid).all() or not np.isfinite(guard_uv).all()
                    or not np.isfinite(guard_depth).all()):
                return None
        residual_uv = torch.as_tensor(uv, dtype=torch.float64, device=gpu) - target
        residual = residual_uv if fixed_normals is None else (residual_uv * fixed_normals).sum(1)
        length = (torch.linalg.vector_norm(residual, dim=1) if fixed_normals is None
                  else residual.abs())
        delta = float(huber_delta_px)
        rho = torch.where(length <= delta, .5 * length.square(), delta * (length - .5 * delta))
        cost = (base_weight * rho).sum()
        if not bool(torch.isfinite(cost)):
            return None
        jacobian = None
        if with_jacobian:
            raw_jacobian = np.asarray(result[3], dtype=np.float64)
            if raw_jacobian.shape != (2 * len(world), 4) or not np.isfinite(raw_jacobian).all():
                return None
            jacobian = torch.as_tensor(raw_jacobian, dtype=torch.float64, device=gpu)
            if fixed_normals is not None:
                jacobian = (jacobian.reshape(len(world), 2, 4) * fixed_normals[:, :, None]).sum(1)
        return dict(residual=residual, residual_uv=residual_uv, length=length, cost=cost, jacobian=jacobian)

    def rank_of(value):
        robust = base_weight * torch.minimum(
            torch.ones_like(value["length"]),
            float(huber_delta_px) / value["length"].clamp_min(1e-30))
        row_weight = robust.repeat_interleave(2) if fixed_normals is None else robust
        weighted_jacobian = value["jacobian"] * row_weight.sqrt()[:, None]
        sv = torch.linalg.svdvals(weighted_jacobian)
        rank = int((sv > sv.max() * float(rank_rtol)).sum()) if sv.numel() and float(sv.max()) > 0 else 0
        return rank, sv.detach().cpu().tolist(), row_weight

    def finish(view, status, value, reverted=False):
        def metrics(record):
            if record is None:
                return None
            weight_sum = float(base_weight.sum())
            return dict(weighted_huber_cost=float(record["cost"]),
                        weighted_rms_native_px=(float(((base_weight * record["length"].square()).sum()
                                                      / weight_sum).sqrt()) if weight_sum else None),
                        residual_uv_native_px=record["residual_uv"].detach().cpu().numpy(),
                        residual_norm_native_px=record["length"].detach().cpu().numpy(),
                        residual_normal_native_px=(None if fixed_normals is None
                                                   else record["residual"].detach().cpu().numpy()),
                        euclidean_residual_norm_native_px=torch.linalg.vector_norm(
                            record["residual_uv"], dim=1).detach().cpu().numpy())
        camera = calibration.camera25(view)
        return dict(
            candidate_view=view, camera25=camera, source_binding=binding,
            geometry_source_binding=dict(geometry_source_binding),
            point_source_binding=dict(point_source_binding), geometry_id=str(geometry_id),
            model_revision=str(model_revision), status=status, before=metrics(before), after=metrics(value),
            rank=observed_rank, weighted_jacobian_singular_values=singular_values,
            accepted_steps=accepted_steps, rejected_trials=rejected_trials,
            reverted_to_initial=reverted, trial_history=trials,
            initial_state4=initial_state, candidate_state4=view.state4(),
            effective_pp_offset_px_preserved=list(offset),
            domain_scope=("correspondence_and_supplied_original_curve_samples"
                          if guard_world is not None else "correspondence_points_only"),
            domain_guard_count=0 if guard_world is None else len(guard_world),
            numerical_backend="CUDA torch.float64 normal_system_rank_and_solve; original CPU Brown projection",
            optimized_coordinates=["rvec_world_to_camera_x", "rvec_world_to_camera_y",
                                   "rvec_world_to_camera_z", "log_f_ratio"],
            shared_calibration_frozen=True, world_geometry_frozen=True,
            residual_kind="vector_uv" if fixed_normals is None else "fixed_source_normal_scalar",
            source_normals=None if normals is None else normals.copy(),
            tangential_residual_constrained=fixed_normals is None,
            correspondence_kind=("unverified_world_point_correspondence" if fixed_normals is None
                                 else "unverified_curve_profile_candidate_no_tangent_identity"),
            point_correspondence_verified_by_this_tool=False, is_pixel_ground_truth=False,
            localization_accuracy_validated=False, huber_delta_px=float(huber_delta_px),
        )

    before = project(initial_view, True)
    if before is None:
        return finish(initial_view, "invalid_initial_projection_no_refinement", None)
    current = before
    observed_rank, singular_values, row_weight = rank_of(current)
    if observed_rank < 4:
        return finish(initial_view, "unobservable_rank_below_four", before)
    view = initial_view
    state = torch.as_tensor(initial_state, dtype=torch.float64, device=gpu)
    damping = float(initial_damping)
    status = "iteration_limit"
    for iteration in range(max_iterations):
        observed_rank, singular_values, row_weight = rank_of(current)
        if observed_rank < 4:
            return finish(initial_view, "unobservable_rank_below_four_reverted", before, True)
        jacobian = current["jacobian"]
        residual = current["residual"].reshape(-1)
        normal = jacobian.T @ (row_weight[:, None] * jacobian)
        gradient = jacobian.T @ (row_weight * residual)
        diagonal = normal.diag().clamp_min(1e-12)
        try:
            step = torch.linalg.solve(normal + damping * torch.diag(diagonal), -gradient)
        except torch.linalg.LinAlgError:
            rejected_trials += 1
            trials.append(dict(iteration=iteration, accepted=False, reason="gpu_linear_solve_failed",
                               damping=damping))
            damping *= 10.0
            continue
        if not bool(torch.isfinite(step).all()):
            rejected_trials += 1
            trials.append(dict(iteration=iteration, accepted=False, reason="nonfinite_gpu_step",
                               damping=damping))
            damping *= 10.0
            continue
        step_norm = float(torch.linalg.vector_norm(step))
        if step_norm < 1e-10:
            status = "converged_step"
            break
        proposed_state = state + step
        try:
            proposed_view = ViewState.from_state4(
                proposed_state.detach().cpu().numpy(), effective_pp_offset_px=offset)
            trial = project(proposed_view, True)
        except (ValueError, OverflowError, FloatingPointError):
            trial = None
        if trial is None:
            rejected_trials += 1
            trials.append(dict(iteration=iteration, accepted=False, reason="invalid_optical_domain",
                               damping=damping, step_norm=step_norm))
            # The invalid projection never enters the objective or the returned
            # state. Back off the LM step through the unchanged physical guard.
            damping *= 10.0
            continue
        trial_rank, _, _ = rank_of(trial)
        if trial_rank < 4:
            rejected_trials += 1
            trials.append(dict(iteration=iteration, accepted=False, reason="unobservable_trial_rank",
                               trial_rank=trial_rank, damping=damping, step_norm=step_norm))
            damping *= 10.0
            continue
        accepted = float(trial["cost"]) < float(current["cost"])
        trials.append(dict(iteration=iteration, accepted=accepted,
                           before_cost=float(current["cost"]), trial_cost=float(trial["cost"]),
                           damping=damping, step_norm=step_norm))
        if accepted:
            accepted_steps += 1
            state, view, current = proposed_state, proposed_view, trial
            damping = max(damping / 3.0, 1e-12)
            status = "candidate_true_cost_decreased"
        else:
            rejected_trials += 1
            damping *= 10.0
    observed_rank, singular_values, _ = rank_of(current)
    if not accepted_steps and rejected_trials:
        status = "no_cost_decreasing_legal_step_initial"
    return finish(view, status, current)


def guided_source_reading(
    rgb,
    no_prior_probs,
    expected_curves_2d,
    *,
    classes,
    rgb_source_binding,
    probability_source_binding,
    guide_source_binding,
    probabilities_conditioned_on_geometry=True,
    search_radius_px=16.,
    sample_step_px=.5,
    minimum_rgb_contrast=.04,
    maximum_edge_width_px=4.,
    minimum_type_probability=.2,
    maximum_shadow_probability=.5,
    ambiguity_ratio=.6,
    source_exclusion_mask=None,
    exclusion_source_binding=None,
):
    """Read native RGB near source-bound 2D hints, without fitting any camera.

    Each guide has id, kind (a class name), primitive, points_uv, physical_role,
    origin and identity_authority; normals/segment_ids are optional. Origins
    are user_confirmed_visual_supervision or model_projection_search_hint.
    Ordered points imply within-segment connectivity only. Sampling indices
    always refer to the caller's original points, with no resampling.

    Actual output locations come from the strongest RGB evidence over the
    complete supplied normal range. A declared painted_stroke_center_axis also
    needs signal=red_excess/blue_excess/dark_band and two measured transitions.
    Every other boundary uses one transition. Neither guides nor fitted shape
    samples enter actual_source_uv. Gaps remain gaps. Neural type/shadow values
    are disagreement diagnostics, never vetoes of confirmed source annotations
    or of pure-RGB quality. A supplied source exclusion mask describes known
    occluded source pixels; this function does not infer unseen occlusions.

    Projection hints carry expected_id only, unless the caller separately
    declares identity_confirmed=True with a source identity_authority. Human
    annotations retain their supplied native UV, identity and supervision in
    confirmed_source_uv even when this automatic reader cannot measure them.
    An explicitly source-confirmed rgb_transition_polarity may restrict a disk
    border to darkening_along_normal or lightening_along_normal. That restriction
    uses signed actual RGB luminance gradients, never the guide's distance.
    """
    torch = _torch()
    import torch.nn.functional as F
    binding = _same_frame(rgb_source_binding, probability_source_binding)
    _same_frame(binding, guide_source_binding)
    if probabilities_conditioned_on_geometry is not False:
        raise ValueError("Guided RGB reading requires explicitly declared no-prior probabilities")
    rgb = _cuda_tensor(rgb, dimensions=4)
    probabilities = _cuda_tensor(no_prior_probs, dimensions=4)
    if rgb.shape[:2] != (1, 3) or min(rgb.shape[-2:]) < 3:
        raise ValueError("Supply one full native RGB frame [1,3,H,W]")
    h, w = rgb.shape[-2:]
    names = tuple(classes)
    if (not names or len(set(names)) != len(names) or probabilities.shape != (1, len(names), h, w)
            or probabilities.device != rgb.device):
        raise ValueError("Unique class names and native no-prior probabilities must share the RGB domain")
    if tuple(binding.get("native_hw", (h, w))) != (h, w):
        raise ValueError("Native RGB dimensions differ from its source binding")
    if (binding.get("image_domain", "native_brown_pixels") != "native_brown_pixels"
            or binding.get("crop_xyxy") not in (None, [0, 0, w, h], (0, 0, w, h))):
        raise ValueError("Keep the full original native source pixels, without rectification")
    if rgb.dtype == torch.uint8:
        values = rgb.float()/255.
    elif rgb.dtype in (torch.float32, torch.float64):
        if (not bool(torch.isfinite(rgb).all()) or bool(((rgb < 0) | (rgb > 1)).any())
                or not bool(((rgb*255)-(rgb*255).round()).abs().le(1e-3).all())):
            raise ValueError("Floating RGB must preserve original uint8 levels divided by 255")
        values = rgb.float()
    else:
        raise ValueError("Use original uint8 or lossless normalized FP32/FP64 RGB")
    if (not probabilities.is_floating_point() or not bool(torch.isfinite(probabilities).all())
            or bool(((probabilities < 0) | (probabilities > 1)).any())
            or not bool(torch.allclose(probabilities.float().sum(1),
                                      torch.ones_like(probabilities[:, 0], dtype=torch.float32),
                                      atol=1e-3, rtol=1e-3))):
        raise ValueError("Probabilities must be finite, normalized and source-bound")
    exclusion = None
    if source_exclusion_mask is not None:
        if exclusion_source_binding is None:
            raise ValueError("A known source exclusion mask needs an explicit frame binding")
        _same_frame(binding, exclusion_source_binding)
        exclusion = _cuda_tensor(source_exclusion_mask, dimensions=4)
        if exclusion.shape != (1, 1, h, w) or exclusion.device != rgb.device or exclusion.dtype != torch.bool:
            raise ValueError("Known source exclusions must be boolean native CUDA [1,1,H,W]")
    shadow = (probabilities[:, names.index("shadow_or_reflection"):names.index("shadow_or_reflection")+1]
              if "shadow_or_reflection" in names else None)
    readings = []
    global_exclusion = exclusion
    any_exclusion_used = exclusion is not None

    def cuda_array(value, dtype, name, dimensions):
        if isinstance(value, torch.Tensor):
            result = _cuda_tensor(value, dimensions=dimensions)
            if result.device != rgb.device:
                raise ValueError(f"{name} must share the source CUDA device")
            return result.to(dtype=dtype)
        result = torch.as_tensor(value, device=rgb.device, dtype=dtype)
        if result.ndim != dimensions:
            raise ValueError(f"{name} has an invalid number of dimensions")
        return result

    def restrict_source_transition(profiles, polarity):
        # Gray-disc upper/lower transitions can coexist within one search band.
        # Their known source-material transition selects a sign, not a location.
        sign = -1. if polarity == "darkening_along_normal" else 1.
        luminance = profiles["rgb_profile"].mean(-1)
        valid = profiles["rgb_gradient_valid_mask"]
        step = profiles["metadata"]["sample_step_native_px"]
        gradient = (sign*(luminance[:, 1:]-luminance[:, :-1])/step).clamp_min(0)
        safe = torch.where(valid & torch.isfinite(gradient), gradient, -torch.inf)
        height, index = safe.max(1)
        pos = torch.arange(gradient.shape[1], device=rgb.device)[None]
        half = valid & (gradient >= height[:, None]*.5) & (gradient > 0)
        left = torch.where((pos < index[:, None]) & ~half, pos, -1).amax(1)
        right = torch.where((pos > index[:, None]) & ~half, pos, gradient.shape[1]).amin(1)
        width = (right-left-1).float()*step
        previous = F.pad(safe[:, :-1], (1, 0), value=-float("inf"))
        following = F.pad(safe[:, 1:], (0, 1), value=-float("inf"))
        local = valid & (safe >= previous) & (safe > following)
        competing = (local & ((pos-index[:, None]).abs()*step > 1.)
                     & (gradient >= height[:, None]*ambiguity_ratio)
                     & (gradient >= minimum_rgb_contrast/max(1., maximum_edge_width_px))).sum(1)
        profile_valid = profiles["valid_mask"]
        contrast = (torch.where(profile_valid, luminance, -torch.inf).amax(1)
                    - torch.where(profile_valid, luminance, torch.inf).amin(1))
        offset = profiles["rgb_gradient_offsets_native_px"][index]
        location = profiles["points_native_uv"]+profiles["normals_unit"]*offset[:, None]
        good = valid.any(1) & torch.isfinite(height) & (height > 0)
        location = location.masked_fill(~good[:, None], float("nan"))
        profiles["rgb_gradient_profile"] = gradient.masked_fill(~valid, float("nan"))
        row = torch.arange(len(index), device=rgb.device)
        shadow_profile = profiles.get("shadow_probability_profile")
        shadow_value = ((shadow_profile[row, index]+shadow_profile[row, index+1])*.5
                        if shadow_profile is not None else None)
        profiles["rgb_peak"].update(uv=location, offset_px=offset, gradient=height,
                                    contrast=contrast, fwhm_px=width,
                                    competing_peaks=competing, shadow_probability=shadow_value)
        profiles["flags"].update(no_rgb_contrast=~good | (contrast < minimum_rgb_contrast),
                                 blur_review=width > maximum_edge_width_px,
                                 multi_edge_ambiguity=competing > 0,
                                 peak_at_search_boundary=(index == 0) | (index == gradient.shape[1]-1))
        delta = values.new_tensor([[-1., -1.], [0., -1.], [1., -1.], [-1., 0.],
                                   [0., 0.], [1., 0.], [-1., 1.], [0., 1.], [1., 1.]])
        uv = location[:, None]+delta
        complete = (torch.isfinite(uv).all(-1) & (uv[..., 0] >= 0) & (uv[..., 0] <= w-1)
                    & (uv[..., 1] >= 0) & (uv[..., 1] <= h-1)).all(1)
        grid = torch.stack((2*(uv[..., 0]+.5)/w-1, 2*(uv[..., 1]+.5)/h-1), -1)[None]
        grid = torch.nan_to_num(grid, nan=2., posinf=2., neginf=-2.)
        gray = F.grid_sample(values, grid, align_corners=False)[0].mean(0)
        profiles["local_sharpness_at_rgb_peak"] = dict(
            patch_complete=complete,
            laplacian=(gray[:, 1]+gray[:, 3]+gray[:, 5]+gray[:, 7]-4*gray[:, 4]).abs(),
            gradient=((gray[:, 5]-gray[:, 3]).square()+(gray[:, 7]-gray[:, 1]).square()).sqrt()*.5,
            variance=gray.var(1, correction=0))
        profiles["metadata"]["source_transition_polarity"] = polarity

    for guide in expected_curves_2d:
        if not isinstance(guide, Mapping):
            raise ValueError("Every source guide must be an explicit mapping")
        if "source_binding" in guide:
            _same_frame(binding, guide["source_binding"])
        exclusion = global_exclusion
        if guide.get("source_exclusion_mask") is not None:
            if guide.get("source_exclusion_binding") is None:
                raise ValueError("A per-guide source exclusion needs its native frame binding")
            _same_frame(binding, guide["source_exclusion_binding"])
            own_exclusion = _cuda_tensor(guide["source_exclusion_mask"], dimensions=4)
            if (own_exclusion.shape != (1, 1, h, w) or own_exclusion.device != rgb.device
                    or own_exclusion.dtype != torch.bool):
                raise ValueError("Per-guide exclusions must be boolean native CUDA [1,1,H,W]")
            exclusion = own_exclusion if exclusion is None else exclusion | own_exclusion
            any_exclusion_used = True
        origin = guide.get("origin")
        if origin not in ("user_confirmed_visual_supervision", "model_projection_search_hint",
                          "source_annotation_search_hint", "temporal_source_search_hint"):
            raise ValueError("Declare direct human, projection, source-annotation or temporal search origin")
        expected_id, authority = guide.get("id"), guide.get("identity_authority")
        if expected_id is None or not str(expected_id):
            raise ValueError("A guide requires an explicit expected identity")
        human = origin == "user_confirmed_visual_supervision"
        if human and not authority:
            raise ValueError("Human-confirmed identity needs its source authority")
        identity_confirmed = human or (guide.get("identity_confirmed") is True and bool(authority))
        kind, primitive, role = guide.get("kind"), guide.get("primitive"), guide.get("physical_role")
        if kind not in names:
            raise ValueError(f"Guide kind must name an actual semantic class: {kind}")
        if primitive not in ("line", "circle", "conic", "free_curve") or not role:
            raise ValueError("Declare the expected source primitive and its physical boundary/axis role")
        points = cuda_array(guide.get("points_uv", guide.get("uv")), torch.float32, "points_uv", 2)
        if points.shape[1:] != (2,) or not len(points):
            raise ValueError("A source guide must contain nonempty native Nx2 points")
        n = len(points)
        if "segment_ids" in guide:
            raw_segments = guide["segment_ids"]
            if isinstance(raw_segments, torch.Tensor) and raw_segments.dtype not in (
                    torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError("Segment identities must be integers")
            segments = cuda_array(raw_segments, torch.int64, "segment_ids", 1)
            if segments.shape != (n,):
                raise ValueError("Supply one segment identity for each original guide point")
        else:
            segments = torch.zeros(n, device=rgb.device, dtype=torch.int64)
        if "normals" in guide:
            normals = cuda_array(guide["normals"], torch.float32, "normals", 2)
            if normals.shape != points.shape:
                raise ValueError("Normals must match every original source guide point")
        else:
            # Tangents use immediate neighbors in the same declared segment.
            previous, following = torch.roll(points, 1, 0), torch.roll(points, -1, 0)
            has_previous = segments == torch.roll(segments, 1, 0)
            has_following = segments == torch.roll(segments, -1, 0)
            if not guide.get("closed", False):
                has_previous[0], has_following[-1] = False, False
            tangent = torch.where(has_following[:, None], following-points, torch.zeros_like(points))
            tangent += torch.where(has_previous[:, None], points-previous, torch.zeros_like(points))
            normals = torch.stack((-tangent[:, 1], tangent[:, 0]), 1)
        radius = float(guide.get("search_radius_px", search_radius_px))
        profiles = measure_boundary_profiles(
            values, probabilities, points, normals, class_index=names.index(kind),
            search_radius=radius, sample_step=float(guide.get("sample_step_px", sample_step_px)),
            rgb_source_binding=binding, probability_source_binding=binding,
            projection_source_binding=binding, segment_ids=segments,
            shadow_probabilities=shadow, shadow_source_binding=binding if shadow is not None else None,
            minimum_contrast=minimum_rgb_contrast, maximum_edge_width=maximum_edge_width_px,
            minimum_typed_probability=minimum_type_probability, ambiguity_ratio=ambiguity_ratio)
        polarity = guide.get("rgb_transition_polarity")
        if polarity is not None:
            if (polarity not in ("darkening_along_normal", "lightening_along_normal")
                    or role != "filled_disc_boundary"
                    or (not human and not guide.get("transition_authority"))):
                raise ValueError("Transition polarity needs an explicit source-confirmed filled-disc material boundary")
            restrict_source_transition(profiles, polarity)
        if role == "painted_stroke_center_axis":
            observation = measure_paint_band(
                profiles, signal=guide.get("signal"), physical_role=role,
                source_binding=binding, range_source_binding=binding,
                probabilities_conditioned_on_geometry=False, minimum_signal_contrast=minimum_rgb_contrast,
                maximum_edge_width_px=maximum_edge_width_px,
                minimum_type_probability=minimum_type_probability,
                maximum_shadow_probability=maximum_shadow_probability, ambiguity_ratio=ambiguity_ratio)
            candidate_uv = observation["all_center_uv"]
            quality_mask = observation["source_colour_pair_quality_mask"].clone()
            quality_weights = observation["source_colour_pair_weights"]
            flags = {k: v for k, v in observation["rejection_flags"].items()
                     if k not in ("weak_type_at_optical_center", "shadow_on_band_or_border")}
            disagreement = dict(type_weak=observation["rejection_flags"]["weak_type_at_optical_center"],
                                shadow_high=observation["rejection_flags"].get("shadow_on_band_or_border"),
                                type_probability=observation["typed_at_optical_center"],
                                shadow_probability=observation["shadow_at_band"])
            support = dict(method="opposite_source_colour_transitions_midpoint",
                           left_border_uv=observation["left_border_uv"],
                           right_border_uv=observation["right_border_uv"],
                           band_width_px=observation["source_band_width_native_px"],
                           left_fwhm_px=observation["left_transition_fwhm_px"],
                           right_fwhm_px=observation["right_transition_fwhm_px"],
                           normal_interval_px=observation["optical_center_support_interval_px"],
                           interval_relative_to="guide_uv_not_measured_center",
                           is_calibrated_uncertainty=False)
        else:
            observation = select_profile_correspondences(
                profiles, source_binding=binding, geometry_source_binding=binding,
                point_source_binding=binding, probability_is_geometry_conditioned=False,
                minimum_typed_at_rgb_peak=minimum_type_probability,
                maximum_shadow_at_rgb_peak=maximum_shadow_probability,
                maximum_edge_width=maximum_edge_width_px)
            candidate_uv = observation["all_candidate_uv"]
            quality_mask = observation["source_rgb_quality_mask"].clone()
            quality_weights = observation["source_rgb_quality_weights"]
            flags = {k: v for k, v in observation["rejection_flags"].items()
                     if k not in ("typed_multimodal", "weak_type_at_rgb_peak", "shadow_at_rgb_peak")}
            disagreement = dict(type_weak=observation["rejection_flags"]["weak_type_at_rgb_peak"],
                                type_multimodal=observation["rejection_flags"]["typed_multimodal"],
                                shadow_high=observation["rejection_flags"].get("shadow_at_rgb_peak"),
                                type_probability=observation["typed_probability_at_rgb_peak"],
                                shadow_probability=observation["shadow_probability_at_rgb_peak"])
            width, offset = profiles["rgb_peak"]["fwhm_px"], observation["rgb_offset_native_px"]
            support = dict(method="single_source_RGB_material_transition",
                           fwhm_px=width, normal_interval_px=torch.stack((offset-width*.5, offset+width*.5), 1),
                           rgb_transition_polarity=polarity,
                           interval_relative_to="guide_uv_not_measured_center",
                           is_calibrated_uncertainty=False)
        if exclusion is not None:
            # A stroke needs both source transitions clear, as well as its midpoint.
            locations = ([candidate_uv, support["left_border_uv"], support["right_border_uv"]]
                         if role == "painted_stroke_center_axis" else [candidate_uv])
            excluded = torch.zeros(n, device=rgb.device, dtype=torch.bool)
            for uv in locations:
                grid = torch.stack((2*(uv[:, 0]+.5)/w-1, 2*(uv[:, 1]+.5)/h-1), -1)[None, :, None]
                grid = torch.nan_to_num(grid, nan=2., posinf=2., neginf=-2.)
                excluded |= F.grid_sample(exclusion.float(), grid, mode="nearest",
                                          padding_mode="zeros", align_corners=False)[0, 0, :, 0] > .5
            flags["known_source_occlusion"] = excluded
            quality_mask &= ~excluded
        temporal_valid = guide.get("temporal_flow_supported")
        if temporal_valid is not None:
            temporal_valid = cuda_array(temporal_valid, torch.bool, "temporal_flow_supported", 1)
            if temporal_valid.shape != (n,):
                raise ValueError("Temporal validity must bind every guide position")
            flags["temporal_identity_unresolved"] = ~temporal_valid
            quality_mask &= temporal_valid
        quality_mask &= torch.isfinite(candidate_uv).all(1) & (quality_weights > 0)
        indices = torch.nonzero(quality_mask, as_tuple=False).flatten()
        actual = candidate_uv[indices]
        # Only consecutive original indices from one declared segment are joined.
        runs = []
        index_list, segment_list = indices.detach().cpu().tolist(), segments[indices].detach().cpu().tolist()
        start = 0
        for end in range(1, len(index_list)+1):
            if end == len(index_list) or (index_list[end] != index_list[end-1]+1
                                         or segment_list[end] != segment_list[end-1]):
                runs.append(dict(actual_source_uv=actual[start:end], source_indices=indices[start:end],
                                 segment_id=segment_list[start]))
                start = end
        characterization = dict(line=None, image_conic=None, fitted_samples_are_source_pixels=False)
        for shape, minimum, key in (("line", 3, "line"), ("conic", 6, "image_conic")):
            if len(actual) >= minimum:
                try:
                    characterization[key] = primitive_diagnostic(actual, kind=shape)
                except ValueError as exc:
                    characterization[key] = dict(valid=False, reason=str(exc), diagnostic_only=True)
        peak_candidates = None
        if role == "painted_stroke_center_axis":
            peak_candidates = measure_all_paint_band_candidates(
                profiles, source_binding=binding, range_source_binding=binding,
                maximum_edge_width_px=maximum_edge_width_px)
            if exclusion is not None and len(peak_candidates["actual_peak_uv"]):
                excluded = torch.zeros(len(peak_candidates["actual_peak_uv"]), device=rgb.device, dtype=torch.bool)
                for coordinates in (peak_candidates["actual_peak_uv"], peak_candidates["left_border_uv"],
                                    peak_candidates["right_border_uv"]):
                    grid = torch.stack((2*(coordinates[:, 0]+.5)/w-1,
                                        2*(coordinates[:, 1]+.5)/h-1), -1)[None, :, None]
                    excluded |= F.grid_sample(exclusion.float(), grid, mode="nearest",
                                               align_corners=False)[0, 0, :, 0] > .5
                peak_candidates["source_valid"] &= ~excluded
                peak_candidates["known_source_occlusion"] = excluded
        else:
            # Keep every measurable source peak, including ambiguous/wide peaks.
            # A later named image-primitive reading can use all of this evidence.
            gradients = profiles["rgb_gradient_profile"]
            pair_valid = profiles["rgb_gradient_valid_mask"] & torch.isfinite(gradients)
            safe = torch.where(pair_valid, gradients, -torch.inf)
            previous = F.pad(safe[:, :-1], (1, 0), value=-float("inf"))
            following = F.pad(safe[:, 1:], (0, 1), value=-float("inf"))
            minimum_gradient = minimum_rgb_contrast/max(1., maximum_edge_width_px)
            local = pair_valid & (safe >= previous) & (safe > following) & (safe >= minimum_gradient)
            local &= ~profiles["flags"]["invalid_point_or_normal"][:, None]
            local &= (profiles["rgb_peak"]["contrast"] >= minimum_rgb_contrast)[:, None]
            row, column = torch.nonzero(local, as_tuple=True)
            g = gradients[row]
            height = gradients[row, column]
            p = torch.arange(gradients.shape[1], device=rgb.device)[None]
            half = pair_valid[row] & (g >= height[:, None]*.5) & (g > 0)
            lo = torch.where((p < column[:, None]) & ~half, p, -1).amax(1)
            hi = torch.where((p > column[:, None]) & ~half, p, gradients.shape[1]).amin(1)
            step = profiles["metadata"]["sample_step_native_px"]
            width = (hi-lo-1).float()*step
            offset = profiles["rgb_gradient_offsets_native_px"][column]
            locations = profiles["points_native_uv"][row]+profiles["normals_unit"][row]*offset[:, None]
            clipped = (lo < 0) | (hi >= gradients.shape[1])
            excluded = torch.zeros(len(row), device=rgb.device, dtype=torch.bool)
            if exclusion is not None and len(row):
                grid = torch.stack((2*(locations[:, 0]+.5)/w-1,
                                    2*(locations[:, 1]+.5)/h-1), -1)[None, :, None]
                excluded = F.grid_sample(exclusion.float(), grid, mode="nearest",
                                         align_corners=False)[0, 0, :, 0] > .5
            source_valid = ~clipped & ~excluded & torch.isfinite(locations).all(1)
            q = height/(height+minimum_gradient)/(1+width.clamp_min(1))
            peak_candidates = dict(
                actual_peak_uv=locations, source_indices=row, peak_sample_indices=column,
                normals=profiles["normals_unit"][row], segment_ids=segments[row],
                gradient=height, fwhm_px=width, offset_px=offset,
                normal_interval_px=torch.stack((
                    profiles["rgb_gradient_offsets_native_px"][(lo+1).clamp(0, gradients.shape[1]-1)]-step*.5,
                    profiles["rgb_gradient_offsets_native_px"][(hi-1).clamp(0, gradients.shape[1]-1)]+step*.5), 1),
                source_quality_weights=q, source_valid=source_valid,
                blur=width > maximum_edge_width_px, profile_boundary_clipped=clipped,
                known_source_occlusion=excluded,
                typed_probability=(profiles["typed_probability_profile"][row, column]
                                   + profiles["typed_probability_profile"][row, column+1])*.5,
                rgb_transition_polarity=polarity,
                material_outward_transition=guide.get("material_outward_transition", "dark_to_light"),
                metadata=dict(all_source_peaks_above_minimum_gradient_retained=True,
                              minimum_gradient=float(minimum_gradient),
                              source_measurement_is_exact=False, camera_or_world_used=False,
                              is_calibrated_uncertainty=False))
        if temporal_valid is not None and peak_candidates is not None:
            peak_candidates["source_valid"] &= temporal_valid[peak_candidates["source_indices"]]
            peak_candidates["temporal_identity_unresolved"] = ~temporal_valid[peak_candidates["source_indices"]]
        readings.append(dict(
            expected_id=expected_id, physical_id=expected_id if identity_confirmed else None,
            identity_authority=authority if identity_confirmed else None, origin=origin,
            kind=kind, primitive=primitive, physical_role=role,
            actual_source_uv=actual, source_indices=indices, segment_ids=segments[indices],
            measured_runs=runs, source_quality_mask=quality_mask,
            source_quality_weights=torch.where(quality_mask, quality_weights, 0.),
            flags=flags, model_disagreement=disagreement, support=support,
            shape_characterization=characterization,
            source_peak_candidates=peak_candidates,
            source_group_id=guide.get("source_group_id"),
            visible_arc_branch=guide.get("visible_arc_branch"),
            confirmed_source_uv=points.clone() if human else None,
            guide=dict(uv=points, normals=profiles["normals_unit"], segment_ids=segments,
                       closed=bool(guide.get("closed", False)),
                       source_binding=binding),
            metadata=dict(source_point_count=n, actual_measured_count=len(indices),
                          position_rule="strongest_actual_RGB_evidence_in_full_supplied_normal_range",
                          guide_position_is_measurement=False, projection_distance_used=False,
                          user_confirmed_supervision_preserved=human, model_disagreement_is_veto=False,
                          unmeasured_sections_preserved=True, occlusion_filter_available=exclusion is not None,
                          missing_sections_extrapolated=False, camera_or_world_fitted=False,
                          is_pixel_ground_truth=False, localization_accuracy_validated=False)))
    return dict(source_binding=binding, native_hw=[h, w], readings=readings,
                metadata=dict(probabilities_conditioned_on_geometry=False,
                              actual_source_uv_contains_only_measured_source_locations=True,
                              original_guide_indices_preserved=True, camera_or_world_fitted=False,
                              source_exclusion_filter_available=any_exclusion_used,
                              user_confirmed_visual_supervision_has_priority=True,
                              model_disagreement_is_veto=False, is_pixel_ground_truth=False,
                              localization_accuracy_validated=False))


def source_annotations_to_guides(
    annotations,
    *,
    source_binding,
    annotation_source_binding,
    role_definitions=None,
    verified_identity_group_ids=(),
    default_search_radius_px=18.,
    classes=("background", "ice_contact", "board_top", "ice_paint", "ad_boundary", "shadow_or_reflection"),
):
    """Frame-bound annotation adapter, independent of PTS, programme or camera.

    Human source drawings, including historical user_trace_weak records, retain
    the user's highest visual authority. This adapter never reclassifies them
    using network confidence. Old AI/source-search UV remain search hints and
    are never promoted to human supervision or training labels. Separately
    verified identity groups must be declared explicitly by the caller.

    role_definitions maps source feature/curve IDs to physical_role or to saved
    source-definition records. Schematic colour is never read. RGB transition
    polarity is copied only from an explicit source annotation/definition; a
    source-confirmed gray_disc_vs_white_ice definition can interpret a named
    upper/lower branch with source-native normal orientation. Unresolved paint
    roles are returned explicitly instead of treating every circle as a stroke.
    """
    binding = _same_frame(source_binding, annotation_source_binding)
    if not np.isfinite(default_search_radius_px) or default_search_radius_px <= 0:
        raise ValueError("Search radius must be finite, positive native pixels")
    definitions = dict(role_definitions or {})
    verified_groups = set(verified_identity_group_ids)
    guides, unresolved = [], []
    human_states = {"user_confirmed_visual_supervision", "user_trace_weak"}
    role_map = {
        "named_filled_disc_color_transition_boundary": "filled_disc_boundary",
        "source_confirmed_painted_band_center_axis": "painted_stroke_center_axis",
        "explicit_source_painted_band_optical_center_axis": "painted_stroke_center_axis",
        "nominal_painted_stroke_center_axis": "painted_stroke_center_axis",
        "painted_stroke_center_axis": "painted_stroke_center_axis",
    }
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            raise ValueError("Each source annotation must be a mapping")
        if "source_binding" in annotation:
            _same_frame(binding, annotation["source_binding"])
        for name, expected in (("source_pts", binding["source_pts"]), ("source_sha256", binding["source_sha256"])):
            if name in annotation and annotation[name] != expected:
                raise ValueError("An annotation belongs to another source frame")
        kind = annotation.get("kind")
        identity = annotation.get("feature_id", annotation.get("id", annotation.get("group_id")))
        group = annotation.get("group_id", identity)
        primitive = annotation.get("primitive", "free_curve")
        points = annotation.get("uv", annotation.get("points_uv"))
        if kind not in classes or identity is None or points is None:
            unresolved.append(dict(group_id=group, reason="missing_source_identity_kind_or_native_UV"))
            continue
        if primitive not in ("line", "circle", "conic", "free_curve"):
            unresolved.append(dict(group_id=group, reason="unsupported_explicit_source_primitive"))
            continue
        revision_indices = None
        original_points = points
        if "source_active_supervision_mask" in annotation or annotation.get("superseded_by_user_boundary_revision"):
            from .vision_supervision import active_source_supervision_mask
            active = active_source_supervision_mask({**annotation, "uv": points})
            revision_indices = np.flatnonzero(active)
            if not len(revision_indices):
                unresolved.append(dict(group_id=group, reason="superseded_by_later_direct_user_boundary_revision"))
                continue
            points = np.asarray(original_points)[revision_indices].tolist()
        authority = annotation.get("authority", annotation.get("identity_authority"))
        authority_map = authority if isinstance(authority, Mapping) else {}
        human = (annotation.get("state") in human_states
                 or annotation.get("manual_annotation_ground_truth") is True
                 or authority_map.get("human_authority") is True
                 or authority_map.get("origin") == "direct_user_provided_drawing")
        if human and not authority:
            unresolved.append(dict(group_id=group, reason="human_source_annotation_missing_recorded_authority"))
            continue
        branch = annotation.get("user_drawing_branch", annotation.get("visible_arc_branch"))
        # Group names are recorded source semantics, not projected locations.
        if branch not in ("upper", "lower"):
            branch = ("upper" if str(group).endswith("_upper") or "_outer_upper_" in str(group)
                      else "lower" if str(group).endswith("_lower") or "_outer_lower_" in str(group) else None)
        subrole = annotation.get("feature_subrole", annotation.get("ring_role"))
        if subrole is None:
            for suffix in ("outer", "middle", "inner", "center"):
                if f"_{suffix}_" in str(group) or str(group).endswith(f"_{suffix}"):
                    subrole = suffix
                    break
        curve_id = str(identity) if ":" in str(identity) or subrole is None else f"{identity}:{subrole}"
        definition = definitions.get(curve_id, definitions.get(str(identity), {}))
        if not definition:
            candidates = [item for item in definitions.values() if isinstance(item, Mapping)
                          and str(item.get("feature_id")) == str(identity)]
            roles = {item.get("physical_role", role_map.get(item.get("semantic_definition")))
                     for item in candidates}
            roles.discard(None)
            if len(roles) == 1:
                definition = dict(physical_role=roles.pop(),
                                  role_authority="consistent_saved_source_feature_definitions",
                                  source_definition_curve_ids=[item.get("curve_id") for item in candidates])
        if isinstance(definition, str):
            definition = dict(physical_role=definition)
        if not isinstance(definition, Mapping):
            raise ValueError("Source role definitions must be explicit mappings or physical role strings")
        role = annotation.get("physical_role", definition.get("physical_role"))
        if role is None:
            role = role_map.get(definition.get("semantic_definition"))
        if role is None and kind == "board_top":
            role = "board_top_edge"
        elif role is None and kind == "ice_contact":
            role = "ice_board_contact"
        elif role is None and kind == "ad_boundary":
            # Printed contours, holes and physical panel borders are different
            # source roles even if the semantic head calls them ad_boundary.
            boundary_role = annotation.get("boundary_role")
            printed_roles = {
                "external_printed_ink_component": "printed_pattern_external_contour",
                "internal_ink_hole": "printed_pattern_internal_hole_contour",
                "print_background_outer_boundary": "printed_background_external_contour",
            }
            role = printed_roles.get(boundary_role)
            if role is None:
                role = ("printed_stroke_outer_boundary" if "outermost_visible_circular_print_stroke" in
                        str(annotation.get("target_role", "")) else "advertisement_visible_boundary")
        elif role is None and kind == "ice_paint" and primitive == "line":
            role = "painted_stroke_center_axis"
        if not role:
            unresolved.append(dict(group_id=group, feature_id=identity, curve_id=curve_id,
                                   reason="source_paint_boundary_vs_stroke_axis_role_unresolved"))
            continue
        named_identity = curve_id if kind == "ice_paint" and primitive in ("circle", "conic") and subrole in (
            "outer", "middle", "inner", "center") else identity
        guide = dict(
            id=named_identity, annotation_feature_id=identity, source_group_id=group, kind=kind, primitive=primitive,
            points_uv=points, physical_role=role,
            origin="user_confirmed_visual_supervision" if human else "source_annotation_search_hint",
            identity_authority=authority,
            identity_confirmed=human or group in verified_groups or annotation.get("source_identity_verified") is True,
            source_binding=binding, closed=bool(annotation.get("annotation_closed", annotation.get("closed", False))),
            visible_arc_branch=branch,
            search_radius_px=float(annotation.get("search_radius_px", default_search_radius_px)),
            source_annotation_state=annotation.get("state"),
            source_role_definition=definition,
            metadata=dict(annotation_adapter_created_training_target=False,
                          original_manual_visual_supervision_preserved=human,
                          old_AI_UV_is_search_hint_not_label=not human,
                          source_role_is_expected_observation_definition_not_pixel_truth=True))
        if "normal" in annotation or "normals" in annotation:
            guide["normals"] = annotation.get("normal", annotation.get("normals"))
        if "segment_ids" in annotation:
            guide["segment_ids"] = annotation["segment_ids"]
        if "source_visibility_evidence" in annotation:
            visibility_evidence = annotation["source_visibility_evidence"]
            if not source_visibility_matches_frame(binding, visibility_evidence):
                raise ValueError("Source visibility evidence belongs to another native RGB frame")
            guide["source_visibility_evidence"] = dict(visibility_evidence)
        if "source_visible_mask" in annotation:
            visibility = np.asarray(annotation["source_visible_mask"])
            if visibility.dtype != np.bool_ or visibility.shape != (len(original_points),):
                raise ValueError("Source guide visibility needs one explicit boolean per original position")
            guide["source_visible_mask"] = (visibility if revision_indices is None
                                             else visibility[revision_indices]).tolist()
        if revision_indices is not None:
            # User authority retires only the explicitly corrected positions.
            # Compute directions on the original curve, never across its gap.
            native_points = np.asarray(original_points, float)
            normals = annotation.get("normal", annotation.get("normals"))
            if normals is None:
                tangent = np.gradient(native_points, axis=0) if len(native_points) > 1 else np.zeros_like(native_points)
                normals = np.column_stack((-tangent[:, 1], tangent[:, 0]))
            normals = np.asarray(normals)
            if normals.shape != native_points.shape:
                raise ValueError("Revised source normals must match the complete original curve")
            guide["normals"] = normals[revision_indices].tolist()
            original_segments = np.asarray(annotation.get("segment_ids", np.zeros(len(native_points), int)))
            if original_segments.shape != (len(native_points),):
                raise ValueError("Revised source segments must match the complete original curve")
            breaks = ((np.diff(revision_indices) != 1) |
                      (np.diff(original_segments[revision_indices]) != 0))
            guide["segment_ids"] = np.r_[0, np.cumsum(breaks)].tolist()
            guide["original_source_point_indices"] = revision_indices.tolist()
            guide["closed"] = guide["closed"] and len(revision_indices) == len(native_points)
            guide["metadata"].update(original_source_position_count=len(native_points),
                superseded_source_position_count=len(native_points)-len(revision_indices),
                retired_gap_connected_as_source_curve=False)
        if role == "painted_stroke_center_axis":
            guide["signal"] = annotation.get("signal", definition.get("source_signal", "dark_band"))
        polarity = annotation.get("rgb_transition_polarity", definition.get("rgb_transition_polarity"))
        if (polarity is None and role == "filled_disc_boundary" and branch in ("upper", "lower")
                and definition.get("source_material_transition") == "gray_disc_vs_white_ice"):
            normals = guide.get("normals")
            y_sign = 1.
            if normals is not None:
                values = normals.detach().cpu().numpy() if hasattr(normals, "detach") else np.asarray(normals)
                y_sign = float(np.sign(np.median(values[:, 1])))
            darkening = (branch == "upper") == (y_sign >= 0)
            polarity = "darkening_along_normal" if darkening else "lightening_along_normal"
        if polarity is not None:
            guide["rgb_transition_polarity"] = polarity
            guide["transition_authority"] = annotation.get("transition_authority", definition.get("source_material_authority"))
            if not human and not guide["transition_authority"]:
                unresolved.append(dict(group_id=group, reason="nonhuman_transition_polarity_missing_source_authority"))
                guide.pop("rgb_transition_polarity")
        guides.append(guide)
    return dict(source_binding=binding, guides=guides, unresolved=unresolved,
                metadata=dict(source_annotation_count=len(annotations), guide_count=len(guides),
                              source_roles_unresolved=len(unresolved), camera_or_world_used=False,
                              pixel_training_labels_generated=False,
                              human_source_visual_supervision_has_priority=True,
                              localization_accuracy_validated=False))


def measure_all_paint_band_candidates(profiles, *, source_binding, range_source_binding,
                                      maximum_edge_width_px=4.):
    """All complete native red/blue/dark source-band midpoints and noise data.

    Two actual opposite signed gradient peaks define every candidate midpoint.
    Noise and native colour quantization supply engineering signal scales;
    faint bands are retained, not denied by a universal contrast threshold.
    Multiple bands, low signal/noise and broad transitions remain explicit.
    These measurements do not determine physical identity or exact paint pixels.
    """
    torch = _torch()
    import torch.nn.functional as F
    binding = _same_frame(source_binding, range_source_binding)
    _same_frame(binding, profiles["source_binding"])
    if not profiles["metadata"].get("source_evidence_only"):
        raise ValueError("Band candidates require actual native source RGB profiles")
    rgb = _cuda_tensor(profiles["rgb_profile"], dimensions=3)
    valid = _cuda_tensor(profiles["valid_mask"], dimensions=2).bool()
    n, k, _ = rgb.shape
    if k < 7 or not np.isfinite(maximum_edge_width_px) or maximum_edge_width_px <= 0:
        raise ValueError("Complete source bands require at least seven samples and a positive edge-width flag scale")
    step = profiles["metadata"]["sample_step_native_px"]
    offsets = profiles["offsets_native_px"]
    grad_offsets = profiles["rgb_gradient_offsets_native_px"]
    samples = (rgb[..., 0]-.5*(rgb[..., 1]+rgb[..., 2]),
               rgb[..., 2]-.5*(rgb[..., 0]+rgb[..., 1]), 1-rgb.mean(-1))
    pieces = []
    signal_names = ("red_excess", "blue_excess", "dark_band")
    row_valid = torch.isfinite(profiles["points_native_uv"]).all(1) & torch.isfinite(profiles["normals_unit"]).all(1)
    position = torch.arange(k-1, device=rgb.device)

    def widths_at_peaks(gradient, peak_mask):
        scores = gradient[:, None, :]
        height = gradient[:, :, None]
        half = (scores >= height*.5) & (scores > 0)
        p = position[None, None]
        index = position[None, :, None]
        left = torch.where((p < index) & ~half, p, -1).amax(2)
        right = torch.where((p > index) & ~half, p, k-1).amin(2)
        width = (right-left-1).float()*step
        return width, (left < 0) | (right >= k-1)

    for signal_index, value in enumerate(samples):
        value = value.masked_fill(~valid, float("nan"))
        highpass = (value[:, 2:]-2*value[:, 1:-1]+value[:, :-2])
        hp_median = torch.nanmedian(highpass, 1).values
        sigma = (1.4826*torch.nanmedian((highpass-hp_median[:, None]).abs(), 1).values/(6.**.5)).clamp_min(.5/255.)
        gradient = (value[:, 1:]-value[:, :-1])/step
        pair_valid = valid[:, 1:] & valid[:, :-1] & row_valid[:, None]
        median_gradient = torch.nanmedian(gradient, 1).values
        noise_gradient = (1.4826*torch.nanmedian((gradient-median_gradient[:, None]).abs(), 1).values).clamp_min(.5/255.)
        # Keep low-SNR real peaks for full-segment evidence aggregation.
        source_gradient_floor = torch.minimum(noise_gradient*.75, noise_gradient.new_full(noise_gradient.shape, 1/255.)).clamp_min(.5/255.)
        positive = gradient.clamp_min(0)
        negative = (-gradient).clamp_min(0)
        positive_safe = torch.where(pair_valid, positive, -torch.inf)
        negative_safe = torch.where(pair_valid, negative, -torch.inf)
        positive_mask = (pair_valid & (positive_safe >= F.pad(positive_safe[:, :-1], (1, 0), value=-torch.inf))
                         & (positive_safe > F.pad(positive_safe[:, 1:], (0, 1), value=-torch.inf))
                         & (positive > source_gradient_floor[:, None]))
        negative_mask = (pair_valid & (negative_safe >= F.pad(negative_safe[:, :-1], (1, 0), value=-torch.inf))
                         & (negative_safe > F.pad(negative_safe[:, 1:], (0, 1), value=-torch.inf))
                         & (negative > source_gradient_floor[:, None]))
        pair = positive_mask[:, :, None] & negative_mask[:, None, :]
        left, right = position[None, :, None], position[None, None, :]
        pair &= (right > left) & (left >= 2) & (right < k-4)
        # Local source background on both sides, not a global frame/colour prior.
        outside_left = (value[:, (position-1).clamp_min(0)]+value[:, (position-2).clamp_min(0)])*.5
        outside_right = (value[:, (position+2).clamp_max(k-1)]+value[:, (position+3).clamp_max(k-1)])*.5
        background = (outside_left[:, :, None]+outside_right[:, None, :])*.5
        cumulative = F.pad(torch.nan_to_num(value).cumsum(1), (1, 0))
        # Interior includes the material side of both measured transitions.
        interior = (cumulative[:, (position+1)][..., None, :]-cumulative[:, position+1][:, :, None])
        interior /= (right-left).clamp_min(1)
        depth = interior-background
        # Reject pairs spanning several separate colour components: every
        # interior sample must support this source material elevation.
        source_minimum = torch.where(torch.arange(k, device=rgb.device)[None, None] > position[None, :, None],
                                     value[:, None, :], torch.inf).cummin(2).values[:, :, :k-1]
        pair &= torch.isfinite(depth) & (depth > .5/255.) & (source_minimum >= background+depth*.1)
        pair &= torch.isfinite(outside_left)[:, :, None] & torch.isfinite(outside_right)[:, None, :]
        row, li, ri = torch.nonzero(pair, as_tuple=True)
        if not len(row):
            continue
        left_width, left_clipped = widths_at_peaks(positive, positive_mask)
        right_width, right_clipped = widths_at_peaks(negative, negative_mask)
        lw, rw = left_width[row, li], right_width[row, ri]
        offset = (grad_offsets[li]+grad_offsets[ri])*.5
        center_uv = profiles["points_native_uv"][row]+profiles["normals_unit"][row]*offset[:, None]
        left_uv = profiles["points_native_uv"][row]+profiles["normals_unit"][row]*grad_offsets[li, None]
        right_uv = profiles["points_native_uv"][row]+profiles["normals_unit"][row]*grad_offsets[ri, None]
        amplitude = depth[row, li, ri]
        snr = amplitude/sigma[row]
        edge_snr = torch.minimum(positive[row, li], negative[row, ri])/noise_gradient[row]
        weights = snr/(snr+2)*edge_snr/(edge_snr+1)/(1+(lw+rw)*.25)
        pieces.append(dict(
            actual_peak_uv=center_uv, left_border_uv=left_uv, right_border_uv=right_uv,
            source_indices=row, peak_sample_indices=torch.stack((li, ri), 1),
            normals=profiles["normals_unit"][row], segment_ids=profiles["segment_ids"][row],
            gradient=torch.minimum(positive[row, li], negative[row, ri]),
            fwhm_px=(lw+rw)*.5, left_fwhm_px=lw, right_fwhm_px=rw,
            band_width_px=grad_offsets[ri]-grad_offsets[li], offset_px=offset,
            normal_interval_px=torch.stack((offset-(lw+rw)*.25, offset+(lw+rw)*.25), 1),
            source_quality_weights=weights,
            source_valid=~left_clipped[row, li] & ~right_clipped[row, ri] & torch.isfinite(center_uv).all(1),
            blur=(lw > maximum_edge_width_px) | (rw > maximum_edge_width_px),
            signal_index=torch.full_like(row, signal_index), source_colour_amplitude=amplitude,
            source_signal_noise=sigma[row], source_gradient_noise=noise_gradient[row],
            band_snr=snr, edge_snr=edge_snr, low_single_point_snr=(snr < 2) | (edge_snr < 1.5)))
    if pieces:
        result = {key: torch.cat([piece[key] for piece in pieces]) for key in pieces[0]}
    else:
        empty = rgb.new_empty(0)
        result = dict(actual_peak_uv=rgb.new_empty((0, 2)), left_border_uv=rgb.new_empty((0, 2)),
                      right_border_uv=rgb.new_empty((0, 2)), source_indices=torch.empty(0,device=rgb.device,dtype=torch.long),
                      peak_sample_indices=torch.empty((0,2),device=rgb.device,dtype=torch.long),
                      normals=rgb.new_empty((0,2)),segment_ids=torch.empty(0,device=rgb.device,dtype=torch.long),
                      gradient=empty,fwhm_px=empty,left_fwhm_px=empty,right_fwhm_px=empty,
                      band_width_px=empty,offset_px=empty,normal_interval_px=rgb.new_empty((0,2)),
                      source_quality_weights=empty,source_valid=torch.empty(0,device=rgb.device,dtype=torch.bool),
                      blur=torch.empty(0,device=rgb.device,dtype=torch.bool),
                      signal_index=torch.empty(0,device=rgb.device,dtype=torch.long),source_colour_amplitude=empty,
                      source_signal_noise=empty,source_gradient_noise=empty,band_snr=empty,edge_snr=empty,
                      low_single_point_snr=torch.empty(0,device=rgb.device,dtype=torch.bool))
    result.update(source_binding=binding, signal_names=signal_names,
                  metadata=dict(position_rule="midpoint_of_two_actual_opposite_source_colour_gradient_peaks",
                                local_noise_rule="source_profile_robust_highpass_and_gradient_MAD_native_quantization_floor",
                                noise_estimate_is_calibrated=False, faint_source_bands_retained=True,
                                neural_probability_used=False, guide_distance_used=False,
                                source_measurement_is_exact=False, camera_or_world_used=False))
    return result


def _select_source_graph_shape(uv, profile, gradient, width, quality, primary, *,
                               profile_count, degree, convexity, maximum_residual_px,
                               hypothesis_count, random_seed):
    """GPU graph-shape consensus selecting existing RGB peaks, not new UV."""
    torch = _torch()
    device = uv.device
    m = len(uv)
    counts = torch.bincount(profile, minlength=profile_count)
    active = counts > 0
    fallback = primary[active]
    if int(active.sum()) < max(degree+1, 6):
        return fallback, None, False
    center = uv.double().mean(0)
    rotation = torch.eye(2, device=device, dtype=torch.float64)
    if degree <= 2 and not convexity:
        centered = uv.double()-center
        _, vectors = torch.linalg.eigh(centered.T@centered/max(1, len(centered)))
        axis = vectors[:, -1]
        major_native_dimension = int((uv.amax(0)-uv.amin(0)).argmax())
        axis *= torch.where(axis[major_native_dimension] >= 0, 1., -1.)
        rotation = torch.stack((axis, torch.stack((-axis[1], axis[0]))), 1)
    local_uv = (uv.double()-center)@rotation
    scale = (local_uv[:, 0].amax()-local_uv[:, 0].amin())*.5
    if float(scale) <= 1e-6:
        return fallback, None, False
    xy = local_uv/scale
    x, y = xy.unbind(1)
    design = torch.stack([x**j for j in range(degree+1)], 1)
    derivatives = torch.stack([torch.zeros_like(x) if j == 0 else j*x**(j-1)
                               for j in range(degree+1)], 1)
    order = torch.argsort(profile, stable=True)
    starts = counts.cumsum(0)-counts
    profiles = torch.nonzero(active, as_tuple=False).flatten()
    source_mean_x = torch.zeros(profile_count, device=device, dtype=xy.dtype)
    source_mean_x.scatter_add_(0, profile, xy[:, 0])
    source_mean_x /= counts.clamp_min(1)
    profiles = profiles[torch.argsort(source_mean_x[profiles])]
    strata = torch.tensor_split(profiles, degree+1)
    if any(not len(part) for part in strata):
        return fallback, None, False
    generator = torch.Generator(device=device).manual_seed(int(random_seed))
    chosen_profiles = torch.stack([part[torch.randint(len(part), (hypothesis_count,),
                                        device=device, generator=generator)] for part in strata], 1)
    pick = torch.floor(torch.rand(chosen_profiles.shape, device=device, generator=generator)
                       *counts[chosen_profiles]).long()
    sample = order[starts[chosen_profiles]+pick]
    use_primary = torch.rand(sample.shape, device=device, generator=generator) < .65
    sample = torch.where(use_primary, primary[chosen_profiles], sample)
    solution, _, rank, _ = torch.linalg.lstsq(design[sample], y[sample, None], driver="gels")
    coefficients = solution[..., 0]
    sample_rank_valid = torch.linalg.svdvals(design[sample])[:, -1] > 1e-8
    tolerance = torch.maximum(width.double()*.5, width.new_full(width.shape, maximum_residual_px).double())
    quality_max = torch.zeros(profile_count, device=device, dtype=quality.dtype)
    quality_max.scatter_reduce_(0, profile, quality, reduce="amax", include_self=True)
    relative_quality = quality/quality_max[profile].clamp_min(1e-12)
    grid_x = torch.linspace(float(x.min()), float(x.max()), 128, device=device, dtype=torch.float64)
    second_derivatives = torch.stack([torch.zeros_like(grid_x) if j < 2 else
                                     j*(j-1)*grid_x**(j-2) for j in range(degree+1)], 1)

    def evaluate(coefficients):
        predicted = coefficients@design.T
        slope = coefficients@derivatives.T
        residual = (predicted-y[None]).abs()/torch.sqrt(1+slope.square())*scale
        legal = torch.isfinite(coefficients).all(1)
        if convexity:
            curvature = (coefficients@second_derivatives.T)*convexity
            legal &= curvature.amin(1) >= -1e-6
        matching = legal[:, None] & (residual <= tolerance[None])
        score = (.5+.5*relative_quality[None])*torch.exp(-.5*(residual/tolerance[None]).square())
        score = torch.where(matching, score, 0.)
        profile_score = torch.zeros((len(coefficients), profile_count), device=device, dtype=score.dtype)
        profile_score.scatter_reduce_(1, profile[None].expand(len(coefficients), -1), score,
                                      reduce="amax", include_self=True)
        coverage = (profile_score[:, active] > 0).double().mean(1)
        total = .75*coverage+.25*profile_score[:, active].mean(1)
        total.masked_fill_(~legal | (coverage < .5), -torch.inf)
        return total, score, residual, legal, coverage

    candidate_ids = torch.arange(m, device=device)

    def choose(scores):
        best = torch.zeros(profile_count, device=device, dtype=scores.dtype)
        best.scatter_reduce_(0, profile, scores, reduce="amax", include_self=True)
        selected = torch.full((profile_count,), m, device=device, dtype=torch.long)
        selected.scatter_reduce_(0, profile, torch.where((scores == best[profile]) & (scores > 0),
                                                       candidate_ids, m), reduce="amin", include_self=True)
        return selected[selected < m]

    scores, candidate_scores, _, _, _ = evaluate(coefficients)
    scores.masked_fill_(~sample_rank_valid, -torch.inf)
    winner = int(scores.argmax())
    if not bool(torch.isfinite(scores[winner])):
        return fallback, None, False
    best_score, model = float(scores[winner]), coefficients[winner]
    selected = choose(candidate_scores[winner])
    for _ in range(3):
        if len(selected) < degree+1:
            break
        weight = quality[selected].double().clamp_min(1e-8).sqrt()
        refined = torch.linalg.lstsq(design[selected]*weight[:, None],
                                    y[selected, None]*weight[:, None], driver="gels").solution[:, 0]
        value, candidates, _, legal, _ = evaluate(refined[None])
        if bool(legal[0]) and float(value[0]) > best_score:
            model, best_score = refined, float(value[0])
            selected = choose(candidates[0])
        else:
            break
    value, candidates, residual, _, coverage = evaluate(model[None])
    selected = choose(candidates[0])
    competing = False
    for alternative in scores.topk(min(8, len(scores))).indices:
        if not bool(torch.isfinite(scores[alternative])) or float(scores[alternative]) < best_score-.03:
            continue
        other = choose(candidate_scores[alternative])
        mine = torch.full((profile_count,), m, device=device, dtype=torch.long)
        theirs = mine.clone()
        mine[profile[selected]], theirs[profile[other]] = selected, other
        shared = (mine < m) & (theirs < m)
        if int(shared.sum()) >= 6 and float(torch.linalg.vector_norm(
                uv[mine[shared]]-uv[theirs[shared]], dim=1).median()) > maximum_residual_px:
            competing = True
            break
    return selected, dict(
        kind="visible_image_polynomial_arc" if degree > 1 else "source_image_straight_line",
        polynomial_degree=degree, coefficients_normalized=model,
        normalization_center=center, normalization_scale=scale,
        normalization_rotation=rotation,
        visible_source_x_extent=torch.stack((uv[:, 0].amin(), uv[:, 0].amax())),
        source_score=best_score, source_profile_coverage=float(coverage[0]),
        source_residual_px=residual[0, selected], measured_candidate_indices=selected,
        diagnostic_only=True, invisible_circle_geometry_estimated=False,
        fitted_samples_are_source_pixels=False, identity_verified_by_fit=False,
        localization_accuracy_validated=False), competing


def coherent_named_source_reading(
    guided_result,
    *,
    maximum_residual_px=2.,
    hypothesis_count=512,
    random_seed=1729,
):
    """Select source material transitions and coherent named image primitives.

    All coordinates are existing measured RGB peaks. Only declared source
    identities join readings; there is no camera/world input or guide-distance
    score. A CUDA conic hypothesis organizes the actual peak evidence without
    creating any new sampled curve pixels. Signed source transitions must agree
    with entering/leaving the same image ellipse. A low residual remains a shape
    hypothesis, never proof of physical identity or pixel-level correctness.

    A lone branch can supply a visible convex/concave image-arc hypothesis. It
    cannot establish the invisible portion of a disk. Broad peaks stay measured
    with their support and blur flags. All confirmed human positions survive.
    Contact with ice additionally uses discriminative, no-prior neural contact
    evidence to rank measured RGB transitions, retaining RGB-only fallback when
    the semantic field cannot distinguish the candidates. This rule does not
    infer disk instance identities or apply contact ranking to other roles.
    """
    torch = _torch()
    binding = _source_binding(guided_result["source_binding"])
    if not np.isfinite(maximum_residual_px) or maximum_residual_px <= 0:
        raise ValueError("An image-source residual scale must be finite and positive")
    if isinstance(hypothesis_count, bool) or not isinstance(hypothesis_count, int) or hypothesis_count < 1:
        raise ValueError("Use a positive integer source-image hypothesis count")
    result = dict(guided_result)
    readings = [dict(reading) for reading in guided_result["readings"]]
    result["readings"] = readings
    groups = {}

    def source_runs(uv, indices, segments):
        runs = []
        ids = indices.detach().cpu().tolist()
        seg = segments.detach().cpu().tolist()
        start = 0
        for end in range(1, len(ids)+1):
            if end == len(ids) or ids[end] != ids[end-1]+1 or seg[end] != seg[end-1]:
                runs.append(dict(actual_source_uv=uv[start:end], source_indices=indices[start:end],
                                 segment_id=seg[start]))
                start = end
        return runs

    for index, reading in enumerate(readings):
        _same_frame(binding, reading["guide"]["source_binding"])
        pool = reading.get("source_peak_candidates")
        if (reading.get("kind") == "ice_contact"
                and reading.get("physical_role") == "ice_board_contact"
                and pool is not None):
            # Several board-material transitions can share a normal scan. The
            # strongest RGB edge may be the foot-strip upper edge, rather than
            # contact with ice. Rank only measured, sharp RGB candidates using
            # source-bound contact probabilities. Flat or weak semantic fields
            # retain the original RGB reading; guide distance is never a score.
            valid = pool["source_valid"] & ~pool["blur"]
            probability = pool["typed_probability"]
            profile = pool["source_indices"]
            n = len(reading["guide"]["uv"])
            m = len(profile)
            best = probability.new_full((n,), -torch.inf)
            least = probability.new_full((n,), torch.inf)
            best.scatter_reduce_(0, profile, torch.where(valid, probability, -torch.inf),
                                 reduce="amax", include_self=True)
            least.scatter_reduce_(0, profile, torch.where(valid, probability, torch.inf),
                                  reduce="amin", include_self=True)
            # A class-wide plateau cannot distinguish the physical transition.
            discriminative = torch.isfinite(best) & (best >= .2) & (best - least >= .05)
            winner = torch.full((n,), m, dtype=torch.long, device=profile.device)
            winner.scatter_reduce_(0, profile, torch.where(
                valid & (probability == best[profile]),
                torch.arange(m, device=profile.device), m), reduce="amin", include_self=True)
            chosen_profiles = torch.nonzero(discriminative, as_tuple=False).flatten()
            chosen = winner[chosen_profiles]
            old_indices = reading["source_indices"]
            unchanged = ~discriminative[old_indices]
            source_indices = torch.cat((old_indices[unchanged], chosen_profiles))
            source_uv = torch.cat((reading["actual_source_uv"][unchanged],
                                   pool["actual_peak_uv"][chosen]))
            support_width = torch.cat((reading["support"]["fwhm_px"][old_indices[unchanged]],
                                       pool["fwhm_px"][chosen]))
            support_interval = torch.cat((reading["support"]["normal_interval_px"][old_indices[unchanged]],
                                          pool["normal_interval_px"][chosen]))
            source_quality = torch.cat((reading["source_quality_weights"][old_indices[unchanged]],
                                        pool["source_quality_weights"][chosen]))
            order = torch.argsort(source_indices, stable=True)
            source_indices, source_uv = source_indices[order], source_uv[order]
            segment_ids = reading["guide"]["segment_ids"][source_indices]
            reading.update(
                coherent_source_uv=source_uv, coherent_source_indices=source_indices,
                coherent_measured_runs=source_runs(source_uv, source_indices, segment_ids),
                coherent_semantic_candidate_indices=chosen,
                coherent_support=dict(fwhm_px=support_width[order],
                    normal_interval_px=support_interval[order],
                    source_quality_weights=source_quality[order],
                    interval_relative_to="guide_uv_not_measured_center",
                    is_calibrated_uncertainty=False),
                coherent_flags=dict(status="semantic_ranked_measured_RGB_contact_transitions",
                    semantic_ranked_profile_count=len(chosen_profiles),
                    original_RGB_fallback_profile_count=int(unchanged.sum()),
                    semantic_minimum_probability=.2, semantic_minimum_candidate_spread=.05,
                    source_locating_exact=False, camera_or_world_fitted=False,
                    guide_distance_used=False, neural_field_can_change_final_source_UV=True,
                    human_confirmed_positions_preserved=reading["origin"] == "user_confirmed_visual_supervision"))
            continue
        if (reading.get("primitive") == "line" and reading.get("physical_role") == "painted_stroke_center_axis"
                and pool is not None):
            valid = _cuda_tensor(pool["source_valid"], dimensions=1).bool()
            keep = torch.nonzero(valid, as_tuple=False).flatten()
            uv = pool["actual_peak_uv"][keep]
            profile = pool["source_indices"][keep]
            n = len(reading["guide"]["uv"])
            m = len(uv)
            active = torch.bincount(profile, minlength=n) > 0
            relative_quality = pool["source_quality_weights"][keep]
            strongest = uv.new_full((n,), -torch.inf)
            strongest.scatter_reduce_(0, profile, relative_quality, reduce="amax", include_self=True)
            primary = torch.full((n,), m, device=uv.device, dtype=torch.long)
            primary.scatter_reduce_(0, profile, torch.where(relative_quality == strongest[profile],
                                   torch.arange(m, device=uv.device), m), reduce="amin", include_self=True)
            selected, model, competing = _select_source_graph_shape(
                uv, profile, pool["gradient"][keep], pool["fwhm_px"][keep], relative_quality, primary,
                profile_count=n, degree=1, convexity=0., maximum_residual_px=maximum_residual_px,
                hypothesis_count=hypothesis_count, random_seed=random_seed)
            # Brown-native paint axes can show mild visible image curvature.
            # A quadratic is accepted only if actual source evidence improves.
            curved, curved_model, curved_competing = _select_source_graph_shape(
                uv, profile, pool["gradient"][keep], pool["fwhm_px"][keep], relative_quality, primary,
                profile_count=n, degree=2, convexity=0., maximum_residual_px=maximum_residual_px,
                hypothesis_count=hypothesis_count, random_seed=random_seed)
            if curved_model is not None and (model is None or curved_model["source_score"] > model["source_score"]+.03):
                selected, model, competing = curved, curved_model, curved_competing
            indices = keep[selected]
            src = pool["source_indices"][indices]
            order = torch.argsort(src, stable=True)
            indices, src = indices[order], src[order]
            actual = pool["actual_peak_uv"][indices]
            reading.update(
                coherent_source_uv=actual, coherent_source_indices=src,
                coherent_candidate_indices=indices,
                coherent_measured_runs=source_runs(actual, src, pool["segment_ids"][indices]),
                coherent_support=dict(left_border_uv=pool["left_border_uv"][indices],
                                      right_border_uv=pool["right_border_uv"][indices],
                                      fwhm_px=pool["fwhm_px"][indices],
                                      band_width_px=pool["band_width_px"][indices],
                                      normal_interval_px=pool["normal_interval_px"][indices],
                                      interval_relative_to="guide_uv_not_measured_center",
                                      source_signal_noise=pool["source_signal_noise"][indices],
                                      source_gradient_noise=pool["source_gradient_noise"][indices],
                                      band_snr=pool["band_snr"][indices], edge_snr=pool["edge_snr"][indices],
                                      signal_index=pool["signal_index"][indices], signal_names=pool["signal_names"],
                                      is_calibrated_uncertainty=False),
                coherent_flags=dict(status="actual_source_colour_pair_image_axis_hypothesis" if model is not None else
                                           "image_axis_unobservable_source_pairs_only",
                                    blur=pool["blur"][indices],
                                    low_single_point_snr=pool["low_single_point_snr"][indices],
                                    competing_image_shape_hypothesis=competing,
                                    source_locating_exact=False, camera_or_world_fitted=False,
                                    guide_distance_used=False,
                                    human_confirmed_positions_preserved=reading["origin"] == "user_confirmed_visual_supervision"),
                coherent_shape_characterization=model)
            continue
        if (reading.get("primitive") in ("circle", "conic")
                and reading.get("physical_role") in ("filled_disc_boundary", "printed_stroke_outer_boundary",
                                                     "painted_stroke_center_axis")
                and pool is not None):
            key = (str(reading.get("physical_id") or reading["expected_id"]), reading.get("physical_id") is not None)
            groups.setdefault(key, []).append(index)
        else:
            reading.update(coherent_source_uv=reading["actual_source_uv"],
                           coherent_source_indices=reading["source_indices"],
                           coherent_measured_runs=reading["measured_runs"],
                           coherent_flags=dict(status="original_source_reading_no_joint_disk_scope",
                                               source_locating_exact=False))
    group_reports = []

    for (identity, identity_verified), member_indices in groups.items():
        members = [readings[index] for index in member_indices]
        first_pool = members[0]["source_peak_candidates"]
        device = _cuda_tensor(first_pool["actual_peak_uv"], dimensions=2).device
        points, normal, gradients, widths, quality, profile_ids, member_ids, pool_indices, signs = [], [], [], [], [], [], [], [], []
        profile_base = 0
        active_profiles_by_member = []
        for member_index, reading in enumerate(members):
            pool = reading["source_peak_candidates"]
            valid = _cuda_tensor(pool["source_valid"], dimensions=1).bool()
            pos = _cuda_tensor(pool["actual_peak_uv"], dimensions=2)
            if pos.device != device or valid.shape != (len(pos),):
                raise ValueError("Named source peak groups must share one CUDA device")
            indices = torch.nonzero(valid, as_tuple=False).flatten()
            original_profile = pool["source_indices"][indices]
            points.append(pos[indices]); normal.append(pool["normals"][indices])
            gradients.append(pool["gradient"][indices]); widths.append(pool["fwhm_px"][indices])
            quality.append(pool["source_quality_weights"][indices])
            profile_ids.append(original_profile+profile_base)
            member_ids.append(torch.full_like(indices, member_index))
            pool_indices.append(indices)
            polarity = pool.get("rgb_transition_polarity")
            sign = (-1. if polarity == "darkening_along_normal" else
                    1. if polarity == "lightening_along_normal" else 0.)
            # Nested light discs have the inverse material order. Conic normals
            # point outward regardless of which side is darker; keep source polarity.
            if pool.get("material_outward_transition") == "light_to_dark":
                sign *= -1
            signs.append(torch.full((len(indices),), sign, device=device))
            active_profiles_by_member.append(torch.unique(original_profile)+profile_base)
            profile_base += len(reading["guide"]["uv"])
        uv = torch.cat(points); normals = torch.cat(normal)
        g, width, q = torch.cat(gradients), torch.cat(widths), torch.cat(quality)
        profile, member, local_pool = torch.cat(profile_ids), torch.cat(member_ids), torch.cat(pool_indices)
        polarity_sign = torch.cat(signs)
        m, p_count = len(uv), profile_base
        counts = torch.bincount(profile, minlength=p_count)
        active = counts > 0
        primary_height = torch.full((p_count,), -torch.inf, device=device)
        primary_height.scatter_reduce_(0, profile, g, reduce="amax", include_self=True)
        candidate_id = torch.arange(m, device=device)
        primary = torch.full((p_count,), m, device=device, dtype=torch.long)
        primary.scatter_reduce_(0, profile, torch.where(g == primary_height[profile], candidate_id, m),
                                reduce="amin", include_self=True)
        selected = primary[active]
        status = "single_branch_shape_unobservable_source_peaks_only"
        model = None
        best_score, competing_shape = None, False
        best_residual = None
        has_both_polarities = bool((polarity_sign < 0).any() and (polarity_sign > 0).any())
        explicitly_closed_single = len(members) == 1 and members[0]["guide"].get("closed") is True
        multi_sided_single_scope = (len(members) == 1 and m >= 6
                                   and bool((normals[:, 0] < -.25).any()) and bool((normals[:, 0] > .25).any())
                                   and bool((normals[:, 1] < -.25).any()) and bool((normals[:, 1] > .25).any()))
        can_fit = ((len(members) >= 2 or explicitly_closed_single or multi_sided_single_scope)
                   and (has_both_polarities or bool((polarity_sign == 0).all())) and m >= 6
                   and all(len(ids) >= (6 if explicitly_closed_single else 3) for ids in active_profiles_by_member))
        if can_fit:
            xy_center = uv.double().mean(0)
            xy_scale = torch.linalg.vector_norm(uv.double()-xy_center, dim=1).square().mean().sqrt()
            xy = (uv.double()-xy_center)/xy_scale.clamp_min(1e-8)
            x, y = xy.unbind(1)
            design = torch.stack((x*x, x*y, y*y, x, y, torch.ones_like(x)), 1)
            normal64 = normals.double()
            # Profile membership is used only to draw actual source candidates.
            order = torch.argsort(profile, stable=True)
            starts = counts.cumsum(0)-counts
            generator = torch.Generator(device=device).manual_seed(int(random_seed))
            assigned = []
            for member_index, profiles in enumerate(active_profiles_by_member):
                take = 3 if len(members) == 2 else max(1, 6//len(members))
                choice = torch.rand((hypothesis_count, len(profiles)), device=device,
                                    generator=generator).argsort(1)[:, :take]
                assigned.append(profiles[choice])
            chosen_profiles = torch.cat(assigned, 1)[:, :6]
            if chosen_profiles.shape[1] < 6:
                available = torch.nonzero(active, as_tuple=False).flatten()
                extra = torch.randint(len(available), (hypothesis_count, 6-chosen_profiles.shape[1]),
                                      device=device, generator=generator)
                chosen_profiles = torch.cat((chosen_profiles, available[extra]), 1)
            uniform_pick = torch.floor(torch.rand(chosen_profiles.shape, device=device, generator=generator)
                                       * counts[chosen_profiles]).long()
            sample = order[starts[chosen_profiles]+uniform_pick]
            # Primary source-gradient hypotheses are complemented by hypotheses
            # using all other actual peaks; this is not a nearest-hint choice.
            use_primary = torch.rand(sample.shape, device=device, generator=generator) < .65
            sample = torch.where(use_primary, primary[chosen_profiles], sample)
            _, singular, vh = torch.linalg.svd(design[sample], full_matrices=False)
            coefficients = vh[:, -1]
            coefficients *= torch.where((coefficients[:, 0]+coefficients[:, 2]) >= 0, 1., -1.)[:, None]
            sample_rank_valid = singular[:, -2] > singular[:, 0]*1e-8
            tolerance = torch.maximum(width.double()*.5, width.new_full(width.shape, maximum_residual_px).double())

            def evaluate(coefficients):
                a, b, c, d, e, f = coefficients.unbind(1)
                quadratics = torch.stack((a, b*.5, b*.5, c), 1).reshape(-1, 2, 2)
                eigen = torch.linalg.eigvalsh(quadratics)
                determinant = a*c-b*b*.25
                safe_det = torch.where(determinant.abs() > 1e-14, determinant, torch.ones_like(determinant))
                center = torch.stack(((-.5*c*d+.25*b*e)/safe_det,
                                      (.25*b*d-.5*a*e)/safe_det), 1)
                translated = f+.5*(d*center[:, 0]+e*center[:, 1])
                radii_sq = -translated[:, None]/eigen.clamp_min(1e-14)*xy_scale.square()
                source_center = center*xy_scale+xy_center
                bbox_lo, bbox_hi = uv.double().amin(0), uv.double().amax(0)
                span = torch.linalg.vector_norm(bbox_hi-bbox_lo)
                legal = ((eigen[:, 0] > eigen[:, 1]*1e-4) & (translated < 0)
                         & torch.isfinite(radii_sq).all(1) & (radii_sq.amin(1) > .25)
                         & (radii_sq.amax(1) < (span*3).square())
                         & ((source_center >= bbox_lo) & (source_center <= bbox_hi)).all(1))
                gx = 2*a[:, None]*x+b[:, None]*y+d[:, None]
                gy = b[:, None]*x+2*c[:, None]*y+e[:, None]
                residual = (coefficients@design.T).abs()/(gx.square()+gy.square()).sqrt().clamp_min(1e-12)*xy_scale
                orientation = (gx*normal64[:, 0]+gy*normal64[:, 1])*polarity_sign.double()
                matching = legal[:, None] & (residual <= tolerance[None]) & ((orientation > 0) | (polarity_sign == 0)[None])
                q_max = torch.zeros(p_count, device=device, dtype=q.dtype)
                q_max.scatter_reduce_(0, profile, q, reduce="amax", include_self=True)
                source_relative_quality = q/q_max[profile].clamp_min(1e-12)
                fit_score = (.5+.5*source_relative_quality[None])*torch.exp(-.5*(residual/tolerance[None]).square())
                fit_score = torch.where(matching, fit_score, 0.)
                profile_score = torch.zeros((len(coefficients), p_count), device=device, dtype=fit_score.dtype)
                profile_score.scatter_reduce_(1, profile[None].expand(len(coefficients), -1), fit_score,
                                              reduce="amax", include_self=True)
                coverage = []
                branch_score = []
                for profiles in active_profiles_by_member:
                    values = profile_score[:, profiles]
                    covered = (values > 0).double().mean(1)
                    coverage.append(covered)
                    branch_score.append(.65*covered+.35*values.mean(1))
                cover = torch.stack(coverage, 1)
                score = torch.stack(branch_score, 1).mean(1)+cover.amin(1)*.1
                score = score.masked_fill(~legal | (cover.amin(1) < .35), -torch.inf)
                return score, fit_score, residual, legal, cover

            scores, candidate_scores, residuals, legal, cover = evaluate(coefficients)
            scores.masked_fill_(~sample_rank_valid, -torch.inf)

            def choose_peak(scores):
                best = torch.zeros(p_count, device=device, dtype=scores.dtype)
                best.scatter_reduce_(0, profile, scores, reduce="amax", include_self=True)
                chosen = torch.full((p_count,), m, device=device, dtype=torch.long)
                chosen.scatter_reduce_(0, profile,
                                       torch.where((scores == best[profile]) & (scores > 0), candidate_id, m),
                                       reduce="amin", include_self=True)
                return chosen[chosen < m]

            winner = int(scores.argmax())
            if bool(torch.isfinite(scores[winner])):
                model = coefficients[winner]
                best_score = float(scores[winner])
                selected = choose_peak(candidate_scores[winner])
                # Refit only selected, existing source peaks. Accept improved
                # source evidence coverage/quality, never new fitted pixels.
                for _ in range(3):
                    if len(selected) < 6:
                        break
                    weights = q[selected].double().clamp_min(1e-8)
                    _, _, refined_vh = torch.linalg.svd(design[selected]*weights.sqrt()[:, None], full_matrices=False)
                    refined = refined_vh[-1]
                    refined *= torch.where(refined[0]+refined[2] >= 0, 1., -1.)
                    new_score, new_candidates, new_residual, new_legal, _ = evaluate(refined[None])
                    if bool(new_legal[0]) and float(new_score[0]) > best_score:
                        model, best_score = refined, float(new_score[0])
                        selected = choose_peak(new_candidates[0])
                        best_residual = new_residual[0, selected]
                    else:
                        break
                final_score, final_candidates, final_residual, _, final_cover = evaluate(model[None])
                selected = choose_peak(final_candidates[0])
                best_residual = final_residual[0, selected]
                status = "joint_actual_source_peak_image_conic_hypothesis"
                for alternative in scores.topk(min(8, len(scores))).indices:
                    if not bool(torch.isfinite(scores[alternative])) or float(scores[alternative]) < best_score-.03:
                        continue
                    other = choose_peak(candidate_scores[alternative])
                    mine = torch.full((p_count,), m, device=device, dtype=torch.long)
                    theirs = mine.clone()
                    mine[profile[selected]], theirs[profile[other]] = selected, other
                    shared = (mine < m) & (theirs < m)
                    if int(shared.sum()) >= 6 and float(torch.linalg.vector_norm(
                            uv[mine[shared]]-uv[theirs[shared]], dim=1).median()) > maximum_residual_px:
                        competing_shape = True
                        break
                model = dict(coefficients_normalized=model, normalization_center=xy_center,
                             normalization_scale=xy_scale,
                             source_score=best_score, branch_source_coverage=final_cover[0],
                             source_residual_px=best_residual,
                             measured_candidate_indices=selected,
                             diagnostic_only=True, fitted_samples_are_source_pixels=False,
                             identity_verified_by_fit=False, localization_accuracy_validated=False)
            else:
                selected = primary[active]
                status = "no_supported_joint_image_conic_source_peaks_only"
        elif len(members) == 1 and m and (bool((polarity_sign != 0).all())
                                         or members[0].get("visible_arc_branch") in ("upper", "lower")):
            # A visible upper/lower arc is identifiable from source evidence
            # without estimating or drawing its unseen circle completion.
            sign = (-float(polarity_sign[0])*float(torch.sign(normals[:, 1].median()))
                    if bool((polarity_sign != 0).all()) else
                    1. if members[0]["visible_arc_branch"] == "upper" else -1.)
            selected, model, competing_shape = _select_source_graph_shape(
                uv, profile, g, width, q, primary, profile_count=p_count, degree=4,
                convexity=sign, maximum_residual_px=maximum_residual_px,
                hypothesis_count=hypothesis_count, random_seed=random_seed)
            if model is not None:
                status = "visible_actual_source_peak_image_arc_hypothesis"
                best_score = model["source_score"]
        for member_index, reading in enumerate(members):
            picked = selected[member[selected] == member_index]
            pool = reading["source_peak_candidates"]
            pool_index = local_pool[picked]
            src = pool["source_indices"][pool_index]
            order = torch.argsort(src, stable=True)
            pool_index, src = pool_index[order], src[order]
            observed = pool["actual_peak_uv"][pool_index]
            number_per_profile = torch.bincount(pool["source_indices"][pool["source_valid"]],
                                                minlength=len(reading["guide"]["uv"]))
            reading.update(
                coherent_source_uv=observed, coherent_source_indices=src,
                coherent_candidate_indices=pool_index,
                coherent_measured_runs=source_runs(observed, src, pool["segment_ids"][pool_index]),
                coherent_support=dict(fwhm_px=pool["fwhm_px"][pool_index],
                                      normal_interval_px=pool["normal_interval_px"][pool_index],
                                      interval_relative_to="guide_uv_not_measured_peak",
                                      is_calibrated_uncertainty=False),
                coherent_flags=dict(status=status, blur=pool["blur"][pool_index],
                                    multiple_source_peaks=number_per_profile[src] > 1,
                                    competing_image_shape_hypothesis=competing_shape,
                                    source_locating_exact=False, camera_or_world_fitted=False,
                                    guide_distance_used=False, human_confirmed_positions_preserved=True),
                coherent_shape_characterization=model)
        group_reports.append(dict(physical_id=identity if identity_verified else None, expected_id=identity, status=status,
                                  source_peak_count=m, source_profile_count=int(active.sum()),
                                  actual_read_count=len(selected), hypothesis_count=hypothesis_count if can_fit or model is not None else 0,
                                  source_score=best_score, competing_shape_hypothesis=competing_shape,
                                  search_scope_has_multiple_arc_sides=multi_sided_single_scope,
                                  closed_visible_boundary_proven_by_shape_fit=False,
                                  fitted_curve_pixels_in_output=False, source_locating_exact=False))
    result["coherent_groups"] = group_reports
    result["metadata"] = dict(guided_result["metadata"],
                              joint_image_primitive_reading=True,
                              guide_distance_used_for_peak_selection=False,
                              output_pixels_all_existing_RGB_peaks=True,
                              camera_or_world_fitted=False, localization_accuracy_validated=False)
    return result


if __name__ == "__main__":
    import json
    print(json.dumps(gpu_sanity_checks(), ensure_ascii=False))
