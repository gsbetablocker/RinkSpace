"""Source-image support anchors for the existing RinkSpace body state.

Reference implementation, not a fitted result or a second body/skate state.
NumPy only. Geometry is in ONE caller-declared metric world. Image coordinates
are original native Brown pixels, after the project's rolling-shutter projection.
There are no default precision gates, speed limits, or fixed spin centres.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math
import numpy as np


def _array(value, shape, name):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite, shape {shape}")
    return a


def _positive(value, name, *, zero=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite scalar")
    v = float(value)
    if not np.isfinite(v) or (v < 0 if zero else v <= 0):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")
    return v


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, not a rounded timestamp")
    return int(value)


def _skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


@dataclass(frozen=True)
class NativeRegionObservation:
    source_id: str
    shot_id: str
    pts: int
    time_base: tuple[int, int]
    foot: int | None
    kind: str
    centre_uv: tuple[float, float]
    region_shape_px2: tuple[tuple[float, float], tuple[float, float]]
    visibility: str
    contact_state: str
    role: str
    reliability: float
    objective_weight: float
    provenance: str

    def __post_init__(self):
        if not self.source_id or not self.shot_id or not self.provenance:
            raise ValueError("Explicit source, shot, and evidence provenance are required")
        _integer(self.pts, "pts")
        if len(self.time_base) != 2 or any(_integer(v, "time_base") <= 0 for v in self.time_base):
            raise ValueError("time_base must be a positive integer rational")
        if self.foot is not None and (_integer(self.foot, "foot") not in (0, 1)):
            raise ValueError("foot must be anatomical track 0/1 or unknown (None)")
        if self.kind not in ("support_contact_region", "boot_marker_region"):
            raise ValueError("An ankle or boot centroid is not a contact-region observation")
        if self.visibility not in ("visible", "occluded", "unknown"):
            raise ValueError("Unknown visibility category")
        if self.contact_state not in ("supported", "airborne", "unknown"):
            raise ValueError("Unknown contact category")
        if self.role not in ("fit", "review"):
            raise ValueError("role must distinguish fitted and held-out evidence")
        _array(self.centre_uv, (2,), "centre_uv")
        C = _array(self.region_shape_px2, (2, 2), "region_shape_px2")
        if not np.allclose(C, C.T, rtol=0, atol=1e-12):
            raise ValueError("region_shape_px2 must be symmetric positive definite")
        try:
            np.linalg.cholesky(C)
        except np.linalg.LinAlgError as e:
            raise ValueError("region_shape_px2 must be positive definite") from e
        q = _positive(self.reliability, "reliability", zero=True)
        if q > 1:
            raise ValueError("reliability must be in [0, 1]; it is not a calibrated probability")
        _positive(self.objective_weight, "objective_weight")


def _disabled(reason):
    return dict(residual=np.zeros(2), jacobian_native_uv=np.zeros((2, 2)),
                enabled=False, reason=reason, cost=0.,
                measured_error_available=False)


def pixel_region_factor(predicted_native_uv, observation: NativeRegionObservation, *,
                        source_id, shot_id, pts, time_base, predicted_foot,
                        predicted_kind, robust_delta):
    """Soft distance OUTSIDE a source uncertainty ellipse, plus exact d r / d uv.

    region_shape_px2 defines the allowed source-image ellipse; it is NOT a
    calibrated Gaussian covariance. Outside it, a radial pseudo-Huber cost
    limits outlier influence. Reliability and objective weight are fixed
    evidence/configuration inputs, never optimization variables.

    This function accepts an already projected prediction. Use the existing
    moving_projection and its pixel_jac_nodes for the production chain. Do
    not replace that chain with a midpoint pinhole projection.
    """
    o = observation
    if (source_id, shot_id, _integer(pts, "pts"), tuple(time_base)) != (
            o.source_id, o.shot_id, o.pts, tuple(o.time_base)):
        raise ValueError("Source/shot/exact PTS/time base mismatch")
    if predicted_kind != o.kind:
        raise ValueError("Predicted feature and observed feature have different meanings")
    if o.role != "fit":
        return _disabled("review_only_observation")
    if o.foot is None:
        return _disabled("anatomical_foot_unassociated")
    if _integer(predicted_foot, "predicted_foot") != o.foot:
        raise ValueError("Left/right track identity mismatch")
    if o.visibility != "visible":
        return _disabled("visibility:" + o.visibility)
    if o.reliability == 0:
        return _disabled("zero_fixed_reliability")
    if o.kind == "support_contact_region" and o.contact_state != "supported":
        return _disabled("contact:" + o.contact_state)
    uv = _array(predicted_native_uv, (2,), "predicted_native_uv")
    c = _positive(robust_delta, "robust_delta")
    W = np.linalg.solve(np.linalg.cholesky(np.asarray(o.region_shape_px2)), np.eye(2))
    z = W @ (uv - np.asarray(o.centre_uv))
    radius = float(np.linalg.norm(z))
    if radius <= 1.:
        e, De = np.zeros(2), np.zeros((2, 2))
    else:
        e = (1. - 1. / radius) * z
        De = ((1. - 1. / radius) * np.eye(2)
              + np.outer(z, z) / radius**3) @ W
    s = float(e @ e)
    u = math.sqrt(1. + s / c**2)
    k = math.sqrt(2. / (u + 1.))
    dk = -k / (4. * c**2 * u * (u + 1.))
    D = k * np.eye(2) + 2. * dk * np.outer(e, e)
    weight = math.sqrt(o.reliability * o.objective_weight)
    r = weight * k * e
    return dict(residual=r, jacobian_native_uv=weight * D @ De,
                enabled=True, reason="inside_source_region" if radius <= 1. else "outside_source_region",
                cost=float(r @ r / 2.), measured_error_available=True,
                native_pixel_offset=uv - np.asarray(o.centre_uv),
                source_ellipse_radius=radius,
                calibrated_probability=False)


def lowest_runner_witness(arcs, rotation_world, origin_world, *, plane_normal_world):
    """Exact minimum of supplied runner arcs and its local rotation derivative.

    arcs use RinkSpace contact_arcs fields. Only semantic='runner' is accepted;
    a boot shell/sole minimum is not silently relabelled as blade contact.
    Toe-pick support needs its separate semantic geometry, not this adapter.

    The derivative is d X / d eta for R_new = R @ Exp(eta), O fixed. It INCLUDES
    the motion of the geometric argmin along the rocker. For image XY residuals,
    the height-only envelope-theorem derivative is insufficient. All arc
    parameters are fixed for this function; shared geometry updates must
    re-evaluate the function with the trial template.

    Ties, endpoint transitions, and a level/degenerate arc are flagged. Tied
    branches can have the same point but different derivatives. A returned
    Jacobian is then only the selected branch, not a unique derivative of the
    minimum. Do not claim a differentiable measured contact point in those
    situations.
    """
    R = _array(rotation_world, (3, 3), "rotation_world")
    O = _array(origin_world, (3,), "origin_world")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-7, rtol=0) or np.linalg.det(R) <= 0:
        raise ValueError("rotation_world must belong to SO(3)")
    n = _array(plane_normal_world, (3,), "plane_normal_world")
    norm = np.linalg.norm(n)
    if norm == 0:
        raise ValueError("Plane normal must be nonzero")
    n = n / norm
    local_n = R.T @ n
    candidates = []
    for arc in arcs:
        if arc.get("semantic") != "runner":
            raise ValueError("Pass only explicitly selected runner arcs")
        a = _array(arc["axis_cos"], (3,), "axis_cos")
        b = _array(arc["axis_sin"], (3,), "axis_sin")
        centre = _array(arc["center"], (3,), "center")
        radius = _positive(arc["radius_m"], "radius_m")
        lo, hi = float(arc["theta_start_rad"]), float(arc["theta_end_rad"])
        if not np.isfinite([lo, hi]).all() or not 0 < hi - lo < 2 * np.pi:
            raise ValueError("Arc interval must have width in (0, 2 pi)")
        A, B = float(local_n @ a), float(local_n @ b)
        denominator = A*A + B*B
        flat = denominator <= 1e-24 * max(1., float(a@a + b@b))
        choices = [(lo, False), (hi, False)]
        if not flat:
            raw = math.atan2(-B, -A)
            for turn in range(math.floor((lo - raw) / (2*np.pi)) - 1,
                              math.ceil((hi - raw) / (2*np.pi)) + 2):
                theta = raw + turn*2*np.pi
                if lo < theta < hi:
                    choices.append((theta, True))
        for theta, interior in choices:
            q = centre + radius*(a*np.cos(theta) + b*np.sin(theta))
            X = O + R @ q
            dq = np.zeros((3, 3))
            if interior:
                dq_dn = np.outer(radius*(-a*np.sin(theta) + b*np.cos(theta)),
                                 (-B*a + A*b) / denominator)
                dq = dq_dn @ _skew(local_n)
            # At a stationary endpoint the argmin is about to switch between
            # an interior point and a clamped point. The point is continuous,
            # but its tangential derivative need not be. This angular epsilon
            # only detects a floating-point branch tie, not a physical margin.
            angular_eps = 64*np.finfo(float).eps*max(1., abs(theta))
            endpoint_transition = (not interior and not flat and
                abs(math.remainder(theta - raw, 2*np.pi)) <= angular_eps)
            candidates.append(dict(height=float(n @ X), world_point=X, local_point=q,
                arc_id=str(arc["id"]), theta_rad=theta, interior=interior,
                flat_arc=flat, endpoint_transition=bool(endpoint_transition),
                jacobian_origin=np.eye(3),
                jacobian_right_rotation=R @ (-_skew(q) + dq),
                frozen_material_point_jacobian=-R @ _skew(q)))
    if not candidates:
        raise ValueError("At least one runner arc is required")
    candidates.sort(key=lambda row: row["height"])
    result = dict(candidates[0])
    # Numerical tie detection only: this is not an ice-height tolerance.
    tie_eps = 64*np.finfo(float).eps*max(1., abs(result["height"]))
    tied = [r for r in candidates if abs(r["height"]-result["height"]) <= tie_eps]
    distinct = any(np.linalg.norm(r["world_point"]-result["world_point"]) > tie_eps for r in tied[1:])
    jacobian_eps = 64*np.finfo(float).eps*max(
        1., *(float(np.linalg.norm(r["jacobian_right_rotation"])) for r in tied))
    derivatives_differ = any(np.linalg.norm(
        r["jacobian_right_rotation"]-result["jacobian_right_rotation"]) > jacobian_eps
        for r in tied[1:])
    result["distinct_minima_tied"] = bool(distinct)
    result["endpoint_transition"] = any(r["endpoint_transition"] for r in tied)
    result["tied_derivatives_differ"] = bool(derivatives_differ)
    result["differentiability_warning"] = bool(distinct or derivatives_differ or
        result["endpoint_transition"] or any(r["flat_arc"] for r in tied))
    result["interpretation"] = "model geometric runner minimum, not an observed material contact particle"
    return result


def contact_region_plane_lift(observation: NativeRegionObservation, *, camera_origin_world,
                              ray_at_native_uv: Callable, plane_normal_world,
                              plane_offset, plane_height_sigma, extra_world_covariance,
                              pixel_difference_step, minimum_incidence_cosine):
    """Conditional inverse-camera diagnostic/initializer, not a second factor.

    X satisfies plane_normal_world dot X = plane_offset. Both coefficients
    are normalized together, so an arbitrary nonzero normal length does not
    change the plane. plane_height_sigma is a metric distance along the unit
    normal, independent of the scale of the plane equation.

    This is geometry conditioned on the camera at the observation's source
    exposure and observed-row time. camera_origin_world stays fixed during
    the pixel finite differences. The ray callback must use the same fixed
    observed-row camera and Brown model for every perturbed pixel; it must
    not advance camera time with the perturbed row while leaving the origin
    fixed. The resulting Jacobian does not automatically propagate camera,
    row-time, or implicit rolling-shutter uncertainty. Any such uncertainty
    not supplied by extra_world_covariance remains unmodeled, not proved zero.
    Production pixel optimization must still use moving_projection and its
    full derivative chain, not this inverse-camera diagnostic.

    Never add this lifted anchor AND the original pixel/height observation as
    independent evidence. They are derived from the same source information.
    """
    o = observation
    if o.kind != "support_contact_region" or o.contact_state != "supported":
        raise ValueError("Only observed contact regions may be lifted to the ice plane")
    if o.visibility != "visible" or o.foot is None:
        raise ValueError("A visible, associated source contact region is required")
    C = _array(camera_origin_world, (3,), "camera_origin_world")
    n = _array(plane_normal_world, (3,), "plane_normal_world")
    normal_length = np.linalg.norm(n)
    if normal_length == 0:
        raise ValueError("Plane normal must be nonzero")
    n = n / normal_length
    offset = float(plane_offset)
    if not np.isfinite(offset):
        raise ValueError("plane_offset must be finite")
    offset = offset / normal_length
    step = _positive(pixel_difference_step, "pixel_difference_step")
    incidence = _positive(minimum_incidence_cosine, "minimum_incidence_cosine")
    if incidence >= 1:
        raise ValueError("minimum_incidence_cosine must be less than one")
    sigma = _positive(plane_height_sigma, "plane_height_sigma", zero=True)
    extra = _array(extra_world_covariance, (3, 3), "extra_world_covariance")
    if not np.allclose(extra, extra.T, atol=1e-12, rtol=0) or np.linalg.eigvalsh(extra).min() < -1e-12:
        raise ValueError("extra_world_covariance must be positive semidefinite")

    def lift(uv):
        ray = _array(ray_at_native_uv(np.asarray(uv)), (3,), "world_ray")
        norm = np.linalg.norm(ray)
        if norm == 0:
            raise ValueError("Zero ray")
        ray = ray / norm
        denom = float(n @ ray)
        if abs(denom) < incidence:
            raise ValueError("Grazing ray: no reliable plane intersection")
        distance = (offset - n @ C) / denom
        if distance <= 0:
            raise ValueError("Plane intersection is behind the camera")
        return C + distance*ray, ray / denom

    uv = np.asarray(o.centre_uv)
    X, dh = lift(uv)
    J = np.column_stack([(lift(uv + np.eye(2)[axis]*step)[0]
                          - lift(uv - np.eye(2)[axis]*step)[0]) / (2*step)
                         for axis in range(2)])
    covariance = J @ np.asarray(o.region_shape_px2) @ J.T + sigma*sigma*np.outer(dh, dh) + extra
    return dict(position_world=X, jacobian_native_uv=J, world_uncertainty_shape=covariance,
                source_id=o.source_id, shot_id=o.shot_id, pts=o.pts,
                role="conditional_inverse_geometry_only", independent_evidence=False,
                pixel_jacobian_conditioning="fixed_observed_row_camera",
                calibrated_covariance=False)
