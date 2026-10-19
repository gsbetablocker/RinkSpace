"""Auditable, mask-isolated photometric *candidate* homography refinement.

This is a small-motion local optimiser, not a relocaliser or independent check.
The caller supplies ice-only masks in each image's own pixel coordinates and an
initial current-to-reference map. Neither landmarks nor previous error reports
enter this function. All returned geometry uses the supplied image resolution.

Outside-mask values are replaced *before* conversion/filtering. A 5x5 Gaussian
has radius two; every integer sample is inside a mask eroded by three pixels.
Reference bilinear values and their exact derivatives use four such samples.
There are no pyramids, unmasked gradients, ECC calls, or hidden interpolation.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


_SAFE_RADIUS = 3
_MAX_POINTS = 7000
_MIN_POINTS = 160
_MIN_STD = 0.002
_MIN_CORRELATION = 0.85
_HUBER = 0.015


def _isolated(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(image)
    mask = np.asarray(mask)
    if raw.ndim not in (2, 3) or (raw.ndim == 3 and raw.shape[2] != 3):
        raise ValueError("images must be grayscale or three-channel BGR arrays")
    if mask.shape != raw.shape[:2]:
        raise ValueError("mask must match its image's height and width")
    if mask.dtype.kind not in "buif" or not np.all(np.isfinite(mask)):
        raise ValueError("masks must be finite numeric arrays")
    valid = mask > 0
    # Do not let protected values influence conversion, scaling or validation.
    cleaned = np.zeros(raw.shape, dtype=np.float64)
    cleaned[valid] = raw[valid]
    if not np.all(np.isfinite(cleaned)):
        raise ValueError("usable image values must be finite")
    if raw.dtype.kind == "u":
        cleaned /= np.iinfo(raw.dtype).max
    elif raw.dtype.kind == "f":
        if np.any(cleaned < 0) or np.any(cleaned > 1):
            raise ValueError("floating point images must be in [0, 1]")
    else:
        raise ValueError("images must be unsigned integer or [0, 1] float arrays")
    if cleaned.ndim == 3:
        cleaned = cleaned @ np.array([0.114, 0.587, 0.299])
    safe = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    filtered = cv2.GaussianBlur(cleaned, (5, 5), 0.8, borderType=cv2.BORDER_CONSTANT)
    return filtered, safe


def _ranges(indices: np.ndarray) -> list[list[int]]:
    """Lossless, inclusive-exclusive ranges into the saved sampling pool."""
    if len(indices) == 0:
        return []
    boundaries = np.flatnonzero(np.diff(indices) != 1) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(indices)] - 1
    return [[int(indices[a]), int(indices[b]) + 1] for a, b in zip(starts, ends)]


def _project(h: np.ndarray, xy: np.ndarray) -> np.ndarray:
    q = np.c_[xy, np.ones(len(xy))] @ h.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return q[:, :2] / q[:, 2:3]


def _unpack(p: np.ndarray) -> np.ndarray:
    return np.r_[p[:8], 1.0].reshape(3, 3)


def _bilinear(image: np.ndarray, safe: np.ndarray, xy: np.ndarray):
    """Only fetch image values after all four interpolation corners are safe."""
    height, width = image.shape
    finite = np.all(np.isfinite(xy), axis=1)
    in_bounds = finite & (xy[:, 0] >= 0) & (xy[:, 1] >= 0)
    in_bounds &= (xy[:, 0] < width - 1) & (xy[:, 1] < height - 1)
    ix = np.zeros(len(xy), dtype=np.int64)
    iy = np.zeros(len(xy), dtype=np.int64)
    ix[in_bounds] = np.floor(xy[in_bounds, 0]).astype(np.int64)
    iy[in_bounds] = np.floor(xy[in_bounds, 1]).astype(np.int64)
    valid = in_bounds & safe[iy, ix] & safe[iy, ix + 1]
    valid &= safe[iy + 1, ix] & safe[iy + 1, ix + 1]
    ids = np.flatnonzero(valid)
    x, y = ix[ids], iy[ids]
    dx, dy = xy[ids, 0] - x, xy[ids, 1] - y
    a, b = image[y, x], image[y, x + 1]
    c, d = image[y + 1, x], image[y + 1, x + 1]
    values = a*(1-dx)*(1-dy) + b*dx*(1-dy) + c*(1-dx)*dy + d*dx*dy
    gx = (b-a)*(1-dy) + (d-c)*dy
    gy = (c-a)*(1-dx) + (d-b)*dx
    return ids, values, gx, gy


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - np.mean(a), b - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 1e-15 else 0.0


def _loss(residual: np.ndarray) -> float:
    a = np.abs(residual)
    return float(np.mean(np.where(a <= _HUBER, a*a/2, _HUBER*(a-_HUBER/2))))


def _support(points: np.ndarray, shape: tuple[int, int]) -> dict:
    height, width = shape
    if len(points) < 3:
        return {"hull_fraction": 0., "span_x_fraction": 0., "span_y_fraction": 0.}
    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)
    span = np.ptp(points, axis=0)
    return dict(hull_fraction=float(cv2.contourArea(hull)/(width*height)),
                span_x_fraction=float(span[0]/width), span_y_fraction=float(span[1]/height),
                hull_xy=hull.tolist())


def estimate_photometric(reference, current, ref_mask, cur_mask,
                         initial_reference_from_current, *, motion_model="homography",
                         perspective_prior_weight=0.001) -> dict:
    """Return JSON-safe candidate/refusal plus complete fitting-domain evidence.

    H maps current pixels to reference pixels. Gain/bias fit
    ``current_intensity = gain * reference_intensity(H(current_xy)) + bias``.
    ``fit_reference_xy`` are mapped sampling coordinates, NOT independently
    observed correspondences. ``independent_validation`` is always false.
    A refusal has H=None; the diagnostic initial/final model remains in the
    iteration trace and cannot be substituted for the refused map.
    """
    result = dict(schema="rinkspace.masked-photometric.v1", status="unavailable",
                  reason=None, H_reference_from_current=None,
                  matrix_direction="reference_from_current",
                  geometry_level="local_projective", unit="reference_px",
                  independent_validation=False, fit_current_xy=[], fit_reference_xy=[],
                  fit_points_role="photometric_sampling_not_independent_correspondences",
                  sampling_pool_current_xy=[], iterations=[],
                  protection=dict(outside_mask_constant=0., gaussian_kernel=[5, 5],
                      gaussian_sigma=0.8, gaussian_radius_px=2,
                      safe_mask_erosion_radius_px=_SAFE_RADIUS,
                      reference_interpolation="explicit_bilinear_all_four_corners_safe",
                      derivative="analytic_from_same_four_bilinear_values",
                      validity_ranges="inclusive_exclusive_indices_into_sampling_pool",
                      mask_semantics="caller_supplied_positive_pixels_only",
                      mask_correctness_independently_verified=False),
                  limits=dict(max_iterations=35, max_sampling_points=_MAX_POINTS,
                      minimum_sampling_points=_MIN_POINTS, minimum_std_0_to_1=_MIN_STD,
                      minimum_correlation=_MIN_CORRELATION,
                      minimum_initial_support_retained=0.9, huber_scale_0_to_1=_HUBER,
                      maximum_gain=2., minimum_gain=0.5, maximum_abs_bias=0.35),
                  limitations=["small_motion_local_refinement_requires_a_credible_initial_map",
                      "photometric_fit_is_not_independent_geometric_accuracy",
                      "mask_semantics_and_ice_plane_membership_are_callers_responsibility",
                      "outside_fitting_support_is_extrapolation"])

    def fail(reason):
        result["reason"] = reason
        return result

    if motion_model not in ("homography", "affine", "similarity", "homography_regularized"):
        return fail("invalid_motion_model")
    if not np.isfinite(perspective_prior_weight) or perspective_prior_weight < 0:
        return fail("invalid_perspective_prior_weight")
    regularization = perspective_prior_weight if motion_model == "homography_regularized" else 0.
    active = np.array([0,1,2,3,4,5,8,9]) if motion_model == "affine" else np.arange(10)
    basis = None
    if motion_model == "similarity":
        # p = [h00,h01,tx,h10,h11,ty,h20,h21,log_gain,bias].
        # Updates preserve [[a,-b,tx],[b,a,ty],[0,0,1]] exactly; exposure
        # contributes two separate parameters, not additional geometric DOF.
        basis = np.zeros((10, 6))
        basis[0, 0] = basis[4, 0] = 1.
        basis[1, 1], basis[3, 1] = -1., 1.
        basis[2, 2] = basis[5, 3] = basis[8, 4] = basis[9, 5] = 1.
    parameter_count = len(active) if basis is None else basis.shape[1]
    result["motion_model"] = motion_model
    result["model_assumption"] = ("similarity_approximation_not_a_physical_camera_or_metric_model"
        if motion_model == "similarity" else "affine_approximation_not_a_physical_camera_or_metric_model"
        if motion_model == "affine" else "local_projective_photometric_approximation")
    result["perspective_prior"] = dict(weight=regularization, target_normalized_h20_h21=[0.,0.],
        coordinate_normalization="image_center_origin_max_dimension_half_unit",
        objective="mean_huber_data_loss + 0.5*weight*(normalized_h20^2+normalized_h21^2)",
        source="explicit_model_prior_not_observed_landmarks")
    result["optimized_parameter_count_including_gain_bias"] = parameter_count
    if basis is not None:
        result["geometric_parameter_count"] = 4
        result["parameter_constraints"] = ["h00=h11", "h01=-h10", "h20=h21=0", "h22=1"]
        result["parameter_basis"] = basis.tolist()
    try:
        ref, ref_safe = _isolated(reference, ref_mask)
        cur, cur_safe = _isolated(current, cur_mask)
        h0 = np.asarray(initial_reference_from_current, dtype=np.float64)
        if h0.shape != (3, 3) or not np.all(np.isfinite(h0)) or abs(h0[2, 2]) < 1e-10:
            return fail("invalid_initial_homography")
        h0 = h0/h0[2, 2]
        if abs(np.linalg.det(h0)) < 1e-10:
            return fail("singular_initial_homography")
    except (TypeError, ValueError, cv2.error) as exc:
        result["input_error"] = str(exc)
        return fail("invalid_input")
    height, width = cur.shape
    if min(height, width, *ref.shape) < 16:
        return fail("image_too_small")
    # Different-sized inputs are allowed: each normalizer belongs to its image.
    cs, rs = max(cur.shape)/2., max(ref.shape)/2.
    nc = np.array([[1/cs, 0, -(width-1)/(2*cs)],
                   [0, 1/cs, -(height-1)/(2*cs)], [0, 0, 1.]])
    nr = np.array([[1/rs, 0, -(ref.shape[1]-1)/(2*rs)],
                   [0, 1/rs, -(ref.shape[0]-1)/(2*rs)], [0, 0, 1.]])
    nri = np.linalg.inv(nr)
    hn = nr @ h0 @ np.linalg.inv(nc)
    if abs(hn[2, 2]) < 1e-10:
        return fail("initial_homography_has_a_projective_horizon")
    hn /= hn[2, 2]
    result["requested_initial_H_reference_from_current"] = h0.tolist()
    if motion_model in ("affine", "similarity"):
        # First-order approximation at the current image centre. Merely dropping
        # a projective denominator would change both its translation and slope.
        hn[:2,:2] -= np.outer(hn[:2,2], hn[2,:2])
        hn[2,:2] = 0.
        if motion_model == "similarity":
            # Closest orientation-preserving similarity derivative at the
            # current image centre; the requested and constrained seeds are
            # both retained. The model-wrapper calls this with identity.
            a = (hn[0, 0] + hn[1, 1]) / 2.
            b = (hn[1, 0] - hn[0, 1]) / 2.
            hn[:2, :2] = [[a, -b], [b, a]]
        h0 = nri @ hn @ nc
        h0 /= h0[2,2]
    result["model_initial_H_reference_from_current"] = h0.tolist()
    p = np.r_[hn.ravel()[:8], 0., 0.]
    corners = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], float)
    initial_corners = _project(h0, corners)
    initial_denominator = (np.c_[corners, np.ones(4)] @ h0.T)[:, 2]
    initial_area = cv2.contourArea(initial_corners.astype(np.float32), oriented=True)
    if (not np.all(np.isfinite(initial_corners)) or initial_area <= 1e-6
            or np.min(initial_denominator)*np.max(initial_denominator) <= 0):
        return fail("initial_homography_has_a_horizon_or_reflection")
    step = max(1, int(math.ceil(math.sqrt(np.count_nonzero(cur_safe)/_MAX_POINTS))))
    yy, xx = np.mgrid[step//2:height:step, step//2:width:step]
    pool = np.c_[xx.ravel(), yy.ravel()]
    pool = pool[cur_safe[pool[:, 1], pool[:, 0]]]
    if len(pool) > _MAX_POINTS:
        pool = pool[np.linspace(0, len(pool)-1, _MAX_POINTS, dtype=np.int64)]
    initial_ids, _, _, _ = _bilinear(ref, ref_safe, _project(h0, pool))
    pool = pool[initial_ids]
    result["sampling_pool_current_xy"] = pool.tolist()
    result["sampling_grid_step_px"] = step
    initial_support = _support(pool, cur.shape)
    result["initial_support"] = initial_support
    if len(pool) < _MIN_POINTS:
        return fail("insufficient_safe_overlap")
    if (initial_support["hull_fraction"] < .02 or initial_support["span_x_fraction"] < .18
            or initial_support["span_y_fraction"] < .1):
        return fail("insufficient_spatial_support")
    normalized = _project(nc, pool)
    current_values = cur[pool[:, 1], pool[:, 0]]
    # Only safe sampled intensities influence texture and exposure estimates.
    _, reference_values, _, _ = _bilinear(ref, ref_safe, _project(h0, pool))
    result["initial_texture_std"] = dict(reference=float(np.std(reference_values)),
                                        current=float(np.std(current_values)))
    if min(np.std(reference_values), np.std(current_values)) < _MIN_STD:
        return fail("insufficient_texture")
    exposure = np.linalg.lstsq(np.c_[reference_values, np.ones(len(pool))],
                               current_values, rcond=None)[0]
    # A bad initial alignment can give an unphysical OLS gain: start neutral.
    if .5 <= exposure[0] <= 2. and abs(exposure[1]) <= .35:
        p[8:] = [math.log(exposure[0]), exposure[1]]
    max_correction = max(5., min(40., .04*math.hypot(width, height)))
    result["limits"]["maximum_correction_from_initial_at_image_corners_px"] = max_correction

    def evaluate(params, iteration, scale):
        h = nri @ _unpack(params) @ nc
        if not np.all(np.isfinite(params)) or not np.all(np.isfinite(h)) or abs(h[2, 2]) < 1e-10:
            row = dict(evaluation=len(result["iterations"]), iteration=iteration,
                       step_scale=scale, accepted=False, H_reference_from_current=None,
                       gain=None, bias=None, valid_count=0, valid_pool_index_ranges=[],
                       robust_loss=None, guard="nonfinite_or_singular_proposal")
            result["iterations"].append(row)
            return dict(row=row, guard=row["guard"])
        h /= h[2, 2]
        mapped = _project(h, pool)
        ids, values, gx, gy = _bilinear(ref, ref_safe, mapped)
        # Bounding proposals before exp also makes pathological solves safe.
        gain = float(np.exp(np.clip(params[8], -50, 50)))
        residual = gain*values + params[9] - current_values[ids]
        row = dict(evaluation=len(result["iterations"]), iteration=iteration,
                   step_scale=scale, accepted=False, H_reference_from_current=h.tolist(),
                   gain=gain, bias=float(params[9]), valid_count=len(ids),
                   valid_pool_index_ranges=_ranges(ids),
                   robust_loss=_loss(residual) if len(ids) else None)
        result["iterations"].append(row)
        guard = None
        if not .5 <= gain <= 2. or abs(params[9]) > .35:
            guard = "exposure_out_of_bounds"
        elif len(ids) < max(_MIN_POINTS, .9*len(pool)):
            guard = "safe_support_lost"
        hc = np.c_[corners, np.ones(4)] @ h.T
        projected_corners = _project(h, corners)
        area = cv2.contourArea(projected_corners.astype(np.float32), oriented=True)
        if (not np.all(np.isfinite(projected_corners)) or np.min(hc[:, 2])*np.max(hc[:, 2]) <= 0
                or area*initial_area <= 0 or not .65 <= abs(area/initial_area) <= 1.55
                or np.max(np.linalg.norm(projected_corners-initial_corners, axis=1)) > max_correction):
            guard = "motion_or_projective_geometry_out_of_bounds"
        row["guard"] = guard
        return dict(h=h, ids=ids, values=values, gx=gx, gy=gy, mapped=mapped,
                    residual=residual, row=row, guard=guard)

    def jacobian(state, params):
        ids = state["ids"]
        x, y = normalized[ids].T
        hn = _unpack(params)
        d = hn[2, 0]*x + hn[2, 1]*y + 1
        u = (hn[0, 0]*x + hn[0, 1]*y + hn[0, 2])/d
        v = (hn[1, 0]*x + hn[1, 1]*y + hn[1, 2])/d
        gx, gy = state["gx"]*rs, state["gy"]*rs
        gain = np.exp(params[8])
        # Exact derivative of the same safe bilinear interpolation used above.
        geometry = np.column_stack((gx*x, gx*y, gx, gy*x, gy*y, gy,
                                    -gx*u*x-gy*v*x, -gx*u*y-gy*v*y))/d[:, None]
        return np.column_stack((gain*geometry, gain*state["values"], np.ones(len(ids))))

    def prior_loss(params):
        return float(.5*regularization*np.sum(params[6:8]**2))

    def reduced_jacobian(state, params):
        full = jacobian(state, params)
        return full[:, active] if basis is None else full @ basis

    state = evaluate(p, 0, 0.)
    if state["guard"]:
        return fail(state["guard"])
    state["row"]["accepted"] = True
    result["initial_correlation"] = _correlation(state["values"], current_values[state["ids"]])
    result["initial_rmse_0_to_1"] = float(np.sqrt(np.mean(state["residual"]**2)))
    converged = False
    last_rank_ratio = None
    for iteration in range(1, 36):
        jac = reduced_jacobian(state, p)
        weights = np.minimum(1., _HUBER/np.maximum(np.abs(state["residual"]), 1e-15))
        weighted = jac*np.sqrt(weights[:, None])
        columns = np.maximum(np.linalg.norm(weighted, axis=0), 1e-12)
        scaled = weighted/columns
        singular = np.linalg.svd(scaled, compute_uv=False)
        last_rank_ratio = float(singular[-1]/singular[0])
        if last_rank_ratio < 1e-5:
            result["design_rank_ratio"] = last_rank_ratio
            return fail("photometric_system_rank_deficient")
        rhs = -(state["residual"]*np.sqrt(weights))
        if regularization:
            prior_jac = np.zeros((2,len(active)))
            prior_jac[0,6] = prior_jac[1,7] = math.sqrt(len(state["ids"])*regularization)
            weighted = np.vstack([weighted,prior_jac])
            rhs = np.r_[rhs,-prior_jac @ p[active]]
            columns = np.maximum(np.linalg.norm(weighted,axis=0),1e-12)
            scaled = weighted/columns
        # Column-scaled LM damping is numerical; the separate perspective prior
        # above is declared and never counted as observed photometric support.
        step = np.linalg.solve(scaled.T@scaled + 1e-5*np.eye(parameter_count), scaled.T@rhs)/columns
        if basis is None:
            delta = np.zeros(10)
            delta[active] = step
        else:
            delta = basis @ step
        improved = False
        for line in range(9):
            scale = 0.5**line
            proposal = p + scale*delta
            trial = evaluate(proposal, iteration, scale)
            if trial["guard"]:
                continue
            common, a, b = np.intersect1d(state["ids"], trial["ids"], return_indices=True)
            # Compare on identical pixels: dropping difficult pixels is no gain.
            before, after = _loss(state["residual"][a]), _loss(trial["residual"][b])
            trial["row"]["common_count"] = len(common)
            trial["row"]["common_loss_before"] = before
            trial["row"]["common_loss_after"] = after
            trial["row"]["perspective_prior_loss"] = prior_loss(proposal)
            if after + prior_loss(proposal) <= before + prior_loss(p) + 1e-15:
                motion = float(np.max(np.linalg.norm(trial["mapped"][common]-state["mapped"][common], axis=1)))
                trial["row"]["accepted"] = True
                p, state = proposal, trial
                improved = True
                if motion < 1e-3 and np.max(np.abs(scale*delta[8:])) < 1e-5:
                    converged = True
                break
        if converged or not improved:
            break
    ids = state["ids"]
    corr = _correlation(state["values"], current_values[ids])
    final_jac = reduced_jacobian(state, p)
    final_weights = np.minimum(1., _HUBER/np.maximum(np.abs(state["residual"]), 1e-15))
    final_weighted = final_jac*np.sqrt(final_weights[:, None])
    final_scaled = final_weighted/np.maximum(np.linalg.norm(final_weighted, axis=0), 1e-12)
    final_singular = np.linalg.svd(final_scaled, compute_uv=False)
    last_rank_ratio = float(final_singular[-1]/final_singular[0])
    result.update(final_correlation=corr, gain=float(np.exp(p[8])), bias=float(p[9]),
                  final_rmse_0_to_1=float(np.sqrt(np.mean(state["residual"]**2))),
                  design_rank_ratio=last_rank_ratio, converged=converged,
                  safe_support_retained=float(len(ids)/len(pool)),
                  support=_support(pool[ids], cur.shape), final_evaluation=state["row"]["evaluation"],
                  fit_current_xy=pool[ids].tolist(), fit_reference_xy=state["mapped"][ids].tolist())
    result["perspective_prior"]["final_penalty"] = prior_loss(p)
    if last_rank_ratio < 1e-5:
        return fail("photometric_system_rank_deficient")
    final_support = result["support"]
    if (final_support["hull_fraction"] < .02 or final_support["span_x_fraction"] < .18
            or final_support["span_y_fraction"] < .1):
        return fail("insufficient_final_spatial_support")
    if corr < _MIN_CORRELATION:
        return fail("insufficient_photometric_correlation")
    if min(np.std(state["values"]), np.std(current_values[ids])) < _MIN_STD:
        return fail("insufficient_final_texture")
    result.update(status="candidate", reason="photometric_candidate_requires_independent_geometry_check",
                  H_reference_from_current=state["h"].tolist())
    return result
