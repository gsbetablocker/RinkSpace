"""Source localization support for the accepted first-shot extension.

This is a fixed, first-order propagation of the saved native material support.
It never adjusts an accepted UV, refits scene/camera parameters, or treats sparse
representatives of one source estimate as independent observations. The scales
describe localization support; they are not calibrated stochastic noise.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .official_static_surface_dataset import read_reference, reference
from .source_evidence_whitening import latent_whitening_operator, MissingSourceEvidence, SourceWhiteningOperator


def _positive(values, n, label):
    values = np.asarray(values, float)
    if values.shape != (n,) or not np.isfinite(values).all() or (values <= 0).any():
        raise MissingSourceEvidence(label + " requires its measured positive native scales")
    return values


def _influence(design, weight):
    """Fixed source estimator differential; null directions add no noise."""
    design = np.asarray(design, float)
    weight = np.asarray(weight, float)
    root = np.sqrt(weight)
    weighted = root[:, None]*design
    u, singular, vt = np.linalg.svd(weighted, full_matrices=False)
    threshold = np.finfo(float).eps*max(weighted.shape)*np.linalg.norm(weighted, ord="fro")
    keep = singular > threshold
    if not keep.any():
        raise MissingSourceEvidence("Native material readouts identify no source estimator mode")
    return ((vt[keep].T/singular[keep]) @ u[:, keep].T)*root


FINAL_LINE_SCHEMA = "rinkspace.fixed-final-native-support-line.v1"
DERIVED_LINE_ORIGIN = "same_final_native_support_fixed_chart_exact_mean_differential"


def rebuild_final_native_line_estimator(line, feature_id, source_point_ids, original_uv, *, group_id):
    """One fixed-chart mean and its exact native-noise differential.

    The original representative stations and final native support are fixed.
    These are correlated outputs of that support, never additional readouts.
    Native sigma is in the recorded navigation-normal coordinate. Projecting
    the derived mean to its fitted physical normal therefore includes N dot n.
    """
    ids = tuple(source_point_ids)
    uv = np.asarray(original_uv, float)
    if (uv.shape != (len(ids), 2) or not len(ids) or len(set(ids)) != len(ids)
            or not np.isfinite(uv).all() or line.get("current_frame_only") is not True):
        raise MissingSourceEvidence("A complete current-native line and original representative inventory is required")
    rows = line.get("direct_readouts", [])
    if len(rows) < 2:
        raise MissingSourceEvidence("The final native line needs at least two measured readouts")
    points = np.asarray([r["uv"] for r in rows], float)
    sigma = _positive([r["sigma_px"] for r in rows], len(rows), feature_id)
    origin = np.asarray(line["point_origin"], float)
    tangent = np.asarray(line["unit_tangent"], float)
    normal = np.asarray(line["unit_normal"], float)
    if (points.shape != (len(rows), 2) or not np.isfinite(points).all()
            or any(a.shape != (2,) or not np.isfinite(a).all() for a in (origin, tangent, normal))
            or not np.allclose(np.array([tangent, normal]) @ np.array([tangent, normal]).T,
                               np.eye(2), rtol=0, atol=1e-12)):
        raise MissingSourceEvidence("The final native line requires its finite orthonormal recorded chart")
    broad = feature_id.split(".")[-1].split(":")[0] in ("B1", "B2", "C")
    if broad:
        signal = np.asarray([min(r["source_evidence"]["signed_interface_iqr_contrast"]) for r in rows])
        weights = _positive(signal, len(rows), "Actual two-interface material contrast")**2
        weights_policy = "minimum_two_interface_contrast_squared"
    else:
        weights = np.ones(len(rows))
        weights_policy = "uniform_current_native_line_support"
    station = (points-origin) @ tangent
    design = np.column_stack((station, np.ones(len(station))))
    if np.linalg.matrix_rank(design) != 2:
        raise MissingSourceEvidence("Final native line stations do not identify both coefficients")
    influence = _influence(design, weights)
    coefficients = influence @ ((points-origin) @ normal)
    representative_station = (uv-origin) @ tangent
    output_design = np.column_stack((representative_station, np.ones(len(ids))))
    mean = origin + representative_station[:, None]*tangent + (output_design @ coefficients)[:, None]*normal
    physical_normal = normal-coefficients[0]*tangent
    physical_normal /= np.linalg.norm(physical_normal)
    projection = float(physical_normal @ normal)
    basis = np.zeros((len(ids), 2*len(ids)))
    for j in range(len(ids)):
        basis[j, 2*j:2*j+2] = physical_normal
    sensitivity = projection*output_design
    noise_factor = influence*sigma
    operator = latent_whitening_operator(ids, sensitivity, explicit_noise_factor=noise_factor,
        physical_basis=basis, group_id=group_id, covariance_origin=DERIVED_LINE_ORIGIN)
    metadata = dict(schema=FINAL_LINE_SCHEMA, kind="weighted_linear_final_native_support",
        point_origin=origin.tolist(), unit_tangent=tangent.tolist(), unit_normal=normal.tolist(),
        normal_offset_coefficients=coefficients.tolist(), new_physical_normal=physical_normal.tolist(),
        weights_policy=weights_policy, final_support_refit_unconstrained=True,
        family_direction_used_for_candidate_disambiguation_only=True,
        source_sigma_coordinates="recorded_navigation_normal_coverage_px",
        derivative_normal_projection_factor_required=True,
        physical_sensitivity_normal_projection_factor=projection,
        representative_station_fixed_at_original_source_uv=True,
        final_native_readout_count=len(rows),
        final_native_support_canonical_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(),
        D_sha256=hashlib.sha256(influence.tobytes()).hexdigest(),
        native_sigma_sha256=hashlib.sha256(sigma.tobytes()).hexdigest(),
        weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
        noise_factor_sha256=hashlib.sha256(noise_factor.tobytes()).hexdigest(),
        physical_sensitivity_sha256=hashlib.sha256(sensitivity.tobytes()).hexdigest(),
        raw_mean_differential="normal outer [fixed_station,1] times D",
        source_mean_and_W_from_same_final_native_support=True,
        native_readouts_added_as_independent_information=False)
    return dict(uv=mean, stations=representative_station, coefficients=coefficients,
        normal=physical_normal, influence=influence, noise_factor=noise_factor,
        physical_basis=basis, physical_sensitivity=sensitivity, operator=operator, metadata=metadata)


def _normal(points):
    points = np.asarray(points, float)
    if len(points) < 2:
        raise MissingSourceEvidence("A material trace requires at least two source stations")
    tangent = np.gradient(points, axis=0)
    lengths = np.linalg.norm(tangent, axis=1)
    if (lengths <= 0).any():
        raise MissingSourceEvidence("Native material stations do not define a tangent")
    tangent /= lengths[:, None]
    return np.column_stack((-tangent[:, 1], tangent[:, 0]))


@dataclass
class _Curve:
    key: str
    origin: np.ndarray
    tangent: np.ndarray
    coefficients: np.ndarray
    covariance: np.ndarray
    metadata: dict
    linear: bool
    direct_points: np.ndarray | None = None
    direct_normals: np.ndarray | None = None
    noise_factor: np.ndarray | None = None

    def normal(self, uv):
        if self.direct_points is not None:
            index = int(np.argmin(np.linalg.norm(self.direct_points-np.asarray(uv), axis=1)))
            return self.direct_normals[index]
        if self.linear:
            return np.array([-self.tangent[1], self.tangent[0]])
        t = self.station(uv)
        tangent = self.coefficients[1]+2*t*self.coefficients[2]
        return np.array([-tangent[1], tangent[0]])/np.linalg.norm(tangent)

    def station(self, uv):
        point = np.asarray(uv, float)
        if self.linear:
            return float((point - self.origin) @ self.tangent)
        c = self.coefficients
        # The squared distance derivative of a quadratic is a cubic. The
        # accepted UV is used only to locate its station in the saved trace.
        z = c.copy(); z[0] -= point
        polynomial = sum((np.polynomial.polynomial.polymul(z[:, axis],
                           np.array([c[1, axis], 2*c[2, axis]])) for axis in (0, 1)))
        roots = np.polynomial.polynomial.polyroots(polynomial)
        candidates = [0., 1.] + [float(r.real) for r in roots if abs(r.imag) < 1e-8 and 0 <= r.real <= 1]
        return min(candidates, key=lambda t: np.linalg.norm(self.evaluate(t)-point))

    def evaluate(self, t):
        if self.linear:
            return self.origin + self.tangent*t
        return np.array([1., t, t*t]) @ self.coefficients

    def sensitivity(self, uv, *, project_normal=True):
        if self.direct_points is not None:
            index = int(np.argmin(np.linalg.norm(self.direct_points-np.asarray(uv), axis=1)))
            result = np.zeros((2, len(self.direct_points)))
            result[:, index] = self.direct_normals[index]
            return result
        t = self.station(uv)
        if self.linear:
            n = np.array([-self.tangent[1], self.tangent[0]])
            return np.outer(n, [t, 1.])
        basis = np.array([1., t, t*t])
        result = np.zeros((2, 6))
        result[0, ::2] = basis; result[1, 1::2] = basis
        if project_normal:
            tangent = self.coefficients[1] + 2*t*self.coefficients[2]
            n = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
            result = np.outer(n, n) @ result
        return result


def _curve_key(fid, segment):
    address = [fid, segment.get("measurement_origin"), segment.get("source_native_readout_uv"),
               segment.get("source_native_transition_iqr_px"), segment.get("source_curve_degree")]
    return hashlib.sha256(json.dumps(address, separators=(",", ":")).encode()).hexdigest()


def _board_curve(fid, segment, packet_reference):
    p = np.asarray(segment.get("source_native_readout_uv", []), float)
    if not segment.get("representative_points_share_line_estimate"):
        p = np.asarray(segment["uv"], float)
        n = len(p)
        sigmas = next((np.asarray(segment[k], float) for k in ("sigma_px", "localization_sigma_px")
                       if k in segment and len(segment[k]) == n), None)
        if sigmas is None:
            raise MissingSourceEvidence("A direct native board edge requires its original localization scales")
        sigmas = _positive(sigmas, n, fid)
        normal = _normal(p)
        key = _curve_key(fid, segment)
        # Include actual native coordinates in the identity of direct readouts.
        key = hashlib.sha256((key+json.dumps(segment["uv"])).encode()).hexdigest()
        metadata = dict(source_feature_id=fid, source_readout_packet=packet_reference,
            native_readout_count=n, source_sigma_origin="original_direct_native_localization_sigma_px",
            covariance_method="independent_native_profile_normal_support_shared_with_derived_corners")
        return _Curve(key, np.zeros(2), np.zeros(2), np.empty((0, 2)), np.diag(sigmas**2),
                      metadata, False, p, normal, np.diag(sigmas))
    if p.ndim != 2 or p.shape[1:] != (2,) or len(p) < 2 or not np.isfinite(p).all():
        raise MissingSourceEvidence("Saved actual board material readouts are required: " + fid)
    widths = _positive(segment.get("source_native_transition_iqr_px", []), len(p), fid)
    sigma = widths / 1.349
    key = _curve_key(fid, segment)
    metadata = dict(source_feature_id=fid, source_readout_packet=packet_reference,
        native_readout_count=len(p), source_readout_digest=key,
        source_sigma_origin="source_native_transition_iqr_px/1.349",
        covariance_method="fixed_first_order_native_material_localization_support",
        accepted_UV_modified=False, camera_or_scene_fit_used=False)
    smooth = segment.get("source_curve_degree") == 2
    if not smooth:
        fit = cv2.fitLine(p.astype(np.float32), cv2.DIST_HUBER, 0, .001, .001).ravel().astype(float)
        tangent = fit[:2] / np.linalg.norm(fit[:2]); origin = fit[2:]
        n = np.array([-tangent[1], tangent[0]])
        t = (p-origin) @ tangent; residual = (p-origin) @ n
        design = np.column_stack((t, np.ones(len(t))))
        # OpenCV's recorded Huber source estimator uses C=1.345.
        weights = np.minimum(1., 1.345 / np.maximum(np.abs(residual), np.finfo(float).tiny))
        influence = _influence(design, weights)
        noise_factor = influence*sigma
        covariance = noise_factor @ noise_factor.T
        metadata.update(source_estimator="saved_native_cv2_DIST_HUBER_line",
            huber_C=1.345, linearized_on_current_saved_material_readouts=True,
            covariance_parameter_order=["normal_slope_per_native_station", "normal_offset_px"])
        return _Curve(key, origin, tangent, np.empty((0, 2)), covariance, metadata, True,
                      noise_factor=noise_factor)
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if distance[-1] <= 0:
        raise MissingSourceEvidence("Source curve has no measured length")
    t = distance / distance[-1]
    design = np.column_stack((np.ones(len(t)), t, t*t))
    # No nonlinear source re-fit is performed. These source-only control
    # coordinates establish the local chart for the uncertainty differential.
    coefficients = np.linalg.lstsq(design, p, rcond=None)[0]
    residual = design @ coefficients-p
    loss_scale = np.maximum(widths, 1.)
    tangent = coefficients[1]+2*t[:, None]*coefficients[2]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    physical_design = np.zeros((len(p), 6))
    physical_design[:, ::2] = design*normal[:, 0, None]
    physical_design[:, 1::2] = design*normal[:, 1, None]
    normal_residual = np.einsum('ij,ij->i', residual, normal)
    weights = (1+(normal_residual/loss_scale)**2)**(-1.5)/loss_scale**2
    # Only native material normal localization is measured. Conditioning
    # arclength station/X coordinates as noiseless would give a nearly
    # horizontal quadratic an artificial 1e-7-pixel shape variance. Use the
    # physical normal information, retaining its observable shape modes and
    # discarding tangent reparameterization as unmeasured, never as certainty.
    influence = _influence(physical_design, weights)
    noise_factor = influence*sigma
    covariance = noise_factor @ noise_factor.T
    metadata.update(source_estimator="saved_native_quadratic_soft_l1_material_curve",
        chart="quadratic_native_arclength_local_chart",
        chart_coefficients=coefficients.tolist(),
        differential_chart_is_not_a_reestimate_of_accepted_UV=True,
        saved_loss_scale="maximum(source_native_transition_iqr_px,1)",
        covariance_method="native_material_physical_normal_design_readout_influence",
        source_coordinate_fit_conditioning_not_interpreted_as_source_precision=True,
        unmeasured_parametric_tangent_directions_are_not_zero_noise_constraints=True,
        loss_f_scale=1., covariance_parameter_order=["c0x", "c0y", "c1x", "c1y", "c2x", "c2y"])
    return _Curve(key, np.zeros(2), np.zeros(2), coefficients, covariance, metadata, False,
                  noise_factor=noise_factor)


def _block_covariance(curves):
    offsets = {}; cursor = 0
    for curve in curves:
        offsets[curve.key] = cursor; cursor += len(curve.covariance)
    result = np.zeros((cursor, cursor))
    for curve in curves:
        k = offsets[curve.key]; d = len(curve.covariance)
        result[k:k+d, k:k+d] = curve.covariance
    return result, offsets


def _block_noise_factor(curves):
    offsets = {}; cursor = columns = 0
    for curve in curves:
        if curve.noise_factor is None:
            raise MissingSourceEvidence("The original material readout noise factor is required")
        offsets[curve.key] = cursor
        cursor += curve.noise_factor.shape[0]; columns += curve.noise_factor.shape[1]
    result = np.zeros((cursor, columns)); column = 0
    for curve in curves:
        start = offsets[curve.key]; rows, count = curve.noise_factor.shape
        result[start:start+rows, column:column+count] = curve.noise_factor
        column += count
    return result, offsets


def _line_coefficients(points, axis):
    fit = cv2.fitLine(np.asarray(points, np.float32), cv2.DIST_HUBER, 0, .001, .001).ravel().astype(float)
    denominator = fit[axis]
    if abs(denominator) < 1e-10:
        raise MissingSourceEvidence("The saved material side cannot identify its declared line chart")
    a = fit[1-axis]/denominator
    return np.array([a, fit[3-axis]-a*fit[2+axis]])


def _feature_line(feature, charts, offsets, dimension, axis):
    points = []; sensitivity = []
    for segment in feature["segments"]:
        curve = charts[_board_curve(feature["id"], segment, {}).key]
        for point in segment["uv"]:
            # The differential's nominal points belong to the saved source
            # chart. They are never substituted for the approved FIT UV.
            nominal = point if curve.direct_points is not None else curve.evaluate(curve.station(point))
            points.append(nominal)
            block = np.zeros((2, dimension)); start = offsets[curve.key]
            # A boundary has no material correspondence along its tangent.
            # Its derived corner must not receive reparameterization noise.
            block[:, start:start+len(curve.covariance)] = curve.sensitivity(nominal, project_normal=True)
            sensitivity.append(block)
    points = np.asarray(points, float); sensitivity = np.stack(sensitivity)
    initial = _line_coefficients(points, axis)
    initial_normal = np.array([-initial[0], 1.]) if axis == 0 else np.array([1., -initial[0]])
    initial_normal /= np.linalg.norm(initial_normal)
    residual = (points-points.mean(axis=0)) @ initial_normal
    weight = np.minimum(1., 1.345/np.maximum(np.abs(residual), np.finfo(float).tiny))
    weight /= weight.sum()
    center = weight @ points; centered = points-center
    scatter = centered.T @ (weight[:, None]*centered)
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    tangent = eigenvectors[:, -1]; normal = eigenvectors[:, 0]
    gap = float(eigenvalues[-1]-eigenvalues[0])
    if gap <= 0 or abs(tangent[axis]) < 1e-10:
        raise MissingSourceEvidence("Source side chart has no identifiable line orientation")
    a = tangent[1-axis]/tangent[axis]
    coefficient = np.array([a, center[1-axis]-a*center[axis]])
    dcenter = np.einsum('i,ijk->jk', weight, sensitivity)
    # Analytic weighted TLS eigenvector differential; no float32 finite
    # differences can manufacture an independent corner/side noise mode.
    delta_scatter_times_tangent = np.einsum('i,ijk,il,l->jk', weight, sensitivity, centered, tangent)
    delta_scatter_times_tangent += np.einsum('i,ij,ilk,l->jk', weight, centered, sensitivity, tangent)
    dtangent = normal[:, None]*(normal @ delta_scatter_times_tangent)[None, :]/gap
    da = (dtangent[1-axis]*tangent[axis]-tangent[1-axis]*dtangent[axis])/tangent[axis]**2
    db = dcenter[1-axis]-a*dcenter[axis]-center[axis]*da
    return coefficient, np.vstack((da, db))


def _ellipse_normal(feature, points):
    g = feature.get("geometric_interpretation", {})
    if not all(k in g for k in ("center_uv", "axes_px", "angle_degrees")):
        raise MissingSourceEvidence("Saved current-source circle geometry is required for its normal")
    center = np.asarray(g["center_uv"], float); axes = np.asarray(g["axes_px"], float)/2
    angle = np.deg2rad(g["angle_degrees"])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    matrix = rotation @ np.diag(1/axes**2) @ rotation.T
    normal = (np.asarray(points, float)-center) @ matrix
    length = np.linalg.norm(normal, axis=1)
    if (length <= 0).any():
        raise MissingSourceEvidence("The current source circle cannot supply a finite material normal")
    return normal/length[:, None]


def _affine_basis(point):
    x, y = (np.asarray(point, float)-[960., 540.])/1000.
    return np.array([[1., 0., x, y, 0., 0.], [0., 1., 0., 0., x, y]])


def _current_board_profiles(fid, path, frame):
    """Existing board material reader, used for uncertainty support only."""
    from .extract_first_shot_board_source import profile_read
    for name in ("source_image", "mask"):
        if reference(frame[name]["path"]) != frame[name]:
            raise MissingSourceEvidence("Native uncertainty support bytes changed: " + name)
    raw = cv2.imread(frame["source_image"]["path"])
    mask = cv2.imread(frame["mask"]["path"], cv2.IMREAD_GRAYSCALE)
    if raw is None or mask is None:
        raise MissingSourceEvidence("Current native side RGB/mask support is missing")
    ad, role = fid.split("."); number = int(ad[2:])
    weights = [-1., .5, .5] if number == 6 else [1., -.5, -.5] if number in (3, 5, 7) else [.5, -1., .5]
    direction = {"top_edge": [0, 1], "bottom_edge": [0, -1],
                 "left_vertical": [1, 0], "right_vertical": [-1, 0]}[role]
    return profile_read(raw, mask != 0, np.asarray(path, float), weights, 24,
        maximum_blur=18., minimum_amplitude=10., gradient_direction=direction, signal_mode="chroma")[0]


def _supplement_side(fid, source_features, frame):
    ad, role = fid.split("."); side = role.split("_")[0]
    corners = {}
    for feature in source_features.values():
        if not feature["id"].startswith(ad+"."):
            continue
        for segment in feature["segments"]:
            for corner, index in segment.get("corner_vertex_indices", {}).items():
                if corner.startswith(side+"_"):
                    corners[corner.split("_")[1]] = np.asarray(segment["uv"][index], float)
    if not corners:
        raise MissingSourceEvidence("Missing side has no authenticated adjacent source corner")
    if len(corners) == 2:
        anchors = np.array([corners["top"], corners["bottom"]])
    else:
        level, corner = next(iter(corners.items()))
        opposite = "left" if side == "right" else "right"
        other = source_features.get(ad+"."+opposite+"_vertical")
        if other is None:
            horizontal = source_features.get(ad+"."+("bottom" if level == "top" else "top")+"_edge")
            if horizontal is None:
                raise MissingSourceEvidence("Clipped corner has no finite current native side search extent")
            material = np.vstack([s["uv"] for s in horizontal["segments"]])
            direction = np.array([0., 1.])
            span = abs(float(np.median(material[:, 1])-corner[1]))
        else:
            material = np.vstack([s.get("source_native_readout_uv", s["uv"]) for s in other["segments"]])
            direction = cv2.fitLine(material.astype(np.float32), cv2.DIST_HUBER, 0, .001, .001).ravel()[:2].astype(float)
            if direction[1] < 0:
                direction *= -1
            span = float(np.ptp(material[:, 1]))
        # These directions/extents are search navigation only. Only actual
        # current native material transitions supply uncertainty readouts.
        anchors = np.array([corner, corner+direction*span*(1 if level == "top" else -1)])
    distance = np.linalg.norm(anchors[1]-anchors[0])
    path = np.linspace(anchors[0], anchors[1], max(2, int(np.ceil(distance/12))+1))
    measured = [r for r in _current_board_profiles(fid, path, frame) if r is not None]
    if len(measured) < 2:
        raise MissingSourceEvidence("Current native RGB cannot identify missing source side uncertainty: " + fid)
    p = np.asarray([r[0] for r in measured]); widths = np.asarray([r[1] for r in measured])
    segment = dict(uv=[p[0].tolist(), p[-1].tolist()],
        measurement_origin="current_native_joint_straight_material_edge_fit",
        representative_points_share_line_estimate=True,
        source_native_readout_uv=p.tolist(), source_native_transition_iqr_px=widths.tolist())
    return dict(id=fid, segments=[segment], uncertainty_support_only=True,
        source_image=frame["source_image"], missing_original_source_side_record=True,
        current_native_support_uv=p.tolist(), current_native_support_transition_iqr_px=widths.tolist(),
        added_as_FIT_points=False, transported_navigation_points_added_as_readouts=False)


def _manual_local_paint_evidence(point, normal, offsets, profile, valid):
    """Read one human-identified thin stroke's own nearest two shoulders.

    A stronger neighbouring shadow is neither this stroke's valley nor its
    background plateau. Asymmetric native flank contrasts remain asymmetric.
    """
    from scipy.signal import find_peaks
    from .native_ice_material_repair import _crossings
    offsets = np.asarray(offsets, float); profile = np.asarray(profile, float)
    valid = np.asarray(valid, bool)
    center = int(np.argmin(np.abs(offsets)))
    if not valid[center]:
        raise MissingSourceEvidence("Human C0 point has no unmasked current-native profile center")
    lo = hi = center
    while lo > 0 and valid[lo-1]:
        lo -= 1
    while hi+1 < len(valid) and valid[hi+1]:
        hi += 1
    off = offsets[lo:hi+1]; raw = profile[lo:hi+1]
    valleys, valley_data = find_peaks(-raw, plateau_size=(1, None))
    shoulders, shoulder_data = find_peaks(raw, plateau_size=(1, None))
    if not len(valleys):
        raise MissingSourceEvidence("The human-selected C0 stroke has no local native paint valley")
    peak_index = min(range(len(valleys)), key=lambda k: (
        abs(float(np.mean(off[valley_data['left_edges'][k]:valley_data['right_edges'][k]+1]))),
        raw[valleys[k]]))
    first = int(valley_data['left_edges'][peak_index]); last = int(valley_data['right_edges'][peak_index])
    left_candidates = shoulder_data['right_edges'][shoulder_data['right_edges'] < first]
    right_candidates = shoulder_data['left_edges'][shoulder_data['left_edges'] > last]
    if not len(left_candidates) or not len(right_candidates):
        raise MissingSourceEvidence("Both of the selected C0 stroke's own native ice shoulders are required")
    left = int(left_candidates[-1]); right = int(right_candidates[0])
    bottom = float(np.min(raw[first:last+1])); levels = np.array([.25, .5, .75])
    contrasts = np.array([raw[left]-bottom, raw[right]-bottom])
    if not np.isfinite(contrasts).all() or (contrasts <= 0).any():
        raise MissingSourceEvidence("The selected C0 native shoulders have no positive paint contrast")
    ql = _crossings(off[left:first+1], raw[left:first+1], bottom+contrasts[0]*levels, False)
    qr = _crossings(off[last:right+1], raw[last:right+1], bottom+contrasts[1]*levels, True)
    if ql is None or qr is None:
        raise MissingSourceEvidence("The selected C0 stroke's native flank quartiles are unresolved")
    quantile_offsets = np.array([np.sort(ql), np.sort(qr)])
    quantiles = point[None, None]+quantile_offsets[:, :, None]*normal[None, None]
    widths = np.abs(quantile_offsets[:, 2]-quantile_offsets[:, 0])
    sigma = float(np.max(widths)/1.349)
    _positive([sigma], 1, "manual same-stroke material flank support")
    sides = quantiles[:, 1]; measured = sides.mean(0)
    valley_offset = float(np.mean(off[first:last+1]))
    evidence = dict(measurement_origin="human_selected_native_C0_thin_stroke_own_nearest_ice_shoulders",
        profile_normal=normal.tolist(), native_profile_offsets_px=off.tolist(),
        native_median_luminance_profile=raw.tolist(),
        valley_plateau_offset_interval_px=[float(off[first]), float(off[last])],
        valley_luminance=bottom, actual_source_valley_uv=(point+normal*valley_offset).tolist(),
        actual_source_own_shoulders_uv=(point+off[[left, right], None]*normal).tolist(),
        own_shoulder_luminance=raw[[left, right]].tolist(),
        own_flank_paint_contrast=contrasts.tolist(), quantile_fractions=levels.tolist(),
        actual_source_sides_uv=sides.tolist(), actual_source_side_quantiles_uv=quantiles.tolist(),
        actual_source_side_transition_iqr_px=widths.tolist(), transition_sigma_px=sigma,
        source_sigma_formula="max(two_own_flank_transition_IQR_px)/1.349",
        sigma_floor_applied=False, neighbouring_valley_or_shadow_not_used_as_own_shoulder=True,
        complete_band_native_interval_px=[float(off[left]), float(off[right])],
        material_branch_authority="human_accepted_C0_thin_contour_and_current_native_local_valley",
        actual_source_support_outside_person_and_broadcast_masks=True,
        native_profile_shift_px=float((measured-point)@normal))
    return dict(uv=measured.tolist(), sigma_px=sigma, source_evidence=evidence)


def _manual_support(row, j, packet, source_features, frame):
    """Read the human-selected C0 thin contour without borrowing adjacent bands."""
    target = row["target_id"]
    feature = source_features.get(target)
    if feature is None:
        raise MissingSourceEvidence("Human point has no current-source material context: " + target)
    point = row["uv"][j]
    for name in ("source_image", "mask"):
        if reference(frame[name]["path"]) != frame[name]:
            raise MissingSourceEvidence("Manual native support bytes changed: " + name)
    normal = _ellipse_normal(feature, [point])[0]
    raw = cv2.imread(frame["source_image"]["path"])
    if raw is None:
        raise MissingSourceEvidence("The accepted manual point's current native RGB is missing")
    expected = packet["source"].get("decoded_BGR_sha256")
    if expected and hashlib.sha256(raw.tobytes()).hexdigest() != expected:
        raise MissingSourceEvidence("Manual point current RGB differs from its accepted source")
    mask = cv2.imread(frame["mask"]["path"], cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise MissingSourceEvidence("Manual point current native mask is missing")
    tangent = np.array([-normal[1], normal[0]])
    # The four accepted clicks identify the visible narrow C0 stroke. Its
    # locally reviewed native interval excludes the lower broad shadow band.
    offsets = np.arange(-20., 21.)
    support = point+offsets[:, None, None]*normal+np.arange(-3., 4.)[None, :, None]*tangent
    xy = np.rint(support).astype(int)
    inside = ((xy >= [0, 0]) & (xy < [raw.shape[1], raw.shape[0]])).all(axis=2)
    profile = np.full(len(offsets), np.nan); valid = np.zeros(len(offsets), bool)
    luminance = .2126*raw[:, :, 2]+.7152*raw[:, :, 1]+.0722*raw[:, :, 0]
    for k in range(len(offsets)):
        if not inside[k].all():
            continue
        x, y = xy[k].T
        if mask[y, x].any():
            continue
        profile[k] = np.median(luminance[y, x]); valid[k] = True
    measured = _manual_local_paint_evidence(point, normal, offsets, profile, valid)
    sigma = float(measured["sigma_px"])
    _positive([sigma], 1, "manual native paint support")
    return sigma, dict(source_point_id=row["source_point_ids"][j], target_id=target,
        source_image=frame["source_image"], unchanged_accepted_uv=point.tolist(),
        method="current_native_C0_human_selected_thin_stroke_own_flank_localization_support",
        normal_origin="saved_current_source_C0_normal_used_only_for_local_profile_navigation",
        reviewed_local_native_profile_radius_px=20,
        measured_sigma_px=sigma, native_material_evidence=measured["source_evidence"],
        measured_support_midpoint_uv=measured["uv"], accepted_UV_modified=False,
        measured_midpoint_is_not_a_new_fit_point=True)


def prepare_frame_evidence(observations, source_frame, *, preserved_source_scales=None,
                           physical_material_normals_by_point_id=None):
    """Return fixed source scales and correlated operators for every input ID."""
    packet = read_reference(source_frame["packet"])
    pts = source_frame["native_pts"]
    if packet["source"]["native_pts"] != pts:
        raise MissingSourceEvidence("Accepted packet and source evidence use different native exposures")
    native_packets = {}
    for name in ("board_source_readout_packet", "ice_source_readout_packet"):
        if name in packet:
            native_packets[name] = read_reference(packet[name])
            if native_packets[name]["source"]["native_pts"] != pts:
                raise MissingSourceEvidence("Native uncertainty support belongs to a different source exposure")
    board_source = {f["id"]: f for f in native_packets.get("board_source_readout_packet", {}).get("features", [])}
    ice_source = {f["id"]: f for f in native_packets.get("ice_source_readout_packet", {}).get("features", [])}
    # Historical vertical-side pruning retained corner identities and removed
    # some source support. Re-read only the absent uncertainty evidence from
    # the same authenticated native RGB; these are never new FIT members.
    board_source = dict(board_source)
    for row in observations:
        d = row["model_target"]
        if d["kind"] != "ad_corner":
            continue
        side, level = d["role"].split("_")
        fid = "Ad"+str(d["advertisement_id"])+"."+side+"_vertical"
        if fid not in board_source:
            board_source[fid] = _supplement_side(fid, board_source, source_frame)
    repaired_segments = {}
    for fid, feature in list(board_source.items()):
        if not fid.startswith("Ad"):
            continue
        for si, segment in enumerate(feature["segments"]):
            p = segment.get("source_native_readout_uv", [])
            w = segment.get("source_native_transition_iqr_px", [])
            if p and len(p) != len(w):
                readouts = _current_board_profiles(fid, p, source_frame)
                if any(r is None for r in readouts):
                    raise MissingSourceEvidence("Current native profiles cannot restore pruned width address: " + fid)
                fixed = dict(segment, source_native_transition_iqr_px=[r[1] for r in readouts])
                repaired_segments[_curve_key(fid, segment)] = fixed
                copy = dict(feature); copy["segments"] = list(feature["segments"])
                copy["segments"][si] = fixed
                feature = copy; board_source[fid] = copy
    primitive = []; locations = {}; scales = {}; uncertainty = []
    for row in observations:
        for j, pid in enumerate(row["source_point_ids"]):
            if pid in locations:
                raise ValueError("An accepted source ID entered this frame twice")
            locations[pid] = len(primitive); primitive.append((row, j, pid))
            scales[pid] = float(row["sigma"][j])
    grouped = set(); groups = []
    material_normals = physical_material_normals_by_point_id or {}
    if not set(material_normals) <= set(locations):
        raise ValueError("A physical material normal override has no accepted source point")

    def install(items, sensitivity, noise_factor, group_id, evidence, normal_directions):
        ids = [z[2] for z in items]
        if grouped.intersection(ids):
            raise ValueError("One native point would enter two source evidence groups")
        physical_rows = []; physical_sensitivity = []
        for k, direction in enumerate(normal_directions):
            block = sensitivity[2*k:2*k+2]
            if ids[k] in material_normals:
                if direction is None:
                    raise ValueError("A physical corner cannot be converted to a curve normal")
                direction = material_normals[ids[k]]["normal"]
            if direction is None:
                for axis in (0, 1):
                    basis_row = np.zeros(2*len(items)); basis_row[2*k+axis] = 1.
                    physical_rows.append(basis_row); physical_sensitivity.append(block[axis])
            else:
                direction = np.asarray(direction, float); direction /= np.linalg.norm(direction)
                basis_row = np.zeros(2*len(items)); basis_row[2*k:2*k+2] = direction
                physical_rows.append(basis_row); physical_sensitivity.append(direction @ block)
        operator = latent_whitening_operator(ids, np.asarray(physical_sensitivity), group_id=group_id,
            covariance_origin="saved_current_native_material_original_readout_noise_factor",
            explicit_noise_factor=noise_factor, physical_basis=np.asarray(physical_rows))
        propagated_factor = sensitivity @ noise_factor
        for k, pid in enumerate(ids):
            scale = float(np.linalg.norm(propagated_factor[2*k:2*k+2], ord="fro"))
            _positive([scale], 1, "propagated native support " + pid)
            scales[pid] = scale
        group = dict(operator=operator.to_record(), point_indices=[locations[p] for p in ids],
                     source_evidence=evidence)
        changed = {pid:material_normals[pid] for pid in ids if pid in material_normals}
        if changed:
            group["physical_material_normal_correction"] = dict(source_points=changed,
                source_estimator_B_and_noise_factor_L_preserved=True,
                only_physical_scalar_residual_basis_Q_changed=True,
                source_estimator_template_conditional_support_not_an_unbiased_material_estimator=True,
                original_source_active_bounds_and_prior_bias_not_absorbed_into_sigma=True,
                source_coordinate_bias_remains_in_model_residual=True,
                newly_independent_readout_information_added=False)
        groups.append(group)
        grouped.update(ids)

    # Each advertisement's corners and source sides use the same latent noise.
    # Separate visible segments retain separate native estimators unless their
    # saved native readout digest proves they share a source estimator.
    board_groups = defaultdict(list)
    for item in primitive:
        row = item[0]; d = row["model_target"]
        if d["kind"] in ("ad_edge", "ad_corner", "board_edge"):
            key = "Ad"+str(d["advertisement_id"]) if d["kind"].startswith("ad_") else row["source_feature_id"]
            board_groups[key].append(item)
    for key, items in board_groups.items():
        features = {}
        for row, j, pid in items:
            fid = row["source_feature_id"]
            if fid not in board_source:
                raise MissingSourceEvidence("Accepted board feature has no saved source readouts: " + fid)
            features[fid] = board_source[fid]
            if row["model_target"]["kind"] == "ad_corner":
                side, level = row["model_target"]["role"].split("_")
                for fid2 in (key+"."+level+"_edge", key+"."+side+"_vertical"):
                    if fid2 not in board_source:
                        raise MissingSourceEvidence("Accepted corner source side is missing: " + fid2)
                    features[fid2] = board_source[fid2]
        charts = {}
        for fid, feature in features.items():
            for segment in feature["segments"]:
                chart = _board_curve(fid, segment, packet["board_source_readout_packet"])
                charts.setdefault(chart.key, chart)
        noise_factor, offsets = _block_noise_factor(list(charts.values())); dimension = noise_factor.shape[0]
        source_lines = {}; sensitivity = []; normal_directions = []
        for row, j, pid in items:
            d = row["model_target"]; uv = row["uv"][j]
            if d["kind"] == "ad_corner":
                side, level = d["role"].split("_")
                line_ids = (key+"."+level+"_edge", key+"."+side+"_vertical")
                values = []
                for fid, axis in zip(line_ids, (0, 1), strict=True):
                    if fid not in source_lines:
                        source_lines[fid] = _feature_line(features[fid], charts, offsets, dimension, axis)
                    values.append(source_lines[fid])
                (horizontal, dh), (vertical, dv) = values
                implicit = np.array([[-horizontal[0], 1.], [1., -vertical[0]]])
                rhs = np.vstack((uv[0]*dh[0]+dh[1], uv[1]*dv[0]+dv[1]))
                block = np.linalg.solve(implicit, rhs)
                normal_directions.append(None)
            else:
                fid = row["source_feature_id"]; segment = row["source_segment"]
                segment = repaired_segments.get(_curve_key(fid, segment), segment)
                chart_key = _board_curve(fid, segment, packet["board_source_readout_packet"]).key
                if chart_key not in charts:
                    # A page deletion may split/reorder segments. The native
                    # support survives in the copied source segment, keyed by
                    # its material readouts rather than its mutable index.
                    chart = _board_curve(fid, segment, packet["board_source_readout_packet"])
                    raise MissingSourceEvidence("Accepted board native support differs from frozen source: " + chart.key)
                curve = charts[chart_key]; start = offsets[curve.key]
                block = np.zeros((2, dimension)); block[:, start:start+len(curve.covariance)] = curve.sensitivity(uv)
                normal_directions.append(curve.normal(uv))
            sensitivity.append(block)
        evidence = [c.metadata for c in charts.values()]
        evidence.extend(dict(source_feature_id=fid, **{k:v for k,v in feature.items() if k not in ("id", "segments")})
                        for fid, feature in features.items() if feature.get("uncertainty_support_only"))
        if repaired_segments:
            evidence.append(dict(method="fresh_current_native_profile_IQR_restores_historical_untrimmed_width_inventory",
                source_image=source_frame["source_image"], source_coordinates_modified=False,
                repaired_native_readout_supports=[dict(original_digest=k,
                    current_native_readout_uv=s["source_native_readout_uv"],
                    current_native_transition_iqr_px=s["source_native_transition_iqr_px"])
                    for k,s in repaired_segments.items()]))
        install(items, np.vstack(sensitivity), noise_factor, f"{pts}/{key}/native_material_and_corners",
                evidence, normal_directions)

    lines = defaultdict(list); circles = defaultdict(list)
    for item in primitive:
        row = item[0]; feature = row["source_feature"]
        if feature.get("source_line_fit"):
            lines[row["source_feature_id"]].append(item)
        elif row["source_segment"].get("representative_points_share_circle_family_estimate"):
            navigation = feature.get("joint_group_navigation", {})
            house = navigation.get("house_id")
            if not house:
                raise MissingSourceEvidence("A source circle family has no declared shared house")
            circles[house].append(item)
    for fid, items in lines.items():
        feature = ice_source.get(fid)
        if feature is None:
            raise MissingSourceEvidence("Accepted ice line actual native readouts missing: " + fid)
        z = feature["source_line_fit"]; rows = z["direct_readouts"]
        p = np.asarray([r["uv"] for r in rows], float); sigma = _positive([r["sigma_px"] for r in rows], len(rows), fid)
        origin = np.asarray(z["point_origin"], float); tangent = np.asarray(z["unit_tangent"], float)
        normal = np.asarray(z["unit_normal"], float); coefficient = np.asarray(z["normal_offset_coefficients"], float)
        station = (p-origin) @ tangent; design = np.column_stack((station, np.ones(len(station))))
        broad = fid.split(".")[-1].split(":")[0] in ("B1", "B2", "C")
        signal = np.asarray([min(r["source_evidence"]["signed_interface_iqr_contrast"]) for r in rows])
        weights = signal**2 if broad else np.ones(len(rows))
        influence = _influence(design, weights); noise_factor = influence*sigma
        actual_normal = normal-coefficient[0]*tangent; actual_normal /= np.linalg.norm(actual_normal)
        projection = 1.
        contract = z.get("estimator_contract")
        if contract is not None:
            expected_policy = "minimum_two_interface_contrast_squared" if broad else "uniform_current_native_line_support"
            if (contract.get("schema") != FINAL_LINE_SCHEMA
                    or contract.get("kind") != "weighted_linear_final_native_support"
                    or contract.get("weights_policy") != expected_policy
                    or contract.get("final_support_refit_unconstrained") is not True
                    or contract.get("family_direction_used_for_candidate_disambiguation_only") is not True
                    or contract.get("source_sigma_coordinates") != "recorded_navigation_normal_coverage_px"
                    or contract.get("derivative_normal_projection_factor_required") is not True
                    or not np.allclose(coefficient, influence@((p-origin)@normal), rtol=2e-12, atol=1e-12)):
                raise ValueError("Opt-in source line mean and D do not certify the same final native support")
            projection = float(actual_normal@normal)
        sensitivity = []
        for row, j, pid in items:
            t = float((row["uv"][j]-origin) @ tangent)
            sensitivity.append(projection*np.outer(actual_normal, [t, 1.]))
        install(items, np.vstack(sensitivity), noise_factor, f"{pts}/{fid}/source_line", dict(
            source_readout_packet=packet["ice_source_readout_packet"], source_feature_id=fid,
            native_readout_count=len(rows), source_sigma_origin="direct_readouts.sigma_px",
            covariance_method="saved_material_normal_line_fixed_readout_influence",
            conditional_on_recorded_material_branch_and_family_direction=True,
            source_estimator_contract=contract,
            physical_sensitivity_normal_projection_factor=projection), [actual_normal]*len(items))
    for house, items in circles.items():
        features = {fid: f for fid, f in ice_source.items() if f.get("source_circle_fit")
                    and f.get("joint_group_navigation", {}).get("house_id") == house}
        raw_rows = [(f, r) for f in features.values() for r in f["source_circle_fit"]["direct_readouts"]]
        rows = [r for f, r in raw_rows]; sigma = _positive([r["sigma_px"] for r in rows], len(rows), house)
        short = all(f["joint_group_navigation"].get("fixed_shape_from_confirmed_model_and_source_continuity") for f in features.values())
        normals = np.asarray([r["source_evidence"]["profile_normal"] for r in rows], float)
        if short:
            _, singular, axes = np.linalg.svd(normals, full_matrices=False)
            rank = next(iter(features.values()))["joint_group_navigation"]["observable_translation_rank"]
            translation = axes[:rank].T
            design = normals @ translation
            output_basis = lambda f, p: np.outer(_ellipse_normal(f, [p])[0], _ellipse_normal(f, [p])[0]) @ translation
        else:
            design = np.stack([n @ _affine_basis(r["representative_uv"]) for n, r in zip(normals, rows, strict=True)])
            output_basis = lambda f, p: np.outer(_ellipse_normal(f, [p])[0], _ellipse_normal(f, [p])[0]) @ _affine_basis(p)
        residual = np.asarray([float((np.asarray(r["uv"])-r["representative_uv"]) @ n) for r, n in zip(rows, normals, strict=True)])
        weights = (1+(residual/1.5)**2)**(-1.5)
        influence = _influence(design, weights); noise_factor = influence*sigma
        sensitivity = np.vstack([output_basis(features[row["source_feature_id"]], row["uv"][j]) for row, j, pid in items])
        install(items, sensitivity, noise_factor, f"{pts}/{house}/shared_native_circle_family", dict(
            source_readout_packet=packet["ice_source_readout_packet"], house_id=house,
            native_readout_count=len(rows), source_sigma_origin="source_circle_fit.direct_readouts.sigma_px",
            covariance_method="shared_source_normal_affine_localization_support" if not short else "observable_source_normal_translation_support",
            shared_rings=list(features), source_only_loss_f_scale=1.5,
            fixed_navigation_directions_do_not_add_source_noise=True,
            local_chart_is_not_a_refit_of_the_accepted_family=True),
            [_ellipse_normal(features[row["source_feature_id"]], [row["uv"][j]])[0] for row, j, pid in items])
    for row, j, pid in primitive:
        if pid in grouped:
            continue
        if row["source_feature"].get("manual_source_observation"):
            scales[pid], evidence = _manual_support(row, j, packet, ice_source, source_frame)
            uncertainty.append(evidence)
        _positive([scales[pid]], 1, "Accepted source point " + pid)
    if len(scales) != len(primitive):
        raise ValueError("Every accepted native point requires one source scale")
    if preserved_source_scales is not None:
        if set(preserved_source_scales) != set(scales):
            raise ValueError("Preserved source scales differ from the unchanged accepted membership")
        for pid, value in preserved_source_scales.items():
            _positive([value], 1, "Preserved accepted source scale " + pid)
        scales = dict(preserved_source_scales)
    return dict(source_scales_by_point_id=scales, whitening_groups=groups,
        uncertainty_evidence=uncertainty, point_count=len(primitive),
        correlated_point_count=len(grouped), ordinary_point_count=len(primitive)-len(grouped),
        source_UV_modified=False, supplementary_readouts_added_as_FIT_points=0,
        previous_source_scale_map_preserved=preserved_source_scales is not None,
        source_scales_are_localization_support_not_calibrated_noise=True)


def prepare_model_association_frame_evidence(observations, source_frame, overrides_reference, *, preserved_frame):
    """Change only Q for confirmed identities; keep source B/L and all scales.

    Source representatives remain the original conditional source-estimator
    outputs, including their template/bound bias. A changed model material
    uses its own same-exposure native line normal for the residual direction;
    these line readouts are not added as new independent noise information.
    """
    from copy import deepcopy
    fix = read_reference(overrides_reference)
    if (fix.get("schema") != "rinkspace.accepted-source-point-identity-overrides.v1"
            or fix.get("original_source_packets_bindings_UV_IDs_and_approvals_modified") is not False):
        raise ValueError("An unchanged authenticated accepted source-identity bridge is required")
    packet = read_reference(source_frame["packet"])
    native = read_reference(packet["ice_source_readout_packet"])
    registry = read_reference(fix["feature_registry"])["targets"]
    if (native["source"]["native_pts"] != source_frame["native_pts"]
            or packet["source"]["native_pts"] != source_frame["native_pts"]
            or fix["source_video_sha256"] != packet["source"]["source_video_sha256"]):
        raise ValueError("Material-normal evidence belongs to another source exposure")
    features = {f["id"]:f for f in native["features"]}
    items = [item for item in fix["overrides"] if item["native_pts"] == source_frame["native_pts"]]
    if not items or source_frame["index"] <= 200:
        raise ValueError("Only explicitly confirmed appended-frame identities may change physical normals")
    locations = {pid:(row,j) for row in observations for j,pid in enumerate(row["source_point_ids"])}
    corrections = {}
    for item in items:
        pid = item["source_point_id"]
        if pid in corrections or pid not in locations:
            raise ValueError("Duplicate or missing accepted material-normal point")
        row,j = locations[pid]; target = item["model_target_descriptor"]
        if (item.get("identity_confirmed_by_user") is not True
                or item["source_packet"] != source_frame["packet"]
                or item["time_base"] != fix["source_time_base"]
                or target != registry.get(item["model_target_id"])
                or target.get("kind") != "ice_curve"
                or item["original_source_feature_id"] != row["source_feature_id"]
                or item["original_source_segment_index"] != row["source_segment_index"]
                or item["original_source_vertex_index"] != row["original_vertex_indices"][j]
                or item["original_source_model_target_id"] != row["target_id"]
                or not np.array_equal(np.asarray(item["approved_source_UV"],float),row["uv"][j])):
            raise ValueError("A confirmed material normal changed the accepted source address or UV")
        fid = "ice."+target["curve_id"]; feature = features.get(fid)
        if feature is None or not feature.get("source_line_fit"):
            raise MissingSourceEvidence("The confirmed material needs its own current-native line normal: " + fid)
        z = feature["source_line_fit"]; readouts = z.get("direct_readouts", [])
        if not readouts or not z.get("current_frame_only"):
            raise MissingSourceEvidence("The material normal lacks current-native measured line support: " + fid)
        _positive([r["sigma_px"] for r in readouts],len(readouts),fid)
        n = np.asarray(z["unit_normal"],float)-z["normal_offset_coefficients"][0]*np.asarray(z["unit_tangent"],float)
        if n.shape != (2,) or not np.isfinite(n).all() or np.linalg.norm(n) == 0.:
            raise MissingSourceEvidence("The material line has no finite current-native normal: " + fid)
        n /= np.linalg.norm(n)
        source_n = _ellipse_normal(features[row["source_feature_id"]],[row["uv"][j]])[0]
        corrections[pid] = dict(normal=n.tolist(), model_target_id=item["model_target_id"],
            unchanged_source_UV=row["uv"][j].tolist(), original_source_output_normal=source_n.tolist(),
            source_output_to_material_normal_cosine=float(source_n@n),
            model_association_overrides=overrides_reference, source_native_packet=packet["ice_source_readout_packet"],
            physical_normal_source_feature_id=fid, physical_normal_source_field="source_line_fit",
            current_native_readout_count=len(readouts),
            current_native_source_line_metadata={k:v for k,v in z.items() if k != "direct_readouts"},
            current_native_normal_not_taken_from_fitted_world_model=True,
            normal_source_readouts_added_as_independent_information=False)
    rebuilt = prepare_frame_evidence(observations, source_frame,
        preserved_source_scales=preserved_frame["source_scales_by_point_id"],
        physical_material_normals_by_point_id=corrections)
    result = deepcopy(preserved_frame); used = set(); changed_groups = []
    by_ids = {tuple(g["operator"]["source_point_ids"]):g for g in rebuilt["whitening_groups"]}
    for i,old in enumerate(result["whitening_groups"]):
        ids = tuple(old["operator"]["source_point_ids"]); affected = set(ids) & set(corrections)
        if not affected:continue
        new = by_ids[ids]
        if old["point_indices"] != new["point_indices"]:
            raise ValueError("Corrected physical normals reordered original source membership")
        result["whitening_groups"][i] = new; used.update(affected)
        old_operator = SourceWhiteningOperator.from_record(old["operator"])
        new_operator = SourceWhiteningOperator.from_record(new["operator"])
        changed_groups.append(dict(group_id=new["operator"]["group_id"],
            source_point_ids=list(ids), corrected_source_point_ids=sorted(affected),
            original_rank=old_operator.rank, corrected_rank=new_operator.rank,
            original_maximum_absolute_W=float(abs(old_operator.matrix).max()),
            corrected_maximum_absolute_W=float(abs(new_operator.matrix).max()),
            original_point_indices=old["point_indices"], source_output_B_and_native_noise_L_preserved=True))
    if used != set(corrections):
        raise ValueError("A corrected material point has no original correlated source estimator")
    result["corrected_evidence"] = dict(model_association_overrides=overrides_reference,
        source_point_ids=list(corrections), source_estimator_B_and_noise_factor_L_preserved=True,
        physical_material_residual_basis_only_changed=True,
        affected_groups=changed_groups, physical_material_normals_by_point_id=corrections,
        source_scales_and_UV_IDs_unchanged=True, other_groups_reused_without_modification=True,
        original_estimator_template_priors_active_bounds_remain_conditional_support_limitations=True,
        source_estimator_bias_not_removed_by_source_noise_whitening=True)
    return result
