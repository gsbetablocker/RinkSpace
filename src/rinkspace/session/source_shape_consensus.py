"""Organize RGB profile alternatives with the known projected shape intact.

One image transform is shared by every path of an object.
The linearized contour equation supplies search/shape evidence only; returned
observations are unchanged source candidate IDs, never predicted coordinates.
"""
from __future__ import annotations

import numpy as np
import cv2
from .material_rings import ellipse_matrix


def known_shape_profile_hypotheses(measurement, curves, fragments=()):
    """Compare partial arcs without fitting an unconstrained ellipse to them.

    Native query curves set shape and relative geometry. Actual q25..q75
    intervals decide final support; ambiguous source reads remain alternatives.
    This neither supplies physical identity nor qualifies data for fitting.
    """
    curves = np.asarray(curves, int)
    qcurve = np.asarray(measurement['query_curve'], int)
    query = np.asarray(measurement['candidate_query'], int)
    quv = np.asarray(measurement['query_uv'], float)
    qnormal = np.asarray(measurement['query_normal'], float)
    uv = np.asarray(measurement['candidate_uv'], float)
    quantiles = np.asarray(measurement['candidate_side_quantiles_uv'], float).mean(axis=1)
    clean = (np.asarray(measurement['candidate_flags']) & 3) == 0
    use = np.flatnonzero(np.isin(qcurve, curves))
    if len(use) < 5:
        return []
    origin = quv[use].mean(0)
    extent = max(float(np.linalg.norm(np.ptp(quv[use], axis=0))), 1.)
    design = np.full((len(quv), 6), np.nan)
    guide_conics = {}
    for curve in curves:
        qi = np.flatnonzero(qcurve == curve)
        if len(qi) < 5:
            continue
        # Fit only the known projected guide. No observed point determines its
        # aspect ratio or turns a straight reflection into a free tiny ellipse.
        try:
            center, matrix = ellipse_matrix(quv[qi])
        except (ValueError, cv2.error):
            continue
        guide_conics[int(curve)] = (center, matrix)
        gradient = (quv[qi]-center)@matrix
        gradient /= np.maximum(np.linalg.norm(gradient, axis=1, keepdims=True), 1e-12)
        denominator = np.sum(gradient*qnormal[qi], axis=1)
        good = np.abs(denominator) > 1e-6
        qi, gradient, denominator = qi[good], gradient[good], denominator[good]
        relative = (quv[qi]-origin)/extent
        design[qi] = np.c_[gradient, gradient[:, 0, None]*relative,
                           gradient[:, 1, None]*relative]/denominator[:, None]
    ids = np.flatnonzero(clean & np.isin(qcurve[query], curves) & np.isfinite(design[query]).all(1))
    if len(ids) < 5:
        return []
    local_query = query[ids]
    measured = np.sum((uv[ids]-quv[local_query])*qnormal[local_query], axis=1)
    ends = np.sum((quantiles[ids]-quv[local_query, None])*qnormal[local_query, None], axis=2)
    lower, upper = ends.min(1), ends.max(1)
    x = design[local_query]
    if np.linalg.matrix_rank(x[:, :2]) < 2:
        return []
    groups = [np.flatnonzero(local_query == qi) for qi in np.unique(local_query)]
    lookup = {int(cid): i for i, cid in enumerate(ids)}
    translation = np.eye(6)[:, :2]
    similarity = np.zeros((6, 4)); similarity[:, :2] = translation
    similarity[[2, 5], 2] = 1.; similarity[3, 3] = -1.; similarity[4, 3] = 1.
    modes = [('translation', translation), ('similarity', similarity)]
    if len(np.unique(qcurve[local_query])) >= 2:
        # The image aspect may be wrong before the camera is refined. Multiple
        # nested interfaces can support ONE common aspect/shear correction;
        # fitting a separate ellipse per layer would discard the known shape.
        symmetric = np.zeros((6, 5)); symmetric[:, :2] = translation
        symmetric[2, 2] = 1.; symmetric[5, 3] = 1.; symmetric[[3, 4], 4] = 1.
        modes.append(('shared_symmetric_affine', symmetric))
    starts = []
    # A cropped arc often cannot constrain scale/rotation independently. Keep
    # the translation-only explanation as an explicit alternative, preserving
    # those known shape dimensions instead of inventing an extreme ellipse.
    for mode, basis in modes:
        local_x = x@basis; dimensions = basis.shape[1]
        if np.linalg.matrix_rank(local_x) < dimensions:
            continue
        starts.append((mode, basis, np.zeros(6)))
        for fragment in fragments:
            if fragment['curve'] not in curves:
                continue
            rows = [lookup[i] for i in fragment['candidate_ids'] if i in lookup]
            if len(rows) >= 5 and np.linalg.matrix_rank(local_x[rows]) == dimensions:
                parameters = basis@np.linalg.lstsq(local_x[rows], measured[rows], rcond=1e-8)[0]
                starts.append((mode, basis, parameters))
        # Every measured displacement mode is tried, without random trials.
        for offset in np.unique(np.rint(measured/4.)*4.):
            parameters = basis@np.linalg.lstsq(local_x, np.full(len(x), offset), rcond=1e-8)[0]
            starts.append((mode, basis, parameters))
    alternatives = {}
    for mode, basis, parameters in starts:
        local_x = x@basis; dimensions = basis.shape[1]
        seen = set()
        while True:
            predicted = x@parameters
            error = np.abs(measured-predicted)
            chosen = np.array([g[np.argmin(error[g])] for g in groups], int)
            chosen = chosen[error[chosen] <= 4.]
            key = tuple(chosen)
            if len(chosen) < 5 or key in seen or np.linalg.matrix_rank(local_x[chosen]) < dimensions:
                break
            seen.add(key)
            parameters = basis@np.linalg.lstsq(local_x[chosen], measured[chosen], rcond=1e-8)[0]
            # A single arc retains its aspect. Multiple measured layers may
            # change aspect together, preserving their relative geometry.
            if not np.isfinite(parameters).all():
                break
            linear = np.eye(2)+parameters[2:].reshape(2, 2)/extent
            scale = np.linalg.svd(linear, compute_uv=False)
            if np.linalg.det(linear) <= 0. or scale.min() <= 1e-8*max(1., scale.max()):
                break
            predicted = x@parameters
            supported = (predicted >= lower) & (predicted <= upper)
            # The linearization may organize candidates but is not evidence
            # of exact shape support. Check the same transform on the actual
            # finite q25..q75 material intervals before returning any IDs.
            inverse = np.linalg.inv(linear)
            for curve, (center, matrix) in guide_conics.items():
                rows = np.flatnonzero(qcurve[local_query] == curve)
                transformed_center = (center-origin)@linear.T+origin+parameters[:2]
                transformed_matrix = inverse.T@matrix@inverse
                endpoints = quantiles[ids[rows]][:, [0, 2]]-transformed_center
                value = np.einsum('nqi,ij,nqj->nq', endpoints, transformed_matrix, endpoints)-1.
                supported[rows] &= value[:, 0]*value[:, 1] <= 0.
            unique = np.zeros(len(ids), bool)
            for group in groups:
                good = group[supported[group]]
                if len(good) == 1:
                    unique[good[0]] = True
            selected = np.flatnonzero(unique)
            if len(selected) < 5:
                continue
            if mode == 'shared_symmetric_affine':
                _, layer_counts = np.unique(qcurve[local_query[selected]], return_counts=True)
                if (layer_counts >= 5).sum() < 2:
                    continue  # One short arc cannot relax the known aspect.
            # Do not claim an object from a handful of coincident profile tips.
            source_span = float(np.linalg.norm(np.ptp(uv[ids[selected]], axis=0)))
            if source_span < 56.:
                continue
            selected_ids = ids[selected].tolist()
            item = dict(candidate_ids=selected_ids, supported_source_queries=len(selected),
                supported_source_span_px=source_span, parameters=parameters.tolist(),
                origin_uv=origin.tolist(), parameter_extent=extent,
                transform_model=mode,
                known_projected_shape_preserved_to_first_order=True,
                exact_shared_conic_inside_original_transition=True,
                source_transition_intervals_used=True, source_UV_unchanged=True,
                fit_eligible=False, physical_identity_confirmed=False,
                score=float(len(selected)))
            old = alternatives.get(tuple(selected_ids))
            if old is None or item['score'] > old['score']:
                alternatives[tuple(selected_ids)] = item
    return sorted(alternatives.values(), key=lambda item: item['score'], reverse=True)
