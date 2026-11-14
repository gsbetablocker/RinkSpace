"""Select existing query samples by photographed navigation shape, never by UV error."""
from __future__ import annotations

import cv2
import numpy as np

from rinkspace.session.print_shape_queries import KEY_SHAPE_POLICY


def select_query_shape_vertices(points, *, closed):
    """Return original indices retained by the public shape approximation policy.

    The caller supplies finite navigation coordinates in authenticated source arc
    order and an explicit canonical closure flag. Closure only controls the
    simplification of these discrete samples; it makes no visibility claim.
    Returned indices are in original order. No measured UV, fit residual, new
    query, qualification decision, weight or point limit is used or created.
    """
    points = np.asarray(points, np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) == 0
            or not np.isfinite(points).all()):
        raise ValueError('Finite nonempty ordered query coordinates required')
    if type(closed) is not bool:
        raise ValueError('Explicit canonical closure flag required')
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    # Same epsilon and float32 approxPolyDP input as key_shape_stations.
    epsilon = max(KEY_SHAPE_POLICY['approximation_error_px'], diagonal * .008)
    source = points.astype(np.float32)
    if not np.isfinite(source).all():
        raise ValueError('Query coordinates exceed float32 approximation domain')
    vertices = cv2.approxPolyDP(source.reshape(-1, 1, 2), epsilon, closed).reshape(-1, 2)
    indices = []
    for vertex in vertices:
        matches = np.flatnonzero(np.all(source == vertex, axis=1))
        if len(matches) != 1:
            raise ValueError('Approximation vertex has no unique original query sample')
        indices.append(int(matches[0]))
    indices = np.asarray(sorted(set(indices)), np.int64)
    if not len(indices):
        raise ValueError('Approximation returned no original sample')
    return indices, dict(
        schema='rinkspace.saved_query_shape_selection.v1',
        policy=dict(KEY_SHAPE_POLICY), approximation_error_px=epsilon,
        shape_diagonal_px=diagonal, relative_diagonal_error=.008,
        closed=closed, input_samples=len(points), selected_samples=len(indices),
        original_query_coordinates_only=True, source_arc_order_supplied_by_caller=True,
        selected_original_indices=indices.tolist(),
        actual_UV_or_residual_used=False, new_points_created=False,
        qualification_granted=False, weights_changed=False,
    )
