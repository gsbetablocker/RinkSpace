"""A bounded full-state trial conditioned on the clear source contour margin."""
import numpy as np
from scipy.sparse import csr_matrix, vstack


def contour_margins(value, contour, limits, contour_weight):
    """Linearize the same outside-band p95 checked by the nonlinear guard."""
    if contour is None:
        return None
    rows = value['action_metrics'].get('visible_runner_contours', [])
    terms = next(b for b in value['weighted_blocks'] if b.name == 'full_source_skate_structure')
    if len(terms.residual) != 2*len(contour.point_rows) or len(rows) != len(contour.rows):
        raise ValueError('Source protection requires exact assigned contour row order')
    gradient, slack, labels = [], [], []
    for i, row in enumerate(rows):
        if row['id'] != contour.rows[i]['id']:
            raise ValueError('Clear contour and its actual Jacobian identity differ')
        old, radius = limits[row['id']]
        distances = np.asarray(row['outside_visible_band_px'])
        current = float(np.percentile(distances, 95))
        margin = old + .25*radius - current
        if margin > .12*radius or current <= 0.:
            continue
        ordered = np.argsort(distances, kind='stable')
        coordinate = .95*(len(ordered)-1)
        lo, hi = int(np.floor(coordinate)), int(np.ceil(coordinate))
        fraction = coordinate - lo
        source_points = np.flatnonzero(contour.point_rows == i)
        row_jac = csr_matrix((1, terms.jacobian.shape[1]))
        for q, weight in ((lo, 1.-fraction), (hi, fraction)):
            index = ordered[q]
            outside = distances[index]
            if outside <= 0. or weight == 0.:
                continue
            k = source_points[index]
            pair = slice(2*k, 2*k+2)
            norm = np.linalg.norm(terms.residual[pair])
            if norm == 0.:
                continue
            # region_terms uses a circular band and pseudo-Huber delta 2.
            source = contour.factor_rows[k]
            w = np.sqrt(contour_weight*source['objective_weight']*source['reliability'])
            u = np.hypot(1., outside/(2*radius))
            derivative = w*np.sqrt(2./(u+1.))*(u+1.)/(2*u*radius)
            radial = terms.residual[pair] / (norm*derivative)
            row_jac += csr_matrix(radial[None]) @ terms.jacobian[pair] * weight
        if row_jac.nnz:
            gradient.append(row_jac)
            # Leave a fraction of the observed band for nonlinear curvature.
            # The original guard and its exact original limit remain in use.
            slack.append(margin-.04*radius)
            labels.append(row['id'])
    return None if not gradient else (vstack(gradient,format='csr'), np.asarray(slack), labels)


def source_conditioned_direction(gpu, value, bounds, margins, selection):
    """Project the damped joint step onto sparse image-space half spaces."""
    scale = 1./np.maximum(gpu.scale_inv, 1e-6)
    step, linear = gpu.solve_scaled(-value['residual'], scale, damp=.07)
    step /= max(1., np.linalg.norm(step))
    A, slack, labels = margins
    A = (A[:,selection].multiply(scale)).toarray()
    norms = np.einsum('ij,ij->i', A, A)
    initial = (A @ step).tolist()
    for _ in range(8):
        for a, norm, limit in zip(A, norms, slack):
            excess = a @ step - limit
            if excess > 0. and norm > 0.:
                step -= (excess/norm)*a
    step = np.clip(step*scale, *bounds)
    return step, dict(constraint_rows=labels, initial_linear_change_px=initial,
        adjusted_linear_change_px=(A @ (step/scale)).tolist(),
        inner_margin_px=slack.tolist(), linear_solve=linear,
        original_nonlinear_guard_required=True, all_active_controls_retained=True)
