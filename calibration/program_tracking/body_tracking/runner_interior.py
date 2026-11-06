"""Exact longitudinal rocker coordinate with finite endpoint restoration."""
import numpy as np
from .skate_template import _profile


def runner_interior_band(parameters, config=None):
    """Geometry-relative finite prior band; no calibrated contact threshold."""
    config = {} if config is None else config
    _, _, (rear, _, join, front) = _profile(parameters)
    if 'u_min' in config or 'u_max' in config:
        low, high = float(config['u_min']), float(config['u_max'])
    else:
        fraction = float(config.get('buffer_fraction_of_shorter_rocker', .5))
        if not 0 < fraction < 1:
            raise ValueError('Interior rocker buffer fraction must be in (0,1)')
        margin = fraction*min(join-rear, front-join)/(front-rear)
        low, high = margin, 1-margin
    if not 0 < low < high < 1:
        raise ValueError('Finite interior interval must lie strictly inside the runner')
    return np.array([low, high]), min(low, 1-high)


def runner_longitudinal_coordinate(normal, parameters, witness_x=None):
    """Actual clipped u plus C1 tangent continuation at a steel endpoint.

    The clipped minimizer has zero derivative outside its domain. The tangent
    continuation of its stationary coordinate restores descent there, without
    inventing a material/contact point beyond the steel. Inside, both coincide.
    """
    normal = np.asarray(normal, float)
    main, front, (x0, _, _, x1) = _profile(parameters)
    nx, nz = normal[:, 0], normal[:, 2]
    denominator = nx*nx+nz*nz
    if np.any(denominator < 1e-14) or np.any(nz <= 0):
        raise ValueError('Supported runner stationary coordinate is undefined or inverted')
    theta = np.arctan2(-nx, nz)
    low, join, high = main['theta_start_rad'], main['theta_end_rad'], front['theta_end_rad']
    clipped = np.clip(theta, low, high)
    primary = clipped <= join
    radius = np.where(primary, main['radius_m'], front['radius_m'])
    cx = np.where(primary, main['center'][0], front['center'][0])
    exact_x = cx+radius*np.sin(clipped)
    inside = (theta > low) & (theta < high)
    dx_exact = radius*np.cos(clipped)*inside
    slope = np.where(theta < low, main['radius_m']*np.cos(low), front['radius_m']*np.cos(high))
    extension = slope*(theta-clipped)/(x1-x0)
    if witness_x is not None:
        witness_x = np.asarray(witness_x, float)
        if not np.allclose(witness_x, exact_x, rtol=0, atol=(x1-x0)*2e-10):
            raise ValueError('Analytic u differs from exact curved-runner witness')
        exact_x = witness_x
    actual = (exact_x-x0)/(x1-x0)
    dtheta = np.c_[-nz/denominator, np.zeros_like(nx), nx/denominator]
    actual_derivative = dx_exact[:, None]/(x1-x0)*dtheta
    effective_derivative = (dx_exact+slope*(~inside))[:, None]/(x1-x0)*dtheta
    return dict(actual_u=actual, effective_u=actual+extension, endpoint_extension_u=extension,
                actual_derivative_normal=actual_derivative, derivative_normal=effective_derivative,
                stationary_theta_rad=theta, clipped_at_endpoint=~inside)


def longitudinal_barrier(coordinate, band, scale, weights):
    u = coordinate['effective_u']; low, high = band
    excess = np.minimum(u-low, 0.)+np.maximum(u-high, 0.)
    w = np.sqrt(np.asarray(weights, float))/scale
    active = (u < low) | (u > high)
    return w*excess, (w*active)[:, None]*coordinate['derivative_normal']
