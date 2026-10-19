"""Complete local tangent of the existing continuous mount guard certificate.

The guard keeps the quadratic driver's z/z**2 correlation, but bounds its rho
and driver independently within each native interval. This module preserves
that exact certificate and its numeric box; it does not claim a tight bound on
the fully correlated time trajectory. It constructs no physical state or QP.

Outward-rounding values supplied by the original support/quadratic callback
are retained in every current margin. Their machine-ULP padding is frozen in
the derivative. Exact finite extrema ties retain every derivative branch.
"""
import itertools

import numpy as np
from scipy.sparse import csr_matrix, vstack


def guard_mount_tangent(*, value, value_jacobian, lower, upper,
                        log_scale_gradient, support, tilt, tilt_jacobian,
                        driver, driver_jacobian, interval_limits,
                        vector_a, vector_b, vector_a_jacobian,
                        vector_b_jacobian, quadratic_interval_range):
    """Return all current guard margins and complete common-column tangents.

    value is [internal_d_xyz, axial_linear, signed_axis_a, axial_quadratic].
    lower/upper are the existing box in the same working chart as value.
    quadratic_interval_range is the original guard's quadratic_range callback.
    Completely flat polynomial/support extrema with free derivatives have a
    continuum of supporting planes. Fail explicitly rather than silently call
    a finite arbitrary branch set a complete tangent of that degenerate case.
    """
    value = np.asarray(value, dtype=float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    tilt, driver = np.asarray(tilt, float), np.asarray(driver, float)
    limits = np.asarray(interval_limits, float)
    av, bv = np.asarray(vector_a, float), np.asarray(vector_b, float)
    VJ, TJ, DJ = map(csr_matrix, (value_jacobian, tilt_jacobian, driver_jacobian))
    AJ, BJ = map(csr_matrix, (vector_a_jacobian, vector_b_jacobian))
    ls = csr_matrix(np.asarray(log_scale_gradient, float)[None, :])
    n = VJ.shape[1]
    if not (value.shape == (6,) and lower.shape == upper.shape == (3,)
            and np.all(lower < upper) and av.shape == bv.shape == (3,)
            and VJ.shape == (6, n) and TJ.shape == DJ.shape == (len(tilt), n)
            and len(driver) == len(tilt) and limits.shape == (len(tilt)-1, 2)
            and AJ.shape == BJ.shape == (3, n) and ls.shape == (1, n)
            and len(support['intervals']) == len(tilt)-1):
        raise ValueError('Complete original guard arrays and common-column derivatives required')
    for array in (value, lower, upper, tilt, driver, limits, av, bv, ls.data,
                  VJ.data, TJ.data, DJ.data, AJ.data, BJ.data):
        if not np.isfinite(array).all():
            raise ValueError('Finite original guard values and derivatives required')

    rows, margins, origins, ties = [], [], [], []
    total_low, total_high = np.full(3, np.inf), np.full(3, -np.inf)

    def rho_gradients(i, axis, extreme):
        left, right = limits[i]
        beta0, beta1 = tilt[i:i+2]; delta = beta1-beta0
        times = [left, right]
        low, high = sorted((beta0+delta*left, beta0+delta*right))
        if av[axis] == bv[axis] == 0. and value[4] != 0.:
            if AJ.getrow(axis).nnz or BJ.getrow(axis).nnz:
                raise ValueError('Flat trigonometric support has a continuum of tangent branches')
        if delta != 0.:
            phase = float(np.arctan2(bv[axis], av[axis]))
            for k in range(int(np.ceil((low-phase)/np.pi)),
                           int(np.floor((high-phase)/np.pi))+1):
                times.append((phase+k*np.pi-beta0)/delta)
        betas = np.asarray([beta0+delta*t for t in times])
        values = av[axis]*np.cos(betas)+bv[axis]*np.sin(betas)
        target = values.min() if extreme == 0 else values.max()
        selected = np.flatnonzero(values == target)
        output = []
        for index in selected:
            t, beta = times[index], betas[index]
            gradient = (-av[axis]*np.sin(beta)+bv[axis]*np.cos(beta))*((1-t)*TJ.getrow(i)+t*TJ.getrow(i+1))
            gradient += np.cos(beta)*AJ.getrow(axis)+np.sin(beta)*BJ.getrow(axis)
            output.append(gradient)
        if len(output) > 1:
            ties.append(dict(kind='rho', interval=i, axis=axis, extreme=extreme, branches=len(output)))
        return output

    def endpoint_driver_gradients(i, extreme):
        endpoints = limits[i]
        z = np.asarray([(1-t)*driver[i]+t*driver[i+1] for t in endpoints])
        target = z.min() if extreme == 0 else z.max()
        selected = np.flatnonzero(z == target)
        result = [(1-endpoints[k])*DJ.getrow(i)+endpoints[k]*DJ.getrow(i+1)
                  for k in selected]
        if len(result) > 1:
            ties.append(dict(kind='driver', interval=i, extreme=extreme, branches=len(result)))
        return result

    def add(axis, coordinate, jacobian, side, origin):
        bound = lower[axis] if side == 1 else upper[axis]
        rows.append(side*(jacobian+bound*ls))
        margins.append(side*(coordinate-bound))
        origins.append(origin)
        if origin['kind'] != 'internal':
            total_low[axis] = min(total_low[axis], coordinate)
            total_high[axis] = max(total_high[axis], coordinate)

    # Mechanical zero-axial reference for every axis. This is also the entire
    # x/y guard. Global rho extrema include every tied interval origin.
    for axis, extreme in itertools.product(range(3), range(2)):
        target = support['v_min' if extreme == 0 else 'v_max'][axis]
        matching = [record['left'] for record in support['intervals']
                    if np.asarray(record['v_bounds'])[axis, extreme] == target]
        if not matching:
            raise ValueError('Original global rho support lacks its interval origins')
        for i in matching:
            coordinate = value[axis]+value[4]*target
            for rj, side in itertools.product(rho_gradients(i, axis, extreme), (1., -1.)):
                gradient = VJ.getrow(axis)+target*VJ.getrow(4)+value[4]*rj
                add(axis, coordinate, gradient, side,
                    dict(kind='mechanical', axis=axis, extreme=extreme, interval=i, side=int(side)))

    # Every native interval uses the original polynomial minimum/maximum,
    # keeping the z/z^2 relation. Only rho remains independently bounded.
    for record in support['intervals']:
        i = int(record['left']); zlo, zhi = map(float, record['driver'])
        if value[3] == value[5] == 0. and zlo != zhi:
            if VJ.getrow(3).nnz or VJ.getrow(5).nnz:
                raise ValueError('Flat axial polynomial has a continuum of tangent branches')
        bounds, _ = quadratic_interval_range(value[3], value[5], zlo, zhi)
        stations = [(zlo, 'low'), (zhi, 'high')]
        if value[5] != 0.:
            vertex = -value[3]/(2*value[5])
            if zlo <= vertex <= zhi:
                stations.append((vertex, 'vertex'))
        pvalues = np.asarray([value[3]*z+value[5]*z*z for z, _ in stations])
        for extreme, side in ((0, 1.), (1, -1.)):
            target = pvalues.min() if extreme == 0 else pvalues.max()
            selected = np.flatnonzero(pvalues == target)
            if len(selected) > 1:
                ties.append(dict(kind='polynomial', interval=i, extreme=extreme, branches=len(selected)))
            for index in selected:
                z, branch = stations[index]
                if branch == 'vertex':
                    # Envelope theorem: p'(z*)=0, including moving s/b.
                    zgrads = [csr_matrix((1, n))]
                    slope = 0.
                else:
                    zgrads = endpoint_driver_gradients(i, 0 if branch == 'low' else 1)
                    slope = value[3]+2*value[5]*z
                for rho_extreme in (0, 1):
                    rho = np.asarray(record['v_bounds'])[2, rho_extreme]
                    coordinate = value[2]+bounds[extreme]+value[4]*rho
                    for rj, zj in itertools.product(rho_gradients(i, 2, rho_extreme), zgrads):
                        gradient = (VJ.getrow(2)+z*VJ.getrow(3)+rho*VJ.getrow(4)
                                    +z*z*VJ.getrow(5)+value[4]*rj+slope*zj)
                        add(2, coordinate, gradient, side,
                            dict(kind='axial', axis=2, interval=i, extreme=extreme,
                                 rho_extreme=rho_extreme, polynomial_branch=branch, side=int(side)))

    for axis, side in itertools.product(range(3), (1., -1.)):
        add(axis, value[axis], VJ.getrow(axis), side,
            dict(kind='internal', axis=axis, side=int(side)))
    matrix = vstack(rows, format='csr'); matrix.eliminate_zeros()
    margins = np.asarray(margins)
    return dict(matrix=matrix, lower=-margins, current_margins=margins,
        row_origins=origins,
        report=dict(schema='rinkspace.existing-continuous-mount-guard-tangent.v1',
            differentiated_rows=len(margins), exact_tie_branches_retained=ties,
            guard_total_min_chart_working_units=total_low.tolist(), guard_total_max_chart_working_units=total_high.tolist(),
            existing_box_lower_chart_working_units=lower.tolist(), existing_box_upper_chart_working_units=upper.tolist(),
            original_numeric_box_and_nonlinear_guard_unchanged=True,
            z_and_z_squared_polynomial_correlation_preserved=True,
            rho_and_driver_time_correlation_still_interval_separated=True,
            original_outward_rounding_preserved_in_current_margins=True,
            derivative_of_machine_ulp_padding_frozen=True,
            source_observations_weights_and_parameter_count_unchanged=True,
            independent_z_z_squared_corner_box_not_used=True,
            value_and_bounds_unit='Caller chart working units; no metric conversion performed by this kernel',
            axial_polynomial_coefficients_both_zero=bool(value[3] == value[5] == 0.),
            flat_continuum_cases_refused=True,
            flat_continuum_policy='Explicit ValueError; nonlinear guard remains required'))
