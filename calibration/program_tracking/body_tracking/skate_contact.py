"""Exact plane contact witnesses for a rigid figure-skate template.

This is a small geometry/residual kernel, not an estimator of contact state or
force. All input lengths and sigmas are metres in one declared world frame.
Unknown/airborne feet receive nonpenetration rows but no support rows unless
the caller explicitly supplies a SupportBand. No world-XY velocity is fixed.

Features are dictionaries with a unique ``id`` within each skate and one of:
* kind='arc': center, axis_cos, axis_sin, radius_m, theta_start_rad,
  theta_end_rad. p(theta)=center+radius*(axis_cos*cos+axis_sin*sin).
* kind='bezier': control_points, 2..4 Cartesian control points.
* kind='points': points, e.g. all vertices of a triangular solid mesh or pick.

The minimum of height over a triangle mesh occurs at a vertex. For rounded
runner curves use arcs/Bezier segments, not render-sampled points. Each skate
has one nonpenetration row: the worst adjusted clearance over all its features.
Thus subdividing a curve or duplicating mesh vertices does not change cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Plane:
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    offset_m: float = 0.0  # signed height = n dot p + offset

    def values(self):
        n = np.asarray(self.normal, dtype=float)
        if n.shape != (3,) or not np.isfinite(n).all() or not np.isfinite(self.offset_m):
            raise ValueError('Plane must be finite')
        if abs(np.linalg.norm(n) - 1.0) > 1e-8:
            raise ValueError('Plane normal must be unit length; do not silently rescale metres')
        return n, float(self.offset_m)


@dataclass(frozen=True)
class SupportBand:
    """Explicit observed/conditional support, possibly with uncertain foot/edge.

    members lists (skate_id, feature_id). The lowest eligible feature supplies
    one row, not one row per runner point. To express 'one foot is standing but
    identity unknown', list the allowed features from both feet in one band.
    lower_m/upper_m are a geometric tolerance, not calibrated ice mechanics.
    """
    members: tuple[tuple[str, str], ...]
    lower_m: float
    upper_m: float
    sigma_m: float
    label: str = 'explicit_support'


def _vector(value, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError(name + ' must be a finite 3-vector')
    return a


def _points(value, name):
    a = np.asarray(value, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or len(a) == 0 or not np.isfinite(a).all():
        raise ValueError(name + ' must be a nonempty finite (N,3) array')
    return a


def feature_minimum(feature: Mapping, plane: Plane = Plane()):
    """Return an exact height witness and partials at its active parameter.

    At an interior extremum, the envelope theorem permits differentiating the
    height at the fixed witness parameter. At ties the minimum has a set of
    subgradients; a deterministic active branch is returned, not a claim of
    differentiability. The derivative tests deliberately avoid switch points.
    """
    n, offset = plane.values()
    fid = str(feature['id'])
    kind = feature.get('kind', 'arc' if 'radius_m' in feature else None)
    if kind == 'arc':
        c = _vector(feature['center'], 'center')
        u = _vector(feature['axis_cos'], 'axis_cos')
        v = _vector(feature['axis_sin'], 'axis_sin')
        radius = float(feature['radius_m'])
        a, b = float(feature['theta_start_rad']), float(feature['theta_end_rad'])
        if not np.isfinite([radius, a, b]).all() or radius <= 0 or not a < b or b-a > 2*np.pi+1e-12:
            raise ValueError('Arc requires positive radius and increasing span <= 2*pi')
        if max(abs(u@u-1), abs(v@v-1), abs(u@v)) > 1e-8:
            raise ValueError('Arc axes must be orthonormal')
        A, B = radius*(n@u), radius*(n@v)
        candidates = [a, b]
        if np.hypot(A, B) > 1e-15:
            # Both stationary phases are included. Only a handful can lie in
            # an interval of at most 2*pi; no sampling resolution enters here.
            phase = np.arctan2(B, A)
            k0 = int(np.ceil((a-phase)/np.pi))
            k1 = int(np.floor((b-phase)/np.pi))
            candidates.extend(phase+k*np.pi for k in range(k0, k1+1))
        t = np.asarray(candidates)
        points = c + radius*(np.cos(t)[:, None]*u + np.sin(t)[:, None]*v)
        index = int(np.argmin(points@n+offset))
        parameter, p = float(t[index]), points[index]
        partials = dict(center=n.copy(), axis_cos=radius*np.cos(parameter)*n,
                        axis_sin=radius*np.sin(parameter)*n,
                        radius_m=float(n@(u*np.cos(parameter)+v*np.sin(parameter))))
        # Endpoint angles only contribute when that endpoint is the witness.
        angle_derivative = float(radius*n@(-u*np.sin(parameter)+v*np.cos(parameter)))
        partials['theta_start_rad'] = angle_derivative if index == 0 else 0.0
        partials['theta_end_rad'] = angle_derivative if index == 1 else 0.0
    elif kind == 'bezier':
        controls = _points(feature['control_points'], 'control_points')
        degree = len(controls)-1
        if degree not in (1, 2, 3):
            raise ValueError('Only linear, quadratic and cubic Bezier segments are supported')
        heights = controls@n+offset
        # Convert Bernstein heights to power coefficients, then solve the
        # linear/quadratic derivative polynomial including endpoint witnesses.
        coefficients = np.zeros(degree+1)
        for i in range(degree+1):
            for j in range(degree-i+1):
                coefficients[i+j] += heights[i]*comb(degree, i)*comb(degree-i, j)*(-1)**j
        derivative = np.arange(1, degree+1)*coefficients[1:]
        roots = np.polynomial.polynomial.polyroots(derivative) if np.any(derivative) else []
        candidates = [0.0, 1.0]
        candidates.extend(float(r.real) for r in roots if abs(r.imag) < 1e-10 and 0 < r.real < 1)
        t = np.asarray(candidates)
        weights = np.stack([comb(degree, i)*t**i*(1-t)**(degree-i) for i in range(degree+1)], axis=1)
        points = weights@controls
        index = int(np.argmin(points@n+offset))
        parameter, p = float(t[index]), points[index]
        partials = dict(control_points=weights[index, :, None]*n)
    elif kind == 'points':
        points = _points(feature['points'], 'points')
        index = int(np.argmin(points@n+offset))
        parameter, p = index, points[index]
        derivative = np.zeros_like(points)
        derivative[index] = n
        partials = dict(points=derivative)
    else:
        raise ValueError('Unknown feature kind: '+str(kind))
    return dict(feature_id=fid, semantic=str(feature.get('semantic', fid)), kind=kind,
                height_m=float(n@p+offset), point=np.asarray(p).copy(), parameter=parameter,
                height_partials=partials, support_eligible=bool(feature.get('support_eligible', True)))


def template_world_features(template: Mapping, rotation, translation):
    """Adapter for skate_template.build_skate(), preserving feature semantics.

    Use template['support_feature_ids']['glide' or 'pick'] for support members;
    guards are part of nonpenetration, not evidence for a contact point.
    """
    features = [dict(a, kind='arc') for a in template['contact_arcs']]
    features += [dict(g, kind='points') for g in template['contact_point_groups']]
    return rigid_transform_features(features, rotation, translation)


def rigid_transform_features(features: Sequence[Mapping], rotation, translation):
    """Transform the same template used for rendering to world contact geometry."""
    R = np.asarray(rotation, dtype=float)
    t = _vector(translation, 'translation')
    if R.shape != (3, 3) or not np.isfinite(R).all() or not np.allclose(R.T@R, np.eye(3), atol=1e-9) or np.linalg.det(R) <= 0:
        raise ValueError('rotation must be a proper 3D rotation')
    result = []
    for original in features:
        f = dict(original)
        kind = f.get('kind', 'arc' if 'radius_m' in f else None)
        if kind == 'arc':
            f['center'] = R@_vector(f['center'], 'center')+t
            f['axis_cos'] = R@_vector(f['axis_cos'], 'axis_cos')
            f['axis_sin'] = R@_vector(f['axis_sin'], 'axis_sin')
        elif kind in ('points', 'bezier'):
            key = 'points' if kind == 'points' else 'control_points'
            f[key] = _points(f[key], key)@R.T+t
        else:
            raise ValueError('Unknown feature kind: '+str(kind))
        result.append(f)
    return result


def trapezoid_time_weights(times_s, *, normalize_duration=False, singleton_duration_s=None):
    """Quadrature weights independent of frame rate; no implicit 50 Hz constant.

    Integral weights sum to elapsed duration. normalize_duration=True gives a
    duration-mean cost. Use the same convention for comparable solver blocks.
    Duplicate timestamps must be resolved by the caller, not counted twice.
    """
    t = np.asarray(times_s, dtype=float)
    if t.ndim != 1 or len(t) == 0 or not np.isfinite(t).all():
        raise ValueError('times_s must be a finite nonempty 1D array')
    if len(t) == 1:
        if singleton_duration_s is None or not np.isfinite(singleton_duration_s) or singleton_duration_s <= 0:
            raise ValueError('A single frame needs its explicit represented duration')
        weights = np.array([float(singleton_duration_s)])
    else:
        dt = np.diff(t)
        if np.any(dt <= 0):
            raise ValueError('Times must be strictly increasing')
        weights = np.r_[dt[0]/2, (dt[:-1]+dt[1:])/2, dt[-1]/2]
    return weights/weights.sum() if normalize_duration else weights


def frame_residuals(skates: Mapping[str, Sequence[Mapping]], *, nonpenetration_sigma_m: float,
                    supports: Sequence[SupportBand] = (), allowed_penetration_m=0.0,
                    plane: Plane = Plane(), time_weight_s: float = 1.0):
    """Return residuals and exact active witnesses for one shared body state.

    allowed_penetration_m may be a scalar or a mapping by semantic, with a
    'default' entry. This permits small declared runner/pick tolerances while
    the sole, heel, plate and stanchion mesh use zero or separate clearance.
    Tolerances/sigmas are engineering choices, never measured ice stiffness.

    rows[i]['dres_dheight'] * plane.normal @ d(witness_point)/d(state)
    supplies the local Jacobian away from active-set switches. The point must
    use the saved witness parameter; do not differentiate argmin itself.
    """
    n, _ = plane.values()
    sigma = float(nonpenetration_sigma_m)
    weight = float(time_weight_s)
    if not np.isfinite([sigma, weight]).all() or sigma <= 0 or weight <= 0 or not skates:
        raise ValueError('Positive sigma, time weight, and at least one skate required')
    scale = np.sqrt(weight)
    lookup, by_skate = {}, {}
    for foot, features in skates.items():
        if not features:
            raise ValueError('Every skate must provide its contact/solid geometry')
        records = []
        for feature in features:
            witness = feature_minimum(feature, plane)
            key = (str(foot), witness['feature_id'])
            if key in lookup:
                raise ValueError('Duplicate feature identity: '+str(key))
            semantic = witness['semantic']
            tolerance = (allowed_penetration_m.get(semantic, allowed_penetration_m.get('default', 0.0))
                         if isinstance(allowed_penetration_m, Mapping) else allowed_penetration_m)
            tolerance = float(tolerance)
            if not np.isfinite(tolerance) or tolerance < 0:
                raise ValueError('Allowed penetration must be nonnegative and finite')
            record = dict(witness, skate_id=str(foot), allowed_penetration_m=tolerance)
            lookup[key] = record
            records.append(record)
        by_skate[str(foot)] = records
    residuals, rows = [], []
    for foot, records in by_skate.items():
        witness = min(records, key=lambda r: r['height_m']+r['allowed_penetration_m'])
        clearance = witness['height_m']+witness['allowed_penetration_m']
        r = scale*max(-clearance, 0.0)/sigma
        derivative = -scale/sigma if clearance < 0 else 0.0
        residuals.append(r)
        rows.append(dict(witness, row_kind='nonpenetration', label=foot+'_nonpenetration',
                         dres_dheight=derivative, plane_normal=n.copy(),
                         constraint_kind='lower_bound', constraint_value_m=clearance,
                         constraint_lower_m=0.0, constraint_upper_m=None,
                         violation_m=max(-clearance, 0.0)))
    for support in supports:
        if not support.members or len(set(support.members)) != len(support.members):
            raise ValueError('Support needs unique explicit (skate, feature) members')
        if not np.isfinite([support.lower_m, support.upper_m, support.sigma_m]).all() or support.lower_m > support.upper_m or support.sigma_m <= 0:
            raise ValueError('Invalid support height band or sigma')
        try:
            eligible = [lookup[(str(a), str(b))] for a, b in support.members]
        except KeyError as e:
            raise ValueError('Support references missing geometry: '+str(e)) from e
        if any(not row['support_eligible'] for row in eligible):
            raise ValueError('Nonpenetration-only solid guard cannot become a support candidate')
        witness = min(eligible, key=lambda r: r['height_m'])
        h = witness['height_m']
        error = h-float(np.clip(h, support.lower_m, support.upper_m))
        residuals.append(scale*error/support.sigma_m)
        rows.append(dict(witness, row_kind='support', label=support.label,
                         dres_dheight=scale/support.sigma_m if error != 0 else 0.0,
                         plane_normal=n.copy(), members=support.members,
                         constraint_kind='interval', constraint_value_m=h,
                         constraint_lower_m=support.lower_m, constraint_upper_m=support.upper_m,
                         signed_band_error_m=error, violation_m=abs(error)))
    return dict(residuals=np.asarray(residuals), rows=rows)


def contact_jacobian(result, point_jacobian):
    """Chain active witnesses to the caller's existing body/shape Jacobian.

    point_jacobian(row) returns a (3,N) array for that witness's fixed curve
    parameter or mesh vertex. This does not introduce an independent skate
    pose. For sparse spline columns assemble equivalent sparse rows directly.
    """
    output = []
    for row in result['rows']:
        derivative = np.asarray(point_jacobian(row), dtype=float)
        if derivative.ndim != 2 or derivative.shape[0] != 3 or not np.isfinite(derivative).all():
            raise ValueError('point_jacobian must return a finite (3,N) array')
        output.append(row['dres_dheight']*(row['plane_normal']@derivative))
    return np.stack(output)


def constraint_jacobian(result, point_jacobian):
    """Unweighted metre-valued constraint Jacobian for an external solver.

    This pairs with constraint_value_m and the lower/upper bounds in each row.
    Unlike the residual Jacobian it is not zero merely because a constraint
    is currently feasible. Useful for SQP/augmented-Lagrangian feasibility
    control. This module's least-squares rows alone do not guarantee contact.
    """
    output = []
    for row in result['rows']:
        derivative = np.asarray(point_jacobian(row), dtype=float)
        if derivative.ndim != 2 or derivative.shape[0] != 3 or not np.isfinite(derivative).all():
            raise ValueError('point_jacobian must return a finite (3,N) array')
        output.append(row['plane_normal']@derivative)
    return np.stack(output)
