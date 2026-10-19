"""Joint hard rink geometry and Brown camera fit for a few wide views.

Extends JointSceneProblem without changing its geometric parameterization.
``problem['wide_angle_camera']`` (optionally overridden per installation) is::

    {'fit': ['k2_image', 'k3_image', 'p1_image', 'p2_image',
             'principal_x_px', 'principal_y_px'],
     'scale_image': 960., 'initial': {}, 'bounds': {}, 'priors': {}}

Every entry in ``initial``, ``bounds`` and ``priors`` uses those parameter names.
Omitted coefficients are zero and omitted principal points use the input frame.
Existing installation radial_distortion continues to own k1_image; its scale
must match scale_image. Coefficients use ideal image coordinates divided by
scale_image, so their magnitude does not change merely when focal changes.
Principal point and distortion are shared by frames of one installation.
Optional ``s1_image`` through ``s4_image`` add OpenCV's thin-prism model;
they are never fitted unless explicitly included in ``fit``/``active``.
Optional ``log_fy_over_fx`` fits shared nonuniform image scaling, initialized
from the input frame aspect ratio. It is fixed at that input value otherwise.
Optional ``k4_image`` through ``k6_image`` enable the OpenCV rational radial
denominator. Its positivity is guarded on the central rays, not offscreen ice.
No scale gauge is silently added: callers must declare any fixed unit anchor.
"""
from __future__ import annotations

import copy
import inspect

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

from .joint_scene import JointSceneProblem
from .scene_camera import Camera, nearest_polyline_vectors


BASE_OPTICAL_PARAMETERS = ('k2_image', 'k3_image', 'p1_image', 'p2_image',
                           'principal_x_px', 'principal_y_px')
THIN_PRISM_PARAMETERS = ('s1_image', 's2_image', 's3_image', 's4_image')
RATIONAL_PARAMETERS = ('k4_image', 'k5_image', 'k6_image')
OPTICAL_PARAMETERS = BASE_OPTICAL_PARAMETERS + THIN_PRISM_PARAMETERS + ('log_fy_over_fx',) + RATIONAL_PARAMETERS


def rational_denominator(normalized_xy, distortion):
    """OpenCV radial denominator at each ray; one for a polynomial model."""
    xy = np.asarray(normalized_xy, float).reshape(-1, 2)
    d = np.asarray(distortion, float).reshape(-1)
    if len(d) < 8:
        return np.ones(len(xy))
    r2 = np.sum(xy*xy, axis=1)
    return 1.+r2*(d[5]+r2*(d[6]+r2*d[7]))


def rational_minimum_denominator(normalized_xy, distortion):
    """Exact minimum along each axis-to-ray segment, including cubic extrema.

    This excludes positive outer branches beyond a pole even if the endpoint's
    denominator has become positive again. No offscreen world curve is guarded.
    """
    xy = np.asarray(normalized_xy, float).reshape(-1, 2)
    d = np.asarray(distortion, float).reshape(-1)
    result = np.minimum(1., rational_denominator(xy, d))
    if len(d) < 8 or not np.any(d[5:8]):
        return result
    r2 = np.sum(xy*xy, axis=1)
    coefficients = np.trim_zeros(np.array([3.*d[7], 2.*d[6], d[5]]), 'f')
    roots = np.roots(coefficients) if len(coefficients) > 1 else []
    for root in roots:
        if abs(np.imag(root)) > 1e-12 or np.real(root) <= 0:
            continue
        t = float(np.real(root))
        value = 1.+t*(d[5]+t*(d[6]+t*d[7]))
        result = np.where(r2 >= t, np.minimum(result, value), result)
    return result


def _brown_map_jacobian(xy, distortion):
    d = np.zeros(12); supplied = np.asarray(distortion, float).reshape(-1)
    if len(supplied) not in (0, 4, 5, 8, 12):
        raise ValueError('central Brown/rational/thin-prism inverse requires 0, 4, 5, 8 or 12 coefficients')
    d[:len(supplied)] = supplied
    k1, k2, p1, p2, k3 = d[:5]
    k4, k5, k6 = d[5:8]
    s1, s2, s3, s4 = d[8:12]
    x, y = np.asarray(xy, float).T
    r2 = x*x+y*y; numerator = 1.+k1*r2+k2*r2*r2+k3*r2*r2*r2
    denominator = 1.+k4*r2+k5*r2*r2+k6*r2*r2*r2
    # Finite recovery residual at a pole; invalid denominator rays are excluded
    # by every central-domain caller and separately penalized by sensor guards.
    safe_denominator = np.where(np.abs(denominator) >= 1e-12, denominator,
                                np.where(denominator < 0, -1e-12, 1e-12))
    radial = numerator/safe_denominator
    numerator_slope = k1+2.*k2*r2+3.*k3*r2*r2
    denominator_slope = k4+2.*k5*r2+3.*k6*r2*r2
    slope = (numerator_slope*safe_denominator-numerator*denominator_slope)/safe_denominator**2
    predicted = np.column_stack([x*radial+2.*p1*x*y+p2*(r2+2.*x*x)+s1*r2+s2*r2*r2,
                                 y*radial+p1*(r2+2.*y*y)+2.*p2*x*y+s3*r2+s4*r2*r2])
    j11 = radial+2.*x*x*slope+2.*p1*y+6.*p2*x+2.*x*(s1+2.*s2*r2)
    j12 = 2.*x*y*slope+2.*p1*x+2.*p2*y+2.*y*(s1+2.*s2*r2)
    j21 = 2.*x*y*slope+2.*p1*x+2.*p2*y+2.*x*(s3+2.*s4*r2)
    j22 = radial+2.*y*y*slope+6.*p1*y+2.*p2*x+2.*y*(s3+2.*s4*r2)
    # Thin-prism J is not symmetric. Positive definite symmetric part is a
    # conservative sufficient monotonicity/orientation condition. For ordinary
    # Brown distortion this exactly reduces to its former minimum eigenvalue.
    eigen = .5*(j11+j22-np.sqrt((j11-j22)**2+(j12+j21)**2))
    return predicted, j11, j12, j21, j22, eigen


def brown_central_inverse(camera, pixels_uv, *, continuation_steps=8,
                          max_iterations=24, tolerance_px=1e-5):
    """Invert raw pixels on the positive-Jacobian branch connected to the axis.

    Homotopy increases the target from the principal point to each requested
    pixel. Damped Newton steps remain on positive Jacobians; failed or unreachable
    targets are explicitly invalid. Unlike fixed-point undistortPoints this also
    returns the reprojection error. Inputs and returned arrays are vectorized.
    """
    pixels = np.asarray(pixels_uv, float).reshape(-1, 2)
    focal = np.diag(camera.K_px)[:2]
    target = (pixels-camera.K_px[:2, 2])/focal
    n = len(target); xy = np.zeros_like(target)
    alive = np.isfinite(target).all(axis=1)
    tolerance = tolerance_px/float(np.max(focal))
    for stage in range(1, continuation_steps+1):
        goal = target * (stage/continuation_steps)
        converged = ~alive
        for _ in range(max_iterations):
            indices = np.flatnonzero(alive & ~converged)
            if not len(indices):
                break
            z = xy[indices]
            predicted, a, b, c, d, eigen = _brown_map_jacobian(z, camera.distortion_opencv)
            error = predicted-goal[indices]; err2 = np.sum(error*error, axis=1)
            done = err2 <= tolerance*tolerance
            converged[indices[done]] = True
            indices, z, error, err2, a, b, c, d, eigen = [v[~done] for v in (indices, z, error, err2, a, b, c, d, eigen)]
            if not len(indices):
                continue
            determinant = a*d-b*c
            valid = (eigen > 1e-9) & (determinant > 1e-16) & (rational_minimum_denominator(z, camera.distortion_opencv) > 1e-9)
            alive[indices[~valid]] = False
            indices, z, error, err2, a, b, c, d, determinant = [v[valid] for v in (indices, z, error, err2, a, b, c, d, determinant)]
            if not len(indices):
                continue
            step = np.column_stack([(d*error[:, 0]-b*error[:, 1])/determinant,
                                     (a*error[:, 1]-c*error[:, 0])/determinant])
            # Bound the ray change so Newton cannot jump between distant roots.
            step *= np.minimum(1., .15/np.maximum(np.linalg.norm(step, axis=1), 1e-30))[:, None]
            factor = np.ones(len(indices)); accepted = np.zeros(len(indices), bool)
            for _ in range(18):
                todo = np.flatnonzero(~accepted)
                if not len(todo):
                    break
                proposal = z[todo]-factor[todo, None]*step[todo]
                pred, _, _, _, _, eig = _brown_map_jacobian(proposal, camera.distortion_opencv)
                _, _, _, _, _, mid_eig = _brown_map_jacobian((z[todo]+proposal)/2., camera.distortion_opencv)
                improves = np.sum((pred-goal[indices[todo]])**2, axis=1) < err2[todo]
                good = improves & (eig > 1e-9) & (mid_eig > 1e-9) & (rational_minimum_denominator(proposal, camera.distortion_opencv) > 1e-9)
                take = todo[good]
                xy[indices[take]] = proposal[good]; accepted[take] = True
                factor[todo[~good]] *= .5
            alive[indices[~accepted]] = False
        alive &= converged
    prediction, _, _, _, _, eigen = _brown_map_jacobian(xy, camera.distortion_opencv)
    error = np.linalg.norm((prediction-target)*focal, axis=1)
    denominator = rational_minimum_denominator(xy, camera.distortion_opencv)
    valid = alive & (eigen > 1e-9) & (denominator > 1e-9) & (error <= tolerance_px*2.)
    return dict(normalized_xy=xy, valid=valid, reprojection_error_px=error,
                minimum_eigenvalue=eigen, minimum_denominator_on_axis_path=denominator)


def brown_projection_validity(camera, points_xyz, *, tolerance_px=.05):
    """Project only the real central lens branch; preserve invalid gaps in curves.

    The returned dict contains uv, depth, valid and per-point numerical evidence.
    Callers must connect adjacent valid samples only, never join over a bad gap.
    No image dimensions or crop assumptions enter this optical-domain decision.
    """
    points = np.asarray(points_xyz, float).reshape(-1, 3)
    uv, depth = camera.project(points)
    rays = (points-camera.center_xyz) @ camera.R.T
    normalized = rays[:, :2]/np.maximum(depth[:, None], 1e-12)
    eigen = brown_minimum_eigenvalue(normalized, camera.distortion_opencv)
    denominator = rational_minimum_denominator(normalized, camera.distortion_opencv)
    good = (depth > 1e-6) & np.isfinite(uv).all(axis=1) & (eigen > 1e-9) & (denominator > 1e-9)
    ray_error = np.full(len(points), np.inf); inverse_error = np.full(len(points), np.inf)
    indices = np.flatnonzero(good)
    if len(indices):
        inverse = brown_central_inverse(camera, uv[indices])
        ray_error[indices] = np.linalg.norm((inverse['normalized_xy']-normalized[indices])*np.diag(camera.K_px)[:2], axis=1)
        inverse_error[indices] = inverse['reprojection_error_px']
        good[indices] &= inverse['valid'] & (ray_error[indices] <= tolerance_px)
    return dict(uv=uv, depth=depth, valid=good, minimum_eigenvalue=eigen,
                minimum_denominator_on_axis_path=denominator,
                inverse_ray_error_px=ray_error, inverse_reprojection_error_px=inverse_error)


def nearest_valid_polyline_vectors(points_uv, polyline_uv, valid):
    """Closest valid adjacent segment; invalid gaps are never bridged.

    Return observed-minus-projected vectors and the original starting segment
    indices. Empty indices explicitly mean that no physical segment is usable.
    """
    observed = np.asarray(points_uv, float).reshape(-1, 2)
    projected = np.asarray(polyline_uv, float).reshape(-1, 2)
    valid = np.asarray(valid, bool)
    indices = np.flatnonzero(valid[:-1] & valid[1:])
    if not len(indices):
        return np.full_like(observed, 1e3), np.empty(0, dtype=int)
    starts = projected[indices]; directions = projected[indices+1]-starts
    den = np.sum(directions*directions, axis=1)
    t = np.einsum('nsi,si->ns', observed[:, None, :]-starts[None, :, :], directions)/np.maximum(den, 1e-20)
    closest = starts[None, :, :]+np.clip(t, 0., 1.)[:, :, None]*directions[None, :, :]
    delta = observed[:, None, :]-closest
    chosen = np.argmin(np.sum(delta*delta, axis=2), axis=1)
    return delta[np.arange(len(observed)), chosen], indices[chosen]


def brown_minimum_eigenvalue(normalized_xy, distortion_opencv):
    """Smallest eigenvalue of the symmetric part of the distortion Jacobian.

    Brown's J is symmetric; thin-prism J generally is not. Positive values are
    a conservative sufficient condition against folds for either model.
    This sampled diagnostic is not a global proof of inversion.
    """
    xy = np.asarray(normalized_xy, float).reshape(-1, 2)
    d = np.zeros(5)
    supplied = np.asarray(distortion_opencv, float).reshape(-1)
    if len(supplied) in (8, 12):
        return _brown_map_jacobian(xy, supplied)[-1]
    if len(supplied) not in (0, 4, 5):
        raise ValueError('wide Brown model requires 0, 4 or 5 coefficients')
    d[:len(supplied)] = supplied
    k1, k2, p1, p2, k3 = d
    xx, yy = xy[:, 0], xy[:, 1]
    r2 = xx * xx + yy * yy
    radial = 1. + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
    slope = k1 + 2.*k2*r2 + 3.*k3*r2*r2
    a = radial + 2.*xx*xx*slope + 2.*p1*yy + 6.*p2*xx
    b = 2.*xx*yy*slope + 2.*p1*xx + 2.*p2*yy
    c = radial + 2.*yy*yy*slope + 6.*p1*yy + 2.*p2*xx
    return .5 * (a + c - np.sqrt((a-c)**2 + 4.*b*b))


class WideAngleJointSceneProblem(JointSceneProblem):
    """Dense numerical Jacobian includes all geometry and optical dependencies."""

    def __init__(self, scene, problem):
        problem = copy.deepcopy(problem)
        # Compact spelling used by the wide-frame fitting command.
        if 'wide_angle_optics' in problem and 'wide_angle_camera' not in problem:
            compact = problem['wide_angle_optics']
            names = {'cx': 'principal_x_px', 'cy': 'principal_y_px'}
            config = {k: copy.deepcopy(v) for k, v in compact.items() if k != 'active'}
            config['fit'] = [names.get(name, name) for name in compact.get('active', BASE_OPTICAL_PARAMETERS)]
            for key in ('initial', 'bounds', 'priors'):
                if key in config:
                    config[key] = {names.get(name, name): value for name, value in config[key].items()}
            problem['wide_angle_camera'] = config
        self._initializing_wide = True
        self.optical_specs = {}
        self.optical_indices = {}
        # The parent k1 convenience interface historically rejects unequal fx/fy.
        # Its distortion coefficient is normalized by fx and remains valid for
        # an unequal output fy. Bypass only that construction-time legacy guard;
        # restore the exact input aspect before building this class's cameras.
        original_aspects = {f['id']: float(f.get('fy_over_fx', 1.)) for f in problem['frames']}
        if any(not np.isfinite(v) or v <= 0 for v in original_aspects.values()):
            raise ValueError('positive finite input fy_over_fx required')
        for frame in problem['frames']:
            frame['fy_over_fx'] = 1.
        super().__init__(scene, problem)
        for fid, frame in self.frames.items():
            frame['fy_over_fx'] = original_aspects[fid]
        values = self.x0.tolist()
        lower, upper = [a.tolist() for a in self.bounds]
        default = copy.deepcopy(self.problem.get('wide_angle_camera', {}))
        for iid, installation in self.installations.items():
            config = {**default, **copy.deepcopy(installation.get('wide_angle_camera', {}))}
            if not config or config.get('enabled') is False:
                continue
            related = [f for f in self.frames.values() if f['installation_id'] == iid]
            if not related:
                continue
            if any(f.get('fy_over_fx', 1.) != related[0].get('fy_over_fx', 1.) for f in related):
                raise ValueError('shared wide optics require the same input fy_over_fx')
            if any(tuple(f['image_size_wh']) != tuple(related[0]['image_size_wh']) for f in related):
                raise ValueError('shared wide optics require the same image dimensions')
            if any(tuple(f['principal_point_px']) != tuple(related[0]['principal_point_px']) for f in related):
                raise ValueError('shared wide optics require the same initial principal point')
            if any(np.any(f.get('distortion_opencv', [])) for f in related):
                raise ValueError('wide optics use image-normalized coefficients; clear fixed OpenCV distortion')
            radial = installation.get('radial_distortion', {})
            scale = float(config.get('scale_image', radial.get('scale_image', max(related[0]['image_size_wh'])/2.)))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('positive scale_image required')
            if radial and not np.isclose(scale, radial['scale_image'], atol=1e-12, rtol=0):
                raise ValueError('k1 and higher Brown coefficients must use the same image scale')
            fit = config.get('fit', list(BASE_OPTICAL_PARAMETERS))
            if not isinstance(fit, list) or len(set(fit)) != len(fit) or set(fit) - set(OPTICAL_PARAMETERS):
                raise ValueError('fit must list unique supported wide optical parameter names')
            w, h = map(float, related[0]['image_size_wh'])
            initial = dict(k2_image=0., k3_image=0., p1_image=0., p2_image=0.,
                           k4_image=0., k5_image=0., k6_image=0.,
                           s1_image=0., s2_image=0., s3_image=0., s4_image=0.,
                           log_fy_over_fx=float(np.log(related[0].get('fy_over_fx', 1.))),
                           principal_x_px=float(related[0]['principal_point_px'][0]),
                           principal_y_px=float(related[0]['principal_point_px'][1]))
            initial.update(config.get('initial', {}))
            bounds = dict(k2_image=[-1., 1.], k3_image=[-1., 1.],
                          k4_image=[-10., 10.], k5_image=[-10., 10.], k6_image=[-10., 10.],
                          p1_image=[-.1, .1], p2_image=[-.1, .1],
                          s1_image=[-2., 2.], s2_image=[-2., 2.],
                          s3_image=[-2., 2.], s4_image=[-2., 2.],
                          log_fy_over_fx=[min(float(np.log(.8)), initial['log_fy_over_fx']-.1),
                                          max(float(np.log(1.25)), initial['log_fy_over_fx']+.1)],
                          principal_x_px=[0., w], principal_y_px=[0., h])
            bounds.update(config.get('bounds', {}))
            priors = config.get('priors', {})
            if any(set(v)-set(OPTICAL_PARAMETERS) for v in (initial, bounds, priors)):
                raise ValueError('unknown wide optical parameter')
            self.optical_specs[iid] = dict(scale_image=scale, initial=initial,
                                          bounds=bounds, fit=fit, priors=priors,
                                          rational_enabled=bool(set(fit)&set(RATIONAL_PARAMETERS)
                                              or any(initial[n] != 0 for n in RATIONAL_PARAMETERS)),
                                          thin_prism_enabled=bool(set(fit)&set(THIN_PRISM_PARAMETERS)
                                              or any(initial[n] != 0 for n in THIN_PRISM_PARAMETERS)))
            indices = self.optical_indices[iid] = {}
            for name in OPTICAL_PARAMETERS:
                value = float(initial[name]); lo, hi = map(float, bounds[name])
                if not np.isfinite([value, lo, hi]).all() or not lo <= value <= hi or not lo < hi:
                    raise ValueError(f'{iid}:{name}: invalid initial value/bounds')
                if name not in fit:
                    continue
                index = len(values); indices[name] = index
                values.append(value); lower.append(lo); upper.append(hi)
                self.parameter_names.append(f'installation:{iid}:{name}')
                if name in priors:
                    prior = priors[name]
                    if not np.isfinite([prior['mean'], prior['sigma']]).all() or prior['sigma'] <= 0:
                        raise ValueError('optical prior needs finite mean and positive sigma')
                    self.parameter_priors.append((index, copy.deepcopy(prior)))
        self.x0 = np.asarray(values, float)
        self.bounds = np.asarray(lower, float), np.asarray(upper, float)
        self._initializing_wide = False
        self._domain_uv = {}
        for fid, frame in self.frames.items():
            if frame['installation_id'] in self.optical_specs:
                w, h = frame['image_size_wh']
                xx, yy = np.meshgrid(np.linspace(0., w-1., 13), np.linspace(0., h-1., 9))
                self._domain_uv[fid] = np.column_stack([xx.ravel(), yy.ravel()])
        self.sensor_coverage_specs = {}
        self.sensor_coverage_slices = {}
        coverage = copy.deepcopy(self.problem.get('raw_sensor_coverage', {}))
        if coverage.get('enabled', False):
            count = int(coverage.get('edge_samples', 3))
            if count < 2:
                raise ValueError('raw sensor coverage requires at least two samples per edge')
            values = self.x0.tolist(); lower, upper = [a.tolist() for a in self.bounds]
            for fid in self._domain_uv:
                w, h = self.frames[fid]['image_size_wh']
                sx = np.linspace(0., w-1., count); sy = np.linspace(0., h-1., count)
                raw_uv = np.unique(np.vstack([np.column_stack([sx, np.zeros(count)]),
                    np.column_stack([sx, np.full(count, h-1.)]),
                    np.column_stack([np.zeros(count), sy]),
                    np.column_stack([np.full(count, w-1.), sy])]), axis=0)
                spec = dict(raw_pixels_uv=raw_uv.tolist(), edge_samples=count,
                    pixel_weight=float(coverage.get('pixel_weight', 10.)),
                    jacobian_weight=float(coverage.get('jacobian_weight', 1000.)),
                    minimum_eigenvalue=float(coverage.get('minimum_eigenvalue', .02)),
                    minimum_denominator=float(coverage.get('minimum_denominator', .001)),
                    denominator_weight=float(coverage.get('denominator_weight', 1000.)),
                    path_samples=int(coverage.get('path_samples', 17)),
                    ray_bounds=list(coverage.get('ray_bounds', [-10., 10.])),
                    provenance='Numerical camera-domain constraints at known sensor pixels; not observed rink landmarks or physical measurements.')
                if (not np.isfinite([spec['pixel_weight'], spec['jacobian_weight'], spec['minimum_eigenvalue'], spec['minimum_denominator'], spec['denominator_weight']]).all()
                    or min(spec['pixel_weight'], spec['jacobian_weight'], spec['minimum_eigenvalue'], spec['minimum_denominator'], spec['denominator_weight']) <= 0
                    or spec['path_samples'] < 3 or not spec['ray_bounds'][0] < 0 < spec['ray_bounds'][1]):
                    raise ValueError('invalid raw sensor coverage weights/domain')
                self.sensor_coverage_specs[fid] = spec
                start = len(values)
                for index in range(len(raw_uv)):
                    for axis in ('x', 'y'):
                        self.parameter_names.append(f'sensor_domain:{fid}:anchor_{index}:ideal_ray_{axis}')
                        values.append(0.); lower.append(spec['ray_bounds'][0]); upper.append(spec['ray_bounds'][1])
                self.sensor_coverage_slices[fid] = slice(start, len(values))
            self.x0 = np.asarray(values, float)
            self.bounds = np.asarray(lower, float), np.asarray(upper, float)
            self.x0 = self.initialize_sensor_coverage(self.x0)
        if not np.isfinite(self.residual(self.x0)).all():
            raise ValueError('nonfinite wide initial residual')

    def optical_values(self, x, iid):
        spec = self.optical_specs[iid]
        values = dict(spec['initial'])
        values.update({name: float(x[index]) for name, index in self.optical_indices[iid].items()})
        return values

    def optics_report(self, x):
        """Serialize fitted and fixed optical values, domains and frame cameras."""
        installations = {}
        for iid, spec in self.optical_specs.items():
            radial = self.installations[iid].get('radial_distortion')
            k1 = 0. if radial is None else float(x[self.radial_indices[iid]] if iid in self.radial_indices else radial['initial_k1_image'])
            installations[iid] = {
                **copy.deepcopy(spec), 'values': {'k1_image': k1, **self.optical_values(x, iid)},
                'fy_over_fx': float(np.exp(self.optical_values(x, iid)['log_fy_over_fx'])),
                'active': (['k1_image'] if iid in self.radial_indices else []) + list(spec['fit']),
                'fixed': (['k1_image'] if iid not in self.radial_indices else []) + [n for n in OPTICAL_PARAMETERS if n not in spec['fit']],
            }
        prism = any(s['thin_prism_enabled'] for s in self.optical_specs.values())
        rational = any(s['rational_enabled'] for s in self.optical_specs.values())
        return {'model': 'Brown k1/k2/k3/p1/p2' + (' with OpenCV rational denominator k4/k5/k6' if rational else '') + (' with OpenCV thin-prism s1/s2/s3/s4' if prism else '') + '; fy/fx fixed to input unless explicitly active; zero skew',
                'coefficient_conversion': 'a=fx/scale; OpenCV=[k1_image*a^2,k2_image*a^4,p1_image*a,p2_image*a,k3_image*a^6]; rational appends [k4_image*a^2,k5_image*a^4,k6_image*a^6]; prism extends to 12 with [s1_image*a,s2_image*a^3,s3_image*a,s4_image*a^3]; omitted coefficients are zero',
                'domain_guard': 'positive symmetric part of the full quotient Jacobian; positive rational denominator including every cubic extremum along axis-to-ray paths; full-Jacobian Newton central inverse and raw sensor coverage checks',
                'installations': installations,
                'image_domain_minimum_eigenvalues': self.domain_minimum_eigenvalues(x),
                'image_domain_minimum_denominators': self.domain_minimum_denominators(x),
                'raw_sensor_coverage_constraints': self.sensor_coverage_report(x),
                'frame_cameras': {fid: self.camera(x, fid).to_dict() for fid in self.frames}}

    def initialize_sensor_coverage(self, x):
        """Initialize auxiliary rays after a caller transfers a warm camera fit.

        Return a copy; geometry and camera parameters are unchanged. A failed
        inverse supplies its last central-branch iterate as a numerical seed,
        which has a nonzero, explicitly reported sensor coverage residual.
        """
        initialized = np.asarray(x, float).copy()
        for fid, spec in self.sensor_coverage_specs.items():
            camera = self.camera(initialized, fid)
            inverse = brown_central_inverse(camera, spec['raw_pixels_uv'])
            where = self.sensor_coverage_slices[fid]
            initialized[where] = np.clip(inverse['normalized_xy'].ravel(),
                                         self.bounds[0][where]+1e-9, self.bounds[1][where]-1e-9)
            spec['initialization'] = dict(method='central-branch damped Newton on the supplied warm camera',
                converged_count=int(inverse['valid'].sum()), anchor_count=len(inverse['valid']),
                maximum_seed_reprojection_error_px=float(inverse['reprojection_error_px'].max()),
                camera=camera.to_dict())
        return initialized

    def _sensor_coverage_terms(self, x, fid):
        spec = self.sensor_coverage_specs[fid]
        camera = self.camera(x, fid)
        rays = np.asarray(x[self.sensor_coverage_slices[fid]]).reshape(-1, 2)
        predicted, _, _, _, _, _ = _brown_map_jacobian(rays, camera.distortion_opencv)
        pixel_delta = predicted*np.diag(camera.K_px)[:2]+camera.K_px[:2, 2]-spec['raw_pixels_uv']
        fractions = np.linspace(0., 1., spec['path_samples'])
        path = rays[:, None, :]*fractions[None, :, None]
        eigen = brown_minimum_eigenvalue(path.reshape(-1, 2), camera.distortion_opencv).reshape(len(rays), -1)
        return pixel_delta, eigen

    def sensor_coverage_report(self, x):
        result = {}
        for fid, spec in self.sensor_coverage_specs.items():
            delta, eigen = self._sensor_coverage_terms(x, fid)
            result[fid] = {**copy.deepcopy(spec),
                'auxiliary_ideal_rays_xy': np.asarray(x[self.sensor_coverage_slices[fid]]).reshape(-1, 2).tolist(),
                'reprojection_error_px': np.linalg.norm(delta, axis=1).tolist(),
                'minimum_eigenvalue_along_each_axis_to_ray_path': eigen.min(axis=1).tolist(),
                'minimum_denominator_along_each_axis_to_ray_path': rational_minimum_denominator(
                    np.asarray(x[self.sensor_coverage_slices[fid]]).reshape(-1, 2), self.camera(x, fid).distortion_opencv).tolist(),
                'interpretation': 'Auxiliary ray coordinates are numerical variables for sensor coverage, not new image observations; full raw-image inverse validation is reported separately.'}
        return result

    def camera(self, x, fid):
        camera = super().camera(x, fid)
        iid = self.frames[fid]['installation_id']
        if self._initializing_wide or iid not in self.optical_specs:
            return camera
        spec = self.optical_specs[iid]; values = self.optical_values(x, iid)
        focal = camera.K_px[0, 0]; ratio = focal / spec['scale_image']
        K = camera.K_px.copy()
        K[0, 2], K[1, 2] = values['principal_x_px'], values['principal_y_px']
        K[1, 1] = focal*np.exp(values['log_fy_over_fx'])
        d = np.zeros(12 if spec['thin_prism_enabled'] else 8 if spec['rational_enabled'] else 5)
        if len(camera.distortion_opencv):
            d[0] = camera.distortion_opencv[0]
        d[1] = values['k2_image'] * ratio**4
        d[2] = values['p1_image'] * ratio
        d[3] = values['p2_image'] * ratio
        d[4] = values['k3_image'] * ratio**6
        if spec['rational_enabled']:
            d[5:8] = [values['k4_image']*ratio**2, values['k5_image']*ratio**4,
                      values['k6_image']*ratio**6]
        if spec['thin_prism_enabled']:
            d[8:] = [values['s1_image']*ratio, values['s2_image']*ratio**3,
                     values['s3_image']*ratio, values['s4_image']*ratio**3]
        return Camera(camera.center_xyz, camera.rvec_world_to_camera, K, d)

    def _error(self, x, observation, uv, geometry, cameras):
        if self._initializing_wide or self.frames[observation['frame_id']]['installation_id'] not in self.optical_specs:
            return super()._error(x, observation, uv, geometry, cameras)
        xyz = geometry.points(observation['target'], observation['type'], self.samples)
        camera = cameras[observation['frame_id']]
        projected, depth = camera.project(xyz)
        rays = (xyz-camera.center_xyz) @ camera.R.T
        normalized = rays[:, :2] / np.maximum(rays[:, 2:3], 1e-8)
        eigen = brown_minimum_eigenvalue(normalized, camera.distortion_opencv)
        denominator = rational_minimum_denominator(normalized, camera.distortion_opencv)
        if observation['type'] == 'point':
            delta = np.where(np.isfinite(projected), projected-uv, 1e3)
            return delta, depth, float(min(eigen.min(), denominator.min()))
        valid = (depth > 1e-6) & (eigen > 1e-9) & (denominator > 1e-9) & np.isfinite(projected).all(axis=1)
        # Reject distant positive-Jacobian islands separated from the central
        # lens branch. The exact inverse check remains in rendering and sensor
        # acceptance; this inexpensive ray-path check is used during fitting.
        for alpha in (.25, .5, .75):
            valid &= brown_minimum_eigenvalue(normalized*alpha, camera.distortion_opencv) > 1e-9
        delta, matched = nearest_valid_polyline_vectors(uv, projected, valid)
        if not len(matched):
            # No valid correspondence exists: keep a visible failure residual,
            # with a domain recovery direction rather than dropping evidence.
            return delta, np.array([float(depth.max())]), float(eigen.max())
        endpoints = np.r_[matched, matched+1]
        # Off-screen polynomial extensions are neither observed rays nor guard
        # constraints. Only the actual matched segments may affect this guard.
        return delta, depth[endpoints], float(eigen[endpoints].min())

    def domain_minimum_eigenvalues(self, x):
        result = {}
        for fid, uv in self._domain_uv.items():
            camera = self.camera(x, fid)
            normalized = (uv-camera.K_px[:2, 2])/np.diag(camera.K_px)[:2]
            result[fid] = float(np.min(brown_minimum_eigenvalue(normalized, camera.distortion_opencv)))
        return result

    def domain_minimum_denominators(self, x):
        result = {}
        for fid, uv in self._domain_uv.items():
            if not self.optical_specs[self.frames[fid]['installation_id']]['rational_enabled']:
                continue
            camera = self.camera(x, fid)
            normalized = (uv-camera.K_px[:2, 2])/np.diag(camera.K_px)[:2]
            result[fid] = float(rational_minimum_denominator(normalized, camera.distortion_opencv).min())
        return result

    def raw_sensor_domain_diagnostics(self, x, *, columns=65, rows=37):
        """Check actual raw sensor pixels, not just the smaller ideal rectangle."""
        result = {}
        for fid in self._domain_uv:
            frame = self.frames[fid]; w, h = frame['image_size_wh']
            xx, yy = np.meshgrid(np.linspace(0., w-1., columns), np.linspace(0., h-1., rows))
            uv = np.column_stack([xx.ravel(), yy.ravel()])
            inverse = brown_central_inverse(self.camera(x, fid), uv)
            valid = inverse['valid']; invalid = uv[~valid]
            result[fid] = dict(samples=int(len(uv)), central_inverse_valid=int(valid.sum()),
                all_sampled_pixels_in_central_domain=bool(valid.all()),
                minimum_eigenvalue_on_inverse_rays=float(inverse['minimum_eigenvalue'][valid].min()) if valid.any() else None,
                minimum_denominator_on_axis_to_inverse_rays=float(inverse['minimum_denominator_on_axis_path'][valid].min()) if valid.any() else None,
                maximum_reprojection_error_px=float(inverse['reprojection_error_px'][valid].max()) if valid.any() else None,
                invalid_sample_bounds_uv=None if not len(invalid) else [invalid.min(axis=0).tolist(), invalid.max(axis=0).tolist()],
                note='Sampled coverage of raw sensor domain; outer polynomial branches are not camera rays.')
        return result

    def residual(self, x):
        base = super().residual(x)
        if self._initializing_wide:
            return base
        # Fixed row count through all evaluations, including when no fold exists.
        guards = np.array([min(value-1e-5, 0.)*1e3
                           for value in self.domain_minimum_eigenvalues(x).values()])
        parts = [base, guards]
        parts.append(np.array([min(value-.001, 0.)*1e3 for value in self.domain_minimum_denominators(x).values()]))
        for fid, spec in self.sensor_coverage_specs.items():
            delta, eigen = self._sensor_coverage_terms(x, fid)
            parts.append((delta*spec['pixel_weight']).ravel())
            parts.append((np.minimum(eigen-spec['minimum_eigenvalue'], 0.)*spec['jacobian_weight']).ravel())
            if self.optical_specs[self.frames[fid]['installation_id']]['rational_enabled']:
                rays = np.asarray(x[self.sensor_coverage_slices[fid]]).reshape(-1, 2)
                denominator = rational_minimum_denominator(rays, self.camera(x, fid).distortion_opencv)
                parts.append(np.minimum(denominator-spec['minimum_denominator'], 0.)*spec['denominator_weight'])
        return np.concatenate(parts)

    @property
    def jac_sparsity(self):
        # Safe full dependency pattern for callers; solve uses a dense Jacobian.
        return csr_matrix(np.ones((len(self.residual(self.x0)), len(self.x0)), dtype=np.int8))

    def summarize(self, x):
        summary = super().summarize(x)
        if not self._initializing_wide:
            domain = self.domain_minimum_eigenvalues(x)
            summary['wide_brown_domain_minimum_eigenvalues'] = domain
            denominators = self.domain_minimum_denominators(x)
            summary['wide_brown_domain_minimum_denominators'] = denominators
            summary['fit_geometry_valid'] &= all(v > 0. for v in denominators.values())
            summary['fit_geometry_valid'] &= all(v > 0. for v in domain.values())
            sensor = self.raw_sensor_domain_diagnostics(x)
            summary['wide_brown_raw_sensor_domain'] = sensor
            summary['raw_sensor_coverage_constraints'] = self.sensor_coverage_report(x)
            summary['fit_geometry_valid'] &= all(v['all_sampled_pixels_in_central_domain'] for v in sensor.values())
            summary['brown_fold_check'] = 'full Jacobian on matched valid curve segments and 13x9 ideal grid; central-branch inverse consistency on 65x37 raw sensor grid'
        return summary

    def solve(self, max_nfev=None, *, verbose=0, callback=None, x0=None,
              loss='soft_l1', f_scale=1., **kwargs):
        """Jointly vary the existing geometry/pose variables and added optics.

        ``kwargs`` forwards tolerances and diff_step to scipy least_squares.
        Check observations remain excluded from the objective.
        """
        if not len(self.x0):
            raise ValueError('no variable parameters')
        initial = self.x0 if x0 is None else np.asarray(x0, float)
        if initial.shape != self.x0.shape or not np.isfinite(initial).all():
            raise ValueError('warm start must match parameter vector size and be finite')
        if np.any(initial < self.bounds[0]) or np.any(initial > self.bounds[1]):
            raise ValueError('warm start outside parameter bounds')
        options = dict(method='trf', tr_solver='exact', x_scale='jac', loss=loss,
                       f_scale=f_scale, max_nfev=max_nfev, verbose=verbose)
        options.update(kwargs)
        if callback is not None:
            if 'callback' not in inspect.signature(least_squares).parameters:
                raise ValueError('installed SciPy does not support least_squares callback')
            options['callback'] = callback
        result = least_squares(self.residual, initial, bounds=self.bounds, **options)
        self.result = result
        jac = np.asarray(result.jac)
        norms = np.linalg.norm(jac, axis=0)
        singular = np.linalg.svd(jac/np.maximum(norms, 1e-15), compute_uv=False)
        threshold = singular[0]*1e-6 if len(singular) else 0.
        after = self.summarize(result.x)
        installations = []
        for iid, install in self.installations.items():
            related = next((fid for fid, f in self.frames.items() if f['installation_id'] == iid), None)
            if related is None:
                continue
            radial = install.get('radial_distortion')
            entry = dict(id=iid, physical_camera_id=install.get('physical_camera_id'),
                         center_xyz=self.camera(result.x, related).center_xyz.tolist(),
                         position_status='conditional_joint_candidate_NOT_measured_installation',
                         center_bounds_xyz=install.get('center_bounds_xyz'),
                         radial_k1_image=None if radial is None else float(result.x[self.radial_indices[iid]] if iid in self.radial_indices else radial['initial_k1_image']),
                         radial_scale_image=None if radial is None else radial['scale_image'])
            if iid in self.optical_specs:
                entry['wide_angle_camera'] = {**copy.deepcopy(self.optical_specs[iid]),
                                             'values': self.optical_values(result.x, iid)}
            installations.append(entry)
        return dict(schema_version='rinkspace.wide-angle-joint-scene-fit.v1',
            synthetic=self.problem.get('synthetic', False), scene_sha256=self.scene.sha256,
            scene_geometry_held_fixed=not bool(self.scene_specs or self.landmark_slices),
            rink_geometry_held_fixed=not bool(self.scene_specs), physical_accuracy_verified=False,
            accepted_geometry_candidate=bool(after['fit_geometry_valid']),
            coordinate_gauge=self.coordinate_gauge_description(),
            hard_geometry_constraints=copy.deepcopy(self.hard_geometry_constraints),
            optimizer=dict(success=bool(result.success), message=str(result.message),
                           nfev=int(result.nfev), njev=int(result.njev or 0),
                           cost=float(result.cost), optimality=float(result.optimality),
                           warm_started=x0 is not None, jacobian='dense_full_dependency',
                           loss=loss, f_scale=f_scale,
                           near_bounds_parameters=[self.parameter_names[i] for i in np.flatnonzero(result.active_mask)]),
            parameter_names=list(self.parameter_names), parameter_values=result.x.tolist(),
            parameter_changes=self.parameter_changes(result.x), retained_parameters=self.retained_parameters,
            parameter_bounds=[a.tolist() for a in self.bounds],
            parameter_priors=[dict(parameter=self.parameter_names[i], **p) for i, p in self.parameter_priors],
            board_landmarks=self.landmark_estimates(result.x), before=self.summarize(self.x0), after=after,
            installations=installations,
            frames=[{**{k: v for k, v in f.items() if not k.startswith('initial_')},
                     'camera': self.camera(result.x, fid).to_dict()} for fid, f in self.frames.items()],
            local_numerical_identifiability=dict(column_normalized_singular_values=singular.tolist(),
                rank_at_relative_1e_6=int(np.sum(singular > threshold)), parameter_count=len(result.x),
                zero_sensitivity_parameters=[self.parameter_names[i] for i in np.flatnonzero(norms < 1e-12)],
                note='Local numerical diagnostic, not physical uncertainty or global uniqueness.'),
            limitations=[self.coordinate_gauge_description(),
                         'Added optical coefficients can remain weakly identifiable from one planar view.',
                         'Sampled Brown fold checks are not a global proof of invertibility.',
                         'Real image alignment must be inspected separately; optimizer success is not visual acceptance.'])


WideAngleJointProblem = WideAngleJointSceneProblem
