"""Named, observed ice contours constrain cameras, never ellipse centres.

All input points are visible measured samples in the camera sample domain.
Physical groups have fixed total weight; a dense trace is not many landmarks.
The conic equation is used only in the undistorted pinhole domain. Its first
order image-normal residual is not an independent geometric accuracy check.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .multisurface_camera import Camera
from .whole_rink import undistort_observed_points


@dataclass
class GroundCurve:
    frame: int
    physical_id: str
    kind: str
    equation_model: np.ndarray
    observed_xy: np.ndarray
    sigma_px: float = 3.
    group_weight: float = 1.
    observation_id: str = ''
    tangents_xy: np.ndarray | None = None
    tangent_sigma_rad: np.ndarray | None = None


def model_equation(primitive):
    p = np.asarray(primitive['points_model'], float)
    if primitive['kind'] == 'circle':
        centre = (p.min(axis=0) + p.max(axis=0)) / 2
        radius = float(np.median(np.linalg.norm(p - centre, axis=1)))
        a, b = centre
        return 'circle', np.array([[1., 0., -a], [0., 1., -b], [-a, -b, a*a+b*b-radius*radius]])
    if primitive['kind'] == 'axis_line':
        line = np.cross(np.r_[p[0], 1.], np.r_[p[-1], 1.])
        return 'line', line / np.linalg.norm(line[:2])
    raise ValueError('Only identified full circles or axis lines have analytic equations')


def curve_residual_h(h, kind, equation, points):
    """Signed geometric distance to the projected ellipse/supporting line.

    A first-order algebraic normal distance severely understates the error at
    the ends of a flat ellipse. Use a double-precision closest-ellipse solve.
    """
    inverse = np.linalg.inv(np.asarray(h, float))
    xy = np.c_[np.asarray(points, float), np.ones(len(points))]
    if kind == 'circle':
        q = inverse.T @ equation @ inverse
        centre=-np.linalg.solve(q[:2,:2],q[:2,2])
        constant=-(centre @ q[:2,:2] @ centre+2*q[:2,2] @ centre+q[2,2])
        values,vectors=np.linalg.eigh(q[:2,:2]/constant)
        if np.any(values<=0) or not np.isfinite(values).all():
            gradient=2*(xy @ q)[:,:2]
            return np.einsum('ni,ij,nj->n',xy,q,xy)/np.maximum(np.linalg.norm(gradient,axis=1),1e-12)
        axes=1/np.sqrt(values)  # eigh increasing: a >= b
        local=(np.asarray(points,float)-centre) @ vectors
        p=np.abs(local);a,b=axes
        inside=np.sum((p/axes)**2,axis=1)<1
        lower=np.where(inside,-b*b+b*p[:,1],0.)
        upper=np.linalg.norm(p*axes,axis=1)
        for _ in range(40):
            t=(lower+upper)/2
            value=np.sum((p*axes/np.maximum(t[:,None]+axes**2,1e-25))**2,axis=1)
            lower=np.where(value>1,t,lower);upper=np.where(value>1,upper,t)
        t=(lower+upper)/2
        closest=p*axes**2/np.maximum(t[:,None]+axes**2,1e-25)
        on_axis=p[:,1]<1e-10
        if np.any(on_axis):
            fraction=a*p[on_axis,0]/max(a*a-b*b,1e-20)
            closest[on_axis,0]=a*np.minimum(fraction,1)
            closest[on_axis,1]=b*np.sqrt(np.maximum(1-fraction**2,0))
        distance=np.linalg.norm(p-closest,axis=1)
        return np.where(inside,-distance,distance)
    if kind == 'line':
        line = inverse.T @ equation
        return (xy @ line) / max(np.linalg.norm(line[:2]), 1e-12)
    raise ValueError('Unknown measured curve type')


def curve_residual(camera, curve):
    points, valid = undistort_observed_points(curve.observed_xy, camera.distortion)
    result = curve_residual_h(camera.H, curve.kind, curve.equation_model, points)
    # Radial observations are transformed through the same camera chain;
    # retain their sample-pixel interpretation to first order only.
    return np.where(valid, result, 1e4)


def curve_direction_residual_h(h,curve):
    """Measured local arc direction versus the projected conic normal.

    Missing/cropped endpoints never acquire a synthetic tangent. This tests
    image orientation, not a correspondence with an ellipse centre or a
    completed arc. Returns sine-angle errors, separately from pixel errors.
    """
    if curve.tangents_xy is None:
        return np.empty(0)
    inverse=np.linalg.inv(h)
    if curve.kind=='circle':
        q=inverse.T @ curve.equation_model @ inverse
        normal=(np.c_[curve.observed_xy,np.ones(len(curve.observed_xy))] @ q)[:,:2]
    else:
        line=inverse.T @ curve.equation_model
        normal=np.repeat(line[None,:2],len(curve.observed_xy),axis=0)
    normal=normal/np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
    return np.sum(normal*curve.tangents_xy,axis=1)


def curve_direction_residual(camera,curve):
    if curve.tangents_xy is None:
        return np.empty(0)
    if camera.distortion is None:
        return curve_direction_residual_h(camera.H,curve)
    from dataclasses import replace
    points,valid=undistort_observed_points(curve.observed_xy,camera.distortion)
    before,v1=undistort_observed_points(curve.observed_xy-curve.tangents_xy,camera.distortion)
    after,v2=undistort_observed_points(curve.observed_xy+curve.tangents_xy,camera.distortion)
    tangent=after-before
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-12)
    changed=replace(curve,observed_xy=points,tangents_xy=tangent)
    return np.where(valid&v1&v2,curve_direction_residual_h(camera.H,changed),1.)


def observations_to_curves(frame, observations, model, scale=.5):
    primitives = {p['id']: p for p in model['structure']['primitives']}
    groups = []
    for item in observations:
        identity = item.get('selected_model_id')
        if not identity or item.get('association_status') in {'conflict', 'masked_out', 'not_detected'}:
            continue
        if identity not in primitives:
            raise ValueError('Observed identity is not in the bound model')
        explicit_segments=item.get('visible_segments_source_px')
        ranges=item.get('visible_segment_ranges')
        if explicit_segments:
            parts=[np.asarray(p,float).reshape(-1,2) for p in explicit_segments if len(p)]
            if not parts:continue
            offsets=np.r_[0,np.cumsum([len(p) for p in parts])]
            ranges=list(zip(offsets[:-1],offsets[1:]))
            points=np.concatenate(parts)*scale
        else:
            points = np.asarray(item.get('points_source_px', []), float).reshape(-1, 2) * scale
        if len(points) < 3 or not np.isfinite(points).all():
            continue
        sigma=max(float(item.get('measurement_uncertainty_source_px', 6.))*scale,.75)
        tangents=np.zeros_like(points);tangent_sigma=np.ones(len(points))
        ranges=ranges or [[0,len(points)]]
        for start,stop in ranges:
            start=max(int(start),0);stop=min(int(stop),len(points))
            for j in range(start+1,stop-1):
                before=points[j]-points[j-1];after=points[j+1]-points[j]
                if min(np.linalg.norm(before),np.linalg.norm(after))<1e-6 or np.dot(before,after)<0:
                    continue
                chord=points[j+1]-points[j-1];length=np.linalg.norm(chord)
                if length>2*sigma:
                    tangents[j]=chord/length
                    tangent_sigma[j]=max(.08,2*sigma/length)
        # Uniform trace thinning, independent of model projection.
        if len(points) > 32:
            take=np.linspace(0,len(points)-1,32).round().astype(int)
            points=points[take];tangents=tangents[take];tangent_sigma=tangent_sigma[take]
        kind, equation = model_equation(primitives[identity])
        groups.append(GroundCurve(frame, identity, kind, equation, points,
                    sigma,float(item.get('group_weight', 1.)),item.get('observation_id',''),
                    tangents,tangent_sigma))
    return groups


def fit_camera_curves(initial, curves, *, max_nfev=180, adjust_centre=False,
                      adjust_radial=False):
    """Diagnostic/local hypothesis fit. Extra freedoms are opt-in comparisons.

    No curve identity is chosen by this function. One visible ellipse is not
    enough to declare a full-rink localization; caller retains alternatives.
    """
    if not curves:
        raise ValueError('A camera fit needs actual identified visible contours')
    base = initial
    n = 4 + 3*adjust_centre + adjust_radial
    def unpack(x):
        centre = base.centre.copy()
        if adjust_centre:
            centre += x[4:7] * 10.
        radial = base.radial_ray_k1 + (x[-1] if adjust_radial else 0.)
        return Camera(Rotation.from_rotvec(x[:3]).as_matrix() @ base.rotation,
                      centre, base.focal*np.exp(x[3]), base.principal.copy(), base.size, radial)
    def residual(x):
        camera = unpack(x)
        values = [curve_residual(camera, c) / c.sigma_px * np.sqrt(16*c.group_weight/len(c.observed_xy)) for c in curves]
        for c in curves:
            if c.tangents_xy is not None:
                values.append(curve_direction_residual(camera,c)/c.tangent_sigma_rad*np.sqrt(16*c.group_weight/len(c.observed_xy)))
        if adjust_centre:
            # A broad conditional regularizer, explicitly reported, not evidence.
            values.append(x[4:7] / 5.)
            values.append(np.array([max(.25-camera.centre[2], 0)*20]))
        if adjust_radial:
            values.append(np.array([x[-1]/.3]))
        return np.concatenate(values)
    bounds = np.array([.8,.8,.8,1.5]+([5.,5.,3.] if adjust_centre else [])+([.4] if adjust_radial else []))
    fit = least_squares(residual, np.zeros(n), bounds=(-bounds,bounds), loss='soft_l1',
                        f_scale=2., x_scale='jac', max_nfev=max_nfev,
                        ftol=1e-8, xtol=1e-8, gtol=1e-7)
    camera = unpack(fit.x)
    singular = np.linalg.svd(fit.jac[:sum(len(c.observed_xy) for c in curves)], compute_uv=False)
    groups = [dict(physical_id=c.physical_id, observation_id=c.observation_id,
                   initial_median_px=float(np.median(np.abs(curve_residual(base,c)))),
                   median_px=float(np.median(np.abs(curve_residual(camera,c)))),
                   p95_px=float(np.percentile(np.abs(curve_residual(camera,c)),95)),
                   sigma_sample_px=c.sigma_px, group_weight=c.group_weight,
                   measured_sample_count=len(c.observed_xy)) for c in curves]
    return camera, dict(solver_converged=bool(fit.success), nfev=int(fit.nfev), message=str(fit.message),
                       initial_cost=float(np.sum(residual(np.zeros(n))**2)), final_cost=float(2*fit.cost),
                       parameter_step=fit.x.tolist(), jacobian_singular_values=singular.tolist(),
                       physical_group_count=len({c.physical_id for c in curves}), groups=groups,
                       adjusted_centre=adjust_centre, adjusted_radial=adjust_radial,
                       centre_prior_sigma_model_units=50. if adjust_centre else None,
                       radial_prior_sigma=.3 if adjust_radial else None,
                       independent_accuracy_validated=False)
