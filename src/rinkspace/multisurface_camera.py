"""A shared conditional camera for ice, ground contact and elevated boards.

Board pixels constrain world ray directions under an explicitly conditional
fixed optical centre. They never become Z=0 ground points. Ground correspondences
are intersected with Z=0; contact/marking curves have separate residual groups.
Image agreement is a test of this approximation, not proof of a fixed tripod.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np
from scipy.optimize import least_squares

from .whole_rink import undistort_observed_points, point_polyline_distances


@dataclass
class Camera:
    rotation: np.ndarray
    centre: np.ndarray
    focal: float
    principal: np.ndarray
    size: tuple[int, int] = (960, 540)
    radial_ray_k1: float = 0.

    @property
    def K(self):
        cx, cy = self.principal
        return np.array([[self.focal, 0., cx], [0., self.focal, cy], [0., 0., 1.]])

    @property
    def H(self):
        h = self.K @ np.column_stack((self.rotation[:, :2], -self.rotation @ self.centre))
        return h / h[2, 2]

    @property
    def distortion(self):
        if not self.radial_ray_k1:
            return None
        return dict(k1=self.radial_ray_k1*(self.size[0]/self.focal)**2,
                    centre_image=self.principal.tolist(), scale_image=self.size[0],
                    status='conditional_fixed_lens_radial_coefficient_not_independently_calibrated')

    def to_dict(self):
        return dict(K=self.K.tolist(), R=self.rotation.tolist(), C=self.centre.tolist(),
                    t=(-self.rotation @ self.centre).tolist(), focal_sample_px=self.focal,
                    principal_sample_px=self.principal.tolist(), sample_size=list(self.size),
                    radial_ray_k1=self.radial_ray_k1,
                    H_undistorted_sample_from_model=self.H.tolist(),
                    radial_distortion=self.distortion,
                    centre_units='conditional_rink_width_100_not_metres',
                    optical_centre_assumption='fixed_within_this_observed_shot_candidate')


def camera_from_homography(h, size=(960, 540), distortion=None):
    """Conditionally decompose a metric-ratio plane; report inconsistency.

    Uses both orthogonality and equal column norm equations for square pixels,
    with centred principal point. A nonphysical H is not silently certified.
    """
    h = np.asarray(h, float)
    cx, cy = np.asarray(size, float)/2
    a = h[:2, 0]-np.array([cx, cy])*h[2, 0]
    b = h[:2, 1]-np.array([cx, cy])*h[2, 1]
    numerator = np.array([np.dot(a, b), np.dot(a, a)-np.dot(b, b)])
    denominator = np.array([-h[2, 0]*h[2, 1], h[2, 1]**2-h[2, 0]**2])
    estimates = np.divide(numerator, denominator, out=np.full(2, np.nan), where=np.abs(denominator)>1e-12)
    valid = estimates[np.isfinite(estimates)&(estimates>0)]
    f = float(np.sqrt(np.median(valid))) if len(valid) else 2.*size[0]
    f = np.clip(f, .35*size[0], 20.*size[0])
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.]])
    q = np.linalg.solve(K, h)
    scale = 2./(np.linalg.norm(q[:, 0])+np.linalg.norm(q[:, 1]))
    candidates = []
    for sign in (1., -1.):
        r1, r2, t = (sign*scale*q).T
        u, _, vt = np.linalg.svd(np.column_stack((r1, r2, np.cross(r1, r2))))
        r = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
        centre = -r.T @ t
        candidates.append((centre[2] > 0 and t[2] > 0, r, centre))
    _, r, centre = max(candidates, key=lambda row: row[0])
    kr = float((distortion or {}).get('k1', 0.))*(f/float((distortion or {}).get('scale_image', size[0])))**2
    camera = Camera(r, centre, f, np.array([cx, cy]), tuple(size), kr)
    diagnostic = dict(focal_squared_estimates=estimates.tolist(),
                      positive_focal_equation_count=len(valid),
                      positive_camera_height=bool(centre[2]>0),
                      independent_calibration=False,
                      assumptions=['square_pixels', 'centred_principal_point', 'conditional_rink_aspect_ratio'])
    return camera, diagnostic


def world_rays(camera, pixels):
    pixels = np.asarray(pixels, float).reshape(-1, 2)
    raw, valid = undistort_observed_points(pixels, camera.distortion)
    rays = np.column_stack(((raw-camera.principal)/camera.focal, np.ones(len(raw)))) @ camera.rotation
    return rays, valid


def ground_points(camera, pixels):
    rays, valid = world_rays(camera, pixels)
    depth = -camera.centre[2]/np.where(np.abs(rays[:, 2])>1e-9, rays[:, 2], np.nan)
    points = camera.centre+depth[:, None]*rays
    valid &= np.isfinite(points).all(axis=1)&(depth>0)
    return points, valid


def project_camera(camera, points, *, rays=False):
    points = np.asarray(points, float).reshape(-1, 3)
    q = (points if rays else points-camera.centre) @ camera.rotation.T
    valid = np.isfinite(q).all(axis=1)&(q[:, 2]>1e-7)
    normalized = q[:, :2]/np.where(valid, q[:, 2], np.nan)[:, None]
    radius_squared=np.sum(normalized**2,axis=1)
    radial_scale=1+camera.radial_ray_k1*radius_squared
    valid &= (radial_scale>0)&(1+3*camera.radial_ray_k1*radius_squared>0)
    normal_distorted = normalized*radial_scale[:, None]
    normal_distorted[~valid]=np.nan
    return normal_distorted*camera.focal+camera.principal, valid


def fit_shared_camera(initial, correspondences, curves=(), *, max_nfev=35,
                      translation_sigma=None, centre_prior=None):
    """Fit rotation/zoom to separately typed measured groups.

    Each correspondence: surface, reference_camera, reference_xy, current_xy.
    Each curve: surface ice_marking/ice_board_contact, world_polyline Nx3,
    observed_xy Mx2, sigma_px. Elevated points use ray residuals exclusively.
    """
    groups = []
    for item in correspondences:
        surface = item['surface']
        if surface not in {'ice_marking', 'board_surface'}:
            raise ValueError('Unrecognised point observation surface')
        if 'world_points' in item:
            points=np.asarray(item['world_points'],float).reshape(-1,3)
            representation=item.get('representation','world_points' if surface=='ice_marking' else None)
            if representation not in {'world_points','world_rays'}:
                raise ValueError('Explicit board representation must distinguish XYZ from rays')
            if surface=='ice_marking' and representation!='world_points':
                raise ValueError('Ice must use ground world_points')
            ok=np.isfinite(points).all(axis=1)
            if surface=='ice_marking' and not np.allclose(points[:,2],0,atol=1e-8):
                raise ValueError('Ice points must be on Z=0')
        else:
            other = item['reference_camera']
            if surface == 'board_surface':
                if not np.allclose(other.centre,initial.centre,rtol=0,atol=1e-7):
                    raise ValueError('Board ray route requires the same conditional optical centre')
                representation='world_rays'
                points, ok = world_rays(other, item['reference_xy'])
            else:
                representation='world_points'
                points, ok = ground_points(other, item['reference_xy'])
        current = np.asarray(item['current_xy'], float).reshape(-1, 2)
        ok &= np.isfinite(current).all(axis=1)
        if np.count_nonzero(ok)<4:
            continue
        sigma = float(item.get('sigma_px', 1.5))
        # Limit each physical channel/temporal edge rather than allowing its
        # feature count to swamp the independent geometry channels.
        weight = np.sqrt(min(len(current[ok]), 40)/len(current[ok]))/sigma
        groups.append((surface, representation, points[ok], current[ok], weight))
    curve_groups=[]
    for item in curves:
        if item['surface'] not in {'ice_marking', 'ice_board_contact'}:
            raise ValueError('Elevated board cannot enter the ground-curve route')
        world=np.asarray(item['world_polyline'],float)
        if world.shape[1] != 3 or not np.allclose(world[:,2],0):
            raise ValueError('Ground curve must have Z=0')
        observed=np.asarray(item['observed_xy'],float).reshape(-1,2)
        if len(observed): curve_groups.append((item['surface'],world,observed,float(item.get('sigma_px',3.))))
    if not groups:
        return None, dict(reason='curves_without_enough_point_correspondences' if curve_groups else 'no_actual_typed_correspondences', success=False,
                          available_ground_curve_groups=len(curve_groups))

    if translation_sigma is not None and any(representation=='world_rays' for _,representation,*_ in groups):
        raise ValueError('Moving centre requires board XYZ, not fixed-centre rays')
    translated=translation_sigma is not None
    def unpack(x):
        r=cv2.Rodrigues(x[:3])[0] @ initial.rotation
        c=initial.centre+(x[4:7] if translated else 0.)
        return Camera(r, c, initial.focal*np.exp(x[3]),
                      initial.principal.copy(), initial.size, initial.radial_ray_k1)

    def residual(x):
        camera=unpack(x)
        values=[]
        for surface, representation, points, observed, weight in groups:
            prediction, good=project_camera(camera,points,rays=representation=='world_rays')
            delta=prediction-observed
            delta[~good]=1e3
            values.append((delta*weight).ravel())
        for _,world,observed,sigma in curve_groups:
            prediction,good=project_camera(camera,world)
            error=point_polyline_distances(observed,prediction,good)
            values.append(error/sigma*np.sqrt(min(8,len(error))/len(error)))
        if translated:
            prior=initial.centre if centre_prior is None else np.asarray(centre_prior,float)
            values.append((camera.centre-prior)/float(translation_sigma))
        # No velocity prior and no anchor-age stop. Only bounded numerical
        # rotation/zoom search; the accepted result still needs image checks.
        return np.concatenate(values)

    size=7 if translated else 4
    lower=[-.3,-.3,-.3,-.3]+([-5.]*3 if translated else [])
    upper=[.3,.3,.3,.3]+([5.]*3 if translated else [])
    fit=least_squares(residual,np.zeros(size),bounds=(lower,upper),
                      loss='soft_l1',f_scale=1.,max_nfev=max_nfev,xtol=1e-7,ftol=1e-7,gtol=1e-6)
    camera=unpack(fit.x)
    report={}
    for surface in ('ice_marking','board_surface'):
        errs=[]
        for label,representation,points,observed,_ in groups:
            if label==surface:
                pred,good=project_camera(camera,points,rays=representation=='world_rays')
                errs.extend(np.linalg.norm(pred[good]-observed[good],axis=1).tolist())
        report[surface]={'point_count':len(errs),'median_px':float(np.median(errs)) if errs else None,
                         'p95_px':float(np.percentile(errs,95)) if errs else None}
    report['ground_curves']=[{'surface':surface,'count':len(observed),
        'p95_px':float(np.percentile(point_polyline_distances(observed,*project_camera(camera,world)),95))}
        for surface,world,observed,_ in curve_groups]
    report.update(success=bool(fit.success),solver_converged=bool(fit.success),solver_status=int(fit.status),solver_message=str(fit.message),nfev=int(fit.nfev),parameters=fit.x.tolist(),
                  parameter_bound_hits=fit.active_mask.tolist(),robust_final_cost=float(fit.cost),initial_squared_residual_sum=float(np.sum(residual(np.zeros(size))**2)),
                  board_used_as_ground=False,fixed_centre_is_conditional=not translated,
                  centre_translation_sigma_model_units=translation_sigma,
                  centre_adjustment_model_units=(camera.centre-initial.centre).tolist(),
                  geometric_accuracy_validated=False)
    return camera, report


def projection_chain(camera, source_size=(1920, 1080)):
    return dict(H_reference_undistorted_from_model=camera.H.tolist(),
                reference_radial_distortion=camera.distortion,
                H_current_sample_from_reference_raw=np.eye(3).tolist(),
                H_source_image_from_sample=np.diag([source_size[0]/camera.size[0],source_size[1]/camera.size[1],1.]).tolist(),
                reference_visible_branch_sign=float(np.sign((-camera.rotation@camera.centre)[2])),
                raw_transfer_visible_branch_sign=1.,
                relative_geometry='shared_camera_rotation_zoom_with_typed_observations',
                reference_id='conditional_camera_current_frame')
