"""Refit reference cameras while keeping every rink geometry value fixed.

The output is a conditional square-pixel camera, not a surveyed calibration.
Raw image observations retain their geometry class and whole physical groups
share weight; samples along one fuzzy line do not become independent points.
"""
from __future__ import annotations

import copy
import cv2
import numpy as np
from scipy.optimize import least_squares

from .whole_rink import observation_residual, _pack_h, _unpack_h


def camera_matrix(focal_px, image_size):
    w, h = image_size
    return np.array([[focal_px, 0., w/2], [0., focal_px, h/2], [0., 0., 1.]])


def decompose_ground_homography(homography, image_size, focal_px=None):
    """Return a proper rotation and positive-depth seed, including f ambiguity."""
    h = np.asarray(homography, float)
    width, height = image_size
    centered = np.array([[1., 0., -width/2], [0., 1., -height/2], [0., 0., 1.]]) @ h
    a, b = centered[:, 0], centered[:, 1]
    values = []
    for numerator, denominator in [(a[:2] @ b[:2], a[2]*b[2]),
                                  (a[:2] @ a[:2]-b[:2] @ b[:2], a[2]**2-b[2]**2)]:
        values.append(float(-numerator/denominator) if abs(denominator)>1e-14 else None)
    positive = [v for v in values if v is not None and v>0]
    if focal_px is None:
        focal_px = np.sqrt(np.median(positive)) if positive else width*2
    focal_px = float(focal_px)
    focal_px = float(np.clip(focal_px, width*.3, width*12))
    k = camera_matrix(focal_px, image_size)
    q = np.linalg.solve(k, h)
    scale = 2/(np.linalg.norm(q[:, 0])+np.linalg.norm(q[:, 1]))
    if q[2, 2]*scale < 0:
        scale = -scale
    r1, r2, t = q[:, 0]*scale, q[:, 1]*scale, q[:, 2]*scale
    u, _, vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    r = u @ np.diag([1., 1., np.linalg.det(u@vt)]) @ vt
    return {"K": k, "R": r, "t": t, "focal_px": focal_px,
            "orthogonality_focal_squared_candidates": values}


def ground_homography(camera):
    k, r, t = np.asarray(camera['K']), np.asarray(camera['R']), np.asarray(camera['t'])
    h = k @ np.column_stack([r[:, 0], r[:, 1], t])
    return h/h[2, 2]


def _weights(observations):
    group_counts = {}
    for ob in observations:
        group_counts[ob['feature_id']] = group_counts.get(ob['feature_id'], 0) + 1
    return [np.sqrt(ob.get('effective_independent_samples', 2) /
                    (len(ob['points_image'])*group_counts[ob['feature_id']])) /
            ob.get('sigma_image_px', 3.) for ob in observations]


def residual_summary(observations, features, parameters, h):
    summary = []
    for ob in observations:
        raw = observation_residual(ob, features[ob['feature_id']], parameters, h,
                                   visible_branch_sign=1)
        error = np.linalg.norm(raw.reshape(-1,2),axis=1) if ob.get('type')=='point' else np.abs(raw)
        summary.append({'observation_id': ob['id'], 'feature_id': ob['feature_id'],
                        'surface_type': ob.get('surface_type', 'ice_marking'),
                        'role': ob.get('role', 'fit'), 'sample_count': len(error),
                        'median_px': float(np.median(error)), 'p95_px': float(np.percentile(error, 95)),
                        'max_px': float(np.max(error)), 'errors_px': error.tolist(),
                        'independent_validation': False})
    return summary


def refit_fixed_geometry(model, view_id, *, include_development_checks=True, max_nfev=800):
    """Compare old H, free-H refit and one K/R/t camera on unchanged geometry."""
    view = next(v for v in model['view_poses'] if v['id']==view_id)
    size = view['sample_size']
    features = {f['id']: f for f in model['features']}
    parameters = {k: p['value'] for k, p in model['parameter_estimates'].items()}
    observations = [copy.deepcopy(o) for o in model['observations'] if o['view_id']==view_id]
    for ob in observations:
        ob['surface_type'] = 'ice_board_contact' if ob['feature_id'] in {
            'far_side', 'near_side', 'left_end', 'right_end'} else 'ice_marking'
        ob['annotation_origin'] = 'preserved_v07_sparse_original_source_annotation'
        if ob.get('role')=='check' and include_development_checks:
            ob['previous_role'] = 'check'
            ob['role'] = 'fit'
            ob['development_reuse'] = 'Used for v09 camera repair; no longer held out.'
    fit_observations = [ob for ob in observations if ob.get('role', 'fit')=='fit']
    weights = _weights(fit_observations)

    def errors(h):
        try:
            return np.concatenate([observation_residual(ob, features[ob['feature_id']], parameters, h,
                                   visible_branch_sign=1)*w for ob, w in zip(fit_observations, weights)])
        except (ValueError, np.linalg.LinAlgError):
            return np.full(sum(len(ob['points_image'])*(2 if ob.get('type')=='point' else 1)
                               for ob in fit_observations), 1e6)

    old_h = np.asarray(view['H_image_from_model'])
    free = least_squares(lambda x: errors(_unpack_h(x)), _pack_h(old_h), loss='soft_l1',
                         x_scale='jac', max_nfev=max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    free_h = _unpack_h(free.x)
    seed = decompose_ground_homography(free_h, size)
    rvec = cv2.Rodrigues(seed['R'])[0].ravel()
    x0 = np.r_[rvec, seed['t']/100., np.log(seed['focal_px'])]

    def unpack(x):
        return {'K': camera_matrix(np.exp(x[6]), size), 'R': cv2.Rodrigues(x[:3])[0], 't': x[3:6]*100.}

    lower = np.r_[np.full(6, -np.inf), np.log(size[0]*.3)]
    upper = np.r_[np.full(6, np.inf), np.log(size[0]*12)]
    physical = least_squares(lambda x: errors(ground_homography(unpack(x))), x0,
        bounds=(lower, upper), loss='soft_l1', x_scale='jac', max_nfev=max_nfev,
        ftol=1e-10, xtol=1e-10, gtol=1e-10)
    camera = unpack(physical.x)
    physical_h = ground_homography(camera)
    center = -camera['R'].T @ camera['t']
    return {'view_id': view_id, 'source_id': view['source_id'], 'source_sha256': view['source_sha256'],
        'source_pts': view['pts'], 'source_time_s': view['source_time_s'], 'source_time_base': view['time_base'],
        'image_size': size, 'source_size': view['source_size'], 'image': view['image'],
        'sample_from_source': np.diag([size[0]/view['source_size'][0], size[1]/view['source_size'][1], 1.]).tolist(),
        'geometry_fixed': True, 'geometry_units': 'conditional_normalized_width_100_not_metres',
        'status': 'development_camera_candidate_not_independently_validated',
        'image_domain': 'raw_px_with_zero_distortion_approximation_not_calibrated',
        'radial_distortion': {'k1': 0., 'status': 'zero_distortion_approximation_not_estimated'},
        'observations': observations,
        'baseline': {'H_image_from_model': old_h.tolist(), 'residuals': residual_summary(observations, features, parameters, old_h)},
        'free_homography': {'H_image_from_model': free_h.tolist(), 'H_model_from_image': np.linalg.inv(free_h).tolist(),
            'solver_success': bool(free.success), 'nfev': free.nfev,
            'residuals': residual_summary(observations, features, parameters, free_h)},
        'camera': {**{k: v.tolist() for k, v in camera.items()}, 'center_model': center.tolist(),
            'focal_px': float(camera['K'][0, 0]), 'square_pixels': True, 'principal_point': 'image_center_assumed',
            'H_image_from_model': physical_h.tolist(), 'H_model_from_image': np.linalg.inv(physical_h).tolist(),
            'solver_success': bool(physical.success), 'nfev': physical.nfev,
            'jacobian_rank': int(np.linalg.matrix_rank(physical.jac)), 'unknown_count': 7,
            'orthogonality_focal_squared_seed_candidates': seed['orthogonality_focal_squared_candidates'],
            'residuals': residual_summary(observations, features, parameters, physical_h)},
        'limitations': ['Fixed conditional rink geometry can be wrong despite a camera fit.',
            'Only observed line normals and visible curve fragments enter residuals; along-line identity remains weak.',
            'Camera centre and focal length are conditional on square pixels, centred principal point and zero lens distortion.',
            'This does not establish a fixed camera centre across time.']}


def refit_shared_center(model, reference_results, *, max_nfev=1000):
    """Fit a testable fixed-centre camera group without averaging homographies.

    Observations are reprojected using common C and separate R/f. This is a
    hypothesis comparison, never an automatic declaration that pan/zoom holds.
    """
    if len(reference_results)<2:
        raise ValueError('A shared-centre comparison needs at least two views')
    features={f['id']:f for f in model['features']}
    parameters={k:p['value'] for k,p in model['parameter_estimates'].items()}
    center=np.mean([r['camera']['center_model'] for r in reference_results],axis=0)
    x0=list(center/100.)
    for row in reference_results:
        x0.extend(cv2.Rodrigues(np.asarray(row['camera']['R']))[0].ravel())
        x0.append(np.log(row['camera']['focal_px']))
    observations=[[o for o in row['observations'] if o.get('role','fit')=='fit'] for row in reference_results]
    weights=[_weights(obs) for obs in observations]

    def unpack(x,index):
        r=cv2.Rodrigues(x[3+4*index:6+4*index])[0]
        return {'K':camera_matrix(np.exp(x[6+4*index]),reference_results[index]['image_size']),
                'R':r,'t':-r@(x[:3]*100.)}

    def residual(x):
        out=[]
        for index,(obs,ww) in enumerate(zip(observations,weights)):
            h=ground_homography(unpack(x,index))
            out.extend(observation_residual(o,features[o['feature_id']],parameters,h,
                         visible_branch_sign=1)*w for o,w in zip(obs,ww))
        return np.concatenate(out)

    lower=np.full(len(x0),-np.inf);upper=np.full(len(x0),np.inf)
    for index,row in enumerate(reference_results):
        lower[6+4*index]=np.log(row['image_size'][0]*.3)
        upper[6+4*index]=np.log(row['image_size'][0]*12)
    result=least_squares(residual,x0,bounds=(lower,upper),loss='soft_l1',x_scale='jac',
                         max_nfev=max_nfev,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    cameras=[]
    for index,row in enumerate(reference_results):
        camera=unpack(result.x,index);h=ground_homography(camera)
        cameras.append({'view_id':row['view_id'],**{k:v.tolist() for k,v in camera.items()},
          'focal_px':float(camera['K'][0,0]),'center_model':(result.x[:3]*100.).tolist(),
          'H_image_from_model':h.tolist(),'H_model_from_image':np.linalg.inv(h).tolist(),
          'residuals':residual_summary(row['observations'],features,parameters,h)})
    return {'status':'fixed_center_hypothesis_requires_image_motion_check',
            'geometry_fixed':True,'independent_validation':False,
            'view_ids':[r['view_id'] for r in reference_results],
            'center_model':(result.x[:3]*100.).tolist(),'cameras':cameras,
            'solver_success':bool(result.success),'jacobian_rank':int(np.linalg.matrix_rank(result.jac)),
            'unknown_count':len(result.x),'nfev':result.nfev}
