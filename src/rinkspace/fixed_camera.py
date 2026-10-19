"""Versioned physical-camera hypotheses tied to one rink model.

Background pixels are rays under a same-centre hypothesis; they never become
ice coordinates. A fit is a candidate until separately withheld ice agrees.
"""
from __future__ import annotations

from dataclasses import replace
import cv2
import numpy as np
from scipy.optimize import least_squares

from .multisurface_camera import Camera, world_rays, project_camera


def camera_from_record(record):
    value = record.get('camera', record)
    size = tuple(value.get('sample_size', record.get('sample_size', [960, 540])))
    focal = value.get('focal_sample_px', value.get('focal_px'))
    if focal is None:
        focal = value['K'][0][0]
    return Camera(np.asarray(value['R'], float),
                  np.asarray(value.get('C', value.get('center_model')), float),
                  float(focal), np.asarray(value.get('principal_sample_px', np.asarray(size)/2), float),
                  size, float(value.get('radial_ray_k1', 0.)))


def _summary(errors):
    errors = np.asarray(errors, float)
    finite = errors[np.isfinite(errors)]
    return {'count': int(len(errors)), 'finite_count': int(len(finite)),
            'median_px': float(np.median(finite)) if len(finite) else None,
            'p95_px': float(np.percentile(finite, 95)) if len(finite) else None,
            'max_px': float(np.max(finite)) if len(finite) else None}


def rotation_zoom_h(reference_camera, current_camera):
    if not np.allclose(reference_camera.centre, current_camera.centre, atol=1e-7, rtol=0):
        raise ValueError('Scene-ray transfer requires one installation optical centre')
    if reference_camera.radial_ray_k1 or current_camera.radial_ray_k1:
        raise ValueError('A raw-pixel scene homography cannot include radial distortion')
    h = current_camera.K @ current_camera.rotation @ reference_camera.rotation.T @ np.linalg.inv(reference_camera.K)
    return h / h[2, 2]


def fit_rotation_zoom(reference_camera, reference_xy, current_xy, surface_labels=None,
                      *, initial_camera=None, max_nfev=60, fit_mask=None,
                      reference_installation_id=None, current_installation_id=None):
    """Solve current R/f with fixed C, optionally holding all ice out of fitting.

    Input points are actual corresponding raw sample pixels. Caller identifies
    static support and camera installation; feature matching alone does not
    establish physical identity. `fit_mask=False` points never enter residuals.
    Three orientation coordinates include a diagnostic roll freedom; this is
    deliberately reported rather than silently claiming a calibrated PT head.
    """
    if reference_installation_id is not None and current_installation_id is not None and reference_installation_id != current_installation_id:
        raise ValueError('Cannot transfer background rays between installations')
    reference_xy = np.asarray(reference_xy, float).reshape(-1, 2)
    current_xy = np.asarray(current_xy, float).reshape(-1, 2)
    if reference_xy.shape != current_xy.shape:
        raise ValueError('Corresponding arrays must have the same length')
    labels = np.asarray(surface_labels if surface_labels is not None else ['static_scene']*len(reference_xy))
    if len(labels) != len(reference_xy):
        raise ValueError('One surface label is required per correspondence')
    base = initial_camera or reference_camera
    if not np.allclose(base.centre, reference_camera.centre, atol=1e-7, rtol=0):
        raise ValueError('Initial and reference camera centres must be shared')
    rays, valid = world_rays(reference_camera, reference_xy)
    valid &= np.isfinite(current_xy).all(axis=1)
    training = valid.copy() if fit_mask is None else valid & np.asarray(fit_mask, bool)
    if np.count_nonzero(training) < 4:
        return None, {'success': False, 'reason': 'fewer_than_four_static_correspondences'}

    def unpack(x):
        return replace(base, rotation=cv2.Rodrigues(x[:3])[0] @ base.rotation,
                       focal=base.focal*np.exp(x[3]))

    def residual(x):
        prediction, good = project_camera(unpack(x), rays[training], rays=True)
        delta = prediction-current_xy[training]
        delta[~good] = 1e4
        return delta.ravel()

    # Initial H only seeds the numerical search. Final transfer must lie in
    # K_current R_current R_reference^-1 K_reference^-1, four parameters.
    x0 = np.zeros(4)
    h, _ = cv2.findHomography(reference_xy[training], current_xy[training], cv2.RANSAC, 4.)
    if h is not None and len(reference_xy[training]) >= 6:
        best = np.inf
        for ratio in np.geomspace(.35, 3., 35):
            k = base.K.copy(); k[0,0] = k[1,1] = base.focal*ratio
            q = np.linalg.solve(k, h @ reference_camera.K)
            if np.linalg.det(q) < 0: q = -q
            u, _, vt = np.linalg.svd(q)
            r = u @ np.diag([1.,1.,np.linalg.det(u@vt)]) @ vt
            total = r @ reference_camera.rotation @ base.rotation.T
            candidate = np.r_[cv2.Rodrigues(total)[0].ravel(), np.log(ratio)]
            if np.max(np.abs(candidate[:3])) <= 1.5:
                cost = np.median(np.linalg.norm(residual(candidate).reshape(-1,2),axis=1))
                if cost < best: x0, best = candidate, cost
    fit = least_squares(residual, x0, bounds=([-1.6]*3+[-1.5], [1.6]*3+[1.5]),
                        loss='soft_l1', f_scale=2., max_nfev=max_nfev,
                        x_scale='jac', ftol=1e-8, xtol=1e-8, gtol=1e-7)
    camera = unpack(fit.x)
    prediction, good = project_camera(camera, rays, rays=True)
    errors = np.linalg.norm(prediction-current_xy, axis=1)
    errors[~good | ~valid] = np.nan
    surfaces = {str(label): {**_summary(errors[labels==label]),
                 'fit_count': int(np.count_nonzero(training & (labels==label))),
                 'withheld_count': int(np.count_nonzero(~training & valid & (labels==label)))} for label in np.unique(labels)}
    return camera, {'success': bool(fit.success), 'nfev': int(fit.nfev),
        'message': str(fit.message), 'parameter_step': fit.x.tolist(),
        'training': _summary(errors[training]), 'withheld': _summary(errors[~training & valid]),
        'surfaces': surfaces, 'errors_px': [float(e) if np.isfinite(e) else None for e in errors],
        'fit_mask': training.tolist(), 'data_jacobian_rank': int(np.linalg.matrix_rank(fit.jac)),
        'fixed_center': True, 'center_adjustment': [0.,0.,0.],
        'orientation_parameter_count': 3, 'roll_is_allowed_diagnostic_freedom': True,
        'board_points_used_as_ground': False, 'independent_accuracy_validated': False,
        'H_current_from_reference': rotation_zoom_h(reference_camera, camera).tolist() if not camera.radial_ray_k1 else None}


def sift_static_matches(image_a, image_b, mask_a=None, mask_b=None, *, max_features=5000, ratio=.76):
    """Reciprocal SIFT matches with support restricted to source-defined masks."""
    sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=.007, edgeThreshold=14)
    def describe(image, mask):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
        if mask is None: mask=np.full(gray.shape,255,np.uint8)
        isolated = np.where(mask>0, gray, 0).astype(np.uint8)
        keys, descriptors = sift.detectAndCompute(isolated, mask)
        if descriptors is None: return np.empty((0,2)), np.empty((0,128),np.float32)
        distance=cv2.distanceTransform(mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        keep=[i for i,k in enumerate(keys) if distance[min(round(k.pt[1]),gray.shape[0]-1),min(round(k.pt[0]),gray.shape[1]-1)] >= max(3.,2.*k.size)]
        return np.asarray([keys[i].pt for i in keep],float).reshape(-1,2), descriptors[keep]
    a,da=describe(image_a,mask_a);b,db=describe(image_b,mask_b)
    if min(len(a),len(b)) < 2: return np.empty((0,2)),np.empty((0,2)),{'keypoint_counts':[len(a),len(b)],'match_count':0}
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    backward={m.queryIdx:m.trainIdx for pair in matcher.knnMatch(db,da,k=2) if len(pair)==2 for m,n in [pair] if m.distance<ratio*n.distance}
    matches=[m for pair in matcher.knnMatch(da,db,k=2) if len(pair)==2 for m,n in [pair] if m.distance<ratio*n.distance and backward.get(m.trainIdx)==m.queryIdx]
    aa=np.asarray([a[m.queryIdx] for m in matches],float).reshape(-1,2)
    bb=np.asarray([b[m.trainIdx] for m in matches],float).reshape(-1,2)
    if len(aa)>=4:
        _,inliers=cv2.findHomography(aa,bb,cv2.USAC_MAGSAC,3.)
        keep=inliers.ravel().astype(bool) if inliers is not None else np.zeros(len(aa),bool)
    else: keep=np.ones(len(aa),bool)
    return aa[keep],bb[keep],{'keypoint_counts':[len(a),len(b)],'reciprocal_matches':len(matches),
                             'match_count':int(np.count_nonzero(keep)), 'free_H_used_only_for_match_outlier_rejection':True}
