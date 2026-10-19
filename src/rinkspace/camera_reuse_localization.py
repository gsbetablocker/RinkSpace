"""Image-evidence retrieval and fixed-installation camera reuse.

Every retrieval estimates a fresh orientation and focal length, including a
return to the same physical camera. Static board pixels are rays, never ice
points. An appearance match proposes an installation; spatially distributed
point checks test that proposal. All outputs remain conditional candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from .geometry import project
from .multisurface_camera import Camera, world_rays, project_camera
from .local_registration import ice_mask


@dataclass(frozen=True)
class ReuseConfig:
    min_matches: int = 14
    ransac_px: float = 3.
    max_fit_median_px: float = 2.5
    max_check_p95_px: float = 7.
    retrieve_every_s: float = .5
    keyframe_every_s: float = .4
    max_anchor_age_s: float = 8.
    max_lk_gap_s: float = .1
    require_ice_for_retrieval: bool = True


def polygon_mask(shape, polygons):
    mask = np.zeros(shape, np.uint8)
    for p in polygons:
        a = np.asarray(p, float)
        if a.ndim == 2 and a.shape[1] == 2 and len(a) >= 3:
            cv2.fillPoly(mask, [np.rint(a).astype(np.int32)], 255)
    return mask


def observation_masks(image, ice_polygons=(), foreground_polygons=(), foreground_mask=None):
    h, w = image.shape[:2]
    ice = (polygon_mask((h, w), ice_polygons) if ice_polygons
           else ice_mask(image)['ice'])
    valid = np.full((h, w), 255, np.uint8)
    valid[:max(20, h//14)] = 0
    valid[-max(20, h//14):] = 0
    # Exclusion masks are observations supplied by the caller. A white image
    # region alone is not evidence that a feature is on the ice.
    people = polygon_mask((h, w), foreground_polygons) if foreground_mask is None else np.asarray(foreground_mask, np.uint8)
    if people.shape != (h, w):
        raise ValueError('Person mask and image sample domains differ')
    valid[cv2.dilate(people, np.ones((9, 9), np.uint8)) > 0] = 0
    ice = cv2.bitwise_and(ice, valid)
    return valid, ice


def descriptors(image, valid):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keys, desc = cv2.SIFT_create(nfeatures=5000, contrastThreshold=.007,
                                edgeThreshold=14).detectAndCompute(gray, valid)
    xy = np.asarray([k.pt for k in keys], np.float32).reshape(-1, 2)
    if desc is not None:
        desc = np.sqrt(desc/(desc.sum(axis=1, keepdims=True)+1e-12))
    return xy, desc


def match_features(reference, current, config):
    a, da = reference
    b, db = current
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return None
    matcher = cv2.BFMatcher()
    pairs = matcher.knnMatch(da, db, k=2)
    backward = {pair[0].queryIdx:pair[0].trainIdx for pair in matcher.knnMatch(db,da,k=2)
                if len(pair)==2 and pair[0].distance < .76*pair[1].distance}
    candidates = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .76*pair[1].distance
                  and backward.get(pair[0].trainIdx)==pair[0].queryIdx]
    # One current feature cannot count as several independent matches.
    unique, reference_positions = {}, set()
    for m in sorted(candidates, key=lambda x: x.distance):
        target_key=tuple(np.rint(b[m.trainIdx]/3).astype(int))
        source_key=tuple(np.rint(a[m.queryIdx]/3).astype(int))
        if source_key not in reference_positions and target_key not in unique:
            unique[target_key]=m
            reference_positions.add(source_key)
    candidates = list(unique.values())
    if len(candidates) < config.min_matches:
        return None
    left = a[[m.queryIdx for m in candidates]]
    right = b[[m.trainIdx for m in candidates]]
    H, mask = cv2.findHomography(left, right, cv2.RANSAC, config.ransac_px,
                                 maxIters=2500, confidence=.999)
    if H is None or mask is None or mask.sum() < config.min_matches:
        return None
    keep = mask.ravel().astype(bool)
    left, right = left[keep], right[keep]
    if np.ptp(right[:, 0]) < 90 or np.ptp(right[:, 1]) < 22:
        return None
    return left, right, H


def camera_projection_chain(camera):
    return dict(H_reference_undistorted_from_model=camera.H.tolist(),
                reference_radial_distortion=camera.distortion,
                H_current_sample_from_reference_raw=np.eye(3).tolist(),
                reference_visible_branch_sign=1, raw_transfer_visible_branch_sign=1,
                reference_sample_size=list(camera.size), current_sample_size=list(camera.size),
                relative_geometry='fixed_installation_rotation_zoom_candidate')


def relative_camera_error(reference, camera, left, right):
    rays, good = world_rays(reference, left)
    prediction, front = project_camera(camera, rays, rays=True)
    good &= front & np.isfinite(prediction).all(axis=1)
    error = np.full(len(left), 1e6)
    error[good] = np.linalg.norm(prediction[good]-right[good], axis=1)
    return error


def unused_ice_check(ref, image, camera, current_valid=None):
    """Test a background proposal against unused textured ice patches.

    Forward/backward LK and raw patch contrast reject white-on-white matches.
    This is a cross-surface diagnostic, not independent rink ground truth.
    """
    a_gray=cv2.cvtColor(ref['image'],cv2.COLOR_BGR2GRAY)
    current_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Compare actual pixels in a common angular sampling. Raw image patches
    # have different scales after zoom, which would reject valid camera reuse.
    # This warp predicts where to read; LK below still measures an independent
    # image displacement and no synthetic texture enters the observation.
    if '_rays_grid' not in ref:
        yy,xx=np.indices(a_gray.shape)
        grid=np.c_[xx.ravel(),yy.ravel()].astype(float)
        ref['_rays_grid']=world_rays(ref['camera'],grid)
    grid_rays,grid_valid=ref['_rays_grid']
    mapped,front=project_camera(camera,grid_rays,rays=True)
    mapped[~(grid_valid&front)]=-1000
    maps=mapped.reshape(*a_gray.shape,2).astype(np.float32)
    b_gray=cv2.remap(current_gray,maps[:,:,0],maps[:,:,1],cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)
    support=cv2.remap(np.full(current_gray.shape,255,np.uint8) if current_valid is None else current_valid,
                     maps[:,:,0],maps[:,:,1],cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
    safe=cv2.erode(ref['ice_mask'],np.ones((15,15),np.uint8))
    safe=cv2.bitwise_and(safe,cv2.erode(support,np.ones((19,19),np.uint8)))
    points=cv2.goodFeaturesToTrack(a_gray,300,.002,9,mask=safe,blockSize=7)
    if points is None:
        return {'count':0,'passed':False,'reason':'no_textured_reference_ice'}
    points=points.reshape(-1,2)
    points=points.astype(np.float32)
    if len(points)<4:
        return {'count':int(len(points)),'passed':False,'reason':'no_overlap_with_reference_ice'}
    measured,ok,_=cv2.calcOpticalFlowPyrLK(a_gray,b_gray,points[:,None],points[:,None].copy(),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,winSize=(21,21),maxLevel=2)
    if measured is None or ok is None:
        return {'count':0,'passed':False,'reason':'ice_flow_missing'}
    back,okback,_=cv2.calcOpticalFlowPyrLK(b_gray,a_gray,measured,points[:,None].copy(),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,winSize=(21,21),maxLevel=2)
    if back is None or okback is None:
        return {'count':0,'passed':False,'reason':'ice_reverse_flow_missing'}
    measured,back=measured.reshape(-1,2),back.reshape(-1,2)
    good=ok.ravel().astype(bool)&okback.ravel().astype(bool)&(np.linalg.norm(back-points,axis=1)<.7)
    good &= (measured[:,0]>12)&(measured[:,0]<image.shape[1]-12)&(measured[:,1]>12)&(measured[:,1]<image.shape[0]-12)
    # Photometric support is measured after the candidate's view normalization.
    # Constant white ice is never an accepted correspondence.
    for j in np.flatnonzero(good):
        pa=cv2.getRectSubPix(a_gray,(15,15),tuple(points[j])).astype(float)
        pb=cv2.getRectSubPix(b_gray,(15,15),tuple(measured[j])).astype(float)
        sa,sb=pa.std(),pb.std()
        corr=float(np.mean((pa-pa.mean())*(pb-pb.mean()))/max(sa*sb,1e-9))
        if sa<3 or sb<3 or corr<.8: good[j]=False
    rays,_=world_rays(ref['camera'],points[good])
    predicted,_=project_camera(camera,rays,rays=True)
    observed_rays,_=world_rays(ref['camera'],measured[good])
    observed,_=project_camera(camera,observed_rays,rays=True)
    error=np.linalg.norm(observed-predicted,axis=1)
    spread=np.ptp(observed,axis=0) if good.any() else np.zeros(2)
    p95=float(np.percentile(error,95)) if len(error) else None
    return {'count':int(good.sum()),'p95_px':p95,'span_px':spread.tolist(),
            'passed':bool(len(error)>=4 and spread[0]>70 and p95<7),
            'reference_xy':points[good].tolist(),'measured_xy':observed.tolist(),
            'measured_rectified_reference_xy':measured[good].tolist(),
            'predicted_xy':predicted.tolist(),
            'role':'unused_ice_patch_transfer_check_not_independent_rink_geometry'}


def fit_evidence(reference_camera, left, right, reference_ice, config):
    from .fixed_camera import fit_rotation_zoom
    left, right = np.asarray(left), np.asarray(right)
    coords = np.rint(left).astype(int)
    coords[:, 0] = np.clip(coords[:, 0], 0, reference_ice.shape[1]-1)
    coords[:, 1] = np.clip(coords[:, 1], 0, reference_ice.shape[0]-1)
    ice = reference_ice[coords[:, 1], coords[:, 0]] > 0
    # Prefer all ice matches as an unused surface check when background is
    # sufficient. Otherwise interleave fixed spatial indices as a point check;
    # neither case is independent rink geometry or a new physical landmark.
    if np.count_nonzero(~ice) >= config.min_matches and np.count_nonzero(ice) >= 4:
        fit = ~ice
        check = ice
        check_kind = 'ice_points_excluded_from_rotation_zoom_fit'
    else:
        order = np.lexsort((left[:, 1], left[:, 0]))
        check = np.zeros(len(left), bool)
        check[order[::5]] = True
        fit = ~check
        check_kind = 'spatially_interleaved_points_not_independent_physical_groups'
    if np.count_nonzero(fit) < 8:
        return None, {'reason': 'insufficient_fit_points_after_withholding'}
    camera, diag = fit_rotation_zoom(reference_camera, left[fit], right[fit], max_nfev=45)
    if camera is None:
        return None, {'reason': 'fixed_centre_fit_failed', 'solver': diag}
    error = relative_camera_error(reference_camera, camera, left, right)
    fit_median = float(np.median(error[fit]))
    check_p95 = float(np.percentile(error[check], 95)) if check.any() else None
    accepted = bool(fit_median <= config.max_fit_median_px and check_p95 is not None
                    and check_p95 <= config.max_check_p95_px)
    report = dict(reason='conditional_image_check_passed' if accepted else 'rotation_zoom_image_check_failed',
                  fit_point_count=int(fit.sum()), check_point_count=int(check.sum()),
                  ice_point_count=int(ice.sum()), check_kind=check_kind,
                  fit_median_px=fit_median, check_p95_px=check_p95,
                  independent_geometry_validation=False,
                  fit_mask=fit.tolist(), check_mask=check.tolist(), errors_px=error.tolist(),
                  solver_success=bool(diag.get('success', False)))
    return (camera if accepted else None), report


class CameraReuseLocalizer:
    def __init__(self, references, *, config=None):
        self.config = config or ReuseConfig()
        self.references = references
        self.active = None
        self.last_time = None
        self.last_retrieval = -float('inf')
        self.last_failures = []
        self.sequence = 0

    def reset(self):
        """A cut discards orientation/zoom, even if the next camera ID repeats."""
        self.active = None
        self.last_time = None
        self.last_retrieval = -float('inf')
        self.sequence += 1

    def retrieve(self, image, valid, ice, source_time):
        features = descriptors(image, valid)
        options, failures = [], []
        for ref in self.references:
            matched = match_features(ref['descriptors'], features, self.config)
            if matched is None:
                continue
            left, right, _ = matched
            camera, diag = fit_evidence(ref['camera'], left, right, ref['ice_mask'], self.config)
            if camera is None:
                failures.append({'reference_id': ref['id'], **diag})
                continue
            if self.config.require_ice_for_retrieval:
                if diag['check_kind']=='ice_points_excluded_from_rotation_zoom_fit':
                    ice_check={'passed':True,'count':diag['check_point_count'],
                               'p95_px':diag['check_p95_px'],'role':'unused_SIFT_ice_correspondences'}
                else:
                    ice_check=unused_ice_check(ref,image,camera,valid)
                diag['unused_ice_check']=ice_check
                if not ice_check['passed']:
                    failures.append({'reference_id':ref['id'],'reason':'camera_identity_has_no_compatible_unused_ice_check',
                                     'unused_ice_check':ice_check})
                    continue
            options.append((len(left)/(1+diag['check_p95_px']), ref, camera, left, right, diag))
        self.last_failures = sorted(failures, key=lambda r: r.get('check_p95_px', 1e9))[:3]
        if not options:
            return None
        options.sort(key=lambda x: x[0], reverse=True)
        best = options[0]
        # Different installations must agree in model space before a weak
        # appearance ranking is allowed to decide between them.
        if len(options)>1:
            other = options[1]
            if other[0] > best[0]*.8:
                probe = np.asarray([[200, 350], [480, 400], [750, 350]], float)
                a = project(np.linalg.inv(best[2].H), probe)
                b = project(np.linalg.inv(other[2].H), probe)
                if np.max(np.linalg.norm(a-b, axis=1)) > 5:
                    self.last_failures.append({'reason':'ambiguous_installation_retrieval'})
                    return None
        _, ref, camera, left, right, diag = best
        return dict(camera=camera, reference=ref, left=left, right=right,
                    evidence=diag, anchor_time=source_time, anchor_id=ref['id'],
                    route='reference_library_retrieval')

    def track(self, image, valid, source_time):
        ref = self.active
        if ref is None or source_time-ref['anchor_time'] > self.config.max_anchor_age_s:
            return None
        if self.last_time is not None and source_time-self.last_time > self.config.max_lk_gap_s:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        a = cv2.goodFeaturesToTrack(ref['gray'], maxCorners=650, qualityLevel=.008,
                                   minDistance=7, mask=ref['valid'], blockSize=7)
        if a is None or len(a) < self.config.min_matches:
            return None
        b, ok, _ = cv2.calcOpticalFlowPyrLK(ref['gray'], gray, a, None, winSize=(25, 25), maxLevel=4)
        if b is None:
            return None
        aa, back, _ = cv2.calcOpticalFlowPyrLK(gray, ref['gray'], b, None, winSize=(25, 25), maxLevel=4)
        if aa is None:
            return None
        a, b, aa = a.reshape(-1, 2), b.reshape(-1, 2), aa.reshape(-1, 2)
        good = ok.ravel().astype(bool) & back.ravel().astype(bool) & (np.linalg.norm(a-aa, axis=1)<.8)
        xy = np.rint(b).astype(int)
        inside = (xy[:, 0]>=0)&(xy[:, 0]<valid.shape[1])&(xy[:, 1]>=0)&(xy[:, 1]<valid.shape[0])
        good &= inside
        selected = np.flatnonzero(good)
        good[selected] &= valid[xy[selected, 1], xy[selected, 0]] > 0
        left, right = a[good], b[good]
        if len(left) < self.config.min_matches:
            return None
        H, m = cv2.findHomography(left, right, cv2.RANSAC, self.config.ransac_px)
        if H is None or m is None or m.sum()<self.config.min_matches:
            return None
        left, right = left[m.ravel()>0], right[m.ravel()>0]
        camera, diag = fit_evidence(ref['camera'], left, right, ref['ice_mask'], self.config)
        if camera is None:
            return None
        return dict(camera=camera, reference=ref['reference'], left=left, right=right,
                    evidence=diag, anchor_time=ref['anchor_time'], anchor_id=ref['anchor_id'],
                    route='observed_LK_rotation_zoom', parent_time=ref['key_time'])

    def process(self, image, source_time, *, ice_polygons=(), foreground_polygons=(), foreground_mask=None, cut=False):
        if cut:
            self.reset()
        valid, ice = observation_masks(image, ice_polygons, foreground_polygons, foreground_mask)
        result = None
        if self.active is None or source_time-self.last_retrieval >= self.config.retrieve_every_s:
            result = self.retrieve(image, valid, ice, source_time)
            self.last_retrieval = source_time
        if result is None:
            result = self.track(image, valid, source_time)
        self.last_time = source_time
        if result is not None and (self.active is None or result['route']=='reference_library_retrieval'
                                   or source_time-self.active['key_time']>=self.config.keyframe_every_s):
            self.active = {**result, 'gray': cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                           'valid': valid, 'ice_mask': ice, 'key_time': source_time}
        return result
