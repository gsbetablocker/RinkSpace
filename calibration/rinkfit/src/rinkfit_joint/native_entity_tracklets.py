"""Source-only native tracklets of previously qualified physical ad corners.

LK is a navigation proposal. Every emitted mean is re-read at the current
exposure as the intersection of two actual material edges with the existing
production corner operator. No model projection or camera enters this module.
Identity inheritance is explicit; this reader cannot qualify an unknown anchor.
"""
from __future__ import annotations

import numpy as np
import cv2
from .entity_reacquire import remeasure_reacquired_corner
from .entity_corners import remeasure_corner, CORNER_SOURCE_OPERATOR
from .source_worker import _channels

VERSION = 'native_fixed_entity_material_tracklet_v1'
# These are the existing R3 source-navigation checks, applied only to actual RGB.
NAVIGATION = dict(reciprocal_max_px=1., patch_ncc_min=.75,
    lk_window_px=31, lk_pyramid_levels=3, patch_radius_px=16,
    navigation_is_measurement=False, world_model_used=False)


def _gray(rgb):
    return cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2GRAY)


def corner_inward_tangents(role):
    if role not in {'left_top', 'right_top', 'left_bottom', 'right_bottom'}:
        raise ValueError('An established finite physical corner role is required')
    return [[1. if role.startswith('left') else -1., 0.],
            [0., 1. if role.endswith('top') else -1.]]


def _patch(rgb, uv):
    r = NAVIGATION['patch_radius_px']
    q = np.asarray(uv, float)
    h, w = rgb.shape[:2]
    if not (r+2 <= q[0] < w-r-2 and r+2 <= q[1] < h-r-2):
        return None
    return cv2.getRectSubPix(_gray(rgb).astype(np.float32), (2*r+1, 2*r+1), tuple(q))


def _ncc(a, b):
    a = a.astype(float).ravel(); b = b.astype(float).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/d) if d > 1e-9 else -1.


def read_material_corner(rgb, mask, seed_uv, edge_tangents, *, channels=None):
    """Two-edge reacquisition followed by the unchanged production operator."""
    rgb = np.asarray(rgb, np.uint8); mask = np.asarray(mask, bool)
    q = np.asarray(seed_uv, float)
    if rgb.shape != (1080,1920,3) or mask.shape != (1080,1920):
        raise ValueError('Actual native 1920x1080 RGB and independent current mask required')
    u,v = np.rint(q).astype(int)
    if not (50 <= u < 1870 and 50 <= v < 1030):
        return dict(valid=False, reason='current_corner_patch_outside_native_image')
    origin = np.array([u-48,v-48]); patch = rgb[v-48:v+49,u-48:u+49]
    pmask = mask[v-48:v+49,u-48:u+49]
    if pmask.any():
        return dict(valid=False, reason='current_source_corner_support_patch_masked')
    reacquired = remeasure_reacquired_corner(patch, dict(corner_uv_in_patch=(q-origin).tolist(),
        edge_tangents=edge_tangents), pmask)
    if not reacquired.get('valid'):
        return dict(valid=False, reason=reacquired['reason'], reacquisition=reacquired)
    nav = np.asarray(reacquired['current_intersection_uv'])+origin
    directions = [f['tangent'] for f in reacquired['edge_fits']]
    actual = remeasure_corner(_channels(rgb) if channels is None else channels, nav, directions, mask)
    if not actual.get('valid'):
        return dict(valid=False, reason=actual['reason'], reacquisition=reacquired, source_operator=actual)
    return dict(valid=True, reason='two_current_material_edges_intersection',
        current_uv=actual['current_intersection_uv'], edge_tangents=directions,
        operator_version=CORNER_SOURCE_OPERATOR['version'], source_operator=actual,
        reacquisition=reacquired, native_patch_origin_uv=origin.tolist(),
        pixel_support_status='current_native_two_material_edges',
        source_mean_pixels_from_model=False, numeric_statistics_ready=False,
        statistical_objective_eligible=False, source_identity_inherited_not_established=True)


def track_material_corner(previous_rgb, current_rgb, previous_uv, edge_tangents, mask, *, current_channels=None):
    """One same-shot native step; failure remains an explicit terminal reason."""
    old = _gray(previous_rgb); now = _gray(current_rgb)
    p = np.asarray(previous_uv,np.float32).reshape(1,1,2)
    params = dict(winSize=(31,31), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001))
    q, status, error = cv2.calcOpticalFlowPyrLK(old,now,p,None,**params)
    if q is None or not status[0,0]:
        return dict(valid=False, reason='current_native_LK_navigation_failed')
    back, status2, error2 = cv2.calcOpticalFlowPyrLK(now,old,q,None,**params)
    if back is None or not status2[0,0]:
        return dict(valid=False, reason='reverse_native_LK_navigation_failed')
    reciprocal = float(np.linalg.norm(back-p))
    if reciprocal > NAVIGATION['reciprocal_max_px']:
        return dict(valid=False, reason='actual_native_reciprocal_navigation_above_existing1px',
            reciprocal_error_px=reciprocal)
    seed = q[0,0].astype(float)
    a,b = _patch(previous_rgb,previous_uv),_patch(current_rgb,seed)
    correlation = _ncc(a,b) if a is not None and b is not None else -1.
    if correlation < NAVIGATION['patch_ncc_min']:
        return dict(valid=False, reason='actual_native_patch_navigation_below_existing075',
            patch_ncc=correlation, reciprocal_error_px=reciprocal)
    result = read_material_corner(current_rgb,mask,seed,edge_tangents,channels=current_channels)
    result.update(navigation_seed_uv=seed.tolist(), reciprocal_error_px=reciprocal,
        patch_ncc=correlation, actual_previous_uv=np.asarray(previous_uv).tolist(),
        navigation_configuration=NAVIGATION)
    return result
