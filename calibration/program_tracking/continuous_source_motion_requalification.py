"""Source-only candidate classification of original hint-0 motion matches.

Original LK coordinates and indices stay immutable. The old pose-guided blue
edge can be advertisement #4's upper edge, not the board rim. Its pixels and
the yellow edge serve only as broad image search windows. Candidate pairs
also need two-view LK/FB, SAM clearance and local print or ice-paint evidence.
This module does not certify a world edge, board height or final plane label.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np


PATCH_SIZE = 11
AD_FACE_TOP_MARGIN_PX = 15
AD_FACE_BOTTOM_MARGIN_PX = 10
ICE_MARGIN_PX = 12

# Statuses align with original raw pair row order, not a filtered/copy UV list.
STATUS = {
    'not_original_hint0': 0,
    'flow_or_inlier_fail': 1,
    'source_roi_missing': 2,
    'sam_person_or_hud': 3,
    'outside_semantic_rois': 4,
    'texture_or_appearance_fail': 5,
    'ad_print_context_missing': 6,
    'ice_paint_or_white_context_missing': 7,
    'candidate_ad_print': 20,
    'candidate_painted_ice': 21,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _mask_at(mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
    x = np.clip(np.rint(xy[:, 0]).astype(int), 0, mask.shape[1] - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(int), 0, mask.shape[0] - 1)
    return mask[y, x]


def _source_roi(edge: dict, pts: int, x: np.ndarray) -> np.ndarray:
    """Interpolate old source colour edges solely as an image search window."""
    stamps = edge['pts']
    i = int(np.searchsorted(stamps, pts))
    if i == len(stamps) or int(stamps[i]) != int(pts):
        return np.full((2, len(x)), np.nan)
    lo, hi = map(int, edge['offsets'][i:i + 2])
    uv, ids = edge['observed_uv'][lo:hi], edge['edge_id'][lo:hi]
    result = []
    for identity in (0, 1):
        points = uv[ids == identity]
        if len(points) < 6:
            return np.full((2, len(x)), np.nan)
        points = points[np.argsort(points[:, 0])]
        xx, inverse = np.unique(points[:, 0], return_inverse=True)
        yy = np.asarray([np.median(points[inverse == j, 1])
                         for j in range(len(xx))])
        result.append(np.interp(x, xx, yy, left=np.nan, right=np.nan))
    return np.asarray(result)


def _patch_checks(frame0: np.ndarray, frame1: np.ndarray,
                  uv0: np.ndarray, uv1: np.ndarray) -> tuple[np.ndarray, ...]:
    """Check two-view texture, appearance, ad chroma and ice paint/context."""
    count = len(uv0)
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2HSV)
    hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    texture = np.zeros(count, bool)
    appearance = np.zeros(count, bool)
    painted = np.zeros(count, bool)
    ad_chroma = np.zeros(count, bool)
    ice_white = np.zeros(count, bool)
    ncc = np.full(count, np.nan, np.float32)
    for i, (a, b) in enumerate(zip(uv0, uv1)):
        if (min(a[0], b[0], a[1], b[1]) < PATCH_SIZE
                or max(a[0], b[0]) >= frame0.shape[1] - PATCH_SIZE
                or max(a[1], b[1]) >= frame0.shape[0] - PATCH_SIZE):
            continue
        p = cv2.getRectSubPix(gray0, (PATCH_SIZE, PATCH_SIZE),
                              tuple(map(float, a))).astype(float)
        q = cv2.getRectSubPix(gray1, (PATCH_SIZE, PATCH_SIZE),
                              tuple(map(float, b))).astype(float)
        p -= p.mean()
        q -= q.mean()
        vp, vq = np.linalg.norm(p), np.linalg.norm(q)
        texture[i] = vp / np.sqrt(p.size) >= 7 and vq / np.sqrt(q.size) >= 7
        if vp > 0 and vq > 0:
            ncc[i] = float(np.sum(p * q) / (vp * vq))
            appearance[i] = ncc[i] >= .55
        paint_views, chroma_views, white_views = [], [], []
        for hsv, point in ((hsv0, a), (hsv1, b)):
            patch = cv2.getRectSubPix(hsv, (PATCH_SIZE, PATCH_SIZE),
                                      tuple(map(float, point)))
            h, s, v = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2]
            red = (((h <= 15) | (h >= 170)) & (s >= 24) & (v >= 65))
            blue = ((h >= 86) & (h <= 137) & (s >= 24) & (v >= 55))
            paint_views.append((np.count_nonzero(red) >= 5,
                                np.count_nonzero(blue) >= 5))
            chroma_views.append(np.count_nonzero((s >= 35) & (v >= 55)) >= 10)
            white_views.append(np.count_nonzero((s <= 40) & (v >= 145)) >= 16)
        painted[i] = any(paint_views[0][j] and paint_views[1][j]
                         for j in (0, 1))
        ad_chroma[i] = all(chroma_views)
        ice_white[i] = all(white_views)
    return texture, appearance, painted, ad_chroma, ice_white, ncc


def qualify_original_pair(*, motion: dict, edge: dict, pair_index: int,
                          frame0: np.ndarray, frame1: np.ndarray,
                          person0: np.ndarray, person1: np.ndarray) -> dict:
    """Classify one pair without changing any original source UV or hint."""
    source_pts, target_pts = map(int, motion['pair_pts'][pair_index])
    if (target_pts - source_pts != 256 or frame0.shape != frame1.shape
            or frame0.shape[:2] != (1080, 1920)):
        raise ValueError('Expected adjacent 1920x1080 source exposures')
    lo, hi = map(int, motion['offsets'][pair_index:pair_index + 2])
    uv0 = np.asarray(motion['xy0'][lo:hi], np.float64)
    uv1 = np.asarray(motion['xy1'][lo:hi], np.float64)
    hint0 = motion['region_hint'][lo:hi] == 0
    good = (motion['homography_inlier'][lo:hi].astype(bool)
            & (motion['fb_error_px'][lo:hi] < 2.5)
            & np.isfinite(uv0).all(axis=1) & np.isfinite(uv1).all(axis=1))
    unknown = hint0 & good
    bounds0 = _source_roi(edge, source_pts, uv0[:, 0])
    bounds1 = _source_roi(edge, target_pts, uv1[:, 0])
    upper0, lower0 = bounds0
    upper1, lower1 = bounds1
    roi_valid = (np.isfinite(bounds0).all(axis=0)
                 & np.isfinite(bounds1).all(axis=0)
                 & (lower0 - upper0 > 24) & (lower1 - upper1 > 24))
    ad_roi = (unknown & roi_valid
              & (uv0[:, 1] > upper0 + AD_FACE_TOP_MARGIN_PX)
              & (uv0[:, 1] < lower0 - AD_FACE_BOTTOM_MARGIN_PX)
              & (uv1[:, 1] > upper1 + AD_FACE_TOP_MARGIN_PX)
              & (uv1[:, 1] < lower1 - AD_FACE_BOTTOM_MARGIN_PX))
    ice_roi = (unknown & roi_valid
               & (uv0[:, 1] > lower0 + ICE_MARGIN_PX)
               & (uv1[:, 1] > lower1 + ICE_MARGIN_PX))
    person = _mask_at(person0, uv0) | _mask_at(person1, uv1)
    ad_roi &= ~person
    ice_roi &= ~person
    candidates = np.flatnonzero(ad_roi | ice_roi)
    texture, appearance, paint, ad_chroma, ice_white, ncc = _patch_checks(
        frame0, frame1, uv0[candidates], uv1[candidates])
    ad_local = ad_roi[candidates] & texture & appearance & ad_chroma
    ice_local = ice_roi[candidates] & texture & appearance & paint & ice_white
    selected_ad = candidates[ad_local]
    selected_ice = candidates[ice_local]
    status = np.full(hi - lo, STATUS['not_original_hint0'], np.uint8)
    status[hint0] = STATUS['flow_or_inlier_fail']
    status[unknown] = STATUS['source_roi_missing']
    status[unknown & roi_valid] = STATUS['outside_semantic_rois']
    status[unknown & roi_valid & person] = STATUS['sam_person_or_hud']
    status[candidates] = STATUS['texture_or_appearance_fail']
    status[candidates[texture & appearance & ad_roi[candidates]]] = STATUS['ad_print_context_missing']
    status[candidates[texture & appearance & ice_roi[candidates]]] = STATUS['ice_paint_or_white_context_missing']
    status[selected_ad] = STATUS['candidate_ad_print']
    status[selected_ice] = STATUS['candidate_painted_ice']
    return {
        'source_pts': source_pts, 'target_pts': target_pts,
        'raw_pair_index': int(pair_index), 'original_offset_range': [lo, hi],
        'candidate_ad_print_local_indices': selected_ad.tolist(),
        'candidate_painted_ice_local_indices': selected_ice.tolist(),
        'status_code_by_local_index': status.tolist(),
        'counts': {
            'raw_uv': hi - lo,
            'good_unknown': int(unknown.sum()),
            'source_roi_supported': int(np.count_nonzero(unknown & roi_valid)),
            'sam_overlap': int(np.count_nonzero(unknown & person)),
            'ad_face_roi_sam_clear': int(ad_roi.sum()),
            'ice_roi_sam_clear': int(ice_roi.sum()),
            'two_view_texture_appearance': int((texture & appearance).sum()),
            'two_view_ad_chroma': int(ad_chroma.sum()),
            'two_view_paint': int(paint.sum()),
            'two_view_ice_white_context': int(ice_white.sum()),
            'candidate_ad_print': len(selected_ad),
            'candidate_painted_ice': len(selected_ice),
        },
        'ncc_median_of_candidates': (float(np.median(ncc[ad_local | ice_local]))
                                     if np.any(ad_local | ice_local) else None),
        'classification_limit': ('Source-only candidates for review. Pose-guided '
                                 'edges are ROI guides, never a certified board '
                                 'rim or world height. No factor is attached.'),
    }
