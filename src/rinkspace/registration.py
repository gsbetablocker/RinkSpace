"""CPU SIFT baseline for masked planar registration, deliberately candidate-only.

Automatic split-match residuals are diagnostics, not independent physical-ground
truth: repeated logos and reflections can give low errors while being wrong.
"""
from __future__ import annotations
from typing import Any
import cv2
import numpy as np
from .geometry import fit_homography, project


def masked_pair(reference: np.ndarray, current: np.ndarray, reference_mask: np.ndarray,
                current_mask: np.ndarray, ratio: float = 0.72, seed: int = 7) -> dict[str, Any]:
    if reference.shape[:2] != reference_mask.shape or current.shape[:2] != current_mask.shape:
        raise ValueError("Image/mask resolution mismatch")
    detector = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.015, edgeThreshold=12)
    k_ref, d_ref = detector.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), reference_mask)
    k_cur, d_cur = detector.detectAndCompute(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), current_mask)
    base = {"status": "unavailable", "matrix_direction": "H_reference_from_current",
            "geometry_level": "projective", "independent_validation": False,
            "map_usable_for_metric_output": False}
    if d_ref is None or d_cur is None or min(len(d_ref), len(d_cur)) < 2:
        return {**base, "reason": "insufficient_masked_features"}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(d_cur, d_ref, k=2)
    reverse = matcher.knnMatch(d_ref, d_cur, k=2)
    back = {m.queryIdx: m.trainIdx for pair in reverse if len(pair) == 2
            for m, n in [pair] if m.distance < ratio * n.distance}
    good = [m for pair in forward if len(pair) == 2 for m, n in [pair]
            if m.distance < ratio * n.distance and back.get(m.trainIdx) == m.queryIdx]
    if len(good) < 16:
        return {**base, "reason": "too_few_mutual_matches", "matches": len(good)}
    a = np.array([k_cur[m.queryIdx].pt for m in good], dtype=float)
    b = np.array([k_ref[m.trainIdx].pt for m in good], dtype=float)
    order = np.random.default_rng(seed).permutation(len(a))
    n_check = max(4, len(order) // 4)
    check, train = order[:n_check], order[n_check:]
    try:
        h, inliers = fit_homography(a[train], b[train])
        err = np.linalg.norm(project(h, a[check]) - b[check], axis=1)
    except ValueError as exc:
        return {**base, "reason": str(exc), "matches": len(good)}
    hull = cv2.convexHull(a[train][inliers].astype(np.float32))
    return {**base, "status": "candidate", "reason": "not_independently_validated",
            "H_reference_from_current": h.tolist(), "matches": len(good),
            "train_inliers": int(np.sum(inliers)), "train_count": len(train),
            "diagnostic_check_count": len(check), "diagnostic_median_reference_px": float(np.median(err)),
            "diagnostic_p95_reference_px": float(np.percentile(err, 95)),
            "fit_support_current_xy": hull.reshape(-1, 2).tolist(),
            "inlier_hull_fraction_of_full_frame": float(cv2.contourArea(hull) / (current.shape[0] * current.shape[1])),
            "warning": "Automatic match holdout; not physically independent landmark validation."}


def adjacent_lk(previous: np.ndarray, current: np.ndarray, previous_mask: np.ndarray,
                current_mask: np.ndarray, seed: int = 7) -> dict[str, Any]:
    """Short-baseline forward/backward LK; output maps current into previous.

    Uses a spatial-cell diagnostic split. This is still not independent ground
    truth, and composing successive outputs accumulates drift.
    """
    if previous.shape != current.shape or previous.shape[:2] != previous_mask.shape or current.shape[:2] != current_mask.shape:
        raise ValueError("LK images and masks must share dimensions")
    a_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    base = {"status": "unavailable", "matrix_direction": "H_reference_from_current",
            "reference_is_previous_sample": True, "pair_algorithm": "shi-tomasi+forward-backward-lk+ransac",
            "geometry_level": "projective", "independent_validation": False,
            "map_usable_for_metric_output": False}
    x = cv2.goodFeaturesToTrack(a_gray, 1200, .003, 8, mask=previous_mask, blockSize=7)
    if x is None or len(x) < 20:
        return {**base, "reason": "insufficient_masked_features"}
    y, st, _ = cv2.calcOpticalFlowPyrLK(a_gray, b_gray, x, None, winSize=(31,31), maxLevel=4)
    if y is None:
        return {**base, "reason": "forward_flow_failed"}
    z, sb, _ = cv2.calcOpticalFlowPyrLK(b_gray, a_gray, y, None, winSize=(31,31), maxLevel=4)
    if z is None:
        return {**base, "reason": "backward_flow_failed"}
    a, b, c = x[:,0], y[:,0], z[:,0]
    height, width = current.shape[:2]
    finite = np.all(np.isfinite(b), axis=1) & np.all(np.isfinite(c), axis=1)
    safe = np.nan_to_num(b, nan=-10000, posinf=-10000, neginf=-10000)
    ij = np.rint(safe).astype(np.int64)
    ij[:,0] = np.clip(ij[:,0], 0, width-1); ij[:,1] = np.clip(ij[:,1], 0, height-1)
    ok = (finite & (st[:,0]>0) & (sb[:,0]>0) & (np.linalg.norm(a-c,axis=1)<1)
          & (b[:,0]>=0) & (b[:,0]<width) & (b[:,1]>=0) & (b[:,1]<height)
          & (current_mask[ij[:,1],ij[:,0]]>0))
    a, b = a[ok], b[ok]
    if len(a) < 24:
        return {**base, "reason": "too_few_consistent_tracks", "matches": len(a)}
    cells = np.floor(a[:,0]/80).astype(int) + np.floor(a[:,1]/80).astype(int) * 1000
    unique = np.unique(cells)
    chosen = np.random.default_rng(seed).permutation(unique)[:max(1,len(unique)//4)]
    check = np.isin(cells,chosen)
    train = ~check
    if np.sum(train) < 12 or np.sum(check) < 4:
        return {**base, "reason": "insufficient_spatial_diagnostic_split", "matches": len(a)}
    try:
        h, inliers = fit_homography(b[train],a[train],2.0)
        err = np.linalg.norm(project(h,b[check])-a[check],axis=1)
    except ValueError as exc:
        return {**base, "reason": str(exc), "matches": len(a)}
    hull = cv2.convexHull(b[train][inliers].astype(np.float32))
    return {**base, "status": "candidate", "reason": "not_independently_validated",
            "H_reference_from_current": h.tolist(), "matches": len(a), "train_count": int(train.sum()),
            "train_inliers": int(inliers.sum()), "diagnostic_check_count": int(check.sum()),
            "diagnostic_split": "spatial_cells_not_physical_landmark_ground_truth",
            "diagnostic_median_reference_px": float(np.median(err)),
            "diagnostic_p95_reference_px": float(np.percentile(err,95)),
            "fit_support_current_xy": hull.reshape(-1,2).tolist(),
            "inlier_hull_fraction_of_full_frame": float(cv2.contourArea(hull)/(width*height)),
            "warning": "Pairwise residual only. No global drift or metric-scale validation."}
