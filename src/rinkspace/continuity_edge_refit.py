"""Controlled camera-edge ablation on frozen, already observed ice point pairs.

No image acquisition, masking, model geometry or anchor pose changes here. The
four-DOF similarity and regularized plane are fitted to identical training pairs.
Every fourth pair is reserved for checking, not used to choose regularization.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np

from .geometry import project

ALGORITHM_VERSION = 'frozen-observed-edges-camera-refit-1'


@dataclass(frozen=True)
class EdgeRefitConfig:
    max_holdout_p95_px_at_960: float = 2.
    image_sigma_px_at_960: float = 1.7
    anisotropy_shear_prior_scale: float = .003
    perspective_prior_scale: float = .003
    max_normalized_perspective: float = .025
    max_normalized_condition: float = 50.

    def __post_init__(self):
        if any(value <= 0 for value in asdict(self).values()):
            raise ValueError('All edge refit engineering limits must be positive')


def similarity_fit(current, target):
    """Least-squares orientation-preserving similarity, with no RANSAC redraw."""
    x, y = np.asarray(current, float), np.asarray(target, float)
    if len(x) < 3 or x.shape != y.shape or x.shape[1:] != (2,) or not np.isfinite([x, y]).all():
        raise ValueError('Need finite paired 2D observations')
    a, b = x.mean(axis=0), y.mean(axis=0)
    xx, yy = x-a, y-b
    denominator = float(np.sum(xx*xx))
    if denominator < 1e-9 or np.linalg.matrix_rank(xx) < 2:
        raise ValueError('Image observations lack two-dimensional support')
    u, singular, vt = np.linalg.svd(xx.T @ yy)
    correction = np.eye(2)
    correction[-1, -1] = 1 if np.linalg.det(u @ vt) > 0 else -1
    rotation = u @ correction @ vt
    scale = float(np.sum(singular*np.diag(correction))/denominator)
    if scale <= 0:
        raise ValueError('Reflected or collapsed relative image fit')
    h = np.eye(3)
    h[:2, :2], h[:2, 2] = scale*rotation.T, b-scale*rotation.T @ a
    return h


def regularized_plane_fit(current, target, size, config):
    """Normalized plane least squares with camera-shape priors.

    The penalty suppresses anisotropy, shear and perspective unsupported by the
    image pairs. It does not constrain a pan or isotropic zoom to zero. Iterated
    denominator weights approximate geometric rather than algebraic residual.
    This is fitting an edge; no homography coefficients are interpolated in time.
    """
    width, height = size
    n = np.array([[1/width, 0., -.5], [0., 1/width, -height/(2*width)], [0., 0., 1.]])
    x, y = project(n, current), project(n, target)
    initial = n @ similarity_fit(current, target) @ np.linalg.inv(n)
    design = np.zeros((len(x)*2, 8)); target_vector = y.ravel()
    design[0::2, :3] = np.c_[x, np.ones(len(x))]
    design[1::2, 3:6] = np.c_[x, np.ones(len(x))]
    design[0::2, 6:8] = -y[:, 0, None]*x
    design[1::2, 6:8] = -y[:, 1, None]*x
    shape_prior = np.array([[1., 0, 0, 0, -1, 0, 0, 0],
                            [0, 1., 0, 1., 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1., 0],
                            [0, 0, 0, 0, 0, 0, 0, 1.]])
    sigma_normalized = config.image_sigma_px_at_960/960
    strength = np.sqrt(len(x))*sigma_normalized/np.array([
        config.anisotropy_shear_prior_scale, config.anisotropy_shear_prior_scale,
        config.perspective_prior_scale, config.perspective_prior_scale])
    prior = shape_prior*strength[:, None]
    h = initial
    for _ in range(4):
        depth = np.c_[x, np.ones(len(x))] @ h[2]
        if np.min(depth) <= .1:
            raise ValueError('Plane fit approaches the observed-image horizon')
        weight = np.repeat(1/depth, 2)
        solution = np.linalg.lstsq(np.vstack([design*weight[:, None], prior]),
            np.r_[target_vector*weight, np.zeros(len(prior))], rcond=None)[0]
        h = np.r_[solution, 1.].reshape(3, 3)
    if np.max(np.abs(h[2, :2])) > config.max_normalized_perspective:
        raise ValueError('Relative perspective exceeds declared camera-shape bound')
    return np.linalg.inv(n) @ h @ n


def fit_frozen_edge(current, target, size, strategy, config=None):
    config = config or EdgeRefitConfig()
    if strategy not in {'similarity', 'regularized_plane'}:
        raise ValueError('Unknown camera edge experiment')
    x, y = np.asarray(current, float), np.asarray(target, float)
    base = {'strategy': strategy, 'config': asdict(config),
        'source_pairs_are_frozen_original_plane_inliers': True,
        'independent_geometry_validation': False,
        'holdout_rule': 'ordinal modulo 4 equals 0, fixed before scores', 'accepted': False}
    if x.ndim != 2 or x.shape != y.shape or x.shape[1:] != (2,) or len(x) < 14 or not np.isfinite([x, y]).all():
        return {**base, 'reason': 'insufficient_frozen_actual_pairs'}
    heldout = np.arange(len(x)) % 4 == 0
    try:
        h = similarity_fit(x[~heldout], y[~heldout]) if strategy == 'similarity' else regularized_plane_fit(x[~heldout], y[~heldout], size, config)
        normalizer = np.diag([1/size[0], 1/size[1], 1.])
        condition = float(np.linalg.cond(normalizer @ h @ np.linalg.inv(normalizer)))
        residual = np.linalg.norm(project(h, x)-y, axis=1)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {**base, 'reason': str(exc)}
    p95 = float(np.percentile(residual[heldout], 95))
    accepted = (np.isfinite(residual).all() and p95 <= config.max_holdout_p95_px_at_960*size[0]/960 and
                condition < config.max_normalized_condition)
    return {**base, 'accepted': bool(accepted), 'reason': 'heldout_actual_pairs_within_old_pixel_budget' if accepted else 'heldout_relative_residual_or_condition_exceeded',
        'H_keyframe_from_current': h.tolist(), 'point_count': len(x),
        'holdout_count': int(heldout.sum()), 'holdout_p95_sample_px': p95,
        'all_pair_p95_sample_px': float(np.percentile(residual, 95)),
        'all_pair_median_sample_px': float(np.median(residual)),
        'normalized_condition': condition}


def refit_frames(frames, strategy, config=None):
    """Recompute local gauge along observed parent edges; never create a map."""
    config = config or EdgeRefitConfig()
    result = copy.deepcopy(frames)
    solved, report, root_for = {}, [], {}
    for row in result:
        local = row.get('local_registration', {})
        source_key = (row.get('source_sha256'), row.get('shot_id'), row.get('camera_component_id'))
        current_key = (*source_key, row['sample_index'])
        local['original_H_reference_from_current'] = local.get('H_reference_from_current')
        local['original_H_keyframe_from_current'] = local.get('H_keyframe_from_current')
        row['forward_status_before_edge_refit'] = row['status']
        row['forward_model_bridge_before_edge_refit'] = row.get('model_bridge')
        row.update(status='unavailable', reason='edge_refit_requires_graph_model_anchor',
                   projection_chain=None, H_image_from_model=None, H_model_from_image=None, anchor_id=None)
        if local.get('status') != 'candidate' or local.get('H_reference_from_current') is None:
            continue
        parent = local.get('edge_keyframe_sample_index')
        parent_key = (*source_key, parent)
        fit = None
        if not local.get('reference_frame_only') and parent_key in solved:
            fit = fit_frozen_edge(local.get('fit_current_xy', []), local.get('fit_keyframe_xy', []),
                                  row['sample_size'], strategy, config)
            local['edge_camera_refit'] = fit
            report.append({'sample_index': row['sample_index'], 'source_time_s': row['source_time_s'], **fit})
        if fit is not None and fit['accepted']:
            h = solved[parent_key] @ np.asarray(fit['H_keyframe_from_current'])
            h /= h[2, 2]
            solved[current_key], root_for[current_key] = h, root_for[parent_key]
            local.update(H_reference_from_current=h.tolist(), H_keyframe_from_current=fit['H_keyframe_from_current'],
                fit_reference_xy=project(solved[parent_key], local['fit_keyframe_xy']).tolist(),
                inlier_p95_reference_px=fit['all_pair_p95_sample_px'],
                inlier_median_reference_px=fit['all_pair_median_sample_px'],
                reason='frozen_ice_pairs_camera_edge_refit_candidate')
        else:
            # The existing image establishes only a fresh local gauge. Its
            # failed old edge remains in the archive; no old map bridge survives.
            solved[current_key], root_for[current_key] = np.eye(3), row['sample_index']
            local.update(H_reference_from_current=np.eye(3).tolist(), reference_frame_only=True,
                         reason='unanchored_refit_component_origin')
            local['refit_break_reason'] = fit['reason'] if fit else 'original_local_origin_or_missing_parent'
        row['camera_component_id'] = f"{source_key[2]}:edge-refit-{strategy}:{root_for[current_key]}"
        local['refit_original_camera_component_id'] = source_key[2]
    return {'frames': result, 'edge_comparisons': report, 'strategy': strategy,
            'config': asdict(config), 'algorithm_version': ALGORITHM_VERSION,
            'actual_points_masks_anchors_model_unchanged': True}
