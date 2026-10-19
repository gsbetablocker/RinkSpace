"""Weak cross-camera identity checks from a few visible support regions.

These regions are neither measured blade contacts nor centre-of-mass points.
They can contradict a camera hypothesis, but cannot solve a camera. Inferred
live-time continuity only produces a soft shortlist; only an explicitly
confirmed programme-time relation permits pruning the active hypothesis list.
Camera matrices, published localization states and the rink model never change.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json

import numpy as np

from .whole_rink import undistort_observed_points

ALGORITHM_VERSION = 'coarse-support-cross-camera-identity-1'
CONFIRMED_TIME = frozenset({'confirmed_programme_continuity', 'confirmed_live_programme_continuity'})
REVIEWED_TIME = frozenset({'visually_reviewed_candidate_not_verified_live_clock'})


@dataclass(frozen=True)
class AthleteContinuityConfig:
    max_observation_gap_s: float = .3
    # A deliberately loose compatibility allowance for this normalized model;
    # it is not a speed, a measured displacement or a statistical error bound.
    allowed_region_separation_model_units: float = 20.
    model_region_padding_units: float = 5.
    source_support_padding_px_at_1920: float = 12.
    expected_model_unit: str = 'normalized_rink_width_100'
    boundary_samples_per_side: int = 17

    def __post_init__(self):
        numeric = [self.max_observation_gap_s, self.allowed_region_separation_model_units,
                   self.model_region_padding_units, self.source_support_padding_px_at_1920]
        if any(not np.isfinite(v) or v <= 0 for v in numeric) or self.boundary_samples_per_side < 3:
            raise ValueError('Coarse support budgets must be finite and positive')


def _matrix(value):
    matrix = np.asarray(value, float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError('invalid_projection_chain_matrix')
    return matrix


def _project(matrix, points):
    homogeneous = np.c_[points, np.ones(len(points))] @ matrix.T
    if not np.isfinite(homogeneous).all() or np.any(abs(homogeneous[:, 2]) < 1e-9):
        raise ValueError('support_region_crosses_projection_horizon')
    if np.any(homogeneous[:, 2] > 0) and np.any(homogeneous[:, 2] < 0):
        raise ValueError('support_region_crosses_projection_horizon')
    return homogeneous[:, :2]/homogeneous[:, 2:]


def source_points_to_model(chain, points):
    """Invert source scaling, raw-reference transfer, lens, then rink pose.

    Non-zero radial distortion is inverted only on its central monotonic
    branch. Never apply an approximate raw homography in place of this inverse.
    Every supplied point must belong to both declared visible branches.
    """
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('invalid_source_support_points')
    source = _matrix(chain['H_source_image_from_sample'])
    transfer = _matrix(chain['H_current_sample_from_reference_raw'])
    reference = _matrix(chain['H_reference_undistorted_from_model'])
    sample_xy = _project(np.linalg.inv(source), points)
    raw_reference = _project(np.linalg.inv(transfer), sample_xy)
    raw_branch = chain.get('raw_transfer_visible_branch_sign', 1)
    reference_branch = chain.get('reference_visible_branch_sign', 1)
    if raw_branch not in (-1, 1) or reference_branch not in (-1, 1):
        raise ValueError('invalid_projection_branch')
    if np.any(raw_branch*(np.c_[raw_reference, np.ones(len(points))] @ transfer[2]) <= 1e-9):
        raise ValueError('support_outside_raw_transfer_visible_branch')
    ideal_reference, valid = undistort_observed_points(raw_reference, chain.get('reference_radial_distortion'))
    if not np.all(valid):
        raise ValueError('support_outside_invertible_radial_domain')
    model_xy = _project(np.linalg.inv(reference), ideal_reference)
    if np.any(reference_branch*(np.c_[model_xy, np.ones(len(points))] @ reference[2]) <= 1e-9):
        raise ValueError('support_outside_model_visible_branch')
    return model_xy


def _time(record):
    return Fraction(record['pts'])*Fraction(record['time_base'])


def _source_key(record):
    return record.get('source_sha256'), _time(record)


def _forbidden_role(row):
    metadata = row.get('source_shot_metadata') or {}
    values = [row.get('source_role'), row.get('role'), metadata.get('role'), metadata.get('label')]
    return any(any(word in str(value).lower() for word in ('replay', 'slow_motion', 'slow-motion',
        'non_programme', 'time_jump', 'insert')) for value in values if value is not None)


def _camera_hypotheses(row):
    hypotheses = row.get('model_anchor_hypotheses') or []
    if row.get('status') == 'ambiguous':
        # An ambiguous prior must retain every interpretation, never just the
        # new raw model_anchor_candidate or the first stored matrix.
        return hypotheses
    if row.get('projection_chain'):
        return [{'reference_id': row.get('anchor_id'), 'projection_chain': row['projection_chain'],
                 'role': 'published_camera_candidate', 'anchor_observation': row.get('uncertainty')}]
    return hypotheses


def _support_error(row, observation):
    if observation.get('coordinate_domain') != 'raw_source_image_pixels':
        return 'support_coordinate_domain_not_raw_source_pixels'
    if observation.get('not_body_bbox_bottom') is not True:
        return 'support_is_not_distinguished_from_body_box'
    if observation.get('observation_type') not in {
            'manual_coarse_visible_blade_support_candidate_regions', 'confirmed_visible_blade_support_regions'}:
        return 'no_reviewed_visible_support_regions'
    if observation.get('state') not in {
            'visible_near_ice_support_candidates_not_exact_contact', 'confirmed_visible_blade_support'}:
        return 'airborne_occluded_or_unconfirmed_support'
    if observation.get('source_id') != row.get('source_id') or observation.get('source_sha256') != row.get('source_sha256'):
        return 'support_source_identity_mismatch'
    if row.get('width') and row.get('height') and observation.get('image_size') != [row['width'], row['height']]:
        return 'support_original_image_size_mismatch'
    if abs(float(_time(observation))-float(observation['source_time_s'])) > 1e-8:
        return 'support_pts_time_mismatch'
    if _forbidden_role(row):
        return 'replay_slow_motion_or_time_jump_disables_athlete_factor'
    return None


def _region(row, observation, hypothesis, config):
    regions = observation.get('support_regions_source_px')
    if not isinstance(regions, list) or not regions:
        raise ValueError('no_visible_support_polygon')
    polygons = [np.asarray(region, float) for region in regions]
    size = np.asarray(observation['image_size'], float)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError('invalid_support_image_size')
    for polygon in polygons:
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
            raise ValueError('invalid_support_polygon')
        if np.any(polygon < 0) or np.any(polygon > size):
            raise ValueError('support_polygon_outside_source_image')
    # The bounding rectangle includes both possible feet and all points between
    # them. It does not assert which blade, if any, is the actual contact.
    points = np.concatenate(polygons)
    error = (row.get('uncertainty') or {}).get('estimated_error_px_at_960')
    if error is None:
        error = (row.get('camera_uncertainty') or {}).get('engineering_range_sample_px', 0.)
        error = float(error)*960/row.get('sample_size', [960])[0]
    lineage_error = (hypothesis.get('anchor_observation') or {}).get('engineering_range_sample_px', 0.)
    error = max(float(error), float(lineage_error)*960/row.get('sample_size', [960])[0])
    if not np.isfinite(error) or error < 0:
        raise ValueError('invalid_camera_engineering_uncertainty')
    padding = config.source_support_padding_px_at_1920*size[0]/1920+error*size[0]/960
    lo, hi = points.min(axis=0)-padding, points.max(axis=0)+padding
    if np.any(lo < 0) or np.any(hi >= size):
        raise ValueError('expanded_support_leaves_source_image')
    x, y = np.linspace(lo[0], hi[0], config.boundary_samples_per_side), np.linspace(lo[1], hi[1], config.boundary_samples_per_side)
    # A dense rectangle also detects projective branch changes across its
    # interior. The extra model-space padding is an explicit loose allowance,
    # not a claim of statistically calibrated camera uncertainty.
    xx, yy = np.meshgrid(x, y)
    model_xy = source_points_to_model(hypothesis['projection_chain'], np.c_[xx.ravel(), yy.ravel()])
    bounds = np.r_[model_xy.min(axis=0)-config.model_region_padding_units,
                   model_xy.max(axis=0)+config.model_region_padding_units]
    return {'bounds_model': bounds.tolist(), 'unit': config.expected_model_unit,
            'source_support_envelope_px': [*lo.tolist(), *hi.tolist()],
            'model_region_padding_units': config.model_region_padding_units,
            'source_padding_px': padding, 'region_kind': 'coarse_possible_support_envelope',
            'not_measured_contact_or_com_projection': True,
            'engineering_envelope_not_statistical_confidence_interval': True}


def _separation(first, second):
    a, b = np.asarray(first, float), np.asarray(second, float)
    return float(np.linalg.norm(np.maximum(0., np.maximum(a[:2]-b[2:], b[:2]-a[2:]))))


def _ground_state(row, observation, config):
    error = _support_error(row, observation)
    result = {'status': 'unused' if error else 'coarse_camera_dependent_support_candidates',
        'unused_reason': error, 'source_pts': observation['pts'], 'source_time_base': observation['time_base'],
        'source_time_s': observation['source_time_s'], 'source_image_sha256': observation.get('image_sha256'),
        'used_for_camera_pose_solve': False, 'independent_validation': False,
        'body_box_is_not_contact': True, 'regions_by_camera_hypothesis': []}
    if error:
        return result
    hypotheses = _camera_hypotheses(row)
    if not hypotheses:
        result.update(status='unused', unused_reason='no_camera_hypotheses_for_support_region')
        return result
    for index, hypothesis in enumerate(hypotheses):
        record = {'hypothesis_index': index, 'reference_id': hypothesis.get('reference_id'),
                  'role': hypothesis.get('role'), 'mapped': False}
        try:
            record.update(_region(row, observation, hypothesis, config), mapped=True)
        except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
            record['unused_reason'] = str(exc)
        result['regions_by_camera_hypothesis'].append(record)
    if not any(item['mapped'] for item in result['regions_by_camera_hypothesis']):
        result.update(status='unused', unused_reason='no_invertible_camera_support_mapping')
    return result


def apply_athlete_continuity(frames, constraints, config=None):
    """Return copied frames and explicit cross-cut compatibility diagnostics.

    ``constraints`` uses ``rinkspace.coarse-athlete-continuity-evidence.v1``.
    Unreviewed programme labels and source PTS alone never enable the factor.
    Candidate time continuity only marks a soft shortlist, retaining every
    original camera hypothesis. Confirmed time continuity can prune a stored
    ambiguous hypothesis list, with originals retained; it still cannot publish
    a pose, grant independent validation, or create a precise athlete track.
    """
    config = config or AthleteContinuityConfig()
    result = copy.deepcopy(frames)
    diagnostics = {'algorithm': ALGORITHM_VERSION, 'config': asdict(config), 'comparisons': [],
        'camera_pose_solved_from_athlete': False, 'camera_projection_modified': False,
        'published_localization_state_modified': False, 'metric_speed_or_precise_trajectory_output': False,
        'independent_validation': False, 'used_frame_count': 0}
    if not constraints:
        diagnostics['unused_reason'] = 'no_reviewed_athlete_constraints'
        return {'frames': result, 'diagnostics': diagnostics}
    diagnostics['constraint_content_sha256'] = hashlib.sha256(json.dumps(constraints, ensure_ascii=False,
        sort_keys=True, separators=(',', ':')).encode('utf8')).hexdigest()
    temporal = constraints.get('source_time_continuity') or {}
    time_status = temporal.get('status')
    if time_status not in CONFIRMED_TIME | REVIEWED_TIME:
        diagnostics['unused_reason'] = 'programme_time_continuity_not_visually_reviewed'
        return {'frames': result, 'diagnostics': diagnostics}
    mode = 'confirmed_time_candidate_pruning' if time_status in CONFIRMED_TIME else 'reviewed_time_soft_candidate_filter'
    diagnostics['mode'] = mode
    interval = temporal.get('interval_s', [])
    cut = temporal.get('cut_s')
    if len(interval) != 2 or not isinstance(cut, (float, int)) or not interval[0] <= cut <= interval[1]:
        diagnostics['unused_reason'] = 'missing_reviewed_cut_interval'
        return {'frames': result, 'diagnostics': diagnostics}
    rows = {_source_key(row): row for row in result}
    observations = []
    for observation in constraints.get('observations', []):
        row = rows.get(_source_key(observation))
        if row is None:
            diagnostics['comparisons'].append({'source_time_s': observation['source_time_s'],
                                               'unused_reason': 'annotated_source_frame_not_in_solve_output'})
            continue
        if row.get('unit') != config.expected_model_unit:
            row['athlete_ground_state'] = {'status': 'unused', 'unused_reason': 'model_unit_does_not_match_loose_region_budget'}
            continue
        state = _ground_state(row, observation, config)
        row['athlete_ground_state'] = state
        observations.append((row, observation, state))
    before = [item for item in observations if interval[0] <= item[1]['source_time_s'] < cut]
    after = [item for item in observations if cut <= item[1]['source_time_s'] <= interval[1]]
    if not before or not after:
        diagnostics['unused_reason'] = 'no_reviewed_support_pair_on_both_sides_of_cut'
        return {'frames': result, 'diagnostics': diagnostics}
    previous, prior_observation, prior = max(before, key=lambda item: item[1]['source_time_s'])
    for current, observation, ground in after:
        record = {'before_pts': previous['pts'], 'after_pts': current['pts'], 'time_base': current['time_base'],
            'before_source_time_s': previous['source_time_s'], 'after_source_time_s': current['source_time_s'],
            'time_continuity_status': time_status, 'mode': mode, 'constant_velocity_assumed': False,
            'independent_validation': False, 'used': False, 'hypotheses': []}
        diagnostics['comparisons'].append(record)
        current['athlete_continuity'] = record
        if previous.get('shot_id') == current.get('shot_id') or previous.get('shot_id') is None or current.get('shot_id') is None:
            record['unused_reason'] = 'no_distinct_camera_shots_at_reviewed_cut';continue
        if previous.get('model_sha256') != current.get('model_sha256') or not previous.get('model_sha256'):
            record['unused_reason'] = 'cross_camera_model_identity_mismatch';continue
        if current['source_sha256'] != previous['source_sha256']:
            record['unused_reason'] = 'cross_camera_source_identity_mismatch';continue
        if not 0 < float(_time(current)-_time(previous)) <= config.max_observation_gap_s:
            record['unused_reason'] = 'reviewed_support_gap_exceeds_configured_neighbourhood';continue
        prior_regions = prior['regions_by_camera_hypothesis']
        if prior['status'] == 'unused' or ground['status'] == 'unused':
            record['unused_reason'] = prior.get('unused_reason') or ground.get('unused_reason');continue
        if not prior_regions or not all(region['mapped'] for region in prior_regions):
            record['unused_reason'] = 'not_all_previous_camera_hypotheses_have_ground_regions';continue
        for candidate in ground['regions_by_camera_hypothesis']:
            assessment = {'hypothesis_index': candidate['hypothesis_index'], 'reference_id': candidate.get('reference_id')}
            if not candidate['mapped']:
                assessment.update(compatibility='unknown', unused_reason=candidate['unused_reason'])
            else:
                distances = [_separation(candidate['bounds_model'], prior_region['bounds_model']) for prior_region in prior_regions]
                compatible = [i for i, distance in enumerate(distances) if distance <= config.allowed_region_separation_model_units]
                assessment.update(compatibility='compatible_with_possible_prior_region' if compatible else 'incompatible_with_all_possible_prior_regions',
                    possible_prior_hypothesis_indices=compatible, separation_from_each_prior_region_model_units=distances,
                    allowed_region_separation_model_units=config.allowed_region_separation_model_units)
            record['hypotheses'].append(assessment)
        disfavored = [item['hypothesis_index'] for item in record['hypotheses'] if item['compatibility']=='incompatible_with_all_possible_prior_regions']
        retained = [item['hypothesis_index'] for item in record['hypotheses'] if item['hypothesis_index'] not in disfavored]
        record.update(used=True, soft_disfavored_hypothesis_indices=disfavored, soft_retained_hypothesis_indices=retained,
            previous_camera_hypothesis_count=len(prior_regions), active_hypotheses_pruned=False)
        diagnostics['used_frame_count'] += 1
        # All rejected is a contradiction in the input assumptions, not proof
        # that one of them may be selected anyway. Preserve the entire set.
        if not retained:
            record['filter_status'] = 'all_camera_candidates_conflict_with_support_assumption_kept_for_review'
        elif mode == 'confirmed_time_candidate_pruning' and current.get('status') == 'ambiguous' and disfavored:
            original = copy.deepcopy(current['model_anchor_hypotheses'])
            current['model_anchor_hypotheses_before_athlete_filter'] = original
            current['model_anchor_hypotheses'] = [original[i] for i in retained]
            record.update(active_hypotheses_pruned=True, filter_status='camera_identity_candidates_pruned_without_pose_promotion')
        else:
            record['filter_status'] = 'soft_shortlist_only_no_camera_pose_or_status_change'
    if not diagnostics['used_frame_count']:
        diagnostics['unused_reason'] = 'no_cross_camera_pair_with_complete_camera_and_support_evidence'
    return {'frames': result, 'diagnostics': diagnostics}
