"""Quality-limited image observation chains, separate from blind prediction.

Error budgets are engineering scores, not calibrated confidence intervals.
An unanchored local identity is never a world pose. Camera components can be
restarted without changing the fixed rink or pretending to recover a map.
"""
from __future__ import annotations

import copy
import numpy as np


def register_edge(nodes, local, sample, width):
    """Keep an observed parent edge and its non-resetting uncertainty cost."""
    if local.get('status') != 'candidate':
        return
    index = sample['sample_index']
    if local.get('reference_frame_only'):
        nodes[index] = {'parent': None, 'random_variance': 0., 'systematic_px': 0.}
        return
    parent = local.get('edge_keyframe_sample_index')
    if parent not in nodes:
        return
    p95 = float(local.get('inlier_p95_reference_px', 99))*960/width
    count = max(1, local.get('train_inliers', 0))
    area = max(.005, local.get('support_fraction', 0))
    # Account for limited spatial leverage and correlated fitting bias. A new
    # derived keyframe inherits this path; it is not a fresh absolute anchor.
    leverage = max(1., np.sqrt(.15/area))
    random = max(.03, p95/np.sqrt(count))*leverage
    bias = .025*p95*leverage
    nodes[index] = {'parent': parent, 'random_variance': float(random**2),
                    'systematic_px': float(bias)}


def path_uncertainty(nodes, first, second, local, width):
    def walk(start):
        result, variance, bias = {}, 0., 0.
        while start in nodes and start not in result:
            result[start] = (variance, bias)
            edge = nodes[start]
            variance += edge['random_variance']; bias += edge['systematic_px']
            start = edge['parent']
        return result
    a, b = walk(first), walk(second)
    common = a.keys() & b.keys()
    if not common:
        return {'kind': 'engineering_not_calibrated_confidence',
                'estimated_error_px_at_960': None, 'reason': 'no_observed_path_to_anchor', 'accepted': False}
    join = min(common, key=lambda k: sum(a[k])+sum(b[k]))
    variance = a[join][0]+b[join][0]
    bias = a[join][1]+b[join][1]
    # Normalized-coordinate condition avoids interpreting translation in px
    # as singularity. This tests the observed edge, not the arbitrary gauge.
    h = np.asarray(local.get('H_keyframe_from_current', np.eye(3)), float)
    s = np.diag([1/width, 1/(width*9/16), 1.])
    condition = float(np.linalg.cond(s @ h @ np.linalg.inv(s)))
    extrapolation = max(1., np.sqrt(.15/max(.005, local.get('support_fraction', .15))))
    estimate = (1.+2*np.sqrt(variance)+bias)*extrapolation
    return {'kind': 'engineering_not_calibrated_confidence',
            'estimated_error_px_at_960': float(estimate),
            'accumulated_variance': float(variance), 'accumulated_bias_allowance_px': float(bias),
            'normalized_edge_condition': condition, 'extrapolation_multiplier': float(extrapolation),
            'accepted': bool(np.isfinite(condition) and condition < 50),
            'reference': 'actual_observed_edge_path_to_real_anchor; derived_keyframes_do_not_reset_cost'}


def push_observed(tracker, image, sample, force_break=False):
    from .geometry import project
    from .whole_rink_localization import (_source_timestamp, frame_masks, _registrar,
        _compact_fit, _polygons, _projection_chain, _model_probe_disagreement,
        _distortion_parameters)
    cfg = tracker.config
    time = _source_timestamp(sample)
    if image.shape[:2] != (cfg.sample_height, cfg.sample_width):
        raise ValueError('Image does not match configured processing dimensions')
    if any(isinstance(sample.get(k), bool) or not isinstance(sample.get(k), int) or sample[k] <= 0
           for k in ('width', 'height')):
        raise ValueError('Source pixel dimensions must be positive integers')
    changed_source = tracker.last_source_sha256 is not None and sample['source_sha256'] != tracker.last_source_sha256
    broken = (tracker.last_time is None or force_break or changed_source or
        (tracker.last_shot is not None and sample.get('shot_id') != tracker.last_shot) or
        (tracker.last_time is not None and not 0 < time-tracker.last_time <= .6))
    if broken:
        tracker.reset()
    masks = frame_masks(image)
    local = tracker.local.push(image, sample, masks, force_break=broken)
    restarted = local.get('reason') == 'recovery_gap_exceeded_requires_explicit_new_component'
    if restarted:
        # A new image component may be observed before it is attached to the
        # rink. It must not inherit a disconnected component's map bridge.
        tracker.reset()
        local = tracker.local.push(image, sample, masks, force_break=True)
    register_edge(tracker.observed_edges, local, sample, image.shape[1])
    component = f"{sample['source_sha256']}:{sample.get('shot_id')}:{tracker.propagation_segment}"
    row = {**sample, 'model_id': tracker.model['model_id'], 'model_revision': tracker.model['revision'],
        'model_sha256': tracker.model_sha256, 'status': 'unavailable',
        'reason': 'cut_waiting_for_relocalization' if broken else 'unanchored_local_component',
        'geometry_level': tracker.model['geometry_level'], 'unit': tracker.model['unit'],
        'independent_validation': False, 'map_usable_for_metric_output': False,
        'H_image_from_model': None, 'H_model_from_image': None, 'projection_chain': None,
        'anchor_id': None, 'sample_size': [image.shape[1], image.shape[0]],
        'ice_polygons_sample': _polygons(masks['ice']), 'foreground_polygons_sample': _polygons(masks['foreground']),
        'mask_status': masks['mask_status'], 'foreground_method': masks['foreground_method'],
        'precise_person_segmentation': False, 'fit_support_polygon_sample': [],
        'whole_structure_overlay': 'conditional_extrapolation_on_full_visible_ice_not_independent_support',
        'local_registration': _compact_fit(local), 'break_applied': bool(broken),
        'propagation_segment_id': tracker.propagation_segment, 'camera_component_id': component,
        'source_identity_break_applied': bool(changed_source),
        'athlete_ground_state': {'status': 'unavailable', 'reason': 'no_confirmed_support_region',
            'used_for_camera': False, 'body_box_is_not_contact': True},
        'last_image_observation_age_s': 0. if local.get('status') == 'candidate' and not local.get('reference_frame_only') else None}
    if restarted:
        row['local_restart_reason'] = 'unanchored_component_after_local_observation_gap'
    age = None if tracker.last_anchor_time is None else time-tracker.last_anchor_time
    row['anchor_age_s'] = age
    lh = local.get('H_reference_from_current')
    prior, uncertainty = None, None
    if tracker.bridge is not None and lh is not None and local['status'] == 'candidate':
        uncertainty = path_uncertainty(tracker.observed_edges, tracker.anchor_sample_index,
            sample['sample_index'], local, image.shape[1])
        row['uncertainty'] = uncertainty
        if uncertainty['accepted'] and uncertainty['estimated_error_px_at_960'] <= cfg.max_engineering_error_px_at_960:
            prior = np.linalg.inv(np.asarray(lh)) @ tracker.bridge
        else:
            row['reason'] = 'accumulated_uncertainty_exceeded'
    elif tracker.bridge is not None:
        row['reason'] = 'local_observation_lost'
    attempt = broken or restarted or sample['sample_index'] % cfg.reobserve_every == 0
    candidate, attempts = tracker._reobserve(image, masks, sample, prior) if attempt else (None, [])
    row['reobservation_attempts'] = attempts
    ref, h = None, None
    if candidate is not None:
        ref, h, fit = candidate
        probe = project(np.linalg.inv(ref['transform']), np.mean(np.asarray(fit['fit_reference_xy']), axis=0)[None])[0]
        chain = _projection_chain(ref, h, probe, image.shape[1::-1], (sample['width'], sample['height']))
        row['model_anchor_candidate'] = {'projection_chain': copy.deepcopy(chain),
            'reference_id': ref['view']['id'], 'source_time_s': time, 'pts': sample['pts'],
            'time_base': sample['time_base'], 'fit': _compact_fit(fit), 'independent_validation': False}
        row['ice_reobservation'] = _compact_fit(fit)
        if prior is not None:
            previous = _projection_chain(tracker.last_anchor, prior, tracker.reference_branch_probe,
                image.shape[1::-1], (sample['width'], sample['height']))
            consistency = _model_probe_disagreement(tracker.model, previous, chain,
                masks['usable'], cfg.anchor_conflict_p95_px_at_960*image.shape[1]/960)
            row['model_anchor_consistency'] = consistency
            if consistency['conflict']:
                tracker.pending_conflict = {'first_conflict_source_time_s': time,
                    'model_anchor_consistency': consistency, 'model_anchor_hypotheses': [
                        {'role': 'observed_chain_prior_not_ground_truth', 'reference_id': tracker.last_anchor['view']['id'],
                         'projection_chain': previous, 'at_source_time_s': time},
                        {'role': 'fresh_model_observation_not_ground_truth', 'reference_id': ref['view']['id'],
                         'projection_chain': chain, 'at_source_time_s': time}]}
        tracker.anchor_lineage.append({'reference_id': ref['view']['id'], 'pts': sample['pts'],
            'time_base': sample['time_base'], 'sample_index': sample['sample_index']})
        if tracker.pending_conflict is None:
            tracker.last_anchor, tracker.last_anchor_time = ref, time
            tracker.anchor_sample_index, tracker.reference_branch_probe = sample['sample_index'], probe
            tracker.bridge = np.asarray(lh) @ h if lh is not None and local['status'] == 'candidate' else None
            row.update(status='model_reobserved', reason='fixed_model_reference_ice_reobserved',
                anchor_age_s=0., fit_support_polygon_sample=fit['support_polygon'],
                uncertainty={'kind': 'engineering_not_calibrated_confidence', 'estimated_error_px_at_960': 1.,
                    'absolute_model_error_included': False})
    elif prior is not None and tracker.pending_conflict is None:
        ref, h = tracker.last_anchor, prior
        row.update(status='image_tracked', reason='observed_ice_chain_since_model_reobservation',
            fit_support_polygon_sample=local.get('support_polygon', []))
    row['anchor_lineage'] = copy.deepcopy(tracker.anchor_lineage)
    if tracker.pending_conflict is not None:
        row.update(copy.deepcopy(tracker.pending_conflict), status='ambiguous', reason='model_anchor_conflict',
            conflict_policy='retain_component_hypotheses_until_joint_image_evidence_or_new_camera_component',
            propagation_barrier=False)
    elif row['status'] in {'model_reobserved', 'image_tracked'}:
        view = ref['view']
        distortion = _distortion_parameters(view.get('radial_distortion'))
        nonlinear = distortion is not None and distortion['k1'] != 0
        row.update(anchor_id=view['id'],
            model_reference={'view_id': view['id'], 'image_sha256': view['image_sha256'],
                'source_sha256': view['source_sha256'], 'source_id': view.get('source_id'),
                'pts': view['pts'], 'time_base': view['time_base'], 'image': view['image'],
                'sample_size': view['sample_size'], 'image_domain': view.get('image_domain', 'raw_px_distortion_unchecked')},
            image_domain='raw_reference_transfer_approximation' if nonlinear else 'raw_px_distortion_unchecked',
            projection_chain=_projection_chain(ref, h, tracker.reference_branch_probe,
                image.shape[1::-1], (sample['width'], sample['height'])),
            cumulative_drift_independently_validated=False,
            model_bridge={'H_local_from_reference_sample': tracker.bridge.tolist() if tracker.bridge is not None else None,
                'anchor_sample_index': tracker.anchor_sample_index, 'component_id': component})
        if not nonlinear:
            raw_scale = np.diag([sample['width']/image.shape[1], sample['height']/image.shape[0], 1.])
            raw_h = raw_scale @ h @ ref['transform'] @ np.asarray(view['H_image_from_model'])
            row['H_image_from_model'], row['H_model_from_image'] = raw_h.tolist(), np.linalg.inv(raw_h).tolist()
    tracker.last_time, tracker.last_shot = time, sample.get('shot_id')
    tracker.last_source_sha256 = sample['source_sha256']
    return row
