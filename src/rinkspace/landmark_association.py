"""Multi-contour, conditional landmark association with explicit alternatives.

This module consumes image-only detector output. It never reads intervention
labels, invents observed arcs, or treats a fitted ellipse centre as a rink point.
Camera references are broad search priors; ambiguity remains in the result.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import combinations
import numpy as np

from .landmark_camera import curve_residual, fit_camera_curves, observations_to_curves
from .multisurface_camera import Camera, ground_points, project_camera
from .landmark_band_profile import evaluate_outer_boundary_cue, point_array_sha256


@dataclass(frozen=True)
class AssociationConfig:
    broad_search_sample_px: float = 180.
    maximum_hypotheses: int = 32
    maximum_seeds: int = 3
    per_identity_proposals: int = 5
    maximum_fit_evaluations: int = 55
    min_source_span_px: float = 100.
    min_distinct_groups: int = 2
    compatible_group_p90_sample_px: float = 8.
    compatible_group_median_sample_px: float = 4.
    relative_ambiguity_margin: float = .2
    absolute_ambiguity_margin: float = .75
    minimum_circle_angular_support_deg: float = 90.
    # Any blue curve could be blue_left. Do not use uncertain blue_right
    # assignments to leak the frozen blue_left physical validation group.
    exclude_blue_from_fit: bool = True
    independent_source_budget_fraction: float = .5


def source_complementary_pairs(observations, *, maximum_pairs=8, profiles=None,
                               allow_blue=False, eligible_observation_ids=None):
    """Image-only complementary arc proposals, with no camera/ID gate.

    Long upper/lower paint boundaries often remain near straight under a low
    camera. Their source-coordinate overlap and separation provide an alternate
    start when the current rink projection is too far away to guide matching.
    """
    candidates = []
    for index, observation in enumerate(observations):
        if observation.get('association_status') == 'masked_out' or (observation.get('colour_class') == 'blue' and not allow_blue):
            continue
        if eligible_observation_ids is not None and observation['observation_id'] not in eligible_observation_ids:continue
        points=np.asarray(observation.get('points_source_px',[]),float).reshape(-1,2)
        if len(points)<10 or not np.isfinite(points).all():continue
        span=np.ptp(points,axis=0)
        if span.max()<150:continue
        candidates.append((index,points,span))
    pairs=[]
    for (i,a,sa),(j,b,sb) in combinations(candidates,2):
        if observations[i].get('colour_class')!=observations[j].get('colour_class'):continue
        # Use a principal image direction, not rink/model axes or predicted IDs.
        major=int(np.argmax(sa+sb));minor=1-major
        overlap=min(a[:,major].max(),b[:,major].max())-max(a[:,major].min(),b[:,major].min())
        minimum_span=min(sa[major],sb[major]);separation=abs(np.median(a[:,minor])-np.median(b[:,minor]))
        if overlap<.55*minimum_span or separation<max(7.,.012*minimum_span) or separation>.45*max(sa[major],sb[major]):continue
        if min(sa[major],sb[major])<.45*max(sa[major],sb[major]):continue
        score=minimum_span*(overlap/minimum_span)+.2*(sa[major]+sb[major])
        brightness=[]
        midpoint=(np.median(a[:,minor])+np.median(b[:,minor]))/2
        for index,points in [(i,a),(j,b)]:
            profile=(profiles or {}).get(observations[index]['observation_id'])
            status='unknown'
            if profile:
                if profile['point_array_sha256']!=point_array_sha256(points):raise ValueError('Source pair brightness points do not match')
                good=np.asarray(profile['gradient_reliable'],bool)
                gradient=np.asarray(profile['gradient_unit_source_xy'])
                if good.sum()>=5 and good.mean()>=.4:
                    alignment=gradient[good,minor]*np.sign(np.median(points[:,minor])-midpoint)
                    if np.mean(alignment>.35)>=.7 and np.mean(alignment<-.35)<=.15:status='source_pair_outward_brightness_supported'
                    elif np.mean(alignment<-.35)>=.7 and np.mean(alignment>.35)<=.15:status='source_pair_inward_brightness_supported'
            brightness.append(status)
        # This is a camera-independent shape proposal. It does not call the
        # image ellipse centre a projected physical rink centre or assign IDs.
        outward=brightness.count('source_pair_outward_brightness_supported')
        conflict='source_pair_inward_brightness_supported' in brightness
        ranking_score=score*(1+outward) if not conflict else score*.1
        pairs.append({'observation_indices':(i,j),'source_span_score':float(score),
                      'source_ranking_score':float(ranking_score),
                      'source_pair_brightness':brightness,
                      'source_pair_brightness_conflict':conflict,
                      'source_overlap_fraction':float(overlap/minimum_span),
                      'source_separation_px':float(separation)})
    return sorted(pairs,key=lambda p:-p['source_ranking_score'])[:maximum_pairs]


def camera_from_record(record):
    k = np.asarray(record['K'], float)
    return Camera(np.asarray(record['R'], float),
                  np.asarray(record.get('C', record.get('center_model')), float),
                  float(record.get('focal_sample_px', record.get('focal_px', k[0, 0]))),
                  np.asarray(record.get('principal_sample_px', k[:2, 2]), float),
                  tuple(record.get('sample_size', (960, 540))),
                  float(record.get('radial_ray_k1', 0.)))


def candidate_model_ids(observation, model, *, exclude_blue=True):
    """Colour/shape classes generate possibilities, never final identities."""
    colour = observation.get('colour_class', 'gray')
    result = []
    for primitive in model['structure']['primitives']:
        identity, kind = primitive['id'], primitive['kind']
        if kind == 'circle':
            if colour == 'blue':
                if identity == 'centre_circle' and not exclude_blue:
                    result.append(identity)
            elif colour == 'red':
                if identity.startswith(('R', 'F', 'big_red')):
                    result.append(identity)
            elif not identity.startswith('big_red'):
                # A grazing-view circle can be classified line/arc_or_line.
                result.append(identity)
        elif kind == 'axis_line' and colour == 'red' and identity == 'red_center':
            result.append(identity)
        elif kind == 'axis_line' and colour == 'blue' and not exclude_blue and identity.startswith('blue_'):
            result.append(identity)
    return result


def _points(observation, sample_size, source_size):
    p = np.asarray(observation.get('points_source_px', []), float).reshape(-1, 2)
    if len(p) > 40:
        p = p[np.linspace(0, len(p)-1, 40).round().astype(int)]
    return p * (np.asarray(sample_size)/np.asarray(source_size))


def _curve(index, observation, identity, model, sample_size, source_size):
    scale = float(np.mean(np.asarray(sample_size)/np.asarray(source_size)))
    item = dict(observation, selected_model_id=identity, association_status='associated',
                group_weight=1., observation_id=str(index),
                measurement_uncertainty_source_px=max(4., observation.get('measurement_uncertainty_source_px', 6.)))
    return observations_to_curves(0, [item], model, scale=scale)[0]


def _stats(camera, curve):
    error = np.abs(curve_residual(camera, curve))
    ground, valid = ground_points(camera, curve.observed_xy)
    if not valid.all() or not np.isfinite(error).all():
        return None
    return {'median_px': float(np.median(error)), 'p90_px': float(np.percentile(error, 90)),
            'p95_px': float(np.percentile(error, 95)), 'maximum_px': float(error.max())}


def _visible_primitive(camera, primitive, margin):
    points = np.asarray(primitive['points_model'], float)
    xy, valid = project_camera(camera, np.c_[points, np.zeros(len(points))])
    xy = xy[valid]
    if not len(xy):
        return False
    return bool(np.all(xy.max(axis=0) >= -margin) and
                np.all(xy.min(axis=0) <= np.asarray(camera.size)+margin))


def _angular_support(camera, curve, points):
    raw = np.c_[points, np.ones(len(points))] @ np.linalg.inv(camera.H).T
    xy = raw[:, :2]/raw[:, 2:3]
    centre = -curve.equation_model[:2, 2]
    angles = np.sort(np.mod(np.arctan2(xy[:, 1]-centre[1], xy[:, 0]-centre[0]), 2*np.pi))
    gaps = np.diff(np.r_[angles, angles[0]+2*np.pi])
    return float(np.degrees(2*np.pi-gaps.max()))


def _band_cue(camera, curve, observation, profile, sample_size, source_size):
    if profile is None:
        return {'status':'unknown_no_image_brightness_profile','source_outer_support_bonus':0.}
    points=np.asarray(observation['points_source_px'],float)
    if profile.get('point_array_sha256')!=point_array_sha256(points):
        raise ValueError('Brightness profile must bind unchanged source points')
    scale=np.asarray(sample_size)/np.asarray(source_size)
    measured=deepcopy(curve);measured.observed_xy=points*scale
    derivatives=[]
    # Differentiate in original distorted source pixels. curve_residual carries
    # raw sample pixels through the declared radial camera chain before fitting.
    for axis in range(2):
        delta=np.zeros(2);delta[axis]=scale[axis]
        plus=deepcopy(measured);minus=deepcopy(measured)
        plus.observed_xy+=delta;minus.observed_xy-=delta
        derivatives.append((curve_residual(camera,plus)-curve_residual(camera,minus))/2)
    cue=evaluate_outer_boundary_cue(profile,np.c_[derivatives[0],derivatives[1]])
    return {k:v for k,v in cue.items() if k not in {'outward_alignment_cosine','usable_points'}} | {
        'profile_point_array_sha256':profile['point_array_sha256'],
        'source_outer_support_bonus':1.5 if cue['status']=='brightness_supports_outer_side_of_this_model_hypothesis' else 0.}


def _compatible_groups(camera, curve_options, observations, config, profiles=None, sample_size=(960,540), source_size=(1920,1080)):
    """Choose complementary observations jointly under one camera.

    Complementary visible fragments can support one circle, but their total
    weight remains one. A tiny nearly straight scratch cannot identify a ring.
    """
    options = []
    for (index, identity), curve in curve_options.items():
        stats = _stats(camera, curve)
        if stats and stats['median_px'] <= config.compatible_group_median_sample_px and stats['p90_px'] <= config.compatible_group_p90_sample_px:
            cue=_band_cue(camera,curve,observations[index],(profiles or {}).get(observations[index]['observation_id']),sample_size,source_size)
            if cue['status']=='brightness_supports_inner_side_of_this_model_hypothesis':continue
            span = float(np.linalg.norm(np.ptp(curve.observed_xy, axis=0)))
            score = stats['median_px'] + .35*stats['p90_px'] - .4*np.log1p(span/50)-cue['source_outer_support_bonus']
            options.append((score, index, identity, stats, cue))
    grouped, taken_indices = {}, set()
    for score, index, identity, stats, cue in sorted(options,key=lambda row:row[0]):
        if index in taken_indices:
            continue
        rows = grouped.setdefault(identity, [])
        if len(rows) >= 6:
            continue
        taken_indices.add(index)
        rows.append((index, stats, score, cue))
    picked = []
    for identity, rows in grouped.items():
        curve = curve_options[rows[0][0], identity]
        points = np.concatenate([curve_options[row[0], identity].observed_xy for row in rows])
        angular = _angular_support(camera, curve, points) if curve.kind == 'circle' else None
        if angular is not None and angular < config.minimum_circle_angular_support_deg:
            continue
        # One aggregate score per physical ink pattern, not per sampled pixel.
        stats = {'median_px': float(np.mean([row[1]['median_px'] for row in rows])),
                 'p90_px': max(row[1]['p90_px'] for row in rows),
                 'p95_px': max(row[1]['p95_px'] for row in rows)}
        score = stats['median_px']+.35*stats['p90_px']-.7*np.log1p(np.linalg.norm(np.ptp(points, axis=0))/50)-float(np.mean([row[3]['source_outer_support_bonus'] for row in rows]))
        picked.append({'observation_indices': [row[0] for row in rows], 'physical_id': identity,
                       'observation_ids': [observations[row[0]]['observation_id'] for row in rows],
                       'residual_summary_sample_px': stats, 'group_weight': 1.,
                       'visible_angular_support_degrees': angular, 'score': float(score)})
        picked[-1]['brightness_side_cues']=[row[3] for row in rows]
    return picked


def associate_frame(frame, model, camera_seeds, *, config=None, allow_partial_existing_pose=False,
                    band_profiles=None, qualified_model_ids=None):
    """Fit bounded multiple physical-group hypotheses from actual contours.

    camera_seeds contains (provenance_label, Camera). Missing input frame poses
    do not prevent callers from supplying known reference-camera groups.
    Returned cameras/identities remain conditional candidates, not calibration.
    """
    config = config or AssociationConfig()
    observations = deepcopy(frame.get('landmark_observations', []))
    source_size = frame.get('source_size', [1920, 1080])
    sample_size = frame.get('sample_size', [960, 540])
    profiles={row['observation_id']:row for row in (band_profiles or [])}
    curve_options = {}
    for index, observation in enumerate(observations):
        observation['used_by_solver'] = False
        observation['manual_confirmed'] = False
        observation['selected_model_id'] = None
        if observation.get('association_status') == 'masked_out':
            continue
        if config.exclude_blue_from_fit and observation.get('colour_class') == 'blue':
            observation.update(candidate_model_ids=['blue_left', 'blue_right'],
                               rejection_stage='physical_group_holdout',
                               rejection_reason='blue_left_frozen_for_evaluation_blue_identity_unresolved',
                               association_status='detected_unidentified')
            continue
        points = np.asarray(observation.get('points_source_px', []), float).reshape(-1, 2)
        if len(points) < 8 or not np.isfinite(points).all() or np.linalg.norm(np.ptp(points, axis=0)) < config.min_source_span_px:
            observation.update(rejection_stage='shape_support', rejection_reason='too_little_visible_curve_extent')
            continue
        identities = candidate_model_ids(observation, model, exclude_blue=config.exclude_blue_from_fit)
        if qualified_model_ids is not None:
            allowed=set(qualified_model_ids.get(observation['observation_id'],[]))
            identities=[identity for identity in identities if identity in allowed]
        observation['candidate_model_ids'] = identities
        for identity in identities:
            curve_options[index, identity] = _curve(index, observation, identity, model, sample_size, source_size)

    primitives = {p['id']: p for p in model['structure']['primitives']}
    proposals, partial_proposals = [], []
    for seed_index, (label, seed) in enumerate(camera_seeds[:config.maximum_seeds]):
        by_id = {}
        for (index, identity), curve in curve_options.items():
            if not _visible_primitive(seed, primitives[identity], config.broad_search_sample_px):
                continue
            stats = _stats(seed, curve)
            if not stats or stats['p90_px'] > config.broad_search_sample_px:
                continue
            span = float(np.linalg.norm(np.ptp(curve.observed_xy, axis=0)))
            # A long visible arc is more informative than a convenient scratch.
            score = stats['median_px'] + .3*stats['p90_px'] - 8*np.log1p(span/80)
            by_id.setdefault(identity, []).append((score, index))
        shortlist = []
        for identity, rows in by_id.items():
            nearest = sorted(rows)[:config.per_identity_proposals]
            longest = sorted(rows, key=lambda row: -np.linalg.norm(np.ptp(curve_options[row[1], identity].observed_xy, axis=0)))[:3]
            shortlist.extend((identity, score, index) for score, index in sorted(set(nearest+longest)))
        partial_proposals.extend((score, seed_index, ((index, identity),)) for identity, score, index in shortlist)
        for a, b in combinations(shortlist, 2):
            if a[0] == b[0] or a[2] == b[2]:
                continue
            pa, pb = curve_options[a[2], a[0]].observed_xy, curve_options[b[2], b[0]].observed_xy
            if np.linalg.norm(np.median(pa, axis=0)-np.median(pb, axis=0)) < 50:
                continue
            proposals.append((a[1]+b[1], seed_index, ((a[2], a[0]), (b[2], b[0]))))
    # Allocate the bounded search across reference seeds, not just one current
    # camera. References can propose hypotheses even when the pose is missing.
    ordered, count_per_seed = [], {}
    per_seed = max(1, config.maximum_hypotheses//max(1, min(len(camera_seeds), config.maximum_seeds)))
    for proposal in sorted(proposals):
        seed_index = proposal[1]
        if count_per_seed.get(seed_index, 0) >= per_seed:
            continue
        count_per_seed[seed_index] = count_per_seed.get(seed_index, 0)+1
        ordered.append(proposal)
    source_budget=max(0,int(config.maximum_hypotheses*config.independent_source_budget_fraction))
    eligible_ids=({observations[index]['observation_id'] for (index,identity),curve in curve_options.items()
                   if curve.kind=='circle'} if qualified_model_ids is not None else None)
    source_pairs=source_complementary_pairs(observations,profiles=profiles,
                    allow_blue=not config.exclude_blue_from_fit,eligible_observation_ids=eligible_ids)
    independent=[]
    for rank,source_pair in enumerate(source_pairs):
        if source_pair['source_pair_brightness_conflict']:continue
        i,j=source_pair['observation_indices']
        for seed_index,(_,seed) in enumerate(camera_seeds[:config.maximum_seeds]):
            for identity,primitive in primitives.items():
                if primitive['kind']!='circle' or (i,identity) not in curve_options or (j,identity) not in curve_options:continue
                # A camera-family/heading prior may select model regions, but
                # no measured-curve residual may exclude the source-driven pair.
                if not _visible_primitive(seed,primitive,config.broad_search_sample_px):continue
                independent.append((rank,seed_index,((i,identity),(j,identity))))
    independent=independent[:source_budget]
    independent_keys={(seed_index,pair) for _,seed_index,pair in independent}
    ordered=ordered[:config.maximum_hypotheses-len(independent)]+independent
    hypotheses = []
    for _, seed_index, pair in ordered[:config.maximum_hypotheses]:
        label, seed = camera_seeds[seed_index]
        curves = [curve_options[key] for key in pair]
        source_driven=(seed_index,pair) in independent_keys
        required_groups=1 if source_driven and allow_partial_existing_pose else config.min_distinct_groups
        try:
            camera, diagnostics = fit_camera_curves(seed, curves, max_nfev=config.maximum_fit_evaluations)
            groups = _compatible_groups(camera, curve_options, observations, config,profiles,sample_size,source_size)
            if source_driven:
                identity=pair[0][1];rows=[_stats(camera,c) for c in curves]
                cues=[_band_cue(camera,c,observations[key[0]],profiles.get(observations[key[0]]['observation_id']),sample_size,source_size) for c,key in zip(curves,pair)]
                if all(row and row['p90_px']<=config.compatible_group_p90_sample_px for row in rows) and not any(cue['status']=='brightness_supports_inner_side_of_this_model_hypothesis' for cue in cues):
                    points=np.concatenate([c.observed_xy for c in curves])
                    angular=_angular_support(camera,curves[0],points)
                    if angular>=config.minimum_circle_angular_support_deg:
                        selected_indices={key[0] for key in pair}
                        groups=[g for g in groups if g['physical_id']!=identity and not selected_indices.intersection(g['observation_indices'])]
                        summary={'median_px':float(np.mean([row['median_px'] for row in rows])),
                                 'p90_px':max(row['p90_px'] for row in rows),'p95_px':max(row['p95_px'] for row in rows)}
                        groups.append({'observation_indices':[key[0] for key in pair],'physical_id':identity,
                                       'observation_ids':[observations[key[0]]['observation_id'] for key in pair],
                                       'residual_summary_sample_px':summary,'group_weight':1.,
                                       'visible_angular_support_degrees':angular,
                                       'source_complementary_pair':True,
                                       'brightness_side_cues':cues,
                                       'score':summary['median_px']+.35*summary['p90_px']-1.5*np.log1p(np.linalg.norm(np.ptp(points,axis=0))/50)-float(np.mean([cue['source_outer_support_bonus'] for cue in cues]))})
            if len(groups) < required_groups:
                continue
            # Refit the jointly explained physical groups with capped weights.
            refined = []
            for g in groups:
                for index in g['observation_indices']:
                    curve = deepcopy(curve_options[index, g['physical_id']])
                    curve.group_weight = 1./len(g['observation_indices'])
                    refined.append(curve)
            camera, diagnostics = fit_camera_curves(camera, refined, max_nfev=config.maximum_fit_evaluations)
            rescored = _compatible_groups(camera, curve_options, observations, config,profiles,sample_size,source_size)
            # Preserve the independently nominated outer-pair hypothesis. Do
            # not silently replace its long source arcs by closest short edges.
            for old_group in groups:
                if old_group.get('source_complementary_pair'):
                    rescored=[g for g in rescored if g['physical_id']!=old_group['physical_id'] and not set(old_group['observation_indices']).intersection(g['observation_indices'])]
                    rows=[_stats(camera,curve_options[i,old_group['physical_id']]) for i in old_group['observation_indices']]
                    if all(row and row['p90_px']<=config.compatible_group_p90_sample_px for row in rows):
                        old_group=deepcopy(old_group)
                        old_group['residual_summary_sample_px']={key:float(np.mean([r[key] for r in rows])) for key in ['median_px','p90_px','p95_px']}
                        rescored.append(old_group)
            groups=rescored
            if len(groups) < required_groups:
                continue
            score = float(np.mean([g['score'] for g in groups])) - 2*len(groups)
            hypotheses.append({'score': score, 'seed': label, 'groups': groups,
                               'camera': camera.to_dict(), 'fit': diagnostics,
                               'source_driven_complementary_arc_start':source_driven,
                               'conditional_same_optical_centre_prior': True,
                               'identity_signature': sorted((index, g['physical_id']) for g in groups for index in g['observation_indices'])})
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
    partial_attempts = 0
    if not hypotheses and allow_partial_existing_pose:
        # A one-pattern correction can be attached to an already anchored
        # camera graph. It is never sufficient for new full-rink initialization.
        for _, seed_index, pair in sorted(partial_proposals)[:min(8, config.maximum_hypotheses)]:
            label, seed = camera_seeds[seed_index]
            partial_attempts += 1
            try:
                camera, diagnostics = fit_camera_curves(seed, [curve_options[key] for key in pair],
                                                        max_nfev=config.maximum_fit_evaluations)
                groups = _compatible_groups(camera, curve_options, observations, config,profiles,sample_size,source_size)
                if not groups:
                    continue
                score = float(np.mean([g['score'] for g in groups]))-2*len(groups)
                hypotheses.append({'score':score,'seed':label,'groups':groups,'camera':camera.to_dict(),
                                   'fit':diagnostics,'partial_correction_requires_existing_graph_anchor':True,
                                   'conditional_same_optical_centre_prior':True,
                                   'identity_signature':sorted((i,g['physical_id']) for g in groups for i in g['observation_indices'])})
            except (ValueError,np.linalg.LinAlgError,FloatingPointError):
                continue
    # Duplicate starts must not masquerade as distinct competing explanations.
    unique = {}
    for candidate in hypotheses:
        key = tuple(map(tuple, candidate['identity_signature']))
        if key not in unique or candidate['score'] < unique[key]['score']:
            unique[key] = candidate
    hypotheses = sorted(unique.values(), key=lambda row: row['score'])
    best = hypotheses[0] if hypotheses else None
    threshold = (config.absolute_ambiguity_margin + config.relative_ambiguity_margin*abs(best['score'])) if best else None
    alternatives = [h for h in hypotheses if h['score'] <= best['score']+threshold] if best else []
    gap = hypotheses[1]['score']-best['score'] if len(hypotheses) > 1 else None
    for index, observation in enumerate(observations):
        if observation.get('rejection_stage') in {'physical_group_holdout', 'shape_support'} or observation.get('association_status') == 'masked_out':
            continue
        possibilities = sorted({g['physical_id'] for h in alternatives for g in h['groups'] if index in g['observation_indices']})
        used_in_best = next((g for g in best['groups'] if index in g['observation_indices']), None) if best else None
        if used_in_best:
            identity = used_in_best['physical_id']
            # Observation absence from an alternative is also weak support;
            # do not turn a coincidentally good local normal fit into certainty.
            agrees = all(any(index in g['observation_indices'] and g['physical_id'] == identity for g in h['groups']) for h in alternatives)
            observation.update(candidate_model_ids=possibilities,
                               selected_model_id=identity if agrees else None,
                               association_status='associated' if agrees else 'detected_unidentified',
                               identity_basis=['joint_multiple_physical_contours', 'visible_curve_normal_residual', 'broad_reference_camera_search_prior'],
                               residual_summary_sample_px=used_in_best['residual_summary_sample_px'],
                               brightness_side_evidence=used_in_best.get('brightness_side_cues',[]),
                               group_weight=1./len(used_in_best['observation_indices']), association_match_gap=gap,
                               rejection_stage='solver_pending' if agrees else 'association',
                               rejection_reason='candidate_identity_not_yet_used_by_global_solver' if agrees else 'competing_joint_camera_identity_hypotheses')
        else:
            observation.update(candidate_model_ids=possibilities or observation.get('candidate_model_ids', []),
                               association_status='detected_unidentified',
                               rejection_stage='association',
                               rejection_reason='no_multi_structure_camera_explanation' if best is None else 'not_supported_by_best_multi_structure_camera')
    agreed_ids = {o['selected_model_id'] for o in observations if o.get('selected_model_id')}
    # A partial consistent observation may correct an existing graph without
    # being sufficient to initialize a previously unlocalized camera.
    rank_ok = bool(best and len(best['fit'].get('jacobian_singular_values', [])) >= 4 and
                   best['fit']['jacobian_singular_values'][-1] > 1e-6*best['fit']['jacobian_singular_values'][0])
    initialization = bool(best and len(agreed_ids) >= config.min_distinct_groups and
                          best['fit']['solver_converged'] and rank_ok and
                          (gap is not None and gap > threshold))
    return {'landmark_observations': observations,
            'landmark_camera_candidate': best['camera'] if best else None,
            'landmark_initialization_accepted': initialization,
            'landmark_candidate_provenance': {'source': 'multiple_actual_curves_and_reference_camera_priors',
                                             'common_selected_physical_ids': sorted(agreed_ids),
                                             'one_group_can_correct_existing_pose_not_initialize_whole_rink': True,
                                             'initialization_is_conditional_not_independently_validated': True},
            'association': {'algorithm_version': 'joint-visible-contour-association-v10-4',
                            'config': asdict(config), 'hypotheses_attempted': len(ordered[:config.maximum_hypotheses])+partial_attempts,
                            'independent_source_hypotheses_attempted':len(independent),
                            'source_complementary_pair_proposals':source_pairs,
                            'hypotheses_surviving': len(hypotheses), 'candidate_hypotheses': hypotheses,
                            'best_camera_candidate': best['camera'] if best else None,
                            'match_gap': gap, 'ambiguity_score_tolerance': threshold,
                            'plausible_identity_hypothesis_count': len(alternatives),
                            'manual_intervention_inputs_read': False,
                            'missing_current_pose_does_not_disable_reference_search': True,
                            'fit_residual_is_not_independent_validation': True,
                            'source_points_modified': False}}
