"""Conditional identity disambiguation using actual same-shot image edges.

Neighbour cameras may drift. Their reprojections only rank broad competing
identities; they are not accuracy labels and are not added as fixed poses.
Elevated board points use world rays, never ground-plane intersections.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from .landmark_association import camera_from_record
from .multisurface_camera import ground_points, world_rays, project_camera
from .whole_rink import point_polyline_distances


def contact_category_conflicts(hypothesis, observations, scene_frame):
    """Prevent an observed board-contact edge from being renamed a ring.

    This uses a different measured geometric category, not a predicted camera
    location. The boundary itself remains an uncertain contact observation.
    """
    contact=scene_frame.get('channels',{}).get('ice_board_contact',{}).get('observation') or {}
    boundary=np.asarray(contact.get('observed_xy',[]),float).reshape(-1,2)
    flags=np.asarray(contact.get('inlier_flags',[True]*len(boundary)),bool)
    if len(flags)!=len(boundary):raise ValueError('Contact flags do not match observed points')
    boundary=boundary[flags]
    if len(boundary)<4:
        return []
    boundary=boundary[np.argsort(boundary[:,0])]
    tolerance=max(12.,2*float(contact.get('uncertainty_source_px',6.)))
    conflicts=[]
    for group in hypothesis['groups']:
        identity=group['physical_id']
        if not (identity.startswith(('F','R')) or identity in {'centre_circle','big_red_left','big_red_right'}):
            continue
        for index in group['observation_indices']:
            points=np.asarray(observations[index]['points_source_px'],float)
            # Avoid creating observed support across an unmeasured span.
            covered=np.zeros(len(points),bool)
            distance=np.full(len(points),np.inf)
            for a,b in zip(boundary[:-1],boundary[1:]):
                if b[0]-a[0]>100.:continue
                covered |= (points[:,0]>=a[0]-10.) & (points[:,0]<=b[0]+10.)
                distance=np.minimum(distance,point_polyline_distances(points,np.array([a,b])))
            fraction=float(np.mean(distance[covered]<tolerance)) if covered.any() else 0.
            if covered.sum()>=6 and covered.mean()>=.35 and fraction>=.7:
                conflicts.append({'observation_index':index,'physical_id':identity,
                                  'contact_overlap_fraction':fraction,
                                  'measured_contact_coverage_fraction':float(covered.mean()),
                                  'tolerance_source_px':tolerance})
    return conflicts


def edge_residual(reference_camera, candidate_camera, previous_xy, current_xy, surface):
    if surface == 'board_surface':
        if not np.allclose(reference_camera.centre, candidate_camera.centre, atol=1e-6, rtol=0):
            return None
        points, valid = world_rays(reference_camera, previous_xy)
    elif surface in {'ice_texture', 'ice_marking'}:
        points, valid = ground_points(reference_camera, previous_xy)
    else:
        raise ValueError('Only explicitly typed image correspondences are accepted')
    projected, good = project_camera(candidate_camera, points, rays=surface == 'board_surface')
    valid &= good & np.isfinite(current_xy).all(axis=1)
    if valid.sum() < 6 or valid.mean() < .6:
        return None
    errors = np.linalg.norm(projected[valid] - np.asarray(current_xy)[valid], axis=1)
    return {'surface': surface, 'point_count': int(valid.sum()),
            'median_sample_px': float(np.median(errors)), 'p90_sample_px': float(np.percentile(errors,90))}


def resolve_frame(frame, baseline_by_pts, scene_by_pts, arrays, *, scale_px=25., ambiguity_margin=1.):
    """Rank identities, preserving every image hypothesis and rejected trace."""
    out = deepcopy(frame)
    hypotheses = out.get('association', {}).get('candidate_hypotheses', [])
    if not hypotheses:
        return out
    pts = int(frame['pts'])
    old = baseline_by_pts.get(pts, {})
    observed = scene_by_pts.get(pts, {})
    edges = []
    # Both adjacent and persistent-track long edges remain actual measured
    # pixel pairs. A frame across a cut cannot supply the reference pose.
    for channel in ('ice_texture', 'ice_marking', 'board_surface'):
        channel_data = observed.get('channels', {}).get(channel, {})
        for link in [channel_data, channel_data.get('long_edge', {})]:
            prefix = link.get('npz_prefix')
            refpts = link.get('previous_source_pts', link.get('reference_source_pts'))
            reference = baseline_by_pts.get(refpts, {})
            if not prefix or not reference.get('camera') or reference.get('shot_id') != frame.get('shot_id'):
                continue
            suffix='_previous_xy' if 'previous_source_pts' in link else '_reference_xy'
            names = [prefix + suffix, prefix+'_current_xy']
            if not all(name in arrays for name in names):
                continue
            xy = [np.asarray(arrays[name],float)*.5 for name in names]
            edges.append((channel, refpts, camera_from_record(reference['camera']), *xy))
    scored = []
    for index, hypothesis in enumerate(hypotheses):
        candidate = camera_from_record(hypothesis['camera'])
        groups = []
        for channel, refpts, reference, previous, current in edges:
            result = edge_residual(reference, candidate, previous, current, channel)
            if result:
                groups.append(dict(result, reference_source_pts=refpts))
        # Equal channel weight, not a vote for the channel with most corners.
        channels = sorted({g['surface'] for g in groups})
        channel_errors = [float(np.median([g['median_sample_px'] for g in groups if g['surface']==c])) for c in channels]
        motion = float(np.mean(channel_errors)) if channel_errors else None
        score = float(hypothesis['score']) + (motion/scale_px if motion is not None else 0.)
        category_conflicts=contact_category_conflicts(hypothesis,out['landmark_observations'],observed)
        scored.append(dict(hypothesis_index=index, combined_score=score,
                           contour_score=float(hypothesis['score']),
                           conditional_neighbour_median_sample_px=motion, image_edge_groups=groups,
                           geometric_category_conflicts=category_conflicts))
    scored.sort(key=lambda h:(bool(h['geometric_category_conflicts']),not bool(h['image_edge_groups']),h['combined_score']))
    if not edges or not any(s['image_edge_groups'] for s in scored):
        out['graph_identity_resolution'] = {'status':'no_anchored_same_shot_image_edge',
            'accuracy_validated':False, 'hypotheses':scored}
        selected_groups=[{'physical_id':o['selected_model_id'],'observation_indices':[i]}
            for i,o in enumerate(out.get('landmark_observations',[])) if o.get('selected_model_id')]
        conflicts=contact_category_conflicts({'groups':selected_groups},out.get('landmark_observations',[]),observed)
        for conflict in conflicts:
            out['landmark_observations'][conflict['observation_index']].update(
                selected_model_id=None,used_by_solver=False,association_status='conflict',
                rejection_stage='geometric_category',rejection_reason='measured_contact_edge_cannot_also_be_ring_arc')
        # Do not inject an unresolved best pose into the bundle.
        if not any(o.get('selected_model_id') for o in out.get('landmark_observations',[])):
            out['landmark_camera_candidate']=None
            out['landmark_initialization_accepted']=False
        elif conflicts:
            # Its numerical fit included a misclassified surface. Keep the
            # remaining observations, but do not bootstrap from that fit.
            out['landmark_camera_candidate']=None
            out['landmark_initialization_accepted']=False
        return out
    eligible=[s for s in scored if s['image_edge_groups'] and not s['geometric_category_conflicts']]
    if not eligible:
        out['graph_identity_resolution']={'status':'no_comparable_hypothesis_with_typed_image_support',
                                         'hypotheses':scored,'accuracy_validated':False}
        out['landmark_camera_candidate']=None
        out['landmark_initialization_accepted']=False
        for observation in out['landmark_observations']:
            observation.update(selected_model_id=None,used_by_solver=False)
        return out
    best = eligible[0]
    competitive = [s for s in scored if s['image_edge_groups'] and not s['geometric_category_conflicts'] and s['combined_score'] <= best['combined_score']+ambiguity_margin]
    mappings=[]
    for row in competitive:
        mapping={}
        for group in hypotheses[row['hypothesis_index']]['groups']:
            for index in group['observation_indices']:
                mapping[index]=group
        mappings.append(mapping)
    best_hypothesis=hypotheses[best['hypothesis_index']]
    best_groups={i:g for g in best_hypothesis['groups'] for i in g['observation_indices']}
    for index, observation in enumerate(out.get('landmark_observations',[])):
        # Eligibility already excludes withheld blue axis lines. A blue centre
        # circle is a different physical structure and must receive the same
        # image-edge ambiguity check as every other eligible landmark.
        if observation.get('association_status')=='masked_out':
            continue
        group=best_groups.get(index)
        agreed=group and all(m.get(index,{}).get('physical_id')==group['physical_id'] for m in mappings)
        observation['selected_model_id']=group['physical_id'] if agreed else None
        observation['used_by_solver']=False
        if agreed:
            observation.update(association_status='associated',group_weight=1./len(group['observation_indices']),
                residual_summary_sample_px=group['residual_summary_sample_px'],
                rejection_stage=None,rejection_reason=None,
                identity_basis=['actual_visible_contours','conditional_same_shot_image_edge_ranking'],
                manual_confirmed=False)
        elif group:
            observation.update(association_status='detected_unidentified',rejection_stage='association',
                rejection_reason='multiple_identities_remain_after_actual_image_edge_ranking')
    selected={o['selected_model_id'] for o in out['landmark_observations'] if o.get('selected_model_id')}
    out['landmark_camera_candidate']=best_hypothesis['camera'] if selected else None
    out['landmark_initialization_accepted']=bool(len(selected)>=2 and len(competitive)==1)
    out['graph_identity_resolution']={'status':'conditional_identity_ranking_not_geometric_validation',
        'neighbour_pose_source':'frozen_v09_candidate_may_contain_drift',
        'old_pose_is_not_added_as_an_observation':True,
        'scale_sample_px':scale_px,'ambiguity_score_margin':ambiguity_margin,
        'hypotheses':scored,'selected_hypothesis_index':best['hypothesis_index'],
        'competitive_hypothesis_indices':[s['hypothesis_index'] for s in competitive],
        'selected_physical_ids':sorted(selected),'accuracy_validated':False}
    return out
