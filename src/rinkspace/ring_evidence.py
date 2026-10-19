"""Image-only ring evidence, kept separate from camera and physical identity.

Detection searches native pixels even when a conservative ice mask is empty or
wrong. This deliberately creates unclassified scene-edge candidates: an edge
outside the old ice mask does not automatically acquire a ground-plane role.
"""
from __future__ import annotations
import gzip
import json
from functools import lru_cache
import cv2
import numpy as np

from .landmark_observations import detect_landmarks, LandmarkDetectionConfig, polygon_mask
from .landmark_association import candidate_model_ids
from .landmark_source_qualification import source_shape
from .landmark_band_profile import measure_band_profiles, point_array_sha256


@lru_cache(maxsize=32)
def _gradient_support_radius(sigma):
    """Measure this OpenCV float Gaussian's actual finite support, plus Sobel.

    GaussianBlur(..., (0,0), sigma) on CV32F currently uses four sigma, so a
    nominal three-sigma guard would still expose the measured gradient to a
    nearby person. An impulse uses the same dtype/kernel selection as profiles.
    """
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Profile Gaussian sigma must be finite and positive')
    half = max(16, int(np.ceil(8 * sigma)) + 4)
    impulse = np.zeros((1, 2 * half + 1), np.float32)
    impulse[0, half] = 1
    support = np.flatnonzero(cv2.GaussianBlur(impulse, (0, 0), sigma)[0] != 0)
    return max(int(np.max(abs(support - half))), int(np.ceil(3 * sigma))) + 1


def _touches_foreground(mask, points):
    """Any bilinear source-pixel footprint touches the binary foreground."""
    points = np.asarray(points, float)
    low = np.floor(points).astype(np.int64)
    high = np.ceil(points).astype(np.int64)
    touched = np.zeros(points.shape[:-1], bool)
    for xx in (low[..., 0], high[..., 0]):
        for yy in (low[..., 1], high[..., 1]):
            valid = (xx >= 0) & (yy >= 0) & (xx < mask.shape[1]) & (yy < mask.shape[0])
            touched[valid] |= mask[yy[valid], xx[valid]] > 0
    return touched


def qualify_profile_foreground(profiles, observations, foreground):
    """Return qualification copies; keep all point/profile measurements intact.

    ``foreground`` must be the native-size binary person/graphics mask, not a
    960-wide sample mask. This is also usable on immutable saved frame evidence
    to produce a sidecar: it never mutates the supplied profiles or observations.
    Profile samples within +/-12px and every actual multiscale Gaussian/Sobel
    support must be clear. Existing rejection reasons and false flags survive.
    """
    foreground = np.asarray(foreground)
    if foreground.ndim != 2:
        raise ValueError('Native foreground must be a two-dimensional mask')
    binary = (foreground > 0).astype(np.uint8)
    if len(profiles) != len(observations):
        raise ValueError('One source-bound profile per observation required')
    by_id = {o['observation_id']: o for o in observations}
    if len(by_id) != len(observations):
        raise ValueError('Observation identities are not unique')
    guarded_masks, result = {}, []
    for profile in profiles:
        observation = by_id.get(profile['observation_id'])
        if observation is None:
            raise ValueError('Profile is not bound to a supplied observation')
        if observation.get('source_size', [foreground.shape[1], foreground.shape[0]]) != [foreground.shape[1], foreground.shape[0]]:
            raise ValueError('Foreground dimensions differ from native observation coordinates')
        points = np.asarray(observation['points_source_px'], float)
        count = len(points)
        if profile['point_array_sha256'] != point_array_sha256(points):
            raise ValueError('Profile source-point hash changed')
        unit = np.asarray(profile['gradient_unit_source_xy'], float)
        reliable = np.asarray(profile['gradient_reliable'], bool)
        reasons = [list(r) for r in profile['unknown_reasons']]
        if unit.shape != points.shape or reliable.shape != (count,) or len(reasons) != count:
            raise ValueError('Profile qualification arrays do not match source points')
        if not np.isfinite(points).all() or not np.isfinite(unit).all():
            raise ValueError('Profile coordinates and gradient directions must be finite')
        sigmas = profile.get('config', {}).get('gradient_sigmas_source_px', [profile['gradient_sigma_source_px']])
        radius = max(_gradient_support_radius(float(s)) for s in sigmas)
        if radius not in guarded_masks:
            guarded_masks[radius] = cv2.dilate(binary, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
        near_gradient = _touches_foreground(guarded_masks[radius], points)
        offsets = np.asarray(profile['profile_offsets_source_px'], float)
        offsets = offsets[abs(offsets) <= 12]
        if not len(offsets):
            raise ValueError('Profile has no declared samples within +/-12 source pixels')
        query = points[:, None, :] + unit[:, None, :] * offsets[None, :, None]
        near_profile = _touches_foreground(binary, query).any(axis=1)
        rejected = near_gradient | near_profile
        qualified = reliable & ~rejected
        for index in np.flatnonzero(rejected):
            if 'person_near_profile' not in reasons[index]:
                reasons[index].append('person_near_profile')
        updated = dict(profile)
        updated.update(gradient_reliable=qualified.tolist(), unknown_reasons=reasons,
            reliable_point_count=int(qualified.sum()),
            foreground_profile_qualification={
                'version': 'native_profile_foreground_guard_v1',
                'profile_sample_limit_source_px': 12,
                'gradient_sigmas_source_px': list(sigmas),
                'gradient_square_support_radius_source_px': radius,
                'support_includes_sobel_and_bilinear_sampling': True,
                'person_near_profile': rejected.tolist(),
                'sample_footprint_touches_foreground': near_profile.tolist(),
                'gradient_support_touches_foreground': near_gradient.tolist(),
                'original_points_and_profile_values_changed': False,
                'original_median_measurements_remain_pre_foreground_qualification': True,
            })
        # Existing median values remain untouched, like every measured array.
        # New fields name aggregates over the smaller qualified set explicitly.
        gradients = np.asarray(profile['gradients_source_xy'], float)
        contrasts = np.asarray(profile['contrast_source_levels'], float)
        updated['foreground_qualified_median_gradient_source_xy'] = np.median(gradients[qualified], axis=0).tolist() if qualified.any() else None
        updated['foreground_qualified_median_contrast_source_levels'] = float(np.median(contrasts[qualified])) if qualified.any() else None
        result.append(updated)
    return result


def native_ring_evidence(image, row, model, *, person_mask_sample=None, maximum_candidates=120):
    """Read visible source edges without a model projection or supplied ring ROI."""
    height,width=image.shape[:2]
    old_ice=polygon_mask(image.shape,row.get('ice_polygons_sample',[]),row.get('sample_size',[960,540]))
    old_people=polygon_mask(image.shape,row.get('foreground_polygons_sample',[]),row.get('sample_size',[960,540]))
    foreground=(old_people.copy() if person_mask_sample is None else
                cv2.resize(person_mask_sample,(width,height),interpolation=cv2.INTER_NEAREST))
    # Broadcast overlays occupy image coordinates, never a physical landmark.
    foreground[int(height*.87):,int(width*.86):]=255
    search=np.full((height,width),255,np.uint8)
    search[:3]=0;search[-3:]=0;search[:,:3]=0;search[:,-3:]=0
    observations,diagnostic=detect_landmarks(image,search,foreground,
        conservative_mask=old_people,sample_size=(960,540),
        config=LandmarkDetectionConfig(maximum_candidates=maximum_candidates))
    profiles=measure_band_profiles(image,observations)
    for item in observations:
        item['observation_id']=f"{row['source_id']}:{row['pts']}:{item['observation_id']}"
        xy=np.asarray(item['points_source_px'],float)
        ij=np.rint(xy).astype(int)
        shape=source_shape(item)
        ids=candidate_model_ids(item,model,exclude_blue=False)
        item.update(source_size=[width,height],source_shape=shape,
            candidate_model_ids=ids,selected_model_id=None,
            association_status='masked_out' if item['association_status']=='masked_out' else 'detected_unidentified',
            used_by_solver=False,solver_role='not_used_identity_unresolved',
            identity_basis=['source_colour_shape_candidates_only_no_pose_no_manual_label'],
            original_ice_mask_support_fraction=float(np.mean(old_ice[ij[:,1],ij[:,0]]>0)),
            original_foreground_mask_fraction=float(np.mean(old_people[ij[:,1],ij[:,0]]>0)),
            surface_status='unclassified_scene_edge_candidate_not_yet_verified_on_ice',
            search_method='native_full_image_edges_independent_of_coarse_ice_and_model',
            rejection_stage='identity_and_surface_pending' if item['association_status']!='masked_out' else 'foreground_mask',
            rejection_reason='actual_edge_detected_physical_identity_and_plane_membership_unresolved' if item['association_status']!='masked_out' else item['rejection_reason'])
    for profile,item in zip(profiles,observations):profile['observation_id']=item['observation_id']
    profiles=qualify_profile_foreground(profiles,observations,foreground)
    return observations,profiles,{
        **diagnostic,'input_size':[width,height], 'native_pixels':True,
        'detection_search_uses_camera':False,'detection_search_uses_reported_ring_roi':False,
        'search_mask_fraction':float(np.mean(search>0)),
        'old_ice_mask_fraction':float(np.mean(old_ice>0)),
        'foreground_source':'lossless_SAM2_sample_mask_nearest_scaled' if person_mask_sample is not None else 'legacy_coarse_mask',
        'counts':{'detected':len(observations),'identity_unresolved':sum(o['association_status']=='detected_unidentified' for o in observations),
                  'masked_out':sum(o['association_status']=='masked_out' for o in observations),'solver_used':0}}


def load_ring_evidence(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8') as handle:return json.load(handle)


def iter_ring_frames(path):
    """Stream the versioned native evidence frames without loading all profiles.

    The recorder writes the top-level frames array last. Limit the metadata
    search and decode complete frame objects; truncated gzip/JSON raises rather
    than silently turning missing evidence into an empty detection result.
    """
    import re
    opener=gzip.open if str(path).endswith('.gz') else open
    decoder=json.JSONDecoder()
    with opener(path,'rt',encoding='utf-8') as handle:
        buffer=''
        while True:
            chunk=handle.read(65536)
            if not chunk:raise ValueError('Native evidence has no frames array')
            buffer+=chunk
            marker=re.search(r'"frames"\s*:\s*\[',buffer)
            if marker:
                # Confirm this is a top-level key, not text nested in metadata.
                metadata=json.loads(buffer[:marker.start()]+'"frames": []}')
                if not isinstance(metadata,dict):raise ValueError('Invalid native evidence metadata')
                buffer=buffer[marker.end():];break
            if len(buffer)>4*1024*1024:raise ValueError('Native evidence metadata exceeds schema limit')
        first=True
        while True:
            buffer=buffer.lstrip()
            while not buffer:
                chunk=handle.read(1024*1024)
                if not chunk:raise ValueError('Truncated native evidence array')
                buffer=chunk.lstrip()
            if buffer.startswith(']'):
                trailer=buffer[1:]+handle.read()
                if trailer.strip()!='}':raise ValueError('Unexpected native evidence trailer')
                return
            if not first:
                if not buffer.startswith(','):raise ValueError('Missing native evidence frame separator')
                buffer=buffer[1:].lstrip()
                while not buffer:
                    chunk=handle.read(1024*1024)
                    if not chunk:raise ValueError('Truncated native evidence frame')
                    buffer=chunk.lstrip()
                if buffer.startswith(']'):raise ValueError('Trailing native evidence frame separator')
            while True:
                try:
                    frame,end=decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    chunk=handle.read(1024*1024)
                    if not chunk:raise ValueError('Truncated native evidence frame')
                    buffer+=chunk
            if not isinstance(frame,dict) or 'pts' not in frame:raise ValueError('Invalid native evidence frame')
            yield frame
            first=False
            buffer=buffer[end:]
