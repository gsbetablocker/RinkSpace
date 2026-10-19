"""Measured advert-detail tracks -> joint observations on any board surface.

Names never identify a physical instance, and no high camera is required.
All surface coordinates are free initial guesses, not measured PPT controls.
"""
from __future__ import annotations

import copy

import numpy as np

from .camera_identity import resolve_frame_camera
from .joint_scene import CORNERS, STRAIGHT_SIDES


def surface_initialization(track, parameters, camera=None, pixel=None):
    side = track['board_side']
    if side not in (*STRAIGHT_SIDES, *CORNERS):
        raise ValueError(f'unknown identified board surface: {side}')
    bounds = parameters['bounds']
    spec = {'side': side, 'initial_z_fraction': .5, 'bounds_z_fraction': [0., 1.]}
    if side in CORNERS:
        spec.update(along_parameterization='corner_fraction', initial_along=.5, bounds_along=[0., 1.])
    else:
        axis = 0 if side in ('upper_long_side', 'lower_long_side') else 1
        spec.update(initial_along=(bounds[axis]+bounds[axis+2])/2.,
                    bounds_along=[bounds[axis]+1e-4, bounds[axis+2]-1e-4])
        if track.get('allow_corner') is not None:
            spec['allow_corner'] = track['allow_corner']
    method = 'surface_midpoint_numeric_initial_guess_only'
    diagnostic = None
    if camera is not None and pixel is not None:
        from .board_intersection import intersect_board_surface
        diagnostic = intersect_board_surface(camera, pixel, side, parameters,
                                              allow_corner=track.get('allow_corner'))
        if diagnostic['valid']:
            xyz = diagnostic['xyz']
            if side in CORNERS:
                spec['initial_along'] = float(diagnostic['along'])
            else:
                spec['initial_along'] = float(np.clip(xyz[axis], *spec['bounds_along']))
            spec['initial_z_fraction'] = float(np.clip(xyz[2]/parameters['board_height'], 0., 1.))
            method = 'conditional_camera_ray_to_identified_surface_initial_guess_only'
    return spec, {'method': method, 'ray_intersection': diagnostic,
                  'original_user_schematic_xyz_used_as_measurement': False}


def assemble_tracks(data, parameters, *, initial_landmarks=(), cameras=None, identity_records=()):
    """Keep fit/check roles, per-installation groups and physical-instance IDs."""
    frames = [resolve_frame_camera(f, data.get('installations', [])) for f in data['frames']]
    by_frame = {f['id']: f for f in frames}
    if len(by_frame) != len(frames):
        raise ValueError('duplicate source frame IDs')
    previous = {p['id']: p for p in initial_landmarks}
    identities = {r['advertisement_instance_id']: r for r in identity_records}
    cameras = cameras or {}
    rows, landmarks, seen = [], [], set()
    for track in data['landmarks']:
        lid, instance = track['track_id'], track['advertisement_instance_id']
        if not instance or not isinstance(instance, str):
            raise ValueError('a physical advertisement_instance_id is required, not a brand name')
        if lid in seen:
            raise ValueError(f'duplicate detail track {lid}')
        seen.add(lid)
        observations = track['observations']
        if not observations:
            raise ValueError(f'{lid}: no measured pixels')
        fit_observations = [o for o in observations if o.get('use', track.get('use', data.get('use'))) == 'fit']
        if not fit_observations and lid not in previous:
            raise ValueError(f'{lid}: check-only detail has no frozen surface point; use the separate frozen-board anchor validation instead of a midpoint guess')
        if not fit_observations and not all(key in previous[lid] for key in ('along','z_fraction')):
            raise ValueError(f'{lid}: check-only detail requires a frozen estimate, not a numerical initializer')
        initialization = None
        if lid in previous:
            initial = copy.deepcopy(previous[lid])
            if initial['advertisement_instance_id'] != instance or initial['side'] != track['board_side']:
                raise ValueError(f'{lid}: previous initializer belongs to another instance/surface')
            for key in ('allow_corner','along_parameterization'):
                if key in track and track[key]!=initial.get(key):
                    raise ValueError(f'{lid}: previous initializer conflicts with {key}')
            initialization = {'method': 'reuse_previous_free_initializer_not_measurement',
                              'original_user_schematic_xyz_used_as_measurement': False}
            if not fit_observations:
                initial['initial_along'] = initial['along']
                initial['initial_z_fraction'] = initial['z_fraction']
                initialization['method'] = 'provided_frozen_conditional_estimate_for_check_only'
        else:
            anchor = next((o for o in observations if o['frame_id'] in cameras and
                           o.get('use', track.get('use', data.get('use'))) == 'fit'), None)
            initial, initialization = surface_initialization(
                track, parameters, cameras.get(anchor['frame_id']) if anchor else None,
                anchor['pixels_uv'] if anchor else None)
        evidence = []
        for observation in observations:
            frame = by_frame[observation['frame_id']]
            use = observation.get('use', track.get('use', data.get('use')))
            if use not in ('fit', 'check'):
                raise ValueError(f'{lid}: explicit fit/check use required')
            uv = np.asarray(observation['pixels_uv'], float)
            if uv.shape != (2,) or not np.isfinite(uv).all():
                raise ValueError(f'{lid}: one measured finite image point is required')
            oid = f"{frame['id']}_{lid}"
            rows.append(dict(id=oid, frame_id=frame['id'], type='point',
                target=dict(kind='board_landmark', landmark_id=lid),
                physical_group_id=instance, advertisement_instance_id=instance, use=use,
                pixels_uv=[uv.tolist()], sigma_px=observation['sigma_px'],
                correlation_group_id=f"{instance}:{frame['installation_id']}",
                measurement_origin='image_measured_static_detail', method=observation.get('method'),
                source_pts=frame['source_pts'], time_base=frame['time_base'],
                identity_evidence=copy.deepcopy(identities.get(instance, track.get('identity_evidence'))),
                tracking_diagnostics={k: copy.deepcopy(v) for k,v in observation.items()
                                      if k not in ('pixels_uv','frame_id')}))
            if use == 'fit':
                evidence.append(oid)
        initial.update(id=lid, advertisement_instance_id=instance, fit=bool(evidence),
                       evidence_observation_ids=evidence)
        initial['provenance'] = {**initial.get('provenance', {}), 'initialization': initialization,
                                 'identity_evidence': copy.deepcopy(identities.get(instance, track.get('identity_evidence')))}
        landmarks.append(initial)
    used_frames = {o['frame_id'] for o in rows}
    return dict(schema_version='rinkspace.board-detail-observations.v2',
                frames=[f for f in frames if f['id'] in used_frames], observations=rows,
                board_landmarks=landmarks, installations=copy.deepcopy(data.get('installations', [])),
                unknown_board_points_are_joint_variables=True,
                supported_board_surfaces=list(STRAIGHT_SIDES+CORNERS),
                fit_observation_count=sum(o['use']=='fit' for o in rows),
                check_observation_count=sum(o['use']=='check' for o in rows),
                weighting='Per physical advertisement instance and installation; tracked pixels remain correlated',
                shot_audits=copy.deepcopy(data.get('shot_audits', [])))
