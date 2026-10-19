"""Physical camera identity routing, separate from conditional pose estimates."""
from __future__ import annotations

import copy


CAMERA_IDS = ('camera_main_long_side', 'camera_secondary_long_side',
              'camera_low_end', 'camera_high_oblique')
WOMEN_INSTALLATIONS = dict(zip(CAMERA_IDS, ('women_sp_main', 'women_sp_secondary',
                                          'women_sp_low', 'women_sp_high')))
ALIASES = {key: camera for camera, installation in WOMEN_INSTALLATIONS.items()
           for key in (camera, camera.removeprefix('camera_'), installation)}
ALIASES.update(high_oblique_overview='camera_high_oblique')
LEGACY_INSTALLATIONS = {
    'oxXu1avhrvs_camera_main_long_side_observed_keyframes': 'women_sp_main',
}


def normalized_installations(installations):
    result, seen = [], set()
    for original in installations:
        record = copy.deepcopy(original)
        record['id'] = LEGACY_INSTALLATIONS.get(record['id'], record['id'])
        if record['id'] in seen:
            raise ValueError('duplicate installation declarations after alias normalization')
        seen.add(record['id'])
        result.append(record)
    return result


def resolve_frame_camera(frame, installations=()):
    """Preserve explicit installations; never infer 'main' from 'not high'.

    A recognized installation ID can fill a missing camera field. This records
    a naming inference only; independent source-shot evidence remains separate.
    Unknown or conflicting identities fail visibly instead of entering a fit.
    """
    registry = {i['id']: i for i in normalized_installations(installations)}
    claims = {}
    for key in ('physical_camera_id', 'camera_id'):
        value = frame.get(key)
        if value is not None:
            if value not in ALIASES:
                raise ValueError(f"{frame.get('id')}: unresolved {key}={value!r}")
            claims[key] = ALIASES[value]
    iid = LEGACY_INSTALLATIONS.get(frame.get('installation_id'), frame.get('installation_id'))
    if iid in registry:
        physical = registry[iid].get('physical_camera_id')
        if physical not in ALIASES:
            raise ValueError(f'{iid}: installation needs a specific physical_camera_id')
        claims['installation_registry'] = ALIASES[physical]
    elif iid in ALIASES:
        claims['installation_id'] = ALIASES[iid]
    if not claims:
        raise ValueError(f"{frame.get('id')}: missing a specific physical camera identity")
    if len(set(claims.values())) != 1:
        raise ValueError(f"{frame.get('id')}: conflicting camera identities {claims}")
    camera = next(iter(claims.values()))
    result = copy.deepcopy(frame)
    result.setdefault('original_camera_identity_fields',
                      {key: frame.get(key) for key in ('camera_id', 'physical_camera_id', 'installation_id')})
    result['camera_id'] = result['physical_camera_id'] = camera
    generic_installation_name = iid in ALIASES and iid not in WOMEN_INSTALLATIONS.values() and iid not in registry
    if iid is None or generic_installation_name:
        source = frame.get('source_id')
        if not source:
            raise ValueError(f"{frame.get('id')}: source_id or explicit installation_id required")
        iid = (WOMEN_INSTALLATIONS[camera] if source == 'oxXu1avhrvs'
               else f'{source}:{camera}:installation_unverified')
    result['installation_id'] = iid
    result['camera_identity_resolution'] = {
        'claims': claims, 'method': 'explicit_alias_or_installation_registry',
        'independent_visual_verification': False,
        'note': 'Naming resolution is not source-shot verification or surveyed camera position.'}
    return result


def installation_records(frames, supplied=(), legacy_main_center=None, start=0):
    """One independent center per installation; defaults are free initial guesses."""
    supplied = normalized_installations(supplied)
    registry = {i['id']: copy.deepcopy(i) for i in supplied}
    if len(registry) != len(supplied):
        raise ValueError('duplicate installation declarations')
    groups = {}
    for frame in frames:
        normalized = resolve_frame_camera(frame, supplied)
        iid, camera = normalized['installation_id'], normalized['physical_camera_id']
        if iid in groups and groups[iid] != camera:
            raise ValueError(f'{iid}: multiple physical cameras cannot share an installation')
        groups[iid] = camera
    main = list(legacy_main_center) if legacy_main_center is not None else [0., -80., 8.]
    guesses = {
        'camera_main_long_side': (main, [[-100., -500., .25], [100., -50.001, 200.]]),
        'camera_secondary_long_side': ([53., -80., 8.], [[-200., -500., .25], [300., -50.001, 200.]]),
        'camera_low_end': ([-120., -32., 2.], [[-500., -200., .1], [-100.001, 200., 100.]]),
        'camera_high_oblique': ([100.9435315, -58.8067769, 24.8013901], [[70., -400., 5.], [400., -50.001, 250.]]),
    }
    result = []
    for iid, camera in groups.items():
        if iid in registry:
            record = registry[iid]
            record['physical_camera_id'] = camera
        else:
            center, bounds = copy.deepcopy(guesses[camera])
            if start and camera == 'camera_main_long_side':
                center = [-10., -95., 12.]
            if start and camera == 'camera_high_oblique':
                center = [120., -90., 40.]
            record = dict(id=iid, physical_camera_id=camera, initial_center_xyz=center,
                          fit_center=True, center_bounds_xyz=bounds,
                          position_scope='Only the explicitly named physical camera and installation epoch',
                          initialization_status='conditional_free_initial_guess_NOT_measured_position',
                          initialization_basis=('historical conditional candidate' if camera in
                                                ('camera_main_long_side', 'camera_high_oblique') else
                                                'user-confirmed approximate side/direction only; numeric coordinates are guesses'))
            if camera == 'camera_high_oblique':
                record['radial_distortion'] = dict(initial_k1_image=-.4664955, fit=True,
                                                   bounds=[-1.1, .3], scale_image=1920.)
        result.append(record)
    return result
