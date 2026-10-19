"""Admit native advertisement corner intervals without changing source pixels.

The caller supplies the shared model's actual corners and an explicit physical
corner permutation. Image clockwise labels alone do not select world corners.
"""
from pathlib import Path
from copy import deepcopy
import hashlib,json
import cv2,numpy as np
from z190_continuous_source_bundle import SOURCE_SHA


def admit_native_ad_corner_intervals(path,aliases,world_corners,image_corner_to_world_role,
                                    *,expected_observer,expected_sha256=None):
    path=Path(path);data=path.read_bytes();sha=hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and sha!=expected_sha256:
        raise ValueError('Frozen advertisement corner source changed')
    source=json.loads(data.decode('utf8'))
    if source['source_video_sha256']!=SOURCE_SHA or source['source_pts_timebase']!=[1,15360]:
        raise ValueError('Advertisement source video/native clock mismatch')
    if source['native_size_wh']!=[1920,1080]or source['observer_installation']!=expected_observer:
        raise ValueError('Advertisement observing installation/native dimensions changed')
    if source['selected_source_before_projection_or_new_fit']is not True:
        raise ValueError('Advertisement corner admission must precede model fitting')
    roles=image_corner_to_world_role
    if set(roles)!={'TL','TR','BR','BL'} or len(set(roles.values()))!=4:
        raise ValueError('Explicit physical corner permutation required')
    corners={name:np.asarray(value,float)for name,value in world_corners.items()}
    if any(roles[name]not in corners for name in roles):
        raise ValueError('A source-visible corner is missing in the shared model')
    if any(corners[roles[name]].shape!=(3,)or not np.isfinite(corners[roles[name]]).all()for name in roles):
        raise ValueError('Finite shared world corners required')
    frames={int(row['source_native_pts']):row for row in source['frames']}
    checked={};groups={};held=set()
    for original in source['source_rows']:
        row=deepcopy(original);native=int(row['source_native_pts']);actual=int(row['source_actual_pts'])
        if int(aliases[native])!=actual or row['split']not in('fit','held'):
            raise ValueError('Ad corner actual exposure/split mismatch')
        if row['source_visible']is not True or row['corner_manually_qualified']is not True or row['corner_occluded']is not False:
            raise ValueError('An occluded/inferred corner cannot become an observed target')
        if row['model_projection_used_to_generate_or_select_uv']is not False or row['frozen_before_fit']is not True:
            raise ValueError('Ad corner pixels must be independently frozen source evidence')
        bounds=np.asarray(row['native_support_bounds_xy_px'],float)
        uv=np.asarray(row['uv'],float);sigma=np.asarray(row['sigma_native_px'],float)
        if bounds.shape!=(2,2)or uv.shape!=(2,)or sigma.shape!=(2,)or not np.isfinite(np.r_[bounds.ravel(),uv,sigma]).all():
            raise ValueError('Finite source UV, XY support and engineering sigma required')
        if np.any(bounds[:,0]>bounds[:,1])or np.any(uv<bounds[:,0])or np.any(uv>bounds[:,1])or np.any(sigma<=0):
            raise ValueError('Corner locator must remain inside its positive-scale source interval')
        # Existing absolute rows use one isotropic native scale per point.
        if sigma[0]!=sigma[1]:raise ValueError('Absolute corner adapter requires the original isotropic XY scale')
        frame=frames[native]
        if frame['source_actual_pts']!=actual or frame['split']!=row['split']:
            raise ValueError('Frozen frame/corner role mismatch')
        raw=Path(row['raw_png'])
        if str(raw)not in checked:
            if hashlib.sha256(raw.read_bytes()).hexdigest()!=row['raw_png_sha256']:
                raise ValueError('Native advertisement source PNG changed')
            pixels=cv2.imread(str(raw),cv2.IMREAD_COLOR)
            if pixels is None or pixels.shape!=(1080,1920,3)or hashlib.sha256(pixels.tobytes()).hexdigest()!=row['native_bgr_sha256']:
                raise ValueError('Native advertisement pixels/dimensions changed')
            checked[str(raw)]=row['raw_png_sha256']
        key=(actual,row['split']);group=groups.setdefault(key,[])
        if any(r['image_corner_clockwise']==row['image_corner_clockwise']for r in group):
            raise ValueError('Duplicate physical corner for one actual exposure')
        group.append(row)
        if row['split']=='held':held.add(actual)
    fit_groups=[];held_groups=[]
    for (actual,split),rows in sorted(groups.items()):
        rows.sort(key=lambda r:('TL','TR','BR','BL').index(r['image_corner_clockwise']))
        world_roles=[roles[r['image_corner_clockwise']]for r in rows]
        assembled=dict(id=f'Ad{source["physical_ad_id"]}_qualified_corners_{actual}',pts=actual,split=split,
            physical_ad_id=source['physical_ad_id'],point_correspondence=True,
            curve_observation_kind='native_ad_corner_interval',absolute_support_volume_quadratic=True,
            world=np.asarray([corners[role]for role in world_roles]),
            uv=np.asarray([r['uv']for r in rows],float),sigma=np.asarray([r['sigma_native_px'][0]for r in rows]),
            point_support_bounds_xy_px=np.asarray([r['native_support_bounds_xy_px']for r in rows],float),
            source_native_pts=[r['source_native_pts']for r in rows],source_rows=rows,
            physical_world_corner_roles=world_roles,image_corner_to_world_role=dict(roles),
            source_qualification=dict(path=str(path),sha256=sha,identity_authority=source['identity_authority']),
            interval_locator_is_not_exact_corner=True,world_geometry_is_shared_actual_model=True)
        (fit_groups if split=='fit'else held_groups).append((actual,[assembled]))
    if any(p in held for p,_ in fit_groups):raise ValueError('Held actual aliases cannot become fit')
    manifest=dict(path=str(path),sha256=sha,physical_ad_id=source['physical_ad_id'],
        native_pixels_and_sigma_roles_unchanged=True,raw_native_sources_verified=len(checked),
        fit_actual=len(fit_groups),fit_corners=sum(len(r['uv'])for _,rows in fit_groups for r in rows),
        held_actual=len(held_groups),held_corners=sum(len(r['uv'])for _,rows in held_groups for r in rows),
        explicit_world_corner_permutation=dict(roles),point_support_zero_inside_original_interval=True)
    return fit_groups,held_groups,manifest
