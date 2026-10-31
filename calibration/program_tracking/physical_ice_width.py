"""Current source profiles across the physical width of straight ice paint.

The scene and Brown camera define a transverse search direction only. Both paint
interfaces are still read from the current image. A physical midpoint is formed
between those measured interfaces on z=0, then projected back to native pixels.
Circles and board material boundaries deliberately do not use this operator.
"""
from __future__ import annotations
import cv2
import numpy as np
from .projection import ground_xy,project_fast


def _ground(camera,pixels):
    return camera.ground_xy(pixels) if hasattr(camera,'ground_xy') else ground_xy(camera,pixels)


def _project(camera,xyz):
    return camera.project(xyz) if hasattr(camera,'project') else project_fast(camera,xyz)


def physical_width_axis(path):
    """World-plane unit axis perpendicular to this named physical straight line."""
    xyz=np.asarray(path['xyz'],float)
    tangent=xyz[-1,:2]-xyz[0,:2]
    length=float(np.linalg.norm(tangent))
    if not np.isfinite(length) or length<1e-9:
        raise ValueError('Physical width needs a finite straight line')
    tangent/=length
    return np.array([-tangent[1],tangent[0]])


def projected_width_profile(camera,path,query_uv,*,image_hw,radius_px=90.,step_px=1.):
    """Project a dense *physical* transverse section around a source search UV.

    ``offsets`` are search coordinates in locally scaled source pixels; they are
    not physical width measurements. ``world_offsets`` and ``source_uv`` retain
    the exact Brown-projected curve, including its nonuniform perspective scale.
    ``image_hw`` is the actual decoded RGB shape, never a nominal source size.
    """
    image_hw=np.asarray(image_hw)
    if image_hw.shape!=(2,) or image_hw.dtype.kind not in 'iu' or np.any(image_hw<=0):
        raise ValueError('Physical width requires positive integer decoded image height/width')
    h,w=map(int,image_hw)
    query=np.asarray(query_uv,float).reshape(2)
    xy,valid=_ground(camera,query[None])
    if not valid[0]:return None
    axis=physical_width_axis(path)
    probe,_=_project(camera,np.c_[xy[0]+np.array([-.01,.01])[:,None]*axis,np.zeros(2)])
    derivative=(probe[1]-probe[0])/.02
    scale=float(np.linalg.norm(derivative))
    if not np.isfinite(scale) or scale<1e-6:return None
    offsets=np.arange(-radius_px,radius_px+step_px*.5,step_px)
    world_offsets=offsets/scale
    world_xy=xy[0]+world_offsets[:,None]*axis
    uv,depth=_project(camera,np.c_[world_xy,np.zeros(len(world_xy))])
    valid=np.isfinite(uv).all(1)&(depth>.05)&(uv>=[0.,0.]).all(1)&(uv<=[w-1,h-1]).all(1)
    return dict(query_uv=query,query_world_xy=xy[0],physical_width_axis_xy=axis,
                query_width_direction_uv=derivative/scale,offsets=offsets,
                world_offsets=world_offsets,world_xy=world_xy,source_uv=uv,valid=valid,image_hw=(h,w))


def source_at_profile_offsets(profile,offsets):
    values=np.asarray(offsets,float)
    uv=np.asarray(profile['source_uv'])
    return np.stack([np.interp(values,profile['offsets'],uv[:,axis])for axis in (0,1)],axis=-1)


def ground_midpoint_from_source_sides(camera,sides_uv):
    sides=np.asarray(sides_uv,float).reshape(2,2)
    xy,valid=_ground(camera,sides)
    if not valid.all():return None
    center_xy=xy.mean(0)
    center,depth=_project(camera,np.r_[center_xy,0.][None])
    if not np.isfinite(center).all() or depth[0]<=.05:return None
    return center[0],center_xy,xy


def sample_physical_width_field(rgb,profile,signal='dark',field=None):
    """Read native RGB on the exact projected section; no predicted values."""
    if tuple(np.asarray(rgb).shape[:2])!=profile['image_hw']:
        raise ValueError('Physical profile and decoded RGB image dimensions differ')
    if field is not None and np.asarray(field).shape!=profile['image_hw']:
        raise ValueError('Physical profile field and decoded RGB image dimensions differ')
    if field is None:
        rgb=np.asarray(rgb,np.float32)
        if signal=='dark':field=.2126*rgb[...,0]+.7152*rgb[...,1]+.0722*rgb[...,2]
        elif signal=='blue':field=rgb[...,2]-.5*(rgb[...,0]+rgb[...,1])
        elif signal=='red':field=rgb[...,0]-.5*(rgb[...,1]+rgb[...,2])
        else:raise ValueError('Unsupported source material signal')
    uv=np.asarray(profile['source_uv'],np.float32)
    return cv2.remap(field,uv[:,0][None],uv[:,1][None],cv2.INTER_LINEAR)[0]


def valid_profile_contributors(profile,exclusion_mask):
    """Require every bilinear source contributor in-frame and unexcluded."""
    exclusion_mask=np.asarray(exclusion_mask,dtype=bool)
    if exclusion_mask.shape!=profile['image_hw']:
        raise ValueError('Physical profile exclusion mask and decoded RGB image dimensions differ')
    uv=np.asarray(profile['source_uv'],float)
    finite=np.isfinite(uv).all(1)
    floor=np.floor(np.where(np.isfinite(uv),uv,-100.)).astype(int)
    h,w=exclusion_mask.shape
    valid=np.asarray(profile['valid'],bool)&finite&(floor>=[0,0]).all(1)&(floor<=[w-2,h-2]).all(1)
    ids=np.flatnonzero(valid)
    for dy,dx in ((0,0),(0,1),(1,0),(1,1)):
        valid[ids]&=~exclusion_mask[floor[ids,1]+dy,floor[ids,0]+dx]
    return valid


def restore_physical_profile_readout(camera,profile,scalar_row):
    """Convert a measured scalar two-interface profile into native evidence.

    Scalar readers are called at point=(0,0), normal=(1,0). Their first UV
    coordinate is an actually measured profile level crossing, not a scene UV.
    """
    from copy import deepcopy
    row=deepcopy(scalar_row);e=row['source_evidence']
    scalar_sides=np.asarray(e['actual_source_sides_uv'])[:,0]
    scalar_quantiles=np.asarray(e['actual_source_side_quantiles_uv'])[...,0]
    sides=source_at_profile_offsets(profile,scalar_sides)
    quantiles=source_at_profile_offsets(profile,scalar_quantiles)
    result=ground_midpoint_from_source_sides(camera,sides)
    if result is None:return None
    center,world_center,world_sides=result
    e.update(query_uv=profile['query_uv'].tolist(),profile_normal=profile['query_width_direction_uv'].tolist(),
             actual_source_sides_uv=sides.tolist(),actual_source_side_quantiles_uv=quantiles.tolist(),
             native_profile_shift_px=float((center-profile['query_uv'])@profile['query_width_direction_uv']),
             physical_width_axis_xy=profile['physical_width_axis_xy'].tolist(),
             actual_measured_sides_ground_xy=world_sides.tolist(),actual_measured_midpoint_ground_xy=world_center.tolist(),
             width_midpoint_operator='inverse_Brown_z0_measured_two_interfaces_ground_average_then_native_Brown_projection',
             width_profile_operator='exact_Brown_projection_of_physical_ice_plane_transverse_section',
             scene_camera_used_only_for_width_direction_and_transverse_midpoint_coordinate_conversion=True,
             measured_boundary_pixels_from_current_source=True,
             physical_midpoint_minus_image_arithmetic_midpoint_px=(center-sides.mean(0)).tolist())
    row['uv']=np.round(center,4).tolist()
    return row
