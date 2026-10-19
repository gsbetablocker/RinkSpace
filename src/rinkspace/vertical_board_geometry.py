"""Conditional ray intersection with a bounded straight vertical board surface.

The scene plane supplies depth. Pure camera rotation supplies no triangulated
depth, and these inferred board locations are never measured ice points.
Curved rink ends must be separate bounded segments or remain unknown.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .multisurface_camera import Camera, project_camera, world_rays


@dataclass(frozen=True)
class VerticalBoardSegment:
    segment_id: str
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    geometry_kind: str = 'straight_vertical_segment'
    geometry_status: str = 'conditional_on_frozen_rink_geometry_not_surveyed'

    def __post_init__(self):
        if self.geometry_kind != 'straight_vertical_segment':
            raise ValueError('Curved boards cannot be intersected as one straight plane')
        values=np.asarray([self.start_xy,self.end_xy],float)
        if values.shape!=(2,2) or not np.isfinite(values).all():
            raise ValueError('A bounded board needs finite XY endpoints')
        if np.linalg.norm(values[1]-values[0])<1e-9:
            raise ValueError('A bounded board segment needs nonzero length')

    def to_dict(self):
        return {'segment_id':self.segment_id,'start_xy':list(self.start_xy),
          'end_xy':list(self.end_xy),'geometry_kind':self.geometry_kind,
          'geometry_status':self.geometry_status,'vertical_axis':'model_+Z',
          'ground_height':0.,'units':'conditional_rink_width_100_not_metres'}


def far_side_straight_segment(model):
    """Use only the straight part of frozen far_side, excluding both corners."""
    values={k:p['value'] for k,p in model['parameter_estimates'].items()}
    half_length=float(values['half_length']);radius=float(values['corner_radius'])
    if not 0<radius<half_length:
        raise ValueError('Invalid conditional rounded-rectangle parameters')
    feature=next((f for f in model['features'] if f['id']=='far_side'),None)
    if feature is None or feature.get('kind')!='axis_line' or feature.get('axis')!='y' or feature.get('coordinate')!=50:
        raise ValueError('This helper requires explicit straight far_side at Y=50')
    return VerticalBoardSegment('far_side_straight',(-half_length+radius,50.),(half_length-radius,50.))


def intersect_reference_board(camera: Camera, pixels, segment: VerticalBoardSegment, *,
                              height_interval=None, endpoint_margin=0.):
    """Infer XYZ only by combining a raw pixel ray and a declared board plane.

    Positive height is required. An optional height_interval is a caller-declared
    search bound, never a measured board height or confidence interval. Invalid
    rows are returned as null XYZ and carry a concrete reason.
    """
    if segment.geometry_kind!='straight_vertical_segment':
        raise ValueError('Curved boards must not use a single plane')
    pixels=np.asarray(pixels,float).reshape(-1,2)
    if endpoint_margin<0:
        raise ValueError('Endpoint margin cannot expand the known straight region')
    if height_interval is not None:
        lower,upper=map(float,height_interval)
        if not 0<=lower<upper or not np.isfinite([lower,upper]).all():
            raise ValueError('Height interval must be finite and positive')
    a,b=np.asarray(segment.start_xy,float),np.asarray(segment.end_xy,float)
    direction=b-a;length=np.linalg.norm(direction);tangent=direction/length
    if 2*endpoint_margin>=length:
        raise ValueError('Endpoint margin removes the entire straight segment')
    normal=np.array([-tangent[1],tangent[0],0.])
    plane_origin=np.r_[a,0.]
    rays,ray_valid=world_rays(camera,pixels)
    denominator=rays@normal
    depth=np.divide((plane_origin-camera.centre)@normal,denominator,
                    out=np.full(len(rays),np.nan),where=np.abs(denominator)>1e-9)
    xyz=camera.centre+depth[:,None]*rays
    along=(xyz[:,:2]-a)@tangent
    rows=[]
    for index,point in enumerate(xyz):
        reason=None
        if not ray_valid[index] or not np.isfinite(pixels[index]).all():
            reason='invalid_raw_pixel_or_distortion_branch'
        elif abs(denominator[index])<=1e-9:
            reason='ray_parallel_to_board_plane'
        elif not np.isfinite(point).all() or depth[index]<=0:
            reason='board_plane_behind_camera'
        elif not endpoint_margin<=along[index]<=length-endpoint_margin:
            reason='outside_known_straight_segment_corner_or_other_surface'
        elif point[2]<=0:
            reason='nonpositive_height_not_board_surface'
        elif height_interval is not None and not lower<=point[2]<=upper:
            reason='outside_declared_conditional_height_search_bound'
        rows.append({'reference_xy':pixels[index].tolist(),'valid':reason is None,
                     'world_xyz':point.tolist() if reason is None else None,
                     'reason':reason,'surface_type':'board_surface',
                     'representation':'world_points',
                     'geometry_status':'conditional_plane_ray_intersection_not_measured_or_triangulated',
                     'height_status':'conditional_estimate_not_measured'})
    return {'segment':segment.to_dict(),'camera':camera.to_dict(),'observations':rows,
            'height_interval':None if height_interval is None else list(height_interval),
            'height_interval_status':'declared_search_bound_not_measurement',
            'endpoint_margin':float(endpoint_margin),
            'board_used_as_ice':False,'depth_source':'assumed_board_plane_and_reference_camera',
            'pure_rotation_triangulation':False}


def reproject_board(camera: Camera, records):
    """Use the same camera projection as ground points, with actual board Z."""
    rows=records['observations'] if isinstance(records,dict) else records
    projected=[]
    for row in rows:
        if row.get('surface_type')!='board_surface' or row.get('representation')!='world_points':
            raise ValueError('Expected typed elevated board XYZ observations')
        if not row.get('valid') or row.get('world_xyz') is None:
            projected.append({'valid':False,'image_xy':None,'reason':row.get('reason','invalid_board_location')})
            continue
        point=np.asarray(row['world_xyz'],float)
        if point.shape!=(3,) or not np.isfinite(point).all() or point[2]<=0:
            raise ValueError('Board XYZ must have finite positive height, never Z=0')
        xy,valid=project_camera(camera,[point],rays=False)
        projected.append({'valid':bool(valid[0]),'image_xy':xy[0].tolist() if valid[0] else None,
                          'reason':None if valid[0] else 'outside_camera_projection_domain'})
    return projected


def triangulation_observability(camera_a: Camera, pixels_a, camera_b: Camera, pixels_b):
    """Report whether translated-view triangulation is even observable.

    This diagnostic does not triangulate or manufacture a depth from pure pan.
    Baseline is in conditional rink units and is not a physical accuracy bound.
    """
    a,valid_a=world_rays(camera_a,pixels_a);b,valid_b=world_rays(camera_b,pixels_b)
    if len(a)!=len(b):
        raise ValueError('Corresponding pixel arrays must have equal length')
    baseline=float(np.linalg.norm(camera_b.centre-camera_a.centre))
    a=a/np.linalg.norm(a,axis=1)[:,None];b=b/np.linalg.norm(b,axis=1)[:,None]
    angles=np.rad2deg(np.arccos(np.clip(np.sum(a*b,axis=1),-1,1)))
    same_center=baseline<=1e-9
    return {'baseline_model_units':baseline,'same_optical_center':same_center,
            'depth_observable':False if same_center else None,
            'reason':'pure_rotation_cannot_triangulate_depth' if same_center else 'nonzero_baseline_requires_further_parallax_and_uncertainty_checks',
            'ray_angle_degrees':[float(v) if ok else None for v,ok in zip(angles,valid_a&valid_b)],
            'returned_depth':None}
