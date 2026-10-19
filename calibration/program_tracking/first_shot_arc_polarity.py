"""Original-RGB material-side evidence for white holes in dark print.

Source gradients are measured once on authenticated native pixels. They are
association eligibility, never new UV, an independent point or a fitted normal.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np


def source_light_gradients(image, uv):
    import cv2
    image=np.asarray(image);uv=np.asarray(uv,float)
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('Original uint8 native BGR required')
    if uv.ndim!=2 or uv.shape[1]!=2 or not np.isfinite(uv).all():
        raise ValueError('Original finite source UV required')
    if (np.any(uv<1.) or np.any(uv[:,0]>image.shape[1]-2) or np.any(uv[:,1]>image.shape[0]-2)):
        raise ValueError('Original source gradient stencil leaves native sensor')
    gray=image.astype(np.float64)@np.array([.114,.587,.299])
    gradient=[cv2.Sobel(gray,cv2.CV_64F,dx,dy,ksize=3,scale=.125)for dx,dy in ((1,0),(0,1))]
    x=uv[:,0].astype(np.float32)[:,None];y=uv[:,1].astype(np.float32)[:,None]
    values=np.column_stack([cv2.remap(g,x,y,cv2.INTER_LINEAR).ravel()for g in gradient])
    # Each normalized Sobel kernel has L1 norm 1; nonnegative bilinear
    # interpolation preserves the half-gray-level input quantization bound.
    # This is NOT a calibrated optical/localization/noise covariance.
    return values, .5


def material_side_nearest(observed,projected,segment_valid,gradients,component_bound):
    observed=np.asarray(observed,float);p=np.asarray(projected,float)
    g=np.asarray(gradients,float);bound=np.asarray(component_bound,float)
    if (p.ndim!=3 or p.shape[0]!=len(observed) or p.shape[2]!=2 or g.shape!=observed.shape
            or bound.shape!=(len(observed),) or not np.isfinite(np.r_[g.ravel(),bound]).all()
            or np.any(bound<0) or not np.array_equal(p[:,0],p[:,-1])):
        raise ValueError('One measured gradient and one closed physical white-hole path per source row required')
    direction=np.diff(p,axis=1);length2=np.sum(direction*direction,axis=2)
    area=np.sum(p[:,:-1,0]*p[:,1:,1]-p[:,:-1,1]*p[:,1:,0],axis=1)
    if np.any(~np.isfinite(area))or np.any(area==0):
        raise ValueError('Physical white-hole projected interior is unresolved')
    normal=np.stack((-direction[:,:,1],direction[:,:,0]),axis=2)
    normal*=np.sign(area)[:,None,None]/np.sqrt(np.maximum(length2,1e-300))[:,:,None]
    score=np.einsum('nej,nj->ne',normal,g)
    uncertainty=bound[:,None]*np.sum(abs(normal),axis=2)
    eligible=np.asarray(segment_valid,bool)&(length2>1e-4)&(score>uncertainty)
    if np.any(~eligible.any(axis=1)):
        error=ValueError('No finite model boundary has the original RGB white-inside material side')
        error.report=dict(source_rows=np.flatnonzero(~eligible.any(axis=1)).tolist(),gradient=g.tolist(),
            component_bound=bound.tolist(),projected_area=area.tolist(),valid_edge_counts=np.asarray(segment_valid).sum(axis=1).tolist(),
            maximum_side_score=score.max(axis=1).tolist(),minimum_quantization_bound=uncertainty.min(axis=1).tolist(),
            projected=p.tolist(),segment_valid=np.asarray(segment_valid).tolist())
        raise error
    t=np.clip(np.sum((observed[:,None]-p[:,:-1])*direction,axis=2)/np.maximum(length2,1e-300),0.,1.)
    feet=p[:,:-1]+t[:,:,None]*direction;cost=np.sum((feet-observed[:,None])**2,axis=2);cost[~eligible]=np.inf
    selected=np.argmin(cost,axis=1)
    return selected,feet[np.arange(len(observed)),selected],dict(
        association_method='original_RGB_white_inside_side_qualified_nearest',
        source_points=len(observed),world_edges=cost.shape[1],
        original_squared_pixel_distance_sum=float(cost[np.arange(len(observed)),selected].sum()),
        source_arc_order_used_as_material_orientation=False,source_order_or_world_winding_invariant=True,
        eligible_edge_counts=eligible.sum(axis=1).tolist(),
        minimum_selected_material_side_margin=float(np.min((score-uncertainty)[np.arange(len(observed)),selected])))


def bind_material_side_evidence(frames,manifest_reference,*,require_all=True):
    from .first_shot_source_transition import read,reference
    manifest=read(manifest_reference)
    if (manifest.get('schema')!='rinkspace.original-RGB-white-hole-side.v1'
            or manifest.get('producer')!=reference(__file__)
            or manifest.get('source_UV_ID_W_and_native_PTS_unchanged')is not True):
        raise ValueError('Explicit original-RGB white-hole-side evidence required')
    for ref in manifest['evidence_references']:
        if reference(ref['path'])!=ref:raise ValueError('Original source material-side evidence changed')
    lookup={(int(x['native_pts']),x['source_point_id']):x for x in manifest['records']}
    if len(lookup)!=len(manifest['records']):raise ValueError('Repeated source material-side evidence')
    used=set()
    for frame in frames:
        for row in frame['rows']:
            for j,pid in enumerate(row['source_point_ids']):
                address=int(frame['native_pts']),pid
                if address not in lookup:continue
                item=lookup[address]
                if (len(row['uv'])!=1 or j!=0 or row['source_packet']!=item['source_packet']
                        or not np.array_equal(row['uv'][0],item['source_uv'])
                        or row['target'].get('kind')!='ad_print'
                        or row['target'].get('advertisement_id')!=item['advertisement_id']
                        or row['target'].get('contour_key')!=item['contour_key']
                        or row['target'].get('contour_role')!='printed_internal_hole_boundary'):
                    raise ValueError('Material-side evidence changed its original UV, source packet or verified white-hole identity')
                row['_source_material_side']=dict(gradient=deepcopy(item['gradient']),component_bound=float(item['component_bound']),
                    reference=deepcopy(manifest_reference),material='white_hole_inside_dark_print')
                used.add(address)
    if require_all and used!=set(lookup):raise ValueError('Some original material-side evidence is absent from current source')
    return dict(manifest=deepcopy(manifest_reference),source_points=len(used),original_UV_ID_W_and_PTS_unchanged=True,
                source_order_does_not_certify_material_direction=True)
