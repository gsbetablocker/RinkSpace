"""Pre-estimator native support admission for versioned source rereads.

This is the same original sample_chroma and current-worker stencil as source_worker_v2.
It rejects source support before line/circle family estimation, preserving every
failed row. No image, mask, raw coordinate or ROI is changed.
"""
from __future__ import annotations
import numpy as np

def _contexts(uv, evidence):
    values=[np.asarray(uv,dtype=float).reshape(1,2)]
    for key in ('query_uv','actual_source_sides_uv','actual_source_side_quantiles_uv'):
        if key in evidence:
            a=np.asarray(evidence[key],dtype=float).reshape(-1,2)
            if not np.isfinite(a).all(): raise ValueError('Nonfinite source profile context')
            values.append(a)
    return np.vstack(values)

def _between(points):
    points=np.asarray(points,float).reshape(-1,2)
    if len(points)<2:return points
    # All original profile contexts lie on the profile normal. Pick the farthest
    # pair without assuming quantiles are a sigma or a new acceptance bound.
    distances=np.sum((points[:,None,:]-points[None,:,:])**2,axis=2)
    i,j=np.unravel_index(int(np.argmax(distances)),distances.shape)
    n=max(1,int(np.ceil(np.sqrt(distances[i,j])/.25)))
    return points[i]+np.linspace(0.,1.,n+1)[:,None]*(points[j]-points[i])


def support_pixel_audit(sites,mask,local_records=(),*,gaussian_radius=3):
    """Exact finite stencil of v1 Gaussian(.8, truncate4)+order-1 sampling.

    This evaluates the reader's existing stencil, not an expanded mask/ROI.
    Gaussian reflect boundary conditions and bilinear zero weights are explicit.
    """
    sites=np.asarray(sites,float).reshape(-1,2);h,w=mask.shape
    if not len(sites):return {'sites':0,'status':'missing_reader_support','person_mask_pixels':0,'static_exclusion_ids':[]}
    if not np.isfinite(sites).all():raise ValueError('Nonfinite reader sites')
    outside=bool(((sites<0)|(sites>np.array([w-1,h-1]))).any())
    q=np.clip(sites,[0,0],[w-1,h-1]);floor=np.floor(q).astype(int)
    # ceil==floor on integer coordinates, avoiding falsely including zero-weight taps.
    ceil=np.ceil(q).astype(int)
    centers=np.vstack([np.column_stack((xs,ys)) for xs in (floor[:,0],ceil[:,0]) for ys in (floor[:,1],ceil[:,1])])
    centerids=np.unique(centers[:,1]*w+centers[:,0]);centers=np.column_stack((centerids%w,centerids//w))
    off=np.asarray([(x,y) for y in range(-gaussian_radius,gaussian_radius+1) for x in range(-gaussian_radius,gaussian_radius+1)])
    pixels=(centers[:,None,:]+off[None,:,:]).reshape(-1,2)
    for axis,size in ((0,w),(1,h)):
        z=pixels[:,axis]%(2*size);pixels[:,axis]=np.where(z<size,z,2*size-1-z)
    ids=np.unique(pixels[:,1]*w+pixels[:,0]);pixels=np.column_stack((ids%w,ids//w))
    hit=pixels[mask[pixels[:,1],pixels[:,0]]!=0]
    local=[]
    for r in local_records:
        box=np.asarray(r['source_roi_xyxy']);inside=((pixels>=box[:2])&(pixels<box[2:])).all(axis=1)
        if inside.any():local.append(r.get('id',r.get('exclusion_id',r['source_roi_xyxy'])))
    return {'sites':len(sites),'native_pixels':len(pixels),'outside_native_image':outside,
        'person_mask_pixels':len(hit),'first_person_mask_pixel':hit[0].tolist() if len(hit) else None,
        'static_exclusion_ids':local,'status':'masked_reader_support' if len(hit) or local else 'reader_support_clear'}

def original_source_profile_audit(uv,evidence,mask,local_records=()):
    """Recorded source readout footprint, distinct from the v1 +/-3 worker.

    sample_chroma uses normal offsets at 1 px and tangent integer -3..3,
    float32 1/32 remap quantization, with raw chroma/luminance (no 2-D
    Gaussian). Full recorded interval is source support, not just q25/q75.
    """
    n=np.asarray(evidence.get('profile_normal',[]),float)
    q=np.asarray(evidence.get('query_uv',[]),float)
    interval=None;interval_key=None
    for key in ('whole_transition_native_interval_px','complete_band_native_interval_px','complete_band_native_interval'):
        if key in evidence:
            interval=np.asarray(evidence[key],float);interval_key=key;break
    if interval is not None:
        if n.shape!=(2,) or q.shape!=(2,) or interval.shape!=(2,) or not np.isfinite(np.r_[n,q,interval]).all() or interval[1]<interval[0]:
            raise ValueError('Invalid recorded full native source profile')
        if not np.isclose(np.linalg.norm(n),1.,rtol=0,atol=2e-6):raise ValueError('Recorded profile normal is not unit')
        offsets=interval[0]+np.arange(int(np.floor(interval[1]-interval[0]+.25))+1)
        tangent=np.array([-n[1],n[0]])
        sites=q[None,None,:]+offsets[None,:,None]*n+np.arange(-3,4)[:,None,None]*tangent
        sites=np.rint(sites.astype(np.float32)*32)/32
        guard=support_pixel_audit(sites.reshape(-1,2),mask,local_records,gaussian_radius=0)
        guard.update(stencil='recorded_sample_chroma_full_interval_tangent7_cv2_bilinear32',interval_key=interval_key,normal_interval_px=interval.tolist(),profile_tangent_offsets_px=[-3,3])
    else:
        # Preserve the declared two-side/quantile context even where a source
        # did not record the entire historical search interval. No invented ROI.
        guard=support_pixel_audit(_between(_contexts(uv,evidence)),mask,local_records,gaussian_radius=0)
        guard.update(stencil='recorded_context_continuous_interval_only',full_historical_reader_interval_recorded=False)
    return guard



def filter_source_family_readouts(rows, mask, records=(), *, operator='material_interface'):
    kept, rejected = [], []
    for row in rows:
        uv=np.asarray(row['uv'],float);evidence=row.get('source_evidence') or {}
        normal=np.asarray(evidence.get('profile_normal',[]),float)
        if normal.shape!=(2,) or not np.isfinite(normal).all() or np.linalg.norm(normal)<1e-8:
            rejected.append(dict(native_readout=row,reason='missing_finite_source_profile_normal_before_family_fit'))
            continue
        normal=normal/np.linalg.norm(normal)
        seeds=np.asarray(evidence.get('actual_source_sides_uv',[uv]),float).reshape(-1,2) if operator=='paint_center' else uv[None]
        sites=np.vstack([p+np.arange(-3.,3.0001,.25)[:,None]*normal for p in seeds])
        current=support_pixel_audit(sites,mask,records)
        original=original_source_profile_audit(uv,evidence,mask,records)
        reason=None
        if current.get('outside_native_image') or original.get('outside_native_image'):
            reason='original_or_current_reader_footprint_outside_native_image_before_family_fit'
        elif current.get('person_mask_pixels') or original.get('person_mask_pixels') or current.get('static_exclusion_ids') or original.get('static_exclusion_ids'):
            reason='original_or_current_reader_footprint_intersects_excluded_source_before_family_fit'
        audit=dict(current_reader_support=current,original_source_profile=original)
        if reason:
            rejected.append(dict(native_readout=row,reason=reason,source_family_support_admission=audit))
        else:
            row['source_evidence']['source_family_support_admission']=audit
            kept.append(row)
    return kept,rejected
