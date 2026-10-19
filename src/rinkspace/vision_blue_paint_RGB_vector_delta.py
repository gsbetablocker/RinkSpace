"""Independent RGB-vector blue-paint readouts; frozen readers remain unchanged.

The original chroma measurements remain an exact prefix. New source queries
copy search navigation only. Both new interface locations and their midpoint
come from the current original RGB. Supersession is explicit and separate from
world-role exclusion; it is not permission to remove an admitted old row until
the replacement passes its source identity and pixel gates.
"""
from __future__ import annotations

import hashlib
import cv2
import numpy as np

from .vision_programme_ice_review import _batched_actual_profiles,_candidate_context_excluded
from .vision_source_shapes import _MaterialProfileFrame


def _native_rgb_profiles(rgb,points,normals,intervals,*,backend='cpu'):
    count=len(points)
    size=int(np.ceil(np.max(intervals[:,1]-intervals[:,0])))+1 if count else 0
    offsets=intervals[:,:1]+np.arange(size,dtype=float)[None]
    valid=offsets<=intervals[:,1:]+.25
    if not count:return offsets,np.empty((0,size,3)),np.empty((0,size,3)),valid
    tangent=np.c_[-normals[:,1],normals[:,0]]
    samples=(points[:,None,None]+normals[:,None,None]*offsets[:,None,:,None]
        +tangent[:,None,None]*np.arange(-2,3)[None,:,None,None])
    xy=np.rint(samples.astype(np.float32)*32.)/32.
    if backend=='cpu':
        xy=xy.reshape(-1,size,2)
        raw=cv2.remap(rgb.astype(np.float32),xy[...,0],xy[...,1],cv2.INTER_LINEAR)
        profiles=np.median(raw.reshape(count,5,size,3),axis=1)
    elif backend=='cuda':
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA actual RGB-vector readout requested without CUDA')
        with torch.inference_mode():
            coordinates=torch.as_tensor(xy,dtype=torch.float64,device='cuda')
            base=torch.floor(coordinates).long();fraction=(coordinates-base).float()
            fx,fy=fraction[...,0],fraction[...,1]
            image=torch.as_tensor(np.ascontiguousarray(rgb),dtype=torch.float32,device='cuda')
            height,width=rgb.shape[:2];parts=[]
            for dx,dy,weight in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)):
                x,y=base[...,0]+dx,base[...,1]+dy
                inside=(x>=0)&(x<width)&(y>=0)&(y<height)
                values=image[y.clamp(0,height-1),x.clamp(0,width-1)]
                parts.append(values*weight[...,None]*inside[...,None])
            values=(parts[0]+parts[1])+(parts[2]+parts[3])
            profiles=values.median(dim=1).values.cpu().numpy()
    else:raise ValueError('Explicit CPU or CUDA source sampling backend required')
    for i in range(count):
        end=int(valid[i].sum())
        if end<size:profiles[i,end:]=profiles[i,end-1]
    smooth=np.stack([cv2.GaussianBlur(profiles[:,:,k].astype(float),(5,1),1.,
        borderType=cv2.BORDER_REPLICATE) for k in range(3)],axis=-1)
    return offsets,profiles,smooth,valid


def _rgb_vector_components(point,normal,offsets,rgb,smooth,luma,luma_smooth):
    """All observed darkened RGB components and their two signed interfaces."""
    if len(offsets)<20:return []
    second=np.diff(luma_smooth,n=2)
    noise=float(1.4826*np.median(np.abs(second-np.median(second))))
    flank=min(10,max(3,len(offsets)//10))
    background=np.linspace(float(np.median(luma[:flank])),float(np.median(luma[-flank:])),len(luma))
    material=np.maximum(background-luma_smooth,0.)
    active=material>max(2.,3*noise,.1*float(material.max()))
    starts=np.flatnonzero(active&~np.r_[False,active[:-1]])
    ends=np.flatnonzero(active&~np.r_[active[1:],False]);components=[]
    for a,b in zip(starts,ends):
        if b-a<3:continue
        if components and a-components[-1][1]-1<=3:components[-1][1]=int(b)
        else:components.append([int(a),int(b)])
    gradient=np.diff(smooth,axis=0);output=[]
    for a,b in components:
        lo,hi=max(0,a-10),min(len(offsets)-1,b+10)
        mass=material[lo:hi+1];total=float(mass.sum())
        if total<=0:continue
        # Observed intensity divides two interface searches. It never supplies
        # an output center and no neighboring width supplies either boundary.
        split_offset=float(np.interp(.5,(np.cumsum(mass)-mass*.5)/total,offsets[lo:hi+1]))
        split=int(np.clip(np.searchsorted(offsets,split_offset),lo+1,hi-1))
        inside=int(a+np.argmin(luma_smooth[a:b+1]))
        paint=np.median(rgb[max(a,inside-1):min(b+1,inside+2)],axis=0)
        outside=np.asarray([np.median(rgb[lo:min(lo+4,split)],axis=0),
            np.median(rgb[max(split+1,hi-3):hi+1],axis=0)])
        directions=np.asarray([paint-outside[0],outside[1]-paint],float)
        lengths=np.linalg.norm(directions,axis=1)
        if (lengths<=1.).any():continue
        directions/=lengths[:,None]
        quantiles=[];amounts=[]
        for begin,end,direction in ((lo,split,directions[0]),(split,hi,directions[1])):
            weight=np.maximum(gradient[begin:end]@direction,0.);amount=float(weight.sum())
            if amount<=max(1.,3*noise):break
            quantile=np.interp([.25,.5,.75],np.r_[0.,np.cumsum(weight)]/amount,offsets[begin:end+1])
            quantiles.append(point+normal*quantile[:,None]);amounts.append(amount)
        if len(quantiles)!=2:continue
        quantiles=np.asarray(quantiles,float);sides=quantiles[:,1]
        output.append(dict(uv=sides.mean(0),sides=sides,quantiles=quantiles,
            cropped=lo==0 or hi==len(offsets)-1,component_profile_indices=[a,b],
            full_RGB_interface_profile_intervals=[[lo,split],[split,hi]],
            actual_paint_prototype_RGB=paint.tolist(),actual_exterior_RGB=outside.tolist(),
            actual_signed_RGB_gradient_directions=directions.tolist(),actual_signed_RGB_gradient_amount=amounts,
            source_noise_luminance_second_difference=noise))
    return output


def append_blue_paint_RGB_vector_readouts(rgb,measurement,exclusion_mask,inventory,*,
        source_ice_scope_mask,query_indices=None,curve_ids=('B1:segment0','B2:segment0'),
        profile_backend='cpu',keep_profiles=False):
    """Read blue/gray paint material at selected full original source searches.

    This produces source candidates, not identities or FIT rows. Call the
    ordinary temporal selector and strict normalizer on the returned measurement.
    A caller may supply pending original query IDs to limit a delta to failures.
    New unknown candidates cannot authorize removal from a previously admitted
    pack. Preserve the returned supersession receipt for that later exact merge.
    """
    rgb=np.asarray(rgb);binding=measurement['source_binding']
    digest=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
    bgr=hashlib.sha256(np.ascontiguousarray(rgb[:,:,::-1]).tobytes()).hexdigest()
    if (digest!=measurement['decoded_native_RGB_sha256']
            or binding.get('decoded_native_BGR_sha256',bgr)!=bgr):
        raise ValueError('Blue RGB-vector delta requires its own original native RGB/BGR')
    if any(np.asarray(measurement[name]).dtype!=np.dtype(np.float64)
            for name in ('query_uv','query_normal','query_interval')):
        raise ValueError('Restore exact original source navigation before the RGB-vector delta')
    mask=np.asarray(exclusion_mask,bool)|~np.asarray(source_ice_scope_mask,bool)
    if mask.shape!=rgb.shape[:2]:raise ValueError('Current native physical exclusion and ice scope required')
    allowed={i for i,v in enumerate(inventory) if v['curve_id'] in curve_ids
        and v['operator']=='paint_center' and v['material_signal']=='blue'}
    qcurve=np.asarray(measurement['query_curve']);original_queries=len(qcurve)
    queries=(np.flatnonzero(np.isin(qcurve,list(allowed))) if query_indices is None
        else np.unique(np.asarray(query_indices,dtype=np.int64)))
    if (np.any(queries<0) or np.any(queries>=original_queries)
            or any(int(qcurve[q]) not in allowed for q in queries)):
        raise ValueError('Delta queries must belong to the declared existing blue paint curves')
    points=np.asarray(measurement['query_uv'])[queries];normals=np.asarray(measurement['query_normal'])[queries]
    intervals=np.asarray(measurement['query_interval'])[queries]
    frame=_MaterialProfileFrame(rgb,binding,digest)
    offsets,full_rgb,smooth_rgb,valid=_native_rgb_profiles(rgb,points,normals,intervals,backend=profile_backend)
    if len(queries):
        sampler=_batched_actual_profiles
        if profile_backend=='cuda':
            from .vision_source_cuda_profiles import batched_actual_profiles_cuda
            sampler=batched_actual_profiles_cuda
        other,luma,luma_smooth,other_valid=sampler(frame,points,normals,intervals,np.array(['dark']*len(points)))
        if not np.array_equal(offsets,other) or not np.array_equal(valid,other_valid):
            raise ValueError('RGB and luminance source sampling domains differ')
    else:luma=luma_smooth=np.empty(offsets.shape)
    result=dict(measurement);old_count=len(measurement['candidate_uv'])
    old_flags=np.asarray(measurement['candidate_flags'],np.uint8)
    old_candidate_query=np.asarray(measurement['candidate_query'])
    added={key:[] for key in ('uv','sides','query','flags','side_quantiles_uv')}
    components=[];superseded=[];supersessions=[];statuses=np.full(len(queries),2,np.uint8)
    for i,query in enumerate(queries):
        use=valid[i]
        observed=_rgb_vector_components(points[i],normals[i],offsets[i,use],full_rgb[i,use],
            smooth_rgb[i,use],luma[i,use],luma_smooth[i,use])
        previous=np.flatnonzero(old_candidate_query==query)
        complete=False
        for candidate in observed:
            blocked=any(_candidate_context_excluded(mask,p,normals[i]) for p in [candidate['uv'],*candidate['sides']])
            flags=int(candidate['cropped'])|(int(blocked)<<1)|(int(len(observed)>1)<<2)|8
            complete |= not (flags&3)
            new_index=old_count+len(added['uv'])
            added['uv'].append(candidate['uv']);added['sides'].append(candidate['sides'])
            added['query'].append(original_queries+i);added['flags'].append(flags)
            added['side_quantiles_uv'].append(candidate['quantiles'])
            components.append(dict(candidate_index=new_index,original_query_index=int(query),
                appended_query_index=original_queries+i,
                **{key:value for key,value in candidate.items() if key not in ('uv','sides','quantiles')}))
            supersessions.append(dict(new_candidate_index=new_index,original_query_index=int(query),
                supersedes_original_candidate_indices=previous.tolist(),
                source_replacement_admission_required=True))
        if observed:statuses[i]=1 if len(observed)==1 and complete else 3
        if complete:superseded.extend(previous.tolist())
    for key,value in (('curve',qcurve[queries]),('uv',points),('normal',normals),('interval',intervals),('status',statuses)):
        result['query_'+key]=np.concatenate([np.asarray(measurement['query_'+key]),value])
    for key,value in added.items():
        original=np.asarray(measurement['candidate_'+key])
        new=np.asarray(value,dtype=original.dtype).reshape((-1,)+original.shape[1:])
        result['candidate_'+key]=np.concatenate([original,new])
    superseded=np.unique(np.asarray(superseded,np.int64))
    # Supersession has a literal receipt. This is not a new material-role
    # negative and is not deletion from the active source pack.
    result['candidate_flags'][superseded] |= np.uint8(1)
    receipt=dict(kind='actual_blue_RGB_vector_source_delta_candidates',source_binding=dict(binding),
        decoded_native_RGB_sha256=digest,original_query_prefix_length=original_queries,
        original_candidate_prefix_length=old_count,source_query_indices=queries.tolist(),
        query_curve=qcurve[queries].tolist(),query_uv=points.tolist(),query_normal=normals.tolist(),query_interval=intervals.tolist(),
        original_superseded_candidate_indices=superseded.tolist(),
        original_flags_before_supersession=old_flags[superseded].tolist(),
        supersession_reason='superseded_by_current_actual_RGB_vector_interface_readout',
        supersession_is_not_world_role_negative=True,new_replacement_admission_required_to_retire_old_pack_rows=True,
        candidate_supersession=supersessions,actual_RGB_vector_components=components,
        source_measurement_method='observed_signed_RGB_vector_interface_q50_midpoint',
        width_used_as_center_tolerance_or_filled_boundary=False,projected_UV_used_as_source_pixel_truth=False,
        source_readout_quantization_step_px=1.,source_interpolation_grid_px=1/32.,
        physical_identity_confirmed_by_reader=False,full_programme_accuracy_review_complete=False)
    result['source_blue_RGB_vector_queries']=receipt
    result['source_readout_quantization_step_px']=1.;result['source_interpolation_grid_px']=1/32.
    output=dict(measurement=result,source_blue_RGB_vector_queries=receipt,
        superseded_original_candidate_indices=superseded,
        source_candidates_need_temporal_identity_and_accuracy_review=True)
    if keep_profiles:output['actual_source_profiles']=dict(query_indices=queries,offsets=offsets,RGB=full_rgb,
        smoothed_RGB=smooth_rgb,valid=valid,luminance=luma,smoothed_luminance=luma_smooth)
    return output


def remeasure_own_native_blue_paint_RGB_vector_suffix(rgb,measurement,exclusion_mask,inventory,*,
        source_ice_scope_mask,canonical_receipt,profile_backend='cpu',keep_profiles=False):
    """Read the same search stations in the alias's own native source pixels."""
    own=measurement['source_binding'];old=canonical_receipt['source_binding']
    for key in ('source_sha256','time_base','actual_pts','shot_id'):
        if own.get(key)!=old.get(key):raise ValueError('Own-native RGB delta belongs to another source exposure')
    if len(measurement['query_curve'])!=canonical_receipt['original_query_prefix_length']:
        raise ValueError('Own-native source query prefix differs from canonical RGB-vector receipt')
    queries=np.asarray(canonical_receipt['source_query_indices'],np.int64)
    for key in ('curve','uv','normal','interval'):
        if not np.array_equal(np.asarray(measurement['query_'+key])[queries],np.asarray(canonical_receipt['query_'+key])):
            raise ValueError('Own-native exact source navigation differs; no observed UV may be copied')
    output=append_blue_paint_RGB_vector_readouts(rgb,measurement,exclusion_mask,inventory,
        source_ice_scope_mask=source_ice_scope_mask,query_indices=queries,
        profile_backend=profile_backend,keep_profiles=keep_profiles)
    output['source_blue_RGB_vector_queries']['canonical_navigation_source_binding']=dict(old)
    output['source_blue_RGB_vector_queries']['own_native_RGB_profiles_read']=True
    output['source_blue_RGB_vector_queries']['canonical_observed_UV_copied']=False
    return output
