"""Current RGB profile re-entry around an authorized named-material NAV.

The NAV supplies search stations and named-carrier context, never output UVs
or a pixel-error acceptance radius. Auxiliary measurements do not become fit
observations. Original UV, sides, quartiles, sigma and proof locators survive.
"""
from __future__ import annotations

import numpy as np

from .gray_source_contour_review import (
    analyze_current_gray_native_contour_pixels,_material_signature_intervals,
)


class _OwnNativeFloat32:
    def __init__(self,values):self.values=values
    def astype(self,dtype):
        if np.dtype(dtype)!=np.dtype(np.float32):raise ValueError('Original producer native float32 operator required')
        return self.values


class _OwnBatchedNativeFrame:
    def __init__(self,frame,native_rgb_field):
        self.rgb=_OwnNativeFloat32(native_rgb_field)
        key='gray_review_same_producer_native_luminance_float32'
        if key not in frame.material_signals:
            values=frame.luminance.astype(np.float32);values.flags.writeable=False
            frame.material_signals[key]=values
        self.luminance=_OwnNativeFloat32(frame.material_signals[key])


def same_producer_actual_gray_profiles(frame,locations,normals,intervals,native_rgb_field):
    """Reuse original production median/native gradient-quartile operator."""
    from rinkspace.vision_programme_ice_review import _batched_actual_profiles,_actual_profile_candidates
    arrays=[np.asarray(x,float) for x in (locations,normals,intervals)]
    # Equal native sample counts prevent the public ragged padding sampler
    # from reading another query's extra, out-of-scope search interval.
    sizes=np.ceil(arrays[2][:,1]-arrays[2][:,0]).astype(int)+1
    readings=[None]*len(arrays[0]);cached=_OwnBatchedNativeFrame(frame,native_rgb_field)
    for size in np.unique(sizes):
        ids=np.flatnonzero(sizes==size)
        offsets,profiles,smooth,valid=_batched_actual_profiles(cached,
            *(array[ids]for array in arrays),np.full(len(ids),'dark'))
        for j,i in enumerate(ids):
            readings[i]=_actual_profile_candidates(arrays[0][i],arrays[1][i],offsets[j,valid[j]],
                profiles[j,valid[j]],smooth[j,valid[j]],operator='paint_center',
                signal='dark',transition=None,with_quantiles=True)
    return readings


def _sample_scope(mask,samples):
    xy=np.rint(np.asarray(samples,np.float32)*32.)/32.
    lo=np.floor(xy).astype(np.int64);hi=np.ceil(xy).astype(np.int64)
    for x in (lo[...,0],hi[...,0]):
        for y in (lo[...,1],hi[...,1]):
            inside=(x>=0)&(y>=0)&(x<mask.shape[1])&(y<mask.shape[0])
            if not inside.all()or not mask[y[inside],x[inside]].all():return False
    return True


def _available_source_interval(mask,location,normal,radius):
    """Clip navigation search to one actually supported source interval."""
    offsets=np.arange(-radius,radius+.25,1.)
    offsets=np.r_[offsets[0]-2,offsets[0]-1,offsets,offsets[-1]+1,offsets[-1]+2]
    tangent=np.array([-normal[1],normal[0]])
    good=[]
    for offset in offsets:
        samples=location+normal*offset+tangent[None]*np.arange(-2,3)[:,None]
        good.append(_sample_scope(mask,samples))
    good=np.asarray(good,bool);runs=np.split(np.flatnonzero(good),np.flatnonzero(np.diff(np.flatnonzero(good))>1)+1)
    for run in runs:
        if len(run)<10:continue
        lo=float(offsets[run[0]]+2.);hi=float(offsets[run[-1]]-2.)
        if lo<=0<=hi and hi-lo>=6:return [lo,hi]
    return None


def _navigation_patch(point,polyline,spacing,footprint):
    path=np.asarray(polyline,float);d=np.diff(path,axis=0);length=np.linalg.norm(d,axis=1)
    usable=length>0
    if not usable.any():raise ValueError('Visible named NAV carrier needs nonzero segments')
    fraction=np.divide(np.einsum('ij,ij->i',point-path[:-1],d),length**2,
        out=np.zeros(len(d)),where=usable);fraction=np.clip(fraction,0.,1.)
    feet=path[:-1]+fraction[:,None]*d
    nearest=int(np.argmin(np.where(usable,np.linalg.norm(feet-point,axis=1),np.inf)))
    station=np.r_[0.,np.cumsum(length)];center=station[nearest]+fraction[nearest]*length[nearest]
    offsets=np.arange(-footprint,footprint+spacing*.01,spacing)
    offsets=np.unique(np.r_[offsets,0.])
    target=center+offsets;inside=(target>=station[0])&(target<=station[-1]);target=target[inside]
    indices=np.minimum(np.searchsorted(station,target,side='right')-1,len(d)-1)
    indices=np.maximum(indices,0);unit=d[indices]/length[indices,None]
    coordinates=path[indices]+(target-station[indices])[:,None]*unit
    coordinates+=point-feet[nearest]
    normals=np.c_[-unit[:,1],unit[:,0]]
    center_index=int(np.flatnonzero(target==center)[0])
    return coordinates,normals,unit,center_index


def review_named_gray_point_with_current_profiles(frame,point,sides,quantiles,*,
        source_valid_mask,navigation_polyline,native_quantization_step_px,
        source_interpolation_grid_px,navigation_sampling_spacing_px,
        navigation_normal_search_radius_px,source_neighbor_navigation_footprint_px,
        named_gray_material_signature_bounds,native_rgb_field):
    """Return one exact original point's current-source qualification evidence."""
    point=np.asarray(point,float);sides=np.asarray(sides,float);q=np.asarray(quantiles,float)
    mask=np.asarray(source_valid_mask,bool)
    base=dict(original_numeric_pixel_values_changed=False,auxiliary_pixels_are_not_new_FIT_observations=True,
        model_UV_distance_used_as_pixel_accuracy_acceptance=False,
        NAV_used_only_for_named_material_search_and_carrier_identity=True,
        native_quantization_step_px=float(native_quantization_step_px),
        source_interpolation_grid_px=float(source_interpolation_grid_px),
        auxiliary_actual_profile_operator='original_producer_5_native_tangent_median_signed_gradient_q25_q50_q75',
        original_native_RGB_and_luminance_float32_caches_are_rebuilt_per_immutable_frame=True)
    def pending(reason,**extra):return dict(base,status=reason,**extra)
    if not np.isfinite(q).all()or not np.array_equal(sides,q[:,1])or not np.array_equal(point,sides.mean(0)):
        return pending('pending_original_literal_q50_or_readout_support_missing')
    locations,normals,tangents,center=_navigation_patch(point,navigation_polyline,
        navigation_sampling_spacing_px,source_neighbor_navigation_footprint_px)
    radius=max(float(navigation_normal_search_radius_px),
        float(np.linalg.norm(sides[1]-sides[0])*.5+6.))
    valid_queries=[];aux=[];query_evidence=[];query_intervals=[]
    for qi,(location,normal)in enumerate(zip(locations,normals)):
        offsets=np.arange(-radius,radius+.25,1.)
        halo=np.r_[offsets[0]-2,offsets[0]-1,offsets,offsets[-1]+1,offsets[-1]+2]
        tangent=np.array([-normal[1],normal[0]])
        samples=location[None,None]+normal[None,None]*halo[None,:,None]+tangent[None,None]*np.arange(-2,3)[:,None,None]
        interval=[-radius,radius]
        clipped=False
        if not _sample_scope(mask,samples):
            interval=_available_source_interval(mask,location,normal,radius);clipped=True
            if interval is None:
                query_evidence.append(dict(query_index=qi,status='pending_full_actual_profile_halo_and_bilinear_source_scope'))
                continue
        valid_queries.append(qi);query_intervals.append(interval)
        query_evidence.append(dict(query_index=qi,
            actual_supported_search_interval_px=interval,source_search_clipped_to_actual_mask_or_boundary=clipped,
            no_scope_excluded_RGB_read=True))
    readings=same_producer_actual_gray_profiles(frame,locations[valid_queries],normals[valid_queries],
        query_intervals,native_rgb_field) if valid_queries else []
    for qi,candidates in zip(valid_queries,readings):
        indices=[]
        for uv,ss,source_bad,qq in candidates:
            if source_bad:continue
            indices.append(len(aux));aux.append(dict(uv=uv,sides=ss,quantiles=qq,
                auxiliary_query_index=qi,named_NAV_tangent=tangents[qi],
                source_flags=dict(original_production_truncated=False)))
        evidence=next(e for e in query_evidence if e['query_index']==qi)
        evidence.update(actual_complete_gray_band_candidates=len(indices),auxiliary_candidate_indices=indices)
    center_candidates=[i for i,a in enumerate(aux)if a['auxiliary_query_index']==center]
    match=[];normal=normals[center]
    for i in center_candidates:
        aq=aux[i]['quantiles']
        original=q@normal;current=aq@normal
        if all(max(float(original[s].min()-.5*native_quantization_step_px),float(current[s].min()-.5*native_quantization_step_px))
                <=min(float(original[s].max()+.5*native_quantization_step_px),float(current[s].max()+.5*native_quantization_step_px))for s in range(2)):
            match.append(i)
    if len(match)!=1:return pending('pending_own_q50_band_and_auxiliary_source_profile_not_unique',
        matching_actual_center_band_count=len(match),auxiliary_query_evidence=query_evidence)
    points=np.asarray([point]+[a['uv']for a in aux]);ss=np.asarray([sides]+[a['sides']for a in aux]);qq=np.asarray([q]+[a['quantiles']for a in aux])
    axis=tangents[center]
    geometry=analyze_current_gray_native_contour_pixels(frame.rgb,points,ss,qq,
        source_valid_mask=mask,source_axis=axis,native_quantization_step_px=native_quantization_step_px,
        native_rgb_field=native_rgb_field)
    if not geometry['valid'][0]:return pending('pending_own_RGB_two_interfaces_native_scope_or_material_missing')
    lo,hi=_material_signature_intervals(geometry['colours']);eligible=np.zeros(len(points),bool)
    for i in range(len(points)):
        interval=geometry['tangent_intervals'][i]
        if not geometry['valid'][i]or interval is None or interval[0]>interval[1]:continue
        if not np.isfinite(lo[i]).all()or not np.isfinite(hi[i]).all():continue
        if np.any(np.maximum(lo[i],named_gray_material_signature_bounds[0])>np.minimum(hi[i],named_gray_material_signature_bounds[1])):continue
        direction=axis if i==0 else aux[i-1]['named_NAV_tangent']
        expected_slope=float(direction@np.array([-axis[1],axis[0]]))/float(direction@axis)
        if interval[0]<=expected_slope<=interval[1]:eligible[i]=True
    if not eligible[0]:return pending('pending_current_actual_source_material_or_contour_competes_with_named_carrier_NAV',
        actual_original_tangent_interval=geometry['tangent_intervals'][0],auxiliary_query_evidence=query_evidence)
    reached={0};stack=[0]
    while stack:
        for j in geometry['neighbors'][stack.pop()]:
            if eligible[j]and j not in reached:reached.add(j);stack.append(j)
    original_band=match[0]+1
    if original_band not in reached:return pending('pending_original_q50_and_current_auxiliary_contour_continuation_unresolved')
    named_queries={aux[i-1]['auxiliary_query_index']for i in reached if i>0}
    if len(named_queries)<2:return pending('pending_actual_two_dimensional_contour_support_missing')
    # A second compatible band at the same navigation station is unresolved;
    # nearest-to-NAV or the number of points never chooses a source branch.
    for qi in named_queries:
        same=[i+1 for i,a in enumerate(aux)if a['auxiliary_query_index']==qi and eligible[i+1]]
        if len(same)>1:return pending('pending_multiple_current_actual_gray_contours_at_named_NAV_station')
    return dict(base,status='current_own_RGB_named_gray_contour_and_q50_source_profile_supported',
        actual_original_tangent_interval=geometry['tangent_intervals'][0],
        actual_connected_auxiliary_native_station_count=len(named_queries),
        source_profile_candidate_count=len(aux),auxiliary_query_evidence=query_evidence)
