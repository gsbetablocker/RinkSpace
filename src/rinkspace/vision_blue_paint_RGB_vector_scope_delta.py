"""Blue RGB-vector source searches confined to a connected actual ice scope.

The original query prefix remains immutable. Appended searches can only shrink
to native samples supported by the same source image, including the median
tangent footprint, bilinear pixels and smoothing halo. This supplies candidates;
no width, model point or source identity is inferred.
"""
from __future__ import annotations

import numpy as np

from .vision_blue_paint_RGB_vector_delta import append_blue_paint_RGB_vector_readouts


def _supported_interval(mask, point, normal, interval):
    offsets=np.arange(interval[0],interval[1]+.25,1.,dtype=np.float64)
    tangent=np.array([-normal[1],normal[0]])
    samples=(point[None,None]+normal[None,None]*offsets[:,None,None]
        +tangent[None,None]*np.arange(-2,3)[None,:,None])
    xy=np.rint(samples.astype(np.float32)*32.)/32.
    lo=np.floor(xy).astype(np.int64);hi=np.ceil(xy).astype(np.int64)
    valid=np.ones(len(offsets),bool)
    for xx in (lo[...,0],hi[...,0]):
        for yy in (lo[...,1],hi[...,1]):
            inside=(xx>=0)&(yy>=0)&(xx<mask.shape[1])&(yy<mask.shape[0])
            good=np.zeros(inside.shape,bool)
            good[inside]=mask[yy[inside],xx[inside]]
            valid&=good.all(1)
    ids=np.flatnonzero(valid)
    runs=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1)
    for run in runs:
        if not len(run):continue
        # Keep two native samples on each side for the existing five-sample
        # smoothing kernel. The source centre only selects a connected scope,
        # never a component or its measured q50 location.
        first,last=float(offsets[run[0]]+2.),float(offsets[run[-1]]-2.)
        if first<=0<=last and last-first>=20.:
            return np.array([first,last]),dict(native_scope_run=[float(offsets[run[0]]),float(offsets[run[-1]])],
                source_interval_smoothing_halo_px=2.,source_interpolation_grid_px=1./32.,
                tangent_native_sample_offsets_px=[-2,-1,0,1,2])
    return None,None


def append_blue_paint_scope_RGB_vector_readouts(rgb,measurement,exclusion_mask,inventory,*,
        source_ice_scope_mask,query_indices=None,curve_ids=('B1:segment0','B2:segment0'),
        profile_backend='cpu',keep_profiles=False):
    """Append whole-band q50 candidates without reading outside actual ice."""
    mask=np.asarray(source_ice_scope_mask,bool)&~np.asarray(exclusion_mask,bool)
    allowed={i for i,d in enumerate(inventory)if d['curve_id']in curve_ids
        and d['operator']=='paint_center'and d['material_signal']=='blue'}
    curves=np.asarray(measurement['query_curve'])
    selected=(np.flatnonzero(np.isin(curves,list(allowed)))if query_indices is None
        else np.unique(np.asarray(query_indices,np.int64)))
    if (np.any(selected<0)or np.any(selected>=len(curves))
            or any(int(curves[i])not in allowed for i in selected)):
        raise ValueError('Scope successor needs original declared blue-paint query indices')
    intervals=np.asarray(measurement['query_interval']).copy();usable=[];evidence=[]
    for i in selected:
        clipped,proof=_supported_interval(mask,np.asarray(measurement['query_uv'])[i],
            np.asarray(measurement['query_normal'])[i],intervals[i])
        record=dict(original_query_index=int(i),original_search_interval_px=intervals[i].tolist(),
            original_query_UV_and_normal_changed=False,source_profile_reads_outside_actual_ice_scope=False)
        if clipped is None:
            evidence.append(dict(record,status='pending_connected_complete_actual_ice_profile_scope_missing'))
            continue
        usable.append(int(i));intervals[i]=clipped
        evidence.append(dict(record,status='actual_ice_connected_profile_scope_supported',
            appended_search_interval_px=clipped.tolist(),**proof))
    original_queries=len(curves)
    temporary=dict(measurement,query_interval=intervals)
    output=append_blue_paint_RGB_vector_readouts(rgb,temporary,exclusion_mask,inventory,
        source_ice_scope_mask=source_ice_scope_mask,query_indices=np.asarray(usable,np.int64),
        curve_ids=curve_ids,profile_backend=profile_backend,keep_profiles=keep_profiles)
    current=output['measurement']
    for name in ('curve','uv','normal','interval','status'):
        key='query_'+name
        current[key][:original_queries]=np.asarray(measurement[key])
        if not np.array_equal(current[key][:original_queries],np.asarray(measurement[key])):
            raise ValueError('Scope successor changed an original frozen source query')
    receipt=dict(schema='rinkspace.actual-ice-scope-blue-RGB-vector-query-successor.v1',
        source_binding=dict(measurement['source_binding']),query_source_scope_evidence=evidence,
        original_query_prefix_bitexact_preserved=True,appended_queries_use_actual_supported_search_intervals=True,
        original_query_complete_profile_replay_claimed=False,
        model_UV_or_paint_width_used_as_source_boundary=False,source_FIT_admitted=False)
    current['source_blue_RGB_vector_scope']=receipt
    output['source_blue_RGB_vector_scope']=receipt
    output['source_blue_RGB_vector_queries']['actual_ice_scope_successor']=receipt
    return output
