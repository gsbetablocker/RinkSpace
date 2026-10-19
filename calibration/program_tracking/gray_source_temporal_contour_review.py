"""Own-image gray contour continuation from only previously verified pixels.

Bidirectional source LK provides normal navigation. It never supplies output
UVs, independent 2D point correspondences, or a substitute for current source
material, tangent, scope and competing-contour checks.
"""
from __future__ import annotations

import hashlib
import cv2
import numpy as np

from .gray_source_contour_review import (
    analyze_current_gray_native_contour_pixels,_tangent_interval,
    _material_signature_intervals,
)


def _own_rgb(rgb,binding):
    rgb=np.asarray(rgb)
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
        raise ValueError('Exact own native RGB required')
    if hashlib.sha256(np.ascontiguousarray(rgb[:,:,::-1]).tobytes()).hexdigest()!=binding['decoded_native_BGR_sha256']:
        raise ValueError('Native RGB differs from its exact source binding')
    return rgb


def review_gray_source_temporal_continuation(previous_rgb,previous_uv,previous_sides,previous_quantiles,
        current_rgb,uv,sides,quantiles,*,previous_binding,current_binding,
        previous_verified_source_paths,previous_source_review,
        previous_source_valid_mask,current_source_valid_mask,native_quantization_step_px):
    """Check all current points; unresolved points retain their original evidence.

    The 24px source search footprint is the existing public source-context
    navigation footprint. It cannot accept a pixel. Acceptance here requires
    a unique current RGB contour with its own exact interfaces, material and
    tangent support. Source LK sample counts never become anchor counts.
    """
    from rinkspace.vision_ice_temporal_identity import _curve_local_navigation
    previous_rgb=_own_rgb(previous_rgb,previous_binding);current_rgb=_own_rgb(current_rgb,current_binding)
    p=np.asarray(previous_uv,float);ps=np.asarray(previous_sides,float);pq=np.asarray(previous_quantiles,float)
    points=np.asarray(uv,float);sides=np.asarray(sides,float);q=np.asarray(quantiles,float)
    n=len(points);paths=[list(path)for path in previous_verified_source_paths]
    flat=[i for path in paths for i in path]
    if (previous_source_review.get('source_binding')!=previous_binding
            or previous_source_review.get('model_coordinates_or_fit_errors_used') is not False
            or not set(flat).issubset(previous_source_review['source_shape_verified_original_local_point_indices'])
            or len(flat)!=len(set(flat)) or not flat or any(not path for path in paths)
            or any(type(i)is not int or not 0<=i<len(p)for i in flat)
            or any(previous_binding[k]!=current_binding[k]for k in ('source_sha256','time_base','shot_id'))
            or previous_binding['actual_pts']==current_binding['actual_pts']):
        raise ValueError('Only exact same-cut previously source-verified points may navigate')
    base=dict(schema='rinkspace.own-RGB-gray-temporal-contour-review.v1',source_binding=dict(current_binding),
        previous_source_binding=dict(previous_binding),original_source_points_removed=0,
        original_numeric_pixel_values_changed=False,model_coordinates_or_fit_errors_used=False,
        source_shape_check_is_not_independent_named_identity_certificate=True,source_FIT_admitted=False,
        previous_source_anchor_point_count=len(flat),LK_sample_count_is_not_independent_anchor_count=True,
        normal_navigation_is_not_two_dimensional_point_correspondence=True)
    def unresolved(reason,motion=None):
        return dict(base,point_decisions=[dict(original_local_point_index=i,status=reason)for i in range(n)],
            pending_original_local_point_indices=list(range(n)),source_shape_verified_original_local_point_indices=[],
            current_verified_source_paths=[],source_normal_navigation=motion)
    if not any(len(path)>=2 for path in paths):
        return unresolved('pending_current_two_dimensional_source_tangent_requires_actual_auxiliary_support')
    track=dict(actual_pts=previous_binding['actual_pts'],navigation_pts=previous_binding['actual_pts'],
        actual_uv=p[flat],navigation_uv=p[flat],candidate_sides=ps[flat])
    prev_gray=cv2.cvtColor(previous_rgb,cv2.COLOR_RGB2GRAY);current_gray=cv2.cvtColor(current_rgb,cv2.COLOR_RGB2GRAY)
    nav,motion=_curve_local_navigation(prev_gray,current_gray,~np.asarray(previous_source_valid_mask,bool),
        ~np.asarray(current_source_valid_mask,bool),track)
    if motion.get('source_RGB_tracking_confirmed')is not True:
        return unresolved('pending_current_own_RGB_normal_navigation_missing',motion)
    centered=nav-nav.mean(0);axis=np.linalg.svd(centered,full_matrices=False)[2][0]
    if axis[0]<0:axis=-axis
    # These are transported previous readout supports for source navigation,
    # explicitly separate from all unchanged current measured pixel arrays.
    transported_q=pq[flat]+(np.asarray(nav,float)-p[flat])[:,None,None,:]
    lookup={i:pos for pos,i in enumerate(flat)};expected=[]
    for path in paths:
        if len(path)<2:continue
        ids=[lookup[i]for i in path]
        interval=_tangent_interval(np.asarray(nav)[ids],transported_q[ids],axis,native_quantization_step_px)
        if interval is not None and interval[0]<=interval[1]:expected.append(interval)
        else:
            # A visibly curved source path need not have one global slope.
            for a,b in zip(ids[:-1],ids[1:]):
                interval=_tangent_interval(np.asarray(nav)[[a,b]],transported_q[[a,b]],axis,native_quantization_step_px)
                if interval is not None and interval[0]<=interval[1]:expected.append(interval)
    if not expected:return unresolved('pending_previous_source_tangent_support_unresolved',motion)
    previous_geometry=analyze_current_gray_native_contour_pixels(previous_rgb,p[flat],ps[flat],pq[flat],
        source_valid_mask=previous_source_valid_mask,source_axis=axis,native_quantization_step_px=native_quantization_step_px)
    plo,phi=_material_signature_intervals(previous_geometry['colours'])
    usable=previous_geometry['valid']&np.isfinite(plo).all((1,2))&np.isfinite(phi).all((1,2))
    if not usable.all():return unresolved('pending_previous_own_source_material_or_scope_support_missing',motion)
    material_bounds=[plo.min(0),phi.max(0)]
    geometry=analyze_current_gray_native_contour_pixels(current_rgb,points,sides,q,
        source_valid_mask=current_source_valid_mask,source_axis=axis,native_quantization_step_px=native_quantization_step_px)
    clo,chi=_material_signature_intervals(geometry['colours']);eligible=np.zeros(n,bool);reasons=[]
    for i in range(n):
        interval=geometry['tangent_intervals'][i]
        if not geometry['valid'][i]:reason='pending_current_own_RGB_interfaces_quantiles_or_source_scope'
        elif not np.isfinite(clo[i]).all()or not np.isfinite(chi[i]).all():reason='pending_current_native_gray_material_signature_unresolved'
        elif np.any(np.maximum(clo[i],material_bounds[0])>np.minimum(chi[i],material_bounds[1])):reason='pending_current_native_material_competes_with_previous_verified_gray'
        elif interval is None:reason='pending_current_actual_two_dimensional_tangent_support_unresolved'
        elif interval[0]>interval[1]:reason='pending_current_multiple_actual_contour_directions'
        elif not any(max(interval[0],e[0])<=min(interval[1],e[1])for e in expected):reason='pending_current_actual_contour_direction_competes_with_verified_source_navigation'
        else:reason=None;eligible[i]=True
        reasons.append(reason)
    components=[];component_id=np.full(n,-1,int)
    for i in np.flatnonzero(eligible):
        if component_id[i]>=0:continue
        comp={int(i)};stack=[int(i)];component_id[i]=len(components)
        while stack:
            for j in geometry['neighbors'][stack.pop()]:
                if eligible[j]and component_id[j]<0:
                    component_id[j]=len(components);comp.add(j);stack.append(j)
        components.append(sorted(comp))
    found=set();ambiguous=set();navigation_matches=[]
    for owner,target in zip(flat,nav):
        # This existing footprint locates candidate source contours. Its
        # distance never appears in a point's precision or fitting weight.
        near=np.flatnonzero(eligible&(np.linalg.norm(points-target,axis=1)<=24.))
        matches=set(int(component_id[i])for i in near)
        navigation_matches.append(dict(previous_verified_local_point_index=owner,
            current_source_candidate_indices=near.tolist(),current_source_components=sorted(matches)))
        if len(matches)==1:found.update(matches)
        elif len(matches)>1:ambiguous.update(matches)
    found-=ambiguous;accepted=[];pending=[];decisions=[]
    for i in range(n):
        reason=reasons[i]
        if reason is None:
            if component_id[i]in ambiguous:reason='pending_multiple_current_RGB_contours_compete_for_same_source_navigation'
            elif component_id[i]not in found:reason='pending_current_named_source_contour_lineage_unresolved'
            else:reason='current_own_RGB_contour_unique_source_continuation_supported';accepted.append(i)
        if reason.startswith('pending_'):pending.append(i)
        decisions.append(dict(original_local_point_index=i,status=reason,
            actual_tangent_slope_interval=geometry['tangent_intervals'][i],
            actual_contour_neighbor_indices=sorted(geometry['neighbors'][i]),
            current_RGB_contour_component=int(component_id[i])))
    current_paths=[]
    for cid in sorted(found):
        current_paths.append(sorted(components[cid],key=lambda i:float(points[i]@axis)))
    return dict(base,point_decisions=decisions,pending_original_local_point_indices=pending,
        source_shape_verified_original_local_point_indices=accepted,current_verified_source_paths=current_paths,
        source_normal_navigation=motion,previous_own_RGB_tangent_navigation_intervals=expected,
        source_navigation_matches=navigation_matches,actual_two_interface_continuation=geometry['continuation'],
        normal_LK_gray_bindings=dict(previous_gray_sha256=hashlib.sha256(prev_gray.tobytes()).hexdigest(),
            current_gray_sha256=hashlib.sha256(current_gray.tobytes()).hexdigest()),
        source_contour_competition_unresolved_components=sorted(ambiguous))
