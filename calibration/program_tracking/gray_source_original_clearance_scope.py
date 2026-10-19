"""Reuse only an authenticated own-source candidate's exact box context.

This is a mask adapter, not material naming or a point-admission certificate.
The current contour guard must still read both interfaces and reject competing
strokes. No person contour, other box, unrelated exclusion, or source-negative
pixel can be cleared.
"""
from __future__ import annotations

import hashlib
import numpy as np

from .strict_model_scope import authenticate_source_reference
from rinkspace.vision_ice_local_clearance import _context_pixels


def apply_exact_current_original_candidate_clearance(source_valid_mask, *,
        source_binding, original_candidate_index, original_candidate_uv,
        original_query_normal, original_candidate_sides_uv, clearance_proof,
        primary_person_box_mask, hard_source_exclusion_mask):
    """Return a copied current mask and a receipt for an exact original proof.

    ``hard_source_exclusion_mask`` must contain full measured/dilated SAM2,
    other-person boxes, all unrelated exclusions and actual source-negative
    pixels. The caller derives it from this current original image. A proof
    from a canonical presentation cannot clear another native presentation.
    """
    valid=np.asarray(source_valid_mask,bool)
    primary=np.asarray(primary_person_box_mask,bool)
    hard=np.asarray(hard_source_exclusion_mask,bool)
    if valid.ndim!=2 or primary.shape!=valid.shape or hard.shape!=valid.shape:
        raise ValueError('One complete current native source mask domain required')
    output=valid.copy()
    if not isinstance(clearance_proof,dict) or not clearance_proof:
        return dict(source_valid_mask=output,receipt=dict(status='no_original_candidate_local_clearance',cleared_pixels=0))
    proof=clearance_proof
    if proof.get('kind')!='known_static_ice_material_visible_inside_primary_detector_box':
        raise ValueError('Unknown original local source-clearance proof')
    if type(original_candidate_index)is not int or proof.get('original_candidate_index')!=original_candidate_index:
        raise ValueError('Original candidate locator differs from clearance proof')
    for name in ('actual_pts','native_pts','shot_id','source_sha256','time_base','decoded_native_BGR_sha256'):
        if name not in source_binding or proof.get(name)!=source_binding[name]:
            raise ValueError('Clearance is not this exact current original image: '+name)
    required=('source_identity_confirmed','measured_person_dilated_contour_and_other_person_boxes_excluded',
        'only_primary_box_context_exclusion_removed','actual_source_painted_ice_membership_confirmed')
    if any(proof.get(k)is not True for k in required):
        raise ValueError('Original source-clearance material/scope proof incomplete')
    if (proof.get('model_projection_or_fit_residual_used_for_identity')is not False
            or proof.get('mask_probability_used_as_visibility_truth')is not False):
        raise ValueError('Model pixels or mask probabilities cannot establish source visibility')
    authenticate_source_reference(proof['source_identity_authority'])
    point=np.asarray(original_candidate_uv,float)
    normal=np.asarray(original_query_normal,float)
    sides=np.asarray(original_candidate_sides_uv,float)
    if (point.shape!=(2,)or normal.shape!=(2,)or sides.shape!=(2,2)
            or not np.isfinite(np.r_[point,normal,sides.ravel()]).all()):
        raise ValueError('Exact original measured candidate/query fields required')
    context=proof['current_complete_material_context']
    # This reproduces the public producer's footprint; no new extent or
    # tolerance is invented, and a fine q50 cannot replace the raw query UV.
    extent=max(24.,float(np.linalg.norm(sides[1]-sides[0]))*.5+12.)
    if (not np.array_equal(np.asarray(context['source_center_uv']),point)
            or not np.array_equal(np.asarray(context['source_normal']),normal)
            or context.get('normal_interval_px')!=[-extent,extent]
            or context.get('tangent_interval_px')!=[-3,3]):
        raise ValueError('Original candidate context differs from its measured source fields')
    pixels=_context_pixels(point,normal,extent)
    if (context.get('actual_native_context_pixel_count')!=len(pixels)
            or context.get('actual_native_context_pixel_indices_sha256')!=hashlib.sha256(pixels.tobytes()).hexdigest()):
        raise ValueError('Original exact source-context pixel certificate changed')
    if ((pixels<0).any()or(pixels[:,0]>=valid.shape[1]).any()or(pixels[:,1]>=valid.shape[0]).any()):
        raise ValueError('Original context is outside this native source image')
    x,y=pixels.T
    if hard[y,x].any():
        return dict(source_valid_mask=output,receipt=dict(status='pending_current_hard_source_mask_intersects_original_context',
            original_candidate_index=original_candidate_index,cleared_pixels=0,
            original_context_pixels_sha256=context['actual_native_context_pixel_indices_sha256']))
    # Only pixels inside the original uniquely associated primary box change.
    # A historical proof cannot expand an ice mask or clear an entire box.
    allowed=primary[y,x]
    count=int((~output[y[allowed],x[allowed]]).sum())
    output[y[allowed],x[allowed]]=True
    return dict(source_valid_mask=output,receipt=dict(status='original_exact_current_primary_box_context_reused',
        original_candidate_index=original_candidate_index,cleared_pixels=count,
        original_context_pixel_entries=len(pixels),original_context_pixels_sha256=context['actual_native_context_pixel_indices_sha256'],
        only_certified_primary_box_context_changed=True,hard_source_exclusions_cleared=False,
        source_named_identity_or_pixel_accuracy_admitted=False))
