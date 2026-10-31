"""Reread existing contact queries from ordered source material interfaces.

Neighbouring columns describe the same visible material band. They are search
support, not extra fit observations. Every returned UV is the public reader's
current RGB half-transition; neither the guide nor a fitted curve supplies it.
"""
from __future__ import annotations

import numpy as np


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def read_board_contacts(rgb, old_uv, exclusion, *, binding, authority, spacing=24.):
    import torch
    from rinkspace.vision_source_topology import read_board_foot_material_profiles
    from rinkspace.session.chromatic_line_navigation import source_board_foot_component_support

    old=np.asarray(old_uv, float).reshape(-1,2)
    mask=np.asarray(exclusion, bool)
    if len(old)<2 or not np.isfinite(old).all() or spacing<=0:
        raise ValueError('At least two finite source navigation queries and positive spacing required')
    if mask.shape!=rgb.shape[:2]:
        raise ValueError('Source exclusion and original RGB differ')
    order=np.argsort(old[:,0],kind='stable');nav=old[order]
    columns=np.arange(np.ceil(nav[0,0]/spacing)*spacing,nav[-1,0],spacing)
    columns=columns[~np.isin(columns,old[:,0])]
    extra=np.c_[columns,np.interp(columns,nav[:,0],nav[:,1])]
    unsorted=np.vstack([old,extra]);sort=np.argsort(unsorted[:,0],kind='stable')
    queries=unsorted[sort];old_query_indices=np.argsort(sort)[:len(old)]
    component_support,component_evidence=source_board_foot_component_support(rgb,mask)
    packet=read_board_foot_material_profiles(rgb,
        dict(source_binding=binding,instances=[dict(object_id='current_source_exclusion',
            kind='occluder',source_binding=binding,region_mask=torch.as_tensor(mask))]),
        dict(source_binding=binding),material_role_authority=authority,
        source_query_points_uv=queries,device='cpu',sample_spacing_px=spacing,
        source_material_component_support=component_support)
    lower=packet['readings'][0];pairs=packet['same_material_band_pair_evidence']
    fall=pairs['first_iceward_yellow_material_fall_evidence'];profiles=packet['original_RGB_profiles']
    intervals=_array(fall['actual_warm_gray_80_to_20_fall_support_native_px'])
    accepted=_array(lower['primary_role_samples']).astype(bool).copy()
    accepted &= _array(fall['supported']).astype(bool)
    accepted &= _array(fall['actual_warm_gray_80_to_20_fall_support_complete']).astype(bool)
    accepted &= _array(pairs['competing_same_band_pairs'])==0
    sampled=_array(profiles['sampled_native_uv'])
    offsets=_array(profiles['offsets_native_px'])
    entry_offsets=_array(profiles['rgb_gradient_offsets_native_px'])[_array(pairs['entry_index']).astype(int)]
    mask_conflict=np.zeros(len(queries),bool)
    for i in range(len(queries)):
        # Check the actually measured yellow material and its lower fall,
        # including derivative support. The old contact mask is never used.
        support=(offsets>=entry_offsets[i]-.5)&(offsets<=intervals[i,1]+.5)
        pixels=np.rint(sampled[i,support]).astype(int)
        if not len(pixels):mask_conflict[i]=True;continue
        valid=(pixels>=0).all(1)&(pixels<[mask.shape[1],mask.shape[0]]).all(1)
        mask_conflict[i]=(not valid.all()) or mask[pixels[valid,1],pixels[valid,0]].any()
    accepted &= ~mask_conflict
    measured=_array(lower['candidate_source_uv'])
    accepted &= np.isfinite(measured).all(1)
    original=old_query_indices
    old_outside=(intervals[original,0]>0)|(intervals[original,1]<0)
    return dict(schema='rinkspace.current-source-board-contact-readings.v1',
        source_binding=binding,queries=queries,old_query_indices=original,
        candidate_source_uv=measured,source_material_accepted=accepted,
        old_query_supported=accepted[original],old_outside_measured_transition=old_outside,
        old_query_fresh_uv=measured[original],source_mask_conflict=mask_conflict,
        transition_offsets_80_20=intervals,material_entry_offsets=entry_offsets,
        competing_same_band_pairs=_array(pairs['competing_same_band_pairs']),
        original_RGB_profiles={key:_array(profiles[key]) for key in
            ('rgb_profile','sampled_native_uv','offsets_native_px','valid_mask')},
        source_role_flags=lower['flags'],material_role_authority=authority,
        source_query_spacing_px=float(spacing),source_2D_material_component_evidence=component_evidence,
        source_2D_material_selected_pair_support_fraction=_array(pairs['selected_source_component_support_fraction']),
        dense_queries_are_fit_observations=False,
        source_UV_synthesized=False,fit_eligible=False)
