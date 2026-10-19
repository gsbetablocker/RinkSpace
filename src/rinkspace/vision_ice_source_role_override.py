"""Read source-confirmed event roles again in the shared current original RGB.

Frozen historical measurements are never relabeled into another operator.
An override changes the effective semantic inventory and replaces the affected
curve's candidates with new native source readouts, with original clock proof.
"""
from __future__ import annotations

import numpy as np

from .vision_programme_ice_review import (
    _actual_profile_candidates,_batched_actual_profiles,_candidate_context_excluded,STATUS)
from .vision_source_shapes import _MaterialProfileFrame


def effective_source_role_inventory(inventory,overrides):
    effective=[dict(v) for v in inventory]
    known={v['curve_id']:v for v in effective}
    for name,role in overrides.items():
        if name not in known:raise ValueError('Source override names an unmodeled curve')
        if role.get('confirmed_measurement_operator') not in ('paint_center','material_interface'):
            raise ValueError('An actual source operator is required')
        if role.get('confirmed_material_signal') not in ('red','blue','dark'):
            raise ValueError('An actual source material signal is required')
        authority=role.get('source_role_evidence',{})
        references=[authority] if 'path' in authority else [v for v in authority.values() if isinstance(v,dict) and 'path' in v]
        if not references:
            raise ValueError('Source role override requires preserved authenticated evidence')
        from calibration.program_tracking.strict_model_scope import authenticate_source_reference
        try:
            for reference in references:authenticate_source_reference(reference)
        except (OSError,ValueError,KeyError) as error:
            raise ValueError('Source role override requires preserved authenticated evidence') from error
        if (known[name].get('source_role_authority') == 'authenticated_feedback_finite_width_midline'
                and role['confirmed_measurement_operator'] != 'paint_center'):
            raise ValueError('Current authenticated finite-width midline cannot be replaced by a stale material-interface override')
        known[name].update(operator=role['confirmed_measurement_operator'],
            material_signal=role['confirmed_material_signal'],
            material_transition=role.get('confirmed_material_transition'),
            source_role_authority=role.get('semantic_definition','source_confirmed_event_material_role'),
            source_role_evidence=dict(authority))
    return effective


def remeasure_confirmed_source_roles(rgb,measurement,exclusion_mask,inventory,overrides,*,profile_backend='cpu'):
    """Return a complete current measurement overlay and its effective roles.

    Old candidates of affected curves become invalid; all other candidates
    retain their array positions, and new real measurements occupy appended slots.
    ``original_candidate_indices`` therefore refer to this returned overlay.
    A caller must preserve this overlay for delayed temporal row normalization.
    The old cache/inventory are unchanged. No image decoding or fitting occurs.
    """
    effective=effective_source_role_inventory(inventory,overrides)
    frame=_MaterialProfileFrame(rgb,measurement['source_binding'],measurement['decoded_native_RGB_sha256'])
    mask=np.asarray(exclusion_mask,bool)
    if mask.shape!=rgb.shape[:2]:raise ValueError('Source role measurement needs this frame native mask')
    qcurve=np.asarray(measurement['query_curve'])
    query=np.asarray(measurement['candidate_query'])
    curve=np.asarray([v['curve_id'] for v in inventory])
    affected=np.flatnonzero(np.isin(curve,list(overrides)))
    qi=np.flatnonzero(np.isin(qcurve,affected))
    result=dict(measurement)
    # Keep unaffected indices stable. Old affected candidates are permanently
    # excluded from this overlay, never granted a new material interpretation.
    result['candidate_flags']=np.asarray(measurement['candidate_flags']).copy()
    result['candidate_flags'][np.isin(qcurve[query],affected)] |= 1
    result['candidate_source_role_remeasured']=np.zeros(len(query),bool)
    if not len(qi):
        result['effective_source_role_inventory']=effective
        return result,effective
    points=np.asarray(measurement['query_uv'],float)[qi]
    normals=np.asarray(measurement['query_normal'],float)[qi]
    intervals=np.asarray(measurement['query_interval'],float)[qi]
    signals=np.array([effective[int(qcurve[i])]['material_signal'] for i in qi])
    if profile_backend=='cuda':
        from .vision_source_cuda_profiles import batched_actual_profiles_cuda
        offsets,profiles,smooth,valid=batched_actual_profiles_cuda(frame,points,normals,intervals,signals)
    elif profile_backend=='cpu':
        offsets,profiles,smooth,valid=_batched_actual_profiles(frame,points,normals,intervals,signals)
    else:raise ValueError('Current source profile backend must be cpu or cuda')
    added={key:[] for key in ('candidate_uv','candidate_sides','candidate_query','candidate_flags','candidate_side_quantiles_uv')}
    statuses=np.asarray(measurement.get('query_status',np.zeros(len(qcurve),np.uint8))).copy()
    for local,global_q in enumerate(qi):
        role=effective[int(qcurve[global_q])];take=valid[local]
        values=_actual_profile_candidates(points[local],normals[local],offsets[local,take],profiles[local,take],smooth[local,take],
            operator=role['operator'],signal=role['material_signal'],transition=role['material_transition'],with_quantiles=True)
        clean=0
        for uv,sides,bad,quantiles in values:
            excluded=any(_candidate_context_excluded(mask,p,normals[local]) for p in (uv,*sides))
            added['candidate_uv'].append(uv);added['candidate_sides'].append(sides)
            added['candidate_query'].append(global_q)
            added['candidate_flags'].append(int(bad)|(int(excluded)<<1)|(int(len(values)>1)<<2)|8)
            added['candidate_side_quantiles_uv'].append(quantiles)
            clean+=not bad and not excluded
        statuses[global_q]=STATUS['source_material_missing'] if not values else (
            STATUS['actual_RGB_candidate_pending_review'] if clean==len(values)==1 else STATUS['competing_or_unresolved_material'])
    shapes={'candidate_uv':(-1,2),'candidate_sides':(-1,2,2),'candidate_side_quantiles_uv':(-1,2,3,2),
        'candidate_query':(-1,),'candidate_flags':(-1,)}
    for key,items in added.items():
        old=result.get(key)
        if old is None and key=='candidate_side_quantiles_uv':old=np.full((len(query),2,3,2),np.nan)
        dtype=np.float64 if key=='candidate_side_quantiles_uv' else np.asarray(old).dtype
        new=np.asarray(items,dtype=dtype).reshape(shapes[key])
        result[key]=np.concatenate([old,new])
    result.update(query_status=statuses,candidate_source_role_remeasured=np.r_[np.zeros(len(query),bool),np.ones(len(added['candidate_uv']),bool)],
        effective_source_role_inventory=effective,source_readout_quantization_step_px=1.,source_interpolation_grid_px=1./32.,
        source_role_override_actual_RGB_remeasured=True,
        source_role_override_evidence={name:role['source_role_evidence'] for name,role in overrides.items()},
        source_role_override_old_measurements_admitted=False)
    return result,effective
