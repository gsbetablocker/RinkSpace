"""Source identity may use the same grid as context, never as pixel truth.

Historical projection-independent facts remain literal. The new branch
requires measured material, field order, physical scope and source continuity
with the exact original frame binding and authenticated source authority.
"""
from __future__ import annotations

from copy import deepcopy


GRID_CONTEXT_FIELDS = (
    'model_projection_used_as_navigation_and_identity_context',
    'projected_UV_used_as_source_pixel_truth',
    'source_identity_not_confirmed_from_projection_alone',
    'fit_residual_used_for_identity',
    'source_identity_navigation_evidence',
)


def source_identity_context_fields(value):
    if value.get('model_projection_used_as_navigation_and_identity_context') is not True:
        return {}
    fields={key:deepcopy(value[key]) for key in GRID_CONTEXT_FIELDS if key in value}
    if 'model_projection_or_fit_residual_used_for_identity' in value:
        fields['model_projection_or_fit_residual_used_for_identity']=value['model_projection_or_fit_residual_used_for_identity']
    return fields


def admissible_source_identity_context(value, *, expected=None, source=None):
    """Validate old independent evidence or the explicit measured-grid branch."""
    if value.get('model_projection_used_as_navigation_and_identity_context') is not True:
        return value.get('model_projection_or_fit_residual_used_for_identity') is False
    if (value.get('projected_UV_used_as_source_pixel_truth') is not False
            or value.get('source_identity_not_confirmed_from_projection_alone') is not True
            or value.get('fit_residual_used_for_identity') is not False
            or value.get('model_projection_or_fit_residual_used_for_identity') not in (None,True)
            or value.get('model_projection_or_fit_residual_used_for_identity') is False):
        return False
    evidence=value.get('source_identity_navigation_evidence')
    if (not isinstance(evidence,dict)
            or evidence.get('kind')!='actual_RGB_material_field_order_and_continuity_with_model_navigation'
            or any(evidence.get(key) is not True for key in (
                'actual_material_correspondence_confirmed','actual_field_order_confirmed',
                'physical_model_scope_confirmed','same_shot_source_continuity_confirmed'))):
        return False
    binding=evidence.get('source_binding',{})
    if (not isinstance(binding,dict)
            or any(type(binding.get(key)) is not int for key in ('actual_pts','native_pts'))
            or len(binding.get('source_sha256',''))!=64
            or len(binding.get('decoded_native_BGR_sha256',''))!=64
            or not binding.get('time_base') or not binding.get('shot_id')):
        return False
    expected=value if expected is None else expected
    frame=dict(expected.get('source_binding',{}))
    frame.update({key:expected[key] for key in ('actual_pts','native_pts','shot_id') if key in expected})
    frame['source_sha256']=expected.get('source_sha256',expected.get('video_sha256',frame.get('source_sha256',frame.get('video_sha256'))))
    frame['time_base']=expected.get('time_base',frame.get('time_base'))
    frame['decoded_native_BGR_sha256']=expected.get('decoded_native_BGR_sha256',
        expected.get('decoded_BGR_sha256',frame.get('decoded_native_BGR_sha256',frame.get('decoded_BGR_sha256'))))
    if source is not None:
        frame.update(source_sha256=source['video_sha256'],time_base=source['time_base'])
    for key in ('source_sha256','time_base','actual_pts','native_pts','decoded_native_BGR_sha256','shot_id'):
        if frame.get(key) is None or frame[key]!=binding[key]:return False
    from .strict_model_scope import authenticate_source_reference
    for key in ('source_identity_authority','source_model_navigation_reference'):
        reference=evidence.get(key)
        if not isinstance(reference,dict) or not reference.get('path') or len(reference.get('sha256',''))!=64:
            return False
        authenticate_source_reference(reference)
    return True
