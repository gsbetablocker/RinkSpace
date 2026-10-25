from copy import deepcopy

import numpy as np
import pytest

from calibration.program_tracking.strict_absolute_source import (
    StrictAbsolutePackWriter,StrictAbsoluteSourcePack,normalize_selected_ice_rows)
from calibration.program_tracking.source_identity_navigation import admissible_source_identity_context
from calibration.program_tracking.test_strict_absolute_source import fixture_inputs


def _grid_context(binding,row):
    authority=row['temporal_evidence']['source_identity_authority']
    return dict(model_projection_used_as_navigation_and_identity_context=True,
        projected_UV_used_as_source_pixel_truth=False,
        source_identity_not_confirmed_from_projection_alone=True,
        fit_residual_used_for_identity=False,
        model_projection_or_fit_residual_used_for_identity=True,
        source_identity_navigation_evidence=dict(
            kind='actual_RGB_material_field_order_and_continuity_with_model_navigation',
            source_binding=dict(binding,shot_id=row['shot_id']),
            actual_material_correspondence_confirmed=True,actual_field_order_confirmed=True,
            physical_model_scope_confirmed=True,same_shot_source_continuity_confirmed=True,
            source_identity_authority=authority,source_model_navigation_reference=authority))


def test_actual_centres_with_grid_identity_context_pass_without_forging_old_flag(tmp_path):
    binding,row,measurement,inventory=fixture_inputs(tmp_path)
    context=_grid_context(binding,row)
    row.update(context);row['source_binding']=dict(binding,shot_id=row['shot_id'])
    row['temporal_evidence'].update(context)
    normalized=normalize_selected_ice_rows([row],measurement,
        source_binding=binding,curve_inventory=inventory)
    assert len(normalized['rows'])==1
    endpoint=next(iter(normalized['review'].records.values()))
    assert endpoint['model_projection_or_fit_residual_used_for_identity'] is True
    writer=StrictAbsolutePackWriter(tmp_path/'grid_context.npz',source=normalized['review'].source,
        expected_actual_pts=[100])
    writer.add_row(normalized['rows'][0],normalized['review']);writer.mark_exposure_processed(100)
    receipt=writer.finish();pack=StrictAbsoluteSourcePack(receipt['path'],expected_sha256=receipt['sha256'])
    assert np.array_equal(pack.values['uv'],measurement['candidate_sides'].astype(float).mean(1))
    assert pack.rows[0]['model_projection_used_as_navigation_and_identity_context'] is True


def test_projection_only_wrong_RGB_and_hidden_residual_identity_stay_rejected(tmp_path):
    binding,row,measurement,inventory=fixture_inputs(tmp_path)
    context=_grid_context(binding,row)
    value=dict(row,source_binding=dict(binding,shot_id=row['shot_id']),**context)
    assert admissible_source_identity_context(value)
    for key in ('projected_UV_used_as_source_pixel_truth','fit_residual_used_for_identity'):
        changed=deepcopy(value);changed[key]=True
        assert not admissible_source_identity_context(changed)
    for key in ('actual_material_correspondence_confirmed','actual_field_order_confirmed',
                'physical_model_scope_confirmed','same_shot_source_continuity_confirmed'):
        changed=deepcopy(value);changed['source_identity_navigation_evidence'][key]=False
        assert not admissible_source_identity_context(changed)
    changed=deepcopy(value);changed['model_projection_or_fit_residual_used_for_identity']=False
    assert not admissible_source_identity_context(changed)
    changed=deepcopy(value)
    changed['source_identity_navigation_evidence']['source_binding']['decoded_native_BGR_sha256']='c'*64
    assert not admissible_source_identity_context(changed)
    changed=deepcopy(value);changed['source_identity_navigation_evidence']['source_binding']['native_pts']=110.
    assert not admissible_source_identity_context(changed)


def test_grid_context_anchor_requires_actual_second_exposure_continuity(tmp_path):
    from calibration.program_tracking.strict_absolute_source import _temporal_identity
    binding,row,measurement,inventory=fixture_inputs(tmp_path)
    context=_grid_context(binding,row)
    row.update(context);row['source_binding']=dict(binding,shot_id=row['shot_id'])
    row['temporal_evidence'].update(context,
        kind='actual_RGB_material_field_order_and_continuity_with_model_navigation',
        previous_actual_pts=90,previous_shot_id=row['shot_id'],previous_decoded_BGR_sha256='d'*64,
        previous_frame_RGB_measured=True,source_RGB_tracking_confirmed=True,
        current_feature_semantics_remeasured=True)
    assert _temporal_identity(row)
    row['temporal_evidence']['previous_actual_pts']=100
    assert not _temporal_identity(row)


def test_current_native_source_navigation_keeps_nonprojection_proof_through_writer(tmp_path):
    from calibration.program_tracking.strict_absolute_source import _temporal_identity
    binding,row,measurement,inventory=fixture_inputs(tmp_path)
    context=_grid_context(binding,row)
    authority=context['source_identity_navigation_evidence']['source_identity_authority']
    context.update(model_projection_used_as_navigation_and_identity_context=False,
        model_projection_or_fit_residual_used_for_identity=False,verified_source_guides_used_as_navigation=True)
    ev=context['source_identity_navigation_evidence']
    ev.update(kind='actual_RGB_material_field_order_and_continuity_with_native_source_guides',
        same_native_camera_reference=None,same_scene_reference=authority,
        source_identity_authority=dict(authority,original_feature_identity_authorities={'line':authority}))
    row.update(context);row['source_binding']=dict(binding,shot_id=row['shot_id'])
    row['temporal_evidence'].update(context,kind=ev['kind'],source_identity_authority=ev['source_identity_authority'],
        previous_actual_pts=90,previous_shot_id=row['shot_id'],previous_decoded_BGR_sha256='d'*64,
        previous_frame_RGB_measured=True,source_RGB_tracking_confirmed=True,current_feature_semantics_remeasured=True)
    assert admissible_source_identity_context(row) and _temporal_identity(row)
    normalized=normalize_selected_ice_rows([row],measurement,source_binding=binding,curve_inventory=inventory)
    assert len(normalized['rows'])==1
    writer=StrictAbsolutePackWriter(tmp_path/'native_source_context.npz',source=normalized['review'].source,expected_actual_pts=[100])
    writer.add_row(normalized['rows'][0],normalized['review']);writer.mark_exposure_processed(100)
    receipt=writer.finish();pack=StrictAbsoluteSourcePack(receipt['path'],expected_sha256=receipt['sha256'])
    assert np.array_equal(pack.values['uv'],measurement['candidate_sides'].astype(float).mean(1))
    assert pack.rows[0]['model_projection_used_as_navigation_and_identity_context'] is False
    assert pack.rows[0]['verified_source_guides_used_as_navigation'] is True
    wrong=deepcopy(row);wrong['source_identity_navigation_evidence']['same_native_camera_reference']=authority
    assert not admissible_source_identity_context(wrong)
