import hashlib
import json

import cv2
import numpy as np

from rinkspace.vision_ice_model_context_identity import (
    SameStateIceModelContextIdentity,source_paint_pixel_accuracy_check,_actual_curved_paint_structure)
from rinkspace.vision_ice_temporal_identity import TemporalIceIdentitySelector


def source(pts,shift=0):
    rgb=np.full((140,240,3),220,np.uint8)
    for x in range(240):rgb[50+shift:74+shift,x]=[80+x%17,90+x%11,190]
    rgb=cv2.GaussianBlur(rgb,(3,3),.6)
    uv=np.array([[x,61.5+shift] for x in [45,75,105,135,165,195]],np.float32)
    sides=np.stack([uv-[0,12.],uv+[0,12.]],axis=1)
    m=dict(source_binding=dict(source_sha256='a'*64,time_base='1/1000',actual_pts=pts,source_pts=pts,native_pts=pts,
        shot_id='one',decoded_native_BGR_sha256=hashlib.sha256(rgb[:,:,::-1].tobytes()).hexdigest()),
        decoded_native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        candidate_uv=uv,candidate_sides=sides,candidate_query=np.arange(len(uv)),candidate_flags=np.full(len(uv),8,np.uint8),
        query_curve=np.zeros(len(uv),np.int16),query_normal=np.tile([0.,1.],(len(uv),1)))
    return rgb,m


def test_grid_context_requires_two_actual_source_exposures_and_keeps_measured_centers(tmp_path):
    authority=tmp_path/'authority.json';authority.write_text('{}')
    reference=dict(path=str(authority),sha256=hashlib.sha256(authority.read_bytes()).hexdigest())
    inv=[dict(curve_id='B1:segment0',feature_id='B1',world_curve_role='segment0',operator='paint_center',material_signal='blue')]
    context=SameStateIceModelContextIdentity.__new__(SameStateIceModelContextIdentity)
    context.inventory=inv;context.source_sha256='a'*64;context.time_base='1/1000';context.authority=reference
    context.state_reference=context.scene_reference=context.camera_reference=reference
    context.source_negative_materials=[];context.feature_kind={'B1':'polyline'}
    context.previous_gray=context.previous_mask=context.previous_binding=None;context.previous_tracks={}
    context.navigation=lambda m:[[np.array([[0.,m['candidate_uv'][0,1]],[239.,m['candidate_uv'][0,1]]])]]
    selector=TemporalIceIdentitySelector(inv,source_sha256='a'*64,time_base='1/1000',model_context_identity=context)
    rgb,m=source(10);mask=np.zeros(rgb.shape[:2],bool)
    assert not selector.consume(rgb,m,mask,source_ice_scope_mask=~mask)['rows']
    rgb,m=source(20,2);rows=selector.consume(rgb,m,mask,source_ice_scope_mask=~mask)['rows']
    assert len(rows)==1
    row=rows[0]
    assert row['temporal_evidence']['kind']=='actual_RGB_material_field_order_and_continuity_with_model_navigation'
    assert row['temporal_evidence']['previous_actual_pts']==10
    assert 'model_projection_or_fit_residual_used_for_identity' not in row
    assert row['projected_UV_used_as_source_pixel_truth'] is False
    assert np.array_equal(row['uv'],np.asarray(row['source_sides_uv']).mean(1))
    from calibration.program_tracking.strict_absolute_source import _temporal_identity
    assert _temporal_identity(row)


def test_current_source_paint_neighbor_jump_is_held_without_changing_raw_values():
    rgb=np.full((140,240,3),220,np.uint8);rgb[57:68]=[190,110,110]
    x=np.arange(25,226,25);uv=np.c_[x,np.full(len(x),62.)].astype(np.float32)
    uv[4,1]=80.;rgb[76:85,121:130]=[222,220,220]
    sides=np.stack([uv-[0,5.],uv+[0,5.]],axis=1)
    m=dict(source_binding=dict(decoded_native_BGR_sha256=hashlib.sha256(rgb[:,:,::-1].tobytes()).hexdigest()),
        decoded_native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),candidate_uv=uv,candidate_sides=sides,
        candidate_flags=np.full(len(uv),8,np.uint8))
    before=sides.copy();flags=m['candidate_flags'].copy()
    result=source_paint_pixel_accuracy_check(rgb,m,np.arange(len(uv)),dict(feature_id='C',operator='paint_center',material_signal='red'))
    assert 4 not in result['accepted_candidate_indices']
    assert np.array_equal(sides,before) and np.array_equal(m['candidate_flags'],flags)
    assert result['pending'] and result['pixel_values_changed'] is False


def test_straight_red_segment_cannot_establish_circle_identity():
    points=np.c_[np.arange(0.,240.,20.),np.full(12,60.)]
    okay,evidence=_actual_curved_paint_structure(points,np.arange(len(points)))
    assert not okay and evidence['actual_source_tangent_span_rad']==0
