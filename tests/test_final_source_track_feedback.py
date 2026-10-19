from copy import deepcopy
import hashlib

import numpy as np
from rinkspace.vision_ice_model_context_identity import SameStateIceModelContextIdentity
from rinkspace.vision_ice_temporal_identity import TemporalIceIdentitySelector
from test_vision_ice_model_context_identity import source


def selector_with_context(tmp_path):
    authority=tmp_path/'authority.json';authority.write_text('{}')
    reference=dict(path=str(authority),sha256=hashlib.sha256(authority.read_bytes()).hexdigest())
    inventory=[dict(curve_id='B1:segment0',feature_id='B1',world_curve_role='segment0',
        operator='paint_center',material_signal='blue')]
    context=SameStateIceModelContextIdentity.__new__(SameStateIceModelContextIdentity)
    context.inventory=inventory;context.source_sha256='a'*64;context.time_base='1/1000';context.authority=reference
    context.state_reference=context.scene_reference=context.camera_reference=reference
    context.source_negative_materials=[];context.feature_kind={'B1':'polyline'}
    context.previous_gray=context.previous_mask=context.previous_binding=None;context.previous_tracks={}
    context.navigation=lambda m:[[np.array([[0.,m['candidate_uv'][0,1]],[239.,m['candidate_uv'][0,1]]])]]
    return TemporalIceIdentitySelector(inventory,source_sha256='a'*64,time_base='1/1000',
        model_context_identity=context)


def consume(selector,pts,shift=0):
    rgb,measurement=source(pts,shift);mask=np.zeros(rgb.shape[:2],bool)
    selected=selector.consume(rgb,measurement,mask,source_ice_scope_mask=~mask)
    return rgb,measurement,selected


def test_first_frame_material_preparation_survives_until_real_two_image_identity(tmp_path):
    selector=selector_with_context(tmp_path)
    rgb,m,selected=consume(selector,10)
    assert selected['rows']==[]
    selector.retain_final_source_rows(rgb,m,selected,final_source_node_ids=np.arange(len(m['candidate_uv'])))
    assert selector.model_context_identity.previous_tracks['B1:segment0']['source_context_identity_pending'] is True
    rgb,m,selected=consume(selector,20,2)
    assert len(selected['rows'])==1
    assert selected['rows'][0]['temporal_evidence']['previous_actual_pts']==10


def test_rejected_final_material_cannot_become_next_frame_identity_evidence(tmp_path):
    selector=selector_with_context(tmp_path)
    rgb,m,selected=consume(selector,10)
    selector.retain_final_source_rows(rgb,m,selected,final_source_node_ids=np.arange(len(m['candidate_uv'])))
    rgb,m,selected=consume(selector,20,2)
    assert selected['rows']
    # The current entity had been selected, then its exact source support was
    # rejected by the final graph. It must not revert to first-frame pending.
    selected['rows']=[]
    selector.retain_final_source_rows(rgb,m,selected,final_source_node_ids=np.arange(len(m['candidate_uv'])))
    assert selector.tracks=={}
    assert selector.model_context_identity.previous_tracks=={}
    _,_,next_selected=consume(selector,30,3)
    assert next_selected['rows']==[]


def test_final_q50_subset_rebuilds_all_source_track_arrays_from_same_RGB(tmp_path):
    selector=selector_with_context(tmp_path)
    rgb,m,selected=consume(selector,10)
    selector.retain_final_source_rows(rgb,m,selected,final_source_node_ids=np.arange(len(m['candidate_uv'])))
    rgb,m,selected=consume(selector,20,2)
    row=deepcopy(selected['rows'][0]);ids=np.asarray(row['original_candidate_indices'])[1:4]
    sides=np.asarray(m['candidate_sides'],float)
    quantiles=np.repeat(sides[:,:,None,:],3,axis=2)
    quantiles[:,:,0,1]-=.3;quantiles[:,:,1,1]+=.125;quantiles[:,:,2,1]+=.3
    m['candidate_side_quantiles_uv']=quantiles
    row['original_candidate_indices']=ids.tolist()
    row['uv']=quantiles[ids,:,1].mean(1).tolist()
    row['source_sides_uv']=quantiles[ids,:,1].tolist()
    selected['rows']=[row]
    selector.retain_final_source_rows(rgb,m,selected,final_source_node_ids=ids)
    for track in (selector.tracks['B1:segment0'],selector.model_context_identity.previous_tracks['B1:segment0']):
        assert np.array_equal(track['source_candidate_indices'],ids)
        assert np.array_equal(track['actual_uv'],quantiles[ids,:,1].mean(1))
        assert np.array_equal(track['candidate_sides'],quantiles[ids,:,1])
        assert len(track['material_descriptors'])==len(track['source_normals'])==len(track['source_query_indices'])==3
        assert track['actual_pts']==track['navigation_pts']==20


def test_pending_preparation_does_not_retain_rejected_graph_nodes(tmp_path):
    selector=selector_with_context(tmp_path)
    rgb,m,selected=consume(selector,10)
    selector.retain_final_source_rows(rgb,m,selected,final_source_node_ids=np.empty(0,np.int64))
    assert selector.model_context_identity.previous_tracks=={}
    _,_,next_selected=consume(selector,20,2)
    assert next_selected['rows']==[]


def test_historical_missing_material_keeps_reentry_without_stale_image_LK(tmp_path,monkeypatch):
    selector=selector_with_context(tmp_path)
    for pts,shift in ((10,0),(20,2)):
        rgb,m,selected=consume(selector,pts,shift)
        selector.retain_final_source_rows(rgb,m,selected,final_source_node_ids=np.arange(len(m['candidate_uv'])))
    rgb,m=source(30,3);mask=np.zeros(rgb.shape[:2],bool)
    m['candidate_flags']|=np.uint8(2)
    selected=selector.consume(rgb,m,mask,source_ice_scope_mask=~mask)
    assert selected['rows']==[]
    selector.retain_final_source_rows(rgb,m,selected,final_source_node_ids=np.empty(0,np.int64))
    held=selector.tracks['B1:segment0']
    assert held['actual_pts']==20 and held['navigation_pts']==30
    def reject_stale_LK(*args,**kwargs):
        raise AssertionError('Older actual pixels cannot be tracked in the current held gray image')
    monkeypatch.setattr('cv2.calcOpticalFlowPyrLK',reject_stale_LK)
    rgb,m=source(40,4);m['candidate_flags']|=np.uint8(2)
    assert selector.consume(rgb,m,mask,source_ice_scope_mask=~mask)['rows']==[]
