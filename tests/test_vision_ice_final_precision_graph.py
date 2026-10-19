import copy
import hashlib

import cv2
import numpy as np
import pytest

from rinkspace.vision_ice_final_precision_graph import PreciseSourceIceMaterialIdentityGraph


def source(tmp_path):
    rgb=np.full((140,240,3),220,np.uint8)
    for x in range(240):rgb[50:73,x]=[80+x%17,90+x%11,190]
    rgb=cv2.GaussianBlur(rgb,(3,3),.6)
    uv=np.array([[x,61.] for x in [45,75,105,135,165,195]],np.float32)
    sides=np.stack([uv-[0,11.5],uv+[0,11.5]],axis=1)
    quantiles=np.repeat(sides[:,:,None],3,axis=2)
    quantiles[:,:,0,1]-=.2;quantiles[:,:,2,1]+=.2
    measurement=dict(source_binding=dict(source_sha256='a'*64,source_pts=10,actual_pts=10,native_pts=10,
        time_base='1/1000',shot_id='one',decoded_native_BGR_sha256=hashlib.sha256(rgb[:,:,::-1].tobytes()).hexdigest()),
        decoded_native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        candidate_uv=uv,candidate_sides=sides,candidate_query=np.arange(len(uv)),
        query_curve=np.zeros(len(uv),np.int16),query_normal=np.tile([0.,1.],(len(uv),1)),
        candidate_flags=np.full(len(uv),8,np.uint8),candidate_side_quantiles_uv=quantiles)
    path=tmp_path/'source.json';path.write_text('{}')
    authority=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    seed=dict(curve_id='B1:segment0',actual_pts=10,native_pts=10,shot_id='one',source_sha256='a'*64,
        time_base='1/1000',surface_and_physical_identity_confirmed=True,source_material_class='static_painted_ice',
        authority=authority,decoded_native_RGB_sha256=measurement['decoded_native_RGB_sha256'])
    row=dict(actual_pts=10,native_pts=10,shot_id='one',source_sha256='a'*64,time_base='1/1000',
        curve_id='B1:segment0',feature_id='B1',world_curve_role='segment0',physical_material_id='B1:segment0',
        original_candidate_indices=list(range(6)),source_candidate_indices=list(range(6)),source_query_indices=list(range(6)),
        uv=sides.mean(1).tolist(),source_sides_uv=sides.tolist(),actual_source_RGB_measured=True,
        static_material_confirmed=True,pixel_location_confirmed=True,model_projection_or_fit_residual_used_for_identity=False,
        decoded_native_RGB_sha256=measurement['decoded_native_RGB_sha256'],
        decoded_native_BGR_sha256=measurement['source_binding']['decoded_native_BGR_sha256'],
        source_seed_authority=authority,source_seed_reference=seed,
        admission_status='admitted_source_material_center_temporal_identity',
        temporal_evidence=dict(kind='confirmed_original_source_identity_anchor',physical_material_id='B1:segment0',
            shot_id='one',current_actual_pts=10,same_physical_entity_confirmed=True,current_frame_RGB_measured=True,
            model_projection_or_fit_residual_used_for_identity=False,source_identity_authority=authority,source_identity_confirmed=True))
    inventory=[dict(curve_id='B1:segment0',feature_id='B1',world_curve_role='segment0',operator='paint_center',material_signal='blue')]
    graph=PreciseSourceIceMaterialIdentityGraph(inventory,source_sha256='a'*64,time_base='1/1000')
    return graph,rgb,measurement,row


def test_final_q50_colour_failure_also_removes_writer_point_and_preserves_other_proofs(tmp_path):
    graph,rgb,measurement,row=source(tmp_path)
    measurement['candidate_side_quantiles_uv'][0,:,:,1]-=35
    selected=dict(rows=[row],decisions=[]);mask=np.zeros(rgb.shape[:2],bool)
    result=graph.consume(rgb,measurement,mask,source_ice_scope_mask=~mask,selected_rows=selected)
    assert 0 not in graph.frames[-1]['ids']
    assert selected['rows'][0]['original_candidate_indices']==list(range(1,6))
    pending=result['source_current_final_pixel_accuracy']['pending_actual_source_candidates']
    assert pending[0]['original_candidate_index']==0 and not pending[0]['preserved_graph_node_exists']
    assert graph.preserve_current_source_row_proofs(selected)['preserved_current_source_rows']==1
    receipt=graph.save_compact_state(tmp_path/'graph.npz')
    assert receipt['source_precision_pending_reference']['sha256']
    assert measurement['candidate_uv'][0,1]==61.  # No old readout was changed.


def test_final_node_and_row_use_exact_measured_q50_readouts(tmp_path):
    graph,rgb,measurement,row=source(tmp_path)
    measurement['candidate_side_quantiles_uv'][:,:,:,1]+=.031
    selected=dict(rows=[row],decisions=[]);mask=np.zeros(rgb.shape[:2],bool)
    graph.consume(rgb,measurement,mask,source_ice_scope_mask=~mask,selected_rows=selected)
    fine=measurement['candidate_side_quantiles_uv'][:,:,1]
    assert np.array_equal(selected['rows'][0]['source_sides_uv'],fine)
    assert np.array_equal(selected['rows'][0]['uv'],fine.mean(1))
    assert graph.preserve_current_source_row_proofs(selected)['preserved_current_source_rows']==1


def test_final_precision_cannot_repair_forged_selector_uv_by_nearest_candidate(tmp_path):
    graph,rgb,measurement,row=source(tmp_path)
    row=copy.deepcopy(row);row['uv'][0][1]+=1e-8
    mask=np.zeros(rgb.shape[:2],bool)
    with pytest.raises(ValueError,match='exact saved raw or q50'):
        graph.consume(rgb,measurement,mask,source_ice_scope_mask=~mask,selected_rows=dict(rows=[row]))


def test_rejected_selector_anchor_cannot_label_delayed_nodes_or_become_future_parent(tmp_path):
    from rinkspace import vision_ice_final_precision_graph as module
    graph,rgb,measurement,row=source(tmp_path)
    mask=np.zeros(rgb.shape[:2],bool)
    # The raw lower interface context ends at y=80.5 (rounds to 80). Its
    # actually measured q50 in the next source image crosses the excluded
    # native pixel at 81. Neither color support nor center-only scope sees it.
    mask[81,45]=True
    graph.consume(rgb,measurement,mask,source_ice_scope_mask=~mask)
    assert 0 in graph.frames[0]['ids']
    current=copy.deepcopy(measurement)
    current['source_binding'].update(source_pts=20,actual_pts=20,native_pts=20)
    current['candidate_side_quantiles_uv'][0,:,:,1]+=.031
    row=module._subset_source_row(row,np.array([True,False,False,False,False,False]))
    row.update(actual_pts=20,native_pts=20)
    row['temporal_evidence']['current_actual_pts']=20
    selected=dict(rows=[row],decisions=[])
    result=graph.consume(rgb,current,mask,source_ice_scope_mask=~mask,selected_rows=selected)
    assert not selected['rows'] and not result['rows']
    assert all(not frame['label'] for frame in graph.frames)
    assert 0 not in graph.frames[-1]['ids']
    assert graph.preserve_current_source_row_proofs(selected)['preserved_current_source_rows']==0
    current['source_binding'].update(source_pts=30,actual_pts=30,native_pts=30)
    third=graph.consume(rgb,current,mask,source_ice_scope_mask=~mask)
    assert not third['rows'] and all(not frame['label'] for frame in graph.frames)
    assert all(graph.frames[-2]['ids'][parent[1]]!=0 for parent in graph.frames[-1]['parents'].values())


def test_filtered_final_source_row_slices_all_point_support_and_clearance(tmp_path):
    graph,rgb,measurement,row=source(tmp_path)
    row['source_readout_uncertainty']=dict(
        actual_readout_blur_interval_px=[[-.2,.2]]*6,
        normal_interval_px=[[-.7,.7]]*6,quantization_margin_px=[.5]*6,
        actual_side_gradient_quantiles_uv=measurement['candidate_side_quantiles_uv'].tolist())
    row['sigma_px']=[.7]*6
    row['continuity_evidence']=dict(current_actual_paint_widths_px=[23.]*6,
        current_source_RGB_material_context_correlations=[1.]*6)
    row['candidate_local_source_clearance']={0:{'candidate':0},1:{'candidate':1}}
    original=copy.deepcopy(row)
    measurement['candidate_side_quantiles_uv'][0,:,:,1]-=35
    selected=dict(rows=[row],decisions=[]);mask=np.zeros(rgb.shape[:2],bool)
    graph.consume(rgb,measurement,mask,source_ice_scope_mask=~mask,selected_rows=selected)
    final=selected['rows'][0]
    assert final['original_candidate_indices']==list(range(1,6))
    assert all(len(value)==5 for value in final['source_readout_uncertainty'].values())
    assert np.array_equal(final['source_readout_uncertainty']['actual_side_gradient_quantiles_uv'],
        np.asarray(original['source_readout_uncertainty']['actual_side_gradient_quantiles_uv'])[1:])
    assert len(final['sigma_px'])==5
    assert all(len(value)==5 for value in final['continuity_evidence'].values())
    assert final['candidate_local_source_clearance']=={1:{'candidate':1}}
    assert row==original


def test_unselected_final_node_rejections_remain_in_saved_source_evidence(tmp_path):
    import json
    graph,rgb,measurement,row=source(tmp_path)
    mask=np.zeros(rgb.shape[:2],bool);mask[81,45]=True
    measurement['candidate_side_quantiles_uv'][0,:,:,1]+=.031
    result=graph.consume(rgb,measurement,mask,source_ice_scope_mask=~mask)
    review=result['source_current_final_pixel_accuracy']
    assert not review['pending_actual_source_candidates']
    assert review['final_source_node_rejections']['0']=='final_q50_source_context_is_excluded_or_surface_unknown'
    receipt=graph.save_compact_state(tmp_path/'unselected.npz')
    with np.load(receipt['source_precision_pending_reference']['path'],allow_pickle=False) as saved:
        evidence=json.loads(saved['pending_actual_source_receipts_utf8'].tobytes().decode())
    assert evidence['source_precision_receipts'][0]['final_source_node_rejections']==review['final_source_node_rejections']
    assert evidence['source_points_not_admitted'] is True
