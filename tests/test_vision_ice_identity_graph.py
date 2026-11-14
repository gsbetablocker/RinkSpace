import hashlib
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from rinkspace.vision_ice_identity_graph import IceSourceMaterialIdentityGraph


def frame(pts, shift=0, shot='one'):
    rgb=np.full((140,240,3),220,np.uint8)
    for x in range(240):
        rgb[50+shift:73+shift,x]=[80+(x%17),90+(x%11),190]
    rgb=cv2.GaussianBlur(rgb,(3,3),.6)
    uv=np.array([[x,61.+shift] for x in [45,75,105,135,165,195]],np.float32)
    sides=np.stack([uv-[0,11.5],uv+[0,11.5]],axis=1)
    measurement=dict(source_binding=dict(source_sha256='a'*64,source_pts=pts,time_base='1/1000',shot_id=shot),
        decoded_native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        candidate_uv=uv,candidate_sides=sides,candidate_query=np.arange(len(uv)),
        query_curve=np.zeros(len(uv),np.int16),query_normal=np.tile([0.,1.],(len(uv),1)),
        candidate_flags=np.full(len(uv),8,np.uint8))
    return rgb,measurement


def inventory():
    return [dict(curve_id='B1:segment0',feature_id='B1',world_curve_role='segment0',operator='paint_center')]


def anchor(measurement):
    return dict(curve_id='B1:segment0',physical_material_id='B1:segment0',
        original_candidate_indices=list(range(len(measurement['candidate_uv']))),
        source_seed_authority=dict(path='source-identity-anchor.json',sha256='b'*64),
        source_seed_reference=dict(actual_pts=measurement['source_binding']['source_pts']),
        admission_status='admitted_source_material_center_temporal_identity')


def offline_anchor(rgb,measurement,path):
    pts=measurement['source_binding']['source_pts'];sha='a'*64
    reference=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    seed=dict(curve_id='B1:segment0',actual_pts=pts,source_sha256=sha,time_base='1/1000',
        surface_and_physical_identity_confirmed=True,authority=reference)
    row=dict(anchor(measurement),actual_pts=pts,native_pts=pts,shot_id='one',source_sha256=sha,time_base='1/1000',
        decoded_native_RGB_sha256=measurement['decoded_native_RGB_sha256'],
        decoded_native_BGR_sha256=hashlib.sha256(rgb[:,:,::-1].tobytes()).hexdigest(),
        actual_source_RGB_measured=True,static_material_confirmed=True,pixel_location_confirmed=True,
        model_projection_or_fit_residual_used_for_identity=False,source_seed_authority=reference,source_seed_reference=seed,
        uv=measurement['candidate_sides'].astype(float).mean(1).tolist(),source_sides_uv=measurement['candidate_sides'].tolist(),
        source_query_indices=measurement['candidate_query'].tolist(),
        temporal_evidence=dict(kind='confirmed_original_source_identity_anchor',physical_material_id='B1:segment0',shot_id='one',
            current_actual_pts=pts,same_physical_entity_confirmed=True,current_frame_RGB_measured=True,
            model_projection_or_fit_residual_used_for_identity=False,source_identity_authority=reference,source_identity_confirmed=True))
    return row


def exact_review(row,path,image):
    from calibration.program_tracking.strict_model_scope import ExactSourceScopeReview,REVIEW_SCHEMA
    evidence=dict(path=str(image),sha256=hashlib.sha256(image.read_bytes()).hexdigest(),kind='reviewed_original_image')
    records=[dict(actual_pts=row['actual_pts'],native_pts=row['native_pts'],uv=uv,scope='ice',decision='allow',
        physical_material_id=row['physical_material_id'],static_material_confirmed=True,pixel_location_confirmed=True,
        review_authority='human_original_source_review',original_source_image_actually_reviewed=True,reviewer='test source reviewer',
        model_projection_or_fit_residual_used_for_identity=False,review_note='Specific original static paint source readout',
        decoded_BGR_sha256=row['decoded_native_BGR_sha256'],source_image_evidence=evidence) for uv in row['uv']]
    path.write_text(json.dumps(dict(schema=REVIEW_SCHEMA,source=dict(video_sha256=row['source_sha256'],time_base=row['time_base'],
        pixel_domain='native1920x1080'),reviewed_endpoints=records)))
    return ExactSourceScopeReview.load(path)


def test_future_anchor_reaches_only_real_source_side_midpoints():
    graph=IceSourceMaterialIdentityGraph(inventory(),source_sha256='a'*64,time_base='1/1000')
    cleared=np.zeros((140,240),bool)
    scope=~cleared
    first=frame(10,0);middle=frame(20,2);last=frame(30,4)
    assert not graph.consume(*first,cleared,source_ice_scope_mask=scope)['rows']
    assert not graph.consume(*middle,cleared,source_ice_scope_mask=scope)['rows']
    result=graph.consume(*last,cleared,source_ice_scope_mask=scope,selected_rows=[anchor(last[1])])
    assert {row['actual_pts'] for row in result['rows']}=={10,20}
    for row in result['rows']:
        assert row['temporal_evidence']['direction']=='backward'
        assert row['temporal_evidence']['previous_actual_pts']>row['actual_pts']
        assert np.array_equal(row['uv'],np.asarray(row['source_sides_uv']).mean(1))
        assert row['model_projection_used_as_measurement'] is False
        assert 'sigma_px' not in row  # No real gradient blur readout was supplied.


def test_saved_graph_export_recovers_source_anchor_and_real_edges_without_mutating_emitted(tmp_path):
    path=tmp_path/'source.json';path.write_text('{}')
    graph=IceSourceMaterialIdentityGraph(inventory(),source_sha256='a'*64,time_base='1/1000')
    cleared=np.zeros((140,240),bool)
    first=frame(10);middle=frame(20,2);last=frame(30,4)
    row=offline_anchor(*middle,path)
    row['source_seed_reference'].update(native_pts=20,shot_id='one',source_material_class='static_painted_ice',decoded_native_RGB_sha256=middle[1]['decoded_native_RGB_sha256'])
    graph.consume(*first,cleared,source_ice_scope_mask=~cleared)
    graph.consume(*middle,cleared,source_ice_scope_mask=~cleared,selected_rows=[row])
    graph.consume(*last,cleared,source_ice_scope_mask=~cleared)
    saved=tmp_path/'graph.npz';receipt=graph.save_compact_state(saved)
    restored=IceSourceMaterialIdentityGraph.load_compact_state(saved)
    before=[set(f['emitted']) for f in restored.frames]
    result=restored.export_confirmed_source_rows(source_graph_reference=receipt,stream=True)
    rows=list(result['rows'])
    assert {r['actual_pts'] for r in rows}=={10,20,30}
    assert next(r for r in rows if r['actual_pts']==20)['temporal_evidence']['kind']=='confirmed_original_source_identity_anchor'
    assert not result['pending'] and result['source_RGB_decoded_again'] is False
    assert before==[f['emitted'] for f in restored.frames]
    for r in rows:
        assert np.array_equal(r['uv'],np.asarray(r['source_sides_uv']).mean(1))
        assert r['recovered_from_preserved_actual_material_graph']==receipt


def test_saved_anchor_without_original_hash_binding_cannot_be_recovered_as_identity(tmp_path):
    path=tmp_path/'source.json';path.write_text('{}')
    graph=IceSourceMaterialIdentityGraph(inventory(),source_sha256='a'*64,time_base='1/1000')
    rgb,m=frame(10);row=offline_anchor(rgb,m,path)
    row['source_seed_reference'].update(native_pts=10,shot_id='one',source_material_class='static_painted_ice')  # Current original RGB identity proof was lost.
    mask=np.zeros(rgb.shape[:2],bool)
    graph.consume(rgb,m,mask,source_ice_scope_mask=~mask,selected_rows=[row])
    result=graph.export_confirmed_source_rows()
    assert not result['rows'] and result['source_proof_reachable_nodes']==0
    assert result['pending'][0]['confirmed_labels_without_recoverable_source_proof']==6


def test_future_compact_graph_keeps_original_selector_continuation_without_graph_edges(tmp_path):
    path=tmp_path/'source.json';path.write_text('{}')
    graph=IceSourceMaterialIdentityGraph(inventory(),source_sha256='a'*64,time_base='1/1000')
    first=frame(10);last=frame(20,45)  # Too far apart for the compact LK graph.
    mask=np.zeros(first[0].shape[:2],bool)
    source=offline_anchor(*first,path)
    source['source_seed_reference'].update(native_pts=10,shot_id='one',source_material_class='static_painted_ice',
        decoded_native_RGB_sha256=first[1]['decoded_native_RGB_sha256'])
    graph.consume(*first,mask,source_ice_scope_mask=~mask,selected_rows=[source])
    row=offline_anchor(*last,path);row['source_seed_reference']=source['source_seed_reference']
    row['temporal_evidence']=dict(kind='same_shot_actual_RGB_temporal_identity',physical_material_id='B1:segment0',
        shot_id='one',current_actual_pts=20,previous_actual_pts=10,previous_shot_id='one',
        previous_decoded_BGR_sha256=source['decoded_native_BGR_sha256'],same_physical_entity_confirmed=True,
        current_frame_RGB_measured=True,previous_frame_RGB_measured=True,source_RGB_tracking_confirmed=True,
        current_feature_semantics_remeasured=True,model_projection_or_fit_residual_used_for_identity=False,
        source_RGB_tracking_method='actual_selector_source_material_reentry')
    graph.consume(*last,mask,source_ice_scope_mask=~mask,selected_rows=[row])
    graph.frames[-1]['parents'].clear();graph.frames[0]['children'].clear()
    assert graph.preserve_current_source_row_proofs([row])['node_admission_changed'] is False
    saved=tmp_path/'graph.npz';receipt=graph.save_compact_state(saved)
    restored=IceSourceMaterialIdentityGraph.load_compact_state(saved)
    recovered=restored.export_confirmed_source_rows(source_graph_reference=receipt)
    final=next(r for r in recovered['rows'] if r['actual_pts']==20)
    assert final['temporal_evidence']==row['temporal_evidence']
    assert final['recovered_original_temporal_proof_without_reconstruction'] is True
    assert final['uv']==row['uv'] and not recovered['pending']
    assert recovered['preserved_original_temporal_proof_nodes']==6


def test_preserved_selector_proof_rejects_changed_actual_readout_and_authority(tmp_path):
    path=tmp_path/'source.json';path.write_text('{}')
    graph=IceSourceMaterialIdentityGraph(inventory(),source_sha256='a'*64,time_base='1/1000')
    rgb,m=frame(10);mask=np.zeros(rgb.shape[:2],bool);row=offline_anchor(rgb,m,path)
    row['source_seed_reference'].update(native_pts=10,shot_id='one',source_material_class='static_painted_ice',
        decoded_native_RGB_sha256=m['decoded_native_RGB_sha256'])
    graph.consume(rgb,m,mask,source_ice_scope_mask=~mask,selected_rows=[row])
    changed=copy.deepcopy(row);changed['uv'][0][1]+=1e-9
    with pytest.raises(ValueError,match='exact source readouts'):graph.preserve_current_source_row_proofs([changed])
    assert not graph.frames[-1].get('preserved_source_row_proofs')
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError):graph.preserve_current_source_row_proofs([row])
    assert not graph.frames[-1].get('preserved_source_row_proofs')


def test_unknown_ice_scope_and_cuts_do_not_inherit_identity():
    graph=IceSourceMaterialIdentityGraph(inventory(),source_sha256='a'*64,time_base='1/1000')
    cleared=np.zeros((140,240),bool)
    first=frame(10,0)
    graph.consume(*first,cleared,source_ice_scope_mask=~cleared)
    last=frame(20,2,shot='two')
    result=graph.consume(*last,cleared,source_ice_scope_mask=~cleared,selected_rows=[anchor(last[1])])
    assert not result['rows']
    third=frame(30,4,shot='two')
    assert not graph.consume(*third,cleared,source_ice_scope_mask=cleared)['rows']


def test_graph_cannot_bind_candidate_to_other_original_RGB():
    graph=IceSourceMaterialIdentityGraph(inventory(),source_sha256='a'*64,time_base='1/1000')
    rgb,measurement=frame(10)
    changed=rgb.copy();changed[0,0]=0
    with pytest.raises(ValueError,match='exact measured'):
        graph.consume(changed,measurement,np.zeros(rgb.shape[:2],bool),source_ice_scope_mask=np.ones(rgb.shape[:2],bool))


def test_empty_source_measurement_still_records_catalog_without_admission():
    graph=IceSourceMaterialIdentityGraph(inventory(),source_sha256='a'*64,time_base='1/1000')
    rgb,measurement=frame(10)
    for key in ['candidate_uv','candidate_sides','candidate_query','candidate_flags','query_curve','query_normal']:
        measurement[key]=measurement[key][:0]
    mask=np.zeros(rgb.shape[:2],bool)
    assert graph.source_support_candidate_indices(measurement,mask,source_ice_scope_mask=~mask).size==0
    result=graph.consume(rgb,measurement,mask,source_ice_scope_mask=~mask)
    assert not result['rows'] and result['source_catalog_entry']['actual_pts']==10
    assert result['source_catalog_entry']['actual_material_nodes']==0


def test_saved_pending_graph_replays_real_future_anchor_without_source_images(tmp_path):
    graph=IceSourceMaterialIdentityGraph(inventory(),source_sha256='a'*64,time_base='1/1000')
    cleared=np.zeros((140,240),bool)
    last=None
    for pts,shift in [(10,0),(20,2),(30,4)]:
        last=frame(pts,shift)
        graph.consume(*last,cleared,source_ice_scope_mask=~cleared)
    path=tmp_path/'pending-source-graph.npz'
    receipt=graph.save_compact_state(path)
    assert receipt['source_frames']==3 and receipt['confirmed_nodes']==0
    restored=IceSourceMaterialIdentityGraph.load_compact_state(path)
    assert restored.previous_gray is None and restored.previous_mask is None
    authority=tmp_path/'reviewed-source.json';authority.write_text(json.dumps(dict(source_identity_reviewed=True)))
    row=offline_anchor(*last,authority)
    image=tmp_path/'source.png';cv2.imwrite(str(image),last[0][:,:,::-1])
    review=exact_review(row,tmp_path/'review.json',image)
    result=restored.apply_confirmed_source_rows([row],source_scope_review=review)
    assert {row['actual_pts'] for row in result['rows']}=={10,20}
    assert all(np.array_equal(r['uv'],np.asarray(r['source_sides_uv']).mean(1)) for r in result['rows'])
    with pytest.raises(FileExistsError):graph.save_compact_state(path)


def test_restored_graph_refuses_navigation_hint_as_source_anchor(tmp_path):
    graph=IceSourceMaterialIdentityGraph(inventory(),source_sha256='a'*64,time_base='1/1000')
    rgb,m=frame(10);clear=np.zeros(rgb.shape[:2],bool)
    graph.consume(rgb,m,clear,source_ice_scope_mask=~clear)
    path=tmp_path/'pending-source-graph.npz';graph.save_compact_state(path)
    restored=IceSourceMaterialIdentityGraph.load_compact_state(path)
    with pytest.raises(ValueError,match='navigation hint'):
        restored.apply_confirmed_source_rows([dict(anchor(m),actual_pts=10)])


@pytest.mark.parametrize('changed',['uv','sides','native_pts','BGR','authority','temporal'])
def test_offline_anchor_must_match_actual_readouts_clock_and_authority(tmp_path,changed):
    graph=IceSourceMaterialIdentityGraph(inventory(),source_sha256='a'*64,time_base='1/1000')
    rgb,m=frame(10);clear=np.zeros(rgb.shape[:2],bool)
    graph.consume(rgb,m,clear,source_ice_scope_mask=~clear)
    path=tmp_path/'pending.npz';graph.save_compact_state(path)
    restored=IceSourceMaterialIdentityGraph.load_compact_state(path)
    authority=tmp_path/'reviewed.json';authority.write_text('{}')
    good=offline_anchor(rgb,m,authority);bad=copy.deepcopy(good)
    image=tmp_path/'source.png';cv2.imwrite(str(image),rgb[:,:,::-1])
    review=exact_review(good,tmp_path/'review.json',image)
    if changed=='uv':bad['uv'][0][1]+=.001
    elif changed=='sides':bad['source_sides_uv'][0][0][1]+=.001
    elif changed=='native_pts':bad['native_pts']+=1
    elif changed=='BGR':bad['decoded_native_BGR_sha256']='c'*64
    elif changed=='authority':authority.write_text('{"changed":true}')
    else:bad['temporal_evidence']['current_actual_pts']+=1
    with pytest.raises(ValueError):restored.apply_confirmed_source_rows([good,bad],source_scope_review=review)
    assert not restored.frames[0]['label']


def test_normalized_float64_gradient_q50_anchor_is_exact_not_approximately_accepted(tmp_path):
    from rinkspace.vision_source_shapes import _MaterialProfileFrame
    from rinkspace.vision_programme_ice_review import _batched_actual_profiles,_actual_profile_candidates
    rgb,m=frame(10)
    points=m['candidate_uv'].astype(float)+[0.,.00001]
    normal=m['query_normal'];interval=np.tile([-35.,35.],(len(points),1))
    source=_MaterialProfileFrame(rgb,m['source_binding'],m['decoded_native_RGB_sha256'])
    offset,profile,smooth,valid=_batched_actual_profiles(source,points,normal,interval,np.full(len(points),'blue'))
    readings=[_actual_profile_candidates(p,n,o[v],a[v],s[v],operator='paint_center',signal='blue',transition=None,
        with_quantiles=True)[0] for p,n,o,a,s,v in zip(points,normal,offset,profile,smooth,valid)]
    m['candidate_uv']=np.asarray([r[0] for r in readings],np.float32)
    m['candidate_sides']=np.asarray([r[1] for r in readings],np.float32)
    m['candidate_side_quantiles_uv']=np.asarray([r[3] for r in readings],np.float64)
    graph=IceSourceMaterialIdentityGraph(inventory(),source_sha256='a'*64,time_base='1/1000')
    mask=np.zeros(rgb.shape[:2],bool);graph.consume(rgb,m,mask,source_ice_scope_mask=~mask)
    path=tmp_path/'pending.npz';graph.save_compact_state(path)
    authority=tmp_path/'identity.json';authority.write_text('{}')
    row=offline_anchor(rgb,m,authority);q50=m['candidate_side_quantiles_uv'][:,:,1]
    row['source_sides_uv']=q50.tolist();row['uv']=q50.mean(1).tolist()
    assert not np.array_equal(q50,m['candidate_sides'].astype(float))
    image=tmp_path/'source.png';cv2.imwrite(str(image),rgb[:,:,::-1])
    review=exact_review(row,tmp_path/'review.json',image)
    restored=IceSourceMaterialIdentityGraph.load_compact_state(path)
    restored.apply_confirmed_source_rows([row],source_scope_review=review)
    assert {v['anchor_source_readout_coordinate_domain'] for v in restored.frames[0]['label'].values()}=={'actual_gradient_q50_float64'}
    restored=IceSourceMaterialIdentityGraph.load_compact_state(path);changed=copy.deepcopy(row);changed['uv'][0][1]+=1e-9
    with pytest.raises(ValueError,match='saved readouts'):restored.apply_confirmed_source_rows([changed],source_scope_review=review)


def source_translation_correspondence(graph, current, path, *, shift):
    """Exact correspondence for this test's rendered vertical translation."""
    before=graph.frames[-1]
    support=np.concatenate([before['uv'][:,None],before['sides']],axis=1)
    target=support+[0.,shift]
    valid=np.ones(support.shape[:2],bool)
    errors=np.zeros(support.shape[:2],np.float32)
    np.savez_compressed(path,source_support_uv=support,target_support_uv=target,
        valid_FB=valid,FB_error_native_px=errors)
    rgb,m=current;binding=m['source_binding']
    previous=dict(source_sha256=graph.source_sha256,actual_pts=before['pts'],
        native_pts=before['binding']['native_pts'],shot_id=before['binding']['shot_id'],time_base=graph.time_base,
        decoded_native_RGB_sha256=before['RGB_sha256'])
    now=dict(source_sha256=graph.source_sha256,actual_pts=binding['source_pts'],native_pts=binding['source_pts'],
        shot_id=binding['shot_id'],time_base=graph.time_base,decoded_native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
    return dict(source_support_uv=support,target_support_uv=target,valid_FB=valid,FB_error_native_px=errors,
        previous_binding=previous,current_binding=now,method='rendered_source_translation_forward_backward',
        raw_reference=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


@pytest.mark.parametrize('anchored',[False,True])
def test_external_correspondence_links_actual_readouts_without_LK_or_new_semantic_authority(tmp_path,monkeypatch,anchored):
    import rinkspace.vision_ice_identity_graph as module
    graph=IceSourceMaterialIdentityGraph(inventory(),source_sha256='a'*64,time_base='1/1000')
    first,last=frame(10),frame(20,45)
    mask=np.zeros(first[0].shape[:2],bool)
    graph.consume(*first,mask,source_ice_scope_mask=~mask)
    flow=source_translation_correspondence(graph,last,tmp_path/'translation.npz',shift=45)
    def no_LK(*args,**kwargs):raise AssertionError('External source flow must not invoke LK')
    monkeypatch.setattr(module,'_point_source_flow',no_LK)
    rows=[anchor(last[1])] if anchored else []
    result=graph.consume(*last,mask,source_ice_scope_mask=~mask,selected_rows=rows,source_flow_correspondence=flow)
    assert len(graph.frames[1]['parents'])==6
    assert np.array_equal(graph.frames[1]['uv'],last[1]['candidate_sides'].mean(1))
    assert all(link[2]['source_RGB_matching_method']==flow['method'] for link in graph.frames[1]['parents'].values())
    assert all(link[2]['source_flow_correspondence_reference']==flow['raw_reference'] for link in graph.frames[1]['parents'].values())
    assert all('source_forward_backward_LK_error_px' not in link[2] for link in graph.frames[1]['parents'].values())
    if anchored:
        assert {row['actual_pts'] for row in result['rows']}=={10}
        temporal=result['rows'][0]['temporal_evidence']
        assert temporal['source_RGB_tracking_method']==flow['method']
        assert temporal['source_RGB_tracking_methods']==[flow['method']]
    else:
        assert not result['rows'] and all(not f['label'] for f in graph.frames)


@pytest.mark.parametrize('change',[
    'previous_pts','current_pts','native_pts','previous_RGB','current_RGB','source','shot','clock','support',
    'support_shape','target_nonfinite','valid_type','error_negative','error_nonfinite','error_shape',
    'method','raw_hash','unknown_field'])
def test_external_correspondence_rejects_wrong_binding_or_malformed_support_before_mutating_graph(tmp_path,change):
    graph=IceSourceMaterialIdentityGraph(inventory(),source_sha256='a'*64,time_base='1/1000')
    first,last=frame(10),frame(20,2);mask=np.zeros(first[0].shape[:2],bool)
    graph.consume(*first,mask,source_ice_scope_mask=~mask)
    flow=source_translation_correspondence(graph,last,tmp_path/'translation.npz',shift=2)
    if change=='previous_pts':flow['previous_binding']['actual_pts']+=1
    elif change=='current_pts':flow['current_binding']['actual_pts']+=1
    elif change=='native_pts':flow['current_binding']['native_pts']+=1
    elif change=='previous_RGB':flow['previous_binding']['decoded_native_RGB_sha256']='c'*64
    elif change=='current_RGB':flow['current_binding']['decoded_native_RGB_sha256']='c'*64
    elif change=='source':flow['previous_binding']['source_sha256']='c'*64
    elif change=='shot':flow['current_binding']['shot_id']='other'
    elif change=='clock':flow['previous_binding']['time_base']='1/50'
    elif change=='support':flow['source_support_uv'][0,0,1]+=1e-9
    elif change=='support_shape':flow['source_support_uv']=flow['source_support_uv'][:,:2]
    elif change=='target_nonfinite':flow['target_support_uv'][0,0,0]=np.nan
    elif change=='valid_type':flow['valid_FB']=flow['valid_FB'].astype(np.uint8)
    elif change=='error_negative':flow['FB_error_native_px'][0,0]=-.01
    elif change=='error_nonfinite':flow['FB_error_native_px'][0,0]=np.inf
    elif change=='error_shape':flow['FB_error_native_px']=flow['FB_error_native_px'][:,0]
    elif change=='method':flow['method']=''
    elif change=='raw_hash':flow['raw_reference']['sha256']='c'*64
    else:flow['unknown']=True
    previous_gray=graph.previous_gray
    with pytest.raises(ValueError,match='External source flow'):
        graph.consume(*last,mask,source_ice_scope_mask=~mask,source_flow_correspondence=flow)
    assert len(graph.frames)==len(graph.catalog)==1 and graph.previous_gray is previous_gray
    assert not graph.frames[0]['parents'] and not graph.frames[0]['children']


@pytest.mark.parametrize('gate',['FB','valid','previous_mask','current_mask','scope','target_bounds'])
def test_external_correspondence_still_requires_original_FB_and_both_frame_pixel_gates(tmp_path,gate):
    graph=IceSourceMaterialIdentityGraph(inventory(),source_sha256='a'*64,time_base='1/1000')
    first,last=frame(10),frame(20,2);mask=np.zeros(first[0].shape[:2],bool)
    before_mask=mask.copy()
    if gate=='previous_mask':
        pix=np.rint(first[1]['candidate_sides']).astype(int).reshape(-1,2)
        before_mask[pix[:,1],pix[:,0]]=True
    graph.consume(*first,before_mask,source_ice_scope_mask=~mask)
    flow=source_translation_correspondence(graph,last,tmp_path/'translation.npz',shift=2)
    current_mask=mask.copy();scope=~mask
    if gate=='FB':flow['FB_error_native_px'][:]=2.01
    elif gate=='valid':flow['valid_FB'][:]=False
    elif gate in ('previous_mask','current_mask','scope'):
        flow['valid_FB'][:,0]=False  # Only the actually occluded side supports remain.
        if gate!='previous_mask':
            pix=np.rint(flow['target_support_uv'][:,1:]).astype(int).reshape(-1,2)
            if gate=='current_mask':current_mask[pix[:,1],pix[:,0]]=True
            else:scope[pix[:,1],pix[:,0]]=False
    else:flow['target_support_uv'][...,0]=-100.
    result=graph.consume(*last,current_mask,source_ice_scope_mask=scope,selected_rows=[anchor(last[1])],
        source_flow_correspondence=flow)
    assert not graph.frames[1]['parents'] and not result['rows']


def test_external_correspondence_FB_boundary_and_one_accepted_support_keep_normal_median_rule(tmp_path):
    from rinkspace.vision_ice_identity_graph import _source_support_flow_predictions
    graph=IceSourceMaterialIdentityGraph(inventory(),source_sha256='a'*64,time_base='1/1000')
    first,last=frame(10),frame(20,2);mask=np.zeros(first[0].shape[:2],bool)
    graph.consume(*first,mask,source_ice_scope_mask=~mask)
    flow=source_translation_correspondence(graph,last,tmp_path/'translation.npz',shift=2)
    flow['valid_FB'][:,:2]=False;flow['FB_error_native_px'][:]=2.
    graph.consume(*last,mask,source_ice_scope_mask=~mask,source_flow_correspondence=flow)
    assert len(graph.frames[1]['parents'])==6
    support=flow['source_support_uv'];delta=np.array([[12.,1.],[-8.,3.],[10.,5.]])
    predicted,good,error=_source_support_flow_predictions(graph.frames[0],support,support+delta,
        np.ones(support.shape[:2],bool),np.zeros(support.shape[:2]),mask,mask)
    np.testing.assert_array_equal(predicted,graph.frames[0]['uv']+[0.,3.])
    assert good.all() and not error.any()


@pytest.mark.parametrize('empty_frame',[False,True])
def test_external_correspondence_cannot_invent_previous_material_nodes(tmp_path,empty_frame):
    graph=IceSourceMaterialIdentityGraph(inventory(),source_sha256='a'*64,time_base='1/1000')
    first,last=frame(10),frame(20,2);mask=np.zeros(first[0].shape[:2],bool)
    if empty_frame:
        for key in ('candidate_uv','candidate_sides','candidate_query','candidate_flags','query_curve','query_normal'):
            first[1][key]=first[1][key][:0]
        graph.consume(*first,mask,source_ice_scope_mask=~mask)
    with pytest.raises(ValueError,match='previous actual material nodes'):
        graph.consume(*last,mask,source_ice_scope_mask=~mask,source_flow_correspondence={})


def test_external_correspondence_evidence_survives_compact_roundtrip_and_export(tmp_path):
    graph=IceSourceMaterialIdentityGraph(inventory(),source_sha256='a'*64,time_base='1/1000')
    first,last=frame(10),frame(20,2);mask=np.zeros(first[0].shape[:2],bool)
    authority=tmp_path/'authority.json';authority.write_text('{}')
    row=offline_anchor(*first,authority)
    row['source_seed_reference'].update(native_pts=10,shot_id='one',source_material_class='static_painted_ice',
        decoded_native_RGB_sha256=first[1]['decoded_native_RGB_sha256'])
    graph.consume(*first,mask,source_ice_scope_mask=~mask,selected_rows=[row])
    flow=source_translation_correspondence(graph,last,tmp_path/'translation.npz',shift=2)
    graph.consume(*last,mask,source_ice_scope_mask=~mask,source_flow_correspondence=flow)
    saved=tmp_path/'graph.npz';reference=graph.save_compact_state(saved)
    restored=IceSourceMaterialIdentityGraph.load_compact_state(saved)
    assert restored.frames[1]['parents']==graph.frames[1]['parents']
    assert restored.frames[1]['source_flow_correspondence_receipt']==graph.frames[1]['source_flow_correspondence_receipt']
    exported=restored.export_confirmed_source_rows(source_graph_reference=reference)
    recovered=next(r for r in exported['rows'] if r['actual_pts']==20)['temporal_evidence']
    assert recovered['source_RGB_tracking_method']=='preserved_'+flow['method']
    assert recovered['source_RGB_tracking_methods']==[flow['method']]
    assert all(link['source_flow_correspondence_reference']==flow['raw_reference']
               for link in recovered['actual_material_adjacency_evidence'])


def test_mixed_external_and_LK_evidence_names_all_actual_matching_methods():
    from rinkspace.vision_ice_identity_graph import _tracking_method_evidence
    evidence=_tracking_method_evidence([{},dict(source_RGB_matching_method='actual_GPUFlow_FB')])
    assert evidence['source_RGB_tracking_method']=='mixed_actual_material_support_correspondence_and_RGB_context'
    assert evidence['source_RGB_tracking_methods']==['actual_GPUFlow_FB','bidirectional_actual_material_support_LK_and_RGB_context']


def saved_translation_graph(tmp_path, *, shifts=(0,2,4), empty_middle=False, masks=None):
    graph=IceSourceMaterialIdentityGraph(inventory(),source_sha256='a'*64,time_base='1/1000')
    exposures=[frame(10*(i+1),shift) for i,shift in enumerate(shifts)]
    cleared=np.zeros(exposures[0][0].shape[:2],bool)
    if masks is None:masks=[cleared for _ in exposures]
    authority=tmp_path/'source-anchor.json';authority.write_text('{}')
    source=offline_anchor(*exposures[0],authority)
    source['source_seed_reference'].update(native_pts=10,shot_id='one',source_material_class='static_painted_ice',
        decoded_native_RGB_sha256=exposures[0][1]['decoded_native_RGB_sha256'])
    for i,(rgb,m) in enumerate(exposures):
        if empty_middle and i==1:
            for key in ('candidate_uv','candidate_sides','candidate_query','candidate_flags','query_curve','query_normal'):
                m[key]=m[key][:0]
        graph.consume(rgb,m,masks[i],source_ice_scope_mask=~cleared,selected_rows=[source] if not i else [])
    original=tmp_path/'original-LK.npz';graph.save_compact_state(original)
    restored=IceSourceMaterialIdentityGraph.load_compact_state(original)
    correspondences=[];frame_masks=[]
    for i,f in enumerate(restored.frames):
        binding=dict(source_sha256='a'*64,actual_pts=f['pts'],native_pts=f['binding']['native_pts'],
            shot_id=f['binding']['shot_id'],time_base='1/1000',decoded_native_RGB_sha256=f['RGB_sha256'])
        mask_path=tmp_path/f'mask-{i}.npz'
        np.savez_compressed(mask_path,native_hw=np.array(masks[i].shape,np.int32),
            mask_packbits=np.packbits(masks[i].reshape(-1),bitorder='little'))
        frame_masks.append(dict(source_binding=binding,mask=masks[i],
            raw_reference=dict(path=str(mask_path),sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest())))
        if i:
            previous=SimpleNamespace(frames=[restored.frames[i-1]],source_sha256='a'*64,time_base='1/1000')
            correspondences.append(source_translation_correspondence(previous,exposures[i],
                tmp_path/f'pair-{i}.npz',shift=shifts[i]-shifts[i-1]))
    return restored,correspondences,frame_masks


def test_offline_source_flow_relinks_saved_nodes_without_RGB_profiles_or_array_changes(tmp_path,monkeypatch):
    import rinkspace.vision_ice_identity_graph as module
    original,pairs,masks=saved_translation_graph(tmp_path,shifts=(0,45))
    old_digest=hashlib.sha256((tmp_path/'original-LK.npz').read_bytes()).hexdigest()
    def no_RGB(*args,**kwargs):raise AssertionError('Offline relinking must not decode RGB, run LK, or reread profiles')
    monkeypatch.setattr(module,'_point_source_flow',no_RGB)
    monkeypatch.setattr(module,'_measured_material_descriptors',no_RGB)
    monkeypatch.setattr(IceSourceMaterialIdentityGraph,'_nodes',no_RGB)
    monkeypatch.setattr(cv2,'cvtColor',no_RGB)
    accessed=[]
    class LazyMasks:
        def __len__(self):return len(masks)
        def __getitem__(self,index):accessed.append(index);return masks[index]
    rebuilt=original.relink_source_flow_correspondences(pairs,frame_masks=LazyMasks())
    assert accessed==[0,1] and len(rebuilt.frames[1]['parents'])==6
    assert rebuilt is not original and rebuilt.previous_gray is rebuilt.previous_mask is None
    for before,after in zip(original.frames,rebuilt.frames):
        for key,value in before.items():
            if isinstance(value,np.ndarray):
                assert after[key] is value and not value.flags.writeable
        assert before['binding']==after['binding']
    receipt=rebuilt.source_flow_relink_receipt
    assert receipt['source_frames']==2 and receipt['adjacent_source_pairs']==1
    assert receipt['actual_source_adjacency_edges']==6 and receipt['explicit_original_anchor_nodes']==6
    assert not receipt['source_RGB_decoded_again'] and not receipt['source_quantiles_remeasured']
    assert rebuilt.inventory==original.inventory
    assert not original.frames[1]['label'] and not original.frames[1]['parents']
    assert hashlib.sha256((tmp_path/'original-LK.npz').read_bytes()).hexdigest()==old_digest
    saved=tmp_path/'external-flow-graph.npz';reference=rebuilt.save_compact_state(saved)
    rows=rebuilt.export_confirmed_source_rows(source_graph_reference=reference)['rows']
    assert {r['actual_pts'] for r in rows}=={10,20}
    assert all(r['uv']==original.frames[i]['uv'].tolist() for i,r in enumerate(rows))
    reloaded=IceSourceMaterialIdentityGraph.load_compact_state(saved)
    assert reloaded.source_flow_relink_receipt==receipt
    assert reloaded.frames[1]['parents']==rebuilt.frames[1]['parents']
    assert reloaded.frames[0]['source_flow_mask_receipt']==rebuilt.frames[0]['source_flow_mask_receipt']


def test_offline_relink_does_not_reuse_old_forward_labels_as_independent_anchors(tmp_path):
    original,pairs,masks=saved_translation_graph(tmp_path)
    assert all(proof['direction']=='forward' for proof in original.frames[1]['label'].values())
    old_labels=[copy.deepcopy(f['label']) for f in original.frames]
    old_edges=sum(len(f['parents']) for f in original.frames)
    for pair in pairs:pair['valid_FB'][:]=False
    rebuilt=original.relink_source_flow_correspondences(pairs,frame_masks=masks)
    assert len(rebuilt.frames[0]['label'])==6
    assert all(not f['label'] for f in rebuilt.frames[1:])
    assert old_labels==[f['label'] for f in original.frames]
    assert rebuilt.source_flow_relink_receipt['original_actual_source_adjacency_edges']==old_edges
    assert rebuilt.source_flow_relink_receipt['actual_source_adjacency_edges']==0
    assert not rebuilt.source_flow_relink_receipt['old_propagated_labels_reused_as_anchors']


def test_offline_zero_node_frame_keeps_zero_edges_and_cannot_be_skipped(tmp_path):
    original,pairs,masks=saved_translation_graph(tmp_path,empty_middle=True)
    rebuilt=original.relink_source_flow_correspondences(pairs,frame_masks=masks)
    assert [len(f['ids']) for f in rebuilt.frames]==[6,0,6]
    assert all(not f['parents'] for f in rebuilt.frames)
    assert not rebuilt.frames[2]['label']
    assert rebuilt.source_flow_relink_receipt['zero_node_source_pairs']==2
    assert rebuilt.source_flow_relink_receipt['adjacent_source_pairs']==2
    with pytest.raises(ValueError,match='every saved adjacent'):
        original.relink_source_flow_correspondences(pairs[:1],frame_masks=masks)


@pytest.mark.parametrize('change',['pair_time','pair_source','pair_UV','pair_order','mask_time','mask_bytes','mask_hash'])
def test_offline_relink_rejects_wrong_original_pair_or_preserved_mask(tmp_path,change):
    original,pairs,masks=saved_translation_graph(tmp_path)
    if change=='pair_time':pairs[0]['current_binding']['actual_pts']+=1
    elif change=='pair_source':pairs[0]['previous_binding']['source_sha256']='c'*64
    elif change=='pair_UV':pairs[0]['source_support_uv'][0,1,0]+=1e-9
    elif change=='pair_order':pairs.reverse()
    elif change=='mask_time':masks[0]['source_binding']['actual_pts']+=1
    elif change=='mask_bytes':
        masks[0]['mask']=masks[0]['mask'].copy();masks[0]['mask'][0,0]=True
    else:masks[0]['raw_reference']['sha256']='c'*64
    old_labels=[copy.deepcopy(f['label']) for f in original.frames]
    with pytest.raises(ValueError):original.relink_source_flow_correspondences(pairs,frame_masks=masks)
    assert old_labels==[f['label'] for f in original.frames]


@pytest.mark.parametrize('masked_frame',[0,1])
def test_offline_relink_checks_original_before_and_current_combined_masks(tmp_path,masked_frame):
    masks=[np.zeros((140,240),bool) for _ in range(2)]
    source=frame(10*(masked_frame+1),2*masked_frame)[1]['candidate_sides']
    pix=np.rint(source).astype(int).reshape(-1,2)
    masks[masked_frame][pix[:,1],pix[:,0]]=True
    original,pairs,records=saved_translation_graph(tmp_path,shifts=(0,2),masks=masks)
    pairs[0]['valid_FB'][:,0]=False
    rebuilt=original.relink_source_flow_correspondences(pairs,frame_masks=records)
    assert not rebuilt.frames[1]['parents'] and not rebuilt.frames[1]['label']


def test_legacy_compact_graph_without_method_groups_keeps_LK_evidence(tmp_path):
    original,_,_=saved_translation_graph(tmp_path,shifts=(0,2))
    with np.load(tmp_path/'original-LK.npz',allow_pickle=False) as saved:
        fields={key:saved[key] for key in saved.files if key!='actual_source_edge_evidence_group_indices'}
    metadata=json.loads(str(fields['metadata_json'].item()))
    metadata.pop('actual_source_edge_evidence_groups')
    fields['metadata_json']=json.dumps(metadata)
    path=tmp_path/'legacy-LK.npz';np.savez_compressed(path,**fields)
    restored=IceSourceMaterialIdentityGraph.load_compact_state(path)
    assert len(restored.frames[1]['parents'])==len(original.frames[1]['parents'])==6
    for node,(_,_,evidence) in restored.frames[1]['parents'].items():
        assert evidence['source_forward_backward_LK_error_px']==original.frames[1]['parents'][node][2]['source_forward_backward_LK_error_px']
        assert 'source_flow_correspondence_reference' not in evidence



def incremental_added_node_graph(tmp_path, *, with_guard=False):
    masks=None
    if with_guard:
        masks=[np.zeros((140,240),bool)for _ in range(4)]
        for value in masks:value[0,0]=True
    old,pairs,records=saved_translation_graph(tmp_path,shifts=(0,2,4,6),masks=masks)
    guard_ref=None
    if with_guard:
        guard=np.zeros((140,240),np.uint8);guard[0,0]=255
        guard_path=tmp_path/'original-guard.png';cv2.imwrite(str(guard_path),guard)
        guard_ref=dict(path=str(guard_path),sha256=hashlib.sha256(guard_path.read_bytes()).hexdigest())
        for record in records:
            path=Path(record['raw_reference']['path'])
            np.savez_compressed(path,native_hw=np.array(record['mask'].shape,np.int32),
                mask_packbits=np.packbits(record['mask'].reshape(-1),bitorder='little'),
                metadata_json=json.dumps(dict(original_guard=guard_ref)))
            record['raw_reference']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    baseline=old.relink_source_flow_correspondences(pairs,frame_masks=records)
    baseline_path=tmp_path/'baseline-flow.npz';baseline.save_compact_state(baseline_path)
    baseline=IceSourceMaterialIdentityGraph.load_compact_state(baseline_path)
    extended=IceSourceMaterialIdentityGraph.load_compact_state(baseline_path)
    value=extended.frames[1]
    for key in ('ids','query_ids','uv','sides','normal','width','curve','descriptor','raw_uv','quantiles'):
        if key not in value:continue
        added=value[key][:1].copy()
        if key in ('ids','query_ids'):added[:]=99
        elif key in ('uv','sides','raw_uv','quantiles'):added+=np.array([6.,0.])
        value[key]=np.concatenate([value[key],added])
    path=tmp_path/'extended-nodes.npz';extended.save_compact_state(path)
    extended=IceSourceMaterialIdentityGraph.load_compact_state(path)
    previous=SimpleNamespace(frames=[extended.frames[1]],source_sha256='a'*64,time_base='1/1000')
    changed={0:pairs[0],1:source_translation_correspondence(previous,frame(30,4),
        tmp_path/'changed-pair-1.npz',shift=2.)}
    return extended,baseline,changed,records,guard_ref


def test_incremental_relink_only_changed_pairs_reuses_old_edges_without_propagated_seeds(tmp_path,monkeypatch):
    import rinkspace.vision_ice_identity_graph as module
    extended,baseline,pairs,masks,_=incremental_added_node_graph(tmp_path)
    pairs[1]['valid_FB'][:]=False
    def no_RGB(*args,**kwargs):raise AssertionError('Cached incremental relinking must not decode RGB or run LK')
    monkeypatch.setattr(module,'_point_source_flow',no_RGB)
    monkeypatch.setattr(module,'_measured_material_descriptors',no_RGB)
    rebuilt=extended.relink_changed_source_flow_correspondences(baseline,pairs,frame_masks=masks)
    receipt=rebuilt.source_flow_relink_receipt
    assert receipt['added_material_nodes']==1 and receipt['changed_node_frame_ordinals']==[1]
    assert receipt['changed_pair_ordinals']==[0,1]
    assert receipt['actual_relinked_source_pairs']==2 and receipt['reused_source_pairs']==1
    assert len(rebuilt.frames[1]['parents'])==7 and not rebuilt.frames[2]['parents']
    assert len(rebuilt.frames[3]['parents'])==6
    assert not rebuilt.frames[2]['label'] and not rebuilt.frames[3]['label']
    assert receipt['explicit_original_anchor_nodes']==6
    assert not receipt['old_propagated_labels_reused_as_anchors']
    assert rebuilt.frames[3]['parents']==baseline.frames[3]['parents']
    assert len(baseline.frames[1]['ids'])==6 and len(extended.frames[1]['ids'])==7


@pytest.mark.parametrize('change',['missing_pair','extra_pair','old_readout'])
def test_incremental_relink_rejects_incomplete_changed_scope_or_changed_old_pixels(tmp_path,change):
    extended,baseline,pairs,masks,_=incremental_added_node_graph(tmp_path)
    if change=='missing_pair':pairs.pop(0)
    elif change=='extra_pair':pairs[2]=pairs[1]
    else:
        extended.frames[0]['uv']=extended.frames[0]['uv'].copy()
        extended.frames[0]['uv'][0,0]+=1e-9
    with pytest.raises(ValueError):
        extended.relink_changed_source_flow_correspondences(baseline,pairs,frame_masks=masks)


@pytest.mark.parametrize('change',['shrink_scope','release_guard'])
def test_incremental_relink_never_excludes_old_clear_pixels_or_releases_original_guard(tmp_path,change):
    extended,baseline,pairs,masks,guard_ref=incremental_added_node_graph(tmp_path,with_guard=True)
    value=masks[1]['mask'].copy()
    value[1,1]=True if change=='shrink_scope' else value[1,1]
    if change=='release_guard':value[0,0]=False
    path=tmp_path/'changed-mask.npz'
    np.savez_compressed(path,native_hw=np.array(value.shape,np.int32),
        mask_packbits=np.packbits(value.reshape(-1),bitorder='little'),
        metadata_json=json.dumps(dict(original_guard=guard_ref)))
    masks[1]=dict(source_binding=masks[1]['source_binding'],mask=value,
        raw_reference=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    with pytest.raises(ValueError):
        extended.relink_changed_source_flow_correspondences(baseline,pairs,frame_masks=masks)
