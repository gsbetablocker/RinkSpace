import hashlib
import copy
import json

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
