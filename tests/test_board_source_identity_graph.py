"""Board identity uses actual source links and its own exact surface authority."""
import copy
import hashlib
import json

import cv2
import numpy as np
import pytest

from rinkspace.vision_ice_identity_graph import IceSourceMaterialIdentityGraph
from test_vision_ice_identity_graph import frame, offline_anchor


def board_inventory():
    return [dict(curve_id='board:top',feature_id='board',world_curve_role='top',operator='material_interface')]


def board_frame(pts,shift=0):
    rgb,m=frame(pts,shift)
    # One measured interface, repeated for the graph's two-support layout.
    m['candidate_uv']=m['candidate_sides'][:,1].copy()
    m['candidate_sides']=np.repeat(m['candidate_uv'][:,None],2,axis=1)
    return rgb,m


def source_anchor(rgb,m,authority):
    row=offline_anchor(rgb,m,authority)
    row.update(curve_id='board:top',physical_material_id='board:top',scope='board',
        admission_status='admitted_source_physical_boundary_temporal_identity')
    row['temporal_evidence']['physical_material_id']='board:top'
    row['source_seed_reference'].update(curve_id='board:top',native_pts=row['native_pts'],shot_id='one',
        source_material_class='static_board_material',decoded_native_RGB_sha256=m['decoded_native_RGB_sha256'])
    return row


def exact_source_review(row,path,image,scope='board'):
    from calibration.program_tracking.strict_model_scope import ExactSourceScopeReview,REVIEW_SCHEMA
    evidence=dict(path=str(image),sha256=hashlib.sha256(image.read_bytes()).hexdigest(),kind='reviewed_original_image')
    endpoints=[dict(actual_pts=row['actual_pts'],native_pts=row['native_pts'],uv=uv,scope=scope,decision='allow',
        physical_material_id='board:top',static_material_confirmed=True,pixel_location_confirmed=True,
        review_authority='agent_original_source_review',original_source_image_actually_reviewed=True,
        reviewer='synthetic fixture source review',model_projection_or_fit_residual_used_for_identity=False,
        review_note='Exact static boundary in this fixture source image',
        decoded_BGR_sha256=row['decoded_native_BGR_sha256'],source_image_evidence=evidence) for uv in row['uv']]
    path.write_text(json.dumps(dict(schema=REVIEW_SCHEMA,
        source=dict(video_sha256=row['source_sha256'],time_base=row['time_base'],pixel_domain='native1920x1080'),
        reviewed_endpoints=endpoints)))
    return ExactSourceScopeReview.load(path)


def saved_board_graph(tmp_path,surface='board'):
    graph=IceSourceMaterialIdentityGraph(board_inventory(),source_sha256='a'*64,time_base='1/1000',source_surface=surface)
    first=board_frame(10);last=board_frame(20,2);mask=np.zeros(first[0].shape[:2],bool)
    graph.consume(*first,mask,source_scope_mask=~mask)
    graph.consume(*last,mask,source_scope_mask=~mask)
    path=tmp_path/'unseeded.npz';graph.save_compact_state(path)
    authority=tmp_path/'authority.json';authority.write_text('{}')
    row=source_anchor(*last,authority)
    image=tmp_path/'source.png';assert cv2.imwrite(str(image),last[0][:,:,::-1])
    return path,row,image


def test_board_anchor_recovers_only_measured_interfaces_and_persists_surface(tmp_path):
    path,row,image=saved_board_graph(tmp_path)
    graph=IceSourceMaterialIdentityGraph.load_compact_state(path)
    assert graph.source_surface=='board'
    before=[f['uv'].copy() for f in graph.frames]
    review=exact_source_review(row,tmp_path/'review.json',image)
    graph.apply_confirmed_source_rows([row],source_scope_review=review)
    saved=tmp_path/'confirmed.npz';receipt=graph.save_compact_state(saved)
    restored=IceSourceMaterialIdentityGraph.load_compact_state(saved)
    result=restored.export_confirmed_source_rows(source_graph_reference=receipt)
    assert {r['actual_pts'] for r in result['rows']}=={10,20}
    for r in result['rows']:
        assert r['scope']=='board'
        assert r['source_seed_reference']['source_material_class']=='static_board_material'
        assert np.array_equal(r['uv'],np.asarray(r['source_sides_uv'])[:,0])
    assert all(np.array_equal(a,f['uv']) for a,f in zip(before,restored.frames))
    with pytest.raises(ValueError,match='identity proofs'):
        IceSourceMaterialIdentityGraph.load_compact_state(saved,source_surface='ice')


@pytest.mark.parametrize('connected',[True,False])
def test_saved_board_export_does_not_promote_same_seed_exposure_propagation_to_anchor(tmp_path,connected):
    path,row,image=saved_board_graph(tmp_path)
    graph=IceSourceMaterialIdentityGraph.load_compact_state(path)
    previous,current=graph.frames
    # A saved previous node can link more than one current candidate. Review
    # only candidates 0/1; candidates 2/3 must inherit identity via frame 10.
    for target,anchor_node in ((2,0),(3,1)):
        old_fi,old_node,_=current['parents'][target]
        previous['children'][old_node]=[edge for edge in previous['children'][old_node] if edge[:2]!=(1,target)]
        pfi,pnode,evidence=current['parents'][anchor_node]
        assert old_fi==pfi==0
        current['parents'][target]=(pfi,pnode,evidence)
        previous['children'][pnode].append((1,target,evidence))
    for key in ('original_candidate_indices','source_query_indices','uv','source_sides_uv'):
        row[key]=row[key][:2]
    review=exact_source_review(row,tmp_path/'review.json',image)
    graph.apply_confirmed_source_rows([row],source_scope_review=review)
    assert [current['label'][node]['direction'] for node in range(4)]==['anchor','anchor','forward','forward']
    if not connected:
        # Keep the saved edges, but make their intermediate source readouts
        # ineligible. Propagated labels must then remain pending.
        previous['source_pixel_accuracy_pending_nodes']={0,1}
    saved=tmp_path/'confirmed.npz';receipt=graph.save_compact_state(saved)
    restored=IceSourceMaterialIdentityGraph.load_compact_state(saved)
    before=[copy.deepcopy(f) for f in restored.frames]
    result=restored.export_confirmed_source_rows(source_graph_reference=receipt)
    assert result['recoverable_source_seed_nodes']==2
    assert result['source_proof_reachable_nodes']==(6 if connected else 2)
    anchors=[r for r in result['rows'] if r['temporal_evidence']['kind']=='confirmed_original_source_identity_anchor']
    assert len(anchors)==1 and anchors[0]['actual_pts']==20
    assert anchors[0]['original_candidate_indices']==[0,1]
    propagated=[r for r in result['rows'] if r['actual_pts']==20 and r not in anchors]
    if connected:
        assert len(propagated)==1 and propagated[0]['original_candidate_indices']==[2,3]
        evidence=propagated[0]['temporal_evidence']
        assert evidence['kind']=='same_shot_actual_RGB_temporal_identity'
        assert evidence['direction']=='forward' and evidence['previous_actual_pts']==10
        assert evidence['previous_decoded_BGR_sha256']==previous['binding']['decoded_BGR_sha256']
        assert len(evidence['actual_material_adjacency_evidence'])==2 and not result['pending']
    else:
        assert not propagated
        pending=next(p for p in result['pending'] if p['actual_pts']==20)
        assert pending['confirmed_labels_without_recoverable_source_proof']==2
    for recovered in result['rows']:
        original=before[0 if recovered['actual_pts']==10 else 1]
        indices=recovered['original_candidate_indices']
        assert np.array_equal(recovered['uv'],original['uv'][indices])
        assert np.array_equal(recovered['source_sides_uv'],original['sides'][indices])
        assert recovered['source_seed_authority']==row['source_seed_authority']
        assert recovered['source_seed_reference']==row['source_seed_reference']
    for original,unchanged in zip(before,restored.frames):
        assert original['label']==unchanged['label'] and original['emitted']==unchanged['emitted']


@pytest.mark.parametrize('wrong',['ice_scope','paint_seed','modified_uv'])
def test_board_anchor_rejects_wrong_surface_or_changed_source_before_labelling(tmp_path,wrong):
    path,row,image=saved_board_graph(tmp_path)
    graph=IceSourceMaterialIdentityGraph.load_compact_state(path)
    review=exact_source_review(row,tmp_path/'review.json',image,scope='ice' if wrong=='ice_scope' else 'board')
    if wrong=='paint_seed':row['source_seed_reference']['source_material_class']='static_painted_ice'
    if wrong=='modified_uv':row['uv'][0][1]+=.001
    with pytest.raises(ValueError):graph.apply_confirmed_source_rows([row],source_scope_review=review)
    assert all(not f['label'] for f in graph.frames)


def test_only_an_unseeded_graph_can_change_surface_without_new_source_reads(tmp_path):
    path,row,image=saved_board_graph(tmp_path,surface='ice')
    original=IceSourceMaterialIdentityGraph.load_compact_state(path)
    restored=IceSourceMaterialIdentityGraph.load_compact_state(path,source_surface='board')
    assert restored.source_surface=='board' and not any(f['label'] for f in restored.frames)
    for old,new in zip(original.frames,restored.frames):
        for key in ('uv','sides','normal','descriptor','ids','query_ids'):
            assert np.array_equal(old[key],new[key])
        assert old['parents']==new['parents'] and old['children']==new['children']


def test_board_contract_does_not_accept_paint_centers_or_ambiguous_scope_arguments():
    with pytest.raises(ValueError,match='interfaces'):
        IceSourceMaterialIdentityGraph([dict(operator='paint_center')],source_sha256='a'*64,time_base='1/1000',source_surface='board')
    with pytest.raises(ValueError,match='ice or board'):
        IceSourceMaterialIdentityGraph([],source_sha256='a'*64,time_base='1/1000',source_surface='stands')
    graph=IceSourceMaterialIdentityGraph(board_inventory(),source_sha256='a'*64,time_base='1/1000',source_surface='board')
    rgb,m=board_frame(10);mask=np.zeros(rgb.shape[:2],bool)
    with pytest.raises(ValueError,match='exactly one'):
        graph.consume(rgb,m,mask,source_scope_mask=~mask,source_ice_scope_mask=~mask)
