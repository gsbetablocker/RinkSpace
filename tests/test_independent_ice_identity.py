"""Synthetic current-source identity integration; no real capture or native fit."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

PROGRAMME=Path(__file__).resolve().parents[1]/'events/ankara_2026/programmes/okada_mei_sp'
sys.path.insert(0,str(PROGRAMME))
import capture_independent_ice as ice

SOURCE='a'*64


def source_frame(y=100.,curve=0):
    yy=np.arange(240)[:,None]
    rgb=np.broadcast_to((230.-100.*np.exp(-.5*((yy-y)/4.)**2))[:,:,None],(240,400,3)).astype(np.uint8).copy()
    uv=np.c_[np.arange(6)*24.+40.,np.full(6,y)].astype(np.float32)
    sides=uv[:,None]+np.array([[0.,-4.],[0.,4.]])
    m=dict(candidate_query=np.arange(6),query_curve=np.full(6,curve),query_uv=uv.copy(),
        query_normal=np.tile([0.,1.],(6,1)),candidate_uv=uv.copy(),candidate_sides=sides,
        candidate_side_quantiles_uv=sides[:,:,None]+np.array([[0.,-.5],[0.,0.],[0.,.5]]),
        candidate_flags=np.zeros(6,int))
    fragments=[dict(curve=curve,candidate_ids=list(range(6)),source_visible_candidate_pieces=[list(range(6))],
                    organization='unique_whole_source_line_family',physical_identity=None)]
    return rgb,m,fragments


def paths():
    return [dict(native_path_id=name,curve_id=name,operator='paint_center',material_signal='dark',
                 material_transition=None,feature_id='C' if i else 'other') for i,name in enumerate(['query.wrong_name','ice.C'])]


def write_reference(tmp_path,*,approved=True,identity='ice.C'):
    rgb,_,_=source_frame();folder=tmp_path/'references'/'s07';folder.mkdir(parents=True)
    bgr_sha=hashlib.sha256(rgb.tobytes()).hexdigest()
    frame=dict(pts=512,fit_eligible=False,source_video_sha256=SOURCE,source_BGR_sha256=bgr_sha,
        rows=[dict(path_id=identity,uv=[float(x),100.],model_UV_synthesized=False) for x in (52,76,100,124)])
    source=folder/'source_rows.jsonl';source.write_text(json.dumps(frame)+'\n',encoding='utf8')
    group=dict(shot='s07',pts=512,path_id=identity,source_rows_file_sha256=ice.digest(source),
               approved=approved,evidence='synthetic independently reviewed source contour')
    approval=tmp_path/'approval.json';ice.write(approval,dict(schema='rinkspace.independent_line_anchor_approval.v1',
        source_video_sha256=SOURCE,groups=[group]))
    return folder.parent,approval,frame


def navigation(tmp_path,supported=(True,True,True)):
    shot=dict(shot_id='s07',processing=dict(first_pts=512,last_pts=1024,stop_pts_exclusive=1280))
    folder=tmp_path/'s07'/'navigation';folder.mkdir(parents=True)
    h=np.array([[1.,0.,0.],[0.,1.,5.],[0.,0.,1.]])
    np.savez(folder/'source_graph.npz',pts=np.array([512,768,1024],np.int64),
             supported=np.asarray(supported,bool),adjacent_h=np.tile(h,(2,1,1)))
    return ice.load_navigation_graph(tmp_path,shot)


def test_pending_reference_cannot_seed_without_explicit_group_approval(tmp_path):
    capture,approval,frame=write_reference(tmp_path,approved=False)
    groups,note=ice.approval_sources(capture,approval,['s07'],SOURCE)
    assert groups==[] and note['groups']==0
    assert ice.anchor_records(groups,SOURCE)==[]
    assert ice.approval_sources(capture,None,['s07'],SOURCE)[0]==[]
    doc=json.loads(approval.read_text());doc['groups'][0]['approved']=True;ice.write(approval,doc)
    groups,_=ice.approval_sources(capture,approval,['s07'],SOURCE)
    records=ice.anchor_records(groups,SOURCE)
    assert len(records)==1 and records[0]['reference_fit_eligible'] is False
    np.testing.assert_array_equal(records[0]['uv'],[r['uv'] for r in frame['rows']])


@pytest.mark.parametrize('bad',['schema','source','hash','evidence'])
def test_approval_requires_current_schema_video_file_hash_and_evidence(tmp_path,bad):
    capture,approval,_=write_reference(tmp_path)
    doc=json.loads(approval.read_text())
    if bad=='schema':doc['schema']='wrong'
    elif bad=='source':doc['source_video_sha256']='f'*64
    elif bad=='hash':doc['groups'][0]['source_rows_file_sha256']='0'*64
    else:doc['groups'][0]['evidence']=[]
    ice.write(approval,doc)
    with pytest.raises(ValueError):ice.approval_sources(capture,approval,['s07'],SOURCE)


def test_anchor_matches_actual_same_contour_across_search_names_not_curve_id(tmp_path):
    capture,approval,frame=write_reference(tmp_path)
    declared,_=ice.approval_sources(capture,approval,['s07'],SOURCE)
    groups=ice.anchor_records(declared,SOURCE)
    _,m,fragments=source_frame()
    anchors,attempts=ice.match_anchor_fragments(groups,512,m,fragments,frame['source_BGR_sha256'],True)
    assert len(anchors)==1 and anchors[0]['identity']=='ice.C' and fragments[0]['curve']==0
    assert anchors[0]['support']['qualified'] and anchors[0]['support']['span_px']>=56.
    assert attempts[0]['status']=='matched_actual_source_contour'
    # A failed incoming H does not invalidate an independently approved RGB anchor.
    assert ice.match_anchor_fragments(groups,512,m,fragments,frame['source_BGR_sha256'],False)[0]
    assert ice.match_anchor_fragments(groups,512,m,fragments,'0'*64,True)[0]==[]
    absent=[dict(groups[0],uv=groups[0]['uv']+[0.,20.])]
    assert ice.match_anchor_fragments(absent,512,m,fragments,frame['source_BGR_sha256'],True)[0]==[]


def test_supported_cumulative_navigation_is_search_only_and_cannot_cross_missing_graph(tmp_path):
    graph,note=navigation(tmp_path)
    assert note['sha256']==ice.digest(note['path'])
    h,_=ice.navigation_between(graph,512,1024)
    np.testing.assert_allclose(h,[[1,0,0],[0,1,10],[0,0,1]])
    reverse,_=ice.navigation_between(graph,1024,512)
    np.testing.assert_allclose(reverse@h,np.eye(3))
    uv=np.c_[[50.,80.,110.,140.],np.full(4,100.)];original=uv.copy()
    reference=dict(path_id='ice.C',pts=512,uv=uv,approval_group=0,source_rows_file_sha256='b'*64)
    curves={'ice.C':dict(uv=np.array([[0.,1.],[399.,1.]]),valid=np.ones(2,bool))}
    moved,receipt=ice.source_reference_curves(curves,paths(),[reference],1024,graph,400,240)
    np.testing.assert_allclose(moved['ice.C']['uv'][:,1],110.)
    np.testing.assert_array_equal(uv,original)
    assert receipt[0]['search_only'] and not receipt[0]['observation_UV_created']
    graph['supported'][1]=False;graph['segments'][2]=1
    assert ice.navigation_between(graph,512,1024)[0] is None
    unchanged,receipt=ice.source_reference_curves(curves,paths(),[reference],1024,graph,400,240)
    assert receipt==[] and unchanged['ice.C'] is curves['ice.C']
    assert ice.navigation_between(None,512,768)[0] is None


def test_real_bidirectional_rgb_links_propagate_future_anchor_without_changing_uv():
    inventory=paths();frames=[];nodes=[];measurements=[];rgbs=[]
    mask=np.zeros((240,400),bool)
    for i in range(3):
        rgb,m,f=source_frame(100.+i*5.,curve=i%2)
        rgbs.append(rgb);measurements.append(m);nodes.append(ice.source_nodes(rgb,m,f,inventory))
        frames.append(dict(pts=512+i*256,fragments=f,cross_identity_conflicts=[]))
    edges=[]
    h=np.array([[1.,0.,0.],[0.,1.,5.],[0.,0.,1.]])
    for i in range(2):
        linked=ice.link_fragments(rgbs[i],rgbs[i+1],nodes[i],nodes[i+1],mask,mask,h)
        assert len(linked)==1 and linked[0]['unambiguous'] and linked[0]['source_only']
        edges.append(linked)
    anchors=[dict(pts=1024,fragment=0,identity='ice.C')]
    labels=ice.propagate_identities(frames,edges,anchors)
    assert len(labels)==3 and all(r['identities']==['ice.C'] for r in labels)
    original=measurements[0]['candidate_uv'].copy()
    rows,pending,_=ice.identity_rows(measurements[0],frames[0]['fragments'],inventory,[labels[0]],[])
    assert rows and not pending
    assert all(r['path_id']=='ice.C' and r['query_path_id']=='query.wrong_name' for r in rows)
    assert all(r['uv'] in original.tolist() for r in rows)
    np.testing.assert_array_equal(measurements[0]['candidate_uv'],original)
    ambiguous=ice.propagate_identities(frames,edges,anchors+[dict(pts=512,fragment=0,identity='ice.other')])
    assert all(r['status']=='identity_conflict' for r in ambiguous)
    assert ice.identity_rows(measurements[0],frames[0]['fragments'],inventory,[ambiguous[0]],[])[0]==[]


def test_unqualified_adjacent_background_map_only_initializes_actual_rgb_search(tmp_path):
    from link_legacy_ice_sources import search_only_adjacent_navigation
    graph,_=navigation(tmp_path,supported=(True,False,True))
    assert ice.navigation_between(graph,512,768)[0] is None
    h,status=search_only_adjacent_navigation(graph,512,768)
    assert status=='finite_raw_adjacent_H_search_only_background_support_unqualified'
    np.testing.assert_array_equal(h,graph['adjacent_h'][0])
    assert search_only_adjacent_navigation(graph,512,1024)[0] is None
    assert search_only_adjacent_navigation(graph,768,512)[0] is None
    # The helper does not change the qualified cumulative-navigation contract.
    assert ice.navigation_between(graph,512,1024)[0] is None
    assert not graph['supported'][1]
    a,ma,fa=source_frame(100.);b,mb,fb=source_frame(105.)
    old=ice.source_nodes(a,ma,fa,paths());now=ice.source_nodes(b,mb,fb,paths())
    mask=np.zeros(a.shape[:2],bool)
    links=ice.link_fragments(a,b,old,now,mask,mask,h)
    assert len(links)==1 and links[0]['unambiguous']
    assert links[0]['source_only'] and not links[0]['identity_assumed_from_curve_name']
    # A finite H itself cannot supply a material match from unrelated RGB.
    assert ice.link_fragments(a,np.full_like(b,230),old,now,mask,mask,h)==[]


@pytest.mark.parametrize('bad',[np.zeros((3,3)),np.full((3,3),np.nan)])
def test_search_only_adjacent_navigation_rejects_unusable_raw_steps(tmp_path,bad):
    from link_legacy_ice_sources import search_only_adjacent_navigation
    graph,_=navigation(tmp_path,supported=(True,False,True));graph['adjacent_h'][0]=bad
    assert search_only_adjacent_navigation(graph,512,768)[0] is None


def test_search_only_navigation_preserves_supported_map_exactly(tmp_path):
    from link_legacy_ice_sources import search_only_adjacent_navigation
    graph,_=navigation(tmp_path)
    h,status=ice.navigation_between(graph,768,1024)
    actual,actual_status=search_only_adjacent_navigation(graph,768,1024)
    np.testing.assert_array_equal(actual,h)
    assert actual_status==status


def test_same_frame_duplicate_queries_merge_only_original_pixels_at_48px_arc_spacing():
    rgb,m,f=source_frame();inventory=paths()
    for key in ('candidate_query','candidate_flags'):m[key]=np.tile(m[key],2)
    for key in ('candidate_uv','candidate_sides','candidate_side_quantiles_uv'):m[key]=np.concatenate([m[key],m[key]])
    # Independent query strips sample the same actual source contour.
    m['candidate_query'][6:]+=6
    for key in ('query_uv','query_normal'):m[key]=np.concatenate([m[key],m[key]])
    m['query_curve']=np.r_[np.zeros(6,int),np.ones(6,int)]
    f.append(dict(f[0],curve=1,candidate_ids=list(range(6,12))))
    nodes=ice.source_nodes(rgb,m,f,inventory);conflicts=ice.same_frame_conflicts(nodes,f)
    assert conflicts and conflicts[0]['supporting_readouts']>=3
    labels=[dict(fragment=i,status='anchored',identities=['ice.C']) for i in range(2)]
    rows,pending,merges=ice.identity_rows(m,f,inventory,labels,conflicts)
    assert not pending and len(rows)==4
    assert len({tuple(r['uv']) for r in rows})==4
    assert all(r['uv'] in m['candidate_uv'].tolist() for r in rows)
    assert merges[0]['spacing_px']==48. and merges[0]['interpolated_points_created']==0
    labels[1]['identities']=['ice.other']
    rows,pending,_=ice.identity_rows(m,f,inventory,labels,conflicts)
    assert rows==[] and len(pending)==2


def test_same_frame_identity_conflict_keeps_entire_linked_components_pending():
    frames=[dict(cross_identity_conflicts=[]),dict(cross_identity_conflicts=[dict(a=0,b=1)])]
    labels=[dict(frame=i,fragment=j,identities=[name],status='anchored',proofs={str(j):None})
            for i in range(2) for j,name in enumerate(['ice.C','ice.other'])]
    resolved=ice.resolve_same_frame_identity_conflicts(frames,labels)
    assert all(row['status']=='identity_conflict' and row['identities']==['ice.C','ice.other'] for row in resolved)


@pytest.mark.parametrize('prior_fit',[False,True])
def test_same_frame_pending_C_capture_is_navigation_only_never_approval(tmp_path,prior_fit):
    capture=tmp_path/'search';folder=capture/'s07';folder.mkdir(parents=True)
    ice.write(capture/'binding.json',dict(source_video_sha256=SOURCE))
    rgb,m,_=source_frame();bgr=hashlib.sha256(rgb.tobytes()).hexdigest()
    record=dict(pts=512,source_video_sha256=SOURCE,source_BGR_sha256=bgr,fit_eligible=prior_fit,
        rows=[dict(path_id='ice.C',uv=uv.tolist(),model_UV_synthesized=False) for uv in m['candidate_uv']])
    (folder/'source_rows.jsonl').write_text(json.dumps(record)+'\n',encoding='utf8')
    records,note=ice.source_search_records(capture,['s07','s08'],SOURCE,'ice.C')
    assert note['missing_shots']==['s08']
    curves={'ice.C':dict(uv=np.array([[0.,1.],[399.,1.]]),valid=np.ones(2,bool))}
    moved,receipt=ice.same_frame_center_search(curves,'ice.C',records['s07',512],bgr,400,240)
    np.testing.assert_allclose(moved['ice.C']['uv'][:,1],100.)
    assert receipt['search_only'] and not receipt['inherited_fit_eligibility'] and not receipt['inherited_identity']
    assert receipt['observation_UV_created'] is False and receipt['identity_established'] is False
    assert ice.same_frame_center_search(curves,'ice.C',records['s07',512],'0'*64)[0] is curves
    assert ice.approval_sources(capture,None,['s07'],SOURCE)[0]==[]


def test_source_floor_limit_is_ice_only_and_missing_limit_stays_explicit():
    rgb=np.full((140,500,3),230,np.uint8);rgb[50:60,20:480]=[210,190,30]
    original=np.zeros((140,500),bool)
    ice_mask,receipt=ice.ice_source_mask(rgb,original)
    assert receipt['status']=='long_source_kick_plate_supported' and receipt['navigation_only']
    assert ice_mask[:50].all() and not ice_mask[60:].any()
    assert not original.any()  # Shared advertisement exclusion is unchanged.
    absent,note=ice.ice_source_mask(np.full_like(rgb,230),original)
    assert not absent.any() and note['status']=='source_kick_plate_unresolved'


def test_final_rows_keep_raw_measurement_hashes_and_unanchored_frames_pending(tmp_path):
    folder=tmp_path/'capture';folder.mkdir();frames=[];candidate_records=[];raw_hashes={}
    rgb,m,f=source_frame();source_image=folder/'source.png'
    ice.cv2.imwrite(str(source_image),rgb);image_sha=hashlib.sha256(rgb.tobytes()).hexdigest()
    for i in range(2):
        pts=512+i*256;raw=folder/f'{pts}.npz';np.savez(raw,**m);sha=ice.digest(raw);raw_hashes[raw]=sha
        rows=[ice.measured_row(m,c,0,paths()[0]) for c in f[0]['candidate_ids']]
        candidate_records.append(dict(pts=pts,rows=rows,fit_eligible=False,source_BGR_sha256=image_sha,
            source_video_sha256=SOURCE,raw_source_measurements=str(raw)))
        frames.append(dict(pts=pts,fragments=f,cross_identity_conflicts=[],raw_sha256=sha,navigation_supported=i==0,
                           source_image=str(source_image) if i==0 else None))
    candidate=folder/'candidate_source_rows.jsonl';candidate.write_text(''.join(json.dumps(r)+'\n' for r in candidate_records),encoding='utf8')
    candidate_sha=ice.digest(candidate);(folder/'source_links.jsonl').write_text('')
    result=ice.finalize_identity_capture(folder,frames,[[]],[dict(pts=512,fragment=0,identity='ice.C')],paths(),dict(anchor_approval='synthetic'))
    output=[json.loads(line) for line in (folder/'source_rows.jsonl').read_text().splitlines()]
    assert output[0]['fit_eligible'] and all(r['path_id']=='ice.C' for r in output[0]['rows'])
    assert ice.cv2.imread(output[0]['identity_review_image']) is not None
    assert hashlib.sha256(ice.cv2.imread(str(source_image)).tobytes()).hexdigest()==image_sha
    assert not output[1]['fit_eligible'] and output[1]['rows']==[]
    assert output[1]['pending_fragments'][0]['reason']=='no_unique_approved_anchor_connection'
    assert result['qualified_rows']>0 and result['pending_fragments']==1
    assert ice.digest(candidate)==candidate_sha
    assert all(ice.digest(path)==sha for path,sha in raw_hashes.items())
    for name in ('identity_graph.json','source_labels.json','pending_source_rows.jsonl'):assert (folder/name).is_file()


def test_grouped_reader_maps_query_and_candidate_indices_without_changing_uv():
    def reader(marker):
        result=dict(query_curve=np.array([0],np.int16),query_uv=np.array([[marker,20.]],np.float32),
            query_normal=np.array([[0.,1.]]),query_interval=np.array([[-40.,40.]]),query_status=np.array([1]),
            candidate_query=np.array([0],np.int32),candidate_uv=np.array([[marker,21.]],np.float32),
            candidate_flags=np.array([0]),curve_status=np.array([marker],np.uint8))
        return SimpleNamespace(read=lambda *a,**kw: result)
    result=ice.read_material_groups([(reader(2),[1]),(reader(3),[0])],None,512,None,'s07',{},paths())
    np.testing.assert_array_equal(result['query_curve'],[1,0])
    np.testing.assert_array_equal(result['candidate_query'],[0,1])
    np.testing.assert_array_equal(result['candidate_uv'],[[2.,21.],[3.,21.]])
    np.testing.assert_array_equal(result['curve_status'],[3,2])


def test_cli_exposes_selected_shots_and_explicit_navigation_and_approval(monkeypatch):
    calls=[];monkeypatch.setattr(ice,'capture',lambda args:calls.append(args))
    monkeypatch.setattr(sys,'argv',['capture_independent_ice.py','--root','BOOT','--run-root','prepared',
        '--state','state.npz','--output','new','--shots','s08','--anchor-capture','references',
        '--anchor-approval','approval.json','--source-reference-navigation','--source-search-capture','centerline'])
    ice.main();args=calls[0]
    assert args.shots==['s08'] and args.anchor_approval==Path('approval.json')
    assert args.source_reference_navigation and args.source_search_capture==Path('centerline')


def test_incoming_graph_failure_blocks_only_that_edge_not_self_or_next_edge(tmp_path):
    graph,_=navigation(tmp_path,supported=(True,False,True))
    assert ice.navigation_between(graph,512,768)[0] is None
    assert ice.navigation_between(graph,512,1024)[0] is None
    np.testing.assert_array_equal(ice.navigation_between(graph,768,768)[0],np.eye(3))
    h,_=ice.navigation_between(graph,768,1024)
    np.testing.assert_allclose(h,[[1,0,0],[0,1,5],[0,0,1]])
    _,m,f=source_frame()
    rows,pending,_=ice.identity_rows(m,f,paths(),[dict(fragment=0,status='anchored',identities=['ice.C'])],[],False)
    assert rows and not pending


def test_approved_actual_source_signal_override_changes_readout_only(tmp_path):
    capture,approval,_=write_reference(tmp_path)
    source=capture/'s07/source_rows.jsonl';record=json.loads(source.read_text())
    for row in record['rows']:row['material_signal']='dark'
    source.write_text(json.dumps(record)+'\n',encoding='utf8')
    document=json.loads(approval.read_text());document['groups'][0]['source_rows_file_sha256']=ice.digest(source);ice.write(approval,document)
    declared,_=ice.approval_sources(capture,approval,['s07'],SOURCE);groups=ice.anchor_records(declared,SOURCE)
    inventory=paths();inventory[1]['material_signal']='red'
    modified,note=ice.anchor_signal_paths(inventory,groups)
    assert modified[1]['material_signal']=='dark' and inventory[1]['material_signal']=='red'
    assert note[0]['source_rows_file_sha256']==[ice.digest(source)]
    assert not note[0]['native_material_changed'] and not note[0]['identity_established']
    assert not record['fit_eligible']
    with pytest.raises(ValueError,match='disagree'):
        ice.anchor_signal_paths(inventory,groups+[dict(groups[0],material_signal='blue')])


def test_explicit_capture_pts_decode_only_contiguous_selected_runs(monkeypatch):
    windows=[]
    def frames(index,source):
        window=source['processing'];windows.append((window['first_pts'],window['last_pts']))
        for pts in range(window['first_pts'],window['last_pts']+256,256):yield pts,np.zeros((2,2,3),np.uint8)
    monkeypatch.setattr(ice,'frames',frames)
    source=dict(processing=dict(first_pts=512,last_pts=2304,stop_pts_exclusive=2560))
    found=list(ice.capture_frames({},source,[768,1024,2048]))
    assert windows==[(768,1024),(2048,2048)]
    assert [record[0] for record in found]==[1,2,6]
    assert [record[1][0] for record in found]==[768,1024,2048]
