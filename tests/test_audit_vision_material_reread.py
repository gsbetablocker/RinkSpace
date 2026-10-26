"""Same-exposure refinement must replace contributions and preserve history."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('torch')

from rinkspace.vision_material_reader import CurrentNativeMaterialReader
from rinkspace.vision_ice_identity_graph import IceSourceMaterialIdentityGraph


def _nested_reader(monkeypatch):
    import rinkspace.vision_ice_source_guides as profiles
    rgb=np.full((64,64,3),150,np.uint8)
    binding=dict(source_sha256='a'*64,source_pts=256,time_base='1/12800',shot_id='s01',native_hw=[64,64])
    scene=dict(path='native.json',sha256='c'*64)
    definitions=[dict(curve_id=f'{name}:center',feature_id=name,world_curve_role='center',operator='material_interface',
        material_signal='dark',material_transition='light_to_dark',native_path_id=f'native/{name}:center') for name in ('R3','R4')]
    def bind(guides):
        return [dict(g,native_geometry=dict(path_id=g['id'],scene=scene),primitive='circle',physical_role='filled_disc_boundary') for g in guides],{}
    reader=CurrentNativeMaterialReader(SimpleNamespace(ice_inventory=lambda:definitions,refs=[scene],bind=bind),device='cpu')
    reader.scope=('a'*64,'1/12800','s01');reader.inventory=definitions
    reader.source_context=SimpleNamespace(prepare=lambda *a:None,navigation_kind='actual_source_fixture')
    reader.selector=SimpleNamespace(consume=lambda *a,**k:dict(rows=[]),retain_final_source_rows=lambda *a,**k:{},latest_current_source_context_reentry=[])
    reader.graph=SimpleNamespace(consume=lambda *a,**k:dict(rows=[]),frames=[dict(ids=np.arange(4))])
    reader.output_queries={}
    monkeypatch.setattr(profiles,'read_actual_source_query_profiles',lambda *a,**k:dict(query_curve=[],candidate_query=np.empty(0,int)))
    def expand(rgb,measurement,excluded,*a,**kwargs):
        if excluded.any():return dict(rows=[])
        points=np.array([[10.,20.],[20.,20.],[40.,40.],[50.,40.]])
        measurement.update(query_uv=points,query_normal=np.tile([0.,1.],(4,1)),query_curve=np.array([0,0,1,1]),
            query_interval=np.array([[-16.,28.],[-23.,32.],[-18.,34.],[-24.,31.]]),
            candidate_query=np.arange(4),candidate_uv=points,candidate_sides=np.repeat(points[:,None,:],2,axis=1),
            candidate_flags=np.zeros(4,np.uint8))
        return dict(rows=[dict(curve_id=d['curve_id'],original_candidate_indices=[2*i,2*i+1],uv=points[2*i:2*i+2].tolist()) for i,d in enumerate(definitions)])
    reader.house_expander=SimpleNamespace(consume=expand)
    return reader,rgb,binding


def test_same_rgb_reread_rebuilds_nested_outputs_and_consumes_changed_masks(monkeypatch):
    reader,rgb,binding=_nested_reader(monkeypatch)
    near=dict(qualified_ice_visible_mask=np.ones(rgb.shape[:2],bool))
    guides=[];first=dict(readings=[])
    reader.refine_ice(rgb,binding,guides,first,near_board=near)
    assert len(first['readings'])==2
    np.testing.assert_array_equal(guides[0]['material_profile_intervals_px'],[[-16.,28.],[-23.,32.]])
    np.testing.assert_array_equal(guides[1]['material_profile_intervals_px'],[[-18.,34.],[-24.,31.]])
    repeated_guides=[];repeated=dict(readings=[])
    reader.refine_ice(rgb,binding,repeated_guides,repeated,near_board=near)
    assert len(repeated_guides)==len(repeated['readings'])==2
    for old,new in zip(first['readings'],repeated['readings']):
        np.testing.assert_array_equal(old['actual_source_uv'],new['actual_source_uv'])
    excluded=np.zeros(rgb.shape[:2],bool);excluded[20,10]=True
    revised=dict(readings=[])
    result=reader.refine_ice(rgb,binding,[],revised,near_board=near,global_exclusion=excluded)
    assert result['current_named_readout_count']==0
    assert revised['readings']==[]
    assert len(reader.output_queries)==1
    changed=rgb.copy();changed[0,0,0]+=1
    with pytest.raises(ValueError,match='Same PTS changed RGB'):
        reader.refine_ice(changed,binding,[],dict(readings=[]),near_board=near)


def test_failed_same_frame_revision_keeps_the_committed_reader_state(monkeypatch):
    import rinkspace.vision_ice_source_guides as profiles
    reader,rgb,binding=_nested_reader(monkeypatch)
    near=dict(qualified_ice_visible_mask=np.ones(rgb.shape[:2],bool))
    original=reader.refine_ice(rgb,binding,[],dict(readings=[]),near_board=near)
    before=dict(reader.output_queries)
    def fail(*args,**kwargs):raise RuntimeError('source reread failed')
    monkeypatch.setattr(profiles,'read_actual_source_query_profiles',fail)
    out=dict(readings=[]);guides=[]
    with pytest.raises(RuntimeError,match='source reread failed'):
        reader.refine_ice(rgb,binding,guides,out,near_board=near)
    assert reader.last is original
    assert reader.output_queries.keys()==before.keys()
    assert reader.output_queries[256] is before[256]
    assert out['readings']==guides==[]


def test_graph_revision_retracts_only_current_backward_labels_and_preserves_native_nodes(monkeypatch):
    import rinkspace.vision_ice_identity_graph as graph_module
    definitions=[dict(curve_id=name,feature_id=name,world_curve_role='boundary',operator='material_interface') for name in ('A','B')]
    class Graph(IceSourceMaterialIdentityGraph):
        def _nodes(self,rgb,measurement,mask):
            uv=np.array([[10.,10.],[20.,10.],[10.,20.],[20.,20.]])
            return dict(ids=np.arange(4),query_ids=np.arange(4),uv=uv,raw_uv=uv.copy(),
                sides=np.stack((uv-[0.,1.],uv+[0.,1.]),axis=1),normal=np.tile([0.,1.],(4,1)),
                width=np.ones(4)*2,curve=np.array([0,0,1,1]),descriptor=np.ones((4,3)),
                parents={},children={},label={},emitted=set())
    monkeypatch.setattr(graph_module,'_actual_material_links',lambda *a,**k:[(i,i,0.,1.) for i in range(4)])
    graph=Graph(definitions,source_sha256='a'*64,time_base='1/12800',device='cpu')
    rgb=np.full((32,32,3),150,np.uint8);mask=np.zeros(rgb.shape[:2],bool)
    def consume(pts,anchors=()):
        binding=dict(source_sha256='a'*64,source_pts=pts,time_base='1/12800',shot_id='shot')
        measurement=dict(source_binding=binding,decoded_native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
        rows=[dict(admission_status='admitted_source_physical_boundary_temporal_identity',
            original_candidate_indices=[2*index,2*index+1],physical_material_id=definitions[index]['curve_id'],
            source_seed_authority={'kind':'source_fixture'},source_seed_reference={'actual_pts':pts}) for index in anchors]
        return graph.consume(rgb,measurement,mask,source_ice_scope_mask=~mask,selected_rows=rows)
    consume(0,[0])
    original_nodes=graph.frames[0]['uv']
    graph.checkpoint_source_frame()
    before=graph._source_frame_checkpoint
    first=consume(256,[1])
    assert any(row['curve_id']=='B' and row['actual_pts']==0 for row in first['rows'])
    assert set(graph.frames[0]['label'])=={0,1,2,3}
    assert set(before['frames'][0]['label'])=={0,1}
    graph.restore_source_frame_state(before)
    graph.checkpoint_source_frame()
    consume(256)
    assert len(graph.frames)==2 and [row['actual_pts'] for row in graph.catalog]==[0,256]
    assert set(graph.frames[0]['label'])=={0,1}
    assert graph.frames[0]['uv'] is original_nodes
    assert all(len(links)==1 for links in graph.frames[0]['children'].values())
    graph.checkpoint_source_frame()
    future=consume(512,[1])
    assert {row['actual_pts'] for row in future['rows'] if row['curve_id']=='B'}=={0,256}


def test_actual_profiles_replace_same_exposure_mask_without_advancing_graph():
    import torch
    rgb=np.full((80,100,3),190,np.uint8);rgb[36:44,10:90]=50
    scene=dict(path='fixture.json',sha256='c'*64)
    definition=dict(curve_id='line:segment0',feature_id='line',world_curve_role='segment0',operator='paint_center',
        material_signal='dark',material_transition=None,native_path_id='native/line',native_kind='ice_segment')
    reader=CurrentNativeMaterialReader(SimpleNamespace(ice_inventory=lambda:[definition],refs=[scene]),device='cpu')
    def read(pts,masked=False):
        binding=dict(source_sha256='a'*64,source_pts=pts,time_base='1/12800',shot_id='shot',native_hw=[80,100])
        uv=np.c_[np.arange(25.,76.,10.),np.full(6,39.5)]
        guide=dict(id='native/line',kind='ice_paint',points_uv=uv,segment_ids=np.zeros(6,int),source_binding=binding,
            material_profile_intervals_px=np.tile([-16.,28.],(6,1)),
            native_geometry=dict(path_id='native/line',scene=scene),origin='user_confirmed_visual_supervision',identity_confirmed=True,
            identity_authority=dict(kind='user_confirmed_visual_supervision',reference='synthetic_fixture'),
            source_exclusion_mask=np.full((80,100),masked,bool),source_exclusion_binding=binding)
        row=dict(guide=dict(uv=torch.as_tensor(uv),normals=torch.tensor([[0.,1.]]*6)),
            actual_source_uv=torch.empty((0,2)),source_indices=torch.empty(0,dtype=torch.long),metadata={})
        result=reader.refine_ice(rgb,binding,[guide],dict(readings=[row]),
            near_board=dict(qualified_ice_visible_mask=np.ones((80,100),bool)))
        np.testing.assert_array_equal(result['measurement']['query_interval'],guide['material_profile_intervals_px'])
        assert reader.selector.model_context_identity is reader.source_context
        return row
    assert len(read(0)['actual_source_uv'])==6
    historical_nodes=reader.graph.frames[0]['uv']
    assert len(read(256)['actual_source_uv'])==6
    assert len(read(256,masked=True)['actual_source_uv'])==0
    assert [frame['pts'] for frame in reader.graph.frames]==[0,256]
    assert reader.graph.frames[0]['uv'] is historical_nodes
    assert len(read(512)['actual_source_uv'])==6
    assert [frame['pts'] for frame in reader.graph.frames]==[0,256,512]


def test_same_frame_refinement_updates_next_frame_navigation(monkeypatch):
    from test_vision_tracking_frame import reader_and_spies,source_rgb,binding,guide
    reader,_=reader_and_spies(monkeypatch)
    reader.read(source_rgb(8),binding(0),guides_2d=[guide()],reset=True,extract_source_components=False)
    original_predict=reader.vision.predict
    def repeated(*args,**kwargs):
        result=original_predict(*args,**kwargs)
        result.update(memory_update='reused_same_source_frame',new_observation=False)
        return result
    monkeypatch.setattr(reader.vision,'predict',repeated)
    reader.read(source_rgb(8),binding(0),guides_2d=[dict(guide(),id='revised_contact')],extract_source_components=False)
    assert reader.active_guides[0]['id']=='revised_contact'
