"""Guide-local exclusions remain local through real RGB profile/graph gates."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from rinkspace.vision_ice_source_guides import (
    apply_query_exclusions, read_actual_source_query_profiles,
)
from rinkspace.vision_material_reader import CurrentNativeMaterialReader


def source():
    rgb=np.full((96,128,3),220,np.uint8)
    rgb[40:57]=80
    binding=dict(source_sha256='a'*64,source_pts=10,time_base='1/1000',
        shot_id='s01',native_hw=[96,128])
    inventory=[dict(curve_id=name,feature_id=name,world_curve_role='segment0',
        native_path_id='native/'+name,operator='paint_center',
        material_signal='dark',material_transition=None) for name in ('A','B')]
    empty=np.zeros(rgb.shape[:2],bool)
    restricted=empty.copy();restricted[30:67]=True
    return rgb,binding,inventory,empty,restricted


def profiles(rgb,binding,inventory,global_mask,masks=None):
    return read_actual_source_query_profiles(rgb,binding,inventory,
        [[32.,48.],[96.,48.]],[[0.,1.]]*2,[[-24.,24.]]*2,[0,1],global_mask,
        query_guide_indices=[0,1],guide_exclusion_masks=masks)


def test_actual_rgb_profiles_use_each_guides_own_mask_and_keep_global_fallback():
    rgb,binding,inventory,empty,restricted=source()
    measured=profiles(rgb,binding,inventory,empty,{0:restricted,1:empty})
    np.testing.assert_array_equal(measured['candidate_query'],[0,1])
    np.testing.assert_allclose(measured['candidate_uv'],[[32.,48.],[96.,48.]])
    assert measured['candidate_flags'].tolist()==[10,8]
    # Swap ownership: eligibility follows its guide, not first/last order.
    swapped=profiles(rgb,binding,inventory,empty,{0:empty,1:restricted})
    assert swapped['candidate_flags'].tolist()==[8,10]
    default=profiles(rgb,binding,inventory,restricted)
    assert default['candidate_flags'].tolist()==[10,10]
    # A final per-guide reviewed exception never clears another guide or an
    # unassigned query's global restriction.
    exception=profiles(rgb,binding,inventory,restricted,{1:empty})
    assert exception['candidate_flags'].tolist()==[10,8]


@pytest.mark.parametrize('global_ignore',[False,True])
def test_native_material_entry_passes_scoped_masks_without_cross_guide_union(monkeypatch,global_ignore):
    import rinkspace.vision_ice_source_guides as module
    rgb,binding,inventory,empty,restricted=source()
    reader=CurrentNativeMaterialReader(SimpleNamespace(ice_inventory=lambda:inventory),device='cpu')
    reader.scope=('a'*64,'1/1000','s01');reader.inventory=inventory
    reader.selector=SimpleNamespace();reader.graph=SimpleNamespace()
    reader.output_queries={}
    guides=[];readings=[]
    for index,definition in enumerate(inventory):
        points=np.array([[32.+64*index,48.]])
        guides.append(dict(id=definition['curve_id'],kind='ice_paint',source_binding=binding,
            points_uv=points,native_geometry={'path_id':definition['native_path_id']},
            source_exclusion_mask=restricted if index==0 else empty,
            source_exclusion_binding=binding,material_profile_half_extent_px=24.))
        readings.append(dict(guide=dict(uv=points,normals=np.array([[0.,1.]]))))
    actual=module.read_actual_source_query_profiles
    def capture(*args,**kwargs):
        measurement=actual(*args,**kwargs)
        assert measurement['candidate_flags'].tolist()==[10,8]
        assert kwargs['query_guide_indices']==[0,1]
        raise RuntimeError('scoped RGB readout verified')
    monkeypatch.setattr(module,'read_actual_source_query_profiles',capture)
    with pytest.raises(RuntimeError,match='scoped RGB readout verified'):
        reader.refine_ice(rgb,binding,guides,dict(readings=readings),near_board={},
            global_exclusion=restricted if global_ignore else empty)


def test_final_q50_scoped_exclusion_removes_graph_node_without_affecting_other_guide(tmp_path):
    import test_vision_ice_final_precision_graph as fixture
    graph,rgb,measurement,row=fixture.source(tmp_path)
    original=deepcopy(measurement)
    mask=np.zeros(rgb.shape[:2],bool);mask[81,45]=True
    # Raw source context ends at y=80.5; exact q50 moves it into pixel 81.
    measurement['candidate_side_quantiles_uv'][0,:,:,1]+=.031
    apply_query_exclusions(measurement,np.zeros_like(mask),
        query_guide_indices=[0,1,1,1,1,1],guide_exclusion_masks={0:mask,1:np.zeros_like(mask)})
    assert measurement['candidate_flags'][0]&2
    assert np.all((measurement['candidate_flags'][1:]&2)==0)
    selected=dict(rows=[row])
    graph.consume(rgb,measurement,np.zeros_like(mask),source_ice_scope_mask=np.ones_like(mask),
        selected_rows=selected)
    assert 0 not in graph.frames[-1]['ids']
    assert selected['rows'][0]['original_candidate_indices']==list(range(1,6))
    np.testing.assert_array_equal(measurement['candidate_uv'],original['candidate_uv'])
    np.testing.assert_array_equal(measurement['candidate_sides'],original['candidate_sides'])


def test_nested_discovery_does_not_borrow_another_curves_mask_or_promote_blocked_authority():
    torch=pytest.importorskip('torch')
    rgb,binding,inventory,empty,restricted=source()
    scene={'path':'current.json','sha256':'c'*64}
    def bind(guides):
        return [dict(g,native_geometry={'path_id':g['id'],'scene':scene},
            primitive='line',physical_role='painted_stroke_center_axis') for g in guides],{}
    geometry=SimpleNamespace(ice_inventory=lambda:inventory,refs=[scene],bind=bind)
    reader=CurrentNativeMaterialReader(geometry,device='cpu')
    reader.scope=('a'*64,'1/1000','s01');reader.inventory=inventory
    reader.output_queries={}
    reader.source_context=SimpleNamespace(prepare=lambda *a:None,navigation_kind='test')
    reader.selector=SimpleNamespace(consume=lambda *a,**k:dict(rows=[]),
        retain_final_source_rows=lambda *a,**k:{},latest_current_source_context_reentry=[])
    seen={}
    def expand(rgb,measurement,mask,**kwargs):
        assert not mask.any()  # A's restriction is not a global discovery veto.
        points=np.array([[32.,48.],[40.,48.],[88.,48.],[96.,48.]])
        measurement.update(query_uv=points,query_normal=np.tile([0.,1.],(4,1)),
            query_interval=np.tile([-18.,18.],(4,1)),
            query_curve=np.array([0,0,1,1]),candidate_query=np.arange(4),
            candidate_uv=points.copy(),candidate_sides=np.repeat(points[:,None],2,axis=1),
            candidate_flags=np.full(4,8,np.uint8),candidate_side_quantiles_uv=np.full((4,2,3,2),np.nan))
        return dict(rows=[dict(curve_id=definition['curve_id'],original_candidate_indices=[2*i,2*i+1],
            uv=points[2*i:2*i+2].tolist()) for i,definition in enumerate(inventory)])
    reader.house_expander=SimpleNamespace(consume=expand,inventory=inventory)
    def consume(rgb,measurement,mask,**kwargs):
        seen['flags']=measurement['candidate_flags'].tolist()
        seen['names']=[row['curve_id'] for row in kwargs['selected_rows']['rows']]
        return dict(rows=[])
    reader.graph=SimpleNamespace(consume=consume,frames=[dict(ids=np.array([2,3]))])
    guide=dict(id='A',kind='ice_paint',source_binding=binding,points_uv=np.array([[32.,48.]]),
        native_geometry={'path_id':'native/A','scene':scene},temporal_flow_supported=[False],
        source_exclusion_mask=restricted,source_exclusion_binding=binding)
    reading=dict(guide=dict(uv=torch.tensor([[32.,48.]]),normals=torch.tensor([[0.,1.]])),
        actual_source_uv=torch.empty((0,2)),source_indices=torch.empty(0,dtype=torch.long),metadata={})
    guides=[guide]
    reader.refine_ice(rgb,binding,guides,dict(readings=[reading]),
        near_board=dict(qualified_ice_visible_mask=np.ones_like(empty)))
    assert seen==dict(flags=[10,10,8,8],names=['B'])
    assert guides[1]['identity_confirmed'] is False
    assert guides[1]['identity_authority']['source_rows']==[]
    assert guides[2]['identity_confirmed'] is True
    assert not guides[2]['source_exclusion_mask'].any()


def test_real_global_box_clearance_preserves_overlapping_guide_only_veto():
    import test_vision_ice_local_clearance as fixture
    seed=dict(curve_id='R4:outer',actual_pts=20,source_sha256='a'*64,time_base='1/1000',shot_id='one',
        source_material_class='static_painted_ice',surface_and_physical_identity_confirmed=True,
        authority=dict(path='actual-original-source-identity.json',sha256='b'*64),
        navigation_uv=[[270.,170.],[170.,270.]])
    reader=fixture.setup([seed])
    first,last=fixture.material(10),fixture.material(20)
    person=np.zeros(first[0].shape[:2],bool)
    fixture.consume(reader,*first,person)
    # The actual two-RGB material/flow/silhouette checks grant both original
    # candidates local clearance from the coarse global detector box.
    cleared=fixture.consume(reader,*last,person)
    measurement=cleared['measurement']
    assert measurement['candidate_flags'].tolist()==[8,8]
    assert set(cleared['candidate_local_source_clearance'])=={0,1}
    original_uv=measurement['candidate_uv'].copy()
    global_box=np.ones_like(person)
    protected=np.zeros_like(person);protected[270,170]=True
    assert global_box[270,170] and not cleared['exclusion_mask'][270,170]
    apply_query_exclusions(measurement,cleared['exclusion_mask'],query_guide_indices=[0,1],
        guide_exclusion_masks={0:global_box,1:global_box},
        guide_protected_masks={0:person,1:protected},
        cleared_global_mask=global_box&~cleared['exclusion_mask'])
    assert measurement['candidate_flags'].tolist()==[8,10]
    np.testing.assert_array_equal(measurement['candidate_uv'],original_uv)


def test_source_reader_retains_explicit_mask_component_when_it_overlaps_global(monkeypatch):
    import test_vision_tracking_frame as fixture
    reader,_=fixture.reader_and_spies(monkeypatch)
    reader.read(fixture.source_rgb(8),fixture.binding(0),guides_2d=[fixture.guide()],
        ignored_regions=[{'x_range_source_px':[2,5]},
                         {'x_range_source_px':[3,4],'applies_only_to_feature_ids':['contact']}],
        reset=True,extract_source_components=False)
    guide=reader.active_guides[0]
    assert guide['source_exclusion_mask'][0,0,4,3]
    assert reader.active_global_exclusion[0,0,4,3]
    assert guide['source_guide_exclusion_mask'][0,0,4,3]
    assert guide['source_exclusion_protected_mask'][0,0,4,3]
    assert not guide['source_exclusion_protected_mask'][0,0,4,2]
