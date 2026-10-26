"""The actual entry preserves known source work and rejects unrelated geometry."""
from types import SimpleNamespace
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from rinkspace.vision_material_reader import CurrentNativeMaterialReader, install_readout
from rinkspace.vision_native import NativeSourceGeometry


def test_native_output_rejects_removed_or_foreign_paths():
    geo=NativeSourceGeometry.__new__(NativeSourceGeometry)
    scene={'path':'current.json','sha256':'a'*64}
    row=dict(id='native/retained',layer='advertisement',measurement='contour')
    geo.paths={'Ad2.retained':row};geo.refs=[scene]
    guide=dict(id='Ad2.removed',native_geometry=dict(path_id='native/retained',scene=scene))
    with pytest.raises(ValueError,match='retained current native'):
        geo.check_output([guide],dict(readings=[{}]))
    guide['id']='Ad2.retained';guide['native_geometry']['scene']={**scene,'sha256':'b'*64}
    with pytest.raises(ValueError,match='another native scene'):
        geo.check_output([guide],dict(readings=[{}]))


def test_install_readout_preserves_accepted_uv_and_does_not_bridge_missing_query():
    accepted=[[1.,2.],[3.,4.],[5.,6.]]
    row=dict(actual_source_uv=torch.tensor([[2.,3.]]),source_indices=torch.tensor([0]),
        confirmed_source_uv=accepted,metadata={},support={'method':'existing_current_RGB'})
    guide=dict(points_uv=np.zeros((3,2)),segment_ids=[0,0,0])
    install_readout(row,guide,[0,2],[[10.,20.],[30.,40.]],method='test_current_RGB')
    assert row['confirmed_source_uv'] is accepted
    assert len(row['measured_runs'])==2
    assert row['pre_material_readout']['actual_source_uv'].tolist()==[[2.,3.]]
    assert row['actual_source_uv'].tolist()==[[10.,20.],[30.,40.]]


def test_print_half_transition_uses_current_rgb_and_respects_mask_and_hole_identity():
    # Native dimensions are required by the existing reusable source profiler.
    rgb=np.full((1080,1920,3),200,np.uint8);rgb[:,:600]=[70,40,100]
    binding=dict(source_sha256='a'*64,source_pts=256,time_base='1/12800',shot_id='one',native_hw=[1080,1920])
    definition=dict(id='print_instance/Ad2::Ad2.hole',layer='advertisement',
        role='letter_hole_boundary',parent_contour_key='Ad2.outer')
    geometry=SimpleNamespace(ice_inventory=lambda:[],lookup=lambda name:definition)
    reader=CurrentNativeMaterialReader(geometry,device='cpu')
    original=np.array([[600.,200.],[600.,300.],[600.,400.]])
    mask=np.zeros(rgb.shape[:2],bool);mask[299:302,:]=True
    guide=dict(id='Ad2.hole',kind='ad_boundary',points_uv=original,segment_ids=[0,0,0],
        source_binding=binding,source_exclusion_mask=mask,source_exclusion_binding=binding,normals=np.zeros_like(original))
    row=dict(actual_source_uv=torch.empty((0,2)),source_indices=torch.empty(0,dtype=torch.long),
        metadata={},confirmed_source_uv=original.copy(),guide=dict(normals=torch.zeros((3,2))))
    result=reader.refine_print(rgb,binding,[guide],dict(readings=[row]),
        near_board={'vertical_face_admission_mask':np.ones(rgb.shape[:2],bool)})
    assert row['source_indices'].tolist()==[0,2]
    assert np.max(np.abs(row['actual_source_uv'].numpy()[:,0]-599.5))<.4
    np.testing.assert_array_equal(guide['points_uv'],original)
    np.testing.assert_array_equal(row['confirmed_source_uv'],original)
    assert result['readings'][0]['native_role']=='letter_hole_boundary'
    assert result['readings'][0]['parent_contour_key']=='Ad2.outer'
    np.testing.assert_array_equal(row['guide']['normals'].numpy()[[0,2]],[[-1.,0.],[-1.,0.]])
    np.testing.assert_array_equal(guide['normals'],np.zeros_like(original))
    from rinkspace.session.source_observations import qualify_measurement
    row.update(native_path_id=definition['id'],expected_id=definition['id'],native_measurement='contour')
    candidates=qualify_measurement(dict(binding=binding,rgb_sha256='b'*64),row,
        dict(identity_confirmed=True,identity_authority={'kind':'test_source'}))
    assert [c.normal for c in candidates]==[[-1.,0.],[-1.,0.]]
    # A real source change moves the readout; no old point is copied as truth.
    shifted=rgb.copy();shifted[:,:602]=[70,40,100]
    later=dict(binding,source_pts=512)
    guide.update(source_binding=later,source_exclusion_binding=later)
    row2=dict(actual_source_uv=torch.empty((0,2)),source_indices=torch.empty(0,dtype=torch.long),metadata={})
    reader.refine_print(shifted,later,[guide],dict(readings=[row2]),near_board={})
    assert np.max(np.abs(row2['actual_source_uv'].numpy()[:,0]-601.5))<.4


def test_dense_face_requires_paired_current_named_material_and_keeps_exclusions(monkeypatch):
    from rinkspace.vision_board_reader import dense_front_face_admission
    import calibration.program_tracking.source_board_measurements as source
    x=np.arange(600.,881.,28)
    band=dict(candidate_id='paired',blue_upper_uv=np.c_[x,np.full(len(x),220.)],
        yellow_lower_uv=np.c_[x,np.full(len(x),500.)],
        blue_transition_interval_y_px=np.ones((len(x),2)),yellow_transition_interval_y_px=np.ones((len(x),2)))
    monkeypatch.setattr(source,'measure_board_source_frame',lambda *a,**k:dict(boundary_candidates=[band]))
    guides=[dict(id='top',kind='board_top',source_board_view='front_face'),
        dict(id='foot',kind='ice_contact',source_board_view='front_face')]
    rows=[dict(actual_source_uv=torch.tensor([[700.,220.]])),dict(actual_source_uv=torch.tensor([[760.,500.]]))]
    rgb=np.zeros((1080,1920,3),np.uint8);excluded=np.zeros(rgb.shape[:2],bool);excluded[300:400,700:720]=True
    mask,receipt,_=dense_front_face_admission(rgb,dict(source_pts=256),guides,dict(readings=rows),excluded,device='cpu')
    assert mask[0,0,280,750] and not mask[0,0,350,710]
    assert not mask[0,0,210,750] and not mask[0,0,510,750]
    guides[0]['source_board_view']='top_face_only'
    mask,receipt,_=dense_front_face_admission(rgb,dict(source_pts=256),guides,dict(readings=rows),excluded,device='cpu')
    assert not mask.any()


def test_ice_material_precision_cannot_revive_invalid_temporal_flow(monkeypatch):
    from rinkspace.vision_material_reader import CurrentNativeMaterialReader
    import rinkspace.vision_ice_source_guides as profiles
    import hashlib
    rgb=np.full((64,64,3),150,np.uint8)
    binding=dict(source_sha256='a'*64,source_pts=256,time_base='1/12800',shot_id='s01',native_hw=[64,64])
    definition=dict(curve_id='line:segment0',feature_id='line',world_curve_role='segment0',
        operator='paint_center',material_signal='dark',material_transition=None,native_path_id='native/line')
    reader=CurrentNativeMaterialReader(SimpleNamespace(ice_inventory=lambda:[definition],refs=[{'path':'current.json','sha256':'c'*64}]),device='cpu')
    reader.reset(binding)
    guide=dict(id='ice.line:segment0',kind='ice_paint',source_binding=binding,
        native_geometry={'path_id':'native/line'},points_uv=np.array([[20.,20.],[30.,20.]]),
        temporal_flow_supported=np.array([False,True]),segment_ids=[0,0])
    reading=dict(guide=dict(uv=torch.tensor(guide['points_uv']),normals=torch.tensor([[0.,1.],[0.,1.]])),
        actual_source_uv=torch.empty((0,2)),source_indices=torch.empty(0,dtype=torch.long),metadata={})
    def capture(rgb,binding,inventory,points,normals,intervals,curves,mask,**kwargs):
        assert np.asarray(points).tolist()==[[30.,20.]]
        raise RuntimeError('query_filter_verified')
    monkeypatch.setattr(profiles,'read_actual_source_query_profiles',capture)
    with pytest.raises(RuntimeError,match='query_filter_verified'):
        reader.refine_ice(rgb,binding,[guide],dict(readings=[reading]),near_board={})


def test_authenticated_saved_identity_reaches_current_rgb_precision_queries(monkeypatch):
    from rinkspace.vision_material_reader import CurrentNativeMaterialReader
    definition=dict(curve_id='line:segment0',feature_id='line',world_curve_role='segment0',
        native_kind='ice_segment',operator='paint_center',material_signal='dark',
        material_transition=None,native_path_id='native/line')
    scene={'path':'current.json','sha256':'c'*64}
    geometry=SimpleNamespace(ice_inventory=lambda:[definition],refs=[scene])
    reader=CurrentNativeMaterialReader(geometry,device='cpu')
    binding=dict(source_sha256='a'*64,source_pts=256,time_base='1/12800',shot_id='s02',native_hw=[64,64])
    reader.reset(binding)
    rgb=np.full((64,64,3),190,np.uint8);rgb[26:31]=70
    points=np.array([[14.,26.],[28.,26.],[42.,26.]])
    authority=dict(kind='authenticated_existing_source_material_identity',source_binding=binding,
        current_native_path_id='native/line',source_records=[dict(id='original-human-point',source_ref={'sha256':'d'*64})],
        independent_identity_basis='exact_original_human_named_point_or_typed_physical_board_material',
        identity_from_projection_alone=False,human_pixel_acceptance=False)
    guide=dict(id='ice.line:segment0',kind='ice_paint',points_uv=points,source_binding=binding,
        native_geometry=dict(path_id='native/line',scene=scene),segment_ids=[0,0,0],
        identity_confirmed=True,identity_authority=authority,origin='source_annotation_search_hint')
    reading=dict(guide=dict(uv=torch.tensor(points),normals=torch.tensor([[0.,1.]]*3)),
        actual_source_uv=torch.empty((0,2)),source_indices=torch.empty(0,dtype=torch.long),metadata={})
    def check_current_support(image,measurement,mask,*,seeds,**kwargs):
        assert len(seeds)==1 and seeds[0]['authority']==authority
        assert reader.source_context.navigation_flags['verified_source_guides_used_as_navigation'] is True
        assert seeds[0]['is_exact_centerline_pixel_ground_truth'] is False
        candidates=np.asarray(measurement['candidate_uv'])
        assert len(candidates)>0 and np.max(np.abs(candidates[:,1]-28.))<.6
        assert not np.array_equal(candidates,points)
        raise RuntimeError('authenticated identity and fresh RGB profiles reached selector')
    reader.selector.consume=check_current_support
    with pytest.raises(RuntimeError,match='fresh RGB profiles'):
        reader.refine_ice(rgb,binding,[guide],dict(readings=[reading]),
            near_board=dict(qualified_ice_visible_mask=np.ones((64,64),bool)))
    np.testing.assert_array_equal(points,guide['points_uv'])


def test_saved_source_identity_continuation_requires_same_shot_actual_flow():
    from rinkspace.vision_material_reader import _stored_source_identity
    source=dict(source_sha256='a'*64,source_pts=256,time_base='1/12800',shot_id='s02')
    guide=dict(native_geometry={'path_id':'native/line'},origin='temporal_source_search_hint',
        temporal_flow_supported=np.array([True,False]),hint_previous_source_binding=source,
        identity_authority=dict(kind='authenticated_existing_source_material_identity',source_binding=source,
            source_records=[{'id':'original'}],independent_identity_basis='original_human_point',
            current_native_path_id='native/line',identity_from_projection_alone=False))
    assert _stored_source_identity(guide,dict(source,source_pts=512))
    assert not _stored_source_identity(guide,dict(source,source_pts=512,shot_id='s03'))
    assert not _stored_source_identity(dict(guide,temporal_flow_supported=np.array([False,False])),dict(source,source_pts=512))


def test_later_identity_revision_updates_only_its_original_frame_and_preserves_manual_uv():
    from rinkspace.session.source_read import apply_delayed_source_readouts
    from copy import deepcopy
    binding=dict(source_sha256='a'*64,source_pts=256,time_base='1/12800',shot_id='one',
        native_hw=[64,64],decoded_native_rgb_sha256='b'*64)
    scene=dict(path='current_native.json',sha256='c'*64)
    row=dict(expected_id='ice.line',native_path_id='native/line',source_indices=[0],actual_source_uv=[[2.,10.]],
        guide=dict(segment_ids=[0,0,0]),metadata=dict(source_point_count=3),confirmed_source_uv=[[1.,10.],[9.,10.]],
        source_group_id='original_correlated_group')
    original=dict(source_binding=binding,native_geometry=dict(scene=scene),guided=dict(readings=[row]))
    revision=dict(source_binding=binding,evidence_source_binding=dict(binding,source_pts=512),native_scene=scene,
        decoded_native_RGB_sha256='b'*64,guide_index=0,expected_id='ice.line',native_path_id='native/line',
        source_indices=[2],actual_source_uv=[[8.5,10.125]])
    before=deepcopy(original)
    result=apply_delayed_source_readouts(original,[revision]);updated=result['guided']['readings'][0]
    assert original==before
    assert updated['source_indices']==[0,2] and len(updated['measured_runs'])==2
    assert updated['confirmed_source_uv']==row['confirmed_source_uv']
    assert updated['source_group_id']=='original_correlated_group'
    assert result['source_binding']['source_pts']==256
    bad=deepcopy(revision);bad['evidence_source_binding']['shot_id']='two'
    with pytest.raises(ValueError,match='this shot'):
        apply_delayed_source_readouts(original,[bad])
    bad=deepcopy(revision);bad['native_scene']['sha256']='d'*64
    with pytest.raises(ValueError,match='another native geometry'):
        apply_delayed_source_readouts(original,[bad])


def test_delayed_graph_rows_retain_original_query_and_native_path():
    geometry=SimpleNamespace(ice_inventory=lambda:[])
    reader=CurrentNativeMaterialReader(geometry,device='cpu')
    binding=dict(source_sha256='a'*64,source_pts=256,time_base='1/12800',shot_id='one')
    reader.output_queries={256:dict(source_binding=binding,RGB_sha256='b'*64,
        candidate_query=np.array([1,0]),owners=np.array([3,3]),local_indices=np.array([8,4]),
        guides={3:dict(expected_id='ice.line',native_path_id='native/line',curve_id='line',native_scene={})})}
    row=dict(actual_pts=256,source_binding=binding,decoded_native_RGB_sha256='b'*64,curve_id='line',
        original_candidate_indices=[0,1],uv=[[40.,10.],[80.,10.]],temporal_evidence={'direction':'backward'},source_seed_authority={})
    result=reader.delayed_readouts(dict(rows=[row]),dict(binding,source_pts=512))
    assert result[0]['guide_index']==3 and result[0]['source_indices']==[4,8]
    assert result[0]['source_binding']['source_pts']==256
    bad=dict(row,curve_id='different_line')
    with pytest.raises(ValueError,match='changed native path'):
        reader.delayed_readouts(dict(rows=[bad]),dict(binding,source_pts=512))


def test_native_gray_context_uses_only_current_source_navigation_and_preserves_gaps():
    from rinkspace.vision_material_reader import CurrentNativeSourceMaterialContext
    scene=dict(path='native.json',sha256='c'*64)
    definition=dict(curve_id='house_connection_F3_R3:segment0',feature_id='house_connection_F3_R3',
        native_path_id='native/F3',native_kind='ice_segment',operator='paint_center',material_signal='dark')
    binding=dict(source_sha256='a'*64,source_pts=256,time_base='1/12800',shot_id='one',native_hw=[64,64])
    context=CurrentNativeSourceMaterialContext(SimpleNamespace(refs=[scene]),[definition],binding)
    points=np.array([[10.,20.],[20.,20.],[30.,20.],[40.,20.],[50.,20.]])
    guide=dict(kind='ice_paint',id='ice.F3',points_uv=points,segment_ids=[0]*5,source_binding=binding,
        native_geometry=dict(scene=scene,path_id='native/F3'),identity_confirmed=True,
        identity_authority={'kind':'user_confirmed_visual_supervision','reference':'original'},
        origin='temporal_source_search_hint',temporal_flow_supported=[True,True,False,True,True])
    context.prepare([guide],binding)
    paths=context.navigation({'source_binding':binding})
    assert len(paths[0])==2 and sum(map(len,paths[0]))==4
    assert context.camera_reference is None
    assert context.navigation_flags['model_projection_used_as_navigation_and_identity_context'] is False
    assert context._source_context_output_curve_allowed(definition)
    np.testing.assert_array_equal(points,guide['points_uv'])
    wrong=dict(guide,native_geometry=dict(scene={'path':'old.json','sha256':'d'*64},path_id='native/F3'))
    with pytest.raises(ValueError,match='current native'):
        context.prepare([wrong],binding)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='native optical flow uses CUDA')
def test_front_face_continuity_transports_only_verified_region_and_reapplies_exclusions():
    from rinkspace.vision_board_reader import continue_front_face_admission
    previous=torch.zeros((1,1,16,24),device='cuda',dtype=torch.bool)
    previous[:,:,4:12,3:10]=True
    current=torch.zeros_like(previous);exclusion=torch.zeros_like(previous)
    backward=torch.zeros((1,2,16,24),device='cuda');backward[:,0]=-2
    valid=torch.ones_like(previous);valid[:,:,:,8]=False
    exclusion[:,:,6:9,6]=True
    a=dict(source_sha256='a'*64,source_pts=256,time_base='1/12800',shot_id='s01',native_hw=[16,24])
    b=dict(a,source_pts=512)
    flow=SimpleNamespace(backward=backward,valid_backward=valid,
        metadata=dict(source_bindings=[a,b],source_alignment_verified=True))
    actual,proof=continue_front_face_admission(current,previous,b,a,flow,exclusion)
    expected=torch.zeros_like(previous);expected[:,:,4:12,5:12]=True
    expected&=valid&~exclusion
    assert torch.equal(actual,expected)
    assert not proof['source_UV_created'] and proof['current_RGB_reread_required']
    assert not torch.any(actual&exclusion) and not torch.any(actual&~valid)
    assert torch.equal(previous[:,:,4:12,3:10],torch.ones((1,1,8,7),device='cuda',dtype=torch.bool))
    with pytest.raises(ValueError,match='source frames'):
        continue_front_face_admission(current,previous,dict(b,shot_id='s02'),a,flow,exclusion)


def test_new_nested_material_rows_survive_saved_delayed_revision(monkeypatch):
    import hashlib
    import rinkspace.vision_ice_source_guides as profiles
    from rinkspace.vision import _source_json
    from rinkspace.session.source_read import apply_delayed_source_readouts
    rgb=np.full((64,64,3),150,np.uint8)
    binding=dict(source_sha256='a'*64,source_pts=256,time_base='1/12800',shot_id='s01',native_hw=[64,64],
        decoded_native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
    scene=dict(path='current_native.json',sha256='c'*64)
    definitions=[dict(curve_id=f'{name}:center',feature_id=name,world_curve_role='center',operator='material_interface',
        material_signal='dark',material_transition='light_to_dark',native_path_id=f'native/{name}:center') for name in ('R3','R4')]
    def bind(guides):
        return [dict(g,native_geometry=dict(path_id=g['id'],scene=scene),primitive='circle',
            physical_role='filled_disc_boundary') for g in guides],{}
    geometry=SimpleNamespace(ice_inventory=lambda:definitions,refs=[scene],bind=bind)
    reader=CurrentNativeMaterialReader(geometry,device='cpu')
    reader.scope=('a'*64,'1/12800','s01');reader.inventory=definitions
    reader.source_context=SimpleNamespace(prepare=lambda *a:None,navigation_kind='test_actual_source')
    reader.selector=SimpleNamespace(consume=lambda *a,**k:dict(rows=[]),retain_final_source_rows=lambda *a,**k:{},
        latest_current_source_context_reentry=[])
    reader.graph=SimpleNamespace(consume=lambda *a,**k:dict(rows=[]),frames=[dict(ids=np.arange(4))])
    reader.output_queries={}
    monkeypatch.setattr(profiles,'read_actual_source_query_profiles',lambda *a,**k:dict(query_curve=[]))
    def expand(rgb,measurement,*a,**kwargs):
        points=np.array([[10.,20.],[20.,20.],[40.,40.],[50.,40.]])
        measurement.update(query_uv=points,query_normal=np.tile([0.,1.],(4,1)),
            query_interval=np.tile([-128.,128.],(4,1)),
            query_curve=np.array([0,0,1,1]),candidate_query=np.arange(4),
            candidate_uv=points.copy(),candidate_sides=np.repeat(points[:,None],2,axis=1),
            candidate_flags=np.full(4,8,np.uint8))
        return dict(rows=[dict(curve_id=d['curve_id'],original_candidate_indices=[2*i,2*i+1],
            uv=points[2*i:2*i+2].tolist()) for i,d in enumerate(definitions)])
    reader.house_expander=SimpleNamespace(consume=expand)
    guides=[];guided=dict(readings=[])
    reader.refine_ice(rgb,binding,guides,guided,near_board=dict(qualified_ice_visible_mask=np.ones((64,64),bool)))
    for guide,row in zip(guides,guided['readings']):row['native_path_id']=guide['native_geometry']['path_id']
    saved=_source_json(dict(source_binding=binding,native_geometry=dict(scene=scene),guided=guided))
    for gi,row in enumerate(saved['guided']['readings']):
        assert row['guide']['segment_ids']==[0,1]
        revision=dict(source_binding=binding,evidence_source_binding=dict(binding,source_pts=512),native_scene=scene,
            decoded_native_RGB_sha256=binding['decoded_native_rgb_sha256'],guide_index=gi,expected_id=row['expected_id'],
            native_path_id=row['native_path_id'],source_indices=row['source_indices'],actual_source_uv=row['actual_source_uv'])
        updated=apply_delayed_source_readouts(saved,[revision])['guided']['readings'][gi]
        assert len(updated['measured_runs'])==2 and updated['segment_ids']==[0,1]
        assert updated['actual_source_uv']==row['actual_source_uv']


def test_legacy_revision_recovers_saved_segments_without_inventing_missing_connectivity():
    from rinkspace.session.source_read import _original_query_segments
    row=dict(guide={},metadata=dict(source_point_count=3),source_indices=[0,2],segment_ids=[7,9],
        measured_runs=[dict(source_indices=[0],segment_id=7),dict(source_indices=[2],segment_id=9)])
    assert _original_query_segments(row,[0,2])==[7,9]
    assert 'segment_ids' not in row['guide']
    with pytest.raises(ValueError,match='cannot invent'):
        _original_query_segments(row,[0,1,2])


def test_native_context_discovers_identity_only_after_two_real_material_exposures(tmp_path):
    import hashlib
    from rinkspace.vision_material_reader import CurrentNativeSourceMaterialContext
    from rinkspace.vision_ice_temporal_identity import TemporalIceIdentitySelector
    from test_vision_ice_model_context_identity import source
    from calibration.program_tracking.strict_absolute_source import _temporal_identity
    path=tmp_path/'native.json';path.write_text('{}')
    scene=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    definition=dict(curve_id='B1:segment0',feature_id='B1',world_curve_role='segment0',
        native_path_id='native/B1',native_kind='ice_segment',operator='paint_center',material_signal='blue')
    _,first=source(10)
    context=CurrentNativeSourceMaterialContext(SimpleNamespace(refs=[scene]),[definition],first['source_binding'])
    selector=TemporalIceIdentitySelector([definition],source_sha256='a'*64,time_base='1/1000',model_context_identity=context)
    for pts,shift in ((10,0),(20,2)):
        rgb,measurement=source(pts,shift);binding=measurement['source_binding'];mask=np.zeros(rgb.shape[:2],bool)
        navigation=np.array([[0.,64.5+shift],[239.,64.5+shift]])
        query=dict(id='native/B1',kind='ice_paint',points_uv=navigation,segment_ids=[0,0],source_binding=binding,
            identity_confirmed=False,identity_authority=None,origin='model_projection_search_hint',
            native_geometry=dict(scene=scene,path_id='native/B1'),
            model_search_projection=dict(model_revision=scene['sha256'],native_pts=pts,camera_revision='c'*64))
        context.prepare([query],binding)
        result=selector.consume(rgb,measurement,mask,source_ice_scope_mask=~mask)
        assert query['identity_confirmed'] is False and context.authorized=={}
        if pts==10:assert not result['rows']
    assert len(result['rows'])==1 and _temporal_identity(result['rows'][0])
    row=result['rows'][0]
    np.testing.assert_array_equal(row['uv'],np.asarray(row['source_sides_uv']).mean(1))
    assert not np.allclose(np.asarray(row['uv'])[:,1],navigation[0,1])
    assert row['source_identity_navigation_evidence']['actual_field_order_confirmed']
    assert row['projected_UV_used_as_source_pixel_truth'] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest()==scene['sha256']
    query['model_search_projection']['native_pts']=10
    with pytest.raises(ValueError,match='exposure'):context.prepare([query],binding)


def test_unknown_saved_name_cannot_supply_its_own_field_order_context():
    from rinkspace.vision_material_reader import CurrentNativeSourceMaterialContext
    scene=dict(path='current.json',sha256='c'*64)
    definition=dict(curve_id='line:segment0',feature_id='line',world_curve_role='segment0',
        native_path_id='native/line',native_kind='ice_segment',operator='paint_center',material_signal='dark')
    binding=dict(source_sha256='a'*64,source_pts=256,time_base='1/12800',shot_id='s02')
    context=CurrentNativeSourceMaterialContext(SimpleNamespace(refs=[scene]),[definition],binding)
    guide=dict(kind='ice_paint',id='native/line',points_uv=[[0.,10.],[40.,10.]],segment_ids=[0,0],
        source_binding=binding,native_geometry=dict(scene=scene,path_id='native/line'),identity_confirmed=False)
    context.prepare([guide],binding)
    assert context.navigation({'source_binding':binding})==[[]]
    assert not context._source_context_output_curve_allowed(definition)
