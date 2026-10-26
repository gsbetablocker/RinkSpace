"""Model locations are current-frame search hints, never accepted source points."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rinkspace.vision_navigation import (NativeNavigation, PROJECTION_TAG,
    _selected_indices, merge_navigation_guides)
from rinkspace.vision_native import NativeSourceGeometry, file_ref


ROOT = Path(__file__).resolve().parents[1]


def source_binding(pts=0, shot='one'):
    return dict(source_sha256='a'*64, source_pts=pts, time_base='1/12800',
                shot_id=shot, native_hw=[10, 12])


def navigation_guide(binding, *, identity='path', x=2.):
    return dict(id=identity, points_uv=[[x, 4.], [x, 6.]], source_binding=binding,
        origin='model_projection_search_hint', navigation_provenance=PROJECTION_TAG,
        identity_confirmed=False, identity_authority=None, hint_is_source_measurement=False,
        kind='ice_contact', primitive='free_curve', physical_role='ice_board_contact')


def test_navigation_refreshes_only_unconfirmed_projection_and_retains_source_track():
    binding = source_binding(256)
    old = navigation_guide(binding, x=3.)
    old.update(origin='temporal_source_search_hint')
    fresh = navigation_guide(binding, x=7.)
    trusted = dict(navigation_guide(binding, identity='trusted'),
        origin='user_confirmed_visual_supervision', identity_confirmed=True,
        identity_authority={'reference':'accepted-source.json'})
    conflicting = navigation_guide(binding, identity='trusted', x=1.)
    geometry = SimpleNamespace(lookup=lambda name: {'id':name.replace('alias', 'trusted')})
    conflicting['id'] = 'alias'
    merged, report = merge_navigation_guides([old, trusted], [fresh, conflicting],
        binding=binding, geometry=geometry)
    assert merged[0] is trusted and len(merged) == 2
    np.testing.assert_array_equal(merged[1]['points_uv'], fresh['points_uv'])
    assert merged[1]['identity_confirmed'] is False
    assert report['refreshed_pending_path_ids'] == ['path']
    assert report['preserved_source_or_verified_path_ids'] == ['alias']
    assert not report['identity_confirmed_by_navigation']
    # No in-frame projected path also retires a stale pending projection.
    merged, _ = merge_navigation_guides([old, trusted], [], binding=binding)
    assert merged == [trusted]


@pytest.mark.parametrize('invalid', ['identity', 'authority', 'origin', 'source_frame'])
def test_additional_navigation_cannot_smuggle_identity_or_another_exposure(invalid):
    binding = source_binding()
    extra = navigation_guide(binding)
    if invalid == 'identity': extra['identity_confirmed'] = True
    if invalid == 'authority': extra['identity_authority'] = {'reference':'query-only'}
    if invalid == 'origin': extra['origin'] = 'user_confirmed_visual_supervision'
    if invalid == 'source_frame': extra['source_binding'] = source_binding(256)
    with pytest.raises(ValueError):
        merge_navigation_guides([], [extra], binding=binding)


def test_clipped_or_invalid_native_samples_do_not_bridge_a_query_segment():
    uv = np.array([[0.,0.], [4.,0.], [8.,0.], [12.,0.], [16.,0.], [20.,0.]])
    valid = np.array([True,True,False,True,True,True])
    indices, segments = _selected_indices(uv, valid, spacing=8., maximum_gap=96.)
    assert indices.tolist() == [0,1,3,5]
    assert segments.tolist() == [0,0,1,1]
    np.testing.assert_array_equal(uv[indices], [[0.,0.],[4.,0.],[12.,0.],[20.,0.]])


def test_source_query_coverage_keeps_independent_native_identity_context():
    from rinkspace.vision_association import associate
    scene=dict(path='native.json',sha256='c'*64)
    geometry=SimpleNamespace(refs=[scene],lookup=lambda name:dict(id=name,layer='ice'),bind=lambda rows:(rows,{}))
    binding=source_binding(256)
    source=dict(id='line',candidate_kind='saved_source_readout',points_uv=[[2.,4.],[2.,6.]],
        segment_ids=[0,0],source_binding=binding,source_candidates=[])
    projection=dict(navigation_guide(binding,identity='line',x=3.),candidate_kind='native_projection',segment_ids=[0,0],
        model_search_projection=dict(model_revision=scene['sha256'],native_pts=256,camera_revision='d'*64))
    dispatched=associate({'binding':binding},[source,projection],geometry)
    assert len(dispatched['queries'])==1
    query=dispatched['queries'][0]
    np.testing.assert_array_equal(query['points_uv'],source['points_uv'])
    np.testing.assert_array_equal(query['native_identity_navigation']['points_uv'],projection['points_uv'])
    assert query['identity_confirmed'] is False and query['identity_authority'] is None
    assert 'native_identity_navigation' not in source


@pytest.fixture(scope='module')
def native_navigation():
    library = os.environ.get('RINKSPACE_CORE_LIBRARY')
    seeds = os.environ.get('RINKSPACE_NAVIGATION_SEED_SET')
    if not library or not seeds:
        pytest.skip('explicit native library and exported current navigation seed set required')
    geometry = NativeSourceGeometry(ROOT/'events/ankara_2026/scene/field/model.json', library)
    scope = json.loads((ROOT/'events/ankara_2026/programmes/okada_mei_sp/rebuild_scope.json').read_text(encoding='utf-8-sig'))
    return NativeNavigation(geometry, seeds, scope=scope)


def native_binding(navigation, shot='s02'):
    camera = navigation.cameras[shot][0]
    return dict(source_sha256=navigation.source['sha256'], source_pts=camera['first_pts'],
        time_base=str(navigation.clock), shot_id=shot, native_hw=navigation.hw)


@pytest.mark.parametrize('shot', ['s02','s05','s07','s09'])
def test_current_native_motion_projects_real_paths_without_source_or_identity_claims(native_navigation, shot):
    navigation = native_navigation
    binding = native_binding(navigation, shot)
    guides, receipt = navigation.project(binding)
    assert guides and receipt['new_observations'] == 0
    assert receipt['current_native_path_count'] == len(navigation.geometry.samples['cuts'])
    unlocalized = navigation.geometry.samples['inventory']['unlocalized_print_instance_ids']
    assert receipt['unlocalized_print_instance_ids'] == unlocalized
    assert receipt['model']['scene_content_revision'] == navigation.scene_content_revision
    assert receipt['model']['scene_dependencies'] == navigation.geometry.refs
    camera, controls, _, _ = navigation.cameras[shot]
    t = (binding['source_pts']-camera['reference_pts'])*camera['time_base_num']/camera['time_base_den']
    for guide in guides:
        assert guide['origin'] == 'model_projection_search_hint'
        assert guide['identity_confirmed'] is False and guide['identity_authority'] is None
        assert 'actual_source_uv' not in guide and not guide['hint_is_source_measurement']
        assert not guide['model_search_projection']['image_alignment_checked']
        assert not any(guide['id'].startswith(identity+'::') for identity in unlocalized)
    chosen = [guide for guide in guides if guide['kind'] in ('ice_contact','board_top')] + guides[:1]
    for guide in chosen:
        projected = navigation.geometry.core.project_motion(camera['rig'], controls,
            camera['first_control_time'], camera['control_dt'], navigation.hw[0], t,
            navigation.xyz[guide['native_sample_indices']])
        assert np.all(projected['status'] == 0)
        np.testing.assert_array_equal(projected['uv'], np.asarray(guide['points_uv']))
        points = np.asarray(guide['points_uv'])
        assert np.all(points >= 0) and np.all(points <= np.array(navigation.hw[::-1])-1)
        assert len(guide['segment_ids']) == len(points)


@pytest.mark.parametrize('invalid', ['source_sha','clock','size','source_id','shot','pts_grid','pts_type','outside_domain'])
def test_projection_rejects_wrong_native_source_binding(native_navigation, invalid):
    binding = native_binding(native_navigation)
    if invalid == 'source_sha': binding['source_sha256'] = 'f'*64
    if invalid == 'clock': binding['time_base'] = '1/1000'
    if invalid == 'size': binding['native_hw'] = [720,1280]
    if invalid == 'source_id': binding['source_id'] = 'other-video'
    if invalid == 'shot': binding['shot_id'] = 's01'
    if invalid == 'pts_grid': binding['source_pts'] += 1
    if invalid == 'pts_type': binding['source_pts'] = float(binding['source_pts'])
    if invalid == 'outside_domain': binding['source_pts'] -= native_navigation.step
    with pytest.raises(ValueError): native_navigation.project(binding)


@pytest.mark.parametrize('dependency', ['geometry','camera'])
def test_projection_rejects_changed_dependency(native_navigation, tmp_path, monkeypatch, dependency):
    path = tmp_path/'dependency.json'
    path.write_text('{}',encoding='utf8')
    owner = native_navigation.geometry if dependency == 'geometry' else native_navigation
    monkeypatch.setattr(owner, 'refs', owner.refs+[file_ref(path)])
    path.write_text('{"changed":true}',encoding='utf8')
    with pytest.raises(ValueError, match='changed'):
        native_navigation.project(native_binding(native_navigation))


def test_additional_projection_reaches_real_reader_without_turning_off_reidentification(monkeypatch):
    torch = pytest.importorskip('torch')
    from rinkspace.vision import SourceRinkReader
    from rinkspace import vision_geometry, vision_near_board, vision_feedback
    monkeypatch.setattr(vision_near_board, 'configured_near_evidence', lambda: {})
    reader = SourceRinkReader.__new__(SourceRinkReader)
    reader.active_guides, reader.active_binding, reader.active_global_exclusion = [], None, None
    module = SimpleNamespace(device='cpu', classes=('background','ice_contact'), last_motion=None,
        model_usage=lambda: dict(sha256='b'*64,path='CPU-test-predictor'))
    def predict(rgb, binding, reset=False):
        module.current_image = torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float()/255
        module.current_prediction = dict(source_binding=binding, prediction_source_binding=binding,
            class_probabilities=torch.zeros((1,2,*rgb.shape[:2])),
            memory_update='shot_reset' if reset else 'image_aligned', new_observation=True,
            model_sha256='b'*64, source_clock_evidence={}, motion={})
        return module.current_prediction
    module.predict = predict
    module.reset = lambda: None
    reader.vision = module
    searches, queries, feedback_queries = [], [], []
    def search(rgb, binding, **kwargs):
        searches.append(binding['source_pts'])
        return [], dict(status='test-no-source-correspondence')
    reader.reidentifier = SimpleNamespace(person=SimpleNamespace(weights_sha256='c'*64),
        exclusions=lambda rgb,binding: (np.zeros(rgb.shape[:2],bool),[],dict(candidates=[])), search=search)
    reader.reidentification_active = False
    original_begin = vision_feedback.begin_read
    def begin(binding, guides):
        feedback_queries.append([guide['id'] for guide in guides])
        return original_begin(binding,guides)
    monkeypatch.setattr(vision_feedback, 'begin_read', begin)
    def read_rgb(image, probability, guides, **kwargs):
        guide, = guides
        queries.append(dict(id=guide['id'], uv=np.asarray(guide['points_uv']).tolist(),
            origin=guide['origin'], confirmed=guide['identity_confirmed']))
        # Actual source signal is deliberately far from either model query.
        x = int(torch.argmax(image[0,0,4]))
        return dict(readings=[dict(expected_id=guide['id'], physical_id=None, identity_authority=None,
            actual_source_uv=torch.tensor([[x,4.],[x,6.]]), source_indices=torch.tensor([0,1]))],metadata={})
    monkeypatch.setattr(vision_geometry,'guided_source_reading',read_rgb)
    monkeypatch.setattr(vision_geometry,'coherent_named_source_reading',lambda result:result)
    rgb = np.zeros((10,12,3),np.uint8);rgb[:,9,0]=255
    first = reader.read(rgb, source_binding(), guides_2d=[], reset=True, extract_source_components=False,
        additional_search_guides=[navigation_guide(source_binding(),x=2.)])
    second = reader.read(rgb, source_binding(256), extract_source_components=False,
        additional_search_guides=[navigation_guide(source_binding(256),x=6.)])
    assert searches == [0,256] and reader.reidentification_active
    assert feedback_queries[0] == ['path']
    assert all('path' in ids for ids in feedback_queries)
    assert [query['uv'] for query in queries] == [[[2.,4.],[2.,6.]], [[6.,4.],[6.,6.]]]
    assert all(query['origin']=='model_projection_search_hint' and not query['confirmed'] for query in queries)
    for result in (first,second):
        reading, = result['guided']['readings']
        assert reading['actual_source_uv'].tolist() == [[9.,4.],[9.,6.]]
        assert reading['physical_id'] is None and not result['hint_pixels_used_as_output']
        assert result['navigation_dispatch']['dispatched_path_count'] == 1
    assert second['navigation_dispatch']['refreshed_pending_path_ids'] == ['path']
    assert reader.active_guides[0]['identity_confirmed'] is False


def test_formal_source_read_request_dispatches_navigation_on_each_native_frame(tmp_path, monkeypatch):
    """Exercise the real request entry, keeping first-frame acceptance separate."""
    from fractions import Fraction
    import hashlib
    import sys
    from rinkspace import vision, vision_navigation
    from rinkspace.session.source_read import run_source_read
    video = tmp_path/'source.mp4'; video.write_bytes(b'fixture-original-source')
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    scope = dict(event_id='event',programme_id='programme',required_shot_ids=['one'],
        source=dict(id='video',sha256=digest,width=12,height=10,registered_path=str(video)),
        source_clock=dict(time_base_num=1,time_base_den=12800,native_pts_step=256),
        rig_bindings=dict(ids={'rig':'installation'}),
        shots=[dict(id='one',rig_id='rig',processing=dict(first_pts=0,stop_pts_exclusive=512))])
    scope_path = tmp_path/'scope.json';scope_path.write_text(json.dumps(scope),encoding='utf8')
    seed_path = tmp_path/'navigation.json';seed_path.write_text('{}',encoding='utf8')
    request = dict(programme_scope=str(scope_path),vision_config='stub-predictor.json',
        sequences=[dict(shot_id='one',first_pts=0,stop_pts_exclusive=512)],
        navigation=dict(seed_set='navigation.json',search_radius_px=24.,sample_spacing_px=8.))
    request_path=tmp_path/'request.json';request_path.write_text(json.dumps(request),encoding='utf8')
    image = np.zeros((10,12,3),np.uint8)
    frames = [SimpleNamespace(pts=pts,to_ndarray=lambda format: image.copy()) for pts in (0,256)]
    class Container:
        streams=SimpleNamespace(video=[SimpleNamespace(time_base=Fraction(1,12800))])
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def seek(self,*args,**kwargs): pass
        def decode(self,*args): return iter(frames)
    monkeypatch.setitem(sys.modules,'av',SimpleNamespace(open=lambda path:Container()))
    calls, projected = [], []
    class Reader:
        def __init__(self,*args,**kwargs):
            self.native_geometry=SimpleNamespace(refs=[],assert_unchanged=lambda:None)
            self.vision=SimpleNamespace(model_usage=lambda:{'fixture':True})
        def read(self,rgb,binding,**kwargs):
            calls.append(kwargs)
            assert kwargs['additional_search_guides'][0]['source_binding']==binding
            return dict(source_binding=binding,guided=dict(readings=[]),
                memory_update='shot_reset' if kwargs['reset'] else 'image_aligned',
                new_observation=True,source_feedback={},native_geometry={},navigation_dispatch={})
    class Navigation:
        def __init__(self,geometry,path,**kwargs):
            assert path==seed_path
            assert kwargs['scope']==scope
            self.refs=[file_ref(path)]
        def project(self,binding):
            projected.append(binding['source_pts'])
            return [navigation_guide(binding)],dict(query_point_count=2,new_observations=0)
        def assert_unchanged(self): pass
    monkeypatch.setattr(vision,'SourceRinkReader',Reader)
    monkeypatch.setattr(vision_navigation,'NativeNavigation',Navigation)
    output=tmp_path/'output'
    result=run_source_read(request_path,output)
    assert result['frames']==2 and projected==[0,256]
    assert calls[0]['guides_2d']==[] and calls[1]['guides_2d'] is None
    assert [row['reset'] for row in calls]==[True,False]
    for pts in (0,256):
        record=json.loads((output/f'one_{pts}.json').read_text())
        assert not record['accepted_reference_used_for_guides']
        assert record['native_navigation']['new_observations']==0
    receipt=json.loads((output/'result.json').read_text())
    assert receipt['navigation_refs']==[file_ref(seed_path)]


def test_material_context_reports_projection_navigation_without_authorizing_its_identity():
    from rinkspace.vision_material_reader import CurrentNativeSourceMaterialContext
    scene=dict(path='current-scene.json',sha256='c'*64)
    inventory=[dict(curve_id=name,feature_id=name,native_path_id='native/'+name,
        native_kind='ice_segment',operator='paint_center',material_signal='dark')
        for name in ('pending','confirmed')]
    binding=source_binding()
    context=CurrentNativeSourceMaterialContext(SimpleNamespace(refs=[scene]),inventory,binding)
    pending=dict(navigation_guide(binding,identity='native/pending'),kind='ice_paint',
        native_geometry=dict(scene=scene,path_id='native/pending'),
        model_search_projection=dict(model_revision=scene['sha256'],native_pts=binding['source_pts']))
    confirmed=dict(navigation_guide(binding,identity='native/confirmed'),kind='ice_paint',
        native_geometry=dict(scene=scene,path_id='native/confirmed'),
        origin='user_confirmed_visual_supervision',identity_confirmed=True,
        identity_authority=dict(kind='user_confirmed_visual_supervision',reference='exact-original-source'))
    context.prepare([pending,confirmed],binding)
    assert context.navigation_flags['model_projection_used_for_navigation'] is True
    assert context.navigation_flags['model_projection_used_for_identity_confirmation'] is False
    assert context.navigation_flags['model_projection_used_as_navigation_and_identity_context'] is True
    assert context._source_context_output_curve_allowed(inventory[0])
    assert pending['identity_confirmed'] is False and pending['identity_authority'] is None
    assert context._source_context_output_curve_allowed(inventory[1])
    assert context.authority['projected_navigation_path_ids']==['native/pending']
    assert context.authority['original_feature_identity_authorities']=={
        'confirmed':confirmed['identity_authority']}
    context.prepare([pending],binding)
    assert not context.navigation_flags['verified_source_guides_used_as_navigation']
    assert not context.authorized
    context.prepare([confirmed],binding)
    assert not context.navigation_flags['model_projection_used_for_navigation']
    assert context.navigation_flags['verified_source_guides_used_as_navigation']


def test_navigation_fills_unread_part_of_already_named_path():
    binding=source_binding()
    old=dict(navigation_guide(binding),origin='user_confirmed_visual_supervision',
        identity_confirmed=True,identity_authority={'reference':'accepted-source'},
        points_uv=[[2.,0.],[2.,10.]],segment_ids=[0,0])
    fresh=dict(navigation_guide(binding),points_uv=[[2.,0.],[2.,10.],[2.,20.],[2.,30.]],segment_ids=[0,0,0,0])
    merged,report=merge_navigation_guides([old],[fresh],binding=binding)
    assert merged[0] is old and len(merged)==2
    assert merged[1]['points_uv'][-1].tolist()==[2.,30.]
    assert merged[1]['identity_confirmed'] is False
    assert merged[1]['query_coverage']['preserved_existing_source_runs']
