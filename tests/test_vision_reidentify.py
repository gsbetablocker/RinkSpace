"""The cut entry must reacquire identities, then reread current RGB via tracking."""
from types import SimpleNamespace
import numpy as np
import pytest
from rinkspace.vision_native import NativeSourceGeometry
from test_vision_tracking_frame import reader_and_spies, source_rgb, binding, guide


def test_native_source_alias_resolves_only_one_installed_subpath():
    geometry = NativeSourceGeometry.__new__(NativeSourceGeometry)
    row = dict(id='print_instance/Ad7::Ad7.top_edge/path0')
    geometry.paths = {'Ad7.top_edge/path0': row}
    assert geometry.lookup('Ad7.top_edge') is row
    assert geometry.lookup('Ad7') is None
    geometry.paths['Ad7.top_edge/path1'] = dict(id='other')
    assert geometry.lookup('Ad7.top_edge') is None


def test_cut_entry_uses_reidentification_then_current_RGB_within_shot(monkeypatch):
    reader, calls = reader_and_spies(monkeypatch)
    original_predict = reader.vision.predict
    reader.vision.last_motion = None
    def predict(rgb, current, reset=False):
        result = original_predict(rgb, current, reset)
        result['memory_update'] = 'shot_reset' if reset else 'image_aligned'
        return result
    reader.vision.predict = predict
    searches = []
    def search(rgb, current, **kwargs):
        searches.append((current['shot_id'], kwargs['reset']))
        recovered = {**guide(), 'id':'Ad7.top_edge/path0', 'kind':'ad_boundary',
            'source_binding':current, 'identity_authority':{'kind':'test_authenticated_source_identity'}}
        return ([recovered] if kwargs['reset'] else []), {'status':'reacquired' if kwargs['reset'] else 'tracking'}
    reader.reidentifier = SimpleNamespace(search=search,
        exclusions=lambda rgb,b:(np.zeros(rgb.shape[:2],bool),[],{'candidates':[]}),
        person=SimpleNamespace(weights_sha256='c'*64))
    reader.reidentification_active = False
    reader.active_guides = [{**guide(), 'id':'old_shot_identity'}]
    first_binding = {**binding(256),'shot_id':'two'}
    first = reader.read(source_rgb(8), first_binding, guides_2d=[], reset=True, extract_source_components=False)
    assert [g['id'] for g in reader.active_guides] == ['Ad7.top_edge/path0']
    second = reader.read(source_rgb(9), {**binding(512),'shot_id':'two'}, extract_source_components=False)
    assert searches == [('two',True),('two',False)]
    assert first['guided']['readings'][0]['actual_source_uv'].tolist() == [[8.,4.],[8.,6.]]
    assert second['guided']['readings'][0]['actual_source_uv'].tolist() == [[9.,4.],[9.,6.]]
    assert second['guided']['readings'][0]['temporal_hint_previous_source_binding']['source_pts'] == 256
    assert calls == {'global':0,'guided':2}
    assert not first['hint_pixels_used_as_output'] and not second['hint_pixels_used_as_output']


def test_explicit_accepted_source_is_not_replaced_by_reidentification(monkeypatch):
    reader, _ = reader_and_spies(monkeypatch)
    def forbidden(*args,**kwargs):
        pytest.fail('Accepted explicit guide must not be replaced by automatic reacquisition')
    exclusions=[]
    def current_exclusions(rgb,current):
        exclusions.append(current['source_pts'])
        return np.zeros(rgb.shape[:2],bool),[],dict(candidates=[])
    reader.reidentifier = SimpleNamespace(exclusions=current_exclusions, search=forbidden)
    result=reader.read(source_rgb(8),binding(0),guides_2d=[guide()],reset=True,extract_source_components=False)
    assert result['source_reidentification'] is None
    assert exclusions == [0]  # Accepted identity never bypasses current occlusion detection.


def test_reidentifier_rejects_foreign_source_before_matching():
    from rinkspace.vision_reidentify import AcceptedSourceReidentifier
    reader=AcceptedSourceReidentifier.__new__(AcceptedSourceReidentifier)
    reader.spec={'source_sha256':'a'*64}
    with pytest.raises(ValueError,match='different source'):
        reader.search(np.zeros((10,12,3),np.uint8),{**binding(0),'source_sha256':'b'*64})


def test_actual_entry_uses_current_SAM_pixels_not_person_rectangle(monkeypatch):
    import torch
    from rinkspace.vision_reidentify import AcceptedSourceReidentifier
    import rinkspace.vision_source_topology as topology
    from rinkspace import vision_geometry
    reader,_=reader_and_spies(monkeypatch)
    original_predict=reader.vision.predict
    def predict(rgb,current,reset=False):
        result=original_predict(rgb,current,reset)
        result['memory_update']='shot_reset' if reset else 'image_aligned'
        return result
    reader.vision.predict=predict;reader.vision.last_motion=None
    detector=AcceptedSourceReidentifier.__new__(AcceptedSourceReidentifier)
    detector.person=SimpleNamespace(read=lambda rgb,b:[dict(box_xyxy=[1,1,11,9],source_binding=b)])
    monkeypatch.setattr(topology,'source_graphic_rectangle_candidates',
        lambda rgb,b:dict(candidates=[dict(box_xyxy=[10,8,11,9],source_binding=b)]))
    reader.reidentifier=SimpleNamespace(exclusions=detector.exclusions,
        matcher=SimpleNamespace(read=lambda *a,**k:dict(prompts=[])),
        accepted_ad_anchors=lambda *a:[],advertisement_guides=lambda *a,**k:([],[]))
    reader.config=dict(source_instances=True);reader.reidentification_active=False
    def instances(rgb,b,**kwargs):
        person=torch.zeros((10,12),dtype=torch.bool)
        person[2:8,2:5] = b['source_pts']==0
        person[2:8,7:10] = b['source_pts']!=0
        graphic=torch.zeros_like(person);graphic[8,10]=True
        return dict(source_binding=b,reviewed_source_exclusion_mask=torch.zeros_like(person),
            instances=[dict(object_id=1,kind='occluder',region_mask=person,source_binding=b),
                       dict(object_id=2,kind='broadcast_overlay',region_mask=graphic,source_binding=b)])
    reader.instance_provider=SimpleNamespace(read_auto=instances,device='cpu',reread_rgb=False)
    def read(image,probability,guides,**kwargs):
        points=torch.tensor(guides[0]['points_uv']) if not torch.is_tensor(guides[0]['points_uv']) else guides[0]['points_uv']
        xy=points.long();mask=guides[0]['source_exclusion_mask'][0,0]
        good=(~mask[xy[:,1],xy[:,0]])&(image[0,0,xy[:,1],xy[:,0]]>.5)
        return dict(readings=[dict(actual_source_uv=points[good],source_indices=torch.where(good)[0])],metadata={})
    monkeypatch.setattr(vision_geometry,'guided_source_reading',read)
    source_guide=dict(id='Ad2.print.G',kind='ad_boundary',points_uv=[[8.,4.],[3.,4.],[10.,8.]])
    first=reader.read(source_rgb(8),binding(0),guides_2d=[source_guide],reset=True,extract_source_components=False)
    second=reader.read(source_rgb(3),binding(256),extract_source_components=False)
    assert first['guided']['readings'][0]['actual_source_uv'].tolist()==[[8.,4.]]
    assert second['guided']['readings'][0]['actual_source_uv'].tolist()==[[3.,4.]]
    assert not first['source_instances']['reviewed_source_exclusion_mask'].any()
    assert first['source_instances']['current_source_exclusion_mask'][4,3]
    assert not first['source_instances']['current_source_exclusion_mask'][4,8]
    assert second['source_instances']['current_source_exclusion_mask'][8,10]
    assert not second['source_instances']['current_source_pixel_exclusion_evidence']['person_detector_rectangles_used_as_exclusion']


def test_current_SAM_exclusion_refuses_another_exposure():
    import torch
    from rinkspace.vision_review import current_source_pixel_exclusion
    packet=dict(source_binding=binding(0),instances=[dict(kind='occluder',source_binding=binding(256),
        region_mask=torch.ones((10,12),dtype=torch.bool))])
    with pytest.raises(ValueError,match='source frames'):
        current_source_pixel_exclusion(packet,binding(0),native_hw=(10,12),device='cpu')
