from copy import deepcopy
from types import SimpleNamespace
import hashlib
import numpy as np
import pytest
from rinkspace.vision_candidate_admission import admit_material_candidates


def fixture():
    rgb=np.zeros((12,14,3),np.uint8)
    binding=dict(source_sha256='a'*64,source_pts=100,time_base='1/1000',shot_id='s01',native_hw=[12,14])
    definition=dict(id='ice.A:outer',layer='ice',measurement='contour')
    scene=dict(path='synthetic-scene',sha256='b'*64)
    def bind(guides):
        return [dict(g,native_geometry=dict(path_id=g['id'],scene=scene)) for g in guides],dict(unresolved=[])
    geometry=SimpleNamespace(assert_unchanged=lambda:None, refs=[scene],lookup=lambda id:definition if id=='ice.A:outer' else None,bind=bind)
    proposal=dict(id='shape-1',normals=np.tile([1.,0.],(4,1)),human_confirmed_instance_identity_initialized=True,
        human_confirmed_identity_source_binding=binding,source_identity_association=dict(status='observed_RGB_material_layer_under_human_named_parent'))
    row=dict(id='shape-1',physical_curve_id='ice.A:outer',source_binding=binding,physical_identity_verified=True,
        identity_authority=[dict(kind='saved_human_reference')],physical_role='filled_disc_boundary',native_material_role_matches=True,
        fitted_or_flow_pixels_used_as_measurements=False,actual_source_uv=np.array([[4.,3.],[4.,4.],[4.,7.]]),
        original_source_profile_indices=np.array([0,1,3]),source_quality_flags=dict(competing_same_sign_material_transition_peaks=np.zeros(3,bool)))
    packet=dict(source_binding=binding,source_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),guides=[proposal],source_shape_readings=[row])
    return rgb,binding,packet,geometry


def test_qualified_native_queries_preserve_gaps_and_require_current_rgb_reread():
    rgb,b,p,g=fixture();r=admit_material_candidates(rgb,b,p,g)
    query=r['accepted_current_query_guides'][0]
    assert query['segment_ids'].tolist()==[0,0,1]
    assert query['original_source_profile_indices'].tolist()==[0,1,3]
    assert r['actual_RGB_reread_required'] and not r['candidate_UV_used_as_final_measurements']
    assert query['identity_authority']['human_pixel_acceptance'] is False


@pytest.mark.parametrize('change', ['unknown','competing','wrong_role','wrong_native','duplicate'])
def test_unresolved_identity_and_competition_do_not_become_guides(change):
    rgb,b,p,g=fixture();row=p['source_shape_readings'][0]
    if change=='unknown':row['physical_identity_verified']=False
    if change=='competing':row['source_quality_flags']['competing_same_sign_material_transition_peaks'][:]=True
    if change=='wrong_role':row['physical_role']='painted_stroke_center_axis'
    if change=='wrong_native':row['physical_curve_id']='ice.absent'
    if change=='duplicate':p['source_shape_readings'].append(deepcopy(row))
    r=admit_material_candidates(rgb,b,p,g)
    assert not r['accepted_current_query_guides'] and r['pending_identity']


def test_ambiguous_profile_does_not_erase_separate_readable_material():
    rgb,b,p,g=fixture();row=p['source_shape_readings'][0]
    row['source_quality_flags']['competing_same_sign_material_transition_peaks'][0]=True
    original=row['actual_source_uv'].copy()
    result=admit_material_candidates(rgb,b,p,g)
    query=result['accepted_current_query_guides'][0]
    np.testing.assert_array_equal(query['points_uv'],original[1:])
    np.testing.assert_array_equal(row['actual_source_uv'],original)
    assert result['pending_identity'][0]['original_source_profile_indices']==[0]


def test_changed_rgb_or_hidden_candidate_cannot_be_admitted():
    rgb,b,p,g=fixture();other=rgb.copy();other[0,0]=1
    with pytest.raises(ValueError,match='RGB'):admit_material_candidates(other,b,p,g)
    excluded=np.zeros(rgb.shape[:2],bool);excluded[3,4]=True
    r=admit_material_candidates(rgb,b,p,g,exclusion=excluded)
    assert not r['accepted_current_query_guides'] and r['rejected']


def test_existing_path_only_covers_its_measured_visible_runs():
    rgb,b,p,g=fixture()
    uv=np.c_[np.full(9,4.),np.arange(2.,11.)]
    p['guides'][0]['normals']=np.tile([1.,0.],(len(uv),1))
    row=p['source_shape_readings'][0]
    row.update(actual_source_uv=uv,original_source_profile_indices=np.arange(len(uv)),
        source_quality_flags=dict(competing_same_sign_material_transition_peaks=np.zeros(len(uv),bool)))
    old=dict(id='ice.A:outer',native_geometry=dict(path_id='ice.A:outer'),
        points_uv=uv,segment_ids=np.zeros(len(uv),int),last_measured_query_indices=[0,1,2])
    result=admit_material_candidates(rgb,b,p,g,existing_guides=[old])
    query=result['accepted_current_query_guides'][0]
    assert query['original_source_profile_indices'][-1]==8
    assert np.array_equal(query['points_uv'],uv[query['original_source_profile_indices']])
    assert query['query_coverage']['preserved_existing_source_runs']
    assert old['last_measured_query_indices']==[0,1,2]
    old['last_measured_query_indices']=list(range(len(uv)))
    result=admit_material_candidates(rgb,b,p,g,existing_guides=[old])
    assert not result['accepted_current_query_guides']
    assert result['rejected'][0]['reason']=='already_measured_source_query_support'


def test_reader_remeasures_admitted_candidate_and_updates_next_frame(monkeypatch):
    import torch
    from test_vision_tracking_frame import reader_and_spies
    rgb,b,p,g=fixture();rgb[:,9,0]=255;p['source_RGB_sha256']=hashlib.sha256(rgb.tobytes()).hexdigest()
    reader,calls=reader_and_spies(monkeypatch);reader.vision.last_motion=None
    reader.candidate_admission_enabled=True
    g.check_output=lambda guides,readings: dict(checked_count=len(readings['readings']))
    reader.native_geometry=g
    transactions=[]
    def refine(rgb,binding,guides,guided,**kw):
        transactions.append((binding['source_pts'],len(guides)))
        return {}
    reader.material_reader=SimpleNamespace(refine_ice=refine,refine_print=lambda *a,**k:{},read_unassigned_material=lambda *a,**k:p)
    result=reader.read(rgb,b,guides_2d=[],reset=True,extract_source_components=False)
    admitted=result['native_material']['candidate_admission']['accepted_current_query_guides']
    assert len(admitted)==1 and calls['guided']==1
    assert transactions==[(100,0),(100,1)]
    # The deterministic test precision reader locates the actual red RGB stripe.
    # Candidate navigation was x=4; it must not be copied as the measurement.
    assert result['guided']['readings'][0]['actual_source_uv'].tolist()==[[9.,4.],[9.,6.]]
    assert reader.active_guides[0]['points_uv'][:2].tolist()==[[9.,4.],[9.,6.]]
    assert result['native_material']['output_scope']['checked_count']==1
    assert result['native_material']['candidate_admission_enabled'] is True


@pytest.mark.parametrize("setting", [None, False, True, "true", 1])
def test_candidate_admission_requires_explicit_true_config(monkeypatch, tmp_path, setting):
    from rinkspace import vision
    config = dict(final_model="unused-model", flow_weights="unused-flow", device="cpu")
    if setting is not None:
        config["enable_candidate_admission"] = setting
    monkeypatch.setattr(vision, "configure", lambda _: (config, tmp_path))
    monkeypatch.setattr(vision, "RinkVisionModule", lambda *a, **k: SimpleNamespace(device="cpu"))
    reader = vision.SourceRinkReader("unused-config")
    assert reader.candidate_admission_enabled is (setting is True)


def test_default_reader_preserves_queries_without_activating_candidate_admission(monkeypatch):
    from rinkspace import vision_candidate_admission
    from test_vision_tracking_frame import reader_and_spies
    rgb, binding, packet, geometry = fixture()
    reader, calls = reader_and_spies(monkeypatch)
    reader.vision.last_motion = None
    geometry.check_output = lambda guides, guided: dict(checked_count=len(guided["readings"]))
    reader.native_geometry = geometry
    transactions = []
    def refine(rgb, binding, guides, guided, **kwargs):
        transactions.append(len(guides))
        return {}
    reader.material_reader = SimpleNamespace(
        refine_ice=refine, refine_print=lambda *a, **k: {},
        read_unassigned_material=lambda *a, **k: packet)
    def unexpected_admission(*args, **kwargs):
        pytest.fail("Unvalidated F30 admission must not run in the default reader")
    monkeypatch.setattr(vision_candidate_admission, "admit_material_candidates", unexpected_admission)
    result = reader.read(rgb, binding, guides_2d=[], reset=True, extract_source_components=False)
    assert transactions == [0]
    assert calls["guided"] == 0
    assert result["guided"]["readings"] == []
    assert reader.active_guides == []
    assert result["native_material"]["candidate_admission_enabled"] is False
    assert "candidate_admission" not in result["native_material"]
    assert result["native_material"]["unassigned_material_search"] is packet
