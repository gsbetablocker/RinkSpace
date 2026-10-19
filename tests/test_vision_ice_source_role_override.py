import hashlib

import cv2
import numpy as np
import pytest

from rinkspace.vision_ice_source_role_override import remeasure_confirmed_source_roles
from rinkspace.vision_ice_nested_material_identity import _source_nested_profile


def source_measurement():
    rgb=np.full((180,240,3),210,np.uint8)
    rgb[78:103]=[210,75,75]
    rgb=cv2.GaussianBlur(rgb,(3,3),.6)
    uv=np.array([[60.,89.],[120.,89.],[180.,89.]],np.float32)
    measurement=dict(source_binding=dict(source_sha256='a'*64,source_pts=10,time_base='1/1000',shot_id='one'),
        decoded_native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        candidate_uv=uv,candidate_sides=np.stack([uv-[0,12.5],uv+[0,12.5]],1).astype(np.float32),
        candidate_query=np.arange(3,dtype=np.int32),candidate_flags=np.full(3,8,np.uint8),
        query_uv=uv,query_normal=np.tile([0.,1.],(3,1)).astype(np.float32),query_interval=np.tile([-50.,50.],(3,1)),
        query_curve=np.zeros(3,np.int16))
    inventory=[dict(curve_id='goal_crease_left:boundary',operator='unresolved',material_signal='dark',material_transition=None)]
    return rgb,measurement,inventory


def test_current_red_role_requires_new_actual_sides_and_keeps_old_dark_invalid(tmp_path):
    evidence=tmp_path/'source-role.json';evidence.write_text('actual source role authority')
    overrides={'goal_crease_left:boundary':dict(confirmed_measurement_operator='paint_center',confirmed_material_signal='red',
        source_role_evidence=dict(path=str(evidence),sha256=hashlib.sha256(evidence.read_bytes()).hexdigest()))}
    rgb,measurement,inventory=source_measurement()
    result,effective=remeasure_confirmed_source_roles(rgb,measurement,np.zeros(rgb.shape[:2],bool),inventory,overrides)
    assert np.all(result['candidate_flags'][:3]&1)
    assert np.allclose(result['candidate_sides'][3:].mean(1),result['candidate_uv'][3:],atol=1e-5)
    assert np.allclose(result['candidate_uv'][3:,1],90.,atol=1.)
    assert np.isfinite(result['candidate_side_quantiles_uv'][3:]).all()
    assert effective[0]['material_signal']=='red' and inventory[0]['operator']=='unresolved'
    assert not (measurement['candidate_flags']&1).any()
    evidence.write_text('modified authority')
    with pytest.raises(ValueError,match='authenticated'):
        remeasure_confirmed_source_roles(rgb,measurement,np.zeros(rgb.shape[:2],bool),inventory,overrides)


def test_actual_nested_material_profile_stops_at_an_occluding_person():
    rgb=np.full((340,340,3),220,np.uint8)
    for radius,colour in [(100,100),(75,220),(50,100),(25,220)]:
        cv2.circle(rgb,(170,170),radius,(colour,)*3,-1)
    rgb=cv2.GaussianBlur(rgb,(3,3),.6)
    mask=np.zeros(rgb.shape[:2],bool)
    chain=_source_nested_profile(rgb,mask,np.array([270.,170.]),np.array([-1.,0.]))
    assert [v['sign'] for v in chain]==[-1,1,-1,1]
    assert np.allclose([v['offset'] for v in chain],[0.,25.,50.,75.],atol=2.)
    mask[:,185:215]=True
    partial=_source_nested_profile(rgb,mask,np.array([270.,170.]),np.array([-1.,0.]))
    assert len(partial)<=2  # Missing source is never bridged to center layers.
