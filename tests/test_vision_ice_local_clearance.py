import hashlib

import cv2
import numpy as np
import pytest

from rinkspace.vision_ice_local_clearance import KnownIceMaterialBoxClearance


def material(pts):
    rgb=np.full((340,340,3),220,np.uint8)
    cv2.circle(rgb,(170,170),100,(100,)*3,-1)
    uv=np.array([[270.,170.],[170.,270.]],np.float32)
    m=dict(source_binding=dict(source_sha256='a'*64,source_pts=pts,time_base='1/1000',shot_id='one'),
        decoded_native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        candidate_uv=uv,candidate_sides=np.repeat(uv[:,None],2,axis=1),candidate_query=np.arange(2),
        candidate_flags=np.full(2,10,np.uint8),query_curve=np.zeros(2,np.int16),query_normal=np.array([[1.,0.],[0.,1.]]))
    return rgb,m


def setup(records=()):
    inventory=[dict(curve_id='R4:outer',feature_id='R4',operator='material_interface',material_signal='dark',material_transition='dark_to_light')]
    return KnownIceMaterialBoxClearance(inventory,source_sha256='a'*64,time_base='1/1000',seed_registry=records)


def consume(reader,rgb,m,person):
    box=np.ones(rgb.shape[:2],bool)
    return reader.consume(rgb,m,box,source_ice_scope_mask=~box,primary_box_mask=box,
        measured_person_mask=person,other_person_boxes_mask=np.zeros_like(person))


def test_unidentified_material_is_not_released_by_segmentation_background():
    reader=setup();first=material(10);last=material(20)
    clear=np.zeros(first[0].shape[:2],bool)
    consume(reader,*first,clear)
    result=consume(reader,*last,clear)
    assert not result['candidate_local_source_clearance']
    assert result['exclusion_mask'].all() and not result['source_ice_scope_mask'].any()
    assert np.array_equal(result['measurement']['candidate_flags'],last[1]['candidate_flags'])


def test_known_source_identity_cannot_clear_measured_person_pixels():
    seed=dict(curve_id='R4:outer',actual_pts=20,source_sha256='a'*64,time_base='1/1000',shot_id='one',
        source_material_class='static_painted_ice',surface_and_physical_identity_confirmed=True,
        authority=dict(path='actual-original-source-identity.json',sha256='b'*64),navigation_uv=[[270.,170.],[170.,270.]])
    reader=setup([seed]);first=material(10);last=material(20)
    clear=np.zeros(first[0].shape[:2],bool)
    consume(reader,*first,clear)
    result=consume(reader,*last,~clear)
    assert not result['candidate_local_source_clearance'] and result['exclusion_mask'].all()


def test_local_clearance_rejects_another_source_image():
    reader=setup();rgb,m=material(10)
    rgb=rgb.copy();rgb[0,0]=0
    with pytest.raises(ValueError,match='exact original RGB'):
        consume(reader,rgb,m,np.zeros(rgb.shape[:2],bool))
