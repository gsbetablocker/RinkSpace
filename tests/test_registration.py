import numpy as np
from rinkspace.registration import masked_pair


def test_empty_ice_texture_reports_unavailable():
    a=np.full((200,300,3),240,np.uint8);m=np.ones((200,300),np.uint8)*255
    r=masked_pair(a,a,m,m)
    assert r['status']=='unavailable'
    assert not r['map_usable_for_metric_output']


def test_lk_textureless_unavailable():
    from rinkspace.registration import adjacent_lk
    image=np.zeros((160,240,3),np.uint8);mask=np.full(image.shape[:2],255,np.uint8)
    result=adjacent_lk(image,image,mask,mask)
    assert result['status']=='unavailable'
    assert result['independent_validation'] is False


def test_lk_known_translation_candidate_only():
    import cv2
    from rinkspace.registration import adjacent_lk
    from rinkspace.geometry import project
    rng=np.random.default_rng(4)
    image=cv2.GaussianBlur(rng.integers(0,256,(400,640,3),dtype=np.uint8),(5,5),0)
    translated=cv2.warpAffine(image,np.array([[1,0,4],[0,1,3]],np.float32),(640,400))
    mask=np.zeros(image.shape[:2],np.uint8);mask[30:-30,30:-30]=255
    result=adjacent_lk(image,translated,mask,mask)
    assert result['status']=='candidate'
    points=np.array([[100,100],[200,250],[530,320]],float)
    back=project(result['H_reference_from_current'],points+np.array([4,3]))
    assert np.max(np.linalg.norm(back-points,axis=1))<.4
    assert not result['map_usable_for_metric_output']
    assert not result['independent_validation']
