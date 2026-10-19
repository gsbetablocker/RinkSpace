import cv2
import numpy as np
from rinkspace.rotation_photometric import estimate_rotation, camera_h, RotationPhotometricRegistrar
from rinkspace.geometry import project


def case():
    rng=np.random.default_rng(814)
    ref=cv2.GaussianBlur(rng.uniform(.2,.8,(180,260)),(7,7),1.3)
    y,x=np.indices(ref.shape)
    ref+=.04*np.sin(x/15)+.03*np.cos(y/19)
    h=camera_h(ref.shape,260,264,cv2.Rodrigues(np.array([.003,-.009,.002]))[0])
    cur=cv2.warpPerspective(ref,np.linalg.inv(h),(260,180))
    mask=np.full(ref.shape,255,np.uint8)
    mask[:8]=mask[-8:]=0;mask[50:100,100:140]=0
    return ref,cur,mask,h


def test_known_camera_motion_and_zoom_recovered_without_shear():
    ref,cur,mask,truth=case()
    r=estimate_rotation(ref,cur,mask,mask,reference_focal=260,initial_current_focal=260,initial_rotation=np.eye(3))
    assert r['status']=='candidate',r['reason']
    points=[[25,25],[230,25],[230,150],[25,150]]
    assert np.max(np.linalg.norm(project(r['H_reference_from_current'],points)-project(truth,points),axis=1))<.15
    assert abs(r['current_focal_px']-264)<.3
    rotation=np.asarray(r['rotation_reference_from_current'])
    np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-10)


def test_excluded_pixels_cannot_change_any_pose_or_iteration():
    ref,cur,mask,_=case()
    args=dict(reference_focal=260,initial_current_focal=260,initial_rotation=np.eye(3))
    a=estimate_rotation(ref,cur,mask,mask,**args)
    ref[mask==0]=np.nan;cur[mask==0]=999999
    b=estimate_rotation(ref,cur,mask,mask,**args)
    assert a==b


def test_uniform_ice_and_explicit_cut_do_not_fabricate_compensation():
    ref,cur,mask,_=case()
    raw=np.uint8(ref*255)
    tracker=RotationPhotometricRegistrar()
    masks=dict(usable=mask,mask_status='synthetic',ice_fraction=1.,usable_fraction=.8)
    def sample(i):return dict(sample_index=i,source_frame_index=i,pts=i,time_base='1/50',source_time_s=i/50)
    a=tracker.push(raw,sample(0),masks)
    b=tracker.push(np.full(raw.shape,150,np.uint8),sample(1),masks)
    c=tracker.push(raw,sample(2),masks,force_break=True)
    assert a['reference_frame_only'] and c['reference_frame_only']
    assert b['H_reference_from_current'] is None
    assert a['graph_component_id']!=c['graph_component_id']
