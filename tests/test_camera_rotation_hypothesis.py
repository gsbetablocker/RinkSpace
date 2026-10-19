"""Scientific separation checks for the isolated off-plane camera experiment."""
import cv2
import numpy as np
import pytest
pytest.importorskip("scipy", reason="isolated camera hypothesis uses the optional advanced dependencies")
from rinkspace.camera_rotation_hypothesis import CameraRotationZoomHypothesis, source_background_mask


def sample(i):
    return dict(sample_index=i,source_frame_index=i,source_time_s=i/50,pts=i,time_base="1/50")


def images():
    rng=np.random.default_rng(41)
    gray=cv2.GaussianBlur(rng.integers(0,256,(270,480),dtype=np.uint8),(3,3),.6)
    first=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    k=np.array([[480.,0,240],[0,480.,135],[0,0,1]])
    h=k@cv2.Rodrigues(np.array([-.001,.004,.001]))[0]@np.linalg.inv(k)
    second=cv2.warpPerspective(first,np.linalg.inv(h),(480,270))
    mask=np.zeros((270,480),np.uint8);mask[35:170,35:445]=255
    return first,second,mask,h


def test_camera_prediction_is_labelled_and_recovers_synthetic_rotation():
    a,b,mask,h=images()
    tracker=CameraRotationZoomHypothesis(480.)
    first=tracker.push(a,sample(0),mask);second=tracker.push(b,sample(1),mask)
    assert first['reference_frame_only']
    assert second['status']=='candidate'
    assert second['training_plane']=='unverified_off_plane_background'
    assert second['ice_plane_mapping_established'] is False
    assert 'H_reference_from_current' not in second
    got=np.asarray(second['H_scene_reference_from_current'])
    probes=np.array([[100,80,1],[350,100,1],[240,160,1.]])
    expected=probes@h.T;expected=expected[:,:2]/expected[:,2:]
    actual=probes@got.T;actual=actual[:,:2]/actual[:,2:]
    assert np.max(np.linalg.norm(actual-expected,axis=1))<.5
    assert second['jacobian_condition']<1e6


def test_all_outside_mask_values_cannot_influence_camera_estimate():
    a,b,mask,_=images();rng=np.random.default_rng(7)
    alternate=[]
    for im in (a,b):
        changed=im.copy();changed[mask==0]=rng.integers(0,256,(int((mask==0).sum()),3),dtype=np.uint8)
        alternate.append(changed)
    runs=[]
    for aa,bb in [(a,b),alternate]:
        tracker=CameraRotationZoomHypothesis(480.)
        tracker.push(aa,sample(0),mask);runs.append(tracker.push(bb,sample(1),mask))
    assert runs[0]['status']=='candidate'
    assert runs[0]['H_scene_reference_from_current']==runs[1]['H_scene_reference_from_current']
    assert runs[0]['fit_current_xy']==runs[1]['fit_current_xy']


def test_source_mask_cannot_consume_ice_or_person_pixels():
    im=np.full((300,480,3),180,np.uint8)
    cv2.line(im,(0,110),(479,110),(220,80,10),6)
    correction={'include_polygons':[[[0,220],[479,220],[479,299],[0,299]]],
                'exclude_polygons':[[[0,0],[479,0],[479,219],[0,219]],[[0,0],[100,0],[100,80],[0,80]]]}
    people=[dict(expanded_xyxy=[230,80,270,230])]
    first=source_background_mask(im,correction,people)
    changed=im.copy();forbidden=(first['ice_guard']>0)|(first['people_and_graphics']>0)
    changed[forbidden]=np.random.default_rng(4).integers(0,256,(int(forbidden.sum()),3),dtype=np.uint8)
    second=source_background_mask(changed,correction,people)
    assert np.count_nonzero(first['training'])>0
    assert not np.any(first['training'][forbidden])
    assert np.array_equal(first['training'],second['training'])


def test_textureless_background_stays_unavailable():
    im=np.full((270,480,3),150,np.uint8);mask=np.zeros((270,480),np.uint8);mask[50:150,50:430]=255
    row=CameraRotationZoomHypothesis(480.).push(im,sample(0),mask)
    assert row['status']=='unavailable'
    assert row['H_scene_reference_from_current'] is None


def test_explicit_building_polygons_are_not_an_inverted_ice_mask():
    from rinkspace.camera_rotation_hypothesis import source_building_mask
    im=np.full((300,480,3),180,np.uint8)
    correction={'include_polygons':[[[0,220],[479,220],[479,299],[0,299]]],
                'exclude_polygons':[[[0,0],[479,0],[479,219],[0,219]]]}
    annotation={'include_polygons':[[[100,20],[400,20],[400,100],[100,100]]],
                'exclude_polygons':[[[230,20],[260,20],[260,100],[230,100]]],
                'review_status':'manual_source_only_keyframe'}
    masks=source_building_mask(im,correction,[],annotation)
    assert masks['training'][50,120]>0
    assert masks['training'][50,240]==0
    assert masks['training'][150,120]==0
    assert masks['training'][250,120]==0
    assert masks['annotation_status']=='manual_source_only_keyframe'


def test_zoom_is_carried_into_new_observed_keyframe_and_remains_rotation():
    a,_,mask,_=images();tracker=CameraRotationZoomHypothesis(480.)
    tracker.push(a,sample(0),mask)
    kr=np.array([[480.,0,240],[0,480.,135],[0,0,1.]])
    for i,f,angle in [(20,490.,.004),(21,492.,.005)]:
        kc=np.array([[f,0,240],[0,f,135],[0,0,1.]])
        h=kr@cv2.Rodrigues(np.array([-.001,angle,.001]))[0]@np.linalg.inv(kc)
        im=cv2.warpPerspective(a,np.linalg.inv(h),(480,270))
        row=tracker.push(im,sample(i),mask)
        assert row['status']=='candidate'
        if i==20:
            assert tracker.engine.keys[-1]['sample_index']==20
            assert tracker.engine.keys[-1]['focal_px']==row['current_focal_px']
        got_k=np.array([[row['current_focal_px'],0,240],[0,row['current_focal_px'],135],[0,0,1.]])
        rotation=np.linalg.inv(kr)@np.asarray(row['H_scene_reference_from_current'])@got_k
        rotation/=np.cbrt(np.linalg.det(rotation))
        assert np.max(np.abs(rotation.T@rotation-np.eye(3)))<1e-8
        assert abs(row['current_focal_px']-f)<2.


def test_sift_reobserves_large_motion_without_reading_excluded_pixels():
    a,_,_,_=images();mask=np.zeros(a.shape[:2],np.uint8);mask[25:245,25:455]=255
    kr=np.array([[480.,0,240],[0,480.,135],[0,0,1.]])
    h=kr@cv2.Rodrigues(np.array([0.,.12,0.]))[0]@np.linalg.inv(kr)
    b=cv2.warpPerspective(a,np.linalg.inv(h),(480,270))
    results=[]
    for altered in [False,True]:
        tracker=CameraRotationZoomHypothesis(480.,sift_reobservation=True)
        rng=np.random.default_rng(92)
        for i,im in enumerate([a,b]):
            used=im.copy()
            if altered:used[mask==0]=rng.integers(0,256,(int(np.count_nonzero(mask==0)),3),dtype=np.uint8)
            row=tracker.push(used,sample(i),mask)
        results.append(row)
    assert results[0]['status']=='candidate'
    assert 'SIFT' in results[0]['pair_algorithm']
    assert results[0]['H_scene_reference_from_current']==results[1]['H_scene_reference_from_current']
    assert results[0]['fit_current_xy']==results[1]['fit_current_xy']
