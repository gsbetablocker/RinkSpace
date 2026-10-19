"""Synthetic integration contracts; no real-rink accuracy claim."""
import copy
from fractions import Fraction
import numpy as np
import pytest
from rinkspace.geometry import project
from rinkspace.rink_reference import (RinkReference, compose_model_from_image,
    validate_frame_map, project_model_segments, check_source_group_split)


def model(level='projective', unit='reference_px'):
    return RinkReference('synthetic-rink', 'test-v1', level, unit,
                         'synthetic origin and orientation', ('synthetic_geometry',), ('synthetic_scale',))


def record():
    return {'model_id': 'synthetic-rink', 'model_revision': 'test-v1',
            'source_sha256': 'a'*64, 'pts': 130, 'time_base': '1/50',
            'matrix_direction': 'H_model_from_image', 'image_domain': 'undistorted_px',
            'status': 'observed', 'anchor_id': 'synthetic-anchor',
            'H_model_from_image': np.eye(3).tolist()}


def test_legacy_bridge_with_different_scales_and_crops():
    c = np.array([[1.7,.2,30],[.1,1.2,8],[.001,.002,1.]])
    local = np.array([[1.1,.03,10],[-.05,.9,20],[.001,0,1.]])
    cur = np.array([[.5,0,-20],[0,.4,-8],[0,0,1.]])
    anchor = np.array([[.25,0,-4],[0,.3,-2],[0,0,1.]])
    xy = np.array([[30.,40.],[100.,70.],[80.,100.]])
    sequential = project(c,project(np.linalg.inv(anchor),project(local,project(cur,xy))))
    combined = compose_model_from_image(c,local,cur,anchor)
    assert np.allclose(project(combined,xy),sequential)
    assert np.allclose(project(np.linalg.inv(combined),sequential),xy)


def test_inverse_overlay_direction():
    r=record();r['H_model_from_image']=[[.5,0,10],[0,.5,20],[0,0,1]]
    result=project_model_segments(model(),r,[[[10,20],[20,20]]],(100,100))
    assert np.allclose(result['segments_image'],[[[0,0],[20,0]]])


def test_viewport_clipping_and_outside_rejection():
    result=project_model_segments(model(),record(),[[[-20,20],[200,20]],[[-20,-30],[-10,-5]]],(100,50))
    assert np.allclose(result['segments_image'],[[[0,20],[99,20]]])
    assert result['rejected_viewport_count']==1
    assert result['requires_ice_support_and_foreground_clipping']


def test_horizon_rejected_instead_of_connecting_infinity():
    h_image=np.array([[1.,0,0],[0,1.,0],[1.,0,-5.]])
    r=record();r['H_model_from_image']=np.linalg.inv(h_image).tolist()
    result=project_model_segments(model(),r,[[[4.,1.],[6.,1.]]],(100,100))
    assert result['segments_image']==[] and result['rejected_horizon_count']==1


@pytest.mark.parametrize('status',['unavailable','ambiguous'])
def test_no_fake_grid_for_missing_or_ambiguous(status):
    r=record();r.update(status=status,H_model_from_image=None,anchor_id=None)
    assert project_model_segments(model(),r,[[[1,1],[5,5]]],(100,100))['segments_image']==[]


@pytest.mark.parametrize('field,value',[
    ('model_id','another-rink'),('model_revision','v2'),('source_sha256','not-a-hash'),
    ('pts',True),('pts',1.2),('time_base','0'),('time_base','1/0'),
    ('matrix_direction','H_image_from_model'),('image_domain','CSS'),('status','validated'),
    ('anchor_id',None),('H_model_from_image',[[0,0,0]]*3)])
def test_invalid_contract_rejected(field,value):
    r=record();r[field]=value
    with pytest.raises(ValueError): validate_frame_map(r,model())


def test_exact_source_time_and_not_accuracy_validation():
    state=validate_frame_map(record(),model())
    assert state['source_time']==Fraction(13,5)
    assert not state['accuracy_validation_performed']


def test_predicted_requires_provenance():
    r=record();r['status']='predicted'
    with pytest.raises(ValueError):validate_frame_map(r,model())
    r['propagated_from']=['anchor_left','anchor_right']
    assert validate_frame_map(r,model())['status']=='predicted'


def test_missing_transform_cannot_be_identity():
    r=record();r['status']='unavailable'
    with pytest.raises(ValueError):validate_frame_map(r,model())


def test_projective_not_square_or_metres():
    for kind in ['equal_spacing','metre']:
        with pytest.raises(ValueError):model().grid_label(kind)
    assert '非等距' in model().grid_label()


def test_euclidean_not_metres():
    m=model('euclidean_unscaled','arbitrary_unit')
    assert '任意单位' in m.grid_label('equal_spacing')
    with pytest.raises(ValueError):m.grid_label('metre')


@pytest.mark.parametrize('level',['metric_assumed','metric_verified'])
def test_metric_needs_evidence_refs(level):
    with pytest.raises(ValueError):RinkReference('id','v1',level,'m','x-y')
    with pytest.raises(ValueError):RinkReference('id','v1',level,'m','x-y',('geometry',))
    assert '米制' in model(level,'m').grid_label('metre')


def test_raw_distortion_requires_explicit_approximation():
    r=record();r['image_domain']='raw_px_distortion_unchecked'
    with pytest.raises(ValueError):project_model_segments(model(),r,[[[1,1],[5,5]]],(10,10))
    assert project_model_segments(model(),r,[[[1,1],[5,5]]],(10,10),allow_approximate_raw=True)['lens_distortion_approximation']


def test_duplicate_sources_are_not_independent():
    check_source_group_split(['same_camera_clip'],['different_clip'])
    with pytest.raises(ValueError):check_source_group_split(['same_camera_clip'],['same_camera_clip'])


def test_singular_and_nonfinite_transforms_rejected():
    for bad in [np.zeros((3,3)),np.full((3,3),np.nan)]:
        with pytest.raises(ValueError):compose_model_from_image(bad,np.eye(3),np.eye(3),np.eye(3))


def test_unbuilt_template_not_validated():
    with pytest.raises(ValueError):RinkReference(None,None,'projective','reference_px',None)


def test_equal_geometry_unit_contract():
    with pytest.raises(ValueError):model('projective','m')
    with pytest.raises(ValueError):model('euclidean_unscaled','reference_px')
