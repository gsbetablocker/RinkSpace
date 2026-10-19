import hashlib
import copy

import cv2
import numpy as np
import pytest

from rinkspace.vision_ice_source_guides import (append_actual_source_guided_queries,
    remeasure_actual_source_guided_queries,ActualIceSourceGuideTracker)


def source(shift=0,native=10):
    rgb=np.full((260,320,3),210,np.uint8)
    rgb[170+shift:195+shift]=[210,65,65]
    rgb=cv2.GaussianBlur(rgb,(3,3),.6)
    uv=np.array([[60.,40.]],np.float64)
    measurement=dict(source_binding=dict(source_sha256='a'*64,source_pts=10,actual_pts=10,native_pts=native,
        time_base='1/1000',shot_id='one'),decoded_native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        query_uv=uv,query_normal=np.array([[0.,1.]],np.float64),query_interval=np.array([[-20.,20.]],np.float64),
        query_curve=np.zeros(1,np.int16),query_status=np.array([2],np.uint8),
        candidate_uv=np.empty((0,2),np.float32),candidate_sides=np.empty((0,2,2),np.float32),
        candidate_query=np.empty(0,np.int32),candidate_flags=np.empty(0,np.uint8))
    inventory=[dict(curve_id='C:segment0',operator='paint_center',material_signal='red',material_transition=None)]
    return rgb,measurement,inventory


def guide(rgb,measurement,authority):
    b=measurement['source_binding']
    return dict(curve_id='C:segment0',navigation_uv=[[40.,175.],[280.,175.]],actual_pts=10,native_pts=b['native_pts'],
        shot_id='one',source_sha256='a'*64,time_base='1/1000',decoded_native_RGB_sha256=measurement['decoded_native_RGB_sha256'],
        decoded_native_BGR_sha256=hashlib.sha256(rgb[:,:,::-1].tobytes()).hexdigest(),authority=authority,
        original_source_image_actually_reviewed=True,physical_identity_from_model_projection=False,
        normal_search_half_extent_px=60.)


def test_source_guide_reads_obvious_paint_outside_frozen_camera_queries(tmp_path):
    path=tmp_path/'actual-source-identity.json';path.write_text('{}')
    authority=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    rgb,m,inv=source();g=guide(rgb,m,authority)
    result=append_actual_source_guided_queries(rgb,m,np.zeros(rgb.shape[:2],bool),inv,[g])
    assert np.array_equal(result['query_uv'][:1],m['query_uv']) and len(m['candidate_uv'])==0
    assert len(result['candidate_uv'])>20
    assert np.all(np.abs(result['candidate_uv'][:,1]-182.)<1.)
    assert np.all(result['candidate_flags']&8)  # A guide is not admission.
    assert np.isfinite(result['candidate_side_quantiles_uv']).all()
    assert not np.any(result['candidate_uv'][:,1]==175.)  # Navigation is not output truth.
    receipt=result['source_guided_queries']
    assert receipt['original_query_prefix_length']==1 and receipt['original_candidate_prefix_length']==0
    assert np.array_equal(receipt['query_uv'],result['query_uv'][1:])


def test_alias_guide_rereads_own_native_sides_instead_of_copying_canonical_uv(tmp_path):
    path=tmp_path/'identity.json';path.write_text('{}')
    authority=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    rgb,m,inv=source();mask=np.zeros(rgb.shape[:2],bool)
    canonical=append_actual_source_guided_queries(rgb,m,mask,inv,[guide(rgb,m,authority)])
    alias,own,inv=source(shift=3,native=11)
    observed=remeasure_actual_source_guided_queries(alias,own,mask,inv,canonical['source_guided_queries'])
    assert np.array_equal(observed['query_uv'],canonical['query_uv'])
    assert np.all(np.abs(observed['candidate_uv'][:,1]-185.)<1.)
    assert not np.array_equal(observed['candidate_uv'],canonical['candidate_uv'])
    assert observed['source_guided_queries']['own_native_RGB_profiles_read'] is True
    assert observed['source_guided_queries']['canonical_observed_UV_copied'] is False


def test_another_source_image_cannot_reuse_a_manual_guide_authority(tmp_path):
    path=tmp_path/'identity.json';path.write_text('{}')
    authority=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    rgb,m,inv=source();g=guide(rgb,m,authority)
    changed,own,_=source(shift=3)
    with pytest.raises(ValueError,match='reviewed original RGB'):
        append_actual_source_guided_queries(changed,own,np.zeros(rgb.shape[:2],bool),inv,[g])


def test_missing_LK_keeps_only_search_and_reads_current_material_without_promoting_it(tmp_path,monkeypatch):
    path=tmp_path/'identity.json';path.write_text('{}')
    authority=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    old,m,inv=source();mask=np.zeros(old.shape[:2],bool)
    tracker=ActualIceSourceGuideTracker(inv,source_sha256='a'*64,time_base='1/1000')
    tracker.previous=dict(pts=10,binding=m['source_binding'],gray=cv2.cvtColor(old,cv2.COLOR_RGB2GRAY),mask=mask,BGR_sha256=hashlib.sha256(old[:,:,::-1].tobytes()).hexdigest())
    tracker.supports=[dict(curve_id='C:segment0',curve_index=0,uv=np.array([[60.,182.],[260.,182.]]),
        sides=np.array([[[60.,170.],[60.,195.]],[[260.,170.],[260.,195.]]]),normal=np.array([[0.,1.],[0.,1.]]),
        actual_pts=10,decoded_BGR_sha256=tracker.previous['BGR_sha256'],authority=authority)]
    def missing(*args):
        support=args[2]
        return support['uv'],np.zeros(len(support['uv']),bool),np.full(len(support['uv']),np.nan)
    monkeypatch.setattr('rinkspace.vision_ice_identity_graph._point_source_flow',missing)
    rgb,own,_=source(shift=3,native=20)
    own['source_binding'].update(source_pts=20,actual_pts=20)
    result=tracker.consume(rgb,own,mask,source_ice_scope_mask=~mask)
    measured=result['measurement']
    assert np.all(np.abs(measured['candidate_uv'][:,1]-185.)<1.)
    assert np.all(measured['candidate_flags']&8)
    assert measured['source_guided_queries']['guides'][0]['current_RGB_tracking_confirmed_by_guide'] is False
    assert measured['source_guided_queries']['guides'][0]['previous_actual_pts']==10
    tracker.remember_selected_source_rows([])
    # A missing/admission-free frame must not discard the known-source search,
    # or apply stale material coordinates to the immediately previous RGB.
    def stale_flow(*args):raise AssertionError('Stale source points are not LK inputs')
    monkeypatch.setattr('rinkspace.vision_ice_identity_graph._point_source_flow',stale_flow)
    absent,empty,_=source(native=30);absent[:]=210
    empty['source_binding'].update(source_pts=30,actual_pts=30)
    empty['decoded_native_RGB_sha256']=hashlib.sha256(absent.tobytes()).hexdigest()
    missing=tracker.consume(absent,empty,mask,source_ice_scope_mask=~mask)['measurement']
    assert len(missing['query_uv'])>1 and len(missing['candidate_uv'])==0


def test_remember_final_current_source_overlay_accepts_exact_appended_readouts(tmp_path,monkeypatch):
    path=tmp_path/'identity.json';path.write_text('{}')
    authority=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    rgb,m,inv=source();mask=np.zeros(rgb.shape[:2],bool)
    tracker=ActualIceSourceGuideTracker(inv,source_sha256='a'*64,time_base='1/1000')
    original=tracker.consume(rgb,m,mask,source_ice_scope_mask=~mask)['measurement']
    final=append_actual_source_guided_queries(rgb,original,mask,inv,[guide(rgb,m,authority)])
    ids=np.arange(2)
    sides=np.asarray(final['candidate_sides'],float)[ids]
    row=dict(curve_id='C:segment0',actual_pts=10,decoded_native_RGB_sha256=m['decoded_native_RGB_sha256'],
        actual_source_RGB_measured=True,static_material_confirmed=True,pixel_location_confirmed=True,
        model_projection_or_fit_residual_used_for_identity=False,
        admission_status='admitted_source_material_center_temporal_identity',source_seed_authority=authority,
        original_candidate_indices=ids.tolist(),uv=sides.mean(1).tolist(),source_sides_uv=sides.tolist())
    monkeypatch.setattr('calibration.program_tracking.strict_absolute_source._temporal_identity',lambda row:True)
    tracker.remember_selected_source_rows([row],measurement=final)
    assert len(tracker.supports)==1 and np.array_equal(tracker.supports[0]['sides'],sides)
    assert 'measurement' not in tracker.previous and 'readout_prefix' not in tracker.previous


def test_remember_final_overlay_rejects_replaced_source_prefix_or_clock():
    rgb,m,inv=source();mask=np.zeros(rgb.shape[:2],bool)
    tracker=ActualIceSourceGuideTracker(inv,source_sha256='a'*64,time_base='1/1000')
    measured=tracker.consume(rgb,m,mask,source_ice_scope_mask=~mask)['measurement']
    wrong=copy.deepcopy(measured);wrong['query_uv'][0,1]+=1
    with pytest.raises(ValueError,match='readout prefix'):
        tracker.remember_selected_source_rows([],measurement=wrong)
    wrong=copy.deepcopy(measured);wrong['source_binding']['native_pts']=11
    with pytest.raises(ValueError,match='source clock'):
        tracker.remember_selected_source_rows([],measurement=wrong)


@pytest.mark.parametrize('forged',[False,True])
def test_final_q50_source_support_is_kept_exact_without_using_raw_sides(tmp_path,monkeypatch,forged):
    path=tmp_path/'identity.json';path.write_text('{}')
    authority=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    rgb,m,inv=source();mask=np.zeros(rgb.shape[:2],bool)
    tracker=ActualIceSourceGuideTracker(inv,source_sha256='a'*64,time_base='1/1000')
    original=tracker.consume(rgb,m,mask,source_ice_scope_mask=~mask)['measurement']
    final=append_actual_source_guided_queries(rgb,original,mask,inv,[guide(rgb,m,authority)])
    ids=np.arange(2)
    sides=np.asarray(final['candidate_side_quantiles_uv'],float)[ids,:,1]
    row=dict(curve_id='C:segment0',actual_pts=10,decoded_native_RGB_sha256=m['decoded_native_RGB_sha256'],
        actual_source_RGB_measured=True,static_material_confirmed=True,pixel_location_confirmed=True,
        model_projection_or_fit_residual_used_for_identity=False,
        admission_status='admitted_source_material_center_temporal_identity',source_seed_authority=authority,
        original_candidate_indices=ids.tolist(),uv=sides.mean(1).tolist(),source_sides_uv=sides.tolist())
    monkeypatch.setattr('calibration.program_tracking.strict_absolute_source._temporal_identity',lambda row:True)
    if forged:
        row['uv'][0][1]+=1e-8
        with pytest.raises(ValueError,match='exact current raw or measured q50'):
            tracker.remember_selected_source_rows([row],measurement=final)
    else:
        tracker.remember_selected_source_rows([row],measurement=final)
        assert np.array_equal(tracker.supports[0]['uv'],sides.mean(1))
        assert np.array_equal(tracker.supports[0]['sides'],sides)
