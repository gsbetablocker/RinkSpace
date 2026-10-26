import copy,hashlib,importlib.util,json
import os
from pathlib import Path, PureWindowsPath
import numpy as np
import pytest
from rinkspace.vision_ice_model_context_identity import SameStateIceModelContextIdentity
from rinkspace.vision_ice_single_gray_support_identity import ConfirmedSingleGraySupportModelContextIdentity,_same_navigation_carrier,F3,DISPLAY_ROW3
from test_vision_ice_model_context_identity import source

PREDECESSOR=Path(r'D:\CodexData\RinkSpace_Data\events\ankara_2026\diagnostics\current56_shared_active_set_filter_20261009\systematic_continuation_20261010\full_programme_strict_source_review_20261012\model_context_pre_single_gray_support_20261013\vision_ice_model_context_identity_predecessor.py')
# A Windows absolute path is one oversized POSIX filename, so stat() itself
# can fail before pytest evaluates the historical-fixture skip condition.
PREDECESSOR_AVAILABLE = ((os.name == 'nt' or not PureWindowsPath(str(PREDECESSOR)).is_absolute())
                         and PREDECESSOR.is_file())

def context(cls,tmp_path,*,gray=False):
    authority=tmp_path/'authority.json'
    if not authority.exists():authority.write_text('{}')
    ref=dict(path=str(authority),sha256=hashlib.sha256(authority.read_bytes()).hexdigest())
    c=cls.__new__(cls)
    names=[F3,DISPLAY_ROW3,'lane3:edge0'] if gray else ['B1:segment0']
    c.inventory=[dict(curve_id=n,feature_id=n.split(':')[0],world_curve_role='segment0',operator='paint_center',material_signal='dark' if gray else 'blue') for n in names]
    c.source_sha256='a'*64;c.time_base='1/1000';c.authority=ref;c.visible_F3_authority=ref
    c.state_reference=c.scene_reference=c.camera_reference=ref;c.source_negative_materials=[]
    c.feature_kind={n.split(':')[0]:'polyline' for n in names}
    c.previous_gray=c.previous_mask=c.previous_binding=None;c.previous_tracks={}
    c._single_gray_support_by_candidate={};c.latest_single_gray_support_certificate=None
    c.paths=[dict(xyz=np.array([[0.,0.,0.],[200.,0.,0.]])),dict(xyz=np.array([[-50.,0.,0.],[250.,0.,0.]])),dict(xyz=np.array([[0.,10.,0.],[200.,10.,0.]]))]
    c.navigation=lambda m:([np.array([[0.,m['candidate_uv'][0,1]],[239.,m['candidate_uv'][0,1]]])],)*2+([np.array([[0.,120.],[239.,120.]])],) if gray else [[np.array([[0.,m['candidate_uv'][0,1]],[239.,m['candidate_uv'][0,1]]])]]
    return c

def gray_source(pts,shift=0):
    rgb,m=source(pts,shift)
    rgb[50+shift:74+shift]=np.mean(rgb[50+shift:74+shift],axis=2).astype(np.uint8)[...,None]
    m['decoded_native_RGB_sha256']=hashlib.sha256(rgb.tobytes()).hexdigest()
    m['source_binding']['decoded_native_BGR_sha256']=hashlib.sha256(rgb[:,:,::-1].tobytes()).hexdigest()
    q=np.repeat(np.asarray(m['candidate_sides'])[:,:,None,:],3,axis=2)
    q[:,:,0,1]-=.25;q[:,:,2,1]+=.25;m['candidate_side_quantiles_uv']=q
    return rgb,m

def literal(v):
    if isinstance(v,np.ndarray):return dict(dtype=v.dtype.str,shape=v.shape,bytes=v.tobytes().hex())
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:literal(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [literal(x) for x in v]
    return v

@pytest.mark.skipif(not PREDECESSOR_AVAILABLE, reason="Requires the recorded predecessor source; not distributed in the public fixture")
def test_default_context_two_exposures_is_bitexact_to_saved_predecessor(tmp_path):
    spec=importlib.util.spec_from_file_location('rinkspace._single_gray_saved_predecessor',PREDECESSOR)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    a=context(old.SameStateIceModelContextIdentity,tmp_path);b=context(SameStateIceModelContextIdentity,tmp_path)
    for pts,shift in [(10,0),(20,2)]:
        rgb,m=source(pts,shift);mask=np.zeros(rgb.shape[:2],bool)
        first=a.consume(rgb,copy.deepcopy(m),mask,source_ice_scope_mask=~mask)
        second=b.consume(rgb,copy.deepcopy(m),mask,source_ice_scope_mask=~mask)
        assert literal(first)==literal(second)
        assert literal(a.previous_tracks)==literal(b.previous_tracks)

def test_source_qualified_single_F3_support_recovers_two_image_seed_without_row3_output(tmp_path):
    old=context(SameStateIceModelContextIdentity,tmp_path,gray=True)
    new=context(ConfirmedSingleGraySupportModelContextIdentity,tmp_path,gray=True)
    for pts,shift in [(10,0),(20,2)]:
        rgb,m=gray_source(pts,shift);mask=np.zeros(rgb.shape[:2],bool)
        before=copy.deepcopy(m)
        baseline=old.consume(rgb,m,mask,source_ice_scope_mask=~mask)
        result=new.consume(rgb,m,mask,source_ice_scope_mask=~mask)
        assert not baseline['seeds']
        assert literal(m)==literal(before)
    assert [s['curve_id'] for s in result['seeds']]==[F3]
    e=result['seeds'][0]['source_identity_navigation_evidence']['source_single_physical_gray_support']
    assert e['every_other_Lane_house_or_material_competitor_retained']
    assert e['whole_curve_physical_alias_equation'] is False
    assert np.array_equal(result['seeds'][0]['navigation_uv'],m['candidate_side_quantiles_uv'][:,:,1].mean(1))

def test_competitor_exemption_only_for_qualified_measured_support_and_keeps_Lane(tmp_path):
    c=context(ConfirmedSingleGraySupportModelContextIdentity,tmp_path,gray=True)
    rgb,m=gray_source(10);mask=np.zeros(rgb.shape[:2],bool)
    m['candidate_flags'][1]|=2;m['candidate_side_quantiles_uv'][2]=np.nan
    mask[60:64,132:138]=True
    identities=np.tile([0,1,2],(6,1))
    result=c._source_field_order_competitors(rgb,m,mask,0,np.arange(6),identities)
    assert not result[:,0].any() and result[:,2].all()
    assert result[1,1] and result[2,1] and result[3,1]
    assert not result[[0,4,5],1].any()
    assert c._source_field_order_competitors(rgb,m,mask,2,np.arange(6),identities)[:,1].all()

def test_nearby_parallel_navigation_carrier_is_not_same_material_support(tmp_path):
    a=np.array([[0.,0.,0.],[200.,0.,0.]])
    assert _same_navigation_carrier(a,np.array([[-50.,0.,0.],[250.,0.,0.]]))
    assert not _same_navigation_carrier(a,a+[0,1e-4,0])
    c=context(ConfirmedSingleGraySupportModelContextIdentity,tmp_path,gray=True)
    c.paths[1]['xyz']+=np.array([0.,1e-4,0.])
    rgb,m=gray_source(10);mask=np.zeros(rgb.shape[:2],bool);identities=np.tile([0,1,2],(6,1))
    result=c._source_field_order_competitors(rgb,m,mask,0,np.arange(6),identities)
    assert result[:,1:].all()
