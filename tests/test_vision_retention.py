import copy
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from rinkspace.vision_retention import source_retention_snapshot,source_retention_regressions

BINDING=dict(source_sha256='a'*64,time_base='1/15360',source_pts=256,
             native_hw=[8,8],shot_id='source-shot',installation_id='camera')
CLASSES=('background','ice_contact','ice_paint')

def snapshot(*,bad=(),region_bad=(),primitive_bad=()):
    semantic=torch.zeros((1,3,8,8));semantic[:,0]=1
    semantic[:,:,2,2]=torch.tensor([0.,1.,0.]);semantic[:,:,2,4]=torch.tensor([0.,1.,0.])
    for x,y in bad:semantic[:,:,y,x]=torch.tensor([1.,0.,0.])
    for x,y in region_bad:semantic[:,:,y,x]=torch.tensor([0.,1.,0.])
    primitive=torch.zeros((1,3,8,8));primitive[:,2]=1
    primitive[:,:,2,2]=torch.tensor([0.,1.,0.]);primitive[:,:,2,4]=torch.tensor([0.,1.,0.])
    for x,y in primitive_bad:primitive[:,:,y,x]=torch.tensor([1.,0.,0.])
    annotation=dict(group_id='confirmed-original-positions',kind='ice_contact',primitive='circle',
        uv=np.array([[2.,2.],[4.,2.]]),source_point_indices=np.array([3,9]),state='user_confirmed_visual_supervision')
    mask=torch.zeros((1,1,8,8),dtype=torch.bool);mask[:,:,5:7,1:4]=True
    region=dict(region_id='confirmed-contact-absent',excluded_kinds=['ice_contact'])
    return source_retention_snapshot(semantic,primitive,source_binding=BINDING,annotations=[annotation],
                                     classes=CLASSES,reviewed_regions=[(region,mask)])

def test_new_lost_original_point_is_not_hidden_by_other_recovered_position():
    baseline=snapshot(bad=[(2,2)])
    current=snapshot(bad=[(4,2)])
    result=source_retention_regressions(current,baseline)
    assert not result['retained_without_new_classification_errors']
    assert result['newly_lost_confirmed_support'][0]['newly_wrong_original_indices']==[9]

def test_reject_region_keeps_other_class_open_but_detects_new_contact():
    baseline=snapshot()
    result=source_retention_regressions(snapshot(region_bad=[(2,5)]),baseline)
    assert result['newly_lost_confirmed_support'][0]['newly_wrong_original_indices']==[42]
    assert baseline['reviewed_regions'][0]['primitive_wrong_native_flat_indices'] is None

def test_known_circle_primitive_regression_is_separate_from_semantic_class():
    result=source_retention_regressions(snapshot(primitive_bad=[(4,2)]),snapshot())
    assert result['newly_lost_confirmed_support'][0]['field']=='primitive_wrong_source_point_indices'
    assert result['newly_lost_confirmed_support'][0]['newly_wrong_original_indices']==[9]

def test_retained_snapshot_is_not_pixel_accuracy_acceptance():
    result=source_retention_regressions(snapshot(),snapshot())
    assert result['retained_without_new_classification_errors']
    assert result['physical_boundary_or_independent_accuracy_accepted'] is False

@pytest.mark.parametrize('changed',('source','indices','mask','kind','missing_group'))
def test_changed_source_or_authority_is_rejected(changed):
    baseline=snapshot();current=copy.deepcopy(baseline)
    if changed=='source':current['source_binding']['source_pts']+=256
    if changed=='indices':current['human_groups'][0]['source_point_indices']=[3,10]
    if changed=='mask':current['reviewed_regions'][0]['native_mask_sha256']='b'*64
    if changed=='kind':current['human_groups'][0]['kind']='ice_paint'
    if changed=='missing_group':current['human_groups']=[]
    with pytest.raises(ValueError):source_retention_regressions(current,baseline)
