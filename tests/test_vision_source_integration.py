from copy import deepcopy
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from rinkspace.vision_supervision import attach_source_support_integration, confirmed_position_interval_localization_loss
from rinkspace.vision import _annotations_in_crop, interval_supervision, primitive_supervision

def source(duplicate=False):
    line=dict(group_id='first',physical_curve_id='same_paint',kind='ice_paint',primitive='line',
        uv=[[2.,5.],[8.,5.],[18.,5.]],normal=[[0.,1.]]*3,interval_px=[-1.,1.],
        state='user_confirmed_visual_supervision',authority=dict(human_authority=True))
    annotations=[line]
    if duplicate: annotations.append({**deepcopy(line),'group_id':'duplicate'})
    return dict(source_hash='a'*64,time_base='1/25',source_pts=10,annotations=annotations)

def losses(sample):
    sample=attach_source_support_integration(dict(samples=[sample]))['samples'][0]
    annotations=_annotations_in_crop(sample,(0,0),(16,24))
    torch.manual_seed(9)
    logits=torch.randn(1,3,16,24,requires_grad=True)
    semantic=interval_supervision(logits,annotations,['background','ice_paint','other'])[0]
    primitive=primitive_supervision(logits,annotations)
    localization=confirmed_position_interval_localization_loss(logits,annotations,
        ['background','ice_paint','other'],original_sample=sample,crop_origin_xy=(0,0),profile_radius_px=3,rank_weight=1.)[0]
    total=semantic+primitive+localization
    total.backward()
    return torch.stack([semantic,primitive,localization]).detach(),logits.grad

def test_duplicate_curve_path_does_not_change_network_losses_or_gradient():
    first,gradient=losses(source())
    copied,copied_gradient=losses(source(True))
    torch.testing.assert_close(first,copied,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(gradient,copied_gradient,atol=1e-6,rtol=1e-6)

def test_visible_curve_mass_preserves_original_points_without_occlusion_bridge():
    sample=source();line=sample['annotations'][0]
    line['source_visible_mask']=[True,False,True]
    out=attach_source_support_integration(dict(samples=[sample]))['samples'][0]
    assert out['annotations'][0]['source_integration_weights']==[.5,0.,.5]
    assert out['annotations'][0]['uv']==line['uv']
    assert 'source_integration_weights' not in line
    selected=_annotations_in_crop(out,(0,0),(16,24))[0]
    assert selected['source_point_indices'].tolist()==[0,2]
    np.testing.assert_array_equal(selected['source_integration_weights'],[.5,.5])
