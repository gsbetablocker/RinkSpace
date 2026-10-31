import json

import numpy as np
import pytest

from rinkspace.session.native_material_association import source_interface_material_support


def _source(contrasts, roles=None):
    count=len(contrasts)
    if roles is None:roles=np.arange(count)%2
    inventory=[dict(world_curve_role=role,operator='material_interface',material_signal='dark',
        material_transition='light_to_dark' if role=='middle' else 'dark_to_light')
        for role in ('middle','outer')]
    rgb=np.full((80,24*count+30,3),210.,np.float32)
    uv=np.c_[20.+24*np.arange(count),np.full(count,40.)]
    quantiles=np.empty((count,2,3,2))
    base=np.array([200.,210.,220.])
    for index,(x,_) in enumerate(uv):
        x=int(x);paint=base-contrasts[index]
        if roles[index]==0:
            rgb[:40,x-8:x+9]=base;rgb[40:,x-8:x+9]=paint
        else:
            rgb[:40,x-8:x+9]=paint;rgb[40:,x-8:x+9]=base
        quantiles[index]=np.array([[[x,39.],[x,40.],[x,41.]]]*2)
    measurement=dict(candidate_uv=uv,candidate_query=np.arange(count),query_curve=np.asarray(roles,int),
        query_normal=np.tile([0.,1.],(count,1)),candidate_side_quantiles_uv=quantiles,
        candidate_flags=np.zeros(count,np.uint8))
    return rgb,measurement,inventory


def _set_flanks(rgb, measured, index, paint, background):
    x=int(measured['candidate_uv'][index,0])
    negative_y=[36,33,30];positive_y=[44,47,50]
    paint_positive=measured['query_curve'][index]==0
    for y,value in zip(positive_y if paint_positive else negative_y,paint):rgb[y,x]=value
    for y,value in zip(negative_y if paint_positive else positive_y,background):rgb[y,x]=value


def test_clear_weak_interface_keeps_support_when_unrelated_stronger_interface_is_added():
    rgb,measured,inventory=_source([20.,100.],roles=[0,1])
    _set_flanks(rgb,measured,0,[190.,191.,189.],[210.,211.,209.])
    before={key:value.copy() for key,value in measured.items()}
    weak_only,first=source_interface_material_support(rgb,measured,inventory,[0])
    support,receipt=source_interface_material_support(rgb,measured,inventory,[0,1])
    np.testing.assert_array_equal(weak_only,[True,False])
    np.testing.assert_array_equal(support,[True,True])
    assert first['supported_ids']==[0]
    np.testing.assert_allclose(receipt['contrast'],[20.,100.])
    assert receipt['paint_flank_spread']==[2.,0.]
    assert receipt['background_flank_spread']==[2.,0.]
    assert receipt['flanks_separated']==[True,True]
    assert receipt['source_separation_luma']==[18.,100.]
    assert receipt['threshold_method']=='local_source_flank_separation'
    assert receipt['otsu_threshold_luma'] is None
    assert not receipt['physical_line_absence_asserted']
    json.dumps(receipt,allow_nan=False)
    for key,value in measured.items():np.testing.assert_array_equal(value,before[key])


@pytest.mark.parametrize('paint,background,separation',[
    ([180.,190.,220.],[200.,210.,230.],-20.),
    ([180.,190.,200.],[200.,210.,220.],0.),
    ([150.,180.,190.],[195.,210.,240.],5.),
    ([150.,180.,185.],[190.,215.,225.],5.),
])
def test_overlapping_or_varying_flanks_are_unresolved_despite_positive_contrast(paint,background,separation):
    rgb,measured,inventory=_source([20.])
    _set_flanks(rgb,measured,0,paint,background)
    support,receipt=source_interface_material_support(rgb,measured,inventory,[0])
    assert receipt['contrast'][0]>0.
    assert receipt['source_separation_luma']==[separation]
    assert receipt['flanks_separated']==[separation>0.]
    assert not support.any()
    assert receipt['support_status']==['unresolved']


def test_small_positive_difference_cannot_qualify_touching_flank_trends():
    rgb,measured,inventory=_source([4.])
    _set_flanks(rgb,measured,0,[190.,192.,194.],[194.,196.,198.])
    support,receipt=source_interface_material_support(rgb,measured,inventory,[0])
    assert receipt['contrast']==receipt['paint_flank_spread']==receipt['background_flank_spread']==[4.]
    assert receipt['source_separation_luma']==[0.]
    assert not support.any()


def test_dark_transition_roles_use_opposite_paint_flanks_and_signed_contrast():
    rgb,measured,inventory=_source([20.,20.,-20.,-20.])
    support,receipt=source_interface_material_support(rgb,measured,inventory,[0,1,2,3])
    np.testing.assert_array_equal(support,[True,True,False,False])
    np.testing.assert_allclose(receipt['paint_median_luma'],[190.,190.,230.,230.])
    np.testing.assert_allclose(receipt['contrast'],[20.,20.,-20.,-20.])
    assert receipt['flanks_separated']==[True,True,False,False]


def test_masked_and_missing_source_flanks_remain_unresolved():
    rgb,measured,inventory=_source([2.,10.,200.,80.])
    mask=np.zeros(rgb.shape[:2],bool)
    mask[47,int(measured['candidate_uv'][2,0])]=True
    measured['candidate_side_quantiles_uv'][3]=np.nan
    support,receipt=source_interface_material_support(rgb,measured,inventory,[0,1,2,3],mask)
    np.testing.assert_array_equal(support,[True,True,False,False])
    assert receipt['source_valid']==[True,True,False,False]
    assert receipt['contrast'][3] is None
    assert receipt['source_separation_luma'][3] is None
    assert receipt['support_status'][2:]==['unresolved','unresolved']
    json.dumps(receipt,allow_nan=False)


def test_cropped_flagged_and_masked_interfaces_cannot_be_qualified_by_separated_flanks():
    rgb,measured,inventory=_source([20.]*5)
    measured['candidate_uv'][1,1]=4.
    measured['candidate_side_quantiles_uv'][1,:,:,1]=[3.,4.,5.]
    measured['candidate_flags'][2]=1
    measured['candidate_flags'][3]=2
    mask=np.zeros(rgb.shape[:2],bool)
    mask[40,int(measured['candidate_uv'][4,0])]=True
    support,receipt=source_interface_material_support(rgb,measured,inventory,np.arange(5),mask)
    np.testing.assert_array_equal(support,[True,False,False,False,False])
    assert receipt['source_valid']==[True,False,False,False,False]
    assert receipt['source_separation_luma'][1] is None
    assert receipt['flanks_separated'][2:]==[True,True,True]
    json.dumps(receipt,allow_nan=False)


@pytest.mark.parametrize('contrast',[0.,.25,3.25])
def test_uniform_source_contrast_uses_local_evidence_without_a_global_floor(contrast):
    rgb,measured,inventory=_source([contrast]*4)
    support,receipt=source_interface_material_support(rgb,measured,inventory,[0,1,2,3])
    np.testing.assert_array_equal(support,np.full(4,contrast>0.))
    assert receipt['otsu_threshold_luma'] is None
    assert receipt['threshold_method']=='local_source_flank_separation'


def test_empty_selection_is_safe_and_unselected_pixels_cannot_change_support():
    rgb,measured,inventory=_source([4.,20.,200.])
    support,receipt=source_interface_material_support(rgb,measured,inventory,[0,1])
    np.testing.assert_array_equal(support,[True,True,False])
    assert receipt['support_status'][2]=='not_selected'
    empty,empty_receipt=source_interface_material_support(rgb,measured,inventory,[])
    assert not empty.any() and empty_receipt['otsu_threshold_luma'] is None
    zero,zero_receipt=source_interface_material_support(rgb,{'candidate_uv':np.empty((0,2))},inventory,[])
    assert zero.shape==(0,) and zero_receipt['contrast']==[]
    assert zero_receipt['flanks_separated']==[] and zero_receipt['source_separation_luma']==[]


@pytest.mark.parametrize('direction',[-1,1])
def test_uniform_illumination_profile_is_rejected_by_upstream_search_boundary_flag(direction):
    from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader

    inventory=[dict(curve_id=f'lighting:{role}',feature_id='lighting',world_curve_role=role,
        operator='material_interface',material_signal='dark',
        material_transition='light_to_dark' if role=='middle' else 'dark_to_light',
        native_path_id=f'lighting/{role}',xyz=np.zeros((2,3))) for role in ('middle','outer')]
    rgb=np.broadcast_to((128+direction*np.arange(128))[:,None,None],(128,128,3)).astype(np.uint8)
    projections={p['native_path_id']:dict(uv=np.array([[16.,64.],[112.,64.]]),valid=np.ones(2,bool))
        for p in inventory}
    context=dict(source_sha256='a'*64,time_base='1/12800',
        authority=[dict(path='synthetic_uniform_illumination',sha256='b'*64)])
    reader=IceProgrammeSourceReader(None,source_sha256='a'*64,time_base='1/12800',
        native_paths=inventory,search_radius_px=32.,prepared_source_context=context)
    measured=reader.read(rgb,256,None,np.zeros(rgb.shape[:2],bool),
        shot_id='uniform_illumination',native_projections=projections)
    assert len(measured['candidate_uv'])>0
    assert (measured['candidate_flags']&1).all()
    support,receipt=source_interface_material_support(rgb,measured,inventory,
        np.arange(len(measured['candidate_uv'])))
    # A monotone ramp separates the sampled flanks. The complete-profile
    # candidate reaches the search boundary, so the upstream flag rejects it.
    assert all(receipt['flanks_separated'])
    assert not any(receipt['source_valid'])
    assert not support.any()
