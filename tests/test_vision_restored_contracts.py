"""Regressions for restored material meaning, temporal identity and top-deck exclusion."""
import cv2
import numpy as np
import pytest
from rinkspace.vision_near_board import read_near_board, exclude_top_face_prints


def binding():
    return dict(source_sha256='a'*64, source_pts=256, time_base='1/15360',shot_id='synthetic',native_hw=[1080,1920])


def test_near_cap_ice_edge_excludes_outside_print_but_far_blue_strip_is_not_near_cap():
    rgb=np.full((1080,1920,3),225,np.uint8)
    rgb[190:210,:]=[20,45,140]
    assert read_near_board(rgb,binding())['exclusion_pixel_count']==0
    cv2.fillPoly(rgb,[np.array([[0,650],[330,1079],[0,1079]])],(20,45,140))
    result=read_near_board(rgb,binding())
    assert result['readings'] and result['exclusion_pixel_count']>30000
    mask=result['ad_exclusion_mask']
    assert mask[900,30] and not mask[900,800]
    uv=np.array([[30.,900.],[800.,900.]],np.float32)
    packet=dict(source_binding=binding(),instances=[dict(kind='advertisement_panel',region_mask=np.ones((1080,1920),bool),
        boundary_readings=[dict(actual_source_uv=uv,source_indices=np.array([0,1]))])])
    result=exclude_top_face_prints(packet,result)
    row=result['instances'][0]['boundary_readings'][0]
    np.testing.assert_array_equal(row['actual_source_uv'],[[800,900]])
    np.testing.assert_array_equal(row['top_face_excluded_source_uv'],[[30,900]])


def test_top_deck_mask_cannot_transfer_to_another_native_exposure():
    rgb=np.zeros((1080,1920,3),np.uint8)
    result=read_near_board(rgb,binding())
    with pytest.raises(ValueError,match='different source frames'):
        exclude_top_face_prints(dict(source_binding={**binding(),'source_pts':512},instances=[]),result)


def test_temporally_invalid_circle_peaks_cannot_reenter_through_joint_shape_fit():
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('CUDA source reader')
    from rinkspace.vision_geometry import guided_source_reading,coherent_named_source_reading
    rgb=np.full((120,300,3),215,np.uint8)
    cv2.ellipse(rgb,(150,60),(110,30),0,0,360,(165,165,165),-1)
    points=np.c_[np.linspace(45,255,18),np.full(18,60.)-30*np.sqrt(1-((np.linspace(45,255,18)-150)/110)**2)]
    b={**binding(), 'native_hw':[120,300]}
    image=torch.tensor(rgb,device='cuda').permute(2,0,1)[None].float()/255
    prob=torch.zeros((1,2,120,300),device='cuda');prob[:,1]=1
    guide=dict(id='circle',kind='ice_paint',primitive='circle',physical_role='filled_disc_boundary',
        points_uv=points, normals=np.tile([0.,1.],(len(points),1)),source_binding=b,
        origin='temporal_source_search_hint',identity_authority={'source':'prior_confirmed'},identity_confirmed=True,
        temporal_flow_supported=np.zeros(len(points),bool),search_radius_px=8.,visible_arc_branch='upper')
    result=guided_source_reading(image,prob,[guide],classes=['background','ice_paint'],rgb_source_binding=b,
        probability_source_binding=b,guide_source_binding=b,probabilities_conditioned_on_geometry=False)
    result=coherent_named_source_reading(result)
    row=result['readings'][0]
    assert len(row['actual_source_uv'])==0 and len(row['coherent_source_uv'])==0
    assert not row['source_peak_candidates']['source_valid'].any()


@pytest.mark.parametrize('outward',['dark_to_light','light_to_dark'])
def test_circle_joint_shape_obeys_both_dark_and_light_inside_materials(outward):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('CUDA source reader')
    from rinkspace.vision_geometry import guided_source_reading,coherent_named_source_reading
    outer,inner=(215,155) if outward=='dark_to_light' else (155,215)
    rgb=np.full((160,360,3),outer,np.uint8)
    cv2.ellipse(rgb,(180,80),(140,45),0,0,360,(inner,inner,inner),-1)
    rgb=cv2.GaussianBlur(rgb,(0,0),1.)
    b={**binding(), 'native_hw':[160,360]}
    image=torch.tensor(rgb,device='cuda').permute(2,0,1)[None].float()/255
    prob=torch.zeros((1,2,160,360),device='cuda');prob[:,1]=1
    guides=[]
    for branch,sign in [('upper',-1),('lower',1)]:
        x=np.linspace(50,310,25);y=80+sign*45*np.sqrt(1-((x-180)/140)**2)
        lightens=(sign>0)==(outward=='dark_to_light')
        guides.append(dict(id='circle',kind='ice_paint',primitive='circle',physical_role='filled_disc_boundary',
            points_uv=np.c_[x,y],normals=np.tile([0.,1.],(len(x),1)),source_binding=b,
            origin='user_confirmed_visual_supervision',identity_authority={'synthetic_test':True},
            rgb_transition_polarity='lightening_along_normal' if lightens else 'darkening_along_normal',
            material_outward_transition=outward,visible_arc_branch=branch,search_radius_px=7.))
    result=guided_source_reading(image,prob,guides,classes=['background','ice_paint'],rgb_source_binding=b,
        probability_source_binding=b,guide_source_binding=b,probabilities_conditioned_on_geometry=False)
    result=coherent_named_source_reading(result)
    for row in result['readings']:
        assert row['coherent_flags']['status']=='joint_actual_source_peak_image_conic_hypothesis'
        assert len(row['coherent_source_uv'])>=20


def test_excluded_ad_point_breaks_runs_and_clears_quality():
    mask=np.zeros((20,30),bool);mask[10,10]=True
    b={**binding(),'native_hw':[20,30]}
    uv=np.array([[5.,10.],[10.,10.],[15.,10.]],np.float32)
    row=dict(actual_source_uv=uv,source_indices=np.arange(3),segment_ids=np.zeros(3,int),
        source_quality_mask=np.ones(3,bool),measured_runs=[dict(actual_source_uv=uv)])
    packet=dict(source_binding=b,instances=[dict(kind='advertisement_panel',boundary_readings=[row])])
    near=dict(source_binding=b,ad_exclusion_mask=mask,exclusion_pixel_count=1,source_reader='test')
    exclude_top_face_prints(packet,near)
    assert [len(r['actual_source_uv']) for r in row['measured_runs']]==[1,1]
    np.testing.assert_array_equal(row['source_indices'],[0,2])
    np.testing.assert_array_equal(row['source_quality_mask'],[True,False,True])


def test_qualified_near_board_rejects_changed_native_pixels():
    from rinkspace.vision_near_board import configured_near_evidence
    from rinkspace.vision_near_board_qualified import qualified_index,reread_qualified_near_board
    ref=configured_near_evidence().get('official_gray_generation')
    if ref is None:pytest.skip('No event source generation installed')
    manifest,index=qualified_index(ref['path'],ref['sha256'])
    source=manifest['source']
    b=dict(source_sha256=source['sha256'],source_pts=next(iter(index)),time_base=source['time_base'],shot_id='s06')
    with pytest.raises(ValueError,match='different native source pixels'):
        reread_qualified_near_board(np.zeros((1080,1920,3),np.uint8),b,ref)


def test_source_json_keeps_numpy_material_flags_boolean():
    import json
    from rinkspace.vision import _source_json
    packet={'direct':np.bool_(True),'pending':np.bool_(False),'pts':np.int64(1342208)}
    assert json.loads(json.dumps(_source_json(packet),allow_nan=False))=={'direct':True,'pending':False,'pts':1342208}
