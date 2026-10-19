import copy
import numpy as np

from rinkspace.landmark_source_qualification import qualify_source_frame, source_shape
from rinkspace.landmark_band_profile import point_array_sha256
from rinkspace.landmark_association import AssociationConfig, associate_frame


def _fixture(colour='gray', curved=True, upper_reliable=True):
    x=np.linspace(0,600,61)
    bend=12*(2*x/600-1)**2 if curved else np.zeros_like(x)
    points=[np.c_[x,300+bend],np.c_[x,350-bend]]
    observations=[];profiles=[]
    for i,p in enumerate(points):
        observations.append({'observation_id':str(i),'colour_class':colour,
            'association_status':'detected_unidentified','points_source_px':p.tolist(),
            'visible_segment_ranges':[[0,len(p)]],'image_scale_from_source':[1,1]})
        profiles.append({'observation_id':str(i),'point_array_sha256':point_array_sha256(p),
            'gradient_reliable':[bool(i or upper_reliable)]*len(p),
            'gradient_unit_source_xy':[[0,-1 if i==0 else 1]]*len(p)})
    model={'structure':{'primitives':[{'id':identity,'kind':kind} for identity,kind in
        [('R1','circle'),('F2','circle'),('centre_circle','circle'),('red_center','axis_line'),
         ('blue_left','axis_line'),('blue_right','axis_line'),('F_centre_alignment_guide','guide')]]}}
    return {'landmark_observations':observations},model,profiles


def test_source_curved_outer_pair_has_circle_eligibility_without_camera_or_identity():
    frame,model,profiles=_fixture()
    before=copy.deepcopy(frame)
    result=qualify_source_frame(frame,model,profiles)
    assert frame==before
    assert all(r['source_circle_evidence_eligible'] for r in result['observations'])
    assert result['qualified_model_ids']=={'0':['F2','R1'],'1':['F2','R1']}
    assert all(not r['camera_projection_used_for_qualification'] for r in result['observations'])


def test_parallel_edges_cannot_become_circle_from_an_external_fitted_camera():
    frame,model,profiles=_fixture(curved=False)
    result=qualify_source_frame(frame,model,profiles)
    assert all(not ids for ids in result['qualified_model_ids'].values())
    assert all(r['weak_shape_does_not_mean_not_a_real_ring'] for r in result['observations'])
    assert all(r['association_status']=='detected_unidentified' for r in frame['landmark_observations'])


def test_one_sided_brightness_is_not_complementary_outer_evidence():
    frame,model,profiles=_fixture(upper_reliable=False)
    result=qualify_source_frame(frame,model,profiles)
    assert all(not ids for ids in result['qualified_model_ids'].values())


def test_centre_colour_allows_blue_circle_without_using_heldout_blue_lines():
    frame,model,profiles=_fixture(colour='blue')
    result=qualify_source_frame(frame,model,profiles)
    assert result['qualified_model_ids']=={'0':['centre_circle'],'1':['centre_circle']}
    frame['landmark_observations'][0]['colour_class']='red'
    result=qualify_source_frame(frame,model,profiles)
    assert all('centre_circle' not in ids for ids in result['qualified_model_ids'].values())


def test_explicit_qualification_is_applied_before_camera_curve_options():
    frame,model,profiles=_fixture()
    # No points_model/equations exist in the tiny model: an excluded circle
    # reaching the camera constructor would fail, regardless of pose fitness.
    result=associate_frame(frame,model,[],config=AssociationConfig(exclude_blue_from_fit=False),
        qualified_model_ids={'0':[],'1':[]})
    assert result['association']['hypotheses_attempted']==0
    assert all(o['selected_model_id'] is None for o in result['landmark_observations'])


def test_partial_curved_arc_uses_chromatic_support_without_inventing_missing_half():
    from rinkspace.landmark_source_qualification_v7 import qualify_source_frame_v7
    frame,model,profiles=_fixture()
    frame['landmark_observations']=frame['landmark_observations'][:1]
    before=copy.deepcopy(frame)
    profile={'observation_id':'0','point_array_sha256':profiles[0]['point_array_sha256'],
        'colour_evidence':'blue_ink_supported','median_chromatic_excess_br_bg_levels':[4.8,1.2],
        'blue_excess_point_fraction':.94,'reliable_point_count':61}
    result=qualify_source_frame_v7(frame,model,profiles,chromatic_profiles=[profile])
    assert frame==before
    assert result['qualified_model_ids']['0']==['centre_circle']
    assert result['observations'][0]['partial_visible_arc_only']
    assert result['observations'][0]['recommended_measurement_uncertainty_source_px']==8.
    profile['colour_evidence']='unknown'
    result=qualify_source_frame_v7(frame,model,profiles,chromatic_profiles=[profile])
    assert set(result['qualified_model_ids']['0'])=={'R1','F2','centre_circle'}
    assert result['observations'][0]['one_arc_or_colour_is_not_sufficient_full_rink_initialization']
