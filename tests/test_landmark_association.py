from copy import deepcopy
import numpy as np

from rinkspace.landmark_association import (
    AssociationConfig, associate_frame, candidate_model_ids, source_complementary_pairs,
)
from rinkspace.multisurface_camera import Camera, project_camera
from rinkspace.landmark_band_profile import point_array_sha256


def fixture():
    camera = Camera(np.array([[1, 0, 0], [0, -.554700196, -.832050294],
                              [0, .832050294, -.554700196]]),
                    np.array([0., -12., 8.]), 700., np.array([480., 270.]))
    angle = np.linspace(0, 2*np.pi, 80, endpoint=False)
    primitives, observations = [], []
    for identity, x in [('F2', -4.), ('F3', 4.)]:
        points = np.c_[x+2*np.cos(angle), 2+2*np.sin(angle)]
        primitives.append({'id':identity,'kind':'circle','points_model':points.tolist()})
        observed, valid = project_camera(camera, np.c_[points,np.zeros(len(points))])
        assert valid.all()
        observations.append({'observation_id':identity+'_image_only','kind':'arc_or_line',
                             'colour_class':'gray','points_source_px':(observed*2).tolist(),
                             'association_status':'detected_unidentified','used_by_solver':False,
                             'measurement_uncertainty_source_px':4.})
    model = {'structure':{'primitives':primitives}}
    frame = {'source_size':[1920,1080],'sample_size':[960,540],
             'landmark_observations':observations,'pts':100,'time_base':'1/50'}
    return camera, model, frame


def test_two_real_visible_curves_receive_joint_candidate_ids_and_capped_physical_weights():
    camera, model, frame = fixture()
    original = deepcopy(frame)
    result = associate_frame(frame,model,[('known_same_event_reference',camera)],
                             config=AssociationConfig(maximum_hypotheses=12,maximum_fit_evaluations=25))
    chosen = [o for o in result['landmark_observations'] if o['selected_model_id']]
    assert {o['selected_model_id'] for o in chosen} == {'F2','F3'}
    assert all(o['used_by_solver'] is False for o in chosen)
    assert all(o['association_status']=='associated' for o in chosen)
    for identity in {'F2','F3'}:
        assert sum(o['group_weight'] for o in chosen if o['selected_model_id']==identity)<=1.000001
    assert result['association']['manual_intervention_inputs_read'] is False
    assert frame==original


def test_one_circle_does_not_claim_full_rink_camera_and_frozen_blue_does_not_enter_fit():
    camera, model, frame = fixture()
    frame['landmark_observations']=frame['landmark_observations'][:1]
    frame['landmark_observations'].append({'observation_id':'blue', 'kind':'line','colour_class':'blue',
        'points_source_px':[[20,300],[60,300],[100,300],[140,300],[180,300],[220,300],[260,300],[300,300]],
        'association_status':'detected_unidentified'})
    result=associate_frame(frame,model,[('reference_without_current_pose',camera)])
    assert result['association']['best_camera_candidate'] is None
    assert all(o['selected_model_id'] is None for o in result['landmark_observations'])
    blue=result['landmark_observations'][-1]
    assert blue['rejection_stage']=='physical_group_holdout'
    assert blue['candidate_model_ids']==['blue_left','blue_right']


def test_gray_flat_arc_can_propose_circles_but_masked_pixels_cannot_identify_one():
    camera, model, frame = fixture()
    assert set(candidate_model_ids({'kind':'line','colour_class':'gray'},model))=={'F2','F3'}
    frame['landmark_observations'][0]['association_status']='masked_out'
    result=associate_frame(frame,model,[('known_reference',camera)])
    assert result['association']['best_camera_candidate'] is None
    assert result['landmark_observations'][0]['association_status']=='masked_out'


def test_no_seed_still_records_detector_evidence_and_rejection_without_inventing_a_pose():
    _, model, frame=fixture()
    result=associate_frame(frame,model,[])
    assert result['association']['hypotheses_attempted']==0
    assert result['association']['best_camera_candidate'] is None
    assert result['landmark_observations'][0]['points_source_px']==frame['landmark_observations'][0]['points_source_px']
    assert result['landmark_observations'][0]['rejection_reason']=='no_multi_structure_camera_explanation'


def test_one_consistent_circle_can_correct_existing_graph_without_new_initialization():
    camera, model, frame=fixture()
    frame['landmark_observations']=frame['landmark_observations'][:1]
    result=associate_frame(frame,model,[('existing_anchored_camera',camera)],
                           allow_partial_existing_pose=True,
                           config=AssociationConfig(maximum_hypotheses=8,maximum_fit_evaluations=20))
    assert result['landmark_observations'][0]['selected_model_id']=='F2'
    assert result['landmark_observations'][0]['used_by_solver'] is False
    assert result['landmark_camera_candidate'] is not None
    assert result['landmark_initialization_accepted'] is False
    assert result['association']['candidate_hypotheses'][0]['partial_correction_requires_existing_graph_anchor'] is True


def test_source_complementary_proposals_depend_on_pixels_not_prediction_distance_or_ids():
    x=np.linspace(100,900,50)
    a=np.c_[x,700+.0001*(x-500)**2]
    b=np.c_[x,800-.0001*(x-500)**2]
    observations=[{'observation_id':'anonymous_lower','kind':'arc_or_line','colour_class':'gray',
                   'points_source_px':a.tolist(),'association_status':'detected_unidentified'},
                  {'observation_id':'anonymous_upper','kind':'line','colour_class':'gray',
                   'points_source_px':b.tolist(),'association_status':'detected_unidentified'}]
    pairs=source_complementary_pairs(observations)
    assert pairs[0]['observation_indices']==(0,1)
    assert pairs[0]['source_overlap_fraction']==1
    assert pairs[0]['source_separation_px']>50
    changed=deepcopy(observations)
    for row in changed:
        row.update(observation_id='different',selected_model_id='wrong_prediction',
                   predicted_distance_sample_px=1000000)
    assert source_complementary_pairs(changed)==pairs
    changed[1]['association_status']='masked_out'
    assert source_complementary_pairs(changed)==[]


def test_source_brightness_prior_prefers_opposite_outer_sides_and_preserves_unknown():
    x=np.linspace(100,900,40)
    observations=[];profiles={}
    for number,y,gradient in [(0,700,[0,-1]),(1,800,[0,1])]:
        points=np.c_[x,np.full(len(x),y)]
        identity=str(number)
        observations.append({'observation_id':identity,'colour_class':'gray','points_source_px':points.tolist()})
        profiles[identity]={'point_array_sha256':point_array_sha256(points),
                            'gradient_reliable':[True]*len(x),
                            'gradient_unit_source_xy':[gradient]*len(x)}
    pair=source_complementary_pairs(observations,profiles=profiles)[0]
    assert pair['source_pair_brightness']==['source_pair_outward_brightness_supported']*2
    assert pair['source_ranking_score']==3*pair['source_span_score']
    profiles['0']['gradient_reliable']=[False]*len(x)
    unknown=source_complementary_pairs(observations,profiles=profiles)[0]
    assert unknown['source_pair_brightness'][0]=='unknown'
    assert not unknown['source_pair_brightness_conflict']
    assert unknown['source_ranking_score']>=unknown['source_span_score']
