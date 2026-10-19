import copy

import numpy as np
import pytest

from rinkspace.athlete_continuity import apply_athlete_continuity, source_points_to_model


def chain(translation=0., radial=None):
    return {'H_reference_undistorted_from_model': [[10.,0.,500.],[0.,10.,500.],[0.,0.,1.]],
            'H_current_sample_from_reference_raw': [[1.,0.,translation],[0.,1.,0.],[0.,0.,1.]],
            'H_source_image_from_sample': np.eye(3).tolist(),
            'reference_visible_branch_sign':1,'raw_transfer_visible_branch_sign':1,
            'reference_radial_distortion':radial}


def hypothesis(translation):
    return {'reference_id':f'ref_{translation}','role':'real_anchor_propagated_through_actual_image_edges',
            'projection_chain':chain(translation)}


def fixture():
    frames=[]
    observations=[]
    for i in range(2):
        frames.append({'source_id':'skater','source_sha256':'source','model_sha256':'model',
            'model_id':'event','model_revision':'v07','unit':'normalized_rink_width_100',
            'source_time_s':1+i*.02,'pts':50+i,'time_base':'1/50',
            'shot_id':str(i),'width':1920,'height':1080,'sample_size':[1920,1080],
            'status':'model_reobserved' if i==0 else 'ambiguous',
            'projection_chain':chain() if i==0 else None,'H_model_from_image':None,
            'model_anchor_hypotheses':[] if i==0 else [hypothesis(0.),hypothesis(1000.)],
            'source_shot_metadata':{'role':'programme_candidate'}})
        observations.append({'source_id':'skater','source_sha256':'source','source_time_s':1+i*.02,
            'pts':50+i,'time_base':'1/50','image_size':[1920,1080],
            'coordinate_domain':'raw_source_image_pixels','not_body_bbox_bottom':True,
            'observation_type':'manual_coarse_visible_blade_support_candidate_regions',
            'state':'visible_near_ice_support_candidates_not_exact_contact',
            'support_regions_source_px':[[[500,500],[520,500],[520,520],[500,520]]],
            'manual_confirmed_by_user':False})
    constraints={'schema_version':'rinkspace.coarse-athlete-continuity-evidence.v1',
        'source_time_continuity':{'status':'visually_reviewed_candidate_not_verified_live_clock',
                                'interval_s':[.98,1.1],'cut_s':1.02},'observations':observations}
    return frames,constraints


def test_reviewed_time_candidate_only_softly_disfavors_wrong_end_without_pose_promotion():
    frames,constraints=fixture();original=copy.deepcopy(frames)
    result=apply_athlete_continuity(frames,constraints)
    row=result['frames'][1]
    assert row['athlete_continuity']['soft_disfavored_hypothesis_indices']==[1]
    assert not row['athlete_continuity']['active_hypotheses_pruned']
    assert len(row['model_anchor_hypotheses'])==2
    assert row['projection_chain'] is None and row['status']=='ambiguous'
    assert result['diagnostics']['used_frame_count']==1
    assert not result['diagnostics']['metric_speed_or_precise_trajectory_output']
    assert frames==original


def test_ambiguous_previous_camera_propagates_all_regions_instead_of_picking_first():
    frames,constraints=fixture()
    frames[0].update(status='ambiguous',projection_chain=None,
                     model_anchor_hypotheses=[hypothesis(0.),hypothesis(1000.)])
    frames[1]['model_anchor_hypotheses'].append(hypothesis(2000.))
    result=apply_athlete_continuity(frames,constraints)
    check=result['frames'][1]['athlete_continuity']
    assert check['previous_camera_hypothesis_count']==2
    assert check['soft_retained_hypothesis_indices']==[0,1]
    assert check['soft_disfavored_hypothesis_indices']==[2]
    assert check['hypotheses'][1]['possible_prior_hypothesis_indices']==[1]


def test_confirmed_time_can_prune_but_retains_originals_and_never_solves_camera():
    frames,constraints=fixture()
    constraints['source_time_continuity']['status']='confirmed_programme_continuity'
    result=apply_athlete_continuity(frames,constraints)
    row=result['frames'][1]
    assert len(row['model_anchor_hypotheses'])==1
    assert len(row['model_anchor_hypotheses_before_athlete_filter'])==2
    assert row['status']=='ambiguous' and row['projection_chain'] is None
    assert row['athlete_continuity']['active_hypotheses_pruned']
    assert not result['diagnostics']['camera_pose_solved_from_athlete']


@pytest.mark.parametrize('change,reason',[
    ('unreviewed_time','programme_time_continuity_not_visually_reviewed'),
    ('airborne','airborne_occluded_or_unconfirmed_support'),
    ('bodybox','support_is_not_distinguished_from_body_box'),
    ('replay','replay_slow_motion_or_time_jump_disables_athlete_factor'),
    ('missing_camera','no_camera_hypotheses_for_support_region'),
    ('same_shot','no_distinct_camera_shots_at_reviewed_cut'),
])
def test_missing_or_inappropriate_evidence_returns_unused_without_affecting_camera(change,reason):
    frames,constraints=fixture()
    if change=='unreviewed_time':constraints['source_time_continuity']['status']='programme_candidate'
    if change=='airborne':constraints['observations'][0]['state']='airborne'
    if change=='bodybox':constraints['observations'][0]['not_body_bbox_bottom']=False
    if change=='replay':frames[0]['source_shot_metadata']['role']='replay_candidate'
    if change=='missing_camera':frames[0].update(status='unavailable',projection_chain=None)
    if change=='same_shot':frames[1]['shot_id']=frames[0]['shot_id']
    result=apply_athlete_continuity(frames,constraints)
    assert result['diagnostics']['used_frame_count']==0
    assert reason in str(result['diagnostics'])
    assert all(a['projection_chain']==b['projection_chain'] and a['status']==b['status']
               for a,b in zip(result['frames'],frames))


def test_unknown_previous_hypothesis_disables_exclusion_of_all_future_candidates():
    frames,constraints=fixture()
    bad=hypothesis(1000.);bad['projection_chain']['H_source_image_from_sample']=None
    frames[0].update(status='ambiguous',projection_chain=None,model_anchor_hypotheses=[hypothesis(0.),bad])
    result=apply_athlete_continuity(frames,constraints)
    check=result['frames'][1]['athlete_continuity']
    assert check['unused_reason']=='not_all_previous_camera_hypotheses_have_ground_regions'
    assert not check['used']
    assert len(result['frames'][1]['model_anchor_hypotheses'])==2


def test_radial_chain_inverse_applies_source_scaling_transfer_and_lens_exactly_once():
    radial={'k1':-.08,'centre_image':[500.,500.],'scale_image':1000.}
    projection=chain(100.,radial)
    projection['H_source_image_from_sample']=[[2.,0.,0.],[0.,2.,0.],[0.,0.,1.]]
    model=np.array([[0.,0.],[10.,5.],[-12.,6.]])
    ideal=model*10+500
    delta=ideal-np.array(radial['centre_image'])
    raw=500+delta*(1+radial['k1']*np.sum((delta/radial['scale_image'])**2,axis=1))[:,None]
    pixels=(raw+[100,0])*2
    assert np.allclose(source_points_to_model(projection,pixels),model,atol=1e-8)
    projection['H_reference_undistorted_from_model']=(np.array(projection['H_reference_undistorted_from_model'])*-1).tolist()
    projection['reference_visible_branch_sign']=-1
    assert np.allclose(source_points_to_model(projection,pixels),model,atol=1e-8)


def test_folded_radial_region_is_unused_not_mapped_using_a_fake_raw_homography():
    projection=chain(radial={'k1':-1.,'centre_image':[0.,0.],'scale_image':100.})
    with pytest.raises(ValueError,match='invertible_radial_domain'):
        source_points_to_model(projection,np.array([[100.,100.]]))
