import copy
import json
import numpy as np
import pytest
from calibration.program_tracking.body_tracking.body_source_revision import revise_body_observations, reassess_lower_limb_quality_population


def fixture():
    pts=np.array([10,20,30],dtype=np.int64)
    ids=np.tile(np.arange(17),(3,1));ids[0,13:17]=[14,13,16,15]
    obs=dict(pts=pts,uv=np.zeros((3,17,2)),sigma_px=np.ones((3,17)),confidence=np.ones((3,17)),use=np.ones((3,17),bool),model_joint_ids=ids)
    obs['body_joint_assignment_provenance_json']=np.array(json.dumps(dict(kind='source_observation_correspondence',source_measurements_unchanged=True,source_pts=[10])))
    exclusions=[dict(source_pts=20,joint_id=14,reason='source knee occluded'),dict(source_pts=20,joint_id=16,reason='source ankle occluded')]
    updated,revision=revise_body_observations(obs,[],exclusions,{'source_review':'immutable image receipt'})
    baseline=dict(source={'shoe':[2.,4.]},torso=dict(observations=11,pixel_p95=9.),turn_episodes={'spin/torso':dict(observations=12,pixel_p95=8.),'spin/lower_limbs':dict(observations=12,pixel_p95=7.)},flight_guard={'caps':{'H':.3}},stage_reset_allowed=False)
    projection=np.arange(3*17*2,dtype=float).reshape(3,17,2)
    kwargs=dict(reference_projection=projection,reference_pts=pts,turn_source_pts={'spin':pts},reference_identity={'path':'saved.npz','sha256':'immutable'})
    return baseline,obs,updated,revision,kwargs


def call(values):
    baseline,obs,updated,revision,kwargs=values
    return reassess_lower_limb_quality_population(baseline,obs,updated,{'path':'new_obs.npz','sha256':'new'},revision,**kwargs)


def test_same_pose_population_reassessment_keeps_numeric_limits_and_flight_caps():
    values=fixture();original=copy.deepcopy(values[0]);result=call(values)
    assert values[0]==original
    for key in ('source','torso','flight_guard','stage_reset_allowed'):assert result[key]==original[key]
    assert result['turn_episodes']['spin/lower_limbs']==dict(observations=10,pixel_p95=7.)
    assert result['turn_episodes']['spin/torso']==original['turn_episodes']['spin/torso']
    replay=result['body_source_revision']['same_pose_replay']['spin/lower_limbs']
    projected=values[4]['reference_projection'][np.arange(3)[:,None],values[1]['model_joint_ids']]
    pixels=np.linalg.norm(projected-values[1]['uv'],axis=-1)
    assert replay['same_saved_pose_before']['pixel_p95']==np.percentile(pixels[:,13:17],95)
    assert replay['same_saved_pose_after']['observations']==10
    assert replay['retained_limit_px']==9.
    assert not result['body_source_revision']['accuracy_improvement_claim']
    assert result['body_source_revision']['same_pose_torso_replay']['observations']==12
    assert result['torso']['observations']==11


def test_replay_handles_reviewed_bilateral_mapping_and_mixed_exits_in_one_packet():
    baseline,obs,_,_,kwargs=fixture()
    obs['uv'][1,15]=kwargs['reference_projection'][1,16]
    updated,revision=revise_body_observations(obs,[dict(source_pts=20,joint_pairs=[[15,16]])],
        [dict(source_pts=30,joint_id=j,reason='same-frame duplicate in source') for j in (9,15)],{})
    result=call((baseline,obs,updated,revision,kwargs))
    revised=result['body_source_revision']
    assert revised['excluded_lower_limb_rows']==1
    assert revised['excluded_rows']==2
    replay=revised['same_pose_replay']['spin/lower_limbs']
    assert replay['surviving_rows']==11 and replay['retired_rows']==1 and replay['retained_limit_px']==9.
    assert replay['same_saved_pose_before']['observations']==12
    assert replay['same_saved_pose_after']['observations']==11
    frame=np.arange(3)[:,None]
    mask=updated['use'][:,13:17]
    errors=np.linalg.norm(kwargs['reference_projection'][frame,updated['model_joint_ids']]-obs['uv'],axis=-1)
    assert replay['same_saved_pose_after']['pixel_p95']==np.percentile(errors[:,13:17][mask],95)


@pytest.mark.parametrize('failure',['raw_uv','identity','undeclared','added','count','clock','missing_episode','nonfinite','reference_identity','non_native_turn'])
def test_reassessment_rejects_unbound_or_changed_populations(failure):
    values=list(fixture());baseline,obs,updated,revision,kwargs=values
    if failure=='raw_uv':updated['uv'][1,14,0]+=1.
    elif failure=='identity':updated['model_joint_ids'][1,13:15]=[14,13]
    elif failure=='undeclared':revision['derived_use_exclusions'].pop()
    elif failure=='added':obs['use'][0,1]=False
    elif failure=='count':baseline['turn_episodes']['spin/lower_limbs']['observations']=13
    elif failure=='clock':kwargs['reference_pts']=np.array([10,21,30])
    elif failure=='missing_episode':kwargs['turn_source_pts']={}
    elif failure=='nonfinite':kwargs['reference_projection'][0,14,0]=np.nan
    elif failure=='reference_identity':kwargs['reference_identity']={}
    elif failure=='non_native_turn':kwargs['turn_source_pts']={'spin':np.array([10,21,30])}
    with pytest.raises((ValueError,AssertionError)):call(values)
