import pytest
from fit_native_multicamera import selected_scope
SHOTS=[dict(id=f's{i+1:02}',camera_index=c)for i,c in enumerate([0,1,2,0,2,1,0,2,0])]

def test_new_shot_motion_scope_does_not_include_other_shots_on_same_rig():
    assert selected_scope(SHOTS,3,'motion',shot_ids=['s07','s08','s09'])==({0,2},{'s07','s08','s09'})

def test_full_joint_default_still_contains_every_camera_and_shot():
    assert selected_scope(SHOTS,3,'joint')==({0,1,2},{s['id']for s in SHOTS})

@pytest.mark.parametrize('phase,cameras,shots',[('joint',None,['s09']),('motion',None,['s10']),('motion',None,['s09','s09']),('motion',[1],['s09']),('motion',None,[])])
def test_invalid_or_reduced_joint_scopes_are_rejected(phase,cameras,shots):
    with pytest.raises(ValueError):selected_scope(SHOTS,3,phase,cameras,shots)
