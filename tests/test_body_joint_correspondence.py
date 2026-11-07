"""A source label repair changes correspondence/Jacobian, never source pixels."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from calibration.program_tracking.body_tracking.fullbody_fit import FullBodyProblem,body_joint_assignment


def test_bilateral_correspondence_uses_target_model_columns_and_original_sigma():
    # Construct only the actual pixel factor, independent of model assets/FK.
    p=FullBodyProblem.__new__(FullBodyProblem)
    p.of=np.array([0]);p.oj=np.array([15]);p.model_oj=np.array([16]);p.frames=np.array([0])
    p.layout=SimpleNamespace(body_landmarks=np.arange(17),point_count=17)
    p.obs={'uv':np.zeros((1,17,2)),'sigma_px':np.ones((1,17))*3.}
    uv=np.zeros((1,17,2));uv[0,15]=[100,200];uv[0,16]=[4,-2]
    before=p.obs['uv'].copy();r,A,_,_=p.body_factors(uv)
    assert np.array_equal(p.obs['uv'],before)
    assert set(A.indices)=={32,33}
    d=np.arange(34).reshape(1,17,2)/100.;h=1e-6
    numeric=(p.body_factors(uv+h*d)[0]-p.body_factors(uv-h*d)[0])/(2*h)
    np.testing.assert_allclose(A@d.ravel(),numeric,atol=2e-9)


def test_assignment_cannot_alias_points_change_joint_kind_or_omit_pts_provenance():
    ids=np.arange(17)[None].copy();ids[:,15:17]=[16,15]
    o={'pts':np.array([256]),'model_joint_ids':ids,'body_joint_assignment_provenance_json':np.array(json.dumps(dict(kind='source_observation_correspondence',source_pts=[256],source_measurements_unchanged=True)))}
    np.testing.assert_array_equal(body_joint_assignment(o),ids)
    o['model_joint_ids']=ids.copy();o['model_joint_ids'][0,15]=15
    with pytest.raises(ValueError):body_joint_assignment(o)
    o['model_joint_ids']=ids;o['body_joint_assignment_provenance_json']=np.array('{}')
    with pytest.raises(ValueError):body_joint_assignment(o)
