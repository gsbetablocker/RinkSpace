from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from calibration.program_tracking.body_tracking.action_live import GlideActionFactors


def test_collision_mesh_runner_semantic_does_not_supply_material_tangent():
    factor=GlideActionFactors.__new__(GlideActionFactors)
    arc=dict(center=[0.,0.,0.],radius_m=1.,axis_cos=[1.,0.,0.],axis_sin=[0.,0.,1.],semantic='runner')
    factor.binding=SimpleNamespace(ratio=1.,packet=lambda:dict(template=dict(contact_arcs=[arc])))
    state=dict(R=np.broadcast_to(np.eye(3),(2,2,3,3)).copy(),O=np.zeros((2,2,3)))
    # The collision mesh guard is lower than the exact arc and also uses
    # runner semantics. It must never become a zero blade-motion tangent.
    q=np.broadcast_to([[1.,0.,0.],[0.,0.,-2.]],(2,2,2,3)).copy()
    features=dict(semantics=['runner','runner'],height=np.broadcast_to([0.,-2.],(2,2,2)),local_point=q)
    with patch('calibration.program_tracking.body_tracking.action_live.batch_features',return_value=features):
        result=factor.features(state).reshape(2,2,18)
    np.testing.assert_allclose(result[:,:,12:15],np.broadcast_to([1.,0.,0.],(2,2,3)))
    np.testing.assert_allclose(np.linalg.norm(result[:,:,15:18],axis=-1),1.)
