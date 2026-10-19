import numpy as np
from rinkfit_start.atlas_upgrade import estimate_chart


def test_unknown_Q_is_estimated_in_its_actual_chart():
    Q=np.array([[1.2,.3,340.],[.1,.8,120.],[.0002,-.0003,1.]])
    A=[np.array([[1000.,0,960.],[0,1000.,540.],[.1*k,.05*k,1.]]) for k in (0,1,2)]
    H=[a@np.linalg.inv(Q) for a in A]
    result=estimate_chart(H,A);actual=np.array(result['Q_metric_to_component_atlas']);actual/=actual[2,2]
    np.testing.assert_allclose(actual,Q,atol=1e-7)
    assert not result['metric_truth_established']
