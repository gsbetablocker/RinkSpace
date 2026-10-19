import numpy as np
import pytest

from calibration.solver_runtime import install
install()
from structured_programme_source_candidate.source_step_curvature_tightening import constraints_on_declared_free_chart


class OriginalTruthProducer:
    active=np.array([0,1,3])
    def __init__(self):
        self.calls=[]
        self.lower=np.array([.3,-.7])
        self.response={'nominal_carrier':np.array([.2]),'retained_face_mask':np.ones(2,bool)}
    def constraints(self,q,selected,**options):
        assert set(self.active)<=set(selected)
        self.calls.append(np.array(selected))
        full=np.array([[2.,5.,0.,-3.],[-2.,-5.,0.,3.]])
        return dict(matrix=full[:,selected],lower=self.lower,
                    step_carrier_response=self.response,report={'original_face_order':[7,12]})


def test_fixed_pullback_preserves_original_faces_targets_and_carrier_response():
    producer=OriginalTruthProducer()
    result=constraints_on_declared_free_chart(producer,np.zeros(4),[3,0],[1],full_stroke=True)
    np.testing.assert_array_equal(producer.calls[0],[3,0,1])
    np.testing.assert_array_equal(result['matrix'],[[-3.,2.],[3.,-2.]])
    assert result['lower'] is producer.lower and result['step_carrier_response'] is producer.response
    assert result['report']['original_face_order']==[7,12]
    # A nonzero derivative of a declared fixed coordinate is legitimately
    # multiplied by its exact zero displacement, rather than silently dropped.
    assert result['report']['declared_fixed_tangent_pullback']['original_fixed_derivative_nonzero_count']==2
    step=np.array([.4,-.2])
    lifted=np.array([step[1],0.,0.,step[0]])
    np.testing.assert_array_equal(result['matrix']@step,np.array([[2.,5.,0.,-3.],[-2.,-5.,0.,3.]])@lifted)


def test_missing_nonfixed_dependency_rejected_before_original_producer():
    producer=OriginalTruthProducer()
    with pytest.raises(ValueError,match='nonfixed original truth dependency'):
        constraints_on_declared_free_chart(producer,np.zeros(4),[0],[1])
    assert producer.calls==[]


def test_complete_chart_default_returns_original_objects():
    producer=OriginalTruthProducer()
    result=constraints_on_declared_free_chart(producer,np.zeros(4),[0,1,3])
    assert result['lower'] is producer.lower and result['step_carrier_response'] is producer.response
    assert 'declared_fixed_tangent_pullback' not in result['report']


@pytest.mark.parametrize('selected,fixed', [([0,0,3],[1]),([0,3],[1,1]),([0,3],[4])])
def test_invalid_chart_identity_rejected(selected,fixed):
    with pytest.raises(ValueError,match='Exact distinct'):
        constraints_on_declared_free_chart(OriginalTruthProducer(),np.zeros(4),selected,fixed)
