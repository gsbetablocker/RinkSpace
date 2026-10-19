import numpy as np

from calibration.solver_runtime.strict_source_dataset_adapter import StrictDatasetFactor
from calibration.solver_runtime.source_direction_probe import qualify_source_direction_records


class CornerFactor(StrictDatasetFactor):
    def __init__(self):
        self.parameter_count=17
        self._evaluation_q=self._world_cache_q=self._camera_cache_q=None
        self._world_cache={};self._derivative_cache={};self._camera_cache={}

    def _produce_world(self,q,target):
        world=np.array([[-1.,0.,0.],[0.,0.,0.],[0.,1.,0.]])
        world[:,:2]+=q[15:17]
        derivative=np.zeros((3,3,2));derivative[:,0,0]=1.;derivative[:,1,1]=1.
        return world,(np.array([15,16]),derivative)

    def _camera(self,q,actual,jacobian):
        camera=np.zeros(25);camera[:15]=q[:15];camera[:3]*=100.;camera[8:11]*=1000.
        return camera,np.arange(15),np.eye(15)


def actual_corner_records(uv,direction=None):
    from calibration.solver_runtime import install
    install()
    factor=CornerFactor();q=np.zeros(17);q[2]=-.1;q[6:8]=np.log(1000.);q[8:10]=[.96,.54]
    row=dict(actual_pts=100,model_target=dict(kind='corner'),uv=np.asarray(uv,float),sigma=np.ones(len(uv)))
    factor._preflight_probe_records={};factor._preflight_probe_label='nominal';factor._prepare_evaluation(q)
    _,matrix=factor._evaluate_row(q,row,True)
    if direction is None:direction=np.r_[np.zeros(15),1.,0.]
    h=1e-6
    for label,offset in (('plus',h),('minus',-h)):
        value=q+offset*direction;factor._prepare_evaluation(value);factor._preflight_probe_label=label
        factor._evaluate_row(value,row,False)
    return factor._preflight_probe_records,matrix,direction,h


def test_polyline_vertex_full_original_branch_J_passes_without_claiming_unavailable_smooth_trial_FD():
    records,matrix,direction,h=actual_corner_records([[960.,539.]])
    result=qualify_source_direction_records(records,matrix,direction,h)
    assert result['passed']
    assert result['unrestricted_changed_queries']==1
    assert not result['unrestricted_stable_FD_available']
    assert result['unrestricted_stable_queries_passed'] is None
    assert result['source_FIT_pixels_removed']==0
    assert result['unrestricted_all_query_FD_maximum_difference']>1.
    assert result['original_nominal_segment_and_clamp_fixed_FD_relative_error']<1e-4


def test_smooth_true_source_trials_and_fixed_branch_both_qualify():
    records,matrix,direction,h=actual_corner_records([[935.,538.]],np.random.default_rng(91).normal(size=17))
    result=qualify_source_direction_records(records,matrix,direction,h)
    assert result['passed'] and result['unrestricted_stable_queries_passed']
    assert result['unrestricted_stable_queries']==1
    assert not result['changed_segment_or_clamp_queries']


def test_wrong_complete_J_still_fails_even_when_true_trials_cross_polyline_clamp():
    records,matrix,direction,h=actual_corner_records([[960.,539.]])
    result=qualify_source_direction_records(records,matrix*1.1,direction,h)
    assert not result['passed']
    assert not result['complete_original_branch_J_passed']
