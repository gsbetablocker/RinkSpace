"""A cached seed J must not consume the full-J reserve twice."""
import numpy as np
from calibration.program_tracking.body_tracking.skate_run_policy import jacobian_reserve_seconds


def test_same_state_cached_j_can_start_with_less_than_one_cold_j_remaining():
    x=np.array([1.,2.]);cache={'jac':np.eye(2)}
    assert jacobian_reserve_seconds(x,cache,x.copy(),138.)==0.
    assert jacobian_reserve_seconds(x,cache,x+np.array([0.,1e-10]),138.)==138.
    assert jacobian_reserve_seconds(x,{'jac':None},x,138.)==138.
    assert jacobian_reserve_seconds(None,None,x,138.)==138.
