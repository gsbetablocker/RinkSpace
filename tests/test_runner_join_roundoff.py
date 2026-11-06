import numpy as np
from calibration.program_tracking.body_tracking.runner_interior import canonical_runner_witness,RunnerCoordinateDomainError
from calibration.program_tracking.body_tracking.skate_template import SkateParameters,_profile,profile_height
import pytest


def test_join_height_tie_uses_convex_stationary_coordinate():
    p=SkateParameters();main,secondary,_=_profile(p);theta=main['theta_end_rad']+1e-9
    n=np.array([[-np.sin(theta),0.,np.cos(theta)]]);x=main['center'][0]+main['radius_m']*np.sin(main['theta_end_rad'])
    old=np.array([[x,0.,profile_height(np.array([x]),p)[0]]]);point,c=canonical_runner_witness(n,p,old)
    expected=secondary['center'][0]+secondary['radius_m']*np.sin(theta)
    np.testing.assert_allclose(point[0,0],expected,atol=1e-15,rtol=0)
    assert point[0,0]>old[0,0]


def test_genuinely_wrong_witness_is_not_silently_accepted():
    p=SkateParameters();wrong=np.array([[.23,0.,.1]])
    with pytest.raises(RunnerCoordinateDomainError):canonical_runner_witness(np.array([[0.,0.,1.]]),p,wrong)
