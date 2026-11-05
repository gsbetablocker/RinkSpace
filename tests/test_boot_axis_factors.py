import numpy as np
import pytest
from calibration.program_tracking.body_tracking.boot_axis_factors import axis_region,finite_axis_term


def test_unoriented_axis_and_finite_band():
    centre,half=axis_region([0.,0.],[100.,0.],[3.,4.],[3.,4.])
    assert half>0
    for heel,toe in [(np.array([0.,0.]),np.array([100.,0.])),(np.array([100.,0.]),np.array([0.,0.]))]:
        r,j,error=finite_axis_term(heel,toe,centre,half,2.)
        assert abs(r)<1e-12
        np.testing.assert_array_equal(j,np.zeros(4))
        assert error<1e-12


def test_axis_pixel_jacobian_including_heel_and_toe():
    x=np.array([12.,7.,56.,83.]);centre,half=axis_region([0.,0.],[100.,0.],[2.,3.],[2.,3.])
    r,j,_=finite_axis_term(x[:2],x[2:],centre,half,3.5)
    assert r>0
    h=1e-5;n=[]
    for i in range(4):
        d=np.zeros(4);d[i]=h
        a=x+d;b=x-d
        n.append((finite_axis_term(a[:2],a[2:],centre,half,3.5)[0]-finite_axis_term(b[:2],b[2:],centre,half,3.5)[0])/(2*h))
    np.testing.assert_allclose(j,n,rtol=1e-7,atol=1e-8)


def test_overlapping_source_regions_do_not_define_axis():
    with pytest.raises(ValueError,match='unobservable'):
        axis_region([0.,0.],[2.,0.],[4.,4.],[4.,4.])
