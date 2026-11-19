import numpy as np

from calibration.program_tracking.body_tracking.programme_skate_evidence import native_point_response


def test_detector_response_derivative_at_zero_and_large_pixel_errors():
    error=np.array([[0.,0.],[.00001,-.00002],[9.,12.],[18.,0.],[220.,-160.]])
    radius=np.array([6.,6.,8.,6.,6.]);weight=np.array([1.,2.,1.,3.,1.])
    for choice in (np.zeros(5,bool),np.ones(5,bool),np.array([False,True,False,True,True])):
        value,J=native_point_response(error,radius,weight,cauchy=choice)
        assert np.isfinite(value).all() and np.isfinite(J).all()
        for axis in range(2):
            delta=np.zeros_like(error);delta[:,axis]=1.e-5
            numerical=(native_point_response(error+delta,radius,weight,cauchy=choice)[0]-
                       native_point_response(error-delta,radius,weight,cauchy=choice)[0])/2.e-5
            np.testing.assert_allclose(J[:,:,axis],numerical,rtol=2.e-6,atol=1.e-8)


def test_gross_detector_error_loses_influence_without_dropping_the_row():
    error=np.array([[18.,0.],[180.,0.],[1800.,0.]])
    radius=np.full(3,6.);weight=np.ones(3)
    value,J=native_point_response(error,radius,weight,cauchy=np.ones(3,bool))
    influence=np.einsum('ni,nij->nj',value,J)[:,0]
    assert np.all(influence>0) and np.all(np.diff(influence)<0)
    assert np.all(np.diff(np.sum(value*value,axis=1))>0)
    old,_=native_point_response(error,radius,weight)
    expected=error/radius[:,None]/(1+np.sum(error*error,axis=1)/(9*radius**2))[:,None]**.25
    np.testing.assert_allclose(old,expected,rtol=1.e-15,atol=1.e-15)
