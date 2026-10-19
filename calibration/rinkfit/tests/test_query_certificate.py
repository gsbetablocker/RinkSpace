import math
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np

from rinkfit_start.native_camera import NativeCamera
from rinkfit_start.native_query_certificate import (
    Interval, _exp, _log_positive_point, _sincos,
    certify_profile_roots,
)


def _decimal_trig(value, cosine=False):
    with localcontext() as context:
        context.prec=80
        x=Decimal.from_float(value)
        term=Decimal(1) if cosine else x
        total=term
        for k in range(1,100):
            j=2*k if cosine else 2*k+1
            term=-term*x*x/Decimal(j*(j-1))
            total+=term
        return +total


def test_elementary_intervals_contain_independent_high_precision_values():
    values=np.array([-4.,-math.pi,-1.,-.125,0.,.125,1.,math.pi,4.])
    sine,cosine=_sincos(Interval(values))
    for i,value in enumerate(values):
        assert Decimal.from_float(float(sine.lo[i])) <= _decimal_trig(value) <= Decimal.from_float(float(sine.hi[i]))
        assert Decimal.from_float(float(cosine.lo[i])) <= _decimal_trig(value,True) <= Decimal.from_float(float(cosine.hi[i]))
    values=np.array([-20.,-1.,0.,.125,1.,10.,20.])
    result=_exp(Interval(values))
    with localcontext() as context:
        context.prec=80
        for i,value in enumerate(values):
            expected=Decimal.from_float(float(value)).exp()
            assert Decimal.from_float(float(result.lo[i]))<=expected<=Decimal.from_float(float(result.hi[i]))
    for value in [.1,1.,2.,100.,12345.]:
        result=_log_positive_point(value)
        with localcontext() as context:
            context.prec=80
            expected=Decimal.from_float(value).ln()
            assert Decimal.from_float(float(result.lo))<=expected<=Decimal.from_float(float(result.hi))


def _synthetic():
    scene=SimpleNamespace(values=[np.array([-1.,0.,1.,0.])],path=lambda m,p:dict(kind=0,geometry_block=0))
    controls=np.zeros((10,3));controls[:,2]=math.log(100.)
    camera=NativeCamera(dict(shared_head=dict(R0=np.eye(3).tolist(),n=[0.,1.,0.],b=[1.,0.,0.],P=[0.,0.,-2.],d=[0.,0.,0.],a=0.),
        camera_controls=dict(basis=dict(dt=.02,first_control_time=-.04,n_controls=10),controls=controls.tolist()),
        optics=dict(f_ref=100.,principal_point=[960.,540.],k1=[0.,0.],k2=[0.,0.],p1=0.,p2=0.,radius_max=2.),
        rolling_shutter=dict(signed_readout_seconds=.02)))
    curves=dict(numerator_u=np.array([[960.,0.,0.,0.]]),numerator_v=np.array([[540.,1.,0.,0.]]),denominator=np.array([[1.,0.,0.,0.]]),bounds=np.array([[-20.,20.]]))
    return scene,camera,curves


def test_krawczyk_proves_transverse_root_and_encloses_analytic_jacobian():
    scene,camera,curves=_synthetic()
    result=certify_profile_roots(scene,camera,[0],[0],[.5],[0.],curves,[0.],[540.])
    assert result['certified'].tolist()==[True]
    true_jacobian=np.diag([100.,-1.,1.])
    assert np.all(result['jacobian_lower'][0]<=true_jacobian)
    assert np.all(true_jacobian<=result['jacobian_upper'][0])
    assert np.all(result['krawczyk_lower'][0]>result['box_lower'][0])
    assert np.all(result['krawczyk_upper'][0]<result['box_upper'][0])
    assert result['preconditioned_jacobian_contraction_upper'][0]<1


def test_wrong_center_parallel_query_and_finite_boundary_do_not_certify():
    scene,camera,curves=_synthetic()
    wrong=certify_profile_roots(scene,camera,[0],[0],[.6],[0.],curves,[0.],[540.])
    assert not wrong['certified'][0]
    parallel=dict(curves,numerator_u=np.array([[960.,1.,0.,0.]]),numerator_v=np.array([[540.,0.,0.,0.]]))
    singular=certify_profile_roots(scene,camera,[0],[0],[.5],[0.],parallel,[0.],[540.])
    assert not singular['certified'][0]
    finite=certify_profile_roots(scene,camera,[0],[0],[0.],[0.],curves,[0.],[540.])
    assert not finite['certified'][0]
    assert finite['reason'][0]=='local_box_crosses_finite_station_alpha_or_row_boundary'


def test_clock_knot_box_is_explicitly_uncertified():
    scene,camera,curves=_synthetic()
    result=certify_profile_roots(scene,camera,[0],[0],[.5],[0.],curves,[0.],[539.5])
    assert not result['certified'][0]
    assert result['reason'][0]=='cubic_knot_or_allocated_time_domain_crossing'


def test_nonsingular_equation_root_in_folded_Brown_region_does_not_certify():
    scene,camera,curves=_synthetic()
    camera.optics['k1']=[-10.,0.]
    curves=dict(curves,numerator_u=np.array([[969.375,0.,0.,0.]]))
    result=certify_profile_roots(scene,camera,[0],[0],[.75],[0.],curves,[0.],[540.])
    assert not result['certified'][0]
    assert result['reason'][0]=='Brown_no_fold_entire_root_box_uncertified'
    assert result['box_brown_no_fold_lower'][0]<0
