from types import SimpleNamespace

import numpy as np
import pytest

from calibration.program_tracking.brown_inverse_rows import brown_central_inverse_rows
from rinkspace.calibration_companion.wide_angle_camera import brown_central_inverse


def scalar(K,D,uv,**options):
    rows=[brown_central_inverse(SimpleNamespace(K_px=k,distortion_opencv=d),p[None],**options)
        for k,d,p in zip(K,D,uv,strict=True)]
    return {key:np.concatenate([row[key]for row in rows])for key in rows[0]}


def same(actual,expected):
    assert set(actual)==set(expected)
    for key in actual:
        assert actual[key].dtype==expected[key].dtype
        assert actual[key].shape==expected[key].shape
        assert actual[key].tobytes()==expected[key].tobytes(),key


@pytest.mark.parametrize('tolerance',[1e-5,1e-9])
def test_same_original_algorithm_with_distinct_lenses_and_invalid_rows(tolerance):
    rng=np.random.default_rng(1017);n=24
    K=np.repeat(np.eye(3)[None],n,axis=0)
    K[:,0,0]=rng.uniform(400.,3500.,n);K[:,1,1]=rng.uniform(500.,3100.,n)
    K[:,0,2]=rng.uniform(700.,1100.,n);K[:,1,2]=rng.uniform(400.,650.,n)
    D=np.zeros((n,12));D[:,:5]=rng.normal(size=(n,5))*[.05,.01,.002,.002,.001]
    D[:,8:]=rng.normal(size=(n,4))*.0001
    uv=rng.uniform([0.,0.],[1920.,1080.],size=(n,2))
    # Distinct rational poles/extrema keep the untouched scalar implementation.
    D[4,5:8]=[-.8,.1,.01];D[8,5:8]=[.05,-.01,.003]
    D[12,0]=-2.;uv[12]=[5000.,5000.]
    D[16]=0.;uv[16]=1e8
    uv[17]=np.nan;uv[18]=np.inf;uv[19]=1e200
    options=dict(tolerance_px=tolerance,continuation_steps=8,max_iterations=24)
    with np.errstate(all='ignore'):
        expected=scalar(K,D,uv,**options)
        actual=brown_central_inverse_rows(K,D,uv,**options)
        same(actual,expected)
        order=rng.permutation(n)
        shuffled=brown_central_inverse_rows(K[order],D[order],uv[order],**options)
    same({key:value[np.argsort(order)]for key,value in shuffled.items()},expected)
    assert actual['valid'].any()and not actual['valid'].all()


def test_one_exact_lens_per_original_pixel_required():
    with pytest.raises(ValueError,match='One exact K'):
        brown_central_inverse_rows(np.eye(3),np.zeros((2,12)),np.zeros((2,2)))


def test_zero_focal_keeps_original_scalar_exception_semantics():
    K=np.eye(3)[None];K[0,0,0]=0.;K[0,1,1]=0.
    D=np.zeros((1,12));uv=np.ones((1,2))
    with np.errstate(all='ignore'):
        with pytest.raises(ZeroDivisionError):scalar(K,D,uv,tolerance_px=1e-9)
        with pytest.raises(ZeroDivisionError):brown_central_inverse_rows(K,D,uv,tolerance_px=1e-9)
