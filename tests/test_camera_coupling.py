"""The fixed delivery-mode model shares lens functions, not scene coordinates."""
import numpy as np
import pytest
from rinkspace.domain.camera_coupling import build_multicamera_directions,change_zoom_reference
from rinkspace.domain.camera_experiment import CAMERA_NAMES


def rigs_and_coupling():
    rigs=np.zeros((3,43));rigs[:,:9]=np.eye(3).ravel();rigs[:,9:12]=[0,0,1]
    rigs[:,12:15]=[1,0,0];rigs[:,24]=[10000,5000,7000];rigs[:,25]=1
    rigs[:,26:28]=[959.5,539.5];rigs[:,38]=np.log([8000,4000,6000])
    rigs[:2,[31,32,33,34,39,40]]=[.1,.2,.3,.4,.5,.6]
    group=dict(id='radial',camera_indices=[0,1],shared_names=['k10','k11','k20','k21','wide_k1','wide_k2'],
        delivery_magnifications=[2,1],reference_focal_base_pixels=5000,wide_boundary_base_log_f=np.log(4000))
    return rigs,dict(groups=[group])


def test_same_physical_focal_uses_the_same_radial_function():
    rigs,config=rigs_and_coupling();active=[tuple(range(31))]*3
    rd,meta=build_multicamera_directions(rigs,active,np.ones(31),config)
    assert rd.shape==(93,87)
    step=np.linspace(-.01,.01,87);local=(rd@step).reshape(3,31)
    for storage,chart in [(31,20),(32,21),(33,22),(34,23),(39,27),(40,28)]:
        rigs[:,storage]+=local[:,chart]
    physical_focal=np.array([2500.,4500.,9000.])
    functions=[]
    for i,d in [(0,2.),(1,1.)]:
        f=physical_focal*d;z=np.log(f/rigs[i,24]);w=np.maximum(rigs[i,38]-np.log(f),0.)**2
        functions.append(np.c_[rigs[i,31]+rigs[i,32]*z+rigs[i,39]*w,
                                rigs[i,33]+rigs[i,34]*z+rigs[i,40]*w])
    np.testing.assert_allclose(functions[0],functions[1],atol=1e-14)
    # Unit-specific tangential terms retain separate directions.
    assert not np.array_equal(rd[24],rd[31+24])


def test_shared_manifold_and_mode_mismatch_are_rejected():
    rigs,config=rigs_and_coupling();active=[tuple(range(31))]*3
    bad=rigs.copy();bad[1,31]+=.001
    with pytest.raises(ValueError,match='manifold'):build_multicamera_directions(bad,active,np.ones(31),config)
    bad=rigs.copy();bad[0,24]+=1
    with pytest.raises(ValueError,match='reference'):build_multicamera_directions(bad,active,np.ones(31),config)


def test_reference_change_preserves_complete_forward_and_optics():
    rigs,_=rigs_and_coupling();old=rigs[0].copy()
    old[20:24]=[.001,.0005,.0011,.0007];old[29:31]=[2,-3]
    new=change_zoom_reference(old,7300.)
    f=np.geomspace(3000.,20000.,20);a=np.log(f/old[24]);b=np.log(f/new[24])
    np.testing.assert_allclose(old[20]+old[22]*a+old[23]*a*a,new[20]+new[22]*b+new[23]*b*b,atol=1e-16)
    for base,slope in [(26,29),(27,30),(31,32),(33,34)]:
        np.testing.assert_allclose(old[base]+old[slope]*a,new[base]+new[slope]*b,atol=1e-13)
    assert new[38]==old[38]
