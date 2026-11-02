"""The fixed delivery-mode model shares lens functions, not scene coordinates."""
import json
import numpy as np
import pytest
from rinkspace.domain.camera_coupling import (
    build_multicamera_directions,change_zoom_reference,read_camera_coupling,
)
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
    bad=rigs.copy();bad[0,38]+=.001
    with pytest.raises(ValueError,match='wide basis'):build_multicamera_directions(bad,active,np.ones(31),config)


def test_reference_change_preserves_complete_forward_and_optics():
    rigs,_=rigs_and_coupling();old=rigs[0].copy()
    old[20:24]=[.001,.0005,.0011,.0007];old[29:31]=[2,-3]
    new=change_zoom_reference(old,7300.)
    f=np.geomspace(3000.,20000.,20);a=np.log(f/old[24]);b=np.log(f/new[24])
    np.testing.assert_allclose(old[20]+old[22]*a+old[23]*a*a,new[20]+new[22]*b+new[23]*b*b,atol=1e-16)
    for base,slope in [(26,29),(27,30),(31,32),(33,34)]:
        np.testing.assert_allclose(old[base]+old[slope]*a,new[base]+new[slope]*b,atol=1e-13)
    assert new[38]==old[38]


def readout_only_coupling():
    return dict(schema='rinkspace.camera_coupling.v1',id='shared_readout',
        hardware_calibrated=False,basis='Explicit engineering comparison, not field calibration',
        groups=[dict(id='timing',camera_indices=[0,1,2],shared_names=['readout'])])


def test_readout_only_group_preserves_distinct_optical_references(tmp_path):
    rigs,_=rigs_and_coupling();rigs[:,37]=.008;before=rigs.copy()
    path=tmp_path/'readout.json';path.write_text(json.dumps(readout_only_coupling()),encoding='utf-8')
    config=read_camera_coupling(path)
    scales=np.linspace(.1,3.1,31);active=[tuple(range(31))]*3
    rd,meta=build_multicamera_directions(rigs,active,scales,config)
    assert rd.shape==(93,91)
    assert meta['unique_camera_columns']==91
    chart=CAMERA_NAMES.index('readout')
    rows=rd[np.arange(3)*31+chart]
    np.testing.assert_array_equal(rows,np.repeat(rows[:1],3,axis=0))
    assert np.count_nonzero(rows[0])==1
    assert rows[0].max()==scales[chart]
    step=np.linspace(-.01,.01,91);local=(rd@step).reshape(3,31)
    np.testing.assert_array_equal(local[:,chart],np.repeat(local[0,chart],3))
    np.testing.assert_array_equal(rigs,before)
    assert meta['shared_coordinates']==[
        dict(group='timing',parameter='readout',cameras=[0,1,2],active=True)]
    # Unrelated optical directions stay camera-local.
    assert not np.array_equal(rd[20],rd[31+20])


def test_shared_readout_can_be_inactive_for_all_cameras():
    rigs,_=rigs_and_coupling();rigs[:,37]=.008
    active=[(20,)]*3
    rd,meta=build_multicamera_directions(rigs,active,np.ones(31),readout_only_coupling())
    assert rd.shape==(93,3)
    assert not meta['shared_coordinates'][0]['active']
    assert not rd[np.arange(3)*31+CAMERA_NAMES.index('readout')].any()


@pytest.mark.parametrize('active_readout',[True,False])
def test_shared_readout_requires_exact_initial_equality(active_readout):
    rigs,_=rigs_and_coupling();rigs[:,37]=.008
    rigs[2,37]=np.nextafter(rigs[2,37],np.inf)
    chart=CAMERA_NAMES.index('readout');active=[(chart,) if active_readout else ()]*3
    with pytest.raises(ValueError,match='manifold: readout'):
        build_multicamera_directions(rigs,active,np.ones(31),readout_only_coupling())


@pytest.mark.parametrize('selected_cameras',[(0,),(0,1)])
def test_shared_readout_rejects_partial_activation(selected_cameras):
    rigs,_=rigs_and_coupling();rigs[:,37]=.008
    chart=CAMERA_NAMES.index('readout')
    active=[(chart,) if i in selected_cameras else () for i in range(3)]
    with pytest.raises(ValueError,match='Select all linked cameras'):
        build_multicamera_directions(rigs,active,np.ones(31),readout_only_coupling())


@pytest.mark.parametrize('names',[['k10'],['readout','k10']])
@pytest.mark.parametrize('missing',[
    'delivery_magnifications','reference_focal_base_pixels','wide_boundary_base_log_f'])
def test_optical_and_mixed_groups_still_require_optical_conventions(tmp_path,names,missing):
    _,legacy=rigs_and_coupling();config=readout_only_coupling()
    group=legacy['groups'][0];group['shared_names']=names;del group[missing]
    config['groups']=[group]
    path=tmp_path/'optics.json';path.write_text(json.dumps(config),encoding='utf-8')
    with pytest.raises(KeyError,match=missing):read_camera_coupling(path)


@pytest.mark.parametrize('names',[['k10'],['readout','k10']])
@pytest.mark.parametrize(('storage','message'),[(24,'reference'),(38,'wide basis')])
def test_optical_and_mixed_groups_still_reject_misaligned_references(names,storage,message):
    rigs,config=rigs_and_coupling();config['groups'][0]['shared_names']=names
    rigs[0,storage]+=.001
    with pytest.raises(ValueError,match=message):
        build_multicamera_directions(rigs,[tuple(range(31))]*3,np.ones(31),config)
