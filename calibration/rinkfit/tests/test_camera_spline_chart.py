from copy import deepcopy
import numpy as np
import pytest
from rinkfit_start.camera_spline_chart import UniformCubicTimeChart,cardinal_refinement,cubic_design
from rinkfit_start.ad_chart import PrintOnlyAdChart
from rinkfit_start.native_scene import camera_from_event


def model_fixture(n=52):
    g=np.zeros(32);g[:5]=[.5,.12,.08,1.91,.02];g[11:14]=[0.,0.,.05];g[20:22]=[960.,540.]
    blocks=[dict(name='global32',values=g.tolist(),scales=[1.]*32,constant=True)]
    for i in range(n):
        blocks.append(dict(name=f'control/{i:04d}',values=[.01*np.sin(.4*i)+.003*(-1)**i,.008*np.cos(.3*i),np.log(1000)+.1*np.sin(.1*i)],
            scales=[.1,.1,1.],lower=[-20,-20,np.log(100)],upper=[20,20,np.log(1e6)]))
    blocks[3]['values'][:2]=[0.,0.];blocks[3]['fixed_indices']=[0,1]
    ai=len(blocks);blocks.append(dict(name='ad/8',values=[.3,.1,.001,.018],scales=[1,.1,.02,.02],lower=[-1,.01,0,0],upper=[1,.4,.02,.02]))
    qi=len(blocks);blocks.append(dict(name='print/example',values=[1,0,0,0,1,0],scales=[1.]*6,lower=[-3,-2,-1,-2,-3,-1],upper=[3,2,1,2,3,1]))
    return dict(blocks=blocks,camera=dict(R0_seed=np.eye(3).ravel().tolist(),n_seed=[0,0,1],b_seed=[1,0,0],n_tangent1=[1,0,0],n_tangent2=[0,1,0],
        f_ref=1000.,radius_max=.4,dt=.02,first_control_time=-.04,n_controls=n,control_block_begin=1,dependent_control=2,reference_indices=[1,2,3,4],reference_weights=[1/6,4/6,1/6,0.],tau_bound=.02),
        materials=[dict(label='Ad8.print',paths=[dict(kind=2,advertisement_block=ai,geometry_block=qi,points=[[0,0],[1,1]])])])


@pytest.mark.parametrize('multiple',[2,4,8])
def test_exact_knot_refinement_including_both_padding_and_derivatives(multiple):
    first=-.04;dt=.02;n=52;R,basis=cardinal_refinement(first,dt,n,multiple)
    rng=np.random.default_rng(17);coarse=rng.normal(size=(R.shape[1],3));fine=R@coarse
    times=np.linspace(first+dt,first+(n-2)*dt,337)
    for derivative,tolerance in [(0,3e-14),(1,4e-12),(2,6e-10)]:
        a=cubic_design(times,first,dt,n,derivative)@fine
        b=cubic_design(times,basis['first_control_time'],basis['dt'],len(coarse),derivative)@coarse
        np.testing.assert_allclose(a,b,atol=tolerance,rtol=2e-13)
    np.testing.assert_allclose(R@np.ones(len(coarse)),1.,atol=1e-15)


def test_projection_changes_the_parent_and_preserves_angle_gauge_and_fine_registration():
    original=model_fixture();chart=UniformCubicTimeChart(original,4)
    with pytest.raises(ValueError,match='outside the coarse family'):chart.recover()
    projected,report=chart.project_model();q,membership=chart.recover(projected)
    assert membership['max_fine_coefficient_membership_error']<1e-11
    assert report['new_temporal_model_hypothesis'] and report['projection_does_not_establish_image_fit']
    assert report['exact_refinement_function_difference']<1e-12
    assert projected['camera']==original['camera']
    assert camera_from_event(projected).reference_error<1e-14
    assert report['coarse_time_free_scalars']<report['fine_time_free_scalars']
    assert np.max(abs(camera_from_event(projected).controls-camera_from_event(original).controls))>1e-4
    again=chart.write_controls(q,projected)
    np.testing.assert_allclose(camera_from_event(again).controls,camera_from_event(projected).controls,atol=2e-13)


def test_combined_lift_removes_every_fine_time_coordinate_and_preserves_the_family():
    original=model_fixture();time=UniformCubicTimeChart(original,2);projected,_=time.project_model();print_chart=PrintOnlyAdChart(projected,(8,))
    C,scales,names,info=time.compose_physical(print_chart,projected)
    assert sum(name.startswith('time/coarse/')for name in names)==3*time.nc-2
    assert not any(name.startswith('control/')for name in names)
    assert C.shape[1]==info['non_time_columns']+3*time.nc-2
    rng=np.random.default_rng(31);delta=rng.normal(size=C.shape[1])*1e-6
    candidate=print_chart.unpack(print_chart.pack()+C@(scales*delta))
    _,membership=time.recover(candidate)
    assert membership['max_fine_coefficient_membership_error']<1e-11
    assert camera_from_event(candidate).reference_error<1e-14
    assert candidate['camera']==projected['camera']


@pytest.mark.parametrize('multiple',[2,4,8])
def test_local_knot_removal_is_exact_and_keeps_outside_storage_bitwise(multiple):
    from rinkfit_start.camera_spline_chart import LocalCubicTimeChart,time_chart_from_model
    m=model_fixture();local=LocalCubicTimeChart(m,multiple,.22,.58)
    projected,proof=local.project_model();fine=camera_from_event(projected).controls
    assert proof['outside_storage_bitwise_preserved']
    assert proof['both_source_paddings_unchanged']
    assert proof['removed_knots']>0
    assert np.max(abs(local.knot_constraints@fine))<2e-13
    old=np.array([b['values']for b in m['blocks'][1:53]])
    new=np.array([b['values']for b in projected['blocks'][1:53]])
    time=-.04+.02*np.arange(52);outside=(time<.22-1e-12)|(time>.58+1e-12)
    assert np.array_equal(old[outside].view(np.uint64),new[outside].view(np.uint64))
    assert np.max(abs(new[~outside]-old[~outside]))>1e-6
    resumed=time_chart_from_model(projected)
    assert isinstance(resumed,LocalCubicTimeChart)
    _,membership=resumed.recover(projected)
    assert membership['max_fine_coefficient_membership_error']<1e-11
    # Removing these knots preserves every global polynomial of degree<=3.
    for degree in range(4):
        c=(np.arange(52)/52.)**degree
        np.testing.assert_allclose(local.R@c[local.retained],c,atol=1e-13)
