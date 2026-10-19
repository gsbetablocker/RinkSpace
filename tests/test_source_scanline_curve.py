import numpy as np
import pytest

from calibration.solver_runtime import source_scanline_curve as s


def test_explicit_side_full_curve_and_paired_layout():
    ring = np.array([[-2.,-1.], [3.,-1.], [4.,1.], [-3.,1.], [-2.,-1.]])
    observed = np.array([[100.,0.], [-100.,.5]])
    assert s.projected_horizontal_scanline_value(ring, np.ones(5,bool), observed, side='right') == pytest.approx([-96.5,0.,103.75,0.])
    assert s.projected_horizontal_scanline_value(ring, np.ones(5,bool), observed, side='left') == pytest.approx([-102.5,0.,97.25,0.])
    # The extreme branch is independent of which source x is closest.
    changed = observed.copy();changed[:,0]=0.
    r,a=s.projected_horizontal_scanline_value(ring,np.ones(5,bool),changed,side='right',with_association=True)
    assert a['segment_indices'].tolist()==[1,1]
    assert a['ROI_or_observed_x_used_to_select_branch'] is False


def test_compact_endpoint_gradient_matches_coordinate_fd():
    points=np.array([[-2.,-2.],[4.,3.],[0.,5.]])
    uv=np.array([[2.,.3],[3.,1.7]])
    raw,association,gradient=s.projected_horizontal_scanline_value_and_gradient(points,np.ones(3,bool),uv,side='right')
    for endpoint in range(2):
        for axis in range(2):
            h=1e-6;p=points.copy();m=points.copy();p[endpoint,axis]+=h;m[endpoint,axis]-=h
            fd=(s.projected_horizontal_scanline_chart_response(p,np.ones(3,bool),uv,association)-s.projected_horizontal_scanline_chart_response(m,np.ones(3,bool),uv,association))/(2*h)
            assert fd.reshape(-1,2)[:,0]==pytest.approx(gradient[:,endpoint,axis],abs=7e-10)
            assert np.array_equal(fd.reshape(-1,2)[:,1],np.zeros(2))


def test_duplicate_vertices_are_single_crossing_not_extra_weight():
    points=np.array([[1.,-2.],[2.,0.],[2.,0.],[3.,2.]])
    uv=np.array([[0.,0.]])
    raw,a,g=s.projected_horizontal_scanline_value_and_gradient(points,np.ones(4,bool),uv,side='right')
    assert raw.tolist()==[2.,0.]
    assert a['regular'].tolist()==[True]
    assert a['duplicate_witness_count'].tolist()==[2]
    assert s.projected_horizontal_scanline_chart_response(points,np.ones(4,bool),uv,a)==pytest.approx(raw)


@pytest.mark.parametrize('points', [np.array([[0.,1.],[2.,0.],[4.,1.]]),np.array([[0.,0.],[4.,0.]])])
def test_tangent_or_coincident_value_has_no_fictitious_derivative(points):
    raw,a=s.projected_horizontal_scanline_value(points,np.ones(len(points),bool),[[0.,0.]],side='right',with_association=True)
    assert np.isfinite(raw).all()
    assert not a['regular'][0]
    with pytest.raises(s.SourceScanlineDomainError,match='no regular'):
        s.projected_horizontal_scanline_chart_response(points,np.ones(len(points),bool),[[0.,0.]],a)


def test_no_intersection_never_falls_back_to_nearest_foot():
    points=np.array([[0.,1.],[4.,2.]])
    with pytest.raises(s.SourceScanlineDomainError,match='No positive-central'):
        s.projected_horizontal_scanline_value(points,np.ones(2,bool),[[2.,0.]],side='right')
    with pytest.raises(s.SourceScanlineDomainError):
        s.projected_horizontal_scanline_value(points,[False,True],[[2.,1.5]],side='right')


def test_physical_endpoint_is_not_unbounded_regular_station():
    points=np.array([[0.,0.],[2.,2.]])
    raw,a=s.projected_horizontal_scanline_value(points,np.ones(2,bool),[[0.,0.]],side='right',with_association=True)
    assert raw.tolist()==[0.,0.]
    with pytest.raises(s.SourceScanlineDomainError):
        s.projected_horizontal_scanline_chart_response(points,np.ones(2,bool),[[0.,0.]],a)


def test_chart_refuses_observation_change_and_slope_flip():
    points=np.array([[0.,-1.],[3.,2.]])
    raw,a=s.projected_horizontal_scanline_value(points,np.ones(2,bool),[[2.,0.]],side='right',with_association=True)
    with pytest.raises(ValueError):s.projected_horizontal_scanline_chart_response(points,np.ones(2,bool),[[3.,0.]],a)
    with pytest.raises(s.SourceScanlineDomainError):s.projected_horizontal_scanline_chart_response(points[::-1],np.ones(2,bool),[[2.,0.]],a)


def test_full_brown_camera_and_world_directional_chain():
    from calibration.solver_runtime import install
    install()
    camera=np.zeros(25);camera[:3]=[.3,-.2,-8.];camera[3:6]=[.03,-.04,.02]
    camera[6:8]=np.log([1100.,1080.]);camera[8:10]=[960.,540.]
    camera[11:15]=[.2,-.02,.002,-.003];camera[23:25]=[.015,-.01]
    theta=np.linspace(0,2*np.pi,361);world=np.c_[2*np.cos(theta),1.5*np.sin(theta),np.zeros(len(theta))]
    uv=np.array([[1100.,520.],[1200.,590.]])
    points,valid=s._project(camera,world)
    raw,a,gradient=s.projected_horizontal_scanline_value_and_gradient(points,valid,uv,side='right')
    assert s.horizontal_scanline_curve_value(camera,world,uv,side='right')==pytest.approx(raw)
    cam_d=np.zeros(25);cam_d[[0,1,3,4,6,7,8,9,11,12,13,23,24]]=[.2,-.1,.01,-.02,.03,-.02,2.,-1.,.03,-.01,.002,.001,-.002]
    world_d=np.c_[.1*np.cos(theta),-.05*np.sin(theta),np.full(len(theta),.03)]
    for dc,dw in ((cam_d,np.zeros_like(world)),(np.zeros(25),world_d),(cam_d,world_d)):
        h=1e-6;p,pv=s._project(camera+h*dc,world+h*dw);m,mv=s._project(camera-h*dc,world-h*dw)
        projected_d=(p-m)/(2*h);indices=a['segment_indices']
        chain=np.einsum('nei,nei->n',gradient,np.stack([projected_d[indices],projected_d[indices+1]],axis=1))
        actual=(s.horizontal_scanline_curve_value(camera+h*dc,world+h*dw,uv,side='right')-s.horizontal_scanline_curve_value(camera-h*dc,world-h*dw,uv,side='right'))/(2*h)
        chart=(s.horizontal_scanline_curve_chart_response(camera+h*dc,world+h*dw,uv,a)-s.horizontal_scanline_curve_chart_response(camera-h*dc,world-h*dw,uv,a))/(2*h)
        assert actual[::2]==pytest.approx(chain,abs=2e-7)
        assert chart==pytest.approx(actual,abs=2e-7)
        assert np.array_equal(actual[1::2],np.zeros(2))
