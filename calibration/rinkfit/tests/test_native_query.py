import numpy as np
from rinkfit_start.native_query import rational_query,recover_query_roots


def curves():
    return {'numerator_u':np.array([[0.,.01,0.,0.]]),'numerator_v':np.array([[0.,0.,.0001,0.]]),
        'denominator':np.array([[1.,0.,0.,0.]]),'bounds':np.array([[0.,100.]])}


class Scene:
    def world_rows(self,m,b,s):
        return np.c_[s,1-s,np.zeros(len(s))],(s>=0)&(s<=1),None


class Camera:
    def project_rs(self,w,t):
        return {'uv':w[:,:2].copy(),'valid':np.isfinite(w).all(axis=1)}


def test_original_query_coordinate_is_not_pixel_distance():
    q,tangent,valid=rational_query(curves(),np.array([60.]))
    np.testing.assert_allclose(q,[[.6,.36]],atol=1e-15)
    np.testing.assert_allclose(tangent,[[.01,.012]],atol=1e-15)
    assert valid.all()
    out=recover_query_roots(Scene(),Camera(),[0],[0],[.6],[0.],curves(),[60.])
    exact=(np.sqrt(5)-1)/2
    np.testing.assert_allclose(out['stations'],[exact],atol=1e-8)
    np.testing.assert_allclose(out['alpha'],[100*exact],atol=1e-7)
    assert out['valid'].all()
    assert not out['relevant_root_interval_certified'].any()


def test_root_outside_original_finite_query_is_a_failure():
    c=curves();c['bounds']=np.array([[0.,50.]])
    out=recover_query_roots(Scene(),Camera(),[0],[0],[.6],[0.],c,[40.])
    assert not out['valid'].any()


def finite_row_fixture():
    from types import SimpleNamespace
    from rinkfit_start.native_camera import NativeCamera
    q=np.array([-1.,-4.,1.,16.])
    def world_rows(m,b,s):
        s=np.asarray(s)
        return np.c_[q[0]*(1-s)+q[2]*s,q[1]*(1-s)+q[3]*s,np.zeros(len(s))],(s>=0)&(s<=1),None
    scene=SimpleNamespace(values=[q],path=lambda m,b:dict(kind=0,geometry_block=0),world_rows=world_rows)
    controls=np.zeros((10,3));controls[:,2]=np.log(100.)
    camera=NativeCamera(dict(shared_head=dict(R0=np.eye(3).tolist(),n=[0.,1.,0.],b=[1.,0.,0.],P=[0.,0.,-2.],d=[0.,0.,0.],a=0.),
        camera_controls=dict(basis=dict(dt=.02,first_control_time=-.04,n_controls=10),controls=controls.tolist()),
        optics=dict(f_ref=100.,principal_point=[960.,540.],k1=[0.,0.],k2=[0.,0.],p1=0.,p2=0.,radius_max=20.),
        rolling_shutter=dict(signed_readout_seconds=.02)))
    query=dict(numerator_u=np.array([[960.,1.,0.,0.]]),numerator_v=np.array([[800.,0.,0.,0.]]),
        denominator=np.array([[1.,0.,0.,0.]]),bounds=np.array([[-20.,20.]]))
    return scene,camera,query


def test_finite_three_variable_fallback_recovers_when_saved_station_has_no_RS_root():
    scene,camera,c=finite_row_fixture()
    world,_,_=scene.world_rows([0],[0],np.array([.9]))
    assert not camera.project_rs(world,np.array([0.]))['bracketed'][0]
    out=recover_query_roots(scene,camera,[0],[0],[.9],[0.],c,[0.])
    assert out['valid'][0] and out['relevant_root_interval_certified'][0]
    assert out['finite_three_variable_fallback_succeeded'][0]
    np.testing.assert_allclose(out['stations'],[.46],atol=1e-10)
    np.testing.assert_allclose(out['alpha'],[-4.],atol=1e-8)
    np.testing.assert_allclose(out['row'],[800.],atol=1e-7)
    assert out['query_equality_px'][0]<1e-7


def test_three_variable_fallback_does_not_admit_root_outside_original_alpha_bounds():
    scene,camera,c=finite_row_fixture();c=dict(c,bounds=np.array([[0.,20.]]))
    out=recover_query_roots(scene,camera,[0],[0],[.9],[0.],c,[1.])
    assert not out['valid'][0]
    assert out['finite_three_variable_fallback_attempted'][0]
    assert not out['finite_three_variable_fallback_succeeded'][0]
    assert not out['relevant_root_interval_certified'][0]


def test_existing_successful_row_is_byte_unchanged_when_another_row_needs_fallback():
    scene,camera,c=finite_row_fixture()
    successful=recover_query_roots(scene,camera,[0],[0],[.46],[0.],c,[-4.])
    batch={k:np.repeat(v,2,axis=0)for k,v in c.items()}
    out=recover_query_roots(scene,camera,[0,0],[0,0],[.46,.9],[0.,0.],batch,[-4.,0.])
    for key,value in successful.items():
        if isinstance(value,np.ndarray) and key not in ['finite_three_variable_fallback_reason']:
            assert out[key][:1].tobytes()==value.tobytes(),key
        elif isinstance(value,dict):
            for name,array in value.items():
                if isinstance(array,np.ndarray):assert out[key][name][:1].tobytes()==array.tobytes(),name
