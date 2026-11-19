import numpy as np

from calibration.program_tracking.body_tracking.source_foot_foreground import field_sample


def test_native_distance_field_derivative_and_outside_extension():
    y,x=np.mgrid[:12,:16];field=(.2*x*x+.3*y*y)[None]
    bounds=np.array([[100.,200.,260.,320.]])
    uv=np.array([[[153.,247.],[75.,360.],[300.,277.]]])
    value,gradient=field_sample(field,bounds,uv)
    for axis in (0,1):
        step=np.zeros_like(uv);step[...,axis]=1.e-5
        plus=field_sample(field,bounds,uv+step)[0]
        minus=field_sample(field,bounds,uv-step)[0]
        np.testing.assert_allclose((plus-minus)/2.e-5,gradient[...,axis],rtol=1.e-7,atol=1.e-7)
    assert np.isfinite(value).all()


def test_hidden_foot_inside_source_person_has_no_attraction():
    field=np.zeros((1,12,16));bounds=np.array([[100.,200.,260.,320.]])
    value,gradient=field_sample(field,bounds,np.array([[[145.,245.],[240.,290.]]]))
    np.testing.assert_array_equal(value,0.)
    np.testing.assert_array_equal(gradient,0.)


def test_cropped_boot_is_not_pulled_onto_visible_calf():
    field=np.full((1,12,16),40.);bounds=np.array([[100.,800.,300.,1080.]])
    uv=np.array([[[210.,1100.],[210.,1074.],[210.,1020.]]])
    value,gradient=field_sample(field,bounds,uv,image_size=[1920,1080])
    np.testing.assert_allclose(value,[[0.,5.5,40.]])
    np.testing.assert_allclose(gradient,[[[0.,0.],[0.,-1.],[0.,0.]]])
    for axis in (0,1):
        step=np.zeros_like(uv);step[...,axis]=1.e-5
        plus=field_sample(field,bounds,uv+step,image_size=[1920,1080])[0]
        minus=field_sample(field,bounds,uv-step,image_size=[1920,1080])[0]
        np.testing.assert_allclose((plus-minus)/2.e-5,gradient[...,axis],atol=1.e-7)


def test_registered_cuff_rejects_sideways_shank_and_has_actual_tangent_derivative():
    from types import SimpleNamespace
    from scipy.sparse import block_diag
    from calibration.program_tracking.body_tracking.source_foot_foreground import RegisteredBootCuff

    class Delta(np.ndarray):
        def chain(self,local):return block_diag(list(local),format='csr')
    class State:
        def increments(self,x,times):return np.asarray(x).reshape(len(times),69).view(Delta)
    class Binding:
        ratio=1.
        def parameters(self):
            return np.tile(np.eye(3),(2,1,1)),np.tile([-.05,0.,-.15],(2,1)),None
        def packet(self):
            return dict(parameters=SimpleNamespace(sole_length_m=.23),xs=np.array([0.,.0644,.23]),
                        floor=np.full(3,.02),rise=np.full(3,.18),width=np.full(3,.05))
    joints=np.zeros((22,3));joints[[4,5],2]=1.
    reference=SimpleNamespace(body=SimpleNamespace(J=joints),at=lambda t:dict(
        local_R=np.broadcast_to(np.eye(3),(len(t),21,3,3))))
    problem=SimpleNamespace(reference=reference,binding=Binding(),body=SimpleNamespace(state=State()))
    factor=RegisteredBootCuff(problem,np.array([0.,.02,.04]))
    x=np.zeros(3*69);bad=x.reshape(3,69);bad[:,6+3*(7-1)]=1.1
    result=factor.evaluate(x,True)
    neutral=factor.evaluate(np.zeros_like(x))
    assert np.linalg.norm(result['residual'])>10*np.linalg.norm(neutral['residual'])
    v=np.random.default_rng(3).normal(size=x.size);v/=np.linalg.norm(v)
    numerical=(factor.evaluate(x+1.e-6*v)['residual']-factor.evaluate(x-1.e-6*v)['residual'])/2.e-6
    np.testing.assert_allclose(numerical,result['jac'][:,17:]@v,rtol=2.e-6,atol=2.e-8)
