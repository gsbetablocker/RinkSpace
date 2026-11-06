import numpy as np
from types import SimpleNamespace
from scipy.spatial.transform import Rotation
from calibration.program_tracking.body_tracking.runner_interior import runner_interior_band, runner_longitudinal_coordinate, longitudinal_barrier
from calibration.program_tracking.body_tracking.skate_template import SkateParameters, build_skate
from calibration.program_tracking.body_tracking.skate_physics import batch_features


def normals(degrees):
    t=np.deg2rad(degrees)
    return np.c_[-np.sin(t),np.zeros(len(t)),np.cos(t)]


def test_longitudinal_coordinate_matches_exact_curved_runner_and_has_endpoint_descent():
    p=SkateParameters();template=build_skate(p)
    binding=SimpleNamespace(ratio=1.,packet=lambda:dict(template=template))
    angles=np.array([-15.,-2.9,0.,6.,12.,30.])
    R=Rotation.from_rotvec(np.c_[np.zeros(6),np.deg2rad(angles),np.zeros(6)]).as_matrix()
    features=batch_features(binding,np.repeat(R[:,None],2,axis=1),np.zeros((6,2,3)))
    ids=[i for i,n in enumerate(features['ids']) if n in template['support_feature_ids']['glide']]
    pick=np.asarray(ids)[np.argmin(features['height'][:,1,ids],axis=1)]
    x=features['local_point'][np.arange(6),1,pick,0]
    c=runner_longitudinal_coordinate(normals(angles),p,x)
    assert abs(c['actual_u'][0])<1e-12 and abs(c['actual_u'][-1]-1)<1e-12
    assert c['effective_u'][0]<0 and c['effective_u'][-1]>1
    np.testing.assert_array_equal(c['actual_derivative_normal'][[0,-1]],np.zeros((2,3)))
    assert np.linalg.norm(c['derivative_normal'][0])>0 and np.linalg.norm(c['derivative_normal'][-1])>0


def test_actual_and_endpoint_restoration_coordinate_derivatives():
    p=SkateParameters();v=normals([-15.,-2.9,0.,6.,12.,30.]);d=np.tile([.23,-.17,.12],(6,1));h=1e-6
    c=runner_longitudinal_coordinate(v,p)
    a=runner_longitudinal_coordinate(v+h*d,p);b=runner_longitudinal_coordinate(v-h*d,p)
    for name,derivative in [('actual_u','actual_derivative_normal'),('effective_u','derivative_normal')]:
        np.testing.assert_allclose((a[name]-b[name])/(2*h),np.sum(c[derivative]*d,axis=1),rtol=2e-8,atol=2e-9)


def test_finite_band_has_no_center_attraction_and_analytic_barrier_derivative():
    p=SkateParameters();band,scale=runner_interior_band(p);v=normals([-15.,-2.9,0.,6.,12.,30.]);w=np.arange(1,7)*.04
    c=runner_longitudinal_coordinate(v,p);r,J=longitudinal_barrier(c,band,scale,w)
    assert r[0]<0 and r[-1]>0 and r[2]==0 and r[3]==0
    d=np.tile([.23,-.17,.12],(6,1));h=1e-6
    a=longitudinal_barrier(runner_longitudinal_coordinate(v+h*d,p),band,scale,w)[0]
    b=longitudinal_barrier(runner_longitudinal_coordinate(v-h*d,p),band,scale,w)[0]
    np.testing.assert_allclose((a-b)/(2*h),np.sum(J*d,axis=1),rtol=2e-8,atol=2e-9)


def test_real_bank_preserved_without_forcing_flat_runner():
    p=SkateParameters();band,scale=runner_interior_band(p)
    R=Rotation.from_rotvec(np.c_[np.deg2rad([0,20,40]),np.zeros((3,2))]).as_matrix()
    c=runner_longitudinal_coordinate(R[:,2],p);r,_=longitudinal_barrier(c,band,scale,np.ones(3))
    np.testing.assert_array_equal(r,np.zeros(3))
