"""Tangent-cone oracle and actual GPU checks at physical parameter bounds."""
from pathlib import Path
import os
import shutil
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def cone_program(tmp_path_factory):
    compiler = shutil.which('g++') or shutil.which('cl')
    if compiler is None:
        pytest.skip('existing C++ compiler required')
    directory = tmp_path_factory.mktemp('stationarity')
    code = (ROOT/'native/src/gpu_stationarity.cuh').read_text(encoding='utf-8')
    code = code[code.index('template<class Equality>'):code.index('__global__ void lm_stationarity_project')]
    cpp = directory/'cone.cpp'
    cpp.write_text('#include "rinkspace/core_types.hpp"\n#include <vector>\n#include <iostream>\n#include <iomanip>\nusing namespace rinkspace;\n'+code+r'''
int main(){int q,m;std::cin>>q>>m;std::vector<double> g(q),a(size_t(q)*m),dual(m),p(q);std::vector<int> eq(m);
 for(auto& x:g)std::cin>>x;for(auto& x:a)std::cin>>x;for(auto& x:eq)std::cin>>x;double certificate=0;
 bool ok=stationarity_cone(q,m,g.data(),a.data(),dual.data(),p.data(),[&](int i){return eq[i]!=0;},1e-12,certificate);
 std::cout<<ok<<' '<<std::setprecision(17)<<certificate;for(double x:p)std::cout<<' '<<x;return 0;}
''', encoding='utf-8')
    executable = directory/('cone.exe' if os.name=='nt' else 'cone')
    if Path(compiler).stem.lower()=='cl':
        command=[compiler,'/nologo','/std:c++17','/EHsc','/O1','/I'+str(ROOT/'native/include'),str(cpp),'/Fe:'+str(executable)]
    else:
        command=[compiler,'-std=c++17','-O1','-I'+str(ROOT/'native/include'),str(cpp),'-o',str(executable)]
    subprocess.run(command,cwd=directory,capture_output=True,check=True)
    return executable


@pytest.mark.parametrize('gradient,matrix,equality', [
    ([3.,0.], [[1.,0.]], [1]),
    ([3.,1.], [[1.,0.]], [1]),
    ([3.,0.], [[1.,0.]], [0]),
    ([-3.,0.], [[-1.,0.]], [0]),
    ([3.,1.], [[1.,1.]], [1]),
    ([3.,3.], [[1.,1.]], [0]),
    ([3.,-1.], [[1.,1.],[1.,0.]], [0,0]),
    ([3.,-1.], [[1.,1.],[1.,1.],[1.,0.]], [0,0,0]),
    ([3.,1.], [[1.,0.],[0.,1.]], [1,1]),
    ([3.,1.], np.empty((0,2)), []),
    # Near-parallel physical faces caused the real S01+S04 stopping check to
    # exhaust 4096 scalar sweeps despite a well-defined feasible descent.
    ([3.,.11,2.], [[1.,0.,0.],[np.cos(.07),np.sin(.07),0.]], [0,0]),
    ([3.,.11,2.], [[0.,0.,0.],[np.cos(.07),np.sin(.07),0.],[1.,0.,0.]], [0,0,0]),
    ([3.,-.11,2.], [[1.,0.,0.],[np.cos(.07),np.sin(.07),0.]], [0,0]),
    ([3.,.11,2.], [[1.,0.,0.],[np.cos(.07),np.sin(.07),0.]], [1,0]),
    ([3.,.11,2.], [[1.,0.,0.],[np.cos(.07),np.sin(.07),0.]], [1,1]),
    ([3.,.11,2.], [[1.,0.,0.],[-np.cos(.07),-np.sin(.07),0.]], [0,0]),
])
def test_original_kernel_cone_matches_independent_cpu_oracle(cone_program, gradient, matrix, equality):
    from scipy.optimize import minimize
    gradient=np.asarray(gradient);matrix=np.asarray(matrix).reshape(-1,len(gradient));equality=np.asarray(equality,bool)
    parts=[str(len(gradient)),str(len(matrix)),*map(str,gradient),*map(str,matrix.ravel()),*map(str,equality.astype(int))]
    result=subprocess.run([str(cone_program)],input=' '.join(parts),text=True,capture_output=True,check=True,timeout=10)
    values=np.fromstring(result.stdout,sep=' ')
    assert values[0]==1, result.stdout
    constraints=[]
    for eq in (True,False):
        rows=matrix[equality==eq]
        if len(rows):constraints.append({'type':'eq' if eq else 'ineq','fun':lambda x,a=rows:a@x,'jac':lambda x,a=rows:a})
    oracle=minimize(lambda x:.5*np.sum((x+gradient)**2),np.zeros(len(gradient)),jac=lambda x:x+gradient,
                    constraints=constraints,method='SLSQP',options={'ftol':1e-13,'maxiter':1000})
    assert oracle.success, oracle.message
    np.testing.assert_allclose(values[2:],oracle.x,atol=2e-11,rtol=0.)
    assert values[1]>=np.max(np.abs(values[2:]))


def optical_problem(*, offset, bound, coupled, solver_options=None):
    from rinkspace.core.native import NativeCore
    from rinkspace.core.gpu import ResidentResidualProblem
    from test_g2_gpu import G2GPU
    fixture=G2GPU();fixture.setUp();kw=fixture.kw
    core=NativeCore(os.environ['RINKSPACE_G4_LIBRARY'])
    kw['controls']=np.tile([0.,0.,np.log(100.)],(8,1))
    with ResidentResidualProblem(core,**kw) as p:
        observed=p.evaluate('cuda')['values'][:,3:5]+np.asarray(offset)
    p=ResidentResidualProblem(core,**{**kw,'observed':observed})
    directions=np.zeros((27,2));directions[15]=[1.,1.] if coupled else [1.,0.]
    directions[16]=[1.,-1.] if coupled else [0.,1.]
    lower=np.full(38,-np.inf);upper=np.full(38,np.inf)
    lower[26],upper[26]=bound
    p.configure_camera_parameters(directions=directions,max_readout=.08,lower=lower,upper=upper)
    p.configure_pixel_linearization(directions=np.zeros((8,3,2)),local_ids=np.full(4,-1),local_scales=np.zeros(0))
    p.configure_solver(lower=np.zeros(0),upper=np.zeros(0),device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30,**(solver_options or {}))
    return p


@pytest.mark.skipif(not os.environ.get('RINKSPACE_G4_LIBRARY'),reason='explicit CUDA/cuDSS library required')
@pytest.mark.parametrize('offset,bound,coupled', [
    ([3.,0.], [50.,50.], False),
    ([3.,0.], [50.,50.], True),
    ([-3.,0.], [50.,np.inf], False),
    ([3.,0.], [-np.inf,50.], True),
])
def test_gpu_constrained_optimum_stops_before_any_trial(offset,bound,coupled):
    with optical_problem(offset=offset,bound=bound,coupled=coupled) as p:
        before=p.export_state();result=p.solve(progress=lambda _:False)
        assert result['numerically_converged'], result
        assert result['trials']==0
        assert result['projected_gradient_inf']<1e-10
        after=p.export_state()
        for key in before:np.testing.assert_array_equal(after[key],before[key])


@pytest.mark.skipif(not os.environ.get('RINKSPACE_G4_LIBRARY'),reason='explicit CUDA/cuDSS library required')
@pytest.mark.parametrize('coupled',[False,True])
def test_gpu_fixed_physical_coordinate_keeps_feasible_descent(coupled):
    with optical_problem(offset=[3.,2.],bound=[50.,50.],coupled=coupled) as p:
        history=[]
        def once(record):history.append(record);return False
        result=p.solve(progress=once)
        assert result['accepted']==1, result
        assert history[0]['projected_gradient_before']>1.
        assert result['projected_gradient_inf']==result['stationarity_diagnostics']['constraint_projected_gradient_inf']
        assert 0.<result['projected_gradient_inf']<history[0]['projected_gradient_before']
        state=p.export_state();assert state['rig'][26]==50.
        assert state['rig'][27]>51.9
        assert p.solve()['numerically_converged']


@pytest.mark.skipif(not os.environ.get('RINKSPACE_G4_LIBRARY'),reason='explicit CUDA/cuDSS library required')
def test_gpu_interior_optics_are_not_mistaken_for_active_bounds():
    with optical_problem(offset=[.2,.1],bound=[49.,51.],coupled=True) as p:
        result=p.solve();assert result['numerically_converged'], result
        state=p.export_state();np.testing.assert_allclose(state['rig'][26:28],[50.2,50.1],atol=1e-7,rtol=0.)


@pytest.mark.skipif(not os.environ.get('RINKSPACE_G4_LIBRARY'),reason='explicit CUDA/cuDSS library required')
@pytest.mark.parametrize('camera_count',[1,2])
def test_static_axis_face_occurs_once_per_physical_camera(camera_count):
    from rinkspace.core.gpu import ResidentResidualProblem
    from test_camera_envelope import CameraEnvelopeTests
    from test_g2_gpu import G2GPU
    fixture=G2GPU();fixture.setUp();mount=CameraEnvelopeTests();mount.setUp()
    rig=mount.rig.copy();angle=np.radians(5.)
    rig[12:15]=[np.cos(angle),0.,np.sin(angle)];rig[27]=540.
    rigs=np.repeat(rig[None,:],camera_count,axis=0)
    controls=np.tile([0.,.15,np.log(100.)],(32,1))
    kw={**fixture.kw,'rig':rigs,'camera_ids':np.repeat(np.arange(camera_count,dtype=np.int32),4),
        'controls':controls,'image_height':1080}
    for key in ('codes','stations','times','observed'):kw[key]=np.concatenate([fixture.kw[key]]*camera_count)
    with ResidentResidualProblem(fixture.core,**kw) as p:
        directions=np.zeros((27*camera_count,camera_count))
        for camera in range(camera_count):directions[27*camera+5,camera]=.01
        p.configure_camera_parameters(directions=directions,max_readout=.08)
        for camera in range(camera_count):
            p.configure_camera_envelope(forward_bounds=mount.limits[:2],pupil_bounds=mount.limits[2:4],
                max_pan_tilt=mount.limits[4],max_tilt_deviation=mount.limits[5],
                elevation_bounds=mount.limits[6:8],roll_bounds=mount.limits[8:10],
                frame_time_intervals=np.array([[0.,10.],[15.,25.]]),camera_index=camera)
        p.configure_pixel_linearization(directions=np.zeros((32,3,camera_count)),
            local_ids=np.full(4*camera_count,-1,np.int32),local_scales=np.zeros(0))
        p.configure_solver(lower=[],upper=[],device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
        diagnostic=p.stationarity_diagnostics(include_constraints=True)
        records=diagnostic['constraint_records'];mask=(records[:,0]==3)&(records[:,11]==1)
        assert mask.sum()==camera_count
        np.testing.assert_array_equal(np.sort(records[mask,1]),np.arange(camera_count))
        assert np.all(records[mask,10]>0)
        assert diagnostic['physical_feasibility_violation']<1e-12
        assert diagnostic['cone_primal_residual']<1e-10


def planted_gradient(make, gradient):
    """Independent linear algebra plants the original objective's gradient."""
    with make() as p:
        values=p.evaluate('cuda')['values'][:,3:5]
        p.linearize();jac=p.probe_rows(np.arange(len(values)))[:,:,:len(gradient)].reshape(-1,len(gradient))
        residual=np.linalg.lstsq(jac.T,np.asarray(gradient),rcond=None)[0]
        np.testing.assert_allclose(jac.T@residual,gradient,atol=1e-11,rtol=0.)
    return values-residual.reshape(-1,2)


def assert_initial_constrained_optimum(make, gradient):
    observed=planted_gradient(make,gradient)
    with make(observed) as p:
        before=p.export_state();result=p.solve(progress=lambda _:False)
        assert result['numerically_converged'] and result['trials']==0, result
        for key,value in before.items():np.testing.assert_array_equal(p.export_state()[key],value)


@pytest.mark.skipif(not os.environ.get('RINKSPACE_G4_LIBRARY'),reason='explicit CUDA/cuDSS library required')
def test_gpu_scene_bound_uses_the_current_positive_chart_pullback(monkeypatch):
    from rinkspace.core.gpu import ResidentResidualProblem
    from test_gpu_joint_scene import JointSceneGPU
    fixture=JointSceneGPU();fixture.setUp()
    original=ResidentResidualProblem.configure_scene_parameters
    def bounded(self,**kwargs):
        lower=np.full(len(kwargs['values']),-np.inf);upper=np.full(len(lower),np.inf)
        lower[43]=upper[43]=kwargs['values'][43]
        return original(self,**kwargs,lower=lower,upper=upper)
    monkeypatch.setattr(ResidentResidualProblem,'configure_scene_parameters',bounded)
    directions=np.zeros((63,2));directions[43,0]=directions[44,1]=.001
    charts=np.zeros(63,np.int32);charts[43]=1
    def make(observed=None):
        p=fixture.make(directions=directions,observed=observed,charts=charts)
        p.configure_pixel_linearization(directions=np.zeros((8,3,2)),local_ids=np.full(30,-1),local_scales=np.zeros(0))
        p.configure_solver(lower=[],upper=[],device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
        return p
    assert_initial_constrained_optimum(make,[1.,0.])


@pytest.mark.skipif(not os.environ.get('RINKSPACE_G4_LIBRARY'),reason='explicit CUDA/cuDSS library required')
@pytest.mark.parametrize('symmetric,positive_chart',[(False,False),(True,False),(True,True)])
def test_gpu_shared_lane_goal_boundary_uses_the_coupled_cone(symmetric,positive_chart):
    from rinkspace.core.gpu import ResidentResidualProblem
    from test_g2_gpu import G2GPU
    fixture=G2GPU();fixture.setUp();kw=fixture.kw;core=fixture.core
    kw.update(options=22 if symmetric else 6,codes=np.repeat(np.arange(2,dtype=np.int32),4),stations=np.tile([0.,.1,.2,.3],2),times=np.zeros(8),observed=np.zeros((8,2)))
    if symmetric:
        kw['layout']=list(kw['layout']);kw['layout'][23]=kw['layout'][31]=kw['layout'][32]=0.
    goal_column=30 if symmetric else 29
    state=np.r_[kw['board'],kw['layout']];directions=np.zeros((41,2));directions[16,0]=directions[goal_column,1]=.001
    charts=np.zeros(41,np.int32)
    if positive_chart:charts[goal_column]=1
    normal=np.array([1.,(1. if symmetric else -1.)*(state[goal_column] if positive_chart else 1.)])
    assert state[16]==(-state[goal_column] if symmetric else state[goal_column])
    def make(observed=None):
        p=ResidentResidualProblem(core,**{**kw,'observed':kw['observed'] if observed is None else observed})
        paths=[]
        for code in (300,403):
            kind,parameters=core.shared_geometry(kw['board'],kw['layout'],code,options=kw['options']);paths.append(dict(kind=kind,parameters=parameters))
        p.configure_geometry(paths)
        p.configure_scene_parameters(values=state,records=np.array([[1,300,*([0]*10)],[1,403,*([0]*10)]],np.int32),data=np.zeros(1),directions=directions,charts=charts)
        p.configure_pixel_linearization(directions=np.zeros((8,3,2)),local_ids=np.full(8,-1),local_scales=np.zeros(0))
        p.configure_solver(lower=[],upper=[],device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
        return p
    assert_initial_constrained_optimum(make,normal)
    # A gradient with both forbidden normal and feasible tangent components
    # must descend along the coupled face, not freeze either parameter.
    with make() as p:
        uv=p.evaluate('cuda')['values'][:,3:5]
        p.linearize();jac=p.probe_rows(np.arange(8))[:,:,:2].reshape(-1,2)
    gradient=.001*(normal+2*np.array([normal[1],-normal[0]]))
    observed=uv-np.linalg.lstsq(jac.T,gradient,rcond=None)[0].reshape(-1,2)
    with make(observed) as p:
        initial=p.evaluate_errors();old_cost=.5*np.dot(initial,initial)
        result=p.solve(progress=lambda _:False)
        assert result['accepted']==1,result
        after=p.export_state()['layout']
        goal=-after[24] if symmetric else after[23]
        assert after[10]-goal>=-1e-9
        assert (after[10]-kw['layout'][10])*(-1 if symmetric else 1)>0
        np.testing.assert_array_equal(p.evaluate('cuda')['status'],0)
        errors=p.evaluate_errors();assert .5*np.dot(errors,errors)<old_cost


@pytest.mark.skipif(not os.environ.get('RINKSPACE_G4_LIBRARY'),reason='explicit CUDA/cuDSS library required')
@pytest.mark.parametrize('family',['mount','forward_envelope'])
def test_gpu_nonlinear_physical_boundary_uses_its_original_objective_gradient(family):
    from rinkspace.core.gpu import ResidentResidualProblem
    from test_camera_envelope import CameraEnvelopeTests
    from test_g2_gpu import G2GPU
    camera=CameraEnvelopeTests();camera.setUp();fixture=G2GPU();fixture.setUp()
    rig=camera.rig.copy();rig[37]=.02
    if family=='mount':rig[16]=-fixture.kw['board'][1]
    controls=np.tile([0.,.15,np.log(100.)],(8,1))
    kw={**fixture.kw,'rig':rig,'controls':controls,'codes':np.zeros(4,np.int32)}
    directions=np.zeros((27,2));directions[7 if family=='mount' else 11,0]=1.;directions[15,1]=1.
    def make(observed=None):
        p=ResidentResidualProblem(fixture.core,**{**kw,'observed':kw['observed'] if observed is None else observed})
        p.configure_geometry([dict(kind=4,vertices=[[0.,3.,0.]],parameters=[])])
        p.configure_camera_parameters(directions=directions,max_readout=.08)
        if family=='mount':p.configure_mount_constraints(side='negative_y')
        else:p.configure_camera_envelope(forward_bounds=[.1,.3],pupil_bounds=[-.1,.1],max_pan_tilt=camera.limits[4],
            max_tilt_deviation=camera.limits[5],elevation_bounds=camera.limits[6:8],roll_bounds=camera.limits[8:10])
        p.configure_pixel_linearization(directions=np.zeros((8,3,2)),local_ids=np.full(4,-1),local_scales=np.zeros(0))
        p.configure_solver(lower=[],upper=[],device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
        return p
    assert_initial_constrained_optimum(make,[-1. if family=='mount' else 1.,0.])
