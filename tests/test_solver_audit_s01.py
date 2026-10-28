"""Independent constrained quadratic oracle and compact current-state checks."""
import os
import numpy as np
import pytest
from rinkspace.core.gpu import ResidentResidualProblem
from test_g2_gpu import G2GPU
from test_constrained_stationarity import optical_problem

pytestmark=pytest.mark.skipif(not os.environ.get('RINKSPACE_G4_LIBRARY'),reason='explicit CUDA/cuDSS build required')


def test_constraint_diagnostics_export_matches_original_gradient_without_update():
    with optical_problem(offset=[3.,0.],bound=[50.,50.],coupled=True) as p:
        before=p.export_state();d=p.stationarity_diagnostics(include_constraints=True);after=p.export_state()
        for key in before:np.testing.assert_array_equal(before[key],after[key])
        assert d['raw_retained_gradient_inf']>1.
        assert d['constraint_projected_gradient_inf']<1e-10
        a=d['normalized_constraint_jacobian'];dual=d['constraint_records'][:,9]
        np.testing.assert_allclose(d['projected_retained_descent']+d['raw_retained_gradient'],a.T@dual,atol=1e-11,rtol=0.)
        np.testing.assert_allclose(np.linalg.norm(a,axis=1),1.,atol=1e-14)
        assert d['cone_primal_residual']<1e-11
        assert d['cone_complementarity_residual']<1e-11
        assert d['cone_stationarity_residual']<1e-11
        assert d['physical_feasibility_violation']==0


def test_compact_errors_and_selected_rows_equal_complete_evaluation():
    f=G2GPU();f.setUp()
    with ResidentResidualProblem(f.core,**f.kw) as p:
        full=p.evaluate('cuda');start=p.diagnostic_download_bytes
        errors=p.evaluate_errors()
        np.testing.assert_allclose(errors,np.linalg.norm(full['values'][:,9:11],axis=1),rtol=2e-15,atol=1e-13)
        assert p.diagnostic_download_bytes-start==8*p.count
        rows=np.array([3,1,1,0]);selected=p.evaluate_rows(rows)
        np.testing.assert_array_equal(selected['status'],full['status'][rows])
        np.testing.assert_array_equal(selected['values'],full['values'][rows])
        assert p.evaluate_rows(np.array([],np.int32))['values'].shape==(0,11)
        with pytest.raises(ValueError):p.evaluate_rows(np.array([p.count]))


def test_tiny_cost_progress_with_nonzero_gradient_returns_nonconvergence():
    with optical_problem(offset=[3.,2.],bound=[40.,60.],coupled=False,
                         solver_options=dict(damping_metric='jacobi',initial_lambda=1e10)) as p:
        records=[]
        def progress(record):
            records.append(record)
            return len(records)<5 # Bound a regression of the previous bug.
        result=p.solve(progress=progress,gradient_tolerance=1e-12,relative_cost_tolerance=1e-8)
        assert result['stop_reason']=='cost_progress_stagnation_without_convergence'
        assert not result['numerically_converged']
        assert result['constraint_projected_gradient_inf']>1.
        assert len(records)==1 and records[0]['summary']['accepted']


def test_warm_station_refinement_and_final_global_search_keep_same_optimum():
    f=G2GPU();f.setUp();path=dict(kind=1,parameters=[0.,0.,0.,.3,0.,2*np.pi])
    target=np.array([.05,.3,.6,.95]);kw={**f.kw,'codes':np.zeros(4,np.int32),'stations':target}
    with ResidentResidualProblem(f.core,**kw) as p:
        p.configure_geometry([path]);observed=p.evaluate('cuda')['values'][:,3:5]
    with ResidentResidualProblem(f.core,**{**kw,'stations':target+.005,'observed':observed}) as p:
        p.configure_geometry([path])
        p.configure_pixel_linearization(directions=f.directions,local_ids=np.arange(4,dtype=np.int32),local_scales=np.full(4,.001))
        p.configure_solver(lower=np.zeros(4),upper=np.ones(4),device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
        before=p.export_state();local=p.refine_curve_stations(mode='local');after=p.export_state()
        assert local['native_curve_samples']==0 and local['failed_rows']==0
        assert local['pixel_cost_decrease']>0
        np.testing.assert_allclose(after['stations'],target,atol=1e-8,rtol=0.)
        for key in ('rig','controls','board','layout'):np.testing.assert_array_equal(before[key],after[key])
        assert np.all((after['stations']>=0)&(after['stations']<=1))
        global_check=p.refine_curve_stations(mode='global')
        assert global_check['native_curve_samples']>0
        assert global_check['pixel_cost_decrease']<1e-12
        assert np.max(p.evaluate_errors())<1e-7


@pytest.mark.parametrize('station,u0,u1,v,damping',[
    (.5,60.8,50.,48.8,1e-10),(.2,35.,45.,48.8,.001),
    (.99,72.,54.,49.,.1),(.8,35.,45.,55.,.01),
    (.01,72.,54.,48.8,1.),(.4,60.8,50.,55.,1e-8),
])
def test_bounded_joint_step_matches_independent_cpu_quadratic_oracle(station,u0,u1,v,damping):
    from scipy.optimize import minimize
    f=G2GPU();f.setUp();rig=np.asarray(f.kw['rig']).copy();rig[37]=0.
    kw={**f.kw,'rig':rig,'controls':np.tile([0.,0.,np.log(100.)],(8,1)),
        'codes':np.array([0,1],np.int32),'stations':np.array([station,0.]),
        'times':np.zeros(2),'observed':np.array([[u0,v],[u1,v]])}
    paths=[dict(kind=4,parameters=[],vertices=[[-.9,0.,0.],[.3,0.,0.]]),dict(kind=4,parameters=[],vertices=[[0.,0.,0.]])]
    # Analytic pinhole projection: u=20+40*s+a, v=50+a+b; anchor u=50+a.
    # Scale the station increment by 40 to condition the independent CPU solve.
    j=np.array([[1.,0.,1.],[1.,1.,0.],[1.,0.,0.],[1.,1.,0.]])
    residual=np.array([20+40*station-u0,50-v,50-u1,50-v])
    h=j.T@j+damping*np.diag([4.,2.,1.]);g=j.T@residual
    oracle=minimize(lambda x:.5*x@h@x+g@x,np.zeros(3),jac=lambda x:h@x+g,
        bounds=[(None,None),(None,None),(-40*station,40*(1-station))],
        constraints=[dict(type='ineq',fun=lambda x:x[0]+x[1],jac=lambda x:np.array([1.,1.,0.]))],
        method='SLSQP',options={'ftol':1e-12,'maxiter':1000})
    assert oracle.success,oracle.message
    with ResidentResidualProblem(f.core,**kw) as p:
        p.configure_geometry(paths);rd=np.zeros((27,2));rd[15,0]=1.;rd[16,:]=1.
        lower=np.full(38,-np.inf);upper=np.full(38,np.inf);lower[27]=50.
        p.configure_camera_parameters(directions=rd,lower=lower,upper=upper,max_readout=.02)
        p.configure_pixel_linearization(directions=np.zeros((8,3,2)),local_ids=np.array([0,-1],np.int32),local_scales=np.ones(1))
        p.configure_solver(lower=[0.],upper=[1.],device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30,damping_metric='jacobi',initial_lambda=damping)
        record=p.lm_step();summary=record['summary'];state=p.export_state()
        assert summary['accepted'] and summary['reason_code']==6,summary
        a=state['rig'][26]-50.;dy=state['rig'][27]-50.
        step=np.array([a,dy-a,40*(state['stations'][0]-station)])
        np.testing.assert_allclose(step,oracle.x,atol=2e-6,rtol=1e-8)
        assert summary['trial_domain_failures']==0
        assert summary['predicted_decrease']>0 and summary['actual_decrease']>0
        assert all(value>=0 for value in record['trial_objective_components_ms'].values())


def test_boundary_restoration_inset_remains_active_in_stopping_cone():
    from test_camera_envelope import CameraEnvelopeTests
    from test_constrained_stationarity import assert_initial_constrained_optimum
    c=CameraEnvelopeTests();c.setUp();f=G2GPU();f.setUp()
    rig=c.rig.copy();rig[37]=.02
    # The production boundary projector uses this physical inset. It is still
    # an active face at binary64 resolution, with an outward objective gradient.
    rig[20]=.1+4*(128*np.finfo(float).eps)
    controls=np.tile([0.,.15,np.log(100.)],(8,1))
    kw={**f.kw,'rig':rig,'controls':controls,'codes':np.zeros(4,np.int32)}
    directions=np.zeros((27,2));directions[11,0]=1.;directions[15,1]=1.
    def make(observed=None):
        p=ResidentResidualProblem(f.core,**{**kw,'observed':kw['observed'] if observed is None else observed})
        p.configure_geometry([dict(kind=4,vertices=[[0.,3.,0.]],parameters=[])])
        p.configure_camera_parameters(directions=directions,max_readout=.08)
        p.configure_camera_envelope(forward_bounds=[.1,.3],pupil_bounds=[-.1,.1],max_pan_tilt=c.limits[4],
            max_tilt_deviation=c.limits[5],elevation_bounds=c.limits[6:8],roll_bounds=c.limits[8:10])
        p.configure_pixel_linearization(directions=np.zeros((8,3,2)),local_ids=np.full(4,-1),local_scales=[])
        p.configure_solver(lower=[],upper=[],device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
        return p
    assert_initial_constrained_optimum(make,[1.,0.])
