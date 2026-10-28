"""API-level CUDA checks for the audit's station and physical-bound repairs."""
import os

import numpy as np
import pytest

from rinkspace.core.gpu import ResidentResidualProblem
import test_g2_gpu as source_fixture

pytestmark=pytest.mark.skipif(not os.environ.get('RINKSPACE_G4_LIBRARY'),reason='explicit CUDA/cuDSS library required')


@pytest.mark.parametrize('case', ['finite','unwrapped','lower','upper','unbounded','fixed'])
def test_gpu_periodic_refinement_retains_declared_support_and_other_state(case):
    fixture=source_fixture.G2GPU();fixture.setUp();core=fixture.core;kw=fixture.kw
    period=core.board_perimeter(kw['board']);offset=0. if case=='finite' else period
    original=offset+.45*period
    lower=offset+.4*period if case not in ('upper','unbounded') else -np.inf
    upper=offset+.5*period if case not in ('lower','unbounded') else np.inf
    if case=='fixed':lower=upper=original
    kw={**kw,'controls':np.tile([0.,0.,np.log(100.)],(8,1)),
        'codes':np.array([0,1],np.int32),'stations':np.array([.9*period,0.]),
        'times':np.zeros(2),'observed':np.zeros((2,2))}
    paths=[dict(kind=7,parameters=[.5]),dict(kind=4,vertices=[[0.,0.,0.]],parameters=[])]
    with ResidentResidualProblem(core,**kw) as p:
        p.configure_geometry(paths);observed=p.evaluate('cuda')['values'][:,3:5]
    kw.update(observed=observed,stations=np.array([original,0.]))
    with ResidentResidualProblem(core,**kw) as p:
        p.configure_geometry(paths)
        p.configure_pixel_linearization(directions=np.zeros((8,3,1)),local_ids=np.array([0,-1]),local_scales=np.ones(1))
        p.configure_solver(lower=[lower],upper=[upper],device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
        before=p.export_state();old=p.evaluate('cuda');result=p.refine_curve_stations();after=p.export_state();new=p.evaluate('cuda')
        assert lower<=after['stations'][0]<=upper
        assert after['stations'][1]==before['stations'][1]
        np.testing.assert_array_equal(new['status'],0)
        old_cost=.5*np.sum(old['values'][:,9:11]**2,axis=1);new_cost=.5*np.sum(new['values'][:,9:11]**2,axis=1)
        assert np.all(new_cost<=old_cost)
        assert result['failed_rows']==0
        if case=='fixed':np.testing.assert_array_equal(after['stations'],before['stations'])
        else:assert new_cost[0]<old_cost[0]
        for key in before.keys()-{'stations'}:np.testing.assert_array_equal(after[key],before[key])


def boundary_problem(angle, *, narrow=False):
    fixture=source_fixture.G2GPU();fixture.setUp();core=fixture.core;kw=fixture.kw
    rig=np.asarray(kw['rig']).copy();rig[26:28]=0.
    kw={**kw,'rig':rig,'controls':np.tile([0.,0.,np.log(100.)],(8,1))}
    with ResidentResidualProblem(core,**kw) as p:observed=p.evaluate('cuda')['values'][:,3:5]
    if not narrow:observed+=np.array([0.,1.])
    p=ResidentResidualProblem(core,**{**kw,'observed':observed})
    directions=np.zeros((27,2));directions[15]=[1.,0.] if narrow else [1.,-angle];directions[16]=[1.,angle]
    lower=np.full(38,-np.inf);upper=np.full(38,np.inf)
    if narrow:lower[27]=-1e-15;upper[27]=1e-15
    else:lower[26]=0.;upper[27]=0.
    p.configure_camera_parameters(directions=directions,max_readout=.08,lower=lower,upper=upper)
    p.configure_pixel_linearization(directions=np.zeros((8,3,2)),local_ids=np.full(4,-1),local_scales=np.zeros(0))
    p.configure_solver(lower=[],upper=[],device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
    return p


@pytest.mark.parametrize('dual_backend', ['cpu','gpu'])
def test_gpu_legal_narrow_coupled_bounds_return_without_a_false_step(dual_backend,monkeypatch):
    monkeypatch.setenv('RINKSPACE_SMALL_DUAL_BACKEND',dual_backend)
    with boundary_problem(1.,narrow=True) as p:
        before=p.export_state();result=p.lm_step()['summary'];assert not result['accepted']
        for key,value in before.items():np.testing.assert_array_equal(p.export_state()[key],value)
        p.request_stop();assert p.solve()['stop_reason']=='stopped_by_caller'


@pytest.mark.parametrize('dual_backend', ['cpu','gpu'])
def test_gpu_nearly_parallel_faces_preserve_constrained_optimum_and_stop_boundary(dual_backend,monkeypatch):
    monkeypatch.setenv('RINKSPACE_SMALL_DUAL_BACKEND',dual_backend)
    with boundary_problem(.001) as p:
        before=p.export_state()
        # The Schur-metric solve replaces the old slow Euclidean projection.
        # Its constrained optimum is the input state, so it must reject a trial
        # with no decrease and return at a safe boundary.
        result=p.lm_step()['summary'];assert not result['accepted'],result
        assert result['trial_domain_failures']==0
        for key,value in before.items():np.testing.assert_array_equal(p.export_state()[key],value)
        p.request_stop();assert p.solve()['stop_reason']=='stopped_by_caller'


def test_native_missing_readout_support_has_no_derivative_channel_access():
    from test_camera_envelope import CameraEnvelopeTests
    fixture=CameraEnvelopeTests();fixture.setUp()
    fixture.rig[37]=.02
    controls=np.tile([0.,.15,np.log(100.)],(8,1));limits=np.r_[fixture.limits,np.full(9,np.inf),1.,6.]
    result=fixture.core.evaluate_camera_envelope(fixture.rig,controls,0.,1.,limits)
    assert result['kind']==10 and result['channel']==-1
    assert result['derivative_order']==0 and result['derivative']==0.
    assert result['margin']==-np.inf
