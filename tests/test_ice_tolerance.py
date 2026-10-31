"""Ice tolerance geometry, derivatives and finite synthetic fit regressions."""
import os
import numpy as np
import pytest
from rinkspace.core.native import NativeCore, NativeError
from rinkspace.core.gpu import ResidentResidualProblem
from rinkspace.domain.ice_tolerance import DEVIATION_COUNT, LAYOUT_SIZE
import test_g2_gpu as fixture


def nominal():
    return np.array([-.3,.2,0.,.1,.03,.1,.35,.65,.31,.07,-.42,-.43,-.41,-.44,.28,.28,.28,.28,
        0.,.12,.06,-.24,.24,0.,.45,.3,0.,.11,0.,.035,.025,0.,0.,.05,.07,-.23,0.,0.,0.])


@pytest.fixture
def core():
    library=os.environ.get('RINKSPACE_CORE_LIBRARY')
    if not library:pytest.skip('explicit native CUDA library required')
    return NativeCore(library)


def test_zero_deviations_keep_every_ice_primitive_and_legacy_option(core):
    board=[.5,.25,.1,.12,1.85,.02]
    codes=[*range(100,132),*range(200,204),*range(300,312),*range(320,324),*range(400,412),*range(420,424)]
    q=nominal();extended=np.r_[q,np.zeros(DEVIATION_COUNT)]
    assert len(extended)==LAYOUT_SIZE
    for code in codes:
        before=core.shared_geometry(board,q,code,options=63)
        after=core.shared_geometry(board,extended,code,options=127)
        assert before[0]==after[0]
        np.testing.assert_allclose(after[1],before[1],rtol=0,atol=2e-13)
    for bad in (.010001,-.010001):
        extended[39+16]=bad
        with pytest.raises(NativeError):core.shared_geometry(board,extended,100,options=127)


def test_cuda_all_new_scene_columns_and_independent_fd(core,monkeypatch):
    monkeypatch.setenv('RINKSPACE_G2_LIBRARY',str(core.library_path))
    seed=fixture.G2GPU();seed.setUp()
    codes=np.array([*range(100,132),*range(200,204),*range(300,312),*range(320,324),*range(400,412),*range(420,424)])
    board=np.array([.5,.25,.1,.12,1.85,.02])
    values=np.r_[board,nominal(),np.zeros(DEVIATION_COUNT)]
    indices=6+39+np.arange(DEVIATION_COUNT)
    directions=np.eye(len(values))[:,indices]*.001
    rows=len(codes)*3
    records=np.zeros((len(codes),12),np.int32);records[:,0]=1;records[:,1]=codes
    def make(state,attach=True):
        kw={**seed.kw, 'board':state[:6], 'layout':state[6:], 'options':127,
            'codes':np.repeat(np.arange(len(codes),dtype=np.int32),3),
            'stations':np.tile([.17,.51,.83],len(codes)),
            'times':np.linspace(0,.1,rows),'observed':np.zeros((rows,2))}
        p=ResidentResidualProblem(core,**kw)
        try:
            paths=[]
            for code in codes:
                kind,parameters=core.shared_geometry(state[:6],state[6:],int(code),options=127)
                paths.append(dict(kind=kind,parameters=parameters))
            p.configure_geometry(paths)
            if attach:p.configure_scene_parameters(values=state,records=records,data=np.zeros(1),directions=directions)
        except BaseException:p.close();raise
        return p
    with make(values) as problem:
        sparse=problem.configure_pixel_linearization(directions=np.zeros((8,3,DEVIATION_COUNT)),
            local_ids=np.full(rows,-1,np.int32),local_scales=np.zeros(0))
        assert sparse['max_row_q'] < DEVIATION_COUNT/4, 'unrelated ice deviations must not inflate each observation block'
        problem.linearize('cuda');jac=problem.probe_rows(np.arange(rows))[:,:,:DEVIATION_COUNT]
        problem.linearize('cpu')
        np.testing.assert_allclose(jac,problem.probe_rows(np.arange(rows),'cpu')[:,:,:DEVIATION_COUNT],atol=3e-10,rtol=1e-8)
        assert np.isfinite(jac).all()
        assert np.all(np.linalg.norm(jac,axis=(0,1))>0)
        for j in [0,3,4,8,12,16,47,48,52,56,60,64,68,72,77,82,86,90,95,99,103]:
            uv=[];epsilon=.002
            for sign in (-1,1):
                with make(values+sign*epsilon*directions[:,j],attach=False) as sample:
                    ev=sample.evaluate('cuda');np.testing.assert_array_equal(ev['status'],0)
                    uv.append(ev['values'][:,3:5])
            np.testing.assert_allclose(jac[:,:,j],(uv[1]-uv[0])/(2*epsilon),atol=3e-7,rtol=3e-4,
                                       err_msg=f'ice deviation {j}')
    bad=values.copy();bad[6+39+16]=.010001
    with pytest.raises(RuntimeError,match='bounded ice geometry'):
        with make(bad,attach=False):pass


def fit_fixture(core, code, active, target_parameters):
    """One synthetic finite LM trial; footage and saved production states absent."""
    seed=fixture.G2GPU();seed.setUp()
    board=np.array([.5,.25,.1,.12,1.85,.02])
    values=np.r_[board,nominal(),np.zeros(DEVIATION_COUNT)]
    kind,parameters=core.shared_geometry(board,values[6:],code,options=127)
    rows=17
    kw={**seed.kw,'board':board,'layout':values[6:],'options':127,
        'codes':np.zeros(rows,np.int32),'stations':np.linspace(.03,.97,rows),
        'times':np.linspace(0,.1,rows),'observed':np.zeros((rows,2))}
    with ResidentResidualProblem(core,**kw) as target:
        target.configure_geometry([dict(kind=kind,parameters=target_parameters)])
        kw['observed']=target.evaluate('cuda')['values'][:,3:5].copy()
    directions=np.zeros((len(values),len(active)))
    for j,index in enumerate(active):directions[6+39+index,j]=1.
    p=ResidentResidualProblem(core,**kw)
    p.configure_geometry([dict(kind=kind,parameters=parameters)])
    records=np.zeros((1,12),np.int32);records[0,:2]=[1,code]
    # No caller bounds: mandatory +/-1% is imposed by native attach itself.
    p.configure_scene_parameters(values=values,records=records,data=np.zeros(1),directions=directions)
    p.configure_pixel_linearization(directions=np.zeros((8,3,len(active))),
        local_ids=np.full(rows,-1,np.int32),local_scales=np.zeros(0))
    p.configure_solver(lower=np.zeros(0),upper=np.zeros(0),
        device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30,
        damping_metric='jacobi',initial_lambda=.001)
    return p,values


@pytest.mark.parametrize('sign',[-1.,1.])
def test_one_synthetic_fit_stops_at_mandatory_nominal_radius_bound(core,monkeypatch,sign):
    monkeypatch.setenv('RINKSPACE_G2_LIBRARY',str(core.library_path))
    monkeypatch.setenv('RINKSPACE_SMALL_DUAL_BACKEND','cpu')
    monkeypatch.setenv('RINKSPACE_STATIONARITY_BACKEND','cpu')
    q=nominal();board=[.5,.25,.1,.12,1.85,.02]
    _,target=core.shared_geometry(board,np.r_[q,np.zeros(DEVIATION_COUNT)],100,options=127)
    target[3]*=1+sign*.02
    with fit_fixture(core,100,[16],target)[0] as p:
        step=p.lm_step()['summary']
        assert step['accepted'],step
        assert step['trial_domain_failures']==0
        assert step['trial_cost']<step['old_cost']
        delta=p.export_state()['layout'][39+16]
        assert abs(delta)<=.01
        assert delta==pytest.approx(sign*.01,abs=2e-8)
        p.linearize('cuda')
        diagnostic=p.stationarity_diagnostics(include_constraints=True)
        assert diagnostic['invalid']==0
        assert diagnostic['physical_feasibility_violation']<1e-10


def test_one_synthetic_fit_respects_coupled_collinear_endpoint_face(core,monkeypatch):
    monkeypatch.setenv('RINKSPACE_G2_LIBRARY',str(core.library_path))
    monkeypatch.setenv('RINKSPACE_SMALL_DUAL_BACKEND','cpu')
    monkeypatch.setenv('RINKSPACE_STATIONARITY_BACKEND','cpu')
    q=nominal();width=q[9];y=-1.5*q[3]
    target=[q[8]+.015*width,y+width/2,0.,q[8]+.005*width,y-width/2,0.]
    with fit_fixture(core,301,[56,60],target)[0] as p:
        step=p.lm_step()['summary']
        assert step['accepted'],step
        assert step['trial_domain_failures']==0
        state=p.export_state();delta=state['layout'][39:]
        assert np.max(np.abs(delta))<=.01
        _,line=core.shared_geometry(state['board'],state['layout'],301,options=127)
        assert max(abs(line[0]-q[8]),abs(line[3]-q[8]))<=.01*width+1e-11
        assert line[0]-q[8]==pytest.approx(.01*width,abs=2e-8)
        p.linearize('cuda')
        diagnostic=p.stationarity_diagnostics(include_constraints=True)
        assert diagnostic['invalid']==0
        assert diagnostic['physical_feasibility_violation']<1e-10
        assert np.any(diagnostic['constraint_records'][:,0]==6)

@pytest.mark.parametrize('dual_backend',['cpu','gpu'])
def test_more_than_64_constraint_cuts_preserve_feasible_descent(core,monkeypatch,dual_backend):
    """107 independent ice deviations need more cuts than the initial workspace.

    Synthetic UV targets are first-order offsets beyond the mandatory bounds.
    A fixed 64-cut fallback used to erase the shared step at this same trigger.
    """
    monkeypatch.setenv('RINKSPACE_G2_LIBRARY',str(core.library_path))
    monkeypatch.setenv('RINKSPACE_SMALL_DUAL_BACKEND',dual_backend)
    monkeypatch.setenv('RINKSPACE_STATIONARITY_BACKEND','cpu')
    seed=fixture.G2GPU();seed.setUp()
    codes=np.array([*range(100,132),*range(200,204),*range(300,312),*range(320,324),*range(400,412),*range(420,424)])
    values=np.r_[[.5,.25,.1,.12,1.85,.02],nominal(),np.zeros(DEVIATION_COUNT)]
    q=DEVIATION_COUNT;directions=np.eye(len(values))[:,45:]*.001
    paths=[]
    for code in codes:
        kind,parameters=core.shared_geometry(values[:6],values[6:],int(code),options=127)
        paths.append(dict(kind=kind,parameters=parameters))
    records=np.zeros((len(codes),12),np.int32);records[:,0]=1;records[:,1]=codes
    rows=len(codes)*9
    kw={**seed.kw,'board':values[:6],'layout':values[6:],'options':127,
        'codes':np.repeat(np.arange(len(codes),dtype=np.int32),9),
        'stations':np.tile(np.linspace(.03,.97,9),len(codes)),
        'times':np.linspace(0,.1,rows),'observed':np.zeros((rows,2))}
    def make():
        p=ResidentResidualProblem(core,**kw)
        try:
            p.configure_geometry(paths)
            p.configure_scene_parameters(values=values,records=records,data=np.zeros(1),directions=directions)
            p.configure_pixel_linearization(directions=np.zeros((8,3,q)),
                local_ids=np.full(rows,-1,np.int32),local_scales=np.zeros(0))
        except BaseException:p.close();raise
        return p
    with make() as p:
        uv=p.evaluate('cuda')['values'][:,3:5]
        p.linearize('cuda');jac=p.probe_rows(np.arange(rows))[:,:,:q]
    kw['observed']=uv+np.einsum('ijk,k->ij',jac,np.full(q,30.))
    with make() as p:
        p.configure_solver(lower=np.zeros(0),upper=np.zeros(0),
            device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30,
            damping_metric='jacobi',initial_lambda=.001)
        allocated=p.solver_info['owned_device_bytes']
        step=p.lm_step()['summary']
        assert step['accepted'],step
        assert step['trial_domain_failures']==0
        assert step['trial_cost']<.7*step['old_cost']
        # Growth from 64 to 128 retains gradients, responses and the pitched
        # Gram matrix. No extra point/station storage is allocated by this step.
        growth=(2*q*64+(128**2-64**2)+3*64)*8
        assert step['solver_device_bytes']-allocated>=growth
        state=p.export_state()
        assert np.max(np.abs(state['layout'][39:]))<=.01
        np.testing.assert_array_equal(p.evaluate('cuda')['status'],0)
        p.linearize('cuda');diagnostic=p.stationarity_diagnostics(include_constraints=True)
        assert diagnostic['physical_feasibility_violation']<1e-10
