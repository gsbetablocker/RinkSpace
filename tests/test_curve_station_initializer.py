"""Test the repository initializer with a deterministic projector, not a GPU."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

REPO=Path(os.environ.get('RINKSPACE_AUDIT_REPO',Path(__file__).resolve().parents[1]))

def support(path):
    values=np.asarray(path.get('support',path.get('finite_support',[0.,1.])),float)
    if values.shape!=(2,) or not np.isfinite(values).all() or not 0<=values[0]<=values[1]<=1:
        raise ValueError('invalid finite support in fixture')
    return tuple(values)

@pytest.fixture
def initialize():
    path=REPO/'src/rinkspace/session/gpu_geometry.py'
    tree=ast.parse(path.read_text());tree.body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='initialize_curve_stations']
    scope={'finite_path_support':support}
    exec(compile(tree,str(path),'exec'),scope)
    return scope['initialize_curve_stations']

class FixtureResident:
    def __init__(self,native,parameters):self.native=native;self.parameters=parameters
    def __enter__(self):return self
    def __exit__(self,*a):pass
    def configure_projection_time_support(self,**kw):self.native.time_support=kw
    def evaluate(self,backend):
        s=self.parameters['stations'];pts=self.parameters['pts'];n=len(s)
        values=np.zeros((n,11));values[:,3]=s*100
        status=np.isin(pts,self.native.failed_pts).astype(np.int32)
        return {'values':values,'status':status}

class FixtureNative:
    def __init__(self,kind=0,bounds=(0.,1.),failed_pts=()):
        self.paths={'p':dict(id='p',kind=kind,support=bounds,knots=[0.,.3,.7,1.])}
        self.scene={'board':[1.]};self.core=SimpleNamespace(board_perimeter=lambda b:4.)
        self.failed_pts=failed_pts;self.calls=[]
    def resident(self,**kw):
        s=np.asarray(kw['stations'])
        for name,x in zip(kw['path_ids'],s):
            p=self.paths[name]
            if p['kind']!=7:
                lo,hi=support(p)
                if not lo<=x<=hi:raise ValueError('original station outside selected native scene support')
        self.calls.append(kw)
        return FixtureResident(self,kw)

def call(fn,native,stations,*,kinds=None,pts=None,observed=None):
    n=len(stations)
    return fn(native,camera={},controls=np.zeros((4,3)),path_ids=['p']*n,stations=stations,
        pts=np.arange(n,dtype=np.int64)+1 if pts is None else np.asarray(pts,np.int64),
        observed=np.tile([73.125,0.],(n,1)) if observed is None else np.asarray(observed,float),
        kinds=[1]*n if kinds is None else kinds,backend='cpu')

@pytest.mark.parametrize('kind',[0,1,6,8])
def test_truncated_material_domain_not_renormalized(initialize,kind):
    native=FixtureNative(kind,bounds=(.2,.8));seed=np.array([.5]);before=seed.copy()
    result,report=call(initialize,native,seed)
    np.testing.assert_allclose(result,[.73125],rtol=0,atol=1e-15)
    np.testing.assert_array_equal(seed,before)
    grid=native.calls[0]['stations'];assert grid.min()==.2 and grid.max()==.8
    assert not report['unresolved']

@pytest.mark.parametrize('kind',[0,8])
def test_single_station_support_remains_valid(initialize,kind):
    result,report=call(initialize,FixtureNative(kind,bounds=(.3,.3)),[.3])
    np.testing.assert_array_equal(result,[.3]);assert not report['unresolved']

def test_integer_seed_not_truncate_fractional_solution(initialize):
    seed=np.array([0],np.int32);result,report=call(initialize,FixtureNative(),seed)
    np.testing.assert_allclose(result,[.73125],rtol=0,atol=1e-15)
    assert result.dtype==np.dtype('float64');np.testing.assert_array_equal(seed,[0])

@pytest.mark.parametrize('n',[0,2])
def test_no_curve_rows_are_noop(initialize,n):
    native=FixtureNative();seed=np.zeros(n)
    result,report=call(initialize,native,seed,kinds=[0]*n)
    np.testing.assert_array_equal(result,seed)
    assert not native.calls and report==dict(unresolved=[],groups=0,seed_error_percentiles=[])

def test_all_projection_failures_return_records_not_percentile_exception(initialize):
    native=FixtureNative(failed_pts=[1,2]);seed=np.array([.4,.6]);result,report=call(initialize,native,seed)
    np.testing.assert_array_equal(result,seed)
    assert len(report['unresolved'])==2 and report['seed_error_percentiles']==[]

def test_mixed_failure_retains_every_observation(initialize):
    native=FixtureNative(failed_pts=[2]);result,report=call(initialize,native,[.4,.6])
    np.testing.assert_allclose(result,[.73125,.6],rtol=0,atol=1e-15)
    assert len(report['unresolved'])==1 and len(report['seed_error_percentiles'])==4

def test_periodic_course_keeps_existing_sampling_chart(initialize):
    native=FixtureNative(7,bounds=None);result,report=call(initialize,native,[.5])
    np.testing.assert_allclose(result,[.73125],rtol=0,atol=1e-15)
    grid=native.calls[0]['stations'];assert len(grid)==1025 and grid.min()==0 and grid.max()==4
