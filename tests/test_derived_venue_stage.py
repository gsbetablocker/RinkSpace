from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import hashlib

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.program_tracking.first_shot_derived_venue_stage import DerivedVenueStage


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@dataclass
class Block:
    r: object
    J: object
    quadratic: object
    scales: object
    row_ledger: object
    loss_weights: object


class Chart:
    count = 6
    fixed_canonical_rows = np.array([1])
    released_rows = (2,)
    recipe_reference = ref(__file__)
    def identity(self):
        return dict(schema='nonlinear-chain-test-fixture', derived_rows=[1])
    def effective_q_and_chain(self, q):
        value = q.copy(); value[1] = 2*q[0]**2
        E = np.eye(6); E[1] = 0.; E[1,0] = 4*q[0]
        return value, csr_matrix(E)
    def pullback_constraints(self, original, q, physical_columns, *, canonical_columns, fixed_columns):
        _, E = self.effective_q_and_chain(q)
        return dict(original, matrix=csr_matrix(original['matrix'])@E[physical_columns][:,canonical_columns])


class Physical:
    parameter_count = 6
    programme_count = 4
    shared = np.arange(5)
    fixed = np.array([5])
    scales = np.ones(2)
    quadratic = np.zeros(2, bool)
    source_producers = []
    recovery_references = []
    def __init__(self):
        self.programme = SimpleNamespace(name='existing-owner')
        self.seen = []
        self.state = SimpleNamespace(world_parameter_indices=np.array([0,1,2]),
            scene=lambda q: dict(physical=q.copy()))
        self.recovery = lambda parent, q: dict(q=q.copy(), report={})
    def solver_kwargs(self):
        return dict(shared_parameter_bounds={1:(4.,12.)}, global_schur_count=5,
            global_schur_indices=self.shared, fixed_parameter_indices=self.fixed)
    def checkpoint_arrays(self):
        return dict(static_venue_initial_camera_state=np.array([-3.,-4.]),old_reference=np.array([17]))
    def checkpoint_identity(self):
        return dict(schema='physical-fixture', original=True)
    def guard(self, q):
        self.seen.append(q.copy()); return True
    def source_blocks(self, q, jacobian):
        self.seen.append(q.copy())
        J=np.array([[0.,1.,1.,0.,0.,0.],[0.,0.,0.,1.,-1.,0.]])
        yield Block(J@q,csr_matrix(J)if jacobian else None,self.quadratic,self.scales,
                    dict(clock='independent'),np.array([2.,3.]))
    def world(self, q, jacobian):
        return {'curve':q[:3][None]}, {'curve':np.eye(3)[None]}if jacobian else None
    def raw_programme(self, q, **kwargs):return q[:4]
    def split_state(self, q):return q[:4],q[4:]
    def raw_static(self, q):return q[4:]
    def raw(self, q, **kwargs):return q.copy()
    def cost(self, q):return float(q@q)
    def linear_constraints(self, q, selected):
        assert np.array_equal(selected,self.shared)
        M=np.zeros((1,5));M[0,1]=3.
        return dict(matrix=M,lower=np.array([7.-3*q[1]]),unchanged='retained',report={'original':True})
    def recover(self, parent, q):return self.recovery(parent,q)
    def complete_state_payload(self,q):return dict(q_augmented=q.copy(),qbase=q[:3].copy(),programme_q_augmented=q[:4].copy())
    def close(self):self.closed=True


@pytest.fixture
def wrapped():
    physical=Physical();q=np.array([2.,999.,1.,4.,5.,0.])
    aux={'static_venue_initial_camera_state':np.array([-9.,-8.]),'old_reference':np.array([17])}
    result=DerivedVenueStage(physical,Chart(),canonical_start=q,inherited_aux=aux,entry_reference=ref(__file__))
    return result,physical,q,aux


def test_all_values_and_derivatives_use_same_physical_state(wrapped):
    stage,physical,q,_=wrapped
    r,J=stage.evaluate(q,True);qe,E=stage.effective_q_and_chain(q)
    np.testing.assert_array_equal(physical.seen[-1],qe)
    np.testing.assert_array_equal(r,[9.,-1.])
    np.testing.assert_array_equal(J.toarray()[0],[8.,0.,1.,0.,0.,0.])
    h=1e-6;d=np.array([.1,0.,.2,.3,.4,0.])
    np.testing.assert_allclose((stage.evaluate(q+h*d)-stage.evaluate(q-h*d))/(2*h),J@d,atol=2e-8)
    np.testing.assert_array_equal(stage.state.scene(q)['physical'],qe)
    np.testing.assert_array_equal(stage.raw(q),qe)
    np.testing.assert_array_equal(stage.split_state(q)[0],qe[:4])
    assert stage.cost(q)==qe@qe
    worlds,D=stage.world(q,True)
    np.testing.assert_array_equal(worlds['curve'],qe[:3][None])
    np.testing.assert_array_equal(D['curve'][0],[[1.,0.,0.],[8.,0.,0.],[0.,0.,1.]])


def test_original_bounds_and_halfspaces_are_pulled_back_at_current_physical_value(wrapped):
    stage,_,q,_=wrapped
    result=stage.linear_constraints(q,stage.shared)
    assert result['unchanged']=='retained'
    np.testing.assert_array_equal(result['lower'],[-17.,-4.,-4.])
    np.testing.assert_array_equal(result['matrix'].toarray(),[[24.,0.,0.,0.],[8.,0.,0.,0.],[-8.,0.,0.,0.]])
    assert stage.solver_kwargs()['shared_parameter_bounds']=={}
    invalid=q.copy();invalid[0]=1.
    with pytest.raises(ValueError,match='original physical bound'):stage.guard(invalid)


def test_raw_fixed_elimination_and_aux_parent_identity(wrapped):
    stage,_,q,aux=wrapped
    np.testing.assert_array_equal(stage.shared,[0,2,3,4])
    np.testing.assert_array_equal(stage.fixed,[5,1])
    np.testing.assert_array_equal(stage.start,q)
    for k,v in aux.items():np.testing.assert_array_equal(stage.checkpoint_arrays()[k],v)
    record=stage.complete_state_payload(q)
    np.testing.assert_array_equal(record['q_augmented'],q)
    assert record['physical_q_augmented'][1]==8.
    np.testing.assert_array_equal(record['qbase'],q[:3])
    np.testing.assert_array_equal(record['physical_qbase'],[2.,8.,1.])
    assert stage.checkpoint_identity()['derived_venue_stage']['chart']['installed_in_this_explicit_stage']
    assert record['canonical_q_sha256']!=record['physical_q_sha256']
    changed=q.copy();changed[1]+=1.
    with pytest.raises(ValueError,match='Fixed raw'):stage.evaluate(changed)


def test_recovery_requires_exact_chart_consistency(wrapped):
    stage,physical,q,_=wrapped
    physical.recovery=lambda parent,value:dict(q=value+np.array([0.,0.,.5,0.,0.,0.]),report={'kept':True})
    result=stage.recover(q,q)
    assert result['q'][1]==999. and result['q'][2]==1.5 and result['report']['kept']
    physical.recovery=lambda parent,value:dict(q=value+np.array([.1,0.,0.,0.,0.,0.]),report={})
    with pytest.raises(ValueError,match='incompatible with the derived chart'):stage.recover(q,q)


def test_wrapper_does_not_mutate_physical_owner(wrapped):
    stage,physical,_,_=wrapped
    np.testing.assert_array_equal(physical.shared,np.arange(5))
    np.testing.assert_array_equal(physical.fixed,[5])
    assert physical.source_producers==[]
    assert stage.programme is physical.programme
