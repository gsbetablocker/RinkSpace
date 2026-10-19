from dataclasses import dataclass, replace
from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from calibration.solver_runtime.current_principal_point_working_prior import (
    CurrentPP6WorkingPriorBlock, SourceWithCurrentPP6WorkingPrior,
    wrap_current_principal_point_working_prior)


@dataclass(frozen=True)
class Calibration:
    installation_id: str
    principal_x_fraction: tuple
    principal_y_fraction: tuple
    output_mode: object


@dataclass(frozen=True)
class Chart:
    indices: tuple
    def calibration(self, base, q):
        return replace(base,principal_x_fraction=(base.principal_x_fraction[0]+q[self.indices[0]]/1920.,),
                       principal_y_fraction=(base.principal_y_fraction[0]+q[self.indices[1]]/1080.,))


def fixture():
    from calibration.solver_runtime import install
    install()
    from z190_streaming_source_normal import SourceRowBlock
    permutation=np.array([4,0,8,2,6,1,9,3,7,5])
    layout=SimpleNamespace(canonical_indices=lambda ix:permutation[np.asarray(ix,int)])
    block=SourceRowBlock(np.array([1.,2.]),csr_matrix(np.array([[1.]+[0.]*9,[0.,2.]+[0.]*8])),
        np.array([False,True]),np.array([4.,7.]),dict(family='original'))
    source=SimpleNamespace(count=10,quadratic=block.quadratic,scales=block.scales,
        contract=dict(source_rows=2),parameter_layout=layout,
        evaluation_core=lambda q:np.asarray(q)[permutation],
        blocks=lambda q,jacobian=True:iter([block]),marker='preserved')
    mode=SimpleNamespace(domain_wh=(1920,1080),final_wh=(1920,1080),A=np.eye(3))
    charts={};blocks=[];ids=('main','secondary','low')
    for i,name in enumerate(ids):
        cal=Calibration(name,(.42+i*.01,),(.62-i*.01,),mode)
        chart=Chart((3*i,3*i+1,3*i+2));charts[name]=chart
        blocks.append(SimpleNamespace(bundle=SimpleNamespace(calibration=cal),
            optics_base_calibration=cal,optics_chart=chart))
    return SimpleNamespace(ids=ids,blocks=blocks,installation_optics=charts),source,block


def test_reference_is_original_image_center_and_exact_current_J():
    state,source,_=fixture();factor=CurrentPP6WorkingPriorBlock(state,source)
    q=np.linspace(-4,5,10);original=q.copy();r,J=factor.evaluate(q)
    assert np.array_equal(q,original)
    np.testing.assert_allclose(r,((factor.native_principal_points(q)-[960,540])/32).ravel())
    assert J.nnz==6
    for column in range(len(q)):
        plus=q.copy();minus=q.copy();plus[column]+=.001;minus[column]-=.001
        np.testing.assert_allclose((factor.evaluate(plus,False)-factor.evaluate(minus,False))/.002,
                                  J.toarray()[:,column],atol=2e-10,rtol=0)
    assert factor.contract['original_target_not_recentered_on_current_or_seed_fit']
    assert factor.contract['prior_counted_once_per_physical_installation']
    assert len(factor.contract['implementation_sources']) == 2
    for record in factor.contract['implementation_sources']:
        assert Path(record['path']).is_file() and len(record['sha256']) == 64


def test_appended_rows_preserve_original_block_values_J_weights_and_slots():
    state,base,original=fixture();source=wrap_current_principal_point_working_prior(base,state)
    blocks=list(source.blocks(np.zeros(10),True))
    assert blocks[0] is original
    assert source.count==10 and len(source.scales)==8
    assert np.array_equal(source.scales[:2],base.scales)
    assert np.array_equal(source.quadratic[:2],base.quadratic)
    assert blocks[1].row_ledger['original_start']==2
    assert blocks[1].row_ledger['original_stop']==8
    assert blocks[1].row_ledger['source_role']=='engineering_working_hypothesis_prior'
    assert source.marker=='preserved'
    blocks=list(source.blocks(np.zeros(10),False));assert blocks[1].J is None
    with pytest.raises(ValueError,match='already installed'):
        wrap_current_principal_point_working_prior(source,state)


def test_reject_non_native_domain_and_inconsistent_physical_installation():
    state,source,_=fixture()
    state.blocks[0].optics_base_calibration.output_mode.A[0,0]=2.
    with pytest.raises(ValueError,match='native identity'):
        CurrentPP6WorkingPriorBlock(state,source)
    state,source,_=fixture()
    other=state.blocks[0]
    state.blocks.append(SimpleNamespace(bundle=other.bundle,optics_chart=other.optics_chart,
        optics_base_calibration=replace(other.optics_base_calibration,principal_x_fraction=(.5,))))
    with pytest.raises(ValueError,match='producers disagree'):
        CurrentPP6WorkingPriorBlock(state,source)
