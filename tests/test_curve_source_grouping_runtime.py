from copy import deepcopy
import hashlib
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.solver_runtime import install
install()
from z190_streaming_source_normal import (SourceRowBlock, accumulate_source_normal,
    replay_source_cost, replay_native_step_maxima, _checked_block, _array_digest, _ledger_digest)
from z190_joint_source_solver import source_cost
from structured_programme_source_candidate.lagged_jacobian_source import LaggedJacobianSource
from calibration.solver_runtime.source_loss import SourceLossWeights, source_irls_weights
from calibration.solver_runtime.curve_source_grouping import (
    SCHEMA, wrap_curve_source_grouping, source_contract_sha256,
)
from calibration.solver_runtime.source_point_qualification import file_sha256, original_uv_sha256


def accumulated(block, profile='frozen-original-profile', retain=False):
    return accumulate_source_normal(lambda: iter([block]), block.J.shape[1],
        expected_row_count=len(block.r), frozen_profile_token=profile,
        retain_native_replay=retain, source_state=np.zeros(block.J.shape[1]) if retain else None)


def test_weighted_original_streaming_huber_cost_gradient_normal_and_native_units():
    # The upstream source already applied actual profile/working sigma. This
    # adapter must not undo profile normalization or move the robust cutoff.
    profile=np.array([.25,.81,.49,1.,.36,1.])
    native=np.array([.2,-1.,3.,-8.,20.,-30.])
    r=native*np.sqrt(profile)/2.
    J=csr_matrix(np.arange(18.).reshape(6,3)*np.sqrt(profile)[:,None]/2.+.1)
    mask=np.array([False,False,False,False,True,True]);scales=2./np.sqrt(profile)
    a=np.array([.1,.1,.2,.2,.3,.3]);c=np.array([1.,1.,3.,3.,2.,2.])
    payload=SourceLossWeights(a,c)
    block=SourceRowBlock(r,J,mask,scales,dict(family='real-normalized-layout',original_start=0,original_stop=6),payload)
    result=accumulated(block,retain=True)
    working=r*c;Jworking=J.toarray()*c[:,None]
    robust=np.minimum(1.,2./np.maximum(abs(working),1e-12));robust[mask]=1.
    expected_H=Jworking.T@((a*robust)[:,None]*Jworking)
    expected_g=Jworking.T@((a*robust)*working)
    losses=np.where(abs(working)<=2.,.5*working**2,2.*abs(working)-2.);losses[mask]=.5*working[mask]**2
    assert np.allclose(result.normal.toarray(),expected_H,rtol=2e-14,atol=2e-14)
    assert np.allclose(result.gradient,expected_g,rtol=2e-14,atol=2e-14)
    assert result.cost==float(a@losses)==source_cost(r,mask,payload)
    assert replay_source_cost(lambda:iter([SourceRowBlock(r,None,mask,scales,block.row_ledger,payload)]),
        result.contract,frozen_profile_token='frozen-original-profile')==result.cost
    step=np.array([.01,-.02,.03])
    native_max=replay_native_step_maxima(lambda:iter([block]),step,result.contract,
        frozen_profile_token='frozen-original-profile',native_replay_cache=result.native_replay_cache,source_state=np.zeros(3))
    assert native_max==float(np.max(abs(J@step)*scales))
    # Finite differences of the optional robust cost agree with the exact g.
    h=1e-6
    numerical=np.array([(source_cost(r+h*J.toarray()[:,i],mask,payload)-source_cost(r-h*J.toarray()[:,i],mask,payload))/(2*h)for i in range(3)])
    assert np.allclose(numerical,result.gradient,rtol=1e-8,atol=1e-7)


def test_default_none_preserves_original_cost_normal_and_exact_inventory_digest():
    r=np.array([-.4,3.5,-20.,10.]);J=csr_matrix([[1.,2.],[3.,4.],[5.,6.],[7.,8.]])
    mask=np.array([False,False,False,True]);scales=np.array([2.,3.,4.,5.]);ledger=dict(family='original',original_start=0,original_stop=4)
    old=SourceRowBlock(r,J,mask,scales,ledger)
    explicit=SourceRowBlock(r,J,mask,scales,ledger,None)
    a,b=accumulated(old),accumulated(explicit)
    assert a.contract==b.contract and a.cost==b.cost
    assert np.array_equal(a.normal.toarray(),b.normal.toarray()) and np.array_equal(a.gradient,b.gradient)
    original_hash=hashlib.sha256();_array_digest(original_hash,mask);_array_digest(original_hash,scales);_ledger_digest(original_hash,ledger)
    assert _checked_block(old,2,True)[-1].inventory_sha256==original_hash.hexdigest()
    robust=np.abs(r[~mask]);exact=float(np.where(robust<=2.,.5*robust*robust,2.*robust-2.).sum())+.5*float(r[mask]@r[mask])
    assert a.cost==exact==source_cost(r,mask)


def test_weighted_source_cache_rejects_new_policy_and_lagged_J_keeps_weights():
    r=np.array([0.,5.,-10.,3.]);J=csr_matrix([[1.,2.],[3.,4.],[5.,6.],[7.,8.]])
    mask=np.zeros(4,bool);scales=np.ones(4);ledger=dict(family='original',original_start=0,original_stop=4)
    block=SourceRowBlock(r,J,mask,scales,ledger,SourceLossWeights([.25]*4))
    normal=accumulated(block,retain=True)
    model=LaggedJacobianSource();assert model.begin(np.zeros(2),'frozen-original-profile')is False
    model.capture(normal.native_replay_cache);assert model.begin(np.ones(2)*.01,'frozen-original-profile')is True
    supplied=SourceRowBlock(r,None,mask,scales,ledger,block.loss_weights)
    reused=list(model.blocks(lambda:iter([supplied])))[0]
    assert reused.loss_weights is block.loss_weights
    assert accumulated(reused).cost==normal.cost
    changed=SourceRowBlock(r,None,mask,scales,ledger,SourceLossWeights([.5]*4))
    with pytest.raises(ValueError,match='row order/ledger/mask/native scales changed'):
        replay_source_cost(lambda:iter([changed]),normal.contract,frozen_profile_token='frozen-original-profile')
    original=SourceRowBlock(r,None,mask,scales,ledger)
    with pytest.raises(ValueError):
        replay_source_cost(lambda:iter([original]),normal.contract,frozen_profile_token='frozen-original-profile')


def test_loss_payload_is_immutable_including_worker_pickle_transport():
    payload=SourceLossWeights([.25,.75],[2.,3.])
    for value in (payload,pickle.loads(pickle.dumps(payload))):
        with pytest.raises(ValueError):value.integration_weights.setflags(write=True)
        with pytest.raises(ValueError):value.working_normalization_multiplier[0]=99.
        assert value.identity_digest==payload.identity_digest


def setup_policy(tmp_path):
    source_row=dict(id='raw_curve',pts=100,split='fit',uv=[[0.,0.],[10.,0.]],sigma_px=4.)
    parent=tmp_path/'raw_packet.json';parent.write_text(json.dumps(dict(rows=[source_row])),encoding='utf8')
    class Source:
        count=2;updater=SimpleNamespace(original_profile=np.array([.4,.8]))
        contract=dict(schema='existing-complete-original-source',source_packet=str(parent))
        scales=np.full(6,4.);quadratic=np.zeros(6,bool)
        def guard(self,q):return True
        def blocks(self,q,jacobian=True):
            # First two slots are upstream profile-weighted motion, which the
            # policy does not select. No grouping scan or numerical change.
            r=np.array([3.,-12.,4.,-20.,80.,-4.])+np.array([[1.,2.],[2.,1.],[1.,3.],[3.,2.],[4.,2.],[2.,4.]])@q
            J=csr_matrix([[1.,2.],[2.,1.],[1.,3.],[3.,2.],[4.,2.],[2.,4.]]) if jacobian else None
            yield SourceRowBlock(r,J,self.quadratic,self.scales,dict(family='programme:test',original_start=0,original_stop=6))
    source=Source()
    record=dict(id='typed_row',split='fit',source_video_sha256='a'*64,actual_pts=100,physical_curve_id='C0',operator_id='finite-source-curve',
        uv=source_row['uv'],stored_sigma_px=4.,working_sigma_px=4.,prior_sampling_multiplier=1.,
        sampling_kind=['measured','measured'],point_integration_mass=[5.,5.],
        source_row_id='raw_curve',original_uv_float64_sha256=original_uv_sha256(source_row),
        source_row_reference=dict(path=str(parent),sha256=file_sha256(parent),format='json',pointer='/rows/0'),
        physical_identity_basis='source_authenticated',physical_identity_authority=dict(path=str(parent),sha256=file_sha256(parent)),
        block_family='programme:test',block_start=0,block_stop=6,global_scalar_indices=[[2,3],[4,5]])
    policy=dict(schema=SCHEMA,budget_policy='equal_physical_curve',curve_budget=1.,
        source_contract_sha256=source_contract_sha256(source.contract),source_scalar_count=6,
        source_scales_float64_sha256=hashlib.sha256(source.scales.astype('<f8').tobytes()).hexdigest(),
        source_quadratic_bool_sha256=hashlib.sha256(source.quadratic.tobytes()).hexdigest(),records=[record])
    path=tmp_path/'grouping.json';path.write_text(json.dumps(policy),encoding='utf8')
    return source,path,policy,parent


def test_complete_wrapper_keeps_raw_source_and_profiles_and_untargeted_rows(tmp_path):
    source,path,_,_=setup_policy(tmp_path)
    assert wrap_curve_source_grouping(source,None) is source
    grouped=wrap_curve_source_grouping(source,path)
    assert grouped.updater is source.updater and grouped.scales is source.scales
    old=list(source.blocks(np.zeros(2)))[0];new=list(grouped.blocks(np.zeros(2)))[0]
    assert np.array_equal(new.r,old.r) and np.array_equal(new.J.data,old.J.data)
    assert new.scales is old.scales
    assert np.array_equal(new.loss_weights.integration_weights,[1.,1.,.5,.5,.5,.5])
    assert grouped.guard(np.zeros(2)) is True
    result=accumulated(new)
    assert result.cost==source_cost(old.r,old.quadratic,new.loss_weights)
    assert grouped.contract['source_curve_integration']['group_policy_precomputed_once'] is True


@pytest.mark.parametrize('mutation',['source_packet','UV','source_scales','source_contract','wrong_slot','HELD','lambda'])
def test_grouping_policy_and_source_changes_fail_closed(tmp_path,mutation):
    source,path,policy,parent=setup_policy(tmp_path)
    if mutation=='source_packet':parent.write_text(parent.read_text()+' ',encoding='utf8')
    elif mutation=='UV':policy['records'][0]['uv'][0][0]+=1.
    elif mutation=='source_scales':source.scales=source.scales.copy();source.scales[2]=99.
    elif mutation=='source_contract':source.contract=dict(source.contract,changed=True)
    elif mutation=='wrong_slot':policy['records'][0]['global_scalar_indices']=[[5,6],[2,3]]
    elif mutation=='HELD':policy['records'][0]['split']='holdout'
    else:policy['curve_budget']=2.
    path.write_text(json.dumps(policy),encoding='utf8')
    with pytest.raises(ValueError):wrap_curve_source_grouping(source,path)
