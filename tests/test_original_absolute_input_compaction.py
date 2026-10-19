from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.solver_runtime.original_absolute_input_compaction import (slice_observed_points, SourceWithAbsoluteInputCompaction,
    original_points_to_active,compose_selected_scalar_positions,reindex_retirement_block_binding)
from calibration.solver_runtime.source_loss import SourceLossWeights, source_loss_checkpoint_arrays, source_loss_from_checkpoint


def test_observed_input_slice_preserves_full_finite_primitive_and_source_arrays():
    row=dict(uv=[[0.,0.],[1.,1.],[2.,2.]],sigma=[7.,7.,7.],normal=[[0.,1.]]*3,
        normal_interval=[[-2.,2.]]*3,world=[[0.,0.,0.],[20.,0.,0.],[20.,20.,0.]],
        point_correspondence=False,id='finite_RGB_curve',split='fit')
    saved=deepcopy(row);active=slice_observed_points(row,[0,2])
    assert active['uv']==[row['uv'][0],row['uv'][2]] and active['sigma']==[7.,7.]
    assert active['world'] is row['world'] and len(active['world'])==3
    assert row==saved and active['normal_interval']==[[-2.,2.],[-2.,2.]]
    with pytest.raises(ValueError,match='Ordered'):slice_observed_points(row,[2,0])
    with pytest.raises(ValueError,match='Point correspondence'):slice_observed_points(dict(row,point_correspondence=True),[0,2])
    assert original_points_to_active(active,[2]).tolist()==[1] # deleted point1 moves a later correction
    with pytest.raises(ValueError,match='physically removed'):original_points_to_active(active,[1])
    active['uv'][0][0]+=1.
    with pytest.raises(ValueError,match='bytes changed'):original_points_to_active(active,[2])


def test_only_declared_retirement_slots_and_low_range_are_reindexed():
    previous={1478186:9,1478187:10,777:-1,888:20}
    keep=np.array([0,1,2,3,4,5,6,7,8,9,10,15,20])
    mapped=compose_selected_scalar_positions(previous,0,0,keep,21)
    assert mapped=={1478186:9,1478187:10,777:-1,888:12}
    assert len(mapped)==len(previous)
    # The shape/rows are true current ranges; all old clock/source/map facts
    # remain exactly the same objects, not rewritten into current positions.
    before=dict(family='programme:low',original_start=50,original_stop=150,motion_point_count=40)
    history={'sourceUV':[[1.,2.]],'original_global_slots':[99,100]}
    prequalified=dict(compact_block_binding=before,qualified_records=history)
    state=SimpleNamespace(original_motion_compaction_retirement_prequalification=prequalified)
    report=[dict(block='low',original_start=50,original_rows=100,compact_start=40,compact_rows=98,motion_point_count=40)]
    composition=reindex_retirement_block_binding(state,report)
    assert composition['before']==before and composition['after']==dict(family='programme:low',original_start=40,original_stop=138,motion_point_count=40)
    assert prequalified['qualified_records']is history and history['original_global_slots']==[99,100]
    with pytest.raises(ValueError,match='before absolute'):reindex_retirement_block_binding(state,report)


def test_live_source_integration_actual_huber_normal_and_checkpoint_duplicate_invariance(tmp_path):
    from calibration.solver_runtime import install
    install()
    from z190_streaming_source_normal import SourceRowBlock,accumulate_source_normal,_checked_block
    # Include residuals outside Huber2; weights cannot scale the robust cutoff.
    r=np.array([.3,-.4,4.,-5.,12.,-13.]);J=csr_matrix(np.arange(18.).reshape(6,3)+1)
    mask=np.zeros(6,bool);sigma=np.full(6,4*np.sqrt(3));a=np.repeat([.2,.3,.5],2)*3
    weights=SourceLossWeights(a)
    raw=SourceRowBlock(r,J,mask,sigma,dict(family='programme:main7',original_start=0,original_stop=6))
    base=SimpleNamespace(contract={'schema':'original-complete-source'},blocks=lambda q,jacobian:iter([raw]))
    state=SimpleNamespace(original_absolute_input_compaction_loss_weights={'programme:main7':weights},
        original_absolute_input_compaction_contract={'schema':'source-certified-current-curves'})
    source=SourceWithAbsoluteInputCompaction(base,state);block=next(source.blocks(np.zeros(3),True))
    result=accumulate_source_normal(lambda:iter([block]),3,expected_row_count=6,frozen_profile_token='original-profile')
    robust=np.minimum(1.,2/abs(r));expected_g=J.T@(a*robust*r)
    expected_H=J.T@J.multiply((a*robust)[:,None])
    assert np.allclose(result.gradient,expected_g) and np.allclose(result.normal.toarray(),expected_H.toarray())
    losses=np.where(abs(r)<=2,.5*r*r,2*abs(r)-2)
    assert result.cost==pytest.approx(a@losses)
    assert np.sum(a[::2]/sigma[::2]**2)==pytest.approx(1/16)
    duplicated=SourceRowBlock(np.repeat(r,2),csr_matrix(np.repeat(J.toarray(),2,axis=0)),np.repeat(mask,2),
        np.repeat(sigma,2),dict(family='real-exact-duplicate-replay'),SourceLossWeights(np.repeat(a,2)/2))
    duplicate=accumulate_source_normal(lambda:iter([duplicated]),3,expected_row_count=12,frozen_profile_token='original-profile')
    assert duplicate.cost==pytest.approx(result.cost)
    assert np.allclose(duplicate.gradient,result.gradient) and np.allclose(duplicate.normal.toarray(),result.normal.toarray())
    path=tmp_path/'actual_source_block.npz';np.savez(path,**source_loss_checkpoint_arrays(weights,len(r)))
    with np.load(path,allow_pickle=False)as z:restored=source_loss_from_checkpoint(z)
    assert restored.identity_digest==weights.identity_digest and not restored.integration_weights.flags.writeable
    unweighted=_checked_block(raw,3,True)[-1];weighted=_checked_block(block,3,True)[-1]
    assert unweighted.inventory_sha256!=weighted.inventory_sha256
    assert source_loss_checkpoint_arrays(None,6)=={} and source_loss_from_checkpoint({})is None
    assert raw.loss_weights is None and block.scales is raw.scales and block.J is raw.J


def test_physical_sigma_separates_historical_density_in_huber_tail_and_live_normal(tmp_path):
    from calibration.solver_runtime import install
    install()
    from z190_streaming_source_normal import SourceRowBlock,accumulate_source_normal,replay_source_cost
    # One physical source, including errors well beyond its true 8px Huber
    # cutoff. Historical N changes the stored normalization, never its loss.
    error=np.array([3.,-4.,12.,-20.,30.,-40.]);physical_J=np.arange(18.).reshape(6,3)+1.
    mass=np.repeat([.2,.3,.5],2);quadratic=np.zeros(6,bool);results=[]
    for n in (3,48):
        prior=np.full(6,np.sqrt(n));stored_sigma=np.full(6,4*np.sqrt(n))
        r=error/stored_sigma;J=csr_matrix(physical_J/stored_sigma[:,None])
        weights=SourceLossWeights(mass,prior)
        raw=SourceRowBlock(r,J,quadratic,stored_sigma,dict(family='programme:main7',original_start=0,original_stop=6))
        source=SourceWithAbsoluteInputCompaction(SimpleNamespace(contract={'schema':'same-physical-source'},blocks=lambda q,jacobian:iter([raw])),
            SimpleNamespace(original_absolute_input_compaction_loss_weights={'programme:main7':weights},original_absolute_input_compaction_contract={'schema':'physical-sigma4-source'}))
        block=next(source.blocks(None,True));result=accumulate_source_normal(lambda:iter([block]),3,expected_row_count=6,frozen_profile_token='physical-source')
        work=error/4.;workJ=physical_J/4.;robust=np.minimum(1.,2/abs(work))
        assert result.cost==pytest.approx(mass@np.where(abs(work)<=2,.5*work**2,2*abs(work)-2))
        assert np.allclose(result.gradient,workJ.T@(mass*robust*work))
        assert np.allclose(result.normal.toarray(),workJ.T@((mass*robust)[:,None]*workJ))
        assert block.r is raw.r and block.J is raw.J and block.scales is raw.scales
        results.append(result)
        old=SourceRowBlock(r,J,quadratic,stored_sigma,raw.row_ledger,SourceLossWeights(mass*prior**2))
        with pytest.raises(ValueError,match='row order/ledger/mask/native scales changed'):
            replay_source_cost(lambda:iter([old]),result.contract,frozen_profile_token='physical-source')
        path=tmp_path/f'physical_sigma{n}.npz';np.savez(path,**source_loss_checkpoint_arrays(weights,6))
        with np.load(path,allow_pickle=False)as z:restored=source_loss_from_checkpoint(z)
        assert restored.identity_digest==weights.identity_digest
    assert results[0].cost==pytest.approx(results[1].cost)
    assert np.allclose(results[0].gradient,results[1].gradient)
    assert np.allclose(results[0].normal.toarray(),results[1].normal.toarray())
