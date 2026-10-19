import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.solver_runtime.source_loss import (
    SourceLossWeights, source_loss_checkpoint_arrays, weighted_source_cost,
)

MODULE=Path(__file__).parents[1]/'calibration/solver_runtime/legacy/structured_programme_source_candidate/existing_main7_trial_row_cost.py'
spec=importlib.util.spec_from_file_location('existing_main7_row_cost_under_test',MODULE)
diagnostic=importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def _fixture(motion,absolute,priors,start=0):
    n=motion//2;count=motion+absolute+priors
    row=dict(id='retained_RGB_curve',pts=123,split='fit',uv=np.zeros((absolute//2,2)),
        sigma=np.full(absolute//2,4.))
    mask=np.r_[np.zeros(motion+absolute,bool),np.ones(priors,bool)]
    sigma=np.r_[np.full(motion+absolute,4.),np.ones(priors)]
    bundle=SimpleNamespace(motion_rows=motion,source_residual_rows=motion+absolute,
        a=np.zeros((n,2)),b=np.ones((n,2)),ia=np.zeros(n,int),ib=np.zeros(n,int),
        motion_sigma=np.full(motion,2.),stamps=np.arange(priors))
    owner=SimpleNamespace(name='main7',bundle=bundle,groups=[(123,[row])],quadratic=mask)
    prior=SimpleNamespace(name='main1',quadratic=np.zeros(start,bool))
    state=SimpleNamespace(blocks=[prior,owner] if start else [owner])
    block=SimpleNamespace(r=np.zeros(count),quadratic=mask,scales=sigma,loss_weights=None,
        row_ledger=dict(family='programme:main7',original_start=start,original_stop=start+count))
    return state,block


def _baseline(tmp_path,block,r,loss=None):
    directory=tmp_path/'initial_normal_source_blocks';directory.mkdir()
    (directory/'completed_prefix.json').write_text(json.dumps(dict(
        frozen_profile_token='profiles',q_sha256='q',source_manifest_sha256='source',
        completed_blocks=[dict(family='programme:main7')])),encoding='utf8')
    np.savez(directory/'000.npz',r=r,quadratic=block.quadratic,scales=block.scales,
        ledger_json=np.asarray(json.dumps(block.row_ledger)),frozen_profile_token='profiles',
        q_sha256='q',**source_loss_checkpoint_arrays(loss,len(r)))


def test_current_actual_main7_size_after_nearboard_compaction_has_no_old_partition_gate():
    # Actual production main7 counts: the reviewed 49 nearboard rows remove
    # 2910 absolute scalars, while motion and physical RF roll priors stay.
    state,block=_fixture(801078,50358-2910,1253,start=97884)
    _,motion,absolute,priors=diagnostic._main7_partition(block,state)
    assert (motion,absolute,priors,len(block.r))==(801078,47448,1253,849779)
    block.row_ledger['original_start']+=2
    with pytest.raises(ValueError,match='not applicable'):
        diagnostic._main7_partition(block,state)


def test_report_uses_current_shared_huber_integration_and_working_normalization(tmp_path):
    state,block=_fixture(4,4,2,start=3)
    initial=np.array([.3,-4.,1.,-2.,.5,8.,-9.,2.,.5,-.3])
    block.r=np.array([4.,-.2,-3.,2.,1.,-4.,.8,-8.,3.,-.4])
    weights=np.array([1.,.2,.5,2.,.1,.9,.3,.7,1.,1.])
    normalization=np.array([1.,2.,.5,1.,3.,.5,2.,1.,1.,1.])
    block.loss_weights=SourceLossWeights(weights,normalization)
    _baseline(tmp_path,block,initial,block.loss_weights)
    before_r=block.r.copy();before_uv=state.blocks[-1].groups[0][1][0]['uv'].copy()
    path=diagnostic.write_existing_main7_trial_row_cost(tmp_path,np.zeros(2),'profiles',block,state,1)
    report=json.loads(path.read_text(encoding='utf8'))
    assert report['current_native_layout']['complete_scalar_rows']==10
    assert report['actual_before_cost']==pytest.approx(weighted_source_cost(initial,block.quadratic,block.loss_weights))
    assert report['actual_after_cost']==pytest.approx(weighted_source_cost(block.r,block.quadratic,block.loss_weights))
    assert sum(p['actual_after_cost'] for p in report['actual_source_partition'])==pytest.approx(report['actual_after_cost'])
    expected=weighted_source_cost(block.r[4:8],block.quadratic[4:8],SourceLossWeights(weights[4:8],normalization[4:8]))
    assert report['largest_actual_absolute_source_cost_increases'][0]['actual_after_cost']==pytest.approx(expected)
    assert report['extra_source_calls']==0 and np.array_equal(block.r,before_r)
    assert np.array_equal(state.blocks[-1].groups[0][1][0]['uv'],before_uv)
    assert not np.isclose(report['actual_after_cost'],weighted_source_cost(block.r,block.quadratic,None))


def test_saved_baseline_loss_or_layout_mismatch_is_not_comparable(tmp_path):
    state,block=_fixture(4,4,2)
    _baseline(tmp_path,block,np.zeros(10),SourceLossWeights(np.ones(10)))
    with pytest.raises(ValueError,match='loss inventory'):
        diagnostic.write_existing_main7_trial_row_cost(tmp_path,np.zeros(2),'profiles',block,state,1)


def test_ambiguous_native_or_roll_partition_is_explicitly_not_applicable():
    state,block=_fixture(4,4,2)
    state.blocks[0].bundle.source_residual_rows+=2
    with pytest.raises(ValueError,match='not applicable'):
        diagnostic._main7_partition(block,state)


def test_live_main1_baseline_requires_no_full_matrix_cache_and_is_not_aliased(tmp_path):
    state,block=_fixture(4,4,2)
    state.blocks[0].name='main1';block.row_ledger['family']='programme:main1'
    block.r[:]=np.arange(10)/3.
    initial=block.r.copy()
    baseline=diagnostic.capture_existing_programme_baseline(np.zeros(2),'profiles',block,state,'source')
    block.r[:4]+=2.
    path=diagnostic.write_existing_main7_trial_row_cost(tmp_path,np.ones(2),'profiles',block,state,1,baseline=baseline)
    report=json.loads(path.read_text(encoding='utf8'))
    assert not (tmp_path/'initial_normal_source_blocks').exists()
    assert report['observed_programme_family']=='programme:main1'
    assert report['residual_baseline_only_no_J_retained']
    assert np.array_equal(baseline['r'],initial)
    parts={x['family']:x for x in report['actual_source_partition']}
    assert parts['motion']['actual_cost_delta']>0.
    assert parts['absolute_original_FIT']['actual_cost_delta']==0.
    assert parts['physical_roll_prior']['actual_cost_delta']==0.
    with pytest.raises(ValueError,match='profile differs'):
        diagnostic.write_existing_main7_trial_row_cost(tmp_path,np.ones(2),'changed',block,state,2,baseline=baseline)
