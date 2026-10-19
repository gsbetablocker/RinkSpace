"""Guard actual head deltas without inventing targets or consuming a rejection."""
from copy import deepcopy
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from rinkspace.vision_head_guard import (build_head_support_cache,evaluate_head_retention,
    adamw_head_proposal,guarded_head_backtrack,commit_guarded_head_proposal,
    native_cuda_original_supervision_loss)
from rinkspace.vision_retention import source_retention_snapshot

CLASSES=['background','board_top','ice_contact']


def fixture(*,fractional=False,retired=False):
    head=torch.nn.Conv2d(2,3,1)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[-1.,0.],[1.,0.],[0.,1.]])[:,:,None,None]);head.bias.zero_()
    feature=torch.zeros(1,2,8,8);feature[:,0]=1.;feature[0,0,0,0]=-1.
    uv=[[3.25,2.5],[4.,4.]] if fractional else [[3.,3.],[4.,4.]]
    original=dict(group_id='human',kind='board_top',state='user_confirmed_visual_supervision',
        primitive='free_curve',uv=uv,interval_px=[-1.5,1.5],authority='direct_user',
        source_active_supervision_mask=[not retired,True],source_visible_mask=[True,True])
    sample=dict(source_hash='a'*64,source_pts=256,time_base='1/15360',frame_size_wh=[8,8],annotations=[original])
    indices=np.array([1] if retired else [0,1]);selected={**original,'uv':np.array(uv)[indices],'source_point_indices':indices}
    binding=dict(source_sha256='a'*64,source_pts=256,time_base='1/15360',shot_id='native_shot',native_hw=[8,8])
    with torch.no_grad():probabilities=head(feature).softmax(1)
    primitive=torch.zeros(1,3,8,8);primitive[:,2]=1.
    positive=dict(region_id='known_background',kind='background')
    rejected=dict(region_id='known_not_contact',excluded_kinds=['ice_contact'])
    bgmask=torch.zeros(1,1,8,8,dtype=torch.bool);bgmask[:,:,0,0]=True
    rejectmask=torch.zeros_like(bgmask);rejectmask[:,:,7,7]=True
    regions=[(positive,bgmask),(rejected,rejectmask)]
    snapshot=source_retention_snapshot(probabilities,primitive,source_binding=binding,annotations=[selected],
        classes=CLASSES,reviewed_regions=regions)
    cache=build_head_support_cache(feature,probabilities,sample=sample,source_binding=binding,classes=CLASSES,
        reference_weight=head.weight,reference_bias=head.bias,reference_snapshot=snapshot,
        reviewed_regions=regions,reference_precision='fp32')
    return head,feature,sample,snapshot,cache


def test_native_bilinear_corner_cache_preserves_retired_gap_and_complete_inventory():
    head,_,_,_,cache=fixture(fractional=True,retired=True)
    assert cache['CPU_proxy_baseline_decisions_verified']
    assert cache['groups'][0]['source_point_indices'].tolist()==[1]
    assert cache['groups'][0]['corner_features'].shape==(1,4,2)
    result=evaluate_head_retention([cache],head.weight,head.bias,expected_source_keys=[cache['source_key']])
    assert result['retained']
    with pytest.raises(ValueError,match='incomplete'):
        evaluate_head_retention([cache],head.weight,head.bias,expected_source_keys=[cache['source_key'],['other','1/15360',256]])


def test_candidate_cannot_sacrifice_an_old_correct_manual_point_for_lower_loss():
    head,_,_,_,cache=fixture()
    proposal=dict(before_weight=head.weight.detach().clone(),before_bias=head.bias.detach().clone(),
        proposed_weight=head.weight.detach().clone(),proposed_bias=torch.tensor([0.,-5.,0.]),
        source_head_sha256=cache['reference_head_sha256'])
    def objective(weight,bias):return dict(loss=float(10+bias[1]),sample_count=2,source_key=cache['source_key'])
    decision=guarded_head_backtrack([cache],proposal,expected_source_keys=[cache['source_key']],
        localization_evaluator=objective,scales=(1.,),precision='fp32')
    assert not decision['accepted'] and decision['trials'][0]['localization_improved']
    assert decision['trials'][0]['retention']['findings'][0]['newly_wrong_original_indices']==[0,1]


def test_source_reviewed_reject_only_is_kept_without_assigning_unknown_background():
    head,_,_,_,cache=fixture()
    bias=head.bias.detach().clone();bias[2]=5.
    result=evaluate_head_retention([cache],head.weight,bias,expected_source_keys=[cache['source_key']])
    region=next(r for r in result['findings'] if r.get('region_id')=='known_not_contact')
    assert region['newly_wrong_native_flat_indices']==[63] and region['unknown_pixels_not_included']
    assert cache['unknown_pixels_assigned_labels'] is False


def test_proxy_decision_mismatch_fails_closed_and_cannot_reset_original_baseline():
    head,_,_,_,cache=fixture();cache['CPU_proxy_baseline_decisions_verified']=False
    with pytest.raises(ValueError,match='did not reproduce'):
        evaluate_head_retention([cache],head.weight,head.bias,expected_source_keys=[cache['source_key']])


def test_backtrack_keeps_known_support_and_requires_a_real_supported_localization_decrease():
    head,_,_,_,cache=fixture()
    proposal=dict(before_weight=head.weight.detach().clone(),before_bias=head.bias.detach().clone(),
        proposed_weight=head.weight.detach().clone(),proposed_bias=torch.tensor([0.,-5.,0.]),
        source_head_sha256=cache['reference_head_sha256'])
    objective=lambda weight,bias:dict(loss=float(10+bias[1]),sample_count=2,source_key=cache['source_key'])
    decision=guarded_head_backtrack([cache],proposal,expected_source_keys=[cache['source_key']],
        localization_evaluator=objective,scales=(1.,.125),precision='fp32')
    assert decision['accepted'] and decision['scale']==.125
    with pytest.raises(ValueError,match='actual supported source'):
        guarded_head_backtrack([cache],proposal,expected_source_keys=[cache['source_key']],
            localization_evaluator=lambda w,b:dict(loss=0.,sample_count=0),scales=(1.,))
    changing=lambda w,b:dict(loss=float(10+b[1]),sample_count=1 if float(b[1]) else 2,source_key=cache['source_key'])
    with pytest.raises(ValueError,match='silently changed'):
        guarded_head_backtrack([cache],proposal,expected_source_keys=[cache['source_key']],localization_evaluator=changing,scales=(.125,))


def test_adam_proposal_preserves_real_parameters_moments_and_matches_an_actual_step():
    head,_,_,_,cache=fixture();frozen=torch.nn.Parameter(torch.tensor([4.]),requires_grad=False)
    optimizer=torch.optim.AdamW([head.weight,head.bias,frozen],lr=1e-3,weight_decay=.02)
    for p in (head.weight,head.bias):p.grad=torch.full_like(p,.25)
    optimizer.step();before=deepcopy(optimizer.state_dict());weight=head.weight.detach().clone();bias=head.bias.detach().clone()
    for p in (head.weight,head.bias):p.grad=torch.full_like(p,-.5)
    proposed=adamw_head_proposal(head,optimizer)
    assert torch.equal(head.weight,weight) and torch.equal(head.bias,bias)
    assert all(torch.equal(before['state'][key]['exp_avg'],optimizer.state_dict()['state'][key]['exp_avg']) for key in before['state'])
    optimizer.step()
    assert torch.equal(proposed['proposed_weight'],head.weight) and torch.equal(proposed['proposed_bias'],head.bias)
    assert torch.equal(frozen,torch.tensor([4.]))


def test_unfrozen_shared_gradients_or_stale_optimizer_state_cannot_be_committed():
    head,_,_,_,cache=fixture();shared=torch.nn.Parameter(torch.tensor([4.]))
    optimizer=torch.optim.AdamW([head.weight,head.bias,shared],lr=1e-3)
    for p in (head.weight,head.bias):p.grad=torch.full_like(p,.25)
    shared.grad=torch.ones_like(shared)
    with pytest.raises(ValueError,match='must remain frozen'):adamw_head_proposal(head,optimizer)
    shared.grad=None;proposal=adamw_head_proposal(head,optimizer)
    objective=lambda w,b:dict(loss=float(w.square().sum()),sample_count=2,source_key=cache['source_key'])
    decision=guarded_head_backtrack([cache],proposal,expected_source_keys=[cache['source_key']],
        localization_evaluator=objective,scales=(1.,),precision='fp32')
    assert decision['accepted']
    optimizer.param_groups[0]['lr']=.1
    with pytest.raises(ValueError,match='moments/settings changed'):commit_guarded_head_proposal(head,optimizer,proposal,decision)


def test_rejected_proposal_never_advances_optimizer_or_creates_success():
    head,_,_,_,cache=fixture();optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3)
    for p in head.parameters():p.grad=torch.ones_like(p)
    proposal=adamw_head_proposal(head,optimizer)
    with pytest.raises(ValueError,match='Rejected parameter'):
        commit_guarded_head_proposal(head,optimizer,proposal,dict(accepted=False,scale=0.))
    assert not optimizer.state


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Original native CUDA arithmetic required')
def test_native_cuda_guard_uses_full_source_arithmetic_and_keeps_reject_only_regions():
    head,feature,sample,snapshot,_=fixture(fractional=True)
    head=head.cuda();feature=feature.cuda()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        probabilities=head(feature).float().softmax(1)
    primitive=torch.zeros(1,3,8,8,device='cuda');primitive[:,2]=1.
    positive=dict(region_id='known_background',kind='background')
    rejected=dict(region_id='known_not_contact',excluded_kinds=['ice_contact'])
    bgmask=torch.zeros(1,1,8,8,dtype=torch.bool,device='cuda');bgmask[:,:,0,0]=True
    rejectmask=torch.zeros_like(bgmask);rejectmask[:,:,7,7]=True
    regions=[(positive,bgmask),(rejected,rejectmask)]
    annotation={**sample['annotations'][0],'source_point_indices':np.array([0,1])}
    snapshot=source_retention_snapshot(probabilities,primitive.softmax(1),
        source_binding=snapshot['source_binding'],annotations=[annotation],classes=CLASSES,reviewed_regions=regions)
    cache=build_head_support_cache(feature,probabilities,sample=sample,
        source_binding=snapshot['source_binding'],classes=CLASSES,reference_weight=head.weight,
        reference_bias=head.bias,reference_snapshot=snapshot,reviewed_regions=regions,
        retain_native_cuda_features=True)
    assert cache['native_GPU_baseline_decisions_verified']
    # A CPU sparse proxy can disagree; the verified native arithmetic remains
    # authoritative, and unknown regions still acquire no replacement label.
    cache['CPU_proxy_baseline_decisions_verified']=False
    assert evaluate_head_retention([cache],head.weight,head.bias,
        expected_source_keys=[cache['source_key']])['retained']
    bias=head.bias.detach().clone();bias[2]=5.
    result=evaluate_head_retention([cache],head.weight,bias,expected_source_keys=[cache['source_key']])
    assert not result['retained']
    assert any(row.get('region_id')=='known_not_contact' for row in result['findings'])
    cache['whole_shot_role']='development'
    with pytest.raises(ValueError,match='training-shot'):
        native_cuda_original_supervision_loss([cache],head.weight,head.bias)
    cache['whole_shot_role']='train'
    loss=native_cuda_original_supervision_loss([cache],head.weight,head.bias)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(head.weight.grad).all()
    assert head.weight.grad.count_nonzero()>0
    assert cache['regions'][1]['kind'] is None
    assert cache['regions'][1]['excluded_kinds']==['ice_contact']
