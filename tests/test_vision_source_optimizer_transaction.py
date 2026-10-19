import pytest

torch=pytest.importorskip('torch')
from rinkspace.vision_source_optimizer_transaction import source_optimizer_proposal,guarded_source_optimizer_step


def test_rejected_actual_source_update_restores_parameters_and_moments():
    parameter=torch.nn.Parameter(torch.tensor([1.]))
    optimizer=torch.optim.AdamW([parameter],lr=.1)
    parameter.square().sum().backward()
    proposal=source_optimizer_proposal(optimizer)
    assert parameter.item()==1. and not optimizer.state
    decision=guarded_source_optimizer_step(optimizer,proposal,
        source_objective=lambda:dict(loss=float(parameter.detach().square()),support_count=1,source_keys=['native110']),
        native_retention=lambda:dict(retained=False),minimum_scale=.0625)
    assert decision['accepted'] is False
    assert parameter.item()==1. and not optimizer.state


def test_backtracked_update_preserves_other_known_source_and_commits_actual_moments():
    parameter=torch.nn.Parameter(torch.tensor([1.]))
    optimizer=torch.optim.AdamW([parameter],lr=.1)
    parameter.square().sum().backward()
    proposal=source_optimizer_proposal(optimizer)
    decision=guarded_source_optimizer_step(optimizer,proposal,
        source_objective=lambda:dict(loss=float(parameter.detach().square()),support_count=1,source_keys=['native110']),
        native_retention=lambda:dict(retained=parameter.item()>.92),minimum_scale=.0625)
    assert decision['accepted'] is True and decision['scale']==.5
    assert float(optimizer.state[parameter]['step'])==1
    assert torch.equal(optimizer.state[parameter]['exp_avg'],proposal['next_optimizer_state'][0]['exp_avg'])


def test_changed_exact_source_support_rejects_and_restores_update():
    parameter=torch.nn.Parameter(torch.tensor([1.]))
    optimizer=torch.optim.AdamW([parameter],lr=.1)
    parameter.square().sum().backward()
    proposal=source_optimizer_proposal(optimizer)
    with pytest.raises(ValueError,match='changed known source support'):
        guarded_source_optimizer_step(optimizer,proposal,
            source_objective=lambda:dict(loss=float(parameter.detach().square()),support_count=1,
                source_keys=['native110' if parameter.item()==1. else 'native111']),
            native_retention=lambda:dict(retained=True),minimum_scale=.0625)
    assert parameter.item()==1. and not optimizer.state


def test_nonfinite_existing_moments_and_invalid_scale_fail_without_mutation():
    parameter=torch.nn.Parameter(torch.tensor([1.]))
    optimizer=torch.optim.AdamW([parameter],lr=.1)
    parameter.square().sum().backward();optimizer.step()
    optimizer.state[parameter]['exp_avg'].fill_(float('nan'))
    with pytest.raises(ValueError,match='finite'):
        source_optimizer_proposal(optimizer)
    optimizer.state[parameter]['exp_avg'].zero_()
    proposal=source_optimizer_proposal(optimizer)
    with pytest.raises(ValueError,match='fraction'):
        guarded_source_optimizer_step(optimizer,proposal,source_objective=lambda:None,
            native_retention=lambda:None,minimum_scale=0.)


def test_multi_parameter_copy_failure_restores_existing_optimizer_moments(monkeypatch):
    from rinkspace import vision_source_optimizer_transaction as transaction
    a=torch.nn.Parameter(torch.tensor([1.]));b=torch.nn.Parameter(torch.tensor([2.]))
    optimizer=torch.optim.AdamW([a,b],lr=.1)
    (a.square()+b.square()).sum().backward();optimizer.step();optimizer.zero_grad()
    (a.square()+b.square()).sum().backward()
    proposal=source_optimizer_proposal(optimizer)
    original_deepcopy=transaction.deepcopy
    def fail_second_state(value):
        if value is proposal['next_optimizer_state'][1]:raise RuntimeError('simulated moment-copy failure')
        return original_deepcopy(value)
    monkeypatch.setattr(transaction,'deepcopy',fail_second_state)
    with pytest.raises(RuntimeError,match='moment-copy'):
        guarded_source_optimizer_step(optimizer,proposal,
            source_objective=lambda:dict(loss=float((a.detach().square()+b.detach().square()).sum()),
                support_count=2,source_keys=['native110']),native_retention=lambda:dict(retained=True))
    for parameter,before,state in zip(proposal['parameters'],proposal['before'],proposal['previous_optimizer_state']):
        assert torch.equal(parameter,before)
        assert torch.equal(optimizer.state[parameter]['step'],state['step'])
        assert torch.equal(optimizer.state[parameter]['exp_avg'],state['exp_avg'])
        assert torch.equal(optimizer.state[parameter]['exp_avg_sq'],state['exp_avg_sq'])


def test_lower_known_role_loss_cannot_lose_a_previously_correct_feedback_point():
    parameter=torch.nn.Parameter(torch.tensor([1.]))
    optimizer=torch.optim.AdamW([parameter],lr=.1)
    parameter.square().sum().backward();proposal=source_optimizer_proposal(optimizer)
    decision=guarded_source_optimizer_step(optimizer,proposal,
        source_objective=lambda:dict(loss=float(parameter.detach().square()),support_count=2,
            source_keys=['native110'],known_semantic_correct_original_indices={'110':[0] if parameter.item()==1. else [1]}),
        native_retention=lambda:dict(retained=True),minimum_scale=.0625)
    assert decision['accepted'] is False and parameter.item()==1. and not optimizer.state
