"""Finite network proposals checked on unchanged actual source supervision.

The existing optimizer supplies the proposed gradients and moments. Rejected
proposals change neither trained parameters nor optimizer state. Source loss
and native retention are callbacks, so this module creates no labels or masks.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch


def _finite_state(value):
    if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(_finite_state(v) for v in value.values())
    if isinstance(value,(tuple,list)):return all(_finite_state(v) for v in value)
    if isinstance(value,float):return math.isfinite(value)
    return True


def source_optimizer_proposal(optimizer):
    if not isinstance(optimizer,torch.optim.AdamW):
        raise ValueError('Source correction preserves the existing AdamW update')
    originals,clones,groups=[],{},[]
    for group in optimizer.param_groups:
        members=[]
        for parameter in group['params']:
            if parameter.grad is None:continue
            if (not parameter.requires_grad or not torch.isfinite(parameter.grad).all()
                    or not torch.isfinite(parameter).all()
                    or not _finite_state(optimizer.state.get(parameter))):
                raise ValueError('A source proposal needs finite actual trainable gradients')
            clone=torch.nn.Parameter(parameter.detach().clone())
            clone.grad=parameter.grad.detach().clone()
            clones[parameter]=clone;originals.append(parameter);members.append(clone)
        if members:groups.append(dict(deepcopy({k:v for k,v in group.items() if k!='params'}),params=members))
    if not originals:raise ValueError('No current source gradient exists')
    temporary=torch.optim.AdamW(groups)
    for parameter,clone in clones.items():
        if parameter in optimizer.state:temporary.state[clone]=deepcopy(optimizer.state[parameter])
    temporary.step()
    if any(not torch.isfinite(clones[p]).all() or not _finite_state(temporary.state[clones[p]]) for p in originals):
        raise FloatingPointError('The actual source optimizer proposal is not finite')
    return dict(parameters=originals,before=[p.detach().clone() for p in originals],
        proposed=[clones[p].detach().clone() for p in originals],
        previous_optimizer_state=[deepcopy(optimizer.state.get(p)) for p in originals],
        next_optimizer_state=[deepcopy(temporary.state[clones[p]]) for p in originals],
        previous_group_settings=[deepcopy({k:v for k,v in g.items() if k!='params'}) for g in optimizer.param_groups])


def _same_state(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(_same_state(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(_same_state(x,y) for x,y in zip(a,b))
    return a==b


def guarded_source_optimizer_step(optimizer,proposal,*,source_objective,native_retention,
        minimum_scale=None):
    """Backtrack to the representable update, preserving every old source label.

    ``source_objective`` must return loss, support_count and exact source_keys.
    Native retention is evaluated only for an actual source improvement. A
    false return restores every original parameter and leaves moments intact.
    """
    parameters,before,proposed=proposal['parameters'],proposal['before'],proposal['proposed']
    if (minimum_scale is not None and (isinstance(minimum_scale,bool)
            or not math.isfinite(minimum_scale) or not 0<minimum_scale<=1)):
        raise ValueError('Minimum source-proposal scale must be a finite fraction in (0,1]')
    if not _finite_state([before,proposed,proposal['previous_optimizer_state'],proposal['next_optimizer_state']]):
        raise FloatingPointError('A source transaction cannot evaluate nonfinite parameters or moments')
    if any(not torch.equal(p,b) for p,b in zip(parameters,before)):
        raise ValueError('Network changed after its source proposal')
    if (not _same_state(proposal['previous_group_settings'],
            [{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups])
            or any(not _same_state(optimizer.state.get(p),s) for p,s in
                zip(parameters,proposal['previous_optimizer_state']))):
        raise ValueError('Optimizer changed after its source proposal')
    prior=source_objective()
    if prior['support_count']<1 or not math.isfinite(prior['loss']):
        raise ValueError('A source transaction needs finite original known support')
    scale=1.;trials=[];committed=False
    try:
        while minimum_scale is None or scale>=minimum_scale:
            candidates=[a+scale*(b-a) for a,b in zip(before,proposed)]
            if all(torch.equal(a,b) for a,b in zip(before,candidates)):break
            with torch.no_grad():
                for parameter,value in zip(parameters,candidates):parameter.copy_(value)
            after=source_objective()
            if after['support_count']!=prior['support_count'] or after['source_keys']!=prior['source_keys']:
                raise ValueError('A source transaction changed known source support')
            old_correct=prior.get('known_semantic_correct_original_indices')
            new_correct=after.get('known_semantic_correct_original_indices')
            roles_retained=True
            if old_correct is not None or new_correct is not None:
                if old_correct is None or new_correct is None or old_correct.keys()!=new_correct.keys():
                    raise ValueError('Known semantic feedback source identities changed')
                roles_retained=all(set(old_correct[k]).issubset(new_correct[k]) for k in old_correct)
            improved=math.isfinite(after['loss']) and after['loss']<prior['loss']
            retained=native_retention() if improved and roles_retained else None
            trials.append(dict(scale=scale,loss=after['loss'],actual_source_improved=improved,
                previously_correct_semantic_feedback_retained=roles_retained,
                original_native_retention=retained))
            if improved and roles_retained and retained['retained']:
                next_states=[deepcopy(state) for state in proposal['next_optimizer_state']]
                for parameter,state in zip(parameters,next_states):optimizer.state[parameter]=state
                committed=True
                return dict(accepted=True,scale=scale,before=prior,after=after,trials=trials,
                    original_optimizer_or_parameters_modified=True)
            scale*=.5
        return dict(accepted=False,scale=0.,before=prior,trials=trials,
            original_optimizer_or_parameters_modified=False)
    finally:
        if not committed:
            with torch.no_grad():
                for parameter,value in zip(parameters,before):parameter.copy_(value)
            for parameter,state in zip(parameters,proposal['previous_optimizer_state']):
                if state is None:optimizer.state.pop(parameter,None)
                else:optimizer.state[parameter]=state
