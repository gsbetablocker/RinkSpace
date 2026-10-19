"""Finite semantic-head proposals on actual frozen source features.

All support is sampled after image-only forward. This guard adds no labels,
does not advance recurrent state, and cannot certify an optical boundary.
CPU head arithmetic must reproduce the captured baseline's original decisions;
a mismatched proxy fails closed. The native runtime guard remains required.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import numpy as np
import torch
from torch.nn import functional as F

from .vision_supervision import (active_source_supervision_mask,
    annotation_revision_sha256, source_sample_key)


def _tensor_sha(value):
    value=value.detach().cpu().contiguous()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _optimizer_head_signature(head,optimizer):
    def encode(value):
        if isinstance(value,torch.Tensor):return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=_tensor_sha(value))
        if isinstance(value,dict):return {str(k):encode(v) for k,v in sorted(value.items(),key=lambda item:str(item[0]))}
        if isinstance(value,(tuple,list)):return [encode(v) for v in value]
        return value
    records=[]
    for parameter in (head.weight,head.bias):
        groups=[group for group in optimizer.param_groups if any(parameter is member for member in group['params'])]
        if len(groups)!=1:raise ValueError('Head parameter needs exactly one optimizer group')
        records.append(dict(state=encode(optimizer.state.get(parameter,{})),
            settings=encode({k:v for k,v in groups[0].items() if k!='params'})))
    return hashlib.sha256(json.dumps(records,sort_keys=True,allow_nan=False).encode()).hexdigest()


def _head(weight,bias):
    if (weight.ndim!=4 or weight.shape[2:]!=(1,1) or bias.shape!=(weight.shape[0],) or
            not weight.is_floating_point() or not bias.is_floating_point() or
            not bool(torch.isfinite(weight).all()) or not bool(torch.isfinite(bias).all())):
        raise ValueError('Guard requires the finite existing semantic 1x1 weight and bias')
    return weight.detach().cpu().float().reshape(weight.shape[0],weight.shape[1]),bias.detach().cpu().float()


def _scores(features,weight,bias,precision):
    w,b=_head(weight,bias)
    if features.shape[-1]!=w.shape[1]:raise ValueError('Frozen source feature channels and head disagree')
    if precision=='bf16':
        return F.linear(features.to(torch.bfloat16),w.to(torch.bfloat16),b.to(torch.bfloat16)).float()
    if precision=='fp32':return F.linear(features.float(),w,b)
    raise ValueError('Head precision must be explicitly bf16 or fp32')


def _corners(field,uv):
    """Use the existing align_corners=True source sampling convention."""
    _,height,width=field.shape
    points=torch.as_tensor(uv,dtype=torch.float32)
    grid=points*points.new_tensor([2/(width-1),2/(height-1)])-1
    native=(grid+1)*grid.new_tensor([(width-1)/2,(height-1)/2])
    low=native.floor().long();fraction=native-low
    features=[];weights=[];valid=[]
    for dx,dy in ((0,0),(0,1),(1,0),(1,1)):
        x,y=low[:,0]+dx,low[:,1]+dy
        inside=(x>=0)&(x<width)&(y>=0)&(y<height)
        sampled=field[:,y.clamp(0,height-1),x.clamp(0,width-1)].T
        features.append(sampled)
        weights.append((fraction[:,0] if dx else 1-fraction[:,0])*(fraction[:,1] if dy else 1-fraction[:,1]))
        valid.append(inside)
    return torch.stack(features,1),torch.stack(weights,1),torch.stack(valid,1),grid


def _manual_probability(group,weight,bias,precision):
    pixels=_scores(group['corner_features'],weight,bias,precision).softmax(-1)
    return (pixels*group['corner_weights'][...,None]*group['corner_inbounds'][...,None]).sum(1)


def _native_cuda_probabilities(cache,weight,bias,precision):
    """Reuse the actual native CUDA field and convolution protocol exactly."""
    features=cache['native_CUDA_refined_features']
    if features.device.type!='cuda' or precision!='bf16':
        raise ValueError('Native CUDA head support needs the captured BF16 source protocol')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        logits=F.conv2d(features,weight.to(features.device),bias.to(features.device))
    return logits.float().softmax(1)


def _native_cuda_group_probability(probabilities,group):
    grid=group['source_grid'].to(probabilities.device)
    return F.grid_sample(probabilities,grid[None,:,None],align_corners=True)[0,:,:,0].T.cpu()


def native_cuda_original_supervision_loss(caches,weight,bias):
    """Replay only original known labels from complete training shots.

    Frozen actual image features avoid another encoder/decode pass. Every
    original active human point is included, including originally wrong model
    decisions. Reject-only regions retain only their original excluded classes.
    Development/test sources can constrain acceptance but never enter this loss.
    """
    losses=[]
    for cache in caches:
        if cache.get('whole_shot_role')!='train':
            raise ValueError('Original supervision replay requires a complete training-shot role')
        if not cache.get('native_GPU_baseline_decisions_verified'):
            raise ValueError('Original source replay requires verified actual native CUDA features')
        features=cache['native_CUDA_refined_features']
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            logits=F.conv2d(features,weight.to(features.device),bias.to(features.device))
        probabilities=logits.float().softmax(1)
        for group in cache['groups']:
            grid=group['source_grid'].to(features.device)
            actual=F.grid_sample(probabilities,grid[None,:,None],align_corners=True)[0,:,:,0].T
            losses.append(-actual[:,group['expected_class']].clamp_min(torch.finfo(torch.float32).tiny).log().mean())
        field=logits[0].float().flatten(1).T
        for region in cache['regions']:
            scores=field[region['native_flat_indices'].to(features.device)]
            if not len(scores):continue
            if region['kind'] is not None:
                target=torch.full((len(scores),),cache['classes'].index(region['kind']),
                    device=features.device,dtype=torch.long)
                losses.append(F.cross_entropy(scores,target))
            else:
                allowed=[i for i,name in enumerate(cache['classes']) if name not in region['excluded_kinds']]
                losses.append((torch.logsumexp(scores,dim=1)-torch.logsumexp(scores[:,allowed],dim=1)).mean())
    if not losses:raise ValueError('Original supervision replay has no known source support')
    return torch.stack(losses).mean()


def build_head_support_cache(refined,probabilities,*,sample,source_binding,classes,
        reference_weight,reference_bias,reference_snapshot,reviewed_regions=(),reference_precision='bf16',
        retain_native_cuda_features=False):
    """Cache every highest active-visible UV and each known region after forward.

    ``reference_snapshot`` is the existing immutable native guard result, never
    reconstructed from the proposal. The captured posterior must match that
    record exactly by original index. Any CPU head baseline decision mismatch
    is reported and makes proposal acceptance unavailable, rather than changing
    the labels, the baseline or the position interval.
    """
    names=tuple(classes);key=source_sample_key(sample)
    if source_sample_key(source_binding)!=key or source_sample_key(reference_snapshot['source_binding'])!=key:
        raise ValueError('Head support and original guard belong to different source frames')
    if reference_snapshot['classes']!=list(names):raise ValueError('Head support vocabulary changed')
    if (refined.ndim!=4 or refined.shape[0]!=1 or probabilities.shape!=(1,len(names),*refined.shape[-2:]) or
            min(refined.shape[-2:])<2 or not bool(torch.isfinite(refined).all()) or
            not bool(torch.isfinite(probabilities).all()) or bool((probabilities<0).any()) or
            bool((probabilities>1).any()) or len(set(names))!=len(names)):
        raise ValueError('Head support needs actual aligned native features and class probabilities')
    if not torch.allclose(probabilities.sum(1),torch.ones_like(probabilities[:,0]),atol=1e-3):
        raise ValueError('Captured class probabilities are not normalized')
    _head(reference_weight,reference_bias)
    height,width=refined.shape[-2:]
    if sample.get('frame_size_wh')!=[width,height]:raise ValueError('Head support must use full native source dimensions')
    field=refined[0].detach().cpu().float();posterior=probabilities[0].detach().cpu().float()
    originals={a['group_id']:a for a in sample['annotations']}
    if len(originals)!=len(sample['annotations']):raise ValueError('Head support has duplicate original groups')
    previous={a['group_id']:a for a in reference_snapshot['human_groups']}
    groups=[];proxy_mismatches=[]
    for identity,original in originals.items():
        if original.get('state')!='user_confirmed_visual_supervision' or original.get('kind') not in names:continue
        uv=np.asarray(original['uv'],float)
        active=np.asarray(active_source_supervision_mask(original),bool)
        visible=np.asarray(original.get('source_visible_mask',[True]*len(uv)))
        if visible.dtype!=np.bool_ or visible.shape!=active.shape:raise ValueError('Head support visibility must preserve all original indices')
        eligible=active&visible&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
        indices=np.flatnonzero(eligible)
        if not len(indices):continue
        original_reference=previous.get(identity)
        if (original_reference is None or original_reference['source_point_indices']!=indices.tolist() or
                original_reference['kind']!=original['kind']):
            raise ValueError('Head cache cannot omit or revise original highest visible support')
        features,weights,inbounds,grid=_corners(field,uv[indices])
        actual=F.grid_sample(posterior[None],grid[None,:,None],align_corners=True)[0,:,:,0].T
        expected=names.index(original['kind']);wrong=actual.argmax(1)!=expected
        if indices[wrong.numpy()].tolist()!=original_reference['semantic_wrong_source_point_indices']:
            raise ValueError('Captured posterior differs from the immutable native classification baseline')
        group=dict(group_id=identity,kind=original['kind'],expected_class=expected,
            original_annotation_sha256=annotation_revision_sha256(original),source_point_indices=indices,
            original_uv=uv[indices].copy(),corner_features=features,corner_weights=weights,corner_inbounds=inbounds,
            source_grid=grid,
            baseline_probability=actual,baseline_correct=~wrong,
            original_position_count=len(uv),retired_source_positions_used=0,
            model_confidence_filter_used=False,position_intervals_changed=False)
        proxy=_manual_probability(group,reference_weight,reference_bias,reference_precision)
        mismatch=(proxy.argmax(1)!=actual.argmax(1)).numpy()
        if mismatch.any():proxy_mismatches.append(dict(group_id=identity,original_indices=indices[mismatch].tolist()))
        groups.append(group)
    if {g['group_id'] for g in groups}!=set(previous):raise ValueError('Head cache lost an original confirmed group')
    regions=[];old_regions={r['region_id']:r for r in reference_snapshot['reviewed_regions']}
    for region,mask in reviewed_regions:
        identity=region.get('region_id')
        if (not identity or mask.dtype!=torch.bool or mask.shape!=(1,1,height,width) or identity not in old_regions):
            raise ValueError('Head region support must use original named native masks')
        mask=mask.detach().cpu();flat=torch.nonzero(mask[0,0].flatten(),as_tuple=False)[:,0]
        features=field.reshape(field.shape[0],-1).T[flat]
        actual=posterior.reshape(len(names),-1).T[flat]
        expected=region.get('kind');excluded=tuple(region.get('excluded_kinds',()))
        if expected in names and not excluded:
            bad=actual.argmax(1)!=names.index(expected)
        elif expected is None and excluded and len(set(excluded))==len(excluded) and all(x in names for x in excluded):
            bad=torch.isin(actual.argmax(1),torch.tensor([names.index(x) for x in excluded]))
        else:raise ValueError('Known region must retain its positive or reject-only original role')
        old=old_regions[identity]
        if (old['kind']!=expected or old['excluded_kinds']!=list(excluded) or old['supervised_pixels']!=len(flat) or
                old['native_mask_sha256']!=hashlib.sha256(mask.numpy().tobytes()).hexdigest() or
                flat[bad].tolist()!=old['semantic_wrong_native_flat_indices']):
            raise ValueError('Known native region mask or baseline classification changed')
        proxy=_scores(features,reference_weight,reference_bias,reference_precision).softmax(-1)
        mismatch=proxy.argmax(1)!=actual.argmax(1)
        if bool(mismatch.any()):proxy_mismatches.append(dict(region_id=identity,native_flat_indices=flat[mismatch].tolist()))
        regions.append(dict(region_id=identity,kind=expected,excluded_kinds=list(excluded),
            native_flat_indices=flat,features=features,baseline_probability=actual,baseline_correct=~bad))
    if {r['region_id'] for r in regions}!=set(old_regions):raise ValueError('Head cache lost known source regions')
    cache=dict(schema='rinkspace.frozen-source-head-support.v1',source_key=list(key),classes=list(names),
        source_binding=deepcopy(source_binding),native_hw=[height,width],groups=groups,regions=regions,
        reference_precision=reference_precision,reference_head_sha256=[_tensor_sha(reference_weight),_tensor_sha(reference_bias)],
        original_feature_dtype=str(refined.dtype),source_feature_sha256=_tensor_sha(refined),
        CPU_proxy_baseline_decisions_verified=not proxy_mismatches,CPU_proxy_mismatches=proxy_mismatches,
        human_UV_used_as_model_input=False,unknown_pixels_assigned_labels=False,independent_accuracy_accepted=False)
    if retain_native_cuda_features:
        if refined.device.type!='cuda' or reference_precision!='bf16':
            raise ValueError('Native head cache requires the original CUDA BF16 forward')
        cache['native_CUDA_refined_features']=refined.detach().clone()
        actual=_native_cuda_probabilities(cache,reference_weight,reference_bias,reference_precision)
        mismatches=[]
        for group in groups:
            labels=_native_cuda_group_probability(actual,group).argmax(1)
            before=group['baseline_probability'].argmax(1)
            changed=labels!=before
            if bool(changed.any()):mismatches.append(dict(group_id=group['group_id'],
                original_indices=group['source_point_indices'][changed.numpy()].tolist()))
        labels=actual.argmax(1)[0].flatten()
        for region in regions:
            current=labels[region['native_flat_indices'].to(labels.device)].cpu()
            before=region['baseline_probability'].argmax(1)
            changed=current!=before
            if bool(changed.any()):mismatches.append(dict(region_id=region['region_id'],
                native_flat_indices=region['native_flat_indices'][changed].tolist()))
        cache['native_GPU_baseline_decisions_verified']=not mismatches
        cache['native_GPU_baseline_mismatches']=mismatches
    return cache


def evaluate_head_retention(caches,weight,bias,*,precision=None,expected_source_keys):
    """Finite candidate probabilities; preserve every previously correct index."""
    keys=[tuple(c['source_key']) for c in caches]
    if len(set(keys))!=len(keys) or set(keys)!=set(tuple(k) for k in expected_source_keys):
        raise ValueError('Head constraint cannot use an incomplete or duplicate source inventory')
    findings=[]
    for cache in caches:
        native='native_CUDA_refined_features' in cache
        verified=(cache.get('native_GPU_baseline_decisions_verified') if native
            else cache['CPU_proxy_baseline_decisions_verified'])
        if cache.get('schema')!='rinkspace.frozen-source-head-support.v1' or not verified:
            raise ValueError('CPU head proxy did not reproduce the original native baseline decisions')
        mode=precision or cache['reference_precision'];names=cache['classes']
        probabilities=_native_cuda_probabilities(cache,weight,bias,mode) if native else None
        for group in cache['groups']:
            pp=(_native_cuda_group_probability(probabilities,group) if native else
                _manual_probability(group,weight,bias,mode))
            lost=group['baseline_correct']&(pp.argmax(1)!=group['expected_class'])
            if bool(lost.any()):findings.append(dict(source_key=cache['source_key'],group_id=group['group_id'],kind=group['kind'],
                newly_wrong_original_indices=group['source_point_indices'][lost.numpy()].tolist(),
                original_source_UV=group['original_uv'][lost.numpy()].tolist(),
                candidate_probabilities=pp[lost].tolist(),previous_probabilities=group['baseline_probability'][lost].tolist()))
        for region in cache['regions']:
            label=(probabilities.argmax(1)[0].flatten()[region['native_flat_indices'].to(probabilities.device)].cpu()
                if native else _scores(region['features'],weight,bias,mode).argmax(-1))
            wrong=(label!=names.index(region['kind']) if region['kind'] is not None else
                torch.isin(label,torch.tensor([names.index(x) for x in region['excluded_kinds']])))
            lost=region['baseline_correct']&wrong
            if bool(lost.any()):findings.append(dict(source_key=cache['source_key'],region_id=region['region_id'],
                newly_wrong_native_flat_indices=region['native_flat_indices'][lost].tolist(),
                unknown_pixels_not_included=True))
    return dict(retained=not findings,findings=findings,checked_source_count=len(caches),
        actual_native_runtime_guard_still_required=True,classification_is_not_RGB_edge_precision=True)


def frozen_head_localization(refined,weight,bias,*,sample,annotations,classes,crop_origin_xy,
        precision='bf16',profile_step_px=1.,rank_weight=1.):
    """Evaluate the existing loss on this actual frozen source/crop only."""
    from .vision_supervision import confirmed_position_interval_localization_loss
    w,b=_head(weight,bias);feature=refined.detach().cpu()
    dtype=torch.bfloat16 if precision=='bf16' else torch.float32 if precision=='fp32' else None
    if dtype is None:raise ValueError('Head precision must be explicitly bf16 or fp32')
    logits=F.conv2d(feature.to(dtype),w[:,:,None,None].to(dtype),b.to(dtype)).float()
    loss,diagnostics=confirmed_position_interval_localization_loss(logits,annotations,classes,
        original_sample=sample,crop_origin_xy=crop_origin_xy,profile_step_px=profile_step_px,rank_weight=rank_weight)
    return dict(loss=float(loss),sample_count=sum(d['sample_count'] for d in diagnostics),
        diagnostics=diagnostics,source_key=list(source_sample_key(sample)),precision=precision,
        original_intervals_normals_and_authority_changed=False,unknown_pixels_assigned_background=False)


def adamw_head_proposal(head,optimizer):
    """Run AdamW on two private clones, preserving original parameters/state.

    Use the actual already-clipped gradients, loaded moments and param-group
    settings. Frozen shared/private parameters must have grad=None. This makes
    a real finite Adam proposal, not an inferred historic batch or new update.
    """
    if not isinstance(optimizer,torch.optim.AdamW):raise ValueError('Only the existing AdamW optimizer protocol is supported')
    _head(head.weight,head.bias)
    originals=[head.weight,head.bias]
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in originals):
        raise ValueError('Adam head proposal requires both actual finite clipped gradients')
    for group in optimizer.param_groups:
        if group.get('differentiable'):raise ValueError('Differentiable AdamW proposals are unsupported')
        if any(p is not head.weight and p is not head.bias and p.grad is not None for p in group['params']):
            raise ValueError('Shared and primitive optimizer parameters must remain frozen')
    clones=[torch.nn.Parameter(p.detach().clone()) for p in originals]
    groups=[];seen=[]
    for group in optimizer.param_groups:
        chosen=[(p,q) for p,q in zip(originals,clones) if any(p is member for member in group['params'])]
        if chosen:
            groups.append({**{k:deepcopy(v) for k,v in group.items() if k!='params'},'params':[q for _,q in chosen]})
            seen.extend(p for p,_ in chosen)
    if len(seen)!=2 or any(sum(p is x for x in seen)!=1 for p in originals):
        raise ValueError('Both head parameters need their unique original optimizer group')
    temporary=torch.optim.AdamW(groups)
    for original,clone in zip(originals,clones):
        clone.grad=original.grad.detach().clone()
        if original in optimizer.state:temporary.state[clone]=deepcopy(optimizer.state[original])
    temporary.step()
    return dict(before_weight=head.weight.detach().clone(),before_bias=head.bias.detach().clone(),
        proposed_weight=clones[0].detach().clone(),proposed_bias=clones[1].detach().clone(),
        next_weight_optimizer_state=deepcopy(temporary.state[clones[0]]),
        next_bias_optimizer_state=deepcopy(temporary.state[clones[1]]),
        source_head_sha256=[_tensor_sha(head.weight),_tensor_sha(head.bias)],
        source_optimizer_head_sha256=_optimizer_head_signature(head,optimizer),
        original_optimizer_or_parameters_modified=False,actual_clipped_gradients_used=True)


def guarded_head_backtrack(caches,proposal,*,expected_source_keys,localization_evaluator,
        precision=None,scales=(1.,.5,.25,.125,.0625)):
    """Accept a finite existing-head delta only if retention and localization pass."""
    scales=tuple(scales)
    if not scales or any(isinstance(s,bool) or not np.isfinite(s) or not 0<s<=1 for s in scales) or len(set(scales))!=len(scales):
        raise ValueError('Proposal scales must be distinct finite positive fractions at most one')
    before=localization_evaluator(proposal['before_weight'],proposal['before_bias'])
    if before.get('sample_count',0)<1 or not np.isfinite(before['loss']):raise ValueError('Localization guard requires actual supported source positions')
    trials=[]
    for scale in scales:
        weight=proposal['before_weight']+scale*(proposal['proposed_weight']-proposal['before_weight'])
        bias=proposal['before_bias']+scale*(proposal['proposed_bias']-proposal['before_bias'])
        retention=evaluate_head_retention(caches,weight,bias,precision=precision,expected_source_keys=expected_source_keys)
        after=localization_evaluator(weight,bias)
        if after.get('sample_count')!=before['sample_count'] or after.get('source_key')!=before.get('source_key'):
            raise ValueError('Localization candidate silently changed source support')
        improved=np.isfinite(after['loss']) and after['loss']<before['loss']
        trials.append(dict(scale=float(scale),retention=retention,localization=after,localization_improved=bool(improved)))
        if retention['retained'] and improved:
            return dict(accepted=True,scale=float(scale),weight=weight,bias=bias,
                localization_before=before,trials=trials,source_head_sha256=proposal['source_head_sha256'],
                actual_native_guard_still_required=True,optimizer_success_record_not_written=True)
    return dict(accepted=False,scale=0.,localization_before=before,trials=trials,
        source_head_sha256=proposal['source_head_sha256'],original_optimizer_or_parameters_modified=False,
        optimizer_success_record_not_written=True)


def commit_guarded_head_proposal(head,optimizer,proposal,decision):
    """Commit accepted head tensors and their proposed moments, never a rejection."""
    if not decision.get('accepted') or not 0<decision.get('scale',0)<=1:
        raise ValueError('Rejected parameter proposals cannot become successful optimizer steps')
    signature=[_tensor_sha(head.weight),_tensor_sha(head.bias)]
    if signature!=proposal['source_head_sha256'] or signature!=decision['source_head_sha256']:
        raise ValueError('Head parameters changed after their guarded proposal')
    if _optimizer_head_signature(head,optimizer)!=proposal['source_optimizer_head_sha256']:
        raise ValueError('Optimizer head moments/settings changed after their guarded proposal')
    scale=decision['scale']
    for name in ('weight','bias'):
        expected=proposal['before_'+name]+scale*(proposal['proposed_'+name]-proposal['before_'+name])
        if not torch.equal(expected,decision[name]):raise ValueError('Accepted head delta changed after evaluation')
    with torch.no_grad():
        head.weight.copy_(decision['weight']);head.bias.copy_(decision['bias'])
    optimizer.state[head.weight]=deepcopy(proposal['next_weight_optimizer_state'])
    optimizer.state[head.bias]=deepcopy(proposal['next_bias_optimizer_state'])
    return dict(actual_head_optimizer_step_committed=True,applied_delta_scale=float(scale),
        shared_or_primitive_parameters_updated=False,backtracked_moments_use_actual_proposed_gradient=True)
