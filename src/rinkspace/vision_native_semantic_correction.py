"""Correct known source roles through the existing native feature network.

Original human supervision is replayed from complete training shots. Every
parameter proposal must improve the unchanged actual correction source and
retain all originally correct native human/region classifications. No fitted
scene, unknown pixels, generated masks or position labels enter this stage.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def train_native_semantic_correction(*,config,network,optimizer,saved,inputs,manifest,plan,
        checkpoint,checkpoint_sha256,steps=None):
    import av
    import torch
    from torch.nn import functional as F
    from .vision import (_image,_tensor,_annotations_in_crop,interval_supervision,
        primitive_supervision,source_reviewed_region_masks,write_json,semantic_exclusion_loss)
    from .vision_network import masked_region_loss
    from .vision_supervision import confirmed_position_interval_localization_loss
    from .vision_retention import source_retention_snapshot,source_retention_regressions
    from .source_semantic_feedback import known_source_semantic_point_loss
    from .strict_source_supervision import strict_negative_point_loss
    from .vision_source_optimizer_transaction import source_optimizer_proposal,guarded_source_optimizer_step
    from .vision_retention_hard_examples import (NativeRetentionHardExamples,
        protect_feedback_from_conflicting_original_regions)
    from calibration.program_tracking.strict_model_scope import authenticate_source_reference,file_sha

    if inputs.semantic_feedback is None or config.get('training_update_scope')!='all_trainable':
        raise ValueError('Native feature correction requires known semantic evidence and existing trainable features')
    classes=config['classes'];device=config['device'];spec=config['strict_source_stream_training']
    source=inputs.pack.metadata['source']
    original_guards=[s for s in manifest['samples'] if s.get('regions') or any(
        a.get('state')=='user_confirmed_visual_supervision' for a in s['annotations'])]
    if not original_guards:raise ValueError('Native source correction needs original retained supervision')
    guards=[]
    for sample in original_guards:
        rgb=_image(sample['frame_path'],sample.get('frame_key'));h,w=rgb.shape[:2]
        images=_tensor(rgb[None],device)
        annotations=_annotations_in_crop(sample,(0,0),(h,w))
        regions=[(dict(r,region_id=r.get('region_id',str(i))),mask) for i,(r,mask) in enumerate(
            source_reviewed_region_masks(sample,(0,0),(h,w),device))]
        binding=dict(source_sha256=sample['source_hash'],source_pts=sample['source_pts'],time_base=sample['time_base'],
            shot_id=sample['source_shot_scope']['shot_id'],native_hw=[h,w])
        guards.append(dict(images=images,annotations=annotations,regions=regions,sample=sample,binding=binding))

    def forward(images):
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            return network(images,hidden=None)

    def snapshots():
        network.eval();result=[]
        with torch.no_grad():
            for item in guards:
                output=forward(item['images'])
                result.append(source_retention_snapshot(output['logits'][:,0].float().softmax(1),
                    output['primitive_logits'][:,0].float().softmax(1),source_binding=item['binding'],
                    annotations=item['annotations'],classes=classes,reviewed_regions=item['regions']))
                del output
        return result

    baseline=snapshots()
    for item,snapshot in zip(guards,baseline):
        item['hard_examples']=NativeRetentionHardExamples(binding=snapshot['source_binding'],
            annotations=item['annotations'],classes=classes,reviewed_regions=item['regions'],
            whole_shot_role=item['sample']['source_shot_scope']['split'])
    human_by_native={int(s['source_pts']):s for s in manifest['samples'] if s['source_hash']==source['video_sha256']
        and Fraction(s['time_base'])==Fraction(source['time_base'])}
    targets={}
    for native in inputs.semantic_feedback.frame_bindings:
        human=human_by_native.get(native)
        supervision=inputs.supervision_catalog(native,human_annotations=human['annotations'] if human else ())
        if supervision['semantic_annotations']:targets[native]=supervision
    if not targets:raise ValueError('No known semantic correction belongs to a complete training shot')
    source_path=inputs.exact_frame_binding(min(targets))['source_path']
    authenticate_source_reference(dict(path=source_path,sha256=source['video_sha256']))
    with av.open(source_path) as container:
        stream=container.streams.video[0];stream.thread_count=int(spec.get('cpu_threads',1))
        if Fraction(stream.time_base)!=Fraction(source['time_base']):raise ValueError('Native correction clock differs')
        container.seek(min(targets),stream=stream,backward=True)
        for frame in container.decode(stream):
            if frame.pts is None or int(frame.pts)<min(targets):continue
            if int(frame.pts)>max(targets):break
            native=int(frame.pts)
            if native not in targets:continue
            bgr=frame.to_ndarray(format='bgr24');binding=inputs.exact_frame_binding(native)
            human=human_by_native.get(native);human_annotations=human['annotations'] if human else ()
            if human and not np.array_equal(bgr[...,::-1],_image(human['frame_path'],human.get('frame_key'))):
                raise ValueError('Known role feedback and original human supervision use different actual RGB')
            supervision=inputs.frame_supervision(bgr,binding,human_annotations=human_annotations)
            current_regions=list(source_reviewed_region_masks(human,(0,0),(1080,1920),device)) if human else []
            protect_feedback_from_conflicting_original_regions(supervision['semantic_annotations'],current_regions,
                native_hw=(1080,1920))
            sample=dict(annotations=supervision['annotations'],source_hash=source['video_sha256'],
                source_pts=native,time_base=source['time_base'])
            targets[native]=dict(supervision,images=_tensor(np.ascontiguousarray(bgr[...,::-1])[None],device),
                binding=binding,sample=sample,human_annotations=human_annotations)
    if any('images' not in target for target in targets.values()):raise ValueError('Not every actual correction frame was read')

    def original_loss(item,output):
        logits=output['logits'][:,0]
        positive,_=interval_supervision(logits,item['annotations'],classes)
        position,_=confirmed_position_interval_localization_loss(logits,item['annotations'],classes,
            original_sample=item['sample'],crop_origin_xy=(0,0),rank_weight=config.get('human_position_rank_weight',0.))
        primitive=primitive_supervision(output['primitive_logits'][:,0],item['annotations'])
        region_losses=[]
        for region,mask in item['regions']:
            if region.get('excluded_kinds'):
                value,_=semantic_exclusion_loss(logits,mask,region['excluded_kinds'],classes,item['annotations'])
            else:
                value,_=masked_region_loss(logits,output['primitive_logits'][:,0],mask,
                    classes.index(region['kind']),('line','circle','other').index(region.get('primitive','other')))
            region_losses.append(value)
        return (positive+config.get('human_position_localization_weight',1.)*position
            +config.get('primitive_supervision_weight',.2)*primitive
            +(torch.stack(region_losses).mean() if region_losses else logits.sum()*0))

    def correction_loss(target,output):
        logits=output['logits'][:,0];h,w=logits.shape[-2:]
        annotations=_annotations_in_crop(target['sample'],(0,0),(h,w))
        positive,details=interval_supervision(logits,annotations,classes)
        position,_=confirmed_position_interval_localization_loss(logits,annotations,classes,
            original_sample=target['sample'],crop_origin_xy=(0,0),rank_weight=config.get('human_position_rank_weight',0.))
        negative,ncount=strict_negative_point_loss(logits,target['negatives'],classes,human_annotations=annotations)
        semantic,scount=known_source_semantic_point_loss(logits,target['semantic_annotations'],classes,
            source_binding=target['binding'],human_annotations=target['human_annotations'])
        return positive+config.get('human_position_localization_weight',1.)*position+negative+semantic,dict(
            support_count=sum(d['sample_count'] for d in details)+ncount+scount,semantic_count=scount,
            known_role_loss=semantic,role_absence_loss=negative)

    def objective():
        network.eval();losses=[];combined_losses=[];support=correct=semantic_count=0;correct_indices={}
        with torch.no_grad():
            for native,target in sorted(targets.items()):
                output=forward(target['images']);loss,counts=correction_loss(target,output)
                combined_losses.append(float(loss));losses.append(float(counts['known_role_loss']))
                support+=counts['semantic_count'];semantic_count+=counts['semantic_count']
                probabilities=output['logits'][:,0].float().softmax(1);h,w=probabilities.shape[-2:]
                correct_indices[str(native)]=[]
                for annotation in target['semantic_annotations']:
                    uv=torch.as_tensor(annotation['uv'],device=device,dtype=torch.float32)
                    grid=uv*uv.new_tensor([2/(w-1),2/(h-1)])-1
                    labels=F.grid_sample(probabilities,grid[None,:,None],align_corners=True)[0,:,:,0].argmax(0)
                    good=labels==classes.index(annotation['kind']);correct+=int(good.sum())
                    correct_indices[str(native)].extend(np.asarray(annotation['original_point_indices'])[good.cpu().numpy()].tolist())
                del output
        return dict(loss=float(np.mean(losses)),support_count=support,
            combined_actual_source_loss=float(np.mean(combined_losses)),
            source_keys=[[source['video_sha256'],source['time_base'],native] for native in sorted(targets)],
            known_semantic_correct_points=correct,known_semantic_point_count=semantic_count,
            known_semantic_correct_original_indices=correct_indices)

    def retention():
        comparisons=[source_retention_regressions(current,prior) for current,prior in zip(snapshots(),baseline)]
        for item,comparison in zip(guards,comparisons):item['hard_examples'].observe(comparison)
        return dict(retained=all(x['retained_without_new_classification_errors'] for x in comparisons),
            original_native_frame_count=len(comparisons),newly_wrong_counts=[sum(f['newly_wrong_count']
                for f in x['newly_lost_confirmed_support']) for x in comparisons],
            existing_source_hard_example_points=sum(item['hard_examples'].point_count for item in guards))

    initial=objective();last=initial;accepted=0;run_id=time.time_ns()
    directory=Path(config['training_run_directory']).resolve()
    if not directory.is_relative_to(Path(config['data_directory']).resolve()):
        raise ValueError('Native training artifacts must stay in configured Data')
    directory.mkdir(parents=True,exist_ok=True)
    ledger=directory/f'native_semantic_correction_{run_id}.jsonl'
    wanted=steps if steps is not None else config['training_steps']
    if isinstance(wanted,bool) or not isinstance(wanted,int) or wanted<1:
        raise ValueError('Native correction uses the positive configured training-step allocation')
    replay=[g for g in guards if g['sample']['source_shot_scope']['split']=='train']
    if not replay:raise ValueError('Original semantic correction replay has no complete training shots')
    for index in range(wanted):
        if last['known_semantic_correct_points']==last['known_semantic_point_count']:break
        network.train();optimizer.zero_grad(set_to_none=True)
        prior_constraint_events=sum(g['hard_examples'].observed_events for g in replay)
        for item in replay:
            output=forward(item['images'])
            hard=item['hard_examples'].loss(output['logits'][:,0],output['primitive_logits'][:,0],binding=item['binding'])
            loss=(original_loss(item,output)+config.get('human_position_localization_weight',1.)*hard)/len(replay)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite original-source replay')
            loss.backward();del loss,output
        for target in targets.values():
            output=forward(target['images']);_,counts=correction_loss(target,output)
            loss=counts['known_role_loss']+counts['role_absence_loss']
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite actual semantic correction')
            (loss/len(targets)).backward();del loss,output
        norm=torch.nn.utils.clip_grad_norm_(network.parameters(),5.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite actual native gradient')
        proposal=source_optimizer_proposal(optimizer)
        decision=guarded_source_optimizer_step(optimizer,proposal,source_objective=objective,native_retention=retention)
        with ledger.open('a',encoding='utf8') as saved_ledger:
            saved_ledger.write(json.dumps(dict(proposal_index=index,decision=decision,
                original_replay_frames=len(replay),known_correction_native_pts=sorted(targets)),separators=(',',':'))+'\n')
        if not decision['accepted']:
            if sum(g['hard_examples'].observed_events for g in replay)>prior_constraint_events:
                continue
            break
        accepted+=1;last=decision['after']
    final=objective();kept=retention()
    inputs.semantic_feedback.verify_evidence_unchanged();authenticate_source_reference(inputs.authority)
    if spec.get('negative_inputs'):authenticate_source_reference(spec['negative_inputs'])
    if file_sha(checkpoint)!=checkpoint_sha256:raise ValueError('The original resumed model changed during correction')
    output_path=directory/f'checkpoint_native_semantic_source_candidate_{run_id}.pt'
    if accepted:
        torch.save(dict(saved,state_dict=network.state_dict(),optimizer=optimizer.state_dict(),
            step=int(saved.get('step',0))+accepted,strict_source_supervision=spec,
            strict_source_training_update_scope='all_trainable',
            native_source_objective='known_semantic_role_CE_with_original_supervision_replay',
            strict_source_human_retention_failed=not kept['retained'],
            strict_source_resume_checkpoint_sha256=checkpoint_sha256,
            strict_source_previous_optimizer_state_available=saved.get('optimizer') is not None),output_path)
    report=dict(schema='rinkspace.native-semantic-source-correction.v1',
        checkpoint=str(output_path) if accepted else None,successful_optimizer_steps=accepted,
        initial_source=initial,final_source=final,original_native_retention=kept,
        source_supervision=spec,actual_source_consumption=dict(path=str(ledger),sha256=file_sha(ledger)) if ledger.exists() else None,
        parameter_acceptance_objective='known_semantic_role_CE',
        hard_examples_are_unchanged_original_source_labels=True,
        existing_source_hard_example_points=sum(item['hard_examples'].point_count for item in guards),
        cpu_threads=int(spec.get('cpu_threads',1)),device=device,whole_shot_plan_unchanged=True,
        unknown_pixels_used_as_labels=False,new_position_targets_created=False,formal_model_replaced=False,
        candidate_only=True,independent_source_accuracy_validated=False)
    write_json(directory/f'native_semantic_correction_{run_id}.json',report)
    return report
