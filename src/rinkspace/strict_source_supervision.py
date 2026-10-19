"""Strict source-observation bridge to the existing vision supervision losses.

Every positive is an admitted current RGB measurement, with actual readout
uncertainty. Unknown/pending/excluded-by-default pixels are neither positives
nor negatives. The streaming trainer receives shared decoded original frames;
it never exports PNGs or uses temporal predictions as training positions.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from calibration.program_tracking.strict_absolute_source import (
    StrictAbsoluteSourcePack, _row_admission_problem, _exact_binding,
)
from calibration.program_tracking.strict_model_scope import (
    ExactSourceScopeReview, REVIEW_SCHEMA, authenticate_source_reference, file_sha,
)


NEGATIVE_SCHEMA='rinkspace.strict-source-role-absence-pixels.v1'


def _actual_normal_and_support(row):
    uv=np.asarray(row['uv'],float).reshape(-1,2)
    evidence=row.get('source_readout_uncertainty',{})
    if (evidence.get('kind')!='actual_source_readout_blur_and_quantization'
            or evidence.get('actual_source_RGB_measured') is not True
            or evidence.get('paint_width_used_as_center_tolerance') is not False
            or not evidence.get('measurement_method')
            or not isinstance(evidence.get('quantization_step_px'),(int,float))
            or evidence['quantization_step_px']<0):
        return None
    interval=np.asarray(evidence.get('normal_interval_px',[]),float)
    blur=np.asarray(evidence.get('actual_readout_blur_interval_px',[]),float)
    if interval.shape==(2,):interval=np.tile(interval,(len(uv),1))
    if blur.shape==(2,):blur=np.tile(blur,(len(uv),1))
    if (interval.shape!=(len(uv),2) or blur.shape!=interval.shape
            or not np.isfinite(interval).all() or not np.isfinite(blur).all()
            or np.any(interval[:,0]>0) or np.any(interval[:,1]<0)
            or np.any(blur[:,0]>blur[:,1])):
        return None
    if row['observation_semantics']=='paint_centerline':
        sides=np.asarray(row['source_sides_uv'],float)
        normal=sides[:,1]-sides[:,0]
    elif row.get('source_normal_measured_from_actual_RGB') is True:
        normal=np.asarray(row.get('normal',[]),float)
    elif row.get('source_neighbour_adjacency_confirmed') is True and len(uv)>1:
        tangent=np.gradient(uv,axis=0)
        normal=np.c_[-tangent[:,1],tangent[:,0]]
    else:
        return None
    if normal.shape!=(len(uv),2) or not np.isfinite(normal).all():return None
    length=np.linalg.norm(normal,axis=1)
    if np.any(length<=0):return None
    return normal/length[:,None],interval


def strict_positive_annotations(rows, *, authority, classes, human_annotations=()):
    """Return ordinary source_reviewed_interval records; never exact pixel GT.

    The complete wide painted band is retained as measurement evidence only.
    It cannot become a centerline tolerance. Missing blur/quantization support
    withholds that row from training without changing its source observations.
    """
    authenticate_source_reference(authority)
    return _positive_annotations_from_authenticated_pack(rows, authority=authority,
        classes=classes, human_annotations=human_annotations)


def _positive_annotations_from_authenticated_pack(rows, *, authority, classes, human_annotations=()):
    """Internal formatter; the input pack has already been SHA-authenticated."""
    annotations=[]
    for i,row in enumerate(rows):
        if row.get('split')!='fit':continue
        if _row_admission_problem(row):continue
        source=_actual_normal_and_support(row)
        if source is None:continue
        kind=row.get('vision_semantic_kind')
        if kind is None and row['scope']=='ice':kind='ice_paint'
        if kind is None and row['observation_semantics'] in ('advertisement_boundary','advertisement_corner'):
            kind='ad_boundary'
        if kind not in classes:continue
        # Human positions/semantics are immutable and have higher authority.
        # A source interval must not supply a conflicting label at their UVs.
        uv=np.asarray(row['uv'],float).reshape(-1,2)
        for human in human_annotations:
            if human.get('state')!='user_confirmed_visual_supervision':continue
            if human.get('feature_id')==row['feature_id'] and human.get('kind')!=kind:
                raise ValueError('Source supervision conflicts with the direct human feature identity')
        normal,interval=source
        annotations.append(dict(group_id=f"strict_source_{row['family']}_{row['actual_pts']}_{i}",
            feature_id=row['feature_id'],kind=kind,primitive=row.get('vision_primitive'),
            source_curve_id=row.get('curve_id'),physical_material_id=row.get('physical_material_id'),
            boundary_role=row['observation_semantics'],uv=uv.tolist(),normal=normal.tolist(),
            interval_px=interval.tolist(),state='source_reviewed_interval',authority=deepcopy(authority),
            identity_authority=deepcopy(row['temporal_evidence']),source_identity_verified=True,
            actual_source_RGB_locations_used=True,propagated_search_UV_used_as_labels=False,
            model_projection_used_to_generate_or_select_uv=False,pixel_ground_truth=False,
            is_calibrated_uncertainty=False,new_human_positions_created=False,
            source_readout_uncertainty=deepcopy(row['source_readout_uncertainty']),
            original_point_indices=np.asarray(row.get('original_point_indices',np.arange(len(uv)))).tolist()))
    return annotations


def write_strict_negative_source_pack(path,review):
    """Export only explicitly reviewed role absence; pending is not negative."""
    path=Path(path).resolve()
    if path.exists():raise ValueError('Do not overwrite source negative evidence')
    rows=[]
    for proof in review.records.values():
        if (proof['decision']!='exclude' or proof.get('reject_only_authority')!='actual_visual_role_absence'
                or proof.get('actual_source_RGB_role_absence_confirmed') is not True
                or proof.get('confidence_used_as_rejection_evidence') is not False
                or proof.get('model_probability_used_as_rejection_evidence') is not False):
            continue
        kinds=proof.get('excluded_kinds')
        if not isinstance(kinds,list) or not kinds or len(set(kinds))!=len(kinds):
            raise ValueError('Reviewed role absence needs explicit distinct excluded semantic kinds')
        rows.append(deepcopy(proof))
    review.verify_evidence_unchanged()
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,uv=np.asarray([r.pop('uv') for r in rows],float).reshape(-1,2),
        rows_json=json.dumps(rows,ensure_ascii=False,separators=(',',':')),
        metadata_json=json.dumps(dict(schema=NEGATIVE_SCHEMA,source=review.source,
            default_exclusions_used_as_negatives=False,pending_used_as_truth=False,
            other_positive_class_assigned=False),separators=(',',':')))
    return dict(path=str(path),sha256=file_sha(path),points=len(rows))


class StrictSourceTrainingInputs:
    def __init__(self,absolute_reference,*,negative_reference=None,semantic_reference=None,shot_plan=None,classes):
        self.authority=deepcopy(absolute_reference)
        self.pack=StrictAbsoluteSourcePack(absolute_reference['path'],expected_sha256=absolute_reference['sha256'])
        self.classes=tuple(classes)
        self.semantic_feedback=None
        self.shot_plan=deepcopy(shot_plan)
        self.rows_by_native={};self.negative_by_native={};self.frame_bindings={}
        proof_by_material={(p['actual_pts'],p['physical_material_id']):p for p in self.pack.proof_templates}
        for row in self.pack.fit_rows+self.pack.held_rows:
            key=int(row['native_pts'])
            self.rows_by_native.setdefault(key,[]).append(row)
            proof=proof_by_material[(row['actual_pts'],row['physical_material_id'])]
            value=dict(actual_pts=row['actual_pts'],native_pts=key,decoded_BGR_sha256=row['decoded_BGR_sha256'],
                source_path=proof['source_image_evidence']['path'],shot_id=row['shot_id'])
            if key in self.frame_bindings and self.frame_bindings[key]!=value:
                raise ValueError('One native source frame has competing exact source identities')
            self.frame_bindings[key]=value
        if negative_reference is not None:
            authenticate_source_reference(negative_reference)
            with np.load(negative_reference['path'],allow_pickle=False) as z:
                metadata=json.loads(str(z['metadata_json'].item()))
                rows=json.loads(str(z['rows_json'].item()));uv=z['uv'].copy()
            if metadata.get('schema')!=NEGATIVE_SCHEMA or metadata['source']!=self.pack.metadata['source']:
                raise ValueError('Source negative pack identity differs')
            proofs=[dict(row,uv=point.tolist()) for row,point in zip(rows,uv)]
            if len(rows)!=len(uv):raise ValueError('Negative source proof/UV count differs')
            review=ExactSourceScopeReview(dict(schema=REVIEW_SCHEMA,source=metadata['source'],reviewed_endpoints=proofs))
            for proof in review.records.values():
                if (proof['decision']!='exclude' or proof.get('reject_only_authority')!='actual_visual_role_absence'
                        or proof.get('actual_source_RGB_role_absence_confirmed') is not True
                        or proof.get('confidence_used_as_rejection_evidence') is not False
                        or proof.get('model_probability_used_as_rejection_evidence') is not False):
                    raise ValueError('Default exclusion or pending cannot become negative supervision')
                if not proof.get('excluded_kinds') or any(k not in classes for k in proof['excluded_kinds']):
                    raise ValueError('Reviewed negative roles differ from vision semantic classes')
                self.negative_by_native.setdefault(int(proof['native_pts']),[]).append(proof)
                key=int(proof['native_pts'])
                value=dict(actual_pts=proof['actual_pts'],native_pts=key,decoded_BGR_sha256=proof['decoded_BGR_sha256'],
                    source_path=proof['source_image_evidence']['path'],shot_id=proof.get('shot_id'))
                previous=self.frame_bindings.get(key)
                if previous is not None and any(previous[k]!=value[k] for k in
                        ('actual_pts','native_pts','decoded_BGR_sha256','source_path')):
                    raise ValueError('Negative and positive source frame identities differ')
                if previous is None:self.frame_bindings[key]=value
        if semantic_reference is not None:
            from .source_semantic_feedback import KnownSourceSemanticFeedback
            if shot_plan is None:raise ValueError('Known semantic source feedback needs the unchanged whole-shot plan')
            feedback=KnownSourceSemanticFeedback(semantic_reference,classes=classes)
            if feedback.source!=self.pack.metadata['source']:
                raise ValueError('Known semantic feedback and admitted source clocks differ')
            for native,binding in feedback.frame_bindings.items():
                value={k:binding[k] for k in ('actual_pts','native_pts','decoded_BGR_sha256','shot_id')}
                value['source_path']=binding['source_path']
                previous=self.frame_bindings.get(native)
                if previous is not None and previous!=value:
                    raise ValueError('Known semantic and static source frame identities differ')
                self.frame_bindings[native]=value
            self.semantic_feedback=feedback

    def exact_frame_binding(self,native):
        item=self.frame_bindings[int(native)]
        source=self.pack.metadata['source']
        return dict(item,source_sha256=source['video_sha256'],time_base=source['time_base'])

    def supervision_catalog(self,native,*,human_annotations=()):
        """Source metadata for planning; this does not replace frame RGB checks.

        StrictAbsoluteSourcePack verified its immutable source file and every
        proof on construction. Avoid hashing the full pack once per crop/frame.
        The training entry also authenticates it again after consumption.
        """
        semantic=(self.semantic_feedback.catalog(self.exact_frame_binding(native),shot_plan=self.shot_plan,
            human_annotations=human_annotations) if self.semantic_feedback is not None else [])
        return dict(annotations=list(deepcopy(human_annotations))+
            _positive_annotations_from_authenticated_pack(self.rows_by_native.get(int(native),[]),
                authority=self.authority,classes=self.classes,human_annotations=human_annotations),
            negatives=deepcopy([p for p in self.negative_by_native.get(int(native),[]) if p.get('split')!='held']),
            semantic_annotations=semantic)

    def frame_supervision(self,bgr,binding,*,human_annotations=()):
        binding=_exact_binding(binding)
        digest=hashlib.sha256(np.ascontiguousarray(bgr).tobytes()).hexdigest()
        if digest!=binding['decoded_BGR_sha256']:
            raise ValueError('Shared source RGB stream differs from exact training evidence')
        native=binding['native_pts']
        rows=self.rows_by_native.get(native,[])
        for row in rows:
            if row['actual_pts']!=binding['actual_pts'] or row['decoded_BGR_sha256']!=digest:
                raise ValueError('Training row belongs to another original exposure')
        for proof in self.negative_by_native.get(native,[]):
            if proof['actual_pts']!=binding['actual_pts'] or proof['decoded_BGR_sha256']!=digest:
                raise ValueError('Negative source evidence belongs to another original exposure')
        result=self.supervision_catalog(native,human_annotations=human_annotations)
        if self.semantic_feedback is not None:
            result['semantic_annotations']=self.semantic_feedback.frame_supervision(bgr,binding,
                shot_plan=self.shot_plan,human_annotations=human_annotations)
        return result


def strict_negative_point_loss(logits, negatives, classes, *, origin_xy=(0,0),human_annotations=()):
    """Source role-absence loss at exact UV; no replacement positive label."""
    import torch
    from torch.nn import functional as F
    losses=[];count=0;h,w=logits.shape[-2:]
    def bilinear_support(point):
        low=np.floor(point).astype(int);fraction=np.asarray(point)-low
        return {(int(low[0]+dx),int(low[1]+dy)) for dx,dy in ((0,0),(0,1),(1,0),(1,1))
            if (fraction[0] if dx else 1-fraction[0])*(fraction[1] if dy else 1-fraction[1])>0}
    for proof in negatives:
        uv=np.asarray(proof['uv'],float)-origin_xy
        if not (0<=uv[0]<w and 0<=uv[1]<h):continue
        excluded=proof['excluded_kinds']
        if not excluded or len(set(excluded))!=len(excluded) or len(excluded)>=len(classes):
            raise ValueError('Reviewed role absence must leave an unspecified possible semantic role')
        for human in human_annotations:
            if human.get('state')!='user_confirmed_visual_supervision' or human.get('kind') not in excluded:continue
            positions=np.asarray(human['uv'],float).reshape(-1,2)
            protected=set().union(*(bilinear_support(p) for p in positions)) if len(positions) else set()
            if bilinear_support(uv)&protected:
                raise ValueError('Negative source evidence conflicts with direct human pixels')
        point=torch.as_tensor(uv,device=logits.device,dtype=torch.float32)
        grid=point*point.new_tensor([2/(w-1),2/(h-1)])-1
        value=F.grid_sample(logits.float(),grid[None,None,None],align_corners=True)[0,:,0,0]
        allowed=[i for i,kind in enumerate(classes) if kind not in excluded]
        losses.append(torch.logsumexp(value,dim=0)-torch.logsumexp(value[allowed],dim=0));count+=1
    return (torch.stack(losses).mean() if losses else logits.sum()*0),count


class StrictSourceStreamingTrainer:
    """Use each shared source frame once; train crop batches with existing losses."""
    def __init__(self,network,optimizer,inputs,*,classes,device,crop_size,human_position_weight=1.,primitive_weight=.2,
            position_rank_weight=0.,guarded_head_support=None,original_source_replay=False):
        self.network,self.optimizer,self.inputs=network,optimizer,inputs
        self.classes,self.device,self.crop_size=classes,device,int(crop_size)
        self.human_position_weight,self.primitive_weight=human_position_weight,primitive_weight
        self.position_rank_weight=position_rank_weight
        self.guarded_head_support=guarded_head_support
        self.original_source_replay=bool(original_source_replay)
        self.successful_steps=0
        self.source_crops_consumed=0

    def consume_frame(self,bgr,binding,*,shot_role,human_annotations=(),max_optimizer_steps=None,crop_origins=None):
        import torch
        from .vision import (_tensor,_annotations_in_crop,interval_supervision,primitive_supervision)
        from .vision_supervision import confirmed_position_interval_localization_loss
        supervision=self.inputs.frame_supervision(bgr,binding,human_annotations=human_annotations)
        if shot_role!='train':
            return dict(optimizer_steps=0,reason='whole_shot_not_training')
        annotations,negatives=supervision['annotations'],supervision['negatives']
        semantic=supervision.get('semantic_annotations',[])
        if not annotations and not negatives and not semantic:
            return dict(optimizer_steps=0,reason='no_authoritative_source_support')
        rgb=np.ascontiguousarray(bgr[...,::-1]);h,w=rgb.shape[:2];size=self.crop_size
        tiles=set()
        locations=([p for a in annotations for p in a['uv']]+[r['uv'] for r in negatives]
            +[p for a in semantic for p in a['uv']])
        for x,y in locations:
            tiles.add((min(int(x)//size*size,max(0,w-size)),min(int(y)//size*size,max(0,h-size))))
        if crop_origins is not None:
            selected_tiles={tuple(map(int,point)) for point in crop_origins}
            if not selected_tiles.issubset(tiles):
                raise ValueError('Scheduled crop has no authoritative source target')
            tiles=selected_tiles
        if not tiles:return dict(optimizer_steps=0,reason='no_scheduled_source_crops')
        images=_tensor(rgb[None],self.device)
        amp=torch.device(self.device).type=='cuda'
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
            context=self.network.encode_context(images)
        source_sample=dict(annotations=annotations,source_hash=binding.get('source_sha256',binding.get('video_sha256')),
            source_pts=int(binding['native_pts']),time_base=binding['time_base'])
        stats=[]
        for x,y in sorted(tiles,key=lambda point:(point[1],point[0])):
            if max_optimizer_steps is not None and len(stats)>=max_optimizer_steps:break
            crop=rgb[y:y+size,x:x+size];ch,cw=crop.shape[:2]
            self.source_crops_consumed+=1
            selected=_annotations_in_crop(source_sample,(x,y),(ch,cw))
            self.optimizer.zero_grad(set_to_none=True)
            refined=[]
            hook=(self.network.semantic_head.register_forward_pre_hook(
                lambda module,args:refined.append(args[0].detach()))
                if self.guarded_head_support is not None else None)
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False,enabled=amp):
                output=self.network(_tensor(crop[None],self.device),context_features=context,
                    context_crop=(x,y,cw,ch),context_native_hw=(h,w))
                if hook is not None:hook.remove()
                logits=output['logits'][:,0]
                positive,details=interval_supervision(logits,selected,self.classes)
                human,_=confirmed_position_interval_localization_loss(logits,selected,self.classes,
                    original_sample=source_sample,crop_origin_xy=(x,y),rank_weight=self.position_rank_weight)
                primitive=primitive_supervision(output['primitive_logits'][:,0],selected)
                negative,negative_count=strict_negative_point_loss(logits,negatives,self.classes,
                    origin_xy=(x,y),human_annotations=selected)
                from .source_semantic_feedback import known_source_semantic_point_loss
                known_role,known_role_count=known_source_semantic_point_loss(logits,semantic,self.classes,
                    source_binding=binding,origin_xy=(x,y),human_annotations=human_annotations)
                loss=positive+self.human_position_weight*human+self.primitive_weight*primitive+negative+known_role
                if self.original_source_replay:
                    from .vision_head_guard import native_cuda_original_supervision_loss
                    replay=native_cuda_original_supervision_loss(
                        [cache for cache in self.guarded_head_support if cache['whole_shot_role']=='train'],
                        self.network.semantic_head.weight,self.network.semantic_head.bias)
                    loss=loss+self.human_position_weight*replay
            if not details and not negative_count and not known_role_count:continue
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite strict source training loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.network.parameters(),5.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite strict source training gradient')
            applied_scale=1.
            if self.guarded_head_support is None:
                self.optimizer.step()
            else:
                from .vision_head_guard import (adamw_head_proposal,guarded_head_backtrack,
                    commit_guarded_head_proposal)
                if len(refined)!=1:raise ValueError('Guarded correction needs this actual crop feature field')
                def actual_source_objective(weight,bias):
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                        scores=torch.nn.functional.conv2d(refined[0],weight,bias)
                        positive,details=interval_supervision(scores,selected,self.classes)
                        human,_=confirmed_position_interval_localization_loss(scores,selected,self.classes,
                            original_sample=source_sample,crop_origin_xy=(x,y),rank_weight=self.position_rank_weight)
                        negative,negative_count=strict_negative_point_loss(scores,negatives,self.classes,
                            origin_xy=(x,y),human_annotations=selected)
                        known_role,known_role_count=known_source_semantic_point_loss(scores,semantic,self.classes,
                            source_binding=binding,origin_xy=(x,y),human_annotations=human_annotations)
                        value=positive+self.human_position_weight*human+negative+known_role
                    return dict(loss=float(value),sample_count=sum(d['sample_count'] for d in details)+negative_count+known_role_count,
                        source_key=[source_sample['source_hash'],source_sample['time_base'],source_sample['source_pts']])
                proposal=adamw_head_proposal(self.network.semantic_head,self.optimizer)
                decision=guarded_head_backtrack(self.guarded_head_support,proposal,
                    expected_source_keys=[cache['source_key'] for cache in self.guarded_head_support],
                    localization_evaluator=actual_source_objective)
                if not decision['accepted']:
                    stats.append(dict(crop_xy=[x,y],optimizer_step_committed=False,
                        reason='no_source_improvement_preserving_original_confirmed_support'))
                    continue
                commit_guarded_head_proposal(self.network.semantic_head,self.optimizer,proposal,decision)
                applied_scale=decision['scale']
            self.successful_steps+=1
            stats.append(dict(crop_xy=[x,y],positive_points=sum(d['sample_count'] for d in details),
                negative_points=negative_count,known_semantic_points=known_role_count,
                loss=float(loss.detach()),optimizer_step_committed=True,
                applied_head_delta_scale=float(applied_scale)))
        return dict(optimizer_steps=sum(row.get('optimizer_step_committed',True) for row in stats),
            actual_pts=int(binding['actual_pts']),native_pts=int(binding['native_pts']),
            crop_batches=stats,temporal_prediction_used_as_position=False)


def train_strict_source_stream(config_path,*,steps=None,resume=False,candidate=False,resume_checkpoint=None,
        retention_guard_every=None,update_scope=None,learning_rate=None,position_rank_weight=None,retention_reference=None):
    """Explicit vision.train hook; one seek then sequential exact-frame decoding."""
    import av
    import torch
    from .vision import (configure,load_training_manifest,_image,write_json,_tensor,_annotations_in_crop,
        source_reviewed_region_masks,validate_training_sources)
    from .vision_network import RinkVisionNet
    from .vision_supervision import source_sample_key
    from .vision_retention import source_retention_snapshot,source_retention_regressions
    from .strict_source_training_schedule import source_crop_schedule
    config,data=configure(config_path);spec=config['strict_source_stream_training']
    torch.set_num_threads(int(spec.get('cpu_threads',1)))
    import cv2
    cv2.setNumThreads(int(spec.get('cpu_threads',1)))
    if not torch.cuda.is_available():raise RuntimeError('Configured GPU is required for strict source training')
    if not resume:raise ValueError('Strict source correction resumes the current trained model; no silent head reset')
    checkpoint=Path(resume_checkpoint or spec['resume_checkpoint']).resolve()
    checkpoint_sha256=file_sha(checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved.get('classes') is not None and saved['classes']!=config['classes']:
        raise ValueError('Strict source correction cannot change the trained semantic vocabulary')
    network=RinkVisionNet(**saved['model_config']).to(config['device'])
    network.load_state_dict(saved['state_dict'],strict=True)
    actual_scope=update_scope or config.get('training_update_scope','all_trainable')
    if actual_scope not in ('all_trainable','semantic_head'):raise ValueError('Unknown vision update scope')
    optimizer=torch.optim.AdamW([p for p in network.parameters() if p.requires_grad],
        lr=learning_rate or config['learning_rate'],weight_decay=1e-4)
    if saved.get('optimizer') is not None:optimizer.load_state_dict(saved['optimizer'])
    if learning_rate is not None:
        for group in optimizer.param_groups:group['lr']=float(learning_rate)
    if actual_scope=='semantic_head':
        for name,parameter in network.named_parameters():parameter.requires_grad_(name.startswith('semantic_head.'))
    guarded_head=spec.get('guarded_semantic_head_updates',False)
    if guarded_head and actual_scope!='semantic_head':
        raise ValueError('Frozen source feature guards require only the existing semantic head to update')
    manifest,plan=load_training_manifest({k:v for k,v in config.items() if k!='strict_source_stream_training'},data)
    validate_training_sources(manifest)
    inputs=StrictSourceTrainingInputs(spec['absolute_inputs'],negative_reference=spec.get('negative_inputs'),
        semantic_reference=spec.get('semantic_inputs'),shot_plan=plan,classes=config['classes'])
    if spec.get('native_semantic_source_transactions'):
        if actual_scope!='all_trainable':
            raise ValueError('Native feature correction requires the actual all-trainable update scope')
        from .vision_native_semantic_correction import train_native_semantic_correction
        return train_native_semantic_correction(config=config,network=network,optimizer=optimizer,saved=saved,
            inputs=inputs,manifest=manifest,plan=plan,checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,steps=steps)
    human_by_native={}
    source=inputs.pack.metadata['source']
    for sample in manifest['samples']:
        if source_sample_key(sample)[:2]!=(source['video_sha256'],str(Fraction(source['time_base']))):continue
        human_by_native[int(sample['source_pts'])]=sample
    frame_bindings=inputs.frame_bindings
    if not frame_bindings:raise ValueError('Strict source pack has no admitted observations; pending cannot train as truth')
    trainer=StrictSourceStreamingTrainer(network,optimizer,inputs,classes=config['classes'],device=config['device'],
        crop_size=config['crop_size'],human_position_weight=config.get('human_position_localization_weight',1.),
        primitive_weight=config.get('primitive_supervision_weight',.2),
        position_rank_weight=position_rank_weight if position_rank_weight is not None else config.get('human_position_rank_weight',0.))
    wanted_steps=steps if steps is not None else config.get('training_steps')
    run_data=Path(config.get('training_run_directory',data)).resolve()
    if not run_data.is_relative_to(data):raise ValueError('Strict training artifacts must stay in configured Data')
    run_data.mkdir(parents=True,exist_ok=True)
    source_path=next(iter(frame_bindings.values()))['source_path']
    authenticate_source_reference(dict(path=source_path,sha256=source['video_sha256']))
    start,stop=min(frame_bindings),max(frame_bindings)
    def native_shot_role(native_pts):
        binding=frame_bindings[int(native_pts)]
        matches=[r for r in plan['shots'] if r['source_sha256']==source['video_sha256']
            and Fraction(r['time_base'])==Fraction(source['time_base'])
            and r['id']==binding['shot_id']]
        if not matches:return 'unlisted_not_training'
        if len(matches)!=1:raise ValueError('Training source frame needs one verified complete native shot')
        if matches[0].get('whole_shot') is not True:raise ValueError('Framewise training splits are forbidden')
        return matches[0]['split']
    for native in frame_bindings:native_shot_role(native)
    catalog=[];semantic_catalog=[]
    for native in sorted(frame_bindings):
        if native_shot_role(native)!='train':continue
        human=human_by_native.get(native)
        supervision=inputs.supervision_catalog(native,human_annotations=human['annotations'] if human else ())
        catalog.append((native,supervision['annotations'],supervision['negatives']))
        semantic_catalog.append((native,supervision['semantic_annotations']))
    schedule,schedule_receipt=source_crop_schedule(catalog,crop_size=config['crop_size'],
        native_hw=(1080,1920),optimizer_steps=wanted_steps,semantic_catalog=semantic_catalog)
    if not schedule:raise ValueError('No source crops have an authoritative whole-shot training role')
    start,stop=min(schedule),max(schedule)
    guards=[s for s in manifest['samples'] if s.get('regions') or any(
        a.get('state')=='user_confirmed_visual_supervision' for a in s['annotations'])]
    if not guards:raise ValueError('Source correction needs original human/source region retention evidence')
    if retention_reference is not None:
        raise ValueError('The original guard reference includes a different temporal-forward protocol; use this branch source-bound baseline')
    cadence=retention_guard_every if retention_guard_every is not None else config.get('source_retention_guard_every',20)
    if isinstance(cadence,bool) or not isinstance(cadence,int) or cadence<1:
        raise ValueError('Source retention guard cadence must remain explicit and positive')
    head_support=[]
    def retention_snapshots(*,capture_head_support=False):
        network.eval();snapshots=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for sample in guards:
                rgb=_image(sample['frame_path'],sample.get('frame_key'));h,w=rgb.shape[:2]
                refined=[]
                hook=(network.semantic_head.register_forward_pre_hook(
                    lambda module,args:refined.append(args[0].detach())) if capture_head_support else None)
                output=network(_tensor(rgb[None],config['device']))
                if hook is not None:hook.remove()
                annotations=_annotations_in_crop(sample,(0,0),(h,w))
                regions=[(dict(region,region_id=region.get('region_id',str(i))),mask)
                    for i,(region,mask) in enumerate(source_reviewed_region_masks(sample,(0,0),(h,w),config['device']))]
                snapshot=source_retention_snapshot(output['logits'][:,0].float().softmax(1),
                    output['primitive_logits'][:,0].float().softmax(1),
                    source_binding=dict(source_sha256=sample['source_hash'],source_pts=sample['source_pts'],
                        time_base=sample['time_base'],shot_id=sample['source_shot_scope']['shot_id'],native_hw=[h,w]),
                    annotations=annotations,classes=config['classes'],reviewed_regions=regions)
                snapshots.append(snapshot)
                if capture_head_support:
                    from .vision_head_guard import build_head_support_cache
                    if len(refined)!=1:raise ValueError('Original source guard lost its actual native features')
                    support=build_head_support_cache(refined[0],output['logits'][:,0].float().softmax(1),
                        sample=sample,source_binding=snapshot['source_binding'],classes=config['classes'],
                        reference_weight=network.semantic_head.weight,reference_bias=network.semantic_head.bias,
                        reference_snapshot=snapshot,reviewed_regions=regions,retain_native_cuda_features=True)
                    if not support['native_GPU_baseline_decisions_verified']:
                        raise ValueError('Frozen source head guard does not reproduce the actual native baseline')
                    support['whole_shot_role']=sample['source_shot_scope']['split']
                    head_support.append(support)
                del output
        network.train();return snapshots
    baseline=retention_snapshots(capture_head_support=guarded_head);guard_failed=False;last_guard=0
    if guarded_head:trainer.guarded_head_support=head_support
    trainer.original_source_replay=spec.get('original_source_replay_during_head_update',False)
    if trainer.original_source_replay and not guarded_head:
        raise ValueError('Frozen original-source replay requires guarded semantic-head updates')
    comparisons=[]
    run_id=time.time_ns();consumption_path=run_data/f'strict_source_training_consumption_{run_id}.jsonl'
    consumed=0;network.train()
    def check_guard():
        return [source_retention_regressions(current,prior)
            for current,prior in zip(retention_snapshots(),baseline)]
    with av.open(source_path) as container:
        stream=container.streams.video[0]
        stream.thread_count=int(spec.get('cpu_threads',1))
        if Fraction(stream.time_base)!=Fraction(source['time_base']):raise ValueError('Source training native clock differs')
        container.seek(start,stream=stream,backward=True)
        for frame in container.decode(stream):
            if frame.pts is None or int(frame.pts)<start:continue
            if int(frame.pts)>stop:break
            item=frame_bindings.get(int(frame.pts))
            if item is None or int(frame.pts) not in schedule:continue
            bgr=frame.to_ndarray(format='bgr24')
            binding=dict(source_sha256=source['video_sha256'],source_path=source_path,time_base=source['time_base'],
                native_pts=int(frame.pts),actual_pts=item['actual_pts'],decoded_native_BGR_sha256=item['decoded_BGR_sha256'],
                shot_id=item['shot_id'])
            human=human_by_native.get(int(frame.pts));annotations=human['annotations'] if human else ()
            if human and not np.array_equal(bgr[...,::-1],_image(human['frame_path'],human.get('frame_key'))):
                raise ValueError('Human source frame differs from shared exact video frame')
            result=trainer.consume_frame(bgr,binding,shot_role=native_shot_role(int(frame.pts)),
                human_annotations=annotations,crop_origins=schedule[int(frame.pts)],
                max_optimizer_steps=None if wanted_steps is None
                    else max(0,wanted_steps-trainer.successful_steps))
            if result.get('crop_batches'):
                consumed+=1
                with consumption_path.open('a',encoding='utf8') as ledger:
                    ledger.write(json.dumps(result,separators=(',',':'))+'\n')
            if trainer.successful_steps-last_guard>=cadence:
                comparisons=check_guard();last_guard=trainer.successful_steps
                guard_failed=any(not r['retained_without_new_classification_errors'] for r in comparisons)
                if guard_failed:break
            if wanted_steps is not None and trainer.successful_steps>=wanted_steps:break
    if not trainer.successful_steps:raise ValueError('No current source rows had authoritative training support in training shots')
    authenticate_source_reference(inputs.authority)
    if spec.get('negative_inputs'):authenticate_source_reference(spec['negative_inputs'])
    if inputs.semantic_feedback is not None:inputs.semantic_feedback.verify_evidence_unchanged()
    if last_guard!=trainer.successful_steps:
        comparisons=check_guard()
        guard_failed=any(not r['retained_without_new_classification_errors'] for r in comparisons)
    output=run_data/f'checkpoint_strict_source_candidate_{run_id}.pt'
    torch.save(dict(saved,model_config=network.model_config,state_dict=network.state_dict(),optimizer=optimizer.state_dict(),
        step=int(saved.get('step',0))+trainer.successful_steps,strict_source_supervision=spec,
        strict_source_training_update_scope=actual_scope,strict_source_human_retention_failed=guard_failed,
        strict_source_resume_checkpoint_sha256=checkpoint_sha256,
        strict_source_previous_optimizer_state_available=saved.get('optimizer') is not None),output)
    if file_sha(checkpoint)!=checkpoint_sha256:
        raise ValueError('The original resumed vision checkpoint changed during source correction')
    report=dict(schema='rinkspace.strict-source-stream-training-consumption.v1',checkpoint=str(output),
        source_supervision=spec,successful_optimizer_steps=trainer.successful_steps,frames_consumed=consumed,
        actual_source_consumption=dict(path=str(consumption_path),sha256=file_sha(consumption_path)),
        human_retention_failed=guard_failed,status=('source_retention_failed' if guard_failed else
            'candidate_trained' if wanted_steps is None or trainer.successful_steps>=wanted_steps else
            'source_stream_consumed_partial_parameter_updates'),
        actual_source_crops_consumed=trainer.source_crops_consumed,
        all_scheduled_source_crops_consumed=trainer.source_crops_consumed==schedule_receipt['scheduled_crop_count'],
        requested_optimizer_steps_completed=wanted_steps is None or trainer.successful_steps>=wanted_steps,
        original_source_retention_comparisons=comparisons,
        source_frames_exported=0,per_frame_seek=False,pending_used_as_truth=False,
        crop_schedule=schedule_receipt,
        resume_checkpoint=dict(path=str(checkpoint),sha256=checkpoint_sha256,
            trained_weights_preserved_at_start=True,
            previous_optimizer_state_available=saved.get('optimizer') is not None),
        full_shot_accuracy_validated=False,candidate_only=True)
    write_json(run_data/f'strict_source_training_consumption_{run_id}.json',report)
    return report
