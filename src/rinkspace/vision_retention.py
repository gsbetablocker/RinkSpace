"""Compare actual source predictions with existing confirmed supervision.

This is a regression guard, not a pixel-accuracy or independent-validation claim.
Unknown pixels acquire no target. Full native fields and original point indices
remain source-bound; a lower aggregate loss cannot hide a newly lost position.
"""
from __future__ import annotations
import hashlib
import numpy as np


def source_retention_snapshot(semantic_probabilities, primitive_probabilities, *,
        source_binding, annotations, classes, reviewed_regions=()):
    import torch
    from torch.nn import functional as F
    from .vision_instances import source_binding as normalize_binding
    names=tuple(classes)
    semantic=semantic_probabilities
    primitive=primitive_probabilities
    if (semantic.ndim!=4 or semantic.shape[:2]!=(1,len(names)) or
            primitive.shape!=(1,3,*semantic.shape[-2:]) or len(set(names))!=len(names)):
        raise ValueError('Retention checks need actual single-frame native semantic and primitive fields')
    if semantic.device!=primitive.device:
        raise ValueError('Retention probability fields must share one source device')
    for value in (semantic,primitive):
        if (not bool(torch.isfinite(value).all()) or bool((value<0).any()) or
                bool((value>1).any()) or not torch.allclose(value.sum(1),torch.ones_like(value[:,0]),atol=1e-3)):
            raise ValueError('Retention fields must be normalized finite probabilities')
    height,width=semantic.shape[-2:]
    binding=normalize_binding(source_binding,(height,width))
    groups=[]
    for annotation in annotations:
        if annotation.get('state')!='user_confirmed_visual_supervision' or annotation.get('kind') not in names:
            continue
        uv=np.asarray(annotation['uv'],float)
        indices=np.asarray(annotation['source_point_indices'])
        if (uv.shape!=(len(indices),2) or indices.dtype.kind not in 'iu' or
                not np.isfinite(uv).all() or np.any(uv<0) or
                np.any(uv[:,0]>=width) or np.any(uv[:,1]>=height) or
                np.any(indices<0) or np.any(np.diff(indices)<=0)):
            raise ValueError('Retention needs unchanged ordered original visible source indices')
        points=torch.as_tensor(uv,device=semantic.device,dtype=torch.float32)
        grid=points*points.new_tensor([2/(width-1),2/(height-1)])-1
        def labels(field):
            return F.grid_sample(field.float(),grid[None,:,None],align_corners=True)[0,:,:,0].argmax(0).cpu().numpy()
        bad=labels(semantic)!=names.index(annotation['kind'])
        declared=annotation.get('primitive')
        primitive_bad=(labels(primitive)!=('line','circle','other').index(declared)
                       if declared in ('line','circle','other') else None)
        groups.append(dict(group_id=annotation['group_id'],kind=annotation['kind'],primitive=declared,
            source_point_indices=indices.tolist(),semantic_wrong_source_point_indices=indices[bad].tolist(),
            primitive_wrong_source_point_indices=indices[primitive_bad].tolist() if primitive_bad is not None else None,
            human_positions_filtered_by_model_confidence=False))
    if len({row['group_id'] for row in groups})!=len(groups):
        raise ValueError('Retention requires unique confirmed source groups')
    semantic_label=semantic.argmax(1)[0]
    primitive_label=primitive.argmax(1)[0]
    regions=[]
    for region,mask in reviewed_regions:
        if (mask.dtype!=torch.bool or mask.shape!=(1,1,height,width) or mask.device!=semantic.device):
            raise ValueError('Reviewed retention mask must have the full native source domain')
        if not region.get('region_id'):
            raise ValueError('Retention regions need stable source identities')
        expected=region.get('kind');excluded=tuple(region.get('excluded_kinds',()))
        if expected in names and not excluded:
            wrong=semantic_label!=names.index(expected)
        elif excluded and expected is None and len(set(excluded))==len(excluded) and all(x in names for x in excluded):
            wrong=torch.zeros_like(semantic_label,dtype=torch.bool)
            for name in excluded:wrong|=semantic_label==names.index(name)
        else:raise ValueError('A reviewed retention region needs its original positive or reject-only authority')
        declared=region.get('primitive','other') if expected is not None else None
        primitive_wrong=(primitive_label!=('line','circle','other').index(declared)
                         if declared in ('line','circle','other') else None)
        def flat_indices(value):
            return torch.nonzero((value&mask[0,0]).flatten(),as_tuple=False)[:,0].cpu().tolist()
        regions.append(dict(region_id=region['region_id'],kind=expected,excluded_kinds=list(excluded),
            primitive=declared,supervised_pixels=int(mask.sum()),
            native_mask_sha256=hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest(),
            semantic_wrong_native_flat_indices=flat_indices(wrong),
            primitive_wrong_native_flat_indices=flat_indices(primitive_wrong) if primitive_wrong is not None else None,
            unlabelled_pixels_assigned_target=False))
    if len({row['region_id'] for row in regions})!=len(regions):
        raise ValueError('Retention requires unique reviewed region identities')
    return dict(schema='rinkspace.confirmed-source-retention-snapshot.v1',source_binding=binding,
        classes=list(names),human_groups=groups,reviewed_regions=regions,
        original_RGB_measured_boundary_accuracy_accepted=False,independent_validation=False)


def source_retention_regressions(current,baseline):
    from .vision_instances import same_frame
    same_frame(current['source_binding'],baseline['source_binding'])
    if current['classes']!=baseline['classes']:
        raise ValueError('Retention cannot compare changed semantic vocabularies')
    findings=[]
    for collection,identity,wrong_fields,invariants in (
        ('human_groups','group_id',('semantic_wrong_source_point_indices','primitive_wrong_source_point_indices'),
         ('kind','primitive','source_point_indices')),
        ('reviewed_regions','region_id',('semantic_wrong_native_flat_indices','primitive_wrong_native_flat_indices'),
         ('kind','excluded_kinds','primitive','supervised_pixels','native_mask_sha256'))):
        old={row[identity]:row for row in baseline[collection]}
        new={row[identity]:row for row in current[collection]}
        if old.keys()!=new.keys():raise ValueError('Retention comparison cannot lose confirmed supervision groups or regions')
        for key,row in new.items():
            previous=old[key]
            if any(row[field]!=previous[field] for field in invariants):
                raise ValueError('Retention authority, native support or original point indices changed')
            for field in wrong_fields:
                if (row[field] is None)!=(previous[field] is None):
                    raise ValueError('Retention primitive supervision changed')
                if row[field] is None:continue
                lost=sorted(set(row[field])-set(previous[field]))
                if lost:findings.append(dict(collection=collection,identity=key,field=field,
                    newly_wrong_count=len(lost),newly_wrong_original_indices=lost,
                    prior_wrong_count=len(previous[field]),current_wrong_count=len(row[field])))
    return dict(source_binding=current['source_binding'],newly_lost_confirmed_support=findings,
        retained_without_new_classification_errors=not findings,
        physical_boundary_or_independent_accuracy_accepted=False)
