"""Replay actual lost source supervision without inventing targets.

An original native retention failure only selects a harder existing label.
Its original UV/index or reviewed mask pixel remains unchanged. Model scores,
unknown masks and failed observations never supply a replacement class.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .vision_instances import same_frame,source_binding


class NativeRetentionHardExamples:
    def __init__(self,*,binding,annotations,classes,reviewed_regions=(),whole_shot_role):
        import torch
        self.binding=source_binding(binding,binding['native_hw'])
        self.classes=tuple(classes);self.whole_shot_role=whole_shot_role
        height,width=self.binding['native_hw']
        self.definitions={};self.weights=defaultdict(dict);self.observed_events=0
        for annotation in annotations:
            if annotation.get('state')!='user_confirmed_visual_supervision':continue
            indices=np.asarray(annotation['source_point_indices'])
            uv=np.asarray(annotation['uv'],np.float64)
            if (indices.dtype.kind not in 'iu' or uv.shape!=(len(indices),2)
                    or len(set(indices.tolist()))!=len(indices) or not np.isfinite(uv).all()
                    or np.any(uv<0) or np.any(uv>=[width,height])):
                raise ValueError('Hard examples must retain exact original visible source indices')
            positions={int(i):p.copy() for i,p in zip(indices,uv)}
            for field,label,names in (
                ('semantic_wrong_source_point_indices',annotation.get('kind'),self.classes),
                ('primitive_wrong_source_point_indices',annotation.get('primitive'),('line','circle','other'))):
                if label not in names:continue
                key=('human_groups',annotation['group_id'],field)
                if key in self.definitions:raise ValueError('Duplicate original hard-example source group')
                self.definitions[key]=dict(domain='original_UV',positions=positions,
                    expected=names.index(label),primitive=field.startswith('primitive'),allowed=None)
        for region,mask in reviewed_regions:
            if (region.get('state') not in (None,'source_reviewed_region') or not region.get('authority')
                    or mask.dtype!=torch.bool or mask.shape!=(1,1,height,width)):
                raise ValueError('Only original reviewed region pixels can become hard examples')
            kind=region.get('kind');excluded=region.get('excluded_kinds',())
            if kind in self.classes and not excluded:
                expected=self.classes.index(kind);allowed=None
            elif kind is None and excluded and all(k in self.classes for k in excluded) and len(excluded)<len(self.classes):
                expected=None;allowed=[i for i,k in enumerate(self.classes) if k not in excluded]
            else:raise ValueError('Original hard region role is missing or competing')
            for field,label in (
                ('semantic_wrong_native_flat_indices',kind),
                ('primitive_wrong_native_flat_indices',region.get('primitive','other') if kind is not None else None)):
                primitive=field.startswith('primitive')
                if primitive and label not in ('line','circle','other'):continue
                key=('reviewed_regions',region['region_id'],field)
                if key in self.definitions:raise ValueError('Duplicate original hard-example source region')
                self.definitions[key]=dict(domain='reviewed_native_pixel',mask=mask.detach(),
                    expected=('line','circle','other').index(label) if primitive else expected,
                    primitive=primitive,allowed=None if primitive else allowed)

    def observe(self,comparison):
        same_frame(self.binding,comparison['source_binding']);added=0
        for finding in comparison['newly_lost_confirmed_support']:
            key=(finding['collection'],finding['identity'],finding['field'])
            if key not in self.definitions:raise ValueError('A lost point has no original known source role')
            definition=self.definitions[key];indices=finding['newly_wrong_original_indices']
            if len(indices)!=len(set(indices)) or any(isinstance(i,bool) or not isinstance(i,int) or i<0 for i in indices):
                raise ValueError('Hard-example indices must preserve the original distinct source locators')
            for index in indices:
                if definition['domain']=='original_UV':
                    if index not in definition['positions']:raise ValueError('Lost human source index is not originally visible')
                elif index>=definition['mask'].numel() or not bool(definition['mask'].flatten()[index]):
                    raise ValueError('Unknown region pixels cannot become hard-example supervision')
                if index not in self.weights[key]:added+=1
                self.weights[key][index]=self.weights[key].get(index,0)+1
                self.observed_events+=1
        return added

    @property
    def point_count(self):return sum(len(weights) for weights in self.weights.values())

    def loss(self,semantic_logits,primitive_logits,*,binding):
        import torch
        from torch.nn import functional as F
        same_frame(self.binding,binding)
        if self.whole_shot_role!='train':raise ValueError('Held whole shots cannot enter hard-example replay loss')
        height,width=self.binding['native_hw'];losses=[]
        if semantic_logits.shape!=(1,len(self.classes),height,width) or primitive_logits.shape!=(1,3,height,width):
            raise ValueError('Hard replay requires these original full native source fields')
        for key,weights in self.weights.items():
            if not weights:continue
            definition=self.definitions[key];indices=sorted(weights)
            logits=primitive_logits if definition['primitive'] else semantic_logits
            if definition['domain']=='original_UV':
                uv=torch.as_tensor(np.asarray([definition['positions'][i] for i in indices]),device=logits.device,dtype=torch.float32)
                grid=uv*uv.new_tensor([2/(width-1),2/(height-1)])-1
                scores=F.grid_sample(logits.float(),grid[None,:,None],align_corners=True)[0,:,:,0].T
            else:
                scores=logits[0].float().flatten(1).T[torch.as_tensor(indices,device=logits.device)]
            if definition['expected'] is not None:
                targets=torch.full((len(indices),),definition['expected'],device=logits.device,dtype=torch.long)
                values=F.cross_entropy(scores,targets,reduction='none')
            else:
                values=torch.logsumexp(scores,dim=1)-torch.logsumexp(scores[:,definition['allowed']],dim=1)
            weight=torch.as_tensor([weights[i] for i in indices],device=logits.device,dtype=torch.float32)
            losses.append((values*weight).mean())
        return torch.stack(losses).mean() if losses else semantic_logits.sum()*0+primitive_logits.sum()*0


def protect_feedback_from_conflicting_original_regions(annotations,reviewed_regions,*,native_hw):
    """Check known feedback's actual bilinear support against old known roles."""
    height,width=native_hw
    for annotation in annotations:
        for point in np.asarray(annotation['uv'],float):
            low=np.floor(point).astype(int);fraction=point-low
            for dx,dy in ((0,0),(0,1),(1,0),(1,1)):
                x,y=low+[dx,dy]
                weight=(fraction[0] if dx else 1-fraction[0])*(fraction[1] if dy else 1-fraction[1])
                if weight<=0 or not (0<=x<width and 0<=y<height):continue
                for region,mask in reviewed_regions:
                    if not bool(mask[0,0,y,x]):continue
                    if (region.get('kind') is not None and region['kind']!=annotation['kind']
                            or annotation['kind'] in region.get('excluded_kinds',())):
                        raise ValueError('Known semantic feedback conflicts with an original reviewed source region')
