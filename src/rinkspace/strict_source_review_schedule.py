"""Prioritize explicitly admitted current source corrections in the crop plan."""
from __future__ import annotations

import numpy as np

from .strict_source_training_schedule import source_crop_schedule


def reviewed_source_crop_schedule(catalog, *, crop_size, native_hw, optimizer_steps, reviewed_source_sha256s=()):
    """Keep old entity coverage and every actual crop of the current admission.

    This changes crop selection only. Source labels, loss weights, configured
    optimizer steps and the older schedule implementation remain unchanged.
    """
    all_schedule,record=source_crop_schedule(catalog,crop_size=crop_size,native_hw=native_hw,optimizer_steps=None)
    def keys(schedule):return {(int(n),int(x),int(y)) for n,tiles in schedule.items() for x,y in tiles}
    candidates=keys(all_schedule)
    old_required=max(1,record['required_source_entity_count']+record['mandatory_actual_negative_crop_count']+
        record['mandatory_known_semantic_crop_count'])
    old_schedule,_=source_crop_schedule(catalog,crop_size=crop_size,native_hw=native_hw,optimizer_steps=old_required)
    mandatory=keys(old_schedule);reviewed=set();height,width=map(int,native_hw);size=int(crop_size)
    admitted=set(reviewed_source_sha256s)
    for native,annotations,_ in catalog:
        for annotation in annotations:
            authority=annotation.get('authority')
            if not isinstance(authority,dict) or authority.get('sha256') not in admitted:continue
            for x,y in np.asarray(annotation['uv'],float):
                reviewed.add((int(native),min(int(x)//size*size,max(0,width-size)),
                    min(int(y)//size*size,max(0,height-size))))
    if not reviewed.issubset(candidates):raise ValueError('A current correction crop has no authoritative source location')
    mandatory.update(reviewed)
    if len(mandatory)>optimizer_steps:
        raise ValueError('Existing configured steps do not cover current reviewed corrections and original source entities')
    selected=set(mandatory);remaining=sorted(candidates-mandatory)
    count=min(int(optimizer_steps)-len(selected),len(remaining))
    if count:
        selected.update(remaining[int(i)] for i in np.rint(np.linspace(0,len(remaining)-1,count)).astype(int))
    schedule={}
    for native,x,y in sorted(selected):schedule.setdefault(native,[]).append((x,y))
    record.update(scheduled_crop_count=len(selected),source_stream_coverage=[min(schedule),max(schedule)] if schedule else [],
        mandatory_current_review_crop_count=len(reviewed),all_current_review_crops_scheduled=reviewed.issubset(selected),
        current_review_source_sha256s=sorted(admitted),original_entity_coverage_preserved=True,
        configured_optimizer_steps_unchanged=True)
    return schedule,record
