"""Distribute the existing optimizer budget over the actual correction stream."""
from __future__ import annotations

import numpy as np


def source_crop_schedule(catalog, *, crop_size, native_hw, optimizer_steps=None,semantic_catalog=()):
    """Keep real negative corrections and each source entity in the schedule.

    Taking the first configured number of crops would exhaust the budget before
    later user corrections. Candidate crops here come only from authoritative
    source locations. Remaining slots span the complete ordered source stream.
    """
    height, width = map(int, native_hw)
    size = int(crop_size)
    if size <= 0:
        raise ValueError('Positive source crop size required')
    candidates, mandatory, entities = set(), set(), set()
    negative_keys = set()
    semantic_keys = set()
    def tile(point):
        x, y = np.asarray(point, float)
        if not np.isfinite([x,y]).all() or not (0 <= x < width and 0 <= y < height):
            raise ValueError('Training crop source point is outside its native image')
        return (min(int(x)//size*size, max(0,width-size)),
                min(int(y)//size*size, max(0,height-size)))
    for native, annotations, negatives in sorted(catalog, key=lambda row:row[0]):
        for annotation in annotations:
            if annotation.get('state') not in ('source_reviewed_interval','user_confirmed_visual_supervision'):
                continue
            locations = np.asarray(annotation['uv'], float).reshape(-1,2)
            keys = {(int(native), *tile(point)) for point in locations}
            candidates.update(keys)
            entity = (annotation.get('source_curve_id') or annotation.get('physical_material_id')
                or annotation.get('feature_id'), annotation.get('kind'), annotation.get('boundary_role'))
            if keys and entity not in entities:
                mandatory.add(min(keys))
                entities.add(entity)
        for proof in negatives:
            key = (int(native), *tile(proof['uv']))
            candidates.add(key)
            negative_keys.add(key)
    mandatory.update(negative_keys)
    for native, annotations in semantic_catalog:
        for annotation in annotations:
            if (annotation.get('state')!='human_confirmed_semantic_role_at_existing_source_uv'
                    or annotation.get('semantic_supervision_only') is not True
                    or annotation.get('position_supervision_allowed') is not False
                    or annotation.get('fitting_admissible') is not False
                    or annotation.get('vision_training_role')!='train'):
                raise ValueError('Only authenticated known training-source roles can require semantic crops')
            keys={(int(native),*tile(point)) for point in annotation['uv']}
            candidates.update(keys);semantic_keys.update(keys)
    mandatory.update(semantic_keys)
    if optimizer_steps is None:
        selected = candidates
    else:
        if isinstance(optimizer_steps,bool) or int(optimizer_steps) != optimizer_steps or optimizer_steps <= 0:
            raise ValueError('Positive configured optimizer-step budget required')
        if len(mandatory) > optimizer_steps:
            raise ValueError('Configured training steps cannot cover all actual corrections and source entities')
        selected = set(mandatory)
        remaining = sorted(candidates-mandatory)
        count = min(int(optimizer_steps)-len(selected), len(remaining))
        if count:
            for index in np.rint(np.linspace(0,len(remaining)-1,count)).astype(int):
                selected.add(remaining[int(index)])
    by_native = {}
    for native, x, y in sorted(selected):
        by_native.setdefault(native, []).append((x,y))
    return by_native, dict(candidate_crop_count=len(candidates), scheduled_crop_count=len(selected),
        required_source_entity_count=len(entities), mandatory_actual_negative_crop_count=len(negative_keys),
        all_actual_negative_crops_scheduled=negative_keys.issubset(selected),
        mandatory_known_semantic_crop_count=len(semantic_keys),
        all_known_semantic_crops_scheduled=semantic_keys.issubset(selected),
        source_stream_coverage=[min(by_native),max(by_native)] if by_native else [],
        initial_prefix_only_training=False)
