"""Export fixed source labels on their exact source clock, without geometry inference."""
import json

import numpy as np


def apply_source_support(packet, evidence, *, shot_id=None):
    """Update and return a packet using explicit single-foot source evidence.

    ``evidence`` is the fixed list returned by ``source_evidence``. Labels are
    never propagated to neighbours or selected using predicted foot heights.
    Unknown/ambiguous rows leave the incoming contact labels untouched. Source
    phase and all model geometry fields are deliberately outside this mapping.
    """
    pts = np.asarray(packet['pts'])
    if pts.ndim != 1 or pts.dtype.kind not in 'iu' or np.any(np.diff(pts) <= 0):
        raise ValueError('Support export requires unique ordered integer source PTS')
    if 'source_pts' in packet and not np.array_equal(packet['source_pts'], pts):
        raise ValueError('Support export source PTS identity differs')
    time_base = np.asarray(packet['time_base'])
    if time_base.shape != (2,) or time_base.dtype.kind not in 'iu' or np.any(time_base <= 0):
        raise ValueError('Support export requires an explicit integer source time base')
    shape = (len(pts), 2)
    modes = np.asarray(packet.get('skate_contact_mode_mid', np.full(shape, 'unknown')), dtype=str)
    confidence = np.asarray(packet.get('skate_contact_confidence_mid', np.zeros(shape)), float).copy()
    if modes.shape != shape or confidence.shape != shape:
        raise ValueError('Support export contact labels must have one row per source PTS and foot')
    records = list(evidence)
    width = max([len('unknown'), *(len(str(r.get('mode', ''))) for r in records),
                 *(len(v) for v in modes.ravel())])
    modes = modes.astype(f'<U{width}', copy=True)
    source_modes = np.full(shape, 'unknown', dtype=f'<U{width}')
    source_confidence = np.zeros(shape)
    source_ids = np.full(shape, '', dtype=object)
    lookup = {int(p): i for i, p in enumerate(pts)}
    candidates, audit = {}, []
    for row in records:
        item = dict(id=row.get('id'), pts=row.get('pts'), status='unmapped')
        audit.append(item)
        p = row.get('pts')
        if isinstance(p, (bool, np.bool_)) or not isinstance(p, (int, np.integer)):
            raise ValueError('Support evidence PTS must remain an original integer')
        item['pts'] = int(p)
        if 'source_time_base' in row and not np.array_equal(row['source_time_base'], time_base):
            raise ValueError('Support evidence source time base differs')
        if shot_id is not None and row.get('shot_id', shot_id) != shot_id:
            raise ValueError('Support evidence shot identity differs')
        if int(p) not in lookup:
            item['status'] = 'source_exposure_absent';continue
        feet = row.get('feet', [])
        if (len(feet) != 1 or isinstance(feet[0], (bool, np.bool_))
                or not isinstance(feet[0], (int, np.integer)) or feet[0] not in (0, 1)):
            item['status'] = 'ambiguous_physical_foot';continue
        mode = row.get('mode', 'unknown')
        if mode not in ('glide', 'pick', 'airborne'):
            item['status'] = 'unknown_contact_mode';continue
        value = row.get('confidence')
        if (not row.get('source') or 'source_time_base' not in row
                or isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
                or not np.isfinite(value) or not 0 <= value <= 1):
            item['status'] = 'missing_source_label_provenance';continue
        key = (lookup[int(p)], int(feet[0]))
        candidates.setdefault(key, []).append((mode, float(value), row.get('id', ''), item))
    for (i, foot), group in candidates.items():
        if len({v[0] for v in group}) != 1:
            for *_, item in group:item['status'] = 'conflicting_source_modes'
            continue
        mode = group[0][0]
        value = min(v[1] for v in group)
        modes[i, foot] = source_modes[i, foot] = mode
        confidence[i, foot] = source_confidence[i, foot] = value
        source_ids[i, foot] = json.dumps([v[2] for v in group], ensure_ascii=False)
        for *_, item in group:item['status'] = 'mapped_exact_source_exposure'
    packet.update(skate_contact_mode_mid=modes, skate_contact_confidence_mid=confidence,
        source_support_mode_mid=source_modes, source_support_confidence_mid=source_confidence,
        source_support_evidence_ids_mid=np.asarray(source_ids.tolist(), dtype=str),
        source_support_export_json=np.array(json.dumps(dict(
            schema='rinkspace.source-support-export.v1', source_time_base=time_base.tolist(), shot_id=shot_id,
            mapping='explicit source mode and single physical foot at exact integer PTS only',
            duplicate_confidence='minimum fixed source confidence for agreeing labels',
            geometry_used_for_labels=False, source_action_phase_is_contact_evidence=False,
            rows=audit), sort_keys=True)))
    return packet
