"""Exact native-frame foreground/material exclusions, independent of person masks.

These rectangles reject source support; their coordinates are never observations.
No record is inferred from a model residual or a zero person-mask pixel.
"""
from __future__ import annotations
import numpy as np


def source_records(source, records):
    result = []
    for record in records:
        if int(record['native_pts']) != int(source['native_pts']):
            continue
        for key in ('time_base', 'source_video_sha256', 'png_sha256', 'mask_sha256'):
            if record.get(key) != source.get(key):
                raise ValueError('Static-material exclusion source binding changed: ' + key)
        if int(record['programme_frame_index_1based']) != int(source['programme_frame_index_1based']):
            raise ValueError('Static-material exclusion frame index changed')
        if int(record['shot_index']) != int(source.get('shot_index', source.get('shot_id'))):
            raise ValueError('Static-material exclusion shot changed')
        if record.get('scope') != 'all_static_ice_material':
            raise ValueError('Unsupported static-material exclusion scope')
        if not record.get('source_roi_is_not_observation') or not record.get('current_native_source_visually_confirmed'):
            raise ValueError('Unconfirmed static-material source exclusion')
        box = np.asarray(record['source_roi_xyxy'], dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or not ((box[:2] >= 0).all() and (box[2:] <= [1920, 1080]).all() and (box[2:] > box[:2]).all()):
            raise ValueError('Invalid native static-material exclusion rectangle')
        if not (box == np.rint(box)).all():
            raise ValueError('Native exclusion rectangle requires pixel bounds')
        result.append(record)
    return result


def exclusion_mask(records, native_hw=(1080, 1920)):
    mask = np.zeros(native_hw, dtype=bool)
    for record in records:
        x0, y0, x1, y1 = map(int, record['source_roi_xyxy'])
        mask[y0:y1, x0:x1] = True
    return mask


def _segment_hits(a, b, box):
    lo, hi = 0.0, 1.0
    delta = b - a
    for axis in range(2):
        if delta[axis] == 0:
            if a[axis] < box[axis] or a[axis] >= box[axis+2]:
                return False
        else:
            first, last = sorted(((box[axis]-a[axis])/delta[axis], (box[axis+2]-a[axis])/delta[axis]))
            lo, hi = max(lo, first), min(hi, last)
    return lo <= hi


def profile_roi_hits(row, records):
    evidence = row.get('source_evidence') or {}
    points = [np.asarray(row['uv'], dtype=float).reshape(-1, 2)]
    for key in ('query_uv', 'actual_source_sides_uv', 'actual_source_side_quantiles_uv'):
        if key in evidence:
            points.append(np.asarray(evidence[key], dtype=float).reshape(-1, 2))
    points = np.vstack(points)
    if not np.isfinite(points).all():
        raise ValueError('Nonfinite native material support')
    # Recorded contexts lie on their measured profile normal. The entire
    # interval between contexts is support, including the two paint flanks.
    hits = []
    for record in records:
        box = np.asarray(record['source_roi_xyxy'], dtype=float)
        inside = ((points >= box[:2]) & (points < box[2:])).all(axis=1).any()
        crossing = any(_segment_hits(a, b, box) for i, a in enumerate(points) for b in points[i+1:])
        if inside or crossing:
            hits.append(record)
    return hits


def filter_source_readouts(rows, records):
    kept, rejected = [], []
    for row in rows:
        hits = profile_roi_hits(row, records)
        if hits:
            rejected.append(dict(native_readout=row, static_material_exclusion_records=hits,
                                 reason='current_native_foreground_blur_or_unresolved_static_material_support'))
        else:
            kept.append(row)
    return kept, rejected
