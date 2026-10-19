"""Archive complete uncertain source estimators without slicing correlated W.

This is an explicit source change, never residual-based outlier rejection.
The caller authenticates its reviewed manifest before supplying these exact
native addresses and original operator hashes. Every retained estimator and
ordinary component keeps its original likelihood, order and scale.
"""
from copy import deepcopy
import hashlib

import numpy as np


def _sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def quarantine_complete_groups(frames, specifications, *, evidence_reference):
    from .first_shot_fit import FrozenWhitening
    specs = {}
    for item in specifications:
        key = int(item['native_pts']), tuple(map(str, item['source_point_ids']))
        if (key in specs or not key[1] or len(set(key[1])) != len(key[1])
                or not item.get('reason') or not item.get('evidence_references')
                or item.get('material_identity_confirmed') is not False):
            raise ValueError('Distinct reviewed uncertain complete source groups required')
        specs[key] = item
    if not specs:
        raise ValueError('An empty source quarantine is not a source transition')
    result, receipts, retained_formal, used = [], [], [], set()
    formal_offset = point_count = 0
    for frame in frames:
        old_w = FrozenWhitening(frame)
        flattened = [(row, j) for row in frame['rows'] for j in range(len(row['uv']))]
        ids = [str(row['source_point_ids'][j]) for row, j in flattened]
        keep = np.ones(len(ids), bool)
        dropped = set()
        archive = []
        for gi, group in enumerate(frame['whitening_groups']):
            indices = np.asarray(group['point_indices'], int)
            group_ids = tuple(ids[i] for i in indices)
            key = int(frame['native_pts']), group_ids
            if key not in specs:
                continue
            item = specs[key]
            matrix = group['operator'].matrix
            if (_sha(matrix) != item['operator_matrix_sha256']
                    or list(matrix.shape) != item['operator_matrix_shape']):
                raise ValueError('Quarantine does not match the original complete W')
            if not keep[indices].all():
                raise ValueError('Source estimators overlap in a quarantine')
            keep[indices] = False
            dropped.add(gi)
            used.add(key)
            archive.append(dict(native_pts=int(frame['native_pts']),
                source_point_ids=list(group_ids), original_complete_group=group,
                original_rows=[dict(source_point_id=ids[i], UV=flattened[i][0]['uv'][flattened[i][1]].copy(),
                    sigma=float(flattened[i][0]['sigma'][flattened[i][1]]),
                    original_target=deepcopy(flattened[i][0]['target'])) for i in indices],
                review=deepcopy(item), source_transition=evidence_reference,
                excluded_from_FIT=True, original_source_evidence_modified=False))
        if not dropped:
            result.append(frame)
            retained_formal.extend(range(formal_offset, formal_offset + old_w.count))
            formal_offset += old_w.count
            point_count += len(ids)
            continue
        mapping = np.full(len(ids), -1, int)
        mapping[keep] = np.arange(int(keep.sum()))
        rows, cursor = [], 0
        for row in frame['rows']:
            mask = keep[cursor:cursor + len(row['uv'])]
            # A reviewed estimator must own complete rows. Do not implicitly
            # split native queries, finite supports or covariance estimators.
            if mask.any() and not mask.all():
                raise ValueError('Complete group quarantine cannot split a source row')
            if mask.all():
                rows.append(row)
            cursor += len(row['uv'])
        groups = []
        selected_formal = []
        for gi, (group, (_, matrix, dest)) in enumerate(zip(frame['whitening_groups'], old_w.groups, strict=True)):
            if gi in dropped:
                continue
            indices = np.asarray(group['point_indices'], int)
            if not keep[indices].all():
                raise ValueError('Quarantine intersects an unreviewed correlated estimator')
            groups.append(dict(group, point_indices=mapping[indices].tolist()))
            selected_formal.extend(range(dest.start, dest.stop))
        ordinary_keep = np.repeat(keep, 2)[old_w.ordinary]
        selected_formal.extend((old_w.ordinary_start + np.flatnonzero(ordinary_keep)).tolist())
        new = dict(frame, rows=rows, whitening_groups=groups, source_point_count=int(keep.sum()),
            inactive_source_groups=list(frame.get('inactive_source_groups', [])) + archive,
            complete_group_quarantine_revision=evidence_reference)
        new_w = FrozenWhitening(new)
        selected_formal = np.asarray(selected_formal, int)
        if (len(selected_formal) != new_w.count
                or not np.array_equal(old_w.scales[selected_formal], new_w.scales)
                or not np.array_equal(old_w.ordinary_scales[ordinary_keep], new_w.ordinary_scales)):
            raise ValueError('Quarantine changed retained formal scales/order')
        raw_mapping = np.full(2*len(ids), -1, int)
        raw_mapping[np.repeat(keep, 2)] = np.arange(2*int(keep.sum()))
        retained_old_groups = [g for gi, g in enumerate(old_w.groups) if gi not in dropped]
        for (old_indices, A, old_dest), (new_indices, B, new_dest) in zip(retained_old_groups, new_w.groups, strict=True):
            if not np.array_equal(A, B) or not np.array_equal(raw_mapping[old_indices], new_indices):
                raise ValueError('Quarantine altered a retained correlated W or membership order')
        if not np.array_equal(raw_mapping[old_w.ordinary[ordinary_keep]], new_w.ordinary):
            raise ValueError('Quarantine changed retained ordinary components')
        retained_formal.extend((formal_offset + selected_formal).tolist())
        receipts.append(dict(frame=int(frame['index']), native_pts=int(frame['native_pts']),
            removed_complete_groups=len(dropped), removed_source_points=int((~keep).sum()),
            removed_formal_rows=old_w.count-new_w.count,
            retained_UV_ID_target_sigma_and_rows_same_objects=True,
            retained_correlated_W_values_and_member_order_byte_exact=True,
            no_correlated_matrix_rows_or_columns_sliced=True))
        result.append(new)
        formal_offset += old_w.count
        point_count += int(keep.sum())
    if used != set(specs):
        raise ValueError('Reviewed complete group is absent from this exact native inventory')
    indices = np.asarray(retained_formal, np.int64)
    return result, dict(native_frames=len(result), source_points=point_count,
        old_formal_rows=formal_offset, formal_rows=len(indices),
        removed_complete_groups=len(used), frames=receipts,
        retained_formal_indices=indices,
        retained_formal_indices_sha256=_sha(indices),
        original_uncertain_estimator_and_evidence_archived=True,
        camera_or_scene_parameters_modified=False)
