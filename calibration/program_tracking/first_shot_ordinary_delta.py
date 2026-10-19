"""Source-addressed ordinary edits while preserving every unrelated native row."""
from copy import deepcopy
import numpy as np
from .first_shot_revision import revise_frames
from .first_shot_source_transition import read, compare_source_frames


def apply_ordinary_delta(frames, revision_reference, binding, *, policy=None):
    revision = read(revision_reference)
    for field in ('added_native_readouts', 'correlated_operator_replacements', 'correlated_estimator_replacements'):
        if revision.get(field):
            raise ValueError('Ordinary identity delta cannot change native readouts or operators')
    changes = {(int(x['native_pts']), str(x['source_point_id'])): x
               for x in revision.get('identity_corrections', [])}
    quarantines = {(int(x['native_pts']), str(x['source_point_id'])): x
                   for x in revision.get('quarantined_correspondences', [])}
    required = set(changes) | set(quarantines)
    if (len(changes) != len(revision.get('identity_corrections', []))
            or len(quarantines) != len(revision.get('quarantined_correspondences', []))
            or set(changes) & set(quarantines)):
        raise ValueError('Duplicate or conflicting ordinary source address')
    selected, touched, found = [], {}, set()
    for frame in frames:
        pts = int(frame['native_pts'])
        grouped = {int(i) for group in frame['whitening_groups'] for i in group['point_indices']}
        rows, indices, cursor = [], set(), 0
        for ri, row in enumerate(frame['rows']):
            n = len(row['uv'])
            addresses = {(pts, str(pid)) for pid in row['source_point_ids']}
            hit = addresses & required
            if hit:
                if (any(i in grouped for i in range(cursor, cursor+n))
                        or len(row.get('native_line_queries', ()))):
                    raise ValueError('Ordinary edit cannot split a correlated or native-query row')
                rows.append(row); indices.add(ri); found.update(hit)
            cursor += n
        if rows:
            selected.append(dict(frame, rows=rows, whitening_groups=[],
                source_point_count=sum(len(row['uv']) for row in rows)))
            touched[pts] = indices
    if found != required:
        raise ValueError('Ordinary delta refers to a missing active source address')
    changed, _, proof = revise_frames(selected, revision_reference, binding, policy=policy)
    changed = {int(f['native_pts']): f for f in changed}
    result = []
    for original in frames:
        pts = int(original['native_pts'])
        if pts not in changed:
            result.append(original); continue
        revised = changed[pts]
        by_id = {str(row['source_point_ids'][0]): row for row in revised['rows']}
        expected_ids = {str(pid) for ri, row in enumerate(original['rows'])
                        if ri in touched[pts] for pid in row['source_point_ids']
                        if (pts, str(pid)) not in quarantines}
        if set(by_id) != expected_ids:
            raise ValueError('Edited ordinary row inventory differs from exact retained IDs')
        keep, rows = [], []
        for ri, row in enumerate(original['rows']):
            if ri not in touched[pts]:
                rows.append(row); keep.extend([True]*len(row['uv'])); continue
            for pid in row['source_point_ids']:
                retained = (pts, str(pid)) not in quarantines
                keep.append(retained)
                if retained: rows.append(by_id[str(pid)])
        keep = np.asarray(keep, bool)
        mapping = np.full(len(keep), -1, int)
        mapping[keep] = np.arange(int(keep.sum()))
        groups = []
        for group in original['whitening_groups']:
            old = np.asarray(group['point_indices'], int)
            if not keep[old].all():
                raise ValueError('Ordinary delta removed a correlated member')
            new = mapping[old].tolist()
            groups.append(group if new == list(group['point_indices']) else dict(group, point_indices=new))
        result.append(dict(original, rows=rows, whitening_groups=groups,
            source_point_count=int(keep.sum()),
            inactive_source_correspondences=deepcopy(original.get('inactive_source_correspondences', []))
                + revised.get('inactive_source_correspondences', []),
            original_approved_source_point_count=original.get('original_approved_source_point_count', original['source_point_count']),
            post_native_ordinary_revision=revision_reference))
    aliases = {key: dict(value, before_target=value['original_target'], after_target=value['target'])
               for key, value in changes.items()}
    excluded = {key: dict(value, before_target=value['original_target']) for key, value in quarantines.items()}
    scope = compare_source_frames(frames, result, aliases, {}, excluded)
    proof.update(unrelated_native_query_rows_preserved_as_complete_original_objects=True,
        original_correlated_operators_preserved=True, complete_source_comparison=scope)
    return result, proof
