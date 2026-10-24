"""Bind source-image verdicts and exchange representatives without adding rows.

This is not an image-identity estimator. Callers must supply independently
reviewed source-image evidence. Partial verdict coverage never proves all rows.
"""
from __future__ import annotations
import numpy as np


class SourceIdentityError(ValueError):
    pass


def inspect_verdicts(records, fit_rows, verdicts):
    """Read-only exact-row check; records is a sparse mapping by raw row.

    Source ID, UV, PTS, shot, material and path are all required. An identity
    change requires a new verdict; changing a name cannot clear an old finding.
    """
    fit = set(map(int, fit_rows)); seen = set(); rejected = []; ambiguous = []
    confirmed = []; blockers = []
    for v in verdicts:
        i = int(v['raw_row'])
        if i in seen: raise SourceIdentityError('Duplicate source verdict')
        seen.add(i)
        if i not in records: raise SourceIdentityError('Missing actual source row')
        r = records[i]
        for key in ('source_row_id', 'native_pts', 'shot_index', 'material_index', 'path'):
            if r[key] != v[key]:
                raise SourceIdentityError('Source verdict binding changed: '+str(i)+'/'+key)
        if not np.array_equal(np.asarray(r['uv'], dtype=np.float64),
                              np.asarray(v['uv'], dtype=np.float64)):
            raise SourceIdentityError('Source verdict UV changed: '+str(i))
        if not v.get('evidence') or not any(isinstance(v.get(k),str) and v[k].strip() for k in ('finding','reason')):
            raise SourceIdentityError('Missing source-image evidence or finding')
        if v.get('uses_fitted_model_projection') is not False:
            raise SourceIdentityError('Identity verdict must be source-image-only')
        decision = v.get('decision')
        if decision == 'confirmed': confirmed.append(i)
        elif decision == 'rejected': rejected.append(i)
        elif decision == 'ambiguous': ambiguous.append(i)
        else: raise SourceIdentityError('Unsupported source verdict')
        if decision != 'confirmed' and i in fit: blockers.append(i)
    return dict(schema='rinkfit.source_identity_verdict_check.v1',
                listed_rows=len(seen), confirmed_rows=sorted(confirmed),
                rejected_rows=sorted(rejected), ambiguous_rows=sorted(ambiguous),
                blocking_fit_rows=sorted(blockers), passed=not blockers,
                scope='listed_source_rows_only', all_source_identity_proven=False,
                raw_rows_changed=False, fit_rows_added=False)


def require_verdicts(records, fit_rows, verdicts):
    report = inspect_verdicts(records, fit_rows, verdicts)
    if report['blocking_fit_rows']:
        raise SourceIdentityError('Rejected/ambiguous source identities remain in fit: '+
                                  ','.join(map(str, report['blocking_fit_rows'])))
    return report


def exchange_representatives(fit_rows, weights, records, swaps):
    """Pure, explicit one-for-one exchanges of existing source rows.

    Each record must carry a source-only identity_decision and an exact
    budget_group_id. This function deliberately cannot choose candidates,
    rebind a material, alter raw observations, issue W, or migrate AL state.
    Only rank-zero engineering weights can be exchanged here. Correlated W
    groups require their own whole-block transformation and are refused.
    """
    fit = np.asarray(fit_rows, dtype=np.int64)
    if len(np.unique(fit)) != len(fit): raise SourceIdentityError('Duplicate fit rows')
    old_weights = np.asarray(weights, dtype=np.float64)
    if (not np.isfinite(old_weights).all()) or (old_weights < 0).any():
        raise SourceIdentityError('Invalid engineering weights')
    if (fit < 0).any() or (fit >= len(old_weights)).any():
        raise SourceIdentityError('Fit row outside full raw denominator')
    if np.count_nonzero(old_weights) != len(fit) or not np.all(old_weights[fit] > 0):
        raise SourceIdentityError('Weights must be the actual selected objective weights')
    selected = set(map(int, fit)); used = set(); result = old_weights.copy()
    for item in swaps:
        old, new = int(item['old_row']), int(item['new_row'])
        if old in used or new in used or old == new:
            raise SourceIdentityError('Duplicate/conflicting representative exchange')
        used.update((old, new))
        if old not in selected or new in selected or new < 0 or new >= len(result):
            raise SourceIdentityError('Exchange must replace one existing fit row')
        a, b = records[old], records[new]
        if a['identity_decision'] not in ('rejected', 'ambiguous'):
            raise SourceIdentityError('Old representative has no source-identity failure')
        if b['identity_decision'] != 'confirmed' or not b.get('identity_evidence'):
            raise SourceIdentityError('Replacement identity lacks independent source evidence')
        if b.get('uses_fitted_model_projection') is not False:
            raise SourceIdentityError('Model residual cannot select a replacement identity')
        if b['admission'] != 1 or b['fit_role'] != 1 or b['frame_role'] != 'training':
            raise SourceIdentityError('Replacement is not eligible training source')
        for key in ('shot_index', 'native_pts', 'material_index', 'kind',
                    'measurement_operator', 'budget_group_id'):
            if a[key] != b[key]:
                raise SourceIdentityError('Exchange changes source stratum: '+key)
        if not a['budget_group_id'] or a['W_rank'] != 0 or b['W_rank'] != 0:
            raise SourceIdentityError('Exchange requires same rank-zero engineering budget')
        if not a.get('path_allowed') or not b.get('path_allowed'):
            raise SourceIdentityError('Source finite-branch authority missing')
        result[new], result[old] = result[old], 0.
        selected.remove(old); selected.add(new)
    output = np.asarray(sorted(selected), dtype=np.uint64)
    if len(output) != len(fit) or np.count_nonzero(result) != len(fit):
        raise AssertionError('Representative count changed')
    # A transfer preserves each individual weight exactly, including roundoff.
    if not np.array_equal(np.sort(result[result > 0]), np.sort(old_weights[old_weights > 0])):
        raise AssertionError('Engineering budget changed')
    return output, result


def validate_run_binding(packet_path, selection, verdict_document, check_reference):
    """Adapter for dense_joint_run; reads only listed source rows and fit IDs."""
    import json
    from pathlib import Path
    from .compact_identity_withhold_v2 import _source_identities
    from .dense_joint_packet import _array
    doc_path = Path(packet_path).resolve()
    doc = json.loads(doc_path.read_text(encoding='utf-8-sig'))
    requested = [int(v['raw_row']) for v in verdict_document['rows']]
    if verdict_document.get('schema') != 'rinkfit.source_identity_verdicts.v1':
        raise SourceIdentityError('Unknown source identity verdict schema')
    if verdict_document.get('source_packet_sha256') != selection['source_packet']['sha256']:
        raise SourceIdentityError('Identity verdict source generation changed')
    for ref in verdict_document['evidence_files']: check_reference(ref)
    arrays = {}; visited = set()
    while True:
        identity = str(doc_path).casefold()
        if identity in visited: raise SourceIdentityError("Cyclic source array lineage")
        visited.add(identity)
        for key in ('uv', 'native_pts', 'shot_index', 'material_index', 'path'):
            if key not in arrays and key in doc.get('arrays', {}): arrays[key] = (doc['arrays'][key], doc_path.parent)
        if len(arrays) == 5: break
        if 'dense_parent_packet' not in doc: raise SourceIdentityError('Missing original arrays')
        parent = dict(doc['dense_parent_packet'])
        parent_path = Path(parent['file'])
        parent['file'] = str(parent_path if parent_path.is_absolute() else doc_path.parent / parent_path)
        doc_path = Path(check_reference(parent)).resolve()
        doc = json.loads(doc_path.read_text(encoding='utf-8-sig'))
    loaded = {}
    for key, (ref, base) in arrays.items():
        loaded[key] = _array(ref, base)
    ids, _ = _source_identities(packet_path, requested)
    records = {i:dict(source_row_id=ids[i], **{k:v[i].tolist() for k,v in loaded.items()}) for i in requested}
    fit = np.fromfile(check_reference(selection['fit_rows']), dtype='<u8')
    if len(fit) != selection['fit_count']: raise SourceIdentityError('Fit selection length changed')
    return require_verdicts(records, fit, verdict_document['rows'])
