"""Explicit source exclusions and exact marginal statistics over immutable rows.

An inactive row remains in every raw source/state/prediction array.  It is never
represented by an artificial zero residual or by a column slice of the old W.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .data import sha256_file


def _path(root, spec):
    return Path(root)/spec.get('file', spec.get('path', ''))


def _json(root, spec):
    p = _path(root, spec)
    if sha256_file(p) != spec['sha256']:
        raise ValueError('Source exclusion asset checksum changed: '+str(p))
    return json.loads(p.read_text(encoding='utf-8'))


def _binary(root, spec):
    p = _path(root, spec)
    if sha256_file(p) != spec['sha256']:
        raise ValueError('Source exclusion array checksum changed: '+str(p))
    a = np.fromfile(p, dtype=spec['dtype'])
    if a.size != int(np.prod(spec['shape'])):
        raise ValueError('Source exclusion array shape mismatch')
    return a.reshape(spec['shape'])


def _signature(value):
    """Descriptor identity excludes relocation, never shape/type/content."""
    if isinstance(value, dict):
        return {k:_signature(v) for k,v in value.items() if k not in ['file','path']}
    if isinstance(value, list):return [_signature(v) for v in value]
    return value


def load_exclusion(manifest, manifest_path, arrays, n, parent_loader):
    keys = ['source_exclusion_parent_manifest','source_exclusion',
            'active_source_mask','active_statistics','active_observations']
    present = [k in manifest for k in keys]
    if not any(present):
        return {'enabled':False,'active_mask':np.ones(n,dtype=bool),'records':[],
                'replacements':{},'active_observations':n}
    if not all(present):raise ValueError('Incomplete source exclusion manifest')
    root = Path(manifest_path).parent
    parent = manifest['source_exclusion_parent_manifest']
    pm, _, _, inherited = parent_loader(_path(root,parent), n,
                                       expected_sha256=parent['sha256'])
    if inherited.get('source_exclusion',{}).get('enabled'):
        raise ValueError('Stacked source exclusions require an explicit merged transition')
    if set(manifest)-set(keys) != set(pm):
        raise ValueError('Source exclusion changed unrelated semantic fields')
    for k,v in pm.items():
        if _signature(manifest[k]) != _signature(v):
            raise ValueError('Source exclusion changed parent semantic identity: '+k)
    ex = _json(root,manifest['source_exclusion'])
    if (ex.get('schema')!='rinkfit.confirmed_source_exclusion.v1'
            or ex.get('parent_semantic_sha256')!=parent['sha256']
            or ex.get('raw_observations')!=n
            or ex.get('active_observations')!=manifest['active_observations']
            or ex.get('raw_arrays_W_E_and_record_bytes_remain_immutable') is not True):
        raise ValueError('Source exclusion authority/source registry mismatch')
    mask = _binary(root,manifest['active_source_mask'])
    if (mask.dtype!=np.dtype('u1') or mask.shape!=(n,)
            or np.any((mask!=0)&(mask!=1))):
        raise ValueError('Active source mask requires a complete binary uint8 vector')
    active = mask.astype(bool); records = ex.get('exclusions',[])
    excluded = [r['source_row'] for r in records]
    # This transition is deliberately scoped to the one current user grant.
    if excluded != [25537] or np.flatnonzero(~active).tolist()!=excluded:
        raise ValueError('Source exclusion exceeds the single-row user authorization')
    if np.any(arrays['kind'][excluded]!=0):
        raise ValueError('This exclusion is limited to the authorized ordinary source row')
    for line in _path(root,manifest['assets']['source_runs.jsonl']).read_text(encoding='utf-8').splitlines():
        if line.strip() and set(json.loads(line)['source_rows']).intersection(excluded):
            raise ValueError('An ordered source-run exclusion needs a separate topology transition')
    if int(active.sum())!=manifest['active_observations']:
        raise ValueError('Active source count mismatch')
    for r in records:
        auth = _json(root,r['direct_user_authorization'])
        if (auth.get('schema')!='rinkfit.user_authorized_source_exclusion.v1'
                or auth.get('authority')!='direct_current_user_instruction'
                or not auth.get('user_message')
                or auth.get('user_authorizes_excluding_only_this_source_row') is not True
                or auth.get('exclude_other_source_rows') is not False
                or auth.get('raw_arrays_and_record_bytes_remain_immutable') is not True
                or auth.get('engineering_budget_for_other_rows_remains_2px') is not True
                or auth.get('original_npz_source_identity')!=ex['source_identity']
                or auth.get('source_row')!=r['source_row']
                or auth.get('native_pts')!=r['native_pts']
                or auth.get('original_source_point_id')!=r['source_point_id']
                or auth.get('original_source_label')!=r['original_source_label']
                or auth.get('source_uv')!=r['source_uv']
                or _signature(auth.get('source_record'))!=_signature(r.get('source_record'))):
            raise ValueError('Single-row exclusion lacks exact direct user authority')
        _json(root,r['source_record'])
    stats = _json(root,manifest['active_statistics'])
    stats_root = _path(root,manifest['active_statistics']).parent
    if (stats.get('schema')!='rinkfit.active_statistics.v1'
            or stats.get('parent_semantic_sha256')!=parent['sha256']
            or stats.get('source_exclusion_sha256')!=manifest['source_exclusion']['sha256']
            or stats.get('active_source_mask_sha256')!=manifest['active_source_mask']['sha256']
            or stats.get('active_observations')!=int(active.sum())
            or stats.get('original_groups_asset_sha256')!=manifest['assets']['groups.jsonl']['sha256']
            or stats.get('original_weights_array_sha256')!=manifest['arrays']['weights']['sha256']):
        raise ValueError('Active statistics identity mismatch')
    groups = [json.loads(l) for l in _path(root,manifest['assets']['groups.jsonl']).read_text(encoding='utf-8').splitlines() if l.strip()]
    affected = {}
    for i,g in enumerate(groups):
        rows = arrays['group_rows'][g['member_start']:g['member_start']+g['member_count']].astype(np.int64)
        if np.any(~active[rows]):affected[g['group_id']] = (i,g,rows)
    replacements = {}
    for replacement in stats.get('replacements',[]):
        gid = replacement['group_id']
        if gid not in affected or gid in replacements:
            raise ValueError('Unexpected or duplicate active statistical replacement')
        ordinal,g,old_rows = affected[gid]
        retained = old_rows[active[old_rows]]
        if replacement['source_rows']!=retained.tolist() or replacement['rank']!=g['rank']:
            raise ValueError('Active marginal membership/rank changed')
        w = g['weight_start']; rank = g['rank']; old_m = len(old_rows)
        old_W = arrays['weights'][w:w+rank*2*old_m].reshape(rank,2*old_m)
        matrix_hash = hashlib.sha256(old_W.astype('<f8').tobytes()).hexdigest()
        if replacement['original_matrix_sha256']!=matrix_hash:
            raise ValueError('Original statistical operator identity changed')
        proof = _json(stats_root,replacement['proof']); source = _json(stats_root,proof['original_source_packet'])
        features = [f for f in source['features'] if f['id']==proof['feature_id']]
        if len(features)!=1:raise ValueError('Ambiguous original line covariance')
        line = features[0]['source_line_fit']
        C = np.asarray(line['covariance_ab'],float)
        coefficients = np.asarray(line['coefficient_y_equals_ax_plus_b'],float)
        normal = np.array([-coefficients[0],1.]);normal/=np.linalg.norm(normal)
        old_uv = np.asarray(proof['original_uv_native'],float)
        retained_uv = old_uv[active[old_rows]]
        def design(uv):
            basis = np.c_[uv[:,0],np.ones(len(uv))]*normal[1]
            return (normal[None,:,None]*basis[:,None,:]).reshape(2*len(uv),2)
        J0 = design(old_uv); J = design(retained_uv)
        W = _binary(stats_root,replacement['W'])
        B = np.linalg.pinv(J); metric = B.T@np.linalg.inv(C)@B
        if (W.shape!=(rank,2*len(retained)) or W.dtype!=np.dtype('<f8')
                or not np.isfinite(W).all() or rank!=2
                or proof['original_group_ordinal']!=ordinal
                or proof['original_source_rows']!=old_rows.tolist()
                or proof['source_rows']!=retained.tolist()
                or proof['excluded_source_rows']!=old_rows[~active[old_rows]].tolist()
                or not np.array_equal(C,proof['coefficient_covariance_ab'])
                or not np.array_equal(coefficients,proof['line_coefficients_ab'])
                or not np.array_equal(J,proof['measurement_design'])
                or not np.array_equal(J0,proof['original_measurement_design'])
                or not np.allclose(W.T@W,metric,rtol=1e-9,atol=1e-11)
                or not np.allclose(W@(J@C@J.T)@W.T,np.eye(rank),rtol=1e-9,atol=1e-9)
                or not np.allclose(old_W@(J0@C@J0.T)@old_W.T,np.eye(rank),rtol=1e-9,atol=1e-9)):
            raise ValueError('Active W is not the exact original-noise covariance marginal')
        replacements[gid] = {'source_rows':retained,'W':W,'rank':rank,
                             'proof':proof,'original_source_rows':old_rows}
    formal = sum(g['rank'] for g in groups)+2*int(active[arrays['ordinary_rows'].astype(np.int64)].sum())
    if (set(replacements)!=set(affected) or len(groups)!=stats['original_group_count']
            or stats['remaining_original_groups_bitexact']!=len(groups)-len(replacements)
            or formal!=stats['formal_residual_rows']):
        raise ValueError('Incomplete active statistical membership/rank ledger')
    return {'enabled':True,'active_mask':active,'records':records,
            'replacements':replacements,'active_observations':int(active.sum()),
            'source_identity':ex['source_identity'],'budget_sha256':ex['original_engineering_budget_sha256'],
            'source_exclusion_sha256':manifest['source_exclusion']['sha256'],
            'active_source_mask_sha256':manifest['active_source_mask']['sha256'],
            'active_statistics_sha256':manifest['active_statistics']['sha256'],
            'statistics':stats}


def verify_exclusion_source(info, obs, source_identity, budget_sha256):
    if not info['enabled']:return
    if info['source_identity']!=source_identity or info['budget_sha256']!=budget_sha256:
        raise ValueError('Exclusion source/budget identity mismatch')
    for r in info['records']:
        i=r['source_row']
        if (int(obs.native_pts[i])!=r['native_pts']
                or str(obs.source_point_ids[i])!=r['source_point_id']
                or str(obs.source_target_labels[i])!=r['original_source_label']
                or not np.array_equal(obs.source_uv[i],r['source_uv'])):
            raise ValueError('Excluded row does not match its immutable source composite key')
    for replacement in info['replacements'].values():
        p=replacement['proof'];ii=replacement['original_source_rows']
        if (not np.array_equal(obs.source_uv[ii],p['original_uv_native'])
                or obs.source_point_ids[ii].tolist()!=p['original_source_point_ids']):
            raise ValueError('Marginal statistic source rows/UV changed')
