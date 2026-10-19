"""Evidence-scoped source-audit revisions above an immutable semantic parent.

A Codex source review is distinct from a direct human confirmation.  Original
source rows and W stay archived; only the new branch's active likelihood changes.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
from .data import sha256_file


def covariance_marginal(original_W, keep_coordinates, source_rank_threshold):
    """Marginalize the original supported covariance, never slice old W columns.

    The original full-row-rank W defines covariance C=W^+ (W^+)^T on its
    measured subspace.  Select C's retained coordinates, then whiten that
    marginal with an SVD of its factor.  Original numerical source rank is
    retained; no diagonal floor or newly inferred tiny mode is introduced.
    """
    W=np.asarray(original_W,float);keep=np.asarray(keep_coordinates,bool)
    if W.ndim!=2 or keep.shape!=(W.shape[1],) or not np.isfinite(W).all():
        raise ValueError('Invalid original covariance marginal input')
    _,s,vt=np.linalg.svd(W,full_matrices=False)
    if np.any(s<=0):raise ValueError('Original whitening operator is not full row rank')
    factor=vt.T/s
    retained=factor[keep]
    u,sv,_=np.linalg.svd(retained,full_matrices=False)
    backward=np.finfo(float).eps*max(retained.shape)*np.linalg.norm(retained,ord='fro')
    threshold=max(float(source_rank_threshold),float(backward))
    positive=sv>threshold
    if not positive.any():raise ValueError('Retained source covariance has no measured mode')
    marginal_W=(u[:,positive]/sv[positive]).T
    evidence={'source_rank_threshold':float(source_rank_threshold),'SVD_backward_error_bound':float(backward),
        'numerical_rank_threshold':threshold,'singular_values':sv.tolist(),'retained_rank':int(positive.sum()),
        'original_whitening_covariance_identity_error':float(np.linalg.norm(W@factor@factor.T@W.T-np.eye(len(W)),ord='fro')),
        'marginal_whitening_covariance_identity_error':float(np.linalg.norm(marginal_W@retained@retained.T@marginal_W.T-np.eye(len(marginal_W)),ord='fro'))}
    return marginal_W,retained,evidence


def _path(root,spec):return Path(root)/spec.get('path',spec.get('file',''))
def _json(root,spec):
    p=_path(root,spec)
    if sha256_file(p)!=spec['sha256']:raise ValueError('Source-audit evidence checksum changed: '+str(p))
    return json.loads(p.read_text(encoding='utf-8'))
def _binary(root,spec):
    p=_path(root,spec)
    if sha256_file(p)!=spec['sha256']:raise ValueError('Source-audit binary checksum changed: '+str(p))
    a=np.fromfile(p,dtype=spec['dtype'])
    if a.size!=int(np.prod(spec['shape'])):raise ValueError('Source-audit array shape changed')
    return a.reshape(spec['shape'])



def _verify_flattened_statistics(root, stats, replacements, groups):
    """A compatibility table is an exact alias of the validated covariance ledger."""
    flat=stats.get('replacements')
    if not isinstance(flat,list):raise ValueError('Missing canonical flattened source statistics')
    seen=set();ordinals={g['group_id']:i for i,g in enumerate(groups)}
    for rep in flat:
        gid=rep.get('group_id')
        if gid in seen or gid not in replacements:raise ValueError('Duplicate or unvalidated flattened source statistics')
        seen.add(gid);canonical=replacements[gid]
        if (('original_group_ordinal'in rep and rep['original_group_ordinal']!=ordinals[gid])
            or ('original_matrix_sha256'in rep and rep['original_matrix_sha256']!=groups[ordinals[gid]]['matrix_sha256'])
            or rep.get('source_rows')!=canonical['source_rows'].tolist()
            or rep.get('rank')!=canonical['rank']):
            raise ValueError('Flattened source statistics membership differs')
        W=_binary(root,rep['W']);proof=_json(root,rep['proof'])
        if not np.array_equal(W,canonical['W'])or proof!=canonical['proof']:
            raise ValueError('Flattened W differs from validated source covariance')
    if seen!=set(replacements):raise ValueError('Incomplete flattened source statistics')


def _source_registry_rows(root, spec, selected, n):
    source=_path(root,spec)
    if sha256_file(source)!=spec['sha256']:raise ValueError('Source registry checksum changed')
    found={};count=0
    with source.open(encoding='utf-8')as stream:
        for line in stream:
            if not line.strip():continue
            row=json.loads(line)
            if row.get('source_row')!=count:raise ValueError('Original source registry row order changed')
            if count in selected:found[count]=row
            count+=1
    if count!=n or set(found)!=set(selected):raise ValueError('Incomplete original source registry')
    return found


def _verify_native_source_key(root, record, evidence, canonical):
    original=_json(root,record['source_record'])
    declared_registry=_json(root,evidence['source_registry_record'])
    if original!=canonical or declared_registry!=canonical or canonical.get('source_row')!=record['source_row']:
        raise ValueError('Audit source record is not the immutable original registry row')
    packet=_json(root,canonical['source_packet'])
    declared_packet=_json(root,evidence['source_packet'])
    if declared_packet!=packet:raise ValueError('Audit source packet differs from original registry')
    feature_id=canonical['source_feature_id'];features=[f for f in packet['features']if f['id']==feature_id]
    if len(features)!=1:raise ValueError('Original source feature is ambiguous')
    si=canonical['source_segment_index'];vi=canonical['original_vertex_index']
    if (type(si)is not int or type(vi)is not int or si<0 or vi<0
        or canonical.get('native_query_id')is not None):
        raise ValueError('Audit requires an exact original recorded vertex identity')
    try:uv=features[0]['segments'][si]['uv'][vi]
    except (KeyError,IndexError,TypeError):raise ValueError('Original source vertex does not exist')
    point_id=f'{feature_id}/segment/{si}/vertex/{vi}'
    source=packet['source'];frame=evidence['exact_native_frame'];pts=record['native_pts']
    if (record['source_point_id']!=point_id or record['original_source_label']!=feature_id
        or record['source_uv']!=uv or pts!=packet['source_pts'] or pts!=source['native_pts']
        or pts!=source.get('actual_pts',pts) or pts!=frame.get('native_pts')
        or canonical['snapshot_frame']!=packet['programme_frame_index_1based']
        or frame.get('sha256')!=source['png_sha256']
        or [frame.get('height'),frame.get('width')]!=source['native_hw']):
        raise ValueError('Audit PTS, UV, ID, label or native frame differs from original source')


def load_source_audit_revision(path,m,n,parent_loader):
    root=Path(path).parent
    parent=m['source_audit_parent_manifest']
    pm,a,_,inherited=parent_loader(_path(root,parent),n,expected_sha256=parent['sha256'])
    added={'source_audit_parent_manifest','source_audit_revision'}
    changed={'active_source_mask','active_statistics','active_observations'}
    if set(m)!=set(pm)|added or any(m[k]!=v for k,v in pm.items()if k not in changed):
        raise ValueError('Source-audit revision changed unrelated semantic content')
    old=inherited.get('source_exclusion',{})
    if not old.get('enabled'):
        raise ValueError('Audit revision requires an unchanged historical exclusion authority')
    historical=np.flatnonzero(~old['active_mask']).tolist()
    if inherited.get('source_audit_revision'):raise ValueError('Audit revisions require an explicit merged successor')
    rev=_json(root,m['source_audit_revision'])
    newly=rev.get('new_audit_exclusions',[])
    if (not isinstance(newly,list) or not newly or any(type(r)is not int or not 0<=r<n for r in newly)
        or newly!=sorted(set(newly)) or set(newly)&set(historical)):
        raise ValueError('Audit rows must be distinct new declared source rows')
    expected=old['active_mask'].copy();expected[newly]=False
    if (rev.get('schema')!='rinkfit.native-source-audit-revision.v1' or rev.get('parent_semantic_sha256')!=parent['sha256']
        or rev.get('raw_observations')!=n or rev.get('source_identity')!=old['source_identity']
        or rev.get('historical_human_authorized_exclusions')!=historical
        or rev.get('active_observations')!=int(expected.sum())
        or rev.get('raw_rows_UV_ID_labels_W_E_remain_immutable')is not True
        or rev.get('not_a_direct_human_confirmation')is not True):
        raise ValueError('Invalid distinct source-audit authority/source identity')
    exclusions=rev.get('exclusions',[])
    if [r.get('source_row')for r in exclusions]!=newly:raise ValueError('Unreviewed source rows in this audit revision')
    active=_binary(root,m['active_source_mask'])
    if active.dtype!=np.dtype('u1')or active.shape!=(n,)or np.any((active!=0)&(active!=1)):
        raise ValueError('Invalid complete source-audit active mask')
    if not np.array_equal(active.astype(bool),expected)or m['active_observations']!=int(expected.sum()):
        raise ValueError('Source-audit active difference exceeds individually reviewed row')
    registry=_source_registry_rows(root,m['assets']['source_rows.jsonl'],set(newly),n)
    for record in exclusions:
        r=record['source_row'];evidence=_json(root,record['native_image_audit'])
        if (a['kind'][r]!=0 or evidence.get('schema')!='rinkfit.native-image-visibility-audit.v1'
            or evidence.get('authority')!='codex_native_source_audit_under_user_request'
            or evidence.get('not_a_direct_human_confirmation')is not True
            or evidence.get('user_personally_confirmed_pixel')is not False
            or evidence.get('source_row')!=r or evidence.get('native_pts')!=record['native_pts']
            or evidence.get('source_uv')!=record['source_uv']
            or evidence.get('original_source_point_id')!=record['source_point_id']
            or evidence.get('original_source_label')!=record['original_source_label']
            or evidence.get('review_applies_to_other_source_rows')is not False
            or evidence.get('classification')!='foreground_occlusion_not_static_board_material'
            or evidence.get('native_video_PTS_redecode_BGR_bitexact')is not True
            or not evidence.get('current_user_request_scope') or not evidence.get('project_rule_applied')):
            raise ValueError('Audit row lacks exact native-source review authority')
        for key in ['exact_native_frame','visual_diagnostic','source_packet','source_registry_record','project_rules']:
            spec=evidence[key]
            if sha256_file(_path(root,spec))!=spec['sha256']:raise ValueError('Audit source asset changed: '+key)
        _verify_native_source_key(root,record,evidence,registry[r])
    for line in _path(root,m['assets']['source_runs.jsonl']).read_text(encoding='utf8').splitlines():
        if line.strip()and set(newly)&set(json.loads(line)['source_rows']):
            raise ValueError('Ordered run exclusion needs a separate topology revision')
    stats=_json(root,m['active_statistics'])
    if (stats.get('schema')!='rinkfit.source-audit-active-statistics.v1'
        or stats.get('parent_semantic_sha256')!=parent['sha256']
        or stats.get('source_audit_revision_sha256')!=m['source_audit_revision']['sha256']
        or stats.get('active_source_mask_sha256')!=m['active_source_mask']['sha256']):
        raise ValueError('Audit statistics identity mismatch')
    groups=[json.loads(l)for l in _path(root,m['assets']['groups.jsonl']).read_text(encoding='utf8').splitlines()if l.strip()]
    replacements=copy.deepcopy(old['replacements']);found=set()
    for rep in stats.get('new_replacements',[]):
        gid=rep['group_id'];ordinal=rep['original_group_ordinal'];g=groups[ordinal]
        if gid in found:raise ValueError('Duplicate new source-audit replacement')
        ii=a['group_rows'][g['member_start']:g['member_start']+g['member_count']].astype(np.int64)
        retained=ii[expected[ii]]
        if gid!=g['group_id']or not set(newly)&set(ii.tolist()) or rep['source_rows']!=retained.tolist():
            raise ValueError('Unexpected source-audit statistical replacement')
        W=a['weights'][g['weight_start']:g['weight_start']+g['rank']*2*len(ii)].reshape(g['rank'],2*len(ii))
        keep=np.repeat(expected[ii],2)
        threshold=g['numerical_rank_evidence']['numerical_rank_threshold']
        reference,factor,rank_proof=covariance_marginal(W,keep,threshold)
        new=_binary(root,rep['W']);proof=_json(root,rep['proof'])
        if (new.shape!=reference.shape or rep['rank']!=len(reference)
            or not np.allclose(new.T@new,reference.T@reference,rtol=1e-10,atol=1e-11)
            or proof['original_source_rows']!=ii.tolist()or proof['source_rows']!=retained.tolist()
            or proof['excluded_source_rows']!=ii[~expected[ii]].tolist()or proof['original_W_sha256']!=g['matrix_sha256']
            or proof['method']!='retained_original_supported_covariance_marginal'
            or not np.allclose(new@factor@factor.T@new.T,np.eye(len(new)),rtol=1e-10,atol=1e-10)):
            raise ValueError('Audit W is not the original covariance marginal')
        replacements[gid]={'source_rows':retained,'W':new,'rank':len(new),'proof':proof,'original_source_rows':ii}
        found.add(gid)
    affected={g['group_id']for g in groups if set(newly)&set(a['group_rows'][g['member_start']:g['member_start']+g['member_count']].tolist())}
    if found!=affected:raise ValueError('Incomplete source-audit group marginalization')
    _verify_flattened_statistics(root,stats,replacements,groups)
    formal=sum((replacements[g['group_id']]['rank']if g['group_id']in replacements else g['rank'])for g in groups)+2*int(expected[a['ordinary_rows'].astype(int)].sum())
    if formal!=stats['formal_residual_rows']or stats['active_observations']!=int(expected.sum()):
        raise ValueError('Incomplete source-audit formal likelihood ledger')
    result=copy.copy(inherited);ex=copy.copy(old)
    ex.update(active_mask=expected,records=old['records']+exclusions,replacements=replacements,active_observations=int(expected.sum()),
        source_exclusion_sha256=m['source_audit_revision']['sha256'],active_source_mask_sha256=m['active_source_mask']['sha256'],
        active_statistics_sha256=m['active_statistics']['sha256'],statistics=stats,
        historical_source_exclusion_sha256=old['source_exclusion_sha256'])
    result['source_exclusion']=ex
    result['source_audit_revision']={'path':str(_path(root,m['source_audit_revision'])),'sha256':m['source_audit_revision']['sha256'],
        'new_excluded_source_rows':newly,'not_a_direct_human_confirmation':True}
    return m,a,Path(path),result
