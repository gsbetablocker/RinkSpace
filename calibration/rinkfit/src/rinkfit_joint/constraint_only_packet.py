"""All admitted sources keep engineering constraints without invented noise.

Exact source-readout aliases are retained raw but consumed only once. Known
source groups are marginalized through their full covariance, never W columns.
"""
from pathlib import Path
import json
from collections import Counter
import numpy as np
from .io_utils import file_sha256,read_jsonl,write_json
from .packet_storage import write_packet
from .source_statistics import supported_factor_whitening


def compile_constraint_only_scope(parent_packet,output,*,actual_source_aliases=None):
    parent=Path(parent_packet).resolve();p=json.loads(parent.read_text(encoding='utf8'))
    out=Path(output).resolve()
    if out.exists():raise ValueError('new immutable all-source scope required')
    out.mkdir(parents=True)
    checked={}
    def ref(path):
        path=Path(path).resolve();key=str(path)
        if key not in checked:checked[key]=file_sha256(path)
        return {'file':key,'sha256':checked[key]}
    def verify(d):
        f=Path(d.get('file',d.get('path')));f=f if f.is_absolute() else parent.parent/f
        if ref(f)['sha256']!=d['sha256']:raise ValueError('changed immutable source dependency: '+str(f))
        return f
    def array(k):
        d=p['arrays'][k];return np.memmap(verify(d),dtype=d['dtype'],mode='r',shape=tuple(d['shape']))
    n=int(p['row_count']);ad=np.array(array('admission'));role=array('fit_role');before=(ad==1)&(role==1)
    pts=array('native_pts');uv=array('uv');mi=array('material_index');kind=array('kind');E=array('E')
    ids=p['source_row_ids'];byid={(x,int(pts[i])):i for i,x in enumerate(ids)}
    aliases=[];alias_ref=None
    if actual_source_aliases:
        ap=Path(actual_source_aliases).resolve();index=json.loads(ap.read_text(encoding='utf8'));alias_ref=ref(ap)
        verify(index['audit_manifest']);verify(index['producer_script'])
        for a in read_jsonl(verify(index['aliases'])):
            i=byid.get((a['source_row_id'],int(a['native_pts'])))
            if i is None or not before[i]:continue
            j=byid.get((a['actual_source_readout_alias_of'],int(a['native_pts'])))
            if j is None or not before[j]:raise ValueError('alias canonical readout not currently admitted')
            if i==j or i<int(p['new_source_start']):raise ValueError('unexpected inherited/self readout alias')
            if not np.array_equal(uv[i],uv[j]) or mi[i]!=mi[j] or kind[i]!=kind[j] or E[i]!=E[j]:
                raise ValueError('exact readout alias conflicts with current measurement/material/E')
            if a.get('duplicate_source_geometry_I_or_conditional_W_consumption_allowed')is not False:raise ValueError('alias source policy not explicit')
            aliases.append({**a,'joint_row':i,'canonical_joint_row':j,'physical_source_identity_withdrawn':False,'independent_consumption_admission':0})
        alias_rows={x['joint_row']for x in aliases}
        if len(alias_rows)!=len(aliases):raise ValueError('duplicate alias decisions')
        if any(x['canonical_joint_row']in alias_rows for x in aliases):raise ValueError('unresolved chained source aliases')
        ad[list(alias_rows)]=0
        af=out/'admission.u8';ad.tofile(af);p['arrays']['admission']={**p['arrays']['admission'],**ref(af)}
    active=(ad==1)&(role==1)
    base=parent.parent.parent
    source_journals={Path(x['file']).parent.name:x for x in p.get('source_family_journal_refs',[])}
    manual=None;source_refs={}
    def unknown_source_refs(rows):
        nonlocal manual
        datasets={p['source_dataset_ids'][i]for i in rows};refs=[]
        for dataset in sorted(datasets):
            if dataset.startswith('actual_geometry:'):
                gate=dataset.split(':')[1]
                if gate not in source_refs:
                    journal=source_journals.get(gate)
                    if journal is None:raise ValueError('unknown-noise family lacks its actual source journal')
                    refs0=[ref(verify(journal)),ref(base/gate/'manifest.json')]
                    if gate=='board_readout_selected_scope_v1':refs0.append(ref(base/'board_readout_selected_scope_v2/manifest.json'))
                    source_refs[gate]=refs0
                refs.extend(source_refs[gate])
            elif dataset=='current_native_s02':
                gate='manual_source_geometry_v3'
                if manual is None:
                    mr=base/gate/'geometry_rows.jsonl.gz';manual={x['source_row_id']for x in read_jsonl(mr)}
                    if len(manual)!=76:raise ValueError('manual source closure is not the actual 76 clicks')
                    source_refs[gate]=[ref(mr),ref(base/gate/'manifest.json')]
                if any(ids[i]not in manual for i in rows if p['source_dataset_ids'][i]==dataset):raise ValueError('automatic readout cannot borrow manual unknown-noise proof')
                refs.extend(source_refs[gate])
            else:raise ValueError('unclassified unknown-noise source dataset: '+dataset)
        return list({d['file']:d for d in refs}.values())
    partition=np.zeros(n,np.int16);groups=[];proofs=[];unknown_rows=0;unknown_groups=0;known_rows=0;known_rank=0
    bundle=out/'canonical_covariance_W.f64';stream=bundle.open('wb');pending=[]
    alias_kinds=Counter();whole=[]
    for g in p['groups']:
        rows=np.asarray(g['rows'],int);keep=active[rows]
        if not keep.any():
            whole.append({'id':g['id'],'original_rows':rows.tolist(),'rank':g['rank'],'parent_W':g['W']});continue
        retained=rows[keep].tolist();stat=g.get('statistic_kind')
        if stat=='initialization_surrogate':
            refs=unknown_source_refs(retained)
            new={**g,'rows':retained,'rank':0,'W':[],'statistic_kind':'constraint_only_unknown_noise',
                'unknown_noise_source_refs':refs,'parent_initialization_W':g['W'],
                'unknown_noise_source_has_statistical_precision':False,
                'E_RS_Query_and_finite_material_constraints_retained':True}
            unknown_rows+=len(retained);unknown_groups+=1
        elif stat in {'original','native_joint_readout','propagated_source_covariance'}:
            new={**g,'rows':retained}
            if not keep.all():
                if stat=='original':raise ValueError('actual-source alias must not modify original S01 groups')
                wd=g['W'];wf=verify(wd)
                W=np.asarray(np.memmap(wf,dtype=wd['dtype'],mode='r',offset=wd.get('offset_bytes',0),shape=tuple(wd['shape'])))
                U,s,V=np.linalg.svd(W,full_matrices=False)
                if not np.all(s>0):raise ValueError('parent supported whitening lost numerical rank')
                F=(V.T/s)@U.T;coords=np.ravel(np.c_[2*np.flatnonzero(keep),2*np.flatnonzero(keep)+1]);selected=F[coords]
                M,proof=supported_factor_whitening(selected)
                # This checks the selected covariance support, not a sliced precision.
                C=selected@selected.T
                e=float(np.max(np.abs(M@C@M.T-np.eye(len(M))),initial=0.))
                if e>1e-7:raise ValueError('exact canonical covariance whitening failed')
                offset=stream.tell();M.astype('<f8').tofile(stream);spec={'file':str(bundle),'sha256':'pending','dtype':'<f8','shape':list(M.shape),'offset_bytes':offset};pending.append(spec)
                new.update(rank=len(M),W=spec)
                proofs.append({'id':g['id'],'parent_W':wd,'parent_rows':rows.tolist(),'canonical_retained_rows':retained,
                    'removed_exact_alias_rows':rows[~keep].tolist(),'parent_rank':g['rank'],'retained_rank':len(M),
                    'W_columns_sliced':False,'full_parent_covariance_marginal':True,'whitening_identity_max_abs':e,'support':proof})
            else:verify(g['W'])
            if stat!='original':known_rows+=len(retained);known_rank+=int(new['rank'])
        elif stat=='constraint_only_unknown_noise':
            if g['rank']!=0 or g['W']!=[] or not g.get('unknown_noise_source_refs'):raise ValueError('invalid inherited unknown-noise source constraint group')
            for d in g['unknown_noise_source_refs']:verify(d)
            new={**g,'rows':retained};unknown_rows+=len(retained);unknown_groups+=1
        else:raise ValueError('unrecognized active statistic kind: '+str(stat))
        alias_kinds[stat]+=int((~keep).sum());groups.append(new);partition[np.asarray(retained,int)]+=1
    stream.close();bd=ref(bundle)
    for d in pending:d['sha256']=bd['sha256']
    independent=[int(i)for i in p.get('independent_rows',[])if active[i]];partition[independent]+=1
    if not np.array_equal(partition,active.astype(np.int16)):raise ValueError('all-source active statistical partition mismatch')
    rank=2*len(independent)+sum(int(g['rank'])for g in groups)
    original_rank=2*sum(i<int(p['new_source_start'])for i in independent)+sum(int(g['rank'])for g in groups if g['statistic_kind']=='original')
    write_json(out/'canonical_alias_membership.json',{'parent_packet':ref(parent),'actual_source_aliases':alias_ref,'rows':aliases,
        'raw_UV_source_clock_material_identity_unchanged':True,'only_independent_consumption_removed':True})
    write_json(out/'covariance_marginal_proofs.json',proofs);write_json(out/'fully_inactive_parent_groups.json',whole)
    p.update(groups=groups,independent_rows=independent,active_rows=int(active.sum()),joint_statistics_rank=rank,
        current_original_W_rank=original_rank,new_source_statistics_rank=known_rank,
        preserved_conditional_statistics_rows=known_rows,preserved_conditional_statistics_rank=known_rank,
        new_source_statistics_status='original S01 and known conditional readout statistics plus all unknown-noise engineering constraints without statistical precision',
        recapture_rows=np.flatnonzero(active&(np.arange(n)>=int(p['new_source_start']))).tolist(),
        parent_all_source_statistical_packet=ref(parent),
        unknown_noise_source_rows_keep_engineering_constraints=True,
        unknown_noise_source_statistical_precision_supplied=False,
        constraint_only_unknown_noise_rows=unknown_rows,constraint_only_unknown_noise_groups=unknown_groups,
        initialization_objective_only=False,formal_new_source_statistical_optimum_claimed=False,
        physical_noise_likelihood_claimed=False,physical_total_error_accuracy_claimed=False,
        canonical_actual_source_alias_membership=ref(out/'canonical_alias_membership.json'),
        covariance_marginal_proof=ref(out/'covariance_marginal_proofs.json'),
        source_mean_source_clock_material_or_four_aux_changed=False,
        statistical_objective_interpretation='known conditional native readout likelihood plus all admitted engineering constraints; unknown noise contributes no statistical precision')
    write_packet(out/'packet.json',p)
    report={'schema':'rinkfit.all_source_constraint_only_packet.v1','packet':ref(out/'packet.json'),'parent_packet':ref(parent),
        'raw_rows':n,'previous_active_rows':int(before.sum()),'active_rows':int(active.sum()),'exact_readout_aliases_not_independently_consumed':len(aliases),
        'alias_rows_by_parent_statistic_kind':dict(alias_kinds),'unknown_noise_constraint_only_rows':unknown_rows,'unknown_noise_constraint_only_groups':unknown_groups,
        'conditional_native_rows':known_rows,'conditional_native_rank':known_rank,'original_S01_rank':original_rank,'total_statistical_rank':rank,
        'partial_full_covariance_marginal_groups':len(proofs),'fully_inactive_parent_groups':len(whole),
        'all_active_rows_exactly_one_statistical_or_unknown_noise_partition':True,'unknown_noise_rows_role_or_E_RS_Query_constraints_removed':False,
        'W_floor_ridge_or_precision_invented':False,'original_S01_W_and_all_unchanged_arrays_shared_by_reference':True,
        'actual_source_refs':source_refs,'statistical_optimum_or_total_physical_noise_calibration_claimed':False}
    write_json(out/'report.json',report)
    return report
