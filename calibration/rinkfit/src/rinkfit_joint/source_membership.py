"""Withdraw source membership without deleting raw observations or changing UV.

Physical/definition withdrawal changes admission. Numerical initializer holding
is a separate fit_role mechanism and is never conflated with this operation.
"""
from pathlib import Path
from .packet_storage import write_packet
import json
import numpy as np
from .io_utils import file_sha256,write_json
from .source_statistics import supported_factor_whitening

def withdraw_source_rows(parent_packet,decisions,output,inherited_source_authority=None):
    src=Path(parent_packet);p=json.loads(src.read_text(encoding='utf8'));out=Path(output)
    authority=None;authorized_inherited=set()
    if inherited_source_authority is not None:
        ap=Path(inherited_source_authority);authority=json.loads(ap.read_text(encoding='utf8'))
        if authority.get('schema')!='rinkfit.evidence_specific_inherited_source_withdrawal_authority.v1' or authority.get('preserve_original_historical_state_and_W_files')is not True:raise ValueError('evidence-specific historical source authority required')
        authorized_inherited={int(x)for x in authority['joint_rows']}
        authority={'file':str(ap.resolve()),'sha256':file_sha256(ap),'scope':authority['scope']}

    if out.exists():raise ValueError('new source membership generation required')
    out.mkdir(parents=True);n=p['row_count'];start=p['new_source_start'];byid={rid:i for i,rid in enumerate(p['source_row_ids'])};checked={}
    def locate(ref):
        f=Path(ref['file']);f=f if f.is_absolute() else src.parent/f;k=(str(f),ref['sha256'])
        if k not in checked:
            if file_sha256(f)!=ref['sha256']:raise ValueError('changed source numerical dependency')
            checked[k]=True
        return f
    arrays={k:{**v,'file':str(locate(v))} for k,v in p['arrays'].items()};ad=np.fromfile(arrays['admission']['file'],arrays['admission']['dtype']);role=np.fromfile(arrays['fit_role']['file'],arrays['fit_role']['dtype']);before=ad.copy();records=[]
    for d in decisions:
        rid=d['source_row_id'];i=int(d['joint_row'])if 'joint_row'in d else byid.get(rid)
        if i is not None and (i<0 or i>=n or p['source_row_ids'][i]!=rid):raise ValueError('source withdrawal immutable row identity conflict')
        if i is not None and 'native_pts'in d:
            desc=arrays['native_pts'];native=np.memmap(desc['file'],dtype=desc['dtype'],mode='r',shape=(n,))
            if int(native[i])!=int(d['native_pts']):raise ValueError('source withdrawal native clock identity conflict')
        rec={**d,'joint_row':i,'previous_admission':None if i is None else int(ad[i]),'fit_role_changed':False}
        if i is not None:
            if i<start and i not in authorized_inherited:raise ValueError('s01 source membership requires its own evidence-specific authority audit')
            ad[i]=0;rec['current_admission']=0
        records.append(rec)
    af=out/'admission.bin';ad.tofile(af);arrays['admission']={**arrays['admission'],'file':af.name,'sha256':file_sha256(af)}
    active=(ad==1)&(role==1);groups=[];proofs=[];whole_group_withdrawals=[];bundle=out/'marginal_geometry_W.f64';stream=bundle.open('wb');pending_refs=[]
    partition=np.zeros(n,np.int16);independent=[r for r in p['independent_rows'] if active[r]];partition[independent]+=1;rank=2*len(independent)
    for g in p['groups']:
        rows=np.asarray(g['rows'],int);keep=active[rows]
        if not keep.any():
            whole_group_withdrawals.append(dict(id=g['id'],rows=rows.tolist(),rank=g['rank'],parent_W=g['W'],reason='all member source identities withdrawn; full source W file preserved'))
            continue
        ref={**g['W'],'file':str(locate(g['W']))}
        if keep.all():new=g.copy();new['W']=ref
        else:
            if any(int(i)not in authorized_inherited for i in rows[~keep]if i<start):raise ValueError('unexpected s01 partial statistical mutation')
            W=np.memmap(ref['file'],dtype=ref['dtype'],mode='r',offset=ref.get('offset_bytes',0),shape=tuple(ref['shape']))
            if g.get('original_numerical_scheme')=='fixed_group_and_shot_native_pixel_geometry':
                scale=float(g['geometry_weight']['scale']);expected=np.eye(2*len(rows))*scale
                if not np.array_equal(W,expected):raise ValueError('geometric covariance special case lacks exact diagonal proof')
                M=np.eye(2*int(keep.sum()),dtype='<f8')*scale
                proof={'derivation':'C=(1/scale^2)I; exact selected covariance submatrix; W_marginal=scale*I','scale':scale,'source_scale_reference_counts_unchanged':True}
            else:
                U,s,V=np.linalg.svd(W,full_matrices=False);F=(V.T/s)@U.T;coords=np.ravel(np.c_[2*np.flatnonzero(keep),2*np.flatnonzero(keep)+1]);M,proof=supported_factor_whitening(F[coords])
            offset=stream.tell();M.astype('<f8').tofile(stream);spec={'file':bundle.name,'sha256':'pending_closed_bundle','dtype':'<f8','shape':list(M.shape),'offset_bytes':offset};pending_refs.append(spec)
            new={**g,'rows':rows[keep].tolist(),'rank':len(M),'W':spec};proofs.append({'id':g['id'],'original_rows':rows.tolist(),'retained_rows':rows[keep].tolist(),'original_W_sha256':g['W']['sha256'],'W_columns_sliced':False,'proof':proof})
        groups.append(new);rank+=new['rank'];partition[np.asarray(new['rows'],int)]+=1
    stream.close();digest=file_sha256(bundle)
    for spec in pending_refs:spec['sha256']=digest
    if not np.array_equal(partition,active.astype(np.int16)):raise ValueError('corrected full W partition does not match corrected source membership')
    current_original_rank=2*sum(i<start for i in independent)+sum(g['rank']for g in groups if all(i<start for i in g['rows']))
    result={**p,'current_original_W_rank':current_original_rank,'arrays':arrays,'groups':groups,'independent_rows':independent,'active_rows':int(active.sum()),'joint_statistics_rank':rank,
        'new_source_statistics_rank':rank-current_original_rank,'recapture_rows':np.flatnonzero(active&(np.arange(n)>=start)).tolist(),
        'source_membership_revision':{'parent':{'file':str(src),'sha256':file_sha256(src)},'decisions':'source_withdrawals.json','reason':'existing user measurement definitions and actual RGB source proof take priority over previous numeric/promotion receipts','physical_source_withdrawal_is_not_numerical_RS_failure':True,'inherited_source_authority':authority},
        'formal_new_source_statistical_optimum_claimed':False,'physical_source_audit_complete':False,'s01_inherited_definition_and_RGB_audit_pending':True}
    write_json(out/'source_withdrawals.json',records);result['source_membership_revision']['decisions_sha256']=file_sha256(out/'source_withdrawals.json')
    write_packet(out/'packet.json',result);write_json(out/'marginal_covariance_proofs.json',proofs);write_json(out/'whole_group_withdrawals.json',whole_group_withdrawals)
    report={'schema':'rinkfit.source_membership_correction.v1','raw_rows':n,'previous_active_rows':int(np.sum((before==1)&(role==1))),'corrected_active_rows':int(active.sum()),'source_rows_withdrawn_from_previous_active':int(np.sum((before==1)&(ad==0)&(role==1))),
        'source_mean_source_clock_all_aux_or_fit_role_changed':False,'original_s01_W_file_unchanged':True,'original_s01_joint_likelihood_membership_changed':bool(authorized_inherited),'current_original_W_rank':current_original_rank,'whole_statistical_groups_withdrawn':len(whole_group_withdrawals),'partial_covariance_marginal_groups':len(proofs),'packet_sha256':file_sha256(out/'packet.json'),'definition_or_RGB_pending_is_not_prediction_outlier_rejection':True}
    write_json(out/'report.json',report);return report
