"""Transparent numerical initialization scope; never changes source admission."""
import json
from pathlib import Path
import numpy as np
from .io_utils import file_sha256,write_json,read_jsonl
from .source_statistics import supported_factor_whitening

def bootstrap_scope(parent_packet,recapture,seed,output,*,valid_field="RS_valid",initialization_time_model="implicit_rolling_shutter_root"):
    if valid_field not in {"RS_valid","numerical_initialization_seed_valid"}:raise ValueError("unknown initialization seed predicate")
    if (valid_field=="RS_valid")!=(initialization_time_model=="implicit_rolling_shutter_root"):raise ValueError("seed predicate and time model do not agree")
    parent_path=Path(parent_packet);root=parent_path.parent;packet=json.loads(parent_path.read_text(encoding='utf8'));n=packet['row_count'];start=packet['new_source_start']
    out=Path(output)
    if out.exists():raise ValueError('new bootstrap scope required')
    out.mkdir(parents=True)
    checked={}
    def locate(ref):
        path=Path(ref['file']);path=path if path.is_absolute() else root/path
        key=(str(path.resolve()),ref['sha256']);signature=(path.stat().st_size,path.stat().st_mtime_ns)
        if key in checked and checked[key]!=signature:raise ValueError('parent numerical dependency changed during bootstrap')
        if key not in checked:
            if file_sha256(path)!=ref['sha256']:raise ValueError('changed parent numerical dependency')
            checked[key]=signature
        return path
    arrays={k:{**v,'file':str(locate(v))} for k,v in packet['arrays'].items()}
    old={k:np.fromfile(v['file'],v['dtype']).reshape(v['shape']) for k,v in arrays.items()}
    rec=list(read_jsonl(recapture));rr=np.asarray([r['joint_row'] for r in rec],int)
    expected=np.asarray(packet['recapture_rows'],int)
    if len(np.unique(rr))!=len(rr) or set(rr)!=set(expected):raise ValueError('recapture did not account for every active new source once')
    good=np.asarray([r.get(valid_field) is True for r in rec]);failed=rr[~good];successful=rr[good]
    active=(old['admission']==1)&(old['fit_role']==1);active[failed]=False
    role=old['fit_role'].copy();role[failed]=2
    role_path=out/'fit_role.bin';role.tofile(role_path);arrays['fit_role']={**arrays['fit_role'],'file':role_path.name,'sha256':file_sha256(role_path)}
    seed_path=Path(seed);sd=json.loads(seed_path.read_text(encoding='utf8'))
    for key in ['station','path','row','alpha']:
        ref=sd['arrays'][key];path=Path(ref['file']);path=path if path.is_absolute() else seed_path.parent/path
        if file_sha256(path)!=ref['sha256']:raise ValueError('changed recapture four-auxiliary seed')
        value=np.fromfile(path,ref['dtype']).reshape(ref['shape'])
        if value.shape!=(n,) or value[:start].tobytes()!=old[key][:start].tobytes():
            raise ValueError('recapture changed original s01 auxiliary prefix: '+key)
        arrays[key]={**ref,'file':str(path)}
    groups=[];marginal_proofs=[];rank=2*len(packet['independent_rows']);partition=np.zeros(n,np.int16);partition[np.asarray(packet['independent_rows'],int)]+=1
    for group in packet['groups']:
        ids=np.asarray(group['rows'],int);keep=active[ids]
        if not keep.any():continue
        spec=dict(group['W'])
        if keep.all():spec['file']=str(locate(spec));grank=group['rank']
        else:
            if np.any(ids<start):raise ValueError('bootstrap may not change s01 original W groups')
            path=locate(spec);allw=np.fromfile(path,spec['dtype']);offset=spec.get('offset_bytes',0)//8;W=allw[offset:offset+group['rank']*2*len(ids)].reshape(group['rank'],2*len(ids))
            U,s,V=np.linalg.svd(W,full_matrices=False)
            if np.any(s<=0):raise ValueError('bootstrap parent W lost declared row rank')
            F=(V.T/s)@U.T;coords=np.ravel(np.c_[2*np.flatnonzero(keep),2*np.flatnonzero(keep)+1]);newW,proof=supported_factor_whitening(F[coords])
            name=f'W_marginal_{len(marginal_proofs):05d}.f64';newW.astype('<f8').tofile(out/name)
            spec={'file':name,'dtype':'<f8','shape':list(newW.shape),'sha256':file_sha256(out/name)};grank=len(newW)
            marginal_proofs.append({'id':group['id'],'original_W_sha256':group['W']['sha256'],'source_rows':ids.tolist(),'retained_rows':ids[keep].tolist(),
                'marginal_covariance_not_sliced_W':True,'original_rank':group['rank'],'new_rank':grank,'proof':proof})
        group={**group,'rows':ids[keep].tolist(),'W':spec,'rank':grank};groups.append(group);rank+=grank;partition[ids[keep]]+=1
    if not np.array_equal(partition,active.astype(np.int16)):raise ValueError('bootstrap source W partition differs from temporary fit role')
    result={**packet,'arrays':arrays,'groups':groups,'active_rows':int(active.sum()),'admitted_rows':int(np.sum(old['admission']==1)),
        'joint_statistics_rank':rank,'new_source_statistics_rank':rank-packet['s01_original_W_rank'],
        'recapture_rows':successful.tolist(),'temporary_initialization_not_projectable_rows':failed.tolist(),
        'temporary_source_scope':True,'source_admission_changed':False,'source_exclusions_added':False,
        'numerical_seed_failure_does_not_reject_sources':True,'initialization_seed_validation_field':valid_field,
        'initialization_time_model':initialization_time_model,'implicit_RS_validity_claimed_for_observed_time_initialization':False,
        'failed_RS_rows_are_not_rejected_sources':valid_field=='RS_valid','restore_parent_full_source_packet_after_initialization':{'file':str(parent_path),'sha256':file_sha256(parent_path)},
        'initialization_recapture':{'file':str(Path(recapture)),'sha256':file_sha256(recapture)},
        'four_auxiliary_seed':{'file':str(seed_path),'sha256':file_sha256(seed_path)},
        'formal_new_source_statistical_optimum_claimed':False}
    write_json(out/'packet.json',result);write_json(out/'marginal_covariance_proofs.json',marginal_proofs)
    report={'schema':'rinkfit.temporary_joint_initialization_scope.v1','raw_rows':n,'source_admitted_rows':result['admitted_rows'],
        'temporary_fit_rows':int(active.sum()),'successful_new_seed_rows':len(successful),'temporarily_invalid_seed_rows':len(failed),
        'seed_validation_field':valid_field,'initialization_time_model':initialization_time_model,
        'successful_new_RS_rows':len(successful) if valid_field=='RS_valid' else None,'not_projectable_new_rows':len(failed),
        's01_original_W_and_auxiliaries_unchanged':True,'exact_covariance_marginal_groups':len(marginal_proofs),'source_exclusions_added':0,
        'joint_statistics_rank':rank,'packet_sha256':file_sha256(out/'packet.json')}
    write_json(out/'report.json',report);return report
