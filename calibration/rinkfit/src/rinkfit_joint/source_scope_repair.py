"""Evidence-specific source membership migration; never a same-problem AL step.

This module does not fit, manufacture source qualification, infer a missing
covariance, or change raw means, Queries, E, finite identities or source clocks.
"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from .source_statistics import supported_factor_whitening


def marginalize_group(group, retained_rows, load_whitener):
    """Select the exact supported covariance, including rank-zero constraints."""
    old=list(map(int,group['rows'])); keep=list(map(int,retained_rows))
    if len(set(old))!=len(old) or len(set(keep))!=len(keep):
        raise ValueError('Repeated source row in statistical group')
    if [x for x in old if x in set(keep)]!=keep:
        raise ValueError('Membership selection must preserve complete source order')
    if not keep:return None,None,dict(method='whole_group_retired',new_rank=0)
    if group.get('statistic_kind')=='constraint_only_unknown_noise':
        if group['rank']!=0 or group['W']!=[]:
            raise ValueError('Unknown-noise constraints must carry rank zero and no W')
        return {**group,'rows':keep},None,dict(method='rank_zero_unknown_noise_membership_only',original_rows=old,retained_rows=keep,new_rank=0,statistical_precision_invented=False,W_columns_sliced=False)
    if keep==old:return copy.deepcopy(group),None,dict(method='unchanged_complete_group',new_rank=int(group['rank']))
    W=np.asarray(load_whitener(group['W']),float)
    if W.ndim!=2 or W.shape!=(int(group['rank']),2*len(old)) or not np.isfinite(W).all():
        raise ValueError('Invalid complete parent group whitener')
    if not len(W):raise ValueError('Unclassified empty whitener')
    U,s,V=np.linalg.svd(W,full_matrices=False)
    if np.any(s<=np.finfo(float).eps*max(W.shape)*s[0]*8):
        raise ValueError('Advertised parent supported rank is not positive')
    # W^+ is a covariance factor. Select rows in C, never columns in W.
    F=(V.T/s)@U.T
    positions=[old.index(i)for i in keep]
    coords=np.asarray([2*j+k for j in positions for k in (0,1)],int)
    selected=F[coords]; C=selected@selected.T
    M,support=supported_factor_whitening(selected)
    whitening_error=float(np.max(abs(M@C@M.T-np.eye(len(M))),initial=0))
    result={**group,'rows':keep,'rank':len(M),'W':None}
    proof=dict(method='recover_supported_covariance_factor_select_source_coordinates_rewhiten',original_rows=old,retained_rows=keep,selected_parent_positions=positions,original_rank=int(group['rank']),new_rank=len(M),W_columns_sliced=False,precision_columns_sliced=False,MC_rerun=False,original_means_Q_normals_unchanged=True,whitening_identity_max_abs_error=whitening_error,support=support)
    return result,M,proof


def map_constraint_values(old_ids,old_values,new_ids):
    """Exact IDs retain duals; new constraints get zero; retired duals are archived."""
    if len(old_ids)!=len(old_values) or len(set(old_ids))!=len(old_ids) or len(set(new_ids))!=len(new_ids):
        raise ValueError('Invalid complete dual schema')
    values=dict(zip(old_ids,old_values)); newset=set(new_ids)
    new=[values.get(key,0.)for key in new_ids]
    retired={key:value for key,value in values.items()if key not in newset}
    return new,dict(retired=retired,new_zero_ids=[key for key in new_ids if key not in values],retained_exact_ids=[key for key in new_ids if key in values],source_membership_changed=True,same_problem_AL_monotonicity_claimed=False)


def map_named_parameters(old,new,retired_row_indices,*,new_row_start):
    """Pure bridge primitive. Actual native snapshot/token must be rebuilt natively.

    Old inactive auxiliaries are retained in explicit full-row history. Only
    parameters belonging to source-withdrawn rows may leave the active registry.
    """
    before={x['name']:x for x in old}; after={x['name']:x for x in new}
    if len(before)!=len(old) or len(after)!=len(new):raise ValueError('Repeated parameter name')
    def row_of(name):
        prefix,sep,tail=name.partition('/')
        return int(tail)if sep and prefix in('station','row','alpha')and tail.isdigit()else None
    retired=set(map(int,retired_row_indices));out=copy.deepcopy(new);added=[];removed=[]
    for x in out:
        prior=before.get(x['name'])
        if prior is not None:
            if len(prior['values'])!=len(x['values'])or prior.get('equality_auxiliary')!=x.get('equality_auxiliary'):
                raise ValueError('Named parameter layout changed: '+x['name'])
            x['values']=copy.deepcopy(prior['values'])
        else:
            index=row_of(x['name'])
            if not(index is not None and index>=new_row_start)and not x['name'].startswith('measurement/paint_half_width/'):
                raise ValueError('Unqualified newly introduced parameter: '+x['name'])
            added.append(x['name'])
    for name,x in before.items():
        if name not in after:
            if row_of(name)not in retired:raise ValueError('Unqualified retired physical parameter: '+name)
            removed.append(copy.deepcopy(x))
    return out,dict(retired_parameter_values=removed,added_numerical_parameter_names=added,all_retained_named_values_exact=True,retired_full_row_aux_must_remain_in_explicit_history=True,native_state_identity_and_token_rebuild_required=True,same_problem_AL_monotonicity_claimed=False)


def withdraw_source_membership(parent_packet,decisions,output,authority_ref):
    """Write one immutable membership revision, preserving every raw observation.

    Authority must already be the closed-source qualification index. This API
    never reactivates a row. It accepts only explicit source keys and per-ID
    proof decisions, and supports both measured W and rank-zero engineering.
    """
    import json
    from .io_utils import file_sha256,write_json
    from .packet_storage import write_packet
    src=Path(parent_packet).resolve();out=Path(output).resolve()
    if out.exists():raise ValueError('Immutable source membership output exists')
    authority=Path(authority_ref.get('file',authority_ref.get('path','')))
    if file_sha256(authority)!=authority_ref['sha256']:raise ValueError('Changed closed-source withdrawal authority')
    auth=json.loads(authority.read_text(encoding='utf8'))
    if auth.get('schema')!='rinkfit.closed_board_scope_assembly_index.v1'or auth.get('status')!='complete':
        raise ValueError('Withdrawal authority is not a closed source-scope qualification')
    packet=json.loads(src.read_text(encoding='utf8'));n=int(packet['row_count']);checked={}
    def locate(d):
        f=Path(d['file']);f=f if f.is_absolute()else src.parent/f;k=(str(f),d['sha256'])
        if k not in checked:
            if file_sha256(f)!=d['sha256']:raise ValueError('Changed parent numerical dependency')
            checked[k]=True
        return f.resolve()
    import sqlite3
    from .io_utils import object_sha256
    for key in ('source_scope','primary_complete_audit','independent_mask_supplement','independent_mask_recipe'):
        r=auth[key];f=Path(r.get('file',r.get('path','')))
        if file_sha256(f)!=r['sha256']:raise ValueError('Changed complete qualification authority '+key)
    import gzip,hashlib
    scope=json.loads(Path(auth['source_scope']['file']).read_text(encoding='utf8'))
    taskref=scope['frame_tasks'];taskfile=Path(taskref['file'])
    if file_sha256(taskfile)!=taskref['sha256']:raise ValueError('Changed original source exposure inventory')
    opener=gzip.open if taskfile.suffix=='.gz'else open
    with opener(taskfile,'rt',encoding='utf8')as stream:
        actual_pts=[int(json.loads(line)['native_pts'])for line in stream if line.strip()]
    actual_unique=sorted(set(actual_pts))
    if not actual_unique or len(actual_pts)!=len(actual_unique)or int(scope['development_exposures'])!=len(actual_unique)or int(auth['native_exposures'])!=len(actual_unique):
        raise ValueError('Closed qualification differs from actual bound source exposure scope')
    pts_digest=hashlib.sha256(np.asarray(actual_unique,dtype='<i8').tobytes()).hexdigest()
    if auth.get('native_pts_sha256')!=pts_digest:raise ValueError('Closed qualification exact native clock coverage differs')
    dbref=auth['database'];dbfile=Path(dbref['file'])
    if file_sha256(dbfile)!=dbref['sha256']:raise ValueError('Changed per-ID qualification index')
    db=sqlite3.connect('file:'+dbfile.as_posix()+'?mode=ro',uri=True)
    arrays={k:{**d,'file':str(locate(d))}for k,d in packet['arrays'].items()}
    def array(name):
        d=arrays[name];return np.memmap(d['file'],dtype=d['dtype'],mode='r',offset=int(d.get('offset_bytes',0)),shape=tuple(d['shape']))
    oldad=array('admission');roles=array('fit_role');pts=array('native_pts');uv=array('uv');ad=np.asarray(oldad).copy();seen=set();records=[]
    for d in decisions:
        i=int(d['joint_row'])
        if i in seen or not 0<=i<n:raise ValueError('Invalid/repeated explicit withdrawal row')
        seen.add(i)
        if i<int(packet['new_source_start']):raise ValueError('Inherited S01 withdrawal needs a separate source-specific authority')
        if packet['source_row_ids'][i]!=d['source_row_id']or packet['source_dataset_ids'][i]!=d['source_dataset_id']or int(pts[i])!=int(d['native_pts']):raise ValueError('Source-key withdrawal redirects raw identity')
        record=db.execute('SELECT data,operator FROM rows WHERE sid=? AND pts=?',(d['source_row_id'],int(d['native_pts']))).fetchone()
        if record is None:raise ValueError('No closed per-ID qualification for requested withdrawal')
        approved=json.loads(record[0])
        for key in ('physical_role_status','operator_status'):
            if d.get(key)!=approved[key]:raise ValueError('Requested withdrawal changes actual reviewed status')
        if d.get('exact_actual_readout_alias_of')!=approved['actual_readout_alias_of']:
            raise ValueError('Requested withdrawal changes actual alias authority')
        if not np.array_equal(np.asarray(approved['source_current_uv'],float),uv[i]):raise ValueError('Requested withdrawal redirects reviewed source mean')
        expected=object_sha256(dict(native_pts=int(pts[i]),current_uv=uv[i].tolist(),source_row_id=d['source_row_id'],measurement_operator=record[1]))
        if expected!=approved['current_measurement_sha256']:raise ValueError('Requested withdrawal redirects reviewed source operator')
        if approved['actual_current_RGB_evidence']not in d.get('evidence_refs',[]):raise ValueError('Requested withdrawal omits actual closed per-ID review')
        if d.get('physical_role_status')=='pass'and d.get('operator_status')=='pass'and not d.get('exact_actual_readout_alias_of'):
            raise ValueError('A dual-pass independent source cannot be withdrawn by this physical qualification operation')
        if not d.get('evidence_refs')or d.get('reason')not in('wrong_role','unresolved_physical_or_operator','exact_actual_readout_alias'):
            raise ValueError('Withdrawal lacks per-ID physical/operator/alias source evidence')
        for r in d['evidence_refs']:
            f=Path(r.get('file',r.get('path','')));f=f if f.is_absolute()else authority.parent/f
            locate(dict(file=str(f),sha256=r['sha256']))
        records.append({**d,'previous_admission':int(ad[i]),'current_admission':0,'fit_role_changed':False});ad[i]=0
    db.close()
    active=(ad==1)&(roles==1);partition=np.zeros(n,np.uint8);independent=[int(i)for i in packet['independent_rows']if active[i]];partition[independent]+=1
    out.mkdir(parents=True);af=out/'admission.bin';ad.tofile(af);arrays['admission']={**arrays['admission'],'file':str(af),'sha256':file_sha256(af),'offset_bytes':0}
    groups=[];proofs=[];pending=[];bundle=out/'marginal_W.f64';rank=2*len(independent)
    with bundle.open('wb')as stream:
        for old in packet['groups']:
            retained=[int(i)for i in old['rows']if active[i]]
            def load(d):return np.memmap(locate(d),dtype=d['dtype'],mode='r',offset=int(d.get('offset_bytes',0)),shape=tuple(d['shape']))
            group,M,proof=marginalize_group(old,retained,load);proofs.append({'group_id':old['id'],**proof})
            if group is None:continue
            if M is not None:
                offset=stream.tell();M.astype('<f8').tofile(stream);spec=dict(file=str(bundle),sha256='pending_closed_bundle',dtype='<f8',shape=list(M.shape),offset_bytes=offset);group['W']=spec;pending.append(spec)
            elif group['W']!=[]:group['W']={**group['W'],'file':str(locate(group['W']))}
            groups.append(group);rank+=int(group['rank']);partition[retained]+=1
    digest=file_sha256(bundle)
    for d in pending:d['sha256']=digest
    if not np.array_equal(partition,active.astype(np.uint8)):raise ValueError('Corrected source groups do not partition exact active source membership')
    original_rank=2*sum(i<int(packet['new_source_start'])for i in independent)+sum(int(g['rank'])for g in groups if all(int(i)<int(packet['new_source_start'])for i in g['rows']))
    result={**packet,'arrays':arrays,'groups':groups,'independent_rows':independent,'active_rows':int(active.sum()),'joint_statistics_rank':rank,'current_original_W_rank':original_rank,'new_source_statistics_rank':rank-original_rank,'recapture_rows':np.flatnonzero(active&(np.arange(n)>=int(packet['new_source_start']))).tolist(),'constraint_only_unknown_noise_rows':sum(len(g['rows'])for g in groups if g.get('statistic_kind')=='constraint_only_unknown_noise'),'constraint_only_unknown_noise_groups':sum(g.get('statistic_kind')=='constraint_only_unknown_noise'for g in groups),'source_membership_revision':dict(parent=dict(file=str(src),sha256=file_sha256(src)),authority=authority_ref,physical_source_withdrawal_is_not_numerical_outlier_rejection=True,raw_source_rows_removed=False,requires_native_complete_state_membership_bridge=True,same_problem_AL_monotonicity_claimed=False),'formal_new_source_statistical_optimum_claimed':False}
    # Its arrays/active coverage changed. An old certificate can never certify it.
    result.pop('board_material_source_qualification',None)
    write_json(out/'source_withdrawals.json',records);write_json(out/'covariance_marginal_proofs.json',proofs)
    result['source_membership_revision']['decisions']=dict(file=str(out/'source_withdrawals.json'),sha256=file_sha256(out/'source_withdrawals.json'))
    write_packet(out/'packet.json',result)
    report=dict(schema='rinkfit.evidence_specific_source_membership_repair.v1',parent_packet=dict(file=str(src),sha256=file_sha256(src)),packet=dict(file=str(out/'packet.json'),sha256=file_sha256(out/'packet.json')),raw_rows=n,previous_active_rows=int(np.sum((oldad==1)&(roles==1))),current_active_rows=int(active.sum()),withdrawn_active_rows=int(np.sum((oldad==1)&(ad==0)&(roles==1))),all_raw_source_mean_Query_E_clock_fit_role_and_four_aux_arrays_unchanged=True,source_rows_removed_from_raw_ledger=False,old_source_covariance_W_files_preserved=True,rank_zero_groups_not_assigned_precision=True,MC_rerun=False,current_CP_or_model_modified=False,native_source_qualification_certificate_required=True,native_membership_state_bridge_required=True,same_problem_AL_monotonicity_claimed=False)
    write_json(out/'report.json',report);return report


def finalize_appended_finite_sources(packet,folder,source_rows,approved_pairs,*,new_source_start,model_ref):
    """Finalize actual frozen Queries for singles and paired flanks after one append.

    This writes only the appended array tails. Station/RS witnesses remain
    explicitly pending until the existing native multi-start initializer runs.
    """
    import json
    from .io_utils import object_sha256,file_sha256,write_json
    folder=Path(folder).resolve();byid={sid:new_source_start+j for j,sid in enumerate(packet['source_row_ids'][new_source_start:])}
    sources={r['source_row_id']:r for r in source_rows}
    if set(byid)!=set(sources)or len(byid)!=len(source_rows):raise ValueError('Appended source identities differ from final frozen qualification')
    queries=list(packet.get('queries',[]));pairs=list(packet.get('stroke_pairs',[]));patches={}
    for rid,r in sources.items():
        i=byid[rid];seed=np.asarray(r['source_profile_query_uv'],float);normal=np.asarray(r['source_profile_normal'],float);bounds=np.asarray(r['source_profile_alpha_bounds'],float);alpha=float(r['source_profile_current_alpha']);uv=np.asarray(r['current_uv'],float)
        if seed.shape!=(2,)or normal.shape!=(2,)or bounds.shape!=(2,)or not np.isfinite(np.r_[seed,normal,bounds,alpha]).all()or abs(np.linalg.norm(normal)-1)>1e-10 or not bounds[0]<=alpha<=bounds[1]or np.linalg.norm(seed+alpha*normal-uv)>1e-8:
            raise ValueError('New actual source is outside its frozen finite Query')
        patches[i]=dict(profile_normal=normal,query_uv=seed,alpha_bounds=bounds,alpha=alpha,stroke_pair_index=-1,stroke_source_side=-1)
        queries.append(dict(row_index=i,alpha=bounds.tolist(),native_uv=[(seed+a*normal).tolist()for a in bounds],normal=normal.tolist(),source_row_id=rid,source_factor_row_id=rid,source_acquisition_window_only=True))
    for entry in approved_pairs:
        pair=entry['pair'];ids=pair['factor_row_ids'];indices=[byid[rid]for rid in ids];mi=int(entry['joint_material_index']);pi=len(pairs);uv=np.asarray(pair['actual_flank_uv'],float)
        if len(ids)!=2 or entry.get('source_identity_eligible')is not True or packet['material_binding']['materials'][mi]['label']!=entry['actual_target_id']:raise ValueError('New paint pair actual material identity differs')
        for side,(i,rid)in enumerate(zip(indices,ids)):
            if sources[rid]['current_uv']!=uv[side].tolist()or int(sources[rid]['joint_material_index'])!=mi:raise ValueError('Pair side order or actual physical mean changed')
            patches[i]['stroke_pair_index']=pi;patches[i]['stroke_source_side']=side
        measured=object_sha256(dict(native_pts=int(pair['native_pts']),source_row_id=pair['id'],actual_flank_uv=uv.tolist(),measurement_operator='finite_width_paint_material_flank_pair'))
        if measured!=pair['source_mean_sha256']or measured!=entry['actual_4D_source_mean_sha256']:raise ValueError('Pair whole4D source mean changed')
        pairs.append(dict(id=pair['id'],rows=indices,material_index=mi,target_id=entry['actual_target_id'],shot_id=pair['shot_id'],native_pts=int(pair['native_pts']),width_scope_id=pair['width_scope_id'],width_scope=pair.get('width_scope','per_actual_transverse_source_section'),allowed_world_side_assignments=pair['allowed_world_side_assignments'],actual_flank_uv=uv.tolist(),source_normal=pair['source_normal'],source_mean_sha256=measured,source_family_id=pair['source_family_id'],source_refs=pair.get('source_refs',[dict(source_pair_id=pair['id'],source_admission_receipt_sha256=entry['source_admission_receipt_sha256'])])))
    for name in ('profile_normal','query_uv','alpha_bounds','alpha','stroke_pair_index','stroke_source_side'):
        d=packet['arrays'][name];a=np.memmap(d['file'],dtype=d['dtype'],mode='r+',offset=int(d.get('offset_bytes',0)),shape=tuple(d['shape']))
        for i,values in patches.items():a[i]=values[name]
        a.flush();del a;d['sha256']=file_sha256(d['file'])
    packet['queries']=queries;packet['stroke_pairs']=pairs;packet['required_measurement_width_model_ref']=model_ref;packet['new_source_genuine_Query_indices']=sorted(patches);packet['new_source_finite_RS_initialization_required']=True;packet['source_placeholder_station_half_is_not_solver_ready']=True
    write_json(folder/'appended_finite_sources_Query_pairs.json',dict(new_source_start=new_source_start,source_rows=len(sources),actual_pairs=len(approved_pairs),query_indices=sorted(patches),old_aux_prefix_changed=False,new_station_RS_witnesses_are_pending=True,source_queries_are_acquisition_windows_not_added_precision=True))
    return packet
