"""Bind usable same-mean conditional source statistics to an existing packet.

Statistically pending rows retain source admission and are held by fit role.
Normal-only readouts get their actual deterministic finite normal Query, never
an artificial tangential covariance. Original historical source rows stay fixed.
"""
from pathlib import Path
from .packet_storage import write_packet
import json
import numpy as np
from .io_utils import read_jsonl,write_json,file_sha256,object_sha256
from .source_statistics import supported_factor_whitening


def bind_outer_conditional_statistics(parent_packet,statistics_registry,solver_rows,source_journal,gate_manifest,output,*,preserve_existing_new_statistics=False,mixed_finite_source_queries=False,require_statistics_only_update=False,source_recall_indexes=None):
    src=Path(parent_packet).resolve();out=Path(output).resolve()
    if out.exists():raise ValueError('new immutable conditional packet generation required')
    for name in ('SOURCE_IDENTITY_CERTIFICATE_RECALLED.json','SOURCE_IDENTITY_EVIDENCE_RECALL.json'):
        if (Path(gate_manifest).resolve().parent/name).exists():raise ValueError('Source admission certificate was recalled: '+str(Path(gate_manifest).resolve().parent/name))
    recalled=set();recall_refs=[]
    for item in (source_recall_indexes or []):
        f=Path(item.get('file',item.get('path'))).resolve()
        if file_sha256(f)!=item['sha256']:raise ValueError('Source qualification recall evidence changed')
        m=json.loads(f.read_text(encoding='utf8'));ref=m['membership'];mf=Path(ref['file'])
        if file_sha256(mf)!=ref['sha256']:raise ValueError('Source recall row membership changed')
        for row in read_jsonl(mf):
            if row.get('old_source_qualification_withdrawn'):recalled.update(row.get('old_factor_row_ids',[row['source_row_id']]))
        recall_refs.append(dict(file=str(f),sha256=item['sha256']))
    p=json.loads(src.read_text(encoding='utf8'));registry=json.loads(Path(statistics_registry).read_text(encoding='utf8'));statistics_ref=dict(file=str(Path(statistics_registry).resolve()),sha256=file_sha256(statistics_registry));cache={}
    def locate(ref):
        f=Path(ref['file']);f=f if f.is_absolute()else src.parent/f;k=(str(f),ref['sha256'])
        if k not in cache:
            if file_sha256(f)!=ref['sha256']:raise ValueError('changed parent numeric dependency')
            cache[k]=True
        return f.resolve()
    arrays={k:{**v,'file':str(locate(v))}for k,v in p['arrays'].items()}
    def arr(name):
        v=arrays[name];return np.fromfile(v['file'],v['dtype']).reshape(v['shape'])
    ad=arr('admission');role=arr('fit_role');original_role=role.copy();pts=arr('native_pts');uv=arr('uv');mat=arr('material_index');alpha=arr('alpha');bounds=arr('alpha_bounds');start=p['new_source_start'];N=p['row_count'];ids=p['source_row_ids'];byid={}
    qualified={r['source_row_id']:r for r in read_jsonl(solver_rows)};needed=set(qualified);journal={}
    # A raw SID can survive several immutable source generations. Inactive
    # historical copies must not shadow the one admitted current measurement.
    # Match the exact clock/mean/material and require a unique current version;
    # this selects existing admission, it never revives an older raw row.
    sid_last={};source_version_selections=[]
    for i,rid in enumerate(ids[start:],start):
        if rid not in needed:continue
        sid_last[rid]=i;q=qualified[rid]
        if ad[i]!=1 or (require_statistics_only_update and role[i]!=1):continue
        if int(pts[i])!=int(q['native_pts'])or not np.array_equal(uv[i],np.asarray(q['current_uv']))or int(mat[i])!=int(q['joint_material_index']):continue
        if rid in byid:raise ValueError('Conditional source has multiple admitted exact measurement versions: '+rid)
        byid[rid]=i
    for rid,i in byid.items():
        if sid_last[rid]!=i:source_version_selections.append(dict(source_row_id=rid,selected_existing_active_index=i,inactive_later_raw_index=sid_last[rid],native_pts=int(pts[i]),source_dataset_id=(p['source_dataset_ids'][i]if 'source_dataset_ids'in p else None)))
    if set(byid)!=needed:raise ValueError('Conditional source has no admitted exact mean/clock/material version: '+str(sorted(needed-set(byid))[:5]))
    if needed&recalled:raise ValueError('Conditional statistics cannot restore a recalled source qualification')
    if any(rid in recalled and ad[i]==1 and role[i]==1 for i,rid in enumerate(ids)):raise ValueError('Current source packet still consumes a recalled source row')
    # The admission receipt binds the complete immutable source row. Keep only
    # fields actually needed for this same-mean numeric/query comparison.
    journal_fields=('source_row_id','native_pts','source_uv','current_uv','joint_material_id','measurement_operator_resolved','normal','navigation_seed_uv','operator_kind','frozen_source_query','kind','entity_station_fixed')
    for r in read_jsonl(source_journal):
        if r['source_row_id']in needed:
            minimal={key:r[key]for key in journal_fields if key in r}
            if 'source_measurement_audit'in r and 'profile'in r['source_measurement_audit']:
                profile=r['source_measurement_audit']['profile'];minimal['source_measurement_audit']={'profile':{key:profile[key]for key in ('channel','normal_offsets')if key in profile}}
            journal[r['source_row_id']]=minimal
    if set(journal)!=needed:raise ValueError('formal solver gate references missing actual source')
    accepted={};held=[];fixed_entity_arrays={}
    for rid,g in qualified.items():
        if g.get('statistical_objective_eligible')is not True or g.get('solver_eligible')is not True or g.get('numeric_statistics_ready')is not True or g.get('statistic_kind')!='native_joint_readout':raise ValueError('source gate did not sign conditional statistical objective eligibility')
        if rid not in byid:raise ValueError('conditional readout must already exist in full source packet')
        i=byid[rid];r=journal[rid]
        if ad[i]!=1:raise ValueError('conditional statistics cannot revive a physically withdrawn source')
        for key in ('native_pts','source_uv','current_uv','joint_material_id','measurement_operator_resolved'):
            if g[key]!=r[key]:raise ValueError('formal statistical source gate redirects same-mean source '+key)
        if int(pts[i])!=int(r['native_pts'])or not np.array_equal(uv[i],np.asarray(r['current_uv']))or int(mat[i])!=int(g['joint_material_index']):raise ValueError('existing packet differs from exact source mean/clock/material')
        if mixed_finite_source_queries:
            if r.get('operator_kind')=='physical_corner_2d':
                if not fixed_entity_arrays:fixed_entity_arrays.update(path=arr('path'),kind=arr('kind'))
                path=int(fixed_entity_arrays['path'][i]);mi=int(mat[i]);point=p['material_binding']['materials'][mi]['paths'][path]
                if int(fixed_entity_arrays['kind'][i])!=1 or not g.get('entity_station_fixed')or not point.get('physical_point')or r.get('frozen_source_query'):raise ValueError('Fixed physical entity was replaced by a sliding Query')
            elif r.get('operator_kind')not in ('continuous_cubic_point','paint_flank')or not r.get('frozen_source_query'):raise ValueError('mixed source needs its exact qualified finite Query')
        elif r['source_measurement_audit']['profile']['channel']!=2:raise ValueError('outer conditional adapter expects actual frozen colour operator')
        accepted[rid]=i
    accepted_indices=set(accepted.values())
    original_active=(ad==1)&(role==1)
    if not preserve_existing_new_statistics:role[np.arange(N)>=start]=2
    for i in accepted.values():role[i]=1
    active=(ad==1)&(role==1);groups=[];partition=np.zeros(N,np.int16);independent=[int(i)for i in p['independent_rows']if active[i]];partition[independent]+=1;rank=2*len(independent);original_rank=rank
    for g in p['groups']:
        rows=np.asarray(g['rows'],int)
        if np.all(rows<start):
            if not active[rows].all():raise ValueError('inherited packet does not already reflect historical source membership')
            groups.append({**g,'W':{**g['W'],'file':str(locate(g['W']))}});rank+=g['rank'];original_rank+=g['rank'];partition[rows]+=1
        elif np.any(rows<start):raise ValueError('mixed historical and new statistical group')
        elif preserve_existing_new_statistics:
            overlap=np.asarray([int(i)in accepted_indices for i in rows])
            if overlap.any():
                if not overlap.all():
                    if g.get('statistic_kind')!='constraint_only_unknown_noise':raise ValueError('conditional replacement needs exact marginal of any remaining old group members')
                    remaining=rows[~overlap]
                    if g['rank']!=0 or g['W']!=[] or not g.get('unknown_noise_source_refs'):raise ValueError('invalid unknown-noise constraint-only source group')
                    groups.append({**g,'rows':remaining.tolist()});partition[remaining]+=1
                continue
            if not active[rows].all():raise ValueError('preserved source group is not entirely active')
            if g.get('statistic_kind')=='constraint_only_unknown_noise':
                if g['rank']!=0 or g['W']!=[] or not g.get('unknown_noise_source_refs'):raise ValueError('unknown-noise constraints must not acquire invented precision')
                groups.append(g.copy());partition[rows]+=1
            else:
                groups.append({**g,'W':{**g['W'],'file':str(locate(g['W']))}});rank+=g['rank'];partition[rows]+=1
    out.mkdir(parents=True)
    version_QA=dict(schema='rinkfit.unique_existing_admitted_measurement_version_selection.v1',qualified_rows=len(needed),unique_exact_active_versions=len(byid),last_SID_entry_shadowed_active_versions=len(source_version_selections),shadowed_version_selections=source_version_selections,source_admission_mean_clock_material_or_row_order_changed=False)
    write_json(out/'existing_source_version_selection_QA.json',version_QA)
    bundle=out/'conditional_marginal_W.f64';marginals=[];pending=[];covered=set()
    with bundle.open('wb')as stream:
        for gid,g in registry.items():
            rows=g['row_ids'];keep=np.asarray([rid in accepted for rid in rows]);kept=[rid for rid,k in zip(rows,keep)if k]
            if not kept:continue
            if g.get('statistic_kind')!='native_joint_readout'or not g.get('numeric_whitening_verified')or g.get('actual_mean_replay_max_px',float('inf'))>1e-9:raise ValueError('conditional source group is not numerically same-mean verified')
            if any(rid in covered for rid in kept):raise ValueError('actual source readout counted in multiple statistical groups')
            for rid in kept:
                r=journal[rid];q=qualified[rid]
                if q['statistical_group_id']!=gid:raise ValueError('formal source gate group identity conflict')
                source=object_sha256(dict(native_pts=r['native_pts'],source_uv=r['source_uv'],source_row_id=rid));current=object_sha256(dict(native_pts=r['native_pts'],current_uv=r['current_uv'],source_row_id=rid,measurement_operator=r['measurement_operator_resolved']))
                if g['source_measurements_sha256'][rid]!=source or g['current_measurements_sha256'][rid]!=current:raise ValueError('conditional W uses different actual source mean/operator')
            ref=g['W'];f=Path(ref['file'])
            if not f.is_absolute():f=Path(statistics_registry).resolve().parent/f
            if file_sha256(f)!=ref['sha256']:raise ValueError('conditional W bytes changed')
            W=np.memmap(f,dtype=ref['dtype'],mode='r',offset=ref.get('offset_bytes',0),shape=tuple(ref['shape']))
            if W.shape!=(g['rank'],2*len(rows)):raise ValueError('source covariance column order must be row0.u,v row1.u,v')
            if keep.all():spec={**ref,'file':str(f.resolve())};newrank=g['rank']
            else:
                U,s,V=np.linalg.svd(W,full_matrices=False)
                if np.any(s<=0):raise ValueError('source W contains unsupported advertised rows')
                F=(V.T/s)@U.T;coords=np.ravel(np.c_[2*np.flatnonzero(keep),2*np.flatnonzero(keep)+1]);M,proof=supported_factor_whitening(F[coords]);offset=stream.tell();M.astype('<f8').tofile(stream);spec=dict(file=str(bundle),sha256='pending_closed_bundle',dtype='<f8',shape=list(M.shape),offset_bytes=offset);pending.append(spec);newrank=len(M);marginals.append(dict(group_id=gid,parent_W={**ref,'file':str(f)},original_row_ids=rows,retained_row_ids=kept,source_covariance_selected_exactly=True,W_columns_sliced=False,proof=proof))
            indices=[accepted[rid]for rid in kept];groups.append(dict(id=gid,source_group_id=gid,rows=indices,rank=newrank,W=spec,statistic_kind='native_joint_readout',original_numerical_scheme=g.get('original_numerical_scheme','same whole-native RGB nonlinear component covariance and supported whitening'),conditional_model=g['conditional_model'],noise_calibration_status=g.get('noise_calibration_status','explicit conditional encoded-RGB model; physical total error uncalibrated'),physical_total_error_calibrated=False,source_statistics_ref=statistics_ref))
            rank+=newrank;partition[indices]+=1;covered.update(kept)
    digest=file_sha256(bundle)
    for spec in pending:spec['sha256']=digest
    if covered!=needed or not np.array_equal(partition,active.astype(np.int16)):raise ValueError('formal conditional statistics do not partition the active source exactly')
    queries=list(p.get('queries',[]));existing_queries={q['row_index']:q for q in queries};queries_added=0;queries_reused=0
    for rid,i in accepted.items():
        r=journal[rid]
        if r.get('operator_kind')=='physical_corner_2d':
            if i in existing_queries:raise ValueError('A fixed physical entity cannot gain a profile Query')
            continue
        if mixed_finite_source_queries:
            frozen=r['frozen_source_query'];normal=np.asarray(frozen['normal'],float);lo,hi=map(float,frozen['alpha']);native_uv=np.asarray(frozen['native_uv'],float);seed=native_uv[0]-lo*normal
            if not lo<hi or native_uv.shape!=(2,2)or np.linalg.norm(native_uv[1]-(seed+hi*normal))>1e-9:raise ValueError('invalid actual finite source Query geometry')
        else:
            normal=np.asarray(r['normal'],float);seed=np.asarray(r['navigation_seed_uv'],float);off=np.asarray(r['source_measurement_audit']['profile']['normal_offsets'],float);lo=float(off.min());hi=float(off.max());native_uv=np.asarray([seed+lo*normal,seed+hi*normal])
            if lo != -3.0 or hi != 3.0:raise ValueError('actual source query differs from frozen acquisition window')
        if abs(np.linalg.norm(normal)-1)>1e-9:raise ValueError('actual source Query normal is not unit')
        expected=dict(row_index=i,alpha=[lo,hi],native_uv=native_uv.tolist(),normal=normal.tolist(),source_row_id=rid,actual_acquisition_window_preserved=True,deterministic_query_is_not_tangential_noise_precision=True)
        if i in existing_queries:
            old_query=existing_queries[i]
            if any(old_query.get(key)!=expected[key]for key in ('alpha','native_uv','normal'))or old_query.get('source_row_id',old_query.get('source_factor_row_id'))!=rid:raise ValueError('replacement statistics redirects an existing frozen source Query')
            if not(old_query.get('actual_acquisition_window_preserved')or old_query.get('source_acquisition_window_only')):raise ValueError('existing Query has no actual source window authority')
            queries_reused+=1;continue
        a=float(np.dot(uv[i]-seed,normal));alpha[i]=a;bounds[i]=[lo,hi];queries.append(expected);queries_added+=1
    for name,value in [('fit_role',role),('alpha',alpha),('alpha_bounds',bounds)]:
        original=np.memmap(arrays[name]['file'],dtype=arrays[name]['dtype'],mode='r',shape=tuple(arrays[name]['shape']))
        if np.array_equal(value,original):continue
        f=out/(name+'.bin');value.tofile(f);arrays[name]={**arrays[name],'file':str(f),'sha256':file_sha256(f)}
    refs=[dict(file=str(Path(x).resolve()),sha256=file_sha256(x))for x in (statistics_registry,solver_rows,source_journal,gate_manifest)]+recall_refs
    result={**p,'arrays':arrays,'groups':groups,'independent_rows':independent,'queries':queries,'active_rows':int(active.sum()),'joint_statistics_rank':rank,'current_original_W_rank':original_rank,'new_source_statistics_rank':rank-original_rank,'recapture_rows':(p['recapture_rows']if preserve_existing_new_statistics else sorted(accepted.values())),'initialization_objective_only':any(g.get('statistic_kind')=='initialization_surrogate'for g in groups),'physical_noise_likelihood_claimed':True,'physical_noise_model_is_explicitly_conditional':True,'physical_total_error_accuracy_claimed':False,'formal_new_source_statistical_optimum_claimed':False,'new_source_statistics_status':('mixed unknown-noise source geometry initialization and same-mean nonlinear native RGB conditional covariance'if preserve_existing_new_statistics else 'usable same-mean whole-native-exposure nonlinear encoded-RGB conditional covariance; broader total physical error uncalibrated'),'conditional_source_objective_is_total_physical_noise_model':False,'source_noise_statistics_supplied_for_appended_rows':True,'conditional_statistics_parent':dict(file=str(src),sha256=file_sha256(src)),'conditional_statistics_gate_refs':refs,'conditional_statistics_held_new_rows':int(np.sum((ad==1)&(np.arange(N)>=start)&(role==2))),'new_role2_reason':('current role states retained'if preserve_existing_new_statistics else 'source geometry admission retained; numerical statistics pending for this formal objective'),'source_admission_source_UV_current_UV_native_clock_or_original_S01_aux_changed':False,'new_source_query_alpha_initialization_changed':bool(queries_added)}
    result['source_qualification_recall_refs']=recall_refs
    result['source_manifests']=p.get('source_manifests',[])+refs
    result['constraint_only_unknown_noise_rows']=sum(len(g['rows'])for g in groups if g.get('statistic_kind')=='constraint_only_unknown_noise')
    result['constraint_only_unknown_noise_groups']=sum(g.get('statistic_kind')=='constraint_only_unknown_noise'for g in groups)
    result['source_noise_statistics_supplied_for_appended_rows']=result['constraint_only_unknown_noise_rows']==0
    result['new_source_rows_with_conditional_W']=sum(len(g['rows'])for g in groups if g.get('statistic_kind')=='native_joint_readout')
    if result['constraint_only_unknown_noise_rows']:
        result['unknown_noise_source_rows_keep_engineering_constraints']=True;result['unknown_noise_source_statistical_precision_supplied']=False
        result['new_source_statistics_status']='same-mean conditional native statistics plus all unknown-noise source engineering constraints; no invented unknown-noise precision'
    if require_statistics_only_update:
        changed=[key for key,desc in arrays.items()if any(desc[k]!=p['arrays'][key][k]for k in ('sha256','dtype','shape'))]
        structural=('queries','source_row_ids','source_dataset_ids','source_family_ids','allowed_paths','source_station_domains','stroke_pairs','material_binding')
        mismatched=[key for key in structural if result.get(key)!=p.get(key)]
        if changed or mismatched or queries_added or not np.array_equal(original_role,role)or int(active.sum())!=p['active_rows']:
            raise ValueError('Statistics-only update changes source/Query/state identity '+str((changed,mismatched,queries_added)))
        identity=dict(schema='rinkfit.fixed_source_scope_statistics_only_update_QA.v1',passed=True,parent_packet=dict(file=str(src),sha256=file_sha256(src)),
            raw_rows=N,active_rows=int(active.sum()),all_original_numeric_array_digests_shapes_dtypes_unchanged=True,changed_numeric_arrays=changed,
            full_Query_objects_and_sourceIDs_roles_material_binding_station_domains_stroke_pairs_unchanged=True,changed_structural_fields=mismatched,
            new_queries_added=queries_added,old_queries_reused_unchanged=queries_reused,source_native_UV_PTS_E_4aux_unchanged=True,
            source_statistical_group_and_refs_replacement_only=True,current_actual_checkpoint_primal_dual_should_be_inherited=True,
            no_full_source_or_nuisance_recapture_needed_for_statistics_replacement=True,original_S01_statistical_rank=original_rank,
            mixed_same_exposure_W_crossCov_not_split=True,parent_full_covariance_marginalization_not_W_column_slicing=True)
        write_json(out/'statistics_only_identity_QA.json',identity)
        result['source_statistical_group_update_only']=True
        result['statistics_only_identity_QA']=dict(file=str(out/'statistics_only_identity_QA.json'),sha256=file_sha256(out/'statistics_only_identity_QA.json'))
        result['statistical_replacement_requires_source_or_nuisance_recapture']=False
    write_packet(out/'packet.json',result);write_json(out/'marginal_covariance_proofs.json',marginals)
    report=dict(schema='rinkfit.formal_conditional_outer_statistics_packet.v1',raw_rows=N,source_admission_unchanged=True,current_native_UV_unchanged=True,active_rows=int(active.sum()),conditional_new_active_rows=len(accepted),conditional_new_rank=sum(g['rank']for g in groups if g.get('statistic_kind')=='native_joint_readout'and all(i>=start for i in g['rows'])),total_new_objective_rank=rank-original_rank,unknown_noise_geometry_rank=sum(g['rank']for g in groups if g.get('statistic_kind')=='initialization_surrogate'),held_new_source_qualified_rows=result['conditional_statistics_held_new_rows'],current_S01_rank=original_rank,physical_total_noise_calibrated=False,initialization_surrogate_groups_in_objective=sum(g.get('statistic_kind')=='initialization_surrogate'for g in groups),preserved_existing_new_statistics=preserve_existing_new_statistics,actual_source_queries_added=queries_added,actual_source_queries_reused_unchanged=queries_reused,mixed_actual_finite_sources=mixed_finite_source_queries,source_window_expanded_for_prediction=False,statistical_W_columns_sliced=False,partial_covariance_marginal_groups=len(marginals),packet_sha256=file_sha256(out/'packet.json'))
    write_json(out/'report.json',report);return report


def retain_conditional_noise_objective(parent_packet,output):
    """Hold unknown-noise geometry rows while keeping their physical admission.

    Original S01 statistics and every source-qualified conditional native group
    are retained. A role hold is not a source rejection or an observed UV edit.
    """
    src=Path(parent_packet).resolve();out=Path(output).resolve()
    if out.exists():raise ValueError('new immutable conditional objective generation required')
    p=json.loads(src.read_text(encoding='utf8'));arrays={};checked={}
    def resolve(v):
        f=Path(v['file']);f=f if f.is_absolute()else src.parent/f;k=(str(f),v['sha256'])
        if k not in checked:
            if file_sha256(f)!=v['sha256']:raise ValueError('changed current conditional dependency')
            checked[k]=True
        return f.resolve()
    for k,v in p['arrays'].items():arrays[k]={**v,'file':str(resolve(v))}
    def load(k):
        v=arrays[k];return np.fromfile(v['file'],v['dtype']).reshape(v['shape'])
    ad=load('admission');role=load('fit_role');start=p['new_source_start'];groups=[];held=[]
    for g in p['groups']:
        if g.get('statistic_kind')=='initialization_surrogate':
            if any(i<start for i in g['rows']):raise ValueError('cannot hold original S01 source as new unknown-noise geometry')
            held.extend(g['rows']);role[g['rows']]=2
        else:
            if any(i>=start for i in g['rows'])and g.get('statistic_kind')!='native_joint_readout':raise ValueError('new source statistics are not explicitly conditional native readouts')
            groups.append({**g,'W':{**g['W'],'file':str(resolve(g['W']))}})
    active=(ad==1)&(role==1);part=np.zeros(len(ad),np.int16);independent=[i for i in p['independent_rows']if active[i]];part[independent]+=1;rank=2*len(independent)
    for g in groups:
        if not active[g['rows']].all():raise ValueError('current supported statistical group is not fully active')
        part[g['rows']]+=1;rank+=g['rank']
    if not np.array_equal(part,active.astype(np.int16)):raise ValueError('conditional objective does not partition active readouts')
    out.mkdir(parents=True);f=out/'fit_role.bin';role.tofile(f);arrays['fit_role']={**arrays['fit_role'],'file':str(f),'sha256':file_sha256(f)};conditional_rank=rank-p['current_original_W_rank'];result={**p,'arrays':arrays,'groups':groups,'independent_rows':independent,'active_rows':int(active.sum()),'joint_statistics_rank':rank,'new_source_statistics_rank':conditional_rank,'recapture_rows':np.flatnonzero(active&(np.arange(len(ad))>=start)).tolist(),'initialization_objective_only':False,'physical_noise_likelihood_claimed':True,'physical_noise_model_is_explicitly_conditional':True,'physical_total_error_accuracy_claimed':False,'new_source_statistics_status':'original S01 statistics plus source-qualified same-mean whole-exposure encoded-RGB conditional source statistics; unknown-noise source geometry held by role only','conditional_statistics_held_new_rows':len(held),'new_role2_reason':'source geometry rows remain physically admitted; source-specific explicit conditional noise statistics are pending','pure_conditional_scope_parent':dict(file=str(src),sha256=file_sha256(src)),'source_admission_or_current_UV_changed':False}
    write_packet(out/'packet.json',result);report=dict(schema='rinkfit.source_qualified_conditional_noise_scope.v1',raw_rows=p['row_count'],active_rows=int(active.sum()),original_S01_active_rows=int(active[:start].sum()),new_conditional_active_rows=int(active[start:].sum()),unknown_noise_geometry_role_held_rows=len(held),role_held_geometry_rows_source_rejected=0,current_original_S01_rank=p['current_original_W_rank'],new_conditional_rank=conditional_rank,total_rank=rank,initialization_surrogate_group_count=0,full_physical_error_model_calibrated=False,UV_PTS_admission_W_or_original_four_aux_changed=False,packet_sha256=file_sha256(out/'packet.json'));write_json(out/'report.json',report);return report
