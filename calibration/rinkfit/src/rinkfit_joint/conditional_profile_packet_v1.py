"""Install current whole-exposure conditional W without rewriting profile Queries.

Only unknown-noise rank-zero rows covered by an exact closed exposure change
statistics. Raw rows, finite Query49, UV, roles, E, and all nuisance arrays are
inherited as bytes. The remaining unknown geometry constraints stay present.
"""
from pathlib import Path
import json
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,file_sha256,object_sha256
from .packet_storage import write_packet
from .r_inner_prefix_v1 import array
from .board_native_qualification_reader_v1 import assert_current
from .board_profile_statistics_v1 import SCHEMA as GATE_SCHEMA,ref,checked

SCHEMA='rinkfit.statistics_only_exact_profile49_packet_update.v1'

def bind(parent_packet,statistics_gate,output):
    pp=Path(parent_packet).resolve();gp=Path(statistics_gate).resolve();out=Path(output).resolve()
    if out.exists():raise ValueError('New immutable exact-profile statistics packet required')
    gate=read_json(gp);p=read_json(pp);seen=set()
    if gate['schema']!=GATE_SCHEMA or gate['packet']!=ref(pp)or gate['issuer']!=ref(Path(__file__).with_name('board_profile_statistics_v1.py')):raise ValueError('Unknown or redirected current profile statistics gate')
    qp=checked(gate['native_source_qualification'],seen);assert_current(qp,pp)
    sp=checked(gate['statistics_registry'],seen);registry=read_json(sp);solver=list(read_jsonl(checked(gate['solver_rows'],seen)))
    if len(solver)!=gate['eligible_conditional_rows']:raise ValueError('Numerical gate source denominator changed')
    arrays={k:array(p,k,pp.parent,seen)for k in p['arrays']};active=(arrays['admission']==1)&(arrays['fit_role']==1);N=p['row_count'];selected={};groups_selected={}
    for row in solver:
        i=int(row['global_row']);rid=row['source_row_id']
        if not 0<=i<N or i in selected or not active[i]or p['source_row_ids'][i]!=rid or p['source_dataset_ids'][i]!=row['source_dataset_id']or int(arrays['native_pts'][i])!=row['native_pts']or int(arrays['material_index'][i])!=row['joint_material_index']or not np.array_equal(arrays['uv'][i],np.asarray(row['current_uv'])):raise ValueError('Statistics gate redirects an original active source row')
        if any(row.get(k)is not True for k in ('solver_eligible','numeric_statistics_ready','statistical_objective_eligible'))or row['statistic_kind']!='native_joint_readout'or row['conditional_model_is_total_error_calibration']is not False:raise ValueError('Unknown noise was assigned statistical precision')
        gid=row['statistical_group_id'];g=registry[gid]
        if object_sha256(g)!=row['statistics_definition_sha256']:raise ValueError('Closed conditional statistic definition changed')
        selected[i]=row;groups_selected.setdefault(gid,[]).append(i)
    wanted=set(selected);partition=np.zeros(N,np.int16);groups=[];oldrank=2*len(p['independent_rows'])
    original_partition=np.zeros(N,np.int16);original_partition[p['independent_rows']]+=1
    for group in p['groups']:original_partition[group['rows']]+=1
    if not np.array_equal(original_partition,active.astype(np.int16)):raise ValueError('Parent active source partition is not exact')
    if wanted.intersection(p['independent_rows']):raise ValueError('A real point statistic cannot be replaced by a profile covariance')
    partition[p['independent_rows']]+=1
    for g in p['groups']:
        rows=list(map(int,g['rows']));overlap=wanted.intersection(rows);oldrank+=int(g['rank'])
        if overlap:
            if g['statistic_kind']!='constraint_only_unknown_noise'or g['rank']!=0 or g['W']!=[]or not g.get('unknown_noise_source_refs'):raise ValueError('Replacement would discard an existing known joint statistic')
            remaining=[i for i in rows if i not in wanted]
            if remaining:groups.append(dict(g,rows=remaining));partition[remaining]+=1
            continue
        copy=dict(g)
        if isinstance(g['W'],dict):copy['W']=dict(g['W'],file=str(checked(g['W'],seen)))
        groups.append(copy);partition[rows]+=1
    addedrank=0;consumed=[]
    for gid,indices in groups_selected.items():
        g=registry[gid];byid={selected[i]['source_row_id']:i for i in indices}
        if set(byid)!=set(g['row_ids'])or len(byid)!=len(g['row_ids']):raise ValueError('Full-exposure W was split into a successful subset')
        rows=[byid[rid]for rid in g['row_ids']];Wref=dict(g['W'],file=str(checked(g['W'],seen)))
        if Wref['shape']!=[g['rank'],2*len(rows)]or not 0<g['rank']<=len(rows):raise ValueError('Conditional normal W has invalid support dimension')
        groups.append(dict(id=gid,source_group_id=gid,rows=rows,rank=int(g['rank']),W=Wref,statistic_kind='native_joint_readout',original_numerical_scheme=g['original_numerical_scheme'],conditional_model=g['conditional_model'],noise_calibration_status=g['noise_calibration_status'],physical_total_error_calibrated=False,source_statistics_ref=ref(sp),current_source_gate_ref=ref(gp),all_original49_Query_and_source_UV_preserved=True));partition[rows]+=1;addedrank+=int(g['rank']);consumed.append(dict(group_id=gid,rows=len(rows),rank=g['rank'],W=Wref,full_exposure_cross_terms_consumed=True))
    if not np.array_equal(partition,active.astype(np.int16)):raise ValueError('Statistics-only groups do not preserve the full active source partition')
    result=dict(p);result['arrays']={k:dict(v,file=str((Path(v['file'])if Path(v['file']).is_absolute()else pp.parent/v['file']).resolve()))for k,v in p['arrays'].items()};result['groups']=groups;result['joint_statistics_rank']=oldrank+addedrank;result['new_source_statistics_rank']=int(p.get('new_source_statistics_rank',0))+addedrank
    result['new_source_statistics_status']='same-mean whole-native-exposure conditional W plus retained unknown-noise source engineering constraints; total physical noise uncalibrated'
    refs=[ref(gp),ref(sp),ref(qp),ref(__file__)];result['source_manifests']=p.get('source_manifests',[])+refs
    unknown=sum(len(g['rows'])for g in groups if g.get('statistic_kind')=='constraint_only_unknown_noise')
    result['constraint_only_unknown_noise_rows']=unknown;result['constraint_only_unknown_noise_groups']=sum(g.get('statistic_kind')=='constraint_only_unknown_noise'for g in groups)
    result['new_source_rows_with_conditional_W']=sum(len(g['rows'])for g in groups if g.get('statistic_kind')=='native_joint_readout')
    result['source_noise_statistics_supplied_for_appended_rows']=unknown==0;result['unknown_noise_source_rows_keep_engineering_constraints']=bool(unknown);result['unknown_source_noise_statistical_precision_supplied']=False
    result['physical_noise_model_is_explicitly_conditional']=True;result['conditional_source_objective_is_total_physical_noise_model']=False;result['physical_total_error_accuracy_claimed']=False;result['formal_new_source_statistical_optimum_claimed']=False
    result['exact_profile_statistics_update']=dict(schema=SCHEMA,parent_packet=ref(pp),source_gate=ref(gp),statistics_registry=ref(sp),native_source_qualification_used_for_parent=ref(qp),issuer=ref(__file__),closed_exposure_groups=len(consumed),source_rows=len(selected),added_conditional_rank=addedrank,all_original_arrays_source_IDs_roles_Query49_E_and_aux_bytes_preserved=True,unknown_noise_rows_retained=unknown,old_known_W_or_cross_terms_changed=False,native_source_qualification_must_be_reissued_for_new_packet_SHA=True,physical_total_noise_calibrated=False,quality_accepted=False)
    out.mkdir(parents=True);write_packet(out/'packet.json',result)
    # Re-read the actual emitted packet, rather than treating construction as QA.
    current=read_json(out/'packet.json')
    for name,desc in p['arrays'].items():
        actual=current['arrays'][name]
        if any(actual[k]!=desc[k]for k in ('dtype','shape','sha256')):raise ValueError('Statistics update changed source or nuisance bytes')
    for name in ('queries','source_row_ids','source_dataset_ids','allowed_paths','source_station_domains','independent_rows'):
        if current.get(name)!=p.get(name):raise ValueError('Statistics update changed original '+name)
    report=dict(schema=SCHEMA,parent_packet=ref(pp),packet=ref(out/'packet.json'),statistics_gate=ref(gp),consumer=ref(__file__),raw_rows=N,active_rows=int(active.sum()),original_known_rank=oldrank,added_conditional_rank=addedrank,actual_known_rank=result['joint_statistics_rank'],new_conditional_rows=len(selected),new_exposure_groups=consumed,all_active_source_partition_exactly_once=True,all_original_arrays_and_queries_byte_identity_preserved=True,unknown_noise_rows_retained=unknown,source_membership_or_means_changed=False,native_requalification_and_new_runtime_packet_binding_required=True,running_packet_modified=False,formal_trial_accepted=False,physical_Solver_started=False,quality_accepted=False)
    write_json(out/'update_receipt.json',report);return report
