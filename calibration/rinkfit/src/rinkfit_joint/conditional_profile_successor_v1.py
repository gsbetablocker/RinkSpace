"""Read-only proof of the existing exact-profile statistics-only packet update.

Source authority remains the strict parent qualification reader. The accepted
change is exactly the pinned conditional_profile_packet_v1.bind construction;
no additional source kind, UV, Query, role, auxiliary or arbitrary W is allowed.
"""
from pathlib import Path
import numpy as np
from .io_utils import read_json,read_jsonl,file_sha256,object_sha256
from .r_inner_prefix_v1 import array
from .board_native_qualification_reader_v1 import assert_current
from .board_profile_statistics_v1 import ref,checked,SCHEMA as GATE_SCHEMA
from . import conditional_profile_packet_v1 as binding

BINDING_SHA='a122282ab625bd633d3901198cd272ee8afd52887380e2aeb9ea7da5f78e3dcb'
SCHEMA='rinkfit.current_exact_profile49_statistics_only_successor.v1'

def assert_statistics_only_current(packet_path):
    pp=Path(packet_path).resolve();p=read_json(pp);seen=set();update=p.get('exact_profile_statistics_update')
    if not update or update['schema']!=binding.SCHEMA:raise ValueError('No recognized exact-profile statistics-only update')
    issuer=Path(binding.__file__).resolve()
    if file_sha256(issuer)!=BINDING_SHA or update['issuer']!=ref(issuer):raise ValueError('Statistics update does not use the frozen actual binding consumer')
    parent_path=checked(update['parent_packet'],seen);parent=read_json(parent_path);gp=checked(update['source_gate'],seen);gate=read_json(gp)
    if gate['schema']!=GATE_SCHEMA or gate['packet']!=ref(parent_path)or gate['issuer']!=ref(Path(__file__).with_name('board_profile_statistics_v1.py')):raise ValueError('Current statistics gate has another source or estimator')
    qp=checked(gate['native_source_qualification'],seen);source_report=assert_current(qp,parent_path)
    if update['native_source_qualification_used_for_parent']!=ref(qp):raise ValueError('Statistics update changed current source authority')
    sp=checked(gate['statistics_registry'],seen);registry=read_json(sp)
    if update['statistics_registry']!=ref(sp):raise ValueError('Statistics update redirects the measured covariance')
    solver=list(read_jsonl(checked(gate['solver_rows'],seen)));arrays={k:array(parent,k,parent_path.parent,seen)for k in parent['arrays']};active=(arrays['admission']==1)&(arrays['fit_role']==1);N=parent['row_count'];selected={};groups_selected={}
    for row in solver:
        i=int(row['global_row']);rid=row['source_row_id']
        if not 0<=i<N or i in selected or not active[i]or parent['source_row_ids'][i]!=rid or parent['source_dataset_ids'][i]!=row['source_dataset_id']or int(arrays['native_pts'][i])!=row['native_pts']or int(arrays['material_index'][i])!=row['joint_material_index']or not np.array_equal(arrays['uv'][i],np.asarray(row['current_uv'])):raise ValueError('Statistics gate changed an original source identity or mean')
        if any(row.get(k)is not True for k in ('solver_eligible','numeric_statistics_ready','statistical_objective_eligible'))or row['statistic_kind']!='native_joint_readout'or row['conditional_model_is_total_error_calibration']is not False:raise ValueError('Unknown source noise was relabeled as calibrated')
        gid=row['statistical_group_id'];g=registry[gid]
        if object_sha256(g)!=row['statistics_definition_sha256']:raise ValueError('Closed conditional statistic changed after numeric gate')
        selected[i]=row;groups_selected.setdefault(gid,[]).append(i)
    if len(solver)!=gate['eligible_conditional_rows']:raise ValueError('Statistics gate row denominator differs')
    wanted=set(selected);groups=[];partition=np.zeros(N,np.int16);partition[parent['independent_rows']]+=1;oldrank=2*len(parent['independent_rows'])
    if wanted.intersection(parent['independent_rows']):raise ValueError('A known original independent source was replaced')
    for group in parent['groups']:
        rows=list(map(int,group['rows']));overlap=wanted.intersection(rows);oldrank+=int(group['rank'])
        if overlap:
            if group['statistic_kind']!='constraint_only_unknown_noise'or group['rank']!=0 or group['W']!=[]or not group.get('unknown_noise_source_refs'):raise ValueError('Statistics update discarded original known W/cross terms')
            remaining=[i for i in rows if i not in wanted]
            if remaining:groups.append(dict(group,rows=remaining));partition[remaining]+=1
        else:
            copy=dict(group)
            if isinstance(group['W'],dict):copy['W']=dict(group['W'],file=str(checked(group['W'],seen)))
            groups.append(copy);partition[rows]+=1
    addedrank=0
    for gid,indices in groups_selected.items():
        group=registry[gid];byid={selected[i]['source_row_id']:i for i in indices}
        if set(byid)!=set(group['row_ids'])or len(byid)!=len(group['row_ids']):raise ValueError('Full actual exposure W was cut to a successful subset')
        rows=[byid[rid]for rid in group['row_ids']];W=dict(group['W'],file=str(checked(group['W'],seen)))
        if W['shape']!=[group['rank'],2*len(rows)]or not 0<group['rank']<=len(rows):raise ValueError('Conditional normal statistic has invalid support')
        groups.append(dict(id=gid,source_group_id=gid,rows=rows,rank=int(group['rank']),W=W,statistic_kind='native_joint_readout',original_numerical_scheme=group['original_numerical_scheme'],conditional_model=group['conditional_model'],noise_calibration_status=group['noise_calibration_status'],physical_total_error_calibrated=False,source_statistics_ref=ref(sp),current_source_gate_ref=ref(gp),all_original49_Query_and_source_UV_preserved=True));partition[rows]+=1;addedrank+=int(group['rank'])
    if not np.array_equal(partition,active.astype(np.int16)):raise ValueError('Statistics successor lost or repeated an active source')
    unknown=sum(len(g['rows'])for g in groups if g.get('statistic_kind')=='constraint_only_unknown_noise');expected=dict(parent)
    expected['arrays']={k:dict(v,file=str((Path(v['file'])if Path(v['file']).is_absolute()else parent_path.parent/v['file']).resolve()))for k,v in parent['arrays'].items()};expected['groups']=groups;expected['joint_statistics_rank']=oldrank+addedrank;expected['new_source_statistics_rank']=int(parent.get('new_source_statistics_rank',0))+addedrank
    expected['new_source_statistics_status']='same-mean whole-native-exposure conditional W plus retained unknown-noise source engineering constraints; total physical noise uncalibrated'
    expected['source_manifests']=parent.get('source_manifests',[])+[ref(gp),ref(sp),ref(qp),ref(issuer)]
    expected['constraint_only_unknown_noise_rows']=unknown;expected['constraint_only_unknown_noise_groups']=sum(g.get('statistic_kind')=='constraint_only_unknown_noise'for g in groups);expected['new_source_rows_with_conditional_W']=sum(len(g['rows'])for g in groups if g.get('statistic_kind')=='native_joint_readout');expected['source_noise_statistics_supplied_for_appended_rows']=unknown==0;expected['unknown_noise_source_rows_keep_engineering_constraints']=bool(unknown);expected['unknown_source_noise_statistical_precision_supplied']=False;expected['physical_noise_model_is_explicitly_conditional']=True;expected['conditional_source_objective_is_total_physical_noise_model']=False;expected['physical_total_error_accuracy_claimed']=False;expected['formal_new_source_statistical_optimum_claimed']=False
    expected['exact_profile_statistics_update']=dict(schema=binding.SCHEMA,parent_packet=ref(parent_path),source_gate=ref(gp),statistics_registry=ref(sp),native_source_qualification_used_for_parent=ref(qp),issuer=ref(issuer),closed_exposure_groups=len(groups_selected),source_rows=len(selected),added_conditional_rank=addedrank,all_original_arrays_source_IDs_roles_Query49_E_and_aux_bytes_preserved=True,unknown_noise_rows_retained=unknown,old_known_W_or_cross_terms_changed=False,native_source_qualification_must_be_reissued_for_new_packet_SHA=True,physical_total_noise_calibrated=False,quality_accepted=False)
    changed=sorted(k for k in set(parent)|set(p)if parent.get(k)!=p.get(k));mismatches=sorted(k for k in set(expected)|set(p)if expected.get(k)!=p.get(k))
    if mismatches:raise ValueError('Statistics-only successor has additional or redirected changes: '+str(mismatches))
    return dict(schema=SCHEMA,passed=True,packet=ref(pp),expected_source_parent=ref(parent_path),parent_native_source_qualification=ref(qp),parent_source_validation=source_report,statistics_gate=ref(gp),statistics_registry=ref(sp),frozen_binding_consumer=ref(issuer),reader=ref(__file__),actual_changed_top_level_fields=changed,raw_rows=N,active_rows=int(active.sum()),new_conditional_rows=len(selected),new_conditional_groups=len(groups_selected),added_conditional_rank=addedrank,unknown_noise_rows_retained=unknown,all_source_arrays_IDs_roles_Queries49_E_and_aux_exactly_preserved=True,original_known_W_and_all_cross_terms_preserved=True,new_source_admission_issued=False,physical_total_noise_calibrated=False,quality_accepted=False)
