"""Bind closed same-mean conditional RGB statistics to current native board proof.

The source authority is the shared native qualification reader. This numerical
consumer cannot promote a new material, move a mean, shorten a finite Query,
or assign a variance to an unmeasured source. All exposure cross terms survive.
"""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .r_inner_prefix_v1 import array
from .board_native_qualification_reader_v1 import assert_current

SCHEMA='rinkfit.current_board_profile_conditional_statistics_gate.v1'

def ref(path):
    path=Path(path).resolve()
    return dict(file=str(path),sha256=file_sha256(path))

def checked(spec,seen):
    path=Path(spec.get('file',spec.get('path'))).resolve()
    key=(str(path),spec['sha256'])
    if key not in seen:
        if file_sha256(path)!=spec['sha256']:raise ValueError('Conditional statistics dependency changed: '+str(path))
        seen.add(key)
    return path

def build(packet_path,native_qualification,statistics_registry,output):
    pp=Path(packet_path).resolve();qp=Path(native_qualification).resolve();sp=Path(statistics_registry).resolve();out=Path(output).resolve()
    if out.exists():raise ValueError('New immutable profile statistics gate required')
    source=assert_current(qp,pp,return_records=True)
    records=source['records_by_global_row'];p=read_json(pp);registry=read_json(sp);seen=set()
    arrays={k:array(p,k,pp.parent,seen)for k in ('uv','native_pts','shot_index','kind','material_index','admission','fit_role','profile_normal')}
    byid={r['source_row_id']:r for r in records.values()};byframe=defaultdict(set)
    if len(byid)!=len(records):raise ValueError('Current qualified source IDs are not unique')
    for r in records.values():byframe[(r['shot_id'],int(r['native_pts']))].add(r['source_row_id'])
    all_stats_ids=set();closed_ids=set();usable={};pending=[]
    for gid,g in registry.items():
        ids=g['row_ids']
        if gid!=g['source_group_id']or len(ids)!=len(set(ids))or all_stats_ids.intersection(ids):raise ValueError('Repeated or redirected same-exposure readout statistic')
        if any(rid not in byid for rid in ids):raise ValueError('Statistic contains a source outside current strict qualification')
        all_stats_ids.update(ids);identity={(byid[rid]['shot_id'],int(byid[rid]['native_pts']))for rid in ids}
        if len(identity)!=1:raise ValueError('Conditional frame covariance crosses actual exposures')
        identity=next(iter(identity))
        if set(ids)!=byframe[identity]:raise ValueError('Whole actual exposure statistic omitted a qualified current readout')
        if g['shot_id']!=identity[0]or int(g['native_pts'])!=identity[1]:raise ValueError('Statistic source exposure identity differs')
        if g.get('actual_mean_replay_max_px')!=0. or g.get('float32_channel_pipeline_error')!=0.:raise ValueError('Actual complete RGB estimator mean was not exact')
        if g.get('physical_total_error_calibrated')is not False or g.get('conditional_model_is_total_error_calibration')is not False:raise ValueError('Conditional RGB model cannot certify total physical source noise')
        if not g.get('shared_full_native_RGB_domain')or g.get('independent_rows_or_PSD_floor_added')or g.get('tangent_information_from_navigation_seed_claimed'):raise ValueError('Actual joint noise model or Query information was changed')
        nr=g['native_receipt']
        if int(nr['native_pts'])!=identity[1]or nr['shot_id']!=identity[0]or nr.get('native_RGB_resampled')is not False or nr.get('training_role_verified')is not True:raise ValueError('Conditional statistic does not use actual native training RGB')
        for rid in ids:
            r=byid[rid];i=int(r['global_row']);actual=r['actual_frame']['native_receipt']
            if nr['source_video_sha256']!=actual['source_video_sha256']or nr['decoded_native_BGR_sha256']!=actual['decoded_native_BGR_sha256']or nr['decoded_native_RGB_sha256']!=r['actual_native_RGB_sha256']:raise ValueError('Shared actual RGB/mask/source clock identity differs')
            expected=r['original_operator_review']['frozen_peak_recipe']
            if checked(g['frozen_operator'],seen)!=checked(expected,seen):raise ValueError('Conditional statistics use another original estimator')
            # The shared reader has already proved actual role/operator/profile49.
            # Retain all 49 original Query samples instead of replacing them by ends.
            q=next_query(p,i)
            offsets=np.asarray(r['normal_offsets'],dtype='<f8');normal=np.asarray(r['normal'],dtype='<f8');seed=np.asarray(r['seed_uv'],dtype='<f8')
            if offsets.shape!=(49,)or not np.array_equal(np.asarray(q['alpha'],dtype='<f8'),offsets)or not np.array_equal(np.asarray(q['normal'],dtype='<f8'),normal):raise ValueError('Original finite profile Query changed')
            if not np.array_equal(np.asarray(q['native_uv'],dtype='<f8'),seed+offsets[:,None]*normal):raise ValueError('Original49 Query samples do not exactly consume seed+alpha*normal')
            if not np.array_equal(arrays['uv'][i],np.asarray(r['current_uv'],dtype='<f8'))or int(arrays['kind'][i])==1:raise ValueError('Profile source was changed into a fixed entity')
        if not g.get('W')or g.get('rank',0)==0:
            if g.get('numeric_whitening_verified'):raise ValueError('Missing W advertises numeric precision')
            pending.append(dict(group_id=gid,row_ids=ids,reason=g.get('status','no_measured_covariance_support')));continue
        if g.get('statistic_kind')!='native_joint_readout'or g.get('numeric_whitening_verified')is not True:raise ValueError('Unverified conditional readout statistic')
        wf=checked(g['W'],seen);zf=checked(g['replay_arrays'],seen);n=len(ids)
        W=np.fromfile(wf,dtype=g['W']['dtype'],offset=g['W'].get('offset_bytes',0),count=int(np.prod(g['W']['shape']))).reshape(g['W']['shape'])
        if W.shape!=(g['rank'],2*n)or not np.isfinite(W).all():raise ValueError('Conditional W row/column order or arithmetic invalid')
        with np.load(zf,allow_pickle=False)as z:
            if z['source_row_ids'].tolist()!=ids:raise ValueError('Covariance replay changed source column order')
            means=np.asarray(z['actual_mean_uv']);normals=np.asarray(z['profile_normals']);C=np.asarray(z['normal_covariance_px2'])
        expected_uv=np.asarray([byid[rid]['current_uv']for rid in ids]);expected_normals=np.asarray([byid[rid]['normal']for rid in ids])
        if not np.array_equal(means,expected_uv)or not np.array_equal(normals,expected_normals)or C.shape!=(n,n)or not np.isfinite(C).all():raise ValueError('Covariance does not preserve original means/normals')
        B=np.zeros((n,2*n))
        for j,norm in enumerate(normals):B[j,2*j:2*j+2]=norm
        Wn=W@B.T
        # Same arithmetic support check used by supported_factor_whitening.
        if not np.allclose(Wn@C@Wn.T,np.eye(len(W)),rtol=1e-7,atol=1e-8)or not np.allclose(W,Wn@B,rtol=1e-7,atol=1e-8):raise ValueError('Conditional normal support whitening or no-tangent contract failed')
        usable[gid]=dict(g,source_statistics_definition_sha256=object_sha256(g));closed_ids.update(ids)
    needed=set(all_stats_ids);journals={byid[rid]['original_source_journal']['file']:byid[rid]['original_source_journal']['sha256']for rid in needed};original={}
    fields=('source_row_id','source_family_id','shot_id','native_pts','source_uv','current_uv','joint_material_id','joint_material_index','measurement_operator_resolved','normal','navigation_seed_uv','kind','entity_station_fixed')
    for journal,digest in journals.items():
        checked(dict(file=journal,sha256=digest),seen)
        for row in read_jsonl(journal):
            rid=row['source_row_id']
            if rid not in needed:continue
            if rid in original:raise ValueError('Same source ID repeated in original qualified journals')
            original[rid]={k:row[k]for k in fields}
    if set(original)!=needed:raise ValueError('Original immutable source row is missing')
    solver=[];eligible_groups=[]
    for gid,g in usable.items():
        for rid in g['row_ids']:
            r=byid[rid];row=original[rid]
            if row['native_pts']!=r['native_pts']or row['shot_id']!=r['shot_id']or row['joint_material_index']!=r['material_index']or not np.array_equal(np.asarray(row['current_uv'],dtype='<f8'),np.asarray(r['current_uv'],dtype='<f8'))or not np.array_equal(np.asarray(row['normal'],dtype='<f8'),np.asarray(r['normal'],dtype='<f8'))or not np.array_equal(np.asarray(row['navigation_seed_uv'],dtype='<f8'),np.asarray(r['seed_uv'],dtype='<f8')):raise ValueError('Original source journal/strict current proof differs')
            sh=object_sha256(dict(native_pts=row['native_pts'],source_uv=row['source_uv'],source_row_id=rid));ch=object_sha256(dict(native_pts=row['native_pts'],current_uv=row['current_uv'],source_row_id=rid,measurement_operator=row['measurement_operator_resolved']))
            if g['source_measurements_sha256'][rid]!=sh or g['current_measurements_sha256'][rid]!=ch:raise ValueError('W uses a different exact original/current readout')
            solver.append(dict(row,global_row=int(r['global_row']),source_dataset_id=r['source_dataset_id'],statistical_group_id=gid,statistic_kind='native_joint_readout',solver_eligible=True,numeric_statistics_ready=True,statistical_objective_eligible=True,conditional_model_is_total_error_calibration=False,native_source_qualification=ref(qp),statistics_definition_sha256=g['source_statistics_definition_sha256']))
        eligible_groups.append(dict(group_id=gid,rows=len(g['row_ids']),rank=g['rank'],W=g['W'],full_exposure_cross_terms_preserved=True))
    out.mkdir(parents=True);write_jsonl(out/'solver_rows.jsonl.gz',solver)
    report=dict(schema=SCHEMA,packet=ref(pp),native_source_qualification=ref(qp),native_reader=ref(Path(__file__).with_name('board_native_qualification_reader_v1.py')),issuer=ref(__file__),statistics_registry=ref(sp),solver_rows=ref(out/'solver_rows.jsonl.gz'),source_qualified_rows=len(records),closed_statistics_exposures=len(registry),closed_statistics_rows=len(all_stats_ids),eligible_conditional_rows=len(solver),eligible_conditional_groups=eligible_groups,closed_without_positive_W=pending,unclosed_or_no_W_source_rows_retained=len(records)-len(solver),all_original49_Query_samples_verified=True,original_arrays_and_queries_changed=False,current_source_membership_changed=False,physical_total_noise_calibrated=False,unknown_source_noise_precision_invented=False,quality_accepted=False)
    write_json(out/'manifest.json',report);return report

def next_query(packet,i):
    # Built once per loaded packet without changing its serialized data.
    cache=packet.get('_conditional_profile_query_index')
    if cache is None:cache={int(q['row_index']):q for q in packet['queries']};packet['_conditional_profile_query_index']=cache
    if i not in cache:raise ValueError('Current original profile Query missing')
    return cache[i]
