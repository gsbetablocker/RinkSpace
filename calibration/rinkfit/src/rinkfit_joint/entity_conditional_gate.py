"""Sign new fixed physical entities only after same-mean whole-RGB joint MC."""
from pathlib import Path
import json
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .role_guard import FrameRoleGuard


def promote(project,run,source_qualification,statistics,outdir):
    run=Path(run);out=Path(outdir);source=Path(source_qualification);stats=Path(statistics)
    if out.exists():raise ValueError('New immutable entity statistical gate required')
    cache={}
    def verify(ref):
        p=Path(ref.get('file',ref.get('path'))).resolve();key=str(p)
        if key not in cache:cache[key]=file_sha256(p)
        if cache[key]!=ref['sha256']:raise ValueError('Changed entity source/statistic dependency '+key)
        return p
    sm=read_json(source)
    if sm['schema']!='rinkfit.new_actual_type1_source_qualification_gate.v1':raise ValueError('Wrong source qualification')
    original=verify(sm['input_cumulative_journal']);qr=verify(sm['qualification_receipts'])
    for k in ('current_model','source_mean_and_floating_estimator_replay_QA'):verify(sm[k])
    physical={r['source_row_id']:r for r in read_jsonl(qr)}
    registry=read_json(stats);lookup={};group_cache={}
    for gid,record in registry.items():
        for rid in record['row_ids']:
            if rid in lookup:raise ValueError('Entity readout appears in two independent source-noise groups')
            lookup[rid]=gid
    guard=FrameRoleGuard(run/'roles/frame_roles.json');solver=[];decisions=[];receipts=[]
    def actual_group(gid):
        if gid in group_cache:return group_cache[gid]
        g=registry[gid]
        if g['statistic_kind']!='native_joint_readout'or g.get('statistics_use')!='conditional_noise_objective'or not g['numeric_whitening_verified']or g['actual_mean_replay_max_px']>1e-9 or g['float32_channel_pipeline_error']!=0 or g['physical_total_error_calibrated']is not False:
            raise ValueError('Entity statistics are not explicit exact-mean conditional native RGB noise')
        if not g.get('shared_full_native_RGB_domain')or g['trials']!=256:
            raise ValueError('Separate/crop entity covariance cannot replace whole-exposure union')
        for name in ('W','frozen_operator','replay_arrays'):verify(g[name])
        for ref in g.get('entity_frozen_source_operator_refs',[])+g.get('source_mask_refs',[])+g.get('entity_source_dependencies',[]):verify(ref)
        with np.load(g['replay_arrays']['file'])as z:
            if str(z['schema'])!='rinkfit.mixed_native_readout_component_covariance.v1':raise ValueError('Entity source has no 2-D component covariance layout')
            data={k:z[k].copy()for k in ('source_row_ids','component_row_index','component_directions','component_covariance_px2','actual_mean_uv')}
        W=np.fromfile(g['W']['file'],dtype=g['W']['dtype']).reshape(g['W']['shape'])
        if W.shape!=(g['rank'],2*len(g['row_ids'])):raise ValueError('Whole mixed W has wrong actual row order')
        if list(map(str,data['source_row_ids']))!=g['row_ids']:raise ValueError('Component source row order differs from W')
        # Full crossCov remains in the parent; this gate never extracts W columns.
        group_cache[gid]=(g,data);return g,data
    for row in read_jsonl(original):
        rid=row['source_row_id'];q=physical[rid];pts=int(row['native_pts'])
        guard.assert_training_pixels([pts],shot_id=row['shot_id'])
        sourcehash=object_sha256(dict(source_row_id=rid,native_pts=pts,current_2D_UV=row['current_uv'],estimator='native_two_edge_quadratic_intersection_v1'))
        if sourcehash!=q['current_2D_source_mean_sha256']or q['original_UV_preserved']!=row['source_uv']or q['existing_material_index']!=row['joint_material_index']:
            raise ValueError('New entity mean differs from its physical source qualification')
        if not q['new_actual_source_geometry_qualified']or q['entity_kind']!=1 or not q['physical_point']:
            raise ValueError('Entity source was not qualified as a fixed physical point')
        decision=dict(source_row_id=rid,native_pts=pts,new_source_version=q['new_source_version'],source_geometry_eligible=True,original_raw_flags_preserved=q['original_flags_preserved'],solver_eligible=False,statistical_objective_eligible=False,numeric_statistics_ready=False)
        gid=lookup.get(rid)
        if gid is None:
            decisions.append(dict(decision,statistical_pending_reason='whole_exposure_same_mean_entity_noise_not_closed_or_estimator_domain_failure'));continue
        g,data=actual_group(gid);j=g['row_ids'].index(rid);columns=np.flatnonzero(data['component_row_index']==j)
        if len(columns)!=2 or not np.array_equal(data['component_directions'][columns],np.eye(2))or not np.array_equal(data['actual_mean_uv'][j],row['current_uv']):
            raise ValueError('Physical entity was replaced by a sliding normal-only source')
        marginal=data['component_covariance_px2'][np.ix_(columns,columns)];s=np.linalg.eigvalsh(marginal)
        if np.any(s<=np.finfo(float).eps*max(abs(s).max(),np.finfo(float).tiny)*16):
            decisions.append(dict(decision,statistical_pending_reason='two_physical_source_directions_do_not_both_have_supported_conditional_variance',component_eigenvalues=s.tolist()));continue
        immutable=object_sha256(dict(native_pts=pts,source_uv=row['source_uv'],source_row_id=rid))
        current=object_sha256(dict(native_pts=pts,current_uv=row['current_uv'],source_row_id=rid,measurement_operator=row['measurement_operator_resolved']))
        if g['source_measurements_sha256'][rid]!=immutable or g['current_measurements_sha256'][rid]!=current:
            raise ValueError('Whole-exposure W redirects original/current entity means')
        nr=g['native_receipt'];fr=row['current_native_frame_source']
        if nr['native_pts']!=pts or nr['shot_id']!=row['shot_id']or nr['decoded_native_RGB_sha256']!=fr['RGB_sha256']or nr['native_RGB_resampled']or not nr['training_role_verified']:
            raise ValueError('Entity statistics use another source exposure or processing clock')
        verify(dict(file=row['native_png'],sha256=row['native_png_sha256']))
        op=row['current_corner_operator'];verify(op['pure_remeasure_entry']);verify(dict(file=op['native_mask_file'],sha256=op['native_mask_sha256']))
        if op['pure_remeasure_entry']['sha256']!=q['pure_remeasure_entry_sha256']or not op['all_support_pixels_unmasked']:
            raise ValueError('Current entity source operator or visibility changed')
        receipt=dict(schema='rinkfit.new_actual_physical_entity_conditional_admission.v1',source_row_id=rid,native_pts=pts,
            source_qualification=dict(file=str(source.resolve()),sha256=file_sha256(source)),source_qualification_receipt_sha256=object_sha256(q),
            immutable_source_measurement_sha256=immutable,current_measurement_sha256=current,current_2D_source_mean_sha256=sourcehash,
            statistical_group_id=gid,statistics_definition_sha256=object_sha256(g),kind=1,entity_station_fixed=True,
            fixed_physical_point_path=q['existing_point_path'],material_index=q['existing_material_index'],
            whole_exposure_cross_covariance_retained=True,exact_parent_covariance_marginal_required=True,
            original_raw_flags_not_overwritten=True,physical_total_error_calibrated=False)
        receipts.append(receipt);suffix=dict(solver_eligible=True,statistical_objective_eligible=True,numeric_statistics_ready=True,
            statistic_kind='native_joint_readout',statistics_use='conditional_noise_objective',statistical_group_id=gid,receipt_sha256=object_sha256(receipt))
        decisions.append(dict(decision,**suffix))
        solver.append(dict(row,**suffix,kind=1,entity_station_fixed=True,new_measurement_version=q['new_source_version'],original_raw_source_flags_preserved=True))
    if len(decisions)!=len(physical):raise ValueError('Entity source qualification ledger was not fully consumed')
    for path,digest in cache.items():
        if file_sha256(path)!=digest:raise ValueError('Source/statistics changed during entity admission')
    out.mkdir();write_json(out/'source_rows.ref.json',dict(file=str(original),sha256=file_sha256(original)))
    write_jsonl(out/'solver_rows.jsonl.gz',solver);write_jsonl(out/'promotion_decisions.jsonl.gz',decisions);write_jsonl(out/'receipts.jsonl.gz',receipts)
    report=dict(schema='rinkfit.current_physical_entity_conditional_gate.v1',raw_new_entity_rows=len(decisions),source_geometry_qualified_rows=len(physical),conditional_rows=len(solver),statistics_pending_rows=len(decisions)-len(solver),
        whole_exposure_Cov_exact_marginal_required=True,source_qualification=dict(file=str(source.resolve()),sha256=file_sha256(source)),statistics_registry=dict(file=str(stats.resolve()),sha256=file_sha256(stats)),
        original_raw_source_flags_and_UV_not_overwritten=True,all_entities_fixed_physical_point_correspondences=True,independent_crop_or_corner_W_added=False,physical_total_error_calibrated=False)
    write_json(out/'manifest.json',report);return report
