"""Admit new full-wide C0 pairs against exact same-exposure 4-D statistics."""
from pathlib import Path
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256,object_sha256
from .role_guard import FrameRoleGuard
from .measurement_contract import EventMeasurementContract


def promote(project,run,binding_manifest,factor_rows,statistics,outdir):
    run=Path(run);out=Path(outdir);binding_file=Path(binding_manifest);stats=Path(statistics)
    if out.exists():raise ValueError('New immutable recovered-pair admission generation required')
    for name in ('SOURCE_IDENTITY_CERTIFICATE_RECALLED.json','SOURCE_IDENTITY_EVIDENCE_RECALL.json'):
        if (binding_file.parent/name).exists():raise ValueError('Recovered-pair source qualification was recalled: '+str(binding_file.parent/name))
    cache={}
    def verify(ref):
        p=Path(ref.get('file',ref.get('path'))).resolve();key=str(p)
        if key not in cache:cache[key]=file_sha256(p)
        if cache[key]!=ref['sha256']:raise ValueError('Changed recovered-pair source/statistic '+key)
        return p
    bm=read_json(binding_file)
    if bm['schema']!='rinkfit.new_C0_actual_4D_source_binding_gate.v1':raise ValueError('Wrong actual pair source binding gate')
    journal=verify(bm['source_journal']);bindings={r['source_row_id']:r for r in read_jsonl(verify(bm['binding_receipts']))};model=read_json(verify(bm['current_model']));material=model['materials'][144]
    if material['label']!='ice.C0:outer':raise ValueError('C0 model material identity differs')
    actual_rows={r['source_row_id']:r for r in read_jsonl(journal)};factors={r['source_row_id']:r for r in read_jsonl(factor_rows)if r.get('operator_kind')=='wide_paint_flank'}
    registry=read_json(stats);lookup={}
    for gid,g in registry.items():
        for rid in g['row_ids']:
            if rid in lookup:raise ValueError('Same recovered pair appears in independent noise groups')
            lookup[rid]=gid
    guard=FrameRoleGuard(run/'roles/frame_roles.json');contract=EventMeasurementContract(project,'ankara_2026')
    if contract.expected('ice.C0:outer')['operator']!='paint_center':raise ValueError('C0 is not defined as a finite-width stroke')
    decisions=[];receipts=[];solver=[];approved=[];geometry=[];group_cache={};masks={};source_proofs={}
    for rid,row in actual_rows.items():
        pts=int(row['native_pts']);guard.assert_training_pixels([pts],shot_id=row['shot_id']);b=bindings[rid];binding=b['new_measurement_binding'];pairid='restored_C0_fullwide/'+rid;ids=[pairid+'/flank/'+str(i)for i in (0,1)]
        qhash=object_sha256(dict(source_row_id=rid,native_pts=pts,actual_4D_UV=row['actual_pair_4D_UV'],estimator='current_C0_full_wide_peak_pair_then_two_flank_v1'))
        if not b['new_measurement_source_geometry_qualified']or b['current_4D_source_mean_sha256']!=qhash or row['current_4D_source_mean_sha256']!=qhash:raise ValueError('New 4-D mean differs from physical source qualification')
        if binding['joint_material_index']!=144 or binding['existing_path_id']!='w054/circle'or binding['existing_path_kind']!=1 or binding['image_midpoint_used_as_world_center_projection']:raise ValueError('Actual C0 pair is not bound to the existing circular stroke')
        version=row['source_remeasurement_version'];parent=run/version;sm=read_json(parent/'manifest.json');qr=verify(sm['source_qualification_receipts'])
        if version not in source_proofs:source_proofs[version]={q['source_row_id']:q for q in read_jsonl(qr)}
        sourceq=source_proofs[version][rid]
        if not sourceq['new_actual_source_geometry_qualified']or not sourceq['original_UV_and_flags_preserved']or not sourceq['source_RGB_material_condition']['passed']or not sourceq['midpoint_is_not_model_center_projection']or sourceq['current_4D_source_mean_sha256']!=qhash:raise ValueError('Full-wide C0 physical source qualification is incomplete')
        verify(sourceq['same_actual_C0_identity_ref']);verify(row['source_identity_ref']);packet=read_json(verify(row['actual_source_packet']));value=packet
        for token in row['actual_source_packet']['pointer'].lstrip('/').split('/'):
            token=token.replace('~1','/').replace('~0','~');value=value[int(token)]if isinstance(value,list)else value[token]
        if value!=row['source_uv']:raise ValueError('Original inner readout/edited source pointer changed')
        fr=row['current_native_frame_source'];verify(fr);maskfile=verify(dict(file=fr['mask_file'],sha256=fr['mask_sha256']));op=row['current_pair_operator'];verify(op['pure_remeasure_entry'])
        if op['name']!='current_C0_full_wide_peak_pair_then_two_flank_v1'or not op['discrete_peak_pair_selection_must_be_replayed_in_every_MC_draw']or not op['material_identity_classification_frozen_in_MC']or not op['support_is_union_of_all_possible_discrete_peak_branches']:raise ValueError('C0 full-wide source operator was replaced by a local window')
        if op['pure_remeasure_entry']['sha256']!=sourceq['pure_remeasure_entry_sha256']:raise ValueError('C0 source replay entry differs')
        if str(maskfile)not in masks:
            from PIL import Image
            masks[str(maskfile)]=np.asarray(Image.open(maskfile)).astype(bool)
        mask=masks[str(maskfile)]
        for y,x0,x1 in op['source_dependency_envelope_pixel_runs_y_xfirst_xlast']:
            if np.any(mask[int(y),int(x0):int(x1)+1]):raise ValueError('Possible full-wide estimator branch crosses source visibility mask')
        spec=dict(id=pairid,source_row_ids=ids,material_index=144,width_scope_id=binding['width_scope_id'],allowed_world_side_assignments=binding['world_side_assignment_nuisance'],source_refs=[dict(file=str(binding_file.resolve()),sha256=file_sha256(binding_file)),dict(file=str(journal),sha256=file_sha256(journal))],source_parent_row_id=rid)
        geometry.append(spec);decision=dict(source_row_id=rid,native_pts=pts,pair_id=pairid,source_geometry_eligible=True,solver_eligible=False,numeric_statistics_ready=False,statistical_objective_eligible=False,original_inner_UV_or_flags_not_reactivated=True)
        gids={lookup.get(sid)for sid in ids}
        if None in gids or len(gids)!=1:
            decisions.append(dict(decision,statistical_pending_reason='complete_pair_estimator_domain_or_whole_exposure_noise_pending'));continue
        gid=next(iter(gids));g=registry[gid]
        if g['statistic_kind']!='native_joint_readout'or g['statistics_use']!='conditional_noise_objective'or not g['numeric_whitening_verified']or not g['shared_full_native_RGB_domain']or g['trials']!=256 or g['actual_mean_replay_max_px']>1e-9 or g['float32_channel_pipeline_error']!=0 or g['physical_total_error_calibrated']is not False or pairid not in g['complete_wide_pair_estimators_replayed']:raise ValueError('New C0 pair did not use the whole-native full-wide conditional estimator')
        for k in ('W','frozen_operator','replay_arrays'):verify(g[k])
        for ref in g.get('entity_frozen_source_operator_refs',[])+g.get('source_mask_refs',[])+g.get('entity_source_dependencies',[]):verify(ref)
        if gid not in group_cache:
            with np.load(g['replay_arrays']['file'])as z:group_cache[gid]={k:z[k].copy()for k in ('source_row_ids','component_row_index','component_directions','component_covariance_px2','actual_mean_uv')}
        data=group_cache[gid]
        if list(map(str,data['source_row_ids']))!=g['row_ids']:raise ValueError('Full component covariance source row order differs')
        for side,sid in enumerate(ids):
            f=factors[sid];uv=row['actual_pair_4D_UV'][side];j=g['row_ids'].index(sid);columns=np.flatnonzero(data['component_row_index']==j)
            if f['operator_kind']!='wide_paint_flank'or f['source_pair_id']!=pairid or f['source_uv']!=uv or f['current_uv']!=uv or f['joint_material_index']!=144:raise ValueError('C0 flattened factor redirects the actual ordered flank mean')
            if len(columns)!=1 or not np.array_equal(data['component_directions'][columns[0]],op['normal'])or np.max(abs(data['actual_mean_uv'][j]-uv))>1e-9:raise ValueError('C0 normal component/actual mean differs')
            sh=object_sha256(dict(native_pts=pts,source_uv=uv,source_row_id=sid));ch=object_sha256(dict(native_pts=pts,current_uv=uv,source_row_id=sid,measurement_operator='finite_width_paint_material_flank'))
            if g['source_measurements_sha256'][sid]!=sh or g['current_measurements_sha256'][sid]!=ch:raise ValueError('C0 W source/current mean differs')
        pairhash=object_sha256(dict(native_pts=pts,source_row_id=pairid,actual_flank_uv=row['actual_pair_4D_UV'],measurement_operator='finite_width_paint_material_flank_pair'))
        if g['current_pair_measurements_sha256'][pairid]!=pairhash:raise ValueError('Joint W ordered 4-D pair tuple differs')
        nr=g['native_receipt']
        if int(nr['native_pts'])!=pts or nr['shot_id']!=row['shot_id']or nr['decoded_native_RGB_sha256']!=fr['RGB_sha256']or nr['native_RGB_resampled']or not nr['training_role_verified']:raise ValueError('C0 W uses another native RGB exposure')
        receipt=dict(schema='rinkfit.full_wide_C0_pair_conditional_source_admission.v1',source_row_id=rid,pair_id=pairid,native_pts=pts,current_4D_source_mean_sha256=qhash,current_pair_measurements_sha256=pairhash,statistical_group_id=gid,statistics_definition_sha256=object_sha256(g),source_binding=dict(file=str(binding_file.resolve()),sha256=file_sha256(binding_file)),source_qualification_receipt_sha256=object_sha256(sourceq),finite_width_actual_flank_columns_retained=True,world_width_scope=binding['width_scope_id'],midpoint_used_as_world_projection=False,full_wide_branch_selection_replayed=True,source_branch_switch_count=g['wide_pair_branch_switch_counts'][pairid],exact_full_parent_covariance_marginal_required=True,physical_total_error_calibrated=False)
        receipts.append(receipt);suffix=dict(solver_eligible=True,numeric_statistics_ready=True,statistical_objective_eligible=True,statistics_use='conditional_noise_objective',statistic_kind='native_joint_readout',statistical_group_id=gid,source_admission_receipt_sha256=object_sha256(receipt))
        decisions.append(dict(decision,**suffix));approved.append(dict(spec,**suffix,current_4D_source_mean_sha256=qhash))
        solver.extend(dict(factors[sid],**suffix)for sid in ids)
    for p,sha in cache.items():
        if file_sha256(p)!=sha:raise ValueError('Source/statistics changed during pair admission')
    contract.verify_unchanged();out.mkdir();write_json(out/'geometry_pairs.json',dict(pairs=geometry));write_json(out/'approved_pairs.json',dict(pairs=approved));write_jsonl(out/'solver_rows.jsonl.gz',solver);write_jsonl(out/'promotion_decisions.jsonl.gz',decisions);write_jsonl(out/'receipts.jsonl.gz',receipts)
    report=dict(schema='rinkfit.current_full_wide_recovered_C0_pair_conditional_gate.v1',source_geometry_qualified_pairs=len(geometry),conditional_pairs=len(approved),conditional_rows=len(solver),pending_conditional_pairs=len(geometry)-len(approved),source_binding=dict(file=str(binding_file.resolve()),sha256=file_sha256(binding_file)),factor_rows=dict(file=str(Path(factor_rows).resolve()),sha256=file_sha256(factor_rows)),statistics_registry=dict(file=str(stats.resolve()),sha256=file_sha256(stats)),exact_parent_covariance_marginal_required=True,whole_exposure_cross_covariance_retained=True,source_UV_flags_or_mean_changed=False,physical_total_error_calibrated=False)
    write_json(out/'manifest.json',report);return report
