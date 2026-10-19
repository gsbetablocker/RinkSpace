"""Bind already reviewed RGB exclusions to immutable original FIT sources.

No new pixel reading, fitting, profile update or solver. Slot verification
reconstructs only absolute group order, using saved complete block/motion sizes.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from calibration.solver_runtime import install, status
install()
from calibration.solver_runtime.original_source_point_qualification import FAMILY, OriginalSourcePointQualification
from calibration.solver_runtime.source_point_qualification import SCHEMA, file_sha256, original_uv_sha256
from structured_programme_source_candidate.original_ice_pixel_correction import install_original_ice_pixel_correction
from z190_streaming_source_normal import SourceRowBlock

BASE=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
RUN=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/priority_witness_complete_cycle_qc38_20261010')
EVIDENCE=BASE/'original_F2_pending_point_qualification_evidence_20261010.json'
PIXEL_CORRECTION=BASE/'original_connected_F2_and_direct_user_ice_correction_DERIVED_20261010.json'
OUTPUT=BASE/'original_F2_pending_point_qualification_DERIVED_20261010.json'
RECEIPT=BASE/'original_F2_pending_point_qualification_interface_verification_20261010.json'


def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,nargs='+',default=[EVIDENCE])
    parser.add_argument('--pixel-correction',type=Path,default=PIXEL_CORRECTION)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--receipt',type=Path,default=RECEIPT)
    parser.add_argument('--previous-qualification',type=Path)
    parser.add_argument('--update-config',type=Path)
    opt=parser.parse_args(argv)
    proofs=[(p,json.loads(p.read_text(encoding='utf8'))) for p in opt.evidence]
    evidence=proofs[0][1]
    parent=evidence['original_absolute_packet']
    if file_sha256(parent['path'])!=parent['sha256']: raise ValueError('Original packet changed')
    original=pickle.loads(Path(parent['path']).read_bytes())['programme_absolute_rows']
    records=[]
    for evidence_path,proof in proofs:
        if (proof['original_absolute_packet']!=parent
                or proof['source_video_sha256']!=evidence['source_video_sha256']
                or proof['time_base']!=evidence['time_base']):
            raise ValueError('Merged source qualification must retain one exact original packet and clock')
        for entry in proof['entries']:
            item=original[entry['original_packet_row_index']]; row=item['source_row']
            indices=entry['excluded_point_indices']
            if (original_uv_sha256(row)!=entry['original_full_uv_float64_sha256']
                    or not np.array_equal(np.asarray(row['uv'])[indices],entry['original_selected_uv'])
                    or not np.array_equal(np.asarray(row['sigma'])[indices],entry['original_selected_sigma'])):
                raise ValueError('Reviewed original UV/sigma identity changed')
            raw=entry['source_RGB']
            record=dict(original_packet_row_index=entry['original_packet_row_index'],
                locator={k:item[k] for k in ('block','collection_role','group_actual_pts','group_order','row_order')},
                row_id=entry['original_id'],source_pts=entry['original_pts'],
                original_uv_float64_sha256=entry['original_full_uv_float64_sha256'],
                excluded_point_indices=indices,review_basis='original_rgb_only',reason=entry['reason'],
                decision_is_not_from_model_residual_or_cost=entry['decision_is_not_from_model_residual_or_cost'],
                review_authority=dict(path=str(evidence_path),sha256=file_sha256(evidence_path)),
                native_rgb_evidence=[dict(path=raw['path'],sha256=raw['sha256'],source_pts=raw['pts'],
                    source_video_sha256=evidence['source_video_sha256'],time_base=evidence['time_base'])])
            records.append(record)
    manifest=dict(schema=SCHEMA,source_family=FAMILY,
        original_packet=dict(**parent,format='original_absolute_compact_pickle'),
        source_video_sha256=evidence['source_video_sha256'],time_base=evidence['time_base'],records=records,
        original_packet_bytes_modified=False,UV_sigma_world_split_modified=False,
        replacement_observation_added=False,qualification_is_source_only_not_camera_accuracy=True)
    write(opt.output,manifest)
    # Saved source summaries provide actual post-compaction block sizes. They
    # are not interpreted as a new graph or used to compute a camera objective.
    input_path=RUN/'input_manifest.json'; block_path=RUN/'actual_backend_and_source_continuity.json'
    source_input=json.loads(input_path.read_text(encoding='utf8'))
    saved=json.loads(block_path.read_text(encoding='utf8'))['actual_family_rows'][:6]
    camera_path=Path(json.loads((RUN/'latest_source_review.json').read_text(encoding='utf8'))['camera_path'])
    with np.load(camera_path,allow_pickle=False) as camera:
        metadata=json.loads(str(camera['metadata_json']))
        if (metadata['source_sha256']!=evidence['source_video_sha256']
                or metadata['source_time_base']!=evidence['time_base']):
            raise ValueError('Saved complete camera exposure inventory belongs to another source')
        aliases={int(p):int(a) for p,a in zip(camera['pts'],camera['exposure_representative_pts'])}
    blocks=[]
    for index,ledger in enumerate(saved):
        name=ledger['family'].split(':',1)[1]
        items=[x for x in original if x['block']==name and x['collection_role']=='FIT']
        groups=[None]*(max(x['group_order'] for x in items)+1)
        for item in items:
            group=int(item['group_order']); position=int(item['row_order']); actual=int(item['group_actual_pts'])
            if groups[group] is None: groups[group]=(actual,[])
            pts,rows=groups[group]
            if pts!=actual or position!=len(rows): raise ValueError('Original group order changed')
            rows.append(deepcopy(item['source_row']))
            source=int(item['source_row']['pts'])
            if source in aliases and aliases[source]!=actual: raise ValueError('Original source aliases conflict')
            aliases[source]=actual; aliases[actual]=actual
        if any(g is None for g in groups): raise ValueError('Original FIT group absent')
        blocks.append(SimpleNamespace(name=name,groups=groups,
            bundle=SimpleNamespace(motion_rows=source_input['original_motion_rows'][index]),
            quadratic=np.zeros(ledger['original_stop']-ledger['original_start'],bool)))
    state=SimpleNamespace(blocks=blocks,original=SimpleNamespace(aliases=aliases))
    qualification=OriginalSourcePointQualification(opt.output)
    qualification.qualify_state(state)
    original_slots=qualification.contract['active_runtime_global_scalar_slots']
    # Reuse the actual pixel installer before mask binding, verifying that all
    # All certified points (including direct user drawings) stay active.
    correction=install_original_ice_pixel_correction(state,opt.pixel_correction)
    qualification=OriginalSourcePointQualification(opt.output)
    qualification.qualify_state(state)
    if original_slots!=qualification.contract['active_runtime_global_scalar_slots']:
        raise ValueError('Pixel correction changed original source scalar layout')
    masks=[]
    for (family,start,stop),slots in qualification._blocks.items():
        # This probes structural mask algebra/layout only. It is deliberately
        # not a native source cost/accuracy measurement or full state replay.
        local=slots-start
        residual=np.ones(stop-start)
        matrix=csr_matrix((np.arange(1.,len(local)+1.),(local,np.zeros(len(local),int))),shape=(stop-start,1))
        before=SourceRowBlock(residual,matrix,np.zeros(stop-start,bool),np.full(stop-start,3.),
            dict(family=family,original_start=start,original_stop=stop,source_sha256=evidence['source_video_sha256']))
        after,key=qualification.block(before,True)
        if (key!=(family,start,stop) or np.any(after.r[local]) or after.J.nnz!=0
                or not np.all(before.r==1.) or before.J.nnz!=len(local)
                or after.scales is not before.scales or after.quadratic is not before.quadratic):
            raise ValueError('Source mask scalar preservation failed')
        masks.append(dict(family=family,block_start=start,block_stop=stop,
            global_scalar_slots=slots.tolist(),inactive_R_and_J_structural_zero=True,
            original_buffers_masks_and_scales_preserved=True))
    report=dict(schema=SCHEMA+'.real-source-interface-verification',
        runtime_status=status(),packet=dict(path=str(opt.output),sha256=file_sha256(opt.output)),
        source_evidence=[dict(path=str(p),sha256=file_sha256(p)) for p,_ in proofs],
        saved_complete_input=dict(path=str(input_path),sha256=file_sha256(input_path)),
        saved_complete_block_summary=dict(path=str(block_path),sha256=file_sha256(block_path)),
        saved_complete_exposure_alias_inventory=dict(path=str(camera_path),sha256=file_sha256(camera_path)),
        actual_original_packet_and_full_FIT_group_order_verified=True,
        original_compacted_motion_prefix_sizes_from_complete_saved_state=True,
        source_only_qualification=qualification.contract,masks=masks,
        certified_pixel_correction_composition=dict(path=str(opt.pixel_correction),sha256=file_sha256(opt.pixel_correction),
            rows=len(correction['corrected_rows']),points=sum(len(x['point_indices'])for x in correction['corrected_rows']),
            all_certified_corrected_and_user_points_unretired=True),
        structural_mask_probe_is_not_native_cost_or_accuracy=True,
        full_programme_graph_loaded=False,production_solver_started=False,
        fresh_production_profiles_J_and_truth_target_revision_still_required=True,accuracy_accepted=False)
    if opt.previous_qualification is not None:
        previous=OriginalSourcePointQualification(opt.previous_qualification)
        previous.qualify_state(state)
        from structured_programme_source_candidate.truth_source_role_migration import target
        def mask_target(contract):
            # Probe the production target-hash function with the actual
            # qualification contract. This is not a new full graph manifest.
            return target(dict(full_parameters=source_input['full_parameters'],source_rows=source_input['source_rows'],
                source_contract=contract))
        before,after=mask_target(previous.contract),mask_target(qualification.contract)
        if before==after: raise ValueError('Source target fingerprint failed to track changed point eligibility')
        report['source_target_fingerprint_probe']=dict(previous_qualification=dict(path=str(opt.previous_qualification),
            sha256=file_sha256(opt.previous_qualification)),previous_mask_target=before,new_mask_target=after,
            parameter_and_row_counts_unchanged=True,mask_contract_SHA_changes_production_target_hash=True,
            probe_is_not_actual_new_complete_startup_target=True,
            old_complete_truth_target_requires_explicit_same_q_revision=True,
            no_old_complete_linearization_or_append_cache_reuse_authorized=True)
    if opt.update_config is not None:
        config=json.loads(opt.update_config.read_text(encoding='utf8'))
        original_config=deepcopy(config)
        source=config['source_options']
        before_options=deepcopy(source)
        old_correction=source.get('--original-ice-pixel-correction-source')
        old_qualification=source.get('--original-source-point-qualification-source')
        source['--original-ice-pixel-correction-source']=str(opt.pixel_correction.resolve())
        source['--original-source-point-qualification-source']=str(opt.output.resolve())
        details=config['continuation']['source_RGB_qualification_20261010']
        details.update(previous_original_pixel_correction_source=old_correction,
            original_pixel_correction_source=str(opt.pixel_correction.resolve()),
            original_pixel_correction_source_sha256=file_sha256(opt.pixel_correction),
            original_pixel_corrected_rows=len(correction['corrected_rows']),
            original_pixel_corrected_points=sum(len(x['point_indices']) for x in correction['corrected_rows']),
            previous_original_source_point_qualification=old_qualification,
            original_source_point_qualification_source=str(opt.output.resolve()),
            original_source_point_qualification_source_sha256=file_sha256(opt.output),
            original_source_point_qualification_selected=True,
            original_absolute_inactive_points=qualification.contract['active_FIT_points_retired'],
            original_absolute_inactive_scalars=qualification.contract['active_FIT_scalars_retired'],
            source_correction_and_qualification_composition_verified=True,
            composition_verification_receipt=str(opt.receipt.resolve()),
            next_start_requires_fresh_profiles_and_J=True,
            complete_truth_target_revision_required=True,solver_run_started=False)
        changed={key for key in set(before_options)|set(source) if before_options.get(key)!=source.get(key)}
        if changed-{'--original-ice-pixel-correction-source','--original-source-point-qualification-source'}:
            raise ValueError('Source qualification unexpectedly changed another production option')
        for key in ('complete_accepted_state','checkpoint_file_sha256','truth_restoration_state',
                'truth_restoration_state_sha256','state'):
            if config['continuation'][key]!=original_config['continuation'][key]:
                raise ValueError('Source qualification changed production seed or optimizer identity')
        seed_keys=('--camera-seed','--world-init','--truth-restoration-state','--initial-damping')
        if any(source.get(key)!=before_options.get(key) for key in seed_keys):
            raise ValueError('Source qualification changed q seed/truth/damping')
        write(opt.update_config,config)
        report['selected_production_input_configuration']=dict(path=str(opt.update_config.resolve()),
            sha256=file_sha256(opt.update_config),changed_source_options=sorted(changed),
            camera_world_truth_seed_damping_and_complete_checkpoint_identity_unchanged=True,
            source_target_changed_no_old_J_profile_reuse=True,production_solver_started=False)
    write(opt.receipt,report)
    print(json.dumps(dict(packet=str(opt.output),receipt=str(opt.receipt),points=qualification.contract['active_FIT_points_retired'],
        global_scalar_slots=original_slots),ensure_ascii=True))


if __name__=='__main__': main()
