"""Seal native material tracklets as geometry-only source chunks.

This gate consumes exact original entity qualification, current source RGB
bindings, each actual reader result/replay, and a separately recorded native
pixel review. It never manufactures statistical precision or changes old UV.
"""
from pathlib import Path
from collections import Counter
import json
import numpy as np
from .io_utils import read_json, read_jsonl, write_json, write_jsonl, file_sha256, object_sha256
from .typed_entity_append import validate_new_entity


def seal_qualified_tracklets(acquisitions, review, output):
    out=Path(output)
    if out.exists():raise ValueError('New immutable tracklet qualification generation required')
    refs={}
    def ref(path):
        p=Path(path).resolve();h=file_sha256(p);refs[str(p)]=h
        return dict(file=str(p),sha256=h)
    def verify(r):
        p=Path(r.get('file',r.get('path')))
        if ref(p)['sha256']!=r['sha256']:raise ValueError('Changed native source artifact: '+str(p))
        return p
    rev=read_json(review);reviewref=ref(review)
    if rev.get('authority')!='assistant_native_pixel_review' or not rev.get('two_physical_ad_material_edges_reviewed'):
        raise ValueError('Actual native material review is required; numeric success is insufficient')
    for item in rev['evidence_refs']:verify(item)
    maskledger={}
    for source_ref in rev['source_bound_actor_ledgers']:
        for exposure in read_jsonl(verify(source_ref)):
            maskledger[int(exposure['native_pts'])]=exposure
    accepted=set(rev['accepted_source_row_ids'])
    rows=[];receipts=[];mapping={};counts=Counter();parents=[];model_ref=None
    for location in acquisitions:
        p=Path(location);a=read_json(p);parents.append(ref(p))
        if a.get('schema')!='rinkfit.native_fixed_entity_tracklets_acquisition.v1':raise ValueError('Wrong native acquisition type')
        verify(a['raw_source_rows']);verify(a['source_mean_replay_QA']);verify(a['anchor_proofs']);verify(a['source_reader'])
        model=read_json(verify(a['material_binding_model']))
        if model_ref is not None and model_ref!=a['material_binding_model']:raise ValueError('Tracklet material model differs')
        model_ref=a['material_binding_model'];anchors={x['source_row_id']:x for x in read_json(a['anchor_proofs']['file'])}
        replays={x['source_row_id']:x for x in read_json(a['source_mean_replay_QA']['file'])['rows']}
        for row in read_jsonl(a['raw_source_rows']['file']):
            rid=row['source_row_id']
            if rid not in accepted:continue
            if any(x['source_row_id']==rid for x in rows):raise ValueError('Repeated actual native readout')
            q=row['source_operator_result'];anchor=anchors[row['anchor_source_row_id']];qm=read_json(verify(anchor['parent_qualification_manifest']));qr=next(x for x in read_jsonl(verify(qm['qualification_receipts'])) if x['source_row_id']==anchor['source_row_id'])
            if object_sha256(qr)!=anchor['parent_qualification_record_sha256'] or not qr['new_actual_source_geometry_qualified'] or not qr['physical_point']:raise ValueError('Unqualified original fixed material identity')
            if not q['valid'] or q['source_mean_pixels_from_model'] or q['operator_version']!='native_two_edge_quadratic_intersection_v1' or q['reciprocal_error_px']>1 or q['patch_ncc']<.75:raise ValueError('Current native material tracklet did not close')
            if q['current_uv']!=row['current_uv'] or not replays[rid]['exact_current_source_operator_replay'] or replays[rid]['max_abs_replay_px']!=0:raise ValueError('Current actual mean/replay mismatch')
            b=row['native_source_binding']
            if b['source_video_sha256']!=a['source_video']['sha256'] or b['native_pts']!=row['native_pts'] or b['time_base']!='1/12800' or not b['training_role_verified']:raise ValueError('Wrong native source identity')
            ld=maskledger.get(int(row['native_pts']))
            if ld is None or ld['shot_id']!=row['shot_id']:raise ValueError('Missing independent current actor source ledger')
            qxy=np.asarray(row['current_uv']);halfwidth=30.
            for actor in ld.get('current_person_boxes',[]):
                ab=actor['source_binding']
                if ab['source_pts']!=row['native_pts'] or ab['decoded_native_rgb_sha256']!=b['decoded_native_RGB_sha256']:raise ValueError('Actor source ledger has wrong current RGB')
                x1,y1,x2,y2=actor['box_xyxy']
                if qxy[0]+halfwidth>=x1-4 and qxy[0]-halfwidth<=x2+4 and qxy[1]+halfwidth>=y1-4 and qxy[1]-halfwidth<=y2+4:raise ValueError('Current material support overlaps actor exclusion')
            fr=read_json(verify(row['feedback_receipt']['receipt']))
            if fr['actual_native_RGB_sha256']!=b['decoded_native_RGB_sha256'] or fr['source']['pts']!=row['native_pts'] or fr.get('blockers'):raise ValueError('Current feedback reader evidence is stale or blocked')
            mi=int(row['joint_material_index']);mat=model['materials'][mi]
            if mat['label']!=row['target_id'] or mi!=qr['existing_material_index']:raise ValueError('Cross-instance material transfer')
            path=next(x for x in mat['paths'] if x['id']==qr['existing_point_path'])
            if not path.get('physical_point'):raise ValueError('Static physical corner lost its fixed point path')
            receipt=dict(schema='rinkfit.native_tracklet_actual_type1_source_qualification.v1',source_row_id=rid,native_pts=row['native_pts'],target_id=row['target_id'],existing_material_index=mi,existing_point_path=qr['existing_point_path'],point_path_kind=path['kind'],physical_point=True,entity_kind=1,new_actual_source_geometry_qualified=True,entity_station_fixed=True,original_flags_preserved=dict(manual_reference_preserved=False),original_UV_preserved=row['source_uv'],current_2D_source_mean_sha256=object_sha256(dict(source_row_id=rid,native_pts=row['native_pts'],current_2D_UV=row['current_uv'],estimator='native_two_edge_quadratic_intersection_v1')),solver_eligible=False,statistical_objective_eligible=False,numeric_statistics_ready=False,statistics_status='unknown_for_current_generation',source_identity_anchor=anchor['parent_qualification_manifest'],source_identity_anchor_record_sha256=object_sha256(qr),native_pixel_review=reviewref,old_raw_source_UV_or_membership_changed=False)
            new=dict(row,physical_track_id=row['source_family_id'],source_family_id=row['source_family_id']+'/'+str(row['native_pts']),source_geometry_eligible=True,source_geometry_review_pending=False,source_qualification_receipt_sha256=object_sha256(receipt))
            validate_new_entity(new,receipt,mat,mi);rows.append(new);receipts.append(receipt);counts[row['shot_id']]+=1
            mapping[new['joint_material_id']]=dict(material_index=mi,geometry_ready=True)
    if accepted!={x['source_row_id'] for x in rows}:raise ValueError('Pixel review lists absent source IDs')
    out.mkdir(parents=True);write_jsonl(out/'source_rows.jsonl.gz',rows);write_jsonl(out/'geometry_rows.jsonl.gz',rows);write_jsonl(out/'qualification_receipts.jsonl.gz',receipts)
    write_json(out/'material_map.json',dict(semantic_to_material=mapping))
    replay=dict(schema='rinkfit.native_tracklet_current_mean_replay_QA.v1',source_acquisitions=parents,all_current_source_means_replayed=True,all_deterministic_replay_max_abs_px=0.,statistical_covariance_supplied=False);write_json(out/'source_mean_replay_QA.json',replay)
    qualification=dict(schema='rinkfit.new_actual_type1_source_qualification_gate.v1',input_cumulative_journal=ref(out/'source_rows.jsonl.gz'),qualification_receipts=ref(out/'qualification_receipts.jsonl.gz'),current_model=model_ref,source_mean_and_floating_estimator_replay_QA=ref(out/'source_mean_replay_QA.json'),old_raw_source_flags_false_are_not_overwritten=True,new_measurement_version_geometry_qualification_only=True,qualified_new_measurement_rows=len(rows),native_pixel_review=reviewref)
    write_json(out/'qualification_gate.json',qualification)
    manifest=dict(schema='rinkfit.actual_source_gate_reference_adapter.v1',all_requested_frames_accounted=True,source_dataset_tag='native_entity_tracklets_recovery_1019_v1',source_rows_ref=ref(out/'source_rows.jsonl.gz'),geometry_rows_ref=ref(out/'geometry_rows.jsonl.gz'),source_gate_manifest=ref(out/'qualification_gate.json'),actual_source_rows=len(rows),by_shot=dict(counts),all_new_sources_statistical_status='unknown_for_current_generation',current_source_geometry_qualification_is_not_noise_precision=True)
    write_json(out/'chunk_manifest.json',manifest)
    result=dict(schema='rinkfit.native_tracklet_qualified_source_bundle.v1',source_acquisitions=parents,source_review=reviewref,closed_chunks=[ref(out/'chunk_manifest.json')],typed_entity_qualification=[ref(out/'qualification_gate.json')],material_map=ref(out/'material_map.json'),source_rows=len(rows),by_shot=dict(counts),source_ledger_preserved=True,whole_same_exposure_statistical_status='unavailable_for_new_generation_engineering_only',model=model_ref)
    for path, digest in refs.items():
        if file_sha256(path)!=digest:raise ValueError('Source changed while sealing tracklets: '+path)
    write_json(out/'manifest.json',result);return result
