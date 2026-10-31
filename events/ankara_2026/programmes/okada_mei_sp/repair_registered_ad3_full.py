"""Stream original RGB Ad3 exterior repairs for current S07/S09 registration.

Existing whole-print source registrations navigate searches. This program never
loads fitted cameras, emits model UV, or deletes old measurements. Qualification
and packets remain separate. Per-frame actual profile proof preserves all reads.
"""
from pathlib import Path
import argparse, hashlib, json, shutil, time
from collections import Counter
import cv2
import numpy as np

from rinkspace.geometry import project
from rinkspace.session.registered_panel_edges import read_registered_panel_edges
from prepare_independent_navigation import frames
from capture_independent_prints import material_guard, source_patch_check
from capture_independent_materials import source_exclusion, spaced_indices, plain, write
from repair_registered_print_edges import sha
from audit_registered_exterior_rows import audit_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shots', nargs='+', default=['s09', 's07'])
    a = p.parse_args(); root=a.root.resolve(); out=a.output.resolve(); cv2.setNumThreads(2)
    if out.exists(): raise FileExistsError(out)
    if not a.shots or set(a.shots)-{'s07','s09'} or len(set(a.shots)) != len(a.shots):
        raise ValueError('Only distinct current S07/S09 shots may be repaired')
    index=json.loads((root/'index.json').read_text('utf8'))
    source=Path(index['source']['resolved_path']); source_sha=sha(source)
    if source_sha != index['source']['actual_sha256']: raise ValueError('Original source hash differs')
    registration=root/'registered_full'; source_binding=json.loads((registration/'binding.json').read_text('utf8'))
    if source_binding['source_sha256'] != source_sha: raise ValueError('Registration source hash differs')
    bank_file=root/'print_references/references.json'; bank=json.loads(bank_file.read_text('utf8'))
    references={}
    for meta in bank['references']:
        if meta['ad_id'] != 'Ad3': continue
        z=dict(np.load(meta['descriptor_npz'], allow_pickle=False))
        z['source_image']=cv2.imread(str(root/meta['shot_id']/'originals'/f'{meta["shot_id"]}_pts{meta["pts"]}.png'))
        z['reference_mask']=cv2.imread(meta['descriptor_mask'],cv2.IMREAD_GRAYSCALE)>0
        if hashlib.sha256(z['source_image'].tobytes()).hexdigest() != meta['source_BGR_sha256']:
            raise ValueError('Original reference pixels differ')
        if sha(meta['descriptor_npz']) != meta['descriptor_npz_sha256']: raise ValueError('Reference NPZ differs')
        references[meta['pts']]=(meta,z)
    project_root=Path(__file__).resolve().parents[4]
    implementation_files=[Path(__file__),Path(__file__).with_name('audit_registered_exterior_rows.py'),
        Path(__file__).with_name('capture_independent_prints.py'),Path(__file__).with_name('capture_independent_materials.py'),
        Path(__file__).with_name('prepare_independent_navigation.py'),
        project_root/'src/rinkspace/session/registered_panel_edges.py',
        project_root/'src/rinkspace/session/panel_boundaries.py',project_root/'src/rinkspace/session/material_tracks.py']
    implementation={str(f):sha(f) for f in implementation_files}
    binding=dict(schema='rinkspace.independent_material_capture.v1',source_sha256=source_sha,
        source_video=str(source),independent_source_capture=True,legacy_new_shot_inputs_read=False,
        appearance_references=str(bank_file),reference_bank_sha256=sha(bank_file),
        source_registration=str(registration),selected_shot_ids=a.shots,selected_entities=['Ad3'],
        implementation_sha256=implementation,source_feature_match_is_boundary_measurement=False,
        native_camera_prediction_is_identity_evidence=False,native_state_loaded=False,
        query_spacing_px=24.,fit_spacing_px=48.,short_feature_minimum_samples=4,
        no_native_residual_selection=True,fit_eligible=False,
        review_state='pending_machine_source_exterior_qualification')
    out.mkdir(parents=True);write(out/'binding.json',binding);summaries={};start=time.monotonic()
    for sid in a.shots:
        shot=next(s for s in index['shots']if s['shot_id']==sid);folder=out/sid;folder.mkdir();proof_dir=folder/'profiles';proof_dir.mkdir()
        reg_file=registration/sid/'registration.jsonl';reg_sha=sha(reg_file);cached={}
        with reg_file.open(encoding='utf8')as stream:
            for line_number,line in enumerate(stream,1):
                record=json.loads(line);items=[n for n in record['prints']if n.get('ad')=='Ad3'
                    and n.get('status')=='source_content_registered'and n.get('source_pattern_check',{}).get('supported')]
                if len(items)==1:
                    n=items[0];cached[record['pts']]=dict(reference_pts=n['reference_pts'],
                        H_current_from_reference=n['H_current_from_reference'],source_pattern_check=n['source_pattern_check'],
                        parent_source_graph_pts=n.get('source_graph_parent_pts'),source_registration_line=line_number)
        old_file=root/'qualified_full'/sid/'source_rows.jsonl';old_sha=sha(old_file);old={}
        with old_file.open(encoding='utf8')as stream:
            for line_number,line in enumerate(stream,1):
                r=json.loads(line);rows=[(i,q)for i,q in enumerate(r['rows'])if q.get('entity')=='Ad3']
                if rows:old[r['pts']]=(line_number,r['source_BGR_sha256'],rows)
        old_packet=root/'packets_full'/sid/'source_row_manifest.jsonl';old_locations={}
        with old_packet.open(encoding='utf8')as stream:
            for line in stream:
                loc=json.loads(line)
                if loc['path_id'].startswith('print_instance/Ad3::'):
                    old_locations.setdefault((loc['pts'],loc['path_id'],tuple(loc['uv'])),[]).append(loc)
        counts=Counter();coverage=[];by_path=Counter();source_retractions=Counter()
        with (folder/'source_rows.jsonl').open('w',encoding='utf8',buffering=1)as rows_out, (folder/'registration.jsonl').open('w',encoding='utf8',buffering=1)as audit, (folder/'old_row_retraction_evidence.jsonl').open('w',encoding='utf8',buffering=1)as retractions:
            for frame_index,(pts,image)in enumerate(frames(index,shot)):
                rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);bgr_sha=hashlib.sha256(image.tobytes()).hexdigest()
                mask,mask_path=source_exclusion(root,sid,pts);mask_sha=sha(mask_path)
                rows=[];notes=[];proof=None;record=None;topology=None
                if pts in cached:
                    item=cached[pts];meta,z=references[item['reference_pts']];h=np.asarray(item['H_current_from_reference'])
                    check=source_patch_check(image,mask,meta,z,h);counts['registered_support_frames']+=1
                    note=dict(ad='Ad3',reference_pts=meta['pts'],H_current_from_reference=h.tolist(),
                        source_registration_line=item['source_registration_line'],source_registration_sha256=reg_sha,
                        cached_pattern_check=item['source_pattern_check'],source_pattern_check=check,
                        source_graph_parent_pts=item['parent_source_graph_pts'],registration_is_navigation_only=True)
                    if check['supported']:
                        counts['current_pattern_supported_frames']+=1
                        uv=project(h,z['projected_uv']);offset=z['edge_offsets'];paths=[uv[offset[j]:offset[j+1]]for j in range(4)]
                        proof_arrays={}
                        def collect(stage,arrays):
                            nonlocal topology
                            if stage=='source_topology':topology=arrays
                            proof_arrays.update({stage+'__'+key:value for key,value in arrays.items()})
                        local,record=read_registered_panel_edges(rgb,paths,mask,binding=dict(source_sha256=source_sha,
                            source_pts=pts,time_base='1/12800',shot_id=sid,independent_source_pattern_match_verified=True),
                            search_radius=20.,spacing=24.,candidate_material_validator=lambda row:material_guard(rgb,row,'Ad3')[0],proof_sink=collect)
                        proof_path=proof_dir/f'{pts}.npz';np.savez_compressed(proof_path,**proof_arrays)
                        proof=dict(path=str(proof_path),sha256=sha(proof_path),
                            profile_arrays_prefix='actual_source_envelope_search__',topology_arrays_prefix='source_topology__')
                        counts['actual_RGB_candidates']+=record['candidate_count'];counts['organized_source_candidates']+=len(local)
                        selected={c['candidate_id']:c for c in record['candidate_records']if c['candidate_id']in record['selected_candidate_ids']}
                        for ci,path_id in enumerate(meta['native_edge_path_ids']):
                            group=[q for q in local if q['curve']==ci]
                            for at in spaced_indices([q['uv']for q in group],spacing=48.):
                                q=group[at];trace=next(c for c in selected.values()if c['curve']==ci and c['uv']==q['uv'])
                                rows.append(dict(q,path_id=path_id,entity='Ad3',capture_family='source_registered_exterior_repair',
                                    source_candidate_id=trace['candidate_id'],source_profile_query=trace['stage_query'],
                                    source_topology_probes=trace['source_topology_probes'],actual_source_sides=trace['actual_source_sides'],
                                    source_colour=material_guard(rgb,q,'Ad3')[1],source_reference_pts=meta['pts'],source_profile_proof=proof))
                        note.update(status='source_content_registered',source_readout=record,profile_proof=proof)
                    else:note['status']='source_pattern_pending'
                    notes.append(note)
                if pts in old:
                    line_number,old_bgr,old_rows=old[pts]
                    evidence=audit_rows(rgb,topology,mask,old_rows,record['candidate_records']if record else [],
                        source_hash_matches=old_bgr==bgr_sha)
                    for e in evidence:
                        e['old_observation_locations']=old_locations.get((pts,e['path_id'],tuple(e['uv'])),[])
                    for path_id in sorted({q['path_id']for _,q in old_rows}):
                        chosen=[e for e in evidence if e['path_id']==path_id];new_count=sum(q['path_id']==path_id for q in rows)
                        group=dict(shot_id=sid,pts=pts,path_id=path_id,old_source_capture=str(old_file),old_source_capture_sha256=old_sha,
                            old_source_line=line_number,source_BGR_sha256=bgr_sha,source_hash_matches=old_bgr==bgr_sha,
                            source_only_retraction_evidence=True,no_replacement_is_not_withdrawal_evidence=True,
                            old_rows=len(chosen),source_verified_bad_rows=sum(e['withdrawal_authorized_by_this_audit']for e in chosen),
                            keep_unproven_old_rows=sum(not e['withdrawal_authorized_by_this_audit']for e in chosen),
                            new_source_replacement_rows=new_count,empty_group_replacement_source_supported=bool(not new_count and chosen
                                and all(e['withdrawal_authorized_by_this_audit']for e in chosen)),profile_proof=proof,evidence=chosen)
                        retractions.write(json.dumps(plain(group),ensure_ascii=False)+'\n')
                        source_retractions['source_verified_bad_rows']+=group['source_verified_bad_rows']
                        source_retractions['kept_unproven_old_rows']+=group['keep_unproven_old_rows']
                        source_retractions['empty_group_replacement_source_supported']+=group['empty_group_replacement_source_supported']
                row_record=dict(pts=pts,source_BGR_sha256=bgr_sha,source_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                    source_video_sha256=source_sha,mask_path=str(mask_path),mask_sha256=mask_sha,
                    fit_eligible=False,review_state='pending_machine_source_exterior_qualification',rows=rows,pending_rows=[],
                    source_profile_proof=proof,all_actual_candidates_retained_at=str(folder/'registration.jsonl'))
                rows_out.write(json.dumps(plain(row_record),ensure_ascii=False)+'\n')
                audit.write(json.dumps(plain(dict(pts=pts,prints=notes,fit_eligible=False)),ensure_ascii=False)+'\n')
                counts['native_frames']+=1;counts['frames_with_new_rows']+=bool(rows);counts['rows']+=len(rows)
                by_path.update(q['path_id']for q in rows);coverage.append(dict(pts=pts,rows_by_path=dict(Counter(q['path_id']for q in rows))))
                if frame_index%100==0:print(json.dumps(dict(shot=sid,frames=counts['native_frames'],rows=counts['rows'],source_supported=counts['current_pattern_supported_frames'],seconds=time.monotonic()-start)),flush=True)
        # Packet motion is disposable initialization; source-update preserves the
        # currently fitted controls and does not read these packet pose values.
        pose_source=root/sid/'navigation/poses_reanchored.npz';shutil.copyfile(pose_source,folder/'poses.npz')
        summary=dict(counts,rows_by_path=dict(by_path),source_only_old_row_audit=dict(source_retractions),
            old_source_capture=str(old_file),old_source_capture_sha256=old_sha,
            old_packet_source_manifest=str(old_packet),old_packet_source_manifest_sha256=sha(old_packet),
            source_registration=str(reg_file),source_registration_sha256=reg_sha,
            pose_initialization_source=str(pose_source),pose_initialization_source_sha256=sha(pose_source),
            pixel_fit_executed=False,fit_eligible=False)
        write(folder/'coverage.json',dict(summary,frames=coverage));summaries[sid]=summary
        write(out/'report.json',dict(binding,shots=summaries,seconds=time.monotonic()-start))
    for path,digest in implementation.items():
        if sha(path)!=digest:raise RuntimeError('Implementation changed during capture: '+path)
    write(out/'report.json',dict(binding,shots=summaries,seconds=time.monotonic()-start,complete=True))
    print(json.dumps(dict(output=str(out),complete=True,shots=summaries),ensure_ascii=True),flush=True)


if __name__=='__main__':main()