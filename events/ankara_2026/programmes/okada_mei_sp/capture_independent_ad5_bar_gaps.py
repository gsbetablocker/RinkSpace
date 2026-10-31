"""Source-supported Ad5 blue/green bar boundaries in three approved short gaps.

Reference image motion navigates the known source contours only. Original RGB
q50s, unchanged public closed-shape qualification, actual colour and current SAM
support supply every row. Reviewed pilot q50s are retained exactly; no weights.
"""
from pathlib import Path
import argparse,copy,hashlib,json,shutil,time
from collections import Counter
import cv2,numpy as np
from probe_independent_ad5_bars import CASES,actual_guard
from capture_independent_badge import exact_frames,sha,write
from capture_independent_badge_full import read_profile_proof
from capture_independent_ice import load_navigation_graph,navigation_between
from capture_independent_materials import source_exclusion,spaced_indices,plain
from capture_independent_prints import source_patch_check
from rinkspace.camera_reuse_localization import descriptors,match_features,ReuseConfig
from rinkspace.geometry import project
from build_independent_shot_packets import build

WINDOWS={'s07':(1702912,1705216),'s08':(1777152,1781248),'s09':(1887488,1891328)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.root.resolve();out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    cv2.setNumThreads(2);started=time.monotonic();index=json.loads((root/'index.json').read_text('utf8'));video=Path(index['source']['resolved_path'])
    if sha(video)!=index['source']['actual_sha256']:raise ValueError('Actual original source changed')
    pilot=root/'ad5_bars_pilot';pilot_binding=json.loads((pilot/'binding.json').read_text('utf8'));pilot_records={r['shot_id']:r for l in(pilot/'source_rows.jsonl').open(encoding='utf8')if(r:=json.loads(l))};out.mkdir(parents=True);capture=out/'capture';qualified=out/'qualified';capture.mkdir();qualified.mkdir()
    approval=dict(authority='Explicit current main-session original-image pilot review and task instruction',physical_instance='Ad5',native_paths=pilot_binding['native_paths'],known_rigs_and_Ad4_adjacent_source_context=pilot_binding['physical_identity_authority'],physical_instance_layout_check=pilot_binding['layout_check'],approved_intervals={sid:dict(first_pts=v[0],last_pts=v[1],native_step=256,frames=(v[1]-v[0])//256+1)for sid,v in WINDOWS.items()},reviewed_pilot_source_rows=str(pilot/'source_rows.jsonl'),reviewed_pilot_source_rows_sha256=sha(pilot/'source_rows.jsonl'),pilot_RGB_proofs={sid:str(pilot/f'{sid}_{r["pts"]}_actual_bars_crop.png')for sid,r in pilot_records.items()},original_51_pilot_q50s_must_be_preserved=True,not_a_repeated_text_only_identity_assignment=True)
    approval_path=out/'source_identity_approval.json';write(approval_path,approval);approval_sha=sha(approval_path)
    files=[Path(__file__),Path(actual_guard.__code__.co_filename),Path(read_profile_proof.__code__.co_filename),Path(source_exclusion.__code__.co_filename),Path(source_patch_check.__code__.co_filename),Path(build.__code__.co_filename)]
    binding=dict(source_video_sha256=index['source']['actual_sha256'],source_sha256=index['source']['actual_sha256'],independent_source_capture=True,legacy_new_shot_point_inputs_read=False,native_camera_projection_called=False,new_weights_created=False,source_identity_approval=str(approval_path),source_identity_approval_sha256=approval_sha,source_pilot=str(pilot),source_pilot_binding_sha256=sha(pilot/'binding.json'),source_query_spacing_px=24.,target_source_spacing_px=48.,short_closed_shape_minimum_samples=4,public_reader_thresholds_changed=False,physical_identity_scope=approval['approved_intervals'],native_paths=pilot_binding['native_paths'],fit_eligible=False,fit_executed=False,implementation_sha256={str(p.resolve()):sha(p)for p in files})
    write(capture/'binding.json',binding);summaries={};pilot_preserved=0;total=0;proof_paths=[]
    for sid,refpts,blue_roi,green_roi in CASES:
        first,last=WINDOWS[sid];requested=list(range(first,last+256,256));images=exact_frames(video,requested);ref_record=pilot_records[sid];reference=images[refpts]
        if hashlib.sha256(reference.tobytes()).hexdigest()!=ref_record['source_BGR_sha256']:raise ValueError('Reviewed pilot original RGB changed')
        reference_exclusion,_=source_exclusion(root,sid,refpts);bounds=np.array([blue_roi,green_roi]);roi=[*bounds[:,:2].min(0).tolist(),*bounds[:,2:].max(0).tolist()];x0,y0,x1,y1=roi
        patch_mask=np.zeros((1080,1920),'uint8');patch_mask[y0:y1+1,x0:x1+1]=255;patch_mask[reference_exclusion]=0;context_mask=np.zeros_like(patch_mask);context_mask[max(0,y0-48):min(1080,y1+49),max(0,x0-48):min(1920,x1+49)]=255;context_mask[reference_exclusion]=0
        reference_features=descriptors(reference,context_mask);photo=dict(source_image=reference,reference_mask=patch_mask>0);meta=dict(roi_xyxy_inclusive=roi);shot=next(s for s in index['shots']if s['shot_id']==sid);graph,graph_note=load_navigation_graph(root,shot)
        search={c:np.load(pilot/'profiles'/f'{sid}_{refpts}_{c}.npz',allow_pickle=False)['search_uv']for c in('blue','green')}
        folder=capture/sid;folder.mkdir();(folder/'profiles').mkdir();qfolder=qualified/sid;qfolder.mkdir();np.savez_compressed(folder/'reference_RootSIFT_navigation.npz',feature_uv=reference_features[0],descriptors=reference_features[1]);write(folder/'reference_source.json',dict(shot=sid,pts=refpts,source_BGR_sha256=ref_record['source_BGR_sha256'],original_source=ref_record['source_original'],photo_ROI=roi,descriptor_context_margin_px=48,context_features_are_navigation_only=True,source_graph=graph_note))
        frames=[];by_path=Counter();raw_count=0;failures=Counter();frame_rows={}
        with(folder/'source_rows.jsonl').open('x',encoding='utf8')as raw_out,(qfolder/'source_rows.jsonl').open('x',encoding='utf8')as rows_out,(folder/'registration.jsonl').open('x',encoding='utf8')as notes_out:
            for pt in requested:
                image=images[pt];rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);ex,mp=source_exclusion(root,sid,pt);rows=[];pending=[];bar_notes=[];attempts=[];h=None;patch=None;basis=None;root_matches=0
                if pt==refpts:
                    h=np.eye(3);patch=source_patch_check(image,ex,meta,photo,h);basis='exact_reviewed_pilot_source_rows_retained';rows=copy.deepcopy(ref_record['rows']);pending=copy.deepcopy(ref_record['pending_rows'])
                    for row in rows:
                        guard=actual_guard(row,ex,row['bar_colour'])
                        if guard['guard_reasons']:raise ValueError('Current fine source guard no longer supports an approved pilot row')
                        row.update(fit_eligible=True,review_state='source_supported_approved_Ad5_bar_boundary',source_identity_approval=str(approval_path),source_identity_approval_sha256=approval_sha)
                    for before,after in zip(ref_record['rows'],rows,strict=True):
                        for key in('uv','normal','actual_source_quantiles','actual_source_sides','candidate_flags'):assert json.dumps(before[key],sort_keys=True)==json.dumps(after[key],sort_keys=True)
                    pilot_preserved+=len(rows)
                else:
                    hh,reason=navigation_between(graph,refpts,pt)
                    if hh is not None:
                        check=source_patch_check(image,ex,meta,photo,hh);attempts.append(dict(basis='supported_source_graph_navigation',navigation_status=reason,H_current_from_reference=hh.tolist(),actual_source_photo_check=check))
                        if check['supported']:h=hh;patch=check;basis='supported_source_graph_and_current_original_photo'
                    if h is None:
                        matched=match_features(reference_features,descriptors(image,(~ex).astype('uint8')*255),ReuseConfig(min_matches=6,ransac_px=3.));root_matches=0 if matched is None else len(matched[0])
                        if matched is not None:
                            hh=matched[2];check=source_patch_check(image,ex,meta,photo,hh);attempts.append(dict(basis='public_RootSIFT_source_context',matches=len(matched[0]),H_current_from_reference=hh.tolist(),actual_source_photo_check=check))
                            if check['supported']:h=hh;patch=check;basis='public_RootSIFT_context_and_current_original_photo'
                    if h is not None:
                        for colour in('blue','green'):
                            geometry=project(h,search[colour]);profile=folder/'profiles'/f'{pt}_{colour}.npz';public,readout,raw,arrays=read_profile_proof(rgb,geometry,ex,dict(source_sha256=index['source']['actual_sha256'],source_pts=pt,time_base='1/12800',shot_id=sid,decoded_native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()),material_signal='blue'if colour=='blue'else'dark');np.savez_compressed(profile,**arrays);raw_count+=len(raw);selected_uv={tuple(r['uv'])for r in public};eligible=[];raw_rows=[]
                            for row in raw:
                                np.testing.assert_array_equal(row['uv'],np.array(row['actual_source_quantiles'])[0,1]);guard=actual_guard(row,ex,colour);reasons=list(guard['guard_reasons']);organized=tuple(row['uv'])in selected_uv
                                if not organized:reasons.append('not_selected_by_unchanged_public_closed_native_shape_organization')
                                item=dict(row,path_id=pilot_binding['native_paths'][colour],entity='Ad5',bar_colour=colour,source_profile_proof=str(profile),source_profile_sha256=sha(profile),actual_source_guard=guard,public_closed_shape_qualified=organized,machine_certifiable_source_boundary=not reasons,pending_reasons=reasons,fit_eligible=False,review_state='raw_actual_source_boundary_candidate',native_material_metadata=pilot_binding['native_path_metadata'][colour],source_identity_approval=str(approval_path),source_identity_approval_sha256=approval_sha,native_or_H_vertices_are_not_observations=True)
                                raw_rows.append(item);failures.update(reasons)
                                if not reasons:eligible.append(item)
                            for i in spaced_indices([r['uv']for r in eligible],spacing=48.):rows.append(dict(eligible[i],fit_eligible=True,review_state='source_supported_approved_Ad5_bar_boundary'))
                            pending.extend(raw_rows);bar_notes.append(dict(colour=colour,source_profile_proof=str(profile),public_reader_record=readout,all_actual_exterior_candidates=raw_rows,machine_certifiable_before_spacing=len(eligible)))
                    else:failures['no_supported_actual_source_photo_navigation']+=1
                record=dict(shot_id=sid,pts=pt,source_video_sha256=index['source']['actual_sha256'],source_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),source_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),mask_path=str(mp),mask_sha256=sha(mp),rows=rows,pending_rows=pending,fit_eligible=bool(rows),review_state='source_supported_approved_Ad5_bar_boundary'if rows else'pending_no_supported_source_boundary',source_identity_approval=str(approval_path),source_identity_approval_sha256=approval_sha,reference_source_pts=refpts,original_reviewed_pilot_source=str(pilot/'source_rows.jsonl'),original_pilot_source_sha256=sha(pilot/'source_rows.jsonl'))
                rows_out.write(json.dumps(plain(record),ensure_ascii=False,allow_nan=False)+'\n');raw_out.write(json.dumps(plain(dict(record,rows=pending,fit_eligible=False)),ensure_ascii=False,allow_nan=False)+'\n');notes_out.write(json.dumps(plain(dict(pts=pt,source_navigation_basis=basis,H_current_from_reference=None if h is None else h.tolist(),photo_check=patch,navigation_attempts=attempts,bars=bar_notes,exact_pilot_rows_preserved=pt==refpts,source_graph=graph_note)),ensure_ascii=False,allow_nan=False)+'\n')
                by_path.update(r['path_id']for r in rows);frame_rows[pt]=record;frames.append(dict(pts=pt,rows=len(rows),rows_by_native_path=dict(Counter(r['path_id']for r in rows)),source_navigation_basis=basis,photo_status=None if patch is None else patch['status'],RootSIFT_matches=root_matches));total+=len(rows);print(json.dumps(dict(shot=sid,pts=pt,rows=len(rows),basis=basis)),flush=True)
        pose=root/sid/'navigation/poses_reanchored.npz';shutil.copyfile(pose,qfolder/'poses.npz')
        for pt in(first,last):
            record=frame_rows[pt];original=out/f'{sid}_{pt}_original.png';cv2.imwrite(str(original),images[pt]);proof=images[pt].copy()
            for row in record['rows']:
                cv2.circle(proof,tuple(np.rint(row['uv']).astype(int)),3,(0,255,0),1,cv2.LINE_AA)
                for side in np.array(row['actual_source_quantiles']):cv2.line(proof,tuple(np.rint(side[0]).astype(int)),tuple(np.rint(side[2]).astype(int)),(255,0,255),1)
            cv2.putText(proof,f'Ad5 blue/green source q50 | {sid} PTS {pt} | {len(record["rows"])} source-supported | NO FIT',(20,35),0,.65,(0,220,255),2);full=out/f'{sid}_{pt}_source_supported_bars.jpg';cv2.imwrite(str(full),proof);crop=None
            if record['rows']:
                uv=np.array([r['uv']for r in record['rows']]);lo=np.maximum(np.floor(uv.min(0)-25).astype(int),[0,0]);hi=np.minimum(np.ceil(uv.max(0)+26).astype(int),[1920,1080]);crop=out/f'{sid}_{pt}_source_supported_bars_crop.png';cv2.imwrite(str(crop),proof[lo[1]:hi[1],lo[0]:hi[0]])
            proof_paths.append(dict(shot=sid,pts=pt,original=str(original),point_proof=str(full),crop_proof=None if crop is None else str(crop)))
        if graph_note.get('sha256')and sha(graph_note['path'])!=graph_note['sha256']:raise ValueError('Source navigation graph changed during capture')
        summaries[sid]=dict(requested_frames=len(requested),qualified_frames=sum(bool(f['rows'])for f in frames),rows=sum(f['rows']for f in frames),missing_native_PTS=[f['pts']for f in frames if not f['rows']],rows_by_native_path=dict(by_path),raw_nonpilot_exterior_candidates=raw_count,pending_reasons=dict(failures),frames=frames)
    write(qualified/'binding.json',dict(binding,fit_eligible=True,source_capture=str(capture),source_capture_binding_sha256=sha(capture/'binding.json'),qualification_method='Explicit approved bounded Ad5 instances; actual-source photo navigation; unchanged public component/profile/closed-native-shape reader; same RGB colour and full quantile SAM/HUD guards; no synthesized UV or weights',pilot_rows_preserved=pilot_preserved))
    build(root,qualified,out/'source_packets',shot_ids=['s07','s08','s09']);report=dict(binding,complete=True,shots=summaries,total_requested_frames=43,total_qualified_frames=sum(s['qualified_frames']for s in summaries.values()),total_rows=total,all_51_reviewed_pilot_rows_preserved_exactly=pilot_preserved==51,source_packet_root=str(out/'source_packets'),packets={sid:str(out/'source_packets'/sid)for sid in WINDOWS},packet_manifest=str(out/'source_packets/packets_manifest.json'),qualified_source_root=str(qualified),raw_evidence_root=str(capture),representative_source_proofs=proof_paths,neighbor_frames_processed=0,other_native_frames_generated=False,registered_panel_edges_file_modified=False,fit_or_upsert_executed=False,seconds=time.monotonic()-started);write(out/'report.json',report);print(json.dumps(dict(complete=True,rows=total,qualified_frames=report['total_qualified_frames'],packets=report['packets'],seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()