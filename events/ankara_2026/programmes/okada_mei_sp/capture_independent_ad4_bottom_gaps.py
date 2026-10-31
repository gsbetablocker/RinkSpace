"""Complete current Ad4 top-only source groups with actual RGB bottom reads.

Current native point input selects exposures only. Independently captured whole-
print registration navigates the unchanged public RGB exterior reader. Only its
actual bottom q50s enter this new packet; all other candidates stay in evidence.
"""
from pathlib import Path
import argparse,copy,hashlib,json,shutil,time
from collections import Counter
import cv2,numpy as np
from rinkspace.geometry import project
from rinkspace.session.registered_panel_edges import read_registered_panel_edges
from capture_independent_badge import exact_frames,sha
from capture_independent_prints import material_guard,source_patch_check
from capture_independent_materials import source_exclusion,spaced_indices,plain
from capture_independent_ad4_glyph_gap import safe,write
from build_independent_shot_packets import build

TOP='print_instance/Ad4::Ad4.top_edge/path0'
BOTTOM='print_instance/Ad4::Ad4.bottom_edge/path0'
PRIORITY_FIRST,PRIORITY_LAST,PILOT=1894400,1912320,1906176


def select_scope(prepared):
    manifest=json.loads((prepared/'joint_manifest.json').read_text('utf8'));mapping=json.loads((prepared/'mapping.json').read_text('utf8'))
    identity=sha(prepared/'point_observations.npz')
    if identity!=manifest['point_input_sha256']or identity!=mapping['point_input_sha256']:raise ValueError('Current prepared point identity differs')
    points=np.load(prepared/'point_observations.npz',allow_pickle=False);names=mapping['path_ids'];scopes={}
    for shot in manifest['shots']:
        if shot['id']not in ('s07','s08','s09'):continue
        lo,hi=shot['row_begin'],shot['row_stop'];pts=points['pts'][lo:hi];indices=points['path_indices'][lo:hi];groups={}
        for pt in np.unique(pts):groups[int(pt)]=[names[int(i)]for i in np.unique(indices[pts==pt])]
        requested=[pt for pt,paths in groups.items()if paths==[TOP]]
        zeros=[]
        if shot['id']=='s09':
            zeros=[pt for pt in range(PRIORITY_FIRST,PRIORITY_LAST+256,256)if not groups.get(pt)]
            requested+=zeros
        scopes[shot['id']]=dict(requested_native_PTS=sorted(set(requested)),top_only_exposures=len(requested)-len(zeros),priority_zero_native_PTS=zeros,original_source_groups={str(pt):groups.get(pt,[])for pt in requested})
    return scopes,dict(root=str(prepared),manifest_sha256=sha(prepared/'joint_manifest.json'),mapping_sha256=sha(prepared/'mapping.json'),point_input_sha256=identity,selection='Every current S07-S09 exposure whose only path is Ad4.top_edge, plus missing exposures in explicitly reviewed S09 148.0-149.4s Ad4 segment; no residual, native projection or camera state read')


def proof_image(out,sid,pt,image,rows):
    original=out/f'{sid}_{pt}_original.png';cv2.imwrite(str(original),image);overlay=image.copy()
    for row in rows:
        qq=np.asarray(row['actual_source_quantiles'])[0]
        cv2.line(overlay,tuple(np.rint(qq[0]).astype(int)),tuple(np.rint(qq[2]).astype(int)),(255,0,255),1)
        cv2.circle(overlay,tuple(np.rint(row['uv']).astype(int)),3,(0,0,255),1,cv2.LINE_AA)
    cv2.putText(overlay,f'Ad4 BOTTOM | {sid} PTS {pt} | {len(rows)} actual RGB q50 | NO FIT',(20,35),0,.65,(0,220,255),2)
    full=out/f'{sid}_{pt}_actual_bottom.jpg';cv2.imwrite(str(full),overlay);crop=None
    if rows:
        uv=np.asarray([r['uv']for r in rows]);lo=np.maximum(np.floor(uv.min(0)-25).astype(int),[0,0]);hi=np.minimum(np.ceil(uv.max(0)+26).astype(int),[1920,1080]);crop=out/f'{sid}_{pt}_actual_bottom_crop.png';cv2.imwrite(str(crop),overlay[lo[1]:hi[1],lo[0]:hi[0]])
    return dict(shot=sid,pts=pt,rows=len(rows),source_original=str(original),full_proof=str(full),crop_proof=None if crop is None else str(crop))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--prepared-root',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-scope',type=Path,help='Explicit source PTS scope; admits selected exterior sides, with optional known JUNIOR fallback')
    p.add_argument('--preserve-source-capture',type=Path,help='Qualified independent source capture whose existing PTS/path groups stay exact; only newly supported groups are added')
    p.add_argument('--only-missing-source-frames',action='store_true',help='Reuse supported prior source records verbatim; decode/read only remaining empty frames')
    p.add_argument('--reuse-completed-capture',type=Path,help='Reuse complete JSONL records from an interrupted current-source capture; preserve partial file and all its proofs')
    p.add_argument('--shots',nargs='+',default=['s07','s08']);p.add_argument('--pts',nargs='+',type=int);p.add_argument('--edges',nargs='+',choices=['top','right','bottom','left'],default=['right','left']);p.add_argument('--junior-glyphs',action='store_true');p.add_argument('--glyph-atlas',type=Path,help='Explicit same-original-reference semantic atlas; pending/fit_eligible=false paths are excluded');p.add_argument('--search-radius',type=float,default=20.,help='Source-navigation normal search radius; all material and qualification guards remain unchanged')
    a=p.parse_args()
    if a.source_scope is not None:return capture_explicit_source_scope(a)
    if a.prepared_root is None:p.error('--prepared-root is required without --source-scope')
    root=a.root.resolve();prepared=a.prepared_root.resolve();out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    started=time.monotonic();cv2.setNumThreads(2);scopes,current=select_scope(prepared);index=json.loads((root/'index.json').read_text('utf8'));video=Path(index['source']['resolved_path']);video_sha=sha(video)
    if video_sha!=index['source']['actual_sha256']:raise ValueError('Actual original video differs')
    bank_path=root/'print_references/references.json';meta=next(v for v in json.loads(bank_path.read_text('utf8'))['references']if v['ad_id']=='Ad4');bank=dict(np.load(meta['descriptor_npz'],allow_pickle=False));reference=root/meta['shot_id']/'originals'/f'{meta["shot_id"]}_pts{meta["pts"]}.png';bank['source_image']=cv2.imread(str(reference));bank['reference_mask']=cv2.imread(meta['descriptor_mask'],0)>0
    if hashlib.sha256(bank['source_image'].tobytes()).hexdigest()!=meta['source_BGR_sha256']or sha(meta['descriptor_npz'])!=meta['descriptor_npz_sha256']:raise ValueError('Independent Ad4 reference differs')
    out.mkdir(parents=True);capture=out/'capture';qualified=out/'qualified';capture.mkdir();qualified.mkdir();write(out/'current_source_scope.json',dict(current=current,shots=scopes))
    approval=dict(authority='Current primary-session explicit original-image Ad4 identification and instruction to complete single-top-only intervals',physical_entity='Ad4',source_reference=str(reference),source_reference_BGR_sha256=meta['source_BGR_sha256'],physical_identity_basis='Known main-camera and end-camera installation plus previously established Ad4 physical-source reference and neighboring print context; complete current source pattern is freshly verified per exposure',scope=scopes,admitted_native_path=BOTTOM,other_edges_and_glyphs_not_admitted=True,source_reference_correspondence_is_navigation_only=True)
    approval_path=out/'source_identity_approval.json';write(approval_path,approval);approval_sha=sha(approval_path)
    files=[Path(__file__),Path(read_registered_panel_edges.__code__.co_filename),Path(material_guard.__code__.co_filename),Path(source_exclusion.__code__.co_filename),Path(build.__code__.co_filename)]
    binding=dict(source_sha256=video_sha,source_video_sha256=video_sha,source_video=str(video),independent_source_capture=True,legacy_new_shot_inputs_read=False,native_state_loaded=False,native_projection_called=False,source_scope=current,source_identity_approval=str(approval_path),source_identity_approval_sha256=approval_sha,query_spacing_px=24.,fit_spacing_px=48.,short_feature_minimum_samples=4,public_qualification_thresholds_changed=False,admitted_native_path=BOTTOM,other_entities_unchanged=True,fit_eligible=False,fit_executed=False,implementation_sha256={str(f.resolve()):sha(f)for f in files})
    write(capture/'binding.json',binding);summaries={};proofs=[];all_records={};total=0;pilot_preserved=False
    for sid in ('s09','s08','s07'):
        requested=scopes[sid]['requested_native_PTS'];folder=capture/sid;folder.mkdir();profile_dir=folder/'profiles';profile_dir.mkdir();qfolder=qualified/sid;qfolder.mkdir();regfile=root/'registered_full'/sid/'registration.jsonl';regsha=sha(regfile);cached={}
        with regfile.open(encoding='utf8')as stream:
            for line_number,line in enumerate(stream,1):
                rr=json.loads(line)
                if rr['pts']not in requested:continue
                ad=next((v for v in rr.get('prints',[])if v.get('ad')=='Ad4'and v.get('status')=='source_content_registered'and v.get('source_pattern_check',{}).get('supported')is True),None)
                if ad is not None:cached[rr['pts']]=dict(ad,source_registration_line=line_number)
        images=exact_frames(video,requested)if requested else{};frames=[];reasons=Counter();frame_records={};proof_pts={requested[0],requested[-1]}if requested else set()
        if sid=='s09':proof_pts|={PILOT,1900032}
        with(folder/'source_rows.jsonl').open('x',encoding='utf8',buffering=1)as raw_out,(folder/'registration.jsonl').open('x',encoding='utf8',buffering=1)as audit,(qfolder/'source_rows.jsonl').open('x',encoding='utf8',buffering=1)as selected_out:
            for pt in requested:
                image=images[pt];rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);ex,maskpath=source_exclusion(root,sid,pt);item=cached.get(pt);check=None;record=None;rows=[];pending=[];profile=None;nav=None
                if item is not None:
                    h=np.asarray(item['H_current_from_reference']);check=source_patch_check(image,ex,meta,bank,h)
                    if check['supported']:
                        guide=project(h,bank['projected_uv']);offsets=bank['edge_offsets'];paths=[guide[offsets[i]:offsets[i+1]]for i in range(4)];arrays={}
                        def collect(stage,values):arrays.update({stage+'__'+k:v for k,v in values.items()})
                        local,record=read_registered_panel_edges(rgb,paths,ex,binding=dict(source_sha256=video_sha,source_pts=pt,time_base='1/12800',shot_id=sid,independent_source_pattern_match_verified=True),search_radius=20.,spacing=24.,candidate_material_validator=lambda row:material_guard(rgb,row,'Ad4')[0],proof_sink=collect)
                        dest=profile_dir/f'{pt}.npz';np.savez_compressed(dest,**arrays);profile=dict(path=str(dest),sha256=sha(dest),profile_arrays_prefix='actual_source_envelope_search__',topology_arrays_prefix='source_topology__');bottom=[v for v in local if v['curve']==2]
                        for k in spaced_indices([v['uv']for v in bottom],spacing=48.):
                            row=bottom[k];trace=next(c for c in record['candidate_records']if c['curve']==2 and c['uv']==row['uv']and c['disposition']=='source_line_organized_actual_measurement');np.testing.assert_array_equal(row['uv'],np.asarray(row['actual_source_quantiles'])[0,1]);assert trace['candidate_flags']==0
                            rows.append(dict(row,path_id=BOTTOM,entity='Ad4',source_candidate_id=trace['candidate_id'],source_profile_query=trace['stage_query'],source_topology_probes=trace['source_topology_probes'],actual_source_sides=trace['actual_source_sides'],source_colour=material_guard(rgb,row,'Ad4')[1],source_reference_pts=meta['pts'],source_profile_proof=profile,source_identity_approval=str(approval_path),source_identity_approval_sha256=approval_sha,capture_family='independent_source_bottom_completion',native_or_H_vertices_used_as_observations=False,fit_eligible=True,review_state='source_supported_public_RGB_exterior_bottom'))
                        pending=record['candidate_records'];reasons.update(reason for v in pending for reason in v['pending_reasons'])
                        if sid=='s09'and pt==PILOT:
                            pilot=json.loads((root/'ad4_bottom_pilot_1906176/s09/source_rows.jsonl').read_text('utf8'));old=[v for v in pilot['rows']if v['path_id']==BOTTOM]
                            assert len(old)==len(rows)==22
                            for before,after in zip(old,rows,strict=True):
                                for key in ['uv','normal','actual_source_quantiles','actual_source_sides','source_topology_probes']:assert json.dumps(before[key],sort_keys=True)==json.dumps(after[key],sort_keys=True)
                            pilot_preserved=True
                    else:reasons['fresh_original_source_pattern_unsupported']+=1
                    nav=dict(source_registration_path=str(regfile),source_registration_sha256=regsha,source_registration_line=item['source_registration_line'],reference_pts=item['reference_pts'],source_graph_parent_pts=item.get('source_graph_parent_pts'),H_current_from_reference=item['H_current_from_reference'],cached_source_pattern_check=item['source_pattern_check'],fresh_source_pattern_check=check,navigation_not_observations=True)
                else:reasons['no_supported_current_independent_Ad4_registration']+=1
                result=dict(shot_id=sid,pts=pt,source_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),source_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),source_video_sha256=video_sha,mask_path=str(maskpath),mask_sha256=sha(maskpath),rows=rows,pending_rows=pending,fit_eligible=bool(rows),review_state='source_supported_public_RGB_exterior_bottom'if rows else'pending_no_supported_actual_bottom',source_registration=nav,source_profile_proof=profile,source_identity_approval=str(approval_path),source_identity_approval_sha256=approval_sha)
                raw_out.write(json.dumps(safe(dict(result,rows=pending,fit_eligible=False)),ensure_ascii=False,allow_nan=False)+'\n');selected_out.write(json.dumps(safe(result),ensure_ascii=False,allow_nan=False)+'\n');audit.write(json.dumps(safe(dict(pts=pt,source_registration=nav,public_source_readout=record,source_profile_proof=profile)),ensure_ascii=False,allow_nan=False)+'\n');frame_records[pt]=result;total+=len(rows);frames.append(dict(pts=pt,source_seconds=pt/12800,rows=len(rows),whole_print_highpass_NCC=None if check is None else check.get('highpass_pattern_correlation'),source_registration_supported=item is not None));print(json.dumps(dict(shot=sid,pts=pt,rows=len(rows))),flush=True)
                if pt in proof_pts:proofs.append(proof_image(out,sid,pt,image,rows))
        shutil.copyfile(root/sid/'navigation/poses_reanchored.npz',qfolder/'poses.npz');assert sha(regfile)==regsha
        all_records[sid]=frame_records;summaries[sid]=dict(requested_frames=len(requested),top_only_frames=scopes[sid]['top_only_exposures'],priority_zero_frames=len(scopes[sid]['priority_zero_native_PTS']),qualified_frames=sum(bool(f['rows'])for f in frames),rows=sum(f['rows']for f in frames),missing_native_PTS=[f['pts']for f in frames if not f['rows']],pending_reasons=dict(reasons),frames=frames)
        del images
    write(qualified/'binding.json',dict(binding,fit_eligible=True,qualification_method='Established physical Ad4 source identity; freshly verified whole visible source pattern; unchanged public actual RGB exterior topology/material/SAM guards and source-line qualification; q50 only; bottom-only 48px thinning',source_capture=str(capture),source_capture_binding_sha256=sha(capture/'binding.json'),source_navigation_pose_packet_initialization_only=True,current_fitted_motion_must_be_preserved_by_source_update=True))
    selected=[sid for sid in ('s07','s08','s09')if summaries[sid]['rows']]
    build(root,qualified,out/'source_packets',shot_ids=selected)
    packet_hashes={}
    for sid in selected:
        arrays=dict(np.load(out/'source_packets'/sid/'point_observations.npz',allow_pickle=False));expected=[(pt,row)for pt in scopes[sid]['requested_native_PTS']for row in all_records[sid][pt]['rows']];assert np.array_equal(arrays['uv'],[row['uv']for pt,row in expected]);assert np.array_equal(arrays['pts'],[pt for pt,row in expected]);assert len(set(arrays['observation_ids']))==len(expected);assert set(json.loads((out/'source_packets'/sid/'mapping.json').read_text('utf8'))['path_ids'])=={BOTTOM};packet_hashes[sid]=sha(out/'source_packets'/sid/'point_observations.npz')
    assert sha(prepared/'point_observations.npz')==current['point_input_sha256']and sha(prepared/'joint_manifest.json')==current['manifest_sha256'];report=dict(complete=True,current_source_input=current,shots=summaries,total_rows=total,packets={sid:str(out/'source_packets'/sid)for sid in selected},packet_sha256=packet_hashes,representative_original_RGB_proofs=proofs,pilot_22_bottom_rows_preserved_exactly=pilot_preserved,admitted_native_path=BOTTOM,source_only_pixel_and_time_preservation=True,public_thresholds_changed=False,glyphs_or_new_holes_admitted=False,model_camera_or_point_input_modified=False,fit_or_upsert_executed=False,seconds=time.monotonic()-started)
    write(out/'report.json',report);print(json.dumps(dict(complete=True,rows=total,shots={sid:dict(requested=v['requested_frames'],qualified=v['qualified_frames'],rows=v['rows'])for sid,v in summaries.items()},packets=report['packets'],seconds=report['seconds'])),flush=True)


def capture_explicit_source_scope(a):
    """Capture real exterior sides and known source-glyph fallback at explicit PTS.

    The scope is a source-coverage selector. Existing independent whole-print H
    charts navigate public readers; fitted state and residuals are never loaded.
    A glyph atlas is used only after its whole-word and separated-current-ink
    checks. Failed frames remain recorded, including all actual edge candidates.
    """
    import rinkspace.session.print_patterns as pp
    from capture_independent_ad4_glyph_gap import read_public,save_profiles
    root=a.root.resolve();out=a.output.resolve();scope_path=a.source_scope.resolve(strict=True)
    if out.exists():raise FileExistsError(out)
    if any(sid not in ('s07','s08','s09')for sid in a.shots):raise ValueError('Unknown independent source shot')
    started=time.monotonic();cv2.setNumThreads(2);scope=json.loads(scope_path.read_text('utf8'));index=json.loads((root/'index.json').read_text('utf8'));video=Path(index['source']['resolved_path']);video_sha=sha(video)
    if video_sha!=index['source']['actual_sha256']:raise ValueError('Actual original video differs')
    requested={sid:sorted(set(scope[sid]['pts']) & set(a.pts))if a.pts else sorted(set(scope[sid]['pts']))for sid in a.shots};prior_records={};prior_evidence={}
    if a.preserve_source_capture:
        prior_root=a.preserve_source_capture.resolve(strict=True)
        for sid in a.shots:
            path=prior_root/sid/'source_rows.jsonl';prior_records[sid]={r['pts']:r for r in map(json.loads,path.read_text('utf8').splitlines())};prior_evidence[sid]=dict(path=str(path),sha256=sha(path),preservation='Every previous qualified PTS/path group remains byte-equivalent in its original measurement rows; new source groups only')
    reused_completed={sid:set()for sid in a.shots}
    if a.reuse_completed_capture:
        reuse_root=a.reuse_completed_capture.resolve(strict=True)
        for sid in a.shots:
            path=reuse_root/sid/'source_rows.jsonl'
            if not path.exists():continue
            reused=[]
            with path.open(encoding='utf8')as stream:
                for number,line in enumerate(stream,1):
                    try:record=json.loads(line)
                    except json.JSONDecodeError:
                        if not line.endswith('\n'):break
                        raise
                    pt=record['pts'];previous=prior_records.get(sid,{}).get(pt)
                    if previous is not None:
                        assert previous['source_BGR_sha256']==record['source_BGR_sha256']
                        for name in {r['path_id']for r in previous['rows']}:
                            assert [r for r in previous['rows']if r['path_id']==name]==[r for r in record['rows']if r['path_id']==name]
                    prior_records.setdefault(sid,{})[pt]=record;reused_completed[sid].add(pt);reused.append(dict(pts=pt,line=number))
            prior_evidence.setdefault(sid,{})['reused_completed_capture']=dict(path=str(path),sha256=sha(path),complete_records=reused)
    if a.only_missing_source_frames and not a.preserve_source_capture:raise ValueError('Missing-frame-only mode needs the previous qualified source capture')
    if not any(requested.values()):raise ValueError('Empty explicit source scope')
    bank_path=root/'print_references/references.json';meta=next(v for v in json.loads(bank_path.read_text('utf8'))['references']if v['ad_id']=='Ad4');bank=dict(np.load(meta['descriptor_npz'],allow_pickle=False));reference=root/meta['shot_id']/'originals'/f'{meta["shot_id"]}_pts{meta["pts"]}.png';bank['source_image']=cv2.imread(str(reference));bank['reference_mask']=cv2.imread(meta['descriptor_mask'],0)>0
    if hashlib.sha256(bank['source_image'].tobytes()).hexdigest()!=meta['source_BGR_sha256']or sha(meta['descriptor_npz'])!=meta['descriptor_npz_sha256']:raise ValueError('Independent Ad4 reference differs')
    atlas_path=a.glyph_atlas.resolve(strict=True)if a.glyph_atlas else root/'ad4_glyph_pilot_2179328/reference_atlas.json';pattern=None;native_map={};allowed=set()
    if a.junior_glyphs or a.glyph_atlas:
        atlas=json.loads(atlas_path.read_text('utf8'));atlas_image=cv2.imread(atlas['source_image'])
        if hashlib.sha256(atlas_image.tobytes()).hexdigest()!=atlas['source_BGR_sha256']:raise ValueError('Independent glyph source RGB differs')
        if atlas['source_BGR_sha256']!=meta['source_BGR_sha256']:raise ValueError('Whole-print and glyph charts need the same source reference')
        if a.glyph_atlas:
            atlas=dict(atlas,paths=[v for v in atlas['paths']if v.get('fit_eligible')is not False and not v.get('admission_pending',False)and not v['path_id'].endswith('_inner')])
            known={v['id']for v in json.loads((root/'native_world.json').read_text('utf8'))['cuts']}
            if not atlas['paths']or any(v['native_path_id']not in known or not v['native_path_id'].startswith('print_instance/Ad4::')for v in atlas['paths']):raise ValueError('Explicit atlas has no eligible known Ad4 native exterior paths')
        pattern=pp.PrintPattern(atlas,atlas_image);native_map={'print_instance/Ad4::'+v['path_id']:v['native_path_id']for v in atlas['paths']};allowed=set(native_map.values())if a.glyph_atlas else{v for k,v in native_map.items()if k.rsplit('_',1)[-1]in set('JUNIO')}
        if not a.glyph_atlas and len(allowed)!=5:raise ValueError('Only reviewed J/U/N/I/O exterior paths allowed')
    side_numbers={'top':0,'right':1,'bottom':2,'left':3};side_paths=['print_instance/Ad4::Ad4.'+v+'/path0'for v in('top_edge','right_vertical','bottom_edge','left_vertical')];selected_curves={side_numbers[v]for v in a.edges}
    out.mkdir(parents=True);capture=out/'capture';qualified=out/'qualified';capture.mkdir();qualified.mkdir();write(out/'current_source_scope.json',dict(source_scope=str(scope_path),source_scope_sha256=sha(scope_path),requested_native_PTS=requested,selection='explicit current source-coverage PTS; no native state or residual read'))
    approval=dict(authority='Current primary-session explicit source-completion task for known Ad4 instance in known main/end camera source context',physical_entity='Ad4',source_reference=str(reference),source_reference_BGR_sha256=meta['source_BGR_sha256'],requested_native_PTS=requested,allowed_edges=a.edges,allowed_glyphs=sorted(allowed),source_H_and_contours_navigation_only=True)
    approval_path=out/'source_identity_approval.json';write(approval_path,approval);approval_sha=sha(approval_path)
    reader_path=Path(read_registered_panel_edges.__code__.co_filename)
    files=[Path(__file__),reader_path,reader_path.parent/'panel_boundaries.py',reader_path.parent.parent/'vision_source_roles.py',Path(pp.__file__),Path(source_patch_check.__code__.co_filename),Path(source_exclusion.__code__.co_filename),Path(build.__code__.co_filename)]
    binding=dict(source_sha256=video_sha,source_video_sha256=video_sha,source_video=str(video),independent_source_capture=True,legacy_new_shot_inputs_read=False,native_state_loaded=False,native_projection_called=False,source_scope=str(scope_path),source_scope_sha256=sha(scope_path),source_identity_approval=str(approval_path),source_identity_approval_sha256=approval_sha,source_glyph_atlas=str(atlas_path)if pattern is not None else None,source_glyph_atlas_sha256=sha(atlas_path)if pattern is not None else None,query_spacing_px=24.,glyph_query_spacing_px=8.,source_navigation_search_radius_px=a.search_radius,fit_spacing_px=48.,short_feature_minimum_samples=4,public_qualification_thresholds_changed=False,admitted_native_paths=sorted({side_paths[i]for i in selected_curves}|allowed),fit_eligible=False,fit_executed=False,implementation_sha256={str(f.resolve()):sha(f)for f in files})
    binding['preserved_source_capture']=prior_evidence;write(capture/'binding.json',binding);summaries={};all_records={};proofs=[];total=0
    for sid in a.shots:
        pts=requested[sid];folder=capture/sid;folder.mkdir();profiles=folder/'profiles';profiles.mkdir();qfolder=qualified/sid;qfolder.mkdir();regfile=root/'registered_full'/sid/'registration.jsonl';regsha=sha(regfile);cached={}
        for line_number,line in enumerate(regfile.open(encoding='utf8'),1):
            rr=json.loads(line)
            if rr['pts']not in pts:continue
            item=next((v for v in rr.get('prints',[])if v.get('ad')=='Ad4'and v.get('status')=='source_content_registered'and v.get('source_pattern_check',{}).get('supported')is True),None)
            if item is not None:cached[rr['pts']]=dict(item,source_registration_line=line_number)
        read_pts=[pt for pt in pts if pt not in reused_completed[sid]and not(a.only_missing_source_frames and prior_records.get(sid,{}).get(pt,{}).get('rows'))]
        images=exact_frames(video,read_pts)if read_pts else{};records={};frame_summaries=[];reasons=Counter();proof_pts={read_pts[0],read_pts[len(read_pts)//2],read_pts[-1]}if read_pts else set()
        with(folder/'source_rows.jsonl').open('x',encoding='utf8',buffering=1)as raw,(folder/'registration.jsonl').open('x',encoding='utf8',buffering=1)as audit,(qfolder/'source_rows.jsonl').open('x',encoding='utf8',buffering=1)as selected:
            for pt in pts:
                if pt not in images:
                    record=copy.deepcopy(prior_records[sid][pt]);record['reused_prior_source_capture']=prior_evidence[sid];record['source_read_executed_this_capture']=False
                    selected.write(json.dumps(safe(record),ensure_ascii=False,allow_nan=False)+'\n');raw.write(json.dumps(safe(dict(pts=pt,rows=[],fit_eligible=False,reused_prior_source_capture=prior_evidence[sid],source_read_executed_this_capture=False)),ensure_ascii=False,allow_nan=False)+'\n');audit.write(json.dumps(safe(dict(pts=pt,reused_prior_source_capture=prior_evidence[sid],source_read_executed_this_capture=False)),ensure_ascii=False,allow_nan=False)+'\n');records[pt]=record;total+=len(record['rows']);frame_summaries.append(dict(pts=pt,seconds=pt/12800,rows=len(record['rows']),paths=dict(Counter(v['path_id']for v in record['rows'])),source_read_executed_this_capture=False));continue
                image=images[pt];rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);ex,mp=source_exclusion(root,sid,pt);item=cached.get(pt);rows=[];pending=[];check=None;edge_record=None;glyph_registration=None;glyph_readout=None;glyph_actual=[];profile=None;glyph_profile=None
                if item is not None:
                    h=np.asarray(item['H_current_from_reference']);check=source_patch_check(image,ex,meta,bank,h)
                    if check['supported']:
                        guide=project(h,bank['projected_uv']);offsets=bank['edge_offsets'];paths=[guide[offsets[i]:offsets[i+1]]for i in range(4)];arrays={}
                        def collect(stage,values):arrays.update({stage+'__'+k:v for k,v in values.items()})
                        local,edge_record=read_registered_panel_edges(rgb,paths,ex,binding=dict(source_sha256=video_sha,source_pts=pt,time_base='1/12800',shot_id=sid,independent_source_pattern_match_verified=True),search_radius=a.search_radius,spacing=24.,candidate_material_validator=lambda row:material_guard(rgb,row,'Ad4')[0],proof_sink=collect)
                        dest=profiles/f'{pt}_edges.npz';np.savez_compressed(dest,**arrays);profile=dict(path=str(dest),sha256=sha(dest),profile_arrays_prefix='actual_source_envelope_search__',topology_arrays_prefix='source_topology__')
                        for ci in sorted(selected_curves):
                            group=[v for v in local if v['curve']==ci]
                            for k in spaced_indices([v['uv']for v in group],spacing=48.):
                                row=group[k];trace=next(v for v in edge_record['candidate_records']if v['curve']==ci and v['uv']==row['uv']and v['disposition']=='source_line_organized_actual_measurement');np.testing.assert_array_equal(row['uv'],np.asarray(row['actual_source_quantiles'])[0,1]);assert trace['candidate_flags']==0
                                rows.append(dict(row,path_id=side_paths[ci],entity='Ad4',source_candidate_id=trace['candidate_id'],source_profile_query=trace['stage_query'],source_topology_probes=trace['source_topology_probes'],actual_source_sides=trace['actual_source_sides'],source_colour=material_guard(rgb,row,'Ad4')[1],source_reference_pts=meta['pts'],source_profile_proof=profile,capture_family='independent_source_exterior_completion',native_or_H_vertices_used_as_observations=False,fit_eligible=True,review_state='source_supported_public_RGB_exterior'))
                        pending=edge_record['candidate_records'];reasons.update(r for v in pending for r in v['pending_reasons'])
                        if pattern is not None and not rows:
                            glyph_registration=pattern.register(image,ex.astype('uint8'),h)
                            if glyph_registration['status']=='source_pattern_registered':
                                glyph_actual,glyph_readout,captured=read_public(pattern,image,ex,glyph_registration,dict(source_sha256=video_sha,source_pts=pt,time_base='1/12800',shot_id=sid,decoded_native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
                                dest=profiles/f'{pt}_glyphs.npz';save_profiles(dest,captured);glyph_profile=dict(path=str(dest),sha256=sha(dest));
                                for row in glyph_actual:
                                    reading=captured['public_readings'][row['candidate']];np.testing.assert_array_equal(row['uv'],reading['actual_source_uv']);row['atlas_path_id']=row['path_id'];row['path_id']=native_map[row['path_id']];row.update(measurement_operator='actual_RGB_white_ink_interface_q10_q50_q90',source_quantile_semantics='q10/q50/q90',model_UV_synthesized=False,native_or_H_vertices_used_as_observations=False,source_profile_proof=glyph_profile)
                                for path in sorted(allowed):
                                    group=[v for v in glyph_actual if v['path_id']==path];rows.extend(dict(group[k],fit_eligible=True,capture_family='independent_source_known_glyph_completion',review_state='source_supported_public_RGB_separated_glyph')for k in spaced_indices([v['uv']for v in group],spacing=48.))
                                reasons.update(glyph_readout.get('reasons',{}))
                            else:reasons['JUNIOR_'+glyph_registration['status']]+=1
                    else:reasons['fresh_whole_print_source_pattern_unsupported']+=1
                else:reasons['no_supported_current_independent_Ad4_registration']+=1
                for row in rows:row.update(source_identity_approval=str(approval_path),source_identity_approval_sha256=approval_sha)
                preservation=None
                prior=prior_records.get(sid,{}).get(pt)
                if prior is not None and prior.get('fit_eligible')is True:
                    if prior['source_BGR_sha256']!=hashlib.sha256(image.tobytes()).hexdigest():raise ValueError('Preserved original-source RGB differs')
                    old_paths={r['path_id']for r in prior['rows']};new_rows=[r for r in rows if r['path_id']not in old_paths]
                    rows=copy.deepcopy(prior['rows'])+new_rows;preservation=dict(prior_evidence[sid],preserved_rows=len(prior['rows']),preserved_native_paths=sorted(old_paths),new_group_rows=len(new_rows))
                record=dict(shot_id=sid,pts=pt,source_video_sha256=video_sha,source_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),source_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),mask_path=str(mp),mask_sha256=sha(mp),rows=rows,pending_rows=pending,fit_eligible=bool(rows),review_state='source_supported_RGB'if rows else'pending_no_supported_transverse_material',source_profile_proof=profile,glyph_source_profiles=glyph_profile,source_group_preservation=preservation,source_identity_approval=str(approval_path),source_identity_approval_sha256=approval_sha)
                selected.write(json.dumps(safe(record),ensure_ascii=False,allow_nan=False)+'\n');raw.write(json.dumps(safe(dict(record,rows=pending,all_actual_glyph_rows=glyph_actual,fit_eligible=False)),ensure_ascii=False,allow_nan=False)+'\n');audit.write(json.dumps(safe(dict(pts=pt,source_registration=item,source_registration_path=str(regfile),source_registration_sha256=regsha,fresh_source_pattern_check=check,public_edge_readout=edge_record,word_registration=glyph_registration,word_readout=glyph_readout)),ensure_ascii=False,allow_nan=False)+'\n');records[pt]=record;total+=len(rows)
                frame_summaries.append(dict(pts=pt,seconds=pt/12800,rows=len(rows),paths=dict(Counter(v['path_id']for v in rows)),source_read_executed_this_capture=True,source_registration_supported=item is not None,word_status=None if glyph_registration is None else glyph_registration['status']));print(json.dumps(dict(shot=sid,pts=pt,rows=len(rows),paths=frame_summaries[-1]['paths'])),flush=True)
                if pt in proof_pts:
                    source_path=out/f'{sid}_{pt}_original.png';cv2.imwrite(str(source_path),image);proof=image.copy()
                    for row in rows:cv2.circle(proof,tuple(np.rint(row['uv']).astype(int)),3,(0,255,0),1,cv2.LINE_AA)
                    path=out/f'{sid}_{pt}_actualproof.jpg';cv2.imwrite(str(path),proof);proofs.append(dict(shot=sid,pts=pt,source_original=str(source_path),actualproof=str(path),rows=len(rows)))
        shutil.copyfile(root/sid/'navigation/poses_reanchored.npz',qfolder/'poses.npz');assert sha(regfile)==regsha
        all_records[sid]=records;summaries[sid]=dict(requested_frames=len(pts),freshly_read_frames=len(read_pts),reused_source_frames=len(pts)-len(read_pts),qualified_frames=sum(bool(v['rows'])for v in records.values()),rows=sum(len(v['rows'])for v in records.values()),missing_native_PTS=[pt for pt,v in records.items()if not v['rows']],rows_by_path=dict(Counter(v['path_id']for rec in records.values()for v in rec['rows'])),pending_reasons=dict(reasons),frames=frame_summaries)
        del images
    write(qualified/'binding.json',dict(binding,fit_eligible=True,source_capture=str(capture),source_capture_binding_sha256=sha(capture/'binding.json'),qualification_method='Fresh whole-print original source support; unchanged public actual RGB exterior topology/material/SAM guards, or whole JUNIOR registration and separated current letters; actual q50 only',source_navigation_pose_packet_initialization_only=True,current_fitted_motion_must_be_preserved_by_source_update=True))
    selected_shots=[sid for sid in a.shots if summaries[sid]['rows']];packet_hashes={}
    if selected_shots:
        build(root,qualified,out/'source_packets',shot_ids=selected_shots)
        for sid in selected_shots:
            packet=out/'source_packets'/sid;arrays=dict(np.load(packet/'point_observations.npz',allow_pickle=False));expected=[(pt,row)for pt in requested[sid]for row in all_records[sid][pt]['rows']];np.testing.assert_array_equal(arrays['pts'],[pt for pt,row in expected]);np.testing.assert_array_equal(arrays['uv'],[row['uv']for pt,row in expected]);assert len(set(arrays['observation_ids']))==len(expected);packet_hashes[sid]=sha(packet/'point_observations.npz')
    report=dict(complete=True,shots=summaries,total_rows=total,packets={sid:str(out/'source_packets'/sid)for sid in selected_shots},packet_sha256=packet_hashes,representative_source_proofs=proofs,all_packet_UV_PTS_exact=True,public_thresholds_changed=False,native_projection_called=False,current_native_state_read=False,source_scope_sha256=sha(scope_path),source_only_pixel_and_time_preservation=True,fit_or_upsert_executed=False,seconds=time.monotonic()-started)
    write(out/'report.json',report);print(json.dumps(dict(complete=True,rows=total,shots={sid:{k:v[k]for k in('requested_frames','qualified_frames','rows')}for sid,v in summaries.items()},packets=report['packets'],seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
