"""Source-supported Ad4 JUNIOR glyphs for the explicitly reviewed 15-frame gap.

Only actual original-RGB transition q50s enter the packet. The approved source
atlas and image registrations navigate queries; no native projection is called.
The reviewed central frame keeps its exact original 31 measurement rows.
"""
from pathlib import Path
import argparse,copy,hashlib,json,shutil,sys,time
from collections import Counter
import cv2,numpy as np
import rinkspace.session.print_patterns as pp
from rinkspace.camera_reuse_localization import descriptors,match_features,ReuseConfig
from capture_independent_badge import exact_frames,sha
from capture_independent_materials import source_exclusion,spaced_indices,plain
from capture_independent_prints import source_patch_check
from build_independent_shot_packets import build

FIRST,LAST,CENTER=2177280,2180864,2179328


def safe(value):
    value=plain(value)
    if isinstance(value,float)and not np.isfinite(value):return None
    if isinstance(value,dict):return{k:safe(v)for k,v in value.items()}
    if isinstance(value,list):return[safe(v)for v in value]
    return value


def write(path,value):Path(path).write_text(json.dumps(safe(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def read_public(pattern,image,mask,registration,binding):
    captured={};original=pp.read_source_material_transition_profiles
    def telemetry(rgb,**kwargs):
        captured['query_path_labels']=list(sys._getframe(1).f_locals.get('labels',[]));kwargs['keep_profiles']=True
        result=original(rgb,**kwargs);captured.update(public_readings=result['readings'],search_uv=np.array(kwargs['search_points_uv']),normals=np.array(kwargs['normals']),intervals=np.array(kwargs['search_intervals_px']));return result
    pp.read_source_material_transition_profiles=telemetry
    try:rows,report=pattern.read_paths(image,mask.astype('uint8'),registration,binding,spacing=8.)
    finally:pp.read_source_material_transition_profiles=original
    return rows,report,captured


def save_profiles(path,captured):
    readings=captured.get('public_readings',[]);arrays={k:captured[k]for k in('search_uv','normals','intervals')if k in captured}
    arrays['query_path_labels']=np.array(captured.get('query_path_labels',[]),dtype='U96')
    if readings:
        count=len(readings);length=max(len(r['actual_source_normal_offsets_px'])for r in readings);arrays['profile_lengths']=np.array([len(r['actual_source_normal_offsets_px'])for r in readings])
        for key in('actual_source_normal_offsets_px','actual_source_luminance_profile','actual_source_smoothed_luminance_profile','actual_source_signed_luminance_gradient','actual_source_profile_valid'):
            values=np.full((count,length),np.nan)
            for i,r in enumerate(readings):values[i,:len(r[key])]=r[key]
            arrays[key]=values
        arrays['source_q10_q50_q90_normal_offsets_px']=np.array([r['source_q10_q50_q90_normal_offsets_px']for r in readings]);arrays['actual_source_transition_candidate_uv']=np.array([r['actual_source_transition_candidate_uv']for r in readings]);arrays['actual_source_uv']=np.full((count,2),np.nan)
        for i,r in enumerate(readings):
            if np.asarray(r['actual_source_uv']).shape==(2,):arrays['actual_source_uv'][i]=r['actual_source_uv']
        names=sorted({name for r in readings for name in r['source_flags']});arrays['source_flag_names']=np.array(names,dtype='U64');arrays['source_flags']=np.array([[r['source_flags'].get(name,False)for name in names]for r in readings])
    np.savez_compressed(path,**arrays)


def current_top_only_scope(prepared_root, probe_pts):
    """Select current source groups, never a model residual or fitted trajectory."""
    prepared=Path(prepared_root).resolve(strict=True)
    manifest_path=prepared/'joint_manifest.json';point_path=prepared/'point_observations.npz';mapping_path=prepared/'mapping.json'
    manifest=json.loads(manifest_path.read_text('utf8'));mapping=json.loads(mapping_path.read_text('utf8'))
    if sha(point_path)!=manifest['point_input_sha256']:raise ValueError('Current prepared point hash differs')
    shot=next(s for s in manifest['shots']if s['id']=='s09');points=np.load(point_path,allow_pickle=False)
    lo,hi=shot['row_begin'],shot['row_stop'];pts=points['pts'][lo:hi];indices=points['path_indices'][lo:hi]
    names=mapping['path_ids'];groups={}
    for pt in np.unique(pts):
        current=indices[pts==pt];groups[int(pt)]={names[int(i)]:int(np.sum(current==i))for i in np.unique(current)}
    top='print_instance/Ad4::Ad4.top_edge/path0';bottom='print_instance/Ad4::Ad4.bottom_edge/path0'
    if set(groups.get(probe_pts,{}))!={top}:raise ValueError('The diagnostic exposure must currently be Ad4 top-only')
    first=probe_pts
    while first-256>=shot['first_pts']and set(groups.get(first-256,{}))<={top}:first-=256
    last=probe_pts
    while last+256<CENTER and set(groups.get(last+256,{}))<={top,bottom}:last+=256
    requested=[p for p in range(first,last+256,256)if set(groups.get(p,{}))<={top}]
    if not requested:raise ValueError('No source-only top/empty groups selected')
    targets=list(range(first,CENTER+256,256));neighbors=[p for p in targets if p not in requested]
    audit=dict(selection='Contiguous current Ad4 top-only run containing probe, continued to nearest existing approved JUNIOR glyph group; existing two-edge groups are navigation only',probe_pts=probe_pts,prepared_root=str(prepared),prepared_manifest_sha256=sha(manifest_path),prepared_mapping_sha256=sha(mapping_path),prepared_point_input_sha256=sha(point_path),first_pts=first,last_pts=last,requested_native_PTS=requested,source_navigation_only_PTS=neighbors,counts=dict(requested=len(requested),top_only=sum(set(groups.get(p,{}))=={top}for p in requested),empty=sum(not groups.get(p,{})for p in requested)),current_groups=[dict(pts=p,path_counts=groups.get(p,{}),requested=p in requested)for p in range(first,last+256,256)],no_native_state_or_residual_used=True)
    return requested,neighbors,sorted(targets,reverse=True),audit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--prepared-root',type=Path,help='Select current S09 Ad4 top-only source groups around --probe-pts, instead of the original 15-frame gap');p.add_argument('--probe-pts',type=int,default=2170880);a=p.parse_args();root=a.root.resolve();out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    cv2.setNumThreads(2);started=time.monotonic();index=json.loads((root/'index.json').read_text('utf8'));video=Path(index['source']['resolved_path'])
    if sha(video)!=index['source']['actual_sha256']:raise ValueError('Actual original video differs')
    pilot=root/'ad4_glyph_pilot_2179328';atlas_path=pilot/'reference_atlas.json';atlas=json.loads(atlas_path.read_text('utf8'));pilot_record=json.loads((pilot/'source_rows.jsonl').read_text('utf8'));center_registration=json.loads((pilot/'word_registration.json').read_text('utf8'))
    reference_path=Path(atlas['source_image']);reference=cv2.imread(str(reference_path))
    if hashlib.sha256(reference.tobytes()).hexdigest()!=atlas['source_BGR_sha256']:raise ValueError('Approved independent source glyph reference differs')
    pattern=pp.PrintPattern(atlas,reference);native_map={'print_instance/Ad4::'+p['path_id']:p['native_path_id']for p in atlas['paths']};allowed={v for k,v in native_map.items()if k.rsplit('_',1)[-1]in set('JUNIO')}
    if len(allowed)!=5:raise ValueError('Only reviewed J/U/N/I/O exterior native paths are allowed')
    bank_path=root/'print_references/references.json';meta=next(m for m in json.loads(bank_path.read_text('utf8'))['references']if m['ad_id']=='Ad4'and m['shot_id']=='s09');z=dict(np.load(meta['descriptor_npz'],allow_pickle=False));z['source_image']=reference;z['reference_mask']=cv2.imread(meta['descriptor_mask'],0)>0
    requested=list(range(FIRST,LAST+256,256));neighbors=[FIRST-256,LAST+256];targets=sorted([*requested,*neighbors]);scope=None
    if a.prepared_root is not None:requested,neighbors,targets,scope=current_top_only_scope(a.prepared_root,a.probe_pts)
    first,last=min(requested),max(requested);images=exact_frames(video,targets)
    source_registrations={};source_registration_path=None;source_registration_sha=None
    if scope is not None:
        source_registration_path=root/'registered_full/s09/registration.jsonl';source_registration_sha=sha(source_registration_path)
        for line in source_registration_path.open(encoding='utf8'):
            rr=json.loads(line)
            if rr['pts'] not in targets:continue
            ad=next((v for v in rr.get('prints',[])if v.get('ad')=='Ad4'),None)
            if ad is not None and ad.get('source_pattern_check',{}).get('supported') is True:
                source_registrations[rr['pts']]=dict(H_current_from_reference=ad['H_current_from_reference'],previous_source_pattern_check=ad['source_pattern_check'],source_graph_parent_pts=ad.get('source_graph_parent_pts'),registration_file=str(source_registration_path),registration_file_sha256=source_registration_sha,only_source_search_navigation_reused=True)
        scope['source_registration_path']=str(source_registration_path);scope['source_registration_sha256']=source_registration_sha
    if hashlib.sha256(images[CENTER].tobytes()).hexdigest()!=pilot_record['source_BGR_sha256']:raise ValueError('Approved exact central source RGB differs')
    out.mkdir(parents=True);
    if scope is not None:write(out/'current_source_scope.json',scope)
    capture=out/'capture';folder=capture/'s09';folder.mkdir(parents=True);profiles=folder/'profiles';profiles.mkdir();qualified=out/'qualified';(qualified/'s09').mkdir(parents=True)
    approval=dict(authority='Explicit current primary-session source-image review and task instruction',approved_center_pts=CENTER,approved_center_actual_q50_rows=31,source_identity='Ad4 JUNIOR banner in known main-camera source context',approved_interval=dict(first_pts=first,last_pts=last,native_step=256,frames=len(requested),requested_native_PTS=requested),current_source_group_audit=scope,neighbor_anchor_pts_navigation_only=neighbors,only_native_glyph_exterior_paths=sorted(allowed),O_inner_and_R_not_admitted=True,reference_atlas=str(atlas_path),reference_atlas_sha256=sha(atlas_path),original_pending_source_rows=str(pilot/'source_rows.jsonl'),original_pending_source_rows_sha256=sha(pilot/'source_rows.jsonl'),original_RootSIFT_banner_and_candidate_evidence=str(pilot/'report.json'),original_evidence_sha256=sha(pilot/'report.json'),physical_identity_not_inferred_from_H_or_native_projection=True)
    approval_path=out/'source_identity_approval.json';write(approval_path,approval);approval_sha=sha(approval_path)
    files=[Path(__file__),Path(pp.__file__),Path(source_exclusion.__code__.co_filename),Path(build.__code__.co_filename)];binding=dict(source_video_sha256=index['source']['actual_sha256'],source_sha256=index['source']['actual_sha256'],independent_source_capture=True,legacy_new_shot_inputs_read=False,source_identity_approval=str(approval_path),source_identity_approval_sha256=approval_sha,source_atlas=str(atlas_path),source_atlas_sha256=sha(atlas_path),native_projection_called=False,native_state_loaded=False,query_spacing_px=8.,fit_spacing_px=48.,short_feature_minimum_samples=4,public_qualification_thresholds_changed=False,quantile_operator='actual signed RGB luminance q10/q50/q90, q50 output',implementation_sha256={str(p.resolve()):sha(p)for p in files},fit_eligible=False,fit_executed=False)
    write(capture/'binding.json',binding);summaries=[];frame_rows={};previous=None;total=0;all_reasons=Counter()
    with(folder/'source_rows.jsonl').open('x',encoding='utf8')as raw_out,(qualified/'s09/source_rows.jsonl').open('x',encoding='utf8')as selected_out,(folder/'registration.jsonl').open('x',encoding='utf8')as registration_out:
        for pt in targets:
            image=images[pt];rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);ex,mp=source_exclusion(root,'s09',pt);bgr_sha=hashlib.sha256(image.tobytes()).hexdigest();rows=[];unthinned=[];readout=None;captured={};direct=None;banner_check=None;attempts=[];reg=None;basis=None;profile=None
            if pt==CENTER:
                reg=center_registration;readout=pilot_record['source_readout'];unthinned=copy.deepcopy(pilot_record['all_unthinned_public_reader_rows']);rows=copy.deepcopy(pilot_record['rows']);profile=str(pilot/'all_source_transition_profiles.json');basis='exact_reviewed_31_actual_q50_rows_preserved';attempts=[dict(basis='reviewed_original_pilot',registration=reg,evidence=str(pilot/'report.json'))]
            else:
                seeds=[]
                source_registration=source_registrations.get(pt)
                if source_registration is not None:
                    hh=np.asarray(source_registration['H_current_from_reference']);check=source_patch_check(image,ex,meta,z,hh)
                    attempts.append(dict(basis='current_independent_registered_full_Ad4_source_H_navigation_only',evidence=source_registration,fresh_original_source_banner_check=check))
                    if check['supported']:seeds.append(('current_independent_registered_full_Ad4_source_H_navigation_only',hh));banner_check=check
                if not seeds:
                    direct=match_features((z['keypoints'],z['descriptors']),descriptors(image,(~ex).astype('uint8')*255),ReuseConfig(min_matches=6,ransac_px=3.))
                    if direct is not None:seeds.append(('current_source_RootSIFT',direct[2]));banner_check=source_patch_check(image,ex,meta,z,direct[2])
                seeds.append(('approved_central_source_word_H_navigation_only',np.array(center_registration['H'])))
                if previous is not None:seeds.append(('previous_actual_source_word_registration_navigation_only',previous))
                choices=[]
                for source_basis,h in seeds:
                    if not np.isfinite(h).all():continue
                    candidate=pattern.register(image,ex.astype('uint8'),h);attempts.append(dict(basis=source_basis,initial_source_H=np.asarray(h).tolist(),registration=candidate))
                    if candidate['status']=='source_pattern_registered':choices.append((candidate['correlation'],source_basis,candidate))
                if choices:
                    _,basis,reg=max(choices,key=lambda x:x[0]);unthinned,readout,captured=read_public(pattern,image,ex,reg,dict(source_sha256=index['source']['actual_sha256'],source_pts=pt,time_base='1/12800',shot_id='s09',decoded_native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
                    profile=profiles/f'{pt}.npz';save_profiles(profile,captured);write(profiles/f'{pt}_public_readings.json',captured)
                    for row in unthinned:
                        reading=captured['public_readings'][row['candidate']];np.testing.assert_array_equal(row['uv'],reading['actual_source_uv']);np.testing.assert_array_equal(row['actual_normal_quantiles'],reading['source_q10_q50_q90_normal_offsets_px']);row['atlas_path_id']=row['path_id'];row['path_id']=native_map[row['path_id']];row.update(measurement_operator='actual_RGB_white_ink_interface_q10_q50_q90',model_UV_synthesized=False,source_quantile_semantics='q10/q50/q90, not q25/q50/q75',native_or_H_vertices_used_as_observations=False)
                    for path in sorted(allowed):
                        group=[r for r in unthinned if r['path_id']==path];rows.extend(copy.deepcopy(group[i])for i in spaced_indices([r['uv']for r in group],spacing=48.))
            if reg is not None and reg['status']=='source_pattern_registered':previous=np.array(reg['H'])
            in_scope=pt in requested
            for row in rows:row.update(fit_eligible=in_scope,review_state='source_supported_approved_glyphs_with_public_RGB_shape_checks'if in_scope else'neighbor_source_navigation_anchor_only',source_identity_approval=str(approval_path),source_identity_approval_sha256=approval_sha,source_profile_proof=str(profile),source_original_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
            if pt==CENTER:
                for before,after in zip(pilot_record['rows'],rows,strict=True):
                    for key in('uv','normal','actual_normal_quantiles','actual_current_material_sides','source_profile_flags'):assert json.dumps(before[key],sort_keys=True)==json.dumps(after[key],sort_keys=True)
            pending=[dict(r,fit_eligible=False,pending_reason='outside_reviewed_visible_glyph_scope')for r in unthinned if r['path_id']not in allowed]
            record=dict(pts=pt,source_video_sha256=index['source']['actual_sha256'],source_BGR_sha256=bgr_sha,source_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),mask_path=str(mp),mask_sha256=sha(mp),rows=rows if in_scope else[],pending_rows=pending,fit_eligible=bool(rows)and in_scope,review_state='source_supported_approved_glyphs_with_public_RGB_shape_checks'if rows and in_scope else'pending_no_admitted_source_glyphs'if in_scope else'neighbor_source_anchor_navigation_only',source_readout=readout,source_profiles=None if profile is None else str(profile),source_identity_approval=str(approval_path),source_identity_approval_sha256=approval_sha,source_atlas=str(atlas_path),source_atlas_sha256=sha(atlas_path),reference_atlas_semantics_pending_review=False,original_pilot_evidence=str(pilot/'report.json'))
            raw_record=dict(record,rows=unthinned,fit_eligible=False,all_public_reader_actual_rows_preserved=True);raw_out.write(json.dumps(safe(raw_record),ensure_ascii=False,allow_nan=False)+'\n')
            if in_scope:selected_out.write(json.dumps(safe(record),ensure_ascii=False,allow_nan=False)+'\n')
            registration_out.write(json.dumps(safe(dict(pts=pt,RootSIFT_matches=0 if direct is None else len(direct[0]),initial_banner_patch_check_for_navigation_only=banner_check,source_word_registration_attempts=attempts,chosen_source_word_basis=basis,chosen_registration=reg,source_readout=readout,source_profiles=None if profile is None else str(profile),native_projection_called=False)),ensure_ascii=False,allow_nan=False)+'\n')
            if readout:all_reasons.update(readout.get('reasons',{}))
            admitted=len(rows)if in_scope else 0
            summaries.append(dict(pts=pt,requested=in_scope,rows=admitted,neighbor_anchor_actual_rows=len(rows)if not in_scope else 0,public_actual_rows=len(unthinned),word_status=None if reg is None else reg['status'],word_correlation=None if reg is None else reg.get('correlation'),chosen_source_word_basis=basis,public_readout=readout));frame_rows[pt]=record;total+=admitted
            print(json.dumps(dict(pts=pt,requested=in_scope,rows=admitted,actual_public_rows=len(unthinned),word_correlation=None if reg is None else reg.get('correlation'))),flush=True)
    if scope is not None:
        (qualified/'s09/source_rows.jsonl').write_text(''.join(json.dumps(safe(frame_rows[p]),ensure_ascii=False,allow_nan=False)+'\n'for p in requested),encoding='utf8')
    pose_source=root/'s09/navigation/poses_reanchored.npz';shutil.copyfile(pose_source,qualified/'s09/poses.npz');write(qualified/'binding.json',dict(binding,fit_eligible=True,source_capture=str(capture),source_capture_binding_sha256=sha(capture/'binding.json'),source_identity_approval=str(approval_path),source_identity_approval_sha256=approval_sha,qualification_method='Approved one-window Ad4 instance and known visible glyphs; unchanged public whole-word photographic registration, separated actual ink components and original RGB q50 profiles; no current native projection'))
    proofs=[]
    for pt in sorted({first,a.probe_pts if scope is not None else CENTER,last}):
        image=images[pt].copy();record=frame_rows[pt];original=out/f's09_pts{pt}_original.png';cv2.imwrite(str(original),image);proof=image.copy()
        for row in record['rows']:cv2.circle(proof,tuple(np.rint(row['uv']).astype(int)),3,(0,255,0),1,cv2.LINE_AA)
        cv2.putText(proof,f'Ad4 JUNIOR | S09 PTS {pt} | {len(record["rows"])} actual source-supported q50 | NO FIT',(20,35),0,.65,(0,220,255),2,cv2.LINE_AA);full=out/f'{pt}_source_supported_glyphs.jpg';cv2.imwrite(str(full),proof)
        crop=None
        if record['rows']:
            q=np.array([r['uv']for r in record['rows']]);lo=np.maximum(np.floor(q.min(0)-25).astype(int),[0,0]);hi=np.minimum(np.ceil(q.max(0)+26).astype(int),[1920,1080]);crop=out/f'{pt}_source_supported_glyphs_crop.png';cv2.imwrite(str(crop),proof[lo[1]:hi[1],lo[0]:hi[0]])
        proofs.append(dict(pts=pt,source_original=str(original),full_point_proof=str(full),crop_proof=None if crop is None else str(crop)))
    build(root,qualified,out/'source_packets',shot_ids=['s09']);packet=out/'source_packets/s09';arrays=dict(np.load(packet/'point_observations.npz',allow_pickle=False));mapping=json.loads((packet/'mapping.json').read_text('utf8'));expected=[(pt,row)for pt in requested for row in frame_rows[pt]['rows']]
    assert np.array_equal(arrays['pts'],[p for p,r in expected]);assert np.array_equal(arrays['uv'],[r['uv']for p,r in expected]);assert set(mapping['path_ids'])<=allowed;assert len(set(arrays['observation_ids']))==len(arrays['pts']);assert (len(frame_rows[CENTER]['rows'])==31 if CENTER in requested else frame_rows[CENTER]['rows']==[])
    center_uv=arrays['uv'][arrays['pts']==CENTER]
    if CENTER in requested:assert np.array_equal(center_uv,[r['uv']for r in pilot_record['rows']])
    else:assert not len(center_uv)
    if source_registration_path is not None and sha(source_registration_path)!=source_registration_sha:raise ValueError('Current source registration changed during read')
    missing=[pt for pt in requested if not frame_rows[pt]['rows']];report=dict(complete=True,requested_native_interval=dict(first_pts=first,last_pts=last,frames=len(requested),step=256,requested_native_PTS=requested),current_source_group_audit=scope,neighbor_anchors_navigation_only=neighbors,qualified_frames=len(requested)-len(missing),rows=total,missing_requested_native_PTS=missing,rows_by_path=dict(Counter(r['path_id']for p,r in expected)),frames=summaries,representative_source_proofs=proofs,source_packet=str(packet),packet_manifest=str(out/'source_packets/packets_manifest.json'),packet_sha256=sha(packet/'point_observations.npz'),source_rows=str(qualified/'s09/source_rows.jsonl'),raw_source_evidence=str(folder/'source_rows.jsonl'),registration_evidence=str(folder/'registration.jsonl'),source_identity_approval=str(approval_path),source_identity_approval_sha256=approval_sha,approved_31_center_q50_rows_preserved_exactly=True,approved_center_is_navigation_only=CENTER not in requested,all_packet_UV_PTS_exact=True,native_projection_called=False,current_camera_or_station_state_modified=False,public_thresholds_changed=False,O_inner_and_R_rows_admitted=False,source_readout_rejection_counts=dict(all_reasons),fit_executed=False,seconds=time.monotonic()-started)
    write(out/'report.json',report);print(json.dumps({k:report[k]for k in['qualified_frames','rows','missing_requested_native_PTS','source_packet','seconds']}),flush=True)


if __name__=='__main__':main()