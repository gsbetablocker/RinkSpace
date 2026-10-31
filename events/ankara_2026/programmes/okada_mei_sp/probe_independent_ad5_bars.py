"""Three current-source Ad5 blue/green native-bar exterior pilots, all pending.

Material-chart contours and independently supplied source windows navigate only.
The public RGB component/profile/closed-shape reader supplies every q50. No
native camera projection, historical point input, solver, or new weight is used.
"""
from pathlib import Path
import argparse,hashlib,json
from collections import Counter
import cv2,numpy as np
from capture_native_panels import panel_material_metadata
from capture_independent_badge import exact_frames,sha,write
from capture_independent_badge_full import read_profile_proof
from capture_independent_materials import source_exclusion,spaced_indices,plain

CASES=[('s07',1703936,[625,454,785,483],[625,520,785,550]),
       ('s08',1779200,[500,395,655,422],[500,455,655,484]),
       ('s09',1889536,[525,420,740,454],[525,499,740,535])]


def actual_guard(row,exclusion,colour):
    uv=np.array(row['uv']);n=np.array(row['normal']);qq=np.array(row['actual_source_quantiles']);offset=(qq[:,[0,2]].reshape(-1,2)-uv)@n;lo,hi=float(offset.min()-3),float(offset.max()+3)
    xy=uv+np.linspace(lo,hi,max(2,int(np.ceil((hi-lo)*4))+1))[:,None]*n;reasons=[]
    inside=bool((xy>=0).all()and(xy<=[exclusion.shape[1]-1,exclusion.shape[0]-1]).all())
    if not inside:reasons.append('actual_quantile_guard_crosses_original_image_boundary')
    else:
        lower,upper=np.floor(xy).astype(int),np.ceil(xy).astype(int)
        if any(exclusion[y[:,1],x[:,0]].any()for x in(lower,upper)for y in(lower,upper)):reasons.append('actual_quantile_guard_crosses_current_fine_SAM_or_HUD')
    c=np.array(row['actual_source_sides']['inside_outside_source_RGB'])[0];channel=2 if colour=='blue'else 1;other=[i for i in range(3)if i!=channel];colour_supported=bool(c[channel]>max(c[other]))
    if not colour_supported:reasons.append('actual_inward_RGB_does_not_support_expected_bar_colour')
    return dict(actual_quantile_normal_guard_px=[lo,hi],all_source_pixel_contributors_checked=True,inward_RGB=c.tolist(),expected_colour=colour,inward_expected_channel_dominant=colour_supported,guard_reasons=reasons)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.root.resolve();out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    cv2.setNumThreads(2);index=json.loads((root/'index.json').read_text('utf8'));video=Path(index['source']['resolved_path'])
    if sha(video)!=index['source']['actual_sha256']:raise ValueError('Actual original source differs')
    scene_manifest=root.parent/'prepared_joint_ice/joint_manifest.json';manifest=json.loads(scene_manifest.read_text('utf8'));scene_path=Path(manifest['scene_path']);scene=json.loads(scene_path.read_text('utf8'));metadata=panel_material_metadata(scene,scene_path)
    world=dict(np.load(root/'native_world.npz',allow_pickle=False));cuts=json.loads((root/'native_world.json').read_text('utf8'))['cuts'];catalogue={};positions={}
    for instance in('Ad4','Ad5','Ad17','Ad18'):
        candidates=[c for c in cuts if c['id'].startswith('print_instance/'+instance+'::')];cut=next((c for c in candidates if 'blue_bar' in c['id']),candidates[0]);xyz=world['xyz'][cut['begin']:cut['end']];positions[instance]=dict(world_min=xyz.min(0).tolist(),world_max=xyz.max(0).tolist(),used_for_physical_layout_check_only=True)
    for colour in('blue','green'):
        matches=[(pid,m)for pid,m in metadata.items()if pid.startswith('print_instance/Ad5::')and pid.endswith('_'+colour+'_bar/path0')]
        if len(matches)!=1:raise ValueError('Unique current native Ad5 bar path is required')
        pid,m=matches[0];asset=Path(m['asset_path'])
        if sha(asset)!=m['asset_sha256']:raise ValueError('Current native bar asset hash changed')
        data=json.loads(asset.read_text('utf8'));template=next(t for t in data['templates']if t['id']==m['template_id']);path=next(p for p in template['paths']if p['id']==m['template_path_id']);cut=next(c for c in cuts if c['id']==pid)
        if not path['closed']or not cut['closed']or path['role']!='printed_ink_external_boundary':raise ValueError('Only modeled closed printed-bar exterior is allowed')
        catalogue[colour]=dict(native_path_id=pid,metadata=m,material_uv=np.array(path['material_uv'],float),closed=True,role=path['role'])
    images=exact_frames(video,[pt for sid,pt,b,g in CASES]);out.mkdir(parents=True);(out/'profiles').mkdir();summaries=[];all_machine=[];files=[Path(__file__),Path(read_profile_proof.__code__.co_filename),Path(source_exclusion.__code__.co_filename)]
    binding=dict(source_video=str(video),source_video_sha256=index['source']['actual_sha256'],independent_source_capture=True,legacy_new_shot_point_inputs_read=False,native_camera_projection_called=False,native_shape_vertices_used_as_observations=False,new_weights_created=False,physical_instance='Ad5',physical_instance_scope=[dict(shot=s,pts=p)for s,p,b,g in CASES],physical_identity_authority='Current primary task and independent source review: 40th-anniversary/MEIWA content beside Ad4; known main camera at negative Y for S07/S09 and known end camera at negative X/negative Y for S08',same_artwork_Ad18_not_identity_evidence=True,current_catalogue_layout=positions,layout_check='Ad5 and adjacent Ad4 occupy positive-Y board; repeated Ad18 and Ad17 occupy opposite negative-Y board; known rig/context declaration chooses Ad5, not text alone',native_paths={c:v['native_path_id']for c,v in catalogue.items()},native_path_metadata={c:v['metadata']for c,v in catalogue.items()},source_query_spacing_px=24.,target_source_spacing_px=48.,short_closed_shape_minimum_samples=4,public_reader_thresholds_changed=False,green_reader='unchanged public dark-material component exterior plus actual inward G-dominant RGB support; no new green-weighted image signal',fit_eligible=False,review_state='pending_primary_original_image_pilot_review',implementation_sha256={str(p.resolve()):sha(p)for p in files})
    write(out/'binding.json',binding)
    with(out/'source_rows.jsonl').open('x',encoding='utf8')as output:
        for sid,pt,blue,green in CASES:
            image=images[pt];given=root.parent/'source_gap_review'/f'{sid}_{pt}_original.png';provided=cv2.imread(str(given))
            if provided is None or not np.array_equal(provided,image):raise ValueError('Provided original-frame extraction differs from exact native video PTS')
            rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);ex,mp=source_exclusion(root,sid,pt);rows=[];pending=[];notes=[]
            for colour,roi in(('blue',blue),('green',green)):
                item=catalogue[colour];uv=item['material_uv'];lo,hi=uv.min(0),uv.max(0);box=np.array(roi,float);search=box[:2]+(uv-lo)/(hi-lo)*(box[2:]-box[:2]);profile=out/'profiles'/f'{sid}_{pt}_{colour}.npz'
                public,readout,raw,arrays=read_profile_proof(rgb,search,ex,dict(source_sha256=index['source']['actual_sha256'],source_pts=pt,time_base='1/12800',shot_id=sid,decoded_native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()),material_signal='blue'if colour=='blue'else'dark');np.savez_compressed(profile,**arrays)
                public_uv={tuple(r['uv'])for r in public};certifiable=[];raw_rows=[]
                for order,row in enumerate(raw):
                    np.testing.assert_array_equal(row['uv'],np.array(row['actual_source_quantiles'])[0,1]);guard=actual_guard(row,ex,colour);organized=tuple(row['uv'])in public_uv;reasons=list(guard['guard_reasons'])
                    if not organized:reasons.append('not_selected_by_unchanged_public_closed_native_shape_organization')
                    record=dict(row,path_id=item['native_path_id'],entity='Ad5',bar_colour=colour,source_profile_proof=str(profile),source_profile_sha256=sha(profile),actual_source_guard=guard,public_closed_shape_qualified=organized,machine_certifiable_source_boundary=not reasons,pending_reasons=reasons,fit_eligible=False,review_state='pending_primary_original_source_pilot_review',native_material_metadata=item['metadata'],source_search_window_only=roi,native_or_bbox_vertices_are_not_observations=True)
                    raw_rows.append(record)
                    if not reasons:certifiable.append(record)
                selected=[certifiable[i]for i in spaced_indices([r['uv']for r in certifiable],spacing=48.)];rows.extend(selected);pending.extend(raw_rows)
                note=dict(colour=colour,native_path_id=item['native_path_id'],reader_material_signal='blue'if colour=='blue'else'dark',source_search_window=roi,native_material_chart_contour_used_as_search_only=True,public_reader_record=readout,all_actual_exterior_candidates=raw_rows,public_actual_rows=len(public),machine_certifiable_before_spacing=len(certifiable),spaced_pilot_rows=len(selected),source_profile_proof=str(profile),source_profile_sha256=sha(profile));notes.append(note)
            frame=dict(shot_id=sid,pts=pt,source_video_sha256=index['source']['actual_sha256'],source_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),source_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),source_original=str(given),source_original_file_sha256=sha(given),mask_path=str(mp),mask_sha256=sha(mp),rows=rows,pending_rows=pending,fit_eligible=False,review_state='pending_primary_original_source_pilot_review',physical_identity_binding=str(out/'binding.json'),machine_certifiable_rows=len(rows))
            output.write(json.dumps(plain(frame),ensure_ascii=False,allow_nan=False)+'\n');write(out/f'{sid}_{pt}_readout.json',dict(frame=frame,bars=notes))
            proof=image.copy()
            for q in pending:cv2.circle(proof,tuple(np.rint(q['uv']).astype(int)),2,(0,130,255),1,cv2.LINE_AA)
            for q in rows:
                cv2.circle(proof,tuple(np.rint(q['uv']).astype(int)),3,(0,255,0),1,cv2.LINE_AA)
                for side in np.array(q['actual_source_quantiles']):cv2.line(proof,tuple(np.rint(side[0]).astype(int)),tuple(np.rint(side[2]).astype(int)),(255,0,255),1,cv2.LINE_AA)
            cv2.putText(proof,f'Ad5 actual blue/green bar exterior | {sid} PTS {pt} | {len(rows)} source q50 PENDING | NO FIT',(20,35),0,.65,(0,220,255),2,cv2.LINE_AA)
            full=out/f'{sid}_{pt}_actual_bars_pending.jpg';cv2.imwrite(str(full),proof)
            bounds=np.array([blue,green]);crop_lo=np.maximum(bounds[:,:2].min(0)-30,[0,0]);crop_hi=np.minimum(bounds[:,2:].max(0)+31,[1920,1080]);crop=out/f'{sid}_{pt}_actual_bars_crop.png';cv2.imwrite(str(crop),proof[crop_lo[1]:crop_hi[1],crop_lo[0]:crop_hi[0]])
            summary=dict(shot=sid,pts=pt,source_time_seconds=pt/12800.,source_original=str(given),source_point_proof=str(full),source_crop_proof=str(crop),machine_certifiable_rows=len(rows),rows_by_native_path=dict(Counter(q['path_id']for q in rows)),bar_results=[dict(colour=n['colour'],native_path_id=n['native_path_id'],public_actual_rows=n['public_actual_rows'],machine_certifiable_before_spacing=n['machine_certifiable_before_spacing'],spaced_pilot_rows=n['spaced_pilot_rows'],reader_status=n['public_reader_record'].get('status'))for n in notes],fit_eligible=False);summaries.append(summary);all_machine.extend(rows);print(json.dumps(plain(summary),ensure_ascii=True),flush=True)
    report=dict(binding,frames=summaries,total_machine_certifiable_pilot_rows=len(all_machine),raw_actual_exterior_candidates=sum(n['public_actual_rows']for s in summaries for n in s['bar_results']),source_rows=str(out/'source_rows.jsonl'),all_actual_UV_exactly_equal_source_q50=True,source_profiles_and_component_masks_preserved=True,registered_panel_edges_file_modified=False,no_fit_packet_created=True,fit_executed=False,other_frames_processed=False)
    write(out/'report.json',report)


if __name__=='__main__':main()