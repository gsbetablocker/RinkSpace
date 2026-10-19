"""Append approximate user R1/R2 outer strokes, preserving off-image context.

The original attachment is registered through source RGB texture. Actual drawn
centres, rather than a fitted ellipse, constrain the existing R-side house-ring
producer. Black-padding strokes stay context and are not native RGB evidence.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image
from register_user_native230024192_all_visible import ROOT,BASE,SOURCE,SOURCE_SHA,SUFFIX,ref,write,sha,canonical_sha,register,sparse_indices

ATTACHMENT=Path('C:/Users/REDACTED/AppData/Local/Temp/codex-clipboard-private-attachment-00029.png')
NEXT=SUFFIX+'_R1_R2_outer'


def normals(points):
    points=np.asarray(points,float);arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))];out=[]
    for i,p in enumerate(points):
        local=points[abs(arc-arc[i])<=15.5];centered=local-local.mean(0)
        _,vectors=np.linalg.eigh(centered.T@centered);tangent=vectors[:,-1]
        direction=points[min(i+1,len(points)-1)]-points[max(i-1,0)]
        if tangent@direction<0:tangent=-tangent
        out.append([-tangent[1],tangent[0]])
    return np.asarray(out),arc


def drawn_ring(marked,roi,inverse):
    x0,y0,x1,y1=roi;color=np.array([4,36,51]);crop=marked[y0:y1,x0:x1]
    mask=(np.max(abs(crop.astype(np.int16)-color),axis=2)<=3).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    label=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);yy,xx=np.nonzero(labels==label);xx+=x0;yy+=y0
    middle=.5*(yy.min()+yy.max());upper=[];lower=[];groups={}
    for x in np.unique(xx):
        values=yy[xx==x]
        for kind,choice in [('upper',values[values<=middle]),('lower',values[values>middle])]:
            if len(choice):
                center=[float(x),float(np.median(choice))];(upper if kind=='upper'else lower).append(center)
                groups[tuple(center)]=np.c_[np.full(len(choice),x),choice].astype(float)
    screen=np.asarray(upper+lower[::-1]);native=screen@inverse[:,:2].T+inverse[:,2]
    all_normals,arc=normals(native);span=[];cell_info=[]
    for p,s,n in zip(native,screen,all_normals):
        actual=groups[tuple(s)];mapped=actual@inverse[:,:2].T+inverse[:,2];signed=(mapped-p)@n
        half=.5*np.sum(abs(n@inverse[:,:2]));span.append([signed.min()-half,signed.max()+half])
        cell_info.append(dict(pixel_count=len(actual),screenshot_bbox_xyxy=np.r_[actual.min(0),actual.max(0)].tolist()))
    inside=np.all(native>=0,axis=1)&np.all(native<[1920,1080],axis=1)
    return screen,native,all_normals,np.asarray(span),inside,cell_info


def main():
    cv2.setNumThreads(1)
    authority_path=BASE/f'human_registered_hard_truth_authorities_{NEXT}.json'
    if authority_path.exists():raise FileExistsError('Preserve prior ring source')
    saved=BASE/ATTACHMENT.name
    if not saved.exists():shutil.copyfile(ATTACHMENT,saved)
    assert sha(saved)==sha(ATTACHMENT)
    marked=np.asarray(Image.open(saved).convert('RGB'));rgb=np.asarray(Image.open(SOURCE).convert('RGB'))
    if marked.shape!=(1125,2184,3):raise ValueError('Read the original attachment dimensions')
    inverse,registration=register(rgb,marked)
    parent_ice=BASE/f'ice_sources_{SUFFIX}.json';ice=json.loads(parent_ice.read_text(encoding='utf8'))
    parent_authority=BASE/f'human_registered_hard_truth_authorities_{SUFFIX}.json';authority=json.loads(parent_authority.read_text(encoding='utf8'))
    parent_guard=BASE/f'human_registered_full_stroke_hard_guard_{SUFFIX}.json';guard=json.loads(parent_guard.read_text(encoding='utf8'))
    with np.load(guard['guard_arrays_reference']['path'],allow_pickle=False)as previous:arrays={k:previous[k].copy()for k in previous.files}
    parent_caps=BASE/f'human_registered_stroke_end_2D_source_footprint_support_{SUFFIX}.json';caps=json.loads(parent_caps.read_text(encoding='utf8'))
    facts=[];new_records=[];new_guards=[]
    ice_path=BASE/f'ice_sources_{NEXT}.json'
    for row,roi in [(2,(0,470,490,560)),(1,(700,475,1370,565))]:
        screen,all_uv,all_normals,all_spans,visible,cells=drawn_ring(marked,roi,inverse)
        uv=all_uv[visible];screen_visible=screen[visible];normal=all_normals[visible];spans=all_spans[visible]
        visible_indices=np.flatnonzero(visible);chosen,arc=sparse_indices(uv);manual=8.;reg=registration['registration_uncertainty_native_px']
        bounds=spans+np.array([-reg-manual,reg+manual]);identifier=f'user_native230024192_R{row}_outer_approx_20261008'
        curve=dict(kind='house_ring',side='R',row=row,ring='outer');trace_path=BASE/f'user_native230024192_R{row}_outer_approx_registered_stroke.json'
        write(trace_path,dict(schema='rinkspace.registered-user-source-stroke.v1',id=identifier,user_label=f'R{row}外圈',physical_curve=curve,
            registration=registration,original_attachment=ref(saved),source_pts=230024192,time_base='1/15360',source_raw_frame=str(SOURCE),source_file_sha256=sha(SOURCE),
            user_stroke_raw_screenshot_px=screen_visible.tolist(),user_stroke_raw_native_px=uv.tolist(),
            original_all_drawn_stroke_screenshot_px=screen.tolist(),all_drawn_native_coordinates_including_off_image=all_uv.tolist(),
            native_frame_visible_original_vertex_indices=visible_indices.tolist(),off_image_original_vertex_indices=np.flatnonzero(~visible).tolist(),
            off_image_stroke_native_px=all_uv[~visible].tolist(),off_image_stroke_scope='approximate human extrapolated location/shape context; not source RGB pixels and not admitted to this native FIT/guard',
            source_RGB_actual_measurement=False,direct_human_approximate_position_shape=True,ellipse_parameters_not_measured=True,
            physical_endpoint_or_corner_observation=False,normal_interval_px=bounds.tolist(),practical_manual_location_uncertainty_native_px=manual,
            practical_uncertainty_is_user_pixel_precision=False))
        item=dict(id=identifier,pts=230024192,actual_pts=230024192,feature_id=f'R{row}',curve=curve,split='fit',geometry_role='user_approximate_house_outer_ring_axis',
            uv=uv[chosen].tolist(),normal=normal[chosen].tolist(),normal_interval_px=bounds[chosen].tolist(),sigma_px=[manual]*len(chosen),
            source_optical_normal_support_authorized=False,source_raw_frame=str(SOURCE),source_file_sha256=sha(SOURCE),source_user_annotation=ref(saved),source_annotation_registration=ref(trace_path),
            user_label=f'R{row}外圈',user_confirmed_source_axis=True,direct_human_GT=True,source_RGB_actual_measurement=False,
            model_projection_used_to_generate_or_select_uv=False,point_correspondence=False,curve_observation_kind='normal_interval',endpoint_observation=False,
            practical_manual_location_uncertainty_native_px=manual,practical_uncertainty_is_user_pixel_precision=False,
            physical_ring_producer_mapping=dict(side='R',house_row=row,ring='outer',current_radius_parameter_index=17,current_radius_multiplier=1.,
                current_house_center_parameters=[12,13,14,15,16],current_side_sign=1.,F_side_used=False),
            sourceUV_origin='direct approximately hand-drawn external house ring, inverse verified source-RGB display registration',
            complete_human_stroke=dict(reference=ref(trace_path),native_coordinates_field='user_stroke_raw_native_px',sample_count=len(uv)),
            sigma_semantics='8 native px approximate manual engineering scale; exact coloured pixel/registration span kept separately; not calibrated physical paint or ellipse precision')
        index=len(ice['rows']);ice['rows'].append(item)
        record=dict(id=identifier,family='native_ice_curve',packet_reference={},original_row_index=index,source_row_canonical_json_sha256=canonical_sha(item),
            pts=230024192,actual_pts=230024192,time_base='1/15360',source_video_sha256=SOURCE_SHA,curve=curve,geometry_role=item['geometry_role'],
            uv=item['uv'],normal=item['normal'],normal_interval_px=item['normal_interval_px'],lower_bound_px=bounds[chosen,0].tolist(),upper_bound_px=bounds[chosen,1].tolist(),
            direct_human_GT=True,direct_human_approximate_position_shape=True,model_or_RGB_transport_used_to_generate_UV=False,historical_sigma_used_as_hard_tolerance=False,
            source_raw_frame=str(SOURCE),source_file_sha256=sha(SOURCE),source_annotation_attachment=ref(saved),source_annotation_registration=ref(trace_path),
            endpoint_observation=False,complete_human_stroke=item['complete_human_stroke'],
            practical_manual_location_uncertainty_native_px=manual,practical_uncertainty_is_user_pixel_precision=False)
        new_records.append(record);gi=len(guard['records']);prefix=f'record_{gi:02d}'
        for key,value in [('uv',uv),('normal',normal),('normal_interval_px',bounds),('annotation_stroke_normal_interval_px',spans),('screenshot_xy',screen_visible),('source_arc_length_px',arc)]:arrays[prefix+'_'+key]=value
        grow=dict(id=identifier,authority_record_index=gi,authority_source_row_canonical_json_sha256=canonical_sha(item),family='native_ice_curve',curve=curve,geometry_role=item['geometry_role'],
            pts=230024192,actual_pts=230024192,vertex_count=len(uv),array_prefix=prefix,source_raw_frame=str(SOURCE),source_file_sha256=sha(SOURCE),
            source_annotation_attachment=ref(saved),source_annotation_registration=ref(trace_path),full_trace_is_guard_only=True,dense_pixels_are_not_independent_FIT_residuals=True,endpoint_observation=False)
        guard['records'].append(grow);new_guards.append(grow)
        for vi,name in [(0,'first'),(len(uv)-1,'last')]:
            n=normal[vi];tangent=np.array([-n[1],n[0]]);radius=float(np.max(abs(bounds[vi])));cell=cells[int(visible_indices[vi])]
            caps['records'].append(dict(id=identifier,source_full_guard_record_index=gi,source_vertex_index=vi,stroke_end=name,pts=230024192,actual_pts=230024192,
                uv=uv[vi].tolist(),normal=n.tolist(),tangent=tangent.tolist(),normal_cell_support_px=bounds[vi].tolist(),tangent_cell_support_px=[-radius,radius],
                source_annotation_attachment=ref(saved),source_annotation_registration=ref(trace_path),source_raw_frame=str(SOURCE),source_file_sha256=sha(SOURCE),
                selected_actual_coloured_pixel_count=cell['pixel_count'],selected_screenshot_bbox_xyxy=cell['screenshot_bbox_xyxy'],physical_endpoint_or_corner_observation=False,
                rectangular_marker_joint_not_free_cap=False,normal_band_from_frozen_guard_not_replaced=True,registration_uncertainty_native_px=reg,
                practical_manual_location_uncertainty_native_px=manual,source_frame_edge_or_marker_join_is_not_physical_endpoint=True))
        facts.append(dict(id=identifier,user_label=f'R{row}外圈',physical_curve=curve,registration=registration,trace_reference=ref(trace_path),
            all_drawn_vertices=len(all_uv),native_frame_vertices=len(uv),off_image_context_vertices=int((~visible).sum()),FIT_queries=len(chosen),
            native_visible_bbox_xyxy=np.r_[uv.min(0),uv.max(0)].tolist(),off_image_drawing_is_RGB_measurement=False,ellipse_parameters_or_physical_endpoints_measured=False))
    ice['user_approximate_R1_R2_outer_append']=dict(parent=ref(parent_ice),source_pts=230024192,added_ids=[r['id']for r in new_records],original_rows_exact_prefix=True)
    write(ice_path,ice)
    for record in new_records:record['packet_reference']=ref(ice_path)
    authority['records'].extend(new_records);authority['R1_R2_approximate_outer_append']=dict(parent=ref(parent_authority),attachment=ref(saved),registration=registration)
    write(authority_path,authority)
    guard_arrays=BASE/f'human_registered_full_stroke_hard_guard_{NEXT}.npz';np.savez_compressed(guard_arrays,**arrays)
    guard_path=BASE/f'human_registered_full_stroke_hard_guard_{NEXT}.json'
    guard.update(primary_authority_reference=ref(authority_path),guard_arrays_reference=ref(guard_arrays),record_count=len(guard['records']),
        full_stroke_vertex_count=sum(r['vertex_count']for r in guard['records']),primary_UV_count=sum(len(r['uv'])for r in authority['records']))
    write(guard_path,guard)
    caps_path=BASE/f'human_registered_stroke_end_2D_source_footprint_support_{NEXT}.json';caps.update(primary_authority_reference=ref(authority_path),full_trace_reference=ref(guard_path),full_trace_arrays_reference=ref(guard_arrays),endpoint_record_count=len(caps['records']));write(caps_path,caps)
    previous_confirmed=BASE/f'user_confirmed_native_source_priority_authorities_{SUFFIX}.json';confirmed=json.loads(previous_confirmed.read_text(encoding='utf8'))
    for record in confirmed['records']:
        if record['family']=='native_ice_curve':record['packet_reference_before_R1_R2_append']=deepcopy(record['packet_reference']);record['packet_reference']=ref(ice_path)
    confirmed_path=BASE/f'user_confirmed_native_source_priority_authorities_{NEXT}.json';write(confirmed_path,confirmed)
    cfg_path=ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_resume_user230024192_20261008.json';cfg=json.loads(cfg_path.read_text(encoding='utf8'))
    old_policy=Path(cfg['source_options']['--practical-human-location-envelope']);policy=json.loads(old_policy.read_text(encoding='utf8'))
    native=[r for r in policy['records']if r['family']=='native'];wide=[r for r in policy['records']if r['family']=='wide']
    native.extend(dict(family='native',id=r['id'],full_vertices=r['complete_human_stroke']['sample_count'],normal_expansion_px=0.,tangent_expansion_px=0.,
        engineering_uncorrelated_location_scale_px=8.,manual_location_uncertainty_native_px=8.,normal_and_tangent_manual_location_already_bound_in_source=True,
        recovery_uncertainty_is_not_additional_support_expansion=True,location_scale_is_not_a_user_accuracy_threshold=True)for r in new_records)
    policy['records']=native+wide;policy['authority_references']['native']=ref(authority_path);policy['previous_R1_R2_policy_reference']=ref(old_policy)
    policy_path=BASE/'fixed_practical_manual_location_envelope_full_user230024192_R1_R2_v1.json';write(policy_path,policy)
    replacements={'--additional-ice-source':ice_path,'--direct-human-truth-authority':authority_path,'--direct-human-full-stroke-guard':guard_path,
        '--direct-human-stroke-cap-support':caps_path,'--user-confirmed-source-truth-authority':confirmed_path,'--practical-human-location-envelope':policy_path}
    for key,path in replacements.items():cfg['source_options'][key]=str(path);cfg['new_source_references'][key]=ref(path)
    added_scalars=2*sum(len(r['uv'])for r in new_records);cfg['new_source_scalar_rows']+=added_scalars;cfg['planned_source_rows']+=added_scalars;cfg['new_manual_stroke_groups']+=2
    cfg['latest_R1_R2_outer_approximate_human_append']=dict(attachment=ref(saved),records=facts,all_previous_targets_preserved=True,manual_normal_error_added_once=True)
    write(cfg_path,cfg)
    fact_path=ROOT/'events/ankara_2026/model_constraints/user_native230024192_all_visible_20261008.json';fact=json.loads(fact_path.read_text(encoding='utf8'))
    fact['latest_R1_R2_outer_user_approximate_shape']=dict(attachment=ref(saved),registration=registration,records=facts,
        authority=ref(authority_path),guard=ref(guard_path),ice_packet=ref(ice_path),caps=ref(caps_path),confirmed_source_authority=ref(confirmed_path),manual_normal_error_added_once=True)
    fact['hard_truth_authority']=ref(authority_path);fact['full_stroke_guard']=ref(guard_path);fact['source_2D_coverage_support']=ref(caps_path);fact['ice_packet']=ref(ice_path);fact['confirmed_native_source_priority_authority']=ref(confirmed_path)
    fact['full_nine_shot_resume_inputs']['sha256']=sha(cfg_path);write(fact_path,fact)
    print(json.dumps(dict(ice=str(ice_path),authority=str(authority_path),guard=str(guard_path),caps=str(caps_path),confirmed=str(confirmed_path),
        full_resume_config=str(cfg_path),new_groups=2,newFITscalars=added_scalars,total_native_manual_groups=len(authority['records']),total_caps=len(caps['records']),facts=facts),ensure_ascii=False))


if __name__=='__main__':main()
