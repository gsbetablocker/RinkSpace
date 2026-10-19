"""Append the user's partial A4 arc on the existing native hockey-circle.

Drawn pixels constrain only the covered visible arc. The drawing endpoints do
not identify a physical circle endpoint, center, radius or angular station.
All previous manual and real RGB source packages remain immutable.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image
from register_user_native230024192_all_visible import ROOT,BASE,SOURCE,SOURCE_SHA,SUFFIX,ref,write,sha,canonical_sha,register,sparse_indices
from append_user_native230024192_R1_R2_outer import normals

ATTACHMENT=Path('C:/Users/REDACTED/AppData/Local/Temp/codex-clipboard-private-attachment-00027.png')
PREVIOUS=SUFFIX+'_R1_R2_outer'
NEXT=PREVIOUS+'_A4_arc'


def main():
    cv2.setNumThreads(1)
    authority_path=BASE/f'human_registered_hard_truth_authorities_{NEXT}.json'
    if authority_path.exists():raise FileExistsError('Preserve prior A4 registration')
    saved=BASE/ATTACHMENT.name
    if not saved.exists():shutil.copyfile(ATTACHMENT,saved)
    assert sha(saved)==sha(ATTACHMENT)
    marked=np.asarray(Image.open(saved).convert('RGB'));rgb=np.asarray(Image.open(SOURCE).convert('RGB'))
    inverse,registration=register(rgb,marked)
    mask=(np.max(abs(marked.astype(np.int16)-np.array([21,96,130])),axis=2)<=3)
    mask[:900]=False;yy,xx=np.nonzero(mask)
    centers=[];pixels=[]
    for x in np.unique(xx):
        y=yy[xx==x];centers.append([float(x),float(np.median(y))]);pixels.append(np.c_[np.full(len(y),x),y].astype(float))
    screen_all=np.asarray(centers);uv_all=screen_all@inverse[:,:2].T+inverse[:,2]
    visible=np.all(uv_all>=0,axis=1)&np.all(uv_all<[1920,1080],axis=1);indices=np.flatnonzero(visible)
    screen=screen_all[visible];uv=uv_all[visible];normal,arc=normals(uv)
    spans=[];cells=[]
    for index,p,n in zip(indices,uv,normal):
        actual=pixels[int(index)];mapped=actual@inverse[:,:2].T+inverse[:,2];signed=(mapped-p)@n
        half=.5*np.sum(abs(n@inverse[:,:2]));spans.append([signed.min()-half,signed.max()+half])
        cells.append(dict(pixel_count=len(actual),screenshot_bbox_xyxy=np.r_[actual.min(0),actual.max(0)].tolist()))
    spans=np.asarray(spans);manual=6.;reg=registration['registration_uncertainty_native_px'];bounds=spans+np.array([-manual-reg,manual+reg])
    chosen,_=sparse_indices(uv);identifier='user_native230024192_A4_visible_arc_20261008';curve=dict(kind='hockey_circle',feature_id='A4')
    trace_path=BASE/'user_native230024192_A4_visible_arc_registered_stroke.json'
    write(trace_path,dict(schema='rinkspace.registered-user-source-stroke.v1',id=identifier,user_label='A4',physical_curve=curve,registration=registration,
        original_attachment=ref(saved),source_pts=230024192,time_base='1/15360',source_raw_frame=str(SOURCE),source_file_sha256=sha(SOURCE),
        user_stroke_raw_screenshot_px=screen.tolist(),user_stroke_raw_native_px=uv.tolist(),original_all_drawn_stroke_screenshot_px=screen_all.tolist(),
        all_drawn_native_coordinates_including_off_image=uv_all.tolist(),native_frame_visible_original_vertex_indices=indices.tolist(),
        off_image_original_vertex_indices=np.flatnonzero(~visible).tolist(),off_image_stroke_native_px=uv_all[~visible].tolist(),
        source_RGB_actual_measurement=False,direct_human_approximate_position_shape=True,partial_visible_arc_only=True,
        full_circle_or_hidden_arc_observation=False,physical_endpoint_or_corner_observation=False,circle_center_or_radius_as_image_measurement=False,
        normal_interval_px=bounds.tolist(),practical_manual_location_uncertainty_native_px=manual,practical_uncertainty_is_user_pixel_precision=False))
    parent_ice=BASE/f'ice_sources_{PREVIOUS}.json';ice=json.loads(parent_ice.read_text(encoding='utf8'));ice_path=BASE/f'ice_sources_{NEXT}.json'
    parent_authority=BASE/f'human_registered_hard_truth_authorities_{PREVIOUS}.json';authority=json.loads(parent_authority.read_text(encoding='utf8'))
    parent_guard=BASE/f'human_registered_full_stroke_hard_guard_{PREVIOUS}.json';guard=json.loads(parent_guard.read_text(encoding='utf8'))
    with np.load(guard['guard_arrays_reference']['path'],allow_pickle=False)as old:arrays={k:old[k].copy()for k in old.files}
    caps=json.loads((BASE/f'human_registered_stroke_end_2D_source_footprint_support_{PREVIOUS}.json').read_text(encoding='utf8'))
    item=dict(id=identifier,pts=230024192,actual_pts=230024192,feature_id='A4',curve=curve,split='fit',geometry_role='user_approximate_visible_A4_painted_arc_axis',
        uv=uv[chosen].tolist(),normal=normal[chosen].tolist(),normal_interval_px=bounds[chosen].tolist(),sigma_px=[manual]*len(chosen),
        source_optical_normal_support_authorized=False,source_raw_frame=str(SOURCE),source_file_sha256=sha(SOURCE),source_user_annotation=ref(saved),source_annotation_registration=ref(trace_path),
        user_label='A4',user_confirmed_source_axis=True,direct_human_GT=True,source_RGB_actual_measurement=False,model_projection_used_to_generate_or_select_uv=False,
        point_correspondence=False,curve_observation_kind='normal_interval',endpoint_observation=False,partial_visible_arc_only=True,full_circle_or_hidden_arc_observation=False,
        practical_manual_location_uncertainty_native_px=manual,practical_uncertainty_is_user_pixel_precision=False,
        physical_circle_producer_mapping=dict(existing_feature_id='A4',current_radius_parameter_index=3,side='left',across_sign=-1,center_or_radius_not_observed=True),
        sourceUV_origin='direct user partial visible A4 curve, inverse verified source-RGB display registration',
        complete_human_stroke=dict(reference=ref(trace_path),native_coordinates_field='user_stroke_raw_native_px',sample_count=len(uv)),
        sigma_semantics='6px manual engineering scale; drawn point/registration footprint included once, no pixel-exact physical precision')
    row_index=len(ice['rows']);ice['rows'].append(item);ice['user_A4_partial_visible_arc_append']=dict(parent=ref(parent_ice),source_pts=230024192,added_ids=[identifier],all_previous_rows_exact_prefix=True);write(ice_path,ice)
    record=dict(id=identifier,family='native_ice_curve',packet_reference=ref(ice_path),original_row_index=row_index,source_row_canonical_json_sha256=canonical_sha(item),
        pts=230024192,actual_pts=230024192,time_base='1/15360',source_video_sha256=SOURCE_SHA,curve=curve,geometry_role=item['geometry_role'],
        uv=item['uv'],normal=item['normal'],normal_interval_px=item['normal_interval_px'],lower_bound_px=bounds[chosen,0].tolist(),upper_bound_px=bounds[chosen,1].tolist(),
        direct_human_GT=True,direct_human_approximate_position_shape=True,model_or_RGB_transport_used_to_generate_UV=False,historical_sigma_used_as_hard_tolerance=False,
        source_raw_frame=str(SOURCE),source_file_sha256=sha(SOURCE),source_annotation_attachment=ref(saved),source_annotation_registration=ref(trace_path),endpoint_observation=False,
        complete_human_stroke=item['complete_human_stroke'],partial_visible_arc_only=True,practical_manual_location_uncertainty_native_px=manual,practical_uncertainty_is_user_pixel_precision=False)
    authority['records'].append(record);authority['A4_visible_arc_append']=dict(parent=ref(parent_authority),attachment=ref(saved),registration=registration);write(authority_path,authority)
    gi=len(guard['records']);prefix=f'record_{gi:02d}'
    for key,value in [('uv',uv),('normal',normal),('normal_interval_px',bounds),('annotation_stroke_normal_interval_px',spans),('screenshot_xy',screen),('source_arc_length_px',arc)]:arrays[prefix+'_'+key]=value
    guard['records'].append(dict(id=identifier,authority_record_index=gi,authority_source_row_canonical_json_sha256=canonical_sha(item),family='native_ice_curve',curve=curve,
        geometry_role=item['geometry_role'],pts=230024192,actual_pts=230024192,vertex_count=len(uv),array_prefix=prefix,source_raw_frame=str(SOURCE),source_file_sha256=sha(SOURCE),
        source_annotation_attachment=ref(saved),source_annotation_registration=ref(trace_path),full_trace_is_guard_only=True,dense_pixels_are_not_independent_FIT_residuals=True,
        endpoint_observation=False,partial_visible_arc_only=True))
    guard_arrays=BASE/f'human_registered_full_stroke_hard_guard_{NEXT}.npz';np.savez_compressed(guard_arrays,**arrays)
    guard_path=BASE/f'human_registered_full_stroke_hard_guard_{NEXT}.json';guard.update(primary_authority_reference=ref(authority_path),guard_arrays_reference=ref(guard_arrays),record_count=len(guard['records']),
        full_stroke_vertex_count=sum(r['vertex_count']for r in guard['records']),primary_UV_count=sum(len(r['uv'])for r in authority['records']));write(guard_path,guard)
    for vi,name in [(0,'first'),(len(uv)-1,'last')]:
        n=normal[vi];tangent=np.array([-n[1],n[0]]);radius=float(np.max(abs(bounds[vi])));cell=cells[vi]
        caps['records'].append(dict(id=identifier,source_full_guard_record_index=gi,source_vertex_index=vi,stroke_end=name,pts=230024192,actual_pts=230024192,uv=uv[vi].tolist(),normal=n.tolist(),tangent=tangent.tolist(),
            normal_cell_support_px=bounds[vi].tolist(),tangent_cell_support_px=[-radius,radius],source_annotation_attachment=ref(saved),source_annotation_registration=ref(trace_path),
            source_raw_frame=str(SOURCE),source_file_sha256=sha(SOURCE),selected_actual_coloured_pixel_count=cell['pixel_count'],selected_screenshot_bbox_xyxy=cell['screenshot_bbox_xyxy'],
            physical_endpoint_or_corner_observation=False,rectangular_marker_joint_not_free_cap=False,normal_band_from_frozen_guard_not_replaced=True,
            registration_uncertainty_native_px=reg,practical_manual_location_uncertainty_native_px=manual,source_frame_edge_or_drawing_end_is_not_physical_circle_endpoint=True))
    caps_path=BASE/f'human_registered_stroke_end_2D_source_footprint_support_{NEXT}.json';caps.update(primary_authority_reference=ref(authority_path),full_trace_reference=ref(guard_path),full_trace_arrays_reference=ref(guard_arrays),endpoint_record_count=len(caps['records']));write(caps_path,caps)
    previous_confirmed=BASE/f'user_confirmed_native_source_priority_authorities_{PREVIOUS}.json';confirmed=json.loads(previous_confirmed.read_text(encoding='utf8'))
    for r in confirmed['records']:
        if r['family']=='native_ice_curve':r['packet_reference_before_A4_append']=deepcopy(r['packet_reference']);r['packet_reference']=ref(ice_path)
    confirmed_path=BASE/f'user_confirmed_native_source_priority_authorities_{NEXT}.json';write(confirmed_path,confirmed)
    cfg_path=ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_resume_user230024192_20261008.json';cfg=json.loads(cfg_path.read_text(encoding='utf8'))
    previous_policy=Path(cfg['source_options']['--practical-human-location-envelope']);policy=json.loads(previous_policy.read_text(encoding='utf8'));native=[r for r in policy['records']if r['family']=='native'];wide=[r for r in policy['records']if r['family']=='wide']
    native.append(dict(family='native',id=identifier,full_vertices=len(uv),normal_expansion_px=0.,tangent_expansion_px=0.,engineering_uncorrelated_location_scale_px=manual,
        manual_location_uncertainty_native_px=manual,normal_and_tangent_manual_location_already_bound_in_source=True,recovery_uncertainty_is_not_additional_support_expansion=True,
        location_scale_is_not_a_user_accuracy_threshold=True,partial_visible_arc_only=True))
    policy['records']=native+wide;policy['authority_references']['native']=ref(authority_path);policy['previous_A4_policy_reference']=ref(previous_policy)
    policy_path=BASE/'fixed_practical_manual_location_envelope_full_user230024192_R1_R2_A4_v1.json';write(policy_path,policy)
    replacements={'--additional-ice-source':ice_path,'--direct-human-truth-authority':authority_path,'--direct-human-full-stroke-guard':guard_path,'--direct-human-stroke-cap-support':caps_path,'--user-confirmed-source-truth-authority':confirmed_path,'--practical-human-location-envelope':policy_path}
    for key,path in replacements.items():cfg['source_options'][key]=str(path);cfg['new_source_references'][key]=ref(path)
    added_scalars=2*len(chosen);cfg['new_source_scalar_rows']+=added_scalars;cfg['planned_source_rows']+=added_scalars;cfg['new_manual_stroke_groups']+=1
    receipt=dict(id=identifier,attachment=ref(saved),trace_reference=ref(trace_path),curve=curve,source_pts=230024192,partial_visible_arc_only=True,
        native_vertices=len(uv),FIT_queries=len(chosen),native_visible_bbox_xyxy=np.r_[uv.min(0),uv.max(0)].tolist(),manual_error_added_once=True,
        endpoints_have_no_physical_circle_endpoint_or_angle_meaning=True)
    cfg['latest_A4_partial_visible_arc_human_append']=receipt;write(cfg_path,cfg)
    fact_path=ROOT/'events/ankara_2026/model_constraints/user_native230024192_all_visible_20261008.json';fact=json.loads(fact_path.read_text(encoding='utf8'))
    fact['latest_A4_partial_visible_arc_user']=dict(**receipt,registration=registration,authority=ref(authority_path),guard=ref(guard_path),ice_packet=ref(ice_path),caps=ref(caps_path),confirmed_source_authority=ref(confirmed_path))
    fact['hard_truth_authority']=ref(authority_path);fact['full_stroke_guard']=ref(guard_path);fact['source_2D_coverage_support']=ref(caps_path);fact['ice_packet']=ref(ice_path);fact['confirmed_native_source_priority_authority']=ref(confirmed_path);fact['full_nine_shot_resume_inputs']['sha256']=sha(cfg_path)
    fact['A3_A4_direct_source_inventory']=dict(before_this_A4_drawing=dict(registered_human_A3_groups=0,registered_human_A4_groups=0,
        additional_native_ice_A3_FIT_groups=0,additional_native_ice_A3_HELD_groups=0,additional_native_ice_A4_FIT=[dict(id='native_A4_visible_red_near_arc_230011136',pts=230011136,points=5)],
        additional_native_ice_A4_HELD_groups=0,complete_original_programme_absolute_A3_groups=0,complete_original_programme_absolute_A4_groups=0),
        after_this_drawing=dict(registered_human_A3_groups=0,registered_human_A4_groups=1,A4_source_pts=230024192),model_arcs_are_not_source_measurements=True)
    write(fact_path,fact)
    print(json.dumps(dict(ice=str(ice_path),authority=str(authority_path),guard=str(guard_path),caps=str(caps_path),confirmed=str(confirmed_path),
        receipt=receipt,total_manual_groups=len(authority['records']),total_caps=len(caps['records']),planned_full_source_rows=cfg['planned_source_rows']),ensure_ascii=False))


if __name__=='__main__':main()
