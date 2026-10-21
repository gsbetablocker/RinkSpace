"""Read local actual board interfaces using authenticated print as search context.

Source-to-source transforms carry only search queries. Output means always come
from a current native RGB signed material profile and the frozen cubic49 reader.
"""
from pathlib import Path
import hashlib
import numpy as np
import torch
import cv2
from .io_utils import read_json,file_sha256
from .outer_cubic_source import prepare_cubic_channels,replay_outer_cubic,CONFIG as CUBIC_CONFIG
from .board_source_requalification_v2 import _np
from rinkspace.vision_source_topology import (read_source_board_rail_material_profiles,
    read_board_foot_material_profiles,board_foot_material_authority)

CONFIG=dict(version='source_print_context_local_actual_material_profiles_v1',source_context_is_navigation_only=True,profile_radius_native_px=128.,existing_profile_navigation_spacing_native_px=8.,existing_cubic_source_column_spacing_native_px=16.,unknown_noise_not_1px=True)

def read_printed_board_native(rgb,mask,receipt,contexts,*,project,evidence_directory):
    project=Path(project);image=np.asarray(rgb);mask=np.asarray(mask,bool);h,w=mask.shape
    if image.shape!=(h,w,3)or image.dtype!=np.uint8 or hashlib.sha256(image.tobytes()).hexdigest()!=receipt['decoded_native_RGB_sha256']:raise ValueError('Actual native print readout RGB differs')
    if receipt['time_base']!='1/12800' or receipt['source_video_sha256']!='e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d':raise ValueError('Wrong native programme processing source')
    binding=dict(source_sha256=receipt['source_video_sha256'],source_pts=receipt['native_pts'],time_base='1/12800',shot_id=receipt['shot_id'],native_hw=[h,w]);cap=[];lower=[]
    for pc in contexts:
        if pc['actual_native_RGB_sha256']!=receipt['decoded_native_RGB_sha256']or pc['native_pts']!=receipt['native_pts']or pc['shot_id']!=receipt['shot_id']or not pc['current_source_entity_identity_confirmed']:raise ValueError('Print context from another RGB or unresolved entity')
        source=next(r for r in pc['source_context_references']if 'source_datasets' in r.get('file','')and r['file'].endswith('.json'))
        if file_sha256(source['file'])!=source['sha256']:raise ValueError('Reviewed source feature context changed')
        f=read_json(source['file']);number=pc['physical_board_instance'];byid={q['id']:q for q in f['features']};tt=np.concatenate([np.asarray(s['uv'],float)for s in byid[number+'.top_edge']['segments']]);bb=np.concatenate([np.asarray(s['uv'],float)for s in byid[number+'.bottom_edge']['segments']]);bounds=np.r_[np.minimum(tt.min(0),bb.min(0)),np.maximum(tt.max(0),bb.max(0))];lo,hi=bounds[:2],bounds[2:];poly=np.array([lo,[hi[0],lo[1]],hi,[lo[0],hi[1]]],np.float32);H=cv2.getPerspectiveTransform(poly,np.asarray(pc['current_source_print_body_polygon_uv'],np.float32))
        foot=np.concatenate([np.asarray(s.get('source_native_readout_uv',s['uv']),float)for q in f['features']if q['id'].startswith('board_contact.')for s in q['segments']]);foot=foot[(foot[:,0]>=lo[0])&(foot[:,0]<=hi[0])]
        mapped=cv2.perspectiveTransform(foot.astype(np.float32)[None],H)[0]if len(foot)else foot
        lower.extend(mapped.tolist());cap.extend(pc['current_source_cap_search_navigation_uv'])
    instances=[dict(object_id='same_exposure_person_graphics',kind='occluder',region_mask=torch.as_tensor(mask),source_binding=binding)]
    cap=np.asarray(cap,float).reshape(-1,2);lower=np.asarray(lower,float).reshape(-1,2)
    if not len(cap)or not len(lower):return [],dict(status='unknown_missing_authenticated_current_print_navigation',all_candidates=[],source_noise_W_or_precision_issued=False)
    # Existing 8px profile spacing is a search carrier, never an emitted UV.
    def source_queries(points):
        points=points[np.isfinite(points).all(1)];points=points[(points[:,0]>=2)&(points[:,0]<=w-3)&(points[:,1]>=2)&(points[:,1]<=h-3)];xs=np.arange(2,w-2,CONFIG['existing_profile_navigation_spacing_native_px']);queries=[]
        for x in xs:
            near=points[abs(points[:,0]-x)<=8]
            if len(near):queries.append(near[np.argmin(abs(near[:,0]-x))])
        return np.asarray(queries,float).reshape(-1,2)
    cap=source_queries(cap);lower=source_queries(lower)
    rail=read_source_board_rail_material_profiles(image,dict(source_binding=binding,RGB_sha256=receipt['decoded_native_RGB_sha256'],actual_source_top_uv=cap),source_instances=instances,sample_spacing_px=1)
    authority=board_foot_material_authority(project/'events/ankara_2026/model_constraints/ice_contact_material_roles_user_confirmed_20261006.json');authority=dict(authority,original_confirmation_source_sha256=authority['source_sha256'],source_sha256=binding['source_sha256'],same_event_material_fact_application=True,original_annotation_pixels_transferred=False)
    foot=read_board_foot_material_profiles(image,dict(source_binding=binding,instances=instances),dict(source_binding=binding),material_role_authority=authority,device='cpu',search_radius_px=CONFIG['profile_radius_native_px'],sample_spacing_px=CONFIG['existing_profile_navigation_spacing_native_px'],source_query_points_uv=lower)
    candidates=[]
    for e in rail['source_material_role_evidence']:
        if e['competing_current_source_person_support']or e['selected_optical_peak_at_profile_search_boundary']:continue
        qi=e['source_query_index'];candidates.append(('board.top',np.asarray(e['actual_source_material_entry_uv'],float),np.asarray(rail['actual_original_RGB_profiles']['normals_unit'][qi],float),dict(current_blue_transition=e['entry_observed_same_signed_material_transition'],source_profile_query_index=qi)))
    pp=foot['original_RGB_profiles'];reading=foot['readings'][0];ids=np.flatnonzero(_np(reading['primary_role_samples']))
    for qi in ids:candidates.append(('board.contact',_np(reading['candidate_source_uv'][qi]).astype(float),_np(pp['normals_unit'][qi]).astype(float),dict(source_profile_query_index=int(qi),current_yellow_first_iceward_fall_support=_np(foot['source_material_transition_support']['observed_native_warm_colour_fall_support_endpoints_uv'][qi]).tolist())))
    # Retain one actual already-read query per historical 16px source cell.
    # This does not interpolate missing pixels or impose an observation count.
    chosen={}
    for role,seed,normal,evidence in candidates:
        key=(role,int(seed[0]//CONFIG['existing_cubic_source_column_spacing_native_px']))
        if key not in chosen or abs(seed[0]-(key[1]+.5)*16)<abs(chosen[key][1][0]-(key[1]+.5)*16):chosen[key]=(role,seed,normal,evidence)
    coeff=prepare_cubic_channels(image);rows=[];failed=[]
    for i,(role,seed,normal,evidence)in enumerate(sorted(chosen.values(),key=lambda q:(q[0],q[1][0]))):
        nav=dict(navigation_seed_uv=seed.tolist(),normal=normal.tolist(),source_measurement_audit=dict(profile=dict(channel=2,polarity=1)));a=replay_outer_cubic(image,nav,coefficients=coeff)
        if not a['numeric_valid']or a['synthetic_window_boundary_count']:
            failed.append(dict(material_role=role,actual_RGB_seed=seed.tolist(),cubic=a));continue
        uv=np.asarray(a['current_uv']);xy=np.rint(uv).astype(int)
        if not(0<=xy[0]<w and 0<=xy[1]<h)or mask[xy[1],xy[0]]:
            failed.append(dict(material_role=role,actual_RGB_seed=seed.tolist(),reason='source_pixel_person_graphic_or_native_domain_exclusion'));continue
        profile=dict(channel=2,channel_definition='B-(R+G)/2',polarity=1,normal_offsets=a['actual_profile_normal_offsets'],intensity=a['actual_profile_intensity'],peak_gradient=a['signed_peak_gradient'],offset_px=a['normal_offset_px'],interpolation='continuous_cubic',valid=True,configuration=CUBIC_CONFIG)
        rows.append(dict(source_row_id='board_print_native/'+receipt['shot_id']+'/'+str(receipt['native_pts'])+'/'+role+'/'+str(i),native_pts=receipt['native_pts'],shot_id=receipt['shot_id'],time_base='1/12800',material_role=role,target_id=role,measurement_operator='board_upper_material_edge'if role=='board.top'else'ice_board_interface',measurement_operator_resolved='board_upper_material_edge'if role=='board.top'else'ice_board_interface',navigation_seed_uv=seed.tolist(),normal=normal.tolist(),current_uv=uv.tolist(),source_uv=uv.tolist(),source_measurement_audit=dict(current_uv=uv.tolist(),normal=normal.tolist(),profile=profile,continuous_cubic_material_profile=a,source_RGB_material_evidence=evidence,current_mean_from_actual_material_profile_not_context_H=True,source_noise_W_or_precision_issued=False)))
    return rows,dict(schema='rinkfit.actual_print_context_board_material_readout.v1',configuration=CONFIG,source_binding=binding,actual_native_RGB_sha256=receipt['decoded_native_RGB_sha256'],actual_blue_profile_queries=len(cap),actual_yellow_profile_queries=len(lower),actual_optical_candidates=len(candidates),selected_actual_query_cells=len(chosen),failed_candidate_count=len(failed),failed_candidates=failed,all_raw_optical_candidate_denominators_retained=True,actual_top_all_role_reviews=rail['source_material_role_evidence'],actual_top_missing_role_queries=rail['missing_or_competing_source_roles'],actual_foot_all_query_candidate_UV=_np(reading['candidate_source_uv']),actual_foot_all_query_primary=reading['primary_role_samples'].cpu().numpy(),current_print_contexts=contexts,source_H_or_world_UV_emitted_as_measurement=False,source_noise_W_or_precision_issued=False)
