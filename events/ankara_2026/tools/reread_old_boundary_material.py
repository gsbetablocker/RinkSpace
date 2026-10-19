"""Reread frozen old TOP/contact pixels from actual RGB; never change sources."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from rinkspace.vision_source_topology import (_original_RGB_profiles_on_CPU,
    _ordered_board_foot_profile_batches,read_source_board_rail_material_profiles,
    observed_signed_material_transition,source_graphic_rectangle_candidates,
    source_graphic_material_support_conflicts)
from events.ankara_2026.tools.read_registered_human_source_shots import NativeDecoder,normals,plain,sha,SOURCE_SHA,blue_material

def write(path,data):Path(path).write_text(json.dumps(plain(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def contact_material(rgb,binding,uv):
    uv=np.asarray(uv,float);direction=normals(uv);profiles=_original_RGB_profiles_on_CPU(rgb,uv,direction,48,binding)
    ordered=_ordered_board_foot_profile_batches(profiles,torch.zeros(rgb.shape[:2],dtype=torch.bool),np.full(len(uv),rgb.shape[0]*.25),None)
    rows=np.arange(len(uv));offset=ordered['selected_exit_offset_native_px'].numpy()
    lower=profiles['points_native_uv'].numpy()+profiles['normals_unit'].numpy()*offset[:,None]
    entry_offset=profiles['rgb_gradient_offsets_native_px'][ordered['entry_index']].numpy()
    upper=profiles['points_native_uv'].numpy()+profiles['normals_unit'].numpy()*entry_offset[:,None]
    unique=ordered['usable'].numpy()&~ordered['missing_same_band'].numpy()&(ordered['competing_same_band_pairs'].numpy()==0)
    graphics=source_graphic_rectangle_candidates(rgb,binding)
    graphic_support=source_graphic_material_support_conflicts(profiles,ordered,graphics)
    graphic_conflict=np.asarray(plain(graphic_support['selected_material_support_overlaps_source_graphic']),bool)
    for rectangle in graphics['candidates']:
        x0,y0,x1,y1=rectangle['box_xyxy']
        graphic_conflict|=((lower[:,0]>=x0)&(lower[:,0]<x1)&(lower[:,1]>=y0)&(lower[:,1]<y1))
        graphic_conflict|=((uv[:,0]>=x0)&(uv[:,0]<x1)&(uv[:,1]>=y0)&(uv[:,1]<y1))
    local_unique=unique.copy();unique &= ~graphic_conflict
    gamma=ordered['gamma'].numpy();light=ordered['native_RGB_luminance'].numpy()
    fall=plain(ordered['first_iceward_yellow_material_fall_evidence']);valid=profiles['valid_mask'].numpy()
    center=int(np.argmin(abs(profiles['offsets_native_px'].numpy())))
    intervals=np.asarray(fall['actual_warm_gray_80_to_20_fall_support_native_px'])
    classifications=[]
    for i,p in enumerate(uv):
        a=int(ordered['entry_index'][i]);gradient=np.diff(gamma[i])/.5
        support=observed_signed_material_transition(gamma[i],gradient,valid[i,:-1]&valid[i,1:],a,1)
        indices=support['source_indices'];entry_bounds=[float(profiles['rgb_gradient_offsets_native_px'][indices[0]]-.25),float(profiles['rgb_gradient_offsets_native_px'][indices[-1]]+.25)]if len(indices)else None
        outside=bool(np.any(p<0)or np.any(p>=[1920,1080]));unique[i]&=not outside
        lower_support=intervals[i];complete=bool(fall['actual_warm_gray_80_to_20_fall_support_complete'][i])
        at_upper=bool(unique[i]and entry_bounds and entry_bounds[0]<=0<=entry_bounds[1]and not(lower_support[0]<=0<=lower_support[1]))
        at_gray=bool(unique[i]and complete and lower_support[1]<0 and gamma[i,center]<=fall['actual_gray_side_gamma'][i]+.2*(fall['dominant_warm_peak_gamma'][i]-fall['actual_gray_side_gamma'][i]))
        gray_shadow=at_gray and bool(light[i,center]<np.nanmedian(light[i,-12:])-3/255)
        wrong=at_upper or at_gray
        reason=('old_lower_contact_label_on_observed_yellow_upper_interface'if at_upper else
            'old_contact_label_on_gray_shadow_after_resolved_yellow_fall'if gray_shadow else
            'old_contact_label_on_gray_surface_after_resolved_yellow_fall'if at_gray else
            'actual_RGB_graphic_role_conflict'if graphic_conflict[i]else'actual_material_role_ambiguous'if not unique[i]else'no_proven_source_role_error')
        classifications.append(dict(original_uv_index=i,original_uv=p,fresh_material_entry_uv=upper[i],
            fresh_first_yellow_gray_fall_uv=lower[i],fresh_edge_minus_old_uv=lower[i]-p,
            unique_actual_material_hypothesis=bool(unique[i]),local_material_disagreement_candidate=wrong,reason=reason,
            unique_local_warm_pair_before_RGB_graphic_review=bool(local_unique[i]),actual_RGB_graphic_role_conflict=bool(graphic_conflict[i]),
            ambiguous_material_quarantine_candidate=not bool(unique[i]),outframe=outside,
            entry_signed_optical_support_native_px=entry_bounds,first_fall_80_to_20_optical_support_native_px=lower_support,
            optical_blur_support_is_not_physical_GT=True,original_point_gamma=float(gamma[i,center]),
            original_point_luminance=float(light[i,center]),actual_gray_side_gamma=fall['actual_gray_side_gamma'][i],
            class_probability_used_to_rank_edge=False,world_facet_binding_association_pending=True,
            source_correction_admission_requires_main_review=True))
    return dict(classifications=classifications,fresh_material_entry_uv=upper,fresh_material_lower_uv=lower,
        unique_mask=unique,first_fall_evidence=fall,all_ordered_pair_indices=ordered['all_ordered_pair_indices_profile_entry_exit'],
        source_RGB_graphic_role_context=graphics,source_RGB_graphic_support_context=graphic_support,
        original_navigation_uv_not_truth=uv,normal_direction=direction)

def top_material(rgb,binding,uv):
    uv=np.asarray(uv,float);pending=np.arange(len(uv));records=[];layers=[]
    while len(pending):
        mat=blue_material(rgb,binding,uv[pending]);layers.append(mat);selected=np.asarray(mat['queried_original_indices'],int)
        measured={int(ev['source_query_index']):(entry,exit_,ev,bool(good))for entry,exit_,ev,good in zip(mat['entry_uv'],mat['internal_exit_uv'],mat['evidence'],mat['unique_entry_role'])}
        for q,local in enumerate(selected):
            i=int(pending[local]);actual=measured.get(q);outside=bool(np.any(uv[i]<0)or np.any(uv[i]>=[1920,1080]))
            unique=bool(actual and actual[3]and not outside);at_internal=False
            if actual:
                entry,exit_,ev,_=actual;eb=ev['entry_observed_same_signed_material_transition']['actual_native_normal_interval_px'];xb=ev['exit_observed_same_signed_material_transition']['actual_native_normal_interval_px']
                at_internal=bool(unique and xb[0]<=0<=xb[1]and not(eb[0]<=0<=eb[1]))
            records.append(dict(original_uv_index=i,original_uv=uv[i],fresh_material_entry_uv=actual[0]if actual else None,
                fresh_blue_white_internal_exit_uv=actual[1]if actual else None,
                fresh_edge_minus_old_uv=np.asarray(actual[0])-uv[i]if actual else None,
                unique_actual_material_hypothesis=unique,local_material_disagreement_candidate=at_internal,
                reason='old_TOP_label_on_blue_to_white_internal_interface'if at_internal else'actual_material_role_ambiguous'if not unique else'no_proven_source_role_error',
                ambiguous_material_quarantine_candidate=not unique,outframe=outside,
                actual_source_role_evidence=actual[2]if actual else None,
                world_facet_binding_association_pending=True,source_correction_admission_requires_main_review=True))
        pending=np.delete(pending,selected)
    records.sort(key=lambda x:x['original_uv_index']);return dict(classifications=records,material_layers=layers)

def development(out):
    base=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/observations/vision_training/continuous_review/native_cut_pipeline/user_ice_contact_correction')
    reports=[]
    for filename,uvfield,imagefield,hashfield,pts in [('user_red_contact_uv.json','red_line_uv','frame_path','frame_sha256',36208896),
        ('user_iceedge2_contact_crop_source_uv.json',None,'source_frame_path','source_frame_SHA256',228372480)]:
        source=base/filename;record=json.loads(source.read_text(encoding='utf8'))
        uv=np.asarray(record[uvfield]if uvfield else record['coordinate_mapping']['native_210_column_uv'],float)[::8]
        if sha(record[imagefield])!=record[hashfield]:raise ValueError('Development source image changed')
        rgb=np.asarray(Image.open(record[imagefield]).convert('RGB'));binding=dict(source_sha256=SOURCE_SHA,source_pts=pts,time_base='1/15360',native_hw=[1080,1920],shot_id='development_reference_only_'+str(pts))
        result=contact_material(rgb,binding,uv);mask=result['unique_mask'];delta=((result['fresh_material_lower_uv']-uv)*result['normal_direction']).sum(1)
        for point in result['classifications']:
            point['automatic_material_disagreement_reason']=point['reason']
            point['reason']='development_operator_comparison_only; direct_human_authority_unchanged'
            point['local_material_disagreement_candidate']=False
            point['human_authority_must_not_be_retired_from_this_diagnostic']=True
        reports.append(dict(source_pts=pts,human_source=str(source),human_source_sha256=sha(source),source_frame_path=record[imagefield],
            source_frame_sha256=record[hashfield],development_and_previously_trained_not_independent_validation=True,
            input_points=len(uv),unique_material_hypotheses=int(mask.sum()),
            unique_median_human_normal_distance_px=float(np.median(abs(delta[mask])))if mask.any()else None,
            unique_maximum_human_normal_distance_px=float(np.max(abs(delta[mask])))if mask.any()else None,
            development_GT_2p5px_disagreement_indices=np.flatnonzero(mask&(abs(delta)>2.5)),
            raw_human_uv_changed=False,reading=result))
    write(out/'previously_trained_human_development_regression.json',reports)
    print(json.dumps([dict(source_pts=r['source_pts'],input_points=r['input_points'],unique=r['unique_material_hypotheses'],median=r['unique_median_human_normal_distance_px'],maximum=r['unique_maximum_human_normal_distance_px'])for r in reports]),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--old-row-index',type=Path,required=True);p.add_argument('--output',type=Path,required=True);o=p.parse_args()
    if o.output.exists()or not o.output.resolve().is_relative_to(Path('D:/CodexData/RinkSpace_Data')):raise ValueError('New Data output required')
    o.output.mkdir(parents=True);torch.set_num_threads(2);start=time.perf_counter();development(o.output)
    frozen=json.loads(o.old_row_index.read_text(encoding='utf8'))
    if sha(o.old_row_index)!='aef268b0f9b046c7644f665eff32dc769fbf6237a798ab91a6bd5dc46833000f':raise ValueError('Frozen index differs')
    groups=defaultdict(list)
    for row in frozen['records']:groups[(row['shot'],int(row['group_actual_pts']))].append(row)
    decoder=NativeDecoder('D:/CodexData/RinkSpace_Data/events/ankara_2026/media/originals/oxXu1avhrvs.mp4');records=[];counts=Counter()
    try:
        for (shot,pts),rows in sorted(groups.items()):
            rgb=decoder.exact(pts);binding=dict(source_sha256=SOURCE_SHA,source_pts=pts,time_base='1/15360',native_hw=[1080,1920],shot_id=shot)
            for row in rows:
                role=row['semantic_role'];uv=np.asarray(row['original_uv'],dtype='<f8')
                if hashlib.sha256(uv.tobytes()).hexdigest()!=row['original_full_uv_float64_sha256']:raise ValueError('Original UV digest differs')
                if role not in('board_top','ice_contact','ice_board_contact'):continue
                result=top_material(rgb,binding,uv)if role=='board_top'else contact_material(rgb,binding,uv)
                if [v['original_uv_index']for v in result['classifications']]!=list(range(len(uv))):raise ValueError('Incomplete original query')
                rec=dict(original_source=row,actual_source_binding=binding,actual_decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),reading=result)
                records.append(rec);counts['rows']+=1;counts['points']+=len(uv)
                for v in result['classifications']:
                    counts['local_material_disagreement_candidates']+=int(v['local_material_disagreement_candidate'])
                    counts['ambiguous_material_quarantine_candidates']+=int(v['ambiguous_material_quarantine_candidate'])
                    counts['unique_material_hypotheses']+=int(v['unique_actual_material_hypothesis'])
            counts['actual_exposures_RGB_read']+=1
            if counts['actual_exposures_RGB_read']%10==0:print(json.dumps(dict(stage='old_actual_RGB_reread',actual=counts['actual_exposures_RGB_read'],expected=len(groups),pts=pts,counts=dict(counts))),flush=True)
        write(o.output/'old_actual_material_role_review_index_v2.json',dict(schema='rinkspace.actual-old-source-material-role-review.v2',
            original_index_path=str(o.old_row_index),original_index_sha256=sha(o.old_row_index),records=records,
            original_source_packets_changed=False,source_active_admission=False,world_binding_review_separate=True))
        write(o.output/'receipt.json',dict(status='complete_actual_RGB_reread',counts=dict(counts),expected_actual_exposures=len(groups),
            code_sha256={str(path):sha(path)for path in (Path(__file__),ROOT/'src/rinkspace/vision_source_topology.py')},
            no_new_NN_forward_or_training=True,no_camera_or_model_projection_navigation=True,no_human_UV_changed=True,
            no_source_sigma_or_roles_changed=True,local_material_disagreement_is_not_proven_source_error=True,
            wall_seconds=time.perf_counter()-start))
        print(json.dumps(dict(status='complete_actual_RGB_reread',counts=dict(counts),wall_seconds=time.perf_counter()-start)),flush=True)
    finally:decoder.close()

if __name__=='__main__':main()
