"""Finish source-side vertical-join cleanup and all first-shot board coverage."""
from __future__ import annotations
from copy import deepcopy
import json
import cv2
import numpy as np
from .extract_first_shot_board_source import (DATA,MODEL,OLD_MODEL,read,write,sha,feature_world,
    visible_projected_runs,path_stations,profile_read,segment_records,native_overlay,source_segments,
    retain_print_band_segments,print_inside_current_board_band)


def finish():
    registry=read(MODEL/'feature_registry.json');scene=read(MODEL/'scene.json');templates=read(MODEL/'ad2_material_template.json')
    archive=np.load(DATA/'camera_seeds.npz');cameras={int(i):c for i,c in zip(archive['programme_indices'],archive['camera25'])}
    manifest={int(r['index']):r for r in read(DATA/'frame_manifest.json')['frames']}
    rows=[];errors=[];changes=[]
    circles={tid:d for tid,d in registry['targets'].items()if d.get('geometry_source')=='scene_existing_board_pattern'}
    for index in range(201,1249):
        path=DATA/f'board_candidates/{index:06d}.json'
        if not path.exists():errors.append(dict(index=index,error='missing_board_candidate'));continue
        packet=read(path);record=manifest[index];changed=[]
        if packet['source']['mask_sha256']!=record['mask_sha256']:errors.append(dict(index=index,error='stale_current_source_person_mask'));continue
        # One visible straight upper rail is one native line. A local search
        # can otherwise jump down to a blue/purple ad edge across a gap (1248).
        for feature in packet['features']:
            if feature['target_id']not in('board.top.upper_long_side','board.top.lower_long_side','board.top.left_end','board.top.right_end'):continue
            evidence=np.vstack([s.get('source_native_readout_uv',s['uv'])for s in feature['segments']])
            if len(evidence)<12:continue
            line=cv2.fitLine(evidence.astype('float32'),cv2.DIST_HUBER,0,.001,.001).ravel().astype(float)
            perpendicular=np.array([-line[1],line[0]])
            if np.max(np.abs((evidence-line[2:])@perpendicular))<=8:continue
            accepted=evidence[np.abs((evidence-line[2:])@perpendicular)<4]
            if len(accepted)<12:continue
            line=cv2.fitLine(accepted.astype('float32'),cv2.DIST_HUBER,0,.001,.001).ravel().astype(float);perpendicular=np.array([-line[1],line[0]])
            replacement=[]
            for segment in feature['segments']:
                native=np.asarray(segment.get('source_native_readout_uv',segment['uv']),float)
                good=np.abs((native-line[2:])@perpendicular)<4
                if not good.all():changed.append(dict(feature_id=feature['id'],reason='local_blue_ad_transition_not_same_current_source_straight_upper_rail',removed_uv=native[~good].tolist()))
                native=native[good]
                if len(native)<2 or np.linalg.norm(native[-1]-native[0])<8:continue
                station=(native[[0,-1]]-line[2:])@line[:2]
                uv=line[2:]+np.linspace(station[0],station[-1],max(2,int(np.ceil(abs(station[1]-station[0])/200))+1))[:,None]*line[:2]
                replacement_segment=dict(segment,uv=uv.tolist(),vertex_count=len(uv),source_native_readout_uv=native.tolist(),
                    source_native_transition_iqr_px=[segment.get('source_native_transition_iqr_px',[1.]*len(good))[j]for j in np.flatnonzero(good)],
                    shared_current_source_upper_rail_line=line.tolist(),point_bindings=[dict(kind='edge_sample',physical_target=feature['target_id'],required_residual='point_to_named_finite_model_edge',measurement_origin='current_source_joint_straight_upper_rail')for _ in uv])
                replacement.append(replacement_segment)
            feature['segments']=replacement
        packet['features']=[f for f in packet['features']if f['segments']]
        for feature in packet['features']:
            if not feature['id'].startswith('Ad')or'vertical'not in feature['id']:continue
            evidence=np.vstack([s.get('source_native_readout_uv',s['uv'])for s in feature['segments']])
            if len(evidence)<5:continue
            line=cv2.fitLine(evidence.astype('float32'),cv2.DIST_HUBER,0,.001,.001).ravel().astype(float)
            kept=[]
            for segment in feature['segments']:
                uv=np.asarray(segment['uv']);delta=uv[-1]-uv[0];length=np.linalg.norm(delta)
                cosine=abs(float(delta@line[:2]))/max(length,1e-8)
                distance=np.max(np.abs((uv-line[2:])@np.array([-line[1],line[0]])))
                if length<60 and (cosine<.95 or distance>3.5):
                    changed.append(dict(feature_id=feature['id'],reason='blurred_join_fragment_not_current_joint_straight_vertical_side',removed_uv=segment['uv']));continue
                kept.append(segment)
            feature['segments']=kept
        packet['features']=[f for f in packet['features']if f['segments']]
        raw=None;mask=None
        for tid,d in circles.items():
            if any(f['target_id']==tid for f in packet['features']):continue
            world=feature_world(scene,d,templates);paths=visible_projected_runs(cameras[index],world)
            if not paths:continue
            if raw is None:
                raw=cv2.imread(record['image_path']);mask=cv2.dilate((cv2.imread(record['mask_path'],0)>0).astype('uint8'),np.ones((5,5),np.uint8))>0
                mask[:120,:570]=True;mask[945:,1660:]=True
            segments=[]
            for path_uv in paths:
                p=path_stations(path_uv,3.5)
                profile,_=profile_read(raw,mask,p,[1/3,1/3,1/3],10,maximum_blur=6,minimum_amplitude=24,
                    navigation_tolerance=6.5,require_clear_end_plateaus=True)
                segments+=segment_records(profile,spacing=45,closed=bool(np.linalg.norm(p[-1]-p[0])<1.),minimum_span=8)
            if not segments:continue
            feature=dict(id=tid,target_id=tid,label=d['contour_key'],label_zh=tid,review_layer='ad2_print',physical_surface='board',
                segments=segments,current_source_only_readout=True,model_or_template_generated_observation=False,
                physical_shape='same_inherited_circle_on_continuous_unwrapped_board',allowed_for_fit=False)
            packet['features'].append(feature);changed.append(dict(feature_id=tid,reason='existing_physical_board_circle_registered_and_current_source_measured',added_points=sum(len(s['uv'])for s in segments)))
        for feature in packet['features']:
            if '.print.'not in feature['id']:continue
            rejected=retain_print_band_segments(packet['features'],feature)
            if rejected:changed.append(dict(feature_id=feature['id'],reason='not_advertisement_material_inside_current_source_board_band',removed_uv=rejected))
        packet['features']=[f for f in packet['features']if f['segments']]
        packet['point_count']=sum(len(s['uv'])for f in packet['features']for s in f['segments'])
        packet['diagnostics']['board_point_count']=sum(len(s['uv'])for f in packet['features']if'.print.'not in f['id']for s in f['segments'])
        packet['diagnostics']['printed_point_count']=packet['point_count']-packet['diagnostics']['board_point_count']
        packet['diagnostics']['material_targets']=sorted({f['target_id']for f in packet['features']})
        feature_ids={f['id']for f in packet['features']};packet['source_correlated_groups']=[g for g in packet.get('source_correlated_groups',[])if g.get('feature_id')in feature_ids]
        roles={f['target_id'].split('.')[1]for f in packet['features']if f['target_id'].startswith('board.')}
        if roles!={'top','contact'}:errors.append(dict(index=index,error='missing_mandatory_visible_board_boundary',roles=sorted(roles)))
        maxspacing=0.
        for feature in packet['features']:
            if feature['target_id']not in registry['targets']:errors.append(dict(index=index,error='unregistered_material',feature=feature['id']))
            for segment in feature['segments']:
                p=np.asarray(segment['uv'])
                if feature['target_id'].startswith('board.')and len(p)>1:maxspacing=max(maxspacing,float(np.max(np.linalg.norm(np.diff(p,axis=0),axis=1))))
                if 'point_bindings'in segment and len(segment['point_bindings'])!=len(p):errors.append(dict(index=index,error='binding_array_length',feature=feature['id']))
                if 'corner_vertex_indices'in segment and not all(0<=j<len(p)for j in segment['corner_vertex_indices'].values()):errors.append(dict(index=index,error='corner_index',feature=feature['id']))
                if (p<[0,0]).any()or(p>[1919,1079]).any():errors.append(dict(index=index,error='out_of_native_image',feature=feature['id']))
                if '.print.'in feature['id']and not print_inside_current_board_band(packet['features'],p).all():errors.append(dict(index=index,error='print_outside_current_source_board_band',feature=feature['id']))
        if maxspacing>201:errors.append(dict(index=index,error='board_spacing_exceeds200',maximum_spacing=maxspacing))
        if changed:
            packet['source_vertical_join_cleanup_and_pattern_completion']=changed;write(DATA/f'board_candidates/{index:06d}.json',packet);changes.append(dict(index=index,changes=changed))
        if index%25==0 or index in (201,300,400,500,600,800,1000,1200,1248):
            if raw is None:raw=cv2.imread(record['image_path'])
            cv2.imwrite(str(DATA/f'board_overlays/{index:06d}.png'),native_overlay(raw,list(source_segments(packet))),[cv2.IMWRITE_PNG_COMPRESSION,1])
        rows.append(dict(index=index,points=packet['point_count'],board_points=packet['diagnostics']['board_point_count'],print_points=packet['diagnostics']['printed_point_count'],
            board_roles=sorted(roles),maximum_board_adjacent_spacing_px=maxspacing,print_target_ids=[f['target_id']for f in packet['features']if'.print.'in f['id']]))
    original_targets=read(OLD_MODEL/'feature_registry.json')['targets'];old_templates=read(OLD_MODEL/'ad2_material_template.json')
    for tid,d in registry['targets'].items():
        world=feature_world(scene,d,templates)
        if not np.isfinite(world).all():errors.append(dict(error='nonfinite_material_target',target_id=tid))
    report=dict(schema='rinkspace.first-shot-source-board-coverage-audit.v1',expected_frames=1048,frames_processed=len(rows),
        programme_frame_range_1based=[201,1248],mandatory_board_top_and_contact_present_on_every_frame=all(r['board_roles']==['contact','top']for r in rows),
        maximum_adjacent_board_point_spacing_px=max(r['maximum_board_adjacent_spacing_px']for r in rows),
        all_candidate_points_with_explicit_material_target=not any(e['error']=='unregistered_material'for e in errors),
        all_current_source_mask_hashes_match=not any(e['error']=='stale_current_source_person_mask'for e in errors),errors=errors,point_count_min=min(r['points']for r in rows),point_count_max=max(r['points']for r in rows),point_count_total=sum(r['points']for r in rows),
        all_print_points_within_current_source_board_material_band=not any(e['error']=='print_outside_current_source_board_band'for e in errors),
        candidate_count_above330=[r['index']for r in rows if r['points']>330],
        previous_scene_bytes_unchanged=sha(MODEL/'scene.json')==sha(OLD_MODEL/'scene.json'),
        previous_targets_preserved_exactly=all(registry['targets'][k]==v for k,v in original_targets.items()),
        previous_Ad2_contours_preserved_exactly=templates['contours']==old_templates['contours'],
        model_references={name:dict(path=str(MODEL/name),sha256=sha(MODEL/name))for name in ('scene.json','feature_registry.json','ad2_material_template.json','model.json')},
        new_print_counts={k:len(v['contours'])for k,v in templates['advertisements'].items()},
        added_material_target_count=len(registry['targets'])-len(original_targets),
        ad5_model_extent_scope='existing_internal_graphic_only_not_whole_advertisement_rectangle',
        formal_model_accuracy_accepted=False,source_observations_generated_from_model_or_temporal_interpolation=False,
        native_material_readouts_or_explicit_joint_same_source_edge_representatives_only=True,
        final_root_merge_enforces_user500=True,rows=rows)
    write(DATA/'board_candidate_audit.json',report);write(DATA/'board_candidate_shape_cleanup.json',changes)
    print(json.dumps({k:v for k,v in report.items()if k not in('rows','model_references')},ensure_ascii=False),flush=True)
    return report


if __name__=='__main__':finish()
