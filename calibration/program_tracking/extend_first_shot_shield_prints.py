"""Register clear Ad12 source print on its existing continuous shield chart."""
from __future__ import annotations
from copy import deepcopy
import cv2
import numpy as np
from .extract_first_shot_board_source import (DATA,MODEL,OLD_MODEL,read,write,sha,feature_world,
    path_stations,profile_read,segment_records,visible_projected_runs,retain_print_band_segments,
    native_overlay,source_segments)
from .projection import project_fast

ROLES=('upper_left_printed_outline_corner','upper_right_printed_outline_corner','bottom_printed_outline_tip')


def register():
    scene=read(MODEL/'scene.json');registry=read(MODEL/'feature_registry.json');templates=read(MODEL/'ad2_material_template.json')
    source=DATA/'frames/000850.png';raw=cv2.imread(str(source));luma=raw.mean(2)
    support=np.zeros((1080,1920),bool);support[512:650,76:275]=True
    ink=(luma<145)&support
    contours,hierarchy=cv2.findContours(ink.astype('uint8')*255,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    items=[]
    for j,c in enumerate(contours):
        x,y,w,h=cv2.boundingRect(c);area=abs(cv2.contourArea(c))
        if area<5 or w<2 or h<7:continue
        hole=bool(hierarchy[0,j,3]>=0)
        if hole:continue  # Tiny emblem detail is not required to lock the clear outer material.
        p=c[:,0]
        if any(not support[np.clip(p[:,1]+dy,0,1079),np.clip(p[:,0]+dx,0,1919)].all()for dx,dy in ((1,0),(-1,0),(0,1),(0,-1))):continue
        approx=cv2.approxPolyDP(c,.55,True)[:,0].astype(float)
        if len(approx)<3:continue
        items.append((x,y,w,h,area,approx))
    shield=max(items,key=lambda row:row[4])
    p=shield[-1];top=p[p[:,1]<p[:,1].min()+4]
    anchors=np.float32([top[np.argmin(top[:,0])],top[np.argmax(top[:,0])],p[np.argmax(p[:,1])]])
    basis=np.column_stack((anchors,np.ones(3)))
    records={}
    for j,(x,y,w,h,area,p)in enumerate(sorted(items,key=lambda row:(row[0],row[1]))):
        key='shield_print_component'+f'{j:03d}';closed=np.vstack((p,p[:1]))
        weights=np.column_stack((closed,np.ones(len(closed))))@np.linalg.inv(basis)
        records[key]=dict(id=key,advertisement_id=12,closed=True,
            role='printed_shield_outer_boundary'if area==shield[4]else'printed_letter_external_boundary',
            source_native_uv=closed.tolist(),source_native_bbox_xywh=[x,y,w,h],
            material_anchor_weights=weights.tolist(),material_coordinate_domain='continuous_board_station_height_from_existing_shield_three_anchors',
            source_native_boundary_uncertainty_px=2.,initialization_accuracy_accepted=False)
        registry['targets']['Ad12.print.'+key]=dict(kind='ad_print',advertisement_id=12,contour_key=key,
            geometry_source='scene_shield_anchor_material_chart',contour_role=records[key]['role'],
            residual='point_to_named_finite_print_contour',template='ad2_material_template.json',
            material_coordinate_domain=records[key]['material_coordinate_domain'],geometry_status='current_source_material_initializer_accuracy_not_accepted')
    templates['advertisements']['12']=dict(advertisement_number=12,contours=records,
        source=dict(path=str(source),sha256=sha(source),programme_frame_index_1based=850),
        source_shield_anchor_native_uv=anchors.tolist(),existing_model_anchor_roles=list(ROLES),
        source_anchor_pixels_are_current_source_material_initializer=True,
        physical_identity_authority='historical_user_Ad12_shield_and_Ankara_text_annotation',
        physical_chart='same_continuous_board_station_height_as_existing_shield12',
        planar_3d_rectangle_required=False,formal_accuracy_accepted=False)
    model_id='ankara_2026.framewise_8a28_first_shot_20261014'
    registry.update(model_id=model_id,created_date_local='2026-10-14',parent_model_id=read(OLD_MODEL/'model.json').get('model_id'),
        parent_feature_registry=dict(path=str(OLD_MODEL/'feature_registry.json'),sha256=sha(OLD_MODEL/'feature_registry.json')))
    templates.update(model_id=model_id,created_date_local='2026-10-14',parent_material_template=dict(path=str(OLD_MODEL/'ad2_material_template.json'),sha256=sha(OLD_MODEL/'ad2_material_template.json')))
    write(MODEL/'feature_registry.json',registry);write(MODEL/'ad2_material_template.json',templates)
    manifest=read(MODEL/'model.json');manifest.update(model_id=model_id,created_date_local='2026-10-14')
    manifest['feature_registry']['sha256']=sha(MODEL/'feature_registry.json');manifest['material_templates']['sha256']=sha(MODEL/'ad2_material_template.json');write(MODEL/'model.json',manifest)
    for target in registry['targets'].values():
        if not np.isfinite(feature_world(scene,target,templates)).all():raise ValueError('Nonfinite named print material')
    print('Ad12 material components',len(records),'current source anchors',anchors.tolist(),flush=True)


def complete():
    scene=read(MODEL/'scene.json');registry=read(MODEL/'feature_registry.json');templates=read(MODEL/'ad2_material_template.json')
    archive=np.load(DATA/'camera_seeds.npz');cameras={int(i):c for i,c in zip(archive['programme_indices'],archive['camera25'])}
    manifest={int(r['index']):r for r in read(DATA/'frame_manifest.json')['frames']}
    targets={tid:d for tid,d in registry['targets'].items()if d.get('geometry_source')=='scene_shield_anchor_material_chart'}
    source_contours=templates['advertisements']['12']['contours'];source_raw=cv2.imread(str(DATA/'frames/000850.png'))
    anchor_native=np.float32(templates['advertisements']['12']['source_shield_anchor_native_uv'])
    anchor_world=np.array([scene['shared_board_patterns']['shield_fixed_material_points_xyz_scene_units']['12'][role]for role in ROLES])
    rows=[]
    for index in range(201,1249):
        packet=read(DATA/f'board_candidates/{index:06d}.json');packet['features']=[f for f in packet['features']if not f['target_id'].startswith('Ad12.print.')]
        camera=cameras[index];projected,depth=project_fast(camera,anchor_world)
        if (depth<=0).any()or projected[:,0].max()<-30 or projected[:,0].min()>1950:continue
        record=manifest[index];raw=cv2.imread(record['image_path']);mask=cv2.dilate((cv2.imread(record['mask_path'],0)>0).astype('uint8'),np.ones((5,5),np.uint8))>0
        mask[:120,:570]=True;mask[945:,1660:]=True
        warp=cv2.getAffineTransform(anchor_native,projected.astype('float32'))
        polygon=cv2.transform(np.float32([[[76,512],[275,512],[275,650],[76,650]]]),warp)[0]
        x0,y0=np.floor(polygon.min(0)).astype(int);x1,y1=np.ceil(polygon.max(0)).astype(int)
        offset=np.zeros(2);score=None
        if 25<x0<x1<1895 and 20<y0<y1<1060:
            expected=cv2.warpAffine(source_raw,warp,(1920,1080))[y0:y1,x0:x1]
            search=raw[y0-15:y1+15,x0-20:x1+20]
            response=cv2.matchTemplate(cv2.cvtColor(search,cv2.COLOR_BGR2GRAY),cv2.cvtColor(expected,cv2.COLOR_BGR2GRAY),cv2.TM_CCOEFF_NORMED)
            _,score,_,location=cv2.minMaxLoc(response)
            if score>.45:offset=np.array(location)-[20,15]
        added=[]
        for tid,target in targets.items():
            world=feature_world(scene,target,templates);paths=visible_projected_runs(camera,world);segments=[]
            for path in paths:
                p=path_stations(path+offset,3.)
                readouts,_=profile_read(raw,mask,p,[1/3]*3,6,maximum_blur=5.5,minimum_amplitude=20.,navigation_tolerance=5.,require_clear_end_plateaus=True)
                segments+=segment_records(readouts,spacing=45,closed=np.linalg.norm(p[-1]-p[0])<1.,minimum_span=8.)
            if not segments:continue
            feature=dict(id=tid,target_id=tid,label=target['contour_key'],label_zh=tid,physical_surface='board',review_layer='ad2_print',
                segments=segments,allowed_for_fit=False,current_source_only_readout=True,model_or_template_generated_observation=False,
                source_alignment_navigation=dict(native_template_match_offset_xy=offset.tolist(),match_score=score,coordinates_are_navigation_not_observations=True))
            retain_print_band_segments(packet['features'],feature)
            if feature['segments']:packet['features'].append(feature);added.append(tid)
        if added:
            from .single_frame_source_review import compact_review_points
            count=sum(len(s['uv'])for f in packet['features']for s in f['segments'])
            if count>300:
                glyphs=[f for f in packet['features']if'.print.'in f['id']];board_count=sum(len(s['uv'])for f in packet['features']if'.print.'not in f['id']for s in f['segments'])
                compact_review_points(glyphs,300-board_count)
            packet['point_count']=sum(len(s['uv'])for f in packet['features']for s in f['segments']);packet['diagnostics']['printed_point_count']=sum(len(s['uv'])for f in packet['features']if'.print.'in f['id']for s in f['segments'])
            packet['diagnostics']['material_targets']=sorted({f['target_id']for f in packet['features']})
            write(DATA/f'board_candidates/{index:06d}.json',packet);rows.append(dict(index=index,new_target_ids=added,points=packet['point_count']))
        if index%25==0 or index in(834,850):cv2.imwrite(str(DATA/f'board_overlays/{index:06d}.png'),native_overlay(raw,list(source_segments(packet))),[cv2.IMWRITE_PNG_COMPRESSION,1])
    write(DATA/'Ad12_current_source_completion.json',rows);print('Ad12 source frames',len(rows),'range',(rows[0]['index'],rows[-1]['index'])if rows else None,flush=True)


if __name__=='__main__':register();complete()
