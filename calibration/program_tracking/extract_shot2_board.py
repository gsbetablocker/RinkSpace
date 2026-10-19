"""Explicit selected-frame board extraction for the native 50 fps second shot.

The original first-shot readout primitives are shared. All paths, selected
indices, cameras and model inputs are supplied by this CLI. Projection only
navigates identity; every readout comes from the current native RGB.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import cv2
import numpy as np
from .extract_first_shot_board_source import (
    read, write, sha, path_stations, profile_read, segment_records,
    visible_projected_runs, refined_ad_mapping, transform,
    retain_print_band_segments, native_overlay, source_segments,
)
from .framewise_model import feature_world
from .projection import project_fast
from .repair_first_shot_board_patterns import circle_feature
from .recover_clear_first_shot_print import current_clear_outline
from .prune_first_shot_board_uncertain_print import fragment_rejection


def visible_source_runs(points, valid):
    """Split a contour where a source mask/crop hid ink; never close that gap."""
    points=np.asarray(points,float);valid=np.asarray(valid,bool)
    if valid.all():return [(points,True)]
    cut=int(np.flatnonzero(~valid)[0]);order=np.r_[np.arange(cut+1,len(points)),np.arange(cut+1)]
    result=[];run=[]
    for j in order:
        if valid[j]:run.append(points[j])
        else:
            if len(run)>=4:result.append((np.asarray(run),False))
            run=[]
    return result


def build_current_rectangle_charts(data,model):
    """Initialize missing/distorted print shapes from selected current RGB.

    The measured board sides only initialize each continuous-material chart.
    Chart vertices are model initializers, not the final review observations.
    """
    scene=read(model/'scene.json');registry=read(model/'feature_registry.json');templates=read(model/'ad2_material_template.json')
    records={int(r['index']):r for r in read(data/'frame_manifest.json')['frames']}
    archive=np.load(data/'camera_seeds.npz');cameras=dict(zip(archive['programme_indices'],archive['camera25']))
    receipts=[]
    for number,index in ((1,1670),(2,1746),(3,1811),(7,1249)):
        record=records[index];raw=cv2.imread(record['image_path'])
        person=cv2.dilate((cv2.imread(record['mask_path'],0)>0).astype('uint8'),np.ones((5,5),np.uint8))>0
        packet=read(data/f'board_candidates/{index:06d}.json')
        mapping=refined_ad_mapping(packet['features'],cameras[index],scene,registry,number)
        if mapping is None:raise ValueError(f'No source material edges for Ad{number} chart')
        quad=transform([[0,0],[1,0],[1,1],[0,1]],mapping)
        inside=np.zeros(raw.shape[:2],np.uint8);cv2.fillConvexPoly(inside,np.rint(quad).astype('int32'),1)
        inside=cv2.erode(inside,np.ones((7,7),np.uint8))>0
        support=inside&~person
        b,g,r=raw.transpose(2,0,1).astype(float)
        bright=(np.minimum.reduce((b,g,r))>(120 if number==7 else 150))&(np.minimum.reduce((b,g,r))/np.maximum(np.maximum.reduce((b,g,r)),1)>(.74 if number==7 else .82))
        ink=bright&support
        contours,hierarchy=cv2.findContours(ink.astype('uint8')*255,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
        valid_region=cv2.erode(support.astype('uint8'),np.ones((3,3),np.uint8))>0
        entries=[]
        for j,contour in enumerate(contours):
            p=contour[:,0];x,y,w,h=cv2.boundingRect(contour);area=abs(cv2.contourArea(contour))
            if area<15 or w<4 or h<5:continue
            hole=bool(hierarchy[0,j,3]>=0)
            # QR pixels are useful at three clearly identifiable finder blocks
            # and their outer frame, without hundreds of blurred bit cells.
            if number==1 and x<540 and 435<y<690 and not (w>170 and h>170):continue
            good=valid_region[p[:,1],p[:,0]]
            for k,(path,closed) in enumerate(visible_source_runs(p,good)):
                if np.linalg.norm(np.ptp(path,axis=0))<12:continue
                path=cv2.approxPolyDP(path.astype('float32')[:,None],.8,closed)[:,0].astype(float)
                if len(path)<3:continue
                material=transform(path,np.linalg.inv(mapping))
                if ((material<0).any() or (material>1).any()):continue
                entries.append((x,y,hole,j,k,path,material,closed))
        if number==2:
            # The pale blue curved ribbons are finite printed bands. Both
            # material edges remain separate contours, never a centreline.
            blue=(b-r>15)&(b-g>15)&support
            cs,_=cv2.findContours(blue.astype('uint8')*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            for component,c in enumerate(cs):
                p=c[:,0];x,y,w,h=cv2.boundingRect(c)
                if abs(cv2.contourArea(c))<60 or max(w,h)<60:continue
                for k,(path,closed) in enumerate(visible_source_runs(p,valid_region[p[:,1],p[:,0]])):
                    if np.linalg.norm(np.ptp(path,axis=0))<24:continue
                    path=cv2.approxPolyDP(path.astype('float32')[:,None],.8,closed)[:,0].astype(float)
                    material=transform(path,np.linalg.inv(mapping))
                    if len(path)>=3 and not ((material<0).any() or (material>1).any()):entries.append((x,y,False,-3,component*100+k,path,material,closed))
        if number==1:
            gray=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)
            for finder,(x0,y0,x1,y1) in enumerate(((331,473,390,536),(456,471,512,533),(332,590,391,651))):
                roi=gray[y0:y1,x0:x1];_,binary=cv2.threshold(roi,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                cs,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                choices=[]
                for c in cs:
                    x,y,w,h=cv2.boundingRect(c);area=abs(cv2.contourArea(c))
                    if w<20 or h<20 or x<=1 or y<=1 or x+w>=x1-x0-1 or y+h>=y1-y0-1:continue
                    choices.append((area,c))
                if not choices:continue
                path=max(choices,key=lambda item:item[0])[1][:,0].astype(float)+[x0,y0]
                path=cv2.approxPolyDP(path.astype('float32')[:,None],1.,True)[:,0].astype(float)
                entries.append((x0,y0,False,-1,finder,path,transform(path,np.linalg.inv(mapping)),True))
            # Current gray gradients determine the retained sides; these
            # finder outlines and phone silhouettes only navigate the read.
            navigators=[
                [[336,474],[384,474],[384,515],[336,515]],
                [[461,473],[504,473],[504,516],[461,516]],
                [[336,611],[384,611],[384,648],[336,648]],
                [[1385,455],[1505,453],[1535,476],[1534,699],[1368,695],[1374,476]],
                [[1600,468],[1718,491],[1738,515],[1715,704],[1566,693],[1576,491]],
            ]
            for j,path in enumerate(navigators):
                path=np.asarray(path,float)
                # The sketch has no measured vertices. Keep its search chart
                # strictly inside the finite advertisement material domain.
                material=np.clip(transform(path,np.linalg.inv(mapping)),.001,.999)
                entries.append((float(path[:,0].min()),float(path[:,1].min()),False,-2,j,path,material,True))
        entries.sort(key=lambda row:(row[0],row[1],row[2],row[3],row[4]))
        charts={}
        # Current charts replace the same ad's uncertain old print initializer.
        registry['targets']={k:v for k,v in registry['targets'].items() if not k.startswith(f'Ad{number}.print.')}
        for j,(x,y,hole,component,run,path,material,closed) in enumerate(entries):
            key=f'shot2_native_component{j:03d}'+('_inner' if hole else '')
            charts[key]=dict(id=key,advertisement_id=number,closed=closed,role='visible_blue_ribbon_material_boundary' if component==-3 else 'visible_printed_hole_boundary' if hole else 'visible_printed_external_boundary',material_uv=material.tolist(),source_native_uv=path.tolist(),source_frame_index=index,source_signal_BGR_weights=[1,0,-1] if component==-3 else [1/3]*3,material_coordinate_domain='ad_rectangle_u_left_to_right_v_top_to_bottom',initialization_accuracy_accepted=False,source_occlusion_gap_not_closed=True,source_path_is_hand_navigation=component==-2)
            registry['targets'][f'Ad{number}.print.{key}']=dict(kind='ad_print',advertisement_id=number,contour_key=key,contour_role=charts[key]['role'],residual='point_to_named_finite_print_contour',template='ad2_material_template.json',material_coordinate_domain='ad_rectangle_u_left_to_right_v_top_to_bottom',geometry_status='current_native_source_material_initializer_accuracy_not_accepted',current_selected_source_chart=True)
            if component==-2 and run>=3:
                registry['targets'][f'Ad{number}.print.{key}']['minimum_current_amplitude']=12.
                registry['targets'][f'Ad{number}.print.{key}']['contour_role']='visible_phone_outer_outline'
            if component==-3:registry['targets'][f'Ad{number}.print.{key}']['minimum_current_amplitude']=14.
            if number==7:
                registry['targets'][f'Ad{number}.print.{key}'].update(minimum_current_amplitude=15.,maximum_current_blur=10.,source_profile_radius=10.,source_navigation_tolerance=7.)
        templates.setdefault('advertisements',{})[str(number)]=dict(schema='rinkspace.selected-current-source-material-chart.v1',advertisement_number=number,contours=charts,source=dict(path=record['image_path'],sha256=sha(record['image_path']),mask_path=record['mask_path'],mask_sha256=sha(record['mask_path']),programme_frame_index_1based=index,native_pts=record['source']['native_pts']),source_to_material_homography=np.linalg.inv(mapping).tolist(),source_corners_are_navigation_not_precision_measurements=True,physical_chart='continuous_board_station_and_height',planar_3d_ad_surface_required=False,initialization_accuracy_accepted=False)
        preview=raw.copy()
        for c in charts.values():cv2.polylines(preview,[np.rint(c['source_native_uv']).astype('int32')],c['closed'],(0,0,255),1)
        cv2.imwrite(str(data/f'Ad{number}_source_chart.png'),preview)
        receipts.append(dict(advertisement_number=number,source_frame_index=index,components=len(charts)))
    write(model/'feature_registry.json',registry);write(model/'ad2_material_template.json',templates)
    manifest=read(model/'model.json');manifest['feature_registry']['sha256']=sha(model/'feature_registry.json');manifest['material_templates']['sha256']=sha(model/'ad2_material_template.json')
    manifest['current_selected_source_material_charts']=receipts;write(model/'model.json',manifest)
    write(data/'current_board_charts.json',receipts)
    print(json.dumps(receipts),flush=True)


def build_current_end_prints(data,model):
    """Add current Can lettering and Ankara badge/text on their known board.

    Their inherited support is the verified right-end vertical board plane.
    A projected source-to-plane chart initializes print geometry only. Current
    source profiles supply all review coordinates, with no unknown depth.
    """
    scene=read(model/'scene.json');registry=read(model/'feature_registry.json');templates=read(model/'ad2_material_template.json')
    # The blue shield already has a finite continuous-material anchor chart;
    # it is a printed pattern, without a fabricated rectangular ad boundary.
    scene['advertisements']=[a for a in scene['advertisements'] if int(a['user_number'])!=25]
    archive=np.load(data/'camera_seeds.npz');cameras=dict(zip(archive['programme_indices'],archive['camera25']))
    records={int(r['index']):r for r in read(data/'frame_manifest.json')['frames']};receipts=[]
    for number,index,roi,bright in ((22,1469,(1446,498,1688,713),True),(24,1469,(332,463,434,625),False),(25,1469,(55,473,178,589),False)):
        record=records[index];raw=cv2.imread(record['image_path']);person=cv2.dilate((cv2.imread(record['mask_path'],0)>0).astype('uint8'),np.ones((5,5),np.uint8))>0
        x0,y0,x1,y1=roi;support=np.zeros(raw.shape[:2],np.uint8);support[y0:y1,x0:x1]=1
        if number in (22,24):
            if number==22:
                existing=next(a for a in scene['advertisements'] if int(a['user_number'])==number)
                outer=np.asarray(existing['external_contours'][0]['world_xyz_scene_units'])
            else:outer=np.asarray(scene['additional_board_circle_patterns']['circles']['24']['circle_outer_stroke_xyz_scene_units'])
            native,depth=project_fast(cameras[index],outer)
            shape=np.zeros(raw.shape[:2],np.uint8);cv2.fillPoly(shape,[np.rint(native).astype('int32')],1)
            support &=cv2.erode(shape,np.ones((19 if number==22 else 13,19 if number==22 else 13),np.uint8))
        support=(support>0)&~person
        gray=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)
        ink=((gray>165) if bright else (gray<145))&support
        contours,hierarchy=cv2.findContours(ink.astype('uint8')*255,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
        valid_region=cv2.erode(support.astype('uint8'),np.ones((3,3),np.uint8))>0
        board_x=float(scene['shared_board_patterns']['shield_fixed_material_points_xyz_scene_units']['25']['upper_left_printed_outline_corner'][0])
        yz=np.float32([[12,0],[20,0],[20,4],[12,4]])
        world=np.column_stack((np.full(4,board_x),yz));native,depth=project_fast(cameras[index],world)
        mapping=cv2.getPerspectiveTransform(native.astype('float32'),yz)
        items=[]
        for j,c in enumerate(contours):
            p=c[:,0];x,y,w,h=cv2.boundingRect(c);area=abs(cv2.contourArea(c))
            if area<6 or w<3 or h<4:continue
            hole=bool(hierarchy[0,j,3]>=0)
            for path,closed in visible_source_runs(p,valid_region[p[:,1],p[:,0]]):
                if np.linalg.norm(np.ptp(path,axis=0))<8:continue
                path=cv2.approxPolyDP(path.astype('float32')[:,None],.6,closed)[:,0].astype(float)
                if len(path)<3:continue
                material=transform(path,mapping);xyz=np.column_stack((np.full(len(path),board_x),material))
                items.append((x,y,hole,path,xyz,closed))
        items.sort(key=lambda item:(item[0],item[1],item[2]))
        advertisement=next((a for a in scene['advertisements'] if int(a['user_number'])==number),None)
        if number==24 and advertisement is None:
            advertisement=dict(user_number=24,external_contours=[],print_pattern_only=True,geometry_accuracy_accepted=False)
            scene['advertisements'].append(advertisement)
        if number!=25:advertisement['external_contours']=[c for c in advertisement.get('external_contours',[]) if not c['id'].startswith('shot2_native_')]
        registry['targets']={k:v for k,v in registry['targets'].items() if not (k.startswith(f'Ad{number}.print.shot2_native_'))}
        preview=raw.copy()
        shield_charts={}
        for j,(x,y,hole,path,xyz,closed) in enumerate(items):
            key=f'shot2_native_component{j:03d}'+('_inner' if hole else '')
            if closed:xyz=np.vstack((xyz,xyz[:1]))
            if number==25:
                roles=('upper_left_printed_outline_corner','upper_right_printed_outline_corner','bottom_printed_outline_tip')
                anchors=np.asarray([scene['shared_board_patterns']['shield_fixed_material_points_xyz_scene_units']['25'][role] for role in roles])
                shared=scene['shared_board_patterns'];old_coordinates=np.asarray(shared['instance_material_coordinates_m']['shield25'])
                scale=float(np.median(old_coordinates[:,1]/anchors[:,2]))
                course=np.asarray(shared['course_xy_m']);stations=np.asarray(shared['course_station_m'])
                material_anchors=[]
                for anchor in anchors:
                    delta=course[1:]-course[:-1]
                    fractions=np.clip(np.sum((anchor[:2]*scale-course[:-1])*delta,axis=1)/np.maximum(np.sum(delta*delta,axis=1),1.e-12),0,1)
                    nearest=course[:-1]+fractions[:,None]*delta
                    owner=int(np.argmin(np.linalg.norm(nearest-anchor[:2]*scale,axis=1)))
                    material_anchors.append([float(stations[owner]+fractions[owner]*(stations[owner+1]-stations[owner])),float(anchor[2]*scale)])
                # The previously unregistered opposite badge kept the base
                # badge's station. Register its actual unchanged named XYZ
                # anchors on this candidate's same continuous perimeter.
                shared['instance_material_coordinates_m']['shield25']=material_anchors
                basis=np.column_stack((anchors[:,1:],np.ones(3)))
                weights=np.column_stack((xyz[:,1:],np.ones(len(xyz))))@np.linalg.inv(basis)
                shield_charts[key]=dict(id=key,advertisement_id=25,closed=closed,material_anchor_weights=weights.tolist(),role='visible_blue_shield_or_Ankara_text_boundary',source_native_uv=path.tolist(),source_frame_index=index,material_coordinate_domain='continuous_board_station_height_from_existing_shield_three_anchors',initialization_accuracy_accepted=False)
            else:
                advertisement['external_contours'].append(dict(id=key,world_xyz_scene_units=xyz.tolist(),closed=closed,source_native_uv=path.tolist(),source_frame_index=index,source_path=record['image_path'],source_png_sha256=sha(record['image_path']),geometry_initialization_only=True,model_accuracy_accepted=False,source_material_plane='existing_right_end_board_body',source_to_material_yz_homography=mapping.tolist()))
            registry['targets'][f'Ad{number}.print.{key}']=dict(kind='ad_print',advertisement_id=number,contour_key=key,geometry_source='scene_shield_anchor_material_chart' if number==25 else 'scene_existing_external_contour',residual='point_to_named_finite_print_contour',geometry_status='current_native_source_initializer_on_confirmed_board_body',current_selected_source_chart=True,contour_closed=closed,minimum_current_amplitude=18.)
            cv2.polylines(preview,[np.rint(path).astype('int32')],closed,(0,0,255),1)
        if number==25:templates.setdefault('advertisements',{})['25']=dict(advertisement_number=25,contours=shield_charts,source=dict(path=record['image_path'],sha256=sha(record['image_path']),programme_frame_index_1based=index),physical_chart='existing_continuous_shield_station_height_anchor_chart',planar_3d_rectangle_required=False,initialization_accuracy_accepted=False)
        if number==24:
            descriptor=scene.setdefault('source_conditioned_board_prints',dict(schema='rinkspace.current-source-conditioned-board-print-contours.v1',patterns=[]))
            descriptor['patterns']=[p for p in descriptor['patterns'] if int(p['advertisement_id'])!=24]
            descriptor['patterns'].append(dict(advertisement_id=24,geometry_initialization_only=True,accuracy_accepted=False,external_contours=[dict(role=c['id'],xyz_scene_units=c['world_xyz_scene_units']) for c in advertisement['external_contours']]))
        cv2.imwrite(str(data/f'Ad{number}_source_chart.png'),preview)
        receipts.append(dict(advertisement_number=number,source_frame_index=index,components=len(items),initializer_only=True))
    write(model/'scene.json',scene);write(model/'feature_registry.json',registry);write(model/'ad2_material_template.json',templates)
    manifest=read(model/'model.json');manifest['scene']['sha256']=sha(model/'scene.json');manifest['feature_registry']['sha256']=sha(model/'feature_registry.json')
    manifest['material_templates']['sha256']=sha(model/'ad2_material_template.json')
    manifest['current_selected_source_end_print_charts']=receipts;write(model/'model.json',manifest)
    write(data/'current_end_print_charts.json',receipts);print(json.dumps(receipts),flush=True)


def prune_source_ad_boundary_confusions(data,model):
    """Reject ads/print whose source pixels actually name another material."""
    scene=read(model/'scene.json');registry=read(model/'feature_registry.json')
    archive=np.load(data/'camera_seeds.npz');cameras=dict(zip(archive['programme_indices'],archive['camera25']))
    manifest=read(data/'frame_manifest.json');changes=[]
    for record in manifest['frames']:
        index=int(record['index']);packet=read(data/f'board_candidates/{index:06d}.json')
        source_features=deepcopy(packet['features'])
        contact=np.vstack([s['uv'] for f in source_features if f['target_id'].startswith('board.contact.') for s in f['segments']]);contact=contact[np.argsort(contact[:,0])]
        mappings={n:refined_ad_mapping(source_features,cameras[index],scene,registry,n) for n in (1,2,3,4,6,7)}
        rejected=[]
        for feature in packet['features']:
            if not feature['target_id'].startswith('Ad'):continue
            descriptor=registry['targets'][feature['target_id']];kind=descriptor['kind'];number=int(descriptor['advertisement_id']);retained=[]
            for segment in feature['segments']:
                uv=np.asarray(segment['uv']);good=np.ones(len(uv),bool);reason=None
                if kind=='ad_edge':
                    bottom=np.interp(uv[:,0],contact[:,0],contact[:,1])
                    good &=uv[:,1]<bottom-10.
                    reason='yellow_kickplate_or_ice_contact_is_not_this_advertisement_edge'
                if kind=='ad_print' and mappings.get(number) is not None:
                    quad=transform([[0,0],[1,0],[1,1],[0,1]],mappings[number]).astype('float32')
                    distances=np.array([cv2.pointPolygonTest(quad,tuple(p),True) for p in uv])
                    good &= distances>5.
                    reason='print_outside_or_indistinguishable_from_current_source_advertisement_border'
                if good.all():retained.append(segment);continue
                rejected.append(dict(target_id=feature['target_id'],reason=reason,source_native_uv=uv[~good].tolist()))
                runs=[];run=[]
                for j,accept in enumerate(good):
                    if accept:run.append(j)
                    else:
                        if len(run)>=2:runs.append(run)
                        run=[]
                if len(run)>=2:runs.append(run)
                for keep in runs:
                    replacement=deepcopy(segment)
                    for key,value in segment.items():
                        if isinstance(value,list) and len(value)==len(uv) and key not in ('source_native_readout_uv','source_native_transition_iqr_px'):
                            replacement[key]=[value[j] for j in keep]
                    replacement.update(uv=uv[keep].tolist(),vertex_count=len(keep),closed=False,current_source_ad_border_identity_checked=True)
                    if 'corner_vertex_indices' in replacement:replacement['corner_vertex_indices']={k:keep.index(v) for k,v in segment['corner_vertex_indices'].items() if v in keep}
                    retained.append(replacement)
            feature['segments']=retained
        if not rejected:continue
        packet['features']=[f for f in packet['features'] if f['segments']]
        packet.setdefault('source_material_identity_rejections',[]).extend(rejected)
        packet['point_count']=sum(len(s['uv']) for f in packet['features'] for s in f['segments'])
        packet['diagnostics']['board_point_count']=sum(len(s['uv']) for f in packet['features'] if '.print.' not in f['id'] for s in f['segments'])
        packet['diagnostics']['printed_point_count']=packet['point_count']-packet['diagnostics']['board_point_count']
        packet['diagnostics']['material_targets']=sorted({f['target_id'] for f in packet['features']})
        ids={f['id'] for f in packet['features']};packet['source_correlated_groups']=[g for g in packet.get('source_correlated_groups',[]) if g.get('feature_id') in ids]
        write(data/f'board_candidates/{index:06d}.json',packet)
        raw=cv2.imread(record['image_path']);cv2.imwrite(str(data/f'board_overlays/{index:06d}.png'),native_overlay(raw,list(source_segments(packet))))
        changes.append(dict(index=index,rejected=rejected,remaining_points=packet['point_count']))
    write(data/'board_boundary_identity_rejections.json',dict(rows=changes,current_source_identity_only=True,source_points_not_moved=True))
    print('Boundary identity pruned',[(r['index'],r['remaining_points']) for r in changes],flush=True)

def extract_frame(index,record,camera,scene,registry,templates,*,data,overlay=False):
    raw=cv2.imread(record['image_path']);mask=cv2.imread(record['mask_path'],0)
    if mask is None:raise ValueError('Current source person exclusion is not ready')
    mask=cv2.dilate((mask>0).astype(np.uint8),np.ones((5,5),np.uint8))>0
    # Broadcast overlays never become physical board entities.
    mask[:120,:570]=True;mask[945:,1660:]=True
    features=[];diagnostics=[]
    targets=registry['targets']
    advertisement_numbers=sorted({int(t['advertisement_id']) for t in targets.values() if t['kind']=='ad_edge' and all(f"Ad{int(t['advertisement_id'])}.{role}" in targets for role in ('left_top','right_top','right_bottom','left_bottom'))})
    for tid,target in targets.items():
        kind=target['kind']
        if kind not in ('board_edge','ad_edge'):continue
        if kind=='board_edge' and target.get('part') is None:continue
        world=feature_world(scene,target,templates)
        all_segments=[]
        for projected in visible_projected_runs(camera,world):
            path=path_stations(projected,12)
            if np.linalg.norm(path[-1]-path[0])<10:continue
            if kind=='board_edge':
                weights=[1,-.5,-.5]if target['role']=='top'else[-1,.5,.5]
                gradient_direction=[0,1]if target['role']=='top'else[0,-1];mode=None
            else:
                number=int(target['advertisement_id']);weights=([-1,.5,.5]if number==6 else [1,-.5,-.5]if number in (3,7)else [.5,-1,.5]if number in (2,4)else[1,-.5,-.5])
                gradient_direction={'top_edge':[0,1],'bottom_edge':[0,-1],'left_vertical':[1,0],'right_vertical':[-1,0]}[target['role']];mode='chroma'
            rows,_=profile_read(raw,mask,path,weights,36 if kind=='board_edge' else 24,
                maximum_blur=18.,minimum_amplitude=10.,gradient_direction=gradient_direction,signal_mode=mode)
            straight=(kind=='board_edge' and target.get('part')in ('upper_long_side','lower_long_side','left_end','right_end'))or(kind=='ad_edge'and (int(target['advertisement_id'])in (3,4,5,6)or 'vertical'in target['role']))
            all_segments+=segment_records(rows,spacing=200,straight=straight,smooth_curve=not straight,target_id=tid)
        if not all_segments:continue
        if kind=='board_edge':fid=('board_top'if target['role']=='top'else'board_contact')+'.'+str(target['part']);layer='board_top'if target['role']=='top'else'board_contact'
        else:fid=tid;layer='ad2_border'
        features.append(dict(id=fid,label_zh=tid,physical_surface='board',target_id=tid,world_curve=tid,
            review_layer=layer,segments=all_segments,allowed_for_fit=False))
    # One physical vertical material has one current-source line. Short
    # diagonal chroma fragments at a blurred/rounded printed join do not form
    # separate model sides. All visible runs use the jointly read straight side.
    for feature in features:
        if not feature['id'].startswith('Ad') or 'vertical'not in feature['id']:continue
        evidence=np.vstack([s.get('source_native_readout_uv',s['uv'])for s in feature['segments']])
        if len(evidence)<5:continue
        line=cv2.fitLine(evidence.astype('float32'),cv2.DIST_HUBER,0,.001,.001).ravel().astype(float)
        perpendicular=np.array([-line[1],line[0]])
        errors=np.abs((evidence-line[2:])@perpendicular)
        accepted=evidence[errors<3.5]
        if len(accepted)<5:continue
        line=cv2.fitLine(accepted.astype('float32'),cv2.DIST_HUBER,0,.001,.001).ravel().astype(float);perpendicular=np.array([-line[1],line[0]])
        replacement=[]
        for segment in feature['segments']:
            native=np.asarray(segment.get('source_native_readout_uv',segment['uv']),float)
            good=np.abs((native-line[2:])@perpendicular)<3.5;native=native[good]
            if len(native)<2 or np.linalg.norm(native[-1]-native[0])<8:continue
            stations=(native[[0,-1]]-line[2:])@line[:2]
            locations=np.linspace(stations[0],stations[-1],max(2,int(np.ceil(abs(stations[1]-stations[0])/200))+1))
            uv=line[2:]+locations[:,None]*line[:2]
            segment.update(uv=uv.tolist(),vertex_count=len(uv),source_native_readout_uv=native.tolist(),
                point_bindings=[dict(kind='edge_sample',physical_target=feature['target_id'],required_residual='point_to_named_finite_model_edge',measurement_origin='current_source_joint_vertical_side_line')for _ in uv],
                shared_current_source_vertical_line=line.tolist(),printed_corner_radius_not_model_feature=True)
            replacement.append(segment)
        feature['segments']=replacement
    features=[f for f in features if f['segments']]
    # Source corners are finite same-source intersections, emitted once at the
    # relevant horizontal edge endpoint only when both materials were measured.
    for number in advertisement_numbers:
        by={f['id']:f for f in features}
        mapping=refined_ad_mapping(features,camera,scene,registry,number)
        if mapping is None:continue
        quad=transform([[0,0],[1,0],[1,1],[0,1]],mapping)
        for point,side,level in zip(quad,['left','right','right','left'],['top','top','bottom','bottom']):
            fid=f'Ad{number}.{level}_edge';vertical=f'Ad{number}.{side}_vertical';f=by.get(fid)
            if not f or vertical not in by:continue
            x,y=np.rint(point).astype(int)
            if not(4<=x<1916 and 4<=y<1076)or mask[y-3:y+4,x-3:x+4].any():continue
            # Only replace an adjacent actual readout. No disconnected edge
            # endpoint or clipped image boundary is promoted to a corner.
            candidates=[(np.linalg.norm(np.asarray(s['uv'][j])-point),s,j)for s in f['segments']for j in (0,len(s['uv'])-1)]
            distance,s,j=min(candidates,key=lambda x:x[0])
            if distance>24:continue
            s['uv'][j]=point.tolist();s['point_bindings'][j]=dict(kind='corner',physical_target=f'Ad{number}.{side}_{level}',required_residual='point_to_corresponding_model_corner_2d',measurement_origin='current_source_measured_adjacent_edge_intersection')
            s.setdefault('corner_vertex_indices',{})[side+'_'+level]=j
    mappings={n:refined_ad_mapping(features,camera,scene,registry,n)for n in advertisement_numbers}
    for tid,target in targets.items():
        if target['kind']!='ad_print':continue
        n=int(target['advertisement_id']);world=feature_world(scene,target,templates)
        if target.get('contour_key') in ('outermost_visible_circular_print_stroke','outer_circle_print_stroke'):
            projected,depth=project_fast(camera,world)
            if (depth<=0).any() or not np.isfinite(projected).all() or projected[:,0].max()<-40 or projected[:,0].min()>1959:continue
            feature,diagnostic=circle_feature(raw,mask,projected,features)
            if feature:
                feature.update(id=tid,target_id=tid,label=target['contour_key'],label_zh=tid)
                features.append(feature)
            diagnostics.append(dict(target_id=tid,**diagnostic))
            continue
        if target.get('geometry_source')in ('scene_existing_external_contour','scene_existing_board_pattern','scene_shield_anchor_material_chart'):
            paths=visible_projected_runs(camera,world);weights=[1/3,1/3,1/3];closed=bool(target.get('contour_closed',True))
        else:
            t=templates.get('advertisements',{}).get(str(n),templates);contour=t['contours'][target['contour_key']]
            mapping=mappings.get(n)
            if mapping is None:continue
            p=transform(contour['material_uv'],mapping)
            if n==2 and not target.get('current_selected_source_chart'):weights=[0,1,0]
            else:weights=contour.get('source_signal_BGR_weights',[1,1,1])
            closed=bool(contour['closed']);paths=[p]
        all_segments=[]
        for p in paths:
            if len(p)<3:continue
            if closed and np.linalg.norm(p[-1]-p[0])>.1:p=np.vstack([p,p[:1]])
            if closed and 'inner' not in tid and np.ptp(p,axis=0).min()>=12 and not target.get('current_selected_source_chart'):
                bright=n in (3,6,7,16,19,20)
                clear=current_clear_outline(raw,mask,p,bright=bright)
                if clear:
                    all_segments+=clear
                    continue
            p=path_stations(p,3.5)
            current_chart=bool(target.get('current_selected_source_chart'))
            rows,_=profile_read(raw,mask,p,weights,float(target.get('source_profile_radius',6 if current_chart else 10)),maximum_blur=float(target.get('maximum_current_blur',6.)),minimum_amplitude=float(target.get('minimum_current_amplitude',24.)),navigation_tolerance=float(target.get('source_navigation_tolerance',4.5 if current_chart else 6.5)),require_clear_end_plateaus=not current_chart)
            # A nearly complete glyph may have adjacent endpoints beside one
            # missing profile. Its full current shape still supports identity;
            # endpoint distance alone must not erase that visible outline.
            segments=segment_records(rows,spacing=45,closed=closed,minimum_span=.5 if current_chart else 8.)
            if current_chart:
                actual=np.asarray([row[0] for row in rows if row is not None])
                support_fraction=len(actual)/max(len(rows),1)
                if len(actual)>=8 and support_fraction>=.60 and np.ptp(actual,axis=0).min()>=3:
                    for segment in segments:
                        segment['source_clear_component_identity_verified_by_current_shape']=True
                        segment['native_connected_component_rgb_profile_support_fraction']=support_fraction
            all_segments+=segments
        if not all_segments:continue
        features.append(dict(id=tid,target_id=tid,label=target['contour_key'],label_zh=tid,physical_surface='board',review_layer='ad2_print',
            world_correspondence_status='named_same_physical_print_component',segments=all_segments,allowed_for_fit=False,
            current_source_only_readout=True,model_or_template_generated_observation=False))
    for feature in features:
        if '.print.'in feature['id']:retain_print_band_segments(features,feature)
    for feature in features:
        rejected=[];retained=[]
        for segment in feature['segments']:
            reason=fragment_rejection(feature,segment)
            if reason:rejected.append(dict(uv=segment['uv'],reason=reason))
            else:retained.append(segment)
        feature['segments']=retained
        if rejected:diagnostics.append(dict(target_id=feature['target_id'],rejected_unresolved_fragments=rejected))
    features=[f for f in features if f['segments']]
    count=sum(len(s['uv'])for f in features for s in f['segments'])
    # The source family is not independently capped. The parent merges ice
    # and board observations, preserving every material under the global 500.
    for f in features:
        cleaned=[]
        for s in f['segments']:
            p=np.asarray(s['uv']);good=(p[:,0]>=0)&(p[:,0]<=1919)&(p[:,1]>=0)&(p[:,1]<=1079)
            xy=np.rint(np.clip(p,[0,0],[1919,1079])).astype(int);good&=~mask[xy[:,1],xy[:,0]]
            keep=np.flatnonzero(good).tolist()
            if len(keep)<2:continue
            if not good.all():
                for key in ('uv','point_bindings','transition_iqr_px','localization_sigma_px'):
                    if key in s:s[key]=[s[key][j]for j in keep]
                if 'corner_vertex_indices'in s:s['corner_vertex_indices']={k:keep.index(v)for k,v in s['corner_vertex_indices'].items()if v in keep}
                s.update(vertex_count=len(keep),closed=False)
            cleaned.append(s)
        f['segments']=cleaned
    features=[f for f in features if f['segments']]
    count=sum(len(s['uv'])for f in features for s in f['segments'])
    source=deepcopy(record['source']);source.update(mask_path=record['mask_path'],mask_sha256=sha(record['mask_path']),source_points_extracted=True,optimization_performed=False)
    packet=dict(schema='rinkspace.single-frame-source-review.v1',source=source,source_pts=source['native_pts'],
        programme_frame_index_1based=index,batch_sequence_index_1based=index,frame_count=1,features=features,point_count=count,
        status='board_print_current_native_candidate',user_accepted=False,user_hard_point_ceiling=500,
        observation_coordinates_from_projection=False,optimization_performed=False,
        excluded_features_by_user=['blue_strip_lower_edge','yellow_strip_upper_edge','board_panel_seam'],
        unresolved_source_regions=[],source_correlated_groups=[dict(id=f['target_id']+'_current_source_shared_line',feature_id=f['id'],
            source='same_current_native_edge_joint_line_fit',representative_points_share_line_estimate=True)for f in features if any(s.get('representative_points_share_line_estimate')for s in f['segments'])],
        diagnostics=dict(board_point_count=sum(len(s['uv'])for f in features if '.print.'not in f['id']for s in f['segments']),
            printed_point_count=sum(len(s['uv'])for f in features if '.print.'in f['id']for s in f['segments']),
            material_targets=sorted({f['target_id']for f in features}),native_readout_admits_only_clear_profile_pairs=True,
            source_blur_suppresses_unresolved_glyph_detail=True,readout_details=diagnostics))
    write(data/f'board_candidates/{index:06d}.json',packet)
    if overlay:
        (data/'board_overlays').mkdir(exist_ok=True);cv2.imwrite(str(data/f'board_overlays/{index:06d}.png'),native_overlay(raw,list(source_segments(packet))),[cv2.IMWRITE_PNG_COMPRESSION,1])
    return dict(index=index,points=count,board_points=packet['diagnostics']['board_point_count'],print_points=packet['diagnostics']['printed_point_count'],
        ads=sorted({int(f['target_id'].split('.')[0][2:])for f in features if f['target_id'].startswith('Ad')}),
        board_roles=sorted({f['target_id'].split('.')[1]for f in features if f['target_id'].startswith('board.')}))


def run(data, model, indices=None):
    cv2.setNumThreads(2)
    scene=read(model/'scene.json');registry=read(model/'feature_registry.json')
    templates=read(model/'ad2_material_template.json')
    records={int(r['index']):r for r in read(data/'frame_manifest.json')['frames']}
    archive=np.load(data/'camera_seeds.npz')
    cameras={int(i):c for i,c in zip(archive['programme_indices'],archive['camera25'])}
    active=list(indices if indices is not None else records)
    for folder in ('board_candidates','board_overlays'):(data/folder).mkdir(parents=True,exist_ok=True)
    results=[];failures=[]
    for index in active:
        try:result=extract_frame(index,records[index],cameras[index],scene,registry,templates,data=data,overlay=True)
        except Exception as error:
            failures.append(dict(index=index,error=repr(error)));print(json.dumps(failures[-1]),flush=True);continue
        results.append(result);print(json.dumps(result),flush=True)
    write(data/'board_counts.json',results);write(data/'board_failures.json',failures)
    return results

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--indices')
    parser.add_argument('--build-charts',action='store_true')
    parser.add_argument('--build-end-prints',action='store_true')
    parser.add_argument('--prune-boundary-confusions',action='store_true')
    args=parser.parse_args()
    if args.build_charts:build_current_rectangle_charts(args.data,args.model)
    elif args.build_end_prints:build_current_end_prints(args.data,args.model)
    elif args.prune_boundary_confusions:prune_source_ad_boundary_confusions(args.data,args.model)
    else:run(args.data,args.model,list(map(int,args.indices.split(','))) if args.indices else None)
