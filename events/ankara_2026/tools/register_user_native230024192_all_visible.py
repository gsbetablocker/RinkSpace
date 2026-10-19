"""Append the user's same-frame ice axes and advertisement boundaries.

Only actual coloured drawing pixels supply image observations. Texture-based
registration maps the attachments to the original native RGB. Historical
packets, authorities and full-stroke arrays remain intact.
"""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
DATA = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
BASE = DATA/'evidence/user_video_review_20261008/c134_full_programme/registered_user_complete_stroke_source_append_20261008'
SOURCE = BASE.parent/'20_lower_long_side_ice_contact_source_native_230024192.png'
ICE = Path('C:/Users/REDACTED/AppData/Local/Temp/codex-clipboard-private-attachment-00028.png')
ADS = Path('C:/Users/REDACTED/AppData/Local/Temp/codex-clipboard-private-attachment-00030.png')
SUFFIX = 'with_user_native230024192_20261008'
SOURCE_SHA = 'f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf8')).hexdigest()


def register(rgb, marked):
    sift = cv2.SIFT_create(nfeatures=3000)
    def descriptors(image):
        mask=np.zeros(image.shape[:2],np.uint8)
        # Unmarked wall/windows identify image scale, translation and rotation.
        mask[:int(image.shape[0]*.29), int(image.shape[1]*.28):]=255
        return sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),mask)
    a,da=descriptors(rgb); b,db=descriptors(marked)
    matches=[m for m,n in cv2.BFMatcher().knnMatch(da,db,k=2)if m.distance<.75*n.distance]
    p=np.float32([a[m.queryIdx].pt for m in matches]); q=np.float32([b[m.trainIdx].pt for m in matches])
    matrix,selected=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=1.5,maxIters=5000)
    selected=selected[:,0].astype(bool)
    errors=np.linalg.norm(p@matrix[:,:2].T+matrix[:,2]-q,axis=1)[selected]
    if len(errors)<50:raise ValueError('Insufficient original RGB wall matches')
    inverse=cv2.invertAffineTransform(matrix)
    scale=float(np.sqrt(np.linalg.det(matrix[:,:2])))
    return inverse,dict(native_wh=[1920,1080],attachment_wh=[marked.shape[1],marked.shape[0]],
        source_to_attachment_affine=matrix.tolist(),native_from_attachment_affine=inverse.tolist(),
        method='SIFT actual unchanged source wall/window RGB; similarity RANSAC',
        texture_inliers=int(len(errors)),median_error_attachment_px=float(np.median(errors)),
        p95_error_attachment_px=float(np.quantile(errors,.95)),registration_uncertainty_native_px=float(np.quantile(errors,.95)/scale),
        source_camera_or_geometry_used=False,original_RGB_registration_not_model_projection=True)


def trace(marked,inverse,color,start,end,*,exclude_junctions=False):
    start=np.asarray(start,float);end=np.asarray(end,float);delta=end-start
    length=np.linalg.norm(delta);tangent=delta/length;normal=np.array([-tangent[1],tangent[0]])
    yy,xx=np.nonzero(np.all(marked==np.asarray(color,np.uint8),axis=2));pixels=np.c_[xx,yy].astype(float)
    along=(pixels-start)@tangent;across=(pixels-start)@normal
    margin=3. if exclude_junctions else 0.
    selected=(along>=margin)&(along<=length-margin)&(abs(across)<5.)
    pixels=pixels[selected]
    major=0 if abs(delta[0])>=abs(delta[1])else 1
    grouped=defaultdict(list)
    for pixel in pixels:grouped[int(pixel[major])].append(pixel)
    centers=[];parts=[]
    for key in sorted(grouped,reverse=bool(delta[major]<0)):
        part=np.asarray(grouped[key]);center=np.median(part,axis=0)
        native=center@inverse[:,:2].T+inverse[:,2]
        if np.any(native<0)or np.any(native>=[1920,1080]):continue
        centers.append(center);parts.append(part)
    if len(centers)<6:raise ValueError('User stroke not extracted')
    screen=np.asarray(centers);native=screen@inverse[:,:2].T+inverse[:,2]
    native_tangent=inverse[:,:2]@tangent;native_tangent/=np.linalg.norm(native_tangent)
    native_normal=np.array([-native_tangent[1],native_tangent[0]])
    normals=np.tile(native_normal,(len(native),1));spans=[];cells=[]
    half=.5*np.sum(abs(native_normal@inverse[:,:2]))
    for center,part in zip(native,parts):
        transformed=part@inverse[:,:2].T+inverse[:,2];signed=(transformed-center)@native_normal
        spans.append([float(signed.min()-half),float(signed.max()+half)])
        cells.append(dict(pixel_count=len(part),screenshot_bbox_xyxy=np.r_[part.min(0),part.max(0)].tolist()))
    return screen,native,normals,np.asarray(spans),cells


def sparse_indices(native,spacing=31.):
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(native,axis=0),axis=1))]
    # Sparse real source pixels supply the soft factor; complete correlated
    # drawing centres remain guard vertices, not independent FIT weights.
    chosen=[0]
    for i in range(1,len(native)):
        if arc[i]-arc[chosen[-1]]>=spacing:chosen.append(i)
    if chosen[-1]!=len(native)-1:chosen.append(len(native)-1)
    return np.asarray(chosen,int),arc


def main():
    cv2.setNumThreads(1)
    output_authority=BASE/f'human_registered_hard_truth_authorities_{SUFFIX}.json'
    if output_authority.exists():raise FileExistsError('Preserve existing registration')
    rgb=np.asarray(Image.open(SOURCE).convert('RGB'))
    attachments={};registration={};images={};matrices={}
    for key,path in [('ice',ICE),('ads',ADS)]:
        saved=BASE/path.name
        if not saved.exists():shutil.copyfile(path,saved)
        if sha(path)!=sha(saved):raise ValueError('Saved attachment differs')
        images[key]=np.asarray(Image.open(saved).convert('RGB'))
        matrices[key],registration[key]=register(rgb,images[key]);attachments[key]=ref(saved)
    specifications=[
        ('ice','B2',dict(kind='current_ice_segment',feature_id='B2',segment_index=0),(0,176,80),(3,548),(1546,558),'native_ice_curve'),
        ('ice','C0',dict(kind='current_ice_segment',feature_id='C',segment_index=0),(255,0,0),(2,609),(1709,615),'native_ice_curve'),
        ('ice','B1',dict(kind='current_ice_segment',feature_id='B1',segment_index=0),(0,176,80),(2,702),(1977,710),'native_ice_curve'),
        ('ice','Lane1 上长边',dict(kind='current_ice_segment',feature_id='lane1',segment_index=0),(21,96,130),(2,813),(557,506),'native_ice_curve'),
        ('ice','Lane1 中线',dict(kind='curling_centreline',row=1),(21,96,130),(754,1125),(854,527),'native_ice_curve'),
        ('ice','Lane1 下长边',dict(kind='current_ice_segment',feature_id='lane1',segment_index=2),(21,96,130),(1156,516),(1989,1109),'native_ice_curve'),
    ]
    ad_corners={17:[(1545,418),(2003,480),(1991,667),(1543,530)],18:[(1452,416),(1467,417),(1463,487),(1449,484)],
                19:[(1355,391),(1400,398),(1397,484),(1354,471)],20:[(924,379),(1259,387),(1258,462),(923,453)]}
    for number,corners in ad_corners.items():
        scope='internal_graphic_only'if number==18 else'external_border'
        for edge,a,b in [('top_edge',0,1),('right_vertical',1,2),('bottom_edge',3,2),('left_vertical',0,3)]:
            specifications.append(('ads',f'ad{number} {edge}',dict(kind='advertisement_boundary',advertisement_number=number,edge=edge,extent_scope=scope),
                (21,96,130),corners[a],corners[b],'native_board_edge'))
    board_parent=BASE/'board_parent77_plus_user_confirmed230501120_top_contact_v1.json'
    ice_parent=BASE/'ice_parent51_plus_user19_right_goal_baseline_registered_complete_stroke_FIT_CANDIDATE_v2_HUMAN_LABEL_SUPPORT.json'
    packets={'native_board_edge':json.loads(board_parent.read_text(encoding='utf8')),'native_ice_curve':json.loads(ice_parent.read_text(encoding='utf8'))}
    packet_paths={'native_board_edge':BASE/f'board_sources_{SUFFIX}.json','native_ice_curve':BASE/f'ice_sources_{SUFFIX}.json'}
    authority=json.loads((BASE/'human_registered_hard_truth_authorities_v1.json').read_text(encoding='utf8'))
    guard=json.loads((BASE/'human_registered_full_stroke_hard_guard_v1.json').read_text(encoding='utf8'))
    with np.load(guard['guard_arrays_reference']['path'],allow_pickle=False)as existing:arrays={k:existing[k].copy()for k in existing.files}
    cap=json.loads((BASE/'human_registered_stroke_end_2D_source_footprint_support_v2.json').read_text(encoding='utf8'))
    new_rows=[];new_records=[];fact_lines=[];full_total=0
    for ordinal,(key,label,curve,color,start,end,family)in enumerate(specifications):
        screen,uv,normals,spans,cells=trace(images[key],matrices[key],color,start,end,exclude_junctions=key=='ads')
        selected,arc=sparse_indices(uv);reg=registration[key]['registration_uncertainty_native_px']
        # User explicitly rejected pixel/subpixel precision for manual drawing.
        manual=6.;bounds=spans+np.array([-reg-manual,reg+manual])
        identifier=f'user_native230024192_{label.replace(" ","_")}_20261008'
        trace_path=BASE/f'annotated_native230024192_stroke_{ordinal:02d}.json'
        write(trace_path,dict(schema='rinkspace.registered-user-source-stroke.v1',id=identifier,user_label=label,
            physical_curve=curve,registration=registration[key],original_attachment=attachments[key],
            source_pts=230024192,time_base='1/15360',source_raw_frame=str(SOURCE),source_file_sha256=sha(SOURCE),
            user_stroke_raw_screenshot_px=screen.tolist(),user_stroke_raw_native_px=uv.tolist(),
            source_only_reading=True,model_projection_used=False,endpoint_observation=False,
            normal_interval_px=bounds.tolist(),practical_manual_location_uncertainty_native_px=manual,
            practical_uncertainty_is_user_pixel_precision=False,drawing_pixels_are_correlated=True))
        identity=curve.get('feature_id',f'Ad{curve.get("advertisement_number")}')
        if curve['kind']=='curling_centreline':identity='house_connection_F1_R1'
        role=('internal_graphic_only_boundary'if curve.get('extent_scope')=='internal_graphic_only'else
              'external_print_boundary'if family=='native_board_edge'else'nominal_painted_stroke_center_axis')
        row=dict(id=identifier,pts=230024192,actual_pts=230024192,feature_id=identity,curve=curve,split='fit',geometry_role=role,
            uv=uv[selected].tolist(),normal=normals[selected].tolist(),normal_interval_px=bounds[selected].tolist(),sigma_px=[6.]*len(selected),
            point_correspondence=False,curve_observation_kind='normal_interval',endpoint_observation=False,
            source_optical_normal_support_authorized=False,source_raw_frame=str(SOURCE),source_file_sha256=sha(SOURCE),
            source_user_annotation=attachments[key],source_annotation_registration=ref(trace_path),
            user_label=label,user_confirmed_source_axis=True,direct_human_GT=True,
            sourceUV_origin='direct_user_coloured_drawing_registered_to_original_native_RGB',
            model_projection_used_to_generate_or_select_uv=False,source_RGB_actual_measurement=False,
            annotation_stroke_normal_interval_px=spans[selected].tolist(),registration_uncertainty_native_px=reg,
            practical_manual_location_uncertainty_native_px=manual,practical_uncertainty_is_user_pixel_precision=False,
            sigma_semantics='6px engineering soft-factor scale; hard visual position uses drawing plus independent registration and manual location envelope')
        index=len(packets[family]['rows']);packets[family]['rows'].append(row);new_rows.append((family,index,row))
        record=dict(id=identifier,family=family,original_row_index=index,source_row_canonical_json_sha256=canonical_sha(row),
            pts=230024192,actual_pts=230024192,time_base='1/15360',source_video_sha256=SOURCE_SHA,curve=curve,geometry_role=role,
            uv=row['uv'],normal=row['normal'],normal_interval_px=row['normal_interval_px'],
            lower_bound_px=bounds[selected,0].tolist(),upper_bound_px=bounds[selected,1].tolist(),
            direct_human_GT=True,model_or_RGB_transport_used_to_generate_UV=False,historical_sigma_used_as_hard_tolerance=False,
            source_raw_frame=str(SOURCE),source_file_sha256=sha(SOURCE),source_annotation_registration=ref(trace_path),source_annotation_attachment=attachments[key],
            user_label=label,endpoint_observation=False,authority_temporal_scope=dict(native_pts_start_inclusive=230024192,native_pts_stop_inclusive=230024192,propagation_to_other_exposures_is_GT=False),
            complete_human_stroke=dict(reference=ref(trace_path),native_coordinates_field='user_stroke_raw_native_px',sample_count=len(uv)))
        new_records.append(record)
        gi=len(guard['records']);prefix=f'record_{gi:02d}'
        for field,value in [('uv',uv),('normal',normals),('normal_interval_px',bounds),('annotation_stroke_normal_interval_px',spans),('screenshot_xy',screen),('source_arc_length_px',arc)]:arrays[prefix+'_'+field]=value
        guard['records'].append(dict(id=identifier,authority_record_index=gi,authority_source_row_canonical_json_sha256=canonical_sha(row),
            family=family,curve=curve,geometry_role=role,pts=230024192,actual_pts=230024192,vertex_count=len(uv),array_prefix=prefix,
            source_raw_frame=str(SOURCE),source_file_sha256=sha(SOURCE),source_annotation_registration=ref(trace_path),source_annotation_attachment=attachments[key],
            full_trace_is_guard_only=True,dense_pixels_are_not_independent_FIT_residuals=True,endpoint_observation=False))
        for vi,name in [(0,'first'),(len(uv)-1,'last')]:
            normal=normals[vi];tangent=np.array([-normal[1],normal[0]])
            radius=float(np.max(abs(bounds[vi])))
            cap['records'].append(dict(id=identifier,source_full_guard_record_index=gi,source_vertex_index=vi,stroke_end=name,
                pts=230024192,actual_pts=230024192,uv=uv[vi].tolist(),normal=normal.tolist(),tangent=tangent.tolist(),
                normal_cell_support_px=bounds[vi].tolist(),tangent_cell_support_px=[-radius,radius],
                registration_uncertainty_native_px=reg,practical_manual_location_uncertainty_native_px=manual,
                selected_actual_coloured_pixel_count=cells[vi]['pixel_count'],selected_screenshot_bbox_xyxy=cells[vi]['screenshot_bbox_xyxy'],
                source_annotation_attachment=attachments[key],source_annotation_registration=ref(trace_path),
                source_raw_frame=str(SOURCE),source_file_sha256=sha(SOURCE),physical_endpoint_or_corner_observation=False,
                rectangular_marker_joint_not_free_cap=key=='ads',normal_band_from_frozen_guard_not_replaced=True,
                selection='actual coloured source pixels; marker junction omitted, drawing coverage endpoint has no physical endpoint meaning'))
        full_total+=len(uv);fact_lines.append(dict(id=identifier,user_label=label,physical_curve=curve,trace_reference=ref(trace_path),native_bounds_xyxy=np.r_[uv.min(0),uv.max(0)].tolist(),raw_drawing_vertices=len(uv),sparse_queries=len(selected)))
    for family,packet in packets.items():
        packet['new_direct_human_same_frame_append']=dict(source_pts=230024192,added_ids=[r['id']for f,_,r in new_rows if f==family],old_source_rows_preserved_exactly=True)
        write(packet_paths[family],packet)
    for record in new_records:record['packet_reference']=ref(packet_paths[record['family']])
    authority['records'].extend(new_records);authority['source_authority_append']=dict(attachments=attachments,registration=registration,old_authorities_preserved=True)
    write(output_authority,authority)
    guard_arrays=BASE/f'human_registered_full_stroke_hard_guard_{SUFFIX}.npz';np.savez_compressed(guard_arrays,**arrays)
    guard_path=BASE/f'human_registered_full_stroke_hard_guard_{SUFFIX}.json'
    guard.update(primary_authority_reference=ref(output_authority),guard_arrays_reference=ref(guard_arrays),record_count=len(guard['records']),
        full_stroke_vertex_count=sum(r['vertex_count']for r in guard['records']),primary_UV_count=sum(len(r['uv'])for r in authority['records']))
    write(guard_path,guard)
    cap_path=BASE/f'human_registered_stroke_end_2D_source_footprint_support_{SUFFIX}.json'
    cap.update(primary_authority_reference=ref(output_authority),full_trace_reference=ref(guard_path),full_trace_arrays_reference=ref(guard_arrays),endpoint_record_count=len(cap['records']))
    write(cap_path,cap)
    confirmed_parent=BASE/'user_confirmed_native_source_priority_authorities_v1.json'
    confirmed=json.loads(confirmed_parent.read_text(encoding='utf8'))
    for item in confirmed['records']:
        parent_packet=json.loads(Path(item['packet_reference']['path']).read_text(encoding='utf8'))
        if parent_packet['rows'][item['original_row_index']]!=packets[item['family']]['rows'][item['original_row_index']]:
            raise ValueError('Appended packet changed existing confirmed source row')
        item['historical_packet_reference']=deepcopy(item['packet_reference'])
        item['packet_reference']=ref(packet_paths[item['family']])
    confirmed['append_binding']=dict(parent_authority=ref(confirmed_parent),all_confirmed_original_UV_sigma_source_rows_unchanged=True)
    confirmed_path=BASE/f'user_confirmed_native_source_priority_authorities_{SUFFIX}.json'
    write(confirmed_path,confirmed)
    fact_path=ROOT/'events/ankara_2026/model_constraints/user_native230024192_all_visible_20261008.json'
    fact=dict(schema='rinkspace.user-confirmed-same-frame-visible-geometry.v1',event_id='ankara_2026',programme_id='okada_mei_sp',source_pts=230024192,
        source_time='04:09:35.533333',time_base='1/15360',source_video_sha256=SOURCE_SHA,attachments=attachments,registration=registration,
        user_marked_strokes=fact_lines,advertisement_identities=[dict(number=n,user_label=f'ad{n}',drawn_blue_boundary=n<=20)for n in range(17,24)],
        Ad18_blue_rectangle_scope='existing internal_graphic_only left printed subgraphic; user drawing is not judged missing',
        unboxed_Ad21_22_23='identity authority only; no invented rectangle or pixel contour',
        red_line_label_alias=dict(user_label='C0',existing_producer_id='C',role='central_transverse_red_painted_axis',user_label_preserved=True),
        hard_truth_authority=ref(output_authority),full_stroke_guard=ref(guard_path),source_2D_coverage_support=ref(cap_path),
        confirmed_native_source_priority_authority=ref(confirmed_path),
        board_packet=ref(packet_paths['native_board_edge']),ice_packet=ref(packet_paths['native_ice_curve']),
        previous_human_authorities_and_arrays_preserved=True,independent_accuracy_acceptance=False)
    write(fact_path,fact)
    print(json.dumps(dict(fact=str(fact_path),board=str(packet_paths['native_board_edge']),ice=str(packet_paths['native_ice_curve']),authority=str(output_authority),
        guard=str(guard_path),cap=str(cap_path),new_stroke_groups=len(new_records),new_vertices=full_total,registration=registration),ensure_ascii=False))


if __name__=='__main__':main()
