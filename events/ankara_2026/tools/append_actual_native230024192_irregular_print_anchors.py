"""Bind same-frame RGB measurements to existing frozen Ad21-23 ink materials.

Human input confirms identities. Pixels are independently measured from source
RGB, and never become directly drawn human truth. No current camera, world,
solver state or model projection enters this tool.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).resolve().parent)]
import cv2
import numpy as np
from measure_user_bad_Ad22_23_frozen_material_20261008 import source_matches
from register_user_native230024192_all_visible import BASE,SOURCE,ref,write,SOURCE_SHA

PARENT=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/observations/vision_training/continuous_review/user_fullvideo_systematic_source_append_20261008/native_six_parent_plus_Ad22_Ad23_bad_time_actual_frozen_ink_ADMITTED_v1.json')
OUT=BASE/'native230024192_actual_RGB_source_candidate'
PACKET=OUT/'native_six_parent_plus_Ad21_22_23_native230024192_actual_frozen_ink_ADMITTED_v1.json'
ROIS={21:((615,368,831,431),(562,355,790,434)),22:((410,355,590,465),(229,348,360,439)),23:((1300,260,1665,380),(0,350,172,435))}


def main():
    cv2.setNumThreads(1)
    if PACKET.exists():raise FileExistsError('Preserve prior admitted source')
    packet=json.loads(PARENT.read_text(encoding='utf8'));parent=deepcopy(packet)
    target=cv2.imread(str(SOURCE));new=[];reviews=[];crops=[]
    for number,(r0,r1)in ROIS.items():
        candidates=[r for r in parent['source_anchor_observations']if r['advertisement_number']==number and r['split']=='fit']
        original=(min(candidates,key=lambda r:abs(r['native_pts']-230024192))if number!=23 else next(r for r in candidates if r['native_pts']==230123008))
        source=cv2.imread(original['source_raw_frame']['path'])
        assert ref(Path(original['source_raw_frame']['path']))==original['source_raw_frame']
        g0,g1,H,evidence=source_matches(source,target,r0,r1,number)
        u=np.asarray(original['uv'],np.float32).reshape(-1,1,2);guess=cv2.perspectiveTransform(u,H)
        # Local RGB search after source-only registration is stable across
        # the print scale change. Final pixels live on the unwarped target.
        warped=cv2.warpPerspective(g0,H,(1920,1080),flags=cv2.INTER_LINEAR)
        options=dict(winSize=(21,21),maxLevel=2,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,40,.001),minEigThreshold=1e-5,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        v,forward,error=cv2.calcOpticalFlowPyrLK(warped,g1,guess.copy(),guess.copy(),**options)
        back,back_status,_=cv2.calcOpticalFlowPyrLK(g1,warped,v.copy(),guess.copy(),**options)
        original_back=cv2.perspectiveTransform(back,np.linalg.inv(H));fb=np.linalg.norm(original_back[:,0]-u[:,0],axis=1)
        he=np.linalg.norm(v[:,0]-guess[:,0],axis=1);x0,y0,x1,y1=r1
        keep=forward.ravel().astype(bool)&back_status.ravel().astype(bool)&(fb<=.75)&(he<=3.)
        keep&=(v[:,0,0]>=x0+2)&(v[:,0,0]<x1-2)&(v[:,0,1]>=y0+3)&(v[:,0,1]<y1-3)
        ids=np.flatnonzero(keep)
        if not len(ids):raise ValueError(f'No reliable visible actual Ad{number} point')
        init=deepcopy(original['initialization']);init['material_coordinates_normalized']=[init['material_coordinates_normalized'][i]for i in ids]
        row=dict(id=f'Ad{number}_user_identity_native230024192_actual_frozen_ink_fit',advertisement_number=number,
            pts=230024192,native_pts=230024192,actual_pts=230024192,time_base='1/15360',split='fit',source_raw_frame=ref(SOURCE),
            uv=v[ids,0].astype(float).tolist(),sigma_px=deepcopy(original['sigma_px']),initialization=init,
            physical_identity_authority=dict(path=str(ROOT/'events/ankara_2026/model_constraints/user_native230024192_all_visible_20261008.json'),kind='direct user same-frame labelled advertisement identity'),
            source_role='actual source RGB same existing named frozen printed-material ink point; not human drawn pixels',
            direct_human_GT=False,source_RGB_actual_measurement=True,model_camera_or_q_projection_not_used_for_UV_selection=True,
            original_HELD_source_pixels_or_tracks_not_promoted=True,new_world_XYZ_or_shape_or_location_parameters_supplied=False,
            actual_old_FIT_same_ink_indices=ids.tolist(),old_FIT_material_reference=dict(parent_packet=ref(PARENT),id=original['id'],
                native_pts=original['native_pts'],original_source_raw_frame=original['source_raw_frame'],source_fields_and_material_coordinates_not_reestimated=True),
            source_only_material_review_pending=False,actual_both_native_RGB_same_ink_components_qualified=True,source_subpixel_precision_not_certified=True,
            source_source_tracking=dict(**evidence,LK_source_preconditioning='actual source RGB homography only; target points measured on unwarped native RGB',
                LK_window_native_px=21,LK_pyramid_maxLevel=2,source_forward_backward_gate_native_px=.75,source_LK_vs_SIFT_H_gate_native_px=3.,
                native_forward_backward_px=fb[ids].tolist(),native_LK_vs_source_H_px=he[ids].tolist()))
        new.append(row)
        contours=OUT/f'Ad{number}_actual_native230024192_visible_external_ink_contours_SOURCE_CANDIDATE.json'
        contour_obj=json.loads(contours.read_text(encoding='utf8'))
        assert contour_obj['world_or_camera_projection_used']is False
        assert contour_obj['source_binding']['source_pts']==230024192
        reviews.append(dict(advertisement_number=number,original_source_frame=original['source_raw_frame'],current_source_frame=ref(SOURCE),
            existing_real_same_frame_print_contours=ref(contours),external_contour_measurement_is_human_drawing=False,
            external_component_count=len(contour_obj['contours']),partial_open_component_count=sum(not c['closed']for c in contour_obj['contours']),
            same_ink_source_coordinates=u[ids,0].astype(float).tolist(),same_ink_target_coordinates=row['uv'],
            original_FIT_indices=ids.tolist(),rejected_original_FIT_indices=np.flatnonzero(~keep).tolist(),
            normalized_material_coordinates_preserved=True,source_original_sigma_preserved=True,human_identity_pixel_measurement_separate=True))
        # Small source-native pair crop provides review of every admitted
        # ink identity; no full diagnostic video or repeated whole frame.
        for image,roi,points,key in [(source,r0,u[ids,0],'old'),(target,r1,v[ids,0],'new')]:
            ax,ay,bx,by=roi;crop=image[ay:by,ax:bx].copy()
            for index,point in enumerate(points):
                p=tuple(np.rint(point-[ax,ay]).astype(int));cv2.circle(crop,p,2,(0,255,255),1)
                cv2.putText(crop,str(index),(p[0]+3,p[1]-3),cv2.FONT_HERSHEY_SIMPLEX,.28,(0,255,0),1)
            head=np.full((24,crop.shape[1],3),240,np.uint8);cv2.putText(head,f'Ad{number} {key} {len(ids)} actual inks',(3,16),0,.38,(10,10,10),1)
            crops.append(np.vstack((head,crop)))
        print(json.dumps(dict(ad=number,original_FIT_points=len(u),new_visible_measured_points=len(ids),kept_original_indices=ids.tolist(),
            source_only_registration=evidence['source_only_initialization_method'],median_FB_native_px=float(np.median(fb[ids])),max_FB_native_px=float(fb[ids].max())),ensure_ascii=False),flush=True)
    packet['source_anchor_observations'].extend(new)
    if 'fit_row_order'in packet:packet['fit_row_order'].extend(r['id']for r in new)
    packet['same_frame_user_identity_actual_RGB_append']=dict(parent_packet=ref(PARENT),source_pts=230024192,
        new_source_ids=[r['id']for r in new],new_measured_FIT_points=sum(len(r['uv'])for r in new),
        old_rows_anchors_templates_and_HELD_exact=True,new_human_drawing_GT=False,model_projection_as_source_measurement=False,
        no_rectangle_or_paint_endpoint_or_hidden_component_invented=True,source_contour_reviews=reviews)
    assert packet['rows']==parent['rows']and packet['frozen_source_print_templates']==parent['frozen_source_print_templates']
    assert packet['source_anchor_observations'][:len(parent['source_anchor_observations'])]==parent['source_anchor_observations']
    qa=OUT/'Ad21_22_23_native230024192_actual_frozen_ink_RGB_review_v1.json'
    write(qa,dict(schema='rinkspace.actual-native-irregular-print-ink-source-review.v1',source_pts=230024192,time_base='1/15360',
        source_video_sha256=SOURCE_SHA,source_raw_frame=ref(SOURCE),parent_packet=ref(PARENT),source_pair_reviews=reviews,
        direct_human_identity_authority=True,direct_human_pixel_truth=False,source_RGB_only_measurement=True,
        frozen_normalized_templates_preserved=True,model_or_world_or_camera_not_loaded=True,HELD_not_promoted_to_FIT=True,accuracy_accepted=False))
    packet['same_frame_user_identity_actual_RGB_append']['review_reference']=ref(qa)
    write(PACKET,packet)
    canvas=np.full((sum(c.shape[0]for c in crops)+5*(len(crops)-1),max(c.shape[1]for c in crops),3),240,np.uint8);y=0
    for crop in crops:canvas[y:y+crop.shape[0],:crop.shape[1]]=crop;y+=crop.shape[0]+5
    view=OUT/'Ad21_22_23_native230024192_actual_frozen_ink_pair_review.png';cv2.imwrite(str(view),canvas)
    fact_path=ROOT/'events/ankara_2026/model_constraints/user_native230024192_all_visible_20261008.json'
    fact=json.loads(fact_path.read_text(encoding='utf8'))
    fact['unboxed_Ad21_22_23_actual_RGB_followup']=dict(packet=ref(PACKET),source_review=ref(qa),
        source_pair_preview=ref(view),source_pts=230024192,new_FIT_points=sum(len(r['uv'])for r in new),
        human_confirmation_scope='advertisement identity only',pixel_measurement_scope='actual native RGB same frozen printed ink',
        direct_human_pixel_truth=False,original_source_HELD_and_templates_preserved=True,
        rectangle_added=False,source_only_contour_references=[r['existing_real_same_frame_print_contours']for r in reviews])
    write(fact_path,fact)
    print(json.dumps(dict(packet=ref(PACKET),source_review=ref(qa),small_actual_RGB_pair_image=ref(view),new_source_groups=3,new_FIT_points=sum(len(r['uv'])for r in new)),ensure_ascii=False))


if __name__=='__main__':main()
