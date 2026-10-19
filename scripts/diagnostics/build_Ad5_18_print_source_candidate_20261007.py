"""Independent next-target source candidate; frozen six-ad v1 is untouched.

The old four-corner MEIWA internal graphic remains an actual old material.
These free ink components describe the wider visible print group separately.
Ad18 gets no copied camera, world point, rectangle, or unsupported anchor.
"""
from copy import deepcopy
from pathlib import Path
import hashlib,json
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DATA=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
OUT=DATA/'observations/vision_training/continuous_review/Ad4_17_18_19_source_absolute_20261007'
PARENT=DATA/'diagnostics/irregular_six_ads_source_revisit_20261007/native_board_tracks_plus_six_irregular_ads_source_FIT_HELD_v1.json'
FRAMES=DATA/'programmes/okada_mei_sp/results/aligned_complete_20261007_478afaf77615/nine_shot_motion_regression_v2/native_board_motion_source_frames'
P0,P1=230338304,230330624
ROI=[1124,365,1890,510]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_path(pts):
    p=OUT/f'raw_native_{pts}.png'
    return p if p.exists()else FRAMES/f'native_{pts}.png'
def ref(pts):
    p=source_path(pts);return dict(path=str(p),sha256=sha(p))

def main():
    parent_bytes=PARENT.read_bytes();packet=json.loads(parent_bytes)
    packet['parent_source_packet']=dict(path=str(PARENT),sha256=sha(PARENT))
    packet['parent_source_admission']=packet.pop('source_admission')
    raw=cv2.imread(str(source_path(P0)));target=cv2.imread(str(source_path(P1)))
    gray=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY);g1=cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)
    x0,y0,x1,y1=ROI;region=gray[y0:y1,x0:x1]
    threshold=float(np.quantile(region,.85))-16.
    contours,h=cv2.findContours((region<threshold).astype(np.uint8)*255,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    external=[];holes=[];clipped=[]
    for i,c in enumerate(contours):
        if abs(cv2.contourArea(c))<3. or len(c)<7:continue
        points=c[:,0]
        if np.any((points[:,0]<=0)|(points[:,0]>=x1-x0-1)|(points[:,1]<=0)|(points[:,1]>=y1-y0-1)):
            clipped.append(dict(component=i,reason='actual component touches crop; closing edge not fabricated'));continue
        depth=0;ancestor=h[0,i,3]
        while ancestor>=0:depth+=1;ancestor=h[0,ancestor,3]
        hole=depth%2==1
        uv=cv2.approxPolyDP(c,.6,True)[:,0]+[x0,y0]
        item=dict(id=f'Ad5:whole_print_component{i}',uv=uv.astype(float).tolist(),closed=True,
            role='print_internal_hole'if hole else'actual_printed_component_external_contour',
            physical_shape='free_polyline',source_native_pixel_uncertainty_px=3.,
            measurement='native ink luminance boundary candidate, not a subpixel ground truth')
        (holes if hole else external).append(item)
    mask=np.zeros(gray.shape,np.uint8);mask[y0:y1,x0:x1]=255
    sift=cv2.SIFT_create(nfeatures=5000,contrastThreshold=.02)
    k0,d0=sift.detectAndCompute(gray,mask);k1,d1=sift.detectAndCompute(g1,None)
    matcher=cv2.BFMatcher();forward=matcher.knnMatch(d0,d1,k=2);reverse=matcher.knnMatch(d1,d0,k=2)
    back={m.queryIdx:m.trainIdx for m,n in reverse if m.distance<.72*n.distance}
    pairs=sorted([(m.queryIdx,m.trainIdx,float(m.distance))for m,n in forward
        if m.distance<.72*n.distance and back.get(m.trainIdx)==m.queryIdx],key=lambda r:r[2])
    unique=[]
    for i,j,distance in pairs:
        u=np.asarray(k0[i].pt);v=np.asarray(k1[j].pt)
        if any(np.linalg.norm(u-a[0])<1 or np.linalg.norm(v-a[1])<1 for a in unique):continue
        unique.append((u,v,distance))
    u=np.float32([r[0]for r in unique]);v=np.float32([r[1]for r in unique])
    if len(u)<8:raise ValueError('Insufficient actual same-print temporal texture')
    source_H,keep=cv2.findHomography(u,v,cv2.RANSAC,2.,maxIters=2000,confidence=.999)
    if keep is None:raise ValueError('Source consistency failed')
    keep=keep.ravel().astype(bool);u=u[keep];v=v[keep]
    held=np.arange(len(u))%3==0
    if (~held).sum()<6 or held.sum()<3:raise ValueError('Actual FIT and development-held points required')
    # This is a source-only consistency check. Both splits were involved in
    # pre-split selection above, so HELD must retain its development identity.
    fit_H,_=cv2.findHomography(u[~held],v[~held],0)
    source_errors=np.linalg.norm(cv2.perspectiveTransform(u[None],fit_H)[0]-v,axis=1)
    corners=np.float32([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    target_box=cv2.perspectiveTransform(corners[None],source_H)[0]
    target_origin=np.maximum(np.floor(target_box.min(axis=0)-20).astype(int),0)
    target_stop=np.minimum(np.ceil(target_box.max(axis=0)+20).astype(int),[target.shape[1],target.shape[0]])
    source_crop=raw[y0:y1,x0:x1];target_crop=target[target_origin[1]:target_stop[1],target_origin[0]:target_stop[0]]
    review_paths=[]
    for start in range(0,len(u),20):
        left=cv2.resize(source_crop,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
        right=cv2.resize(target_crop,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
        montage=np.full((max(left.shape[0],right.shape[0])+44,left.shape[1]+right.shape[1]+24,3),245,np.uint8)
        montage[44:44+left.shape[0],:left.shape[1]]=left
        offset=left.shape[1]+24;montage[44:44+right.shape[0],offset:]=right
        cv2.putText(montage,f'native {P0} -> {P1}, pairs {start}..{min(start+19,len(u)-1)}',(8,27),cv2.FONT_HERSHEY_SIMPLEX,.7,(30,30,30),2,cv2.LINE_AA)
        for j in range(start,min(start+20,len(u))):
            color=(255,0,200)if held[j]else(0,220,70)
            a=np.rint(2*(u[j]-[x0,y0])+[0,44]).astype(int)
            b=np.rint(2*(v[j]-target_origin)+[offset,44]).astype(int)
            for point in (a,b):
                cv2.circle(montage,tuple(point),5,color,1,cv2.LINE_AA)
                cv2.putText(montage,str(j),tuple(point+[4,-4]),cv2.FONT_HERSHEY_SIMPLEX,.38,color,1,cv2.LINE_AA)
        path=OUT/f'Ad5_whole_print_temporal_correspondences_{start:02d}_{start+19:02d}_CANDIDATE.png'
        cv2.imwrite(str(path),montage);review_paths.append(str(path))
    review=dict(source_pts=P0,target_pts=P1,source_frame=ref(P0),target_frame=ref(P1),
        source_H_is_only_RGB_consistency_not_model_accuracy=True,
        development_held_screened_before_split_and_not_independent_validation=True,
        source_only_fit_H_uv_error_px={split:dict(count=int(selected.sum()),median=float(np.median(source_errors[selected])),max=float(source_errors[selected].max()))for split,selected in [('fit',~held),('development_held',held)]},
        correspondence_review_images=review_paths,review_status='original RGB visual review pending')
    (OUT/'Ad5_whole_print_source_only_correspondence_review_CANDIDATE.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for split,selected in [('fit',~held),('holdout',held)]:
        packet['rows'].append(dict(id=f'Ad5_whole_print_{P0}_{P1}_{split}',advertisement_id=5,
            surface_id='boards_inner_surface',same_static_material_confirmed=True,
            source_pts=P0,target_pts=P1,source_frame=ref(P0),target_frame=ref(P1),
            uv0=u[selected].astype(float).tolist(),uv1=v[selected].astype(float).tolist(),sigma_px=6.,split=split,
            source_meaning='real same-instance printed texture; not same brand identity alone or nearest outline',
            selection='source-only mutual SIFT and pre-split source projective screening; original RGB review pending',
            holdout_role='development holdout from solver FIT; not untouched source generalization validation'))
    packet['source_conditioned_print_contours'].append(dict(advertisement_id=5,anchor_pts=P0,anchor_frame=ref(P0),
        identity_authority='Existing original Ad5 source identity plus same-event neighboring prints and user number/content catalogue; recheck before new-target admission',
        identity_catalog=str(ROOT/'configs/rink_model/ankara_2026_advertisements_user_confirmed_20260924.json'),
        source_only_navigation_roi=ROI,navigation_roi_is_geometry=False,
        external_contours=external,internal_holes_recorded_separately=holes,unknown_or_clipped_components=clipped,
        current_camera_course_conditioned_shape_not_measured_XYZ=True,
        filled_rectangle_circle_or_symmetry_shape_assumption=False,accuracy_accepted=False,
        whole_print_scope_distinct_from_retained_old_internal_blue_white_green_graphic=True,
        source_print_components=['40th anniversary ink','blue/white/green internal MEIWA graphic','black Japanese text']))
    packet['confirmed_same_print_pairs_without_current_counterpart_anchor']=[dict(numbers=[5,18],
        same_print_dimensions_authority=dict(path=str(ROOT/'events/ankara_2026/model_constraints/user_advertisement_pairs_20261005.json'),
            sha256=sha(ROOT/'events/ankara_2026/model_constraints/user_advertisement_pairs_20261005.json')),
        Ad18_source_in_current_programme_not_yet_found=True,Ad18_contour_or_camera_or_world_not_copied=True)]
    packet['review_pending_before_production_admission']=True
    packet['new_candidate_scope']='Next-target v2 only; old internal Ad5/18 material and all frozen v1 rows remain untouched. No current counterpart18 source admitted.'
    preview=raw[y0:y1,x0:x1].copy()
    for c in external:cv2.polylines(preview,[np.rint(np.asarray(c['uv'])-[x0,y0]).astype(np.int32)],True,(0,255,255),1,cv2.LINE_AA)
    cv2.imwrite(str(OUT/f'Ad5_whole_print_native_{P0}_contour_CANDIDATE.png'),cv2.resize(preview,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST))
    path=OUT/'native_board_tracks_six_v1_plus_Ad5_whole_print_v2_CANDIDATE.json'
    path.write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    assert PARENT.read_bytes()==parent_bytes
    print(json.dumps(dict(path=str(path),sha256=sha(path),Ad5_components=len(external),holes=len(holes),
        fit=int((~held).sum()),development_held=int(held.sum()),Ad18_native_anchor_found=False)))

if __name__=='__main__':main()
