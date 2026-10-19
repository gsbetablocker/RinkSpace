"""Source-only free print contours and independent material correspondences.

All navigation below was read from native video and the user-confirmed print
order. No model projection or world coordinates enter this source packet.
"""
from pathlib import Path
from copy import deepcopy
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'calibration/program_tracking')]
import cv2
import numpy as np
from calibration.program_tracking.render_result import decode_selected

DATA=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
OUT=DATA/'diagnostics/irregular_six_ads_source_revisit_20261007'
SOURCES=[(8,231013120,[180,528,557,636]),(9,229387520,[1380,500,1608,677]),
    (10,230379776,[1102,451,1775,647]),(21,230022144,[615,368,831,431]),
    (22,230066432,[1630,291,1815,416]),(23,230123008,[1310,266,1648,363])]
PARENT=DATA/'programmes/okada_mei_sp/results/aligned_complete_20261007_478afaf77615/nine_shot_motion_regression_v2/native_board_motion_all_nine_shots_typed_source.json'
AUTHORITY=ROOT/'events/ankara_2026/model_constraints/user_advertisement_pairs_20261005.json'
# The clearer actual print supplies its free contour; its counterpart keeps
# its independently observed physical position and its own native evidence.
PRINT_PAIRS=[(8,21),(22,9),(10,23)]


def append_shared_print_material(packet):
    """Match real print texture across confirmed identical printed instances.

    A source-only homography qualifies descriptors; it is never a world or
    print shape input. Repeated SIFT orientations at the same pixel are one
    material point. The placement fiducial is excluded from FIT and HELD.
    Pre-split descriptor screening sees all points, so HELD is a development
    holdout from solver FIT, not untouched generalization validation.
    """
    facts=json.loads(AUTHORITY.read_text(encoding='utf8'))['constraint']
    if not (facts['same_printed_template'] and facts['shared_printed_dimensions']
            and facts['physical_locations_independent']):
        raise ValueError('Explicit user shared-print dimensions required')
    sources={n:(pts,roi)for n,pts,roi in SOURCES}
    sift=cv2.SIFT_create(nfeatures=5000,contrastThreshold=.02)
    matcher=cv2.BFMatcher(cv2.NORM_L2);cache={};audit=[]
    def features(number):
        if number not in cache:
            pts,roi=sources[number];raw=cv2.imread(str(OUT/f'raw_{pts}.png'))
            gray=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY);mask=np.zeros(gray.shape,np.uint8)
            x0,y0,x1,y1=roi;mask[y0:y1,x0:x1]=255
            keypoints,descriptors=sift.detectAndCompute(gray,mask)
            cache[number]=(raw,keypoints,descriptors)
        return cache[number]
    packet['shared_print_templates']=[]
    for canonical,counterpart in PRINT_PAIRS:
        p0,roi0=sources[canonical];p1,roi1=sources[counterpart]
        raw0,k0,d0=features(canonical);raw1,k1,d1=features(counterpart)
        forward=matcher.knnMatch(d0,d1,k=2);reverse=matcher.knnMatch(d1,d0,k=2)
        back={m.queryIdx:m.trainIdx for m,n in reverse if m.distance<.75*n.distance}
        pairs=sorted([(m.queryIdx,m.trainIdx,float(m.distance))for m,n in forward
            if m.distance<.75*n.distance and back.get(m.trainIdx)==m.queryIdx],key=lambda r:r[2])
        # Different detector orientations do not create independent samples.
        unique=[]
        for i,j,distance in pairs:
            u=np.asarray(k0[i].pt);v=np.asarray(k1[j].pt)
            if any(np.linalg.norm(u-old[0])<1. or np.linalg.norm(v-old[1])<1.
                   for old in unique):continue
            unique.append((u,v,distance))
        u=np.float32([r[0]for r in unique]);v=np.float32([r[1]for r in unique])
        if len(u)<7:raise ValueError(f'Insufficient unique source print points: {canonical}/{counterpart}')
        _,keep=cv2.findHomography(u,v,cv2.RANSAC,2.,maxIters=2000,confidence=.999)
        if keep is None:raise ValueError(f'No independent source consistency: {canonical}/{counterpart}')
        ids=np.flatnonzero(keep.ravel());u=u[ids];v=v[ids]
        if len(u)<7:raise ValueError(f'Insufficient source inliers: {canonical}/{counterpart}')
        # Lowest mutual descriptor distance among qualified points gives a
        # real corresponding internal print fiducial, not an invented center.
        fiducial0=u[0].copy();fiducial1=v[0].copy();u=u[1:];v=v[1:]
        held=np.arange(len(u))%3==0
        if (~held).sum()<4 or held.sum()<2:raise ValueError('Real independent shared-print split required')
        H,_=cv2.findHomography(u[~held],v[~held],0)
        if H is None:raise ValueError('Degenerate source-only print correspondence')
        error=np.linalg.norm(cv2.perspectiveTransform(u[held,None,:],H)[:,0]-v[held],axis=1)
        pair_id=f'Ad{canonical}_Ad{counterpart}_same_user_confirmed_print'
        template=dict(id=pair_id,canonical_advertisement_id=canonical,counterpart_advertisement_id=counterpart,
            canonical_anchor_pts=p0,counterpart_anchor_pts=p1,
            canonical_anchor_frame=ref(p0),counterpart_anchor_frame=ref(p1),
            canonical_fiducial_uv=fiducial0.astype(float).tolist(),counterpart_fiducial_uv=fiducial1.astype(float).tolist(),
            fiducial_role='same real mutually matched internal printed-material keypoint; excluded from source FIT/HELD',
            shared_dimensions_authority=dict(path=str(AUTHORITY),sha256=sha(AUTHORITY)),
            identical_print_template_and_dimensions=True,physical_locations_independent=True,
            developed_station_orientation=1,free_outline_from_actual_canonical_native_frame=True,
            geometry_or_camera_projection_used_to_select_source=False,
            pixel_measurement_accuracy_accepted=False)
        packet['shared_print_templates'].append(template)
        for split,selected in [('fit',~held),('holdout',held)]:
            packet['rows'].append(dict(id=f'{pair_id}_native_material_{split}',advertisement_id=canonical,
                counterpart_advertisement_id=counterpart,shared_print_pair_id=pair_id,
                surface_id='boards_inner_surface',same_static_material_confirmed=False,
                same_print_template_material_confirmed=True,source_pts=p0,target_pts=p1,
                source_frame=ref(p0),target_frame=ref(p1),uv0=u[selected].astype(float).tolist(),
                uv1=v[selected].astype(float).tolist(),sigma_px=6.,split=split,
                holdout_role='development source holdout; never included in solver FIT; source descriptor screening used pre-split points',
                selection='source-only mutual SIFT, deduplicated native pixel locations and original RGB review; identical real print material across distinct instances',
                source_meaning='same named ink material in the user-confirmed shared printed template; not same physical board and not closest outline'))
        # Diagnostic crops retain original native pixels at integer magnification.
        crops=[]
        for raw,roi,points,fiducial in [(raw0,roi0,u,fiducial0),(raw1,roi1,v,fiducial1)]:
            x0,y0,x1,y1=roi;crop=raw[y0:y1,x0:x1].copy()
            for i,point in enumerate(points):
                pos=tuple(np.rint(point-[x0,y0]).astype(int));colour=(255,0,255)if held[i]else(0,255,0)
                cv2.circle(crop,pos,2,colour,-1);cv2.putText(crop,str(i),pos,cv2.FONT_HERSHEY_SIMPLEX,.3,colour,1,cv2.LINE_AA)
            cv2.drawMarker(crop,tuple(np.rint(fiducial-[x0,y0]).astype(int)),(0,255,255),cv2.MARKER_CROSS,8,1)
            crops.append(cv2.resize(crop,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST))
        canvas=np.zeros((sum(c.shape[0]for c in crops)+16,max(c.shape[1]for c in crops),3),np.uint8)
        offset=0
        for crop in crops:canvas[offset:offset+crop.shape[0],:crop.shape[1]]=crop;offset+=crop.shape[0]+16
        cv2.imwrite(str(OUT/f'{pair_id}_native_correspondence_review.png'),canvas)
        audit.append(dict(pair_id=pair_id,unique_native_material_points=len(u),fit=int((~held).sum()),held=int(held.sum()),
            source_only_projective_held_error_px=error.astype(float).tolist(),
            homography_is_descriptor_screen_only_not_physical_geometry=True,
            fiducial_excluded_from_fit_and_held=True,
            held_is_untouched_generalization_validation=False,original_RGB_review_pending=True))
    packet['shared_print_material_source_audit']=audit

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ref(pts):
    path=OUT/f'raw_{pts}.png'
    return dict(path=str(path),sha256=sha(path))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    shots=json.loads((ROOT/'calibration/program_tracking/source_shots.json').read_text(encoding='utf8'))['shots'][1:10]
    paired={}
    for number,pts,roi in SOURCES:
        shot=next(s for s in shots if s['start_pts']<=pts<s['end_pts'])
        paired[pts]=[p for p in [pts-6144,pts+6144,pts+12288]if shot['start_pts']<=p<shot['end_pts']]
    stamps=sorted({p for pts,targets in paired.items()for p in [pts,*targets]})
    missing=[p for p in stamps if not (OUT/f'raw_{p}.png').exists()]
    for pts,raw in decode_selected(DATA/'media/originals/oxXu1avhrvs.mp4',missing):
        if not cv2.imwrite(str(OUT/f'raw_{pts}.png'),raw):raise OSError(pts)
    packet=json.loads(PARENT.read_text(encoding='utf8'))
    packet['parent_source_packet']=dict(path=str(PARENT),sha256=sha(PARENT))
    packet['source_conditioned_print_contours']=[]
    sift=cv2.SIFT_create(nfeatures=5000,contrastThreshold=.025,edgeThreshold=12)
    cache={}
    def features(pts):
        if pts not in cache:
            raw=cv2.imread(str(OUT/f'raw_{pts}.png'));gray=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)
            kp,d=sift.detectAndCompute(gray,None)
            cache[pts]=(raw,gray,kp,d)
        return cache[pts]
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    audits=[]
    for number,pts,roi in SOURCES:
        raw,gray,kp,d=features(pts);x0,y0,x1,y1=roi;region=gray[y0:y1,x0:x1]
        background=float(np.quantile(region,.85));threshold=background-16.
        mask=(region<threshold).astype(np.uint8)*255
        contours,h=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        outline=[];holes=[];skipped=[]
        for ci,c in enumerate(contours):
            points=c[:,0];area=abs(cv2.contourArea(c))
            if area<3 or len(points)<7:continue
            touches=np.any((points[:,0]<=0)|(points[:,0]>=x1-x0-1)|(points[:,1]<=0)|(points[:,1]>=y1-y0-1))
            if touches:
                skipped.append(dict(component=ci,reason='navigation_crop_touched; no fabricated closing edge'))
                continue
            # This selects original contour vertices; it does not fit a shape.
            vertices=cv2.approxPolyDP(c,.6,True)[:,0]+[x0,y0]
            record=dict(id=f'Ad{number}:component{ci}',uv=vertices.astype(float).tolist(),closed=True,
                role='print_internal_hole' if h[0,ci,3]>=0 else 'actual_printed_component_external_contour',
                extraction='native luminance ink component boundary; engineering blur uncertainty',
                source_native_pixel_uncertainty_px=3.,physical_shape='free_polyline')
            (holes if h[0,ci,3]>=0 else outline).append(record)
        item=dict(advertisement_id=number,anchor_pts=pts,anchor_frame=ref(pts),
            identity_authority='actual neighbouring physical print order and user-confirmed 1..26 catalogue',
            identity_catalog=str(ROOT/'configs/rink_model/ankara_2026_advertisements_user_confirmed_20260924.json'),
            source_only_navigation_roi=roi,navigation_roi_is_geometry=False,
            external_contours=outline,internal_holes_recorded_separately=holes,
            threshold_gray=threshold,unknown_or_clipped_components=skipped,
            current_camera_course_conditioned_shape_not_measured_XYZ=True,
            filled_rectangle_circle_or_symmetry_shape_assumption=False,
            accuracy_accepted=False)
        packet['source_conditioned_print_contours'].append(item)
        preview=raw[y0:y1,x0:x1].copy()
        for c in outline:cv2.polylines(preview,[np.rint(np.array(c['uv'])-[x0,y0]).astype(np.int32)],True,(0,255,255),1,cv2.LINE_AA)
        cv2.imwrite(str(OUT/f'Ad{number}_source_outer_contour_review.png'),preview)
        source_indices=np.array([i for i,k in enumerate(kp)if x0+3<k.pt[0]<x1-3 and y0+3<k.pt[1]<y1-3],int)
        for target in paired[pts]:
            _,g1,k1,d1=features(target)
            audit=dict(advertisement_id=number,source_pts=pts,target_pts=target,source_keypoints=len(source_indices))
            if len(source_indices)<8 or d1 is None:
                audit['failure']='insufficient source-only print texture';audits.append(audit);continue
            forward=matcher.knnMatch(d[source_indices],d1,k=2)
            reverse=matcher.knnMatch(d1,d[source_indices],k=2)
            backward={m.queryIdx:m.trainIdx for m,n in reverse if m.distance<.72*n.distance}
            matched=[(source_indices[m.queryIdx],m.trainIdx,m.distance)for m,n in forward
                if m.distance<.72*n.distance and backward.get(m.trainIdx)==m.queryIdx]
            unique=[]
            for i,j,distance in sorted(matched,key=lambda r:r[2]):
                if any(np.linalg.norm(np.asarray(kp[i].pt)-kp[a].pt)<1.
                       or np.linalg.norm(np.asarray(k1[j].pt)-k1[b].pt)<1.
                       for a,b,_ in unique):continue
                unique.append((i,j,distance))
            matched=unique[:80]
            u=np.float32([kp[i].pt for i,j,_ in matched]);v=np.float32([k1[j].pt for i,j,_ in matched])
            if len(u)<8:
                audit['failure']='insufficient mutual true print descriptors';audits.append(audit);continue
            H,keep=cv2.findHomography(u,v,cv2.RANSAC,2.,maxIters=1000,confidence=.999)
            if H is None:
                audit['failure']='source-only geometric consistency failed';audits.append(audit);continue
            keep=keep.ravel().astype(bool);u=u[keep];v=v[keep]
            # The check is independent of rink geometry. Keep every third
            # ordered matched point as HELD, never use it in source FIT.
            held=np.arange(len(u))%3==0
            if (~held).sum()<6 or held.sum()<3:
                audit['failure']='too few consistent print points';audits.append(audit);continue
            Hfit,_=cv2.findHomography(u[~held],v[~held],0)
            error=np.linalg.norm(cv2.perspectiveTransform(u[held,None,:],Hfit)[:,0]-v[held],axis=1)
            if float(np.median(error))>1.5:
                audit['failure']='independent source-only HELD residual too large';audits.append(audit);continue
            for split,selected in [('fit',~held),('holdout',held)]:
                packet['rows'].append(dict(id=f'Ad{number}_native_material_{pts}_{target}_{split}',
                    advertisement_id=number,surface_id='boards_inner_surface',same_static_material_confirmed=True,
                    source_pts=pts,target_pts=target,source_frame=ref(pts),target_frame=ref(target),
                    uv0=u[selected].astype(float).tolist(),uv1=v[selected].astype(float).tolist(),
                    sigma_px=6.,split=split,
                    holdout_role='development source holdout; never included in solver FIT; descriptor/RANSAC screening and source-only held review used for admission',
                    selection='source-only mutual SIFT and source projective-consistency; same visible printed instance',
                    source_meaning='matched printed material texture points; not nearest points on an outline'))
            audit.update(points=len(u),fit=int((~held).sum()),held=int(held.sum()),source_H_held_median_px=float(np.median(error)))
            audits.append(audit)
    packet['irregular_six_ad_source_audit']=audits
    append_shared_print_material(packet)
    packet['review_pending_before_production_admission']=True
    target=OUT/'native_board_tracks_plus_six_irregular_ads_source_CANDIDATE.json'
    target.write_text(json.dumps(packet,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(path=str(target),contours=[dict(number=r['advertisement_id'],components=len(r['external_contours']),holes=len(r['internal_holes_recorded_separately']))for r in packet['source_conditioned_print_contours']],tracks=audits),ensure_ascii=False))

if __name__=='__main__':main()
