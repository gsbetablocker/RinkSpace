"""Source-only new native anchors for existing irregular printed materials.

Use original FIT anchor ink identities and frozen normalized coordinates.
Mutual SIFT initializes deep LK; no camera, q, world ray or geometry is read.
Every point remains pending until its two native RGB endpoints are viewed.
"""
from copy import deepcopy
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'src')];sys.dont_write_bytecode=True
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np
from measure_user_multitime_board_edges_20261008 import OUT,EVIDENCE,SOURCE_SHA,ref,read,write_new
cv2.setNumThreads(1)
PARENT=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/irregular_six_ads_source_revisit_20261007/native_board_tracks_six_frozen_normalized_shapes_source_anchors_v2_CANDIDATE.json')
TARGET=EVIDENCE/'06_Ad22_Ad23_large_shift_source_native_230036992.png'
ROIS={22:((1625,280,1830,425),(410,355,590,465)),23:((1300,260,1665,380),(75,365,350,450))}


def source_matches(a,b,r0,r1,number):
    g0,g1=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
    masks=[]
    for roi in(r0,r1):
        mask=np.zeros(g0.shape,np.uint8);x0,y0,x1,y1=roi;mask[y0:y1,x0:x1]=255;masks.append(mask)
    sift=cv2.SIFT_create(nfeatures=1500,contrastThreshold=.008,edgeThreshold=15)
    k0,d0=sift.detectAndCompute(g0,masks[0]);k1,d1=sift.detectAndCompute(g1,masks[1])
    if d0 is None or d1 is None:raise ValueError('No source same-print descriptor support')
    matcher=cv2.BFMatcher();fwd=matcher.knnMatch(d0,d1,k=2);back=matcher.knnMatch(d1,d0,k=2)
    reverse={m.queryIdx:m.trainIdx for m,n in back if m.distance<.78*n.distance}
    chosen=[m for m,n in fwd if m.distance<.78*n.distance and reverse.get(m.trainIdx)==m.queryIdx]
    # Multiple SIFT orientations at one point are one source datum.
    unique=[]
    for m in sorted(chosen,key=lambda x:x.distance):
        u=np.asarray(k0[m.queryIdx].pt);v=np.asarray(k1[m.trainIdx].pt)
        if any(np.linalg.norm(u-p[0])<1 or np.linalg.norm(v-p[1])<1 for p in unique):continue
        unique.append((u,v,float(m.distance)))
    assert len(unique)>=5,(len(k0),len(k1),len(unique))
    u=np.asarray([r[0]for r in unique],np.float64);v=np.asarray([r[1]for r in unique],np.float64)
    H,keep=cv2.findHomography(u,v,cv2.RANSAC,2.,maxIters=4000,confidence=.999)
    method='source_projective_consensus'
    if H is None or int(keep.sum())<5:
        # A blurred small shield has too few descriptor landmarks for eight
        # projective DOF. A four-DOF source similarity only initializes the
        # independently checked native LK; it does not define final pixels.
        A,keep=cv2.estimateAffinePartial2D(u,v,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=4000,confidence=.999)
        if A is not None and int(keep.sum())>=4:
            H=np.vstack((A,[0.,0.,1.]));method='source_similarity_consensus_initialization_only'
        else:
            assert number==22,'Insufficient direct descriptor correspondence for this source'
            # Exact source colour extents of the distinctive red shield give
            # a source-only image-registration initialization. No old model
            # silhouette or solved camera enters this registration.
            boxes=[]
            for image,roi in((a,r0),(b,r1)):
                x0,y0,x1,y1=roi;hsv=cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2HSV)
                mask=(((hsv[:,:,0]<13)|(hsv[:,:,0]>165))&(hsv[:,:,1]>30)&(hsv[:,:,2]<245)).astype(np.uint8)
                n,labels,stats,_=cv2.connectedComponentsWithStats(mask)
                label=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));yy,xx=np.where(labels==label)
                boxes.append([int(xx.min()+x0-7),int(yy.min()+y0-7),int(xx.max()+x0+8),int(yy.max()+y0+8)])
            p,q=boxes;x0,y0,x1,y1=p;tx0,ty0,tx1,ty1=q
            template=g0[y0:y1,x0:x1].astype(np.float32)/255.
            image=cv2.resize(g1[ty0:ty1,tx0:tx1],(x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.
            score,warp=cv2.findTransformECC(template,image,np.eye(3,dtype=np.float32),cv2.MOTION_HOMOGRAPHY,
                (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,120,1e-6),None,5)
            assert score>.85,score
            old_to_crop=np.array([[1.,0.,-x0],[0.,1.,-y0],[0.,0.,1.]])
            crop_to_native=np.array([[(tx1-tx0)/(x1-x0),0.,tx0],[0.,(ty1-ty0)/(y1-y0),ty0],[0.,0.,1.]])
            H=crop_to_native@warp@old_to_crop
            method='whole_actual_red_shield_ECC_source_registration_initialization_only'
            registration=dict(source_colour_extent_ROIs=boxes,photometric_ECC=float(score),
                normalized_crop_warp_matrix=warp.tolist(),registered_image_pixels_not_used_as_measurement=True)
    report=dict(mutual_ratio=.78,mutual_unique_descriptor_matches=len(unique),
        source_only_projective_inliers=int(keep.sum()),source_only_initialization_method=method,source_only_H=H.tolist(),
        source_RGB_matches=[dict(source_uv=a.tolist(),target_uv=b.tolist(),descriptor_distance=d,inlier=bool(k))for(a,b,d),k in zip(unique,keep.ravel())],
        H_role='RGB point-search initialization and independent same-material consistency only; not measured world or per-state model warp')
    if method.startswith('whole_actual'):report['source_only_photometric_registration']=registration
    return g0,g1,H,report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--authority',required=True);ap.add_argument('--only',type=int,choices=[22,23]);args=ap.parse_args();assert args.authority.strip()
    OUT.mkdir(parents=True,exist_ok=True);old_bytes=PARENT.read_bytes();parent=read(PARENT)
    target=cv2.imread(str(TARGET));assert target.shape==(1080,1920,3)
    aliases_path=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_unlocked_geometry_af8cb_board739_canonical_gate_20261008/complete_physical_camera_model_UNACCEPTED_a90e01e5d788434d.npz')
    with np.load(aliases_path,allow_pickle=False)as z:
        aliases=dict(zip(map(int,z['pts']),map(int,z['exposure_representative_pts'])))
    new=[];reviews=[];sheets=[]
    for number,(r0,r1) in ROIS.items():
        if args.only is not None and number!=args.only:continue
        original=next(r for r in parent['source_anchor_observations']if r['advertisement_number']==number and r['split']=='fit')
        assert ref(Path(original['source_raw_frame']['path']))==original['source_raw_frame']
        source=cv2.imread(original['source_raw_frame']['path']);assert source.shape==(1080,1920,3)
        g0,g1,H,evidence=source_matches(source,target,r0,r1,number)
        u=np.asarray(original['uv'],np.float32).reshape(-1,1,2)
        guess=cv2.perspectiveTransform(u,H).astype(np.float32)
        options=dict(winSize=(21,21),maxLevel=5,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001),minEigThreshold=1e-5,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        if number==23:
            # The two real source views are 1200px apart and differ strongly
            # in print scale. Compensate only the source-image registration
            # for local point search; final UV is read on the native target.
            # Forward/backward consistency is mapped back to the old native
            # source coordinates, retaining the same .75px original gate.
            warped_source=cv2.warpPerspective(g0,H,(g1.shape[1],g1.shape[0]),flags=cv2.INTER_LINEAR)
            local_options={**options,'maxLevel':2}
            v,forward,error=cv2.calcOpticalFlowPyrLK(warped_source,g1,guess.copy(),guess.copy(),**local_options)
            registered_back,status_back,error_back=cv2.calcOpticalFlowPyrLK(g1,warped_source,v.copy(),guess.copy(),**local_options)
            back=cv2.perspectiveTransform(registered_back,np.linalg.inv(H)).astype(np.float32)
            evidence['LK_source_preconditioning']='Source-RGB-only projective image resampling for native target local point search; maxLevel2 for residual local search; no model/camera/world input; final UV on unwarped native target'
        else:
            v,forward,error=cv2.calcOpticalFlowPyrLK(g0,g1,u,guess.copy(),**options)
            back,status_back,error_back=cv2.calcOpticalFlowPyrLK(g1,g0,v,u.copy(),**options)
        fb=np.linalg.norm(back[:,0]-u[:,0],axis=1);h_error=np.linalg.norm(v[:,0]-guess[:,0],axis=1)
        x0,y0,x1,y1=r1
        keep=forward.ravel().astype(bool)&status_back.ravel().astype(bool)&(fb<=.75)&(h_error<=3.)
        keep&=(v[:,0,0]>=x0+5)&(v[:,0,0]<x1-5)&(v[:,0,1]>=y0+5)&(v[:,0,1]<y1-5)
        indices=np.flatnonzero(keep)
        print(json.dumps(dict(source_only_stage='native_LK_forward_backward_checked',advertisement_number=number,
            kept=len(indices),total=len(u),registration=evidence,forward_count=int(forward.sum()),
            backward_count=int(status_back.sum()),FB_native_px_summary=np.quantile(fb,[0,.5,1]).tolist(),
            LK_vs_source_H_px_summary=np.quantile(h_error,[0,.5,1]).tolist(),
            guessed_UV_extent=np.vstack((guess[:,0].min(axis=0),guess[:,0].max(axis=0))).tolist(),
            measured_UV_extent=np.vstack((v[:,0].min(axis=0),v[:,0].max(axis=0))).tolist())),flush=True)
        # Each point is an existing physical ink identity, so a small number
        # of verified points is still a valid source measurement. No minimum
        # frame feature count is a precision gate or a user requirement.
        assert len(indices)>0,(number,len(indices))
        init=deepcopy(original['initialization']);init['material_coordinates_normalized']=[init['material_coordinates_normalized'][i]for i in indices]
        row=dict(id=f'Ad{number}_user_bad_native230036992_actual_frozen_ink_anchor_fit',split='fit',
            advertisement_number=number,pts=230036992,native_pts=230036992,actual_pts=aliases[230036992],time_base='1/15360',
            source_raw_frame=ref(TARGET),uv=v[indices,0].astype(float).tolist(),sigma_px=deepcopy(original['sigma_px']),
            initialization=init,source_role='actual same printed-material point at newly measured native source exposure',
            actual_old_FIT_same_ink_indices=indices.tolist(),old_FIT_material_reference=dict(parent_packet=ref(PARENT),id=original['id'],
                native_pts=original['native_pts'],original_source_raw_frame=deepcopy(original['source_raw_frame']),
                source_fields_and_material_coordinates_not_reestimated=True),
            original_HELD_source_pixels_or_tracks_not_promoted=True,model_camera_or_q_projection_not_used_for_UV_selection=True,
            new_world_XYZ_or_shape_or_location_parameters_supplied=False,source_only_material_review_pending=True,
            source_source_tracking=dict(**evidence,LK_window_native_px=21,LK_pyramid_maxLevel=(2 if number==23 else 5),
                source_forward_backward_gate_native_px=.75,source_LK_vs_SIFT_H_gate_native_px=3.,
                native_forward_backward_px=fb[indices].tolist(),native_LK_vs_source_H_px=h_error[indices].tolist()))
        new.append(row)
        records=[dict(old_FIT_anchor_point_index=int(i),source_UV=u[i,0].astype(float).tolist(),target_UV=v[i,0].astype(float).tolist(),
            normalized_material_coordinates=init['material_coordinates_normalized'][j],FB_native_px=float(fb[i]),
            LK_vs_source_H_native_px=float(h_error[i]),original_sigma=deepcopy(original['sigma_px']))for j,i in enumerate(indices)]
        # The old source has higher native print scale; show both crops at
        # their native resolution, never warp target pixels to the reference.
        for start in range(0,len(records),12):
            selected=records[start:start+12];crops=[]
            for bgr,roi,key in((source,r0,'source_UV'),(target,r1,'target_UV')):
                x0,y0,x1,y1=roi;crop=bgr[y0:y1,x0:x1].copy()
                for j,r in enumerate(selected,start):
                    xy=np.rint(np.asarray(r[key])-[x0,y0]).astype(int)
                    cv2.circle(crop,tuple(xy),3,(0,255,255),1)
                    cv2.putText(crop,str(j),tuple(xy+[4,-4]),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,0),1,cv2.LINE_AA)
                crops.append(crop)
            page=np.full((max(c.shape[0]for c in crops)+35,sum(c.shape[1]for c in crops)+20,3),245,np.uint8)
            x=0
            for crop in crops:page[35:35+crop.shape[0],x:x+crop.shape[1]]=crop;x+=crop.shape[1]+20
            cv2.putText(page,f'Ad{number} oldFIT {original["native_pts"]} -> 230036992 points{start}..{start+len(selected)-1}',(6,24),cv2.FONT_HERSHEY_SIMPLEX,.43,(20,20,20),1,cv2.LINE_AA)
            path=OUT/f'Ad{number}_230036992_same_actual_ink_SOURCE_review_{start:03d}.png'
            if path.exists():np.testing.assert_array_equal(cv2.imread(str(path)),page)
            else:assert cv2.imwrite(str(path),page)
            sheets.append(dict(advertisement_number=number,start_point=start,stop_point=start+len(selected),**ref(path)))
        reviews.append(dict(advertisement_number=number,source_frame=deepcopy(original['source_raw_frame']),target_frame=ref(TARGET),
            points=records,source_navigation_ROIs_native_xyxy=[r0,r1],source_only_rejected_point_count=int((~keep).sum()),
            source_only_rejected_indices=np.flatnonzero(~keep).tolist(),actual_both_endpoint_ink_review_pending=True))
    candidate=dict(schema='rinkspace.user-fullvideo-irregular-print-frozen-material-native-absolute-source-candidate.v1',
        source_video_sha256=SOURCE_SHA,time_base='1/15360',parent=ref(PARENT),new_source_anchor_rows=new,
        source_only_pair_reviews=reviews,source_marker_review_pages=sheets,existing_frozen_template_world_initializer_preserved=True,
        parent72_track_and12_anchor_UV_sigma_split_and_initialization_unchanged=True,
        source_camera_q_world_graph_or_solver_not_loaded=True,HELD_not_promoted_to_FIT=True,
        authority=args.authority,material_and_pixel_correspondence_actual_RGB_review_pending=True,accuracy_accepted=False)
    candidate_path=OUT/'Ad22_Ad23_bad_time_same_frozen_ink_actual_native_anchor_CANDIDATE_v1.json';write_new(candidate_path,candidate)
    packet=deepcopy(parent);packet['source_anchor_observations']+=deepcopy(new)
    packet.update(parent_source_packet=ref(PARENT),source_append_user_bad_Ad22_23=dict(candidate=ref(candidate_path),
        old_track_rows_and_anchor_rows_exact_prefix_preserved=True,newFITpoints=sum(len(r['uv'])for r in new),
        newHELDpoints=0,new_geometry_parameters=0,new_camera_parameters=0,actual_material_review_pending=True),
        solver_target_updated_here=False,source_review_pending_before_production_admission=True)
    assert packet['rows']==parent['rows'] and packet['frozen_source_print_templates']==parent['frozen_source_print_templates']
    assert packet['source_anchor_observations'][:len(parent['source_anchor_observations'])]==parent['source_anchor_observations']
    assert PARENT.read_bytes()==old_bytes
    output=OUT/'native_six_parent_plus_Ad22_Ad23_bad_time_actual_frozen_ink_SOURCE_REVIEW_PENDING_v1.json';write_new(output,packet)
    print(json.dumps(dict(candidate=ref(candidate_path),packet=ref(output),newFITpoints=sum(len(r['uv'])for r in new),
        groups=[dict(ad=r['advertisement_number'],points=len(r['uv']),mutualMatches=r['source_source_tracking']['mutual_unique_descriptor_matches'],inliers=r['source_source_tracking']['source_only_projective_inliers'])for r in new],sheets=sheets)),flush=True)


if __name__=='__main__':main()
