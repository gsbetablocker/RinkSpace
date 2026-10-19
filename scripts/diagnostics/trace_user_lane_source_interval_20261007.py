"""Trace direct user axes using native source texture only; no model imports."""
from pathlib import Path
import argparse
import hashlib
import json
import av
import cv2
import numpy as np

cv2.setNumThreads(1)
cv2.setRNGSeed(107)
DATA = Path(r"D:/CodexData/RinkSpace_Data/events/ankara_2026")
SOURCE = DATA / "media/originals/oxXu1avhrvs.mp4"
PACKET = DATA / "observations/vision_training/continuous_review/cut6_clear_source_constraints/complete_source_clear_anchor_append_20261007/ice_clear_anchors_user_lane2_lane3_20261007.json"
BEGIN, END = 230501120, 230507776


def hom(H, points):
    return cv2.perspectiveTransform(np.asarray(points, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)


def stats(a):
    a = np.asarray(a, float)
    return dict(count=len(a), median=float(np.median(a)), p90=float(np.percentile(a, 90)), maximum=float(np.max(a))) if len(a) else dict(count=0)


def fb_tracks(a, b, p, seed=None):
    p = np.asarray(p, np.float32).reshape(-1, 1, 2)
    settings = dict(winSize=(41, 41), maxLevel=5,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, .003))
    initial = None if seed is None else hom(seed,p).astype(np.float32).reshape(-1,1,2)
    if seed is not None: settings["flags"] = cv2.OPTFLOW_USE_INITIAL_FLOW
    q, s, _ = cv2.calcOpticalFlowPyrLK(a, b, p, initial, **settings)
    reverse_initial = None if seed is None else hom(np.linalg.inv(seed),q).astype(np.float32).reshape(-1,1,2)
    back, t, _ = cv2.calcOpticalFlowPyrLK(b, a, q, reverse_initial, **settings)
    error = np.linalg.norm(p.reshape(-1, 2)-back.reshape(-1, 2), axis=1)
    q = q.reshape(-1, 2)
    valid = (s.ravel()!=0) & (t.ravel()!=0) & (error<1.0)
    valid &= (q[:, 0]>2) & (q[:, 0]<1917) & (q[:, 1]>2) & (q[:, 1]<1077)
    return q, valid, error


def profile(gray, uv, tangent, span=40):
    tangent = np.asarray(tangent, float); tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    d = np.arange(-span, span+1, dtype=np.float32)
    t = np.arange(-6, 7, dtype=np.float32)
    xy = np.asarray(uv)[None, None, :] + d[:, None, None]*normal + t[None, :, None]*tangent
    signal = cv2.remap(gray.astype(np.float32, copy=False), xy[..., 0].astype(np.float32),
        xy[..., 1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.median(signal, axis=1), normal


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", required=True); args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    packet = json.loads(PACKET.read_text(encoding="utf-8-sig"))
    user = [r for r in packet["rows"] if r.get("user_confirmed_source_axis")]
    frames, receipts = {}, []
    with av.open(str(SOURCE)) as container:
        stream = container.streams.video[0]; stream.thread_count=1
        container.seek(BEGIN-15360, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts>END: break
            if frame.pts<BEGIN: continue
            bgr = frame.to_ndarray(format="bgr24")
            frames[int(frame.pts)] = bgr
            receipts.append(dict(pts=int(frame.pts), time_base=str(frame.time_base),
                decoded_BGR_sha256=hashlib.sha256(bgr.tobytes()).hexdigest(), native_hw=list(bgr.shape[:2])))
    stamps = sorted(frames)
    if stamps != list(range(BEGIN, END+1, 256)):
        raise ValueError("Missing native display PTS in complete interval")
    first = cv2.imread(user[0]["source_raw_frame"])
    if not np.array_equal(first, frames[BEGIN]):
        raise ValueError("Decoded first RGB differs from authenticated user source")
    gray = {p: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for p, f in frames.items()}
    # Explicit, source-reviewed ice-only mask avoids athlete, board and fixed UI.
    mask = np.zeros((1080,1920), np.uint8)
    mask[615:1077, 20:535] = 255
    mask[615:935, 1090:1895] = 255
    mask[1015:1077, 1090:1895] = 255
    mask[935:1015, 1090:1640] = 255
    board_mask=np.zeros_like(mask)
    board_mask[305:570,20:575]=255
    board_mask[305:570,1150:1895]=255
    clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(8,8))
    motion_gray = {p: clahe.apply(g) for p, g in gray.items()}
    cumulative = np.eye(3)
    originals = np.concatenate([np.asarray(r["uv"],float) for r in user])
    direct = originals.copy()
    alive = np.ones(len(originals), bool)
    direct_history = {BEGIN: direct.tolist()}
    rows, H_by_pts, alive_by_pts = [], {BEGIN: cumulative.copy()}, {BEGIN: alive.tolist()}
    for pa, pb in zip(stamps[:-1], stamps[1:]):
        board_points=cv2.goodFeaturesToTrack(gray[pa],maxCorners=900,qualityLevel=.012,
            minDistance=9,mask=board_mask,blockSize=7).reshape(-1,2)
        board_q,board_valid,board_fb=fb_tracks(gray[pa],gray[pb],board_points)
        bi=np.rint(board_q).astype(int).clip([0,0],[1919,1079]);board_valid &= board_mask[bi[:,1],bi[:,0]]!=0
        bp,bq=board_points[board_valid],board_q[board_valid]
        seed,board_inliers=cv2.findHomography(bp,bq,cv2.RANSAC,1.25,maxIters=7000,confidence=.999)
        if seed is None: raise RuntimeError(f"No measured source board seed at {pb}")
        points = cv2.goodFeaturesToTrack(motion_gray[pa], maxCorners=1800, qualityLevel=.008,
            minDistance=9, mask=mask, blockSize=7).reshape(-1,2)
        tracked, valid, fb = fb_tracks(motion_gray[pa], motion_gray[pb], points,seed)
        qi = np.rint(tracked).astype(int).clip([0,0],[1919,1079])
        valid &= mask[qi[:,1],qi[:,0]]!=0
        p, q, fb = points[valid], tracked[valid], fb[valid]
        held = np.arange(len(p))%3==0
        H, inliers = cv2.findHomography(p[~held], q[~held], cv2.RANSAC, 1.25,
            maxIters=7000, confidence=.999)
        if H is None: raise RuntimeError(f"No source ice homography at {pb}")
        H /= H[2,2]
        # Reverse source estimate is independent, not inverse(H).
        reverse, _ = cv2.findHomography(q[~held], p[~held], cv2.RANSAC, 1.25,
            maxIters=7000, confidence=.999)
        res = np.linalg.norm(hom(H,p)-q,axis=1)
        reverse_res = np.linalg.norm(hom(reverse,hom(H,p))-p,axis=1)
        cumulative = H @ cumulative; cumulative /= cumulative[2,2]
        H_by_pts[pb] = cumulative.copy()
        qdirect, direct_valid, direct_fb = fb_tracks(motion_gray[pa], motion_gray[pb], direct,seed)
        direct = qdirect
        alive &= direct_valid
        direct_history[pb] = direct.tolist(); alive_by_pts[pb] = alive.tolist()
        predicted = hom(cumulative, originals)
        rows.append(dict(from_pts=pa,to_pts=pb,source_tracks=len(p),fit_inliers=int(inliers.sum()),
            board_seed_only_not_used_as_ice_geometry=dict(source_tracks=len(bp),inliers=int(board_inliers.sum()),
                inlier_residual_px=stats(np.linalg.norm(hom(seed,bp[board_inliers.ravel()!=0])-bq[board_inliers.ravel()!=0],axis=1)),
                measured_source_board_H=seed.tolist()),
            withheld_point_residual_px=stats(res[held]),fit_point_residual_px=stats(res[~held]),
            source_track_forward_backward_px=stats(fb),
            source_H_independent_reverse_roundtrip_px=stats(reverse_res),
            source_H=H.tolist(),source_cumulative_H=cumulative.tolist(),
            direct_user_LK_uv=direct.tolist(),direct_user_LK_alive=alive.tolist(),
            direct_user_LK_step_fb_px=direct_fb.tolist(),
            source_H_user_uv=predicted.tolist(),
            direct_LK_minus_ice_H_native_px=(direct-predicted).tolist()))
    # Direct first->last check uses source ice tracks, never any camera/world.
    corners = cv2.goodFeaturesToTrack(motion_gray[BEGIN], maxCorners=2200,qualityLevel=.005,
        minDistance=8,mask=mask,blockSize=7).reshape(-1,2)
    tracked, valid, fb = fb_tracks(motion_gray[BEGIN],motion_gray[END],corners,cumulative)
    qi=np.rint(tracked).astype(int).clip([0,0],[1919,1079]);valid &= mask[qi[:,1],qi[:,0]]!=0
    p,q=corners[valid],tracked[valid]
    direct_check = dict(source_tracks=len(p), failure="Insufficient source LK endpoint correspondences")
    if len(p)>=4:
        HD, direct_ins = cv2.findHomography(p,q,cv2.RANSAC,1.5,maxIters=7000,confidence=.999)
        if HD is not None:
            direct_check = dict(source_tracks=len(p),inliers=int(direct_ins.sum()),H=HD.tolist(),
                residual_px=stats(np.linalg.norm(hom(HD,p)-q,axis=1)),
                cumulative_vs_direct_user_px=(hom(cumulative,originals)-hom(HD,originals)).tolist())
    sift=cv2.SIFT_create(nfeatures=4500,contrastThreshold=.007,edgeThreshold=15)
    ka,da=sift.detectAndCompute(motion_gray[BEGIN],mask)
    kb,db=sift.detectAndCompute(motion_gray[END],mask)
    sift_check=dict(failure="No source endpoint SIFT descriptors")
    if da is not None and db is not None:
        matcher=cv2.BFMatcher()
        forward=matcher.knnMatch(da,db,k=2);reverse=matcher.knnMatch(db,da,k=2)
        back={m.queryIdx:m.trainIdx for m,n in reverse if m.distance<.72*n.distance}
        mutual=[m for m,n in forward if m.distance<.72*n.distance and back.get(m.trainIdx)==m.queryIdx]
        sift_check=dict(mutual_source_matches=len(mutual),failure="Insufficient source endpoint SIFT matches")
        if len(mutual)>=8:
            p=np.array([ka[m.queryIdx].pt for m in mutual]);q=np.array([kb[m.trainIdx].pt for m in mutual])
            HS,ins=cv2.findHomography(p,q,cv2.RANSAC,2.,maxIters=7000,confidence=.999)
            if HS is not None:
                sift_check=dict(mutual_source_matches=len(mutual),inliers=int(ins.sum()),H=HS.tolist(),
                    residual_px=stats(np.linalg.norm(hom(HS,p[ins.ravel()!=0])-q[ins.ravel()!=0],axis=1)),
                    cumulative_vs_direct_user_px=(hom(cumulative,originals)-hom(HS,originals)).tolist())
    # Preserve raw normal profiles around transported annotations. Automatic
    # dark-ridge minima are not silently promoted to physical upper/lower edges.
    sampled = []
    chosen = [BEGIN, 230503168, 230505216, 230506752, END]
    panels = []
    for pts in chosen:
        shown = frames[pts].copy()
        H=H_by_pts[pts]
        start = 0
        for source, color in zip(user, ((40,125,240),(210,120,25))):
            uv=np.asarray(source["uv"],float);transferred=hom(H,uv)
            line_extended=np.array([[0., np.polyval(np.polyfit(uv[:,0],uv[:,1],1),0.)],
                [1919.,np.polyval(np.polyfit(uv[:,0],uv[:,1],1),1919.)]])
            line=hom(H,line_extended)
            cv2.line(shown,tuple(np.rint(line[0]).astype(int)),tuple(np.rint(line[1]).astype(int)),color,1,cv2.LINE_AA)
            profiles=[]
            tangent=hom(H,np.array([uv[0],uv[0]+[10.,float(10*(uv[-1,1]-uv[0,1])/(uv[-1,0]-uv[0,0]))]]))
            tangent=tangent[1]-tangent[0]
            for k, xy in enumerate(transferred):
                inside=30<xy[0]<1890 and 45<xy[1]<1035
                if not inside: continue
                signal, normal=profile(gray[pts],xy,tangent)
                lk=np.array(direct_history[pts])[start+k]
                profiles.append(dict(original_user_index=k,source_H_uv=xy.tolist(),
                    direct_LK_uv=lk.tolist(),direct_LK_alive=alive_by_pts[pts][start+k],
                    unit_source_normal=normal.tolist(),normal_profile_offset_px=list(range(-40,41)),
                    median_13_tangent_pixel_gray=signal.tolist()))
                cv2.circle(shown,tuple(np.rint(xy).astype(int)),4,color,1,cv2.LINE_AA)
                if alive_by_pts[pts][start+k]:
                    cv2.drawMarker(shown,tuple(np.rint(lk).astype(int)),(200,30,210),cv2.MARKER_CROSS,11,1,cv2.LINE_AA)
            sampled.append(dict(pts=pts,source_user_id=source["id"],source_profile_probes=profiles,
                extended_image_line_for_navigation_only_not_admitted_source_pixels=line.tolist()))
            start += len(uv)
        for r in packet["rows"]:
            if r["pts"]==pts and r["id"].startswith("cut6_source_"):
                for xy in r["uv"]:
                    cv2.drawMarker(shown,tuple(np.rint(xy).astype(int)),(10,10,250),cv2.MARKER_TILTED_CROSS,13,2,cv2.LINE_AA)
        crop=shown[790:1080].copy();head=np.full((35,1920,3),245,np.uint8)
        cv2.putText(head,f"PTS {pts} | source-only ice-H traced user lines; magenta direct LK; RED old later UV | extended line = navigation, NOT source measurement",(10,25),0,.5,(20,20,20),1)
        panels.append(np.vstack((head,crop)))
    image=out/"native_source_lane_interval_navigation.png"
    cv2.imwrite(str(image),np.vstack(panels))
    report=dict(schema="rinkspace.user-lane-source-only-native-interval-trace.v1",source_video=str(SOURCE),
        source_video_sha256_from_existing_ledger="f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891",
        first_decoded_BGR_equals_existing_user_RGB_bitexact=True,decoded_native_receipts=receipts,
        complete_display_interval_pts=stamps,source_ice_ROIs_xyxy=[[20,615,535,1077],[1090,615,1895,935],[1090,1015,1895,1077],[1090,935,1640,1015]],
        original_user_rows=[dict(id=r["id"],curve=r["curve"],uv=r["uv"]) for r in user],
        per_pair_source_tracking=rows,direct_source_endpoint_tracking_check=direct_check,
        independent_source_SIFT_endpoint_check=sift_check,
        source_normal_profiles=sampled,native_interval_navigation_image=str(image),
        source_axis_packet_not_admitted_or_modified=True,camera_world_network_and_solver_not_imported=True,
        limitations=["Ice homography is a source-image motion hypothesis; not a calibrated world model.",
            "Scratch, reflection and blur changes can spoil source correspondence; FB and held residuals must be reviewed.",
            "Whole line extrapolation is for source navigation only; no unseen point is accepted as measurement.",
            "User physical side identity is inherited only after reliable source texture correspondence; gray-ridge centre is not automatically upper/lower edge."])
    path=out/"user_lane_source_interval_trace.json"
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(report=str(path),navigation=str(image),
        worst_pair_held_p90=max(r["withheld_point_residual_px"]["p90"] for r in rows),
        final_surviving_direct_LK=int(alive.sum()),direct_check=direct_check),ensure_ascii=False))


if __name__=="__main__": main()
