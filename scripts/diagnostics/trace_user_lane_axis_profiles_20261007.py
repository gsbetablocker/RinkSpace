"""Source-only normal-profile tracking of user-confirmed physical Lane sides."""
from pathlib import Path
import argparse
import hashlib
import json
import av
import cv2
import numpy as np
from trace_user_lane_source_interval_20261007 import DATA, SOURCE, PACKET, BEGIN, END, hom, profile, fb_tracks


def norm(signal):
    signal = np.asarray(signal,float)
    signal=cv2.GaussianBlur(signal.reshape(-1,1),(1,5),1.1).reshape(-1)
    x=np.linspace(-1.,1.,len(signal))
    signal -= np.polyval(np.polyfit(x,signal,1),x)
    strength=float(np.percentile(signal,95)-np.percentile(signal,5))
    return signal/max(1e-20,np.linalg.norm(signal)), strength


def match(template, gray, guess, tangent, radius=24, span=28):
    ref,ref_strength=norm(template)
    unit_t=np.asarray(tangent,float);unit_t/=np.linalg.norm(unit_t)
    n=np.array([-unit_t[1],unit_t[0]])
    deltas=np.arange(-radius,radius+.01,.5)
    signals=[];scores=[];strengths=[]
    for delta in deltas:
        xy=guess+delta*n
        signal,_=profile(gray,xy,unit_t,span)
        value,strength=norm(signal)
        scores.append(float(ref@value));signals.append(signal);strengths.append(strength)
    j=int(np.argmax(scores));best=deltas[j]
    if 0<j<len(deltas)-1:
        a,b,c=scores[j-1:j+2]
        if a-2*b+c<0:
            best+=float(np.clip(.25*(a-c)/(a-2*b+c),-.5,.5))
    xy=guess+best*n
    selected,normal=profile(gray,xy,unit_t,span)
    support=deltas[np.asarray(scores)>=scores[j]-.01]
    full_patch=(6<xy[0]<1913 and span+2<xy[1]<1077-span)
    ui_overlap=(xy[0]+abs(span*n[0])+6>1690 and xy[0]-abs(span*n[0])-6<1865
        and xy[1]+abs(span*n[1])+6>944 and xy[1]-abs(span*n[1])-6<1000)
    return xy,selected,dict(ncc=float(scores[j]),normal_delta_px=float(best),
        score_within_001_support_px=[float(support.min()-best),float(support.max()-best)],
        reference_profile_strength=ref_strength,current_profile_strength=float(strengths[j]),
        source_normal=normal.tolist(),normal_search_offsets_px=deltas.tolist(),normal_search_NCC=scores,
        full_source_profile_visible=bool(full_patch),source_profile_overlaps_fixed_UI=bool(ui_overlap),
        hit_search_boundary=bool(j in (0,len(deltas)-1)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",required=True);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    packet=json.loads(PACKET.read_text(encoding="utf-8-sig"))
    user=[r for r in packet["rows"] if r.get("user_confirmed_source_axis")]
    frames={};receipts=[]
    with av.open(str(SOURCE)) as container:
        stream=container.streams.video[0];stream.thread_count=1
        container.seek(BEGIN-15360,stream=stream,backward=True,any_frame=False)
        for f in container.decode(stream):
            if f.pts>END:break
            if f.pts<BEGIN:continue
            bgr=f.to_ndarray(format="bgr24");frames[int(f.pts)]=bgr
            receipts.append(dict(pts=int(f.pts),time_base=str(f.time_base),native_BGR_sha256=hashlib.sha256(bgr.tobytes()).hexdigest()))
    stamps=sorted(frames)
    if stamps!=list(range(BEGIN,END+1,256)):raise ValueError("Incomplete native source interval")
    if not np.array_equal(cv2.imread(user[0]["source_raw_frame"]),frames[BEGIN]):raise ValueError("User source changed")
    gray={p:cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) for p,f in frames.items()}
    sample_gray={p:g.astype(np.float32) for p,g in gray.items()}
    board_mask=np.zeros((1080,1920),np.uint8);board_mask[305:570,20:575]=255;board_mask[305:570,1150:1895]=255
    initial=np.concatenate([np.array(r["uv"],float) for r in user]);points=initial.copy()
    tangent=np.concatenate([np.tile((np.array(r["uv"])[-1]-np.array(r["uv"])[0]),(len(r["uv"]),1)) for r in user])
    tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    templates=[profile(sample_gray[BEGIN],xy,t,span=28)[0] for xy,t in zip(points,tangent)]
    first_templates=[s.copy() for s in templates];first_tangent=tangent.copy()
    alive=np.ones(len(points),bool);history=[];by_pts={BEGIN:points.copy()};cumulative=np.eye(3)
    for pa,pb in zip(stamps[:-1],stamps[1:]):
        board_points=cv2.goodFeaturesToTrack(gray[pa],maxCorners=900,qualityLevel=.012,minDistance=9,mask=board_mask,blockSize=7).reshape(-1,2)
        bq,valid,fb=fb_tracks(gray[pa],gray[pb],board_points)
        bi=np.rint(bq).astype(int).clip([0,0],[1919,1079]);valid &= board_mask[bi[:,1],bi[:,0]]!=0
        bp,bq=board_points[valid],bq[valid]
        affine,ins=cv2.estimateAffinePartial2D(bp,bq,method=cv2.RANSAC,ransacReprojThreshold=1.25,maxIters=5000,confidence=.999,refineIters=15)
        if affine is None:raise RuntimeError(f"No actual board motion at {pb}")
        H=np.eye(3);H[:2]=affine;cumulative=H@cumulative
        guesses=hom(H,points);next_tangent=tangent@affine[:2,:2].T;next_tangent/=np.linalg.norm(next_tangent,axis=1)[:,None]
        items=[];next_templates=[];next_points=[]
        for k,(p,q,t,nt,template) in enumerate(zip(points,guesses,tangent,next_tangent,templates)):
            target,signal,quality=match(template,sample_gray[pb],q,nt)
            back_guess=hom(np.linalg.inv(H),[target])[0]
            back,_,reverse=match(signal,sample_gray[pa],back_guess,t)
            pn=np.array([-t[1],t[0]])
            normal_fb=float((back-p)@pn)
            independent,_,endpoint=match(first_templates[k],sample_gray[pb],target,nt)
            endpoint_normal=float((independent-target)@np.array([-nt[1],nt[0]]))
            valid=quality["ncc"]>.72 and quality["current_profile_strength"]>3.5 and quality["full_source_profile_visible"]
            valid &= not quality["source_profile_overlaps_fixed_UI"]
            valid &= not quality["hit_search_boundary"] and abs(normal_fb)<1.5
            valid &= endpoint["ncc"]>.65 and abs(endpoint_normal)<3.
            alive[k]&=valid
            items.append(dict(original_user_index=k,source_axis_uv=target.tolist(),source_previous_uv=p.tolist(),
                source_board_guess_uv=q.tolist(),source_tangent=nt.tolist(),profile_match=quality,
                reverse_profile_match=reverse,step_forward_backward_normal_px=normal_fb,
                independent_original_template_match=endpoint,independent_original_template_normal_delta_px=endpoint_normal,
                chain_alive=bool(alive[k]),current_step_valid=bool(valid),raw_gray_normal_profile_offset_px=list(range(-28,29)),
                raw_gray_normal_profile=signal.tolist()))
            next_templates.append(signal);next_points.append(target)
        points=np.array(next_points);templates=next_templates;tangent=next_tangent
        by_pts[pb]=points.copy()
        inmask=ins.ravel()!=0
        history.append(dict(from_pts=pa,to_pts=pb,source_board_initialization_only=dict(source_tracks=len(bp),inliers=int(ins.sum()),
            source_similarity_affine=affine.tolist(),source_inlier_median_residual_px=float(np.median(np.linalg.norm(hom(H,bp[inmask])-bq[inmask],axis=1)))),
            source_axis_normal_tracking=items))
    # Native source crops show only actual annotation/correspondence markers.
    panels=[]
    for pts in [BEGIN,230503168,230505216,230506752,END]:
        shown=frames[pts].copy();xy=by_pts[pts]
        alivepts=np.ones(len(xy),bool) if pts==BEGIN else np.array([r["chain_alive"] for r in next(h for h in history if h["to_pts"]==pts)["source_axis_normal_tracking"]])
        for k,p in enumerate(xy):
            if not (0<p[0]<1920 and 0<p[1]<1080):continue
            color=(40,125,240) if k<6 else (210,120,25)
            cv2.circle(shown,tuple(np.rint(p).astype(int)),4,color,1 if alivepts[k] else 2,cv2.LINE_AA)
            cv2.putText(shown,f"{k}:{'OK' if alivepts[k] else 'LOST'}",tuple(np.rint(p+[6,-6]).astype(int)),0,.4,color,1,cv2.LINE_AA)
        for source in packet["rows"]:
            if source["pts"]==pts and source["id"].startswith("cut6_source_"):
                for p in source["uv"]:cv2.drawMarker(shown,tuple(np.rint(p).astype(int)),(10,10,250),cv2.MARKER_TILTED_CROSS,13,2,cv2.LINE_AA)
        head=np.full((35,1920,3),245,np.uint8)
        cv2.putText(head,f"PTS {pts} | source RGB normal-profile correspondence; coloured points labelled OK/LOST; red old UV; no projected model",(10,25),0,.52,(20,20,20),1)
        panels.append(np.vstack((head,shown[790:1080])))
    image=out/"native_source_profile_correspondence.png";cv2.imwrite(str(image),np.vstack(panels))
    report=dict(schema="rinkspace.user-lane-source-only-normal-profile-trace.v1",source_video=str(SOURCE),
        original_user_source_packet=str(PACKET),original_user_source_packet_sha256=hashlib.sha256(PACKET.read_bytes()).hexdigest(),
        decoded_native_receipts=receipts,complete_display_interval_pts=stamps,
        original_registered_user_rows=[dict(id=r["id"],curve=r["curve"],uv=r["uv"]) for r in user],
        source_board_ROI_xyxy=[[20,305,575,570],[1150,305,1895,570]],per_pair_source_profile_trace=history,
        final_alive_indices=np.flatnonzero(alive).tolist(),native_diagnostic=str(image),
        normal_profile_axis_keeps_original_user_axis_not_dark_ridge_minimum=True,
        normal_profile_span_px=28,
        board_source_similarity_used_for_broad_search_initialization_only=True,
        camera_world_model_or_network_not_imported=True,input_packets_states_and_solver_unchanged=True,
        tracked_axes_not_yet_admitted_as_fit_or_holdout=True,
        limitations=["Aperture prevents tangent position accuracy; this trace measures source normal correspondence.",
            "Forward-backward and original-template normal agreement are necessary but not sufficient for physical accuracy.",
            "Blur and scratch/reflection variation remain in raw normal profiles; no unobserved end points or pixels filled."])
    path=out/"user_lane_source_normal_profile_trace.json";path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(report=str(path),image=str(image),final_alive_indices=report["final_alive_indices"],
        final=[dict(index=r["original_user_index"],uv=r["source_axis_uv"],alive=r["chain_alive"],ncc=r["profile_match"]["ncc"],
            fb=r["step_forward_backward_normal_px"],original_template_delta=r["independent_original_template_normal_delta_px"]) for r in history[-1]["source_axis_normal_tracking"]]),ensure_ascii=False))


if __name__=="__main__":main()
