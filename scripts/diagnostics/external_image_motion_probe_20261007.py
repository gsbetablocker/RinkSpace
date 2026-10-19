"""Independent original-RGB static-scene motion probe; no old UV or fitting.

The homographies are image diagnostics, not calibrated rink/camera models.
All masks are declared pixel ROIs; movement and unknown depth remain limitations.
"""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse, hashlib, json, time
from pathlib import Path
import av
import cv2
import numpy as np

cv2.setNumThreads(1)
cv2.setRNGSeed(917)

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def native_frames(path, stamps):
    wanted=set(stamps); result={}; receipts=[]
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        stream.thread_count=1
        container.seek(min(stamps)-15360,stream=stream,backward=True,any_frame=False)
        for f in container.decode(stream):
            if f.pts in wanted:
                bgr=f.to_ndarray(format="bgr24")
                result[f.pts]=bgr
                receipts.append(dict(pts=f.pts,time_base=str(f.time_base),native_hw=list(bgr.shape[:2]),native_bgr_sha256=hashlib.sha256(bgr.tobytes()).hexdigest()))
            if f.pts is not None and f.pts>max(stamps): break
    if set(result)!=wanted: raise RuntimeError(f"Missing native PTS {sorted(wanted-set(result))}")
    return result,receipts

def stats(v):
    a=np.asarray(v,float)
    return dict(count=len(a),median=float(np.median(a)),p95=float(np.percentile(a,95)),maximum=float(np.max(a))) if len(a) else dict(count=0)

def hom_apply(h,p):
    return cv2.perspectiveTransform(np.asarray(p,np.float64).reshape(-1,1,2),h).reshape(-1,2)

def track_pair(a,b,mask):
    aa=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY);bb=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
    p=cv2.goodFeaturesToTrack(aa,maxCorners=1400,qualityLevel=.012,minDistance=9,mask=mask,blockSize=7)
    if p is None: raise RuntimeError("No source features")
    q,s,e=cv2.calcOpticalFlowPyrLK(aa,bb,p,None,winSize=(31,31),maxLevel=5,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.005))
    z,t,_=cv2.calcOpticalFlowPyrLK(bb,aa,q,None,winSize=(31,31),maxLevel=5,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.005))
    p=p.reshape(-1,2);q=q.reshape(-1,2);fb=np.linalg.norm(p-z.reshape(-1,2),axis=1)
    h,w=mask.shape
    valid=(s.ravel()!=0)&(t.ravel()!=0)&(fb<.8)&(q[:,0]>1)&(q[:,0]<w-2)&(q[:,1]>1)&(q[:,1]<h-2)
    qi=np.rint(q).astype(int); qi[:,0]=qi[:,0].clip(0,w-1);qi[:,1]=qi[:,1].clip(0,h-1)
    valid&=mask[qi[:,1],qi[:,0]]!=0
    p=p[valid];q=q[valid];fb=fb[valid]
    if len(p)<12: raise RuntimeError(f"Only {len(p)} valid tracks")
    # Deterministic 2/3 training; the held points never estimate the homography.
    test=(np.arange(len(p))%3)==0
    hm,ins=cv2.findHomography(p[~test],q[~test],cv2.RANSAC,1.5,maxIters=5000,confidence=.999)
    if hm is None: raise RuntimeError("No image homography")
    res=np.linalg.norm(hom_apply(hm,p)-q,axis=1)
    keep=res<2.0
    return p,q,fb,hm,test,keep,res

def unpack(c):
    r=cv2.Rodrigues(c[3:6])[0]
    k=np.array([[np.exp(c[6]),c[10],c[8]],[0,np.exp(c[7]),c[9]],[0,0,1.]])
    return r,k,np.asarray(c[11:25])

def model_transfer(ca,cb,p,plane_z=None):
    ra,ka,da=unpack(ca);rb,kb,db=unpack(cb)
    rays=cv2.undistortPointsIter(p.reshape(-1,1,2).astype(float),ka,da,None,None,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,80,1e-12)).reshape(-1,2)
    rays=np.c_[rays,np.ones(len(rays))]@ra
    if plane_z is None:
        xyzcam=rays@rb.T
    else:
        scale=(plane_z-ca[2])/rays[:,2]
        xyz=ca[:3]+scale[:,None]*rays
        xyzcam=(xyz-cb[:3])@rb.T
    q=cv2.projectPoints(xyzcam,np.zeros(3),np.zeros(3),kb,db)[0].reshape(-1,2)
    valid=np.isfinite(q).all(axis=1)&(xyzcam[:,2]>0)
    return q,valid

def inverse_roundtrip(c,p):
    r,k,d=unpack(c)
    u=cv2.undistortPointsIter(p.reshape(-1,1,2).astype(float),k,d,None,None,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,80,1e-12)).reshape(-1,2)
    back=cv2.projectPoints(np.c_[u,np.ones(len(u))],np.zeros(3),np.zeros(3),k,d)[0].reshape(-1,2)
    return float(np.max(np.linalg.norm(back-p,axis=1)))

def sift_pair(a,b,mask):
    sift=cv2.SIFT_create(nfeatures=2400,contrastThreshold=.025)
    ka,da=sift.detectAndCompute(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),mask)
    kb,db=sift.detectAndCompute(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),mask)
    if da is None or db is None:return dict(failure="missing descriptors")
    matcher=cv2.BFMatcher()
    f=matcher.knnMatch(da,db,k=2);r=matcher.knnMatch(db,da,k=2)
    rev={m.queryIdx:m.trainIdx for m,n in r if m.distance<.72*n.distance}
    matches=[m for m,n in f if m.distance<.72*n.distance and rev.get(m.trainIdx)==m.queryIdx]
    if len(matches)<10:return dict(mutual_ratio_matches=len(matches),failure="too few mutual matches")
    p=np.asarray([ka[m.queryIdx].pt for m in matches]);q=np.asarray([kb[m.trainIdx].pt for m in matches])
    hm,ins=cv2.findHomography(p,q,cv2.RANSAC,2,maxIters=4000,confidence=.999)
    return dict(mutual_ratio_matches=len(matches),ransac_inliers=int(ins.sum()),median_motion_px=float(np.median(np.linalg.norm(p[ins.ravel()!=0]-q[ins.ravel()!=0],axis=1))),H=hm.tolist())

def draw(a,b,p,q,hm,keep,pred,title):
    # Native-size drawing, then one 1600-pixel review canvas; arrows scaled x8.
    shown=b.copy();step=max(1,int(np.sum(keep)//70))
    idx=np.flatnonzero(keep)[::step]
    for j in idx:
        x=tuple(np.rint(q[j]).astype(int));cv2.circle(shown,x,4,(50,255,80),1)
        if pred is not None:
            end=tuple(np.rint(q[j]+8*(pred[j]-q[j])).astype(int))
            cv2.arrowedLine(shown,x,end,(20,20,255),2,tipLength=.2)
    cv2.putText(shown,title,(18,1030),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),3)
    cv2.putText(shown,title,(18,1030),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
    return cv2.resize(shown,(1600,900),interpolation=cv2.INTER_AREA)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--secondary-pair",action="store_true");ap.add_argument("--motion-pairs");args=ap.parse_args()
    if args.secondary_pair:
        secondary_pair_probe(args)
        return
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    source=Path(r"D:\CodexData\RinkSpace_Data\events\ankara_2026\media\originals\oxXu1avhrvs.mp4")
    stamps=[230112512,230144256,230231552]
    t0=time.perf_counter();frames,receipts=native_frames(source,stamps)
    for p in stamps: cv2.imwrite(str(out/f"native_{p}.jpg"),frames[p],[cv2.IMWRITE_JPEG_QUALITY,95])
    video=Path(r"D:\CodexData\RinkSpace_Data\events\ankara_2026\calibration\runs\solver_acceleration_20261003\okada_user_priorities_shared_actual_fit_fast_20261005\complete_physical_camera_model_UNACCEPTED_917c204dafde4027.npz")
    current=Path(r"D:\CodexData\RinkSpace_Data\events\ankara_2026\calibration\runs\solver_acceleration_20261003\source_corrected_RF_complete_joint_20261007\complete_physical_camera_model_UNACCEPTED_07f5492daacd7c0a.npz")
    models={}
    for name,path in [("video_4075",video),("current_a8d2",current)]:
        d=np.load(path,allow_pickle=False);models[name]=(d,{int(p):i for i,p in enumerate(d["pts"])})
    # Physical static material only: far print board and ice side regions.
    # No body segmentation, old observations, model predictions or marker IDs select points.
    roi={"board":[[30,305,630,420],[1120,285,1880,420]],"ice":[[30,450,630,920],[1120,450,1880,920]],"unmasked_with_UI":[[25,25,1880,1030]]}
    report=dict(schema="rinkspace.independent-original-rgb-motion-probe.v1",source_video=str(source),source_sha256_from_verified_source_ledger="f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891",decoded_native_receipts=receipts,old_source_UV_used=False,old_model_used_for_feature_selection=False,camera_models={name:dict(path=str(path),sha256=sha(path)) for name,path in [("video_4075",video),("current_a8d2",current)]},roi_rectangles=roi,image_baseline_fit_test_split="index mod3: 2/3 estimate and 1/3 withheld",diagnostic_not_final_model=True,experiments=[],limitations=["Static ROI assumption visually reviewed; not semantic segmentation.","Ice features include actual paint and scratch texture; reflections can be nonrigid.","Board rotation-only transfer ignores lever-arm parallax; ice transfer uses model Z=0 plane.","Single shot and three source frames cannot validate all nine shots or absolute geometry.","Clock shift scan only tests relative-motion consistency locally; not source PTS/exposure provenance."])
    for pa,pb in [(stamps[0],stamps[1]),(stamps[1],stamps[2]),(stamps[0],stamps[2])]:
        for material,boxes in roi.items():
            mask=np.zeros((1080,1920),np.uint8)
            for x0,y0,x1,y1 in boxes: mask[y0:y1,x0:x1]=255
            if material=="unmasked_with_UI": mask[180:980,630:1120]=0
            entry=dict(pts=[pa,pb],seconds=[pa/15360,pb/15360],material=material)
            try:
                p,q,fb,hm,test,keep,res=track_pair(frames[pa],frames[pb],mask)
            except RuntimeError as e:
                entry["failure"]=str(e);report["experiments"].append(entry);continue
            clean=test&keep
            entry.update(total_tracks=len(p),baseline_H=hm.tolist(),baseline_all_held_test_error_px=stats(res[test]),baseline_good_held_test_error_px=stats(res[clean]),raw_observed_displacement_px=stats(np.linalg.norm(q[clean]-p[clean],axis=1)),tracked_fb_error_px=stats(fb[clean]),baseline_good_test_count=int(np.sum(clean)),baseline_good_all_count=int(np.sum(keep)),models={})
            np.savez_compressed(out/f"tracks_{pa}_{pb}_{material}.npz",p=p,q=q,fb=fb,H=hm,test=test,image_baseline_keep=keep,image_baseline_residual=res)
            for name,(d,indices) in models.items():
                ca=d["camera25"][indices[pa]];cb=d["camera25"][indices[pb]]
                pred,valid=model_transfer(ca,cb,p,0. if material=="ice" else None)
                selected=clean&valid
                all_selected=test&valid
                err=pred-q
                mp=dict(transfer_kind="ice_Z0_full_model" if material=="ice" else "camera_rotation_only_no_lever_parallax",inverse_Brown_roundtrip_max_error_px=inverse_roundtrip(ca,p[selected]),raw_predicted_displacement_px=stats(np.linalg.norm(pred[selected]-p[selected],axis=1)),prediction_vs_original_RGB_track_error_px=stats(np.linalg.norm(err[selected],axis=1)),mean_signed_error_xy_px=np.mean(err[selected],axis=0).tolist(),median_signed_error_xy_px=np.median(err[selected],axis=0).tolist(),source_actual_alias_pts=[int(d["exposure_representative_pts"][indices[pa]]),int(d["exposure_representative_pts"][indices[pb]])],clock_global_shift_scan=[])
                mp["all_held_test_valid_model_error_px"]=stats(np.linalg.norm(err[all_selected],axis=1))
                mp["all_held_test_valid_same_points_baseline_error_px"]=stats(res[all_selected])
                mp["all_held_test_valid_same_points_RGB_motion_px"]=stats(np.linalg.norm(q[all_selected]-p[all_selected],axis=1))
                mp["held_test_invalid_model_count"]=int(np.sum(test&~valid))
                for offset in [-15360,-7680,-3840,-1536,-768,-256,0,256,768,1536,3840,7680,15360]:
                    if pa+offset not in indices or pb+offset not in indices:continue
                    pp,v=model_transfer(d["camera25"][indices[pa+offset]],d["camera25"][indices[pb+offset]],p,0. if material=="ice" else None)
                    ss=clean&v;mp["clock_global_shift_scan"].append(dict(offset_native_pts=offset,offset_seconds=offset/15360,median_error_px=float(np.median(np.linalg.norm(pp[ss]-q[ss],axis=1)))))
                entry["models"][name]=mp
                if material!="unmasked_with_UI":
                    image=draw(frames[pa],frames[pb],p,q,hm,keep,pred,f"{material} {pa}->{pb} {name}: green RGB track; red residual x8")
                    cv2.imwrite(str(out/f"probe_{pa}_{pb}_{material}_{name}.jpg"),image,[cv2.IMWRITE_JPEG_QUALITY,92])
            if pa==stamps[0] and pb==stamps[2] and material in ("board","ice"):
                entry["independent_SIFT_descriptor_check"]=sift_pair(frames[pa],frames[pb],mask)
                left=frames[pa].copy();right=frames[pb].copy()
                good=np.flatnonzero(clean)[::max(1,np.sum(clean)//16)]
                for number,j in enumerate(good):
                    for im,point in [(left,p[j]),(right,q[j])]:
                        xy=tuple(np.rint(point).astype(int));cv2.circle(im,xy,6,(30,255,80),2);cv2.putText(im,str(number),xy,cv2.FONT_HERSHEY_SIMPLEX,.65,(10,30,255),2)
                crop=(245,440) if material=="board" else (430,940)
                panel=np.hstack([left[crop[0]:crop[1]],right[crop[0]:crop[1]]])
                cv2.imwrite(str(out/f"correspondence_{pa}_{pb}_{material}.jpg"),cv2.resize(panel,(1920,panel.shape[0]//2)),[cv2.IMWRITE_JPEG_QUALITY,95])
            report["experiments"].append(entry)
    report["wall_seconds"]=time.perf_counter()-t0
    (out/"independent_motion_probe.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    for e in report["experiments"]:
        print(e["pts"],e["material"],e.get("baseline_all_held_test_error_px"),{k:v["prediction_vs_original_RGB_track_error_px"] for k,v in e.get("models",{}).items()},e.get("failure"))

def secondary_pair_probe(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    source=Path(r"D:\CodexData\RinkSpace_Data\events\ankara_2026\media\originals\oxXu1avhrvs.mp4")
    stamps=[230507264,230507776]
    frames,receipts=native_frames(source,stamps);a,b=[frames[p] for p in stamps]
    for p in stamps:cv2.imwrite(str(out/f"native_{p}.png"),frames[p])
    report=dict(schema="rinkspace.secondary-largest-motion-conflict-native-RGB-probe.v1",source_video=str(source),source_sha256_from_existing_verified_ledger="f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891",native_pts=stamps,dt_seconds=(stamps[1]-stamps[0])/15360,receipts=receipts,independent_baselines=[],actual_old_motion_pairs=[],limitations=["No training or physical camera fit is performed.","Image homography cannot establish absolute world geometry.","Actual old motion pairs are diagnostic queries only, never baseline measurements.","Board camera-rotation prediction ignores optical-centre parallax."])
    root=Path(r"D:\CodexData\RinkSpace_Data\events\ankara_2026\calibration\runs\solver_acceleration_20261003\source_corrected_RF_complete_joint_20261007")
    models={name:np.load(root/file,allow_pickle=False) for name,file in [("initial_9bd4","complete_physical_camera_model_UNACCEPTED_7b0edfe83f8c8c6e.npz"),("accepted_a8d2","complete_physical_camera_model_UNACCEPTED_07f5492daacd7c0a.npz")]}
    models["parent_3311"]=np.load(root.parent/"mainspin_all_tripods_outside_user_domain_v2_actual_fit_nodump_20261006"/"complete_physical_camera_model_UNACCEPTED_09900e4e4b41d05b.npz",allow_pickle=False)
    report['model_receipts']={name:dict(path=str(d.fid.name),sha256=sha(d.fid.name)) for name,d in models.items()}
    h,w=a.shape[:2]
    for material,boxes in [("board",[[1160,290,1890,570]]),("ice",[[1200,590,1890,915]])]:
        mask=np.zeros((h,w),np.uint8)
        for x0,y0,x1,y1 in boxes:mask[y0:y1,x0:x1]=255
        entry=dict(material=material,roi=boxes)
        try:
            p,q,fb,H,test,keep,res=track_pair(a,b,mask)
            entry.update(source_feature_pairs=len(p),test_point_count=int(test.sum()),baseline_test_error_px=stats(res[test]),actual_RGB_test_motion_px=stats(np.linalg.norm(q[test]-p[test],axis=1)),actual_RGB_test_median_vector=np.median(q[test]-p[test],axis=0).tolist(),H=H.tolist(),independent_SIFT=sift_pair(a,b,mask),models={})
            np.savez_compressed(out/f"independent_tracks_{material}.npz",p=p,q=q,fb=fb,H=H,test=test,baseline_keep=keep)
            for name,d in models.items():
                ca,cb=[d['camera25'][np.flatnonzero(d['pts']==s)[0]] for s in stamps]
                pp,v=model_transfer(ca,cb,p,0. if material=="ice" else None);selected=test&v
                entry['models'][name]=dict(all_test_valid_count=int(selected.sum()),model_test_error_px=stats(np.linalg.norm(pp[selected]-q[selected],axis=1)),same_test_baseline_error_px=stats(res[selected]),model_test_median_vector=np.median(pp[selected]-p[selected],axis=0).tolist(),actual_alias_pts=[int(d['exposure_representative_pts'][np.flatnonzero(d['pts']==s)[0]]) for s in stamps])
                ra=cv2.Rodrigues(ca[3:6])[0];rb=cv2.Rodrigues(cb[3:6])[0]
                rv=cv2.Rodrigues(rb@ra.T)[0].ravel()
                entry['models'][name]['relative_rotation_degrees']=float(np.linalg.norm(rv)*180/np.pi)
                entry['models'][name]['focal_change_percent']=float((np.exp(cb[6]-ca[6])-1)*100)
                entry['models'][name]['optical_center_difference_scene_units']=(cb[:3]-ca[:3]).tolist()
                cv2.imwrite(str(out/f"independent_{material}_{name}_model_error_x4.png"),draw(a,b,p,q,H,keep,q+(pp-q)/2,f"{material} {stamps[0]}->{stamps[1]} {name}: RGB track + model residual x4"))
        except RuntimeError as ex:entry['failure']=str(ex)
        report['independent_baselines'].append(entry)
    if args.motion_pairs:
        pair_path=Path(args.motion_pairs);raw=json.loads(pair_path.read_text(encoding='utf-8'))
        rows=raw if isinstance(raw,list) else raw.get('pair_rows',raw.get('pairs',raw.get('largest_motion_pair_cost_drops',[])))
        pairs=[r for r in rows if r.get('source_pts',raw.get('source_pts') if isinstance(raw,dict) else None)==stamps[0] and r.get('target_pts',raw.get('target_pts') if isinstance(raw,dict) else None)==stamps[1]]
        report['actual_motion_pair_source']=dict(path=str(pair_path),sha256=sha(pair_path),queried_pairs=len(pairs),not_used_for_independent_feature_detection=True)
        ga=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY);gb=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
        xy=np.asarray([r['source_uv'] for r in pairs],np.float32).reshape(-1,1,2)
        if len(xy):
            independently,s,e=cv2.calcOpticalFlowPyrLK(ga,gb,xy,None,winSize=(31,31),maxLevel=5,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,60,.003))
            back,t,ee=cv2.calcOpticalFlowPyrLK(gb,ga,independently,None,winSize=(31,31),maxLevel=5,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,60,.003))
            for i,r in enumerate(pairs):
                p=np.asarray(r['source_uv']);old=np.asarray(r['target_uv']);fresh=independently[i,0]
                x,y=np.rint(p).astype(int);half=15
                template=ga[y-half:y+half+1,x-half:x+half+1]
                x0=max(half,x-125);x1=min(w-half-1,x+125);y0=max(half,y-25);y1=min(h-half-1,y+25)
                search=gb[y0-half:y1+half+1,x0-half:x1+half+1]
                corr=cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED)
                _,peak,_,loc=cv2.minMaxLoc(corr);patch=np.array([x0+loc[0],y0+loc[1]],float)
                entry=dict(original_bundle_pair_index=r.get('original_bundle_pair_index'),source_uv=p.tolist(),old_target_uv=old.tolist(),fresh_LK_target_uv=fresh.tolist(),fresh_LK_valid=bool(s[i,0] and t[i,0]),fresh_LK_fb_error_px=float(np.linalg.norm(back[i,0]-p)),fresh_LK_vs_old_target_error_px=float(np.linalg.norm(fresh-old)),fresh_LK_displacement=(fresh-p).tolist(),source_patch_std=float(np.std(template)),independent_patch_search_box=[int(x0),int(y0),int(x1),int(y1)],patch_NCC_max=float(peak),patch_best_target_uv=patch.tolist(),patch_vs_old_target_error_px=float(np.linalg.norm(patch-old)))
                report['actual_old_motion_pairs'].append(entry)
            report['old_pairs_vs_fresh_LK_all_error_px']=stats([r['fresh_LK_vs_old_target_error_px'] for r in report['actual_old_motion_pairs']])
            report['old_pairs_vs_fresh_LK_valid_FB_under_point8_error_px']=stats([r['fresh_LK_vs_old_target_error_px'] for r in report['actual_old_motion_pairs'] if r['fresh_LK_valid'] and r['fresh_LK_fb_error_px']<.8])
            report['old_pairs_vs_independent_patch_all_error_px']=stats([r['patch_vs_old_target_error_px'] for r in report['actual_old_motion_pairs']])
            canvas=np.hstack([a[260:760,1100:1920],b[260:760,1100:1920]])
            for i,r in enumerate(report['actual_old_motion_pairs']):
                for off,point in [(0,r['source_uv']),(820,r['fresh_LK_target_uv'])]:
                    x,y=np.rint(point).astype(int);xy=(int(x-1100+off),int(y-260))
                    if 0<=xy[0]<canvas.shape[1] and 0<=xy[1]<canvas.shape[0]:cv2.circle(canvas,xy,5,(30,255,80),2);cv2.putText(canvas,str(i),xy,cv2.FONT_HERSHEY_SIMPLEX,.5,(10,20,255),1)
            cv2.imwrite(str(out/'actual_motion_source_and_fresh_RGB_tracks.png'),canvas)
    (out/'secondary_pair_native_RGB_probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False)[:12000])

if __name__=="__main__": main()
