"""Source-image support-blade tracking; propagated pixels remain derived evidence."""
from pathlib import Path
import json
import numpy as np
import cv2
from rinkspace.viewer3d.source import SourceVideo
from .rebuild_prepare import digest
from .fullbody_round import write_json


def _flow_track(images,start,stop,centre):
    n=len(images);out=np.full((n,2),np.nan);quality=np.zeros(n);out[start]=centre;quality[start]=1
    step=1 if stop>start else -1;current=np.array(centre,float)
    for i in range(start,stop,step):
        j=i+step;mask=np.zeros(images[i].shape,np.uint8);x,y=np.rint(current).astype(int)
        cv2.rectangle(mask,(x-74,y-95),(x+74,y-5),255,-1)
        pts=cv2.goodFeaturesToTrack(images[i],maxCorners=80,qualityLevel=.015,minDistance=3,mask=mask,blockSize=5)
        if pts is None or len(pts)<6:break
        forward,status,error=cv2.calcOpticalFlowPyrLK(images[i],images[j],pts,None,winSize=(25,25),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
        back,back_status,_=cv2.calcOpticalFlowPyrLK(images[j],images[i],forward,None,winSize=(25,25),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
        fb=np.linalg.norm(back[:,0]-pts[:,0],axis=1);good=status[:,0].astype(bool)&back_status[:,0].astype(bool)&(fb<1.5)
        if good.sum()<6:break
        M,inliers=cv2.estimateAffinePartial2D(pts[good,0],forward[good,0],method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=1000,confidence=.99)
        if M is None or inliers.sum()<6:break
        current=M[:,:2]@current+M[:,2];out[j]=current;quality[j]=float(inliers.mean())*float(np.exp(-np.median(fb[good])))
    return out,quality


def build_track(project,base,old_output,output):
    output.mkdir(parents=True,exist_ok=True);cache=output/'source_support_track.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False)as z:return {k:z[k]for k in z.files}
    programme=project/'events/ankara_2026/programmes/okada_mei_sp'
    anchors=json.loads((programme/'spin_support_midpoint_anchors_1104.json').read_text(encoding='utf8'))
    check=json.loads((old_output/'source_review/support_direct_check_v2.json').read_text(encoding='utf8'))
    updates={r['native_pts']:r for r in check['frames'] if r['blade_midpoint_px']is not None}
    rows=[dict(r)for r in anchors['anchors']if 82.4*12800<=r['pts']<=92.46*12800]
    for row in rows:
        if row['pts']in updates:
            row['centre_uv']=updates[row['pts']]['blade_midpoint_px'];row['region_half_extent_uv']=[8,7]
        row['feature_semantics']='visible support blade middle region; not ankle, not moving geometric minimum'
    original=json.loads((old_output/'support_and_free_shoe_regions.json').read_text(encoding='utf8'))
    review={int(r['pts'])for r in original['observations']if r['role']=='review'}
    excluded={p+d for p in review for d in (-256,0,256)}
    with np.load(base/'action_integration_1103/source_cache/spin_s05.native50.npz',allow_pickle=False)as z:allpts=z['pts']
    wanted=allpts[(allpts>=rows[0]['pts'])&(allpts<=rows[-1]['pts'])];pts=np.array([p for p in wanted if int(p)not in excluded],np.int64)
    assert all(r['pts']not in excluded for r in rows)
    crop=np.array([700,640,1400,1040]);origin=crop[:2]
    source=json.loads((base/'action_integration_1103/spin_s05/contract.json').read_text(encoding='utf8'))['source']['path']
    video=SourceVideo(source,(1,12800));images=[]
    try:
        for pt in pts:
            rgb=video.read(int(pt));images.append(cv2.cvtColor(rgb[crop[1]:crop[3],crop[0]:crop[2]],cv2.COLOR_RGB2GRAY))
    finally:video.close()
    images=np.asarray(images);uv=np.full((len(pts),2),np.nan);confidence=np.zeros(len(pts));bidirectional=[]
    lookup={int(p):i for i,p in enumerate(pts)}
    for a,b in zip(rows[:-1],rows[1:]):
        ia,ib=lookup[a['pts']],lookup[b['pts']];pa=np.array(a['centre_uv'])-origin;pb=np.array(b['centre_uv'])-origin
        forward,qf=_flow_track(images,ia,ib,pa);back,qb=_flow_track(images,ib,ia,pb)
        idx=np.arange(ia,ib+1);u=(pts[idx]-pts[ia])/(pts[ib]-pts[ia]);goodf=np.isfinite(forward[idx]).all(1);goodb=np.isfinite(back[idx]).all(1)
        # Blend source-material flow in each direction; no image-derived
        # coordinate is supplied by the frozen review frames or their neighbours.
        w1=(1-u)*qf[idx]*goodf;w2=u*qb[idx]*goodb;den=w1+w2
        middle=np.zeros((len(idx),2));use=den>0
        middle[use]=(np.nan_to_num(forward[idx][use])*w1[use,None]+np.nan_to_num(back[idx][use])*w2[use,None])/den[use,None]
        both=goodf&goodb;disagreement=np.full(len(idx),np.nan);disagreement[both]=np.linalg.norm(forward[idx][both]-back[idx][both],axis=1)
        reliability=np.clip(den,0,1);reliability[both]*=np.exp(-disagreement[both]/35.)
        # If feature flow fails, keep an explicit unknown, not an invented point.
        use&=reliability>.08;uv[idx[use]]=middle[use]+origin;confidence[idx[use]]=reliability[use]
        uv[ia]=a['centre_uv'];uv[ib]=b['centre_uv'];confidence[[ia,ib]]=1
        bidirectional.append(dict(start_pts=a['pts'],end_pts=b['pts'],valid_frames=int(use.sum()),total_frames=len(idx),median_forward_backward_disagreement_px=float(np.nanmedian(disagreement))if both.any()else None))
    native=pts.copy();valid=np.isfinite(uv).all(1);assert np.all(np.diff(pts)>0)
    # Source pixels at 10 Hz plus the admitted seed frames enter optimization.
    # Native-50Hz flow remains available for initialization and independent plots.
    chosen=valid&((np.arange(len(pts))%5)==0);chosen|=np.isin(pts,[r['pts']for r in rows])
    np.savez_compressed(cache,pts=pts,uv=uv,confidence=confidence,valid=valid,optimization_use=chosen,native_crop_xyxy=crop,time_base=np.array([1,12800]))
    document=dict(schema='rinkspace.source-support-derived-track.v1',source_sha256=anchors['source_sha256'],shot_id='s05',time_base=[1,12800],pixel_domain='native Brown 1920x1080',support_foot=1,source_path=source,
      source_anchor_rows=rows,excluded_review_pts=sorted(excluded),intervals=bidirectional,frames=len(pts),valid_flow_frames=int(valid.sum()),optimization_exposures=int(chosen.sum()),
      provenance='adjacent source-image KLT forward/backward + robust local affine propagation from admitted blade-middle development regions; no body/network predictions used as blade observations',
      not_per_frame_manual_truth=True,no_review_pixels_consumed=True,unknown_pts=pts[~valid].tolist(),
      uncertainties=['flow follows nearby boot/steel material features; longitudinal blade midpoint changes appearance under rotation','partial visibility and source blur','confidence is an engineering reliability score, not calibrated probability'],
      input_anchor_document_sha256=digest(programme/'spin_support_midpoint_anchors_1104.json'),track_npz_sha256=digest(cache))
    write_json(output/'source_support_track.json',document)
    print(json.dumps(dict(event='source_support_track_ready',frames=len(pts),valid=int(valid.sum()),optimization_exposures=int(chosen.sum()),unknown=int((~valid).sum()))),flush=True)
    return dict(pts=pts,uv=uv,confidence=confidence,valid=valid,optimization_use=chosen)
