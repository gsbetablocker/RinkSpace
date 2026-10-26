"""Carry a checked source-pattern identity through adjacent authentic exposures."""
from pathlib import Path
import sys,json,time,collections
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_entity_registration_s01 as reg
b=reg.b;OUT=reg.OUT
PRIMARY=[1,3,70,77,4,6,7,9,75,76,11,12,71,16,72,43,44,50,67,68]

def score(r):
    if r['method']=='same_authentic_source_exposure_identity':return 10.
    return r.get('full_pattern_score',{}).get('pattern_correlation_median',0)+min(r.get('prior_inliers',0),100)*.001

def step(ref,H,previous,pt):
    _,a,ma=reg.current(previous);_,c,mc=reg.current(pt);lo=ref['lo'];hi=ref['hi'];poly=cv2.perspectiveTransform(np.array([[lo,[hi[0],lo[1]],hi,[lo[0],hi[1]]]],np.float32),H)[0]
    if not np.isfinite(poly).all():return None,'source_chart_outside_view'
    roi=np.zeros_like(a);cv2.fillPoly(roi,[np.rint(poly).astype(int)],255);roi=cv2.dilate(roi,np.ones((9,9),np.uint8));roi[cv2.dilate(ma,np.ones((9,9),np.uint8))>0]=0;roi[:180]=0
    x=cv2.goodFeaturesToTrack(a,maxCorners=650,qualityLevel=.015,minDistance=5,mask=roi,blockSize=5)
    if x is None or len(x)<12:return None,'insufficient_unoccluded_identity_features'
    y,st,_=cv2.calcOpticalFlowPyrLK(a,c,x,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01));xx,sb,_=cv2.calcOpticalFlowPyrLK(c,a,y,None,winSize=(21,21),maxLevel=3)
    if y is None or xx is None:return None,'source_track_failed'
    old=x[:,0];new=y[:,0];ii=np.rint(new).astype(int);inside=(ii[:,0]>5)&(ii[:,0]<1914)&(ii[:,1]>185)&(ii[:,1]<1074);valid=st[:,0].astype(bool)&sb[:,0].astype(bool)&inside&(np.linalg.norm(xx[:,0]-old,axis=1)<.8)
    rr=np.flatnonzero(inside);mask=cv2.dilate(mc,np.ones((9,9),np.uint8));valid[rr]&=mask[ii[rr,1],ii[rr,0]]==0;old=old[valid];new=new[valid]
    if len(old)<12:return None,'insufficient_consistent_source_tracks'
    A,inlier=cv2.estimateAffine2D(old,new,method=cv2.RANSAC,ransacReprojThreshold=1.25,maxIters=2500,confidence=.999,refineIters=15)
    if A is None:return None,'source_feature_transform_unresolved'
    good=inlier[:,0].astype(bool);span=np.ptp(old[good],axis=0)
    if good.sum()<10 or good.mean()<.65 or min(span)<5 or cv2.contourArea(cv2.convexHull(old[good]))<60:return None,'source_feature_support_degenerate'
    T=np.vstack([A,[0,0,1]]);nextH=T@H
    if reg.geometry(nextH,ref) is None:return None,'source_transform_invalid'
    error=np.linalg.norm(cv2.transform(old[None],A)[0]-new,axis=1);return dict(H=nextH.tolist(),feature_inliers=int(good.sum()),feature_pairs=len(old),source_track_inlier_fraction=float(good.mean()),source_feature_span=span.tolist(),source_track_p95=float(np.percentile(error[good],95))),None

def run(anchor_file=None, output_file=None):
    anchor_file=Path(anchor_file) if anchor_file is not None else OUT/'source_pattern_anchors.json'
    output_file=Path(output_file) if output_file is not None else OUT/'source_identity_charts.json'
    cv2.setNumThreads(2);anchors=b.read(anchor_file);ats=b.read(b.OUT/'atlases.json')['atlases'];source_pts=sorted(reg.VIS);index={p:i for i,p in enumerate(source_pts)};output=collections.defaultdict(dict);failures=[];start=time.monotonic()
    # Direct source-pattern evidence remains available for every chart.
    for ai,pts in anchors['records'].items():
        for pt,r in pts.items():output[int(ai)][int(pt)]=dict(r,anchor_pts=int(pt),source_track_steps=0)
    for ai in PRIMARY:
        if str(ai) not in anchors['records']:continue
        ref=reg.reference(ats[ai]);available={int(pt):r for pt,r in anchors['records'][str(ai)].items()};seen=set();seeds=sorted(available,key=lambda p:-score(available[p]))
        for seed in seeds:
            if seed in seen:continue
            seen.add(seed)
            for direction in [-1,1]:
                H=np.asarray(available[seed]['H'],float);prev=seed;last_anchor=seed;since_anchor=0
                for j in range(index[seed]+direction,len(source_pts) if direction>0 else -1,direction):
                    pt=source_pts[j]
                    if pt in seen:break
                    result,reason=step(ref,H,prev,pt)
                    if result is None:
                        failures.append(dict(atlas_id=ai,previous_pts=prev,pts=pt,reason=reason));break
                    proposal=np.asarray(result['H']);nextH=proposal;method='adjacent_actual_RGB_identity_track';disagreement=None
                    if pt in available:
                        direct=np.asarray(available[pt]['H']);q=ref['points'][ref['probe']];a=cv2.perspectiveTransform(q[None],proposal)[0];d=cv2.perspectiveTransform(q[None],direct)[0];delta=np.linalg.norm(a-d,axis=1);disagreement=float(np.median(delta))
                        # A direct chart cannot silently jump by a letter. Strong same-exposure references
                        # can re-anchor; a disagreement remains an explicit source-identity conflict.
                        if disagreement<3 or available[pt]['method']=='same_authentic_source_exposure_identity':
                            nextH=direct;method='adjacent_RGB_track_consistent_with_direct_source_pattern';last_anchor=pt;since_anchor=0
                        else:
                            output[ai][pt]['source_track_conflict_px']=disagreement
                            if disagreement>8:
                                failures.append(dict(atlas_id=ai,previous_pts=prev,pts=pt,reason='direct_chart_disagrees_with_adjacent_source_identity',disagreement_px=disagreement));break
                    since_anchor+=1
                    if pt not in available:
                        output[ai][pt]=dict(result,H=nextH.tolist(),atlas_id=ai,pts=pt,method=method,anchor_pts=last_anchor,previous_pts=prev,source_track_steps=since_anchor,root_seed_pts=seed,source_BGR_sha256=reg.VIS[pt]['source_BGR_sha256'],navigation_not_observed_UV=True)
                    else:output[ai][pt]['adjacent_source_check']=dict(previous_pts=prev,disagreement_px=disagreement,feature_inliers=result['feature_inliers'],p95=result['source_track_p95'])
                    seen.add(pt);H=nextH;prev=pt
        print(json.dumps(dict(stage='source_identity_tracks',atlas=ai,ad=ats[ai]['ad'],direct_anchors=len(available),mapped_source_frames=len(output[ai]),seconds=round(time.monotonic()-start,1))),flush=True)
    b.write(output_file,dict(schema='rinkspace.actual_source_identity_charts.v1',records={str(ai):{str(pt):r for pt,r in pts.items()} for ai,pts in output.items()},tracking_failures=failures,source_anchor_file_sha256=b.sha(anchor_file),native_projection_used=False,measurements_created_by_tracking=0))
    print(json.dumps(dict(stage='source_identity_tracks_done',mapped_chart_frames=sum(map(len,output.values())),failures=len(failures))),flush=True)
if __name__=='__main__':run()
