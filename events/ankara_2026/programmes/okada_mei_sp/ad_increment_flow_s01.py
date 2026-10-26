"""Actual-exposure feature tracks carry source-pattern identity into adjacent exposures."""
from pathlib import Path
import sys,cv2,numpy as np,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_increment_contours_s01 as c
b=c.b;OUT=c.OUT

def run():
    if (OUT/'increment_manifest.json').exists():raise ValueError('Frozen increment cannot be rewritten; prepare a new increment')
    cv2.setNumThreads(2);d=b.load();vis={f['pts']:f for f in b.read(b.OUT/'visibility.json')['frames']};pts=sorted(map(int,np.unique(d['pts'][d['ad']==7])));anchor=445440;refproof=b.OUT/'verified_final'/f'ad7_{anchor}.json';base=next(r for r in b.read(refproof)['registrations'] if r['atlas_id']==79 and r['method']=='independent_actual_source_SIFT_affine');H0=np.asarray(base['H']);records={};steps=[]
    def frame(pt):
        f=vis[pt];im=cv2.imread(f['image'])
        if hashlib.sha256(im.tobytes()).hexdigest()!=f['source_BGR_sha256']:raise ValueError('Source changed')
        return cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),cv2.imread(f['mask_path'],0)
    for direction in [-1,1]:
        H=H0.copy();previous=anchor;gray,person=frame(anchor);indices=pts[:pts.index(anchor)][::-1] if direction<0 else pts[pts.index(anchor)+1:]
        for pt in indices:
            current,occlusion=frame(pt);poly=cv2.perspectiveTransform(np.array([[[0,0],[2500,0],[2500,340],[0,340]]],np.float32),H)[0];roi=np.zeros_like(gray);cv2.fillPoly(roi,[np.rint(poly).astype(int)],255);roi[cv2.dilate(person,np.ones((11,11),np.uint8))>0]=0
            features=cv2.goodFeaturesToTrack(gray,maxCorners=700,qualityLevel=.015,minDistance=5,mask=roi,blockSize=5)
            if features is None or len(features)<14:break
            nextp,status,err=cv2.calcOpticalFlowPyrLK(gray,current,features,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01));back,sb,_=cv2.calcOpticalFlowPyrLK(current,gray,nextp,None,winSize=(21,21),maxLevel=3)
            x=features[:,0];y=nextp[:,0];rounded=np.rint(y).astype(int);inside=(rounded[:,0]>=6)&(rounded[:,0]<1914)&(rounded[:,1]>=6)&(rounded[:,1]<1074);valid=status[:,0].astype(bool)&sb[:,0].astype(bool)&inside&(np.linalg.norm(back[:,0]-x,axis=1)<1.0)
            inbounds=np.flatnonzero(inside);dilated=cv2.dilate(occlusion,np.ones((11,11),np.uint8));valid[inbounds]&=dilated[rounded[inbounds,1],rounded[inbounds,0]]==0
            x=x[valid];y=y[valid]
            if len(x)<14:break
            A,inliers=cv2.estimateAffine2D(x,y,method=cv2.RANSAC,ransacReprojThreshold=1.25,maxIters=2000,confidence=.995,refineIters=15)
            if A is None:break
            good=inliers[:,0].astype(bool)
            if good.sum()<12 or good.mean()<.65:break
            span=np.ptp(x[good],axis=0)
            if min(span)<20:break
            step=np.vstack([A,[0,0,1]]);H=step@H;res=np.linalg.norm(cv2.transform(x[None],A)[0]-y,axis=1);reg=dict(H=H.tolist(),atlas_id=79,inliers=int(good.sum()),p95=float(np.percentile(res[good],95)),method='actual_source_adjacent_exposure_forward_backward_feature_track',anchor_pts=anchor,previous_pts=previous,source_identity_anchor=str(refproof),source_identity_anchor_sha256=b.sha(refproof),source_BGR_sha256=vis[pt]['source_BGR_sha256'],tracking_not_used_as_observed_UV=True)
            if not c.usable(reg):break
            records[str(pt)]=reg;steps.append(dict(pts=pt,previous_pts=previous,source_pairs=len(x),inliers=int(good.sum()),inlier_ratio=float(good.mean()),span=span.tolist(),p95=reg['p95']));previous=pt;gray=current;person=occlusion
    b.write(OUT/'source_flow_navigation.json',dict(anchor_pts=anchor,anchor_registration=base,records=records,steps=steps,purpose='Identity navigation only. Every exported observation must be remeasured from that exposure RGB.',native_geometry_used=False));print(json.dumps(dict(navigation_frames=len(records),min_pts=min(map(int,records)),max_pts=max(map(int,records)))),flush=True)
if __name__=='__main__':run()
