"""Bridge original print-image correspondences through a real same-shot image.

The final registration is fitted to original canonical SIFT points and actual
forward/backward-checked RGB tracks in the target exposure. No 3D projection.
"""
from pathlib import Path
import json
import av,cv2,numpy as np
from .dense_joint_proofs import _ref
from .io_utils import write_json


def bridge(*,source_video,atlas,registrations,bridge_image,bridge_pts,local_registrations,output):
    if Path(output).exists():raise FileExistsError('Immutable bridge registration exists')
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
    a=read(atlas);base=read(registrations);local={int(x['native_pts']):x for x in read(local_registrations)}
    anchor=next(x for x in base if int(x['native_pts'])==bridge_pts)['candidates'][0]
    original=cv2.imread(a['source_image']['file'],0);middle=cv2.imread(str(bridge_image),0)
    src=np.asarray(anchor['source_inliers'],np.float32);mid=np.asarray(anchor['target_inliers'],np.float32)
    x,y,X,Y=map(int,a['crop']);refcrop=original[y:Y,x:X];T=np.array([[1.,0,-x],[0,1.,-y],[0,0,1.]])
    wanted={pt for pt,x in local.items()if x['candidates']};new={};criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,60,.01)
    if wanted:
      with av.open(str(source_video))as con:
        stream=con.streams.video[0];con.seek(min(wanted),stream=stream,backward=True)
        for frame in con.decode(stream):
            pt=int(frame.pts)
            if pt>max(wanted):break
            if pt not in wanted:continue
            gray=frame.to_ndarray(format='gray');candidates=[]
            for first in local[pt]['candidates']:
                Hmid=np.asarray(first['H'],float);initial=cv2.perspectiveTransform(mid[None],Hmid)[0].astype(np.float32)
                target,status,_=cv2.calcOpticalFlowPyrLK(middle,gray,mid[:,None],initial[:,None].copy(),winSize=(31,31),maxLevel=4,criteria=criteria,flags=cv2.OPTFLOW_USE_INITIAL_FLOW,minEigThreshold=1e-5)
                if target is None:continue
                back,back_status,_=cv2.calcOpticalFlowPyrLK(gray,middle,target,mid[:,None].copy(),winSize=(31,31),maxLevel=4,criteria=criteria,flags=cv2.OPTFLOW_USE_INITIAL_FLOW,minEigThreshold=1e-5)
                if back is None:continue
                target=target[:,0];fb=np.linalg.norm(back[:,0]-mid,axis=1)
                valid=(status[:,0]>0)&(back_status[:,0]>0)&(fb<=2.)&np.isfinite(target).all(1)&(target[:,0]>=0)&(target[:,0]<gray.shape[1])&(target[:,1]>=0)&(target[:,1]<gray.shape[0])
                if valid.sum()<8:continue
                p=src[valid];q=target[valid];fits=[]
                A,inside=cv2.estimateAffine2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=3.,maxIters=5000,confidence=.999)
                if A is not None:fits.append(('affine',np.r_[A,[[0,0,1]]],inside))
                H,inside=cv2.findHomography(p,q,cv2.RANSAC,3.,maxIters=5000,confidence=.999)
                if H is not None:fits.append(('homography',H,inside))
                for kind,H,inside in fits:
                    inside=inside[:,0].astype(bool);n=int(inside.sum())
                    if n<8:continue
                    points=p[inside];hull=cv2.convexHull(points).reshape(-1,2);span=np.ptp(points,axis=0);den=np.c_[hull,np.ones(len(hull))]@H[2]
                    if min(den)*max(den)<=0 or np.linalg.det(H)/(den.mean()**3)<=0 or span[0]<40 or span[1]<10 or cv2.contourArea(hull)<150:continue
                    inv=T@np.linalg.inv(H);warp=cv2.warpPerspective(gray,inv,(X-x,Y-y));visible=cv2.warpPerspective(np.ones(gray.shape,np.uint8),inv,(X-x,Y-y))>0;support=np.zeros(refcrop.shape,np.uint8);cv2.fillConvexPoly(support,np.rint(hull-[x,y]).astype(np.int32),255);use=visible&(support>0);aa=refcrop[use].astype(float);bb=warp[use].astype(float);corr=float(np.corrcoef(aa,bb)[0,1])if len(aa)>100 and aa.std()>5 and bb.std()>5 else -1.
                    error=np.linalg.norm(cv2.perspectiveTransform(points[None],H)[0]-q[inside],axis=1);p95=float(np.quantile(error,.95))
                    if corr<.65 or p95>3.:continue
                    candidates.append(dict(kind=kind,H=H.tolist(),inliers=n,correlation=corr,registration_p95_px=p95,reference_pts=a['native_pts'],source_convex_hull=hull.tolist(),source_inliers=points.tolist(),target_inliers=q[inside].tolist(),actual_image_bridge_pts=bridge_pts,bridge_forward_backward_max_px=float(fb[valid][inside].max()),source_bridge_inlier_image=_ref(bridge_image),original_registration=_ref(registrations),local_registration=_ref(local_registrations),reader=_ref(__file__),only_inlier_footprint_authorized=True,whole_panel_registration_claimed=False,source_3D_projection_used=False))
            candidates.sort(key=lambda x:(x['correlation'],x['inliers']),reverse=True);new[pt]=candidates
    for entry in base:
        pt=int(entry['native_pts'])
        if not entry['candidates']and new.get(pt):entry['candidates']=new[pt]
    write_json(output,base)
    return dict(original_supported=sum(bool(x['candidates'])for x in read(registrations)),new_supported=sum(bool(x['candidates'])for x in base),added_supported=sum(bool(x)for x in new.values()),actual_native_image_bridge_pts=bridge_pts,source_3D_projection_used=False)
