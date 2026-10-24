"""Independent local original-image registration for print identity evidence.

Source pixels only navigate the search ROI. No fitted camera, world coordinates,
material residuals, alternate pixel measurements or persistent model correction.
"""
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import hashlib,time
import av,cv2,numpy as np
from .io_utils import write_json


def register(*,source_video,reference_image,reference_pts,crop,targets,output,workers=4):
    if Path(output).exists():raise FileExistsError('Immutable registration exists')
    cv2.setNumThreads(1);original=cv2.imread(str(reference_image),0)
    x,y,X,Y=map(int,crop);rcrop=original[y:Y,x:X]
    T=np.array([[1.,0,-x],[0,1.,-y],[0,0,1.]])
    def features(gray,origin):
        enlarged=cv2.resize(gray,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
        k,d=cv2.SIFT_create(nfeatures=6000,contrastThreshold=.008).detectAndCompute(enlarged,None)
        return np.array([z.pt for z in k],np.float32)/3+origin,d
    rxy,dr=features(rcrop,np.array([x,y]));started=time.perf_counter()
    def work(pt,gray):
        source_uv=np.asarray(targets[pt],float);lo=np.maximum([0,0],np.floor(source_uv.min(0)-80)).astype(int);hi=np.minimum([gray.shape[1],gray.shape[0]],np.ceil(source_uv.max(0)+80)).astype(int)
        result=dict(native_pts=pt,candidates=[],source_gray_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),source_UV_used_for_ROI_only=True)
        if np.any(hi<=lo):return result
        k,dt=features(gray[lo[1]:hi[1],lo[0]:hi[0]],lo)
        if dt is None or dr is None or len(dt)<2 or len(dr)<2:return result
        matcher=cv2.BFMatcher();forward=[a for a,b in matcher.knnMatch(dr,dt,k=2)if a.distance<.78*b.distance]
        reverse={a.queryIdx:a.trainIdx for a,b in matcher.knnMatch(dt,dr,k=2)if a.distance<.78*b.distance}
        good=[a for a in forward if reverse.get(a.trainIdx)==a.queryIdx]
        result['mutual_matches']=len(good)
        if len(good)<8:return result
        src=rxy[[g.queryIdx for g in good]];dst=k[[g.trainIdx for g in good]];fits=[]
        A,inside=cv2.estimateAffine2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=3.,maxIters=5000,confidence=.999)
        if A is not None:fits.append(('affine',np.r_[A,[[0,0,1]]],inside))
        H,inside=cv2.findHomography(src,dst,cv2.RANSAC,3.,maxIters=5000,confidence=.999)
        if H is not None:fits.append(('homography',H,inside))
        for kind,H,inside in fits:
            inside=inside.ravel().astype(bool);n=int(inside.sum());points=src[inside].astype(np.float32)
            if n<8:continue
            hull=cv2.convexHull(points).reshape(-1,2);span=np.ptp(points,axis=0);den=np.c_[hull,np.ones(len(hull))]@H[2]
            if min(den)*max(den)<=0 or np.linalg.det(H)/(den.mean()**3)<=0 or span[0]<40 or span[1]<10 or cv2.contourArea(hull)<150:continue
            warp=cv2.warpPerspective(gray,T@np.linalg.inv(H),(X-x,Y-y));valid=cv2.warpPerspective(np.ones(gray.shape,np.uint8),T@np.linalg.inv(H),(X-x,Y-y))>0;support=np.zeros(rcrop.shape,np.uint8);cv2.fillConvexPoly(support,np.rint(hull-[x,y]).astype(np.int32),255);use=valid&(support>0);a=rcrop[use].astype(float);b=warp[use].astype(float)
            corr=float(np.corrcoef(a,b)[0,1])if len(a)>100 and a.std()>5 and b.std()>5 else -1.
            residual=np.linalg.norm(cv2.perspectiveTransform(points[None],H)[0]-dst[inside],axis=1);p95=float(np.quantile(residual,.95))
            if corr<.65 or p95>3.:continue
            result['candidates'].append(dict(kind=kind,H=H.tolist(),inliers=n,correlation=corr,registration_p95_px=p95,reference_pts=int(reference_pts),source_convex_hull=hull.tolist(),source_inliers=points.tolist(),target_inliers=dst[inside].tolist(),only_inlier_footprint_authorized=True,whole_panel_registration_claimed=False))
        result['candidates'].sort(key=lambda q:(q['correlation'],q['inliers']),reverse=True);return result
    results=[];pending=set()
    with ThreadPoolExecutor(max_workers=workers)as pool,av.open(str(source_video))as con:
        stream=con.streams.video[0];stream.thread_count=1
        if str(stream.time_base)!='1/12800':raise ValueError('Native clock differs')
        con.seek(min(targets),stream=stream,backward=True)
        for frame in con.decode(stream):
            pt=int(frame.pts)
            if pt>max(targets):break
            if pt not in targets:continue
            pending.add(pool.submit(work,pt,frame.to_ndarray(format='gray')))
            if len(pending)>=2*workers:
                done,pending=wait(pending,return_when=FIRST_COMPLETED);results.extend(f.result()for f in done)
            if len(results)and len(results)%100==0:print('source_registration',len(results),'supported',sum(bool(x['candidates'])for x in results),'seconds',round(time.perf_counter()-started,1),flush=True)
        results.extend(f.result()for f in pending)
    if {x['native_pts']for x in results}!=set(targets):raise ValueError('Source registration exposure census incomplete')
    write_json(output,sorted(results,key=lambda x:x['native_pts']))
    return dict(frames=len(results),supported=sum(bool(x['candidates'])for x in results),seconds=time.perf_counter()-started,reference_features=len(rxy),source_projection_used=False)
