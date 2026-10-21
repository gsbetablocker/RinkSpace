import os
for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib,av,cv2,numpy as np
O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_v1');R=O.parent
items=json.loads((O/'historical_input.json').read_text(encoding='utf8'));scope=json.loads(Path(r'D:/CodexData/RinkSpace_Data/runs/okada_joint9_1018a/programme_scope.json').read_text(encoding='utf8'));source=scope['source'];cv2.setNumThreads(1)
unique={r['source_raw_frame']:r for r in items};dims=(320,180);valid=np.ones((180,320),bool);valid[3:44,3:95]=False;valid[154:170,275:318]=False

def normalized(im):
 a=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),dims,interpolation=cv2.INTER_AREA).astype('float32');a=cv2.GaussianBlur(a,(3,3),0)[valid];a-=a.mean();return a/max(float(np.linalg.norm(a)),1e-20)
old=[cv2.imread(p)for p in unique];assert all(x is not None for x in old);vec=np.stack([normalized(a)for a in old]);best=[dict(score=-2.)for a in old];scores=[]
with av.open(source['path'])as c:
 st=c.streams.video[0];c.seek(948992,stream=st,backward=True)
 for frame in c.decode(st):
  p=int(frame.pts)
  if p<948992:continue
  if p>1425408:break
  raw=frame.to_ndarray(format='bgr24');v=vec@normalized(raw);scores.append([p,*v.tolist()])
  for i,s in enumerate(v):
   if float(s)>best[i]['score']:best[i]=dict(score=float(s),native_pts=p)
np.save(O/'historical_image_search_scores.npy',np.asarray(scores));wanted={x['native_pts']for x in best};frames={}
with av.open(source['path'])as c:
 st=c.streams.video[0];c.seek(min(wanted),stream=st,backward=True)
 for frame in c.decode(st):
  p=int(frame.pts)
  if p>max(wanted):break
  if p in wanted:frames[p]=frame.to_ndarray(format='bgr24')
matcher=cv2.BFMatcher(cv2.NORM_L2);sift=cv2.SIFT_create();guides=[];matches=[]
for i,(path,r)in enumerate(unique.items()):
 p=best[i]['native_pts'];a=old[i];b=frames[p];ga=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY);gb=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY);mask=np.ones(ga.shape,'uint8')*255;mask[10:265,10:570]=0;mask[925:1020,1650:1919]=0;ka,da=sift.detectAndCompute(ga,mask);kb,db=sift.detectAndCompute(gb,mask);pairs=matcher.knnMatch(da,db,k=2);good=[m for m,n in pairs if m.distance<.75*n.distance];pa=np.float64([ka[m.queryIdx].pt for m in good]);pb=np.float64([kb[m.trainIdx].pt for m in good]);H,inside=cv2.findHomography(pa,pb,cv2.USAC_MAGSAC,3.);assert H is not None;inside=inside[:,0]>0;projected=cv2.perspectiveTransform(pa[None],H)[0];err=np.linalg.norm(projected-pb,axis=1);ref=dict(old_native_pts=r['native_pts'],old_source_clock=r['source_clock'],old_source_raw_frame=path,official_native_pts=p,official_time_base='1/12800',official_BGR_sha256=hashlib.sha256(b.tobytes()).hexdigest(),coarse_photometric_correlation=best[i]['score'],actual_image_matches=len(good),actual_image_inliers=int(inside.sum()),inlier_error_native_px=dict(median=float(np.median(err[inside])),p95=float(np.quantile(err[inside],.95))),new_image_from_old_image_H=H.tolist(),clock_offset_not_assumed=True,source_position_guidance_only_not_new_measurement=True)
 matches.append(ref);cv2.imwrite(str(O/f'historical_match_{p}.jpg'),cv2.resize(np.concatenate((a,b),axis=1),(1920,540)),[cv2.IMWRITE_JPEG_QUALITY,94])
 for it in items:
  if it['source_raw_frame']!=path:continue
  doc=json.loads(Path(it['file']).read_text(encoding='utf8'));uv=np.asarray(doc['native_source_uv'],float);new=cv2.perspectiveTransform(uv[None],H)[0];guides.append(dict(feature_id=it['feature_id'],curve=it['curve'],official_native_pts=p,old_annotation_file=it['file'],old_annotation_sha256=hashlib.sha256(Path(it['file']).read_bytes()).hexdigest(),old_source_native_uv=uv.tolist(),official_source_navigation_uv=new.tolist(),user_statement=it['user_statement'],registration=ref,measurement_origin='registered_historical_user_guidance_only',fresh_current_RGB_midpoint_or_material_interface_required=True,do_not_emit_as_observation_or_independent_point=True))
result=dict(schema='rinkspace.old_user_guidance_actual_image_registration.v1',source=source,official_search_first_pts=948992,official_search_last_pts=1425408,actual_native_frames_searched=len(scores),matches=matches,guides=guides,fixed_source_clock_conversion_used=False,new_observations_issued=False)
(O/'historical_guidance_registered.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(dict(matches=matches,guidance_count=len(guides)),ensure_ascii=False))
