"""Stateless source-to-current board-object navigation, never UV interpolation."""
from pathlib import Path
import argparse,json,hashlib
from collections import Counter
import cv2,numpy as np
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256
from .role_guard import FrameRoleGuard

def _H(q,H):return cv2.perspectiveTransform(np.asarray(q,np.float64)[None],H)[0]
def transfer(project,run,anchor_review,outdir):
 run=Path(run);out=Path(outdir);ar=Path(anchor_review);original=read_json(ar);guard=FrameRoleGuard(run/'roles/frame_roles.json');manifest=read_json(run/'native_selected_v2/manifest.json');data=next(p for p in run.parents if p.name=='RinkSpace_Data');scope=read_json(run/'programme_scope.json')
 if out.exists():raise ValueError('New immutable transfer required')
 out.mkdir(parents=True);cv2.setNumThreads(1);sift=cv2.SIFT_create(7000,contrastThreshold=.008);anchor={};rows=[];decisions=[];counts=Counter();frames_by_id={f['shot_id']:[]for f in original['frames']}
 for f in original['frames']:
  pts=int(f['native_pts']);sid=f['shot_id'];guard.assert_training_pixels([pts],shot_id=sid)
  if file_sha256(f['native_png'])!=f['native_png_sha256']:raise ValueError('Actual reviewed anchor PNG changed')
  im=cv2.imread(f['native_png']);mask=np.zeros(im.shape[:2],np.uint8);x0,y0,x1,y1=f['board_instance_roi_xyxy'];y0=max(y0,f['top_blue_upper_search_y_range'][1]+4);y1=min(y1,f['contact_yellow_lower_search_y_range'][0]-4);mask[y0:y1,x0:x1]=255;f=dict(f,descriptor_body_roi_xyxy=[x0,y0,x1,y1]);nr=int(sid[1:]);mp=data/'events/ankara_2026/programmes/okada_mei_sp/results'/f'shot{nr}_review_20261014';mp=mp/'r2/frame_manifest.json'if nr==2 else mp/'frame_manifest.json'
  fm={int(q.get('native_pts',q.get('source_pts'))):q for q in read_json(mp)['frames']};frames_by_id[sid]=fm;mf=fm[pts];pm=cv2.imread(mf.get('mask_path')or mf['source']['mask_path'],0)>0;mask[pm]=0;kp,d=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),mask);points=np.asarray([k.pt for k in kp]);good=(points[:,0]>=x0+1)&(points[:,0]<x1-1)&(points[:,1]>=y0+1)&(points[:,1]<y1-1);anchor[sid]=(f,points[good],d[good])
 for fr in manifest['frames']:
  pts=int(fr['native_pts']);sid=fr['shot_id']
  if sid not in anchor:continue
  guard.assert_training_pixels([pts],shot_id=sid);a,A,da=anchor[sid]
  if pts==int(a['native_pts']):decisions.append(dict(native_pts=pts,shot_id=sid,status='original_direct_current_board_review_retained'));continue
  png=run/'native_selected_v2'/fr['native_png']
  if file_sha256(png)!=fr['native_png_sha256']:raise ValueError('Actual current PNG changed')
  im=cv2.imread(str(png));bgrsha=hashlib.sha256(im.tobytes()).hexdigest();rgbsha=hashlib.sha256(np.ascontiguousarray(im[:,:,::-1]).tobytes()).hexdigest();mf=frames_by_id[sid][pts];mp=Path(mf.get('mask_path')or mf['source']['mask_path']);msha=mf.get('mask_sha256')or mf.get('source',{}).get('mask_sha256')
  if msha and file_sha256(mp)!=msha:raise ValueError('Actual current mask changed')
  pm=cv2.imread(str(mp),0)>0;mask=(~pm).astype(np.uint8)*255;kp,db=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),mask);B=np.asarray([k.pt for k in kp]);reason=None
  if da is None or db is None:reason='actual_source_descriptors_missing';matches=[]
  else:
   matcher=cv2.BFMatcher();fw=matcher.knnMatch(da,db,k=2);bw=matcher.knnMatch(db,da,k=2);back={m.queryIdx:m.trainIdx for m,n in bw if m.distance<.7*n.distance};matches=[m for m,n in fw if m.distance<.7*n.distance and back.get(m.trainIdx)==m.queryIdx]
  receipt=dict(native_pts=pts,shot_id=sid,native_png=str(png),native_png_sha256=fr['native_png_sha256'],decoded_native_BGR_sha256=bgrsha,actual_native_RGB_sha256=rgbsha,anchor_native_pts=int(a['native_pts']),anchor_native_png=a['native_png'],anchor_native_png_sha256=a['native_png_sha256'],anchor_board_review=dict(path=str(ar),sha256=file_sha256(ar)),anchor_descriptor_body_roi_xyxy=a['descriptor_body_roi_xyxy'],actual_mutual_RGB_matches=len(matches),source_clock='same official 1/12800 native PTS',role_policy='both actual exposures individually prelocked development; stateless direct image comparison, no tracking or interpolation through intervening frames',homography_is_navigation_only=True,pixels_from_model=False)
  if not reason and len(matches)<30:reason='insufficient_unique_actual_board_RGB_correspondence'
  if not reason:
   aa=np.asarray([A[m.queryIdx]for m in matches]);bb=np.asarray([B[m.trainIdx]for m in matches]);H,keep=cv2.findHomography(aa,bb,cv2.RANSAC,2.5);keep=keep.reshape(-1).astype(bool)if keep is not None else np.zeros(len(aa),bool)
   if H is None or keep.sum()<30:reason='actual_correspondence_not_one_supported_source_object'
   else:
    aa=aa[keep];bb=bb[keep];err=np.linalg.norm(_H(aa,H)-bb,axis=1);span=np.ptp(aa,axis=0);probe=np.array([aa.mean(0),aa.mean(0)+[3,0],aa.mean(0)+[0,3]]);dest=_H(probe,H);det=np.linalg.det(np.array([dest[1]-dest[0],dest[2]-dest[0]]));receipt.update(actual_inlier_source_uv=aa.tolist(),actual_inlier_current_uv=bb.tolist(),source_to_current_navigation_H=H.tolist(),actual_inliers=int(len(aa)),current_reprojection_error_q95=float(np.quantile(err,.95)),source_inlier_extent_xy=span.tolist(),source_orientation_determinant=float(det))
    if det<=0 or span[0]<150 or span[1]<20 or np.quantile(err,.95)>2.5:reason='current_source_object_extent_or_orientation_ambiguous'
    else:
     x0,y0,x1,y1=a['board_instance_roi_xyxy'];tx0=max(x0,float(aa[:,0].min())-20);tx1=min(x1,float(aa[:,0].max())+20);upper=np.asarray([[tx0,a['top_blue_upper_search_y_range'][0]],[tx1,a['top_blue_upper_search_y_range'][0]],[tx0,a['top_blue_upper_search_y_range'][1]],[tx1,a['top_blue_upper_search_y_range'][1]]]);lower=upper.copy();lower[:2,1]=a['contact_yellow_lower_search_y_range'][0];lower[2:,1]=a['contact_yellow_lower_search_y_range'][1];ut=_H(upper,H);lt=_H(lower,H);allp=np.r_[ut,lt];lo=np.maximum([0,0],np.floor(allp.min(0)-12)).astype(int);hi=np.minimum([1920,1080],np.ceil(allp.max(0)+13)).astype(int);top=[max(0,int(np.floor(ut[:,1].min()-12))),min(1079,int(np.ceil(ut[:,1].max()+12)))];bottom=[max(0,int(np.floor(lt[:,1].min()-12))),min(1079,int(np.ceil(lt[:,1].max()+12)))];receipt.update(board_instance_roi_xyxy=[*lo.tolist(),*hi.tolist()],top_blue_upper_search_y_range=top,contact_yellow_lower_search_y_range=bottom,top_blue_upper_visible=True,contact_yellow_lower_visible=True,person_and_overlay_masks_still_required=True,ROI_is_search_domain_only=True,fit_UV_created=False,physical_semantic_authority='current_actual_board_object_RGB_correspondence_to_direct_current_review; source_band_profiles_still_required',source_review_notes='Direct native RGB correspondences only navigate the board object. Current blue upper and yellow lower must independently pass local actual material reading.')
     if hi[0]-lo[0]<120 or hi[1]-lo[1]<40 or hi[1]-lo[1]>800 or top[0]>=bottom[1]:reason='current_object_navigation_footprint_unbounded_or_material_order_ambiguous'
  receipt['status']=reason or'actual_RGB_board_object_correspondence_navigation_ready';decisions.append(receipt);counts.update([receipt['status']])
  if not reason:rows.append(receipt)
  if len(decisions)%20==0:print(json.dumps(dict(frames=len(decisions),ready=len(rows),counts=dict(counts))),flush=True)
 write_json(out/'board_instance_regions.json',dict(schema='rinkfit.joint9.source_correspondence_board_instance_navigation.v1',frames=rows,anchor_review=dict(path=str(ar),sha256=file_sha256(ar)),no_fitted_camera_or_model_used=True,source_UV_created=False));write_jsonl(out/'decisions.jsonl.gz',decisions);result=dict(frames=len(decisions),source_object_navigation_ready_frames=len(rows),counts=dict(counts),fit_UV_created=False,no_source_role_promoted_without_current_material_reading=True);write_json(out/'manifest.json',result);print(json.dumps(result));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--run',required=True);p.add_argument('--anchor-review',required=True);p.add_argument('--out',required=True);a=p.parse_args();transfer(a.project,a.run,a.anchor_review,a.out)
