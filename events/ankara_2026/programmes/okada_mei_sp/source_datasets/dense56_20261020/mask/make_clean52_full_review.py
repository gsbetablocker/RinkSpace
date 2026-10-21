exec(open(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_footwear_v1/build.py',encoding='utf8').read().split('def main():')[0])
poses={r['native_pts']:r for r in [json.loads(l)for l in gzip.open(O/'pose_proposals.jsonl.gz','rt',encoding='utf8')]};clean={r['native_pts']:r for r in [json.loads(l)for l in gzip.open(O/'clean_parent52_v2/index.jsonl.gz','rt',encoding='utf8')]};wanted=set(json.load(open(O/'clean_parent52_v2/source_visual_review.json',encoding='utf8'))['known_white_shoe_review']);cells=[];meta=[];page=0
with av.open(source['path'])as c:
 st=c.streams.video[0];c.seek(min(wanted),stream=st,backward=True)
 for fr in c.decode(st):
  pts=int(fr.pts)
  if pts>max(wanted):break
  if pts not in wanted:continue
  rgb=fr.to_ndarray(format='rgb24');p=poses[pts];k=p['selected_keypoints_uv']
  if k is None:k=max(p['person_proposals'],key=lambda r:r['box_xyxy'][3]-r['box_xyxy'][1])['keypoints_uv']
  feet=np.array(k)[15:17];x0=max(0,int(feet[:,0].min())-130);x1=min(1920,int(feet[:,0].max())+131);y0=max(0,int(feet[:,1].min())-100);y1=min(1080,int(feet[:,1].max())+161);im=rgb[:,:,::-1].copy();mask=cv2.imread(clean[pts]['mask_file'],0);cnt,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cnt,-1,(255,0,0),1)
  crop=cv2.resize(im,(960,540));cv2.putText(crop,str(pts),(8,525),0,.7,(0,0,255),2);cells.append(crop);meta.append(dict(native_pts=pts))
  if len(cells)==4:cv2.imwrite(str(O/'clean52_footwear_delta'/f'full_native_{page:02d}.jpg'),np.vstack([np.hstack(cells[:2]),np.hstack(cells[2:])]),[cv2.IMWRITE_JPEG_QUALITY,98]);page+=1;cells=[]
if cells:
 cells +=[np.full((540,960,3),235,'u1')for _ in range(4-len(cells))];cv2.imwrite(str(O/'clean52_footwear_delta'/f'full_native_{page:02d}.jpg'),np.vstack([np.hstack(cells[:2]),np.hstack(cells[2:])]),[cv2.IMWRITE_JPEG_QUALITY,98])
(O/'clean52_footwear_delta/full_native_index.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
