exec(open(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_footwear_v1/build.py',encoding='utf8').read().split('def main():')[0])
POSE={json.loads(l)['native_pts']:json.loads(l)for l in gzip.open(O/'pose_selection_v2.jsonl.gz','rt',encoding='utf8')};CLEAN={json.loads(l)['native_pts']:json.loads(l)for l in gzip.open(O/'clean_parent52_v2/index.jsonl.gz','rt',encoding='utf8')}
def main():
 out=O/'manual_arm_foot_source_v2';out.mkdir(exist_ok=True);(out/'delta_masks').mkdir(exist_ok=True);wanted={1187584,1187840,1188096,1188352};frames={}
 with av.open(source['path'])as c:
  st=c.streams.video[0];c.seek(min(wanted),stream=st,backward=True)
  for fr in c.decode(st):
   pts=int(fr.pts)
   if pts>max(wanted):break
   if pts in wanted:frames[pts]=fr.to_ndarray(format='rgb24')
 gray={p:cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)for p,rgb in frames.items()};anchor=1188352;seed=np.array([[620,398],[556,386],[706,402]],'f4').reshape(-1,1,2);tracks={anchor:(seed.reshape(-1,2),np.zeros(3))}
 for order in [sorted(p for p in wanted if anchor<p<1200000),sorted([p for p in wanted if p<anchor],reverse=True)]:
  prev=anchor;q=seed.copy()
  for pts in order:
   f,st,er=cv2.calcOpticalFlowPyrLK(gray[prev],gray[pts],q,None,winSize=(41,41),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01));back,bs,be=cv2.calcOpticalFlowPyrLK(gray[pts],gray[prev],f,None,winSize=(41,41),maxLevel=4);fb=np.linalg.norm(back[:,0]-q[:,0],axis=1);fb[(st[:,0]==0)|(bs[:,0]==0)]=999;tracks[pts]=(f[:,0],fb);q=f;prev=pts
 overrides={1187584:[[768,318],[740,300],[700,270]],1187840:[[774,356],[730,335],[672,291]],1188096:[[755,380],[680,365],[600,341]],1188352:[[620,398],[556,386],[706,402]],1235200:[[811,582],[815,609]]};sam=Sam2SourceInstances(r'D:/CodexData/MotionTrail_Archive_20261019/models/sam2.1-hiera-tiny',device='cuda',reread_rgb=False,contour_stride=16);rows=[];cells=[];page=0
 def save():
  nonlocal cells,page
  if not cells:return
  cells +=[np.full((300,600,3),235,'u1')for _ in range(6-len(cells))];cv2.imwrite(str(out/f'QA_{page:02d}.jpg'),np.vstack([np.hstack(cells[k:k+2])for k in range(0,6,2)]),[cv2.IMWRITE_JPEG_QUALITY,98]);cells=[];page+=1
 with gzip.open(out/'index.jsonl.gz','wt',encoding='utf8')as w:
  for pts,rgb in sorted(frames.items()):
   parent=CLEAN.get(pts,MASK_INDEX[pts]);assert TH(rgb.tobytes()).hexdigest()==parent['source_RGB_sha256'];base=cv2.imread(parent['mask_file'],0)>0;pp=overrides.get(pts);mode='source_visual_review_current_RGB_positive_points';fb=[]
   if pp is None:
    uv,ff=tracks[pts];pp=[];mode='actual_native_RGB_LK_FB_search_navigation_then_current_SAM2_pixels';fb=ff.tolist()
    for i,(u,e)in enumerate(zip(uv,ff)):
     x,y=np.rint(u).astype(int)
     if e>2 or not(4<=x<1916 and 4<=y<1076):continue
     if i!=1 and rgb[y,x].max()>160:continue
     pp.append(u.tolist())
   if not pp:rows.append(dict(native_pts=pts,status='no_source_supported_prompt',source_RGB_sha256=parent['source_RGB_sha256']));continue
   if pts==1235200:box=[790,540,900,670]
   elif pts==1188352:box=[520,355,825,435]
   else:
    kp=POSE[pts]['selected_keypoints_uv'];xy=np.array(pp)
    if kp is not None:
     kk=np.array(kp);poss=[kk[[5,7,9]],kk[[6,8,10]]];part=min(poss,key=lambda a:np.min(np.linalg.norm(a[:,None]-xy[None],axis=2)));xy=np.vstack([xy,part])
    box=[max(0,float(xy[:,0].min()-45)),max(0,float(xy[:,1].min()-35)),min(1920,float(xy[:,0].max()+45)),min(1080,float(xy[:,1].max()+35))]
   arm_boxes={1187584:[665,235,855,390],1187840:[630,260,855,415],1188096:[555,305,855,440],1188352:[520,355,825,435]}
   if pts in arm_boxes:box=arm_boxes[pts]
   bind=dict(source_sha256=source['sha256'],source_pts=pts,time_base='1/12800',native_hw=[1080,1920],shot_id=parent['shot_id'],decoded_native_rgb_sha256=parent['source_RGB_sha256']);prompt=dict(object_id=1,kind='occluder',origin='original_rgb_candidate',source_binding=bind,exclusion_authority=None,points_uv=pp,point_labels=[1]*len(pp),box_xyxy=box);result=sam.read(rgb,bind,prompts=[prompt],reset=True);delta=result['occluder_region_candidate_mask'].detach().cpu().numpy().astype(bool);df=out/'delta_masks'/f'{pts}.png';cv2.imwrite(str(df),delta.astype('u1')*255,[cv2.IMWRITE_PNG_COMPRESSION,3]);r=dict(native_pts=pts,shot_id=parent['shot_id'],source_RGB_sha256=parent['source_RGB_sha256'],source_BGR_sha256=parent['source_BGR_sha256'],delta_mask_file=str(df),delta_mask_sha256=TH(df.read_bytes()).hexdigest(),added_pixels=int((delta&~base).sum()),points_uv=pp,point_RGB=[rgb[int(round(u[1])),int(round(u[0]))].tolist()for u in pp],box_search_only=box,source_navigation_kind=mode,native_RGB_LK_FB_error=fb,point_role='person_foreground_SAM_positive_prompt_not_fitting_observation',source_pixel_mask_is_actual_SAM_not_ROI=True);w.write(json.dumps(r,separators=(',',':'))+'\n');rows.append(r)
   ys,xs=np.nonzero(delta);x0=max(0,int(xs.min())-35);x1=min(1920,int(xs.max())+36);y0=max(0,int(ys.min())-40);y1=min(1080,int(ys.max())+40);im=rgb[:,:,::-1].copy();raw=im[y0:y1,x0:x1].copy();cnt,_=cv2.findContours(base.astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cnt,-1,(255,0,0),1);cnt,_=cv2.findContours(delta.astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cnt,-1,(0,255,0),1);cell=np.full((300,600,3),235,'u1');cell[:275,:300]=cv2.resize(raw,(300,275));cell[:275,300:]=cv2.resize(im[y0:y1,x0:x1],(300,275));cv2.putText(cell,str(pts)+' +'+str(r['added_pixels']),(5,293),0,.6,(0,0,255),1);cells.append(cell)
   if len(cells)==6:save()
 save();(out/'manifest.json').write_text(json.dumps(dict(complete=True,frames=len(rows),source_positive_points_or_actual_LK_FB_used_only_for_search=True,source_masks=SAm if False else'actual SAM2 source pixels',QA_pages=page),indent=2),encoding='utf8')
if __name__=='__main__':main()
