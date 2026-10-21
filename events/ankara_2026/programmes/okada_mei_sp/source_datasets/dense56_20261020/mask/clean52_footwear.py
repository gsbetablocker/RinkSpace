exec(open(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_footwear_v1/build.py',encoding='utf8').read().split('def main():')[0])
POSE={json.loads(l)['native_pts']:json.loads(l)for l in gzip.open(O/'pose_proposals.jsonl.gz','rt',encoding='utf8')};COMP={json.loads(l)['native_pts']:json.loads(l)for l in gzip.open(O/'index.jsonl.gz','rt',encoding='utf8')}
def pose_warm(rgb,mask,p):
 props=[];audit=[];kp=p.get('selected_keypoints_uv');cf=p.get('selected_keypoint_confidence');state='selected_pose_navigation'
 if kp is None:
  persons=p.get('person_proposals',[])
  if not persons:return [],dict(status='no_pose_proposal_current_source_visibility_unknown')
  who=max(persons,key=lambda e:e['box_xyxy'][3]-e['box_xyxy'][1]);kp=who['keypoints_uv'];cf=who['keypoint_confidence'];state='unselected_pose_search_proposal_current_entity_unknown'
 for ki in(15,16):
  if cf[ki]<.5:audit.append(dict(keypoint=ki,status='low_confidence_search_only'));continue
  ankle=np.array(kp[ki]);knee=np.array(kp[ki-2]);v=ankle-knee;v/=max(np.linalg.norm(v),1);radius=125;x0=max(0,int(ankle[0])-radius);x1=min(1920,int(ankle[0])+radius);y0=max(0,int(ankle[1])-radius);y1=min(1080,int(ankle[1])+radius)
  rr=rgb[y0:y1,x0:x1].astype('i2');m=mask[y0:y1,x0:x1];warm=(rr[:,:,0]-rr[:,:,2]>5)&(abs(rr[:,:,0]-rr[:,:,1])<12)&(rr[:,:,1]>70)&(rr[:,:,2]<235)&~m;warm=cv2.morphologyEx(warm.astype('u1'),cv2.MORPH_CLOSE,np.ones((3,3),'u1'));n,lab,stats,cen=cv2.connectedComponentsWithStats(warm,8);count=0
  for k,(x,y,w,h,area)in enumerate(stats[1:],1):
   if area<150 or w>165 or h>190 or w<6 or h<8 or w/h>3.2 or x==0 or y==0 or x+w>=x1-x0 or y+h>=y1-y0:continue
   cc=lab==k;yy,xx=np.nonzero(cc);uv=np.c_[xx+x0,yy+y0];dist=np.linalg.norm(uv-ankle,axis=1);along=(uv-ankle)@v
   if dist.min()>65 or np.median(along)<-15 or np.median(along)>110:continue
   core=cv2.distanceTransform(cc.astype('u1'),cv2.DIST_L2,5);core[dist.reshape(-1).shape[0]:]if False else None;zy,zx=np.unravel_index(np.argmax(core),core.shape);point=[x0+int(zx),y0+int(zy)]
   if core.max()<3 or any(np.linalg.norm(np.array(point)-q['points_uv'][0])<25 for q in props):continue
   props.append(dict(points_uv=[point],point_labels=[1],box_xyxy=[max(0,x0+int(x)-14),max(0,y0+int(y)-14),min(1920,x0+int(x+w)+14),min(1080,y0+int(y+h)+25)],actual_RGB_at_positive_point=rgb[point[1],point[0]].tolist(),component_bbox=[x0+int(x),y0+int(y),x0+int(x+w),y0+int(y+h)],current_warm_component_native_area=int(area),ankle_pose_proposal_uv=ankle.tolist(),pose_keypoint_confidence=float(cf[ki]),pose_navigation_state=state,prompt_role='current_RGB_warm_white_component_near_image_pose_ankle_search_only'));count+=1
  audit.append(dict(keypoint=ki,status='current_native_RGB_component_candidates'if count else'no_current_RGB_unmasked_warm_white_component',candidates=count,ankle_search_uv=ankle.tolist(),ankle_confidence=float(cf[ki])))
 return props,dict(status=state,keypoints=audit)

from scipy.ndimage import distance_transform_edt
CLEAN={r['native_pts']:r for r in [json.loads(l)for l in gzip.open(O/'clean_parent52_v2/index_candidate.jsonl.gz','rt',encoding='utf8')]}
def main():
 out=O/'clean52_footwear_delta';out.mkdir(exist_ok=True);(out/'delta_masks').mkdir(exist_ok=True);sam=Sam2SourceInstances(r'D:/CodexData/MotionTrail_Archive_20261019/models/sam2.1-hiera-tiny',device='cuda',reread_rgb=False,contour_stride=16);rows=[];cells=[];page=0
 def save():
  nonlocal cells,page
  if not cells:return
  cells +=[np.full((250,320,3),235,'u1')for _ in range(16-len(cells))];cv2.imwrite(str(out/f'QA_{page:02d}.jpg'),np.vstack([np.hstack(cells[k:k+4])for k in range(0,16,4)]),[cv2.IMWRITE_JPEG_QUALITY,98]);cells=[];page+=1
 with av.open(source['path'])as c,gzip.open(out/'index.jsonl.gz','wt',encoding='utf8')as w:
  st=c.streams.video[0];c.seek(min(CLEAN),stream=st,backward=True)
  for fr in c.decode(st):
   pts=int(fr.pts)
   if pts>max(CLEAN):break
   if pts not in CLEAN:continue
   cr=CLEAN[pts];rgb=fr.to_ndarray(format='rgb24');assert TH(rgb.tobytes()).hexdigest()==cr['source_RGB_sha256'];mf=cr['current_SAM']['file'];assert TH(Path(mf).read_bytes()).hexdigest()==cr['current_SAM']['sha256'];mask=cv2.imread(mf,0)>0;props=propose(rgb,mask,pts);more,_=pose_warm(rgb,mask,POSE[pts])
   for p in more:
    if any(np.linalg.norm(np.array(p['points_uv'][0])-q['points_uv'][0])<25 for q in props):continue
    bx0,by0,bx1,by1=p['component_bbox'];x0=max(0,bx0-45);x1=min(1920,bx1+45);y0=max(0,by0-45);y1=min(1080,by1+45);rr=rgb[y0:y1,x0:x1].astype('i2');mm=mask[y0:y1,x0:x1];near,inds=distance_transform_edt(~mm,return_indices=True);nearest=rr[inds[0],inds[1]];ankle=np.array(p['ankle_pose_proposal_uv']);dist=np.hypot(inds[1]+x0-ankle[0],inds[0]+y0-ankle[1]);warm=(rr[:,:,0]-rr[:,:,2]>5)&(abs(rr[:,:,0]-rr[:,:,1])<12)&~mm;warm[:by0-y0]=False;warm[by1-y0:]=False;warm[:,:bx0-x0]=False;warm[:,bx1-x0:]=False;attach=warm&(near<20)&(nearest[:,:,0]-nearest[:,:,2]>15)&(dist<65)
    if attach.sum()>=12:props.append(p)
   r=dict(native_pts=pts,shot_id=cr['shot_id'],source_RGB_sha256=cr['source_RGB_sha256'],source_BGR_sha256=cr['source_BGR_sha256'],base_person_mask_file=mf,base_person_mask_sha256=cr['current_SAM']['sha256'],prompts=props,added_pixels=0,delta_mask_file=None)
   if props:
    bind=dict(source_sha256=source['sha256'],source_pts=pts,time_base='1/12800',native_hw=[1080,1920],shot_id=cr['shot_id'],decoded_native_rgb_sha256=cr['source_RGB_sha256']);prompts=[dict(object_id=i+1,kind='occluder',origin='original_rgb_candidate',source_binding=bind,exclusion_authority=None,**p)for i,p in enumerate(props)];result=sam.read(rgb,bind,prompts=prompts,reset=True);delta=result['occluder_region_candidate_mask'].detach().cpu().numpy().astype(bool);added=delta&~mask;area=int(added.sum());df=out/'delta_masks'/f'{pts}.png';cv2.imwrite(str(df),delta.astype('u1')*255,[cv2.IMWRITE_PNG_COMPRESSION,3]);r.update(delta_mask_file=str(df),delta_mask_sha256=TH(df.read_bytes()).hexdigest(),added_pixels=area)
    if area:
     ys,xs=np.nonzero(added);x0=max(0,int(xs.min())-35);x1=min(1920,int(xs.max())+36);y0=max(0,int(ys.min())-75);y1=min(1080,int(ys.max())+45);im=rgb[:,:,::-1].copy();raw=im[y0:y1,x0:x1].copy();cnt,_=cv2.findContours(mask.astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cnt,-1,(255,0,0),1);cnt,_=cv2.findContours(delta.astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cnt,-1,(0,255,0),1);cell=np.full((250,320,3),235,'u1');cell[:230,:160]=cv2.resize(raw,(160,230));cell[:230,160:]=cv2.resize(im[y0:y1,x0:x1],(160,230));cv2.putText(cell,str(pts)+' +'+str(area),(5,245),0,.5,(0,0,255),1);cells.append(cell)
     if len(cells)==16:save()
   rows.append(r);w.write(json.dumps(r,separators=(',',':'))+'\n')
 save();(out/'manifest.json').write_text(json.dumps(dict(complete=True,native_frames_scanned=len(rows),new_SAM_frames=sum(bool(r['added_pixels'])for r in rows),source_person_base='clean_parent52_v2 current_SAM candidate raw',all_original_data_preserved=True,QA_pages=page),indent=2),encoding='utf8');print(json.dumps(dict(frames=len(rows),changed=sum(bool(r['added_pixels'])for r in rows),QA_pages=page)),flush=True)
if __name__=='__main__':main()
