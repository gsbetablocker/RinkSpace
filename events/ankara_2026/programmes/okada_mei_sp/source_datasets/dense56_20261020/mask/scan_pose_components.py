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

def main():
 rows=[];stats=__import__('collections').Counter();out=O/'pose_component_scan';out.mkdir(exist_ok=True)
 with av.open(source['path'])as c,gzip.open(out/'candidates.jsonl.gz','wt',encoding='utf8')as w:
  st=c.streams.video[0];c.seek(948992,stream=st,backward=True)
  for fr in c.decode(st):
   pts=int(fr.pts)
   if pts>1425408:break
   if pts<948992:continue
   rgb=fr.to_ndarray(format='rgb24');assert TH(rgb.tobytes()).hexdigest()==POSE[pts]['source_RGB_sha256'];mask=cv2.imread(COMP[pts]['mask_file'],0)>0;props,audit=pose_warm(rgb,mask,POSE[pts]);r=dict(native_pts=pts,shot_id=POSE[pts]['shot_id'],source_RGB_sha256=POSE[pts]['source_RGB_sha256'],source_BGR_sha256=POSE[pts]['source_BGR_sha256'],prompts=props,current_source_audit=audit);w.write(json.dumps(r,separators=(',',':'))+'\n');rows.append(r);stats['frames_scanned']+=1;stats['frames_with_new_RGB_components']+=bool(props);stats['components']+=len(props)
   if len(rows)%256==0:print(json.dumps(dict(stats)),flush=True)
 assert len(rows)==1862;(out/'manifest.json').write_text(json.dumps(dict(complete=True,counts=dict(stats),all_native_RGB_hash_matches=True,pose_proposals_are_search_only=True),indent=2),encoding='utf8');print(json.dumps(dict(stats)),flush=True)
if __name__=='__main__':main()
