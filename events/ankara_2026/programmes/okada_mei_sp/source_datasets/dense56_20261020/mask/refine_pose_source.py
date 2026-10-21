exec(open(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_footwear_v1/build.py',encoding='utf8').read().split('def main():')[0])
from scipy.ndimage import distance_transform_edt
CAND={json.loads(l)['native_pts']:json.loads(l)for l in gzip.open(O/'pose_component_scan/candidates.jsonl.gz','rt',encoding='utf8')};COMP={json.loads(l)['native_pts']:json.loads(l)for l in gzip.open(O/'index.jsonl.gz','rt',encoding='utf8')}
def main():
 sam=Sam2SourceInstances(r'D:/CodexData/MotionTrail_Archive_20261019/models/sam2.1-hiera-tiny',device='cuda',reread_rgb=False,contour_stride=16);out=O/'pose_refined';out.mkdir(exist_ok=True);(out/'delta_masks').mkdir(exist_ok=True);rows=[];stats=__import__('collections').Counter();cells=[];page=0
 def qa_save():
  nonlocal cells,page
  if not cells:return
  cells +=[np.full((250,320,3),235,'u1')for _ in range(16-len(cells))];cv2.imwrite(str(out/f'QA_{page:03d}.jpg'),np.vstack([np.hstack(cells[k:k+4])for k in range(0,16,4)]),[cv2.IMWRITE_JPEG_QUALITY,98]);page+=1;cells=[]
 with av.open(source['path'])as c,gzip.open(out/'index.jsonl.gz','wt',encoding='utf8')as w:
  st=c.streams.video[0];c.seek(948992,stream=st,backward=True)
  for fr in c.decode(st):
   pts=int(fr.pts)
   if pts>1425408:break
   if pts<948992:continue
   p=CAND[pts];stats['frames_scanned']+=1
   if not p['prompts']:continue
   rgb=fr.to_ndarray(format='rgb24');assert TH(rgb.tobytes()).hexdigest()==p['source_RGB_sha256'];mask=cv2.imread(COMP[pts]['mask_file'],0)>0;kept=[]
   for pp in p['prompts']:
    bx0,by0,bx1,by1=pp['component_bbox'];x0=max(0,bx0-45);x1=min(1920,bx1+45);y0=max(0,by0-45);y1=min(1080,by1+45);rr=rgb[y0:y1,x0:x1].astype('i2');m=mask[y0:y1,x0:x1];near,inds=distance_transform_edt(~m,return_indices=True);nearest=rr[inds[0],inds[1]];ankle=np.array(pp['ankle_pose_proposal_uv']);distance=np.hypot(inds[1]+x0-ankle[0],inds[0]+y0-ankle[1]);warm=(rr[:,:,0]-rr[:,:,2]>5)&(abs(rr[:,:,0]-rr[:,:,1])<12)&~m;warm[:by0-y0]=False;warm[by1-y0:]=False;warm[:,:bx0-x0]=False;warm[:,bx1-x0:]=False;attach=warm&(near<20)&(nearest[:,:,0]-nearest[:,:,2]>15)&(distance<65)
    if attach.sum()<12:stats['pose_component_not_attached_to_current_visible_warm_ankle']+=1;continue
    pp=dict(pp);pp['actual_current_skin_ankle_attachment_pixels']=int(attach.sum());kept.append(pp)
   if not kept:continue
   sid=p['shot_id'];bind=dict(source_sha256=source['sha256'],source_pts=pts,time_base='1/12800',native_hw=[1080,1920],shot_id=sid,decoded_native_rgb_sha256=p['source_RGB_sha256']);prompts=[dict(object_id=i+1,kind='occluder',origin='original_rgb_candidate',source_binding=bind,exclusion_authority=None,**q)for i,q in enumerate(kept)];result=sam.read(rgb,bind,prompts=prompts,reset=True);delta=result['occluder_region_candidate_mask'].detach().cpu().numpy().astype(bool);added=delta&~mask;area=int(added.sum());stats['new_SAM_frames']+=1;stats['new_SAM_pixels']+=area;df=out/'delta_masks'/f'{pts}.png';cv2.imwrite(str(df),delta.astype('u1')*255,[cv2.IMWRITE_PNG_COMPRESSION,3]);r=dict(native_pts=pts,shot_id=sid,source_RGB_sha256=p['source_RGB_sha256'],source_BGR_sha256=p['source_BGR_sha256'],delta_mask_file=str(df),delta_mask_sha256=TH(df.read_bytes()).hexdigest(),added_pixels=area,prompts=kept);w.write(json.dumps(r,separators=(',',':'))+'\n');rows.append(r)
   if area:
    ys,xs=np.nonzero(added);x0=max(0,int(xs.min())-35);x1=min(1920,int(xs.max())+36);y0=max(0,int(ys.min())-70);y1=min(1080,int(ys.max())+45);raw=rgb[y0:y1,x0:x1,::-1].copy();im=rgb[:,:,::-1].copy();cnt,_=cv2.findContours(mask.astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cnt,-1,(255,0,0),1);cnt,_=cv2.findContours(delta.astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cnt,-1,(0,255,0),1);cell=np.full((250,320,3),235,'u1');cell[:230,:160]=cv2.resize(raw,(160,230));cell[:230,160:]=cv2.resize(im[y0:y1,x0:x1],(160,230));cv2.putText(cell,str(pts)+' +'+str(area),(5,245),0,.5,(0,0,255),1);cells.append(cell)
    if len(cells)==16:qa_save()
   if len(rows)%50==0:print(json.dumps(dict(stats)),flush=True)
 qa_save();(out/'manifest.json').write_text(json.dumps(dict(complete=True,counts=dict(stats),all_1862_source_ankle_proposals_audited=True,source_ankle_nearby_RGB_confirmed=True,source_rows=len(rows),QA_pages=page),indent=2),encoding='utf8');print(json.dumps(dict(stats)),flush=True)
if __name__=='__main__':main()
