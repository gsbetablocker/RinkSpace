from source_reader_anchored import *
chosen=[997632,1235200,1357568,1415168,1418240,1424896];allpts=set(chosen+[p-256 for p in chosen]);ims={};receipt=[]
seq=readj(O/'sequential_probe_receipt.json');rec={v['native_pts']:v for v in seq['records']if v.get('actual_adjacent_board_LK')}
with av.open(readj(J/'programme_scope.json')['source']['path'])as c:
 s=c.streams.video[0];c.seek(min(allpts),stream=s,backward=True)
 for f in c.decode(s):
  p=int(f.pts)
  if p>max(allpts):break
  if p in allpts:ims[p]=f.to_ndarray(format='bgr24')
for p in chosen:
 current_anchor=readj(O/'anchored_frames'/(str(p)+'.json'));motion=current_anchor['temporal_source_check'];uv0=np.array(motion.get('source_uv',[]));uv1=np.array(motion.get('current_uv',[]));delta=uv1-uv0;tiles=[];roles={}
 for pp in [p-256,p]:
  im=ims[pp];out=im.copy();ar=readj(O/'anchored_frames'/(str(pp)+'.json'));rr={ro:dict(y=zz['full_native_y'],meta=zz['material_band'])for ro,zz in ar['roles'].items()};qf=O/'qualified_frames'/(str(pp)+'.json');q=readj(qf)if qf.exists()else None;m=mask_for(pp)[0]|current_graphics(im)[0]
  for ro,col in [('board.top',(0,255,0)),('board.contact',(0,60,255))]:
   z=rr[ro]
   points=q['roles'][ro]['trace_uv']if q else[[x,z['y'][x]]for x in range(0,1920,4)if not z['meta']['full_material_occlusion'][x]]
   for x,y in points:
    x=int(x);y=int(round(y))
    if 0<=y<1080 and not m[y,x]:cv2.circle(out,(x,y),1,col,-1)
  if pp==p:
   for u,v in zip(uv0,uv1):
    x,y=np.rint(v).astype(int);old=np.rint(u).astype(int);cv2.arrowedLine(out,tuple(old),tuple([x,y]),(255,255,0),1,tipLength=.15)
  lo=max(0,int(np.percentile(rr['board.top']['y'],5))-25);hi=min(1080,int(np.percentile(rr['board.contact']['y'],95))+30);tile=cv2.resize(out[lo:hi],(1280,320));cv2.putText(tile,str(pp)+' current source'+(' + actual board LK'if pp==p else''),(10,20),0,.6,(255,0,255),2);tiles.append(tile)
 path=O/(str(p)+'_jump_source_before_after.jpg');cv2.imwrite(str(path),np.vstack(tiles));old_anchor=readj(O/'anchored_frames'/(str(p-256)+'.json'));change={}
 for ro in ['board.top','board.contact']:
  aa=np.array(old_anchor['roles'][ro]['full_native_y']);bb=np.array(current_anchor['roles'][ro]['full_native_y']);occ=np.array(old_anchor['roles'][ro]['material_band']['full_material_occlusion'])|np.array(current_anchor['roles'][ro]['material_band']['full_material_occlusion']);dd=abs(bb-aa);change[ro]=dict(median_same_x_change_px=float(np.median(dd[~occ])),q95_same_x_change_px=float(np.percentile(dd[~occ],95)))
 receipt.append(dict(native_pts=p,previous_native_pts=p-256,current_and_previous_BGR_sha256=[hashlib.sha256(ims[k].tobytes()).hexdigest()for k in[p-256,p]],boundary_same_x_change=change,actual_board_texture_LK=motion,median_actual_LK_displacement_px=np.median(delta,axis=0).tolist()if len(delta)else None,anchored_files=[dict(file=str(O/'anchored_frames'/(str(pp)+'.json')),sha256=sha(O/'anchored_frames'/(str(pp)+'.json')))for pp in[p-256,p]],qualified_overlay=all((O/'qualified_frames'/(str(pp)+'.json')).exists()for pp in[p-256,p]),source_image=str(path),non_source_feedback_or_KF_switch_jump_removed=True,affine_rejected_when_inliers_insufficient=motion['image_navigation_H']is None))
write(O/'known_jump_source_review.json',dict(complete=True,source_reader_sha256=sha(O/'source_reader_anchored.py'),full_actual_sequence_probe_frames=seq['tested_frame_reads'],records=receipt,source_image_review_required=True,model_projection_or_fit_used=False))
print([(r['native_pts'],r['boundary_same_x_change'],r['median_actual_LK_displacement_px'])for r in receipt],flush=True)
