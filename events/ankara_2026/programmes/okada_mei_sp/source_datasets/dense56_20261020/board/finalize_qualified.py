import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import av,cv2,numpy as np,json,gzip,hashlib,time,copy
from pathlib import Path
from scipy.ndimage import median_filter
from source_reader_anchored import O,M,R,J,P,PTS,readj,sha,write,sid,current_graphics
cv2.setNumThreads(1)
def spans(v):
 d=np.diff(np.r_[False,v,False].astype('i1'));return [(int(a),int(b-1))for a,b in zip(np.flatnonzero(d==1),np.flatnonzero(d==-1))]
while not(M/'index.jsonl.gz').exists():time.sleep(.2)
mi={}
with gzip.open(M/'index.jsonl.gz','rt',encoding='utf8')as f:
 for l in f:
  q=json.loads(l);mi[int(q['native_pts'])]=q
binding=readj(O/'final_mask_binding.json');final_index=Path(binding['file']);assert sha(final_index)==binding['sha256'];assert readj(binding['manifest_file'])['accepted_for_final_source_qualification'];fmi={}
with gzip.open(final_index,'rt',encoding='utf8')as f:
 for line in f:
  row=json.loads(line);fmi[int(row['native_pts'])]=row
assert set(fmi)==set(PTS)
qualification_lock=O/'qualification_active.lock';qualification_lock_fd=os.open(str(qualification_lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(qualification_lock_fd,str(os.getpid()).encode())
(O/'qualified_frames').mkdir(exist_ok=True)
t=time.perf_counter();counts={};bykind={};byrole={};frames=[];changedFrames=0;changesN=0;actorN=0;badOrder=0;temporal=[];prevY=None;weakN=0
frows=gzip.open(O/'qualified_source_rows.stream.jsonl.gz','wt',encoding='utf8',compresslevel=6);fcov=gzip.open(O/'qualified_coverage.stream.jsonl.gz','wt',encoding='utf8',compresslevel=6);flocal=gzip.open(O/'local_material_requalification.stream.jsonl.gz','wt',encoding='utf8');factor=gzip.open(O/'actor_overlap_exclusions.stream.jsonl.gz','wt',encoding='utf8')
scope=readj(J/'programme_scope.json');xs=np.arange(1920)
with av.open(scope['source']['path'])as c:
 s=c.streams.video[0];c.seek(PTS[0],stream=s,backward=True)
 for ff in c.decode(s):
  p=int(ff.pts)
  if p<PTS[0]:continue
  if p>PTS[-1]:break
  src=O/'anchored_frames'/(str(p)+'.json')
  while not src.exists():time.sleep(.15)
  raw=readj(src);im=ff.to_ndarray(format='bgr24');bh=hashlib.sha256(im.tobytes()).hexdigest()
  assert bh==raw['decoded_native_BGR_sha256']==mi[p]['source_BGR_sha256']
  mp=Path(fmi[p]['mask_file']);mh=sha(mp);assert mh==fmi[p]['mask_sha256'];person_mask=cv2.imread(str(mp),0)>0;unknown_ref=fmi[p].get('source_unknown_exclusion_mask_ref');unknown=np.zeros_like(person_mask)
  if unknown_ref:
   up=Path(unknown_ref['file']);assert sha(up)==unknown_ref['sha256'];unknown=cv2.imread(str(up),0)>0
  mask=cv2.dilate((person_mask|unknown).astype('u1'),np.ones((17,17),'u1'))>0
  if fmi[p].get('source_BGR_sha256'):assert fmi[p]['source_BGR_sha256']==bh
  graphics,graphics_receipt=current_graphics(im)
  raw.setdefault('source_reader_person_pixel_mask_path',raw['mask_path']);raw.setdefault('source_reader_person_pixel_mask_sha256',sha(raw['source_reader_person_pixel_mask_path']));raw['mask_path']=str(mp);raw['mask_sha256']=mh;raw['final_eligibility_mask_index']=str(final_index);raw['final_eligibility_mask_index_sha256']=binding['sha256'];raw['final_eligibility_mask_only_requalification']=True;raw['source_unknown_exclusion_mask_ref']=unknown_ref;raw['unknown_exclusion_ref']=unknown_ref;raw['source_visibility_guard_px']=8;raw['person_mask_includes_unknown_exclusion']=False
  for ro,rr0 in raw['roles'].items():
   yi0=np.clip(np.rint(rr0['full_native_y']).astype(int),0,1079);rr0['material_band'].setdefault('source_reader_occluded_x_spans',spans(np.array(rr0['material_band']['full_material_occlusion'],bool)));rr0['material_band']['full_material_occlusion']=(mask[yi0,xs]|graphics[yi0,xs]).tolist()
   if ro=='board.top':
    ly=np.rint(rr0['material_band']['full_blue_cap_lower_y']).astype(int);ld=np.array(rr0['material_band']['full_blue_cap_lower_direct'],bool)
    for d in range(-5,6):ld&=~mask[np.clip(ly+d,0,1079),xs]
    rr0['material_band']['full_blue_cap_lower_direct']=ld.tolist();rr0['material_band']['full_blue_cap_lower_localization_radius']=[v if ld[k]else None for k,v in enumerate(rr0['material_band']['full_blue_cap_lower_localization_radius'])]
  write(src,raw);ah=sha(src)
  near=cv2.dilate(mask.astype('u1'),np.ones((21,21),'u1'))>0;bm=np.zeros_like(mask)
  for box in mi[p].get('current_boxes',[]):
   a,b,d,e=np.rint(box['box_xyxy']).astype(int);bm[max(0,b):min(1080,e),max(0,a):min(1920,d)]=True
  rgb=im[:,:,::-1].astype('f4');ch=rgb[:,:,2]-(rgb[:,:,0]+rgb[:,:,1])/2;sm=cv2.GaussianBlur(ch,(3,3),.7);gr=np.gradient(sm,axis=0);roles={};rows=[];proof=[];actors=[];currentYs={};changed=0
  for role,rr in raw['roles'].items():
   y=np.array(rr['full_native_y'],float);oldy=y.copy();direct=np.array(rr['full_direct'],bool);gradient=np.array(rr['full_gradient'],float);before=np.array(rr['full_chroma_before'],float);after=np.array(rr['full_chroma_after'],float);med=median_filter(y,129,mode='nearest');localbad=np.zeros(1920,bool);kinds={};localexc=np.zeros(1920,bool)
   # Arrays are the current native pixel reader's frozen UV; no second curve-based relocation.
   for item in rr['material_band'].get('rejected_local_material_candidates_and_actual_current_reread',[]):proof.append(dict(role=role,**item))
   yi=np.clip(np.rint(y).astype(int),0,1079);occ=mask[yi,xs]|graphics[yi,xs]|localexc|np.array(rr['material_band']['full_material_occlusion'],bool);inframe=(y>=2)&(y<1078)
   # Retain final-parent actor checks: bbox alone never removes a source point.
   refy=np.clip(np.rint(med).astype(int),0,1079);offs=range(-7,-1)if role=='board.contact'else range(1,8);vv=np.stack([ch[np.clip(refy+d,0,1079),xs]for d in offs]);strength=np.maximum(0,-np.min(vv,axis=0))if role=='board.contact'else np.maximum(0,np.max(vv,axis=0));ref=median_filter(strength,129,mode='nearest');suspect=inframe&~occ&near[yi,xs]&bm[yi,xs]&(ref>8)&(strength<.6*ref)
   for x in np.flatnonzero(suspect):actors.append(dict(role=role,original_source_row_id=f'dense56_board/{p}/{role}/{int(x)}',original_uv=[int(x),float(y[x])],reason='actual_current_material_absent_relative_to_adjacent_same_static_surface_inside_person_box_near_SAM; excluded_not_moved',current_core_strength=float(strength[x]),adjacent_same_material_strength=float(ref[x])))
   occ|=suspect;visible=inframe&~occ;ss=[];uv=[]
   for a,b in spans(visible):
    seg=f'{sid(p)}/{p}/{role}/{a}-{b}';dx=np.arange(a,b+1,4,dtype=int)
    if not len(dx)or dx[-1]!=b:dx=np.r_[dx,b]
    sp=dict(segment_id=seg,x_range=[a,b],visibility='visible',source_kind='current_source_material_band_connected_span',direct_columns=int(direct[a:b+1].sum()),blurred_columns=int((~direct[a:b+1]).sum()),candidate_points=len(dx),endpoints_uv=[[a,float(y[a])],[b,float(y[b])]])
    ss.append(sp)
    for x0 in dx:
     x=int(x0);v=float(y[x]);di=bool(direct[x]);kind=kinds.get(x,'current_native_material_transition'if di else'blurred_current_material_plus_same_frame_span_continuity');slope=float((y[min(1919,x+1)]-y[max(0,x-1)])/max(1,min(1919,x+1)-max(0,x-1)));normal=np.array([-slope,1.]);normal/=np.linalg.norm(normal);width=max(1. if di else 6.,6/max(float(gradient[x]),.5));support=[(np.array([x,v])-normal*width).tolist(),(np.array([x,v])+normal*width).tolist()]
     row=dict(shot_id=sid(p),native_pts=p,time_base='1/12800',source_video_sha256=scope['source']['sha256'],decoded_native_BGR_sha256=bh,feature_id=role,role=role,segment_id=seg,source_row_id=f'dense56_board/{p}/{role}/{x}',uv=[x,v],source_kind=kind,visibility='visible',occlusion=False,pixel_support=dict(chroma_gradient=float(gradient[x]),chroma_before=float(before[x]),chroma_after=float(after[x]),current_pixel_profile_reread=True,uncertainty_kind='photometric_localization_support_not_calibrated_confidence',position_interval_endpoints_uv=support,interval_is_not_sigma=True,weak_local_query_interval_not_independent_source=not di),pixels_measured_in_this_frame=di,source_evidence_measured_in_this_frame=di,current_native_pixel_profile_queried=True,samples_independent=False,source_navigation_native_pts=p,model_or_world_projection_used=False,current_person_pixel_mask_sha256=mh,mask_sha256=mh,source_visibility_guard_px=8,anchored_file=str(src),anchored_file_sha256=ah)
     if x in kinds:row['original_rejected_local_candidate_uv']=[x,float(oldy[x])]
     rows.append(row);uv.append([x,v]);bykind[kind]=bykind.get(kind,0)+1;byrole[role]=byrole.get(role,0)+1
    weakN+=int((~direct[a:b+1]).sum())
   roles[role]=dict(trace_uv=uv,spans=ss,visible_columns=int(visible.sum()),occluded_columns=int(occ.sum()),direct_columns=int((direct&visible).sum()),blurred_columns=int((~direct&visible).sum()),excluded_x_spans=[[a,b]for a,b in spans(~visible)],actor_overlap_x_spans=[[a,b]for a,b in spans(suspect|localexc)],local_pixel_return_requalified_x_spans=[[a,b]for a,b in spans(localbad&~localexc)],all_visible_connected_spans_have_endpoints=True,blurred_visible_not_dropped=True)
   currentYs[role]=y
  order=currentYs['board.top']>=currentYs['board.contact'];badOrder+=int(order.any())
  if prevY is not None and sid(p)==prevY[0]:
   jump=max(float(np.median(abs(currentYs[ro]-prevY[1][ro])))for ro in currentYs)
   if jump>15:temporal.append(dict(native_pts=p,median_jump_px=jump))
  prevY=(sid(p),currentYs)
  q=dict(schema='rinkfit.dense_native_board_source_frame.v1',native_pts=p,shot_id=sid(p),rig_id=raw['rig_id'],time_base='1/12800',decoded_native_BGR_sha256=bh,source_video_sha256=scope['source']['sha256'],roles=roles,rows=rows,mask_path=str(mp),mask_sha256=mh,mask_kind='final_actual_current_native_person_and_footwear_pixel_mask',source_unknown_exclusion_mask_ref=unknown_ref,unknown_exclusion_ref=unknown_ref,source_visibility_guard_px=8,unknown_exclusion_is_not_person_mask=True,anchored_file=str(src),anchored_file_sha256=ah,graphics_receipt=graphics_receipt,qualified_source_generation='s05_s06_current_complete_material_source_v5',frame_point_budget_applied=False,dense_candidates_not_independent_fit_samples=True,source_original_UV_modified=False,local_material_requalification=dict(actual_current_RGB_reread=True,original_anchored_source_UV_preserved=True,changes=proof,model_residual_used=False),local_actor_overlap_exclusions=actors)
  dst=O/'qualified_frames'/(str(p)+'.json');write(dst,q)
  for row in rows:frows.write(json.dumps(row,separators=(',',':'))+'\n')
  fcov.write(json.dumps(dict(native_pts=p,shot_id=sid(p),rows=len(rows),roles=roles,decoded_native_BGR_sha256=bh,anchored_file=str(src),anchored_file_sha256=ah,qualified_file=str(dst),qualified_file_sha256=sha(dst),mask_path=str(mp),mask_sha256=mh,source_unknown_exclusion_mask_ref=unknown_ref,unknown_exclusion_ref=unknown_ref,source_visibility_guard_px=8),separators=(',',':'))+'\n')
  if proof:flocal.write(json.dumps(dict(native_pts=p,shot_id=sid(p),changes=proof,anchored_file_sha256=ah),separators=(',',':'))+'\n');changedFrames+=1
  if actors:factor.write(json.dumps(dict(native_pts=p,shot_id=sid(p),excluded=actors),separators=(',',':'))+'\n')
  changesN+=len(proof);actorN+=len(actors);frames.append(p);counts[sid(p)]=counts.get(sid(p),0)+len(rows)
  if len(frames)%25==0 or len(frames)==1:
   for f in [frows,fcov,flocal,factor]:f.flush()
   pr=dict(qualified_frames=len(frames),total=1862,last_native_pts=p,source_candidate_rows_by_shot=counts,local_frames_reread=changedFrames,elapsed_seconds=time.perf_counter()-t);write(O/'qualified_progress.json',pr);print(json.dumps(pr),flush=True)
assert frames==PTS
for f in [frows,fcov,flocal,factor]:f.close()
for n in ['qualified_source_rows','qualified_coverage','local_material_requalification','actor_overlap_exclusions']:(O/(n+'.stream.jsonl.gz')).replace(O/(n+'.jsonl.gz'))
review=dict(schema='rinkfit.full_native_board_material_source_review.v1',complete=False,native_frames=len(frames),frame_counts={'s05':1376,'s06':486},correct_original_keyframes=52,source_original_UV_files_verified_unchanged=True,source_candidates=sum(counts.values()),counts_by_source_kind=bykind,counts_by_role=byrole,counts_by_shot=counts,initial_global_order_and_temporal_checks=dict(bad_top_contact_order=badOrder,median_temporal_jump_over15px=temporal),local_material_requalification=dict(complete=True,current_native_frames_checked=1862,frames_reread_for_local_wrong_return=changedFrames,local_candidates_reviewed=changesN,original_anchored_UV_arrays_unchanged=True,mask_eligibility_metadata_bound_before_release=True,recipe_sha256=sha(__file__)),actor_overlap=dict(complete=True,native_frames_source_checked=1862,excluded_source_columns=actorN,original_UV_retained_in_receipt=True,root_masks_unchanged=True),source_noise_W_or_independence_issued=False,model_or_world_projection_used_for_UV=False,raw_navigation_source_arrays_immutable=True,quality_scope='source UV material membership and visible span coverage for raw-point overlay; not physical-model fit acceptance',visual_self_review_pending=True)
write(O/'source_quality_review.json',review)
manifest=dict(schema='rinkfit.dense_native_board_source_generation.v1',complete=True,native_frames=1862,shots={'s05':1376,'s06':486},source_candidate_rows_by_shot=counts,files={n:dict(file=str(O/n),sha256=sha(O/n))for n in ['qualified_source_rows.jsonl.gz','qualified_coverage.jsonl.gz','source_reader_anchored.py','finalize_qualified.py','source_anchor_recipe.json','local_material_requalification.jsonl.gz','actor_overlap_exclusions.jsonl.gz','source_quality_review.json']},authority=dict(file=str(P/'events/ankara_2026/model_constraints/user_board_edge_semantics_20261010.json'),sha256=sha(P/'events/ankara_2026/model_constraints/user_board_edge_semantics_20261010.json')),definition={'board.top':'upper edge of blue rim at top of vertical board','board.contact':'lower edge of yellow kickplate at actual ice interface'},source_clock='1/12800',source_fps=50,source_uv_from_model=False,source_visible_blurred_spans_preserved=True,person_bbox_used_as_final_exclusion=False,full_native_mask_refs_per_frame=True,candidate_density_not_independence=True,root_must_apply_combined_500_point_per_frame_budget=True,source_quality_status='pending_complete_scope_source_self_review',quality_accepted_for_raw_point_overlay=False,physical_model_quality_accepted=False,all1862_anchored_file_sha256_bound=True,original_source_UV_or_historical_packet_modified=False,PTS_linear_interpolation_used=False)
write(O/'qualified_manifest.json',manifest);print(json.dumps(manifest),flush=True)

os.close(qualification_lock_fd);qualification_lock.unlink()
