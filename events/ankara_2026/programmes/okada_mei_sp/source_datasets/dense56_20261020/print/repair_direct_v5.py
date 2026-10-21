import os,sys
sys.dont_write_bytecode=True
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import av,cv2,numpy as np,json,gzip,hashlib,time,base64,importlib.util
from pathlib import Path
from scipy.spatial import cKDTree
O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_print_v1');R=O.parent
s=importlib.util.spec_from_file_location('v5',O/'patch_v5_native.py');v=importlib.util.module_from_spec(s);s.loader.exec_module(v);h=v.h;op=v.op;cv2.setNumThreads(2)
F=O/'source_direct_v5_frames';M=O/'source_direct_v5_motion';T=O/'semantic_direct_v5';D=O/'body_direct_delta_v5'
for d in(F,M,T):d.mkdir(exist_ok=True)
def sha(b):return hashlib.sha256(b).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,j):Path(p).write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf8')
def ref(p):return dict(file=str(p),sha256=sha(Path(p).read_bytes()))
def bodymask(pts):
 p=D/f'{pts}.png'
 if not p.exists():p=Path(load(load(O/'source_v5_frames'/f'{pts}.json')['body_material_ref']['file'])['body_material_mask_ref']['file'])
 return cv2.imread(str(p),0)
def same(a,b):return(a<=1300992)==(b<=1300992)
def semantic_current(rgb,valid,pts):
 # Actual review of these changed current source patterns is stored before final qualification.
 if pts not in REVIEWED_EXTRA:return v.semantics.current_static_print_semantics(rgb,valid,pts)
 color=((rgb[:,:,2].astype(float)-rgb[:,:,0].astype(float)>12)|(rgb[:,:,0].astype(float)-rgb[:,:,2].astype(float)>12))&(valid>0);contours,_=cv2.findContours(color.astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);mask=np.zeros((1080,1920),'u1');cv2.drawContours(mask,contours,-1,255,cv2.FILLED);mask[valid==0]=0
 return mask,dict(native_pts=pts,independent_static_print_semantic_support=True,current_native_RGB_component_recognition='current recognizable fixed printed sponsor/ISU board pattern, reviewed original RGB; blue/red ink components and enclosed print holes reread every exposure',current_RGB_blue_or_red_material_contrast_native_DN=12,source_review_ref=ref(O/'direct_component_source_review.json'),world_material_identity_not_issued=True,fixed_2D_material_point_correspondence_not_issued=True,normal_only_material_boundary_observation=True,current_visible_print_material_pixels=int((mask>0).sum()))
REVIEWED_EXTRA=[]
def run():
 global REVIEWED_EXTRA
 scope=load(D/'manifest.json');changed={q['native_pts']for q in scope['changed']};motion=set(changed)|{p+256 for p in changed if p<1425408 and same(p,p+256)};needed=set(motion)|{p-256 for p in motion if p>948992 and same(p,p-256)};seed=op.seeds();counter=300000000;pg=None;pv=None;ppts=None;patchprev=None;patchuv=[];patchids=[];patchfeatures=[];canonical=O/'rows_direct_v5_delta.jsonl.gz';records=[];reviewtiles=[];reviewrecords=[];start=time.perf_counter()
 # Review classification is completed by the source QA step; no semantic promotion until then.
 with av.open(str(h.V))as c,canonical.open('ab')as dest:
  st=c.streams.video[0];c.seek(min(needed),stream=st,backward=True)
  for frame in c.decode(st):
   pts=int(frame.pts)
   if pts>max(needed):break
   if pts not in needed:continue
   base=load(O/'source_v5_frames'/f'{pts}.json');rgb=frame.to_ndarray(format='rgb24');assert sha(rgb.tobytes())==base['decoded_native_RGB_sha256'];gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);valid=bodymask(pts);mv=cv2.erode(valid,np.ones((21,21),'u1'),borderType=cv2.BORDER_CONSTANT,borderValue=0);sid=base['shot_id'];r=dict(base)
   if pts in motion:
    previous=pg if ppts==pts-256 and same(ppts,pts)else None;mo=h.motion(previous,gray,pv if previous is not None else None,mv,pts,sid,ppts if previous is not None else None);mo.update(full_native_21px_LK_patch_current_material_and_actor_and_graphics_exclusion=True,actor_guard_native_px=8,source_unknown_exclusion_guard_native_px=8,composite_mask_index_ref=base['composite_mask_index_ref'],current_source_mask_ref=dict(file=base['mask_file'],sha256=base['mask_sha256']),source_unknown_exclusion_ref=base.get('unknown_exclusion_ref'),final_board_ref=dict(file=base['final_board_file'],sha256=base['final_board_sha256']),qualified_direct_component_repair=True);mp=M/f'{pts}.json';write(mp,mo);r['source_motion_file']=str(mp);r['source_motion_sha256']=sha(mp.read_bytes());r['source_motion_valid_2D_tracks']=len(mo['source_uv'])
   if pts in changed:
    rows=v.rowsfrom(base);oldbody=load(base['body_material_ref']['file']);old=cv2.imread(oldbody['body_material_mask_ref']['file'],0)>0;extra=(valid>0)&~old;extra=cv2.dilate(extra.astype('u1'),np.ones((9,9),'u1'))>0;extra&=valid>0;surface=v.profiles(rows,valid);tree=cKDTree(np.asarray([q['uv']for q,a in zip(rows,surface)if a],float).reshape(-1,2))if surface.any()else None;_,uv,norm,contrast,offset,cids=v.extraedges(rgb,extra,gray)
    if len(uv):
     qq=np.rint(uv[:,None,:]+norm[:,None,:]*np.arange(-3,3.01,.25)[None,:,None]).astype(int);good=(valid[np.clip(qq[:,:,1],0,1079),np.clip(qq[:,:,0],0,1919)]>0).all(1);uv=uv[good];norm=norm[good];contrast=contrast[good];offset=offset[good];cids=cids[good]
    distance=tree.query(uv)[0]if tree is not None and len(uv)else np.zeros(len(uv))+1e9;keep=distance>2;uv=uv[keep];norm=norm[keep];contrast=contrast[keep];offset=offset[keep];cids=cids[keep];flow=op.track(pg if patchprev==pts-256 else None,gray,patchuv if patchprev==pts-256 else [],uv);qq=np.rint(uv).astype(int)
    if len(uv):
     ax=max(0,int(qq[:,0].min())-8);bx=min(1920,int(qq[:,0].max())+9);ay=max(0,int(qq[:,1].min())-8);by=min(1080,int(qq[:,1].max())+9);ev=cv2.cornerEigenValsAndVecs(gray[ay:by,ax:bx],5,3)[qq[:,1]-ay,qq[:,0]-ax,:2]
    else:ev=np.zeros((0,2))
    twod=np.min(ev,axis=1)/np.maximum(np.max(ev,axis=1),1e-12)if len(uv)else np.zeros(0);sr=seed.get(pts,[]);stree=cKDTree(np.asarray([q['uv']for q in sr]))if sr else None;sd,si=stree.query(uv,distance_upper_bound=3)if stree is not None and len(uv)else(np.zeros(len(uv))+1e9,np.zeros(len(uv),int));used={q['point_id']for q in rows};patch=[];ids=[];features=[]
    for i,p in enumerate(uv):
     counter+=1;prior=flow.get(i);pid=patchids[prior[0]]if prior else f'{sid}.v5_direct_delta.{counter}'
     if pid in used:pid=f'{sid}.v5_direct_delta.{counter}'
     used.add(pid);semantic=sr[int(si[i])]['feature_id']if stree is not None and sd[i]<3 else patchfeatures[prior[0]]if prior else None;ids.append(pid);features.append(semantic);patch.append(dict(shot_id=sid,native_pts=pts,feature_id=semantic or f'{sid}.board_print.direct_component.{pts}.{int(cids[i])}',segment_id=f'{pts}.direct_delta_contour.{int(cids[i])}',point_id=pid,uv=[round(float(z),5)for z in p],source_kind='board_ink_material_half_transition'if semantic else'board_body_material_transition_unassigned_print_component',physical_surface='board',board_instance=semantic.split('.')[0]if semantic else None,crosswalk_status='current_pixels_confirmed_board_material_world_component_unbound',world_material_crosswalk='unknown_not_used',pixel_support=dict(normal=[round(float(z),5)for z in norm[i]],contrast_native_DN=round(float(contrast[i]),3),half_transition_offset_px=round(float(offset[i]),5),local_structure_eigenvalue_ratio=round(float(twod[i]),5),two_dimensional_texture_support=bool(twod[i]>.1),complete_plus_minus_3px_RGB_profile_qualified=True,actor_guard_native_px=8,qualified_direct_contact_and_actual_cap_body_component_supported=True,actual_new_domain_RGB_reread=True),FB_evidence=dict(previous_native_pts=patchprev,error_px=prior[1],current_edge_relocalization_px=prior[2],world_point_correspondence=False)if prior else dict(status='new_current_native_pixel_measurement_no_temporal_match'),source_position_precision_unknown=True))
    bb=gzip.compress(''.join(json.dumps(q,ensure_ascii=False,separators=(',',':'))+'\n'for q in patch).encode(),compresslevel=5,mtime=0);pos=dest.tell();dest.write(bb);dest.flush();part=dict(canonical_file=str(canonical),gzip_member_offset=pos,gzip_member_bytes=len(bb),gzip_member_sha256=sha(bb),rows=len(patch),generation='actual_current_RGB_qualified_direct_contact_new_body_component_delta');parts=base['source_parts']+[part];surface=np.r_[surface,np.ones(len(patch),bool)];r.update(source_parts=parts,source_parts_binding_sha256=sha(json.dumps(parts,sort_keys=True,separators=(',',':')).encode()),source_rows=len(surface),surface_eligible_uint8_base64=base64.b64encode(surface.astype('u1').tobytes()).decode(),surface_eligible_rows=int(surface.sum()),body_material_ref=ref(D/f'{pts}.json'),retained_prior_raw_UV_and_ID_rows=len(rows),qualified_direct_repair_new_rows=len(patch),new_current_RGB_domain_pixels=int(extra.sum()));sm,srec=v.semantics.current_static_print_semantics(rgb,valid,pts);sp=T/f'{pts}.png';cv2.imwrite(str(sp),sm);sf=T/f'{pts}.json';srec.update(decoded_native_RGB_sha256=base['decoded_native_RGB_sha256'],mask_ref=ref(sp));write(sf,srec);r['semantic_normal_ref']=ref(sf);patchuv=uv;patchids=ids;patchfeatures=features;patchprev=pts
    tile=cv2.resize(rgb[:,:,::-1][150:620],(640,157));cv2.putText(tile,str(pts),(4,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1);reviewtiles.append(tile);reviewrecords.append(dict(native_pts=pts,source_RGB_sha256=base['decoded_native_RGB_sha256'],source_current_material_component=ref(D/f'{pts}.json')))
   if pts in motion:write(F/f'{pts}.json',r);records.append(dict(native_pts=pts,source_ref=ref(F/f'{pts}.json'),body_changed=pts in changed,new_rows=r.get('qualified_direct_repair_new_rows',0)))
   pg=gray;pv=mv;ppts=pts
 if reviewtiles:
  while len(reviewtiles)%3:reviewtiles.append(np.zeros_like(reviewtiles[0]))
  cv2.imwrite(str(O/'direct_components_current_RGB.jpg'),cv2.vconcat([cv2.hconcat(reviewtiles[i:i+3])for i in range(0,len(reviewtiles),3)]),[cv2.IMWRITE_JPEG_QUALITY,94])
 write(O/'direct_component_source_review.json',dict(source_current_RGB_cases=reviewrecords,all_22_current_RGB_decoded=True,visual_review_pending=True,current_qualified_direct_contact_and_cap_component_proof=True))
 write(O/'source_direct_v5_manifest.json',dict(complete=True,body_changed_frames=len(changed),actual_RGB_exposures_reread=len(needed),motion_records_recomputed=len(motion),new_raw_rows=sum(q['new_rows']for q in records),canonical_delta=ref(canonical),parent_source_manifest=ref(O/'source_v5_manifest.json'),body_direct_domain_recipe=ref(O/'body_qualified_direct_domain.py'),actual_delta_source_reader=ref(Path(__file__)),source_frames_directory=str(F),unmodified_source_frames_fallback=str(O/'source_v5_frames'),records=records,all_original_members_preserved=True,elapsed_seconds=time.perf_counter()-start));print(json.dumps(dict(body_frames=len(changed),actual_RGB=len(needed),motion_frames=len(motion),new_rows=sum(q['new_rows']for q in records),elapsed=time.perf_counter()-start)))
if __name__=='__main__':run()
