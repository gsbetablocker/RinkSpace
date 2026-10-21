import os,sys
sys.dont_write_bytecode=True
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import av,cv2,numpy as np,json,gzip,hashlib,time,importlib.util,base64
from scipy.spatial import cKDTree
from pathlib import Path
O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_print_v1');R=O.parent
F=O/'source_v5_frames';M=O/'source_v5_motion';D=O/'body_v5';G=O/'graphics_v5';T=O/'semantics_v5'
for d in (F,M,D,G,T):d.mkdir(exist_ok=True)
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);h=importlib.util.module_from_spec(s);s.loader.exec_module(h);return h
h=module('accepted_source',O/'measure_release.py');op=h.op
body=module('current_material',O/'body_material_domain.py');graphics=module('current_graphics',R/'dense56_v1/source_graphics.py');semantics=module('static_print',O/'static_print_semantics.py');cv2.setNumThreads(2)
def sha(b):return hashlib.sha256(b).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,j):Path(p).write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf8')
def fileref(p):return dict(file=str(p),sha256=sha(Path(p).read_bytes()))
def rowsfrom(ref):
 rows=[]
 for part in ref['source_parts']:
  with open(part['canonical_file'],'rb')as f:f.seek(part['gzip_member_offset']);member=f.read(part['gzip_member_bytes'])
  assert sha(member)==part['gzip_member_sha256'];rr=[json.loads(q)for q in gzip.decompress(member).decode('utf8').splitlines()];assert len(rr)==part['rows'];rows.extend(rr)
 return rows

def profiles(rows,valid):
 if not rows:return np.zeros(0,bool)
 xy=np.asarray([r['uv']for r in rows],float);normal=np.asarray([r.get('pixel_support',{}).get('normal',[0.,0.])for r in rows],float);q=np.rint(xy[:,None,:]+normal[:,None,:]*np.arange(-3,3.01,.25)[None,:,None]).astype(int);inside=(q[:,:,0]>=0)&(q[:,:,0]<1920)&(q[:,:,1]>=0)&(q[:,:,1]<1080);out=(inside&(valid[np.clip(q[:,:,1],0,1079),np.clip(q[:,:,0],0,1919)]>0)).all(1)
 original=np.array([r['source_kind'].startswith('preserved')for r in rows]);p=np.rint(xy).astype(int);er=cv2.erode(valid,np.ones((7,7),'u1'));a=(p[:,0]>=3)&(p[:,0]<1917)&(p[:,1]>=3)&(p[:,1]<1077);out[original]=a[original]&(er[np.clip(p[original,1],0,1079),np.clip(p[original,0],0,1919)]>0)
 hasnormal=np.array([r.get('pixel_support',{}).get('normal')is not None for r in rows]);out&=original|hasnormal
 return out

def extraedges(rgb,extra,gray):
 if not extra.any():return gray,np.zeros((0,2)),np.zeros((0,2)),np.zeros(0),np.zeros(0),np.zeros(0,int)
 yy,xx=np.where(extra);a=max(0,int(xx.min())-12);b=min(1920,int(xx.max())+13);u=max(0,int(yy.min())-12);v=min(1080,int(yy.max())+13);_,uv,norm,contrast,offset,cids=op.source_edges(rgb[u:v,a:b],extra[u:v,a:b].astype('u1')*255);uv+=np.array([a,u]);return gray,uv,norm,contrast,offset,cids

def current_body(bp,rgb,mask,gr,sr,unknown):
 board=load(bp);valid,nav,receipt=body.body_material_domain(board,rgb,mask,gr,[x['uv']for x in sr],source_unknown=unknown);return board,valid,receipt

def run():
 config=load(O/'final_bindings_v5.json');B=Path(config['board_directory']);mi=Path(config['composite_mask_index']);assert config['board_audit_released']is True;assert config['composite_mask_released']is True
 for k in ('board_release_ref','composite_mask_manifest_ref'):
  rr=config[k];assert sha(Path(rr['file']).read_bytes())==rr['sha256']
 maskmanifest=load(config['composite_mask_manifest_ref']['file']);assert maskmanifest['accepted_for_final_source_qualification']is True;assert sha(mi.read_bytes())==maskmanifest['index_sha256'];maskidx={}
 with gzip.open(mi,'rt',encoding='utf8')as f:
  for s in f:j=json.loads(s);maskidx[int(j['native_pts'])]=j
 assert len(maskidx)==1862;seed=op.seeds();refs=[];counter=200000000;pg=None;pv=None;patchuv=[];patchids=[];patchfeatures=[];psid=None;ppts=None;started=time.perf_counter();canonical=O/'rows_v5_delta.jsonl.gz'
 with av.open(str(h.V))as c,canonical.open('ab')as dest,(O/'source_v5_index.stream.jsonl').open('w',encoding='utf8')as ind:
  st=c.streams.video[0];assert str(st.time_base)=='1/12800';c.seek(948992,stream=st,backward=True)
  for frame in c.decode(st):
   pts=int(frame.pts)
   if pts<948992:continue
   if pts>1425408:break
   sid='s05'if pts<=1300992 else's06';bp=B/f'{pts}.json';br=bp.read_bytes();base=load(O/'source_v4_frames'/f'{pts}.json');rgb=frame.to_ndarray(format='rgb24');bgr=rgb[:,:,::-1].copy();rgbsha=sha(rgb.tobytes());bgrsha=sha(bgr.tobytes());assert rgbsha==base['decoded_native_RGB_sha256'];assert bgrsha==base['decoded_native_BGR_sha256'];mr=maskidx[pts];maskp=Path(mr['mask_file']);assert sha(maskp.read_bytes())==mr['mask_sha256'];assert mr['source_RGB_sha256']==rgbsha and mr['source_BGR_sha256']==bgrsha;mask=cv2.imread(str(maskp),0)>0;ur=mr.get('source_unknown_exclusion_mask_ref');unknown=np.zeros((1080,1920),bool)
   if ur is not None:assert sha(Path(ur['file']).read_bytes())==ur['sha256'];unknown=cv2.imread(ur['file'],0)>0
   gr,greceipt=graphics.current_graphics(bgr);sr=seed.get(pts,[]);board,valid,receipt=current_body(bp,rgb,mask,gr,sr,unknown);assert bgrsha==board['decoded_native_BGR_sha256'];gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);motionvalid=cv2.erode(valid,np.ones((21,21),'u1'),borderType=cv2.BORDER_CONSTANT,borderValue=0);existing=F/f'{pts}.json'
   if sid!=psid:pg=None;pv=None;patchuv=[];patchids=[];patchfeatures=[];ppts=None
   if existing.exists():
    ref=load(existing);assert ref['final_board_sha256']==sha(br)and ref['mask_sha256']==mr['mask_sha256'];part=ref['source_parts'][-1]
    with open(part['canonical_file'],'rb')as f:f.seek(part['gzip_member_offset']);bb=f.read(part['gzip_member_bytes'])
    assert sha(bb)==part['gzip_member_sha256'];patch=[json.loads(s)for s in gzip.decompress(bb).decode('utf8').splitlines()];patchuv=np.asarray([r['uv']for r in patch],np.float32).reshape(-1,2);patchids=[r['point_id']for r in patch];counter=max([counter]+[int(pid.rsplit('.',1)[1])for pid in patchids if '.v5_native_delta.'in pid]);patchfeatures=[r['feature_id']if r.get('board_instance')else None for r in patch];pg=gray;pv=motionvalid;psid=sid;ppts=pts;refs.append(ref);continue
   rows=rowsfrom(base);oldboardp=Path(base['final_board_file']);assert sha(oldboardp.read_bytes())==base['final_board_sha256'];oldmask=cv2.imread(base['mask_file'],0)>0;oldvalid,*_=h.bands(load(oldboardp),oldmask,rgb);extra=(valid>0)&(oldvalid==0);extra=cv2.dilate(extra.astype('u1'),np.ones((9,9),'u1'))>0;extra&=valid>0
   mo=h.motion(pg,gray,pv,motionvalid,pts,sid,ppts);mo.update(full_native_21px_LK_patch_current_material_and_actor_and_graphics_exclusion=True,actor_guard_native_px=8,composite_mask_index_ref=fileref(mi),current_source_mask_ref=fileref(maskp),source_unknown_exclusion_ref=ur,source_unknown_exclusion_guard_native_px=8,final_board_ref=fileref(bp));mp=M/f'{pts}.json';write(mp,mo);surface=profiles(rows,valid);oldxy=np.asarray([r['uv']for r,a in zip(rows,surface)if a],float).reshape(-1,2);tree=cKDTree(oldxy)if len(oldxy)else None
   _,uv,norm,contrast,offset,cids=extraedges(rgb,extra,gray)
   if len(uv):
    q=np.rint(uv[:,None,:]+norm[:,None,:]*np.arange(-3,3.01,.25)[None,:,None]).astype(int);good=(valid[np.clip(q[:,:,1],0,1079),np.clip(q[:,:,0],0,1919)]>0).all(1);uv=uv[good];norm=norm[good];contrast=contrast[good];offset=offset[good];cids=cids[good]
   distance=tree.query(uv)[0]if tree is not None and len(uv)else np.zeros(len(uv))+1e9;add=np.flatnonzero(distance>2);uv=uv[add];norm=norm[add];contrast=contrast[add];offset=offset[add];cids=cids[add];flow=op.track(pg,gray,patchuv,uv);qq=np.rint(uv).astype(int)
   if len(uv):
    ax=max(0,int(qq[:,0].min())-8);bx=min(1920,int(qq[:,0].max())+9);ay=max(0,int(qq[:,1].min())-8);by=min(1080,int(qq[:,1].max())+9);eigen=cv2.cornerEigenValsAndVecs(gray[ay:by,ax:bx],5,3);ev=eigen[qq[:,1]-ay,qq[:,0]-ax,:2]
   else:ev=np.zeros((0,2))
   twod=np.min(ev,axis=1)/np.maximum(np.max(ev,axis=1),1e-12)if len(uv)else np.zeros(0);stree=cKDTree(np.asarray([r['uv']for r in sr],float))if sr else None;sdist,six=stree.query(uv,distance_upper_bound=3)if stree is not None and len(uv)else(np.zeros(len(uv))+1e9,np.zeros(len(uv),int));patch=[];ids=[];features=[];used={r['point_id']for r in rows}
   for i,p in enumerate(uv):
    counter+=1;prior=flow.get(i);pid=patchids[prior[0]]if prior else f'{sid}.v5_native_delta.{counter}'
    if pid in used:pid=f'{sid}.v5_native_delta.{counter}'
    used.add(pid);semantic=sr[int(six[i])]['feature_id']if stree is not None and sdist[i]<3 else patchfeatures[prior[0]]if prior else None;ids.append(pid);features.append(semantic)
    patch.append(dict(shot_id=sid,native_pts=pts,feature_id=semantic or f'{sid}.board_print.v5_component.{pts}.{int(cids[i])}',segment_id=f'{pts}.v5_delta_contour.{int(cids[i])}',point_id=pid,uv=[round(float(z),5)for z in p],source_kind='board_ink_material_half_transition'if semantic else'board_body_material_transition_unassigned_print_component',physical_surface='board',board_instance=semantic.split('.')[0]if semantic else None,crosswalk_status='source_frame_label_with_current_pixel_support'if semantic else'board_material_current_pixels_confirmed_world_component_unbound',world_material_crosswalk='unknown_not_used',pixel_support=dict(normal=[round(float(z),5)for z in norm[i]],contrast_native_DN=round(float(contrast[i]),3),half_transition_offset_px=round(float(offset[i]),5),local_structure_eigenvalue_ratio=round(float(twod[i]),5),two_dimensional_texture_support=bool(twod[i]>.1),current_real_board_body_material_component_supported=True,complete_plus_minus_3px_RGB_profile_qualified=True,actor_guard_native_px=8,actual_new_board_body_difference_RGB_reread=True),FB_evidence=dict(previous_native_pts=ppts,error_px=prior[1],current_edge_relocalization_px=prior[2],world_point_correspondence=False)if prior else dict(status='new_current_native_pixel_measurement_no_temporal_match'),source_position_precision_unknown=True))
   txt=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n'for r in patch).encode();member=gzip.compress(txt,compresslevel=5,mtime=0);pos=dest.tell();dest.write(member);dest.flush();part=dict(canonical_file=str(canonical),gzip_member_offset=pos,gzip_member_bytes=len(member),gzip_member_sha256=sha(member),rows=len(patch),generation='actual_v5_new_body_material_difference_current_RGB');parts=base['source_parts']+[part];surface=np.r_[surface,np.ones(len(patch),bool)]
   maskfile=D/f'{pts}.png';cv2.imwrite(str(maskfile),valid);bodyfile=D/f'{pts}.json';receipt.update(native_pts=pts,decoded_native_RGB_sha256=rgbsha,final_board_ref=fileref(bp),mask_ref=fileref(maskp),unknown_exclusion_ref=ur,body_material_mask_ref=fileref(maskfile));write(bodyfile,receipt);gp=G/f'{pts}.png';cv2.imwrite(str(gp),gr.astype('u1')*255);gf=G/f'{pts}.json';write(gf,dict(native_pts=pts,decoded_native_RGB_sha256=rgbsha,mask_ref=fileref(gp),current_RGB_presence=greceipt,actual_shared_graphics_recipe=fileref(R/'dense56_v1/source_graphics.py')));smp,sreceipt=semantics.current_static_print_semantics(rgb,valid,pts);sp=T/f'{pts}.png';cv2.imwrite(str(sp),smp);sf=T/f'{pts}.json';sreceipt.update(decoded_native_RGB_sha256=rgbsha,mask_ref=fileref(sp));write(sf,sreceipt)
   ref=dict(schema='rinkspace.final_native_board_print_source_refs.v2',shot_id=sid,native_pts=pts,time_base='1/12800',decoded_native_RGB_sha256=rgbsha,decoded_native_BGR_sha256=bgrsha,source_parts=parts,source_parts_binding_sha256=sha(json.dumps(parts,sort_keys=True,separators=(',',':')).encode()),source_rows=len(surface),surface_eligible_uint8_base64=base64.b64encode(surface.astype('u1').tobytes()).decode(),surface_eligible_rows=int(surface.sum()),retained_raw_UV_and_ID_rows=len(rows),new_current_RGB_half_transition_rows=len(patch),current_RGB_new_interior_pixels=int(extra.sum()),final_board_file=str(bp),final_board_sha256=sha(br),mask_file=str(maskp),mask_sha256=mr['mask_sha256'],unknown_exclusion_ref=ur,composite_mask_index_ref=fileref(mi),source_motion_file=str(mp),source_motion_sha256=sha(mp.read_bytes()),source_motion_valid_2D_tracks=len(mo['source_uv']),body_material_ref=fileref(bodyfile),graphics_ref=fileref(gf),semantic_normal_ref=fileref(sf),source_sigma_unknown=True,old_correct_source_UV_or_ID_not_moved=True,all_source_unknown_and_conflict_raw_rows_preserved=True,no_camera_model_or_solver=True)
   write(existing,ref);ind.write(json.dumps(ref,separators=(',',':'))+'\n');ind.flush();refs.append(ref);pg=gray;pv=motionvalid;patchuv=uv;patchids=ids;patchfeatures=features;psid=sid;ppts=pts
   if len(refs)%25==0:pr=dict(frames=len(refs),pts=pts,new_delta_rows=sum(r['new_current_RGB_half_transition_rows']for r in refs),source_rows=sum(r['source_rows']for r in refs),static_2D_tracks=len(mo['source_uv']),elapsed_seconds=time.perf_counter()-started);write(O/'source_v5_progress.json',pr);print(json.dumps(pr),flush=True)
 assert len(refs)==1862
 with gzip.open(O/'source_v5_index.jsonl.gz','wt',encoding='utf8')as f:
  for r in refs:f.write(json.dumps(r,separators=(',',':'))+'\n')
 write(O/'source_v5_manifest.json',dict(schema='rinkspace.current_native_advertisement_v5_actual_difference_generation.v1',complete=True,frames=1862,parent_source_manifest=fileref(O/'source_v4_manifest.json'),delta_canonical_file=str(canonical),delta_canonical_sha256=sha(canonical.read_bytes()),new_delta_rows=sum(r['new_current_RGB_half_transition_rows']for r in refs),retained_raw_UV_and_ID_rows=sum(r['retained_raw_UV_and_ID_rows']for r in refs),frame_refs_directory=str(F),final_bindings_ref=fileref(O/'final_bindings_v5.json'),all_current_native_RGB_read=True,new_surface_difference_current_pixels_reread=True,old_UV_and_ID_not_moved=True,no_camera_model_or_solver=True,elapsed_seconds=time.perf_counter()-started))
if __name__=='__main__':run()
