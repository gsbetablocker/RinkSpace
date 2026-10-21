import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import av,cv2,numpy as np,json,gzip,hashlib,time,collections,importlib.util,base64
from pathlib import Path
from scipy.spatial import cKDTree
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');R=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2');O=R/'dense56_print_v1';B=R/'dense56_board_v1/shape_v3/anchored_frames';F=O/'source_release_frames';M=O/'source_release_motion'
for d in (O,F,M):d.mkdir(exist_ok=True)
SHOTS=[('s05',948992,1300992,'shot5_20261014'),('s06',1301248,1425408,'shot6_20261014')]
SCOPE=json.loads(Path(r'D:/CodexData/RinkSpace_Data/runs/okada_joint9_1018a/programme_scope.json').read_text(encoding='utf8'));V=Path(SCOPE['source']['path']);SHA=SCOPE['source']['sha256']
spec=importlib.util.spec_from_file_location('successful_native_source',R/'dense_print_v1/dense_print_source.py');op=importlib.util.module_from_spec(spec);spec.loader.exec_module(op);op.SHOTS=SHOTS;cv2.setNumThreads(2)
BLUR_GUARD=8;PROFILE=3
QA={948992,950528,1015552,1055232,1124864,1129472,1205760,1290496,1300992,1301248,1330944,1363456,1395456,1425408}
def sha(b):return hashlib.sha256(b).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf8')
def waitload(p):
 while True:
  try:return load(p)
  except(FileNotFoundError,json.JSONDecodeError):time.sleep(.35)
def wait(p):
 while not p.exists():time.sleep(.35)
def inside(uv,mask):
 x,y=np.rint(uv).astype(int);return 0<=x<1920 and 0<=y<1080 and bool(mask[y,x])
def qualified_row(r,valid):
 if r['source_kind'].startswith('preserved'):
  x,y=np.rint(r['uv']).astype(int)
  return bool(3<=x<1917 and 3<=y<1077 and bool(valid[y-3:y+4,x-3:x+4].all()))
 normal=r.get('pixel_support',{}).get('normal')
 if normal is None:return False
 uv=np.asarray(r['uv'],float);q=uv[None,:]+np.asarray(normal)[None,:]*np.arange(-3,3.01,.25)[:,None];xy=np.rint(q).astype(int)
 return bool(((xy[:,0]>=0)&(xy[:,0]<1920)&(xy[:,1]>=0)&(xy[:,1]<1080)).all())and bool((valid[xy[:,1],xy[:,0]]>0).all())
def bodytop(rgb,top):
 # Current image blue-cap to white-board transition is a surface classification only.
 # It excludes unverified cap highlights from the advertisement reader, never issues a source anchor UV.
 chroma=rgb[:,:,2].astype(np.float32)-rgb[:,:,0].astype(np.float32);off=np.arange(49,dtype=np.float32);x=np.repeat(np.arange(1920,dtype=np.float32)[:,None],49,axis=1);y=top[:,None].astype('f4')+off[None,:];p=cv2.remap(chroma,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE);p=cv2.GaussianBlur(p,(1,15),0)
 high=np.median(p[:,3:8],axis=1);low=np.min((p[:,7:31]+p[:,8:32]+p[:,9:33])/3,axis=1);mid=(high+low)/2;below=p<mid[:,None];cons=below[:,6:39]&below[:,7:40]&below[:,8:41];has=cons.any(1);index=cons.argmax(1)+6;confidence=(high>12)&((high-low)>10)&has
 # When source contrast is insufficient, only the minimum cap margin remains excluded;
 # no inferred cap contour becomes a fixed material point.
 cut=top+np.where(confidence,index,6);return cut,confidence,high,low
def bands(board,actor,rgb):
 y=[np.asarray(board['roles'][role]['full_native_y'],float)for role in ('board.top','board.contact')];assert all(len(a)==1920 for a in y)
 yy=np.arange(1080)[:,None];finite=np.isfinite(y[0]+y[1])&(y[1]>y[0]+8)
 cap,capconf,blue,white=bodytop(rgb,y[0]);raw=(yy>cap[None,:]+1)&(yy<y[1][None,:]-4)&finite[None,:]
 # Conservative actor blur guard is an exclusion margin, never a calibrated blur estimate.
 # Additional 3px disc excludes the complete +/-3 RGB half-transition sampling profile.
 bad=cv2.dilate(actor.astype('u1'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*BLUR_GUARD+1,)*2))>0
 valid=raw.copy();valid&=~bad
 valid[:170,:620]=False;valid[900:,1600:]=False
 return valid.astype('u1')*255,raw.astype('u1')*255,bad,cap,capconf,blue,white

def motion(pg,g,pvalid,valid,pts,sid,ppts):
 out=dict(schema='rinkspace.anchor_source_board_body_static_2D_LK_FB.v1',native_pts=pts,previous_native_pts=ppts,shot_id=sid,cross_cut=False,source_world_projection_used=False)
 if pg is None:out.update(source_uv=[],current_uv=[],forward_backward_error_px=[],status='first_shot_exposure_no_cross_cut');return out
 p=cv2.goodFeaturesToTrack(pg,maxCorners=2000,qualityLevel=.015,minDistance=7,mask=pvalid,blockSize=5)
 if p is None:out.update(source_uv=[],current_uv=[],forward_backward_error_px=[],status='no_unique_board_texture_corner_in_previous_current_RGB');return out
 q,s,e=cv2.calcOpticalFlowPyrLK(pg,g,p,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,25,.01),minEigThreshold=1e-4)
 back,t,eb=cv2.calcOpticalFlowPyrLK(g,pg,q,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,25,.01),minEigThreshold=1e-4)
 fb=np.linalg.norm(back[:,0]-p[:,0],axis=1);xy=np.rint(q[:,0]).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<1920)&(xy[:,1]>=0)&(xy[:,1]<1080);keep=(s[:,0]>0)&(t[:,0]>0)&(fb<.75)&inside;keep&=valid[np.clip(xy[:,1],0,1079),np.clip(xy[:,0],0,1919)]>0
 out.update(candidate_2D_corners=len(p),source_uv=p[keep,0].tolist(),current_uv=q[keep,0].tolist(),forward_backward_error_px=fb[keep].tolist(),status='actual_current_native_board_texture_forward_backward_measured',source_navigation_only=False,pixel_coordinates_not_model_prediction=True)
 return out

def original():
 seed=op.seeds();manual=collections.defaultdict(list);inventory=[]
 for sid,a,b,folder in SHOTS:
  for f in sorted((P/'events/ankara_2026/programmes/okada_mei_sp/source_datasets'/folder/'frames').glob('*.json')):
   br=f.read_bytes();j=json.loads(br);pts=int(j['source_pts']);n=0
   for fi,feature in enumerate(j.get('features',[])):
    if not feature.get('id','').startswith('manual.') or feature.get('physical_surface')!='board':continue
    for si,s in enumerate(feature.get('segments',[])):
     if s.get('measurement_origin')!='user_manual_native_image_click':continue
     for vi,uv in enumerate(s.get('uv',[])):
      manual[pts].append(dict(shot_id=sid,native_pts=pts,feature_id=feature.get('target_id',feature['id']),segment_id=feature['id'],point_id=feature['id'],uv=uv,source_kind='preserved_user_manual_native_image_click',physical_surface='board',board_instance=feature.get('target_id','').split('.')[0],crosswalk_status='existing_user_original_frame_entity_label',world_material_crosswalk='not_reissued_here',source_file=str(f),source_sha256=sha(br),source_pointer=f'/features/{fi}/segments/{si}/uv/{vi}',original_UV_unchanged=True));n+=1
   inventory.append(dict(file=str(f),sha256=sha(br),native_pts=pts,advertisement_readout_rows=len(seed.get(pts,[])),manual_rows=n))
 write(O/'original_source_inventory.json',dict(frames=inventory,source_readout_rows=sum(map(len,seed.values())),manual_rows=sum(map(len,manual.values())),deleted_points_not_recovered_from_history=True,source_UV_not_replaced_by_fitted_uv=True))
 return seed,manual

def run():
 seed,manual=original();counter=20000000;prevgray=None;prevvalid=None;prevuv=[];previds=[];prevfeatures=[];prevsid=None;prevpts=None;refs=[];start=time.perf_counter();canonical=O/'rows_release.jsonl.gz'
 with av.open(str(V))as c,canonical.open('ab')as dest,(O/'source_release_index.stream.jsonl').open('w',encoding='utf8')as ind:
  st=c.streams.video[0];assert str(st.time_base)=='1/12800';c.seek(948992,stream=st,backward=True)
  for frame in c.decode(st):
   pts=int(frame.pts)
   if pts<948992:continue
   if pts>1425408:break
   sid='s05'if pts<=1300992 else's06';bp=B/f'{pts}.json';board=waitload(bp);br=bp.read_bytes();maskp=R/'dense56_masks_v1/masks'/f'{pts}.png';wait(maskp);mask=cv2.imread(str(maskp),0)>0
   rgb=frame.to_ndarray(format='rgb24');bgrsha=sha(rgb[:,:,::-1].copy().tobytes());assert bgrsha==board['decoded_native_BGR_sha256'];valid,raw,bad,cap,capconf,blue,white=bands(board,mask,rgb);capdir=O/'bodycap';capdir.mkdir(exist_ok=True);capfile=capdir/f'{pts}.json';write(capfile,dict(native_pts=pts,decoded_native_RGB_sha256=sha(rgb.tobytes()),board_file=str(bp),board_file_sha256=sha(br),current_native_body_top_classification_y=cap.tolist(),current_native_blue_plateau_chroma=blue.tolist(),current_native_white_gap_chroma=white.tolist(),classification_contrast_supported=capconf.tolist(),classification_not_source_anchor_or_world_correspondence=True,cap_highlights_not_confirmed_as_static_material_entities=True))
   existing=F/f'{pts}.json'
   if existing.exists():
    ref=load(existing);rows=[]
    for part in ref['source_parts']:
     with open(part['canonical_file'],'rb')as f:f.seek(part['gzip_member_offset']);member=f.read(part['gzip_member_bytes'])
     assert sha(member)==part['gzip_member_sha256'];rows.extend(json.loads(q)for q in gzip.decompress(member).decode('utf8').splitlines())
    tr=[r for r in rows if not r['source_kind'].startswith('preserved')];prevgray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);prevvalid=valid;prevuv=np.asarray([r['uv']for r in tr],np.float32).reshape(-1,2);previds=[r['point_id']for r in tr];prevfeatures=[r['feature_id']if r.get('board_instance')else None for r in tr];prevsid=sid;prevpts=pts;refs.append(ref);continue
   gray,newuv,norm,contrast,offset,cids=op.source_edges(rgb,valid)
   # Every half-transition profile lies in the measured board body and beyond SAM+blur exclusion, not just its centre.
   if len(newuv):
    t=np.arange(-3,3.01,.25,dtype=np.float32);q=newuv[:,None,:]+norm[:,None,:]*t[None,:,None];x=np.rint(q[:,:,0]).astype(int);y=np.rint(q[:,:,1]).astype(int);ok=(valid[np.clip(y,0,1079),np.clip(x,0,1919)]>0).all(1);newuv=newuv[ok];norm=norm[ok];contrast=contrast[ok];offset=offset[ok];cids=cids[ok]
   baseparts=[];baserows=[];basepath=O/'source_final_frames'/f'{pts}.json'
   if not basepath.exists():basepath=O/'source8_frames'/f'{pts}.json'
   if not basepath.exists():basepath=O/'source_frames'/f'{pts}.json'
   if basepath.exists():
    base=load(basepath)
    for part in base['source_parts']:
     with open(part['canonical_file'],'rb')as z:z.seek(part['gzip_member_offset']);member=z.read(part['gzip_member_bytes'])
     assert sha(member)==part['gzip_member_sha256'];partrows=[json.loads(q)for q in gzip.decompress(member).decode('utf8').splitlines()];assert len(partrows)==part['rows'];baserows.extend(partrows)
    baseparts=base['source_parts'];tree=cKDTree(np.asarray([r['uv']for r in baserows],float));dist=tree.query(newuv)[0]if len(newuv)else np.zeros(0);keep=dist>2.;newuv=newuv[keep];norm=norm[keep];contrast=contrast[keep];offset=offset[keep];cids=cids[keep]
   if sid!=prevsid:prevgray=None;prevvalid=None;prevuv=[];previds=[];prevfeatures=[];prevpts=None
   mo=motion(prevgray,gray,prevvalid,valid,pts,sid,prevpts);mp=M/f'{pts}.json';write(mp,mo)
   flow=op.track(prevgray,gray,prevuv,newuv);eigen=cv2.cornerEigenValsAndVecs(gray,5,3);qxy=np.rint(newuv).astype(int);ev=eigen[qxy[:,1],qxy[:,0],:2]if len(newuv)else np.zeros((0,2));twod=np.min(ev,axis=1)/np.maximum(np.max(ev,axis=1),1e-12)if len(newuv)else np.zeros(0)
   sr=seed.get(pts,[]);stree=cKDTree(np.array([r['uv']for r in sr],float))if sr else None;sdist,six=stree.query(newuv,distance_upper_bound=3)if stree is not None and len(newuv)else(np.zeros(len(newuv))+1e9,np.zeros(len(newuv),int))
   rows=list(baserows);ids=[];features=[];usedids={r['point_id']for r in rows}
   for i,p in enumerate(newuv):
    counter+=1;prior=flow.get(i);pid=previds[prior[0]]if prior else f'{sid}.native_release_print.{counter}';semantic=None
    if stree is not None and sdist[i]<3:semantic=sr[int(six[i])]['feature_id']
    elif prior:semantic=prevfeatures[prior[0]]
    if pid in usedids:pid=f'{sid}.native_release_print.{counter}'
    usedids.add(pid);ids.append(pid);features.append(semantic)
    row=dict(shot_id=sid,native_pts=pts,feature_id=semantic or f'{sid}.board_print.component.{pts}.{int(cids[i])}',segment_id=f'{pts}.native_contour.{int(cids[i])}',point_id=pid,uv=[round(float(z),5)for z in p],source_kind='board_ink_material_half_transition'if semantic else'board_body_material_transition_unassigned_print_component',physical_surface='board',board_instance=semantic.split('.')[0]if semantic else None,crosswalk_status='source_frame_label_with_current_pixel_support'if semantic else'board_material_current_pixels_confirmed_world_component_unbound',world_material_crosswalk='unknown_not_used',pixel_support=dict(normal=[round(float(z),5)for z in norm[i]],contrast_native_DN=round(float(contrast[i]),3),half_transition_offset_px=round(float(offset[i]),5),local_structure_eigenvalue_ratio=round(float(twod[i]),5),two_dimensional_texture_support=bool(twod[i]>.1),final_current_board_band_supported=True,complete_plus_minus_3px_RGB_profile_qualified=True,actor_SAM2_full_profile_excluded=True,conservative_actor_blur_guard_native_px=BLUR_GUARD),mask_evidence='source_frame_ref',FB_evidence=dict(previous_native_pts=prevpts,error_px=prior[1],current_edge_relocalization_px=prior[2],world_point_correspondence=False)if prior else dict(status='new_current_native_pixel_measurement_no_temporal_match'),source_position_precision_unknown=True)
    if stree is not None and sdist[i]<3:row['seed_source_file']=sr[int(six[i])]['source_file'];row['seed_source_pointer']=sr[int(six[i])]['source_pointer']
    rows.append(row)
   surface=[qualified_row(r,valid)for r in rows];existing_original={tuple(r['uv'])for r in rows if r['source_kind'].startswith('preserved')}
   for j,s in enumerate(sr):
    if tuple(s['uv'])in existing_original:continue
    passed=qualified_row(dict(source_kind='preserved_original',uv=s['uv']),valid);rows.append(dict(shot_id=sid,native_pts=pts,feature_id=s['feature_id'],segment_id=s['feature_id']+'.original_readout',point_id=f'{sid}.{pts}.original.{j}',uv=s['uv'],source_kind='preserved_original_native_advertisement_readout',physical_surface='board'if passed else'original_board_readout_current_visibility_unqualified',board_instance=s['feature_id'].split('.')[0],crosswalk_status='original_source_frame_local_identity',world_material_crosswalk='unknown_not_used',pixel_support=dict(origin=s['origin'],original_UV_unchanged=True,final_current_board_band_supported=passed,complete_plus_minus_3px_RGB_profile_qualified=passed,conservative_actor_blur_guard_native_px=BLUR_GUARD),seed_source_file=s['source_file'],seed_source_pointer=s['source_pointer'],mask_evidence='source_frame_ref',FB_evidence=dict(status='exact_original_current_exposure_readout'),source_position_precision_unknown=True));surface.append(passed)
   for m in manual.get(pts,[]):
    if m['point_id']in usedids:continue
    passed=qualified_row(m,valid);row=dict(m);row['pixel_support']=dict(original_UV_unchanged=True,final_current_board_band_supported=passed,complete_plus_minus_3px_RGB_profile_qualified=passed,conservative_actor_blur_guard_native_px=BLUR_GUARD);row['FB_evidence']=dict(status='exact_original_user_click_current_exposure');rows.append(row);surface.append(passed)
   assert len({r['point_id']for r in rows})==len(rows)
   txt=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n'for r in rows[len(baserows):]).encode('utf8');member=gzip.compress(txt,compresslevel=5,mtime=0);pos=dest.tell();dest.write(member);dest.flush();part=dict(canonical_file=str(canonical),gzip_member_offset=pos,gzip_member_bytes=len(member),gzip_member_sha256=sha(member),rows=len(rows)-len(baserows),generation='current_native_RGB_half_transition_guard8_delta_final_anchored_board')
   surface=np.asarray(surface,'u1');ref=dict(schema='rinkspace.final_native_board_print_source_refs.v1',shot_id=sid,native_pts=pts,time_base='1/12800',decoded_native_BGR_sha256=bgrsha,decoded_native_RGB_sha256=sha(rgb.tobytes()),source_parts=baseparts+[part],source_parts_binding_sha256=sha(json.dumps(baseparts+[part],sort_keys=True,separators=(',',':')).encode()),source_rows=len(rows),surface_eligible_uint8_base64=base64.b64encode(surface.tobytes()).decode('ascii'),surface_eligible_rows=int(surface.sum()),surface_excluded_rows=int((surface==0).sum()),new_current_RGB_half_transition_rows=len(newuv),exact_original_readouts=len(sr),exact_user_clicks=len(manual.get(pts,[])),current_RGB_board_band_pixels=int((raw>0).sum()),current_RGB_profile_and_blur_qualified_pixels=int((valid>0).sum()),final_board_file=str(bp),final_board_sha256=sha(br),mask_file=str(maskp),mask_sha256=sha(maskp.read_bytes()),source_motion_file=str(mp),source_motion_sha256=sha(mp.read_bytes()),source_motion_valid_2D_tracks=len(mo['source_uv']),body_cap_surface_file=str(capfile),body_cap_surface_sha256=sha(capfile.read_bytes()),actor_blur_guard_native_px=BLUR_GUARD,guard_is_conservative_exclusion_not_observed_blur_estimate=True,half_transition_profile_native_px=PROFILE,source_sigma_unknown=True,original_good_source_UV_or_ID_moved=False,no_world_camera_model_or_solver=True)
   write(F/f'{pts}.json',ref);ind.write(json.dumps(ref,separators=(',',':'))+'\n');ind.flush();refs.append(ref)
   tr=[r for r in rows if not r['source_kind'].startswith('preserved')];prevgray=gray;prevvalid=valid;prevuv=np.asarray([r['uv']for r in tr],np.float32).reshape(-1,2);previds=[r['point_id']for r in tr];prevfeatures=[r['feature_id']if r.get('board_instance')else None for r in tr];prevsid=sid;prevpts=pts
   if len(refs)%25==0:pr=dict(frames=len(refs),pts=pts,total_rows=sum(r['source_rows']for r in refs),current_rows=len(rows),source_static_2D_tracks=len(mo['source_uv']),elapsed_seconds=time.perf_counter()-start);write(O/'source_release_progress.json',pr);print(json.dumps(pr),flush=True)
   if pts in QA:
    show=rgb[:,:,::-1].copy()
    for r,a in zip(rows,surface):
     if a:x,y=np.rint(r['uv']).astype(int);cv2.circle(show,(int(x),int(y)),1,(0,255,255),-1,cv2.LINE_AA)
    cv2.imwrite(str(O/f'{sid}_{pts}_source_QA.jpg'),show,[cv2.IMWRITE_JPEG_QUALITY,93])
 assert len(refs)==1862,len(refs)
 with gzip.open(O/'source_release_index.jsonl.gz','wt',encoding='utf8')as f:
  for r in refs:f.write(json.dumps(r,separators=(',',':'))+'\n')
 write(O/'source_release_manifest.json',dict(schema='rinkspace.current_native_dense_print_source_generation.v1',complete=True,frames=len(refs),first_pts=948992,last_pts=1425408,native_step_pts=256,source_time_base='1/12800',source_video_sha256=SHA,canonical_rows_file=str(canonical),canonical_rows_sha256=sha(canonical.read_bytes()),frame_refs_directory=str(F),per_shot={sid:dict(frames=sum(r['shot_id']==sid for r in refs),source_rows=sum(r['source_rows']for r in refs if r['shot_id']==sid),surface_rows=sum(r['surface_eligible_rows']for r in refs if r['shot_id']==sid),original_readouts=sum(r['exact_original_readouts']for r in refs if r['shot_id']==sid))for sid,*_ in SHOTS},no_source_pixel_coordinates_moved=True,deleted_source_points_not_restored=True,all_current_native_RGB_frames_reread=True,static_qualification_required=True,actor_SAM2_conservative_blur_guard_native_px=BLUR_GUARD,no_camera_model_or_solver=True,elapsed_seconds=time.perf_counter()-start))
if __name__=='__main__':run()
