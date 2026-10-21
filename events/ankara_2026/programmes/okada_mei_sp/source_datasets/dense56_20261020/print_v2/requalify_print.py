"""Requalify immutable real advertisement UV using newly accepted source domains.
No source row, ID, or compressed member is rewritten. No model projection is used.
"""
import os,sys
sys.dont_write_bytecode=True
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,base64,collections,gzip,hashlib,json,time,importlib.util
from pathlib import Path
import av,cv2,numpy as np
from scipy.spatial import cKDTree
cv2.setNumThreads(2)
SOURCE_SHA='e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d'
SHOT_RANGES={'s05':(948992,1300992),'s06':(1301248,1425408)}
CODES={0:'unknown_no_same_shot_adjacent_static_material_support',1:'new_domain_rebound_actual_adjacent_static_motion_normal_or_2D_supported',2:'preserved_original_native_known_advertisement_material_readout_current_domain_qualified',3:'preserved_user_manual_native_click_current_domain_qualified',4:'actual_adjacent_motion_conflict_outside_static_board_motion_support',5:'unknown_local_motion_or_complete_two_exposure_2D_patch_support_unavailable',6:'excluded_by_complete_current_source_material_person_unknown_graphics_or_near_board_domain',7:'independent_recognized_static_print_semantic_support_normal_only_unbound_no_2D_world_correspondence'}
TYPES={0:'unqualified',1:'normal_only_static_material_boundary_unbound',2:'actual_image_2D_texture_tracking_supported_no_world_identity',3:'preserved_original_native_print_readout',4:'preserved_user_native_click'}
def digest(b):return hashlib.sha256(b).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  while True:
   b=f.read(8*1024*1024)
   if not b:break
   h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,j):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(j,ensure_ascii=False,separators=(',',':')),encoding='utf8')
def ref(p):return dict(file=str(p),sha256=sha(p))
def checked(r):
 p=Path(r['file']);assert sha(p)==r['sha256'],f'Changed pinned dependency {p}';return p
def jref(r):return load(checked(r))
def maskref(r):
 p=checked(r);m=cv2.imread(str(p),0);assert m is not None and m.shape==(1080,1920);return m>0

def accepted(r,label):
 m=jref(r)
 assert m.get('complete') is True,f'{label} is not complete'
 assert any(m.get(k)is True for k in ('accepted','accepted_for_final_source_qualification','accepted_for_source_qualification')),f'{label} has no accepted frozen source receipt'
 return m

def readindex(r):
 p=checked(r);rows={}
 with gzip.open(p,'rt',encoding='utf8')as f:
  for line in f:
   j=json.loads(line);n=int(j['native_pts']);assert n not in rows;rows[n]=j
 return rows

def raw_rows(f):
 out=[]
 for m in f['source_members']:
  with open(m['file'],'rb')as s:s.seek(m['offset_bytes']);b=s.read(m['length_bytes'])
  assert len(b)==m['length_bytes']and digest(b)==m['sha256']
  rows=[json.loads(q)for q in gzip.decompress(b).decode('utf8').splitlines()]
  assert len(rows)==m['row_count'];out.extend(rows)
 assert len(out)==f['source_row_count']
 return out

def at(q,m):
 q=np.asarray(q,float).reshape(-1,2);inside=np.isfinite(q).all(1)&(q[:,0]>=0)&(q[:,0]<=1919)&(q[:,1]>=0)&(q[:,1]<=1079)
 safe=np.where(np.isfinite(q),q,0);p=np.rint(safe).astype(int)
 return inside&m[np.clip(p[:,1],0,1079),np.clip(p[:,0],0,1919)]

def profile_rows(rows,valid):
 n=len(rows);xy=np.asarray([r['uv']for r in rows],float).reshape(-1,2)
 normal=np.asarray([r.get('pixel_support',{}).get('normal')or[0.,0.]for r in rows],float).reshape(-1,2)
 normal_ok=np.isfinite(normal).all(1)&(np.linalg.norm(normal,axis=1)>.5)
 safe=np.where(np.isfinite(normal),normal,0);points=xy[:,None,:]+safe[:,None,:]*np.arange(-3,3.01,.25)[None,:,None]
 out=at(points.reshape(-1,2),valid).reshape(n,25).all(1)&normal_ok
 original=np.asarray([r['source_kind'].startswith('preserved')for r in rows],bool)
 er=cv2.erode(valid.astype('u1'),np.ones((7,7),'u1'),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
 out[original]=at(xy[original],er)
 return out,xy,normal,normal_ok

def b64(a):return base64.b64encode(np.asarray(a,dtype='u1').tobytes()).decode('ascii')
def arr(f,key):return np.frombuffer(base64.b64decode(f[key+'_uint8_base64']),dtype='u1')

class LazyDelta:
 def __init__(self,p):self.path=p;self.stream=None
 def tell(self):return self.stream.tell()if self.stream is not None else 0
 def write(self,b):
  if self.stream is None:self.stream=self.path.open('wb')
  self.stream.write(b)
 def flush(self):
  if self.stream is not None:self.stream.flush()
 def close(self):
  if self.stream is not None:self.stream.close()

def module_from_ref(name,r):
 p=checked(r);sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def track21(previous,gray,uv,current_motionvalid):
 out={}
 if previous is None or len(uv)==0:return out
 pi=np.flatnonzero(previous['surface']&previous['row_motionvalid']);a=previous['xy'][pi].astype('f4').reshape(-1,1,2)
 if not len(a):return out
 b,s,e=cv2.calcOpticalFlowPyrLK(previous['gray'],gray,a,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,25,.01),minEigThreshold=1e-4)
 c,t,eb=cv2.calcOpticalFlowPyrLK(gray,previous['gray'],b,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,25,.01),minEigThreshold=1e-4)
 fb=np.linalg.norm(a[:,0]-c[:,0],axis=1);tree=cKDTree(uv);distance,ix=tree.query(b[:,0],distance_upper_bound=2);used=set();valid=at(b[:,0],current_motionvalid)
 for k in np.argsort(distance):
  i=int(ix[k])
  if s[k,0]and t[k,0]and fb[k]<.75 and valid[k]and distance[k]<2 and i not in used:out[i]=(int(pi[k]),float(fb[k]),float(distance[k]));used.add(i)
 return out

def new_recovery_rows(rows,rgb,gray,restored,valid,motionvalid,previous,sid,pts,helpers,out,delta_stream):
 if not restored.any():return [],None
 yy,xx=np.where(restored);x0=max(0,int(xx.min())-12);x1=min(1920,int(xx.max())+13);y0=max(0,int(yy.min())-12);y1=min(1080,int(yy.max())+13)
 _,uv,norm,contrast,offset,cids=helpers['dense'].source_edges(rgb[y0:y1,x0:x1],restored[y0:y1,x0:x1].astype('u1')*255);uv+=np.asarray([x0,y0]);original=np.asarray([r['uv']for r in rows],float)
 if not len(uv):return [],None
 distance=cKDTree(original).query(uv)[0]if len(original)else np.full(len(uv),np.inf);keep=distance>2;uv=uv[keep];norm=norm[keep];contrast=contrast[keep];offset=offset[keep];cids=cids[keep]
 if not len(uv):return [],None
 flow=track21(previous,gray,uv,motionvalid)if previous is not None and previous['sid']==sid else{};xy=np.rint(uv).astype(int);eigen=cv2.cornerEigenValsAndVecs(gray,5,3);e=eigen[xy[:,1],xy[:,0],:2];ratio=np.min(e,axis=1)/np.maximum(np.max(e,axis=1),1e-12);used={r['point_id']for r in rows};new=[]
 for i,p in enumerate(uv):
  prior=flow.get(i);pid=previous['rows'][prior[0]]['point_id']if prior else f'{sid}.source_nonperson_recovery.{pts}.{i}'
  if pid in used:pid=f'{sid}.source_nonperson_recovery.{pts}.{i}'
  used.add(pid);new.append(dict(shot_id=sid,native_pts=pts,feature_id=f'{sid}.recovered_static_print_component.{pts}.{int(cids[i])}',segment_id=f'{pts}.source_recovered_contour.{int(cids[i])}',point_id=pid,uv=[round(float(v),5)for v in p],source_kind='actual_current_RGB_board_print_half_transition_after_verified_nonperson_reclassification',physical_surface='board',board_instance=None,crosswalk_status='source_material_confirmed_world_component_unbound',world_material_crosswalk='unknown_not_used',pixel_support=dict(normal=[round(float(v),5)for v in norm[i]],contrast_native_DN=round(float(contrast[i]),3),half_transition_offset_px=round(float(offset[i]),5),local_structure_eigenvalue_ratio=round(float(ratio[i]),5),two_dimensional_texture_support=bool(ratio[i]>.1),source_RGB_reread_only_in_verified_old_nonperson_hole=True,actor_guard_native_px=8),FB_evidence=dict(previous_native_pts=previous['pts'],error_px=prior[1],current_edge_relocalization_px=prior[2],complete_21px_window_checked_in_both_exposures=True,world_point_correspondence=False)if prior else dict(status='new_source_current_RGB_half_transition_no_temporal_match'),source_position_precision_unknown=True))
 newvalid,*_=profile_rows(new,valid);new=[r for r,k in zip(new,newvalid)if k]
 if not new:return [],None
 text=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n'for r in new).encode();member=gzip.compress(text,compresslevel=5,mtime=0);position=delta_stream.tell();delta_stream.write(member);delta_stream.flush();m=dict(file=str(out/'rows_source_recovery.jsonl.gz'),offset_bytes=position,length_bytes=len(member),sha256=digest(member),row_count=len(new),source_generation='actual_RGB_verified_nonperson_hole_material_reread')
 return new,m

def new_recovery_motion(prev,cur):
 if prev is None or not prev['restored'].any():return np.zeros((0,2)),np.zeros((0,2)),np.zeros(0)
 region=prev['motionvalid']&prev['restored'];p=cv2.goodFeaturesToTrack(prev['gray'],maxCorners=2000,qualityLevel=.015,minDistance=7,mask=region.astype('u1')*255,blockSize=5)
 if p is None:return np.zeros((0,2)),np.zeros((0,2)),np.zeros(0)
 q,s,e=cv2.calcOpticalFlowPyrLK(prev['gray'],cur['gray'],p,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,25,.01),minEigThreshold=1e-4);b,t,eb=cv2.calcOpticalFlowPyrLK(cur['gray'],prev['gray'],q,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,25,.01),minEigThreshold=1e-4)
 fb=np.linalg.norm(b[:,0]-p[:,0],axis=1);keep=(s[:,0]>0)&(t[:,0]>0)&np.isfinite(fb)&(fb<.75)&at(q[:,0],cur['motionvalid']);return p[keep,0].astype(float),q[keep,0].astype(float),fb[keep].astype(float)

def current_frame(pts,bgr,old,mi,near,cfg,out,helpers,boardcoverage,previous,delta_stream):
 f=jref(dict(file=old['file'],sha256=old['sha256']));checked(f['static_qualification_ref']);checked(f['final_board_ref']);rgb=bgr[:,:,::-1].copy();rgbsha=digest(rgb.tobytes());bgrsha=digest(bgr.tobytes());sid=f['shot_id']
 assert f['native_pts']==pts and f['decoded_native_RGB_sha256']==rgbsha and f['decoded_native_BGR_sha256']==bgrsha
 mr=mi[pts];assert mr['source_RGB_sha256']==rgbsha and mr['source_BGR_sha256']==bgrsha
 actorref=dict(file=mr['mask_file'],sha256=mr['mask_sha256']);actor=maskref(actorref)
 baseactor=maskref(f['mask_ref']);lost=baseactor&~actor;disposition_ref=mr.get('verified_old_nonperson_disposition_ref')
 if lost.any():
  assert disposition_ref is not None,'Old person pixels removed without precise source disposition'
  disposition=jref(disposition_ref);assert disposition['native_pts']==pts and disposition['source_BGR_sha256']==bgrsha and disposition['base_mask_sha256']==f['mask_ref']['sha256']and disposition['old_true_person_pixels_preserved']is True
  nonperson=maskref(disposition['verified_old_nonperson_mask_ref']);assert not(lost&~nonperson).any(),'Removed old person pixels outside source-verified nonperson bitmap'
 ur=mr.get('source_unknown_exclusion_mask_ref');unknown=maskref(ur)if ur else np.zeros_like(actor)
 oldur=f.get('unknown_exclusion_ref');oldunknown=maskref(oldur)if oldur else np.zeros_like(actor);assert not(oldunknown&~unknown).any(),'Accepted effective source unknown generation dropped prior unknown pixels'
 body=jref(f['body_material_ref']);assert body['decoded_native_RGB_sha256']==rgbsha
 baseline=maskref(body['body_material_mask_ref']);grj=jref(f['graphics_ref']);assert grj['decoded_native_RGB_sha256']==rgbsha;graphics=maskref(grj['mask_ref'])
 personclear=cv2.distanceTransform((~actor).astype('u1'),cv2.DIST_L2,3)>8
 unknownclear=cv2.distanceTransform((~unknown).astype('u1'),cv2.DIST_L2,3)>8
 bodyvisible=np.ones_like(actor);outside=np.zeros_like(actor);nr=None
 if sid=='s06':
  nr=dict(near[pts]);nr['frame_ref']=nr.get('frame_ref')or dict(file=nr['frame_file'],sha256=nr['frame_sha256']);nj=jref(nr['frame_ref']);assert nj['native_pts']==pts and nj['source_RGB_sha256']==rgbsha and nj['source_BGR_sha256']==bgrsha;assert all(nj[k]==nr[k]for k in ('board_body_visible_mask_ref','near_outside_exclusion_mask_ref'));assert nr['shot_id']==sid and nr['source_RGB_sha256']==rgbsha and nr['source_BGR_sha256']==bgrsha
  bodyvisible=maskref(nr['board_body_visible_mask_ref']);outside=maskref(nr['near_outside_exclusion_mask_ref']);checked(nr['frame_ref'])
 restored=np.zeros_like(actor);recovery={}
 if lost.any():
  board=jref(f['final_board_ref']);assert board['decoded_native_BGR_sha256']==bgrsha;cv=boardcoverage[pts];qr=dict(file=cv['qualified_file'],sha256=cv['qualified_file_sha256']);qualified=jref(qr);assert qualified['decoded_native_BGR_sha256']==bgrsha
  fresh,_,photo=helpers['body'].body_material_domain(board,rgb,actor,graphics,source_unknown=unknown);direct,di=helpers['direct'].qualified_direct_body(board,qualified,actor,unknown,graphics)
  halo=cv2.dilate(lost.astype('u1'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))>0;restored=((fresh>0)|(direct>0))&halo&~baseline&personclear&unknownclear&~graphics&~outside
  # The old near-body PNG inherited the old false-person hole. Only this exact
  # source-verified hole may recover from independently reread cap/foot material.
  bodyvisible|=restored;recovery=dict(actual_current_RGB_body_recipe=helpers['refs']['body'],actual_qualified_direct_body_recipe=helpers['refs']['direct'],qualified_board_ref=qr,current_photometric_body_receipt=photo,current_qualified_direct_body_receipt=di,recovery_limited_to_verified_removed_nonperson_and_original_8px_guard=True,recovered_body_pixels=int(restored.sum()))
 valid=(baseline|restored)&personclear&unknownclear&~graphics&bodyvisible&~outside
 vr=out/'body'/f'{pts}.png';cv2.imwrite(str(vr),valid.astype('u1')*255)
 proof=dict(schema='rinkspace.rebound_actual_source_advertisement_visible_body.v2',shot_id=sid,native_pts=pts,decoded_native_RGB_sha256=rgbsha,decoded_native_BGR_sha256=bgrsha,parent_body_material_ref=f['body_material_ref'],verified_old_nonperson_disposition_ref=disposition_ref,source_verified_old_nonperson_removed_pixels=int(lost.sum()),body_material_mask_ref=ref(vr),mask_ref=actorref,unknown_exclusion_ref=ur,graphics_ref=f['graphics_ref'],composite_mask_index_ref=cfg['mask_index_ref'],near_board_index_ref=cfg.get('near_index_ref')if sid=='s06'else None,near_board_frame_ref=nr['frame_ref']if nr else None,board_body_visible_mask_ref=nr['board_body_visible_mask_ref']if nr else None,near_outside_exclusion_mask_ref=nr['near_outside_exclusion_mask_ref']if nr else None,source_material_pixels_before=int(baseline.sum()),source_material_pixels_after=int(valid.sum()),removed_by_person_guard=int((baseline&~personclear).sum()),removed_by_unknown_guard=int((baseline&~unknownclear).sum()),new_unknown_source_exclusion_pixels=int((unknown&~oldunknown).sum()),removed_by_near_domain=int((baseline&(~bodyvisible|outside)).sum()),person_guard_native_px=8,source_unknown_guard_native_px=8,guard_is_engineering_exclusion_not_sigma=True,current_native_RGB_verified=True,old_source_domain_never_enlarged_by_requalification=not bool(restored.any()),actual_verified_nonperson_body_recovery=recovery,domain_not_source_point_or_model_projection=True)
 bp=out/'body'/f'{pts}.json';write(bp,proof)
 sj=jref(f['semantic_normal_ref']);assert sj['decoded_native_RGB_sha256']==rgbsha;sm=maskref(sj['mask_ref'])&valid
 sp=out/'semantic'/f'{pts}.png';cv2.imwrite(str(sp),sm.astype('u1')*255)
 sj2=dict(sj,schema='rinkspace.rebound_independent_current_print_semantics.v2',parent_semantic_normal_ref=f['semantic_normal_ref'],mask_ref=ref(sp),new_current_body_material_ref=ref(bp),same_source_semantic_recognition_reused=True,semantic_mask_requalified_with_new_person_and_visible_material=True,world_material_identity_not_issued=True,fixed_2D_material_point_correspondence_not_issued=True)
 sf=out/'semantic'/f'{pts}.json';write(sf,sj2)
 rows=raw_rows(f);oldcount=len(rows);oldstatus=arr(f,'status').copy();assert len(oldstatus)==oldcount;gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);mv=cv2.erode(valid.astype('u1'),np.ones((21,21),'u1'),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
 newrows,member=new_recovery_rows(rows,rgb,gray,restored,valid,mv,previous,sid,pts,helpers,out,delta_stream)
 if member is not None:
  f['source_members']=f['source_members']+[member];f['source_parts_binding_sha256']=digest(json.dumps(f['source_members'],sort_keys=True,separators=(',',':')).encode());f['source_row_count']+=len(newrows);f['new_actual_RGB_material_recovery_rows']=len(newrows);f['parent_preserved_original_source_row_count']=oldcount;rows+=newrows
 surface,xy,norm,normok=profile_rows(rows,valid);n=len(rows)
 semantic=np.zeros(n,bool)
 if sj['independent_static_print_semantic_support']:
  guard=cv2.dilate(sm.astype('u1'),np.ones((7,7),'u1'))>0
  contrast=np.asarray([r.get('pixel_support',{}).get('contrast_native_DN',0)for r in rows],float)
  semantic=at(xy,guard)&surface&normok&(contrast>=12)
 code=np.zeros(n,'u1')
 for i,r in enumerate(rows):
  if not surface[i]:continue
  if r['source_kind']=='preserved_user_manual_native_image_click':code[i]=3
  elif r['source_kind']=='preserved_original_native_advertisement_readout':code[i]=2
 mv=cv2.erode(valid.astype('u1'),np.ones((21,21),'u1'),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
 oldcode=np.r_[oldstatus,np.zeros(len(newrows),'u1')];assert len(oldcode)==n
 ids={r['point_id']:i for i,r in enumerate(rows)};assert len(ids)==n
 return dict(pts=pts,sid=sid,source=f,rows=rows,xy=xy,ids=ids,surface=surface,normal=norm,normal_ok=normok,row_texture=np.asarray([bool(r.get('pixel_support',{}).get('two_dimensional_texture_support'))for r in rows]),semantic=semantic,code=code,back=np.zeros(n,'u1'),forward=np.zeros(n,'u1'),backkind=np.zeros(n,'u1'),forwardkind=np.zeros(n,'u1'),motionerror=np.full(n,np.nan,np.float32),motionvalid=mv,row_motionvalid=at(xy,mv),motion_refs=[],reasons=collections.Counter(),oldstatus=oldcode,body_ref=ref(bp),semantic_ref=ref(sf),person_ref=actorref,unknown_ref=ur,mask_index_ref=cfg['mask_index_ref'],source_RGB_sha256=rgbsha,source_BGR_sha256=bgrsha,near_ref=nr['frame_ref']if nr else None,valid=valid,gray=gray,restored=restored,new_source_rows=len(newrows),old_source_rows=oldcount)

def rebind_motion(prev,cur,out):
 oldref=cur['source']['static_motion_ref'];m=jref(oldref);p=m.get('previous_native_pts')
 assert m['native_pts']==cur['pts']and m['shot_id']==cur['sid']
 src=np.asarray(m.get('source_uv',[]),float).reshape(-1,2);dst=np.asarray(m.get('current_uv',[]),float).reshape(-1,2);fb=np.asarray(m.get('forward_backward_error_px',[]),float);assert len(src)==len(dst)==len(fb)
 same=prev is not None and prev['sid']==cur['sid']and cur['pts']-prev['pts']==256
 if not same:assert len(src)==0,'A cut has source tracks';keep=np.zeros(len(src),bool)
 else:
  assert p==prev['pts'],'Actual LK source exposure mismatch'
  source_patch=at(src,prev['motionvalid']);current_patch=at(dst,cur['motionvalid']);keep=np.isfinite(src).all(1)&np.isfinite(dst).all(1)&np.isfinite(fb)&(fb<.75)&source_patch&current_patch
 retained_src=src[keep];retained_dst=dst[keep];retained_fb=fb[keep];newsrc,newdst,newfb=new_recovery_motion(prev,cur)if same else(np.zeros((0,2)),np.zeros((0,2)),np.zeros(0))
 if len(newsrc)and len(retained_src):
  distinct=cKDTree(retained_src).query(newsrc)[0]>2;newsrc=newsrc[distinct];newdst=newdst[distinct];newfb=newfb[distinct]
 allsrc=np.vstack((retained_src,newsrc));alldst=np.vstack((retained_dst,newdst));allfb=np.r_[retained_fb,newfb]
 mm=dict(m,schema='rinkspace.new_domain_rebound_actual_static_2D_LK_FB.v2',source_uv=allsrc.tolist(),current_uv=alldst.tolist(),forward_backward_error_px=allfb.tolist(),new_actual_RGB_recovered_domain_LK_track_count=len(newsrc),new_actual_RGB_recovered_domain_LK_start_index=len(retained_src),parent_actual_LK_FB_ref=oldref,composite_mask_index_ref=cur['mask_index_ref'],source_unknown_exclusion_ref=cur['unknown_ref'],parent_source_uv_count=len(src),retained_source_track_indices_0based=np.flatnonzero(keep).tolist(),rejected_source_track_indices_0based=np.flatnonzero(~keep).tolist(),retained_source_uv_count=int(keep.sum()),rejected_source_uv_count=int((~keep).sum()),source_patch_rejected=int((~source_patch).sum())if same else 0,current_patch_rejected=int((~current_patch).sum())if same else 0,current_source_RGB_sha256=cur['source_RGB_sha256'],previous_source_RGB_sha256=prev['source_RGB_sha256']if same else None,current_source_mask_ref=cur['person_ref'],previous_source_mask_ref=prev['person_ref']if same else None,current_source_unknown_exclusion_ref=cur['unknown_ref'],previous_source_unknown_exclusion_ref=prev['unknown_ref']if same else None,current_body_material_ref=cur['body_ref'],previous_body_material_ref=prev['body_ref']if same else None,current_near_board_frame_ref=cur['near_ref'],previous_near_board_frame_ref=prev['near_ref']if same else None,complete_21x21_patch_required_in_both_real_exposure_domains=True,raw_LK_FB_coordinates_reused_bitwise_same_unchanged_source_RGB=True,old_static_status_not_reused=True,source_or_current_UV_not_moved=True,source_world_projection_used=False,cross_cut=False)
 mp=out/'motion'/f'{cur["pts"]}.json';write(mp,mm);cur['motion_ref']=ref(mp)
 if same:cur['motion_refs'].append(cur['motion_ref']);prev['motion_refs'].append(cur['motion_ref'])
 return allsrc,alldst,allfb

def qualify_pair(prev,cur,src,dst):
 if prev is None or prev['sid']!=cur['sid']or cur['pts']-prev['pts']!=256:return
 common=[(i,prev['ids'][r['point_id']])for i,r in enumerate(cur['rows'])if r['point_id']in prev['ids']and cur['surface'][i]and prev['surface'][prev['ids'][r['point_id']]]]
 if not common:return
 ci=np.asarray([x[0]for x in common],int);pi=np.asarray([x[1]for x in common],int)
 if len(src)<8:
  cur['back'][ci]=5;prev['forward'][pi]=5;cur['reasons']['real_static_board_LK_support_less_than_8']+=len(ci);return
 delta=dst-src;tree=cKDTree(src);dist,ix=tree.query(prev['xy'][pi],k=min(12,len(src)));d=delta[ix];med=np.median(d,axis=1);res=np.linalg.norm(d-med[:,None,:],axis=2);keep=res<=np.maximum(3.,np.median(res,axis=1)*3)[:,None]
 expected=np.nanmedian(np.where(keep[:,:,None],d,np.nan),axis=1);expected[keep.sum(1)<4]=np.nan
 support=(dist[:,min(3,dist.shape[1]-1)]<=320)&np.isfinite(expected).all(1);error=cur['xy'][ci]-prev['xy'][pi]-expected
 twod_candidate=cur['row_texture'][ci]|prev['row_texture'][pi]
 twod_patch=cur['row_texture'][ci]&prev['row_texture'][pi]&cur['row_motionvalid'][ci]&prev['row_motionvalid'][pi]
 bad2d=support&twod_candidate&~twod_patch
 unavailable=~support|bad2d
 cur['back'][ci[unavailable]]=5;prev['forward'][pi[unavailable]]=5
 badn=int(bad2d.sum());cur['reasons']['2D_raw_point_complete_two_exposure_texture_and_21px_patch_rejected']+=badn;prev['reasons']['2D_raw_point_complete_two_exposure_texture_and_21px_patch_rejected']+=badn
 kind=np.where(twod_candidate,2,1).astype('u1');normal=cur['normal'][ci];normal=normal/np.maximum(np.linalg.norm(normal,axis=1),1e-12)[:,None]
 measured=support&~bad2d&(twod_candidate|cur['normal_ok'][ci]);e=np.where(twod_candidate,np.linalg.norm(error,axis=1),np.abs(np.einsum('ij,ij->i',error,normal)))
 status=np.where(e<=4.,1,4).astype('u1');cur['back'][ci[measured]]=status[measured];prev['forward'][pi[measured]]=status[measured];cur['backkind'][ci[measured]]=kind[measured];prev['forwardkind'][pi[measured]]=kind[measured];cur['motionerror'][ci[measured]]=e[measured]
 for k in (1,2):
  for st,label in ((1,'supported'),(4,'conflict')):
   count=int((measured&(kind==k)&(status==st)).sum());cur['reasons'][f'backward_{k}D_'+label]+=count;prev['reasons'][f'forward_{k}D_'+label]+=count

def finish(frame,out,cfg):
 code=frame['code'];back=frame['back'];forward=frame['forward'];auto=code==0;passed=(back==1)|(forward==1);conflict=((back==4)|(forward==4))&~passed;unavailable=((back==5)|(forward==5))&~passed&~conflict
 code[auto&passed]=1;code[auto&conflict]=4;code[auto&unavailable]=5
 code[frame['semantic']&((code==0)|(code==5))]=7;code[~frame['surface']]=6
 eligible=np.isin(code,[1,2,3,7]);texture=(code==1)&(((frame['backkind']==2)&(back==1))|((frame['forwardkind']==2)&(forward==1)));normal=(code==7)|((code==1)&~texture)
 types=np.zeros(len(code),'u1');types[normal]=1;types[texture]=2;types[code==2]=3;types[code==3]=4
 old=frame['oldstatus'];oldel=np.isin(old,[1,2,3,7]);transition=collections.Counter(f'{a}->{b}'for a,b in zip(old,code));changes=np.flatnonzero(old!=code)
 transitionfile=out/'transitions'/f'{frame["pts"]}.json.gz'
 with gzip.open(transitionfile,'wt',encoding='utf8')as f:
  json.dump(dict(native_pts=frame['pts'],shot_id=frame['sid'],changed_source_row_indices_0based=changes.tolist(),old_status=[int(old[i])for i in changes],new_status=[int(code[i])for i in changes],source_point_ids=[frame['rows'][int(i)]['point_id']for i in changes],source_UV_not_rewritten=True),f,separators=(',',':'))
 proof=dict(schema='rinkspace.requalified_same_shot_actual_material_static_status.v2',shot_id=frame['sid'],native_pts=frame['pts'],source_row_count=len(code),eligible_rows=int(eligible.sum()),status_code_definition=CODES,status_counts=dict(collections.Counter(map(int,code))),old_status_to_new_status_counts=dict(transition),old_eligible_rows=int(oldel.sum()),old_eligible_new_excluded=int((oldel&~eligible).sum()),old_excluded_new_eligible=int((~oldel&eligible).sum()),new_body_material_ref=frame['body_ref'],semantic_normal_ref=frame['semantic_ref'],actual_adjacent_static_motion_refs=frame['motion_refs'],parent_static_qualification_ref=frame['source']['static_qualification_ref'],parent_source_frame_ref=dict(file=frame['source']['_pinned_source_frame_file'],sha256=frame['source']['_pinned_source_frame_sha256']),transition_ledger_ref=ref(transitionfile),new_effective_person_index_ref=cfg['mask_index_ref'],near_board_index_ref=cfg.get('near_index_ref')if frame['sid']=='s06'else None,raw_members_byte_identical=True,new_actual_RGB_source_measurement_rows=frame['new_source_rows'],old_preserved_source_rows=frame['old_source_rows'],actual_source_restored_body_pixels=int(frame['restored'].sum()),old_static_status_not_reused=True,complete_21x21_patch_in_both_exposures_required_for_2D=True,genuine_2D_texture_structure_required_in_both_exposure_raw_readouts=True,failed_2D_patch_cannot_fall_back_normal_without_independent_static_semantics=True,normal_only_static_material_edges_do_not_claim_tangent_or_world_identity=True,engineering_local_motion_capture_native_px=4,engineering_local_nearest4_radius_native_px=320,engineering_threshold_is_not_sigma=True,world_correspondence_and_independent_weights_not_issued=True,reason_counts=dict(frame['reasons']))
 sf=out/'static'/f'{frame["pts"]}.json';write(sf,proof)
 oldf=frame['source'];final={k:v for k,v in oldf.items()if not k.startswith('_pinned_')};final.update(schema='rinkspace.final_dense_advertisement_source_consumer_frame.v2',eligible_rows=int(eligible.sum()),eligible_uint8_base64=b64(eligible),status_uint8_base64=b64(code),status_code_definition=CODES,normal_only_uint8_base64=b64(normal),texture_2D_tracking_uint8_base64=b64(texture),observation_type_uint8_base64=b64(types),observation_type_definition=TYPES,status_counts=proof['status_counts'],mask_ref=frame['person_ref'],unknown_exclusion_ref=frame['unknown_ref'],body_material_ref=frame['body_ref'],static_motion_ref=frame['motion_ref'],adjacent_static_motion_refs=frame['motion_refs'],static_qualification_ref=ref(sf),semantic_normal_ref=frame['semantic_ref'],composite_mask_index_ref=cfg['mask_index_ref'],near_board_index_ref=cfg.get('near_index_ref')if frame['sid']=='s06'else None,near_board_frame_ref=frame['near_ref'],parent_failed_v1_source_frame_ref=proof['parent_source_frame_ref'],source_UV_or_ID_moved=False,source_members_immutable_byte_identical=True,source_detail_file=str(sf),static_detail_file=str(sf),new_qualification_uses_both_real_exposure_domains=True)
 fp=out/'frames'/f'{frame["pts"]}.json';write(fp,final)
 index=dict(native_pts=frame['pts'],shot_id=frame['sid'],file=str(fp),sha256=sha(fp),source_row_count=len(code),eligible_rows=int(eligible.sum()),decoded_native_BGR_sha256=frame['source_BGR_sha256'],decoded_native_RGB_sha256=frame['source_RGB_sha256'],final_board_ref=oldf['final_board_ref'],mask_ref=frame['person_ref'],unknown_exclusion_ref=frame['unknown_ref'],body_material_ref=frame['body_ref'],static_motion_ref=frame['motion_ref'],source_parts_binding_sha256=oldf['source_parts_binding_sha256'])
 summary=dict(shot_id=frame['sid'],native_pts=frame['pts'],source_row_count=len(code),new_actual_RGB_source_rows=frame['new_source_rows'],actual_restored_body_pixels=int(frame['restored'].sum()),eligible_rows=int(eligible.sum()),normal_only_rows=int(normal.sum()),texture_2D_rows=int(texture.sum()),semantic_normal_rows=int((code==7).sum()),old_eligible_rows=int(oldel.sum()),old_eligible_new_excluded=proof['old_eligible_new_excluded'],old_excluded_new_eligible=proof['old_excluded_new_eligible'],status_counts=proof['status_counts'])
 return index,summary

def run(configpath):
 cfg=load(configpath);oldm=jref(cfg['parent_advert_manifest_ref']);assert oldm['complete']is True and oldm['frames']==1862 and oldm['source_video_sha256']==SOURCE_SHA
 maskm=accepted(cfg['mask_manifest_ref'],'effective person generation');mi=readindex(cfg['mask_index_ref']);assert len(mi)==maskm['frames'],'Accepted mask scope and index count differ';old=readindex(oldm['perframe_index']);near={}
 shots=cfg.get('shots',['s05','s06']);assert shots and set(shots)<=set(SHOT_RANGES)
 if 's06'in shots:accepted(cfg['near_manifest_ref'],'near board visible material generation');near=readindex(cfg['near_index_ref']);assert len(near)==486;assert all(p in near for p in range(*[1301248,1425408+1,256]))
 wanted=[p for sid in shots for p in range(SHOT_RANGES[sid][0],SHOT_RANGES[sid][1]+1,256)];wanted.sort();assert all(p in mi and p in old for p in wanted)
 out=Path(cfg['output_directory']);assert not out.exists(),f'Output already exists: {out}';out.mkdir(parents=True)
 for name in ('body','semantic','motion','static','frames','transitions'):(out/name).mkdir()
 deps=[cfg['parent_advert_manifest_ref'],cfg['mask_manifest_ref'],cfg['mask_index_ref']]+([cfg['near_manifest_ref'],cfg['near_index_ref']]if 's06'in shots else[])+[ref(__file__),ref(configpath)]
 for k in ('executed_small_source_dependencies','executed_review_source_dependencies'):
  for r in oldm.get(k,[]):checked(r)
 for m in oldm['canonical_member_files']:checked(m)
 helpers={};boardcoverage={}
 if any(mi[p].get('verified_old_nonperson_disposition_ref')for p in wanted):
  assert cfg.get('board_material_manifest_ref')is not None,'Actual source disposition body reread needs the frozen board manifest'
  bm=jref(cfg['board_material_manifest_ref']);assert bm['complete']is True and bm['quality_accepted_for_raw_point_overlay']is True;br=bm['files']['qualified_coverage.jsonl.gz'];boardcoverage=readindex(br);deps.extend([cfg['board_material_manifest_ref'],br]);refs={}
  for key,name in (('body','body_material_domain.py'),('direct','body_qualified_direct_domain.py'),('dense','dense_print_source.py')):
   rr=next(r for r in oldm['executed_small_source_dependencies']if Path(r['file']).name==name);refs[key]=rr;helpers[key]=module_from_ref('recovery_'+key,rr)
  helpers['refs']=refs
 delta_stream=LazyDelta(out/'rows_source_recovery.jsonl.gz')
 started=time.perf_counter();previous=None;indices=[];summary=[]
 def emit(fr):
  i,s=finish(fr,out,cfg);indices.append(i);summary.append(s)
  if len(indices)%25==0:
   pr=dict(frames=len(indices),expected_frames=len(wanted),native_pts=s['native_pts'],eligible_rows=sum(x['eligible_rows']for x in summary),elapsed_seconds=time.perf_counter()-started);write(out/'progress.json',pr);print(json.dumps(pr),flush=True)
 try:
  with av.open(cfg['source_video'])as c:
   st=c.streams.video[0];assert str(st.time_base)=='1/12800';c.seek(wanted[0],stream=st,backward=True);seen=0
   for vf in c.decode(st):
    p=int(vf.pts)
    if p<wanted[0]:continue
    if p>wanted[-1]:break
    if p not in old or old[p]['shot_id']not in shots:continue
    assert p==wanted[seen],f'Native source clock gap {p}';seen+=1
    bgr=vf.to_ndarray(format='bgr24');cur=current_frame(p,bgr,old[p],mi,near,cfg,out,helpers,boardcoverage,previous,delta_stream);cur['source']['_pinned_source_frame_file']=old[p]['file'];cur['source']['_pinned_source_frame_sha256']=old[p]['sha256']
    src,dst,fb=rebind_motion(previous,cur,out);qualify_pair(previous,cur,src,dst)
    if previous is not None:emit(previous)
    previous=cur
   assert seen==len(wanted)
  if previous is not None:emit(previous)
  assert len(indices)==len(wanted);delta_stream.close()
  canonical=list(oldm['canonical_member_files']);deltafile=out/'rows_source_recovery.jsonl.gz'
  if deltafile.exists():canonical.append(dict(ref(deltafile),bytes=deltafile.stat().st_size,generation='actual_current_RGB_source_recovery_only_when_verified_disposition_required'))
  ip=out/'index.jsonl.gz'
  with gzip.open(ip,'wt',encoding='utf8')as f:
   for i in indices:f.write(json.dumps(i,separators=(',',':'))+'\n')
  sp=out/'coverage.jsonl.gz'
  with gzip.open(sp,'wt',encoding='utf8')as f:
   for s in summary:f.write(json.dumps(s,separators=(',',':'))+'\n')
  for d in deps:checked(d)
  for m in oldm['canonical_member_files']:checked(m)
  per={}
  for sid in shots:
   ss=[s for s in summary if s['shot_id']==sid];per[sid]=dict(frames=len(ss),source_rows=sum(s['source_row_count']for s in ss),eligible_rows=sum(s['eligible_rows']for s in ss),normal_only_rows=sum(s['normal_only_rows']for s in ss),texture_2D_rows=sum(s['texture_2D_rows']for s in ss),semantic_normal_rows=sum(s['semantic_normal_rows']for s in ss),minimum_eligible_per_frame=min(s['eligible_rows']for s in ss),maximum_eligible_per_frame=max(s['eligible_rows']for s in ss),new_actual_RGB_source_rows=sum(s['new_actual_RGB_source_rows']for s in ss),actual_restored_body_pixels=sum(s['actual_restored_body_pixels']for s in ss),old_eligible_rows=sum(s['old_eligible_rows']for s in ss),old_eligible_new_excluded=sum(s['old_eligible_new_excluded']for s in ss),old_excluded_new_eligible=sum(s['old_excluded_new_eligible']for s in ss),status_counts=dict(sum((collections.Counter(s['status_counts'])for s in ss),collections.Counter())))
  manifest=dict(schema='rinkspace.requalified_real_native_advertisement_generation.v2',complete=True,accepted_for_source_consumer=True,accepted_for_final_source_qualification=True,frames=len(indices),first_pts=wanted[0],last_pts=wanted[-1],native_step_pts=256,source_time_base='1/12800',source_video_sha256=SOURCE_SHA,per_shot=per,consumer_frame_directory=str(out/'frames'),perframe_index=ref(ip),coverage_ref=ref(sp),canonical_member_files=canonical,parent_advert_manifest_ref=cfg['parent_advert_manifest_ref'],composite_mask_index=cfg['mask_index_ref'],composite_mask_manifest=cfg['mask_manifest_ref'],near_board_index=cfg.get('near_index_ref')if 's06'in shots else None,near_board_manifest=cfg.get('near_manifest_ref')if 's06'in shots else None,executed_small_source_dependencies=[ref(__file__)],pinned_dependencies=deps,all_current_native_source_RGB_verified=True,old_correct_UV_and_ID_and_raw_members_bitwise_preserved=True,new_two_exposure_21px_source_domains_actually_consumed=True,static_qualifications_recomputed_from_newly_valid_actual_LK_measurements=True,raw_LK_FB_uv_reused_no_unchanged_pixel_recomputation=True,failed_2D_patch_only_can_use_independent_semantics_for_normal_fallback=True,source_UV_or_ID_moved=False,source_acceptance_scope='current source domain and static qualification; complete video release visual review remains separate',world_material_crosswalk_not_issued=True,independent_weights_not_issued=True,no_camera_model_or_solver=True,unknown_and_failed_raw_rows_preserved=True,source_candidate_count_is_not_delivered_display_count=True,elapsed_seconds=time.perf_counter()-started)
  write(out/'manifest.json',manifest);print(json.dumps(manifest),flush=True)
 except Exception as e:
  delta_stream.close();write(out/'FAILED.json',dict(complete=False,accepted_for_source_consumer=False,error=repr(e),frames_finished=len(indices)));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);run(ap.parse_args().config)
