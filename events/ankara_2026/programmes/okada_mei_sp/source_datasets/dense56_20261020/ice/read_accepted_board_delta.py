import json
from pathlib import Path
BINDING_FILE=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_ice_v1/final_board_binding.json');BINDING=json.load(open(BINDING_FILE,encoding='utf8'))
assert BINDING.get('accepted_for_source_overlay')is True,'Await accepted final board generation'
BOARD_NAME=BINDING['board_generation'];OLD_BOARD_NAME=BINDING.get('previous_board_generation_for_delta','shape_v4')
exec(Path(__file__).with_name('reader56_exact.py').read_text(encoding='utf8').split('def main():')[0].replace('shape_v3',BOARD_NAME).replace('SAM2_8px','SAM2_footwear_composite_8px').replace('qualified_v3_boardfoot','qualified_'+BOARD_NAME+'_boardfoot').replace("mp=R/'dense56_masks_v1/masks'/f'{pts}.png';","mp=Path(mask_index[pts]['mask_file']);").replace("old=cv2.dilate((raw>0).astype('u1'),np.ones((11,11),'u1'))>0", "old_raw=cv2.imread(str(mask_index[pts].get('parent_mask_file',mp)),0);old=cv2.dilate((old_raw>0).astype('u1'),np.ones((11,11),'u1'))>0"))
FINAL_PERSON_INDEX=Path(BINDING['final_person_mask_index_file'])
assert TH(FINAL_PERSON_INDEX.read_bytes()).hexdigest()==BINDING['final_person_mask_index_sha256'],'Composite person mask index changed'
mask_index={json.loads(l)['native_pts']:json.loads(l)for l in gzip.open(FINAL_PERSON_INDEX,'rt',encoding='utf8')}
FINAL_PERSON_MANIFEST=Path(BINDING['final_person_mask_manifest_file'])
assert TH(FINAL_PERSON_MANIFEST.read_bytes()).hexdigest()==BINDING['final_person_mask_manifest_sha256']
assert json.load(open(FINAL_PERSON_MANIFEST,encoding='utf8'))['accepted_for_final_source_qualification']is True

BASE_MASK_FRAME=mask_frame
LAST_PERSON_MASK_GUARD=None
LAST_SOURCE_UNKNOWN_GUARD=None
def mask_frame(rgb,pts,sid):
 global LAST_PERSON_MASK_GUARD,LAST_SOURCE_UNKNOWN_GUARD
 mask,info=BASE_MASK_FRAME(rgb,pts,sid);mi=mask_index[pts];person=cv2.imread(mi['mask_file'],0)>0;LAST_PERSON_MASK_GUARD=cv2.dilate(person.astype('u1'),np.ones((17,17),'u1'))>0;ref=mi.get('source_unknown_exclusion_mask_ref');unknown=np.zeros_like(mask)
 if ref:
  uf=Path(ref['file']);assert TH(uf.read_bytes()).hexdigest()==ref['sha256'];assert ref['source_RGB_sha256']==TH(rgb.tobytes()).hexdigest();unknown=cv2.imread(str(uf),0)>0;unknown=cv2.dilate(unknown.astype('u1'),np.ones((17,17),'u1'))>0;mask|=unknown
 LAST_SOURCE_UNKNOWN_GUARD=unknown;info.update(person_mask_semantics=mi['person_mask_semantics'],clean_person_base_mask_file=mi['clean_person_base_mask_file'],clean_person_base_mask_sha256=mi['clean_person_base_mask_sha256'],original_comparison_parent_mask_file=mi['parent_mask_file'],person_composite_index_file=str(FINAL_PERSON_INDEX),person_composite_index_sha256=BINDING['final_person_mask_index_sha256'],source_unknown_exclusion_mask_ref=ref,source_unknown_guard_pixels=8,person_and_source_unknown_masks_separately_typed=True);info['source_domain']+='_and_separately_typed_current_source_unknown_guard'
 return mask,info


spots=collections.defaultdict(list)
for sid,folder in [('s05','shot5_20261014'),('s06','shot6_20261014')]:
 for g in manual[sid].get('static_spot_rois',[]):
  x0,y0,x1,y1=g['source_roi_xyxy'];poly=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],float);stamp=int(g['native_pts']);spots[(sid,g['target_id'])].append(dict(pts=stamp,base=warp(poly,np.linalg.inv(S.matrix(stamp,sid))),source_file=str(sp/folder/'ice_source_guides.json')))
goals=collections.defaultdict(list)
for stamp,data in keyframes.items():
 sid=next(s for s,(a,b)in scopes.items()if a<=stamp<=b)
 for fe in data['features']:
  key=fe.get('world_curve','');g=fe.get('geometric_interpretation')
  if not key.startswith('goal_crease')or not g or'axes_px'not in g:continue
  inv=np.linalg.inv(S.matrix(stamp,sid));uv=ellipse_points(g);goals[(sid,key)].append(dict(pts=stamp,base=warp(uv,inv),center_base=warp(np.asarray(g['center_uv'])[None],inv)[0],source_file='prior source-only inferred ellipse guides navigation, not blue material observation'))

def read_red(rgb,pts,sid,mask,frame):
 rows=[];summ=[];H=S.matrix(pts,sid)
 for(ss,target),gg in spots.items():
  if ss!=sid:continue
  g=min(gg,key=lambda a:abs(a['pts']-pts));quad=warp(g['base'],H);lo=np.maximum([4,4],np.floor(quad.min(0)-[35,16])).astype(int);hi=np.minimum([1916,1076],np.ceil(quad.max(0)+[35,16])).astype(int)
  if(hi-lo<[12,8]).any():continue
  x0,y0=lo;x1,y1=hi;field=rgb[y0:y1,x0:x1,0].astype(float)-rgb[y0:y1,x0:x1,1];smooth=cv2.GaussianBlur(field.astype('f4'),(5,5),1);base=float(np.percentile(smooth,25));peak=float(np.percentile(smooth,95));cut=max(3.,base+.45*(peak-base));component=(smooth>cut)&~mask[y0:y1,x0:x1];n,labels,stats,centroids=cv2.connectedComponentsWithStats(component.astype('u1'),8);candidates=[];expected=quad.mean(0)-lo
  for k,(x,y,w,h,area)in enumerate(stats[1:],1):
   if area<35 or area>.65*component.size or w<8 or h<3 or w/h>25 or h/w>2:continue
   if x<=1 or y<=1 or x+w>=x1-x0-1 or y+h>=y1-y0-1:continue
   cx,cy=centroids[k];score=np.linalg.norm((np.array([cx,cy])-expected)/np.maximum((hi-lo)/2,1));candidates.append((score,k))
  if not candidates:summ.append(dict(feature_id=target,status='current_ROI_no_complete_red_material_component'));continue
  candidates.sort()
  if len(candidates)>1 and candidates[1][0]-candidates[0][0]<.15:summ.append(dict(feature_id=target,status='competing_actual_red_components_material_identity_unknown'));continue
  lab=candidates[0][1];contours,_=cv2.findContours((labels==lab).astype('u1'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);cont=max(contours,key=len)[:,0,:].astype(float)+lo
  if len(cont)<12:continue
  dist=np.r_[0,np.cumsum(np.linalg.norm(np.diff(cont,axis=0),axis=1))];choose=np.r_[0,np.flatnonzero(np.diff(np.floor(dist/8))>0)+1];q=cont[choose];cent=centroids[lab]+lo;normal=q-cent;normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1);ii=np.tile([-18.,18.],(len(q),1));off,pr,sm,val=_batched_actual_profiles(frame,q,normal,ii,np.full(len(q),'red'));kept=0
  for j in range(len(q)):
   c=int(val[j].sum());options=_actual_profile_candidates(q[j],normal[j],off[j,:c],pr[j,:c],sm[j,:c],operator='material_interface',signal='red',transition='light_to_dark',with_quantiles=True);eligible=[]
   for u,edges,bad,qs in options:
    if bad or not context_clear(mask,u,edges,normal[j],qs):continue
    pos=(qs[0]-q[j])@normal[j];contrast=abs(float(np.interp(pos[2],off[j,:c],sm[j,:c])-np.interp(pos[0],off[j,:c],sm[j,:c])));shift=float((u-q[j])@normal[j])
    if contrast<2.5 or abs(shift)>7:continue
    eligible.append((abs(shift),u,qs,contrast))
   if not eligible:continue
   eligible.sort(key=lambda z:z[0]);_,u,qs,contrast=eligible[0];rows.append(dict(shot_id=sid,native_pts=pts,feature_id=target,curve_id=target,definition='confirmed_static_ice_red_material_component_boundary',uv=u.tolist(),paired_edge_uv=None,source_kind='actual_current_native_RGB_red_material_boundary',source_UV_is_direct_measurement=True,surface='ice',source_uv_from_model=False,source_sha256=source_sha,native_time_base='1/12800',pixel_support=dict(material_signal='red_minus_green',current_chroma_component_threshold=cut,signed_interface_quartiles_uv=qs[0].tolist(),signed_interface_IQR_contrast=contrast,profile_normal=normal[j].tolist(),source_RGB_center=rgb[int(round(u[1])),int(round(u[0]))].tolist()),navigation=dict(source_roi_native_pts=g['pts'],source_file=g['source_file'],ROI_used_only_to_locate_search=True),occlusion='current_native_SAM2_real_boardfoot_score_timer_and_pink_reflection_excluded',status='direct_current_pixel_supported_material_identity_requires_continuous_source_qualification',visible_span_id=f'{sid}/{pts}/{target}/0',source_row_id=f'dense56_static/{sid}/{pts}/{target}/{j}',world_entity_binding='unbound_not_for_physical_fit'));kept+=1
  summ.append(dict(feature_id=target,status='actual_current_red_interface_candidates'if kept else'current_material_interface_unresolved',rows=kept,actual_ROI_xyxy=[int(x0),int(y0),int(x1),int(y1)]))
 return rows,summ

def read_goal(rgb,pts,sid,mask,frame):
 rows=[];summary=[];H=S.matrix(pts,sid)
 for(ss,key),gg in goals.items():
  if ss!=sid:continue
  g=min(gg,key=lambda z:abs(z['pts']-pts));uv=warp(g['base'],H);center=warp(g['center_base'][None],H)[0];p=dict(operator='material_interface',material_signal='blue',material_transition='light_to_dark',_projected_center=center);q,nn,ii=_source_queries(p,uv,np.ones(len(uv)),rgb.shape[:2],8.,32.)
  if not len(q):continue
  off,pr,sm,valid=_batched_actual_profiles(frame,q,nn,ii,np.full(len(q),'blue'));count=0
  for i in range(len(q)):
   c=int(valid[i].sum());options=_actual_profile_candidates(q[i],nn[i],off[i,:c],pr[i,:c],sm[i,:c],operator='material_interface',signal='blue',transition='light_to_dark',with_quantiles=True);opts=[]
   for u,edges,bad,qs in options:
    if bad or not context_clear(mask,u,edges,nn[i],qs):continue
    pos=(qs[0]-q[i])@nn[i];contrast=abs(float(np.interp(pos[2],off[i,:c],sm[i,:c])-np.interp(pos[0],off[i,:c],sm[i,:c])));shift=float((u-q[i])@nn[i])
    inner=u-nn[i]*8;outer=u+nn[i]*8;xy=np.rint([inner,outer]).astype(int)
    if(xy<4).any()or(xy>[1915,1075]).any():continue
    vals=rgb[xy[:,1],xy[:,0]].astype(float);bluefill=vals[0,2]-vals[0,0];outerblue=vals[1,2]-vals[1,0]
    if contrast<2 or abs(shift)>20 or bluefill<3 or bluefill-outerblue<1.5:continue
    opts.append((abs(shift),u,qs,contrast,vals))
   if not opts:continue
   opts.sort(key=lambda z:z[0]);_,u,qs,contrast,vals=opts[0];feature='ice.'+key;rows.append(dict(shot_id=sid,native_pts=pts,feature_id=feature,curve_id=key,definition='filled_blue_goal_area_material_boundary',uv=u.tolist(),paired_edge_uv=None,source_kind='actual_current_native_RGB_blue_filled_area_boundary',source_UV_is_direct_measurement=True,surface='ice',source_uv_from_model=False,source_sha256=source_sha,native_time_base='1/12800',pixel_support=dict(material_signal='blue_minus_red',signed_interface_quartiles_uv=qs[0].tolist(),signed_interface_IQR_contrast=contrast,profile_normal=nn[i].tolist(),inner_outer_RGB=vals.tolist(),not_target_ring_or_width_midline=True),navigation=dict(source_guide_native_pts=g['pts'],source_file=g['source_file'],ROI_only=True),occlusion='current_native_SAM2_real_boardfoot_graphics_reflection_excluded',status='direct_current_pixel_supported_blue_filled_area',visible_span_id=f'{sid}/{pts}/{key}/blue_fill',source_row_id=f'dense56_goal/{sid}/{pts}/{key}/{i}',world_entity_binding='actual_material_source_annotation_blue_region_boundary'));count+=1
  summary.append(dict(feature_id='ice.'+key,queries=len(q),rows=count,blue_fill_boundary_operator=True))
 return rows,summary

import multiprocessing as mp
ORIGINAL_CURVE_NAV=curve_nav;TARGET_KEYS=None
def curve_nav(sid,pts):
 nav=ORIGINAL_CURVE_NAV(sid,pts)
 return nav if TARGET_KEYS is None else{k:v for k,v in nav.items()if k in TARGET_KEYS}
def worker_read(task):
 global TARGET_KEYS
 rgb,pts,sid,keys,border3=task;TARGET_KEYS=set(keys);rr,cov,controls=read_frame(rgb,pts,sid)
 if '__red_static__'in keys or '__goal_fill__'in keys:
  mask,ginfo=mask_frame(rgb,pts,sid);frame=_MaterialProfileFrame(rgb,dict(source_pts=pts,source_sha256=source_sha,time_base='1/12800',native_hw=[1080,1920],shot_id=sid))
  if '__red_static__'in keys:extra,_=read_red(rgb,pts,sid,mask,frame);rr.extend(extra)
  if '__goal_fill__'in keys:extra,_=read_goal(rgb,pts,sid,mask,frame);rr.extend(extra)
  cov.update(ginfo)
 cov['native_RGB_sha256']=TH(rgb.tobytes()).hexdigest()
 for r in rr:
  r['source_row_id']='dense56_final_board_'+BOARD_NAME+'/'+r['source_row_id'];ps=r['pixel_support']
  if 'side_interface_quartiles_uv'not in ps and 'signed_interface_quartiles_uv'in ps:ps['side_interface_quartiles_uv']=[ps['signed_interface_quartiles_uv']]
  if 'normal'not in ps and 'profile_normal'in ps:ps['normal']=ps['profile_normal']
  r['canonical_generation']='dense56_final_source_domain_complete_affected_source_curve_reread';r['source_kind']='actual_current_native_RGB_complete_affected_curve_reread_final_source_domain';r['repaired_source_domain_reread']=True;r['reread_reason']='final_person_mask_opened_full_source_frame'if pts in BINDING.get('source_domain_force_reread_native_pts',[])else'final_actual_'+BOARD_NAME+'_board_domain_opened';support=np.vstack([np.asarray(r['uv'])[None],np.asarray(r['paired_edge_uv']).reshape(-1,2)if r['paired_edge_uv']is not None else np.empty((0,2)),np.asarray(r['pixel_support']['side_interface_quartiles_uv']).reshape(-1,2)]);r['pixel_support']['newly_opened_by_final_boardfoot_support']=bool((support[:,1]<=np.interp(support[:,0],np.arange(1920),border3)).any())
 cov['replaces_complete_feature_ids']=['ice.'+k if k.split(':')[0]not in ['A3','A4']else'unbound.ice.paired_paint_band.'+k for k in keys if not k.startswith('__')]+(['unbound.ice.flat_red_static_spot']if '__red_static__'in keys else[])+(['ice.'+k for(ss,k)in goals if ss==sid]if '__goal_fill__'in keys else[]);cov['point_candidates']=len(rr);cov['actual_complete_affected_curve_pixels_reread']=True;cov['final_board_new_support_rows']=sum(r['pixel_support']['newly_opened_by_final_boardfoot_support']for r in rr)
 return rr,cov,controls

def affected_curves(rgb,pts,sid,border3,border4):
 change=border4<border3-.01;nav=ORIGINAL_CURVE_NAV(sid,pts);keys=[]
 if pts in BINDING.get('source_domain_force_reread_native_pts',[]):
  keys=[key for key,(uv,p,g)in nav.items()if not key.startswith('goal_crease')and p['operator']!='unresolved']
  if any(ss==sid for ss,target in spots):keys.append('__red_static__')
  if any(ss==sid for ss,key in goals):keys.append('__goal_fill__')
  return keys
 if not change.any():return keys
 for key,(uv,p,g)in nav.items():
  if key.startswith('goal_crease')or p['operator']=='unresolved':continue
  radius=190 if p['feature_id']in ['B1','B2','C']else 64 if p['feature_id']in ['A1','A2','A3','A4']else 36
  q,nn,ii=_source_queries(p,uv,np.ones(len(uv)),rgb.shape[:2],8.,radius)
  if not len(q):continue
  changed=cv2.dilate(change.astype('u1')[None],np.ones((1,radius*2+1),'u1'))[0]>0;lower=cv2.erode(border4.astype('f4')[None],np.ones((1,radius*2+1),'u1'))[0];upper=cv2.dilate(border3.astype('f4')[None],np.ones((1,radius*2+1),'u1'))[0];x=np.clip(np.rint(q[:,0]).astype(int),0,1919);ylo=q[:,1]-radius*abs(nn[:,1]);yhi=q[:,1]+radius*abs(nn[:,1]);possible=changed[x]&(ylo<=upper[x])&(yhi>=lower[x])
  if possible.any():keys.append(key)

 # The same actual-domain change is checked against the small source red/blue ROI inventories.
 for(ss,target),gg in spots.items():
  if ss!=sid:continue
  g=min(gg,key=lambda a:abs(a['pts']-pts));quad=warp(g['base'],S.matrix(pts,sid));x0=max(0,int(np.floor(quad[:,0].min()-35)));x1=min(1920,int(np.ceil(quad[:,0].max()+35)));y0=quad[:,1].min()-34;y1=quad[:,1].max()+34
  if x1>x0 and change[x0:x1].any()and y0<=np.max(border3[x0:x1])and y1>=np.min(border4[x0:x1]):keys.append('__red_static__');break
 for(ss,key),gg in goals.items():
  if ss!=sid:continue
  g=min(gg,key=lambda a:abs(a['pts']-pts));uv=warp(g['base'],S.matrix(pts,sid));center=warp(g['center_base'][None],S.matrix(pts,sid))[0];q,nn,ii=_source_queries(dict(operator='material_interface',_projected_center=center),uv,np.ones(len(uv)),rgb.shape[:2],8.,32.)
  if not len(q):continue
  x=np.clip(np.rint(q[:,0]).astype(int),0,1919);ch=cv2.dilate(change.astype('u1')[None],np.ones((1,65),'u1'))[0]>0
  if (ch[x]&(q[:,1]-32*abs(nn[:,1])<=border3[x]+32)&(q[:,1]+32*abs(nn[:,1])>=border4[x]-32)).any():keys.append('__goal_fill__');break
 return keys

def main_delta():
 out=O/'accepted_board_delta_layer_unique_v3';out.mkdir(exist_ok=False);a=948992;b=1425408;batch=[];total=0;nf=0;newrows=0;audits=[];start=time.monotonic();stats=collections.Counter();features=collections.Counter();boarddir=R/'dense56_board_v1'/BOARD_NAME
 with mp.get_context('spawn').Pool(8)as pool,gzip.open(out/'rows.jsonl.gz','wt',encoding='utf8')as wr,gzip.open(out/'coverage.jsonl.gz','wt',encoding='utf8')as wc,gzip.open(out/'source_band_controls.jsonl.gz','wt',encoding='utf8')as wctrl,gzip.open(out/'all1862_domain_audit.jsonl.gz','wt',encoding='utf8')as wa,av.open(str(D/'events/ankara_2026/programmes/okada_mei_sp/Mei OKADA (JPN) ｜ Junior Women Short Program ｜ Ankara 2026.mp4'))as con:
  def consume(batch):
   nonlocal nf,total,newrows
   results=pool.map(worker_read,batch,chunksize=1)
   for (rgb,pts,sid,keys,border3),(rr,cov,ctrl)in zip(batch,results):
    for m in ctrl:wctrl.write(json.dumps(m,separators=(',',':'))+'\n')
    for r in rr:
     if r['navigation'].get('whole_current_source_band'):r['navigation']['source_file']=str(out/'source_band_controls.jsonl.gz')
     wr.write(json.dumps(r,separators=(',',':'))+'\n');features[r['feature_id']]+=1
    wc.write(json.dumps(cov,separators=(',',':'))+'\n');total+=len(rr);newrows+=cov['final_board_new_support_rows'];nf+=1
    if nf%32==1 or cov['final_board_new_support_rows']>30:
     im=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
     for r in rr:cv2.circle(im,tuple(np.rint(r['uv']).astype(int)),2,(40,255,80)if r['pixel_support']['newly_opened_by_final_boardfoot_support']else(0,255,255),-1)
     cv2.imwrite(str(out/f'{pts}.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,95])
   wr.flush();wc.flush();wctrl.flush();print(json.dumps(dict(reread_frames=nf,rows=total,actual_new_domain_support_rows=newrows,last_pts=pts,elapsed_seconds=time.monotonic()-start)),flush=True)
  st=con.streams.video[0];con.seek(a,stream=st,backward=True)
  for fr in con.decode(st):
   pts=int(fr.pts)
   if pts>b:break
   if pts<a:continue
   sid=next(s for s,(aa,bb)in scopes.items()if aa<=pts<=bb);qp4=boarddir/'qualified_frames'/f'{pts}.json';qp3=R/'dense56_board_v1'/OLD_BOARD_NAME/'qualified_frames'/f'{pts}.json'
   while not qp4.exists():time.sleep(1)
   q3=json.load(open(qp3,encoding='utf8'));q4=json.load(open(qp4,encoding='utf8'));rgb=fr.to_ndarray(format='rgb24');_,border3,_=current_domains(rgb[:,:,::-1].copy(),q3);_,border4,_=current_domains(rgb[:,:,::-1].copy(),q4);keys=affected_curves(rgb,pts,sid,border3,border4);change=np.maximum(0,border3-border4);openpix=int(np.maximum(0,np.ceil(border3)-np.ceil(border4)).sum());audit=dict(shot_id=sid,native_pts=pts,native_RGB_sha256=TH(rgb.tobytes()).hexdigest(),old_qualified_board_file=str(qp3),old_qualified_board_sha256=TH(qp3.read_bytes()).hexdigest(),final_qualified_board_file=str(qp4),final_qualified_board_sha256=TH(qp4.read_bytes()).hexdigest(),maximum_opened_height_px=float(change.max()),newly_opened_native_pixels=openpix,complete_curve_features_reread=keys,current_person_mask_opened_domain_forced_full_source_reread=pts in BINDING.get('source_domain_force_reread_native_pts',[]),source_navigation_only_no_source_UV_manufactured=True);wa.write(json.dumps(audit,separators=(',',':'))+'\n');audits.append(audit);stats['native_frames_audited']+=1;stats['frames_with_opened_pixels']+=int(openpix>0);stats['frames_with_affected_curve_queries']+=int(bool(keys))
   if keys:batch.append((rgb,pts,sid,keys,border3))
   if len(batch)==16:consume(batch);batch=[]
  if batch:consume(batch)
 assert stats['native_frames_audited']==1862
 manifest=dict(schema='rinkspace.final_board_source_domain_affected_curve_reread.v1',complete=True,audit=dict(stats),reread_frames=nf,rows=total,new_domain_material_support_rows=newrows,feature_counts=dict(features),rows_file=str(out/'rows.jsonl.gz'),coverage_file=str(out/'coverage.jsonl.gz'),audit_file=str(out/'all1862_domain_audit.jsonl.gz'),source_band_controls_file=str(out/'source_band_controls.jsonl.gz'),current_full_affected_source_curve_reread_not_old_points_filter=True,all_other_raws_unchanged=True,final_board=BOARD_NAME,forced_current_person_mask_source_domain_reread_native_pts=BINDING.get('source_domain_force_reread_native_pts',[]),consume='replace listed complete source feature IDs for native frame, then use final accepted source board eligibility for every retained parent/delta/supplement row')
 for k in ['rows_file','coverage_file','audit_file','source_band_controls_file']:manifest[k.replace('_file','_sha256')]=TH(Path(manifest[k]).read_bytes()).hexdigest()
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');print(json.dumps(manifest),flush=True)
if __name__=='__main__':main_delta()
