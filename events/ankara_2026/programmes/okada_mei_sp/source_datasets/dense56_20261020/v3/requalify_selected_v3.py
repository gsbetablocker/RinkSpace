"""Requalify frozen source points without moving UV or refilling the display ceiling."""
import os
for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from collections import Counter,defaultdict
import sys,json,gzip,hashlib,base64,time,shutil,itertools
import numpy as np,cv2,av
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O))
from assemble_source_generation_v2 import supports
from source_surface_qualification import current_domains,rejection_reason

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,j):Path(p).write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf8')
def ref(p):return dict(file=str(p),sha256=sha(p))
def stream(p):
 with gzip.open(p,'rt',encoding='utf8')as f:
  for l in f:
   if l.strip():yield json.loads(l)
def groups(p):
 key=None;batch=[]
 for r in stream(p):
  if key is not None and r['native_pts']!=key:yield key,batch;batch=[]
  key=r['native_pts'];batch.append(r)
 if key is not None:yield key,batch
class Lines:
 def __init__(self,p):self.f=gzip.open(p,'rt',encoding='utf8');self.n=-1
 def get(self,n):
  if n<=self.n:raise ValueError('Canonical source line is not forward ordered')
  line=None
  while self.n<n:line=next(self.f);self.n+=1
  return json.loads(line)
 def close(self):self.f.close()

def run(cf):
 cfg=read(cf);pins=[ref(__file__),ref(cf),ref(O/'assemble_source_generation_v2.py'),ref(O/'source_surface_qualification.py')]
 def checked(r):
  assert sha(r['file'])==r['sha256'],'Frozen source input changed '+r['file'];pins.append(r);return Path(r['file'])
 parent=read(checked(cfg['parent_generation_ref']));assert parent['complete']and parent['frames']==1376 and parent['shot_ids']==['s05']
 maskm=read(checked(cfg['mask_manifest_ref']));assert maskm['complete']and maskm['accepted_for_final_source_qualification'];mi={r['native_pts']:r for r in stream(checked(cfg['mask_index_ref']))}
 adm=read(checked(cfg['advert_manifest_ref']));assert adm['complete']and adm['accepted_for_final_source_qualification'];ai={r['native_pts']:r for r in stream(checked(adm['perframe_index']))}
 rm=read(checked(cfg['reflection_manifest_ref']));assert rm['complete']and rm['review_status']=='qualified_source_uncertain_exclusions_visual_QA_complete'and rm['source_immutable_fullfile_SHA_verified'];rv=read(checked(rm['visual_QA']));assert rv['status']=='qualified_source_uncertain_exclusions_only'and rv['current_RGB_and_BGR_bindings_verified'];rb={r['native_pts']:r for r in stream(checked(rm['native_bindings']))};red=defaultdict(dict)
 for r in stream(checked(rm['exclusions'])):
  assert r['source_UV_preserved']and not r['model_or_residual_used'];red[r['native_pts']][r['source_row_id']]=r
 rc={r['native_pts']:r for r in stream(checked(rm['coverage']))}
 expected=list(range(948992,1300992+1,256));assert sorted(mi)==sorted(ai)==sorted(rc)==sorted(rb)==expected
 for r in parent['files'].values():checked(r)
 source=cfg['source'];checked(source);target=Path(cfg['output']);assert not target.exists();target.mkdir(parents=True)
 ice=Lines(checked(cfg['ice_rows_ref']));original=av.open(source['file']);st=original.streams.video[0];assert str(st.time_base)=='1/12800';original.seek(expected[0],stream=st,backward=True);decoder=iter(original.decode(st))
 def current(p):
  for fr in decoder:
   if int(fr.pts)<p:continue
   assert int(fr.pts)==p;return fr.to_ndarray(format='bgr24')
  raise ValueError('Original exposure missing')
 names=['selected_rows','selected_frames','coverage','consumer_bindings'];outputs={k:gzip.open(target/(k+'.jsonl.gz'),'wt',encoding='utf8',compresslevel=4)for k in names};delta=gzip.open(target/'new_exclusions.jsonl.gz','wt',encoding='utf8',compresslevel=4)
 def emit(k,j):outputs[k].write(json.dumps(j,ensure_ascii=False,separators=(',',':'))+'\n')
 cov=iter(stream(parent['files']['coverage.jsonl.gz']['file']));bindings=iter(stream(parent['files']['consumer_bindings.jsonl.gz']['file']));sf=iter(stream(parent['files']['selected_frames.jsonl.gz']['file']));stats=Counter();parts=Counter();classes=Counter();kinds=Counter();features=Counter();counts=[];byreason=Counter();changedframes=[];start=time.perf_counter();cache={}
 try:
  for p,rows in groups(parent['files']['selected_rows.jsonl.gz']['file']):
   c=next(cov);b=next(bindings);frame=next(sf);assert c['native_pts']==b['native_pts']==frame['native_pts']==p;assert len(rows)==len(frame['points'])
   m=mi[p];assert m['source_BGR_sha256']==b['source_BGR_sha256']==frame['decoded_native_BGR_sha256']and m['source_RGB_sha256']==b['source_RGB_sha256']==rc[p]['source_RGB_sha256']==rb[p]['native_RGB_sha256'];assert m['source_BGR_sha256']==rb[p]['native_BGR_sha256']
   changed=m['mask_sha256']!=b['mask']['sha256']or(m.get('source_unknown_exclusion_mask_ref')or{}).get('sha256')!=(b.get('source_unknown_exclusion_mask_ref')or{}).get('sha256')
   distance=None;unknown_distance=None;rawboard=None;border=None;graphics=None;domain=c['source_domain_qualification']
   if changed:
    assert sha(m['mask_file'])==m['mask_sha256'];person=cv2.imread(m['mask_file'],0);assert person is not None and person.shape==(1080,1920);oldperson=cv2.imread(b['mask']['file'],0);assert sha(b['mask']['file'])==b['mask']['sha256']and not((oldperson>0)&(person==0)).any();distance=cv2.distanceTransform((person==0).astype('u1'),cv2.DIST_L2,5)
    ur=m.get('source_unknown_exclusion_mask_ref')
    if ur:
     checked(ur);um=cv2.imread(ur['file'],0);assert um is not None and um.shape==(1080,1920);unknown_distance=cv2.distanceTransform((um==0).astype('u1'),cv2.DIST_L2,5)
     our=b.get('source_unknown_exclusion_mask_ref')
     if our:assert sha(our['file'])==our['sha256']and not((cv2.imread(our['file'],0)>0)&(um==0)).any()
    raw=current(p);assert hashlib.sha256(raw.tobytes()).hexdigest()==m['source_BGR_sha256'];assert hashlib.sha256(raw[:,:,::-1].copy().tobytes()).hexdigest()==m['source_RGB_sha256'];rawboard=read(b['board']['file']);assert sha(b['board']['file'])==b['board']['sha256'];graphics,border,domain=current_domains(raw,rawboard)
    validcontact=[]
    for r in rawboard['rows']:
     if r['feature_id']!='board.contact':continue
     q=supports(r,'board');good=np.isfinite(q).all(1)&(q[:,0]>=0)&(q[:,0]<=1919)&(q[:,1]>=0)&(q[:,1]<=1079);xy=np.rint(q[good]).astype(int)
     ok=len(xy)>0 and not(distance[xy[:,1],xy[:,0]]<=8).any()and not graphics[xy[:,1],xy[:,0]].any()
     if ok and unknown_distance is not None:ok=not(unknown_distance[xy[:,1],xy[:,0]]<=8).any()
     if ok:validcontact.append(r['uv'])
    assert len(validcontact)>=2;db=dict(rawboard,roles=dict(rawboard['roles']));db['roles']['board.contact']=dict(rawboard['roles']['board.contact'],trace_uv=validcontact);graphics,border,domain=current_domains(raw,db);domain['current_contact_valid_samples_after_shoe_delta']=len(validcontact);domain['contact_connection_is_exclusion_only_not_source_observation']=True;stats['actual_source_RGB_for_shoe_changed_frames_verified']+=1
   adref=ai[p];newad=Path(adref['file']).resolve()!=Path(b['advert']['file']).resolve()or adref['sha256']!=b['advert']['sha256'];ad=None;eligible=None;adstatus=None;adtypes=None
   if newad or changed:
    assert sha(adref['file'])==adref['sha256'];ad=read(adref['file']);assert ad['decoded_native_BGR_sha256']==m['source_BGR_sha256']and ad['decoded_native_RGB_sha256']==m['source_RGB_sha256']and ad['mask_ref']['sha256']==m['mask_sha256'];assert(ad.get('unknown_exclusion_ref')or{}).get('sha256')==(m.get('source_unknown_exclusion_mask_ref')or{}).get('sha256');eligible=base64.b64decode(ad['eligible_uint8_base64']);adstatus=base64.b64decode(ad['status_uint8_base64']);adtypes={k:base64.b64decode(ad[k+'_uint8_base64'])for k in('normal_only','texture_2D_tracking','observation_type')}
   source_raw={};needed=sorted({r['source_pixel_or_annotation_ref']['jsonl_line_0based']for r in rows if changed and r['physical_surface']=='ice'})
   for line in needed:
    ir=ice.get(line);assert ir['native_pts']==p;source_raw[line]=ir
   ads=None;kept=[];removed=[];frameparts=Counter();framefeatures=Counter();framekinds=Counter();framewhy=Counter();typechanges=0
   for point in rows:
    r=dict(point);rr=r['source_pixel_or_annotation_ref'];part='ice'if r['physical_surface']=='ice'else'board'if r['feature_id'].startswith('board.')else'print';why=None;proof=None;tc=False
    if part=='ice'and r.get('source_point_id')in red[p]:
     proof=red[p][r['source_point_id']];assert proof['uv']==r['uv']and proof['feature_id']==r['feature_id']and proof['source_RGB_sha256']==m['source_RGB_sha256'];why=proof['reason']
    if part=='print'and ad is not None:
     i=rr['qualification_row_0based'];assert 0<=i<len(eligible)
     if not eligible[i]:why='new_shoe_two_exposure_or_current_material_qualification_status_'+str(adstatus[i])
     else:
      ot=r.get('observation_type_code');r.update(source_qualification_status_code=int(adstatus[i]),source_qualification_status_definition=ad['status_code_definition'].get(str(adstatus[i])),normal_only=bool(adtypes['normal_only'][i]),texture_2D_tracking=bool(adtypes['texture_2D_tracking'][i]),observation_type_code=int(adtypes['observation_type'][i]),observation_type_definition=ad['observation_type_definition'][str(adtypes['observation_type'][i])]);r['measurement_definition']=r['observation_type_definition'];nr=dict(rr,qualification_file=adref['file'],qualification_status_code=int(adstatus[i]),normal_only=r['normal_only'],texture_2D_tracking=r['texture_2D_tracking'],observation_type_code=r['observation_type_code'],observation_type_definition=r['observation_type_definition']);r['source_pixel_or_annotation_ref']=nr;tc=ot!=r['observation_type_code']
    if changed and why is None:
     if part=='ice':sr=source_raw[rr['jsonl_line_0based']]
     elif part=='board':sr=rawboard['rows'][int(rr['json_pointer'].rsplit('/',1)[1])]
     else:
      if ads is None:
       ads=[]
       for member in ad['source_members']:
        path=member['file']
        if path not in cache:cache[path]=open(path,'rb')
        hh=cache[path];hh.seek(member['offset_bytes']);blob=hh.read(member['length_bytes']);assert hashlib.sha256(blob).hexdigest()==member['sha256'];ads.extend(json.loads(l)for l in gzip.decompress(blob).splitlines()if l.strip())
      sr=dict(ads[rr['qualification_row_0based']],_root_qualification_status=int(adstatus[rr['qualification_row_0based']]))
     assert sr['native_pts']==p and sr['uv']==r['uv']and sr['feature_id']==r['feature_id'];q=supports(sr,part);inside=np.isfinite(q).all(1)&(q[:,0]>=0)&(q[:,0]<=1919)&(q[:,1]>=0)&(q[:,1]<=1079);xy=np.rint(q[inside]).astype(int)
     if not inside.all()and part=='ice':why='source_material_profile_outside_native_exposure'
     elif len(xy)==0:why='source_material_profile_unavailable'
     elif(distance[xy[:,1],xy[:,0]]<=8).any():why='new_continuous_raised_shoe_person_and_material_support_guard'
     elif unknown_distance is not None and(unknown_distance[xy[:,1],xy[:,0]]<=8).any():why='new_continuous_raised_shoe_source_uncertainty_guard'
     else:why=rejection_reason(sr,part,graphics,border)
    if why:
     er=dict(shot_id='s05',native_pts=p,source_row_id=r.get('source_point_id'),selected_row_id=r['selected_row_id'],feature_id=r['feature_id'],uv=r['uv'],reason=why,source_kind=r['source_kind'],raw_ref=rr,reflection_material_proof=proof,original_source_UV_preserved=True,physical_solver_or_world_binding_used=False);delta.write(json.dumps(er,ensure_ascii=False,separators=(',',':'))+'\n');removed.append(er);framewhy[why]+=1;byreason[why]+=1;stats['removed_selected_rows']+=1;continue
    assert r['uv']==point['uv'];kept.append(r);typechanges+=tc;frameparts[part]+=1;framefeatures[r['feature_id']]+=1;framekinds[r['source_kind']]+=1;classes[r['display_class']]+=1;kinds[r['source_kind']]+=1;features[r['feature_id']]+=1;parts[part]+=1
   keepids={r['selected_row_id']for r in kept};frame['points']=[r for r in frame['points']if r['selected_row_id']in keepids];assert len(frame['points'])==len(kept)<=500 and len(kept)>0
   for r in kept:emit('selected_rows',r)
   emit('selected_frames',frame);counts.append(len(kept));stats['frames']+=1;stats['kept_selected_rows']+=len(kept);stats['print_type_changes_after_shoe_delta']+=typechanges
   if removed or typechanges:changedframes.append(p)
   nc=dict(c);nc['upstream_candidate_rows_by_part']=nc.pop('candidate_rows_by_part');nc['upstream_rejected_source_rows']=nc.pop('rejected_rows');nc['upstream_qualified_board_rows_by_source_branch']=nc.pop('qualified_board_rows_by_source_branch');nc['upstream_selected_rows_by_part']=c['selected_rows_by_part'];nc['upstream_selection']=c['selection'];nc.update(selected_rows_by_part=dict(frameparts),selected_feature_counts=dict(framefeatures),selected_source_kind_counts=dict(framekinds),new_v3_source_exclusions=dict(framewhy),source_domain_qualification=domain,selection=dict(selected=len(kept),display_ceiling=500,previous_qualified_source_rows_subset=True,refill_to_500=False,source_UV_moved=False),upstream_generation=cfg['parent_generation_ref'],current_material_qualification=cfg['reflection_manifest_ref'],current_actor_generation=cfg['mask_manifest_ref']);emit('coverage',nc)
   nb=dict(b);nb['upstream_v2_consumer_binding']=b;nb['mask']=dict(file=m['mask_file'],sha256=m['mask_sha256']);nb['source_unknown_exclusion_mask_ref']=m.get('source_unknown_exclusion_mask_ref');nb['advert']=dict(file=adref['file'],sha256=adref['sha256']);nb['current_actor_generation']=cfg['mask_manifest_ref'];nb['current_bluegray_material_qualification']=cfg['reflection_manifest_ref'];nb['source_requalification_script']=ref(__file__)
   if ad is not None:nb['advert_actual_source_proofs']={k:ad[k]for k in('static_motion_ref','static_qualification_ref','body_material_ref','graphics_ref','semantic_normal_ref','adjacent_static_motion_refs')}
   emit('consumer_bindings',nb)
   if stats['frames']%50==0:write(target/'progress.json',dict(frames=stats['frames'],total=1376,kept_rows=stats['kept_selected_rows'],removed_rows=stats['removed_selected_rows'],native_pts=p,elapsed_seconds=time.perf_counter()-start))
  assert stats['frames']==1376;assert next(cov,None)is None and next(bindings,None)is None and next(sf,None)is None
 finally:
  for f in outputs.values():f.close()
  delta.close();ice.close();original.close()
  for f in cache.values():f.close()
 dst=target/'excluded.jsonl.gz'
 old_ex=iter(groups(parent['files']['excluded.jsonl.gz']['file']));new_ex=iter(groups(target/'new_exclusions.jsonl.gz'));oe=next(old_ex,None);ne=next(new_ex,None);last_ex=None
 with gzip.open(dst,'wt',encoding='utf8',compresslevel=4)as out:
  while oe is not None or ne is not None:
   ep=min(q[0]for q in(oe,ne)if q is not None);assert last_ex is None or ep>last_ex;last_ex=ep
   for which,q in enumerate((oe,ne)):
    if q is not None and q[0]==ep:
     for row in q[1]:out.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
     if which==0:oe=next(old_ex,None)
     else:ne=next(new_ex,None)
 for r in pins:assert sha(r['file'])==r['sha256']
 files={k+'.jsonl.gz':ref(target/(k+'.jsonl.gz'))for k in names+['excluded']}
 manifest=dict(schema='rinkspace.source_only_dense56_generation.v3',generation_id='dense56_v3',complete=True,shot_ids=['s05'],native_intervals=[dict(shot_id='s05',first_pts=948992,last_pts=1300992)],frames=1376,point_rows=sum(counts),point_count_per_native_frame=dict(min=min(counts),max=max(counts)),by_part=dict(parts),by_shot={'s05':dict(parts)},features_by_shot={'s05':dict(features)},source_kind_counts=dict(kinds),display_classes=dict(classes),manual_source_rows=classes.get('manual',0),rejected_source_rows_by_actual_reason=dict(byreason),actual_source_requalification_counts=dict(stats),source_correction_rows_applied=stats['removed_selected_rows'],files=files,parent_pins=pins,parent_source_generation=cfg['parent_generation_ref'],current_actor_generation=cfg['mask_manifest_ref'],current_advertisement_generation=cfg['advert_manifest_ref'],current_material_qualification=cfg['reflection_manifest_ref'],source_selected_frames_changed_native_pts=changedframes,source_UV_moved=False,source_and_selected_row_IDs_preserved=True,old_generations_unchanged=True,physical_solver_or_live_packet_changed=False,selected_rows_are_independent_measurements=False,physical_world_correspondence_or_independent_weights_issued=False,refill_to_500=False,all_visible_features_perfect_claimed=False,elapsed_seconds=time.perf_counter()-start)
 write(target/'manifest.json',manifest);write(target/'video_config.json',dict(purpose='user_requested_source_points_review',source=source,selected_frame_file=files['selected_frames.jsonl.gz'],shots=[dict(id='s05',first_pts=948992,last_pts=1300992)]));print(json.dumps(dict(complete=True,manifest=ref(target/'manifest.json'),counts=dict(stats),per_frame=manifest['point_count_per_native_frame']),ensure_ascii=False),flush=True)
if __name__=='__main__':run(sys.argv[1])
