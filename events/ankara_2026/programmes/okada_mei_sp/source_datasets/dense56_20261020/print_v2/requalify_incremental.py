"""Incremental real-source print qualification; reuse only byte-identical actual inputs."""
import os,sys
sys.dont_write_bytecode=True
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,base64,collections,gzip,hashlib,importlib.util,json,time
from pathlib import Path
import av,cv2,numpy as np
cv2.setNumThreads(2)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(file=str(p),sha256=sha(p))
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def checked(r):assert sha(r['file'])==r['sha256'];return Path(r['file'])
def readindex(r):
 rows={}
 with gzip.open(checked(r),'rt',encoding='utf8')as f:
  for s in f:j=json.loads(s);assert j['native_pts']not in rows;rows[j['native_pts']]=j
 return rows

def sig_index(row,is_mask=False):
 if is_mask:return dict(native_pts=row['native_pts'],source_RGB_sha256=row['source_RGB_sha256'],source_BGR_sha256=row['source_BGR_sha256'],person_mask_sha256=row['mask_sha256'],source_unknown_sha256=(row.get('source_unknown_exclusion_mask_ref')or{}).get('sha256'))
 return dict(native_pts=row['native_pts'],source_RGB_sha256=row['decoded_native_RGB_sha256'],source_BGR_sha256=row['decoded_native_BGR_sha256'],person_mask_sha256=row['mask_ref']['sha256'],source_unknown_sha256=(row.get('unknown_exclusion_ref')or{}).get('sha256'))

def read_equivalent(b,p,bgr,index,mi):
 ir=index[p];f=b.jref(dict(file=ir['file'],sha256=ir['sha256']));assert sig_index(ir)==sig_index(mi[p],True)
 assert hashlib.sha256(bgr.tobytes()).hexdigest()==f['decoded_native_BGR_sha256'];rgb=bgr[:,:,::-1].copy();assert hashlib.sha256(rgb.tobytes()).hexdigest()==f['decoded_native_RGB_sha256']
 body=b.jref(f['body_material_ref']);valid=b.maskref(body['body_material_mask_ref']);sj=b.jref(f['semantic_normal_ref']);assert sj['decoded_native_RGB_sha256']==f['decoded_native_RGB_sha256'];sm=b.maskref(sj['mask_ref']);b.checked(f['graphics_ref']);b.checked(f['static_qualification_ref']);b.checked(f['final_board_ref']);b.checked(f['mask_ref'])
 if f.get('unknown_exclusion_ref'):b.checked(f['unknown_exclusion_ref'])
 rows=b.raw_rows(f);old=b.arr(f,'status').copy();surface=old!=6;assert len(rows)==len(old)
 xy=np.asarray([r['uv']for r in rows],float);normal=np.asarray([r.get('pixel_support',{}).get('normal')or[0.,0.]for r in rows],float);normalok=np.isfinite(normal).all(1)&(np.linalg.norm(normal,axis=1)>.5);semantic=np.zeros(len(rows),bool)
 if sj['independent_static_print_semantic_support']:
  sem=cv2.dilate(sm.astype('u1'),np.ones((7,7),'u1'))>0;contrast=np.asarray([r.get('pixel_support',{}).get('contrast_native_DN',0)for r in rows]);semantic=b.at(xy,sem)&surface&normalok&(contrast>=12)
 code=np.zeros(len(rows),'u1')
 for i,r in enumerate(rows):
  if not surface[i]:continue
  if r['source_kind']=='preserved_user_manual_native_image_click':code[i]=3
  elif r['source_kind']=='preserved_original_native_advertisement_readout':code[i]=2
 mv=cv2.erode(valid.astype('u1'),np.ones((21,21),'u1'),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
 f['_pinned_source_frame_file']=ir['file'];f['_pinned_source_frame_sha256']=ir['sha256']
 return dict(pts=p,sid=f['shot_id'],source=f,rows=rows,xy=xy,ids={r['point_id']:i for i,r in enumerate(rows)},surface=surface,normal=normal,normal_ok=normalok,row_texture=np.asarray([bool(r.get('pixel_support',{}).get('two_dimensional_texture_support'))for r in rows]),semantic=semantic,code=code,back=np.zeros(len(rows),'u1'),forward=np.zeros(len(rows),'u1'),backkind=np.zeros(len(rows),'u1'),forwardkind=np.zeros(len(rows),'u1'),motionerror=np.full(len(rows),np.nan,np.float32),motionvalid=mv,row_motionvalid=b.at(xy,mv),motion_refs=[],motion_ref=f['static_motion_ref'],reasons=collections.Counter(),oldstatus=old,body_ref=f['body_material_ref'],semantic_ref=f['semantic_normal_ref'],person_ref=f['mask_ref'],unknown_ref=f.get('unknown_exclusion_ref'),mask_index_ref=f['composite_mask_index_ref'],source_RGB_sha256=f['decoded_native_RGB_sha256'],source_BGR_sha256=f['decoded_native_BGR_sha256'],near_ref=f.get('near_board_frame_ref'),valid=valid,gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),restored=np.zeros_like(valid),new_source_rows=0,old_source_rows=len(rows))

def run(configfile):
 cfg=load(configfile);bpath=checked(cfg['requalifier_ref']);spec=importlib.util.spec_from_file_location('frozen_real_print_qualifier',bpath);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
 bm=b.jref(cfg['base_print_manifest_ref']);assert bm['complete']is True and bm['accepted_for_final_source_qualification']is True;assert set(bm['per_shot'])=={'s05'}and bm['frames']==1376
 mm=b.accepted(cfg['mask_manifest_ref'],'new effective source generation');mi=readindex(cfg['mask_index_ref']);index=readindex(bm['perframe_index']);assert len(index)==1376 and len(mi)==mm['frames'];assert all(p in mi for p in index)
 source_sha=b.SOURCE_SHA;assert bm['source_video_sha256']==source_sha and mm['source_video_sha256']==source_sha
 out=Path(cfg['output_directory']);assert not out.exists();out.mkdir(parents=True)
 for name in ('body','semantic','motion','static','frames','transitions'):(out/name).mkdir()
 pins=[cfg['requalifier_ref'],cfg['base_print_manifest_ref'],cfg['mask_manifest_ref'],cfg['mask_index_ref'],ref(__file__),ref(configfile)]
 if cfg.get('change_scope_ref'):checked(cfg['change_scope_ref']);pins.append(cfg['change_scope_ref'])
 for r in bm['canonical_member_files']:checked(r)
 changed=set();signature_changes=[]
 for p,ir in index.items():
  old=sig_index(ir);new=sig_index(mi[p],True);assert old['source_RGB_sha256']==new['source_RGB_sha256']and old['source_BGR_sha256']==new['source_BGR_sha256']
  if old!=new:
   f=b.jref(dict(file=ir['file'],sha256=ir['sha256']));oldp=b.maskref(f['mask_ref']);newp=b.maskref(dict(file=mi[p]['mask_file'],sha256=mi[p]['mask_sha256']));oldur=f.get('unknown_exclusion_ref');newur=mi[p].get('source_unknown_exclusion_mask_ref');oldu=b.maskref(oldur)if oldur else np.zeros_like(oldp);newu=b.maskref(newur)if newur else np.zeros_like(oldp)
   assert not(oldp&~newp).any(),'Incremental append-only correction removed person pixels';assert not(oldu&~newu).any(),'Incremental append-only correction removed source unknown pixels'
   pixeldifference=int(np.count_nonzero(oldp!=newp))+int(np.count_nonzero(oldu!=newu));assert pixeldifference>0,'SHA-only alias change must be handled separately, not called a source correction';changed.add(p);signature_changes.append(dict(native_pts=p,old=old,new=new,source_pixel_difference=pixeldifference))
 assert changed,'No actual source changes for incremental generation'
 pairs=changed|{p+256 for p in changed if p+256 in index};qual=changed|{p+d for p in changed for d in(-256,256)if p+d in index};context=qual|{p+d for p in qual for d in(-256,256)if p+d in index}
 sortedpts=sorted(context);groups=[]
 for p in sortedpts:
  if not groups or p!=groups[-1][-1]+256:groups.append([p])
  else:groups[-1].append(p)
 b.write(out/'actual_change_scope.json',dict(changed_source_domain_pts=sorted(changed),changed_pair_current_pts=sorted(pairs),requalified_pts=sorted(qual),actual_RGB_context_pts=sortedpts,change_details=signature_changes,world_or_model_used=False))
 basecov=readindex(bm['coverage_ref']);replaced={};newcov={};delta=b.LazyDelta(out/'rows_source_recovery.jsonl.gz');started=time.perf_counter();readframes=0;previous=None
 def emit(frame):
  ir,s=b.finish(frame,out,cfg);replaced[frame['pts']]=ir;newcov[frame['pts']]=s
 def pair(prev,cur):
  if cur['pts']in pairs:return b.rebind_motion(prev,cur,out)[:2]
  r=cur['source']['static_motion_ref'];m=b.jref(r);assert m['native_pts']==cur['pts']and m['previous_native_pts']==prev['pts'];assert (m['current_source_mask_ref']['sha256']==cur['person_ref']['sha256']and m['previous_source_mask_ref']['sha256']==prev['person_ref']['sha256']);assert (m.get('current_source_unknown_exclusion_ref')or{}).get('sha256')==(cur['unknown_ref']or{}).get('sha256')and(m.get('previous_source_unknown_exclusion_ref')or{}).get('sha256')==(prev['unknown_ref']or{}).get('sha256');assert m['current_body_material_ref']==cur['body_ref']and m['previous_body_material_ref']==prev['body_ref']
  src=np.asarray(m['source_uv'],float).reshape(-1,2);dst=np.asarray(m['current_uv'],float).reshape(-1,2);cur['motion_ref']=r;cur['motion_refs'].append(r);prev['motion_refs'].append(r);return src,dst
 try:
  with av.open(cfg['source_video'])as c:
   st=c.streams.video[0];assert str(st.time_base)=='1/12800'
   for group in groups:
    previous=None;c.seek(group[0],stream=st,backward=True);i=0
    for vf in c.decode(st):
     p=int(vf.pts)
     if p<group[0]:continue
     if p>group[-1]:break
     assert p==group[i];i+=1;readframes+=1;bgr=vf.to_ndarray(format='bgr24')
     if p in changed:
      cur=b.current_frame(p,bgr,index[p],mi,{},cfg,out,{}, {},previous,delta);cur['source']['_pinned_source_frame_file']=index[p]['file'];cur['source']['_pinned_source_frame_sha256']=index[p]['sha256'];cur['motion_ref']=cur['source']['static_motion_ref']
     else:cur=read_equivalent(b,p,bgr,index,mi)
     if previous is not None and(cur['pts']in qual or previous['pts']in qual):
      src,dst=pair(previous,cur);b.qualify_pair(previous,cur,src,dst)
     elif previous is None and p in pairs and p==min(index):b.rebind_motion(None,cur,out)
     if previous is not None and previous['pts']in qual:emit(previous)
     previous=cur
     if readframes%25==0:
      pr=dict(actual_RGB_read_frames=readframes,expected_context_frames=len(context),qualification_frames_finished=len(replaced),changed_source_domains=len(changed),elapsed_seconds=time.perf_counter()-started);b.write(out/'progress.json',pr);print(json.dumps(pr),flush=True)
    assert i==len(group)
    if previous is not None and previous['pts']in qual:emit(previous)
  delta.close();assert set(replaced)==qual and readframes==len(context);assert not(out/'rows_source_recovery.jsonl.gz').exists(),'Append-only unknown correction must not add unneeded source UV'
  ip=out/'index.jsonl.gz';cp=out/'coverage.jsonl.gz';rp=out/'unchanged_input_reuse.jsonl.gz';allcov=[]
  with gzip.open(ip,'wt',encoding='utf8')as fi,gzip.open(cp,'wt',encoding='utf8')as fc,gzip.open(rp,'wt',encoding='utf8')as fr:
   for p in sorted(index):
    if p in replaced:ir=replaced[p];ss=newcov[p]
    else:
     ir=index[p];checked(dict(file=ir['file'],sha256=ir['sha256']));ss=dict(basecov[p]);ss.update(old_eligible_rows=ss['eligible_rows'],old_eligible_new_excluded=0,old_excluded_new_eligible=0);neighbors=[n for n in(p-256,p,p+256)if n in index]
     for n in neighbors:assert sig_index(index[n])==sig_index(mi[n],True)
     pp=[n for n in(p,p+256)if n in index];assert not(set(pp)&pairs);fr.write(json.dumps(dict(native_pts=p,source_frame_ref=dict(file=ir['file'],sha256=ir['sha256']),actual_inputs_equivalence=[dict(native_pts=n,base=sig_index(index[n]),new=sig_index(mi[n],True))for n in neighbors],unchanged_pair_current_pts=pp,original_body_semantic_motion_and_qualification_refs_reused=True,same_alias_alone_not_used_as_evidence=True),separators=(',',':'))+'\n')
    allcov.append(ss);fi.write(json.dumps(ir,separators=(',',':'))+'\n');fc.write(json.dumps(ss,separators=(',',':'))+'\n')
  for r in pins:checked(r)
  for r in bm['canonical_member_files']:checked(r)
  counts=collections.Counter()
  for s in allcov:counts.update({int(k):int(v)for k,v in s['status_counts'].items()})
  per=dict(frames=1376,source_rows=sum(s['source_row_count']for s in allcov),eligible_rows=sum(s['eligible_rows']for s in allcov),normal_only_rows=sum(s['normal_only_rows']for s in allcov),texture_2D_rows=sum(s['texture_2D_rows']for s in allcov),semantic_normal_rows=sum(s['semantic_normal_rows']for s in allcov),minimum_eligible_per_frame=min(s['eligible_rows']for s in allcov),maximum_eligible_per_frame=max(s['eligible_rows']for s in allcov),old_eligible_rows=bm['per_shot']['s05']['eligible_rows'],old_eligible_new_excluded=sum(s['old_eligible_new_excluded']for s in allcov),old_excluded_new_eligible=sum(s['old_excluded_new_eligible']for s in allcov),status_counts=dict(counts),new_actual_RGB_source_rows=0)
  result=dict(schema='rinkspace.incremental_actual_source_print_qualification.v3',complete=True,accepted_for_source_consumer=True,accepted_for_final_source_qualification=True,frames=1376,first_pts=min(index),last_pts=max(index),native_step_pts=256,source_time_base='1/12800',source_video_sha256=source_sha,per_shot={'s05':per},perframe_index=ref(ip),coverage_ref=ref(cp),base_print_manifest_ref=cfg['base_print_manifest_ref'],composite_mask_index=cfg['mask_index_ref'],composite_mask_manifest=cfg['mask_manifest_ref'],canonical_member_files=bm['canonical_member_files'],source_actual_change_scope=ref(out/'actual_change_scope.json'),unchanged_input_reuse_ref=ref(rp),requalified_frames=len(qual),changed_source_domains=len(changed),changed_21px_motion_pairs=sum(p-256 in index for p in pairs),changed_motion_proof_files=len(pairs),newly_decoded_RGB_context_frames=len(context),unchanged_frame_qualification_refs_reused=1376-len(qual),all_output_source_RGB_identity_proven=True,all_changed_two_exposure_LK_patch_domains_actually_requalified=True,old_correct_source_UV_and_all_original_members_bitwise_preserved=True,new_source_UV_rows=0,no_model_projection_or_world_identity_or_independent_weights_issued=True,executed_small_source_dependencies=[ref(__file__),cfg['requalifier_ref']],pinned_dependencies=pins,elapsed_seconds=time.perf_counter()-started)
  b.write(out/'manifest.json',result);print(json.dumps(result),flush=True)
 except Exception as e:delta.close();b.write(out/'FAILED.json',dict(error=repr(e),complete=False,requalified_frames=len(replaced)));raise
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--config',required=True);run(a.parse_args().config)
