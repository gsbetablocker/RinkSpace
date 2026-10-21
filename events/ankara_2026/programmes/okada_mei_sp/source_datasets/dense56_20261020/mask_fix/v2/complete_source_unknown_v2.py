"""Install source-uncertain SAM components omitted by an automatic prompt gate."""
import os
for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from collections import defaultdict,Counter
import json,gzip,hashlib,sys,time
import av,cv2,numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O.parent))
from source_graphics import current_graphics
V=O/'effective_v1';F=O/'effective_v2';policy=Path(sys.argv[1])
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ref(p):return dict(file=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,j):Path(p).write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf8')
def lines(p):
 with gzip.open(p,'rt',encoding='utf8')as f:
  for l in f:
   if l.strip():yield json.loads(l),l
if F.exists():raise FileExistsError(F)
parent=read(V/'s05_manifest.json');assert parent['accepted_for_final_source_qualification']and parent['frames']==1376
pi=parent['index'];assert sha(pi['file'])==pi['sha256']
oldlines=list(lines(pi['file']));bypts={r['native_pts']:r for r,l in oldlines};decisions=[r for r,l in lines(V/'background_source_disposition.jsonl.gz')]
rule=read(policy);assert rule['source_semantics_qualification_complete'];reject={(r['native_pts'],r['object_index'],r['component'])for r in rule['rejected_source_components']}
miss=defaultdict(list)
for d in decisions:
 if d['qualification']!='current_source_semantically_reviewed_true_person_components':continue
 cc=[c for c in d['rejected_components']if c['reason']=='no_current_human_prompt_or_confirmed_person_overlap_source_component_unqualified']
 if cc:assert d['shot_id']=='s05';miss[d['native_pts']].append((d,cc))
assert len(miss)==107 and sum(map(len,miss.values()))==147
source=read(O.parent/'consumer_s05.json')['source'];assert sha(source['file'])==source['sha256']
pins=[ref(__file__),ref(policy),ref(V/'s05_manifest.json'),ref(pi['file']),ref(V/'background_source_disposition.jsonl.gz'),ref(O.parent/'source_graphics.py')]
F.mkdir();(F/'unknown').mkdir();newrows={};receipt=[];counter=Counter();start=time.perf_counter()
with av.open(source['file'])as src:
 st=src.streams.video[0];assert str(st.time_base)=='1/12800';src.seek(min(miss),stream=st,backward=True)
 for fr in src.decode(st):
  p=int(fr.pts)
  if p>max(miss):break
  if p not in miss:continue
  row=bypts[p];raw=fr.to_ndarray(format='bgr24');bh=hashlib.sha256(raw.tobytes()).hexdigest();rh=hashlib.sha256(raw[:,:,::-1].copy().tobytes()).hexdigest();assert bh==row['source_BGR_sha256']and rh==row['source_RGB_sha256']
  gm,_=current_graphics(raw);delta=np.zeros((1080,1920),bool);componentproof=[]
  for d,cc in miss[p]:
   rr=d['raw_source_mask_ref'];assert sha(rr['file'])==rr['sha256'];sam=cv2.imread(rr['file'],0);assert sam is not None and sam.shape==(1080,1920);m=(sam>0)&~gm;n,labels,stats,_=cv2.connectedComponentsWithStats(m.astype('u1'),8)
   for c in cc:
    j=c['component'];assert 0<j<n and stats[j,:4].tolist()==c['bbox_xywh']and int(stats[j,4])==c['pixels']
    key=(p,d['object_index'],j);rej=key in reject;comp=labels==j
    if not rej:delta|=comp
    item=dict(native_pts=p,object_index=d['object_index'],component=j,bbox_xywh=c['bbox_xywh'],pixels=c['pixels'],raw_source_mask_ref=rr,source_BGR_sha256=bh,source_RGB_sha256=rh,source_type='rejected_source_confirmed_nonperson_component'if rej else'source_uncertain_foreground_component_not_scene_observation',reason=next((r['reason']for r in rule['rejected_source_components']if (r['native_pts'],r['object_index'],r['component'])==key),'Semantically reviewed person instance; automatic prompt overlap failure cannot certify that its exact SAM component exposes static scene material.'),unknown_is_not_person_pixel_certificate=True,model_or_pose_pixels_issued=False)
    componentproof.append(item);receipt.append(item);counter['rejected_confirmed_nonperson_components'if rej else'source_components_installed_as_independent_unknown']+=1
  ur=row.get('source_unknown_exclusion_mask_ref');old=np.zeros_like(delta)
  if ur:assert sha(ur['file'])==ur['sha256'];old=cv2.imread(ur['file'],0)>0
  union=old|delta;added=int((union&~old).sum());counter['source_exposures_actually_reread']+=1;counter['new_source_unknown_pixels']+=added
  if added:
   uf=F/'unknown'/f'{p}.png';cv2.imwrite(str(uf),union.astype('u1')*255,[cv2.IMWRITE_PNG_COMPRESSION,3]);nr=dict(row);nr['source_unknown_exclusion_mask_ref']=dict(ref(uf),native_pts=p,shot_id='s05',source_BGR_sha256=bh,source_RGB_sha256=rh,type='preserved_source_unknown_OR_source_uncertain_person_instance_SAM_component',reason='Source-qualified independent exclusion; no new person or static-scene truth asserted.',pixel_count=int(union.sum()),visibility_guard_pixels=8,parent_unknown_exclusion_ref=ur,component_source_proofs=componentproof);nr['additional_source_uncertain_component_layers']=componentproof;nr['parent_effective_v1_index_ref']=pi;nr['added_source_unknown_pixels_this_delta']=added;newrows[p]=nr;counter['frames_with_actual_new_unknown_pixels']+=1
  assert not(old&~union).any()
  if counter['source_exposures_actually_reread']%20==0:print(json.dumps(dict(frames=counter['source_exposures_actually_reread'],changed_frames=len(newrows),elapsed_seconds=time.perf_counter()-start)),flush=True)
assert counter['source_exposures_actually_reread']==107
known={(r['native_pts'],r['object_index'],r['component'])for r in receipt};assert reject<=known
qual=dict(complete=True,source_unknown_not_true_person_certification=True,parent_qualification=parent['source_qualification'],component_policy=ref(policy),source_instances_semantically_reviewed=True,no_new_SAM_or_model_inference=True,all_current_source_RGB_for_changed_candidates_verified=True,all_source_components_reconstructed_from_exact_frozen_SAM_bytes=True,source_component_count=len(receipt),counts=dict(counter),old_true_person_and_unknown_preserved=True,source_original_UV_unchanged=True)
qf=F/'source_qualification.json';write(qf,qual)
ip=F/'s05_index.jsonl.gz'
with gzip.open(ip,'wt',encoding='utf8',compresslevel=4)as w:
 for row,line in oldlines:
  p=row['native_pts']
  if p in newrows:newrows[p]['source_qualification_ref']=ref(qf);w.write(json.dumps(newrows[p],separators=(',',':'))+'\n')
  else:w.write(line)
rf=F/'component_dispositions.json';write(rf,dict(complete=True,components=receipt,actual_changed_native_pts=sorted(newrows),counts=dict(counter),frozen_parent_inputs=pins,all_old_person_and_unknown_preserved=True,source_UV_not_changed=True))
for r in pins:assert sha(r['file'])==r['sha256']
m=dict(parent);m.update(schema='rinkspace.current_native_source_person_and_typed_unknown_effective.v4',index=ref(ip),index_file=str(ip),index_sha256=sha(ip),source_reader=ref(__file__),source_qualification=ref(qf),parent_effective_generation=ref(V/'s05_manifest.json'),parent_effective_index=pi,source_component_delta=ref(rf),source_component_delta_counts=dict(counter),actual_changed_native_pts=sorted(newrows),unaffected_s06_generation=ref(V/'s06_manifest.json'),all_source_components_closure_complete=True,complete=True,sealed=True,accepted=True,accepted_for_final_source_qualification=True)
m['counts']=dict(parent['counts']);m['counts']['added_source_unknown_pixels']+=counter['new_source_unknown_pixels'];m['counts']['frames_with_any_source_unknown']=sum(bool((newrows.get(r['native_pts'])or r).get('source_unknown_exclusion_mask_ref'))for r,l in oldlines)
write(F/'s05_manifest.json',m);print(json.dumps(dict(complete=True,manifest=ref(F/'s05_manifest.json'),index=ref(ip),counts=dict(counter),changed_native_pts=sorted(newrows)),ensure_ascii=False),flush=True)
