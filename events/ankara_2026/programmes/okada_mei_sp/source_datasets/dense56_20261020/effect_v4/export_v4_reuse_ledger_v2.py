from pathlib import Path
import json,gzip,hashlib
O=Path(__file__).resolve().parent;T=O/'source_diff_v4'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def ref(p):return dict(file=str(p),sha256=sha(p))
def rows(p):
 with gzip.open(p,'rt',encoding='utf8') as f:
  for l in f:yield json.loads(l)
def key(r):return r['shot_id'],r['native_pts']
def sig(o):
 if isinstance(o,dict):
  if 'file' in o and 'sha256' in o:return o['sha256']
  return {k:sig(v) for k,v in sorted(o.items())}
 if isinstance(o,list):return [sig(v) for v in o]
 return o
def digest(o):return hashlib.sha256(json.dumps(o,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf8')).hexdigest()
vp=O/'generation_v3/manifest.json';np=O/'generation_v4/manifest.json';sp=T/'v3_to_v4_subset.json'
assert sha(vp)=='7ad095fc2aca1072337b13ceb27250a6dc55f9d22f8699d568218674af4cf86a'
assert sha(np)=='1d3f4ac6077f27a86f83e47977d80b7ac89cc95a307fa51a6bdf190a464b59b4'
assert sha(sp)=='8dcb31daf7f51bc45c2cb05acd264a078f16cba17033a334d6ce4cb36dd8a2d7'
v,n,s=read(vp),read(np),read(sp);relevant=('mask','source_unknown_exclusion_mask_ref','graphics_mask','board','anchored_board','actual_surface_override','upstream_board_actor_mask','advert','advert_actual_source_proofs')
sets={field:{key(r) for r in s[field]} for field in ('source_evidence_reuse_frames','exact_entire_overlay_record_equal_frames','actual_effective_source_input_SHA_equal_frames')}
def bindings(m):
 r=m['files']['consumer_bindings.jsonl.gz'];assert sha(r['file'])==r['sha256']
 out={}
 for q in rows(r['file']):
  k=key(q);assert k not in out;out[k]=dict(rgb=q['source_RGB_sha256'],bgr=q['source_BGR_sha256'],inputs=digest({f:sig(q.get(f)) for f in relevant}))
 return out
vb,nb=bindings(v),bindings(n);assert list(vb)==list(nb) and len(vb)==1862
ledger=T/'frame_reuse_ledger_v2.jsonl.gz';assert not ledger.exists()
with gzip.open(ledger,'wt',encoding='utf8',compresslevel=4) as f:
 for k,a in vb.items():
  b=nb[k];assert (a['rgb'],a['bgr'])==(b['rgb'],b['bgr']);eq=a['inputs']==b['inputs'];assert eq==(k in sets['actual_effective_source_input_SHA_equal_frames'])
  f.write(json.dumps(dict(shot_id=k[0],native_pts=k[1],time_base='1/12800',source_RGB_sha256=a['rgb'],source_BGR_sha256=a['bgr'],prior_v3_effective_input_sha256=a['inputs'],final_v4_effective_input_sha256=b['inputs'],entire_overlay_records_exactly_equal=k in sets['exact_entire_overlay_record_equal_frames'],source_evidence_reuse=k in sets['source_evidence_reuse_frames']),separators=(',',':'))+'\n')
def ranges(keys):
 out=[]
 for sid,p in sorted(keys):
  if out and out[-1]['shot_id']==sid and out[-1]['last_native_pts']+256==p:out[-1]['last_native_pts']=p;out[-1]['frames']+=1
  else:out.append(dict(shot_id=sid,first_native_pts=p,last_native_pts=p,frames=1))
 return out
reuse=sets['source_evidence_reuse_frames'];changed=set(vb)-reuse
assert len(changed)+len(reuse)==1862
scope=dict(schema='rinkspace.final_v4_same_source_reuse_scope.v1',complete=True,prior_generation=ref(vp),final_generation=ref(np),actual_full_subset_audit=ref(sp),per_native_frame_SHA_ledger=ref(ledger),ledger_rows=1862,all_native_source_RGB_and_BGR_equal=True,reusable_frames=len(reuse),reusable_ranges=ranges(reuse),new_source_QA_required_frames=len(changed),new_source_QA_required_ranges=ranges(changed),selected_row_changes=s['changed_native_frames'],actual_effective_input_files_rehashed_by_full_subset_audit=s['verified_effective_input_files'],entire_overlay_records_equal_and_effective_input_bytes_equal_required_for_reuse=True,source_image_QA_not_replaced=True,code=ref(__file__))
p=T/'reuse_scope.json';assert not p.exists();p.write_text(json.dumps(scope,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(scope,ensure_ascii=False))
