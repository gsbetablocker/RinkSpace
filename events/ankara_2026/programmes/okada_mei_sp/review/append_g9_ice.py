import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
from pathlib import Path
import sys,json,time,numpy as np
root=Path(r'D:/OneDrive/codex_projs/RinkSpace');sys.path.insert(0,str(root/'calibration/rinkfit/src'))
from rinkfit_joint.reference_source_append import append_constraint_source_chunks
from rinkfit_joint.qualified_bundle_append import verify_qualified_bundle,rebind_qualified_bundle_collection,COLLECTION_SCHEMA,BUNDLE_SCHEMAS
# This named protocol is issued by the independently executed frozen finite-paint gate.
BUNDLE_SCHEMAS.add('rinkfit.native_finite_paint_midline_qualified_bundle.v1')
from rinkfit_joint.io_utils import file_sha256,write_json,read_jsonl
rec=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019')
parent=rec/'g8_ice/packet.json';out=rec/'g9_ice';collection_dir=rec/'g9_sources'
if out.exists() or collection_dir.exists():raise ValueError('Use a new immutable source generation')
collection_dir.mkdir()
def ref(p):return dict(file=str(Path(p).resolve()),sha256=file_sha256(p))
names=['hist_short_v1/ring_history_qualified','hist_s09_v1/qualified_ring','hist_s09_v1/qualified_C0_chain_v2','hist_s07_v1/chain_qualified_v3']
while not all((rec/n/'manifest.json').exists()for n in names):time.sleep(1)
bundlerefs=[ref(rec/n/'manifest.json')for n in names]
bundles=[verify_qualified_bundle(r)for r in bundlerefs]
mapping={};byshot={};roles=None;source=None
for b in bundles:
 a=json.loads(Path(b['acquisition']['file']).read_text(encoding='utf8'))
 if roles is not None and a['source_roles']!=roles:raise ValueError('Mixed source exposure role registries')
 roles=a['source_roles']
 if source is not None and a['source_video']!=source:raise ValueError('Mixed official source or native clock')
 source=a['source_video']
 for key,value in json.loads(Path(b['material_map']['file']).read_text())['semantic_to_material'].items():
  if key in mapping and mapping[key]!=value:raise ValueError('Conflicting finite physical material identity')
  mapping[key]=value
 for sid,n in b['by_shot'].items():byshot[sid]=byshot.get(sid,0)+n
write_json(collection_dir/'material_map.json',dict(semantic_to_material=mapping))
collection=dict(schema=COLLECTION_SCHEMA,qualified_source_bundles=bundlerefs,model=bundles[0]['model'],source_video=source,source_roles=roles,
 closed_chunks=[r for b in bundles for r in b['closed_chunks']],material_map=ref(collection_dir/'material_map.json'),
 qualified_source_rows=sum(byshot.values()),by_shot=byshot,source_ledger_preserved=True,
 source_qualification_issued_by_collection=False,old_raw_source_UV_or_membership_changed=False,new_statistical_precision_invented=False,finite_paint_protocol_gate=ref(root/'calibration/rinkfit/src/rinkfit_joint/finite_paint_gate.py'),collection_compiler=ref(Path(__file__)))
write_json(collection_dir/'manifest.json',collection)
verify_qualified_bundle(ref(collection_dir/'manifest.json'))
passed=ref(rec/'g8_ice/native_qualification.json')
start=time.monotonic()
def finalize(packet,folder,old):
 n=old['row_count']
 for a,b in zip(old['groups'],packet['groups']):
  for key in ('rows','rank','statistic_kind'):assert a.get(key)==b.get(key)
  if isinstance(a['W'],dict):assert all(a['W'].get(k)==b['W'].get(k)for k in ('sha256','shape','dtype','offset_bytes'))
  else:assert a['W']==b['W']
 for g in packet['groups'][len(old['groups']):]:assert g['rank']==0 and g['W']==[] and g['unknown_noise_source_refs']
 result=rebind_qualified_bundle_collection(parent,packet,folder,ref(collection_dir/'manifest.json'),parent_document=old,native_parent_qualification_ref=passed)
 def arr(k):
  d=result['arrays'][k];return np.memmap(d['file'],dtype=d['dtype'],mode='r',offset=int(d.get('offset_bytes',0)),shape=tuple(d['shape']))
 ad,role,kind,shot,mat,uv,pts=(arr(k)for k in ('admission','fit_role','kind','shot_index','material_index','uv','native_pts'))
 before={s:int(((ad[:n]==1)&(role[:n]==1)&(shot[:n]==i)).sum())for i,s in enumerate(packet['shot_ids'])}
 after={s:int(((ad==1)&(role==1)&(shot==i)).sum())for i,s in enumerate(packet['shot_ids'])}
 delta={s:after[s]-before[s]for s in before}
 assert delta=={s:byshot.get(s,0)for s in before}
 assert np.all(kind[n:]==0)
 aliases=[];nearest=[]
 for i in np.flatnonzero((ad==1)&(role==1)&(np.arange(len(ad))>=n)):
  prior=np.flatnonzero((ad[:i]==1)&(role[:i]==1)&(shot[:i]==shot[i])&(mat[:i]==mat[i])&(pts[:i]==pts[i]))
  if not len(prior):continue
  dist=np.linalg.norm(uv[prior]-uv[i],axis=1);j=int(np.argmin(dist))
  nearest.append(dict(row=int(i),prior=int(prior[j]),distance=float(dist[j])))
  if dist[j]<.25:aliases.append(nearest[-1])
 write_json(folder/'alias_QA.json',dict(passed=not aliases,unresolved_quarter_pixel_remeasurement_aliases=aliases,nearest_same_material_same_exposure=nearest,old_rows_removed=False))
 if aliases:raise ValueError('Actual same-exposure remeasurement aliases must be closed before production append')
 write_json(folder/'prefix_QA.json',dict(passed=True,old_raw_rows=n,new_raw_rows=packet['row_count'],old_arrays_IDs_UV_admission_four_aux_byte_identical=True,
  old_W_Query_partition_unchanged=True,old_active_by_shot=before,new_active_by_shot=after,qualified_delta_by_shot=delta,new_statistical_rank=0,
  circle_material_path_kind_is_not_fixed_entity_observation_kind=True,collection_ref=ref(collection_dir/'manifest.json')))
 return result
report=append_constraint_source_chunks(parent,collection['closed_chunks'],collection['model']['file'],collection['material_map']['file'],roles['file'],out,finalize_packet_with_parent=finalize)
write_json(out/'append_execution.json',dict(actual_pid=os.getpid(),elapsed_seconds=time.monotonic()-start,report=report,no_MC=True,no_fit=True))
print(json.dumps(report,ensure_ascii=False),flush=True)
