"""Reuse completed physical outer selections; reread only their frozen means."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,json,gzip,hashlib
import av,cv2,numpy as np
from scipy.ndimage import map_coordinates
from .io_utils import read_json,read_jsonl,write_json,file_sha256,object_sha256
from .role_guard import FrameRoleGuard
from .measurement_contract import EventMeasurementContract
from .outer_cubic_source import prepare_cubic_channels,acquire_outer_cubic,CONFIG

def revise(project,run,parent_manifest,outdir,skip_chunks=0):
 project=Path(project);run=Path(run);parent_manifest=Path(parent_manifest);out=Path(outdir)
 if out.exists():raise ValueError('New immutable continuous outer-source generation required')
 parent=read_json(parent_manifest);chunks=parent['chunks'][skip_chunks:];bypts=defaultdict(list);originals=[]
 for chunk in chunks:
  files={Path(f['path']).name:Path(f['path']) for f in chunk['files']}
  for f in chunk['files']:
   if file_sha256(f['path'])!=f['sha256']:raise ValueError('Completed original physical source chunk changed')
  for r in read_jsonl(files['rows.jsonl.gz']):bypts[int(r['native_pts'])].append(r)
  originals.append(dict(files=chunk['files'],source_frames=chunk['frames']))
 guard=FrameRoleGuard(run/'roles/frame_roles.json');scope=read_json(run/'programme_scope.json');video=Path(scope['source']['path']);data=next(x for x in run.parents if x.name=='RinkSpace_Data');contract=EventMeasurementContract(project,'ankara_2026')
 if file_sha256(video)!=scope['source']['sha256']:raise ValueError('Official native video changed')
 out.mkdir(parents=True);(out/'frame_certificates').mkdir();(out/'chunks').mkdir();(out/'code').mkdir();code=Path(__file__).parent;refs=[]
 for name in ('outer_cubic_revision.py','outer_cubic_source.py','source_worker.py'):
  src=code/name;dst=out/'code'/name;dst.write_bytes(src.read_bytes());refs.append(dict(path=str(dst),sha256=file_sha256(dst),original_path=str(src)))
 write_json(out/'policy.json',dict(schema='rinkfit.current_outer_continuous_cubic_revision.v1',parent_manifest=dict(path=str(parent_manifest),sha256=file_sha256(parent_manifest)),parent_original_chunk_range=[skip_chunks,skip_chunks+len(chunks)],source_material_identity_and_color_topology_reused_without_redetection=True,original_source_uv_preserved=True,old_bilinear_W_reused=False,configuration=CONFIG,source_measurement_contract_sha256=contract.digest,code_snapshots=refs));counts=Counter();seen=set();newchunks=[];rows=[];profiles=[];index=[];puv=[];pn=[];frows=[];off=np.asarray(CONFIG['normal_offsets']);toff=np.arange(-2,3,dtype=float)
 def flush():
  nonlocal rows,profiles,index,puv,pn,frows
  if not frows:return
  d=out/'chunks'/f'{len(newchunks):04d}';d.mkdir()
  with gzip.open(d/'rows.jsonl.gz','wt',encoding='utf8') as f:
   for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
  np.savez_compressed(d/'profiles.npz',intensity=np.asarray(profiles,np.float32).reshape(-1,5,len(off)),uv=np.asarray(puv,float).reshape(-1,2),normal=np.asarray(pn,float).reshape(-1,2),channel=np.full(len(profiles),2,np.uint8),normal_offsets=off,tangent_offsets=toff);write_json(d/'profile_index.json',index);write_json(d/'frames.json',frows);newchunks.append(dict(frames=len(frows),rows=len(rows),files=[dict(path=str(d/n),sha256=file_sha256(d/n)) for n in ('rows.jsonl.gz','profiles.npz','profile_index.json','frames.json')]));rows=[];profiles=[];index=[];puv=[];pn=[];frows=[];write_json(out/'progress.json',dict(completed_source_frames=len(seen),counts=dict(counts),chunks=newchunks))
 if not bypts:raise ValueError('No original current source rows in requested completed chunk range')
 with av.open(str(video)) as container,gzip.open(out/'frame_receipts.jsonl.gz','wt',encoding='utf8') as receipts:
  stream=container.streams.video[0]
  for frame in container.decode(stream):
   pts=int(frame.pts)
   if pts not in bypts:
    if pts>max(bypts):break
    continue
   oldrows=bypts[pts];sid=oldrows[0]['shot_id'];guard.assert_training_pixels([pts],shot_id=sid);bgr=frame.to_ndarray(format='bgr24');sha=hashlib.sha256(bgr.tobytes()).hexdigest()
   if any(r['decoded_native_BGR_sha256']!=sha for r in oldrows):raise ValueError('Actual current pixels differ from frozen material selection')
   rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);coeff=prepare_cubic_channels(rgb);new=[];qualified=[];parent_proofs=[];sourcecert=None
   for original in oldrows:
    r=dict(original);a=acquire_outer_cubic(rgb,original,coefficients=coeff);r['previous_bilinear_current_uv']=original['current_uv'];r['previous_bilinear_source_measurement_audit']=original['source_measurement_audit'];r['previous_physical_source_certificate']=original['measurement_contract_certificate'];r['source_operator_version']=CONFIG['version'];r['current_uv']=a['current_uv'] if a['base_source_material_qualified'] else None;r['source_material_locally_qualified']=a['base_source_material_qualified'];r['source_measurement_audit']={**original['source_measurement_audit'],'status':a['status'],'current_uv':r['current_uv'],'continuous_cubic_material_profile':a,'original_material_selection_seed_uv':original['navigation_seed_uv'],'profile':dict(channel=2,polarity=original['source_measurement_audit']['profile']['polarity'],normal_offsets=CONFIG['normal_offsets'],interpolation='cubic',peak_location_rule=CONFIG['location'])};r['numeric_statistics_ready']=False;r['solver_eligible']=False;r['auxiliary_profiles_diagnostic_only']=True;r['old_bilinear_W_reused']=False
    if r['current_uv'] is not None:
     n=np.asarray(r['normal']);t=np.array([-n[1],n[0]]);q=np.asarray(r['navigation_seed_uv']);xy=q+toff[:,None,None]*t+off[None,:,None]*n;p=map_coordinates(coeff[2],[xy[:,:,1].ravel(),xy[:,:,0].ravel()],order=3,mode='nearest',prefilter=False).reshape(5,len(off));index.append(dict(profile_index=len(profiles),source_row_id=r['source_row_id'],source_family_id=r['source_family_id'],shot_id=sid,native_pts=pts,side=0,decoded_native_BGR_sha256=sha));profiles.append(p);puv.append(q);pn.append(n);qualified.append(r);counts['actual_continuous_cubic_mean_rows']+=1
    else:counts['actual_same_material_or_window_pending_rows']+=1
    counts['source_rows']+=1;counts['base_flat_peak_interval_rows']+=int(bool(a['flat_peak_intervals']));counts['base_window_boundary_rows']+=a['synthetic_window_boundary_count'];new.append(r)
    for ref in original['measurement_contract_certificate']['source_evidence']:
     if ref not in parent_proofs:parent_proofs.append(ref)
   if parent_proofs:
    first=data/parent_proofs[0]['path']
    if file_sha256(first)!=parent_proofs[0]['sha256']:raise ValueError('Original physical material source certificate changed')
    sourcecert=read_json(first)
   path=out/'frame_certificates'/f'{pts}.json';write_json(path,dict(native_pts=pts,shot_id=sid,measurement_contract_sha256=contract.digest,native_video_sha256=scope['source']['sha256'],decoded_native_BGR_sha256=sha,original_physical_source_evidence=parent_proofs,actual_outer_material_certificates=sourcecert['actual_outer_material_certificates'],actual_outer_source_rows=[{k:v for k,v in r.items() if k!='measurement_contract_certificate'} for r in qualified],localization_code_snapshots=refs,source_color_identity_redetected=False,new_localization_configuration=CONFIG));proof=dict(path=str(path.resolve().relative_to(data)),sha256=file_sha256(path),role='physical_source_definition_certificate')
   for r in new:
    cert=dict(r['measurement_contract_certificate']);cert['current_measurement_sha256']=object_sha256(dict(native_pts=pts,current_uv=r['current_uv'],source_row_id=r['source_row_id']));cert['source_evidence']=[proof];cert['operator_selected_correct_material_transition']=bool(r['source_material_locally_qualified']);cert['actual_RGB_material_identity_verified']=bool(r['source_material_locally_qualified']);r['measurement_contract_certificate']=cert
   rows.extend(new);seen.add(pts);receipt=dict(native_pts=pts,shot_id=sid,source_rows=len(new),source_qualified_rows=len(qualified),source_BGR_sha256=sha,source_certificate=str(path),source_certificate_sha256=proof['sha256'],material_redetection_performed=False);frows.append(receipt);receipts.write(json.dumps(receipt,separators=(',',':'))+'\n')
   if len(frows)>=60:flush();print(json.dumps(dict(source_frames=len(seen),counts=dict(counts))),flush=True)
 flush()
 if seen!=set(bypts):raise ValueError('Frozen source means not all replayed')
 contract.verify_unchanged();result=dict(schema='rinkfit.actual_outer_continuous_cubic_source_generation.v1',parent_manifest=dict(path=str(parent_manifest),sha256=file_sha256(parent_manifest)),parent_chunk_range=[skip_chunks,skip_chunks+len(chunks)],parent_completed_scope_frames=sum(c['frames'] for c in chunks),actual_source_frames_relocalized=len(seen),counts=dict(counts),chunks=newchunks,all_gap_frames_accounted=True,all_original_source_rows_accounted=counts['source_rows']==sum(len(v) for v in bypts.values()),source_uv_preserved=True,physical_material_selection_not_redetected=True,old_bilinear_W_reused=False,new_source_requires_same_cubic_W=True,protected_pixels_consumed=False,code_snapshots=refs);write_json(out/'manifest.json',result);return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--run',required=True);p.add_argument('--parent-manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--skip-chunks',type=int,default=0);a=p.parse_args();print(json.dumps(revise(a.project,a.run,a.parent_manifest,a.out,a.skip_chunks)))
