"""Full development-frame exterior reread with one shared physical operator."""
from pathlib import Path
from collections import Counter
from fractions import Fraction
import argparse,json,gzip,hashlib,time
import av,cv2,numpy as np
from .io_utils import read_json,read_jsonl,write_json,file_sha256,object_sha256
from .role_guard import FrameRoleGuard,DENY_TRAIN
from .measurement_contract import EventMeasurementContract
from .board_outer_source import source_outer_boundary_rows,CONFIG
from .outer_cubic_source import prepare_cubic_channels,CONFIG as CUBIC_CONFIG
from scipy.ndimage import map_coordinates

def outer_batch(project,run,outdir):
 project=Path(project);run=Path(run);out=Path(outdir)
 if out.exists():raise ValueError('New immutable outer-source generation required')
 scope=read_json(run/'programme_scope.json');video=Path(scope['source']['path']);guard=FrameRoleGuard(run/'roles/frame_roles.json');contract=EventMeasurementContract(project,'ankara_2026');data=next(x for x in run.parents if x.name=='RinkSpace_Data')
 if file_sha256(video)!=scope['source']['sha256']:raise ValueError('Official video changed')
 ledgers={int(r['native_pts']):r for n in ('dense_s03_v1','dense_remaining_v1') for r in read_jsonl(run/n/'frame_ledger.jsonl.gz')};pending={int(r['native_pts']) for r in read_jsonl(run/'board_outer_full_v1/frame_receipts.jsonl.gz') if r['source_rows']==0};targets={p:r for p,r in ledgers.items() if r['role'] not in DENY_TRAIN and p in pending};code=Path(__file__).parent;code_refs=[dict(path=str(code/n),sha256=file_sha256(code/n)) for n in ('board_outer_recovery.py','board_outer_source.py','outer_cubic_source.py','measurement_contract.py')];out.mkdir(parents=True);(out/'frame_certificates').mkdir();(out/'chunks').mkdir();write_json(out/'policy.json',dict(configuration=CONFIG,contract_sha256=contract.digest,authority_references=contract.references,actor_exclusion='same-exposure exact original person body boxes + 4px margin; downward reflection exclusion only when actual current board-contact proves the person is foreground on ice; original graphics exclusions; no new detection',native_video_sha256=scope['source']['sha256'],source_pixels_from_model=False,code_references=code_refs));counts=Counter();seen=set();chunkrows=[];profiles=[];puv=[];pn=[];index=[];chunkframes=[];chunks=[];off=np.asarray(CUBIC_CONFIG['normal_offsets']);toff=np.arange(-2,3,dtype=float);start=time.perf_counter()
 def flush():
  nonlocal chunkrows,profiles,puv,pn,index,chunkframes
  if not chunkframes:return
  d=out/'chunks'/f'{len(chunks):04d}';d.mkdir()
  with gzip.open(d/'rows.jsonl.gz','wt',encoding='utf8') as f:
   for r in chunkrows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
  np.savez_compressed(d/'profiles.npz',intensity=np.asarray(profiles,np.float32).reshape(-1,5,len(off)),uv=np.asarray(puv,float).reshape(-1,2),normal=np.asarray(pn,float).reshape(-1,2),channel=np.full(len(profiles),2,np.uint8),normal_offsets=off,tangent_offsets=toff)
  write_json(d/'profile_index.json',index);write_json(d/'frames.json',chunkframes);chunks.append(dict(path=str(d),frames=len(chunkframes),rows=len(chunkrows),files=[dict(path=str(d/n),sha256=file_sha256(d/n)) for n in ('rows.jsonl.gz','profiles.npz','profile_index.json','frames.json')]));chunkrows=[];profiles=[];puv=[];pn=[];index=[];chunkframes=[];write_json(out/'progress.json',dict(completed_frames=len(seen),counts=dict(counts),chunks=chunks))
 with av.open(str(video)) as container,gzip.open(out/'frame_receipts.jsonl.gz','wt',encoding='utf8') as receipts:
  stream=container.streams.video[0]
  if Fraction(stream.time_base)!=Fraction(1,12800):raise ValueError('Official native time base changed')
  for frame in container.decode(stream):
   pts=int(frame.pts)
   if pts not in targets:
    if pts>max(targets):break
    continue
   fr=targets[pts];sid=fr['shot_id'];guard.assert_training_pixels([pts],shot_id=sid);bgr=frame.to_ndarray(format='bgr24');bgrsha=hashlib.sha256(bgr.tobytes()).hexdigest()
   if bgrsha!=fr['decoded_native_BGR_sha256']:raise ValueError('Original current detector exposure differs from actual native RGB')
   rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);rgbsha=hashlib.sha256(rgb.tobytes()).hexdigest();mask=np.zeros(rgb.shape[:2],bool)
   for b in fr.get('current_person_boxes',[]):
    binding=b['source_binding']
    if binding.get('source_pts')!=pts or binding.get('decoded_native_rgb_sha256')!=rgbsha:raise ValueError('Current person exclusion belongs to another exposure')
    q=np.asarray(b['box_xyxy']);a=np.maximum([0,0],np.floor(q[:2]-4).astype(int));end=np.minimum([1920,1080],np.ceil(q[2:]+4).astype(int));mask[a[1]:end[1],a[0]:end[0]]=True
   mask[:170,:620]=True;mask[900:,1600:]=True;fresh,cert=source_outer_boundary_rows(rgb,mask);contacts=[r['current_uv'] for r in fresh if r['material_role']=='board.contact'];reflection_extended=0
   if contacts:
    contact_y=float(np.median(np.asarray(contacts)[:,1]))
    for b in fr.get('current_person_boxes',[]):
     q=np.asarray(b['box_xyxy']);a=np.maximum([0,0],np.floor(q[:2]-4).astype(int));end=np.minimum([1920,1080],np.ceil(q[2:]+4).astype(int))
     if float(q[3])>contact_y+8:
      mask[end[1]:1080,a[0]:end[0]]=True;reflection_extended+=1
    if reflection_extended:fresh,cert=source_outer_boundary_rows(rgb,mask)
   channels=prepare_cubic_channels(rgb);certfile=out/'frame_certificates'/f'{pts}.json';certref=dict(path=str(certfile.resolve().relative_to(data)),role='physical_source_definition_certificate');current=[]
   for j,r in enumerate(fresh):
    role=r['material_role'];operator=r['measurement_operator'];edge='top' if role=='board.top' else 'contact';key='|'.join((role,'board',operator,edge));rid=f'outer_recovery_v2/{sid}/{pts}/{edge}/{j}';family=f'outer_recovery_v2/{sid}/{pts}/{edge}';r.update(shot_id=sid,native_pts=pts,time_base='1/12800',source_row_id=rid,source_family_id=family,joint_material_id=key,point_correspondence=False,anchor_pts=pts,anchor_source_row_id=None,source_acquisition_mode='current_native_user_defined_outer_material_topology',source_identity_basis='current_RGB_blue_upper_yellow_lower_same_board_body',decoded_native_BGR_sha256=bgrsha,current_pixels_measured=True,pixels_from_model=False,numeric_statistics_ready=False,solver_eligible=False,source_material_locally_qualified=True,source_operator_version=CUBIC_CONFIG['version'],role='development',finite_model_support='entire existing continuous board perimeter H/0; no free strip height')
    expected=contract.expected(role,'board_edge');r['measurement_contract_certificate']=dict(contract_sha256=contract.digest,source_row_id=rid,native_pts=pts,joint_material_id=key,operator=operator,physical_role=expected['physical_role'],current_measurement_sha256=object_sha256(dict(native_pts=pts,current_uv=r['current_uv'],source_row_id=rid)),actual_RGB_material_identity_verified=True,operator_selected_correct_material_transition=True,pixels_from_model=False,source_evidence=[certref],actual_source_certificate_index=r['source_definition_certificate_index'])
    n=np.asarray(r['normal']);t=np.array([-n[1],n[0]]);q=np.asarray(r['navigation_seed_uv']);xy=q+toff[:,None,None]*t+off[None,:,None]*n;profile=map_coordinates(channels[2],[xy[:,:,1].ravel(),xy[:,:,0].ravel()],order=3,mode='nearest',prefilter=False).reshape(5,len(off));r['auxiliary_profiles_diagnostic_only']=True;index.append(dict(profile_index=len(profiles),source_row_id=rid,source_family_id=family,shot_id=sid,native_pts=pts,side=0,decoded_native_BGR_sha256=bgrsha));profiles.append(profile);puv.append(q);pn.append(n);current.append(r)
   # The source artifact is self-contained; its SHA is bound into row certs only
   # after all actual source rows are written, avoiding a circular file hash.
   write_json(certfile,dict(shot_id=sid,native_pts=pts,native_video_sha256=scope['source']['sha256'],decoded_native_BGR_sha256=bgrsha,current_exclusion_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),actual_outer_material_certificates=cert,actual_outer_source_rows=[{k:v for k,v in r.items() if k!='measurement_contract_certificate'} for r in current],measurement_contract_sha256=contract.digest,code_references=code_refs))
   certsha=file_sha256(certfile)
   for r in current:r['measurement_contract_certificate']['source_evidence'][0]['sha256']=certsha
   chunkrows.extend(current);seen.add(pts);receipt=dict(shot_id=sid,native_pts=pts,status='actual_user_outer_boundary_source_measured' if current else 'actual_outer_material_order_missing_ambiguous_or_excluded',source_rows=len(current),source_BGR_sha256=bgrsha,source_certificate=str(certfile),source_certificate_sha256=certsha,current_exclusion_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),protected_pixels_consumed=False);chunkframes.append(receipt);receipts.write(json.dumps(receipt,separators=(',',':'))+'\n');counts['frames_with_actual_outer_source' if current else 'source_pending_frames']+=1;counts['actual_outer_rows']+=len(current)
   if len(chunkframes)>=120:flush();print(json.dumps(dict(frames=len(seen),counts=dict(counts))),flush=True)
 flush()
 if seen!=set(targets):raise ValueError('Missing actual development exposure')
 for ref in code_refs:
  if file_sha256(ref['path'])!=ref['sha256']:raise ValueError('Outer source implementation changed within generation')
 contract.verify_unchanged();result=dict(schema='rinkfit.actual_pending_frames_outer_board_recovery.v2',development_frames=len(seen),protected_frames_not_consumed=len(ledgers)-len(targets),counts=dict(counts),chunks=chunks,source_uv_is_first_actual_native_measurement=True,source_uv_from_model=False,contract_sha256=contract.digest,code_references=code_refs,all_development_frames_accounted=False,all_missing_frames_accounted=True,all_gap_frames_accounted=True,original_pending_frames=len(pending),original_complete_source_manifest=dict(path=str(run/'board_outer_full_v1/manifest.json'),sha256=file_sha256(run/'board_outer_full_v1/manifest.json')),numeric_statistics_ready=False,solver_admitted_rows=0,elapsed_seconds=time.perf_counter()-start);write_json(out/'manifest.json',result);return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(outer_batch(a.project,a.run,a.out)))
