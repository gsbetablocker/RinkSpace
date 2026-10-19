"""Read each development exposure at native PTS; flow is navigation only.

Current RGB profiles, person exclusions and source anchor identities remain
separate from final solver admission. No model residual can reject a source.
Protected exposures are not consumed, and tracking resets at every role gap.
"""
from pathlib import Path
from collections import defaultdict,Counter
from fractions import Fraction
import argparse,gzip,hashlib,json,time,sys
import av,cv2,numpy as np
from .io_utils import read_json,read_jsonl,write_json,file_sha256
from .role_guard import FrameRoleGuard,DENY_TRAIN
from .measurement_contract import EventMeasurementContract
from .physical_source_operator import prepare_current_source,remeasure_physical_source,_sample as semantic_sample,CONFIG as SEMANTIC_OPERATOR_CONFIG

def _write(f,x):f.write(json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
def _channels(rgb):
 a=rgb.astype(np.float32);c=np.stack((.2126*a[:,:,0]+.7152*a[:,:,1]+.0722*a[:,:,2],a[:,:,0]-a[:,:,1],a[:,:,2]-.5*(a[:,:,0]+a[:,:,1])))
 return np.stack([cv2.GaussianBlur(x,(7,7),.8,borderType=cv2.BORDER_REFLECT) for x in c])
def _sample_patches(gray,uv,r=6):
 offsets=np.asarray([(x,y) for y in range(-r,r+1) for x in range(-r,r+1)],np.float32)
 q=np.asarray(uv,np.float32)[:,None,:]+offsets
 return cv2.remap(gray,q[:,:,0],q[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
def _ncc(a,b):
 a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True)
 return np.sum(a*b,1)/np.maximum(np.sqrt(np.sum(a*a,1)*np.sum(b*b,1)),1e-8)
def _seed_rows(source,decisions,material_map,*,project,audit_file=None):
 contract=EventMeasurementContract(project,'ankara_2026');final={r['source_row_id']:r for r in read_jsonl(decisions)};rejected=[];reasons=Counter();scanned=0
 geometry={k:v for k,v in read_json(material_map)['semantic_to_material'].items() if v.get('geometry_ready')}
 by=defaultdict(list)
 for r in read_jsonl(source):
  scanned+=1;errors=contract.seed_errors(r,final.get(r['source_row_id'],{}))
  if r.get('joint_material_id') not in geometry:errors.append('seed_shared_finite_geometry_not_ready')
  if r.get('current_uv') is None:errors.append('seed_actual_current_UV_unavailable')
  if errors:
   rejected.append(dict(source_row_id=r['source_row_id'],shot_id=r['shot_id'],native_pts=r['native_pts'],target_id=r.get('target_id'),reasons=sorted(set(errors))));reasons.update(set(errors));continue
  m=r['source_measurement_audit'];n=m.get('normal');point=bool(r.get('point_correspondence'))
  if not point and (n is None or not np.isfinite(n).all()):continue
  uv=np.asarray(r['current_uv'],float);sides=m.get('flank_uv');operator=r['measurement_operator_resolved']
  if operator=='paint_center' and (sides is None or len(sides)!=2):continue
  by[(r['shot_id'],int(r['native_pts']))].append(dict(source_row_id=r['source_row_id'],source_family_id=r['source_family_id'],joint_material_id=r['joint_material_id'],target_id=r['target_id'],material_index=geometry[r['joint_material_id']]['material_index'],uv=uv.tolist(),normal=n,flank_uv=sides,operator=operator,entity_point=point,anchor_pts=int(r['native_pts']),identity_source='final source admission plus current material/operator contract',category=r.get('category'),typed_material_descriptor=r.get('typed_material_descriptor'),anchor_measurement_contract_certificate=r.get('measurement_contract_certificate')))
 # Only identical seed identities at identical UV are aliases, never extra information.
 for key,rows in by.items():
  seen=set();out=[]
  for r in rows:
   tag=(r['joint_material_id'],tuple(np.round(r['uv'],7)))
   if tag not in seen:seen.add(tag);out.append(r)
  by[key]=out
 contract.verify_unchanged()
 if audit_file:
  audit_file=Path(audit_file)
  if audit_file.exists():raise ValueError('A new immutable seed-gate audit generation is required')
  with gzip.open(str(audit_file)+'.rows.jsonl.gz','wt',encoding='utf8') as f:
   for r in rejected:_write(f,r)
  write_json(audit_file,dict(schema='rinkfit.joint9.source_seed_contract_gate.v1',scanned_rows=scanned,accepted_seed_rows=sum(len(x) for x in by.values()),rejected_rows=len(rejected),reason_counts=dict(reasons),measurement_contract_sha256=contract.digest,existing_fact_references=contract.references,final_admission=dict(path=str(decisions),sha256=file_sha256(decisions)),source_rows=dict(path=str(source),sha256=file_sha256(source)),actual_RGB_remeasurement_performed_by_this_gate=False,definition_match_alone_can_admit_source=False))
 return by

def measure(run_dir,project,*,out_dir,source,decisions,material_map,shot_ids=None):
 started=time.perf_counter();run=Path(run_dir);project=Path(project);out=Path(out_dir)
 if out.exists():raise ValueError('A new dense generation is required')
 scope=read_json(run/'programme_scope.json');roles=FrameRoleGuard(run/'roles/frame_roles.json');video=Path(scope['source']['path'])
 if file_sha256(video)!=scope['source']['sha256']:raise ValueError('Official programme source changed')
 seeds=_seed_rows(source,decisions,material_map,project=project,audit_file=out.with_name(out.name+'.seed_gate.json'));
 if not seeds:raise ValueError('No seed passed current physical source semantics and final admission; actual remeasurement and re-promotion required')
 shots={s['id']:s for s in scope['shots'] if s['id']!='s01' and (shot_ids is None or s['id'] in shot_ids)}
 expected={p:s['id'] for s in shots.values() for p in range(s['first_pts'],s['last_pts']+1,256)}
 sys.path[:0]=[str(project),str(project/'src')]
 from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
 add_geometry_dependencies()
 import torch
 from rinkspace.vision_instances import SourcePersonBoxes,Sam2SourceInstances
 torch.set_num_threads(2);cv2.setNumThreads(1)
 out.mkdir(parents=True);(out/'chunks').mkdir()
 detector=SourcePersonBoxes(project.parent/'FSpathing/models/yolo11n-pose.pt',dependency_paths=[project.parent/'FSpathing/.venv/Lib/site-packages'],runtime_directory=out/'detector_runtime',device='cuda')
 provider=Sam2SourceInstances(project.parent/'MotionTrail_Codex_Starter_v1/models/sam2.1-hiera-tiny',device='cuda',reread_rgb=False,contour_stride=16)
 config=dict(schema='rinkfit.dense_native_policy.v2',source_operator=SEMANTIC_OPERATOR_CONFIG,old_absolute_gradient_reader_enabled=False,profile_artifact_role='diagnostic actual frozen-material cubic profiles; full source operator receipts stored per row; not formal noise W',source_video_sha256=scope['source']['sha256'],time_base='1/12800',native_size=[1920,1080],flow='bidirectional pyramidal LK, navigation only',flow_forward_backward_limit_px=1.,minimum_current_anchor_patch_ncc=.75,maximum_one_way_anchor_age_seconds=.6,person_exclusion='independent current YOLO boxes union current SAM2 foreground masks, conservative 12px exterior',maximum_model_residual_input=None,protected_source_pixels_used=False,seed_source=dict(path=str(source),sha256=file_sha256(source)),identity_decisions=dict(path=str(decisions),sha256=file_sha256(decisions)),material_map=dict(path=str(material_map),sha256=file_sha256(material_map)),numeric_W_ready=False,solver_admission=False)
 write_json(out/'policy.json',config);counts=Counter();shotcounts=defaultdict(Counter);chunks=[];frames=[];dense_rows=[];prof=[];puv=[];pn=[];pch=[];pindex=[];processed=set();last_pts=None;sidprev=None;active=[];prev=None;anchor_patches=None
 normal_off=np.arange(-3,3.0001,.25);tangent_off=np.arange(-2,3,dtype=float)
 def flush():
  nonlocal dense_rows,prof,puv,pn,pch,pindex,frames
  if not frames:return
  n=len(chunks);folder=out/'chunks'/f'{n:04d}';folder.mkdir()
  with gzip.open(folder/'rows.jsonl.gz','wt',encoding='utf8') as f:
   for r in dense_rows:_write(f,r)
  np.savez_compressed(folder/'profiles.npz',intensity=np.asarray(prof,np.float32).reshape(-1,5,25),uv=np.asarray(puv,float).reshape(-1,2),normal=np.asarray(pn,float).reshape(-1,2),channel=np.asarray(pch,np.uint8),normal_offsets=normal_off,tangent_offsets=tangent_off)
  write_json(folder/'frames.json',frames);write_json(folder/'profile_index.json',pindex)
  desc=dict(chunk=n,first_pts=frames[0]['native_pts'],last_pts=frames[-1]['native_pts'],frames=len(frames),source_rows=len(dense_rows),actual_profiles=len(prof),files=[dict(path=str(folder/x),sha256=file_sha256(folder/x)) for x in ('rows.jsonl.gz','profiles.npz','frames.json','profile_index.json')]);chunks.append(desc);write_json(out/'progress.json',dict(chunks=chunks,counts=dict(counts),shot_counts={k:dict(v) for k,v in shotcounts.items()},elapsed_seconds=time.perf_counter()-started));dense_rows=[];prof=[];puv=[];pn=[];pch=[];pindex=[];frames=[]
 with av.open(str(video)) as container,gzip.open(out/'frame_ledger.jsonl.gz','wt',encoding='utf8') as ledger:
  stream=container.streams.video[0]
  if Fraction(stream.time_base)!=Fraction(1,12800):raise ValueError('Native PTS time base changed')
  for frame in container.decode(stream):
   pts=int(frame.pts)
   if pts not in expected:
    if pts>max(expected):break
    continue
   sid=expected[pts];processed.add(pts);role=roles.roles[pts]['role'];s=shots[sid];idx=s['programme_frames'][0]+(pts-s['first_pts'])//256
   base=dict(shot_id=sid,rig_id=s['rig_id'],native_pts=pts,programme_frame_index_1based=idx,time_base='1/12800',role=role)
   if role in DENY_TRAIN:
    record={**base,'status':'protected_deferred_until_model_frozen','current_rgb_consumed_for_training':False,'tracked_through_for_training':False,'source_rows':0};_write(ledger,record);counts['protected_deferred']+=1;shotcounts[sid]['protected_deferred']+=1;prev=None;active=[];last_pts=None;sidprev=None;continue
   roles.assert_training_pixels([pts],shot_id=sid)
   bgr=frame.to_ndarray(format='bgr24');rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);digest=hashlib.sha256(bgr.tobytes()).hexdigest();gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
   binding=dict(source_sha256=scope['source']['sha256'],source_pts=pts,time_base='1/12800',native_hw=[1080,1920],shot_id=sid,decoded_native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
   boxes=detector.read(rgb,binding);mask=np.zeros((1080,1920),np.uint8)
   prompts=[]
   for i,box in enumerate(boxes):
    q=np.asarray(box['box_xyxy'],float);a=np.maximum([0,0],np.floor(q[:2]-12).astype(int));b=np.minimum([1920,1080],np.ceil(q[2:]+12).astype(int));mask[a[1]:b[1],a[0]:b[0]]=1
    if q[3]>500 and q[3]-q[1]>100:prompts.append(dict(object_id=i+1,kind='occluder',origin='original_rgb_candidate',source_binding=binding,box_xyxy=q.tolist(),exclusion_authority=None))
   if prompts:
    current=provider.read(rgb,binding,prompts=prompts,reset=True);sam=current['occluder_region_candidate_mask'].detach().cpu().numpy().astype(np.uint8);mask|=cv2.dilate(sam,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))
   # Broadcast graphics remain outside the static physical scene. This fixed
   # conservative screen region is an exclusion, not a material boundary.
   mask[0:170,0:620]=1
   rejected=Counter();candidate_count=0
   if sid!=sidprev or last_pts is None or pts-last_pts!=256:active=[];prev=None
   if active and prev is not None:
    old=np.asarray([x['uv'] for x in active],np.float32).reshape(-1,1,2)
    nxt,st,err=cv2.calcOpticalFlowPyrLK(prev,gray,old,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    back,bs,be=cv2.calcOpticalFlowPyrLK(gray,prev,nxt,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01));fb=np.linalg.norm(back[:,0]-old[:,0],axis=1);corr=_ncc(np.stack([x['anchor_patch'] for x in active]),_sample_patches(gray,nxt[:,0]))
    keep=[]
    for i,x in enumerate(active):
     q=nxt[i,0];cause=None
     if not(st[i,0] and bs[i,0]) or not np.isfinite(q).all() or fb[i]>1.:cause='flow_navigation_failed'
     elif not (8<q[0]<1911 and 8<q[1]<1071):cause='source_outside_native_profile_domain'
     elif mask[int(round(q[1])),int(round(q[0]))]:cause='current_person_or_overlay_exclusion'
     elif corr[i]<.75:cause='current_anchor_appearance_identity_not_supported'
     elif (pts-x['anchor_pts'])/12800>.6:cause='one_way_source_identity_requires_new_anchor'
     if cause:rejected[cause]+=1;continue
     x=dict(x,uv=q.astype(float).tolist(),flow_fb_px=float(fb[i]),anchor_patch_ncc=float(corr[i]));keep.append(x)
    active=keep
   currentseeds=seeds.get((sid,pts),[])
   if currentseeds:
    active=[]
    patches=_sample_patches(gray,[x['uv'] for x in currentseeds])
    for x,patch in zip(currentseeds,patches):
     u,v=np.rint(x['uv']).astype(int)
     if not(8<u<1911 and 8<v<1071) or mask[v,u]:rejected['anchor_excluded_current_person_or_edge']+=1;continue
     active.append(dict(x,anchor_patch=patch,flank_center=x['uv'],flow_fb_px=0.,anchor_patch_ncc=1.))
   source_context=prepare_current_source(rgb,mask,project);measured=[]
   for x in active:
    candidate_count+=1;q=np.asarray(x['uv'],float);sides=x.get('flank_uv')
    if sides is not None:
     # Flow displaces both previously measured material flanks equally only
     # as navigation; each current flank is independently re-read below.
     sides=np.asarray(sides)+(q-np.asarray(x.get('flank_center',q)))
    if x['entity_point']:
     corner=cv2.cornerSubPix(gray,q.astype(np.float32).reshape(1,1,2),(4,4),(-1,-1),(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.001))[0,0]
     if np.linalg.norm(corner-q)>3.:rejected['current_corner_source_refinement_outside_local_window']+=1;continue
     q=corner.astype(float)
    rr=remeasure_physical_source(source_context,x,q,normal=x.get('normal'),flanks=sides)
    if x['entity_point']:rr['cornerSubPix_used_for_navigation_only']=True
    if rr.get('current_uv') is None:rejected[rr['status']]+=1;continue
    uv=np.asarray(rr['current_uv']);u,v=np.rint(uv).astype(int)
    if not(8<u<1911 and 8<v<1071) or mask[v,u]:rejected['remeasured_source_excluded_current_person_or_edge']+=1;continue
    rowid=f'dense/{sid}/{pts}/{x["source_row_id"]}';row={**base,'source_row_id':rowid,'anchor_source_row_id':x['source_row_id'],'anchor_pts':x['anchor_pts'],'source_family_id':f'{sid}/{pts}/{x["source_family_id"]}','joint_material_id':x['joint_material_id'],'material_index':x['material_index'],'target_id':x['target_id'],'measurement_operator':rr['measurement_operator'],'measurement_contract_sha256':rr['measurement_contract_sha256'],'source_material_operator_version':SEMANTIC_OPERATOR_CONFIG['version'],'point_correspondence':x['entity_point'],'source_uv':uv.tolist(),'current_uv':uv.tolist(),'navigation_seed_uv':q.tolist(),'navigation_flank_seed_uv':None if sides is None else np.asarray(sides).tolist(),'source_measurement_audit':rr,'flow_fb_px':x['flow_fb_px'],'anchor_patch_ncc':x['anchor_patch_ncc'],'decoded_native_BGR_sha256':digest,'current_pixels_measured':True,'pixels_from_model':False,'numeric_statistics_ready':False,'solver_eligible':False,'pending_reason':'current-family numeric W and cross-frame identity closure required'}
    dense_rows.append(row);x['uv']=uv.tolist();x['flank_uv']=rr.get('flank_uv',x.get('flank_uv'));x['flank_center']=uv.tolist();measured.append(x)
    n=rr.get('normal')
    if n is not None:
     n=np.asarray(n,float);t=np.array([-n[1],n[0]])
     for side,point in enumerate(rr.get('flank_uv') or [uv.tolist()]):
      coords=np.asarray(point)+tangent_off[:,None,None]*t+normal_off[None,:,None]*n;allc=semantic_sample(source_context['spline_channels'],coords.reshape(-1,2)).reshape(5,25,3);ch=(rr.get('source_material_receipt') or {}).get('channel',(rr.get('profile') or {}).get('channel'))
      if ch is None:continue
      ch=int(ch);pi=len(prof);prof.append(allc[:,:,ch]);puv.append(point);pn.append(n);pch.append(ch);pindex.append(dict(profile_index=pi,source_row_id=rowid,source_family_id=row['source_family_id'],shot_id=sid,native_pts=pts,side=side,decoded_native_BGR_sha256=digest,channel_selected_by_actual_material_operator=True,diagnostic_profile_only=True,complete_measurement_operator_receipt_is_in_source_row=True,not_a_complete_independent_pixel_noise_model=True))
   active=measured;prev=gray;last_pts=pts;sidprev=sid
   record={**base,'status':'actual_current_native_profile_measured' if measured else 'no_source_supported_family_requires_automatic_reacquisition','decoded_native_BGR_sha256':digest,'source_rows':len(measured),'candidate_rows':candidate_count,'rejected_navigation_or_source':dict(rejected),'current_person_boxes':boxes,'current_SAM2_inference_executed':bool(prompts),'source_person_exclusion_pixels':int(mask.sum()),'source_mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),'current_rgb_consumed_for_training':True,'numeric_W_ready':False,'solver_admission':False};_write(ledger,record);frames.append(record);counts['development_frames']+=1;counts['actual_current_profile_rows']+=len(measured);counts['development_frames_with_support']+=bool(measured);shotcounts[sid]['development_frames']+=1;shotcounts[sid]['source_rows']+=len(measured)
   if len(frames)>=50:flush();print(json.dumps(dict(pts=pts,shot=sid,counts=dict(counts),elapsed=time.perf_counter()-started)),flush=True)
 flush()
 if processed!=set(expected):raise ValueError('Native source ledger scope incomplete')
 result=dict(schema='rinkfit.joint9.dense_native_receipt.v1',counts=dict(counts),shot_counts={k:dict(v) for k,v in shotcounts.items()},source_frames_accounted=len(processed),expected_source_frames=len(expected),all_native_pts_accounted=processed==set(expected),actual_pixel_provenance=True,current_source_rows_are_not_predictions=True,protected_pixels_for_training=False,numeric_W_ready=False,solver_admission=False,all_frames_material_support_complete=False,chunks=chunks,elapsed_seconds=time.perf_counter()-started,policy_sha256=file_sha256(out/'policy.json'),frame_ledger_sha256=file_sha256(out/'frame_ledger.jsonl.gz'))
 write_json(out/'manifest.json',result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--project',required=True);p.add_argument('--out',required=True);p.add_argument('--source',required=True);p.add_argument('--decisions',required=True);p.add_argument('--material-map',required=True);p.add_argument('--shots',nargs='*');a=p.parse_args();print(json.dumps(measure(a.run,a.project,out_dir=a.out,source=a.source,decisions=a.decisions,material_map=a.material_map,shot_ids=a.shots),ensure_ascii=False))
if __name__=='__main__':main()
