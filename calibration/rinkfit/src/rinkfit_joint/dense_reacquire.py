"""Backwards native reacquisition of forward-pass gaps without crossing roles.

Only missing development exposures are released.  Right-anchor image matching
navigates a fresh source profile; UV is never copied from flow or a model.
"""

from rinkspace.storage import resolve_asset
from pathlib import Path
from collections import defaultdict,Counter
from fractions import Fraction
import argparse,gzip,hashlib,json,sys,time
import av,cv2,numpy as np
from .io_utils import read_json,read_jsonl,write_json,write_jsonl,file_sha256
from .role_guard import FrameRoleGuard,DENY_TRAIN
from .dense_native import _seed_rows,_sample_patches,_ncc,_channels
from .source_worker import _sample
from .physical_source_operator import prepare_current_source,remeasure_physical_source,_sample as _sample_current_cubic

def reacquire(run,project,first_pass,source,decisions,material_map,out,shot_ids=None):
 run=Path(run);project=Path(project);first=Path(first_pass);out=Path(out)
 if out.exists():raise ValueError('A new native reacquisition generation is required')
 start=time.perf_counter();scope=read_json(run/'programme_scope.json');video=Path(scope['source']['path']);guard=FrameRoleGuard(run/'roles/frame_roles.json')
 if file_sha256(video)!=scope['source']['sha256']:raise ValueError('Wrong official source video')
 ledger={int(r['native_pts']):r for r in read_jsonl(first/'frame_ledger.jsonl.gz')};seeds=_seed_rows(source,decisions,material_map,project=project,audit_file=out.with_name(out.name+'_seed_gate.json'));missing=[r for r in ledger.values() if r['status']=='no_source_supported_family_requires_automatic_reacquisition' and (shot_ids is None or r['shot_id'] in shot_ids)];sessions=defaultdict(list);unavailable=[]
 for r in missing:
  sid=r['shot_id'];p=r['native_pts'];candidates=sorted(q for s,q in seeds if s==sid and q>=p and (q-p)/12800<=.6)
  selected=None;reasons=[]
  for q in candidates:
   try:guard.assert_training_track(p,q,shot_id=sid)
   except ValueError as e:reasons.append(str(e));continue
   selected=q;break
  if selected is None:unavailable.append({'shot_id':sid,'native_pts':p,'status':'no_right_source_anchor_within_guarded_identity_interval','reason':reasons or ['no_eligible_current_source_anchor_in_local_interval'],'source_rows':0})
  else:sessions[(sid,selected)].append(p)
 out.mkdir(parents=True);(out/'native_cache').mkdir();(out/'mask_cache').mkdir();write_json(out/'policy.json',{'schema':'rinkfit.native_gap_reacquisition_policy.v1','source_video_sha256':scope['source']['sha256'],'first_pass':{'path':str(first.resolve()),'sha256':file_sha256(first/'frame_ledger.jsonl.gz')},'direction':'backwards from actual right selected source anchor','maximum_right_anchor_age_seconds':.6,'flow_is_navigation_only':True,'current_native_profiles_required':True,'current_entity_corner_remeasurement':'cornerSubPix plus two-direction native structure','cached_current_masks_and_RGB_proofs':True,'role_guard_required_for_entire_interval':True,'source_coordinates_from_model':False,'source_uv_generated_from_flow':False,'formal_statistics_ready':False})
 sys.path[:0]=[str(project),str(project/'src')]
 from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
 add_geometry_dependencies()
 import torch
 from rinkspace.vision_instances import Sam2SourceInstances
 torch.set_num_threads(2);cv2.setNumThreads(1);provider=Sam2SourceInstances(resolve_asset(project, project/'models/cache/sam2.1-hiera-tiny'),device='cuda',reread_rgb=False,contour_stride=16)
 result_rows=[];receipts=list(unavailable);profiles=[];profile_index=[];puv=[];pn=[];pch=[];normal_off=np.arange(-3,3.0001,.25);tangent_off=np.arange(-2,3,dtype=float);counts=Counter()
 def mask_for(pts,rgb,binding):
  cache=out/'mask_cache'/f'{pts}.png'
  if cache.is_file():return (cv2.imread(str(cache),cv2.IMREAD_GRAYSCALE)>0).astype(np.uint8)
  record=ledger[pts];mask=np.zeros((1080,1920),np.uint8);prompts=[]
  # Reuse the first pass current detector boxes bound to this exact RGB/PTS.
  # SAM was not cached in first pass, so only this missing interval repeats it.
  for i,box in enumerate(record.get('current_person_boxes',[])):
   old=box['source_binding']
   if old.get('source_pts')!=pts or old.get('decoded_native_rgb_sha256')!=binding['decoded_native_rgb_sha256']:raise ValueError('Cached current person box belongs to different source pixels')
   q=np.asarray(box['box_xyxy'],float);a=np.maximum([0,0],np.floor(q[:2]-12).astype(int));b=np.minimum([1920,1080],np.ceil(q[2:]+12).astype(int));mask[a[1]:b[1],a[0]:b[0]]=1
   if q[3]>500 and q[3]-q[1]>100:prompts.append(dict(object_id=i+1,kind='occluder',origin='original_rgb_candidate',source_binding=binding,box_xyxy=q.tolist(),exclusion_authority=None))
  if prompts:
   current=provider.read(rgb,binding,prompts=prompts,reset=True);sam=current['occluder_region_candidate_mask'].detach().cpu().numpy().astype(np.uint8);mask|=cv2.dilate(sam,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))
  mask[:170,:620]=1;mask[900:,1600:]=1;cv2.imwrite(str(cache),mask*255);return mask
 for (sid,right),targets in sorted(sessions.items()):
  left=min(targets);guard.assert_training_track(left,right,shot_id=sid);frames={};needed=set(range(left,right+1,256))
  with av.open(str(video)) as container:
   stream=container.streams.video[0]
   if Fraction(stream.time_base)!=Fraction(1,12800):raise ValueError('Non-native source clock')
   container.seek(left,backward=True,any_frame=False,stream=stream)
   for frame in container.decode(stream):
    pts=int(frame.pts)
    if pts>right:break
    if pts not in needed:continue
    guard.assert_training_pixels([pts],shot_id=sid);bgr=frame.to_ndarray(format='bgr24');digest=hashlib.sha256(bgr.tobytes()).hexdigest()
    if digest!=ledger[pts]['decoded_native_BGR_sha256']:raise ValueError('Reacquisition frame pixels disagree with first pass')
    frames[pts]=bgr;cv2.imwrite(str(out/'native_cache'/f'{pts}.png'),bgr,[cv2.IMWRITE_PNG_COMPRESSION,2])
  if set(frames)!=needed:raise ValueError('Missing official native exposure in backwards interval')
  active=[];prev=None
  for pts in range(right,left-1,-256):
   bgr=frames[pts];rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);rgbsha=hashlib.sha256(rgb.tobytes()).hexdigest();bgrsha=hashlib.sha256(bgr.tobytes()).hexdigest();binding=dict(source_sha256=scope['source']['sha256'],source_pts=pts,time_base='1/12800',native_hw=[1080,1920],shot_id=sid,decoded_native_rgb_sha256=rgbsha);mask=mask_for(pts,rgb,binding);rejected=Counter()
   if pts==right:
    current=seeds[(sid,right)];patches=_sample_patches(gray,[x['uv'] for x in current]);active=[]
    for x,patch in zip(current,patches):
     u,v=np.rint(x['uv']).astype(int)
     if not(8<u<1911 and 8<v<1071) or mask[v,u]:continue
     active.append(dict(x,anchor_patch=patch,flank_center=x['uv']))
   elif active:
    old=np.asarray([x['uv'] for x in active],np.float32).reshape(-1,1,2);nxt,st,_=cv2.calcOpticalFlowPyrLK(prev,gray,old,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01));back,bs,_=cv2.calcOpticalFlowPyrLK(gray,prev,nxt,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01));fb=np.linalg.norm(back[:,0]-old[:,0],axis=1);corr=_ncc(np.stack([x['anchor_patch'] for x in active]),_sample_patches(gray,nxt[:,0]));keep=[]
    for i,x in enumerate(active):
     q=nxt[i,0]
     if not(st[i,0] and bs[i,0]) or not np.isfinite(q).all() or fb[i]>1.:rejected['reverse_flow_navigation_not_verified']+=1;continue
     if not(8<q[0]<1911 and 8<q[1]<1071) or mask[int(round(q[1])),int(round(q[0]))]:rejected['current_actor_graphics_or_profile_domain_exclusion']+=1;continue
     if corr[i]<.75:rejected['current_source_appearance_not_supported']+=1;continue
     keep.append(dict(x,uv=q.astype(float).tolist(),flow_fb_px=float(fb[i]),anchor_patch_ncc=float(corr[i])))
    active=keep
   source_context=prepare_current_source(rgb,mask,project);channels=source_context['channels'];measured=[];frame_rows=[]
   for x in active:
    q=np.asarray(x['uv'],float);sides=x.get('flank_uv')
    if sides is not None:sides=np.asarray(sides)+(q-np.asarray(x.get('flank_center',q)))
    if x['entity_point']:
     corner=cv2.cornerSubPix(gray,q.astype(np.float32).reshape(1,1,2),(4,4),(-1,-1),(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.001))[0,0]
     if np.linalg.norm(corner-q)>3.:rejected['current_corner_refinement_outside_real_local_window']+=1;continue
     q=corner.astype(float)
    rr=remeasure_physical_source(source_context,dict(target_id=x['target_id'],category='ad_corner' if x['entity_point'] else {'poster_boundary':'ad_edge','ink_boundary':'ad_print'}.get(x['operator'])),q,x.get('normal'),sides)
    if rr.get('current_uv') is None:rejected[rr['status']]+=1;continue
    uv=np.asarray(rr['current_uv']);u,v=np.rint(uv).astype(int)
    if not(8<u<1911 and 8<v<1071) or mask[v,u]:rejected['remeasured_source_current_occlusion']+=1;continue
    if x['entity_point'] and rr.get('physical_entity_source_verified') is not True:rejected['entity_corner_second_direction_not_supported']+=1;continue
    old=dict(x);old['uv']=uv.tolist();old['flank_uv']=rr.get('flank_uv',x.get('flank_uv'));old['flank_center']=uv.tolist();measured.append(old)
    if pts not in targets:continue
    rowid=f"dense_reverse/{sid}/{pts}/{x['source_row_id']}";row={'shot_id':sid,'native_pts':pts,'time_base':'1/12800','source_row_id':rowid,'anchor_source_row_id':x['source_row_id'],'anchor_pts':right,'source_family_id':f"{sid}/{pts}/{x['source_family_id']}",'joint_material_id':x['joint_material_id'],'material_index':x['material_index'],'target_id':x['target_id'],'measurement_operator':x['operator'],'point_correspondence':x['entity_point'],'source_uv':uv.tolist(),'current_uv':uv.tolist(),'navigation_seed_uv':q.tolist(),'measurement_contract_sha256':rr.get('measurement_contract_sha256'),'auxiliary_profiles_diagnostic_only':True,'source_measurement_audit':rr,'reverse_flow_fb_px':x.get('flow_fb_px',0.),'anchor_patch_ncc':x.get('anchor_patch_ncc',1.),'decoded_native_BGR_sha256':bgrsha,'current_pixels_measured':True,'pixels_from_model':False,'numeric_statistics_ready':False,'solver_eligible':False,'role':'development','pending_reason':'source-family numeric statistics and physical replay required'};result_rows.append(row);frame_rows.append(row)
    n=rr.get('normal')
    if n is not None:
     n=np.asarray(n,float);t=np.array([-n[1],n[0]])
     for side,point in enumerate(rr.get('flank_uv') or [uv.tolist()]):
      coords=np.asarray(point)+tangent_off[:,None,None]*t+normal_off[None,:,None]*n;allc=_sample_current_cubic(source_context['spline_channels'],coords.reshape(-1,2)).reshape(5,25,3);gradient=np.gradient(allc[2],.25,axis=0);ch=int((rr.get('source_material_receipt') or {}).get('channel',(rr.get('profile') or {}).get('channel',0)));i=len(profiles);profiles.append(allc[:,:,ch]);puv.append(point);pn.append(n);pch.append(ch);profile_index.append({'profile_index':i,'source_row_id':rowid,'source_family_id':row['source_family_id'],'native_pts':pts,'shot_id':sid,'side':side,'decoded_native_BGR_sha256':bgrsha})
   active=measured;prev=gray
   if pts in targets:
    receipt={'shot_id':sid,'native_pts':pts,'status':'actual_current_source_reacquired' if frame_rows else 'backwards_current_source_still_unresolved','source_rows':len(frame_rows),'right_anchor_pts':right,'track_interval_role_guard_verified':True,'source_BGR_sha256':bgrsha,'source_mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),'native_png':str(out/'native_cache'/f'{pts}.png'),'native_png_sha256':file_sha256(out/'native_cache'/f'{pts}.png'),'current_mask_png':str(out/'mask_cache'/f'{pts}.png'),'current_mask_png_sha256':file_sha256(out/'mask_cache'/f'{pts}.png'),'rejections':dict(rejected),'current_native_UV_not_flow_output':True,'numeric_statistics_ready':False};receipts.append(receipt);counts['recovered_frames' if frame_rows else 'still_unresolved_frames']+=1;counts['recovered_actual_source_rows']+=len(frame_rows)
  print(json.dumps({'shot_id':sid,'right_anchor':right,'gap_frames':len(targets),'counts':dict(counts)}),flush=True)
 write_jsonl(out/'rows.jsonl.gz',result_rows);write_jsonl(out/'frame_receipts.jsonl.gz',receipts);write_json(out/'profile_index.json',profile_index);np.savez_compressed(out/'profiles.npz',intensity=np.asarray(profiles,np.float32).reshape(-1,5,25),uv=np.asarray(puv,float).reshape(-1,2),normal=np.asarray(pn,float).reshape(-1,2),channel=np.asarray(pch,np.uint8),normal_offsets=normal_off,tangent_offsets=tangent_off)
 result={'schema':'rinkfit.joint9.native_gap_reacquisition.v1','input_missing_frames':len(missing),'right_anchor_sessions':len(sessions),'counts':dict(counts),'no_guarded_right_anchor':len(unavailable),'all_missing_frames_accounted':len(receipts)==len(missing),'current_source_rows':len(result_rows),'actual_profiles':len(profiles),'source_video_sha256':scope['source']['sha256'],'source_role_file_sha256':file_sha256(run/'roles/frame_roles.json'),'protected_pixels_consumed':False,'cached_native_frames':len(list((out/'native_cache').glob('*.png'))),'numeric_statistics_ready':False,'physical_projection_replay_pending':True,'elapsed_seconds':time.perf_counter()-start};write_json(out/'manifest.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--project',required=True);p.add_argument('--first-pass',required=True);p.add_argument('--source',required=True);p.add_argument('--decisions',required=True);p.add_argument('--material-map',required=True);p.add_argument('--out',required=True);p.add_argument('--shots',nargs='*');a=p.parse_args();print(json.dumps(reacquire(a.run,a.project,a.first_pass,a.source,a.decisions,a.material_map,a.out,a.shots)))
