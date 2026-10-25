"""Frozen-state evaluation of protected native exposures; never training input.

The frozen model supplies a search region only. Material identity and UV are
read from current paired blue/body/yellow topology; no model UV becomes data.
This is a board-interface accuracy diagnostic, not full-scene certification.
"""
from pathlib import Path
from fractions import Fraction
from collections import Counter,defaultdict
import argparse,gzip,hashlib,json,sys,time
import av,cv2,numpy as np
from scipy.spatial import cKDTree
from .io_utils import read_json,write_json,file_sha256
from .role_guard import FrameRoleGuard,DENY_TRAIN
from .joint_model import graph_from_joint,as_event_view
from .camera_graph import CameraGraph
from .joint_replay import fast_native_project,render_visibility
from .measurement_contract import EventMeasurementContract
from .joint_solver import assert_source_identity_current
from .bound_source_guard import BoundSourceIdentityGuard
from rinkfit_start.native_scene import NativeScene

def _verify(ref):
 if file_sha256(ref['path'])!=ref['sha256']:raise ValueError('Frozen dependency changed: '+ref['path'])
 return read_json(ref['path'])

def validate_freeze(freeze_file,run):
 f=read_json(freeze_file)
 assert_source_identity_current(f['packet']['path'])
 if f.get('schema')!='rinkfit.joint9.frozen_joint_state.v1' or not f.get('completed_joint_optimization') or not f.get('all_shared_blocks_released') or not f.get('all_active_sources_true_RS_replay_valid'):raise ValueError('A complete, released, true-RS replayed immutable joint state is required')
 for name in ('model','packet','complete_state','full_source_replay','source_admission','frame_roles','source_semantic_audit'):_verify(f[name])
 semantics=_verify(f['source_semantic_audit'])
 if semantics.get('all_fit_sources_physical_semantics_verified') is not True or semantics.get('active_packet_sha256')!=f['packet']['sha256']:raise ValueError('A pre-audit or semantically unverified state cannot consume protected evaluation RGB')
 if f['frame_roles']['sha256']!=file_sha256(Path(run)/'roles/frame_roles.json'):raise ValueError('Evaluation role manifest differs from the frozen state')
 return f

def _measure_current(rgb,mask,context):
 # This is the unique source-only outer-boundary reader; there is deliberately
 # no fallback to a generic gradient or the withdrawn inner-band interfaces.
 from .board_outer_source import source_outer_boundary_rows
 rows,certificates=source_outer_boundary_rows(rgb,mask,current_board_context=context)
 for row in rows:
  role=row.get('material_role')
  if role not in ('board.top','board.contact'):raise ValueError('Source reader did not establish the user-confirmed outer boundary role')
  if row.get('source_uv')!=row.get('current_uv'):raise ValueError('Evaluation source UV must be its first actual current measurement')
  if row.get('source_material_role_requalification',{}).get('status')!='pass':raise ValueError('Evaluation requires actual same-exposure material-role qualification')
  row['material_label']=role
 return rows,certificates

def _frozen_board_search_context(scene,camera,local,*,native_pts,shot_id,bgrsha,rgbsha,model_ref,source_sha256,output):
 # The source reader still proves actual blue/body/yellow material in this RGB.
 # These projected points supply no fit UV, identities, or visibility claims.
 grid=np.linspace(0,scene.perimeter,2049);world=np.concatenate([scene.board(grid,h) for h in (scene.height,0.)]);uv,good,_=fast_native_project(camera,world,local);record=[dict(label='board.top',begin=0,end=2049,board_material_surface=True,board_top_boundary=True),dict(label='board.contact',begin=2049,end=len(world),board_material_surface=True)];good&=render_visibility(camera,scene,world,record,local,uv);good&=np.isfinite(uv).all(1)&(uv[:,0]>=-100)&(uv[:,0]<=2020)&(uv[:,1]>=-100)&(uv[:,1]<=1180)
 a=uv[:2049][good[:2049]];b=uv[2049:][good[2049:]]
 if len(a)<2 or len(b)<2:return None
 a=a[np.argsort(a[:,0])];b=b[np.argsort(b[:,0])];a=a[np.unique(np.linspace(0,len(a)-1,min(64,len(a))).round().astype(int))];b=b[np.unique(np.linspace(0,len(b)-1,min(64,len(b))).round().astype(int))];alluv=np.concatenate([a,b]);top=[max(0,int(np.floor(a[:,1].min()-80))),min(1080,int(np.ceil(a[:,1].max()+80)))];foot=[max(0,int(np.floor(b[:,1].min()-80))),min(1080,int(np.ceil(b[:,1].max()+80)))];box=[max(0,int(np.floor(alluv[:,0].min()-80))),max(0,min(top[0],foot[0])),min(1920,int(np.ceil(alluv[:,0].max()+80))),min(1080,max(top[1],foot[1]))]
 if box[2]<=box[0] or box[3]<=box[1] or top[1]<=top[0] or foot[1]<=foot[0]:return None
 frame=dict(native_pts=int(native_pts),shot_id=shot_id,time_base='1/12800',source_video_sha256=source_sha256,decoded_native_BGR_sha256=bgrsha,board_instance_roi_xyxy=box,top_blue_upper_search_y_range=top,contact_yellow_lower_search_y_range=foot,top_blue_upper_visible=True,contact_yellow_lower_visible=True,visibility_flags_mean_search_proposal_only=True,source_top_readout_navigation_uv=a.tolist(),source_contact_readout_navigation_uv=b.tolist(),source_model_navigation=model_ref,physical_semantic_authority='current_joint_model_projection_search_only; current_native_material_identity_unproved',source_object_identity_not_proved_by_navigation=True,pixels_from_model=False,ROI_is_search_domain_only=True,fit_UV_created=False,source_fit_UV_created=False,current_RGB_still_required=True)
 path=Path(output)/('search_'+str(int(native_pts))+'.json')
 if path.exists():raise ValueError('Frozen per-exposure search receipt exists')
 path.write_text(json.dumps(dict(schema='rinkfit.joint9.current_joint_model_board_navigation.v1',frames=[frame],source_model_navigation=model_ref,source_identity_certified=False,source_fit_UV_created=False,protected_pixels_used_for_training=False),separators=(',',':')),encoding='utf8')
 return dict(native_pts=int(native_pts),shot_id=shot_id,actual_native_RGB_sha256=rgbsha,source_review=dict(path=str(path.resolve()),sha256=file_sha256(path)))

def _project_rows(scene,camera,local,height_fraction,rows):
 def world_at(s):return scene.board(np.asarray(s)*scene.perimeter,scene.height*height_fraction)
 source=np.asarray([r['current_uv'] for r in rows]);n=len(source);grid=np.linspace(0,1,2049);world=world_at(grid);uv,good,_=fast_native_project(camera,world,local);record=[dict(label='board.top' if height_fraction==1. else 'board.contact',begin=0,end=len(world),board_material_surface=True,board_top_boundary=height_fraction==1.)];good&=render_visibility(camera,scene,world,record,local,uv)
 if not good.any():return [dict(valid=False,reason='no_visible_finite_same_material_projection') for r in rows]
 ids=np.flatnonzero(good);_,nn=cKDTree(uv[good]).query(source);s=grid[ids[nn]];active=np.ones(n,bool)
 # Deterministic finite-curve nearest-point refinement after actual source read.
 # This is a nuisance-coordinate diagnostic, with no fitted theta changes.
 for step in range(10):
  w=world_at(s);pred,ok,eq=fast_native_project(camera,w,local);lo=np.maximum(0,s-1e-6);hi=np.minimum(1,s+1e-6);a,ag,_=fast_native_project(camera,world_at(lo),local);b,bg,_=fast_native_project(camera,world_at(hi),local);derivative=(b-a)/np.maximum(hi-lo,1e-12)[:,None];den=np.sum(derivative**2,axis=1);active&=ok&ag&bg&np.isfinite(den)&(den>0);move=np.divide(np.sum(derivative*(pred-source),axis=1),den,out=np.zeros(n),where=active);s=np.clip(s-np.clip(move,-.01,.01),0,1)
 pred,ok,eq=fast_native_project(camera,world_at(s),local);ok&=active;result=[]
 for i,r in enumerate(rows):
  normal=np.asarray(r['normal']);error=pred[i]-source[i];result.append(dict(valid=bool(ok[i]),reason='' if ok[i] else 'true_RS_or_finite_refinement_invalid',finite_station=float(s[i]),predicted_uv=pred[i].tolist() if np.isfinite(pred[i]).all() else None,normal_error_px=float(abs(error@normal)) if ok[i] else None,euclidean_error_px=float(np.linalg.norm(error)) if ok[i] else None,RS_equality_px=float(eq[i]) if np.isfinite(eq[i]) else None,finite_global_closest_point_certified=False))
 return result

def evaluate(run,project,freeze_file,output,*,compiled_dir=None,gpu_dll=None,gpu_validation=None):
 run=Path(run);project=Path(project);out=Path(output)
 if out.exists():raise ValueError('Frozen evaluation requires a new immutable generation')
 entry_refs=[dict(path=str(p),sha256=file_sha256(p))for p in (Path(freeze_file).resolve(),run/'programme_scope.json')]
 frozen=validate_freeze(freeze_file,run);model=_verify(frozen['model']);scope=read_json(run/'programme_scope.json');guard=FrameRoleGuard(run/'roles/frame_roles.json');expected={pts:r for pts,r in guard.roles.items() if r['role'] in DENY_TRAIN};cg=CameraGraph(graph_from_joint(model));scene=NativeScene(as_event_view(model,'s01'));lookup={m['label']:i for i,m in enumerate(model['materials'])}
 # Import before any protected source RGB is consumed. A missing corrected
 # source reader is a hard error, never a reason to fall back to the old one.
 from .board_outer_source import source_outer_boundary_rows
 contract=EventMeasurementContract(project,'ankara_2026');contract.verify_unchanged()
 if not any(x.startswith('board.top.') for x in lookup) or not any(x.startswith('board.contact.') for x in lookup):raise ValueError('Frozen model lacks existing physical H/0 board boundaries')
 if file_sha256(scope['source']['path'])!=scope['source']['sha256']:raise ValueError('Official native source changed')
 reader_files=[Path(__file__).with_name(n)for n in ('board_outer_source.py','board_local_source.py','board_source_requalification.py','current_person_exclusion.py','measurement_contract.py')]+[project/'src/rinkspace/vision_source_topology.py']
 source_binding=BoundSourceIdentityGuard(frozen['packet']['path'],packet_sha256=frozen['packet']['sha256'],extra_refs=[frozen[n]for n in ('model','complete_state','full_source_replay','source_admission','frame_roles','source_semantic_audit')]+[scope['source']]+entry_refs+list(contract.references.values())+[dict(path=str(p),sha256=file_sha256(p))for p in reader_files])
 out.mkdir(parents=True);sys.path[:0]=[str(project),str(project/'src')]
 from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
 add_geometry_dependencies()
 import torch
 from rinkspace.vision_instances import SourcePersonBoxes,Sam2SourceInstances
 torch.set_num_threads(2);cv2.setNumThreads(1)
 detector=SourcePersonBoxes(project/'models/cache/yolo11n-pose.pt',dependency_paths=[project/'.venv-gpu/Lib/site-packages'],runtime_directory=out/'detector_runtime',device='cuda');provider=Sam2SourceInstances(project/'models/cache/sam2.1-hiera-tiny',device='cuda',reread_rgb=False,contour_stride=16)
 if compiled_dir:
  from .joint_acceleration import install
  worlds=np.concatenate([scene.board(np.linspace(0,scene.perimeter,2049),scene.height*h) for h in (1.,0.)]);install(cg,frozen['model']['path'],out/'acceleration',worlds,compiled_dir,gpu_dll=gpu_dll,gpu_validation=gpu_validation)
 counts=Counter();shotcounts=defaultdict(Counter);errors=[];processed=set();started=time.perf_counter()
 with av.open(scope['source']['path']) as source,gzip.open(out/'frames.jsonl.gz','wt',encoding='utf8') as frames,gzip.open(out/'source_rows.jsonl.gz','wt',encoding='utf8') as rowsfile:
  stream=source.streams.video[0]
  if Fraction(stream.time_base)!=Fraction(1,12800):raise ValueError('Native source time base changed')
  for frame in source.decode(stream):
   pts=int(frame.pts)
   if pts not in expected:
    if pts>max(expected):break
    continue
   source_binding.check()
   role=expected[pts];sid=role['shot_id'];processed.add(pts);bgr=frame.to_ndarray(format='bgr24');rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);bgrsha=hashlib.sha256(bgr.tobytes()).hexdigest();binding=dict(source_sha256=scope['source']['sha256'],source_pts=pts,time_base='1/12800',native_hw=[1080,1920],shot_id=sid,decoded_native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest());boxes=detector.read(rgb,binding)
   from .current_person_exclusion import current_person_exclusion_mask
   local=float(cg.local_seconds(sid,np.array([pts],np.int64))[0]);camera=cg.native_camera(sid);context=_frozen_board_search_context(scene,camera,local,native_pts=pts,shot_id=sid,bgrsha=bgrsha,rgbsha=binding['decoded_native_rgb_sha256'],model_ref=frozen['model'],source_sha256=scope['source']['sha256'],output=out)
   if context is not None:context['native_video_decode_receipt']=dict(native_pts=pts,time_base='1/12800',source_video_sha256=scope['source']['sha256'],decoded_native_BGR_sha256=bgrsha,decoded_native_RGB_sha256=binding['decoded_native_rgb_sha256'],shot_id=sid,native_RGB_resampled=False,source_role='sealed_validation',validation_role_verified=True,training_pixels_created=False)
   mask,exclusion=current_person_exclusion_mask(rgb,boxes,native_pts=pts,current_board_context=context);mask=mask.astype(np.uint8);prompts=[]
   for i in exclusion['reflection_columns_extended_only_for_foreground_boxes']:
    q=boxes[i]['box_xyxy'];prompts.append(dict(object_id=i+1,kind='occluder',origin='original_rgb_candidate',source_binding=binding,box_xyxy=q,exclusion_authority=None))
   if prompts:mask|=cv2.dilate(provider.read(rgb,binding,prompts=prompts,reset=True)['occluder_region_candidate_mask'].detach().cpu().numpy().astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))
   
   current,certs=_measure_current(rgb,mask,context) if context else ([],[])
   for label in sorted({r['material_label'] for r in current}):
    select=[r for r in current if r['material_label']==label];predictions=_project_rows(scene,camera,local,1. if label=='board.top' else 0.,select)
    for i,(r,pred) in enumerate(zip(select,predictions)):
     row=dict(**r,**pred,shot_id=sid,native_pts=pts,time_base='1/12800',source_row_id=f'frozen/{sid}/{pts}/{label}/{i}',role='sealed_validation',original_reserved_role=role['role'],model_sha256=frozen['model']['sha256'],decoded_native_BGR_sha256=bgrsha,current_RGB_measured=True,training_eligible=False,independence_certified=False);rowsfile.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');counts['measured_rows']+=1;counts['valid_rows' if pred['valid'] else 'invalid_rows']+=1
     if pred['valid']:errors.append(pred['normal_error_px'])
   status='actual_current_board_interfaces_measured' if current else 'source_paired_material_topology_unavailable';fr=dict(shot_id=sid,native_pts=pts,time_base='1/12800',original_reserved_role=role['role'],role='sealed_validation',status=status,source_rows=len(current),decoded_native_BGR_sha256=bgrsha,model_sha256=frozen['model']['sha256'],source_material_certificates=certs,current_person_reflection_exclusion=exclusion,model_used_for_source_identity=False,model_used_for_source_search_only=True,model_UV_used_as_source_readout=False,tracking_across_exposures=False,training_eligible=False,independence_certified=False);frames.write(json.dumps(fr,separators=(',',':'),allow_nan=False)+'\n');counts['frames']+=1;counts['frames_with_source' if current else 'frames_without_supported_source']+=1;shotcounts[sid][status]+=1
   if counts['frames']%20==0:write_json(out/'progress.json',dict(counts=dict(counts),expected=len(expected),elapsed_seconds=time.perf_counter()-started))
 if processed!=set(expected):raise ValueError('Frozen native frame coverage incomplete')
 validate_freeze(freeze_file,run);source_guard_receipt=source_binding.finish();err=np.asarray(errors);receipt=dict(schema='rinkfit.joint9.frozen_native_board_validation.v1',frozen_state=dict(path=str(Path(freeze_file).resolve()),sha256=file_sha256(freeze_file)),model=frozen['model'],source=scope['source'],frames_expected=len(expected),counts=dict(counts),by_shot={s:dict(c) for s,c in shotcounts.items()},normal_error_px=dict(max=float(err.max()) if len(err) else None,median=float(np.median(err)) if len(err) else None,p95=float(np.quantile(err,.95)) if len(err) else None),every_protected_exposure_accounted=True,training_outputs_created=False,protected_pixels_used_for_training=False,source_profile_covariance_claimed=False,independence_certified=False,lineage_reason='Historical template/navigation exposure lineage is not certified independent',scope='two shared physical board material interfaces only; no full ice/ad accuracy claim',source_policy=dict(reader='rinkfit_joint.board_outer_source.source_outer_boundary_rows',actor_exclusion='same public current_person_exclusion_mask: four-pixel current body boxes, only below actually measured current yellow LOWER contact extend reflection columns; optional same-exposure foreground SAM2 adds exclusions',actor_exclusion_reader_sha256=file_sha256(Path(__file__).parent/'current_person_exclusion.py'),reader_sha256=file_sha256(Path(__file__).parent/'board_outer_source.py'),measurement_contract_sha256=contract.digest,existing_user_fact_references=contract.references,physical_outer_boundary_support='blue upper H / yellow lower 0, reflection excluded'),scoped_source_integrity=source_guard_receipt,continuous_visual_review_complete=False,whole_scene_accuracy_pass_claimed=False,elapsed_seconds=time.perf_counter()-started,files={n:dict(path=str(out/n),sha256=file_sha256(out/n)) for n in ('frames.jsonl.gz','source_rows.jsonl.gz')});write_json(out/'manifest.json',receipt);return receipt

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--project',required=True);p.add_argument('--freeze',required=True);p.add_argument('--out',required=True);p.add_argument('--compiled-dir');p.add_argument('--gpu-dll');p.add_argument('--gpu-validation');a=p.parse_args();print(json.dumps(evaluate(a.run,a.project,a.freeze,a.out,compiled_dir=a.compiled_dir,gpu_dll=a.gpu_dll,gpu_validation=a.gpu_validation),ensure_ascii=False))
