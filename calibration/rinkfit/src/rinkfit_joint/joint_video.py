"""Native-clock same-model full programme overlay, after source improvement review."""
from pathlib import Path
from fractions import Fraction
import argparse,gzip,hashlib,json,subprocess,time
import av,numpy as np
from .camera_graph import CameraGraph
from .joint_model import graph_from_joint,as_event_view,validate_physical_domain
from .joint_replay import inventory,fast_native_project,render_visibility,draw_fast
from .joint_display import CurrentSourcePoints,SharedModelTopView
from .joint_solver import assert_source_identity_current
from .bound_source_guard import BoundSourceIdentityGuard
from .io_utils import read_json,write_json,file_sha256
from rinkfit_start.native_scene import NativeScene

def verified(reference):
 p=Path(reference['path'])
 if file_sha256(p)!=reference['sha256']:raise ValueError('Review evidence SHA mismatch: '+str(p))
 return read_json(p)


_FIRST4=(('s01','main',256256,575488),('s02','secondary',575744,720640),('s03','end',720896,745728),('s04','main',745984,948736))
_FIRST6=_FIRST4+(('s05','end',948992,1300992),('s06','secondary',1301248,1425408))

def _selected_scope(model,scope,cg,scope_mode):
 """Validate the actual selected source interval without certifying other shots."""
 if scope_mode=='programme':
  cg.validate(require_ready=True)
  return 8434,'joint9'
 if scope_mode not in ('first4','first6'):raise ValueError('Unknown native replay scope mode')
 selected=_FIRST4 if scope_mode=='first4' else _FIRST6
 if not validate_physical_domain(model)['passed']:raise ValueError('Selected physical model is outside its actual scene/material domain')
 if set(model['shots'])!={s[0] for s in selected}:raise ValueError('Selected replay requires the actual scoped physical model')
 if len(scope['shots'])!=len(selected):raise ValueError('Selected replay scope contains other shots')
 clock=model['source_clock']
 if clock.get('time_base')!=[1,12800] or clock.get('frame_step_pts')!=256 or clock.get('image_size')!=[1920,1080] or clock['source_video_sha256']!=scope['source']['sha256']:raise ValueError('Selected source clock mismatch')
 for s,(sid,rid,first,last) in zip(scope['shots'],selected):
  if (s['id'],s['rig_id'],s['first_pts'],s['last_pts'])!=(sid,rid,first,last):raise ValueError('Selected native source interval or rig identity mismatch')
  if model['shots'][sid]['rig_id']!=rid:raise ValueError('Selected physical rig changed')
  stamps=np.arange(first,last+1,256,dtype=np.int64);cg.local_seconds(sid,stamps);camera=cg.native_camera(sid)
  allowed=float(cg.graph['rigs'][rid].get('tau_bound',.02));duration=(last-first)/12800.
  if not np.isfinite(allowed) or allowed<=0 or abs(camera.tau)>allowed or camera.domain[0]>-allowed/2+1e-12 or camera.domain[1]<duration+allowed/2-1e-12:raise ValueError('Selected camera lacks full native readout support: '+sid)
  if not camera.optical_domain_certificate()['passed']:raise ValueError('Selected optical domain is invalid: '+sid)
 return sum((last-first)//256+1 for _,_,first,last in selected),scope_mode

def render(model_file,scope_file,output,improvement_receipt,*,compiled_dir=None,gpu_dll=None,gpu_validation=None,ffmpeg='ffmpeg',scope_mode='programme'):
 model_file=Path(model_file).resolve();scope_file=Path(scope_file).resolve();out=Path(output).resolve()
 if out.exists():raise ValueError('Full video requires a new immutable generation')
 entry_refs=[dict(path=str(p),sha256=file_sha256(p))for p in (model_file,scope_file,Path(improvement_receipt).resolve())]
 model=read_json(model_file);scope=read_json(scope_file);sha=entry_refs[0]['sha256'];approval=read_json(improvement_receipt)
 expected_schema='rinkfit.'+('joint9' if scope_mode=='programme' else scope_mode)+'.clear_improvement_review.v1'
 if approval.get('schema')!=expected_schema or not approval.get('clear_improvement'):raise ValueError('Actual source improvement review is required before full video')
 if scope_mode in ('first4','first6'):
  selected=_FIRST4 if scope_mode=='first4' else _FIRST6
  if approval.get('all_selected_shots_reviewed') is not True or approval.get('reviewed_shot_ids')!=[s[0] for s in selected]:raise ValueError('Improvement review must cover every selected complete shot')
 elif not approval.get('all_nine_shots_reviewed'):raise ValueError('Actual nine-shot source improvement review is required')
 if approval['model']['sha256']!=sha or file_sha256(approval['model']['path'])!=sha:raise ValueError('Improvement review belongs to a different physical state')
 if approval['native_source_video_sha256']!=scope['source']['sha256']:raise ValueError('Improvement review names another source')
 quality=verified(approval['quality']);visual=verified(approval['visual_review']);verified(approval['comparison_quality']);verified(approval['whole_range_dense_evidence']);semantics=verified(approval['source_semantic_audit'])
 if semantics.get('all_fit_sources_physical_semantics_verified') is not True or semantics.get('active_packet_sha256')!=quality['packet']['sha256']:raise ValueError('Whole active source semantics must be revalidated after the systemic source-operator audit')
 if quality['model']['sha256']!=sha or visual['model']['sha256']!=sha or set(quality['by_shot'])!=set(model['shots']):raise ValueError('Improvement evidence is not the same selected physical state')
 if file_sha256(scope['source']['path'])!=scope['source']['sha256']:raise ValueError('Native source changed')
 packet_ref=quality['packet']
 assert_source_identity_current(packet_ref['path'])
 if file_sha256(packet_ref['path'])!=packet_ref['sha256']:raise ValueError('Current-point packet differs from the reviewed state')
 evaluation_ref=approval.get('sealed_evaluation');evaluation=None
 if evaluation_ref is not None:
  evaluation=verified(evaluation_ref)
  if evaluation['model']['sha256']!=sha:raise ValueError('Sealed current source dots belong to another frozen model')
 evaluation_refs=[evaluation_ref,evaluation['files']['source_rows.jsonl.gz']]if evaluation is not None else []
 source_binding=BoundSourceIdentityGuard(packet_ref['path'],packet_sha256=packet_ref['sha256'],extra_refs=[approval[n]for n in ('model','quality','visual_review','comparison_quality','whole_range_dense_evidence','source_semantic_audit')]+[scope['source']]+entry_refs+evaluation_refs)
 cg=CameraGraph(graph_from_joint(model));expected_count,scope_tag=_selected_scope(model,scope,cg,scope_mode);world,records=inventory(model);scene=NativeScene(as_event_view(model,'s01'));points=CurrentSourcePoints(packet_ref['path'],model,scope,sealed_evaluation=evaluation_ref['path'] if evaluation_ref else None);top=SharedModelTopView(world,records,scene)
 out.mkdir(parents=True);started=time.perf_counter();acceleration=None
 if compiled_dir:
  from .joint_acceleration import install
  acceleration=install(cg,model_file,out/'acceleration',world,compiled_dir,gpu_dll=gpu_dll,gpu_validation=gpu_validation,shots=[s['id']for s in scope['shots']])
 first=min(s['first_pts'] for s in scope['shots']);last=max(s['last_pts'] for s in scope['shots']);expected=list(range(first,last+1,256))
 if len(expected)!=expected_count:raise ValueError('Scope is not the complete selected native interval')
 silent=out/'overlay_silent.mp4';frames=0;by_shot={s['id']:0 for s in scope['shots']};lastpts=None
 with av.open(scope['source']['path']) as source,av.open(str(silent),'w') as sink,gzip.open(out/'frame_replay.jsonl.gz','wt',encoding='utf8') as ledger:
  stream=source.streams.video[0]
  if Fraction(stream.time_base)!=Fraction(1,12800):raise ValueError('Native source time base changed')
  target=sink.add_stream('libx264',rate=Fraction(50,1));target.width=2560;target.height=1080;target.pix_fmt='yuv420p';target.time_base=Fraction(1,12800);target.options={'crf':'18','preset':'fast'}
  for frame in source.decode(stream):
   pts=int(frame.pts)
   if pts<first:continue
   if pts>last:break
   if frames>=expected_count or pts!=expected[frames]:raise ValueError('Missing, duplicate or reordered native source exposure')
   shot=next(s for s in scope['shots'] if s['first_pts']<=pts<=s['last_pts']);sid=shot['id'];raw=frame.to_ndarray(format='bgr24');camera=cg.native_camera(sid);local=float(cg.local_seconds(sid,np.array([pts],np.int64))[0]);uv,good,eq=fast_native_project(camera,world,local);good&=render_visibility(camera,scene,world,records,local,uv)
   title=f'{sid} source={pts/12800:.3f}s PTS={pts} model={sha[:12]} L=1';image,metrics=draw_fast(raw,uv,good,records,title);point_metrics=points.draw(image,pts);image=np.concatenate((image,top.frame(camera,local,sid)),axis=1)
   result=av.VideoFrame.from_ndarray(image,format='bgr24');result.pts=pts-first;result.time_base=Fraction(1,12800)
   for packet in target.encode(result):sink.mux(packet)
   row=dict(programme_frame_index_1based=frames+1,shot_id=sid,rig_id=shot['rig_id'],native_pts=pts,time_base='1/12800',source_seconds=pts/12800.,model_sha256=sha,decoded_native_BGR_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),rendered_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),path_samples=len(world),valid_visible_path_samples=int(good.sum()),maximum_certified_root_equality_px=float(np.max(np.abs(eq[good]),initial=0.)),**metrics,**point_metrics)
   ledger.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');frames+=1;by_shot[sid]+=1;lastpts=pts
   if frames%50==0:source_binding.check()
   if frames%50==0:write_json(out/'progress.json',dict(frames=frames,total=expected_count,current_shot=sid,native_pts=pts,elapsed_seconds=time.perf_counter()-started,model_sha256=sha))
  for packet in target.encode(None):sink.mux(packet)
 source_binding.check()
 if frames!=expected_count or lastpts!=last or file_sha256(model_file)!=sha:raise ValueError('Full video replay did not close against the immutable model/scope')
 final=out/('okada_'+scope_tag+'_overlay.mp4');duration=(last-first+256)/12800.
 command=[str(ffmpeg),'-hide_banner','-nostdin','-y','-ss',f'{first/12800.:.8f}','-i',scope['source']['path'],'-i',str(silent),'-map','1:v:0','-map','0:a:0?','-c:v','copy','-c:a','aac','-t',f'{duration:.8f}','-movflags','+faststart',str(final)]
 with (out/'audio_mux.log').open('w',encoding='utf8') as log:subprocess.run(command,stdout=log,stderr=log,check=True)
 decoded=[]
 with av.open(str(final)) as check:
  st=check.streams.video[0]
  for f in check.decode(st):decoded.append(Fraction(f.pts)*st.time_base)
 if len(decoded)!=expected_count or any(t!=Fraction(i,50) for i,t in enumerate(decoded)):raise ValueError('Encoded overlay lost or shifted a native video exposure')
 source_guard_receipt=source_binding.finish()
 manifest=dict(schema='rinkfit.'+scope_tag+'.complete_native_overlay.v1',reviewed_shot_ids=[s['id']for s in scope['shots']],whole_programme_claim=scope_mode=='programme',model=dict(path=str(model_file),sha256=sha),scope=dict(path=str(scope_file),sha256=file_sha256(scope_file)),source=scope['source'],improvement_review=dict(path=str(Path(improvement_receipt).resolve()),sha256=file_sha256(improvement_receipt)),video=dict(path=str(final),sha256=file_sha256(final)),frame_replay=dict(path=str(out/'frame_replay.jsonl.gz'),sha256=file_sha256(out/'frame_replay.jsonl.gz')),frames=frames,by_shot=by_shot,source_seconds_inclusive=[first/12800.,last/12800.],source_stop_exclusive=(last+256)/12800.,output_native_frame_times_verified=True,all_material_paths_enabled=True,all_ad_numbers_enabled=True,all_ice_identifiers_enabled=True,right_side_shared_model_top_view=True,original_native_view_hw=[1080,1920],output_hw=[1080,2560],scoped_source_integrity=source_guard_receipt,current_source_points=points.packet_ref,sealed_evaluation_points=points.evaluation_ref,source_points_are_current_native_UV_only=True,no_historical_or_excluded_points_drawn=True,same_model_for_all_shots=True,acceleration=acceleration,continuous_visual_review_complete=False,source_accuracy_pass_claimed=False,metric_scale_verified=False,playback='PotPlayer local path only',elapsed_seconds=time.perf_counter()-started)
 write_json(out/'manifest.json',manifest);return manifest

def render_first4(model_file,scope_file,output,improvement_receipt,**kwargs):
 """Explicit S01–S04 consumer; the default whole-programme API is unchanged."""
 return render(model_file,scope_file,output,improvement_receipt,scope_mode='first4',**kwargs)

def render_first6(model_file,scope_file,output,improvement_receipt,**kwargs):
 """Explicit S01–S06 replay at the original continuous 50 Hz source clock."""
 return render(model_file,scope_file,output,improvement_receipt,scope_mode='first6',**kwargs)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--scope',required=True);p.add_argument('--out',required=True);p.add_argument('--improvement-receipt',required=True);p.add_argument('--compiled-dir');p.add_argument('--gpu-dll');p.add_argument('--gpu-validation');p.add_argument('--ffmpeg',default='ffmpeg');p.add_argument('--scope-mode',choices=['programme','first4','first6'],default='programme');a=p.parse_args();print(json.dumps(render(a.model,a.scope,a.out,a.improvement_receipt,compiled_dir=a.compiled_dir,gpu_dll=a.gpu_dll,gpu_validation=a.gpu_validation,ffmpeg=a.ffmpeg,scope_mode=a.scope_mode),ensure_ascii=False))
