"""Raw-pixel four-shot / two-camera native joint LM."""
from pathlib import Path
import argparse,collections,ctypes,json,os,threading,time,traceback
import numpy as np
os.environ.setdefault('CUDA_PATH',r'D:/Program Files/CUDA')
os.environ.setdefault('RINKSPACE_SMALL_DUAL_BACKEND','cpu')
import prepare_native_multicamera as prepared
import fit_native_s01_joint as single
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline,continue_exposure_halos
from rinkspace.domain.camera_experiment import load_camera_experiment,build_camera_directions
from rinkspace.session.gpu_geometry import NativeSceneGPU,scene_coordinate_charts


def safe(x):
 if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [safe(v) for v in x]
 if isinstance(x,np.ndarray):return safe(x.tolist())
 if isinstance(x,(float,np.floating)) and not np.isfinite(x):return str(x)
 return x.item() if isinstance(x,np.generic) else x


def solve(args):
 root=prepared.RUN;m=prepared.read(root/'joint_manifest.json');z=prepared.npz(root/'point_observations.npz');mapping=prepared.read(root/'mapping.json')
 out=args.output;out.mkdir(parents=True,exist_ok=True);start=time.monotonic();log=(out/'solve.jsonl').open('a',encoding='utf-8',buffering=1);progress_log=(out/'fit_progress.log').open('a',encoding='utf-8',buffering=1)
 print('MAIN LOG: '+str(out/'solve.jsonl'),flush=True)
 def emit(event,**values):log.write(json.dumps(safe(dict(event=event,elapsed_s=time.monotonic()-start,**values)),ensure_ascii=False,allow_nan=False)+'\n')
 if prepared.digest(root/'point_observations.npz')!=m['point_input_sha256'] or prepared.digest(m['scene_path'])!=m['scene_file_sha256']:raise ValueError('Current input or native scene differs')
 if prepared.digest(m['native_library'])!=m['native_library_sha256']:raise ValueError('Selected solver library differs')
 v=prepared.npz(args.resume or root/'initial_pixels.npz');model=SceneParameterModel(prepared.read(m['scene_path']),Path(m['scene_path']),optimize_scene=True)
 prepared.validate_snapshot(v,model.initial_state.layout,mapping,m,z);state=ParameterState(model.initial_state.layout,v['scene_values']);core=NativeCore(m['native_library'])
 timeline=ShotMotionTimeline(m['shots'],m['control_dt'],m['first_control_time']);times=timeline.row_times(z['pts']);timeline.assert_support_separation(z['pts'],m['max_readout'])
 camera_ids=np.empty(len(z['pts']),np.int32)
 for s in m['shots']:camera_ids[s['row_begin']:s['row_stop']]=s['camera_index']
 selected_cameras={0,1} if args.phase=='joint' else {1}
 rows=np.flatnonzero(np.isin(camera_ids,list(selected_cameras)))
 paths=[mapping['path_ids'][i] for i in z['path_indices']];native=NativeSceneGPU(m['scene_path'],core,parameter_state=state)
 if args.phase in ('secondary_alignment','secondary_pose_alignment'):
  # Pose bootstrap while the newly supplied ice identities are being checked.
  # This selects factors for initialization only; the input is never rewritten.
  rows=rows[[native.paths[paths[i]]['layer'] in ('board','advertisement') for i in rows]]
 for shot in m['shots']:
  sl=slice(shot['control_begin'],shot['control_stop'])
  interval=[(shot[k]-shot['reference_pts'])*shot['time_base_num']/shot['time_base_den'] for k in ('first_pts','last_pts')]
  closed=continue_exposure_halos(v['controls'][sl],shot['first_control_time'],m['control_dt'],interval)
  if not np.allclose(closed,v['controls'][sl],rtol=2e-12,atol=2e-9):raise ValueError('Resume requires explicit endpoint-cubic exposure-halo initialization: '+shot['id'])
 motion,labels=timeline.directions(continue_endpoint_cubics=True);keep=[j for j,label in enumerate(labels) if next(s['camera_index'] for s in m['shots'] if s['id']==label[0]) in selected_cameras]
 motion=motion[:,:,keep];labels=[labels[j] for j in keep];nm=len(labels)
 profile=prepared.HERE/'camera_experiments/joint/square_center_pp_joint.json'
 rig=v['rig'].copy();indices=[]
 for i in range(2):
  rig[i],ix,experiment=load_camera_experiment(profile,rig[i])
  # Only a starting-pose solve: the joint phase always restores every chart
  # direction from the accepted square/centred-principal-point profile.
  if args.phase in ('secondary_pose','secondary_pose_alignment'):ix=(0,1,2,6,7,8)
  indices.append(ix if i in selected_cameras else ())
 nc=sum(map(len,indices));dependencies={b['id'] for b in state.blocks.values() if b['kind'] in ('board_shape','shared_geometry')}
 for path in set(paths):
  if '::' in path:dependencies.update(model.dependencies(path))
 active=state.active_columns(dependencies) if args.phase=='joint' else np.zeros(0,int)
 q=nm+nc+len(active);directions=np.zeros((len(v['controls']),3,q));directions[:,:,:nm]=.001*motion;del motion
 rd=np.zeros((62,q));sd=np.zeros((len(state.values),q));charts=scene_coordinate_charts(state)
 scales=np.r_[np.array([.001]*6+[.001]*3+[.00025]*3+[.00025]*3+[1.,1.,.001,1.,1.,.03,.03,.1,.1,.003,.003,.001]),.05,.1,.003,.003]
 offset=nm
 for i in range(2):
  if not len(indices[i]):continue
  local,_=build_camera_directions(rig[i],indices[i],scales,motion_columns=0,scene_columns=0)
  rd[i*31:(i+1)*31,offset:offset+len(indices[i])]=local;offset+=len(indices[i])
 for j,k in enumerate(active):
  b=next(b for b in state.blocks.values() if b['available'] and b['offset']<=k<b['offset']+b['size']);kind=b['kind']
  step=.01 if charts[k]>0 or kind=='print_size' else (.0002 if kind=='print_placement' and k-b['offset']<2 else (.001 if kind=='print_placement' else (.002 if kind.startswith('print_') else .001)))
  sd[k,nm+nc+j]=step
 metres=prepared.read(prepared.ROOT/'events/ankara_2026/model_constraints/ankara_2026_metric_scale_assumed_v1.json')['converted_span_xy_m'][0]
 bounds=[single.mount_configuration(metres,rig[i]) for i in range(2)]
 lower=np.concatenate([b[0] for b in bounds]);upper=np.concatenate([b[1] for b in bounds])
 for i in range(2):
  for k in (26,27,28,29,30):lower[i*43+k]=upper[i*43+k]=rig[i,k]
 rp=[paths[i] for i in rows];curve=np.flatnonzero(z['kind'][rows]==1);local=np.full(len(rows),-1,np.int32);local[curve]=np.arange(len(curve))
 periodic=np.array([native.paths[rp[i]]['kind']==7 for i in curve]);lo=np.where(periodic,-np.inf,0.);hi=np.where(periodic,np.inf,1.)
 activity=dict(phase=args.phase,rows=len(rows),all_rows=len(z['pts']),motion_columns=nm,camera_columns_per_rig=list(map(len,indices)),scene_columns=len(active),retained_columns=q,independent_camera_parameters=True,source_weight_fields=False,robust_loss=False,source_error_allowance=False,camera_model=experiment['profile'],conditional_metres_per_L=metres,numeric_ranges_approved_by_user=False,native_library=m['native_library'],point_input_sha256=m['point_input_sha256'],motion_labels=labels)
 activity['initialization_only']=args.phase!='joint'
 activity['active_camera_chart_indices_per_rig']=[list(ix) for ix in indices]
 activity['profile_trial_stations']=True
 activity['exposure_halos']='endpoint cubic continuation in both initial controls and every native update direction'
 prepared.write(out/'activity.json',safe(activity));emit('compile',**activity)
 camera={**prepared.read(m['cameras'][0]['camera_definition']),'first_control_time':timeline.first,'control_dt':timeline.dt}
 last_cost=None;last_summary=None
 def quality(errors,step,status,**extra):
  nonlocal last_cost,last_summary
  good=np.isfinite(errors);per={}
  for s in m['shots']:
   sel=(rows>=s['row_begin'])&(rows<s['row_stop']);e=errors[sel&good]
   if not sel.any():continue
   per[s['id']]=dict(rows=int(sel.sum()),invalid=int(np.count_nonzero(sel&~good)),cost=float(.5*np.dot(e,e)),rms=float(np.sqrt(np.mean(e*e))) if len(e) else None,median=float(np.median(e)) if len(e) else None,p95=float(np.percentile(e,95)) if len(e) else None,p99=float(np.percentile(e,99)) if len(e) else None,maximum=float(e.max()) if len(e) else None,above_20px=int(np.count_nonzero(e>20)))
  cost=float(.5*np.dot(errors[good],errors[good]));summary=dict(step=int(step),status=status,cost=cost,invalid=int(np.count_nonzero(~good)),shots=per,**extra)
  prepared.write(out/'current_quality.pending.json',safe(summary));(out/'current_quality.pending.json').replace(out/'current_quality.json')
  hot=np.flatnonzero(good&(errors>20));hot=hot[np.argsort(errors[hot])[::-1]];view=resident.evaluate_rows(hot.astype(np.int32)) if len(hot) else None
  observations=[]
  for j,k in enumerate(hot):
   i=int(rows[k]);observations.append(dict(row=i,observation_id=str(z['observation_ids'][i]),source_row=int(z['source_rows'][i]),pts=int(z['pts'][i]),camera_index=int(camera_ids[i]),path_id=paths[i],uv=z['uv'][i].tolist(),fitted_uv=view['values'][j,3:5].tolist(),error_px=float(errors[k])))
  prepared.write(out/'current_error_hotspots.pending.json',dict(**safe(summary),observations=observations));(out/'current_error_hotspots.pending.json').replace(out/'current_error_hotspots.json')
  progress_log.write(f"{int(step):5d} {time.monotonic()-start:10.3f} {cost:18.7f} {0. if last_cost is None else last_cost-cost:14.7f} {status} "+' '.join(f"{key}:rms={d['rms']:.4f},max={d['maximum']:.3f},gt20={d['above_20px']}" for key,d in per.items())+'\n');last_cost=cost
  last_summary=summary;emit('quality',**summary);return summary
 with native.resident(camera=camera,rigs=rig,camera_ids=camera_ids[rows],controls=v['controls'],path_ids=rp,stations=v['stations'][rows],pts=z['pts'][rows],observed=z['uv'][rows],scene_directions=sd,motion_times=times[rows]) as resident:
  resident.configure_camera_parameters(directions=rd,lower=lower,upper=upper,max_readout=m['max_readout'])
  errors=resident.evaluate_errors();initial=quality(errors,0,'seed')
  if args.evaluate_only:
   np.savez(out/'evaluation_pixels.npz',**resident.evaluate('cuda'),rows=rows);return
  if initial['invalid']:raise ValueError('Initial projection invalid; initialize it without excluding observations')
  for i in selected_cameras:
   resident.configure_mount_constraints(side='negative_y',camera_index=i)
   windows=timeline.frame_windows()[[j for j,s in enumerate(m['shots']) if s['camera_index']==i]]
   resident.configure_camera_envelope(**bounds[i][2],frame_time_intervals=windows,camera_index=i)
  emit('linearization',metrics=resident.configure_pixel_linearization(directions=directions,local_ids=local,local_scales=np.full(len(curve),.001)))
  class MemoryStatus(ctypes.Structure):
   _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
  memory=MemoryStatus();memory.length=ctypes.sizeof(memory);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
  emit('solver',metrics=resident.configure_solver(lower=lo,upper=hi,device_budget_bytes=resident.creation['free_after_alloc_bytes'],host_budget_bytes=memory.avail_phys,damping_metric='jacobi_image',initial_lambda=args.initial_lambda,profile_trial_stations=True))
  def checkpoint():
   current=resident.export_state();stations=v['stations'].copy();stations[rows]=current['stations'];current['stations']=stations;current['source_rows']=z['source_rows']
   prepared.bind_snapshot(current,state.layout,mapping,m);prepared.validate_snapshot(current,state.layout,mapping,m,z)
   if args.phase!='joint':
    if not np.array_equal(current['scene_values'],v['scene_values']) or not np.array_equal(current['rig'][0],v['rig'][0]):raise RuntimeError('Main-camera baseline changed during secondary initialization')
    for shot in m['shots']:
     if shot['camera_index']==0 and not np.array_equal(current['controls'][shot['control_begin']:shot['control_stop']],v['controls'][shot['control_begin']:shot['control_stop']]):raise RuntimeError('Main-camera motion changed during secondary initialization')
   np.savez(out/'accepted_pixels_pending.npz',**current);(out/'accepted_pixels_pending.npz').replace(out/'accepted_pixels_latest.npz');return current
  emit('initial_curve_search',metrics=resident.refine_curve_stations(mode='global'));checkpoint();quality(resident.evaluate_errors(),0,'initialized')
  stop=threading.Event()
  def watch():
   while not stop.wait(.25):
    if (out/'stop_requested').exists():resident.request_stop();return
  watcher=threading.Thread(target=watch,daemon=True);watcher.start()
  accepted=0
  def report(record):
   nonlocal accepted
   emit('trial',**record)
   if record['summary']['accepted']:
    accepted+=1
    checkpoint();quality(resident.evaluate_errors(),int(record['summary']['trial_count']),'accepted',curve_refinement='included_in_native_candidate_cost')
   else:
    progress_log.write(f"{int(record['summary']['trial_count']):5d} {time.monotonic()-start:10.3f} {last_cost:18.7f} {0.:14.7f} rejected "+' '.join(f"{key}:rms={d['rms']:.4f},max={d['maximum']:.3f},gt20={d['above_20px']}" for key,d in last_summary['shots'].items())+'\n')
   if args.trials is not None and record['summary']['trial_count']>=args.trials:return False
   return not (out/'stop_requested').exists()
  try:
   result=resident.solve(progress=report,gradient_tolerance=1e-6,step_tolerance=1e-6,relative_cost_tolerance=1e-10)
   current=checkpoint();qfinal=quality(resident.evaluate_errors(),int(result['trials']),'finished');np.savez(out/'evaluation_pixels.npz',**resident.evaluate('cuda'),rows=rows)
   prepared.write(out/'result.json',safe(dict(solver=result,quality=qfinal,accepted=accepted,activity=activity,visual_acceptance=False)));emit('finished',result=result,quality=qfinal)
  finally:stop.set();watcher.join();log.close();progress_log.close()


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['secondary_alignment','secondary_initialize','secondary_pose_alignment','secondary_pose','joint'],default='joint');p.add_argument('--resume',type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--evaluate-only',action='store_true');p.add_argument('--initial-lambda',type=float,default=0.);p.add_argument('--trials',type=int)
 a=p.parse_args()
 try:solve(a)
 except BaseException:
  a.output.mkdir(parents=True,exist_ok=True);(a.output/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
