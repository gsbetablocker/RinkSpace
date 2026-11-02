"""Raw-pixel joint LM for the shots and physical cameras in a prepared run."""
from pathlib import Path
import argparse,collections,ctypes,hashlib,json,os,threading,time,traceback
import numpy as np
os.environ.setdefault('CUDA_PATH',r'D:/Program Files/CUDA')
os.environ.setdefault('RINKSPACE_SMALL_DUAL_BACKEND','cpu')
os.environ.setdefault('RINKSPACE_STATIONARITY_BACKEND','cpu')
import prepare_native_multicamera as prepared
import fit_native_s01_joint as single
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline,continue_exposure_halos
from rinkspace.domain.camera_experiment import CAMERA_NAMES,_SCALAR_STORAGE_INDEX,load_camera_experiment,build_camera_directions
from rinkspace.session.gpu_geometry import NativeSceneGPU,scene_coordinate_charts
from rinkspace.domain.camera_coupling import read_camera_coupling,build_multicamera_directions


def safe(x):
 if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [safe(v) for v in x]
 if isinstance(x,np.ndarray):return safe(x.tolist())
 if isinstance(x,(float,np.floating)) and not np.isfinite(x):return str(x)
 return x.item() if isinstance(x,np.generic) else x


def _load_joint_camera_profile(path,rig):
 path=Path(path).resolve()
 copied,indices,metadata=load_camera_experiment(path,rig)
 if metadata['scene_mode']!='joint_current':raise ValueError('Multi-camera adapter requires camera profile scene_mode=joint_current')
 metadata['profile_path']=str(path)
 metadata['profile_file_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
 return copied,indices,metadata


def selected_scope(shots,camera_count,phase,camera_indices=None,shot_ids=None):
 if shot_ids is not None:
  if phase!='motion':raise ValueError('Shot-only scope is initialization-only and requires phase=motion')
  wanted=set(shot_ids);lookup={s['id']:s for s in shots}
  if not wanted or len(wanted)!=len(shot_ids) or not wanted<=lookup.keys():raise ValueError('Invalid explicit motion shot selection')
  cameras={lookup[s]['camera_index']for s in wanted}
  if camera_indices is not None and set(camera_indices)!=cameras:raise ValueError('Camera and shot initialization selections differ')
 else:
  cameras=set(camera_indices)if camera_indices is not None else(set(range(camera_count))if phase=='joint'else{1})
  wanted={s['id']for s in shots if s['camera_index']in cameras}
 if not cameras or not cameras<=set(range(camera_count)):raise ValueError('Selected camera indices are outside this prepared run')
 return cameras,wanted


def select_camera_parameter_names(indices,names):
 """Narrow existing camera charts without reordering or opening a direction."""
 if names is None:return indices
 wanted=set(names);unknown=wanted-set(CAMERA_NAMES)
 if unknown:raise ValueError('Unknown camera parameter names: '+str(sorted(unknown)))
 available={CAMERA_NAMES[k] for selected in indices for k in selected}
 inactive=wanted-available
 if inactive:raise ValueError('Requested camera parameters have no active profile/phase directions: '+str(sorted(inactive)))
 return [tuple(k for k in selected if CAMERA_NAMES[k] in wanted) for selected in indices]


def select_scene_block_ids(state,active,phase,block_ids):
 """Restrict the original dependent scene columns to explicit existing IDs."""
 if block_ids is None:return active
 if phase!='joint':raise ValueError('Scene block selection requires phase=joint')
 wanted=set(block_ids);unknown=wanted-state.blocks.keys()
 if unknown:raise ValueError('Unknown scene block IDs: '+str(sorted(unknown)))
 inactive=[identity for identity in sorted(wanted) if not np.isin(state.active_columns([identity]),active).any()]
 if inactive:raise ValueError('Requested scene blocks have no active dependency directions: '+str(inactive))
 return active[np.isin(active,state.active_columns(wanted))]


def validate_active_selection_checkpoint(current,initial,*,freeze_motion=False,frozen_scene_columns=(),frozen_camera_scalars=()):
 """Check explicitly frozen coordinates against the exact numeric seed."""
 def same_bytes(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
 if freeze_motion and not same_bytes(current['controls'],initial['controls']):raise RuntimeError('Frozen motion controls changed')
 columns=np.asarray(frozen_scene_columns,dtype=int)
 if len(columns) and not np.array_equal(current['scene_values'][columns],initial['scene_values'][columns]):raise RuntimeError('Unselected scene coordinates changed')
 for camera,names in enumerate(frozen_camera_scalars):
  columns=[_SCALAR_STORAGE_INDEX[name] for name in names]
  if not np.array_equal(current['rig'][camera,columns],initial['rig'][camera,columns]):raise RuntimeError('Unselected camera scalar coordinates changed: '+str(camera))


def solve(args):
 if getattr(args,'skip_initial_station_search',False) and not getattr(args,'stationarity_only',False):raise ValueError('Skipping initial station search is only supported for exact-state stationarity diagnostics')
 freeze_motion=getattr(args,'freeze_motion',False);camera_names=getattr(args,'camera_parameter_names',None);scene_ids=getattr(args,'scene_block_ids',None)
 if scene_ids is not None and args.phase!='joint':raise ValueError('Scene block selection requires phase=joint')
 has_active_selection=freeze_motion or camera_names is not None or scene_ids is not None
 start=time.monotonic()
 root=getattr(args,'run_root',None) or prepared.RUN;m=prepared.read(root/'joint_manifest.json');z=prepared.npz(root/'point_observations.npz');mapping=prepared.read(root/'mapping.json')
 working_envelope_path=prepared.camera_working_envelope_path(m)
 library_revision=None
 if getattr(args,'native_library',None):
  library_revision=dict(previous_library=m['native_library'],previous_sha256=m['native_library_sha256'],library=str(args.native_library),sha256=prepared.digest(args.native_library))
  m=dict(m,native_library=str(args.native_library),native_library_sha256=library_revision['sha256'])
 out=args.output;out.mkdir(parents=True,exist_ok=True);log=(out/'solve.jsonl').open('a',encoding='utf-8',buffering=1);progress_log=(out/'fit_progress.log').open('a',encoding='utf-8',buffering=1)
 print('MAIN LOG: '+str(out/'solve.jsonl'),flush=True)
 (out/'process_identity.json').write_text(json.dumps({'pid':os.getpid()}),encoding='utf8')
 def emit(event,**values):log.write(json.dumps(safe(dict(event=event,elapsed_s=time.monotonic()-start,**values)),ensure_ascii=False,allow_nan=False)+'\n')
 if prepared.digest(root/'point_observations.npz')!=m['point_input_sha256'] or prepared.digest(m['scene_path'])!=m['scene_file_sha256']:raise ValueError('Current input or native scene differs')
 if prepared.digest(m['native_library'])!=m['native_library_sha256']:raise ValueError('Selected solver library differs')
 v=prepared.npz(args.resume or root/'initial_pixels.npz');model=SceneParameterModel(prepared.read(m['scene_path']),Path(m['scene_path']),optimize_scene=True)
 prepared.validate_snapshot(v,model.initial_state.layout,mapping,m,z);state=ParameterState(model.initial_state.layout,v['scene_values']);core=NativeCore(m['native_library'])
 timeline=ShotMotionTimeline(m['shots'],m['control_dt'],m['first_control_time']);times=timeline.row_times(z['pts']);timeline.assert_support_separation(z['pts'],m['max_readout'])
 camera_ids=np.empty(len(z['pts']),np.int32)
 for s in m['shots']:camera_ids[s['row_begin']:s['row_stop']]=s['camera_index']
 camera_count=len(m['cameras'])
 requested=getattr(args,'camera_indices',None)
 selected_cameras,selected_shots=selected_scope(m['shots'],camera_count,args.phase,requested,getattr(args,'shot_ids',None))
 row_scope=np.zeros(len(z['pts']),bool)
 for shot in m['shots']:
  if shot['id']in selected_shots:row_scope[shot['row_begin']:shot['row_stop']]=True
 rows=np.flatnonzero(row_scope)
 paths=[mapping['path_ids'][i] for i in z['path_indices']];native=NativeSceneGPU(m['scene_path'],core,parameter_state=state)
 if args.phase in ('secondary_alignment','secondary_pose_alignment'):
  # Pose bootstrap while the newly supplied ice identities are being checked.
  # This selects factors for initialization only; the input is never rewritten.
  rows=rows[[native.paths[paths[i]]['layer'] in ('board','advertisement') for i in rows]]
 anchor_selection=None
 if args.phase=='camera_alignment':
  established={paths[i].split('::',1)[0] for i in np.flatnonzero(~np.isin(camera_ids,list(selected_cameras))) if '::' in paths[i]}
  prior_rows=rows
  rows=rows[[native.paths[paths[i]]['layer']=='board' or ('::' in paths[i] and paths[i].split('::',1)[0] in established) for i in rows]]
  if not len(rows):raise ValueError('New camera has no board or established advertisement anchors')
  anchor_selection=dict(basis='board_and_same_physical_print_instances_observed_by_existing_cameras',
   candidate_rows=len(prior_rows),initialization_rows=len(rows),other_rows_preserved_in_input=True,
   established_instances=sorted(established),geometry_frozen=True)
 source_selection=None
 if getattr(args,'source_selection',None):
  source_selection=prepared.read(args.source_selection)
  if source_selection.get('point_input_sha256')!=m['point_input_sha256']:raise ValueError('Source selection belongs to different observations')
  pending=np.zeros(len(z['pts']),bool);selection_receipts=[]
  for decision in source_selection['pending_materials']:
   shot=next(s for s in m['shots'] if s['id']==decision['shot'])
   path_index=mapping['path_ids'].index(decision['path_id'])
   mask=np.zeros(len(pending),bool);mask[shot['row_begin']:shot['row_stop']]=True
   mask&=z['path_indices']==path_index
   if not mask.any() or not decision.get('evidence'):raise ValueError('Pending material requires present source rows and explicit evidence')
   pending|=mask;selection_receipts.append(dict(**decision,rows=int(mask.sum()),source_frames=int(len(np.unique(z['pts'][mask])))))
  rows=rows[~pending[rows]]
  if not len(rows):raise ValueError('Source qualification removed every active factor')
  source_selection=dict(path=str(args.source_selection),sha256=prepared.digest(args.source_selection),
   pending_materials=selection_receipts,pending_rows=int(pending.sum()),all_source_rows_preserved=True,
   source_identity_resolved=False,full_input_accuracy_claim=False)
 for shot in m['shots']:
  sl=slice(shot['control_begin'],shot['control_stop'])
  interval=timeline.local_exposure_time(shot,np.array([shot['first_pts'],shot['last_pts']],np.int64))
  closed=continue_exposure_halos(v['controls'][sl],shot['first_control_time'],m['control_dt'],interval)
  if not np.allclose(closed,v['controls'][sl],rtol=2e-12,atol=2e-9):raise ValueError('Resume requires explicit endpoint-cubic exposure-halo initialization: '+shot['id'])
 motion_storage=os.environ.get('RINKSPACE_MOTION_DIRECTION_STORAGE','sparse')
 if motion_storage not in ('dense','sparse'):raise ValueError('Motion direction storage must be dense or sparse')
 motion,labels=timeline.directions(continue_endpoint_cubics=True,sparse=motion_storage=='sparse');keep=[j for j,label in enumerate(labels) if not freeze_motion and label[0] in selected_shots]
 motion=motion[:,keep] if motion_storage=='sparse' else motion[:,:,keep];labels=[labels[j] for j in keep]
 endpoint_zoom_shape_configuration=None
 if getattr(args,'endpoint_zoom_shape',None):
  from rinkspace.domain.motion_shape_anchor import retain_endpoint_zoom_shape
  motion,labels,endpoint_zoom_shape_configuration=retain_endpoint_zoom_shape(motion,labels,m['shots'],timeline.dt,m['max_readout'],args.endpoint_zoom_shape,v['controls'])
 nm=len(labels)
 parameterization=m.get('camera_parameterization',{})
 per_rig_names=parameterization.get('active_names_per_rig')
 if per_rig_names is not None:
  if not isinstance(per_rig_names,list) or len(per_rig_names)!=camera_count or any(not isinstance(names,list) or any(not isinstance(name,str) for name in names) or len(set(names))!=len(names) for names in per_rig_names):raise ValueError('Per-camera active names must give one distinct-name list per rig')

 profile=Path(getattr(args,'camera_profile',None) or parameterization.get('profile_path') or prepared.HERE/'camera_experiments/joint/square_center_pp_joint.json').resolve()
 if parameterization and prepared.digest(profile)!=parameterization['profile_sha256']:raise ValueError('Declared camera profile differs')
 coupling_path=getattr(args,'camera_coupling',None) or parameterization.get('coupling_path')
 coupling=read_camera_coupling(coupling_path) if coupling_path else None
 if parameterization.get('coupling_sha256') and prepared.digest(coupling_path)!=parameterization['coupling_sha256']:raise ValueError('Declared camera coupling differs')
 rig=v['rig'].copy();indices=[];profile_indices=[]
 for i in range(camera_count):
  rig[i],ix,experiment=_load_joint_camera_profile(profile,rig[i])
  if per_rig_names is not None:ix=select_camera_parameter_names([ix],per_rig_names[i])[0]
  profile_indices.append(ix)
  # Pose bootstraps use their existing pose-only chart; joint runs use all
  # active directions declared by the selected profile.
  if args.phase in ('secondary_pose','secondary_pose_alignment','camera_pose'):ix=tuple(k for k in (0,1,2,6,7,8) if per_rig_names is None or k in ix)
  if args.phase=='motion':ix=()
  indices.append(ix if i in selected_cameras else ())
 select_camera_parameter_names(profile_indices,camera_names)
 indices=select_camera_parameter_names(indices,camera_names)
 scales=np.r_[np.array([.001]*6+[.001]*3+[.00025]*3+[.00025]*3+[1.,1.,.001,1.,1.,.03,.03,.1,.1,.003,.003,.001]),.05,.1,.003,.003]
 camera_rd,camera_layout=build_multicamera_directions(rig,indices,scales,coupling)
 nc=camera_rd.shape[1];dependencies={b['id'] for b in state.blocks.values() if b['kind'] in ('board_shape','shared_geometry')}
 for path in set(paths):
  if '::' in path:dependencies.update(model.dependencies(path))
 active=state.active_columns(dependencies) if args.phase=='joint' else np.zeros(0,int)
 active=select_scene_block_ids(state,active,args.phase,scene_ids)
 charts=scene_coordinate_charts(state);ice_configuration=None;ice_chart_request=getattr(args,'ice_chart','auto')
 if args.phase=='joint' and ice_chart_request!='legacy' and hasattr(core._lib,'rs_gpu_ice_actual_chart_version') and m.get('scene_path'):
  from rinkspace.domain.ice_actual_chart import actual_ice_chart_configuration
  if native.scene.get('shared_geometry_options')==127:
   ice_configuration=actual_ice_chart_configuration(state.values,charts,active,relative_limit=native.ice_relative_limit)
 if ice_configuration is not None:charts=ice_configuration['charts'];active=ice_configuration['active']
 elif ice_chart_request=='actual':raise ValueError('Actual ice chart is unavailable or external nominal/deviation constraints require legacy coordinates')
 q=nm+nc+len(active)
 if q==0:raise ValueError('Active selection leaves no global parameter columns')
 frozen_scene_columns=np.setdiff1d(np.arange(len(state.values)),active) if scene_ids is not None else np.zeros(0,int)
 # Actual ice retractions may export new nominal witnesses even though their
 # direction columns were removed. They are derived state, not frozen values.
 if ice_configuration is not None:frozen_scene_columns=frozen_scene_columns[~np.isin(charts[frozen_scene_columns],(12,13))]
 frozen_camera_scalars=[[name for name in _SCALAR_STORAGE_INDEX if CAMERA_NAMES.index(name) not in selected] for selected in indices] if camera_names is not None or per_rig_names is not None else []
 if motion_storage=='sparse':
  from scipy.sparse import csr_matrix,hstack
  directions=hstack((.001*motion,csr_matrix((3*len(v['controls']),q-nm))),format='csr')
 else:
  directions=np.zeros((len(v['controls']),3,q));directions[:,:,:nm]=.001*motion
 del motion
 rd=np.zeros((31*camera_count,q));sd=np.zeros((len(state.values),q))
 rd[:,nm:nm+nc]=camera_rd
 for j,k in enumerate(active):
  b=next(b for b in state.blocks.values() if b['available'] and b['offset']<=k<b['offset']+b['size']);kind=b['kind']
  step=.01 if charts[k]>0 or kind=='print_size' else (.0002 if kind=='print_placement' and k-b['offset']<2 else (.001 if kind=='print_placement' else (.002 if kind.startswith('print_') else .001)))
  if ice_configuration is not None:step=ice_configuration['steps'].get(int(k),step)
  sd[k,nm+nc+j]=step
 metres=prepared.read(prepared.ROOT/'events/ankara_2026/model_constraints/ankara_2026_metric_scale_assumed_v1.json')['converted_span_xy_m'][0]
 bounds=[single.mount_configuration(metres,rig[i],working_envelope_path) for i in range(camera_count)]
 lower=np.concatenate([b[0] for b in bounds]);upper=np.concatenate([b[1] for b in bounds])
 for i in range(camera_count):
  for k in (26,27,28,29,30):lower[i*43+k]=upper[i*43+k]=rig[i,k]
 rp=[paths[i] for i in rows];curve=np.flatnonzero(z['kind'][rows]==1);local=np.full(len(rows),-1,np.int32);local[curve]=np.arange(len(curve))
 periodic=np.array([native.paths[rp[i]]['kind']==7 for i in curve]);lo=np.where(periodic,-np.inf,0.);hi=np.where(periodic,np.inf,1.)
 activity=dict(phase=args.phase,rows=len(rows),all_rows=len(z['pts']),motion_columns=nm,camera_columns_per_rig=list(map(len,indices)),scene_columns=len(active),retained_columns=q,independent_camera_parameters=camera_layout['independent_camera_parameters'],unique_camera_columns=nc,camera_column_layout=camera_layout,source_weight_fields=False,robust_loss=False,source_error_allowance=False,camera_model=experiment['profile'],conditional_metres_per_L=metres,numeric_ranges_approved_by_user=False,native_library=m['native_library'],point_input_sha256=m['point_input_sha256'],motion_labels=labels)
 activity['initialization_only']=args.phase!='joint' or source_selection is not None or has_active_selection
 if has_active_selection:
  activity['active_selection']=dict(freeze_motion=freeze_motion,camera_parameter_names=camera_names,scene_block_ids=scene_ids,
   active_camera_parameter_names_per_rig=[[CAMERA_NAMES[k] for k in selected] for selected in indices],
   frozen_camera_scalar_names_per_rig=frozen_camera_scalars,frozen_scene_columns=frozen_scene_columns,
   all_source_rows_preserved=True,all_prepared_rows_retained=len(rows)==len(z['pts']),
   raw_pixel_objective_unchanged=True,source_observations_or_weights_changed=False)
 activity['endpoint_zoom_shape_configuration']=endpoint_zoom_shape_configuration
 activity['source_selection']=source_selection
 activity['initialization_anchor_selection']=anchor_selection
 activity['selected_camera_indices']=sorted(selected_cameras)
 activity['selected_shot_ids']=[s['id']for s in m['shots']if s['id']in selected_shots]
 activity['active_camera_chart_indices_per_rig']=[list(ix) for ix in indices]
 if per_rig_names is not None:
  activity['declared_active_names_per_rig']=per_rig_names
  activity['structurally_frozen_camera_scalar_names_per_rig']=frozen_camera_scalars
 activity['camera_profile']=dict(path=experiment['profile_path'],sha256=experiment['profile_file_sha256'],profile=experiment['profile'])
 activity['camera_working_envelope']=dict(path=str(bounds[0][3]),sha256=prepared.digest(bounds[0][3]),explicit_binding=working_envelope_path is not None)
 activity['profile_trial_stations']=True
 activity['ice_chart']=dict(version='actual_x_log_radius_v1' if ice_configuration is not None else 'legacy',
  nominal_witness_coordinates_removed=6 if ice_configuration is not None else 0,
  stored_layout='layout146/options127',physical_tolerance=native.ice_relative_limit,raw_pixel_objective_unchanged=True)
 activity['native_library_revision']=library_revision
 activity['stationarity_backend']=os.environ.get('RINKSPACE_STATIONARITY_BACKEND','gpu')
 activity['backend_environment']={key:os.environ.get(key) for key in (
  'RINKSPACE_SMALL_DUAL_BACKEND','RINKSPACE_STATIONARITY_BACKEND','RINKSPACE_STATIONARITY_FACE_BACKEND',
  'RINKSPACE_STATIONARITY_CACHE','RINKSPACE_STATIONARITY_CPU_SPARSE','RINKSPACE_TRACE_DUAL_INPUT','RINKSPACE_TRACE_LM_PROJECTIONS')}
 activity['run_root']=str(root.resolve());activity['full_state']=str(args.resume) if args.resume else str(root/'initial_pixels.npz')
 activity['full_state_sha256']=prepared.digest(Path(activity['full_state']))
 activity['scene_sha256']=m['scene_file_sha256'];activity['native_library_sha256']=m['native_library_sha256']
 activity['output_mode']='compact_diagnostics' if getattr(args,'compact_diagnostics',False) else 'full_hotspots'
 activity['motion_direction_storage']=motion_storage
 activity['motion_direction_values']=directions.nnz if motion_storage=='sparse' else int(np.count_nonzero(directions))
 activity['motion_direction_bytes']=int(directions.data.nbytes+directions.indices.nbytes+directions.indptr.nbytes) if motion_storage=='sparse' else directions.nbytes
 activity['exposure_halos']='endpoint cubic continuation in both initial controls and every native update direction'
 prepared.write(out/'activity.json',safe(activity));emit('compile',**activity)
 camera={**prepared.read(m['cameras'][0]['camera_definition']),'first_control_time':timeline.first,'control_dt':timeline.dt}
 last_cost=None;last_summary=None
 def quality(errors,step,status,**extra):
  nonlocal last_cost,last_summary
  quality_started=time.monotonic()
  good=np.isfinite(errors);per={}
  for s in m['shots']:
   sel=(rows>=s['row_begin'])&(rows<s['row_stop']);e=errors[sel&good]
   if not sel.any():continue
   per[s['id']]=dict(rows=int(sel.sum()),invalid=int(np.count_nonzero(sel&~good)),cost=float(.5*np.dot(e,e)),rms=float(np.sqrt(np.mean(e*e))) if len(e) else None,median=float(np.median(e)) if len(e) else None,p95=float(np.percentile(e,95)) if len(e) else None,p99=float(np.percentile(e,99)) if len(e) else None,maximum=float(e.max()) if len(e) else None,above_20px=int(np.count_nonzero(e>20)))
  cost=float(.5*np.dot(errors[good],errors[good]));summary=dict(step=int(step),status=status,cost=cost,invalid=int(np.count_nonzero(~good)),shots=per,**extra)
  prepared.write(out/'current_quality.pending.json',safe(summary));(out/'current_quality.pending.json').replace(out/'current_quality.json')
  if getattr(args,'compact_diagnostics',False):
   # Retain every row's residual without serializing the same source pixels
   # and strings hundreds of thousands of times during camera bootstrap.
   np.savez_compressed(out/'current_residuals.pending.npz',rows=rows,errors=errors)
   (out/'current_residuals.pending.npz').replace(out/'current_residuals.npz')
   entity_ids=np.asarray(z['path_indices'][rows]);count=np.bincount(entity_ids[good],minlength=len(mapping['path_ids']))
   squared=np.bincount(entity_ids[good],weights=errors[good]**2,minlength=len(count))
   maximum=np.zeros(len(count));np.maximum.at(maximum,entity_ids[good],errors[good])
   entities=[dict(path_id=mapping['path_ids'][i],rows=int(count[i]),rms=float(np.sqrt(squared[i]/count[i])),maximum=float(maximum[i])) for i in np.flatnonzero(count)]
   detail=dict(schema='rinkspace.full_residual_array_and_entity_summary.v1',**safe(summary),
    input_path=str(root/'point_observations.npz'),input_sha256=m['point_input_sha256'],
    residuals=str(out/'current_residuals.npz'),all_rows_retained=True,entities=entities)
  else:
   hot=np.flatnonzero(good&(errors>20));hot=hot[np.argsort(errors[hot])[::-1]];view=resident.evaluate_rows(hot.astype(np.int32)) if len(hot) else None
   observations=[]
   for j,k in enumerate(hot):
    i=int(rows[k]);observations.append(dict(row=i,observation_id=str(z['observation_ids'][i]),source_row=int(z['source_rows'][i]),pts=int(z['pts'][i]),camera_index=int(camera_ids[i]),path_id=paths[i],uv=z['uv'][i].tolist(),fitted_uv=view['values'][j,3:5].tolist(),error_px=float(errors[k])))
   detail=dict(**safe(summary),observations=observations)
  prepared.write(out/'current_error_hotspots.pending.json',detail);(out/'current_error_hotspots.pending.json').replace(out/'current_error_hotspots.json')
  progress_log.write(f"{int(step):5d} {time.monotonic()-start:10.3f} {cost:18.7f} {0. if last_cost is None else last_cost-cost:14.7f} {status} "+' '.join(f"{key}:rms={d['rms']:.4f},max={d['maximum']:.3f},gt20={d['above_20px']}" for key,d in per.items())+'\n');last_cost=cost
  last_summary=summary;emit('quality',quality_wall_ms=1000*(time.monotonic()-quality_started),**summary);return summary
 with native.resident(camera=camera,rigs=rig,camera_ids=camera_ids[rows],controls=v['controls'],path_ids=rp,stations=v['stations'][rows],pts=z['pts'][rows],observed=z['uv'][rows],scene_directions=sd,scene_charts=charts,motion_times=times[rows]) as resident:
  resident.configure_camera_parameters(directions=rd,lower=lower,upper=upper,max_readout=m['max_readout'])
  errors=resident.evaluate_errors();initial=quality(errors,0,'seed')
  if args.evaluate_only and not getattr(args,'profile_evaluation_stations',False):
   np.savez(out/'evaluation_pixels.npz',**resident.evaluate('cuda'),rows=rows);return
  if initial['invalid']:raise ValueError('Initial projection invalid; initialize it without excluding observations')
  for i in selected_cameras:
   resident.configure_mount_constraints(side=m['cameras'][i].get('mount_side','negative_y'),camera_index=i)
   windows=timeline.frame_windows()[[j for j,s in enumerate(m['shots']) if s['camera_index']==i]]
   resident.configure_camera_envelope(**bounds[i][2],frame_time_intervals=windows,camera_index=i)
  emit('linearization',metrics=resident.configure_pixel_linearization(directions=directions,local_ids=local,local_scales=np.full(len(curve),.001)))
  del directions
  class MemoryStatus(ctypes.Structure):
   _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
  memory=MemoryStatus();memory.length=ctypes.sizeof(memory);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
  emit('solver',metrics=resident.configure_solver(lower=lo,upper=hi,device_budget_bytes=resident.creation['free_after_alloc_bytes'],host_budget_bytes=memory.avail_phys,damping_metric='jacobi_image',initial_lambda=args.initial_lambda,profile_trial_stations=True))
  def checkpoint():
   checkpoint_started=time.monotonic()
   current=resident.export_state()
   if endpoint_zoom_shape_configuration is not None:
    from rinkspace.domain.motion_shape_anchor import validate_endpoint_zoom_shape
    validate_endpoint_zoom_shape(current['controls'],endpoint_zoom_shape_configuration)
   stations=v['stations'].copy();stations[rows]=current['stations'];current['stations']=stations;current['source_rows']=z['source_rows']
   prepared.bind_snapshot(current,state.layout,mapping,m);prepared.validate_snapshot(current,state.layout,mapping,m,z)
   if coupling:build_multicamera_directions(current['rig'],indices,scales,coupling)
   if has_active_selection or per_rig_names is not None:validate_active_selection_checkpoint(current,{**v,'rig':rig},freeze_motion=freeze_motion,frozen_scene_columns=frozen_scene_columns,frozen_camera_scalars=frozen_camera_scalars)
   if args.phase!='joint' and not np.array_equal(current['scene_values'],v['scene_values']):raise RuntimeError('Frozen scene changed during camera initialization')
   if args.phase=='motion' and not np.array_equal(current['rig'],v['rig']):raise RuntimeError('Frozen physical cameras changed during motion initialization')
   for i in set(range(camera_count))-selected_cameras:
    if not np.array_equal(current['rig'][i],v['rig'][i]):raise RuntimeError('An unselected physical camera changed')
   for shot in m['shots']:
    if shot['id'] not in selected_shots and not np.array_equal(current['controls'][shot['control_begin']:shot['control_stop']],v['controls'][shot['control_begin']:shot['control_stop']]):raise RuntimeError('An unselected shot trajectory changed')
   np.savez(out/'accepted_pixels_pending.npz',**current);(out/'accepted_pixels_pending.npz').replace(out/'accepted_pixels_latest.npz')
   emit('checkpoint',checkpoint_wall_ms=1000*(time.monotonic()-checkpoint_started));return current
  def current_errors():
   errors_started=time.monotonic();result=resident.evaluate_errors()
   emit('errors',errors_wall_ms=1000*(time.monotonic()-errors_started));return result
  if getattr(args,'skip_initial_station_search',False):
   emit('initial_curve_search_skipped',reason='exact_state_stationarity_diagnostic')
  else:emit('initial_curve_search',metrics=resident.refine_curve_stations(mode='global'))
  initialized=checkpoint();np.savez(out/'initialized_pixels.npz',**initialized);quality(current_errors(),0,'initialized')
  if args.evaluate_only:
   # Compare the same source UV on their finite physical paths. Refitting
   # nuisance stations avoids treating a changed tangent coordinate as a
   # camera error; no global/camera/motion parameter is optimized here.
   for name in ('rig','controls','scene_values'):
    if not np.array_equal(initialized[name],v[name]):raise RuntimeError('Evaluation changed global parameters: '+name)
   np.savez(out/'evaluation_pixels.npz',**resident.evaluate('cuda'),rows=rows)
   emit('evaluation_only_finished',profiled_curve_stations=True,global_parameters_unchanged=True)
   log.close();progress_log.close();return
  if getattr(args,'stationarity_only',False):
   diagnostic=resident.stationarity_diagnostics(include_constraints=True)
   arrays={k:v for k,v in diagnostic.items() if isinstance(v,np.ndarray)}
   np.savez_compressed(out/'stationarity_arrays.npz',**arrays)
   summary={k:v for k,v in diagnostic.items() if k not in arrays}
   summary['arrays_path']=str(out/'stationarity_arrays.npz')
   prepared.write(out/'stationarity.json',safe(summary));emit('stationarity_only',diagnostic=summary)
   return
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
    checkpoint();quality(current_errors(),int(record['summary']['trial_count']),'accepted',curve_refinement='included_in_native_candidate_cost')
   else:
    progress_log.write(f"{int(record['summary']['trial_count']):5d} {time.monotonic()-start:10.3f} {last_cost:18.7f} {0.:14.7f} rejected "+' '.join(f"{key}:rms={d['rms']:.4f},max={d['maximum']:.3f},gt20={d['above_20px']}" for key,d in last_summary['shots'].items())+'\n')
   if args.trials is not None and record['summary']['trial_count']>=args.trials:return False
   return not (out/'stop_requested').exists()
  try:
   cost_tolerance=getattr(args,'relative_cost_tolerance',1e-10)
   if not np.isfinite(cost_tolerance) or cost_tolerance<=0:raise ValueError('Positive finite relative cost tolerance required')
   emit('continuous_solve_options',gradient_tolerance=1e-6,step_tolerance=1e-6,relative_cost_tolerance=cost_tolerance)
   result=resident.solve(progress=report,gradient_tolerance=1e-6,step_tolerance=1e-6,relative_cost_tolerance=cost_tolerance)
   current=checkpoint();qfinal=quality(current_errors(),int(result['trials']),'finished');final_output_started=time.monotonic()
   np.savez(out/'evaluation_pixels.npz',**resident.evaluate('cuda'),rows=rows);emit('final_evaluation_output',wall_ms=1000*(time.monotonic()-final_output_started))
   prepared.write(out/'result.json',safe(dict(solver=result,quality=qfinal,accepted=accepted,activity=activity,visual_acceptance=False)));emit('finished',result=result,quality=qfinal)
  finally:stop.set();watcher.join();log.close();progress_log.close()


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['secondary_alignment','secondary_initialize','secondary_pose_alignment','secondary_pose','camera_alignment','camera_initialize','camera_pose','motion','joint'],default='joint');p.add_argument('--resume',type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--evaluate-only',action='store_true');p.add_argument('--initial-lambda',type=float,default=0.);p.add_argument('--trials',type=int)
 p.add_argument('--run-root',type=Path);p.add_argument('--camera-indices',nargs='+',type=int)
 p.add_argument('--freeze-motion',action='store_true',help='Initialization only: freeze every existing motion control direction')
 p.add_argument('--camera-parameter-names',nargs='*',help='Initialization only: retain these already active camera names; an empty list freezes all camera directions')
 p.add_argument('--scene-block-ids',nargs='*',help='Joint initialization only: retain these active dependent scene block IDs; an empty list freezes the scene')
 p.add_argument('--compact-diagnostics',action='store_true')
 p.add_argument('--endpoint-zoom-shape',type=Path,help='Retain explicitly source-initialized endpoint zoom variation with a free level offset')
 p.add_argument('--shot-ids',nargs='+',help='Explicit shot motion initialization only; all rig/scene and other shots stay fixed')
 p.add_argument('--source-selection',type=Path)
 p.add_argument('--native-library',type=Path)
 p.add_argument('--camera-profile',type=Path,help='Explicit joint_current camera profile; default is square_center_pp_joint')
 p.add_argument('--camera-coupling',type=Path,help='Explicit shared scalar camera model; requires a compatible numerical seed')
 p.add_argument('--stationarity-only',action='store_true')
 p.add_argument('--skip-initial-station-search',action='store_true',help='Only with --stationarity-only: diagnose the exact supplied full state')
 p.add_argument('--ice-chart',choices=['auto','legacy','actual'],default='auto')
 p.add_argument('--profile-evaluation-stations',action='store_true')
 p.add_argument('--relative-cost-tolerance',type=float,default=1e-10)
 a=p.parse_args()
 try:solve(a)
 except BaseException:
  a.output.mkdir(parents=True,exist_ok=True);(a.output/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
