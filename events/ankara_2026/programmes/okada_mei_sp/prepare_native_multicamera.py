"""Prepare four shots without importing package geometry or camera parameters."""
from pathlib import Path
import argparse,collections,gzip,hashlib,io,json,zipfile,os
import numpy as np
os.environ.setdefault('CUDA_PATH',r'D:/Program Files/CUDA')
import prepare_native_joint_shots as old
import fit_native_s01_joint as single
from initialize_native_joint_principal_point import constant_principal_initial_state
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline,refine_uniform_controls
from rinkspace.domain.camera import complete_camera_rig
from rinkspace.session.gpu_geometry import NativeSceneGPU,initialize_curve_stations
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
RUN=Path(r'D:/CodexData/RinkSpace_Data/runs/native_joint_s01_s02_s04_s06_20261028')
OLD=old.RUN
BASE=OLD/'camera_models/center_pp_square/review/candidate_pixels.npz'
PACKAGES={
 's02':Path(r'D:/CodexData/RinkSpace_Data/runs/native_points_s02_20261027/source_repair/identity_repair/S02_native_points_identity_repaired_20261027.zip'),
 's06':Path(r'D:/CodexData/RinkSpace_Data/runs/native_points_s06_20261027/S06_native_points_20261027.zip')}
read=old.read;write=old.write;digest=old.digest;npz=old.npz
FIELDS=old.FIELDS


def snapshot_identity(layout,mapping,manifest):
 result=dict(schema='rinkspace.multicamera_joint_dependencies.v1',scene_revision=layout['scene_content_revision'],
  layout_revision=layout['layout_revision'],point_input_sha256=manifest['point_input_sha256'],
  paths_sha256=hashlib.sha256(json.dumps(mapping['path_ids'],separators=(',',':')).encode()).hexdigest(),
  shots=manifest['shots'],cameras=manifest['cameras'],control_dt=manifest['control_dt'],first_control_time=manifest['first_control_time'],max_readout=manifest['max_readout'])
 if 'camera_parameterization' in manifest:result['camera_parameterization']=manifest['camera_parameterization']
 return result


def bind_snapshot(v,layout,mapping,manifest):
 v['joint_numeric_dependencies_json']=np.array(json.dumps(snapshot_identity(layout,mapping,manifest),sort_keys=True,separators=(',',':')))
 v['point_input_sha256']=np.array(manifest['point_input_sha256']);v['native_library_provenance_sha256']=np.array(manifest['native_library_sha256'])
 return v


def validate_snapshot(v,layout,mapping,manifest,z):
 if json.loads(str(v['joint_numeric_dependencies_json']))!=snapshot_identity(layout,mapping,manifest):raise ValueError('Multi-camera state dependencies differ')
 if str(v['point_input_sha256'])!=manifest['point_input_sha256'] or not np.array_equal(v['source_rows'],z['source_rows']):raise ValueError('State point identity differs')
 for key,shape in dict(rig=(len(manifest['cameras']),43),controls=(manifest['shots'][-1]['control_stop'],3),stations=z['stations'].shape,scene_values=(layout['size'],)).items():
  if v[key].shape!=shape or not np.isfinite(v[key]).all():raise ValueError('Invalid state array: '+key)
 if not np.array_equal(v['stations'][z['kind']==0],z['stations'][z['kind']==0]):raise ValueError('A fixed material point changed')
 if np.any(v['rig'][:,28]!=0) or np.any(v['rig'][:,26:28]!=[959.5,539.5]) or np.any(v['rig'][:,29:31]!=0):raise ValueError('Square centered camera baseline changed')


def package_points(shot,path,native):
 with zipfile.ZipFile(path) as archive:
  raw=archive.read('points/point_observations.npz');z=dict(np.load(io.BytesIO(raw),allow_pickle=False));m=json.loads(archive.read('points/mapping.json'))
  if set(z)!=FIELDS:raise ValueError('Only seven plain point fields are permitted')
  if hashlib.sha256(raw).hexdigest()!=m['point_input_sha256']:raise ValueError('Package current input checksum differs')
  if m['scene_revision']!=native.scene_revision or set(m['path_ids'])-native.paths.keys():raise ValueError('Package native entity/model differs')
  if m['pixel_domain']!='native_distorted' or m['image_size']!=[1920,1080]:raise ValueError('Package pixel domain differs')
  if len(set(z['observation_ids']))!=len(z['pts']):raise ValueError('Duplicate package identity')
  a,b={'s02':(575744,720640),'s06':(1301248,1425408)}[shot]
  if not np.array_equal(np.unique(z['pts']),np.arange(a,b+1,256)):raise ValueError('Incomplete native shot clock')
  if not np.isfinite(z['uv']).all() or np.any(z['uv']<0) or np.any(z['uv']>[1919,1079]) or not np.isin(z['kind'],[0,1]).all():raise ValueError('Invalid package pixel/kind')
  index={str(k):i for i,k in enumerate(z['observation_ids'])};seen=set();families=collections.Counter();human=0
  records=archive.read('points/point_records.jsonl.gz')
  for line in gzip.decompress(records).splitlines():
   r=json.loads(line);i=index[r['observation_id']]
   if i in seen or int(z['pts'][i])!=r['pts'] or z['kind'][i]!=r['kind'] or not np.array_equal(z['uv'][i],r['uv']) or m['path_ids'][z['path_indices'][i]]!=r['native_path_id']:raise ValueError('Current source record differs from package point')
   if r.get('source_row',int(z['source_rows'][i]))!=int(z['source_rows'][i]):raise ValueError('Current source row differs')
   seen.add(i);families[r.get('review_order',{}).get('family','unspecified')]+=1;human+=bool(r.get('human_acceptance'))
  if len(seen)!=len(z['pts']):raise ValueError('Source records do not cover all supplied points')
  folder=RUN/('source_'+shot);folder.mkdir(exist_ok=True)
  (folder/'point_observations.npz').write_bytes(raw);write(folder/'mapping.json',m);(folder/'point_records.jsonl.gz').write_bytes(records)
  for name in ['remaining_issues.json','verification.json','final_source_audit.json']:
   if name in archive.namelist():(folder/name).write_bytes(archive.read(name))
  return z,m,dict(package=str(path),package_sha256=digest(path),rows=len(z['pts']),frames=len(np.unique(z['pts'])),source_records_checked=len(seen),families=dict(families),human_accepted_rows=human,package_geometry_imported=False,package_cameras_imported=False)


def prepare(library):
 RUN.mkdir(exist_ok=True)
 if (RUN/'joint_manifest.json').exists():raise FileExistsError('Prepared input exists')
 manifest=read(OLD/'joint_manifest.json');base=npz(BASE);za=npz(OLD/'point_observations.npz');ma=read(OLD/'mapping.json')
 model=SceneParameterModel(read(manifest['scene_path']),Path(manifest['scene_path']),optimize_scene=True)
 old.validate_joint_snapshot(base,model.initial_state.layout,ma,manifest,za)
 before={str(p):digest(p) for p in [BASE,OLD/'point_observations.npz',Path(manifest['scene_path'])]}
 core=NativeCore(library);native=NativeSceneGPU(manifest['scene_path'],core,parameter_state=ParameterState(model.initial_state.layout,base['scene_values']))
 packages={shot:package_points(shot,path,native) for shot,path in PACKAGES.items()}
 dt=manifest['control_dt'];cameras=[dict(id='main_long_side',camera_definition=manifest['shared_camera_definition']),dict(id='secondary_long_side',camera_definition=str(HERE/'ad_cameras/s02.json'))]
 sources={};shots=[];points=[];stations=[];controls=[];rigs=[base['rig'].copy()];audits={};seed_notes={};rbegin=cbegin=0
 secondary_cameras={shot:read(HERE/('ad_cameras/s02.json' if shot=='s02' else 'ad1_cameras/s06.json')) for shot in PACKAGES}
 original_secondary=complete_camera_rig(secondary_cameras['s02'])
 if len(original_secondary)==38:
  focal_controls=np.concatenate([np.load((HERE/('ad_cameras' if shot=='s02' else 'ad1_cameras'))/camera['controls_ref'])[1:-1,2] for shot,camera in secondary_cameras.items()])
  original_secondary=np.r_[original_secondary,np.median(focal_controls),np.zeros(4)]
 rig=original_secondary.copy()
 # Preserve the original optical centre when replacing the old metre-scale
 # lever decomposition with the existing labelled compact-tripod working seed.
 initial_controls=np.load(HERE/'ad_cameras'/secondary_cameras['s02']['controls_ref'])
 ref=core.evaluate_motion_camera(rig,initial_controls,secondary_cameras['s02']['first_control_time'],secondary_cameras['s02']['control_dt'],[0.])['centre'][0]
 scale=read(ROOT/'events/ankara_2026/model_constraints/ankara_2026_metric_scale_assumed_v1.json')['converted_span_xy_m'][0]
 lower,upper,envelope,range_path,range_fact=single.mount_configuration(scale,rig)
 rig[18:21]=np.asarray(range_fact['working_ranges']['complete_tilt_axis_to_projection_centre_camera_m']['initial_reference'])/scale
 rig[21]=np.clip(rig[21],lower[21],upper[21]);rig[22:24]=0.
 new=core.evaluate_motion_camera(rig,initial_controls,secondary_cameras['s02']['first_control_time'],secondary_cameras['s02']['control_dt'],[0.])['centre'][0]
 rig[15:18]+=ref-new;rig[17]=np.clip(rig[17],lower[17],upper[17]);rig[28]=0.;rig[37]=0. # global-exposure initialization; readout is fitted afterward
 for shot in ['s01','s02','s04','s06']:
  if shot in ('s01','s04'):
   oldshot=next(s for s in manifest['shots'] if s['id']==shot);rs=slice(oldshot['row_begin'],oldshot['row_stop']);cs=slice(oldshot['control_begin'],oldshot['control_stop'])
   z={k:v[rs].copy() for k,v in za.items()};m=ma;cc=base['controls'][cs].copy();ss=base['stations'][rs].copy();info={**oldshot,'camera_index':0};source=str(OLD/'point_observations.npz')
  else:
   z,m,audit=packages[shot];audits[shot]=audit;camera=secondary_cameras[shot]
   cc,first=refine_uniform_controls(np.load((HERE/('ad_cameras' if shot=='s02' else 'ad1_cameras'))/camera['controls_ref']),camera['first_control_time'],camera['control_dt'],dt)
   cc=cc[1:-1];first+=dt # discard outer coefficients outside source/readout support
   moved,note=constant_principal_initial_state(core,dict(rig=rig,controls=cc),[959.5,539.5]);cc=moved['controls'];sr=moved['rig']
   if len(rigs)==1:rigs.append(sr)
   elif not np.array_equal(rigs[1],sr):raise ValueError('Two secondary shots acquired distinct physical rigs')
   camera={**camera,'rig':sr.tolist(),'first_control_time':first,'control_dt':dt}
   print('INITIALIZE '+shot,flush=True)
   ss,station_note=initialize_curve_stations(native,camera=camera,controls=cc,path_ids=[m['path_ids'][i] for i in z['path_indices']],stations=z['stations'],pts=z['pts'],observed=z['uv'],kinds=z['kind'],half_seconds=.02)
   if station_note['unresolved']:raise ValueError('Native curve initialization unresolved: '+str(station_note['unresolved'][:10]))
   info=dict(id=shot,camera_index=1,reference_pts=camera['reference_pts'],time_base_num=1,time_base_den=12800,first_control_time=first,knot_stride=10,motion_reference_time=0.,first_pts=int(z['pts'].min()),last_pts=int(z['pts'].max()),stop_pts_exclusive=int(z['pts'].max()+256))
   seed_notes[shot]=dict(principal_transfer=note,stations=station_note,camera_definition_seed_only=str(cameras[1]['camera_definition']));source=str(RUN/('source_'+shot)/'point_observations.npz')
  info.update(row_begin=rbegin,row_stop=rbegin+len(z['pts']),control_begin=cbegin,control_stop=cbegin+len(cc));rbegin=info['row_stop'];cbegin=info['control_stop']
  sources[shot]=dict(path=source,sha256=digest(source));shots.append(info);points.append((z,m));stations.append(ss);controls.append(cc)
 paths=list(dict.fromkeys(path for z,m in points for path in m['path_ids']));lookup={p:i for i,p in enumerate(paths)}
 z={key:np.concatenate([v[key] for v,_ in points]) for key in FIELDS if key!='path_indices'}
 z['path_indices']=np.concatenate([np.array([lookup[m['path_ids'][i]] for i in v['path_indices']],np.int32) for v,m in points])
 if len(set(z['observation_ids']))!=len(z['pts']):raise ValueError('Cross-shot observation IDs collide')
 np.savez(RUN/'point_observations.npz',**z);mapping=dict(path_ids=paths,scene_revision=native.scene_revision);write(RUN/'mapping.json',mapping)
 timeline=ShotMotionTimeline(shots,dt);max_readout=timeline.max_readout();timeline.assert_support_separation(z['pts'],max_readout)
 manifest=dict(schema='rinkspace.multicamera_disjoint_shots.v1',shots=shots,cameras=cameras,control_dt=dt,first_control_time=0.,max_readout=max_readout,
  point_input_sha256=digest(RUN/'point_observations.npz'),scene_path=str(manifest['scene_path']),scene_file_sha256=digest(manifest['scene_path']),scene_revision=native.scene_revision,
  native_library=str(library),native_library_sha256=digest(library),source_inputs=sources,baseline_snapshot=str(BASE),baseline_sha256=digest(BASE),
  camera_model=dict(fx_over_fy=1,principal_point=[959.5,539.5],principal_zoom_slope=[0.,0.]),import_audit=audits,
  source_pts_unchanged=True,shot_motion_continuity_across_edits=False,independent_installations_and_optics=True,
  objective='0.5*sum((predicted_uv-original_uv)^2); every retained point contributes equally',small_constraint_dual_backend='cpu')
 v=dict(controls=np.concatenate(controls),rig=np.stack(rigs),scene_values=base['scene_values'].copy(),stations=np.concatenate(stations),source_rows=z['source_rows'])
 bind_snapshot(v,model.initial_state.layout,mapping,manifest);validate_snapshot(v,model.initial_state.layout,mapping,manifest,z)
 np.savez(RUN/'initial_pixels.npz',**v);write(RUN/'joint_manifest.json',manifest)
 for p,sha in before.items():
  if digest(p)!=sha:raise ValueError('Existing baseline or native authority was overwritten')
 write(RUN/'import_receipt.json',dict(rows=len(z['pts']),shots=shots,protected_before=before,protected_after={p:digest(p) for p in before},baseline_scene_values_exactly_preserved=True,baseline_shot_arrays_preserved=True,only_current_seven_fields=True,weighted_or_robust_objective=False,seed_notes=seed_notes,secondary_rig_original=original_secondary.tolist(),secondary_rig_initial=rigs[1].tolist(),secondary_mount_ranges_are_existing_assistant_working_estimates=True))
 print(json.dumps(dict(rows=len(z['pts']),frames=len(np.unique(z['pts'])),initial=str(RUN/'initial_pixels.npz')),ensure_ascii=False),flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);prepare(p.parse_args().library)
