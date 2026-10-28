"""Transfer the saved secondary image fit to the selected square/centered model."""
from pathlib import Path
import json,numpy as np
import prepare_native_multicamera as p
import fit_native_s01_joint as single
from initialize_native_joint_principal_point import constant_principal_initial_state
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.domain.camera import complete_camera_rig
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline,refine_uniform_controls
from rinkspace.session.gpu_geometry import NativeSceneGPU,initialize_curve_stations


def initialize():
 m=p.read(p.RUN/'joint_manifest.json');z=p.npz(p.RUN/'point_observations.npz');mapping=p.read(p.RUN/'mapping.json');v=p.npz(p.RUN/'initial_pixels.npz')
 model=SceneParameterModel(p.read(m['scene_path']),Path(m['scene_path']),optimize_scene=True);p.validate_snapshot(v,model.initial_state.layout,mapping,m,z)
 core=NativeCore(m['native_library']);native=NativeSceneGPU(m['scene_path'],core,parameter_state=ParameterState(model.initial_state.layout,v['scene_values']))
 seed_path=Path(r'D:/CodexData/RinkSpace_Data/runs/native_points_s02_20261027/source_repair/identity_repair/navigation_camera.npz');seed=p.npz(seed_path)
 rig=np.r_[seed['rig'],v['rig'][1,38],np.zeros(4)];eta=rig[28];vertical=np.exp(-eta);fshift=eta/2
 rig[17]*=vertical;rig[19]*=vertical;rig[9:11]*=vertical;rig[9:12]/=np.linalg.norm(rig[9:12])
 r=rig[:9].reshape(3,3).copy();forward=r[2].copy();forward[2]*=vertical;forward/=np.linalg.norm(forward);right=r[0]-np.dot(r[0],forward)*forward;right/=np.linalg.norm(right)
 rig[:9]=np.array([right,np.cross(forward,right),forward]).ravel();axis=rig[:9].reshape(3,3)@rig[9:12];tilt=rig[12:15]-np.dot(rig[12:15],axis)*axis;rig[12:15]=tilt/np.linalg.norm(tilt)
 rig[24]*=np.exp(fshift);rig[38]+=fshift;rig[28]=0.;rig[37]=0.
 cams={s:p.read(p.HERE/('ad_cameras/s02.json' if s=='s02' else 'ad1_cameras/s06.json')) for s in ['s02','s06']};cc={}
 for shot,camera in cams.items():
  c=(seed['controls'] if shot=='s02' else np.load(p.HERE/'ad1_cameras'/camera['controls_ref'])).copy();c[:,1]*=vertical;c[:,2]+=fshift;cc[shot]=c
 scale=p.read(p.ROOT/'events/ankara_2026/model_constraints/ankara_2026_metric_scale_assumed_v1.json')['converted_span_xy_m'][0]
 lo,hi,envelope,range_path,fact=single.mount_configuration(scale,rig)
 c=cams['s02'];old_centre=core.evaluate_motion_camera(rig,cc['s02'],c['first_control_time'],c['control_dt'],[0.])['centre'][0]
 rig[18:21]=np.asarray(fact['working_ranges']['complete_tilt_axis_to_projection_centre_camera_m']['initial_reference'])/scale;rig[21]=np.clip(rig[21],lo[21],hi[21]);rig[22:24]=0.
 new_centre=core.evaluate_motion_camera(rig,cc['s02'],c['first_control_time'],c['control_dt'],[0.])['centre'][0];rig[15:18]+=old_centre-new_centre;rig[17]=np.clip(rig[17],lo[17],hi[17]);rig[16]=min(rig[16],-native.scene['board'][1]-.01/scale)
 before={k:a.copy() for k,a in v.items()};notes={}
 for shot in m['shots']:
  if shot['camera_index']!=1:continue
  camera=cams[shot['id']];fine,first=refine_uniform_controls(cc[shot['id']],camera['first_control_time'],camera['control_dt'],m['control_dt']);fine=fine[1:-1];first+=m['control_dt']
  moved,note=constant_principal_initial_state(core,dict(rig=rig,controls=fine),[959.5,539.5]);cs=slice(shot['control_begin'],shot['control_stop']);rs=slice(shot['row_begin'],shot['row_stop'])
  v['controls'][cs]=moved['controls'];v['rig'][1]=moved['rig'];camera={**camera,'rig':moved['rig'].tolist(),'first_control_time':first,'control_dt':m['control_dt']}
  print('INITIALIZE '+shot['id'],flush=True)
  v['stations'][rs],station_note=initialize_curve_stations(native,camera=camera,controls=moved['controls'],path_ids=[mapping['path_ids'][i] for i in z['path_indices'][rs]],stations=z['stations'][rs],pts=z['pts'][rs],observed=z['uv'][rs],kinds=z['kind'][rs],half_seconds=m['max_readout']/2)
  if station_note['unresolved']:raise ValueError('Unresolved native initial curve: '+str(station_note['unresolved'][:10]))
  notes[shot['id']]=dict(principal_transfer=note,stations=station_note)
 p.bind_snapshot(v,model.initial_state.layout,mapping,m);p.validate_snapshot(v,model.initial_state.layout,mapping,m,z)
 for shot in m['shots']:
  if shot['camera_index']==0:
   for key,a,b in [('controls',shot['control_begin'],shot['control_stop']),('stations',shot['row_begin'],shot['row_stop'])]:
    if not np.array_equal(v[key][a:b],before[key][a:b]):raise ValueError('Main baseline changed')
 if not np.array_equal(v['rig'][0],before['rig'][0]) or not np.array_equal(v['scene_values'],before['scene_values']):raise ValueError('Native scene/main rig changed')
 out=p.RUN/'secondary_seed.npz';np.savez(out,**v);p.write(out.with_suffix('.json'),dict(seed=str(seed_path),sha256=p.digest(seed_path),scene_values_from_secondary_seed_ignored=True,baseline_geometry_preserved=True,source_points_preserved=True,original_aspect=float(np.exp(eta)),camera_vertical_lift=float(vertical),new_camera= v['rig'][1].tolist(),notes=notes))
 print(str(out),flush=True)


if __name__=='__main__':initialize()
