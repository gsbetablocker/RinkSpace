"""Reuse the existing SHA-bound camera ABI for a joint read-only renderer.

A shot camera comes from CameraGraph, preserving its reference-angle rotation.
No event-view camera conversion, fitted values or source measurements are used.
"""
from pathlib import Path
import sys,time
import numpy as np
from .io_utils import file_sha256,write_json
from rinkfit_start.native_camera import NativeCamera

def install(graph,model_file,output,world,compiled_dir,*,gpu_dll=None,gpu_validation=None,shots=None):
 directory=Path(compiled_dir).resolve();out=Path(output);out.mkdir(exist_ok=False)
 for f in ('bound_camera_sensor.py','compiled_camera_guarded_sensor.py','camera_at.dll'):
  if not (directory/f).is_file():raise ValueError('Existing compiled ABI is missing: '+str(directory/f))
 sys.path.insert(0,str(directory))
 from bound_camera_sensor import BoundCompiledAt
 reference=dict(path=str(Path(model_file).resolve()),sha256=file_sha256(model_file))
 import json
 model=json.loads(Path(model_file).read_text(encoding='utf8'));reports=[]
 selected=list(graph.graph['shots']) if shots is None else list(shots)
 if len(selected)!=len(set(selected)) or set(selected)-set(graph.graph['shots']):raise ValueError('Unknown or duplicated acceleration shot')
 for sid in selected:
  camera=graph.native_camera(sid);original=NativeCamera.at.__get__(camera,NativeCamera)
  # Three source exposures and the complete same-model path inventory test the
  # active display domain, cut-back reference angles and off-screen predicates.
  scope=graph.graph['shots'][sid]['scope'];duration=(scope['last_pts']-scope['first_pts'])/12800.
  adapter=BoundCompiledAt(camera,reference,model);camera.at=adapter
  start=time.perf_counter();witness=[];passed=True
  for t in (0.,duration/2,duration):
   a=original(world,t);b=camera.at(world,t)
   flags=bool(np.array_equal(a['valid'],b['valid']));finite=np.isfinite(a['uv']).all(axis=-1)&np.isfinite(b['uv']).all(axis=-1)
   display=finite&a['valid']&(a['uv'][:,1]>=-1)&(a['uv'][:,1]<=1080)&(a['uv'][:,0]>=-1920)&(a['uv'][:,0]<=3840)
   delta=np.max(np.abs(a['uv'][display]-b['uv'][display]),initial=0.)
   witness.append(dict(local_seconds=t,points=len(world),display_domain_points=int(display.sum()),valid_flags_exact=flags,maximum_display_uv_difference_px=float(delta)))
   passed &= flags and delta<=1e-6
  adapter.assert_bound_model(reference)
  if not passed:camera.at=original;adapter.close()
  gpu=None
  if passed and gpu_dll is not None:
   if gpu_validation is None:raise ValueError('GPU requires an existing completed backend validation')
   from rinkfit_start.gpu_camera import validate_then_install
   validate_then_install(camera,gpu_dll,gpu_validation,reference,output_dir=out/sid/'gpu_shadow')
   gpu=getattr(camera,'gpu_acceleration',None)
  reports.append(dict(shot_id=sid,camera_state_digest=adapter.state_digest,compiled_installed=bool(passed),witnesses=witness,display_uv_tolerance_px=1e-6,seconds=time.perf_counter()-start,gpu=gpu,shot_reference_angles_preserved=True))
 receipt=dict(schema='rinkfit.joint9.read_only_camera_acceleration.v1',model=reference,compiled_abi={f:dict(path=str(directory/f),sha256=file_sha256(directory/f)) for f in ('bound_camera_sensor.py','compiled_camera_guarded_sensor.py','camera_at.dll')},shots=reports,all_CPU_shadows_passed=all(r['compiled_installed'] for r in reports),source_pixel_accuracy_claim=False,changes_fitted_parameters=False)
 write_json(out/'manifest.json',receipt);return receipt
