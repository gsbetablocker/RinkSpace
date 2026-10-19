"""Persistent FP64 acceleration for an explicitly SHA-bound existing CPU camera.

Only measured, model-specific batch sizes can be installed for automatic routing.
The original CPU/NumPy boundary predicates and analytic RS certificates remain
in charge. Rebinding creates a new device context; controls are uploaded once.
"""
from __future__ import annotations
import ctypes
import hashlib
import json
from pathlib import Path
import threading
import numpy as np

_D=ctypes.POINTER(ctypes.c_double)
_B=ctypes.POINTER(ctypes.c_uint8)
def _ptr(x):return x.ctypes.data_as(_D)
def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _head(c):return np.r_[c.R0.reshape(-1),c.n,c.b,c.P,c.d,c.a,c.head.get('pupil_linear',0.),c.head.get('pupil_quadratic',0.)].astype('<f8')
def _optics(c):
 o=c.optics;k1=list(o['k1'])+[0.];k2=list(o['k2'])+[0.]
 if len(o['k1'])>2 or len(o['k2'])>2:raise ValueError('GPU camera requires the existing affine zoom optics')
 return np.array([o['f_ref'],o['radius_max'],*o['principal_point'],o.get('eta',0.),*o.get('principal_linear',[0.,0.]),k1[0],k1[1],k2[0],k2[1],o['p1'],o['p2']],dtype='<f8')

class GPUCameraAt:
 def __init__(self,camera,dll,*,validation=None,min_batch=None,benchmark_only=False):
  self.context=None;self._owns_cpu=False;self.camera=camera;self.cpu_at=camera.at;self._lock=threading.RLock();self.stats=dict(gpu_calls=0,gpu_points=0,cpu_small_calls=0,cpu_boundary_calls=0,cpu_time_calls=0,cpu_error_calls=0)
  if not hasattr(self.cpu_at,'assert_bound_model'):raise TypeError('An existing SHA-bound CPU camera is required')
  binding=self.cpu_at.assert_bound_model();self.model_reference=dict(binding['model']);self.state_digest=self.cpu_at.state_digest
  self.payload={k:v.copy()for k,v in self.cpu_at.payload.items()};self.basis=dict(self.cpu_at.basis);self.bound_state_digest=self.state_digest
  self.dll=Path(dll).resolve();self.dll_sha256=_sha(self.dll)
  if validation is None and not benchmark_only:raise ValueError('A completed current-model speed/accuracy validation is required for automatic routing')
  self.validation=None
  if validation is not None:
   self.validation=json.loads(Path(validation).read_text());v=self.validation
   if not v.get('passed')or v.get('model_sha256')!=self.model_reference['sha256']or v.get('state_digest')!=self.state_digest or v.get('dll_sha256')!=self.dll_sha256:raise ValueError('GPU validation does not bind this exact model, camera state and DLL')
   self.min_batch=int(v['min_batch'])
  else:self.min_batch=int(min_batch or 1)
  if min_batch is not None and validation is not None and int(min_batch)<self.min_batch:raise ValueError('Requested GPU threshold is below the measured safe threshold')
  if min_batch is not None:self.min_batch=int(min_batch)
  self._model_stat=self._stat()
  self.lib=ctypes.CDLL(str(self.dll));lib=self.lib
  lib.rf_gpu_camera_error.argtypes=[];lib.rf_gpu_camera_error.restype=ctypes.c_char_p
  lib.rf_gpu_camera_create.argtypes=[_D,_D,_D,ctypes.c_int,ctypes.c_double,ctypes.c_double];lib.rf_gpu_camera_create.restype=ctypes.c_void_p
  lib.rf_gpu_camera_destroy.argtypes=[ctypes.c_void_p];lib.rf_gpu_camera_destroy.restype=None
  lib.rf_gpu_camera_at.argtypes=[ctypes.c_void_p,ctypes.c_uint64,_D,_D,_D,_B];lib.rf_gpu_camera_at.restype=ctypes.c_int
  lib.rf_gpu_camera_capacity.argtypes=[ctypes.c_void_p];lib.rf_gpu_camera_capacity.restype=ctypes.c_uint64
  self.context=lib.rf_gpu_camera_create(_ptr(self.payload['head']),_ptr(self.payload['optics']),_ptr(self.payload['controls']),len(camera.controls),camera.first,camera.dt)
  if not self.context:raise RuntimeError(lib.rf_gpu_camera_error().decode())
 def _stat(self):
  s=Path(self.model_reference['path']).stat();return s.st_size,s.st_mtime_ns
 def _assert_immutable(self):
  if not self.context or not self.cpu_at.context:raise RuntimeError('GPU/CPU camera context is retired; explicitly bind the current model again')
  c=self.camera
  if (self.cpu_at.state_digest!=self.state_digest or c.first!=self.basis['first_control_time']or c.dt!=self.basis['dt']or list(c.domain)!=self.basis['domain']or c.tau!=self.basis['signed_readout_seconds']or not np.array_equal(c.controls,self.payload['controls'])or not np.array_equal(_head(c),self.payload['head'])or not np.array_equal(_optics(c),self.payload['optics'])):raise ValueError('Camera parameters changed after GPU model binding; rebuild from the current exact model')
  if self._stat()!=self._model_stat:
   if _sha(self.model_reference['path'])!=self.model_reference['sha256']:raise ValueError('GPU camera model SHA256 mismatch; rebind required')
   self._model_stat=self._stat()
 def assert_bound_model(self,reference=None):
  self._assert_immutable();return self.cpu_at.assert_bound_model(reference)
 def snapshot(self,out,bindings=None):return self.cpu_at.snapshot(out,bindings)
 def close(self):
  with self._lock:
   if self.context:self.lib.rf_gpu_camera_destroy(self.context);self.context=None
   if self._owns_cpu and hasattr(self,'cpu_at'):self.cpu_at.close();self._owns_cpu=False
 def __del__(self):
  try:self.close()
  except Exception:pass
 @property
 def capacity(self):return int(self.lib.rf_gpu_camera_capacity(self.context))
 def __call__(self,world,time):
  with self._lock:return self._call(world,time)
 def _call(self,world,time):
  self._assert_immutable();X=np.asarray(world,float);t=np.asarray(time,float);shape=np.broadcast_shapes(X.shape[:-1],t.shape);n=int(np.prod(shape,dtype=int))if shape else 1
  if n<self.min_batch:self.stats['cpu_small_calls']+=1;return self.cpu_at(world,time)
  X=np.ascontiguousarray(np.broadcast_to(X,shape+(3,)),dtype='<f8');t=np.ascontiguousarray(np.broadcast_to(t,shape),dtype='<f8')
  if np.any(~np.isfinite(t))or np.any(t<self.camera.domain[0])or np.any(t>self.camera.domain[1]):self.stats['cpu_time_calls']+=1;return self.cpu_at(world,time)
  out=np.empty((n,9),'<f8');flags=np.empty(n,'u1')
  code=self.lib.rf_gpu_camera_at(self.context,n,_ptr(X),_ptr(t),_ptr(out),flags.ctypes.data_as(_B))
  if code or np.any(flags>1):
   self.stats['cpu_error_calls']+=1;return self.cpu_at(world,time)
  eps=np.finfo(float).eps;r2=out[:,8]*out[:,8];limit=float(self.camera.optics['radius_max'])**2
  brown=np.abs(r2-limit)<=512*eps*(1+np.abs(r2)+limit)
  scale=1+np.max(np.abs(X),axis=-1).reshape(-1)+float(np.max(np.abs(self.camera.P)))+float(np.max(np.abs(self.camera.d)))
  depth=np.abs(out[:,2])<=512*eps*scale
  sensor=np.minimum(np.abs(out[:,1]),np.abs(out[:,1]-1079.))<=512*eps*(1+np.abs(out[:,1])+1079.)
  if np.any(brown|depth|sensor):
   self.stats['cpu_boundary_calls']+=1;return self.cpu_at.original_at(world,time)
  self.stats['gpu_calls']+=1;self.stats['gpu_points']+=n
  return dict(uv=out[:,:2].reshape(shape+(2,)),valid=flags.astype(bool).reshape(shape),depth=out[:,2].reshape(shape),camera_xyz=out[:,3:6].reshape(shape+(3,)),logf=out[:,6].reshape(shape),z=out[:,7].reshape(shape),ray_radius=out[:,8].reshape(shape))

def install_camera_at(camera,dll,*,validation,min_batch=None):
 """Install only after a model/state/DLL-bound benchmark passed; CPU stays alive."""
 adapter=GPUCameraAt(camera,dll,validation=validation,min_batch=min_batch);adapter._owns_cpu=True;camera.at=adapter;return camera

def uninstall_camera_at(camera):
 """Restore the still-bound CPU route and retire only device resources."""
 adapter=camera.at
 if isinstance(adapter,GPUCameraAt):
  with adapter._lock:
   if adapter.context:adapter.lib.rf_gpu_camera_destroy(adapter.context);adapter.context=None
   camera.at=adapter.cpu_at
   adapter._owns_cpu=False
 return camera


def validate_then_install(camera,dll,existing_validation,model_ref=None,*,output_dir=None):
 """Rebind each new theta and run a deterministic same-theta CPU shadow check.

 The seed report must be a complete measured validation of this backend/DLL.
 A different model gets 256 deterministic camera-domain witnesses across its
 time allocation, plus immutable state/SHA checks. This is runtime backend
 evidence, not full-raw acceptance of that new theta. Final CPU certification
 and acceptance remain unchanged. Failure leaves the existing CPU route alive.
 """
 if isinstance(camera.at,GPUCameraAt):
  camera.at.assert_bound_model(model_ref);return camera
 cpu=camera.at
 if not hasattr(cpu,'assert_bound_model'):raise TypeError('A SHA-bound CPU camera is required')
 binding=cpu.assert_bound_model(model_ref);seed_path=Path(existing_validation);seed=json.loads(seed_path.read_text())
 if not seed.get('passed')or seed.get('dll_sha256')!=_sha(dll)or seed.get('implementation_sha256')!=_sha(__file__):
  camera.gpu_acceleration=dict(enabled=False,reason='backend_validation_missing_or_changed');return camera
 if seed.get('model_sha256')==binding['model']['sha256']and seed.get('state_digest')==cpu.state_digest:
  install_camera_at(camera,dll,validation=existing_validation);camera.gpu_acceleration=dict(enabled=True,validation=str(seed_path),evidence='full_fixed_model');return camera
 adapter=None
 try:
  from .native_camera import rotation
  # Four rays per time cover front, behind-camera and Brown-domain-invalid
  # outputs. Time points stay inside allocated support, including endpoints.
  times=np.repeat(np.linspace(camera.domain[0],camera.domain[1],64),4)
  abf=camera.spline(times);alpha,beta,logf=abf.T;z=logf-np.log(float(camera.optics['f_ref']))
  R=rotation(camera.b,beta)@camera.R0@rotation(camera.n,alpha)
  normal=np.cross(camera.n,np.einsum('...ji,j->...i',R,camera.b));normal/=np.linalg.norm(normal,axis=1)[:,None]
  lever=np.broadcast_to(camera.d,(256,3)).copy();lever[:,2]+=camera.head.get('pupil_linear',0.)*z+camera.head.get('pupil_quadratic',0.)*z*z
  C=camera.P+np.einsum('...ji,...j->...i',R,lever)+camera.a*normal
  radius=float(camera.optics['radius_max']);rays=np.tile(np.array([[0.,0.,1.],[.4*radius,.2*radius,1.],[1.1*radius,0.,1.],[.1,.1,-1.]]),(64,1))
  world=C+np.einsum('...ji,...j->...i',R,rays)
  adapter=GPUCameraAt(camera,dll,min_batch=1,benchmark_only=True);a=cpu(world,times);b=adapter(world,times)
  differences={};passed=adapter.stats['gpu_calls']>0 and adapter.stats['cpu_error_calls']==0
  for key,x in a.items():
   y=b[key];finite=np.isfinite(x);same_finite=np.array_equal(finite,np.isfinite(y))
   if x.dtype.kind=='b':same=bool(np.array_equal(x,y));differences[key]=dict(exact=same);passed &=same
   else:
    delta=float(np.max(np.abs(x[finite]-y[finite]),initial=0.));differences[key]=dict(max_abs=delta,same_finite_mask=bool(same_finite));passed &=same_finite and delta<=float(seed['numeric_equivalence_absolute_tolerance'])
  adapter.assert_bound_model(model_ref)
  report=dict(schema='rinkfit.gpu_camera_runtime_shadow.v1',passed=bool(passed),model_sha256=binding['model']['sha256'],state_digest=cpu.state_digest,dll_sha256=adapter.dll_sha256,implementation_sha256=_sha(__file__),min_batch=seed['min_batch'],numeric_equivalence_absolute_tolerance=seed['numeric_equivalence_absolute_tolerance'],shadow_points=256,shadow_time_points=64,full_raw_current_theta_acceptance=False,seed_validation=dict(path=str(seed_path),sha256=_sha(seed_path)),differences=differences,stats=dict(adapter.stats),original_CPU_final_certificate_required=True)
  out=Path(output_dir)if output_dir is not None else seed_path.parent/'runtime_shadow'
  if 'OneDrive'in str(out):raise ValueError('Runtime shadow evidence must be written in Data')
  out.mkdir(parents=True,exist_ok=True);dest=out/(binding['model']['sha256']+'.json');dest.write_text(json.dumps(report,indent=2))
  if not passed:
   adapter.close();camera.gpu_acceleration=dict(enabled=False,reason='runtime_shadow_failed',validation=str(dest));return camera
  adapter.min_batch=int(seed['min_batch']);adapter.validation=report;adapter._owns_cpu=True;camera.at=adapter;camera.gpu_acceleration=dict(enabled=True,validation=str(dest),evidence='same_backend_current_theta_256_point_shadow');return camera
 except (ValueError,RuntimeError,OSError)as exc:
  if adapter is not None:adapter.close()
  camera.gpu_acceleration=dict(enabled=False,reason='runtime_shadow_exception',detail=str(exc));return camera
