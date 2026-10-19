"""Fixed-theta camera acceleration checks only; never captures/updates S or solves."""
from pathlib import Path
import argparse,hashlib,json,sys,time,numpy as np
P=Path(__file__).resolve().parents[4];sys.path.insert(0,str(P/'calibration/rinkfit/src'))
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--dll',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();R=a.run_root;O=a.output;O.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(R/'capture_cpu_i9/compiled_camera'))
from bound_camera_sensor import bind_camera_from_model
from rinkfit_start.native_scene import NativeScene,camera_from_event
from rinkfit_start.gpu_camera import GPUCameraAt,install_camera_at,uninstall_camera_at
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
CP=a.checkpoint;cp=json.loads(CP.read_text());MP=Path(cp['model']['path']);model=json.loads(MP.read_text());assert sha(MP)==cp['model']['sha256']
inputs=[CP,MP,R/'semantic_component_v4/manifest.json',R/'backend_input/pts.i64',R/'backend_input/uv.f64',R/'backend_input/budget.f64']
for d in cp['row_state_files'].values():
 f=CP.parent/d['file'];assert sha(f)==d['sha256'];inputs.append(f)
sem_path=R/'semantic_component_v4/manifest.json';sem=json.loads(sem_path.read_text())
def array(d,base=sem_path.parent):
 f=base/d['file'];assert sha(f)==d['sha256'];inputs.append(f);return np.fromfile(f,d['dtype']).reshape(d['shape'])
mi=array(sem['resolved_material_index']);active=array(sem['active_source_mask']).astype(bool);kind=array(sem['arrays']['kind']);assert len(mi)==182017 and np.flatnonzero(~active).tolist()==[25537]
s=np.fromfile(CP.parent/cp['row_state_files']['station']['file'],'<f8');br=np.fromfile(CP.parent/cp['row_state_files']['branch']['file'],'<u4');v=np.fromfile(CP.parent/cp['row_state_files']['row']['file'],'<f8');pts=np.fromfile(R/'backend_input/pts.i64','<i8');t=(pts-256256)/12800.
assert len(np.unique(pts))==1248
for ref in [sem['source_arrays'],sem['source_snapshot']]:
 f=Path(ref['path']);assert sha(f)==ref['sha256'];inputs.append(f)
inputs += [R/'capture_cpu_i9/compiled_camera/bound_camera_sensor.py',R/'capture_cpu_i9/compiled_camera/compiled_camera_guarded_sensor.py',R/'capture_cpu_i9/compiled_camera/camera_at.dll',R/'model_checks/gpu_full/exact_math.cuh',R/'model_checks/gpu_full/batch.cu',P/'calibration/rinkfit/src/rinkfit_start/native_camera.py',P/'calibration/rinkfit/src/rinkfit_start/gpu_camera.py',Path(__file__),*sorted((P/'calibration/rinkfit/cpp/gpu_camera').glob('*.cu')),*sorted((P/'calibration/rinkfit/cpp/gpu_camera').glob('*.cuh'))]
before={str(f):sha(f)for f in inputs};scene=NativeScene(model);X,finite,reason=scene.world_rows(mi,br,s)
np.savez(O/'fixed_inputs.npz',world=X,time=t,station=s,branch=br,row=v,pts=pts,active=active,kind=kind)
cam=bind_camera_from_model(cp['model']);base=cam.at;native=camera_from_event(model);gpu=GPUCameraAt(cam,a.dll,min_batch=1,benchmark_only=True)
checks=[];timings=[];saved={}
def compare(name,b,z):
 differences={};ok=True
 for k,x in b.items():
  if isinstance(x,dict):
   sub=compare(name+'.'+k,x,z[k]);differences[k]=sub;ok &=sub['passed'];continue
  xx=np.asarray(x);yy=np.asarray(z[k]);same_finite=bool(np.array_equal(np.isfinite(xx),np.isfinite(yy)))
  if xx.dtype.kind=='b':d=dict(exact=bool(np.array_equal(xx,yy)),different=int(np.count_nonzero(xx!=yy)));ok &=d['exact']
  elif xx.dtype.kind in 'fiu':
   finite_diff=np.isfinite(xx)&np.isfinite(yy);maximum=float(np.max(np.abs(xx[finite_diff]-yy[finite_diff]),initial=0.));d=dict(max_abs=maximum,same_finite_mask=same_finite,exact_bytes=xx.dtype==yy.dtype and xx.tobytes()==yy.tobytes());ok &=same_finite and maximum<=1e-8
  else:d=dict(equal=x==z[k]);ok &=d['equal']
  differences[k]=d
 return dict(name=name,passed=bool(ok),differences=differences)
def bench(name,xx,tt,repeats=10):
 r={}
 for label,fn in [('native_numpy',native.at),('bound_cpp',base),('gpu_persistent',gpu)]:
  fn(xx,tt);samples=[]
  for _ in range(repeats):
   st=time.perf_counter();fn(xx,tt);samples.append(time.perf_counter()-st)
  r[label]=dict(median_s=float(np.median(samples)),minimum_s=min(samples),samples_s=samples)
 r.update(name=name,shape=list(np.broadcast_shapes(np.asarray(xx).shape[:-1],np.asarray(tt).shape)),speedup_vs_bound_cpp=r['bound_cpp']['median_s']/r['gpu_persistent']['median_s'],speedup_vs_native=r['native_numpy']['median_s']/r['gpu_persistent']['median_s']);timings.append(r);print('BENCH',json.dumps(r),flush=True)
for label,shift in [('saved',cam.tau*(v/1079.-.5)),('top',np.full(len(v),-cam.tau/2)),('bottom',np.full(len(v),cam.tau/2)),('frame',np.zeros(len(v)))]:
 tt=t+shift;b=base(X,tt);z=gpu(X,tt);nn=native.at(X,tt);checks.extend([compare(label+'_cpp',b,z),compare(label+'_numpy',nn,z)])
 for k,x in z.items():saved[label+'_'+k]=x
 print('CHECK',label,checks[-2]['passed'],checks[-1]['passed'],flush=True)
np.savez(O/'full_raw_camera_outputs.npz',**saved)
for n in [1,16,64,256,1025,4096,16384,65536,len(v)]:bench('saved_'+str(n),X[:n],t[:n]+cam.tau*(v[:n]/1079.-.5),repeats=15 if n<20000 else 7)
ordinary=np.flatnonzero(active&(kind==0));assert len(ordinary)==179363
bench('ordinary_179363',X[ordinary],t[ordinary]+cam.tau*(v[ordinary]/1079.-.5),repeats=7)
# Recreate actual grid call shape at fixed legal geometry without selecting S.
grid=np.unique(np.r_[np.linspace(0.,1.,4097),s[:32]]);ii=ordinary[0];GX,_,_=scene.world_rows(np.full(len(grid),mi[ii]),np.full(len(grid),br[ii]),grid);GT=np.full(len(grid),t[ii]);bench('grid_4097_plus_saved',GX,GT,repeats=15);checks.append(compare('grid_at',base(GX,GT),gpu(GX,GT)))
rs=[]
for label,xx,tt in [('raw_182017',X,t),('grid',GX,GT),('4k',X[:4096],t[:4096])]:
 cam.at=base;st=time.perf_counter();b=cam.project_rs(xx,tt);ct=time.perf_counter()-st
 cam.at=gpu;st=time.perf_counter();z=cam.project_rs(xx,tt);gt=time.perf_counter()-st
 d=compare('trueRS_'+label,b,z);checks.append(d);rs.append(dict(name=label,CPU_seconds=ct,GPU_seconds=gt,speedup=ct/gt,valid_count=int(z['valid'].sum()),passed=d['passed']))
 np.savez(O/('trueRS_'+label+'.npz'),**{k:x for k,x in z.items()if isinstance(x,np.ndarray)})
 print('TRUERS',json.dumps(rs[-1]),flush=True)
cam.at=base
# Original exact boundary path, shapes, bad times, immutable state and retirement.
edge=[]
for name,xx,tt in [('scalar',X[0],t[0]),('broadcast',X[:16,None,:],t[:4][None,:]),('domain_start',X[:16],np.full(16,cam.domain[0])),('domain_end',X[:16],np.full(16,cam.domain[1])),('nan_world',np.full((16,3),np.nan),t[:16])]:edge.append(compare(name,base(xx,tt),gpu(xx,tt)))
for name,tt in [('outside',cam.domain[0]-.01),('nan_time',np.nan)]:
 aerr=berr=None
 try:base(X[0],tt)
 except Exception as e:aerr=type(e).__name__
 try:gpu(X[0],tt)
 except Exception as e:berr=type(e).__name__
 edge.append(dict(name=name,passed=aerr==berr and aerr is not None,CPU_exception=aerr,GPU_exception=berr))
old=cam.controls[10,0];cam.controls[10,0]=old+.01
try:gpu(X[:2],t[:2]);edge.append(dict(name='mutated_controls_refused',passed=False))
except ValueError:edge.append(dict(name='mutated_controls_refused',passed=True))
finally:cam.controls[10,0]=old
try:gpu.assert_bound_model(dict(path=cp['model']['path'],sha256='0'*64));edge.append(dict(name='wrong_SHA_refused',passed=False))
except ValueError:edge.append(dict(name='wrong_SHA_refused',passed=True))
checks.extend(edge)
checks.append(dict(name='GPU_actually_used_no_silent_error_fallback',passed=gpu.stats['gpu_calls']>0 and gpu.stats['cpu_error_calls']==0,stats=dict(gpu.stats)))
after={str(f):sha(f)for f in inputs};unchanged=before==after
# Activate only measured batch ranges with a sustained margin, including all
# larger measured shapes. Unknown model states require their own evidence.
ordered=[r for r in timings if r['name'].startswith('saved_')];threshold=None
for j,r in enumerate(ordered):
 if all(max(z['gpu_persistent']['samples_s'])<min(z['bound_cpp']['samples_s'])for z in [*ordered[j:],*timings[len(ordered):]]):threshold=int(r['name'].split('_')[-1]);break
passed=bool(all(z['passed']for z in checks)and unchanged and threshold is not None)
report=dict(schema='rinkfit.gpu_camera_validation.v1',passed=passed,min_batch=threshold,model_sha256=cp['model']['sha256'],checkpoint=dict(path=str(CP),sha256=sha(CP)),state_digest=gpu.state_digest,dll_sha256=sha(a.dll),implementation_sha256=sha(P/'calibration/rinkfit/src/rinkfit_start/gpu_camera.py'),numeric_equivalence_absolute_tolerance=1e-8,FP64_byte_equality_not_claimed=True,strict_valid_and_finite_status_equal=all(z['passed']for z in checks),checks=checks,timings=timings,trueRS=rs,stats=gpu.stats,capacity=gpu.capacity,immutable_input_hashes_before=before,immutable_input_hashes_after=after,inputs_unchanged=unchanged,no_S_capture_or_optimization=True,original_CPU_analytic_RS_certificate_retained=True)
(O/'validation.json').write_text(json.dumps(report,indent=2));print('REPORT',json.dumps({k:report[k]for k in ['passed','min_batch','model_sha256','state_digest','stats','inputs_unchanged']}),flush=True)
gpu.close()
if not passed:raise SystemExit(2)
