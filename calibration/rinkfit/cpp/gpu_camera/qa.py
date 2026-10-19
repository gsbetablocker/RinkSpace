"""Fixed-theta lifecycle/boundary QA. Does not optimize or adopt parameters."""
from pathlib import Path
import hashlib,json,sys,numpy as np
P=Path(__file__).resolve().parents[4];R=Path('D:/CodexData/RinkSpace_Data/runs/okada_s1_r11');O=R/'gpu_accel/camera';sys.path.insert(0,str(P/'calibration/rinkfit/src'));sys.path.insert(0,str(R/'capture_cpu_i9/compiled_camera'))
from bound_camera_sensor import bind_camera_from_model
from rinkfit_start.gpu_camera import GPUCameraAt,validate_then_install,uninstall_camera_at
from rinkfit_start.native_camera import rotation
from scipy.optimize import brentq
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
CP=R/'physical_i20_captured_prepare/prepared/checkpoints/i00020_1792282640978672300/manifest.json';cp=json.loads(CP.read_text());ref=cp['model'];before={str(CP):sha(CP),ref['path']:sha(ref['path'])};checks=[]
c=bind_camera_from_model(ref);cpu=c.at;data=np.load(O/'fixed_inputs.npz');X=data['world'];t=data['time'];dll=O/'gpu_camera.dll';policy=O/'validation_enabled.json'
# Invalid policy must never retire or replace the original CPU context.
failed=dict(json.loads(policy.read_text()));failed['passed']=False;(O/'deliberately_invalid_validation.json').write_text(json.dumps(failed))
try:GPUCameraAt(c,dll,validation=O/'deliberately_invalid_validation.json');checks.append(dict(name='bad_policy_refused',passed=False))
except ValueError:checks.append(dict(name='bad_policy_refused',passed=bool(cpu.context)))
validate_then_install(c,dll,O/'deliberately_invalid_validation.json',ref);checks.append(dict(name='failed_auto_install_keeps_CPU',passed=c.at is cpu and bool(cpu.context)))
validate_then_install(c,dll,policy,ref);g=c.at;checks.append(dict(name='full_model_installed',passed=isinstance(g,GPUCameraAt)))
z=c.at(X[:4096],t[:4096]);c.at(X[:64],t[:64]);checks.append(dict(name='large_GPU_small_CPU_route',passed=g.stats['gpu_calls']==1 and g.stats['cpu_small_calls']==1,stats=dict(g.stats)))
address=int(g.context);capacity=g.capacity;c.at(X[:4096],t[:4096]);c.at(X[:64],t[:64]);checks.append(dict(name='persistent_context_and_capacity',passed=int(g.context)==address and g.capacity==capacity))
# Construct rays at the same fixed camera pose, solely to exercise original
# strict sensor/depth/Brown boundaries; there is no material station search.
g.min_batch=1;tm=float(t[10000]);abf=c.spline(tm);al,be,lf=abf;zz=lf-np.log(c.optics['f_ref']);RR=rotation(c.b,be)@c.R0@rotation(c.n,al);normal=np.cross(c.n,RR.T@c.b);normal/=np.linalg.norm(normal);lever=c.d.copy();lever[2]+=c.head.get('pupil_linear',0.)*zz+c.head.get('pupil_quadratic',0.)*zz*zz;C=c.P+RR.T@lever+c.a*normal
ray=lambda y:C+RR.T@np.array([0.,y,1.])
def cmp(name,world):
 count=g.stats['cpu_boundary_calls'];a=cpu.original_at(world,tm);b=g(world,tm);checks.append(dict(name=name,passed=all(np.array_equal(a[k],b[k],equal_nan=True)for k in a)and g.stats['cpu_boundary_calls']==count+1,original_boundary_fallback=g.stats['cpu_boundary_calls']==count+1))
for row in [0.,1079.]:
 y=brentq(lambda y:float(cpu.original_at(ray(y),tm)['uv'][1])-row,-float(c.optics['radius_max']),float(c.optics['radius_max']),xtol=1e-15);cmp('sensor_'+str(row),ray(y))
cmp('zero_depth',C);cmp('Brown_disk_edge',C+RR.T@np.array([float(c.optics['radius_max']),0.,1.]))
g.min_batch=4096
# Same physical model, deliberately different JSON serialization/SHA. This
# exercises new-SHA rebind and shadow with absolutely unchanged theta/S.
model=json.loads(Path(ref['path']).read_text());copy=O/'same_theta_reencoded.json';copy.write_text(json.dumps(model,indent=1));newref=dict(path=str(copy),sha256=sha(copy));assert newref['sha256']!=ref['sha256'];new=bind_camera_from_model(newref);validate_then_install(new,dll,policy,newref,output_dir=O/'runtime_shadow');ng=new.at;checks.append(dict(name='new_SHA_same_theta_shadow_and_new_context',passed=isinstance(ng,GPUCameraAt)and int(ng.context)!=address,detail=new.gpu_acceleration))
ng.close();retired=False
try:ng(X[:4096],t[:4096])
except RuntimeError:retired=True
checks.append(dict(name='retired_context_refused',passed=retired))
uninstall_camera_at(c);checks.append(dict(name='uninstall_restores_live_CPU',passed=c.at is cpu and bool(cpu.context)));cpu(X[:16],t[:16]);cpu.close()
after={str(CP):sha(CP),ref['path']:sha(ref['path'])};checks.append(dict(name='source_model_checkpoint_unchanged',passed=before==after));passed=all(x['passed']for x in checks)
report=dict(passed=passed,checks=checks,original_before=before,original_after=after,physically_identical_reencoding_only=True,no_S_capture_or_optimization=True);(O/'QA.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));raise SystemExit(0 if passed else 2)
