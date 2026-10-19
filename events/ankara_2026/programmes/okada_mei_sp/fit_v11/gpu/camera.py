"""Event binding for the measured camera backend; original CPU remains owner."""
from pathlib import Path
import hashlib,importlib.util,json

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def install(camera,model_ref,output_dir,policy_path=None):
 policy_path=Path(policy_path)if policy_path else Path(__file__).with_name('compute.json')
 if not policy_path.exists():camera.gpu_acceleration=dict(enabled=False,reason='no_GPU_policy');return camera
 policy=json.loads(policy_path.read_text(encoding='utf8'));entry=policy['camera']
 if not entry.get('enabled'):camera.gpu_acceleration=dict(enabled=False,reason='GPU_camera_disabled');return camera
 try:
  for key in ['module','dll','validation']:
   d=entry[key]
   if sha(d['path'])!=d['sha256']:raise ValueError('GPU binding changed '+key)
  spec=importlib.util.spec_from_file_location('rinkfit_start.gpu_camera',entry['module']['path']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  camera=module.validate_then_install(camera,entry['dll']['path'],entry['validation']['path'],model_ref=model_ref,output_dir=Path(output_dir)/'gpu_camera_shadow')
  camera.gpu_acceleration['compute_policy']=dict(path=str(policy_path),sha256=sha(policy_path))
  return camera
 except (OSError,ValueError,RuntimeError)as exc:
  camera.gpu_acceleration=dict(enabled=False,reason='GPU_policy_failed_using_original_CPU',detail=str(exc));return camera
