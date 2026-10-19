"""Current continuation adapter: verified CUDA UV transport, same CPU fit logic.
This entry does not start on import. Invocation remains a caller decision.
"""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,os

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def verify(d):
 p=Path(d['path']).resolve()
 if sha(p)!=d['sha256']:raise ValueError('Changed GPU continuation authority '+str(p))
 return p

def run(checkpoint,context,output,stop_file=None):
 policy_path=Path(__file__).with_name('compute.json');policy=read(policy_path)
 if policy['schema']!='rinkfit.okada_verified_GPU_compute_policy.v1'or not policy['whole_uv']['enabled']:raise ValueError('Whole UV GPU backend is not enabled')
 validation=read(verify(policy['whole_uv']['validation']))
 if not validation['passed']or not validation['complete_state_unchanged']or validation['end_to_end_speedup']<=1:raise ValueError('No verified GPU transport benefit')
 resident=policy['whole_uv']['resident_transport'];recipe=read(verify(resident['recipe']))
 if recipe['backend_validation']!=policy['whole_uv']['validation']or recipe['validated_backend_build']!=validation['build']:raise ValueError('GPU validation and exact backend build differ')
 verify(validation['build']);verify(recipe['executable'])
 for file,digest in recipe['sources'].items():
  if sha(file)!=digest:raise ValueError('Changed frozen GPU worker source '+file)
 if (validation['raw_rows'],validation['active_rows'],validation['profiles'],validation['original_W_rank'])!=(182017,182016,961,214112):raise ValueError('GPU source validation is incomplete')
 D=Path(output).resolve();D.mkdir(exist_ok=False);C=read(context);base=read(verify(C['source_input']))
 # Keep original native CPU helpers and all fitting stages. The new resident
 # worker replaces only raw UV partials without losing dataset residency.
 base['resident_transport']=copy.deepcopy(resident)
 C['preparer']=policy['preparer_driver'];verify(C['preparer'])
 ip=D/'gpu_source_input.json';ip.write_text(json.dumps(base,indent=2),encoding='utf8')
 C['source_input']=ref(ip);ct=D/'gpu_context.json';ct.write_text(json.dumps(C,indent=2),encoding='utf8')
 parent=verify(policy['continuation_prepare_parent']);module_spec=importlib.util.spec_from_file_location('verified_continuation_prepare',parent);module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
 (D/'GPU_execution.json').write_text(json.dumps(dict(entry=ref(__file__),policy=ref(policy_path),context=ref(context),checkpoint=ref(checkpoint),parent=ref(parent),only_compute_backend_changed=True),indent=2),encoding='utf8')
 old_path=os.environ.get('PATH','');os.environ['PATH']=';'.join(recipe['runtime_bin_paths'])+';'+old_path
 try:result=module.run(checkpoint,ct,D/'runtime',stop_file)
 finally:os.environ['PATH']=old_path
 result['GPU_compute_policy']=ref(policy_path);(D/'prepared.json').write_text(json.dumps(result,indent=2),encoding='utf8');return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--context',required=True);p.add_argument('--output',required=True);p.add_argument('--stop-file');a=p.parse_args();run(a.checkpoint,a.context,a.output,a.stop_file)
