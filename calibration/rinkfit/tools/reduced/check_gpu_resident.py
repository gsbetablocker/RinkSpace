"""Fixed-checkpoint CPU/GPU resident transport comparison. No fitting."""
from pathlib import Path
import argparse,json,sys,time
from check_gpu_transport import sha,ref,read,verify,compare_fixed_outputs
from rinkfit_start.resident_transport import ResidentTransportClient

def run(cp,gpu,backend_build,validation,out,cpu):
 cp=Path(cp).resolve();gpu=Path(gpu).resolve();out=Path(out).resolve();out.mkdir(exist_ok=False);doc=read(cp)
 vd=read(validation)
 if not vd['passed']or vd['build']!=ref(Path(backend_build)/'recipe.json'):raise ValueError('Current full CPU/GPU backend validation required')
 recipe=read(gpu/'recipe.json');recipe.update(backend_validation=ref(Path(validation)),validated_backend_build=vd['build'],same_exact_projection_RS_profile_W_math=True)
 rows=read(verify(read(Path(backend_build)/'recipe.json')['rows_recipe']))
 for path,digest in rows['inputs'].items():
  if sha(path)!=digest:raise ValueError('Changed CUDA source '+path)
  recipe['sources'][path]=digest
 (gpu/'recipe.json').write_text(json.dumps(recipe,indent=2),encoding='utf8')
 paths=[cp,verify(doc['model']),verify(doc['linearization'])]+[verify(d,cp.parent)for d in doc['row_state_files'].values()];before={str(p):sha(p)for p in paths};runs=[]
 for name,build in [('cpu',Path(cpu)),('gpu',gpu)]:
  begin=time.perf_counter()
  with ResidentTransportClient(build,out/(name+'_owner'))as client:
   result=client.transport('whole_uv',cp,out/name)
  wall=time.perf_counter()-begin
  receipt=read(verify(result['receipt']));report=read(verify(result['manifest']))
  runs.append(dict(backend=name,wall_seconds=result['request_seconds'],cold_including_dataset_initialization_seconds=wall,client_request_seconds=result['client_wall_seconds'],report=result['manifest'],operator_seconds=report['elapsed_seconds'],complete_receipt=result['receipt']))
  print(json.dumps(runs[-1]),flush=True)
 result=compare_fixed_outputs(cp,gpu,out,runs,before,paths,doc)
 result['schema']='rinkfit.full_fixed_state_resident_GPU_transport_check.v1';result['warm_timing_excludes_once_per_worker_dataset_initialization']=True;result['CPU_worker_recipe']=ref(Path(cpu)/'recipe.json');result['fresh_theta_and_4aux_each_request']=True
 (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf8');return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--gpu',required=True);p.add_argument('--backend-build',required=True);p.add_argument('--validation',required=True);p.add_argument('--out',required=True);p.add_argument('--cpu',required=True);a=p.parse_args();run(a.checkpoint,a.gpu,a.backend_build,a.validation,a.out,a.cpu)
