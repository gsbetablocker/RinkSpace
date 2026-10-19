"""Compare one selected-source fixed-state GPU transport against its actual CPU result."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,time,sys
import numpy as np
from scipy import sparse
P=Path(__file__).resolve().parents[4];sys.path.insert(0,str(P/'calibration/rinkfit/src'))
from rinkfit_start.target_prepare import selected_scope
from rinkfit_start.resident_scope import ResidentTransportClient

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def verify(d,base=None):
 p=Path(d.get('path',d.get('file')));p=p if p.is_absolute()else Path(base)/p
 if sha(p)!=d['sha256']:raise ValueError('Changed authority '+str(p))
 return p

def compare_fixed_outputs(cp,build,out,runs,before,paths,doc):
 a,b=[read(Path(x['report']['path']))for x in runs];comparisons={};failures=[];n=doc['raw_rows'];active_rows=doc['active_rows'];profiles=a['profile_count'];rank=a['full_W_rank'];cpu=Path(runs[0]['report']['path']).parent;gpu=Path(runs[1]['report']['path']).parent
 for k in ['checkpoint','model','packet','semantic','raw_rows','active_rows','profile_count','entity_count','full_W_rank','statistical_partition_exactly_once','original_W_half_cost','groups']:
  if a[k]!=b[k]:failures.append('metadata '+k)
 for family in ['arrays']:
  for k,da in a[family].items():
   db=b[family][k];pa=verify(da,cpu);pb=verify(db,gpu)
   if da['dtype']!=db['dtype']or da['shape']!=db['shape']:failures.append('shape '+family+'/'+k);continue
   x=np.memmap(pa,mode='r',dtype=da['dtype']);y=np.memmap(pb,mode='r',dtype=db['dtype']);same=sha(pa)==sha(pb)
   maximum=0.;count=0
   if not same:
    if x.dtype.kind!='f':count=int(np.count_nonzero(x!=y))
    else:
     # Preserve the pre-existing 2e-8 relative/absolute derivative gate.
     # CPU primal and every W/statistical array remain byte identical.
     for start in range(0,len(x),1000000):
      xx=x[start:start+1000000];yy=y[start:start+1000000];finite=np.isfinite(xx)&np.isfinite(yy);mask=(np.isnan(xx)&np.isnan(yy))|(xx==yy)
      delta=np.abs(xx[finite]-yy[finite])/(1+np.abs(xx[finite]));maximum=max(maximum,float(delta.max(initial=0)));count+=int(np.count_nonzero(~finite&~mask))+int(np.count_nonzero(delta>2e-8))
     if family=='arrays'and k not in ['station_UV_right','station_UV_left','profile_implicit_rcond']:count=max(1,count)
   if count:failures.append(family+'/'+k)
   comparisons[family+'/'+k]=dict(bitwise_equal=same,max_normalized_error=maximum,over_original_gate=count,elements=len(x))
 # A derivative exactly zero in one backend can be nonzero by rounding in
 # the other. Compare their full source/block/component union without dropping
 # small entries or changing the established derivative tolerance.
 def csr(report,base):
  d=report['physical_UV_CSR']
  def load(key):return np.fromfile(verify(d[key],base),dtype=d[key]['dtype'])
  offset=load('row_offsets');block=load('block_id');component=load('component');values=load('values')
  if len(offset)!=2*n+1 or offset[0]!=0 or offset[-1]!=len(values) or np.any(offset[1:]<offset[:-1]):raise ValueError('CSR full raw source identity invalid')
  if len(block)!=len(values)or len(component)!=len(values)or np.any(component>=32)or np.any(block>=len(read(verify(doc['model']))['blocks'])):raise ValueError('CSR column identity invalid')
  columns=block.astype(np.int64)*32+component
  matrix=sparse.csr_matrix((values,columns,offset),shape=(2*n,len(read(verify(doc['model']))['blocks'])*32))
  if not np.isfinite(matrix.data).all():raise ValueError('Nonfinite original UV derivative')
  sp=verify(a['semantic']);sm=read(sp);ad=sm['active_source_mask'];mask=np.fromfile(verify(ad,sp.parent),dtype=ad['dtype']).astype(bool)
  if np.any(np.diff(offset).reshape(n,2)[~mask]!=0):raise ValueError('Excluded source row reentered')
  return matrix
 A=csr(a,cpu);B=csr(b,gpu);delta=(B-A).tocsr();delta.eliminate_zeros();rowids=np.repeat(np.arange(delta.shape[0],dtype=np.int32),np.diff(delta.indptr));av=np.asarray(A[rowids,delta.indices]).reshape(-1)
 norm=np.abs(delta.data)/(1+np.abs(av));count=int(np.count_nonzero(norm>2e-8));maximum=float(norm.max(initial=0))
 comparisons['physical_UV_CSR/full_source_block_component_union']=dict(cpu_nnz=A.nnz,gpu_nnz=B.nnz,different_entries=delta.nnz,max_normalized_error=maximum,over_original_gate=count,no_small_derivative_discard=True)
 if count:failures.append('physical_UV_CSR/full_source_block_component_union')
 after={str(p):sha(p)for p in paths}
 if before!=after:failures.append('state changed')
 result=dict(schema='rinkfit.full_fixed_state_GPU_transport_check.v1',passed=not failures,no_Solve=True,no_station_refinement=True,no_checkpoint_written=True,complete_state_unchanged=before==after,checkpoint=ref(cp),build=ref(build/'build_recipe.json'),raw_rows=n,active_rows=active_rows,profiles=profiles,original_W_rank=rank,original_derivative_gate=2e-8,runs=runs,end_to_end_speedup=runs[0]['wall_seconds']/runs[1]['wall_seconds'],comparisons=comparisons,failures=failures)
 (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items()if k!='comparisons'}),flush=True)
 if failures:raise RuntimeError('Fixed-state CPU/GPU mismatch; do not enable')
 return result


def check(checkpoint,build,cpu_report,output):
 cp,B,O=Path(checkpoint).resolve(),Path(build).resolve(),Path(output).resolve();O.mkdir(exist_ok=False)
 recipe=read(B/'build_recipe.json');doc=read(cp);lm=read(verify(doc['linearization']));scope=selected_scope(verify(lm['packet']),verify(lm['semantic']))
 for f,digest in recipe['source_inputs'].items():
  if sha(f)!=digest:raise ValueError('Changed GPU build source '+f)
 paths=[cp,verify(doc['model']),verify(doc['linearization']),verify(lm['packet']),verify(lm['semantic'])]
 paths.extend(verify(d,cp.parent)for d in doc['row_state_files'].values());before={str(p):sha(p)for p in paths}
 client=ResidentTransportClient(B,O/'owner');client._scope=scope;expected,data=client._transport_state(cp)
 env={k.upper():v for k,v in os.environ.items()};env['PATH']=';'.join(recipe['runtime_bin_paths'])+';'+env.get('PATH','');env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 proc=subprocess.Popen([str(verify(recipe['executable']))],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 logfile=(O/'worker.log').open('x',encoding='utf8')
 def exchange(request):
  proc.stdin.write(json.dumps(request)+'\n');proc.stdin.flush()
  while True:
   line=proc.stdout.readline()
   if not line:raise RuntimeError('GPU worker closed without response')
   logfile.write(line);logfile.flush()
   if line.startswith('RESIDENT_RESPONSE '):
    response=json.loads(line[len('RESIDENT_RESPONSE '):])
    if not response.get('ok'):raise RuntimeError(str(response))
    return response
 try:
  pk=read(verify(lm['packet']));q=pk['query_packet'];init=exchange(dict(op='init',packet=lm['packet'],semantic=lm['semantic'],query=dict(path=q['manifest'],sha256=q['sha256']),reader_policy_sha256=recipe['reader_header_sha256'],output=str(O/'dataset')))['result']
  if init['raw_rows']!=scope['raw_rows'] or init['active_rows']!=scope['active_rows'] or not init['complete_source_validation_performed']:raise ValueError('GPU source init incomplete')
  begin=time.perf_counter();response=exchange(dict(op='whole_uv',input=ref(cp),output=str(O/'gpu'),authority=init['authority'],expected_state=expected));elapsed=time.perf_counter()-begin;r=response['result']
  if r['state']!=expected or r['authority']!=init['authority']:raise ValueError('GPU request identity mismatch')
  cpu=read(cpu_report);gpu=read(verify(r['report']));runs=[dict(backend='cpu',wall_seconds=cpu['elapsed_seconds'],report=ref(cpu_report),operator_seconds=cpu['elapsed_seconds']),dict(backend='gpu',wall_seconds=elapsed,report=r['report'],operator_seconds=gpu['elapsed_seconds'])]
 finally:
  try:
   if proc.poll()is None:exchange(dict(op='close'));proc.stdin.close();proc.wait()
  finally:
   if proc.poll()is None:proc.kill();proc.wait()
   logfile.close()
 result=compare_fixed_outputs(cp,B,O,runs,before,paths,doc)
 result['selected_target_scope']=scope;result['comparison_scope']='same complete model and four auxiliary arrays; no root/J result reused';(O/'report.json').write_text(json.dumps(result,indent=2),encoding='utf8')
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--build',required=True);p.add_argument('--cpu-report',required=True);p.add_argument('--output',required=True);a=p.parse_args();check(a.checkpoint,a.build,a.cpu_report,a.output)
