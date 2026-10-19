"""Compare the whole fixed-state CUDA operator against CPU; never Solve."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,time
import numpy as np
from scipy import sparse

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def ref(p):return dict(path=str(p),sha256=sha(p))
def verify(d,base=None):
 p=Path(d.get('path',d.get('file')));p=p if p.is_absolute()else Path(base)/p
 if sha(p)!=d['sha256']:raise ValueError('Changed authority '+str(p))
 return p

def check(cp,build,out):
 cp=Path(cp).resolve();build=Path(build).resolve();out=Path(out).resolve();out.mkdir(exist_ok=False)
 recipe=read(build/'recipe.json');doc=read(cp)
 rows_recipe=read(verify(recipe['rows_recipe']))
 for file,digest in rows_recipe['inputs'].items():
  if sha(file)!=digest:raise ValueError('Changed GPU row implementation '+file)
 paths=[cp,verify(doc['model']),verify(doc['linearization'])]
 for d in doc['row_state_files'].values():paths.append(verify(d,cp.parent))
 lm=read(paths[2]);paths.extend([verify(lm['packet']),verify(lm['semantic'])]);before={str(p):sha(p)for p in paths}
 for p,h in recipe['source_inputs'].items():
  if sha(p)!=h:raise ValueError('Changed operator source '+p)
 env={k.upper():v for k,v in os.environ.items()};env['PATH']=';'.join(recipe['runtime_bin_paths'])+';'+env.get('PATH','')
 runs=[]
 for name in ['cpu','gpu']:
  dest=out/name;exe=verify(recipe['executables'][name]);begin=time.perf_counter()
  with(out/(name+'.log')).open('x',encoding='utf8')as f:r=subprocess.run([str(exe),str(cp),str(dest)],cwd='C:/',env=env,stdout=f,stderr=subprocess.STDOUT)
  elapsed=time.perf_counter()-begin
  if r.returncode:raise RuntimeError('Fixed-state operator failed '+name)
  report=read(dest/'manifest.json');runs.append(dict(backend=name,wall_seconds=elapsed,report=ref(dest/'manifest.json'),operator_seconds=report['elapsed_seconds']))
  print(json.dumps(runs[-1]),flush=True)
 return compare_fixed_outputs(cp,build,out,runs,before,paths,doc)

def compare_fixed_outputs(cp,build,out,runs,before,paths,doc):
 a,b=[read(Path(x['report']['path']))for x in runs];comparisons={};failures=[]
 for k in ['checkpoint','model','packet','semantic','raw_rows','active_rows','profile_count','entity_count','full_W_rank','statistical_partition_exactly_once','original_W_half_cost','groups']:
  if a[k]!=b[k]:failures.append('metadata '+k)
 for family in ['arrays']:
  for k,da in a[family].items():
   db=b[family][k];pa=verify(da,out/'cpu');pb=verify(db,out/'gpu')
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
  if len(offset)!=2*182017+1 or offset[0]!=0 or offset[-1]!=len(values) or np.any(offset[1:]<offset[:-1]):raise ValueError('CSR full raw source identity invalid')
  if len(block)!=len(values)or len(component)!=len(values)or np.any(component>=32)or np.any(block>=len(read(verify(doc['model']))['blocks'])):raise ValueError('CSR column identity invalid')
  columns=block.astype(np.int64)*32+component
  matrix=sparse.csr_matrix((values,columns,offset),shape=(2*182017,len(read(verify(doc['model']))['blocks'])*32))
  if not np.isfinite(matrix.data).all():raise ValueError('Nonfinite original UV derivative')
  if np.any(np.diff(offset)[2*25537:2*25537+2]!=0):raise ValueError('Excluded source row reentered')
  return matrix
 A=csr(a,out/'cpu');B=csr(b,out/'gpu');delta=(B-A).tocsr();delta.eliminate_zeros();rowids=np.repeat(np.arange(delta.shape[0],dtype=np.int32),np.diff(delta.indptr));av=np.asarray(A[rowids,delta.indices]).reshape(-1)
 norm=np.abs(delta.data)/(1+np.abs(av));count=int(np.count_nonzero(norm>2e-8));maximum=float(norm.max(initial=0))
 comparisons['physical_UV_CSR/full_source_block_component_union']=dict(cpu_nnz=A.nnz,gpu_nnz=B.nnz,different_entries=delta.nnz,max_normalized_error=maximum,over_original_gate=count,no_small_derivative_discard=True)
 if count:failures.append('physical_UV_CSR/full_source_block_component_union')
 after={str(p):sha(p)for p in paths}
 if before!=after:failures.append('state changed')
 result=dict(schema='rinkfit.full_fixed_state_GPU_transport_check.v1',passed=not failures,no_Solve=True,no_station_refinement=True,no_checkpoint_written=True,complete_state_unchanged=before==after,checkpoint=ref(cp),build=ref(build/'recipe.json'),raw_rows=182017,active_rows=182016,profiles=961,original_W_rank=214112,original_derivative_gate=2e-8,runs=runs,end_to_end_speedup=runs[0]['wall_seconds']/runs[1]['wall_seconds'],comparisons=comparisons,failures=failures)
 (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items()if k!='comparisons'}),flush=True)
 if failures:raise RuntimeError('Fixed-state CPU/GPU mismatch; do not enable')
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--build',required=True);p.add_argument('--out',required=True);a=p.parse_args();check(a.checkpoint,a.build,a.out)
