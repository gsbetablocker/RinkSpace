"""Current self-contained Ad packet into original d78 qbar/whole-W operators.
Only current data and complete state are read. This module does not optimize.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf8')
def arr(d,base):return np.fromfile(Path(base)/d['file'],d['dtype']).reshape(d['shape'])
def desc(p,dtype,n):return dict(file=str(Path(p).resolve()),dtype=dtype,shape=[n],sha256=sha(p))

def run(source,operators,output,compiled_binding=None,resume_current=False,model_override=None,initial_dir=None,qbar_operator_dir=None):
 I,B,O=map(lambda p:Path(p).resolve(),(source,operators,output))
 O.mkdir(parents=True,exist_ok=resume_current)
 PK=I/'packet/manifest.json';SP=I/'semantic/manifest.json';M=Path(model_override).resolve() if model_override else I/'model.json';AUX=Path(initial_dir).resolve() if initial_dir else I;pk=read(PK);sem=read(SP);N=int(pk['observations'])
 from rinkfit_start.semantic_resolution import load_semantics
 _,sa,_,si=load_semantics(SP,N,expected_sha256=sha(SP))
 if pk['semantic_packet']['sha256']!=sha(SP):raise ValueError('Current packet/semantic mismatch')
 prep=Path(__file__).resolve().parents[2]/'src/rinkfit_start/prepare_qbar_input.py'
 cmd=[sys.executable,str(prep),'--model',str(M),'--initial-dir',str(AUX),'--semantic-manifest',str(SP),'--packet',str(PK),'--output',str(O/'qbar_input')]
 if resume_current and (O/'qbar_input/manifest.json').exists():
  qi=O/'qbar_input/manifest.json';prev=read(qi)
  if prev['model']!=ref(M) or prev['semantic']!=ref(SP):raise ValueError('Resume is only for unchanged current model/semantic')
  prev['packet']=ref(PK);dump(qi,prev)
 else:subprocess.run(cmd,check=True,stdout=sys.stdout,stderr=subprocess.STDOUT)
 QB=Path(qbar_operator_dir).resolve()if qbar_operator_dir else B/'qbar'
 br=read(QB/'recipe.json');env=os.environ.copy();env['PATH']=os.pathsep.join(br['runtime_bin_paths'])+os.pathsep+env['PATH']
 qcmd=[str(QB/'qbar_sparse.exe'),str(O/'qbar_input/manifest.json'),str(O/'qbar')]
 if resume_current and (O/'qbar/manifest.json').exists():
  previous_q=read(O/'qbar/manifest.json')
  if previous_q['input']!=ref(O/'qbar_input/manifest.json'):raise ValueError('Current saved qbar belongs to a different input')
 else:
  print('CURRENT_QBAR_OPERATOR',json.dumps(dict(executable=ref(QB/'qbar_sparse.exe'),recipe=ref(QB/'recipe.json'))),flush=True)
  with (O/'qbar.log').open('x',encoding='utf8',buffering=1)as qlog:
   process=subprocess.Popen(qcmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace')
   for line in process.stdout:qlog.write(line);print(line.rstrip(),flush=True)
   code=process.wait()
  if code:raise RuntimeError('Original current qbar rejected input: '+str(O/'qbar.log'))
 Q=O/'qbar/manifest.json';q=read(Q);selected=arr(q['source_rows'],Q.parent).astype(int)
 station=np.fromfile(AUX/'station.f64','<f8');row=np.fromfile(AUX/'initial_row.f64','<f8')
 station[selected]=arr(q['linearization_station'],Q.parent);row[selected]=arr(q['linearization_row'],Q.parent)
 if not np.isfinite(station).all()or not np.isfinite(row).all():raise ValueError('Original qbar produced nonfinite complete auxiliaries')
 CP=O/'state';CP.mkdir(exist_ok=resume_current);(CP/'model.json').write_bytes(M.read_bytes())
 station.tofile(CP/'station.f64');row.tofile(CP/'initial_row.f64')
 for name in ['path.u32','initial_alpha.f64']:(CP/name).write_bytes((AUX/name).read_bytes())
 qvalues=arr(q['q'],Q.parent);finite=qvalues[np.isfinite(qvalues)]
 if not len(finite):raise ValueError('No finite current q values')
 cp=dict(schema='rinkfit.complete_continuous_reduced_SLP_checkpoint.v1',raw_rows=N,active_rows=int(si['source_exclusion']['active_mask'].sum()),model=ref(M),linearization=ref(Q),b=float(max(1.,np.max(finite))),row_state_files={},current_input_scope=pk['current_input_scope'],no_Solve=True)
 for key,name,dtype in [('station','station.f64','<f8'),('branch','path.u32','<u4'),('row','initial_row.f64','<f8'),('alpha','initial_alpha.f64','<f8')]:cp['row_state_files'][key]=desc(CP/name,dtype,N)
 C=CP/'checkpoint.json';dump(C,cp)
 from rinkfit_start.resident_transport import ResidentTransportClient
 run_directory=O/'resident_run';serial=0
 while run_directory.exists():serial+=1;run_directory=O/('resident_run_retry'+str(serial).zfill(2))
 def completed(role,target):
  manifest=target/'manifest.json'
  if not(resume_current and manifest.is_file()):return False
  report=read(manifest)
  for key,expected in [('checkpoint',ref(C)),('model',ref(M)),('packet',ref(PK)),('semantic',ref(SP))]:
   if report.get(key)!=expected:raise ValueError('A saved current '+role+' result belongs to another '+key)
  profiles=int(np.count_nonzero((sa['kind']==2)&si['source_exclusion']['active_mask']))
  if report.get('raw_rows')!=N or report.get('profiles'if role=='profile_implicit'else'profile_count')!=profiles:raise ValueError('Saved current operation source accounting differs')
  if role=='whole_uv'and(report.get('full_W_rank')!=pk['current_input_scope']['formal_W_rank']or not report.get('statistical_partition_exactly_once')):raise ValueError('Saved complete W operation accounting differs')
  print('RESUMED_CURRENT_COMPLETED_OPERATION',role,str(manifest),flush=True);return True
 with ResidentTransportClient(B/'resident',run_directory) as client:
  if not completed('whole_uv',O/'transport'):client.transport('whole_uv',C,O/'transport')
  if not completed('profile_implicit',O/'profile_domains'):client.transport('profile_implicit',C,O/'profile_domains')
 T=O/'transport/manifest.json';tr=read(T)
 if not np.all(arr(tr['arrays']['status'],T.parent)[si['source_exclusion']['active_mask']]==0):raise ValueError('Current whole UV contains unresolved rows')
 prepared=dict(schema='rinkfit.current_original_prepared.v1',checkpoint=ref(C),linearization=ref(Q),transport=ref(T),profile_domains=ref(O/'profile_domains/manifest.json'),base_directory=str(CP),raw_rows=N,active_rows=cp['active_rows'],no_Solve=True,current_input_scope=pk['current_input_scope'])
 P=O/'prepared.json';dump(P,prepared)
 print('ORIGINAL_CURRENT_GPU_PREPARED',json.dumps(prepared),flush=True)
 from current_station_curvature import run as curvature_run
 curvature_run(T,O/'curvature',Path(__file__).resolve().parents[2]/'src',compiled_binding)
 from rinkfit_start.station_radius import run as adaptive_run
 adaptive_run(O/'curvature/manifest.json',O/'station_trust')
 print('ORIGINAL_CURRENT_READY',json.dumps(dict(prepared=ref(P),curvature=ref(O/'station_trust/manifest.json'))),flush=True)
 return prepared

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--operators',required=True);p.add_argument('--output',required=True);p.add_argument('--compiled-binding');p.add_argument('--resume-current',action='store_true');p.add_argument('--model');p.add_argument('--initial-dir');p.add_argument('--qbar-operator-dir');a=p.parse_args();run(a.input,a.operators,a.output,a.compiled_binding,a.resume_current,a.model,a.initial_dir,a.qbar_operator_dir)
