"""Immutable reader-aware exact-Jet evaluation, with safe boundary stopping."""
from pathlib import Path
import argparse,json,hashlib,sys,os,subprocess,time,shutil
sys.stdout.reconfigure(encoding='utf-8')
p=argparse.ArgumentParser();p.add_argument('--build',required=True);p.add_argument('--stop-file');p.add_argument('input_manifest');p.add_argument('output_directory');a=p.parse_args()
B=Path(a.build).resolve();I=Path(a.input_manifest).resolve();O=Path(a.output_directory).resolve();S=Path(a.stop_file).resolve() if a.stop_file else None
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'));ref=lambda p:dict(path=str(p),sha256=sha(p))
recipe=read(B/'recipe.json');assert recipe['schema']=='rinkfit.reduced_reader_snapshot_build.v1' and recipe['tool']=='qbar' and recipe['source_component_recovery'] is True
exe=B/'qbar_sparse.exe';assert sha(exe)==recipe['executable_sha256'];assert all(sha(B/d['file'])==d['sha256'] for d in recipe['frozen_sources'].values())
doc=read(I);assert doc['schema']=='rinkfit.qbar_sparse_linearization_input.v1'
def verify(d,root):
 key='path' if 'path' in d else 'file';path=root/d[key];assert sha(path)==d['sha256'],str(path);return path
for key in ['packet','model','semantic']:verify(doc[key],I.parent)
sem_path=verify(doc['semantic'],I.parent);sem=read(sem_path);component=None
from rinkfit_start.semantic_resolution import load_semantics
_,_,_,extra=load_semantics(sem_path,doc['raw_rows'],expected_sha256=doc['semantic']['sha256'])
proof=extra.get('source_component_recovery')
if proof:
 digest=proof['evidence_sha256'];assert digest in recipe['source_component_evidence_policies'],'reader has another pinned source-component policy'
 assert len(proof['bindings'])==recipe['source_component_evidence_policies'][digest]['changed']
 component=dict(proof_sha256=proof['overlay_sha256'],evidence_sha256=digest,changed=len(proof['bindings']))
for d in doc['arrays'].values():verify(d,I.parent)
if isinstance(doc.get('source_rows'),dict):verify(doc['source_rows'],I.parent)
if O.exists() and any(O.iterdir()):raise ValueError('Never overwrite a prior linearization/driver identity')
O.mkdir(parents=True,exist_ok=True);shutil.copyfile(__file__,O/'driver_frozen.py');started=time.time();command=[str(exe),str(I),str(O)];identity=dict(schema='rinkfit.qbar_reader_snapshot_execution.v1',input_path=str(I),input_sha256=sha(I),command=command,executable_sha256=sha(exe),recipe_sha256=sha(B/'recipe.json'),driver_sha256=sha(__file__),started_unix=started,no_Solve=True,reader_header_sha256=recipe['reader_header_sha256'],source_component_recovery=component,stop_file=str(S) if S else None)
if S and S.exists():
 identity.update(returncode=3,status='STOPPED_BEFORE_READONLY_EVALUATION',elapsed_seconds=0.);(O/'driver_identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8');print(json.dumps(identity));raise SystemExit(3)
env={k.upper():v for k,v in os.environ.items()};env['PATH']=';'.join(recipe['runtime_bin_paths'])+';'+env.get('PATH','');env.update(PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
with(O/'driver.log').open('x',encoding='utf-8',buffering=1) as log:
 child=subprocess.Popen(command,cwd=B,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW);identity['actual_pid']=child.pid;print('QBAR_READER_SNAPSHOT_STARTED',child.pid,flush=True)
 for line in child.stdout:log.write(line);print(line.rstrip(),flush=True)
 code=child.wait()
assert sha(I)==identity['input_sha256'],'input changed during exact-Jet evaluation'
if code==0:
 result=read(O/'manifest.json');assert result['input']['sha256']==sha(I) and result['model']==doc['model'] and result['semantic']==doc['semantic'] and result['packet']==doc['packet']
identity.update(returncode=code,status='COMPLETE' if code==0 else 'FAILED_READER_OR_NUMERIC_EVALUATION',stop_requested_during_evaluation=bool(S and S.exists()),elapsed_seconds=time.time()-started)
(O/'driver_identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8');print(json.dumps(identity));raise SystemExit(code)
