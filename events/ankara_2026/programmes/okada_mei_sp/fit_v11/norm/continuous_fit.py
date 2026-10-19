"""Continuous finite-visible original-S / original-W Phase-I initialization.
Each accepted state is complete. Native physical replay and final KKT are never
inferred from affine, LP or conic success. Immutable operator versions per run.
"""
from pathlib import Path
import argparse,json,hashlib,os,sys,subprocess,time,shutil,traceback

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def verify(r):
 p=Path(r['path']).resolve()
 if sha(p)!=r['sha256']:raise ValueError('Changed frozen operator or state '+str(p))
 return p

def run(input_path,output):
 IP=Path(input_path).resolve();cfg=read(IP)
 if cfg['schema']!='rinkfit.continuous_finite_visible_original_W_SLP_input.v1':raise ValueError('Typed continuous scope required')
 refs=['checkpoint','source_context','prepare_entry','curvature_entry','adaptive_entry','primary_entry','primary_recipe','W_bundle','W_maker','normalizer','adopter','compiled_binding','membership_builder']
 for k in refs:verify(cfg[k])
 B=read(verify(cfg['W_bundle']));sources={Path(x['path']).name:x for x in B['frozen_sources']}
 for x in B['frozen_sources']:verify(x)
 for x in cfg.get('frozen_operator_sources',[]):verify(x)
 D=Path(output).resolve();D.mkdir(exist_ok=False);shutil.copyfile(__file__,D/'driver_frozen.py');(D/'input_frozen.json').write_text(json.dumps(cfg,indent=2));STOP=D/'STOP';current=cfg['checkpoint'];begin=time.monotonic();history=[];status='UNFINISHED';diagnostic={};cycle=0;membership=cfg.get('initial_source_membership')
 if membership:verify(membership)
 env={k.upper():v for k,v in os.environ.items()};env.update(PYTHONIOENCODING='utf8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 def atomic(p,value):
  p=Path(p);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(value,indent=2));os.replace(q,p)
 def emit(kind,**fields):
  v=dict(kind=kind,time=time.time(),**fields);print(json.dumps(v),flush=True)
  with(D/'events.jsonl').open('a',encoding='utf8')as f:f.write(json.dumps(v)+'\n')
 class BoundaryStop(Exception):pass
 def stage(name,command,allow_after_stop=False,stop_after=True):
  if STOP.exists()and not allow_after_stop:raise BoundaryStop()
  T=time.monotonic();emit('STAGE_START',stage=name,command=command)
  with(D/(name+'.log')).open('x',encoding='utf8')as f:
   child=subprocess.Popen(command,cwd=D,env=env,stdout=f,stderr=subprocess.STDOUT);stop_seen=False
   while True:
    try:code=child.wait(timeout=.25);break
    except subprocess.TimeoutExpired:
     if STOP.exists()and not stop_seen:emit('STOP_SEEN_WAITING_CURRENT_COMPLETE_OPERATOR_BOUNDARY',stage=name,pid=child.pid);stop_seen=True
  receipt=dict(command=command,returncode=code,elapsed_seconds=time.monotonic()-T,log=ref(D/(name+'.log')));atomic(D/(name+'_execution.json'),receipt);emit('STAGE_END',stage=name,**receipt)
  if code:raise ValueError('Stage '+name+' failed; preserve exact log and complete parent')
  if STOP.exists()and stop_after and not allow_after_stop:raise BoundaryStop()
 try:
  while True:
   if STOP.exists():raise BoundaryStop()
   for x in cfg.get('frozen_operator_sources',[]):verify(x)
   tag='cycle_%05d'%cycle;P=D/tag;P.mkdir();prepared=cfg.get('first_prepared')if cycle==0 else None
   if not prepared:
    stage(tag+'_prepare',[sys.executable,'-X','utf8',str(verify(cfg['prepare_entry'])),'--checkpoint',str(verify(current)),'--context',str(verify(cfg['source_context'])),'--output',str(P/'prepare'),'--stop-file',str(STOP)]);prepared=read(P/'prepare/prepared.json')
   for k in ['checkpoint','linearization','transport','profile_domains']:verify(prepared[k])
   current=prepared['checkpoint'];cp=read(verify(current));TR=float(cp['control']['trust']);atomic(P/'prepared.json',prepared);atomic(D/'current.json',current)
   curvature=cfg.get('first_curvature')if cycle==0 else None
   if not curvature:
    package=Path(prepared['base_directory']).parents[1]/'python_frozen'
    stage(tag+'_C2',[sys.executable,'-X','utf8',str(verify(cfg['curvature_entry'])),'--transport',str(verify(prepared['transport'])),'--output',str(P/'C2'),'--python-package',str(package),'--compiled-binding',str(verify(cfg['compiled_binding']))]);stage(tag+'_adaptive_C2',[sys.executable,'-X','utf8',str(verify(cfg['adaptive_entry'])),'--curvature',str(P/'C2/manifest.json'),'--output',str(P/'adaptive_C2')]);curvature=ref(P/'adaptive_C2/manifest.json')
   verify(curvature);cd=read(verify(curvature))
   for k in ['checkpoint','transport']:
    if cd[k]['sha256']!=prepared[k]['sha256']:raise ValueError('Stale current numerical S trust '+k)
   primary=cfg.get('first_primary_report')if cycle==0 else None
   if not primary:
    primary_input=dict(schema='rinkfit.allsource_SOC_primary_input.v1',prepared=ref(P/'prepared.json'),station_TR_manifest=curvature,trust_scaled=TR,conic_threads=16)
    atomic(P/'primary_input.json',primary_input)
    cmd=[sys.executable,'-X','utf8',str(verify(cfg['primary_entry'])),'--input',str(P/'primary_input.json'),'--output',str(P/'primary')];cmd+=cfg.get('primary_arguments',[]);stage(tag+'_primary',cmd);primary=ref(P/'primary/report.json')
   primarydoc=read(verify(primary))
   if not primarydoc['ready_for_conditional_W']:raise ValueError('No strict all-source descending primary working target; no fallback physical step')
   if primarydoc['transport']['sha256']!=prepared['transport']['sha256']or primarydoc['station_TR_manifest']['sha256']!=curvature['sha256']:raise ValueError('Mixed primary and complete current state')
   make_command=[sys.executable,'-X','utf8',str(verify(cfg['W_maker'])),'--primary-report',str(verify(primary)),'--bundle',str(verify(cfg['W_bundle'])),'--trust',str(TR),'--output',str(P/'W_input.json')]
   if membership:make_command+=['--source-membership',str(verify(membership))]
   stage(tag+'_make_W',make_command);stage(tag+'_W',[sys.executable,'-X','utf8',str(verify(sources['run_finite_visible_W_secondary.py'])),'--input',str(P/'W_input.json'),'--output',str(P/'W'),'--stop-file',str(STOP)])
   Wraw=read(P/'W/report.json')
   if not Wraw['proposed']:raise ValueError('Full original-W coupled conditional proposal failed original gates')
   before=Path(prepared['base_directory'])/'report.json';stage(tag+'_normalize',[sys.executable,'-X','utf8',str(verify(cfg['normalizer'])),'--secondary-report',str(P/'W/report.json'),'--before-value-report',str(before),'--output',str(P/'proposal')]);proposal=ref(P/'proposal/report.json');q=read(verify(proposal))
   stage(tag+'_line',[sys.executable,'-X','utf8',str(verify(sources['finite_visible_line_search.py'])),'--proposal',str(verify(proposal)),'--initial-dir',prepared['base_directory'],'--semantic-manifest',str(verify(q['semantic'])),'--output',str(P/'line'),'--stop-file',str(STOP)],allow_after_stop=False,stop_after=False);line=read(P/'line/report.json')
   if not line.get('accepted_candidate')and STOP.exists():raise BoundaryStop()
   if not line.get('accepted_candidate'):raise ValueError('No actual legal nonlinear Phase-I descent; diagnose charts/curvature, do not loop micro-backtracking')
   ai=dict(schema='rinkfit.external_curvature_finite_S_original_W_adoption_input.v2',parent_checkpoint=current,primary_report=primary,W_report=proposal,frozen_line_bundle=cfg['W_bundle'],actual_line_report=ref(P/'line/report.json'),before_value_report=ref(before),group_W_chart=ref(P/'W/group_W_chart/manifest.json'),select_main=True,selection_reason='Strict actual full-source Phase-I max-q descent; original fullW/quantiles are recorded separately, not final accuracy/KKT.',iteration_semantics='One complete accepted original physical+raw-N auxiliary state increments parent iteration by one')
   atomic(P/'adoption_input.json',ai);stage(tag+'_adopt',[sys.executable,'-X','utf8',str(verify(cfg['adopter'])),'--input',str(P/'adoption_input.json'),'--output',str(P/'accepted')],allow_after_stop=True);result=read(P/'accepted/result.json');current=result['current_complete_checkpoint'];nd=read(verify(current));atomic(D/'current.json',current)
   member_command=[sys.executable,'-X','utf8',str(verify(cfg['membership_builder'])),'--report',str(P/'W/report.json'),'--output',str(P/'next_source_membership')]
   if membership:member_command+=['--previous',str(verify(membership))]
   stage(tag+'_membership',member_command,allow_after_stop=True);membership=ref(P/'next_source_membership/manifest.json')
   rec=dict(checkpoint=current,prepared=ref(P/'prepared.json'),primary_report=primary,proposal=proposal,actual_line=ref(P/'line/report.json'),metrics=nd['metrics'],trust=nd['control']['trust'],next_source_membership=membership);history.append(rec);atomic(D/'history.json',history);emit('COMPLETE_ACCEPTED_STATE',**rec)
   if STOP.exists():raise BoundaryStop()
   if nd['b']<=1:status='PHASE_I_BUDGET_REACHED_ORIGINAL_PHASE_II_AND_FINAL_FRESH_KKT_REQUIRED';break
   last=nd['control']['history'][-1]
   if last['ratio']<cfg.get('poor_prediction_ratio',.1):raise ValueError('Actual/model ratio exposes poor prediction; complete state saved before structural diagnosis')
   cycle+=1
 except BoundaryStop:status='STOPPED_AT_COMPLETE_STATE'
 except BaseException as e:status='DIAGNOSTIC_UNFINISHED';diagnostic=dict(error=str(e),traceback=traceback.format_exc());emit('DIAGNOSTIC_BOUNDARY',**diagnostic)
 finally:atomic(D/'result.json',dict(schema='rinkfit.continuous_finite_visible_original_W_SLP_result.v1',status=status,current_complete_checkpoint=current,history=history,diagnostic=diagnostic,driver_wall_seconds=time.monotonic()-begin,source_membership=membership,original_raw_rows=182017,active_rows=182016,source_W_E_unchanged=True,final_fit_accepted=False,PhaseII_joint_station_optimum=False,full_true_KKT_certified=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.input,a.output)
