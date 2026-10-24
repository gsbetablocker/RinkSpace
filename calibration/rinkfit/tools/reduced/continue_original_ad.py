"""Orchestrate the restored original d78 primary/W/full2 programs.
No optimizer or mathematical objective is implemented in this entry point.
"""
from pathlib import Path
import argparse,contextlib,hashlib,json,os,shutil,subprocess,sys,time,traceback
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf8')
def engineering_max(error,budget,kind,normal,active):
 coverage=np.linalg.norm(error,axis=1);profile=np.asarray(kind)==2
 coverage[profile]=abs(np.einsum('ij,ij->i',normal[profile],error[profile]))
 return float(np.max((coverage[active]/budget[active])**2))

def protect_current_model(before,after,full):
 if len(before['blocks'])!=len(after['blocks'])or before['camera']!=after['camera']or before['materials']!=after['materials']:raise ValueError('A solver proposal changed current static geometry or model topology')
 protected=0
 if not full:
  for a,b in zip(before['blocks'],after['blocks']):
   if not a['name'].startswith(('ad/','print/')):
    if a!=b:raise ValueError('Original camera/ice/board changed: '+a['name'])
    protected+=1
  return protected
 dep=int(before['camera']['control_block_begin'])+int(before['camera']['dependent_control'])
 for i,(a,b)in enumerate(zip(before['blocks'],after['blocks'])):
  if np.asarray(a['values']).shape!=np.asarray(b['values']).shape:raise ValueError('Original physical block dimension changed: '+a['name'])
  if {k:v for k,v in a.items()if k!='values'}!={k:v for k,v in b.items()if k!='values'}:raise ValueError('Original parameter domain/scale/owner changed: '+a['name'])
  dependent_witness=a['name']in {'ad/'+str(ad)for ad in before['current_input_scope']['print_only_ad_ids']}
  if a.get('constant',False)and not dependent_witness:
   if a['values']!=b['values']:raise ValueError('Fixed current physical block changed: '+a['name'])
   protected+=1
  for j in a.get('fixed_indices',[]):
   if i==dep and j in (0,1):continue # original exact dependent-angle gauge storage
   if a['values'][j]!=b['values'][j]:raise ValueError('Original fixed component changed: '+a['name'])
 return protected

def run(inp,prepared,output,compiled_dir,resume_stages=False,existing_primary=None,budget_report=None,minimax_only=False,gpu_camera_dll=None,gpu_camera_validation=None):
 O=Path(output).resolve();O.mkdir(parents=True,exist_ok=resume_stages)
 with (O/'main.log').open('a' if resume_stages else 'x',encoding='utf8',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
  try:
   print(json.dumps(dict(kind='worker',pid=os.getpid(),route='original_d78_primary_wholeW_full2',input=ref(inp))),flush=True)
   from primary_ad_chart import run as primary
   from conditional_W_ad_chart import run as secondary
   initial_transport=read(read(prepared)['transport']['path'])
   if minimax_only:
    if budget_report is not None or read(initial_transport['semantic']['path'])['current_input_scope']['formal_W_rank']!=0:raise ValueError('Pure original primary is only for the declared no-statistical-factor source scope')
    trbase=Path(read(prepared)['transport']['path']).parent;PK0=read(initial_transport['packet']['path']);SP0=read(initial_transport['semantic']['path'])
    def a0(d,base):return np.fromfile(Path(base)/d['file'],d['dtype']).reshape(d['shape'])
    e0=a0(initial_transport['arrays']['error_uv'],trbase).reshape(-1,2);E0=a0(PK0['arrays']['engineering_budget_px'],Path(initial_transport['packet']['path']).parent);active0=a0(SP0['active_source_mask'],Path(initial_transport['semantic']['path']).parent).astype(bool)
    initial_cost=engineering_max(e0,E0,a0(SP0['arrays']['kind'],Path(initial_transport['semantic']['path']).parent),a0(SP0['arrays']['profile_normal'],Path(initial_transport['semantic']['path']).parent),active0)
   else:initial_cost=initial_transport['original_W_half_cost']
   (O/'step_cost.log').write_text('第0步  cost='+str(initial_cost)+'\n',encoding='utf8')
   primary_dir=Path(existing_primary).resolve() if existing_primary else O/'primary'
   start=time.monotonic()
   if existing_primary:
    if read(primary_dir/'assembly.json')['input']!=ref(inp):raise ValueError('Existing primary belongs to another current input')
   elif not (resume_stages and (primary_dir/'report.json').exists()):primary(inp,primary_dir,True)
   elif read(primary_dir/'assembly.json')['input']!=ref(inp):raise ValueError('Cannot resume changed primary input')
   if not minimax_only and not (resume_stages and (O/'secondary/report.json').exists()):secondary(primary_dir,O/'secondary',budget_from_full_report=budget_report)
   P=read(prepared);CP=read(P['checkpoint']['path']);cm=read((primary_dir if minimax_only else O/'secondary')/'candidate_manifest.json');seed=O/'seed';seed.mkdir(exist_ok=resume_stages)
   names=dict(station='station.f64',branch='path.u32',row='initial_row.f64',alpha='initial_alpha.f64')
   seed_cp=dict(CP);seed_cp['row_state_files']={}
   for key,name in names.items():
    d=cm['row_state_files'][key];src=Path(d.get('path',d.get('file')));shutil.copyfile(src,seed/name);seed_cp['row_state_files'][key]=dict(file=str(seed/name),dtype=d['dtype'],shape=d['shape'],sha256=sha(seed/name))
   dump(seed/'checkpoint.json',seed_cp)
   tr=read(P['transport']['path']);packet=Path(tr['packet']['path']);semantic=Path(tr['semantic']['path'])
   full_out=O/'full2'
   retry=0
   while full_out.exists() and any(full_out.iterdir()):
    retry+=1;full_out=O/('full2_retry'+str(retry).zfill(2))
   command=[sys.executable,str(Path(__file__).with_name('recompute_complete.py')),'--model',cm['model']['path'],'--initial-dir',str(seed),'--semantic-manifest',str(semantic),'--packet',str(packet),'--output',str(full_out),'--compiled-dir',str(compiled_dir),'--run',str(O)]
   if gpu_camera_dll:command+=['--gpu-camera-dll',str(gpu_camera_dll),'--gpu-camera-validation',str(gpu_camera_validation)]
   subprocess.run(command,stdout=log,stderr=log,check=True)
   full=read(full_out/'report.json');best=full['best'];baseline=read(P['transport']['path']);PK=read(packet);S=read(semantic)
   def ar(d,base):return np.fromfile(Path(base)/d['file'],d['dtype']).reshape(d['shape'])
   error=ar(baseline['arrays']['error_uv'],Path(P['transport']['path']).parent).reshape(-1,2);E=ar(PK['arrays']['engineering_budget_px'],packet.parent);active=ar(S['active_source_mask'],semantic.parent).astype(bool);baseline_b=engineering_max(error,E,ar(S['arrays']['kind'],semantic.parent),ar(S['arrays']['profile_normal'],semantic.parent),active)
   before=read(CP['model']['path']);after=read(full_out/'model.json');full=bool(S['current_input_scope'].get('full_physical_solver',False));protected=protect_current_model(before,after,full)
   allowed_budget=float(read(budget_report)['best']['max_q']) if budget_report else baseline_b
   improved=(best['max_q']<baseline_b) if minimax_only else (best['max_q']<=allowed_budget and best['whole_original_W_half_cost']<baseline['original_W_half_cost']) if budget_report else (best['max_q']<baseline_b or (best['max_q']<=baseline_b and best['whole_original_W_half_cost']<baseline['original_W_half_cost']))
   actual_cost=best['max_q'] if minimax_only else best['whole_original_W_half_cost']
   result=dict(actual_objective='original_engineering_max_q' if minimax_only else 'original_whole_W_half_cost',current_actual_cost=actual_cost,schema='rinkfit.actual_original_stage_acceptance.v1',accepted=bool(improved),baseline_b=baseline_b,fixed_allowed_budget=allowed_budget,baseline_W_half_cost=baseline['original_W_half_cost'],current_b=best['max_q'],current_W_half_cost=best['whole_original_W_half_cost'],protected_blocks=protected,full_physical_solver=full,model=ref(full_out/'model.json'),complete_state=ref(full_out/'report.json'),elapsed_seconds=time.monotonic()-start,not_final_visual_or_KKT_success=True)
   dump(O/'acceptance.json',result)
   if improved:
    with (O/'step_cost.log').open('a',encoding='utf8') as progress:progress.write('第1步  cost='+str(actual_cost)+'\n')
    print(json.dumps(dict(kind='accepted',accepted_steps_this_process=1,event=dict(cost=actual_cost,actual_minimax_b=best['max_q']),result=result)),flush=True)
   else:print(json.dumps(dict(kind='candidate_rejected',reason='actual original minimax/whole-W objective did not improve',result=result)),flush=True)
  except BaseException:
   traceback.print_exc();raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--prepared',required=True);p.add_argument('--output',required=True);p.add_argument('--compiled-dir',required=True);p.add_argument('--resume-stages',action='store_true');p.add_argument('--existing-primary');p.add_argument('--budget-report');p.add_argument('--minimax-only',action='store_true');p.add_argument('--gpu-camera-dll');p.add_argument('--gpu-camera-validation');a=p.parse_args();run(a.input,a.prepared,a.output,a.compiled_dir,a.resume_stages,a.existing_primary,a.budget_report,a.minimax_only,a.gpu_camera_dll,a.gpu_camera_validation)
