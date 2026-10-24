"""Continue accepted complete states through unchanged original d78 programs."""
from pathlib import Path
import contextlib,importlib.util,json,os,sys,time,traceback,argparse
from current_ad_prepare import run as prepare
from continue_original_ad import run as solve
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def run(base,output,initial_complete=None,budget_report=None,operators=None,operator_spec=None,initial_step=1):
 B=Path(base).resolve();O=Path(output).resolve();O.mkdir(exist_ok=False)
 latest=Path(initial_complete).resolve() if initial_complete else B/'solve_baseline01/full2';cost=float(read(latest/'report.json')['best']['whole_original_W_half_cost']);step=int(initial_step)
 fixed_report=Path(budget_report).resolve() if budget_report else B/'prepare02/baseline_actual.json';operators=Path(operators).resolve() if operators else B/'operators/current02';operator_spec=Path(operator_spec).resolve() if operator_spec else B/'operator_spec.json'
 progress=O/'step_cost.log';progress.write_text('第'+str(step)+'步  cost='+str(cost)+'\n',encoding='utf8')
 compiled=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_s1_r11/capture_cpu_i9/compiled_camera')
 recipe=Path(__file__).parent/'original/input.py';sp=importlib.util.spec_from_file_location('_original_input_builder',recipe);builder=importlib.util.module_from_spec(sp);sp.loader.exec_module(builder)
 with (O/'main.log').open('x',encoding='utf8',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
  print(json.dumps(dict(kind='worker',pid=os.getpid(),route='original_d78_baseline_W_continuation',initial_step=step,initial_cost=cost,fixed_budget=read(fixed_report)['best']['max_q'])),flush=True)
  while True:
   if (O/'STOP').exists():print(json.dumps(dict(kind='stopped_at_original_stage_boundary',accepted_steps=step)),flush=True);break
   iteration=O/('i'+str(step+1).zfill(5));iteration.mkdir()
   try:
    prepare(B/'input',operators,iteration/'prepared',compiled/'bound_camera_sensor.py',model_override=latest/'model.json',initial_dir=latest)
    builder.run(iteration/'prepared/prepared.json',iteration/'prepared/station_trust/manifest.json',.0128,iteration/'input',operator_spec)
    if (O/'STOP').exists():print(json.dumps(dict(kind='stopped_at_original_stage_boundary',accepted_steps=step)),flush=True);break
    solve(iteration/'input/input.json',iteration/'prepared/prepared.json',iteration/'solve',compiled,budget_report=str(fixed_report))
    result=read(iteration/'solve/acceptance.json')
    if not result['accepted']:
     print(json.dumps(dict(kind='actual_original_no_decrease',accepted_steps=step,result=result,not_final_convergence=True)),flush=True);break
    newcost=float(result['current_W_half_cost'])
    roundoff=64*sys.float_info.epsilon*(1+abs(cost)+abs(newcost))
    if cost-newcost<=roundoff:
     print(json.dumps(dict(kind='actual_original_numerical_stagnation',accepted_steps=step,before=cost,after=newcost,roundoff=roundoff,result=result,not_final_convergence=True)),flush=True);break
    latest=Path(result['model']['path']).parent;step+=1;cost=newcost
    (O/'current.json').write_text(json.dumps(dict(step=step,cost=cost,model=result['model'],complete_state=result['complete_state'],fixed_allowed_budget=result['fixed_allowed_budget']),indent=2),encoding='utf8')
    with progress.open('a',encoding='utf8') as f:f.write('第'+str(step)+'步  cost='+str(cost)+'\n')
    print(json.dumps(dict(kind='accepted',accepted_steps_this_process=step,event=dict(cost=cost,actual_minimax_b=result['current_b']),result=result)),flush=True)
   except BaseException:
    traceback.print_exc();raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True);p.add_argument('--initial-complete');p.add_argument('--budget-report');p.add_argument('--operators');p.add_argument('--operator-spec');p.add_argument('--initial-step',type=int,default=1);a=p.parse_args();run(a.base,a.output,a.initial_complete,a.budget_report,a.operators,a.operator_spec,a.initial_step)
