"""Run current full S01 through the restored original d78 programs.

This entry point only joins the existing full2, prepare, primary and conditional
W programs. It defines no new objective, optimizer, physical model or source
qualification. Every accepted candidate retains its complete native state.
"""
from pathlib import Path
import argparse,contextlib,importlib.util,json,os,subprocess,sys,time,traceback

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'calibration/solve2'))
sys.path.insert(0,str(ROOT/'calibration/rinkfit/src'))
from export_original_full import export
from export_original_ad import sha,save_json
from current_ad_prepare import run as prepare
from continue_original_ad import run as solve
from current_input_identity import same_numerical_input


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def operator_spec(target):
    math=Path(__file__).parent/'original/math';norm=math/'norm'
    paths=dict(strict_norm_source=norm/'fixed_entity_norm_constraints.py',full_norm_source=norm/'source_affine_norm_constraints.py',
        profile_domain_source=math/'profile_domain_constraints.py',coupled_A_source=math/'coupled_A.py',group_W_source=math/'group_W_station.py',
        secondary_source=Path(__file__).with_name('conditional_W_ad_chart.py'),station_TR_limiter_source=math/'station_curvature_TR_chart.py',
        print_det_guard_source=math/'print_det_convex_guard.py')
    # A current source declaration, not a history/qualification dependency.
    path=target/'operator_spec.json'
    if path.exists():
        previous=read(path)
        if all(previous[k]['sha256']==sha(v)for k,v in paths.items()):return path
        serial=1
        while(target/f'operator_spec_r{serial:02}.json').exists():serial+=1
        path=target/f'operator_spec_r{serial:02}.json';sources=target/f'operator_sources_r{serial:02}.json'
    else:sources=target/'operator_sources.json'
    save_json(sources,dict(schema='rinkfit.current_restored_original_math_sources.v1',
        source_inputs={str(path.resolve()):sha(path)for path in paths.values()},current_input_only=True))
    spec=dict(schema='rinkfit.joint_full_lawful_S_frozen_operator_spec.v1',**{k:ref(v)for k,v in paths.items()},group_W_recipe=ref(sources),operator_bundle=ref(sources))
    save_json(path,spec);return path


def validate_args(a):
    if a.resume_current:
        required=('input_dir','baseline_dir')
        mode='Current complete-state continuation'
    else:
        required=('prepared','native_model','original_input','feature_prepared')
        mode='Current input export'
    missing=['--'+name.replace('_','-')for name in required if not getattr(a,name,None)]
    if missing:raise ValueError(mode+' requires '+', '.join(missing))


def run(a):
    validate_args(a)
    out=Path(a.output).resolve();out.mkdir(exist_ok=a.resume_current)
    progress=out/'step_cost.log'
    if not progress.exists():progress.write_text('',encoding='utf8')
    main=out/a.main_log
    with main.open('x',encoding='utf8',buffering=1)as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            inp=Path(a.input_dir).resolve()if a.input_dir else out/'input'
            source_model=Path(a.baseline_dir).resolve()/'model.json'if a.resume_current else Path(a.native_model).resolve()
            print(json.dumps(dict(kind='worker',pid=os.getpid(),route='original_d78_full_S01_primary_conditionalW_full2',
                source_model=str(source_model),no_single_ad_physical_mask=True)),flush=True)
            receipt=read(inp/'mapping.json')if a.resume_current else export(a.prepared,a.native_model,a.original_input,a.feature_prepared,inp,data_root=a.data_root,policy_path=a.scope_policy)
            print(json.dumps(dict(kind='current_full_input',raw_rows=receipt['raw_total_rows'],active_rows=receipt['active_rows'],
                W_rank=receipt['W_rank'],active_Query=receipt['active_Query_rows'],registration=read(inp/'registration.json'))),flush=True)
            compiled=Path(a.compiled_dir).resolve();operators=Path(a.operators).resolve();spec=operator_spec(out)
            baseline_path=Path(a.baseline_dir).resolve()if a.baseline_dir else out/'baseline'
            if a.baseline_dir and not(baseline_path/'report.json').is_file():raise ValueError('The explicitly selected complete current-scope baseline is absent')
            retry=0
            while baseline_path.exists()and not(baseline_path/'report.json').is_file():
                retry+=1;baseline_path=out/('baseline_retry'+str(retry).zfill(2))
            command=[sys.executable,str(Path(__file__).with_name('recompute_complete.py')),'--model',str(inp/'model.json'),
                '--initial-dir',str(inp),'--semantic-manifest',str(inp/'semantic/manifest.json'),'--packet',str(inp/'packet/manifest.json'),
                '--output',str(baseline_path),'--compiled-dir',str(compiled),'--run',str(out)]
            if a.gpu_camera_dll:command+=['--gpu-camera-dll',a.gpu_camera_dll,'--gpu-camera-validation',a.gpu_camera_validation]
            print(json.dumps(dict(kind='original_full2_baseline_start',command=command)),flush=True)
            if not(a.resume_current and (baseline_path/'report.json').is_file()):subprocess.run(command,stdout=log,stderr=log,check=True)
            latest=baseline_path;baseline=read(latest/'report.json')
            if(latest/'visual_review.json').is_file()and read(latest/'visual_review.json').get('visual_accepted')is False:raise ValueError('The selected complete physical state was visually rejected; use the corrected current input with the accepted reference state')
            if baseline['packet']!=ref(inp/'packet/manifest.json'):
                if not same_numerical_input(baseline['packet']['path'],inp/'packet/manifest.json'):raise ValueError('Saved baseline belongs to a different numerical current input')
                print(json.dumps(dict(kind='current_metadata_compatible_baseline',previous_packet=baseline['packet'],current_packet=ref(inp/'packet/manifest.json'),same_numerical_sources_W_model_and_selection=True)),flush=True)
            cost=float(baseline['best']['whole_original_W_half_cost'])
            if not progress.read_text(encoding='utf8').strip():progress.write_text('第0步  cost='+str(cost)+'\n',encoding='utf8')
            print(json.dumps(dict(kind='baseline',step=0,cost=cost,engineering_b=baseline['best']['max_q'],
                complete_state=ref(latest/'report.json'),physical_change=False)),flush=True)
            loader=importlib.util.spec_from_file_location('_current_original_full_input',Path(__file__).parent/'original/input.py')
            builder=importlib.util.module_from_spec(loader);loader.loader.exec_module(builder)
            step=0
            if a.resume_current and(out/'current.json').is_file():
                current=read(out/'current.json')
                if current.get('visual_accepted')is False:raise ValueError('Saved numerical candidate was visually rejected and cannot be the next physical initializer')
                if sha(current['model']['path'])!=current['model']['sha256']or sha(current['complete_state']['path'])!=current['complete_state']['sha256']:raise ValueError('Current accepted complete state changed')
                latest=Path(current['complete_state']['path']).parent;cost=float(current['cost']);step=int(current['step'])
                print(json.dumps(dict(kind='resumed_current_accepted_complete_state',step=step,cost=cost,complete_state=current['complete_state'],original_fixed_baseline=ref(baseline_path/'report.json'))),flush=True)
            first_iteration=True
            while True:
                if(out/'STOP').exists():
                    print(json.dumps(dict(kind='stopped_at_original_stage_boundary',accepted_steps=step)),flush=True);break
                resume_stage=bool(first_iteration and a.resume_stage)
                if resume_stage:
                    iteration=Path(a.resume_stage).resolve();iteration.relative_to(out)
                    if not iteration.is_dir():raise ValueError('The current partial original stage is absent')
                else:
                    iteration=out/('i'+str(step+1).zfill(5));retry=0
                    while iteration.exists():retry+=1;iteration=out/('i'+str(step+1).zfill(5)+'_retry'+str(retry).zfill(2))
                    iteration.mkdir()
                first_iteration=False
                print(json.dumps(dict(kind='original_current_stage_start',next_step=step+1,parent_model=ref(latest/'model.json'))),flush=True)
                P=iteration/'prepared/prepared.json';trust_manifest=iteration/'prepared/station_trust/manifest.json'
                if resume_stage and P.is_file()and trust_manifest.is_file():
                    pr=read(P);tr=read(pr['transport']['path']);cp=read(pr['checkpoint']['path'])
                    if cp['model']!=ref(latest/'model.json')or tr['packet']!=ref(inp/'packet/manifest.json')or tr['semantic']!=ref(inp/'semantic/manifest.json')or read(trust_manifest)['checkpoint']!=pr['checkpoint']:raise ValueError('Saved current preparation belongs to another complete state')
                    print(json.dumps(dict(kind='resumed_current_completed_preparation',prepared=ref(P))),flush=True)
                else:prepare(inp,operators,iteration/'prepared',compiled/'bound_camera_sensor.py',resume_current=resume_stage,model_override=latest/'model.json',initial_dir=latest,qbar_operator_dir=a.qbar_operator_dir)
                linearization=None;existing_primary=None
                if resume_stage:
                    for candidate in sorted(iteration.glob('solve*/primary'),key=lambda x:x.stat().st_mtime_ns,reverse=True):
                        if not(candidate/'report.json').is_file():continue
                        assembly=read(candidate/'assembly.json');saved_input=Path(assembly['input']['path']);saved=read(saved_input)
                        if assembly['input']!=ref(saved_input)or saved['prepared']!=ref(P):continue
                        if assembly['authority']['model']!=ref(latest/'model.json')or assembly['authority']['packet']!=ref(inp/'packet/manifest.json')or assembly['authority']['semantic']!=ref(inp/'semantic/manifest.json'):continue
                        if assembly['source']['sha256']!=sha(Path(__file__).with_name('primary_ad_chart.py')):continue
                        linearization=saved_input.parent;existing_primary=candidate
                        print(json.dumps(dict(kind='resumed_current_completed_primary',primary=ref(candidate/'report.json'),assembly=ref(candidate/'assembly.json'),no_reassembly_or_primary_resolve=True)),flush=True);break
                for candidate in sorted(iteration.glob('linearization*'),key=lambda x:x.stat().st_mtime_ns,reverse=True):
                    if linearization is not None:break
                    if(candidate/'input.json').is_file():
                        saved=read(candidate/'input.json')
                        if saved['prepared']==ref(P)and saved['operator_spec']==ref(spec):linearization=candidate;break
                if linearization is None:
                    linearization=iteration/'linearization';serial=0
                    while linearization.exists():serial+=1;linearization=iteration/('linearization_retry'+str(serial).zfill(2))
                    builder.run(P,trust_manifest,.0128,linearization,spec)
                if(out/'STOP').exists():
                    print(json.dumps(dict(kind='stopped_at_original_stage_boundary',accepted_steps=step)),flush=True);break
                solve_output=iteration/'solve';serial=0
                while solve_output.exists():serial+=1;solve_output=iteration/('solve_retry'+str(serial).zfill(2))
                solve(linearization/'input.json',P,solve_output,compiled,existing_primary=existing_primary,
                    budget_report=str(baseline_path/'report.json'),gpu_camera_dll=a.gpu_camera_dll,gpu_camera_validation=a.gpu_camera_validation)
                result=read(solve_output/'acceptance.json')
                if not result['accepted']:
                    print(json.dumps(dict(kind='actual_original_no_decrease',accepted_steps=step,result=result,not_final_convergence=True)),flush=True);break
                newcost=float(result['current_W_half_cost']);roundoff=64*sys.float_info.epsilon*(1+abs(cost)+abs(newcost))
                if cost-newcost<=roundoff:
                    print(json.dumps(dict(kind='actual_original_numerical_stagnation',accepted_steps=step,before=cost,after=newcost,
                        roundoff=roundoff,result=result,not_final_convergence=True)),flush=True);break
                latest=Path(result['model']['path']).parent;cost=newcost;step+=1
                save_json(out/'current.json',dict(step=step,cost=cost,model=result['model'],complete_state=result['complete_state'],
                    fixed_allowed_budget=result['fixed_allowed_budget'],full_S01_candidate_only=True,original_best_not_overwritten=True))
                with progress.open('a',encoding='utf8')as f:f.write('第'+str(step)+'步  cost='+str(cost)+'\n')
                print(json.dumps(dict(kind='accepted',accepted_steps_this_process=step,event=dict(cost=cost,actual_minimax_b=result['current_b']),result=result)),flush=True)
        except BaseException:
            traceback.print_exc();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('operators','compiled-dir','output'):p.add_argument('--'+name,required=True)
    for name in ('prepared','native-model','original-input','feature-prepared'):p.add_argument('--'+name,help='Required only when exporting a new current input; unused with --resume-current')
    p.add_argument('--scope-policy');p.add_argument('--data-root',default='D:/CodexData/RinkSpace_Data')
    p.add_argument('--gpu-camera-dll');p.add_argument('--gpu-camera-validation')
    p.add_argument('--resume-current',action='store_true');p.add_argument('--main-log',default='main.log')
    p.add_argument('--input-dir')
    p.add_argument('--resume-stage')
    p.add_argument('--qbar-operator-dir',help='Existing original row-identical qbar execution variant; does not replace values/resident operators')
    p.add_argument('--baseline-dir',help='Saved complete current-scope baseline, including explicitly migrated source deltas')
    a=p.parse_args()
    try:validate_args(a)
    except ValueError as error:p.error(str(error))
    run(a)
