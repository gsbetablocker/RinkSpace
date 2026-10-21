from pathlib import Path
import sys,json,subprocess,hashlib,time
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');R=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2');O=R/'c_local_v3';N=R/'native_source_v11';S=Path(__file__).parent
setup={};driver=P/'events/ankara_2026/tools/build_event_backend.py';save=sys.argv
try:
 sys.argv=[str(driver),str(O)];exec(compile(driver.read_text(encoding='utf8').split('sources=')[0],str(driver),'exec'),setup)
finally:sys.argv=save
recipe=json.loads((N/'recipe.json').read_text(encoding='utf8'));flags=[x for x in recipe['commands'][0]['command'][1:]if x.startswith(('/nologo','/std:','/EH','/O2','/MD','/utf-8','/bigobj','/D','/I'))];flags.append('/I'+str(P/'calibration/rinkfit/cpp/include'));libs=[x for x in recipe['commands'][-1]['command']if x.endswith('.lib')];env=dict(setup['env']);env['PATH']=';'.join(recipe['runtime_bin_paths'])+';'+env.get('PATH','')
cmd=[str(setup['cl']),*flags,str(S/'metadata_guard_v3.cpp'),*[str(N/(k+'.obj'))for k in ['event_model','joint_model','joint_packet']],'/Fo:'+str(O/'probe.obj'),'/Fe:'+str(O/'probe.exe'),*libs]
t=time.perf_counter();run=subprocess.run(cmd,cwd=O,env=env,capture_output=True,text=True,errors='replace');(O/'build.log').write_text(run.stdout+run.stderr,encoding='utf8');(O/'build_recipe.json').write_text(json.dumps(dict(command=cmd,returncode=run.returncode,seconds=time.perf_counter()-t,Solve_started=False),indent=2),encoding='utf8')
if run.returncode:raise RuntimeError('metadata probe build failed')
def ref(f):return dict(file=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
binding=dict(registry=ref(R/'c_local_v1/source_branch_registry_v2.json'),source_gate=ref(R/'c_local_v1/source_gate_v2.json'),source_freeze=ref(O/'source_freeze_v3.json'),consumer_module_ref=ref(P/'calibration/rinkfit/cpp/include/rinkfit/joint_source_query_branch_registry_v2.hpp'))
(O/'production_binding_v3.json').write_text(json.dumps(binding,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
cmd=[str(O/'probe.exe'),str(R/'accepted_C_operator_prefix_v2/model.json'),str(R/'R33_prefix_v2/packet.json'),str(O/'production_binding_v3.json'),str(R/'gpu_recovery_v6c/candidate/explicit_nuisance'),str(O/'metadata_guard_QA.json')]
t=time.perf_counter();run=subprocess.run(cmd,cwd=O,env=env,capture_output=True,text=True,errors='replace');(O/'stdout.json').write_text(run.stdout,encoding='utf8');(O/'stderr.log').write_text(run.stderr,encoding='utf8');print(json.dumps(dict(returncode=run.returncode,seconds=time.perf_counter()-t,stderr=run.stderr[-3000:],report=str(O/'metadata_guard_QA.json'))),flush=True)
if run.returncode:raise RuntimeError('metadata guard actual R33 probe failed')
