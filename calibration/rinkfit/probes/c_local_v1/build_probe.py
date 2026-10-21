from pathlib import Path
import os,sys,json,subprocess,hashlib,time
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');R=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2');O=R/'c_local_v1';N=R/'native_source_v10c';S=Path(__file__).parent
setup={};driver=P/'events/ankara_2026/tools/build_event_backend.py';save=sys.argv
try:
 sys.argv=[str(driver),str(O)];exec(compile(driver.read_text(encoding='utf8').split('sources=')[0],str(driver),'exec'),setup)
finally:sys.argv=save
recipe=json.loads((N/'recipe.json').read_text(encoding='utf8'));flags=[x for x in recipe['commands'][0]['command'][1:] if x.startswith(('/nologo','/std:','/EH','/O2','/MD','/utf-8','/bigobj','/D','/I'))];flags.append('/I'+str(S));libs=[x for x in recipe['commands'][-1]['command'] if x.endswith('.lib')];env=dict(setup['env']);env['PATH']=';'.join(recipe['runtime_bin_paths'])+';'+env.get('PATH','')
cmd=[str(setup['cl']),*flags,str(S/'probe.cpp'),*[str(N/(n+'.obj')) for n in ['event_model','joint_model','joint_packet']],'/Fo:'+str(O/'probe.obj'),'/Fe:'+str(O/'probe.exe'),*libs]
t=time.perf_counter();run=subprocess.run(cmd,cwd=O,env=env,capture_output=True,text=True,errors='replace');(O/'build.log').write_text(run.stdout+run.stderr,encoding='utf8');(O/'build_recipe.json').write_text(json.dumps(dict(command=cmd,returncode=run.returncode,seconds=time.perf_counter()-t,source_sha256={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in [S/'probe.cpp',S/'c_source_branch.hpp']},reused_frozen_objects=True,Solve_linked_or_started=False),indent=2),encoding='utf8')
if run.returncode:raise RuntimeError('independent math probe compile failed, see '+str(O/'build.log'))
cmd=[str(O/'probe.exe'),str(R/'gpu_recovery_v6c/candidate/model.json'),str(R/'accepted_C_width_prefix_v1/packet.json'),str(R/'gpu_recovery_v6c/candidate/explicit_nuisance'),str(O/'source_branch_rows.jsonl'),str(O)]
t=time.perf_counter();run=subprocess.run(cmd,cwd=O,env=env,capture_output=True,text=True,errors='replace');(O/'probe_stdout.json').write_text(run.stdout,encoding='utf8');(O/'probe_stderr.log').write_text(run.stderr,encoding='utf8');print(json.dumps(dict(returncode=run.returncode,seconds=time.perf_counter()-t,report=str(O/'local_root_report.json'),stderr=run.stderr[-2000:])),flush=True)
if run.returncode:raise RuntimeError('independent math probe failed')
