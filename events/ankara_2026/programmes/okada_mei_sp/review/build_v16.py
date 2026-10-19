import sys,json,pathlib,shutil,hashlib,subprocess
w=pathlib.Path('D:/CodexData/RinkSpace_Data/runs/okada_rec_1019');root=pathlib.Path('D:/OneDrive/codex_projs/RinkSpace');old=w/'build_v15';out=w/'build_v16'
out.mkdir();(out/'src').mkdir();shutil.copytree(root/'calibration/rinkfit/cpp/include',out/'include')
recipe=json.loads((old/'recipe.json').read_text('utf8'))
setup={};driver=pathlib.Path('D:/CodexData/RinkSpace_Data/runs/okada_s1_r11/model_checks/build_event_backend.py');save=sys.argv;sys.argv=[str(driver),str(out)];exec(compile(driver.read_text(encoding='utf8').split('sources=')[0],str(driver),'exec'),setup);sys.argv=save
cmd=recipe['command'];flags=[s for s in cmd[1:]if s.startswith('/') and s!='/c' and not s.startswith('/Fo:') and not s.startswith('/Fe:')]
flags=['/I'+str(out/'include'),*flags]
source=out/'src/joint_recovery.cpp';shutil.copyfile(root/'calibration/rinkfit/cpp/src/joint_recovery.cpp',source)
command=[cmd[0],*flags,'/c',str(source),'/Fo:'+str(out/'joint_recovery.obj')]
r=subprocess.run(command,cwd=out,env=setup['env'],capture_output=True,text=True,errors='replace')
(out/'joint_recovery.log').write_text(r.stdout+r.stderr,'utf8')
if r.returncode:print(r.stdout+r.stderr);raise SystemExit(r.returncode)
link=[str(out/'joint_recovery.obj')if pathlib.Path(s).name=='joint_recovery.obj'else '/Fe:'+str(out/'joint_fit.exe')if s.startswith('/Fe:')else s for s in recipe['link_command']]
r=subprocess.run(link,cwd=out,env=setup['env'],capture_output=True,text=True,errors='replace');(out/'link.log').write_text(r.stdout+r.stderr,'utf8')
if r.returncode:print(r.stdout+r.stderr);raise SystemExit(r.returncode)
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
recipe.update(parent_recipe=dict(file=str(old/'recipe.json'),sha256=sha(old/'recipe.json')),command=command,link_command=link,
 source_files={**recipe['source_files'],**{str(p.relative_to(out)):sha(p)for p in (out/'src').glob('*.cpp')},**{str(p.relative_to(out)):sha(p)for p in (out/'include').rglob('*.hpp')}},
 executable_sha256=sha(out/'joint_fit.exe'),returncode=0)
(out/'recipe.json').write_text(json.dumps(recipe,indent=2),'utf8');print(json.dumps(dict(executable=str(out/'joint_fit.exe'),sha256=recipe['executable_sha256'])),flush=True)
