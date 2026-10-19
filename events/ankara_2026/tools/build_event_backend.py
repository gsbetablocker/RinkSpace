from pathlib import Path
import os, subprocess, sys, json, hashlib
sys.stdout.reconfigure(encoding='utf-8')
root=Path(r'D:\OneDrive\codex_projs\RinkSpace\calibration\rinkfit')
out=Path(sys.argv[1]) if len(sys.argv)>1 else Path(r'D:\CodexData\RinkSpace_Data\native\rf11');out.mkdir(parents=True,exist_ok=True)
vc=Path(r'D:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat')
line='cmd.exe /d /s /c '+chr(34)+'call '+chr(34)+str(vc)+chr(34)+' >nul && set'+chr(34)
r=subprocess.run(line,capture_output=True,text=True,errors='replace',check=True)
env={k.upper():v for k,v in os.environ.items()};env.update({k.upper():v for k,v in (x.split('=',1)for x in r.stdout.splitlines()if '=' in x and not x.startswith('='))})
cl=Path(r'D:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe')
deps=Path(r'D:\CodexData\RinkSpace_Data\runtime_tools\rf_deps')
inc=[root/'cpp/include',deps/'ceres-solver/Library/include',deps/'eigen/Library/include/eigen3',deps/'glog/Library/include',deps/'gflags/Library/include',deps/'nlohmann_json/Library/include']
bins=[deps/'ceres-solver/Library/bin',deps/'glog/Library/bin',deps/'gflags/Library/bin',Path(r'D:\CodexData\RinkSpace_program_tracking_tmp\gtsam-env\Library\bin')]
env['PATH']=';'.join(str(p)for p in bins)+';'+env['PATH'];env.update({k:'1'for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']})
sources=[root/'cpp/src'/x for x in ['augmented_lagrangian.cpp','checkpoint.cpp','ceres_backend.cpp','event_model.cpp','event_builder.cpp','event_fit.cpp']]
inputs=sources+sorted((root/'cpp/include').rglob('*.hpp'))+sorted((root/'cpp/include').rglob('*.h'))
input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in inputs}
exe=out/'rinkfit_event_fit.exe'
args=[str(cl),'/nologo','/std:c++17','/EHsc','/O2','/MD','/utf-8','/bigobj','/MP4','/DGLOG_USE_GLOG_EXPORT','/DGLOG_USE_GFLAGS','/DGLOG_NO_ABBREVIATED_SEVERITIES',*[('/I'+str(p))for p in inc],*[str(p)for p in sources],'/Fo:'+str(out)+os.sep,'/Fe:'+str(exe),str(deps/'ceres-solver/Library/lib/ceres.lib'),str(deps/'glog/Library/lib/glog.lib'),str(deps/'gflags/Library/lib/gflags.lib')]
r=subprocess.run(args,env=env,cwd=out,capture_output=True,text=True,errors='replace');print(r.stdout);print(r.stderr);(out/'direct_build.log').write_text(r.stdout+r.stderr,encoding='utf-8')
if r.returncode:raise SystemExit(r.returncode)
if input_hashes!={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in inputs}:raise RuntimeError('Compilation inputs changed during build; do not use this executable')
recipe={'sources':input_hashes,'translation_units':[str(p)for p in sources],'project_headers_included':True,'command':args,'runtime_bin_paths':[str(p)for p in bins],'executable_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'compiler':str(cl)}
(out/'direct_build_recipe.json').write_text(json.dumps(recipe,indent=2)+'\n',encoding='utf-8')
r=subprocess.run([str(exe),'--capabilities-json'],env=env,cwd=out,capture_output=True,text=True,errors='replace');print(r.returncode);print(r.stdout);print(r.stderr)
(out/'capabilities.json').write_text(r.stdout,encoding='utf-8')
raise SystemExit(r.returncode)
