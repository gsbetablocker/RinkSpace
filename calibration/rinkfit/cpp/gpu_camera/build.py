"""Build the FP64 camera DLL into Data using the existing CUDA/MSVC install."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
S=Path(__file__).resolve().parent;O=a.output.resolve();O.mkdir(parents=True,exist_ok=True)
if 'OneDrive' in str(O):raise ValueError('Build artifacts must be outside OneDrive')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs={str(p):sha(p)for p in [S/'camera.cu',S/'exact_math.cuh',S/'abi.hpp',Path(__file__)]}
vc=Path('D:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat')
e=subprocess.run('cmd.exe /d /s /c '+chr(34)+'call '+chr(34)+str(vc)+chr(34)+' >nul && set'+chr(34),capture_output=True,text=True,check=True)
env={k.upper():v for k,v in os.environ.items()};env.update({k.upper():v for k,v in(x.split('=',1)for x in e.stdout.splitlines()if '='in x and not x.startswith('='))})
cuda=Path(env.get('CUDA_PATH','D:/Program Files/CUDA'));env['PATH']=str(cuda/'bin')+';'+env['PATH'];(O/'tmp').mkdir(exist_ok=True);env['TMP']=env['TEMP']=str(O/'tmp')
cmd=[str(cuda/'bin/nvcc.exe'),'--use-local-env','-std=c++17','-O3','-arch=sm_120','--fmad=false','--prec-div=true','--prec-sqrt=true','--expt-relaxed-constexpr','-Xcompiler','/MD,/utf-8,/fp:strict','-shared',str(S/'camera.cu'),'-o',str(O/'gpu_camera.dll')]
(O/'build_env.json').write_text(json.dumps({k:v for k,v in env.items()if k in ['PATH','INCLUDE','LIB','VCTOOLSINSTALLDIR','COMSPEC','CUDA_PATH']},indent=2));(O/'build_command.json').write_text(json.dumps(cmd));r=subprocess.run(cmd+['-v'],cwd='C:/',env=env,capture_output=True,text=True,errors='replace');(O/'build.log').write_text(r.stdout+r.stderr,encoding='utf8');print(r.stdout,r.stderr,flush=True)
if r.returncode:raise SystemExit(r.returncode)
assert inputs=={k:sha(Path(k))for k in inputs}
recipe=dict(schema='rinkfit.gpu_camera_build.v1',inputs=inputs,command=cmd,FP64=True,no_fast_math=True,architecture='sm_120',dll=dict(path=str(O/'gpu_camera.dll'),sha256=sha(O/'gpu_camera.dll')))
(O/'build_recipe.json').write_text(json.dumps(recipe,indent=2));print(json.dumps(recipe),flush=True)
