"""Build same-state CPU/CUDA whole-W transport operators, without optimization."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, subprocess

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def build(out,rows):
 here=Path(__file__).resolve().parent;core=here.parents[1];out=Path(out).resolve();rows=Path(rows).resolve()
 out.mkdir(parents=True,exist_ok=True)
 spec=importlib.util.spec_from_file_location('row_build_environment',core/'cpp/gpu_rows/build.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 os.environ.setdefault('CUDA_PATH',r'D:\Program Files\CUDA');env=module.environment();deps=module.D;cuda=Path(env['CUDA_PATH'])
 cl=Path(r'D:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe')
 includes=[core/'cpp/include',core/'cpp/gpu_rows',deps/'ceres-solver/Library/include',deps/'eigen/Library/include/eigen3',deps/'glog/Library/include',deps/'gflags/Library/include',deps/'nlohmann_json/Library/include']
 sources=[here/'whole_W_transport.cpp',here/'transport_qbar_base.cpp']+list((core/'cpp/include/rinkfit').glob('*.hpp'))+list((core/'cpp/gpu_rows').glob('*.hpp'))
 native=[rows/'event_model.obj',rows/'row_uv.lib'];identities={str(p):sha(p)for p in sources+native}
 common=[str(cl),'/nologo','/std:c++17','/EHsc','/O2','/MD','/utf-8','/bigobj','/DGLOG_USE_GLOG_EXPORT','/DGLOG_USE_GFLAGS','/DGLOG_NO_ABBREVIATED_SEVERITIES',*['/I'+str(p)for p in includes]]
 libs=[str(deps/(name+'/Library/lib/'+lib+'.lib'))for name,lib in [('ceres-solver','ceres'),('glog','glog'),('gflags','gflags')]]
 commands=[];executables={}
 for backend in ['cpu','gpu']:
  dest=out/backend;dest.mkdir(exist_ok=True);exe=dest/'whole_W_transport.exe'
  command=common+(['/DRINKFIT_GPU_ROWS']if backend=='gpu'else[])+[str(here/'whole_W_transport.cpp'),str(rows/'event_model.obj'),'/Fo:'+str(dest)+os.sep,'/Fe:'+str(exe)]+libs
  if backend=='gpu':command += [str(rows/'row_uv.lib'),str(cuda/'lib/x64/cudart.lib')]
  result=subprocess.run(command,cwd='C:/',env=env,capture_output=True,text=True,errors='replace');(dest/'build.log').write_text(result.stdout+result.stderr,encoding='utf8')
  print(backend,result.returncode,result.stdout,result.stderr,flush=True)
  if result.returncode:raise RuntimeError('Transport build failed: '+backend)
  commands.append(command);executables[backend]=ref(exe)
 if identities!={str(p):sha(p)for p in sources+native}:raise RuntimeError('Build inputs changed')
 bins=[cuda/'bin',deps/'ceres-solver/Library/bin',deps/'glog/Library/bin',deps/'gflags/Library/bin',Path(r'D:\CodexData\RinkSpace_program_tracking_tmp\gtsam-env\Library\bin')]
 recipe=dict(schema='rinkfit.whole_W_CPU_CUDA_same_math_build.v1',no_Solve=True,model_unchanged=True,CPU_primal_RS_profile_W=True,source_inputs=identities,commands=commands,executables=executables,runtime_bin_paths=[str(p)for p in bins],rows_recipe=ref(rows/('build_recipe_current.json'if(rows/'build_recipe_current.json').exists()else'build_recipe.json')))
 (out/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n',encoding='utf8');return recipe
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--rows',required=True);a=p.parse_args();build(a.out,a.rows)
