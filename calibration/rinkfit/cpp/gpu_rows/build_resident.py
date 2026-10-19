"""Build a verified resident GPU whole-UV worker; never runs a solve.

The original resident source-authority/full-aux protocol and qbar/values/profile
adapters are retained. Only whole_uv uses the current public CUDA partials.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,shutil
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('D:/CodexData/RinkSpace_Data/runs/okada_s1_r11/gpu_accel/resident'));a=p.parse_args()
P=Path(__file__).resolve().parents[4];S=Path(__file__).resolve().parent;R=Path('D:/CodexData/RinkSpace_Data/runs/okada_s1_r11');M=R/'model_checks';V=M/'resident_transport_proto_v3';O=a.output.resolve();O.mkdir(parents=True,exist_ok=True)
if 'OneDrive'in str(O):raise ValueError('Build artifacts must remain in Data')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ref=lambda p:dict(path=str(p),sha256=sha(p))
old=json.loads((V/'recipe.json').read_text());whole=P/'calibration/rinkfit/tools/reduced/whole_W_transport.cpp';base=P/'calibration/rinkfit/tools/reduced/transport_qbar_base.cpp'
worker=S/'resident_worker.cpp';header=S/'resident_dataset.hpp';profile=V/'profile_implicit_adapter.cpp';objects=[V/'qbar_adapter.obj',V/'values_adapter.obj',R/'gpu_accel/rows/event_model.obj'];rowlib=R/'gpu_accel/rows/row_uv.lib'
originals=[whole,base,worker,header,profile,V/'recipe.json',V/'resident_worker.cpp',V/'resident_dataset.hpp',V/'build_policy.hpp',V/'transport_math_only.inc',rowlib,R/'gpu_accel/rows/build_recipe.json',*objects,*sorted((V/'frozen_include/rinkfit').glob('*.hpp')),*sorted(S.glob('row_uv_cache.*')),S/'abi.hpp']
before={str(f):sha(f)for f in originals};assert sha(worker)==sha(V/'resident_worker.cpp')and sha(header)==sha(V/'resident_dataset.hpp')
# Snapshot-aware reader headers are the verified resident implementation.
# All unchanged physical headers must match the freshly compiled GPU objects.
for f in(P/'calibration/rinkfit/cpp/include/rinkfit').glob('*.hpp'):
 if f.name not in ['event_io.hpp','event_packet.hpp']and(V/'frozen_include/rinkfit'/f.name).exists():assert sha(f)==sha(V/'frozen_include/rinkfit'/f.name),f
assert sha(P/'calibration/rinkfit/cpp/include/rinkfit/event_packet.hpp')==old['original_reader_policy_sha256']
shutil.copytree(V/'frozen_include',O/'frozen_include',dirs_exist_ok=True);shutil.copyfile(V/'build_policy.hpp',O/'build_policy.hpp')
math_prefix=base.read_text().split('int main(int argc,char**argv)',1)[0];assert 'struct CSR'in math_prefix
(O/'transport_math_only.inc').write_text(math_prefix)
text=whole.read_text();body=text[text.index('int main(int argc,char**argv)'):]
old_loader='EventPacket packet(pp);packet.load_semantics(sp);packet.load_query(std::filesystem::u8path(packet.manifest.at("query_packet").at("manifest").get<std::string>()),packet.root.parent_path()/"query_initialization.json");'
assert body.count(old_loader)==1
body=body.replace(old_loader,'const EventPacket& packet=resident::dataset(pp,sp);').replace('int main(int argc,char**argv)','int run(int argc,char**argv)',1)
pre='''#include "resident_dataset.hpp"
#include "rinkfit/event_model.hpp"
#include "rinkfit/event_packet.hpp"
#include <ceres/dynamic_autodiff_cost_function.h>
#include <ceres/jet.h>
#include <Eigen/LU>
#include <chrono>
#include <iostream>
#include <iomanip>
#include "row_uv_cache.hpp"
'''
adapter=O/'whole_uv_adapter.cpp';adapter.write_text(pre+'namespace whole_transport_GPU_v1 {\n#include "transport_math_only.inc"\n'+body+'\n}\nint resident_whole_uv_main(int argc,char**argv){return whole_transport_GPU_v1::run(argc,argv);}\n')
setup={};save=sys.argv;sys.argv=[sys.argv[0],str(O)];driver=M/'build_event_backend.py';exec(compile(driver.read_text().split('sources=[')[0],str(driver),'exec'),setup);sys.argv=save
inc=[S,O,O/'frozen_include']+[x for x in setup['inc']if 'rinkfit'not in str(x).lower()];exe=O/'resident_worker.exe';cuda=Path('D:/Program Files/CUDA');libs=[setup['deps']/(x+'/Library/lib/'+y+'.lib')for x,y in [('ceres-solver','ceres'),('glog','glog'),('gflags','gflags')]]
sources=[worker,adapter,profile];cmd=[str(setup['cl']),'/nologo','/std:c++17','/EHsc','/O2','/MD','/utf-8','/bigobj','/MP2','/DRINKFIT_GPU_ROWS','/DGLOG_USE_GLOG_EXPORT','/DGLOG_USE_GFLAGS','/DGLOG_NO_ABBREVIATED_SEVERITIES',*['/I'+str(x)for x in inc],*[str(x)for x in sources],*[str(x)for x in objects],str(rowlib),str(cuda/'lib/x64/cudart.lib'),'/Fo:'+str(O)+os.sep,'/Fe:'+str(exe),*[str(x)for x in libs]]
proc=subprocess.run(cmd,cwd='C:/',env=setup['env'],capture_output=True,text=True,errors='replace');(O/'build.log').write_text(proc.stdout+proc.stderr);print(proc.stdout,proc.stderr,flush=True)
if proc.returncode:raise SystemExit(proc.returncode)
assert before=={str(f):sha(f)for f in originals}
recipe=dict(schema='rinkfit.resident_verified_GPU_transport.v1',no_Solve=True,no_kernel_changed=False,CPU_primal_RS_profile_W=True,raw_UV_GPU=True,supported_ops=['qbar','values','whole_uv','profile_implicit'],original_resident_v3=ref(V/'recipe.json'),original_reader_policy_sha256=old['original_reader_policy_sha256'],original_whole_UV=ref(whole),original_transport_math=ref(base),whole_source_authority_full_four_auxiliary_protocol_unchanged=True,original_qbar_values_profile_retained=True,adapter_changes=['whole_uv only: replace complete cold EventPacket loader by verified resident immutable EventPacket reference','current public RINKFIT_GPU_ROWS raw UV implementation; CPU primal/RS/profile/W retained','row_uv_cache.hpp included outside namespace','model and all four auxiliaries verified before and after every request'],source_inputs_before=before,source_inputs_after={str(f):sha(f)for f in originals},sources={str(f):sha(f)for f in [*sources,*objects,rowlib,O/'transport_math_only.inc',header,O/'build_policy.hpp',Path(__file__),*sorted((O/'frozen_include/rinkfit').glob('*.hpp'))]},command=cmd,compiler_cwd='C:/',executable=ref(exe),runtime_bin_paths=[str(cuda/'bin'),*old['runtime_bin_paths']],rows_recipe=ref(R/'gpu_accel/rows/build_recipe.json'))
(O/'recipe.json').write_text(json.dumps(recipe,indent=2));print(json.dumps(dict(recipe=ref(O/'recipe.json'),exe=ref(exe))),flush=True)
