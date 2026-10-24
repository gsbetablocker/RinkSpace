"""Build new-source readers/operators without changing historical build snapshots.

Only the existing CUDA raw-buffer UV library is reused. EventModel, qbar/values
and resident adapters compile against the selected frozen reader.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def frozen_headers(directory):
 return sorted(f for f in (Path(directory)/'frozen_include').rglob('*') if f.is_file())

def adapt_current_values_source(text):
 old='memberships[i]!=(packet.is_active(i)?1:0)'
 new='memberships[i]!=(packet.current_input?int(packet.statistical_active_mask[i]):(packet.is_active(i)?1:0))'
 if old not in text and new not in text:raise ValueError('Current values statistical partition signature changed')
 return text.replace(old,new).replace('{"raw_material_labels",171}','{"raw_material_labels",packet.material_labels.size()}').replace('{"frames",1248}','{"frames",packet.manifest.at("frame_count")}')

def build(run,header,output):
 R,H,O=Path(run).resolve(),Path(header).resolve(),Path(output).resolve();M=R/'model_checks';P=Path(__file__).resolve().parents[4]
 rows_origin=read(R/'gpu_accel/rows/build_recipe.json')
 for name in ['row_uv.cu','abi.hpp','exact_math.cuh']:
  path=P/'calibration/rinkfit/cpp/gpu_rows'/name
  if rows_origin['inputs'].get(str(path))!=sha(path):raise ValueError('Pure CUDA input changed: '+name)
 if O.exists():raise FileExistsError(str(O))
 O.mkdir(parents=True);setup={};save=sys.argv;sys.argv=[sys.argv[0],str(O)];driver=M/'build_event_backend.py';exec(compile(driver.read_text().split('sources=[')[0],str(driver),'exec'),setup);sys.argv=save
 old=M/'qbar_finite_refine_v7_build';oldr=dict(runtime_bin_paths=[str(p)for p in setup['bins']],source_component_evidence_policies={});V=R/'gpu_accel/resident';source_header=H
 for name in ['qbar','values','whole_cpu','profile_cpu','resident']:
  D=O/name;D.mkdir();shutil.copytree(V/'frozen_include',D/'frozen_include');shutil.copyfile(H,D/'frozen_include/rinkfit/event_packet.hpp')
  extras=[*H.parent.glob('*.hpp'),P/'calibration/rinkfit/cpp/include/rinkfit/source_audit_layer.inc',P/'calibration/rinkfit/cpp/include/rinkfit/source_covariance.hpp']
  for extra in extras:
   if extra.name!='event_packet.hpp' and extra.is_file():shutil.copyfile(extra,D/'frozen_include/rinkfit'/extra.name)
  shutil.copyfile(P/'calibration/rinkfit/cpp/src/event_model.cpp',D/'event_model.cpp')
 libs=[setup['deps']/(x+'/Library/lib/'+y+'.lib')for x,y in [('ceres-solver','ceres'),('glog','glog'),('gflags','gflags')]]
 flags=['/nologo','/std:c++17','/EHsc','/O2','/MD','/utf-8','/bigobj','/DGLOG_USE_GLOG_EXPORT','/DGLOG_USE_GFLAGS','/DGLOG_NO_ABBREVIATED_SEVERITIES']
 def compile_one(name,sources,exe,extra=(),gpu=False):
  D=O/name;inc=[D,D/'frozen_include',P/'calibration/rinkfit/cpp/gpu_rows']+[x for x in setup['inc']if 'rinkfit'not in str(x).lower()]
  cmd=[str(setup['cl']),*flags,*(['/DRINKFIT_GPU_ROWS']if gpu else[]),*['/I'+str(x)for x in inc],*map(str,sources),str(D/'event_model.cpp'),*map(str,extra),'/Fo:'+str(D)+os.sep,'/Fe:'+str(D/exe),*map(str,libs)]
  result=subprocess.run(cmd,cwd=D,env=setup['env'],capture_output=True,text=True,errors='replace');(D/'build.log').write_text(result.stdout+result.stderr,encoding='utf8')
  if result.returncode:raise RuntimeError(name+' compile failed: '+str(D/'build.log'))
  return cmd
 recipes={};caps=dict(source_component_recovery=True,source_audit_revision=True)
 active_profile_count='std::count_if(packet.active_rows.begin(),packet.active_rows.end(),[&](auto i){return packet.kind[i]==2;})'
 for name,origin,frozen_name,exe in [('qbar',M/'qbar_finite_refine_v7.cpp','qbar_finite_refine_v7.cpp','qbar_sparse.exe'),('values',M/'values_rf29_partial_build/reduced_values_frozen.cpp','reduced_values_frozen.cpp','reduced_state_values.exe')]:
  D=O/name;src=D/frozen_name;text=origin.read_text(encoding='utf8')
  src.write_text(adapt_current_values_source(text) if name=='values' else text,encoding='utf8');cmd=compile_one(name,[src],exe)
  frozen={str(f):dict(file=str(f.relative_to(D)),sha256=sha(f))for f in frozen_headers(D)}
  r=dict(schema='rinkfit.reduced_reader_snapshot_build.v1',tool=name,no_Solve=True,source_sha256=sha(src),math_source=ref(origin),command=cmd,
    frozen_sources=frozen,event_model_source=ref(D/'event_model.cpp'),reader_header_sha256=sha(H),source_component_recovery=True,source_audit_revision=True,source_component_evidence_policies=oldr['source_component_evidence_policies'],capabilities=caps,
    event_model_object=ref(D/'event_model.obj'),runtime_bin_paths=oldr['runtime_bin_paths'],executable_sha256=sha(D/exe),arithmetic_math_unchanged=True,
    supports_partial_linearization_values=name=='values',current_reader_build=True)
  (D/'recipe.json').write_text(json.dumps(r,indent=2),encoding='utf8');recipes[name]=dict(executable=ref(D/exe),recipe=ref(D/'recipe.json'))
 print('SOURCE_QBAR_VALUES_BUILT',json.dumps(recipes),flush=True)
 whole=P/'calibration/rinkfit/tools/reduced/whole_W_transport.cpp';transport=P/'calibration/rinkfit/tools/reduced/transport_qbar_base.cpp';profile=M/'profile_implicit_transport_build/profile_implicit_transport.cpp'
 for name,origin,filename,exe in [('whole_cpu',whole,'whole.cpp','whole_W_transport.exe'),('profile_cpu',profile,'profile.cpp','profile_implicit_transport.exe')]:
  D=O/name;shutil.copyfile(transport,D/'transport_qbar_base.cpp');text=origin.read_text()
  if name=='whole_cpu':
   text=text.replace('n!=182017','n!=packet.manifest.at("observations").get<std::size_t>()').replace('packet.active_rows.size()!=182016','packet.active_rows.size()!=doc.at("active_rows").get<std::size_t>()').replace('wres.size()!=214112','wres.size()!=packet.formal_residual_rows').replace('membership[i]!=(packet.is_active(i)?1:0)','membership[i]!=(packet.current_input?int(packet.statistical_active_mask[i]):(packet.is_active(i)?1:0))').replace('profiles!=961','profiles!='+active_profile_count).replace('profiles!=packet.queries.size()','profiles!='+active_profile_count)
  text=text.replace('rows.size()!=961','rows.size()!='+active_profile_count)
  src=D/filename;src.write_text(text,encoding='utf8');cmd=compile_one(name,[src],exe)
  r=dict(schema='rinkfit.source_target_operator_build.v1',command=cmd,executable_sha256=sha(D/exe),source_sha256=sha(src),runtime_bin_paths=oldr['runtime_bin_paths'],reader_header_sha256=sha(H),source_inputs={str(f):sha(f)for f in [origin,transport,src,D/'event_model.cpp',*frozen_headers(D)]},selected_rank_from_validated_packet=True)
  (D/'recipe.json').write_text(json.dumps(r,indent=2),encoding='utf8');recipes[name]=dict(executable=ref(D/exe),recipe=ref(D/'recipe.json'),source_files=[ref(src),ref(D/'transport_qbar_base.cpp')])
 D=O/'resident';S=P/'calibration/rinkfit/cpp/gpu_rows';worker=S/'resident_worker.cpp';shutil.copyfile(S/'resident_dataset.hpp',D/'resident_dataset.hpp');shutil.copyfile(worker,D/'resident_worker.cpp')
 (D/'build_policy.hpp').write_text('#pragma once\ninline constexpr const char* ORIGINAL_READER_SHA="'+sha(H)+'";\n')
 prefix=transport.read_text().split('int main(int argc,char**argv)',1)[0];(D/'transport_math_only.inc').write_text(prefix,encoding='utf8')
 body=(O/'whole_cpu/whole.cpp').read_text();body=body[body.index('int main(int argc,char**argv)'):]
 old_loader='EventPacket packet(pp);packet.load_semantics(sp);packet.load_query(std::filesystem::u8path(packet.manifest.at("query_packet").at("manifest").get<std::string>()),packet.root.parent_path()/"query_initialization.json");'
 if body.count(old_loader)!=1:raise ValueError('Original complete whole-UV loader signature changed')
 body=body.replace(old_loader,'const EventPacket& packet=resident::dataset(pp,sp);').replace('int main(int argc,char**argv)','int run(int argc,char**argv)',1)
 pre='#include "resident_dataset.hpp"\n#include "rinkfit/event_model.hpp"\n#include "rinkfit/event_packet.hpp"\n#include <ceres/dynamic_autodiff_cost_function.h>\n#include <ceres/jet.h>\n#include <Eigen/LU>\n#include <chrono>\n#include <iostream>\n#include <iomanip>\n#include "row_uv_cache.hpp"\n'
 (D/'whole_adapter.cpp').write_text(pre+'namespace source_whole_GPU {\n#include "transport_math_only.inc"\n'+body+'\n}\nint resident_whole_uv_main(int argc,char**argv){return source_whole_GPU::run(argc,argv); }\n',encoding='utf8')
 (D/'profile_adapter.cpp').write_text((M/'resident_transport_proto_v3/profile_implicit_adapter.cpp').read_text().replace('rows.size()!=961','rows.size()!='+active_profile_count),encoding='utf8')
 (D/'unsupported.cpp').write_text('#include <stdexcept>\nint resident_qbar_main(int,char**){throw std::runtime_error("source-target worker exposes only whole_uv and profile_implicit");}\nint resident_values_main(int,char**){throw std::runtime_error("source-target worker exposes only whole_uv and profile_implicit");}\n')
 rowlib=R/'gpu_accel/rows/row_uv.obj';cuda=Path('D:/Program Files/CUDA');shutil.copyfile(S/'row_uv_cache.cpp',D/'row_uv_cache.cpp');sources=[D/f for f in ['resident_worker.cpp','whole_adapter.cpp','profile_adapter.cpp','unsupported.cpp','row_uv_cache.cpp']]
 cmd=compile_one('resident',sources,'resident_worker.exe',[rowlib,cuda/'lib/x64/cudart.lib'],gpu=True)
 inputs={str(f):sha(f)for f in [*sources,D/'resident_dataset.hpp',D/'build_policy.hpp',D/'transport_math_only.inc',D/'event_model.obj',D/'event_model.cpp',rowlib,S/'row_uv_cache.hpp',S/'abi.hpp',S/'row_uv.cu',S/'exact_math.cuh',*frozen_headers(D)]}
 r=dict(schema='rinkfit.source_target_GPU_transport_build.v1',executable=ref(D/'resident_worker.exe'),reader_header_sha256=sha(H),source_inputs=inputs,sources=inputs,command=cmd,runtime_bin_paths=[str(cuda/'bin'),*oldr['runtime_bin_paths']],CPU_primal_RS_profile_W=True,raw_UV_GPU=True,no_old_qbar_values_objects_reused=True,no_old_EventPacket_wrappers_reused=True,only_pure_CUDA_raw_buffer_object_reused=True,supported_ops=['whole_uv','profile_implicit'],pure_GPU_rows_recipe=ref(R/'gpu_accel/rows/build_recipe.json'))
 (D/'build_recipe.json').write_text(json.dumps(r,indent=2),encoding='utf8');recipes['resident_build']=ref(D/'build_recipe.json')
 (O/'operators.json').write_text(json.dumps(recipes,indent=2),encoding='utf8');print('SOURCE_TARGET_OPERATORS_BUILT',json.dumps(ref(O/'operators.json')),flush=True);return O/'operators.json'

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--header',required=True);p.add_argument('--output',required=True);a=p.parse_args();build(a.run,a.header,a.output)
