"""Build a separate current Ceres CUDA policy bridge; never replace a loaded DLL."""
import hashlib,json,subprocess
from pathlib import Path
repo=Path(__file__).resolve().parents[3];data=Path('D:/CodexData/RinkSpace_Data')
base=json.loads((data/'runs/solve2_native/build_06dc5b56a9_acce52/build.json').read_text(encoding='utf-8-sig'))
source=Path(__file__).with_name('gpu_batch_ceres.cpp');digest=hashlib.sha256(source.read_bytes()).hexdigest()
out=data/'runs/solve2_native'/('gpu_'+digest[:12]);out.mkdir(parents=True,exist_ok=True)
binary=out/'rinksolve_gpu_batch.dll';obj=out/'gpu_batch_ceres.obj';manifest=out/'build.json'
flags=[a for a in base['flags'] if not a.startswith('/Fo') and '.cpp' not in a]+['"'+str(source)+'"','/Fo"'+str(obj)+'"']
links=[a for a in base['link_flags'] if not a.startswith('/OUT:') and '.obj' not in a]+['/OUT:"'+str(binary)+'"','"'+str(obj)+'"']
if not binary.exists():
 (out/'compile.rsp').write_text('\n'.join(flags),encoding='utf8');(out/'link.rsp').write_text('\n'.join(links),encoding='utf8')
 command='call "'+base['compiler_setup']+'" >nul && cl.exe @"'+str(out/'compile.rsp')+'" && link.exe @"'+str(out/'link.rsp')+'"'
 batch=out/'build.cmd';batch.write_text('@echo off\n'+command+'\n',encoding='utf8')
 result=subprocess.run(['cmd.exe','/d','/c',str(batch)],capture_output=True)
 (out/'build.log').write_bytes(result.stdout+b'\n'+result.stderr)
 if result.returncode or not binary.exists():raise RuntimeError('separate GPU bridge build failed: '+str(out/'build.log'))
record={**base,'source':str(source),'source_sha256':digest,'binary':str(binary),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'base_bridge_source_sha256':base['source_sha256'],'build_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'log':str(out/'build.log'),'available_gpu_policies':{'cgnr':'CUDA_SPARSE / JACOBI / FP64','dense_schur':'CUDA dense Cholesky / FP64; CPU Schur elimination'},'unchanged_cpu_policies':['sparse_schur','sparse_normal_cholesky','dense_qr','iterative_schur'],'actual_Ceres_solve_validated':False,'flags':flags,'link_flags':links,'flags_file':str(out/'compile.rsp'),'link_flags_file':str(out/'link.rsp')}
manifest.write_text(json.dumps(record,indent=2),encoding='utf8');print(manifest)
