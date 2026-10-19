"""Build only the explicit source-image CUDA shadow; outputs remain in Data."""
from pathlib import Path
import argparse,subprocess,os,json,hashlib,time
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out.resolve();out.mkdir();tmp=out/'tmp';tmp.mkdir()
    vc=Path(r'D:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat')
    setup=subprocess.run('cmd.exe /d /s /c '+chr(34)+'call '+chr(34)+str(vc)+chr(34)+' >nul && set'+chr(34),capture_output=True,text=True,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    env={k.upper():v for k,v in os.environ.items()}
    env.update({k.upper():v for k,v in(x.split('=',1)for x in setup.stdout.splitlines()if '='in x and not x.startswith('='))})
    cuda=Path(r'D:/Program Files/CUDA')
    env['PATH']=str(cuda/'bin')+';'+env['PATH'];env['TMP']=env['TEMP']=str(tmp)
    src=Path(__file__).with_name('source_filter.cu');library=out/'source_filter.dll'
    cmd=[str(cuda/'bin/nvcc.exe'),'-v','--use-local-env','-shared','-std=c++17','-O3','-arch=sm_120','--fmad=false','--prec-div=true','--prec-sqrt=true','-Xcompiler','/MD,/utf-8',str(src),'-o',str(library)]
    t=time.perf_counter();r=subprocess.run(cmd,cwd=out,env=env,capture_output=True,text=True,errors='replace',creationflags=subprocess.CREATE_NO_WINDOW)
    (out/'build.log').write_text(r.stdout+r.stderr,encoding='utf8')
    record=dict(command=cmd,source=dict(file=str(src),sha256=hashlib.sha256(src.read_bytes()).hexdigest()),returncode=r.returncode,seconds=time.perf_counter()-t)
    if library.exists():record['library']=dict(file=str(library),sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    (out/'build_recipe.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    print(json.dumps(record),flush=True)
    raise SystemExit(r.returncode)
if __name__=='__main__':main()
