"""Build the shared joint solver, reusing only matching current objects."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse, hashlib, importlib.util, json, os, re, shutil, subprocess, time

ROOT=Path(__file__).resolve().parents[3]
SOURCES=('augmented_lagrangian','checkpoint','ceres_backend','event_model','event_builder','joint_model','joint_packet','joint_builder','joint_capture','joint_al','joint_gpu_cache','joint_gpu_driver','joint_recovery','joint_fit')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def source_validation_identity(cpp):
    """Shared compile-time ABI over the actual native source-input consumers."""
    cpp=Path(cpp);inc=cpp/'include';seen=set()
    def visit(path):
        if path in seen:return
        seen.add(path)
        for name in re.findall(r'#include\s+"([^"]+)"',path.read_text('utf8')):
            q=path.parent/name
            if not q.is_file():q=inc/name
            if q.is_file():visit(q)
    roots=[inc/'rinkfit/joint_dense_source_guard_v1.hpp',
      inc/'rinkfit/joint_effective_source_guard_v1.hpp',
      inc/'rinkfit/joint_dense_capture_weights_v1.hpp',
      inc/'rinkfit/joint_engineering_budget_v2.hpp',
      cpp/'src/joint_packet.cpp',cpp/'src/joint_model.cpp',cpp/'src/event_model.cpp']
    for path in roots:visit(path)
    values={str(p):digest(p) for p in sorted(seen)}
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest(),values


def save_source_snapshot(destination, cpp, sources, executable_sha, validator_abi):
    """Copy exact compiled inputs; never alias writable source files."""
    destination=Path(destination).resolve();cpp=Path(cpp).resolve()
    destination.mkdir(parents=True,exist_ok=True);entries=[]
    for original,expected in sorted(sources.items()):
        original=Path(original).resolve();relative=original.relative_to(cpp)
        if original.suffix not in ('.hpp','.cpp'):
            raise ValueError('Compiled source snapshot only accepts C++ source/header files')
        data=original.read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected:
            raise RuntimeError('Source changed before immutable snapshot: '+str(original))
        target=destination/'sources'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if digest(target)!=expected:raise RuntimeError('Immutable source snapshot changed: '+str(target))
        else:
            with target.open('xb') as stream:stream.write(data)
        entries.append(dict(original_file=str(original),file=str(target),sha256=expected,byte_count=len(data)))
    manifest=dict(schema='rinkfit.compiled_source_snapshot.v1',executable_sha256=executable_sha,
                  source_validation_abi_hash=validator_abi,sources=entries,byte_copies_not_links=True)
    path=destination/'source_snapshot.json';data=(json.dumps(manifest,indent=2)+'\n').encode('utf8')
    if path.exists():
        if path.read_bytes()!=data:raise RuntimeError('Immutable source snapshot manifest changed')
    else:
        with path.open('xb') as stream:stream.write(data)
    return dict(file=str(path),sha256=digest(path))


def build(output,compiler=None,jobs=3,ceres_prefix=None):
    started=time.perf_counter();out=Path(output).resolve();cpp=ROOT/'calibration/rinkfit/cpp';inc=cpp/'include';out.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('rinkfit_existing_environment',cpp/'gpu_rows/build.py');environment=importlib.util.module_from_spec(spec);spec.loader.exec_module(environment)
    env=environment.environment();prefix=Path(ceres_prefix).resolve()if ceres_prefix else ROOT/'environments/cg22p';runtime=[str(prefix/'bin'),str(environment.D/'glog/Library/bin'),str(environment.D/'gflags/Library/bin')]
    cuda=os.environ.get('CUDA_PATH')
    if cuda:runtime.append(str(Path(cuda)/'bin/x64'))
    env['PATH']=os.pathsep.join(runtime)+os.pathsep+env['PATH'];compiler=compiler or shutil.which('cl.exe',path=env['PATH'])
    if not compiler:raise RuntimeError('MSVC compiler not available in the existing build environment')
    (out/'tmp').mkdir(exist_ok=True);env['TMP']=env['TEMP']=str(out/'tmp')
    sources=[cpp/'src'/(n+'.cpp') for n in SOURCES];tracked=sources+sorted(inc.rglob('*.hpp'));initial={str(p):digest(p)for p in tracked}
    old=json.loads((out/'recipe.json').read_text('utf8'))if(out/'recipe.json').exists()else{}
    ceres_headers={p.relative_to(prefix/'include').as_posix():digest(p)for p in (prefix/'include/ceres').rglob('*.h')}
    previous_ceres_headers=old.get('ceres_public_headers')
    if previous_ceres_headers is None:
        previous_ceres_headers={p.relative_to(ROOT/'environments/cg22p/include').as_posix():digest(p)for p in (ROOT/'environments/cg22p/include/ceres').rglob('*.h')}
    ceres_headers_unchanged=ceres_headers==previous_ceres_headers
    def dependencies(path,seen=None):
        seen=set()if seen is None else seen
        if path in seen:return seen
        seen.add(path)
        for name in re.findall(r'#include\s+"([^"]+)"',path.read_text('utf8')):
            q=path.parent/name
            if not q.is_file():q=inc/name
            if q.is_file():dependencies(q,seen)
        return seen
    validator=inc/'rinkfit/joint_dense_source_guard_v1.hpp'
    abi,validator_sources=source_validation_identity(cpp)
    forced=out/'validation_build.hpp';forced.write_text('#define RINKFIT_SOURCE_VALIDATOR_SHA "'+abi+'"\n','utf8')
    includes=[inc,prefix/'include',environment.D/'eigen/Library/include/eigen3',environment.D/'glog/Library/include',environment.D/'gflags/Library/include',environment.D/'nlohmann_json/Library/include']
    common=[str(compiler),'/nologo','/std:c++17','/EHsc','/O2','/MD','/utf-8','/bigobj','/DGLOG_USE_GLOG_EXPORT','/DGLOG_USE_GFLAGS','/DGLOG_NO_ABBREVIATED_SEVERITIES',*['/I'+str(p)for p in includes],'/FI'+str(forced)]
    commands=[];objects={};previous=old.get('sources_and_headers',{})
    def compile_one(path):
        obj=out/(path.stem+'.obj');deps=dependencies(path);known=old.get('objects',{}).get(path.stem)
        same=all(previous.get(str(p))==initial[str(p)]for p in deps)
        reusable=obj.exists()and same and ceres_headers_unchanged and (old.get('source_validation_abi_hash')==abi or validator not in deps)
        if reusable and known:reusable=known['sha256']==digest(obj)
        if reusable and not known:reusable=old.get('source_bytes_unchanged_during_build')is True and any(c.get('code')==0 and '/c'in c.get('command',[])and str(path)in c['command']for c in old.get('commands',[]))
        if reusable:
            result=dict(reused_object=str(obj),sha256=digest(obj),source_unchanged=True,code=0);print(path.name,'reused',flush=True)
        else:
            command=common+['/c',str(path),'/Fo:'+str(obj)];r=subprocess.run(command,cwd=out,env=env,capture_output=True,text=True,errors='replace');(out/(path.stem+'.log')).write_text(r.stdout+r.stderr,'utf8');result=dict(command=command,code=r.returncode)
            print(path.name,r.returncode,flush=True)
            if r.returncode:raise RuntimeError(path.name+' failed; see '+str(out/(path.stem+'.log')))
        commands.append(result);objects[path.stem]=dict(file=str(obj),sha256=digest(obj))
    with ThreadPoolExecutor(max_workers=jobs)as workers:list(workers.map(compile_one,sources))
    if initial!={str(p):digest(p)for p in tracked}:raise RuntimeError('Source changed during build; current executable not replaced')
    temporary=out/'rinkfit_joint_fit.build.exe';command=[str(compiler),'/nologo',*[str(out/(p.stem+'.obj'))for p in sources],str(prefix/'lib/ceres.lib'),str(environment.D/'glog/Library/lib/glog.lib'),str(environment.D/'gflags/Library/lib/gflags.lib'),'/Fe:'+str(temporary)]
    r=subprocess.run(command,cwd=out,env=env,capture_output=True,text=True,errors='replace');(out/'link.log').write_text(r.stdout+r.stderr,'utf8');commands.append(dict(command=command,code=r.returncode))
    if r.returncode:raise RuntimeError('Native link failed; see '+str(out/'link.log'))
    r=subprocess.run([str(temporary),'--capabilities'],cwd=ROOT,env=env,capture_output=True,text=True,errors='replace');(out/'capabilities.log').write_text(r.stdout+r.stderr,'utf8')
    if r.returncode:raise RuntimeError('Native capabilities entry failed')
    executable=out/'rinkfit_joint_fit.exe';os.replace(temporary,executable)
    recipe=dict(schema='rinkfit.audit_native_build.v1',ceres_prefix=str(prefix),ceres_public_headers=ceres_headers,ceres_library_sha256=digest(prefix/'bin/ceres.dll'),ceres_import_library_sha256=digest(prefix/'lib/ceres.lib'),source_files={p.relative_to(cpp).as_posix():h for p,h in((Path(k),v)for k,v in initial.items())},runtime_bin_paths=runtime,executable_sha256=digest(executable),commands=commands,sources_and_headers=initial,source_bytes_unchanged_during_build=True,source_validation_abi_hash=abi,objects=objects,returncode=0,seconds=time.perf_counter()-started,Solve_started=False)
    temp=out/'recipe.build.json';temp.write_text(json.dumps(recipe,indent=2),'utf8');os.replace(temp,out/'recipe.json')
    registered=out/'artifacts'/recipe['executable_sha256'][:16];registered.mkdir(parents=True,exist_ok=True);registered_exe=registered/executable.name
    if not registered_exe.exists():
        try:os.link(executable,registered_exe)
        except OSError as error:
            # Sandboxed Windows volumes can forbid hard links while allowing
            # ordinary writes. Preserve the same immutable, verified bytes.
            if getattr(error,'winerror',None)!=5 and error.errno!=18:raise
            with executable.open('rb')as source,registered_exe.open('xb')as destination:
                shutil.copyfileobj(source,destination,1024*1024)
    if digest(registered_exe)!=recipe['executable_sha256']:
        raise RuntimeError('Registered native artifact differs from the verified build')
    recipe['compiled_source_snapshot_v1']=save_source_snapshot(registered,cpp,initial,recipe['executable_sha256'],abi)
    # Publish the snapshot binding only after every copied input has passed SHA.
    temp=out/'recipe.build.json';temp.write_text(json.dumps(recipe,indent=2),'utf8');os.replace(temp,out/'recipe.json')
    registered_recipe=registered/'recipe.json'
    if not registered_recipe.exists():registered_recipe.write_text(json.dumps(recipe,indent=2),'utf8')
    print(json.dumps(dict(native=str(registered_exe),compiled=sum('command'in c and '/c'in c['command']for c in commands),reused=sum('reused_object'in c for c in commands),seconds=recipe['seconds'])),flush=True)
    return registered_exe

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--compiler');p.add_argument('--jobs',type=int,default=3);p.add_argument('--ceres-prefix');a=p.parse_args();build(a.output,a.compiler,a.jobs,a.ceres_prefix)

if __name__=='__main__':main()
