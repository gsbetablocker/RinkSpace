"""Run an explicit prepared programme/full seed, recording identity and peaks.

This never prepares observations, chooses a reduced scope, or limits trials.
GPU memory is the device-wide NVML high-water sample, not per-process memory.
"""
from __future__ import annotations
import argparse,ctypes as c,hashlib,importlib.metadata,json,os,platform,subprocess,sys,time
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','runner','library','native-source','run-root','seed','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--motion-storage',choices=('dense','sparse'),default='sparse')
    ap.add_argument('--dual-method',choices=('cached','working_set'),default='working_set')
    ap.add_argument('--verify-response-cache',action='store_true',help='diagnostic re-solves; exclude from final timing')
    ap.add_argument('--station-geometry-cache',choices=('0','1'),default='1')
    ap.add_argument('--ice-chart',choices=('auto','legacy','actual'),default='auto')
    ap.add_argument('--initial-lambda',type=float,default=0.)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_PATH=r'D:\Program Files\CUDA',
        RINKSPACE_SMALL_DUAL_BACKEND='cpu',RINKSPACE_STATIONARITY_BACKEND='cpu',
        RINKSPACE_STATIONARITY_FACE_BACKEND='parallel',RINKSPACE_STATIONARITY_CPU_SPARSE='1',RINKSPACE_STATIONARITY_CACHE='1',
        RINKSPACE_VERIFY_RESPONSE_CACHE=str(int(args.verify_response_cache)),RINKSPACE_DENSE_MOTION_ORACLE='0',RINKSPACE_MOTION_DIRECTION_STORAGE=args.motion_storage,
        RINKSPACE_SMALL_DUAL_METHOD=args.dual_method,RINKSPACE_STATION_GEOMETRY_CACHE=args.station_geometry_cache)
    for key in list(env):
        if key.startswith('RINKSPACE_TRACE_'):env.pop(key)
    programme=args.repo/'events/ankara_2026/programmes/okada_mei_sp'
    env['PYTHONPATH']=os.pathsep.join(map(str,(args.repo/'src',programme,args.repo)))
    command=[sys.executable,'-B',str(args.repo/'scripts/run_native_entry.py'),'--library',str(args.library),str(args.runner),'--run-root',str(args.run_root),'--resume',str(args.seed),
        '--native-library',str(args.library),'--output',str(args.out),'--phase','joint','--compact-diagnostics',
        '--initial-lambda',str(args.initial_lambda),'--relative-cost-tolerance','1e-10','--ice-chart',args.ice_chart]
    manifest=json.loads((args.run_root/'joint_manifest.json').read_text(encoding='utf8'))
    native_files={p.relative_to(args.native_source).as_posix():digest(p)
        for p in args.native_source.rglob('*') if p.suffix in ('.cu','.cuh','.h','.hpp','.cpp')}
    python_files=[args.runner,*[args.repo/p for p in ('src/rinkspace/core/gpu.py','src/rinkspace/domain/multi_shot_motion.py',
        'src/rinkspace/domain/motion_chart.py','src/rinkspace/domain/ice_actual_chart.py','src/rinkspace/session/gpu_geometry.py')]]
    hardware=dict(python=sys.version,platform=platform.platform(),processor=platform.processor(),
        packages={k:importlib.metadata.version(k) for k in ('numpy','scipy')})
    if os.name=='nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            hardware['cpu_name']=winreg.QueryValueEx(key,'ProcessorNameString')[0]
        hardware['gpu_query']=subprocess.run([str(Path(os.environ['SystemRoot'])/'System32/nvidia-smi.exe'),
            '--query-gpu=name,driver_version,compute_cap,memory.total','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    receipt=dict(command=command,run_root=str(args.run_root),full_seed=str(args.seed),seed_sha256=digest(args.seed),
        hardware=hardware,benchmark_driver_sha256=digest(__file__),
        library=str(args.library),library_sha256=digest(args.library),native_source=str(args.native_source),native_source_sha256=native_files,
        python_source_sha256={str(p):digest(p) for p in python_files},point_input_sha256=digest(args.run_root/'point_observations.npz'),
        scene_sha256=digest(manifest['scene_path']),environment={k:v for k,v in env.items() if k.startswith('RINKSPACE_') or k=='CUDA_PATH'},
        memory_note='host PeakWorkingSetSize and PeakPagefileUsage from Windows; GPU NVML device-total samples at 0.2 s; includes other processes')
    cache=args.library.parent/'CMakeCache.txt'
    if cache.exists():receipt['cmake_cache']=str(cache);receipt['cmake_cache_sha256']=digest(cache)
    (args.out/'provenance.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    dirty=subprocess.run(['git','-C',str(args.repo),'diff','--binary'],capture_output=True,check=True).stdout
    (args.out/'source_dirty.diff').write_bytes(dirty)
    class Memory(c.Structure):
        _fields_=[('cb',c.c_uint32),('faults',c.c_uint32)]+[(k,c.c_size_t) for k in
            ('peak_working','working','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
    class GPU(c.Structure):_fields_=[('total',c.c_ulonglong),('free',c.c_ulonglong),('used',c.c_ulonglong)]
    class FileTime(c.Structure):
        _fields_=[('low',c.c_uint32),('high',c.c_uint32)]
        def seconds(self):return ((self.high<<32)|self.low)*1e-7
    nvml=None;device=c.c_void_p();gpu=GPU();peak_gpu=None
    try:
        nvml=c.WinDLL('nvml.dll');nvml.nvmlInit_v2.restype=c.c_int
        if nvml.nvmlInit_v2()!=0 or nvml.nvmlDeviceGetHandleByIndex_v2(0,c.byref(device))!=0:raise RuntimeError('NVML unavailable')
        if nvml.nvmlDeviceGetMemoryInfo(device,c.byref(gpu))==0:receipt['gpu_used_before_bytes']=gpu.used
    except (OSError,AttributeError,RuntimeError):nvml=None
    peak_host=peak_commit=0;started=time.perf_counter();interpreter_handle=None;interpreter_pid=None
    if os.name=='nt':
        c.windll.kernel32.OpenProcess.argtypes=[c.c_uint32,c.c_int,c.c_uint32]
        c.windll.kernel32.OpenProcess.restype=c.c_void_p
    with (args.out/'console.log').open('wb') as console,(args.out/'cpu_samples.jsonl').open('w',encoding='utf8') as samples:
        process=subprocess.Popen(command,cwd=args.repo,env=env,stdout=console,stderr=subprocess.STDOUT)
        print(f'MAIN LOG: {args.out / "solve.jsonl"}; PID {process.pid}',flush=True)
        while True:
            if os.name=='nt':
                identity=args.out/'process_identity.json'
                if interpreter_handle is None and identity.exists():
                    interpreter_pid=json.loads(identity.read_text())['pid']
                    interpreter_handle=c.windll.kernel32.OpenProcess(0x0410,False,interpreter_pid)
                memory=Memory();memory.cb=c.sizeof(memory)
                handle=interpreter_handle or int(process._handle)
                if c.windll.psapi.GetProcessMemoryInfo(c.c_void_p(handle),c.byref(memory),c.sizeof(memory)):
                    peak_host=max(peak_host,memory.peak_working);peak_commit=max(peak_commit,memory.peak_pagefile)
                created=FileTime();exited=FileTime();kernel=FileTime();user=FileTime()
                if c.windll.kernel32.GetProcessTimes(c.c_void_p(handle),c.byref(created),c.byref(exited),c.byref(kernel),c.byref(user)):
                    samples.write(json.dumps(dict(elapsed_s=time.perf_counter()-started,user_s=user.seconds(),kernel_s=kernel.seconds(),working_bytes=memory.working))+'\n')
            if nvml is not None and nvml.nvmlDeviceGetMemoryInfo(device,c.byref(gpu))==0:
                peak_gpu=max(peak_gpu or 0,gpu.used)
            try:code=process.wait(timeout=.2);break
            except subprocess.TimeoutExpired:pass
    result=dict(exit_code=code,process_wall_s=time.perf_counter()-started,peak_host_working_bytes=peak_host,
        peak_host_commit_bytes=peak_commit,peak_gpu_device_used_bytes=peak_gpu,gpu_used_before_bytes=receipt.get('gpu_used_before_bytes'),interpreter_pid=interpreter_pid)
    if interpreter_handle:c.windll.kernel32.CloseHandle(c.c_void_p(interpreter_handle))
    (args.out/'process_metrics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result),flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
