"""Launch one RinkSpace worker with D-only caches and a Windows job limit.

No system paging configuration is changed. The limit is an engineering
resource guard for this job, not a source-data limit or accuracy criterion.
"""
from __future__ import annotations
import argparse, ctypes, json, os, subprocess, sys, time
from ctypes import wintypes as w
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path('D:/CodexData/.codex/visualizations/2026/10/02/private-thread-00008/RinkSpace_tmp/runtime_d_only')

class MemoryStatus(ctypes.Structure):
    _fields_ = [('length',w.DWORD),('load',w.DWORD)] + [(n,ctypes.c_ulonglong) for n in
        ('total_physical','available_physical','total_commit','available_commit','total_virtual','available_virtual','extended_virtual')]

class BasicLimits(ctypes.Structure):
    _fields_ = [('process_time',ctypes.c_longlong),('job_time',ctypes.c_longlong),('flags',w.DWORD),
        ('minimum_working',ctypes.c_size_t),('maximum_working',ctypes.c_size_t),('active_processes',w.DWORD),
        ('affinity',ctypes.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]

class IOCounters(ctypes.Structure):
    _fields_ = [(n,ctypes.c_ulonglong) for n in ('read_operations','write_operations','other_operations','read_bytes','write_bytes','other_bytes')]

class ExtendedLimits(ctypes.Structure):
    _fields_ = [('basic',BasicLimits),('io',IOCounters),('process_memory_limit',ctypes.c_size_t),
        ('job_memory_limit',ctypes.c_size_t),('peak_process_memory',ctypes.c_size_t),('peak_job_memory',ctypes.c_size_t)]

class StartupInfo(ctypes.Structure):
    _fields_ = [('cb',w.DWORD),('reserved',w.LPWSTR),('desktop',w.LPWSTR),('title',w.LPWSTR),
        ('x',w.DWORD),('y',w.DWORD),('width',w.DWORD),('height',w.DWORD),('x_chars',w.DWORD),
        ('y_chars',w.DWORD),('fill',w.DWORD),('flags',w.DWORD),('show',w.WORD),('reserved2_size',w.WORD),
        ('reserved2',ctypes.POINTER(ctypes.c_byte)),('stdin',w.HANDLE),('stdout',w.HANDLE),('stderr',w.HANDLE)]

class ProcessInfo(ctypes.Structure):
    _fields_ = [('process',w.HANDLE),('thread',w.HANDLE),('process_id',w.DWORD),('thread_id',w.DWORD)]

def memory_status(kernel):
    m=MemoryStatus();m.length=ctypes.sizeof(m)
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError(ctypes.get_last_error())
    return {n:int(getattr(m,n)) for n in ('total_physical','available_physical','total_commit','available_commit')}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--memory-mib',type=int,required=True)
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--allow-existing-paging',action='store_true',
        help='User explicitly permits normal system paging; keep the physical-memory safety margin without changing Windows settings')
    p.add_argument('script',type=Path);p.add_argument('arguments',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if os.name!='nt':raise RuntimeError('This launcher requires a Windows job object')
    for path in (a.script.resolve(),a.log.resolve(),RUNTIME.resolve()):
        if path.drive.upper()!='D:':raise ValueError('Worker script, logs and runtime directories must be on D:')
    if a.memory_mib<=0:raise ValueError('A positive explicit engineering memory budget is required')
    directories={n:RUNTIME/n for n in ('temp','numba','matplotlib','cache','config','torch','huggingface','ultralytics')}
    for path in directories.values():path.mkdir(parents=True,exist_ok=True)
    a.log.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    for key,folder in {'TEMP':'temp','TMP':'temp','NUMBA_CACHE_DIR':'numba','MPLCONFIGDIR':'matplotlib',
        'XDG_CACHE_HOME':'cache','XDG_CONFIG_HOME':'config','TORCH_HOME':'torch','HF_HOME':'huggingface',
        'YOLO_CONFIG_DIR':'ultralytics','OPENCV_TEMP_PATH':'temp','PIP_CACHE_DIR':'cache'}.items():env[key]=str(directories[folder])
    for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        env[key]='1'
    env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONNOUSERSITE']='1'
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.CreateJobObjectW.argtypes=[ctypes.c_void_p,w.LPCWSTR];k.CreateJobObjectW.restype=w.HANDLE
    k.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD]
    k.QueryInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD,ctypes.c_void_p]
    k.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE]
    k.TerminateJobObject.argtypes=[w.HANDLE,w.UINT]
    k.TerminateProcess.argtypes=[w.HANDLE,w.UINT]
    k.ResumeThread.argtypes=[w.HANDLE];k.ResumeThread.restype=w.DWORD
    k.WaitForSingleObject.argtypes=[w.HANDLE,w.DWORD];k.WaitForSingleObject.restype=w.DWORD
    k.GetExitCodeProcess.argtypes=[w.HANDLE,ctypes.POINTER(w.DWORD)]
    k.CloseHandle.argtypes=[w.HANDLE]
    k.CreateProcessW.argtypes=[w.LPCWSTR,w.LPWSTR,ctypes.c_void_p,ctypes.c_void_p,w.BOOL,w.DWORD,
        ctypes.c_void_p,w.LPCWSTR,ctypes.POINTER(StartupInfo),ctypes.POINTER(ProcessInfo)]
    before=memory_status(k);budget=a.memory_mib*1024**2
    # Keep this job below both physical and commitment headroom. This cannot
    # control unrelated apps or promise that Windows never writes its pagefile.
    headroom=min(int(before['available_physical']*.40),
        int(before['total_commit']*.70)-(before['total_commit']-before['available_commit']))
    if a.allow_existing_paging:headroom=int(before['available_physical']*.75)
    if budget>headroom:raise RuntimeError('Requested job budget exceeds conservative current memory headroom; no worker started')
    job=k.CreateJobObjectW(None,None)
    if not job:raise ctypes.WinError(ctypes.get_last_error())
    limits=ExtendedLimits();limits.basic.flags=0x2000|0x200;limits.job_memory_limit=budget
    process=ProcessInfo();started=False
    report=dict(script=str(a.script.resolve()),memory_budget_mib=a.memory_mib,
        engineering_resource_guard_not_user_capacity_requirement=True,memory_before=before,
        existing_system_paging_explicitly_permitted=a.allow_existing_paging,
        cache_directories={n:str(v) for n,v in directories.items()},system_pagefile_configuration_changed=False)
    try:
        if not k.SetInformationJobObject(job,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        import msvcrt
        with a.log.open('ab',buffering=0) as stream,open(os.devnull,'rb',buffering=0) as empty:
            handles=[msvcrt.get_osfhandle(f.fileno()) for f in (empty,stream)]
            for h in handles:os.set_handle_inheritable(h,True)
            si=StartupInfo();si.cb=ctypes.sizeof(si);si.flags=0x100;si.stdin=handles[0];si.stdout=si.stderr=handles[1]
            command=ctypes.create_unicode_buffer(subprocess.list2cmdline([str(ROOT/'.venv/Scripts/python.exe'),'-B',str(a.script.resolve()),*a.arguments]))
            environment=ctypes.create_unicode_buffer('\0'.join(f'{key}={value}' for key,value in sorted(env.items()))+'\0\0')
            if not k.CreateProcessW(None,command,None,None,True,0x4|0x400,environment,str(ROOT),ctypes.byref(si),ctypes.byref(process)):
                raise ctypes.WinError(ctypes.get_last_error())
            if not k.AssignProcessToJobObject(job,process.process):
                k.TerminateProcess(process.process,1);raise ctypes.WinError(ctypes.get_last_error())
            if k.ResumeThread(process.thread)==0xffffffff:raise ctypes.WinError(ctypes.get_last_error())
            started=True;report['process_id']=int(process.process_id)
            print(json.dumps(dict(stage='D_only_worker_started',**report)),flush=True)
            while k.WaitForSingleObject(process.process,1000)==258:
                current=memory_status(k)
                if current['available_physical']<int(before['total_physical']*.10)or (not a.allow_existing_paging and
                        (current['total_commit']-current['available_commit'])>int(current['total_commit']*.80)):
                    report['stop_reason']='Current physical/commit pressure exceeded engineering resource guard'
                    k.TerminateJobObject(job,137);break
            k.WaitForSingleObject(process.process,10000)
            code=w.DWORD()
            if not k.GetExitCodeProcess(process.process,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
            report['exit_code']=int(code.value)
    except KeyboardInterrupt:
        k.TerminateJobObject(job,130);report['exit_code']=130;report['stop_reason']='Explicit interruption'
    finally:
        if started:
            peaks=ExtendedLimits()
            if k.QueryInformationJobObject(job,9,ctypes.byref(peaks),ctypes.sizeof(peaks),None):
                report['peak_job_accounted_memory_bytes']=int(peaks.peak_job_memory)
        report['memory_after']=memory_status(k)
        for h in (process.thread,process.process,job):
            if h:k.CloseHandle(h)
        a.log.with_suffix(a.log.suffix+'.runtime.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(stage='D_only_worker_finished',**report)),flush=True)
    return report.get('exit_code',1)

if __name__=='__main__':raise SystemExit(main())
