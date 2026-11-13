"""Windows memory telemetry with an optional process-local commit ceiling.

The limit applies only to this solver process. It does not trim or terminate
other applications, and it never changes the mathematical fitting problem.
"""
from pathlib import Path
import ctypes,json,os,threading,time
from ctypes import wintypes

class ProcessMemoryGuard:
    def __init__(self,output,limit_gib=0.):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.limit=int(float(limit_gib)*1024**3);self.peak=0;self._stop=threading.Event()
        if limit_gib<0:raise ValueError('Memory ceiling must be nonnegative')
        if os.name!='nt':raise RuntimeError('Windows process memory guard requested on another platform')
        size=ctypes.c_size_t
        class BASIC(ctypes.Structure):
            _fields_=[('PerProcessUserTimeLimit',ctypes.c_int64),('PerJobUserTimeLimit',ctypes.c_int64),('LimitFlags',wintypes.DWORD),('MinimumWorkingSetSize',size),('MaximumWorkingSetSize',size),('ActiveProcessLimit',wintypes.DWORD),('Affinity',size),('PriorityClass',wintypes.DWORD),('SchedulingClass',wintypes.DWORD)]
        class IO(ctypes.Structure):
            _fields_=[(k,ctypes.c_uint64)for k in('ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]
        class EXTENDED(ctypes.Structure):
            _fields_=[('BasicLimitInformation',BASIC),('IoInfo',IO),('ProcessMemoryLimit',size),('JobMemoryLimit',size),('PeakProcessMemoryUsed',size),('PeakJobMemoryUsed',size)]
        class COUNTERS(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(k,size)for k in('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
        class GLOBAL(ctypes.Structure):
            _fields_=[('dwLength',wintypes.DWORD),('dwMemoryLoad',wintypes.DWORD)]+[(k,ctypes.c_uint64)for k in('ullTotalPhys','ullAvailPhys','ullTotalPageFile','ullAvailPageFile','ullTotalVirtual','ullAvailVirtual','ullAvailExtendedVirtual')]
        self.EXTENDED=EXTENDED;self.COUNTERS=COUNTERS
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
        kernel.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR];kernel.CreateJobObjectW.restype=wintypes.HANDLE
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
        kernel.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]
        kernel.QueryInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.c_void_p]
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
        self.kernel,self.psapi=kernel,psapi;self.process=kernel.GetCurrentProcess();self.job=None
        self.GLOBAL=GLOBAL;kernel.GlobalMemoryStatusEx.argtypes=[ctypes.POINTER(GLOBAL)]
        kernel.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
        if self.limit:
            self.job=kernel.CreateJobObjectW(None,None)
            info=EXTENDED();info.BasicLimitInformation.LimitFlags=0x100;info.ProcessMemoryLimit=self.limit
            if not self.job or not kernel.SetInformationJobObject(self.job,9,ctypes.byref(info),ctypes.sizeof(info)) or not kernel.AssignProcessToJobObject(self.job,self.process):
                raise ctypes.WinError(ctypes.get_last_error())
        self.started=time.perf_counter();self.thread=threading.Thread(target=self._record,daemon=True,name='solver-memory');self.thread.start()

    def snapshot(self):
        counter=self.COUNTERS();counter.cb=ctypes.sizeof(counter)
        if not self.psapi.GetProcessMemoryInfo(self.process,ctypes.byref(counter),counter.cb):raise ctypes.WinError(ctypes.get_last_error())
        peak=int(counter.PeakPagefileUsage)
        if self.job is not None:
            info=self.EXTENDED()
            if not self.kernel.QueryInformationJobObject(self.job,9,ctypes.byref(info),ctypes.sizeof(info),None):raise ctypes.WinError(ctypes.get_last_error())
            peak=int(info.PeakProcessMemoryUsed)
        self.peak=max(self.peak,peak)
        system=self.GLOBAL();system.dwLength=ctypes.sizeof(system)
        if not self.kernel.GlobalMemoryStatusEx(ctypes.byref(system)):raise ctypes.WinError(ctypes.get_last_error())
        times=[wintypes.FILETIME()for _ in range(4)]
        if not self.kernel.GetProcessTimes(self.process,*[ctypes.byref(v)for v in times]):raise ctypes.WinError(ctypes.get_last_error())
        cpu=[(int(v.dwHighDateTime)*2**32+int(v.dwLowDateTime))*1e-7 for v in times[2:]]
        return dict(seconds=time.perf_counter()-self.started,private_bytes=int(counter.PrivateUsage),working_set_bytes=int(counter.WorkingSetSize),peak_working_set_bytes=int(counter.PeakWorkingSetSize),peak_commit_bytes=self.peak,limit_bytes=self.limit,system_free_physical_bytes=int(system.ullAvailPhys),available_commit_bytes=int(system.ullAvailPageFile),cpu_kernel_seconds=cpu[0],cpu_user_seconds=cpu[1],cpu_core_seconds=sum(cpu),pid=os.getpid())

    def _record(self):
        with (self.output/'memory.jsonl').open('a',encoding='utf-8',buffering=1)as stream:
            while not self._stop.is_set():
                stream.write(json.dumps(self.snapshot())+'\n');self._stop.wait(3.)

    def close(self):
        self._stop.set();self.thread.join(timeout=2.)
        (self.output/'memory_summary.json').write_text(json.dumps(self.snapshot(),indent=2),encoding='utf-8')
        # Closing a non-KILL_ON_JOB_CLOSE handle never terminates a process.
        if self.job is not None:
            self.kernel.CloseHandle.argtypes=[wintypes.HANDLE];self.kernel.CloseHandle(self.job)
