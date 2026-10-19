"""Owned-process memory counters for future immutable source timing receipts."""
import os,ctypes
from ctypes import wintypes as w

def snapshot():
    if os.name!='nt':return dict(available=False,reason='Windows private counters unavailable')
    class M(ctypes.Structure):
        _fields_=[('cb',w.DWORD),('PageFaultCount',w.DWORD)]+[(x,ctypes.c_size_t)for x in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=w.HANDLE;k.GetCurrentProcessId.restype=w.DWORD;k.K32GetProcessMemoryInfo.argtypes=[w.HANDLE,ctypes.POINTER(M),w.DWORD];k.K32GetProcessMemoryInfo.restype=w.BOOL
    data=M();data.cb=ctypes.sizeof(data)
    if not k.K32GetProcessMemoryInfo(k.GetCurrentProcess(),ctypes.byref(data),data.cb):return dict(available=False,error=ctypes.get_last_error())
    return dict(available=True,pid=int(k.GetCurrentProcessId()),private_bytes=int(data.PrivateUsage),working_set_bytes=int(data.WorkingSetSize),peak_private_commit_bytes=int(data.PeakPagefileUsage),peak_working_set_bytes=int(data.PeakWorkingSetSize))
