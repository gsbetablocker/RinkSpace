"""Reuse input digests only for the identical Windows file version.

The source proof remains authoritative. Changed files are hashed against its
expected digest; this cache only avoids reading unchanged immutable inputs.
"""
from __future__ import annotations
import ctypes, hashlib, json, os, time
from ctypes import wintypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

class _Basic(ctypes.Structure):
    _fields_=[('creation',ctypes.c_longlong),('access',ctypes.c_longlong),('write',ctypes.c_longlong),('change',ctypes.c_longlong),('attributes',wintypes.DWORD)]
class _Info(ctypes.Structure):
    _fields_=[('attributes',wintypes.DWORD),('creation',wintypes.FILETIME),('access',wintypes.FILETIME),('write',wintypes.FILETIME),('volume',wintypes.DWORD),('size_hi',wintypes.DWORD),('size_lo',wintypes.DWORD),('links',wintypes.DWORD),('index_hi',wintypes.DWORD),('index_lo',wintypes.DWORD)]

def fingerprint(path):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    create=kernel.CreateFileW;create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE];create.restype=wintypes.HANDLE
    infofn=kernel.GetFileInformationByHandle;infofn.argtypes=[wintypes.HANDLE,ctypes.POINTER(_Info)];infofn.restype=wintypes.BOOL
    basicfn=kernel.GetFileInformationByHandleEx;basicfn.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD];basicfn.restype=wintypes.BOOL
    close=kernel.CloseHandle;close.argtypes=[wintypes.HANDLE];close.restype=wintypes.BOOL
    handle=create(str(path),0x80,7,None,3,0,None)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    try:
        info=_Info();basic=_Basic()
        if not infofn(handle,ctypes.byref(info)) or not basicfn(handle,0,ctypes.byref(basic),ctypes.sizeof(basic)):raise ctypes.WinError(ctypes.get_last_error())
        return ':'.join(map(str,[info.volume,(info.index_hi<<32)|info.index_lo,(info.size_hi<<32)|info.size_lo,basic.creation,basic.write,basic.change]))
    finally:close(handle)

def refresh(source_proof:Path,target:Path,threads:int=16):
    if os.name!='nt':raise RuntimeError('Windows file-version cache is Windows-only')
    start=time.perf_counter();proof=json.loads(source_proof.read_text(encoding='utf8'));refs=proof['verified_files']
    previous={}
    if target.exists():
        old=json.loads(target.read_text(encoding='utf8'))
        if old.get('schema')=='rinkfit.verified_input_digest_cache.v1':previous={r['file']:r for r in old['entries']}
    def one(ref):
        path=Path(ref['file']);before=fingerprint(path);old=previous.get(ref['file'])
        if old and old['fingerprint']==before and old['sha256']==ref['sha256']:return old,0,True
        h=hashlib.sha256();size=0
        with path.open('rb')as stream:
            while data:=stream.read(1024*1024):h.update(data);size+=len(data)
        if fingerprint(path)!=before:raise RuntimeError('Input changed during digest verification: '+str(path))
        if h.hexdigest()!=ref['sha256']:raise RuntimeError('Input digest differs from source proof: '+str(path))
        return dict(file=ref['file'],sha256=ref['sha256'],fingerprint=before),size,False
    with ThreadPoolExecutor(max_workers=max(1,threads))as pool:results=list(pool.map(one,refs))
    result=dict(schema='rinkfit.verified_input_digest_cache.v1',entries=[r[0]for r in results],bytes_hashed=sum(r[1]for r in results),unchanged_file_reuses=sum(r[2]for r in results),seconds=time.perf_counter()-start,source_proof=str(source_proof),quality_accepted=False)
    target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_name(target.name+'.tmp.'+str(os.getpid()));temporary.write_text(json.dumps(result,separators=(',',':')),encoding='utf8');os.replace(temporary,target)
    return {k:v for k,v in result.items()if k!='entries'}
