"""Explicit CUDA driver/NVRTC execution with typed resident allocations.

No retired kernels/binaries are loaded. CPU and GPU buffer identity, shape,
dtype and generation remain visible; unavailable CUDA raises a real error.
"""
from __future__ import annotations
from dataclasses import dataclass
import ctypes as C
from pathlib import Path
import hashlib
import json
import os
import threading
import time
import numpy as np

class CudaError(RuntimeError): pass

@dataclass
class DeviceBuffer:
    device: object
    pointer: int
    shape: tuple[int,...]
    dtype: np.dtype
    generation: int
    role: str
    byte_count: int
    closed: bool=False
    def close(self):
        if not self.closed:
            self.device._set_current()
            self.device._check(self.device.driver.cuMemFree_v2(self.pointer),'cuMemFree')
            self.closed=True
    def require(self,shape,dtype,generation=None):
        if self.closed or tuple(shape)!=self.shape or np.dtype(dtype)!=self.dtype or generation is not None and generation!=self.generation:
            raise ValueError('GPU buffer dtype/shape/generation/closed mismatch')
        return self

class CudaDevice:
    def __init__(self,*,device_index=0,cuda_root='D:/Program Files/CUDA',cache_root=None):
        if os.name!='nt': raise CudaError('This configured CUDA runtime is Windows-only')
        self.cuda_root=Path(cuda_root)
        self.cache_root=Path(cache_root or 'D:/CodexData/RinkSpace_Data/cache/solve2_cuda')
        self.cache_root.mkdir(parents=True,exist_ok=True)
        self._dll_path=os.add_dll_directory(str(self.cuda_root/'bin'/'x64'))
        self.driver=C.WinDLL('nvcuda.dll')
        I=C.c_int;U=C.c_uint;P=C.c_void_p;Q=C.c_uint64;Z=C.c_size_t
        declarations={
            'cuInit':([U],I),'cuDeviceGet':([C.POINTER(I),I],I),
            'cuDeviceGetName':([C.c_char_p,I,I],I),'cuDeviceGetAttribute':([C.POINTER(I),I,I],I),
            'cuDevicePrimaryCtxRetain':([C.POINTER(P),I],I),
            'cuDevicePrimaryCtxRelease_v2':([I],I),'cuCtxSetCurrent':([P],I),
            'cuMemAlloc_v2':([C.POINTER(Q),Z],I),'cuMemFree_v2':([Q],I),
            'cuMemcpyHtoD_v2':([Q,P,Z],I),'cuMemcpyDtoH_v2':([P,Q,Z],I),
            'cuModuleLoadData':([C.POINTER(P),P],I),'cuModuleGetFunction':([C.POINTER(P),P,C.c_char_p],I),
            'cuModuleUnload':([P],I),
            'cuLaunchKernel':([P,U,U,U,U,U,U,U,P,C.POINTER(P),C.POINTER(P)],I),
            'cuCtxSynchronize':([],I),'cuGetErrorName':([I,C.POINTER(C.c_char_p)],I),
            'cuEventCreate':([C.POINTER(P),U],I),'cuEventRecord':([P,P],I),
            'cuEventSynchronize':([P],I),'cuEventElapsedTime':([C.POINTER(C.c_float),P,P],I),
            'cuEventDestroy_v2':([P],I)}
        for name,(args,ret) in declarations.items():
            fn=getattr(self.driver,name);fn.argtypes=args;fn.restype=ret
        self._check(self.driver.cuInit(0),'cuInit')
        dev=I();self._check(self.driver.cuDeviceGet(C.byref(dev),device_index),'cuDeviceGet');self.index=dev.value
        self.context=P();self._check(self.driver.cuDevicePrimaryCtxRetain(C.byref(self.context),self.index),'cuDevicePrimaryCtxRetain')
        self._set_current()
        name=C.create_string_buffer(256);self._check(self.driver.cuDeviceGetName(name,256,self.index),'cuDeviceGetName');self.name=name.value.decode()
        major,minor=I(),I()
        self._check(self.driver.cuDeviceGetAttribute(C.byref(major),75,self.index),'compute major')
        self._check(self.driver.cuDeviceGetAttribute(C.byref(minor),76,self.index),'compute minor')
        self.arch=f'compute_{major.value}{minor.value}'
        candidates=sorted((self.cuda_root/'bin'/'x64').glob('nvrtc64_*.dll'))
        if not candidates:raise CudaError('NVRTC library missing')
        self.nvrtc=C.CDLL(str(candidates[-1]))
        for name,args in {
            'nvrtcCreateProgram':[C.POINTER(P),C.c_char_p,C.c_char_p,I,C.POINTER(C.c_char_p),C.POINTER(C.c_char_p)],
            'nvrtcCompileProgram':[P,I,C.POINTER(C.c_char_p)],'nvrtcGetPTXSize':[P,C.POINTER(Z)],
            'nvrtcGetPTX':[P,C.c_char_p],'nvrtcGetProgramLogSize':[P,C.POINTER(Z)],
            'nvrtcGetProgramLog':[P,C.c_char_p],'nvrtcDestroyProgram':[C.POINTER(P)]}.items():
            fn=getattr(self.nvrtc,name);fn.argtypes=args;fn.restype=I
        self.buffers=[];self.modules=[];self._lock=threading.RLock();self.closed=False
    def _set_current(self):self._check(self.driver.cuCtxSetCurrent(self.context),'cuCtxSetCurrent')
    def _check(self,code,operation):
        if code:
            name=C.c_char_p();self.driver.cuGetErrorName(code,C.byref(name))
            raise CudaError(f'{operation}: CUDA {code} {name.value.decode() if name.value else "unknown"}')
    def allocate(self,shape,dtype,*,generation,role='working'):
        shape=tuple(int(x) for x in shape);dtype=np.dtype(dtype)
        if not shape or min(shape)<=0 or dtype.hasobject or generation<0:raise ValueError('invalid GPU allocation contract')
        self._set_current();pointer=C.c_uint64();size=int(np.prod(shape))*dtype.itemsize
        self._check(self.driver.cuMemAlloc_v2(C.byref(pointer),size),'cuMemAlloc')
        b=DeviceBuffer(self,pointer.value,shape,dtype,generation,role,size);self.buffers.append(b);return b
    def upload(self,array,*,generation,role='immutable_source'):
        a=np.asarray(array)
        if not a.flags.c_contiguous:raise ValueError('GPU source must have explicit C-contiguous layout')
        b=self.allocate(a.shape,a.dtype,generation=generation,role=role)
        self._copy(b,a);return b
    def _copy(self,b,a):
        b.require(a.shape,a.dtype);self._set_current()
        self._check(self.driver.cuMemcpyHtoD_v2(b.pointer,C.c_void_p(a.ctypes.data),a.nbytes),'cuMemcpyHtoD')
    def update(self,b,array,*,generation):
        if b.role not in ('parameter','working'):raise ValueError('immutable source GPU allocation cannot be overwritten')
        a=np.asarray(array);b.require(a.shape,a.dtype,generation)
        if not a.flags.c_contiguous:raise ValueError('explicit contiguous update required')
        self._copy(b,a)
    def download(self,b):
        b.require(b.shape,b.dtype);a=np.empty(b.shape,dtype=b.dtype);self._set_current()
        self._check(self.driver.cuMemcpyDtoH_v2(C.c_void_p(a.ctypes.data),b.pointer,a.nbytes),'cuMemcpyDtoH')
        return a
    def compile(self,source_path,kernel_names):
        source_path=Path(source_path);source=source_path.read_bytes()
        options=[f'--gpu-architecture={self.arch}','--std=c++17','--fmad=false']
        key=hashlib.sha256(source+b'\0'+json.dumps(options).encode()).hexdigest()
        ptx_path=self.cache_root/(key+'.ptx');log_path=self.cache_root/(key+'.log')
        if ptx_path.exists():ptx=ptx_path.read_bytes()
        else:
            program=C.c_void_p();code=self.nvrtc.nvrtcCreateProgram(C.byref(program),source,source_path.name.encode(),0,None,None)
            if code:raise CudaError(f'nvrtcCreateProgram {code}')
            try:
                opts=(C.c_char_p*len(options))(*(o.encode() for o in options))
                code=self.nvrtc.nvrtcCompileProgram(program,len(options),opts)
                size=C.c_size_t();self.nvrtc.nvrtcGetProgramLogSize(program,C.byref(size));log=C.create_string_buffer(max(1,size.value))
                self.nvrtc.nvrtcGetProgramLog(program,log);log_path.write_bytes(log.value)
                if code:raise CudaError(f'NVRTC {code}; {log_path}: '+log.value.decode(errors='replace'))
                self.nvrtc.nvrtcGetPTXSize(program,C.byref(size));out=C.create_string_buffer(size.value)
                result=self.nvrtc.nvrtcGetPTX(program,out)
                if result:raise CudaError(f'nvrtcGetPTX {result}')
                ptx=out.raw;ptx_path.write_bytes(ptx)
            finally:self.nvrtc.nvrtcDestroyProgram(C.byref(program))
        self._set_current();module=C.c_void_p();payload=C.create_string_buffer(ptx)
        self._check(self.driver.cuModuleLoadData(C.byref(module),C.cast(payload,C.c_void_p)),'cuModuleLoadData')
        self.modules.append(module);functions={}
        for name in kernel_names:
            fn=C.c_void_p();self._check(self.driver.cuModuleGetFunction(C.byref(fn),module,name.encode()),'cuModuleGetFunction');functions[name]=fn
        return functions,{'source_path':str(source_path.resolve()),'source_sha256':hashlib.sha256(source).hexdigest(),'ptx_path':str(ptx_path),'architecture':self.arch,'options':options}
    def launch(self,kernel,count,arguments,*,threads=128):
        # Scalars must be explicitly typed ctypes, device pointers must be owned buffers.
        values=[C.c_uint64(v.pointer) if isinstance(v,DeviceBuffer) else v for v in arguments]
        if any(isinstance(v,DeviceBuffer) and (v.closed or v.device is not self) for v in arguments):raise ValueError('foreign or closed CUDA buffer')
        refs=(C.c_void_p*len(values))(*(C.cast(C.byref(v),C.c_void_p) for v in values))
        self._set_current();self._check(self.driver.cuLaunchKernel(kernel,(count+threads-1)//threads,1,1,threads,1,1,0,None,refs,None),'cuLaunchKernel')
    def timed(self,callback):
        self._set_current();start=C.c_void_p();end=C.c_void_p()
        self._check(self.driver.cuEventCreate(C.byref(start),0),'cuEventCreate')
        self._check(self.driver.cuEventCreate(C.byref(end),0),'cuEventCreate')
        try:
            self._check(self.driver.cuEventRecord(start,None),'cuEventRecord');callback()
            self._check(self.driver.cuEventRecord(end,None),'cuEventRecord');self._check(self.driver.cuEventSynchronize(end),'cuEventSynchronize')
            ms=C.c_float();self._check(self.driver.cuEventElapsedTime(C.byref(ms),start,end),'cuEventElapsedTime');return float(ms.value)
        finally:
            self.driver.cuEventDestroy_v2(start);self.driver.cuEventDestroy_v2(end)
    def synchronize(self):self._set_current();self._check(self.driver.cuCtxSynchronize(),'cuCtxSynchronize')
    def close(self):
        if self.closed:return
        self._set_current()
        for b in reversed(self.buffers):b.close()
        for module in reversed(self.modules):self._check(self.driver.cuModuleUnload(module),'cuModuleUnload')
        self._check(self.driver.cuDevicePrimaryCtxRelease_v2(self.index),'cuDevicePrimaryCtxRelease')
        self.closed=True;self._dll_path.close()
    def __enter__(self):return self
    def __exit__(self,*unused):self.close()
