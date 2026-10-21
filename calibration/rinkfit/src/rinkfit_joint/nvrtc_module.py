"""Small explicit NVRTC/CUDA Driver resident module, reused by joint kernels.

Compiles once and keeps GPU buffers resident. No arrays or iterations are dumped.
Numerical compilation defaults preserve FP64 division/sqrt and disable FMA.
"""
from pathlib import Path
import ctypes as C
import hashlib,json,os,time
import numpy as np

class NVRTCModule:
    def __init__(self,source=None,*,ptx=None,name="kernel.cu",output=None,options=(),
                 runtime_bin=None):
        if (source is None)==(ptx is None):raise ValueError("supply exactly one of source or PTX")
        if runtime_bin is None:
            # Reuse the production source-reader runtime resolver: the GPU
            # environment was moved out of OneDrive and must not be rebuilt.
            from .feedback_dispatch import source_reader_python
            runtime_bin=source_reader_python().parents[1]/'Lib/site-packages/torch/lib'
        runtime_bin=Path(runtime_bin)
        if not (runtime_bin/'nvrtc64_120_0.dll').is_file() or not (runtime_bin/'nvrtc-builtins64_128.dll').is_file():
            raise FileNotFoundError('Existing CUDA NVRTC runtime DLLs missing: '+str(runtime_bin))
        self._dll_dir=os.add_dll_directory(str(runtime_bin))
        self._builtins=C.WinDLL(str(Path(runtime_bin)/"nvrtc-builtins64_128.dll"))
        self.nvrtc=C.WinDLL(str(Path(runtime_bin)/"nvrtc64_120_0.dll"))
        self.driver=C.WinDLL("nvcuda.dll");self._ctx=C.c_void_p();self._module=C.c_void_p()
        self.allocations={};self.functions={};self._bind()
        self._check(self.driver.cuInit(0));device=C.c_int()
        self._check(self.driver.cuDeviceGet(C.byref(device),0))
        self._check(self.driver.cuCtxCreate_v2(C.byref(self._ctx),4,device))
        if source is not None:
            code=source.encode("utf8")if isinstance(source,str)else bytes(source)
            ptx=self._compile(code,name,options,output)
        elif isinstance(ptx,(str,Path)):ptx=Path(ptx).read_bytes()
        self._ptx=C.create_string_buffer(bytes(ptx));self.ptx_sha256=hashlib.sha256(bytes(ptx)).hexdigest()
        self._check(self.driver.cuModuleLoadData(C.byref(self._module),self._ptx))
    def _bind(self):
        d=self.driver
        signatures={
          "cuInit":[C.c_uint],"cuDeviceGet":[C.POINTER(C.c_int),C.c_int],
          "cuCtxCreate_v2":[C.POINTER(C.c_void_p),C.c_uint,C.c_int],
          "cuCtxSetCurrent":[C.c_void_p],"cuCtxSynchronize":[],
          "cuCtxDestroy_v2":[C.c_void_p],"cuModuleLoadData":[C.POINTER(C.c_void_p),C.c_void_p],
          "cuModuleGetFunction":[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p],
          "cuModuleUnload":[C.c_void_p],"cuMemAlloc_v2":[C.POINTER(C.c_uint64),C.c_size_t],
          "cuMemFree_v2":[C.c_uint64],"cuMemcpyHtoD_v2":[C.c_uint64,C.c_void_p,C.c_size_t],
          "cuMemcpyDtoH_v2":[C.c_void_p,C.c_uint64,C.c_size_t],
          "cuLaunchKernel":[C.c_void_p]+[C.c_uint]*7+[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]}
        for name,args in signatures.items():f=getattr(d,name);f.argtypes=args;f.restype=C.c_int
        n=self.nvrtc
        n.nvrtcCreateProgram.argtypes=[C.POINTER(C.c_void_p),C.c_char_p,C.c_char_p,C.c_int,C.c_void_p,C.c_void_p]
        n.nvrtcCompileProgram.argtypes=[C.c_void_p,C.c_int,C.POINTER(C.c_char_p)]
        for name in ("nvrtcGetProgramLogSize","nvrtcGetPTXSize"):getattr(n,name).argtypes=[C.c_void_p,C.POINTER(C.c_size_t)]
        for name in ("nvrtcGetProgramLog","nvrtcGetPTX"):getattr(n,name).argtypes=[C.c_void_p,C.c_void_p]
        n.nvrtcDestroyProgram.argtypes=[C.POINTER(C.c_void_p)]
    @staticmethod
    def _check(rc):
        if rc:raise RuntimeError("CUDA Driver error "+str(rc))
    @staticmethod
    def _ncheck(rc):
        if rc:raise RuntimeError("NVRTC API error "+str(rc))
    def _compile(self,code,name,options,output):
        program=C.c_void_p();self._ncheck(self.nvrtc.nvrtcCreateProgram(C.byref(program),code,name.encode(),0,None,None))
        flags=["--gpu-architecture=compute_120","--fmad=false","--prec-div=true","--prec-sqrt=true","--std=c++17",*options]
        encoded=[x.encode()for x in flags];t=time.perf_counter()
        rc=self.nvrtc.nvrtcCompileProgram(program,len(encoded),(C.c_char_p*len(encoded))(*encoded))
        size=C.c_size_t();self._ncheck(self.nvrtc.nvrtcGetProgramLogSize(program,C.byref(size)))
        log=C.create_string_buffer(size.value);self._ncheck(self.nvrtc.nvrtcGetProgramLog(program,log))
        data=None
        if rc==0:
            self._ncheck(self.nvrtc.nvrtcGetPTXSize(program,C.byref(size)));buf=C.create_string_buffer(size.value)
            self._ncheck(self.nvrtc.nvrtcGetPTX(program,buf));data=buf.raw
        self._ncheck(self.nvrtc.nvrtcDestroyProgram(C.byref(program)))
        if output:
            out=Path(output);out.mkdir(parents=True,exist_ok=True)
            if(out/"recipe.json").exists():raise ValueError("new immutable compile output required")
            (out/"kernel.cu").write_bytes(code);(out/"compile.log").write_bytes(log.value)
            recipe=dict(source_sha256=hashlib.sha256(code).hexdigest(),options=flags,returncode=rc,compile_seconds=time.perf_counter()-t,blocking_sync_context=True)
            if data is not None:
                (out/"kernel.ptx").write_bytes(data);recipe["ptx_sha256"]=hashlib.sha256(data).hexdigest()
            (out/"recipe.json").write_text(json.dumps(recipe,indent=2),encoding="utf8")
        if rc:raise RuntimeError("NVRTC compile error "+str(rc)+": "+log.value.decode(errors="replace"))
        return data
    def set_current(self):
        if not self._ctx:raise RuntimeError("CUDA context is closed")
        self._check(self.driver.cuCtxSetCurrent(self._ctx))
    def get_function(self,name):
        self.set_current()
        if name not in self.functions:
            f=C.c_void_p();self._check(self.driver.cuModuleGetFunction(C.byref(f),self._module,name.encode()));self.functions[name]=f
        return self.functions[name]
    def alloc(self,nbytes):
        self.set_current();p=C.c_uint64();self._check(self.driver.cuMemAlloc_v2(C.byref(p),int(nbytes)));self.allocations[p.value]=int(nbytes);return p.value
    def _bounds(self,ptr,nbytes,offset):
        if int(ptr)not in self.allocations or offset<0 or offset+int(nbytes)>self.allocations[int(ptr)]:raise ValueError("GPU allocation bounds exceeded")
    def upload(self,array,ptr=None,*,offset_bytes=0):
        a=np.ascontiguousarray(array)
        if ptr is None:ptr=self.alloc(a.nbytes+offset_bytes)
        self._bounds(ptr,a.nbytes,offset_bytes);self.set_current()
        self._check(self.driver.cuMemcpyHtoD_v2(int(ptr)+offset_bytes,a.ctypes.data,a.nbytes));return int(ptr)
    def download(self,ptr,shape,dtype=np.float64,*,offset_bytes=0):
        a=np.empty(shape,dtype=dtype);self._bounds(ptr,a.nbytes,offset_bytes);self.set_current()
        self._check(self.driver.cuMemcpyDtoH_v2(a.ctypes.data,int(ptr)+offset_bytes,a.nbytes));return a
    def launch(self,name,grid,block,args,*,shared_bytes=0,stream=None,synchronize=False):
        self.set_current()
        def xyz(value):
            v=(value,)if isinstance(value,int)else tuple(value)
            if not 1<=len(v)<=3:raise ValueError("kernel launch dimensions must have one to three axes")
            return v+(1,)*(3-len(v))
        g=xyz(grid);b=xyz(block)
        values=(C.c_void_p*len(args))(*(C.addressof(a)for a in args))
        self._check(self.driver.cuLaunchKernel(self.get_function(name),*g,*b,int(shared_bytes),stream,values,None))
        if synchronize:self.synchronize()
    def synchronize(self):
        self.set_current();self._check(self.driver.cuCtxSynchronize())
    def free(self,ptr):
        self.set_current()
        if int(ptr)not in self.allocations:raise ValueError("unknown GPU allocation")
        self._check(self.driver.cuMemFree_v2(int(ptr)));del self.allocations[int(ptr)]
    def close(self):
        if not self._ctx:return
        self.set_current()
        for ptr in list(self.allocations):self.free(ptr)
        if self._module:self._check(self.driver.cuModuleUnload(self._module));self._module=C.c_void_p()
        self._check(self.driver.cuCtxDestroy_v2(self._ctx));self._ctx=C.c_void_p()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
