"""Driver-only immutable PTX runtime for the same exact native image filters.

No source measurement, noise draw, profile window, or estimator is replaced.
NVRTC is reused from the already installed GPU runtime; no install is performed.
"""
from pathlib import Path
import ctypes as C,json,os,hashlib,time
import numpy as np
from scipy.ndimage._filters import _gaussian_kernel1d

class PTXSourceFilterCUDA:
    def __init__(self,height,width,*,batch=1,kernel_source=None,build_output=None,ptx_file=None,
                 runtime_bin="D:/OneDrive/codex_projs/MotionTrail_Codex_Starter_v1/.venv-gpu/Lib/site-packages/torch/lib"):
        self.h,self.w,self.batch=int(height),int(width),int(batch)
        if min(self.h,self.w)<2 or self.batch<1:raise ValueError("invalid CUDA dimensions")
        self._dll_dir=os.add_dll_directory(runtime_bin)
        self.driver=C.WinDLL("nvcuda.dll")
        self._ctx=C.c_void_p();self._module=C.c_void_p();self.memory=[]
        self._bind()
        self.check(self.driver.cuInit(0))
        dev=C.c_int();self.check(self.driver.cuDeviceGet(C.byref(dev),0))
        self.check(self.driver.cuCtxCreate_v2(C.byref(self._ctx),4,dev))
        if ptx_file is None:raise ValueError('this lightweight runtime only loads an existing immutable PTX')
        ptx=Path(ptx_file).read_bytes()
        self._ptx=C.create_string_buffer(ptx)
        self.check(self.driver.cuModuleLoadData(C.byref(self._module),self._ptx))
        self.functions={}
        for name in ("gauss","convert","spline"):
            f=C.c_void_p();self.check(self.driver.cuModuleGetFunction(C.byref(f),self._module,name.encode()));self.functions[name]=f
        for dtype in (np.float32,np.float32,np.float32,np.float64):
            p=C.c_uint64();self.check(self.driver.cuMemAlloc_v2(C.byref(p),self.h*self.w*self.batch*np.dtype(dtype).itemsize));self.memory.append(p)
        self.kernel=np.asarray(_gaussian_kernel1d(.8,0,3),np.float64)
    def _bind(self):
        d=self.driver
        specs={"cuInit":([C.c_uint],C.c_int),"cuDeviceGet":([C.POINTER(C.c_int),C.c_int],C.c_int),
          "cuCtxCreate_v2":([C.POINTER(C.c_void_p),C.c_uint,C.c_int],C.c_int),"cuCtxDestroy_v2":([C.c_void_p],C.c_int),
          "cuModuleLoadData":([C.POINTER(C.c_void_p),C.c_void_p],C.c_int),"cuModuleGetFunction":([C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p],C.c_int),
          "cuModuleUnload":([C.c_void_p],C.c_int),"cuMemAlloc_v2":([C.POINTER(C.c_uint64),C.c_size_t],C.c_int),
          "cuMemFree_v2":([C.c_uint64],C.c_int),"cuMemcpyHtoD_v2":([C.c_uint64,C.c_void_p,C.c_size_t],C.c_int),
          "cuMemcpyDtoH_v2":([C.c_void_p,C.c_uint64,C.c_size_t],C.c_int),
          "cuLaunchKernel":([C.c_void_p]+[C.c_uint]*7+[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)],C.c_int)}
        for name,(args,res) in specs.items():f=getattr(d,name);f.argtypes=args;f.restype=res
    @staticmethod
    def check(rc):
        if rc:raise RuntimeError("CUDA Driver error "+str(rc))
    @staticmethod
    def ncheck(rc):
        if rc:raise RuntimeError("NVRTC API error "+str(rc))
    def launch(self,name,grid,block,args):
        values=(C.c_void_p*len(args))(*(C.addressof(a)for a in args))
        self.check(self.driver.cuLaunchKernel(self.functions[name],grid,1,1,block,1,1,0,None,values,None))
    def __call__(self,channel,*,include_gaussian=True):
        a=np.ascontiguousarray(channel,dtype=np.float32);single=a.ndim==2
        if single:a=a[None]
        if a.ndim!=3 or a.shape[1:]!=(self.h,self.w) or not 1<=len(a)<=self.batch:raise ValueError("CUDA source filter shape mismatch")
        b=len(a);count=a.size;src,tmp,g,c=self.memory;I=C.c_int;D=C.c_double;U=C.c_uint64
        self.check(self.driver.cuMemcpyHtoD_v2(src,a.ctypes.data,a.nbytes))
        scalars=[D(self.kernel[i])for i in (3,4,5,6)]
        for axis,p,q in ((0,src,tmp),(1,tmp,g)):
            self.launch("gauss",(count+255)//256,256,[U(p.value),U(q.value),I(self.h),I(self.w),I(count),I(axis),*scalars])
        self.launch("convert",(count+255)//256,256,[U(g.value),U(c.value),I(count)])
        z=-0.267949192431122706472553658494127633;gain=(1-z)*(1-1/z)
        for axis,lines in ((0,b*self.w),(1,b*self.h)):
            self.launch("spline",(lines+127)//128,128,[U(c.value),I(self.h),I(self.w),I(b),I(axis),D(z),D(gain)])
        gaussian=np.empty(a.shape,np.float32)if include_gaussian else None;coeff=np.empty(a.shape,np.float64)
        if include_gaussian:self.check(self.driver.cuMemcpyDtoH_v2(gaussian.ctypes.data,g,gaussian.nbytes))
        self.check(self.driver.cuMemcpyDtoH_v2(coeff.ctypes.data,c,coeff.nbytes))
        if not include_gaussian:return coeff[0]if single else coeff
        return (gaussian[0],coeff[0])if single else (gaussian,coeff)
    def close(self):
        for p in self.memory:self.check(self.driver.cuMemFree_v2(p))
        self.memory=[]
        if self._module:self.check(self.driver.cuModuleUnload(self._module));self._module=C.c_void_p()
        if self._ctx:self.check(self.driver.cuCtxDestroy_v2(self._ctx));self._ctx=C.c_void_p()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
