"""RAM-only exact joint CPU oracle used to shadow the FP64 GPU raw-J kernel."""
from __future__ import annotations
import ctypes as C
import os
from pathlib import Path
import numpy as np

class JointCPUOracle:
    def __init__(self,dll,model,layouts,*,runtime_bin_paths=()):
        self._dll_dirs=[]
        if os.name=='nt':
            for path in runtime_bin_paths:
                self._dll_dirs.append(os.add_dll_directory(str(path)))
        self.dll=C.CDLL(str(Path(dll).resolve()))
        self.dll.joint_oracle_last_error.restype=C.c_char_p
        self.dll.joint_oracle_create.argtypes=[C.c_char_p,C.c_char_p]
        self.dll.joint_oracle_create.restype=C.c_void_p
        self.dll.joint_oracle_destroy.argtypes=[C.c_void_p]
        self.dll.joint_oracle_destroy.restype=None
        self.dll.joint_oracle_eval.argtypes=[C.c_void_p,C.c_uint64,*([C.c_void_p]*9),C.c_int32]
        self.dll.joint_oracle_eval.restype=C.c_int64
        self.handle=self.dll.joint_oracle_create(os.fsencode(model),os.fsencode(layouts))
        if not self.handle:raise RuntimeError(self._error())
    def _error(self):
        return self.dll.joint_oracle_last_error().decode('utf8','replace')
    def evaluate(self,layout_index,frame_time,station,row,*,seed_block=None,seed_component=None,seed_aux=None,threads=1):
        if not self.handle:raise RuntimeError('oracle is closed')
        li=np.ascontiguousarray(layout_index,dtype=np.int32).reshape(-1)
        n=len(li)
        def value(a,dtype,default=None):
            if a is None:a=default
            a=np.asarray(a,dtype=dtype)
            if a.ndim==0:a=np.full(n,a.item(),dtype=dtype)
            a=np.ascontiguousarray(a).reshape(-1)
            if len(a)!=n:raise ValueError('oracle arrays must have the same number of requests')
            return a
        t=value(frame_time,np.float64);s=value(station,np.float64);r=value(row,np.float64)
        sb=value(seed_block,np.int32,-1);sc=value(seed_component,np.int32,-1);aux=value(seed_aux,np.int32,0)
        out=np.empty((n,9),np.float64);status=np.empty((n,4),np.int32)
        pointers=[C.c_void_p(a.ctypes.data) for a in (li,t,s,r,sb,sc,aux,out,status)]
        failed=self.dll.joint_oracle_eval(self.handle,n,*pointers,int(threads))
        if failed<0:raise RuntimeError(self._error())
        return dict(values=out,status=status,failed_rows=int(failed),
                    value_columns=('canonical_u','canonical_v','Jet1_du','Jet1_dv','Jet1_u','Jet1_v','depth','ray_radius_sq','brown_lower'),
                    status_columns=('canonical_arithmetic_valid','canonical_physical_status','Jet1_arithmetic_valid','Jet1_physical_status'))
    def close(self):
        if getattr(self,'handle',None):
            self.dll.joint_oracle_destroy(self.handle);self.handle=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
