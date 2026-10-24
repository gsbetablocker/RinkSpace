"""Native exact-W assembly for current sparse block residual/Jacobian batches.

The numerical factor support comes from PreparedS01. Runtime performs one C ABI
call, with OpenMP parallelism over complete source factors. It never builds a
dense N by parameter-count matrix or runs one Python callback per factor.
"""
from __future__ import annotations
from pathlib import Path
import ctypes as C
import json,os,time
import numpy as np

I32P=C.POINTER(C.c_int32);I64P=C.POINTER(C.c_int64);DP=C.POINTER(C.c_double)


class _Input(C.Structure):
    _fields_=[(k,C.c_int32) for k in ("abi_version","nblocks","nraw","nstat")]+[(k,C.c_int64) for k in ("raw_edges","stat_edges","nnz","raw_jac_values","stat_jac_values")]+[
        ("block_sizes",I32P),("raw_edge_offsets",I64P),("raw_edge_blocks",I32P),("raw_jac_offsets",I64P),
        ("stat_edge_offsets",I64P),("stat_edge_blocks",I32P),("stat_jac_offsets",I64P),("stat_residual_offsets",I64P),
        ("raw_stat_owner",I32P),("csr_indptr",I64P),("csr_indices",I64P),("csr_values",DP)]


def _ptr(a,t):
    return a.ctypes.data_as(t) if a is not None else t()


def _typed(a,dtype):
    return np.ascontiguousarray(a,dtype=dtype)


def _buffer(a,size,name,writable=False):
    if not isinstance(a,np.ndarray) or a.dtype!=np.dtype(np.float64) or not a.flags.c_contiguous or a.size!=size:
        raise ValueError(name+" requires a contiguous float64 buffer of declared native layout size")
    if writable and not a.flags.writeable:
        raise ValueError(name+" is readonly")
    return a.reshape(-1)


class WhiteningJacobianPlan:
    """One prepared current source factor graph and precomputed edge routing.

    raw_observation_rows explicitly maps raw factor order to PreparedS01 rows.
    Extra new intersection factors use -1; they have no original statistical W.
    Each original statistical source has exactly one factor owner, allowing a
    compact raw-edge -> statistical matrix base map instead of Wnnz*edge terms.
    Parameter block identity/order/width is shared across both native layouts.
    """
    def __init__(self,prepared,raw_layout,stat_layout,raw_observation_rows,*,manifest):
        started=time.perf_counter()
        if raw_layout.block_ids!=stat_layout.block_ids or raw_layout.block_sizes!=stat_layout.block_sizes:
            raise ValueError("raw/statistical parameter stable IDs, order or widths differ")
        if any(n!=2 for n in raw_layout.residual_sizes):
            raise ValueError("each raw native image factor requires exactly u,v")
        factors=prepared.manifest["statistical_factors"]
        if tuple(f["id"] for f in factors)!=stat_layout.factor_ids or tuple(f["residual_count"] for f in factors)!=stat_layout.residual_sizes:
            raise ValueError("statistical layout must retain the exact current full-W factor order/dimensions")
        rows=np.asarray(raw_observation_rows)
        if rows.ndim!=1 or rows.dtype.kind not in "iu" or len(rows)!=len(raw_layout.factor_ids) or np.any(rows< -1) or np.any(rows>=prepared.row_count):
            raise ValueError("explicit raw-factor/current-observation mapping invalid")
        rows=rows.astype(np.int64,copy=False);valid=rows>=0
        if len(np.unique(rows[valid]))!=np.count_nonzero(valid):
            raise ValueError("original source observation repeated in raw native factor order")
        lookup=np.full(prepared.row_count,-1,dtype=np.int64);lookup[rows[valid]]=np.flatnonzero(valid)
        owner=np.full(len(rows),-1,dtype=np.int32)
        for index,factor in enumerate(factors):
            raw=lookup[np.asarray(factor["rows"],dtype=np.int64)]
            if np.any(raw<0):
                raise ValueError("raw batch omits a complete original W/independent source member")
            if np.any(owner[raw]>=0):
                raise ValueError("source statistical support counted more than once")
            owner[raw]=index
        w=prepared.whitening
        source=lookup[w.indices//2]
        if np.any(source<0):
            raise ValueError("current full W refers to an absent raw native source")
        mapped=2*source+w.indices%2
        record=json.loads(Path(manifest).read_text(encoding="utf-8-sig"))
        if record.get("schema")!="rinksolve.whitening_build.v1" or record.get("status")!="built" or record.get("uses_legacy_solver") is not False:
            raise ValueError("an explicitly built new factor-assembly library is required")
        self._dll_dirs=[os.add_dll_directory(p) for p in record.get("dll_directories",[]) if hasattr(os,"add_dll_directory")]
        self.library=C.CDLL(record["binary"])
        lib=self.library
        lib.wh_abi_version.argtypes=[];lib.wh_abi_version.restype=C.c_int32
        if lib.wh_abi_version()!=1:
            raise ValueError("new whitening native ABI differs")
        lib.wh_create.argtypes=[C.POINTER(_Input),C.c_char_p,C.c_int64];lib.wh_create.restype=C.c_void_p
        lib.wh_destroy.argtypes=[C.c_void_p];lib.wh_destroy.restype=None
        lib.wh_apply.argtypes=[C.c_void_p,DP,DP,DP,DP,C.c_int32,C.c_char_p,C.c_int64];lib.wh_apply.restype=C.c_int32
        arrays={"block_sizes":_typed(raw_layout.block_sizes,np.int32),"raw_edge_offsets":_typed(raw_layout.factor_edge_offsets,np.int64),
            "raw_edge_blocks":_typed(raw_layout.edge_blocks,np.int32),"raw_jac_offsets":_typed(raw_layout.jacobian_offsets,np.int64),
            "stat_edge_offsets":_typed(stat_layout.factor_edge_offsets,np.int64),"stat_edge_blocks":_typed(stat_layout.edge_blocks,np.int32),
            "stat_jac_offsets":_typed(stat_layout.jacobian_offsets,np.int64),"stat_residual_offsets":_typed(stat_layout.residual_offsets,np.int64),
            "raw_stat_owner":owner,"csr_indptr":_typed(w.indptr,np.int64),"csr_indices":_typed(mapped,np.int64),"csr_values":_typed(w.data,np.float64)}
        inp=_Input(1,len(raw_layout.block_ids),len(raw_layout.factor_ids),len(stat_layout.factor_ids),len(raw_layout.edge_blocks),len(stat_layout.edge_blocks),
            w.nnz,raw_layout.jacobian_value_count,stat_layout.jacobian_value_count,
            *[_ptr(arrays[name],dict(_Input._fields_)[name]) for name in arrays])
        error=C.create_string_buffer(4096)
        self.handle=lib.wh_create(C.byref(inp),error,len(error))
        if not self.handle:
            raise ValueError("new whitening plan invalid: "+error.value.decode("utf-8",errors="replace"))
        self.prepared=prepared;self.raw_layout=raw_layout;self.stat_layout=stat_layout
        self.raw_observation_rows=rows.copy();self.raw_observation_rows.setflags(write=False)
        self.raw_statistical_owner=owner;owner.setflags(write=False)
        self.description={"abi_version":1,"raw_factors":len(rows),"original_statistical_factors":len(factors),"W_nonzero_values":w.nnz,
            "raw_jacobian_values":raw_layout.jacobian_value_count,"statistical_jacobian_values":stat_layout.jacobian_value_count,
            "parallel_owner":"one complete statistical factor per OpenMP iteration","runtime_python_factor_loops":0,
            "dense_global_jacobian":False,"preparation_seconds":time.perf_counter()-started,"binary":record["binary"]}

    def apply(self,raw_residual,raw_jacobian=None,residual_out=None,jacobian_out=None,*,num_threads=None):
        if not self.handle:
            raise RuntimeError("closed whitening plan")
        threads=max(1,os.cpu_count() or 1) if num_threads is None else num_threads
        if type(threads) is not int or threads<=0:
            raise ValueError("positive integer CPU thread count required")
        r=_buffer(raw_residual,self.raw_layout.residual_count,"raw residual")
        out=np.empty(self.stat_layout.residual_count,dtype=np.float64) if residual_out is None else _buffer(residual_out,self.stat_layout.residual_count,"statistical residual output",True)
        if raw_jacobian is None:
            if jacobian_out is not None:
                raise ValueError("Jacobian output needs a raw Jacobian")
            j=oj=None
        else:
            j=_buffer(raw_jacobian,self.raw_layout.jacobian_value_count,"raw block Jacobian")
            oj=np.empty(self.stat_layout.jacobian_value_count,dtype=np.float64) if jacobian_out is None else _buffer(jacobian_out,self.stat_layout.jacobian_value_count,"statistical Jacobian output",True)
        if np.shares_memory(r,out) or (oj is not None and (np.shares_memory(j,oj) or np.shares_memory(out,oj) or np.shares_memory(r,oj) or np.shares_memory(j,out))):
            raise ValueError("raw/statistical whitening buffers must not alias")
        error=C.create_string_buffer(4096)
        code=self.library.wh_apply(self.handle,_ptr(r,DP),_ptr(j,DP),_ptr(out,DP),_ptr(oj,DP),threads,error,len(error))
        if code:
            raise ValueError("current source whitening evaluation rejected: "+error.value.decode("utf-8",errors="replace"))
        return out,oj

    def close(self):
        handle=getattr(self,"handle",None)
        if handle:
            self.library.wh_destroy(handle);self.handle=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def __del__(self):self.close()
