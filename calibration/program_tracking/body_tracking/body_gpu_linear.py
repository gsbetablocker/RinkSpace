"""FP64 device-resident CSR products and real cuDSS normal-system solves.

The weighted CSR is assembled by the existing factors. Its values are copied
unchanged to CUDA. Bounded row blocks of J.T form the same FP64 normal matrix
on the device; residual/geometry assembly remains in the existing factors.
"""
from pathlib import Path
import ctypes,os,time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator


class CuDSS:
    def __init__(self):
        root=Path(os.environ.get('RINKSPACE_DATA_ROOT','D:/CodexData/RinkSpace_Data'))/'runtime_tools/cudss/0.8-cu13'
        cuda=Path(os.environ.get('CUDA_PATH','D:/Program Files/CUDA'))
        self.dirs=[os.add_dll_directory(str(p))for p in(root/'bin',cuda/'bin/x64',cuda/'bin')if p.is_dir()]
        self.lib=ctypes.WinDLL(str(root/'bin/cudss64_0.dll'));p=ctypes.c_void_p;i=ctypes.c_int;ll=ctypes.c_int64;s=ctypes.c_size_t
        types=dict(cudssCreate=[ctypes.POINTER(p)],cudssDestroy=[p],cudssConfigCreate=[ctypes.POINTER(p)],cudssConfigDestroy=[p],cudssDataCreate=[p,ctypes.POINTER(p)],cudssDataDestroy=[p,p],cudssSetStream=[p,p],cudssConfigSet=[p,i,p,s],cudssMatrixCreateCsr=[ctypes.POINTER(p),ll,ll,ll,p,p,p,p,i,i,i,i,i,i],cudssMatrixCreateDn=[ctypes.POINTER(p),ll,ll,ll,p,i,i],cudssMatrixDestroy=[p],cudssExecute=[p,i,p,p,p,p,p],cudssDataGet=[p,p,i,p,s,ctypes.POINTER(s)])
        for name,args in types.items():
            fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=i
        self.path=str(root/'bin/cudss64_0.dll')

    def call(self,name,*args):
        status=getattr(self.lib,name)(*args)
        if status:raise RuntimeError(f'{name} cuDSS status {status}')

    def solve(self,A,rhs):
        import torch
        p=ctypes.c_void_p;ptr=lambda t:p(t.data_ptr());handle=p();config=p();data=p();am=p();xm=p();bm=p();created=[]
        x=torch.empty_like(rhs);n=len(rhs);record=dict(backend='cuDSS_FP64_device',library=self.path)
        started=time.perf_counter()
        try:
            self.call('cudssCreate',ctypes.byref(handle));created.append(('cudssDestroy',(handle,)))
            self.call('cudssSetStream',handle,p(torch.cuda.current_stream().cuda_stream))
            self.call('cudssConfigCreate',ctypes.byref(config));created.append(('cudssConfigDestroy',(config,)))
            zero=ctypes.c_int(0)
            for key in(11,15):self.call('cudssConfigSet',config,key,ctypes.byref(zero),ctypes.sizeof(zero))
            self.call('cudssDataCreate',handle,ctypes.byref(data));created.append(('cudssDataDestroy',(handle,data)))
            # CUDA data types: int32=10, FP64=1. SPD/full/base0 matches J.T@J.
            self.call('cudssMatrixCreateCsr',ctypes.byref(am),n,n,A._nnz(),ptr(A.crow_indices()),None,ptr(A.col_indices()),ptr(A.values()),10,10,1,3,0,0);created.append(('cudssMatrixDestroy',(am,)))
            for out,tensor in((xm,x),(bm,rhs)):
                self.call('cudssMatrixCreateDn',ctypes.byref(out),n,1,n,ptr(tensor),1,0);created.append(('cudssMatrixDestroy',(out,)))
            def phase(code,name):
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
                self.call('cudssExecute',handle,code,config,data,am,xm,bm);b.record();b.synchronize();record[name+'_seconds']=a.elapsed_time(b)*.001
            phase(3,'analysis')
            estimates=(ctypes.c_int64*16)();written=ctypes.c_size_t()
            self.call('cudssDataGet',handle,data,13,ctypes.byref(estimates),ctypes.sizeof(estimates),ctypes.byref(written))
            record['memory_estimates_bytes']=list(estimates)[:written.value//8]
            phase(4,'factor');phase(1008,'solve')
            info=ctypes.c_int();self.call('cudssDataGet',handle,data,0,ctypes.byref(info),ctypes.sizeof(info),ctypes.byref(written))
            if info.value:raise ArithmeticError(f'cuDSS numerical info {info.value}')
            record.update(wall_seconds=time.perf_counter()-started,device_allocated_bytes=int(torch.cuda.memory_allocated()),device_reserved_bytes=int(torch.cuda.memory_reserved()))
            return x,record
        finally:
            for name,args in reversed(created):getattr(self.lib,name)(*args)


class GPUJacobian(LinearOperator):
    """A real FP64 CSR operator; no host array impersonates device memory."""
    def __init__(self,matrix,*,shared_prefix=0):
        import torch
        self.torch=torch;self._host=matrix.tocsr();self.prefix=int(shared_prefix);self._gram=None
        if not np.isfinite(matrix.data).all():raise ArithmeticError('Nonfinite actual weighted CSR before GPU upload')
        if self.prefix and matrix.nnz and np.min(matrix.indices)<self.prefix:raise ValueError('Frozen prefix has nonzero Jacobian columns')
        started=time.perf_counter();cols=matrix.shape[1]-self.prefix
        row=torch.from_numpy(matrix.indptr.astype(np.int32,copy=False)).to('cuda')
        index=torch.from_numpy(matrix.indices.astype(np.int32,copy=False)).to('cuda')
        if self.prefix:index-=self.prefix
        values=torch.from_numpy(matrix.data).to('cuda')
        self.device=values.device;self.J=torch.sparse_csr_tensor(row,index,values,size=(matrix.shape[0],cols))
        norm=torch.zeros(cols,dtype=torch.float64,device=self.device)
        for first in range(0,len(values),4_000_000):
            last=min(first+4_000_000,len(values));norm.scatter_add_(0,index[first:last].long(),values[first:last].square())
        self.scale_inv=torch.sqrt(norm).cpu().numpy();torch.cuda.synchronize()
        if not np.isfinite(self.scale_inv).all():raise ArithmeticError('Nonfinite device Jacobian column norm')
        self.transfer=dict(jacobian_H2D_bytes=matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes,jacobian_H2D_seconds=time.perf_counter()-started)
        self._dss=None;self.released=False;super().__init__(dtype=np.dtype('float64'),shape=(matrix.shape[0],cols))

    def release(self):
        """Retire a completed linearization while TRF still holds its wrapper.

        SciPy requests the next Jacobian only after accepting its new fun state.
        Old trust-region operators can remain referenced until the next loop,
        but their device data must not overlap the new FK/J/Gram workspace.
        """
        if self.released:return
        self.J=None;self._gram=None;self._diag=None;self._host=None;self._host_rowptr=None
        self._dss=None;self.released=True
        self.torch.cuda.empty_cache()

    def _matmat(self,x):
        if self.released:raise RuntimeError('Retired GPU linearization used after replacement')
        if not np.isfinite(x).all():raise ArithmeticError('Nonfinite input refused before GPU forward product')
        t=self.torch.as_tensor(np.ascontiguousarray(x),dtype=self.torch.float64,device=self.device)
        result=self.torch.sparse.mm(self.J,t).cpu().numpy()
        if not np.isfinite(result).all():raise ArithmeticError('Nonfinite GPU CSR forward product')
        return result

    def _matvec(self,x):return self._matmat(np.asarray(x).reshape(-1,1)).ravel()

    def _rmatvec(self,x):
        if self.released:raise RuntimeError('Retired GPU linearization used after replacement')
        if not np.isfinite(x).all():raise ArithmeticError('Nonfinite input refused before GPU transpose product')
        t=self.torch.as_tensor(np.ascontiguousarray(x).reshape(-1,1),dtype=self.torch.float64,device=self.device)
        result=self.torch.sparse.mm(self.J.transpose(0,1),t).cpu().numpy().ravel()
        if not np.isfinite(result).all():raise ArithmeticError('Nonfinite GPU CSR transpose product')
        return result

    def _prepare_gram(self):
        if self._gram is None:
            torch=self.torch;started=time.perf_counter();n=self.shape[1]
            # A single full SpGEMM can demand a large symbolic workspace. These
            # are row blocks of ONE global normal matrix, never separate fits.
            block_rows=int(os.environ.get('RINKSPACE_GPU_GRAM_ROWS','96'))
            if block_rows<1:raise ValueError('GPU Gram row block must be positive')
            transpose=self.J.transpose(0,1).to_sparse_csr()
            offsets=transpose.crow_indices().cpu().numpy()
            rows=[];indices=[];data=[];diagonal=[];total=0;kernel_seconds=0.
            for first in range(0,n,block_rows):
                last=min(n,first+block_rows);a,b=map(int,offsets[[first,last]])
                left=torch.sparse_csr_tensor(transpose.crow_indices()[first:last+1]-a,
                    transpose.col_indices()[a:b],transpose.values()[a:b],size=(last-first,self.shape[0]))
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
                gram=torch.sparse.mm(left,self.J)
                # Keep an explicit diagonal even for structurally empty columns
                # so later TRF damping uses the identical complete layout.
                zero_diag=torch.sparse_csr_tensor(torch.arange(last-first+1,dtype=torch.int32,device=self.device),
                    torch.arange(first,last,dtype=torch.int32,device=self.device),
                    torch.zeros(last-first,dtype=torch.float64,device=self.device),size=(last-first,n))
                gram=gram+zero_diag;end.record();end.synchronize();kernel_seconds+=begin.elapsed_time(end)*.001
                row=gram.crow_indices();index=gram.col_indices();values=gram.values()
                # cuSPARSE emits sorted columns. Locate the unique diagonal on
                # device; only the small row pointer is read by scaling later.
                rr=torch.repeat_interleave(torch.arange(first,last,device=self.device),
                    (row[1:]-row[:-1]).long(),output_size=len(values))
                diagonal.append(torch.nonzero(index==rr,as_tuple=False).ravel()+total)
                rows.append(row[1:]+total);indices.append(index);data.append(values);total+=len(values)
                del left,gram,zero_diag,rr
            del transpose
            row=torch.cat([torch.zeros(1,dtype=torch.int32,device=self.device),*rows])
            index=torch.cat(indices);values=torch.cat(data)
            self._diag=torch.cat(diagonal)
            if len(self._diag)!=n:raise ArithmeticError('GPU Gram lost a diagonal or emitted duplicate columns')
            self._gram=(row,index,values);self._host_rowptr=row.cpu().numpy();self._host=None
            torch.cuda.synchronize();self._gram_seconds=time.perf_counter()-started
            self.gram_record=dict(gram_backend='CUDA_FP64_block_SpGEMM',gram_device_block_rows=block_rows,
                gram_GPU_wall_seconds=self._gram_seconds,gram_GPU_kernel_seconds=kernel_seconds,
                gram_H2D_bytes=0,gram_D2H_bytes=self._host_rowptr.nbytes)

    def solve_scaled(self,rhs,scale,diagonal=None,damp=0.):
        torch=self.torch;self._prepare_gram();row,index,base=self._gram;n=self.shape[1]
        s=torch.as_tensor(scale,dtype=torch.float64,device=self.device);values=base.clone()
        for first in range(0,n,1024):
            last=min(first+1024,n);a,b=map(int,self._host_rowptr[[first,last]])
            rr=torch.repeat_interleave(s[first:last],(row[first+1:last+1]-row[first:last]).long(),output_size=b-a)
            values[a:b]*=rr*s[index[a:b].long()]
        penalty=np.full(n,float(damp)**2)
        if diagonal is not None:penalty+=diagonal**2
        values[self._diag]+=torch.as_tensor(penalty,dtype=torch.float64,device=self.device)
        right=scale*self.rmatvec(np.asarray(rhs)[:self.shape[0]])
        if diagonal is not None:right+=diagonal*np.asarray(rhs)[self.shape[0]:]
        b=torch.as_tensor(right,dtype=torch.float64,device=self.device);A=torch.sparse_csr_tensor(row,index,values,size=(n,n))
        if self._dss is None:self._dss=CuDSS()
        x,record=self._dss.solve(A,b)
        error=float((torch.linalg.vector_norm(torch.sparse.mm(A,x[:,None])[:,0]-b)/torch.linalg.vector_norm(b).clamp_min(torch.finfo(torch.float64).tiny)).item())
        record.update(relative_normal_residual=error,normal_nnz=len(values),**self.gram_record,**self.transfer)
        if not np.isfinite(error)or error>1e-8:raise ArithmeticError('GPU normal-system relative residual '+str(error))
        return x.cpu().numpy(),record


def gpu_trf(fun,jac,x0,lower,upper,*,callback,options):
    """The installed SciPy TRF, with its identical adaptive Jacobian scaling."""
    import scipy.optimize._lsq.trf as trf
    from scipy.optimize._optimize import _wrap_callback
    from .body_linear_solver import _LOCK
    old=trf.compute_jac_scale
    def scaling(J,previous=None):
        if not isinstance(J,GPUJacobian):return old(J,previous)
        inv=J.scale_inv.copy()
        if previous is None:inv[inv==0]=1.
        else:inv=np.maximum(inv,previous)
        return 1/inv,inv
    active=None
    def replacing_jac(z):
        nonlocal active
        if active is not None:active.release();active=None
        result=jac(z)
        if isinstance(result,GPUJacobian):active=result
        return result
    with _LOCK:
        trf.compute_jac_scale=scaling
        try:
            # Calling TRF directly avoids least_squares rejecting adaptive
            # scaling for LinearOperator; all values and column norms are real.
            f0=fun(x0)
            if not np.isfinite(f0).all():raise ArithmeticError('Initial state fails the guarded source quality; no nonfinite residual sent to GPU')
            result=trf.trf(fun,replacing_jac,np.asarray(x0),f0,replacing_jac(x0),lower,upper,
                1e-7,1e-7,1e-4,None,'jac',None,'lsmr',dict(options),0,callback=_wrap_callback(callback))
            result.message=f'SciPy TRF status {result.status}; accuracy is reviewed separately'
            return result
        finally:
            trf.compute_jac_scale=old
            if active is not None:active.release()
