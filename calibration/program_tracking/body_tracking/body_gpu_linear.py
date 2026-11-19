"""FP64 device-resident CSR products and real cuDSS normal-system solves.

The weighted CSR is assembled by the existing factors. Its values are copied
unchanged to CUDA. Bounded row blocks of J.T form the same FP64 normal matrix
on the device; residual/geometry assembly remains in the existing factors.
"""
from pathlib import Path
import ctypes,os,time
import json
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator


def _coalesce_gram_tiles(tiles, *, max_rows=1024, max_nnz=2_000_000):
    """Group adjacent Gram tiles for elementwise normal-matrix scaling.

    Reuse the known row/value offsets without a row-pointer download. A single
    tile larger than a budget remains intact; no new matrix entries or changes
    to arithmetic are introduced by grouping the launch ranges.
    """
    if max_rows < 1 or max_nnz < 1:
        raise ValueError('Normal scaling group budgets must be positive')
    groups=[]
    current=None
    for first,last,a,b in tiles:
        if first < 0 or a < 0 or last < first or b < a or (first == last and a != b):
            raise ValueError('Invalid Gram tile range')
        if current is not None:
            if first != current[1] or a != current[3]:
                raise ValueError('Gram tiles must be contiguous in rows and values')
            if last-current[0] <= max_rows and b-current[2] <= max_nnz:
                current=(current[0],last,current[2],b)
                continue
            groups.append(current)
        current=(first,last,a,b)
    if current is not None:
        groups.append(current)
    return groups


class CuDSS:
    def __init__(self):
        root=Path(os.environ.get('RINKSPACE_DATA_ROOT','D:/CodexData/RinkSpace_Data'))/'runtime_tools/cudss/0.8-cu13'
        cuda=Path(os.environ.get('CUDA_PATH','D:/Program Files/CUDA'))
        self.dirs=[os.add_dll_directory(str(p))for p in(root/'bin',cuda/'bin/x64',cuda/'bin')if p.is_dir()]
        self.lib=ctypes.WinDLL(str(root/'bin/cudss64_0.dll'));p=ctypes.c_void_p;i=ctypes.c_int;ll=ctypes.c_int64;s=ctypes.c_size_t
        types=dict(cudssCreate=[ctypes.POINTER(p)],cudssDestroy=[p],cudssConfigCreate=[ctypes.POINTER(p)],cudssConfigDestroy=[p],cudssDataCreate=[p,ctypes.POINTER(p)],cudssDataDestroy=[p,p],cudssSetStream=[p,p],cudssConfigSet=[p,i,p,s],cudssMatrixCreateCsr=[ctypes.POINTER(p),ll,ll,ll,p,p,p,p,i,i,i,i,i,i],cudssMatrixCreateDn=[ctypes.POINTER(p),ll,ll,ll,p,i,i],cudssMatrixDestroy=[p],cudssMatrixSetValues=[p,p],cudssExecute=[p,i,p,p,p,p,p],cudssDataGet=[p,p,i,p,s,ctypes.POINTER(s)])
        for name,args in types.items():
            fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=i
        self.path=str(root/'bin/cudss64_0.dll')

    def call(self,name,*args):
        status=getattr(self.lib,name)(*args)
        if status:raise RuntimeError(f'{name} cuDSS status {status}')

    def solve(self,A,rhs):
        with CuDSSSession(self) as session:
            return session.solve(A,rhs)


class CuDSSSession:
    """One current-stream cuDSS owner. Wrappers borrow resident tensor storage."""
    def __init__(self,runtime=None):
        import torch
        self.torch=torch
        self.runtime=CuDSS() if runtime is None else runtime
        self.call=self.runtime.call
        self.stream=torch.cuda.current_stream()
        self.handle,self.config,self.data,self.am,self.xm,self.bm=(ctypes.c_void_p() for _ in range(6))
        self.A=self.x=self.b=None
        self.analysis_count=self.factor_count=self.solve_count=0
        self.matrix_key=None

    @staticmethod
    def ptr(tensor):
        return ctypes.c_void_p(tensor.data_ptr())

    def __enter__(self):
        try:
            self.call('cudssCreate',ctypes.byref(self.handle))
            self.call('cudssSetStream',self.handle,ctypes.c_void_p(self.stream.cuda_stream))
            self.call('cudssConfigCreate',ctypes.byref(self.config))
            zero=ctypes.c_int(0)
            for key in (11,15):
                self.call('cudssConfigSet',self.config,key,ctypes.byref(zero),ctypes.sizeof(zero))
            return self
        except BaseException:
            self.close()
            raise

    def _release_matrices(self):
        for name in ('bm','xm','am'):
            handle=getattr(self,name)
            if handle:
                self.call('cudssMatrixDestroy',handle)
                setattr(self,name,ctypes.c_void_p())
        if self.data:
            self.call('cudssDataDestroy',self.handle,self.data)
            self.data=ctypes.c_void_p()
        self.A=self.x=self.b=None
        self.matrix_key=None

    def close(self):
        try:
            self.stream.synchronize()
            self._release_matrices()
        finally:
            if self.config:
                self.call('cudssConfigDestroy',self.config)
                self.config=ctypes.c_void_p()
            if self.handle:
                self.call('cudssDestroy',self.handle)
                self.handle=ctypes.c_void_p()

    def release_factorization(self):
        """Retire a completed proposal's matrix before the next nonlinear linearization."""
        self.stream.synchronize()
        self._release_matrices()
        self.torch.cuda.empty_cache()

    def __exit__(self,*exc):
        self.close()

    def matches_structure(self,row,col,shape):
        if self.A is None or self.A.shape!=shape or self.A._nnz()!=col.numel():
            return False
        old_row,old_col=self.A.crow_indices(),self.A.col_indices()
        if row.data_ptr()==old_row.data_ptr() and col.data_ptr()==old_col.data_ptr():
            return True
        return bool(((row==old_row).all() & (col==old_col).all()).item())

    def solve(self,A,rhs,*,matrix_key=None):
        torch=self.torch
        if torch.cuda.current_stream()!=self.stream or A.device!=rhs.device:
            raise ValueError('cuDSS solve changed its CUDA owner')
        started=time.perf_counter()
        same_values=matrix_key is not None and self.matrix_key==matrix_key
        same=self.matches_structure(A.crow_indices(),A.col_indices(),A.shape)
        if same_values and not same:
            raise ValueError('Identical cuDSS matrix key has a different canonical structure')
        setup=time.perf_counter()-started
        n=len(rhs)
        if not same:
            self._release_matrices()
            self.call('cudssDataCreate',self.handle,ctypes.byref(self.data))
            self.x=torch.empty_like(rhs)
            self.b=torch.empty_like(rhs)
            self.A=A
            self.call('cudssMatrixCreateCsr',ctypes.byref(self.am),n,n,A._nnz(),self.ptr(A.crow_indices()),None,
                self.ptr(A.col_indices()),self.ptr(A.values()),10,10,1,3,0,0)
            for handle,tensor in ((self.xm,self.x),(self.bm,self.b)):
                self.call('cudssMatrixCreateDn',ctypes.byref(handle),n,1,n,self.ptr(tensor),1,0)
        elif not same_values and self.A.values().data_ptr()!=A.values().data_ptr():
            # Keep the original structure pointers alive when updating values.
            self.A=torch.sparse_csr_tensor(self.A.crow_indices(),self.A.col_indices(),A.values(),size=A.shape)
            self.call('cudssMatrixSetValues',self.am,self.ptr(self.A.values()))
        self.b.copy_(rhs)
        events={}
        def phase(code,name):
            begin=torch.cuda.Event(enable_timing=True)
            end=torch.cuda.Event(enable_timing=True)
            begin.record(self.stream)
            self.call('cudssExecute',self.handle,code,self.config,self.data,self.am,self.xm,self.bm)
            end.record(self.stream)
            events[name]=begin,end
        written=ctypes.c_size_t()
        if not same:
            phase(3,'analysis')
            events['analysis'][1].synchronize()
            estimates=(ctypes.c_int64*16)()
            self.call('cudssDataGet',self.handle,self.data,13,ctypes.byref(estimates),ctypes.sizeof(estimates),ctypes.byref(written))
            self.estimates=list(estimates)[:written.value//8]
            self.analysis_count+=1
        if not same_values:
            phase(8 if same else 4,'factor')
            self.factor_count+=1
        phase(1008,'solve')
        events['solve'][1].synchronize()
        self.solve_count+=1
        info=ctypes.c_int()
        self.call('cudssDataGet',self.handle,self.data,0,ctypes.byref(info),ctypes.sizeof(info),ctypes.byref(written))
        if info.value:
            raise ArithmeticError(f'cuDSS numerical info {info.value}')
        self.matrix_key=matrix_key
        record={name+'_seconds':(events[name][0].elapsed_time(events[name][1])*.001 if name in events else 0.)
                for name in ('analysis','factor','solve')}
        record.update(backend='cuDSS_FP64_device',library=self.runtime.path,memory_estimates_bytes=self.estimates,
            analysis_reused=same,numeric_factor_reused=same_values,structure_check_seconds=setup,
            session_analysis_count=self.analysis_count,session_factor_count=self.factor_count,
            session_solve_count=self.solve_count,wall_seconds=time.perf_counter()-started,
            device_allocated_bytes=int(torch.cuda.memory_allocated()),device_reserved_bytes=int(torch.cuda.memory_reserved()))
        return self.x,record


class DeviceCSRLayout:
    """Canonical slots retained across numeric evaluations of one control layout."""
    row_batch = 16384

    def __init__(self,*,retain_inactive_slots=True):
        self.shape=None
        self.epoch=0
        self.linearization_id=0
        self.retain_inactive_slots=retain_inactive_slots
        self.row=self.columns=self.row_device=self.columns_device=None

    def upload(self,matrix,prefix,torch,stop=None):
        matrix.sum_duplicates()
        ptr=matrix.indptr.astype(np.int32,copy=False)
        col=matrix.indices.astype(np.int32,copy=False)
        stop=matrix.shape[1] if stop is None else stop
        streamed=stop<matrix.shape[1] and len(col)>0 and col.max()>=stop
        if streamed:
            ptr,col=self._clip_structure(ptr,col,stop)

        def read_values(first,last):
            lo,hi=matrix.indptr[first],matrix.indptr[last]
            values=matrix.data[lo:hi]
            if streamed and ptr[last]-ptr[first]!=hi-lo:
                values=values[matrix.indices[lo:hi]<stop]
            return values

        shape=(matrix.shape,prefix,stop)
        same=self.shape==shape
        equal=same and np.array_equal(ptr,self.row) and np.array_equal(col,self.columns)
        reused=equal
        layout_bytes=0
        active_entries=len(col)
        aligned=(self._merge_slots(matrix.shape[0],stop,ptr,col,read_values,torch)
                 if same and not equal and self.retain_inactive_slots else None)
        if aligned is not None:
            ptr,col,values=aligned
            reused=ptr is self.row
        elif streamed:
            values=torch.empty(active_entries,dtype=torch.float64,device='cuda')
            for first in range(0,matrix.shape[0],self.row_batch):
                last=min(first+self.row_batch,matrix.shape[0])
                values[ptr[first]:ptr[last]].copy_(torch.from_numpy(read_values(first,last)))
        else:
            values=torch.from_numpy(matrix.data).to('cuda')
        if not equal:
            if not reused:
                self.epoch+=1
                self.shape=shape
                # Own the reference structure: callers may reuse and edit a
                # CSR buffer after upload. Aliases would make a changed layout
                # compare equal while the device still holds the old indices.
                self.row=ptr.copy()
                self.columns=col.copy()
                self.row_device=torch.from_numpy(self.row).to('cuda')
                self.columns_device=torch.from_numpy(self.columns).to('cuda')
                if prefix:
                    self.columns_device.sub_(prefix)
                layout_bytes=ptr.nbytes+col.nbytes
        self.linearization_id+=1
        return self.row_device,self.columns_device,values,dict(
            linearization_id=self.linearization_id,jacobian_structure_epoch=self.epoch,
            jacobian_structure_reused=reused,jacobian_structure_H2D_bytes=layout_bytes,
            retained_structural_zero_slots=len(col)-active_entries,
            inactive_entries_skipped=matrix.nnz-active_entries,
            host_slice_value_bytes_avoided=active_entries*matrix.data.itemsize if stop<matrix.shape[1] else 0,
            jacobian_H2D_bytes=layout_bytes+values.numel()*values.element_size())

    def _clip_structure(self,ptr,col,stop):
        """Filter each row without staging a second full array of Jacobian values."""
        clipped_ptr=np.empty_like(ptr);clipped_ptr[0]=0
        clipped_col=np.empty_like(col)
        written=0
        for first in range(0,len(ptr)-1,self.row_batch):
            last=min(first+self.row_batch,len(ptr)-1)
            lo,hi=int(ptr[first]),int(ptr[last]);tile=col[lo:hi]
            keep=tile<stop
            if keep.all():
                counts=ptr[first+1:last+1]-lo
            else:
                cumulative=np.empty(len(keep)+1,dtype=ptr.dtype);cumulative[0]=0
                np.cumsum(keep,dtype=ptr.dtype,out=cumulative[1:])
                counts=cumulative[ptr[first+1:last+1]-lo]
                tile=tile[keep]
            clipped_ptr[first+1:last+1]=written+counts
            clipped_col[written:written+len(tile)]=tile
            written+=len(tile)
        return clipped_ptr,clipped_col[:written]

    def _merge_slots(self,row_count,width,ptr,col,read_values,torch):
        # Merge the actual fixed row support and newly active witnesses.
        # Once a slot is admitted it remains even when its value is zero.
        # Bound the structural overhang independently from the live values.
        slots=[]
        count=0
        extra_limit=(64*1024**2)//(4+8)
        for lo in range(0,row_count,self.row_batch):
            hi=min(lo+self.row_batch,row_count)
            left,right=int(self.row[lo]),int(self.row[hi])
            a,b=int(ptr[lo]),int(ptr[hi])
            previous=self.columns[left:right]
            incoming=col[a:b]
            counts=np.diff(ptr[lo:hi+1])
            old_counts=np.diff(self.row[lo:hi+1])
            if np.array_equal(counts,old_counts) and np.array_equal(incoming,previous):
                indices,counts,mapping=previous,old_counts,None
            else:
                rows=np.arange(hi-lo,dtype=np.int64)*width
                old_keys=np.repeat(rows,old_counts)+previous
                new_keys=np.repeat(rows,counts)+incoming
                union=np.union1d(old_keys,new_keys)
                mapping=np.searchsorted(union,new_keys).astype(np.int32)
                if len(union)==len(old_keys):
                    indices,counts=previous,old_counts
                else:
                    indices=(union%width).astype(np.int32)
                    counts=np.bincount(union//width,minlength=hi-lo)
            slots.append((lo,hi,indices,counts,mapping))
            count+=len(indices)
            if count-b>extra_limit:
                return None
        result=torch.empty(count,dtype=torch.float64,device='cuda')
        growth=count>len(self.columns)
        row=np.empty_like(ptr) if growth else self.row
        columns=np.empty(count,dtype=col.dtype) if growth else self.columns
        if growth:
            row[0]=0
        out=0
        for lo,hi,indices,counts,mapping in slots:
            stop=out+len(indices)
            if growth:
                np.cumsum(counts,out=row[lo+1:hi+1])
                row[lo+1:hi+1]+=out
                columns[out:stop]=indices
            tile=read_values(lo,hi)
            if mapping is not None:
                aligned=np.zeros(len(indices),dtype=tile.dtype)
                aligned[mapping]=tile
                tile=aligned
            result[out:stop].copy_(torch.from_numpy(tile))
            out=stop
        return row,columns,result


class GPUJacobian(LinearOperator):
    """A real FP64 CSR operator; no host array impersonates device memory."""
    def __init__(self,matrix,*,shared_prefix=0,column_stop=None,session=None,layout=None):
        import torch
        self.torch=torch;matrix=matrix.tocsr();self.prefix=int(shared_prefix);self._gram=None
        # A layout's diagnostic counter starts at one and is not a global
        # numeric identity. This token stays unique for this linearization,
        # including when a shared cuDSS session outlives the operator/layout.
        self._numeric_identity=object()
        self._normal_transpose=None;self._transpose_argument=None;self._transpose_result=None
        if not np.isfinite(matrix.data).all():raise ArithmeticError('Nonfinite actual weighted CSR before GPU upload')
        if self.prefix and matrix.nnz and np.min(matrix.indices)<self.prefix:raise ValueError('Frozen prefix has nonzero Jacobian columns')
        stop=matrix.shape[1] if column_stop is None else int(column_stop)
        if not self.prefix<stop<=matrix.shape[1]:raise ValueError('Invalid active Jacobian column window')
        started=time.perf_counter();cols=stop-self.prefix
        layout=DeviceCSRLayout() if layout is None else layout
        row,index,values,transfer=layout.upload(matrix,self.prefix,torch,stop)
        self.device=values.device;self.J=torch.sparse_csr_tensor(row,index,values,size=(matrix.shape[0],cols))
        # A full CUDA CSC-to-CSR conversion temporarily expands this very
        # large Jacobian. Stream exact CPU-transposed row blocks on small
        # devices instead; the normal products and solve still run on CUDA.
        self._host_J=(matrix[:,self.prefix:stop].tocsr()
                      if os.environ.get('RINKSPACE_GPU_HOST_TRANSPOSE')=='1' else None)
        norm=torch.zeros(cols,dtype=torch.float64,device=self.device)
        for first in range(0,len(values),4_000_000):
            last=min(first+4_000_000,len(values));norm.scatter_add_(0,index[first:last].long(),values[first:last].square())
        self.scale_inv=torch.sqrt(norm).cpu().numpy()
        if not np.isfinite(self.scale_inv).all():raise ArithmeticError('Nonfinite device Jacobian column norm')
        self.transfer=dict(transfer,jacobian_H2D_seconds=time.perf_counter()-started)
        self._dss=session;self.released=False;super().__init__(dtype=np.dtype('float64'),shape=(matrix.shape[0],cols))

    def release(self,*,release_factorization=False):
        """Retire a completed linearization while TRF still holds its wrapper.

        SciPy requests the next Jacobian only after accepting its new fun state.
        Old trust-region operators can remain referenced until the next loop,
        but their device data must not overlap the new FK/J/Gram workspace.
        """
        if self.released:return
        session=self._dss
        self.J=None;self._host_J=None;self._gram=None;self._diag=None;self._gram_tiles=None;self._scaling_tiles=None
        self._normal_transpose=None;self._transpose_argument=None;self._transpose_result=None
        self._dss=None;self.released=True
        if release_factorization and session is not None:session.release_factorization()
        else:self.torch.cuda.empty_cache()

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
        argument=np.ascontiguousarray(x).reshape(-1)
        if self._transpose_argument is not None and np.array_equal(argument,self._transpose_argument):
            return self._transpose_result.copy()
        if self._host_J is not None:
            result=np.asarray(self._host_J.T@argument).ravel()
            if not np.isfinite(result).all():raise ArithmeticError('Nonfinite exact host CSR transpose product')
            self._transpose_argument=argument.copy();self._transpose_result=result.copy()
            return result
        t=self.torch.as_tensor(argument[:,None],dtype=self.torch.float64,device=self.device)
        prepared=self._normal_transpose is not None
        operator=self._normal_transpose if prepared else self.J.transpose(0,1)
        result=self.torch.sparse.mm(operator,t).cpu().numpy().ravel()
        if not np.isfinite(result).all():raise ArithmeticError('Nonfinite GPU CSR transpose product')
        self._transpose_argument=argument.copy();self._transpose_result=result.copy()
        if prepared:
            self._normal_transpose=None;del operator
            self.torch.cuda.empty_cache()
        return result

    def _prepare_gram(self):
        if self._gram is None:
            torch=self.torch;started=time.perf_counter();n=self.shape[1]
            # A single full SpGEMM can demand a large symbolic workspace. These
            # are row blocks of ONE global normal matrix, never separate fits.
            block_rows=int(os.environ.get('RINKSPACE_GPU_GRAM_ROWS','96'))
            if block_rows<1:raise ValueError('GPU Gram row block must be positive')
            host_transpose=self._host_J is not None
            transpose=self._host_J.T.tocsr() if host_transpose else self.J.transpose(0,1).to_sparse_csr()
            offsets=transpose.indptr if host_transpose else transpose.crow_indices().cpu().numpy()
            capacity=min(n*n,704*n)
            row=torch.zeros(n+1,dtype=torch.int32,device=self.device)
            index=torch.empty(capacity,dtype=torch.int32,device=self.device)
            data=torch.empty(capacity,dtype=torch.float64,device=self.device)
            diagonal=torch.empty(n,dtype=torch.int64,device=self.device)
            self._gram_tiles=[]
            total=0
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            begin.record()
            for first in range(0,n,block_rows):
                last=min(n,first+block_rows);a,b=map(int,offsets[[first,last]])
                if host_transpose:
                    left=torch.sparse_csr_tensor(
                        torch.as_tensor(transpose.indptr[first:last+1]-a,dtype=torch.int32,device=self.device),
                        torch.as_tensor(transpose.indices[a:b],dtype=torch.int32,device=self.device),
                        torch.as_tensor(transpose.data[a:b],dtype=torch.float64,device=self.device),
                        size=(last-first,self.shape[0]))
                else:
                    left=torch.sparse_csr_tensor(transpose.crow_indices()[first:last+1]-a,
                        transpose.col_indices()[a:b],transpose.values()[a:b],size=(last-first,self.shape[0]))
                gram=torch.sparse.mm(left,self.J)
                # Keep an explicit diagonal even for structurally empty columns
                # so later TRF damping uses the identical complete layout.
                zero_diag=torch.sparse_csr_tensor(torch.arange(last-first+1,dtype=torch.int32,device=self.device),
                    torch.arange(first,last,dtype=torch.int32,device=self.device),
                    torch.zeros(last-first,dtype=torch.float64,device=self.device),size=(last-first,n))
                gram=gram+zero_diag
                grow=gram._nnz()
                stop=total+grow
                if stop>capacity:
                    capacity=max(stop,capacity+capacity//4)
                    new_index=torch.empty(capacity,dtype=index.dtype,device=self.device)
                    new_data=torch.empty(capacity,dtype=data.dtype,device=self.device)
                    new_index[:total].copy_(index[:total]);new_data[:total].copy_(data[:total])
                    index,data=new_index,new_data
                rows=gram.crow_indices();cols=gram.col_indices();values=gram.values()
                # Sorted row/column composite keys find exactly one diagonal
                # per row without a dynamic-size nonzero and host sync.
                rr=torch.repeat_interleave(torch.arange(first,last,device=self.device),
                    (rows[1:]-rows[:-1]).long(),output_size=grow)
                target=torch.arange(first,last,device=self.device)
                diagonal[first:last]=torch.searchsorted(rr*n+cols,target*(n+1))+total
                row[first+1:last+1].copy_(rows[1:]+total)
                index[total:stop].copy_(cols);data[total:stop].copy_(values)
                self._gram_tiles.append((first,last,total,stop))
                total=stop
                del left,gram,zero_diag,rr
            transpose_transfer=(transpose.data.nbytes+transpose.indices.nbytes+
                                transpose.indptr.nbytes) if host_transpose else 0
            self._normal_transpose=None if host_transpose else transpose
            del transpose
            index=index[:total];data=data[:total]
            self._diag=diagonal
            rr=torch.arange(n,device=self.device)
            valid=(diagonal>=row[:-1]) & (diagonal<row[1:]) & (index[diagonal.clamp(0,total-1)]==rr)
            if not bool(valid.all().item()):raise ArithmeticError('GPU Gram lost its canonical diagonal')
            end.record();end.synchronize()
            self._gram=(row,index,data)
            self._scaling_tiles=_coalesce_gram_tiles(self._gram_tiles)
            # Retain the CSR transpose through the first J.T@rhs, avoiding a
            # redundant CSC conversion. That product releases it before cuDSS.
            # Return only completed SpGEMM scratch at this phase boundary.
            torch.cuda.empty_cache()
            self._gram_seconds=time.perf_counter()-started
            self.gram_record=dict(gram_backend='CUDA_FP64_block_SpGEMM',gram_device_block_rows=block_rows,
                gram_GPU_wall_seconds=self._gram_seconds,gram_GPU_event_seconds=begin.elapsed_time(end)*.001,
                gram_H2D_bytes=transpose_transfer,gram_D2H_bytes=1 if host_transpose else offsets.nbytes+1,
                transpose_policy='exact_host_CSR_row_blocks' if host_transpose else 'device_CSC_conversion',
                reverse_product_backend='host_CSR' if host_transpose else 'CUDA',normal_storage_capacity=capacity,
                normal_scaling_groups=len(self._scaling_tiles),
                normal_scaling_max_rows=max((last-first for first,last,a,b in self._scaling_tiles),default=0),
                normal_scaling_max_nnz=max((b-a for first,last,a,b in self._scaling_tiles),default=0))

    def scaled_normal_objective(self,rhs,scale,*,damp=0.):
        """Same FP64 J normal model for a box-constrained linear trial.

        Return (value, gradient) in Jacobian-scaled coordinates. Only vectors
        cross the host boundary; the already assembled normal CSR stays on GPU.
        """
        torch=self.torch;self._prepare_gram();row,index,base=self._gram
        normal=torch.sparse_csr_tensor(row,index,base,size=(self.shape[1],self.shape[1]))
        s=torch.as_tensor(np.asarray(scale),dtype=torch.float64,device=self.device)
        b=torch.as_tensor(scale*self.rmatvec(rhs),dtype=torch.float64,device=self.device)
        penalty=float(damp)**2
        def objective(z):
            t=torch.as_tensor(np.ascontiguousarray(z),dtype=torch.float64,device=self.device)
            product=s*torch.sparse.mm(normal,(s*t)[:,None])[:,0]+penalty*t
            value=(t@(product*.5-b)).item()
            gradient=(product-b).cpu().numpy()
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                raise ArithmeticError('Nonfinite bounded GPU normal objective')
            return float(value),gradient
        return objective

    def solve_scaled(self,rhs,scale,diagonal=None,damp=0.):
        torch=self.torch;self._prepare_gram();row,index,base=self._gram;n=self.shape[1]
        matrix_key=(self._numeric_identity,self.transfer['linearization_id'],scale.tobytes(),
                    None if diagonal is None else diagonal.tobytes(),float(damp))
        if self._dss is not None and self._dss.matrix_key==matrix_key:
            A=self._dss.A
        else:
            same=self._dss is not None and self._dss.matches_structure(row,index,(n,n))
            if same:
                A=self._dss.A
                values=A.values()
                values.copy_(base)
            else:
                values=base.clone()
                A=torch.sparse_csr_tensor(row,index,values,size=(n,n))
            s=torch.as_tensor(scale,dtype=torch.float64,device=self.device)
            for first,last,a,b in self._scaling_tiles:
                rr=torch.repeat_interleave(s[first:last],(row[first+1:last+1]-row[first:last]).long(),output_size=b-a)
                values[a:b]*=rr*torch.index_select(s,0,index[a:b])
            penalty=np.full(n,float(damp)**2)
            if diagonal is not None:penalty+=diagonal**2
            values[self._diag]+=torch.as_tensor(penalty,dtype=torch.float64,device=self.device)
        right=scale*self.rmatvec(np.asarray(rhs)[:self.shape[0]])
        if diagonal is not None:right+=diagonal*np.asarray(rhs)[self.shape[0]:]
        b=torch.as_tensor(right,dtype=torch.float64,device=self.device)
        if self._dss is None:
            x,record=CuDSS().solve(A,b)
        else:
            x,record=self._dss.solve(A,b,matrix_key=matrix_key)
        error=float((torch.linalg.vector_norm(torch.sparse.mm(A,x[:,None])[:,0]-b)/torch.linalg.vector_norm(b).clamp_min(torch.finfo(torch.float64).tiny)).item())
        record.update(relative_normal_residual=error,normal_nnz=A._nnz(),**self.gram_record,**self.transfer)
        if not np.isfinite(error)or error>1e-8:raise ArithmeticError('GPU normal-system relative residual '+str(error))
        return x.cpu().numpy(),record


def gpu_trf(fun,jac,x0,lower,upper,*,callback,options,warm_relative=False,
            release_factorization_on_relinearize=False,log=None):
    """The installed SciPy TRF, with its identical adaptive Jacobian scaling."""
    origin=None
    if warm_relative:
        # The new objective starts near the saved trajectory. Using local
        # increments gives TRF a local initial radius instead of one scaled by
        # the whole displacement from the old prior. Bounds and Jacobian are
        # just translated in the same control coordinates.
        origin=np.asarray(x0).copy()
        original_fun,original_jac,original_callback=fun,jac,callback
        fun=lambda step:original_fun(origin+step)
        jac=lambda step:original_jac(origin+step)
        callback=None if original_callback is None else lambda step:original_callback(origin+step)
        x0=np.zeros_like(origin)
        lower,upper=lower-origin,upper-origin
    import scipy.optimize._lsq.trf as trf
    from inspect import signature
    from scipy.optimize._optimize import _wrap_callback
    from .body_linear_solver import _LOCK
    has_callback = 'callback' in signature(trf.trf).parameters
    old=trf.compute_jac_scale
    old_radius=trf.update_tr_radius
    def radius(*args,**kwargs):
        out=old_radius(*args,**kwargs)
        if log is not None:
            actual,predicted=args[1:3]
            log.write(json.dumps(dict(event='trf_trial_decision',actual_reduction=float(actual),
                predicted_reduction=float(predicted),improves=bool(actual>0),ratio=float(out[1])))+'\n')
        return out
    def scaling(J,previous=None):
        if not isinstance(J,GPUJacobian):return old(J,previous)
        inv=J.scale_inv.copy()
        if previous is None:inv[inv==0]=1.
        else:inv=np.maximum(inv,previous)
        return 1/inv,inv
    active=None
    last_linearized = np.asarray(x0).copy()
    def replacing_jac(z, _residual=None):
        nonlocal active, last_linearized
        if active is not None:
            active.release(release_factorization=release_factorization_on_relinearize)
            active=None
        # Older SciPy calls this after every accepted TRF update. Record the
        # update before building its next Jacobian, as the main solver does.
        if not has_callback and callback is not None and not np.array_equal(z,last_linearized):
            last_linearized = np.array(z,copy=True)
            callback(z)
        result=jac(z)
        if isinstance(result,GPUJacobian):active=result
        return result
    with _LOCK:
        trf.compute_jac_scale=scaling
        trf.update_tr_radius=radius
        try:
            # Calling TRF directly avoids least_squares rejecting adaptive
            # scaling for LinearOperator; all values and column norms are real.
            f0=fun(x0)
            if not np.isfinite(f0).all():raise ArithmeticError('Initial state fails the guarded source quality; no nonfinite residual sent to GPU')
            callback_options = {'callback': _wrap_callback(callback)} if has_callback else {}
            result=trf.trf(fun,replacing_jac,np.asarray(x0),f0,replacing_jac(x0),lower,upper,
                1e-7,1e-7,1e-4,None,'jac',None,'lsmr',dict(options),0,**callback_options)
            if origin is not None:result.x+=origin
            result.message=f'SciPy TRF status {result.status}; accuracy is reviewed separately'
            return result
        finally:
            trf.compute_jac_scale=old
            trf.update_tr_radius=old_radius
            if active is not None:active.release()
