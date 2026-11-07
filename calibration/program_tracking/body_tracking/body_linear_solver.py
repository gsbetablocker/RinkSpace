"""Sparse direct GN subproblem for the existing bounded SciPy TRF objective.

Keeps TRF scaling, diagonal regularization, trust region, bounds and acceptance.
Only the inner damped least-squares direction is solved differently. Unsupported
operators or a failed normal-system accuracy check use the original LSMR.
"""
from contextlib import contextmanager
import threading,time
import numpy as np
from scipy.sparse import issparse,diags
from scipy.sparse.linalg import splu

_LOCK=threading.RLock()

@contextmanager
def sparse_direct_trf(problem):
    if getattr(problem,'linear_solver_backend','lsmr')not in('sparse_direct','cuda_resident'):
        yield;return
    import scipy.optimize._lsq.trf as trf
    thread=threading.get_ident();records=getattr(problem,'linear_solver_statistics',None)
    if records is None:records=[];problem.linear_solver_statistics=records
    with _LOCK:
        old_right,old_regularized,old_lsmr=trf.right_multiplied_operator,trf.regularized_lsq_operator,trf.lsmr
        def right(J,d):
            op=old_right(J,d)
            if threading.get_ident()==thread and (issparse(J)or hasattr(J,'solve_scaled')):op._rinkspace_sparse=(J,np.asarray(d),None)
            return op
        def regularized(J,diag):
            op=old_regularized(J,diag)
            if threading.get_ident()==thread and hasattr(J,'_rinkspace_sparse'):
                matrix,scale,_=J._rinkspace_sparse;op._rinkspace_sparse=(matrix,scale,np.asarray(diag))
            return op
        def direction(op,rhs,**options):
            tagged=getattr(op,'_rinkspace_sparse',None)
            if threading.get_ident()!=thread or tagged is None:return old_lsmr(op,rhs,**options)
            start=time.perf_counter();matrix,scale,diagonal=tagged;row=dict(backend='sparse_direct',columns=matrix.shape[1],rows=matrix.shape[0]);fallback=None
            try:
                if hasattr(matrix,'solve_scaled'):
                    solution,device=matrix.solve_scaled(rhs,scale,diagonal,float(options.get('damp',0.)))
                    row.update(device,total_seconds=time.perf_counter()-start,fallback=False);records.append(row)
                    if getattr(problem,'log',None) is not None:
                        import json;problem.log.write(json.dumps(dict(event='linear_solve',**row))+'\n');problem.log.flush()
                    return(solution,)
                m,n=matrix.shape
                penalty=np.full(n,float(options.get('damp',0.))**2)
                if diagonal is not None:penalty+=diagonal**2
                # Scale the much smaller normal matrix, rather than copying
                # the entire programme Jacobian for every trust-region solve.
                gram=(matrix.T@matrix).tocsc()
                for first in range(0,n,1024):
                    last=min(first+1024,n);lo,hi=map(int,gram.indptr[[first,last]])
                    gram.data[lo:hi]*=scale[gram.indices[lo:hi]]*np.repeat(scale[first:last],np.diff(gram.indptr[first:last+1]))
                gram.setdiag(gram.diagonal()+penalty);gram.sum_duplicates();gram.sort_indices()
                right_hand=scale*np.asarray(matrix.T@np.asarray(rhs)[:m]).ravel()
                if diagonal is not None:right_hand+=diagonal*np.asarray(rhs)[m:]
                row['assembly_seconds']=time.perf_counter()-start;tick=time.perf_counter()
                factor=splu(gram,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.,options={'SymmetricMode':True})
                row['factor_seconds']=time.perf_counter()-tick;tick=time.perf_counter();solution=factor.solve(right_hand)
                denominator=max(float(np.linalg.norm(right_hand)),np.finfo(float).tiny)
                error=float(np.linalg.norm(gram@solution-right_hand)/denominator)
                tolerance=max(float(options.get('atol',1e-6)),float(options.get('btol',1e-6)))*.1
                if not np.isfinite(solution).all() or not np.isfinite(error) or error>tolerance:
                    raise ArithmeticError('normal-system relative residual '+str(error))
                # SuperLU exposes this scalar directly. Asking for L/U merely
                # to count entries materializes two additional large matrices.
                row.update(solve_seconds=time.perf_counter()-tick,normal_nnz=gram.nnz,factor_nnz=int(factor.nnz),relative_normal_residual=error,total_seconds=time.perf_counter()-start,fallback=False)
                records.append(row)
                if getattr(problem,'log',None) is not None:
                    import json;problem.log.write(json.dumps(dict(event='linear_solve',**row))+'\n')
                # The two audited TRF callers consume only element0. Do not
                # fabricate LSMR iteration/condition estimates for a factor solve.
                return (solution,)
            except (RuntimeError,ArithmeticError,ValueError) as exc:fallback=str(exc)
            row.update(total_seconds=time.perf_counter()-start,fallback=True,fallback_reason=fallback);records.append(row)
            if hasattr(matrix,'solve_scaled'):
                if getattr(problem,'log',None) is not None:
                    import json;problem.log.write(json.dumps(dict(event='GPU_linear_failure',**row))+'\n');problem.log.flush()
                raise ArithmeticError('GPU linear solve failed without an unbounded LSMR retry: '+fallback)
            return old_lsmr(op,rhs,**options)
        trf.right_multiplied_operator=right;trf.regularized_lsq_operator=regularized;trf.lsmr=direction
        try:yield
        finally:trf.right_multiplied_operator=old_right;trf.regularized_lsq_operator=old_regularized;trf.lsmr=old_lsmr
