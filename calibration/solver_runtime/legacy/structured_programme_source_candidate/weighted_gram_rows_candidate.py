"""Optional row-parallel Gram for one unchanged weighted source CSR block.

The original normal assembler remains the owner of r/J, weighting, gradient,
contracts and replay.  This function only evaluates sum(WJ_i.T @ WJ_i) for
contiguous complete row chunks, keeping every original parameter column.
Chunk-associated floating summation is equivalent up to roundoff, not bitexact.
"""
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import operator
import os
import time

import numpy as np
from scipy.sparse import csr_matrix

_POOL = None
_POOL_WORKERS = None
_ACTIVE = Lock()
_RESERVE_BYTES = 0
_LAST_REPORT = None


def _original_gram(weighted):
    return (weighted.T @ weighted).tocsc()


def _ranges(rows, jobs):
    edges = np.linspace(0, rows, jobs+1, dtype=np.int64)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _row_view(weighted, start, stop):
    lo, hi = int(weighted.indptr[start]), int(weighted.indptr[stop])
    view = csr_matrix((weighted.data[lo:hi], weighted.indices[lo:hi],
        weighted.indptr[start:stop+1]-lo), shape=(stop-start, weighted.shape[1]), copy=False)
    view.has_sorted_indices = weighted.has_sorted_indices
    view.has_canonical_format = weighted.has_canonical_format
    return view


def _growth_bound(weighted, ranges):
    """Conservative result/storage bound from each chunk's active columns.

    No dense matrix or symbolic Gram is allocated.  CSR input data/indices
    remain views.  Full column shapes and CSC pointers are retained even when
    a contiguous chunk uses only part of the original camera/world columns.
    """
    columns = weighted.shape[1]
    partial_bytes = 0
    for start, stop in ranges:
        lo, hi = int(weighted.indptr[start]), int(weighted.indptr[stop])
        active = np.count_nonzero(np.bincount(weighted.indices[lo:hi], minlength=columns))
        degree = np.diff(weighted.indptr[start:stop+1]).astype(np.int64, copy=False)
        maximum_nnz = min(int(active)*int(active), int(degree @ degree))
        partial_bytes += maximum_nnz*16+(columns+1)*8
    input_bytes = weighted.data.nbytes+weighted.indices.nbytes+weighted.indptr.nbytes
    # All partial results, current reduction plus its replacement, temporary
    # CSC input conversion, per-thread sparse work arrays and thread overhead.
    return 3*partial_bytes+input_bytes+len(ranges)*columns*32+128*1024**2


def weighted_gram(weighted, *, workers=None):
    """Return the original complete CSC Gram with requested row concurrency.

    One source block is active.  Producer/FD jobs must have completed before
    the normal assembler calls this function.  No IPC, disk matrix, source row
    selection, new normalization or solver is introduced. Only nonempty row
    chunks limit the requested concurrency; no memory reserve downgrades it.
    """
    global _POOL, _POOL_WORKERS, _LAST_REPORT
    if not isinstance(weighted, csr_matrix):
        raise TypeError('Original weighted source CSR required')
    try:
        requested = int(os.environ.get('RINKSPACE_NORMAL_WORKERS', '1')) if workers is None else operator.index(workers)
    except (TypeError, ValueError) as exc:
        raise ValueError('Positive integer Gram thread request required') from exc
    if requested < 1:
        raise ValueError('Positive integer Gram thread request required')
    begin = time.perf_counter()
    with _ACTIVE:
        if _POOL is not None and _POOL_WORKERS != requested:
            raise RuntimeError('Gram pool already uses '+str(_POOL_WORKERS)+' threads; '
                'close_weighted_gram_pool() before requesting '+str(requested))
        selected = min(requested, max(1, weighted.shape[0]))
        ranges = _ranges(weighted.shape[0], selected) if selected > 1 else None
        if selected == 1:
            result = _original_gram(weighted)
        else:
            if _POOL is None:
                _POOL = ThreadPoolExecutor(max_workers=requested, thread_name_prefix='rink-gram')
                _POOL_WORKERS = requested
            chunks = [_row_view(weighted, start, stop) for start, stop in ranges]
            futures = [_POOL.submit(_original_gram, chunk) for chunk in chunks]
            result = None
            try:
                # Deterministic original contiguous-row order. Sparse additions
                # only regroup floating sums; no source columns are compressed.
                for index, future in enumerate(futures):
                    part = future.result()
                    result = part if result is None else result+part
                    # A completed Future owns its result too. Drop that
                    # reference once the ordered reduction has consumed it.
                    futures[index] = None
                    del part, future
            finally:
                # Exceptions cannot leave numerical work active during a later
                # source/FD producer stage. No thread is forcefully interrupted.
                for index, future in enumerate(futures):
                    if future is None:
                        continue
                    try:
                        future.result()
                    except Exception:
                        pass
                    finally:
                        futures[index] = None
                future = part = None
            result = result.tocsc()
        _LAST_REPORT = dict(route='original_scipy' if selected == 1 else 'contiguous_row_threads',
            requested_workers=requested, jobs=selected, rows=weighted.shape[0], columns=weighted.shape[1], input_nnz=weighted.nnz,
            result_nnz=result.nnz, estimated_extra_bytes=None,
            engineering_reserve_bytes=_RESERVE_BYTES, wall_seconds=time.perf_counter()-begin,
            memory_admission_enabled=False,
            floating_reduction_order='contiguous row chunks, original chunk order',
            H_bitexact_claim=False, rows_columns_or_source_weighting_changed=False)
        return result


def last_gram_report():
    return None if _LAST_REPORT is None else dict(_LAST_REPORT)


def close_weighted_gram_pool():
    """Finish outstanding work and release only this owned thread pool."""
    global _POOL, _POOL_WORKERS
    with _ACTIVE:
        if _POOL is not None:
            _POOL.shutdown(wait=True)
            _POOL = None
            _POOL_WORKERS = None
