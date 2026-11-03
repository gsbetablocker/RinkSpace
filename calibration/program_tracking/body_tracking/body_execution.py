"""Explicit execution policy for the existing, unchanged body objective.

Only independent local perturbations run concurrently. The SciPy trust region,
shared boot parameters and reference state remain owned by the calling thread.
CUDA batches perturbations through the same reference.world implementation;
the native camera, footwear factors and sparse TRF/LSMR remain CPU operations.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import os
import threading
import numpy as np


_BLAS_LIMIT_LOCK = threading.RLock()


@dataclass(frozen=True)
class BodyExecutionConfig:
    backend: str = 'cpu'
    workers: int = 1
    blas_threads: int = 1
    perturbation_batch: int = 4
    gpu_batch_size: int = 512
    max_batch_samples: int = 4096

    def __post_init__(self):
        if self.backend not in ('cpu', 'cuda'):
            raise ValueError('Body backend must be cpu or cuda')
        for key in ('workers', 'blas_threads', 'perturbation_batch', 'gpu_batch_size', 'max_batch_samples'):
            if getattr(self, key) < 1:
                raise ValueError(key + ' must be positive')
        if self.backend == 'cuda' and self.workers != 1:
            raise ValueError('Use one body CUDA owner; set RINKSPACE_BODY_WORKERS=1')

    @classmethod
    def from_environment(cls):
        prefix = 'RINKSPACE_BODY_'
        return cls(backend=os.environ.get(prefix+'BACKEND', 'cpu').lower(),
                   workers=int(os.environ.get(prefix+'WORKERS', '1')),
                   blas_threads=int(os.environ.get(prefix+'BLAS_THREADS', '1')),
                   perturbation_batch=int(os.environ.get(prefix+'PERTURBATION_BATCH', '4')),
                   gpu_batch_size=int(os.environ.get(prefix+'GPU_BATCH_SIZE', '512')),
                   max_batch_samples=int(os.environ.get(prefix+'MAX_BATCH_SAMPLES', '4096')))


def _slice_state(value, start, stop, total):
    # Metadata (e.g. lower_boot_vertices) is not sample-aligned. Only known
    # dynamic fields are needed by callers; leave all other fields untouched.
    sample_keys = {'P', 'Q', 'local_R', 'native', 'coco', 'gear', 'boot_R', 'boot_O',
                   'frames', 'world', 'upper_deformation',
                   'effective_ankle_relative_rotation', 'blade_R', 'blade_O', 'shank_world'}
    out = {}
    for key, item in value.items():
        if isinstance(item, dict):
            out[key] = _slice_state(item, start, stop, total)
        elif key in sample_keys:
            if not isinstance(item, np.ndarray) or item.shape[0] != total:
                raise ValueError('Unexpected batched body state layout: '+key)
            out[key] = item[start:stop]
        else:
            out[key] = item
    return out


class BodyExecution:
    def __init__(self, config=None):
        self.config = config or BodyExecutionConfig.from_environment()
        self._pool = None
        self._body_backend = None

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def limits(self):
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            if self.config.workers > 1:
                raise RuntimeError('Parallel body evaluation requires threadpoolctl; install it in the selected runtime')
            return nullcontext()
        return threadpool_limits(limits=self.config.blas_threads, user_api='blas')

    def map_columns(self, function, columns):
        """Ordered independent work; drain failures before shared state changes."""
        columns = list(columns)
        with _BLAS_LIMIT_LOCK, self.limits():
            if self.config.workers == 1 or len(columns) <= 1:
                return [function(column) for column in columns]
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=self.config.workers,
                                               thread_name_prefix='rink-body')
            futures = [self._pool.submit(function, column) for column in columns]
            try:
                return [future.result() for future in futures]
            except BaseException:
                for future in futures:
                    future.cancel()
                self.close()
                raise

    def attach(self, reference):
        """Attach to the actual owner, including OpeningReference proxies."""
        owner = reference
        seen = set()
        while 'body' not in vars(owner):
            if id(owner) in seen or 'parent' not in vars(owner):
                raise TypeError('Cannot resolve the CompactBody state owner')
            seen.add(id(owner)); owner = vars(owner)['parent']
        existing = vars(owner).get('_body_execution')
        if existing is self:
            return owner
        if existing is not None:
            raise ValueError('Reference already has an execution owner; reuse it or create a new reference')
        if type(owner.body).__name__ == 'TorchCompactBody':
            raise ValueError('Reference body is already a Torch adapter without this execution owner')
        if self.config.backend == 'cuda':
            from .compact_body_torch import TorchCompactBody
            if isinstance(owner.body, TorchCompactBody):
                raise ValueError('Body already has a CUDA backend; reuse its execution owner')
            owner.body = TorchCompactBody(owner.body, device='cuda',
                                          batch_size=self.config.gpu_batch_size)
            self._body_backend = owner.body
        owner._body_execution = self
        return owner

    def record(self):
        record = dict(asdict(self.config),
                      derivative='same central local finite differences; no global-control differencing',
                      linear_solver='scipy CPU sparse TRF/LSMR',
                      cpu_parallel_axis='independent local parameter perturbations',
                      cuda_scope='compact SMPL-X FK/LBS only; NumPy boundary remains')
        if self._body_backend is not None:
            record['device'] = self._body_backend.backend_info()
        return record

    def state_differences(self, reference, ref, delta, columns, evaluate_state, *, step=1e-5):
        """Differentiate a per-sample state consumer without changing its model.

        CPU uses independent threads; CUDA combines +/- columns into one world
        batch. The consumer always sees the original number/order of samples.
        The batch cap limits the number of columns, not an individual window;
        TorchCompactBody additionally chunks every device operation.
        """
        delta = np.asarray(delta, dtype=np.float64)
        columns = list(map(int, columns))
        if delta.ndim != 2 or delta.shape[1] != 69 or not np.isfinite(delta).all():
            raise ValueError('Expected finite (N,69) body increments')
        if not np.isfinite(step) or step <= 0:
            raise ValueError('Finite-difference step must be positive')
        if any(column < 0 or column >= 69 for column in columns):
            raise ValueError('Body increment column outside 0..68')
        if not columns:
            return []

        def difference(column):
            plus = delta.copy(); minus = delta.copy()
            plus[:, column] += step; minus[:, column] -= step
            a = evaluate_state(reference.world(ref, plus))
            b = evaluate_state(reference.world(ref, minus))
            return (np.asarray(a)-np.asarray(b))/(2*step)

        if self.config.backend == 'cpu':
            return self.map_columns(difference, columns)

        n = len(delta)
        if n == 0:
            raise ValueError('Cannot batch an empty body window')
        width = min(self.config.perturbation_batch,
                    max(1, self.config.max_batch_samples//(2*n)))
        output = []
        with _BLAS_LIMIT_LOCK, self.limits():
            for first in range(0, len(columns), width):
                group = columns[first:first+width]
                count = 2*len(group)
                trial = np.tile(delta, (count, 1))
                for j, column in enumerate(group):
                    trial[(2*j)*n:(2*j+1)*n, column] += step
                    trial[(2*j+1)*n:(2*j+2)*n, column] -= step
                batched_ref = {}
                for key in ('P', 'Q', 'local_R', 'times'):
                    if key in ref:
                        value = np.asarray(ref[key])
                        if value.shape[0] != n:
                            raise ValueError('Reference sample layout differs: '+key)
                        batched_ref[key] = np.concatenate([value]*count, axis=0)
                state = reference.world(batched_ref, trial)
                for j in range(len(group)):
                    plus = _slice_state(state, 2*j*n, (2*j+1)*n, count*n)
                    minus = _slice_state(state, (2*j+1)*n, (2*j+2)*n, count*n)
                    # Consumers may normalize a view in-place; each state owns
                    # a disjoint batch region and is consumed only once.
                    output.append((np.asarray(evaluate_state(plus))-
                                   np.asarray(evaluate_state(minus)))/(2*step))
        return output

    def world_derivatives(self, reference, ref, delta, columns, *, step=1e-5):
        columns = list(map(int, columns))
        values = self.state_differences(reference, ref, delta, columns,
                                        lambda state: state['world'], step=step)
        return zip(columns, values)


def get_body_execution(reference):
    existing = getattr(reference, '_body_execution', None)
    if existing is not None:
        return existing
    execution = BodyExecution()
    execution.attach(reference)
    return execution
