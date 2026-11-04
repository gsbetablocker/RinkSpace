"""Explicit execution policy for the existing, unchanged body objective.

Independent CPU derivative columns share an immutable FK/LBS linearization.
The SciPy trust region, shared boot parameters and reference state remain
owned by the calling thread. CUDA batches the original perturbations through
reference.world; camera, footwear and sparse TRF/LSMR remain CPU operations.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import copy
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
    local_derivatives: str = 'analytic'
    compatibility_backend: str = 'cpu'

    def __post_init__(self):
        if self.compatibility_backend not in ('cpu', 'cuda_resident'):
            raise ValueError('Compatibility backend must be cpu or cuda_resident')
        if self.compatibility_backend == 'cuda_resident' and (self.backend != 'cpu' or self.local_derivatives != 'analytic'):
            raise ValueError('Resident compatibility is the explicit hybrid CPU analytic/CUDA foot backend')
        if self.backend not in ('cpu', 'cuda'):
            raise ValueError('Body backend must be cpu or cuda')
        if self.local_derivatives not in ('analytic', 'finite_difference'):
            raise ValueError('Body local derivatives must be analytic or finite_difference')
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
                   max_batch_samples=int(os.environ.get(prefix+'MAX_BATCH_SAMPLES', '4096')),
                   local_derivatives=os.environ.get(prefix+'LOCAL_DERIVATIVES', 'analytic').lower(),
                   compatibility_backend=os.environ.get(prefix+'COMPATIBILITY_BACKEND', 'cpu').lower())


def _slice_state(value, start, stop, total):
    # Metadata (e.g. lower_boot_vertices) is not sample-aligned. Only known
    # dynamic fields are needed by callers; leave all other fields untouched.
    sample_keys = {'P', 'Q', 'local_R', 'hand_local_R', 'native', 'coco', 'gear', 'boot_R', 'boot_O',
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
        self._compatibility_backend = None
        self._derivative_usage = dict(analytic_world_linearizations=0,
                                     fk_replay_linearizations=0,
                                     surface_replay_linearizations=0,
                                     finite_difference_world_linearizations=0)
        self._last_world_derivative_mode = 'not_evaluated'
        self._last_surface_replay = None

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

    def prepare_compatibility(self,reference,ref,binding):
        if self.config.compatibility_backend != 'cuda_resident':return None
        if self._compatibility_backend is not None:
            raise ValueError('A resident compatibility execution owner belongs to one immutable window')
        from .body_resident_torch import ResidentFootLinearization
        self._compatibility_backend=ResidentFootLinearization(reference,ref,binding)
        return self._compatibility_backend

    def record(self):
        if self.config.backend == 'cuda':
            derivative = 'CUDA batched central local finite differences; compact FK/LBS only'
        elif self.config.local_derivatives == 'finite_difference':
            derivative = 'original central local finite differences; analytic/replay paths explicitly disabled'
        else:
            derivative = 'audited CPU compact FK/pose-blend/LBS analytic chain; exact +/- FK/surface replay for shoe/skin consumers; unsupported/custom consumers retain central differences'
        record = dict(asdict(self.config),
                      derivative=derivative,
                      linear_solver='scipy CPU sparse TRF/LSMR',
                      cpu_parallel_axis='independent local parameter perturbations',
                      cuda_scope=('resident native55 FK/selected skin/compatibility only; CPU camera, body-image analytic derivatives, CSR and TRF remain'
                          if self.config.compatibility_backend=='cuda_resident' else 'compact SMPL-X FK/LBS only for backend=cuda; NumPy boundary remains'),
                      derivative_usage=dict(self._derivative_usage),
                      last_world_derivative_mode=self._last_world_derivative_mode,
                      last_surface_replay=self._last_surface_replay)
        if self._compatibility_backend is not None:
            record['compatibility_device'] = self._compatibility_backend.record()
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
                for key in ('P', 'Q', 'local_R', 'hand_local_R', 'times'):
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

    def _analytic_reference(self, reference):
        if self.config.backend != 'cpu' or self.config.local_derivatives != 'analytic':
            return None
        # Avoid importing body/runtime modules for arbitrary consumer fixtures.
        # In particular, do not unwrap TorchCompactBody into a CPU implementation.
        if type(getattr(reference, 'body', None)).__name__ != 'CompactBody':
            return None
        from .compact_body_derivatives import resolve_reference
        return resolve_reference(reference)

    @staticmethod
    def _validate_local_inputs(ref, delta, columns, step):
        delta = np.asarray(delta, dtype=np.float64)
        if delta.ndim != 2 or delta.shape[1] != 69 or not np.isfinite(delta).all():
            raise ValueError('Expected finite (N,69) body increments')
        if not np.isfinite(step) or step <= 0:
            raise ValueError('Finite-difference step must be positive')
        if any(column < 6 or column >= 69 for column in columns):
            raise ValueError('Expected local body columns in 6..68')
        if np.asarray(ref['P']).shape != (len(delta), 3) or np.asarray(ref['Q']).shape != (len(delta), 3, 3):
            raise ValueError('Reference sample layout differs from body increments')
        return delta

    def native_state_differences(self, reference, ref, delta, columns, evaluate_native, *, step=1e-5,
                                 surface=None, vertex_ids=None):
        """Replay exact +/- native FK without unrelated COCO/equipment work.

        ``evaluate_native(native_local, frames, samples=None)`` must evaluate every
        existing skin/pose-blend/factor dependency.  This only removes the
        compact model and world equipment work that such a consumer discards.
        A supported ``surface`` replays its selected vertices from one frozen
        skinning context and passes them as the third argument.  With no such
        context, the two-argument callback performs its original sampling.
        Return ``None`` for unsupported/custom references, explicit old-FD mode
        or CUDA; callers then keep their original ``state_differences`` path.
        """
        owner = self._analytic_reference(reference)
        if owner is None:
            return None
        columns = list(map(int, columns))
        delta = self._validate_local_inputs(ref, delta, columns, step)
        if not columns:
            return []
        from .compact_body_derivatives import CompactBodyLinearization, _matrices
        with _BLAS_LIMIT_LOCK, self.limits():
            linearization = CompactBodyLinearization(owner.body, ref['local_R'],
                delta[:, 6:].reshape(-1, 21, 3), include_vertices=False,hand_local_R=ref.get('hand_local_R'))
            P = np.asarray(ref['P']) + delta[:, :3]
            Q = np.asarray(ref['Q']) @ _matrices(delta[:, 3:6])

            def body_local(native):
                world = P[:, None] + np.einsum('nij,nkj->nki', Q, native)
                return np.einsum('nji,nkj->nki', Q, world - P[:, None])

            surface_context = None
            if surface is not None:
                from .selected_surface_linearization import SelectedSurfaceLinearization, supports_surface
                # FullSurface normally owns a separately loaded copy of the
                # same compact model.  Object identity is unnecessarily strict,
                # while accepting a different skeleton/scale would be wrong.
                surface_body = getattr(surface, 'body', None)
                same_model = (surface_body is not None and getattr(surface_body, 'scale', None) == owner.body.scale
                    and all(np.array_equal(getattr(surface_body, key, None), getattr(owner.body, key))
                            for key in ('J', 'parents', 'default')))
                if same_model and supports_surface(surface):
                    surface_context = SelectedSurfaceLinearization(surface,
                        body_local(linearization.native), linearization.frames, vertex_ids,hand_local_R=ref.get('hand_local_R'))
                    self._last_surface_replay = surface_context.record()

            def difference(column):
                pair = linearization.kinematic_pair(column, step)
                for state in pair:
                    # Retain the production world->local arithmetic.  Large
                    # root coordinates otherwise change tiny difference noise
                    # in thresholded cavity/collar rows.
                    state['native'] = body_local(state['native'])
                samples = ((None, None) if surface_context is None else
                           surface_context.pair(column, pair[0], pair[1]))
                values = []
                for state, sample in zip(pair, samples):
                    value = (evaluate_native(state['native'], state['frames']) if sample is None else
                             evaluate_native(state['native'], state['frames'], sample))
                    values.append(np.asarray(value))
                return (values[0] - values[1]) / (2 * step)

            values = self.map_columns(difference, columns)
        self._derivative_usage['fk_replay_linearizations'] += 1
        if surface_context is not None:
            self._derivative_usage['surface_replay_linearizations'] += 1
        return values

    def world_derivatives(self, reference, ref, delta, columns, *, step=1e-5):
        columns = list(map(int, columns))
        owner = self._analytic_reference(reference)
        if owner is None or any(c < 6 or c >= 69 for c in columns):
            self._last_world_derivative_mode = ('cuda_central_difference' if self.config.backend == 'cuda' else
                'explicit_finite_difference' if self.config.local_derivatives == 'finite_difference' else
                'unsupported_reference_or_columns_central_difference')
            values = self.state_differences(reference, ref, delta, columns,
                                            lambda state: state['world'], step=step)
            self._derivative_usage['finite_difference_world_linearizations'] += 1
            return zip(columns, values)
        delta = self._validate_local_inputs(ref, delta, columns, step)
        if not columns:
            return iter(())
        from .compact_body_derivatives import CompactBodyLinearization, _matrices, uses_binding
        from .footwear_geometry import equipment_world
        with _BLAS_LIMIT_LOCK, self.limits():
            linearization = CompactBodyLinearization(owner.body, ref['local_R'],
                delta[:, 6:].reshape(-1, 21, 3),hand_local_R=ref.get('hand_local_R'))
            P = np.asarray(ref['P']) + delta[:, :3]
            Q = np.asarray(ref['Q']) @ _matrices(delta[:, 3:6])
            binding = owner.binding if uses_binding(owner) else None
            count = (getattr(getattr(binding, 'layout', None), 'equipment_count', 96)
                     if binding is not None else 96)

            def derivative(column):
                # Body COCO includes all pose blend-shape terms, even for a
                # joint that is outside both leg ancestor trees.
                coco = np.einsum('nij,nkj->nki', Q,
                    linearization.derivative(column, kinematics=False)['coco'])
                if not linearization.affects_joints(column, (4, 5, 7, 8)):
                    gear = np.zeros((len(delta), count, 3), dtype=np.float64)
                else:
                    pair = linearization.kinematic_pair(column, step)
                    if binding is None:
                        trial = [equipment_world(P, Q, state['native'], state['frames'], owner.geometry)[0]
                                 for state in pair]
                    else:
                        # Each worker owns the mutable template cache.  The
                        # production binding's nonsmooth deformation/selection
                        # remains evaluated at the exact two perturbed states.
                        local_binding = copy.copy(binding)
                        local_binding.x = binding.x.copy()
                        trial = []
                        for state in pair:
                            world = P[:, None] + np.einsum('nij,nkj->nki', Q, state['native'])
                            native = np.einsum('nji,nkj->nki', Q, world - P[:, None])
                            trial.append(local_binding.evaluate(P, Q, native, state['frames'])['gear'])
                    gear = ((trial[0] - trial[1]) / (2 * step)).reshape(len(delta), count, 3)
                return np.concatenate((gear, coco), axis=1)

            values = self.map_columns(derivative, columns)
        self._derivative_usage['analytic_world_linearizations'] += 1
        self._last_world_derivative_mode = 'cpu_analytic_coco_exact_fk_gear_difference'
        return zip(columns, values)


def get_body_execution(reference):
    existing = getattr(reference, '_body_execution', None)
    if existing is not None:
        return existing
    execution = BodyExecution()
    execution.attach(reference)
    return execution
