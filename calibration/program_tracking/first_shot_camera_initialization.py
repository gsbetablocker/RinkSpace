"""Restricted camera initialization on one existing full source owner.

All source values, clocks, W and RS operators are unchanged. The temporary
Jacobian holds every world and independent wide-camera coordinate fixed. Its
zero world derivative arrays are local source-worker inputs, never a complete
world Jacobian or an entry in the programme's true derivative cache.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

from .first_shot_source_transition import reference, read


def _sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _install_execution(execution, head_class):
    producer, qa = execution['producer'], execution['QA']
    if reference(producer['path']) != producer or reference(qa['path']) != qa:
        raise ValueError('Camera initialization execution references changed')
    if read(qa).get('passed') is not True:
        raise ValueError('Explicit same-value installer needs passed QA')
    name = '_rinkspace_camera_initializer_execution_' + producer['sha256']
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, producer['path'])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    module = sys.modules[name]
    if 'expected_math_reference' in execution:
        return module.install(expected_math_reference=execution['expected_math_reference'], head_class=head_class)
    return getattr(module, execution.get('function', 'install_execution'))()


def camera_source_chunk(task):
    """Original persistent worker sampler; optional explicit same-value installer.

    An execution option is a separately hashed producer with a passed QA and
    a no-argument install_execution() hook. No option is enabled implicitly.
    The original five-component source task itself is forwarded unchanged.
    """
    execution = None
    if len(task) == 2 and isinstance(task[1], dict):
        task, execution = task
    if execution is not None:
        from . import first_shot_fit as fit
        _install_execution(execution, type(fit._source_worker[0].fixed_head))
    from native_transverse_bundle import source_chunk
    return source_chunk(task)


class _RestrictedState:
    def __init__(self, stage):
        self._stage = stage
        self.count = stage.parameter_count
        self.initial_q = stage.start.copy()
        self.global_shared_indices = stage.shared.copy()
        self.world_parameter_indices = stage.programme.state.world_parameter_indices.copy()
        self.world_parameter_count = len(self.world_parameter_indices)

    def _q(self, q):
        return self._stage._q(q)

    def __getattr__(self, name):
        return getattr(self._stage.base.state, name)


class CameraInitializationStage:
    """One full 7700-vector and original source, with a temporary camera active set."""
    def __init__(self, base_derived_owner, *, entry_reference, initializer_reference):
        self.base = base_derived_owner
        self.programme = self.base.programme
        self.factor = self.base.factor
        self.parameter_count = self.base.parameter_count
        self.programme_count = self.base.programme_count
        self.start = self.base.start.copy()
        if self.start.dtype != np.float64 or self.start.shape != (7700,) or self.programme_count != 7687:
            raise ValueError('The explicit complete fixed-head 7687 plus wide13 chart is required')
        world = np.asarray(self.programme.state.world_parameter_indices, int)
        wide = np.asarray(self.factor.camera_columns, int)
        if len(world) != 171 or len(wide) != 13:
            raise ValueError('The complete original world171 and independent wide13 must be retained')
        self.fixed = np.unique(np.r_[self.base.fixed, world, wide]).astype(int)
        self._base_shared = np.asarray(self.base.shared, int).copy()
        self.shared = self._base_shared[~np.isin(self._base_shared, self.fixed)]
        self._shared_positions = np.flatnonzero(~np.isin(self._base_shared, self.fixed))
        if len(self.base.fixed) != 3780 or len(self._base_shared) != 176 or len(self.fixed) != 3928 or len(self.shared) != 28:
            raise ValueError('Explicit camera initialization active-set dimensions changed')
        self._fixed_bits = self.start[self.fixed].view(np.uint64).copy()
        self.scales, self.quadratic = self.base.scales, self.base.quadratic
        self.source_producers = deepcopy(self.base.source_producers)
        self.recovery_references = deepcopy(self.base.recovery_references)
        self._references = [reference(__file__), deepcopy(entry_reference), deepcopy(initializer_reference)]
        self.execution = None
        config = read(initializer_reference)
        if 'cache_installer_producer' in config or 'cache_installer_QA' in config:
            self.execution = dict(producer=deepcopy(config['cache_installer_producer']),
                QA=deepcopy(config['cache_installer_QA']), function=config.get('cache_installer_function', 'install_execution'))
            if 'cache_installer_math_reference' in config:
                self.execution['expected_math_reference'] = deepcopy(config['cache_installer_math_reference'])
                self._references.append(self.execution['expected_math_reference'])
            self._references.extend((self.execution['producer'], self.execution['QA']))
            if read(self.execution['QA']).get('passed') is not True:
                raise ValueError('Explicit camera execution QA has not passed')
        for ref in self._references:
            if reference(ref['path']) != ref:
                raise ValueError('Camera initialization producer/configuration changed')
            if ref not in self.source_producers:
                self.source_producers.append(deepcopy(ref))
        self.state = _RestrictedState(self)
        self._initial_complete_J_reuse = False
        original_kwargs = self.base.solver_kwargs()
        self._solver_scope = dict(
            original_shared_parameter_bounds={str(k):list(v) for k,v in original_kwargs.get('shared_parameter_bounds', {}).items()},
            original_positive_parameter_indices=list(map(int, original_kwargs.get('positive_parameter_indices', ()))),
            original_conditional_view_protected_parameter_indices=list(map(int, original_kwargs.get('conditional_view_protected_parameter_indices', ()))),
            conditional_view_backtracking=bool(original_kwargs.get('conditional_view_backtracking', False)))
        self.receipt = dict(schema='rinkspace.full-state-camera-only-initialization.v1',
            producer=reference(__file__), entry=deepcopy(entry_reference), initializer=deepcopy(initializer_reference),
            parameter_count=self.parameter_count, programme_parameter_count=self.programme_count,
            original_fixed_count=len(self.base.fixed), fixed_count=len(self.fixed), shared_count=len(self.shared),
            world_fixed_columns=world.tolist(), wide_fixed_columns=wide.tolist(),
            fixed_parameter_indices=self.fixed.tolist(), shared_parameter_indices=self.shared.tolist(),
            all_original_source_values_W_sigma_RS_loss_weights_and_ledgers_unchanged=True,
            restricted_camera_J_not_complete_world_J=True, full_parameter_width_preserved=True,
            world_derivatives_local_zero171_never_cached=True, original_persistent_source_pool_reused=True,
            fixed_coordinate_recovery_changes_rejected_not_clipped=True,
            optional_same_value_execution=deepcopy(self.execution), original_owner_and_geometry_source_not_reconstructed=True,
            actual_base_solver_scope=deepcopy(self._solver_scope),
            original_complete_halfspaces_and_lower_retained=True)
        if self.execution is not None:
            if 'actual_cache_QA_output' not in config:
                raise ValueError('Explicit cache requires actual same-owner six-frame QA output')
            self.receipt['actual_same_owner_cache_QA'] = self.certify_execution(config['actual_cache_QA_output'])
            _install_execution(self.execution, type(self.programme.state.fixed_head))

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _q(self, q):
        q = np.asarray(q)
        if q.dtype != np.float64 or q.shape != self.start.shape or not np.isfinite(q).all():
            raise ValueError('Complete finite float64 camera-initializer state required')
        if not np.array_equal(q[self.fixed].view(np.uint64), self._fixed_bits):
            raise ValueError('Camera initialization changed a fixed world/wide/pose coordinate')
        return q

    def effective_q_and_chain(self, q):
        return self.base.effective_q_and_chain(self._q(q))

    def guard(self, q):
        return self.base.guard(self._q(q))

    def source_blocks(self, q, jacobian):
        q = self._q(q)
        if not jacobian:
            yield from self.base.source_blocks(q, False)
            return
        for ref in self._references:
            if reference(ref['path']) != ref:
                raise ValueError('Frozen camera initialization producer/configuration changed')
        from z190_streaming_source_normal import SourceRowBlock
        self.base.guard(q)
        physical, chain = self.effective_q_and_chain(q)
        prefix = physical[:self.programme_count]
        programme = self.programme
        worlds, _ = programme.world(prefix, False)
        world_count = len(programme.state.world_parameter_indices)
        zero = {key: np.zeros(value.shape + (world_count,)) for key, value in worlds.items()}
        chunks = [list(map(int, group)) for group in
            np.array_split(np.arange(len(programme.bundle.frames)), programme.workers) if len(group)]
        expected = _sha(prefix)
        tasks = [(prefix, worlds, zero, indices, expected) for indices in chunks]
        if self.execution is not None:
            tasks = [(task, self.execution) for task in tasks]
        started = time.perf_counter()
        rows = nnz = 0
        for indices, (r, jac, receipt) in zip(chunks,
                programme._pool().map(camera_source_chunk, tasks, chunksize=1), strict=True):
            if (receipt['augmented_q_sha256'] != expected
                    or len(r) != sum(programme.bundle.whitening[i].count for i in indices)
                    or jac.shape != (len(r), self.programme_count)):
                raise ValueError('Restricted camera worker lost the complete original source contract')
            # Exact literal ledger of CompleteTransverseBundle.source_blocks.
            ledger = dict(family='complete-native-PP-two-axis-transverse',
                binding=programme.bundle.state.binding_reference, two_axis_model=programme.model_reference,
                frames=[programme.bundle.ledgers[i] for i in indices])
            scales = np.concatenate([programme.bundle.whitening[i].scales for i in indices])
            mask = np.concatenate([programme.quadratic[programme.formal_offsets[i]:programme.formal_offsets[i+1]]
                                   for i in indices])
            full = (hstack((jac, csr_matrix((len(r), self.parameter_count-self.programme_count))),
                           format='csr') @ chain).tocsr()
            full.eliminate_zeros()
            fixed = full[:, self.fixed].copy(); fixed.eliminate_zeros()
            if fixed.nnz:
                raise ValueError('Restricted camera J has a nonzero fixed coordinate derivative')
            programme.worker_receipts.append(receipt)
            rows += len(r); nnz += full.nnz
            yield SourceRowBlock(r, full, mask, scales, ledger)
        if rows != len(programme.scales):
            raise ValueError('Restricted camera J missed original programme source rows')
        programme.evaluation_receipts.append(dict(qaug_sha256=expected, jacobian=True,
            source_parameter_chart='restricted camera initialization; not full world J',
            full_parameter_columns=self.programme_count, world_columns=world_count,
            source_rows=rows, J_nonzeros=nnz, source_frames=len(programme.bundle.frames),
            world_values_recomputed_with_original_public_factory=True,
            world_derivatives_local_zero_not_cached=True, elapsed_seconds=time.perf_counter()-started))
        block = self.factor.block(physical, False)
        yield SourceRowBlock(block.r, csr_matrix((len(block.r), self.parameter_count)),
                             block.quadratic, block.scales, block.row_ledger, block.loss_weights)

    def evaluate(self, q, jacobian=False):
        blocks = list(self.source_blocks(q, bool(jacobian)))
        residual = np.concatenate([block.r for block in blocks])
        return (residual, vstack([block.J for block in blocks], format='csr')) if jacobian else residual

    def linear_constraints(self, q, selected):
        self._q(q)
        if not np.array_equal(selected, self.shared):
            raise ValueError('Exact restricted shared-camera coordinate order required')
        original = self.base.linear_constraints(q, self._base_shared.copy())
        if original['matrix'].shape != (len(original['lower']), len(self._base_shared)):
            raise ValueError('Original complete physical halfspace dimensions changed')
        return dict(original, matrix=original['matrix'][:, self._shared_positions])

    def recover(self, parent, proposed):
        result = self.base.recover(self._q(parent), self._q(proposed))
        self._q(result['q'])
        return result

    def raw_programme(self, q, **kwargs):
        return self.base.raw_programme(self._q(q), **kwargs)

    def raw_static(self, q):
        return self.base.raw_static(self._q(q))

    def raw(self, q, **kwargs):
        return self.base.raw(self._q(q), **kwargs)

    def cost(self, q):
        from z190_joint_source_solver import source_cost
        return math.fsum(source_cost(b.r, b.quadratic, b.loss_weights) for b in self.source_blocks(q, False))

    def certify_execution(self, output):
        """Compare six actual full-W camera J blocks on the existing source pool."""
        if self.programme.pool is None:
            raise ValueError('Cache validation requires the existing source pool, never a replacement')
        output = Path(output)
        if output.exists():
            raise ValueError('Actual cache QA must not overwrite previous source evidence')
        from native_transverse_bundle import source_chunk
        physical, _ = self.effective_q_and_chain(self.start)
        prefix = physical[:self.programme_count]
        worlds, _ = self.programme.world(prefix, False)
        zero = {key: np.zeros(value.shape + (171,)) for key,value in worlds.items()}
        indices = [200,384,588,841,1238,1247]
        if [self.programme.bundle.frames[i]['index'] for i in indices] != [201,385,589,842,1239,1248]:
            raise ValueError('Actual source six-frame native indices changed')
        task = (prefix, worlds, zero, indices, _sha(prefix))
        old = next(self.programme._pool().map(source_chunk, [task], chunksize=1))
        current = next(self.programme._pool().map(camera_source_chunk, [(task,self.execution)], chunksize=1))
        def identical(a,b,label):
            if np.asarray(a).dtype != np.asarray(b).dtype or np.asarray(a).shape != np.asarray(b).shape or np.asarray(a).tobytes() != np.asarray(b).tobytes():
                raise ValueError('Actual installation cache changed '+label)
        identical(old[0],current[0],'source residual')
        if old[1].shape != current[1].shape:
            raise ValueError('Actual installation cache changed full camera J shape')
        for name in ('data','indices','indptr'):
            identical(getattr(old[1],name),getattr(current[1],name),'CSR '+name)
        contract = ('augmented_q_sha256','old_q_sha256','PP_sha256','axis_a_working_m',
            'quadratic_b_working_per_logf_squared','frames_0based','world_derivatives_received',
            'backend','full_parameter_count','source_sampler')
        for key in contract:
            if (key in old[2]) != (key in current[2]) or old[2].get(key) != current[2].get(key):
                raise ValueError('Actual cache source worker contract changed: '+key)
        report = dict(schema='rinkspace.camera-initializer-actual-six-source-cache-QA.v1',passed=True,
            producer=reference(__file__),execution=deepcopy(self.execution),q_sha256=_sha(prefix),
            frames_0based=indices,source_rows=len(old[0]),J_shape=list(old[1].shape),J_nonzeros=old[1].nnz,
            original_worker_pid=old[2].get('pid'),cached_worker_pid=current[2].get('pid'),
            r_dtype_shape_and_bytes_exact=True,CSR_data_indices_indptr_shape_and_bytes_exact=True,
            original_source_worker_contract_exact=True,original_persistent_pool_reused=True,
            world_values_only_local_zero171_not_in_real_Dcache=True,parent_cache_not_installed_before_comparison=True,
            full1248_shadow_or_new_owner_or_pool_constructed=False,
            source_contract={key:deepcopy(old[2][key]) for key in contract if key in old[2]})
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
        return reference(output)

    def checkpoint_arrays(self):
        return dict(self.base.checkpoint_arrays(),
            camera_initializer_fixed_columns=self.fixed.copy(),
            camera_initializer_shared_columns=self.shared.copy(),
            camera_initializer_parent_q=self.start.copy(),
            camera_initializer_identity_json=np.asarray(json.dumps(self.receipt, sort_keys=True)))

    def complete_state_payload(self, q):
        result = dict(self.base.complete_state_payload(self._q(q)))
        result.update({'aux__'+key: value for key, value in self.checkpoint_arrays().items()})
        return result

    def checkpoint_identity(self):
        result = deepcopy(self.base.checkpoint_identity())
        result.update(camera_initialization_stage=deepcopy(self.receipt),
            fixed_parameter_indices=self.fixed.tolist(), shared_dense_count=len(self.shared),
            global_shared_indices=self.shared.tolist(), source_producers=deepcopy(self.source_producers),
            initial_exact_complete_current_state_full_J_reuse=False)
        return result

    def solver_kwargs(self):
        result = dict(self.base.solver_kwargs())
        selected = set(map(int,self.shared))
        # Fixed world bounds remain enforced by base.guard and by the complete
        # original halfspaces with their unchanged lower values. The public
        # solver accepts indexed bounds/positivity/protection only in its
        # selected shared coordinates, so retain only the active camera subset.
        result['shared_parameter_bounds'] = {int(k):v for k,v in
            result.get('shared_parameter_bounds',{}).items() if int(k) in selected}
        for name in ('positive_parameter_indices','conditional_view_protected_parameter_indices'):
            if name in result:
                result[name] = np.asarray([int(k) for k in result[name] if int(k) in selected],int)
        result.update(global_schur_count=len(self.shared), global_schur_indices=self.shared.copy(),
            fixed_parameter_indices=self.fixed.copy(), source_block_factory=self.source_blocks,
            shared_linear_constraints=self.linear_constraints, trial_feasibility_recovery=self.recover,
            checkpoint_arrays=self.checkpoint_arrays(), checkpoint_identity=self.checkpoint_identity())
        return result

    def close(self):
        return self.base.close()
