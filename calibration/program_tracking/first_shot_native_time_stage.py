"""Opt-in linear native-PTS pan/tilt/logf subspace of a complete venue stage.

The supplied basis is evaluated at native exposures only. The physical owner
still uses its existing native-angle interpolation during rolling readout.
No knots, source observations, motion limits, or smoothing factors are chosen.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, eye, vstack

from .first_shot_derived_venue_stage import DerivedVenueState, _reference


SCHEMA = 'rinkspace.native-pan-tilt-logf-time-basis.v1'
KINDS = ('pan', 'tilt', 'logf')


def _sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _basis(value):
    if getattr(value, 'dtype', None) != np.dtype(np.float64):
        raise ValueError('Explicit float64 native time basis required')
    matrix = csr_matrix(value, copy=True)
    matrix.sum_duplicates(); matrix.sort_indices(); matrix.eliminate_zeros()
    if (matrix.ndim != 2 or min(matrix.shape) < 1 or matrix.shape[1] > matrix.shape[0]
            or not np.isfinite(matrix.data).all()):
        raise ValueError('Finite nonempty native-by-control basis required')
    return matrix


def recipe_contract(basis, native_pts, control_slots, canonical_start, base_parent_q):
    """Return the literal contract an external, separately saved recipe binds.

    Additional explanation/knot provenance may be saved beside these fields.
    This helper neither selects a basis nor transforms an initializer.
    """
    matrix = _basis(basis)
    pts = np.asarray(native_pts)
    start, parent = np.asarray(canonical_start), np.asarray(base_parent_q)
    if (pts.dtype != np.int64 or pts.shape != (matrix.shape[0],)
            or np.any(np.diff(pts) <= 0)):
        raise ValueError('Exact strictly increasing int64 native PTS required')
    if (start.dtype != np.float64 or parent.dtype != np.float64
            or start.ndim != 1 or start.shape != parent.shape
            or not np.isfinite(start).all() or not np.isfinite(parent).all()):
        raise ValueError('Complete finite float64 canonical initializer and parent required')
    if set(control_slots) != set(KINDS):
        raise ValueError('Explicit separate pan/tilt/logf control slots required')
    slots = {}
    for kind in KINDS:
        indices = np.asarray(control_slots[kind])
        if (indices.dtype.kind not in 'iu' or indices.shape != (matrix.shape[1],)
                or len(np.unique(indices)) != len(indices)
                or np.any(indices < 0) or np.any(indices >= len(start))):
            raise ValueError('Distinct in-range integer control slots matching basis width required')
        slots[kind] = indices.astype(np.int64).tolist()
    all_slots = sum((slots[kind] for kind in KINDS), [])
    if len(set(all_slots)) != len(all_slots):
        raise ValueError('The three native control types must have disjoint slots')
    return dict(schema=SCHEMA,
        basis=dict(format='csr', dtype='float64', shape=list(matrix.shape),
            data_sha256=_sha(matrix.data), indices_int64_sha256=_sha(matrix.indices.astype(np.int64)),
            indptr_int64_sha256=_sha(matrix.indptr.astype(np.int64))),
        native_pts=dict(count=len(pts), int64_sha256=_sha(pts)),
        control_slots=slots, canonical_initial_q_sha256=_sha(start),
        base_canonical_parent_q_sha256=_sha(parent))


class NativeTimeVenueStage:
    """Compose T before one unchanged derived/constant-PP/print-QP owner.

    ``canonical_start`` already contains the externally prepared coefficients.
    Its non-control coordinates must be the exact complete parent's values.
    The base owner must be newly constructed; no running owner is modified.
    """
    def __init__(self, base_derived_owner, *, basis, native_pts, control_slots,
                 canonical_start, recipe_reference, entry_reference):
        base = base_derived_owner
        if (base.parameter_count != 7700 or base.programme_count != 7687
                or len(base.shared) != 172 or len(base.fixed) != 3784):
            raise ValueError('Complete current 7700/172/3784 constant-principal venue stage required')
        if getattr(base, 'native_time_receipt', None) is not None:
            raise ValueError('A native time chart cannot be installed twice')
        self.base = base
        self.parameter_count, self.programme_count = base.parameter_count, base.programme_count
        self.programme, self.physical_stage = base.programme, base.physical_stage
        self.factor = base.factor
        physical_state = self.programme.state
        if physical_state.frame_count != 1248:
            raise ValueError('Exact current 1248 native exposure inventory required')
        self.basis = _basis(basis)
        self.native_pts = np.asarray(native_pts).copy()
        self.start = np.asarray(canonical_start).copy()
        contract = recipe_contract(self.basis, self.native_pts, control_slots, self.start, base.start)
        if (self.start.shape != (self.parameter_count,) or self.basis.shape[0] != 1248
                or not _same(self.native_pts, np.asarray(physical_state.pts, np.int64))):
            raise ValueError('Time basis must use the exact programme native PTS and complete q')
        self.control_slots = {kind: np.asarray(contract['control_slots'][kind], np.int64) for kind in KINDS}
        self.native_columns = np.asarray([physical_state.view_columns(i + 1)[[0, 1, 3]]
                                          for i in range(1248)], np.int64)
        if (self.native_columns.shape != (1248, 3)
                or len(np.unique(self.native_columns)) != self.native_columns.size):
            raise ValueError('Distinct actual native pan/tilt/logf columns required')
        native = self.native_columns.ravel()
        self.shared = np.asarray(base.shared, int).copy()
        old_fixed = np.asarray(base.fixed, int).copy()
        if np.isin(native, np.r_[self.shared, old_fixed]).any():
            raise ValueError('Original native PTZ must be free local coordinates')
        controls = np.concatenate([self.control_slots[kind] for kind in KINDS])
        for j, kind in enumerate(KINDS):
            if not np.isin(self.control_slots[kind], self.native_columns[:, j]).all():
                raise ValueError('Every control slot must retain its original '+kind+' type')
        self.fixed_native = native[~np.isin(native, controls)].copy()
        unchanged = np.setdiff1d(np.arange(self.parameter_count), controls)
        if not _same(self.start[unchanged], np.asarray(base.start)[unchanged]):
            raise ValueError('Only explicit control coefficients may differ from the complete parent')
        self.fixed = np.r_[old_fixed, self.fixed_native]
        self._fixed_bits = self.start[self.fixed].view(np.uint64).copy()
        self._native_mask = np.zeros(self.parameter_count, bool); self._native_mask[native] = True
        tangent = eye(self.parameter_count, format='lil', dtype=np.float64)
        for j, kind in enumerate(KINDS):
            for i, column in enumerate(self.native_columns[:, j]):
                span = slice(self.basis.indptr[i], self.basis.indptr[i + 1])
                tangent.rows[column] = self.control_slots[kind][self.basis.indices[span]].tolist()
                tangent.data[column] = self.basis.data[span].tolist()
        self._time_chain = tangent.tocsr(); self._time_chain.eliminate_zeros()
        self.scales, self.quadratic = base.scales, base.quadratic
        self.source_producers = deepcopy(base.source_producers)
        self.recovery_references = deepcopy(base.recovery_references)
        self._references = deepcopy(getattr(base, '_references', []))
        refs = [_reference(__file__), deepcopy(entry_reference), deepcopy(recipe_reference)]
        for ref in refs:
            if _reference(ref['path']) != {k: ref[k] for k in ('path', 'sha256')}:
                raise ValueError('Explicit native time producer/recipe reference changed')
            if ref not in self._references: self._references.append(deepcopy(ref))
            if ref not in self.source_producers: self.source_producers.append(deepcopy(ref))
        recipe = json.loads(Path(recipe_reference['path']).read_text(encoding='utf-8-sig'))
        if any(recipe.get(key) != value for key, value in contract.items()):
            raise ValueError('External recipe does not bind the literal basis/PTS/slots/initial state/parent')
        self.native_time_receipt = dict(schema='rinkspace.complete-native-time-subspace-stage.v1',
            producers=refs, recipe=deepcopy(recipe_reference), contract=contract,
            parameter_count=self.parameter_count, programme_parameter_count=self.programme_count,
            shared_dense_count=len(self.shared), fixed_parameter_count=len(self.fixed),
            native_PTZoom_columns_by_exposure=self.native_columns.tolist(),
            inherited_fixed_parameter_indices=old_fixed.tolist(),
            newly_fixed_native_history_columns=self.fixed_native.tolist(),
            active_native_control_count=len(controls), source_clock=deepcopy(physical_state.source_clock),
            basis_is_evaluated_only_at_native_PTS=True,
            row_time_basis='existing linear native-angle and logf interpolation after T expansion',
            continuous_C2_or_measured_motion_law_claim=False,
            fixed_R2_installation_and_constant_PP_and_all_physical_guards_retained=True,
            geometry_wide_source_UV_W_sigma_rows_and_loss_unchanged=True,
            additional_temporal_source_rows_or_speed_bounds=False,
            original_base_owner_not_modified=True, derived_chart_evaluated_once_after_T=True,
            parent_source_linearization_not_claimed_equivalent=True,
            initializer_prediction_equivalence_to_parent_not_claimed=True)
        self.state = DerivedVenueState(self)
        self._initial_complete_J_reuse = False
        self._check_producers()
        self.expanded_q_and_chain(self.start)

    def __getattr__(self, name):
        value = getattr(self.base, name)
        if callable(value):
            raise AttributeError('Native time stage needs an explicit q adapter for '+name)
        return value

    def _check_producers(self):
        for ref in self._references:
            if _reference(ref['path']) != {k: ref[k] for k in ('path', 'sha256')}:
                raise ValueError('Frozen native time producer/reference changed')

    def _canonical_q(self, q):
        q = np.asarray(q)
        if q.dtype != np.float64 or q.shape != self.start.shape or not np.isfinite(q).all():
            raise ValueError('Complete finite native-control canonical float64 q required')
        if not _same(q[self.fixed].view(np.uint64), self._fixed_bits):
            raise ValueError('Fixed native history/reference/PP coordinates changed')
        return q

    def expanded_q_and_chain(self, q):
        """Return base-canonical T(q), never a second derived-physical q."""
        self._check_producers(); q = self._canonical_q(q); expanded = q.copy()
        for j, kind in enumerate(KINDS):
            expanded[self.native_columns[:, j]] = self.basis @ q[self.control_slots[kind]]
        if not np.isfinite(expanded).all():
            raise ValueError('Native time expansion leaves finite floating domain')
        return expanded, self._time_chain.copy()

    def effective_q_and_chain(self, q):
        expanded, time_chain = self.expanded_q_and_chain(q)
        physical, derived_chain = self.base.effective_q_and_chain(expanded)
        return physical, (csr_matrix(derived_chain) @ time_chain).tocsr()

    def chart_identity(self):
        return dict(schema='rinkspace.native-time-then-existing-derived-chart.v1',
            native_time=deepcopy(self.native_time_receipt), existing_base_chart=self.base.chart_identity(),
            order='T_native_then_base_derived_exactly_once')

    def state_identity(self, q):
        expanded, _ = self.expanded_q_and_chain(q); physical, _ = self.effective_q_and_chain(q)
        return dict(schema='rinkspace.native-control-base-canonical-effective-identities.v1',
            canonical_q_sha256=_sha(q), expanded_base_canonical_q_sha256=_sha(expanded),
            physical_q_sha256=_sha(physical), chart=self.chart_identity(),
            solver_q_basis='native-control canonical', source_scene_guard_q_basis='physical effective')

    def split_state(self, q):
        expanded, _ = self.expanded_q_and_chain(q)
        return self.base.split_state(expanded)

    def guard(self, q):
        expanded, _ = self.expanded_q_and_chain(q)
        return self.base.guard(expanded)

    def source_blocks(self, q, jacobian):
        expanded, chain = self.expanded_q_and_chain(q)
        for block in self.base.source_blocks(expanded, bool(jacobian)):
            jac = None
            if jacobian:
                if block.J.shape != (len(block.r), self.parameter_count):
                    raise ValueError('Complete base source Jacobian dimensions changed')
                jac = (csr_matrix(block.J) @ chain).tocsr(); jac.eliminate_zeros()
            yield type(block)(block.r, jac, block.quadratic, block.scales,
                              block.row_ledger, block.loss_weights)

    def evaluate(self, q, jacobian=False):
        blocks = list(self.source_blocks(q, jacobian))
        residual = np.concatenate([block.r for block in blocks])
        return (residual, vstack([block.J for block in blocks], format='csr')) if jacobian else residual

    def world(self, q, jacobian=False):
        expanded, _ = self.expanded_q_and_chain(q)
        # T changes no world row/column; base already pulls through derived36.
        return self.base.world(expanded, bool(jacobian))

    def raw_programme(self, q, **kwargs):
        expanded, _ = self.expanded_q_and_chain(q)
        return self.base.raw_programme(expanded, **kwargs)

    def raw_static(self, q):
        expanded, _ = self.expanded_q_and_chain(q)
        return self.base.raw_static(expanded)

    def raw(self, q, **kwargs):
        expanded, _ = self.expanded_q_and_chain(q)
        return self.base.raw(expanded, **kwargs)

    def cost(self, q):
        expanded, _ = self.expanded_q_and_chain(q)
        return self.base.cost(expanded)

    def linear_constraints(self, q, selected):
        if not np.array_equal(selected, self.shared):
            raise ValueError('Exact inherited 172 shared-coordinate order required')
        expanded, chain = self.expanded_q_and_chain(q)
        original = self.base.linear_constraints(expanded, self.shared.copy())
        matrix = original['matrix']; sparse = csr_matrix(matrix)
        if sparse.shape != (len(original['lower']), len(self.shared)):
            raise ValueError('Complete original print/mount/geometry halfspace dimensions changed')
        # The original callback contains shared columns only; retain its
        # current native-pose-dependent lower and pull its declared tangent.
        pulled = sparse @ chain[self.shared]
        outside = np.setdiff1d(np.arange(self.parameter_count), self.shared)
        if pulled[:, outside].nnz:
            raise ValueError('Native time chart moved a declared shared constraint into local coordinates')
        result = pulled[:, self.shared].tocsr()
        if (result - sparse).nnz:
            raise ValueError('Native-only time chart changed the inherited shared tangent')
        # Shared rows are exactly identity under T. Return the literal original
        # matrix, retaining dense signed zeros and the original sparse layout.
        return dict(original, matrix=matrix,
            report=dict(original.get('report', {}), native_time_current_pose_expanded=True,
                original_lower_unchanged=True, original_shared_matrix_right_chain=True))

    def recover(self, parent, proposed):
        old, _ = self.expanded_q_and_chain(parent); trial, _ = self.expanded_q_and_chain(proposed)
        result = self.base.recover(old, trial); recovered = np.asarray(result['q'])
        if recovered.dtype != np.float64 or recovered.shape != trial.shape or not np.isfinite(recovered).all():
            raise ValueError('Base recovery returned an invalid complete canonical q')
        value = self._canonical_q(proposed).copy()
        # Controls are never overwritten by expanded native samples. Only
        # coordinates on which T is identity may inherit physical recovery.
        value[~self._native_mask] = recovered[~self._native_mask]
        restored, _ = self.expanded_q_and_chain(value)
        if not _same(restored, recovered):
            raise ValueError('Base recovery is incompatible with the native time subspace; no inverse or clipping applied')
        return dict(result, q=value, report=dict(result.get('report', {}),
            native_controls_and_fixed_native_history_preserved=True,
            expanded_recovery_matches_base_canonical_q_byte_exact=True))

    def checkpoint_arrays(self):
        expanded, _ = self.expanded_q_and_chain(self.start)
        physical, _ = self.base.effective_q_and_chain(expanded)
        result = {key: np.asarray(value).copy() for key, value in self.base.checkpoint_arrays().items()}
        result.update(native_time_canonical_initial_q=self.start.copy(),
            native_time_expanded_base_initial_q=expanded, native_time_physical_initial_q=physical,
            native_time_basis_data=self.basis.data.copy(),
            native_time_basis_indices=self.basis.indices.astype(np.int64),
            native_time_basis_indptr=self.basis.indptr.astype(np.int64),
            native_time_basis_shape=np.asarray(self.basis.shape, np.int64),
            native_time_pts=self.native_pts.copy(), native_time_native_columns=self.native_columns.copy(),
            native_time_fixed_history_columns=self.fixed_native.copy(),
            native_time_control_slots=np.asarray([self.control_slots[k] for k in KINDS], np.int64),
            native_time_identity_json=np.asarray(json.dumps(self.native_time_receipt, ensure_ascii=False, sort_keys=True)))
        return result

    def complete_state_payload(self, q):
        q = self._canonical_q(q); expanded, _ = self.expanded_q_and_chain(q)
        physical, _ = self.base.effective_q_and_chain(expanded)
        result = dict(self.base.complete_state_payload(expanded))
        if 'qbase' in result:
            result['native_time_expanded_base_qbase'] = np.asarray(result['qbase']).copy()
            result['qbase'] = q[:len(result['qbase'])].copy()
        result.update(q_augmented=q.copy(), physical_q_augmented=physical.copy(),
            native_time_expanded_base_canonical_q=expanded.copy(),
            native_time_expanded_base_programme_q=expanded[:self.programme_count].copy(),
            canonical_programme_q_augmented=q[:self.programme_count].copy(),
            programme_q_augmented=q[:self.programme_count].copy(),
            physical_programme_q_augmented=physical[:self.programme_count].copy(),
            canonical_q_sha256=np.asarray(_sha(q)), physical_q_sha256=np.asarray(_sha(physical)),
            source_scene_guard_q_basis=np.asarray('physical_effective_q_after_T_and_base_derived_once'),
            main_q_augmented_qbase_and_programme_prefix_basis=np.asarray('native-control canonical q'),
            canonical_physical_state_identity_json=np.asarray(json.dumps(self.state_identity(q), sort_keys=True)))
        result.update({'aux__'+key: value for key, value in self.checkpoint_arrays().items()})
        return result

    def checkpoint_identity(self):
        result = deepcopy(self.base.checkpoint_identity())
        result.update(schema='rinkspace.complete-native-time-canonical-venue-state.v1',
            native_time_stage=deepcopy(self.native_time_receipt), source_producers=deepcopy(self.source_producers),
            fixed_parameter_indices=self.fixed.tolist(), global_shared_indices=self.shared.tolist(),
            shared_dense_count=len(self.shared), initial_exact_complete_current_state_full_J_reuse=False)
        return result

    def solver_kwargs(self):
        result = dict(self.base.solver_kwargs())
        for name in ('source_physical_endpoint_event', 'shared_step_curvature_tightening',
                     'bounded_motion_model_factory', 'save_current_linearization_cache'):
            if callable(result.get(name)):
                raise ValueError('Native time chart needs an explicit adapter for '+name)
        result.update(source_block_factory=self.source_blocks, fixed_parameter_indices=self.fixed.copy(),
            global_schur_count=len(self.shared), global_schur_indices=self.shared.copy(),
            shared_linear_constraints=self.linear_constraints, trial_feasibility_recovery=self.recover,
            checkpoint_arrays=self.checkpoint_arrays(), checkpoint_identity=self.checkpoint_identity(),
            reuse_source_jacobian=False)
        return result

    def close(self):
        return self.base.close()
