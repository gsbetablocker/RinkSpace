"""Opt-in canonical/physical chart around an unchanged static venue stage.

The physical stage owns every source operator, worker and physical guard.
This wrapper maps all evaluations to qe and pulls derivatives back to canonical
coordinates. It neither creates an owner nor installs itself in an old run.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix, hstack, issparse, vstack


def _reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class DerivedVenueState:
    """Expose full canonical-state entry points without changing physical state."""
    def __init__(self, owner):
        self._owner = owner
        self.count = owner.parameter_count
        self.initial_q = owner.start.copy()
        self.world_parameter_indices = owner.physical_stage.state.world_parameter_indices.copy()
        self.world_parameter_count = len(self.world_parameter_indices)
        self.global_shared_indices = owner.shared.copy()

    def _q(self, q):
        return self._owner._canonical_q(q)

    def __getattr__(self, name):
        value = getattr(self._owner.physical_stage.state, name)
        if not callable(value):
            return value
        def call(*args, **kwargs):
            if args and isinstance(args[0], np.ndarray):
                if args[0].shape != (self.count,):
                    raise ValueError('Derived venue state calls require the complete canonical q')
                physical, _ = self._owner.effective_q_and_chain(args[0])
                args = (physical, *args[1:])
            return value(*args, **kwargs)
        return call


class DerivedVenueStage:
    """Wrap one newly prepared physical stage; never mutate its methods/start.

    inherited_aux is the complete parent's auxiliary checkpoint mapping with
    the ``aux__`` prefix removed. Historical auxiliary arrays are kept exactly;
    in particular a newly decoded static camera must not replace the parent's
    historical static-camera initializer array.
    """
    def __init__(self, physical_stage, chart, *, canonical_start, inherited_aux,
                 entry_reference, transition_reference=None, producer_references=()):
        start = np.asarray(canonical_start)
        if (start.dtype != np.float64 or start.shape != (physical_stage.parameter_count,)
                or not np.isfinite(start).all() or chart.count != len(start)):
            raise ValueError('Exact complete finite canonical float64 parent required')
        self.physical_stage, self.chart = physical_stage, chart
        self.programme = physical_stage.programme
        self.parameter_count = len(start)
        self.programme_count = physical_stage.programme_count
        self.start = start.copy()
        self.physical_shared = np.asarray(physical_stage.shared, int).copy()
        self.raw_derived = np.asarray(chart.fixed_canonical_rows, int).copy()
        if (len(set(self.raw_derived)) != len(self.raw_derived)
                or not np.isin(self.raw_derived, self.physical_shared).all()):
            raise ValueError('Every distinct raw derived row must belong to original shared coordinates')
        self.shared = self.physical_shared[~np.isin(self.physical_shared, self.raw_derived)]
        old_fixed = np.asarray(physical_stage.fixed, int)
        self.fixed = np.r_[old_fixed, self.raw_derived[~np.isin(self.raw_derived, old_fixed)]]
        self._fixed_bits = start[self.fixed].view(np.uint64).copy()
        self.scales = physical_stage.scales
        self.quadratic = physical_stage.quadratic
        self.source_producers = deepcopy(physical_stage.source_producers)
        self.recovery_references = deepcopy(physical_stage.recovery_references)
        refs = [_reference(__file__), _reference(sys.modules[type(chart).__module__].__file__),
                deepcopy(chart.recipe_reference), deepcopy(entry_reference)]
        if transition_reference is not None:
            refs.append(deepcopy(transition_reference))
        refs.extend(deepcopy(tuple(producer_references)))
        self._references = []
        for ref in refs:
            if ref not in self._references:
                self._references.append(ref)
            if ref not in self.source_producers:
                self.source_producers.append(deepcopy(ref))
        self._check_producers()
        expected = set(physical_stage.checkpoint_arrays())
        if not expected <= set(inherited_aux):
            raise ValueError('Complete inherited auxiliary arrays required: ' + str(sorted(expected-set(inherited_aux))))
        self._inherited_aux = {}
        for key, value in inherited_aux.items():
            array = np.asarray(value)
            if key.startswith('aux__') or array.dtype.hasobject:
                raise ValueError('Safe inherited auxiliary names/arrays required')
            self._inherited_aux[key] = array.copy()
        self._original_kwargs = physical_stage.solver_kwargs()
        self.physical_bounds = {int(k): tuple(map(float, v)) for k, v in
                                self._original_kwargs.get('shared_parameter_bounds', {}).items()}
        for column, interval in self.physical_bounds.items():
            if (column not in self.physical_shared or len(interval) != 2
                    or not np.isfinite(interval).all() or not interval[0] < interval[1]):
                raise ValueError('Original finite physical shared bounds required')
        self.state = DerivedVenueState(self)
        qe, _ = self.effective_q_and_chain(self.start)
        self.stage_receipt = dict(schema='rinkspace.derived-programme-static-venue-stage.v1',
            chart=self.chart_identity(), producers=deepcopy(self._references),
            canonical_initial_q_sha256=_sha(self.start), physical_initial_q_sha256=_sha(qe),
            canonical_parent_all_parameters_byte_exact=True,
            original_physical_shared_indices=self.physical_shared.tolist(),
            effective_shared_indices=self.shared.tolist(), raw_derived_fixed_indices=self.raw_derived.tolist(),
            fixed_parameter_indices=self.fixed.tolist(),
            physical_absolute_bounds={str(k): list(v) for k, v in self.physical_bounds.items()},
            all_absolute_bounds_converted_to_current_physical_halfspaces=True,
            solver_raw_absolute_bounds_empty=True,
            inherited_auxiliary_arrays={k:dict(dtype=v.dtype.str,shape=list(v.shape),sha256=_sha(v))
                                        for k,v in self._inherited_aux.items()},
            source_workers_receive_physical_q=True, source_worker_q_not_claimed_canonical=True,
            scene_guard_source_constraints_and_review_use_same_physical_q=True,
            source_J_right_chain_and_constraint_M_right_chain_required=True,
            raw_derived_columns_true_solver_fixed_elimination=True,
            canonical_checkpoint_and_physical_evaluation_identities_distinct=True,
            initial_r_J_normal_or_old_objective_cache_reused=False,
            pure_common_scale_gauge_proved=False, installed_in_old_stage=False)
        self._initial_complete_J_reuse = False

    def __getattr__(self, name):
        return getattr(self.physical_stage, name)

    def _check_producers(self):
        for ref in self._references:
            if _reference(ref['path']) != {k: ref[k] for k in ('path', 'sha256')}:
                raise ValueError('Frozen derived venue producer/reference changed')

    def _canonical_q(self, q):
        q = np.asarray(q)
        if q.dtype != np.float64 or q.shape != self.start.shape or not np.isfinite(q).all():
            raise ValueError('Complete finite canonical float64 q required')
        if not np.array_equal(np.ascontiguousarray(q[self.fixed]).view(np.uint64), self._fixed_bits):
            raise ValueError('Fixed raw reference/gauge coordinates changed')
        return q

    def effective_q_and_chain(self, q):
        self._check_producers()
        return self.chart.effective_q_and_chain(self._canonical_q(q))

    def chart_identity(self):
        result = deepcopy(self.chart.identity())
        result['prototype_installed_in_running_model_claim'] = result.get('installed_in_running_model')
        result.update(installed_in_running_model=True, installed_in_this_explicit_stage=True,
                      old_physical_stage_not_monkeypatched=True)
        return result

    def split_state(self, q):
        qe, _ = self.effective_q_and_chain(q)
        return self.physical_stage.split_state(qe)

    def state_identity(self, q):
        qe, _ = self.effective_q_and_chain(q)
        return dict(schema='rinkspace.canonical-and-effective-venue-state.v1',
            canonical_q_sha256=_sha(q), physical_q_sha256=_sha(qe),
            chart=self.chart_identity(), source_scene_guard_q_basis='physical_effective_q',
            solver_checkpoint_q_basis='canonical_q', physical_q_is_not_an_accepted_canonical_checkpoint=True)

    def guard(self, q):
        qe, _ = self.effective_q_and_chain(q)
        for column, (lower, upper) in self.physical_bounds.items():
            if not lower <= qe[column] <= upper:
                raise ValueError('Effective q leaves original physical bound: ' + str(column))
        return self.physical_stage.guard(qe)

    def source_blocks(self, q, jacobian):
        qe, chain = self.effective_q_and_chain(q)
        for block in self.physical_stage.source_blocks(qe, bool(jacobian)):
            jac = None
            if jacobian:
                if block.J.shape[1] != self.parameter_count:
                    raise ValueError('Physical complete source Jacobian columns changed')
                jac = (csr_matrix(block.J) @ chain).tocsr()
                jac.eliminate_zeros()
            # Source rows, sigma, W, integration weights and clocks retain the
            # original objects; only the parameter derivative chart changes.
            yield type(block)(block.r, jac, block.quadratic, block.scales,
                              block.row_ledger, block.loss_weights)

    def evaluate(self, q, jacobian=False):
        blocks = list(self.source_blocks(q, jacobian))
        residual = np.concatenate([block.r for block in blocks])
        return (residual, vstack([block.J for block in blocks], format='csr')) if jacobian else residual

    def world(self, q, jacobian=False):
        qe, chain = self.effective_q_and_chain(q)
        worlds, derivatives = self.physical_stage.world(qe, bool(jacobian))
        if not jacobian:
            return worlds, derivatives
        columns = self.state.world_parameter_indices
        tangent = chain[columns]
        outside = np.setdiff1d(np.arange(self.parameter_count), columns)
        if tangent[:, outside].nnz:
            raise ValueError('World chart acquired a non-world canonical dependency')
        tangent = tangent[:, columns].toarray()
        mapped = {key: np.einsum('...i,ij->...j', value, tangent)
                  for key, value in derivatives.items()}
        return worlds, mapped

    def raw_programme(self, q, **kwargs):
        qe, _ = self.effective_q_and_chain(q)
        return self.physical_stage.raw_programme(qe, **kwargs)

    def raw_static(self, q):
        qe, _ = self.effective_q_and_chain(q)
        return self.physical_stage.raw_static(qe)

    def raw(self, q, **kwargs):
        qe, _ = self.effective_q_and_chain(q)
        return self.physical_stage.raw(qe, **kwargs)

    def cost(self, q):
        qe, _ = self.effective_q_and_chain(q)
        return self.physical_stage.cost(qe)

    def linear_constraints(self, q, selected):
        if not np.array_equal(selected, self.shared):
            raise ValueError('Exact effective shared coordinate order required')
        qe, _ = self.effective_q_and_chain(q)
        original = self.physical_stage.linear_constraints(qe, self.physical_shared.copy())
        matrix = original['matrix']
        if matrix.shape != (len(original['lower']), len(self.physical_shared)):
            raise ValueError('Complete physical halfspace dimensions changed')
        rows, lower, bound_rows = [], [], []
        lookup = {int(c): i for i, c in enumerate(self.physical_shared)}
        for column, (lo, hi) in sorted(self.physical_bounds.items()):
            row = np.zeros(len(self.physical_shared)); row[lookup[column]] = 1.
            rows.extend((row, -row)); lower.extend((lo-qe[column], qe[column]-hi))
            bound_rows.append(dict(physical_column=column, lower=lo, upper=hi, current_value=float(qe[column])))
        augmented = dict(original)
        if rows:
            augmented['matrix'] = (vstack((csr_matrix(matrix), csr_matrix(rows)), format='csr')
                                   if issparse(matrix) else np.vstack((matrix, rows)))
            augmented['lower'] = np.r_[original['lower'], lower]
        augmented['report'] = dict(original.get('report', {}),
            original_physical_halfspace_row_slice=[0, len(original['lower'])],
            original_absolute_bounds_as_current_halfspaces=bound_rows,
            physical_bound_units_and_endpoints_unchanged=True)
        return self.chart.pullback_constraints(augmented, self._canonical_q(q),
            self.physical_shared, canonical_columns=self.shared, fixed_columns=self.fixed)

    def recover(self, parent, proposed):
        old, _ = self.effective_q_and_chain(parent)
        trial, _ = self.effective_q_and_chain(proposed)
        result = self.physical_stage.recover(old, trial)
        recovered = np.asarray(result['q'])
        if recovered.shape != self.start.shape or recovered.dtype != np.float64 or not np.isfinite(recovered).all():
            raise ValueError('Original recovery returned an invalid physical state')
        value = self._canonical_q(proposed).copy()
        identity = np.ones(self.parameter_count, bool); identity[self.raw_derived] = False
        value[identity] = recovered[identity]
        physical, _ = self.effective_q_and_chain(value)
        if not np.array_equal(physical.view(np.uint64), recovered.view(np.uint64)):
            raise ValueError('Original physical recovery is incompatible with the derived chart; no inverse or clipping applied')
        return dict(result, q=value, report=dict(result.get('report', {}),
            canonical_recovery_identity_rows_only=True,
            recovered_physical_state_matches_rederived_qe_byte_exact=True,
            derived_coordinates_not_inverse_solved_or_clipped=True))

    def checkpoint_arrays(self):
        qe, _ = self.effective_q_and_chain(self.start)
        return dict({k:v.copy() for k,v in self._inherited_aux.items()},
            derived_chart_canonical_initial_q=self.start.copy(),
            derived_chart_physical_initial_q=qe,
            derived_chart_raw_fixed_columns=self.raw_derived.copy(),
            derived_chart_released_columns=np.asarray(self.chart.released_rows, int),
            derived_chart_identity_json=np.asarray(json.dumps(self.chart_identity(), sort_keys=True)))

    def complete_state_payload(self, q):
        qe, _ = self.effective_q_and_chain(q)
        result = dict(self.physical_stage.complete_state_payload(qe))
        if 'qbase' in result:
            result['physical_qbase'] = np.asarray(result['qbase']).copy()
            result['qbase'] = np.asarray(q)[:len(result['physical_qbase'])].copy()
        result.update(q_augmented=self._canonical_q(q).copy(), physical_q_augmented=qe.copy(),
            canonical_programme_q_augmented=np.asarray(q)[:self.programme_count].copy(),
            programme_q_augmented=np.asarray(q)[:self.programme_count].copy(),
            physical_programme_q_augmented=qe[:self.programme_count].copy(),
            canonical_q_sha256=np.asarray(_sha(q)), physical_q_sha256=np.asarray(_sha(qe)),
            source_scene_guard_q_basis=np.asarray('physical_effective_q'),
            main_q_augmented_qbase_and_programme_prefix_basis=np.asarray('canonical_q'),
            canonical_physical_state_identity_json=np.asarray(json.dumps(self.state_identity(q), sort_keys=True)))
        result.update({'aux__'+k:v for k,v in self.checkpoint_arrays().items()})
        return result

    def checkpoint_identity(self):
        result = deepcopy(self.physical_stage.checkpoint_identity())
        result.update(schema='rinkspace.canonical-derived-programme-static-venue-state.v1',
            derived_venue_stage=deepcopy(self.stage_receipt),
            fixed_parameter_indices=self.fixed.tolist(), shared_dense_count=len(self.shared),
            source_producers=deepcopy(self.source_producers),
            initial_exact_complete_current_state_full_J_reuse=False)
        return result

    def solver_kwargs(self):
        result = dict(self._original_kwargs)
        result.update(source_block_factory=self.source_blocks,
            fixed_parameter_indices=self.fixed.copy(), global_schur_count=len(self.shared),
            global_schur_indices=self.shared.copy(), shared_parameter_bounds={},
            shared_linear_constraints=self.linear_constraints, trial_feasibility_recovery=self.recover,
            checkpoint_arrays=self.checkpoint_arrays(), checkpoint_identity=self.checkpoint_identity())
        return result

    def close(self):
        return self.physical_stage.close()
