"""Explicit programme-source objective; preserve the retired wide fit as history.

The complete canonical checkpoint and shared venue remain unchanged. The old
independent wide camera is fixed and its pixel factor is excluded. Wide images
serve only as references for visibly clear geometric relationships.
"""
from copy import deepcopy
import math

import numpy as np
from scipy.sparse import csr_matrix, vstack

from .first_shot_source_transition import reference


class ProgrammeSourceObjective:
    def __init__(self, base, *, entry_reference, transition_reference):
        self.base = base
        self.programme = base.programme
        self.parameter_count = int(base.parameter_count)
        self.programme_count = int(base.programme_count)
        self.start = np.asarray(base.start).copy()
        self.factor = base.factor  # Historical camera/clock, retained for export.
        self.retired_camera_columns = np.asarray(self.factor.camera_columns, dtype=np.int64).copy()
        if (len(self.retired_camera_columns) == 0
                or not np.array_equal(self.retired_camera_columns,
                                      np.arange(self.programme_count, self.parameter_count))):
            raise ValueError('Exact independent appended static-camera columns required')
        self.shared = np.asarray(base.shared)[~np.isin(base.shared, self.retired_camera_columns)].copy()
        self.fixed = np.unique(np.r_[base.fixed, self.retired_camera_columns]).astype(np.int64)
        self._fixed_bits = self.start[self.fixed].view(np.uint64).copy()
        self.scales = np.asarray(self.programme.scales).copy()
        self.quadratic = np.asarray(self.programme.quadratic).copy()
        n = len(self.scales)
        if (n <= 0 or n >= len(base.scales)
                or not np.array_equal(self.scales, np.asarray(base.scales)[:n])
                or not np.array_equal(self.quadratic, np.asarray(base.quadratic)[:n])
                or len(base.scales)-n != len(self.factor.scales)):
            raise ValueError('Actual unchanged complete programme source prefix required')
        self.receipt = dict(schema='rinkspace.programme-source-wide-reference-only.v1',
            producer=reference(__file__), entry=deepcopy(entry_reference),
            transition=deepcopy(transition_reference),
            programme_formal_rows=n, retired_static_pixel_rows=len(self.factor.scales),
            retired_fixed_camera_columns=self.retired_camera_columns.tolist(),
            source_UV_full_W_PTS_and_complete_canonical_q_unchanged=True,
            wide_role='visibly clear relative rink-line positions; no wide-camera or pixel fit',
            common_rink_geometry_remains_jointly_adjustable=True,
            objective_changed_explicitly=True, old_combined_cost_not_comparable=True)
        self.physical_bounds = deepcopy(getattr(base, 'physical_bounds', {}))
        self.source_producers = deepcopy(base.source_producers)
        self.recovery_references = deepcopy(base.recovery_references)
        self._references = deepcopy(getattr(base, '_references', []))
        for name in ('source_producers', 'recovery_references', '_references'):
            refs = getattr(self, name)
            for ref in (reference(__file__), entry_reference, transition_reference):
                if ref not in refs:
                    refs.append(deepcopy(ref))
        self.state = base.state
        self._initial_complete_J_reuse = False

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _q(self, q):
        q = np.asarray(q)
        if (q.dtype != np.float64 or q.shape != self.start.shape or not np.isfinite(q).all()
                or not np.array_equal(q[self.fixed].view(np.uint64), self._fixed_bits)):
            raise ValueError('Complete finite programme state with all fixed history coordinates required')
        return q

    def guard(self, q):
        q = self._q(q)
        physical, _ = self.base.effective_q_and_chain(q)
        for column, (lower, upper) in self.physical_bounds.items():
            if not lower <= physical[column] <= upper:
                raise ValueError('Effective programme q leaves original physical bound: '+str(column))
        return self.programme.guard(physical[:self.programme_count])

    def source_blocks(self, q, jacobian):
        q = self._q(q)
        cursor = 0
        # Stop at the exact programme boundary, before the retired factor is
        # evaluated. Every programme block retains its original loss and ledger.
        for block in self.base.source_blocks(q, bool(jacobian)):
            stop = cursor+len(block.r)
            if (stop > len(self.scales)
                    or not np.array_equal(block.scales, self.scales[cursor:stop])
                    or not np.array_equal(block.quadratic, self.quadratic[cursor:stop])):
                raise ValueError('Programme source prefix changed or crossed into retired static rows')
            if jacobian and block.J.shape != (len(block.r), self.parameter_count):
                raise ValueError('Complete canonical source Jacobian columns required')
            yield block
            cursor = stop
            if cursor == len(self.scales):
                return
        raise ValueError('Incomplete original programme source blocks')

    def evaluate(self, q, jacobian=False):
        blocks = list(self.source_blocks(q, jacobian))
        r = np.concatenate([block.r for block in blocks])
        return (r, vstack([block.J for block in blocks], format='csr')) if jacobian else r

    def cost(self, q):
        from z190_joint_source_solver import source_cost
        return math.fsum(source_cost(block.r, block.quadratic, block.loss_weights)
                         for block in self.source_blocks(q, False))

    def raw_programme(self, q, **kwargs):
        return self.base.raw_programme(self._q(q), **kwargs)

    def raw(self, q, **kwargs):
        return self.raw_programme(q, **kwargs)

    def linear_constraints(self, q, selected):
        if not np.array_equal(selected, self.shared):
            raise ValueError('Exact active programme shared-column order required')
        record = self.base.linear_constraints(self._q(q), self.base.shared.copy())
        indices = np.flatnonzero(~np.isin(self.base.shared, self.retired_camera_columns))
        matrix = csr_matrix(record['matrix'])[:, indices].tocsr()
        return dict(record, matrix=matrix, report=dict(record.get('report', {}),
            fixed_static_camera_columns_eliminated=self.retired_camera_columns.tolist(),
            all_existing_physical_rows_and_current_margins_retained=True))

    def recover(self, parent, proposed):
        result = self.base.recover(self._q(parent), self._q(proposed))
        self._q(result['q'])
        return result

    def checkpoint_arrays(self):
        result = self.base.checkpoint_arrays()
        result['programme_only_fixed_static_columns'] = self.retired_camera_columns.copy()
        return result

    def solver_linear_constraints(self, q, selected):
        # The legacy active-QP API still calls np.asarray on its input. Keep
        # the owner domain sparse and bridge only that exact delivery boundary.
        result = self.linear_constraints(q, selected)
        return dict(result, matrix=result['matrix'].toarray())

    def checkpoint_identity(self):
        result = deepcopy(self.base.checkpoint_identity())
        result.update(programme_source_objective=deepcopy(self.receipt),
            fixed_parameter_indices=self.fixed.tolist(), global_shared_indices=self.shared.tolist(),
            shared_dense_count=len(self.shared), source_producers=deepcopy(self.source_producers))
        return result

    def complete_state_payload(self, q):
        result = self.base.complete_state_payload(self._q(q))
        result.update({'aux__'+key:value for key, value in self.checkpoint_arrays().items()})
        return result

    def solver_kwargs(self):
        result = self.base.solver_kwargs()
        result.update(source_block_factory=self.source_blocks,
            fixed_parameter_indices=self.fixed.copy(), global_schur_count=len(self.shared),
            global_schur_indices=self.shared.copy(), shared_linear_constraints=self.solver_linear_constraints,
            trial_feasibility_recovery=self.recover,
            checkpoint_arrays=self.checkpoint_arrays(), checkpoint_identity=self.checkpoint_identity())
        return result
