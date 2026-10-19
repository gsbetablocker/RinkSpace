"""Explicit constant-PP subspace of the unchanged complete derived venue model."""
from copy import deepcopy
import json
import numpy as np

from .first_shot_derived_venue_stage import DerivedVenueStage, DerivedVenueState, _reference

FIXED_PP_ZOOM_COLUMNS = np.array([5168, 5169, 7675, 7676], dtype=int)


class ConstantPrincipalVenueStage(DerivedVenueStage):
    """Keep shared unknown PP0; fix only its four zoom slopes to exact zero."""
    def __init__(self, base_derived_owner, *, entry_reference, optical_policy_reference):
        base = base_derived_owner
        if (not isinstance(base, DerivedVenueStage) or base.parameter_count != 7700
                or len(base.shared) != 176 or len(base.fixed) != 3780):
            raise ValueError('Complete current 7700/176/3780 derived venue stage required')
        columns = FIXED_PP_ZOOM_COLUMNS.copy()
        if (not np.isin(columns, base.shared).all() or np.isin(columns, base.fixed).any()
                or not np.isin([124, 125], base.shared).all()
                or not np.array_equal(base.start[columns].view(np.uint64), np.zeros(4).view(np.uint64))):
            raise ValueError('Explicit zero PP-zoom slopes and free shared constant PP required')
        # The physical owner, workers, source factory and chart remain the same.
        self.__dict__.update(base.__dict__)
        self.base = base
        self.start = base.start.copy()
        self._base_shared = base.shared.copy()
        self._shared_positions = np.flatnonzero(~np.isin(base.shared, columns))
        self.shared = base.shared[self._shared_positions].copy()
        self.fixed = np.r_[base.fixed, columns]
        self._fixed_bits = self.start[self.fixed].view(np.uint64).copy()
        self.source_producers = deepcopy(base.source_producers)
        self._references = deepcopy(base._references)
        refs = [_reference(__file__), deepcopy(entry_reference), deepcopy(optical_policy_reference)]
        for ref in refs:
            if ref not in self._references:
                self._references.append(ref)
            if ref not in self.source_producers:
                self.source_producers.append(deepcopy(ref))
        self._check_producers()
        self.constant_principal_receipt = dict(
            schema='rinkspace.constant-principal-complete-venue-subspace.v1',
            producers=refs, optical_policy=deepcopy(optical_policy_reference),
            fixed_PP_zoom_columns=columns.tolist(), free_shared_constant_PP_columns=[124, 125],
            native_PP_fixed_zero_count=2496, third_pose_fixed_zero_count=1248,
            parameter_count=7700, shared_dense_count=172, fixed_parameter_count=3784,
            complete_parent_q_preserved_byte_exact=True,
            all_inherited_auxiliary_arrays_preserved=True,
            original_world_camera_wide_source_and_halfspaces_preserved=True,
            original_model_equations_modified=False,
            fixed_PP_columns_may_have_nonzero_full_J=True,
            fixed_PP_steps_eliminated_by_original_public_solver=True,
            new_PP_or_Brown_numeric_bound_added=False,
            old_source_r_J_cache_reused=False)
        self.stage_receipt = deepcopy(base.stage_receipt)
        self.stage_receipt.update(effective_shared_indices=self.shared.tolist(),
                                  fixed_parameter_indices=self.fixed.tolist(),
                                  additional_constant_principal_policy=deepcopy(self.constant_principal_receipt))
        self.state = DerivedVenueState(self)

    def guard(self, q):
        self._check_producers()
        return self.base.guard(self._canonical_q(q))

    def source_blocks(self, q, jacobian):
        self._check_producers()
        yield from self.base.source_blocks(self._canonical_q(q), bool(jacobian))

    def linear_constraints(self, q, selected):
        if not np.array_equal(selected, self.shared):
            raise ValueError('Exact constant-principal shared coordinate order required')
        original = self.base.linear_constraints(self._canonical_q(q), self._base_shared.copy())
        if original['matrix'].shape != (len(original['lower']), len(self._base_shared)):
            raise ValueError('Original full derived physical halfspaces changed')
        return dict(original, matrix=original['matrix'][:, self._shared_positions])

    def recover(self, parent, proposed):
        result = self.base.recover(self._canonical_q(parent), self._canonical_q(proposed))
        self._canonical_q(result['q'])
        return result

    def checkpoint_arrays(self):
        result = super().checkpoint_arrays()
        result.update(constant_principal_fixed_PP_zoom_columns=FIXED_PP_ZOOM_COLUMNS.copy(),
                      constant_principal_initial_q=self.start.copy(),
                      constant_principal_identity_json=np.asarray(json.dumps(
                          self.constant_principal_receipt, ensure_ascii=False, sort_keys=True)))
        return result

    def checkpoint_identity(self):
        result = super().checkpoint_identity()
        result.update(constant_principal_point_stage=deepcopy(self.constant_principal_receipt),
                      global_shared_indices=self.shared.tolist())
        return result

    def solver_kwargs(self):
        result = dict(self.base.solver_kwargs())
        selected = set(map(int, self.shared))
        result['shared_parameter_bounds'] = {int(k): v for k, v in
            result.get('shared_parameter_bounds', {}).items() if int(k) in selected}
        for name in ('positive_parameter_indices', 'conditional_view_protected_parameter_indices'):
            if name in result:
                result[name] = np.asarray([int(k) for k in result[name] if int(k) in selected], int)
        result.update(source_block_factory=self.source_blocks,
                      fixed_parameter_indices=self.fixed.copy(), global_schur_count=len(self.shared),
                      global_schur_indices=self.shared.copy(), shared_linear_constraints=self.linear_constraints,
                      trial_feasibility_recovery=self.recover,
                      checkpoint_arrays=self.checkpoint_arrays(), checkpoint_identity=self.checkpoint_identity())
        return result
