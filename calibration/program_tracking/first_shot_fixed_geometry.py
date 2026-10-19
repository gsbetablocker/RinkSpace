"""Freeze explicitly verified world coordinates on the existing solver chart.

The caller supplies the already prepared complete start and its fixed values.
Original half-spaces are evaluated on that complete state before pulling their
step matrix back to free shared coordinates. No prior, box or geometry value is
invented; original source equations and complete-state dimensions remain.
"""
from copy import deepcopy
import hashlib
from types import MethodType

import numpy as np

from .official_static_surface_dataset import reference


def _indices(values, count, label):
    raw = np.asarray(values)
    result = np.asarray(values, dtype=int).reshape(-1)
    if (raw.ndim != 1 or not np.array_equal(raw, result)
            or len(np.unique(result)) != len(result)
            or np.any(result < 0) or np.any(result >= count)):
        raise ValueError('Distinct complete-state ' + label + ' indices required')
    return result


def install_fixed_geometry(owner, *, fixed_parameter_values, fixed_parameter_indices=None,
                           native_pp_values=None, entry_reference=None, value_reference=None):
    """Install last, after original domain/recovery and execution wrappers.

    Default: freeze every world column, including print layouts and any added
    geometry deviations. To retain print-layout fitting, the caller must supply
    an explicit subset; this helper does not choose a scene or parameter values.
    """
    if hasattr(owner, 'fixed_geometry_receipt'):
        raise ValueError('Verified fixed geometry is already installed')
    start = np.asarray(owner.start)
    if start.ndim != 1 or start.dtype != np.float64 or not np.isfinite(start).all():
        raise ValueError('Finite complete float64 prepared start required')
    world = _indices(owner.state.world_parameter_indices, len(start), 'world')
    columns = world.copy() if fixed_parameter_indices is None else _indices(
        fixed_parameter_indices, len(start), 'fixed geometry')
    if not len(columns) or not np.isin(columns, world).all():
        raise ValueError('Only explicit nonempty world-geometry subsets may be fixed')
    values = np.array(fixed_parameter_values, dtype=np.float64, copy=True)
    if values.shape != columns.shape or not np.isfinite(values).all():
        raise ValueError('One explicit finite value per fixed geometry column required')
    geometry_columns = columns.copy()
    pp_columns = np.empty(0, int)
    if native_pp_values is not None:
        pp_columns = _indices(np.concatenate([owner.state.pp_columns(k)
            for k in range(owner.state.frame_count)]), len(start), 'native principal-point')
        pp_values = np.asarray(native_pp_values, dtype=np.float64)
        if pp_values.shape != (owner.state.frame_count, 2) or not np.isfinite(pp_values).all() or np.any(pp_values != 0.):
            raise ValueError('Withdrawal requires explicit all-zero native principal offsets')
        if np.isin(pp_columns, world).any():
            raise ValueError('Native principal-point and world charts overlap')
        columns = np.r_[columns, pp_columns]
        values = np.r_[values, pp_values.ravel()]
    expected = values.view(np.uint64).copy()
    def fixed_state(q, label):
        q = np.asarray(q)
        if q.shape != start.shape or q.dtype != np.float64 or not np.isfinite(q).all():
            raise ValueError('Finite complete float64 ' + label + ' required')
        if not np.array_equal(np.ascontiguousarray(q[columns]).view(np.uint64), expected):
            raise ValueError('Declared fixed geometry changed in ' + label)
        return q
    fixed_state(start, 'prepared start')
    shared = _indices(owner.shared, len(start), 'shared')
    if not np.isin(geometry_columns, shared).all() or np.isin(pp_columns, shared).any():
        raise ValueError('Every fixed world column must belong to the original shared chart')
    original_kwargs, original_identity = owner.solver_kwargs, owner.checkpoint_identity
    original_guard, original_recover = owner.guard, owner.recover
    original_constraints = owner.linear_constraints
    refs = [reference(__file__)]
    for ref in (entry_reference, value_reference):
        if ref is not None and ref not in refs:
            refs.append(deepcopy(ref))
    for ref in refs:
        if reference(ref['path']) != {k: ref[k] for k in ('path', 'sha256')}:
            raise ValueError('Verified fixed geometry reference changed')
    receipt = dict(schema='rinkspace.explicit-fixed-world-geometry.v1',
        producers=deepcopy(refs), fixed_parameter_indices=columns.tolist(),
        fixed_parameter_values=values.tolist(),
        fixed_geometry_indices=geometry_columns.tolist(),
        withdrawn_native_principal_offset_indices=pp_columns.tolist(),
        native_principal_offsets_restored_to_zero=bool(len(pp_columns)),
        fixed_values_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
        original_world_parameter_indices=world.tolist(),
        all_world_geometry_fixed=bool(np.array_equal(np.sort(geometry_columns), np.sort(world))),
        original_shared_parameter_indices=shared.tolist(),
        geometry_initializer_selected_or_changed=False, complete_state_slots_preserved=True,
        original_constraints_evaluated_at_complete_current_state=True,
        fixed_coordinate_current_margin_preserved=True,
        original_constraint_rows_lower_bounds_and_nonmatrix_payload_preserved=True,
        no_new_geometry_bounds_priors_or_source_rows=True,
        original_gauge_field_and_arrays_preserved=True,
        original_recovery_must_preserve_fixed_values_without_clipping=True)
    unused_legacy_metric = (getattr(owner, '_unmeasured_mount_domain', {}).get(
        'conditional_source_metric_may_still_be_built_but_never_projects_to_photo_box') is True)
    receipt['original_full_shared_inspection_for_unused_released_mount_metric'] = unused_legacy_metric
    def check():
        for ref in refs:
            if reference(ref['path']) != {k: ref[k] for k in ('path', 'sha256')}:
                raise ValueError('Frozen fixed geometry producer/value reference changed')
    def constraints(adapter, q, selected):
        check(); fixed_state(q, 'half-space state')
        selected = _indices(selected, len(start), 'effective shared')
        allowed = shared[~np.isin(shared, columns)]
        inspection = unused_legacy_metric and np.array_equal(selected, shared)
        if not inspection and not np.array_equal(selected, allowed):
            raise ValueError('Exact original free shared order required after fixed geometry elimination')
        original = original_constraints(q, shared.copy())
        matrix = original['matrix']
        if getattr(matrix, 'shape', None) != (len(original['lower']), len(shared)):
            raise ValueError('Original complete shared half-space dimensions changed')
        # The historical metric still inspects its original chart after yielding
        # J. The authenticated released-domain recover never applies that metric;
        # actual public QP/step/recovery calls retain exact fixed elimination.
        locations = np.arange(len(shared)) if inspection else np.flatnonzero(~np.isin(shared, columns))
        result = dict(original, matrix=matrix[:, locations])
        result['report'] = dict(original.get('report', {}),
            fixed_geometry_tangent_pullback=dict(original_shared=shared.tolist(),
                effective_shared=selected.tolist(), fixed_geometry=columns.tolist(),
                original_complete_state_margin_and_row_order_preserved=True,
                unused_legacy_metric_full_chart_inspection=inspection,
                actual_public_solver_fixed_elimination=not inspection))
        return result
    def guard(adapter, q):
        check(); fixed_state(q, 'physical guard')
        return original_guard(q)
    def recover(adapter, parent, proposed):
        check(); fixed_state(parent, 'recovery parent'); fixed_state(proposed, 'recovery proposal')
        result = original_recover(parent, proposed)
        fixed_state(result['q'], 'recovery result')
        result = dict(result)
        result['report'] = dict(result.get('report', {}),
            verified_fixed_geometry_preserved=True, fixed_geometry_clipped_after_recovery=False)
        return result
    def identity(adapter):
        result = original_identity()
        result['verified_fixed_geometry'] = deepcopy(receipt)
        return result
    def kwargs(adapter):
        check(); fixed_state(owner.start, 'solver start')
        result = original_kwargs()
        if (not np.array_equal(result.get('global_schur_indices'), shared)
                or result.get('global_schur_count') != len(shared)):
            raise ValueError('Original complete shared solver chart changed')
        previous_fixed = _indices(result.get('fixed_parameter_indices', ()), len(start), 'original fixed')
        # Current programme gauge columns are outside shared. An already held
        # shared column would need its own explicit pullback, not silent omission.
        if np.isin(previous_fixed, shared).any():
            raise ValueError('Install fixed geometry before any other shared-coordinate restriction')
        result['fixed_parameter_indices'] = np.r_[previous_fixed, columns[~np.isin(columns, previous_fixed)]]
        result['shared_linear_constraints'] = owner.linear_constraints
        result['trial_feasibility_recovery'] = owner.recover
        result['checkpoint_identity']['verified_fixed_geometry'] = deepcopy(receipt)
        return result
    owner.linear_constraints = MethodType(constraints, owner)
    owner.guard, owner.recover = MethodType(guard, owner), MethodType(recover, owner)
    owner.checkpoint_identity, owner.solver_kwargs = MethodType(identity, owner), MethodType(kwargs, owner)
    for ref in refs:
        for name in ('source_producers', 'recovery_references'):
            if hasattr(owner, name) and ref not in getattr(owner, name):
                getattr(owner, name).append(deepcopy(ref))
    owner.fixed_geometry_receipt = deepcopy(receipt)
    return deepcopy(receipt)

def scene_camera_parameter_groups(state):
    """Read the actual chart: core scene, print layout, deviations and native PP."""
    original = state.original
    world = _indices(state.world_parameter_indices, state.count, 'world')
    print_groups = original.print_layout_global_indices_by_ad.values()
    prints = _indices(np.concatenate(tuple(print_groups)), state.count, 'print layout')
    deviations = _indices(state.deviation_indices, state.count, 'geometry deviation')
    original_world = _indices(original.world_parameter_indices, state.count, 'original world')
    base = original_world[~np.isin(original_world, prints)]
    pp = _indices(np.concatenate([state.pp_columns(k) for k in range(state.frame_count)]),
        state.count, 'native principal-point')
    if (not np.array_equal(np.sort(world), np.sort(np.r_[base, prints, deviations]))
            or len(np.unique(np.r_[world, pp])) != len(world) + len(pp)
            or len(prints) != sum(len(v) for v in original.print_layout_global_indices_by_ad.values())):
        raise ValueError('Original world/print/deviation/PP chart does not partition exactly')
    return dict(base_scene_geometry=base, print_layout=prints,
        added_geometry_deviations=deviations, native_principal_offsets=pp)
