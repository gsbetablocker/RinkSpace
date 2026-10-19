"""Differentiate the current uniform scale in existing mount halfspaces.

This is an explicit solver callback adapter. It changes no camera, source,
geometry, bound, recovery or objective. Pose-dependent support is still frozen
in the shared-only QP and is checked by the existing nonlinear recovery/guard.
"""
from copy import deepcopy
from types import MethodType

import numpy as np
from scipy.sparse import csr_matrix, issparse, vstack

from .first_shot_derived_inactive import PhysicalCourseChart, DerivedInactiveChart
from .first_shot_source_transition import reference


def corrected_record(record, mount, absolute_lower, log_scale_gradient):
    """Add b/s * d(log s) to d(Mv-b/s), preserving every current margin."""
    matrix = record['matrix']
    prefix = csr_matrix(matrix)[:len(absolute_lower)]
    original = csr_matrix(mount['matrix'])
    if prefix.shape != original.shape or (prefix-original).nnz:
        raise ValueError('Actual unchanged mount prefix required before chart pullback')
    lower = np.asarray(record['lower'])
    if not np.array_equal(lower[:len(absolute_lower)], mount['lower']):
        raise ValueError('Actual unchanged current mount margins required')
    gradient = np.asarray(log_scale_gradient, dtype=np.float64)
    absolute_lower = np.asarray(absolute_lower, dtype=np.float64)
    if gradient.shape != (matrix.shape[1],) or not np.isfinite(gradient).all():
        raise ValueError('Complete finite selected scale gradient required')
    correction = csr_matrix(absolute_lower[:, None]*gradient[None, :])
    updated = vstack((prefix+correction, csr_matrix(matrix)[len(absolute_lower):]), format='csr')
    updated.eliminate_zeros()
    return dict(record, matrix=updated if issparse(matrix) else updated.toarray(),
        report=dict(record.get('report', {}), mount_uniform_scale_tangent=dict(
            producer=reference(__file__), original_mount_row_slice=[0, len(absolute_lower)],
            formula='d(Mv-b/s)=M dv+(b/s) d(log s), with current pose support',
            lower_and_all_existing_bounds_unchanged=True,
            pose_support_still_current_state_only=True,
            original_nonlinear_recovery_guard_and_true_source_cost_required=True,
            new_source_prior_bounds_or_parameters=False)))


def install(owner, *, entry_reference):
    """Install once on a newly constructed owner, at its physical callback."""
    programme = owner.programme
    if hasattr(programme, '_mount_scale_tangent_receipt'):
        raise ValueError('Mount scale tangent must be installed exactly once')
    state, model = programme.state, programme.model
    physical, _ = owner.effective_q_and_chain(owner.start)
    course = PhysicalCourseChart(state, physical[:state.count])
    original = programme.linear_constraints

    def constraints(this, q, selected):
        current = original(q, selected)
        mount = model.mount_linear_constraints(state, q, selected)
        # This exact existing producer exposes absolute b/s and current margins.
        from . import first_shot_fixed_head_optics as physical_mount
        _, _, _, _, absolute, actual_scale = physical_mount._physical_halfspaces(state, q)
        values, jacobian = course(q)
        scale, partials = DerivedInactiveChart._scale(*values[3:])
        if not np.isclose(scale, actual_scale, rtol=32*np.finfo(float).eps, atol=0):
            raise ValueError('Analytic actual-course scale differs from actual mount geometry scale')
        full_gradient = (partials[0]*jacobian[3]+partials[1]*jacobian[4])/scale
        selected = np.asarray(selected, dtype=int)
        omitted = np.setdiff1d(np.arange(len(q)), selected)
        if np.any(full_gradient[omitted] != 0):
            raise ValueError('Physical scale has undeclared non-shared dependencies')
        return corrected_record(current, mount, absolute, full_gradient[selected])

    programme.linear_constraints = MethodType(constraints, programme)
    receipt = dict(schema='rinkspace.existing-mount-uniform-scale-tangent.v1',
        producer=reference(__file__), entry=deepcopy(entry_reference),
        source_projection_camera_geometry_and_objective_unchanged=True,
        existing_mount_lower_bounds_and_recovery_unchanged=True,
        correction_before_derived_and_native_time_pullback=True,
        full_pose_dependent_mount_linearization_claimed=False)
    programme._mount_scale_tangent_receipt = deepcopy(receipt)
    # The constraint is evaluated by all existing outer chart adapters.
    # Record this algorithm producer without rewriting historical auxiliary data.
    layer, seen = owner, set()
    while id(layer) not in seen:
        seen.add(id(layer))
        for name in ('source_producers', '_references', 'recovery_references'):
            refs = vars(layer).get(name)
            if refs is not None:
                for ref in (receipt['producer'], entry_reference):
                    if ref not in refs:
                        refs.append(deepcopy(ref))
        if 'base' in vars(layer):
            layer = vars(layer)['base']
        elif 'physical_stage' in vars(layer):
            layer = vars(layer)['physical_stage']
        else:
            break
    checkpoint_identity = owner.checkpoint_identity

    def identity(this):
        result = checkpoint_identity()
        result['mount_uniform_scale_tangent'] = deepcopy(receipt)
        return result

    owner.checkpoint_identity = MethodType(identity, owner)
    return receipt
