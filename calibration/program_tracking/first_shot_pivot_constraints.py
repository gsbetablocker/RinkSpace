"""Connect the existing fixed-pivot outside face to the first-shot shared QP.

The original outside-or-on fact and nonlinear guard remain authoritative.
This module maps their one installation and physical working coordinates to
an existing public step-constraint helper; it adds no source/prior rows.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np


def _reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def install_pivot_outside_step_constraints(owner):
    """Wrap only owner.linear_constraints and its checkpoint identity."""
    if hasattr(owner, 'pivot_outside_step_receipt'):
        raise ValueError('Fixed-pivot outside step constraints are already installed')
    from installation_mount_candidate import mount_graph
    from calibration.program_tracking.first_shot_physical_state import INSTALLATION
    state = owner.state
    original = state.original
    if state.count != 7674 or original.count != 5172:
        raise ValueError('Explicit complete first-shot 7674/5172 chart required')
    selected = np.asarray(owner.shared, int)
    if selected.shape != (186,) or not np.array_equal(selected, state.global_shared_indices):
        raise ValueError('Original complete 186-coordinate shared solve order required')
    old_contract = deepcopy(original.pivot_contract)
    relation = old_contract.get('by_installation', {}).get(INSTALLATION, {})
    if relation.get('relation') != 'pivot_outside_or_on_current_actual_board_course':
        raise ValueError('Authenticated existing outside-or-on tripod fact required')
    facts = old_contract['facts']
    if _reference(facts['path'])['sha256'] != facts['sha256']:
        raise ValueError('Authenticated original tripod fact changed')
    scale = float(original.scene_unit_working_m)
    if (scale != old_contract['scene_unit_working_m']
            or scale != float(mount_graph.SCENE_UNIT_M)):
        raise ValueError('Original current-course and pivot working scales differ')
    # Source chart construction is an authenticated one-to-one extraction of
    # the original parent's canonical shared slots. Never reuse slots 5/6/7
    # directly as columns of the current native first-shot state.
    pivot = np.arange(original.installation_slice.start,
                      original.installation_slice.start + 3, dtype=int)
    old_world = np.asarray(old_contract['world_parameter_indices'], int)
    parent_slots = np.asarray(original.parent_shared_slots, int)
    evaluation_slots = np.asarray(original.evaluation_shared_slots, int)
    if (not np.array_equal(pivot, [118, 119, 120])
            or not np.array_equal(evaluation_slots[pivot], relation['pivot_parameter_indices'])
            or not np.array_equal(evaluation_slots[:100], old_world)
            or not all(int(c) in selected for c in np.r_[np.arange(100), pivot[:2]])):
        raise ValueError('Exact original parent world/pivot-to-first-shot slot mapping required')
    contract = deepcopy(old_contract)
    contract['world_parameter_indices'] = list(range(100))
    contract['by_installation'] = {INSTALLATION: dict(deepcopy(relation),
                                                   pivot_parameter_indices=pivot.tolist())}
    provider = original.provider.current
    facade = SimpleNamespace(geometry=SimpleNamespace(current=provider),
                            current_user_pivot_outside_constraint_contract=contract)
    previous_constraints = owner.linear_constraints
    previous_identity = owner.checkpoint_identity
    refs = [_reference(__file__), _reference(mount_graph.__file__)]
    for ref in refs:
        if ref not in owner.recovery_references:
            owner.recovery_references.append(ref)
    receipt = dict(schema='rinkspace.first-shot-original-pivot-outside-QP-bridge.v1',
        producers=refs, facts=deepcopy(facts), relation=relation['relation'],
        installation_id=INSTALLATION,
        original_parent_canonical_pivot_columns=parent_slots[pivot].tolist(),
        original_contract_evaluation_pivot_columns=relation['pivot_parameter_indices'],
        original_contract_evaluation_world_columns=old_world.tolist(),
        canonical_to_evaluation_layout_verified=True,
        complete_first_shot_pivot_columns=pivot.tolist(),
        complete_first_shot_course_world_columns=list(range(100)),
        scene_unit_working_m=scale, selected_shared_columns=selected.tolist(),
        value='original.calibration(original_prefix).pivot_xyz * scene_unit_working_m',
        current_actual_course_provider='original.provider.current.print_course_binding/sample_perimeter',
        reused_helper='installation_mount_candidate.mount_graph.pivot_outside_linear_step_constraints',
        source_r_J_and_objective_unchanged=True, previous_shared_QP_rows_unchanged=True,
        original_full_nonlinear_guard_unchanged=True, outside_or_on_boundary_allowed=True,
        numerical_world_derivative_step=contract['numerical_world_derivative_step'],
        inherited_helper_numeric_active_tangent_policy_unchanged=True,
        numerical_resolution_is_not_an_added_physical_clearance=True,
        added_margin_buffer_prior_or_numeric_bound=False,
        existing_camera_mount_photo_window_not_reintroduced=True,
        extra_PP_axis_a_and_ice_transverse5_outside_derivatives_zero=True,
        camera_optical_centre_not_used_as_tripod_pivot=True,
        only_original_user_outside_relation_connected_to_existing_reduced_QP=True)

    def check():
        for ref in refs + [facts]:
            if _reference(ref['path'])['sha256'] != ref['sha256']:
                raise ValueError('Frozen original pivot outside bridge producer/fact changed')

    def mapped_domain(q, columns):
        q = state._q(q)
        old, _ = state.split(q)  # Also activate the actual shared-five context.
        if original.provider.current is not provider:
            raise ValueError('Actual first-shot course producer identity changed')
        evaluation = q.copy()
        # Follow the exact original guard's arithmetic, including roundoff.
        evaluation[pivot] = np.asarray(original.calibration(old).pivot_xyz, float) * scale
        domain = mount_graph.pivot_outside_linear_step_constraints(facade, evaluation, columns)
        domain['report']['first_shot_slot_and_unit_bridge'] = deepcopy(receipt)
        domain['report']['complete_q_sha256'] = hashlib.sha256(q.tobytes()).hexdigest()
        domain['report']['actual_guard_pivot_xyz_working_m'] = evaluation[pivot].tolist()
        return domain

    def linear_constraints(adapter, q, columns):
        check()
        if not np.array_equal(columns, selected):
            raise ValueError('Complete original shared coordinate order required')
        previous = previous_constraints(q, columns)
        outside = mapped_domain(q, columns)
        A, b = np.asarray(previous['matrix'], float), np.asarray(previous['lower'], float)
        P, c = np.asarray(outside['matrix'], float), np.asarray(outside['lower'], float)
        if A.shape != (len(b), len(selected)) or P.shape != (1, len(selected)) or c.shape != (1,):
            raise ValueError('Original shared domains or one installation outside face changed shape')
        result = dict(previous)
        result['matrix'] = np.vstack((A, P))
        result['lower'] = np.r_[b, c]
        result['report'] = dict(kind='original_shared_domains_plus_original_fixed_pivot_outside',
            previous=previous.get('report'), pivot_outside=outside['report'],
            original_row_slice=[0, len(b)], pivot_outside_row_slice=[len(b), len(b) + 1],
            original_QP_prefix_byte_exact=True, objective_rows_added=0,
            nonlinear_trial_guard_still_required=True)
        return result

    def identity(adapter):
        result = previous_identity()
        result['original_fixed_pivot_outside_shared_QP_bridge'] = deepcopy(receipt)
        return result

    owner.linear_constraints = MethodType(linear_constraints, owner)
    owner.checkpoint_identity = MethodType(identity, owner)
    owner.pivot_outside_step_receipt = receipt
    return deepcopy(receipt)
