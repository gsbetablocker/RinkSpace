"""Explicit fixed-head angle/optics owner binding; no source constructor.

All original world and material data stay in their original slots. Nine shared
optics stay at 7674:7683; four shared installation axes follow them. Native third
pose slots and native PP slots are inactive fixed zeros.
"""
from copy import deepcopy
from types import MethodType, SimpleNamespace
import numpy as np
from .first_shot_source_transition import reference

def camera_initializer(parent_q,physical_seed,state,*,effective_parent_q):
    """Project the explicit physical seed to one shared fixed-head surface.

    This inverse initialization uses no source UV and is not an accepted fit.
    The actual new-source residual and physical guard are recomputed afterward.
    """
    import cv2
    from scipy.optimize import least_squares
    parent_q=np.asarray(parent_q,float);physical_seed=np.asarray(physical_seed,float)
    if parent_q.shape!=(7687,)or physical_seed.shape!=(7674,):
        raise ValueError('Exact accepted full parent and complete physical seed required')
    world=np.asarray(state.world_parameter_indices,int)
    if not np.array_equal(physical_seed[world].view(np.uint64),np.asarray(effective_parent_q)[:7674][world].view(np.uint64)):
        raise ValueError('Physical initializer changed actual world or print')
    camera=np.setdiff1d(np.arange(7674),world);prefix=parent_q[:7674].copy();prefix[camera]=physical_seed[camera]
    views=np.asarray([physical_seed[state.view_columns(i)]for i in range(1,state.frame_count+1)])
    rotations=np.asarray([cv2.Rodrigues(v[:3])[0]for v in views])
    head=state.fixed_head
    def residual(s):
        ins=head.installation(s)
        return np.einsum('j,mjk,k->m',ins['b'],rotations,ins['n'])
    def derivative(s):
        ins=head.installation(s)
        return (np.einsum('jc,mjk,k->mc',ins['db'],rotations,ins['n'])
            +np.einsum('j,mjk,kc->mc',ins['b'],rotations,ins['dn']))
    fit=least_squares(residual,np.zeros(4),jac=derivative,xtol=1e-12,ftol=1e-12,gtol=1e-12)
    axes=fit.x.copy();inverse=head.inverse_sequence(rotations,axes,project=True);angles=np.asarray(inverse['angles'])
    if angles.shape!=(state.frame_count,2):raise ValueError('Fixed-head inverse changed the native sequence')
    result=np.r_[prefix,np.zeros(9),axes,parent_q[7674:]]
    for i,angle in enumerate(angles,1):
        cols=state.view_columns(i);result[cols[:2]]=angle;result[cols[2]]=0.
    state._q(result[:state.count])
    if not np.array_equal(result[world].view(np.uint64),parent_q[world].view(np.uint64)):
        raise ValueError('Fixed-head initialization changed canonical world')
    if not np.array_equal(result[state.count:].view(np.uint64),parent_q[7674:].view(np.uint64)):
        raise ValueError('Fixed-head initialization changed independent wide camera')
    departure=np.asarray(inverse['signed_normal_departure_radians'])
    return result,dict(schema='rinkspace.explicit-physical-seed-to-fixed-head-initializer.v1',
        accepted_parent_q_sha256=__import__('hashlib').sha256(parent_q.tobytes()).hexdigest(),
        old_programme_count=7674,new_programme_count=state.count,old_wide_columns=list(range(7674,7687)),
        new_wide_columns=list(range(state.count,len(result))),
        physical_camera_initializer_is_not_an_accepted_fit=True,inverse_projection_used=True,
        inverse_projection_changes_predictions_and_requires_fresh_full_source_r_J=True,
        inverse_sequence_native_order_and_unwrapped_angle_branch_preserved=True,
        canonical_world171_and_independent_wide13_byte_exact=True,
        shared_axes_initial=axes.tolist(),shared_axis_only_rotation_fit=dict(nfev=int(fit.nfev),optimality=float(fit.optimality)),
        original_rotation_surface_departure_radians_rms=float(np.sqrt(np.mean(departure**2))),
        original_rotation_surface_departure_radians_max=float(np.abs(departure).max()),
        unobserved_full_turns_not_identifiable_from_rotation_samples=True)


def install(programme,*,model_reference,start,full_parent_aux,entry_reference,transition_reference):
    """Rebind dynamic original samplers and exact public domains before spawn."""
    from native_transverse_bundle import load_model
    from . import first_shot_fixed_head_optics as module
    from rinkspace import metric_scale
    from .first_shot_fit import FrozenWhitening
    from .finite_source_line_coverage import check_native_line_coverage
    from .first_shot_finite_recovery import install_finite_endpoint_recovery
    from .first_shot_physical_state import INSTALLATION
    from installation_mount_candidate import mount_graph
    from print_domain_shared_constraints import merge_mount_and_print
    if any(getattr(obj,name,None)is not None for obj,name in((programme,'pool'),(programme.bundle,'_source_pool'),(programme.bundle,'_geometry_pool'))):
        raise ValueError('Close all old source/geometry pools before camera chart transition')
    if programme.raw_retention_receipt['verified']or programme.raw_retention_receipt['cache_hits']:
        raise ValueError('New camera chart cannot inherit a populated raw replay cache')
    old_identity=deepcopy(programme.checkpoint_identity());old_state=programme.state
    old_aux={k:np.asarray(v).copy()for k,v in full_parent_aux.items()}
    model=load_model(model_reference);state=model.TransverseNativePrincipalState(old_state);start=state._q(start).copy()
    if len(start)!=7687:raise ValueError('Explicit reduced shared optics width required')
    programme.state=state;programme.model=model;programme.model_reference=deepcopy(model_reference)
    programme.start=start;programme.shared=state.global_shared_indices.copy();programme.fixed=state.fixed_camera_parameter_indices.copy()
    programme.bundle.state=model.make_unshifted_world_state(state)
    for owner,name in((programme,'guard_cache'),(programme,'transverse_world_cache'),(programme.bundle,'_world_cache'),(programme.bundle,'_finite_coverage_cache')):
        getattr(owner,name).clear()
    programme._world_delta=None;programme._geometry_pool_delta=None;programme.metric=None
    programme.initial_J_manifest=None;programme._initial_complete_J_reuse=False
    model.install_dispatch()
    producers=[model_reference,reference(__file__),entry_reference,transition_reference,
        reference(module.__file__),reference(module.optics.__file__),
        reference(__import__(module.FixedPanTilt.__module__,fromlist=['x']).__file__),
        reference(module.existing_quadratic.__file__),
        reference(module.existing_quadratic.camera25_from_calibration_driver.__code__.co_filename),
        reference(module.MOUNT_REFERENCE['path']),reference(metric_scale.__file__)]
    for ref in producers:
        if reference(ref['path'])!=ref:raise ValueError('Frozen shared optics producer/fact changed')
        if ref not in programme.source_producers:programme.source_producers.append(deepcopy(ref))
    original=state.original;original.mount_bounds_active=False
    programme._unmeasured_mount_domain=dict(
        schema='rinkspace.fixed-head-current-physical-recovery-historical-attribute-compatibility.v1',
        historical_attribute_compatibility_only=True,
        current_internal_d_and_continuous_total_and_outside_domains_active=True,
        conditional_source_metric_constructed=False,current_recovery_uses_stale_normal=False,
        historical_released_mount_policy_text_does_not_release_current_physical_domains=True)
    pivot=np.arange(118,121);scale=state.scene_unit_working_m
    contract=deepcopy(original.pivot_contract);relation=deepcopy(contract['by_installation'][INSTALLATION])
    if relation['relation']!='pivot_outside_or_on_current_actual_board_course':raise ValueError('Original outside tripod relation required')
    relation['pivot_parameter_indices']=pivot.tolist();contract['world_parameter_indices']=list(range(100));contract['by_installation']={INSTALLATION:relation}
    facade=SimpleNamespace(geometry=SimpleNamespace(current=original.provider.current),
        current_user_pivot_outside_constraint_contract=contract,installation_optics={})
    def evaluation(q):
        q=state._q(q);old,_=state.split(q);value=q.copy()
        value[pivot]=np.asarray(original.calibration(old).pivot_xyz)*scale
        return value
    def guard(adapter,q):
        q=state._q(q)
        if np.any(q[adapter.fixed]!=0.):raise ValueError('Inactive native PP changed')
        key=(q.tobytes(),model_reference['sha256'],id(adapter.bundle.finite_coverage_inventory))
        if key in adapter.guard_cache:adapter.last_guard_report=adapter.guard_cache[key];return True
        physical=state.guard(q);worlds,_=adapter.world(q,False)
        coverage=check_native_line_coverage(state,q,adapter.bundle.finite_coverage_inventory,worlds)
        report=dict(shared_physical_camera=physical,finite_coverage=coverage,native_PP_all_fixed_zero=True,
            old_four_PP_gauge_not_active=True,internal_d_and_total_offset_boxes_both_active=True)
        adapter.guard_cache[key]=report;adapter.last_guard_report=report;adapter.guard_evaluations+=1
        while len(adapter.guard_cache)>2:adapter.guard_cache.popitem(last=False)
        return True
    def constraints(adapter,q,selected):
        if not np.array_equal(selected,adapter.shared):raise ValueError('Complete new shared optics solve order required')
        current=merge_mount_and_print(state,q,selected,model.mount_linear_constraints(state,q,selected))
        outside=mount_graph.pivot_outside_linear_step_constraints(facade,evaluation(q),selected)
        return dict(matrix=np.r_[current['matrix'],outside['matrix']],lower=np.r_[current['lower'],outside['lower']],
            report=dict(kind='shared_physical_optics_mount_print_and_original_outside',
                current_mount_and_print=current.get('report'),original_outside=outside.get('report'),new_prior_or_source_rows=0))
    def recover(adapter,parent,proposed):
        result=model.recover_mount_trial(state,parent,proposed);q=result['q'];before=evaluation(q)
        domain=mount_graph.pivot_outside_domain(facade,before)
        if not domain['valid']:
            after,proof=mount_graph.reinitialize_invalid_pivots_outside(facade,before,require_coordinate_fd_margin=False)
            q=q.copy();q[pivot[:2]]+=after[pivot[:2]]-before[pivot[:2]]
            if not mount_graph.pivot_outside_domain(facade,evaluation(q))['valid']:raise ValueError('Mapped original outside recovery failed')
            result=dict(q=q,report=dict(result['report'],pivot_outside_original_helper=proof,added_clearance=0.))
        return result
    def arrays(adapter):
        result={k:v.copy()for k,v in old_aux.items()if not k.startswith('static_venue_')and k!='programme_parameter_count'}
        for k,v in old_aux.items():result['historical_parent__'+k]=v.copy()
        result.update(native_PP_gauge_fixed_columns=np.empty(0,np.int64),native_PP_anchor_frames_1based=np.empty(0,np.int64),
            native_PP_fixed_zero_columns=state.reduced_camera_fixed_columns.copy(),third_pose_fixed_zero_columns=state.third_pose_slot_indices.copy(),shared_head_axes_columns=state.head_axes_indices.copy(),shared_nonlinear_optics_columns=state.optics_indices.copy(),
            shared_nonlinear_optics_initial_values=start[state.optics_indices].copy())
        return result
    receipt=dict(schema='rinkspace.explicit-fixed-head-shared-optics-owner-transition.v1',
        previous_complete_source_model_identity=old_identity,new_model=model_reference,producers=producers,
        old_programme_parameter_count=7674,programme_parameter_count=7687,
        native_PP2496_all_fixed_zero_and_not_used_in_projection=True,
        old_four_PP_gauge_and_conditional_source_metric_disabled=True,only_per_native_pan_tilt_zoom_camera_parameters=True,
        same_single_restored_public_source_bundle=True,new_raw_world_guard_source_J_caches=True,
        legacy_constant_working_scale_box_active=False,new_dynamic_internal_and_total_boxes_active=True,
        inactive_third_pose_slot_count=len(state.third_pose_slot_indices),
        historical_jacobian_reuse_admission_attribute=deepcopy(programme._unmeasured_mount_domain),
        internal_d_and_continuous_total_original_user_working_boxes_active=True,
        original_print_outside_positive_focal_and_source_coverage_guards_retained=True,
        dynamic_closest60x30_uniform_units=True,historical_parent_aux_count=len(old_aux),
        full_parent_aux_history_preserved_separately=True,no_accuracy_or_hardware_calibration_claim=True)
    def identity(adapter):
        return dict(schema='rinkspace.complete-fixed-head-shared-optics-programme.v1',binding=adapter.bundle.state.binding_reference,
            association=adapter.association_reference,producer=reference(__file__),source_producers=deepcopy(adapter.source_producers),
            seed=deepcopy(adapter.seed_receipt),parameter_count=state.count,shared_dense_count=len(adapter.shared),
            fixed_parameter_indices=adapter.fixed.tolist(),native_PP_free_coordinate_count=0,
            only_per_native_pan_tilt_zoom_camera_parameters=True,shared_nonlinear_optics=model.provenance(),
            physical_camera_transition=deepcopy(receipt),shared_transverse_geometry=model.base.provenance(),
            old_objective_source_model_identity=deepcopy(old_identity),
            initial_exact_complete_current_state_full_J_reuse=False,old_source_r_J_cache_reused=False)
    def payload(adapter,q):
        q=state._q(q);old,pp=state.split(q)
        return dict(q_augmented=q.copy(),qbase=old.copy(),native_PP_px=pp.copy(),
            shared_nonlinear_optics_values=q[state.optics_indices].copy(),two_axis_a_working_m=np.array([state.axis(q)]),
            transverse_geometry_values=state.deviations(q).copy(),**{'aux__'+k:v for k,v in adapter.checkpoint_arrays().items()})
    def kwargs(adapter):
        return dict(global_schur_count=len(adapter.shared),global_schur_indices=adapter.shared.copy(),
            fixed_parameter_indices=adapter.fixed.copy(),shared_parameter_bounds=state.shared_parameter_bounds,
            shared_linear_constraints=adapter.linear_constraints,trial_feasibility_recovery=adapter.recover,
            source_block_factory=adapter.source_blocks,checkpoint_arrays=adapter.checkpoint_arrays(),
            checkpoint_identity=adapter.checkpoint_identity(),cache_native_source_replay=True,
            reuse_source_jacobian=False,schur_backtracking=True,schur_view_lm_ratio=1.,
            schur_backtracking_aware_damping=True,adaptive_source_backtracking=True,
            initial_damping=adapter.seed_receipt['initial_damping'],
            initial_line_search_fraction=adapter.seed_receipt['initial_line_search_fraction'])
    programme.guard=MethodType(guard,programme);programme.linear_constraints=MethodType(constraints,programme)
    programme.recover=MethodType(recover,programme);programme.checkpoint_arrays=MethodType(arrays,programme)
    programme.checkpoint_identity=MethodType(identity,programme);programme.complete_state_payload=MethodType(payload,programme)
    programme.solver_kwargs=MethodType(kwargs,programme)
    previous_finite=getattr(programme,'finite_endpoint_recovery_receipt',None)
    if hasattr(programme,'finite_endpoint_recovery_receipt'):delattr(programme,'finite_endpoint_recovery_receipt')
    install_finite_endpoint_recovery(programme,allowed_columns=(95,96,97))
    programme.recovery_references=[model_reference,reference(__file__),
        reference(module.MOUNT_REFERENCE['path']),reference(install_finite_endpoint_recovery.__code__.co_filename)]
    receipt['previous_finite_recovery_receipt']=deepcopy(previous_finite)
    receipt['new_dynamic_count_finite_recovery_receipt']=deepcopy(programme.finite_endpoint_recovery_receipt)
    programme.shared_optics_transition_receipt=receipt
    return receipt
def expand_derived_chart(previous_chart,state,*,parameter_count,camera_start):
    """Keep authenticated recipe geometry; append only new shared optics zeros."""
    from .first_shot_derived_inactive import DerivedInactiveChart
    if previous_chart.base_count!=7674 or state.count!=7687:raise ValueError('Explicit7,674-to7,687 fixed-head chart transport required')
    anchor=np.r_[previous_chart.anchor,np.zeros(state.count-previous_chart.base_count)]
    camera=np.setdiff1d(np.arange(state.count),state.world_parameter_indices)
    anchor[camera]=state._q(camera_start)[camera]
    state._q(anchor)  # Geometry-only recipe anchor still obeys inactive-PP chart semantics.
    return DerivedInactiveChart(state,anchor,recipe_reference=previous_chart.recipe_reference,
        parameter_count=parameter_count,release_rows=previous_chart.released_rows,
        admission_dependencies=previous_chart.released_rows,dependency_records=previous_chart.dependency_records)
