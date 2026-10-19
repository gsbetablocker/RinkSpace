"""Audited execution scheduling for the complete named66 source solver.

The registered numerical producers keep their cache identities. Each run also
records the exact execution helpers and transformed function identities. These
hooks schedule original work, reject a trial by an original necessary condition,
or stop a failed optional curvature correction while retaining a direction that
has already passed the original full KKT checks. They do not certify accuracy.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path


def install_named66_execution(driver, diagnostic_directory):
    """Install once, before constructing sources or resident source actors."""
    if hasattr(driver, "_rinkspace_execution_acceleration"):
        return driver._rinkspace_execution_acceleration
    from .optional_soc_runtime import install_optional_SOC_numerical_failure_route
    from .activation_refresh_runtime import install_parent_activated_source_refresh
    from .camera_export_runtime import install_camera_export_runtime
    from .completion_driven_absolute_fd_runtime import install_completion_driven_absolute_fd_runtime
    from .parallel_source_guard import install_parallel_source_guard
    from .parallel_source_validation import install_validation
    from .private_material_snapshot_runtime import install_private_material_snapshot_runtime
    from .priority_feasibility_witness_runtime import install_priority_feasibility_witness_runtime
    from .residual_actor_lifecycle import install_residual_actor_lifecycle
    from .solver_control_runtime import install_solver_control_runtime, source_control_context
    from .same_target_fresh_J_verification import install_same_target_fresh_J_verification
    from .source_gc_observation_runtime import install_source_gc_observation_runtime
    from .trial_precheck_runtime import with_trial_precheck, restoration_precheck_from_callback
    from .restoration_trial_model_consistency import with_restoration_trial_model_consistency
    from .wide_top_only_candidate import install_wide_top_only_runtime

    destination = Path(diagnostic_directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    comparison = os.environ.get('RINKSPACE_COMPLETE_CYCLE_VARIANT', '')
    if comparison not in ('', 'baseline', 'candidate'):
        raise ValueError('Unknown complete cycle comparison variant')
    fd_source = Path(__file__).with_name('completion_driven_absolute_fd_runtime.py')
    if comparison == 'baseline':
        import importlib.util
        fd_source = Path(os.environ['RINKSPACE_BASELINE_FD_SOURCE']).resolve()
        expected = '761fb8a87b612938a57533876c50f6e860106d48bb4b336a579450529262c82c'
        if hashlib.sha256(fd_source.read_bytes()).hexdigest() != expected:
            raise ValueError('The measured 210-second baseline execution snapshot changed')
        spec = importlib.util.spec_from_file_location(
            'calibration.solver_runtime._complete_cycle_baseline_FD', fd_source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        install_completion_driven_absolute_fd_runtime = module.install_completion_driven_absolute_fd_runtime
    receipt = {
        "schema": "rinkspace.complete-source-execution-scheduling.v2",
        "source_GC_observation": install_source_gc_observation_runtime(
            report_directory=destination / 'source_GC_observation'),
        "qp": install_optional_SOC_numerical_failure_route(expand_separation_batches=False,
            parallel_backend=None),
        "explicit_control": install_solver_control_runtime(),
        "phaseI_complete_primal_witness": install_priority_feasibility_witness_runtime(
            report_directory=destination / 'priority_feasibility_witness'),
        "parallel_guard": install_parallel_source_guard(),
        "residual_actor_lifecycle": install_residual_actor_lifecycle(
            report_directory=destination, retain_idle_residual_actors=True),
        "private_material_snapshot": install_private_material_snapshot_runtime(
            report_directory=destination / 'private_material_execution'),
        "absolute_FD_completion_scheduling": install_completion_driven_absolute_fd_runtime(
            report_directory=destination / 'absolute_FD_completion_execution'),
        "parent_activated_source_refresh": install_parent_activated_source_refresh(
            report_directory=destination / 'activation_refresh_execution'),
        "full_source_actor_cache_validation": install_validation(destination, require_residual_actors=False),
        "wide_top_only": install_wide_top_only_runtime(driver),
        "camera_export": install_camera_export_runtime(driver, destination / 'camera_export_execution'),
    }
    if comparison != 'baseline':
        from .startup_npz_member_reuse_runtime import install_startup_npz_member_reuse_runtime
        from .constraint_sample_reuse_runtime import install_constraint_sample_reuse_runtime
        from .tail_world_memo_runtime import install_tail_world_memo_runtime
        from .absolute_material_value_runtime import install_absolute_material_value_runtime
        from .source_actor_payload_runtime import install_source_actor_payload_runtime
        from .shared_actor_geometry_payload_runtime import install_shared_actor_geometry_payload_runtime
        from .confirmed_constraint_columns_runtime import install_confirmed_constraint_columns_runtime
        from .camera_absolute_reuse_runtime import install_camera_absolute_reuse_runtime
        from .camera15_group_prefetch_runtime import install_camera15_group_prefetch_runtime
        from .programme_csr_three_span_merge_runtime import install_programme_csr_three_span_merge_runtime
        from .wide_source_scene_runtime import install_wide_source_scene_runtime
        from .native_factor_columns_runtime import install_native_factor_columns_runtime
        from .review_projection_reuse_runtime import install_review_projection_reuse_runtime
        from .startup_source_auth_reuse_runtime import install_startup_source_auth_reuse_runtime
        from .finite_extent_guard_reuse_runtime import install_finite_extent_guard_reuse_runtime
        from .role_record_index_runtime import install_role_record_index_runtime
        from .role_binding_value_reuse_runtime import install_role_binding_value_reuse_runtime
        from .compaction_binding_reindex_runtime import install_compaction_binding_reindex_runtime
        from .complete_source_pool_lifetime_runtime import install_complete_source_pool_lifetime_runtime
        from .pattern_fd_pool_reuse_runtime import install_pattern_fd_pool_reuse_runtime
        from .static_role_qualification_cache_runtime import install_static_role_qualification_cache_runtime
        from .persistent_qp_face_hint_runtime import install_persistent_qp_face_hint_runtime
        from .persistent_soc_row_hint_runtime import install_persistent_soc_row_hint_runtime
        from .prepared_oob_cache_runtime import install_prepared_oob_cache_runtime
        from .full_priority_LP_backend_runtime import install_full_priority_LP_backend_runtime
        receipt['role_record_index'] = install_role_record_index_runtime(
            report_directory=destination / 'role_record_index_execution')
        receipt['role_binding_value_reuse'] = install_role_binding_value_reuse_runtime(
            report_directory=destination / 'role_binding_value_reuse_execution')
        receipt['compaction_binding_reindex'] = install_compaction_binding_reindex_runtime(
            report_directory=destination / 'compaction_binding_reindex_execution')
        receipt['complete_source_pool_lifetime'] = install_complete_source_pool_lifetime_runtime(
            driver, report_directory=destination / 'complete_source_pool_lifetime_execution')
        receipt['pattern_existing_FD_executor'] = install_pattern_fd_pool_reuse_runtime(
            driver, report_directory=destination / 'pattern_fd_pool_reuse_execution')
        receipt['separated_priority_LP'] = dict(enabled=False, reason='Real source subset backend failed and added work')
        receipt['full_priority_LP_backend'] = dict(enabled=False,
            reason='Complete-model dual simplex returned NotSet; retain original full IPM')
        receipt['full_priority_Clarabel'] = dict(enabled=False,
            reason='Complete original 1e-9 KKT failed; Clarabel added 4.14 seconds before original IPM')
        receipt['finite_extent_guard_reuse'] = install_finite_extent_guard_reuse_runtime(
            report_directory=destination / 'finite_extent_guard_reuse_execution')
        receipt['startup_npz_member_reuse'] = install_startup_npz_member_reuse_runtime(
            report_directory=destination / 'startup_npz_member_reuse_execution')
        receipt['startup_source_auth_reuse'] = install_startup_source_auth_reuse_runtime(
            report_directory=destination / 'startup_source_auth_execution')
        from .config import data_root
        receipt['prepared_OOB_cache'] = dict(enabled=False,
            reason='Authenticated mapped hit took 13.94 seconds; original prepared load retained')
        receipt['persistent_QP_face_hint'] = install_persistent_qp_face_hint_runtime(
            cache_directory=data_root() / 'prepared_full8503' / 'QP_face_hints',
            report_directory=destination / 'QP_face_hint_execution')
        receipt['persistent_SOC_row_hint'] = install_persistent_soc_row_hint_runtime(
            cache_directory=data_root() / 'prepared_full8503' / 'QP_face_hints',
            report_directory=destination / 'SOC_row_hint_execution')
        receipt['static_role_qualification_cache'] = install_static_role_qualification_cache_runtime(
            cache_directory=data_root() / 'prepared_full8503' / 'static_role_qualification',
            report_directory=destination / 'static_role_qualification_execution')
        receipt['constraint_sample_reuse'] = install_constraint_sample_reuse_runtime(
            report_directory=destination / 'constraint_sample_reuse_execution')
        receipt['tail_world_memo'] = install_tail_world_memo_runtime(
            report_directory=destination / 'tail_world_memo_execution')
        receipt['absolute_material_value'] = install_absolute_material_value_runtime(
            report_directory=destination / 'absolute_material_value_execution')
        receipt['source_actor_payload'] = install_source_actor_payload_runtime(
            report_directory=destination / 'source_actor_payload_execution')
        receipt['shared_actor_geometry_payload'] = install_shared_actor_geometry_payload_runtime(
            report_directory=destination / 'shared_actor_geometry_execution')
        receipt['confirmed_constraint_columns'] = install_confirmed_constraint_columns_runtime(
            report_directory=destination / 'confirmed_constraint_columns_execution')
        receipt['native_factor_columns'] = install_native_factor_columns_runtime(
            report_directory=destination / 'native_factor_columns_execution')
        receipt['camera_absolute_reuse'] = install_camera_absolute_reuse_runtime(
            report_directory=destination / 'camera_absolute_reuse_execution')
        receipt['camera15_group_prefetch'] = install_camera15_group_prefetch_runtime(
            report_directory=destination / 'camera15_group_prefetch_execution')
        receipt['programme_CSR_three_span_merge'] = install_programme_csr_three_span_merge_runtime(
            report_directory=destination / 'programme_csr_three_span_merge_execution')
        receipt['wide_numeric_source_scene'] = install_wide_source_scene_runtime(
            driver, report_directory=destination / 'wide_numeric_source_scene_execution')
        receipt['review_projection_reuse'] = install_review_projection_reuse_runtime(
            report_directory=destination / 'review_projection_reuse_execution')
        from .finite_rectangle_station_columns_runtime import install_station_columns_runtime
        receipt['finite_station_columns'] = install_station_columns_runtime(
            report_directory=destination / 'station_columns_execution')
    original_solver = driver.solve_joint_source
    scheduled_solver = with_trial_precheck(original_solver)
    receipt["trial_precheck"] = scheduled_solver.execution_change

    @functools.wraps(original_solver)
    def invoke(evaluate, q, quadratic, scales, solver_output, physicalguard, **kwargs):
        callback = kwargs.get("source_trial_acceptance")
        if callback is not None:
            kwargs["source_trial_precheck"] = restoration_precheck_from_callback(callback)
            kwargs["source_trial_acceptance"] = with_restoration_trial_model_consistency(callback)
        with source_control_context(solver_output):
            return scheduled_solver(evaluate, q, quadratic, scales, solver_output, physicalguard, **kwargs)

    driver.solve_joint_source = invoke
    receipt["same_target_fresh_J_verification"] = install_same_target_fresh_J_verification(
        driver, destination)
    helper_names = (
        "execution_acceleration.py", "optional_soc_runtime.py",
        "parallel_source_guard.py", "parallel_source_validation.py",
        "trial_precheck_runtime.py", "staged_source_guard.py",
        "trial_fraction_source_domain.py", "restoration_trial_model_consistency.py",
        "bounded_motion_schur.py", "motion_bounded_schur_terms.py", "truth_source_location_revision.py",
        "source_scanline_curve.py", "truth_source_scanline_revision.py",
        "original_measured_location_mode.py", "original_measured_transition_locations.py", "truth_original_location_revision.py",
        "wide_top_only_candidate.py",
        "residual_actor_lifecycle.py", "solver_control_runtime.py", "camera_export_runtime.py",
        "completion_driven_absolute_fd_runtime.py",
        "activation_refresh_runtime.py",
        "same_target_fresh_J_verification.py",
        "source_gc_observation_runtime.py",
        "private_material_snapshot_runtime.py",
        "priority_feasibility_witness_runtime.py",
        "physical_signed_margin_SOC.py",
        "complete_cycle_comparison_runtime.py",
    )
    if comparison != 'baseline':
        helper_names += ('constraint_sample_reuse_runtime.py', 'tail_world_memo_runtime.py',
                         'absolute_material_value_runtime.py', 'source_actor_payload_runtime.py',
                         'shared_actor_geometry_payload_runtime.py', 'role_block_binding_runtime.py',
                         'role_view_binding_seed_runtime.py',
                         'native_memo_actor_transport.py', 'full_priority_clarabel_runtime.py',
                         'source_assembly_gc_runtime.py',
                         'camera15_group_prefetch_runtime.py',
                         'programme_csr_three_span_merge_runtime.py', 'native/programme_csr_three_span_merge.c',
                         'wide_source_scene_runtime.py',
                         'source_state_skeleton_cache_runtime.py', 'named66_source_skeleton_adapter_runtime.py',
                         'confirmed_constraint_columns_runtime.py', 'native_factor_columns_runtime.py',
                         'camera_absolute_reuse_runtime.py', 'startup_npz_member_reuse_runtime.py',
                         'review_projection_reuse_runtime.py', 'startup_source_auth_reuse_runtime.py',
                         'finite_extent_guard_reuse_runtime.py', 'role_record_index_runtime.py',
                         'role_binding_value_reuse_runtime.py', 'separated_priority_LP_runtime.py',
                         'compaction_binding_reindex_runtime.py', 'complete_source_pool_lifetime_runtime.py',
                         'pattern_fd_pool_reuse_runtime.py',
                         'static_role_qualification_cache_runtime.py', 'persistent_qp_face_hint_runtime.py',
                         'finite_rectangle_station_columns_runtime.py',
                         'persistent_soc_row_hint_runtime.py', 'prepared_oob_cache_runtime.py', 'full_priority_LP_backend_runtime.py')
    if os.environ.get('RINKSPACE_STARTUP_DIAGNOSTIC_EVIDENCE') and comparison != 'baseline':
        helper_names += ('certified_startup_diagnostic_runtime.py',)
    root = Path(__file__).resolve().parent
    receipt["execution_helpers"] = [
        {"path": str(fd_source if name == 'completion_driven_absolute_fd_runtime.py' else root / name),
         "sha256": hashlib.sha256((fd_source if name == 'completion_driven_absolute_fd_runtime.py' else root / name).read_bytes()).hexdigest()}
        for name in helper_names
    ]
    snapshots = destination / 'execution_sources'
    snapshots.mkdir(parents=True, exist_ok=True)
    for helper in receipt['execution_helpers']:
        source_path = Path(helper['path'])
        archived = snapshots / source_path.name
        contents = source_path.read_bytes()
        if archived.exists() and archived.read_bytes() != contents:
            archived = snapshots / (source_path.stem + '.' + helper['sha256'] + '.py')
        if not archived.exists():
            archived.write_bytes(contents)
        helper['run_source_snapshot'] = str(archived)
    receipt.update(registered_source_files_unchanged=True,
                   original_source_rows_profiles_weights_and_acceptance_retained=True,
                   accuracy_acceptance_implied=False)
    (destination / "execution_acceleration_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    driver._rinkspace_execution_acceleration = receipt
    return receipt
