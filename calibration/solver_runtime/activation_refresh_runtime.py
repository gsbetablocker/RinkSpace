"""Use original parent activation derivatives beside residual-only actors.

Residual workers own independent pickled source graphs and write their own
residual buffers.  The existing parent graph and synchronized frozen profiles
remain the original derivative owner.  This adapter removes only a blanket
actor-presence refusal for that verified execution arrangement; every original
activation, exact-row, complete-family and all-row identity gate remains.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np


def _resident_parent_activation_allowed(programme):
    from structured_programme_source_candidate.residual_only_source_actors import ResidualOnlySourceActors
    from z190_process_source_evaluation import PersistentSourceActors
    manager = programme.actors
    if not isinstance(manager, ResidualOnlySourceActors):
        return False
    blocks = tuple(programme.state.blocks)
    if (len(manager.blocks) != len(blocks)
            or any(left is not right for left, right in zip(manager.blocks, blocks))
            or len(manager.actor_blocks) != len(blocks)
            or any(clone.bundle is not owner.bundle
                   for clone, owner in zip(manager.actor_blocks, blocks))
            or programme.profile_update is not manager.profile_update):
        return False
    delegate = manager.delegate
    return bool(isinstance(delegate, PersistentSourceActors)
                and not delegate.closed and delegate.output_jacobian is False
                and len(delegate.blocks) == len(manager.actor_blocks)
                and all(left is right for left, right in zip(delegate.blocks, manager.actor_blocks)))


def _actor_snapshot(manager):
    delegate = manager.delegate
    return dict(delegate_identity=id(delegate), started=bool(delegate.started),
                closed=bool(delegate.closed), busy=bool(delegate._busy),
                output_jacobian=bool(delegate.output_jacobian),
                actors=[dict(pid=actor['process'].pid, memory_name=actor['spec']['name'],
                             memory_identity=id(actor['memory']),
                             array_identities={key: id(value) for key, value in actor['arrays'].items()})
                        for actor in delegate._actors])


def _programme_from_chain(chain):
    from correlated_roll_prior_candidate.explicit_bias_source import ExplicitBiasSource
    from structured_programme_source_candidate.current_full_source import CurrentProgrammeSource
    bias = next((owner for owner in chain if isinstance(owner, ExplicitBiasSource)), None)
    programme = None if bias is None else getattr(bias.cap.get('iter_source_blocks'), '__self__', None)
    return programme if isinstance(programme, CurrentProgrammeSource) else None


def _original_parent_FD_pool(programme, family):
    matches = [owner for owner in programme.state.blocks if 'programme:' + owner.name == family]
    if len(matches) != 1:
        return None
    return getattr(matches[0], '_absolute_geometry_fd_pool', None)


def _parent_activation_FD_available(pool):
    if pool is None:
        return True
    from .absolute_geometry_fd_pool_candidate import AbsoluteGeometryGroupPool
    if (not isinstance(pool, AbsoluteGeometryGroupPool)
            or not hasattr(type(pool), '_rinkspace_completion_driven_FD_runtime')
            or pool.closed or not pool.owns_executor or pool.parent_geometry is None
            or int(getattr(pool, '_rinkspace_active_geometry_group_iterators', 0)) != 0
            or getattr(pool, '_rinkspace_confirmed_columns_active', False)
            or getattr(pool, '_rinkspace_native_columns_active', False)
            or getattr(pool, '_rinkspace_shared_material_bank_precompute_active', False)
            or int(getattr(pool, '_rinkspace_confirmed_columns_pending_jobs', 0)) != 0
            or int(getattr(pool, '_rinkspace_native_columns_pending_jobs', 0)) != 0
            or int(getattr(pool, '_rinkspace_shared_material_bank_pending_jobs', 0)) != 0):
        return False
    if pool.executor is not None:
        last = getattr(pool, 'last_completion_scheduling', {})
        absolute_idle = (last.get('active_futures_at_exit') == 0
                         and last.get('buffered_batches_at_exit') == 0)
        private_idle = any(
            receipt.get('executor_identity') == id(pool.executor)
            and receipt.get('pending_jobs') == 0
            for receipt in (getattr(pool, name, {}) for name in (
                '_rinkspace_confirmed_columns_idle_receipt',
                '_rinkspace_native_columns_idle_receipt',
                '_rinkspace_shared_material_bank_idle_receipt')))
        if not (absolute_idle or private_idle):
            return False
    return True


def _allow_verified_resident_parent(original):
    source = inspect.getsource(original)
    tree = ast.parse(source)
    changed = []

    class ParentGuard(ast.NodeTransformer):
        def visit_Compare(self, node):
            node = self.generic_visit(node)
            if ast.unparse(node) == 'programme.actors is not None':
                changed.append(node)
                replacement = ast.parse(
                    'programme.actors is not None and not _resident_parent_activation_allowed(programme)',
                    mode='eval').body
                return ast.copy_location(replacement, node)
            return node

    tree = ParentGuard().visit(tree)
    if len(changed) != 1:
        raise RuntimeError('Activation adapter requires exactly one original actor-presence refusal')
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    namespace['_resident_parent_activation_allowed'] = _resident_parent_activation_allowed
    exec(compile(tree, str(Path(__file__).resolve()) + ':original_parent_activation', 'exec'), namespace)
    return namespace[original.__name__], hashlib.sha256(source.encode()).hexdigest()


def install_parent_activated_source_refresh(*, report_directory=None):
    """Install before binding the original activated-source refresh callback."""
    from calibration.solver_runtime import install
    install()
    import structured_programme_source_candidate.activated_small_source_blocks as activated
    from z190_joint_source_solver import _frozen_profile_token

    previous = getattr(activated, '_rinkspace_resident_parent_activation_receipt', None)
    destination = None if report_directory is None else Path(report_directory).resolve()
    if previous is not None:
        if previous['report_directory'] != (None if destination is None else str(destination)):
            raise RuntimeError('Parent activation refresh already installed for another report directory')
        return dict(previous, already_installed=True)
    original_rows = activated.fresh_activated_small_source_rows
    original_family = activated._fresh_complete_programme_family
    permitted_rows, rows_sha = _allow_verified_resident_parent(original_rows)
    permitted_family, family_sha = _allow_verified_resident_parent(original_family)

    def record(**values):
        event = dict(stage='original_parent_activated_source_refresh', wall_time_ns=time.time_ns(), **values)
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)
            with (destination / 'original_parent_activated_source_refresh.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False, separators=(',', ':')) + '\n')
        print(json.dumps(event), flush=True)

    @contextmanager
    def unchanged_parent_and_actors(programme, family, operation, requested_rows=None):
        profile_before = _frozen_profile_token(programme.profile_update)
        if profile_before is None:
            raise RuntimeError('Original frozen parent profiles are required for activation refresh')
        actors_before = _actor_snapshot(programme.actors)
        pool = _original_parent_FD_pool(programme, family)
        if not _parent_activation_FD_available(pool):
            raise RuntimeError('Original activation parent FD requires an idle owned completion manager')
        force_attribute = '_rinkspace_force_parent_activation_FD'
        force_was_present = pool is not None and force_attribute in vars(pool)
        original_force = None if pool is None else getattr(pool, force_attribute, None)
        values = dict(family=family, operation=operation,
                      frozen_profile_token=profile_before, actor_command_sent=False,
                      original_parent_public_operator=True,
                      actor_iterator_busy=actors_before['busy'],
                      activation_FD_parent_only=pool is not None,
                      original_absolute_FD_grouping_projection_and_stencils_preserved=True)
        if requested_rows is not None:
            values['activated_local_rows'] = np.asarray(requested_rows, dtype=np.int64).tolist()
        record(phase='begin', **values)
        began = time.perf_counter()
        succeeded = False
        error = None
        try:
            if pool is not None:
                # Every existing generator/job is complete. Release only this
                # owned idle executor, then use completion's explicit original
                # parent route; never pretend there is memory pressure.
                if pool.executor is not None:
                    pool.release_idle()
                setattr(pool, force_attribute, True)
            yield values
            succeeded = True
        except BaseException as exc:
            error = exc
            raise
        finally:
            if pool is not None:
                if force_was_present:
                    setattr(pool, force_attribute, original_force)
                else:
                    vars(pool).pop(force_attribute, None)
            profile_after = _frozen_profile_token(programme.profile_update)
            actors_after = _actor_snapshot(programme.actors)
            unchanged = profile_before == profile_after and actors_before == actors_after
            record(phase='end', seconds=time.perf_counter() - began,
                   succeeded=succeeded, frozen_profiles_and_actor_transport_unchanged=unchanged,
                   **values)
            if not unchanged and error is None:
                raise RuntimeError('Original parent activation refresh changed frozen profiles or actor transport state')

    def rows_with_resident_parent(source, q, supplied, rows):
        chain = tuple(activated._owners(source))
        programme = _programme_from_chain(chain)
        family = supplied.row_ledger.get('family') if hasattr(supplied, 'row_ledger') else supplied['row_ledger'].get('family')
        if programme is None or not _resident_parent_activation_allowed(programme):
            return original_rows(source, q, supplied, rows)
        if not isinstance(family, str) or not family.startswith('programme:'):
            return original_rows(source, q, supplied, rows)
        if not _parent_activation_FD_available(_original_parent_FD_pool(programme, family)):
            return original_rows(source, q, supplied, rows)
        with unchanged_parent_and_actors(programme, family, 'exact_original_row_subset', rows) as values:
            patch = permitted_rows(source, q, supplied, rows)
            values['exact_row_patch_available'] = patch is not None
            if patch is not None:
                values['refreshed_scalar_row_count'] = len(patch.row_indices)
                values['public_operator_detail'] = patch.detail
            return patch

    def family_with_resident_parent(chain, value, ledger):
        programme = _programme_from_chain(chain)
        if programme is None or not _resident_parent_activation_allowed(programme):
            return original_family(chain, value, ledger)
        if not _parent_activation_FD_available(_original_parent_FD_pool(programme, ledger['family'])):
            return original_family(chain, value, ledger)
        with unchanged_parent_and_actors(programme, ledger['family'], 'exact_original_complete_family') as values:
            result = permitted_family(chain, value, ledger)
            values['exact_complete_family_available'] = result is not None
            if result is not None:
                values['complete_family_rows'] = len(result.r)
            return result

    activated.fresh_activated_small_source_rows = rows_with_resident_parent
    activated._fresh_complete_programme_family = family_with_resident_parent
    receipt = dict(
        schema='rinkspace.resident-parent-activated-source-refresh.v1',
        helper_path=str(Path(__file__).resolve()),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_exact_row_subset_function_sha256=rows_sha,
        original_complete_programme_family_function_sha256=family_sha,
        report_directory=None if destination is None else str(destination),
        blanket_actor_refusal_removed_only_for_verified_residual_only_parent=True,
        original_zero_J_activation_and_current_row_identity_gates_retained=True,
        original_parent_profile_q_source_sigma_mask_scales_and_public_derivatives_retained=True,
        original_motion_rows_keep_complete_family_fallback=True,
        actor_commands_busy_state_shared_output_mapping_and_frozen_profiles_unchanged_checked=True,
        activation_FD_uses_explicit_original_parent_completion_route=True,
        activation_FD_preserves_original_batching_projection_and_domain_stencils=True,
        activation_FD_adds_no_processes_beside_residual_actors=True,
        activation_FD_force_attribute_restored_before_next_normal_full_J=True,
        unknown_or_busy_FD_manager_keeps_original_activation_refusal=True,
        source_graph_constructed=False, registered_source_files_modified=False,
        numerical_refresh_ratio_and_acceptance_changed=False,
        actual_complete_source_accuracy_or_speed_accepted=False)
    activated._rinkspace_resident_parent_activation_receipt = receipt
    return receipt
