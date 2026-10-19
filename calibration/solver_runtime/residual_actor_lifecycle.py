"""Own residual actors across the existing parent-only fresh J stage.

The explicit retention policy preserves idle residual-only workers, profiles
and output mappings while the original parent J and FD iterator runs. The
release policy remains available and lazily restarts the delegate afterward.
Neither policy sends J work to residual actors or changes source selection.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import time


def install_residual_actor_lifecycle(*, report_directory=None, retain_idle_residual_actors=False):
    from structured_programme_source_candidate.residual_only_source_actors import ResidualOnlySourceActors
    from z190_process_source_evaluation import PersistentSourceActors
    from z190_joint_source_solver import _frozen_profile_token

    retain_idle_residual_actors = bool(retain_idle_residual_actors)
    installed = getattr(ResidualOnlySourceActors, '_rinkspace_fresh_J_actor_lifecycle', None)
    destination = None if report_directory is None else Path(report_directory).resolve()
    if installed is not None:
        if installed.get('retains_idle_residual_actors_during_parent_J', False) != retain_idle_residual_actors:
            raise RuntimeError('Residual lifecycle already has another explicit retention policy')
        if installed['report_directory'] != (None if destination is None else str(destination)):
            raise RuntimeError('Residual lifecycle was already installed for another report directory')
        return dict(installed, already_installed=True)
    original = ResidualOnlySourceActors.ordered_evaluations

    def record(manager, phase, **values):
        event = dict(phase=phase, wall_time_ns=time.time_ns(), **values)
        if not hasattr(manager, 'residual_lifecycle_events'):
            manager.residual_lifecycle_events = []
        manager.residual_lifecycle_events.append(event)
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)
            with (destination/'residual_actor_lifecycle.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False, separators=(',', ':')) + '\n')
        print(json.dumps(dict(stage='residual_actor_lifecycle', **event)), flush=True)

    def ordered_with_stage_lifetime(manager, q, jacobian):
        if not jacobian:
            yield from original(manager, q, False)
            return
        delegate = manager.delegate
        if delegate.closed:
            raise RuntimeError('Explicitly closed residual actors cannot perform a fresh J')
        if delegate._busy:
            raise RuntimeError('Fresh parent J requires the original residual actors idle')
        retained = False
        retained_profile = None
        retained_pids = None
        if delegate.started:
            before = _frozen_profile_token(manager.profile_update)
            pids = [actor['process'].pid for actor in delegate._actors]
            output_bytes = sum(actor['spec']['bytes'] for actor in delegate._actors)
            if retain_idle_residual_actors:
                if delegate.output_jacobian or any(
                        actor['spec'].get('output_jacobian', True) for actor in delegate._actors):
                    raise RuntimeError('Only original residual-only actors may be retained during parent J')
                retained = True
                retained_profile, retained_pids = before, pids
                record(manager, 'idle_residuals_retained_during_original_parent_J',
                    owned_actor_pids=pids, original_profile_token=before,
                    shared_output_bytes=output_bytes, output_jacobian=False,
                    actor_J_commands_sent=False, actor_FD_pool_started=False,
                    delegate_identity=id(delegate), replacement_delegate_started=False)
            else:
                record(manager, 'before_idle_residual_release_for_fresh_J',
                    owned_actor_pids=pids, original_profile_token=before, shared_output_bytes=output_bytes)
                began = time.perf_counter()
                # The original close detaches each SHM-backed parent weight array
                # before unmapping it and stops only this delegate's owned workers.
                delegate.close()
                if delegate.started or delegate._actors:
                    raise RuntimeError('Idle residual delegate did not fully close')
                after = _frozen_profile_token(manager.profile_update)
                if after != before:
                    raise RuntimeError('Residual release changed original frozen profile bytes')
                manager.delegate = PersistentSourceActors(manager.actor_blocks,
                    task_root=delegate.task_root, native_threads=delegate.native_threads,
                    q_columns=delegate.q_columns, output_jacobian=False)
                record(manager, 'idle_residuals_released_before_fresh_J',
                    previous_owned_actor_pids=pids, original_profile_token=after,
                    frozen_profiles_byte_exact=True, release_seconds=time.perf_counter()-began,
                    replacement_delegate_started=False, original_blocks_and_parent_FD_unchanged=True)
                manager._report('idle_residuals_released_before_fresh_J')
        succeeded = False
        try:
            # The original parent/attached-FD route stays unchanged. It
            # never starts a lazy delegate or sends any J command to an actor.
            yield from original(manager, q, True)
            succeeded = True
        finally:
            if retained:
                retention_error = None
                after = None
                try:
                    if (manager.delegate is not delegate or delegate.closed
                            or not delegate.started or delegate._busy
                            or [actor['process'].pid for actor in delegate._actors] != retained_pids):
                        raise RuntimeError('Idle residual delegate changed during original parent J')
                    after = _frozen_profile_token(manager.profile_update)
                    if after != retained_profile:
                        raise RuntimeError('Original parent J changed frozen residual profile bytes')
                except Exception as error:
                    retention_error = error
                try:
                    record(manager, 'parent_J_iterator_returned_original_residuals_retained',
                        succeeded=succeeded, owned_actor_pids=retained_pids,
                        resident_residual_workers=len(manager.delegate._actors),
                        original_profile_token=after, frozen_profiles_byte_exact=after == retained_profile,
                        retention_check_error=None if retention_error is None else
                            type(retention_error).__name__+': '+str(retention_error),
                        replacement_delegate_started=False, original_driver_FD_release_callback_retained=True,
                        original_profile_synchronization_and_driver_final_close_retained=True)
                except Exception:
                    if succeeded:
                        raise
                # A new lifecycle audit may fail a successful return, but it
                # must not override the original numerical or control error.
                if succeeded and retention_error is not None:
                    raise retention_error
            else:
                record(manager, 'parent_J_iterator_returned_residuals_remain_lazy',
                    succeeded=succeeded, resident_residual_workers=len(manager.delegate._actors),
                    replacement_delegate_started=manager.delegate.started,
                    restart_owned_by_next_original_profile_residual_or_guard_command=True,
                    original_driver_FD_release_callback_retained=True)

    ResidualOnlySourceActors.ordered_evaluations = ordered_with_stage_lifetime
    receipt = dict(schema='rinkspace.residual-actor-fresh-J-lifetime.v1',
        helper_path=str(Path(__file__).resolve()),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_ordered_evaluations_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
        report_directory=None if destination is None else str(destination),
        releases_only_idle_owned_residual_actors_before_fresh_J=not retain_idle_residual_actors,
        retains_idle_residual_actors_during_parent_J=retain_idle_residual_actors,
        retained_route_uses_original_residual_only_output_and_parent_J=True,
        original_profile_synchronization_and_driver_final_close_retained=True,
        original_parent_J_and_geometry_FD_iterator_unchanged=True,
        lazy_original_restart_and_profile_synchronization=not retain_idle_residual_actors,
        original_profile_synchronization_retained=True,
        frozen_profile_bytes_checked_before_and_after_release=not retain_idle_residual_actors,
        frozen_profile_bytes_checked_across_parent_J=retain_idle_residual_actors,
        no_memory_capacity_threshold_introduced=True, registered_source_files_modified=False,
        accuracy_accepted=False)
    ResidualOnlySourceActors._rinkspace_fresh_J_actor_lifecycle = receipt
    return receipt
