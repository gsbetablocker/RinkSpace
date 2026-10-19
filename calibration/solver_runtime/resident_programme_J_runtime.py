"""Original full programme Jacobians on the six existing isolated actors.

The original material bank is prepared once by the existing FD executor. That
executor is released before full J starts; actors retain their PIDs, original
private blocks and frozen profiles. Every original group FD, native source,
camera column and CSR insertion still runs, with original-order output.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import functools
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import time

import numpy as np


_RECEIPT = None


def _extend_worker_source(source):
    call = "result=block.evaluate(command['q'],jac)"
    ready = "finite_extent_guard_supported=True"
    if source.count(call) != 1 or source.count(ready) != 1:
        raise RuntimeError('Original guarded actor full-J insertion sites changed')
    return source.replace(call, 'result=_rinkspace_complete_J_evaluate(block,command)', 1).replace(
        ready, ready + ',complete_J_shared_bank_supported=True', 1)


def _install_worker_material_scope():
    """Retain the original scope, adding only the current command's bank."""
    from . import absolute_material_value_runtime as material
    from structured_programme_source_candidate.paired_programme_blocks import StructuredProgrammeBlock
    if getattr(StructuredProgrammeBlock, '_rinkspace_actor_bank_scope', False):
        return
    factory = material._programme_material_scope
    tree = ast.parse(textwrap.dedent(inspect.getsource(factory)))
    changes = []
    class BankScope(ast.NodeTransformer):
        def visit_keyword(self, node):
            if node.arg == 'shared_bank' and isinstance(node.value, ast.Constant) and node.value.value is None:
                changes.append(node.lineno)
                node.value = ast.Call(func=ast.Attribute(value=ast.Name(id='_BANK_SCOPE', ctx=ast.Load()),
                    attr='get', ctx=ast.Load()), args=[], keywords=[])
            return self.generic_visit(node)
    tree = BankScope().visit(tree)
    if len(changes) != 1:
        raise RuntimeError('Original programme material scope bank site changed')
    ast.fix_missing_locations(tree)
    namespace = dict(factory.__globals__)
    exec(compile(tree, __file__+'::<actor-programme-bank-scope>', 'exec'), namespace)
    original = StructuredProgrammeBlock._rinkspace_original_material_scoped_evaluate
    StructuredProgrammeBlock._evaluate = namespace[factory.__name__](original)
    StructuredProgrammeBlock._rinkspace_actor_bank_scope = True


def run_resident_programme_J_actor(*args):
    """Picklable Windows entry, retaining the original guarded worker."""
    from . import install
    install()
    from .absolute_material_value_runtime import install_absolute_material_value_runtime
    from .camera_absolute_reuse_runtime import install_camera_absolute_reuse_runtime
    from .finite_extent_guard_reuse_runtime import install_finite_extent_guard_reuse_runtime
    install_absolute_material_value_runtime()
    _install_worker_material_scope()
    install_camera_absolute_reuse_runtime()
    install_finite_extent_guard_reuse_runtime()
    from .parallel_source_guard import run_guarded_source_actor
    return run_guarded_source_actor(*args, complete_J_shared_bank=True)


class _OriginalGroupsInActor:
    """Use the existing batched group producer without starting another pool."""
    def __init__(self, block):
        self.block = block

    def ordered_groups(self, groups, world, absolute, step, *, batched_projection=True):
        from structured_programme_source_candidate.batched_absolute_geometry_fd import evaluate_geometry_group
        block = self.block
        certificates = getattr(block, '_inactive_material_certificates', None)
        if certificates is None:
            certificates = block._inactive_material_certificates = set()
        for ordinal, (actual_pts, rows, camera) in enumerate(groups):
            value = evaluate_geometry_group(block.geometry, world, camera, rows, absolute, step,
                certificates=certificates, batched_projection=batched_projection)
            yield dict(ordinal=ordinal, actual_pts=actual_pts, geo=value)


def _drop_worker_bank(descriptor):
    from . import absolute_material_value_runtime as material
    key = (descriptor['header_path'], descriptor['header_sha256'])
    bank = material._MAPPED_BANKS.pop(key, None)
    if bank is not None:
        bank['entries'].clear()
        bank['mapping']._mmap.close()


def _worker_complete_J_evaluate(block, command):
    if not command['jacobian']:
        return block.evaluate(command['q'], False)
    from . import absolute_material_value_runtime as material
    bank = command.get('_rinkspace_complete_J_material_bank')
    if not isinstance(bank, dict):
        raise ValueError('Resident complete J requires the original current material bank')
    world = np.asarray(block.geometry_values(command['q']), float)
    if bank['root_world_bytes'] != world.tobytes():
        raise ValueError('Resident complete J material bank belongs to another world')
    token = None
    adopted_pool = False
    try:
        material._load_shared_bank(bank)  # original complete file/producer authentication
        if vars(block).get('_absolute_geometry_fd_pool') is not None:
            raise RuntimeError('Resident complete J must not own a nested geometry FD pool')
        token = material._BANK_SCOPE.set(bank)
        block._absolute_geometry_fd_pool = _OriginalGroupsInActor(block)
        adopted_pool = True
        return block.evaluate(command['q'], True)
    finally:
        if adopted_pool:
            vars(block).pop('_absolute_geometry_fd_pool', None)
        if token is not None:
            material._BANK_SCOPE.reset(token)
        # Only this immutable bank's private mapping is detached. Its owned
        # files survive until the parent's existing FD-manager close.
        _drop_worker_bank(bank)


@contextmanager
def _parent_physical_replay_scope(block, q, bank):
    from . import absolute_material_value_runtime as material
    scope = dict(job=None, geometry=block.geometry, root_world_bytes=None,
        root_world_factory=lambda: block.geometry_values(q), shared_bank=bank,
        hits=0, original_value_calls=0, uncacheable_results=0, root_cache_resets=0)
    token = material._SCOPE.set(scope)
    try:
        yield
    finally:
        material._SCOPE.reset(token)


def _check_idle_pool(pool):
    if pool.closed or not pool.owns_executor:
        raise RuntimeError('Resident full J requires the original owned FD manager')
    active = ('_rinkspace_confirmed_columns_active', '_rinkspace_native_columns_active',
              '_rinkspace_shared_material_bank_precompute_active')
    pending = ('_rinkspace_confirmed_columns_pending_jobs', '_rinkspace_native_columns_pending_jobs',
               '_rinkspace_shared_material_bank_pending_jobs', '_rinkspace_active_geometry_group_iterators')
    if any(bool(getattr(pool, name, False)) for name in active) or any(
            int(getattr(pool, name, 0)) != 0 for name in pending):
        raise RuntimeError('Resident full J cannot overlap active original FD jobs')


def _ordered_resident_J(manager, q, record):
    from . import absolute_material_value_runtime as material
    from .absolute_geometry_fd_pool_candidate import AbsoluteGeometryGroupPool
    from z190_joint_source_solver import _frozen_profile_token
    from z190_process_source_evaluation import _output_memory_spec
    delegate = manager.delegate
    if delegate.closed or delegate._busy:
        raise RuntimeError('Resident complete J requires the existing delegate idle and open')
    pools = [getattr(block, '_absolute_geometry_fd_pool', None) for block in manager.blocks]
    pool = pools[0] if pools else None
    if not isinstance(pool, AbsoluteGeometryGroupPool) or any(value is not pool for value in pools):
        raise RuntimeError('Complete programme J requires one unchanged original material FD manager')
    if any(block.geometry is not pool.parent_geometry for block in manager.blocks):
        raise RuntimeError('Complete programme J original shared material owner differs')
    _check_idle_pool(pool)
    before = _frozen_profile_token(manager.profile_update)
    q = np.asarray(q, float)
    world = np.asarray(manager.blocks[0].geometry_values(q), float)
    step = float(__import__('os').environ.get('RINKSPACE_GEOMETRY_FD_STEP_M', '1e-7'))
    begun = time.perf_counter()
    bank = material._prepare_shared_material_bank(pool, world, step)
    if bank is None:
        raise RuntimeError('Original current material bank was unavailable for resident complete J')
    _check_idle_pool(pool)
    preparation_seconds = time.perf_counter()-begun
    release_begin = time.perf_counter()
    released = pool.release_idle()
    if pool.executor is not None:
        raise RuntimeError('Original FD executor was not released before resident complete J')
    record(manager, 'material_bank_ready_FD_executor_released_before_actor_J',
        preparation_seconds=preparation_seconds, release_seconds=time.perf_counter()-release_begin,
        owned_FD_executor_released=released, shared_entries=bank['entries'],
        shared_bank_data_bytes=bank['data_bytes'], original_material_owner_sha256=bank['material_owner_sha256'])
    manager.start()
    if any(not actor.get('complete_J_shared_bank_supported') for actor in delegate._actors):
        raise RuntimeError('Source actor started before resident full-J worker installation')
    pids = [actor['process'].pid for actor in delegate._actors]
    exact_J_spec_bytes = sum(_output_memory_spec(block, delegate.q_columns, True)['bytes']
                            for block in delegate.blocks)
    delegate.resize_output(True)
    if _frozen_profile_token(manager.profile_update) != before:
        raise RuntimeError('Resident J output resize changed frozen profile bytes')
    delegate._rinkspace_complete_J_material_bank = bank
    iterator = None
    completed = False
    replay_seconds = 0.
    try:
        iterator = delegate.ordered_evaluations(q, True)
        for expected, (actor_block, (residual, matrix)) in zip(manager.blocks, iterator, strict=True):
            if actor_block.name != expected.name:
                raise ValueError('Original programme actor output order changed')
            # Restore the original parent's nominal finite planes, scene,
            # calibration and installation binding without evaluating old J.
            began = time.perf_counter()
            with _parent_physical_replay_scope(expected, q, bank):
                expected.local(q)
            expected.bundle.abs_groups = expected.groups
            replay_seconds += time.perf_counter()-began
            # Native replay/cache can retain source arrays after this stage.
            # Copy before restoring residual-sized output mappings.
            yield expected, (residual.copy(), matrix.copy())
        completed = True
    finally:
        if iterator is not None:
            iterator.close()  # original ordered drainage on failure/early close
        vars(delegate).pop('_rinkspace_complete_J_material_bank', None)
        if not delegate.closed:
            delegate.resize_output(False)
        after = _frozen_profile_token(manager.profile_update)
        if after != before:
            raise RuntimeError('Resident complete J changed original frozen profile bytes')
        manager.last_stats = list(delegate.last_stats)
        record(manager, 'resident_complete_J_returned', succeeded=completed,
            owned_actor_pids=pids, actor_pids_retained=pids == [a['process'].pid for a in delegate._actors],
            exact_full_J_output_mapping_bytes=exact_J_spec_bytes,
            resident_output_restored_to_residual=not delegate.output_jacobian,
            parent_physical_local_replay_seconds=replay_seconds,
            frozen_profiles_byte_exact=True, nested_FD_pool_started=False,
            original_FD_manager_retained_for_later_tail=True, original_actor_stats=manager.last_stats)


def install_resident_programme_J_runtime(*, report_directory=None):
    """Install after guard/lifecycle/material, BEFORE file-payload installer."""
    global _RECEIPT
    from . import worker_entry
    from .parallel_source_guard import run_guarded_source_actor
    from structured_programme_source_candidate.residual_only_source_actors import ResidualOnlySourceActors
    from z190_process_source_evaluation import PersistentSourceActors
    if _RECEIPT is not None:
        return _RECEIPT
    if getattr(PersistentSourceActors, '_rinkspace_file_payload_receipt', None) is not None:
        raise RuntimeError('Resident J must install before the file transport captures its worker entry')
    if worker_entry.run_source_actor is not run_guarded_source_actor:
        raise RuntimeError('Original guarded source actor entry must already be installed')
    original_ordered = ResidualOnlySourceActors.ordered_evaluations
    original_send = PersistentSourceActors._send
    original_receive = PersistentSourceActors._receive
    destination = None if report_directory is None else Path(report_directory).resolve()
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)

    def record(manager, phase, **values):
        event = dict(stage='resident_programme_J', phase=phase, wall_time_ns=time.time_ns(), **values)
        manager.resident_J_last_event = event
        if destination is not None:
            with (destination/'resident_programme_J.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False)+'\n')
        print(json.dumps(event), flush=True)

    @functools.wraps(original_send)
    def send(delegate, actor, command):
        bank = getattr(delegate, '_rinkspace_complete_J_material_bank', None)
        if command.get('mode') == 'evaluate' and command.get('jacobian') and bank is not None:
            command = dict(command, _rinkspace_complete_J_material_bank=bank)
        return original_send(delegate, actor, command)

    @functools.wraps(original_receive)
    def receive(delegate, actor):
        reply = original_receive(delegate, actor)
        if reply.get('status') == 'ready':
            actor['complete_J_shared_bank_supported'] = bool(reply.get('complete_J_shared_bank_supported'))
        return reply

    @functools.wraps(original_ordered)
    def ordered(manager, q, jacobian):
        if not jacobian:
            yield from original_ordered(manager, q, False)
            return
        yield from _ordered_resident_J(manager, q, record)

    PersistentSourceActors._send = send
    PersistentSourceActors._receive = receive
    ResidualOnlySourceActors.ordered_evaluations = ordered
    worker_entry.run_source_actor = run_resident_programme_J_actor
    _RECEIPT = dict(schema='rinkspace.resident-original-programme-complete-J.v1',
        helper_path=str(Path(__file__).resolve()), helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_guarded_worker_entry_retained=True, original_source_worker_operations_and_drainage_retained=True,
        original_source_row_group_stencil_camera_native_and_CSR_producers_retained=True,
        six_original_programme_Js_parallel_on_existing_actors=True, no_actor_eviction_or_restart_for_J=True,
        original_bank_prepared_once_before_J=True, existing_FD_executor_released_before_full_actor_J=True,
        original_FD_manager_retained_for_tail_and_native=True, full_J_output_mapping_restored_after_stage=True,
        parent_nominal_physical_local_and_abs_groups_replayed=True,
        original_frozen_profile_bytes_checked_before_and_after=True,
        result_arrays_copied_before_output_mapping_resize=True, baseline_H_g_or_old_J_imported=False,
        registered_source_files_modified=False, actual_complete_graph_certification_pending=True, accuracy_accepted=False)
    if destination is not None:
        (destination/'resident_programme_J_receipt.json').write_text(json.dumps(_RECEIPT, indent=2)+'\n', encoding='utf8')
    return _RECEIPT
