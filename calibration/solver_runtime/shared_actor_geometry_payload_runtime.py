"""Serialize the common, current actor geometry once per owned actor start.

This is transport reuse, not geometry/result reuse across q.  Every actor
unpickles its own complete geometry.  References from its original minimal
block into that geometry's pickle memo retain the original object identity.
"""
from __future__ import annotations

import ast
from contextvars import ContextVar
import functools
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import textwrap
import time
import uuid


_PERSISTENT_TAG = 'rinkspace.current-actor-common-geometry.v1'
_WORKER = None


class SharedActorGeometryPayloadError(RuntimeError):
    """Only the new payload's startup/transport integration failed."""


class _HashingOutput:
    def __init__(self, output):
        self.output = output
        self.digest = hashlib.sha256()
        self.bytes = 0

    def write(self, value):
        # Protocol 5 sends large contiguous ndarray payloads as PickleBuffer;
        # len() is unavailable there and is not a byte count for other views.
        view = value.raw() if isinstance(value, pickle.PickleBuffer) else memoryview(value)
        try:
            count = self.output.write(view)
            if count != view.nbytes:
                raise OSError('Incomplete owned actor payload write')
            self.digest.update(view)
            self.bytes += count
            return count
        finally:
            view.release()


class _PrivatePickler(pickle.Pickler):
    def __init__(self, output, common_memo):
        super().__init__(output, protocol=5)
        self.common_memo = common_memo
        self.references = 0

    def persistent_id(self, value):
        entry = self.common_memo.get(id(value))
        if entry is None:
            return None
        # The memo retains the object, so an id cannot be recycled here.
        if entry[1] is not value:
            raise SharedActorGeometryPayloadError('Common geometry memo identity changed')
        self.references += 1
        return (_PERSISTENT_TAG, entry[0])


class _PrivateUnpickler(pickle.Unpickler):
    def __init__(self, input_file, common_memo):
        super().__init__(input_file)
        self.common_memo = common_memo

    def persistent_load(self, identity):
        if (not isinstance(identity, tuple) or len(identity) != 2
                or identity[0] != _PERSISTENT_TAG or type(identity[1]) is not int
                or identity[1] not in self.common_memo):
            raise SharedActorGeometryPayloadError('Unknown common geometry object reference')
        return self.common_memo[identity[1]]


def _write_pickle(path, value, *, common_memo=None):
    from .source_assembly_gc_runtime import defer_automatic_source_gc
    with defer_automatic_source_gc('owned_actor_pickle_transport'):
        return _write_pickle_unscoped(path, value, common_memo=common_memo)


def _write_pickle_unscoped(path, value, *, common_memo=None):
    with path.open('xb') as output:
        hashing = _HashingOutput(output)
        if common_memo is None:
            pickler = pickle.Pickler(hashing, protocol=5)
            pickler.dump(value)
        else:
            from .native_memo_actor_transport import dump_private_with_common_memo, FORMAT
            dump_private_with_common_memo(hashing, value, common_memo)
        result = dict(path=str(path), owned_name=path.name,
                      owned_transport_root=str(path.parent), bytes=hashing.bytes,
                      sha256=hashing.digest.hexdigest())
        if common_memo is None:
            memo = pickler.memo.copy()
            result['memo_entries'] = len(memo)
            return result, memo
        result.update(transport_format=FORMAT, common_memo_entries=len(common_memo),
                      native_pickle_memo_lookup=True)
        return result


def _open_authenticated(descriptor):
    root = Path(descriptor['owned_transport_root']).resolve()
    path = Path(descriptor['path']).resolve()
    if path.parent != root or path.name != descriptor['owned_name']:
        raise SharedActorGeometryPayloadError('Owned actor payload path changed')
    input_file = path.open('rb')
    try:
        digest = hashlib.sha256()
        size = 0
        while True:
            value = input_file.read(1024 * 1024)
            if not value:
                break
            digest.update(value)
            size += len(value)
        if size != descriptor['bytes'] or digest.hexdigest() != descriptor['sha256']:
            raise SharedActorGeometryPayloadError('Complete current actor payload SHA changed')
        input_file.seek(0)
        return input_file
    except BaseException:
        input_file.close()
        raise


def _load_actor(descriptor):
    try:
        from .native_memo_actor_transport import load_common_then_private, FORMAT
        if descriptor['private'].get('transport_format') != FORMAT:
            raise SharedActorGeometryPayloadError('Native memo transport schema changed')
        with _open_authenticated(descriptor['common']) as common_input:
            with _open_authenticated(descriptor['private']) as private_input:
                geometry, block, common_entries = load_common_then_private(common_input, private_input)
        if common_entries != descriptor['common']['memo_entries']:
            raise SharedActorGeometryPayloadError('Common geometry memo inventory changed')
        if block.geometry is not geometry:
            raise SharedActorGeometryPayloadError('Actor geometry root identity changed')
        expected = descriptor['original_aliases']
        if expected['role_binding_is_geometry_binding']:
            if (block._original_motion_far_board_role_binding
                    is not geometry._original_motion_far_board_role_binding):
                raise SharedActorGeometryPayloadError('Actor role binding owner identity changed')
        if expected['finite_records_partial_owns_geometry']:
            if geometry.finite_board_records.args[0] is not geometry:
                raise SharedActorGeometryPayloadError('Finite record partial geometry identity changed')
        if expected['synchronize_partial_owns_minimal_block']:
            if block.synchronize_course.args[0] is not block:
                raise SharedActorGeometryPayloadError('Original minimal synchronization owner changed')
        return block
    except SharedActorGeometryPayloadError:
        raise
    except Exception as exc:
        # No source evaluation has run. The parent may retry the whole original
        # transport, where any genuine original source error still propagates.
        raise SharedActorGeometryPayloadError(
            'Current common actor payload hydrate failed: '+type(exc).__name__+': '+str(exc)) from exc


def _load_owned_actor(descriptor):
    from .source_assembly_gc_runtime import defer_automatic_source_gc
    from .role_block_binding_runtime import install_owned_actor_role_block_binding_runtime
    with defer_automatic_source_gc('owned_actor_hydration_and_role_view',
                                   descriptor.get('role_block_report_directory')):
        block = _load_actor(descriptor)
        install_owned_actor_role_block_binding_runtime(
            block, report_directory=descriptor.get('role_block_report_directory'))
        return block


def _compile_worker(source):
    # source._actor_worker is already dynamically compiled by the native
    # contract. Reuse the guarded compiler, including its ready capability.
    from .parallel_source_guard import compile_guarded_source_actor
    return compile_guarded_source_actor(payload_loader=_load_owned_actor)


def run_shared_geometry_source_actor(connection, descriptor, task_root,
                                     native_threads, dynamic_modules, spec):
    """Original actor body, with its one load replaced by private hydration."""
    global _WORKER
    try:
        # Import the same project-owned finder/backend contract as the original
        # file actor before touching any legacy type in either pickle stream.
        from . import worker_entry
        from .absolute_material_value_runtime import install_absolute_material_value_runtime
        from .finite_extent_guard_reuse_runtime import install_finite_extent_guard_reuse_runtime
        from .role_record_index_runtime import install_role_record_index_runtime
        from .role_binding_value_reuse_runtime import install_role_binding_value_reuse_runtime
        install_absolute_material_value_runtime()
        install_finite_extent_guard_reuse_runtime()
        install_role_record_index_runtime()
        install_role_binding_value_reuse_runtime()
        if _WORKER is None:
            import z190_process_source_evaluation as source
            _WORKER = _compile_worker(source)
    except Exception as exc:
        try:
            connection.send(dict(status='error', type='SharedActorGeometryPayloadError',
                message='Shared actor bootstrap failed: '+type(exc).__name__+': '+str(exc),
                traceback=''))
        finally:
            connection.close()
        return
    # The unchanged original body owns all SHM/profile/r/J/review operations,
    # original ordered domain errors and BaseException propagation thereafter.
    return _WORKER(connection, descriptor, task_root, native_threads, dynamic_modules, spec)


def _original_aliases(block):
    geometry = block.geometry
    binding = getattr(block, '_original_motion_far_board_role_binding', None)
    finite = getattr(geometry, 'finite_board_records', None)
    return dict(
        role_binding_is_geometry_binding=(binding is not None and binding is
            getattr(geometry, '_original_motion_far_board_role_binding', None)),
        finite_records_partial_owns_geometry=(isinstance(finite, functools.partial)
            and bool(finite.args) and finite.args[0] is geometry),
        synchronize_partial_owns_minimal_block=hasattr(block, '_original_motion_far_board_role_binding'))


def _compile_minimal(original, dump):
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    replaced = []

    class Dump(ast.NodeTransformer):
        def visit_Call(self, node):
            if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == 'pickle' and node.func.attr == 'dumps'):
                if ast.unparse(node) != 'pickle.dumps(small, protocol=5)':
                    raise SharedActorGeometryPayloadError('Original minimal pickle protocol changed')
                replaced.append(True)
                node.func = ast.Name(id='_rinkspace_dump_private_minimal', ctx=ast.Load())
            return self.generic_visit(node)

    tree = Dump().visit(tree)
    if len(replaced) != 1:
        raise SharedActorGeometryPayloadError('Original minimal pickle site changed')
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    namespace['_rinkspace_dump_private_minimal'] = dump
    exec(compile(tree, __file__+':original_minimal_shared_geometry', 'exec'), namespace)
    return namespace['_minimal_pickle'], hashlib.sha256(source.encode()).hexdigest()


def install_shared_actor_geometry_payload_runtime(*, report_directory):
    """Install after the existing exact-file actor transport; no pool starts."""
    import z190_process_source_evaluation as source
    owner = source.PersistentSourceActors
    previous = getattr(owner, '_rinkspace_shared_geometry_payload_receipt', None)
    if previous is not None:
        return previous
    destination = Path(report_directory).resolve()
    transport = destination / 'transient_owned_payloads'
    transport.mkdir(parents=True, exist_ok=True)
    original_start = owner.start
    original_minimal = source._minimal_pickle
    prepared = ContextVar('current_source_actor_common_geometry', default=None)

    def dump_private(value, *, protocol):
        context = prepared.get()
        if context is None or protocol != 5:
            raise SharedActorGeometryPayloadError('Private actor pickle requires current owned scope')
        path = transport / (context['prefix']+'-'+uuid.uuid4().hex+'.private.pickle')
        context['paths'].append(path)
        return _write_pickle(path, value, common_memo=context['memo'])

    minimal, minimal_sha = _compile_minimal(original_minimal, dump_private)
    tree_source = textwrap.dedent(inspect.getsource(original_start))
    tree = ast.parse(tree_source)
    replaced = []

    class Start(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == '_minimal_pickle':
                replaced.append('prepared_private_payload')
                node.func = ast.Name(id='_rinkspace_prepared_private_payload', ctx=ast.Load())
            return self.generic_visit(node)

        def visit_ImportFrom(self, node):
            if (node.module == 'calibration.solver_runtime.worker_entry'
                    and len(node.names) == 1 and node.names[0].name == 'run_source_actor'):
                replaced.append('private_common_geometry_worker')
                return ast.Assign(targets=[ast.Name(id='run_source_actor', ctx=ast.Store())],
                    value=ast.Name(id='_rinkspace_shared_geometry_actor', ctx=ast.Load()))
            return node

    tree = Start().visit(tree)
    if sorted(replaced) != ['prepared_private_payload', 'private_common_geometry_worker']:
        raise SharedActorGeometryPayloadError('Original actor spawn sites changed: '+str(replaced))
    ast.fix_missing_locations(tree)

    def private_payload(block):
        context = prepared.get()
        if context is None or id(block) not in context['blocks']:
            raise SharedActorGeometryPayloadError('Current original actor block inventory changed')
        return context['blocks'][id(block)]

    namespace = dict(inspect.unwrap(original_start).__globals__)
    namespace.update(_rinkspace_prepared_private_payload=private_payload,
                     _rinkspace_shared_geometry_actor=run_shared_geometry_source_actor)
    exec(compile(tree, __file__+':original_start_shared_geometry', 'exec'), namespace)
    shared_start = namespace['start']

    def record(report):
        report['wall_time_ns'] = time.time_ns()
        with (destination/'shared_actor_geometry_execution.jsonl').open('a', encoding='utf8') as output:
            output.write(json.dumps(report, sort_keys=True)+'\n')

    @functools.wraps(original_start)
    def start(self):
        if self.closed or self.started or not self.blocks:
            return original_start(self)
        if self._actors or self._busy:
            return original_start(self)
        context = dict(prefix=uuid.uuid4().hex, paths=[], blocks={}, memo=None)
        token = prepared.set(context)
        begin = time.perf_counter()
        report = dict(route='current_common_geometry_private_actors',
                      original_blocks=[block.name for block in self.blocks],
                      cross_q_or_cross_start_reuse=False, geometry_common_bytes=None,
                      private_actor_bytes=[], original_numeric_payload_pruned=False)
        try:
            try:
                from prepared_cache_helpers import install_mappingproxy_pickle
                install_mappingproxy_pickle()
                geometry = self.blocks[0].geometry
                if any(block.geometry is not geometry for block in self.blocks):
                    raise SharedActorGeometryPayloadError('Original source actors do not share one geometry root')
                common_path = transport / (context['prefix']+'.common.pickle')
                context['paths'].append(common_path)
                common, memo = _write_pickle(common_path, geometry)
                context['memo'] = memo
                report['geometry_common_bytes'] = common['bytes']
                report['geometry_common_sha256'] = common['sha256']
                for block in self.blocks:
                    private, modules = minimal(block)
                    context['blocks'][id(block)] = (dict(common=common, private=private,
                        original_aliases=_original_aliases(block),
                        role_block_report_directory=str(destination/'role_block_binding')), modules)
                    report['private_actor_bytes'].append(dict(block=block.name, **private))
                # No serialization memo or extra common geometry graph is held
                # during child startup. Each child owns its independent load.
                context['memo'] = None
                del memo
                report['parent_pack_seconds'] = time.perf_counter()-begin
            except Exception as exc:
                report.update(route='original_file_actor_transport',
                    fallback_stage='current_common_geometry_packing',
                    fallback_reason=type(exc).__name__+': '+str(exc))
                return original_start(self)
            try:
                result = shared_start(self)
            except Exception as exc:
                # The original start closes/drains only this manager's owned
                # children. Retry only a new transport failure before any
                # numerical actor request, never a later source-domain error.
                transport_failure = (isinstance(exc, source.SourceActorTransportError)
                    or 'SharedActorGeometryPayloadError' in str(exc))
                if not transport_failure:
                    raise
                self.close()
                if self._actors or self.started or self._busy:
                    raise SharedActorGeometryPayloadError('Failed shared actor ownership not drained') from exc
                self.closed = False
                report.update(route='original_file_actor_transport',
                    fallback_stage='current_common_geometry_worker_startup',
                    fallback_reason=type(exc).__name__+': '+str(exc))
                return original_start(self)
            report.update(succeeded=True, independent_private_geometry_per_actor=True,
                original_worker_ready_profiles_SHM_and_source_evaluation_retained=True)
            return result
        finally:
            prepared.reset(token)
            for path in context['paths']:
                if path.parent != transport:
                    raise SharedActorGeometryPayloadError('Owned shared actor payload escaped its root')
                path.unlink(missing_ok=True)
            report['start_seconds'] = time.perf_counter()-begin
            record(report)

    owner.start = start
    receipt = dict(schema='rinkspace.current-actor-common-geometry-private-transport.v1',
        original_start_sha256=hashlib.sha256(tree_source.encode()).hexdigest(),
        original_minimal_clone_sha256=minimal_sha,
        one_current_geometry_serialization_per_start=True,
        every_worker_unpickles_independent_complete_geometry=True,
        common_memo_references_preserve_original_role_partial_owner_identity=True,
        native_C_pickle_memo_replaces_per_object_Python_persistent_callbacks=True,
        complete_current_common_and_private_payload_SHA_checked=True,
        original_numeric_arrays_records_caches_and_current_q_not_pruned=True,
        original_pickle_byte_layout_changed=True,
        source_math_guard_error_order_profiles_rows_and_SHM_unchanged=True,
        private_owned_block_role_view_installed_only_after_child_hydration=True,
        packing_or_transport_integration_failure_uses_original_file_transport=True,
        cross_q_or_cross_start_reuse=False, registered_source_files_modified=False,
        transient_transport_root=str(transport))
    owner._rinkspace_shared_geometry_payload_receipt = receipt
    (destination/'shared_actor_geometry_payload_receipt.json').write_text(
        json.dumps(receipt, sort_keys=True, indent=2)+'\n', encoding='utf8')
    return receipt
