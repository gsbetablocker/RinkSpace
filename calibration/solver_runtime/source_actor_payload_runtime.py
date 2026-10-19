"""Transport unchanged actor pickle bytes outside Windows spawn's startup pipe."""
from __future__ import annotations

import ast
from contextvars import ContextVar
import functools
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import uuid


def run_file_source_actor(connection, descriptor, task_root, native_threads, dynamic_modules, spec):
    """Read the exact original owned payload before entering the original worker."""
    try:
        root = Path(descriptor['owned_transport_root']).resolve()
        path = Path(descriptor['path']).resolve()
        if path.parent != root or path.name != descriptor['owned_name']:
            raise ValueError('Source actor transport path changed')
        payload = path.read_bytes()
        if len(payload) != descriptor['bytes'] or hashlib.sha256(payload).hexdigest() != descriptor['sha256']:
            raise ValueError('Original source actor pickle bytes changed in transit')
        # This file exists only to replace the original transient spawn pipe.
        # It is a fresh UUID file belonging to this one worker, not a checkpoint.
        path.unlink()
        # Windows spawn does not inherit class execution wrappers from the
        # parent. Install the same exact-value reuse before the original entry
        # unpickles and evaluates its private programme block.
        from .absolute_material_value_runtime import install_absolute_material_value_runtime
        from .finite_extent_guard_reuse_runtime import install_finite_extent_guard_reuse_runtime
        from .role_record_index_runtime import install_role_record_index_runtime
        from .role_binding_value_reuse_runtime import install_role_binding_value_reuse_runtime
        install_absolute_material_value_runtime()
        install_finite_extent_guard_reuse_runtime()
        install_role_record_index_runtime()
        install_role_binding_value_reuse_runtime()
        descriptor['original_actor_entry'](
            connection, payload, task_root, native_threads, dynamic_modules, spec)
    except BaseException:
        # Let the original parent's transport failure report the failed worker.
        connection.close()
        raise


def install_source_actor_payload_runtime(*, report_directory):
    from z190_process_source_evaluation import PersistentSourceActors
    from .worker_entry import run_source_actor
    previous = getattr(PersistentSourceActors, '_rinkspace_file_payload_receipt', None)
    if previous is not None:
        return previous
    destination = Path(report_directory).resolve()
    transport = destination / 'transient_owned_payloads'
    transport.mkdir(parents=True, exist_ok=True)
    original = PersistentSourceActors.start
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    changes = []

    class Transport(ast.NodeTransformer):
        def visit_Assign(self, node):
            if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == '_minimal_pickle'
                    and ast.unparse(node.targets[0]) == '(payload, modules)'):
                changes.append('stage_original_pickle_bytes')
                node.value = ast.Call(func=ast.Name(id='_rinkspace_stage_payload', ctx=ast.Load()),
                                      args=[node.value], keywords=[])
            return self.generic_visit(node)

        def visit_ImportFrom(self, node):
            if (node.module == 'calibration.solver_runtime.worker_entry'
                    and len(node.names) == 1 and node.names[0].name == 'run_source_actor'):
                changes.append('original_worker_with_exact_file_payload')
                return ast.Assign(targets=[ast.Name(id='run_source_actor', ctx=ast.Store())],
                                  value=ast.Name(id='_rinkspace_file_actor', ctx=ast.Load()))
            return node

    tree = Transport().visit(tree)
    if sorted(changes) != ['original_worker_with_exact_file_payload', 'stage_original_pickle_bytes']:
        raise ValueError('Original source actor spawn sites changed: '+str(changes))
    ast.fix_missing_locations(tree)
    staged_paths = ContextVar('source_actor_staged_payloads', default=None)

    def stage(original_pair):
        payload, modules = original_pair
        name = uuid.uuid4().hex + '.pickle'
        path = transport / name
        owned = staged_paths.get()
        if owned is None:
            raise RuntimeError('Source actor payload staging requires an owned start scope')
        owned.append(path)
        with path.open('xb') as output:
            output.write(payload)
        descriptor = dict(path=str(path), owned_transport_root=str(transport), owned_name=name,
                          bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                          original_actor_entry=run_source_actor)
        return descriptor, modules

    namespace = dict(original.__globals__)
    namespace.update(_rinkspace_stage_payload=stage, _rinkspace_file_actor=run_file_source_actor)
    exec(compile(tree, __file__+':original_start_file_transport', 'exec'), namespace)
    file_start = namespace['start']

    @functools.wraps(original)
    def start(self):
        owned = []
        token = staged_paths.set(owned)
        try:
            return file_start(self)
        finally:
            staged_paths.reset(token)
            # Original start waits for every ready reply, or closes its owned
            # children on failure. No child needs its transport file now.
            for path in owned:
                if path.parent != transport:
                    raise RuntimeError('Owned source actor transport path escaped its root')
                path.unlink(missing_ok=True)

    PersistentSourceActors.start = start
    receipt = dict(schema='rinkspace.original-actor-pickle-file-transport.v1',
        original_start_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_pickle_bytes_and_original_worker_entry_retained=True,
        all_original_start_order_SHM_profiles_receive_cleanup_retained=True,
        transient_transport_root=str(transport), child_authenticates_complete_payload_SHA=True,
        actor_original_material_value_reuse_installed=True,
        original_actor_entry=run_source_actor.__module__+'.'+run_source_actor.__qualname__,
        parent_cleans_only_this_start_transient_payloads_on_success_or_failure=True,
        cross_state_payload_reuse=False, registered_source_files_modified=False)
    PersistentSourceActors._rinkspace_file_payload_receipt = receipt
    (destination/'source_actor_payload_receipt.json').write_text(
        json.dumps(receipt, indent=2)+'\n', encoding='utf8')
    return receipt
