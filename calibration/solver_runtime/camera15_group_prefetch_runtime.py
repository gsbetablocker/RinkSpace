"""Compute original absolute camera15 and FD spans on the same small jobs.

This is an opt-in, per-family execution hook. No bundle, motion arrays, full
programme J, numerical checkpoint or previously computed derivative is sent
to a worker. The existing completion scheduler and material bank are reused.
"""
from __future__ import annotations

import ast
from concurrent.futures import CancelledError
from contextvars import ContextVar, copy_context
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import struct
import textwrap
import threading
import time

import numpy as np

_SCOPE = ContextVar('rinkspace_camera15_group_prefetch', default=None)
_RECEIPT = None
_REPORT = None
_ORIGINAL_ORDERED = None
_ORIGINAL_CAPTURE = None


def _literal_digest(literal):
    """Value-only digest: pickle memo/alias layouts must not affect admission."""
    result = hashlib.sha256()
    def put(value):
        if type(value) is tuple:
            result.update(b't'+struct.pack('<Q', len(value)))
            for item in value:
                put(item)
        elif type(value) is bytes:
            result.update(b'b'+struct.pack('<Q', len(value))+value)
        elif type(value) is str:
            data = value.encode('utf8')
            result.update(b's'+struct.pack('<Q', len(data))+data)
        elif value is None:
            result.update(b'n')
        elif type(value) in (int, bool):
            result.update(b'i'+str(value).encode('ascii')+b';')
        else:
            raise TypeError('Unsupported exact camera source literal')
    put(literal)
    return result.hexdigest()


def _error_marker(error):
    """Transport diagnostics only; original parent mathematics owns errors."""
    kind = type(error).__module__+'.'+type(error).__qualname__
    try:
        message = str(error)
    except Exception:
        message = 'Original worker exception text unavailable'
    return dict(type=kind, message=message)


def evaluate_camera15_geometry_batch(batch):
    """Picklable, small original-group job on an existing FD worker."""
    from . import absolute_geometry_fd_pool_candidate as source
    from . import absolute_material_value_runtime as material
    from .confirmed_constraint_columns_runtime import _private_material_owner
    from .camera_absolute_reuse_runtime import _literal, _UNSUPPORTED
    source.initialize_absolute_geometry_worker()
    geometry = _private_material_owner(batch['material_owner'])
    bank_token = material._BANK_SCOPE.set(batch.get('_rinkspace_shared_material_bank'))
    common = {key:value for key,value in batch.items() if key != 'groups'}
    results = []
    fd_failed = False
    try:
        for group in batch['groups']:
            job = dict(common, **group)
            nominal_result = nominal_error = nominal_digest = None
            nominal_scope = dict(job=job, geometry=geometry, root_world_bytes=None,
                shared_bank=batch.get('_rinkspace_shared_material_bank'), hits=0,
                original_value_calls=0, uncacheable_results=0, root_cache_resets=0)
            token = material._SCOPE.set(nominal_scope)
            try:
                # The caller already performed its original nominal rebind.
                # A private rebind is admitted only by all literal input bytes.
                moved = [geometry.rebind(row, np.asarray(job['world66'], float))
                         for row in job['rows']]
                literal = _literal(moved)
                if literal is not _UNSUPPORTED:
                    nominal_digest = _literal_digest(literal)
                try:
                    nominal_result = job['absolute'](np.asarray(job['camera25'], float), moved, True)
                except Exception as error:
                    nominal_error = _error_marker(error)
            except Exception as error:
                # Unknown private integration cannot authorize a numerical
                # result. The caller executes the original absolute function.
                nominal_error = _error_marker(error)
                nominal_digest = None
            finally:
                material._SCOPE.reset(token)
            fd_error = None
            if not fd_failed:
                try:
                    result = source._evaluate_one_absolute_geometry_group(job)
                except Exception as error:
                    fd_error = _error_marker(error)
                    fd_failed = True
            else:
                # The original contiguous worker envelope stopped at its
                # first FD error. Later nominal observations still precede it
                # in the original bundle's execution order.
                result = None
            if fd_error is not None or result is None:
                result = dict(ordinal=job['ordinal'], actual_pts=job['actual_pts'],
                    source_group_sha256=job['source_group_sha256'],
                    geo=np.empty((0, 0)), stats=dict(actual_route='deferred_original_FD_error'))
            result.update(camera15_result=nominal_result, camera15_error=nominal_error,
                camera15_input_digest=nominal_digest, camera25_bytes=np.asarray(job['camera25'], float).tobytes(),
                deferred_FD_error=fd_error, FD_after_earlier_envelope_failure=fd_failed and fd_error is None)
            results.append(result)
        return results
    finally:
        material._BANK_SCOPE.reset(bank_token)


class _PrefetchExecutor:
    def __init__(self, executor, owner):
        self.executor, self.owner = executor, owner

    def submit(self, function, batch, *args, **kwargs):
        from . import absolute_geometry_fd_pool_candidate as source
        from . import absolute_material_value_runtime as material
        owned = threading.current_thread() is self.owner.thread
        if not owned:
            return self.executor.submit(function, batch, *args, **kwargs)
        if self.owner.abort.is_set():
            raise CancelledError('Original owned group prefetch was cancelled')
        if function in (source.evaluate_absolute_geometry_worker,
                        material.evaluate_absolute_geometry_with_shared_material_bank):
            function = evaluate_camera15_geometry_batch
        future = self.executor.submit(function, batch, *args, **kwargs)
        self.owner.futures.append(future)
        return future

    def __getattr__(self, name):
        return getattr(self.executor, name)


class _FamilyPrefetch:
    def __init__(self, owner, local, world, pool):
        self.owner, self.local, self.world, self.pool = owner, local, world.copy(), pool
        self.step = float(os.environ.get('RINKSPACE_GEOMETRY_FD_STEP_M', '1e-7'))
        self.condition = threading.Condition()
        self.results, self.futures = [], []
        self.finished = False
        self.error = None
        self.abort = threading.Event()
        self.nominal_position = 0
        self.hits = self.fallbacks = 0
        self.fd_parent_replays = 0
        self.started = time.perf_counter()
        self.had_executor = '_executor' in vars(pool)
        self.saved_executor = vars(pool).get('_executor')
        original_executor = pool._executor
        def executor():
            value = original_executor()
            return None if value is None else _PrefetchExecutor(value, self)
        pool._executor = executor
        context = copy_context()
        self.thread = threading.Thread(target=lambda:context.run(self._run),
            name='original-camera15-FD-feeder', daemon=True)
        try:
            self.thread.start()
        except BaseException:
            if self.had_executor:
                pool._executor = self.saved_executor
            else:
                vars(pool).pop('_executor', None)
            raise

    def _run(self):
        iterator = None
        try:
            groups = ((p, rows, self.owner.bundle.camera(self.local, p))
                      for p,rows in self.owner.groups)
            iterator = _ORIGINAL_ORDERED(self.pool, groups, self.world, self.owner.absolute,
                self.step,
                batched_projection=True, group_count=len(self.owner.groups))
            for result in iterator:
                with self.condition:
                    self.results.append(result)
                    self.condition.notify_all()
                if self.abort.is_set():
                    break
        except BaseException as error:
            self.error = error
        finally:
            try:
                if iterator is not None:
                    iterator.close()
            except BaseException as error:
                if self.error is None:
                    self.error = error
            finally:
                with self.condition:
                    self.finished = True
                    self.condition.notify_all()

    def get(self, position):
        with self.condition:
            self.condition.wait_for(lambda:position < len(self.results) or self.finished or self.abort.is_set())
            if self.abort.is_set():
                raise CancelledError('Original owned group prefetch was cancelled')
            return self.results[position] if position < len(self.results) else None

    def close(self, success):
        if not success:
            self.abort.set()
            with self.condition:
                self.condition.notify_all()
            # Wake the original completion wait with any not-yet-running
            # cancellations before joining its feeder. Already-running native
            # work still drains under the same original pool ownership.
            for future in tuple(self.futures):
                if not future.done():
                    future.cancel()
        self.thread.join()
        # The original scheduler's finally records state but does not own
        # cancellation. Drain exactly these submitted futures before restore.
        for future in self.futures:
            if not success and not future.done():
                future.cancel()
        for future in self.futures:
            try:
                future.result()
            except BaseException:
                pass
        if self.had_executor:
            self.pool._executor = self.saved_executor
        else:
            vars(self.pool).pop('_executor', None)
        if _REPORT is not None:
            event = dict(block=self.owner.name, groups=len(self.owner.groups),
                camera15_private_groups_admitted=self.hits, original_camera15_fallback_groups=self.fallbacks,
                original_parent_FD_error_replay_groups=self.fd_parent_replays,
                original_geometry_FD_spans_consumed=success, original_workers=self.pool.workers,
                no_bundle_motion_graph_or_full_J_transported=True,
                elapsed_seconds=time.perf_counter()-self.started, owned_futures_drained=True,
                feeder_error=None if self.error is None else type(self.error).__name__+': '+str(self.error))
            try:
                with (_REPORT/'camera15_group_prefetch.jsonl').open('a', encoding='utf8') as stream:
                    stream.write(json.dumps(event)+'\n')
            except OSError:
                pass
        self.results.clear()
        self.futures.clear()


def _begin(owner, local, world, jacobian):
    from .absolute_geometry_fd_pool_candidate import AbsoluteGeometryGroupPool
    pool = getattr(owner, '_absolute_geometry_fd_pool', None)
    if (not jacobian or _SCOPE.get() is not None or type(pool) is not AbsoluteGeometryGroupPool
            or pool.closed or not pool.owns_executor or pool.workers <= 1
            or pool.reserve_bytes != 0
            or pool.parent_geometry is None
            or getattr(pool, '_rinkspace_force_parent_activation_FD', False)
            or getattr(pool, '_rinkspace_active_geometry_group_iterators', 0)
            or owner.absolute is not owner.bundle.absolute_evaluator):
        return None
    if any(getattr(pool, key, 0) for key in ('_rinkspace_confirmed_columns_pending_jobs',
            '_rinkspace_native_columns_pending_jobs', '_rinkspace_shared_material_bank_pending_jobs')):
        return None
    try:
        value = _FamilyPrefetch(owner, local, np.asarray(world, float), pool)
    except Exception:
        # No mathematical producer has run in this constructor. Unsupported
        # thread/executor integration leaves the existing original path live.
        return None
    token = _SCOPE.set(value)
    return value, token


def _finish(value, success):
    if value is not None:
        scope, token = value
        try:
            scope.close(success)
        finally:
            _SCOPE.reset(token)


def _capture(bundle, camera, rows, jacobian):
    from . import camera_absolute_reuse_runtime as camera_runtime
    scope = _SCOPE.get()
    reuse = camera_runtime._SCOPE.get()
    if scope is None or not jacobian or bundle is not scope.owner.bundle or reuse is None:
        return _ORIGINAL_CAPTURE(bundle, camera, rows, jacobian)
    position = scope.nominal_position
    scope.nominal_position += 1
    result = scope.get(position)
    camera_key, row_key = camera_runtime._literal(np.asarray(camera)), camera_runtime._literal(rows)
    admitted = (result is not None and result.get('camera15_input_digest') is not None
        and result['camera25_bytes'] == np.asarray(camera, float).tobytes()
        and row_key is not camera_runtime._UNSUPPORTED
        and result['camera15_input_digest'] == _literal_digest(row_key))
    if not admitted:
        scope.fallbacks += 1
        return _ORIGINAL_CAPTURE(bundle, camera, rows, jacobian)
    if result['camera15_error'] is not None:
        # Some original exceptions have non-picklable constructor arguments.
        # Reproduce the mathematical operation at its original consume point,
        # rather than reconstruct an arbitrary exception object from transport.
        # If the private worker alone failed, the actual parent result is valid.
        scope.fallbacks += 1
        return _ORIGINAL_CAPTURE(bundle, camera, rows, jacobian)
    raw = result.pop('camera15_result')
    if (type(raw) is not tuple or len(raw) != 2
            or not all(isinstance(part, np.ndarray) and not part.dtype.hasobject for part in raw)):
        scope.fallbacks += 1
        return _ORIGINAL_CAPTURE(bundle, camera, rows, jacobian)
    scope.hits += 1
    reuse['results'].append((bundle.absolute_evaluator, camera_key, row_key,
                            tuple(part.copy() for part in raw)))
    return raw


def _geometry_spans(pool, groups, world, absolute, step, **kwargs):
    scope = _SCOPE.get()
    if (scope is None or pool is not scope.pool or absolute is not scope.owner.absolute
            or np.asarray(world, float).tobytes() != scope.world.tobytes()):
        yield from _ORIGINAL_ORDERED(pool, groups, world, absolute, step, **kwargs)
        return
    if float(step).hex() != scope.step.hex():
        raise ValueError('Original current FD width changed during group prefetch')
    iterator = iter(enumerate(groups))
    while True:
        envelope = []
        for _ in range(pool.groups_per_job):
            item = next(iterator, None)
            if item is None:
                break
            position, (stamp, rows, camera) = item
            result = scope.get(position)
            if result is None:
                if scope.error is not None:
                    raise scope.error
                raise ValueError('Original geometry group result omitted during prefetch')
            if (result['ordinal'] != position or result['actual_pts'] != stamp
                    or result['source_group_sha256'] != pool_source_digest(rows)
                    or result['camera25_bytes'] != np.asarray(camera, float).tobytes()):
                raise ValueError('Original FD group order/source/camera changed during prefetch')
            envelope.append((position, stamp, rows, camera, result))
        if not envelope:
            break
        # Preserve the original complete contiguous delivery envelope: a
        # later FD error may not publish any earlier prefix from that envelope.
        for position, stamp, rows, camera, result in envelope:
            if (result['deferred_FD_error'] is not None
                    or result['FD_after_earlier_envelope_failure']):
                from . import absolute_geometry_fd_pool_candidate as source
                job = dict(material_owner=pool.material_owner, ordinal=position,
                    actual_pts=int(stamp), rows=rows,
                    camera25=np.asarray(camera, float).copy(),
                    source_group_sha256=pool_source_digest(rows),
                    world66=scope.world.copy(), absolute=absolute, step=float(step),
                    batched_projection=bool(kwargs.get('batched_projection', True)))
                # This is the same original parent fallback function and owner;
                # it does not alter the parent's native backend. Exceptions
                # arise naturally before any prefix of the envelope is yielded.
                replayed = source._evaluate_group_on_owner(job, pool.parent_geometry,
                    pool.parent_certificates, 'original_error_parent_replay')
                scope.fd_parent_replays += 1
                # A private-only failure may succeed here. Fill every group
                # skipped by that worker chunk with actual original parent FD,
                # retaining the original report dictionary already registered
                # by the completion scheduler.
                stats = result['stats']
                stats.clear()
                stats.update(replayed['stats'])
                result.update(replayed, stats=stats, deferred_FD_error=None,
                    FD_after_earlier_envelope_failure=False)
        for _, _, _, _, result in envelope:
            yield result
            result['geo'] = None
    if scope.error is not None:
        raise scope.error


def pool_source_digest(rows):
    from .absolute_geometry_fd_pool_candidate import _source_group_digest
    return _source_group_digest(rows)


def install_camera15_group_prefetch_runtime(*, report_directory=None):
    """Install after material, completion and camera15 reuse, before a run."""
    global _RECEIPT, _REPORT, _ORIGINAL_ORDERED, _ORIGINAL_CAPTURE
    if _RECEIPT is not None:
        return _RECEIPT
    from .absolute_geometry_fd_pool_candidate import AbsoluteGeometryGroupPool
    from . import camera_absolute_reuse_runtime as camera_runtime
    from structured_programme_source_candidate.paired_programme_blocks import StructuredProgrammeBlock
    from z190_fixed_lever_xyz_source_bundle import FixedLeverXYZSourceBundle
    if camera_runtime._RECEIPT is None:
        raise TypeError('Exact original camera15 reuse must already be installed')
    _ORIGINAL_ORDERED = AbsoluteGeometryGroupPool.ordered_groups
    _ORIGINAL_CAPTURE = camera_runtime._capture
    original = StructuredProgrammeBlock.evaluate_camera_geometry
    if original.__code__.co_freevars:
        raise TypeError('Closure-free original programme camera/geometry caller required')
    declared = getattr(StructuredProgrammeBlock, '_rinkspace_original_geometry_group_count_caller', None)
    if declared is None:
        raise TypeError('Original completion scheduling caller identity required')
    from .completion_driven_absolute_fd_runtime import _with_explicit_group_count
    expected, _ = _with_explicit_group_count(declared)
    if expected.__code__ != original.__code__:
        raise TypeError('Installed original group-count caller changed')
    source = textwrap.dedent(inspect.getsource(declared))
    tree = ast.parse(source)
    function = tree.body[0]
    bindings = [i for i,node in enumerate(function.body) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == 'abs_groups' for target in node.targets)]
    tries = [node for node in function.body if isinstance(node, ast.Try)]
    if len(bindings) != 1 or len(tries) != 1:
        raise TypeError('Original nominal absolute rebind/finally boundary changed')
    calls = [node for node in ast.walk(tries[0]) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == 'ordered_groups'
        and isinstance(node.func.value, ast.Name) and node.func.value.id == 'pool']
    if len(calls) != 1:
        raise TypeError('Exactly one original geometry group consumption required')
    call = calls[0]
    call.func = ast.Name('_rinkspace_prefetched_camera15_geometry_spans', ast.Load())
    call.args.insert(0, ast.Name('pool', ast.Load()))
    if not any(item.arg == 'group_count' for item in call.keywords):
        call.keywords.append(ast.keyword(arg='group_count',
            value=ast.parse('len(self.groups)', mode='eval').body))
    function.body[bindings[0]+1:bindings[0]+1] = ast.parse(
        '_rinkspace_camera_prefetch = _rinkspace_start_camera15_prefetch(self, local, world, jacobian)\n'
        '_rinkspace_camera_prefetch_success = False').body
    # Every normal return through the original try marks successful FD use.
    class MarkReturns(ast.NodeTransformer):
        def visit_Return(self, node):
            return [ast.parse('_rinkspace_camera_prefetch_success = True').body[0], node]
    tries[0].body = MarkReturns().visit(ast.Module(body=tries[0].body, type_ignores=[])).body
    tries[0].finalbody.append(ast.parse(
        '_rinkspace_finish_camera15_prefetch(_rinkspace_camera_prefetch, _rinkspace_camera_prefetch_success)').body[0])
    ast.fix_missing_locations(tree)
    globals_ = original.__globals__
    globals_.update(_rinkspace_start_camera15_prefetch=_begin,
        _rinkspace_finish_camera15_prefetch=_finish,
        _rinkspace_prefetched_camera15_geometry_spans=_geometry_spans)
    namespace = {}
    exec(compile(tree, inspect.getsourcefile(original), 'exec'), globals_, namespace)
    original.__code__ = namespace[original.__name__].__code__
    capture_globals = FixedLeverXYZSourceBundle.evaluate.__globals__
    if capture_globals.get('_rinkspace_capture_camera15') is not _ORIGINAL_CAPTURE:
        raise TypeError('Original admitted camera15 capture call changed')
    capture_globals['_rinkspace_capture_camera15'] = _capture
    if report_directory is not None:
        _REPORT = Path(report_directory).resolve()
        _REPORT.mkdir(parents=True, exist_ok=True)
    _RECEIPT = dict(schema='rinkspace.original-camera15-and-geometry-group-prefetch.v1',
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_camera_geometry_caller_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_workers_and_completion_scheduler_retained=True,
        original_bundle_motion_chunk_boundaries_CSR_mapping_and_math_unchanged=True,
        camera25_and_complete_nominal_row_literals_checked=True,
        nominal_and_FD_errors_delivered_at_original_consumption_stages=True,
        no_full_bundle_motion_graph_or_full_source_J_transported=True,
        actual_complete_same_q_source_bytes_and_speedup_pending=True)
    return _RECEIPT
