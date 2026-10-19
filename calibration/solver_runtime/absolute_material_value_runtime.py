"""Exact original material values within original absolute-FD group execution.

Only repeated _material_value results are reused. Original complete-state
validation, FD stencils, source qualification and companion rebind stay active.
Caches are weakly owned outside geometry pickles and clear on root-world change.
"""
from __future__ import annotations

from contextvars import ContextVar
from concurrent.futures import FIRST_COMPLETED, wait
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import time
import uuid
import weakref

import numpy as np


_SCOPE = ContextVar('rinkspace_original_absolute_material_value_scope', default=None)
_VALUES = weakref.WeakKeyDictionary()
_ORIGINAL_GROUP = None
_INSTALLED = False
_RECEIPT = None
_REPORT_DIRECTORY = None
_UNSUPPORTED = object()
_BANK_SCOPE = ContextVar('rinkspace_shared_absolute_material_bank', default=None)
_BINDING_KEYS = weakref.WeakKeyDictionary()
_MAPPED_BANKS = {}
_ORIGINAL_ORDERED_GROUPS = None
_ORIGINAL_POOL_CLOSE = None


def _root_bytes(scope):
    if scope['root_world_bytes'] is None:
        world = scope['root_world_factory']() if scope.get('root_world_factory') is not None else scope['job']['world66']
        scope['root_world_bytes'] = np.asarray(world, float).tobytes()
    return scope['root_world_bytes']


def _binding_keys(geometry):
    previous = _BINDING_KEYS.get(geometry)
    bindings = geometry.bindings
    if previous is None or previous[0] is not bindings or previous[1] != len(bindings):
        previous = (bindings, len(bindings), {id(value): (value, pickle.dumps(key, protocol=5))
                                            for key, value in bindings.items()})
        _BINDING_KEYS[geometry] = previous
    return previous[2]


def _file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _shared_pack(owned, stream):
    """Store exactly owned numeric bytes, including original shape/strides."""
    kind = owned[0]
    if kind == 'array':
        array = owned[1]
        if type(array) is not np.ndarray:
            return _UNSUPPORTED
        align = max(1, array.dtype.alignment)
        stream.write(b'\0' * ((-stream.tell()) % align))
        offset = stream.tell()
        stream.write(array.ravel(order='K').tobytes())
        return ('array', offset, array.dtype, array.shape, array.strides, owned[2], array.nbytes)
    if kind in ('tuple', 'list'):
        entries = [_shared_pack(part, stream) for part in owned[1]]
        return _UNSUPPORTED if any(part is _UNSUPPORTED for part in entries) else (kind, entries)
    return owned


def _shared_unpack(packed, mapping):
    kind = packed[0]
    if kind == 'array':
        _, offset, dtype, shape, strides, writable, size = packed
        if offset < 0 or offset + size > mapping.nbytes:
            raise ValueError('Shared original material numeric span differs')
        array = np.ndarray(shape, dtype=dtype, buffer=mapping, offset=offset, strides=strides)
        array.flags.writeable = False
        return ('array', array, writable)
    if kind in ('tuple', 'list'):
        return (kind, [_shared_unpack(part, mapping) for part in packed[1]])
    return packed


def _load_shared_bank(descriptor):
    key = (descriptor['header_path'], descriptor['header_sha256'])
    if key not in _MAPPED_BANKS:
        header_path, data_path = Path(descriptor['header_path']), Path(descriptor['data_path'])
        if (_file_sha(header_path) != descriptor['header_sha256']
                or data_path.stat().st_size != descriptor['data_bytes']
                or _file_sha(data_path) != descriptor['data_sha256']):
            raise ValueError('Immutable shared original material bank changed')
        header = pickle.loads(header_path.read_bytes())
        if (header['root_world_bytes'] != descriptor['root_world_bytes']
                or header['material_owner_sha256'] != descriptor['material_owner_sha256']):
            raise ValueError('Shared original material root/producer differs')
        mapping = np.memmap(data_path, dtype=np.uint8, mode='r')
        entries = {(material, world): _shared_unpack(value, mapping)
                   for material, world, value in header['entries']}
        _MAPPED_BANKS[key] = dict(mapping=mapping, entries=entries)
    return _MAPPED_BANKS[key]


def _shared_material(scope, geometry, binding, sample_world):
    descriptor = scope.get('shared_bank')
    if descriptor is None or descriptor['root_world_bytes'] != _root_bytes(scope):
        return _UNSUPPORTED
    try:
        pair = _binding_keys(geometry).get(id(binding))
        if pair is None or pair[0] is not binding:
            return _UNSUPPORTED
        bank = _load_shared_bank(descriptor)
        owned = bank['entries'].get((pair[1], np.asarray(sample_world, float).tobytes()))
        if owned is None:
            return _UNSUPPORTED
        scope['shared_hits'] = scope.get('shared_hits', 0) + 1
        return _copy_out(owned)
    except Exception as exc:
        # A missing/incompatible execution cache cannot remove the original
        # numerical producer or substitute a success for an original failure.
        scope['shared_bank_error'] = dict(type=type(exc).__name__, message=str(exc))
        scope['shared_bank'] = None
        return _UNSUPPORTED


def prepare_absolute_material_sample_values(job):
    """One unique world job; original values for all private bindings once."""
    initialize_absolute_material_value_worker()
    from . import absolute_geometry_fd_pool_candidate as source
    snapshot = job['material_owner']
    identity = (snapshot['path'], snapshot['sha256'])
    if identity not in source._PRIVATE_GEOMETRIES:
        if source._sha(snapshot['path']) != snapshot['sha256']:
            raise ValueError('Private original material owner changed before shared value preparation')
        geometry = pickle.loads(Path(snapshot['path']).read_bytes())
        source._PRIVATE_GEOMETRIES[identity] = (geometry, set())
    geometry, _ = source._PRIVATE_GEOMETRIES[identity]
    original = type(geometry)._rinkspace_original_material_value
    begin = time.perf_counter()
    world = geometry._q(np.asarray(job['sample_world'], float))
    world_bytes = np.asarray(world, float).tobytes()
    if world_bytes != job['sample_world_bytes']:
        raise ValueError('Parent/private original effective material world differs')
    entries, failed = [], []
    calls, unsupported = 0, 0
    for key, binding in geometry.bindings.items():
        material = pickle.dumps(key, protocol=5)
        if material in job['already_prepared_material_keys']:
            continue
        try:
            calls += 1
            owned = _owned_value(original(geometry, binding, world))
            if owned is _UNSUPPORTED:
                unsupported += 1
                continue
            entries.append((material, world_bytes, owned))
        except Exception as exc:
            # Preserve the original row/operator error at original execution.
            # This is a speculative cache fill, never an admission decision.
            failed.append(dict(material_key_sha256=hashlib.sha256(material).hexdigest(),
                               type=type(exc).__name__, message=str(exc)))
    return dict(entries=entries, failed=failed, pid=os.getpid(),
                original_value_calls=calls, unsupported_values=unsupported, seconds=time.perf_counter()-begin)


def evaluate_absolute_geometry_with_shared_material_bank(batch):
    """Qualified spawn target, also installs scopes in a borrowed executor."""
    initialize_absolute_material_value_worker()
    from . import absolute_geometry_fd_pool_candidate as source
    return source.evaluate_absolute_geometry_worker(batch)


class _BankExecutor:
    def __init__(self, executor, bank):
        self.executor, self.bank = executor, bank

    def submit(self, function, batch, *args, **kwargs):
        from . import absolute_geometry_fd_pool_candidate as source
        if function is source.evaluate_absolute_geometry_worker:
            batch = dict(batch, _rinkspace_shared_material_bank=self.bank)
            function = evaluate_absolute_geometry_with_shared_material_bank
        return self.executor.submit(function, batch, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.executor, name)


def _bank_identity(pool, world, step):
    return (pool.material_owner['sha256'], np.asarray(world, float).tobytes(), float(step).hex())


def _prepare_shared_material_bank(pool, world, step):
    """Prepare the finite original stencil working set using the SAME pool."""
    if bool(getattr(pool, '_rinkspace_force_parent_activation_FD', False)):
        return None
    identity = _bank_identity(pool, world, step)
    existing = getattr(pool, '_rinkspace_shared_material_bank', None)
    if existing is not None and existing['identity'] == identity:
        return existing
    if _REPORT_DIRECTORY is None or pool.parent_geometry is None:
        return None
    if getattr(pool, '_rinkspace_active_geometry_group_iterators', 0):
        return None
    begin = time.perf_counter()
    geometry = pool.parent_geometry
    from structured_programme_source_candidate.batched_absolute_geometry_fd import _feasible_world_stencil
    root = np.asarray(world, float)
    if root.shape != (geometry.parameter_count,) or not np.isfinite(root).all() or not np.isfinite(step) or step <= 0:
        return None
    samples, failed_stencils = {}, []
    def add(sample):
        value = geometry._q(sample)
        samples.setdefault(np.asarray(value, float).tobytes(), np.asarray(sample, float).copy())
    add(root)
    for column in range(geometry.parameter_count):
        try:
            points, _, _ = _feasible_world_stencil(geometry, root, column, step)
            for sample in points:
                add(sample)
        except Exception as exc:
            failed_stencils.append(dict(column=column, type=type(exc).__name__, message=str(exc)))
    executor = pool._executor()
    if executor is None:
        return None
    pool._rinkspace_shared_material_bank_precompute_active = True
    pool._rinkspace_shared_material_bank_pending_jobs = 0
    report = dict(schema='rinkspace.pool-once-original-material-stencil-values.v1',
        root_world_sha256=hashlib.sha256(identity[1]).hexdigest(), material_owner_sha256=identity[0],
        actual_FD_step=float(step), unique_effective_worlds=len(samples), failed_stencils=failed_stencils,
        original_material_calls=0, unsupported_or_failed_values=0, parent_seed_values=0,
        worker_jobs=0, executor_identity=id(executor), every_original_source_row_and_certificate_check_retained=True,
        only_original_material_value_results_shared=True, no_complete_source_graph_constructed=True)
    directory = _REPORT_DIRECTORY / 'shared_material_values'
    directory.mkdir(parents=True, exist_ok=True)
    name = 'owned_material_' + uuid.uuid4().hex
    data_path, header_path = directory / (name + '.bin'), directory / (name + '.pickle')
    owned_paths = getattr(pool, '_rinkspace_owned_material_bank_paths', [])
    owned_paths.extend((data_path, header_path))
    pool._rinkspace_owned_material_bank_paths = owned_paths
    entries, prepared, material_keys = [], {}, {}
    def intern_material(key):
        return material_keys.setdefault(key, key)
    pending = {}
    result = None
    try:
        with data_path.open('xb') as stream:
            # Existing parent nominal results came from the same original
            # producer already; include them without calculating them again.
            cache = _VALUES.get(geometry)
            if cache is not None and cache['root_world_bytes'] == identity[1]:
                binding_ids = _binding_keys(geometry)
                for (binding_id, sample_bytes), (binding, owned) in cache['entries'].items():
                    pair = binding_ids.get(binding_id)
                    if sample_bytes not in samples or pair is None or pair[0] is not binding:
                        continue
                    packed = _shared_pack(owned, stream)
                    if packed is not _UNSUPPORTED:
                        entries.append((intern_material(pair[1]), sample_bytes, packed))
                        prepared.setdefault(sample_bytes, set()).add(pair[1])
                        report['parent_seed_values'] += 1
            tasks = iter(samples.items())
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < pool.max_inflight:
                    if not pool._headroom('before_shared_material_value_submit'):
                        raise RuntimeError('Original explicit pool headroom refused shared value preparation')
                    item = next(tasks, None)
                    if item is None:
                        exhausted = True
                        break
                    sample_bytes, raw_sample = item
                    job = dict(material_owner=pool.material_owner, sample_world=raw_sample,
                               sample_world_bytes=sample_bytes,
                               already_prepared_material_keys=prepared.get(sample_bytes, set()))
                    future = executor.submit(prepare_absolute_material_sample_values, job)
                    pending[future] = sample_bytes
                    pool._rinkspace_shared_material_bank_pending_jobs = len(pending)
                    report['worker_jobs'] += 1
                if not pending:
                    continue
                completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in completed:
                    sample_bytes = pending.pop(future)
                    value = future.result()
                    report['original_material_calls'] += value['original_value_calls']
                    report['unsupported_or_failed_values'] += len(value['failed']) + value.get('unsupported_values', 0)
                    for material, actual_world, owned in value['entries']:
                        if actual_world != sample_bytes:
                            raise ValueError('Shared worker material result world identity differs')
                        packed = _shared_pack(owned, stream)
                        if packed is not _UNSUPPORTED:
                            entries.append((intern_material(material), sample_bytes, packed))
                    pool._rinkspace_shared_material_bank_pending_jobs = len(pending)
            # memmap cannot map an empty file; an empty bank simply falls back.
            if not entries or stream.tell() == 0:
                raise ValueError('No original owned numeric material values available for sharing')
        header = dict(root_world_bytes=identity[1], material_owner_sha256=identity[0], entries=entries)
        with header_path.open('xb') as stream:
            pickle.dump(header, stream, protocol=5)
        result = dict(identity=identity, root_world_bytes=identity[1], material_owner_sha256=identity[0],
                      header_path=str(header_path), data_path=str(data_path),
                      header_sha256=_file_sha(header_path), data_sha256=_file_sha(data_path),
                      data_bytes=data_path.stat().st_size, entries=len(entries))
        pool._rinkspace_shared_material_bank = result
        report.update(status='ready', shared_entries=len(entries), data_bytes=result['data_bytes'],
                      header_bytes=header_path.stat().st_size, data_sha256=result['data_sha256'],
                      header_sha256=result['header_sha256'])
    except Exception as exc:
        report.update(status='original_material_fallback', error_type=type(exc).__name__, error=str(exc))
        for path in (data_path, header_path):
            if path.is_file():
                path.unlink()
    finally:
        # Drain speculative cache-fill jobs before any original fallback or
        # resource admission. Their errors never certify original source rows.
        if pending:
            wait(tuple(pending))
        pool._rinkspace_shared_material_bank_pending_jobs = 0
        pool._rinkspace_shared_material_bank_precompute_active = False
        pool._rinkspace_shared_material_bank_idle_receipt = dict(
            executor_identity=id(executor), pending_jobs=0, precompute_active=False,
            worker_jobs=report['worker_jobs'], status=report.get('status', 'interrupted'))
        report.update(seconds=time.perf_counter()-begin, pending_jobs=0, precompute_active=False)
        with (_REPORT_DIRECTORY / 'shared_material_preparation.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(report, ensure_ascii=False) + '\n')
    return result


def _ordered_groups_with_shared_material_bank(self, groups_and_cameras, world66, absolute, step, **kwargs):
    bank = None
    try:
        bank = _prepare_shared_material_bank(self, world66, float(step))
    except Exception as exc:
        if _REPORT_DIRECTORY is not None:
            with (_REPORT_DIRECTORY / 'shared_material_preparation.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(dict(status='original_material_fallback',
                    error_type=type(exc).__name__, error=str(exc))) + '\n')
    token = _BANK_SCOPE.set(bank)
    had_executor = '_executor' in vars(self)
    saved_executor = vars(self).get('_executor')
    original_executor = self._executor
    if bank is not None:
        def executor_with_bank():
            executor = original_executor()
            return None if executor is None else _BankExecutor(executor, bank)
        self._executor = executor_with_bank
    try:
        yield from _ORIGINAL_ORDERED_GROUPS(self, groups_and_cameras, world66, absolute, step, **kwargs)
    finally:
        _BANK_SCOPE.reset(token)
        if bank is not None:
            if had_executor:
                self._executor = saved_executor
            else:
                del self._executor


def _programme_material_scope(original):
    @functools.wraps(original)
    def evaluate(self, q, jacobian=True):
        # Original local()/world/_binding admission occurs before lazy root
        # lookup. Keep the current completion caller and every original body.
        scope = dict(job=None, geometry=self.geometry, root_world_bytes=None,
            root_world_factory=lambda: self.geometry_values(q), shared_bank=None,
            hits=0, original_value_calls=0, uncacheable_results=0, root_cache_resets=0)
        token = _SCOPE.set(scope)
        try:
            return original(self, q, jacobian)
        finally:
            _SCOPE.reset(token)
            self._rinkspace_last_material_value_scope = dict(
                jacobian=bool(jacobian), hits=scope['hits'], original_value_calls=scope['original_value_calls'],
                root_cache_resets=scope['root_cache_resets'],
                root_world_sha256=None if scope['root_world_bytes'] is None else hashlib.sha256(scope['root_world_bytes']).hexdigest())
            if _REPORT_DIRECTORY is not None:
                try:
                    with (_REPORT_DIRECTORY / 'parent_material_scope.jsonl').open('a', encoding='utf8') as stream:
                        stream.write(json.dumps(dict(block=getattr(self, 'name', None),
                            **self._rinkspace_last_material_value_scope)) + '\n')
                except OSError as error:
                    self._rinkspace_last_material_value_scope['diagnostic_write_error'] = str(error)
    return evaluate


def _pool_close_with_shared_material_cleanup(self):
    try:
        return _ORIGINAL_POOL_CLOSE(self)
    finally:
        if self.owns_executor and not getattr(self, '_rinkspace_active_geometry_group_iterators', 0):
            # Only fresh exclusively-created execution bank files owned by this
            # pool. The original snapshot, checkpoints and history stay intact.
            paths = getattr(self, '_rinkspace_owned_material_bank_paths', [])
            owned = {str(path) for path in paths}
            for key, bank in list(_MAPPED_BANKS.items()):
                if key[0] in owned:
                    _MAPPED_BANKS.pop(key)
                    bank['entries'].clear()
                    bank['mapping']._mmap.close()
            for path in paths:
                if path.is_file():
                    path.unlink()
            self._rinkspace_owned_material_bank_paths = []
            self._rinkspace_shared_material_bank = None


def _owned_value(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return _UNSUPPORTED
        owned = value.copy(order='K')
        writable = bool(value.flags.writeable)
        owned.flags.writeable = False
        return ('array', owned, writable)
    if type(value) in (tuple, list):
        entries = [_owned_value(part) for part in value]
        if any(entry is _UNSUPPORTED for entry in entries):
            return _UNSUPPORTED
        return ('tuple' if type(value) is tuple else 'list', entries)
    if value is None or isinstance(value, (np.generic, int, float, bool, str, bytes)):
        return ('scalar', value)
    return _UNSUPPORTED


def _copy_out(owned):
    if owned[0] == 'array':
        result = owned[1].copy(order='K')
        result.flags.writeable = owned[2]
        return result
    if owned[0] == 'tuple':
        return tuple(_copy_out(part) for part in owned[1])
    if owned[0] == 'list':
        return [_copy_out(part) for part in owned[1]]
    return owned[1]


def _value_bytes(owned):
    if owned[0] == 'array':
        return owned[1].nbytes
    if owned[0] in ('tuple', 'list'):
        return sum(_value_bytes(part) for part in owned[1])
    return 0


def _material_wrapper(original):
    def exact_material_value(self, binding, sample_world):
        scope = _SCOPE.get()
        if scope is None or scope['geometry'] is not self:
            return original(self, binding, sample_world)
        # This executes only after the original group and rebind admission.
        root = _root_bytes(scope)
        shared = _shared_material(scope, self, binding, sample_world)
        if shared is not _UNSUPPORTED:
            return shared
        cache = _VALUES.get(self)
        if cache is None or cache['root_world_bytes'] != root:
            cache = dict(root_world_bytes=root, entries={}, value_bytes=0)
            _VALUES[self] = cache
            scope['root_cache_resets'] += 1
        # Every physical coordinate remains in the sample-world key. Retaining
        # binding alongside its id also prevents identity reuse after deletion.
        key = (id(binding), np.asarray(sample_world, float).tobytes())
        previous = cache['entries'].get(key)
        if previous is not None and previous[0] is binding:
            scope['hits'] += 1
            return _copy_out(previous[1])
        value = original(self, binding, sample_world)
        owned = _owned_value(value)
        scope['original_value_calls'] += 1
        if owned is _UNSUPPORTED:
            scope['uncacheable_results'] += 1
            return value
        cache['entries'][key] = (binding, owned)
        cache['value_bytes'] += _value_bytes(owned)
        return _copy_out(owned)
    exact_material_value.__name__ = original.__name__
    return exact_material_value


def _evaluate_group_with_material_values(job, geometry, certificates, route):
    # Do not validate or inspect job fields before the original function: its
    # source-digest/shape/physical error priority remains authoritative.
    scope = dict(job=job, geometry=geometry, root_world_bytes=None,
                 shared_bank=job.get('_rinkspace_shared_material_bank', _BANK_SCOPE.get()),
                 hits=0, original_value_calls=0, uncacheable_results=0, root_cache_resets=0)
    token = _SCOPE.set(scope)
    try:
        result = _ORIGINAL_GROUP(job, geometry, certificates, route)
        cache = _VALUES.get(geometry)
        result['stats']['absolute_material_value_runtime'] = dict(
            hits=scope['hits'], original_value_calls=scope['original_value_calls'],
            shared_value_hits=scope.get('shared_hits', 0), shared_bank_error=scope.get('shared_bank_error'),
            uncacheable_results=scope['uncacheable_results'], root_cache_resets=scope['root_cache_resets'],
            current_root_cache_entries=0 if cache is None else len(cache['entries']),
            current_root_cached_value_array_bytes=0 if cache is None else cache['value_bytes'],
            original_validate_stencils_certificates_and_companion_rebind_retained=True)
        return result
    finally:
        _SCOPE.reset(token)


def _install_runtime():
    global _INSTALLED, _ORIGINAL_GROUP, _RECEIPT, _ORIGINAL_ORDERED_GROUPS, _ORIGINAL_POOL_CLOSE
    if _INSTALLED:
        return _RECEIPT
    from . import install
    from . import absolute_geometry_fd_pool_candidate as source
    install()
    from structured_programme_source_candidate.structured_programme_geometry import StructuredProgrammeGeometry
    from structured_programme_source_candidate.paired_programme_blocks import StructuredProgrammeBlock
    original = StructuredProgrammeGeometry._material_value
    group = source._evaluate_group_on_owner
    initializer = source.initialize_absolute_geometry_worker
    ordered = source.AbsoluteGeometryGroupPool.ordered_groups
    programme_evaluate = StructuredProgrammeBlock._evaluate
    if (original.__module__ != 'structured_programme_source_candidate.structured_programme_geometry'
            or original.__name__ != '_material_value'
            or group.__module__ != source.__name__ or group.__name__ != '_evaluate_group_on_owner'
            or initializer.__module__ != source.__name__ or initializer.__name__ != 'initialize_absolute_geometry_worker'):
        raise TypeError('Original source-owned material/group/initializer required')
    helper = Path(__file__).resolve()
    _RECEIPT = dict(schema='rinkspace.original-absolute-material-value-runtime.v1',
        helper_path=str(helper), helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        original_material_value_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
        original_group_evaluate_sha256=hashlib.sha256(inspect.getsource(group).encode()).hexdigest(),
        original_initializer_sha256=hashlib.sha256(inspect.getsource(initializer).encode()).hexdigest(),
        initializer=__name__ + '.initialize_absolute_material_value_worker',
        scope='Original programme _evaluate and _evaluate_group_on_owner; one complete root-world byte state per geometry',
        cache_key='Original binding object identity and every sample-world float64 byte',
        shared_value_key='Original qualified binding key and every effective sample-world float64 byte',
        shared_bank='One exact original stencil working set per root/step/private snapshot, prepared uniquely by existing FD pool',
        original_programme_evaluate_sha256=hashlib.sha256(inspect.getsource(programme_evaluate).encode()).hexdigest(),
        current_geometry_group_caller_left_installed=True,
        parent_nominal_material_values_reused_across_original_rebind_passes=True,
        original_per_row_inactive_certificates_and_companion_checks_not_shared_or_skipped=True,
        shared_precompute_active_attribute='_rinkspace_shared_material_bank_precompute_active',
        shared_precompute_pending_attribute='_rinkspace_shared_material_bank_pending_jobs',
        shared_precompute_idle_receipt_attribute='_rinkspace_shared_material_bank_idle_receipt',
        cache_outside_geometry_pickle=True, original_array_dtype_shape_value_and_writable_flag_retained=True,
        copy_out_prevents_alias_mutation=True, exceptions_not_cached=True,
        original_validation_FD_stencils_source_certificates_and_companion_rebind_retained=True,
        no_guessed_dependency_columns_or_custom_capacity=True,
        actual_complete_same_q_J_and_complete_cycle_AB_pending=True, accuracy_accepted=False)
    StructuredProgrammeGeometry._rinkspace_original_material_value = original
    StructuredProgrammeGeometry._material_value = _material_wrapper(original)
    _ORIGINAL_GROUP = group
    source._rinkspace_original_material_group_evaluate = group
    source._evaluate_group_on_owner = _evaluate_group_with_material_values
    source._rinkspace_original_material_worker_initializer = initializer
    source.initialize_absolute_geometry_worker = initialize_absolute_material_value_worker
    _ORIGINAL_ORDERED_GROUPS = ordered
    source.AbsoluteGeometryGroupPool._rinkspace_material_original_ordered_groups = ordered
    source.AbsoluteGeometryGroupPool.ordered_groups = _ordered_groups_with_shared_material_bank
    _ORIGINAL_POOL_CLOSE = source.AbsoluteGeometryGroupPool.close
    source.AbsoluteGeometryGroupPool.close = _pool_close_with_shared_material_cleanup
    StructuredProgrammeBlock._rinkspace_original_material_scoped_evaluate = programme_evaluate
    StructuredProgrammeBlock._evaluate = _programme_material_scope(programme_evaluate)
    _INSTALLED = True
    return _RECEIPT


def initialize_absolute_material_value_worker():
    """Picklable Windows spawn entry: original bootstrap before cache install."""
    from . import absolute_geometry_fd_pool_candidate as source
    original = getattr(source, '_rinkspace_original_material_worker_initializer', source.initialize_absolute_geometry_worker)
    original()
    from .role_record_index_runtime import install_role_record_index_runtime
    from .role_binding_value_reuse_runtime import install_role_binding_value_reuse_runtime
    install_role_record_index_runtime()
    install_role_binding_value_reuse_runtime()
    _install_runtime()


def install_absolute_material_value_runtime(*, report_directory=None):
    """Install in a new caller-owned run before its absolute-FD pool starts."""
    global _REPORT_DIRECTORY
    destination = None if report_directory is None else Path(report_directory).resolve()
    if _INSTALLED and _REPORT_DIRECTORY != destination:
        raise RuntimeError('Absolute material value runtime already installed for another report directory')
    receipt = _install_runtime()
    _REPORT_DIRECTORY = destination
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'absolute_material_value_runtime_receipt.json').write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return receipt
