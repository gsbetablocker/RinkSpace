"""Protocol5 COW transport for the original prepared source pickle only.

The first necessary original load naturally emits metadata and packed buffers.
Subsequent loads authenticate the original pickle and every derived artifact,
then use the same original reducers with privately writable mapped buffers.
This hook precedes restore_local_priors and every current checkpoint restore.
It does not capture a driver closure, current residual, Jacobian or normal.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import io
import json
import mmap
from pathlib import Path
import pickle
import sys
import textwrap
import time
import uuid

import numpy as np


SCHEMA = 'rinkspace.original-prepared-pickle-protocol5-COW.v1'
_RECEIPT = None
_CONTROLLER = None
_ALIGNMENT = 64


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


class PreparedOutOfBandCache:
    def __init__(self, cache_directory, producer_identity, report_directory=None):
        self.directory = Path(cache_directory).resolve()
        self.producer_identity = producer_identity
        self.report_directory = None if report_directory is None else Path(report_directory).resolve()
        self.events = []

    def _event(self, **value):
        value.update(schema=SCHEMA, wall_time_ns=time.time_ns(),
            current_checkpoint_or_driver_closure_cached=False,
            original_source_pickle_and_downstream_restore_authority_retained=True)
        self.events.append(value)
        print(json.dumps(dict(stage='actual_prepared_source_OOB_cache', **value)), flush=True)
        if self.report_directory is not None:
            try:
                self.report_directory.mkdir(parents=True, exist_ok=True)
                with (self.report_directory/'prepared_oob_cache_events.jsonl').open('a', encoding='utf8') as stream:
                    stream.write(json.dumps(value)+'\n')
            except OSError as error:
                value['report_write_error'] = type(error).__name__+': '+str(error)

    def _identity(self, source):
        return dict(schema=SCHEMA, original_prepared_pickle_path=str(source),
                    original_prepared_pickle_sha256=_sha(source), producer_identity=self.producer_identity)

    def _read(self, directory, identity):
        receipt = json.loads((directory/'prepared_oob_receipt.json').read_text(encoding='utf8'))
        if receipt['identity'] != json.loads(json.dumps(identity)):
            raise ValueError('Original prepared OOB identity differs')
        metadata_path, buffers_path = directory/'metadata.pickle', directory/'packed_buffers.bin'
        metadata = metadata_path.read_bytes()
        if hashlib.sha256(metadata).hexdigest() != receipt['metadata_sha256']:
            raise ValueError('Original prepared metadata bytes changed')
        if _sha(buffers_path) != receipt['packed_buffers_sha256']:
            raise ValueError('Original prepared packed buffer bytes changed')
        packed_bytes = buffers_path.stat().st_size
        if len(metadata) != receipt['metadata_bytes'] or packed_bytes != receipt['packed_file_bytes']:
            raise ValueError('Original prepared metadata/packed file sizes differ')
        spans = receipt['buffers']
        if len(spans) != receipt['buffer_count']:
            raise ValueError('Original prepared buffer count differs')
        previous_stop = 0
        for span in spans:
            offset, size = span['offset'], span['nbytes']
            if (type(offset) is not int or type(size) is not int or offset < previous_stop
                    or size < 0 or offset % _ALIGNMENT or offset+size > packed_bytes):
                raise ValueError('Original prepared ordered aligned buffer bounds differ')
            previous_stop = offset+size
        mapping = None
        views = []
        try:
            if spans:
                with buffers_path.open('rb') as stream:
                    mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY)
                views = [memoryview(mapping)[row['offset']:row['offset']+row['nbytes']] for row in spans]
            result = pickle.loads(metadata, buffers=views)
            # ndarray._frombuffer keeps the mapping alive through its buffer
            # base. Do not attach an mmap/handle to the source state: an actor
            # pickle must still contain only its original model fields.
            return result, receipt
        except BaseException:
            for view in views:
                view.release()
            if mapping is not None:
                try:
                    mapping.close()
                except BufferError:
                    # Any partially reconstructed ndarray owns its view until
                    # exception cleanup/GC. Never invalidate exported memory.
                    pass
            raise

    def _save(self, directory, identity, state):
        if getattr(state, 'source_actors', None) is not None:
            raise ValueError('Only the original pre-actor prepared state can be saved')
        if directory.exists():
            raise ValueError('Existing original prepared OOB artifact is immutable')
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.directory/('.building_'+uuid.uuid4().hex)
        temporary.mkdir()
        spans = []
        metadata_stream = io.BytesIO()
        try:
            with (temporary/'packed_buffers.bin').open('xb') as packed:
                def store_buffer(value):
                    view = value.raw()
                    try:
                        padding = (-packed.tell()) % _ALIGNMENT
                        if padding:
                            packed.write(b'\0'*padding)
                        offset = packed.tell()
                        packed.write(view)
                        spans.append(dict(offset=offset, nbytes=view.nbytes))
                    finally:
                        view.release()
                    return None
                pickle.Pickler(metadata_stream, protocol=5, buffer_callback=store_buffer).dump(state)
                if packed.tell() == 0:
                    # Windows cannot map an empty file, including empty arrays.
                    packed.write(b'\0')
            metadata = metadata_stream.getvalue()
            (temporary/'metadata.pickle').write_bytes(metadata)
            buffers_path = temporary/'packed_buffers.bin'
            receipt = dict(identity=identity, protocol=5, alignment_bytes=_ALIGNMENT,
                metadata_bytes=len(metadata), metadata_sha256=hashlib.sha256(metadata).hexdigest(),
                packed_file_bytes=buffers_path.stat().st_size, packed_buffers_sha256=_sha(buffers_path),
                buffer_count=len(spans), buffer_payload_bytes=sum(row['nbytes'] for row in spans),
                buffers=spans, original_prepared_state_before_current_checkpoint_restore=True,
                current_source_r_J_H_or_driver_closure_added=False, mmap_access='ACCESS_COPY',
                original_pickle_reducers_order_memo_and_READONLY_BUFFER_retained=True)
            (temporary/'prepared_oob_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf8')
            temporary.rename(directory)
            return receipt
        except BaseException:
            # Clean only this fresh, explicitly owned unpublished directory.
            for name in ('metadata.pickle', 'packed_buffers.bin', 'prepared_oob_receipt.json'):
                (temporary/name).unlink(missing_ok=True)
            temporary.rmdir()
            raise

    def load(self, original_prepared_pickle):
        source = Path(original_prepared_pickle).resolve()
        begin = time.perf_counter()
        identity = None; directory = None; reason = 'cache_absent'
        try:
            identity = self._identity(source)
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf8')).hexdigest()
            directory = self.directory/key
            if (directory/'prepared_oob_receipt.json').is_file():
                result, receipt = self._read(directory, identity)
                self._event(action='authenticated_OOB_hit', elapsed_seconds=time.perf_counter()-begin,
                    original_prepared_pickle_sha256=identity['original_prepared_pickle_sha256'],
                    metadata_bytes=receipt['metadata_bytes'], packed_file_bytes=receipt['packed_file_bytes'],
                    buffer_payload_bytes=receipt['buffer_payload_bytes'], buffer_count=receipt['buffer_count'],
                    mmap_access='ACCESS_COPY', immutable_cache_files_written=False)
                return result
        except Exception as error:
            reason = type(error).__name__+': '+str(error)
        # Exactly the original load path; failures here propagate unchanged.
        raw = source.read_bytes()
        result = pickle.loads(raw)
        actual_original_sha = hashlib.sha256(raw).hexdigest()
        del raw
        saved = False; save_error = None; receipt = None
        try:
            if identity is None or directory is None:
                raise ValueError('Original prepared pickle identity was unavailable')
            if actual_original_sha != identity['original_prepared_pickle_sha256']:
                raise ValueError('Original prepared pickle changed during loading')
            receipt = self._save(directory, identity, result)
            saved = True
        except Exception as error:
            save_error = type(error).__name__+': '+str(error)
        self._event(action='original_prepared_load_miss', elapsed_seconds=time.perf_counter()-begin,
            original_prepared_pickle_sha256=actual_original_sha, miss_reason=reason,
            natural_cache_saved=saved, cache_save_error=save_error,
            metadata_bytes=None if receipt is None else receipt['metadata_bytes'],
            buffer_payload_bytes=None if receipt is None else receipt['buffer_payload_bytes'],
            buffer_count=None if receipt is None else receipt['buffer_count'],
            first_natural_cache_export_included_in_actual_cold_elapsed=True)
        return result


def _registered_identity():
    from . import runtime
    status = runtime.status(verify_sources=True)
    runtime.verify_legacy_dependencies()
    root = Path(__file__).resolve().parent
    return dict(registered_source_manifest_sha256=status['source_manifest_sha256'],
        event_factories_manifest_sha256=_sha(root/'event_factories_manifest.json'),
        legacy_dependencies_manifest_sha256=_sha(root/'legacy_dependencies.json'),
        registered_original_source_actual_SHA_verified=True,
        python_cache_tag=sys.implementation.cache_tag, numpy_version=np.__version__,
        prepared_oob_cache_helper_sha256=_sha(__file__))


def install_prepared_oob_cache_runtime(*, cache_directory, report_directory=None):
    """Hook the one original prepared load; install before the next graph load."""
    global _RECEIPT, _CONTROLLER
    if _RECEIPT is not None:
        if _RECEIPT['cache_directory'] != str(Path(cache_directory).resolve()):
            raise RuntimeError('Prepared OOB cache already belongs to another directory')
        return _RECEIPT
    from . import install
    install()
    import benchmark_harness as harness
    original = harness.capture
    if original.__code__.co_freevars:
        raise ValueError('Original prepared capture must remain closure-free')
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    matches = []
    class Loader(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if ast.unparse(node) == 'pickle.loads(cache.read_bytes())':
                matches.append('original_prepared_pickle_load')
                return ast.copy_location(ast.Call(func=ast.Name(id='_rinkspace_prepared_oob_load', ctx=ast.Load()),
                                                 args=[ast.Name(id='cache', ctx=ast.Load())], keywords=[]), node)
            return node
    tree = Loader().visit(tree)
    if matches != ['original_prepared_pickle_load']:
        raise ValueError('Original prepared pickle load boundary changed: '+str(matches))
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, original.__code__.co_firstlineno-1)
    identity = _registered_identity()
    _CONTROLLER = PreparedOutOfBandCache(cache_directory, identity, report_directory)
    original.__globals__['_rinkspace_prepared_oob_load'] = _CONTROLLER.load
    namespace = {}
    exec(compile(tree, inspect.getsourcefile(original), 'exec'), original.__globals__, namespace)
    original.__code__ = namespace[original.__name__].__code__
    helper = Path(__file__).resolve()
    _RECEIPT = dict(schema=SCHEMA, helper_path=str(helper), helper_sha256=_sha(helper),
        cache_directory=str(Path(cache_directory).resolve()), producer_identity=identity,
        original_capture_source_sha256=hashlib.sha256(source.encode('utf8')).hexdigest(),
        AST_change='Only original prepared pickle.loads(cache.read_bytes()) call',
        original_pickle_metadata_and_packed_bytes_actual_SHA_each_entry=True,
        original_readonly_array_opcode_alias_memo_dtype_shape_and_order_retained=True,
        mmap_access='ACCESS_COPY', packed_buffer_alignment_bytes=_ALIGNMENT,
        original_restore_local_priors_and_all_current_checkpoint_restore_unchanged=True,
        first_natural_original_load_and_export_counted_in_cold_elapsed=True,
        original_load_failure_not_swallowed=True, actual_speedup_pending=True)
    if report_directory is not None:
        destination = Path(report_directory).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        (destination/'prepared_oob_cache_installation.json').write_text(json.dumps(_RECEIPT, indent=2)+'\n', encoding='utf8')
    return _RECEIPT


def prepared_oob_cache_snapshot():
    return dict(installation=_RECEIPT, events=[] if _CONTROLLER is None else list(_CONTROLLER.events))
