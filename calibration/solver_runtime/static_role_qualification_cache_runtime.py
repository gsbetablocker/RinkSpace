"""Reuse only original source-only role qualification, never current geometry.

The outer original constructor authenticates and saves naturally on a miss.
Its recursive children remain original. A hit rehashes the complete captured
source closure before restoring only its static certificate fields. Current
camera binding, input compaction, planes, guards and derivatives remain live.
"""
from __future__ import annotations

from contextvars import ContextVar
import copy
import functools
import hashlib
import json
import marshal
from pathlib import Path
import pickle
import sys
import time

import numpy as np


SCHEMA = 'rinkspace.original-source-only-role-qualification-cache.v1'
_ACTIVE = ContextVar('rinkspace_original_role_qualification_dependencies', default=None)
_ORIGINAL_SHA = None
_RECEIPT = None
_CONTROLLER = None
_STATIC_FIELDS = frozenset(('packet_path', 'packet', 'records', 'indices', 'binding_id',
    'source_row_groups', 'source_stamps_by_block', 'source_uv', 'contract'))
_DYNAMIC_FIELDS = frozenset(('source_cameras', 'source_camera_signature', 'source_camera_signatures',
    '_current_binding_cache', '_facet_course_cache', '_source_camera_map_cache', 'exact_cache_statistics'))
_QUALIFICATION_FUNCTIONS = ('_low_records', '_validate_low_review_qualification',
    '_secondary_review_records', '_programme_review_records')
# Explicitly reviewed predecessor: it produced only this same static payload.
# The added exit rehash does not change qualification or certificate values.
_REVIEWED_PREDECESSOR_HELPER_SHA256 = 'e031fee4cfaeab76c5c0f03453ab233b9880a34a304272318ef2adb6c1af9b2f'


def _file_sha(path):
    # Always read bytes on a cache entry. Metadata is never cross-run authority.
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


def _literal(value):
    """Reject graph objects/closures: the payload is exclusively static data."""
    if type(value) in (str, bytes, int, float, bool, type(None)) or isinstance(value, Path):
        return
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError('Object-valued source certificate cannot be cached')
        return
    if isinstance(value, np.generic):
        if value.dtype.hasobject:
            raise TypeError('Object-valued source scalar cannot be cached')
        return
    if type(value) is dict:
        for key, item in value.items():
            _literal(key); _literal(item)
        return
    if type(value) in (tuple, list):
        for item in value:
            _literal(item)
        return
    raise TypeError('Nonliteral original certificate field: ' + type(value).__name__)


def _static_certificate(role):
    if set(role.__dict__) != _STATIC_FIELDS | _DYNAMIC_FIELDS:
        raise TypeError('Original source-only role constructor fields changed')
    for name in ('source_cameras', 'source_camera_signatures', '_source_camera_map_cache'):
        if role.__dict__[name] != {}:
            raise TypeError('Current cameras present at source-only qualification boundary')
    for name in ('source_camera_signature', '_current_binding_cache', '_facet_course_cache'):
        if role.__dict__[name] is not None:
            raise TypeError('Current geometry present at source-only qualification boundary')
    value = {key:role.__dict__[key] for key in _STATIC_FIELDS}
    _literal(value)
    return value


def _restore_certificate(role, value):
    if type(value) is not dict or set(value) != _STATIC_FIELDS:
        raise TypeError('Static original qualification fields differ')
    _literal(value)
    # Unpickling already creates independent arrays/dicts, and each hit does it
    # anew. No source row modified by compaction can leak into another instance.
    role.__dict__.update(value)
    role.source_cameras = {}
    role.source_camera_signature = None
    role.source_camera_signatures = {}
    role._current_binding_cache = None
    role._facet_course_cache = None
    role._source_camera_map_cache = {}
    role.exact_cache_statistics = dict(binding_hits=0, binding_misses=0,
        camera_hits=0, camera_misses=0, facet_group_hits=0, facet_group_misses=0,
        unchanged_facet_binding_hits=0)


def _capture_sha(path):
    digest = _ORIGINAL_SHA(path)
    active = _ACTIVE.get()
    if active is not None:
        key = str(Path(path).resolve())
        prior = active.get(key)
        if prior is not None and prior != digest:
            # The original auth scope will also reject the changed source; do
            # not let an inconsistent construction become a persistent cache.
            active['_inconsistent_source_bytes'] = True
        active[key] = digest
    return digest


def _register_existing_scope_exit_guard(dependencies):
    from .startup_source_auth_reuse_runtime import _ACTIVE as startup_scope
    scope = startup_scope.get()
    if scope is None or scope.closed:
        return False
    # Entry validation used independent actual byte SHA above. Preserve the
    # existing construction scope's original exit rehash as well: one call
    # registers each file and the second marks reuse. Its original producer
    # remains the exit authority, with no injected cached-digest callback.
    for path, digest in dependencies:
        if _ORIGINAL_SHA(path) != digest or _ORIGINAL_SHA(path) != digest:
            raise ValueError('Original startup source auth differs from current cache entry bytes')
    return True


def _registered_identity():
    from . import runtime
    root = Path(__file__).resolve().parent
    files = {}
    for name in ('source_manifest.json', 'event_factories_manifest.json', 'legacy_dependencies.json'):
        path = root/name
        manifest = json.loads(path.read_text(encoding='utf8'))
        files[str(path)] = _file_sha(path)
        rows = (list(manifest.get('modules', {}).values()) + list(manifest.get('native_sources', ()))
                + list(manifest.get('configured_owned_source_closure', ())))
        for row in rows:
            target = (root/row['path']).resolve()
            if str(target) in files:
                continue
            actual = _file_sha(target)
            if actual != row['sha256']:
                raise ValueError('Registered original qualification producer changed: '+str(target))
            files[str(target)] = actual
    runtime.verify_legacy_dependencies()
    return dict(registered_original_files=sorted(files.items()),
                python_cache_tag=sys.implementation.cache_tag, numpy_version=np.__version__)


class _QualificationCache:
    def __init__(self, directory, producer_identity, report_directory):
        self.directory = Path(directory).resolve()
        self.producer_identity = producer_identity
        self.report_directory = None if report_directory is None else Path(report_directory).resolve()
        self.events = []

    def _cache_entries(self, identity):
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf8')).hexdigest()
        yield self.directory/key, identity, None
        if identity['producer_identity'].get('qualification_cache_helper_SHA256') not in (
                None, _REVIEWED_PREDECESSOR_HELPER_SHA256):
            # Change this ONE reviewed identity field only. All actual
            # qualification function, registered/source/helper and runtime
            # identities must still match the prior receipt exactly.
            previous = copy.deepcopy(identity)
            previous['producer_identity']['qualification_cache_helper_SHA256'] = _REVIEWED_PREDECESSOR_HELPER_SHA256
            key = hashlib.sha256(json.dumps(previous, sort_keys=True).encode('utf8')).hexdigest()
            yield self.directory/key, previous, _REVIEWED_PREDECESSOR_HELPER_SHA256

    def _event(self, **value):
        value.update(schema=SCHEMA, wall_time_ns=time.time_ns(),
            current_q_camera_planes_or_compaction_results_reused=False)
        self.events.append(value)
        print(json.dumps(dict(stage='actual_static_role_qualification_cache', **value)), flush=True)
        if self.report_directory is not None:
            try:
                self.report_directory.mkdir(parents=True, exist_ok=True)
                with (self.report_directory/'static_role_qualification_cache_events.jsonl').open('a', encoding='utf8') as stream:
                    stream.write(json.dumps(value)+'\n')
            except OSError as error:
                # Optional reporting cannot reject an original passing source.
                value['report_write_error'] = type(error).__name__+': '+str(error)

    def invoke(self, original, role, packet_path):
        if _ACTIVE.get() is not None:
            return original(role, packet_path)
        begin = time.perf_counter()
        directory = None; identity = None; reason = 'cache_absent'
        try:
            path = Path(packet_path).resolve()
            identity = dict(schema=SCHEMA, packet_path=str(path), packet_sha256=_file_sha(path),
                            producer_identity=self.producer_identity)
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf8')).hexdigest()
            directory = self.directory/key
            for entry_directory, entry_identity, compatibility_sha in self._cache_entries(identity):
                receipt_path = entry_directory/'qualification_receipt.json'
                if not receipt_path.is_file():
                    continue
                receipt = json.loads(receipt_path.read_text(encoding='utf8'))
                if receipt['identity'] != json.loads(json.dumps(entry_identity)):
                    raise ValueError('Original qualification cache identity differs')
                dependencies = receipt['original_dependency_SHA256']
                if not dependencies or any(_file_sha(p) != digest for p, digest in dependencies):
                    raise ValueError('Original qualification source dependency bytes changed')
                payload = entry_directory/'static_certificate.pickle'
                if _file_sha(payload) != receipt['payload_sha256']:
                    raise ValueError('Static qualification cache payload bytes changed')
                with payload.open('rb') as stream:
                    value = pickle.load(stream)
                if str(value['packet_path']) != str(path) or value['binding_id'] != identity['packet_sha256']:
                    raise ValueError('Static qualification packet identity differs')
                exit_guard = _register_existing_scope_exit_guard(dependencies)
                _restore_certificate(role, value)
                self._event(action='authenticated_hit', packet_path=str(path),
                    elapsed_seconds=time.perf_counter()-begin, original_dependency_files=len(dependencies),
                    original_constructor_replayed=False, original_source_rows=len(role.records),
                    payload_bytes=receipt['payload_bytes'], original_startup_auth_exit_rehash_preserved=exit_guard,
                    compatible_predecessor_helper_sha256=compatibility_sha,
                    original_cache_receipt_path=str(receipt_path), original_cache_receipt_unchanged=True)
                return None
        except Exception as error:
            # Optional cache failure cannot change the original exception order.
            # A changed source runs the original constructor and its own gates.
            reason = type(error).__name__+': '+str(error)
        dependencies = {}
        token = _ACTIVE.set(dependencies)
        try:
            result = original(role, packet_path)
        finally:
            _ACTIVE.reset(token)
        saved = False; save_error = None
        try:
            if directory is None or identity is None:
                raise ValueError('No authenticated original packet cache key')
            if dependencies.pop('_inconsistent_source_bytes', False) or not dependencies:
                raise ValueError('Original constructor source bytes were not consistently authenticated')
            if dependencies.get(identity['packet_path']) != identity['packet_sha256']:
                raise ValueError('Original constructor packet SHA differs from entry packet bytes')
            value = _static_certificate(role)
            # Rehash before publishing, including reuse within the existing
            # process-local authentication scope. Stat-only hits never certify
            # a persistent qualification cache for a later process.
            ordered = sorted(dependencies.items())
            if any(_file_sha(p) != digest for p, digest in ordered):
                raise ValueError('Original qualification source changed before cache publication')
            raw = pickle.dumps(value, protocol=5)
            directory.mkdir(parents=True, exist_ok=True)
            payload = directory/'static_certificate.pickle'
            # A bad/changed cache remains historical; do not overwrite it.
            if not payload.exists():
                payload.write_bytes(raw)
                receipt = dict(identity=identity, original_dependency_SHA256=ordered,
                    payload_sha256=hashlib.sha256(raw).hexdigest(), payload_bytes=len(raw),
                    original_constructor_completed=True, original_source_rows=len(role.records),
                    current_q_camera_planes_or_compaction_results_cached=False)
                (directory/'qualification_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf8')
                saved = True
        except Exception as error:
            save_error = type(error).__name__+': '+str(error)
        self._event(action='original_constructor_miss', packet_path=str(Path(packet_path).resolve()),
            elapsed_seconds=time.perf_counter()-begin, miss_reason=reason, cache_saved=saved,
            cache_save_error=save_error, original_dependency_files=len(dependencies),
            original_constructor_replayed=True, original_source_rows=len(role.records))
        return result


def _install_on(module, *, cache_directory, report_directory, producer_identity):
    global _ORIGINAL_SHA
    original = module.OriginalMotionFarBoardRoleBinding.__init__
    controller = _QualificationCache(cache_directory, producer_identity, report_directory)
    _ORIGINAL_SHA = module._sha
    module._sha = _capture_sha

    @functools.wraps(original)
    def constructor(self, packet_path):
        return controller.invoke(original, self, packet_path)

    module.OriginalMotionFarBoardRoleBinding.__init__ = constructor
    return controller


def install_static_role_qualification_cache_runtime(*, cache_directory, report_directory=None):
    """Install once before original role construction; no prebuilt graph needed."""
    global _RECEIPT, _CONTROLLER
    if _RECEIPT is not None:
        if _RECEIPT['cache_directory'] != str(Path(cache_directory).resolve()):
            raise RuntimeError('Static qualification cache is already installed with another directory')
        return _RECEIPT
    from . import install
    install()
    from structured_programme_source_candidate import original_motion_far_board_roles as roles
    identity = _registered_identity()
    functions = [roles.OriginalMotionFarBoardRoleBinding.__init__, roles.OriginalMotionFarBoardRoleBinding._finish_certificate]
    functions += [getattr(roles, name) for name in _QUALIFICATION_FUNCTIONS]
    identity['actual_qualification_function_code_SHA256'] = [
        (f.__qualname__, hashlib.sha256(marshal.dumps(f.__code__)).hexdigest()) for f in functions]
    helper = Path(__file__).resolve()
    identity['qualification_cache_helper_SHA256'] = _file_sha(helper)
    # The optional exact indexing wrapper is part of the active producer too.
    for name in ('role_record_index_runtime.py', 'startup_source_auth_reuse_runtime.py'):
        identity[name] = _file_sha(helper.parent/name)
    _CONTROLLER = _install_on(roles, cache_directory=cache_directory,
        report_directory=report_directory, producer_identity=identity)
    _RECEIPT = dict(schema=SCHEMA, helper_path=str(helper), helper_sha256=_file_sha(helper),
        cache_directory=str(Path(cache_directory).resolve()), producer_identity=identity,
        transparent_outermost_constructor_only=True, nested_original_constructor_unchanged=True,
        complete_original_recursive_SHA_dependency_capture=True,
        each_cache_entry_actual_source_file_SHA_authentication=True,
        registered_producer_SHA_authentication_once_per_install=True,
        active_startup_auth_scope_hit_files_registered_for_original_exit_SHA=True,
        only_reviewed_predecessor_helper_sha256=_REVIEWED_PREDECESSOR_HELPER_SHA256,
        predecessor_qualification_registered_source_and_other_identity_exact_match_required=True,
        static_original_certificate_fields=sorted(_STATIC_FIELDS),
        current_camera_geometry_and_compaction_caches_never_saved=True,
        current_source_qualification_binding_guard_and_J_still_execute=True,
        same_seed_required=False, live_model_or_registered_source_files_modified=False,
        actual_speedup_pending=True)
    if report_directory is not None:
        destination = Path(report_directory).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        (destination/'static_role_qualification_cache_installation.json').write_text(json.dumps(_RECEIPT, indent=2)+'\n', encoding='utf8')
    return _RECEIPT


def static_role_qualification_cache_snapshot():
    return dict(installation=_RECEIPT,
                events=[] if _CONTROLLER is None else copy.deepcopy(_CONTROLLER.events))
