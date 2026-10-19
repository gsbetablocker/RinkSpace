"""Deduplicate original file authentication within one source-build scope.

Only original path-to-SHA functions are wrapped. All schema, source-row,
qualification, domain and current-q checks still execute. A cached SHA hit is
allowed only for the same file identity; every hit file is rehashed using its
original producer before leaving construction. No graph or JSON is cached.
"""
from __future__ import annotations

import ast
from contextvars import ContextVar
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import textwrap
import time
from types import FunctionType

_ACTIVE = ContextVar('rinkspace_original_startup_source_auth', default=None)
_RECEIPT = None
_REPORT_DIRECTORY = None


def _identity(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


class SourceAuthenticationScope:
    def __init__(self, output, report_directory=None):
        self.output = Path(output).resolve()
        self.report_directory = None if report_directory is None else Path(report_directory).resolve()
        self.files = {}
        self.calls = self.hits = self.requested_bytes = self.original_bytes = 0
        self.revalidated_files = self.revalidated_bytes = 0
        self.missing_native_png_authentications = []
        self.closed = False
        self.token = None

    def __enter__(self):
        if _ACTIVE.get() is not None:
            raise RuntimeError('An original startup source authentication scope is already active')
        self.token = _ACTIVE.set(self)
        return self

    def digest(self, value, original):
        path = Path(value).resolve()
        if path.is_relative_to(self.output):
            # Run outputs may legitimately be rewritten; never memoize them.
            return original(value)
        try:
            signature = _identity(path)
        except FileNotFoundError:
            # A deleted disposable PNG can still be authenticated byte for
            # byte against its immutable native cache manifest. Decode only
            # the requested native presentation, verify its original BGR and
            # encoded PNG hashes, and keep both buffers in memory. All caller
            # SHA comparisons remain unchanged and no source cache is made.
            from rinkspace import native_source_cache
            from rinkspace.storage import AssetStore
            storage = AssetStore(Path(__file__).resolve().parents[2])
            digest, receipt = native_source_cache.authenticate_missing_native_png(
                path, resolve_source=storage.source,
                authenticate_file=lambda dependency: self.digest(dependency, native_source_cache._sha))
            self.missing_native_png_authentications.append(receipt)
            return digest
        self.calls += 1
        self.requested_bytes += signature[2]
        entry = self.files.get(path)
        if entry is not None and signature == entry['identity']:
            entry['hits'] += 1
            self.hits += 1
            return entry['sha256']
        digest = original(value)
        if _identity(path) != signature:
            raise ValueError('Original startup source changed during file authentication: ' + str(path))
        if not isinstance(digest, str) or len(digest) != 64:
            raise TypeError('Original path authentication must return its complete SHA256 string')
        self.original_bytes += signature[2]
        self.files[path] = dict(identity=signature, sha256=digest, original=original,
                               original_argument=value, hits=0)
        return digest

    def finish(self, verify=True):
        if self.closed:
            return
        succeeded = False
        try:
            if verify:
                for path, entry in self.files.items():
                    if not entry['hits']:
                        continue
                    # Metadata alone is not the evidence. Rehash every reused
                    # input even if a writer preserved size and mtime.
                    before = _identity(path)
                    actual = entry['original'](entry['original_argument'])
                    self.revalidated_files += 1
                    self.revalidated_bytes += before[2]
                    if before != _identity(path) or actual != entry['sha256']:
                        raise ValueError('Original reused startup source changed before construction completed: ' + str(path))
            succeeded = verify
        finally:
            self.closed = True
            try:
                if self.token is not None:
                    _ACTIVE.reset(self.token)
                    self.token = None
            finally:
                report = dict(schema='rinkspace.original-startup-source-file-auth-reuse.v1',
                    succeeded=succeeded, original_authentication_calls=self.calls,
                    distinct_source_files=len(self.files), original_SHA_reuse_hits=self.hits,
                    requested_file_bytes=self.requested_bytes, original_first_or_changed_file_bytes=self.original_bytes,
                    exit_revalidated_hit_files=self.revalidated_files, exit_revalidated_file_bytes=self.revalidated_bytes,
                    run_output_files_excluded=True, cached_files_after_scope=0,
                    original_schema_qualification_source_rows_sigma_and_current_q_guards_execute=True,
                    file_content_changes_rejected_even_with_preserved_size_and_mtime=True,
                    missing_native_png_authentications=self.missing_native_png_authentications,
                    missing_native_png_buffers_written=0,
                    source_graph_and_JSON_objects_cached=False, wall_time_ns=time.time_ns())
                self.files.clear()
                print(json.dumps(dict(stage='startup_source_authentication_reuse', **report)), flush=True)
                if self.report_directory is not None:
                    self.report_directory.mkdir(parents=True, exist_ok=True)
                    (self.report_directory / 'actual_startup_source_auth_reuse.json').write_text(
                        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.finish()
        else:
            # Retain the original failure/control exception; cache cleanup
            # cannot introduce a second file error that masks it.
            try:
                self.finish(verify=False)
            except Exception:
                pass
        return False


def _source_sha(value, original):
    scope = _ACTIVE.get()
    if scope is None or scope.closed:
        return original(value)
    return scope.digest(value, original)


def _finish_active_scope():
    scope = _ACTIVE.get()
    if scope is not None:
        scope.finish()


def install_startup_source_auth_reuse_runtime(*, report_directory=None):
    global _RECEIPT, _REPORT_DIRECTORY
    destination = None if report_directory is None else Path(report_directory).resolve()
    if _RECEIPT is not None:
        if _RECEIPT['report_directory'] != (None if destination is None else str(destination)):
            raise RuntimeError('Startup source authentication reuse belongs to another run')
        return _RECEIPT
    specifications = (
        ('structured_programme_source_candidate.native_ice_curve_source', '_sha'),
        ('structured_programme_source_candidate.human_stroke_truth', '_sha'),
        ('structured_programme_source_candidate.static_motion_source_role_retirement', '_sha'),
        ('structured_programme_source_candidate.original_motion_input_compaction', '_sha'),
        ('structured_programme_source_candidate.original_motion_far_board_roles', '_sha'),
        ('run_shared_print_pairs_and_mount_full_source_candidate', 'sha'),
    )
    seen, bindings = set(), []
    for module_name, name in specifications:
        module = importlib.import_module(module_name)
        function = getattr(module, name)
        if id(function) in seen:
            continue
        seen.add(id(function))
        if function.__code__.co_freevars or len(inspect.signature(function).parameters) != 1:
            raise TypeError('Original closure-free one-path source SHA producer required')
        source = textwrap.dedent(inspect.getsource(function))
        original = FunctionType(function.__code__, function.__globals__, function.__name__,
                                function.__defaults__, function.__closure__)
        key = '_rinkspace_original_startup_SHA_' + module_name.replace('.', '_') + '_' + name
        function.__globals__[key] = original
        function.__globals__['_rinkspace_startup_source_SHA'] = _source_sha
        tree = ast.parse(source)
        target = tree.body[0]
        argument = target.args.args[0].arg
        target.body = ast.parse('return _rinkspace_startup_source_SHA(' + argument + ', ' + key + ')').body
        ast.fix_missing_locations(tree)
        ast.increment_lineno(tree, function.__code__.co_firstlineno - 1)
        namespace = {}
        exec(compile(tree, inspect.getsourcefile(function), 'exec'), function.__globals__, namespace)
        function.__code__ = namespace[name].__code__
        bindings.append(dict(function=module_name + '.' + name,
                             original_source_sha256=hashlib.sha256(source.encode('utf8')).hexdigest()))
    _REPORT_DIRECTORY = destination
    helper = Path(__file__).resolve()
    _RECEIPT = dict(schema='rinkspace.original-startup-source-file-auth-reuse.v1',
        helper_path=str(helper), helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        report_directory=None if destination is None else str(destination), original_SHA_functions=bindings,
        scope='original source construction only; closes before independent startup derivatives',
        hit_files_revalidated_with_original_SHA_before_scope_exit=True,
        missing_complete_native_PNGs_authenticated_by_original_source_native_PTS_BGR_and_PNG_hashes=True,
        missing_native_PNG_recovery_writes_no_images=True,
        single_read_files_not_read_again_at_exit=True, no_static_graph_or_mutable_JSON_cache=True,
        original_q_dependent_checks_and_original_errors_retained=True, actual_speedup_pending=True)
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'startup_source_auth_reuse_runtime_receipt.json').write_text(
            json.dumps(_RECEIPT, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return _RECEIPT


def attach_startup_source_auth_reuse(tree, namespace, destination):
    """Use the original build marker and diagnostic boundary; no source edit."""
    sites = []
    expected = ast.dump(ast.parse('opt.reuse_complete_linearization is None', mode='eval').body)
    class Close(ast.NodeTransformer):
        def visit_If(self, node):
            self.generic_visit(node)
            if (any(isinstance(part, ast.Compare) and ast.dump(part) == expected for part in ast.walk(node.test))
                    and any(isinstance(part, ast.Constant) and part.value == 'actual_all_small_source_values_and_jacobians'
                            for statement in node.body for part in ast.walk(statement))):
                sites.append('original_independent_diagnostic_boundary')
                return [ast.parse('_rinkspace_finish_startup_source_auth()').body[0], node]
            return node
    tree = Close().visit(tree)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    markers = [index for index, node in enumerate(main.body) if isinstance(node, ast.Expr)
        and any(isinstance(part, ast.Constant) and part.value == 'build_preserved_complete_original_programme_source_graph'
                for part in ast.walk(node))]
    if len(markers) != 1 or sites != ['original_independent_diagnostic_boundary']:
        raise ValueError('Original source-build authentication scope boundaries changed')
    boundary = markers[0]
    body = main.body[boundary:]
    scope = ast.parse('with _rinkspace_startup_source_auth_scope(opt.output):\n    pass').body[0]
    scope.body = body
    main.body[boundary:] = [scope]
    destination = Path(destination).resolve()
    namespace['_rinkspace_startup_source_auth_scope'] = lambda output: SourceAuthenticationScope(output, destination)
    namespace['_rinkspace_finish_startup_source_auth'] = _finish_active_scope
    ast.fix_missing_locations(tree)
    return tree
