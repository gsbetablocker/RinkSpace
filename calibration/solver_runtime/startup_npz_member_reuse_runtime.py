"""Read each original source archive member once in its own load context.

The original secondary role certificates compare thousands of scalar rows
against four NPZ members. NpzFile normally inflates those complete members on
every subscript. All original file hashes, row tests and errors remain active;
only repeated, read-only access within that one open archive is shared.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap


_RECEIPT = None


class _MemberOnceArchive:
    def __init__(self, archive):
        self.archive = archive
        self.members = {}

    def __getitem__(self, key):
        if key not in self.members:
            self.members[key] = self.archive[key]
        return self.members[key]

    def __getattr__(self, name):
        return getattr(self.archive, name)


class _MemberOnceContext:
    def __init__(self, archive):
        self.archive = archive
        self.proxy = None

    def __enter__(self):
        entered = self.archive.__enter__()
        self.proxy = _MemberOnceArchive(entered)
        return self.proxy

    def __exit__(self, *exc):
        try:
            return self.archive.__exit__(*exc)
        finally:
            if self.proxy is not None:
                self.proxy.members.clear()
                self.proxy = None


def _wrap_original_archive_context(function):
    if function.__code__.co_freevars:
        raise TypeError('Original closure-free source certificate required')
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    sites = []

    class Reuse(ast.NodeTransformer):
        def visit_With(self, node):
            self.generic_visit(node)
            for item in node.items:
                if (isinstance(item.optional_vars, ast.Name)
                        and item.optional_vars.id == 'archive'
                        and isinstance(item.context_expr, ast.Call)
                        and ast.unparse(item.context_expr.func) == 'np.load'):
                    # These two exact original bodies only read archive[key]
                    # before scalar/array equality checks. Fail if that source
                    # access changes into an assignment or an unknown use.
                    body = ast.Module(body=node.body, type_ignores=[])
                    parents = {id(child): parent for parent in ast.walk(body) for child in ast.iter_child_nodes(parent)}
                    for access in ast.walk(body):
                        if isinstance(access, ast.Name) and access.id == 'archive':
                            parent = parents.get(id(access))
                            if (not isinstance(access.ctx, ast.Load)
                                    or not isinstance(parent, ast.Subscript)
                                    or parent.value is not access):
                                raise TypeError('Original archive must remain read-only')
                            while isinstance(parent, ast.Subscript):
                                if not isinstance(parent.ctx, ast.Load):
                                    raise TypeError('Original archive member write cannot be reused')
                                parent = parents.get(id(parent))
                        if (isinstance(access, ast.Subscript)
                                and isinstance(access.value, ast.Name)
                                and access.value.id == 'archive'
                                and not isinstance(access.ctx, ast.Load)):
                            raise TypeError('Original archive member write cannot be reused')
                    sites.append(node.lineno)
                    item.context_expr = ast.Call(
                        func=ast.Name(id='_rinkspace_member_once_archive', ctx=ast.Load()),
                        args=[item.context_expr], keywords=[])
            return node

    tree = Reuse().visit(tree)
    if len(sites) != 1:
        raise TypeError('Exactly one original NPZ archive context required: ' + str(sites))
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, function.__code__.co_firstlineno - 1)
    function.__globals__['_rinkspace_member_once_archive'] = _MemberOnceContext
    namespace = {}
    exec(compile(tree, inspect.getsourcefile(function), 'exec'), function.__globals__, namespace)
    function.__code__ = namespace[function.__name__].__code__
    return dict(function=function.__qualname__, original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                original_archive_context_count=len(sites), original_np_load_call_unchanged=True)


def install_startup_npz_member_reuse_runtime(*, report_directory=None):
    """Install before the existing source-role certificate constructors run."""
    global _RECEIPT
    if _RECEIPT is not None:
        return _RECEIPT
    from . import install
    install()
    from structured_programme_source_candidate import original_motion_far_board_roles as original
    functions = [
        _wrap_original_archive_context(original._secondary_review_records),
        _wrap_original_archive_context(original.OriginalMotionFarBoardRoleBinding.__init__),
    ]
    path = Path(__file__).resolve()
    _RECEIPT = dict(schema='rinkspace.original-source-NPZ-member-once-context.v1',
        helper_path=str(path), helper_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        original_functions=functions, cache_scope='One original np.load archive context only',
        lazy_first_member_read_and_original_member_dtype_shape_bytes_retained=True,
        original_allow_pickle_file_hash_clock_UV_sigma_and_all_row_checks_retained=True,
        original_context_close_and_exception_propagation_retained=True,
        no_cross_file_or_cross_run_member_reuse=True, no_source_graph_or_workers_created=True,
        registered_source_files_modified=False, accuracy_accepted=False)
    if report_directory is not None:
        destination = Path(report_directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'startup_npz_member_reuse_receipt.json').write_text(
            json.dumps(_RECEIPT, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return _RECEIPT
