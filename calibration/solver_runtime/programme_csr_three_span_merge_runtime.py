"""Stage the original programme's three CSR insertions into one native copy.

The registered _evaluate body and all mathematical producers stay intact.
Its two existing insert_mount_csr global calls run their original admission
prefixes, then retain private numeric spans. Its final existing selected-motion
insertion performs one row-owned merge. No operator is replaced by a formula,
and unsupported callers/storage/backend use the original three insertions.
"""
from __future__ import annotations

import ast
from contextvars import ContextVar
import ctypes
import dis
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import textwrap
import time

import numpy as np
from scipy.sparse import csr_matrix

from .config import DEFAULT_NATIVE_CACHE_ROOT

SOURCE = Path(__file__).with_name('native')/'programme_csr_three_span_merge.c'
_SCOPE = ContextVar('rinkspace_original_programme_three_CSR_spans', default=None)
_RECEIPT = None
_REPORT = None
_ORIGINAL_MOUNT = _ORIGINAL_SELECTED = None
_MOUNT_ADMISSION = _SELECTED_ADMISSION = None
_KERNELS = None
_LIBRARY = None
_LOAD_ERROR = None


def native_build_directory():
    root = Path(os.environ.get('RINKSPACE_SOLVER_NATIVE_CACHE_ROOT', str(DEFAULT_NATIVE_CACHE_ROOT)))
    return root/'programme_csr_three_span_merge'/hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def _load_native():
    global _KERNELS, _LIBRARY, _LOAD_ERROR
    if _KERNELS is not None:
        return _KERNELS
    try:
        directory = native_build_directory()
        manifest = json.loads((directory/'build.json').read_text(encoding='utf8'))
        binary = directory/manifest['binary']
        if (manifest['source_sha256'] != hashlib.sha256(SOURCE.read_bytes()).hexdigest()
                or manifest['dll_sha256'] != hashlib.sha256(binary.read_bytes()).hexdigest()):
            raise ValueError('Original CSR merge source/binary identity changed')
        library = ctypes.CDLL(str(binary))
        kernels = {}
        for suffix, dtype in (('i32', np.dtype(np.int32)), ('i64', np.dtype(np.int64))):
            prepare = getattr(library, 'merge_prepare_'+suffix)
            prepare.argtypes = ([ctypes.c_int64]*3 + [ctypes.c_void_p]*2
                + [ctypes.c_int64] + [ctypes.c_void_p]*2
                + [ctypes.c_int64] + [ctypes.c_void_p]*3
                + [ctypes.c_int64]*2 + [ctypes.c_void_p]*2
                + [ctypes.c_int] + [ctypes.c_void_p]*2)
            prepare.restype = ctypes.c_int
            fill = getattr(library, 'merge_fill_'+suffix)
            fill.argtypes = ([ctypes.c_int64] + [ctypes.c_void_p]*3
                + [ctypes.c_int64] + [ctypes.c_void_p]*2
                + [ctypes.c_int64] + [ctypes.c_void_p]*3
                + [ctypes.c_int64] + [ctypes.c_void_p]*2
                + [ctypes.c_int] + [ctypes.c_void_p]*3)
            fill.restype = ctypes.c_int
            kernels[dtype] = prepare, fill
        _LIBRARY, _KERNELS, _LOAD_ERROR = library, kernels, None
    except (OSError, ValueError, KeyError, AttributeError) as error:
        _KERNELS, _LOAD_ERROR = {}, type(error).__name__+': '+str(error)
    return _KERNELS


def metadata():
    kernels = _load_native()
    return dict(available=bool(kernels), load_error=_LOAD_ERROR,
        source_path=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        build_directory=str(native_build_directory()),
        implementation='original_three_spans_row_owned_stable_merge_without_float_arithmetic')


def _admission_prefix(original, boundary_name, return_expression):
    """Compile the actual original checks; no source-string replacement."""
    if original.__code__.co_freevars:
        raise TypeError('Original closure-free CSR insertion checks required')
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    function = tree.body[0]
    sites = [i for i, node in enumerate(function.body)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id == boundary_name for target in node.targets)]
    if len(sites) != 1:
        raise TypeError('Original CSR admission/count boundary changed')
    if boundary_name == 'active':
        # The mount helper has no mathematical producer between admission
        # and its first nonzero-count assignment.
        body = function.body[:sites[0]]
    else:
        # Course admission includes the exact original stable column sort;
        # its following overlap loop is performed against virtual spans in C.
        loops = [i for i, node in enumerate(function.body[:sites[0]])
            if isinstance(node, ast.For)]
        if (len(loops) != 1 or loops[0] != sites[0]-2
                or not isinstance(function.body[loops[0]+1], ast.Assign)
                or not any(isinstance(target, ast.Name) and target.id == 'active'
                    for target in function.body[loops[0]+1].targets)):
            raise TypeError('Original selected-row overlap check boundary changed')
        body = function.body[:loops[0]]
    function.name = '_original_admission'
    function.body = body+[ast.Return(ast.parse(return_expression, mode='eval').body)]
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, inspect.getsourcefile(original), 'exec'), original.__globals__, namespace)
    return namespace[function.name], hashlib.sha256(source.encode()).hexdigest()


def _authenticate_registered_tail(original):
    """Authenticate source AST and the actually installed bytecode suffix."""
    if original.__code__.co_freevars:
        raise TypeError('Original closure-free programme insertion caller required')
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    function = tree.body[0]
    expected = ast.parse(
        'matrix=insert_mount_csr(matrix,lens,self.optics_chart.indices,34204)\n'
        'matrix=insert_mount_csr(matrix,mount,self.optics_chart.mount_indices,34195)\n'
        'rows,values=self.course_motion_columns(local,cal)\n'
        "columns=getattr(self,'course_current_motion_geometry_parameter_indices',COURSE_GLOBAL)\n"
        'return residual,insert_selected_motion_geometry_csr(matrix,rows,values,'
        'motion_rows=self.bundle.motion_rows,geometry_columns=columns)\n').body
    if (len(function.body) < 5 or
            [ast.dump(node, include_attributes=False) for node in function.body[-5:]]
            != [ast.dump(node, include_attributes=False) for node in expected]):
        raise TypeError('Original programme three-span insertion suffix changed')
    namespace = {}
    exec(compile(tree, inspect.getsourcefile(original), 'exec'), original.__globals__, namespace)
    declared = namespace[function.name]
    def suffix(function):
        instructions = list(dis.get_instructions(function))
        calls = [i for i, instruction in enumerate(instructions)
            if instruction.opname == 'LOAD_GLOBAL' and instruction.argval == 'insert_mount_csr']
        selected = [i for i, instruction in enumerate(instructions)
            if instruction.opname == 'LOAD_GLOBAL' and instruction.argval == 'insert_selected_motion_geometry_csr']
        if len(calls) != 2 or len(selected) != 1 or selected[0] <= calls[1]:
            raise TypeError('Actual registered original insertion calls changed')
        return [(instruction.opname, instruction.argval,
                 instruction.arg & 1 if instruction.opname == 'LOAD_GLOBAL' else None)
                for instruction in instructions[calls[0]:]]
    if suffix(original) != suffix(declared):
        raise TypeError('Actual registered programme insertion bytecode changed')
    return hashlib.sha256(source.encode()).hexdigest()


def _compatible(matrix):
    return (type(matrix) is csr_matrix and matrix.data.dtype == np.dtype(np.float64)
        and matrix.indices.dtype in (np.dtype(np.int32), np.dtype(np.int64))
        and matrix.indptr.dtype == matrix.indices.dtype
        and all(value.flags.c_contiguous for value in (matrix.data, matrix.indices, matrix.indptr))
        and len(matrix.indptr) == matrix.shape[0]+1 and len(matrix.data) == len(matrix.indices)
        and len(matrix.indptr) > 0 and int(matrix.indptr[0]) == 0
        and int(matrix.indptr[-1]) == len(matrix.data))


class _PendingCSR:
    def __init__(self, scope, matrix, lens, lens_slots):
        self.scope, self.base = scope, matrix
        # Original insertion copies numeric values immediately. Capture only
        # these small three-column spans now, before later original producers.
        self.lens = np.array(lens, dtype=np.float64, order='C', copy=True)
        self.lens_slots = np.asarray(lens_slots, np.int64).copy()
        self.mount = self.mount_slots = self.materialized = None

    def materialize(self):
        if self.materialized is None:
            matrix = _ORIGINAL_MOUNT(self.base, self.lens, self.lens_slots, 34204)
            if self.mount is not None:
                matrix = _ORIGINAL_MOUNT(matrix, self.mount, self.mount_slots, 34195)
            self.materialized = matrix
        return self.materialized


def _original_matrix(matrix):
    return matrix.materialize() if type(matrix) is _PendingCSR else matrix


def _insert_mount(matrix, values, indices, mount_begin):
    scope = _SCOPE.get()
    if scope is None:
        return _ORIGINAL_MOUNT(_original_matrix(matrix), values, indices, mount_begin)
    owner = scope['owner']
    if type(matrix) is _PendingCSR:
        if (matrix is scope['pending'] and scope['phase'] == 1 and mount_begin == 34195
                and indices is owner.optics_chart.mount_indices):
            # The first span's declared 34204..34212 domain is disjoint from
            # this original 34195..34203 domain. The original admission prefix
            # therefore sees the exact same shape, sorted flag and relevant
            # indices on the base matrix as on the virtual first insertion.
            _, actual, slots = _MOUNT_ADMISSION(matrix.base, values, indices, mount_begin)
            matrix.mount = np.array(actual, dtype=np.float64, order='C', copy=True)
            matrix.mount_slots = np.asarray(slots, np.int64).copy()
            scope['phase'] = 2
            return matrix
        scope['fallback'] = 'unrecognized_pending_mount_call'
        return _ORIGINAL_MOUNT(matrix.materialize(), values, indices, mount_begin)
    if (scope['phase'] != 0 or mount_begin != 34204 or indices is not owner.optics_chart.indices):
        return _ORIGINAL_MOUNT(matrix, values, indices, mount_begin)
    if not _compatible(matrix):
        scope['fallback'] = 'unsupported_original_CSR_storage'
        return _ORIGINAL_MOUNT(matrix, values, indices, mount_begin)
    actual_matrix, actual, slots = _MOUNT_ADMISSION(matrix, values, indices, mount_begin)
    scope['pending'] = _PendingCSR(scope, actual_matrix, actual, slots)
    scope['phase'] = 1
    return scope['pending']


def _native_merge(pending, rows, values, columns):
    matrix = pending.base
    if pending.mount is None or not _compatible(matrix):
        return None, 'unrecognized_original_three_span_storage'
    if np.any(rows < 0) or np.any(rows >= matrix.shape[0]):
        # The original admission bounds rows by motion_rows. For unusual
        # motion_rows > matrix rows, replay its per-row overlap/index failures
        # in original order instead of preempting them with vector indexing.
        return None, 'selected_rows_require_original_per_row_error_order'
    try:
        threads = int(os.environ.get('RINKSPACE_CSR_MERGE_THREADS', os.cpu_count() or 1))
    except (TypeError, ValueError):
        return None, 'unsupported_native_thread_setting'
    if threads < 1:
        return None, 'unsupported_native_thread_setting'
    pending.scope['actual_native_merge_threads'] = threads
    prepare, fill = _load_native().get(matrix.indices.dtype, (None, None))
    if prepare is None:
        return None, 'matching_native_merge_binary_unavailable'
    selected_at = np.full(matrix.shape[0], -1, np.int64)
    selected_at[rows] = np.arange(len(rows), dtype=np.int64)
    values = np.ascontiguousarray(values, np.float64)
    columns = np.ascontiguousarray(columns, np.int64)
    pointer = np.empty(matrix.shape[0]+1, matrix.indptr.dtype)
    first_overlap = ctypes.c_int64(-1)
    code = prepare(matrix.shape[0], matrix.shape[1], matrix.nnz,
        matrix.indptr.ctypes.data, matrix.indices.ctypes.data,
        len(pending.lens), pending.lens.ctypes.data, pending.lens_slots.ctypes.data,
        len(pending.mount), pending.mount.ctypes.data, pending.mount_slots.ctypes.data,
        selected_at.ctypes.data, len(rows), len(columns), values.ctypes.data, columns.ctypes.data,
        threads, pointer.ctypes.data, ctypes.byref(first_overlap))
    if code:
        return None, ('original_selected_column_overlap' if code == 2
            else 'unsupported_native_CSR_layout_or_index_capacity')
    data = np.empty(int(pointer[-1]), np.float64)
    output_columns = np.empty(len(data), matrix.indices.dtype)
    code = fill(matrix.shape[0], matrix.indptr.ctypes.data, matrix.indices.ctypes.data,
        matrix.data.ctypes.data,
        len(pending.lens), pending.lens.ctypes.data, pending.lens_slots.ctypes.data,
        len(pending.mount), pending.mount.ctypes.data, pending.mount_slots.ctypes.data,
        selected_at.ctypes.data, len(columns), values.ctypes.data, columns.ctypes.data,
        threads, pointer.ctypes.data, output_columns.ctypes.data, data.ctypes.data)
    if code:
        return None, 'native_output_inventory_not_admitted'
    result = csr_matrix((data, output_columns, pointer), shape=matrix.shape, copy=False)
    result.has_sorted_indices = matrix.has_sorted_indices
    result.has_canonical_format = matrix.has_canonical_format
    return result, None


def _insert_selected(matrix, rows, values, *, motion_rows, geometry_columns=None):
    scope = _SCOPE.get()
    if (scope is None or type(matrix) is not _PendingCSR or matrix is not scope['pending']
            or scope['phase'] != 2):
        return _ORIGINAL_SELECTED(_original_matrix(matrix), rows, values,
            motion_rows=motion_rows, geometry_columns=geometry_columns)
    # Original head validation and stable column sorting run before the native
    # overlap scan. Native rejection replays the actual original functions, so
    # the original first error and any unusual storage behaviour remain owned
    # by their original implementation.
    _, actual_rows, actual_values, columns = _SELECTED_ADMISSION(matrix.base, rows, values,
        motion_rows=motion_rows, geometry_columns=geometry_columns)
    result, reason = _native_merge(matrix, actual_rows, actual_values, columns)
    scope['phase'] = 3
    if result is None:
        scope['fallback'] = reason
        return _ORIGINAL_SELECTED(matrix.materialize(), rows, values,
            motion_rows=motion_rows, geometry_columns=geometry_columns)
    scope['merged'] = True
    scope['selected_motion_rows'] = len(actual_rows)
    scope['result_rows'], scope['result_nnz'] = result.shape[0], result.nnz
    return result


def install_programme_csr_three_span_merge_runtime(*, report_directory=None):
    """Install after camera reuse/prefetch, only before a new owned run."""
    global _RECEIPT, _REPORT, _ORIGINAL_MOUNT, _ORIGINAL_SELECTED
    global _MOUNT_ADMISSION, _SELECTED_ADMISSION
    if _RECEIPT is not None:
        return _RECEIPT
    from . import install
    install()
    from . import camera_absolute_reuse_runtime as camera_runtime
    from structured_programme_source_candidate.paired_programme_blocks import StructuredProgrammeBlock
    from installation_mount_candidate.mount_graph import insert_mount_csr
    if camera_runtime._RECEIPT is None:
        raise TypeError('Original camera15 reuse installation required before CSR merge')
    original = StructuredProgrammeBlock._evaluate
    caller = inspect.unwrap(original)
    caller_sha = _authenticate_registered_tail(caller)
    mount = caller.__globals__.get('insert_mount_csr')
    selected = caller.__globals__.get('insert_selected_motion_geometry_csr')
    if mount is not insert_mount_csr or not inspect.isfunction(selected):
        raise TypeError('Original registered CSR insertion function identities required')
    mount_admission, mount_sha = _admission_prefix(mount, 'active', '(m, values, slots)')
    selected_admission, selected_sha = _admission_prefix(selected, 'counts', '(m, rr, v, columns)')
    backend = metadata()
    directory = None if report_directory is None else Path(report_directory).resolve()
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    _ORIGINAL_MOUNT, _ORIGINAL_SELECTED = mount, selected
    _MOUNT_ADMISSION, _SELECTED_ADMISSION = mount_admission, selected_admission

    @functools.wraps(original)
    def evaluate(owner, q, jacobian=True):
        supported = (jacobian and type(owner) is StructuredProgrammeBlock
            and _SCOPE.get() is None and bool(_load_native()))
        scope = (dict(owner=owner, phase=0, pending=None, merged=False, fallback=None,
                      selected_motion_rows=0, result_rows=0, result_nnz=0,
                      actual_native_merge_threads=None) if supported else None)
        token = _SCOPE.set(scope)
        started = time.perf_counter()
        succeeded = False
        try:
            result = original(owner, q, jacobian)
            succeeded = True
            return result
        finally:
            _SCOPE.reset(token)
            if scope is not None and _REPORT is not None:
                pending = scope['pending']
                value = dict(block=owner.name, pid=os.getpid(), original_call_succeeded=succeeded,
                    original_CSR_spans_merged=scope['merged'], fallback_reason=scope['fallback'],
                    source_rows=scope['result_rows'], result_nnz=scope['result_nnz'],
                    selected_motion_rows=scope['selected_motion_rows'],
                    actual_native_merge_threads=scope['actual_native_merge_threads'],
                    captured_three_column_bytes=0 if pending is None else
                        pending.lens.nbytes+(0 if pending.mount is None else pending.mount.nbytes),
                    complete_programme_seconds=time.perf_counter()-started,
                    no_float_arithmetic_or_source_math_changed=True,
                    wall_time_ns=time.time_ns())
                try:
                    with (_REPORT/'programme_csr_three_span_merge.jsonl').open('a', encoding='utf8') as stream:
                        stream.write(json.dumps(value)+'\n')
                except OSError:
                    pass
    # Preserve the installed material wrapper around the authenticated body.
    # Only the original body's two existing insertion aliases are redirected.
    caller.__globals__['insert_mount_csr'] = _insert_mount
    caller.__globals__['insert_selected_motion_geometry_csr'] = _insert_selected
    StructuredProgrammeBlock._evaluate = evaluate
    _REPORT = directory
    _RECEIPT = dict(schema='rinkspace.original-programme-three-CSR-spans-one-native-merge.v1',
        helper_path=str(Path(__file__).resolve()),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_backend=backend, original_caller_source_sha256=caller_sha,
        original_mount_insert_source_sha256=mount_sha, original_selected_insert_source_sha256=selected_sha,
        original_registered_body_bytecode_suffix_authenticated=True,
        original_lens_mount_course_math_and_error_boundaries_retained=True,
        original_admission_AST_prefixes_execute=True,
        inserted_numeric_values_copied_at_original_consume_points=True,
        sorted_row_merge_copies_double_bytes_without_float_arithmetic=True,
        independent_thread_setting='RINKSPACE_CSR_MERGE_THREADS then RINKSPACE_SCATTER_THREADS then os.cpu_count()',
        unknown_call_dtype_layout_or_backend_uses_original_functions=True,
        complete_cycle_source_bytes_and_speedup_pending=True)
    return _RECEIPT
