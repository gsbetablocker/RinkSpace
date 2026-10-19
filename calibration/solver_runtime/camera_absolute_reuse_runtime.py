"""Reuse an unchanged camera15 source evaluation for its lens/mount columns.

The original programme evaluates the same absolute rows twice at the same
camera. The first result supplies camera columns, the second supplies lens and
mount columns. Only exact, ordered input matches within one call are reused.
"""
from __future__ import annotations

import ast
from contextvars import ContextVar
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import textwrap

import numpy as np


_SCOPE = ContextVar('rinkspace_camera15_result_reuse', default=None)
_UNSUPPORTED = object()
_RECEIPT = None


def _literal(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.fields is not None:
            return _UNSUPPORTED
        literal = ('array', value.dtype.str, value.shape, value.tobytes())
        scope = _SCOPE.get()
        if scope is not None:
            # Repeated groups share material arrays. Keep one immutable byte
            # value per exact array rather than duplicating each row's world.
            return scope['arrays'].setdefault(literal, literal)
        return literal
    if isinstance(value, np.generic):
        return _literal(np.asarray(value))
    if type(value) in (tuple, list):
        parts = tuple(_literal(part) for part in value)
        if any(part is _UNSUPPORTED for part in parts):
            return _UNSUPPORTED
        return (type(value).__name__, parts)
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            return _UNSUPPORTED
        parts = tuple((key, _literal(part)) for key, part in value.items())
        if any(part is _UNSUPPORTED for _, part in parts):
            return _UNSUPPORTED
        return ('dict', parts)
    if value is None or type(value) in (str, bytes, int, bool):
        return (type(value).__name__, value)
    if type(value) is float:
        return ('float', np.float64(value).tobytes())
    return _UNSUPPORTED


def _capture(bundle, camera, rows, jacobian):
    result = bundle.absolute_evaluator(camera, rows, jacobian)
    scope = _SCOPE.get()
    if scope is None or bundle is not scope['owner'].bundle or not jacobian:
        return result
    camera_key = _literal(np.asarray(camera))
    row_key = _literal(rows)
    saved = None
    if (camera_key is not _UNSUPPORTED and row_key is not _UNSUPPORTED
            and type(result) is tuple and len(result) == 2
            and all(isinstance(part, np.ndarray) and not part.dtype.hasobject for part in result)):
        saved = (bundle.absolute_evaluator, camera_key, row_key,
                 tuple(part.copy() for part in result))
    scope['results'].append(saved)
    return result


def _reuse(owner, camera, rows, jacobian):
    scope = _SCOPE.get()
    if scope is None or owner is not scope['owner'] or not jacobian:
        return owner.absolute(camera, rows, jacobian)
    position = scope['position']
    scope['position'] += 1
    saved = scope['results'][position] if position < len(scope['results']) else None
    if (saved is not None and saved[0] is owner.absolute
            and saved[1] == _literal(np.asarray(camera)) and saved[2] == _literal(rows)):
        scope['hits'] += 1
        return tuple(part.copy() for part in saved[3])
    scope['fallbacks'] += 1
    return owner.absolute(camera, rows, jacobian)


def _replace_call(original, match, replacement, name, helper):
    original = inspect.unwrap(original)
    if original.__code__.co_freevars:
        raise TypeError('Source-owned closure-free camera source function required')
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    wanted = ast.dump(ast.parse(match, mode='eval').body, include_attributes=False)
    sites = []

    class Replace(ast.NodeTransformer):
        def visit_Call(self, node):
            if ast.dump(node, include_attributes=False) == wanted:
                sites.append(node.lineno)
                return ast.copy_location(ast.parse(replacement, mode='eval').body, node)
            return self.generic_visit(node)

    tree = Replace().visit(tree)
    if len(sites) != 1:
        raise TypeError('One original camera15 source call required: ' + str(sites))
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, original.__code__.co_firstlineno - 1)
    original.__globals__[name] = helper
    namespace = {}
    exec(compile(tree, inspect.getsourcefile(original), 'exec'), original.__globals__, namespace)
    original.__code__ = namespace[original.__name__].__code__
    return hashlib.sha256(source.encode()).hexdigest()


def install_camera_absolute_reuse_runtime(*, report_directory=None):
    """Install after the material scope; retain that existing outer wrapper."""
    global _RECEIPT
    if _RECEIPT is not None:
        return _RECEIPT
    from structured_programme_source_candidate.paired_programme_blocks import StructuredProgrammeBlock
    from z190_fixed_lever_xyz_source_bundle import FixedLeverXYZSourceBundle

    bundle_sha = _replace_call(FixedLeverXYZSourceBundle.evaluate,
        'self.absolute_evaluator(cam, rows, True)',
        '_rinkspace_capture_camera15(self, cam, rows, True)',
        '_rinkspace_capture_camera15', _capture)
    programme_sha = _replace_call(StructuredProgrammeBlock._evaluate,
        'self.absolute(cam, moved, True)',
        '_rinkspace_reuse_camera15(self, cam, moved, True)',
        '_rinkspace_reuse_camera15', _reuse)
    original_evaluate = StructuredProgrammeBlock.evaluate
    original_text = textwrap.dedent(inspect.getsource(original_evaluate))
    if "if not profiling:return self._evaluate(q,jacobian)" not in original_text:
        raise TypeError('Original optional first-J profiler entry changed')

    @functools.wraps(original_evaluate)
    def evaluate(owner, q, jacobian=True):
        if not jacobian:
            return owner._evaluate(q, jacobian)
        scope = dict(owner=owner, results=[], arrays={}, position=0, hits=0, fallbacks=0)
        token = _SCOPE.set(scope)
        try:
            if os.environ.get('RINKSPACE_SOLVER_PROFILE') == '1':
                return original_evaluate(owner, q, jacobian)
            return owner._evaluate(q, jacobian)
        finally:
            _SCOPE.reset(token)
            if report_directory is not None:
                event = dict(block=owner.name, captured_groups=len(scope['results']),
                    repeated_camera15_calls_reused=scope['hits'],
                    changed_or_unsupported_inputs_original_calls=scope['fallbacks'])
                with (Path(report_directory) / 'camera15_reuse.jsonl').open('a', encoding='utf8') as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + '\n')

    StructuredProgrammeBlock.evaluate = evaluate
    path = Path(__file__).resolve()
    _RECEIPT = dict(schema='rinkspace.exact-camera15-two-pass-reuse.v1',
        helper_path=str(path), helper_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        original_bundle_evaluate_sha256=bundle_sha, original_programme_evaluate_sha256=programme_sha,
        original_profile_entry_sha256=hashlib.sha256(original_text.encode()).hexdigest(),
        cache_scope='One original programme call; original group order, callable, camera and all source row bytes',
        changed_or_unsupported_input_executes_original=True, original_FD_callable_and_worker_protocol_unchanged=True,
        original_binding_mapping_masks_sigma_lens_mount_and_CSR_assembly_retained=True,
        repeated_first_J_cProfile_disabled_by_default=True,
        explicit_profiler_environment='RINKSPACE_SOLVER_PROFILE=1', existing_profiler_evidence_retained=True,
        registered_source_files_modified=False, accuracy_accepted=False)
    if report_directory is not None:
        destination = Path(report_directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'camera_absolute_reuse_receipt.json').write_text(
            json.dumps(_RECEIPT, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return _RECEIPT
