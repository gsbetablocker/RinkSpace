"""Persist only optional SOC separation row indices in their own namespace.

Rows are scheduling hints, including when the preceding optional correction
did not return a direction. Every current problem still executes its original
row separation, full KKT gate and subsequent source guards. Initial-QP hint
files and the original process-local verified hint are never replaced by SOC.
"""
from __future__ import annotations

from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
from .original_halfspace_primal import PRIMAL_CERTIFICATE_SCHEMA

_ATTRIBUTE = '_last_verified_separation_faces'
_SCHEMA = 'rinkspace.optional-SOC-row-index-scheduling-hint.v2'
_MISSING = object()
_RECEIPT = None
_EVENTS = []


def _validated_rows(value, count):
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.dtype.kind not in 'iu':
            raise ValueError('SOC hint must contain one-dimensional integer row indices')
        value = value.tolist()
    if type(value) not in (tuple, list):
        raise ValueError('SOC hint must contain row indices')
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
           or not 0 <= int(i) < count for i in value):
        raise ValueError('SOC hint row index is outside the complete current rows')
    return tuple(sorted({int(i) for i in value}))


def _shape(gradient, lower, upper, constraint_lower):
    return (len(constraint_lower) + int(np.isfinite(lower).sum())
        + int(np.isfinite(upper).sum()), len(gradient))


def _hint_path(directory, scope, shape):
    identity = dict(namespace=_SCHEMA, scope=scope, shape=list(shape),
                    primal_gate=PRIMAL_CERTIFICATE_SCHEMA)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return directory / ('soc_' + key + '.json')


def _record(value, report_directory):
    _EVENTS.append(value)
    try:
        print(json.dumps(dict(stage='optional_SOC_persistent_row_hint', **value)), flush=True)
        if report_directory is not None:
            report_directory.mkdir(parents=True, exist_ok=True)
            with (report_directory / 'SOC_row_hint_events.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(value) + '\n')
    except Exception:
        pass  # Optional reporting cannot mask a current original numeric error.


def _wrap(original, soc_original, holder, context_variable, directory,
          report_directory=None):
    @wraps(original)
    def solve(matrix, gradient, lower, upper, constraint_matrix, constraint_lower):
        context = context_variable.get()
        if context is None:
            return original(matrix, gradient, lower, upper, constraint_matrix, constraint_lower)
        # The original persistent layer writes its current holder after every
        # successful full KKT solve. Bypass only this persistence layer for SOC,
        # so its initial-QP file cannot acquire the optional correction's hint.
        args = (matrix, gradient, lower, upper, constraint_matrix, constraint_lower)
        try:
            shape = _shape(gradient, lower, upper, constraint_lower)
            scope = os.environ.get('RINKSPACE_SOLVER_DATA_ROOT')
            path = _hint_path(directory, scope, shape)
        except Exception:
            return soc_original(*args)
        previous = getattr(holder, _ATTRIBUTE, _MISSING)
        context.pop('last_restricted_face_rows', None)
        try:
            try:
                saved = json.loads(path.read_text(encoding='utf8'))
                if (saved['schema'] != _SCHEMA or saved.get('primal_gate') != PRIMAL_CERTIFICATE_SCHEMA
                        or saved['scope'] != scope
                        or saved['shape'] != list(shape)
                        or saved['authority'] != 'scheduling_only'
                        or saved['solution_or_duals_stored'] is not False):
                    raise ValueError('SOC scheduling hint namespace/scope/shape changed')
                rows = _validated_rows(saved['rows'], shape[0])
                setattr(holder, _ATTRIBUTE, dict(shape=shape, scope=scope, rows=rows))
                _record(dict(action='loaded_SOC_row_indices', rows=len(rows),
                    shape=list(shape), scope=scope, complete_current_KKT_still_required=True), report_directory)
            except (OSError, ValueError, KeyError, TypeError) as error:
                _record(dict(action='original_SOC_separation', reason=type(error).__name__,
                    shape=list(shape), scope=scope), report_directory)
            return soc_original(*args)
        finally:
            # Save only the row coordinates observed by this current original
            # restricted call, never a q/H/g/step/dual or its outcome/status.
            value = context.get('last_restricted_face_rows', _MISSING)
            try:
                if value is not _MISSING:
                    rows = _validated_rows(value, shape[0])
                    payload = dict(schema=_SCHEMA, scope=scope, shape=list(shape),
                        primal_gate=PRIMAL_CERTIFICATE_SCHEMA,
                        rows=list(rows), solution_or_duals_stored=False,
                        authority='scheduling_only')
                    directory.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix('.' + str(os.getpid()) + '.tmp')
                    temporary.write_text(json.dumps(payload) + '\n', encoding='utf8')
                    temporary.replace(path)
                    _record(dict(action='saved_observed_SOC_row_indices', rows=len(rows),
                        shape=list(shape), scope=scope, accepted_solution_claimed=False), report_directory)
            except (OSError, ValueError, KeyError, TypeError) as error:
                _record(dict(action='optional_SOC_hint_save_failed', reason=type(error).__name__), report_directory)
            finally:
                # Restore the actual pre-call verified hint even after a SOC
                # success, optional numerical failure or cancellation.
                if previous is _MISSING:
                    if hasattr(holder, _ATTRIBUTE):
                        delattr(holder, _ATTRIBUTE)
                else:
                    setattr(holder, _ATTRIBUTE, previous)
    solve._rinkspace_persistent_SOC_row_hint = True
    return solve


def install_persistent_soc_row_hint_runtime(*, cache_directory, report_directory=None):
    """Install after optional SOC and the initial-QP persistent outer layer.

The root-owned restricted backend logger must set
``_optional_soc.get()['last_restricted_face_rows']`` to the exact original
``selected`` row coordinates at each restricted call. No logger change is
made here.
"""
    global _RECEIPT
    if _RECEIPT is not None:
        return dict(_RECEIPT)
    import z190_global_schur_linear_step as qp
    from calibration.solver_runtime.optional_soc_runtime import _optional_soc
    original = qp._linearly_constrained_shared_step
    if original.__globals__.get('__name__') != 'calibration.solver_runtime.persistent_qp_face_hint_runtime':
        raise RuntimeError('SOC row hints require the existing initial-QP persistence outer layer')
    closure = inspect.getclosurevars(original).nonlocals
    soc_original = closure.get('original')
    holder = closure.get('holder')
    if (not callable(soc_original) or not callable(holder)
            or getattr(original, '__wrapped__', None) is not soc_original):
        raise RuntimeError('Original persistent QP numeric function and holder required')
    directory = Path(cache_directory)
    report = None if report_directory is None else Path(report_directory)
    qp._linearly_constrained_shared_step = _wrap(original, soc_original, holder,
        _optional_soc, directory, report)
    _RECEIPT = dict(schema='rinkspace.persist-separate-optional-SOC-row-hints.v1',
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cache_directory=str(directory), scheduling_hint_schema=_SCHEMA,
        initial_QP_persistence_unchanged=True, SOC_bypasses_only_initial_hint_IO=True,
        original_holder_restored_after_every_SOC=True,
        original_current_H_g_bounds_constraints_full_KKT_and_guards_unchanged=True,
        previous_q_H_g_step_dual_and_failure_conclusion_not_stored=True,
        selected_rows_capture='optional_soc context last_restricted_face_rows',
        registered_source_modified=False)
    return dict(_RECEIPT)


def snapshot_persistent_soc_row_hint_runtime():
    return dict(installation=None if _RECEIPT is None else dict(_RECEIPT), events=list(_EVENTS))
