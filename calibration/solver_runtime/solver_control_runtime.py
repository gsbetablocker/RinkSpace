"""Observe the original explicit control file during an active Clarabel solve.

No numerical termination limit is introduced. Outside a source-control scope,
the original constructor and solver run unchanged. A requested stop is kept
separate from numerical failures so QP fallbacks cannot consume it.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import functools
import hashlib
import json
from pathlib import Path
import time


class SourceControlStop(BaseException):
    """An explicit control request, never a failed numerical trial."""

    def __init__(self, stop):
        super().__init__(stop['reason'])
        self.stop = stop

    def __reduce__(self):
        # Worker transport must reconstruct the complete explicit stop record,
        # rather than passing the exception's human-readable string to __init__.
        return type(self), (self.stop,)


_active_control = ContextVar('rinkspace_explicit_source_control', default=None)


class SourceControlContext:
    def __init__(self, output, reader):
        self.output = Path(output).resolve()
        self.reader = reader

    def poll(self, info=None):
        return self.reader(self.output)


@contextmanager
def source_control_context(output, *, reader=None):
    if reader is None:
        from z190_joint_source_solver import _control_stop
        reader = _control_stop
    context = SourceControlContext(output, reader)
    token = _active_control.set(context)
    try:
        yield context
    finally:
        _active_control.reset(token)


class _ControlledSolver:
    def __init__(self, solver, context):
        self._solver = solver
        self._context = context

    def __getattr__(self, name):
        return getattr(self._solver, name)

    def solve(self):
        requested = self._context.poll()
        if requested is not None:
            raise SourceControlStop(requested)
        callback_failure = None

        def observe(info):
            nonlocal requested, callback_failure
            # Never propagate a Python exception through the Rust callback.
            try:
                requested = self._context.poll(info)
                if requested is not None:
                    print(json.dumps(dict(stage='source_control_observed_in_QP',
                        wall_time_ns=time.time_ns(),
                        iterations=int(info.iterations),
                        reason=requested['reason'],
                        save_current_linearization_cache=requested['control'].get(
                            'save_current_linearization_cache') is True)), flush=True)
                return requested is not None
            except BaseException as exc:
                callback_failure = exc
                return True

        self._solver.set_termination_callback(observe)
        try:
            solution = self._solver.solve()
        finally:
            self._solver.unset_termination_callback()
        if callback_failure is not None:
            raise callback_failure
        if requested is not None:
            raise SourceControlStop(requested)
        # Cover a request arriving between the final callback and return.
        requested = self._context.poll()
        if requested is not None:
            raise SourceControlStop(requested)
        return solution


def install_solver_control_runtime():
    import clarabel
    original = clarabel.DefaultSolver
    if hasattr(original, '_rinkspace_source_control_receipt'):
        return original._rinkspace_source_control_receipt

    @functools.wraps(original)
    def controlled_solver(*args, **kwargs):
        context = _active_control.get()
        if context is not None:
            requested = context.poll()
            if requested is not None:
                raise SourceControlStop(requested)
        solver = original(*args, **kwargs)
        return solver if context is None else _ControlledSolver(solver, context)

    receipt = dict(schema='rinkspace.explicit-source-control-in-QP.v1',
        helper_path=str(Path(__file__).resolve()),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        clarabel_version=clarabel.__version__,
        original_valid_explicit_control_file_semantics_retained=True,
        backend_termination_callback_checks_each_IPM_iteration=True,
        no_numerical_timeout_or_iteration_limit_added=True,
        cancellation_is_BaseException_not_numerical_failure=True,
        original_checkpoint_save_result_and_driver_cleanup_retained=True)
    controlled_solver._rinkspace_source_control_receipt = receipt
    clarabel.DefaultSolver = controlled_solver
    return receipt
