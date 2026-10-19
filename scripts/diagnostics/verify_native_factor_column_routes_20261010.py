"""Tiny integration/error routing check; no graph, solver or process benchmark."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.solver_runtime import install
install()
import numpy as np
from calibration.solver_runtime import native_factor_columns_runtime as native
from calibration.solver_runtime.confirmed_constraint_columns_runtime import (
    _check_pattern_numerical_closure, _PrivateContextAdmissionError, _PrivateSourceIdentityError)
from calibration.solver_runtime.absolute_geometry_fd_pool_candidate import AbsoluteGeometryGroupPool
from calibration.solver_runtime.solver_control_runtime import SourceControlStop


class Future:
    def __init__(self, job, mode):
        self.job, self.mode = job, mode
    def result(self):
        if self.mode == 'admission':
            raise _PrivateContextAdmissionError('test-private-hydrate')
        if self.mode == 'control':
            raise SourceControlStop({'reason': 'explicit-fixture-control', 'save_current_linearization_cache': True})
        if self.mode == 'identity':
            raise _PrivateSourceIdentityError('test-source-changed')
        result = []
        for position in self.job['positions']:
            column = int(self.job['active'][position])
            error = ValueError('original-column-' + str(position)) if self.mode == 'math' else None
            result.append((position, np.array([float(column)]),
                dict(column=column, source_groups=1, evaluated_scalar_rows=1), error))
        return dict(context_sha256=self.job['context_sha256'], q_sha256=self.job['q_sha256'],
                    material_sha256='fixture', pid=1, results=result)
    def done(self):
        return True
    def cancel(self):
        return False


class Executor:
    def __init__(self, mode):
        self.mode, self.closed = mode, False
    def submit(self, function, job):
        return Future(job, self.mode)
    def shutdown(self, **kwargs):
        self.closed = True


def main():
    _check_pattern_numerical_closure([], [])  # Exercise the corrected real lazy import.
    saved_closure, saved_original = native._closure_context, native._original_fd
    serial_calls, checks = [], []
    try:
        def original(factor, q, rows, active, material, base, evaluate, heights):
            serial_calls.append(np.asarray(active).tolist())
            return np.array([np.asarray(active, float)])
        native._original_fd = lambda: original
        for mode in ('success', 'admission', 'math', 'control', 'identity'):
            serial_calls.clear()
            pool = AbsoluteGeometryGroupPool(dict(path='unused', sha256='fixture', serialized_bytes=1), workers=2)
            executor = pool.executor = Executor(mode)
            pool._rinkspace_native_columns_idle_receipt = dict(executor_identity=id(executor), pending_jobs=0)
            native._closure_context = lambda *args: (('fixture-context', None), pool)
            source = SimpleNamespace()
            error = None
            try:
                result = native._parallel_native_fd(source, np.ones(2), [], np.array([0, 1]), [],
                    np.ones(1), lambda q, groups: np.ones(1), [])
            except BaseException as observed:
                error = observed
            assert not pool._rinkspace_native_columns_active and pool._rinkspace_native_columns_pending_jobs == 0
            if mode == 'success':
                assert error is None and serial_calls == [[1]]
                assert result.tobytes() == np.array([[0., 1.]]).tobytes()
                assert not executor.closed and native._pool_available(pool)
                for name, value in (('_rinkspace_confirmed_columns_active', True),
                    ('_rinkspace_native_columns_active', True),
                    ('_rinkspace_shared_material_bank_precompute_active', True),
                    ('_rinkspace_shared_material_bank_pending_jobs', 1)):
                    setattr(pool, name, value)
                    assert not native._pool_available(pool)
                    delattr(pool, name)
                pool.close()
            elif mode == 'admission':
                assert error is None and serial_calls == [[0, 1]] and executor.closed
            elif mode == 'math':
                assert isinstance(error, ValueError) and str(error) == 'original-column-0'
                assert serial_calls == [] and executor.closed
            elif mode == 'control':
                assert isinstance(error, SourceControlStop) and error.stop['save_current_linearization_cache']
                assert serial_calls == [] and executor.closed
                restored = pickle.loads(pickle.dumps(error))
                assert type(restored) is SourceControlStop and restored.stop == error.stop
            else:
                assert isinstance(error, _PrivateSourceIdentityError) and serial_calls == [] and executor.closed
            checks.append(dict(route=mode, passed=True))
        report = dict(passed=True, cases=checks, corrected_real_course_lazy_import_passed=True,
            original_math_error_order_and_control_BaseException_without_serial_fallback=True,
            admission_failure_drains_owned_jobs_then_complete_original_serial=True,
            three_protocol_busy_and_pending_flags_rejected=True,
            source_identity_error_never_serial_fallback=True)
        destination = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/diagnostics/current_solver_takeover_20261009/native_columns_small')
        (destination / 'native_columns_route_check.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
        print(json.dumps(report), flush=True)
        print(json.dumps({name: hashlib.sha256((ROOT / 'calibration/solver_runtime' / name).read_bytes()).hexdigest()
                          for name in ('native_factor_columns_runtime.py', 'confirmed_constraint_columns_runtime.py')}))
    finally:
        native._closure_context, native._original_fd = saved_closure, saved_original


if __name__ == '__main__':
    main()
