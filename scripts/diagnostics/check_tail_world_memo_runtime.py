"""Small new-hook contract checks; no graph, video, or speed benchmark."""
from __future__ import annotations
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from calibration.solver_runtime.tail_world_memo_runtime import _wrap_factor, _scopes


class Geometry:
    def __init__(self):
        self.calls = 0

    def validate_parameters(self, world):
        self.calls += 1
        if not np.isfinite(world).all() or world[1] <= 0:
            raise ValueError('Original physical world guard')


class Factor:
    def __init__(self):
        self.geometry = Geometry()
        self.world_indices = np.arange(6)
        self.calls = 0
        self.curve = {'kind': 'complete_finite_curve', 'scope': 'original', 'role': 'source'}

    def _world(self, q, curve=None):
        self.calls += 1
        p = np.asarray(q)[self.world_indices]
        self.geometry.validate_parameters(p)
        return np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]]])

    def _residual(self, q, curve):
        world = self._world(q, curve)
        # A caller changing its derived array must not contaminate other FD
        # evaluations or the memo's owned immutable copy.
        original = world.copy()
        world[0, 1] += q[6]
        return np.r_[world.ravel(), original.ravel()]

    def block(self, q, jacobian=True, *, original_start=0):
        if q[7] == 99:
            self._world(q, self.curve)
            raise RuntimeError('Original producer failed')
        r = self._residual(q, self.curve)
        # Repeat nominal inputs as real source nominal/branch evaluation does.
        assert np.array_equal(r, self._residual(q, self.curve))
        matrix = None
        if jacobian:
            columns = []
            for col in range(len(q)):
                p, m = q.copy(), q.copy()
                p[col] += 1e-6
                m[col] -= 1e-6
                columns.append((self._residual(p, self.curve)-self._residual(m, self.curve))/(2e-6))
            matrix = csr_matrix(np.column_stack(columns))
        return SimpleNamespace(r=r, J=matrix, ledger={'original_start': original_start, 'curve': self.curve})


def exact_array(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert memoryview(np.ascontiguousarray(a)).cast('B') == memoryview(np.ascontiguousarray(b)).cast('B')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    opt = parser.parse_args()
    q = np.array([-0., 2., 3., 4., 5., 6., .25, 0.])
    original = Factor().block(q, True, original_start=5)
    receipt = _wrap_factor(Factor, None)
    factor = Factor()
    actual = factor.block(q, True, original_start=5)
    exact_array(actual.r, original.r)
    for field in ('data', 'indices', 'indptr'):
        exact_array(getattr(actual.J, field), getattr(original.J, field))
    assert actual.ledger == original.ledger
    assert factor._rinkspace_last_tail_world_memo_report['hits'] > 0
    assert factor.calls < 18
    assert factor.geometry.calls == factor.calls  # No added hit-only global guard.
    assert not _scopes()
    before = factor.calls
    factor.block(q, True)
    assert factor.calls > before  # No cross-call source reuse.
    bad = q.copy(); bad[7] = 99
    try:
        factor.block(bad, True)
    except RuntimeError as error:
        assert str(error) == 'Original producer failed'
    else:
        raise AssertionError('Original failure was swallowed')
    assert not _scopes() and not factor._rinkspace_last_tail_world_memo_report['succeeded']
    bad_world = q.copy(); bad_world[1] = 0.
    try:
        factor.block(bad_world, True)
    except ValueError as error:
        assert str(error) == 'Original physical world guard'
    else:
        raise AssertionError('Original physical guard was swallowed')
    assert not _scopes()
    factor.block(q, False)
    assert not _scopes()
    # Original concrete world producer: only its minimal physical course is
    # a fixture; the actual NativeBoardEdgeFactor._world function runs.
    from calibration.solver_runtime import install
    install()
    from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
    from calibration.solver_runtime.tail_world_memo_runtime import install_tail_world_memo_runtime
    course = SimpleNamespace(sample_perimeter=lambda value, **kwargs:
        (np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),
         np.array([[0.,0.,value[0]],[1.,0.,value[0]],[0.,1.,value[0]]]), None))
    concrete = NativeBoardEdgeFactor.__new__(NativeBoardEdgeFactor)
    concrete.world_indices = np.arange(6)
    concrete.geometry = SimpleNamespace(validate_parameters=Geometry().validate_parameters,
        print_course_binding=lambda value:(course,value,None))
    concrete._physical_course_scope_overrides = {}
    curve = {'kind':'board_top','source_structure':'far_visible_board','complete_source_role':'original'}
    before_world = concrete._world(q, curve)
    live_receipt = install_tail_world_memo_runtime()
    # Check real source wrapper dispatch outside memo still invokes the same
    # concrete original function without intercepting review/guard behavior.
    exact_array(concrete._world(q, curve), before_world)
    # Exercise the installed concrete producer through the exact active scope
    # used by its block wrapper, without constructing a complete source graph.
    scope = dict(factor=concrete, values={}, hits=0, misses=0, stored_array_bytes=0)
    _scopes()[id(concrete)] = scope
    try:
        nominal = concrete._world(q, curve)
        exact_array(nominal, before_world)
        nominal[0, 2] = 123.  # Returned miss result does not own cached storage.
        exact_array(concrete._world(q, curve), before_world)
        changed_camera = q.copy(); changed_camera[6] += 1.
        exact_array(concrete._world(changed_camera, curve), before_world)
        positive_zero = q.copy(); positive_zero[0] = 0.
        concrete._world(positive_zero, curve)
        concrete._world(q, dict(curve, complete_source_role='another_original_role'))
        assert scope['hits'] == 2 and scope['misses'] == 3
        assert len(scope['values']) == 3
    finally:
        _scopes().pop(id(concrete), None)
        scope['values'].clear()
    report = dict(schema='rinkspace.tail-world-memo-small-contract-check.v1', passed=True,
        exact_residual_CSR_and_ledger=True, independent_array_aliasing_checked=True,
        original_failure_and_scope_cleanup_checked=True, no_cross_call_reuse=True,
        original_physical_guard_failure_retained=True,
        added_global_guard_calls_on_memo_hits=0,
        original_concrete_board_world_function_checked=True,
        full_curve_identity_and_signed_zero_world_keys_distinct=True,
        actual_installed_world_memo_hit_path_checked=True,
        actual_complete_graph_source_J_byte_equivalence_pending=True,
        fixture_receipt=receipt, real_class_installation=live_receipt)
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    opt.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'passed':True,'output':str(opt.output)}))


if __name__ == '__main__':
    main()
