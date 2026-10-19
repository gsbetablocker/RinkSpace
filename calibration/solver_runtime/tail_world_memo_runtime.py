"""Reuse exact world curves only inside one original native factor block.

Camera finite differences repeatedly ask the original ice/board producer for
the identical world curve. The source r/J, dependency stencil, support branch,
and sparse assembly are unchanged. No residual, Jacobian, or world result
survives the block call, and callers receive an independent array each time.
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
from pathlib import Path
import threading

import numpy as np


_LOCAL = threading.local()


def _scopes():
    if not hasattr(_LOCAL, 'scopes'):
        _LOCAL.scopes = {}
    return _LOCAL.scopes


def _curve_key(curve):
    # Retain every field, including physical scope/role metadata. No semantic
    # broadening between curves with similar names or geometry is permitted.
    return json.dumps(curve, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(',', ':'))


def _wrap_factor(cls, report_directory):
    previous = getattr(cls, '_rinkspace_tail_world_memo_receipt', None)
    if previous is not None:
        return previous
    original_world, original_block = cls._world, cls.block
    receipt = dict(
        factor_class=cls.__module__ + '.' + cls.__name__,
        original_world_sha256=hashlib.sha256(inspect.getsource(original_world).encode()).hexdigest(),
        original_block_sha256=hashlib.sha256(inspect.getsource(original_block).encode()).hexdigest(),
        original_block_FD_residual_support_and_CSR_assembly_retained=True,
        cache_key='all original world coordinates as float64 bytes + complete curve JSON',
        cache_lifetime='one original block call; discarded in finally, including failures',
        cache_hit_returns_independent_array=True,
        original_producer_physical_guards_retained_on_exact_key_misses=True,
        extra_full_world_guard_on_camera_only_memo_hits=False,
        cross_state_J_or_residual_reuse=False,
    )

    @functools.wraps(original_world)
    def world(self, q, curve=None):
        scope = _scopes().get(id(self))
        if scope is None or scope['factor'] is not self:
            return original_world(self, q, curve)
        values = np.asarray(q, float)[self.world_indices]
        key = (values.shape, values.dtype.str, values.tobytes(), _curve_key(curve))
        cached = scope['values'].get(key)
        if cached is not None:
            # This exact world and complete curve already passed the original
            # producer in this same block. Calling a different full-world
            # validator here adds work absent from e.g. the original ice
            # producer. Original block/FD optical and physical checks remain.
            scope['hits'] += 1
            return cached.copy()
        result = original_world(self, q, curve)
        if not isinstance(result, np.ndarray) or result.dtype.hasobject:
            raise TypeError('The original native world producer must return a numeric ndarray')
        stored = result.copy()
        stored.setflags(write=False)
        scope['values'][key] = stored
        scope['misses'] += 1
        scope['stored_array_bytes'] += stored.nbytes
        return result

    @functools.wraps(original_block)
    def block(self, q, jacobian=True, *, original_start=0):
        if not jacobian:
            return original_block(self, q, jacobian, original_start=original_start)
        scopes = _scopes()
        previous_scope = scopes.get(id(self))
        # Existing nested evaluation, if introduced by an original producer,
        # stays inside the same exact-key lifetime rather than overwriting it.
        if previous_scope is not None:
            if previous_scope['factor'] is not self:
                raise RuntimeError('Native world memo owner identity changed')
            return original_block(self, q, jacobian, original_start=original_start)
        scope = dict(factor=self, values={}, hits=0, misses=0, stored_array_bytes=0)
        scopes[id(self)] = scope
        succeeded = False
        try:
            result = original_block(self, q, jacobian, original_start=original_start)
            succeeded = True
            return result
        finally:
            scopes.pop(id(self), None)
            stats = dict(schema='rinkspace.one-native-block-exact-world-reuse.v1',
                factor_class=receipt['factor_class'], succeeded=succeeded,
                hits=scope['hits'], original_producer_calls=scope['misses'],
                stored_array_bytes=scope['stored_array_bytes'],
                retained_world_arrays_after_block=0, original_start=int(original_start),
                source_residual_J_stencil_support_and_ledger_unchanged=True,
                cross_state_J_or_residual_reuse=False)
            scope['values'].clear()
            self._rinkspace_last_tail_world_memo_report = stats
            if report_directory is not None:
                destination = report_directory / (cls.__name__ + '.json')
                destination.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n', encoding='utf8')

    cls._world, cls.block = world, block
    cls._rinkspace_tail_world_memo_receipt = receipt
    return receipt


def install_tail_world_memo_runtime(*, report_directory=None):
    """Install before constructing the next run's original native factors."""
    from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
    from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
    destination = None if report_directory is None else Path(report_directory).resolve()
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    helper = Path(__file__).resolve()
    return dict(schema='rinkspace.native-tail-world-producer-exact-reuse.v1',
        helper_path=str(helper), helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        factors=[_wrap_factor(cls, destination) for cls in (NativeIceCurveFactor, NativeBoardEdgeFactor)],
        registered_source_files_unchanged=True, new_processes_or_source_graph_copies=0,
        actual_full_source_J_byte_equivalence_pending=True,
        actual_complete_cycle_speedup_pending=True, accuracy_accepted=False)
