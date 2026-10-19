"""Batch-check every live round6 source-edge Jacobian at one accepted state.

The returned status-2 exceptions identify complete factors for the existing
owner-thread replacement path. This module never edits the graph or manager.
Physical-domain status 4 remains a typed BoardIntersectionError.
"""
from __future__ import annotations

import time

import numpy as np

from continuous_native_edge_optin import NativeEdgeUnsupported


def verify_native_linearization(manager, graph, states):
    """Return (all unsupported factors, diagnostics) without retaining J arrays.

    ``states`` must be the accepted full Vector11 state map. The caller replaces
    every returned complete factor and rebuilds the GTSAM graph just once.
    """
    manager._main_thread()
    started = time.perf_counter()
    live = tuple(record for record in manager._records.values()
                 if not record.fallback)
    required = {int(key) for record in live for key in record.keys}
    values = manager.gtsam.Values()
    for key in sorted(required):
        vector = np.asarray(states[key], np.float64).reshape(-1)
        if vector.shape != (11,) or not np.isfinite(vector).all():
            raise ValueError(f"Accepted state {key} is not finite Vector11")
        values.insert(graph._key(key), vector)

    unsupported = []
    supported = 0
    jacobian_entries_evaluated = 0
    for record in live:
        factor_id = int(record.factor_id)
        if (factor_id < 0 or factor_id >= len(graph._pending)
                or graph._pending[factor_id] is not record.factor):
            raise RuntimeError("Native linearization factor slot changed")
        try:
            # The graph's own factor uses its production GTSAM object and
            # frozen source payload. A temporary Gaussian exists only here.
            gaussian = record.factor.linearize(values)
        except NativeEdgeUnsupported as error:
            if (error.status != 2 or error.factor_id != factor_id
                    or error.source_pts != record.source_pts
                    or error.evaluation_stage != "jacobian"):
                raise RuntimeError(
                    "Native Jacobian unsupported signal changed identity or stage"
                ) from error
            unsupported.append(error)
            continue
        # BoardIntersectionError (status 4) and all unknown failures propagate.
        if gaussian is None:
            raise RuntimeError("Native factor did not return a Gaussian linearization")
        jacobian_entries_evaluated += int(record.options["residual_dim"]) * 11 * len(record.keys)
        supported += 1
        del gaussian

    diagnostics = dict(
        stage="accepted_state_native_jacobian_preflight",
        live_native_factors=len(live), supported_native_factors=supported,
        unsupported_status2=len(unsupported),
        unsupported_factor_ids=[int(error.factor_id) for error in unsupported],
        unsupported_source_pts=[int(error.source_pts) for error in unsupported],
        jacobian_entries_evaluated=jacobian_entries_evaluated,
        graph_rebuilds=0, graph_factors_changed=0,
        retained_jacobians=0,
        elapsed_seconds=time.perf_counter() - started,
    )
    return unsupported, diagnostics
