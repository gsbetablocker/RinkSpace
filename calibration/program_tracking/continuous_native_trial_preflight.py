"""Collect complete native-factor fallbacks at the actual unsupported LM trial."""
from __future__ import annotations

import time
import numpy as np
from continuous_native_edge_optin import NativeEdgeUnsupported
from reference_board_pair import BoardIntersectionError


def trial_unsupported_factors(manager, graph, values):
    """Check all live native residuals without accepting or changing the trial.

    Physical-domain exceptions propagate before any graph mutation.  The solver
    therefore handles the same trial as a domain rejection, preserving its last
    accepted state.  Unsupported factors are replaced only for a finite trial.
    """
    manager._main_thread()
    started=time.perf_counter()
    expected={int(graph._key(key)) for key in graph._pending_ids}
    actual={int(key) for key in values.keys()}
    if actual != expected:
        raise RuntimeError('Captured native trial does not contain the complete graph key set')
    live=tuple(record for record in manager._records.values() if not record.fallback)
    failures=[]
    supported=0
    for record in live:
        if (record.factor_id < 0 or record.factor_id >= len(graph._pending)
                or graph._pending[record.factor_id] is not record.factor):
            raise RuntimeError('Native trial probe factor slot changed')
        try:
            cost=float(record.factor.error(values))
        except NativeEdgeUnsupported as error:
            if (error.factor_id!=record.factor_id or error.source_pts!=record.source_pts
                    or error.evaluation_stage!='residual' or error.status!=2):
                raise RuntimeError('Native residual unsupported identity changed') from error
            # The caller owns this exact trial once, not once per failed factor.
            error.trial_values=None
            error.__traceback__=None
            failures.append(error)
            continue
        except BoardIntersectionError as error:
            # The optimizer and original factor graph are still intact. Expose
            # the probe result on the typed domain error for the retry trace.
            error.trial_probe = dict(
                live_native_factors=len(live),
                supported_native_factors=supported,
                unsupported_status2_before_domain=len(failures),
                unsupported_factor_ids_before_domain=[int(e.factor_id) for e in failures],
                domain_factor_id=getattr(error, 'factor_id', None),
                domain_source_pts=getattr(error, 'source_pts', None),
                elapsed_seconds=time.perf_counter()-started,
                graph_rebuilds=0,
            )
            raise
        if not np.isfinite(cost):
            raise ValueError('Native trial error is nonfinite')
        supported += 1
    return failures,dict(live_native_factors=len(live),
        supported_native_factors=supported,
        unsupported_factor_ids=[int(e.factor_id) for e in failures],
        unsupported_source_pts=[int(e.source_pts) for e in failures],
        unsupported_status2=len(failures),elapsed_seconds=time.perf_counter()-started,
        state_source='complete Values captured by failing native residual',
        physical_domain_failures='propagate before graph mutation',retained_trial_copies=0)
