"""Recover only witnessed finite material right-end lower bounds.

The authenticated query domain and the endpoint mathematics remain owned by
finite_source_line_coverage. This adapter composes its existing initializer
with the caller's recovery, full physical guard and real source-cost gate.
Install any candidate/report recorder around owner.guard before this adapter.
"""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
from pathlib import Path
from types import MethodType

import numpy as np

from .finite_source_line_coverage import (
    FiniteLineCoverageError, initialize_native_line_coverage,
)


def _reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _right_end_witnesses(report, inventory, allowed):
    """Require unambiguous authenticated end-only failures; solve no geometry."""
    if (not isinstance(report, Mapping) or report.get('passed') is not False
            or report.get('unsupported') or not report.get('failures')):
        return None
    sources = {(s.index, s.target_key): s for s in inventory}
    witnesses = []
    for failed in report['failures']:
        if not isinstance(failed, Mapping):
            return None
        source = sources.get((failed.get('index'), failed.get('target_key')))
        if (source is None or failed.get('native_pts') != source.native_pts
                or failed.get('source_feature_id') != source.source_feature_id
                or len(source.end_columns) != 1
                or source.end_columns[0] not in allowed):
            return None
        queries = failed.get('queries', ())
        if (not isinstance(queries, (list, tuple))
                or any(not isinstance(query, Mapping) for query in queries)
                or tuple(query.get('query_id') for query in queries) != source.query_ids):
            return None
        missing = failed.get('missing_queries', ())
        unsupported = [query for query in queries if query.get('supported') is False]
        if (not unsupported or set(missing) != {query['query_id'] for query in unsupported}
                or len(missing) != len(unsupported)):
            return None
        length = failed.get('world_line_length_scene')
        if not isinstance(length, (int, float)) or not np.isfinite(length) or length <= 0.:
            return None
        for query in unsupported:
            candidates = query.get('candidates', ())
            if (not isinstance(candidates, (list, tuple))
                    or any(not isinstance(item, Mapping) for item in candidates)):
                return None
            forward = [item for item in candidates if item.get('forward_central_valid') is True]
            if not forward:
                return None
            # The existing checker has already certified the sensor-row root,
            # source section, central branch and sensor bounds in this flag.
            # Mixed start/end or unresolved branches remain the original failure.
            for item in forward:
                station = item.get('station_scene')
                if (not isinstance(station, (int, float)) or not np.isfinite(station)
                        or station <= length or item.get('inside_finite_line') is not False):
                    return None
            witnesses.append(dict(index=source.index, native_pts=source.native_pts,
                source_feature_id=source.source_feature_id, target_key=source.target_key,
                query_id=query['query_id'], endpoint_column=int(source.end_columns[0]),
                current_world_line_length_scene=float(length),
                forward_station_scene=[float(item['station_scene']) for item in forward],
                actual_endpoint_measured=False, source_supported_lower_bound_only=True))
    return witnesses or None


def install_finite_endpoint_recovery(owner, *, allowed_columns):
    """Compose after mount recovery and guard recording, before solver_kwargs.

    Column numbers are explicit dataset-owned inputs. They must already have
    real authenticated query support in the prepared coverage inventory.
    """
    if hasattr(owner, 'finite_endpoint_recovery_receipt'):
        raise ValueError('Finite endpoint recovery is already installed')
    raw = tuple(allowed_columns)
    if (not raw or any(type(column) is not int for column in raw)
            or len(set(raw)) != len(raw)):
        raise ValueError('Distinct explicit integer endpoint columns required')
    allowed = frozenset(raw)
    count = int(owner.state.count)
    if min(allowed) < 0 or max(allowed) >= count:
        raise ValueError('Endpoint column is outside the complete state')
    inventory = owner.bundle.finite_coverage_inventory
    if inventory is None or getattr(inventory, 'unsupported', ()):
        raise ValueError('A complete authenticated finite-query inventory is required')
    active = {int(column) for source in inventory if source.query_ids
              for column in source.end_columns}
    if not allowed.issubset(active):
        raise ValueError('Endpoint recovery cannot invent support for an unobserved column')
    original_recover = owner.recover
    original_identity = owner.checkpoint_identity
    receipt = dict(producer=_reference(__file__), allowed_endpoint_columns=sorted(allowed),
        active_source_endpoint_columns=sorted(active),
        initializer='existing initialize_native_line_coverage',
        eligibility='actual guard report; only resolved forward roots beyond movable right ends',
        all_other_complete_state_values_byte_exact=True, only_monotonic_endpoint_extension=True,
        endpoint_values_are_source_supported_lower_bounds_not_measurements=True,
        full_physical_guard_and_real_source_cost_acceptance_retained=True,
        guard_recorder_must_be_installed_before_this_recovery=True,
        no_source_prior_sigma_or_noise_rows_added=True)
    owner.finite_endpoint_recovery_receipt = deepcopy(receipt)

    def recover(adapter, parent, proposed):
        if _reference(__file__) != receipt['producer']:
            raise ValueError('Frozen finite endpoint recovery producer changed')
        result = original_recover(parent, proposed)
        if not isinstance(result, dict) or 'q' not in result:
            raise ValueError('Original recovery must return q and its original report')
        before = np.asarray(result['q'], dtype=np.float64).copy()
        if before.shape != (count,) or not np.isfinite(before).all():
            raise ValueError('Original recovery returned an invalid complete state')
        try:
            adapter.guard(before)
        except FiniteLineCoverageError as error:
            witnesses = _right_end_witnesses(error.report, inventory, allowed)
            if witnesses is None:
                raise
            after, initialization = initialize_native_line_coverage(
                adapter.state, before.copy(), inventory,
                lambda value: adapter.world(value, False)[0])
        else:
            # The caller's ordinary guard now has its exact-state cache entry.
            return result
        after = np.asarray(after, dtype=np.float64)
        if after.shape != before.shape or not np.isfinite(after).all():
            raise RuntimeError('Finite endpoint initializer changed the complete-state domain')
        keep = np.ones(count, dtype=bool)
        keep[list(allowed)] = False
        changed = np.flatnonzero(after != before)
        if (before[keep].tobytes() != after[keep].tobytes()
                or not len(changed) or not set(map(int, changed)).issubset(allowed)
                or np.any(after[list(allowed)] < before[list(allowed)])):
            raise RuntimeError('Finite endpoint initializer changed an unrelated value or shortened material')
        # This also retains all ice containment, camera, mount and finite-query
        # checks; its exact-state cache serves the solver's following guard.
        adapter.guard(after)
        previous = result.get('report')
        report = dict(previous) if isinstance(previous, Mapping) else dict(original_recovery_report=previous)
        report['finite_endpoint_recovery'] = dict(
            producer=receipt['producer'], candidate_before_endpoint_sha256=_sha(before),
            candidate_after_endpoint_sha256=_sha(after),
            changed_endpoint_columns=changed.tolist(), witnesses=witnesses,
            initialization=initialization,
            all_non_endpoint_values_byte_exact=True, only_monotonic_extension=True,
            actual_endpoint_measured=False, source_supported_lower_bound_only=True,
            full_guard_after_recovery_passed=True, real_source_cost_still_required=True)
        return dict(result, q=after, report=report)

    def identity(adapter):
        result = dict(original_identity())
        result['finite_endpoint_recovery'] = deepcopy(receipt)
        return result

    owner.recover = MethodType(recover, owner)
    owner.checkpoint_identity = MethodType(identity, owner)
    return deepcopy(receipt)
