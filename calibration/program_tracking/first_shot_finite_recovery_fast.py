"""Reuse exact coverage certificates inside one synchronous endpoint recovery.

The existing recovery, initializer and original full guard execute unchanged.
Only an already completed coverage check for the identical state, complete q,
prepared source inventory and world values can be reused. Other threads and
contexts always call the original checker. No guard-cache format is copied:
the caller's original guard still constructs its own complete certificate.
"""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
from types import MethodType

import numpy as np

from . import finite_source_line_coverage as coverage
from . import first_shot_finite_recovery as original_recovery

_LOCK = threading.Lock()


def _reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _array(h, value):
    value = np.asarray(value)
    h.update(json.dumps([value.dtype.str, value.shape], separators=(',', ':')).encode())
    h.update(np.ascontiguousarray(value).tobytes())


def _inventory_sha(inventory):
    h = hashlib.sha256()
    h.update(json.dumps(_plain(dict(
        requested_extent_columns=getattr(inventory, 'requested_extent_columns', None),
        initializer_strategy=getattr(inventory, 'initializer_strategy', None),
        unsupported=getattr(inventory, 'unsupported', ()))), sort_keys=True).encode())
    for source in inventory:
        metadata = dict(index=source.index, native_pts=source.native_pts,
            target_key=source.target_key, source_feature_id=source.source_feature_id,
            query_ids=source.query_ids, end_columns=source.end_columns,
            start_columns=source.start_columns, evidence=source.evidence)
        h.update(json.dumps(_plain(metadata), sort_keys=True, separators=(',', ':')).encode())
        for name in ('query_uv', 'profile_normal', 'measured_uv'):
            _array(h, getattr(source, name))
    return h.hexdigest()


def _world_sha(worlds):
    h = hashlib.sha256()
    for key in sorted(worlds):
        h.update(json.dumps(key, separators=(',', ':')).encode())
        _array(h, worlds[key])
    return h.hexdigest()


def install_finite_endpoint_recovery_fast(owner, *, allowed_columns):
    """Install instead of, or immediately after, the existing finite helper."""
    if hasattr(owner, 'finite_endpoint_recovery_fast_receipt'):
        raise ValueError('Fast finite endpoint recovery is already installed')
    allowed = tuple(allowed_columns)
    if not hasattr(owner, 'finite_endpoint_recovery_receipt'):
        original_recovery.install_finite_endpoint_recovery(owner, allowed_columns=allowed)
    else:
        previous = owner.finite_endpoint_recovery_receipt
        if (previous['allowed_endpoint_columns'] != sorted(allowed)
                or previous['producer'] != _reference(original_recovery.__file__)):
            raise ValueError('Fast recovery requires the same frozen endpoint recovery and columns')
    slow_recover = owner.recover
    previous_identity = owner.checkpoint_identity
    original_check = coverage.check_native_line_coverage
    inventory = owner.bundle.finite_coverage_inventory
    producer = _reference(__file__)
    checker_producer = _reference(coverage.__file__)
    initializer_producer = _reference(original_recovery.__file__)
    receipt = dict(producer=producer, coverage_producer=checker_producer,
        original_endpoint_recovery_producer=initializer_producer,
        allowed_endpoint_columns=sorted(allowed),
        reuse_scope='one synchronous recovery; identical state, full q bytes, source inventory and world values',
        original_initializer_and_guard_unmodified=True,
        failed_check_reused_only_for_initializer_report_request=True,
        final_passed_coverage_reused_by_original_full_guard=True,
        original_guard_builds_its_own_cache=True,
        every_original_query_still_checked_at_both_distinct_states=True,
        other_threads_delegate_to_original_checker=True,
        checker_restored_in_finally=True,
        real_source_cost_and_all_physical_conditions_unchanged=True)
    owner.finite_endpoint_recovery_fast_receipt = deepcopy(receipt)

    def recover(adapter, parent, proposed):
        for path, expected in ((__file__, producer), (coverage.__file__, checker_producer),
                               (original_recovery.__file__, initializer_producer)):
            if _reference(path) != expected:
                raise ValueError('Frozen finite coverage execution producer changed')
        if coverage.check_native_line_coverage is not original_check:
            raise RuntimeError('Another coverage execution override is already active')
        if not _LOCK.acquire(blocking=False):
            raise RuntimeError('Concurrent endpoint certificate scopes are unsupported')
        thread_id = threading.get_ident()
        certificates = {}
        stats = dict(actual_complete_coverage_checks=0,
            reused_failed_initializer_checks=0, reused_passed_guard_checks=0,
            original_native_queries_checked=0, certificates=[])

        def checked(state, q, sections, worlds, *, detailed=False, raise_on_failure=True):
            if (threading.get_ident() != thread_id or state is not adapter.state
                    or sections is not inventory):
                return original_check(state, q, sections, worlds,
                    detailed=detailed, raise_on_failure=raise_on_failure)
            value = np.asarray(q)
            token = (value.dtype.str, value.shape, np.ascontiguousarray(value).tobytes(), bool(detailed))
            source_sha = _inventory_sha(sections)
            world_sha = _world_sha(worlds)
            previous = certificates.get(token)
            if previous is not None:
                if (source_sha != previous['inventory_sha256']
                        or world_sha != previous['world_sha256']):
                    raise RuntimeError('Identical q acquired different source or world values inside recovery')
                report = previous['report']
                if report.get('passed') is True:
                    stats['reused_passed_guard_checks'] += 1
                    return deepcopy(report)
                if report.get('passed') is False and raise_on_failure is False:
                    stats['reused_failed_initializer_checks'] += 1
                    return deepcopy(report)
            stats['actual_complete_coverage_checks'] += 1
            try:
                report = original_check(state, q, sections, worlds,
                    detailed=detailed, raise_on_failure=raise_on_failure)
            except coverage.FiniteLineCoverageError as error:
                report = error.report
                if isinstance(report, Mapping):
                    certificates[token] = dict(report=deepcopy(report),
                        inventory_sha256=source_sha, world_sha256=world_sha)
                    stats['original_native_queries_checked'] += int(report.get('actual_native_queries', 0))
                raise
            else:
                certificates[token] = dict(report=deepcopy(report),
                    inventory_sha256=source_sha, world_sha256=world_sha)
                stats['original_native_queries_checked'] += int(report.get('actual_native_queries', 0))
                return report

        try:
            coverage.check_native_line_coverage = checked
            result = slow_recover(parent, proposed)
        finally:
            coverage.check_native_line_coverage = original_check
            _LOCK.release()
        for token, certificate in certificates.items():
            stats['certificates'].append(dict(
                complete_q_sha256=hashlib.sha256(token[2]).hexdigest(),
                inventory_sha256=certificate['inventory_sha256'],
                world_sha256=certificate['world_sha256'],
                passed=certificate['report'].get('passed'),
                actual_native_queries=certificate['report'].get('actual_native_queries'),
                detailed=token[3]))
        stats.update(producer=producer, checker_restored=True,
            original_guard_cache_created_by_original_guard=True,
            original_initializer_math_and_real_source_cost_unchanged=True)
        adapter.last_finite_endpoint_fast_certificate = deepcopy(stats)
        report = result.get('report')
        report = dict(report) if isinstance(report, Mapping) else dict(original_recovery_report=report)
        report['finite_endpoint_recovery_fast'] = stats
        return dict(result, report=report)

    def identity(adapter):
        result = dict(previous_identity())
        result['finite_endpoint_recovery_fast'] = deepcopy(receipt)
        return result

    owner.recover = MethodType(recover, owner)
    owner.checkpoint_identity = MethodType(identity, owner)
    return deepcopy(receipt)
