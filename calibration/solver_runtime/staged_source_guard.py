"""Reject from unchanged necessary predicates before an expensive guard.

A probe can reject but can never admit a trial. Every probe-passing trial
still executes the caller's complete, original guard, including its truth
suffix. The caller owns source identity and readiness: initialization must
run in the original order before recovery limits may be read by probes.

Reordering a conjunction preserves admission, but can change the first
exception when several predicates fail. That scheduling change is explicit
and opt-in. Unknown probe failures fall back to the original order.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import time
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class GuardProbe:
    name: str
    check: Callable
    is_necessary_rejection: Callable[[Exception], bool]


@contextmanager
def preserve_programme_camera_bindings(blocks):
    """Restore the small state modified by the audited camera-only query.

    StructuredProgrammeBlock.camera_local() replaces bundle.calibration and
    populates a three-key world-validation cache. It does not synchronize
    programme motion planes/scenes. Require that camera-only interface;
    copying an entire source graph or source arrays is never necessary.
    Exact-input factor query memoization remains permitted: its public
    producers return copies and do not change source/profile/derivative data.
    """
    saved = []
    missing = object()
    for block in blocks:
        if not callable(getattr(block, "camera_local", None)):
            raise TypeError("Early source probes require the camera-only interface")
        cache = vars(block).get("_native_camera_valid_worlds", missing)
        snapshot = None if cache is missing else cache.copy()
        saved.append((block, block.bundle.calibration, cache, snapshot))
    try:
        yield
    finally:
        for block, calibration, cache, snapshot in reversed(saved):
            block.bundle.calibration = calibration
            if cache is missing:
                vars(block).pop("_native_camera_valid_worlds", None)
            else:
                cache.clear()
                cache.update(snapshot)
                block._native_camera_valid_worlds = cache


class StagedSourceGuard:
    """Run original necessary truth predicates first once limits are ready.

    ``ready`` must be read-only and false until every required original limit
    is established. ``probe_context`` restores small physical bindings on
    pass, rejection, and exception. Rejection-priority opt-in changes only
    which original failure is reported for a multiply-invalid trial; the
    source solver must already handle all such failures in the same branch.
    """
    def __init__(self, original_guard, probes, *, ready,
                 probe_context=nullcontext, allow_rejection_priority_change=False):
        self.original_guard = original_guard
        self.probes = tuple(probes)
        self.ready = ready
        self.probe_context = probe_context
        self.allow_rejection_priority_change = bool(allow_rejection_priority_change)
        self.last_report = None
        self.calls = 0
        self.early_rejections = 0
        self.original_calls = 0

    def __call__(self, q):
        started = time.perf_counter()
        self.calls += 1
        report = dict(probes=[], complete_original_guard_executed=False,
                      early_rejected=False, limits_initialized_by_probe=False)
        self.last_report = report
        try:
            if self.ready():
                value = np.asarray(q, float).copy()
                value.flags.writeable = False
                failure = None
                try:
                    with self.probe_context():
                        for probe in self.probes:
                            begin = time.perf_counter()
                            record = dict(name=probe.name)
                            report["probes"].append(record)
                            try:
                                answer = probe.check(value)
                                if answer is not None and not isinstance(answer, (bool, np.bool_)):
                                    raise TypeError("Necessary guard probe must return None/bool or raise")
                                if answer is not None and not bool(answer):
                                    raise ValueError("Unclassified false necessary guard probe")
                                record["passed"] = True
                            except Exception as exc:
                                record.update(passed=False, error_type=type(exc).__name__, error=str(exc))
                                if isinstance(exc, (ValueError, RuntimeError)) and probe.is_necessary_rejection(exc):
                                    failure = exc
                                    record["original_necessary_rejection"] = True
                                else:
                                    record["fallback_to_original_order"] = True
                                break
                            finally:
                                record["seconds"] = time.perf_counter() - begin
                except Exception as exc:
                    # Binding-context/domain failures are not a truth witness.
                    report["probe_context_fallback"] = dict(error_type=type(exc).__name__, error=str(exc))
                    failure = None
                if failure is not None and self.allow_rejection_priority_change:
                    report["early_rejected"] = True
                    report["rejection_priority"] = "original_necessary_predicate_before_complete_guard"
                    self.early_rejections += 1
                    raise failure
            else:
                report["original_order_reason"] = "original_limits_not_ready"
            self.original_calls += 1
            report["complete_original_guard_executed"] = True
            return self.original_guard(q)
        finally:
            report["seconds"] = time.perf_counter() - started


def restoration_progress_precheck(original_decision, previous, actual, *,
                                 common_fraction, step_fraction):
    """Query the original cost-independent necessary restoration conditions.

    The original decision validates finite costs but uses them only in its
    diagnostic report. Zero placeholders are explicitly unmeasured and every
    cost field is removed here. ``reject=False`` means continue to all original
    guards/cost/acceptance; this function never authorizes acceptance.
    """
    if not np.any(np.asarray(previous, float) > 1e-8):
        # The original driver resumes ordinary cost acceptance in this case.
        # A restoration-only predicate must not reject that source-cost trial.
        return dict(reject=False, reason="ordinary_source_objective_pending",
                    report=dict(necessary_only=True, complete_source_cost_pending=True,
                                original_complete_acceptance_still_required=True))
    result = original_decision(previous, actual, common_fraction=common_fraction,
        step_fraction=step_fraction, physical_guard_passed=True,
        previous_source_cost=0., trial_source_cost=0.)
    accepted = result.get("accepted")
    if type(accepted) is not bool:
        raise TypeError("Original restoration decision must return an explicit bool")
    fields = ("protected_zero_groups_inside_original_support",
              "every_previously_violated_group_nonworsening",
              "maximum_actual_original_violation_fraction",
              "required_maximum_original_violation_fraction",
              "restoration_comparison_tolerance_px",
              "comparison_tolerance_basis",
              "maximum_positive_group_growth_px",
              "maximum_positive_group_progress_miss_px",
              "strictly_nonworsening_without_numerical_tolerance",
              "resolvably_improved_positive_groups",
              "measured_violations_not_clipped_or_replaced")
    report = {name: result[name] for name in fields}
    report.update(necessary_only=True, complete_physical_guard_pending=True,
                  complete_source_cost_pending=True,
                  finite_cost_placeholders_used_only_for_original_predicate=True,
                  original_complete_acceptance_still_required=True)
    return dict(reject=not accepted, reason=result["reason"], report=report)


def observe_guard(guard, q):
    """Compact same-state replay outcome; never prepare a graph or solve."""
    value = np.asarray(q, float)
    started = time.perf_counter()
    result = dict(q_sha256=hashlib.sha256(value.tobytes()).hexdigest())
    try:
        answer = guard(value)
        result.update(passed=True, return_value=None if answer is None else bool(answer))
    except (ValueError, RuntimeError) as exc:
        result.update(passed=False, error_type=type(exc).__name__, error=str(exc))
    result["seconds"] = time.perf_counter() - started
    if isinstance(guard, StagedSourceGuard):
        result["staged_guard"] = guard.last_report
    return result


def compare_recorded_guard_outcome(recorded, candidate, *, require_same_exception=True):
    """Reuse a saved original result rather than repeating its costly guard."""
    if recorded["q_sha256"] != candidate["q_sha256"]:
        raise ValueError("Same original complete q required for guard replay")
    passed_equal = recorded["passed"] == candidate["passed"]
    exception_equal = (recorded.get("error_type"), recorded.get("error")) == (
        candidate.get("error_type"), candidate.get("error"))
    return_equal = recorded.get("return_value") == candidate.get("return_value")
    return dict(passed=bool(passed_equal and return_equal and
                           (exception_equal or not require_same_exception)),
                admission_equal=passed_equal, return_equal=return_equal,
                original_exception_equal=exception_equal,
                original_baseline_recomputed=False)
