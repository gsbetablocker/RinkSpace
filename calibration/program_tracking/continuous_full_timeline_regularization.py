"""Opt-in completion of programme-frame R/logf acceleration regularity.

The original official-cadence acceleration factors remain in the graph.  This
module appends the same four-column factor for each otherwise omitted canonical
three-frame window, within one shot only.  Positive exposure-clock intervals
use that clock, including explicitly estimated times.  A zero virtual interval
between *distinct*, unaliased frame nodes uses their native PTS separation for
this prior alone; this is a display-time motion assumption, not an assertion
that the exposures happened at different official times.  Camera evaluation,
alias membership, spline times, source PTS and observations are untouched.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


TICKS_PER_SECOND = 15360.0
_ZERO_TOLERANCE_TICKS = 1e-7
_COLUMNS = (3, 4, 5, 6)
_SPECTRAL_SIGMA = (.5, .5, .5, .5)


def _edge_duration(a: int, b: int, clock: dict[int, float]) -> tuple[float, str]:
    """Return seconds and evidence role for one canonical within-shot edge."""
    virtual_delta = float(clock[b]) - float(clock[a])
    if virtual_delta > _ZERO_TOLERANCE_TICKS:
        return virtual_delta / TICKS_PER_SECOND, "exposure_clock_positive"
    if abs(virtual_delta) <= _ZERO_TOLERANCE_TICKS:
        native_delta = b - a
        if native_delta <= 0:
            raise ValueError(f"Non-increasing native PTS on zero-time edge {a}->{b}")
        return native_delta / TICKS_PER_SECOND, "native_display_time_fallback"
    raise ValueError(f"Exposure clock reverses on edge {a}->{b}")


def add_complete_motion_priors(
    graph: Any,
    model: Any,
    clock: dict[int, float],
    clock_meta: dict[str, Any],
    valid_pairs: set[tuple[int, int]],
    *,
    affine_priors: Any = None,
) -> dict[str, Any]:
    """Append missing R/logf acceleration factors and return JSON-safe provenance.

    Call *after* the original programme graph is built.  Existing factors,
    first-frame targets and every non-frame node remain unchanged.  All inputs
    are checked before the graph is mutated, so an invalid clock cannot leave a
    partially appended batch.  ``affine_priors`` may be NativeAffinePriors and
    receives the same ``add_acceleration_factor`` call as the original path.
    """
    if clock_meta.get("schema") != "rinkspace.continuous-exposure-clock.v1":
        raise ValueError("Expected audited continuous exposure clock metadata")
    records = clock_meta.get("timestamps_by_canonical_source_pts")
    if not isinstance(records, dict):
        raise ValueError("Exposure clock lacks canonical timestamp provenance")

    clock = {int(k): float(v) for k, v in clock.items()}
    valid_pairs = {(int(a), int(b)) for a, b in valid_pairs}
    if any(not np.isfinite(value) for value in clock.values()):
        raise ValueError("Exposure clock contains nonfinite times")

    planned: list[dict[str, Any]] = []
    by_shot: dict[str, dict[str, Any]] = {}
    represented_keys: set[int] = set()
    all_keys: set[int] = set()
    strict_count = 0
    for shot_id in sorted(model.shot_splines):
        stamps = sorted({int(binding.frame_key) for binding in model.frames.values()
                         if binding.shot_id == shot_id})
        if not stamps:
            raise ValueError(f"Shot {shot_id} has no canonical frame keys")
        if len(stamps) != len(set(stamps)) or any(b <= a for a, b in zip(stamps, stamps[1:])):
            raise ValueError(f"Shot {shot_id} has invalid canonical order")
        all_keys.update(stamps)
        for key in stamps:
            if key not in clock or str(key) not in records:
                raise ValueError(f"Clock lacks canonical key {key}")
            record = records[str(key)]
            if (int(record["shot_index"]) != int(shot_id)
                    or abs(float(record["virtual_pts"]) - clock[key]) > 1e-6):
                raise ValueError(f"Clock metadata disagrees at canonical key {key}")
        counts: Counter[str] = Counter()
        shot_planned = 0
        for i in range(1, len(stamps) - 1):
            keys = tuple(stamps[i - 1:i + 2])
            a, b, c = keys
            strict = (a, b) in valid_pairs and (b, c) in valid_pairs
            if strict:
                strict_count += 1
                represented_keys.update(keys)
                # The old factor is already present.  Leave it bitwise alone.
                continue
            dt0, role0 = _edge_duration(a, b, clock)
            dt1, role1 = _edge_duration(b, c, clock)
            if not np.isfinite([dt0, dt1]).all() or min(dt0, dt1) <= 0:
                raise ValueError(f"Nonpositive motion-prior time for {keys}")
            times = (0.0, dt0, dt0 + dt1)
            if role0 == "native_display_time_fallback" or role1 == "native_display_time_fallback":
                counts["zero_virtual_time_fallback"] += 1
                category = "zero_virtual_time_fallback"
            else:
                counts["positive_virtual_time"] += 1
                category = "positive_virtual_time"
            statuses = [str(records[str(key)]["status"]) for key in keys]
            if any(status.startswith("estimated_") for status in statuses):
                counts["contains_estimated_clock"] += 1
            planned.append(dict(
                shot_id=int(shot_id), source_keys=list(keys),
                relative_seconds=list(times), edge_time_roles=[role0, role1],
                canonical_time_statuses=statuses, category=category))
            represented_keys.update(keys)
            shot_planned += 1
        by_shot[str(shot_id)] = dict(
            canonical_frame_count=len(stamps), added_factor_count=shot_planned,
            added_reason_counts=dict(counts),
            first_source_pts=stamps[0], last_source_pts=stamps[-1])

    # Bind the exact legacy implementation only after all validation succeeds.
    if affine_priors is None:
        from fit_continuous_camera import acceleration_factor
        add = acceleration_factor
        backend = "python.fit_continuous_camera.acceleration_factor"
    else:
        add = affine_priors.add_acceleration_factor
        backend = "supplied_affine_priors.add_acceleration_factor"
    for row in planned:
        add(graph, tuple(row["source_keys"]), np.asarray(row["relative_seconds"]),
            list(_COLUMNS), _SPECTRAL_SIGMA)

    if len(all_keys) >= 3 and represented_keys != all_keys:
        # This condition should only occur for an atypically short shot.  The
        # report stays explicit; it does not invent another prior family.
        uncovered = sorted(all_keys - represented_keys)
    else:
        uncovered = []
    return dict(
        schema="rinkspace.complete-programme-motion-priors.v1",
        scope="programme frame R[3:6] and mean logf[6] only; no reference or C prior",
        rule="append omitted consecutive canonical triples within each shot",
        positive_dt_role="audited exposure-clock coordinate, including labelled estimates",
        zero_dt_role="native PTS display-time interval for motion prior only; official exposure identity unchanged",
        virtual_clock_and_aliases_modified=False,
        old_strict_factors_modified=False,
        columns=list(_COLUMNS),
        spectral_sigma_state_per_second_to_three_halves=list(_SPECTRAL_SIGMA),
        factor_backend=backend,
        strict_existing_factor_count=strict_count,
        added_factor_count=len(planned),
        added_positive_virtual_time_count=sum(row["category"] == "positive_virtual_time" for row in planned),
        added_zero_virtual_time_fallback_count=sum(row["category"] == "zero_virtual_time_fallback" for row in planned),
        all_canonical_frame_count=len(all_keys),
        uncovered_canonical_source_pts=uncovered,
        shots=by_shot,
        added_factors=planned,
    )
