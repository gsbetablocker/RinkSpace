"""Opt-in stable-winner Jacobian for the frozen adaptive source-curve target.

The residual always comes from the original adaptive curve. Only complete
factors whose every source row passes interior, central-domain, competition,
curvature and perturbed-state checks use the implicit physical15 Jacobian.
Every other factor calls the original union-topology finite difference.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

import cv2
import numpy as np

from continuous_central_edge_optin import _central_mask, _project15
from projection import from_state
from secondary_nearboard_adaptive_curve import (
    AdaptiveCurveResolutionError, AdaptiveProjectedPolyline,
    _physical_projector, _samples, _interval_distance_by_segment)


_ORIGINAL = None
_COUNTS = Counter()


def _unique_winner_mask(curve, camera, winning_arc, residual,
                        *, nonlocal_world_margin=1., pixel_margin=1.):
    """Conservative screen for a second separated closest-curve branch."""
    projector = _physical_projector(camera, curve.line)
    nodes = curve._refine(curve.line.base_x, projector)
    samples, valid = _samples(nodes, projector)
    full = np.all(valid, axis=1)
    if not np.any(full):
        return np.zeros(len(winning_arc), bool)
    selected = np.flatnonzero(full)
    a, b = samples[selected, 0], samples[selected, 4]
    chord = (a[:, None, :] + np.asarray([.25, .5, .75])[None, :, None]
             * (b-a)[:, None, :])
    sampled_error = np.max(np.linalg.norm(
        samples[selected, 1:4] - chord, axis=2), axis=1)
    lower_arc, upper_arc = nodes[selected], nodes[selected+1]
    unique = np.ones(len(winning_arc), bool)
    for i, ((low, high, y), x) in enumerate(zip(curve.observations, winning_arc)):
        distance = _interval_distance_by_segment(a, b, low, high, y)
        lower_bound = distance - 2.*sampled_error
        separated = ((upper_arc < x-nonlocal_world_margin)
                     | (lower_arc > x+nonlocal_world_margin))
        winner = np.linalg.norm(residual.reshape(-1, 2)[i])
        if np.any(lower_bound[separated] <= winner+pixel_margin):
            unique[i] = False
    return unique


def _local_state_stability(curve, physical, template, winning_arc,
                           dx_dstate, proposed, eligible, *, step_scale=1e-6):
    """Check all 15 local response columns without a global curve re-search."""
    original_arc = curve.line.original_arc
    index = np.clip(np.searchsorted(original_arc, winning_arc, side="right") - 1,
                    0, len(original_arc)-2)
    intervals = np.asarray(curve.observations, np.float64)
    low, high, y = intervals.T
    result = eligible.copy()
    base_j = proposed.reshape(-1, 2, len(physical))
    for column in range(len(physical)):
        width = step_scale*max(1., abs(float(physical[column])))
        pixels = []
        validity = []
        for direction in (1., -1.):
            changed = physical.copy()
            changed[column] += direction*width
            shifted = winning_arc + direction*width*dx_dstate[:, column]
            inside = ((shifted > original_arc[index]+.001)
                      & (shifted < original_arc[index+1]-.001))
            result &= inside
            safe_shifted = np.clip(shifted, original_arc[0], original_arc[-1])
            camera = from_state(changed, template)
            world = curve.line.world(safe_shifted)
            uv, depth = _project15(camera, world)
            central = _central_mask(camera, world, raw_uv=uv, raw_depth=depth)
            result &= central
            finite = np.isfinite(uv).all(axis=1)
            result &= finite
            pixels.append(uv)
            validity.append(central)
        up, down = pixels
        up_residual = up - np.column_stack((np.clip(up[:, 0], low, high), y))
        down_residual = down - np.column_stack((np.clip(down[:, 0], low, high), y))
        finite_difference = (up_residual-down_residual)/(2.*width)
        tolerance = .03 + .02*np.abs(base_j[:, :, column])
        result &= np.all(np.abs(finite_difference-base_j[:, :, column])
                         <= tolerance, axis=1)
    return result


def local_implicit_rows(curve, physical, template, *, vertex_margin=.003,
                        interval_margin_px=.25, curvature_floor=.02,
                        competition_guard=True, state_stability_guard=True,
                        step_scale=1e-6):
    """Return (base residual, provisional J, eligible mask, diagnostics)."""
    camera = from_state(physical, template)
    residual, report = curve.residual(camera)
    xs = np.asarray(report["winning_world_arc"], np.float64)
    original_arc = curve.line.original_arc
    segment = np.clip(np.searchsorted(original_arc, xs, side="right") - 1,
                      0, len(original_arc) - 2)
    distance_to_vertex = np.minimum(xs - original_arc[segment],
                                    original_arc[segment + 1] - xs)
    widths = np.minimum(.001, .2 * distance_to_vertex)
    widths = np.maximum(widths, 1e-8)
    sample_x = np.r_[xs,
                     np.maximum(xs - widths, original_arc[0]),
                     np.minimum(xs + widths, original_arc[-1])]
    world = curve.line.world(sample_x)
    uv, depth, raw_j = _project15(camera, world, with_jacobian=True)
    central = _central_mask(camera, world, raw_uv=uv, raw_depth=depth,
                            raw_j=raw_j)
    count = len(xs)
    pixels, before, after = np.split(uv, 3)
    j0, jbefore, jafter = np.split(raw_j.reshape(-1, 2, 15), 3)
    valid0, valid_before, valid_after = np.split(central, 3)
    tangent = (after - before)/(2. * widths[:, None])
    tangent_j = (jafter - jbefore)/(2. * widths[:, None, None])
    second = (after - 2. * pixels + before)/(widths[:, None]**2)
    intervals = np.asarray(curve.observations, np.float64)
    low, high, y = intervals.T
    point_interval = np.abs(high-low) <= 1e-12
    p_x = np.where(point_interval | (pixels[:, 0] < low)
                   | (pixels[:, 0] > high), 1., 0.)
    observed = np.column_stack((np.clip(pixels[:, 0], low, high), y))
    raw_residual = pixels - observed
    projected_j = j0.copy()
    projected_j[:, 0] *= p_x[:, None]
    numerator = (np.einsum("nij,ni->nj", projected_j, tangent)
                 + np.einsum("ni,nij->nj", raw_residual, tangent_j))
    denominator = (p_x * tangent[:, 0]**2 + tangent[:, 1]**2
                   + np.sum(raw_residual * second, axis=1))
    dx_dstate = -numerator/denominator[:, None]
    answer = j0 + tangent[:, :, None]*dx_dstate[:, None, :]
    answer[:, 0] *= p_x[:, None]
    stationarity = np.sum(raw_residual*tangent, axis=1)
    interval_distance = np.where(point_interval, np.inf,
                                 np.minimum(np.abs(pixels[:, 0]-low),
                                            np.abs(pixels[:, 0]-high)))
    eligible = (valid0 & valid_before & valid_after
                & (distance_to_vertex > vertex_margin)
                & (interval_distance > interval_margin_px)
                & (denominator > curvature_floor*np.sum(tangent*tangent, axis=1))
                & (np.abs(stationarity) < .2)
                & (np.linalg.norm(raw_residual-residual.reshape(-1, 2), axis=1) < 1e-4)
                & np.isfinite(answer).all(axis=(1, 2)))
    if competition_guard:
        unique = _unique_winner_mask(curve, camera, xs, residual)
        eligible &= unique
    else:
        unique = np.ones(count, bool)
    before_stability = eligible.copy()
    if state_stability_guard:
        eligible = _local_state_stability(
            curve, physical, template, xs, dx_dstate, answer, eligible,
            step_scale=step_scale)
    diagnostics = dict(
        count=count, eligible_count=int(eligible.sum()),
        min_vertex_distance=float(distance_to_vertex.min()),
        max_stationarity=float(np.max(np.abs(stationarity))),
        min_relative_curvature=float(np.min(
            denominator/np.maximum(np.sum(tangent*tangent, axis=1), 1e-12))),
        eligible_indices=np.flatnonzero(eligible).tolist(),
        unique_count=int(unique.sum()),
        before_local_state_stability=int(before_stability.sum()),
        original_value_diagnostic=report,
    )
    return residual, answer.reshape(-1, 15), eligible, diagnostics


def install_adaptive_implicit_optin() -> None:
    """Explicitly install the complete-factor fast path before a graph run."""
    global _ORIGINAL
    if _ORIGINAL is not None:
        return
    original = AdaptiveProjectedPolyline.value_and_physical_jacobian

    def guarded(self, state, template25, *, source_keys=(), step_scale=1e-6):
        if np.asarray(state).shape != (15,):
            _COUNTS["fallback_nonphysical15"] += 1
            return original(self, state, template25, source_keys=source_keys,
                            step_scale=step_scale)
        try:
            value, jac, eligible, diagnostics = local_implicit_rows(
                self, state, template25, step_scale=step_scale)
        except AdaptiveCurveResolutionError:
            # A numerical budget failure is not evidence of physical domain
            # failure. Preserve the frozen callback's exact exception.
            raise
        except (FloatingPointError, OverflowError, ValueError, cv2.error):
            _COUNTS["fallback_numeric_guard"] += 1
            return original(self, state, template25, source_keys=source_keys,
                            step_scale=step_scale)
        if not np.all(eligible):
            _COUNTS["fallback_unstable_complete_factor"] += 1
            _COUNTS["fallback_rows"] += int(np.sum(~eligible))
            return original(self, state, template25, source_keys=source_keys,
                            step_scale=step_scale)
        _COUNTS["implicit_complete_factors"] += 1
        _COUNTS["implicit_source_rows"] += len(eligible)
        return value, jac, dict(
            diagnostics["original_value_diagnostic"],
            jacobian_method="stable_winner_implicit_physical15",
            verified_source_rows=len(eligible),
            original_world_curve_unchanged=True,
            original_source_intervals_unchanged=True,
            union_fd_used=False,
        )

    AdaptiveProjectedPolyline.value_and_physical_jacobian = guarded
    _ORIGINAL = original


def uninstall_adaptive_implicit_optin() -> None:
    """Restore the frozen method after an isolated numerical test."""
    global _ORIGINAL
    if _ORIGINAL is not None:
        AdaptiveProjectedPolyline.value_and_physical_jacobian = _ORIGINAL
        _ORIGINAL = None


def adaptive_implicit_optin_metadata() -> dict:
    path = Path(__file__).resolve()
    source = path.parent / "secondary_nearboard_adaptive_curve.py"
    return dict(
        schema="rinkspace.adaptive-curve-implicit-optin.v1",
        installed=_ORIGINAL is not None,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        frozen_curve_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        counts=dict(_COUNTS),
        full_factor_fallback=True,
        residual_target="unchanged_original_adaptive_curve",
        analytic_scope="interior_unique_central_nondegenerate_physical15",
    )
