"""Opt-in Brown15 source-edge Jacobian with central-branch-aware FD fallback.

Call ``install_edge_optin`` before the v13 callback capture.  The original
source UV, quality scales, Huber rule and candidate window remain in that
capture.  This module only supplies full physical15 derivatives and a guard
that retains the captured 1e-6 complete-residual Brown FD whenever a Brown
step changes a discrete candidate, nearest segment, clamp or Huber branch.
It does not run a graph or change the default tracking path.
"""
from __future__ import annotations

import inspect
import hashlib
from collections import Counter
from pathlib import Path

from runtime.bootstrap import add_geometry_dependencies

add_geometry_dependencies()

import cv2
import numpy as np

from acceleration import nearest_segments
from continuous_brown_adapter import adapt_factor
import projection as _projection
from projection import from_state, project_central_valid
from reference_board_pair import BoardIntersectionError
from projection_reuse import reused_projection


_GUARDS: dict[object, CentralBrownEdgeGuard] = {}
_INTERVAL_REPLACEMENTS: dict[object, dict] = {}
_ACTIVE_INTERVAL_REPLACEMENTS: dict[tuple[int, str], object] = {}
_ORIGINALS = None
_ADAPTER_COUNTS = Counter()


def _central_mask(camera, world, *, raw_uv=None, raw_depth=None, raw_j=None):
    """Match project_central_valid's raw/safe predicate exactly."""
    c = np.asarray(camera, float)
    points = np.ascontiguousarray(np.asarray(world, float).reshape(-1, 3))
    if raw_uv is None or raw_depth is None:
        return project_central_valid(c, points)[-1]
    raw_uv = np.asarray(raw_uv, float).reshape(-1, 2)
    raw_depth = np.asarray(raw_depth, float).reshape(-1)
    reused = reused_projection(c, points)
    if (reused is not None
            and np.array_equal(raw_uv, reused[0], equal_nan=True)
            and np.array_equal(raw_depth, reused[1], equal_nan=True)):
        valid = reused[2]
        if raw_j is not None:
            valid &= np.isfinite(np.asarray(raw_j).reshape(-1, 2, 15)).all(axis=(1, 2))
        return valid
    valid = np.isfinite(raw_uv).all(axis=1) & np.isfinite(raw_depth) & (raw_depth > .05)
    if np.any(np.abs(c[11:23]) > 1e-14) and valid.any():
        indices = np.flatnonzero(valid)
        try:
            safe_uv, _, central = _projection._central_projection_backend(c, points[indices])
            central &= np.isfinite(safe_uv).all(axis=1)
            central &= np.linalg.norm(safe_uv - raw_uv[indices], axis=1) <= 1e-5
        except (cv2.error, FloatingPointError, OverflowError):
            central = np.zeros(len(indices), bool)
        valid[indices] = central
    if raw_j is not None:
        valid &= np.isfinite(np.asarray(raw_j).reshape(-1, 2, 15)).all(axis=(1, 2))
    return valid


def _project15(camera, xyz, *, with_jacobian=False):
    """The original OpenCV projection plus its four Brown columns."""
    c = np.asarray(camera, float)
    xyz = np.ascontiguousarray(np.asarray(xyz, float).reshape(-1, 3))
    if not with_jacobian:
        reused = reused_projection(c, xyz)
        if reused is not None:
            return reused[:2]
    if not len(xyz):
        if with_jacobian:
            return np.empty((0, 2)), np.empty(0), np.empty((0, 15))
        return np.empty((0, 2)), np.empty(0)
    rotation, drot = cv2.Rodrigues(c[3:6])
    k = np.array([[np.exp(c[6]), 0., c[8]], [0., np.exp(c[7]), c[9]],
                  [0., 0., 1.]])
    translation = -rotation @ c[:3]
    projected, jac = cv2.projectPoints(xyz, c[3:6], translation, k, c[11:25])
    uv = projected.reshape(-1, 2)
    yf = (uv[:, 1] - c[9]) / k[1, 1]
    uv[:, 0] += c[10] * yf
    depth = (xyz - c[:3]) @ rotation[2]
    if not with_jacobian:
        return uv, depth
    result = np.zeros((2 * len(xyz), 15), float)
    result[:, :3] = jac[:, 3:6] @ (-rotation * 100.)
    dr = drot.reshape(3, 3, 3)
    dt = np.column_stack([-dr[i] @ c[:3] for i in range(3)])
    result[:, 3:6] = jac[:, :3] + jac[:, 3:6] @ dt
    result[:, 6] = jac[:, 6] * k[0, 0]
    result[:, 7] = jac[:, 7] * k[1, 1]
    result[:, 8] = jac[:, 8] * 1000.
    result[:, 9] = jac[:, 9] * 1000.
    result[:, 11:15] = jac[:, 10:14]
    yj = result[1::2].copy()
    yj[:, 9] -= 1000.
    yj /= k[1, 1]
    yj[:, 7] -= yf
    result[0::2] += c[10] * yj
    result[0::2, 10] = yf * 1000.
    return uv, depth, result


def edge_curve_error15(camera, world, observed, segment_index,
                       with_jacobian=False, *, finite_vector=False):
    """Current finite-curve equations, with safe segments and physical15 J."""
    world = np.asarray(world, np.float64)
    observed = np.asarray(observed, np.float64).reshape(-1, 2)
    segments = np.asarray(segment_index, np.int64).reshape(-1)
    if not len(segments):
        raise BoardIntersectionError('source_edge_no_central_segment',
                                     np.arange(len(observed)))
    point_index = np.unique(np.r_[segments, segments + 1])
    first = np.searchsorted(point_index, segments)
    second = np.searchsorted(point_index, segments + 1)
    try:
        if with_jacobian:
            projected, depth, raw_j = _project15(camera, world[point_index],
                                                 with_jacobian=True)
            jac = raw_j.reshape(len(point_index), 2, 15)
        else:
            projected, depth = _project15(camera, world[point_index])
    except cv2.error as error:
        raise BoardIntersectionError('source_edge_projection_failed',
                                     np.arange(len(observed))) from error
    central = _central_mask(camera, world[point_index], raw_uv=projected,
                            raw_depth=depth, raw_j=raw_j if with_jacobian else None)
    midpoint = .5 * (world[segments] + world[segments + 1])
    try:
        midpoint_uv, midpoint_depth = _project15(camera, midpoint)
    except cv2.error as error:
        raise BoardIntersectionError('source_edge_projection_failed',
                                     np.arange(len(observed))) from error
    midpoint_central = _central_mask(camera, midpoint,
                                     raw_uv=midpoint_uv, raw_depth=midpoint_depth)
    start, finish = projected[first], projected[second]
    direction = finish - start
    length2 = np.sum(direction * direction, axis=1)
    valid = (central[first] & central[second] & midpoint_central
             & np.isfinite(start).all(axis=1) & np.isfinite(finish).all(axis=1)
             & (depth[first] > 0) & (depth[second] > 0) & (length2 > 1e-4))
    if not valid.any():
        raise BoardIntersectionError('source_edge_no_central_segment',
                                     np.arange(len(observed)))
    index, raw_t, selected_t, nearest = nearest_segments(
        observed, start, direction, length2, valid, compiled=False)
    a, d = start[index], direction[index]
    j0 = jac[first[index]] if with_jacobian else None
    jd = jac[second[index]] - j0 if with_jacobian else None
    if finite_vector:
        residual = (nearest - observed).ravel()
        if not with_jacobian:
            return residual
        offset = observed - a
        length2_selected = length2[index]
        numerator_j = (-np.einsum('nij,ni->nj', j0, d)
                       + np.einsum('ni,nij->nj', offset, jd))
        length2_j = 2. * np.einsum('ni,nij->nj', d, jd)
        t_j = (numerator_j - raw_t[:, None] * length2_j) / length2_selected[:, None]
        t_j[(raw_t <= 0.) | (raw_t >= 1.)] = 0.
        derivative = j0 + selected_t[:, None, None] * jd + d[:, :, None] * t_j[:, None, :]
        return residual, derivative.reshape(-1, 15)
    norm = np.sqrt(length2[index])
    u = a - observed
    cross = d[:, 0] * u[:, 1] - d[:, 1] * u[:, 0]
    residual = cross / norm
    if not with_jacobian:
        return residual
    dc = (jd[:, 0] * u[:, 1, None] - jd[:, 1] * u[:, 0, None]
          + d[:, 0, None] * j0[:, 1] - d[:, 1, None] * j0[:, 0])
    dn = np.sum(d[:, :, None] * jd, axis=1) / norm[:, None]
    return residual, dc / norm[:, None] - cross[:, None] * dn / norm[:, None]**2


class CentralBrownEdgeGuard:
    """Detect all branch changes across the four exact Brown FD steps."""

    def __init__(self, world, observed, template, quality_scales, sigma,
                 dynamic_candidates, frozen_segments, *, source_pts=None,
                 edge_name=None):
        self.world = np.ascontiguousarray(np.asarray(world, float))
        self.observed = np.ascontiguousarray(np.asarray(observed, float).reshape(-1, 2))
        self.template = np.asarray(template, float).copy()
        self.scales = np.asarray(quality_scales, float).reshape(-1).copy()
        self.sigma = float(sigma)
        self.dynamic = bool(dynamic_candidates)
        self.frozen = np.asarray(frozen_segments, np.int64).copy()
        self.source_pts = source_pts
        self.edge_name = edge_name
        self.observed_sha256 = hashlib.sha256(
            np.ascontiguousarray(self.observed).tobytes()).hexdigest()
        self.quality_sha256 = hashlib.sha256(
            np.ascontiguousarray(self.scales).tobytes()).hexdigest()
        if self.scales.shape != (len(self.observed),):
            raise ValueError('Source quality scale count differs from original UV')
        self.counts = dict(checks=0, unsafe_columns=0, safe_columns=0,
                           guard_exception=0)

    def _signature(self, camera, all_world):
        all_uv, all_depth = _project15(camera, all_world)
        mask = _central_mask(camera, all_world, raw_uv=all_uv,
                             raw_depth=all_depth)
        n = len(self.world)
        uv, depth = all_uv[:n], all_depth[:n]
        safe = mask[:n-1] & mask[1:n] & mask[n:]
        a, b = uv[:-1], uv[1:]
        broad = (np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
                 & (depth[:-1] > 0) & (depth[1:] > 0))
        if self.dynamic:
            low = self.observed.min(axis=0) - 350.
            high = self.observed.max(axis=0) + 350.
            crossing = ((np.minimum(a, b) <= high).all(axis=1)
                        & (np.maximum(a, b) >= low).all(axis=1))
            selected = np.flatnonzero(crossing & broad & safe)
            if not len(selected):
                selected = np.flatnonzero(broad & safe)
        else:
            selected = self.frozen
        if not len(selected):
            raise BoardIntersectionError('source_edge_no_central_segment',
                                         np.arange(len(self.observed)))
        start, finish = uv[selected], uv[selected + 1]
        direction = finish - start
        length2 = np.sum(direction * direction, axis=1)
        valid = broad[selected] & safe[selected] & (length2 > 1e-4)
        if not valid.any():
            raise BoardIntersectionError('source_edge_no_central_segment',
                                         np.arange(len(self.observed)))
        index, raw_t, _, nearest = nearest_segments(
            self.observed, start, direction, length2, valid, compiled=False)
        nearest_segment = selected[index]
        regime = np.where(raw_t <= 0., -1, np.where(raw_t >= 1., 1, 0))
        tail = (np.linalg.norm((nearest - self.observed) / self.scales[:, None],
                               axis=1) > 2. * self.sigma)
        return selected, nearest_segment, regime, tail

    def unsafe_axes(self, physical15, step=1e-6):
        self.counts['checks'] += 1
        try:
            state = np.asarray(physical15, float)
            centre = from_state(state, self.template)
            all_world = np.vstack((self.world,
                                   .5 * (self.world[:-1] + self.world[1:])))
            baseline = self._signature(centre, all_world)
            unsafe = np.zeros(4, bool)
            for local in range(4):
                h = step * max(1., abs(float(state[11 + local])))
                for sign in (-1., 1.):
                    variant = centre.copy()
                    variant[11 + local] += sign * h
                    other = self._signature(variant, all_world)
                    if any(not np.array_equal(x, y) for x, y in zip(other, baseline)):
                        unsafe[local] = True
                        break
            self.counts['unsafe_columns'] += int(unsafe.sum())
            self.counts['safe_columns'] += int((~unsafe).sum())
            return unsafe
        except (BoardIntersectionError, cv2.error, ValueError, FloatingPointError,
                OverflowError):
            self.counts['guard_exception'] += 1
            self.counts['unsafe_columns'] += 4
            return np.ones(4, bool)


def adapt_edge_factor(keys, raw_residual, raw_jacobian, options,
                      active_keys, fixed_states, model, guard, *,
                      derived_frame_keys=None, brown_fd_step=1e-6):
    """Use physical15 J; preserve complete-residual FD for unsafe columns."""
    def guarded_jacobian(states):
        matrices = list(raw_jacobian(states))
        for index, matrix in enumerate(matrices):
            state = np.asarray(states[index], float)
            matrix = np.asarray(matrix, float)
            if state.shape != (15,) or matrix.shape[1] != 15:
                continue
            unsafe = guard.unsafe_axes(state, brown_fd_step)
            if not unsafe.any():
                continue
            matrix = matrix.copy()
            for local in np.flatnonzero(unsafe):
                h = brown_fd_step * max(1., abs(float(state[11 + local])))
                high = [np.asarray(value, float).copy() for value in states]
                low = [np.asarray(value, float).copy() for value in states]
                high[index][11 + local] += h
                low[index][11 + local] -= h
                matrix[:, 11 + local] = (
                    np.asarray(raw_residual(high), float).reshape(-1)
                    - np.asarray(raw_residual(low), float).reshape(-1)) / (2 * h)
            matrices[index] = matrix
        return matrices

    return adapt_factor(keys, raw_residual, guarded_jacobian, options,
                        active_keys, fixed_states, model,
                        derived_frame_keys=derived_frame_keys,
                        brown_fd_step=brown_fd_step)


def registered_edge_adapter(original_keys, raw_residual, raw_jacobian, options,
                            active_keys, fixed_states, model, *,
                            derived_frame_keys=None, brown_fd_step=1e-6):
    """Drop-in adapt_factor entry: only registered source edges get Brown15."""
    guard = _GUARDS.get(raw_residual)
    if guard is None:
        _ADAPTER_COUNTS['generic_factors'] += 1
        return adapt_factor(original_keys, raw_residual, raw_jacobian, options,
                            active_keys, fixed_states, model,
                            derived_frame_keys=derived_frame_keys,
                            brown_fd_step=brown_fd_step)
    if raw_jacobian is None:
        raise ValueError('Registered source-edge factor has no Jacobian')
    _ADAPTER_COUNTS['registered_edge_factors'] += 1
    return adapt_edge_factor(original_keys, raw_residual, raw_jacobian, options,
                             active_keys, fixed_states, model, guard,
                             derived_frame_keys=derived_frame_keys,
                             brown_fd_step=brown_fd_step)


def install_edge_optin():
    """Instrument only source-edge construction, preserving capture provenance."""
    global _ORIGINALS
    import track_sequence
    if _ORIGINALS is not None:
        return
    original_curve = track_sequence._edge_curve_error
    original_huber = track_sequence.pointwise_huber

    def register_huber(residual, jacobians, *args, **kwargs):
        wrapped = original_huber(residual, jacobians, *args, **kwargs)
        frame = inspect.currentframe().f_back
        if frame.f_code.co_name == '_add_source_edge_factors' and (
                frame.f_globals.get('__name__', '').endswith('track_sequence')):
            local = frame.f_locals
            quality = local['source_quality']
            manifest = None if quality is None else quality.manifest
            # Read the same source multiplier without recording the same UV a
            # second time in the original quality diagnostics.
            scales = (np.ones(len(local['observed'])) if manifest is None else
                      np.broadcast_to(manifest.sigma_multiplier(
                          int(local['stamp']), local['observed']),
                          (len(local['observed']),)).copy())
            _GUARDS[wrapped[0]] = CentralBrownEdgeGuard(
                local['xyz'], local['observed'], local['template'], scales,
                local['sigma'], local['dynamic_candidates'], local['segments'],
                source_pts=int(local['stamp']), edge_name=str(local['name']))
        return wrapped

    track_sequence._edge_curve_error = edge_curve_error15
    track_sequence.pointwise_huber = register_huber
    _ORIGINALS = original_curve, original_huber


def guard_for_spec(spec):
    """Find the measured edge guard for a captured source-edge callback."""
    if (spec.get('file') != 'track_sequence.py'
            or spec.get('function') != '_add_source_edge_factors'):
        return None
    callback = spec['args'][1]
    if callback in _INTERVAL_REPLACEMENTS:
        # Native edge specs encode the original nearest-curve residual only.
        # An interval/qualified-only callback has a different objective.
        return None
    return _GUARDS.get(callback)


def note_interval_replacement(original_spec, replacement_spec, *, observed,
                              eligible, edge_name, qualified_only):
    """Record why a reviewed low-end edge must remain a Python factor.

    This is diagnostic registration, never a Brown15 or native-factor guard.
    The native manager may inspect ``interval_replacement_for_spec`` before
    trying ``guard_for_spec``. The original callback remains available for
    source provenance, but its guard cannot represent the interval objective.
    """
    kind = ('track_sequence.py', '_add_source_edge_factors')
    for spec in (original_spec, replacement_spec):
        if (spec.get('file'), spec.get('function')) != kind or spec.get('kind') != 'observation':
            raise ValueError('Interval replacement changed source-edge provenance')
    source_pts = int(original_spec['keys'][0])
    if (tuple(original_spec['keys']) != (source_pts,)
            or tuple(replacement_spec['keys']) != (source_pts,)
            or original_spec['args'][1] is replacement_spec['args'][1]):
        raise ValueError('Interval replacement changed source key or callback identity')
    uv = np.ascontiguousarray(np.asarray(observed, np.float64).reshape(-1, 2))
    flags = np.asarray(eligible, bool).reshape(-1)
    if len(uv) != len(flags) or not len(uv):
        raise ValueError('Interval replacement source UV/eligibility shape changed')
    original_guard = _GUARDS.get(original_spec['args'][1])
    if _ORIGINALS is not None and original_guard is None:
        raise ValueError('Original source-edge guard is missing during interval replacement')
    if original_guard is not None and (
            original_guard.source_pts != source_pts
            or original_guard.edge_name != edge_name
            or original_guard.observed.shape != uv.shape
            or original_guard.observed.tobytes() != uv.tobytes()):
        raise ValueError('Interval replacement differs from registered original UV')
    record = dict(
        source_pts=source_pts, edge_name=str(edge_name),
        original_uv_count=len(uv), source_qualified_uv_count=int(flags.sum()),
        source_unknown_uv_count=int((~flags).sum()),
        qualified_only=bool(qualified_only), original_guard_present=original_guard is not None,
        native_compatible=False,
        reason='source_normal_interval_and_qualified_rows_not_in_native_edge_spec',
    )
    callback = replacement_spec['args'][1]
    _INTERVAL_REPLACEMENTS[callback] = record
    _ACTIVE_INTERVAL_REPLACEMENTS[(source_pts, str(edge_name))] = callback


def interval_replacement_for_spec(spec):
    """Return a copy of the exact objective-compatibility diagnostic, if any."""
    if (spec.get('file'), spec.get('function')) != (
            'track_sequence.py', '_add_source_edge_factors'):
        return None
    record = _INTERVAL_REPLACEMENTS.get(spec['args'][1])
    return None if record is None else dict(record)


def edge_optin_metadata():
    """Small deterministic input/code identity and runtime guard counters."""
    path = Path(__file__).resolve()
    registry = sorted((int(guard.source_pts), str(guard.edge_name),
                       len(guard.observed), guard.observed_sha256,
                       guard.quality_sha256)
                      for guard in _GUARDS.values())
    digest = hashlib.sha256()
    for row in registry:
        digest.update(repr(row).encode('utf8'))
        digest.update(b'\n')
    counts = Counter()
    for guard in _GUARDS.values():
        counts.update(guard.counts)
    interval_rows = [_INTERVAL_REPLACEMENTS[callback]
                     for callback in _ACTIVE_INTERVAL_REPLACEMENTS.values()]
    backend_provenance = None
    if getattr(_projection._central_projection_backend, '__name__', '') == 'safe_project_compatible':
        import continuous_central_brown4_optin as central_backend
        backend_provenance = dict(
            source_sha256=central_backend.SOURCE_SHA256,
            dll_sha256=central_backend.DLL_SHA256,
            adapter_sha256=hashlib.sha256(
                Path(central_backend.__file__).read_bytes()).hexdigest())
    return dict(
        schema='rinkspace.central-source-edge-brown15-optin.v1',
        enabled=_ORIGINALS is not None,
        module_path=str(path),
        module_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        central_backend=getattr(_projection._central_projection_backend,
                                '__name__', repr(_projection._central_projection_backend)),
        central_backend_provenance=backend_provenance,
        registered_factor_count=len(registry),
        registered_original_uv_count=sum(row[2] for row in registry),
        registered_source_input_manifest_sha256=digest.hexdigest(),
        interval_replacement_coverage=dict(
            factors=len(interval_rows),
            original_uv_count=sum(row['original_uv_count'] for row in interval_rows),
            source_qualified_uv_count=sum(row['source_qualified_uv_count'] for row in interval_rows),
            source_unknown_uv_count=sum(row['source_unknown_uv_count'] for row in interval_rows),
            original_guards_present=sum(row['original_guard_present'] for row in interval_rows),
            native_compatible=False,
            reason='source_normal_interval_and_qualified_rows_not_in_native_edge_spec'),
        adapter_counts=dict(_ADAPTER_COUNTS), guard_counts=dict(counts),
        original_uv_and_quality_retained=True,
        complete_residual_fd_step=1e-6,
        unsafe_brown_columns_use_complete_residual_fd=True)


def uninstall_edge_optin():
    """Restore constructors after all callbacks from this capture are retired."""
    global _ORIGINALS
    if _ORIGINALS is None:
        return
    import track_sequence
    track_sequence._edge_curve_error, track_sequence.pointwise_huber = _ORIGINALS
    _ORIGINALS = None
    _GUARDS.clear()
    _INTERVAL_REPLACEMENTS.clear()
    _ACTIVE_INTERVAL_REPLACEMENTS.clear()
    _ADAPTER_COUNTS.clear()
