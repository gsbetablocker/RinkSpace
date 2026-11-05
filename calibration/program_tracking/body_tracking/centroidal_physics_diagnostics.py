"""Small, explicit centroidal diagnostics for a saved articulated trajectory.

Reference implementation for the 036990c RinkSpace audit, not an automatic
acceptance gate, motion filter, or drop-in optimizer. Units are metres/seconds.
Source timestamps, phase evidence, segment mass centres, and segment inertias
must be supplied by the caller. No pelvis-as-COM or point-mass-only shortcut.

Only NumPy and SciPy are used. Angular momentum is reported per total mass, so
relative masses are sufficient; no athlete body mass is invented. The uniform
convex-volume helper remains an anthropometric proxy, not a measured body.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.transform import Rotation

from .segment_mass_proxy import tetraVolumeProperties


def source_seconds(pts, time_base):
    """Elapsed seconds from immutable integer PTS; never infer frame rate."""
    p = np.asarray(pts)
    b = np.asarray(time_base)
    if (p.ndim != 1 or p.dtype.kind not in "iu" or not len(p)
            or b.shape != (2,) or b.dtype.kind not in "iu" or np.any(b <= 0)):
        raise ValueError("Require integer source PTS and positive rational time base")
    # Python integers prevent subtraction overflow with a large signed epoch.
    t = np.array([(int(x) - int(p[0])) * int(b[0]) / int(b[1]) for x in p])
    return _times(t)


def _times(times, minimum=1):
    t = np.asarray(times, float)
    if (t.ndim != 1 or len(t) < minimum or not np.isfinite(t).all()
            or np.any(np.diff(t) <= 0)):
        raise ValueError("Require finite, strictly increasing actual timestamps")
    return t - t[0]


def _finite(value, shape, name):
    a = np.asarray(value, float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return a


def _rotations(value, shape):
    R = _finite(value, shape, "world_from_segment rotations")
    # This is an algebraic SO(3) check, not a physical accuracy tolerance.
    if (not np.allclose(R.swapaxes(-1, -2) @ R, np.eye(3), atol=1e-7, rtol=0)
            or np.any(np.linalg.det(R) <= 0)):
        raise ValueError("Segment orientations must be proper SO(3) rotations")
    return R


def second_difference_operator(times):
    """Nonuniform three-point acceleration; each row uses all three samples."""
    t = _times(times, 3)
    left, right = np.diff(t)[:-1], np.diff(t)[1:]
    D = np.zeros((len(t) - 2, len(t)))
    k = np.arange(len(D))
    D[k, k] = 2 / (left * (left + right))
    D[k, k + 1] = -2 / (left * right)
    D[k, k + 2] = 2 / (right * (left + right))
    return D


def flight_projection(times, com_world_m, gravity_world_mps2, *,
                      temporal_covariance_m2=None):
    """Fit c0,v0 with fixed g on ONE independently confirmed flight interior.

    A position-domain integral of c''=g avoids making raw 50 Hz second
    differences the only physical objective. It does not determine takeoff,
    landing, depth, contact, initial speed, or source alignment by itself.

    Optional covariance is (N,N), in m^2, common across x/y/z. It may contain
    temporal correlations. A full anisotropic covariance belongs upstream.
    Without it this returns an equal-weight geometric diagnostic, not a
    calibrated likelihood or a suggested optimizer weight.
    """
    t = _times(times, 3)
    C = _finite(com_world_m, (len(t), 3), "COM")
    g = _finite(gravity_world_mps2, (3,), "gravity")
    A = np.column_stack((np.ones(len(t)), t))
    q = .5 * t[:, None] ** 2 * g
    D2 = second_difference_operator(t)
    if temporal_covariance_m2 is None:
        L = None
        Aw, Yw = A, C - q
    else:
        covariance = _finite(temporal_covariance_m2, (len(t), len(t)), "covariance")
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-10):
            raise ValueError("Temporal covariance must be symmetric")
        L = np.linalg.cholesky(covariance)
        Aw = solve_triangular(L, A, lower=True)
        Yw = solve_triangular(L, C - q, lower=True)
    parameters = np.linalg.lstsq(Aw, Yw, rcond=None)[0]
    predicted = A @ parameters + q
    residual = C - predicted
    white = None if L is None else solve_triangular(L, residual, lower=True)
    # Gravity's observable curvature after eliminating arbitrary c0,v0.
    gravity_curvature = q - A @ np.linalg.lstsq(A, q, rcond=None)[0]
    result = dict(
        initial_com_world_m=parameters[0], initial_velocity_world_mps=parameters[1],
        fitted_com_world_m=predicted, position_residual_m=residual,
        position_error_max_m=float(np.linalg.norm(residual, axis=1).max()),
        position_error_rms_m=float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))),
        acceleration_world_mps2=D2 @ C,
        acceleration_minus_gravity_mps2=D2 @ C - g,
        acceleration_stencil_indices=np.column_stack((np.arange(len(D2)),
                                                      np.arange(len(D2)) + 1,
                                                      np.arange(len(D2)) + 2)),
        duration_s=float(t[-1]),
        gravity_departure_from_endpoint_chord_m=float(np.linalg.norm(g) * t[-1] ** 2 / 8),
        gravity_curvature_after_affine_removal_m=gravity_curvature,
        independent_position_noise_to_acceleration_gain_per_s2=np.linalg.norm(D2, axis=1),
        whitened_residual=white,
        temporal_covariance_supplied=temporal_covariance_m2 is not None,
        likelihood_calibrated_by_this_module=False,
        acceptance_gate=False,
    )
    if temporal_covariance_m2 is not None:
        result["acceleration_noise_covariance_m2_s4"] = D2 @ covariance @ D2.T
    return result


def discrete_segment_rates(times, centres_world_m, rotation_world_from_segment, *,
                           phase_ids=None, confirmed_phase=None):
    """Diagnostic rates on interior source samples, with explicit stencils.

    Prefer velocities from the SAME canonical continuous FK in optimization.
    Here COM velocities use a local quadratic derivative; world angular rates
    interpolate adjacent SO(3) log rates in actual time. The latter is a
    finite-sampling approximation, not an exact general continuous derivative.
    No filtering or frame deletion occurs. Optional phase_ids/confirmed_phase
    identify which complete stencils remain within one independently confirmed
    phase. Without evidence every stencil is marked invalid for physical use;
    rates remain available as diagnostics. Invalid rows are never auto-used.
    """
    t = _times(times, 3)
    X = np.asarray(centres_world_m, float)
    if X.ndim != 3 or X.shape[0] != len(t) or X.shape[2] != 3 or not np.isfinite(X).all():
        raise ValueError("Segment centres must be finite (N,S,3)")
    N, S, _ = X.shape
    R = _rotations(rotation_world_from_segment, (N, S, 3, 3))
    h = np.diff(t)
    left, right = h[:-1], h[1:]
    velocity = ((-right / (left * (left + right)))[:, None, None] * X[:-2]
                + ((right - left) / (left * right))[:, None, None] * X[1:-1]
                + (left / (right * (left + right)))[:, None, None] * X[2:])
    relative = R[1:] @ R[:-1].swapaxes(-1, -2)
    increment = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(N-1, S, 3)
    interval_omega = increment / h[:, None, None]
    omega = (right[:, None, None] * interval_omega[:-1]
             + left[:, None, None] * interval_omega[1:]) / (left + right)[:, None, None]
    angle = np.linalg.norm(increment, axis=2)
    stencil = np.column_stack((np.arange(N-2), np.arange(N-2)+1, np.arange(N-2)+2))
    intervals = np.column_stack((np.arange(N-1), np.arange(N-1)+1))
    if (phase_ids is None) != (confirmed_phase is None):
        raise ValueError("Supply phase_ids and evidence-derived confirmed_phase together")
    if phase_ids is not None and np.asarray(phase_ids).shape != (N,):
        raise ValueError("Phase evidence must align with source samples")
    valid = (np.zeros(N-2, dtype=bool) if phase_ids is None else
             same_phase_stencil_valid(phase_ids, stencil, confirmed=confirmed_phase))
    interval_valid = (np.zeros(N-1, dtype=bool) if phase_ids is None else
                      same_phase_stencil_valid(phase_ids, intervals, confirmed=confirmed_phase))
    return dict(
        source_indices=np.arange(1, N-1),
        source_stencil_indices=stencil,
        derivative_stencil_valid=valid,
        interval_stencil_valid=interval_valid,
        phase_evidence_supplied=phase_ids is not None,
        segment_velocity_world_mps=velocity,
        segment_omega_world_radps=omega,
        interval_omega_world_radps=interval_omega,
        principal_interval_rotation_rad=angle,
        near_pi_aliasing_warning=angle >= np.pi - 1e-6,
        derivative_origin="unfiltered, timestamp-aware discrete diagnostic; prefer continuous FK rates",
    )


def centroidal_momentum(*, centres_world_m, velocity_world_mps,
                        rotation_world_from_segment, omega_world_radps,
                        segment_relative_masses=None, unit_mass_inertia_segment_m2,
                        segment_mass_fractions=None):
    """Exact algebraic aggregate for the supplied rigid-segment state.

    Inertia is about each segment's OWN mass centre, expressed in its fixed
    local frame, per unit segment mass. R maps that frame into world axes.
    The caller must not give inertia about a joint or world origin here.

    H/M = sum_i w_i [(c_i-C) x (v_i-V) + R_i I_i R_i.T omega_i].
    Includes orbital AND intrinsic terms; root-yaw * Izz is not substituted.
    Supply EITHER positive relative masses (normalized once here), OR the
    prepared proxy's segment_mass_fractions (validated and used unchanged).
    """
    X = np.asarray(centres_world_m, float)
    if X.ndim != 3 or X.shape[2] != 3 or not np.isfinite(X).all():
        raise ValueError("Segment centres must be finite (N,S,3)")
    N, S, _ = X.shape
    V = _finite(velocity_world_mps, X.shape, "segment velocities")
    W = _finite(omega_world_radps, X.shape, "world angular velocities")
    R = _rotations(rotation_world_from_segment, (N, S, 3, 3))
    if (segment_relative_masses is None) == (segment_mass_fractions is None):
        raise ValueError("Supply either relative masses or already normalized mass fractions")
    supplied_fractions = segment_mass_fractions is not None
    m = _finite(segment_mass_fractions if supplied_fractions else segment_relative_masses,
                (S,), "segment masses")
    if np.any(m <= 0) or not np.isfinite(m.sum()):
        raise ValueError("Every declared segment must have positive mass with finite total")
    if supplied_fractions and not np.isclose(m.sum(), 1., atol=1e-12, rtol=1e-12):
        raise ValueError("Declared mass fractions must already sum to one")
    I = _finite(unit_mass_inertia_segment_m2, (S, 3, 3), "unit-mass inertia")
    if not np.allclose(I, I.swapaxes(-1, -2), atol=1e-12, rtol=1e-10):
        raise ValueError("Segment inertia tensors must be symmetric")
    eigen = np.linalg.eigvalsh(I)
    if np.any(eigen < -1e-12) or np.any(eigen[:, 2] > eigen[:, 0] + eigen[:, 1] + 1e-12):
        raise ValueError("Segment inertia must be positive semidefinite with physical principal moments")
    weights = m.copy() if supplied_fractions else m / m.sum()
    C = np.einsum("s,nsc->nc", weights, X)
    VC = np.einsum("s,nsc->nc", weights, V)
    orbital_by_segment = weights[None, :, None] * np.cross(X - C[:, None], V - VC[:, None])
    world_inertia = R @ I[None] @ R.swapaxes(-1, -2)
    intrinsic_by_segment = weights[None, :, None] * np.einsum("nsij,nsj->nsi", world_inertia, W)
    orbital = orbital_by_segment.sum(axis=1)
    intrinsic = intrinsic_by_segment.sum(axis=1)
    return dict(
        com_world_m=C, com_velocity_world_mps=VC,
        specific_angular_momentum_world_m2ps=orbital + intrinsic,
        specific_orbital_momentum_world_m2ps=orbital,
        specific_intrinsic_momentum_world_m2ps=intrinsic,
        specific_angular_momentum_by_segment_world_m2ps=orbital_by_segment + intrinsic_by_segment,
        normalized_mass_fractions=weights,
        specific_H_orbital_world_m2ps=orbital,
        specific_H_intrinsic_world_m2ps=intrinsic,
        input_masses_already_normalized=supplied_fractions,
        absolute_mass_inferred=False, physical_accuracy_certified=False,
    )


def phase_safe_flight_diagnostics(times, com_world_m, confirmed_flight, gravity_world_mps2,
                                  *, specific_H_world_m2ps=None, H_rate_stencil_valid=None):
    """Report each contiguous confirmed run; never bridge unknown/contact.

    confirmed_flight comes from source evidence, NOT predicted shoe height.
    If H comes from discrete rates, H_rate_stencil_valid must be false for
    any sample whose full derivative stencil crosses unknown or contact.
    This diagnostic does not enforce constant H on a supported ice spin.
    """
    t = _times(times)
    C = _finite(com_world_m, (len(t), 3), "COM")
    flight = np.asarray(confirmed_flight)
    if flight.shape != (len(t),) or flight.dtype.kind != "b":
        raise ValueError("confirmed_flight must be an evidence-derived boolean per source sample")
    H = valid = None
    if specific_H_world_m2ps is not None:
        H = _finite(specific_H_world_m2ps, (len(t), 3), "specific H")
        valid = np.asarray(H_rate_stencil_valid)
        if valid.shape != flight.shape or valid.dtype.kind != "b":
            raise ValueError("Explicit all-stencil-valid mask required with angular momentum")
    indices = np.flatnonzero(flight)
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if len(indices) else []
    reports = []
    for idx in runs:
        record = dict(source_indices=idx, source_elapsed_range_s=t[idx[[0, -1]]], source_sample_count=len(idx))
        if len(idx) < 3:
            record["status"] = "insufficient confirmed samples for curvature"
        else:
            record.update(status="diagnostic_only", flight=flight_projection(t[idx], C[idx], gravity_world_mps2))
        if H is not None:
            hi = idx[valid[idx]]
            record["angular_momentum_valid_source_indices"] = hi
            if len(hi) > 1:
                record["specific_H_range_world_m2ps"] = np.ptp(H[hi], axis=0)
                # Interval delta H is angular impulse per mass, not angular acceleration.
                # Do not form a constraint across a missing derivative stencil.
                adjacent = np.diff(hi) == 1
                record["specific_angular_impulse_source_pairs"] = np.stack((hi[:-1][adjacent], hi[1:][adjacent]), axis=1)
                record["specific_angular_impulse_world_m2ps"] = np.diff(H[hi], axis=0)[adjacent]
        reports.append(record)
    return reports


def convex_uniform_volume_properties(points):
    """Uniform convex hull volume, centroid, and centre-based inertia per mass.

    Use on the SAME segment vertex assignments as the existing mass proxy.
    Multiply positions by the single validated metres-per-model-unit scale
    before calling, or scale the output inertia by scale**2 exactly once.
    This does not repair overlapping hulls, density uncertainty, or bad
    anatomical segmentation. A combined rigid skate must explicitly declare
    its template and unmeasured equipment-mass assumption.
    """
    volume, centre, inertia = tetraVolumeProperties(points)
    return dict(volume_m3=volume, centre_m=centre,
                unit_mass_inertia_centre_m2=inertia,
                density_model="uniform convex volume; unmeasured anthropometric proxy")



def same_phase_stencil_valid(phase_ids, source_stencil_indices, *, confirmed):
    """True only when EVERY stencil sample has the same confirmed phase ID.

    Caller phase IDs must distinguish independently supported phase runs;
    confirmed is explicit source evidence, never inferred from model height.
    Indices refer to the full source array, not compressed selected samples.
    """
    phase = np.asarray(phase_ids)
    evidence = np.asarray(confirmed)
    stencil = np.asarray(source_stencil_indices)
    if (phase.ndim != 1 or evidence.shape != phase.shape or evidence.dtype.kind != 'b'
            or stencil.ndim != 2 or stencil.shape[1] < 1 or stencil.dtype.kind not in 'iu'
            or np.any(stencil < 0) or np.any(stencil >= len(phase))):
        raise ValueError('Require aligned phase IDs/boolean evidence and integer source stencils')
    same = np.all(phase[stencil] == phase[stencil[:, :1]], axis=1)
    return same & np.all(evidence[stencil], axis=1)


def flight_position_operator(times, gravity_world_mps2, *, temporal_covariance_m2=None):
    """Linear reference residual eliminating c0/v0 on ONE confirmed flight run.

    residual = operator @ (COM_world_m - gravity_position_m). Its nullspace is
    the arbitrary affine initial position/velocity. With covariance the result
    is dimensionless and correlated time samples are whitened; otherwise it is
    metres. This helper neither selects phase nor calibrates a likelihood.
    """
    t = _times(times, 3)
    g = _finite(gravity_world_mps2, (3,), 'gravity')
    A = np.column_stack((np.ones(len(t)), t))
    W = np.eye(len(t))
    if temporal_covariance_m2 is not None:
        covariance = _finite(temporal_covariance_m2, (len(t), len(t)), 'covariance')
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-10):
            raise ValueError('Temporal covariance must be symmetric')
        W = solve_triangular(np.linalg.cholesky(covariance), W, lower=True)
    Q = np.linalg.qr(W @ A, mode='reduced')[0]
    operator = (np.eye(len(t)) - Q @ Q.T) @ W
    return dict(operator=operator, gravity_position_m=.5*t[:, None]**2*g,
                source_indices=np.arange(len(t)),
                residual_units='dimensionless' if temporal_covariance_m2 is not None else 'm',
                likelihood_calibrated_by_this_module=False, acceptance_gate=False)


def angular_impulse_operator(times, phase_ids, *, confirmed_phase,
                             derivative_stencil_valid):
    """Same-phase adjacent delta-H reference; each row is angular impulse/M.

    H must already be in one world frame and use rates from the same current
    state. Its rate-stencil validity is required even for continuous rates
    (where the caller declares valid analytic support). No division by dt is
    hidden here; delta-H is not an angular acceleration or a conservation gate.
    """
    t = _times(times, 2)
    valid = np.asarray(derivative_stencil_valid)
    if valid.shape != (len(t),) or valid.dtype.kind != 'b':
        raise ValueError('Require explicit H derivative-stencil validity per source sample')
    phase = np.asarray(phase_ids)
    if phase.shape != (len(t),):
        raise ValueError('Phase IDs must align with timestamps')
    pairs = np.column_stack((np.arange(len(t)-1), np.arange(1, len(t))))
    accepted = same_phase_stencil_valid(phase, pairs, confirmed=confirmed_phase) & np.all(valid[pairs], axis=1)
    pairs = pairs[accepted]
    operator = np.zeros((len(pairs), len(t)))
    rows = np.arange(len(pairs))
    operator[rows, pairs[:, 0]] = -1.
    operator[rows, pairs[:, 1]] = 1.
    return dict(operator=operator, source_pairs=pairs,
                interval_duration_s=t[pairs[:, 1]]-t[pairs[:, 0]],
                source_pair_valid=accepted, residual_units='m2/s', acceptance_gate=False)


def ground_force_torque_diagnostics(times, com_world_m, specific_H_world_m2ps,
                                     confirmed_support, gravity_world_mps2, *,
                                     H_rate_stencil_valid, phase_ids=None):
    """Infer external force/M and torque-about-COM/M inside confirmed support.

    f_external/M = COM'' - gravity and torque_external,COM/M = Hdot/M.
    All output derivative rows keep their original source stencils; none may
    cross unknown/contact-phase changes. Hdot also requires EVERY constituent
    H rate stencil to be valid. This is a timestamp-aware discrete diagnostic,
    not a force-plate measurement or a ground-contact feasibility constraint.
    Supported spin is allowed nonzero torque and changing H.
    """
    t = _times(times, 3)
    C = _finite(com_world_m, (len(t), 3), 'COM')
    H = _finite(specific_H_world_m2ps, (len(t), 3), 'specific H')
    g = _finite(gravity_world_mps2, (3,), 'gravity')
    support = np.asarray(confirmed_support)
    hvalid = np.asarray(H_rate_stencil_valid)
    if (support.shape != (len(t),) or support.dtype.kind != 'b'
            or hvalid.shape != support.shape or hvalid.dtype.kind != 'b'):
        raise ValueError('Require evidence-derived support and H stencil boolean masks')
    phase = np.zeros(len(t), dtype=int) if phase_ids is None else np.asarray(phase_ids)
    if phase.shape != (len(t),):
        raise ValueError('Phase IDs must align with source samples')
    stencil = np.column_stack((np.arange(len(t)-2), np.arange(1, len(t)-1), np.arange(2, len(t))))
    safe = same_phase_stencil_valid(phase, stencil, confirmed=support)
    torque_safe = safe & np.all(hvalid[stencil], axis=1)
    left, right = np.diff(t)[:-1], np.diff(t)[1:]
    D1 = np.zeros((len(t)-2, len(t)))
    rows = np.arange(len(t)-2)
    D1[rows, rows] = -right / (left*(left+right))
    D1[rows, rows+1] = (right-left) / (left*right)
    D1[rows, rows+2] = left / (right*(left+right))
    impulse = angular_impulse_operator(t, phase, confirmed_phase=support,
                                       derivative_stencil_valid=hvalid)
    return dict(
        force_source_indices=np.arange(1, len(t)-1)[safe],
        force_source_stencil_indices=stencil[safe],
        force_derivative_stencil_valid=safe,
        specific_total_external_force_world_mps2=second_difference_operator(t)[safe] @ C - g,
        torque_source_indices=np.arange(1, len(t)-1)[torque_safe],
        torque_source_stencil_indices=stencil[torque_safe],
        torque_derivative_stencil_valid=torque_safe,
        specific_external_torque_com_world_m2ps2=D1[torque_safe] @ H,
        specific_angular_impulse_source_pairs=impulse['source_pairs'],
        specific_angular_impulse_world_m2ps=impulse['operator'] @ H,
        derivative_origin='unfiltered, timestamp-aware same-confirmed-phase discrete diagnostic',
        external_torque_about='whole-body COM; same world frame as H',
        absolute_mass_inferred=False, support_feasibility_enforced=False,
        physical_accuracy_certified=False, acceptance_gate=False)
