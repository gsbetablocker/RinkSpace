"""Pure skating factors; live state/Jacobian integration is in action_live.

All vectors are in a single, caller-declared metric world frame. R maps the
skate/root local frame TO that world frame. Velocities use seconds, metres and
radians. The caller supplies actual source PTS, soft scales, and independently
supported phases/confidences. This module never recognizes or relabels actions.

Disabled rows are zero, but are ALWAYS accompanied by enabled/reason metadata.
They must not be counted as measured zero error. All physical scales are
required arguments; no millimetre acceptance threshold is supplied here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


PHASES = frozenset({
    "stable_glide", "pivot", "push", "toe_pick", "flight", "landing",
    "turn", "spin", "unknown",
})
EVIDENCE_ORIGINS = frozenset({
    "video_observation", "official_protocol", "model_inference", "human_review",
})


def _readonly(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, not a rounded float")
    return int(value)


def _finite(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a nonfinite value")
    return result


def _vectors(value, n, name):
    result = _finite(value, name)
    if result.shape == (3,):
        result = np.broadcast_to(result, (n, 3))
    if result.shape != (n, 3):
        raise ValueError(f"{name} must have shape (3,) or ({n},3)")
    return result


def _scalars(value, n, name, *, confidence=False):
    result = _finite(value, name)
    if result.ndim == 0:
        result = np.full(n, float(result))
    if result.shape != (n,):
        raise ValueError(f"{name} must be scalar or shape ({n},)")
    if confidence:
        if np.any((result < 0) | (result > 1)):
            raise ValueError(f"{name} must be in [0,1]")
    elif np.any(result <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return result


def _rotations(value, n, name):
    result = _finite(value, name)
    if result.shape != (n, 3, 3):
        raise ValueError(f"{name} must have shape ({n},3,3)")
    eye = np.eye(3)
    if (not np.allclose(result @ result.swapaxes(1, 2), eye, atol=1e-6, rtol=0)
            or not np.allclose(np.linalg.det(result), 1., atol=1e-6, rtol=0)):
        raise ValueError(f"{name} must contain proper SO(3) rotations")
    return result


def _phases(value, n):
    if isinstance(value, str):
        result = (value,) * n
    else:
        try:
            result = tuple(value)
        except TypeError as exc:
            raise ValueError("phases must be a string or a sequence of strings") from exc
    if len(result) != n or any(not isinstance(p, str) or p not in PHASES for p in result):
        raise ValueError(f"phases must contain {n} entries from {sorted(PHASES)}")
    return result


@dataclass(frozen=True)
class SourceTimeline:
    """Source PTS are immutable copies; elapsed times avoid a large-epoch loss.

    No frame-rate inference, resampling, sorting or duplicate removal occurs.
    Negative initial PTS are allowed. Adjacent PTS must be strictly increasing.
    """

    pts: np.ndarray
    time_base: tuple[int, int]
    source_id: str
    elapsed_s: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must identify the actual video/clock")
        try:
            values = tuple(self.pts)
            base = tuple(self.time_base)
        except TypeError as exc:
            raise ValueError("pts and time_base must be one-dimensional sequences") from exc
        if not values or len(base) != 2:
            raise ValueError("pts must be nonempty and time_base must be (numerator, denominator)")
        integers = [_integer(v, "PTS") for v in values]
        if any(b <= a for a, b in zip(integers, integers[1:])):
            raise ValueError("source PTS must be strictly increasing without duplicates")
        num, den = (_integer(v, "time_base") for v in base)
        if num <= 0 or den <= 0:
            raise ValueError("time_base numerator and denominator must be positive")
        limits = np.iinfo(np.int64)
        if any(v < limits.min or v > limits.max for v in integers):
            raise ValueError("source PTS must fit int64")
        try:
            scale = num / den
            elapsed = np.array([(v - integers[0]) * scale for v in integers])
        except OverflowError as exc:
            raise ValueError("time range is not representable in float64 seconds") from exc
        if (not np.isfinite(scale) or scale <= 0 or not np.isfinite(elapsed).all()
                or (len(elapsed) > 1 and np.any(np.diff(elapsed) <= 0))):
            raise ValueError("PTS differences cannot be represented as positive finite seconds")
        object.__setattr__(self, "pts", _readonly(integers, np.int64))
        object.__setattr__(self, "time_base", (num, den))
        object.__setattr__(self, "elapsed_s", _readonly(elapsed, np.float64))

    def __len__(self):
        return len(self.pts)


def validate_source_timeline(pts, time_base, *, source_id):
    return SourceTimeline(pts, time_base, source_id)


@dataclass(frozen=True)
class FactorRows:
    source_id: str
    pts: np.ndarray
    time_base: tuple[int, int]
    residual: np.ndarray
    enabled: np.ndarray
    reasons: tuple[str, ...]
    diagnostics: Mapping[str, np.ndarray]

    @property
    def status(self):
        if not self.enabled.any():
            return "disabled"
        return "active" if self.enabled.all() else "partially_active"


def _rows(clock, pts, residual, enabled, reasons, diagnostics):
    if not np.isfinite(residual).all() or any(not np.isfinite(v).all() for v in diagnostics.values()):
        raise ValueError("factor arithmetic produced a nonfinite value")
    return FactorRows(
        clock.source_id, _readonly(pts, np.int64), clock.time_base,
        _readonly(residual, np.float64), _readonly(enabled, bool), tuple(reasons),
        {k: _readonly(v) for k, v in diagnostics.items()},
    )


def glide_lateral_residual(
    clock: SourceTimeline, *, rotation_world, origin_velocity_world_mps,
    angular_velocity_world_radps, contact_point_local_m, blade_tangent_local,
    ice_normal_world, phases, confidence, sigma_mps,
) -> FactorRows:
    """Soft lateral MATERIAL-point velocity for independently supported glide.

    origin_velocity and omega are derivatives of the same O(t), R(t) used by
    p(t)=O(t)+R(t)q*. The caller obtains them from its continuous state or a
    separately validated timestamp-aware derivative. q* may vary between rows;
    its derivative is deliberately NOT added to the rigid material velocity.

    Only phase 'stable_glide' enables a row. Pivot, push, toe-pick, flight,
    landing, turn, spin, and unknown disable it by default. Confidence is a
    fixed reliability weight, not a measured probability or solver variable.
    sigma_mps is a caller-provided soft scale, not a physical pass/fail limit.
    """
    if not isinstance(clock, SourceTimeline):
        raise ValueError("clock must be a validated SourceTimeline")
    n = len(clock)
    R = _rotations(rotation_world, n, "rotation_world")
    v = _vectors(origin_velocity_world_mps, n, "origin_velocity_world_mps")
    omega = _vectors(angular_velocity_world_radps, n, "angular_velocity_world_radps")
    q = _vectors(contact_point_local_m, n, "contact_point_local_m")
    tangent = _vectors(blade_tangent_local, n, "blade_tangent_local")
    normal = _vectors(ice_normal_world, n, "ice_normal_world")
    phase = _phases(phases, n)
    weight = _scalars(confidence, n, "confidence", confidence=True)
    sigma = _scalars(sigma_mps, n, "sigma_mps")
    normal_length = np.linalg.norm(normal, axis=1)
    tangent_length = np.linalg.norm(tangent, axis=1)
    if np.any(normal_length < 1e-12) or np.any(tangent_length < 1e-12):
        raise ValueError("normal and blade tangent must be nonzero vectors")
    normal = normal / normal_length[:, None]
    tangent_world = np.einsum("nij,nj->ni", R, tangent / tangent_length[:, None])
    tangent_plane = tangent_world - normal * np.sum(tangent_world * normal, axis=1)[:, None]
    projected_length = np.linalg.norm(tangent_plane, axis=1)
    valid_tangent = projected_length > 1e-10
    tangent_plane = np.divide(
        tangent_plane, projected_length[:, None], out=np.zeros_like(tangent_plane),
        where=valid_tangent[:, None],
    )
    lateral = np.cross(normal, tangent_plane)
    lever = np.einsum("nij,nj->ni", R, q)
    material_velocity = v + np.cross(omega, lever)
    side_speed = np.sum(lateral * material_velocity, axis=1)
    enabled = np.array([p == "stable_glide" for p in phase]) & (weight > 0) & valid_tangent
    reasons = [
        "active" if enabled[i] else
        (f"phase:{phase[i]}" if phase[i] != "stable_glide" else
         ("zero_confidence" if weight[i] <= 0 else "blade_tangent_normal_to_ice"))
        for i in range(n)
    ]
    residual = np.where(enabled, np.sqrt(weight) * side_speed / sigma, 0.)[:, None]
    return _rows(clock, clock.pts, residual, enabled, reasons, {
        "elapsed_s": clock.elapsed_s,
        "material_velocity_world_mps": material_velocity,
        "lateral_velocity_mps": side_speed,
        "along_blade_velocity_mps": np.sum(tangent_plane * material_velocity, axis=1),
        "projected_tangent_valid": valid_tangent,
    })


@dataclass(frozen=True)
class MassWeightedCOM:
    """A declared segment mass model, not an arbitrary root/pelvis trajectory.

    segment_com_world_m are segment MASS CENTRES, not just joint positions.
    This contract verifies arithmetic, names and positive masses; it cannot
    certify the caller's anthropometry. Store mass_model_id and perform mass
    sensitivity checks before assigning a strong ballistic weight.
    """

    source_clock: SourceTimeline
    segment_com_world_m: np.ndarray
    segment_masses_kg: np.ndarray
    segment_names: Sequence[str]
    mass_model_id: str
    position_world_m: np.ndarray = field(init=False)

    def __post_init__(self):
        if not isinstance(self.source_clock, SourceTimeline):
            raise ValueError("CoM requires its actual source clock")
        centres = _finite(self.segment_com_world_m, "segment_com_world_m")
        masses = _finite(self.segment_masses_kg, "segment_masses_kg")
        try:
            names = tuple(self.segment_names)
        except TypeError as exc:
            raise ValueError("segment_names must be a sequence of strings") from exc
        if centres.ndim != 3 or centres.shape[0] < 1 or centres.shape[2] != 3:
            raise ValueError("segment_com_world_m must have shape (N,segments,3)")
        if centres.shape[0] != len(self.source_clock):
            raise ValueError("segment mass centres must align with their source clock")
        count = centres.shape[1]
        if count < 2 or masses.shape != (count,) or len(names) != count:
            raise ValueError("a whole-body CoM requires at least two named mass segments")
        if (any(not isinstance(x, str) or not x.strip() for x in names)
                or len(set(names)) != count or np.any(masses <= 0)):
            raise ValueError("segment names must be unique; every segment mass must be positive")
        if not isinstance(self.mass_model_id, str) or not self.mass_model_id.strip():
            raise ValueError("mass_model_id is required")
        total_mass = masses.sum()
        if not np.isfinite(total_mass):
            raise ValueError("total body mass is not finite")
        position = np.einsum("nsc,s->nc", centres, masses / total_mass)
        if not np.isfinite(position).all():
            raise ValueError("computed mass-weighted CoM is not finite")
        object.__setattr__(self, "segment_com_world_m", _readonly(centres))
        object.__setattr__(self, "segment_masses_kg", _readonly(masses))
        object.__setattr__(self, "segment_names", names)
        object.__setattr__(self, "position_world_m", _readonly(position))


def airborne_ballistic_residual(
    clock: SourceTimeline, com: MassWeightedCOM, *, phases, confidence,
    gravity_world_mps2, sigma_mps2,
) -> FactorRows:
    """Nonuniform three-sample COM acceleration minus explicit world gravity.

    The complete stencil must be 'flight' with positive confidence; contact or
    uncertain boundary samples disable a row. Results belong to the middle
    SOURCE sample, and exact stencil PTS are included. The three-point formula
    is exact for a quadratic trajectory even with nonuniform source PTS.
    """
    if not isinstance(clock, SourceTimeline):
        raise ValueError("clock must be a validated SourceTimeline")
    if not isinstance(com, MassWeightedCOM):
        raise ValueError("com must be MassWeightedCOM; pelvis/root arrays are not accepted")
    if (com.source_clock.source_id != clock.source_id or com.source_clock.time_base != clock.time_base
            or not np.array_equal(com.source_clock.pts, clock.pts)):
        raise ValueError("CoM and factor source timestamps must be identical")
    n = len(clock)
    if n < 3 or com.position_world_m.shape != (n, 3):
        raise ValueError("CoM and source clock must share at least three aligned samples")
    phase = _phases(phases, n)
    weight = _scalars(confidence, n, "confidence", confidence=True)
    sigma = _scalars(sigma_mps2, n - 2, "sigma_mps2")
    gravity = _vectors(gravity_world_mps2, n - 2, "gravity_world_mps2")
    if np.any(np.linalg.norm(gravity, axis=1) < 1e-12):
        raise ValueError("gravity_world_mps2 must be a nonzero world-frame gravity vector")
    dt = np.diff(clock.elapsed_s)
    C = com.position_world_m
    acceleration = 2 * ((C[2:] - C[1:-1]) / dt[1:, None]
                        - (C[1:-1] - C[:-2]) / dt[:-1, None]) / (dt[1:] + dt[:-1])[:, None]
    stencil_weight = np.minimum(np.minimum(weight[:-2], weight[1:-1]), weight[2:])
    in_flight = np.array([all(p == "flight" for p in phase[i:i+3]) for i in range(n-2)])
    enabled = in_flight & (stencil_weight > 0)
    reasons = ["active" if enabled[i] else
               ("stencil_crosses_contact_or_unknown" if not in_flight[i] else "zero_confidence")
               for i in range(n - 2)]
    physical_residual = acceleration - gravity
    residual = np.where(enabled[:, None],
                        np.sqrt(stencil_weight)[:, None] * physical_residual / sigma[:, None], 0.)
    return _rows(clock, clock.pts[1:-1], residual, enabled, reasons, {
        "elapsed_s": clock.elapsed_s[1:-1],
        "stencil_pts": np.stack((clock.pts[:-2], clock.pts[1:-1], clock.pts[2:]), axis=1),
        "com_acceleration_world_mps2": acceleration,
        "acceleration_minus_gravity_world_mps2": physical_residual,
        "stencil_confidence": stencil_weight,
    })


@dataclass(frozen=True)
class RootSO3Diagnostics:
    source_id: str
    source_pts: np.ndarray
    time_base: tuple[int, int]
    velocity_interval_pts: np.ndarray
    velocity_midpoint_elapsed_s: np.ndarray
    omega_world_radps: np.ndarray
    principal_increment_rad: np.ndarray
    near_pi_ambiguous: np.ndarray
    acceleration_stencil_pts: np.ndarray
    acceleration_elapsed_s: np.ndarray
    alpha_world_radps2: np.ndarray
    acceleration_status: str


def root_so3_diagnostics(clock: SourceTimeline, rotations_world) -> RootSO3Diagnostics:
    """World-frame SO(3) finite differences of the FINAL world root rotations.

    Logs of R[i+1]@R[i].T avoid a yaw +/-pi wrap. They cannot recover unobserved
    rotations >=pi between samples; near_pi_ambiguous marks the principal-log
    ambiguity. This is a diagnostic, not smoothing or a zero-velocity prior.
    Acceleration differences neighbouring interval-midpoint angular velocities.
    """
    if not isinstance(clock, SourceTimeline) or len(clock) < 2:
        raise ValueError("root diagnostics require a validated clock with at least two samples")
    n = len(clock)
    R = _rotations(rotations_world, n, "rotations_world")
    relative_world = R[1:] @ R[:-1].swapaxes(1, 2)
    increment = Rotation.from_matrix(relative_world).as_rotvec()
    dt = np.diff(clock.elapsed_s)
    omega = increment / dt[:, None]
    angle = np.linalg.norm(increment, axis=1)
    midpoint = (clock.elapsed_s[:-1] + clock.elapsed_s[1:]) * .5
    alpha = np.diff(omega, axis=0) / np.diff(midpoint)[:, None]
    if not np.isfinite(omega).all() or not np.isfinite(alpha).all():
        raise ValueError("root rotation differences produced nonfinite rates")
    return RootSO3Diagnostics(
        source_id=clock.source_id, source_pts=_readonly(clock.pts), time_base=clock.time_base,
        velocity_interval_pts=_readonly(np.stack((clock.pts[:-1], clock.pts[1:]), axis=1)),
        velocity_midpoint_elapsed_s=_readonly(midpoint), omega_world_radps=_readonly(omega),
        principal_increment_rad=_readonly(angle), near_pi_ambiguous=_readonly(angle >= np.pi - 1e-6),
        acceleration_stencil_pts=_readonly(np.stack((clock.pts[:-2], clock.pts[1:-1], clock.pts[2:]), axis=1)),
        acceleration_elapsed_s=_readonly((midpoint[:-1] + midpoint[1:]) * .5),
        alpha_world_radps2=_readonly(alpha),
        acceleration_status="available" if n >= 3 else "disabled:insufficient_samples",
    )


@dataclass(frozen=True)
class ActionEvidence:
    """Provenance-bearing proposal; no label-to-support or ideal-edge rules.

    `label` is descriptive (e.g. a protocol element or video event). It does not
    activate a factor. The application resolves phases/contact from evidence
    outside the optimizer, explicitly retaining unknowns and disagreements.
    """

    source_id: str
    start_pts: int
    end_pts: int
    origin: str
    label: str
    confidence: float
    support_side: str = "unknown"
    left_edge: str = "unknown"
    right_edge: str = "unknown"
    evidence_pts: tuple[int, ...] = ()
    human_confirmed: bool = False
    score_is_calibrated_probability: bool = False

    def validate(self, clock: SourceTimeline) -> "ActionEvidence":
        if not isinstance(clock, SourceTimeline) or self.source_id != clock.source_id:
            raise ValueError("event and clock source_id must match")
        if not isinstance(self.origin, str) or self.origin not in EVIDENCE_ORIGINS:
            raise ValueError("unknown evidence origin")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("event label must be nonempty")
        start = _integer(self.start_pts, "start_pts")
        end = _integer(self.end_pts, "end_pts")
        if start > end:
            raise ValueError("event start_pts must not exceed end_pts")
        if not isinstance(self.support_side, str) or self.support_side not in {"left", "right", "both", "unknown"}:
            raise ValueError("support_side must retain left/right/both/unknown semantics")
        if (not isinstance(self.left_edge, str) or not isinstance(self.right_edge, str)
                or self.left_edge not in {"inner", "outer", "flat", "unknown"} or self.right_edge not in {
                "inner", "outer", "flat", "unknown"}):
            raise ValueError("edge must be inner/outer/flat/unknown")
        _scalars(self.confidence, 1, "event confidence", confidence=True)
        if not isinstance(self.human_confirmed, bool) or not isinstance(self.score_is_calibrated_probability, bool):
            raise ValueError("confirmation/probability flags must be explicit booleans")
        if self.origin == "official_protocol" and (self.support_side != "unknown"
                or self.left_edge != "unknown" or self.right_edge != "unknown"):
            raise ValueError("protocol labels alone cannot certify the actual support foot or blade edge")
        source_pts = set(map(int, clock.pts))
        if start not in source_pts or end not in source_pts:
            raise ValueError("event boundaries must reference actual source samples in this clock")
        try:
            evidence_pts = tuple(self.evidence_pts)
        except TypeError as exc:
            raise ValueError("evidence_pts must be a sequence of integer source PTS") from exc
        for value in evidence_pts:
            pts = _integer(value, "evidence_pts")
            if pts not in source_pts:
                raise ValueError("evidence_pts contains a sample absent from the source clock")
        return self
