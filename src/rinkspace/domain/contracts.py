"""Small, immutable input contracts. No solver, renderer or historical imports.

Admission is based on source/feature identity and measurement quality. Historical
origin is not a rejection reason; human review of every automatic row is not
required. These checks do not assert visual accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import re
from typing import Mapping


class ContractError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _finite(values: tuple[float, ...]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


@dataclass(frozen=True)
class Source:
    source_id: str
    sha256: str
    width: int
    height: int
    time_base_num: int
    time_base_den: int
    pts: frozenset[int]

    def __post_init__(self) -> None:
        _require(bool(self.source_id), "source_id is required")
        _require(bool(re.fullmatch(r"[0-9a-f]{64}", self.sha256)), "invalid source SHA256")
        _require(all(type(v) is int for v in (self.width, self.height, self.time_base_num, self.time_base_den)), "source dimensions and time base must be integers")
        _require(self.width > 0 and self.height > 0, "invalid source dimensions")
        _require(self.time_base_num > 0 and self.time_base_den > 0, "invalid time base")
        _require(isinstance(self.pts, frozenset) and bool(self.pts), "decoded PTS registry is required")
        _require(all(type(p) is int for p in self.pts), "PTS must be native integers")

    def time(self, pts: int) -> Fraction:
        _require(type(pts) is int and pts in self.pts, "PTS is absent from decoded source index")
        return Fraction(pts * self.time_base_num, self.time_base_den)


@dataclass(frozen=True)
class Feature:
    feature_id: str
    kind: str  # fixed_junction, material_point, finite_curve
    surface: str  # ice, board
    identity: str  # confirmed, unresolved, rejected
    measurement: str  # canonical: centreline, contour (specified targets only), material_point
    parents: tuple[str, ...] = ()
    branch: str | None = None
    has_station_parameter: bool = False
    label: str | None = None

    def __post_init__(self) -> None:
        _require(bool(self.feature_id), "feature_id is required")
        _require(type(self.parents) is tuple, "parent identities must be an immutable tuple")
        _require(self.kind in {"fixed_junction", "material_point", "finite_curve"}, "unknown feature kind")
        _require(self.surface in {"ice", "board"}, "only identified ice/board entities belong to calibration")
        _require(self.identity in {"confirmed", "unresolved", "rejected"}, "unknown identity status")
        _require(self.measurement in {"centreline", "contour", "material_point"}, "measurement semantics are required")
        from .measurement import required_measurement
        _require(self.measurement == required_measurement(self.label or self.feature_id, self.kind),
                 "only the canonical measurement definition is allowed for this entity")
        if self.kind == "fixed_junction":
            _require(len(self.parents) == 2 and self.parents[0] != self.parents[1], "junction needs two distinct parent entities")
            _require(bool(self.branch), "junction branch must be stable and explicit")
        else:
            _require(not self.parents and self.branch is None, "parents/branch only apply to junctions")
        if self.kind != "finite_curve":
            _require(not self.has_station_parameter, "fixed points cannot acquire per-row stations")


@dataclass(frozen=True)
class NoiseGroup:
    group_id: str
    observation_ids: tuple[str, ...]
    covariance_values_ref: str
    covariance_sha256: str
    model: str  # covariance, square_root_information, explicit_nuisance

    def __post_init__(self) -> None:
        _require(bool(self.group_id) and bool(self.observation_ids), "empty noise group")
        _require(type(self.observation_ids) is tuple, "noise group membership must be immutable")
        _require(len(set(self.observation_ids)) == len(self.observation_ids), "duplicate group members")
        _require(bool(self.covariance_values_ref), "noise group must bind an actual values array")
        _require(bool(re.fullmatch(r"[0-9a-f]{64}", self.covariance_sha256)), "invalid noise array hash")
        _require(self.model in {"covariance", "square_root_information", "explicit_nuisance"}, "unknown correlation representation")


@dataclass(frozen=True)
class Observation:
    observation_id: str
    source_id: str
    source_sha256: str
    pts: int
    uv: tuple[float, float]
    feature_id: str
    measurement: str
    origin: str  # human, automatic, legacy_import, synthetic
    source_evidence: str  # measured, mathematical_only, unavailable
    quality: str  # human_confirmed, machine_verified, pending, rejected
    provenance_id: str
    covariance: tuple[float, float, float] | None = None  # xx, xy, yy in source pixels
    noise_group_id: str | None = None
    active: bool = True

    def __post_init__(self) -> None:
        _require(bool(self.observation_id) and bool(self.provenance_id), "observation identity/provenance is required")
        _require(self.measurement in {"centreline", "contour", "material_point"}, "obsolete measurement definition is forbidden")
        _require(type(self.uv) is tuple, "UV must be an immutable tuple")
        _require(type(self.active) is bool, "active must be a boolean")
        _require(type(self.pts) is int, "PTS must be a native integer")
        _require(len(self.uv) == 2 and _finite(self.uv), "UV must contain two finite source-pixel values")
        _require(self.origin in {"human", "automatic", "legacy_import", "synthetic"}, "unknown observation origin")
        _require(self.source_evidence in {"measured", "mathematical_only", "unavailable"}, "unknown evidence type")
        _require(self.quality in {"human_confirmed", "machine_verified", "pending", "rejected"}, "unknown quality decision")
        _require((self.covariance is None) != (self.noise_group_id is None), "provide per-row covariance or a correlated noise group, exclusively")
        if self.covariance is not None:
            _require(type(self.covariance) is tuple, "covariance must be an immutable tuple")
            _require(len(self.covariance) == 3 and _finite(self.covariance), "invalid covariance")
            xx, xy, yy = self.covariance
            _require(xx > 0 and yy > 0 and xx * yy > xy * xy, "pixel covariance must be positive definite")


def validate_dataset(
    sources: Mapping[str, Source],
    features: Mapping[str, Feature],
    observations: tuple[Observation, ...],
    noise_groups: Mapping[str, NoiseGroup] | None = None,
) -> dict[str, int]:
    """Check present inputs once during preparation; no history replay or I/O.

    Numerical covariance values, ranks and junction existence are checked by
    the problem compiler against actual arrays and the native geometry. This
    function deliberately does not pretend strings alone establish them.
    """
    groups = noise_groups or {}
    for sid, source in sources.items():
        _require(sid == source.source_id, "source registry key mismatch")
    for fid, feature in features.items():
        _require(fid == feature.feature_id, "feature registry key mismatch")
        for parent_id in feature.parents:
            _require(parent_id in features, f"missing junction parent: {parent_id}")
            parent = features[parent_id]
            _require(parent.kind == "finite_curve", "junction parents must be named finite entities")
            _require(parent.surface == feature.surface, "junction and parents must share a surface")
    ids = {row.observation_id for row in observations}
    _require(len(ids) == len(observations), "duplicate observation IDs")
    by_id = {row.observation_id: row for row in observations}
    for gid, group in groups.items():
        _require(gid == group.group_id, "noise group registry key mismatch")
        for oid in group.observation_ids:
            _require(oid in by_id, "correlated group references an absent observation")
            _require(by_id[oid].noise_group_id == gid, "correlation membership is not reciprocal")
    active = 0
    for row in observations:
        _require(row.source_id in sources, f"unknown source: {row.source_id}")
        source = sources[row.source_id]
        _require(row.source_sha256 == source.sha256, "observation belongs to a different source file")
        source.time(row.pts)
        _require(0 <= row.uv[0] < source.width and 0 <= row.uv[1] < source.height, "UV is outside native source image")
        _require(row.feature_id in features, f"unknown feature: {row.feature_id}")
        feature = features[row.feature_id]
        _require(row.measurement == feature.measurement, "measurement semantics changed between observation and entity")
        if row.noise_group_id is not None:
            _require(row.noise_group_id in groups, "noise group is missing")
            _require(row.observation_id in groups[row.noise_group_id].observation_ids, "observation omitted from its noise group")
        if row.active:
            _require(feature.identity == "confirmed", "unresolved/rejected entity cannot enter fitting")
            _require(all(features[p].identity == "confirmed" for p in feature.parents), "active junction has unresolved parents")
            _require(row.origin != "synthetic" and row.source_evidence == "measured", "mathematical or synthetic points are not source measurements")
            _require(row.quality in {"human_confirmed", "machine_verified"}, "pending/rejected measurement cannot enter fitting")
            active += 1
    return {"sources": len(sources), "features": len(features), "observations": len(observations), "active": active, "noise_groups": len(groups)}
