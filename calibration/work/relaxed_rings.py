"""Optional per-ring center offsets on one shared relaxed-marker geometry.

This is a diagnostic geometry wrapper.  It does not change marker offsets,
observations, cameras, circle radii, the outer ring, or any source annotation.
Every source frame projects the same world-space circle geometry.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence

import numpy as np

from relaxed_markers import RelaxedMarkerGeometry, apply_marker_offsets
from shared_corner import SharedCornerGeometry


RING_FEATURES = ("F1", "F2", "F3", "F4", "R1", "R2", "R3", "R4")
RING_ROLES = ("middle", "inner", "center")
RING_OFFSET_LIMIT = 0.15

RING_SPECS: list[dict] = [
    {
        "name": f"ring:{feature_id}:{ring_role}:{component}",
        "feature_id": feature_id,
        "ring_role": ring_role,
        "component": component,
        "lower": -RING_OFFSET_LIMIT,
        "upper": RING_OFFSET_LIMIT,
        "scale": RING_OFFSET_LIMIT,
    }
    for feature_id in RING_FEATURES
    for ring_role in RING_ROLES
    for component in ("dx", "dy")
]
RING_SPEC_BY_NAME = {spec["name"]: spec for spec in RING_SPECS}
assert len(RING_SPECS) == len(RING_SPEC_BY_NAME) == 48


def _normalize(values: Mapping[str, float] | Sequence[float]) -> dict[str, float]:
    if isinstance(values, Mapping):
        unknown = set(values) - RING_SPEC_BY_NAME.keys()
        if unknown:
            raise ValueError(f"Unknown ring offsets: {sorted(unknown)}")
        entries = values.items()
    else:
        if len(values) != len(RING_SPECS):
            raise ValueError(f"Expected {len(RING_SPECS)} ring values, got {len(values)}")
        entries = zip((spec["name"] for spec in RING_SPECS), values)
    result = {}
    for name, raw in entries:
        value = float(raw)
        spec = RING_SPEC_BY_NAME[name]
        if not math.isfinite(value) or not spec["lower"] - 1e-12 <= value <= spec["upper"] + 1e-12:
            raise ValueError(f"Ring offset outside trial bounds: {name}={value}")
        if value != 0.0:
            result[name] = value
    return result


def _by_role(feature: dict) -> dict[str, dict]:
    rings = {ring["role"]: ring for ring in feature["rings"]}
    if set(rings) != {"outer", *RING_ROLES}:
        raise ValueError(f"Expected outer/middle/inner/center rings in {feature['id']}")
    return rings


def ring_nesting_checks(geometry: SharedCornerGeometry) -> dict[str, bool]:
    """Strict containment of every painted disk by its immediate parent."""
    checks = {}
    for feature_id in RING_FEATURES:
        rings = _by_role(geometry.features[feature_id])
        ordered = [rings[role] for role in ("outer", *RING_ROLES)]
        for outer, inner in zip(ordered, ordered[1:]):
            distance = float(np.linalg.norm(np.asarray(outer["center_xyz"], float)[:2]
                                            - np.asarray(inner["center_xyz"], float)[:2]))
            checks[f"{feature_id}:{inner['role']}_inside_{outer['role']}"] = (
                float(inner["radius"]) > 0
                and distance + float(inner["radius"]) < float(outer["radius"]) - 1e-9
            )
    return checks


class RelaxedRingGeometry(RelaxedMarkerGeometry):
    """Copy-on-write extension of the marker wrapper; no camera-specific state."""

    def __init__(self, marker_geometry: RelaxedMarkerGeometry, offsets: dict[str, float]):
        self.__dict__ = marker_geometry.__dict__.copy()
        self._ring_baseline = marker_geometry
        self._ring_offsets = offsets.copy()
        self.features = copy.deepcopy(marker_geometry.features)
        self.parameters = copy.deepcopy(marker_geometry.parameters)
        self.parameters["ring_concentricity_policy"] = {
            "status": "released_small_center_offset_trial",
            "roles": list(RING_ROLES),
            "xy_bounds_scene_units": [-RING_OFFSET_LIMIT, RING_OFFSET_LIMIT],
            "all_frames_share_one_world_geometry": True,
            "source_annotations_unchanged": True,
            "radius_and_each_ring_shape_unchanged": True,
        }
        for feature_id in RING_FEATURES:
            rings = _by_role(self.features[feature_id])
            for role in RING_ROLES:
                ring = rings[role]
                ring["center_xyz"][0] += offsets.get(f"ring:{feature_id}:{role}:dx", 0.0)
                ring["center_xyz"][1] += offsets.get(f"ring:{feature_id}:{role}:dy", 0.0)
        # Invalid nesting is reported by hard_geometry_checks and can be
        # penalized by the caller; the wrapper never silently clamps a value.

    def hard_geometry_checks(self) -> dict:
        result = super().hard_geometry_checks()
        checks = result["checks"]
        checks.pop("circle_rings_remain_concentric_with_their_own_centers")
        baseline_concentric = self._ring_baseline.hard_geometry_checks()["checks"]["circle_rings_remain_concentric_with_their_own_centers"]
        result["released"]["circle_rings_remain_concentric_with_their_own_centers"] = {
            "status": "released_not_enforced", "baseline_value": bool(baseline_concentric),
            "scope": "F1-F4/R1-R4 middle, inner and center rings only; each remains a circle",
        }
        checks["C0_A1_A2_rings_still_concentric"] = all(
            all(np.allclose(ring["center_xyz"], self.features[fid]["center_xyz"], atol=1e-9)
                for ring in self.features[fid].get("rings", []))
            for fid in ("C0", "A1", "A2")
        )
        checks["each_ring_nested_without_crossing"] = all(ring_nesting_checks(self).values())
        checks["ring_centers_remain_on_ice_z0"] = all(
            abs(float(ring["center_xyz"][2])) < 1e-10
            for fid in RING_FEATURES for ring in self.features[fid]["rings"]
        )
        checks["outer_and_feature_centers_unchanged_by_ring_wrapper"] = all(
            np.array_equal(self.features[fid]["center_xyz"], self._ring_baseline.features[fid]["center_xyz"])
            and np.array_equal(_by_role(self.features[fid])["outer"]["center_xyz"],
                               _by_role(self._ring_baseline.features[fid])["outer"]["center_xyz"])
            for fid in RING_FEATURES
        )
        checks["all_ring_radii_unchanged_by_ring_wrapper"] = all(
            float(ring["radius"]) == float(_by_role(self._ring_baseline.features[fid])[ring["role"]]["radius"])
            for fid in RING_FEATURES for ring in self.features[fid]["rings"]
        )
        result["all_constraints_satisfied"] = all(checks.values())
        result["ring_nesting"] = ring_nesting_checks(self)
        result["interpretation"] = (
            "Per-ring concentricity is released, not claimed true. Each ring remains a z=0 circle; "
            "radii, outer centers, feature centers, nesting, marker topology and the common rink boundary are retained."
        )
        return result


def apply_ring_offsets(base_marker_geo: SharedCornerGeometry,
                       values: Mapping[str, float] | Sequence[float]) -> RelaxedRingGeometry:
    """Apply sparse name->value offsets or a complete 48-value vector."""
    if not isinstance(base_marker_geo, SharedCornerGeometry):
        raise TypeError("Expected fitted SharedCornerGeometry or RelaxedMarkerGeometry")
    marker = (base_marker_geo if isinstance(base_marker_geo, RelaxedMarkerGeometry)
              else apply_marker_offsets(base_marker_geo, {}))
    return RelaxedRingGeometry(marker, _normalize(values))
