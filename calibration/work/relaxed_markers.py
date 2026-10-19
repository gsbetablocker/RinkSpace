"""Small shared-scene marker offsets without changing source annotations.

All offsets are scene-space physical geometry, shared by every camera.  The
baseline geometry, board/corner profile, feature sizes and marker source fields
are left untouched.  Missing entries in ``values`` mean exactly zero.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence

import numpy as np

from shared_corner import SharedCornerGeometry


CIRCLES = ("F1", "F2", "F3", "F4", "R1", "R2", "R3", "R4", "C0", "A1", "A2")
LANES = ("lane1", "lane2", "lane3", "lane4")
TRANSVERSE_LINES = ("C", "B1", "B2", "goal_line_left", "goal_line_right")
CREASES = ("goal_crease_left", "goal_crease_right")
XY_LIMIT = 0.6
ANGLE_LIMIT = 0.02
PRESERVED_OLD_CHECKS = {
    "eight_equal_outer_radii", "A1_A2_equal_outer_radii",
    "four_equal_corner_radii", "corner_radius_parameter_consistent",
    "B1_B2_nonzero_length", "goal_line_left_goal_line_right_nonzero_length",
    *(f"{fid}_left_open_exactly_three_edges" for fid in LANES),
}


def _spec(feature_id: str, component: str, limit: float) -> dict:
    return {
        "name": f"marker:{feature_id}:{component}",
        "feature_id": feature_id,
        "component": component,
        "lower": -limit,
        "upper": limit,
        "scale": limit,
    }


MARKER_SPECS: list[dict] = []
for _feature in CIRCLES:
    MARKER_SPECS.extend((_spec(_feature, "dx", XY_LIMIT), _spec(_feature, "dy", XY_LIMIT)))
for _feature in LANES:
    for _vertex in range(4):
        MARKER_SPECS.extend((_spec(_feature, f"v{_vertex}_dx", XY_LIMIT),
                             _spec(_feature, f"v{_vertex}_dy", XY_LIMIT)))
for _feature in TRANSVERSE_LINES:
    MARKER_SPECS.extend((_spec(_feature, "upper_dx", XY_LIMIT),
                         _spec(_feature, "lower_dx", XY_LIMIT)))
for _feature in CREASES:
    MARKER_SPECS.extend((_spec(_feature, "dx", XY_LIMIT),
                         _spec(_feature, "dy", XY_LIMIT),
                         _spec(_feature, "angle", ANGLE_LIMIT)))

SPEC_BY_NAME = {spec["name"]: spec for spec in MARKER_SPECS}
assert len(MARKER_SPECS) == len(SPEC_BY_NAME) == 70


def _normalize(values: Mapping[str, float] | Sequence[float]) -> dict[str, float]:
    if isinstance(values, Mapping):
        unknown = set(values) - SPEC_BY_NAME.keys()
        if unknown:
            raise ValueError(f"Unknown marker offsets: {sorted(unknown)}")
        source = values.items()
    else:
        if len(values) != len(MARKER_SPECS):
            raise ValueError(f"Expected {len(MARKER_SPECS)} marker values, got {len(values)}")
        source = zip((s["name"] for s in MARKER_SPECS), values)
    result = {}
    for name, raw in source:
        value = float(raw)
        spec = SPEC_BY_NAME[name]
        if not math.isfinite(value) or not spec["lower"] - 1e-12 <= value <= spec["upper"] + 1e-12:
            raise ValueError(f"Marker offset outside trial bounds: {name}={value}")
        if value != 0.0:
            result[name] = value
    return result


def _inside_rounded(geometry: SharedCornerGeometry, xy: np.ndarray) -> bool:
    x0, y0, x1, y1 = map(float, geometry.parameters["bounds"])
    x, y = map(float, xy)
    if not x0 - 1e-10 <= x <= x1 + 1e-10 or not y0 - 1e-10 <= y <= y1 + 1e-10:
        return False
    dx = max(x0 + geometry.rx - x, 0.0, x - (x1 - geometry.rx)) / geometry.rx
    dy = max(y0 + geometry.ry - y, 0.0, y - (y1 - geometry.ry)) / geometry.ry
    return dx**geometry.exponent + dy**geometry.exponent <= 1.0 + 1e-12


def _clip_supporting_line(geometry: SharedCornerGeometry, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Clip an infinite non-axis-aligned line to the exact superellipse rink."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if not np.allclose([a[2], b[2]], 0.0, atol=1e-12):
        raise ValueError("Marker line must remain on ice z=0")
    direction = b[:2] - a[:2]
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        raise ValueError("Degenerate marker line")
    direction /= length
    midpoint = (a[:2] + b[:2]) / 2
    if not _inside_rounded(geometry, midpoint):
        raise ValueError("Marker supporting line does not cross ice near its original location")
    x0, y0, x1, y1 = map(float, geometry.parameters["bounds"])
    reach = 2 * math.hypot(x1 - x0, y1 - y0)
    def distance_to_edge(sign: float) -> float:
        low, high = 0.0, reach
        if _inside_rounded(geometry, midpoint + sign * high * direction):
            raise ValueError("Marker line clipping search did not leave rink")
        for _ in range(64):
            middle = (low + high) / 2
            if _inside_rounded(geometry, midpoint + sign * middle * direction):
                low = middle
            else:
                high = middle
        return low
    start = midpoint - distance_to_edge(-1.0) * direction
    end = midpoint + distance_to_edge(1.0) * direction
    return np.r_[start, 0.0], np.r_[end, 0.0]


class RelaxedMarkerGeometry(SharedCornerGeometry):
    """Copy-on-write wrapper around a fitted, unchanged shared rink geometry."""

    def __init__(self, base: SharedCornerGeometry, offsets: dict[str, float]):
        if not isinstance(base, SharedCornerGeometry):
            raise TypeError("Expected current SharedCornerGeometry")
        self.__dict__ = base.__dict__.copy()
        self._baseline = base
        self._offsets = offsets.copy()
        self._changed_features = {SPEC_BY_NAME[name]["feature_id"] for name in offsets}
        self.features = copy.deepcopy(base.features)
        self.parameters = copy.deepcopy(base.parameters)
        self.parameters["marker_association_policy"] = {
            "status": "released_small_offset_trial",
            "world_xy_bounds_scene_units": [-XY_LIMIT, XY_LIMIT],
            "crease_rotation_bounds_radians": [-ANGLE_LIMIT, ANGLE_LIMIT],
            "same_physical_markers_across_all_frames": True,
            "source_annotations_unchanged": True,
            "offsets_are_trial_parameter_bounds_not_user_measured_distances": True,
        }
        self._apply()

    def offset(self, feature_id: str, component: str) -> float:
        return self._offsets.get(f"marker:{feature_id}:{component}", 0.0)

    def _apply(self) -> None:
        for fid in CIRCLES:
            dx, dy = self.offset(fid, "dx"), self.offset(fid, "dy")
            if dx == dy == 0.0:
                continue
            feature = self.features[fid]
            feature["center_xyz"][0] += dx
            feature["center_xyz"][1] += dy
            for ring in feature.get("rings", []):
                ring["center_xyz"][0] += dx
                ring["center_xyz"][1] += dy
            feature["marker_offset_status"] = "shared_physical_center_translation; all rings remain concentric"

        for fid in LANES:
            feature = self.features[fid]
            if fid not in self._changed_features:
                continue
            polygon = np.asarray(feature["polygon_xyz"], float)
            if polygon.shape != (4, 3) or feature.get("open_edges") != [3]:
                raise ValueError(f"Unexpected lane topology: {fid}")
            for index in range(4):
                polygon[index, 0] += self.offset(fid, f"v{index}_dx")
                polygon[index, 1] += self.offset(fid, f"v{index}_dy")
            feature["polygon_xyz"] = polygon.tolist()
            feature["centerline_xyz"] = [((polygon[0] + polygon[3]) / 2).tolist(),
                                           ((polygon[1] + polygon[2]) / 2).tolist()]
            feature["left_endpoint_status"] = "unresolved"
            feature["polygon_left_vertices_role"] = "unmeasured_display_proxy_not_physical_endpoints"
            feature["left_endpoint_coordinate_roles"] = {"upper": "unmeasured_display_proxy",
                                                         "lower": "unmeasured_display_proxy"}
            feature["association_fields_status"] = "old shared width, row alignment and right-edge coordinates are historical references only"

        for fid in TRANSVERSE_LINES:
            upper_shift, lower_shift = self.offset(fid, "upper_dx"), self.offset(fid, "lower_dx")
            if upper_shift == lower_shift == 0.0:
                continue
            feature = self.features[fid]
            segment = np.asarray(feature["segments_xyz"][0], float)
            if segment.shape != (2, 3):
                raise ValueError(f"Expected one painted transverse segment for {fid}")
            upper, lower = segment.copy()
            if upper[1] < lower[1]:
                upper, lower = lower, upper
            upper[0] += upper_shift
            lower[0] += lower_shift
            clipped_upper, clipped_lower = _clip_supporting_line(self, upper, lower)
            feature["segments_xyz"] = [[clipped_upper.tolist(), clipped_lower.tolist()]]
            feature["boundary_rule"] = "Tilted physical line extended and clipped to the fitted superellipse ice boundary"
            feature["marker_offset_status"] = "upper/lower X free; no mirrored or parallel line relation enforced"

        for fid in CREASES:
            dx, dy, angle = self.offset(fid, "dx"), self.offset(fid, "dy"), self.offset(fid, "angle")
            if dx == dy == angle == 0.0:
                continue
            feature = self.features[fid]
            polygon = np.asarray(feature["polygon_xyz"], float)
            center = polygon[:, :2].mean(axis=0)
            cosine, sine = math.cos(angle), math.sin(angle)
            rotation = np.array([[cosine, -sine], [sine, cosine]])
            polygon[:, :2] = (polygon[:, :2] - center) @ rotation.T + center + np.array([dx, dy])
            feature["polygon_xyz"] = polygon.tolist()
            feature["marker_offset_status"] = "rigid_xy_pose_only; shape and closed topology preserved"

    def points(self, target: dict, observation_type: str, samples: int) -> np.ndarray:
        if target["kind"] != "polygon_supporting_line":
            return super().points(target, observation_type, samples)
        fid = target.get("feature_id")
        if fid not in LANES or observation_type != "curve":
            raise ValueError("polygon_supporting_line requires a lane curve")
        polygon = np.asarray(self.features[fid]["polygon_xyz"], float)
        index = int(target["index"])
        if not 0 <= index < len(polygon) or index in self.features[fid].get("open_edges", []):
            raise ValueError("Cannot use missing or open lane edge as a supporting line")
        start, end = _clip_supporting_line(self, polygon[index], polygon[(index + 1) % len(polygon)])
        # Only the infinite painted-line location is fitted. In particular,
        # lane left display proxies are not asserted to be physical endpoints.
        return np.linspace(start, end, max(8, samples))

    def hard_geometry_checks(self) -> dict:
        baseline = self._baseline.hard_geometry_checks()
        original = baseline.get("checks", {}) if baseline else {}
        released = {name: {"status": "released_not_enforced", "baseline_value": bool(value)}
                    for name, value in original.items() if name not in PRESERVED_OLD_CHECKS}
        circle_centers = all(
            all(np.allclose(ring["center_xyz"], self.features[fid]["center_xyz"], atol=1e-9)
                for ring in self.features[fid].get("rings", []))
            for fid in CIRCLES
        )
        circle_z = all(abs(float(self.features[fid]["center_xyz"][2])) < 1e-10 for fid in CIRCLES)
        circle_shape = all(all(float(ring["radius"]) > 0 for ring in self.features[fid].get("rings", []))
                           for fid in CIRCLES)
        outer_radii = [float(next(r["radius"] for r in self.features[fid]["rings"] if r["role"] == "outer"))
                       for fid in ("F1", "F2", "F3", "F4", "R1", "R2", "R3", "R4")]
        lane_topology = all(self.features[fid].get("open_edges") == [3]
                            and np.allclose(np.asarray(self.features[fid]["polygon_xyz"])[:, 2], 0)
                            and self.features[fid].get("left_endpoint_status") == "unresolved"
                            and all(_inside_rounded(self, np.asarray(point)[:2])
                                    for point in self.features[fid]["polygon_xyz"])
                            for fid in LANES)
        line_ice = all(np.allclose(np.asarray(self.features[fid]["segments_xyz"])[:, :, 2], 0)
                       and all(_inside_rounded(self, np.asarray(point)[:2])
                               for segment in self.features[fid]["segments_xyz"] for point in segment)
                       for fid in TRANSVERSE_LINES)
        crease_topology = all(self.features[fid].get("closed") is True
                              and np.allclose(np.asarray(self.features[fid]["polygon_xyz"])[:, 2], 0)
                              and all(_inside_rounded(self, np.asarray(point)[:2])
                                      for point in self.features[fid]["polygon_xyz"])
                              for fid in CREASES)
        retained = {
            "circle_rings_remain_concentric_with_their_own_centers": circle_centers,
            "circle_radii_remain_positive": circle_shape,
            "circle_centers_remain_on_ice_z0": circle_z,
            "eight_F_R_outer_radii_remain_equal": bool(np.ptp(outer_radii) < 1e-8),
            "A1_A2_outer_radii_remain_equal": abs(self.features["A1"]["radius"] - self.features["A2"]["radius"]) < 1e-8,
            "four_corners_keep_one_symmetric_profile": bool(baseline.get("corner_profile", {}).get("four_identical")
                                                       and baseline.get("corner_profile", {}).get("reflection_symmetric")),
            "lanes_remain_three_sided_and_left_open_with_unresolved_endpoints": lane_topology,
            "transverse_lines_remain_on_ice_and_inside_boundary": line_ice,
            "creases_remain_closed_shapes_on_ice": crease_topology,
        }
        return {
            "profile": "relaxed_marker_associations_trial",
            "all_constraints_satisfied": all(retained.values()),
            "checks": retained,
            "released": released,
            "corner_profile": baseline.get("corner_profile"),
            "interpretation": "Only retained shape, ice-plane, outer-radius and topology checks are enforced. Former marker alignment, spacing, symmetry, attachment and lane endpoint relations are released, not claimed true.",
        }


def apply_marker_offsets(base_geo: SharedCornerGeometry, values: Mapping[str, float] | Sequence[float]) -> RelaxedMarkerGeometry:
    """Return one modified shared geometry; partial name->value mappings are valid."""
    return RelaxedMarkerGeometry(base_geo, _normalize(values))
