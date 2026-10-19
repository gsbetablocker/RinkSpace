"""Traceable pixel observations from an unaltered user contour layer.

The layer supplies finite drawing support, not world coordinates. A drawn rail
must remain a rail observation; this module never moves it onto the ice plane.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import hashlib
import math
from pathlib import Path

from PIL import Image


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _components(pixels):
    """8-connected support, including every nonzero antialiased alpha pixel."""
    remaining = set(pixels)
    result = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        component, todo = [seed], [seed]
        while todo:
            x, y = todo.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    point = (x + dx, y + dy)
                    if point in remaining:
                        remaining.remove(point)
                        component.append(point)
                        todo.append(point)
        result.append(component)
    return sorted(result, key=lambda group: min(group))


def extract_user_contours(layer_path, source_frame_path, *, source_pts,
                          time_base, component_ids=None, physical_kind=None,
                          expected_layer_sha256=None, expected_frame_sha256=None):
    """Extract alpha-weighted centres and exact finite per-column support.

    Image coordinates use integer pixel centres (x right, y down). A separate
    component becomes a separate observation group. No smoothing, hole filling,
    rescaling, model projection or detector-created observations are applied.
    Contours that fold across a column must be split before using this API.
    """
    layer_path, source_frame_path = Path(layer_path), Path(source_frame_path)
    layer_hash, frame_hash = file_sha256(layer_path), file_sha256(source_frame_path)
    if expected_layer_sha256 and expected_layer_sha256 != layer_hash:
        raise ValueError("Original user layer hash does not match provenance")
    if expected_frame_sha256 and expected_frame_sha256 != frame_hash:
        raise ValueError("Original source frame hash does not match provenance")
    if isinstance(source_pts, bool) or not isinstance(source_pts, int):
        raise ValueError("Source PTS must be an integer in the declared time base")
    clock = Fraction(time_base)
    if clock <= 0:
        raise ValueError("Source time base must be positive")
    with Image.open(layer_path) as layer, Image.open(source_frame_path) as frame:
        if layer.mode != "RGBA":
            raise ValueError("An original RGBA layer with explicit alpha is required")
        if layer.size != frame.size:
            raise ValueError("Layer and source must have identical original dimensions")
        size = list(layer.size)
        alpha = layer.getchannel("A")
        bbox = alpha.getbbox()
        if bbox is None:
            raise ValueError("User layer has no nonzero-alpha contour support")
        data = alpha.load()
        pixels = {(x, y): int(data[x, y])
                  for x in range(bbox[0], bbox[2])
                  for y in range(bbox[1], bbox[3]) if data[x, y] > 0}
    groups = _components(pixels)
    if component_ids is not None and len(component_ids) != len(groups):
        raise ValueError("Component identities must match actual disconnected strokes")
    if component_ids is not None and len(set(component_ids)) != len(component_ids):
        raise ValueError("Contour identities must be unique")
    contours = []
    for index, group in enumerate(groups):
        columns = defaultdict(list)
        for x, y in group:
            columns[x].append(y)
        centres, supports = [], []
        for x, values in sorted(columns.items()):
            values.sort()
            if any(b != a + 1 for a, b in zip(values, values[1:])):
                raise ValueError("A contour folds across x; split it into monotonic observations")
            weight = sum(pixels[x, y] for y in values)
            centre = sum(y * pixels[x, y] for y in values) / weight
            centres.append([x, centre])
            # Support is raster cell extent, not a claimed confidence interval.
            supports.append([x, values[0] - .5, values[-1] + .5, weight])
        identity = component_ids[index] if component_ids else f"user_contour_{index + 1}"
        contours.append({
            "id": identity,
            "physical_kind": physical_kind,
            "observation_group_id": identity,
            "status": "user_traced_image_contour_not_world_measurement",
            "bbox_xyxy_exclusive": [min(x for x, y in group), min(y for x, y in group),
                                    max(x for x, y in group) + 1, max(y for x, y in group) + 1],
            "nonzero_alpha_pixel_count": len(group),
            "observed_centres_source_px": centres,
            "column_support_x_ymin_ymax_alpha_sum": supports,
            "support_width_definition": "full nonzero-alpha raster extent; pixel cells +/-0.5",
            "uncertainty": {
                "drawing_support_is_not_statistical_sigma": True,
                "manual_trace_error_source_px": None,
                "source_edge_blur_error_source_px": None,
                "independent_samples": False,
                "correlation_scope": "one manually drawn connected stroke",
                "world_position_or_height_inferred": False,
            },
        })
    return {
        "schema_version": "rinkspace.user-contour-observations.v1",
        "source": {
            "source_frame": str(source_frame_path.resolve()),
            "source_frame_sha256": frame_hash,
            "user_layer": str(layer_path.resolve()),
            "user_layer_sha256": layer_hash,
            "source_pts": source_pts,
            "time_base": str(clock),
            "source_time_seconds": float(source_pts * clock),
            "image_size_wh": size,
        },
        "image_domain": "original_source_pixels_unmodified",
        "coordinate_definition": "integer pixel centres; x right, y down",
        "H_source_from_user_layer": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        "extraction": {
            "method": "all nonzero alpha; 8-connected strokes; alpha-weighted y centre per original x column",
            "alpha_threshold_exclusive": 0,
            "resize_or_warp": False,
            "smoothing_or_interpolation": False,
            "model_projection_used": False,
            "pixels_outside_original_alpha_added": False,
            "source_or_layer_modified": False,
        },
        "contours": contours,
        "interpretation": "finite user drawing support, with unquantified tracing and source blur uncertainty; no exact physical edge or Z=0 assumption",
    }


def observed_house_connections(scene, *, pairs=None, authority=None,
                               right_extension_fraction=108. / 945.):
    """Return the four user-confirmed longitudinal Lane centre lines.

    The latest user drawing explicitly continues beyond the R house. Its
    approximate relative extension supplies an initial extent, never a video
    pixel observation or a regulation dimension. Current estimated extents
    come from the common scene producer, independently for each line. Keep
    them independent of the rectangular Lane end and use current houses.
    A transverse terminal cap still requires source evidence of physical paint.
    """
    if pairs is None or authority is None:
        raise ValueError("Explicit observed pairs and their authority are required")
    if (not math.isfinite(float(right_extension_fraction))
            or float(right_extension_fraction) < 0.):
        raise ValueError("A finite nonnegative declared centre-line extent initializer is required")
    current_extents = scene.get("independent_lane_right_extent_fractions", {})
    if not isinstance(current_extents, dict):
        raise ValueError("Current independent Lane extents must be an identity-keyed dictionary")
    features = {feature["id"]: feature for feature in scene["ice_features"]}
    result = []
    for left_id, right_id in pairs:
        connection_id = f"house_connection_{left_id}_{right_id}"
        estimated = connection_id in current_extents
        fraction = float(current_extents[connection_id] if estimated else right_extension_fraction)
        if not math.isfinite(fraction) or fraction < 0.:
            raise ValueError("Finite nonnegative current Lane extent required: " + connection_id)
        ends = []
        for identity in (left_id, right_id):
            feature = features[identity]
            point = feature["center_xyz"]
            if (feature.get("kind") != "circle" or len(point) != 3
                    or not all(math.isfinite(float(v)) for v in point)
                    or float(point[2]) != 0.):
                raise ValueError("Confirmed house connection needs finite ice-plane circle centres")
            ends.append([float(v) for v in point])
        original_centres = [point.copy() for point in ends]
        ends[1] = [r + fraction * (r - l)
                   for l, r in zip(*original_centres)]
        result.append({
            "id": connection_id,
            "kind": "polyline",
            "surface_id": "ice_surface",
            "label": f"{left_id}–{right_id}",
            "segments_xyz": [ends],
            "physical_kind": "user_confirmed_longitudinal_lane_centreline",
            "house_ids": [left_id, right_id],
            "passes_through_house_centres_xyz": original_centres,
            "endpoint_scope": "F centre to estimated rightward continuation beyond R centre",
            "right_extent_initializer": {
                "fraction_of_current_FR_separation": float(right_extension_fraction),
                "role": "approximate_user_drawing_initializer_not_measured_video_endpoint",
                "user_drawing": "D:/CodexData/RinkSpace_Data/events/ankara_2026/evidence/user_contours_20261004/user_lane_centrelines_original_20261005.png",
                "user_drawing_sha256": "910f6862f1256a8171a712b6770e18632b0545536eb341cb2f28c24dd566d487",
                "actual_video_terminal_endpoint_identified": False,
                "tied_to_Lane_rectangular_end": False,
                "tied_to_standard_centreline_extension": False,
                "physical_transverse_cap_added": False,
                "used_as_source_observation_or_fit_prior": False,
            },
            "right_extent_current": {
                "fraction_of_current_FR_separation": fraction,
                "role": ("current_independent_geometry_parameter" if estimated
                         else "historical_initializer_compatibility_not_source_verified"),
                "common_scene_field": "independent_lane_right_extent_fractions",
                "independent_parameter_supplied": estimated,
                "other_Lane_extents_tied": False,
                "actual_video_terminal_endpoint_identified": False,
                "used_as_source_observation_or_fit_prior": False,
            },
            "geometry_source": "current scene house centres; user-confirmed real line identity",
            "authority": authority,
            "standard_template_used": False,
            "forces_geometry": False,
        })
    return result


def apply_user_visible_markings(scene, *, pairs, authority):
    """Build the venue inventory from observed paint and confirmed connections.

    The inherited curling standard inventory explicitly has zero source UV.
    Keep it as historical reference data, outside modeled venue paint. Never
    omit a source-backed or unqualified feature, and never condition this
    inventory on a frame, camera, detection score or projection error.
    """
    template_ids = {
        f"curling_sheet{row}_{part}"
        for row in range(1, 5)
        for part in (
            *(f"{mark}_{side}" for mark in (
                "tee_center", "hog_inside", "back_outside",
                "sheet_end_reference", "hack_reference") for side in ("F", "R")),
            "centerline_reference", "sideline_reference_lower", "sideline_reference_upper",
        )
    }
    pair_ids = {f"house_connection_{left}_{right}" for left, right in pairs}
    kept, references, removed = [], {}, []
    for feature in scene.get("unobserved_standard_reference_ice_features", []):
        references[feature["id"]] = feature
    for feature in scene["ice_features"]:
        identity = feature["id"]
        if identity.startswith("curling_sheet"):
            source = feature.get("source", {})
            if (identity not in template_ids
                    or feature.get("geometry_status") != "unobserved_coupled_expected_template"
                    or source.get("UV_count") != 0
                    or source.get("new_observation") is not False):
                raise ValueError("Cannot remove a source-backed or unqualified curling mark: " + identity)
            references[identity] = feature
            removed.append(identity)
        elif identity not in pair_ids:
            kept.append(feature)
    result = dict(scene)
    result["ice_features"] = kept + observed_house_connections(
        scene, pairs=pairs, authority=authority)
    result["unobserved_standard_reference_ice_features"] = list(references.values())
    result["venue_paint_inventory_qualification"] = {
        "schema": "rinkspace.user-confirmed-venue-paint-inventory.v1",
        "excluded_unobserved_standard_template_ids": sorted(references),
        "removed_from_current_venue_inventory_ids": removed,
        "confirmed_house_connection_ids": sorted(pair_ids),
        "authority": authority,
        "original_observed_features_retained": True,
        "standard_references_are_not_modeled_venue_paint": True,
        "no_confidence_or_per_frame_visibility_filter": True,
        "other_geometry_and_source_observations_unchanged": True,
    }
    return result
