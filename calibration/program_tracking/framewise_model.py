"""Named physical model targets for a rebuilt, framewise source dataset.

Only the supplied scene, explicit source rows and declared print materials are
used. No strict-v6 adapter, historical observation graph or optimizer is loaded.
Print u runs left to right; v runs top to bottom on the same finite board as
the advertisement's corners. These material coordinates are model parameters,
not independent world measurements or per-point unknown depths.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


# Existing projection modules also expose their original direct-script imports.
PROGRAM = Path(__file__).resolve().parent
ROOT = PROGRAM.parents[1]
for directory in (PROGRAM, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.append(str(directory))

from rinkspace.packed_assets import read_project_bytes

AD_EDGES = {
    "left_vertical": "left_vertical_xyz",
    "right_vertical": "right_vertical_xyz",
    "top_edge": "top_edge_xyz",
    "bottom_edge": "bottom_edge_xyz",
}
AD_CORNERS = frozenset(("left_top", "left_bottom", "right_top", "right_bottom"))


@lru_cache(maxsize=1)
def _geometry_interfaces():
    from calibration.program_tracking.ad_landmarks import AdGeometry
    from calibration.program_tracking.render_result import advertisement_boundary_paths
    return AdGeometry, advertisement_boundary_paths


@lru_cache(maxsize=1)
def _absolute_interface():
    from calibration.program_tracking.z190_source_camera_diagnostic import absolute_value_jac15
    return absolute_value_jac15


def _xyz(value, *, label, minimum=2):
    result = np.asarray(value, dtype=np.float64)
    if (result.ndim != 2 or result.shape[1:] != (3,)
            or len(result) < minimum or not np.isfinite(result).all()):
        raise ValueError(f"Missing or invalid finite world target: {label}")
    return result.copy()


def _ad_number(target):
    number = target.get("advertisement_id")
    if isinstance(number, bool) or not isinstance(number, (int, np.integer)) or number <= 0:
        raise ValueError("A positive explicit advertisement_id is required")
    return int(number)


def _board_edge(scene, target):
    role = target.get("role")
    if role == "top":
        values = scene.get("board_surface", {}).get("top_edge_xyz", [])
    elif role == "contact":
        values = scene.get("ice_boundary", {}).get("points_xyz", [])
    else:
        raise ValueError("Unknown board edge role: " + str(role))
    path = _xyz(values, label="board_edge:" + role)
    if np.array_equal(path[0], path[-1]):
        path = path[:-1]
    part = target.get("part")
    if "association_policy" in target:
        from .continuous_board_association import continuous_board_path
        bottom = _xyz(scene.get("ice_boundary", {}).get("points_xyz", []), label="ice_boundary")
        if np.array_equal(bottom[0], bottom[-1]):
            bottom = bottom[:-1]
        if len(path) != len(bottom):
            raise ValueError("Continuous board top/contact paths must share the same vertex inventory")
        return continuous_board_path(path, scene.get("ice_boundary", {}).get("parts_counterclockwise", []),
            target["association_policy"], original_part=part)
    if part is None:
        return np.vstack((path, path[:1]))
    records = [p for p in scene.get("ice_boundary", {}).get("parts_counterclockwise", [])
               if p.get("id") == part]
    if len(records) != 1:
        raise ValueError("Missing or ambiguous named board part: " + str(part))
    bottom = _xyz(scene["ice_boundary"].get("points_xyz", []), label="ice_boundary")
    if np.array_equal(bottom[0], bottom[-1]):
        bottom = bottom[:-1]
    if len(path) != len(bottom):
        raise ValueError("Named board top/contact paths must share the same vertex inventory")
    first, count = records[0].get("first_vertex"), records[0].get("vertex_count")
    if (not isinstance(first, int) or not isinstance(count, int)
            or not 0 <= first < len(path) or not 0 < count < len(path)):
        raise ValueError("Invalid named board part vertex range")
    # Part vertices start its physical segments; include their final endpoint.
    return path[(first + np.arange(count + 1)) % len(path)].copy()


def feature_world(scene, target_descriptor, templates=None):
    """Produce the exact named target, refusing unknown or missing identities.

    Returns N-by-3 scene-unit coordinates, with explicit closure preserved.
    An ad_corner has N=1. Missing named edges never fall back to a whole ad.
    """
    if not isinstance(target_descriptor, dict):
        raise ValueError("An explicit model target descriptor is required")
    target = target_descriptor
    kind = target.get("kind")
    if kind == "board_edge":
        return _board_edge(scene, target)
    if kind == "ice_curve":
        from rinkspace.vision_programme_ice_review import ice_curve_inventory
        matches = [r for r in ice_curve_inventory(scene) if r["curve_id"] == target.get("curve_id")]
        if len(matches) != 1:
            raise ValueError("Missing or ambiguous named ice curve: " + str(target.get("curve_id")))
        return _xyz(matches[0]["xyz"], label=target["curve_id"])
    if kind not in ("ad_corner", "ad_edge", "ad_print"):
        raise ValueError("Unknown model target kind: " + str(kind))
    number = _ad_number(target)
    if kind == "ad_print" and target.get("geometry_source") == "scene_shield_anchor_material_chart":
        material = (templates or {}).get("advertisements", {}).get(str(number), {})
        record = material.get("contours", {}).get(target.get("contour_key"))
        if not isinstance(record, dict) or int(record.get("advertisement_id", 0)) != number:
            raise ValueError("Missing named shield print material")
        weights = np.asarray(record.get("material_anchor_weights", []), float)
        if weights.ndim != 2 or weights.shape[1] != 3 or len(weights) < 2 or not np.isfinite(weights).all():
            raise ValueError("Invalid finite shield print anchor coordinates")
        shared = scene.get("shared_board_patterns", {})
        anchors = np.asarray(shared.get("instance_material_coordinates_m", {}).get("shield" + str(number), []), float)
        world_anchor = shared.get("shield_fixed_material_points_xyz_scene_units", {}).get(str(number), {})
        roles = ("upper_left_printed_outline_corner", "upper_right_printed_outline_corner", "bottom_printed_outline_tip")
        if anchors.shape != (3, 2) or any(role not in world_anchor for role in roles):
            raise ValueError("Missing existing shield material anchors")
        # Use the same continuous perimeter station/height chart as those
        # three existing anchors. No planar rectangular surface is introduced.
        scale = float(np.median(anchors[:, 1] / np.array([world_anchor[role][2] for role in roles])))
        station_z = weights @ anchors
        station = np.asarray(shared.get("course_station_m", []), float)
        course = np.asarray(shared.get("course_xy_m", []), float)
        if len(station) != len(course) or len(station) < 2 or scale <= 0:
            raise ValueError("Missing continuous shield perimeter chart")
        if (station_z[:, 0] < station[0]).any() or (station_z[:, 0] > station[-1]).any():
            raise ValueError("Shield print outside finite perimeter")
        xy = np.column_stack([np.interp(station_z[:, 0], station, course[:, j]) for j in (0, 1)])
        path = np.column_stack((xy, station_z[:, 1])) / scale
        if record.get("closed") and not np.array_equal(path[0], path[-1]):
            path = np.vstack((path, path[:1]))
        return _xyz(path, label=f"Ad{number}:{target['contour_key']}")
    if kind == "ad_print" and target.get("geometry_source") == "scene_existing_board_pattern":
        key = target.get("contour_key")
        if key == "outermost_visible_circular_print_stroke":
            records = scene.get("additional_board_circle_patterns", {}).get("circles", {})
            values = records.get(str(number), {}).get("circle_outer_stroke_xyz_scene_units")
        elif key == "outer_circle_print_stroke":
            values = scene.get("shared_board_patterns", {}).get("circle_outer_stroke_paths_xyz_scene_units", {}).get(str(number))
        else:
            raise ValueError("Unknown existing board pattern identity")
        return _xyz(values or [], label=f"Ad{number}:{key}")
    if kind == "ad_print" and target.get("geometry_source") == "scene_existing_external_contour":
        advertisements = [a for a in scene.get("advertisements", [])
                          if int(a["user_number"]) == number]
        records = [c for a in advertisements for c in a.get("external_contours", [])
                   if c.get("id") == target.get("contour_key")]
        if len(records) != 1:
            raise ValueError(f"Missing named existing Ad{number} print contour")
        return _xyz(records[0].get("world_xyz_scene_units", []),
                    label=f"Ad{number}:{target['contour_key']}")
    AdGeometry, boundary_paths = _geometry_interfaces()
    geometry = AdGeometry(scene, number)
    if kind == "ad_corner":
        role = target.get("role")
        if role not in AD_CORNERS or role not in geometry.roles:
            raise ValueError(f"Missing or unknown Ad{number} corner: {role}")
        return geometry.corner(geometry.initial, role)[None, :]
    if kind == "ad_edge":
        role = target.get("role")
        if role not in AD_EDGES:
            raise ValueError(f"Unknown Ad{number} edge: {role}")
        matches = [p for p in boundary_paths(scene)
                   if p["number"] == number and p["role"] == AD_EDGES[role]]
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous named Ad{number} edge: {role}")
        if "extent_scope" in target and target["extent_scope"] != matches[0]["extent"]:
            raise ValueError(f"Ad{number} edge material scope differs from the target")
        return _xyz(matches[0]["xyz"], label=f"Ad{number}:{role}")
    key = target.get("contour_key")
    if not isinstance(key, str) or not key:
        raise ValueError("An explicit printed contour_key is required")
    material_template = (templates or {}).get("advertisements", {}).get(str(number), templates or {})
    record = material_template.get("contours", {}).get(key)
    if not isinstance(record, dict):
        raise ValueError(f"Missing named Ad{number} print contour: {key}")
    if "advertisement_id" in record and int(record["advertisement_id"]) != number:
        raise ValueError("Named print contour belongs to another advertisement")
    if ("advertisement_number" in material_template
            and int(material_template["advertisement_number"]) != number):
        raise ValueError("Source print template belongs to another advertisement")
    uv = np.asarray(record.get("material_uv", []), dtype=np.float64)
    extended = record.get("material_coordinate_domain") == "continuous_board_s_h_relative_to_named_internal_graphic"
    if (uv.ndim != 2 or uv.shape[1:] != (2,) or len(uv) < 2
            or not np.isfinite(uv).all() or (not extended and ((uv < 0).any() or (uv > 1).any()))):
        raise ValueError(f"Invalid finite normalized Ad{number} print material: {key}")
    if not isinstance(record.get("closed"), bool):
        raise ValueError("Print contour must explicitly declare closed or open")
    if "left_top" not in geometry.roles or "left_bottom" not in geometry.roles:
        raise ValueError("Print material requires the advertisement's existing top/bottom geometry")
    if geometry._nominal_width is None:
        raise ValueError("Print material requires both existing advertisement sides")
    stations = geometry._base_s["left"] + geometry._width_sign * geometry._nominal_width * uv[:, 0]
    xy = np.array([geometry._point_at_arc(float(station)) for station in stations])
    top = geometry.corner(geometry.initial, "left_top")[2]
    bottom = geometry.corner(geometry.initial, "left_bottom")[2]
    path = np.column_stack((xy, top + uv[:, 1] * (bottom - top)))
    if record["closed"] and not np.array_equal(path[0], path[-1]):
        path = np.vstack((path, path[:1]))
    return path


def build_row(scene, descriptor, uv, sigma, templates=None, *, row_id=None,
              point_correspondence=None, diagnostic_unit_weights=False):
    """Assemble a named row for the original absolute residual/J interface.

    Sigma is explicitly supplied, never chosen or estimated here. Setting
    diagnostic_unit_weights requires sigma=1 and labels this as pixel-only
    entry verification, not a statistical source precision or solver policy.
    """
    world = feature_world(scene, descriptor, templates)
    observed = np.asarray(uv, dtype=np.float64)
    if (observed.ndim != 2 or observed.shape[1:] != (2,)
            or not len(observed) or not np.isfinite(observed).all()):
        raise ValueError("A nonempty finite N-by-2 original source UV array is required")
    point = descriptor["kind"] == "ad_corner"
    if point_correspondence is not None and (not isinstance(point_correspondence, (bool, np.bool_))
                                             or bool(point_correspondence) != point):
        raise ValueError("Corner/curve target and explicit point_correspondence disagree")
    if point and len(observed) != 1:
        raise ValueError("One named physical corner requires exactly one observed source point")
    try:
        scales = np.broadcast_to(np.asarray(sigma, dtype=np.float64), (len(observed),)).copy()
    except ValueError as exc:
        raise ValueError("Sigma must be scalar or one value per source point") from exc
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Positive finite explicitly supplied source scales are required")
    if diagnostic_unit_weights and not np.array_equal(scales, np.ones(len(observed))):
        raise ValueError("Diagnostic pixel entry verification requires explicit sigma=1")
    return dict(id=row_id, target=deepcopy(descriptor), world=world, uv=observed.copy(),
                sigma=scales, point_correspondence=point,
                diagnostic_unit_weights=bool(diagnostic_unit_weights),
                sigma_semantics=("diagnostic_unit_pixel_scale_not_formal_source_weight"
                                 if diagnostic_unit_weights else "explicit_caller_supplied_source_scale"))


def evaluate_rows(camera25, rows, jacobian=True, *, scene=None, templates=None):
    """Evaluate supplied current-frame rows; return original r or (r, J15).

    The camera Jacobian uses the original conditioned 15-column convention.
    Rows normally come from build_row with the current scene. Supplying scene
    explicitly refreshes every world target from that scene before evaluation.
    This operation does not optimize or load an observation graph.
    """
    camera = np.asarray(camera25, dtype=np.float64)
    if camera.shape != (25,) or not np.isfinite(camera).all():
        raise ValueError("The current frame's finite Brown camera25 is required")
    assembled = []
    for row in rows:
        if "point_correspondence" not in row or "target" not in row or "sigma" not in row:
            raise ValueError("Every row requires target, explicit point_correspondence and sigma")
        if scene is not None:
            current = build_row(scene, row["target"], row["uv"], row["sigma"], templates,
                                row_id=row.get("id"), point_correspondence=row["point_correspondence"],
                                diagnostic_unit_weights=row.get("diagnostic_unit_weights", False))
        else:
            if row["target"].get("kind") not in ("board_edge", "ice_curve", "ad_corner", "ad_edge", "ad_print"):
                raise ValueError("Unknown assembled model target kind")
            expected_point = row["target"]["kind"] == "ad_corner"
            if (not isinstance(row["point_correspondence"], (bool, np.bool_))
                    or bool(row["point_correspondence"]) != expected_point):
                raise ValueError("Corner/curve target and explicit point_correspondence disagree")
            world = _xyz(row.get("world", []), label=str(row.get("id")), minimum=1 if expected_point else 2)
            observed = np.asarray(row["uv"], dtype=float)
            if (observed.ndim != 2 or observed.shape[1:] != (2,) or not len(observed)
                    or not np.isfinite(observed).all() or (expected_point and len(observed) != 1)
                    or (expected_point and len(world) != 1)):
                raise ValueError("Invalid assembled source point/corner inventory")
            try:
                scales = np.broadcast_to(np.asarray(row["sigma"], dtype=float), (len(observed),)).copy()
            except ValueError as exc:
                raise ValueError("Sigma must be scalar or one value per source point") from exc
            if not np.isfinite(scales).all() or (scales <= 0).any():
                raise ValueError("Positive finite explicitly supplied source scales are required")
            if row.get("diagnostic_unit_weights") and not np.array_equal(scales, np.ones(len(observed))):
                raise ValueError("Diagnostic pixel entry verification requires explicit sigma=1")
            current = dict(row, world=world, uv=observed.copy(), sigma=scales)
        for field in ("normal", "normal_interval", "point_support_bounds_xy_px"):
            if field in row:
                current[field] = np.asarray(row[field], dtype=np.float64).copy()
        assembled.append(current)
    if not assembled:
        empty = np.empty(0, dtype=np.float64)
        return (empty, np.empty((0, 15))) if jacobian else empty
    return _absolute_interface()(camera, assembled, jacobian)


def load_bound_frame(model_path, *, diagnostic_unit_weights=False):
    """Read a hash-bound model/dataset directly into the existing residual API.

    This convenience loader is explicitly diagnostic. Formal fitting must
    supply source uncertainty and shared-evidence weighting through build_row;
    creating model entries does not silently choose that statistical policy.
    """
    if not diagnostic_unit_weights:
        raise ValueError('Explicit diagnostic_unit_weights=True required for entry verification')

    def read_reference(record):
        payload=read_project_bytes(record['path'], project_root=ROOT)
        if hashlib.sha256(payload).hexdigest()!=record['sha256']:
            raise ValueError('Model/source reference hash mismatch: '+record['path'])
        return json.loads(payload)

    manifest=json.loads(Path(model_path).read_text(encoding='utf8'))
    if manifest.get('schema')!='rinkspace.framewise-reconstruction-model.v1':
        raise ValueError('An explicit framewise reconstruction model is required')
    scene=read_reference(manifest['scene'])
    registry=read_reference(manifest['feature_registry'])
    templates=read_reference(manifest['material_templates'])
    binding=read_reference(manifest['programme_binding'])
    camera=read_reference(manifest['camera_seed'])
    packet=read_reference(binding['source_packet'])
    if not packet['user_accepted']:
        raise ValueError('The bound source frame has not been accepted')
    if (packet['source']['native_pts']!=binding['source_pts']
            or camera['native_pts']!=binding['source_pts']
            or packet['source']['time_base']!=binding['source_time_base']):
        raise ValueError('Bound source/camera clocks disagree')
    model_id=manifest['model_id']
    if any(item.get('model_id')!=model_id for item in (scene,registry,binding)):
        raise ValueError('Frame model identities disagree')
    features={f['id']:f for f in packet['features']}
    expected={f'{f["id"]}/segment/{si}/vertex/{vi}' for f in packet['features']
              for si,s in enumerate(f['segments']) for vi in range(len(s['uv']))}
    seen=[]; rows=[]
    for observation in binding['observations']:
        fid=observation['source_feature_id'];si=observation['source_segment_index']
        indices=observation['source_vertex_indices']
        ids=[f'{fid}/segment/{si}/vertex/{i}' for i in indices]
        if ids!=observation['source_point_ids']:
            raise ValueError('Point identity differs from its source coordinate index')
        seen.extend(ids)
        uv=np.asarray(features[fid]['segments'][si]['uv'],float)[indices]
        descriptor=registry['targets'][observation['target_id']]
        row=build_row(scene,descriptor,uv,np.ones(len(uv)),templates,
            row_id=observation['id'],point_correspondence=observation['point_correspondence'],
            diagnostic_unit_weights=True)
        row['source_point_ids']=ids
        row['dependence_group']=observation['dependence_group']
        rows.append(row)
    if set(seen)!=expected or len(seen)!=len(expected) or len(seen)!=packet['point_count']:
        raise ValueError('The binding omits or duplicates accepted source points')
    camera25=np.asarray(camera['camera25'],np.float64)
    if hashlib.sha256(camera25.tobytes()).hexdigest()!=camera['camera25_float64_sha256']:
        raise ValueError('Bound camera25 bytes differ from the inherited camera')
    return dict(manifest=manifest,scene=scene,templates=templates,source_packet=packet,
                binding=binding,camera25=camera25,rows=rows)
