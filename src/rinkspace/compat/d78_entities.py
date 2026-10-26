"""One-time complete d78 entity import, then data-only native evaluation.

The frozen scene is kept separate from the shared-structure candidate. Import
composes old chart witnesses once; it never imports a historical evaluator.
Representation vertices are not asserted to be observed landmarks.
"""
from __future__ import annotations
from copy import deepcopy
import math
import numpy as np


def decode_entities(model: dict) -> dict:
    blocks = [np.asarray(b["values"], dtype=np.float64) for b in model["blocks"]]
    if not blocks or blocks[0].shape != (32,) or any(not np.isfinite(b).all() for b in blocks):
        raise ValueError("d78 entity import requires finite complete parameter blocks")
    g = blocks[0]
    materials, paths = [], []
    for mi, material in enumerate(model["materials"]):
        material_id = f"d78_material_{mi:03d}"
        path_ids = []
        for bi, old in enumerate(material["paths"]):
            kind = old["kind"]
            path_id = f"{material_id}/path{bi}"
            path_ids.append(path_id)
            q = blocks[old["geometry_block"]] if old.get("geometry_block", -1) >= 0 else None
            p = {"id": path_id, "entity_id": material_id, "material_index": mi,
                 "branch_index": bi, "label": material["label"], "original_path_id": old["id"],
                 "original_variant_id": old["variant_id"], "closed": bool(old.get("closed", False)),
                 "support": [0., 1.], "historical_physical_point_flag": bool(old.get("physical_point", False)),
                 "vertices_are_observations": False, "original_definition": deepcopy(old),
                 "layer": "ice" if kind in (0, 1) else "board" if kind == 3 else "advertisement"}
            if kind == 0:
                if q is None or q.shape != (4,):
                    raise ValueError(f"malformed segment block at {path_id}")
                p.update(kind="ice_segment", parameters=[float(q[0]), float(q[1]), 0., float(q[2]), float(q[3]), 0.])
            elif kind == 1:
                index = old.get("radius_index", 2)
                if q is None or len(q) <= max(1, index):
                    raise ValueError(f"malformed circle block at {path_id}")
                p.update(kind="ice_circle", parameters=[float(q[0]), float(q[1]), 0., float(q[index]),
                         float(old.get("start_angle", 0.)), float(old.get("end_angle", 2*math.pi))])
            elif kind == 2:
                points = np.asarray(old["points"], dtype=np.float64)
                if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
                    raise ValueError(f"malformed board contour at {path_id}")
                if q is not None:
                    if q.shape != (6,):
                        raise ValueError(f"malformed affine chart at {path_id}")
                    points = np.column_stack((points, np.ones(len(points)))) @ q.reshape(2, 3).T
                offset = old.get("advertisement_offset", 0)
                if type(offset) is not int or offset < 0:
                    raise ValueError(f"invalid advertisement offset at {path_id}")
                ad = blocks[old["advertisement_block"]][offset:offset+4]
                if ad.shape != (4,) or ad[1] <= 0 or ad[3] <= ad[2]:
                    raise ValueError(f"malformed installation block at {path_id}")
                points = points * [ad[1], ad[3]-ad[2]] + [ad[0], ad[2]]
                p.update(kind="board_material_polyline", parameters=[], vertices=points.tolist(),
                         station_parameterization="linear_original_vertex_index", advertisement_offset=offset,
                         advertisement_block=old["advertisement_block"], installation_parameters=ad.tolist())
            elif kind == 3:
                p.update(kind="board_boundary", parameters=[int(old.get("boundary_segment", -1)), float(old.get("fixed_height_fraction", 0.))])
            else:
                raise ValueError(f"unsupported entity primitive {kind!r} at {path_id}")
            paths.append(p)
        materials.append({"id": material_id, "label": material["label"], "path_ids": path_ids,
                          "original_material_metadata": {k: deepcopy(v) for k, v in material.items() if k != "paths"}})
    if len(materials) != 171 or len(paths) != 258:
        raise ValueError(f"expected full d78 inventory 171/258, got {len(materials)}/{len(paths)}")
    return {"schema": "rinkspace.scene.replay.v1", "scene_id": "ankara_2026.d78_frozen", "revision": 1,
            "status": "frozen_projection_replay", "frame": {"frame_id": "d78_L1", "length_unit": "rink_length", "meters_per_unit": None},
            "board": [.5, float(g[0])/2, float(g[1]), float(g[2]), float(g[3]), float(g[4])],
            "materials": materials, "paths": paths, "missing_entities": [],
            "semantic_note": "Frozen representation only. Existing confirmed observations retain their own authority; vertices create none.",
            "fit_executed": False, "user_approval": "not_reviewed"}


KINDS = {"ice_segment": 0, "ice_circle": 1, "board_material_polyline": 2, "board_boundary": 3,
         "xyz_polyline": 4, "installed_artwork_polyline": 5, "ice_ellipse": 6, "board_course": 7}
KIND_NAMES = {value: name for name, value in KINDS.items()}


def native_definition(scene, path, core):
    """One dispatch for shared constraints and both current/explicit replay paths."""
    if "generated_code" in path:
        if (scene.get('schema') in ('rinkspace.field_model.v1', 'rinkspace.curling_lane.v1')
                and 'shared_geometry_options' not in scene):
            raise ValueError('Current shared scene is missing its geometry options')
        return core.shared_geometry(scene["board"], scene["shared_layout"], path["generated_code"],
                                    options=scene.get("shared_geometry_options", 0))
    kind = path.get("native_kind", KINDS.get(path.get("kind")))
    if kind not in KIND_NAMES:
        raise ValueError(f"unsupported entity kind {path.get('kind')} for {path['id']}")
    return kind, path.get("parameters", [])


def sample_path(scene, path, core, stations):
    """No geometry formula here: the unique native kernel evaluates all paths."""
    kind, parameters = native_definition(scene, path, core)
    return np.asarray(core.sample_path(scene["board"], kind, parameters,
                     path.get("vertices", []), stations), dtype=np.float64)


def finite_path_support(path):
    """Read one finite material domain; periodic board paths use no such bound."""
    support = np.asarray(path.get('support', path.get('finite_support', [0., 1.])), dtype=float)
    if (support.shape != (2,) or not np.isfinite(support).all()
            or not 0. <= support[0] <= support[1] <= 1.):
        raise ValueError(f"invalid finite material support for {path['id']}")
    if ('support' in path and 'finite_support' in path
            and not np.array_equal(support, np.asarray(path['finite_support'], dtype=float))):
        raise ValueError(f"conflicting finite material supports for {path['id']}")
    return tuple(map(float, support))


def sample_entities(scene, core, samples=129):
    if samples < 2:
        raise ValueError("at least two representation samples")
    if scene.get('measurement_policy_id') or scene.get('schema') == 'rinkspace.field_model.v1':
        from ..domain.measurement import validate_scene_measurements
        validate_scene_measurements(scene)
    points, cuts, offset = [], [], 0
    for path in scene["paths"]:
        kind, parameters = native_definition(scene, path, core)
        vertices = path.get("vertices", [])
        count = max(samples, 4*len(vertices)) if kind in (2, 5) else samples
        if kind == 7:
            start, stop = 0., core.board_perimeter(scene['board'])
        else:
            start, stop = finite_path_support(path)
        stations = np.linspace(start, stop, count)
        if vertices:
            # Vertex knots retain their original material coordinate. A
            # truncated path must neither draw outside its support nor move
            # the original polyline corners to a newly normalized interval.
            knots = np.linspace(0., 1., len(vertices))
            stations = np.unique(np.r_[stations, knots[(knots >= start) & (knots <= stop)]])
        xyz = np.asarray(core.sample_path(scene['board'], kind, parameters, vertices, stations.tolist()), dtype=np.float64)
        if xyz.shape != (len(stations), 3) or not np.isfinite(xyz).all():
            raise ValueError(f"native entity sampling failed at {path['id']}")
        points.append(xyz)
        closed = kind == 7 or (kind in (1, 6) and abs((parameters[-1]-parameters[-2])*(stop-start)-2*math.pi) < 1e-10)
        cuts.append({**{k: path[k] for k in ("id", "entity_id", "label", "layer")},
                     "closed": bool(path.get('closed', closed) and (kind == 7 or (start == 0. and stop == 1.))), "begin": offset, "end": offset+len(xyz),
                     "kind": KIND_NAMES[kind], "support": None if kind == 7 else path.get("support", path.get('finite_support', [0., 1.])),
                     **({'station_domain':path['station_domain'], 'sampling_interval':[0., stop]} if kind == 7 else {})})
        offset += len(xyz)
    material_ids = ([m['id'] for m in scene['materials']] if 'materials' in scene
                    else list(dict.fromkeys(p['entity_id'] for p in scene['paths'])))
    return {"xyz": np.vstack(points), "cuts": cuts,
            "inventory": {"material_count": len(material_ids), "path_count": len(scene["paths"]),
                          "material_ids": material_ids, "path_ids": [p["id"] for p in scene["paths"]]},
            "missing_entities": scene.get("missing_entities", [])}
