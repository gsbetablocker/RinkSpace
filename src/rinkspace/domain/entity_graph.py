"""Current shared model import and fixed-node graph over the native geometry.

No historical model module is imported. The adapter consumes the saved current
model data once. Graph intersections are derived model nodes, never fabricated
pixel measurements. A fixed node owns no per-frame station parameter.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
from rinkspace.compat.d78_entities import sample_entities, sample_path, native_definition, KIND_NAMES
from rinkspace.core.native import NativeError


def _code(label):
    match = re.fullmatch(r"ice\.([FR])([1-4]):(center|inner|middle|outer)", label)
    if match:
        side, row, ring = match.groups()
        return 100+(int(row)-1+(0 if side == "F" else 4))*4+["center", "inner", "middle", "outer"].index(ring)
    match = re.fullmatch(r"ice\.house_connection_F([1-4])_R\1:segment0", label)
    if match:
        return 200+int(match[1])-1
    match = re.fullmatch(r"ice\.lane([1-4]):edge([0-2])", label)
    if match:
        return 300+(int(match[1])-1)*3+int(match[2])
    names = {"ice.C:segment0": 400, "ice.B1:segment0": 401, "ice.B2:segment0": 402,
             "ice.goal_line_left:segment0": 403, "ice.goal_line_right:segment0": 404,
             "ice.C0:outer": 405, **{f"ice.A{i}:outer": 405+i for i in range(1, 5)},
             "ice.goal_crease_left": 410, "ice.goal_crease_right": 411,
             "ice.goal_trapezoid_left_lower_outer:segment0": 420,
             "ice.goal_trapezoid_left_upper_outer:segment0": 421,
             "ice.goal_trapezoid_right_lower_outer:segment0": 422,
             "ice.goal_trapezoid_right_upper_outer:segment0": 423}
    return names.get(label)


def decode_shared_model(directory) -> dict:
    directory = Path(directory)
    model_path, parameters_path = directory/"model.json", directory/"parameters.npz"
    model = json.loads(model_path.read_text(encoding="utf-8-sig"))
    if model.get("schema") != "rinksolve.runtime_model.v1":
        raise ValueError("unsupported current shared model data format")
    family = model["scene"]["structural_family"]
    if family["layout_version"] != 2 or family["ratios_count"] != 3:
        raise ValueError("shared model adapter requires declared v2, three nested radii")
    with np.load(parameters_path, allow_pickle=False) as arrays:
        values = {block: np.asarray(arrays[key], dtype=np.float64).copy() for block, key in model["state"]["arrays"].items()}
    if any(not np.isfinite(v).all() for v in values.values()):
        raise ValueError("nonfinite shared model parameter")
    group_ids = [family[k] for k in ("curling_block_id", "lane_block_id", "hockey_block_id", "hockey_extra_block_id", "hockey_end_block_id")]
    sizes = [8, 10, 5, 6, 4]
    if any(values[key].shape != (size,) for key, size in zip(group_ids, sizes)):
        raise ValueError("shared structural parameter shape mismatch")
    board = values[family["board_block_id"]]
    if board.shape != (5,):
        raise ValueError("shared board shape mismatch")
    board_parameters = [.5, float(board[0])/2, *map(float, board[1:])]
    curves = {curve["id"]: curve for curve in model["scene"]["curves"]}
    installations = {item["id"]: item for item in model["installations"]}
    paths, material_ids = [], {}
    for original in model["scene"]["curves"]:
        curve, visited = original, set()
        while curve["kind"] == "curve_alias":
            if curve["id"] in visited:
                raise ValueError(f"cyclic alias {curve['id']}")
            visited.add(curve["id"])
            curve = curves[curve["definition"]["target_curve_id"]]
        d = deepcopy(curve["definition"])
        kind, label = d["kind"], curve["label"]
        entity = original["entity_id"]
        material_ids.setdefault(entity, {"id": entity, "label": entity.removeprefix("entity/"), "path_ids": []})["path_ids"].append(original["id"])
        p = {"id": original["id"], "entity_id": entity, "label": original["label"],
             "kind": kind, "support": [0., 1.], "closed": bool(d.get("closed", False)),
             "layer": "ice" if kind.startswith("ice_") else "board" if kind == "board_boundary" else "advertisement",
             "vertices_are_observations": False, "original_definition": deepcopy(original), "parameters": []}
        code = _code(label)
        if code is not None:
            p["generated_code"] = code
            p["kind"] = "ice_circle" if (100 <= code < 132 or 405 <= code <= 411) else "ice_segment"
            p["closed"] = p["kind"] == "ice_circle" and code not in (410, 411)
        elif kind == "installed_artwork_polyline":
            template = curves[d["template_curve_id"]]["definition"] if d.get("template_curve_id") else d
            if template.get("template_curve_id"):
                raise ValueError("template chain must have one canonical static source")
            installation = installations[d["installation_id"]]
            if installation["mapping_family"] != "unwrapped_rectangle_rotation.v1" or installation.get("shared_layout_block_id"):
                raise ValueError(f"unsupported installation mapping {installation['id']}")
            p["parameters"] = np.r_[values[installation["placement_block_id"]], values[installation["size_block_id"]]].tolist()
            p["vertices"] = deepcopy(template["normalized_points"])
            p["installation_id"] = installation["id"]
            p["template_curve_id"] = d.get("template_curve_id", curve["id"])
        elif kind == "board_boundary":
            p["parameters"] = [d["course_segment"], d["height_fraction"]]
        elif kind == "ice_segment":
            q = values[curve["parameter_block_id"]] if curve["parameter_block_id"] else np.asarray(d["endpoints_xy"])
            p["parameters"] = [float(q[0]), float(q[1]), 0., float(q[2]), float(q[3]), 0.]
        elif kind == "ice_circle":
            q = values[curve["parameter_block_id"]] if curve["parameter_block_id"] else None
            xy, radius = (q[:2], q[2]) if q is not None else (d["center_xy"], d["radius"])
            p["parameters"] = [*map(float, xy), 0., float(radius), *map(float, d["angle_interval"])]
        else:
            raise ValueError(f"unsupported shared entity {original['id']}: {kind}")
        if visited:
            p["physical_alias_of"] = curve["id"]
        paths.append(p)
    missing = list(model["scene"]["missing_inventory"])
    # The paired Ad25 artwork exists, but no independent installation was
    # recovered. Its entity is kept visible in inventory, without fake XYZ.
    if "Ad25_placement_unknown" in missing:
        material_ids.setdefault("entity/Ad25", {"id": "entity/Ad25", "label": "Ad25", "path_ids": [], "status": "placement_unknown_template_present"})
    return {"schema": "rinkspace.scene.shared.v1", "scene_id": "ankara_2026.shared_candidate", "revision": 1,
            "status": "unfitted_structure_candidate", "frame": {"frame_id": "ankara2026_combined_initial_L1", "length_unit": "rink_length", "meters_per_unit": None},
            "board": board_parameters, "shared_layout": np.concatenate([values[key] for key in group_ids]).tolist(),
            "shared_layout_blocks": [{"id": key, "offset": sum(sizes[:i]), "size": size} for i, (key, size) in enumerate(zip(group_ids, sizes))],
            "structural_family": deepcopy(family), "parameter_values": {key: value.tolist() for key, value in values.items()},
            "installations": deepcopy(model["installations"]), "artworks": deepcopy(model["artworks"]),
            "materials": list(material_ids.values()), "paths": paths, "missing_entities": missing,
            "source_model": str(model_path.resolve()), "source_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "source_parameters_sha256": hashlib.sha256(parameters_path.read_bytes()).hexdigest(),
            "fit_executed": False, "user_approval": "not_reviewed"}


def resolve_path(scene, path, core):
    if "generated_code" in path or "native_kind" in path:
        kind, parameters = native_definition(scene, path, core)
        return {**path, "kind": KIND_NAMES[kind], "native_kind":kind, "parameters": parameters}
    return path


def carrier_boundary_junctions(resolved_paths):
    """Name existing native carrier ends on the current continuous ice edge.

    This consumes evaluated geometry, never estimates XYZ from display samples
    or creates image observations. Only explicitly declared physical boundary
    intersections qualify; drawing supports and unmeasured M4 attachments do not.
    """
    carriers = [p for p in resolved_paths
                if p.get("native_kind", p.get("kind")) in (0, "ice_segment")
                and p.get("endpoint_roles") == ["derived_ice_boundary_intersection"]*2]
    if not carriers:
        return []
    boundaries = [p for p in resolved_paths
                  if p.get("native_kind", p.get("kind")) in (7, "board_course")
                  and p.get("layer") == "board" and p.get("parameters") == [0.]]
    if len(boundaries) != 1:
        raise ValueError("Declared carrier intersections need one continuous ice boundary")
    boundary = boundaries[0]
    nodes = []
    for carrier in carriers:
        parameters = np.asarray(carrier["parameters"], dtype=float)
        if parameters.shape != (6,) or not np.isfinite(parameters).all():
            raise ValueError("Carrier endpoints must come from a finite native segment")
        parents = [carrier, boundary]
        for branch, xyz in enumerate(parameters.reshape(2, 3)):
            nodes.append({
                "id": "junction/"+"|".join(sorted(p["id"] for p in parents))+f"/{branch}",
                "kind": "FixedJunction", "parents": [p["id"] for p in parents],
                "surface": "ice", "junction_relation": "ice_board_bottom",
                "parent_labels": [p["label"] for p in parents], "branch": branch,
                "xyz": xyz.tolist(), "meaning": "material_centreline_meets_continuous_ice_boundary",
                "source_observation_ids": [], "source_uv": None,
                "semantic_status": "derived_model_node_unobserved",
                "per_frame_station_parameter": False, "coordinate_unit": "rink_length",
                "role": "derived_ice_boundary_intersection", "new_source_observation": False,
            })
    return nodes


def fixed_node_graph(scene, core) -> dict:
    if scene.get('measurement_policy_id') or scene.get('schema') == 'rinkspace.field_model.v1':
        from .measurement import validate_scene_measurements
        validate_scene_measurements(scene)
    resolved = [resolve_path(scene, p, core) for p in scene["paths"] if not p.get("physical_alias_of")]
    paths = [p for p in resolved if p["layer"] == "ice"]
    lines = [p for p in paths if p["kind"] == "ice_segment"]
    circles = [p for p in paths if p["kind"] == "ice_circle" and abs(p["parameters"][5]-p["parameters"][4]-2*math.pi) < 1e-10]
    nodes = []
    def add(parents, branch, xyz, meaning):
        nodes.append({"id": "junction/"+"|".join(sorted(p["id"] for p in parents))+f"/{branch}",
                      "kind": "FixedJunction", "parents": [p["id"] for p in parents], "parent_labels": [p["label"] for p in parents],
                      "branch": branch, "xyz": list(map(float, xyz)), "meaning": meaning,
                      "source_observation_ids": [], "source_uv": None, "semantic_status": "derived_model_node_unobserved",
                      "per_frame_station_parameter": False, "coordinate_unit": "rink_length"})
    for i, a in enumerate(lines):
        for b in lines[i+1:]:
            try:
                xyz = core.line_line(a["parameters"], b["parameters"])
            except NativeError as exc:
                if exc.status not in (2, 3, 4):
                    raise
            else:
                add([a, b], 0, xyz, "intersection_of_finite_material_centrelines")
        for circle in circles:
            for branch in (0, 1):
                try:
                    xyz = core.line_circle(a["parameters"], circle["parameters"][:4], branch)
                except NativeError as exc:
                    if exc.status not in (2, 3, 4):
                        raise
                else:
                    meaning = ("intersection_of_finite_material_centrelines"
                               if circle.get("measurement") == "centreline"
                               else "line_midwidth_axis_crosses_named_ring_material_boundary")
                    add([a, circle], branch, xyz, meaning)
    nodes.extend(carrier_boundary_junctions(resolved))
    # Material corners belong to an installation chart. They are not generated
    # from arbitrary print polyline vertices and are not pixel constraints.
    installation_paths = {}
    for p in scene["paths"]:
        if p.get("installation_id"):
            installation_paths.setdefault(p["installation_id"], p)
    for installation_id, p in installation_paths.items():
        for name, coordinate in zip(("lower_left", "lower_right", "upper_right", "upper_left"), ([0., 0.], [1., 0.], [1., 1.], [0., 1.])):
            q = {**p, "vertices": [coordinate]}
            xyz = sample_path(scene, q, core, [0.])[0]
            nodes.append({"id": f"material/{installation_id}/{name}", "kind": "MaterialPoint", "surface_id": "board/continuous",
                          "installation_id": installation_id, "material_coordinate": coordinate, "corner": name, "xyz": xyz.tolist(),
                          "parents": [installation_id], "parent_labels": [installation_id.removeprefix("installation/")],
                          "meaning": "declared_unwrapped_installation_extent_corner", "source_observation_ids": [], "source_uv": None,
                          "semantic_status": "initial_installation_extent_unobserved", "per_frame_station_parameter": False,
                          "coordinate_unit": "rink_length"})
    for i, node in enumerate(nodes):
        node["display_id"] = f"N{i+1:03d}"
    return {"schema": "rinkspace.fixed_node_graph.v1", "scene_id": scene["scene_id"], "scene_revision": scene["revision"],
            "nodes": nodes, "node_count": len(nodes), "fixed_junctions": sum(n["kind"] == "FixedJunction" for n in nodes),
            "material_points": sum(n["kind"] == "MaterialPoint" for n in nodes), "new_source_measurements": 0,
            "notes": ["固定节点指同一实体关系派生、无逐帧站位，不表示世界坐标固定或已有像素观测。", "数学交点和安装角点未自动绑定UV；原片已有确认观测由PreparedDataset独立保留。", "宽标线以材料宽度中线、靶圈和广告以各自实体边界解释。", "广告长方形在连续围挡展开面定义，允许跨直线与圆弧。"],
            "excluded": ["arbitrary_polyline_vertices", "finite_drawing_support_endpoints_without_physical_corner_meaning", "people", "stands", "reflections"],
            "user_approval": "not_reviewed", "fit_executed": False}


def write_json(path, payload):
    path = Path(path).resolve()
    if len(str(path).encode("utf-16-le"))//2 >= 260 or any(len(p.encode("utf-16-le"))//2 > 255 for p in path.parts):
        raise ValueError(f"Windows path length: {path}")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return str(path)


def render_graph(scene, graph, core, out_dir):
    """Native coordinates only; figures are a structural review, not a fit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sampled = sample_entities(scene, core, samples=129)
    xyz, cuts = sampled["xyz"], sampled["cuts"]
    fig, ax = plt.subplots(figsize=(22, 13), dpi=160)
    for cut in cuts:
        if cut["layer"] == "advertisement":
            continue
        points = xyz[cut["begin"]:cut["end"]]
        color = "#2c8b98" if cut["layer"] == "board" else "#287cc1" if any(x in cut["label"] for x in ("F", "R", "lane")) else "#b95642"
        ax.plot(points[:, 0], points[:, 1], color=color, lw=.9, alpha=.8)
        if cut["layer"] == "ice" and not (":" in cut["label"] and not cut["label"].endswith((":outer", ":segment0", ":edge1"))):
            point = points[len(points)//2]
            ax.text(point[0], point[1], cut["label"].removeprefix("ice."), fontsize=6, color=color)
    junctions = [n for n in graph["nodes"] if n["kind"] == "FixedJunction"]
    for node in junctions:
        x, y, _ = node["xyz"]
        ax.plot(x, y, ".", color="#1b3238", ms=3)
        ax.annotate(node["display_id"], (x, y), xytext=(2, 3), textcoords="offset points", fontsize=4.5, color="#1b3238")
    ax.set_aspect("equal");ax.set_xlabel("X / rink-length units");ax.set_ylabel("Y / rink-length units")
    ax.set_title("Shared scene: all ice/board entities and derived fixed junctions | UNFITTED / UV UNBOUND")
    ax.grid(alpha=.15);fig.tight_layout()
    top = out/"model_nodes.png"
    fig.savefig(top);plt.close(fig)
    # Board diagram uses exact native world samples, with a 3D view to avoid a
    # second station formula in the plotting layer.
    fig = plt.figure(figsize=(22, 12), dpi=150);ax = fig.add_subplot(111, projection="3d")
    ad_labels = set()
    for cut in cuts:
        if cut["layer"] == "ice":
            continue
        points = xyz[cut["begin"]:cut["end"]]
        ax.plot(points[:,0], points[:,1], points[:,2], lw=.7, color="#2c8b98" if cut["layer"] == "board" else "#7c7754")
        match = re.match(r"Ad\d+", cut["label"])
        if match and match[0] not in ad_labels:
            pos = points[len(points)//2];ax.text(*pos, match[0], fontsize=8);ad_labels.add(match[0])
    for node in graph["nodes"]:
        if node["kind"] == "MaterialPoint":
            ax.scatter(*node["xyz"], s=3, color="#8d4b3d")
    ax.set_box_aspect((1, scene["board"][1]*2, .17));ax.view_init(elev=35, azim=-65)
    ax.set_title("Continuous board / paired artwork / installation corners | height visually enlarged; Ad25 placement unknown")
    ax.set_xlabel("X / rink-length");ax.set_ylabel("Y / rink-length");ax.set_zlabel("Z / rink-length")
    fig.tight_layout();board = out/"board_entities.png";fig.savefig(board);plt.close(fig)
    rows = []
    for n in graph["nodes"]:
        rows.append(f"<tr><td>{n['display_id']}</td><td>{n['kind']}</td><td>{' × '.join(n['parent_labels'])}</td><td>{n.get('branch', n.get('corner',''))}</td><td>{', '.join(f'{v:.7f}' for v in n['xyz'])}</td><td>未绑定像素</td></tr>")
    html = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>共同模型与固定节点</title><style>body{font:15px/1.6 system-ui;margin:28px;color:#233c41;background:#f5f7f4}img{max-width:100%;background:white}table{border-collapse:collapse;background:white}td,th{padding:5px 10px;border:1px solid #ccd5d3}.note{padding:15px;background:#fff0d1}code{word-break:break-all}</style><h1>共同模型与固定节点 · 待审核结构候选</h1><p class="note">所有曲线由同一 native 核心生成。尚未重新拟合；图中节点是共享实体关系的数学派生，不是已测UV。原片回放与模型结构迁移审核分别记录，不能用此图宣称贴合通过。</p><p>完整冰球、冰壶、围挡、广告及编号保留；Ad25有配对图案而独立安装位置未知，不虚构位置。圈轮廓、广告轮廓使用其材料边界；宽冰面直线按材料两侧中线解释。尺寸沿用当前候选初值，未验证米制。</p><img src="model_nodes.png"><img src="board_entities.png"><p><a href="scene.json">当前模型数据</a> · <a href="fixed_nodes.json">完整固定节点定义</a></p><h2>节点定义与真实观测边界</h2><p>FixedJunction由两个有限父实体及branch定义，没有逐帧station；场地参数改变时世界坐标由同一native关系更新。MaterialPoint属于同一连续围挡安装坐标；普通广告图案采样顶点不自动成为固定节点。当前表中不新增任何像素约束。真实画面放大由回放审核输出，投影导航也不等于重新标注。</p><table><tr><th>ID</th><th>类型</th><th>父实体</th><th>分支/角</th><th>XYZ 场地长度单位</th><th>源观测</th></tr>"""+"".join(rows)+"</table></html>"
    index = out/"index.html";index.write_text(html, encoding="utf-8")
    return {"top_view": str(top), "board_view": str(board), "index": str(index), "node_count": graph["node_count"], "sampled_path_count": len(cuts), "fit_executed": False}
