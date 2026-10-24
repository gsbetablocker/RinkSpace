"""Render the first native-core slice, with an explicit incomplete-model label.

All world coordinates and junctions come from the native kernel. SVG transforms
only map those coordinates onto a page; they are not a second camera model.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path

from ..core.native import NativeCore


def _flat(segment):
    return [value for point in segment for value in point]


def preview(scene_path: str | Path, library_path: str | Path, output: str | Path) -> dict:
    source = Path(scene_path).resolve()
    raw = source.read_bytes()
    scene = json.loads(raw)
    if scene.get("schema") != "rinkspace.scene.seed.v1" or scene.get("status") != "unfitted_structure_candidate":
        raise ValueError("this first-slice preview only accepts explicitly unfitted scene seeds")
    geometry = scene["geometry"]
    if geometry["board"]["kind"] != "polar_superellipse_course" or geometry["curling"]["kind"] != "paired_equal_rows":
        raise ValueError("unsupported scene family; it must not be silently skipped")
    centre = geometry["centre_circle"]
    if centre["center"] != [0.0, 0.0, 0.0]:
        raise ValueError("centre circle must be constructed at the rink origin")
    group = geometry["curling"]
    p, rows = group["parameters"], group["row_count"]
    core = NativeCore(library_path)
    centres = core.centres(p, rows)
    lanes = core.lanes(p, rows, group["left_render_support"])
    board = geometry["board"]["parameters"]
    perimeter = core.board_perimeter(board)
    stations = [perimeter*i/512 for i in range(513)]
    boundary = core.board_points(board, stations, [0.0]*len(stations))
    top = core.board_points(board, stations, [board[5]]*len(stations))
    curves = [{"id": "board.bottom", "xyz": boundary}, {"id": "board.top", "xyz": top}]
    for h in range(2*rows):
        name = ("F" if h < rows else "R") + str(h % rows+1)
        for ring in group["ring_indices_present"]:
            points = [core.ring(p, rows, h, ring, 2*math.pi*i/96) for i in range(97)]
            curves.append({"id": f"{name}.ring{ring}", "xyz": points})
    junctions = []
    for row, segments in enumerate(lanes):
        for edge, segment in enumerate(segments):
            curves.append({"id": f"lane{row+1}.edge{edge}", "xyz": segment})
        for a, b, branch in ((0, 1, "lower_right"), (1, 2, "upper_right")):
            point = core.line_line(_flat(segments[a]), _flat(segments[b]))
            junctions.append({"id": f"lane{row+1}.{branch}", "parents": [f"lane{row+1}.edge{a}", f"lane{row+1}.edge{b}"],
                              "branch": "unique_finite_line_line", "xyz": point,
                              "source_observation": False, "per_frame_station": False})
    curves.append({"id": "C0.outer", "xyz": [core.centre_circle(centre["radius"], 2*math.pi*i/160) for i in range(161)]})
    output_dir = Path(output).resolve()
    generated = [output_dir/name for name in ("model.svg", "geometry.json", "result.json")]
    if any(path.exists() for path in generated):
        raise ValueError("preview outputs already exist; choose a new candidate output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {"schema": "rinkspace.native_core_preview.v1", "scene_id": scene["scene_id"], "status": scene["status"],
            "scene_sha256": hashlib.sha256(raw).hexdigest(), "native_abi": core.ABI_VERSION,
            "native_library_sha256": hashlib.sha256(core.library_path.read_bytes()).hexdigest(),
            "coordinate_unit": scene["frame"]["length_unit"], "curves": curves, "centres": centres,
            "junctions": junctions, "fit_executed": False, "cuda_executed": False,
            "missing_scope": scene["scope"]["not_yet_imported"], "provenance": scene["provenance"]}
    (output_dir/"geometry.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    # A single uniform scale preserves the geometry. Y is up in the world.
    scale = 1150/(2*board[0])
    def page(point):
        return 720+point[0]*scale, 460-point[1]*scale
    def path(points):
        return " ".join(("M" if i == 0 else "L")+f"{page(q)[0]:.4f},{page(q)[1]:.4f}" for i, q in enumerate(points))
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 990">',
             '<rect width="1440" height="990" fill="#f6f5ef"/>',
             '<style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;fill:#173b42}.title{font-size:36px;font-weight:700}.sub{font-size:19px;fill:#647b7f}.label{font-size:18px}.body{font-size:21px}.small{font-size:17px}</style>',
             '<text x="72" y="69" class="title">新的原生核心 · 结构预览</text>',
             '<text x="72" y="107" class="sub">C++ 直接生成几何与交点；统一尺度。未拟合、未叠加原片，当前不是完整场地迁移。</text>',
             '<rect x="1064" y="42" width="304" height="42" rx="21" fill="#f3dcbf"/>',
             '<text x="1216" y="70" text-anchor="middle" class="small">实际运行输出 / CPU 原生核心</text>']
    parts.append(f'<path d="{path(boundary)}" fill="#eaf1ee" stroke="#365d64" stroke-width="3"/>')
    for curve in curves:
        if curve["id"].startswith("board."):
            continue
        color = "#839d9b" if curve["id"].startswith("lane") else "#16868b"
        if curve["id"].startswith("C0"):
            color = "#6387ad"
        parts.append(f'<path d="{path(curve["xyz"])}" fill="none" stroke="{color}" stroke-width="2"/>')
    for h, q in enumerate(centres):
        x, y = page(q)
        name = ("F" if h < rows else "R")+str(h % rows+1)
        parts.append(f'<text x="{x:.3f}" y="{y-39:.3f}" text-anchor="middle" class="label">{name}</text>')
    for item in junctions:
        x, y = page(item["xyz"])
        parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="4.5" fill="#cf7638"/>')
    x0, y0 = page((0, 0, 0))
    parts += [f'<path d="M{x0-7},{y0}H{x0+7}M{x0},{y0-7}V{y0+7}" stroke="#6387ad" stroke-width="2"/>',
              f'<text x="{x0+13}" y="{y0-10}" class="label">C0 = (0, 0)</text>',
              '<rect x="72" y="790" width="1296" height="142" rx="18" fill="#fffdf7" stroke="#dddcd2"/>',
              '<text x="98" y="826" class="body">已由同一核心保证：圈等大、各行等距、圆心与赛道中线共用位置、C0 固定在原点。</text>',
              '<text x="98" y="861" class="body">橙点：原生有限线段求交；没有逐帧站位。当前仍是模型预测，没有源图像 UV 观测。</text>',
              '<text x="98" y="897" class="small">未完成：其余冰球标记、全部广告迁移、时变相机与原片拟合。左端仅绘制支撑，不生成第四条边。</text>',
              f'<text x="72" y="966" class="small">scene: {html.escape(scene["scene_id"])} · {hashlib.sha256(raw).hexdigest()[:12]} · 原始 d78 文件未改变</text>',
              '</svg>']
    (output_dir/"model.svg").write_text("\n".join(parts)+"\n", encoding="utf-8")
    result = {"model_preview": str(output_dir/"model.svg"), "geometry": str(output_dir/"geometry.json"),
              "status": "native_core_structure_preview_only", "fit_executed": False, "cuda_executed": False,
              "fixed_junctions": len(junctions), "source_observations_created": 0,
              "missing_scope": scene["scope"]["not_yet_imported"]}
    (output_dir/"result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return result
