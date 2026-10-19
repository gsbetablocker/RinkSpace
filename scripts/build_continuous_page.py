"""Export an offline, source-PTS synchronized continuous-shot review.

The HTML embeds geometry; source.mp4 contains every supplied sample at its
original relative PTS. No web server, external CDN, or JSON fetch is needed.
This is a display exporter, never an independent accuracy certification.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import html
import json
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def affine_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 3x3 affine matrix")
    if abs(matrix[2, 2]) < 1e-12:
        raise ValueError("Invalid homogeneous scale")
    matrix = matrix / matrix[2, 2]
    if not np.allclose(matrix[2], [0, 0, 1], atol=1e-10, rtol=0):
        raise ValueError("Projective homography needs a projective renderer; refusing affine approximation")
    if abs(np.linalg.det(matrix[:2, :2])) < 1e-9:
        raise ValueError("Singular affine map")
    return matrix


def projective_matrix(value: Any) -> np.ndarray:
    matrix=np.asarray(value,dtype=float)
    if matrix.shape!=(3,3) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 3x3 projective matrix")
    scale=matrix[2,2] if abs(matrix[2,2])>1e-12 else np.max(np.abs(matrix))
    if abs(scale)<1e-12:
        raise ValueError("Zero projective matrix")
    matrix=matrix/scale
    if np.linalg.matrix_rank(matrix)!=3:
        raise ValueError("Singular projective map")
    return matrix


def map_observed_polygon(matrix: np.ndarray, polygon: list) -> np.ndarray:
    """A linear homogeneous denominator cannot cross zero inside a polygon
    whose vertices all have the same nonzero sign. Check before division.
    """
    points=np.asarray(polygon,dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all():
        raise ValueError("Expected a finite observed polygon")
    homogeneous=np.column_stack((points,np.ones(len(points))))@matrix.T
    denominator=homogeneous[:,2]
    if (np.min(denominator)<=0<=np.max(denominator)) or np.min(np.abs(denominator))<1e-9:
        raise ValueError("Projective horizon crosses or touches an observed display polygon")
    transformed=homogeneous[:,:2]/denominator[:,None]
    if not np.isfinite(transformed).all() or np.max(np.abs(transformed))>1e6:
        raise ValueError("Unbounded observed footprint")
    return transformed


def xy_round(points: Any) -> list:
    return np.asarray(points, dtype=float).round(3).tolist()


def frame_geometry(run: Path, row: dict, person: dict, width: int, height: int,
                   corrections: dict) -> dict:
    correction = row.get("mask_correction") or corrections.get(str(row["sample_index"]), {})
    regions = list(correction.get("include_polygons", []))
    mask_file = run / row["mask_path"]
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != (height, width):
        raise ValueError(f"Missing/mismatched fitting mask: {mask_file}")
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # This outline is a display approximation of the actual fitting mask only.
    mask_paths = [cv2.approxPolyDP(c, 0.6, True)[:, 0, :].tolist()
                  for c in contours if len(c) >= 3 and cv2.contourArea(c) >= 2]
    if not regions:
        outer, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = [c[:, 0, :].tolist() for c in outer if cv2.contourArea(c) > 50]
    boxes = person.get("boxes", [])
    main = max(boxes, key=lambda b: (b["xyxy"][2]-b["xyxy"][0]) *
               (b["xyxy"][3]-b["xyxy"][1]))["xyxy"] if boxes else None
    if main:
        x0, y0, x1, y1 = main
        pad = 8
        regions.append([[max(0, x0-pad), max(0, y0-pad)],
                        [min(width-1, x1+pad), max(0, y0-pad)],
                        [min(width-1, x1+pad), min(height-1, y1+pad)],
                        [max(0, x0-pad), min(height-1, y1+pad)]])
    return {"regions": [xy_round(p) for p in regions],
            "display_exclude": correction.get("display_exclude_polygons", []),
            "mask_paths": mask_paths, "person": xy_round(main) if main else None,
            "fit": xy_round(row.get("fit_current_xy", [])),
            "support": xy_round(row.get("support_polygon", []))}


def encode_source(source: Path, rows: list[dict], destination: Path,
                  width: int, height: int) -> dict:
    absolute = [Fraction(r["pts"]) * Fraction(r["time_base"]) for r in rows]
    if any(b <= a for a, b in zip(absolute, absolute[1:])):
        raise ValueError("Source PTS must strictly increase")
    base = absolute[0]
    time_base = Fraction(rows[0]["time_base"])
    relative_ticks = [(t-base)/time_base for t in absolute]
    if any(t.denominator != 1 for t in relative_ticks):
        raise ValueError("Mixed PTS time bases cannot be represented exactly")
    period = sorted(b-a for a,b in zip(absolute,absolute[1:]))[len(rows)//2-1] if len(rows)>1 else Fraction(1,50)
    rate = Fraction(1, 1) / period
    with av.open(str(destination), "w", options={"movflags": "+faststart"}) as writer:
        stream = writer.add_stream("libx264", rate=rate)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"crf": "25", "preset": "fast", "bf": "0"}
        stream.time_base = stream.codec_context.time_base = time_base
        for row, tick in zip(rows, relative_ticks):
            picture = cv2.imread(str(source / row["image"]))
            if picture is None or picture.shape[:2] != (height, width):
                raise ValueError(f"Missing/mismatched source sample {row['image']}")
            frame = av.VideoFrame.from_ndarray(picture, format="bgr24")
            frame.pts, frame.time_base = int(tick), time_base
            for packet in stream.encode(frame):
                writer.mux(packet)
        for packet in stream.encode():
            writer.mux(packet)
    with av.open(str(destination)) as reader:
        decoded = [Fraction(f.pts) * f.time_base for f in reader.decode(video=0)]
    expected = [t-base for t in absolute]
    if decoded != expected:
        raise ValueError("Encoded preview PTS/count differ from original sample offsets")
    return {"source_time_origin_s": float(base), "decoded_frame_count": len(decoded),
            "exact_relative_pts_verified": True, "display_speed": 1,
            "all_samples_retained": True, "source_sample_dimensions": [width,height],
            "frames": [{"preview_frame_index": i, "preview_time_s": float(t-base),
                        "preview_pts": int(tick), "preview_time_base": str(time_base),
                        **{k:r.get(k) for k in ["sample_index","source_frame_index","pts","time_base","source_time_s"]}}
                       for i,(r,t,tick) in enumerate(zip(rows,absolute,relative_ticks))]}


def build(run: Path, people_path: Path | None, annotations_path: Path | None,
          out: Path, audit_path: Path | None = None, allow_projective: bool = False) -> dict:
    run, out = run.resolve(), out.resolve()
    reg_path = run / "registration.json"
    reg = read_json(reg_path)
    rows = reg["frames"]
    if not rows:
        raise ValueError("No source frames")
    source = Path(reg["sample_directory"])
    width,height = reg["decoded_dimensions"]
    people = read_json(people_path) if people_path else {}
    if people and people.get("source_sha256") != reg["source_sha256"]:
        raise ValueError("Person geometry belongs to a different source")
    person_rows = people.get("frames", [])
    person_rows = {r["sample_index"]:r for r in person_rows} if isinstance(person_rows,list) else person_rows
    annotations = read_json(annotations_path) if annotations_path else {}
    corrections = annotations.get("corrections",annotations)
    audit_path = audit_path or run / "withheld_contour_audit.json"
    audit = read_json(audit_path) if audit_path.exists() else {}
    if audit and (audit.get("source_sha256") != reg["source_sha256"] or
                  audit.get("registration_sha256") != sha256(reg_path)):
        raise ValueError("Independent audit source/run hash mismatch")
    audit_rows = {r["sample_index"]: r for r in audit.get("rows", [])}
    available = [r for r in rows if r.get("H_reference_from_current") is not None]
    if not available:
        raise ValueError("No candidate transforms")
    components = {r.get("graph_component_id") for r in available}
    if len(components) != 1:
        raise ValueError("Disconnected components cannot share one fixed canvas")
    middle = available[len(available)//2]
    validate_matrix=projective_matrix if allow_projective else affine_matrix
    gauge = np.linalg.inv(validate_matrix(middle["H_reference_from_current"]))
    all_points, compact = [], []
    for row in rows:
        geometry = frame_geometry(run,row,person_rows.get(row["sample_index"],{}),width,height,corrections)
        h = None
        if row.get("H_reference_from_current") is not None:
            h = projective_matrix(gauge @ validate_matrix(row["H_reference_from_current"]))
            for polygon in geometry["regions"]:
                transformed = map_observed_polygon(h,polygon)
                all_points.append(transformed)
        a = audit_rows.get(row["sample_index"], {})
        checks = [[*c["source_xy"], *c["predicted_source_xy"],
                   c["distance_to_reference_contour_1920_px"], c["raw_distance_1920_px"],
                   bool(c.get("nearest_segment_interior"))] for c in a.get("checks", [])]
        compact.append({"i":row["sample_index"],"frame":row.get("source_frame_index"),
                        "pts":row["pts"],"tb":row["time_base"],"t":row["source_time_s"],
                        "dt":float(Fraction(row["pts"])*Fraction(row["time_base"])-Fraction(rows[0]["pts"])*Fraction(rows[0]["time_base"])),
                        "h":h.round(11).tolist() if h is not None else None,
                        "status":row["status"],"reason":row.get("reason"),
                        "ref":bool(row.get("reference_frame_only")),**geometry,
                        "audit_status":a.get("status","not_available"),
                        "contours":[xy_round([p["xy"] for p in c["points"]]) for c in a.get("observed_contours",[])],
                        "checks":np.asarray(checks,dtype=float).round(4).tolist() if checks else [],
                        "median":a.get("median_1920_px"),"p95":a.get("p95_1920_px"),
                        "raw_median":a.get("raw_median_1920_px"),"raw_p95":a.get("raw_p95_1920_px"),
                        "audit_ref":row["sample_index"] == audit.get("reference_sample_index")})
    footprint = np.concatenate(all_points)
    low, high = np.floor(footprint.min(0))-15, np.ceil(footprint.max(0))+15
    if np.any(high-low <= 0):
        raise ValueError("Degenerate display footprint")
    data = {"source":reg.get("source_filename"),"sha256":reg["source_sha256"],
            "renderer":"projective_webgl_inverse_h" if allow_projective else "affine_canvas_2d",
            "width":width,"height":height,"rows":compact,"bounds":[low.tolist(),high.tolist()],
            "reference_sample":middle["sample_index"],"reference_time":middle["source_time_s"],
            "gauge":gauge.tolist(),"audit":{"present":bool(audit),"accepted":False,
            "evaluation_role":audit.get("evaluation_role"),
            "model_selection_independence_claimed":audit.get("model_selection_independence_claimed"),
            "independent_data":audit.get("independent_data"),
            "physical_groups":audit.get("physical_groups",0),"reference_time":audit.get("reference_source_time_s"),
            "summary":audit.get("summary"),"raw_summary":audit.get("raw_summary"),
            "limitations":audit.get("limitations",["No independent audit attached."])},
            "summary":reg.get("summary",{})}
    out.mkdir(parents=True,exist_ok=True)
    preview_map = encode_source(source,rows,out/"source.mp4",width,height)
    preview_map.update({"source_sha256":reg["source_sha256"],"source_filename":reg.get("source_filename"),
                        "registration_sha256":sha256(reg_path),"fixed_reference_gauge":gauge.tolist(),
                        "bounds_in_fixed_reference":data["bounds"],"reference_sample":middle["sample_index"],
                        "renderer":data["renderer"],"observed_polygon_denominator_checks":"all observed polygon vertices finite with same nonzero denominator sign",
                        "coordinate_space":"local_projective_reference_pixels_not_metres"})
    (out/"source_pts_map.json").write_text(json.dumps(preview_map,ensure_ascii=False,indent=2),encoding="utf-8")
    js_path = Path(__file__).resolve().parents[1] / "src/rinkspace/static/continuous_review.js"
    embedded = json.dumps(data,ensure_ascii=False,separators=(",",":"),allow_nan=False).replace("<","\\u003c")
    title = f"完整跟拍检查 · {rows[0]['source_time_s']:.2f}–{rows[-1]['source_time_s']:.2f} 秒"
    page = PAGE.replace("__TITLE__",html.escape(title)).replace("__DATA__",embedded).replace("__JS__",js_path.read_text(encoding="utf-8"))
    (out/"index.html").write_text(page,encoding="utf-8")
    manifest={"schema":"rinkspace.continuous-interactive-review.v1","source_sha256":reg["source_sha256"],
              "source_filename":reg.get("source_filename"),"registration_sha256":sha256(reg_path),
              "audit_sha256":sha256(audit_path) if audit else None,
              "exporter_sha256":sha256(Path(__file__)),"javascript_sha256":sha256(js_path),
              "frame_count":len(rows),"first_source_time_s":rows[0]["source_time_s"],
              "last_source_time_s":rows[-1]["source_time_s"],"all_samples_retained":True,
              "exact_relative_pts_verified":True,"independent_validation":False,
              "physical_groups":audit.get("physical_groups",0),"unit":"reference_px",
              "evaluation_role":audit.get("evaluation_role"),
              "model_selection_independence_claimed":audit.get("model_selection_independence_claimed"),
              "renderer":data["renderer"],
              "display_policy":"single fixed reference; whole-shot observed-plane and principal-person union; current frame only; no temporal fill; user zoom only",
              "projective_validation":"finite nonsingular matrices, single component, no horizon crossing/touching any observed polygon after fixed middle-frame gauge",
              "display_precision":{"polygon_px":0.001,"check_px":0.0001,"mask_contour_simplification_px":0.6},
              "limitations":["Source preview is encoded from sampled source JPEGs, not an original bitstream.",
                             "Person rectangle is a projected visual overlay, not a contact or 3D measurement.",
                             "Shape-normal contour discrepancy does not measure material-point or tangential motion accuracy.",
                             "One physical group cannot certify six-landmark or whole-rink accuracy.",
                             "Projective matrices require explicit opt-in and exact inverse-H WebGL rendering; WebGL failure is visibly unavailable, never approximated as affine.",
                             "Disconnected references and projective horizons inside observed polygons are refused."],
              "files":[{"path":p.name,"bytes":p.stat().st_size,"sha256":sha256(p)}
                       for p in [out/"index.html",out/"source.mp4",out/"source_pts_map.json"]]}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return {"out":str(out),"frame_count":len(rows),"bytes":sum(p.stat().st_size for p in out.iterdir() if p.is_file()),
            "bounds":data["bounds"],"exact_relative_pts_verified":True}


PAGE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>:root{color-scheme:dark;font-family:system-ui,"Microsoft YaHei",sans-serif;background:#101820;color:#e3ebf0}*{box-sizing:border-box}body{max-width:1800px;margin:auto;padding:22px}h1{font-size:25px;margin:0 0 8px}h2{font-size:17px;margin:0}p{line-height:1.65;margin:8px 0}.muted{color:#aabac7;font-size:13px}.notice{background:#382b19;border-left:4px solid #eab86c;padding:12px 16px;margin:14px 0}.toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:12px 0}button,input{font:inherit}button{background:#233748;color:#e7f2f8;border:1px solid #496276;border-radius:6px;padding:8px 13px;cursor:pointer}button:hover{background:#314e62}input[type=number]{background:#182631;color:white;border:1px solid #536b7b;width:110px;padding:7px}input[type=range]{width:100%;accent-color:#78d7d2}.main{display:grid;grid-template-columns:1fr 1fr;gap:15px}.card{background:#16232e;border:1px solid #314352;border-radius:8px;padding:12px;min-width:0}.card header{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:9px}canvas{display:block;width:100%;height:auto;background:#13191f;border-radius:3px}#fixed{cursor:grab;touch-action:none}#fixed:active{cursor:grabbing}#timeline{cursor:crosshair}.status{font-variant-numeric:tabular-nums;font-size:14px;line-height:1.7;min-height:26px}.warn{color:#ffd38a}.red{color:#ff8585}.green{color:#61e3ad}.blue{color:#79b8ff}.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px}.lower{display:grid;grid-template-columns:1.05fr 1fr;gap:15px;margin-top:15px}details{margin:12px 0}summary{cursor:pointer}a{color:#8adfda}#decoder{position:absolute;width:1px;height:1px;opacity:0;pointer-events:none}code{word-break:break-all}#timeline{aspect-ratio:1280/260}.quality{font-size:16px;margin:9px 0}.small{font-size:13px}@media(max-width:900px){.main,.lower{grid-template-columns:1fr}body{padding:12px}h1{font-size:21px}}</style></head><body>
<h1>__TITLE__</h1><p id="intro" class="muted"></p>
<div class="notice" id="auditNotice"><strong>连续显示候选，独立精度未通过。</strong>保留全部时间点与尾部失败。这是声明保护区域内、经源图审阅的单组轮廓诊断，只测形状／法向差异；环线切向位移不可辨识，不能当作六处分散地标、物质点轨迹或全场精度。</div>
<video id="decoder" src="source.mp4" muted playsinline preload="auto"></video>
<div class="toolbar"><button id="play" aria-label="播放或暂停">播放</button><button id="previous" aria-label="上一源帧">← 上一帧</button><button id="next" aria-label="下一源帧">下一帧 →</button><label>源时间 <input id="sourceTime" type="number" step="0.02" aria-label="源时间秒"></label><button id="go">跳转</button><label><input id="showAudit" type="checkbox" checked> 独立轮廓叠加</label><span id="frameStatus" class="status"></span></div>
<input id="seek" type="range" min="0" value="0" step="1" aria-label="全部源帧时间轴">
<div class="main"><section class="card"><header><h2>原片 · 当前源帧</h2><span class="muted">完整画面 · 采样代理</span></header><canvas id="original" width="960" height="540"></canvas><p class="legend"><span class="green">绿：独立观测轮廓</span><span class="red">红：参考轮廓预测最近点</span><span>细线：该处形状差异</span></p><div id="sourceStatus" class="status"></div></section>
<section class="card"><header><h2>固定冰面 · 同一参照</h2><div><button id="zoomOut" aria-label="缩小固定画布">−</button> <button id="zoomIn" aria-label="放大固定画布">＋</button> <button id="reset" aria-label="显示全部观察范围">显示全部</button></div></header><canvas id="fixed" width="960" height="540"></canvas><p class="muted">滚轮缩放，按住拖动。视口不自动跟随选手；棋盘区域未填入本帧像素。人体仅为二维投影显示，不是冰面接触测量。若仍见转播文字，它属于屏幕叠加，不能当作场地静物。</p><div id="fixedStatus" class="status"></div></section></div>
<div class="lower"><section class="card"><header><h2>整段独立误差 · 每帧均保留</h2><span class="muted">点击时间线回看源帧</span></header><canvas id="timeline" width="1280" height="260"></canvas><p class="legend"><span class="red">补偿后 P95</span><span class="green">补偿后中位数</span><span class="blue">未补偿 P95</span><span>灰区：无独立观测</span></p><div id="errorStatus" class="quality"></div><p id="summaryStatus" class="small muted"></p><p class="small muted">单位为原始 1920 宽等效像素。虚线 10 px 仅作原定 P95 提案参考；单组轮廓即使小于此线，也不构成正式验收通过。形状参考帧的零值是自比，不计作成功。</p></section>
<section class="card"><header><h2>实际拟合范围</h2><span class="muted">亮：允许域 · 暗：排除域</span></header><canvas id="fit" width="960" height="540"></canvas><p class="muted">橙点是保存的拟合采样，绿色为支持边界；它们不是独立误差。掩膜边界仅为显示简化，原始拟合审计保留在运行目录。</p></section></div>
<details><summary>来源、坐标与复现说明</summary><p id="provenance" class="small"></p><p class="small">固定画布使用一个实际中间源帧的坐标规范，覆盖整段全部显示冰面及主要人物框；不逐帧裁剪、不拼接过去画面、不生成未观测冰面。只有用户操作会改变缩放。透视候选使用逐像素逆投影；浏览器不能支持时明确显示不可用，不近似成仿射。互不连通的参考分量不会拼接到同一画布。</p><p class="small">本轮源审计记录的限制（原文）：</p><ul id="auditLimitations" class="small"></ul><p><a href="source_pts_map.json">完整源 PTS 映射</a> · <a href="manifest.json">精简验收文件与 SHA256</a> · <a href="source.mp4">完整连续源片段</a></p></details>
<script id="review-data" type="application/json">__DATA__</script><script>__JS__</script></body></html>'''


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--people",type=Path)
    parser.add_argument("--annotations",type=Path)
    parser.add_argument("--audit",type=Path)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--projective",action="store_true",help="Opt in to validated projective geometry and exact inverse-H WebGL rendering")
    args=parser.parse_args()
    print(json.dumps(build(args.run,args.people,args.annotations,args.out or args.run/"interactive",args.audit,args.projective),ensure_ascii=False))


if __name__=="__main__":
    main()
