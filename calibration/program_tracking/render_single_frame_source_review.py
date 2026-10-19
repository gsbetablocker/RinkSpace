"""Present one saved source frame and its measured points for user review.

Reads the point packet without changing it. Writes a native PNG in Data and
a local, self-contained HTML control surface beside the OneDrive packet.
There is no fitting, acceptance action, video decode or next-frame operation.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POINTS = ROOT / (
    "events/ankara_2026/programmes/okada_mei_sp/source_datasets/"
    "frame_by_frame_review_20261013/frame_000001_points.json"
)
DEFAULT_MEDIA = Path(
    "D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/"
    "okada_mei_sp/results/frame_by_frame_review_20261013/000001_229114112"
)
LAYERS = {
    "board_top": ("围挡上沿", "#f5b523"),
    "board_contact": ("黄色带本体下缘", "#00d9e9"),
    "ad2_border": ("广告边框", "#24e75c"),
    "ad2_print": ("广告内部轮廓", "#ff4242"),
    "ice_features": ("冰面标线与冰圈", "#9670ff"),
    "unbound_static": ("冰面实体：身份待确认", "#ff86df"),
}


def layer_for(feature_id: str) -> str:
    if feature_id in ("board_top", "board_contact"):
        return feature_id
    if re.match(r"^Ad\d+\.print\.", feature_id):
        return "ad2_print"
    if re.match(r"^Ad\d+\.", feature_id):
        return "ad2_border"
    if feature_id.startswith("ice."):
        return "ice_features"
    raise ValueError(f"Unsupported review feature: {feature_id}")


def source_segments(packet: dict):
    """Retain the saved segment breaks and explicit closure flags."""
    for feature in packet["features"] + packet.get('unbound_static_materials', []):
        layer = ('unbound_static' if feature in packet.get('unbound_static_materials', [])
                 else feature.get('review_layer') or layer_for(feature["id"]))
        if layer not in LAYERS:
            raise ValueError('Unknown review layer: ' + str(layer))
        for segment in feature["segments"]:
            uv = np.asarray(segment["uv"], dtype=float)
            if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all():
                raise ValueError(f"Invalid saved native points: {feature['id']}")
            if ((uv < [0, 0]).any() or (uv > [1919, 1079]).any()):
                raise ValueError(f"Saved point outside native image: {feature['id']}")
            if len(uv):
                yield layer, feature["id"], uv, bool(segment.get("closed", False))


def curve_draw_points(uv, geometry=None):
    """Draw only the supported segment of its saved source-fitted ellipse.

    Dense vertices are display coordinates, never additional observations.
    """
    uv = np.asarray(uv, dtype=float)
    if len(uv) < 2 or not geometry or geometry.get('kind') != 'projected_physical_circle_ellipse':
        return uv
    center = np.asarray(geometry['center_uv'], float)
    axes = np.asarray(geometry['axes_px'], float) / 2
    angle = np.deg2rad(geometry['angle_degrees'])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    local = (uv - center) @ rotation
    theta = np.unwrap(np.arctan2(local[:, 1] / axes[1], local[:, 0] / axes[0]))
    # Interpolate each consecutive supported pair. Segment gaps stay gaps.
    samples = []
    for a, b in zip(theta[:-1], theta[1:]):
        count = max(2, int(np.ceil(abs(b-a) * max(axes) / 3)) + 1)
        samples.extend(np.linspace(a, b, count)[:-1])
    samples.append(theta[-1])
    t = np.asarray(samples)
    return np.c_[axes[0] * np.cos(t), axes[1] * np.sin(t)] @ rotation.T + center


def current_material_display_curves(packet):
    """Return complete saved native material traces, separately from observations."""
    result={}
    if not packet:return result
    for feature in packet['features']+packet.get('unbound_static_materials',[]):
        if feature.get('manual_deletion_applied'):continue
        for segment in feature['segments']:
            if not segment.get('draw_current_native_material_readout'):continue
            if not segment.get('pixels_measured_in_this_frame') or not segment.get('source_evidence_measured_in_this_frame'):
                raise ValueError('Display material trace requires actual current-source readout')
            uv=np.asarray(segment['uv'],float)
            trace=np.asarray(segment['source_native_readout_uv'],float)
            if trace.ndim!=2 or trace.shape[1]!=2 or not np.isfinite(trace).all() or (trace<[0,0]).any() or (trace>[1919,1079]).any():
                raise ValueError('Invalid saved current material display trace')
            if len(uv) and len(trace):
                result[(feature['id'],tuple(uv[0]),tuple(uv[-1]))]=trace
    return result


def display_segments(packet):
    """Expose actual vertices plus optional source-fit geometry for rendering."""
    geometries = {f['id']: f.get('geometric_interpretation') for f in packet['features']}
    materials=current_material_display_curves(packet)
    return [dict(layer=layer, id=fid, uv=uv.tolist(), closed=closed,
                 draw_uv=materials.get((fid,tuple(uv[0]),tuple(uv[-1])),curve_draw_points(uv, geometries.get(fid))).round(4).tolist())
            for layer, fid, uv, closed in source_segments(packet)]


def native_overlay(source: np.ndarray, segments: list, inferred_segments=None, packet=None) -> np.ndarray:
    result = source.copy()
    line_image = source.copy()
    geometries = {f['id']: f.get('geometric_interpretation') for f in packet['features']} if packet else {}
    materials=current_material_display_curves(packet)
    for layer, fid, uv, closed in segments:
        color = tuple(bytes.fromhex(LAYERS[layer][1][1:]))[::-1]
        drawn=materials.get((fid,tuple(uv[0]),tuple(uv[-1])),curve_draw_points(uv, geometries.get(fid)))
        if len(drawn) > 1:
            # A tiny review-only contour may have one observation station;
            # its preserved complete source path still remains visible.
            points = np.rint(drawn * 256).astype(np.int32)
            cv2.polylines(line_image, [points], closed, color, 1, cv2.LINE_AA, shift=8)
    result = cv2.addWeighted(line_image, .75, result, .25, 0)
    for guide in inferred_segments or []:
        if not guide.get('inference_only'):raise ValueError('Display guide must be explicit inference')
        uv=np.asarray(guide['uv'],float)
        for a,b in zip(uv[:-1],uv[1:]):
            length=float(np.linalg.norm(b-a))
            if length==0:continue
            for start in np.arange(0,length,14.):
                ends=np.rint([a+(b-a)*start/length,a+(b-a)*min(start+7,length)/length]).astype(int)
                cv2.line(result,tuple(ends[0]),tuple(ends[1]),(255,112,150),1,cv2.LINE_AA)
    for layer, _, uv, _ in segments:
        color = tuple(bytes.fromhex(LAYERS[layer][1][1:]))[::-1]
        for point in np.rint(uv * 256).astype(np.int32):
            centre = tuple(map(int, point))
            cv2.circle(result, centre, 3 * 256, (15, 15, 15), -1, cv2.LINE_AA, shift=8)
            cv2.circle(result, centre, 2 * 256, color, -1, cv2.LINE_AA, shift=8)
    return result


def svg_layers(segments: list) -> str:
    groups = []
    for layer, (label, color) in LAYERS.items():
        lines, points = [], []
        for current, feature_id, uv, closed in segments:
            if current != layer:
                continue
            title = html.escape(feature_id)
            if len(uv) > 1:
                drawn = np.vstack((uv, uv[:1])) if closed else uv
                values = " ".join(f"{x:.3f},{y:.3f}" for x, y in drawn)
                lines.append(f'<polyline points="{values}"><title>{title}</title></polyline>')
            for x, y in uv:
                points.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="2.5">'
                              f'<title>{title}: ({x:.3f}, {y:.3f})</title></circle>')
        groups.append(
            f'<g data-layer="{layer}" aria-label="{html.escape(label)}">'
            f'<g class="review-lines" fill="none" stroke="{color}" '
            f'stroke-width=".75" stroke-opacity=".85">{"".join(lines)}</g>'
            f'<g class="review-points" fill="{color}" stroke="#111" '
            f'stroke-width=".65">{"".join(points)}</g></g>'
        )
    return "".join(groups)


def review_html(packet: dict, segments: list, source_path: Path, model_path: Path,
                model_binding: dict | None = None) -> str:
    counts = {key: sum(len(uv) for layer, _, uv, _ in segments if layer == key)
              for key in LAYERS}
    controls = "".join(
        f'<label class="layer"><input type="checkbox" data-layer-toggle="{key}" checked>'
        f'<span class="swatch" style="background:{color}"></span>{label}'
        f'<span class="count">{counts[key]} 点</span></label>'
        for key, (label, color) in LAYERS.items()
    )
    source = packet["source"]
    stamp = html.escape(str(source["source_time"]))
    pts = int(source["native_pts"])
    q = html.escape(str(source.get("model_q_sha256", ""))[:8])
    images = json.dumps({"source": source_path.resolve().as_uri(),
                         "model": model_path.resolve().as_uri()}, ensure_ascii=False)
    images = images.replace("</", "<\\/")
    pending_print = any(re.match(r"^Ad\d+\.print\.", feature["id"])
                        and feature.get("world_correspondence_status") == "template_binding_pending"
                        for feature in packet["features"])
    limitation = ("字轮廓已记录本帧原图像素；字形与三维模型的对应尚未完成，未进入求解。"
                  if pending_print else "字轮廓的三维对应状态以当前点数据记录为准。")
    if model_binding:
        limitation = (f"本帧 {packet['point_count']} 点均已建立新模型对应入口。"
                      "字形材料模板来自另一原始曝光，属于待优化初值；尚未重新求解。")
    accepted = bool(packet.get('user_accepted'))
    acceptance_label = '用户验收通过' if accepted else '等待用户验收'
    frame_index=packet.get('batch_sequence_index_1based',source.get('programme_frame_index_1based',1))
    frame_label=f'第 {frame_index} 帧'
    continuation = ('本帧已通过用户验收。' if accepted else
                    '仅呈现这一帧，等待用户检查。')
    regions=packet.get('unresolved_source_regions', [])
    unresolved = (f'<details><summary>未取区域及原因（{len(regions)} 项）</summary><ul>'
                  + ''.join(f"<li>{html.escape(region.get('target_id',''))}：{html.escape(region['note_zh'])}</li>"
                            for region in regions) + '</ul></details>') if regions else ''
    inferred=packet.get('inferred_segments',[])
    inference_svg='<g class="inferred-lines" fill="none" stroke="#9670ff" stroke-width="1" stroke-dasharray="7 7">'
    for guide in inferred:
        if not guide.get('inference_only'):raise ValueError('Display guide must be explicit inference')
        uv=' '.join(f'{x:.3f},{y:.3f}' for x,y in guide['uv'])
        inference_svg+=f'<polyline points="{uv}"><title>{html.escape(guide["reason_zh"])}</title></polyline>'
    inference_svg+='</g>'
    inference_control=('<label><input id="inference" type="checkbox" checked>遮挡处推断（虚线，无取点）</label>' if inferred else '')
    gap_note=('实测点在遮挡处断开；虚线仅表示直线的几何延续，不计入点数。' if inferred else '遮挡或缺失处保留断开。')
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{frame_label}取点验收 · {stamp}</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#16191d;color:#f3f4f6;font:15px/1.5 "Microsoft YaHei",sans-serif}}
header{{padding:14px 20px;border-bottom:1px solid #383e46}}h1{{font-size:20px;margin:0 0 5px}}
p{{margin:4px 0;color:#d0d5dd}}.meta{{font-variant-numeric:tabular-nums}}
.controls{{display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:12px 20px;background:#22272d}}
label{{display:inline-flex;gap:6px;align-items:center;white-space:nowrap}}input{{accent-color:#ddd}}
.swatch{{width:12px;height:12px;border-radius:50%;display:inline-block}}.count{{color:#aeb8c4}}
button,select{{font:inherit;background:#343b45;color:#fff;border:1px solid #5c6674;border-radius:5px;padding:5px 9px}}
button{{cursor:pointer}}.viewport{{overflow:auto;width:100%;padding:0;background:#090a0b}}
.stage{{position:relative;width:100%;max-width:1920px;line-height:0;margin:auto}}
.stage.native{{width:1920px;max-width:none;margin:0}}.stage img{{display:block;width:100%;height:auto}}
.stage svg{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}
.stage.hide-points .review-points,.stage.hide-lines .review-lines{{display:none}}
.stage.hide-inference .inferred-lines{{display:none}}
footer{{padding:10px 20px;color:#aeb8c4;font-size:13px}}
</style></head><body>
<header><h1>{frame_label}取点 · {acceptance_label}</h1>
<p class="meta">源时间 {stamp} · PTS {pts} · 原生 1920 × 1080 · 共 {sum(counts.values())} 点</p>
<p>{continuation}</p>
<p>{limitation}</p><div style="color:#f4c769">{unresolved}</div></header>
<div class="controls"><label>底图 <select id="background"><option value="source">原图</option>
<option value="model">{q} 网格</option></select></label>
<label><input id="points" type="checkbox" checked>点</label>
<label><input id="lines" type="checkbox" checked>细连线</label>
{inference_control}
<button id="fit" type="button">适应窗口</button><button id="native" type="button">1:1 原生像素</button>
{controls}</div>
<div class="viewport"><div class="stage" id="stage">
<img id="base" alt="{frame_label}原生图像" src="{html.escape(source_path.resolve().as_uri(), quote=True)}" width="1920" height="1080">
<svg viewBox="0 0 1920 1080" aria-label="当前帧实际取点">{inference_svg}{svg_layers(segments)}</svg>
</div></div><footer>细连线仅辅助看点，不会自动成为求解约束；{gap_note}网格使用指定 {q} 模型，仅供定位参考。</footer>
<script>
const images={images};
const stage=document.getElementById('stage');
document.getElementById('background').addEventListener('change',e=>{{document.getElementById('base').src=images[e.target.value];}});
document.getElementById('points').addEventListener('change',e=>stage.classList.toggle('hide-points',!e.target.checked));
document.getElementById('lines').addEventListener('change',e=>stage.classList.toggle('hide-lines',!e.target.checked));
document.getElementById('inference')?.addEventListener('change',e=>stage.classList.toggle('hide-inference',!e.target.checked));
document.getElementById('fit').addEventListener('click',()=>stage.classList.remove('native'));
document.getElementById('native').addEventListener('click',()=>stage.classList.add('native'));
document.querySelectorAll('[data-layer-toggle]').forEach(input=>input.addEventListener('change',()=>{{
 document.querySelector(`[data-layer="${{input.dataset.layerToggle}}"]`).style.display=input.checked?'':'none';
}}));
</script></body></html>'''


def run(args) -> dict:
    points_path = args.points.resolve()
    packet = json.loads(points_path.read_text(encoding="utf-8"))
    if (packet.get("schema") != "rinkspace.single-frame-source-review.v1"
            or packet.get("frame_count") != 1
            or int(packet["source"]["native_frame_index_1based"]) < 1):
        raise ValueError("This review entry requires exactly one explicitly selected native frame")
    source = cv2.imread(str(args.source_image), cv2.IMREAD_COLOR)
    model = cv2.imread(str(args.model_image), cv2.IMREAD_COLOR)
    if source is None or source.shape != (1080, 1920, 3):
        raise ValueError("Source image must retain its native 1920x1080 pixels")
    if model is None or model.shape != source.shape:
        raise ValueError("Same-frame model image must use the identical native image domain")
    if hashlib.sha256(source.tobytes()).hexdigest() != packet["source"]["decoded_BGR_sha256"]:
        raise ValueError("Source PNG differs from the point packet's original decoded frame")
    segments = list(source_segments(packet))
    overlay = args.overlay.resolve() if args.overlay else args.source_image.resolve().with_name("points_review.png")
    page = args.html.resolve() if args.html else points_path.with_name("review.html")
    if not overlay.is_relative_to(Path("D:/CodexData/RinkSpace_Data").resolve()):
        raise ValueError("Generated review PNG must stay in the existing Data directory")
    if overlay in (args.source_image.resolve(), args.model_image.resolve()) or page == points_path:
        raise ValueError("Review output must not overwrite an input")
    if not cv2.imwrite(str(overlay), native_overlay(source, segments, packet.get('inferred_segments'))):
        raise OSError(f"Could not write review PNG: {overlay}")
    frame_index=packet.get('batch_sequence_index_1based',packet['source'].get('programme_frame_index_1based',1))
    binding_path=points_path.with_name(f'frame_{frame_index:06d}_model_binding.json')
    model_binding=None
    if binding_path.is_file():
        candidate=json.loads(binding_path.read_text(encoding='utf8'))
        if (candidate.get('source_packet',{}).get('sha256') == hashlib.sha256(points_path.read_bytes()).hexdigest()
                and candidate.get('source_pts') == packet['source']['native_pts']):
            model_binding=candidate
    page.write_text(review_html(packet, segments, args.source_image, args.model_image, model_binding), encoding="utf-8")
    return {"points_review": str(overlay), "review_html": str(page),
            "source_pts": int(packet["source"]["native_pts"]),
            "point_count": sum(len(uv) for _, _, uv, _ in segments)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--source-image", type=Path, default=DEFAULT_MEDIA / "source.png")
    parser.add_argument("--model-image", type=Path, default=DEFAULT_MEDIA / "model_8a28afed.png")
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--html", type=Path)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
