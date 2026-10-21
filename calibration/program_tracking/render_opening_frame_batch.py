"""Render the authorized opening-frame packets without changing source points.

Images and raster overlays stay in Data. The OneDrive HTML embeds only metadata
and point coordinates, and loads each native PNG on demand. No playback, fitting,
video decode, or approval mutation is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from fractions import Fraction
from pathlib import Path
import sys

import cv2

try:
    from .render_single_frame_source_review import LAYERS, native_overlay, source_segments, display_segments
except ImportError:
    from render_single_frame_source_review import LAYERS, native_overlay, source_segments, display_segments


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from rinkspace.packed_assets import iter_project_assets, read_project_bytes, read_project_json
DATA_ROOT = Path("D:/CodexData/RinkSpace_Data")
DEFAULT_DATASET = ROOT / (
    "events/ankara_2026/programmes/okada_mei_sp/source_datasets/"
    "frame_by_frame_review_20261013/opening_100_50fps"
)
DEFAULT_MEDIA = DATA_ROOT / (
    "events/ankara_2026/programmes/okada_mei_sp/results/"
    "frame_by_frame_review_20261013/opening_100_50fps"
)


from rinkspace.storage import AssetStore

def timestamp(source: dict) -> str:
    value = source.get("source_time")
    if value is not None:
        return str(value)
    pts = source.get("native_pts", source.get("actual_pts", source.get("pts")))
    base = source.get("time_base")
    if pts is None or base is None:
        return "未记录源时间"
    milliseconds = round(float(int(pts) * Fraction(str(base))) * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def page_html(frames: list[dict], manifest_path: Path | None = None, *,
              frame_json: list[str] | None = None) -> str:
    controls = "".join(
        f'<label><input type="checkbox" data-layer="{key}" checked>'
        f'<span class="swatch" style="background:{color}"></span>{html.escape(label)} '
        f'<span id="count-{key}" class="muted"></span></label>'
        for key, (label, color) in LAYERS.items()
    )
    if frame_json is None:
        data = json.dumps(dict(frames=frames, layers=LAYERS), ensure_ascii=False, separators=(",", ":"))
    else:
        if len(frame_json) != len(frames):
            raise ValueError("Serialized review frame count differs from the frame inventory")
        data = '{"frames":[' + ','.join(frame_json) + '],"layers":' + json.dumps(
            LAYERS, ensure_ascii=False, separators=(",", ":")) + '}'
    # A packet's descriptive text is data, never executable HTML or JavaScript.
    data = data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    manifest_link = (f'<a href="{html.escape(manifest_path.resolve().as_uri(), quote=True)}">批次数据记录</a>'
                     if manifest_path else "")
    title = f"第 {frames[0]['index']}–{frames[-1]['index']} 帧取点审阅"
    return _PAGE.replace("__TITLE__", title).replace("__FIRST_INDEX__", str(frames[0]['index'])).replace("__LAST_INDEX__", str(frames[-1]['index'])).replace("__FRAME_COUNT__", str(len(frames))).replace(
        "__LAYER_CONTROLS__", controls).replace("__DATA__", data).replace(
        "__MANIFEST_LINK__", manifest_link)


_PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
*{box-sizing:border-box}body{margin:0;color:#ecf0f4;background:#15191e;font:14px/1.5 "Microsoft YaHei",sans-serif}
header{padding:12px 18px;border-bottom:1px solid #3a434e}h1{font-size:19px;margin:0 0 4px}p{margin:3px 0}
.meta{font-variant-numeric:tabular-nums}.muted{color:#aeb9c7}.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:10px 18px;background:#242b34}
button,select,input[type=number]{font:inherit;background:#35404d;border:1px solid #647184;border-radius:4px;color:#fff;padding:4px 8px}
button{cursor:pointer}button:disabled{opacity:.4;cursor:default}input[type=number]{width:75px}label{display:inline-flex;align-items:center;gap:5px;white-space:nowrap}
input{accent-color:#88baff}.swatch{width:10px;height:10px;border-radius:50%}.viewport{width:100%;height:calc(100vh - 235px);min-height:400px;overflow:auto;background:#090b0e}
.stage{position:relative;width:1920px;height:1080px;line-height:0}.stage img{display:block;width:100%;height:100%}.stage svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.stage.hide-points .review-points,.stage.hide-lines .review-lines,.stage.hide-inference .review-inference{display:none}footer{padding:8px 18px;color:#aeb9c7;font-size:12px}a{color:#9bc7ff}
#error{color:#ffadad;display:none}#notes{color:#e4c07a;max-width:1500px}#position{min-width:110px;text-align:center}
</style></head><body>
<header><h1>__TITLE__</h1>
<p class="meta" id="meta"></p><p id="status"></p><details><summary>本帧说明</summary><p id="notes"></p></details><p id="error" role="alert"></p>
</header>
<div class="controls">
<button id="previous" type="button">← 上一帧</button><span id="position" class="meta"></span><button id="next" type="button">下一帧 →</button>
<label>跳到第 <input id="jump" type="number" min="__FIRST_INDEX__" max="__LAST_INDEX__" value="__FIRST_INDEX__"> 帧</label><button id="go" type="button">跳转</button>
<label>底图 <select id="background"><option value="source">原图</option><option value="navigation" id="navigation-option">网格导航</option></select></label>
<label><input id="points" type="checkbox" checked>点</label><label><input id="lines" type="checkbox" checked>细连线</label><label><input id="inference" type="checkbox" checked>遮挡处推断（虚线，无取点）</label>
<label>缩放 <select id="zoom"><option value="fit">适应窗口</option><option value="0.5">50%</option><option value="0.75">75%</option><option value="1" selected>100% 原生像素</option><option value="1.5">150%</option><option value="2">200%</option><option value="3">300%</option></select></label>
<button id="fit" type="button">适应窗口</button><button id="native" type="button">1:1</button>
</div><div class="controls">__LAYER_CONTROLS__</div>
<div class="viewport" id="viewport"><div class="stage" id="stage">
<img id="base" alt="当前帧原生图像" width="1920" height="1080"><svg id="drawing" viewBox="0 0 1920 1080" aria-label="当前帧实际取点"></svg>
</div></div>
<footer>使用左右方向键逐帧；数字输入后按 Enter 跳转。细连线辅助审阅，遮挡处虚线不计点、不参与拟合。网格只供定位。__MANIFEST_LINK__</footer>
<script type="application/json" id="batch-data">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('batch-data').textContent),frames=data.frames,layers=data.layers;
const get=id=>document.getElementById(id),stage=get('stage'),drawing=get('drawing'),base=get('base'),ns='http://www.w3.org/2000/svg';
let current=0;
function node(name,attributes,parent){const element=document.createElementNS(ns,name);for(const [key,value]of Object.entries(attributes))element.setAttribute(key,value);if(parent)parent.appendChild(element);return element;}
function setZoom(value){stage.style.width=(1920*value)+'px';stage.style.height=(1080*value)+'px';}
function draw(){drawing.replaceChildren();const frame=frames[current];for(const [layer,[label,color]]of Object.entries(layers)){
 const toggle=document.querySelector('input[data-layer="'+layer+'"]');if(!toggle.checked)continue;
 const group=node('g',{'data-layer':layer,'aria-label':label},drawing),lineGroup=node('g',{'class':'review-lines','fill':'none','stroke':color,'stroke-width':'.8','stroke-opacity':'.85'},group),pointGroup=node('g',{'class':'review-points','fill':color,'stroke':'#111','stroke-width':'.65'},group);
 for(const segment of frame.segments){if(segment.layer!==layer)continue;if(segment.uv.length>1){const path=segment.draw_uv||segment.uv;const points=segment.closed?[...path,path[0]]:path;const line=node('polyline',{points:points.map(point=>point.join(',')).join(' ')},lineGroup);node('title',{},line).textContent=segment.id;}
  for(const [x,y]of segment.uv){const circle=node('circle',{cx:x,cy:y,r:2.5},pointGroup);node('title',{},circle).textContent=segment.id+': ('+x.toFixed(3)+', '+y.toFixed(3)+')';}
 }
}if(get('inference').checked&&document.querySelector('input[data-layer="ice_features"]').checked){const inferred=node('g',{'class':'review-inference','fill':'none','stroke':'#b486ed','stroke-width':'1','stroke-dasharray':'6 5'},drawing);for(const segment of frame.inferred_segments||[]){const line=node('polyline',{points:segment.uv.map(point=>point.join(',')).join(' ')},inferred);node('title',{},line).textContent=segment.target_id+'：遮挡处推断，无取点';}}}
function show(index){current=Math.max(0,Math.min(frames.length-1,Math.round(index)));const frame=frames[current];
 get('position').textContent=(current+1)+' / '+frames.length;get('jump').value=frame.index;get('previous').disabled=current===0;get('next').disabled=current===frames.length-1;
 get('meta').textContent='第 '+frame.index+' 帧 · 源时间 '+frame.source_time+' · PTS '+frame.pts+' · time_base '+frame.time_base+' · 原生 1920 × 1080 · '+frame.point_count+' 点';
 get('status').textContent=frame.status;get('notes').textContent=frame.notes.join(' ');get('error').style.display='none';
 get('navigation-option').hidden=!frame.navigation;get('navigation-option').disabled=!frame.navigation;if(!frame.navigation&&get('background').value==='navigation')get('background').value='source';
 base.src=get('background').value==='navigation'?frame.navigation:frame.source;base.alt='第 '+frame.index+' 帧，源时间 '+frame.source_time;
 for(const layer of Object.keys(layers))get('count-'+layer).textContent=frame.counts[layer]+' 点';draw();
 history.replaceState(null,'','#frame='+String(frame.index));
}
base.addEventListener('error',()=>{get('error').textContent='当前底图加载失败，请检查本地 PNG 路径及文件权限。';get('error').style.display='block';});
get('previous').addEventListener('click',()=>show(current-1));get('next').addEventListener('click',()=>show(current+1));
function positionForFrame(value){let best=0;for(let i=1;i<frames.length;i++){if(Math.abs(frames[i].index-value)<Math.abs(frames[best].index-value))best=i;}return best;}
function jump(){const value=Number(get('jump').value);if(Number.isFinite(value))show(positionForFrame(value));}
get('go').addEventListener('click',jump);get('jump').addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();jump();}});
get('background').addEventListener('change',()=>show(current));
get('points').addEventListener('change',event=>stage.classList.toggle('hide-points',!event.target.checked));get('lines').addEventListener('change',event=>stage.classList.toggle('hide-lines',!event.target.checked));
get('inference').addEventListener('change',draw);
document.querySelectorAll('input[data-layer]').forEach(input=>input.addEventListener('change',draw));
function fit(){get('zoom').value='fit';setZoom(Math.min(get('viewport').clientWidth/1920,get('viewport').clientHeight/1080));}
get('zoom').addEventListener('change',event=>event.target.value==='fit'?fit():setZoom(Number(event.target.value)));get('native').addEventListener('click',()=>{get('zoom').value='1';setZoom(1);});
get('fit').addEventListener('click',fit);window.addEventListener('resize',()=>{if(get('zoom').value==='fit')fit();});
document.addEventListener('keydown',event=>{if(event.target.closest('input,select,textarea')||event.altKey||event.ctrlKey||event.metaKey)return;if(event.key==='ArrowLeft'){event.preventDefault();show(current-1);}else if(event.key==='ArrowRight'){event.preventDefault();show(current+1);}else if(event.key==='Home'){event.preventDefault();show(0);}else if(event.key==='End'){event.preventDefault();show(frames.length-1);}});
const initial=Number((location.hash.match(/^#frame=(\d+)$/)||[])[1]);show(Number.isFinite(initial)?positionForFrame(initial):0);setZoom(1);
</script></body></html>'''


def run(args) -> dict:
    dataset, media = args.dataset.resolve(), args.media.resolve()
    source_media = (getattr(args, 'source_media', None) or media).resolve()
    accepted_manifest = dataset / "user_acceptance.json"
    accepted_indices = set()
    if accepted_manifest.is_file():
        accepted_record = read_project_json(accepted_manifest, project_root=ROOT)
        for item in accepted_record.get("frames", []):
            packet_file = dataset / "frames" / f"{item['index']:06d}.json"
            if hashlib.sha256(read_project_bytes(packet_file, project_root=ROOT)).hexdigest() != item['packet_sha256']:
                raise ValueError("Accepted frame packet has changed")
            accepted_indices.add(item['index'])
    if not media.is_relative_to(DATA_ROOT.resolve()):
        raise ValueError("Batch review media must remain in RinkSpace_Data")
    point_paths = sorted(iter_project_assets(dataset / "frames", "[0-9][0-9][0-9][0-9][0-9][0-9].json", project_root=ROOT))
    if len(point_paths) != args.expected_frames:
        raise ValueError(f"Expected {args.expected_frames} authorized frame packets; found {len(point_paths)}")
    frames = []
    overlay_dir = media / "overlays"
    if not args.skip_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    first_index = int(point_paths[0].stem)
    for index, point_path in enumerate(point_paths, first_index):
        if point_path.stem != f"{index:06d}":
            raise ValueError("Frame packet indices must be contiguous and one-based")
        packet = read_project_json(point_path, project_root=ROOT)
        source_path = source_media / "frames" / f"{index:06d}.png"
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if source is None or source.shape != (1080, 1920, 3):
            raise ValueError(f"Frame {index} source image is not native 1920x1080")
        claimed_hash = packet.get("source", {}).get("decoded_BGR_sha256")
        if claimed_hash and hashlib.sha256(source.tobytes()).hexdigest() != claimed_hash:
            raise ValueError(f"Frame {index} PNG does not match packet source pixels")
        segments = list(source_segments(packet))
        count = sum(len(uv) for _, _, uv, _ in segments)
        if count != packet.get("point_count"):
            raise ValueError(f"Frame {index} saved point count does not match its actual vertices")
        overlay_path = overlay_dir / f"{index:06d}.png"
        if not args.skip_overlays and not cv2.imwrite(str(overlay_path), native_overlay(source, segments, packet.get('inferred_segments', []), packet=packet)):
            raise OSError(f"Could not write batch review overlay: {overlay_path}")
        navigation = Path(packet.get('source', {}).get('navigation_overlay_path', source_media / "navigation" / f"{index:06d}.png"))
        metadata = packet.get("source", {})
        pts = metadata.get("native_pts", metadata.get("actual_pts", metadata.get("pts", "未记录")))
        frames.append(dict(index=index, source=source_path.as_uri(),
            navigation=navigation.as_uri() if navigation.is_file() else None,
            overlay=overlay_path.as_uri() if overlay_path.is_file() else None,
            source_time=timestamp(metadata), pts=pts, time_base=metadata.get("time_base", "未记录"),
            point_count=count, counts={key:sum(len(uv) for layer, _, uv, _ in segments if layer == key) for key in LAYERS},
            status=("用户已审核通过" if index in accepted_indices else
                    "首帧：历史人工验收参考" if index == 1 else "等待用户验收"),
            notes=[str(region.get("note_zh", region.get("note", ""))) for region in packet.get("unresolved_source_regions", []) if region.get("note_zh", region.get("note"))],
            inferred_segments=packet.get('inferred_segments', []),
            segments=display_segments(packet)))
    manifest = dataset / "batch_manifest.json"
    if not manifest.is_file():
        manifest = media / "batch_manifest.json"
    page = args.html.resolve() if args.html else AssetStore(ROOT).review_page(dataset)
    if page.is_relative_to(ROOT) and not page.is_relative_to(ROOT / "outputs"):
        raise ValueError("Generated review HTML must stay outside public source paths")
    if page in point_paths:
        raise ValueError("Review HTML must not overwrite a point packet")
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(page_html(frames, manifest if manifest.is_file() else None), encoding="utf-8")
    return dict(review_html=str(page), frames=len(frames), total_points=sum(f["point_count"] for f in frames),
                navigation_frames=sum(bool(f["navigation"]) for f in frames), overlays_written=0 if args.skip_overlays else len(frames))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--media", type=Path, default=DEFAULT_MEDIA)
    parser.add_argument("--source-media", type=Path)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--expected-frames", type=int, default=100)
    parser.add_argument("--skip-overlays", action="store_true")
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
