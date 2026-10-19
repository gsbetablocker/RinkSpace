"""Refresh the sole project entry from owned models and existing source records.

No fitting, training, decoding, dependency installation, or video rendering.
The active chat owns CURRENT_TASK; retain its latest block when replacing HTML.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone, timedelta
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TASK = re.compile(r"<!-- CURRENT_TASK_BEGIN -->.*?<!-- CURRENT_TASK_END -->", re.S)
ENTRY = re.compile(r'<script[^>]*id="entry-data"[^>]*>(.*?)</script>', re.S)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def esc(value):
    return html.escape(str(value), quote=True)


def href(path: Path):
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_uri()


def link(path: Path, label=None):
    return f'<a href="{esc(href(path))}">{esc(label or path.name)}</a>'


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_svg(points, *, color="#216f82", width=.25, closed=False):
    if not points:
        return ""
    pairs = " ".join(f"{float(p[0]):.6f},{-float(p[1]):.6f}" for p in points)
    tag = "polygon" if closed else "polyline"
    return f'<{tag} points="{pairs}" fill="none" stroke="{color}" stroke-width="{width}"/>'


def plan(scene):
    """Use the exported model geometry; no schematic replacement or confidence mask."""
    boundary = scene.get("ice_boundary", {}).get("points_xyz", [])
    if not boundary:
        return '<p class="small">模型没有导出冰面轮廓；保留缺失状态。</p>'
    xs, ys = [float(p[0]) for p in boundary], [float(p[1]) for p in boundary]
    left, bottom = min(xs)-9, -max(ys)-9
    width, height = max(xs)-min(xs)+18, max(ys)-min(ys)+18
    parts = [f'<svg class="rink-plan" viewBox="{left} {bottom} {width} {height}" role="img" aria-label="该版本实际冰场标记和已建模广告">']
    parts += [path_svg(c.get("polygon_xyz", []), color="#d6e4e9", width=.12, closed=True)
              for c in scene.get("ice_grid", {}).get("cells", [])]
    parts.append(path_svg(boundary, color="#64798c", width=.5, closed=True))
    for feature in scene.get("ice_features", []):
        label = feature.get("id", "?")
        if feature.get("kind") == "circle":
            rings = feature.get("rings") or [feature]
            for ring in rings:
                center = ring.get("center_xyz", feature.get("center_xyz"))
                if center is not None and ring.get("radius") is not None:
                    parts.append(f'<circle cx="{float(center[0])}" cy="{-float(center[1])}" r="{float(ring["radius"])}" fill="none" stroke="#33859a" stroke-width=".24"/>')
        else:
            segments = feature.get("paint_segments_xyz", feature.get("segments_xyz", []))
            if segments:
                parts += [path_svg(p) for p in segments]
            elif feature.get("polygon_xyz"):
                parts.append(path_svg(feature["polygon_xyz"], closed=bool(feature.get("closed"))))
            if feature.get("centerline_xyz"):
                parts.append(path_svg(feature["centerline_xyz"], color="#447691"))
        center = feature.get("center_xyz")
        if center:
            parts.append(f'<text x="{float(center[0])+.7}" y="{-float(center[1])-.7}" font-size="2.4" fill="#346373">{esc(label)}</text>')
    for ad in scene.get("advertisements", []):
        curves = ad.get("fitted_boundary_curves_xyz", {})
        parts += [path_svg(p, color="#bf7640", width=.45) for p in curves.values() if isinstance(p, list)]
        corners = list(ad.get("corners_xyz_scene_units", {}).values())
        if corners:
            x = sum(p[0] for p in corners)/len(corners)
            y = sum(p[1] for p in corners)/len(corners)
            parts.append(f'<text x="{x}" y="{-y-1.2}" text-anchor="middle" font-size="2.5" fill="#925d31">Ad{esc(ad.get("user_number", "?"))}</text>')
    return "".join(parts)+"</svg>"


def model_panel(scene_path: Path, title: str):
    scene = read(scene_path)
    counts = f'{len(scene.get("ice_features", []))} 个冰面标记 · {len(scene.get("advertisements", []))} 个有几何的广告'
    rows = []
    for name, value in scene.get("parameters", {}).items():
        rows.append(f'<tr><td>{esc(name)}</td><td><pre>{esc(json.dumps(value, ensure_ascii=False, indent=2))}</pre></td></tr>')
    cameras = list(scene_path.parent.glob("*camera*.npz"))
    files = " · ".join([link(scene_path, "完整场地数据")]+[link(p, "相机模型参数") for p in cameras])
    return (f'<details class="surface model"><summary><strong>{esc(title)}</strong> · {esc(counts)} · 未通过完整定位验收</summary>'
            f'<p class="small">场景 SHA {sha(scene_path)[:16]}；以下投影来自此文件的实际几何。米制仍需独立验证，未建模广告不补造。</p>'
            +plan(scene)+f'<p>{files}</p><details><summary>场地参数与模型定义</summary><table><thead><tr><th>参数</th><th>当前值／状态</th></tr></thead><tbody>'
            +"".join(rows)+f'</tbody></table><p class="small">单项固定／自由状态和逐曝光相机由相应运行契约定义；本页面不把导出字段数量当成求解自由度。</p></details></details>')


def source_time(pts, base):
    seconds = int(pts)*base
    hours = int(seconds//3600)
    minutes = int(seconds%3600//60)
    return f"{hours:02d}:{minutes:02d}:{seconds%60:06.3f}"


def programme(event: Path, entry):
    directory = event/"programmes/okada_mei_sp"
    latest = Path(entry["full_video_path"])
    num, den = map(int, entry["source_time_base"].split("/"))
    rows = []
    names = {"main_long_side":"长边主机位", "secondary_long_side":"长边副机位", "low_end":"低位端部"}
    for shot in entry["shots"]:
        status = "存在已知偏差" if shot["validation"] == "known_issue" else "尚未通过完整镜头验收"
        rows.append(f'<li class="shot"><div class="time">{source_time(shot["start_pts"], num/den)}<span class="to">至 {source_time(shot["end_pts_exclusive"], num/den)}</span></div><div class="shot-body"><strong>镜头 {shot["shot_index"]} · {esc(names.get(shot["camera_id"], shot["camera_id"]))}</strong><p>{status} · {shot["frame_count"]} 次原片显示</p></div></li>')
    files = sorted(directory.rglob("*.npz"))+sorted(directory.rglob("FROZEN*.json"))
    retained = " · ".join(link(p) for p in files)
    observations = ROOT/"configs/rink_vision.json"
    return ('<section class="section entry-view" id="programme" data-view="programme" hidden><h1 tabindex="-1">冈田芽衣 · 短节目</h1>'
            '<p>安卡拉赛事的开发节目 · 原片 04:08:36.283–04:11:25.000（结束不含）</p>'
            '<div class="surface"><strong>最新已有完整工作版 · 2026-10-05</strong><p>9 镜头，全部已建模网格、广告编号与骨盆投影；贴合和世界位置仍未验收。用 PotPlayer 播放。</p>'
            f'<div class="path-row"><code class="path">{esc(latest)}</code><button class="copy" data-copy-path="{esc(latest)}">复制视频路径</button></div><p class="copy-status small" role="status"></p>'
            f'<details><summary>节目模型数据与来源</summary><p>{retained or "正在登记模型数据"}</p><p><code>{esc(directory)}</code> · {link(Path(entry["current_video_delivery"]["frozen_inputs_path"]), "原始冻结来源")}</p></details></div>'
            '<div class="program-layout"><div><h3>完整源时间轴</h3><ol class="timeline">'+"".join(rows)+'</ol></div><div><h3>已知问题与核验依据</h3><p>冰面漆线、Lane、靶圈与广告仍有明显偏差；旋转段曾出现共同投影漂移。身体位置和接触不能仅凭轨迹平滑判定准确。</p>'
            f'<p>{link(directory/"model_constraints/user_work_priorities_20261005.json", "八处人工指出的问题与工程取舍")} · {link(observations, "当前视觉样本配置")}</p>'
            '<p class="small">此完整视频使用 10 月 5 日冻结模型，与当前工作候选分别绑定。模型更新不会把旧视频自动升级为新成果。</p></div></div></section>')


def content(entry, current_task):
    event = ROOT/"events/ankara_2026"
    manifest = read(event/"manifest.json")
    resources = {r["id"]:ROOT/r["project_path"] for r in manifest["resources"]}
    models = model_panel(resources["working_scene"], "工作候选 · "+manifest["working_model"]["q_sha256"][:12])
    models += model_panel(resources["registered_v025_scene"], "正式登记基线 · v0.25")
    constraints = sorted((event/"model_constraints").rglob("*.json"))
    links = " · ".join(link(p) for p in constraints)
    return ('<section id="project" class="entry-view" data-view="project"><h1 tabindex="-1">花样滑冰转播的空间分析</h1><p class="lead">为不同赛事建立各自的三维冰场与相机模型，持续定位场地和选手，恢复轨迹并支持运动分析。</p>'
            +current_task
            +'<section class="section"><h2>赛事</h2><a class="surface entry-link" href="#/events/ankara_2026"><strong>2026 JGP 安卡拉</strong><span>查看冰场、广告、机位、共同模型与节目结果 →</span></a>'
            '<p class="small">已有开发赛事；跨赛事能力仍需对应真实画面验证。</p></section>'
            '<details class="section surface" id="folders"><summary>公共能力与文件归属</summary><p>公共代码、环境、最终视觉参数和求解接口留在 OneDrive。场地、广告和相机安装归所属赛事；节目保存源时间、模型绑定及自己的结果。大量原片、帧图、数组、检查点和日志保存在 Data。</p><p>'
            +link(ROOT/"README.md", "项目与技术说明")+'</p></details></section>'
            '<section class="section entry-view" id="event" data-view="event" hidden><h1 tabindex="-1">2026 JGP 安卡拉</h1><p class="lead">共同冰场、广告与机位模型，同赛事节目共享。</p>'
            '<section class="section"><h2>节目与结果</h2><a class="surface entry-link" href="#/events/ankara_2026/programmes/okada_mei_sp"><strong>冈田芽衣 · 短节目</strong><span>完整网格视频、源时间轴、轨迹与待修偏差 →</span></a></section>'
            '<section class="section"><h2>场地与相机模型</h2>'
            +models+f'<details class="surface"><summary>机位、广告与人工约束</summary><p>{links}</p><p>{link(event/"manifest.json", "文件身份、角色与原始来源")}</p><p class="small">这些事实仅适用于本赛事，不成为所有花滑冰场的默认布局或设备。</p></details></section>'
            +'</section>'+programme(event, entry))


JS = r'''<script>
(()=>{
const routes={
  '#/':{view:'project',title:'RinkSpace',crumbs:[['RinkSpace','#/']]},
  '#/events/ankara_2026':{view:'event',title:'2026 JGP 安卡拉 · RinkSpace',crumbs:[['RinkSpace','#/'],['2026 JGP 安卡拉','#/events/ankara_2026']]},
  '#/events/ankara_2026/programmes/okada_mei_sp':{view:'programme',title:'冈田芽衣 · 短节目 · RinkSpace',crumbs:[['RinkSpace','#/'],['2026 JGP 安卡拉','#/events/ankara_2026'],['冈田芽衣 · 短节目','#/events/ankara_2026/programmes/okada_mei_sp']]}
};
function navigate(){
  const original=location.hash;
  const aliases={'#project':'#/','#event':'#/events/ankara_2026','#programme':'#/events/ankara_2026/programmes/okada_mei_sp','#current-task':'#/','#folders':'#/'};
  const route=routes[aliases[original]||original]||routes['#/'];
  document.querySelectorAll('[data-view]').forEach(section=>{section.hidden=section.dataset.view!==route.view;});
  document.title=route.title;
  const breadcrumb=document.getElementById('entry-breadcrumb');breadcrumb.replaceChildren();
  route.crumbs.forEach(([label,target],index)=>{if(index){const divider=document.createElement('span');divider.textContent=' / ';divider.setAttribute('aria-hidden','true');breadcrumb.append(divider);}const item=document.createElement(index===route.crumbs.length-1?'span':'a');item.textContent=label;if(item.tagName==='A')item.href=target;else item.setAttribute('aria-current','page');breadcrumb.append(item);});
  const active=document.querySelector('[data-view="'+route.view+'"]');
  if(original==='#current-task'||original==='#folders'){const target=document.querySelector(original);if(target){if(target.tagName==='DETAILS')target.open=true;target.scrollIntoView();}}
  else {window.scrollTo(0,0);active.querySelector('h1')?.focus({preventScroll:true});}
}
window.addEventListener('hashchange',navigate);navigate();
})();
document.querySelectorAll('[data-copy-path]').forEach(b=>b.addEventListener('click',async()=>{let ok=false;try{await navigator.clipboard.writeText(b.dataset.copyPath);ok=true}catch{let t=document.createElement('textarea');t.value=b.dataset.copyPath;document.body.append(t);t.select();ok=document.execCommand('copy');t.remove()}let s=b.closest('.surface')?.querySelector('.copy-status');if(s)s.textContent=ok?'路径已复制，可粘贴到 PotPlayer。':'请选中路径复制。'}));
if(location.protocol!=='file:')document.querySelectorAll('a[href],img[src]').forEach(el=>{let attr=el.tagName==='IMG'?'src':'href';let h=el.getAttribute(attr);if(h&&!h.startsWith('#')&&!/^[a-z]+:/i.test(h)&&!h.startsWith('/'))el.setAttribute(attr,'/assets/project/'+h);else if(h&&h.startsWith('file:///D:/CodexData/RinkSpace_Data/'))el.setAttribute(attr,'/data/'+h.slice('file:///D:/CodexData/RinkSpace_Data/'.length));});
</script>'''


def replace_capturing_previous(target: Path, replacement: Path, backup: Path):
    """Atomically replace a file and retain exactly the displaced version."""
    if os.name == "nt":
        replace = ctypes.WinDLL("kernel32", use_last_error=True).ReplaceFileW
        replace.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_wchar_p,
                            ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p]
        replace.restype = ctypes.c_int
        if not replace(str(target), str(replacement), str(backup), 0, None, None):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        # An atomic exchange puts the displaced inode at the replacement path.
        # Unlike copying before rename, this captures a concurrent replacement.
        libc = ctypes.CDLL(None, use_errno=True)
        target_bytes, replacement_bytes = os.fsencode(target), os.fsencode(replacement)
        if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
            exchange = libc.renameat2
            exchange.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                                 ctypes.c_char_p, ctypes.c_uint]
            exchange.restype = ctypes.c_int
            result = exchange(-100, target_bytes, -100, replacement_bytes, 2)
        elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
            exchange = libc.renamex_np
            exchange.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            exchange.restype = ctypes.c_int
            result = exchange(target_bytes, replacement_bytes, 2)
        else:
            raise RuntimeError("Atomic file exchange is unavailable; entry was not replaced")
        if result != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(target))
        os.replace(replacement, backup)
    return backup.read_bytes()


def refresh():
    target = ROOT/"START_HERE.html"
    store = read(ROOT/"configs/storage.json")
    temp_root = Path(store["data_root"])/"tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    original = target.read_bytes()
    expected = original
    backups = []
    conflicts = 0
    while True:
        source = original.decode("utf-8-sig")
        current, data = TASK.search(source), ENTRY.search(source)
        if current is None or data is None:
            raise ValueError("Active task and legacy result bindings must remain present")
        head = source.split("</head>", 1)[0]
        head = re.sub(r'<!-- PROJECT_ENTRY_STYLE_BEGIN -->.*?<!-- PROJECT_ENTRY_STYLE_END -->', '', head, flags=re.S)
        head += '<!-- PROJECT_ENTRY_STYLE_BEGIN --><style>.rink-plan{width:100%;display:block;max-height:450px;background:#fff}.model{margin-top:12px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;margin:6px 0}table{width:100%;border-collapse:collapse}td,th{text-align:left;vertical-align:top;padding:8px;border-bottom:1px solid #e1e9ec}.model>summary{font-size:15px}details.surface>summary{list-style-position:inside}.entry-view[hidden]{display:none!important}.entry-view>h1:focus{outline:none}.entry-link{display:flex;justify-content:space-between;align-items:center;gap:16px;text-decoration:none;margin-top:10px}.entry-link strong{font-size:18px}.entry-link span{font-size:14px;color:var(--muted)}.entry-link:hover{border-color:var(--accent)}#entry-breadcrumb{display:block;font-size:14px}header>a{font-size:23px;font-weight:700;text-decoration:none;color:var(--ink)}@media(max-width:650px){.entry-link{align-items:flex-start;flex-direction:column}}</style><!-- PROJECT_ENTRY_STYLE_END --></head>'
        sections = content(json.loads(data.group(1)), current.group())
        body = ('<body><main><header><a href="#/">RinkSpace</a><nav id="entry-breadcrumb" aria-label="当前位置">RinkSpace</nav></header>'
                +sections
                +'<footer>唯一入口：START_HERE.html · <a href="README.md">项目说明</a> · <a href="calibration/solver_runtime/README.md">求解接口</a> · <a href="CHECK_PATHS.cmd">OneDrive 路径检查</a></footer>'
                +'<script type="application/json" id="entry-data">'+data.group(1)+'</script>'+JS+'</main></body></html>')
        output = (head+body).encode("utf-8")
        with tempfile.NamedTemporaryFile(dir=temp_root, suffix=".entry.html", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(output)
        with tempfile.NamedTemporaryFile(dir=temp_root, suffix=".entry.previous.html", delete=False) as handle:
            backup = Path(handle.name)
        backup.unlink()
        backups.append(backup)
        displaced = replace_capturing_previous(target, temporary, backup)
        if displaced != expected:
            # A writer updated the entry during rendering. Its complete version
            # is now in backup; use that CURRENT_TASK and entry-data next time.
            # Our own output is the expected target until another writer acts.
            original, expected = displaced, output
            conflicts += 1
            continue
        for saved in backups:
            saved.unlink(missing_ok=True)
        return {"entry":str(target), "task_block_preserved":True,
                "result_binding_preserved":True, "bytes":len(output),
                "concurrent_versions_reconciled":conflicts}


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(refresh()))
