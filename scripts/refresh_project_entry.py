"""Generate the local results page from the event's designated programme fits."""
from __future__ import annotations

import argparse
import ctypes
import html
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rinkspace.storage import AssetStore

EVENT = Path("events/ankara_2026")
PROGRAMMES = (("okada_mei_sp", "冈田芽依"), ("jin_shuxian_sp", "金书贤"))
PANEL = "<!-- RINKSPACE_RESULT_PANEL -->"
REVIEW = re.compile(r"<!-- CURRENT_TASK_BEGIN -->.*?<!-- CURRENT_TASK_END -->", re.S)


def esc(value):
    return html.escape(str(value), quote=True)


def link(path, label):
    return f'<a href="{esc(Path(path).relative_to(ROOT).as_posix())}">{esc(label)}</a>'


def current_result(directory):
    candidates = []
    for path in directory.rglob("provisional_final_*.json"):
        record = json.loads(path.read_text(encoding="utf-8-sig"))
        designation = record.get("designation", {})
        if (record.get("schema") == "rinkspace.provisional_final_programme_fit.v1"
                and record.get("programme_id") == directory.name
                and designation.get("status") == "provisional_final"):
            candidates.append((designation["designated_on"], path, record))
    if not candidates:
        return None
    newest_date = max(candidate[0] for candidate in candidates)
    latest = [candidate for candidate in candidates if candidate[0] == newest_date]
    if len(latest) != 1:
        raise ValueError(f"Multiple designated programme fits on {newest_date}: {directory}")
    return latest[0]


def programme_section(store, key, name):
    directory = ROOT / EVENT / "programmes" / key
    binding = current_result(directory)
    if binding is None:
        result = '<p>暂未登记完整拟合。</p>'
    else:
        date, record_path, record = binding
        review = record["full_review"]
        first, last = review["original_source_seconds"]
        video_path = Path(review["video"]["path"])
        video_path = store.resolve(video_path if video_path.is_absolute() else store.data / video_path)
        result = (
            f'<p>{esc(date)} 暂定最终版 <strong>{esc(record["version_id"])}</strong>'
            f' · {record["shot_count"]} 个镜头 · {record["source_native_PTS"]["exposures"]:,} 个原生帧。</p>'
            f'<p>原片 {first:.3f}–{last:.3f} 秒。视觉验收见节目页及登记绑定的审阅记录。</p>'
            f'<p>{link(record_path, "版本登记")}</p>'
        )
        if video_path.is_file():
            result += (
                '<div class="surface"><p>本地冰场回放地址，供 PotPlayer 使用</p>'
                f'<code>{esc(video_path)}</code> '
                f'<button type="button" data-copy-path="{esc(video_path)}">复制路径</button>'
                '<p class="copy-status" role="status"></p></div>'
            )
        else:
            result += '<p>本地缺少该版整片回放文件。</p>'
    if (directory / "README.md").is_file():
        result += f'<p>{link(directory / "README.md", "节目范围、结果与人工确认")}</p>'
    return (
        f'<section class="entry-view" id="{esc(key)}" data-view="{esc(key)}" hidden>'
        f'<h1 tabindex="-1">{esc(name)}短节目</h1>{result}</section>'
    )


def render(store, previous):
    old_reviews = REVIEW.search(previous) if PANEL in previous else None
    reviews = old_reviews.group() if old_reviews else "<!-- CURRENT_TASK_BEGIN --><!-- CURRENT_TASK_END -->"
    programme_links = "".join(
        f'<li><a href="#/events/ankara_2026/programmes/{key}">{esc(name)}短节目</a></li>'
        for key, name in PROGRAMMES
    )
    sections = "".join(programme_section(store, key, name) for key, name in PROGRAMMES)
    event_docs = " · ".join(
        link(path, label) for path, label in (
            (ROOT / EVENT / "README.md", "赛事版本与来源"),
            (ROOT / EVENT / "scene/README.md", "共同冰场与广告"),
            (ROOT / "docs/RESULTS_3D_PLAYER_ZH.md", "人体与鞋刀播放器"),
        ) if path.is_file()
    )
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>RinkSpace</title><style>'
        'body{margin:0;background:#f5f7f9;color:#233748;font:16px/1.65 system-ui,sans-serif}'
        'main{max-width:60rem;margin:auto;padding:1.5rem}a{color:#08687a}h1{line-height:1.2}'
        'header{display:flex;gap:1rem;align-items:baseline}header>a{font-weight:700}'
        '.surface{padding:1rem;border:1px solid #dbe4eb;border-radius:.5rem;background:#fff}'
        'code{overflow-wrap:anywhere}details{margin-top:1rem}button{font:inherit;cursor:pointer}'
        '[hidden]{display:none!important}</style></head><body>' + PANEL
        + '<main><header><a href="#/">RinkSpace</a><nav id="entry-breadcrumb" aria-label="当前位置"></nav></header>'
        '<section class="entry-view" data-view="project"><h1 tabindex="-1">花样滑冰转播的空间分析</h1>'
        '<p><a href="#/events/ankara_2026">2026 JGP 安卡拉</a></p>'
        f'<p>{link(ROOT / "README.md", "项目说明")}</p></section>'
        '<section class="entry-view" data-view="event" hidden><h1 tabindex="-1">2026 JGP 安卡拉</h1>'
        f'<ul>{programme_links}</ul><p>{event_docs}</p>'
        f'<details id="source-reviews"><summary>历史片段取点审阅</summary>{reviews}</details></section>'
        + sections
        + JS.replace("__PROJECT_URI__", json.dumps(ROOT.as_uri())).replace("__DATA_URI__", json.dumps(store.data.as_uri()))
        + '</main></body></html>'
    )


JS = r'''<script>
(()=>{
const eventPath='#/events/ankara_2026', okadaPath=eventPath+'/programmes/okada_mei_sp';
const routes={
  '#/':{view:'project',title:'RinkSpace',crumbs:[['RinkSpace','#/']]},
  [eventPath]:{view:'event',title:'2026 JGP 安卡拉',crumbs:[['RinkSpace','#/'],['2026 JGP 安卡拉',eventPath]]},
  [okadaPath]:{view:'okada_mei_sp',title:'冈田芽依短节目',crumbs:[['RinkSpace','#/'],['2026 JGP 安卡拉',eventPath],['冈田芽依短节目',okadaPath]]},
  [eventPath+'/programmes/jin_shuxian_sp']:{view:'jin_shuxian_sp',title:'金书贤短节目',crumbs:[['RinkSpace','#/'],['2026 JGP 安卡拉',eventPath],['金书贤短节目',eventPath+'/programmes/jin_shuxian_sp']]}
};
const aliases={'#project':'#/','#event':eventPath,'#programme':okadaPath,'#current-task':eventPath};
function navigate(){
  const hash=location.hash;
  const route=routes[aliases[hash]||hash]||routes['#/'];
  document.querySelectorAll('[data-view]').forEach(section=>{section.hidden=section.dataset.view!==route.view;});
  document.title=route.title;
  const breadcrumb=document.getElementById('entry-breadcrumb');
  breadcrumb.replaceChildren();
  route.crumbs.forEach(([label,target],index)=>{
    if(index)breadcrumb.append(' / ');
    const item=document.createElement(index===route.crumbs.length-1?'span':'a');
    item.textContent=label;
    if(item.tagName==='A')item.href=target;else item.setAttribute('aria-current','page');
    breadcrumb.append(item);
  });
  window.scrollTo(0,0);
  document.querySelector('[data-view="'+route.view+'"] h1')?.focus({preventScroll:true});
}
window.addEventListener('hashchange',navigate);
navigate();
})();
document.querySelectorAll('[data-copy-path]').forEach(button=>button.addEventListener('click',async()=>{
  const status=button.closest('.surface').querySelector('.copy-status');
  try {
    await navigator.clipboard.writeText(button.dataset.copyPath);
    status.textContent='路径已复制，可粘贴到 PotPlayer。';
  } catch {
    status.textContent='请选中路径复制。';
  }
}));
const projectRoot=__PROJECT_URI__+'/', dataRoot=__DATA_URI__+'/';
document.querySelectorAll('a[href],img[src]').forEach(el=>{
  const attr=el.tagName==='IMG'?'src':'href', h=el.getAttribute(attr);
  if(!h||h.startsWith('#')||h.startsWith('/')||(/^[a-z]+:/i.test(h)&&!h.startsWith('file:')))return;
  const url=new URL(h,projectRoot).href;
  if(location.protocol==='file:')el.setAttribute(attr,url);
  else if(url.startsWith(projectRoot))el.setAttribute(attr,'/assets/project/'+url.slice(projectRoot.length));
  else if(url.startsWith(dataRoot))el.setAttribute(attr,'/assets/data/'+url.slice(dataRoot.length));
});
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
    store = AssetStore(ROOT)
    target = store.project_entry()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        original = target.read_bytes()
    except FileNotFoundError:
        output = render(store, "").encode("utf-8")
        try:
            with target.open("xb") as handle:
                handle.write(output)
            return {"entry": str(target), "bytes": len(output)}
        except FileExistsError:
            original = target.read_bytes()

    expected = original
    temp_root = store.data / "tmp" if store.configured else store.project / "outputs/tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    while True:
        output = render(store, original.decode("utf-8-sig")).encode("utf-8")
        if output == expected:
            return {"entry": str(target), "bytes": len(output)}
        with tempfile.NamedTemporaryFile(dir=temp_root, suffix=".entry.html", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(output)
        backup = temporary.with_suffix(".previous.html")
        try:
            displaced = replace_capturing_previous(target, temporary, backup)
            if displaced == expected:
                return {"entry": str(target), "bytes": len(output)}
            # A source-review writer saved an update during rendering.
            original, expected = displaced, output
        finally:
            temporary.unlink(missing_ok=True)
            backup.unlink(missing_ok=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(refresh()))
