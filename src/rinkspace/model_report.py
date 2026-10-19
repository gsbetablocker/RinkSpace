"""Portable original-video/model review, with an explicit derived-file whitelist.

The report reuses the workbench projection code. It packages supplied compact
proxies, never reads or copies a model source's local_path, retains every input
localization row, and does not promote any scientific acceptance status.
"""
from __future__ import annotations

import copy
from fractions import Fraction
import html
import json
from pathlib import Path
import re
import shutil

from .io import dump_json, load_json, sha256


SCHEMA = 'rinkspace.model-grid-review.v1'
IMAGE_TYPES = {'.jpg', '.jpeg', '.png', '.webp'}
MAX_PROXY_BYTES = 128 * 1024 ** 2
MAX_IMAGE_BYTES = 12 * 1024 ** 2


def _pts(row, prefix=''):
    value = row.get(prefix + 'pts', row.get('source_pts') if not prefix else None)
    base = row.get(prefix + 'time_base', row.get('source_time_base') if not prefix else None)
    if isinstance(value, bool) or not isinstance(value, int) or not isinstance(base, (str, int)):
        raise ValueError(f'Exact {prefix}PTS and rational time_base are required')
    factor = Fraction(base)
    if factor <= 0:
        raise ValueError('time_base must be positive')
    if not prefix and 'pts' in row and 'source_pts' in row:
        source_value, source_base = row['source_pts'], row.get('source_time_base')
        if (isinstance(source_value, bool) or not isinstance(source_value, int)
                or not isinstance(source_base, (str, int))
                or source_value * Fraction(source_base) != value * factor):
            raise ValueError('Source PTS aliases disagree')
    return value, str(base), value * factor


def _rows(record):
    rows = record if isinstance(record, list) else record.get('frames')
    if not isinstance(rows, list) or not rows:
        raise ValueError('A nonempty complete localization / PTS frame list is required')
    return rows


def _input(path, base, *, suffix=None, size=None):
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(base.resolve()) or not resolved.is_file() or path.is_symlink():
        raise ValueError(f'Input must stay inside its derived run directory: {path}')
    if suffix and resolved.suffix.lower() not in suffix:
        raise ValueError(f'Unsupported derived asset type: {path}')
    if size and resolved.stat().st_size > size:
        raise ValueError(f'Derived asset exceeds size guard; export a compact proxy or image first: {path}')
    return resolved


def _json_script(value):
    # A model label must not be able to terminate the inert JSON script element.
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')


def _compact_browser_diagnostics(display):
    """Trim only an isolated display copy; linked frame JSON remains complete.

    Failed frames cannot draw a grid, so their drawing masks need not occupy
    the HTML. Geometry, PTS, source identity, statuses, errors and provenance
    stay in every display row; successful-frame exclusion masks stay exact.
    """
    configurations = []
    keys = {}
    for row in display['frames']:
        if 'foreground_config' in row:
            config = row.pop('foreground_config')
            key = json.dumps(config, sort_keys=True, separators=(',', ':'))
            if key not in keys:
                keys[key] = len(configurations)
                configurations.append(config)
            row['foreground_config_index'] = keys[key]
        if row.get('status') == 'unavailable' and row.get('H_model_from_image') is None:
            row.pop('foreground_polygons_image', None)
            row.pop('person_boxes_image', None)
        for attempt in row.get('reobservation_attempts', []):
            for key in list(attempt):
                if key not in {'anchor_id', 'status', 'reason', 'camera_proposal_inliers', 'proposal_role'}:
                    attempt.pop(key)
    if configurations:
        display['foreground_configs'] = configurations
        if len(configurations) == 1:
            for row in display['frames']:
                row.pop('foreground_config_index', None)
    display['browser_diagnostics_scope'] = 'Compact display summary; full optimizer diagnostics and all masks remain in frames_file.'
    return display


def build_model_delivery(model_dir, run_dirs, out_dir):
    """Create an offline report from one frozen model and complete proxy runs.

    Each run contains localization.json, source.mp4, source_pts_map.json and a
    selected_window with a pre-result selection_basis in localization.json.
    Proxy decoding/ZIP verification is performed by export_model_delivery.py;
    this builder checks identity and exact PTS joins before it copies anything.
    """
    model_dir = Path(model_dir).resolve(strict=True)
    directories = [Path(directory).resolve(strict=True) for directory in run_dirs]
    out = Path(out_dir).resolve()
    if not directories:
        raise ValueError('At least one complete programme run is required')
    if any(directory == out or directory.is_relative_to(out) for directory in [model_dir, *directories]):
        raise ValueError('Report output cannot replace a model or localization input directory')
    model_path = _input(model_dir / 'rink_model.json', model_dir, suffix={'.json'})
    model = load_json(model_path)
    identity = (model.get('model_id'), model.get('revision'))
    if not all(isinstance(value, str) and value for value in identity):
        raise ValueError('Model ID and revision must be nonempty strings')
    staged_runs = []
    used_ids = set()
    for directory in directories:
        record = copy.deepcopy(load_json(_input(directory / 'localization.json', directory, suffix={'.json'})))
        if not isinstance(record, dict):
            raise ValueError('localization.json needs run metadata and frames')
        rows = _rows(record)
        mapping = copy.deepcopy(load_json(_input(directory / 'source_pts_map.json', directory, suffix={'.json'})))
        mapped = _rows(mapping)
        proxy = _input(directory / 'source.mp4', directory, suffix={'.mp4'}, size=MAX_PROXY_BYTES)
        source = record.get('source_sha256') or rows[0].get('source_sha256')
        if not isinstance(source, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', source):
            raise ValueError(f'Programme needs the original source SHA-256: {directory.name}')
        if mapping.get('source_sha256') != source:
            raise ValueError('Proxy map source SHA-256 disagrees with localization')
        proxy_hash = sha256(proxy)
        if proxy_hash.lower() == source.lower():
            raise ValueError('Original source media cannot be copied as a derived proxy')
        if mapping.get('proxy_sha256') != proxy_hash:
            raise ValueError('Proxy SHA-256 does not match its source PTS map')
        if len(rows) != len(mapped):
            raise ValueError('Complete localization and proxy PTS frame counts differ')
        previous_time = None
        first_time = _pts(rows[0])[2]
        proxy_first = _pts(mapped[0], 'proxy_')[2]
        for row, point in zip(rows, mapped):
            for key, expected in [('model_id', identity[0]), ('model_revision', identity[1]), ('source_sha256', source)]:
                actual = row.get(key, record.get(key, record.get('revision') if key == 'model_revision' else None))
                if actual != expected:
                    raise ValueError(f'Frame model/source identity mismatch: {directory.name} / {key}')
                row[key] = actual
            pts, base, time = _pts(row)
            map_pts, map_base, map_time = _pts(point)
            _, _, proxy_time = _pts(point, 'proxy_')
            if time != map_time or time-first_time != proxy_time-proxy_first:
                raise ValueError('Proxy mapping must retain every exact relative source PTS')
            if previous_time is not None and time <= previous_time:
                raise ValueError('Source PTS must remain strictly increasing')
            previous_time = time
            row.update(pts=pts, time_base=base)
            point.update(pts=map_pts, time_base=map_base)
            if row.get('status') not in {'observed', 'predicted', 'candidate', 'unavailable', 'ambiguous'}:
                raise ValueError('Unrecognized localization status in complete frame list')
            if row['status'] in {'unavailable', 'ambiguous'} and row.get('H_model_from_image') is not None:
                raise ValueError('Unavailable or ambiguous frames must not retain a model grid matrix')
        window = copy.deepcopy(record.get('selected_window', {}))
        if not isinstance(window.get('selection_basis'), str) or not window['selection_basis'].strip():
            raise ValueError('localization.selected_window.selection_basis must describe the complete window chosen before localization')
        expected = {'first_pts': rows[0]['pts'], 'last_pts': rows[-1]['pts'],
                    'time_base': rows[0]['time_base'], 'frame_count': len(rows)}
        for key, value in expected.items():
            if key in window and window[key] != value:
                raise ValueError(f'Selected window {key} disagrees with complete supplied frames')
            window[key] = value
        programme_id = record.get('run_id') or directory.name
        if not isinstance(programme_id, str) or not programme_id or programme_id in used_ids:
            raise ValueError('Programme run IDs must be nonempty and unique')
        used_ids.add(programme_id)
        staged_runs.append((directory, record, mapping, proxy, source, window, programme_id))

    out.mkdir(parents=True, exist_ok=True)
    files = []

    def copy_asset(source, target):
        destination = out / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        files.append(target)
        return target

    def write_record(target, value):
        dump_json(out / target, value)
        files.append(target)

    model_file = copy_asset(model_path, 'model/rink_model.json')
    portable_model = copy.deepcopy(model)
    image_paths = {}
    missing_images = []
    # Images may live in another derived output directory, but never the raw
    # source tree. The model's reference list is the explicit image whitelist.
    evidence_root = next((parent for parent in model_dir.parents if parent.name == 'outputs'), model_dir)

    def copy_image(relative):
        if not isinstance(relative, str) or not relative:
            return None
        if relative in image_paths:
            return image_paths[relative]
        source = (model_dir / relative).resolve()
        if (not source.is_relative_to(evidence_root) or source.suffix.lower() not in IMAGE_TYPES
                or not source.is_file() or source.stat().st_size > MAX_IMAGE_BYTES):
            missing_images.append(relative)
            return None
        target = 'images/' + sha256(source)[:16] + source.suffix.lower()
        if target not in files:
            copy_asset(source, target)
        image_paths[relative] = target
        return target

    portable_model['overview_url'] = copy_image(model.get('overview_path'))
    for view in portable_model.get('reference_views', []):
        view['image_url'] = copy_image(view.get('image_path'))
    for landmark in portable_model.get('landmarks', []):
        for observation in landmark.get('observations', []):
            observation['image_url'] = copy_image(observation.get('image_path'))

    programmes, display_runs = [], []
    for index, (directory, record, mapping, proxy, source, window, programme_id) in enumerate(staged_runs):
        prefix = f'programmes/{index + 1:02d}'
        frame_file, map_file, proxy_file = prefix + '/localization.json', prefix + '/source_pts_map.json', prefix + '/source.mp4'
        write_record(frame_file, record)
        write_record(map_file, mapping)
        copy_asset(proxy, proxy_file)
        programmes.append({'programme_id': programme_id, 'source_sha256': source,
            'frames_file': frame_file, 'pts_map_file': map_file, 'proxy_file': proxy_file,
            'selected_window': window})
        display = copy.deepcopy(record)
        display.update(programme_id=programme_id, source_sha256=source, model_id=identity[0],
            model_revision=identity[1], label=record.get('label') or
                f"{record.get('source_id',programme_id)} · {float(_pts(record['frames'][0])[2]):.2f}–{float(_pts(record['frames'][-1])[2]):.2f}s"+
                (' · 冻结后新来源推理' if record.get('role')=='new_source_inference_after_freeze' else ' · 开发/回归'),
            proxy_file=proxy_file, frames_file=frame_file, pts_map_file=map_file, selected_window=window)
        for row, point in zip(display['frames'], mapping['frames']):
            row['media_time_s'] = float(_pts(point, 'proxy_')[2])
            row['source_time_s'] = float(_pts(row)[2])
        display_runs.append(_compact_browser_diagnostics(display))
    static = Path(__file__).with_name('static')
    for source, target in [('model_grid.js', 'geometry.js'), ('model_delivery.js', 'report.js'), ('model_delivery.css', 'report.css')]:
        copy_asset(static / source, target)
    data = {'model': portable_model, 'runs': display_runs, 'missing_reference_images': sorted(set(missing_images)),
            'geometry_independently_validated': False}
    options = ''.join(f'<option value="{index}">{html.escape(run["label"])}</option>' for index, run in enumerate(display_runs))
    assets = ''.join(f'<a href="{html.escape(name)}">{html.escape(name)}</a>' for name in files if name.endswith(('.json', '.mp4')))
    page = _PAGE.replace('__OPTIONS__', options).replace('__ASSETS__', assets).replace('__DATA__', _json_script(data))
    (out / 'index.html').write_text(page, encoding='utf-8')
    files.append('index.html')
    manifest = {'schema_version': SCHEMA, 'entrypoint': 'index.html', 'files': sorted(set(files)),
        'model_file': model_file, 'model_sha256': sha256(model_path), 'programmes': programmes,
        'independent_rink_accuracy_validated': False,
        'missing_reference_images': sorted(set(missing_images)),
        'scope': 'Complete supplied windows, source PTS and candidate model-bound overlays; no independent geometry acceptance claim.'}
    dump_json(out / 'review_manifest.json', manifest)
    return {'entrypoint': str(out / 'index.html'), 'manifest': str(out / 'review_manifest.json'),
            'programmes': len(programmes), 'file_count': len(files) + 1, 'missing_reference_images': sorted(set(missing_images))}


_PAGE = '''<!doctype html>
<html lang="zh-CN" data-static-model-delivery><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>RinkSpace · 共同模型与原片网格</title><link rel="stylesheet" href="report.css"><script src="geometry.js" defer></script><script src="report.js" defer></script></head>
<body><header><div class="brand">RinkSpace <span>共同模型 / 原片网格</span></div><span class="badge">候选演示 · 独立几何未验证</span></header>
<main><section class="intro"><h1>同一模型，回到原片上检查。</h1><p id="modelIdentity"></p><p id="deliverySummary">当前工作台的便携验收页。保留完整选定镜头、断链与失败；网格只画在当前帧的支持范围。</p></section>
<section class="toolbar"><label>节目 / 选定镜头<select id="programmeSelect">__OPTIONS__</select></label><label class="check"><input type="checkbox" id="showGrid" checked> 显示网格</label><label class="check"><input type="checkbox" id="showLandmarks"> 地标编号</label><span id="geometryLabel" class="badge"></span></section>
<section class="layout"><article><div id="videoStage"><video id="video" controls muted playsinline preload="metadata"></video><canvas id="gridCanvas" aria-label="模型坐标网叠加"></canvas></div><div class="status-line"><strong id="status">无法定位</strong><span id="frameIdentity"></span></div><p id="reason"></p><div class="steps"><button id="previous">← 前一帧</button><button id="next">后一帧 →</button><button id="firstCandidate">跳到首个网格候选</button><button id="sourceChain">查看此帧来源链</button><span id="windowSummary"></span></div><input type="range" id="frameSlider" min="0" max="0" value="0" step="1" aria-label="完整片段逐帧"><canvas id="statusStrip" width="1400" height="20" aria-label="完整选定片段定位状态分布"></canvas><p class="legend"><span class="observed">■ 定位有证据</span><span class="predicted">■ 连续推算</span><span class="candidate">■ 待验证候选</span><span class="unavailable">■ 无法定位</span> · 图像对应不代表独立几何通过</p><p id="runSummary"></p></article>
<aside><h2>共同模型 · 辅助视图</h2><canvas id="mapCanvas" width="720" height="450" aria-label="模型来源范围与当前支持范围"></canvas><p>浅色范围为已连接参考图的冰面观测；亮线为当前可见支持区域。</p><a id="overviewLink" hidden target="_blank">打开实拍模型概览 ↗</a><div id="coverage"></div><h3>地标与来源</h3><div id="landmarks"></div></aside></section>
<section class="evidence"><h2>模型多认清了哪里</h2><div id="referenceViews" class="references"></div><h3>仍未知 / 尚待确认</h3><div id="ambiguities"></div><p id="missingImages"></p><p>此前开发镜头继续作回归；本页不把回归改称新盲测，不撤销历史失败。原片没有装入此包，展示视频为有完整源 PTS 对照的代理。</p></section>
<details class="records"><summary>模型、源 PTS 和完整定位记录</summary><div>__ASSETS__<a href="review_manifest.json">review_manifest.json</a></div></details>
<footer>原片保持只读 · 无外网运行依赖 · 射影位置坐标网不代表等距方格或一米格</footer></main>
<dialog id="provenance"><div class="dialog-head"><h2 id="provenanceTitle">来源链</h2><button id="closeProvenance">关闭</button></div><div id="provenanceBody"></div></dialog><script id="review-data" type="application/json">__DATA__</script></body></html>'''
