"""Current print templates and their separately recorded source evidence.

All template contours use the native print chart, including analytic squares.
Missing original pixels, frame files or observation links are reported, never
reconstructed from the model. This preview performs no extraction or fitting.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import html
import json
import math
from pathlib import Path, PureWindowsPath

import numpy as np
from PIL import Image, ImageDraw

from ..core.native import NativeCore
from ..domain.print_assets import sample_template_paths, validate_template_topology
from ..review.fonts import review_font
from .curling_preview import Canvas, INK, GRAY, BG, write

COLORS = {'historical_template_evidence': '#148c6f',
          'current_programme_source_contour': '#ec8f28',
          'current_programme_source_glyph': '#148c6f',
          'template_shape_source': '#148c6f'}
HOLE = '#d34cce'


def _fit(image, size):
    scale = min(size[0]/image.width, size[1]/image.height)
    target = tuple(max(1, round(v*scale)) for v in image.size)
    return image.resize(target, Image.Resampling.LANCZOS), scale


def _points(value):
    points = np.asarray(value, dtype=float)
    if points.shape == (2,):
        points = points[None, :]  # One stored corner, not a generated contour.
    if (points.ndim != 2 or points.shape[1] != 2 or not len(points)
            or not np.all(np.isfinite(points))):
        raise ValueError('finite original source_native_uv is required')
    return points


def _seconds(pts, time_base):
    if type(pts) is not int or time_base is None:
        return None
    try:
        clock = Fraction(time_base)
    except (ValueError, TypeError, ZeroDivisionError):
        return None
    return float(pts*clock) if clock > 0 else None


def source_panels(rows, prefix, out, title):
    """Draw each row's own stored pixels; never imply video-exposure verification."""
    if not rows:
        raise ValueError('source panel requires stored observations')
    frame = Path(rows[0]['source_frame_path'])
    fields = ('source_frame_path', 'source_id', 'source_sha256', 'pts', 'time_base')
    if any(any(r.get(key) != rows[0].get(key) for key in fields) for r in rows):
        raise ValueError('mixed-source rows cannot be drawn on one source image')
    payload = frame.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if any(r.get('source_frame_sha256') and r['source_frame_sha256'] != digest for r in rows):
        raise ValueError('declared native source-frame bytes changed: '+str(frame))
    image = Image.open(frame).convert('RGB')
    points = np.concatenate([_points(r.get('source_native_uv')) for r in rows])
    if np.any(points.min(0) < 0) or np.any(points.max(0) > np.asarray(image.size)-1):
        raise ValueError('source UV is outside the original native frame')
    lo = np.maximum(np.floor(points.min(0)-16).astype(int), 0)
    hi = np.minimum(np.ceil(points.max(0)+16).astype(int), image.size)
    box = tuple(map(int, np.r_[lo, hi]))
    enlarged, scale = _fit(image.crop(box), (1120, 285))
    plain = Image.new('RGB', (1120, 300), BG)
    offset = ((1120-enlarged.width)//2, (300-enlarged.height)//2)
    plain.paste(enlarged, offset)
    overlay = plain.copy(); pen = ImageDraw.Draw(overlay)
    identifiers = []
    font = review_font(18, cjk=False)
    for index, row in enumerate(rows):
        identity = f'{prefix}{index+1:02d}'
        p = (_points(row['source_native_uv'])-lo)*scale+offset
        hole = (row.get('role') or '').endswith('hole_boundary') or 'inner' in (row.get('feature_id') or '')
        color = HOLE if hole else COLORS.get(row.get('evidence_role'), '#148c6f')
        line = p.tolist()
        if row.get('closed') is True and not np.array_equal(p[0], p[-1]):
            line.append(p[0].tolist())
        if len(line) > 1 and type(row.get('closed')) is bool:
            pen.line([tuple(v) for v in line], fill=color, width=2, joint='curve')
        else:
            # Without recorded topology, show the recorded points only.
            for x, y in p:
                pen.ellipse((x-3, y-3, x+3, y+3), fill=color)
        at = np.mean(p, axis=0)
        bounds = pen.textbbox(tuple(at), identity, font=font, anchor='mm')
        pen.rectangle((bounds[0]-2, bounds[1]-1, bounds[2]+2, bounds[3]+1), fill='#142827')
        pen.text(tuple(at), identity, font=font, anchor='mm', fill=color)
        identifiers.append({'display_id': identity, 'feature_id': row.get('feature_id'),
                            'observation_id': row.get('id'), 'closed': row.get('closed')})
    out = Path(out)
    plain_path = out/f'{prefix}_source.png'; overlay_path = out/f'{prefix}_overlay.png'
    plain.save(plain_path); overlay.save(overlay_path)
    return {'title': title, 'prefix': prefix, 'plain': str(plain_path), 'overlay': str(overlay_path),
            'source_frame': str(frame), 'source_frame_sha256': digest, 'native_size': list(image.size),
            'declared_frame_hash_verified': all(bool(r.get('source_frame_sha256')) for r in rows),
            'native_source_verified': False,
            'source_verification': 'stored image and recorded UV only; original video exposure not decoded',
            'crop_native_xyxy': list(box), 'display_scale': scale, 'display_offset': list(offset),
            'source_id': rows[0].get('source_id'), 'source_sha256': rows[0].get('source_sha256'),
            'pts': rows[0].get('pts'), 'time_base': rows[0].get('time_base'),
            'source_seconds': _seconds(rows[0].get('pts'), rows[0].get('time_base')),
            'identifiers': identifiers, 'native_vertex_count': len(points)}


def _source_rows(assets, assets_path):
    """Resolve only explicit frame references at the row's own exposure."""
    result, records = [], []
    for index, row in enumerate(assets.get('observations', [])):
        record = {'row_index': index, 'observation_id': row.get('id'),
                  'feature_id': row.get('feature_id'), 'source_id': row.get('source_id'),
                  'pts': row.get('pts'), 'time_base': row.get('time_base'),
                  'missing': [], 'warnings': []}
        source = assets.get('sources', {}).get(row.get('source_id'), {})
        image = row.get('source_image') or {}
        if not isinstance(image, dict):
            image = {'file': image}
        frame = row.get('source_frame_path') or image.get('file') or image.get('path')
        frame_hash = row.get('source_frame_sha256') or image.get('sha256')
        if not frame:
            # A video registry may retain the first exposure at its top level.
            # Never use that frame for a different row PTS, even on the same video.
            exposures = source.get('exposures', [])+[source]
            matches = [e for e in exposures if type(row.get('pts')) is int
                       and e.get('pts') == row['pts']
                       and e.get('time_base', source.get('time_base')) == row.get('time_base')
                       and e.get('frame_path')]
            paths = {e['frame_path'] for e in matches}
            if len(paths) == 1:
                frame = next(iter(paths))
                hashes = {e['frame_sha256'] for e in matches if e.get('frame_sha256')}
                if len(hashes) > 1:
                    raise ValueError('conflicting registered frame hashes at the same exposure')
                frame_hash = frame_hash or next(iter(hashes), None)
        try:
            _points(row.get('source_native_uv'))
        except (ValueError, TypeError):
            record['missing'].append('source_native_uv_missing_or_invalid')
        if not frame:
            record['missing'].append('source_frame_path_missing')
        else:
            path = Path(frame)
            if not path.is_absolute() and not PureWindowsPath(str(frame)).is_absolute():
                path = assets_path.parent/path
            frame = str(path)
            record['source_frame'] = frame
            if not path.is_file():
                record['missing'].append('source_frame_unavailable')
        for key in ('id', 'feature_id'):
            if not row.get(key):
                record['warnings'].append(key+'_not_recorded')
        if type(row.get('pts')) is not int:
            record['warnings'].append('native_pts_not_recorded')
        if _seconds(row.get('pts'), row.get('time_base')) is None:
            record['warnings'].append('native_clock_unavailable')
        if not frame_hash:
            record['warnings'].append('source_frame_sha256_not_recorded')
        record['status'] = 'unavailable' if record['missing'] else 'ready'
        records.append(record)
        if not record['missing']:
            result.append((record, {**row, 'source_frame_path': frame,
                'source_frame_sha256': frame_hash,
                'source_sha256': row.get('source_sha256') or source.get('video_sha256') or source.get('sha256')}))
    return result, records


def _template_size(assets, template, parameter_state=None):
    from ..domain.print_parameters import parameter_values
    values = None if parameter_state is None else parameter_values(assets, parameter_state)
    matches = [b for b in assets.get('parameter_blocks', []) if b['id'] == template.get('size_class_id')]
    if len(matches) > 1:
        raise ValueError('duplicate template size block')
    current = None if not matches else (matches[0].get('values') if values is None
                                       else values[matches[0]['id']])
    if current is not None:
        block = matches[0]
        if block.get('kind') != 'print_size' or block.get('size') != 2 or len(current) != 2:
            raise ValueError('invalid template size block')
        width, height = map(math.exp, current)
        basis = {'kind': 'shared_model_size', 'block_id': block['id'], 'status': block.get('status'),
                 'coordinate_unit': 'rink_length', 'physical_dimensions_verified': False}
    else:
        reference = template.get('display_reference_size')
        width, height = reference if reference is not None else (1., 1.)
        basis = {'kind': 'recorded_display_reference' if reference is not None else 'unit_material_chart',
                 'coordinate_unit': 'display_only', 'physical_dimensions_verified': False,
                 'missing': 'initialized_shared_model_size'}
        if any(p.get('geometry') for p in template['paths']):
            raise ValueError('analytic print shape requires its initialized shared size')
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError('finite positive template display dimensions required')
    return [float(width), float(height)], basis


def _template_panel(assets, template, core, out, index, *, parameter_state=None):
    validate_template_topology(template)
    dimensions, basis = _template_size(assets, template, parameter_state)
    current_paths = sample_template_paths(assets, template['id'], core, dimensions,
                                         parameter_state=parameter_state)
    shapes = []
    for number, (path, current) in enumerate(zip(template['paths'], current_paths), 1):
        geometry = path.get('geometry')
        shapes.append({'id': path['id'], 'display_id': f'{number:02}', 'closed': path.get('closed'),
            'role': path.get('role'), 'parent_contour_key': path.get('parent_contour_key'),
            'display_fill': path.get('display_fill'), 'geometry': geometry,
            'source_observation_id': path.get('source_observation_id'),
            'source_observation_ids': path.get('source_observation_ids', []),
            **current, 'vertices_are_source_observations': False})
    if not shapes:
        raise ValueError('template has no current paths')
    points = np.vstack([p['native_template_xy'] for p in shapes])
    lo, hi = points.min(0), points.max(0)
    span = np.maximum(hi-lo, np.asarray(dimensions)*1e-9)
    scale = min(2200/span[0], 340/span[1])
    origin = np.array([1200., 390.])-(lo+hi)/2*scale
    canvas = Canvas((2400, 780))
    label = template['id'].rsplit('/', 1)[-1]
    canvas.text((55, 52), f'D1 · {label} · 当前统一模板', 42)
    canvas.text((55, 110), template.get('text') or '已登记的大轮廓与材料边界', 29)
    holes = sum(bool(p['parent_contour_key']) for p in shapes)
    canvas.text((55, 157), f'{len(shapes)} 条当前路径，含 {holes} 条内孔；全部由同一 native print chart 生成。编号对照与缺失清单见网页。', 25)
    for shape in sorted(shapes, key=lambda p: bool(p['parent_contour_key'])):
        p = np.asarray(shape['native_template_xy'])*scale+origin
        hole = bool(shape['parent_contour_key'])
        if shape['closed'] and shape['display_fill'] != 'none' and shape['role'] != 'panel_outer_boundary':
            canvas.draw.polygon([tuple(q) for q in p], fill=BG if hole else '#173d68')
        canvas.line(p, HOLE if hole else '#173d68', 2)
    plain = Canvas(canvas.image.size)
    plain.image = canvas.image.copy()
    plain.draw = ImageDraw.Draw(plain.image)
    for shape in shapes:
        p = np.asarray(shape['native_template_xy'])*scale+origin
        at = (p.min(0)+p.max(0))/2
        hole = bool(shape['parent_contour_key'])
        if shape['role'] == 'panel_outer_boundary':
            direction = shape['id'].rsplit('.', 1)[-1]
            offset = {'top_edge': [0, -26], 'bottom_edge': [0, 26],
                      'left_vertical': [-28, 0], 'right_vertical': [28, 0]}.get(direction, [0, -26])
            at = at+offset
        else:
            at[1] = p[:, 1].max()+20+(int(shape['display_id']) % 2)*16 if hole else p[:, 1].min()-22
        canvas.text(at, shape['display_id'], 18, HOLE if hole else INK, 'mm', True)
    if basis['kind'] == 'shared_model_size':
        caption = f'显示比例来自当前共享尺寸：宽 {dimensions[0]:.8f} L，高 {dimensions[1]:.8f} L；尺寸状态保留原记录。'
    elif basis['kind'] == 'unit_material_chart':
        caption = '缺少共享尺寸与显示比例；只展示无量纲材料坐标，不推断物理长宽。'
    else:
        caption = '缺少已初始化的共享尺寸；按已登记显示比例展示。该比例不作为物理尺寸。'
    missing = sum(not p['source_observation_id'] for p in shapes)
    for card in (plain, canvas):
        card.text((55, 640), caption, 25, GRAY)
        card.text((55, 686), '展开模板采用原材料坐标朝向与示意色；开放边界不补闭合，内孔不填成实体。', 25)
        card.text((55, 732), f'{missing} 条路径未登记直接 source_observation_id；来源及缺失逐项列于网页，不补造像素关联。', 24, GRAY)
    suffix = '' if index == 0 else f'_{index+1:02}'
    image = out/f'template_native{suffix}.png'
    numbered = out/f'template_numbered{suffix}.png'
    geometry_file = out/f'template_geometry{suffix}.json'
    plain.image.save(image)
    canvas.image.save(numbered)
    data = {'native_generated': True, 'native_api': 'shared_native_print_geometry',
            'geometry_evaluator': 'evaluate_print_path', 'template_id': template['id'],
            'display_origin_is_not_installation': True, 'display_dimensions': dimensions,
            'display_basis': basis, 'shape_status': template.get('shape_status'),
            'parameter_state_consumed': parameter_state is not None,
            'internal_geometry_schema': (template.get('internal_geometry') or {}).get('schema'),
            'physical_placement_estimated': False, 'fit_executed': False,
            'new_source_observations': 0, 'shapes': shapes}
    write(geometry_file, data)
    return {'template_id': template['id'], 'image': str(image), 'numbered': str(numbered), 'geometry': str(geometry_file),
            'shape_status': template.get('shape_status'), 'path_count': len(shapes),
            'hole_count': holes, 'direct_observation_links_missing': missing,
            'display_basis': basis, 'shapes': shapes}


def template_panel(assets, core, out):
    """Compatibility entry point for the first current template."""
    return _template_panel(assets, assets['templates'][0], core, Path(out), 0)['image']


def _escape(value):
    return html.escape(str(value)) if value is not None else '未登记'


def render(assets_path, library_path, out_dir, *, parameter_state=None):
    assets_path = Path(assets_path).resolve()
    assets = json.loads(assets_path.read_text(encoding='utf-8-sig'))
    out = Path(out_dir).resolve()
    if out.exists():
        raise FileExistsError('Advertisement review output already exists: '+str(out))
    out.mkdir(parents=True)
    core = NativeCore(library_path)
    templates = [_template_panel(assets, template, core, out, index, parameter_state=parameter_state)
                 for index, template in enumerate(assets['templates'])]
    available, source_records = _source_rows(assets, assets_path)
    grouped = {}
    for record, row in available:
        key = tuple(row.get(k) for k in ('source_frame_path', 'source_id', 'source_sha256',
                                         'pts', 'time_base', 'evidence_role'))
        grouped.setdefault(key, []).append((record, row))
    panels = []
    for index, group in enumerate(grouped.values(), 1):
        rows = [row for _, row in group]
        title = f'来源组 {index} · {len(rows)} 条已有读数 · '+(rows[0].get('evidence_role') or '原记录未分类')
        try:
            panel = source_panels(rows, f'S{index:02}', out, title)
        except (OSError, ValueError) as exc:
            for record, _ in group:
                record.update(status='rejected', reason=str(exc))
            continue
        panels.append(panel)
        for record, _ in group:
            record.update(status='displayed', panel_prefix=panel['prefix'])
    sections = []
    for template in templates:
        rows = ''.join('<tr><td>'+_escape(p['display_id'])+'</td><td>'+_escape(p['id'])+
            '</td><td>'+_escape(p['role'])+'</td><td>'+_escape(p['parent_contour_key'])+
            '</td><td>'+_escape(p['source_observation_id'])+'</td></tr>' for p in template['shapes'])
        sections.append('<h2>'+_escape(template['template_id'])+'</h2><p>原记录形状状态：'+
            _escape(template['shape_status'])+'；该状态原样展示，不推断新的用户验收。</p><img src="'+
            Path(template['image']).name+'"><details><summary>查看同一模板的路径编号</summary><img src="'+
            Path(template['numbered']).name+'"></details><p><a href="'+Path(template['geometry']).name+
            '">同一 native 核心的模板几何</a></p><table><thead><tr><th>图中编号</th><th>路径</th>'
            '<th>角色</th><th>父轮廓</th><th>直接 source_observation_id</th></tr></thead><tbody>'+rows+'</tbody></table>')
    for panel in panels:
        seconds = '未登记' if panel['source_seconds'] is None else f"{panel['source_seconds']:.3f}s"
        sections.append('<h2>'+_escape(panel['title'])+'</h2><p>PTS '+_escape(panel['pts'])+
            ' / '+_escape(panel['time_base'])+' · '+seconds+'；登记图像与已有 UV 对照，未解码核验原视频曝光。</p>'
            '<div class="pair"><img src="'+Path(panel['plain']).name+'"><img src="'+
            Path(panel['overlay']).name+'"></div><p>'+html.escape('；'.join(
                i['display_id']+' = '+(i['feature_id'] or '实体关联未登记') for i in panel['identifiers']))+'</p>')
    missing = [r for r in source_records if r['status'] != 'displayed' or r['warnings']]
    missing_rows = ''.join('<tr><td>'+_escape(r['observation_id'])+'</td><td>'+_escape(r['pts'])+
        '</td><td>'+_escape(r['status'])+'</td><td>'+_escape(', '.join(r['missing']+r['warnings']))+
        '</td><td>'+_escape(r.get('source_frame'))+'</td><td>'+_escape(r.get('reason'))+'</td></tr>' for r in missing)
    sections.append('<h2>来源缺失与限制</h2><p>缺少像素、图像或关联时保持缺失；不会把模板采样点当作原片读数。'
        'unavailable 表示当前环境无法展示该条，rejected 表示已登记证据校验失败。</p>'+
        ('<table><thead><tr><th>已有观测 ID</th><th>PTS</th><th>显示状态</th><th>缺失 / 提示</th><th>登记图像</th><th>原因</th>'
         '</tr></thead><tbody>'+missing_rows+'</tbody></table>' if missing else '<p>所有当前内联来源条目均可展示。</p>'))
    if assets.get('observation_store'):
        sections.append('<p>另有已登记来源库；本次只展示当前资产 JSON 内联的 observation，未展开外部来源库：<code>'+
                        _escape(json.dumps(assets['observation_store'], ensure_ascii=False))+'</code></p>')
    limitations = assets.get('limitations', [])
    sections.append('<p>'+html.escape('；'.join(str(v) for v in limitations))+'</p>')
    label = ' / '.join(t['template_id'].rsplit('/', 1)[-1] for t in templates)
    content = '<!doctype html><meta charset="utf-8"><title>'+_escape(label)+' 模板与来源预览</title>'
    content += '<style>body{font:18px/1.6 system-ui;margin:28px;background:#edf2f2;color:#173a3b}img{max-width:100%}'
    content += '.pair{display:flex;gap:12px}.pair img{width:49%;object-fit:contain}table{border-collapse:collapse;width:100%;font-size:14px}'
    content += 'th,td{padding:7px;text-align:left;border:1px solid #bdccce;overflow-wrap:anywhere}code{overflow-wrap:anywhere}</style>'
    content += '<h1>D1 · '+_escape(label)+' 当前模板与已有来源</h1><p>全开当前模板路径、孔洞和边界；所有颜色与编号仅用于展示。'
    content += '不新增读数，不运行拟合，不改变现有用户批准。</p>'+''.join(sections)
    page = out/'index.html'; page.write_text(content, encoding='utf8')
    receipt = {'schema': 'rinkspace.ad_asset_review.v1', 'assets': str(assets_path),
        'library': str(Path(library_path).resolve()), 'panels': panels, 'source_rows': source_records,
        'card': templates[0]['image'], 'template': templates[0]['image'], 'page': str(page),
        'templates': [{k: v for k, v in t.items() if k != 'shapes'} for t in templates],
        'counts': assets.get('counts', {}), 'displayed_source_rows': sum(r['status'] == 'displayed' for r in source_records),
        'unavailable_source_rows': sum(r['status'] != 'displayed' for r in source_records),
        'external_observation_store_expanded': False, 'fit_executed': False,
        'parameter_state_consumed': parameter_state is not None,
        'new_source_observations': 0, 'whole_advertisement_user_approval_inferred': False}
    write(out/'review.json', receipt)
    return receipt
