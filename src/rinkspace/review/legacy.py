"""Offline reports with traceable samples and separately supplied landmark audits."""
from __future__ import annotations

import html
import json
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from ..geometry import project
from ..io import dump_json, load_json


def independent_audit(registration, observations, reference_width=1920):
    """Check reviewed observations, never generate ground truth with the fitted H.

    Each point belongs to a physical group protected in *every* fitting frame.
    The caller supplies explicitly observed sample coordinates and reference
    coordinates. Missing observations remain missing. Full-shot acceptance is
    withheld unless every non-reference sample has >=4 visible landmarks.
    """
    rows = registration.get('frames', [])
    by_index = {r['sample_index']: r for r in rows}
    scale = reference_width / registration['decoded_dimensions'][0]
    errors, all_errors, checks = [], [], []
    protection_violations = []
    support_violations = []
    visible_by_frame = {}
    groups = {p['physical_landmark_id'] for p in observations if p.get('visible', True)}
    protected_groups = set.intersection(*[
        {region.get('physical_landmark_id') for region in row.get('protected_regions', [])}
        for row in rows]) if rows else set()
    missing_protection = sorted(groups - protected_groups)
    def contains(region, xy):
        if 'polygon' in region:
            return cv2.pointPolygonTest(np.asarray(region['polygon'], np.float32), tuple(map(float, xy)), False) >= 0
        if 'xy' in region:
            return np.linalg.norm(np.asarray(region['xy'])-xy) <= region.get('radius_px', 20)
        return False
    for row in rows:
        for region in row.get('protected_regions', []):
            if any(contains(region, xy) for xy in row.get('fit_current_xy', [])):
                protection_violations.append({'sample_index': row['sample_index'], 'reason': 'fit_points_inside_protected_region'})
        reference_row = by_index.get(row.get('reference_sample_index', 0), {})
        for region in reference_row.get('protected_regions', []):
            if any(contains(region, xy) for xy in row.get('fit_reference_xy', [])):
                protection_violations.append({'sample_index': row['sample_index'], 'reason': 'fit_reference_points_inside_protected_region'})
    for point in observations:
        row = by_index[point['sample_index']]
        check = dict(point)
        check.update(source_frame_index=row.get('source_frame_index'), source_time_s=row['source_time_s'])
        if point.get('visible', True) is False or row.get('H_reference_from_current') is None:
            check.update(status='unavailable', error_px=None)
        elif row.get('reference_sample_index') != point.get('reference_sample_index', 0):
            check.update(status='different_component', error_px=None)
        else:
            regions = [r for r in row.get('protected_regions', []) if r.get('physical_landmark_id') == point['physical_landmark_id']]
            protected_here = any(contains(region, point['source_xy']) for region in regions)
            check['measurement_eligible'] = protected_here and point.get('protected_from_fitting', True)
            check['ineligible_reason'] = None if check['measurement_eligible'] else 'candidate_observation_outside_protected_physical_group'
            if not protected_here:
                protection_violations.append({'sample_index': row['sample_index'], 'reason': 'observed_point_outside_its_protected_group', 'point': point})
            polygon = row.get('support_polygon', [])
            distance = cv2.pointPolygonTest(np.asarray(polygon, np.float32), tuple(map(float,point['source_xy'])), True) if len(polygon) >= 3 else None
            check['distance_inside_support_px'] = distance
            check['outside_support'] = distance is None or distance < 0
            if distance is None or distance < -max(12, registration['decoded_dimensions'][0]*.02):
                support_violations.append({'sample_index': row['sample_index'], 'point_id': point.get('point_id'), 'distance_inside_support_px': distance})
            predicted = project(row['H_reference_from_current'], [point['source_xy']])[0]
            residual = predicted - np.asarray(point['reference_xy'])
            predicted_source = project(np.linalg.inv(row['H_reference_from_current']), [point['reference_xy']])[0]
            error = float(np.linalg.norm(predicted_source - np.asarray(point['source_xy'])))
            check.update(status='observed_check', predicted_reference_xy=predicted.tolist(),
                         residual_reference_px=residual.tolist(), error_px=error,
                         predicted_source_xy=predicted_source.tolist(),
                         error_1920_equivalent_px=error*scale)
            if not row.get('reference_frame_only'):
                all_errors.append(error * scale)
                if check['measurement_eligible']:
                    errors.append(error * scale)
                    visible_by_frame.setdefault(row['sample_index'], set()).add(point['physical_landmark_id'])
        checks.append(check)
    tested = [r for r in rows if not r.get('reference_frame_only')]
    covered = sum(len(visible_by_frame.get(r['sample_index'], [])) >= 4 for r in tested)
    median = float(np.median(errors)) if errors else None
    p95 = float(np.percentile(errors, 95)) if errors else None
    visible_points = [p for p in observations if p.get('visible', True)]
    unique_points = {p.get('point_id', p['physical_landmark_id']) for p in visible_points}
    reference_points = np.asarray([p['reference_xy'] for p in visible_points], np.float32)
    check_hull_fraction = float(cv2.contourArea(cv2.convexHull(reference_points))) / np.prod(registration['decoded_dimensions']) if len(reference_points) >= 3 else 0
    passes = bool(errors and len(groups) >= 6 and not missing_protection and not protection_violations and not support_violations
                  and check_hull_fraction >= .025 and covered == len(tested)
                  and median <= 4 and p95 <= 10)
    return {'status': 'independently_validated_local_region' if passes else 'independent_checks_incomplete_or_failed',
            'accepted': passes, 'reference_resolution_width': reference_width,
            'median_1920_px': median, 'p95_1920_px': p95, 'thresholds': {'median_px': 4, 'p95_px': 10},
            'unique_physical_landmarks': len(groups), 'missing_protection_groups': missing_protection,
            'unique_visible_check_points': len(unique_points), 'check_hull_fraction': check_hull_fraction,
            'coverage_counting_unit': 'distinct_physical_landmark_group',
            'protection_violations': protection_violations,
            'support_violations': support_violations,
            'frames_with_four_visible_checks': covered, 'nonreference_frame_count': len(tested),
            'observation_count': len(errors), 'checks': checks,
            'all_candidate_error_summary': {'count': len(all_errors),
                'median_1920_px': float(np.median(all_errors)) if all_errors else None,
                'p95_1920_px': float(np.percentile(all_errors,95)) if all_errors else None},
            'ineligible_candidate_count': len(all_errors)-len(errors),
            'scope': 'Observed protected ice landmarks only; no metric, off-plane or whole-rink accuracy claim.'}


def _letterbox(image, width, height):
    if image is None:
        return np.zeros((height, width, 3), np.uint8)
    if image.shape[2] == 4:
        yy, xx = np.indices(image.shape[:2])
        background = np.where(((xx//24 + yy//24) % 2)[..., None], 38, 26).astype(np.uint8)
        image = np.where(image[:, :, 3:4] > 0, image[:, :, :3], background)
    scale = min(width/image.shape[1], height/image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1]*scale)), max(1, round(image.shape[0]*scale))))
    canvas = np.full((height, width, 3), 24, np.uint8)
    y, x = (height-resized.shape[0])//2, (width-resized.shape[1])//2
    canvas[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
    return canvas


def attach_independent_audit(out, observations, provenance=None):
    out = Path(out)
    reg = load_json(out / 'registration.json')
    audit = independent_audit(reg, observations)
    audit['provenance'] = provenance or {'type': 'user_reviewed_correspondences'}
    dump_json(out / 'independent_audit.json', audit)
    source = Path(reg['sample_directory'])
    (out / 'errors').mkdir(exist_ok=True)
    for row in reg['frames']:
        checks = [p for p in audit['checks'] if p['sample_index'] == row['sample_index']]
        if not checks:
            continue
        image = cv2.imread(str(source / row['image']))
        values = []
        for point in checks:
            if not point.get('visible', True):
                continue
            xy = tuple(np.rint(point['source_xy']).astype(int))
            color = (70,235,60) if point.get('measurement_eligible', False) else (0,150,255)
            cv2.circle(image, xy, 5, color, 1, cv2.LINE_AA)
            cv2.putText(image, str(point.get('point_id', point['physical_landmark_id'])), (xy[0]+7, xy[1]-7),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (50,255,80), 1, cv2.LINE_AA)
            if point.get('predicted_source_xy'):
                prediction = tuple(np.rint(point['predicted_source_xy']).astype(int))
                cv2.drawMarker(image, prediction, (70,70,255), cv2.MARKER_CROSS, 12, 1)
                cv2.line(image, xy, prediction, (20,210,255), 1)
                if point.get('measurement_eligible', False):
                    values.append(point['error_1920_equivalent_px'])
        header = np.full((62,image.shape[1],3), 22, np.uint8)
        cv2.putText(header, f"WITHHELD CHECK {row['source_time_s']:.3f}s | green eligible / orange invalid / red projection", (12,23),cv2.FONT_HERSHEY_SIMPLEX,.49,(230,230,230),1,cv2.LINE_AA)
        text = f"{len(values)} checks | median {np.median(values):.2f}px P95 {np.percentile(values,95):.2f}px (1920 equivalent)" if values else 'No usable independent error for this source frame'
        cv2.putText(header,text,(12,48),cv2.FONT_HERSHEY_SIMPLEX,.5,(100,220,230),1,cv2.LINE_AA)
        path = f'errors/{row["sample_index"]:06d}.jpg'
        cv2.imwrite(str(out / path), np.vstack((header,image)), [cv2.IMWRITE_JPEG_QUALITY,91])
        row.update(error_image=path, independent={'status': audit['status'], 'accepted': audit['accepted'],
            'check_count': len(values), 'median_error_px': float(np.median(values)) if values else None,
            'p95_error_px': float(np.percentile(values,95)) if values else None,
            'reference_resolution_width': 1920, 'coverage_scope': 'protected visible point observations only'})
    dump_json(out / 'registration.json', reg)
    return audit


def encode_comparison(out, registration):
    rows = registration['frames']
    if not rows:
        raise ValueError('Cannot encode an empty report')
    source = Path(registration['sample_directory'])
    output = av.open(str(out / 'comparison.mp4'), 'w', options={'movflags': '+faststart'})
    stream = output.add_stream('libx264', rate=25)
    stream.width, stream.height, stream.pix_fmt = 1280, 414, 'yuv420p'
    stream.options = {'crf': '24', 'preset': 'fast'}
    stream.time_base = Fraction(1, 1000000)
    stream.codec_context.time_base = Fraction(1, 1000000)
    base = rows[0]['source_time_s']
    preview_rows = []
    for i, row in enumerate(rows):
        src = cv2.imread(str(source / row['image']))
        fixed = cv2.imread(str(out / row['fixed_image']), cv2.IMREAD_UNCHANGED) if row.get('fixed_image') else None
        panel = np.zeros((414, 1280, 3), np.uint8)
        panel[54:, :640] = _letterbox(src, 640, 360)
        panel[54:, 640:] = _letterbox(fixed, 640, 360)
        cv2.putText(panel, f"SOURCE {row['source_time_s']:.3f}s | frame {row.get('source_frame_index')}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, .53, (240,240,240), 1, cv2.LINE_AA)
        cv2.putText(panel, f"FIXED ICE | {row['status']} | {row.get('graph_component_id')}", (650,22), cv2.FONT_HERSHEY_SIMPLEX, .5, (100,225,230), 1, cv2.LINE_AA)
        cv2.putText(panel, 'LOCAL PROJECTIVE / single observed frame / no metric or 3D claims', (12,44), cv2.FONT_HERSHEY_SIMPLEX, .48, (180,180,180), 1, cv2.LINE_AA)
        if fixed is None:
            cv2.putText(panel, 'CHAIN BROKEN - NO MAP', (740,225), cv2.FONT_HERSHEY_SIMPLEX, .75, (100,150,255), 2, cv2.LINE_AA)
        frame = av.VideoFrame.from_ndarray(panel, format='bgr24')
        frame.pts = round((row['source_time_s']-base)*1000000)
        frame.time_base = Fraction(1, 1000000)
        for packet in stream.encode(frame):
            output.mux(packet)
        preview_rows.append({'preview_frame_index': i, 'preview_time_s': row['source_time_s']-base,
                             **{k: row.get(k) for k in ('source_frame_index', 'pts', 'time_base', 'source_time_s')}})
        if i in {0, len(rows)//2, len(rows)-1}:
            cv2.imwrite(str(out / f'comparison-{i:06d}.jpg'), panel)
    for packet in stream.encode():
        output.mux(packet)
    output.close()
    with av.open(str(out / 'comparison.mp4')) as decoded:
        encoded_times = [float(frame.pts * frame.time_base) for frame in decoded.decode(video=0)]
    if len(encoded_times) != len(preview_rows) or any(abs(t-row['preview_time_s']) > 0.00002 for t,row in zip(encoded_times, preview_rows)):
        raise RuntimeError('Comparison encoded timestamps differ from source sample mapping')
    dump_json(out / 'preview_map.json', {'source_sha256': registration['source_sha256'],
        'display_speed': 1, 'sampling': 'selected decoded source frames; no optical interpolation', 'frames': preview_rows})


def build_run_report(out):
    out = Path(out)
    reg = load_json(out / 'registration.json')
    run = load_json(out / 'run.json')
    encode_comparison(out, reg)
    audit = load_json(out / 'independent_audit.json') if (out / 'independent_audit.json').exists() else None
    rows = reg['frames']
    picks = sorted({0, len(rows)//2, len(rows)-1})
    sample_markup = ''
    for index in picks:
        row = rows[index]
        sample_markup += f'<article><h3>源 {row["source_time_s"]:.3f} s · 帧 {row.get("source_frame_index")}</h3><img src="comparison-{index:06d}.jpg"><img src="{row["mask_image"]}"><p>{html.escape(row.get("reason", ""))}</p></article>'
        if row.get('error_image'):
            sample_markup += f'<article><img src="{row["error_image"]}"></article>'
    audit_text = '尚未提供独立观测；拟合残差不能替代独立精度。'
    if audit:
        audit_text = f'检查候选中位数 {audit["median_1920_px"]} px，P95 {audit["p95_1920_px"]} px（1920宽等效）；{audit["frames_with_four_visible_checks"]}/{audit["nonreference_frame_count"]} 非参考帧有≥4可见点；验收：{audit["status"]}。'
        if audit.get('ineligible_candidate_count'):
            all_summary = audit['all_candidate_error_summary']
            audit_text += f' 另有{audit["ineligible_candidate_count"]}个匹配跑出保护物理组，不计作独立观测但全部保留。全部候选的中位/P95为{all_summary["median_1920_px"]:.2f}/{all_summary["p95_1920_px"]:.2f}px；剔除不合格匹配不改变未通过状态。'
        provenance = audit.get('provenance') or {}
        if provenance.get('type') == 'protected_physical_group_patch_correlation_with_AI_visual_review':
            audit_text += ' 本次观测来自完整留出的冰面印记图块相关，已目视审阅，但不是人工亚像素真值；雾化边缘单点约1–3个960宽样图像素的不确定性。六点集中于一个物理印记，分布不足，不能宣称整镜头通过。'
            if provenance.get('iteration_wide_reference_exclusion_verified') is not True:
                audit_text += ' 检查点对未用于拟合；ECC迭代参考采样足迹未完整记录，尚不能证明优化全程像素独立性。最终足迹和当前排除掩膜仅是部分证据。'
    escaped = lambda value: html.escape(str(value))
    check_gallery = ''.join(f'<img src="{escaped(path.name)}" alt="独立检查图 / 源帧审阅">' for path in sorted(out.glob('check-*.jpg')))
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>RinkSpace · {escaped(run['id'])}</title>
<style>body{{max-width:1320px;margin:32px auto;padding:0 24px;font:16px/1.65 system-ui;background:#101722;color:#dee7ef}}h1,h2{{color:#8cdbd8}}video,img{{max-width:100%;background:#171d27}}article,section{{padding:24px;background:#1b2634;margin:20px 0;border-radius:12px}}a{{color:#83d9eb}}code{{overflow-wrap:anywhere}}.badge{{color:#fbd587}}</style>
<h1>原片 / 固定冰面 / 独立证据</h1><p class="badge">{escaped(run['role'])} · {escaped(run['status'])} · 局部投影参照，未建立米制</p>
<p>源时间 {run['start_s']:.3f}–{run['end_s']:.3f} s。原始帧号来自完整解码索引；对照片是有明确源映射的采样预览。</p>
{'<p class="badge">目视复查：' + escaped(run['review_note']) + '</p>' if run.get('review_note') else ''}
<video src="comparison.mp4" controls muted preload="metadata"></video><p><a href="preview_map.json">逐预览帧 → 源 PTS</a> · <a href="registration.json">每帧变换 / 保护区 / 拟合支持点</a> · <a href="run.json">运行配置</a></p>
<section><h2>独立误差证据</h2><p>{escaped(audit_text)}</p>{'<a href="independent_audit.json">独立观测及误差</a>' if audit else ''}
{check_gallery}
<p>亮区用于拟合；暗区为非冰面/遮挡候选；紫色为独立地标保护区；绿色多边形为拟合支持区域。自动掩膜是启发式候选，需目视检查。支持区域外属于外推。</p></section>
{sample_markup}<section><h2>来源与边界</h2><p>SHA-256：<code>{escaped(reg['source_sha256'])}</code></p><p>算法 {escaped(reg['algorithm'])}；配置指纹 {escaped(reg['configuration_sha256'])}。切镜和信息不足会断链，不沿用旧 H。透明/棋盘区域未被当前源帧观测；没有生成缺失冰面。每个断开分量有独立固定画布。</p>
<p>人物只随冰面投影显示，非接触或空中身体不代表地面位置。全场标定、跨镜头拼图、接触轨迹、速度和节目统计按交付 B/C/D 继续。</p></section></html>'''
    (out / 'index.html').write_text(report, encoding='utf-8')
    # Compact export explicitly includes sampled evidence, not raw videos/caches.
    files = ['index.html', 'comparison.mp4', 'preview_map.json', 'registration.json', 'run.json']
    for i in picks:
        files.extend([f'comparison-{i:06d}.jpg', rows[i]['mask_image']])
        if rows[i].get('error_image'):
            files.append(rows[i]['error_image'])
    if audit:
        files.append('independent_audit.json')
    files += [p.name for p in out.glob('check-*.jpg')]
    dump_json(out / 'review_manifest.json', {'entrypoint': 'index.html', 'files': sorted(set(files)),
        'source_video_included': False, 'scope': 'short sampled comparison and auditable geometry; original remains local'})
    # Keep the bundler a normal package module for an installed workbench.
    from ..bundle import make_bundle
    destination = out / 'review.zip'
    if destination.exists():
        destination.unlink()
    make_bundle(out, destination)
    return run
