"""Small offline HTML with explicit failure status, native-PTS evidence and JSON."""
from __future__ import annotations
from pathlib import Path
from fractions import Fraction
import html, json, csv
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from .geometry import error_stats, homography_points, symmetric_errors
from .io import write_json


def homography_diagnostic(graph):
    data, meta = graph['data'], graph['meta']
    r = np.zeros((len(data['i']), 4))
    edges = []
    for n, edge in enumerate(meta['edges']):
        m = data['edge'] == n; H = np.asarray(edge['H'])
        r[m, :2] = homography_points(H, data['a'][m])-data['b'][m]
        r[m, 2:] = homography_points(np.linalg.inv(H), data['b'][m])-data['a'][m]
        test = m & data['test']; tr = m & ~data['test'] & data['screened']
        edges.append(dict(edge=n, i=edge['i'], j=edge['j'], train=error_stats(r[tr]), holdout_all=error_stats(r[test])))
    te = data['test']; screened = te & data['screened']
    return dict(holdout_all=error_stats(r[te]), holdout_screened=error_stats(r[screened]), edges=edges), r


def model_diagnostics(graph, fit, homography):
    meta, data = graph['meta'], graph['data']; s = fit['summary']
    r, valid, R = fit['residual'], fit['valid'], fit['rotations']
    reasons = []
    if not s['optimizer_success']: reasons.append('OPTIMIZER_NOT_CONVERGED')
    if s['bounds_hit']: reasons.append('PARAMETERS_AT_NUMERIC_BOUNDS')
    if s.get('local_condition', 0) > 1e6: reasons.append('WEAK_LOCAL_OBSERVABILITY')
    if meta['holdout_tracks'] < 30: reasons.append('FEW_HOLDOUT_TRACKS')
    if (data['test'] & ~data['screened']).sum() > .15*max(1, data['test'].sum()):
        reasons.append('LARGE_HELDOUT_MATCH_REJECTION')
    if len(meta['frames']) < .8*meta['candidate_frame_count']: reasons.append('PARTIAL_TEMPORAL_COVERAGE')
    hm = homography['holdout_screened']['median_px']; fm = s['holdout_screened']['median_px']
    if hm is not None and fm is not None and fm > max(2., 2.5*hm): reasons.append('CENTRAL_ROTATION_MODEL_MISMATCH')
    if s['holdout_screened']['invalid_count']: reasons.append('INVALID_HELDOUT_RAYS')
    # Rotation-axis excitation, not pan/tilt relative to a physical tripod frame.
    relatives = np.array([R[e['j']] @ R[e['i']].T for e in meta['edges']])
    v = Rotation.from_matrix(relatives).as_rotvec()
    sv = np.linalg.svd(v, compute_uv=False)
    excitation = float(sv[1]/max(sv[0], 1e-12)) if len(sv) >= 2 else 0.
    if excitation < .05: reasons.append('NEAR_SINGLE_AXIS_ROTATION')
    if sv[0] < .005: reasons.append('NEAR_ZERO_ROTATION')
    rows, radii, temporal = [], [], []
    for k in range(4):
        m = data['test'] & data['screened'] & (data['b'][:, 1] >= meta['height']*k/4) & (data['b'][:, 1] < meta['height']*(k+1)/4)
        rows.append(dict(row_band=k, **error_stats(r[m], valid[m])))
    for k in range(3):
        times = np.array([f['pts']*float(Fraction(f['time_base'])) for f in meta['frames']])
        a, b = times.min()+(times.max()-times.min())*k/3, times.min()+(times.max()-times.min())*(k+1)/3
        m = data['test'] & data['screened'] & (times[data['j']] >= a) & (times[data['j']] <= b)
        temporal.append(dict(start_s=a, end_s=b, **error_stats(r[m], valid[m])))
    return dict(status='INCONCLUSIVE' if reasons else 'CONDITIONAL_IMAGE_CONSISTENCY',
        reasons=reasons, physical_camera_certified=False, altitude_identified=False,
        relative_rotation_axis_singular_values=sv, second_axis_ratio=excitation,
        row_bands=rows, time_blocks=temporal,
        limitations=['No metric scale, camera height or tripod lever estimate.',
            'Rolling shutter, noncentral parallax and optical stabilization are not estimated.',
            'Homography consistency does not exclude a dominant plane.',
            'Holdout is track-disjoint, not a separate time sequence or independent acquisition.',
            'Geometry screening is based only on training tracks; rejected holdout counts remain visible.'])


def evidence_images(graph, fit, out):
    """Image annotations only; no synthesized scene content."""
    out.mkdir(exist_ok=True); data, meta = graph['data'], graph['meta']
    errors = np.sqrt(np.sum(fit['residual']**2, axis=1)/2)
    edges = []
    for k, e in enumerate(meta['edges']):
        m = (data['edge'] == k) & data['test'] & data['screened']
        if m.sum() >= 4: edges.append((float(np.median(errors[m])), k))
    selected = sorted({k for _, k in sorted(edges, reverse=True)[:3]} |
                      {k for _, k in sorted(edges)[:2]})
    images = []
    for k in selected:
        e = meta['edges'][k]; i, j = e['i'], e['j']
        paths = [meta['frames'][a].get('image', '') for a in (i, j)]
        if any(not Path(p).is_file() for p in paths): continue
        left, right = [cv2.imread(p) for p in paths]
        if left is None or right is None: continue
        # Fixed readout-size coordinates are scaled once for display only.
        W, H = 720, int(round(720*meta['height']/meta['width']))
        sx, sy = W/meta['width'], H/meta['height']
        canvas = np.concatenate((cv2.resize(left, (W, H)), cv2.resize(right, (W, H))), axis=1)
        m = np.flatnonzero((data['edge'] == k) & data['test'] & data['screened'])
        m = m[np.argsort(errors[m])][-35:]
        for a in m:
            p = tuple(np.rint(data['a'][a]*[sx, sy]).astype(int))
            q = tuple(np.rint(data['b'][a]*[sx, sy]+[W, 0]).astype(int))
            cv2.line(canvas, p, q, (220, 220, 70), 1, cv2.LINE_AA)
            cv2.circle(canvas, p, 3, (30, 240, 240), -1)
            cv2.circle(canvas, q, 3, (30, 240, 240), -1)
            pred = data['b'][a] + fit['residual'][a, :2]
            pp = tuple(np.rint(pred*[sx, sy]+[W, 0]).astype(int))
            if all(abs(v) < 100000 for v in pp): cv2.drawMarker(canvas, pp, (40, 40, 255), cv2.MARKER_CROSS, 10, 1)
        label = f'Held-out tracks | PTS {meta["frames"][i]["pts"]} -> {meta["frames"][j]["pts"]} | yellow=observed red=predicted'
        cv2.rectangle(canvas, (0, 0), (2*W, 28), (20, 20, 20), -1)
        cv2.putText(canvas, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .42, (245, 245, 245), 1, cv2.LINE_AA)
        name = f'pair_{k:03}.jpg'; cv2.imwrite(str(out/name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        images.append(name)
    return images


def render_report(graph, fits, out, comparison=None, profile=None, provenance=None):
    hd, hr = homography_diagnostic(graph)
    write_json(out/'homography_baseline.json', hd)
    diagnostics = {name: model_diagnostics(graph, fit, hd) for name, fit in fits.items()}
    # Predetermined richest family for inspection, not a holdout-selected winner.
    selected = fits.get('free_pp_radial', list(fits.values())[-1])
    images = evidence_images(graph, selected, out/'evidence')
    summaries = {name: fit['summary'] for name, fit in fits.items()}
    result = dict(schema='rinkspace.ptz-verifier.v1', models=summaries, diagnostics=diagnostics,
        graph_summary={k: graph['meta'][k] for k in ('candidate_frame_count','selected_original_indices',
            'training_matches','holdout_matches','training_tracks','holdout_tracks','components')},
        current_comparison=comparison, aspect_profile=profile, provenance=provenance,
        physical_camera_certified=False, production_state_changed=False)
    result['graph_summary'].update(component_policy=graph['meta'].get('component_policy', 'largest'),
        active_component_count=len(graph['meta'].get('anchors', [0])))
    write_json(out/'summary.json', result)
    for family, fit in fits.items():
        with (out/f'{family}_cameras.csv').open('w', encoding='utf-8', newline='') as stream:
            writer=csv.writer(stream)
            writer.writerow(['source_pts','time_base','source_seconds','fx_px','fy_px','cx_px','cy_px','k1','k2','p1','p2', *[f'R_{i}{j}' for i in range(3) for j in range(3)]])
            for frame, optics, rotation in zip(graph['meta']['frames'], fit['optics'], fit['rotations']):
                writer.writerow([frame['pts'], frame['time_base'], frame['pts']*float(Fraction(frame['time_base'])), *optics.tolist(), *rotation.ravel().tolist()])
    def num(x): return '—' if x is None else f'{x:.3f}'
    rows = []
    for name, fit in fits.items():
        s = fit['summary']; d = diagnostics[name]
        rows.append(f'<tr><td>{html.escape(name)}</td><td>{num(s["fx_over_fy"])}</td><td>{num(s["holdout_all"]["median_px"])}</td><td>{num(s["holdout_all"]["p95_px"])}</td><td>{num(s["holdout_screened"]["median_px"])}</td><td>{num(s["holdout_screened"]["p95_px"])}</td><td>{"证据不足" if d["status"] == "INCONCLUSIVE" else "条件性一致"}</td></tr>')
    m = graph['meta']; total = m['candidate_frame_count']; used = len(m['frames'])
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RinkSpace · 独立 PTZ 验证</title><style>
body{{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f5f5f0;color:#1f2933;margin:0;line-height:1.75}}
main{{max-width:1080px;margin:auto;padding:36px 26px}}h1{{font-size:34px;line-height:1.3}}h2{{margin-top:36px}}
.card{{background:white;padding:22px 26px;border-radius:14px;margin:16px 0;border:1px solid #ddd}}
.badge{{font-weight:700;color:#884600}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px 8px;border-bottom:1px solid #ddd;text-align:left}}
img{{width:100%;border-radius:8px;margin:8px 0}}code{{background:#eee;padding:2px 5px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}small{{color:#58616b}}
</style><main><p>RINKSPACE / INDEPENDENT CAMERA CHECK</p><h1>不用冰场几何，<br>检验相机能否解释背景运动。</h1>
<div class="card"><span class="badge">这是条件性图像检验，不是物理标定证书。</span><p>只读取原片；未修改主求解器、场景、相机状态或已认可结果。相机高度与米制尺度不在本验证器的可测范围内。</p></div>
<div class="card"><b>{used} / {total} 个采样曝光进入本次检验</b><p>{m['training_tracks']} 条拟合纹理轨迹 · {m['holdout_tracks']} 条整轨迹留出数据。匹配来自像素纹理，不来自冰场曲线或已有相机解。未连接曝光与全部筛除记录保存在 graph.json / graph_diagnostics.json。</p></div>
<h2>内参模型对照</h2><p>误差单位是原片像素，采用双向转移误差。表中同时显示全部留出与训练单应性筛查后的留出子集。筛查会偏向单应性运动，所以两种口径都必须保留；未参与拟合不等于新时间段或独立采集的数据。</p>
<table><tr><th>模型</th><th>fx / fy</th><th>全部中位数</th><th>全部 P95</th><th>筛后中位数</th><th>筛后 P95</th><th>结论</th></tr>{''.join(rows)}</table>
<p>无约束逐对单应性留出中位数：{num(hd['holdout_screened']['median_px'])} px。它是图像运动参照，不是相机真值；平面场景与纯旋转都可能产生单应性。</p>
<h2>必须看的限制</h2><div class="card"><p>当前独立候选采用固定光心、全局曝光近似。云台偏心、近处围挡视差、滚动快门和防抖都可能使它失效。若候选与当前内参不一致，只能报告“在该近似下不一致”，不能直接宣布当前相机错误。</p><p>主点固定、方形像素、共享低阶径向项均是显式模型假设，不是 Sony Z190 的已知标定规律。多初值是局部搜索；停止状态、触边与弱方向一并列出。</p></div>
<h2>自动挑出的核对画面</h2><p>黄点是未参与相机拟合的匹配，红叉是独立模型的预测；连线仅表示对应关系。人工不必逐帧重标，但这些图也不能替代完整数值检查。</p>
{''.join('<img src="evidence/'+html.escape(p)+'" loading="lazy">' for p in images)}
<h2>机器可读证据</h2><p><a href="summary.json">summary.json</a> · <a href="graph.json">graph.json</a> · <a href="homography_baseline.json">homography_baseline.json</a></p>
<details><summary>模型诊断</summary><pre>{html.escape(json.dumps(diagnostics,ensure_ascii=False,indent=2,default=lambda x:x.tolist()))}</pre></details>'''
    if comparison is not None:
        body += '<h2>现有相机的只读对照</h2><p>独立拟合完成后才加载现有相机。以下均不使用场地几何；均未补偿滚动快门和光心平移，所以不是对原生完整投影的公平替代。</p><table><tr><th>对照</th><th>全部留出中位数</th><th>全部 P95</th><th>筛后中位数</th><th>筛后 P95</th></tr>'
        for title, key in [('原内参＋原瞬时旋转', 'current_rotation_and_optics_midframe'), ('冻结原内参，重新估计旋转', 'current_optics_frozen_rotations_refit')]:
            row = comparison[key]
            body += f'<tr><td>{title}</td><td>{num(row["holdout_all"]["median_px"])}</td><td>{num(row["holdout_all"]["p95_px"])}</td><td>{num(row["holdout_screened"]["median_px"])}</td><td>{num(row["holdout_screened"]["p95_px"])}</td></tr>'
        body += '</table><details><summary>原相机对照的停止状态与完整数字</summary><pre>'+html.escape(json.dumps(comparison,ensure_ascii=False,indent=2))+'</pre></details>'
    if profile is not None:
        body += '<h2>固定像素比例，重新拟合其它参数</h2><p>这是局部参数剖面，不是置信区间。未收敛行不能用来判断某一比例已被排除。</p><table><tr><th>fx/fy</th><th>训练目标</th><th>筛后留出中位数</th><th>优化停止</th></tr>'
        for row in profile:
            body += f'<tr><td>{num(row["fx_over_fy"])}</td><td>{num(row["train_robust_cost"])}</td><td>{num(row["holdout_screened"]["median_px"])}</td><td>{"已满足停止条件" if row["optimizer_success"] else "未收敛"}</td></tr>'
        body += '</table>'
    body += '<p><small>全部输出为独立诊断，不写回 RinkSpace。方法来源与实现边界见交付包 METHODS.md。</small></p></main></html>'
    (out/'report.html').write_text(body, encoding='utf-8')
    return result
