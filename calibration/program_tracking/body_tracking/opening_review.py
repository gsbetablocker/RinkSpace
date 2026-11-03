"""Read-only comparison of the saved opening repair and its boot parent.

All selected source pixels and PTS remain fixed. The report describes conditional
3D geometry, source-pixel agreement and numerical attachment; it does not infer
visible legs, identify a supporting skate, or certify true 3D accuracy.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from rinkspace.path_policy import guard_project_write
from .fullbody_review import load, sha256, jsonable, stats, rotation_degrees, COCO_EDGES


TIME_BASE = 12800
START_PTS = 283136
STATIC_END_PTS = 312832  # 24.440 s, inclusive: 117 real source exposures.
END_PTS = 320000        # 25.000 s, inclusive: 145 exposures in the attachment.
KEY_PTS = (283136, 295936, 308480)  # Exact 22.120, 23.120, 24.100 s.
STATE_KEYS = ('P_world_mid', 'Q_world', 'local_R_mid',
              'native22_world_mid', 'gear_world_mid')
PIXEL_GROUPS = dict(head_shoulders=list(range(7)), arms=list(range(7, 11)))


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.kind in 'fc':
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _inputs(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    paths = dict(parent=root/'boot_integration_1102/fitted/result.npz',
                 fitted=output/'fitted/result.npz', observations=output/'observations.npz',
                 parent_observations=root/'boot_integration_1102/observations.npz',
                 ledger=output/'ledger.json', body_input=root/'input.npz')
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Opening report inputs are not ready; no output written: '+', '.join(missing))
    old, new, obs, parent_obs, inp = [load(paths[key]) for key in
        ('parent', 'fitted', 'observations', 'parent_observations', 'body_input')]
    pts = np.asarray(obs['pts'])
    np.testing.assert_array_equal(pts, np.arange(START_PTS, 575488+256, 256))
    for name, packet in (('parent', old), ('fitted', new),
                         ('parent observations', parent_obs), ('body input', inp)):
        clock = packet.get('pts', packet.get('source_pts'))
        if clock is None:
            raise ValueError('Missing exact source PTS: '+name)
        np.testing.assert_array_equal(clock, pts)
        if 'time_base' in packet:
            np.testing.assert_array_equal(packet['time_base'], [1, TIME_BASE])
    np.testing.assert_array_equal(obs['time_base'], [1, TIME_BASE])
    for packet in (old, new):
        expected = dict(P_world_mid=(1143, 3), Q_world=(1143, 3, 3),
                        local_R_mid=(1143, 21, 3, 3), native22_world_mid=(1143, 22, 3),
                        COCO_uv=(1143, 17, 2))
        for key, shape in expected.items():
            if packet[key].shape != shape:
                raise ValueError('Unexpected state shape: '+key)
        if packet['gear_world_mid'].shape != (1143, 2, 48, 3):
            raise ValueError('Unexpected gear state shape')
    np.testing.assert_equal(new['metres_per_scene_unit'], old['metres_per_scene_unit'])
    return paths, old, new, obs, parent_obs, inp, pts


def planar_motion(P):
    xy = np.asarray(P, float)[:, :2]
    if not np.isfinite(xy).all():
        return dict(frames=len(xy), finite=False, net_displacement_m=None,
                    maximum_pairwise_span_m=None, path_length_m=None)
    pairwise = np.linalg.norm(xy[:, None]-xy[None], axis=-1)
    return dict(frames=len(xy), finite=True,
                net_displacement_m=float(np.linalg.norm(xy[-1]-xy[0])),
                maximum_pairwise_span_m=float(np.max(pairwise)),
                path_length_m=float(np.linalg.norm(np.diff(xy, axis=0), axis=-1).sum()))


def torso_angles(native):
    v = np.asarray(native, float)[:, 12]-np.asarray(native, float)[:, 0]
    length = np.linalg.norm(v, axis=-1)
    ratio = np.divide(v[:, 2], length, out=np.full(len(v), np.nan), where=length > 0)
    return np.rad2deg(np.arccos(np.clip(ratio, -1., 1.)))


def heading_metrics(Q):
    """Expose turn retention without interpreting a degenerate horizontal axis."""
    horizontal = np.asarray(Q, float)[:, :2, 2]
    norm = np.linalg.norm(horizontal, axis=1)
    threshold = 1e-8  # Dimensionless roundoff guard, not an accuracy tolerance.
    valid = np.isfinite(horizontal).all(axis=1) & (norm > threshold)
    unwrapped = np.full(len(Q), np.nan)
    segments = []
    cuts = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    for start, stop in zip(cuts[::2], cuts[1::2]):
        yaw = np.rad2deg(np.unwrap(np.arctan2(horizontal[start:stop, 1], horizontal[start:stop, 0])))
        unwrapped[start:stop] = yaw
        segments.append(dict(first_row=int(start), last_row_inclusive=int(stop-1),
            start_deg=float(yaw[0]), end_deg=float(yaw[-1]), change_deg=float(yaw[-1]-yaw[0]),
            range_deg=[float(yaw.min()), float(yaw.max())]))
    complete = bool(valid.all())
    return dict(axis='root Q third column projected to world XY', valid_exposures=int(valid.sum()),
        invalid_exposures=int((~valid).sum()), horizontal_norm=stats(norm),
        horizontal_degeneracy_threshold=threshold, continuous_over_all_exposures=complete,
        start_deg=float(unwrapped[0]) if complete else None,
        end_deg=float(unwrapped[-1]) if complete else None,
        change_deg=float(unwrapped[-1]-unwrapped[0]) if complete else None,
        range_deg=[float(unwrapped.min()), float(unwrapped.max())] if complete else [None, None],
        unwrapped_heading_deg=unwrapped.tolist(), valid_contiguous_segments=segments,
        interpretation='unwrapped separately within each valid segment; a continuous whole-opening change is reported only when every horizontal projection is nondegenerate; not measured facing direction')


def pixel_metrics(packet, obs, mask):
    error = np.linalg.norm(packet['COCO_uv']-obs['uv'], axis=-1)
    result = {}
    for name, ids in PIXEL_GROUPS.items():
        selected = np.asarray(obs['use'][:, ids], bool) & mask[:, None]
        result[name] = stats(error[:, ids][selected])
        result[name]['joints'] = ids
        result[name]['selected_exposures'] = int(selected.any(axis=1).sum())
    return result


def geometry_metrics(packet, obs, mask):
    native = packet['native22_world_mid'][mask]
    blades = packet['gear_world_mid'][mask, :, 14:, 2]
    minimum = np.min(blades, axis=(1, 2))
    finite = minimum[np.isfinite(minimum)]
    return dict(root_XY=planar_motion(packet['P_world_mid'][mask]),
                torso_from_world_up_deg=stats(torso_angles(native)),
                root_heading=heading_metrics(packet['Q_world'][mask]),
                blade_minimum_z_m=dict(summary=stats(minimum),
                    range=[float(finite.min()), float(finite.max())] if len(finite) else [None, None],
                    per_foot=[stats(np.min(blades[:, foot], axis=1)) for foot in (0, 1)],
                    exposures_below_plane=int(np.count_nonzero(minimum < 0)),
                    definition='minimum saved blade sample Z, frozen ice Z=0; geometry only, support/contact unknown'),
                upper_body_source_px=pixel_metrics(packet, obs, mask))


def seam_metrics(packet, pts, boundary_pts):
    index = int(np.searchsorted(pts, boundary_pts))
    if index >= len(pts) or pts[index] != boundary_pts:
        raise ValueError('Missing attachment boundary source PTS')
    rows = np.arange(max(0, index-2), min(len(pts), index+3))
    dt = np.diff(pts[rows])/TIME_BASE
    velocity = np.diff(packet['P_world_mid'][rows], axis=0)/dt[:, None]
    root = rotation_degrees(packet['Q_world'][rows[1:]], packet['Q_world'][rows[:-1]])
    local = rotation_degrees(packet['local_R_mid'][rows[1:]], packet['local_R_mid'][rows[:-1]])
    return dict(boundary_pts=int(boundary_pts), boundary_seconds=boundary_pts/TIME_BASE,
                nearby_pts=pts[rows].tolist(), edges_pts=np.column_stack((pts[rows[:-1]], pts[rows[1:]])).tolist(),
                velocity_world_m_s=velocity.tolist(), speed_XY_m_s=np.linalg.norm(velocity[:, :2], axis=1).tolist(),
                speed_3D_m_s=np.linalg.norm(velocity, axis=1).tolist(),
                successive_edge_velocity_change_m_s=np.linalg.norm(np.diff(velocity, axis=0), axis=1).tolist(),
                root_angle_step_deg=root.tolist(), local_angle_step_deg=local.tolist(),
                maximum_local_angle_step_deg=np.max(local, axis=1).tolist(),
                root_angle_speed_deg_s=(root/dt).tolist(),
                maximum_local_angle_speed_deg_s=(np.max(local, axis=1)/dt).tolist())


def _finite_checks(packet):
    return {key: dict(shape=list(value.shape), nonfinite=int(np.count_nonzero(~np.isfinite(value))))
            for key, value in packet.items() if np.asarray(value).dtype.kind in 'fc'}


def _fmt(value):
    return '缺失' if value is None else f'{value:.6g}'


def report_text(report):
    lines = ['开头人体修复自动比较（保存结果，只读审计）',
        '共同源时间：22.120–44.960秒，1143帧；开头条件22.120–24.440秒，117帧。',
        '人体调整22.120–25.000秒，共145帧；24.440–25.000秒为衔接。',
        '125帧下肢继续标为推断；近原地与站立包络是条件先验，不是冰刀接触观测。', '']
    if 'preparation_view' in report:
        prep=report['preparation_view']
        lines += ['准备姿势：原片19.960秒，复用冻结长边副机位，仅拟合本曝光pan/tilt/logf。',
            'B2×F2–R2实际scene交点：'+str(prep['intersection_scene']),
            '冰面检查点未用于MEIWA相机拟合；详细逐点误差见preparation_geometry_check.json。',
            '交叉腿左右映射保留两种候选；选择仅作数值初值，未将物理左右认定为事实。', '']
    for name, title in (('parent', '父版'), ('fitted', '修复版')):
        row = report['opening_117_frames'][name]
        p, a, z = row['root_XY'], row['torso_from_world_up_deg'], row['blade_minimum_z_m']
        lines += [title+'：',
            '骨盆XY净位移/最大两点跨度/路径长（条件米）：'+
            '/'.join(_fmt(p[key]) for key in ('net_displacement_m', 'maximum_pairwise_span_m', 'path_length_m')),
            '躯干native12-0离世界向上角P95/max（度）：'+_fmt(a['p95'])+'/'+_fmt(a['max']),
            '每帧刀刃最低Z范围（条件米）：'+str(z['range'])]
        heading = row['root_heading']
        lines.append('根Q第三列水平投影解包heading起止变化/范围（度）：'+
            _fmt(heading['change_deg'])+' / '+str(heading['range_deg'])+
            f'；非退化曝光{heading["valid_exposures"]}/{heading["valid_exposures"]+heading["invalid_exposures"]}')
        for key, label in (('head_shoulders', '头肩0–6'), ('arms', '手臂7–10')):
            s = row['upper_body_source_px'][key]
            lines.append(f'{label}同一use点像素median/P95：{_fmt(s["median"])}/{_fmt(s["p95"])} px；n={s["n"]}，缺失={s["missing_n"]}')
        lines.append('')
    inv = report['invariants']
    lines += ['25.000秒之后逐数组完全相同：'+str(inv['after_25_state_equal']),
        '观测逐数组未修改：'+str(inv['observation_arrays_unchanged']),
        '推断标志保持原125帧：'+str(inv['lower_body_inferred_unchanged_125']),
        '新结果所有浮点数组无NaN/Inf：'+str(inv['fitted_all_float_arrays_finite']),
        '接缝以下为源曝光相邻差分，不是解析C2证明。']
    for mode in ('parent', 'fitted'):
        for seam in report['seams'][mode]:
            lines.append(f'{mode} {seam["boundary_seconds"]:.3f}s附近XY速度（m/s）：'+str(seam['speed_XY_m_s']))
            lines.append('根角增量/最大局部姿态角增量（度）：'+str(seam['root_angle_step_deg'])+' / '+str(seam['maximum_local_angle_step_deg']))
    lines += ['', '像素统计保留全部共同use点分母，非有限预测计入缺失。',
        '当前报告是开发材料；根位置、尺度、站立与不可见下肢未获得独立三维测量，准确性仍待验收。', '']
    return '\n'.join(lines)


def run(root, output):
    """Compare finished exports only; never start a solver or source decoder."""
    output = Path(output).resolve()
    paths, old, new, obs, parent_obs, inp, pts = _inputs(root, output)
    opening = (pts >= START_PTS) & (pts <= STATIC_END_PTS)
    refinement = (pts >= START_PTS) & (pts <= END_PTS)
    after = pts > END_PTS
    if int(opening.sum()) != 117 or int(refinement.sum()) != 145:
        raise ValueError('Unexpected opening source coverage')
    observation_equal = {key: key in parent_obs and _same(value, parent_obs[key]) for key, value in obs.items()}
    observation_equal.update({key: False for key in parent_obs if key not in obs})
    lower_equal = all(_same(packet['source_lower_body_inferred'], inp['lower_body_inferred'])
                      for packet in (old, new)) and int(inp['lower_body_inferred'].sum()) == 125
    outside_equal = {key: _same(new[key][after], old[key][after]) for key in STATE_KEYS}
    finite = dict(parent=_finite_checks(old), fitted=_finite_checks(new))
    finite_new = all(item['nonfinite'] == 0 for item in finite['fitted'].values())
    contracts = dict(observation_arrays_unchanged=all(observation_equal.values()),
                     lower_body_inferred_unchanged_125=bool(lower_equal),
                     refinement_mask_matches_145=_same(new['body_refinement_window_mask'], refinement),
                     static_condition_mask_matches_117=_same(new['opening_static_condition_mask'], opening),
                     opening_inferred_mask_preserves_125=_same(new['opening_inferred_mask'], inp['lower_body_inferred']),
                     after_25_state_equal=all(outside_equal.values()),
                     fitted_all_float_arrays_finite=finite_new)
    report = dict(schema='rinkspace.opening-repair-review.v1', source_time_base=[1, TIME_BASE],
        scope=dict(source_frames=len(pts), source_seconds=[pts[0]/TIME_BASE, pts[-1]/TIME_BASE],
                   opening_condition_seconds=[22.12, 24.44], opening_condition_frames=117,
                   body_refinement_seconds=[22.12, 25.0], body_refinement_frames=145,
                   lower_body_inferred_frames=int(inp['lower_body_inferred'].sum()),
                   after_25_frames=int(after.sum())),
        sources={name: dict(path=str(path), sha256=sha256(path)) for name, path in paths.items()},
        invariants={**contracts, 'all_passed': all(contracts.values())},
        observation_array_equality=observation_equal, after_25_array_equality=outside_equal,
        at_25_array_equality={key: _same(new[key][pts == END_PTS], old[key][pts == END_PTS]) for key in STATE_KEYS},
        numeric_finiteness=finite,
        opening_117_frames={name: geometry_metrics(packet, obs, opening) for name, packet in (('parent', old), ('fitted', new))},
        complete_refinement_145_frames={name: geometry_metrics(packet, obs, refinement) for name, packet in (('parent', old), ('fitted', new))},
        seams={name: [seam_metrics(packet, pts, point) for point in (START_PTS, STATIC_END_PTS, END_PTS)]
               for name, packet in (('parent', old), ('fitted', new))},
        metric_definitions=dict(root='anatomical pelvis, world XY; conditional metres',
            torso='angle of native joint12 minus joint0 to +world Z, 0 upright and 180 inverted',
            pixels='same original observations.use mask, Euclidean source-pixel errors; missing predictions retained',
            motion='finite differences of exact source PTS; not measured centre-of-mass or blade speed'),
        accuracy_accepted=False, limitations=[
            'Near-static root and weak blade-envelope height are engineering conditions, not source contact measurements.',
            'The 125 inferred lower-body exposures remain inferred; no physical supporting skate or blade point is identified.',
            'Source-pixel improvements and numerical attachment do not establish independent 3D accuracy.'])
    camera_path=output/'preparation_camera.json'
    if camera_path.is_file():
        prep=json.loads(camera_path.read_text(encoding='utf8'))
        pose=json.loads((output/'preparation_pose.json').read_text(encoding='utf8'))
        report['preparation_view']=dict(source_pts=255488,source_seconds=19.96,
            intersection_scene=prep['intersection_scene'],intersection_source_uv=prep['intersection_source_uv'],
            fitted_source_groups=prep['fitted_source_groups'],pose=pose,
            geometry_check=json.loads((output/'preparation_geometry_check.json').read_text(encoding='utf8')),
            body_reference_neighbourhood_scale_m=.35,anchor_is_exact_contact=False)
    for filename in ('opening_report.json', 'results.txt'):
        guard_project_write(output/filename)
    (output/'opening_report.json').write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    (output/'results.txt').write_text(report_text(report), encoding='utf8')
    return report


def source_figures(root, output):
    """Draw three exact original images and saved old/new projected skeletons.

    The simple shoulder/hip outline is a model proxy, not a measured contour.
    Unknown lower-body source points are never displayed as observations.
    """
    from .fullbody_review import exact_frames, find_source
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths, old, new, obs, _, inp, pts = _inputs(root, output)
    output = Path(output).resolve()
    selected = np.searchsorted(pts, KEY_PTS)
    np.testing.assert_array_equal(pts[selected], KEY_PTS)
    source = find_source(Path(root).resolve())
    frames = exact_frames(source['path'], KEY_PTS)
    fig, axes = plt.subplots(3, 3, figsize=(18, 10.5), squeeze=False, constrained_layout=True)
    record = dict(schema='rinkspace.opening-source-comparison.v1', source_time_base=[1, TIME_BASE], source=source,
        sources={name: dict(path=str(paths[name]), sha256=sha256(paths[name])) for name in ('parent', 'fitted', 'observations', 'ledger')},
        columns=['exact original source', 'parent saved projection', 'opening repair saved projection'],
        pixels='full original source, unwarped and uncropped; saved rolling-row COCO_uv only',
        legend=dict(predicted_skeleton='#11d6ff', model_torso_proxy='#ee63ff', actual_selected_source_points='#ffbd22'),
        outline='four-segment shoulder/hip proxy from saved model COCO joints, not a source silhouette measurement', frames=[])
    for row, (point, fi) in enumerate(zip(KEY_PTS, selected)):
        image = frames[point]
        h, w = image.shape[:2]
        use = np.asarray(obs['use'][fi], bool).copy()
        if inp['lower_body_inferred'][fi]:
            use[11:] = False
        frame_record = dict(pts=point, seconds=point/TIME_BASE, source_shape=list(image.shape),
            original_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            lower_body_inferred=bool(inp['lower_body_inferred'][fi]),
            displayed_source_joint_ids=np.flatnonzero(use).tolist(), modes={})
        for col, (mode, packet) in enumerate((('raw', None), ('parent', old), ('fitted', new))):
            ax = axes[row, col]
            ax.imshow(image)
            if packet is not None:
                uv = packet['COCO_uv'][fi]
                for a, b in COCO_EDGES:
                    if not np.isfinite(uv[[a, b]]).all():
                        continue
                    inferred = bool(inp['lower_body_inferred'][fi]) and (a >= 11 or b >= 11)
                    ax.plot(uv[[a, b], 0], uv[[a, b], 1], color='#11d6ff',
                            lw=1.0, ls='--' if inferred else '-', alpha=.85)
                for a, b in ((5, 6), (6, 12), (12, 11), (11, 5)):
                    if np.isfinite(uv[[a, b]]).all():
                        ax.plot(uv[[a, b], 0], uv[[a, b], 1], color='#ee63ff', lw=.8,
                                ls='--' if inp['lower_body_inferred'][fi] and (a >= 11 or b >= 11) else '-')
                source_uv = obs['uv'][fi, use]
                ax.scatter(source_uv[:, 0], source_uv[:, 1], s=11, facecolors='none', edgecolors='#ffbd22', linewidths=.8)
                frame_mask = np.zeros(len(pts), bool); frame_mask[fi] = True
                frame_record['modes'][mode] = dict(source_px=pixel_metrics(packet, obs, frame_mask),
                    source_observations_from_current_exposure_only=True,
                    inferred_model_segments_dashed=True)
            ax.set_title(f'{point/TIME_BASE:.3f}s | PTS {point} | {mode}', fontsize=10)
            ax.set_xlim(-.5, w-.5); ax.set_ylim(h-.5, -.5); ax.set_aspect('equal'); ax.axis('off')
        record['frames'].append(frame_record)
    fig.suptitle('Exact source | cyan: predicted skeleton | magenta: model torso proxy | gold: same selected source points\nDashed lower-body predictions remain inferred; no invisible lower-body observations shown', fontsize=11)
    image_path = guard_project_write(output/'source_opening.png')
    metadata_path = guard_project_write(output/'source_opening.json')
    fig.savefig(image_path, dpi=160); plt.close(fig)
    record['image'] = dict(path=str(image_path), sha256=sha256(image_path))
    metadata_path.write_text(json.dumps(jsonable(record), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    return dict(image=str(image_path), source_record=str(metadata_path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--source-figures', action='store_true')
    args = parser.parse_args(argv)
    if args.source_figures:
        result = source_figures(args.root, args.output)
    else:
        report = run(args.root, args.output)
        result = dict(report=str(Path(args.output)/'opening_report.json'), invariants=report['invariants'])
    print(json.dumps(jsonable(result), ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
