"""Review saved opening motion numerically; never fit, decode or view images.

The two candidates use the same current observation mask. All geometry and
rotation diagnostics remain model quantities, not independent motion truth.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


TIME_BASE = 12800
START_PTS, END_PTS = 283136, 320000
RANGES = {
    'waist_and_lowering': (22.12, 23.10),
    'turning': (23.10, 24.44),
    'complete_window': (22.12, 25.00),
}
STATE_KEYS = ('P_world_mid', 'Q_world', 'local_R_mid',
              'native22_world_mid', 'gear_world_mid')
JOINTS = {'right_shoulder': 17, 'right_elbow': 19, 'right_wrist': 21}
PIXEL_GROUPS = {'right_shoulder_6': (6,), 'right_elbow_8': (8,),
                'right_wrist_10': (10,), 'all_body': tuple(range(17))}


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def stats(value):
    value = np.asarray(value, float).ravel()
    good = value[np.isfinite(value)]
    result = dict(selected_n=int(len(value)), finite_n=int(len(good)),
                  missing_n=int(len(value)-len(good)))
    for name, percentile in [('min', 0), ('p05', 5), ('median', 50),
                             ('p95', 95), ('p99', 99), ('max', 100)]:
        result[name] = float(np.percentile(good, percentile)) if len(good) else None
    return result


def rotation_degrees(first, second):
    first, second = np.asarray(first), np.asarray(second)
    relative = np.swapaxes(first, -1, -2) @ second
    shape = relative.shape[:-2]
    flat = relative.reshape(-1, 3, 3)
    good = np.isfinite(flat).all(axis=(1, 2))
    result = np.full(len(flat), np.nan)
    if good.any():
        result[good] = np.rad2deg(Rotation.from_matrix(flat[good]).magnitude())
    return result.reshape(shape)


def axis_angle_from_up(axis):
    axis = np.asarray(axis, float)
    length = np.linalg.norm(axis, axis=-1)
    ratio = np.divide(axis[..., 2], length, out=np.full(length.shape, np.nan),
                      where=length > 0)
    return np.rad2deg(np.arccos(np.clip(ratio, -1., 1.)))


def angle_at_elbow(native):
    upper = native[:, 17]-native[:, 19]
    lower = native[:, 21]-native[:, 19]
    lengths = np.linalg.norm(upper, axis=1)*np.linalg.norm(lower, axis=1)
    cosine = np.divide(np.sum(upper*lower, axis=1), lengths,
                       out=np.full(len(native), np.nan), where=lengths > 0)
    return np.rad2deg(np.arccos(np.clip(cosine, -1., 1.)))


def heading(Q):
    axis = Q[:, :2, 2]
    norm = np.linalg.norm(axis, axis=1)
    valid = np.isfinite(axis).all(axis=1) & (norm > 1e-8)
    result = np.full(len(Q), np.nan)
    cuts = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    for start, stop in zip(cuts[::2], cuts[1::2]):
        result[start:stop] = np.rad2deg(np.unwrap(
            np.arctan2(axis[start:stop, 1], axis[start:stop, 0])))
    return result, norm


def dynamic_reference(raw, pts):
    np.testing.assert_array_equal(raw['source_time_base'], [1, TIME_BASE])
    stamps = np.asarray(raw['raw_official_pts'])
    if stamps.ndim != 1 or len(stamps) < 2 or not np.all(np.diff(stamps) > 0):
        raise ValueError('Raw official PTS must be unique and strictly increasing')
    if raw['raw_legacy_pts'].shape != stamps.shape:
        raise ValueError('Raw legacy and official identities have different lengths')
    if raw['local_R'].shape != (len(stamps), 21, 3, 3):
        raise ValueError('Raw local_R must contain native joints1..21')
    if raw['Q_world'].shape != (len(stamps), 3, 3):
        raise ValueError('Raw world-root rotations have the wrong shape')
    if pts[0] < stamps[0] or pts[-1] > stamps[-1]:
        raise ValueError('Raw motion does not bracket the complete opening; no extrapolation')
    local = np.stack([Slerp(stamps/TIME_BASE, Rotation.from_matrix(raw['local_R'][:, j]))(
        pts/TIME_BASE).as_matrix() for j in range(21)], axis=1)
    Q = Slerp(stamps/TIME_BASE, Rotation.from_matrix(raw['Q_world']))(pts/TIME_BASE).as_matrix()
    right = np.searchsorted(stamps, pts)
    left = np.maximum(right-1, 0)
    exact = stamps[right] == pts
    left[exact] = right[exact]
    return local, Q, dict(raw_official_pts=stamps, raw_legacy_pts=raw['raw_legacy_pts'],
        interpolation_bracket_pts=np.column_stack((stamps[left], stamps[right])),
        interpolated=~exact, interpolation='per-joint SO(3) Slerp; no smoothing or extrapolation')


def extrema(value, pts):
    value = np.asarray(value)
    good = np.isfinite(value)
    if not good.any():
        return dict(minimum=None, maximum=None, minimum_pts=[], maximum_pts=[])
    low, high = value[good].min(), value[good].max()
    return dict(minimum=float(low), maximum=float(high),
                minimum_pts=pts[value == low], maximum_pts=pts[value == high])


def _twist(relative, axis):
    """Signed twist of a right local correction about its child-rest bone axis."""
    result = np.full(len(relative), np.nan)
    good = np.isfinite(relative).all(axis=(1, 2))
    if not good.any():
        return result
    q = Rotation.from_matrix(relative[good]).as_quat()
    along = q[:, :3] @ axis
    identifiable = np.hypot(along, q[:, 3]) > 1e-10
    angles = np.rad2deg(2*np.arctan2(along, q[:, 3]))
    angles = (angles+180.) % 360.-180.
    angles[~identifiable] = np.nan
    result[good] = angles
    return result


def curves(packet, obs, selection, pts, raw_local, raw_Q, rest_joints=None):
    P, Q, local, native, gear = [packet[key][selection] for key in STATE_KEYS]
    wrist = native[:, 21]
    body_wrist = np.einsum('nji,nj->ni', Q, wrist-P)
    pixel_error = np.linalg.norm(packet['COCO_uv'][selection]-obs['uv'][selection], axis=-1)
    use = np.asarray(obs['use'][selection], bool)
    blade = gear[:, :, 14:, 2]
    blade_min = np.min(blade, axis=(1, 2))
    blade_per_foot = np.min(blade, axis=2)
    root_heading, heading_norm = heading(Q)
    raw_heading, raw_heading_norm = heading(raw_Q)
    result = dict(pelvis_world_m=P, right_wrist_body_coordinates_m=body_wrist,
        right_wrist_relative_to_pelvis_world_m=wrist-P,
        right_wrist_distance_to_native_right_hip_m=np.linalg.norm(wrist-native[:, 2], axis=1),
        right_elbow_internal_angle_deg=angle_at_elbow(native),
        right_wrist_local_change_from_window_first_deg=rotation_degrees(local[0, 20], local[:, 20]),
        torso_from_world_up_deg=axis_angle_from_up(native[:, 12]-native[:, 0]),
        head_neck_axis_from_world_up_deg=axis_angle_from_up(native[:, 15]-native[:, 12]),
        root_up_from_world_up_deg=axis_angle_from_up(Q[:, :, 1]),
        root_heading_unwrapped_deg=root_heading, root_forward_horizontal_norm=heading_norm,
        raw_root_heading_unwrapped_deg=raw_heading, raw_root_forward_horizontal_norm=raw_heading_norm,
        root_angle_from_dynamic_raw_deg=rotation_degrees(raw_Q, Q),
        minimum_saved_blade_z_m=blade_min, per_foot_minimum_saved_blade_z_m=blade_per_foot,
        source_pixel_error_px=np.where(use, pixel_error, np.nan), common_source_use=use,
        local_angle_from_dynamic_raw_deg={name: rotation_degrees(raw_local[:, j-1], local[:, j-1])
                                         for name, j in JOINTS.items()})
    if rest_joints is not None:
        twists, axes = {}, {}
        for name, joint, child in [('right_shoulder', 17, 19), ('right_elbow', 19, 21)]:
            axis = rest_joints[child]-rest_joints[joint]
            length = np.linalg.norm(axis)
            if not np.isfinite(axis).all() or length <= 0:
                continue
            axis = axis/length
            relative = raw_local[:, joint-1].transpose(0, 2, 1) @ local[:, joint-1]
            twists[name] = _twist(relative, axis)
            axes[name] = axis
        result['right_local_twist_correction_from_raw_deg'] = twists
        result['twist_child_rest_axes'] = axes
    dt = np.diff(pts)/TIME_BASE
    result['velocity'] = dict(edge_start_pts=pts[:-1], edge_end_pts=pts[1:],
        pelvis_world_m_s=np.diff(P, axis=0)/dt[:, None],
        right_wrist_local_step_deg=rotation_degrees(local[:-1, 20], local[1:, 20]))
    result['velocity']['right_wrist_local_angular_speed_deg_s'] = (
        result['velocity']['right_wrist_local_step_deg']/dt)
    return result


def range_summary(packet, trace, use, pts, selection):
    P = trace['pelvis_world_m'][selection]
    stamps = pts[selection]
    finite_xy = np.isfinite(P[:, :2]).all()
    xy = dict(net_displacement_m=float(np.linalg.norm(P[-1, :2]-P[0, :2])) if finite_xy else None,
        maximum_pairwise_span_m=float(np.max(np.linalg.norm(P[:, None, :2]-P[None, :, :2], axis=-1))) if finite_xy else None,
        path_length_m=float(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1).sum()) if finite_xy else None)
    velocity = np.diff(P, axis=0)/(np.diff(stamps)/TIME_BASE)[:, None]
    pixel = {}
    for name, ids in PIXEL_GROUPS.items():
        mask = np.asarray(use[selection][:, ids], bool)
        errors = trace['source_pixel_error_px'][selection][:, ids]
        pixel[name] = dict(stats(errors[mask]), joint_ids=list(ids),
                           selected_exposures=int(mask.any(axis=1).sum()))
    minimum = trace['minimum_saved_blade_z_m'][selection]
    local_wrist = packet['local_R_mid'][np.searchsorted(packet['pts'], stamps), 20]
    wrist_body = trace['right_wrist_body_coordinates_m'][selection]
    wrist_step = rotation_degrees(local_wrist[:-1], local_wrist[1:])
    result = dict(source_pts=stamps, source_exposures=len(stamps), source_pixel_error_px=pixel,
        pelvis_XY=xy, pelvis_Z_m=dict(start=float(P[0, 2]), end=float(P[-1, 2]),
            end_minus_start=float(P[-1, 2]-P[0, 2]), summary=stats(P[:, 2]), **extrema(P[:, 2], stamps)),
        pelvis_velocity_m_s=dict(x=stats(velocity[:, 0]), y=stats(velocity[:, 1]),
            z=stats(velocity[:, 2]), speed_XY=stats(np.linalg.norm(velocity[:, :2], axis=1)),
            speed_3D=stats(np.linalg.norm(velocity, axis=1)), absolute_vertical=stats(abs(velocity[:, 2]))),
        right_elbow_internal_angle_deg=stats(trace['right_elbow_internal_angle_deg'][selection]),
        right_wrist_local_change_from_range_first_deg=stats(rotation_degrees(local_wrist[0], local_wrist)),
        right_wrist_local_step_deg=stats(wrist_step),
        right_wrist_local_angular_speed_deg_s=stats(wrist_step/(np.diff(stamps)/TIME_BASE)),
        right_wrist_body_coordinates_m=dict(start=wrist_body[0], end=wrist_body[-1],
            axes=[stats(wrist_body[:, axis]) for axis in range(3)]),
        right_wrist_distance_to_native_right_hip_m=stats(trace['right_wrist_distance_to_native_right_hip_m'][selection]),
        local_angle_from_dynamic_raw_deg={name: stats(values[selection]) for name, values in trace['local_angle_from_dynamic_raw_deg'].items()},
        orientation={key: stats(trace[key][selection]) for key in
            ('torso_from_world_up_deg', 'head_neck_axis_from_world_up_deg', 'root_up_from_world_up_deg', 'root_angle_from_dynamic_raw_deg')},
        blade_height_geometry=dict(minimum_saved_z_m=stats(minimum),
            minimum_z_extrema=extrema(minimum, stamps),
            per_foot_minimum_saved_z_m=[stats(trace['per_foot_minimum_saved_blade_z_m'][selection, foot]) for foot in (0, 1)],
            exposures_any_saved_sample_below_zero=int(np.count_nonzero(minimum < 0)),
            exposures_all_saved_samples_above_zero=int(np.count_nonzero(minimum > 0)),
            exposures_missing=int(np.count_nonzero(~np.isfinite(minimum))),
            penetration_depth_m=stats(np.maximum(-minimum, 0)),
            positive_minimum_height_m=stats(np.maximum(minimum, 0))))
    if 'right_local_twist_correction_from_raw_deg' in trace:
        result['right_local_twist_correction_from_raw_deg'] = {
            name: dict(signed=stats(values[selection]), absolute=stats(abs(values[selection])))
            for name, values in trace['right_local_twist_correction_from_raw_deg'].items()}
    h = trace['root_heading_unwrapped_deg'][selection]
    result['root_heading'] = dict(summary_deg=stats(h),
        end_minus_start_deg=float(h[-1]-h[0]) if np.isfinite(h).all() else None,
        all_horizontal_axes_valid=bool(np.isfinite(h).all()))
    return result


def exact_after_boundary(new, boot, pts):
    selected = pts >= END_PTS
    rows = {}
    for key in STATE_KEYS:
        a, b = new[key][selected], boot[key][selected]
        equal = bool(a.dtype == b.dtype and np.array_equal(a, b))
        frame_equal = np.all(a == b, axis=tuple(range(1, a.ndim)))
        delta = abs(a-b)
        rows[key] = dict(exact_equal=equal, dtype_equal=bool(a.dtype == b.dtype),
            differing_exposures=int(np.count_nonzero(~frame_equal)),
            first_differing_pts=int(pts[selected][~frame_equal][0]) if (~frame_equal).any() else None,
            maximum_absolute_difference=float(delta.max()) if np.isfinite(delta).all() else None)
    return dict(boundary_pts=END_PTS, boundary_inclusive=True, source_exposures=int(selected.sum()),
                all_exact_equal=all(value['exact_equal'] for value in rows.values()), arrays=rows)


def make_report(old, new, obs, inp, raw, boot, rest_joints=None):
    pts = np.asarray(inp['pts'])
    if pts.ndim != 1 or pts.dtype.kind not in 'iu' or not np.all(np.diff(pts) > 0):
        raise ValueError('Input must retain strictly increasing integer source PTS')
    for name, packet in [('old', old), ('new', new), ('boot_parent', boot)]:
        np.testing.assert_array_equal(packet['pts'], pts, err_msg=name+' source identities')
        np.testing.assert_array_equal(packet['time_base'], [1, TIME_BASE])
        expected = dict(P_world_mid=(len(pts), 3), Q_world=(len(pts), 3, 3),
            local_R_mid=(len(pts), 21, 3, 3), native22_world_mid=(len(pts), 22, 3),
            gear_world_mid=(len(pts), 2, 48, 3), COCO_uv=(len(pts), 17, 2))
        for key, shape in expected.items():
            if packet[key].shape != shape:
                raise ValueError(name+' unexpected '+key+' shape')
    np.testing.assert_array_equal(obs['pts'], pts)
    if obs['uv'].shape != (len(pts), 17, 2) or obs['use'].shape != (len(pts), 17):
        raise ValueError('Observation UV/use shape mismatch')
    for key in ('metres_per_scene_unit',):
        np.testing.assert_array_equal(new[key], old[key])
        np.testing.assert_array_equal(new[key], boot[key])
    full = (pts >= START_PTS) & (pts <= END_PTS)
    times = pts[full]
    if not len(times) or times[0] != START_PTS or times[-1] != END_PTS:
        raise ValueError('Missing opening boundary exposures')
    raw_local, raw_Q, provenance = dynamic_reference(raw, times)
    traces = {name: curves(packet, obs, full, times, raw_local, raw_Q, rest_joints)
              for name, packet in [('old_opening', old), ('new_motion', new)]}
    intervals = {}
    for name, seconds in RANGES.items():
        selected = (times >= round(seconds[0]*TIME_BASE)) & (times <= round(seconds[1]*TIME_BASE))
        if not selected.any():
            raise ValueError('Missing source coverage for '+name)
        intervals[name] = dict(seconds_inclusive=seconds,
            **{mode: range_summary(packet, traces[mode], obs['use'][full], times, selected)
               for mode, packet in [('old_opening', old), ('new_motion', new)]})
    return dict(schema='rinkspace.opening-motion-review.v1', source_time_base=[1, TIME_BASE],
        source_pts=times, source_seconds=times/TIME_BASE, ranges=intervals, curves=traces,
        motion_reference=provenance, after_25_boot_parent=exact_after_boundary(new, boot, pts),
        source_observation_policy='The identical current observations.use mask and source UV are applied to both candidates; missing predictions remain in the selected denominator.',
        lower_body_inferred=dict(source_pts=pts[inp['lower_body_inferred']],
            count=int(np.asarray(inp['lower_body_inferred'], bool).sum()),
            interpretation='Original visibility/inference flags; raw local motion does not turn hidden legs into observations.'),
        definitions=dict(length='Conditional model metres, same frozen scene scale in all three packets.',
            velocity='Actual anatomical-pelvis differences divided by exact source-PTS intervals; no smoothing, not measured centre-of-mass speed.',
            wrist_body_coordinates='Q_world.T @ (native21_world - P_world); right hip proxy is native2, not a measured waist surface or hand-contact point.',
            elbow='Internal native17-19-21 angle: 180 degrees is geometrically straight.',
            head='native15 minus native12, a model neck-to-head axis; not an observed face direction.',
            rotations='SO(3) geodesic angles; wrist local change and raw mismatch do not measure anatomical correctness.',
            twist='Swing/twist of raw_local_R.T @ fitted_local_R about the corresponding child-rest bone axis. Model diagnostic only; not a physiological joint bound. A singular decomposition is missing.',
            heading='Root third-column world-XY projection, unwrapped only within nondegenerate segments. No preferred turn count and no 360-degree target.',
            blades='Minimum saved blade samples (gear indices14:48) relative to frozen Z=0. Counts strictly above/below zero use no accuracy threshold; positive minimum is a model clearance, not proof of airborne motion or contact.',
            ranges='Intervals include both endpoints; 23.10s is deliberately shared by the first two summaries.'),
        accuracy_accepted=False, independent_3d_truth=False, automatic_quality_winner=None,
        limitations=['The same source observations are development evidence, not independent validation.',
            'Raw network poses and prepared unseen legs remain priors; inferred motion and hand orientation need source review.',
            'Smaller displacement, twist, elbow angle or heading change is not automatically better.',
            'Source images and videos are neither decoded nor examined by this script.'])


def report_text(report):
    def fmt(value):
        return '缺失' if value is None else f'{value:.6g}'
    lines = ['开头动态修复数值审阅（保存结果；未拟合、未读图）',
        '源时间22.120–25.000秒；全部逐帧PTS与曲线见motion_report.json。',
        '长度、旋转、冰刀高度均为条件模型诊断，不是独立真值；不自动选择优劣。',
        '共同use和源UV用于两版；23.10秒同时计入前后两个分段。', '']
    labels = dict(waist_and_lowering='扶腰与下沉22.12–23.10秒',
                  turning='转身23.10–24.44秒', complete_window='全窗22.12–25.00秒')
    for name, interval in report['ranges'].items():
        lines.append(labels[name])
        for mode, label in [('old_opening', '旧opening'), ('new_motion', '新motion')]:
            row = interval[mode]
            xy, z = row['pelvis_XY'], row['pelvis_Z_m']
            lines += [label+f'，{row["source_exposures"]}曝光：',
                '  骨盆XY净移/跨度/路径（条件m）：'+'/'.join(fmt(xy[k]) for k in ('net_displacement_m', 'maximum_pairwise_span_m', 'path_length_m')),
                '  骨盆Z起/止/最低/最高（条件m）：'+'/'.join(fmt(z[k]) for k in ('start', 'end', 'minimum', 'maximum')),
                '  最低/最高源PTS：'+str(z['minimum_pts'])+' / '+str(z['maximum_pts']),
                '  右肘夹角median/P95（度）：'+fmt(row['right_elbow_internal_angle_deg']['median'])+'/'+fmt(row['right_elbow_internal_angle_deg']['p95']),
                '  右腕相对动态raw角误差median/P95（度）：'+fmt(row['local_angle_from_dynamic_raw_deg']['right_wrist']['median'])+'/'+fmt(row['local_angle_from_dynamic_raw_deg']['right_wrist']['p95'])]
            for group, title in [('right_shoulder_6', '右肩6'), ('right_elbow_8', '右肘8'),
                                 ('right_wrist_10', '右腕10'), ('all_body', '全body')]:
                v = row['source_pixel_error_px'][group]
                lines.append('  '+title+'源像素median/P95/max：'+'/'.join(fmt(v[k]) for k in ('median', 'p95', 'max'))+
                    f' px；选用{v["selected_n"]}，缺失{v["missing_n"]}')
            blade = row['blade_height_geometry']
            lines.append('  最低刀高min/P95/max（条件m）：'+'/'.join(fmt(blade['minimum_saved_z_m'][k]) for k in ('min', 'p95', 'max'))+
                f'；含负刀样本{blade["exposures_any_saved_sample_below_zero"]}曝光，所有刀样本高于平面{blade["exposures_all_saved_samples_above_zero"]}曝光（接触未知）')
        lines.append('')
    invariant = report['after_25_boot_parent']
    lines += ['25.000秒起（含边界）P/Q/local_R/native/gear与boot父版精确相同：'+str(invariant['all_exact_equal']),
        '转数不限定为360度；位移、速度和角度减少不作为自动改善判据。',
        '腕朝向、身体坐标手轨迹、肩肘轴向修正、倾角及逐曝光速度详见JSON。',
        '准确性尚未验收；下肢推断与准备曝光不构成独立接触或三维运动证据。', '']
    return '\n'.join(lines)


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    paths = dict(old=root/'opening_repair_1102/fitted/result.npz',
        new=output/'fitted/result.npz', observations=output/'observations.npz',
        input=root/'input.npz', motion_reference=output/'motion_reference.npz',
        boot_parent=root/'boot_integration_1102/fitted/result.npz')
    sources = {key: dict(path=str(path), sha256=digest(path)) for key, path in paths.items()}
    packets = {key: load(path) for key, path in paths.items()}
    rest_joints = None
    compact = root/'compact_smplx.npz'
    if compact.is_file():
        sources['compact_body'] = dict(path=str(compact), sha256=digest(compact))
        geometry = load(compact)
        rest_joints = geometry['J_template']+np.einsum(
            'k,jck->jc', packets['input']['shared_betas'], geometry['J_shapedirs'])
    report = make_report(packets['old'], packets['new'], packets['observations'], packets['input'],
                         packets['motion_reference'], packets['boot_parent'], rest_joints)
    report.update(sources=sources, implementation=dict(path=str(Path(__file__).resolve()), sha256=digest(__file__)))
    report['twist_diagnostic_available'] = rest_joints is not None
    for source in sources.values():
        if digest(source['path']) != source['sha256']:
            raise ValueError('An input changed during review; no report written')
    report = jsonable(report)
    json_path, text_path = output/'motion_report.json', output/'results.txt'
    for path in (json_path, text_path):
        if len(str(path).encode('utf-16-le'))//2 >= 260 or len(path.name.encode('utf-16-le'))//2 > 255:
            raise ValueError('Report path exceeds the project Windows path policy')
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    text_path.write_text(report_text(report), encoding='utf8')
    return dict(report=str(json_path), text=str(text_path),
                after_25_boot_parent_exact=report['after_25_boot_parent']['all_exact_equal'],
                accuracy_accepted=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Existing body_s01 run root')
    parser.add_argument('--output', required=True, help='Existing opening_motion output directory')
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root, args.output), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
