"""Read-only model diagnostics for the source-confirmed opening phase timing."""
from pathlib import Path
import argparse
import hashlib
import io
import json

import numpy as np

from .opening_timing import START, END, QUIET_END, QUIET_BODY_JOINTS
from .opening_motion import MotionWindow
from .opening_motion_review import stats, jsonable, rotation_degrees, extrema, digest

TIME_BASE = 12800
RANGES = dict(quiet=(START, QUIET_END), lowering_turn=(QUIET_END, 24.44), complete=(START, END))
JOINTS = dict(right_shoulder=17, right_elbow=19, right_wrist=21)
KEYS = ('P_world_mid', 'Q_world', 'local_R_mid', 'gear_world_mid')
SUMMARY_START = '[opening_timing_summary_begin]'
SUMMARY_END = '[opening_timing_summary_end]'


def guarded(path):
    path = Path(path).resolve()
    if len(str(path).encode('utf-16-le'))//2 >= 260 or len(path.name.encode('utf-16-le'))//2 > 255:
        raise ValueError('Report path exceeds the project Windows path policy')
    return path


def snapshot(path, keys=None):
    """Hash the exact loaded bytes, even if a running fit replaces this path."""
    path = Path(path).resolve()
    data = path.read_bytes()
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        packet = {key: archive[key].copy() for key in (keys or archive.files)}
    return packet, dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def checked_packet(packet, expected_pts=None):
    pts = np.asarray(packet['pts'])
    if pts.ndim != 1 or pts.dtype.kind not in 'iu' or not np.all(np.diff(pts) > 0):
        raise ValueError('Expected strictly increasing integer source PTS')
    np.testing.assert_array_equal(packet['time_base'], [1, TIME_BASE])
    if expected_pts is not None:
        np.testing.assert_array_equal(pts, expected_pts)
    shapes = dict(P_world_mid=(len(pts), 3), Q_world=(len(pts), 3, 3),
                  local_R_mid=(len(pts), 21, 3, 3), gear_world_mid=(len(pts), 2, 48, 3))
    for key, shape in shapes.items():
        if packet[key].shape != shape or not np.isfinite(packet[key]).all():
            raise ValueError('Invalid or nonfinite saved state: '+key)
    return pts


def from_checkpoint(root, output, path, pts):
    from .opening_timing import build
    saved, identity = snapshot(path)
    project = Path(__file__).resolve().parents[3]
    reference, camera, info, _ = build(project, root, output)
    state = MotionWindow(reference, (START, END), 'full')
    np.testing.assert_array_equal(saved['knots'], state.knots)
    np.testing.assert_array_equal(saved['active'], state.active)
    if saved['x'].shape != (state.size,) or not np.isfinite(saved['x']).all():
        raise ValueError('Checkpoint controls do not match the current motion state')
    indices = np.searchsorted(reference.p['pts'], pts)
    np.testing.assert_array_equal(reference.p['pts'][indices], pts)
    reference.refinements.append((state, saved['x']))
    world = reference.world(reference.at(reference.times[indices]))
    ratio = float(info['length_export_ratio'])
    packet = dict(pts=pts.copy(), time_base=np.array([1, TIME_BASE]),
        P_world_mid=world['P']*ratio, Q_world=world['Q'], local_R_mid=world['local_R'],
        gear_world_mid=world['gear']*ratio)
    identity.update(active_columns=saved['active'], knots=saved['knots'],
                    control_count=state.size, checkpoint_is_final=False)
    return packet, identity, dict(camera=info, length_export_ratio=ratio,
        source_times=reference.times[indices], reference_hashes=reference.hashes)


def angles(first, sequence):
    return dict(final_deg=float(rotation_degrees(first, sequence[-1])),
                change_from_range_first_deg=stats(rotation_degrees(first, sequence)))


def summarize(packet, selected, pts):
    P, Q, local, gear = [packet[key][selected] for key in KEYS]
    stamps = pts[selected]
    xy = P[:, :2]
    span = np.linalg.norm(xy[:, None]-xy[None], axis=-1).max()
    blade = np.min(gear[:, :, 14:, 2], axis=(1, 2))
    body = local[:, np.asarray(QUIET_BODY_JOINTS)-1]
    body_angles = rotation_degrees(body[0], body)
    return dict(source_pts=stamps, first_pts=int(stamps[0]), last_pts=int(stamps[-1]), exposures=len(stamps),
        pelvis_net_XYZ_m=P[-1]-P[0], pelvis_XY_net_m=float(np.linalg.norm(xy[-1]-xy[0])),
        pelvis_XY_span_m=float(span), pelvis_XY_path_m=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
        pelvis_Z=dict(start_m=float(P[0, 2]), end_m=float(P[-1, 2]), net_m=float(P[-1, 2]-P[0, 2]),
                      extrema=extrema(P[:, 2], stamps)),
        root_Q=angles(Q[0], Q),
        quiet_body_local={str(j): angles(local[0, j-1], local[:, j-1]) for j in QUIET_BODY_JOINTS},
        quiet_body_maximum_change_deg=float(body_angles.max()),
        right_upper_limb_local={name: angles(local[0, j-1], local[:, j-1]) for name, j in JOINTS.items()},
        minimum_saved_blade_Z_m=stats(blade), minimum_saved_blade_Z_extrema=extrema(blade, stamps))


def trace(packet):
    P, Q, local, gear = [packet[key] for key in KEYS]
    return dict(pelvis_world_m=P, root_Q_change_from_opening_first_deg=rotation_degrees(Q[0], Q),
        quiet_body_native_joint_ids=QUIET_BODY_JOINTS,
        quiet_body_local_change_from_opening_first_deg=rotation_degrees(
            local[0, np.asarray(QUIET_BODY_JOINTS)-1], local[:, np.asarray(QUIET_BODY_JOINTS)-1]),
        right_upper_limb_local_change_from_opening_first_deg={name: rotation_degrees(local[0, j-1], local[:, j-1])
                                                              for name, j in JOINTS.items()},
        minimum_saved_blade_Z_m=np.min(gear[:, :, 14:, 2], axis=(1, 2)))


def summary_text(report):
    lines = [SUMMARY_START, '本轮源时序校正：22.12–22.90秒主要为手部动作，随后才下探；前候选1.11m位移已被用户否定。',
             '以下是同一原始PTS分母的模型诊断，条件米制；不判定接触或准确性通过。',
             '范围22.120–22.900秒（含两端）：']
    labels = dict(old_opening='旧固定姿态版', rejected_motion='被否定动态版', timing='本轮时序候选')
    for mode, row in report['ranges']['quiet']['candidates'].items():
        net = '/'.join(f'{v:.6g}' for v in row['pelvis_net_XYZ_m'])
        z = row['pelvis_Z']
        lines.append(f"{labels[mode]}：XYZ净移{net} 条件m；XY跨度/路径{row['pelvis_XY_span_m']:.6g}/{row['pelvis_XY_path_m']:.6g} 条件m；"
                     f"PZ起止{z['start_m']:.6g}/{z['end_m']:.6g}，变化{z['net_m']:.6g}，范围{z['extrema']['minimum']:.6g}…{z['extrema']['maximum']:.6g} 条件m；"
                     f"根朝向对首帧最大变化{row['root_Q']['change_from_range_first_deg']['max']:.6g}度；躯干腿局部最大变化{row['quiet_body_maximum_change_deg']:.6g}度")
    lines.extend([SUMMARY_END, ''])
    return '\n'.join(lines)


def run(root, output, checkpoint=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    sources, packets = {}, {}
    selected_pts = None
    for name, path in [('old_opening', root/'opening_repair_1102/fitted/result.npz'),
                       ('rejected_motion', root/'opening_motion_1103/fitted/result.npz')]:
        packet, sources[name] = snapshot(path, ('pts', 'time_base', *KEYS))
        full_pts = checked_packet(packet)
        selected = (full_pts >= round(START*TIME_BASE)) & (full_pts <= round(END*TIME_BASE))
        pts = full_pts[selected]
        if selected_pts is None:
            selected_pts = pts
        else:
            np.testing.assert_array_equal(pts, selected_pts)
        packets[name] = {key: value[selected] for key, value in packet.items() if key in KEYS}
    expected = np.arange(round(START*TIME_BASE), round(END*TIME_BASE)+1, 256, dtype=np.int64)
    np.testing.assert_array_equal(selected_pts, expected)
    replay = None
    if checkpoint:
        current, sources['timing'], replay = from_checkpoint(root, output, checkpoint, selected_pts)
    else:
        current, sources['timing'] = snapshot(output/'fitted/result.npz', ('pts', 'time_base', *KEYS))
        checked_packet(current)
        selected = (current['pts'] >= selected_pts[0]) & (current['pts'] <= selected_pts[-1])
        np.testing.assert_array_equal(current['pts'][selected], selected_pts)
        current = {key: value[selected] if key in KEYS or key == 'pts' else value for key, value in current.items()}
    checked_packet(current, selected_pts)
    packets['timing'] = {key: current[key] for key in KEYS}
    ranges = {}
    for name, seconds in RANGES.items():
        selected = (selected_pts >= round(seconds[0]*TIME_BASE)) & (selected_pts <= round(seconds[1]*TIME_BASE))
        stamps = selected_pts[selected]
        np.testing.assert_array_equal(stamps[[0, -1]], np.rint(np.asarray(seconds)*TIME_BASE).astype(np.int64))
        ranges[name] = dict(seconds_inclusive=seconds,
            candidates={mode: summarize(packet, selected, selected_pts) for mode, packet in packets.items()})
    for name in ('inputs.json', 'ledger.json'):
        path = output/name
        if path.is_file():
            sources[name] = dict(path=str(path), sha256=digest(path))
    report = dict(schema='rinkspace.opening-source-timing-review.v1', source_time_base=[1, TIME_BASE],
        source_pts=selected_pts, source_seconds=selected_pts/TIME_BASE, ranges=ranges,
        curves={name: trace(packet) for name, packet in packets.items()}, sources=sources,
        checkpoint=bool(checkpoint), checkpoint_identity=sources['timing'] if checkpoint else None,
        replay=replay, user_correction='Until about22.90 only hands move; lowering follows. Previous1.11m displacement rejected.',
        implementation=dict(path=str(Path(__file__).resolve()), sha256=digest(__file__)),
        definitions=dict(length='Conditional model metres; checkpoint lengths use the same camera length_export_ratio as saved results.',
            angles='SO(3) geodesic changes from each range first frame; not accumulated turn count or anatomical validation.',
            blade='Minimum saved gear samples14:48 relative to frozen Z=0; not an identified physical contact station.',
            ranges='Both endpoints included;22.90 belongs to both adjacent ranges. All candidates use identical exact source PTS.'),
        accuracy_accepted=False, independent_3d_truth=False, automatic_quality_winner=None)
    report = jsonable(report)
    target = guarded(output/('timing_checkpoint.json' if checkpoint else 'timing_report.json'))
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    if not checkpoint:
        text_path = guarded(output/'results.txt')
        previous = text_path.read_text(encoding='utf8') if text_path.is_file() else ''
        if SUMMARY_START in previous:
            first = previous.index(SUMMARY_START)
            last = previous.index(SUMMARY_END, first)+len(SUMMARY_END)
            previous = previous[:first]+previous[last:]
        text_path.write_text(summary_text(report)+'\n'+previous.lstrip('\r\n'), encoding='utf8')
    quiet = report['ranges']['quiet']['candidates']['timing']
    return dict(report=str(target), checkpoint=bool(checkpoint), quiet_pelvis_net_XYZ_m=quiet['pelvis_net_XYZ_m'],
                quiet_pelvis_XY_span_m=quiet['pelvis_XY_span_m'], quiet_pelvis_Z=quiet['pelvis_Z'], accuracy_accepted=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', help='Replay an existing accepted-controls NPZ instead of a final result')
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root, args.output, args.checkpoint), ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()