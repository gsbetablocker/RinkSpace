"""Copy the full programme bundle and replace numerical seeds from saved curves.

This evaluates saved states only: no solver, objective, contact, inference,
GUI or mesh-cache preparation is started. Original inputs remain unchanged.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[3]), str(Path(__file__).resolve().parents[3] / 'src')]

import numpy as np
from scipy.spatial.transform import Rotation
from .rebuild_prepare import load, digest
from .absolute_motion_state import curve_from_packet, evaluate_curve
from .sequence_skate_state import shoe_curve_from_packet

PROGRAMME = Path('events/ankara_2026/programmes/okada_mei_sp')


def summary(values):
    a = np.asarray(values, float)
    if not np.isfinite(a).all():
        raise ValueError('A saved-state difference is nonfinite')
    return dict(median=float(np.median(a)), p95=float(np.percentile(a, 95)), maximum=float(a.max()))


def angular_degrees(a, b):
    relative = np.swapaxes(a, -1, -2) @ b
    return np.degrees(Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()).reshape(relative.shape[:-2])


def sample_curve(curve, times, *, rotations_only=False):
    # Sparse basis, no derivative blocks or dense programme-by-control matrix.
    keys = ('rotations',) if rotations_only else ('P', 'Q', 'local_R')
    pieces = {key: [] for key in keys}
    for first in range(0, len(times), 512):
        sample = evaluate_curve(curve, times[first:first + 512], with_blocks=False)
        for key in keys:
            pieces[key].append(sample[key])
    return {key: np.concatenate(value) for key, value in pieces.items()}


def file_bindings(folder):
    return {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def prepare(project, data_root, *, source=None, output=None, warm=None, write_bundle=True):
    project, data_root = Path(project).resolve(), Path(data_root).resolve()
    sys.path[:0] = [str(project), str(project / 'src')]
    from .programme_contract import required_clock
    from .blade_current_reference import restore_current_reference
    base = data_root / 'runs/body_s01_20261029'
    root = base / 'trajectory_first_1106'
    source = Path(source).resolve() if source else root / 'programme'
    output = Path(output).resolve() if output else root / 'programme_current'
    warm = Path(warm).resolve() if warm else root / 'programme_primary_gpu_v1/solve/last_accepted.npz'
    stage = base / 'support_correction_1106/joint_grounded'
    parameter = project / PROGRAMME / 'spin_support_grounded_1106.npz'
    raw_path = stage / 'common/exmr_hand_nofoot/fitted/result.npz'
    if source == output or output.is_relative_to(source) or not output.is_relative_to(data_root):
        raise ValueError('Output must be a distinct new bundle under the Data root')
    if write_bundle and output.exists():
        raise FileExistsError('Preserve the existing new bundle: ' + str(output))
    old_hashes = file_bindings(source)
    warm_sha = digest(warm)
    packet, initial, checkpoint = load(source / 'input.npz'), load(source / 'initial.npz'), load(warm)
    expected, shots, tb = required_clock(json.loads((project / PROGRAMME / 'rebuild_scope.json').read_text(encoding='utf8')))
    pts = packet['pts']
    np.testing.assert_array_equal(pts, expected)
    np.testing.assert_array_equal(initial['pts'], pts)
    np.testing.assert_array_equal(packet['time_base'], tb)
    if len(pts) != 8329:
        raise ValueError('Require all 8329 official programme exposures')
    if str(checkpoint['prepared_input_sha256']) != old_hashes['input.npz']:
        raise ValueError('The global warm checkpoint belongs to another prepared input')
    if str(checkpoint['prepared_observations_sha256']) != old_hashes['observations.npz']:
        raise ValueError('The global warm checkpoint belongs to other source observations')
    fixed = load(source / 'fixed_binding.npz')
    np.testing.assert_array_equal(checkpoint['x'][:17], fixed['x'])
    whole_times = (pts - pts[0]).astype(np.float64) * tb[0] / tb[1]
    np.testing.assert_allclose(packet['fit_time_seconds'], whole_times, rtol=0, atol=1e-12)
    curve = curve_from_packet(checkpoint, seed=False)
    shoe_curve = shoe_curve_from_packet(checkpoint, seed=False)
    if shoe_curve is None:
        raise ValueError('The global warm checkpoint has no final relative-shoe curve')
    global_state = sample_curve(curve, whole_times)
    global_state['skate_relative_R'] = sample_curve(shoe_curve, whole_times, rotations_only=True)['rotations']
    ref, _, camera_info, parent_parameter = restore_current_reference(project, stage, parameter, seed=False)
    current = ref.at(ref.times)
    spts = ref.p['pts'].astype(np.int64)
    ids = np.searchsorted(pts, spts)
    if np.any(ids >= len(pts)) or not np.array_equal(pts[ids], spts):
        raise ValueError('Current S05 must map by exact integer PTS to the programme')
    if len(spts) != 951 or spts[0] != 972800 or spts[-1] != 1216000:
        raise ValueError('Require the actual current 951-exposure S05 state')
    local_times = (spts - spts[0]).astype(np.float64) / 12800.
    np.testing.assert_allclose(ref.times, local_times, rtol=0, atol=1e-12)
    np.testing.assert_allclose(whole_times[ids] - (spts[0] - pts[0]) / 12800., local_times, rtol=0, atol=2e-14)
    np.testing.assert_allclose(packet['metres_per_scene_unit'], ref.p['metres_per_scene_unit'], rtol=0, atol=1e-12)
    ratio = float(ref.binding.ratio)
    raw = load(raw_path)
    np.testing.assert_array_equal(raw['pts'], spts)
    differences = dict(P_conditional_m=summary(np.linalg.norm(current['P'] * ratio - raw['P_world_mid'], axis=-1)),
        root_degrees=summary(angular_degrees(raw['Q_world_mid'], current['Q'])),
        local_joint_degrees=summary(angular_degrees(raw['local_R_mid'], current['local_R'])))
    if 'skate_relative_R_mid' in raw:
        differences['relative_shoe_degrees'] = summary(angular_degrees(raw['skate_relative_R_mid'], current['skate_relative_R']))
    if 'hand_local_R_mid' in raw and 'hand_local_R' in current:
        differences['hand_degrees'] = summary(angular_degrees(raw['hand_local_R_mid'], current['hand_local_R']))
    raw_matches_old = dict(P_solver_max_abs=float(np.max(abs(initial['P'][ids] - raw['P_world_mid'] / ratio))),
        Q_max_abs=float(np.max(abs(initial['Q'][ids] - raw['Q_world_mid']))),
        local_R_max_abs=float(np.max(abs(initial['local_R'][ids] - raw['local_R_mid']))))
    differs = any(values['maximum'] > 1e-10 for values in differences.values())
    print(json.dumps(dict(event='current_S05_comparison', differs=differs, frames=len(spts),
                         differences=differences, raw_result_matches_old_programme_initial=raw_matches_old)), flush=True)
    if not differs:
        if digest(warm) != warm_sha or file_bindings(source) != old_hashes:
            raise ValueError('A source changed during read-only comparison')
        stage_summary = json.loads((stage / 'run_summary.json').read_text(encoding='utf8'))
        return dict(schema='rinkspace.programme-current-saved-seed-check.v1',
            source=str(source), output=None, frames=len(pts), time_base=list(tb),
            compared_S05_frames=len(spts), compared_S05_pts=[int(spts[0]), int(spts[-1])],
            differences=differences, raw_result_matches_old_programme_initial=raw_matches_old,
            repair_needed=False, replacement_bundle_created=False,
            conclusion_zh='原始common result已经包含当前保存参数的最终修正；本次未发现漏用13步状态。',
            parent_accepted_steps=stage_summary.get('accepted_steps'),
            parent_parameter=dict(path=str(parameter), sha256=digest(parameter)),
            original_S05_result=dict(path=str(raw_path), sha256=digest(raw_path)),
            global_warm_parameter=dict(path=str(warm), sha256=warm_sha),
            source_file_sha256=old_hashes, source_files_preserved_verified=True,
            clock_mapping=dict(programme_origin_pts=int(pts[0]), S05_origin_pts=int(spts[0]),
                S05_origin_offset_in_programme_seconds=float((spts[0] - pts[0]) / 12800.),
                exact_integer_pts_mapping=True),
            no_fit=True, no_network=True, no_GUI=True, no_mesh_preparation=True,
            code=dict(path=str(Path(__file__).resolve()), sha256=digest(Path(__file__).resolve())))
    updated = {key: value.copy() for key, value in initial.items()}
    for key in ('P', 'Q', 'local_R', 'skate_relative_R'):
        updated[key] = global_state[key].copy()
        updated[key][ids] = current[key]
    if 'hand_local_R' in initial:
        updated['hand_local_R'] = initial['hand_local_R'].copy()
        if 'hand_local_R' not in current:
            raise ValueError('Current S05 has no hand state to restore')
        updated['hand_local_R'][ids] = current['hand_local_R']
    outside = np.ones(len(pts), bool)
    outside[ids] = False
    for key in ('P', 'Q', 'local_R', 'skate_relative_R'):
        np.testing.assert_array_equal(updated[key][outside], global_state[key][outside])
        np.testing.assert_array_equal(updated[key][ids], current[key])
    if 'hand_local_R' in updated:
        np.testing.assert_array_equal(updated['hand_local_R'][outside], initial['hand_local_R'][outside])
    for key in ('P', 'Q', 'local_R', 'skate_relative_R', 'hand_local_R'):
        if key in updated and not np.isfinite(updated[key]).all():
            raise ValueError('Nonfinite updated numerical seed: ' + key)
    report = dict(schema='rinkspace.programme-current-saved-seed.v1',
        source=str(source), output=str(output), frames=len(pts), shots=len(set(shots.tolist())), time_base=list(tb),
        source_file_sha256=old_hashes, parent_parameter=dict(path=str(parameter), sha256=digest(parameter)),
        global_warm_parameter=dict(path=str(warm), sha256=warm_sha, final_curve_fields_used=True,
            seed_curve_or_x_reapplication=False),
        original_S05_result=dict(path=str(raw_path), sha256=digest(raw_path)),
        current_S05_vs_original=differences, raw_S05_matches_old_initial=raw_matches_old,
        current_S05_pts=[int(spts[0]), int(spts[-1])], current_S05_frames=len(ids),
        current_S05_programme_rows=[int(ids[0]), int(ids[-1])],
        clocks=dict(programme_origin_pts=int(pts[0]), S05_origin_pts=int(spts[0]),
            S05_origin_offset_in_programme_seconds=float((spts[0] - pts[0]) / 12800.), exact_integer_pts_mapping=True),
        replacement_policy='All P/Q/local_R/relative-shoe seeds from global accepted final curves; exact current S05 rows override only those 951 PTS; hands elsewhere retain original initial',
        S01_explicit_reoverride=False, S01_policy='global warm final curve retained; original satisfied S01 reoverride left to the main session',
        input_npz_unchanged=True, raw_2D_camera_observations_unchanged=True,
        pose_axis_angle_and_raw_root_prior_not_updated=True,
        seed_role='numerical initialization from saved candidates; not network or independent source truth',
        objective_contact_and_fixed_parameters_unchanged=True,
        no_fit=True, no_network=True, no_GUI=True, no_mesh_preparation=True,
        current_restore_info=ref.current_restore_info, camera_info=camera_info)
    if digest(warm) != warm_sha or file_bindings(source) != old_hashes:
        raise ValueError('A source bundle or global checkpoint changed during read-only sampling')
    if not write_bundle:
        return report
    output.mkdir(parents=True)
    links = []
    for relative in old_hashes:
        if relative == 'initial.npz':
            continue
        src, dst = source / relative, output / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
            mode = 'hardlink'
        except OSError:
            shutil.copy2(src, dst)
            mode = 'copy'
        links.append(dict(path=relative, storage=mode))
    np.savez_compressed(output / 'initial.npz', **updated)
    actual = load(output / 'initial.npz')
    for key in updated:
        np.testing.assert_array_equal(actual[key], updated[key])
    new_hashes = file_bindings(output)
    for key in old_hashes:
        if key != 'initial.npz' and new_hashes[key] != old_hashes[key]:
            raise ValueError('Unexpected source/camera/observation change: ' + key)
    if file_bindings(source) != old_hashes:
        raise ValueError('Original bundle was modified')
    report.update(output_file_sha256=new_hashes, unchanged_file_storage=links,
                  source_files_preserved_verified=True,
                  code=dict(path=str(Path(__file__).resolve()), sha256=digest(Path(__file__).resolve())))
    (output / 'current_seed.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(event='programme_current_seed_created', path=str(output), frames=len(pts),
                         report=str(output / 'current_seed.json'))), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'prepare'])
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--data-root', type=Path, default=Path('D:/CodexData/RinkSpace_Data'))
    parser.add_argument('--report', type=Path, help='Optional new Data JSON receipt; does not create a bundle')
    for name in ('source', 'output', 'warm'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        report = prepare(args.project, args.data_root, source=args.source, output=args.output,
                         warm=args.warm, write_bundle=args.action == 'prepare')
    if args.report:
        target = args.report.resolve()
        if not target.is_relative_to(args.data_root.resolve()) or target.exists():
            raise ValueError('Receipt must be a new path under Data')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(output=report['output'], frames=report['frames'],
        repair_needed=report.get('repair_needed', True), report=str(args.report) if args.report else None,
        no_fit=True)), flush=True)


if __name__ == '__main__':
    main()
