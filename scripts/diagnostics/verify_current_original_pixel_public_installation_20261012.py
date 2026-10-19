"""Check actual absolute FIT input installation without motion/J or a solve.

Uses the complete saved material producer, immutable original FIT groups and
the production installer/qualification/compactor. Motion arrays are empty;
scalar offsets therefore describe this absolute-only check, not a full graph.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np

SYS = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
RUN = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_ce235f_wide_curve_budget_resume_20261012')
Q32 = '917c0fa45f80af8e16c74d0f2d56486fdb500b591357add3dc2b240b3de64cdf'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def array_sha(value):
    return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()


def locate(state, item, *, native=False):
    block = next(b for b in state.blocks if b.name == item['block'])
    groups = block.bundle.abs_groups if native else block.groups
    pts, rows = groups[int(item['group_order'])]
    if int(pts) != int(item['group_actual_pts']):
        raise ValueError('Original source group exposure changed')
    return block, rows[int(item['row_order'])]


def row_start(block, item, *, original=False):
    groups = block.original_absolute_groups if original else block.groups
    gi, ri = int(item['group_order']), int(item['row_order'])
    return sum(2 * len(row['uv']) for _, rows in groups[:gi] for row in rows) + sum(2 * len(row['uv']) for row in groups[gi][1][:ri])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--correction', type=Path, default=SYS / 'original_ice_and_TOP_current_all_records_plus_C0_10396_RGB_centers_20261012.json')
    parser.add_argument('--qualification', type=Path, default=SYS / 'original_absolute_same18_plus_C0_10396_four_RGB_unresolved22_qualification_20261012.json')
    parser.add_argument('--config', type=Path, default=ROOT / 'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_ce235f_wide_curve_budget_20261012.json')
    parser.add_argument('--state-identity', type=Path, default=RUN / 'accepted_states' / Q32 / 'state_identity.json')
    parser.add_argument('--verify-integration-policies', action='store_true',
                        help='Also load the configured existing board/ice sampling adapters on these real absolute inputs; no residual/J.')
    parser.add_argument('--output', type=Path, default=SYS / 'original_C0_62record_public_installation_actual_absolute_inputs_20261012.json')
    args = parser.parse_args()

    from calibration.solver_runtime import install
    install()
    from calibration.solver_runtime.original_absolute_input_compaction import compact_original_absolute_inputs, original_points_to_active
    from calibration.solver_runtime.original_source_point_qualification import OriginalSourcePointQualification
    from calibration.solver_runtime.source_point_qualification import file_sha256
    from rinkspace.vision_camera_initialization import current_complete_source_geometry
    from structured_programme_source_candidate.original_ice_pixel_correction import install_original_ice_pixel_correction, require_painted_retirement_compaction
    from z190_moving_pivot_source_bundle import motion_source_csr_layout

    config = read(args.config)
    options = config['source_options']
    evidence_path = Path(options['--original-absolute-input-compaction-source'])
    correction = read(args.correction)
    original_path = Path(correction['original_absolute_packet']['path'])
    original_sha = file_sha256(original_path)
    if original_sha != correction['original_absolute_packet']['sha256']:
        raise ValueError('Original absolute packet bytes differ')
    original = pickle.loads(original_path.read_bytes())['programme_absolute_rows']
    identity = read(args.state_identity)
    state_path = args.state_identity.parent / 'complete_state_UNACCEPTED.npz'
    geometry, world, geometry_receipt = current_complete_source_geometry(state_path, options['--world-init'])
    with np.load(state_path, allow_pickle=False) as arrays:
        q = arrays['q'].copy()
    if array_sha(q) != identity['q_sha256']:
        raise ValueError('Saved complete q does not match original state identity')
    q_before = q.tobytes()
    world_before = world.tobytes()
    aliases, group_by_block, original_by_locator = {}, {}, {}
    for ordinal, item in enumerate(original):
        row = item['source_row']
        aliases[int(row['pts'])] = int(item['group_actual_pts'])
        aliases[int(item['group_actual_pts'])] = int(item['group_actual_pts'])
        if item['collection_role'] != 'FIT':
            continue
        groups = group_by_block.setdefault(item['block'], [])
        gi, ri = int(item['group_order']), int(item['row_order'])
        if gi == len(groups):
            groups.append((int(item['group_actual_pts']), []))
        if gi >= len(groups) or groups[gi][0] != int(item['group_actual_pts']) or ri != len(groups[gi][1]):
            raise ValueError('Original FIT group/row order is not exact')
        groups[gi][1].append(deepcopy(row))
        original_by_locator[(item['block'], gi, ri)] = item
    stamps = np.asarray(sorted(set(aliases.values())), dtype=np.int64)
    camera_index = {int(p): i for i, p in enumerate(stamps)}
    blocks = []
    for name, groups in group_by_block.items():
        scales = np.concatenate([np.repeat(np.asarray(r['sigma']), 2) for _, rows in groups for r in rows])
        ia = np.empty(0, dtype=np.int32)
        ib = ia.copy()
        indices, pointer, swap = motion_source_csr_layout(ia, ib, camera_index, groups)
        bundle = SimpleNamespace(motion_rows=0, motion_sigma=np.empty(0), a=np.empty((0, 2)), b=np.empty((0, 2)),
            ia=ia, ib=ib, index=camera_index, stamps=stamps, abs_groups=groups,
            native_residual_sigma=scales.copy(), source_residual_rows=len(scales), indices=indices, pointer=pointer, swap_rows=swap)
        blocks.append(SimpleNamespace(name=name, groups=groups, bundle=bundle, quadratic=np.zeros(len(scales), bool)))
    state = SimpleNamespace(q=q, geometry=geometry, blocks=blocks, original=SimpleNamespace(aliases=aliases))
    for record in correction['records']:
        item = original[record['original_packet_row_index']]
        geometry.register_row(item['source_row'], actual_pts=int(item['group_actual_pts']))
    installed = install_original_ice_pixel_correction(state, args.correction)
    require_painted_retirement_compaction(installed, args.qualification, evidence_path)
    qualified_before = OriginalSourcePointQualification(args.qualification)
    qualified_before.qualify_state(state)
    prequalification_points = qualified_before.contract['active_FIT_points_retired']
    all_scales = np.concatenate([b.bundle.native_residual_sigma for b in state.blocks])
    cap = dict(scales=all_scales, quadratic=np.zeros(len(all_scales), bool))
    template = deepcopy(cap)
    compaction = compact_original_absolute_inputs(state, cap, template, evidence_path, qualification_path=args.qualification)
    qualified_after = OriginalSourcePointQualification(args.qualification)
    qualified_after.qualify_state(state)

    for block in blocks:
        bundle = block.bundle
        expected_scalars = sum(2 * len(r['uv']) for _, rows in block.groups for r in rows)
        if (bundle.source_residual_rows != expected_scalars or len(bundle.pointer) != expected_scalars + 1
                or len(bundle.native_residual_sigma) != expected_scalars or bundle.pointer[-1] != len(bundle.indices)
                or not np.all(np.diff(bundle.pointer) == 8)):
            raise ValueError('Actual compacted native input/CSR layout differs')
        for gi, (_, rows) in enumerate(block.groups):
            for ri, row in enumerate(rows):
                item = original_by_locator[(block.name, gi, ri)]
                old = item['source_row']
                if not np.array_equal(row['world'], old['world']):
                    raise ValueError('Original physical WORLD changed')
                keep = row.get('active_original_absolute_input_compaction', {}).get('retained_original_point_indices', list(range(len(old['uv']))))
                if not np.array_equal(row['sigma'], np.asarray(old['sigma'])[keep]):
                    raise ValueError('Retained original stored sigma changed')

    installed_checks = []
    for record in installed['corrected_rows']:
        item = original[record['original_packet_row_index']]
        block, active = locate(state, item)
        _, native = locate(state, item, native=True)
        positions = original_points_to_active(active, record['point_indices'])
        if not np.array_equal(np.asarray(active['uv'])[positions], record['corrected_uv']):
            raise ValueError('Corrected UV does not resolve in actual compact input')
        if not np.array_equal(native['uv'], active['uv']):
            raise ValueError('Native bundle absolute UV view is stale after public installation')
        installed_checks.append(dict(original_packet_row_index=record['original_packet_row_index'], source_pts=int(item['group_actual_pts']),
            row_id=item['source_row']['id'], corrected_original_point_indices=record['point_indices'], corrected_UV_in_native_input=True))

    c0_item = original[10396]
    c0_record = next(r for r in installed['corrected_rows'] if r['original_packet_row_index'] == 10396)
    c0_block, c0 = locate(state, c0_item)
    keep = c0['active_original_absolute_input_compaction']['retained_original_point_indices']
    measured, retired = [0, 1, 2, 4, 7, 8], [3, 5, 6, 9]
    if keep != measured or c0_record['point_indices'] != measured or sorted(c0_record['retired_original_point_indices']) != retired:
        raise ValueError('C0 must retain exactly six measured points and remove four unresolved points')
    before_start, after_start = row_start(c0_block, c0_item, original=True), row_start(c0_block, c0_item)
    pointer = c0_block.bundle.pointer
    if pointer[after_start + 12] - pointer[after_start] != 12 * 8:
        raise ValueError('C0 native CSR does not contain exactly twelve scalar rows')
    if not np.array_equal(c0_block.bundle.native_residual_sigma[after_start:after_start + 12], np.repeat(c0['sigma'], 2)):
        raise ValueError('C0 native sigma is not the retained unchanged original sigma')
    rebound = geometry.rebind(c0, world)
    if q.tobytes() != q_before or world.tobytes() != world_before or file_sha256(original_path) != original_sha:
        raise ValueError('Complete q/world/original packet changed')
    integration_report = None
    if args.verify_integration_policies:
        from calibration.solver_runtime.original_board_curve_integration import wrap_original_board_curve_integration
        from calibration.solver_runtime.original_ice_curve_integration import wrap_original_ice_curve_integration
        for item in original:
            if item['collection_role'] == 'FIT':
                geometry.register_row(item['source_row'], actual_pts=int(item['group_actual_pts']))
        base = SimpleNamespace(count=len(q), scales=cap['scales'], quadratic=cap['quadratic'], contract={})
        board = wrap_original_board_curve_integration(base, state, options['--original-board-curve-integration-policy'])
        ice = wrap_original_ice_curve_integration(board, state, options['--original-ice-curve-integration-policy'])
        integration_report = dict(board=board.original_board_curve_integration,
                                  ice=ice.original_ice_curve_integration,
                                  residual_J_and_complete_motion_graph_not_evaluated=True)
    report = dict(schema='rinkspace.original-pixel-public-installation-absolute-input-check.v1', passed=True,
        correction=dict(path=str(args.correction.resolve()), sha256=file_sha256(args.correction), records=len(installed_checks)),
        qualification=dict(path=str(args.qualification.resolve()), sha256=file_sha256(args.qualification), retired_points_before=prequalification_points,
            retired_points_after=qualified_after.contract['active_FIT_points_retired'], physically_removed_before_native=True),
        compaction=dict(path=str(evidence_path.resolve()), sha256=file_sha256(evidence_path),
            removed_dense_nearboard_points=compaction['removed_dense_nearboard_points'], removed_unqualified_absolute_points=compaction['removed_unqualified_absolute_points']),
        original_state_identity=dict(path=str(args.state_identity.resolve()), q_sha256=identity['q_sha256']),
        geometry=geometry_receipt, all_absolute_FIT_groups_and_original_locators=True,
        absolute_FIT_block_order=[b.name for b in blocks], absolute_FIT_rows=len(original_by_locator),
        original_complete_q_and_world_unchanged=True, original_packet_bytes_unchanged=True, retained_original_WORLD_and_sigma_exact=True,
        public_installer_qualification_compactor_requalification_passed=True, native_input_and_CSR_physically_compacted=True,
        corrected_rows=installed_checks, actual_sampling_integration=integration_report,
        C0=dict(original_packet_row_index=10396, source_pts=int(c0_item['group_actual_pts']), row_id=c0_item['source_row']['id'],
            retained_original_point_indices=keep, retired_original_point_indices=retired,
            corrected_uv=np.asarray(c0['uv']).tolist(), original_scalar_rows=20, native_scalar_rows=12,
            absolute_only_block_scalar_start_before=before_start, absolute_only_block_scalar_start_after=after_start,
            physical_binding=geometry._binding(c0)[1]['contract']['physical_identity'],
            rebound_WORLD_sha256=array_sha(rebound['world']), original_WORLD_sha256=array_sha(c0_item['source_row']['world'])),
        scope=dict(motion_arrays_empty=True, complete_programme_graph_constructed=False, full_state_history_loaded=False,
            residual_or_J_evaluated=False, fit_or_benchmark_run=False, scalar_offsets_are_absolute_only=True, accuracy_accepted=False))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(passed=True, correction_records=len(installed_checks), C0_measured=keep, C0_retired=retired,
        physical_retirement_points=prequalification_points, output=str(args.output)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
