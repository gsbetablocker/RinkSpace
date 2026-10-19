"""Merge the current user review increments, preserving every prior source row."""
from pathlib import Path
import argparse
from copy import deepcopy
import hashlib
import json
import os
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RINKSPACE_MOTION_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
DATA = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
BASE = DATA/'calibration/runs/solver_acceleration_20261003'
HELP = BASE/'manual_constraint_dependency_fd_20261009/printed_component_extent_candidate'
EVID = DATA/'evidence/user_video_review_20261009/a13_complete_programme'
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts/diagnostics'), str(HELP)]
from verify_actual_printed_extent_candidate import load_state, read, sha
from audit_user_left_goal_line_and_ad_edges_candidate import bind_human
from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
from structured_programme_source_candidate.human_stroke_truth import UserConfirmedSourceTruth
import numpy as np


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
        default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, np.generic) else str(v))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item()), encoding='utf-8')
    return path


def fit_rows(packet):
    return 2 * sum(len(row['uv']) for row in packet['rows'] if row['split'] == 'fit')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--human-audit', type=Path, required=True)
    parser.add_argument('--board-source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--output-config', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True)
    original_path = ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_resume_user_review49_20261009.json'
    original = read(original_path)
    config = deepcopy(original)
    oldopts = original['source_options']
    opts = config['source_options']
    incoming = read(args.human_audit)['candidate_source_paths']
    for key, value in incoming.items():
        assert sha(value['path']) == value['sha256']
        opts[key] = value['path']
    oldice = read(oldopts['--additional-ice-source'])
    ice = read(opts['--additional-ice-source'])
    assert ice['rows'][:len(oldice['rows'])] == oldice['rows']
    r4audit = read(EVID/'R4_right_shift_04_08_40_200/actual_R4_public_same_a13_q4bb_source_append_audit.json')
    r4packet = read(r4audit['source_packet']['path'])
    r4rows = r4packet['rows'][-2:]
    assert all(row['id'].startswith('user_identified_R4_native229174016') for row in r4rows)
    ice['rows'].extend(r4rows)
    ice['current_review_increment_merge'] = dict(human_audit=ref(args.human_audit), R4_audit=ref(EVID/'R4_right_shift_04_08_40_200/actual_R4_public_same_a13_q4bb_source_append_audit.json'), old_source_rows_all_preserved=True)
    icepath = write(args.output_dir/'ice_all_current_user_review_rows.json', ice)
    opts['--additional-ice-source'] = str(icepath)
    board = read(args.board_source)
    oldboard = read(oldopts['--additional-board-source'])
    assert board['rows'][:len(oldboard['rows'])] == oldboard['rows']
    opts['--additional-board-source'] = str(args.board_source)
    confirmed = read(opts['--user-confirmed-source-truth-authority'])
    r4confirmed = read(r4audit['confirmed_authority']['path'])
    confirmed['records'].extend(r4confirmed['records'][-2:])
    for record in confirmed['records']:
        packet_path = icepath if record['family'] == 'native_ice_curve' else args.board_source
        records = ice['rows'] if record['family'] == 'native_ice_curve' else board['rows']
        index = next(i for i, row in enumerate(records) if row['id'] == record['id'])
        digest = hashlib.sha256(canonical(records[index]).encode()).hexdigest()
        assert digest == record['source_row_canonical_json_sha256']
        record['packet_reference'] = ref(packet_path)
        record['original_row_index'] = index
    cp = write(args.output_dir/'confirmed_all_current_user_review_rows.json', confirmed)
    opts['--user-confirmed-source-truth-authority'] = str(cp)
    statepath = Path(opts['--camera-seed']).parent
    state, q, identity, error = load_state(statepath, Path(opts['--world-init']), read(statepath.parent.parent/'input_manifest.json'))
    assert error == 0.
    ob = NativeBoardEdgeFactor(state, oldopts['--additional-board-source'], len(q))
    oi = NativeIceCurveFactor(state, oldopts['--additional-ice-source'], len(q))
    nb = NativeBoardEdgeFactor(state, opts['--additional-board-source'], len(q))
    ni = NativeIceCurveFactor(state, opts['--additional-ice-source'], len(q))
    assert np.array_equal(ob._residual(q, ob.rows), nb._residual(q, nb.rows[:len(ob.rows)]))
    assert np.array_equal(oi._residual(q, oi.rows), ni._residual(q, ni.rows[:len(oi.rows)]))
    assert np.array_equal(oi._residual(q, oi.held), ni._residual(q, ni.held))
    # New board holdouts may append; every original one is unchanged.
    assert np.array_equal(ob._residual(q, ob.held), nb._residual(q, nb.held[:len(ob.held)]))
    oh = bind_human(state, ob, oi, oldopts, len(q))
    nh = bind_human(state, nb, ni, opts, len(q))
    for field in ('records', 'full_stroke_records'):
        before = oh._queries_for_records(q, getattr(oh, field))
        after = nh._queries_for_records(q, getattr(nh, field)[:len(getattr(oh, field))])
        assert canonical([v['query'] for v in before]) == canonical([v['query'] for v in after])
    oc = UserConfirmedSourceTruth(state, oldopts['--user-confirmed-source-truth-authority'], ob, oi, parameter_count=len(q))
    nc = UserConfirmedSourceTruth(state, opts['--user-confirmed-source-truth-authority'], nb, ni, parameter_count=len(q))
    ov, ol, ou, od = oc._carriers(q)
    nv, nl, nu, nd = nc._carriers(q)
    for before, after in ((ov,nv),(ol,nl),(ou,nu)):
        assert np.array_equal(before, after[:len(before)])
    assert canonical(od) == canonical(nd[:len(od)])
    humanqueries = nh._queries_for_records(q, nh.full_stroke_records)
    groups = [dict(id=r['row']['id'], maximum_violation_px=float(np.max(np.abs(v['query']['residual'])))) for r,v in zip(nh.records, humanqueries)]
    source_delta = fit_rows(ice) - fit_rows(oldice) + fit_rows(board) - fit_rows(oldboard) + nh.row_count - oh.row_count
    config['planned_source_rows'] = original['planned_source_rows'] + source_delta
    config['truth_group_count'] = len(nh.records) + 2 + len(nc.records)
    config['source_manifest_sha256'] = sha(ROOT/'calibration/solver_runtime/source_manifest.json')
    config['source_references'] = {key:ref(opts[key]) for key in original['source_references']}
    config['source_append_audit'] = ref(args.human_audit)
    config['previous_configuration'] = ref(original_path)
    config['launch_status'] = 'prepared_current_review_combined'
    report = dict(schema='rinkspace.current-review-source-merge-actual-public-audit.v1', human_audit=ref(args.human_audit),
        seed_q_sha256=identity['q_sha256'], current_source_paths={key:ref(opts[key]) for key in config['source_references']},
        old_all_FIT_and_HELD_sources_and_old49_guard_queries_bounds_bitexact=True,
        original43_zero_limits_preserved_by_same_complete_q_initializer=True,
        old_source_scalar_count=original['planned_source_rows'], source_scalar_delta=source_delta,
        total_source_scalar_count=config['planned_source_rows'], truth_group_count=config['truth_group_count'],
        direct_human_full_queries=groups, confirmed_source_queries=nd, accuracy_accepted=False)
    auditpath = write(args.output_dir/'actual_combined_source_guard_audit.json', report)
    config['combined_source_append_audit'] = ref(auditpath)
    write(args.output_config, config)
    print(json.dumps(dict(config=str(args.output_config), config_sha256=sha(args.output_config), audit=str(auditpath),
        source_rows=config['planned_source_rows'], groups=config['truth_group_count'], direct_human_full_queries=groups[-4:], confirmed_source_queries=nd[-2:]), ensure_ascii=False))


if __name__ == '__main__':
    main()
