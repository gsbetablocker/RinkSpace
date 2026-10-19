"""Append source-reviewed R4 material transitions and replay the public factor."""
from pathlib import Path
import hashlib
import json
import os
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RINKSPACE_MOTION_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
DATA = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
BASE = DATA / 'calibration/runs/solver_acceleration_20261003'
HELP = BASE / 'manual_constraint_dependency_fd_20261009/printed_component_extent_candidate'
EVID = DATA / 'evidence/user_video_review_20261009/a13_complete_programme'
OUT = EVID / 'R4_right_shift_04_08_40_200'
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts/diagnostics'), str(HELP)]
from verify_actual_printed_extent_candidate import load_state, read, sha
from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
from structured_programme_source_candidate.human_stroke_truth import UserConfirmedSourceTruth
import numpy as np


def write(name, value):
    path = OUT / name
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def main():
    audit = read(EVID/'centre_C_04_08_44_933/full_truth_append_candidate/same_q4bb_public_C_centreline_append_preserve49_audit.json')
    paths = audit['candidate_source_paths']
    parent = Path(paths['--additional-ice-source']['path'])
    packet = read(parent)
    rows = read(OUT/'actual_R4_two_material_edges_source_candidate.json')['rows']
    old_count = len(packet['rows'])
    packet['rows'].extend(rows)
    packet['R4_user_source_append'] = dict(parent=ref(parent), measurement=ref(OUT/'actual_R4_two_material_edges_source_candidate.json'),
        actual_source_and_marker_crop_visually_reviewed=True, no_model_used_for_source_coordinates=True,
        previous_rows_exact_prefix=True, old_HELD_split_unchanged=True)
    target = write('ice_parent63_plus_user_R4_actual_material_edges_CANDIDATE.json', packet)
    assert read(target)['rows'][:old_count] == read(parent)['rows']
    authorities = read(paths['--user-confirmed-source-truth-authority']['path'])
    for record in authorities['records']:
        if record['family'] == 'native_ice_curve':
            record['packet_reference'] = ref(target)
    for index, row in enumerate(rows, old_count):
        canonical = json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
        authorities['records'].append(dict(id=row['id'], family='native_ice_curve', packet_reference=ref(target),
            original_row_index=index, source_row_canonical_json_sha256=hashlib.sha256(canonical).hexdigest(),
            pts=row['pts'], actual_pts=row['actual_pts'], curve=row['curve'], uv=row['uv'], sigma_px=row['sigma_px'],
            source_raw_frame=row['source_raw_frame'], source_file_sha256=row['source_file_sha256'],
            user_confirmed_physical_identity=True, model_projection_used_to_generate_or_select_uv=False,
            source_sigma_or_UV_changed=False, practical_location_uncertainty_px=row['sigma_px'],
            practical_uncertainty_is_user_pixel_precision=False,
            user_confirmation_reference=ref(OUT/'source_review.json'),
            source_measurement_reference=ref(OUT/'actual_R4_two_material_edges_source_candidate.json')))
    authority = write('confirmed_parent11_plus_user_R4_two_material_edges_CANDIDATE.json', authorities)
    reports = []
    for run, key in (
        ('okada_full9_zero_support_cache_20261009_065854_736', 'a13c816fb886166cb625a216dbe84ccaed1d612a785a337203be488bb8ad7c4c'),
        ('okada_full9_conditional_views_20261009_091848_377', '4bb1086fb6bf46c80903a0d1e49aa96b7cf1e028448f4e52077c4fa4fb66e345'),
    ):
        directory = BASE/run/'accepted_states'/key
        state, q, identity, error = load_state(directory, BASE/'user_corrected_programme_candidate/latest9a8c_world66_initialization_ONLY.json', read(BASE/run/'input_manifest.json'))
        assert error == 0.
        ice = NativeIceCurveFactor(state, target, len(q))
        old = NativeIceCurveFactor(state, parent, len(q))
        assert np.array_equal(old._residual(q, old.rows), ice._residual(q, ice.rows[:len(old.rows)]))
        assert np.array_equal(old._residual(q, old.held), ice._residual(q, ice.held))
        board = NativeBoardEdgeFactor(state, paths['--additional-board-source']['path'], len(q))
        confirmed = UserConfirmedSourceTruth(state, authority, board, ice, parameter_count=len(q))
        _, _, _, details = confirmed._carriers(q)
        selected = ice.rows[len(old.rows):]
        raw = ice._residual(q, selected)
        reports.append(dict(q_sha256=key, actual_camera_archive_reproduce_max=error,
            old_FIT_and_HELD_residuals_bitexact=True,
            new_rows=[dict(id=row['id'], source_uv=row['uv'].tolist(), raw_model_minus_source_xy_px=(raw[j*6:(j+1)*6].reshape(-1,2)*row['sigma'][:,None]).tolist(), sigma_px=row['sigma'].tolist()) for j,row in enumerate(selected)],
            new_confirmed_queries=details[-2:]))
    out = write('actual_R4_public_same_a13_q4bb_source_append_audit.json', dict(
        schema='rinkspace.actual-R4-source-append-public-replay.v1', source_packet=ref(target), confirmed_authority=ref(authority),
        previous_ice_rows_exact_prefix=True, new_FIT_scalar_rows=12, new_confirmed_groups=2,
        reports=reports, actual_original_image_and_source_marker_visually_reviewed=True,
        source_measurement_model_independent=True, source_normals_not_invented=True,
        source_normal_uncertainty_is_not_user_pixel_precision=True, accuracy_accepted=False))
    print(json.dumps(dict(report=str(out), reports=reports), ensure_ascii=False))


if __name__ == '__main__':
    main()
