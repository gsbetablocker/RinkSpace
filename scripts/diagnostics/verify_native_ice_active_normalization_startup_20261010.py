"""Actual q27fa native87 startup diagnostics, without full source graph/solver."""
from pathlib import Path
import json, sys, time, os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RINKSPACE_MOTION_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
DATA = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
RUNS = DATA/'calibration/runs/solver_acceleration_20261003'
HELP = RUNS/'manual_constraint_dependency_fd_20261009/printed_component_extent_candidate'
HERE = DATA/'diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010'
STATE = RUNS/'okada_full9_current_review71_clarabel_20261009_152520_411/accepted_states/27fa214561703310c832f3884656cf16c9461870a44ed3acf808bd866a4eeb47'
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts/diagnostics'), str(HELP)]
from verify_actual_printed_extent_candidate import load_state, read, sha
from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
from structured_programme_source_candidate.check_native_ice_raw_sigma_revision import check_raw_sigma_revision
from structured_programme_source_candidate.native_ice_directional_diagnostic import (
    record_native_ice_directional_diagnostic, check_actual_native_ice_directional_branches)
from structured_programme_source_candidate.fixed_pivot_positive_domain_fd import (
    positive_pivot_height_columns, positive_pivot_guard_direction)
import numpy as np


def main():
    began = time.perf_counter()
    config = read(ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_review85_corrected_compact_20261010.json')
    options = config['source_options']
    state, q, identity, camera_error = load_state(STATE, Path(options['--world-init']), read(STATE.parent.parent/'input_manifest.json'))
    assert camera_error == 0.
    original_q = q.tobytes()
    factor = NativeIceCurveFactor(state, options['--additional-ice-source'], len(q))
    transform = check_raw_sigma_revision(factor, q)
    block = factor.block(q, True)
    active = factor._qualified_active_fit_view()
    assert active is not None and active.row_count == 3318 and sum(2*len(row['uv']) for row in factor.rows) == 3348
    direction = np.zeros(len(q)); columns = np.unique(block.J.indices)
    direction[columns] = np.random.default_rng(19066).uniform(-.1, .1, len(columns))
    step = 1e-6
    direction, pivot_guard = positive_pivot_guard_direction(q, direction, step, positive_pivot_height_columns(state.blocks))
    plus = factor.block(q + step*direction, False).r
    minus = factor.block(q - step*direction, False).r
    predicted = block.J@direction
    out = HERE/'native87_actual_active_startup_diagnostics'
    out.mkdir(exist_ok=True)
    diagnostic = record_native_ice_directional_diagnostic(factor, q, q, block, direction, step,
        plus, minus, predicted, out, dict(source='actual-q27fa-native87-all-component-columns',
                                       packet_sha256=factor.contract['packet_sha256']))
    observed, qualification = check_actual_native_ice_directional_branches(q, block, direction, step,
        plus, minus, predicted, diagnostic)
    assert diagnostic['branch_replay_completed'], diagnostic.get('diagnostic_error')
    assert diagnostic['source_review_excluded_scalar_rows'] == 30
    assert diagnostic['scalar_rows'] == 3318
    assert qualification['passed'], qualification
    assert original_q == q.tobytes()
    result = dict(schema='rinkspace.actual-native87-active-startup-diagnostics.v1', passed=True,
                  source_packet=dict(path=str(factor.packet_path), sha256=sha(factor.packet_path)),
                  exact_q27fa_state_directory=str(STATE), exact_camera_archive_reproduce_max=camera_error,
                  normalization=transform, native_active_scalar_rows=len(block.r), original_scalar_rows=3348,
                  actual_all_component_J_shape=list(block.J.shape), actual_all_component_J_nnz=block.J.nnz,
                  actual_active_columns=len(columns), actual_directional=qualification, pivot_guard=pivot_guard,
                  actual_directional_branch_replay_completed=True, active_original_point_indices_preserved=True,
                  full_source_graph_or_solver_loaded=False, source_math_or_user_truth_changed=False,
                  complete_programme_accuracy_claimed=False, seconds=time.perf_counter()-began)
    path = HERE/'native87_actual_active_startup_diagnostics.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, report=str(path), seconds=result['seconds'],
                         raw_sigma_R_difference=transform['actual_r_vs_v1_times_sqrt_N_max_absolute'],
                         raw_sigma_J_difference=transform['actual_J_vs_v1_times_sqrt_N_max_absolute'],
                         directional=qualification['checked_max_absolute_error'], allowed=qualification['checked_allowed'])), flush=True)


if __name__ == '__main__':
    main()
