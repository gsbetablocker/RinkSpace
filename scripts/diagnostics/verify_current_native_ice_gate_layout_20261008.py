"""Replay the existing small native-ice gate on an explicit complete layout."""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
from scipy.sparse import csr_matrix
from types import SimpleNamespace
from verify_original68_motion_role_production_20261008 import producer_state, arrays, sha
from verify_current_native_primitives_20261007 import state_camera_adapter
from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
from structured_programme_source_candidate.native_ice_directional_diagnostic import (
    record_native_ice_directional_diagnostic, check_actual_native_ice_directional_branches)
from structured_programme_source_candidate.fixed_pivot_positive_domain_fd import (
    positive_pivot_height_columns, positive_pivot_guard_direction)


def main():
    ap = argparse.ArgumentParser()
    for key in ('state', 'auxiliary', 'role-packet', 'ice-packet', 'output'):
        ap.add_argument('--' + key, required=True, type=Path)
    ap.add_argument('--reuse-numeric', type=Path,
                    help='Replay saved exact caller small r/J and direction; no derivative is rebuilt')
    args = ap.parse_args()
    start = time.perf_counter()
    state, q, saved, _, _, _, contract, _ = producer_state(
        args.state, args.auxiliary, args.role_packet)
    identity = json.loads((args.state.parent / 'state_identity.json').read_text())
    adapter, reproduction = state_camera_adapter(saved, arrays(identity['camera_path']), state.geometry)
    for block in state.blocks:
        for stamp in block.bundle.stamps:
            adapter.bypts[int(stamp)] = block
    count = state.structured_programme_coordinate_chart.full_count
    factor = NativeIceCurveFactor(adapter, args.ice_packet, count)
    factor.canonical_parameter_layout = adapter.canonical_parameter_layout
    local = q[:count].copy()
    j_start = time.perf_counter()
    if args.reuse_numeric is None:
        block = factor.block(local, True)
        active = np.unique(block.J.indices)
        direction = np.zeros(count)
        direction[active] = np.random.default_rng(19066).uniform(-.1, .1, len(active))
        step = 1e-6
        direction, pivot = positive_pivot_guard_direction(
            local, direction, step, positive_pivot_height_columns(factor.blocks.values()))
    else:
        old = arrays(args.reuse_numeric)
        assert np.array_equal(old['fullq'], saved['q']) and np.array_equal(old['extraq'], local)
        assert np.array_equal(old['scales'], factor.scales)
        assert np.array_equal(old['quadratic'], factor.quadratic)
        assert tuple(old['J_shape']) == (factor.row_count, factor.parameter_count)
        matrix = csr_matrix((old['J_data'], old['J_indices'], old['J_indptr']), shape=tuple(old['J_shape']))
        block = SimpleNamespace(r=old['base'], J=matrix, scales=factor.scales, quadratic=factor.quadratic)
        direction, step, pivot = old['direction'], float(old['step']), dict(saved_exact_small_direction=True)
    j_seconds = time.perf_counter() - j_start
    # This is the ordinary central gate; explicit print-domain legality is
    # checked first. A physical-boundary case belongs to the existing driver.
    for sign in (-1, 1):
        state.geometry._q((local + sign * step * direction)[state.geometry_parameter_indices])
    plus = factor.block(local + step * direction, False).r
    minus = factor.block(local - step * direction, False).r
    predicted = block.J @ direction
    if args.reuse_numeric is not None:
        assert np.array_equal(old['predicted'], predicted)
        assert np.array_equal(old['plus'], plus) and np.array_equal(old['minus'], minus)
        assert np.array_equal(old['base'], factor.block(local, False).r)
    args.output.mkdir(parents=True, exist_ok=True)
    diag = record_native_ice_directional_diagnostic(
        factor, local, saved['q'], block, direction, step, plus, minus, predicted, args.output,
        dict(state_sha256=sha(args.state), role_packet_sha256=sha(args.role_packet),
             source_manifest_sha256=sha(Path('calibration/solver_runtime/source_manifest.json'))))
    observed, branch = check_actual_native_ice_directional_branches(
        local, block, direction, step, plus, minus, predicted, diag)
    report = dict(schema='rinkspace.actual-native-ice-gate-explicit-layout-check.v1',
                  state=str(args.state), canonical_parameter_count=len(saved['q']),
                  evaluation_component_parameter_count=count, world_count=len(state.geometry_parameter_indices),
                  packet_sha256=contract['packet_sha256'],
                  camera_reproduction=reproduction, actual_J_seconds=j_seconds,
                  original_error=diag['original_error'], allowed=diag['original_allowed'],
                  checked_error=float(np.abs(observed - predicted).max()),
                  branch_qualification=branch, pivot_guard=pivot,
                  exact_complete_to_evaluation_mapping=True,
                  cache_statistics=state.original_motion_far_board_roles.exact_cache_statistics,
                  actual_small_J_reused=args.reuse_numeric is not None,
                  wall_seconds=time.perf_counter() - start, passed=branch['passed'], accuracy_accepted=False)
    output = args.output / 'actual_native_ice_gate_explicit_layout_check.json'
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(report=str(output), sha256=sha(output),
                          J_seconds=j_seconds, error=report['checked_error'],
                          allowed=report['allowed'], passed=report['passed'])), flush=True)
    if not report['passed']:
        raise ValueError('Actual existing native-ice gate remains failed; original numeric evidence saved')


if __name__ == '__main__':
    main()
