"""Directional and source-scope checks for programme repair factors on a saved state."""
import argparse
from copy import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import numpy as np

from calibration.program_tracking.body_tracking.absolute_motion_state import (
    absolute_reference,
    curve_from_packet,
)
from calibration.program_tracking.body_tracking.body_clock import source_seconds
from calibration.program_tracking.body_tracking.continuous_physics import ContinuousMassGeometry
from calibration.program_tracking.body_tracking.phase_ledger import flight_mask
from calibration.program_tracking.body_tracking.programme_com import (
    ModelCOMMotionFactor,
    acceleration_operator,
    com_local_jacobians,
    export_com_diagnostics,
)
from calibration.program_tracking.body_tracking.programme_opening import ProgrammeOpeningFactors
from calibration.program_tracking.body_tracking.rebuild_prepare import digest, load
from calibration.program_tracking.body_tracking.sequence_skate_state import (
    SequenceState,
    perturb_relative,
    shoe_curve_from_packet,
)
from calibration.program_tracking.body_tracking.skate_physics import batch_features, fk_state
from calibration.program_tracking.body_tracking.trajectory_backend import load_reference


def saved_problem(project, prepared, parameter):
    saved = load(parameter)
    root, _, _, _, _ = load_reference(project, prepared)
    reference = absolute_reference(root, curve_from_packet(saved), root=prepared)
    reference.relative_skate_curve = shoe_curve_from_packet(saved)
    reference.binding = copy(root.binding)
    reference.binding.reference = reference
    reference.binding.x = saved['x'][:17]
    reference.binding._packet = None
    reference.binding._parameter_key = None
    state = SequenceState(reference, (reference.times[0], reference.times[-1]))
    np.testing.assert_equal(state.body_size, int(saved['body_parameter_count']))
    np.testing.assert_equal(state.shoe_size, int(saved['relative_skate_parameter_count']))
    return SimpleNamespace(reference=reference, body=SimpleNamespace(state=state), size=17+state.size, physics_times=saved['physics_times']), saved


def errors(exact, central, *, rtol=2e-4, atol=1e-7):
    difference = exact-central
    norm = float(np.linalg.norm(central))
    error = float(np.linalg.norm(difference))
    maximum = float(np.max(np.abs(difference), initial=0))
    return dict(
        central_norm=norm,
        relative_direction_error=error/max(norm, 1e-15),
        max_abs_error=maximum,
        passed=bool(error <= rtol*norm or maximum <= atol),
    )


def nonuniform_clock_check():
    clock = np.array([0., .017, .046, .061, .107, .129])
    expected = np.array([2.5, -1.25, .75])
    quadratic = .5*clock[:, None]**2*expected + clock[:, None]*np.array([.1, -.2, 1.])
    measured = acceleration_operator(clock)@quadratic
    return errors(np.broadcast_to(expected, measured.shape), measured, rtol=0, atol=1e-10)


def local_com_check(problem, mass, rng):
    reference = problem.reference
    state = problem.body.state
    indices = np.linspace(0, len(reference.times)-1, 23, dtype=int)
    times = reference.times[indices]
    base = reference.at(times)
    saved_delta = state.increments(np.zeros(state.size), times)
    # Exercise dExp away from zero as well as the actual saved pose.
    delta = saved_delta.copy()
    delta[:, 3:] += rng.normal(scale=.03, size=delta[:, 3:].shape)
    step = 1e-6

    def centre(increments):
        centres, _ = mass.evaluate(base, increments)
        return mass.com(centres)

    def check_pose(increments):
        _, local, shoe = com_local_jacobians(mass, base, increments)
        body = []
        skates = []
        for column in range(69):
            plus, minus = increments.copy(), increments.copy()
            plus[:, column] += step
            minus[:, column] -= step
            finite = (centre(plus)-centre(minus))/(2*step)
            body.append(errors(local[:, :, column], finite, rtol=1e-5, atol=1e-8))
        for foot in range(2):
            for axis in range(3):
                plus = perturb_relative(increments, foot, axis, step)
                minus = perturb_relative(increments, foot, axis, -step)
                finite = (centre(plus)-centre(minus))/(2*step)
                skates.append(errors(shoe[:, :, foot*3+axis], finite, rtol=1e-5, atol=1e-8))

        def summarize(columns):
            return dict(
                directions=len(columns),
                max_relative_error=max(row['relative_direction_error'] for row in columns),
                max_abs_error=max(row['max_abs_error'] for row in columns),
                failed_columns=[i for i, row in enumerate(columns) if not row['passed']],
                passed=all(row['passed'] for row in columns),
            )

        return dict(body=summarize(body), shoes=summarize(skates))

    return dict(sample_frames=len(times), saved_pose=check_pose(saved_delta), nonzero_local_logs=check_pose(delta))


def direction_columns(state):
    body = np.arange(state.body_size)
    dof = body % 69
    joints = (dof-6)//3+1
    return dict(
        random=np.arange(state.size),
        root=body[dof < 6],
        arms=body[np.isin(joints, [13, 14, 16, 17, 18, 19, 20, 21])],
        legs=body[np.isin(joints, [1, 2, 4, 5, 7, 8, 10, 11])],
        shoes=np.arange(state.body_size, state.size),
    )


def global_check(factor, state, rng, groups, *, step=2e-5):
    x = np.zeros(state.size)
    value = factor.evaluate(x, jacobian=True)
    jacobian = value['jac']
    shared_max = float(np.max(np.abs(jacobian[:, :17].data), initial=0))
    state_jacobian = jacobian[:, 17:]
    directions = {}
    for name, columns in groups.items():
        direction = np.zeros(state.size)
        direction[columns] = rng.standard_normal(len(columns))
        direction /= np.linalg.norm(direction)
        plus = factor.evaluate(x+step*direction)['residual']
        minus = factor.evaluate(x-step*direction)['residual']
        central = (plus-minus)/(2*step)
        exact = state_jacobian@direction
        row = errors(exact, central)
        row['passed'] &= row['central_norm'] > 1e-8
        block_rows = {}
        start = 0
        for block in value['weighted_blocks']:
            stop = start+len(block.residual)
            block_rows[block.name] = errors(exact[start:stop], central[start:stop])
            start = stop
        row['blocks'] = block_rows
        directions[name] = row
    return value, dict(
        rows=jacobian.shape[0],
        state_columns=state_jacobian.shape[1],
        jacobian_nnz=state_jacobian.nnz,
        shared_columns_max_abs=shared_max,
        directions=directions,
        passed=shared_max == 0 and all(
            row['passed'] and all(block['passed'] for block in row['blocks'].values())
            for row in directions.values()
        ),
    )


def com_summary(problem, mass):
    reference, state = problem.reference, problem.body.state
    times = reference.times
    centres, rotations = mass.evaluate(reference.at(times), state.increments(np.zeros(state.size), times))
    exported = export_com_diagnostics(mass, times, centres, rotations)
    acceleration = np.linalg.norm(exported['com_acceleration_world_mps2'], axis=1)
    seconds = source_seconds(reference.p)
    lsp = np.flatnonzero((seconds >= 156.2) & (seconds <= 165.2))
    peak = lsp[np.argmax(acceleration[lsp])]
    return dict(
        source_frames=len(times),
        p50_mps2=float(np.percentile(acceleration, 50)),
        p95_mps2=float(np.percentile(acceleration, 95)),
        max_mps2=float(acceleration.max()),
        LSp_156_2_to_165_2=dict(
            max_mps2=float(acceleration[peak]),
            source_pts=int(reference.p['pts'][peak]),
            source_seconds=float(seconds[peak]),
        ),
    )


def flight_counts(reference):
    pts = reference.p['pts'][1:-1]
    episodes = reference.source_document['source_episode_review']['episodes']
    return {
        episode['episode_id']: int(np.count_nonzero(
            (pts >= episode['start_pts']) & (pts <= episode['end_pts_inclusive'])
        ))
        for episode in episodes if episode['kind'] == 'flight_visible_candidate'
    }


def opening_summary(opening, value, fk):
    metrics = value['metrics']['opening_source_standing']
    heights = metrics['minimum_steel_height_m']
    neutral = dict(fk, hand_local_R=np.broadcast_to(np.eye(3), fk['hand_local_R'].shape))
    neutral_rows = opening.operator@opening.features(neutral).ravel()
    hand = opening.rows['opening_source_hand_waist']
    return dict(
        minimum_steel_height_m=dict(
            min=float(np.min(heights)), median=float(np.median(heights)), max=float(np.max(heights)),
        ),
        quiet_P_range_m=metrics['quiet_P_range_m'],
        hand_waist_max_distance_m=metrics['hand_waist_max_distance_m'],
        saved_hand_vs_neutral_max_abs_hand_residual_change=float(
            np.max(np.abs(neutral_rows[hand]-value['residual'][hand])),
        ),
    )


def opening_scope(opening, value, problem, saved, prepared):
    reference, state = problem.reference, problem.body.state
    standing = next(b for b in value['weighted_blocks'] if b.name == 'opening_source_anonymous_standing')
    source = reference.p['pts']
    quiet = (source >= opening.cfg['start_pts']) & (source <= opening.cfg['quiet_end_pts'])
    observed_pts = np.asarray(standing.provenance['source_pts'])
    x = np.zeros(state.size)
    fk = fk_state(reference, opening.base, state.increments(x, opening.times))
    witnesses = batch_features(reference.binding, fk['R'], fk['O'], fk)
    envelope = witnesses['height'][opening.quiet][:, :, witnesses['eligible']].min(axis=(1, 2))
    used_height = value['metrics']['opening_source_standing']['minimum_steel_height_m']
    height_error = float(np.max(np.abs(envelope-used_height), initial=0))
    xy_columns = 17+np.flatnonzero(np.arange(state.body_size) % 69 < 2)
    xy_max = float(np.max(np.abs(standing.jacobian[:, xy_columns].data), initial=0))
    translated = x.copy()
    translated[:state.body_size].reshape(-1, 69)[:, :2] = [.17, -.11]
    translation_error = float(np.max(np.abs(opening.evaluate(translated)['residual']-value['residual'])))
    hand_path = prepared/'initial.npz'
    initial = load(hand_path)
    hands = initial['hand_local_R'][opening.indices]
    hand_error = float(np.max(np.abs(fk['hand_local_R']-hands)))
    saved_hand_identity = digest(hand_path) == str(saved['parent_hand_parameters_sha256'])
    hand_motion = float(np.max(np.abs(hands-hands[0])))
    summary = opening_summary(opening, value, fk)
    return dict(
        **summary,
        standing_rows=len(observed_pts),
        source_pts_first=int(observed_pts[0]),
        source_pts_last=int(observed_pts[-1]),
        exact_quiet_source_scope=bool(np.array_equal(observed_pts, source[quiet])),
        foot_identity_unknown=bool(standing.provenance['foot_identity_unknown']),
        anonymous_two_skate_height_max_abs_error_m=height_error,
        standing_xy_jacobian_max_abs=xy_max,
        common_xy_translation_residual_max_abs_change=translation_error,
        hand_pose_matches_candidate_parent_sha256=saved_hand_identity,
        hand_pose_max_abs_matrix_error=hand_error,
        saved_hand_max_abs_matrix_change=hand_motion,
        passed=bool(
            len(observed_pts) == 40 and np.array_equal(observed_pts, source[quiet])
            and standing.provenance['foot_identity_unknown'] and height_error < 1e-12
            and xy_max == 0 and translation_error < 1e-8 and saved_hand_identity
            and hand_error < 2e-6
            and summary['saved_hand_vs_neutral_max_abs_hand_residual_change'] > 1e-8
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'prepared', 'parameter', 'policy', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding='utf-8'))
    problem, saved = saved_problem(args.project, args.prepared, args.parameter)
    state = problem.body.state
    rng = np.random.default_rng(1108)
    mass = ContinuousMassGeometry(problem.reference)
    clock = nonuniform_clock_check()
    local = local_com_check(problem, mass, rng)
    com = ModelCOMMotionFactor(problem, mass, policy['com_motion_policy'])
    groups = direction_columns(state)
    _, com_result = global_check(com, state, rng, groups, step=1e-3 if com.quadrature!='native_source_centres'else 2e-5)
    flight = flight_mask(problem.reference, problem.reference.p['pts'])[1:-1]
    com_result['excluded_confirmed_flight_frames'] = int(flight.sum())
    com_result['whole_programme_frames'] = len(problem.reference.times)
    com_result['evaluation_samples'] = len(com.times)
    sampled_flight=flight_mask(problem.reference,com.source_pts_float)[1:-1]
    counts = flight_counts(problem.reference)
    com_result['confirmed_flight_frames_by_episode'] = counts
    com_result['acceleration_magnitude'] = com_summary(problem, mass)
    com_result['passed'] &= bool(
        np.array_equal(com.w == 0, sampled_flight) and len(counts) == 5
        and all(counts.values()) and sum(counts.values()) == flight.sum()
    )
    opening = ProgrammeOpeningFactors(problem, policy['programme_opening'])
    opening_value, opening_result = global_check(opening, state, rng, {'random': groups['random']})
    scope = opening_scope(opening, opening_value, problem, saved, args.prepared)
    passed = (
        clock['passed'] and com_result['passed'] and opening_result['passed'] and scope['passed']
        and all(local[pose][kind]['passed'] for pose in ('saved_pose', 'nonzero_local_logs') for kind in ('body', 'shoes'))
    )
    result = dict(
        candidate=str(args.parameter.resolve()), prepared=str(args.prepared.resolve()),
        policy=str(args.policy.resolve()), passed=passed,
        nonuniform_clock=clock, local_com=local, global_com=com_result,
        global_opening=opening_result, opening_scope=scope,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(report=str(args.output.resolve()), passed=passed)), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
