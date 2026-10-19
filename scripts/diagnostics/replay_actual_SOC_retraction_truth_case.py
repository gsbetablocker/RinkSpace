"""Replay a saved trial's complete truth dependencies without a solve or J.

Trial records retain every shared displacement after recovery.  The public
truth owners must prove that all their dependencies are shared or explicitly
fixed before this partial complete-state replay is allowed.  Unsaved nuisance
coordinates are never claimed to reproduce the full proposed-state hash.
Only stdout is emitted; source images and diagnostic artifacts are not written.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'RINKSPACE_MOTION_THREADS'):
    os.environ[name] = '1'

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts/diagnostics')]
sys.dont_write_bytecode = True

import numpy as np
from calibration.solver_runtime import install
from calibration.solver_runtime.checkpoint_history import read_solver_history


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def emit(**value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def query_vertex(result, vertex=None):
    query = result['query']
    residual = np.asarray(query['residual'], float)
    if residual.ndim == 1:
        residual = residual.reshape(-1, 2)
    if vertex is None:
        vertex, axis = np.unravel_index(np.argmax(abs(residual)), residual.shape)
    else:
        axis = int(np.argmax(abs(residual[vertex])))
    answer = dict(vertex=int(vertex), residual_axis=int(axis),
        maximum_violation_px=float(abs(residual[vertex, axis])),
        source_uv=np.asarray(result.get('uv', []))[vertex].tolist()
            if 'uv' in result else None)
    for key in ('nearest_finite_segment_index', 'nearest_finite_segment_station',
                'nominal_finite_segment_categories', 'signed_normal_px',
                'coverage_tangent_px'):
        if key in query:
            answer[key] = np.asarray(query[key])[vertex].item()
    if 'nearest_finite_segment_station' in answer:
        answer['finite_station_is_interior'] = 0. < answer['nearest_finite_segment_station'] < 1.
    return answer


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run', type=Path, required=True)
    cli.add_argument('--iteration', type=int, default=0)
    cli.add_argument('--trial', type=int, default=3)
    cli.add_argument('--native-group', type=int, default=1)
    cli.add_argument('--expect-recovery-truth-failure', action='store_true')
    args = cli.parse_args()
    install()
    import audit_current_user_ice_truth as public
    from joint_absolute_source_context_20261011 import existing_material_source_byte_alias
    from structured_programme_source_candidate.feasibility_priority_restoration import (
        actual_restoration_trial_decision,
    )
    from installation_mount_candidate.mount_graph import pivot_outside_domain
    run = args.run.resolve()
    config = public.read(run.with_name(run.name + '.bootstrap') / 'input_config.json')
    options = config['source_options']
    manifest = public.read(run / 'input_manifest.json')
    history = read_solver_history(run / 'solver/iterations.json')
    iteration = history[args.iteration]
    trials = [t for t in iteration['global_schur_trial_records'] if t['trial'] == args.trial]
    if len(trials) != 1:
        raise ValueError('One explicitly selected saved trial required')
    trial = trials[0]
    parent = config['continuation']['state']['q_sha256'] if args.iteration == 0 else None
    if parent is None:
        raise ValueError('This replay currently authenticates the saved initial parent only')
    directory = run / 'initial_states' / parent
    state, evaluated_parent, identity, camera_error = public.load_state(
        directory, Path(options['--world-init']), manifest)
    if camera_error != 0.:
        raise ValueError('Public state did not exactly reproduce its saved cameras')
    identity['_accepted_directory'] = str(directory)
    with np.load(directory / 'complete_state_UNACCEPTED.npz', allow_pickle=False) as z:
        canonical_parent = z['q'].copy()
    if digest(canonical_parent) != parent:
        raise ValueError('Saved complete parent q identity differs')
    layout = state.canonical_parameter_layout
    state.current_user_pivot_outside_constraint_contract = manifest['camera_pivot_exterior_constraints']
    selected = np.asarray(iteration['global_schur_linear_report']['global_parameter_indices'], int)
    fixed = np.asarray(iteration['explicit_unobserved_held_state_indices_fixed'], int)
    displacement = np.asarray(trial['shared_step'], float)
    if displacement.shape != selected.shape or not np.isfinite(displacement).all():
        raise ValueError('Complete saved shared displacement required')
    recovered = canonical_parent.copy()
    recovered[selected] += displacement
    proposal = recovered.copy()
    recovery = trial['trial_feasibility_recovery']
    changed = recovery['complete_state_recovery_identity']['changed_complete_state_parameter_indices']
    restored = []
    for change in recovery.get('changes', []):
        columns = np.asarray(change['changed_columns'], int)
        if len(columns) != 2:
            raise ValueError('Only the saved original pivot XY recovery can be reversed')
        before = np.asarray(change['before_pivot_xyz_working_m'], float)[:2]
        after = np.asarray(change['after_pivot_xyz_working_m'], float)[:2]
        if not np.array_equal(recovered[columns], after):
            raise ValueError('Recorded shared trial and recovered pivot differ')
        proposal[columns] = before
        restored.extend(columns.tolist())
    if sorted(restored) != sorted(changed):
        raise ValueError('Every post-SOC recovery coordinate must be explicitly restored')
    board = public.NativeBoardEdgeFactor(state, options['--additional-board-source'], len(canonical_parent))
    ice = public.NativeIceCurveFactor(state, options['--additional-ice-source'], len(canonical_parent))
    human = public.bind_human(state, board, ice, options, len(canonical_parent))
    wide = public.wide_factor(state, identity, options, len(canonical_parent),
        options['--practical-human-location-envelope'])
    with existing_material_source_byte_alias() as source_auth_aliases:
        circle, material, placement = public.existing_LOW_source_owners(state, evaluated_parent,
            directory, options, public.read(directory / 'scene_UNACCEPTED.json'))
    confirmed = public.UserConfirmedSourceTruth(state,
        options['--user-confirmed-source-truth-authority'], board, ice,
        parameter_count=len(canonical_parent), circle_factor=circle,
        material_factor=material, placement_source=placement)
    covered = layout.canonical_to_evaluation[np.r_[selected, fixed]]
    for family, factor in (('native', human), ('wide', wide), ('confirmed', confirmed)):
        if not np.isin(factor.active, covered).all():
            raise ValueError('Unsaved nonshared truth dependency in ' + family)
    snapshots = {}
    pivot_domains = {}
    for label, value in (('parent', canonical_parent), ('before_recovery', proposal),
                         ('after_recovery', recovered)):
        snapshots[label] = public.values(human, wide, confirmed, state.evaluation_q(value))
        pivot_domains[label] = pivot_outside_domain(state, state.evaluation_q(value))
    domain = public.read(run / 'actual_initial_shared_step_domain.json')
    priority = domain['report']['truth_feasibility_priority']
    old = np.concatenate(list(snapshots['parent']['buffers'].values()))
    expected = np.asarray(priority['current_original_group_violations_px'], float)
    if not np.allclose(old, expected, rtol=0., atol=1e-10):
        raise ValueError('Public parent truth values differ from the saved source domain')
    emit(stage='actual_saved_shared_truth_dependencies_authenticated',
        parent_q_sha256=parent, saved_trial=args.trial,
        exact_parent_camera_reproduction_maximum=camera_error,
        every_native_wide_confirmed_dependency_shared_or_explicitly_fixed=True,
        complete_proposal_hash_reproduction_claimed=False,
        saved_complete_recovery_identity=recovery['complete_state_recovery_identity'],
        changed_pivot_columns=changed,
        exact_original_source_byte_aliases=source_auth_aliases,
        source_graph_J_LP_QP_images_or_report_files_created=False)
    critical_vertex = query_vertex(snapshots['after_recovery']['full'][args.native_group])['vertex']
    decisions = {}
    for label in ('before_recovery', 'after_recovery'):
        snapshot = snapshots[label]
        actual = np.concatenate(list(snapshot['buffers'].values()))
        decision = actual_restoration_trial_decision(old, actual,
            common_fraction=priority['demanded_common_original_violation_fraction'],
            step_fraction=trial['line_search_fraction'], physical_guard_passed=True,
            previous_source_cost=0., trial_source_cost=0.)
        decisions[label] = decision
        record = human.records[args.native_group]
        full = snapshot['full'][args.native_group]
        primary = snapshot['primary'][args.native_group]
        details = query_vertex(full)
        details['source_uv'] = human.full_stroke_records[args.native_group]['uv'][details['vertex']].tolist()
        critical = query_vertex(full, critical_vertex)
        critical['source_uv'] = human.full_stroke_records[args.native_group]['uv'][critical_vertex].tolist()
        emit(stage=label, native_group=args.native_group,
            source_id=record['row']['id'], curve=record['row']['curve'],
            actual_native_group_violation_px=float(snapshot['buffers']['native'][args.native_group]),
            primary_maximum_violation_px=float(abs(primary['query']['residual']).max()),
            full_stroke_worst=details, same_critical_vertex_before_after=critical,
            original_pivot_domain=dict(valid=pivot_domains[label]['valid'],
                installations=[dict(id=row['installation_id'],
                    signed_outward_margin_working_m=row['outward_margin_working_m'],
                    valid=row['valid'], nearest_course_segment=row['nearest_segment_index'])
                    for row in pivot_domains[label]['installations']]),
            all86_restoration_necessary_gate=decision,
            physical_guard_result_not_claimed_by_truth_only_replay=True)
    if args.expect_recovery_truth_failure:
        if (not decisions['before_recovery']['accepted']
                or decisions['after_recovery']['accepted']
                or pivot_domains['before_recovery']['valid']
                or not pivot_domains['after_recovery']['valid']):
            raise AssertionError('Saved coupled physical/source recovery regression did not reproduce')
        emit(stage='saved_coupled_physical_source_recovery_regression_passed',
            proposal_truth_passes_but_original_pivot_domain_fails=True,
            recovery_pivot_domain_passes_but_original_truth_fails=True,
            source_support_and_guard_caps_unchanged=True)


if __name__ == '__main__':
    main()
