"""Rebuild the complete source graph using audited native-image pose groups.

The old accepted objective is checked before the new model is constructed.
Every old image measurement and original camera prior target is retained;
only dummy frame nodes and temporal regularization follow the new pose clock.
Source-image equivalence never asserts a new official exposure identity.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

from runtime.bootstrap import add_geometry_dependencies
from rinkspace.storage import resolve_asset
add_geometry_dependencies()
import numpy as np

import fit_continuous_source_identity_repaired as old_entry
from fit_continuous_rig_source_calibrated import release_native_resources, snapshot_dependencies
from continuous_camera_solver import complete_graph_error, solve, save_states, write_json
from continuous_quadratic_rig_optin import add_quadratic_priors
from continuous_full_timeline_regularization import add_complete_motion_priors
from continuous_board_seam_jacobian_optin import install_board_seam_optin, metadata as seam_metadata
from resume_source_identity_repair import load_saved

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TMP = Path('D:/CodexData/RinkSpace_program_tracking_tmp')
base = old_entry.base


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_arguments(output):
    """The frozen source-identity construction, with optimization disabled."""
    return SimpleNamespace(camera='main_long_side',
        warm_group=TMP/'continuous_camera_model_v12_complete_motion/stopped_before_source_identity_repair',
        warm_state='accepted_states.npz',
        prior_state_cache=TMP/'continuous_camera_model_v4_domain_preflight/main_long_side',
        output=output,
        rail_review=TMP/'far_rail_reviewed_source_v1/far_rail_reviewed_source_selection.json',
        rail_review_sha256='61eb15d384d43c16959375c79f6d560d78b8fc291eefaeea7354644e8210fd5b',
        motion_sidecar=TMP/'continuous_source_motion_requalification_v2/main_hint0_source_review_pending.json',
        motion_sidecar_sha256='345da13109dd394288d4de127575d755743d6e784c5dfba4eb12d00b550296e5',
        motion_review=resolve_asset(ROOT, 'evidence/round10_tail_user_issue_20260928/main_hint0_source_review_v2/agent_reviewed_motion_tail_v2_union.json'),
        motion_review_sha256='2083502f51c97314241ad1ca17ad7a81226b854f22f1993514dcb771dd086ea9',
        native_board_first_hit=True, native_central_domain=True,
        central_edge_brown15=True, native_source_edges=True,
        native_edge_threads=8, reference_curve_workers=8,
        native_trial_values=True, native_affine_priors=True,
        adaptive_source_curves=True, implicit_adaptive_curve_jacobian=True,
        initialize_absolute_sources=False, absolute_initializer_resume_from=None,
        nearboard_source_report=None, surface_review=None,
        iterations=0, preflight_only=True)


@contextmanager
def capture_context(context):
    """Capture existing builder inputs without modifying source selection."""
    import historic_development_ads as ads
    import main_nearboard_normal_factors as near
    import continuous_reviewed_surface_motion as motion
    saved = []
    def patch(module, name, recorder):
        fn = getattr(module, name)
        def wrapped(*args, **kwargs):
            result = fn(*args, **kwargs)
            recorder(args, kwargs, result)
            return result
        saved.append((module, name, fn))
        setattr(module, name, wrapped)
    patch(base, 'build_graph', lambda a, k, r: context.update(
        data=a[0], pairs=a[1], model=a[2], prior=a[3], clock=a[4],
        valid_pairs=a[5], graph=r[0], counts=r[1]))
    patch(old_entry, 'attach_reference_factors', lambda a, k, r:
          context.update(references=a[2]))
    patch(ads, 'attach_development_ad_factors', lambda a, k, r:
          context.update(ad_rows=a[4], ad_metadata=r))
    patch(near, 'attach_main_nearboard_normal_factors', lambda a, k, r:
          context.update(near_rows=a[4], near_metadata=r))
    patch(motion, 'attach_reviewed_main_motion_factors', lambda a, k, r:
          context.update(motion_rows=a[4], motion_metadata=k['reviewed_metadata']))
    patch(base, 'export', lambda a, k, r: context.update(official_aliases=dict(a[3])))
    try:
        yield
    finally:
        for module, name, fn in reversed(saved):
            setattr(module, name, fn)


def export_with_pose_identity(model, states, data, pose_aliases, official_aliases, path):
    report = base.export(model, states, data, pose_aliases, path)
    with np.load(path, allow_pickle=False) as z:
        packet = {key: z[key].copy() for key in z.files}
    pts = packet['pts']
    packet['official_exposure_representative_pts'] = np.asarray(
        [official_aliases[int(p)] for p in pts], np.int64)
    packet['exposure_representative_pts'] = packet['official_exposure_representative_pts'].copy()
    packet['source_pose_representative_pts'] = np.asarray(
        [pose_aliases[int(p)] for p in pts], np.int64)
    packet['exposure_representative_semantics'] = np.asarray(
        'unchanged confirmed official exposure identity; use source_pose_representative_pts for the camera state')
    np.savez_compressed(path, **packet)
    report['sha256'] = sha(path)
    return report


def attach_same_ad_observations(graph, context, model, initial):
    from historic_development_ads import attach_development_ad_factors
    reports = []
    # A row's old representative describes its original source observation;
    # a new pose key describes the camera variable. Retain all old UV/weights.
    for original in context['ad_rows']:
        row = dict(original)
        row['original_official_representative_pts'] = row['representative_pts']
        row['representative_pts'] = model.frames[int(row['pts'])].frame_key
        reports.append(attach_development_ad_factors(
            graph, context['data'], model, initial, [row]))
    return dict(original_rows=len(reports), per_source_row=reports,
                source_uv_and_weights_retained=True, newly_deduplicated_rows=0)


def attach_same_near_observations(graph, context, model, initial):
    from main_nearboard_normal_factors import attach_main_nearboard_normal_factors
    reports = []
    for original in context['near_rows']:
        row = dict(original)
        old = int(row['representative_pts'])
        row['original_official_representative_pts'] = old
        row['representative_pts'] = model.frames[old].frame_key
        reports.append(attach_main_nearboard_normal_factors(
            graph, context['data'], model, initial, [row]))
    return dict(original_rows=len(reports), per_source_row=reports,
                source_uv_and_weights_retained=True, newly_deduplicated_rows=0)


def run(options):
    from continuous_source_camera_rebinding import rebind_source_camera
    # Loader is a separate source-only audit; it may not read camera projections.
    from source_image_pose_groups import load_source_pose_groups
    saved_phase, _, warm, provenance = load_saved(options.resume_source)
    continuation = (load_saved(options.resume_clock_source)
                    if options.resume_clock_source is not None else None)
    output = options.output.resolve()
    if output.exists() or not output.is_relative_to(TMP.resolve()):
        raise ValueError('Use a fresh programme TMP output')
    output.mkdir(parents=True)
    resources, context = {}, {}
    started = time.perf_counter()
    install_board_seam_optin()
    try:
        construction = output/'original_graph_reconstruction'
        args = original_arguments(construction)
        with capture_context(context):
            old_entry.run(args, resources)
        old_graph, old_manager = resources['graph'], resources['native_edge_manager']
        rebuilt = json.loads((construction/'phase_00.json').read_text(encoding='utf8'))
        if (set(warm) != set(context['prior'])
                or len(old_graph._pending) != saved_phase['graph_factors_with_source_calibration']
                or rebuilt['factor_counts'] != saved_phase['factor_counts']
                or sha(construction/'source_identity_repair.json') !=
                   saved_phase['source_identity_repair']['sha256']):
            raise ValueError('Original source graph or measurement inventory changed')
        with np.load(options.resume_source/'prior_initializer_states.npz', allow_pickle=False) as z:
            prior = {int(k): v.copy() for k, v in zip(z['keys'], z['states'])}
        if set(prior) != set(context['prior']) or any(
                not np.array_equal(prior[k], context['prior'][k]) for k in prior):
            raise ValueError('Original camera prior targets changed')
        old_cost = complete_graph_error(old_graph, warm, native_edge_manager=old_manager,
            diagnostics_path=output/'old_objective_diagnostics.json')
        if abs(old_cost-provenance['expected_cost']) > max(1e-7, 1e-9*abs(old_cost)):
            raise ValueError('Saved accepted complete objective failed restoration')
        gate = dict(**provenance, restored_cost=old_cost,
                    absolute_difference=abs(old_cost-provenance['expected_cost']),
                    priors_bitwise_equal=True, original_source_manifest_identical=True)
        write_json(output/'original_objective_equivalence.json', gate)
        print(json.dumps(dict(stage='old_source_pose_clock_objective_restored', **gate)), flush=True)

        aliases, clock, clock_meta, valid_pairs, source_meta = load_source_pose_groups(
            options.pose_sidecar, saved_phase, context['official_aliases'],
            options.pose_sidecar_sha256)
        model, initial, new_prior, binding_meta = rebind_source_camera(
            context['model'], warm, prior, aliases, clock)
        # Old factors are released only after the exact complete-objective gate.
        release_native_resources(resources)
        old_graph = old_manager = None
        context.pop('graph', None)
        manager, affine = old_entry.continuity.make_native_backend(args, model, new_prior, resources)
        data = context['data']
        graph, counts = base.build_graph(data, context['pairs'], model, new_prior,
            clock, valid_pairs, .25, native_edge_manager=manager,
            graph_owner=resources, affine_priors=affine)
        graph._adaptive_curve_worker_count = 8
        for key, value in context['counts'].items():
            if key != 'total_with_model_priors' and counts.get(key) != value:
                raise ValueError('A source observation family changed: '+key)
        references = copy.copy(context['references'])
        references.initial = new_prior
        ref_meta = old_entry.attach_reference_factors(graph, data, references, adaptive_curves=True)
        ad_meta = attach_same_ad_observations(graph, context, model, initial)
        original_count = len(graph._pending)
        q_meta = add_quadratic_priors(graph, model, initial, affine_priors=affine)
        t_meta = add_complete_motion_priors(graph, model, clock, clock_meta, valid_pairs,
                                           affine_priors=affine)
        near_meta = attach_same_near_observations(graph, context, model, initial)
        from continuous_reviewed_surface_motion import attach_reviewed_main_motion_factors
        motion_meta = attach_reviewed_main_motion_factors(graph, data, model, initial,
            context['motion_rows'], reviewed_metadata=context['motion_metadata'])
        expected_count = (original_count + 4 + t_meta['added_factor_count']
                          + len(context['near_rows']) + motion_meta['attached_factor_count'])
        if (len(graph._pending) != expected_count
                or motion_meta['attached_original_row_count'] != len(context['motion_rows'])
                or t_meta['uncovered_canonical_source_pts']):
            raise ValueError('Complete source/temporal factor coverage changed unexpectedly')
        preflight = base.preflight(context['pairs'], model, initial)
        write_json(output/'source_geometry_preflight.json', preflight)
        if preflight['invalid_pair_count']:
            raise ValueError('Rebound camera initializer leaves original source geometry domain')
        phase = dict(saved_phase,
            model='Movable C quadratic rig; source-image pose groups and corrected estimated clock',
            frame_clock_virtual_pts={str(p): clock[p] for p in sorted(clock)},
            valid_temporal_acceleration_pairs=sorted(map(list, valid_pairs)),
            exposure_clock=clock_meta, source_pose_groups=source_meta,
            source_pose_sidecar=str(options.pose_sidecar.resolve()),
            source_pose_sidecar_sha256=options.pose_sidecar_sha256,
            source_camera_rebinding=binding_meta, complete_motion_prior=t_meta,
            factor_counts=counts, reference_calibration=ref_meta, quadratic_priors=q_meta,
            development_ad_pose_rebinding=ad_meta, nearboard_pose_rebinding=near_meta,
            source_reviewed_motion_rebinding=motion_meta,
            original_complete_factor_count=original_count,
            warm_estimate=dict(provenance,
                resume_role='saved accepted state before source-pose clock change; never prior targets'),
            graph_factors_with_source_calibration=len(graph._pending),
            graph_nodes_with_references=len(initial), original_source_uv_retained=True,
            original_camera_prior_targets_changed=False, first_camera_fixed=False,
            optical_center_fixed=False, full_programme_accepted=False,
            original_accepted_objective_gate=gate,
            source_pose_entry_sha256=sha(Path(__file__)),
            board_seam_derivative_optin=seam_metadata())
        phase['code_dependencies'] = snapshot_dependencies(output)
        write_json(output/'phase_00.json', phase)
        save_states(output/'prior_initializer_states.npz', new_prior)
        save_states(output/'preflight_initial_states.npz', initial)
        export_with_pose_identity(model, initial, data, aliases,
                                  context['official_aliases'], output/'initial_cameras.npz')
        new_cost = complete_graph_error(graph, initial, native_edge_manager=manager,
            diagnostics_path=output/'new_initial_objective_diagnostics.json')
        write_json(output/'new_initial_objective.json', dict(cost=new_cost,
            source_graph_old_cost=old_cost, directly_comparable_objectives=False,
            source_uv_and_weights_retained=True, temporal_model_changed=True))
        print(json.dumps(dict(stage='complete_source_pose_clock_graph_ready',
            frames=len(model.frames), states=len(initial), factors=len(graph._pending),
            initial_cost=new_cost, previous_cost=old_cost)), flush=True)
        solve_initial = initial
        resume_solver = False
        if continuation is not None:
            old_phase, old_trace, continued, continued_provenance = continuation
            for field in ('frame_clock_virtual_pts', 'factor_counts',
                          'graph_factors_with_source_calibration',
                          'graph_nodes_with_references'):
                if old_phase[field] != phase[field]:
                    raise ValueError('Reconstructed pose-clock graph differs: '+field)
            if (set(continued) != set(initial)
                    or old_phase['source_pose_sidecar_sha256'] != options.pose_sidecar_sha256):
                raise ValueError('Pose-clock resume uses different source grouping or states')
            with np.load(options.resume_clock_source/'prior_initializer_states.npz',
                         allow_pickle=False) as z:
                previous_priors = {int(k): v for k, v in zip(z['keys'], z['states'])}
            if set(previous_priors) != set(new_prior) or any(
                    not np.array_equal(previous_priors[k], new_prior[k]) for k in new_prior):
                raise ValueError('Pose-clock resume changed original prior targets')
            cost = complete_graph_error(graph, continued, native_edge_manager=manager,
                diagnostics_path=output/'pose_clock_resume_objective_diagnostics.json')
            expected = continued_provenance['expected_cost']
            if abs(cost-expected) > max(1e-7, 1e-9*abs(expected)):
                raise ValueError('Saved pose-clock accepted objective was not restored')
            resumed = dict(**continued_provenance, restored_cost=cost,
                           absolute_difference=abs(cost-expected), priors_bitwise_equal=True)
            write_json(output/'pose_clock_resume.json', resumed)
            resumed_trace = dict(old_trace)
            resumed_trace['resume_epochs'] = [*old_trace.get('resume_epochs', []), resumed]
            write_json(output/'solver_trace.json', resumed_trace)
            save_states(output/'accepted_states.npz', continued)
            phase['pose_clock_resume'] = resumed
            write_json(output/'phase_00.json', phase)
            solve_initial, resume_solver = continued, True
        if options.preflight_only:
            return
        solved, trace = solve(graph, solve_initial, output, max_iterations=options.iterations,
                              native_edge_manager=manager, resume=resume_solver)
        report = export_with_pose_identity(model, solved, data, aliases,
            context['official_aliases'], output/'shot_cameras.npz')
        pts = np.asarray(sorted(references.model.frames), np.int64)
        np.savez_compressed(output/'reference_cameras.npz', pts=pts,
            camera25=np.asarray([references.model.camera25(int(p), solved) for p in pts]),
            source_sha256=data['diag']['source_sha256'], model_sha256=data['diag']['model_sha256'])
        report.update(camera_id='main_long_side', shot_indices=list(data['shot_indices']),
            source_sha256=data['diag']['source_sha256'], model_sha256=data['diag']['model_sha256'],
            phase_sha256=sha(output/'phase_00.json'), accepted=False,
            seconds=time.perf_counter()-started,
            scope='Complete source-image-pose-clock repaired main group; real-video review required')
        write_json(output/'summary.json', report)
        print(json.dumps(report), flush=True)
    finally:
        if resources.get('native_edge_manager') is not None:
            write_json(output/'native_source_edge_runtime.json', resources['native_edge_manager'].metadata())
        write_json(output/'board_seam_derivative_runtime.json', seam_metadata())
        context.clear()
        release_native_resources(resources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-source', type=Path, required=True)
    parser.add_argument('--resume-clock-source', type=Path,
                        help='Optional interrupted run of this exact corrected-clock graph')
    parser.add_argument('--pose-sidecar', type=Path, required=True)
    parser.add_argument('--pose-sidecar-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=80)
    parser.add_argument('--preflight-only', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
