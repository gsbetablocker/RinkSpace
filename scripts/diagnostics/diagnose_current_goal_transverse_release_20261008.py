"""Read-only local transverse-release tests on unchanged production NativeICE.

Only this process's feature view changes.  No saved q, source packet, production
module, full motion graph, J matrix, or target is changed.  All FIT rows enter
the original quadratic objective; HELD rows never enter the optimizer.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import json
import os
import sys

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import numpy as np
from scipy.optimize import least_squares
from verify_current_native_primitives_20261007 import state_camera_adapter
from rinkspace.vision_camera_initialization import current_complete_source_geometry


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TransverseReleaseView:
    """Experimental geometry view, preserving actual source values and camera."""
    def __init__(self, original, mode, unit):
        self.original, self.mode, self.unit = original, mode, float(unit)
        self.offset_m = 0.

    def features(self, world):
        from complete_ice_template_candidate.complete_ice_numeric_geometry import _boundary_hit
        source = self.original.features(world)
        if self.offset_m == 0.:
            return source
        result = dict(source)
        frame = source['goal_line_left']['shared_physical_goal_frame']
        v, inward = np.asarray(frame['across_xyz']), np.asarray(frame['inward_xyz'])
        shift = self.offset_m * v / self.unit
        crease = deepcopy(source['goal_crease_left'])
        crease['polygon_xyz'] = (np.asarray(crease['polygon_xyz']) + shift).tolist()
        result['goal_crease_left'] = crease
        if self.mode == 'B_joint_goal_crease_trapezoid_frame':
            carrier = deepcopy(source['goal_line_left'])
            anchor = np.asarray(frame['centre_xyz']) + shift
            carrier['shared_physical_goal_frame']['centre_xyz'] = anchor.tolist()
            # A transverse shift lies on the same supporting goal carrier.
            # Preserve its actual clipped segment and faceoff symmetry.
            result['goal_line_left'] = carrier
            current = self.original.current
            course, cq, _ = current.print_course_binding(world)
            boundary = course.sample_perimeter(cq, with_jacobian=False)[0]
            for sign, label in ((-1, 'lower'), (1, 'upper')):
                name = 'goal_trapezoid_left_' + label + '_outer'
                mark = deepcopy(source[name])
                start = anchor + sign * v * world[10] / (2 * self.unit)
                seed = anchor + sign * v * world[11] / (2 * self.unit)
                end = _boundary_hit(seed, -inward, boundary)
                mark['segments_xyz'] = [np.array([start, end]).tolist()]
                result[name] = mark
        return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--state', type=Path, required=True)
    ap.add_argument('--auxiliary', type=Path, required=True)
    ap.add_argument('--ice-packet', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--resume-output', action='store_true')
    args = ap.parse_args()
    arrays = dict(np.load(args.state, allow_pickle=False))
    identity = json.loads((args.state.parent / 'state_identity.json').read_text(encoding='utf8'))
    camera_path = Path(identity['camera_path'])
    assert sha(camera_path) == identity['camera_sha256']
    cameras = dict(np.load(camera_path, allow_pickle=False))
    provider, _, proof = current_complete_source_geometry(args.state, args.auxiliary)
    state, camera_error = state_camera_adapter(arrays, cameras, provider)
    q = state.evaluation_q(arrays['q']).copy()
    from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
    factor = NativeIceCurveFactor(state, args.ice_packet, len(q))
    assert factor.quadratic.all(), 'Keep the real NativeICE objective, no loss substitution'
    fixed_cameras = factor._cameras(q)
    before = factor.review(q)['rows']
    original_view = factor._ice_material_view
    nominal = factor._residual(q, factor.rows, cameras=fixed_cameras)
    world0 = q[factor.world_indices].copy()
    scene = json.loads((args.state.parent / 'scene_UNACCEPTED.json').read_text(encoding='utf8'))
    saved_features = {f['id']: f for f in scene['ice_features']}
    for row in factor.rows + factor.held:
        if row['curve']['kind'] == 'current_goal_crease':
            saved = np.asarray(saved_features[row['feature_id']]['polygon_xyz'])
            if not np.array_equal(saved[0], saved[-1]):
                saved = np.vstack((saved, saved[0]))
            np.testing.assert_array_equal(factor._world(q, row['curve']), saved)
    boundaries = np.r_[0, np.cumsum([2 * len(row['uv']) for row in factor.rows])]
    baseline_worlds = [factor._world(q, row['curve']).copy() for row in factor.rows]
    report = dict(schema='rinkspace.read-only-current-native-ice-goal-transverse-release.v1',
        state_identity=identity, state_path=str(args.state), state_container_sha256=sha(args.state),
        ice_packet=dict(path=str(args.ice_packet), sha256=sha(args.ice_packet)),
        script_sha256=sha(__file__), world_proof=proof, camera_archive_reproduction_max=camera_error,
        complete_q_and_camera_fixed_except_declared_existing_geometry_columns=True,
        existing_original_UV_PTS_sigma_quadratic_scales_roles_preserved=True,
        all_native_FIT_observation_rows=len(factor.rows), all_native_FIT_scalar_rows=len(nominal),
        HELD_observation_rows=len(factor.held), HELD_used_for_optimization=False,
        objective='Exact production NativeICE sum(0.5 * per-UV normalized vector residual squared)',
        nearest_curve_and_optical_support='Recomputed by the existing uncached production residual on every actual value call',
        baseline_cost=float(.5 * nominal @ nominal), baseline_source_rows=before,
        active_source_or_runtime_changed=False, full_motion_or_J_loaded=False,
        whole_model_accuracy_accepted=False, experiments=[])
    all_source_digest = sha(args.ice_packet)
    if args.resume_output and args.output.exists():
        previous = json.loads(args.output.read_text(encoding='utf8'))
        assert previous['state_identity'] == identity
        assert previous['ice_packet'] == report['ice_packet']
        report['previous_partial_script_sha256'] = previous['script_sha256']
        report['experiments'] = previous['experiments']
    for mode in ('A_left_crease_only', 'B_joint_goal_crease_trapezoid_frame'):
        view = TransverseReleaseView(original_view, mode, factor.unit)
        factor._ice_material_view = view
        for stage in ('offset_only', 'offset_plus_existing_shared_radius9',
                      'offset_plus_radius9_plus_existing_left_goal_distance7'):
            if any(e['mode'] == mode and e['experiment'] == stage for e in report['experiments']):
                continue
            dimensions = 1 if stage == 'offset_only' else 2 if stage.endswith('radius9') else 3
            # Only these dependencies can vary; unchanged FIT values are cached
            # bit-exactly and the final full production residual is rechecked.
            changed = []
            for i, row in enumerate(factor.rows):
                fid = row['curve'].get('feature_id', row['feature_id'])
                dependent = fid == 'goal_crease_left'
                dependent |= dimensions >= 2 and row['curve']['kind'] == 'current_goal_crease'
                dependent |= mode.startswith('B_') and fid.startswith('goal_trapezoid_left_')
                dependent |= dimensions == 3 and (fid.startswith('goal_line_left')
                    or fid.startswith('goal_trapezoid_left_') or fid in ('A3', 'A4'))
                if dependent:
                    changed.append(i)
            selected_rows = [factor.rows[i] for i in changed]
            selected_scalars = np.concatenate([np.arange(boundaries[i], boundaries[i+1]) for i in changed])
            calls = 0
            def trial(x):
                z = q.copy()
                view.offset_m = float(x[0])
                if dimensions >= 2:
                    z[factor.world_indices[9]] = world0[9] * np.exp(x[1])
                if dimensions == 3:
                    z[factor.world_indices[7]] = world0[7] * np.exp(x[2])
                provider.validate_parameters(z[factor.world_indices])
                return z
            def residual(x):
                nonlocal calls
                calls += 1
                z = trial(x)
                value = nominal.copy()
                value[selected_scalars] = factor._residual(z, selected_rows, cameras=fixed_cameras)
                return value
            print(json.dumps(dict(stage='start', mode=mode, experiment=stage,
                changed_FIT_ids=[r['id'] for r in selected_rows])), flush=True)
            result = least_squares(residual, np.zeros(dimensions), jac='3-point',
                x_scale=np.array([.3, .1, .1])[:dimensions],
                ftol=1e-11, xtol=1e-11, gtol=1e-9)
            z = trial(result.x)
            full = factor._residual(z, factor.rows, cameras=fixed_cameras)
            cache_error = float(np.max(np.abs(full - result.fun)))
            assert cache_error < 1e-12, cache_error
            unchanged = [i for i in range(len(factor.rows)) if i not in changed]
            other_world_error = max((float(np.max(abs(factor._world(z, factor.rows[i]['curve'])
                - baseline_worlds[i]))) for i in unchanged), default=0.)
            assert other_world_error < 1e-12, other_world_error
            assert all(np.array_equal(cam, fixed_cameras[pts]) for pts, cam in factor._cameras(z).items())
            after = factor.review(z)['rows']
            # All source rows still use original native UV, raw sigma and roles.
            assert sha(args.ice_packet) == all_source_digest
            entry = dict(mode=mode, experiment=stage, fitted_on_all_original_native_FIT=True,
                success=bool(result.success), termination=str(result.message), actual_value_calls=calls,
                offset_working_m=float(result.x[0]), radius9_working_m=float(z[factor.world_indices[9]]),
                radius9_delta_working_m=float(z[factor.world_indices[9]]-world0[9]),
                goal_distance7_working_m=float(z[factor.world_indices[7]]),
                goal_distance7_delta_working_m=float(z[factor.world_indices[7]]-world0[7]),
                baseline_all_FIT_cost=float(.5*nominal@nominal), after_all_FIT_cost=float(.5*full@full),
                optimality=float(result.optimality), final_full_uncached_residual_vs_cached_max=cache_error,
                unchanged_feature_world_max=other_world_error, source_rows=after,
                source_mean_changes=[dict(id=r['id'], feature_id=r['feature_id'], split=r['split'],
                    pts=r['pts'], points=r['source_uv_count'], before_mean_px=b['mean_native_px'],
                    after_mean_px=r['mean_native_px'], delta_mean_px=r['mean_native_px']-b['mean_native_px'])
                    for b, r in zip(before, after)],
                no_current_state_adoption_or_full_motion_cost_claim=True)
            report['experiments'].append(entry)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
            print(json.dumps(dict(stage='done', mode=mode, experiment=stage,
                offset_m=entry['offset_working_m'], radius_delta_m=entry['radius9_delta_working_m'],
                goal_distance_delta_m=entry['goal_distance7_delta_working_m'], cost=entry['after_all_FIT_cost'],
                goal_rows=[r for r in after if r['feature_id'].startswith('goal_')])), flush=True)
    factor._ice_material_view = original_view
    np.testing.assert_array_equal(factor._residual(q, factor.rows, cameras=fixed_cameras), nominal)
    report['original_baseline_restored_bitexact'] = True
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(output=str(args.output), sha256=sha(args.output), completed=True)), flush=True)


if __name__ == '__main__':
    main()
