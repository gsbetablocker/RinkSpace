"""Replay admitted original points against actual complete-run checkpoints.

Reads saved full-source masks, profiles and immutable cameras. No programme
graph, pickle, new observations, full Jacobian or optimizer is constructed.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RINKSPACE_MOTION_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
sys.dont_write_bytecode = True
import cv2
import numpy as np
from calibration.solver_runtime import install
install()
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from structured_programme_source_candidate.original_motion_far_board_roles import OriginalMotionFarBoardRoleBinding, PAIR
from z190_motion_native import native_motion
from z190_streaming_source_normal import _block_cost


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values, float)
    return dict(median=float(np.median(a)), mean=float(np.mean(a)), p95=float(np.percentile(a, 95)), max=float(a.max()))


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--run', required=True, type=Path)
    cli.add_argument('--state', required=True, action='append', type=Path)
    cli.add_argument('--auxiliary', required=True, type=Path)
    cli.add_argument('--output', required=True, type=Path)
    args = cli.parse_args()
    manifest = read(args.run / 'input_manifest.json')
    certificate = manifest['original_motion_far_board_role_correction']
    role = OriginalMotionFarBoardRoleBinding(certificate['packet_path'])
    assert role.binding_id == certificate['packet_sha256']
    starts = [read(args.run / 'actual_source_blocks' / f'{name}_first_actual_normal.json')['actual_rows']
              for name in ('main1', 'main7')]
    source_start = sum(starts)
    scalar_slots = source_start + np.column_stack((2 * role.indices, 2 * role.indices + 1)).ravel()
    sigma = np.asarray([r['source_sigma_px'] for r in role.records], float)
    uv_a = np.asarray([r['source_uv'] for r in role.records], float)
    uv_b = np.asarray([r['target_uv'] for r in role.records], float)
    retired = manifest['static_motion_source_role_retirement']
    assert retired['retired_motion_point_count'] == 89
    assert certificate['original_profile_buffers_not_reweighted'] is True
    states = []
    for directory in args.state:
        identity = read(directory / 'state_identity.json')
        state_path = directory / 'complete_state_UNACCEPTED.npz'
        assert sha(state_path) == identity['state_npz_sha256']
        with np.load(state_path, allow_pickle=False) as archive:
            q_sha = hashlib.sha256(archive['q'].tobytes()).hexdigest()
            assert q_sha == identity['q_sha256']
            names = archive['aux__block_names'].tolist()
            assert names[2] == 'secondary'
            weights = archive['profile__weights_2'][role.indices].copy()
            mask = archive['quadratic'][scalar_slots]
            scales = archive['scales'][scalar_slots]
            assert not np.any(mask) and np.array_equal(scales, sigma.ravel())
        geometry, world, proof = current_complete_source_geometry(state_path, args.auxiliary)
        assert sha(identity['camera_path']) == identity['camera_sha256']
        with np.load(identity['camera_path'], allow_pickle=False) as archive:
            points = archive['exposure_representative_pts']
            lookup = {int(p): i for i, p in enumerate(points)}
            ca, cb = [archive['camera25'][lookup[p]].copy() for p in PAIR]
        role.source_cameras = {PAIR[0]: ca}
        bound = role.current_binding(geometry, world)
        inputs = dict(a=uv_a, b=uv_b, fa=np.exp(ca[6]), fb=np.exp(cb[6]),
            ra=cv2.Rodrigues(ca[3:6])[0], rb=cv2.Rodrigues(cb[3:6])[0], ca=ca[:3], cb=cb[:3],
            pp=ca[8:10], k1=ca[11], sigma=sigma, profile_weight=weights, jacobian=False,
            da=None, db=None, dca_shared=None, dcb_shared=None, dca_view=None, dcb_view=None)
        pixels, checks, trace = native_motion(**inputs, planes=bound['planes'],
            minimum_source_depth_scene_units=1. / geometry.scene_unit_m, schur=True, return_trace=True)
        normalized = pixels / sigma.ravel()
        expected_profiles = np.minimum(1., 2. / np.maximum(np.abs(normalized), 1e-12)).reshape(-1, 2)
        profile_error = float(np.max(abs(expected_profiles - weights)))
        if profile_error > 1e-8:
            raise ValueError(('Actual full-source profile update does not match admitted operator', q_sha, profile_error))
        untyped, _ = native_motion(**inputs, planes=np.zeros_like(bound['planes']),
            minimum_source_depth_scene_units=1. / geometry.scene_unit_m, schur=True)
        states.append(dict(q_sha256=q_sha, state_directory=str(directory), camera_sha256=identity['camera_sha256'],
            actual_complete_source_profile_weights=weights.tolist(), typed_native_residual_pixels=pixels.reshape(-1, 2).tolist(),
            typed_normalized_residual=normalized.reshape(-1, 2).tolist(), typed_pixels_L2=stats(np.linalg.norm(pixels.reshape(-1, 2), axis=1)),
            actual_30_scalar_Huber2_cost=_block_cost(normalized, mask),
            actual_saved_quadratic_false=bool(not np.any(mask)), actual_saved_native_scales=scales.tolist(),
            typed_operator_profile_update_matches_actual_full_checkpoint_max_error=profile_error,
            same_q_untyped_counterfactual_pixels_L2=stats(np.linalg.norm(untyped.reshape(-1, 2), axis=1)),
            all15_actual_native_trace_known=bool(np.all(trace[:, 12])), all15_unknown_depth_profile_absent=bool(not np.any(trace[:, 9])),
            current_course_facets=bound['selected_course_facets'].tolist(), native_checks=checks))
    report = dict(schema='rinkspace.actual-complete-run-original15-profile-and-motion-replay.v1',
        run=str(args.run), input_manifest_sha256=sha(args.run / 'input_manifest.json'), role_certificate=certificate,
        actual_programme_family='programme:secondary', actual_family_start=source_start,
        actual_complete_source_scalar_slots=scalar_slots.tolist(), original_global_point_indices=role.indices.tolist(),
        source_sigma_pixels=sigma.tolist(), source_profiles_are_actual_complete_checkpoint_arrays=True,
        profiles_control_unknown_depth_estimation_not_extra_cost_weight=True,
        original_motion_30_slots_not_affected_by_low_broadcast_or_Ad7_role_retirement=True,
        numerical_replay_is_not_a_captured_full_J_row_dump=True, full_graph_or_global_J_constructed=False,
        new_source_rows=0, weights_sigma_support_roles_modified=False, accuracy_accepted=False, states=states)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps(dict(output=str(args.output), sha256=sha(args.output), states=[dict(q=s['q_sha256'],
        pixels=s['typed_pixels_L2'], actual_cost=s['actual_30_scalar_Huber2_cost'],
        full_profile_error=s['typed_operator_profile_update_matches_actual_full_checkpoint_max_error']) for s in states])))


if __name__ == '__main__':
    main()
