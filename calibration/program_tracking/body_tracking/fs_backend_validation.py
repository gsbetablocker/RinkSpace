"""Real FS cam7-only preparation and independent evaluation of the shared backend.

Preparation cannot read Theia or another source camera. Evaluation opens those
assets only after a saved backend result exists. No scene plane is fabricated.
"""
from pathlib import Path
import argparse, json, os
import numpy as np
from scipy.spatial.transform import Rotation
from .rebuild_prepare import load, digest
from .fullbody_round import write_json
from .footwear_geometry import INITIAL
from .hand_state import SCHEMA, MEAN
from .fs_jump3d_validation import BODY_PAIRS, distribution, project


def prepare(fs_root, body_base, output, *, ice_policy_path):
    fs_root, body_base, output = map(Path, (fs_root, body_base, output))
    from .ice_soft import IceSoftPolicy
    policy_path=Path(ice_policy_path);policy_document=json.loads(policy_path.read_text(encoding='utf-8'))
    IceSoftPolicy(**policy_document['policy'])
    output.mkdir(parents=True, exist_ok=False)
    source = fs_root/'input_cam7'
    d = load(source/'prepared.npz'); k = load(source/'keypoints.npz')
    raw = load(source/'exmr_hand_nofoot.npz')
    joints = load(source/'all_camera_joints.npz')
    pts = d['source_pts']; n = len(pts)
    for item in (k, raw, joints):
        np.testing.assert_array_equal(item['source_pts'], pts)
    tb = d['source_time_base']; times = (pts-pts[0])*tb[0]/tb[1]
    if n != 300 or not np.array_equal(np.diff(pts), np.full(n-1, 256)):
        raise ValueError('Expected the saved complete synchronized FS trial')
    # The existing shared17 packet contains placeholders. Preserve it and use
    # the real detector output, not its zero replacement, as body evidence.
    coco = k['coco63_native_network'][:, :17].astype(float)
    uv = coco[..., :2]; score = coco[..., 2]
    use = np.isfinite(coco).all(-1) & (score > .25)
    use &= (uv[..., 0] >= 0) & (uv[..., 0] < 1024)
    use &= (uv[..., 1] >= 0) & (uv[..., 1] < 1024)
    if not use.any():
        raise ValueError('No actual cam7 body observations')
    # Only camera 7's released constants are consumed during preparation.
    cameras = load(fs_root/'cameras.npz')
    C = cameras['cam7_C_world_m']; R = d['R_w2c']
    if not np.allclose(R, R[:1], rtol=0, atol=1e-12):
        raise ValueError('FS input camera unexpectedly moves')
    root_cam = joints['hand_nofoot_joints_camera_m'][:, joints['joint_names'].tolist().index('pelvis')]
    P = np.einsum('nij,nj->ni', R.swapaxes(-1, -2), root_cam) + C
    Q = R.swapaxes(-1, -2) @ Rotation.from_rotvec(raw['global_orient']).as_matrix()
    local = Rotation.from_rotvec(raw['body_pose'].reshape(-1, 3)).as_matrix().reshape(n, 21, 3, 3)
    hands = Rotation.from_rotvec(np.stack((raw['left_hand_pose'], raw['right_hand_pose']), 1).reshape(-1, 3)).as_matrix().reshape(n, 2, 15, 3, 3)
    betas = np.median(raw['betas'], axis=0).astype(float)
    p = dict(pts=pts, time_base=tb, source_time_base=tb, fit_time_seconds=times,
        shared_betas=betas, body_scale=np.array(1.), metres_per_scene_unit=np.array(1.),
        source_uv=uv, source_score=score, source_qualified=use,
        source_bbox=d['bbox_xyxy'], lower_body_inferred=use[:, 13:17].sum(1) < 2,
        source_identity_suspect=~use.any(1), pose_axis_angle=raw['body_pose'].reshape(n, 21, 3),
        raw_root_prior_Q_world=Q, image_height=np.array(1024))
    np.savez_compressed(output/'input.npz', **p)
    np.savez_compressed(output/'initial.npz', pts=pts, P=P, Q=Q, local_R=local,
        hand_local_R=hands, skate_relative_R=np.broadcast_to(np.eye(3), (n, 2, 3, 3)),
        hand_pose_schema_version=np.array(SCHEMA), hand_mean_convention=np.array(MEAN),
        hand_pose_origin=np.array('FS own cam7 ExMR hand_nofoot'), hand_prior_sha256=np.array(digest(source/'exmr_hand_nofoot.npz')))
    np.savez_compressed(output/'observations.npz', pts=pts, uv=uv, use=use,
        sigma_px=np.maximum(2., 8.*(1-np.clip(score, 0, 1))))
    np.savez_compressed(output/'export_source.npz', pts=pts, source_pts=pts, time_base=tb,
        source_lower_body_inferred=p['lower_body_inferred'], source_identity_suspect=p['source_identity_suspect'],
        window_mask=np.zeros(n, bool), shared_geometry=INITIAL.copy(), source_observation_use=use)
    np.savez_compressed(output/'static_camera.npz', K_input=d['K_input'], R_w2c=R, C_world_m=C)
    os.link(body_base/'compact_smplx.npz', output/'compact_smplx.npz')
    model = json.loads((body_base/'trajectory_first_1106/programme/model_identity.json').read_text(encoding='utf-8'))
    write_json(output/'model_identity.json', model)
    camera_info = dict(shot='FS_Skater_A_Axel_1_cam7', state_sha256=digest(source/'prepared.npz'),
        camera_asset_sha256=digest(fs_root/'cameras.npz'), input_camera=7,
        metres_per_scene_unit=1., solver_metres_per_scene_unit=1., length_export_ratio=1.,
        source_time_base=tb.tolist(), projection='released static cam7 R/C; known rectified crop K_input; no second distortion',
        image_height=1024, camera_parameters_optimized=False,
        rotation_orthonormalization_max=float(np.max(abs(R[0]-cameras['cam7_R_w2c']))))
    write_json(output/'camera_context.json', camera_info)
    write_json(output/'ice_soft_policy.json',dict(policy=policy_document['policy'],
        origin=dict(path=str(policy_path),sha256=digest(policy_path)),
        active_ice_rows=0,meaning='explicit run configuration; scene plane unknown, no ice factor uses these values'))
    write_json(output/'source_evidence.json', dict(generic_context=True,
        source_episode_review=dict(episodes=[], unknown_reason='No independent source-contact/flight annotation for this trial'),
        scene_contract=dict(ice_plane_known=False, backend_mode='source_only',
            missing=['independently established ice plane normal and offset in released camera world', 'source-confirmed support/blade observations'],
            reference_not_used_to_invent_scene=True, scene_unit='m'),
        root_prior_policy=dict(camera_ray_depth_confidence=.04), supports=[]))
    video = fs_root.parent.parent/'datasets/fs_jump3d/original/skater_A/cam_7/Axel_1.mp4'
    write_json(output/'preparation.json', dict(trial='FS-Jump3D/Skater_A/Axel/Axel_1',
        frames=n, input_camera=7, input_video=str(video), source_time_base=tb.tolist(),
        source_prepared_sha256=digest(source/'prepared.npz'), raw_network_sha256=digest(source/'exmr_hand_nofoot.npz'),
        detector_packet_sha256=digest(source/'keypoints.npz'), used_actual_detector_field='coco63_native_network[:,:17]',
        zero_shared17_preserved=True, own_betas=betas.tolist(), body_scale=1.,
        Okada_shape_scale_attachment_or_pose_used=False, shared_generic_model_only=True,
        reference_Theia_or_other_view_pixels_consumed=False, unknown_scene_contract_retained=True))
    print(json.dumps(dict(event='FS_backend_input_prepared', frames=n, observations=int(use.sum()), output=str(output))), flush=True)
    return output


def prepare_scale_sensitivity(prepared, output):
    """Copy the predeclared cam7 input trials; never use fitted/reference state.

    Only the shared full-body scale and original root position change. All
    other prepared assets are hardlinked, as in the original run helper.
    """
    prepared, output = map(Path, (prepared, output))
    p = load(prepared/'input.npz'); initial = load(prepared/'initial.npz')
    camera = load(prepared/'static_camera.npz')
    if float(p['body_scale']) != 1. or len(p['pts']) != 300:
        raise ValueError('Expected the nominal complete 300-frame FS cam7 input')
    C = np.asarray(camera['C_world_m'], float)
    if C.shape == (300, 3):
        np.testing.assert_array_equal(C, np.repeat(C[:1], 300, axis=0))
        C = C[0]
    if C.shape != (3,):
        raise ValueError('Expected one fixed released cam7 optical centre')
    files = sorted(prepared.iterdir())
    for source in files:
        if not source.is_file():
            raise ValueError('Prepared bundle expected files only: '+str(source))
    cases = [('097', .97), ('103', 1.03)]
    for label, _ in cases:
        if (output/('fs_scale_'+label)).exists():
            raise FileExistsError(output/('fs_scale_'+label))
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for label, scale in cases:
        dest = output/('fs_scale_'+label); dest.mkdir(exist_ok=False)
        linked = []
        for source in files:
            if source.name in ('input.npz', 'initial.npz'):
                continue
            os.link(source, dest/source.name); linked.append(source.name)
        q = {k: v.copy() for k, v in p.items()}
        q['body_scale'] = np.asarray(scale, dtype=p['body_scale'].dtype)
        seed = {k: v.copy() for k, v in initial.items()}
        seed['P'] = C+scale*(seed['P']-C)
        for key in p:
            if key != 'body_scale': np.testing.assert_array_equal(q[key], p[key])
        for key in initial:
            if key != 'P': np.testing.assert_array_equal(seed[key], initial[key])
        np.savez_compressed(dest/'input.npz', **q)
        np.savez_compressed(dest/'initial.npz', **seed)
        record = dict(schema='rinkspace.FS.predeclared-scale-sensitivity.v1',
            source_prepared=str(prepared), source_input_sha256=digest(prepared/'input.npz'),
            source_initial_sha256=digest(prepared/'initial.npz'),
            source_initial='original cam7 initializer; never fitted nominal result',
            whole_trial_frames=300, scale=scale,
            scale_scope='one shared full-body scale for entire trial',
            root_transform='P_new = released_cam7_C_world_m + s * (P_original - released_cam7_C_world_m)',
            camera_center_world_m=C.tolist(), changed_input_fields=['body_scale'], changed_initial_fields=['P'],
            unchanged_initial_fields=sorted(k for k in initial if k != 'P'),
            hardlinked_original_files=sorted(linked),
            generic_shoe_binding='rebuilt by existing trajectory_backend from this scaled subject body',
            COM='recomputed by existing backend from resulting scaled body',
            Theia_or_other_cameras_used=False, nominal_unchanged=True,
            scope='predeclared development sensitivity; not calibrated confidence interval; no candidate ranking; do not transfer FS scale to Okada')
        write_json(dest/'scale_sensitivity.json', record); records.append(record)
    print(json.dumps(dict(event='FS_scale_inputs_prepared',
        cases=[dict(scale=r['scale'], frames=r['whole_trial_frames']) for r in records],
        source=str(prepared)), ensure_ascii=False), flush=True)
    return records


def evaluate(fs_root, prepared, result_path, output):
    fs_root, prepared, result_path, output = map(Path, (fs_root, prepared, result_path, output))
    candidate = load(result_path)  # Result must exist before reference opens.
    if len(candidate['pts']) != 300:
        raise ValueError('Complete trial backend result required')
    output.mkdir(parents=True, exist_ok=False)
    raw = load(fs_root/'input_cam7/all_camera_joints.npz'); cams = load(fs_root/'cameras.npz')
    truth = load(fs_root/'reference_original142.npz')
    native_names = raw['joint_names'].tolist()[:22]; jn = raw['joint_names'].tolist(); tn = truth['names'].tolist()
    ids = [native_names.index(a) for a, _ in BODY_PAIRS]; ti = [tn.index(b) for _, b in BODY_PAIRS]
    gt = truth['world_points_mm']/1000.; mask = truth['valid_mask'][:, ti]
    raw_world = raw['hand_nofoot_joints_camera_m'] @ np.linalg.inv(cams['cam7_R_w2c']).T + cams['cam7_C_world_m']
    predictions = dict(raw_cam7=raw_world[:, :22], actual_backend=candidate['native22_world_mid'])
    arrays = dict(source_pts=candidate['pts'], source_time_base=candidate['time_base'],
        reference_names=truth['names'], reference_valid_mask=truth['valid_mask'], reference_world_points_m=gt)
    results = {}
    for label, X in predictions.items():
        body = np.linalg.norm(X[:, ids]-gt[:, ti], axis=-1); body[~mask] = np.nan
        root = np.linalg.norm(X[:, 0]-gt[:, tn.index('PELVIS')], axis=-1)
        relative = np.linalg.norm((X[:, ids]-X[:, [0]])-(gt[:, ti]-gt[:, [tn.index('PELVIS')]]), axis=-1); relative[~mask] = np.nan
        views = []
        for i in range(1, 13):
            cam = {k[len(f'cam{i}_'):]: v for k, v in cams.items() if k.startswith(f'cam{i}_')}
            uv, front = project(X[:, ids], cam); target, valid = project(gt[:, ti], cam)
            visible = valid & mask & (target[..., 0] >= 0) & (target[..., 0] < 1920) & (target[..., 1] >= 0) & (target[..., 1] < 1088)
            error = np.linalg.norm(uv-target, axis=-1); error[~(visible & front)] = np.nan
            views.append(dict(camera=i, role='input' if i == 7 else 'evaluation_only', native_pixel_error=distribution(error)))
        results[label] = dict(absolute_body_m=distribution(body), pelvis_m=distribution(root),
            root_relative_body_m=distribution(relative), other_camera_checks=views)
        arrays[label+'_native22_world_m'] = X; arrays[label+'_body_error_m'] = body
        arrays[label+'_pelvis_error_m'] = root; arrays[label+'_root_relative_error_m'] = relative
    np.savez_compressed(output/'world_predictions_and_errors.npz', **arrays)
    report = dict(schema='rinkspace.FS.actual-backend-evaluation.v1', complete_frames=300,
        backend_result=str(result_path), backend_result_sha256=digest(result_path),
        backend_input_sha256=digest(prepared/'input.npz'), body_pairs=BODY_PAIRS, variants=results,
        alignment='identity; no fitted translation/rotation/scale; no per-frame Procrustes',
        selection_uses_reference_or_other_cameras=False, reference_role='Theia independent multiview proxy, not contact truth',
        backend_mode='source_only; unknown ice plane retained', accuracy_accepted=False)
    write_json(output/'metrics.json', report)
    print(json.dumps(dict(event='FS_actual_backend_evaluated', pelvis=results)), flush=True)
    return report


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('stage', choices=['prepare', 'evaluate', 'prepare-scale-sensitivity'])
    ap.add_argument('--fs-root', type=Path); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--body-base', type=Path); ap.add_argument('--prepared', type=Path); ap.add_argument('--result', type=Path)
    ap.add_argument('--ice-policy',type=Path)
    a=ap.parse_args()
    required = dict(prepare=('fs_root', 'body_base','ice_policy'), evaluate=('fs_root', 'prepared', 'result'),
        **{'prepare-scale-sensitivity': ('prepared',)})
    for name in required[a.stage]:
        if getattr(a, name) is None: ap.error('--'+name.replace('_', '-')+' is required for '+a.stage)
    if a.stage == 'prepare': prepare(a.fs_root, a.body_base, a.output,ice_policy_path=a.ice_policy)
    elif a.stage == 'evaluate': evaluate(a.fs_root, a.prepared, a.result, a.output)
    else: prepare_scale_sensitivity(a.prepared, a.output)

if __name__ == '__main__': main()
