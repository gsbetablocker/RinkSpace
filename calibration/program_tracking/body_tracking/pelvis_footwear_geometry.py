"""Small, camera-independent bare-foot geometry from saved SMPL-X poses.

This does not infer a new pose or reuse the body's historical world position.
Only the original shape, local articulation and image-conditioned root prior
are retained. Bare plantar patches are not boots, blade contacts or COCO
ankle landmarks. Their indices remain explicit for source review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reconstruct_local_poses(records, raw):
    """Retain stored same-cut source-clock brackets; no temporal gap filling."""
    from .body_prior import _slerp_rotvec
    lookup = {int(p): i for i, p in enumerate(raw['source_pts'])}
    poses, origins = [], []
    for item in records:
        left = int(item['body_local_left_source_pts'])
        right = int(item['body_local_right_source_pts'])
        li, ri = lookup[left], lookup[right]
        if not item['body_local_prior_valid'] or not np.isfinite(
                raw['body_pose'][[li, ri]]).all():
            raise ValueError('Missing original local pose is not filled')
        if raw['chosen_window'][li] < 0 or raw['chosen_window'][ri] < 0:
            raise ValueError('A local-pose bracket lacks an actual network result')
        if left == right:
            pose = raw['body_pose'][li].copy()
            origins.append('original_network_local_pose')
        else:
            if int(raw['shot_slot'][ri] - raw['shot_slot'][li]) != 1:
                raise ValueError('Local articulation cannot cross missing raw slots')
            alpha = np.asarray([item['body_local_interpolation_alpha']], np.float32)
            pose = _slerp_rotvec(raw['body_pose'][li][None].reshape(1, 21, 3),
                                raw['body_pose'][ri][None].reshape(1, 21, 3),
                                alpha).reshape(63)
            origins.append('source_clock_local_rotation_prior_interpolation')
        poses.append(pose)
    return np.asarray(poses, np.float32), np.asarray(origins)


def plantar_patch_indices(vertices, joints, weights):
    """Explicit lower-surface rear/front patches of each modeled bare foot.

    Rest model +Y is up and +Z is the toe direction. These are reproducible
    mesh patches, not a learned anatomical or shoe-surface detector.
    """
    patches, description = [], []
    for side, ankle, foot in [('left', 7, 10), ('right', 8, 11)]:
        mask = (weights[:, ankle] + weights[:, foot] > .65)
        mask &= vertices[:, 1] < joints[ankle, 1] + .005
        candidates = np.flatnonzero(mask)
        if len(candidates) < 25:
            raise ValueError('Expected actual foot surface vertices')
        z_lo, z_hi = np.quantile(vertices[candidates, 2], [.2, .8])
        for anchor, region in [('heel', candidates[vertices[candidates, 2] <= z_lo]),
                               ('toe', candidates[vertices[candidates, 2] >= z_hi])]:
            lower = np.quantile(vertices[region, 1], .2)
            selected = region[vertices[region, 1] <= lower + .003]
            if len(selected) < 3:
                raise ValueError('A plantar surface patch is too small')
            patches.append(selected)
            description.append(dict(side=side, anchor=anchor,
                vertex_indices=selected.tolist(), vertex_count=len(selected),
                rest_patch_centroid_model_m=vertices[selected].mean(axis=0).tolist(),
                dominant_reference_joint=ankle if anchor == 'heel' else foot))
    return patches, description


def rigid_boot_anchors(native_body_rootlocal_model_m, rest_native_joints_model_m,
                       rest_plantar_patches_model_m, saved_body_pose_axis_angle,
                       native_parents):
    """Carry each rest-shape heel/toe by one original ankle7/8 rigid frame.

    Saved local rotations determine the ankle frame; native foot10/11 toe
    articulation does not bend the equipment. No joint pose is optimized.
    Output anchors describe a conditional rigid in-boot foot/sole template,
    while the original deformed bare-foot surface should remain separately
    available for diagnosis.
    """
    body = np.asarray(native_body_rootlocal_model_m, float).reshape(-1, 7, 3)
    pose = np.asarray(saved_body_pose_axis_angle, float).reshape(-1, 21, 3)
    parents = np.asarray(native_parents, int).reshape(22)
    rest_joints = np.asarray(rest_native_joints_model_m, float)
    patches = np.asarray(rest_plantar_patches_model_m, float).reshape(2, 2, 3)
    if len(body) != len(pose) or not np.isfinite(body).all() or not np.isfinite(pose).all():
        raise ValueError('Original finite same-source local body pose is required')
    local = np.tile(np.eye(3), (len(pose), 22, 1, 1))
    local[:, 1:] = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(-1, 21, 3, 3)
    chain = local.copy()
    for joint in range(1, 22):
        if not 0 <= parents[joint] < joint:
            raise ValueError('Original ordered native kinematic parents required')
        chain[:, joint] = chain[:, parents[joint]] @ local[:, joint]
    frame = chain[:, [7, 8]]
    rest_relative = patches - rest_joints[[7, 8], None, :]
    anchors = body[:, 3:5, None] + np.einsum('nfij,faj->nfai', frame, rest_relative)
    up = np.repeat(frame[:, :, :, 1][:, :, None], 2, axis=2)
    return anchors, up, frame, rest_relative


def prepare(packet_path, body_prior_path, raw_path, stature_audit_path,
            upstream_runtime_path, model_path, output_directory):
    packet, audit = read_json(packet_path), read_json(stature_audit_path)
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError('Keep existing geometry and failures intact')
    # Heavy runtime imports are confined to this materialization command.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(upstream_runtime_path)))
    import torch
    from pytorch3d.transforms import axis_angle_to_matrix
    from hmr4d.utils.body_model.smplx_lite import SmplxLite, batch_rigid_transform_v2
    with np.load(body_prior_path, allow_pickle=False) as z:
        beta = z['shared_betas'].copy()
        prior_pts = z['pts'].copy()
        stored_local = z['native22_body_offset_m'].copy()
    with np.load(raw_path, allow_pickle=False) as z:
        # Never read historical world rotations or any global translation.
        raw = {name: z[name].copy() for name in
               ['source_pts', 'official_exposure_representative_pts', 'shot_slot',
                'body_pose', 'global_orient', 'chosen_window', 'K_rect']}
    if digest(body_prior_path) != audit['input_bindings'][str(Path(body_prior_path))]:
        raise ValueError('Shared-beta authority changed')
    if digest(model_path) != audit['input_bindings'][str(Path(model_path))]:
        raise ValueError('Same SMPL-X model is required')
    poses, origins = reconstruct_local_poses(packet['records'], raw)
    pts = np.asarray([r['source_observation_pts'] for r in packet['records']], np.int64)
    model = SmplxLite(model_path=Path(model_path), gender='neutral', num_betas=10)
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        rest_params = dict(body_pose=torch.zeros((1, 1, 63)),
            betas=torch.from_numpy(beta)[None, None], global_orient=torch.zeros((1, 1, 3)),
            transl=torch.zeros((1, 1, 3)))
        rest_vertices = model(**rest_params)[0, 0].numpy()
        rest_skeleton = model.get_skeleton(rest_params['betas'])[0, 0].numpy()
    patches, patch_authority = plantar_patch_indices(rest_vertices, rest_skeleton,
                                                    model.lbs_weights.numpy())
    rest_plantar_height = np.asarray([rest_vertices[p, 1].mean() - rest_vertices[:, 1].min()
                                     for p in patches]).reshape(2, 2)
    indices = [0, 1, 2, 7, 8, 10, 11]
    local, plantar, normals = [], [], []
    # Produce existing articulated surface; do not optimize any articulation.
    for start in range(0, len(pts), 12):
        body = torch.from_numpy(poses[start:start + 12])[None]
        length = body.shape[1]
        params = dict(body_pose=body,
            betas=torch.from_numpy(np.tile(beta, (length, 1)))[None],
            global_orient=torch.zeros((1, length, 3)), transl=torch.zeros((1, length, 3)))
        with torch.inference_mode():
            vertices = model(**params)[0].numpy()
            aa = torch.cat([params['global_orient'], body], dim=-1).reshape(1, length, 22, 3)
            rotations = axis_angle_to_matrix(aa)
            skeleton = model.get_skeleton(params['betas'])[..., :22, :]
            joint, transforms = batch_rigid_transform_v2(rotations, skeleton, model.parents[:22])
            joint = joint[0].numpy()
            transforms = transforms[0].numpy()
        root = joint[:, 0]
        local.extend(joint[:, indices] - root[:, None])
        plantar.extend(np.stack([vertices[:, p].mean(axis=1) - root for p in patches], axis=1))
        # Foot normals are an explicit pose-conditioned equipment attachment
        # approximation: heel uses ankle rotation, toe uses native foot rotation.
        normals.extend(transforms[:, [7, 10, 8, 11], :3, :3][..., 1])
    local = np.asarray(local, np.float32)
    plantar = np.asarray(plantar, np.float32).reshape(-1, 2, 2, 3)
    normals = np.asarray(normals, np.float32).reshape(-1, 2, 2, 3)
    rows = np.searchsorted(prior_pts, pts)
    assert np.array_equal(prior_pts[rows], pts)
    discrepancy = float(np.max(np.abs(local - stored_local[rows][:, indices])))
    if discrepancy > 2e-5:
        raise ValueError(f'Original local pose reconstruction differs by {discrepancy} model m')
    required_raw = sorted(set(int(r[key]) for r in packet['records'] for key in
                             ['body_local_left_source_pts', 'body_local_right_source_pts']))
    raw_indices = [int(np.flatnonzero(raw['source_pts'] == p)[0]) for p in required_raw]
    output.mkdir(parents=True)
    array_path = output / 'bare_plantar_body_local_geometry.npz'
    np.savez_compressed(array_path, actual_pts=pts, native_joint_indices=np.asarray(indices),
        native_body_rootlocal_model_m=local, plantar_heel_toe_body_rootlocal_model_m=plantar,
        plantar_heel_toe_local_up=normals, local_pose_origin=origins,
        rest_plantar_height_above_bare_sole_model_m=rest_plantar_height,
        raw_source_pts=raw['source_pts'][raw_indices],
        raw_exposure_representative_pts=raw['official_exposure_representative_pts'][raw_indices],
        raw_root_R_incam=Rotation.from_rotvec(raw['global_orient'][raw_indices]).as_matrix(),
        raw_global_orient_axis_angle=raw['global_orient'][raw_indices],
        raw_K_rect=raw['K_rect'][raw_indices], raw_chosen_window=raw['chosen_window'][raw_indices],
        body_local_left_source_pts=np.asarray([r['body_local_left_source_pts'] for r in packet['records']]),
        body_local_right_source_pts=np.asarray([r['body_local_right_source_pts'] for r in packet['records']]),
        body_local_interpolation_alpha=np.asarray([r['body_local_interpolation_alpha'] for r in packet['records']]),
        shared_betas=beta, body_scale_nominal=np.asarray(audit['scale_from_bio_stature']['nominal_body_scale']))
    report = dict(schema='rinkspace.camera-independent-bare-foot-hip-geometry.v1',
        actual_exposures=len(pts), source_video_sha256=packet['source_video_sha256'],
        body_model='original neutral SMPL-X shape and saved articulated local pose',
        patch_order=['left heel', 'left toe', 'right heel', 'right toe'],
        native_joint_indices=indices, patch_authority=patch_authority,
        body_scale_nominal=audit['scale_from_bio_stature']['nominal_body_scale'],
        rest_plantar_height_above_bare_sole_model_m=rest_plantar_height.tolist(),
        approximate_body_stature_m=1.53, equipment_is_not_part_of_body_stature=True,
        original_local_pose_max_discrepancy_model_m=discrepancy,
        original_network_local_pose=int(np.sum(origins == 'original_network_local_pose')),
        local_rotation_prior_interpolation=int(np.sum(origins != 'original_network_local_pose')),
        original_translation_or_world_R_read=False, new_pose_inference=False,
        articulated_joint_optimization=False, camera_binding_required=False,
        old_network_K_condition='Saved pose/root prediction used historical rectified K and crop; retained as a conditional prior only.',
        inputs={str(Path(p)): digest(p) for p in [packet_path, body_prior_path, raw_path,
                stature_audit_path, model_path]},
        output=dict(path=str(array_path), sha256=digest(array_path)),
        limitations=['Plantar patches describe a bare SMPL-X mesh, not the visible boot outsole.',
            'Heel/toe attachment normals and local articulation are model assumptions.',
            '153 cm is approximate reported body stature; equipment dimensions are separate metres.',
            'No source anatomical ankle, blade contact or unique world G is established.'])
    (output / 'geometry_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['packet', 'body-prior', 'raw', 'stature-audit', 'upstream-runtime', 'model', 'output']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    result = prepare(args.packet, args.body_prior, args.raw, args.stature_audit,
                     args.upstream_runtime, args.model, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
