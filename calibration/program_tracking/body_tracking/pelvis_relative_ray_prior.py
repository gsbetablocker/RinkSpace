"""Small source-ray depth diagnostic with an explicit relative-body prior.

The source hip midpoint and two ankle pixels constrain one optical depth.
Only root-local hip/ankle offsets and the network's image-conditioned root
rotation are used. No old world translation, ground contact or world root
orientation is reused. Body scale and orientation remain model assumptions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from .camera_bridge import project_world, rays
from .pelvis_proxy_programme import digest, read_npz, json_read, save_json


def relative_ray_depth(hip_uv, ankle_uv, ankle_offsets_model_m, rotation_incam,
                       camera25, metres_per_scene_unit):
    """Fit one conditional camera-Z depth, without a feet-on-ice equation."""
    hip_uv = np.asarray(hip_uv, float).reshape(2)
    ankle_uv = np.asarray(ankle_uv, float).reshape(2, 2)
    relative = np.asarray(ankle_offsets_model_m, float).reshape(2, 3)
    rotation = np.asarray(rotation_incam, float).reshape(3, 3)
    camera = np.asarray(camera25, float)
    scale = float(metres_per_scene_unit)
    if (not np.isfinite(relative).all() or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
            or np.linalg.det(rotation) <= 0 or not np.isfinite(scale) or scale <= 0):
        raise ValueError('Finite metric offsets and an actual proper incam rotation required')
    world_rays, valid = rays(np.vstack([hip_uv, ankle_uv]), camera)
    if not valid.all():
        raise ValueError('A source point has no valid native central-Brown ray')
    current_rotation = cv2.Rodrigues(camera[3:6])[0]
    local_rays = world_rays @ current_rotation.T
    local_rays /= local_rays[:, 2, None]
    relative_cam = relative @ rotation.T
    # In camera coordinates: ankle = lambda * hip_ray + relative_offset.
    # Eliminate the ankle depth from its two normalized image coordinates.
    a = local_rays[0, :2] - local_rays[1:, :2]
    b = relative_cam[:, :2] - local_rays[1:, :2] * relative_cam[:, 2, None]
    information = float(np.sum(a*a))
    if information <= np.finfo(float).eps:
        raise ValueError('Hip and ankle source rays cannot resolve optical depth')
    initial = float(-np.sum(a*b)/information)
    lower = max(1e-6, float(-np.min(relative_cam[:, 2])) + 1e-6)
    if initial <= lower:
        raise ValueError('Body-conditioned source depth is behind an observed ankle')
    relative_world = relative_cam @ current_rotation
    origin_m = camera[:3]*scale

    def geometry(depth):
        proxy_m = origin_m + float(depth)*world_rays[0]
        return proxy_m, proxy_m[None, :] + relative_world

    def residual(value):
        _, ankles_m = geometry(value[0])
        pixels, good = project_world(ankles_m/scale, camera)
        if not good.all():
            raise ValueError('Conditional ankle left the native optical domain')
        return (pixels-ankle_uv).ravel()

    result = least_squares(residual, [initial], bounds=([lower], [np.inf]),
                           xtol=1e-11, ftol=1e-11, gtol=1e-11)
    depth = float(result.x[0])
    proxy_m, ankles_m = geometry(depth)
    proxy_pixels, proxy_good = project_world(proxy_m[None, :]/scale, camera)
    ankle_pixels, ankle_good = project_world(ankles_m/scale, camera)
    if not proxy_good[0] or not ankle_good.all():
        raise ValueError('Conditional reconstruction did not remain on the valid lens branch')
    return dict(optical_depth_assumed_m=depth,
                proxy_world_assumed_m=proxy_m.tolist(),
                model_ankles_world_assumed_m=ankles_m.tolist(),
                model_ankles_projected_native_uv=ankle_pixels.tolist(),
                source_ankle_residual_px=np.linalg.norm(ankle_pixels-ankle_uv, axis=1).tolist(),
                source_proxy_closure_px=float(np.linalg.norm(proxy_pixels[0]-hip_uv)),
                optimization_success=bool(result.success),
                normalized_ray_depth_information=information)


def relative_ray_root_depth(hip_uv, observed_four_uv, four_offsets_model_m,
                            rotation_incam, camera25, metres_per_scene_unit,
                            source_sigma_px=8., root_sigma_degrees=15.):
    """Inspect one rigid-root correction using hips/ankles and a rotation prior.

    The four native model offsets are fixed; source COCO candidates are an
    anatomical approximation. The fitted root rotation is not source truth.
    The source hip mean fixes the proxy ray. There is no feet/ice equation,
    ground-mark fit, temporal term, or articulated-body optimization.
    """
    pixel=np.asarray(hip_uv,float).reshape(2)
    observed=np.asarray(observed_four_uv,float).reshape(4,2)
    offsets=np.asarray(four_offsets_model_m,float).reshape(4,3)
    prior=np.asarray(rotation_incam,float).reshape(3,3)
    camera=np.asarray(camera25,float)
    scale=float(metres_per_scene_unit)
    if not np.isfinite(observed).all() or not np.isfinite(offsets).all():
        raise ValueError('Original finite hips/ankles are required')
    if not np.allclose(prior.T@prior,np.eye(3),atol=1e-7) or np.linalg.det(prior)<=0:
        raise ValueError('A proper image-conditioned network root prior is required')
    direction,good=rays(pixel[None],camera)
    if not good[0]:raise ValueError('Source hip mean has no native Brown ray')
    r_w2c=cv2.Rodrigues(camera[3:6])[0]
    origin=camera[:3]*scale
    initial=relative_ray_depth(pixel,observed[2:],offsets[2:],prior,camera,scale)
    sigma_rotation=np.deg2rad(float(root_sigma_degrees))
    if source_sigma_px<=0 or sigma_rotation<=0:raise ValueError('Explicit positive diagnostic uncertainties required')

    def geometry(q):
        rotation=cv2.Rodrigues(q[1:])[0]@prior
        proxy=origin+np.exp(q[0])*direction[0]
        points=proxy[None]+offsets@rotation.T@r_w2c
        projected,valid=project_world(points/scale,camera)
        return proxy,points,projected,valid,rotation

    def residual(q):
        _,_,projected,valid,_=geometry(q)
        if not valid.all():return np.r_[np.full(8,1e6),q[1:]/sigma_rotation]
        return np.r_[(projected-observed).ravel()/source_sigma_px,q[1:]/sigma_rotation]

    q0=np.r_[np.log(initial['optical_depth_assumed_m']),np.zeros(3)]
    # These broad diagnostic bounds keep one positive camera-depth branch;
    # they are not a measured anatomical orientation or depth interval.
    fit=least_squares(residual,q0,bounds=(np.r_[np.log(.5),np.full(3,-np.pi/3)],
                                        np.r_[np.log(200.),np.full(3,np.pi/3)]),
                      xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=150)
    proxy,points,projected,valid,rotation=geometry(fit.x)
    if not valid.all():raise ValueError('Conditional rigid-root diagnostic left native lens branch')
    data_jacobian=fit.jac[:8]
    singular=np.linalg.svd(data_jacobian,compute_uv=False)
    return dict(optical_depth_assumed_m=float(np.exp(fit.x[0])),
                proxy_world_assumed_m=proxy.tolist(),
                model_four_world_assumed_m=points.tolist(),
                model_four_projected_native_uv=projected.tolist(),
                source_four_residual_px=np.linalg.norm(projected-observed,axis=1).tolist(),
                source_ankle_residual_px=np.linalg.norm(projected[2:]-observed[2:],axis=1).tolist(),
                original_root_fixed_source_ankle_residual_px=initial['source_ankle_residual_px'],
                root_rotation_correction_rvec_rad=fit.x[1:].tolist(),
                root_rotation_correction_angle_degrees=float(np.rad2deg(np.linalg.norm(fit.x[1:]))),
                conditional_root_R_incam=rotation.tolist(),
                source_sigma_px=float(source_sigma_px),root_prior_sigma_degrees=float(root_sigma_degrees),
                data_only_jacobian_singular_values=singular.tolist(),
                source_cost=float(.5*np.sum(fit.fun[:8]**2)),
                rotation_prior_cost=float(.5*np.sum(fit.fun[8:]**2)),
                optimization_success=bool(fit.success),
                status='conditional limited rigid-root/depth source diagnostic; not accepted world position')


def run(packet_path, cameras_path, metric_scale_path, output_path, *, allow_root_correction=False,
        body_scale=1.):
    packet, cameras = json_read(packet_path), read_npz(cameras_path)
    from ..z190_physical_camera import PhysicalCameraModel
    PhysicalCameraModel.load_npz(cameras_path)
    if packet['schema'] not in {
            'rinkspace.tail-minimal-body-relative-and-original-source-ray-packet.v1',
            'rinkspace.tail-minimal-body-relative-and-original-source-ray-packet.v2'}:
        raise ValueError('Typed relative-body and original-source packet required')
    if not np.isfinite(body_scale) or body_scale<=0:
        raise ValueError('A positive explicitly assumed body scale is required')
    from rinkspace.metric_scale import AssumedMetricScale
    scale = AssumedMetricScale.load(metric_scale_path).metres_per_scene_unit
    if packet['source_video_sha256'] != json.loads(str(cameras['metadata_json']))['source_sha256']:
        raise ValueError('Body source rays and current Z190 camera belong to different video')
    for path, expected in packet['input_bindings'].items():
        if digest(Path(path)) != expected:
            raise ValueError('Relative-body/source-pixel input changed: '+path)
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError('Keep the existing source diagnostic intact')
    records = []
    missing=[]
    checked_source_files={}
    source_crop_caches={}
    for original in packet['records']:
        if (original['original_source_coco_indices'] != [11, 12, 15, 16]
                or original['native_model_joint_indices'] != [0, 1, 2, 7, 8]):
            raise ValueError('Body hips/ankles and source COCO point identities changed')
        pts = int(original['source_observation_pts'])
        i = int(np.searchsorted(cameras['pts'], pts))
        if i >= len(cameras['pts']) or cameras['pts'][i] != pts:
            raise ValueError('Exact actual source exposure is absent from the current camera')
        camera = cameras['camera25'][i]
        is_v2=packet['schema'].endswith('.v2')
        if not original['active_source_hip_proxy_candidate_valid']:
            if not is_v2:raise ValueError('Missing source hip proxy is not filled with a body model')
            missing.append(dict(source_observation_pts=pts,reason='source_hip_proxy_unknown'))
            continue
        qualifications = np.asarray(original['diagnostic_score_gate_0_35_qualification'], bool)
        in_source = np.asarray(original['original_source_keypoint_in_source'], bool)
        if not qualifications.all() or not in_source.all():
            if not is_v2:raise ValueError('Clipped or unqualified source ankles are not body-depth observations')
            missing.append(dict(source_observation_pts=pts,reason='original_four_source_points_clipped_or_unqualified'))
            continue
        if is_v2:
            if not original['eligible_for_conditional_root_depth_diagnostic']:
                missing.append(dict(source_observation_pts=pts,reason='typed_source_or_body_prior_unqualified'))
                continue
            if original['root_prior_origin'] not in {
                    'raw_network_incam_direct_exposure',
                    'interpolated_current_world_root_prior_reexpressed_incam'}:
                raise ValueError('Explicit direct or interpolated root prior is required')
            root_prior=np.asarray(original['root_prior_R_incam'])
        else:
            root_prior=np.asarray(original['raw_network_root_R_incam'])
        source = original['source_evidence']
        if source['path'] not in checked_source_files:
            checked_source_files[source['path']]=digest(Path(source['path']))
        if checked_source_files[source['path']] != source['sha256']:
            raise ValueError('Actual source image evidence changed')
        if source.get('kind')=='exact_native_body_crop_cache_row':
            if source['path'] not in source_crop_caches:
                source_crop_caches[source['path']]=read_npz(source['path'])
            crop_packet=source_crop_caches[source['path']]
            row=int(source['row_index'])
            crop=crop_packet[source['array_key']][row]
            boxes=np.asarray(crop_packet['native_crop_xyxy'])
            if boxes.shape==(4,):crop_box=boxes
            elif boxes.shape==(len(crop_packet['pts']),4):crop_box=boxes[row]
            else:raise ValueError('Source crop box must be fixed or explicitly indexed per row')
            if (int(crop_packet['pts'][row])!=source['source_native_pts']
                    or not np.array_equal(crop_box,source['native_crop_xyxy'])
                    or hashlib.sha256(crop.tobytes()).hexdigest()!=source['source_crop_bgr_sha256']):
                raise ValueError('Exact source body crop row changed')
        if is_v2 and hashlib.sha256(camera.tobytes()).hexdigest()!=original['current_camera25_sha256']:
            raise ValueError('Relative root packet belongs to another current camera')
        for member in original['confirmed_native_member_pts']:
            j = int(np.searchsorted(cameras['pts'], int(member)))
            if (cameras['pts'][j] != member
                    or cameras['exposure_representative_pts'][j] != pts
                    or not np.array_equal(cameras['camera25'][j], camera)):
                raise ValueError('Body diagnostic changes actual-exposure aliases')
        if allow_root_correction:
            local=np.asarray(original['native_model_body_root_offsets_model_m'])
            mean=np.asarray(original['native_model_hip_mean_offset_from_joint0_model_m'])
            fitted=relative_ray_root_depth(original['active_source_hip_mean_uv'],
                original['original_source_coco_uv'],body_scale*(local[1:]-mean),
                root_prior,camera,scale)
            root=np.asarray(fitted['conditional_root_R_incam'])
        else:
            fitted = relative_ray_depth(
                original['active_source_hip_mean_uv'],
                np.asarray(original['original_source_coco_uv'])[2:],
                body_scale*np.asarray(original['native_model_ankles_offset_from_native_hip_mean_model_m']),
                root_prior, camera, scale)
            root=root_prior
        rotation = cv2.Rodrigues(camera[3:6])[0]
        body_current_world = rotation.T @ root
        delta_joint0_m = body_scale*body_current_world @ -np.asarray(
            original['native_model_hip_mean_offset_from_joint0_model_m'])
        proxy = np.asarray(fitted['proxy_world_assumed_m'])
        joint0 = proxy + delta_joint0_m
        scale_endpoints = []
        origin_m = camera[:3]*scale
        for relative_body_scale in (.85, 1.15):
            # Scaling the body and its fitted depth together preserves pixels.
            proxy_endpoint = origin_m + relative_body_scale*(proxy-origin_m)
            joint0_endpoint = proxy_endpoint + relative_body_scale*delta_joint0_m
            scale_endpoints.append(dict(body_scale=body_scale*relative_body_scale,
                relative_scale_multiplier=relative_body_scale,
                hip_proxy_P_assumed_m=proxy_endpoint.tolist(),
                model_joint0_P_assumed_m=joint0_endpoint.tolist()))
        records.append(dict(source_observation_pts=pts, source_role=original['role'],
            root_prior_origin=original.get('root_prior_origin','raw_network_incam_direct_exposure'),
            role_for_this_method=('source_hip_and_ankle_pixels_used_in_conditional_rigid_root_depth_fit'
                                 if allow_root_correction else 'source_ankle_pixels_used_in_conditional_depth_fit'),
            independent_validation_for_this_method=False,
            confirmed_native_member_pts=original['confirmed_native_member_pts'],
            current_camera25_sha256=hashlib.sha256(camera.tobytes()).hexdigest(),
            **fitted, model_joint0_world_assumed_m=joint0.tolist(),
            hip_proxy_G_assumed_m=[float(proxy[0]), float(proxy[1]), 0.],
            model_joint0_G_assumed_m=[float(joint0[0]), float(joint0[1]), 0.],
            body_scale_condition_endpoints=scale_endpoints,
            world_position_independently_known=False, blade_contact_observed=False))
    report = dict(schema=('rinkspace.relative-body-four-source-ray-rigid-root-depth-diagnostic.v1'
                         if allow_root_correction else 'rinkspace.relative-body-three-source-ray-depth-diagnostic.v1'),
        accuracy_accepted=False, unique_world_position_known=False,
        pelvis_semantics='COCO hip mean proxy; separately model-derived SMPL-X joint0',
        assumptions=dict(nominal_body_model_scale=float(body_scale),
            body_scale_range=[.85*body_scale,1.15*body_scale],
            range_status='engineering sensitivity, not a measured uncertainty interval',
            root_orientation=('limited source-fitted rigid-root correction around declared image-conditioned prior; old K/crop conditions remain'
                              if allow_root_correction else 'declared image-conditioned incam prior; old K/crop conditions remain'),
            source_and_rotation_sigma_authority='assistant diagnostic engineering assumptions, not measured anatomical uncertainty',
            ankle_semantics='original COCO ankle candidates; not independent measured anatomy or blades'),
        feet_on_ice_equation_used=False, old_world_rotation_or_translation_used=False,
        allow_limited_rigid_root_correction=bool(allow_root_correction),
        articulated_body_pose_optimized=False,user_rough_ground_mark_used_in_fit=False,
        original_source_joint_semantics='COCO hip/ankle candidates approximate native model hip/ankle joints',
        source_ankles_used_to_fit_each_depth=True,
        original_range_held_role_not_a_holdout_for_this_different_method=True,
        current_camera_binding=dict(path=str(Path(cameras_path).resolve()),sha256=digest(cameras_path)),
        source_packet=dict(path=str(Path(packet_path).resolve()),sha256=digest(packet_path)),
        metric_scale_binding=dict(path=str(Path(metric_scale_path).resolve()),sha256=digest(metric_scale_path)),
        records=records,missing_records=missing)
    save_json(output_path, report)
    print(json.dumps([dict(pts=r['source_observation_pts'],
        proxy_h_m=r['proxy_world_assumed_m'][2], joint0_h_m=r['model_joint0_world_assumed_m'][2],
        ankle_residual_px=r['source_ankle_residual_px']) for r in records]))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source-packet', 'cameras', 'metric-scale', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--allow-root-correction',action='store_true')
    parser.add_argument('--body-scale',type=float,default=1.)
    args = parser.parse_args()
    run(args.source_packet, args.cameras, args.metric_scale, args.output,
        allow_root_correction=args.allow_root_correction,body_scale=args.body_scale)
