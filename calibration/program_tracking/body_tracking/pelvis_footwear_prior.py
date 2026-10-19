"""Conditional pelvis depth with separate body and footwear geometry.

The source hipmean remains a ray, not a sole/contact substitute. A four-DOF
rigid root/depth diagnostic preserves original local articulation. Apparent
source skate support supplies a broad, soft clearance condition. Equipment
offsets are metres and are never scaled by the body stature factor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import hashlib

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from .camera_bridge import rays, project_world
from .pelvis_footwear_geometry import digest, read_json
from .pelvis_relative_ray_prior import relative_ray_depth


def equipment_local_edges(plantar_model_m, local_up, body_scale,
                          rest_plantar_height_model_m, effective_upright_lift_m):
    """Bare plantar patch -> approximate blade endpoints, in root-local metres.

    User effective upright lift includes outsole, heel, blade and internal
    foot posture. It is not a measured 9cm rigid distance from a bare vertex.
    Rest plantar-patch height biases are body geometry and are removed once
    to calibrate upright equipped stature exactly to body stature + lift.
    No separate heel wedge, blade height or legacy bridge is added again.
    """
    plantar = np.asarray(plantar_model_m, float).reshape(2, 2, 3)
    up = np.asarray(local_up, float).reshape(2, 2, 3)
    rest_height = np.asarray(rest_plantar_height_model_m, float).reshape(2, 2)
    lift = float(effective_upright_lift_m)
    gear = lift + body_scale * rest_height
    if (not np.isfinite(plantar).all() or not np.isfinite(up).all()
            or not np.isfinite(gear).all() or np.any(gear <= 0) or lift <= 0 or body_scale <= 0):
        raise ValueError('Finite bare-foot geometry and separate positive equipment metres required')
    if not np.allclose(np.linalg.norm(up, axis=-1), 1., atol=2e-5):
        raise ValueError('Equipment attachment normals must be unit vectors')
    return body_scale * plantar - gear[:, :, None] * up


def current_incam_root_prior(geometry, row, camera_packet):
    """Rebind raw network R_incam through the supplied physical camera R.

    Non-slot priors use source-clock SLERP of current-world root priors from
    the actual bracketing raw frames, then return to this actual camera. No
    historical world orientation or global translation is read. Optical C is
    deliberately unrelated to root orientation and remains per exposure.
    """
    raw_lookup = {int(p): i for i, p in enumerate(geometry['raw_source_pts'])}
    left = int(geometry['body_local_left_source_pts'][row])
    right = int(geometry['body_local_right_source_pts'][row])
    worlds = []
    for p in [left, right]:
        i = int(np.searchsorted(camera_packet['pts'], p))
        if i >= len(camera_packet['pts']) or int(camera_packet['pts'][i]) != p:
            raise ValueError('Actual raw source root bracket is absent from camera clock')
        r = cv2.Rodrigues(camera_packet['camera25'][i, 3:6])[0]
        worlds.append(r.T @ geometry['raw_root_R_incam'][raw_lookup[p]])
    actual = int(geometry['actual_pts'][row])
    i = int(np.searchsorted(camera_packet['pts'], actual))
    r = cv2.Rodrigues(camera_packet['camera25'][i, 3:6])[0]
    if left == right:
        q = worlds[0]
        origin = 'raw_network_incam_direct_exposure'
    else:
        alpha = float(geometry['body_local_interpolation_alpha'][row])
        q = Slerp([0., 1.], Rotation.from_matrix(np.stack(worlds)))(alpha).as_matrix()
        origin = 'current_camera_source_clock_world_root_prior_slerp_reexpressed_incam'
    return r @ q, origin


def interval_residual(value, lower, upper):
    return min(value - lower, 0.) + max(value - upper, 0.)


def solve_source_footwear(hip_uv, source_four_uv, native_body_rootlocal_model_m,
                          plantar_model_m, plantar_local_up, root_prior_R_incam,
                          camera25, metres_per_scene_unit, body_scale,
                          effective_upright_lift_m, rest_plantar_height_model_m, support,
                          source_knee_uv=None, native_knee_rootlocal_model_m=None,
                          source_knee_original_qualified=None,
                          cocoreg_lower6_rootlocal_model_m=None,
                          allow_legacy_native_image_mapping=False):
    """One actual source exposure, with no temporal or camera optimization.

    Input native indices are [0,1,2,7,8,10,11]. COCO ankle candidates are boot
    upper-region proxies; their pixel sigma is separate from hips. Support is
    source-qualified *apparent* support, never a measured blade contact label.
    """
    camera = np.asarray(camera25, float)
    scale = float(metres_per_scene_unit)
    local = np.asarray(native_body_rootlocal_model_m, float).reshape(7, 3)
    if not np.allclose(local[0],0.,atol=1e-6):
        raise ValueError('Native geometry must remain relative to model joint0')
    if cocoreg_lower6_rootlocal_model_m is not None:
        image_local_six=np.asarray(cocoreg_lower6_rootlocal_model_m,float).reshape(6,3)
        if not np.isfinite(image_local_six).all():
            raise ValueError('Finite original COCO vertex-regressed local geometry required')
        hipmean=image_local_six[:2].mean(axis=0)
        image_local_four=image_local_six[[0,1,4,5]]
        image_semantics='original_SMPLX_to_SMPL_COCO17_vertex_regressor_11_to_16'
    elif allow_legacy_native_image_mapping:
        hipmean = local[1:3].mean(axis=0)
        image_local_four=local[1:5]
        image_semantics='legacy_diagnostic_native_FK_1_2_4_5_7_8_approximate_COCO_mapping'
    else:
        raise ValueError('COCO source pixels require explicit cocoreg_lower6_rootlocal geometry; native FK image mapping is a legacy diagnostic only')
    observed = np.asarray(source_four_uv, float).reshape(4, 2)
    prior = np.asarray(root_prior_R_incam, float).reshape(3, 3)
    edges = equipment_local_edges(plantar_model_m, plantar_local_up,
                                  body_scale, rest_plantar_height_model_m, effective_upright_lift_m)
    relative_four = body_scale * (image_local_four - hipmean)
    relative_edges = edges - body_scale * hipmean
    direction, valid = rays(np.asarray(hip_uv, float)[None], camera)
    if not valid[0]:
        raise ValueError('A source hip proxy has no valid current-camera ray')
    origin = camera[:3] * scale
    r = cv2.Rodrigues(camera[3:6])[0]
    hip_sigma = float(support['hip_source_sigma_px'])
    ankle_sigma = float(support['boot_upper_ankle_source_sigma_px'])
    rot_sigma = np.deg2rad(float(support['root_prior_sigma_degrees']))
    support_sigma = float(support['support_clearance_sigma_m'])
    lo, hi = map(float, support['apparent_blade_clearance_interval_m'])
    indices = list(map(int, support['apparent_support_model_foot_indices']))
    apply_support = bool(support['apply_support_condition'])
    source_blade_regions = support.get('source_blade_regions_for_pixel_factor', [])
    blade_region_sigma = float(support.get('source_blade_region_sigma_px', 12.))
    if apply_support and (not indices or support['support_mode'] != 'one_of_visible_near_ice_feet'):
        raise ValueError('Explicit source foot correspondence and disjunctive support required')
    source_sigmas = np.array([hip_sigma, hip_sigma, ankle_sigma, ankle_sigma])[:, None]
    use_knees = source_knee_uv is not None
    if use_knees:
        observed_knees = np.asarray(source_knee_uv, float).reshape(2, 2)
        qualified_knees = np.asarray(source_knee_original_qualified, bool).reshape(2)
        image_knees=(image_local_six[2:4]if cocoreg_lower6_rootlocal_model_m is not None
                     else np.asarray(native_knee_rootlocal_model_m,float).reshape(2,3))
        relative_knees = body_scale * (image_knees - hipmean)
        knee_sigma = float(support.get('knee_source_sigma_px', hip_sigma))
        if not np.isfinite(relative_knees).all() or not np.isfinite(observed_knees[qualified_knees]).all():
            raise ValueError('Original qualified knee source and fixed local knee geometry are required')

    def geometry(x):
        incam = cv2.Rodrigues(x[1:])[0] @ prior
        q = r.T @ incam
        proxy = origin + np.exp(x[0]) * direction[0]
        points = proxy[None] + relative_four @ q.T
        blade = proxy[None, None] + relative_edges @ q.T
        pixels, good = project_world(points / scale, camera)
        return proxy, q, incam, points, blade, pixels, good

    def residual(x):
        proxy, q, _, _, blade, pixels, good = geometry(x)
        source = ((pixels - observed) / source_sigmas).ravel() if good.all() else np.full(8, 1e5)
        if use_knees:
            knee_pixels, knee_good = project_world((proxy[None] + relative_knees @ q.T) / scale, camera)
            knee_source = ((knee_pixels - observed_knees) / knee_sigma)[qualified_knees].ravel()
            source = np.r_[source, knee_source if knee_good[qualified_knees].all() else np.full(len(knee_source), 1e5)]
        penalty = []
        if apply_support:
            # A curved blade can bear on one portion; neither heel and toe nor
            # both visible feet are forced simultaneously onto ice.
            foot_min = blade[:, :, 2].min(axis=1)
            chosen = min(indices, key=lambda k: abs(interval_residual(float(foot_min[k]), lo, hi)))
            penalty.append(interval_residual(float(foot_min[chosen]), lo, hi) / support_sigma)
            # Visible equipment should not materially penetrate the modeled
            # ice under this conditional pose. This is a weak physical check.
            penalty.extend(np.minimum(foot_min + .035, 0.) / .06)
        if source_blade_regions:
            blade_uv, blade_good = project_world(blade.reshape(4, 3) / scale, camera)
            blade_uv = blade_uv.reshape(2, 2, 2)
            for region in source_blade_regions:
                k = int(region['conditional_model_foot_index'])
                x0, y0, x1, y1 = map(float, region['native_xyxy'])
                for uv in blade_uv[k]:
                    penalty.extend([interval_residual(float(uv[0]), x0, x1) / blade_region_sigma,
                                    interval_residual(float(uv[1]), y0, y1) / blade_region_sigma])
        return np.r_[source, x[1:] / rot_sigma, penalty]

    initial = relative_ray_depth(hip_uv, observed[2:], relative_four[2:], prior, camera, scale)
    depths = [float(initial['optical_depth_assumed_m'])]
    if apply_support and direction[0, 2] < -1e-6:
        fixed_edges = relative_edges @ (r.T @ prior).T
        target_h = -float(fixed_edges[indices, :, 2].min()) + .5 * (lo + hi)
        alternate = (target_h - origin[2]) / direction[0, 2]
        if .5 < alternate < 200:
            depths.append(alternate)
    fits = []
    for depth in depths:
        result = least_squares(residual, np.r_[np.log(depth), np.zeros(3)],
            bounds=(np.r_[np.log(.5), np.full(3, -np.pi/3)],
                    np.r_[np.log(200.), np.full(3, np.pi/3)]),
            max_nfev=150, xtol=1e-10, ftol=1e-10, gtol=1e-10)
        fits.append(result)
    fit = min(fits, key=lambda f: float(np.sum(f.fun**2)))
    proxy, q, incam, points, blade, pixels, good = geometry(fit.x)
    if not good.all():
        raise ValueError('Conditional fit left the valid native lens branch')
    joint0 = proxy - body_scale * q @ hipmean
    barefoot = joint0[None, None] + body_scale * np.asarray(plantar_model_m) @ q.T
    foot_reference = joint0[None] + body_scale * local[5:] @ q.T
    ground = proxy.copy(); ground[2] = 0.
    joint0_ground = joint0.copy(); joint0_ground[2] = 0.
    ground_uv, ground_valid = project_world(ground[None] / scale, camera)
    joint0_ground_uv,joint0_ground_valid=project_world(joint0_ground[None]/scale,camera)
    joint0_uv,joint0_valid=project_world(joint0[None]/scale,camera)
    foot_min = blade[:, :, 2].min(axis=1)
    selected = (min(indices, key=lambda k: abs(interval_residual(float(foot_min[k]), lo, hi)))
                if apply_support else None)
    blade_uv, _ = project_world(blade.reshape(4, 3) / scale, camera)
    blade_uv = blade_uv.reshape(2, 2, 2)
    region_diagnostics = []
    for region in source_blade_regions:
        k = int(region['conditional_model_foot_index'])
        x0, y0, x1, y1 = map(float, region['native_xyxy'])
        outside = np.array([[interval_residual(float(uv[0]), x0, x1),
                             interval_residual(float(uv[1]), y0, y1)] for uv in blade_uv[k]])
        region_diagnostics.append(dict(conditional_model_foot_index=k,
            source_native_xyxy=region['native_xyxy'], model_equipment_heel_toe_native_uv=blade_uv[k].tolist(),
            outside_region_xy_px=outside.tolist(), outside_region_distance_px=np.linalg.norm(outside, axis=1).tolist(),
            exact_blade_endpoint_identity_claim=False, source_blade_region_used_for_fit=True))
    result = dict(conditional_proxy_P_assumed_m=proxy.tolist(),
        conditional_proxy_G_assumed_m=ground.tolist(),
        estimated_model_joint0_P_assumed_m=joint0.tolist(),
        estimated_model_joint0_G_assumed_m=joint0_ground.tolist(),
        conditional_root_R_incam=incam.tolist(), conditional_Q_current_world=q.tolist(),
        root_rotation_correction_rvec_rad=fit.x[1:].tolist(),
        root_rotation_correction_angle_degrees=float(np.rad2deg(np.linalg.norm(fit.x[1:]))),
        optical_camera_Z_depth_assumed_m=float(np.exp(fit.x[0])),
        body_scale_nominal=float(body_scale), effective_upright_equipment_lift_m=float(effective_upright_lift_m),
        calibrated_heel_toe_plantar_attachment_distance_m=(float(effective_upright_lift_m)
             +body_scale*np.asarray(rest_plantar_height_model_m)).tolist(),
        source_four_residual_px=np.linalg.norm(pixels - observed, axis=1).tolist(),
        model_four_world_assumed_m=points.tolist(), model_four_projected_native_uv=pixels.tolist(),
        native_model_hip_ankle_world_assumed_m=(joint0[None]+body_scale*local[1:5]@q.T).tolist(),
        model_COCO_hipmean_offset_from_native_joint0_rootlocal_model_m=hipmean.tolist(),
        observed_model_joint_semantics=image_semantics,
        source_proxy_semantics='original COCO11/12 pixel mean ray; corresponding model COCO mean, not native FK hip mean or exact joint0',
        model_bare_plantar_world_assumed_m=barefoot.tolist(),
        model_equipment_blade_endpoints_world_assumed_m=blade.tolist(),
        native_model_foot10_11_world_assumed_m=foot_reference.tolist(),
        model_equipment_min_height_each_foot_assumed_m=foot_min.tolist(),
        model_equipment_blade_endpoints_projected_native_uv=blade_uv.tolist(),
        source_blade_region_pixel_factor=region_diagnostics,
        apparent_support_condition_applied=apply_support,
        selected_conditional_support_model_foot_index=selected,
        support_condition_distance_m=(abs(interval_residual(float(foot_min[selected]),lo,hi))
                                      if selected is not None else None),
        proxy_G_projected_native_uv=ground_uv[0].tolist(),
        estimated_model_joint0_P_projected_native_uv=joint0_uv[0].tolist(),
        estimated_model_joint0_G_projected_native_uv=joint0_ground_uv[0].tolist(),
        estimated_model_joint0_G_in_native_picture=bool(joint0_ground_valid[0]and np.all(joint0_ground_uv[0]>=0)
            and joint0_ground_uv[0,0]<1920 and joint0_ground_uv[0,1]<1080),
        proxy_G_in_native_picture=bool(ground_valid[0] and np.all(ground_uv[0] >= 0)
                                     and ground_uv[0,0] < 1920 and ground_uv[0,1] < 1080),
        optimization_success=bool(fit.success), conditional_objective=float(np.sum(fit.fun**2)),
        unique_world_known=False, blade_contact_valid=False,
        source_pixel_constraints_used_for_fit=True,
        status='153cm body plus separate footwear apparent-support working condition; unaccepted')
    if use_knees:
        knee_points = proxy[None] + relative_knees @ q.T
        knee_pixels, knee_good = project_world(knee_points / scale, camera)
        if not knee_good[qualified_knees].all():
            raise ValueError('Qualified conditional knee left the native lens branch')
        result.update(model_knee_world_assumed_m=knee_points.tolist(),
            model_knee_projected_native_uv=knee_pixels.tolist(),
            source_knee_residual_px=np.linalg.norm(knee_pixels-observed_knees, axis=1).tolist(),
            source_knee_original_qualified=qualified_knees.tolist(),
            knee_source_sigma_px=knee_sigma,source_knees_used_for_fit=True,
            model_six_world_assumed_m=np.vstack([points[:2],knee_points,points[2:]]).tolist(),
            model_six_projected_native_uv=np.vstack([pixels[:2],knee_pixels,pixels[2:]]).tolist(),
            source_six_residual_px=np.r_[np.linalg.norm(pixels[:2]-observed[:2],axis=1),
                np.linalg.norm(knee_pixels-observed_knees,axis=1),np.linalg.norm(pixels[2:]-observed[2:],axis=1)].tolist())
    return result


def run(packet_path, geometry_path, support_path, cameras_path, output_directory,
        metres_per_scene_unit=.31353729994266055,
        include_broad_body_scale_sensitivity=False, lower_body_source_path=None,
        allow_legacy_native_image_mapping=False):
    packet, support_packet = read_json(packet_path), read_json(support_path)
    with np.load(geometry_path,allow_pickle=False) as z:
        geometry={k:z[k].copy()for k in z.files}
    with np.load(cameras_path,allow_pickle=False) as z:
        camera={k:z[k].copy()for k in z.files}
    output=Path(output_directory)
    if output.exists():raise FileExistsError('Preserve all preceding candidates')
    camera_meta=json.loads(str(camera['metadata_json']))
    if camera_meta['source_sha256']!=packet['source_video_sha256']:
        raise ValueError('Current camera and source-body packet have different video')
    support_lookup={int(r['source_observation_pts']):r for r in support_packet['records']}
    geometry_report=read_json(Path(geometry_path).with_name('geometry_report.json'))
    if geometry_report['output']['sha256']!=digest(geometry_path):
        raise ValueError('Camera-independent foot geometry changed')
    if support_packet.get('user_equipment_prior_binding'):
        binding=support_packet['user_equipment_prior_binding']
        if digest(binding['path'])!=binding['sha256']:
            raise ValueError('User body/whole-equipment lift authority changed')
    if len(packet['records'])!=len(geometry['actual_pts']):
        raise ValueError('Every actual exposure needs its original geometry row')
    if support_packet['source_packet_sha256']!=digest(packet_path):
        raise ValueError('Support review belongs to a different source packet')
    settings=support_packet['working_conditions']
    lower_body = None
    if lower_body_source_path is not None:
        with np.load(lower_body_source_path,allow_pickle=False) as z:
            lower_body={k:z[k].copy()for k in z.files}
        lower_report=read_json(Path(lower_body_source_path).with_suffix('.json'))
        if lower_report['output']['sha256']!=digest(lower_body_source_path):
            raise ValueError('Original lower-body source packet changed')
        if lower_report['source_packet_sha256']!=digest(packet_path):
            raise ValueError('Six-point packet has a different original hip source')
        if not np.array_equal(lower_body['actual_pts'],geometry['actual_pts']):
            raise ValueError('Six-point source and fixed local geometry clocks differ')
    has_coco_regressor=(lower_body is not None and 'cocoreg_lower6_rootlocal_model_m'in lower_body)
    if not has_coco_regressor and not allow_legacy_native_image_mapping:
        raise ValueError('New body runs require the original COCO regressor correspondence packet; use an explicit legacy flag only to reproduce historical native-mapping diagnostics')
    body_scale=float(geometry['body_scale_nominal'])
    nominal_lift=float(settings['effective_upright_equipment_lift_nominal_m'])
    profiles=[('nominal',body_scale,nominal_lift),
        ('equipment_lift_8cm',body_scale,.08),
        ('equipment_lift_10cm',body_scale,.10)]
    if include_broad_body_scale_sensitivity:
        profiles.extend([('body_scale_85pct',body_scale*.85,nominal_lift),
                         ('body_scale_115pct',body_scale*1.15,nominal_lift)])
    results, missing=[],[]
    for row, source in enumerate(packet['records']):
        p=int(source['source_observation_pts'])
        if p!=int(geometry['actual_pts'][row]):raise ValueError('Geometry actual exposure changed')
        i=int(np.searchsorted(camera['pts'],p))
        if i>=len(camera['pts'])or int(camera['pts'][i])!=p:raise ValueError('Actual PTS absent from camera')
        c=camera['camera25'][i]
        for member in source['confirmed_native_member_pts']:
            j=int(np.searchsorted(camera['pts'],int(member)))
            if (int(camera['pts'][j])!=member or int(camera['exposure_representative_pts'][j])!=p
                    or not np.array_equal(camera['camera25'][j],c)):
                raise ValueError('Current physical camera violates exposure alias closure')
        if not source['eligible_for_conditional_root_depth_diagnostic']:
            missing.append(dict(source_observation_pts=p,reason='original_source_or_local_prior_unqualified'))
            continue
        prior,origin=current_incam_root_prior(geometry,row,camera)
        source_support=support_lookup[p]
        condition={**settings,**source_support}
        kwargs=dict(hip_uv=source['active_source_hip_mean_uv'],
            source_four_uv=source['original_source_coco_uv'],
            native_body_rootlocal_model_m=geometry['native_body_rootlocal_model_m'][row],
            plantar_model_m=geometry['plantar_heel_toe_body_rootlocal_model_m'][row],
            plantar_local_up=geometry['plantar_heel_toe_local_up'][row],
            rest_plantar_height_model_m=geometry['rest_plantar_height_above_bare_sole_model_m'],
            root_prior_R_incam=prior,camera25=c,metres_per_scene_unit=metres_per_scene_unit,support=condition)
        kwargs['allow_legacy_native_image_mapping']=allow_legacy_native_image_mapping
        if lower_body is not None:
            kwargs.update(source_knee_uv=lower_body['source_six_original_native_uv'][row,2:4],
                native_knee_rootlocal_model_m=lower_body['native_knee_rootlocal_model_m'][row],
                source_knee_original_qualified=lower_body['source_six_original_qualified'][row,2:4])
            if has_coco_regressor:
                kwargs['cocoreg_lower6_rootlocal_model_m']=lower_body['cocoreg_lower6_rootlocal_model_m'][row]
        candidates={name:solve_source_footwear(**kwargs,body_scale=s,effective_upright_lift_m=g)
                    for name,s,g in profiles}
        fitted=candidates.pop('nominal')
        fitted.update(source_observation_pts=p,
            confirmed_native_member_pts=source['confirmed_native_member_pts'],
            native_member_pts_within_tail=source['native_member_pts_within_tail'],
            camera25_sha256=hashlib.sha256(c.tobytes()).hexdigest(),
            camera_C_assumed_m=(c[:3]*metres_per_scene_unit).tolist(),
            source_evidence=source['source_evidence'],root_prior_origin=origin,
            root_prior_R_incam=prior.tolist(),source_support=source_support,
            local_pose_origin=str(geometry['local_pose_origin'][row]),
            conditional_parameter_profiles=candidates)
        results.append(fitted)
    # Materialize only the requested actual-source member clocks. Aliases
    # obtain identical geometry from the same fit; no other frame is filled.
    native_pts=np.asarray(sorted(p for r in packet['records']for p in r['native_member_pts_within_tail']),np.int64)
    actual_by_native={int(p):int(r['source_observation_pts'])for r in packet['records']for p in r['native_member_pts_within_tail']}
    fitted_lookup={r['source_observation_pts']:r for r in results}
    reps=np.asarray([actual_by_native[int(p)]for p in native_pts])
    arrays=dict(pts=native_pts,exposure_representative_pts=reps,
        source_valid=np.asarray([int(p)in fitted_lookup for p in reps]),
        unique_world_known=np.zeros(len(native_pts),bool),
        unique_world_G_assumed_m=np.full((len(native_pts),3),np.nan),
        blade_contact_valid=np.zeros(len(native_pts),bool),
        confirmed_alias=native_pts!=reps,
        actual_pts=np.asarray([r['source_observation_pts']for r in results],np.int64))
    source_lookup={int(r['source_observation_pts']):r for r in packet['records']}
    arrays['camera25']=camera['camera25'][np.searchsorted(camera['pts'],native_pts)]
    arrays['camera_C_assumed_m']=arrays['camera25'][:,:3]*metres_per_scene_unit
    arrays['source_four_original_native_uv']=np.asarray([source_lookup[int(p)]['original_source_coco_uv']for p in reps])
    arrays['source_four_original_score']=np.asarray([source_lookup[int(p)]['original_source_coco_score']for p in reps])
    arrays['source_hipmean_original_native_uv']=np.asarray([source_lookup[int(p)]['active_source_hip_mean_uv']for p in reps])
    arrays['source_hip_proxy_original_valid']=np.asarray([source_lookup[int(p)]['active_source_hip_proxy_candidate_valid']for p in reps])
    arrays['source_hip_proxy_original_status']=np.asarray([source_lookup[int(p)]['active_source_hip_proxy_status']for p in reps])
    arrays['body_scale_nominal']=np.asarray(body_scale)
    arrays['effective_upright_equipment_lift_nominal_m']=np.asarray(nominal_lift)
    names=['conditional_proxy_P_assumed_m','conditional_proxy_G_assumed_m',
        'estimated_model_joint0_P_assumed_m','estimated_model_joint0_G_assumed_m',
        'conditional_root_R_incam','conditional_Q_current_world',
        'root_rotation_correction_rvec_rad','source_four_residual_px',
        'model_four_world_assumed_m','model_bare_plantar_world_assumed_m',
        'model_equipment_blade_endpoints_world_assumed_m','native_model_foot10_11_world_assumed_m',
        'model_equipment_min_height_each_foot_assumed_m','root_prior_R_incam',
        'model_equipment_blade_endpoints_projected_native_uv']
    names.extend(['native_model_hip_ankle_world_assumed_m',
                  'model_COCO_hipmean_offset_from_native_joint0_rootlocal_model_m',
                  'estimated_model_joint0_P_projected_native_uv','estimated_model_joint0_G_projected_native_uv'])
    arrays['observed_model_joint_semantics']=np.asarray([fitted_lookup[int(p)]['observed_model_joint_semantics']
        if int(p)in fitted_lookup else 'missing'for p in reps])
    if lower_body is not None:
        lower_rows=np.searchsorted(lower_body['actual_pts'],reps)
        for name in ['source_six_original_native_uv','source_six_original_score','source_six_original_qualified',
                     'native_knee_rootlocal_model_m']:
            arrays[name]=lower_body[name][lower_rows]
        if has_coco_regressor:
            arrays['cocoreg_lower6_rootlocal_model_m']=lower_body['cocoreg_lower6_rootlocal_model_m'][lower_rows]
        names.extend(['model_knee_world_assumed_m','model_knee_projected_native_uv',
                      'source_knee_residual_px','source_six_residual_px','model_six_world_assumed_m',
                      'model_six_projected_native_uv','source_knee_original_qualified'])
    for name in names:
        shape=np.asarray(results[0][name]).shape
        arrays[name]=np.asarray([fitted_lookup[int(p)][name]if int(p)in fitted_lookup
                                else np.full(shape,np.nan)for p in reps])
    arrays['actual_proxy_P_assumed_m']=np.asarray([r['conditional_proxy_P_assumed_m']for r in results])
    arrays['actual_Q_current_world']=np.asarray([r['conditional_Q_current_world']for r in results])
    arrays['source_or_local_missing']=~arrays['source_valid']
    arrays['root_prior_origin']=np.asarray([fitted_lookup[int(p)]['root_prior_origin']if int(p)in fitted_lookup else 'missing'for p in reps])
    arrays['local_pose_origin']=np.asarray([fitted_lookup[int(p)]['local_pose_origin']if int(p)in fitted_lookup else 'missing'for p in reps])
    for profile in ['equipment_lift_8cm','equipment_lift_10cm']:
        for field in ['conditional_proxy_P_assumed_m','conditional_proxy_G_assumed_m',
                      'estimated_model_joint0_P_assumed_m','estimated_model_joint0_G_assumed_m']:
            arrays[profile+'_'+field]=np.asarray([
                fitted_lookup[int(p)]['conditional_parameter_profiles'][profile][field]
                if int(p)in fitted_lookup else np.full(3,np.nan)for p in reps])
    actual_g=np.asarray([r['conditional_proxy_G_assumed_m']for r in results])
    steps=np.linalg.norm(np.diff(actual_g[:,:2],axis=0),axis=1)
    maximum=int(np.argmax(steps))if len(steps)else 0
    stats=dict(native_frames=len(native_pts),actual_exposures=len(results),missing_actual=len(missing),
        canonical_actual_exposures=len(packet['records']),extra_native_aliases=len(native_pts)-len(results),
        confirmed_alias_native_flags=int(np.sum(native_pts!=reps)),
        direct_network_actual=sum(r['root_prior_origin']=='raw_network_incam_direct_exposure'for r in results),
        source_clock_root_prior_interpolated_actual=sum(r['root_prior_origin']!='raw_network_incam_direct_exposure'for r in results),
        all_unique_world_G_unknown=bool(np.isnan(arrays['unique_world_G_assumed_m']).all()),
        conditional_proxy_height_range_assumed_m=np.asarray([r['conditional_proxy_P_assumed_m'][2]for r in results]).min().item(),
        conditional_proxy_height_max_assumed_m=np.asarray([r['conditional_proxy_P_assumed_m'][2]for r in results]).max().item(),
        maximum_actual_G_step_assumed_m=float(steps[maximum]),
        maximum_step_source_pts=[results[maximum]['source_observation_pts'],results[maximum+1]['source_observation_pts']],
        proxy_G_outside_native_actual=sum(not r['proxy_G_in_native_picture']for r in results),
        per_point_source_residual_median_px=np.median([r['source_four_residual_px']for r in results],axis=0).tolist())
    if lower_body is not None:
        stats.update(per_point_six_source_residual_median_px=np.median([r['source_six_residual_px']for r in results],axis=0).tolist(),
                     qualified_original_source_knees=int(lower_body['source_six_original_qualified'][:,2:4].sum()))
    output.mkdir(parents=True)
    code_snapshot=output/'pelvis_footwear_prior.used.py'
    code_snapshot.write_bytes(Path(__file__).read_bytes())
    trace=output/'conditional_body_footwear_working_trace.npz'
    np.savez_compressed(trace,**arrays)
    report=dict(schema='rinkspace.pelvis-body-plus-separate-footwear-working-candidate.v1',
        source_video_sha256=packet['source_video_sha256'],source_time_base='1/15360',
        metres_per_scene_unit=metres_per_scene_unit,statistics=stats,working_conditions=settings,
        inputs={str(Path(p)):digest(p)for p in[packet_path,geometry_path,support_path,cameras_path]},
        code_sha256=digest(Path(__file__)),trace=dict(path=str(trace),sha256=digest(trace)),
        code_snapshot=dict(path=str(code_snapshot),sha256=digest(code_snapshot)),
        body_geometry_report_binding=dict(path=str(Path(geometry_path).with_name('geometry_report.json')),
            sha256=digest(Path(geometry_path).with_name('geometry_report.json'))),
        records=results,missing=missing,
        camera_C_preserved_per_exposure=True,effective_equipment_lift_scaled_by_body_factor=False,
        upright_effective_equipped_stature_nominal_m=1.53+nominal_lift,
        parameter_profiles=[dict(name=name,body_scale=s,effective_upright_lift_m=g)
                            for name,s,g in profiles],
        broad_body_scale_sensitivity_is_bio_stature_uncertainty=False,
        rigid_boot_installation_model=str(geometry.get('equipment_installation_model','unqualified nonrigid historical diagnostic')),
        camera_independent_packet_historical_camera_binding_not_used=True,
        current_world_root_prior_rebuilt_from_actual_raw_incam_R_and_current_camera_R=True,
        historical_world_body_translation_or_R_used=False,source_observations_changed=False,
        original_six_lower_body_source_binding=(dict(path=str(Path(lower_body_source_path)),
            sha256=digest(lower_body_source_path),metadata_sha256=digest(Path(lower_body_source_path).with_suffix('.json')))
            if lower_body_source_path is not None else None),
        source_knees_are_original_COCO_candidates_not_manual_bony_measurements=lower_body is not None,
        observed_model_joint_semantics=results[0]['observed_model_joint_semantics'],
        native_kinematic_arrays_replaced_by_COCO_proxy=False,
        proxy_to_native_joint0_offset_uses_COCO_vertex_regressed_hip_mean=has_coco_regressor,
        smoothing=False,temporal_fit=False,joint_articulation_optimization=False,
        unique_world_known=False,blade_contact_recovered=False,accuracy_accepted=False,
        limitations=['Apparent support is a reviewed source condition, not independently measured ice contact.',
            'COCO ankles lie on boot uppers; anatomical correspondence remains approximate with explicit pixel uncertainty.',
            'Equipment endpoints simplify outsole/blade geometry and rocker; their dimensions are pending engineering priors.',
            'Every source hip/ankle pixel used in this diagnostic is fitted, not independent validation.',
            'Parameter profiles are sensitivity conditions, not a statistical confidence interval.',
            'Changing a physical camera requires a complete source-clock rebind and solve; world trace is not transferable.'])
    report_path=output/'working_candidate_report.json'
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return dict(report=str(report_path),trace=str(trace),statistics=stats)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in['packet','geometry','support','cameras','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--metres-per-scene-unit',type=float,default=.31353729994266055)
    p.add_argument('--include-broad-body-scale-sensitivity',action='store_true',
                   help='Add legacy body-scale +/-15 percent diagnostic profiles; not a bio stature confidence interval.')
    p.add_argument('--lower-body-source',help='Original six-point source/fixed local knees packet; no new pose inference.')
    p.add_argument('--allow-legacy-native-image-mapping',action='store_true',
                   help='Explicitly reproduce historical native-FK to COCO approximation; not a current production correspondence.')
    args=p.parse_args()
    print(json.dumps(run(args.packet,args.geometry,args.support,args.cameras,args.output,
                         args.metres_per_scene_unit,args.include_broad_body_scale_sensitivity,args.lower_body_source,
                         args.allow_legacy_native_image_mapping),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
