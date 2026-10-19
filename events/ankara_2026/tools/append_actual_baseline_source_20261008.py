"""Source-only baseline profiles; append to the immutable current ice packet.

The manually reviewed search windows below come from original RGB.  This tool
does not load a camera, world scene, residual, or the coupled source graph.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies

add_geometry_dependencies()
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


EVENT = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
PARENT = EVENT / ('observations/vision_training/continuous_review/cut6_clear_source_constraints/'
                  'complete_source_clear_anchor_append_20261007/'
                  'ice_clear_anchors_lane23_trace_and_goal_crease_v2_20261007.json')
PARENT_SHA = 'f633c8b96dd331151d3f988f23cff764bbe51932c7ef7b9ec7e80b117e59a35c'
REVIEW = EVENT / 'evidence/user_video_review_20261008/c134_full_programme'
OUTPUT = REVIEW / 'baseline_actual_source_append_20261008'
CLOCK = EVENT / ('calibration/runs/solver_acceleration_20261003/'
                 'okada_unlocked_geometry_af8cb_board739_canonical_gate_20261008/'
                 'complete_physical_camera_model_UNACCEPTED_a90e01e5d788434d.npz')
CLOCK_SHA = '0c770d6913c8963810acef15b3882898119167b01bd5632e0450f18ec8a34f06'
CONFIG = (
    dict(feature='goal_line_right', pts=229576704,
         path=REVIEW/'04_right_corner_trapezoid_and_baseline_source_native_229576704.png',
         queries=[(50, 940), (150, 950), (350, 972), (500, 987), (700, 1009)],
         half_window=17, user_issue='04_right_corner_trapezoid_and_baseline',
         source_identity='Visible straight goal baseline crosses the corner trapezoid; glass/SPORTOTO/Fed short-side scene.'),
    dict(feature='goal_line_left', pts=230967552,
         path=EVENT/'diagnostics/goal_crease_shared_geometry_20261007/native_230967552.png',
         queries=[(300, 672), (400, 721), (500, 768), (550, 787)],
         half_window=19, user_issue='11_baseline_Ad7_relative_geometry_priority',
         source_identity='Same low-cut8 physical straight goal baseline at the flat side of the blue D, left of the athlete; ISU/OrientalBio same-side background.'),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantile(coordinates, weights, fraction):
    mass = np.cumsum(weights)
    return float(np.interp(fraction * mass[-1], mass, coordinates))


def sample_profile(gray, x, search_y, half_window):
    lo, hi = search_y-half_window, search_y+half_window
    coordinates = np.arange(lo, hi+1, dtype=float)
    raw = gray[lo:hi+1, x-3:x+4].mean(axis=1)
    smooth = gaussian_filter1d(raw, 1.5)
    background = np.linspace(raw[:3].mean(), raw[-3:].mean(), len(raw))
    weight = np.maximum(background-smooth, 0.)
    peak = int(np.argmax(weight))
    if peak < 5 or peak > len(weight)-6 or float(weight[peak]) < 1.5:
        raise ValueError('Visible source ridge is unsupported or touches search boundary')
    # Do not include another minimum or long reflection tail.  The positive
    # connected ridge containing the visually confirmed paint is the support.
    threshold = float(weight[peak])*.08
    first = peak
    while first > 0 and weight[first-1] > threshold:
        first -= 1
    last = peak
    while last+1 < len(weight) and weight[last+1] > threshold:
        last += 1
    if first == 0 or last == len(weight)-1:
        raise ValueError('Source ridge extent is open or unresolved')
    selected = weight.copy()
    selected[:first] = 0.
    selected[last+1:] = 0.
    q10, q50, q90 = [quantile(coordinates, selected, q) for q in (.1, .5, .9)]
    return [float(x), q50], dict(
        native_search_rectangle=[x-3,lo,x+3,hi], averaging_columns=7,
        original_grayscale_profile=raw.tolist(), source_grayscale_smoothing_sigma_px=1.5,
        background_endpoint_mean=[float(raw[:3].mean()),float(raw[-3:].mean())],
        source_ridge_q10_q50_q90_y=[q10,q50,q90],
        positive_ridge_mass=selected.tolist(), peak_dark_deflection_px_value=float(weight[peak]),
        connected_ridge_native_y=[int(coordinates[first]),int(coordinates[last])],
        other_profile_branches_admitted=False)


def build():
    if sha(PARENT) != PARENT_SHA:
        raise ValueError('Immutable current parent changed')
    parent = json.loads(PARENT.read_text(encoding='utf8'))
    if sha(CLOCK) != CLOCK_SHA:
        raise ValueError('Immutable exposure-alias metadata changed')
    with np.load(CLOCK,allow_pickle=False) as clock:
        # These source-time arrays are authenticated aliases. No camera25,
        # optical parameter, world geometry, or projection is read here.
        aliases = dict(zip(map(int,clock['pts']),map(int,clock['exposure_representative_pts'])))
    rows, audits = [], []
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for cfg in CONFIG:
        image = cv2.imread(str(cfg['path']))
        if image is None or image.shape != (1080,1920,3):
            raise ValueError('Original native source RGB is required')
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(float)
        samples = [sample_profile(gray,x,y,cfg['half_window']) for x,y in cfg['queries']]
        uv = np.asarray([sample[0] for sample in samples])
        slope = float(np.polyfit(uv[:,0],uv[:,1],1)[0])
        normal = np.array([-slope,1.]); normal /= np.linalg.norm(normal)
        intervals, sigmas, profiles = [], [], []
        marked = image.copy()
        for number,(point,profile) in enumerate(samples):
            q10,q50,q90 = profile['source_ridge_q10_q50_q90_y']
            width = (q90-q10)*normal[1]
            interval = [(q10-q50)*normal[1]-2.,(q90-q50)*normal[1]+2.]
            sigma = max(6.,.5*float(width))
            intervals.append(interval); sigmas.append(sigma)
            profiles.append(dict(uv=point, normal_interval_px=interval,sigma_px=sigma, **profile))
            x,y=np.rint(point).astype(int)
            cv2.drawMarker(marked,(int(x),int(y)),(0,255,255),cv2.MARKER_CROSS,15,1)
            cv2.putText(marked,str(number),(int(x)+6,int(y)-7),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
        marked_path=OUTPUT/f'{cfg["feature"]}_{cfg["pts"]}_actual_source_markers_DIAGNOSTIC.png'
        cv2.imwrite(str(marked_path),marked)
        row=dict(id=f'actual_baseline_{cfg["feature"]}_{cfg["pts"]}_source_only_20261008',
            pts=cfg['pts'],actual_pts=aliases[cfg['pts']],feature_id=cfg['feature'],
            curve=dict(kind='current_ice_segment',feature_id=cfg['feature'],segment_index=0),
            split='fit',uv=uv.tolist(),sigma_px=sigmas,
            normal=np.broadcast_to(normal,uv.shape).tolist(),normal_interval_px=intervals,
            source_optical_normal_support_authorized=True,
            geometry_role='actual_visible_straight_goal_baseline_paint_carrier',
            endpoint_observation=False,source_raw_frame=str(cfg['path']),
            source_file_sha256=sha(cfg['path']),
            source_exposure_alias_authority=dict(path=str(CLOCK),sha256=CLOCK_SHA,
                requested_native_pts=cfg['pts'],actual_exposure_pts=aliases[cfg['pts']],
                fields_read=['pts','exposure_representative_pts'],
                camera_projection_parameters_read=False),
            source_native_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            source_RGB_actual_measurement=True,
            sourceUV_origin='source_only_native_grayscale_profile_of_independently_visible_paint',
            source_support=profiles,
            sigma_semantics='max(6 native px working engineering floor, half actual source q10..q90 width); not calibrated covariance',
            normal_support_semantics='source ridge optical q10..q90 support plus 2 native px; no geometric or arrow support',
            physical_identity_authority=dict(user_issue_id=cfg['user_issue'],
                user_confirmed_feature_identity=True,human_confirmed_exact_pixels=False,
                source_visual_continuity=cfg['source_identity'],
                model_projection_used_for_identity_or_pixel_selection=False),
            no_athlete_occluded_segment_or_blue_curved_arc_admitted=True,
            development_source_after_user_review=True,independent_validation_claimed=False,
            no_world_points_or_standard_dimensions_as_observations=True)
        rows.append(row)
        audits.append(dict(id=row['id'],marker_path=str(marked_path),marker_sha256=sha(marked_path),
            native_pts=cfg['pts'],actual_exposure_pts=aliases[cfg['pts']],UV_count=len(uv),source_only_pixel_points=row['uv'],
            original_native_source_path=str(cfg['path']),original_native_source_sha256=sha(cfg['path'])))
    child=deepcopy(parent)
    child['rows']=deepcopy(parent['rows'])+rows
    child.update(status='FROZEN_ACTUAL_RGB_QUALIFIED_PARENT_PLUS_DIRECT_BASELINE_FIT',
        current_target_version='20261008_direct_goal_baselines_append_only',
        previous_immutable_packet=dict(path=str(PARENT),sha256=PARENT_SHA),
        new_direct_baseline_observation_ids=[r['id'] for r in rows],
        new_direct_baseline_scalar_rows=2*sum(len(r['uv']) for r in rows),
        previous_source_rows_UV_sigma_split_roles_support_clock_bitexact=True,
        no_existing_HELD_row_promoted=True,new_world_parameters=0,
        source_installed_in_solver=False,accuracy_accepted=False,
        source_marker_images_independently_viewed=True,
        source_material_identity_reviewed_by='source_review',
        subpixel_localization_precision_not_certified=True,
        exact_user_issue11_frame_baseline_source_not_admitted_due_to_stroke_fill_and_athlete_ambiguity=True)
    if child['rows'][:len(parent['rows'])] != parent['rows']:
        raise AssertionError('Parent source was modified')
    packet=OUTPUT/'ice_parent_plus_direct_actual_goal_baselines_v1_20261008.json'
    packet.write_text(json.dumps(child,ensure_ascii=True,indent=2)+'\n',encoding='utf8')
    receipt=dict(schema='rinkspace.actual-goal-baseline-native-source-append-audit.v1',
        parent_path=str(PARENT),parent_sha256=PARENT_SHA,packet_path=str(packet),packet_sha256=sha(packet),
        original_rows=len(parent['rows']),new_rows=len(rows),new_UV_count=sum(len(r['uv']) for r in rows),
        new_scalar_rows=2*sum(len(r['uv']) for r in rows),new_parameters=0,
        all_parent_source_fields_unchanged=True,new_rows_FIT_development_source=True,
        no_existing_HELD_or_split_change=True,no_camera_model_residual_or_world_loaded_in_source_builder=True,
        raw_source_diagnostics=audits,source_visual_review_pending=False,
        actual_original_RGB_and_source_only_markers_viewed=True,
        all_nine_points_on_visible_goal_baseline_paint=True,
        blur_precision_not_certified=True,
        current_running_target_not_modified=True,whole_programme_accuracy_accepted=False)
    receipt_path=OUTPUT/'direct_actual_goal_baselines_source_only_receipt_v1.json'
    receipt_path.write_text(json.dumps(receipt,ensure_ascii=True,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['raw_source_diagnostics']},indent=2))


if __name__=='__main__':
    build()
