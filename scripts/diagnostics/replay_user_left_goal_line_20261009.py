"""Replay a user's yellow source-image line against an immutable solved scene.

Image registration only recovers screenshot resizing/cropping. It never uses
the model to select observed source coordinates or changes the solved model.
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'calibration/program_tracking')]
import render_complete_problem_cuts_grid_only as grid
from projection import project_central_valid as project_central_valid_full_brown

cv2, np = grid.cv2, grid.np
cv2.setNumThreads(1)
BASE = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
OUT = BASE / 'evidence/user_video_review_20261009/a13_complete_programme/left_end_line_04_10_38_800'
STATE = BASE / 'calibration/runs/solver_acceleration_20261003/okada_full9_zero_support_cache_20261009_065854_736/accepted_states/a13c816fb886166cb625a216dbe84ccaed1d612a785a337203be488bb8ad7c4c'
PTS = 230995968


def main():
    screenshot = OUT / 'user_left_end_line_yellow_04_10_38_800.png'
    user = cv2.imread(str(screenshot))
    original = next(grid.decode_selected(grid.DEFAULT_SOURCE, [PTS]))[1]
    cv2.imwrite(str(OUT / f'original_native_{PTS}.png'), original)
    # Only original camera pixels participate; right plan, header and footer do not.
    mask = np.zeros(user.shape[:2], np.uint8)
    mask[55:890, :1485] = 255
    sift = cv2.SIFT_create(nfeatures=5000)
    ka, da = sift.detectAndCompute(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(user, cv2.COLOR_BGR2GRAY), mask)
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [a for a, b in pairs if a.distance < .7 * b.distance]
    a = np.float64([ka[m.queryIdx].pt for m in good])
    b = np.float64([kb[m.trainIdx].pt for m in good])
    affine, kept = cv2.estimateAffine2D(a, b, method=cv2.RANSAC,
                                      ransacReprojThreshold=1.5, maxIters=3000,
                                      confidence=.999, refineIters=20)
    if affine is None or int(kept.sum()) < 50:
        raise ValueError('Screenshot/native registration unsupported by enough actual image pixels')
    registration_error = np.linalg.norm(a @ affine[:, :2].T + affine[:, 2] - b, axis=1)[kept.ravel().astype(bool)]
    inverse = cv2.invertAffineTransform(affine)
    yellow = ((user[:, :, 0] < 90) & (user[:, :, 1] > 210) & (user[:, :, 2] > 210)).astype(np.uint8)
    yellow[:550] = 0
    yellow[:, :100] = 0
    yellow[:, 1000:] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(yellow, 8)
    candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_WIDTH] > 500]
    if len(candidates) != 1:
        raise ValueError('Expected exactly the user long diagonal yellow component')
    ys, xs = np.nonzero(labels == candidates[0])
    columns = np.unique(xs)
    full_screenshot = np.array([[x, np.median(ys[xs == x])] for x in columns])
    full_native = full_screenshot @ inverse[:, :2].T + inverse[:, 2]
    inside = (full_native[:, 0] >= 0) & (full_native[:, 0] < 1920) & (full_native[:, 1] >= 0) & (full_native[:, 1] < 1080)
    excluded = int(np.count_nonzero(~inside))
    full_screenshot, full_native, columns = full_screenshot[inside], full_native[inside], columns[inside]
    _, _, vectors = np.linalg.svd(full_native - full_native.mean(axis=0), full_matrices=False)
    tangent = vectors[0]
    if tangent[0] < 0: tangent = -tangent
    normal = np.array([tangent[1], -tangent[0]])
    full_normals = np.tile(normal, (len(full_native), 1))
    annotation_intervals = []
    # Pixel-cell corners bound the drawn raster, not physical paint thickness.
    for x, center in zip(columns, full_native):
        yy = ys[xs == x]
        corners = np.array([[x + dx, y + dy] for y in yy for dx in (-.5, .5) for dy in (-.5, .5)])
        native_corners = corners @ inverse[:, :2].T + inverse[:, 2]
        offsets = (native_corners - center) @ normal
        annotation_intervals.append([float(offsets.min()), float(offsets.max())])
    annotation_intervals = np.asarray(annotation_intervals)
    registration_native = float(np.percentile(registration_error, 95) / np.linalg.svd(affine[:, :2], compute_uv=False).min())
    intervals = annotation_intervals + np.array([-registration_native, registration_native])
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(full_native, axis=0), axis=1))]
    # Reuse the existing native source optical-window spacing convention.
    selected = np.unique(np.r_[0, np.searchsorted(arc, np.arange(31., arc[-1], 31.)), len(arc)-1]).astype(int)
    screenshot_uv, native_uv = full_screenshot[selected], full_native[selected]
    arrays_path = OUT / 'registered_user_yellow_left_goal_line_full_trace_v2.npz'
    np.savez_compressed(arrays_path, uv=full_native, normal=full_normals,
                        normal_interval_px=intervals, annotation_stroke_normal_interval_px=annotation_intervals,
                        screenshot_xy=full_screenshot, source_arc_length_px=arc, selected_trace_indices=selected)
    registration = dict(schema='rinkspace.registered-user-source-stroke.v2',
                        native_pts=PTS, source_clock='04:10:38.800', feature_id='goal_line_left', segment_index=0,
                        source_raw_frame=str(OUT / f'original_native_{PTS}.png'),
                        source_file_sha256=hashlib.sha256((OUT / f'original_native_{PTS}.png').read_bytes()).hexdigest(),
                        user_statement='这个画面附近左底线还是没有完全修好，应该在我画黄线的位置',
                        screenshot={'path':str(screenshot),'sha256':hashlib.sha256(screenshot.read_bytes()).hexdigest()},
                        screenshot_from_native_affine=affine.tolist(), registration_actual_image_inliers=int(kept.sum()),
                        registration_uncertainty_native_px=registration_native,
                        registration_uncertainty_kind='actual_image_registration_inlier_p95_converted_by_minimum_affine_scale',
                        source_trace_arrays={'path':str(arrays_path),'sha256':hashlib.sha256(arrays_path.read_bytes()).hexdigest()},
                        full_trace_vertex_count=len(full_native), excluded_footer_or_outside_native_vertices=excluded,
                        native_source_uv=native_uv.tolist(), normal=full_normals[selected].tolist(),
                        normal_interval_px=intervals[selected].tolist(),
                        annotation_stroke_normal_interval_px=annotation_intervals[selected].tolist(),
                        selected_trace_indices=selected.tolist(), source_query_spacing_native_px=31.,
                        source_query_spacing_is_existing_optical_window_convention=True,
                        actual_yellow_pixels_without_model_projection_selected_the_observation=True,
                        dense_trace_is_guard_only_not_independent_FIT_rows=True,
                        annotation_stroke_width_is_not_physical_paint_or_optical_blur_width=True,
                        point_correspondence=False, endpoint_observation=False,
                        propagated_to_other_exposures=False, independent_holdout=False,
                        source_packet_or_solver_not_modified=True)
    (OUT / 'registered_user_yellow_left_goal_line_source_trace_v2.json').write_text(json.dumps(registration, ensure_ascii=False, indent=2), encoding='utf-8')
    identity = json.loads((STATE / 'state_identity.json').read_text(encoding='utf-8'))
    scene = json.loads((STATE / 'scene_UNACCEPTED.json').read_text(encoding='utf-8'))
    feature = next(r for r in scene['ice_features'] if r['id'] == 'goal_line_left')
    with np.load(identity['camera_path'], allow_pickle=False) as z:
        keys = list(z.files)
        stamp_key = next(k for k in ('native_pts', 'pts') if k in z)
        index = int(np.flatnonzero(z[stamp_key] == PTS)[0])
        camera = z['camera25'][index].copy()
        actual = int(z['exposure_representative_pts'][index])
    paths = [np.linspace(np.array(s[0]), np.array(s[1]), 301) for s in feature['segments_xyz']]
    model = []
    for xyz in paths:
        uv, _, valid = project_central_valid_full_brown(camera, xyz)
        for j in range(len(uv) - 1):
            if valid[j] and valid[j + 1]:
                model.append(uv[j:j + 2])
    segments = np.asarray(model)
    start, vector = segments[:, 0], segments[:, 1] - segments[:, 0]
    length = np.linalg.norm(vector, axis=1)
    t = np.clip(np.einsum('nsi,si->ns', native_uv[:, None] - start, vector) / length**2, 0., 1.)
    feet = start + t[:, :, None] * vector
    distance = np.linalg.norm(feet - native_uv[:, None], axis=2)
    nearest = np.argmin(distance, axis=1)
    normal = np.c_[vector[:, 1], -vector[:, 0]] / length[:, None]
    normal *= np.where(normal[:, 0] >= 0, 1., -1.)[:, None]
    signed = np.einsum('ni,ni->n', feet[np.arange(len(native_uv)), nearest] - native_uv, normal[nearest])
    report = dict(schema='rinkspace.user-yellow-line-native-replay.v1',
                  screenshot=str(screenshot), screenshot_sha256=hashlib.sha256(screenshot.read_bytes()).hexdigest(),
                  native_pts=PTS, actual_exposure_pts=actual, source_clock='04:10:38.800',
                  q_sha256=identity['q_sha256'], feature_id='goal_line_left',
                  screenshot_from_native_affine=affine.tolist(), registration_inliers=int(kept.sum()),
                  registration_error_screenshot_px={'median':float(np.median(registration_error)),
                                                   'p95':float(np.percentile(registration_error,95))},
                  source_coordinates_origin='User yellow line mapped through actual image-only screenshot registration; no model-derived UV.',
                  screenshot_yellow_center_samples=screenshot_uv.tolist(), native_source_uv=native_uv.tolist(),
                  current_model_signed_normal_error_native_px=signed.tolist(),
                  signed_error_definition='positive: model on image-right normal side of user line',
                  current_model_error_summary={'median':float(np.median(signed)), 'min':float(signed.min()),'max':float(signed.max())},
                  source_observation_not_yet_added_to_solver=True,
                  user_annotation_uncertainty_not_inferred_from_model_error=True,
                  no_model_or_camera_modified=True, whole_shot_accuracy_accepted=False)
    report['excluded_footer_or_outside_native_vertices'] = excluded
    report['registered_source_trace_reference'] = str(OUT / 'registered_user_yellow_left_goal_line_source_trace_v2.json')
    target=OUT/'same_a13_user_yellow_goal_line_replay_v2.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'report':str(target), 'registration':report['registration_error_screenshot_px'],
                      'inliers':report['registration_inliers'], 'error':report['current_model_error_summary']},ensure_ascii=False))


if __name__ == '__main__':
    main()
