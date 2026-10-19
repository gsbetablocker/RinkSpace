"""Measure actual R4 material transitions at the user's reported source time.

This source-only reader imports no camera or model. Bands come from the visible
unmarked source; the existing R4 middle/outer transition reader supplies the
same signed-gradient method and engineering scale convention.
"""
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

OUT = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/evidence/user_video_review_20261009/a13_complete_programme/R4_right_shift_04_08_40_200')
PTS = 229174016


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cv2.setNumThreads(1)
    source = OUT / f'actual_source_native_{PTS}.png'
    image = cv2.imread(str(source))
    assert image.shape == (1080, 1920, 3)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    marked = image.copy()
    rows = []
    for ring, bounds, sign, floor, color in (
        ('middle', (590, 740), -1, 15., (0, 230, 0)),
        ('outer', (830, 980), 1, 12., (230, 130, 0)),
    ):
        points, profiles, sigmas = [], [], []
        for y in (615, 620, 625):
            line = gaussian_filter1d(gray[y-2:y+3].mean(axis=0), 3.)
            score = np.gradient(line) * sign
            lo, hi = bounds
            ix = lo + int(np.argmax(score[lo:hi+1]))
            left, center, right = score[ix-1:ix+2]
            denominator = left - 2 * center + right
            dx = float(np.clip(.5 * (left-right) / denominator, -1., 1.)) if abs(denominator) > 1e-12 else 0.
            x = ix + dx
            a, b = ix, ix
            while a > lo and score[a-1] >= .5 * score[ix]:
                a -= 1
            while b < hi and score[b+1] >= .5 * score[ix]:
                b += 1
            # Keep the original working engineering floor; include the complete
            # visible gradient half-peak footprint, never an error-based scale.
            sigma = max(floor, x-(a-.5), b+.5-x)
            points.append([x, float(y)])
            sigmas.append(sigma)
            profiles.append(dict(y=y, source_search_x_interval=[lo, hi],
                source_gradient_half_peak_x_interval=[a-.5, b+.5],
                gradient_sign=sign, gradient_peak=float(score[ix]),
                horizontal_gradient_gaussian_sigma_px=3., vertical_average_rows=5))
            cv2.drawMarker(marked, (round(x), y), color, cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
        rows.append(dict(id=f'user_identified_R4_native{PTS}_{ring}_RGB_edge',
            pts=PTS, actual_pts=PTS, user_report_presentation_pts=229174272,
            feature_id='R4', curve=dict(kind='house_ring', side='R', row=4, ring=ring),
            role='middle_white_red_material_boundary' if ring == 'middle' else 'outer_red_white_material_boundary',
            split='fit', uv=points, sigma_px=sigmas, source_raw_frame=str(source),
            source_file_sha256=sha(source), source_time_s=PTS/15360,
            time_base='1/15360', source_video_sha256='f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891',
            source_search_support=profiles, user_confirmed_physical_identity=True,
            direct_human_GT=False, source_RGB_actual_measurement=True,
            model_projection_used_to_generate_or_select_uv=False, world=None,
            endpoint_observation=False, source_normals_measured=False,
            within_curve_statistical_independence_calibrated=False,
            sigma_basis='Original R4 reader 15px middle / 12px outer engineering floor, enlarged only to include actual gradient half-peak source footprint; not calibrated covariance or user pixel precision.',
            measurement_method='Source-only actual grayscale signed-gradient material transition, existing R4 reader, no model imports.',
            user_confirmation_reference=dict(path=str(OUT/'source_review.json'), sha256=sha(OUT/'source_review.json'))))
    target = OUT / 'actual_R4_two_material_edges_source_candidate.json'
    target.write_text(json.dumps(dict(schema='rinkspace.source-only-R4-material-transitions.v1',
        source_native_pts=PTS, rows=rows, original_source_rows_modified=False,
        source_only_qualified=False, status='pending_actual_marker_view_and_public_factor_replay'),
        ensure_ascii=False, indent=2), encoding='utf-8')
    cv2.imwrite(str(OUT/'actual_R4_material_edges_source_markers.png'), marked)
    cv2.imwrite(str(OUT/'actual_R4_material_edges_source_markers_crop.png'), marked[585:655,560:990])
    print(json.dumps(dict(path=str(target), rows=[dict(id=r['id'], uv=r['uv'], sigma_px=r['sigma_px'], profiles=r['source_search_support']) for r in rows])))


if __name__ == '__main__':
    main()
