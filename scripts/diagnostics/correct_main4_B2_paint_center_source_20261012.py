"""Source-only full painted-band reading; no camera/model/fit is imported.

Finite native scan brackets below were selected by direct RGB inspection of
both ice sides. They enclose the complete B2 band, rather than the interior
codec reversals that the existing candidate reader splits into fragments.
"""
from pathlib import Path
import hashlib
import json
import pickle
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from rinkspace.vision_source_shapes import read_source_dark_stroke_profiles

SYS = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
OUT = SYS / 'same_manual_B2_contact_RGB_review_20261012'
CONFIG = ROOT / 'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_ce235f_wide_curve_budget_20261012.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                                   default=plain) + '\n', encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True)
    config = read(CONFIG)
    parent = read(config['source_options']['--original-ice-pixel-correction-source'])
    original_ref = parent['original_absolute_packet']
    assert ref(original_ref['path']) == original_ref
    items = pickle.loads(Path(original_ref['path']).read_bytes())['programme_absolute_rows']
    item = items[10398]
    old = item['source_row']
    assert item['collection_role'] == 'FIT' and old['id'] == 'B2'
    assert item['group_actual_pts'] == 229702144 and len(old['uv']) == 5
    rawpath = SYS / 'original_ice_FIT_RGB_review/229702144_independently_decoded_RAW_NATIVE.png'
    bgr = cv2.imread(str(rawpath))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rawref = ref(rawpath)
    rgbsha = hashlib.sha256(rgb.tobytes()).hexdigest()
    binding = dict(source_sha256=parent['source_video_sha256'], source_pts=229702144,
                   time_base=parent['time_base'], shot_id='main4', native_hw=[1080, 1920],
                   decoded_native_rgb_sha256=rgbsha)
    candidates = read_source_dark_stroke_profiles(rgb, binding=binding,
        search_points_uv=old['uv'], normals=[[0, 1]] * 5,
        search_intervals_px=[[-45, 30]] * 5, keep_profiles=False)
    # All four finite brackets include both directly viewed ice-side flanks.
    # No normal, UV or bracket is derived from a model, a fitted line or another
    # exposure. The native x remains the original source x.
    brackets = {1: (677, 721), 2: (741, 777), 3: (825, 861), 4: (914, 963)}
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    readings = []
    for index, (lo, hi) in brackets.items():
        x, yold = np.asarray(old['uv'][index], float)
        assert x == round(x)
        ys = np.arange(lo, hi + 1, dtype=float)
        lum = gray[lo:hi + 1, int(x) - 2:int(x) + 3].mean(1)
        endpoints = np.array([lum[:3].mean(), lum[-3:].mean()])
        background = np.linspace(*endpoints, len(lum))
        darkness = np.maximum(background - lum, 0.)
        assert darkness.sum() > 0 and hi < gray.shape[0] and lo > 0
        quantiles = np.interp([.1, .5, .9], (np.cumsum(darkness) - darkness * .5) / darkness.sum(), ys)
        readings.append(dict(original_point_index=index, original_uv=[x, yold],
            corrected_uv=[x, float(quantiles[1])], corrected_normal_interval_px=None,
            method='complete_actual_native_RGB_darkness_mass_q50_with_two_directly_reviewed_ice_flanks',
            source_scan_axis='y', fixed_native_x=x,
            complete_paint_scan_bracket_y=[lo, hi], native_tangent_half_width_px=2,
            measured_luminance_profile=lum, native_scan_y=ys,
            measured_ice_flank_luminance_endpoints=endpoints,
            observed_darkness_mass_profile=darkness,
            source_darkness_q10_q50_q90_y=quantiles,
            physical_edge_role='paint_centerline_of_complete_B2_stroke',
            actual_paint_band_and_both_ice_sides_visually_reviewed=True,
            interior_codec_reversals_not_used_as_paint_borders=True,
            model_used_for_navigation_or_peak_selection=False,
            observed_UV_generated_from_model=False, raw_source_pixels_read=True,
            source_center_is_not_model_projection=True,
            paint_width_is_not_location_deadzone=True,
            source_center_is_not_independently_calibrated_localization_precision=True,
            user_precise_UV_confirmation=False, source_rgb={**rawref, 'decoded_native_RGB_sha256': rgbsha}))
    report = dict(schema='rinkspace.actual-source-B2-complete-stroke-RGB-review.v1',
        original_packet_row_index=10398, original_absolute_packet=original_ref,
        source_binding=binding, source_rgb=rawref, actual_native_RGB_sha256=rgbsha,
        original_uv=old['uv'], original_sigma=old['sigma'],
        public_dark_stroke_candidates_unmodified=candidates,
        public_candidate_gap='Complete physical paint contains internal texture reversals; local dark candidates 3/4 split this one band and do not authenticate its full center.',
        full_band_readings=readings,
        unresolved_points=[dict(original_point_index=0, original_uv=old['uv'][0],
            reason='Actual RGB shows the B2/contact/kickplate intersection; both ice-side B2 interfaces are not separately visible. Reading the kickplate darkness as B2 center would assign the wrong material.')],
        no_model_or_other_frame_or_interpolated_measurement=True,
        TOP_preserved_by_latest_direct_user_confirmation=True,
        current_contact_note='Current row10400 has 8 points; uploaded historical overlay used 4 older points. Its old bad rightmost yellow-band points are not current row10400.',
        contact_current8_directly_viewed=True,
        contact_current8_no_conclusively_wrong_material_point_identified=True,
        contact_subpixel_accuracy_not_certified=True,
        source_correction_does_not_accept_camera_geometry_or_complete_fit=True)
    reportpath = OUT / 'main4_B2_10398_full_band_actual_RGB_readings_20261012.json'
    write(reportpath, report)
    qrecord = dict(original_packet_row_index=10398,
        locator={k: item[k] for k in ('block', 'collection_role', 'group_order', 'row_order', 'group_actual_pts')},
        row_id='B2', source_pts=old['pts'],
        original_uv_float64_sha256=hashlib.sha256(np.asarray(old['uv'], np.float64).tobytes()).hexdigest(),
        excluded_point_indices=[0], review_basis='original_rgb_only',
        reason=report['unresolved_points'][0]['reason'],
        decision_is_not_from_model_residual_or_cost=True,
        review_authority=ref(reportpath), native_rgb_evidence=[{**rawref,
            'source_pts':229702144, 'source_video_sha256':parent['source_video_sha256'], 'time_base':parent['time_base']}])
    qualification = dict(schema='rinkspace.source-only-point-qualification.v1',
        source_family='original_programme_absolute',
        original_packet={**original_ref, 'format':'original_absolute_compact_pickle'},
        source_video_sha256=parent['source_video_sha256'], time_base=parent['time_base'],
        original_packet_bytes_modified=False, UV_sigma_world_split_modified=False,
        replacement_observation_added=False, qualification_is_source_only_not_camera_accuracy=True,
        records=[qrecord])
    qpath = OUT / 'original_B2_10398_point0_RGB_unresolved_qualification_20261012.json'
    write(qpath, qualification)
    indices = list(brackets)
    record = dict(original_packet_row_index=10398, point_indices=indices,
        original_pts=old['pts'], original_id='B2', original_split=old['split'],
        original_collection_role='FIT', original_actual_pts=229702144,
        original_block=item['block'], original_group_order=item['group_order'], original_row_order=item['row_order'],
        original_uv=np.asarray(old['uv'])[indices], original_sigma=np.asarray(old['sigma'])[indices],
        original_full_uv_float64_sha256=qrecord['original_uv_float64_sha256'],
        original_geometry_identity=['line','B2'], physical_identity_changed=False,
        measurement_role='painted_line_center', physical_curve_id='B2',
        physical_edge_role='paint centerline of complete visible B2 stroke; native fixed-x scans',
        paint_footprint_is_not_location_uncertainty=True, remove_original_normal_interval=True,
        corrected_uv=[r['corrected_uv'] for r in readings], corrected_normal_interval_px=None,
        retired_original_point_indices=[0], source_point_qualification_reference=ref(qpath),
        source_only_reading=readings)
    packet = dict(schema=parent['schema'], source_video_sha256=parent['source_video_sha256'],
        time_base=parent['time_base'], original_absolute_packet=original_ref,
        model_projection_used_for_selection=False, historical_packet_modified=False,
        HELD_promoted_to_FIT=False, corrected_point_count=4, native_hw=[1080,1920],
        source_evidence=[{**rawref,'pts':229702144,'decoded_native_RGB_sha256':rgbsha},ref(reportpath)],
        records=[record], pixel_covariance_measured=False, whole_fit_accuracy_accepted=False,
        observed_UV_generated_from_model=False, raw_source_pixels_read=True,
        no_world_radius_sigma_weight_or_user_truth_change=True,
        source_review_complete_painted_row_measured_or_RGB_retired=True)
    packetpath = OUT / 'original_B2_10398_four_actual_RGB_centers_source_correction_20261012.json'
    write(packetpath, packet)
    # Evidence only: original marks red, source-only centers green; right side
    # is the same unmarked native pixel crop at 2x nearest-neighbor.
    canvas = np.full((5 * 225, 900, 3), 255, np.uint8)
    for index, (x, y) in enumerate(np.asarray(old['uv'], int)):
        patch = cv2.resize(bgr[y-60:y+45, x-65:x+65], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        canvas[index*225:index*225+210,:260] = patch
        canvas[index*225:index*225+210,630:890] = patch
        cv2.circle(canvas,(130,index*225+120),4,(0,0,255),1)
        if index in brackets:
            ynew = readings[index-1]['corrected_uv'][1]
            cv2.circle(canvas,(130,index*225+int(round(2*(ynew-y+60)))),4,(0,180,0),1)
        cv2.putText(canvas,f'B2 {index} red=old green=RGB center',(275,index*225+35),cv2.FONT_HERSHEY_SIMPLEX,.52,(0,0,0),1)
        if index == 0:
            cv2.putText(canvas,'RGB role unresolved; retire point',(275,index*225+65),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
    imagepath = OUT / '229702144_B2_original_vs_full_RGB_center_RAW.png'
    cv2.imwrite(str(imagepath), canvas)
    print(json.dumps(dict(packet=ref(packetpath), qualification=ref(qpath), reading=ref(reportpath),
        evidence=ref(imagepath), corrected_uv=record['corrected_uv'],
        source_vertical_changes_px=[r['corrected_uv'][1]-r['original_uv'][1] for r in readings]), ensure_ascii=False))


if __name__ == '__main__':
    main()
