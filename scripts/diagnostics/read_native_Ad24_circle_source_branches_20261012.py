"""Read the existing Ad24 queries in their actual RGB outer-stroke branches.

No model, projection, ellipse, solver or parameter state is used. The native
branch intervals come from the visible outer rim, distinct from its lettering.
The public reader keeps all unresolved material candidates and their flags.
"""
from pathlib import Path
import sys
import json
import hashlib
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from rinkspace.vision_source_shapes import read_source_dark_stroke_profiles

OUT = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/native_board_material_circle_RGB_review_20261012')


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    inputs = json.loads((OUT / 'current_q32_native_board_material_circle_RGB_review_inputs.json').read_text(encoding='utf8'))
    reference = inputs['sources']['C']
    packet_path = Path(reference['path'])
    if hashlib.sha256(packet_path.read_bytes()).hexdigest() != reference['sha256']:
        raise ValueError('Current circle packet identity changed')
    row = json.loads(packet_path.read_text(encoding='utf8'))['rows'][1]
    rawpath = Path(row['source_raw_frame'])
    if hashlib.sha256(rawpath.read_bytes()).hexdigest() != row['source_file_sha256']:
        raise ValueError('Original source frame identity changed')
    bgr = cv2.imread(str(rawpath))
    if hashlib.sha256(bgr.tobytes()).hexdigest() != row['source_native_bgr_pixels_sha256']:
        raise ValueError('Native pixels differ')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgbsha = hashlib.sha256(rgb.tobytes()).hexdigest()
    registry = json.loads((ROOT / 'events/ankara_2026/model_constraints/source_clock_registry.json').read_text(encoding='utf8'))
    source = next(x for x in registry['sources'] if x['time_base'] == '1/15360')
    binding = dict(source_sha256=source['source_sha256'], source_pts=229582592,
        time_base=source['time_base'], shot_id='okada_sp_shot_2', native_hw=[1080, 1920],
        decoded_native_rgb_sha256=rgbsha)
    uv = np.asarray(row['uv'], float)
    normals = np.array([[1., 0.]] * 6 + [[0., 1.]] * 4)
    # Only the actually visible outer thin stroke, retaining both white flanks.
    # These are source scan neighborhoods, never geometric fitted locations.
    absolute_intervals = np.array([[132, 145], [240, 251], [123, 135], [249, 262],
        [126, 138], [140, 152], [472, 494], [476, 495], [669, 690], [680, 700]], float)
    coordinate = np.einsum('ij,ij->i', uv, normals)
    result = read_source_dark_stroke_profiles(rgb, binding=binding, search_points_uv=uv,
        normals=normals, search_intervals_px=absolute_intervals-coordinate[:, None], keep_profiles=True)
    # Keep the full native neighborhood: the local outer material identity was
    # resolved by actual RGB review, not by erasing internal print competition.
    wide_intervals = absolute_intervals + np.array([-20., 20.])
    wide = read_source_dark_stroke_profiles(rgb, binding=binding, search_points_uv=uv,
        normals=normals, search_intervals_px=wide_intervals-coordinate[:, None], keep_profiles=True)
    outer = []
    for support, reading, context in zip(row['source_support'], result['readings'], wide['readings']):
        candidates = reading['all_actual_source_dark_stroke_candidates']
        if len(candidates) != 1:
            raise ValueError('Reviewed exterior branch has no unique local stroke candidate')
        candidate = candidates[0]
        side = support['side']
        interface = candidate['actual_RGB_entering_interface' if side in ('left_outer', 'top_outer')
            else 'actual_RGB_leaving_interface']
        outer.append(dict(original_point_index=reading['source_query_index'], side=side,
            actual_outer_interface_candidate_uv=interface['actual_source_transition_candidate_uv'],
            interface_candidate=interface, local_stroke_candidate=candidate,
            wider_source_competing_materials=context,
            black_stroke_midpoint_not_used_as_outer_contour=True,
            physical_role_basis='Actual RGB continuity of the exterior print circle; internal letter strokes distinct',
            exact_location_or_user_pixel_truth_not_certified=True,
            automatically_accepted_as_measurement=False))
    report = dict(schema='rinkspace.current-circle-actual-outer-stroke-RGB-readings.v1',
        current_packet=reference, packet_row=1, original_row_id=row['id'],
        original_uv=uv, source_RGB=dict(path=str(rawpath), sha256=row['source_file_sha256']),
        actual_native_RGB_sha256=rgbsha, source_binding=binding,
        visible_outer_stroke_search_intervals_absolute_native_px=absolute_intervals,
        physical_role_review='Actual thin exterior circular print stroke; inside lettering excluded by native RGB branch identity',
        reader_result=result, wider_native_source_context=wide,
        outermost_visible_material_interface_candidates=outer,
        black_stroke_center_candidates_are_not_current_outer_curve_observations=True,
        weights_roles_split_world_or_operator_not_changed=True,
        model_projection_or_fitted_shape_used=False, production_source_not_changed=True)
    path = OUT / 'C001_actual_outer_stroke_public_reader_candidates.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=serial), encoding='utf8')
    for reading in result['readings']:
        print(reading['source_query_index'], [(c['source_candidate_index'],
            np.round(c['actual_source_dark_stroke_center_candidate_uv'], 4).tolist(),
            c['disposition'], c['source_flags']) for c in reading['all_actual_source_dark_stroke_candidates']])
    print('Saved', path)


if __name__ == '__main__':
    main()
