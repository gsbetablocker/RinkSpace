"""Read current source/support provenance and one saved B2 RGB counterexample.

No solver, Jacobian, new video decode, source mutation or image export.
"""
from __future__ import annotations

import collections
import hashlib
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import cv2
import numpy as np
from rinkspace.vision_source_shapes import read_source_dark_stroke_profiles

H = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
RUN = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_review86_q276_vertical_print_station_one_cycle_20261011')
Q = '8a15a4ed0c82beb490993db9c58294af592096e8159d9652b5bd5c50217c71eb'
CONFIG = ROOT / 'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_q276_vertical_print_station_one_cycle_20261011.json'
OUT = H / 'current_absolute_support_semantics_and_old_B2_same_RGB_counterexample_q8a15_readonly_20261011.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def reference(path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError(f'Changed source reference: {path}')
    return dict(path=str(path), sha256=digest)


def summaries(rows):
    result = []
    for index, row in rows:
        value = dict(row_index=index, id=row['id'], pts=row.get('pts'), actual_pts=row.get('actual_pts'),
                     curve=row['curve'], split=row['split'], UV_count=len(row['uv']))
        for key in ('measurement_role', 'measurement_operator', 'geometry_role', 'sigma_semantics',
                    'source_optical_normal_support_authorized', 'boundary_location_observation',
                    'source_optical_footprint_is_not_location_zero_cost_interval', 'sigma_normalization_formula'):
            if key in row:
                value[key] = row[key]
        s = np.asarray(row.get('sigma_px', row.get('sigma', [])), float)
        value['stored_sigma_range'] = [float(s.min()), float(s.max())] if s.size else None
        bounds = row.get('normal_interval_px', row.get('normal_interval'))
        value['normal_interval_range'] = None if bounds is None else [float(np.min(bounds)), float(np.max(bounds))]
        result.append(value)
    return result


def main():
    manifest = load(RUN / 'input_manifest.json')
    config = load(CONFIG)
    options = config['source_options']
    refs = [reference(CONFIG), reference(RUN / 'input_manifest.json')]
    packets = {}
    contracts = dict(ice='additional_real_native_ice_curves', board='additional_real_native_board_edges',
                     material='additional_real_native_board_material_points', circle='additional_real_native_board_circles')
    for name, key in contracts.items():
        contract = manifest[key]
        refs.append(reference(contract['packet_path'], contract['packet_sha256']))
        packets[name] = load(contract['packet_path'])
    original_ref = manifest['original_absolute_input_compaction']['original_packet']
    refs.append(reference(original_ref['path'], original_ref['sha256']))
    with Path(original_ref['path']).open('rb') as stream:
        original = pickle.load(stream)
    old_b2 = [(i, r) for i, r in enumerate(original['programme_absolute_rows']) if r['source_row']['id'] == 'B2']
    old = original['programme_absolute_rows'][2995]
    uv = np.asarray(old['source_row']['uv'], float)
    assert old['block'] == 'main7' and old['group_actual_pts'] == 230571776 and len(uv) == 23
    assert 'normal_interval' not in old['source_row'] and 'normal' not in old['source_row']

    correction_ref = manifest['original_ice_pixel_correction']
    refs.append(reference(correction_ref['packet_path'], correction_ref['packet_sha256']))
    correction = load(correction_ref['packet_path'])
    assert not any(r['original_packet_row_index'] == 2995 for r in correction['records'])
    f2_corrections = [dict(record_index=i, original_packet_row_index=r['original_packet_row_index'],
        pts=r['original_pts'], id=r['original_id'], physical_curve_id=r.get('physical_curve_id'),
        point_count=len(r['point_indices']), direct_user_pixel_correction=r.get('direct_user_pixel_correction', False),
        remove_original_normal_interval=r.get('remove_original_normal_interval', False),
        corrected_interval_range=None if r.get('corrected_normal_interval_px') is None else
        [float(np.min(r['corrected_normal_interval_px'])), float(np.max(r['corrected_normal_interval_px']))])
        for i, r in enumerate(correction['records']) if str(r.get('physical_curve_id', '')).startswith('F2')]

    envelope_path = options['--practical-human-location-envelope']
    envelope = load(envelope_path)
    refs.append(reference(envelope_path))
    restoration_path = RUN / 'accepted_states' / Q / 'truth_restoration_state.json'
    restoration = load(restoration_path)
    refs.append(reference(restoration_path))
    guard_rows = []
    for family, values in restoration['families'].items():
        for i, identity in enumerate(values['group_ids']):
            if any(s in identity for s in ('F2', 'B2', 'Ad4', 'Ad24', 'Ad25', 'Ad26')):
                policy = next((r for r in envelope['records'] if r['id'] == identity and r['family'] == family), None)
                guard_rows.append(dict(family=family, index=i, id=identity, current_restoration_limit_px=values['limits'][i],
                                       engineering_restoration_uncertainty_px=values['uncertainties'][i],
                                       practical_location_policy=policy))

    raw = H / 'original_ice_FIT_RGB_review/230571776_independently_decoded_RAW_NATIVE.png'
    receipt_path = raw.parent / 'receipt.json'
    receipt = load(receipt_path)
    raw_ref = next(r for r in receipt['frames'] if r['native_pts'] == 230571776)
    refs.extend([reference(raw, raw_ref['sha256']), reference(receipt_path)])
    bgr = cv2.imread(str(raw), cv2.IMREAD_COLOR)
    assert hashlib.sha256(bgr.tobytes()).hexdigest() == raw_ref['native_bgr24_sha256']
    rgb = bgr[:, :, ::-1].copy()
    origin = Path(old['source_row']['origin'])
    refs.append(reference(origin))
    with np.load(origin, allow_pickle=False) as source:
        saved_uv = source['observed_uv'][source['physical_id'] == 'B2'].astype(float)
        assert np.array_equal(saved_uv, uv)
    provenance_path = origin.parent / 'provenance.json'
    provenance = load(provenance_path)
    parent = next(r for r in provenance['frames'] if r['pts'] == 230571776)
    refs.extend([reference(provenance_path), reference(parent['source'], parent['source_sha256'])])
    with np.load(parent['source'], allow_pickle=False) as source:
        assert np.array_equal(source['observed_uv'][source['physical_id'] == 'B2'].astype(float), uv)
    assert parent['removed'] == []

    # A line fit chooses a source-image search normal only; never output pixels.
    fit = cv2.fitLine(uv.astype(np.float32), cv2.DIST_WELSCH, 3., .01, .01).ravel()
    tangent = fit[:2].astype(float)
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    binding = dict(source_sha256=manifest['additional_real_native_ice_curves']['source_video_sha256'],
                   source_pts=230571776, time_base='1/15360', shot_id='main7', native_hw=[1080, 1920])
    rgb_reads = []
    for half_width in (2, 8):
        reading = read_source_dark_stroke_profiles(rgb, binding=binding, search_points_uv=uv,
            normals=np.tile(normal, (len(uv), 1)), search_intervals_px=np.tile([-32., 32.], (len(uv), 1)),
            tangent_half_width_px=half_width, engineering_support_margin_px=0., keep_profiles=False)
        rows = []
        for row in reading['readings']:
            i = row['source_query_index']
            accepted = [r for r in row['all_actual_source_dark_stroke_candidates']
                        if r['disposition'] == 'measured_dark_stroke_center_candidate']
            candidates = []
            for candidate in accepted:
                first = np.asarray(candidate['actual_RGB_entering_interface']['actual_source_transition_candidate_uv'])
                last = np.asarray(candidate['actual_RGB_leaving_interface']['actual_source_transition_candidate_uv'])
                bounds = [float((first - uv[i]) @ normal), float((last - uv[i]) @ normal)]
                if bounds[0] <= 0 <= bounds[1]:
                    midpoint = .5 * (first + last)
                    candidates.append(dict(two_RGB_interface_q50_offsets_px=bounds, material_midline_uv=midpoint.tolist(),
                        old_minus_two_material_interface_midline_px=float((uv[i] - midpoint) @ normal)))
            item = dict(point_index=i, original_UV=uv[i].tolist(), accepted_band_count=len(accepted),
                        source_bands_bracketing_original_query=candidates)
            rows.append(item)
        values = [r['source_bands_bracketing_original_query'][0]['old_minus_two_material_interface_midline_px']
                  for r in rows if len(r['source_bands_bracketing_original_query']) == 1]
        rgb_reads.append(dict(tangent_half_width_px=half_width, original_query_count=len(uv),
            unique_source_band_bracketing_query_count=len(values), source_normal=normal.tolist(),
            search_interval_native_px=[-32., 32.], signed_error_percentiles=dict(zip(
                ('minimum', 'p05', 'median', 'p95', 'maximum'), map(float, np.percentile(values, [0, 5, 50, 95, 100])))),
            absolute_error_median_px=float(np.median(np.abs(values))),
            absolute_error_maximum_px=float(np.max(np.abs(values))), readings=rows))

    code_files = ['calibration/solver_runtime/legacy/snapshot/three_frame_refit/program_tracking/main7_source_camera_candidate.py',
                  'calibration/program_tracking/ice_refine.py', 'calibration/program_tracking/paint_centerline.py',
                  'src/rinkspace/vision_source_shapes.py',
                  'calibration/solver_runtime/legacy/structured_programme_source_candidate/native_ice_curve_source.py',
                  'calibration/solver_runtime/legacy/structured_programme_source_candidate/native_board_edge_source.py',
                  'calibration/solver_runtime/legacy/structured_programme_source_candidate/human_stroke_truth.py',
                  'calibration/solver_runtime/legacy/structured_programme_source_candidate/original_ice_pixel_correction.py']
    refs.extend(reference(ROOT / name) for name in code_files)
    refs.append(reference(__file__))
    result = dict(schema='rinkspace.current-absolute-source-support-semantics-readonly.v1', current_q_sha256=Q,
        fitting_or_J_generation=False, new_source_images_written=0, source_packets_or_supports_changed=False,
        authenticated_references=refs,
        conclusions=[
            'New F2 rows101..107: q50 boundary locations, optical support disabled,65 FIT pixels over7 exposures, external visible-arc mass; no search/paint interval dead zone.',
            'New B2 rows87..100:34 two-material-interface midpoint FIT pixels over14 exposures; no normal interval. Original row2995 remains a separate23-point raw source with assumed sigma3.',
            'New LOW Ad4 rows185..190:16 midpoint boundary-location FIT pixels over6 exposures; blur intervals preserved as evidence, not applied to position loss.',
            'F2 historical corrected ring boundaries intentionally retain either RGB q10..q90 interface blur or registered user raster support; those are not calibrated noise. User location envelopes and restoration limits are separate engineering rules.',
            'The saved main7 B2 row is not replicated over many exposures: id=B2 has one main7 FIT row. Physical-name aliases must be crosschecked separately in the actual binder.',
            'The available legacy blue reader uses the midpoint of24-percent-of-colour-contrast threshold interfaces, not a colour-mass centroid. Historical generation source SHA is absent, so available code alone does not certify its historical version.',
            'All23 saved B2 queries have one source blue band in the authenticated RGB; midpoint discrepancy stays below1.76px, with22/23 tangent-width sensitivity below1.91px. This falsifies a large colour-mass-center mismatch on this saved row, not all source identity/geometry causes.'
        ],
        new_F2_rows=summaries(list(enumerate(packets['ice']['rows']))[101:108]),
        new_B2_rows=summaries(list(enumerate(packets['ice']['rows']))[87:101]),
        new_LOW_Ad4_rows=summaries(list(enumerate(packets['board']['rows']))[185:191]),
        F2_original_pixel_corrections=f2_corrections, current_protection_limits_and_envelopes=guard_rows,
        original_B2_named_rows=[dict(original_packet_row_index=i, block=r['block'], collection_role=r['collection_role'],
            actual_pts=r['group_actual_pts'], UV_count=len(r['source_row']['uv']), origin=r['source_row'].get('origin'),
            original_normal_interval_present='normal_interval' in r['source_row']) for i, r in old_b2],
        B2_2995_current_source=dict(block='main7', current_programme_FIT_row_index=413,
            source_scalar_start=808558, source_scalar_stop=808604, actual_pts=230571776, UV_count=23,
            sigma_assumed_px=3., corrected_by_current_pixel_packet=False,
            source_measurement_role_or_covariance_or_interval_certificate=False,
            original_RGB_cached_and_authenticated=True, legacy_npz_source_pixels_equal_original_packet=True),
        old_B2_same_RGB_counterexample=rgb_reads,
        limitations=['Source image normal comes from saved UV direction; output border pixels are measured from original RGB, not the fitted line.',
            'Two projected material-interface midpoint is a source-space approximation, not exact Euclidean paint-center metrology.',
            'No source changes or narrower uncertainty proposed. This audit does not claim the remaining absolute geometry bias is fully explained.'])
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(path=str(OUT), sha256=sha(OUT), bytes=OUT.stat().st_size,
        B2_counterexample=[{k:v for k,v in r.items() if k != 'readings'} for r in rgb_reads]), ensure_ascii=False))


if __name__ == '__main__':
    main()
