"""Reread four known source failures using two cached native RGB frames.

No network weights, video decode, model solve or full-state history is loaded.
Source brackets come from saved RGB readings; none are a fitted world curve.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import cv2
import numpy as np
from PIL import Image
from rinkspace.vision_source_fit import OriginalFITSourceReader

SYS = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
RUN = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_ce235f_wide_curve_budget_resume_20261012')
Q32 = '917c0fa45f80af8e16c74d0f2d56486fdb500b591357add3dc2b240b3de64cdf'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def ref(path):
    path = Path(path)
    with path.open('rb') as stream:digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path.resolve()), sha256=digest)


def write(path, value):Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, default=SYS / 'vision_public_source_reread4_actual_RGB_20261012')
    args = parser.parse_args()
    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    packet_path = SYS / 'original_ice_and_TOP_all_records_plus_C0_R4_R3_actual_RGB_20261012.json'
    packet = read(packet_path)
    records = {r['original_packet_row_index']: r for r in packet['records']}
    b2_path = SYS / 'same_manual_B2_contact_RGB_review_20261012/main4_B2_10398_full_band_actual_RGB_readings_20261012.json'
    b2 = read(b2_path)
    b2_correction = read(SYS / 'same_manual_B2_contact_RGB_review_20261012/original_B2_10398_four_actual_RGB_centers_source_correction_20261012.json')['records'][0]
    frames, raw_refs = {}, {}
    for ordinal in (10396, 10397, 11488):
        record = records[ordinal]
        readings = record['source_only_reading']
        raw = readings[0]['source_rgb']
        pts = int(record['original_actual_pts'])
        binding = dict(source_sha256=packet['source_video_sha256'], source_pts=pts, time_base=packet['time_base'],
            shot_id=record['original_block'], native_hw=[1080, 1920],
            decoded_native_rgb_sha256=raw['decoded_native_RGB_sha256'])
        raw_refs[pts] = raw
        definition = dict(source_binding=binding, physical_identity=record['original_geometry_identity'],
            physical_role=record['physical_edge_role'], identity_authority=ref(packet_path),
            original_point_indices=record['point_indices'])
        if ordinal == 10396:
            definition.update(measurement_operator='paired_material_boundary_midline', two_sides_belong_to_same_painted_stroke=True,
                source_search_normals=[r['actual_RGB_entering_interface']['source_search_normal'] for r in readings],
                entering_search_intervals_px=[r['actual_RGB_entering_interface']['source_search_normal_interval_px'] for r in readings],
                leaving_search_intervals_px=[r['actual_RGB_leaving_interface']['source_search_normal_interval_px'] for r in readings])
        else:
            definition.update(measurement_operator='signed_material_transition', material_boundary_role_confirmed=True,
                source_search_normals=[r['actual_source_boundary_reading']['source_search_normal'] for r in readings],
                search_intervals_px=[r['actual_source_boundary_reading']['source_search_normal_interval_px'] for r in readings],
                context_search_intervals_px=[r['actual_source_boundary_reading'].get(
                    'source_broad_navigation_interval_px', r['actual_source_boundary_reading']['source_search_normal_interval_px']) for r in readings],
                transition=readings[0]['actual_source_boundary_reading']['material_transition'])
        entry = dict(original_packet_row_index=ordinal, definition=definition)
        frames.setdefault(pts, dict(source_binding=binding, records=[]))['records'].append(entry)
    binding = frames[229702144]['source_binding']
    # The saved complete-band darkness quantile only splits two search ranges.
    # It is NOT emitted as a centre. Both actual signed boundaries are reread.
    entering, leaving = [], []
    for reading in b2['full_band_readings']:
        index = reading['original_point_index']
        old_y = b2['original_uv'][index][1]
        low, high = reading['complete_paint_scan_bracket_y']
        split = reading['source_darkness_q10_q50_q90_y'][1]
        entering.append([low - old_y, split - old_y])
        leaving.append([split - old_y, high - old_y])
    frames[229702144]['records'].append(dict(original_packet_row_index=10398,
        definition=dict(source_binding=binding, physical_identity=b2_correction['original_geometry_identity'],
            physical_role='painted_stroke_center_axis', identity_authority=ref(b2_path),
            original_point_indices=[r['original_point_index'] for r in b2['full_band_readings']],
            measurement_operator='paired_material_boundary_midline', two_sides_belong_to_same_painted_stroke=True,
            source_search_normals=[[0., 1.]] * len(entering), entering_search_intervals_px=entering,
            leaving_search_intervals_px=leaving, tangent_half_width_px=2)))
    config = read(ROOT / 'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_ce235f_wide_curve_budget_20261012.json')
    request = dict(schema='rinkspace.original-FIT-source-RGB-reread-request.v1',
        original_absolute_packet=packet['original_absolute_packet'],
        complete_state=ref(RUN / 'accepted_states' / Q32 / 'complete_state_UNACCEPTED.npz'),
        world_initializer=ref(config['source_options']['--world-init']), frames=list(frames.values()))
    request_path = output / 'prepared_source_request.json'
    write(request_path, request)
    reader = OriginalFITSourceReader(request_path)
    results, summaries, atlas_images = [], [], []
    for pts, frame in frames.items():
        raw_ref = raw_refs[pts]
        if ref(raw_ref['path'])['sha256'] != raw_ref['sha256']:raise ValueError('Cached raw source bytes differ')
        rgb = np.asarray(Image.open(raw_ref['path']).convert('RGB'))
        result = reader.read(rgb, frame['source_binding'])
        results.append(result)
        for edit in result['row_edits']:
            ordinal = edit['original_packet_row_index']
            original = next(r for r in reader.rows_by_pts[pts] if r['original_packet_row_index'] == ordinal)['source_row']
            actual = {r['original_point_index']: r for r in result['point_decisions'] if r['original_packet_row_index'] == ordinal}
            indices = edit['corrected_original_point_indices']
            uv = np.asarray(edit['corrected_uv'], float).reshape(-1, 2)
            previous = records[ordinal]['corrected_uv'] if ordinal in records else b2_correction['corrected_uv']
            previous_indices = records[ordinal]['point_indices'] if ordinal in records else b2_correction['point_indices']
            previous_by_id = dict(zip(previous_indices, previous))
            compared = [float(np.linalg.norm(np.asarray(actual[i]['actual_source_uv']) - previous_by_id[i])) for i in indices]
            summaries.append(dict(original_packet_row_index=ordinal, source_pts=pts, row_id=original['id'],
                measured_original_point_indices=indices, withheld_original_point_indices=edit['pending_original_point_indices'],
                geometric_midline_or_material_boundary_uv=edit['corrected_uv'],
                old_original_UV_displacement_px=[float(np.linalg.norm(p - np.asarray(original['uv'])[i])) for i, p in zip(indices, uv)],
                prior_corrected_source_UV_displacement_px=compared,
                point_rejection_reasons={str(i): actual[i]['source_flags'] for i in edit['pending_original_point_indices']},
                source_rgb=raw_ref, original_WORLD_sigma_preserved=True))
            crops = []
            for index in range(len(original['uv'])):
                old = np.asarray(original['uv'][index], float)
                candidate = actual[index]['actual_source_uv']
                center = np.asarray(candidate if candidate is not None else old)
                x, y = np.rint(.5 * (center + old)).astype(int)
                x0, y0 = max(0, x - 65), max(0, y - 55)
                crop = rgb[y0:min(rgb.shape[0], y + 56), x0:min(rgb.shape[1], x + 66)].copy()
                cv2.drawMarker(crop, tuple(np.rint(old - [x0, y0]).astype(int)), (255, 50, 40), cv2.MARKER_CROSS, 9, 1)
                if candidate is not None:
                    cv2.drawMarker(crop, tuple(np.rint(np.asarray(candidate) - [x0, y0]).astype(int)), (20, 180, 255), cv2.MARKER_CROSS, 9, 1)
                tile = np.full((140, 165, 3), 250, np.uint8)
                tile[:crop.shape[0], :crop.shape[1]] = crop
                cv2.putText(tile, f'{ordinal} p{index} ' + ('read' if candidate is not None else 'pending'), (3, 132), cv2.FONT_HERSHEY_SIMPLEX, .35, (10, 10, 10), 1)
                crops.append(tile)
            cols = 5
            atlas = np.full((140 * ((len(crops) + cols - 1) // cols), 165 * cols, 3), 245, np.uint8)
            for index, crop in enumerate(crops):atlas[(index // cols) * 140:(index // cols + 1) * 140, (index % cols) * 165:(index % cols + 1) * 165] = crop
            path = output / f'row{ordinal}_actual_source_old_red_new_blue.png'
            Image.fromarray(atlas).save(path)
            atlas_images.append(str(path))
    report = dict(schema='rinkspace.actual-source-four-row-midline-material-verification.v1',
        prepared_request=ref(request_path), cached_raw_source_frames=[raw_refs[p] for p in frames],
        original_geometry_and_packet_prepared_once=True, frames_reread=len(results), row_summaries=summaries,
        visual_atlas_paths=atlas_images, actual_reader_results=results,
        selected_existing_source_brackets_not_new_geometry_or_projection=True,
        no_new_weights_video_decode_fit_or_benchmark=True, accuracy_accepted=False)
    write(output / 'actual_source_reread_receipt.json', report)
    print(json.dumps(dict(rows=[dict(row=r['original_packet_row_index'], measured=r['measured_original_point_indices'], pending=r['withheld_original_point_indices']) for r in summaries],
        receipt=str(output / 'actual_source_reread_receipt.json')), ensure_ascii=False), flush=True)


if __name__ == '__main__':main()
