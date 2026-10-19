"""Read actual ordinary paint sides for already visually reviewed native rows.

No neural/world/camera initialization or optimization. Candidate selection is
limited to one complete source band; competing materials remain unresolved.
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import cv2
import numpy as np
from rinkspace.vision_source_shapes import read_source_dark_stroke_profiles, read_source_chromatic_stroke_centers

SYS = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
OUT = SYS / 'source_pixel_review_additional/native_ice_current_remaining_actual_RGB_review_20261012'


def plain(value):
    if isinstance(value, np.ndarray):return plain(value.tolist())
    if isinstance(value, np.generic):return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):return None
    if isinstance(value, dict):return {key:plain(item) for key,item in value.items()}
    if isinstance(value, (list, tuple)):return [plain(item) for item in value]
    return value


def main():
    inputs = json.loads((OUT/'current_active_remaining_native_ice_RGB_review_inputs.json').read_text(encoding='utf8'))
    packet = json.loads(Path(inputs['packet']['path']).read_text(encoding='utf8'))
    targets = (0,4,6,16,17,18,19,20,38,40,42,44,49,50,82,83,84,85)
    raw_cache, records = {}, []
    for current in inputs['rows']:
        index = current['row_index']
        if index not in targets:continue
        row = packet['rows'][index]
        path = current['raw_source']['path']
        if path not in raw_cache:
            raw = cv2.imread(path)
            raw_cache[path] = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        rgb = raw_cache[path]
        uv = np.asarray(row['uv'], float)
        indices = current['active_original_point_indices']
        if 'normal' in row:
            normals = np.asarray(row['normal'], float)
        elif index in (0,4,6,16,17,18,19,20,49,50):
            # Existing measurements are exact x scans, so preserve that
            # source estimator's vertical reading direction, not a world normal.
            normals = np.tile([0.,1.], (len(uv),1))
        else:
            tangent = np.gradient(uv, axis=0) if len(uv)>1 else np.array([[1.,0.]])
            normals = np.c_[-tangent[:,1],tangent[:,0]]
        binding = dict(source_sha256=inputs['source_video_sha256'], source_pts=current['actual_pts'],
            time_base=inputs['time_base'], shot_id='current_native_source_'+str(current['actual_pts']),
            native_hw=list(rgb.shape[:2]), decoded_native_rgb_sha256=current['decoded_native_RGB_sha256'])
        common = dict(binding=binding, search_points_uv=uv[indices], normals=normals[indices],
            search_intervals_px=[[-32.,32.]]*len(indices), keep_profiles=False)
        if index == 6:
            result = read_source_chromatic_stroke_centers(rgb, material_signal='red_minus_green', **common)
            key = 'all_actual_source_chromatic_stroke_candidates'
        else:
            result = read_source_dark_stroke_profiles(rgb, **common)
            key = 'all_actual_source_dark_stroke_candidates'
        decisions = []
        for original_index, reading in zip(indices, result['readings']):
            candidates = reading[key]
            measurable = [c for c in candidates if np.asarray(c['actual_source_uv']).shape == (2,)
                and not any(c['source_flags'].values())]
            unique = len(candidates) == len(measurable) == 1
            observed = measurable[0]['actual_source_uv'] if unique else None
            decisions.append(dict(original_point_index=original_index, current_uv=uv[original_index],
                candidate_uv=observed, candidate_offset_native_px=None if observed is None else observed-uv[original_index],
                candidate_classification='unique_two_actual_material_interfaces_pending_physical_review' if unique else 'source_band_or_sides_unresolved',
                all_actual_source_candidates=candidates, current_source_UV_sigma_or_qualification_modified=False))
        records.append(dict(row_index=index, id=row['id'], actual_pts=current['actual_pts'], curve=row['curve'],
            current_uv_float64_sha256=current['current_uv_float64_sha256'], source_RGB=current['raw_source'],
            RGB_sha256=result['RGB_sha256'], source_binding=binding, point_decisions=decisions,
            diagnostic_search_only_not_a_loss_or_geometry_constraint=True,
            ordinary_line_geometric_center_from_two_material_sides=True,
            no_nearest_old_UV_candidate_selection=True, candidates_not_FIT_admitted=True))
    path = OUT/'ordinary_paint_two_actual_material_sides_candidate_readings.json'
    path.write_text(json.dumps(plain(dict(schema='rinkspace.current-native-ice-ordinary-paint-side-readings.v1',
        original_packet=inputs['packet'], actual_selected_source_qualification=inputs['qualification_contract'],
        rows=records, source_images_decoded_once_per_cached_RGB=True, camera_or_world_loaded=False,
        source_clock_preserved=True, fitting_observations_modified=False, accuracy_accepted=False)), indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(receipt=str(path),rows=[dict(row=r['row_index'], candidate_indices=[p['original_point_index'] for p in r['point_decisions'] if p['candidate_uv'] is not None]) for r in plain(records)])))


if __name__=='__main__':main()
