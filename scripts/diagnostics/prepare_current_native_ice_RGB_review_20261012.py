"""Current qualified native ice RGB review views; no world/state/solver load."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import cv2
import numpy as np
from calibration.solver_runtime.source_point_qualification import (
    _direct_user_annotation, file_sha256, load_source_point_qualification, original_uv_sha256)
from calibration.solver_runtime.truth_source_scanline_revision import original_qualified_packet

SYS = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
RUN = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_ce235f_wide_curve_budget_resume_20261012')
OUT = SYS / 'source_pixel_review_additional/native_ice_current_remaining_actual_RGB_review_20261012'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def ref(path):return dict(path=str(Path(path).resolve()), sha256=file_sha256(path))


def main():
    config_path = ROOT / 'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_ce235f_wide_curve_budget_20261012.json'
    config = read(config_path)
    manifest_path = RUN / 'input_manifest.json'
    manifest = read(manifest_path)['additional_real_native_ice_curves']
    packet_path = Path(config['source_options']['--additional-ice-source'])
    packet = read(packet_path)
    if (Path(manifest['packet_path']).resolve() != packet_path.resolve()
            or manifest['packet_sha256'] != file_sha256(packet_path)):
        raise ValueError('Actual q32 selected native source differs from current configuration')
    qualification = load_source_point_qualification(packet_path, original_qualified_packet(packet_path, packet),
        source_family='additional_native_ice_curves')
    selected_ids = set(manifest['new_fit_observation_ids'])
    OUT.mkdir(parents=True, exist_ok=True)
    images, rows = {}, []
    for index, row in enumerate(packet['rows']):
        if row['id'] not in selected_ids or row['split'] != 'fit' or row.get('feature_id') in ('C', 'B1', 'B2'):
            continue
        active = np.flatnonzero(qualification.point_mask(row)).tolist()
        if not active:continue
        path = Path(row.get('source_raw_frame', row.get('source_path')))
        digest = row.get('source_file_sha256', row.get('source_frame_sha256'))
        if path not in images:
            if file_sha256(path) != digest:raise ValueError('Original source RGB file changed')
            bgr = cv2.imread(str(path))
            if bgr is None or bgr.shape != (1080, 1920, 3):raise ValueError('Expected native source frame')
            images[path] = bgr
        raw = images[path]
        pts = int(row.get('actual_pts', row['pts']))
        rgb_digest = hashlib.sha256(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
        expected = row.get('source_native_RGB_sha256')
        if expected is not None and expected != rgb_digest:raise ValueError('Bound native RGB pixels differ')
        uv = np.asarray(row['uv'], float)
        protected = _direct_user_annotation(row)
        operator = row.get('curve_observation_kind', 'normal_curve_pixels')
        coordinate_context = operator in ('horizontal_extent', 'clip_extent')
        mark = raw.copy()
        if coordinate_context:
            for i in active:
                x = int(round(uv[i, 0]))
                cv2.line(mark, (x, 300), (x, 950), (0, 200, 255), 1)
        else:
            for i in active:
                point = tuple(np.rint(uv[i]).astype(int))
                cv2.drawMarker(mark, point, (0, 200, 255) if protected else (0, 220, 50), cv2.MARKER_CROSS, 7, 1)
                if len(active) < 25:cv2.putText(mark, str(i), (point[0]+4, point[1]-4), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 220), 1)
        if coordinate_context:
            box = np.array([0, 300, 1200, 950])
        else:
            box = np.r_[np.maximum(0, np.floor(uv[active].min(0)-60)),
                np.minimum([1920, 1080], np.ceil(uv[active].max(0)+61))].astype(int)
        x0, y0, x1, y1 = box
        view = OUT / f'{pts}_row{index:03d}_RAW_CURRENT_UV.png'
        cv2.imwrite(str(view), np.hstack([raw[y0:y1, x0:x1], mark[y0:y1, x0:x1]]))
        pages = []
        # Complete human strokes retain their exact original trace, even in
        # occluded regions. They are reviewed in context and never called RGB
        # measured centres or silently retired by an automatic profile rule.
        if not protected and not coordinate_context:
            for start in range(0, len(active), 20):
                ids = active[start:start+20]
                canvas = np.full((130*((len(ids)+3)//4), 280*4, 3), 250, np.uint8)
                for tile_index, i in enumerate(ids):
                    x, y = np.rint(uv[i]).astype(int)
                    a, b = max(0, min(1780, x-70)), max(0, min(980, y-50))
                    left = raw[b:b+100, a:a+140]
                    right = mark[b:b+100, a:a+140]
                    tile = np.full((130, 280, 3), 250, np.uint8)
                    tile[:100] = np.hstack([left, right])
                    cv2.putText(tile, f'row{index} p{i} [{uv[i,0]:.2f},{uv[i,1]:.2f}]', (3, 119),
                        cv2.FONT_HERSHEY_SIMPLEX, .36, (10, 10, 10), 1)
                    col, r = tile_index%4, tile_index//4
                    canvas[r*130:(r+1)*130, col*280:(col+1)*280] = tile
                page = OUT / f'{pts}_row{index:03d}_points_{start:03d}.png'
                cv2.imwrite(str(page), canvas);pages.append(ref(page))
        rows.append(dict(row_index=index, id=row['id'], actual_pts=pts, source_pts=int(row['pts']),
            curve=row['curve'], physical_definition=row.get('geometry_role', row.get('role', row.get('measurement_role'))),
            observation_operator=operator, source_coordinates_are_context_not_UV_measurements=coordinate_context,
            active_original_point_indices=active, previously_qualified_point_indices=np.flatnonzero(~qualification.point_mask(row)).tolist(),
            current_uv=row['uv'], current_uv_float64_sha256=original_uv_sha256(row), original_sigma=row['sigma_px'],
            direct_user_stroke_protected=protected, raw_source=dict(path=str(path.resolve()), sha256=digest),
            decoded_native_RGB_sha256=rgb_digest, view=ref(view), source_crop_bounds_xyxy=box.tolist(), point_pages=pages))
    output = dict(schema='rinkspace.current-native-ice-source-RGB-review-inputs.v1',
        packet=ref(packet_path), current_config=ref(config_path), q32_input_manifest=ref(manifest_path),
        actual_selected_native_packet_matches_q32=True, qualification_contract=qualification.contract,
        source_video_sha256=packet['source_video_sha256'], time_base=packet['time_base'], rows=rows,
        user_strokes_or_extent_context_never_reinterpreted_as_RGB_UV=True, observations_modified=False,
        model_geometry_projection_history_or_solver_loaded=False, source_review_pending=True)
    path = OUT / 'current_active_remaining_native_ice_RGB_review_inputs.json'
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(input=str(path), row_views=[dict(row=r['row_index'], pts=r['actual_pts'], pages=len(r['point_pages'])) for r in rows])))


if __name__ == '__main__':main()
