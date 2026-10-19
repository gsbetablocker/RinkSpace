"""Current q32 native FIT RGB/UV audit sheets, without projected geometry."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np

RUN = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_ce235f_wide_curve_budget_resume_20261012')
OUT = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/native_board_material_circle_RGB_review_20261012')
FAMILIES = [('B', 'additional_real_native_board_edges'),
            ('M', 'additional_real_native_board_material_points'),
            ('C', 'additional_real_native_board_circles')]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def caption(image, message, x, y, scale=.43):
    cv2.putText(image, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def mark(image, uv, origin=(0, 0), scale=1.):
    for i, value in enumerate(uv):
        x, y = np.rint((np.asarray(value)-origin)*scale).astype(int)
        # A gap at the pixel centre leaves its native colour visible.
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            cv2.line(image, (x+dx*4, y+dy*4), (x+dx*10, y+dy*10), (0, 0, 255), 1, cv2.LINE_AA)
        if len(uv) < 18:
            cv2.putText(image, str(i), (x+8, y-6), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 0, 255), 1, cv2.LINE_AA)


def main():
    OUT.mkdir(exist_ok=True)
    manifest_path = RUN / 'input_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    with np.load(RUN / 'complete_physical_camera_model_UNACCEPTED_5126c97aa85b9576.npz', allow_pickle=False) as z:
        aliases = dict(zip(map(int, z['pts']), map(int, z['exposure_representative_pts'])))
    cache, rows, overview_pages, point_pages = {}, [], [], []
    sources = {}
    for family, key in FAMILIES:
        contract = manifest[key]
        path = Path(contract['packet_path'])
        if sha(path) != contract['packet_sha256']:
            raise ValueError('Saved q32 source packet changed')
        packet = json.loads(path.read_text(encoding='utf8'))
        sources[family] = dict(path=str(path), sha256=contract['packet_sha256'])
        panels, point_panels = [], []
        for index, row in enumerate(packet['rows']):
            if row['split'] != 'fit':
                continue
            raw_path = Path(row.get('source_raw_frame', row.get('source_path')))
            expected = row.get('source_file_sha256', row.get('source_frame_sha256'))
            original_raw_reference = dict(path=str(raw_path), sha256=expected)
            if str(raw_path) not in cache:
                try:
                    if sha(raw_path) != expected:
                        raise ValueError('Source RGB changed: ' + str(raw_path))
                except PermissionError:
                    # Reuse the already decoded byte-equivalent actual frame,
                    # authenticated by the packet's original native BGR hash.
                    root = OUT.parent
                    alternatives = [root / 'source_pixel_review_additional/board_top_contact_material_semantics/original_farCONTACT_full_RGB_scan' / f'{aliases[int(row["pts"])]}_RAW_NATIVE.png',
                                    root / 'original_ice_FIT_RGB_review' / f'{aliases[int(row["pts"])]}_independently_decoded_RAW_NATIVE.png']
                    raw_path = next(p for p in alternatives if p.is_file())
                    expected_bgr = next(row[k] for k in ('source_native_BGR_sha256', 'source_native_bgr_pixels_sha256', 'decoded_source_bgr_sha256') if k in row)
                    decoded = cv2.imread(str(raw_path))
                    if hashlib.sha256(decoded.tobytes()).hexdigest() != expected_bgr:
                        raise ValueError('Reused actual frame native BGR differs')
                    expected = sha(raw_path)
                raw = cv2.imread(str(raw_path))
                if raw is None or raw.shape != (1080, 1920, 3):
                    raise ValueError('Native RGB source required')
                cache[str(raw_path)] = raw
            raw = cache[str(raw_path)]
            pts = aliases[int(row['pts'])]
            if row.get('actual_pts', pts) != pts:
                raise ValueError('Source exposure alias changed')
            uv = np.asarray(row['uv'], float)
            curve = row['curve']
            entity = curve.get('advertisement_number', curve.get('advertisement_id', 'course'))
            role = str(curve.get('edge', curve['kind']))
            if curve.get('kind') == 'shared_board_pattern_material_points':
                role = 'named shield corners/tip'
            if row.get('geometry_role', '').startswith('retained_internal'):
                role += ' INTERNAL graphic'
            tag = f'{family}{index:03d} {pts} Ad{entity} {role}'
            x0, y0 = np.maximum(0, np.floor(uv.min(0)-40)).astype(int)
            x1, y1 = np.minimum([1920, 1080], np.ceil(uv.max(0)+41)).astype(int)
            crop = raw[y0:y1, x0:x1]
            scale = min(1.8, 610/crop.shape[1], 225/crop.shape[0])
            resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            overlay = resized.copy()
            mark(overlay, uv, (x0, y0), scale)
            panel = np.full((270, 1240, 3), 255, np.uint8)
            caption(panel, tag, 7, 18, .52)
            caption(panel, 'native RGB left / current q32 UV right; point crops show exact pixels', 7, 36)
            panel[42:42+len(resized), 0:resized.shape[1]] = resized
            panel[42:42+len(overlay), 625:625+overlay.shape[1]] = overlay
            panels.append(panel)
            for point, value in enumerate(uv):
                cx, cy = np.rint(value).astype(int)
                xa, ya, xb, yb = max(0, cx-14), max(0, cy-14), min(1920, cx+15), min(1080, cy+15)
                original = cv2.resize(raw[ya:yb, xa:xb], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
                marked = original.copy()
                mark(marked, [value], (xa, ya), 4)
                cell = np.full((154, 244, 3), 255, np.uint8)
                caption(cell, f'{family}{index:03d} p{point} {pts}', 3, 14, .39)
                caption(cell, f'Ad{entity} {role[:22]}', 3, 30, .38)
                cell[36:36+len(original), 0:original.shape[1]] = original
                cell[36:36+len(marked), 124:124+marked.shape[1]] = marked
                point_panels.append(cell)
            rows.append(dict(family=family, packet_row=index, id=row['id'], actual_pts=pts,
                native_pts=row['pts'], uv=uv.tolist(), uv_float64_sha256=hashlib.sha256(np.asarray(uv,dtype='<f8').tobytes()).hexdigest(),
                source_rgb=dict(path=str(raw_path), sha256=expected), original_raw_reference=original_raw_reference, curve=curve,
                material_roles=row.get('material_roles'), geometry_role=row.get('geometry_role'),
                direct_user_provenance={k:v for k,v in row.items() if k in ('source_user_annotation','source_annotation_registration','source_annotation_attachment','direct_user_annotation','user_directly_annotated_pixels')},
                source_only_review_pending=True))
        for start in range(0, len(panels), 5):
            image = np.vstack(panels[start:start+5])
            page = OUT / f'{family}_rows_{start//5:02d}_RGB_and_current_UV.png'
            cv2.imwrite(str(page), image)
            overview_pages.append(dict(family=family,path=str(page),row_slice=[start,min(start+5,len(panels))]))
        for start in range(0, len(point_panels), 32):
            image = np.full((8*154, 4*244, 3), 255, np.uint8)
            for i, cell in enumerate(point_panels[start:start+32]):
                x, y = (i%4)*244, (i//4)*154
                image[y:y+154,x:x+244] = cell
            page = OUT / f'{family}_points_{start//32:02d}_RAW4x_and_current_UV.png'
            cv2.imwrite(str(page), image)
            point_pages.append(dict(family=family,path=str(page),active_point_slice=[start,min(start+32,len(point_panels))]))
    inputs = dict(schema='rinkspace.current-native-board-material-circle-RGB-source-review.v1',
        saved_input_manifest=dict(path=str(manifest_path),sha256=sha(manifest_path)), sources=sources,
        rows=rows, overview_pages=overview_pages, point_pages=point_pages,
        actual_source_images_reused=len(cache), observations_selected_from_current_FIT_only=True,
        source_pixels_model_sigma_role_or_split_not_changed=True, model_not_loaded_or_projected=True)
    (OUT / 'current_q32_native_board_material_circle_RGB_review_inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(output=str(OUT),rows=len(rows),source_RGB_frames=len(cache),overview_pages=len(overview_pages),point_pages=len(point_pages))))


if __name__ == '__main__':
    main()
