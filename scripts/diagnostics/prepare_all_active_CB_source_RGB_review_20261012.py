"""Prepare original RGB/current active C/B1/B2 pixel views, without a model."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import pickle
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SYS = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
OUT = SYS / 'same_manual_B2_contact_RGB_review_20261012/all_active_CB_frame_RGB_review'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def plain(x):
    return x.tolist() if isinstance(x, np.ndarray) else x.item() if isinstance(x, np.generic) else str(x)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = read(ROOT / 'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_ce235f_wide_curve_budget_20261012.json')
    opts = config['source_options']
    correction = read(opts['--original-ice-pixel-correction-source'])
    qualification = read(opts['--original-source-point-qualification-source'])
    original_ref = correction['original_absolute_packet']
    assert ref(original_ref['path']) == original_ref
    originals = pickle.loads(Path(original_ref['path']).read_bytes())['programme_absolute_rows']
    changes = {r['original_packet_row_index']: r for r in correction['records']}
    exclusions = {r['original_packet_row_index']: r['excluded_point_indices'] for r in qualification['records']}
    main4corr = read(SYS / 'same_manual_B2_contact_RGB_review_20261012/original_B2_10398_four_actual_RGB_centers_source_correction_20261012.json')
    rows = []
    for index, item in enumerate(originals):
        r = item['source_row']
        if item['collection_role'] != 'FIT' or r['id'] not in ('C', 'B1', 'B2'):
            continue
        uv = np.asarray(r['uv'], float).copy()
        change = changes.get(index)
        if change:
            uv[change['point_indices']] = change['corrected_uv']
        keep = [i for i in range(len(uv)) if i not in exclusions.get(index, [])]
        rawpath = SYS / f"original_ice_FIT_RGB_review/{item['group_actual_pts']}_independently_decoded_RAW_NATIVE.png"
        if not rawpath.is_file():
            raise FileNotFoundError(rawpath)
        rows.append(dict(family='original', packet_row=index, feature=r['id'], actual_pts=item['group_actual_pts'],
            uv=uv[keep], active_original_point_indices=keep, rawpath=str(rawpath), original_uv=np.asarray(r['uv'])[keep],
            correction_applied=bool(change), already_pending_main4_correction=main4corr['records'][0] if index == 10398 else None))
    icepath = Path(opts['--additional-ice-source'])
    ice = read(icepath)
    for index, r in enumerate(ice['rows']):
        feature = r.get('feature_id', r.get('curve', {}).get('feature_id'))
        if feature not in ('C','B1','B2') or r['split'] != 'fit':
            continue
        rows.append(dict(family='native_ice', packet_row=index, feature=feature, actual_pts=r['actual_pts'],
            uv=np.asarray(r['uv'], float), active_original_point_indices=list(range(len(r['uv']))),
            rawpath=r['source_raw_frame'], source_id=r['id'], measurement_role=r.get('measurement_role'),
            authority_kind='direct_user_source_trace' if r['id'].startswith('user_') else 'assistant_source_measurement',
            existing_normal=r.get('normal'), existing_source_support=r.get('normal_interval_px'),
            existing_source_location_support=r.get('source_location_support'), current_source_sigma=r.get('sigma_px')))
    # Views are grouped by the exact actual source exposure; no time sample is
    # omitted and no source row is selected by a residual/model comparison.
    frames = []
    for pts in sorted({r['actual_pts'] for r in rows}):
        group = [r for r in rows if r['actual_pts'] == pts]
        rawpaths = list(dict.fromkeys(r['rawpath'] for r in group))
        rawimages = [cv2.imread(p) for p in rawpaths]
        if any(a is None for a in rawimages):
            raise ValueError(rawpaths)
        # When there are two native caches they must be the same exposure's
        # actual RGB before sharing a canvas.
        if len(rawimages) > 1 and not all(np.array_equal(rawimages[0], a) for a in rawimages[1:]):
            raise ValueError('Two raw cache paths for one exposure differ: ' + str(pts))
        raw = rawimages[0]
        mark = raw.copy()
        colors = {'C': (0,0,255), 'B1': (0,170,0), 'B2': (255,0,255)}
        for r in group:
            for i, uv in zip(r['active_original_point_indices'], r['uv']):
                p = tuple(np.rint(uv).astype(int))
                cv2.drawMarker(mark,p,colors[r['feature']],cv2.MARKER_CROSS,8,1)
                if len(r['uv']) < 25:
                    cv2.putText(mark,f'{r["feature"]}:{i}',(p[0]+4,p[1]-4),cv2.FONT_HERSHEY_SIMPLEX,.35,colors[r['feature']],1)
        title = np.full((35, 3840, 3),255,np.uint8)
        cv2.putText(title,f'{pts}: actual native RGB left; current active UV right; C red/B1 green/B2 magenta',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),1)
        full = np.vstack([title,np.hstack([raw,mark])])
        fullpath = OUT / f'{pts}_FULL_NATIVE_RGB_AND_ALL_CURRENT_CB_UV.png'
        cv2.imwrite(str(fullpath),full)
        rowviews = []
        for r in group:
            uv = np.asarray(r['uv'])
            x0,y0 = np.maximum(0,np.floor(uv.min(0)-45)).astype(int)
            x1,y1 = np.minimum([1920,1080],np.ceil(uv.max(0)+46)).astype(int)
            cropped = np.hstack([raw[y0:y1,x0:x1],mark[y0:y1,x0:x1]])
            croppath = OUT / f'{pts}_{r["family"]}{r["packet_row"]}_{r["feature"]}_RAW_AND_CURRENT_UV.png'
            cv2.imwrite(str(croppath),cropped)
            r.update(source_rgb=ref(r['rawpath']), decoded_native_RGB_sha256=hashlib.sha256(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB).tobytes()).hexdigest(),
                     source_crop_bounds_xyxy=[int(x0),int(y0),int(x1),int(y1)], source_crop_view=ref(croppath))
            rowviews.append(ref(croppath))
        frames.append(dict(actual_pts=pts, view=ref(fullpath), row_views=rowviews,
            source_row_count=len(group), source_only_actual_view_pending=True))
    manifest = dict(schema='rinkspace.all-current-C-B1-B2-native-source-RGB-review-inputs.v1',
        original_packet=original_ref, original_correction=ref(opts['--original-ice-pixel-correction-source']),
        original_qualification=ref(opts['--original-source-point-qualification-source']),
        native_ice_packet=ref(icepath), source_video_sha256=ice['source_video_sha256'], time_base=ice['time_base'],
        rows=rows, frames=frames, all_existing_FIT_CB_rows_retained=True,
        original_source_pixels_and_roles_not_modified=True, frames_not_selected_by_model_or_residual=True,
        observation_qualification_is_not_solver_or_model_accuracy=True)
    manifestpath = OUT / 'all_active_CB_original_and_native_ice_source_RGB_review_inputs_20261012.json'
    manifestpath.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=plain)+'\n',encoding='utf-8')
    print(json.dumps(dict(input_manifest=ref(manifestpath),source_rows=len(rows),source_frames=len(frames),
        actual_pts=[f['actual_pts'] for f in frames]),ensure_ascii=False))


if __name__ == '__main__':
    main()
