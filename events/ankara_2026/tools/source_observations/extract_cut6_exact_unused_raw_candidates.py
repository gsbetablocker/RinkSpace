# Project-owned event tool; source assets and all outputs stay in the original Data directory.
from pathlib import Path
import os, json, hashlib
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import av, cv2, numpy as np
cv2.setNumThreads(1)
O = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/cut6_unused_actual_source_append_v9_20261005')
O.mkdir(parents=True, exist_ok=True)
S = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/media/originals/oxXu1avhrvs.mp4')
P = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/user_reported_grid_priorities_20261005/additional_user_reported_ice_source_admitted_v8_spin_cut6.json')
C = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/body_v8/original_source_clock_rows.json')
A = Path('D:/CodexData/.codex/visualizations/2026/10/02/private-thread-00009/RinkSpace_tmp/z190_full_programme_depth_profile_low8_v1/z190_full_programme_camera_candidate.npz')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
parent = json.loads(P.read_text(encoding='utf-8-sig'))
clock = {r['full_pts']: r for r in json.loads(C.read_text(encoding='utf-8-sig'))}
with np.load(A, allow_pickle=False) as z:
    alias = {int(p): int(a) for p, a in zip(z['pts'], z['exposure_representative_pts'])}
excluded = {230369024,230381312,230384384,230395904,230400512,230415104,230422016,230442752,230499584,230500096,230500352,230500608,230500864,230501120,230504704,230504960,230505216,230505472,230505728,230506240,230506496,230506752,230507008,230507264,230507776,230514944}
selected = [230379776,230407168,230442240,230479360,230512384]
records, panels = [], []
with av.open(str(S)) as c:
    st = c.streams.video[0]
    st.thread_type = 'NONE'
    assert str(st.time_base) == '1/15360'
    for p in selected:
        assert alias[p] not in excluded and clock[p]['match_status'] == 'unique_pixel_match'
        c.seek(p, stream=st, backward=True, any_frame=False)
        for f in c.decode(st):
            if f.pts == p:
                im = f.to_ndarray(format='bgr24')
                break
            if f.pts is not None and f.pts > p:
                raise RuntimeError('Exact PTS absent')
        else:
            raise RuntimeError('Exact PTS absent')
        assert im.shape == (1080,1920,3)
        path = O / f'raw_native_{p}.png'
        assert cv2.imwrite(str(path), im)
        records.append(dict(native_pts=p, actual_pts=alias[p], official_pts=clock[p]['accepted_official_pts'], source_time_s=p/15360, raw_path=str(path), raw_file_sha256=sha(path), raw_bgr_sha256=hashlib.sha256(im.tobytes()).hexdigest(), source_clock_row=clock[p], candidate_only_not_FIT=True))
        panel = cv2.resize(im, (960,540), interpolation=cv2.INTER_AREA)
        cv2.putText(panel, f'native {p} actual {alias[p]} official {clock[p]["accepted_official_pts"]}', (10,25), cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1,cv2.LINE_AA)
        panels.append(panel)
        print(json.dumps({'raw':str(path),'actual':alias[p]}), flush=True)
cv2.imwrite(str(O/'CUT6_UNUSED_ACTUAL_RAW_CANDIDATE_CONTACT_SHEET.jpg'), np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:4]), np.hstack([panels[4],np.zeros_like(panels[0])])]), [cv2.IMWRITE_JPEG_QUALITY,92])
report = dict(schema='rinkspace.cut6-source-only-unused-actual-exposure-candidates.v1', status='RAW_CANDIDATES_PENDING_SOURCE_VISUAL_SELECTION', source_video_path=str(S), source_video_sha256=parent['source_video_sha256'], source_hash_authority=str(P), source_hash_reused_from_verified_parent=True, clock=dict(path=str(C),sha256=sha(C)), actual_alias_authority=parent['actual_alias_provenance'], parent_v8=dict(path=str(P),sha256=sha(P)), exclusion_authority=dict(path='D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/spin_source_long_baseline_20261005/actual50_jitter_source_diagnostic_20261005/cut6_existing_original225_additional_review_exposure_exclusions_METADATA_ONLY.json',sha256='96047a547dacd9fe9dee0044ecacae0da36f1b4c2e8d4c7de0f38415dd37840b'), excluded_actual_pts=sorted(excluded), records=records, model_or_solver_imported=False, source_UV_measured=False, new_video_generated=False)
(O/'CUT6_UNUSED_ACTUAL_RAW_CANDIDATES.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
