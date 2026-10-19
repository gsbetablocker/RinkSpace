"""Local native-PTS camera rebinding for a presentation bridge failure.

Reuses the frozen saved per-exposure camera; does not fit a camera, alter the
source packet, or accept the historical camera/ledger as a validated model.
Only the three missing correspondence runs inside the declared local window
are replaced. Official native source pixels remain the rendering input.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation
from .demo_source import check_output_path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp')
DEFAULT_SOURCE = DATA / 'results/demo_last_shot_20261016/source_packet.npz'
DEFAULT_OUTPUT = DATA / 'results/demo_full_20261016/v3'
LOCAL_FIRST, LOCAL_LAST = 1456896, 1469440  # official 113.82..114.80s, inclusive
SHOT_FIRST, SHOT_LAST = 1425664, 1742592


def _sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def _ref(path):
    return dict(path=str(Path(path)), sha256=_sha(path))


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n', encoding='utf8')


def world_plane_homography(camera25):
    """Scene x/y (z=0) -> official native pixels, same no-Brown demo contract."""
    c = np.asarray(camera25, np.float64)
    if c.shape != (25,) or not np.isfinite(c).all():
        raise ValueError('25 finite camera values required')
    r = Rotation.from_rotvec(c[3:6]).as_matrix()
    k = np.array([[np.exp(c[6]), c[10], c[8]],
                  [0., np.exp(c[7]), c[9]], [0., 0., 1.]])
    h = k @ np.column_stack((r[:, 0], r[:, 1], -r @ c[:3]))
    if abs(np.linalg.det(h)) < 1e-12 or abs(h[2, 2]) < 1e-12:
        raise ValueError('Singular ice mapping')
    return h / h[2, 2]


def project_xy(h, xy):
    xy = np.asarray(xy, np.float64)
    q = np.column_stack((xy, np.ones(len(xy)))) @ np.asarray(h).T
    return q[:, :2] / q[:, 2:3]


def build_patch(source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT):
    source, output = Path(source), Path(output)
    target = check_output_path(output / 'camera_patch.npz')
    receipt_path = check_output_path(output / 'camera_patch.json')
    source_hash = _sha(source)
    original = _read(source.with_suffix('.json'))
    camera_path = Path(original['inputs']['camera']['path'])
    ledger_path = Path(original['inputs']['historical_alignment']['path'])
    for file, key in ((camera_path, 'camera'), (ledger_path, 'historical_alignment')):
        if _sha(file) != original['inputs'][key]['sha256']:
            raise ValueError('Frozen input identity changed: '+key)
    with np.load(source, allow_pickle=False) as z:
        packet = {k: z[k].copy() for k in z.files}
    with np.load(camera_path, allow_pickle=False) as z:
        legacy = {k: z[k].copy() for k in ('pts', 'camera25', 'exposure_representative_pts')}
    index = np.flatnonzero((packet['pts'] >= LOCAL_FIRST) & (packet['pts'] <= LOCAL_LAST))
    pts = packet['pts'][index]
    expected = np.arange(LOCAL_FIRST, LOCAL_LAST+256, 256)
    if not np.array_equal(pts, expected) or not np.all(packet['shot_index'][index] == 7):
        raise ValueError('Declared official native local shot-7 scope required')
    legacy_rows = {int(p): i for i, p in enumerate(legacy['pts'])}
    candidates = {}
    for line in ledger_path.read_text(encoding='utf8').splitlines():
        e = json.loads(line)
        p = e.get('best_official_pts_candidate')
        l = int(e['full_pts'])
        if (p is None or not LOCAL_FIRST <= int(p) <= LOCAL_LAST
                or int(e['shot_index']) != 7 or l not in legacy_rows):
            continue
        row = legacy_rows[l]
        score = (l != int(legacy['exposure_representative_pts'][row]),
                 float(e['best_roi_gray_mae']), l)
        if int(p) not in candidates or score < candidates[int(p)][0]:
            candidates[int(p)] = (score, row, e)
    cameras = packet['camera25'][index].copy()
    before = cameras.copy()
    changed = packet['bridge_mode'][index] != 0
    selected_legacy = np.empty(len(pts), np.int64)
    candidate_mae = np.empty(len(pts), np.float64)
    candidate_margin = np.empty(len(pts), np.float64)
    candidate_accepted = np.empty(len(pts), np.bool_)
    records = []
    for j, p in enumerate(pts):
        if int(p) not in candidates:
            raise ValueError('Missing per-exposure candidate at official PTS '+str(p))
        _, row, entry = candidates[int(p)]
        selected_legacy[j] = legacy['pts'][row]
        candidate_mae[j] = float(entry['best_roi_gray_mae'])
        candidate_margin[j] = float(entry['unique_margin_mae'])
        candidate_accepted[j] = entry.get('accepted_official_pts') == int(p)
        if changed[j]:
            if packet['bridge_mode'][index[j]] != 1 or candidate_accepted[j]:
                raise ValueError('Patch only the declared unaccepted/interpolated rows')
            cameras[j] = legacy['camera25'][row]
        elif not np.array_equal(cameras[j], legacy['camera25'][row]):
            raise ValueError('Direct anchor differs from selected frozen exposure')
        records.append(dict(native_pts=int(p), native_seconds=float(p)/12800,
            source_packet_index=int(index[j]), bridge_before=int(packet['bridge_mode'][index[j]]),
            changed=bool(changed[j]), legacy_candidate_pts=int(selected_legacy[j]),
            accepted_official_pts_in_historical_ledger=bool(candidate_accepted[j]),
            candidate_roi_gray_mae=float(candidate_mae[j]),
            next_candidate_margin_mae=float(candidate_margin[j])))
    hs = np.stack([world_plane_homography(c) for c in cameras])
    old_hs = np.stack([world_plane_homography(c) for c in before])
    # Check the exact public texture -> world -> native contract, not a second
    # custom projection convention. The first/last and internal direct anchors
    # remain byte-identical camera states.
    from .demo_outro import plane_homography
    b, wh = (-98.7, -50.1, 98.7, 50.1), (1536, 781)
    tex_world = np.array([[(b[2]-b[0])/(wh[0]-1), 0., b[0]],
                          [0., -(b[3]-b[1])/(wh[1]-1), b[3]], [0., 0., 1.]])
    errors = []
    for c, h in zip(cameras, hs):
        contract = h @ tex_world
        contract /= contract[2, 2]
        errors.append(float(np.max(np.abs(contract-plane_homography(c, b, wh)))))
    if max(errors) > 1e-8 or not np.array_equal(cameras[~changed], before[~changed]):
        raise AssertionError('Public plane contract or frozen anchors changed')
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, pts=pts, camera25=cameras, original_camera25=before,
        H_world_xy_to_native=hs, original_H_world_xy_to_native=old_hs,
        source_packet_index=index, legacy_candidate_pts=selected_legacy,
        bridge_before=packet['bridge_mode'][index], changed=changed,
        candidate_accepted=candidate_accepted, candidate_roi_gray_mae=candidate_mae,
        candidate_unique_margin_mae=candidate_margin,
        source_packet_sha256=np.asarray(source_hash), time_base=np.asarray('1/12800'),
        native_fps=np.asarray('50/1'), shot_index=np.full(len(pts), 7, np.int16))
    receipt = dict(schema='rinkspace.presentation-camera-local-rebind.v1',
        output=_ref(target), source_packet=_ref(source),
        frozen_camera=_ref(camera_path), historical_ledger=_ref(ledger_path),
        module=_ref(__file__), source_identity=original['source'],
        first_pts=LOCAL_FIRST, last_pts=LOCAL_LAST, native_frames=len(pts),
        changed_frames=int(changed.sum()), unchanged_direct_anchor_frames=int((~changed).sum()),
        direct_anchor_camera25_byte_exact=True,
        public_plane_contract_max_abs_error=max(errors),
        exact_clock='Official native PTS, 256 ticks/frame, time_base 1/12800',
        method='Replace only local linear camera25 bridge rows with their frozen per-exposure historical candidate states; direct anchors preserved',
        new_fitted_points=0, solver_run=False, smoothing_used=False,
        source_packet_unmodified=(_sha(source) == source_hash), body_xy_modified=False,
        candidate_status='Presentation-only candidates previously rejected by the old 1.3 gray-MAE rule; no historical acceptance flag changed',
        mapping_status='Official static ice print review required; numeric continuity alone is not validation',
        limitations=['Frozen camera and scene remain historically UNACCEPTED',
            'Brown terms ignored to match main demonstration plane warp',
            'Existing inferred XY inside this bridge is independently inaccurate; this packet does not repair body or trajectory',
            'No new metric-scale or full-shot model accuracy acceptance'],
        records=records)
    _write(receipt_path, receipt)
    return receipt


def apply_patch(packet, patch_path=DEFAULT_OUTPUT / 'camera_patch.npz'):
    """Change only caller-owned in-memory camera rows, never disk artifacts."""
    with np.load(patch_path, allow_pickle=False) as z:
        patch = {k: z[k].copy() for k in z.files}
    idx = patch['source_packet_index']
    if not np.array_equal(packet['pts'][idx], patch['pts']):
        raise ValueError('Patch/source native PTS mismatch')
    if not np.array_equal(packet['camera25'][idx], patch['original_camera25']):
        raise ValueError('Patch requires exact frozen original local cameras')
    packet['camera25'][idx] = patch['camera25']
    return patch


def review_shot(source=DEFAULT_SOURCE, patch_path=DEFAULT_OUTPUT/'camera_patch.npz',
                output=DEFAULT_OUTPUT):
    """Continuous complete shot-7 diagnostic using the same shared scene.

    This is a presentation-mapping diagnostic, not an original-solver source
    point validation export. All modeled markings, grid, ad bounds/IDs remain
    enabled. Official native source exposure sequence is checked frame by frame.
    """
    import cv2
    from .render_demo_last_shot import decode
    from .render_result import SceneRenderer, _ffmpeg_executable, _available_encoder
    from .body_tracking.pelvis_proxy_programme import PlanView
    from .demo_particle_pass import _encoder_quality
    output = Path(output)
    with np.load(source, allow_pickle=False) as z:
        packet = {k: z[k].copy() for k in z.files}
    patch = apply_patch(packet, patch_path)
    scene = _read(str(packet['scene_path']))
    renderer = SceneRenderer(scene, draw_grid=True, draw_advertisements=True)
    plan = PlanView(scene, 0.)
    right = plan.base.copy()
    right[510:] = 248
    for j, line in enumerate(('Presentation mapping diagnostic',
            'Same frozen model; all modeled paths', 'Official native50 source, shot 7',
            'Local candidate rebind 113.84..114.78s',
            'No new fit; no body/XY update', 'No source-point validation claim')):
        cv2.putText(right, line, (18, 550+j*53), 0, .68, (30, 35, 40), 2, cv2.LINE_AA)
    pts_lookup = {int(p): i for i, p in enumerate(packet['pts'])}
    changed_lookup = {int(p): bool(c) for p, c in zip(patch['pts'], patch['changed'])}
    target = check_output_path(output/'camera_shot7_review.mp4')
    ff = _ffmpeg_executable(); enc, name = _available_encoder(ff)
    enc = _encoder_quality(enc, name)
    command = [ff, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
        '-pixel_format', 'bgr24', '-video_size', '1720x720', '-framerate', '50',
        '-i', 'pipe:0', *enc, '-pix_fmt', 'yuv420p', '-an', '-video_track_timescale',
        '12800', '-movflags', '+faststart', str(target)]
    _write(output/'camera_review_command.json', command)
    count=0; sequence=hashlib.sha256(); snapshots={1456896,1457920,1461760,1465600,1469440}
    with (output/'camera_review_encode.log').open('wb') as log:
        proc=subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log,
                              stdout=subprocess.DEVNULL, creationflags=0x08000000)
        try:
            for p, native in decode(str(packet['source_path']), SHOT_FIRST, SHOT_LAST):
                i=pts_lookup[p]; c=packet['camera25'][i].copy(); c[11:]=0.
                title=('SHOT 7 | source %.2fs PTS %d | %s' %
                    (p/12800., p, 'local candidate camera' if changed_lookup.get(p) else 'frozen direct/bridge camera'))
                left, _ = renderer.draw(native, c, title=title)
                h=world_plane_homography(c)
                for f in scene['ice_features']:
                    if f['kind']=='circle':
                        uv=project_xy(h, [f['center_xyz'][:2]])[0]
                        if 0<=uv[0]<1920 and 0<=uv[1]<1080:
                            cv2.putText(left, f['id'], tuple(np.rint(uv).astype(int)),
                                        0, .55, (20, 20, 230), 2, cv2.LINE_AA)
                joined=np.hstack((left,right))
                frame=cv2.resize(joined,(1720,720),interpolation=cv2.INTER_AREA)
                proc.stdin.write(frame.tobytes()); count+=1
                sequence.update(np.asarray([p],np.int64).tobytes())
                if p in snapshots:cv2.imwrite(str(output/('camera_review_%d.jpg'%p)),frame)
                if count%100==0: print('camera review',count,flush=True)
        finally:
            proc.stdin.close(); code=proc.wait()
        if code:raise RuntimeError('Review encoder failed')
    expected=(SHOT_LAST-SHOT_FIRST)//256+1
    if count!=expected:raise AssertionError('Full-shot native coverage changed')
    result=dict(video=_ref(target), source=str(packet['source_path']),
        first_pts=SHOT_FIRST,last_pts=SHOT_LAST,frames=count,fps=50,time_base='1/12800',
        source_pts_sequence_sha256=sequence.hexdigest(),
        all_model_markings_grid_advertisement_bounds_and_ids=True,
        same_shared_scene_both_views=True, patch=_ref(patch_path),
        diagnostic_not_solver_source_point_validation=True,
        source_model_and_body_unmodified=True)
    _write(output/'camera_review.json',result)
    return result




def build_trajectory_patch(baseline=DATA/'results/demo_full_20261016/v2/trajectory_packet.npz',
                           camera_patch=DEFAULT_OUTPUT/'camera_patch.npz',
                           source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT):
    """Rebind the original saved G.xy from the camera's exact candidate exposure.

    G is the historical body world vertical-projection reference, not a newly
    measured blade/contact. Existing spin and jump revisions must be absent
    from every replaced row. No trajectory, mask, or model file is overwritten.
    """
    baseline,camera_patch,source,output=map(Path,(baseline,camera_patch,source,output))
    baseline_hash=_sha(baseline)
    report=_read(source.with_suffix('.json'))
    body_path=Path(report['inputs']['body']['path'])
    if _sha(body_path)!=report['inputs']['body']['sha256']:
        raise ValueError('Frozen original body identity changed')
    with np.load(baseline,allow_pickle=False) as z:
        traj={k:z[k].copy() for k in z.files}
    with np.load(camera_patch,allow_pickle=False) as z:
        patch={k:z[k].copy() for k in z.files}
    with np.load(body_path,allow_pickle=False) as z:
        body={k:z[k].copy() for k in ('native_pts','native_actual_row','actual_pts',
            'G_world_m','G_native_uv','valid','inferred','metres_per_scene_unit')}
    idx=patch['source_packet_index']; changed=patch['changed']
    if not np.array_equal(traj['pts'][idx],patch['pts']):
        raise ValueError('Trajectory/camera native PTS mismatch')
    if (np.any(traj['spin_id'][idx][changed]!=0)
            or np.any(traj['correction_weight'][idx][changed]!=0)
            or np.any(traj['presentation_linear_segment'][idx][changed])):
        raise ValueError('Original exposure rebinding overlaps protected spin/jump revision')
    if float(traj['unit'])!=float(body['metres_per_scene_unit']):
        raise ValueError('Body/trajectory conditional unit mismatch')
    native_rows={int(p):int(i) for p,i in zip(body['native_pts'],body['native_actual_row'])}
    actual_rows=np.asarray([native_rows[int(p)] for p in patch['legacy_candidate_pts']],np.int64)
    original=traj['xy_m'][idx].copy()
    xy=original.copy(); xy[changed]=body['G_world_m'][actual_rows[changed],:2]
    if not np.array_equal(original[~changed],body['G_world_m'][actual_rows[~changed],:2]):
        raise ValueError('Direct trajectory anchors differ from frozen original G')
    if not np.isfinite(xy).all() or not np.all(body['valid'][actual_rows]):
        raise ValueError('Frozen candidate G must remain finite and valid')
    uv=np.asarray([project_xy(h,[x/float(traj['unit'])])[0]
        for h,x in zip(patch['H_world_xy_to_native'],xy)])
    original_uv=body['G_native_uv'][actual_rows]
    difference=np.linalg.norm(uv-original_uv,axis=1)
    target=check_output_path(output/'trajectory_patch.npz')
    np.savez_compressed(target,pts=patch['pts'],xy_m=xy,original_xy_m=original,
        source_packet_index=idx,legacy_candidate_pts=patch['legacy_candidate_pts'],
        body_actual_row=actual_rows,body_actual_pts=body['actual_pts'][actual_rows],
        original_G_native_uv=original_uv,patched_G_native_uv=uv,changed=changed,
        bridge_before=patch['bridge_before'],candidate_accepted=patch['candidate_accepted'],
        original_body_valid=body['valid'][actual_rows],original_body_inferred=body['inferred'][actual_rows],
        baseline_sha256=np.asarray(baseline_hash),camera_patch_sha256=np.asarray(_sha(camera_patch)),
        time_base=np.asarray('1/12800'),unit=traj['unit'])
    receipt=dict(schema='rinkspace.presentation-trajectory-local-rebind.v1',
        output=_ref(target),baseline=_ref(baseline),camera_patch=_ref(camera_patch),
        source_packet=_ref(source),frozen_body=_ref(body_path),module=_ref(__file__),
        first_pts=int(patch['pts'][0]),last_pts=int(patch['pts'][-1]),
        native_frames=len(idx),changed_frames=int(changed.sum()),
        direct_anchor_xy_byte_exact=True,source_files_unmodified=(_sha(baseline)==baseline_hash),
        body_fit_run=False,new_contact_points=False,xy_semantics='Frozen original G_world_m.xy body vertical projection, not blade contact',
        method='Exact same legacy candidate exposure as camera_patch, native_actual_row -> original G_world_m.xy',
        spin_and_jump_revision_overlap=False,
        original_saved_G_native_uv_reproduction_px=dict(max=float(difference.max()),median=float(np.median(difference))),
        residual_explanation='Same original reference point reproduced; tiny residual is expected because demo ice H ignores Brown terms',
        candidate_status='Presentation only, retains historical unaccepted correspondence status',
        limitations=['Original body G is not an independently verified blade contact or metre scale',
            'Only local missing source correspondences restored; no body/camera/model fit changed'])
    _write(output/'trajectory_patch.json',receipt)
    return receipt


def apply_trajectory_patch(xy_m,pts,patch_path=DEFAULT_OUTPUT/'trajectory_patch.npz'):
    """Replace caller-owned display XY only after exact baseline and PTS checks."""
    with np.load(patch_path,allow_pickle=False) as z:
        patch={k:z[k].copy() for k in z.files}
    idx=patch['source_packet_index']
    if not np.array_equal(np.asarray(pts)[idx],patch['pts']):
        raise ValueError('Trajectory patch native PTS mismatch')
    if not np.array_equal(np.asarray(xy_m)[idx],patch['original_xy_m']):
        raise ValueError('Trajectory patch requires exact v2 baseline local XY')
    xy_m[idx]=patch['xy_m']
    return patch


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--review',action='store_true')
    args=parser.parse_args()
    report=(review_shot(args.source,args.output/'camera_patch.npz',args.output)
            if args.review else build_patch(args.source,args.output))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
