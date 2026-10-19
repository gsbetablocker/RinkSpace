"""Render selected real frames with the complete shared model and mount state.

These timestamped pictures are diagnostic markers, not a continuous-shot
acceptance. The source and plan use the same full rink/ad geometry.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT/'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np
from rinkspace.storage import AssetStore
from z190_physical_camera import PhysicalCameraModel
from z190_continuous_source_bundle import HARD_SCENE_SHA, SOURCE_SHA, SCENE_UNIT_M
from render_result import SceneRenderer, PersonMasks, DEFAULT_MASKS, DEFAULT_PEOPLE, decode_selected
from three_frame_refit.program_tracking.body_tracking.pelvis_proxy_programme import PlanView, source_time


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def render(camera, scene_path, source_path, pts, output, *, expected_scene_sha256=HARD_SCENE_SHA):
    storage = AssetStore(ROOT)
    camera, scene_path = map(storage.resolve, (camera, scene_path))
    source_path = storage.source(source_path)
    output = Path(output).resolve()
    if not output.is_relative_to(Path('D:/CodexData').resolve()):
        raise ValueError('Diagnostic pictures must stay outside OneDrive')
    output.mkdir(parents=True, exist_ok=False)
    if len(expected_scene_sha256)!=64 or digest(scene_path) != expected_scene_sha256:
        raise ValueError('Review scene differs from its explicit frozen identity')
    if source_path != storage.source(ROOT/'data/input/oxXu1avhrvs.mp4'):
        raise ValueError('Use the existing source-bound programme video')
    model, native_pts, aliases, views = PhysicalCameraModel.load_npz(camera)
    with np.load(camera, allow_pickle=False) as z:
        metadata = json.loads(str(z['metadata_json']))
    if metadata['source_sha256'] != SOURCE_SHA:
        raise ValueError('Wrong source camera')
    if any(p not in aliases for p in pts):
        raise ValueError('Requested source frame absent from candidate')
    scene = json.loads(Path(scene_path).read_text(encoding='utf8'))
    renderer, plan = SceneRenderer(scene), PlanView(scene, 0.)
    if renderer.path_counts != plan.path_counts:
        raise ValueError('Source/plan model inventories differ')
    masks = PersonMasks(DEFAULT_MASKS, DEFAULT_PEOPLE)
    records = []
    names = {0: '长边主机位', 1: '长边副机位', 2: '低位端部机位'}
    for p, native in decode_selected(Path(source_path), sorted(set(pts))):
        representative = aliases[p]
        view = views[representative]
        cal = model.calibrations[view.installation_id]
        cam = model.camera25(view)
        raw_path=output/f'{p}_source_whole.jpg'
        if not cv2.imwrite(str(raw_path),native,[cv2.IMWRITE_JPEG_QUALITY,96]):raise OSError(raw_path)
        try:
            mask, mask_status = masks.get(p, require_sam2=True)
        except ValueError:
            mask, mask_status = None, 'no confirmed source mask'
        left, metrics = renderer.draw(native, cam, mask)
        right = plan.base.copy()
        right[510:] = 248
        pivot = np.asarray(cal.pivot_xyz)
        pv, cc = plan.mapxy(np.vstack((pivot[:2], cam[:2])))
        cv2.drawMarker(right, tuple(pv), (30, 30, 230), cv2.MARKER_CROSS, 13, 2)
        cv2.circle(right, tuple(cc), 4, (225, 90, 20), 1, cv2.LINE_AA)
        slot = int(view.installation_id.split('_')[3])
        right = plan.text(right, (24, 535), names[slot]+'：固定云台＋共享非零偏移')
        right = plan.text(right, (24, 579), '红十字：固定云台；蓝圈：当前光心', small=True)
        right = plan.text(right, (24, 621), '平面图采用实际尺度，二者可能接近重合', small=True)
        right = plan.text(right, (24, 669), f'云台高度：{pivot[2]*SCENE_UNIT_M:.3f} m', small=True)
        right = plan.text(right, (24, 710), f'当前光心高度：{cam[2]*SCENE_UNIT_M:.3f} m', small=True)
        lever = np.asarray(cal.lever_xyz)*SCENE_UNIT_M*100.
        right = plan.text(right, (24, 751), f'相机轴偏移 右／上／前（cm）', small=True)
        right = plan.text(right, (24, 790), f'{lever[0]:.1f} ／ {-lever[1]:.1f} ／ {lever[2]:.1f}', small=True)
        right = plan.text(right, (24, 842), '高度相对冰面；摄影师地面高差未量', small=True)
        right = plan.text(right, (24, 889), '条件相机候选；尚未完成连续镜头验收', color=(160, 50, 40), small=True)
        canvas = np.full((1180, 2580, 3), 22, np.uint8)
        canvas[80:1160, :1920], canvas[80:1160, 1920:] = left, right
        canvas = plan.text(canvas, (18, 10), f'Z190 非零安装候选（未验收） | 原片 {source_time(p)} | PTS {p}', color=(245, 245, 245))
        canvas = plan.text(canvas, (18, 45), f'实际曝光 {representative}；全部冰线、网格与已建模广告边界持续显示', color=(220, 220, 220), small=True)
        target = output/f'{p}_nonzero_mount_all_model.jpg'
        cv2.imwrite(str(target), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
        records.append(dict(pts=p, source_time=source_time(p), representative_pts=representative,
            source_pixels_sha256=hashlib.sha256(native.tobytes()).hexdigest(), image=str(target),source_whole=str(raw_path),
            mask_status=mask_status, render_metrics=metrics, pivot_xyz_scene=pivot.tolist(),
            actual_optical_centre_scene=cam[:3].tolist()))
    report = dict(camera=dict(path=str(Path(camera).resolve()), sha256=digest(camera)),
        scene=dict(path=str(Path(scene_path).resolve()), sha256=expected_scene_sha256,
            current_hard_scene=expected_scene_sha256==HARD_SCENE_SHA,accuracy_accepted=False),
        source_path=str(Path(source_path).resolve()), source_sha256=SOURCE_SHA,
        source_model_inventory=renderer.path_counts, plan_model_inventory=plan.path_counts,
        advertisement_ids=plan.advertisement_ids, images=records,
        complete_shot_acceptance=False, video_generated=False)
    (output/'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(images=len(records), output=str(output))), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('camera', 'scene', 'source', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--pts', type=int, nargs='+', required=True)
    parser.add_argument('--expected-scene-sha256',default=HARD_SCENE_SHA,
        help='Explicit frozen diagnostic scene identity; default is the current hard scene')
    args = parser.parse_args()
    render(args.camera, args.scene, args.source, args.pts, args.output,expected_scene_sha256=args.expected_scene_sha256)
