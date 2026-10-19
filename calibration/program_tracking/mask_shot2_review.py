"""Read current person exclusions only for selected official source frames."""
from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np
import torch
from rinkspace.vision_instances import Sam2SourceInstances, SourcePersonBoxes
from calibration.program_tracking.infer_current_person_exclusion_masks import DEFAULT_MODEL
from calibration.program_tracking.infer_continuation_person_exclusion import source_materials

DATA = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/shot2_review_20261014')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def run(indices=None, data=None):
    global DATA
    if data is not None:
        DATA = Path(data)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    manifest = read(DATA / 'frame_manifest.json')
    shot_index = int(manifest['shot_scope']['shot_index'])
    provider = Sam2SourceInstances(DEFAULT_MODEL, device='cuda', reread_rgb=False, contour_stride=16)
    detector = SourcePersonBoxes(ROOT.parent / 'FSpathing/models/yolo11n-pose.pt',
        dependency_paths=[ROOT.parent / 'FSpathing/.venv/Lib/site-packages'],
        runtime_directory=DATA / 'person_detector_runtime', device='cuda')
    tiles = []
    for row in manifest['frames']:
        index = int(row['index'])
        if indices is not None and index not in indices:
            continue
        raw = cv2.imread(row['image_path'])
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        if sha(row['image_path']) != row['png_sha256']:
            raise ValueError('Current source PNG changed')
        binding = dict(source_sha256=row['source']['source_video_sha256'], source_pts=row['native_pts'],
            time_base='1/12800', native_hw=[1080, 1920], shot_id=f'official50_shot{shot_index}_selected',
            decoded_native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
        boxes = detector.read(rgb, binding)
        options = [b for b in boxes if b['box_xyxy'][3] > 430 and
                   b['box_xyxy'][3] - b['box_xyxy'][1] > 100]
        if not options:
            raise ValueError(f'Current foreground person proposal missing: {index}')
        chosen = max(options, key=lambda b: b['box_xyxy'][3] - b['box_xyxy'][1])
        box = np.asarray(chosen['box_xyxy'], float)
        box[:2] = np.maximum(box[:2] - 14, [0, 0])
        box[2:] = np.minimum(box[2:] + 14, [1920, 1080])
        # Box-only SAM prompts can select a spinning skater's dress while
        # omitting visible skin and white boots even inside the box. Reuse the
        # same current-source detector's confident pose proposals solely as
        # positive segmentation prompts, never as measured body/world points.
        prediction = detector.model.predict(source=raw.copy(), device=detector.device.index or 0, verbose=False)[0]
        pose_prompts = []
        if prediction.keypoints is not None and prediction.keypoints.conf is not None:
            pboxes = prediction.boxes.xyxy.cpu().numpy()
            chosen_box = np.asarray(chosen['box_xyxy'], float)
            match = int(np.argmin(np.linalg.norm(pboxes - chosen_box, axis=1)))
            proposed_uv = prediction.keypoints.xy[match].cpu().numpy()
            proposed_conf = prediction.keypoints.conf[match].cpu().numpy()
            for uv, confidence in zip(proposed_uv, proposed_conf):
                if confidence >= .5 and 0 <= uv[0] < 1920 and 0 <= uv[1] < 1080:
                    pose_prompts.append(dict(uv=uv.tolist(), confidence=float(confidence)))
        prompt = dict(object_id=1, kind='occluder', origin='original_rgb_candidate',
            source_binding=binding, box_xyxy=box.tolist(), exclusion_authority=None)
        if pose_prompts:
            prompt.update(points_uv=[p['uv'] for p in pose_prompts],
                          point_labels=[1] * len(pose_prompts))
        result = provider.read(rgb, binding, prompts=[prompt], reset=True)
        candidate = result['occluder_region_candidate_mask'].detach().cpu().numpy()
        scope = np.zeros(candidate.shape, bool)
        x0, y0, x1, y1 = np.rint(box).astype(int)
        scope[y0:y1, x0:x1] = True
        # A detector box is a SAM prompt, not the visible person's extent.
        # Current native spinning/jumping sources can have hands, a raised
        # boot or a whole limb outside this proposal. Keep the SAM source
        # candidate there; the box limits only the local color augmentation.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
        if count <= 1:
            raise ValueError(f'Current foreground person mask empty: {index}')
        main = max(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA])
        candidate = labels == main
        dark, skin, *_ = source_materials(rgb)
        nearby = cv2.dilate(candidate.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
        material = (dark | skin) & scope & nearby
        mask = ((candidate | material).astype(np.uint8) * 255)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        path = DATA / 'masks_current' / f'{index:06d}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        row['mask_path'] = str(path)
        row['mask_sha256'] = sha(path)
        row.pop('mask_pending', None)
        row['source'].update(mask_path=str(path), mask_sha256=sha(path))
        if row.get('source_binding'):
            binding_path=Path(row['source_binding']['path'])
            if sha(binding_path)!=row['source_binding']['sha256']:
                raise ValueError('Current exact source binding changed before mask attachment')
            exact_source=read(binding_path)
            if exact_source['native_pts']!=row['native_pts'] or exact_source['png_sha256']!=row['png_sha256']:
                raise ValueError('Mask attachment is not bound to this current native source')
            exact_source.update(mask_path=str(path),mask_sha256=sha(path))
            write(binding_path,exact_source)
            row['source_binding']['sha256']=sha(binding_path)
        receipt = dict(schema='rinkspace.current-native-source-person-exclusion.v1', index=index,
            source_binding=binding, current_source_prompt=prompt, current_RGB_independently_encoded=True,
            prior_frame_memory_used=False, historical_masks_used=False, mask_path=str(path), mask_sha256=sha(path),
            original_png_sha256=row['png_sha256'], conservative_exclusion_candidate_not_ground_truth=True,
            detector_proposal_used_as_hard_mask_clip=False,
            current_detector_pose_segmentation_prompts=pose_prompts,
            detector_pose_points_are_source_measurements=False,
            checkpoint_path=str(DEFAULT_MODEL), checkpoint_sha256=provider.model_sha256,
            detector_proposal=chosen, human_acceptance=False)
        write(DATA / 'mask_provenance' / f'{index:06d}.json', receipt)
        overlay = raw.copy()
        overlay[mask > 0] = (overlay[mask > 0].astype(float) * .5 + np.array([40, 190, 150]) * .5).astype('uint8')
        cv2.imwrite(str(DATA / 'masks_current' / f'review_{index:06d}.png'), overlay)
        tile = cv2.resize(overlay, (480, 270))
        cv2.putText(tile, str(index), (6, 23), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 1)
        tiles.append(tile)
        write(DATA / 'frame_manifest.json', manifest)
        print(json.dumps(dict(index=index, mask_pixels=int((mask > 0).sum())), ensure_ascii=False), flush=True)
    tiles = []
    for row in manifest['frames']:
        path = DATA / 'masks_current' / f'review_{row["index"]:06d}.png'
        if path.is_file():
            tile = cv2.resize(cv2.imread(str(path)), (480, 270))
            cv2.putText(tile, str(row['index']), (6, 23), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 1)
            tiles.append(tile)
    sheet = np.zeros((270 * ((len(tiles) + 3) // 4), 1920, 3), np.uint8)
    for i, tile in enumerate(tiles):
        y, x = divmod(i, 4)
        sheet[y*270:(y+1)*270, x*480:(x+1)*480] = tile
    cv2.imwrite(str(DATA / 'masks_current/contact_sheet.png'), sheet)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--indices', type=int, nargs='*')
    parser.add_argument('--data', type=Path)
    args = parser.parse_args()
    run(args.indices, args.data)
