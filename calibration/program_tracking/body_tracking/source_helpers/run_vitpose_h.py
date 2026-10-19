"""Run the existing H checkpoint on frozen boxes; no training or smoothing.

The shared input is original RGB PNG + original PTS + xywh detector boxes.
This UDP input path is distinct from the historical GVHMR crop pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
         'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
         'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
         'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
FLIP = [NAMES.index('right_' + name[5:]) if name.startswith('left_') else
        NAMES.index('left_' + name[6:]) if name.startswith('right_') else i
        for i, name in enumerate(NAMES)]
H_SHA256 = '50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc'


def udp_crop(rgb, box):
    """Map original pixels to 192x256 UDP endpoint coordinates, preserving aspect."""
    x, y, width, height = box
    if not np.isfinite(box).all() or width <= 0 or height <= 0:
        raise ValueError('xywh box must be finite with positive width/height')
    center = np.array([x + width / 2, y + height / 2], dtype=np.float32)
    height = max(height, width * 256 / 192) * 1.25
    size = np.array([height * 192 / 256, height], dtype=np.float32)
    gain = np.array([191, 255], dtype=np.float32) / size
    matrix = np.array([[gain[0], 0, (size[0] / 2 - center[0]) * gain[0]],
                       [0, gain[1], (size[1] / 2 - center[1]) * gain[1]]], dtype=np.float32)
    cropped = cv2.warpAffine(rgb, matrix, (192, 256), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    normalized = (cropped.astype(np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    tensor = torch.from_numpy(normalized.astype(np.float32)).permute(2, 0, 1)
    return tensor, center, size / 200, matrix


def load_h(upstream, checkpoint):
    with checkpoint.open('rb') as handle:
        actual = hashlib.file_digest(handle, 'sha256').hexdigest()
    if actual != H_SHA256:
        raise ValueError(f'Unexpected H checkpoint SHA256: {actual}')
    # Import only the isolated pose builder; do not import the old preproc package
    # (its __init__ also imports unrelated HMR/SLAM modules).
    sys.path.insert(0, str(upstream / 'hmr4d/utils/preproc/vitpose_pytorch/src'))
    from vitpose_infer.model_builder import build_model
    pose = build_model('ViTPose_huge_coco_256x192')
    pose.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict'], strict=True)
    spec = importlib.util.spec_from_file_location('h_udp_decoder', upstream / 'hmr4d/utils/kpts/kp2d_utils.py')
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    return pose.cuda().eval(), decoder.keypoints_from_heatmaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.frames.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not rows or len({r['frame_id'] for r in rows}) != len(rows):
        raise ValueError('Nonempty unique frame IDs required')
    args.output.mkdir(parents=True, exist_ok=False)
    wall_start = time.perf_counter()
    model, decode = load_h(args.upstream, args.checkpoint)
    with torch.inference_mode():
        warmup = torch.zeros((2, 3, 256, 192), device='cuda')
        for _ in range(3):
            model(warmup)
        torch.cuda.synchronize()
        del warmup
        torch.cuda.reset_peak_memory_stats()
        forward_seconds = 0.0
        pipeline_start = time.perf_counter()
        arrays = []
        with (args.output / 'predictions.jsonl').open('w', encoding='utf-8') as handle:
            for row in rows:
                out = {key: row[key] for key in ('frame_id', 'source_pts', 'time_base', 'bbox_xywh')}
                out.update(target_status=row.get('target_status', 'unreviewed'), track_id=row.get('track_id'))
                if row['bbox_xywh'] is None:
                    out.update(status='target_unknown' if row.get('target_status') == 'unknown' else 'missing_common_box', keypoints={name: {'xy': None} for name in NAMES})
                    arrays.append(np.full((17, 3), np.nan, dtype=np.float32))
                else:
                    bgr = cv2.imread(str(row['image_path']))
                    if bgr is None:
                        raise FileNotFoundError(row['image_path'])
                    inputs, center, scale, affine = udp_crop(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), row['bbox_xywh'])
                    inputs = inputs[None].cuda()
                    doubled = torch.cat([inputs, inputs.flip(3)])
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    heatmaps = model(doubled)
                    merged = (heatmaps[:1] + heatmaps[1:, FLIP].flip(3)) * 0.5
                    torch.cuda.synchronize()
                    forward_seconds += time.perf_counter() - start
                    xy, scores = decode(merged.cpu().numpy(), center[None], scale[None], use_udp=True)
                    raw = np.concatenate([xy[0], scores[0]], axis=-1)
                    arrays.append(raw)
                    out.update(status='automatic_prediction_accuracy_unknown',
                               original_to_input_affine=affine.tolist(), input_hw=[256, 192],
                               center=center.tolist(), scale_200=scale.tolist(),
                               keypoints={name: {'xy': raw[i, :2].tolist() if scores[0, i, 0] > 0 else None,
                                                 'score': float(scores[0, i, 0])}
                                          for i, name in enumerate(NAMES)})
                handle.write(json.dumps(out, allow_nan=False) + '\n')
        np.savez_compressed(args.output / 'raw_coco17.npz', predictions=np.stack(arrays), frame_ids=[r['frame_id'] for r in rows], names=NAMES)
        torch.cuda.synchronize()
        pipeline_seconds = time.perf_counter() - pipeline_start
    file_hashes = {}
    for relative in ['hmr4d/utils/preproc/vitpose_pytorch/src/vitpose_infer/model_builder.py', 'hmr4d/utils/kpts/kp2d_utils.py']:
        file_hashes[relative] = hashlib.sha256((args.upstream / relative).read_bytes()).hexdigest()
    report = dict(model='existing_vitpose_h', checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=H_SHA256,
                  python=sys.executable, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  input_hw=[256, 192], precision='float32', batch=1, flip_test=True, effective_forward_batch=2,
                  warmup_calls=3, cuda_sync=True, cpu_fallback=False, frame_count=len(rows),
                  pose_forward_seconds=forward_seconds,
                  pose_forward_timing_scope='v2: model + flip unmapping + average, CUDA synchronized; v1 stopped before flip average',
                  target_unknown_frames=sum(r.get('target_status') == 'unknown' for r in rows),
                  pose_forward_ms_per_detected_frame=forward_seconds * 1000 / max(1, sum(r['bbox_xywh'] is not None for r in rows)),
                  png_read_crop_pose_decode_export_seconds=pipeline_seconds,
                  load_warmup_pipeline_seconds=time.perf_counter() - wall_start,
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                  upstream_files_sha256=file_hashes,
                  frames_sha256=hashlib.sha256(args.frames.read_bytes()).hexdigest(),
                  semantics_source='https://raw.githubusercontent.com/ViTAE-Transformer/ViTPose/main/configs/_base_/datasets/coco.py',
                  transform='RGB original -> aspect padded 1.25 xywh -> UDP 192x256 -> ImageNet normalization -> flip average -> DARK/UDP original pixels',
                  historical_pipeline_equivalence=False, training=False, accuracy='unknown; requires independent image review',
                  timing_scope='Original video decode and shared detection are recorded in prepare report; no whole-system benchmark claim here')
    (args.output / 'run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
