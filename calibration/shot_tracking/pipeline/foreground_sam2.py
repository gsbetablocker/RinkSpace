"""Local SAM2 person silhouettes for the 121 native-frame shot trial.

Uses this run's source.mp4 and coarse athlete_bbox only. The project model directory
supplies pretrained model weights/configuration, never images or labels. Each
frame is segmented independently from its own box and torso point prompt.
Outputs stay inside runs/low_end_pm1/sam2_people/.

Run with the verified CUDA 12.8 environment (RTX 5070 Ti):
  D:/OneDrive/codex_projs/RinkSpace/.venv-gpu/Scripts/python.exe -B foreground_sam2.py --mode samples
  D:/OneDrive/codex_projs/RinkSpace/.venv-gpu/Scripts/python.exe -B foreground_sam2.py --mode all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import av
import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "calibration" / "shot_tracking" / "runs" / "low_end_pm1"
OUT = RUN / "sam2_people"
MODEL = ROOT / "models/cache/sam2.1-hiera-tiny"
SAMPLE_INDICES = (0, 60, 120)
FIRST_PTS = 104678400
FRAME_COUNT = 121
PTS_STEP = 256
SIZE_WH = (1920, 1080)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(pending, path)


def input_metadata(model_dir: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    source = read_json(RUN / "source.json")
    if (source["frame_count"], source["original_time_base"], source["width"], source["height"]) != (
            FRAME_COUNT, "1/15360", *SIZE_WH):
        raise ValueError("Shot source metadata changed")
    tracks = np.load(RUN / "pair_tracks.npz")
    boxes = tracks["athlete_bbox"]
    pts = np.load(RUN / "observations.npz")["pts"]
    expected = FIRST_PTS + PTS_STEP*np.arange(FRAME_COUNT)
    if boxes.shape != (FRAME_COUNT, 4) or not np.array_equal(pts, expected):
        raise ValueError("Boxes or original PTS do not cover the 121 native frames")
    if np.any(boxes[:, 0] < 0) or np.any(boxes[:, 1] < 0) or np.any(boxes[:, 2] > SIZE_WH[0]) or np.any(boxes[:, 3] > SIZE_WH[1]) or np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1]):
        raise ValueError("Invalid native athlete bounding boxes")
    clip = RUN / "source.mp4"
    weights = model_dir / "model.safetensors"
    if not clip.is_file() or not weights.is_file():
        raise FileNotFoundError("Local shot clip or SAM2 weights missing")
    meta = {
        "schema_version": "rinkspace.shot_tracking.sam2_people.v1",
        "scope": "01:53:35.000--01:53:37.000, exact original PTS 104678400..104709120",
        "source_clip": "source.mp4", "source_clip_sha256": sha256(clip),
        "original_source": source["original_source"],
        "original_source_sha256": sha256(Path(source["original_source"])),
        "original_time_base": "1/15360", "native_size_wh": list(SIZE_WH),
        "box_source": "pair_tracks.npz:athlete_bbox",
        "pair_tracks_sha256": sha256(RUN / "pair_tracks.npz"),
        "pts_source": "observations.npz:pts",
        "observations_sha256": sha256(RUN / "observations.npz"),
        "model_id": "facebook/sam2.1-hiera-tiny",
        "model_weights_sha256": sha256(weights),
        "model_path_is_external_runtime_only": True,
        "other_project_images_or_labels_used": False,
        "method": "local SAM2 image segmentation, fresh box plus torso point prompt per source frame; no temporal propagation",
        "mask_format": "native 1920x1080 BGRA PNG; all B,G,R,A channels equal 0 background or 255 person foreground",
        "mask_path_base": "runs/low_end_pm1",
        "mask_quality_independently_validated_for_all_frames": False,
        "frame_count_expected": FRAME_COUNT,
    }
    return meta, boxes, pts


def mask_from_sam(processor, model, image: np.ndarray, box: list[int], point: list[float],
                  torch, preprocess: dict, device: str):
    from PIL import Image
    import torch.nn.functional as functional

    height, width = image.shape[:2]
    session = processor.init_video_session(inference_device=device, inference_state_device=device,
        processing_device="cpu", video_storage_device="cpu", dtype=torch.float32)
    processor.add_inputs_to_inference_session(session, frame_idx=0, obj_ids=[1],
        input_boxes=[[box]], input_points=[[[point]]], input_labels=[[[1]]],
        original_size=(height, width))
    # The local runtime has torch and transformers but no torchvision, which
    # Sam2ImageProcessor requires. Reproduce its saved RGB/BILINEAR/scale/mean
    # normalization here; use Sam2VideoProcessor for prompt/session handling.
    size = (preprocess["size"]["width"], preprocess["size"]["height"])
    rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    resized = np.asarray(rgb.resize(size, resample=Image.Resampling.BILINEAR), dtype=np.float32)
    values = resized * float(preprocess["rescale_factor"])
    values = (values - np.asarray(preprocess["image_mean"], np.float32)) / np.asarray(preprocess["image_std"], np.float32)
    pixel_values = torch.from_numpy(np.ascontiguousarray(values.transpose(2, 0, 1))).to(device)
    with torch.inference_mode():
        result = model(inference_session=session, frame=pixel_values, frame_idx=0)
        # This is the local processor's default mask postprocess: bilinear
        # upsampling to original size and a logit threshold of zero.
        full = functional.interpolate(result.pred_masks, (height, width),
                                      mode="bilinear", align_corners=False) > 0
    foreground = full.detach().cpu().numpy().reshape(-1, height, width).any(axis=0)
    return foreground.astype(np.uint8)*255


def save_mask(out: Path, pts: int, mask: np.ndarray) -> tuple[Path, str]:
    path = out / "masks" / f"{pts}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{pts}.pending.png")
    bgra = np.repeat(mask[:, :, None], 4, axis=2)
    if not cv2.imwrite(str(pending), bgra):
        raise OSError(f"Failed to write {pending}")
    decoded = cv2.imread(str(pending), cv2.IMREAD_UNCHANGED)
    gray = cv2.imread(str(pending), cv2.IMREAD_GRAYSCALE)
    if decoded is None or decoded.shape != (SIZE_WH[1], SIZE_WH[0], 4) or not np.array_equal(decoded, bgra) or not np.array_equal(gray, mask):
        raise ValueError(f"Saved mask did not decode identically at PTS {pts}")
    os.replace(pending, path)
    return path, sha256(path)


def save_qa(out: Path, pts: int, image: np.ndarray, mask: np.ndarray,
            box: list[int], point: list[float]) -> None:
    shown = image.copy()
    foreground = mask > 0
    shown[foreground] = (shown[foreground]*.55 + np.array([30, 50, 235])*.45).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(shown, contours, -1, (0, 255, 255), 2)
    cv2.rectangle(shown, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
    cv2.circle(shown, tuple(np.rint(point).astype(int)), 7, (0, 255, 255), -1)
    cv2.imwrite(str(out / f"qa_{pts}.jpg"), shown, [cv2.IMWRITE_JPEG_QUALITY, 92])


def validate_saved(model_dir: Path) -> None:
    metadata, boxes, pts = input_metadata(model_dir)
    manifest = read_json(OUT / "masks.json")
    if any(manifest.get(key) != metadata[key] for key in (
            "source_clip_sha256", "original_source_sha256", "pair_tracks_sha256",
            "observations_sha256", "model_weights_sha256")):
        raise ValueError("Saved mask provenance differs from current source or model")
    rows = manifest["frames"]
    if len(rows) != FRAME_COUNT or [r["frame_index"] for r in rows] != list(range(FRAME_COUNT)):
        raise ValueError("Saved masks do not include all ordered native frames")
    areas, outside_fraction = [], []
    for index, row in enumerate(rows):
        if row["source_pts"] != int(pts[index]) or row["prompt_box_native_xyxy"] != boxes[index].tolist():
            raise ValueError(f"Mask source PTS or prompt changed at frame {index}")
        path = (RUN / row["mask_path"]).resolve()
        if not path.is_relative_to(OUT.resolve()) or sha256(path) != row["mask_sha256"]:
            raise ValueError(f"Mask path or checksum invalid at frame {index}")
        bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if bgra is None or bgra.shape != (SIZE_WH[1], SIZE_WH[0], 4):
            raise ValueError(f"Mask is not native BGRA at frame {index}")
        mask = bgra[:, :, 3]
        if not np.array_equal(bgra[:, :, :3], np.repeat(mask[:, :, None], 3, axis=2)) or not np.isin(mask, [0, 255]).all():
            raise ValueError(f"Mask channels or binary values differ at frame {index}")
        box = boxes[index]
        area = int(np.count_nonzero(mask))
        outside = area - int(np.count_nonzero(mask[box[1]:box[3], box[0]:box[2]]))
        if area != row["mask_foreground_pixels"] or outside != row["mask_outside_prompt_box_pixels"] or area == 0:
            raise ValueError(f"Mask area metadata differs at frame {index}")
        areas.append(area)
        outside_fraction.append(outside/area)
    ratios = np.asarray(areas[1:])/areas[:-1]
    result = {
        "schema_version": "rinkspace.shot_tracking.sam2_mask_integrity.v1",
        "source_pts_first": int(pts[0]), "source_pts_last": int(pts[-1]),
        "native_time_base": "1/15360", "native_size_wh": list(SIZE_WH),
        "frame_count": len(rows), "all_mask_hashes_match": True,
        "all_BGRA_channels_equal_binary_mask": True,
        "all_prompt_boxes_and_pts_match_inputs": True,
        "foreground_pixels_min": min(areas), "foreground_pixels_max": max(areas),
        "adjacent_area_ratio_min": round(float(ratios.min()), 6),
        "adjacent_area_ratio_max": round(float(ratios.max()), 6),
        "max_outside_prompt_box_fraction": round(max(outside_fraction), 6),
        "qa_source_pts_opened": [int(pts[i]) for i in (0, 60, 80, 120)],
        "all_frame_boundary_accuracy_proven": False,
    }
    write_json(OUT / "validation.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("samples", "all", "cuda-check", "validate"), required=True)
    parser.add_argument("--model-dir", type=Path, default=MODEL)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="Default requires CUDA; CPU needs an explicit --device cpu")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.mode == "validate":
        validate_saved(args.model_dir)
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    torch.set_num_threads(args.threads)
    if args.device == "cpu":
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        raise RuntimeError("CUDA is unavailable in this Python. Use the existing .venv-gpu runtime, or explicitly pass --device cpu.")
    if args.mode == "cuda-check" and device != "cuda":
        raise ValueError("cuda-check requires CUDA")
    metadata, boxes, pts = input_metadata(args.model_dir)
    metadata["runtime"] = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                           "device": device, "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
                           "threads": args.threads}
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "masks.json"
    previous = read_json(manifest_path) if manifest_path.is_file() else None
    if previous:
        stable = ("source_clip_sha256", "original_source_sha256", "pair_tracks_sha256",
                  "observations_sha256", "model_weights_sha256")
        if any(previous.get(key) != metadata[key] for key in stable):
            raise ValueError("Existing SAM2 masks use different source, prompts or weights")
        records = {int(row["frame_index"]): row for row in previous["frames"]}
    else:
        records = {}
    target = set(SAMPLE_INDICES if args.mode == "samples" else range(FRAME_COUNT))
    if args.mode == "cuda-check":
        if 0 not in records:
            raise ValueError("cuda-check requires the saved first-frame sample mask")
        target = {0}
    todo = target - records.keys()
    if not todo and args.mode != "cuda-check":
        print(json.dumps({"mode": args.mode, "reused": len(target), "generated": 0}, ensure_ascii=False))
        return
    preprocess = read_json(args.model_dir / "preprocessor_config.json")
    if (preprocess["size"], preprocess["image_mean"], preprocess["image_std"],
            preprocess["resample"], preprocess["do_resize"], preprocess["do_rescale"],
            preprocess["do_normalize"]) != (
            {"height": 1024, "width": 1024}, [.485, .456, .406], [.229, .224, .225],
            2, True, True, True):
        raise ValueError("Unexpected local SAM2 preprocessing configuration")
    metadata["processor_mode"] = "Sam2VideoProcessor prompt/session API; equivalent saved RGB bilinear preprocess and bilinear mask postprocess without missing torchvision"
    processor = Sam2VideoProcessor.__new__(Sam2VideoProcessor)
    processor.target_size = 1024
    processor.point_pad_value = -10
    model = Sam2VideoModel.from_pretrained(str(args.model_dir), local_files_only=True).to(device).eval()
    started = time.monotonic()
    generated = 0
    with av.open(str(RUN / "source.mp4")) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index >= FRAME_COUNT:
                raise ValueError("Source clip has more than 121 decoded frames")
            if (frame.width, frame.height) != SIZE_WH:
                raise ValueError(f"Unexpected source size at frame {index}")
            if index not in todo and not (args.mode == "cuda-check" and index == 0):
                continue
            image = frame.to_ndarray(format="bgr24")
            box = boxes[index].astype(int).tolist()
            point = [float((box[0]+box[2])/2), float(box[1]+.35*(box[3]-box[1]))]
            started_frame = time.monotonic()
            mask = mask_from_sam(processor, model, image, box, point, torch, preprocess, device)
            if args.mode == "cuda-check":
                from PIL import Image

                # This runtime has torchvision: compare our local equivalent
                # preprocessing with the stock saved processor on one frame.
                stock = Sam2VideoProcessor.from_pretrained(str(args.model_dir), local_files_only=True)
                stock_session = stock.init_video_session(inference_device=device,
                    inference_state_device=device, processing_device="cpu",
                    video_storage_device="cpu", dtype=torch.float32)
                stock.add_inputs_to_inference_session(stock_session, frame_idx=0,
                    obj_ids=[1], input_boxes=[[box]], input_points=[[[point]]],
                    input_labels=[[[1]]], original_size=image.shape[:2])
                stock_input = stock(images=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                                    return_tensors="pt").to(device)
                with torch.inference_mode():
                    stock_result = model(inference_session=stock_session,
                                         frame=stock_input.pixel_values[0], frame_idx=0)
                    stock_full = stock.post_process_masks([stock_result.pred_masks],
                        original_sizes=[[image.shape[0], image.shape[1]]])[0]
                stock_mask = stock_full.detach().cpu().numpy().reshape(-1, image.shape[0], image.shape[1]).any(axis=0)
                existing = RUN / records[0]["mask_path"]
                if not existing.is_file() or sha256(existing) != records[0]["mask_sha256"]:
                    raise ValueError("Existing CPU sample mask missing or changed")
                old = cv2.imread(str(existing), cv2.IMREAD_GRAYSCALE)
                if old is None or old.shape != mask.shape:
                    raise ValueError("Existing sample mask has unexpected size")
                overlap = int(np.count_nonzero((old > 0) & (mask > 0)))
                union = int(np.count_nonzero((old > 0) | (mask > 0)))
                stock_overlap = int(np.count_nonzero((old > 0) & stock_mask))
                stock_union = int(np.count_nonzero((old > 0) | stock_mask))
                custom_stock_overlap = int(np.count_nonzero((mask > 0) & stock_mask))
                custom_stock_union = int(np.count_nonzero((mask > 0) | stock_mask))
                result = {"schema_version": "rinkspace.shot_tracking.sam2_cuda_check.v1",
                          "source_pts": int(pts[0]), "existing_mask_sha256": records[0]["mask_sha256"],
                          "model_weights_sha256": metadata["model_weights_sha256"],
                          "runtime": metadata["runtime"],
                          "gpu_mask_pixels": int(np.count_nonzero(mask)),
                          "iou_vs_existing_cpu_mask": round(overlap/union, 6) if union else None,
                          "stock_processor_gpu_mask_pixels": int(np.count_nonzero(stock_mask)),
                          "stock_processor_iou_vs_existing_cpu_mask": round(stock_overlap/stock_union, 6) if stock_union else None,
                          "custom_vs_stock_processor_iou_on_gpu": round(custom_stock_overlap/custom_stock_union, 6) if custom_stock_union else None,
                          "exact_pixel_agreement": round(float(np.mean(old == mask)), 6),
                          "inference_seconds": round(time.monotonic()-started_frame, 3),
                          "mask_written": False}
                write_json(OUT / "cuda_check.json", result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
                return
            path, mask_hash = save_mask(OUT, int(pts[index]), mask)
            if index in SAMPLE_INDICES:
                save_qa(OUT, int(pts[index]), image, mask, box, point)
            inside = mask[box[1]:box[3], box[0]:box[2]]
            records[index] = {
                "frame_index": index, "source_pts": int(pts[index]),
                "source_time_s": round(float(pts[index]/15360), 6),
                "clip_local_pts": int(frame.pts) if frame.pts is not None else None,
                "prompt_box_native_xyxy": box, "prompt_torso_point_native_xy": point,
                "mask_path": str(path.relative_to(RUN)).replace("\\", "/"),
                "mask_sha256": mask_hash,
                "mask_foreground_pixels": int(np.count_nonzero(mask)),
                "mask_outside_prompt_box_pixels": int(np.count_nonzero(mask)-np.count_nonzero(inside)),
                "inference_seconds": round(time.monotonic()-started_frame, 3),
                "status": "sam2_candidate_unreviewed",
            }
            generated += 1
            manifest = metadata | {
                "frame_count_generated": len(records),
                "complete_selected_window": len(records) == FRAME_COUNT,
                "frames": [records[i] for i in sorted(records)],
            }
            write_json(manifest_path, manifest)
            if generated <= 3 or generated % 10 == 0:
                print(json.dumps({"mode": args.mode, "generated": generated, "index": index,
                                  "source_pts": int(pts[index]), "mask_pixels": records[index]["mask_foreground_pixels"],
                                  "frame_s": records[index]["inference_seconds"]}, ensure_ascii=False), flush=True)
    if not target.issubset(records):
        raise ValueError("Source clip ended before all requested frames were decoded")
    print(json.dumps({"mode": args.mode, "reused": len(target)-generated,
                      "generated": generated, "elapsed_s": round(time.monotonic()-started, 3),
                      "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
