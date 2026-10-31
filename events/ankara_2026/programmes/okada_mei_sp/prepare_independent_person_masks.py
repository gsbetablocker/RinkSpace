"""Fresh, source-bound SAM2 person candidates; search boxes never become masks.

Only the new source index, original RGB and local pretrained models are inputs.
Every source frame starts new SAM2 object state. Expanded boxes and current YOLO
arm keypoints are search hints; neither rectangles nor guessed limbs are filled.
Empty outputs, inference errors and unsupported hints remain explicit unknowns.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
import sys
from pathlib import Path
import time

import cv2
import numpy as np

from rinkspace.path_policy import guard_project_write
from rinkspace.review.render import source_frames
from rinkspace.vision_instances import Sam2SourceInstances, SourcePersonBoxes

DATA = Path("D:/CodexData/RinkSpace_Data")
BOOTSTRAP = DATA / "runs/native_programme_20261031/source_bootstrap"
ARM_KEYPOINTS = {7: "left_elbow", 8: "right_elbow", 9: "left_wrist", 10: "right_wrist"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(path)


def checked_shots(index):
    shots = index["shots"]
    ids = [shot["shot_id"] for shot in shots]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Unique new source shot identities are required")
    step = int(index["source_clock"]["native_pts_step"])
    previous = None
    for shot in shots:
        sid, p = shot["shot_id"], shot["processing"]
        if not sid.isascii() or not sid.replace("_", "").isalnum():
            raise ValueError("Unsafe shot ID")
        first, last, stop = (int(p[key]) for key in
                             ("first_pts", "last_pts", "stop_pts_exclusive"))
        if step <= 0 or last < first or (last-first) % step or stop != last+step:
            raise ValueError("Invalid registered processing range: " + sid)
        if (last-first)//step+1 != int(p["native_frame_count"]):
            raise ValueError("Native exposure count differs: " + sid)
        if previous is not None and first <= previous:
            raise ValueError("Source shot order overlaps or reverses")
        previous = last
    return shots


def _box_iou(first, second):
    a, b = np.asarray(first, float), np.asarray(second, float)
    overlap = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    return float(overlap / max((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-overlap, 1e-9))


def current_boxes_and_pose(detector, rgb, binding, confidence):
    """Public source boxes plus current-model pixel hints, never body geometry."""
    boxes = detector.read(rgb, binding)
    hints = [[] for _ in boxes]
    if not boxes:
        return boxes, hints
    # The common box interface deliberately does not expose pose. Read the
    # same local YOLO model on this exact unchanged RGB for arm search hints.
    prediction = detector.model.predict(source=rgb[:, :, ::-1].copy(),
                                        device=detector.device.index or 0, verbose=False)[0]
    if prediction.keypoints is None or prediction.keypoints.conf is None:
        return boxes, hints
    detected = prediction.boxes.xyxy.detach().cpu().numpy()
    xy = prediction.keypoints.xy.detach().cpu().numpy()
    scores = prediction.keypoints.conf.detach().cpu().numpy()
    height, width = rgb.shape[:2]
    for i, box in enumerate(boxes):
        matches = [_box_iou(box["box_xyxy"], other) for other in detected]
        if not matches or max(matches) < .5:
            continue
        match = int(np.argmax(matches))
        for key, name in ARM_KEYPOINTS.items():
            point, score = xy[match, key], float(scores[match, key])
            if (score >= confidence and np.isfinite(point).all() and
                    0 <= point[0] < width and 0 <= point[1] < height):
                hints[i].append(dict(name=name, uv=point.tolist(), confidence=score,
                    provenance="same_current_RGB_local_YOLO_pose_search_hint",
                    is_pixel_ground_truth=False))
    return boxes, hints


def mask_interior_hint(mask, box):
    """Pick an existing mask-interior pixel in the torso search ROI, or none."""
    height, width = mask.shape
    x0, y0, x1, y1 = np.asarray(box, float)
    bw, bh = x1-x0, y1-y0
    a, b = max(0, int(x0+.25*bw)), max(0, int(y0+.25*bh))
    c, d = min(width, int(np.ceil(x1-.25*bw))), min(height, int(np.ceil(y0+.65*bh)))
    eligible = np.zeros_like(mask)
    eligible[b:d, a:c] = mask[b:d, a:c]
    if not eligible.any():
        return None
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    score = np.where(eligible, distance, -1)
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    return [float(x), float(y)]


def expanded_search_box(box, hints, hw, *, up=.4, horizontal=.08, down=.05):
    height, width = hw
    x0, y0, x1, y1 = np.asarray(box, float)
    bw, bh = x1-x0, y1-y0
    expanded = np.array([x0-max(8., horizontal*bw), y0-up*bh,
                         x1+max(8., horizontal*bw), y1+down*bh])
    for hint in hints:
        point = np.asarray(hint["uv"], float)
        expanded[:2] = np.minimum(expanded[:2], point-12.)
        expanded[2:] = np.maximum(expanded[2:], point+12.)
    expanded = np.clip(expanded, [0, 0, 0, 0], [width, height, width, height])
    return expanded.tolist()


def _masks(result):
    return {row["object_id"]: row["region_mask"].detach().cpu().numpy().astype(bool)
            for row in result["instances"] if row["kind"] == "occluder"}


def fresh_person_candidates(sam, rgb, binding, boxes, arm_hints, *, up=.4, horizontal=.08, down=.05):
    """Two same-RGB fresh passes; keep only actual SAM predictions, no box fill."""
    height, width = rgb.shape[:2]
    zero = np.zeros((height, width), bool)
    if not boxes:
        return zero, dict(status="no_current_detector_proposal_missing_support", prompts=[],
                          errors=[], empty_object_ids=[], uncovered_positive_hints=[],
                          unresolved_support=True, candidate_present=False,
                          rectangle_fallback_applied=False, temporal_memory_used=False)
    base_prompts = [dict(object_id=i+1, kind="occluder", origin="original_rgb_candidate",
                         source_binding=binding, box_xyxy=row["box_xyxy"])
                    for i, row in enumerate(boxes)]
    errors, base_masks = [], {}
    try:
        base_masks = _masks(sam.read(rgb, binding, prompts=base_prompts, reset=True))
    except Exception as exc:
        errors.append(dict(pass_name="current_box_fresh", error=f"{type(exc).__name__}: {exc}"))
    refined_prompts, prompt_records = [], []
    for i, row in enumerate(boxes):
        object_id = i+1
        interior = mask_interior_hint(base_masks.get(object_id, zero), row["box_xyxy"])
        hints = list(arm_hints[i])
        points = ([interior] if interior is not None else []) + [hint["uv"] for hint in hints]
        search_box = expanded_search_box(row["box_xyxy"], hints, [height, width],
                                          up=up, horizontal=horizontal, down=down)
        prompt = dict(object_id=object_id, kind="occluder", origin="original_rgb_candidate",
                      source_binding=binding, box_xyxy=search_box)
        if points:
            prompt.update(points_uv=points, point_labels=[1]*len(points))
        refined_prompts.append(prompt)
        prompt_records.append(dict(object_id=object_id, original_detector_box_xyxy=row["box_xyxy"],
            expanded_box_xyxy=search_box, base_mask_interior_torso_search_hint=interior,
            high_confidence_current_arm_search_hints=hints, positive_points_uv=points,
            box_used_as_mask_or_clip=False, hints_are_pixel_ground_truth=False))
    masks = base_masks
    status = "fresh_expanded_box_and_current_hint_SAM2_candidate"
    try:
        masks = _masks(sam.read(rgb, binding, prompts=refined_prompts, reset=True))
        if sam.frame_index != 0 or set(sam.objects) != set(range(1, len(boxes)+1)):
            raise RuntimeError("Fresh frame retained previous object state")
    except Exception as exc:
        errors.append(dict(pass_name="expanded_box_fresh", error=f"{type(exc).__name__}: {exc}"))
        status = "fresh_current_box_candidate_refinement_error"
    combined, empty_ids, uncovered = zero.copy(), [], []
    for record in prompt_records:
        object_id = record["object_id"]
        mask = masks.get(object_id, zero)
        combined |= mask
        if not mask.any():
            empty_ids.append(object_id)
        for point in record["positive_points_uv"]:
            x, y = np.rint(point).astype(int)
            x, y = np.clip([x, y], [0, 0], [width-1, height-1])
            if not mask[y, x]:
                uncovered.append(dict(object_id=object_id, uv=point))
    if not combined.any():
        status = "empty_SAM2_candidate_missing_support"
    return combined, dict(status=status, prompts=prompt_records, errors=errors,
        empty_object_ids=empty_ids, uncovered_positive_hints=uncovered,
        unresolved_support=bool(errors or empty_ids or uncovered),
        candidate_present=bool(combined.any()), rectangle_fallback_applied=False,
        temporal_memory_used=False, SAM2_frame_index=0,
        candidate_does_not_prove_complete_person_coverage=True)


def frame_support_status(candidate_present, diagnostic):
    """A missing mask is pending support, never evidence of unobscured RGB."""
    if not candidate_present:
        return "pending_source_person_support"
    if diagnostic.get("unresolved_support"):
        return "current_RGB_SAM2_candidate_unresolved_search_support"
    return "current_RGB_SAM2_candidate"


def write_support_index(out, rows):
    statuses = []
    for row in rows:
        present = bool(row["mask_nonzero_pixels"])
        status = frame_support_status(present, row["diagnostic"])
        row["frame_support_status"] = status
        statuses.append(dict(source_pts=row["source_binding"]["source_pts"],
            shot_id=row["source_binding"]["shot_id"], status=status,
            current_RGB_segmentation_present=present,
            unresolved_search_support=bool(row["diagnostic"].get("unresolved_support")),
            mask_path=row["mask_path"], mask_sha256=row["mask_sha256"],
            missing_mask_means_unobscured=False,
            search_hint_origin=row["diagnostic"].get("search_hint_origin", "current_RGB_detector")))
    path = Path(out) / "frame_support_status.json"
    write_json(path, dict(schema="rinkspace.source_person_frame_support.v1",
        empty_mask_is_unobscured=False, pixel_accuracy_accepted=False, frames=statuses))
    return dict(path=str(path), sha256=digest(path),
        counts=dict(Counter(row["status"] for row in statuses)),
        pending_source_pts=[row["source_pts"] for row in statuses if not row["current_RGB_segmentation_present"]],
        empty_mask_is_unobscured=False)


def run(args):
    import torch
    started = time.perf_counter()
    index_path = Path(args.index).resolve(strict=True)
    index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    shots = checked_shots(index)
    source = {**index["source"], **index["source_clock"]}
    source_path = Path(source.get("resolved_path", source["registered_path"])).resolve(strict=True)
    if digest(source_path) != source["sha256"]:
        raise ValueError("Original video identity differs from source index")
    height, width = int(source["height"]), int(source["width"])
    clock = Fraction(source["time_base_num"], source["time_base_den"])
    out = guard_project_write(args.output)
    if not out.is_relative_to(DATA.resolve()):
        raise ValueError("Person candidates must remain in Data")
    if out.exists():
        raise FileExistsError("Preserve existing mask output: " + str(out))
    if min(args.dilate_px, args.expand_up, args.expand_horizontal, args.expand_down) < 0:
        raise ValueError("Invalid search expansion or display dilation")
    if not 0 <= args.pose_confidence <= 1:
        raise ValueError("Invalid current pose search confidence")
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(DATA / "runtime_tools/yolo"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    detector = SourcePersonBoxes(args.person_weights, runtime_directory=DATA / "runtime_tools/yolo", device="cuda")
    sam = Sam2SourceInstances(args.sam_checkpoint, device="cuda", reread_rgb=False, contour_stride=32)
    record = dict(schema="rinkspace.independent_source_person_masks.v2", status="running",
        created_utc=datetime.now(timezone.utc).isoformat(),
        inputs=dict(index=str(index_path), index_sha256=digest(index_path), source_video=str(source_path),
            source_sha256=source["sha256"], person_weights=str(args.person_weights),
            person_weights_sha256=detector.weights_sha256, sam_checkpoint=str(args.sam_checkpoint),
            sam_weights_sha256=sam.model_sha256, script=str(Path(__file__).resolve()), script_sha256=digest(__file__),
            public_instances_sha256=digest(Path(__file__).resolve().parents[4] / "src/rinkspace/vision_instances.py")),
        prohibited_prior_observations_tracks_or_advertisement_traces_read=False,
        camera_state_or_rig_binding_used=False, source_time_base=str(clock), native_hw=[height, width],
        binary_values=[0, 255], foreground_value=255,
        meaning="raw source-bound SAM2 person candidate; zero does not prove unobscured pixels",
        method=dict(frame_state="fresh reset for every pass of every actual RGB frame",
            source_boxes_are_search_only=True, rectangle_fallback_applied=False,
            raw_masks_not_clipped_to_prompt_boxes=True,
            current_box_fresh_pass_then_expanded_box_fresh_pass=True,
            expand_up_bbox_heights=args.expand_up, expand_horizontal_bbox_widths=args.expand_horizontal,
            expand_down_bbox_heights=args.expand_down, current_YOLO_arm_confidence=args.pose_confidence,
            current_mask_interior_positive_hint=True, no_old_points_tracks_camera_or_object_ids=True,
            mask_dilation_px=args.dilate_px),
        runtime=dict(python_executable=sys.executable, python_version=sys.version,
            torch=torch.__version__, cuda_runtime=torch.version.cuda,
            transformers=__import__("transformers").__version__, numpy=np.__version__, opencv=cv2.__version__),
        mask_pixel_accuracy_accepted=False, whole_shot_visual_acceptance=False,
        frame_provenance=str(out / "frames.jsonl"), shots=[])
    history = out.parent / "history/prepare_independent_person_masks_before_fresh.py"
    if history.is_file():
        record["previous_generation_source_history"] = dict(path=str(history), sha256=digest(history))
    write_json(out / "manifest.json", record)
    total, totals, unsupported, empty, exceptions = 0, Counter(), 0, 0, 0
    try:
        with (out / "frames.jsonl").open("x", encoding="utf8", buffering=1) as ledger:
            for shot in shots:
                sid, p = shot["shot_id"], shot["processing"]
                target = out / sid
                target.mkdir()
                expected = np.arange(p["first_pts"], p["stop_pts_exclusive"], source["native_pts_step"], dtype=np.int64)
                stamps, counts, shot_unsupported, shot_empty = [], Counter(), 0, 0
                for number, (stamp, bgr) in enumerate(source_frames(source_path, source, expected)):
                    tick = time.perf_counter()
                    rgb = bgr[:, :, ::-1].copy()
                    binding = dict(source_sha256=source["sha256"], source_pts=int(stamp), time_base=str(clock),
                        shot_id=sid, native_hw=[height, width], image_domain="native_brown_pixels")
                    boxes, arm_hints = [], []
                    try:
                        boxes, arm_hints = current_boxes_and_pose(detector, rgb, binding, args.pose_confidence)
                        mask, diagnostic = fresh_person_candidates(sam, rgb, binding, boxes, arm_hints,
                            up=args.expand_up, horizontal=args.expand_horizontal, down=args.expand_down)
                    except Exception as exc:
                        mask = np.zeros((height, width), bool)
                        diagnostic = dict(status="source_inference_error_missing_support", prompts=[],
                            errors=[dict(pass_name="source_person", error=f"{type(exc).__name__}: {exc}")],
                            unresolved_support=True, candidate_present=False, rectangle_fallback_applied=False,
                            temporal_memory_used=False, candidate_does_not_prove_complete_person_coverage=True)
                    if args.dilate_px:
                        side = args.dilate_px*2+1
                        mask = cv2.dilate(mask.astype(np.uint8), np.ones((side, side), np.uint8)).astype(bool)
                    path = target / f"{stamp}.png"
                    if not cv2.imwrite(str(path), mask.astype(np.uint8)*255, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                        raise RuntimeError("Could not save native binary candidate")
                    stamps.append(int(stamp))
                    status = diagnostic["status"]
                    counts[status] += 1
                    totals[status] += 1
                    total += 1
                    unknown = bool(diagnostic["unresolved_support"])
                    absent = not mask.any()
                    shot_unsupported += unknown
                    shot_empty += absent
                    unsupported += unknown
                    empty += absent
                    exceptions += bool(diagnostic["errors"])
                    row = dict(source_binding=binding, decoded_native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                        mask_path=str(path), mask_sha256=digest(path), mask_kind=status,
                        mask_nonzero_pixels=int(mask.sum()), diagnostic=diagnostic,
                        current_source_person_boxes=[dict(box_xyxy=x["box_xyxy"], confidence=x["provenance"]["confidence"])
                                                     for x in boxes],
                        frame_support_status=frame_support_status(bool(mask.any()), diagnostic),
                        rectangle_fallback_applied=False, precise_segmentation_accepted=False,
                        elapsed_seconds=time.perf_counter()-tick)
                    ledger.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    if total % 50 == 0 or number == len(expected)-1:
                        progress = dict(status="running", shot_id=sid, shot_frames=number+1, complete_frames=total,
                            expected_frames=sum(s["processing"]["native_frame_count"] for s in shots),
                            unresolved_support_frames=unsupported, empty_candidate_frames=empty,
                            inference_error_frames=exceptions, elapsed_seconds=time.perf_counter()-started)
                        write_json(out / "progress.json", progress)
                        print(json.dumps(progress), flush=True)
                if stamps != expected.tolist():
                    raise RuntimeError("Actual PTS coverage differs: " + sid)
                for stamp in stamps:
                    saved = cv2.imread(str(target / f"{stamp}.png"), cv2.IMREAD_UNCHANGED)
                    if (saved is None or saved.shape != (height, width) or saved.dtype != np.uint8 or
                            not np.isin(saved, [0, 255]).all()):
                        raise RuntimeError("Saved binary candidate failed verification")
                record["shots"].append(dict(shot_id=sid, processing=p, actual_pts=stamps,
                    decoded_frames=len(stamps), verified_binary_native_pngs=len(stamps),
                    actual_pts_equal_registered_processing=True, mask_kinds=dict(counts),
                    unresolved_support_frames=shot_unsupported, empty_candidate_frames=shot_empty))
                write_json(out / "manifest.json", record)
        expected_count = sum(s["processing"]["native_frame_count"] for s in shots)
        if total != expected_count:
            raise RuntimeError("Actual source coverage differs")
        support_rows = [json.loads(line) for line in (out / "frames.jsonl").read_text(encoding="utf8").splitlines()]
        record["frame_support_status"] = write_support_index(out, support_rows)
        record.update(status="source_pts_and_binary_candidates_verified_support_not_accepted",
            total_frames=total, expected_frames=expected_count, mask_kinds=dict(totals),
            actual_source_pts_coverage_complete=True, unresolved_support_frames=unsupported,
            empty_candidate_frames=empty, inference_error_frames=exceptions,
            frame_provenance_sha256=digest(out / "frames.jsonl"), elapsed_seconds=time.perf_counter()-started)
        write_json(out / "manifest.json", record)
        write_json(out / "coverage.json", {k: record[k] for k in ("status", "source_time_base", "native_hw",
            "total_frames", "expected_frames", "actual_source_pts_coverage_complete", "unresolved_support_frames",
            "empty_candidate_frames", "inference_error_frames", "shots", "mask_pixel_accuracy_accepted")})
        write_json(out / "progress.json", dict(status=record["status"], complete_frames=total,
            expected_frames=expected_count, unresolved_support_frames=unsupported,
            empty_candidate_frames=empty, inference_error_frames=exceptions, elapsed_seconds=record["elapsed_seconds"]))
        print(json.dumps(dict(status=record["status"], total_frames=total, output=str(out),
            unresolved_support_frames=unsupported, empty_candidate_frames=empty,
            inference_error_frames=exceptions, elapsed_seconds=record["elapsed_seconds"])), flush=True)
    except Exception as exc:
        record.update(status="failed_preserved_partial_output", error=f"{type(exc).__name__}: {exc}",
                      complete_frames=total, elapsed_seconds=time.perf_counter()-started)
        write_json(out / "manifest.json", record)
        raise
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=BOOTSTRAP / "index.json")
    parser.add_argument("--output", type=Path, default=BOOTSTRAP / "person_masks_refined/full")
    parser.add_argument("--person-weights", type=Path, default=DATA / "cache/pretrained/yolo11n-pose.pt")
    parser.add_argument("--sam-checkpoint", type=Path, default=DATA / "cache/pretrained/sam2.1-hiera-tiny")
    parser.add_argument("--expand-up", type=float, default=.4)
    parser.add_argument("--expand-horizontal", type=float, default=.08)
    parser.add_argument("--expand-down", type=float, default=.05)
    parser.add_argument("--pose-confidence", type=float, default=.65)
    parser.add_argument("--dilate-px", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
