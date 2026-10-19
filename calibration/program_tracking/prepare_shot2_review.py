"""Select clear official50 shot-two exposures and save exact source/navigation.

All selection metrics use the current official native exposures. Old source
pixel matches choose 8a28 navigation cameras only. No rink points are produced.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "calibration/program_tracking")]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import av
import cv2
import numpy as np
from rinkspace.programme_source_stream import file_sha
from calibration.program_tracking.prepare_continuation_frame_sources import read, timestamp, array_sha
from calibration.program_tracking.prepare_first_shot_review import decoded, write

DATA = Path("D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/shot2_review_20261014")
PARENT = ROOT / "events/ankara_2026/programmes/okada_mei_sp/source_datasets/frame_by_frame_review_20261013"
DEFAULT_MODEL = ROOT / "events/ankara_2026/models/framewise_8a28_first_shot_20261014/repair_20261014/model.json"
PROGRAMME_FIRST_PTS = 256256
START, LAST, STOP = 575744, 720640, 720896


def measure(bgr, previous_gray):
    """Board material contrast and temporal motion are selection aids only."""
    small = cv2.resize(bgr, (960, 540), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    blue = ((hsv[:, :, 0] >= 93) & (hsv[:, :, 0] <= 130)
            & (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 45))
    blue_coverage = blue.mean(axis=1)
    top_candidates = np.flatnonzero((blue_coverage > .28) & (np.arange(540) >= 75)
                                   & (np.arange(540) < 420))
    top = int(top_candidates[0]) if len(top_candidates) else 155
    yellow = ((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 42)
              & (hsv[:, :, 1] >= 65) & (hsv[:, :, 2] >= 70))
    coverage = yellow.mean(axis=1)
    valid = (np.arange(540) > top + 35) & (np.arange(540) < 510)
    bottom = int(np.argmax(np.where(valid, coverage, 0)))
    contact_detected = bool(coverage[bottom] > .12)
    if not contact_detected:
        bottom = min(490, top + 150)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    y0, y1 = max(150, 2 * top), min(1000, max(2 * bottom, 2 * top + 80))
    # The two side ROIs avoid the central skater for selection. They are not
    # segmentation masks and never authorize fitting any included pixel.
    left = gray[y0:y1, 20:600]
    right = gray[y0:y1, 1320:1900]
    lap = np.concatenate([cv2.Laplacian(p, cv2.CV_32F).ravel() for p in (left, right)])
    grad = np.concatenate([np.hypot(cv2.Sobel(p, cv2.CV_32F, 1, 0),
                                  cv2.Sobel(p, cv2.CV_32F, 0, 1)).ravel() for p in (left, right)])
    texture = grad > 12
    lap_var = float(np.var(lap))
    ratio = float(np.mean(np.abs(lap[texture])) / (np.mean(grad[texture]) + 1e-6)) if texture.any() else 0.
    motion_gray = cv2.resize(gray[220:870, :], (480, 163), interpolation=cv2.INTER_AREA).astype(np.float32)
    motion = None
    response = None
    if previous_gray is not None:
        shift, response = cv2.phaseCorrelate(previous_gray, motion_gray)
        motion = float(np.hypot(*shift) * 4)
    return dict(board_selection_ROI_native=[20, y0, 1900, y1],
                center_person_region_excluded_for_metric=[600, y0, 1320, y1],
                blue_rail_color_candidate_y=2 * top,
                yellow_contact_color_candidate_y=2 * bottom,
                yellow_candidate_detected=contact_detected,
                laplacian_variance=lap_var, high_frequency_to_gradient_ratio=ratio,
                temporal_phase_motion_pixels=motion,
                temporal_phase_correlation_response=float(response) if response is not None else None), motion_gray


def contact_sheet(rows, field, path):
    tiles = []
    for r in rows:
        tile = cv2.resize(cv2.imread(r[field]), (480, 270), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f"{r['index']}  {timestamp(r['native_pts'])}", (6, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    for _ in range((-len(tiles)) % 4):
        tiles.append(np.zeros((270, 480, 3), np.uint8))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    if not cv2.imwrite(str(path), sheet):
        raise OSError(path)


def prepare(model_path, count):
    started = time.perf_counter()
    if (DATA / "frame_manifest.json").exists():
        existing = read(DATA / "frame_manifest.json")
        if existing.get("latest_cadence_review"):
            print(json.dumps(dict(stage="latest_cadence_sources_reused", frames=existing["frame_count"])), flush=True)
            return existing
    model = read(model_path)
    opening = read(PARENT / "opening_100_50fps/navigation_mapping.json")
    canonical = read(Path("D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/frame_by_frame_review_20261013/frame_000200_50fps/source_binding.json"))
    source_record = read(ROOT / "events/ankara_2026/programmes/okada_mei_sp/primary_source_selection.json")["primary_source"]
    source = Path(source_record["path"])
    ledger_ref = opening["mapping_evidence"]
    archive_ref = opening["model_seed"]["inherited_camera_archive"]
    representation = canonical["ledger_official_representation"]
    for record in (source_record, ledger_ref, archive_ref, representation, model["scene"]):
        if file_sha(record["path"]) != record["sha256"]:
            raise ValueError("Authenticated input changed: " + record["path"])
    ledger = [dict(json.loads(line), line_1based=i + 1) for i, line in enumerate(Path(ledger_ref["path"]).read_text("utf8").splitlines())]
    shot_rows = [r for r in ledger if r["shot_index"] == 2]
    accepted = {}
    for r in shot_rows:
        if r["match_status"] == "unique_pixel_match" and r.get("accepted_official_pts") is not None:
            accepted.setdefault(int(r["accepted_official_pts"]), []).append(r)
    if set(accepted) != set(range(START, STOP, 256)):
        raise ValueError("Second-shot official exposure identity is incomplete")
    metrics = []
    previous = None
    for pts, bgr in decoded(source, START, STOP):
        values, previous = measure(bgr, previous)
        values.update(native_pts=pts, source_time=timestamp(pts),
                      programme_frame_index_1based=(pts - PROGRAMME_FIRST_PTS) // 256 + 1)
        metrics.append(values)
    # Search +/- 0.12 sec around a 16-frame nominal cadence. First and last
    # windows extend inwards so their coverage distances remain explicit.
    anchors = np.linspace(0, len(metrics) - 1, count)
    selected = []
    windows = []
    for ordinal, anchor in enumerate(anchors):
        lo, hi = max(0, int(round(anchor)) - 6), min(len(metrics), int(round(anchor)) + 7)
        candidates = metrics[lo:hi]
        laps = np.log1p([r["laplacian_variance"] for r in candidates])
        ratios = np.asarray([r["high_frequency_to_gradient_ratio"] for r in candidates])
        # Local normalization avoids comparing different advertisements' ink
        # coverage as if it were the same texture. Motion is recorded, while
        # edge sharpness chooses the final source exposure.
        scores = (laps - laps.min()) / max(float(np.ptp(laps)), 1e-6)
        scores += .8 * (ratios - ratios.min()) / max(float(np.ptp(ratios)), 1e-6)
        scores -= .07 * np.abs(np.arange(lo, hi) - anchor) / 6
        winner = lo + int(np.argmax(scores))
        selected.append(winner)
        windows.append(dict(selection_ordinal_1based=ordinal + 1,
                            nominal_native_pts=START + int(round(anchor)) * 256,
                            candidate_first_pts=metrics[lo]["native_pts"],
                            candidate_last_pts=metrics[hi - 1]["native_pts"],
                            selected_native_pts=metrics[winner]["native_pts"],
                            selected_score=float(scores[winner - lo]),
                            candidates=[dict(native_pts=r["native_pts"], score=float(s),
                                             laplacian_variance=r["laplacian_variance"],
                                             high_frequency_to_gradient_ratio=r["high_frequency_to_gradient_ratio"])
                                        for r, s in zip(candidates, scores)]))
    pts_list = [metrics[i]["native_pts"] for i in selected]
    gaps = np.diff(pts_list) / 12800
    if not np.all((gaps >= .5) & (gaps <= 1.)):
        raise ValueError("Selection intervals do not meet the authorized 0.5–1 sec cadence")
    indices = np.asarray([(p - PROGRAMME_FIRST_PTS) // 256 + 1 for p in pts_list], np.int64)
    for folder in ("frames", "source_bindings", "navigation"):
        (DATA / folder).mkdir(parents=True, exist_ok=True)
    wanted = set(pts_list)
    rows = []
    for pts, bgr in decoded(source, START, STOP):
        if pts not in wanted:
            continue
        index = (pts - PROGRAMME_FIRST_PTS) // 256 + 1
        image = DATA / "frames" / f"{index:06d}.png"
        if not cv2.imwrite(str(image), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise OSError(image)
        if not np.array_equal(bgr, cv2.imread(str(image))):
            raise ValueError("Native PNG roundtrip changed pixels")
        binding = dict(schema="rinkspace.exact-single-native-source-frame.v1",
                       event_id="ankara_2026", programme_id="okada_mei_sp", shot_index=2,
                       programme_frame_index_1based=index, native_frame_index_1based=pts // 256 + 1,
                       native_frame_index_0based=pts // 256, native_pts=pts, actual_pts=pts,
                       time_base="1/12800", source_time=timestamp(pts), source_time_seconds=pts / 12800,
                       source=str(source), source_video_sha256=source_record["sha256"],
                       image_path=str(image), png_sha256=file_sha(image),
                       decoded_BGR_sha256=array_sha(bgr),
                       decoded_RGB_sha256=array_sha(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
                       native_hw=[1080, 1920], original_PTS_verified=True,
                       PNG_roundtrip_lossless_verified=True, optimization_performed=False,
                       ledger_official_representation=representation)
        rows.append(dict(index=index, source=binding, image_path=str(image), path=str(image),
                         native_pts=pts, source_pts=pts, source_time_base="1/12800",
                         source_time=timestamp(pts), source_time_seconds=pts / 12800,
                         native_frame_index_0based=pts // 256, png_sha256=binding["png_sha256"],
                         decoded_native_BGR_sha256=binding["decoded_BGR_sha256"],
                         decoded_native_RGB_sha256=binding["decoded_RGB_sha256"],
                         native_hw=[1080, 1920], exact_native_PTS_and_lossless_PNG_verified=True,
                         selection_metric=metrics[(pts - START) // 256]))
    by_pts = {r["native_pts"]: r for r in rows}
    verified = []
    for pts, bgr in decoded(Path(representation["path"]), START, STOP):
        if pts not in wanted:
            continue
        if array_sha(bgr) != by_pts[pts]["decoded_native_BGR_sha256"]:
            raise ValueError(f"Current official versus ledger official pixels differ at {pts}")
        row = by_pts[pts]
        row["ledger_official_representation_same_frame_BGR_identical"] = True
        row["source"]["ledger_official_representation_same_frame_BGR_identical"] = True
        binding_path = DATA / "source_bindings" / f"{row['index']:06d}.json"
        write(binding_path, row["source"])
        row["source_binding"] = dict(path=str(binding_path), sha256=file_sha(binding_path))
        verified.append(dict(native_pts=pts, decoded_BGR_sha256=array_sha(bgr), decoded_pixels_identical=True))
    if len(verified) != count:
        raise ValueError("Incomplete current-to-ledger selected-frame comparison")
    with np.load(archive_ref["path"], allow_pickle=False) as saved:
        cameras = {k: saved[k].copy() for k in saved.files}
    lookup = {int(p): i for i, p in enumerate(cameras["pts"])}
    map_rows, archive_indices = [], []
    for ordinal, row in enumerate(rows):
        ledger_row = min(accepted[row["native_pts"]], key=lambda r: r["full_pts"])
        archive_index = lookup[int(ledger_row["full_pts"])]
        archive_indices.append(archive_index)
        map_rows.append(dict(sequence_index_1based=row["index"],
                             programme_frame_index_1based=row["index"], official_pts=row["native_pts"],
                             official_time_base="1/12800", official_time_seconds=row["native_pts"] / 12800,
                             legacy60_pts=ledger_row["full_pts"], legacy60_time_base="1/15360",
                             legacy60_shot_index=2, mapping_status="confirmed_pixel_match",
                             pixel_correspondence_confirmed=True, navigation_only=True,
                             ledger_evidence=ledger_row, camera_seed_index_0based=ordinal,
                             inherited_camera_archive_row_0based=archive_index,
                             primary_official50_vs_ledger_official50_BGR_identical=True,
                             camera25_float64_sha256=array_sha(cameras["camera25"][archive_index])))
    ai = np.asarray(archive_indices, np.int64)
    np.savez(DATA / "camera_seeds.npz", programme_indices=indices, pts=np.asarray(pts_list, np.int64),
             legacy60_pts=np.asarray([r["legacy60_pts"] for r in map_rows], np.int64),
             inherited_camera_archive_indices=ai, camera25=cameras["camera25"][ai],
             installation_id=cameras["installation_id"][ai], view_state4=cameras["view_state4"][ai],
             exposure_representative_pts=cameras["exposure_representative_pts"][ai])
    from calibration.program_tracking.render_result import SceneRenderer
    renderer = SceneRenderer(read(model["scene"]["path"]), draw_grid=True, draw_advertisements=True)
    for row, mapping_row, camera in zip(rows, map_rows, cameras["camera25"][ai]):
        raw = cv2.imread(row["image_path"])
        overlay, _ = renderer.draw(raw, camera, person_mask=np.zeros(raw.shape[:2], bool), title=None)
        navigation = DATA / "navigation" / f"{row['index']:06d}.png"
        if not cv2.imwrite(str(navigation), overlay, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise OSError(navigation)
        row["navigation_path"] = str(navigation)
        mapping_row["navigation_overlay"] = dict(path=str(navigation), sha256=file_sha(navigation),
                                                 navigation_only=True, current_source_background=True,
                                                 current_person_mask_applied=False, camera_seed_refitted=False,
                                                 all_original_feature_paths_and_advertisement_ids_enabled=True,
                                                 body_or_trajectory_drawn=False, path_counts=renderer.path_counts)
    boundary = dict(shot_index=2, first_pts=START, last_pts=LAST, stop_pts_exclusive=STOP,
                    time_base="1/12800", first_source_time=timestamp(START),
                    last_source_time=timestamp(LAST), native_frame_count=(STOP - START) // 256,
                    first_programme_index=(START - PROGRAMME_FIRST_PTS) // 256 + 1,
                    last_programme_index=(LAST - PROGRAMME_FIRST_PTS) // 256 + 1,
                    before_cut_pts=START - 256, next_shot_pts=STOP,
                    boundary_status="adjacent unique pixel matches and current source boundary pixels checked")
    selection = dict(schema="rinkspace.native-shot-clear-exposure-selection.v1", shot_scope=boundary,
                     requested_interval_seconds=[.5, 1.], selected_frame_count=len(rows),
                     selected_pts=pts_list, programme_indices=indices.tolist(),
                     selected_interval_seconds=gaps.tolist(),
                     uncovered_start_seconds=(pts_list[0] - START) / 12800,
                     uncovered_end_seconds=(LAST - pts_list[-1]) / 12800,
                     method="local +/-0.12sec board-material high-frequency and Laplacian ranking; human source review remains required",
                     selection_metrics_are_pixel_visibility_or_accuracy_proof=False,
                     all_native_shot_exposures_measured=len(metrics), metrics=metrics, windows=windows,
                     source_points_extracted=False, optimizer_run=False)
    write(DATA / "selection.json", selection)
    verification = dict(schema="rinkspace.exact-current-vs-ledger-official-representation-verification.v1",
                        scope="Selected official50 shot2 source exposures", selected_frame_count=len(rows),
                        current_primary_source=source_record, ledger_official_representation=representation,
                        all_selected_decoded_BGR_identical=True, whole_shot_pixel_equality_claimed=False,
                        checked_via="Exact native PTS decoded BGR SHA256 plus PNG roundtrip", records=verified)
    write(DATA / "official_representation_verification.json", verification)
    mapping = dict(schema="rinkspace.official50-shot-navigation-mapping.v1", event_id="ankara_2026",
                   programme_id="okada_mei_sp", purpose="shot2 selected sources; 8a28 identity navigation only",
                   official_source=source_record, mapping_evidence=ledger_ref,
                   model_reference=dict(path=str(model_path), sha256=file_sha(model_path)),
                   model_seed=deepcopy(opening["model_seed"]), frames=map_rows,
                   official_representation_verification=dict(path=str(DATA / "official_representation_verification.json"),
                                                             sha256=file_sha(DATA / "official_representation_verification.json")),
                   summary={"confirmed_pixel_match": len(rows)}, source_clock_substituted=False,
                   official50_camera_fit_performed=False, optimizer_run=False)
    mapping["model_seed"]["camera_seeds"] = dict(path=str(DATA / "camera_seeds.npz"), sha256=file_sha(DATA / "camera_seeds.npz"))
    mapping["model_seed"]["camera25_shape"] = [len(rows), 25]
    mapping["model_seed"]["scene"] = model["scene"]
    write(DATA / "navigation_mapping.json", mapping)
    manifest = dict(schema="rinkspace.exact-native-selected-source-frames.v1", event_id="ankara_2026",
                    programme_id="okada_mei_sp", source=source_record, shot_scope=boundary,
                    programme_indices=indices.tolist(), start_pts=pts_list[0], last_pts=pts_list[-1],
                    stop_pts_exclusive=pts_list[-1] + 256, native_pts_step=256, frame_count=len(rows),
                    original_PTS_order_verified=True, native_exposures_all_decoded=True,
                    selected_frames_only_saved=True, PNG_roundtrip_lossless_verified=True,
                    source_points_extracted=False, optimization_performed=False,
                    navigation_mapping=dict(path=str(DATA / "navigation_mapping.json"), sha256=file_sha(DATA / "navigation_mapping.json")),
                    selection=dict(path=str(DATA / "selection.json"), sha256=file_sha(DATA / "selection.json")),
                    decoder=dict(library="PyAV", version=av.__version__, cpu_threads=2,
                                 temporal_resampling=False, spatial_resizing_of_source_PNGs=False),
                    frames=rows, source_prepare_elapsed_seconds=time.perf_counter() - started)
    write(DATA / "frame_manifest.json", manifest)
    contact_sheet(rows, "image_path", DATA / "source_contact_sheet.png")
    contact_sheet(rows, "navigation_path", DATA / "navigation_contact_sheet.png")
    print(json.dumps(dict(stage="shot2_sources_ready", frames=len(rows), pts=pts_list,
                          intervals=gaps.tolist(), manifest=str(DATA / "frame_manifest.json"),
                          all_selected_current_vs_ledger_BGR_identical=True,
                          elapsed_seconds=time.perf_counter() - started)), flush=True)
    return manifest


def apply_reviewed_selection():
    """Apply the visually reviewed clear-frame subset, preserving masks/cache."""
    manifest = read(DATA / "frame_manifest.json")
    selection = read(DATA / "selection.json")
    mapping = read(DATA / "navigation_mapping.json")
    verification = read(DATA / "official_representation_verification.json")
    if manifest["programme_indices"] != [r["index"] for r in manifest["frames"]]:
        raise ValueError("Source manifest order changed")
    if 1273 in manifest["programme_indices"] and len(manifest["frames"]) == 13:
        print(json.dumps(dict(stage="reviewed_selection_already_applied", frames=13)), flush=True)
        return manifest
    if len(manifest["frames"]) != 16:
        raise ValueError("Expected the original 16-frame selection before clarity review")
    history_path = DATA / "historical_selection.json"
    if history_path.exists():
        raise ValueError("Historical selection already exists; refuse to overwrite provenance")
    history_path.write_bytes((DATA / "selection.json").read_bytes())
    removed = {1288, 1324, 1358, 1394}
    rows = [deepcopy(r) for r in manifest["frames"] if r["index"] not in removed]
    source_record = manifest["source"]
    representation = verification["ledger_official_representation"]
    new_pts, new_index = 581888, 1273
    bgr = next(decoded(Path(source_record["path"]), new_pts, new_pts + 256))[1]
    old_representation = next(decoded(Path(representation["path"]), new_pts, new_pts + 256))[1]
    if not np.array_equal(bgr, old_representation):
        raise ValueError("New reviewed source differs from ledger official representation")
    image = DATA / "frames" / f"{new_index:06d}.png"
    if image.exists():
        raise ValueError("New source PNG already exists; refuse to replace unseen source")
    if not cv2.imwrite(str(image), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise OSError(image)
    if not np.array_equal(bgr, cv2.imread(str(image))):
        raise ValueError("Reviewed native PNG roundtrip changed pixels")
    binding = dict(schema="rinkspace.exact-single-native-source-frame.v1", event_id="ankara_2026",
                   programme_id="okada_mei_sp", shot_index=2,
                   programme_frame_index_1based=new_index, native_frame_index_1based=new_pts // 256 + 1,
                   native_frame_index_0based=new_pts // 256, native_pts=new_pts, actual_pts=new_pts,
                   time_base="1/12800", source_time=timestamp(new_pts), source_time_seconds=new_pts / 12800,
                   source=source_record["path"], source_video_sha256=source_record["sha256"],
                   image_path=str(image), png_sha256=file_sha(image), decoded_BGR_sha256=array_sha(bgr),
                   decoded_RGB_sha256=array_sha(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
                   native_hw=[1080, 1920], original_PTS_verified=True, PNG_roundtrip_lossless_verified=True,
                   optimization_performed=False, ledger_official_representation=representation,
                   ledger_official_representation_same_frame_BGR_identical=True)
    binding_path = DATA / "source_bindings" / f"{new_index:06d}.json"
    write(binding_path, binding)
    metric = next(r for r in selection["metrics"] if r["native_pts"] == new_pts)
    new_row = dict(index=new_index, source=binding, image_path=str(image), path=str(image),
                   native_pts=new_pts, source_pts=new_pts, source_time_base="1/12800",
                   source_time=timestamp(new_pts), source_time_seconds=new_pts / 12800,
                   native_frame_index_0based=new_pts // 256, png_sha256=binding["png_sha256"],
                   decoded_native_BGR_sha256=binding["decoded_BGR_sha256"],
                   decoded_native_RGB_sha256=binding["decoded_RGB_sha256"], native_hw=[1080, 1920],
                   exact_native_PTS_and_lossless_PNG_verified=True, selection_metric=metric,
                   ledger_official_representation_same_frame_BGR_identical=True,
                   source_binding=dict(path=str(binding_path), sha256=file_sha(binding_path)),
                   mask_pending=True)
    rows.append(new_row)
    rows.sort(key=lambda r: r["native_pts"])
    archive_ref = mapping["model_seed"]["inherited_camera_archive"]
    if file_sha(archive_ref["path"]) != archive_ref["sha256"]:
        raise ValueError("Inherited camera archive changed")
    with np.load(archive_ref["path"], allow_pickle=False) as saved:
        cameras = {k: saved[k].copy() for k in saved.files}
    ledger_ref = mapping["mapping_evidence"]
    ledger = [dict(json.loads(line), line_1based=i + 1) for i, line in enumerate(Path(ledger_ref["path"]).read_text("utf8").splitlines())]
    matches = [r for r in ledger if r["shot_index"] == 2 and r["match_status"] == "unique_pixel_match"
               and r.get("accepted_official_pts") == new_pts]
    ledger_row = min(matches, key=lambda r: r["full_pts"])
    lookup = {int(p): i for i, p in enumerate(cameras["pts"])}
    archive_index = lookup[int(ledger_row["full_pts"])]
    new_mapping = dict(sequence_index_1based=new_index, programme_frame_index_1based=new_index,
                       official_pts=new_pts, official_time_base="1/12800", official_time_seconds=new_pts / 12800,
                       legacy60_pts=ledger_row["full_pts"], legacy60_time_base="1/15360", legacy60_shot_index=2,
                       mapping_status="confirmed_pixel_match", pixel_correspondence_confirmed=True,
                       navigation_only=True, ledger_evidence=ledger_row,
                       inherited_camera_archive_row_0based=archive_index,
                       primary_official50_vs_ledger_official50_BGR_identical=True,
                       camera25_float64_sha256=array_sha(cameras["camera25"][archive_index]))
    map_by_index = {r["programme_frame_index_1based"]: deepcopy(r) for r in mapping["frames"]}
    map_by_index[new_index] = new_mapping
    map_rows = [map_by_index[r["index"]] for r in rows]
    for ordinal, r in enumerate(map_rows):
        r["camera_seed_index_0based"] = ordinal
    ai = np.asarray([r["inherited_camera_archive_row_0based"] for r in map_rows], np.int64)
    indices = np.asarray([r["index"] for r in rows], np.int64)
    pts_list = [r["native_pts"] for r in rows]
    np.savez(DATA / "camera_seeds.npz", programme_indices=indices, pts=np.asarray(pts_list, np.int64),
             legacy60_pts=np.asarray([r["legacy60_pts"] for r in map_rows], np.int64),
             inherited_camera_archive_indices=ai, camera25=cameras["camera25"][ai],
             installation_id=cameras["installation_id"][ai], view_state4=cameras["view_state4"][ai],
             exposure_representative_pts=cameras["exposure_representative_pts"][ai])
    from calibration.program_tracking.render_result import SceneRenderer
    renderer = SceneRenderer(read(mapping["model_seed"]["scene"]["path"]), draw_grid=True, draw_advertisements=True)
    for row, mapped, camera in zip(rows, map_rows, cameras["camera25"][ai]):
        raw = cv2.imread(row["image_path"])
        mask_path = row.get("mask_path")
        mask = cv2.imread(mask_path, 0) > 0 if mask_path else np.zeros(raw.shape[:2], bool)
        if mask_path and file_sha(mask_path) != row["mask_sha256"]:
            raise ValueError("Existing source mask changed")
        overlay, _ = renderer.draw(raw, camera, person_mask=mask, title=None)
        if not np.array_equal(overlay[mask], raw[mask]):
            raise ValueError("Navigation overwrote an existing person exclusion mask")
        navigation = DATA / "navigation" / f"{row['index']:06d}.png"
        if not cv2.imwrite(str(navigation), overlay, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise OSError(navigation)
        row["navigation_path"] = str(navigation)
        mapped["navigation_overlay"] = dict(path=str(navigation), sha256=file_sha(navigation),
                                            navigation_only=True, current_source_background=True,
                                            current_person_mask_applied=bool(mask_path),
                                            current_mask_sha256=row.get("mask_sha256"), camera_seed_refitted=False,
                                            all_original_feature_paths_and_advertisement_ids_enabled=True,
                                            body_or_trajectory_drawn=False, path_counts=renderer.path_counts)
    gap = dict(first_clear_pts=581888, next_clear_pts=623872,
               first_clear_source_time="00:00:45.460", next_clear_source_time="00:00:48.740",
               duration_seconds=3.28, reason="fast camera pan causes strong horizontal board-advertisement motion blur; no independently clear source detail found in expanded candidate review",
               source_evidence_missing=True, inferred_points_added=False)
    review = dict(authority="root source-pixel visual review and current user requirement to avoid strong motion blur",
                  replaced=dict(previous_index=1288, previous_pts=585728, new_index=1273, new_pts=581888),
                  removed_indices=[1324, 1358, 1394], reason="strong horizontal motion blur on rink and advertising material",
                  existing_source_PNGs_masks_and_bindings_deleted=False,
                  historical_selection=dict(path=str(history_path), sha256=file_sha(history_path)),
                  clear_evidence_gaps=[gap], candidate_contact_sheet=str(DATA / "early_motion_candidate_board_crops.png"))
    selection.update(selected_frame_count=13, selected_pts=pts_list, programme_indices=indices.tolist(),
                     selected_interval_seconds=(np.diff(pts_list) / 12800).tolist(),
                     requested_interval_is_approximate=True, clarity_review=review,
                     windows_role="historical automatic candidates retained; current selected_pts govern", clear_evidence_gaps=[gap])
    write(DATA / "selection.json", selection)
    verification["records"] = [r for r in verification["records"] if r["native_pts"] in pts_list]
    verification["records"].append(dict(native_pts=new_pts, decoded_BGR_sha256=array_sha(bgr), decoded_pixels_identical=True))
    verification["records"].sort(key=lambda r: r["native_pts"])
    verification["selected_frame_count"] = 13
    verification["clarity_review"] = review
    if len(verification["records"]) != 13:
        raise ValueError("Selected source representation verification is incomplete")
    write(DATA / "official_representation_verification.json", verification)
    mapping["frames"] = map_rows
    mapping["summary"] = {"confirmed_pixel_match": 13}
    mapping["clarity_review"] = review
    mapping["model_seed"]["camera_seeds"] = dict(path=str(DATA / "camera_seeds.npz"), sha256=file_sha(DATA / "camera_seeds.npz"))
    mapping["model_seed"]["camera25_shape"] = [13, 25]
    mapping["official_representation_verification"] = dict(path=str(DATA / "official_representation_verification.json"),
                                                          sha256=file_sha(DATA / "official_representation_verification.json"))
    write(DATA / "navigation_mapping.json", mapping)
    manifest.update(programme_indices=indices.tolist(), frame_count=13, frames=rows, clarity_review=review,
                    selected_frames_only_saved=False, historical_source_caches_retained=True,
                    navigation_mapping=dict(path=str(DATA / "navigation_mapping.json"), sha256=file_sha(DATA / "navigation_mapping.json")),
                    selection=dict(path=str(DATA / "selection.json"), sha256=file_sha(DATA / "selection.json")))
    write(DATA / "frame_manifest.json", manifest)
    contact_sheet(rows, "image_path", DATA / "source_contact_sheet.png")
    contact_sheet(rows, "navigation_path", DATA / "navigation_contact_sheet.png")
    print(json.dumps(dict(stage="reviewed_clear_selection_ready", frames=13, programme_indices=indices.tolist(),
                          single_missing_mask_index=1273, clear_evidence_gap_seconds=3.28,
                          all_selected_current_vs_ledger_BGR_identical=True,
                          existing_masks_preserved=True)), flush=True)
    return manifest


def refresh_navigation():
    """Refresh identity-navigation images using current per-frame masks."""
    manifest = read(DATA / "frame_manifest.json")
    mapping = read(DATA / "navigation_mapping.json")
    with np.load(DATA / "camera_seeds.npz", allow_pickle=False) as saved:
        if saved["programme_indices"].tolist() != manifest["programme_indices"]:
            raise ValueError("Navigation camera order differs from current source manifest")
        cameras = saved["camera25"].copy()
    from calibration.program_tracking.render_result import SceneRenderer
    renderer = SceneRenderer(read(mapping["model_seed"]["scene"]["path"]), draw_grid=True, draw_advertisements=True)
    for row, mapped, camera in zip(manifest["frames"], mapping["frames"], cameras):
        if row["index"] != mapped["programme_frame_index_1based"]:
            raise ValueError("Navigation source mapping order differs")
        if file_sha(row["image_path"]) != row["png_sha256"]:
            raise ValueError("Navigation source PNG changed")
        raw = cv2.imread(row["image_path"])
        if array_sha(raw) != row["decoded_native_BGR_sha256"]:
            raise ValueError("Navigation source pixels changed")
        if file_sha(row["mask_path"]) != row["mask_sha256"]:
            raise ValueError("Current source mask changed")
        mask = cv2.imread(row["mask_path"], 0) > 0
        overlay, _ = renderer.draw(raw, camera, person_mask=mask, title=None)
        if not np.array_equal(overlay[mask], raw[mask]):
            raise ValueError("Navigation overwrote current person mask")
        path = DATA / "navigation" / f"{row['index']:06d}.png"
        if not cv2.imwrite(str(path), overlay, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise OSError(path)
        row["navigation_path"] = str(path)
        row.pop("mask_pending", None)
        mapped["navigation_overlay"].update(path=str(path), sha256=file_sha(path),
                                             current_person_mask_applied=True,
                                             current_mask_sha256=row["mask_sha256"])
    write(DATA / "navigation_mapping.json", mapping)
    manifest["navigation_mapping"] = dict(path=str(DATA / "navigation_mapping.json"), sha256=file_sha(DATA / "navigation_mapping.json"))
    manifest["selected_frames_only_saved"] = False
    manifest["historical_source_caches_retained"] = True
    write(DATA / "frame_manifest.json", manifest)
    contact_sheet(manifest["frames"], "navigation_path", DATA / "navigation_contact_sheet.png")
    print(json.dumps(dict(stage="current_mask_navigation_refreshed", frames=len(cameras))), flush=True)


def restore_cadence_selection():
    """Use the latest user's hard 0.5–1 sec coverage, retaining blur evidence."""
    manifest = read(DATA / "frame_manifest.json")
    selection = read(DATA / "selection.json")
    mapping = read(DATA / "navigation_mapping.json")
    verification = read(DATA / "official_representation_verification.json")
    wanted_indices = [1249, 1274, 1314, 1358, 1394, 1437, 1469, 1518,
                      1552, 1589, 1624, 1670, 1707, 1746, 1783, 1811]
    if manifest["programme_indices"] == wanted_indices and manifest.get("latest_cadence_review"):
        print(json.dumps(dict(stage="latest_cadence_selection_already_applied", frames=16)), flush=True)
        return manifest
    # These are exact prior records, including the rejected gap, rather than
    # new fitting inputs. Current frames/masks/source bindings stay in place.
    for original, history in (("selection.json", "historical_selection_13_clear.json"),
                              ("frame_manifest.json", "historical_manifest_13_clear.json")):
        target = DATA / history
        if not target.exists():
            target.write_bytes((DATA / original).read_bytes())
    source_record = manifest["source"]
    representation = verification["ledger_official_representation"]
    source_rows = {r["index"]: deepcopy(r) for r in manifest["frames"]}
    metric_by_pts = {r["native_pts"]: r for r in selection["metrics"]}
    verified = []
    for index in wanted_indices:
        pts = PROGRAMME_FIRST_PTS + (index - 1) * 256
        image = DATA / "frames" / f"{index:06d}.png"
        binding_path = DATA / "source_bindings" / f"{index:06d}.json"
        if index in source_rows:
            row = source_rows[index]
        else:
            if image.exists() and binding_path.exists():
                binding = read(binding_path)
                if binding["native_pts"] != pts or file_sha(image) != binding["png_sha256"]:
                    raise ValueError("Preserved source cache identity changed")
                bgr = cv2.imread(str(image))
                if array_sha(bgr) != binding["decoded_BGR_sha256"]:
                    raise ValueError("Preserved source cache pixels changed")
            else:
                bgr = next(decoded(Path(source_record["path"]), pts, pts + 256))[1]
                if not cv2.imwrite(str(image), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    raise OSError(image)
                if not np.array_equal(bgr, cv2.imread(str(image))):
                    raise ValueError("New hard-cadence PNG roundtrip changed pixels")
                binding = dict(schema="rinkspace.exact-single-native-source-frame.v1", event_id="ankara_2026",
                               programme_id="okada_mei_sp", shot_index=2,
                               programme_frame_index_1based=index, native_frame_index_1based=pts // 256 + 1,
                               native_frame_index_0based=pts // 256, native_pts=pts, actual_pts=pts,
                               time_base="1/12800", source_time=timestamp(pts), source_time_seconds=pts / 12800,
                               source=source_record["path"], source_video_sha256=source_record["sha256"],
                               image_path=str(image), png_sha256=file_sha(image), decoded_BGR_sha256=array_sha(bgr),
                               decoded_RGB_sha256=array_sha(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
                               native_hw=[1080, 1920], original_PTS_verified=True,
                               PNG_roundtrip_lossless_verified=True, optimization_performed=False,
                               ledger_official_representation=representation)
            other = next(decoded(Path(representation["path"]), pts, pts + 256))[1]
            if not np.array_equal(bgr, other):
                raise ValueError("Hard-cadence source differs from ledger official exposure")
            binding["ledger_official_representation_same_frame_BGR_identical"] = True
            if not binding_path.exists():
                write(binding_path, binding)
            row = dict(index=index, source=binding, image_path=str(image), path=str(image),
                       native_pts=pts, source_pts=pts, source_time_base="1/12800",
                       source_time=timestamp(pts), source_time_seconds=pts / 12800,
                       native_frame_index_0based=pts // 256, png_sha256=binding["png_sha256"],
                       decoded_native_BGR_sha256=binding["decoded_BGR_sha256"],
                       decoded_native_RGB_sha256=binding["decoded_RGB_sha256"], native_hw=[1080, 1920],
                       exact_native_PTS_and_lossless_PNG_verified=True, selection_metric=metric_by_pts[pts],
                       ledger_official_representation_same_frame_BGR_identical=True,
                       source_binding=dict(path=str(binding_path), sha256=file_sha(binding_path)))
            mask_path = DATA / "masks_current" / f"{index:06d}.png"
            if mask_path.exists():
                row.update(mask_path=str(mask_path), mask_sha256=file_sha(mask_path))
            else:
                row["mask_pending"] = True
            source_rows[index] = row
        if file_sha(row["image_path"]) != row["png_sha256"]:
            raise ValueError("Current source PNG changed during hard-cadence review")
        verified.append(dict(native_pts=pts, decoded_BGR_sha256=row["decoded_native_BGR_sha256"],
                             decoded_pixels_identical=True))
    rows = [source_rows[i] for i in wanted_indices]
    pts_list = [r["native_pts"] for r in rows]
    gaps = np.diff(pts_list) / 12800
    if not np.all((gaps >= .5) & (gaps <= 1.)):
        raise ValueError("Latest hard-cadence coverage failed")
    ledger_ref = mapping["mapping_evidence"]
    ledger = [dict(json.loads(line), line_1based=i + 1) for i, line in enumerate(Path(ledger_ref["path"]).read_text("utf8").splitlines())]
    accepted = {}
    for r in ledger:
        if r["shot_index"] == 2 and r["match_status"] == "unique_pixel_match":
            accepted.setdefault(r["accepted_official_pts"], []).append(r)
    archive_ref = mapping["model_seed"]["inherited_camera_archive"]
    if file_sha(archive_ref["path"]) != archive_ref["sha256"]:
        raise ValueError("Inherited camera archive changed")
    with np.load(archive_ref["path"], allow_pickle=False) as saved:
        cameras = {k: saved[k].copy() for k in saved.files}
    lookup = {int(p): i for i, p in enumerate(cameras["pts"])}
    old_mapping = {r["programme_frame_index_1based"]: deepcopy(r) for r in mapping["frames"]}
    map_rows = []
    for ordinal, row in enumerate(rows):
        if row["index"] in old_mapping:
            mapped = old_mapping[row["index"]]
        else:
            ledger_row = min(accepted[row["native_pts"]], key=lambda r: r["full_pts"])
            ai = lookup[ledger_row["full_pts"]]
            mapped = dict(sequence_index_1based=row["index"], programme_frame_index_1based=row["index"],
                          official_pts=row["native_pts"], official_time_base="1/12800",
                          official_time_seconds=row["native_pts"] / 12800,
                          legacy60_pts=ledger_row["full_pts"], legacy60_time_base="1/15360", legacy60_shot_index=2,
                          mapping_status="confirmed_pixel_match", pixel_correspondence_confirmed=True,
                          navigation_only=True, ledger_evidence=ledger_row,
                          inherited_camera_archive_row_0based=ai,
                          primary_official50_vs_ledger_official50_BGR_identical=True,
                          camera25_float64_sha256=array_sha(cameras["camera25"][ai]))
        mapped["camera_seed_index_0based"] = ordinal
        map_rows.append(mapped)
    ai = np.asarray([r["inherited_camera_archive_row_0based"] for r in map_rows], np.int64)
    np.savez(DATA / "camera_seeds.npz", programme_indices=np.asarray(wanted_indices, np.int64),
             pts=np.asarray(pts_list, np.int64), legacy60_pts=np.asarray([r["legacy60_pts"] for r in map_rows], np.int64),
             inherited_camera_archive_indices=ai, camera25=cameras["camera25"][ai],
             installation_id=cameras["installation_id"][ai], view_state4=cameras["view_state4"][ai],
             exposure_representative_pts=cameras["exposure_representative_pts"][ai])
    from calibration.program_tracking.render_result import SceneRenderer
    renderer = SceneRenderer(read(mapping["model_seed"]["scene"]["path"]), draw_grid=True, draw_advertisements=True)
    for row, mapped, camera in zip(rows, map_rows, cameras["camera25"][ai]):
        raw = cv2.imread(row["image_path"])
        mask_path = row.get("mask_path")
        mask = cv2.imread(mask_path, 0) > 0 if mask_path else np.zeros(raw.shape[:2], bool)
        if mask_path and file_sha(mask_path) != row["mask_sha256"]:
            raise ValueError("Current source mask changed")
        overlay, _ = renderer.draw(raw, camera, person_mask=mask, title=None)
        navigation = DATA / "navigation" / f"{row['index']:06d}.png"
        if not cv2.imwrite(str(navigation), overlay, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise OSError(navigation)
        row["navigation_path"] = str(navigation)
        mapped["navigation_overlay"] = dict(path=str(navigation), sha256=file_sha(navigation),
                                            navigation_only=True, current_source_background=True,
                                            current_person_mask_applied=bool(mask_path),
                                            current_mask_sha256=row.get("mask_sha256"), camera_seed_refitted=False,
                                            all_original_feature_paths_and_advertisement_ids_enabled=True,
                                            body_or_trajectory_drawn=False, path_counts=renderer.path_counts)
    review = dict(authority="latest explicit user instruction: every 0.5–1sec requires a selected exposure; select relatively clearest available pixels and retain residual blur evidence",
                  cadence_interval_is_hard=True, current_selection_pts=pts_list,
                  relative_clarity_visual_review_source=str(DATA / "early_motion_candidate_board_crops.png"),
                  motion_blur_limitations=dict(source_time_range=["00:00:45.480", "00:00:47.880"],
                                               affected_indices=[1274, 1314, 1358, 1394],
                                               description="rapid camera pan leaves horizontal board-advertisement motion blur; these are relatively clearest candidates within required coverage, not clean text-boundary evidence"),
                  historical_clear_subset=dict(path=str(DATA / "historical_selection_13_clear.json"),
                                               sha256=file_sha(DATA / "historical_selection_13_clear.json")),
                  historical_source_PNGs_masks_bindings_retained=True)
    selection.update(selected_frame_count=16, selected_pts=pts_list, programme_indices=wanted_indices,
                     selected_interval_seconds=gaps.tolist(), requested_interval_is_approximate=False,
                     requested_interval_is_hard=True, clear_evidence_gaps=[], latest_cadence_review=review,
                     windows_role="initial automatic metric windows retained as historical diagnostics; latest selected_pts govern",
                     uncovered_start_seconds=(pts_list[0] - START) / 12800,
                     uncovered_end_seconds=(LAST - pts_list[-1]) / 12800)
    selection.pop("clarity_review", None)
    write(DATA / "selection.json", selection)
    verification.update(selected_frame_count=16, records=verified, latest_cadence_review=review)
    verification.pop("clarity_review", None)
    write(DATA / "official_representation_verification.json", verification)
    mapping.update(frames=map_rows, summary={"confirmed_pixel_match": 16}, latest_cadence_review=review)
    mapping.pop("clarity_review", None)
    mapping["model_seed"]["camera_seeds"] = dict(path=str(DATA / "camera_seeds.npz"), sha256=file_sha(DATA / "camera_seeds.npz"))
    mapping["model_seed"]["camera25_shape"] = [16, 25]
    mapping["official_representation_verification"] = dict(path=str(DATA / "official_representation_verification.json"), sha256=file_sha(DATA / "official_representation_verification.json"))
    write(DATA / "navigation_mapping.json", mapping)
    manifest.update(programme_indices=wanted_indices, frame_count=16, frames=rows,
                    latest_cadence_review=review, selected_frames_only_saved=False,
                    historical_source_caches_retained=True,
                    selection=dict(path=str(DATA / "selection.json"), sha256=file_sha(DATA / "selection.json")),
                    navigation_mapping=dict(path=str(DATA / "navigation_mapping.json"), sha256=file_sha(DATA / "navigation_mapping.json")))
    manifest.pop("clarity_review", None)
    write(DATA / "frame_manifest.json", manifest)
    contact_sheet(rows, "image_path", DATA / "source_contact_sheet.png")
    contact_sheet(rows, "navigation_path", DATA / "navigation_contact_sheet.png")
    print(json.dumps(dict(stage="hard_cadence_selection_ready", frames=16, programme_indices=wanted_indices,
                          intervals=gaps.tolist(), pending_masks=[r["index"] for r in rows if "mask_path" not in r],
                          preserved_existing_masks=True)), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--apply-reviewed-selection", action="store_true")
    parser.add_argument("--navigation-only", action="store_true")
    parser.add_argument("--restore-cadence-selection", action="store_true")
    args = parser.parse_args()
    if args.restore_cadence_selection:
        restore_cadence_selection()
    elif args.navigation_only:
        refresh_navigation()
    elif args.apply_reviewed_selection:
        apply_reviewed_selection()
    else:
        prepare(args.model, args.count)
