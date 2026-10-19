"""Conservative, source-pixel-only board-surface support for native LK points.

This is a diagnostic sidecar.  It neither changes the original motion pixels nor
fills missing board-edge samples.  A blue top rail and a locally continuous
yellow kickplate must both be visible in the same original frame; unsupported
columns remain unknown.  The color peak is used to bound an *interior* board
region, not as a precise ice-board-contact measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


SCHEMA = "rinkspace-source-board-support-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_y(points: np.ndarray, x: float, *, max_gap: float = 56.) -> float:
    """Use nearby *measured* source points only; a missing span stays missing."""
    if not len(points):
        return float("nan")
    ordered = points[np.argsort(points[:, 0])]
    xs = ordered[:, 0]
    i = int(np.searchsorted(xs, x))
    if i < len(xs) and abs(xs[i] - x) < 1e-6:
        return float(ordered[i, 1])
    if i == 0 or i == len(xs):
        return float("nan")
    a, b = ordered[i - 1], ordered[i]
    if b[0] - a[0] > max_gap:
        return float("nan")
    return float(a[1] + (x - a[0]) * (b[1] - a[1]) / (b[0] - a[0]))


def _candidate_at_column(image: np.ndarray, score: np.ndarray,
                         exclude: np.ndarray, x: int, top_y: float
                         ) -> tuple[float, float, float] | None:
    """Find an observed narrow yellow-to-neutral transition below the top."""
    h = image.shape[0]
    low = max(151, int(np.ceil(top_y + 45)))
    high = min(h - 51, int(np.floor(top_y + 300)))
    if high <= low:
        return None
    candidates = []
    for y in range(low, high):
        value = float(score[y, x])
        if value < 9. or value < score[y - 1, x] or value < score[y + 1, x]:
            continue
        if np.any(exclude[y - 5:y + 13, max(0, x - 3):x + 4]):
            continue
        below = score[y + 17:y + 37, x]
        if len(below) < 20 or value - float(np.median(below)) < 6.:
            continue
        # The ice below the kickplate is largely neutral.  This prevents a
        # yellow letter inside an advertisement from defining a board foot.
        below_bgr = image[y + 22:y + 42, x].astype(np.float32)
        if (float(np.median(np.abs(below))) > 10.
                or float(np.median(np.ptp(below_bgr, axis=1))) > 25.
                or float(np.median(below_bgr)) < 100.):
            continue
        cutoff = max(5., .5 * value)
        falls = np.flatnonzero(score[y + 2:y + 31, x] < cutoff)
        if not len(falls):
            continue
        if value < 22.:
            above = image[y - 31:y - 9, x].astype(np.float32)
            white = ((above.min(axis=1) >= 140.)
                     & (np.ptp(above, axis=1) <= 35.))
            if np.count_nonzero(white) < 5:
                continue
        foot_y = float(y + 2 + falls[0])
        candidates.append((foot_y, value, value - float(np.median(below))))
    # The physical kickplate is the lowest qualifying stripe before the ice.
    return max(candidates, key=lambda item: item[0]) if candidates else None


def measure_local_support(image: np.ndarray, top_uv: np.ndarray,
                          exclude: np.ndarray | None = None,
                          *, step: int = 14) -> dict:
    """Return only locally supported board-interior strips in source pixels.

    No camera or scene is an input.  A run must contain four successive source
    columns; x gaps, foreground, abrupt y jumps, and weak color remain holes.
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected original uint8 BGR image")
    top_uv = np.asarray(top_uv, np.float64)
    if top_uv.ndim != 2 or top_uv.shape[1] != 2 or not np.isfinite(top_uv).all():
        raise ValueError("Invalid measured top-edge UV")
    if exclude is None:
        exclude = np.zeros(image.shape[:2], bool)
    else:
        exclude = np.asarray(exclude, bool)
        if exclude.shape != image.shape[:2]:
            raise ValueError("Mask/image size mismatch")
    smooth = cv2.GaussianBlur(image, (5, 5), .9).astype(np.float32)
    b, g, r = cv2.split(smooth)
    score = .5 * (r + g) - b
    raw = []
    for x in range(16, image.shape[1] - 16, step):
        top_y = _local_y(top_uv, x)
        if not np.isfinite(top_y):
            continue
        candidate = _candidate_at_column(image, score, exclude, x, top_y)
        if candidate is not None:
            foot_y, strength, prominence = candidate
            raw.append((float(x), float(top_y), foot_y,
                        float(strength), float(prominence)))
    raw = np.asarray(raw, np.float64).reshape(-1, 5)
    if not len(raw):
        return {"raw": raw, "accepted": raw.copy(), "runs": []}
    runs = []
    start = 0
    for i in range(1, len(raw) + 1):
        connected = (i < len(raw) and raw[i, 0] - raw[i - 1, 0] <= step + 1
                     and abs(raw[i, 2] - raw[i - 1, 2]) <= 8.)
        if connected:
            continue
        if i - start >= 4:
            runs.append((start, i))
        start = i
    accepted = np.concatenate([raw[a:b] for a, b in runs]) if runs else raw[:0]
    return {"raw": raw, "accepted": accepted, "runs": runs}


def classify_points(xy: np.ndarray, support: dict,
                    exclude: np.ndarray | None = None,
                    *, interior_margin: float = 16.) -> np.ndarray:
    """Mark original UV inside a directly bounded board strip; unknown=False."""
    xy = np.asarray(xy, np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("Expected N×2 original UV")
    found = np.zeros(len(xy), bool)
    foot = support["accepted"]
    if not len(foot):
        return found
    for i, (x, y) in enumerate(xy):
        if not np.isfinite([x, y]).all():
            continue
        j = int(np.argmin(np.abs(foot[:, 0] - x)))
        if abs(foot[j, 0] - x) > 7.:
            continue
        top_y, foot_y = foot[j, 1:3]
        if top_y + interior_margin < y < foot_y - interior_margin:
            ix, iy = int(round(x)), int(round(y))
            if exclude is not None and (iy < 0 or iy >= exclude.shape[0]
                                        or ix < 0 or ix >= exclude.shape[1]
                                        or exclude[iy, ix]):
                continue
            found[i] = True
    return found


def _source_edges_at(edge_cache: Path, pts: int) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(edge_cache, allow_pickle=False) as z:
        matches = np.flatnonzero(z["pts"] == pts)
        if len(matches) != 1:
            raise ValueError(f"PTS {pts} absent/ambiguous in source-edge cache")
        i = int(matches[0])
        sl = slice(int(z["offsets"][i]), int(z["offsets"][i + 1]))
        uv, edge_id = z["observed_uv"][sl], z["edge_id"][sl]
        top, contact = uv[edge_id == 0], uv[edge_id == 1]
        meta = {key: str(z[key]) for key in
                ("source_sha256", "model_sha256", "track_sha256", "detector_sha256")}
    return top, contact, meta


def _motion_points(motion_cache: Path, pts: int) -> np.ndarray:
    with np.load(motion_cache, allow_pickle=False) as z:
        pairs = z["pair_pts"]
        frames = []
        for side, key in ((0, "xy0"), (1, "xy1")):
            for i in np.flatnonzero(pairs[:, side] == pts):
                sl = slice(int(z["offsets"][i]), int(z["offsets"][i + 1]))
                frames.append(z[key][sl][z["region_hint"][sl] == 0])
    return np.concatenate(frames) if frames else np.empty((0, 2), np.float32)


def _selected_rows(z, pair_index: int, max_matches: int) -> tuple[int, np.ndarray]:
    """Reproduce the frozen v13 selection before adding a source-only gate."""
    begin, end = map(int, z["offsets"][pair_index:pair_index + 2])
    xy0, xy1 = z["xy0"][begin:end], z["xy1"][begin:end]
    fb = z["fb_error_px"][begin:end]
    good = (z["homography_inlier"][begin:end].astype(bool) & (fb < 2.5)
            & np.isfinite(xy0).all(axis=1) & np.isfinite(xy1).all(axis=1))
    indices = np.flatnonzero(good)
    if len(indices) < 8:
        return begin, np.empty(0, np.int32)
    # Exact stable grid occupancy order of track_sequence._distributed_indices.
    points, score = xy0[indices], fb[indices]
    ordered = np.argsort(score, kind="stable")
    col = np.clip((points[:, 0] * 6 / 1920).astype(int), 0, 5)
    row = np.clip((points[:, 1] * 4 / 1080).astype(int), 0, 3)
    occupancy = np.zeros((4, 6), np.int16)
    selected = []
    for local in ordered:
        r, c = row[local], col[local]
        if occupancy[r, c] >= max(2, int(np.ceil(max_matches / 24))):
            continue
        occupancy[r, c] += 1
        selected.append(int(local))
        if len(selected) == max_matches:
            break
    if len(selected) < max_matches:
        picked = set(selected)
        selected.extend(int(local) for local in ordered if local not in picked)
        selected = selected[:max_matches]
    return begin, indices[np.asarray(selected, np.int32)].astype(np.int32)


def _contiguous_runs(pts: list[int]) -> list[tuple[int, int]]:
    if not pts:
        return []
    result = []
    first = previous = pts[0]
    for stamp in pts[1:]:
        if stamp - previous > 256:
            result.append((first, previous))
            first = stamp
        previous = stamp
    result.append((first, previous))
    return result


def _decode_native_runs(source: Path, pts: list[int]):
    """Bounded exact-PTS seeks: one short sequential decode per target run."""
    import av

    wanted = set(pts)
    emitted = set()
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        if (stream.time_base.numerator, stream.time_base.denominator) != (1, 15360):
            raise ValueError("Native source time base changed")
        for start, end in _contiguous_runs(pts):
            container.seek(max(0, start - 15360), stream=stream, backward=True)
            for frame in container.decode(stream):
                stamp = frame.pts
                if stamp is None or stamp < start:
                    continue
                if stamp > end:
                    break
                if stamp in wanted:
                    if stamp in emitted:
                        raise ValueError(f"Duplicate decoded PTS {stamp}")
                    emitted.add(stamp)
                    yield int(stamp), frame.to_ndarray(format="bgr24")
    if emitted != wanted:
        raise ValueError(f"Native decode missed PTS: {sorted(wanted - emitted)[:8]}")


def _source_mask(mask: np.ndarray) -> np.ndarray:
    """Exclude existing SAM2 people and the known burned-in graphics pixels."""
    exclude = np.asarray(mask, bool).copy()
    # Broadcast graphic rectangles verified on this native source.  They are
    # exclusions, never positive board cues; per-UV masks are checked again.
    exclude[24:253, 28:554] = True
    exclude[944:1000, 1690:1865] = True
    return exclude


def build_missing_strict_sidecar(strict_manifest: Path, pair_summary: Path,
                                 source_shots: Path, edge_cache: Path,
                                 motion_cache: Path, output: Path,
                                 *, shot_index: int = 5) -> dict:
    """Classify existing selected hint0 endpoints, never generate new matches."""
    try:
        from .render_result import DEFAULT_MASKS, DEFAULT_PEOPLE, PersonMasks
    except ImportError:
        from render_result import DEFAULT_MASKS, DEFAULT_PEOPLE, PersonMasks

    manifest = json.loads(strict_manifest.read_text(encoding="utf-8"))
    rows = json.loads(pair_summary.read_text(encoding="utf-8"))
    shots = json.loads(source_shots.read_text(encoding="utf-8"))
    source_hash = shots["source_sha256_from_timeline"]
    if source_hash != manifest["source_sha256"]:
        raise ValueError("Source SHA differs from frozen strict manifest")
    if (str(Path(manifest["pair_summary_path"]).resolve()) != str(pair_summary.resolve())
            or manifest["pair_summary_sha256"] != _sha256(pair_summary)):
        raise ValueError("Frozen pair summary differs")
    files = [item for item in manifest["source_files"]
             if item["kind"] == "native" and item["shot_index"] == shot_index]
    if len(files) != 1:
        raise ValueError("Expected one frozen native source for requested shot")
    source_file = files[0]
    if (Path(source_file["path"]).resolve() != motion_cache.resolve()
            or source_file["sha256"] != _sha256(motion_cache)):
        raise ValueError("Native motion cache differs from frozen source")
    with np.load(edge_cache, allow_pickle=False) as z:
        if str(z["source_sha256"]) != source_hash:
            raise ValueError("Source-edge cache differs from native video")
        edge_pts = z["pts"].copy()
        edge_offsets = z["offsets"].copy()
        edge_uv = z["observed_uv"].copy()
        edge_ids = z["edge_id"].copy()
        edge_meta = {key: str(z[key]) for key in
                     ("source_sha256", "model_sha256", "track_sha256", "detector_sha256")}
    # The edge detector was guided by a saved camera-track cache, whereas
    # source_file is the independent original LK motion cache.  Their hashes
    # must be recorded separately, never equated.

    selected_rows = [row for row in rows if row["kind"] == "native"
                     and row["shot_index"] == shot_index
                     and row["strict_board_count"] < 8]
    if not selected_rows:
        raise ValueError("No missing-strict native pairs in requested shot")
    selected_rows.sort(key=lambda row: row["source_pair_index"])
    target_pts = sorted({int(stamp) for row in selected_rows for stamp in row["pts"]})
    pts_to_edge_index = {int(stamp): i for i, stamp in enumerate(edge_pts)}
    if any(stamp not in pts_to_edge_index for stamp in target_pts):
        raise ValueError("Missing exact PTS in source-edge cache")
    with np.load(motion_cache, allow_pickle=False) as z:
        motion = {key: z[key].copy() for key in
                  ("pair_pts", "offsets", "xy0", "xy1", "fb_error_px",
                   "homography_inlier", "region_hint")}
    pair_buffers = {}
    incident: dict[int, list[tuple[int, int]]] = {stamp: [] for stamp in target_pts}
    for row in selected_rows:
        i = int(row["source_pair_index"])
        if (row["source_file_id"] != source_file["id"]
                or list(map(int, motion["pair_pts"][i])) != row["pts"]):
            raise ValueError("Frozen pair index/PTS/source identity mismatch")
        start, selected = _selected_rows(motion, i, source_file["max_matches"])
        if len(selected) != row["selected_v13_pixels"]:
            raise ValueError("Frozen v13 selected-row count mismatch")
        selected = selected[motion["region_hint"][start + selected] == 0]
        pair_buffers[i] = {"row": row, "start": start, "selected": selected,
                           "source_ok": None, "target_ok": None}
        incident[int(row["pts"][0])].append((i, 0))
        incident[int(row["pts"][1])].append((i, 1))

    source_path = Path(shots["source_path"])
    # Verify the actual bytes once; all decoded PTS below come from this file.
    if _sha256(source_path) != source_hash:
        raise ValueError("Original source video bytes differ from timeline SHA")
    people = PersonMasks(DEFAULT_MASKS, DEFAULT_PEOPLE)
    frame_rows = []
    raw_parts, accepted_parts = [], []
    raw_offsets, accepted_offsets = [0], [0]
    qa_pts = set(target_pts) & {230000384, 230009344, 230025984,
                                230044672, 230047488, 230360576, 230361600}
    qa_images: dict[int, np.ndarray] = {}
    for number, (stamp, image) in enumerate(_decode_native_runs(source_path, target_pts), 1):
        if image.shape != (1080, 1920, 3):
            raise ValueError("Native decoded frame size changed")
        mask, mask_status = people.get(stamp, require_sam2=True)
        exclude = _source_mask(mask)
        e = pts_to_edge_index[stamp]
        sl = slice(int(edge_offsets[e]), int(edge_offsets[e + 1]))
        top = edge_uv[sl][edge_ids[sl] == 0]
        contact = edge_uv[sl][edge_ids[sl] == 1]
        support = measure_local_support(image, top, exclude)
        raw_parts.append(support["raw"].astype(np.float32))
        accepted_parts.append(support["accepted"].astype(np.float32))
        raw_offsets.append(raw_offsets[-1] + len(support["raw"]))
        accepted_offsets.append(accepted_offsets[-1] + len(support["accepted"]))
        for pair_index, side in incident[stamp]:
            buf = pair_buffers[pair_index]
            selected = buf["selected"]
            xy = motion["xy0" if side == 0 else "xy1"][buf["start"] + selected]
            buf["source_ok" if side == 0 else "target_ok"] = classify_points(
                xy, support, exclude)
        frame_rows.append({"pts": stamp, "raw_bgr_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                           "sam2_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                           "sam2_status": mask_status, "cached_top_points": len(top),
                           "cached_contact_points": len(contact),
                           "local_candidate_columns": len(support["raw"]),
                           "accepted_source_columns": len(support["accepted"]),
                           "accepted_runs": len(support["runs"]),
                           "evidence_index": number - 1})
        if stamp in qa_pts:
            qa_images[stamp] = image.copy()
        if number % 25 == 0 or number == len(target_pts):
            print(f"source support {number}/{len(target_pts)} PTS={stamp}", flush=True)

    pairs = []
    qa_points: dict[int, list[np.ndarray]] = {stamp: [] for stamp in qa_pts}
    for pair_index, buf in pair_buffers.items():
        source_ok, target_ok = buf["source_ok"], buf["target_ok"]
        if source_ok is None or target_ok is None:
            raise ValueError(f"Pair {pair_index} missed one native endpoint")
        passed = buf["selected"][source_ok & target_ok]
        global_indices = buf["start"] + passed
        xy0, xy1 = motion["xy0"][global_indices], motion["xy1"][global_indices]
        a, b = map(int, buf["row"]["pts"])
        for stamp, uv in ((a, xy0), (b, xy1)):
            if stamp in qa_points:
                qa_points[stamp].append(uv)
        pairs.append({"source_file_id": source_file["id"],
                      "source_pair_index": pair_index, "shot_index": shot_index,
                      "pts": [a, b], "source_point_indices": passed.astype(int).tolist(),
                      "source_global_point_indices": global_indices.astype(int).tolist(),
                      "support_count": len(passed),
                      "source_x_range": [float(xy0[:, 0].min()), float(xy0[:, 0].max())]
                      if len(xy0) else None,
                      "target_x_range": [float(xy1[:, 0].min()), float(xy1[:, 0].max())]
                      if len(xy1) else None,
                      "coverage_note": "local_only_below8" if 0 < len(passed) < 8
                      else "no_supported_pair" if not len(passed) else "spans_to_be_reviewed"})

    output.mkdir(parents=True, exist_ok=True)
    evidence_path = output / "source_board_support_columns.npz"
    np.savez_compressed(evidence_path, pts=np.asarray(target_pts, np.int64),
                        raw_offsets=np.asarray(raw_offsets, np.int64),
                        raw_columns=np.concatenate(raw_parts),
                        accepted_offsets=np.asarray(accepted_offsets, np.int64),
                        accepted_columns=np.concatenate(accepted_parts))
    evidence_sha = _sha256(evidence_path)
    for row in frame_rows:
        row["source_support_evidence_path"] = str(evidence_path)
        row["source_support_evidence_sha256"] = evidence_sha
    qa_overlays = {}
    for stamp, image in qa_images.items():
        q = frame_rows[target_pts.index(stamp)]
        e = q["evidence_index"]
        part = accepted_parts[e]
        overlay = image.copy()
        for x, top_y, foot_y, _, _ in part:
            cv2.circle(overlay, (int(x), int(foot_y)), 2, (0, 180, 255), -1)
        points = np.concatenate(qa_points[stamp]) if qa_points[stamp] else np.empty((0, 2))
        for x, y in points:
            cv2.circle(overlay, (int(round(x)), int(round(y))), 5, (0, 0, 255), 2)
        qa_path = output / f"both_end_source_board_{stamp}.png"
        if not cv2.imwrite(str(qa_path), overlay):
            raise RuntimeError(f"Cannot write {qa_path}")
        qa_overlays[str(stamp)] = {"path": str(qa_path), "sha256": _sha256(qa_path),
                                   "both_end_endpoint_count": len(points)}
    report = {"schema": "rinkspace-source-board-motion-support-v1",
              "source_sha256": source_hash,
              "source_shots_path": str(source_shots), "source_shots_sha256": _sha256(source_shots),
              "source_path": str(source_path),
              "base_strict_manifest_sha256": _sha256(strict_manifest),
              "candidate_scene_sha256": manifest["candidate_scene_sha256"],
              "strict_pair_summary_sha256": _sha256(pair_summary),
              "source_files": [source_file],
              "source_edge_cache": {f"shot{shot_index:02d}": {
                  "path": str(edge_cache), "sha256": _sha256(edge_cache), **edge_meta}},
              "detector_module": {"path": str(Path(__file__).resolve()),
                                  "sha256": _sha256(Path(__file__))},
              "scope": "native original hint0 rows in previously strict<8 pairs only",
              "frame_count": len(frame_rows), "pair_count": len(pairs),
              "both_end_supported_pixels": int(sum(row["support_count"] for row in pairs)),
              "frames": frame_rows, "pairs": pairs, "qa_overlays": qa_overlays,
              "limits": ["Source-only board-interior identity, not a precise contact edge",
                         "No regenerated motion UV, no model projection in source gate",
                         "Thin or weak-color spans and foreground remain unknown",
                         "Fewer than eight pixels are local evidence, not a full pose solution"]}
    report_path = output / "source_board_motion_support.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "frames": len(frame_rows),
                      "pairs": len(pairs),
                      "supported_pixels": report["both_end_supported_pixels"]}, indent=2))
    return report


def inspect_frame(image_path: Path, pts: int, edge_cache: Path,
                  motion_cache: Path, out_dir: Path,
                  *, exclude: np.ndarray | None = None) -> dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode source PNG: {image_path}")
    top, contact, cache_meta = _source_edges_at(edge_cache, pts)
    support = measure_local_support(image, top, exclude)
    motion = _motion_points(motion_cache, pts)
    keep = classify_points(motion, support, exclude)
    overlay = image.copy()
    for x, top_y, foot_y, _, _ in support["accepted"]:
        cv2.line(overlay, (int(x), int(top_y + 16)),
                 (int(x), int(foot_y - 16)), (0, 190, 0), 2)
        cv2.circle(overlay, (int(x), int(foot_y)), 2, (0, 190, 255), -1)
    for x, y in motion[keep]:
        cv2.circle(overlay, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)
    for x, y in contact:
        cv2.circle(overlay, (int(round(x)), int(round(y))), 2, (255, 0, 255), -1)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / f"source_board_support_{pts}.png"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f"Cannot write {overlay_path}")
    return {"pts": pts, "image_path": str(image_path),
            "image_sha256": _sha256(image_path), "source_edge_cache": str(edge_cache),
            "source_edge_cache_sha256": _sha256(edge_cache),
            "motion_cache": str(motion_cache), "motion_cache_sha256": _sha256(motion_cache),
            "source_metadata": cache_meta, "cached_top_points": len(top),
            "cached_contact_points": len(contact),
            "raw_color_columns": len(support["raw"]),
            "accepted_color_columns": len(support["accepted"]),
            "accepted_runs": len(support["runs"]),
            "native_hint0_endpoints": len(motion),
            "locally_bounded_board_endpoints": int(np.count_nonzero(keep)),
            "overlay_path": str(overlay_path), "overlay_sha256": _sha256(overlay_path),
            "semantics": "source-only board-interior support; no new UV, edge or fit factor"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", action="append",
                        help="PTS=path to a cached unmodified native PNG")
    parser.add_argument("--edge-cache", type=Path, required=True)
    parser.add_argument("--motion-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict-manifest", type=Path)
    parser.add_argument("--strict-pair-summary", type=Path)
    parser.add_argument("--source-shots", type=Path)
    parser.add_argument("--shot-index", type=int, default=5)
    parser.add_argument("--sam2-mask", action="store_true",
                        help="Exclude existing original-PTS SAM2 person masks")
    args = parser.parse_args()
    if args.strict_manifest is not None:
        if args.frame or args.strict_pair_summary is None or args.source_shots is None:
            parser.error("Batch sidecar requires strict summary and source shots, without --frame")
        build_missing_strict_sidecar(args.strict_manifest, args.strict_pair_summary,
                                     args.source_shots, args.edge_cache,
                                     args.motion_cache, args.output,
                                     shot_index=args.shot_index)
        return
    if not args.frame:
        parser.error("Need --frame for a sample overlay or --strict-manifest for a sidecar")
    people = None
    if args.sam2_mask:
        try:
            from .render_result import DEFAULT_MASKS, DEFAULT_PEOPLE, PersonMasks
        except ImportError:
            from render_result import DEFAULT_MASKS, DEFAULT_PEOPLE, PersonMasks
        people = PersonMasks(DEFAULT_MASKS, DEFAULT_PEOPLE)
    results = []
    for item in args.frame:
        stamp, filename = item.split("=", 1)
        pts = int(stamp)
        mask = None
        if people is not None:
            mask, _ = people.get(pts, require_sam2=True)
        results.append(inspect_frame(Path(filename), pts, args.edge_cache,
                                     args.motion_cache, args.output, exclude=mask))
    report = {"schema": SCHEMA, "frames": results,
              "mask_source": "existing SAM2 person mask" if people is not None else "none"}
    path = args.output / "source_board_support.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(path), "frames": [
        {k: row[k] for k in ("pts", "raw_color_columns", "accepted_color_columns",
                            "native_hint0_endpoints", "locally_bounded_board_endpoints")}
        for row in results]}, indent=2))


if __name__ == "__main__":
    main()
