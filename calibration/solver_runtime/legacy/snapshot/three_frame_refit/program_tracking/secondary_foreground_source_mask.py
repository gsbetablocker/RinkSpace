"""Original-pixel foreground audit for the secondary long-side camera.

The left foreground ramp in shot 06 is not a modeled rink board.  Detect its
visible ice-side diagonal from each native source frame, then query unaltered
UV in the frozen observation inputs.  No camera state, solver, or residual is
read to construct the mask.  Both the interior and uncertain edge must be
removed from later *known board* factors; the original UV remain in archives.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()

import cv2
import numpy as np

from source_board_support import _decode_native_runs


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TMP = Path("D:/CodexData/RinkSpace_program_tracking_tmp")
SHOTS = HERE / "source_shots.json"
PACKET = TMP / "native_surface_model_comparison_v1/frozen_board_pairs.npz"
MANIFEST = PACKET.with_name("strict_board_manifest.json")
EDGE_ROOT = TMP / "programme_native_evidence_v13/driver_inputs/source_edges"
DEFAULT_OUTPUT = (TMP / "continuous_camera_model_v2" /
                  "secondary_visibility_source_review" / "foreground_source_mask")
SELECTED = (0, 2, 6)
SHOT06_START = 230369024
SHOT06_END = 230518016
ROWS = (700, 750, 800, 850, 900, 950, 1000, 1050)
EDGE_BAND_PX = 20.


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _candidates(gray: np.ndarray, y: int) -> list[tuple[int, float]]:
    """Dark near-side ramp to brighter original ice; independent of a pose."""
    row = cv2.GaussianBlur(gray[y-3:y+4, :500], (11, 3), 0).mean(axis=0)
    contrast = row[14:] - row[:-14]
    valid = ((contrast > 18) & (row[:-14] < 155) & (row[14:] > 145))
    peaks: list[tuple[int, float]] = []
    for x in np.flatnonzero(valid):
        value = float(contrast[x])
        if peaks and x - peaks[-1][0] < 11:
            if value > peaks[-1][1]:
                peaks[-1] = (int(x), value)
        else:
            peaks.append((int(x), value))
    return [(x + 7, value) for x, value in peaks]


def detect_boundary(frame: np.ndarray) -> dict | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    candidates = {y: _candidates(gray, y) for y in ROWS}
    seeds = [(y, x) for y in ROWS for x, value in candidates[y] if value > 22]
    best = None
    for y0, x0 in seeds:
        support = []
        for y in ROWS:
            predicted_x = x0 + .51 * (y-y0)
            close = [(abs(x-predicted_x), x, value)
                     for x, value in candidates[y] if abs(x-predicted_x) < 12]
            if close:
                _, x, value = min(close)
                support.append((y, x, value))
        if len(support) < 3 or support[-1][0]-support[0][0] < 150:
            continue
        ys = np.asarray([item[0] for item in support], np.float64)
        xs = np.asarray([item[1] for item in support], np.float64)
        slope, intercept = np.polyfit(ys, xs, 1)
        if not (.43 < slope < .60
                and np.max(np.abs(xs-(slope*ys+intercept))) <= 9):
            continue
        score = sum(item[2] for item in support)
        if best is None or score > best[0]:
            best = (score, dict(
                slope=float(slope), intercept=float(intercept),
                support_rows=[int(y) for y, _, _ in support],
                support_x=[int(x) for _, x, _ in support],
                max_fit_error_px=float(np.max(np.abs(xs-(slope*ys+intercept))))))
    if best is None:
        return None
    result = best[1]
    # In the board band, the same foreground has a nearly vertical leading
    # edge.  The top of the genuine dark-blue advertisement supplies its
    # independent source-pixel boundary.  White QR squares farther down make
    # a whole-board color average unreliable, hence this narrow top strip.
    line_at_580 = result["slope"]*580 + result["intercept"]
    if line_at_580 > 0:
        hsv = cv2.cvtColor(frame[445:475, :500], cv2.COLOR_BGR2HSV)
        blue = ((hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 135)
                & (hsv[:, :, 1] > 55) & (hsv[:, :, 2] > 25))
        occupied = np.convolve(blue.mean(axis=0), np.ones(9)/9, "same") > .18
        starts = np.flatnonzero(np.r_[occupied[0], occupied[1:] & ~occupied[:-1]])
        require(len(starts) > 0, "Original blue advertisement edge disappeared")
        board_edge = int(starts[0])
        require(25 <= board_edge-line_at_580 <= 70,
                "Foreground/advertisement boundaries disagree in original frame")
        result["board_band_edge_x"] = board_edge
    else:
        result["board_band_edge_x"] = None
    return result


def _edge_x(y: np.ndarray, boundary: dict) -> np.ndarray:
    x = boundary["slope"]*y + boundary["intercept"]
    board = boundary["board_band_edge_x"]
    if board is not None:
        x650 = boundary["slope"]*650 + boundary["intercept"]
        x = np.where(y <= 580, board,
                     np.where(y < 650, board + (x650-board)*(y-580)/70, x))
    return x


def classify_uv(pts: int, xy: np.ndarray, boundaries: dict[int, dict]) -> np.ndarray:
    """0 unaffected, 1 visible foreground interior, 2 uncertain edge."""
    q = np.asarray(xy)
    require(q.ndim == 2 and q.shape[1] == 2, "UV array shape changed")
    out = np.zeros(len(q), np.uint8)
    boundary = boundaries.get(int(pts))
    if boundary is None or not len(q):
        return out
    x_edge = _edge_x(q[:, 1], boundary)
    # The ice-side diagonal is measured at rows 700:1050.  When it enters the
    # board band, the actual blue-advertisement leading edge determines the
    # foreground's right extent at 420:580.  The short 580:650 transition is
    # interpolated between two independently visible source boundaries.
    eligible = (q[:, 1] >= 420) & (q[:, 0] >= 0) & (q[:, 0] < 500)
    out[eligible & (q[:, 0] < x_edge-EDGE_BAND_PX)] = 1
    out[eligible & (np.abs(q[:, 0]-x_edge) <= EDGE_BAND_PX)] = 2
    return out


def _overlay(frame: np.ndarray, pts: int, boundary: dict) -> np.ndarray:
    image = frame.copy()
    for offset, color in ((-EDGE_BAND_PX, (0, 0, 255)),
                          (0, (0, 255, 255)),
                          (EDGE_BAND_PX, (255, 0, 255))):
        yy = np.asarray([420., 580., 650., 1079.])
        xx = _edge_x(yy, boundary)+offset
        points = np.column_stack((np.round(xx), yy)).astype(np.int32)
        cv2.polylines(image, [points], False, color, 3)
    cv2.putText(image, f"SOURCE shot06 PTS {pts}; red interior, magenta unknown edge",
                (8, 33), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 2)
    return image


def original_boundaries(video: Path, output: Path) -> tuple[dict[int, dict], dict]:
    wanted = list(range(SHOT06_START, SHOT06_END, 256))
    require(len(wanted) == 582, "Shot06 original native timeline changed")
    output.mkdir(parents=True, exist_ok=True)
    boundaries: dict[int, dict] = {}
    sampled = {230409984, 230411008, 230412032, 230414080, 230416128,
               230418176, 230420224, 230422784, 230425344, 230427904}
    for pts, frame in _decode_native_runs(video, wanted):
        boundary = detect_boundary(frame)
        if boundary is not None:
            boundaries[pts] = boundary
            if pts in sampled:
                path = output / f"original_boundary_pts{pts}.jpg"
                require(cv2.imwrite(str(path), _overlay(frame, pts, boundary),
                                    [cv2.IMWRITE_JPEG_QUALITY, 91]),
                        f"Could not save original boundary QA: {path}")
    transitions = (230407424, 230408448, 230409472,
                   230428160, 230428928, 230430464)
    for pts, frame in _decode_native_runs(video, list(transitions)):
        crop = frame[:, :500].copy()
        cv2.putText(crop, f"Original source PTS {pts}", (6, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
        path = output / f"original_transition_pts{pts}.jpg"
        require(cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 91]),
                f"Could not save original transition QA: {path}")
    detected = sorted(boundaries)
    require(detected == list(range(230409984, 230428160, 256)),
            "Original diagonal detection interval changed or has a gap")
    # Two opposite-side original pixels explicitly checked by eye.  The
    # 230411008 board advertisement must remain unaffected.
    cases = [(230411008, (100., 552.), 0),
             (230414080, (42., 558.), 1),
             (230416128, (88., 518.), 1)]
    for pts, xy, expected in cases:
        actual = int(classify_uv(pts, np.asarray([xy]), boundaries)[0])
        require(actual == expected,
                f"Original-pixel foreground identity changed at {pts}, {xy}: {actual}")
    x1000 = np.asarray([b["slope"]*1000+b["intercept"]
                        for b in boundaries.values()])
    active_board = [pts for pts, row in boundaries.items()
                    if row["board_band_edge_x"] is not None]
    require(active_board == list(range(230413056, 230425600, 256)),
            "Original foreground/advertisement board-band interval changed")
    other_shots = {}
    timeline = json.loads(SHOTS.read_text(encoding="utf8"))
    for index in (0, 2):
        row = timeline["shots"][index]
        stamps = list(range(int(row["start_pts"]), int(row["end_pts"]), 256))
        found = [pts for pts, frame in _decode_native_runs(video, stamps)
                 if detect_boundary(frame) is not None]
        require(not found, f"Similar original foreground boundary in shot{index:02d}")
        other_shots[str(index)] = dict(native_frames_scanned=len(stamps),
                                       matching_diagonal_frames=0)
    return boundaries, dict(
        shot06_native_frames=len(wanted), observed_diagonal_frames=len(boundaries),
        first_observed_diagonal_pts=detected[0],
        last_observed_diagonal_pts=detected[-1],
        board_band_intrusion_frames=len(active_board),
        first_board_band_intrusion_pts=active_board[0],
        last_board_band_intrusion_pts=active_board[-1],
        other_secondary_shots_scanned=other_shots,
        source_transition_brackets=dict(
            entry=dict(no_clear_wedge_pts=230407424,
                       bottom_only_visible_pts=230408448,
                       first_full_diagonal_pts=230409984),
            exit=dict(last_full_diagonal_pts=230427904,
                      bottom_only_visible_pts=230428160,
                      no_clear_wedge_pts=230428928)),
        x_at_y1000_range_px=[float(x1000.min()), float(x1000.max())],
        fitted_boundary_rule=("dark-to-light ice edge at rows 700:1050; when it enters board band, "
                              "dark-blue real advertisement top-strip leading edge at rows 445:475"),
        definite_interior="x < piecewise source edge - 20px, y >= 420px",
        uncertain_boundary="abs(x - piecewise source edge) <= 20px, y >= 420px",
        exclusion_policy="both definite interior and uncertain boundary excluded as known board",
        unclassified_transition="bottom-only entry/exit outside detected diagonal; no positive board claim",
        original_transition_qa=[str(output / f"original_transition_pts{pts}.jpg")
                                for pts in transitions],
        identity_checks=[dict(pts=p, xy=list(xy), mask_class=expected)
                         for p, xy, expected in cases])


def _frozen_packet_inventory(boundaries: dict[int, dict], manifest: dict) -> dict:
    with np.load(PACKET, allow_pickle=False) as z:
        packet = {name: z[name].copy() for name in
                  ("pair_pts", "shot_index", "source_file_id", "source_pair_index",
                   "offsets", "source_point_index", "source_global_point_index",
                   "xy0", "xy1")}
    sources = {int(row["id"]): row for row in manifest["source_files"]}
    hits = []
    counts = Counter()
    for pair_index, (a, b) in enumerate(packet["pair_pts"]):
        shot = int(packet["shot_index"][pair_index])
        if shot not in SELECTED:
            continue
        lo, hi = map(int, packet["offsets"][pair_index:pair_index+2])
        file_id = int(packet["source_file_id"][pair_index])
        source = sources[file_id]
        family = "auxiliary" if source["kind"] == "auxiliary" else "native"
        counts[family+"_rows_total"] += hi-lo
        left = classify_uv(int(a), packet["xy0"][lo:hi], boundaries)
        right = classify_uv(int(b), packet["xy1"][lo:hi], boundaries)
        for local in np.flatnonzero((left > 0) | (right > 0)):
            global_index = lo + int(local)
            hits.append(dict(
                family=family, shot_index=shot,
                packet_pair_index=pair_index,
                packet_point_global_index=global_index,
                packet_point_local_index=int(local),
                source_file_id=file_id,
                source_file_sha256=source["sha256"],
                source_pair_index=int(packet["source_pair_index"][pair_index]),
                source_point_index=int(packet["source_point_index"][global_index]),
                source_global_point_index=int(packet["source_global_point_index"][global_index]),
                source_pts=int(a), target_pts=int(b),
                xy0=packet["xy0"][global_index].tolist(),
                xy1=packet["xy1"][global_index].tolist(),
                source_mask_class=int(left[local]), target_mask_class=int(right[local])))
            counts[family+"_rows_excluded"] += 1
            counts[family+"_source_interior"] += int(left[local] == 1)
            counts[family+"_source_boundary_unknown"] += int(left[local] == 2)
            counts[family+"_target_interior"] += int(right[local] == 1)
            counts[family+"_target_boundary_unknown"] += int(right[local] == 2)
    return dict(counts=dict(counts), hits=hits)


def _keyframe_and_edge_inventory(boundaries: dict[int, dict], output: Path,
                                 source_sha: str) -> dict:
    # Reconstruct only v13 graph inputs.  Capture raises before graph.update;
    # this performs no optimization and never reads a new fitted camera.
    import conditioned_camera_comparison as driver
    import low_end_calibration_transfer_profile as transfer

    data = driver.prepare(transfer.seed_driver.CONFIG,
                          transfer.seed_driver.CANDIDATE,
                          transfer.seed_driver.MANIFEST,
                          SELECTED, output / "graph_capture",
                          complete_camera_group=True)
    require(data["camera_id"] == "secondary_long_side"
            and data["captured"]["source_hash"] == source_sha,
            "Captured keyframe graph source/camera identity differs")
    links = []
    link_counts = Counter()
    for shot in data["shots"]:
        shot_index = int(shot["index"])
        for pair_index, (a, b, pair) in enumerate(shot["links"]):
            link_counts["pairs_total"] += 1
            link_counts["rows_total"] += len(pair.xy0)
            left = classify_uv(int(a), pair.xy0, boundaries)
            right = classify_uv(int(b), pair.xy1, boundaries)
            for local in np.flatnonzero((left > 0) | (right > 0)):
                links.append(dict(
                    shot_index=shot_index, shot_link_index=pair_index,
                    source_pts=int(a), target_pts=int(b),
                    pair_point_index=int(local),
                    xy0=pair.xy0[local].tolist(), xy1=pair.xy1[local].tolist(),
                    source_mask_class=int(left[local]),
                    target_mask_class=int(right[local])))
                link_counts["rows_excluded"] += 1
    source_edges = []
    edge_counts = Counter()
    config_shots = {int(row["shot_index"]): row for row in data["config"]["shots"]}
    for shot in data["shots"]:
        number = int(shot["index"])
        path = Path(config_shots[number]["source_edges"])
        with np.load(path, allow_pickle=False) as z:
            require(str(z["source_sha256"]) == source_sha,
                    f"Source-edge archive changed original video: shot{number:02d}")
            pts = z["pts"].copy()
            offsets = z["offsets"].copy()
            uv = z["observed_uv"].copy()
            edge_id = z["edge_id"].copy()
        require(np.array_equal(pts, shot["pts"]),
                f"Source-edge PTS changed: shot{number:02d}")
        for frame_index, stamp in enumerate(pts):
            lo, hi = map(int, offsets[frame_index:frame_index+2])
            edge_counts["rows_total"] += hi-lo
            classes = classify_uv(int(stamp), uv[lo:hi], boundaries)
            for local in np.flatnonzero(classes > 0):
                j = lo + int(local)
                source_edges.append(dict(
                    shot_index=number, frame_index=frame_index,
                    pts=int(stamp), source_edge_file=str(path),
                    source_edge_file_sha256=digest(path),
                    source_edge_global_index=j,
                    source_edge_frame_index=int(local),
                    edge_id=int(edge_id[j]),
                    uv=uv[j].tolist(), mask_class=int(classes[local])))
                edge_counts["rows_excluded"] += 1
    return dict(keyframe_surface=dict(counts=dict(link_counts), hits=links),
                source_edge=dict(counts=dict(edge_counts), hits=source_edges))


def audit(output: Path) -> dict:
    source = json.loads(SHOTS.read_text(encoding="utf8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    video = Path(source["source_path"])
    require(digest(PACKET) == manifest["frozen_packet_sha256"]
            and digest(video) == manifest["source_sha256"]
            == source["source_sha256_from_timeline"],
            "Frozen board packet and original video do not bind")
    require(len(source["shots"]) == 10 and
            [source["shots"][i]["camera_id"] for i in SELECTED] ==
            ["secondary_long_side"]*3,
            "Secondary camera classification changed")
    boundaries, boundary_metadata = original_boundaries(video, output)
    frozen = _frozen_packet_inventory(boundaries, manifest)
    others = _keyframe_and_edge_inventory(boundaries, output,
                                          manifest["source_sha256"])
    report = dict(
        schema="rinkspace.secondary-foreground-source-mask.v1",
        source_video_path=str(video.resolve()),
        source_video_sha256=manifest["source_sha256"],
        source_shots_path=str(SHOTS.resolve()), source_shots_sha256=digest(SHOTS),
        frozen_packet_path=str(PACKET.resolve()),
        frozen_packet_sha256=manifest["frozen_packet_sha256"],
        detector_module_path=str(Path(__file__).resolve()),
        detector_module_sha256=digest(Path(__file__)),
        selected_camera_id="secondary_long_side",
        selected_shot_indices=list(SELECTED),
        boundary_metadata=boundary_metadata,
        boundary_by_pts={str(k): v for k, v in boundaries.items()},
        input_families=dict(native_auxiliary=frozen, **others),
        mask_source="native original pixels only; no solved camera, residual, or holdout",
        limitation="Only this visible diagonal foreground class is audited; unsampled other non-board classes remain unverified")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "foreground_exclusion_manifest.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit(args.output)
    print(json.dumps(dict(
        manifest=str((args.output / "foreground_exclusion_manifest.json").resolve()),
        boundary=report["boundary_metadata"],
        families={key: value["counts"]
                  for key, value in report["input_families"].items()}),
        ensure_ascii=False, indent=2))
