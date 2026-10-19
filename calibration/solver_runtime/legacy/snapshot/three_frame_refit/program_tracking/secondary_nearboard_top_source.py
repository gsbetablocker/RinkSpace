"""Source-only ice-side top-edge evidence for the secondary long-side view.

The user identified the grey diagonal in original PTS 230416128 as the top
of the lower long-side rink boards.  Its broad grey surface is not a point on
the scene's zero-thickness vertical ad face.  Only the grey-to-ice edge is
read, as a deliberately broad pixel interval.  This module never changes the
older foreground exclusion packet, original UV, camera model, or scene.

``build_source_report`` reads the already reviewed 71 complete boundaries and
11 entry/exit frames, decoding only those 82 native frames.  The solver-facing
``prepare_secondary_nearboard_top`` reads the frozen report and merges confirmed
same-exposure aliases.  ``attach_secondary_nearboard_top_factors`` is opt-in;
it adds factors without running a solve or changing existing graph factors.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Mapping

import av
import cv2
import numpy as np

try:
    from .continuous_brown_adapter import adapt_factor
    from .projection import from_state, project_central_valid
    from .reference_board_pair import BoardIntersectionError
    from .robust_observations import pointwise_huber
except ImportError:
    from continuous_brown_adapter import adapt_factor
    from projection import from_state, project_central_valid
    from reference_board_pair import BoardIntersectionError
    from robust_observations import pointwise_huber


ROOT=Path('D:/OneDrive/codex_projs/RinkSpace')
TMP = Path("D:/CodexData/RinkSpace_program_tracking_tmp")
SOURCE_SHOTS = ROOT / "three_frame_refit/program_tracking/source_shots.json"
MASK_MANIFEST = (TMP / "continuous_camera_model_v2/secondary_visibility_source_review"
                 / "foreground_source_mask/foreground_exclusion_manifest.json")
TRANSITION_QA = (TMP / "continuous_camera_model_v2/secondary_visibility_source_review"
                 / "render_foreground_source_qa/source_mask_qa.json")
CONFIRMATION_FRAME = (ROOT / "evidence/secondary_visibility_domain_20260927"
                      / "shot06_pts230416128_raw_full.png")
CONFIRMATION_RECORD = (ROOT / "evidence/secondary_visibility_domain_20260927"
                       / "user_confirmed_lower_board_top.json")
DEFAULT_REPORT = (ROOT / "evidence/secondary_visibility_domain_20260927"
                  / "nearboard_top_source_review/source_observations.json")
SOURCE_SHA256 = "f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891"
SHOTS_SHA256 = "feb7f42539bab4a7def84e74eaaf40dd0808710e3c0c0c101e64c381ddc63197"
MASK_SHA256 = "a77abdb70feafcf2245601a179d33d0115278c818a2fa86f2b5738b2c3a376c7"
QA_SHA256 = "04d4eb2ae0848de1e7135f3d9167ff7e943aab7945b78123682e109dbe62cc2e"
CONFIRMATION_PNG_SHA256 = "596e5289e15563a146fad7cf212c2a40e5f719da17903df736d656caa3b02e7b"
CONFIRMATION_RECORD_SHA256 = "8925a96443ca3751e0ffc81c6c8f249b4e1727a64b7a3be5cb3e7c2ff3d23d86"
CONFIRMATION_TEXT = "这个是冰场下长边围挡的上沿"
CONFIRMATION_PTS = 230416128
SHOT06 = 6
SOURCE_TICKS_PER_SECOND = 15360
EDGE_SIGMA_PX = 4.0  # Explicit outside-interval likelihood assumption, not a measured covariance.
MAX_WORLD_STEP = 0.25  # Scene units; subdivision preserves every original straight-side vertex.
MAX_PROJECTED_CHORD_ERROR_PX = 0.25


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _check_user_confirmation() -> None:
    _require(_sha(CONFIRMATION_RECORD) == CONFIRMATION_RECORD_SHA256,
             "User confirmation record hash changed")
    record = json.loads(CONFIRMATION_RECORD.read_text(encoding="utf8"))
    _require(record["user_statement_zh"] == CONFIRMATION_TEXT
             and int(record["source_pts"]) == CONFIRMATION_PTS
             and record["source_video_sha256"] == SOURCE_SHA256
             and record["source_image_sha256"] == CONFIRMATION_PNG_SHA256
             and record["identity"] == "lower_long_side_board_top",
             "Human-confirmed near-board identity changed")


def _read_reviewed_sources() -> tuple[dict, dict, dict]:
    _check_user_confirmation()
    _require(_sha(SOURCE_SHOTS) == SHOTS_SHA256
             and _sha(MASK_MANIFEST) == MASK_SHA256
             and _sha(TRANSITION_QA) == QA_SHA256
             and _sha(CONFIRMATION_FRAME) == CONFIRMATION_PNG_SHA256,
             "Frozen source review or human-confirmed original frame changed")
    shots = json.loads(SOURCE_SHOTS.read_text(encoding="utf8"))
    mask = json.loads(MASK_MANIFEST.read_text(encoding="utf8"))
    qa = json.loads(TRANSITION_QA.read_text(encoding="utf8"))
    _require(shots["source_sha256_from_timeline"] == mask["source_video_sha256"]
             == SOURCE_SHA256 and shots["source_time_base"] == "1/15360"
             and mask["selected_camera_id"] == "secondary_long_side"
             and mask["selected_shot_indices"] == [0, 2, 6]
             and len(mask["boundary_by_pts"]) == 71
             and qa["transition_frames_audited"] == 11,
             "Reviewed secondary source identity or visibility inventory changed")
    counts = {i: int(shots["shots"][i]["source_native_frame_count"])
              for i in (0, 2, 6)}
    _require(counts == {0: 17, 2: 680, 6: 582}
             and sum(counts.values()) == 1279,
             "Complete secondary camera group frame counts changed")
    return shots, mask, qa


def _seed_rows(mask: dict, qa: dict) -> dict[int, tuple[str, list[tuple[int, int]]]]:
    selected: dict[int, tuple[str, list[tuple[int, int]]]] = {}
    for stamp, boundary in mask["boundary_by_pts"].items():
        pts = int(stamp)
        rows = list(map(int, boundary["support_rows"]))
        xs = list(map(int, boundary["support_x"]))
        _require(len(rows) == len(xs) and 3 <= len(rows) <= 8
                 and len(set(rows)) == len(rows), "Invalid reviewed full-boundary support")
        selected[pts] = ("complete_diagonal", list(zip(rows, xs)))
    for frame in qa["frames"]:
        if frame["status"] != "native_pixel_bottom_transition":
            continue
        pts = int(frame["pts"])
        _require(pts not in selected, "Transition duplicates a complete diagonal")
        measured = frame["source_evidence"]["measured_rows"]
        rows = [(int(item["y"]), int(item["edge_x_px"]))
                for item in measured if int(item["edge_x_px"]) >= 0]
        selected[pts] = ("partial_entry_exit", rows)
    expected = list(range(230408448, 230429184 + 256, 256))
    _require(sorted(selected) == expected and len(selected) == 82
             and sorted(int(k) for k in mask["boundary_by_pts"]) ==
                 list(range(230409984, 230427904 + 256, 256)),
             "Reviewed 71+11 source interval no longer matches the native timeline")
    return selected


def _transition_interval(gray: np.ndarray, y: int, seed_x: int
                         ) -> tuple[list[int] | None, str, float | None]:
    """Bracket the raw dark-to-ice rise; endpoints are outward whole pixels.

    The seven-row average incorporates motion blur and local line slope.  A
    q10/q90 luminance rise is visual support, not a statistical confidence
    interval or a subpixel claim about the physical board edge.
    """
    if not (3 <= y < 1077 and 33 <= seed_x < 468):
        return None, "image_border_no_two_plateaux", None
    line = gray[y - 3:y + 4, :500].astype(np.float64).mean(axis=0)
    dark = float(np.median(line[seed_x - 30:seed_x - 16]))
    ice = float(np.median(line[seed_x + 16:seed_x + 31]))
    amplitude = ice - dark
    if amplitude < 18.:
        return None, "weak_grey_ice_contrast", amplitude
    xx = np.arange(seed_x - 15, seed_x + 16)
    values = line[xx]
    crossings = []
    for fraction in (.1, .5, .9):
        level = dark + fraction * amplitude
        candidates = np.flatnonzero((values[:-1] < level) &
                                    (values[1:] >= level))
        if not len(candidates):
            return None, "no_local_rising_transition", amplitude
        index = int(candidates[np.argmin(np.abs(xx[candidates] - seed_x))])
        crossing = (float(xx[index]) +
                    (level - values[index]) / (values[index + 1] - values[index]))
        crossings.append(crossing)
    q10, q50, q90 = crossings
    if not (q10 < q50 < q90 and 1. <= q90 - q10 <= 30.
            and abs(q50 - seed_x) <= 8.):
        return None, "nonunique_or_broad_transition", amplitude
    return [int(np.floor(q10)), int(np.ceil(q90))], "source_interval", amplitude


def _decode_selected_native(source: Path, pts: list[int]):
    wanted = set(pts)
    first, last = pts[0], pts[-1]
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        _require((stream.time_base.numerator, stream.time_base.denominator)
                 == (1, SOURCE_TICKS_PER_SECOND), "Original source time base changed")
        container.seek(max(0, first - SOURCE_TICKS_PER_SECOND),
                       stream=stream, backward=True)
        for frame in container.decode(stream):
            stamp = frame.pts
            if stamp is None or stamp < first:
                continue
            if stamp > last:
                break
            if stamp in wanted:
                image = frame.to_ndarray(format="bgr24")
                _require(image.shape == (1080, 1920, 3)
                         and image.dtype == np.uint8,
                         "Decoded original sensor frame changed shape")
                yield int(stamp), image


def build_source_report(output: Path = DEFAULT_REPORT) -> dict:
    """Read only the reviewed visible window and save auditable intervals."""
    shots, mask, qa = _read_reviewed_sources()
    source = Path(shots["source_path"])
    _require(source.resolve() == Path(mask["source_video_path"]).resolve()
             and _sha(source) == SOURCE_SHA256,
             "Original full programme video hash changed")
    selected = _seed_rows(mask, qa)
    frames = []
    reasons = Counter()
    for pts, image in _decode_selected_native(source, sorted(selected)):
        kind, seeds = selected[pts]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        measurements = []
        unknown = []
        for y, x in seeds:
            bounds, status, contrast = _transition_interval(gray, y, x)
            if bounds is None:
                reasons[status] += 1
                unknown.append(dict(y=y, reviewed_seed_x=x, reason=status))
            else:
                measurements.append(dict(y=y, x_interval_px=bounds,
                                         reviewed_seed_x=x,
                                         source_contrast_grey_levels=round(contrast, 2)))
        frames.append(dict(
            native_pts=pts, source_time_seconds=pts / SOURCE_TICKS_PER_SECOND,
            source_class=kind,
            source_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            measurements=measurements, unknown_rows=unknown))
    _require([row["native_pts"] for row in frames] == sorted(selected),
             "Failed to decode all 82 reviewed native source frames")
    frame_count = sum(bool(row["measurements"]) for row in frames)
    measurement_count = sum(len(row["measurements"]) for row in frames)
    report = dict(
        schema="rinkspace.secondary-lower-long-board-top-source-intervals.v1",
        authority=dict(kind="user_confirmed_object_identity",
                       exact_text=CONFIRMATION_TEXT, original_native_pts=CONFIRMATION_PTS,
                       original_full_frame_png=str(CONFIRMATION_FRAME.resolve()),
                       original_full_frame_png_sha256=CONFIRMATION_PNG_SHA256,
                       scope="identity extends along the continuous shot06 grey strip by source visual continuity; only the named frame was directly user-confirmed"),
        provenance=dict(source_video=str(source.resolve()),
                        source_video_sha256=SOURCE_SHA256,
                        native_time_base="1/15360", sensor_size_wh=[1920, 1080],
                        source_shots_sha256=SHOTS_SHA256,
                        reviewed_foreground_manifest_sha256=MASK_SHA256,
                        reviewed_entry_exit_qa_sha256=QA_SHA256,
                        no_camera_or_model_used_for_source_measurement=True),
        physical_identity=dict(side="lower_long_side", edge="ice_side_of_grey_board_top",
                               excludes="grey top-face texture and far vertical advertisement face",
                               scene_binding="deferred to current data.scene at factor construction"),
        measurement_policy=dict(
            candidate_rows="reviewed 71 complete-boundary support rows and 11 entry/exit source rows only",
            reading="raw BGR to grayscale, mean 7 vertical pixels, dark/ice plateau medians, outward-rounded q10/q90 rise",
            q10_q90_meaning="visible grey-to-ice transition support, not a confidence interval or exact subpixel board edge",
            minimum_contrast_grey_levels=18., near_border_rows_unknown=True,
            mask_edge_not_reused_as_measurement=True,
            unknown_never_interpolated=True),
        coverage=dict(secondary_complete_source_frames=1279,
                      per_shot_frames={"0": 17, "2": 680, "6": 582},
                      complete_diagonal_native_frames=71,
                      partial_entry_exit_native_frames=11,
                      observed_native_frames=frame_count,
                      native_frames_without_qualified_line=82 - frame_count,
                      measured_scan_rows=measurement_count,
                      unknown_scan_row_reasons=dict(sorted(reasons.items())),
                      outside_reviewed_visible_window="unknown, no positive top-line observation"),
        frames=frames,
        old_foreground_exclusion_packet_unchanged=True,
        old_native_pair_uv_unchanged=True,
        solver_started=False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf8")
    return report


def lower_long_side_top(scene: dict, *, max_world_step: float = MAX_WORLD_STEP
                        ) -> tuple[np.ndarray, dict]:
    """Bind the *physical side* from current scene, retaining original vertices."""
    _require(max_world_step > 0 and np.isfinite(max_world_step),
             "Positive world subdivision required")
    bounds = np.asarray(scene["parameters"]["bounds"], np.float64)
    top = np.asarray(scene["board_surface"]["top_edge_xyz"], np.float64)
    _require(bounds.shape == (4,) and top.ndim == 2 and top.shape[1] == 3
             and np.isfinite(top).all(), "Current board scene is malformed")
    y_lower = float(bounds[1])
    indices = np.flatnonzero(np.abs(top[:, 1] - y_lower) <= 1e-7)
    _require(len(indices) >= 3 and np.all(np.diff(indices) == 1)
             and np.all(np.diff(top[indices, 0]) > 0)
             and np.ptp(top[indices, 2]) <= 1e-9,
             "Scene lower physical straight board side is not unique")
    line = top[indices]
    world = [line[0]]
    for a, b in zip(line[:-1], line[1:]):
        parts = int(np.ceil(np.linalg.norm(b - a) / max_world_step))
        for i in range(1, parts + 1):
            world.append(a + i / parts * (b - a))
    xyz = np.asarray(world, np.float64)
    _require(np.allclose(xyz[0], line[0]) and np.allclose(xyz[-1], line[-1])
             and np.max(np.linalg.norm(np.diff(xyz, axis=0), axis=1))
                 <= max_world_step + 1e-10,
             "World subdivision changed scene geometry")
    return xyz, dict(side="lower_long_side", y_scene=y_lower,
                     z_scene=float(line[0, 2]),
                     x_scene=[float(line[0, 0]), float(line[-1, 0])],
                     source_scene_vertex_indices=[int(indices[0]), int(indices[-1])],
                     original_world_vertices=len(line), sampled_world_vertices=len(xyz),
                     maximum_world_step_scene_units=max_world_step,
                     corner_transition_excluded=True,
                     scene_thickness_status=scene["board_surface"].get("thickness_status"))


def _read_report(report_path: Path, expected_sha256: str) -> dict:
    _check_user_confirmation()
    report_path = Path(report_path)
    _require(len(expected_sha256) == 64 and _sha(report_path) == expected_sha256,
             "Caller-pinned near-board source report hash changed")
    report = json.loads(report_path.read_text(encoding="utf8"))
    _require(report["schema"] == "rinkspace.secondary-lower-long-board-top-source-intervals.v1"
             and report["authority"]["exact_text"] == CONFIRMATION_TEXT
             and report["authority"]["original_native_pts"] == CONFIRMATION_PTS
             and report["authority"]["original_full_frame_png_sha256"] == CONFIRMATION_PNG_SHA256
             and report["provenance"]["source_video_sha256"] == SOURCE_SHA256
             and report["provenance"]["reviewed_foreground_manifest_sha256"] == MASK_SHA256
             and report["provenance"]["reviewed_entry_exit_qa_sha256"] == QA_SHA256
             and report["physical_identity"]["side"] == "lower_long_side"
             and report["old_foreground_exclusion_packet_unchanged"] is True
             and report["old_native_pair_uv_unchanged"] is True
             and report["solver_started"] is False,
             "Near-board source report provenance changed")
    return report


def prepare_secondary_nearboard_top(
    data: dict, aliases: Mapping[int, int], *, report_path: Path = DEFAULT_REPORT,
    expected_report_sha256: str
) -> tuple[list[dict], dict]:
    """Merge same-exposure native intervals without changing source PTS identity."""
    _require(data["camera_id"] == "secondary_long_side"
             and data["captured"]["source_hash"] == SOURCE_SHA256,
             "Near-board observations require the complete secondary source group")
    report = _read_report(report_path, expected_report_sha256)
    _require(set(map(int, data["shot_indices"])) == {0, 2, 6}
             and len(data["pts"]) == 1279,
             "Near-board input is not the complete secondary group")
    alias = {int(k): int(v) for k, v in aliases.items()}
    _require(set(alias) == set(map(int, data["pts"])),
             "Exposure alias map must contain every native group frame")
    world, geometry = lower_long_side_top(data["scene"])
    seen = set()
    by_exposure: dict[int, list[dict]] = defaultdict(list)
    for frame in report["frames"]:
        pts = int(frame["native_pts"])
        _require(pts in alias and pts not in seen
                 and 230408448 <= pts <= 230429184,
                 "Report frame is duplicated or outside reviewed shot06")
        seen.add(pts)
        by_exposure[alias[pts]].append(frame)
    _require(len(seen) == 82, "Near-board source report omitted a reviewed frame")
    rows = []
    for representative, frames in sorted(by_exposure.items()):
        members = tuple(sorted(int(frame["native_pts"]) for frame in frames))
        by_y: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for frame in frames:
            for measurement in frame["measurements"]:
                low, high = map(int, measurement["x_interval_px"])
                y = int(measurement["y"])
                _require(0 <= low < high < 500 and 700 <= y <= 1070,
                         "Report source interval is not a visible original-pixel row")
                by_y[y].append((low, high, int(frame["native_pts"])))
        if not by_y:
            continue
        merged = [dict(y=y, x_interval_px=[min(p[0] for p in values),
                                         max(p[1] for p in values)],
                       source_native_pts=sorted(p[2] for p in values))
                  for y, values in sorted(by_y.items())]
        rows.append(dict(representative_pts=representative,
                         reviewed_native_pts=members,
                         measurements=merged,
                         world_xyz=world,
                         source_pixel_sha256={str(f["native_pts"]): f["source_pixel_sha256"]
                                              for f in frames}))
    metadata = dict(report_path=str(Path(report_path).resolve()),
                    report_sha256=expected_report_sha256,
                    user_confirmation_record=str(CONFIRMATION_RECORD.resolve()),
                    user_confirmation_record_sha256=CONFIRMATION_RECORD_SHA256,
                    source_video_sha256=SOURCE_SHA256,
                    human_confirmation_text=CONFIRMATION_TEXT,
                    human_confirmation_pts=CONFIRMATION_PTS,
                    confirmation_frame_sha256=CONFIRMATION_PNG_SHA256,
                    geometry=geometry,
                    reviewed_native_frames=82,
                    measured_native_frames=report["coverage"]["observed_native_frames"],
                    unique_reviewed_exposures=len(by_exposure),
                    graph_exposure_factors=len(rows),
                    measured_graph_scan_rows=sum(len(row["measurements"]) for row in rows),
                    exposure_alias_policy="union same-row intervals of confirmed identical exposures; one factor per representative",
                    source_pixel_interval_policy=report["measurement_policy"],
                    outside_interval_sigma_px=EDGE_SIGMA_PX,
                    sigma_role="explicit modeling assumption, not a measured covariance",
                    source_uv_changed=False, old_foreground_packet_changed=False,
                    solver_started=False)
    return rows, metadata


def _closest_interval_to_curve(a: np.ndarray, b: np.ndarray,
                               low: float, high: float, y: float
                               ) -> tuple[np.ndarray, int]:
    """One 2-D vector from a projected curve to a horizontal pixel interval."""
    direction = b - a
    length2 = np.maximum(np.sum(direction * direction, axis=1), 1e-12)
    residuals = []
    for endpoint in (a, b):
        q = np.column_stack((np.clip(endpoint[:, 0], low, high),
                             np.full(len(endpoint), y)))
        residuals.append(endpoint - q)
    for qx in (low, high):
        q = np.array([qx, y])
        t = np.clip(np.sum((q[None, :] - a) * direction, axis=1) / length2, 0., 1.)
        residuals.append(a + t[:, None] * direction - q)
    # Segment crossings with the observed horizontal interval have zero cost.
    with np.errstate(divide="ignore", invalid="ignore"):
        crossing_t = (y - a[:, 1]) / direction[:, 1]
    crossing_x = a[:, 0] + crossing_t * direction[:, 0]
    crossing = (np.isfinite(crossing_t) & (crossing_t >= 0.) &
                (crossing_t <= 1.) & (crossing_x >= low) &
                (crossing_x <= high))
    candidates = np.stack(residuals, axis=0)
    squared = np.sum(candidates * candidates, axis=2)
    squared[:, crossing] = np.inf
    if np.any(crossing):
        return np.zeros(2), int(np.flatnonzero(crossing)[0])
    kind, segment = np.unravel_index(np.argmin(squared), squared.shape)
    return candidates[kind, segment], int(segment)


def nearboard_interval_residual(camera25: np.ndarray, world_xyz: np.ndarray,
                                measurements: list[dict], *, source_keys=()
                                ) -> np.ndarray:
    """Compare the entire central projected physical lower side with source bands.

    Fixed <=0.25 scene-unit world subdivisions preserve the original straight
    geometry.  Every winning visible chord has a checked central midpoint and
    <=0.25 px sampled chord error; no invalid optical gap is connected.
    """
    xyz = np.asarray(world_xyz, np.float64)
    mids = (xyz[:-1] + xyz[1:]) * .5
    try:
        uv, _, valid = project_central_valid(camera25, xyz)
        uv_mid, _, valid_mid = project_central_valid(camera25, mids)
    except (cv2.error, FloatingPointError, OverflowError) as exc:
        error = BoardIntersectionError("nearboard_projection_domain", np.arange(len(xyz)))
        error.factor_family = "original_secondary_lower_board_top"
        error.source_keys = tuple(source_keys)
        raise error from exc
    segments = np.flatnonzero(valid[:-1] & valid_mid & valid[1:])
    if not len(segments):
        error = BoardIntersectionError("nearboard_no_central_curve", np.arange(len(xyz) - 1))
        error.factor_family = "original_secondary_lower_board_top"
        error.source_keys = tuple(source_keys)
        raise error
    a, b = uv[segments], uv[segments + 1]
    chord_error = np.linalg.norm(uv_mid[segments] - .5 * (a + b), axis=1)
    residuals = []
    winning = []
    for index, measurement in enumerate(measurements):
        low, high = map(float, measurement["x_interval_px"])
        value, selected = _closest_interval_to_curve(
            a, b, low, high, float(measurement["y"]))
        if chord_error[selected] > MAX_PROJECTED_CHORD_ERROR_PX:
            error = BoardIntersectionError("nearboard_curve_underresolved", [index])
            error.factor_family = "original_secondary_lower_board_top"
            error.source_keys = tuple(source_keys)
            error.sampled_chord_error_px = float(chord_error[selected])
            raise error
        residuals.extend(value)
        winning.append(int(segments[selected]))
    # The saved model used to miss a several-pixel Brown arc by projecting
    # long straight-side chords.  Check 1/4 and 3/4 of each winning leaf as
    # well as the midpoint checked above.  Samples outside the central optical
    # branch cannot be bridged by a straight image chord.
    unique = np.asarray(sorted(set(winning)), np.int64)
    fraction = np.asarray([.25, .75], np.float64)
    points = (xyz[unique, None, :]
              + fraction[None, :, None]
              * (xyz[unique + 1] - xyz[unique])[:, None, :])
    extra_uv, _, extra_valid = project_central_valid(
        camera25, points.reshape(-1, 3))
    extra_uv = extra_uv.reshape(len(unique), 2, 2)
    extra_valid = extra_valid.reshape(len(unique), 2)
    projected_chord = (uv[unique, None, :] + fraction[None, :, None]
                       * (uv[unique + 1] - uv[unique])[:, None, :])
    extra_error = np.linalg.norm(extra_uv - projected_chord, axis=2)
    if not extra_valid.all() or np.max(extra_error) > MAX_PROJECTED_CHORD_ERROR_PX:
        error = BoardIntersectionError("nearboard_curve_underresolved", unique)
        error.factor_family = "original_secondary_lower_board_top"
        error.source_keys = tuple(source_keys)
        error.sampled_chord_error_px = float(np.max(
            np.where(extra_valid & np.isfinite(extra_error), extra_error, np.inf)))
        raise error
    result = np.asarray(residuals, np.float64)
    if not np.isfinite(result).all():
        error = BoardIntersectionError("nearboard_curve_nonfinite", [0])
        error.factor_family = "original_secondary_lower_board_top"
        error.source_keys = tuple(source_keys)
        raise error
    return result


def _callback(world_xyz: np.ndarray, measurements: list[dict],
              template25: np.ndarray, source_keys: tuple[int, ...]):
    def one(state):
        physical = np.asarray(state, np.float64)
        _require(physical.shape in ((11,), (15,)) and np.isfinite(physical).all(),
                 "Near-board callback needs one finite physical11/15 camera")
        camera = from_state(physical, template25)
        return nearboard_interval_residual(camera, world_xyz, measurements,
                                           source_keys=source_keys)

    def residual(states):
        return one(states[0])

    def derivative(states):
        state = np.asarray(states[0], np.float64)
        base = one(state)
        result = np.empty((len(base), len(state)))
        for column in range(len(state)):
            step = 1e-6 * max(1., abs(float(state[column])))
            high, low = state.copy(), state.copy()
            high[column] += step
            low[column] -= step
            try:
                result[:, column] = (one(high) - one(low)) / (2. * step)
            except BoardIntersectionError as exc:
                exc.evaluation_stage = "jacobian"
                raise
        _require(np.isfinite(result).all(), "Near-board physical Jacobian nonfinite")
        return [result]

    return residual, derivative


def attach_secondary_nearboard_top_factors(graph, data: dict, model,
                                           initial: Mapping[int, np.ndarray],
                                           rows: list[dict]) -> dict:
    """Opt-in full-group factor adapter; no state is fixed and no solve starts."""
    _require(data["camera_id"] == "secondary_long_side",
             "Near-board factors require the secondary camera group")
    active = set(map(int, initial))
    added = 0
    for row in rows:
        pts = int(row["representative_pts"])
        _require(pts in model.frames and model.frames[pts].camera_id == data["camera_id"],
                 "Near-board exposure is not in the continuous secondary model")
        measurements = row["measurements"]
        _require(bool(measurements), "Empty near-board factor")
        source_keys = tuple(map(int, row["reviewed_native_pts"]))
        raw, jac = _callback(row["world_xyz"], measurements,
                             model.frames[pts].template25, source_keys)
        raw, jac = pointwise_huber(raw, jac, group_size=2,
                                   delta=2. * EDGE_SIGMA_PX)
        sigma = EDGE_SIGMA_PX * np.sqrt(len(measurements))
        adapted = adapt_factor((pts,), raw, jac,
                               dict(residual_dim=2 * len(measurements), sigma=sigma),
                               active, data["states"], model)
        _require(adapted is not None, "Near-board factor has no active camera dependency")
        keys, residual, derivative, options = adapted

        def checked_residual(q, *, fn=residual, native=source_keys):
            try:
                return fn(q)
            except BoardIntersectionError as exc:
                exc.evaluation_stage = "residual"
                exc.factor_family = "original_secondary_lower_board_top"
                exc.source_keys = native
                raise

        def checked_derivative(q, *, fn=derivative, native=source_keys):
            try:
                return fn(q)
            except BoardIntersectionError as exc:
                exc.evaluation_stage = "jacobian"
                exc.factor_family = "original_secondary_lower_board_top"
                exc.source_keys = native
                raise

        graph.add_observation(keys, checked_residual, jac_fn=checked_derivative,
                              **options)
        added += 1
    return dict(factor_family="original_secondary_lower_board_top",
                source_exposure_factors=added,
                original_source_uv_modified=False,
                old_foreground_packet_modified=False,
                existing_graph_factors_modified=False,
                solve_started=False)

