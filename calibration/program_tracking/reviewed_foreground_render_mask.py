"""Source-pixel foreground restoration mask for reviewed secondary footage.

The frozen 71-frame foreground/advertisement boundary supplies the main mask.
At its entry and exit, a small ramp wedge is measured from each current
original frame's lower-left ice pixels.  Neither branch uses a camera pose,
projected grid, fitting residual, or visual acceptance outcome.

``ReviewedForegroundMasks(manifest_path, source_sha256, expected_pts)`` is the
renderer adapter: ``get(pts)`` returns a native 1080x1920 bool foreground mask
or ``None``.  The caller ORs it with its existing person mask before drawing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter
from typing import Iterable

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()

import cv2
import numpy as np

import secondary_foreground_source_mask as frozen_source
from source_board_support import _decode_native_runs


TMP = Path("D:/CodexData/RinkSpace_program_tracking_tmp")
DEFAULT_MANIFEST = (TMP / "continuous_camera_model_v2" /
                    "secondary_visibility_source_review" /
                    "foreground_source_mask" / "foreground_exclusion_manifest.json")
DEFAULT_QA = (TMP / "continuous_camera_model_v2" /
              "secondary_visibility_source_review" / "render_foreground_source_qa")
MANIFEST_SHA256 = "a77abdb70feafcf2245601a179d33d0115278c818a2fa86f2b5738b2c3a376c7"
FRAME_HW = (1080, 1920)
TRANSITION_PTS = tuple(range(230408448, 230409984, 256)) + tuple(
    range(230428160, 230429440, 256))
TRANSITION_ROWS = (900, 930, 960, 990, 1020, 1050, 1070)
FROZEN_BAND_PX = 20.
TRANSITION_BAND_PX = 8.


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _timeline_binding(review: dict, source_shots_path: Path | None) -> dict:
    """Accept an explicit scope extension only when old core bytes match exactly."""
    recorded_path = Path(review["source_shots_path"])
    path = recorded_path if source_shots_path is None else Path(source_shots_path)
    digest = frozen_source.digest(path)
    expected = review["source_shots_sha256"]
    metadata = dict(recorded_path=str(recorded_path.resolve()),
                    recorded_sha256=expected, current_path=str(path.resolve()),
                    current_sha256=digest, analysis_scope_only_extension=False)
    if digest == expected:
        return metadata
    _require(source_shots_path is not None,
             "Source foreground timeline differs; an explicit source timeline is required")
    timeline = json.loads(path.read_text(encoding="utf8"))
    _require("analysis_scope" in timeline,
             "Source foreground timeline changed beyond an explicit analysis scope")
    core = dict(timeline)
    scope = core.pop("analysis_scope")
    # This is the actual historical serialization. The expected review hash
    # remains untouched; any source, cut or other core edit fails this check.
    frozen_bytes = json.dumps(core, ensure_ascii=False, indent=2).replace(
        "\n", "\r\n").encode("utf8")
    import hashlib
    core_digest = hashlib.sha256(frozen_bytes).hexdigest()
    _require(core_digest == expected,
             "Source foreground timeline core differs from the frozen review")
    metadata.update(analysis_scope_only_extension=True,
                    reconstructed_core_sha256=core_digest,
                    reconstructed_core_byte_count=len(frozen_bytes),
                    serialization="UTF-8 json indent=2, CRLF, no trailing newline",
                    current_analysis_scope=scope)
    return metadata


def _transition_row_edge(gray: np.ndarray, y: int) -> int:
    """Measure the left-connected dark ramp against nearby original ice."""
    local = cv2.GaussianBlur(gray[y-4:y+5, :220], (11, 5), 0).mean(axis=0)
    ice = float(np.median(local[150:220]))
    dark = local < ice-22.
    if not dark[0] and local[0] > ice-17.:
        return -1
    # A five-pixel majority bridges one compressed bright fleck within the
    # visible gray wedge, without collecting distant skater/rink pixels.
    contiguous = np.convolve(dark.astype(np.int8), np.ones(5), "same") >= 3
    first_light = np.flatnonzero(~contiguous[:150])
    return int(first_light[0]) if len(first_light) else 150


def _transition_line(frame: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    samples = [(y, _transition_row_edge(gray, y)) for y in TRANSITION_ROWS]
    positive = [(y, x) for y, x in samples if x > 0]
    _require(len(positive) >= 2 and positive[-1][0]-positive[0][0] >= 20,
             "Original lower-left transition wedge lost its pixel boundary")
    ys = np.asarray([y for y, _ in positive], np.float64)
    xs = np.asarray([x for _, x in positive], np.float64)
    slope, intercept = np.polyfit(ys, xs, 1)
    bottom_x = slope*1079+intercept
    error = float(np.max(np.abs(xs-(slope*ys+intercept))))
    _require(.28 <= slope <= .65 and 15 <= bottom_x <= 95 and error <= 5.,
             "Transition ramp edge no longer fits the native source image")
    return dict(slope=float(slope), intercept=float(intercept),
                bottom_x_px=float(bottom_x), max_fit_error_px=error,
                measured_rows=[dict(y=int(y), edge_x_px=int(x))
                               for y, x in samples])


def _filled_left_mask(edge_by_y: np.ndarray, band_px: float,
                      first_y: int) -> tuple[np.ndarray, np.ndarray]:
    """Return definite interior and boundary-unknown masks, tightly cropped."""
    interior = np.zeros(FRAME_HW, bool)
    boundary = np.zeros(FRAME_HW, bool)
    for offset, x_edge in enumerate(edge_by_y):
        y = first_y + offset
        if y < 0 or y >= FRAME_HW[0]:
            continue
        # Prefix semantics are deliberate: every pixel left of the measured
        # edge is the occluding ramp, and only a narrow band reaches ice.
        definitely = max(0, min(FRAME_HW[1], int(np.floor(x_edge-band_px))))
        possibly = max(0, min(FRAME_HW[1], int(np.ceil(x_edge+band_px))))
        if definitely:
            interior[y, :definitely] = True
        if possibly > definitely:
            boundary[y, definitely:possibly] = True
    return interior, boundary


class ReviewedForegroundMask:
    """Original-source foreground occlusion, independently composable with SAM2."""

    def __init__(self, manifest_path: Path = DEFAULT_MANIFEST, *,
                 source_path: Path | None = None,
                 source_sha256: str | None = None,
                 source_shots_path: Path | None = None):
        path = Path(manifest_path)
        _require(frozen_source.digest(path) == MANIFEST_SHA256,
                 "Frozen source foreground manifest changed")
        review = json.loads(path.read_text(encoding="utf8"))
        _require(review["schema"] == "rinkspace.secondary-foreground-source-mask.v1"
                 and review["selected_camera_id"] == "secondary_long_side"
                 and review["selected_shot_indices"] == [0, 2, 6]
                 and review["detector_module_sha256"] ==
                     frozen_source.digest(Path(frozen_source.__file__))
                 and Path(review["detector_module_path"]).resolve() ==
                     Path(frozen_source.__file__).resolve(),
                 "Source foreground provenance differs from the frozen review")
        timeline_binding = _timeline_binding(review, source_shots_path)
        _require(source_path is not None or source_sha256 is not None,
                 "Renderer must bind an original source path or SHA-256")
        if source_path is not None:
            _require(Path(source_path).resolve() ==
                     Path(review["source_video_path"]).resolve()
                     and frozen_source.digest(Path(source_path)) ==
                         review["source_video_sha256"],
                     "Render video is not the reviewed native original")
        if source_sha256 is not None:
            _require(source_sha256 == review["source_video_sha256"],
                     "Render source SHA-256 differs from the source review")
        self.manifest_path = path.resolve()
        self.review = review
        self.boundaries = {int(k): v for k, v in review["boundary_by_pts"].items()}
        _require(set(self.boundaries) == set(range(230409984, 230428160, 256))
                 and len(TRANSITION_PTS) == 11,
                 "Foreground source-time coverage differs")
        self.provenance = dict(
            manifest_path=str(self.manifest_path), manifest_sha256=MANIFEST_SHA256,
            source_video_sha256=review["source_video_sha256"],
            source_timeline_binding=timeline_binding,
            shot_index=6, full_boundary_frame_count=len(self.boundaries),
            full_boundary_first_pts=min(self.boundaries),
            full_boundary_last_pts=max(self.boundaries),
            transition_pts=list(TRANSITION_PTS),
            full_boundary_method="frozen native-image ice edge plus blue-ad leading edge",
            transition_method="per-original-frame lower-left luminance edge; 7 measured rows",
            full_boundary_unknown_band_px=FROZEN_BAND_PX,
            transition_unknown_band_px=TRANSITION_BAND_PX,
            mask_extent="only left foreground prefix; no camera, projection or residual input")

    def foreground(self, pts: int, original_bgr: np.ndarray | None = None
                   ) -> tuple[np.ndarray, dict]:
        """Return bool foreground mask and source evidence for one native PTS."""
        stamp = int(pts)
        if stamp in self.boundaries:
            y = np.arange(420, FRAME_HW[0], dtype=np.float64)
            edge = frozen_source._edge_x(y, self.boundaries[stamp])
            interior, uncertain = _filled_left_mask(edge, FROZEN_BAND_PX, 420)
            status = "frozen_full_source_boundary"
            evidence = dict(board_band_edge_x=self.boundaries[stamp]["board_band_edge_x"],
                            slope=self.boundaries[stamp]["slope"],
                            intercept=self.boundaries[stamp]["intercept"])
        elif stamp in TRANSITION_PTS:
            _require(original_bgr is not None,
                     "Bottom transition requires its exact native source image")
            frame = np.asarray(original_bgr)
            _require(frame.shape == (*FRAME_HW, 3) and frame.dtype == np.uint8,
                     "Foreground mask requires an original 1920x1080 BGR frame")
            evidence = _transition_line(frame)
            y = np.arange(880, FRAME_HW[0], dtype=np.float64)
            edge = evidence["slope"]*y+evidence["intercept"]
            interior, uncertain = _filled_left_mask(edge, TRANSITION_BAND_PX, 880)
            status = "native_pixel_bottom_transition"
        else:
            empty = np.zeros(FRAME_HW, bool)
            return empty, dict(pts=stamp, status="no_reviewed_foreground",
                               interior_pixels=0, uncertain_boundary_pixels=0,
                               foreground_pixels=0)
        mask = interior | uncertain
        _require(not np.any(mask[:, 550:]),
                 "Source foreground mask reached unrelated distant ice")
        metadata = dict(pts=stamp, status=status,
                        interior_pixels=int(interior.sum()),
                        uncertain_boundary_pixels=int(uncertain.sum()),
                        foreground_pixels=int(mask.sum()),
                        source_evidence=evidence)
        return mask, metadata

    def get(self, pts: int, original_bgr: np.ndarray,
            person_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        """Return original-source foreground OR existing person mask for draw()."""
        foreground, metadata = self.foreground(pts, original_bgr)
        if person_mask is None:
            return foreground, metadata
        people = np.asarray(person_mask, bool)
        _require(people.shape == FRAME_HW, "Person and foreground mask sizes differ")
        combined = foreground | people
        return combined, {**metadata,
                          "person_pixels": int(people.sum()),
                          "union_pixels": int(combined.sum()),
                          "composition": "original_source_foreground OR existing_person_mask"}


class ReviewedForegroundMasks:
    """Renderer adapter: exact-PTS ``get`` with no camera or frame dependency.

    Only the eleven bottom-transition frames are decoded from the verified
    original video during initialization.  The seventy-one frozen source
    boundaries are rasterized on demand.  ``None`` means no reviewed foreground
    at that PTS, so an unchanged renderer may keep its existing person mask.
    """

    def __init__(self, manifest_path: Path, source_sha256: str,
                 expected_pts: Iterable[int], *,
                 source_shots_path: Path | None = None):
        stamps = tuple(map(int, expected_pts))
        _require(stamps and len(set(stamps)) == len(stamps),
                 "Renderer expected source PTS are empty or repeated")
        self.expected_pts = set(stamps)
        self.inner = ReviewedForegroundMask(manifest_path,
            source_sha256=source_sha256, source_shots_path=source_shots_path)
        required = set(self.inner.boundaries) | set(TRANSITION_PTS)
        _require(required <= self.expected_pts,
                 "Renderer PTS omit reviewed shot06 foreground frames")
        video = Path(self.inner.review["source_video_path"])
        _require(frozen_source.digest(video) == source_sha256,
                 "Renderer foreground transition decoder differs from original video")
        self.transition_masks: dict[int, np.ndarray] = {}
        self.transition_evidence: dict[int, dict] = {}
        for pts, original in _decode_native_runs(video, list(TRANSITION_PTS)):
            mask, evidence = self.inner.foreground(pts, original)
            _require(evidence["status"] == "native_pixel_bottom_transition"
                     and np.any(mask),
                     f"Original entrance/exit wedge did not produce a mask: {pts}")
            self.transition_masks[pts] = mask
            self.transition_evidence[pts] = evidence
        _require(set(self.transition_masks) == set(TRANSITION_PTS),
                 "Original source transition PTS did not decode exactly")
        self.calls = Counter()
        self.masked_pixel_sum = 0
        self.masked_frames: dict[int, int] = {}

    def get(self, pts: int) -> np.ndarray | None:
        stamp = int(pts)
        _require(stamp in self.expected_pts,
                 f"Render PTS outside verified source timeline: {stamp}")
        if stamp in self.inner.boundaries:
            mask, evidence = self.inner.foreground(stamp)
        elif stamp in self.transition_masks:
            mask = self.transition_masks[stamp]
            evidence = self.transition_evidence[stamp]
        else:
            self.calls["no_reviewed_foreground"] += 1
            return None
        self.calls[evidence["status"]] += 1
        pixels = int(np.count_nonzero(mask))
        self.masked_pixel_sum += pixels
        self.masked_frames[stamp] = pixels
        return mask

    def metadata(self) -> dict:
        return dict(**self.inner.provenance,
                    expected_pts_count=len(self.expected_pts),
                    total_get_calls=int(sum(self.calls.values())),
                    get_calls_by_status=dict(self.calls),
                    unique_masked_pts_count=len(self.masked_frames),
                    masked_source_pixel_sum=self.masked_pixel_sum,
                    masked_pixels_by_pts={str(p): count
                                          for p, count in sorted(self.masked_frames.items())},
                    composition="caller ORs this source foreground mask with existing person mask")


def audit_original_pixels(output: Path, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    """Small original-image QA; no camera files or renderer output are read."""
    review = json.loads(Path(manifest_path).read_text(encoding="utf8"))
    video = Path(review["source_video_path"])
    masker = ReviewedForegroundMask(manifest_path, source_path=video)
    samples = sorted(set(TRANSITION_PTS) |
                     {230407424, 230409984, 230411008, 230414080,
                      230416128, 230422784, 230427904, 230429952})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for pts, frame in _decode_native_runs(video, samples):
        mask, metadata = masker.foreground(pts, frame)
        if pts == 230411008:
            _require(not mask[552, 100], "Genuine blue board advertisement was masked")
        if pts == 230414080:
            _require(mask[558, 42], "Known near foreground was not masked")
        if pts == 230416128:
            _require(mask[518, 88], "Known near foreground was not masked")
        if pts in (230407424, 230429952):
            _require(not np.any(mask), "Unaffected transition-bracket ice was masked")
        illustrated = frame.copy()
        # Semi-transparent tint is confined to the actual mask so reviewers
        # can inspect whether far ice/printed blue advertising stays visible.
        if np.any(mask):
            illustrated[mask] = np.rint(.58*frame[mask] +
                                        .42*np.array([15, 30, 240])).astype(np.uint8)
        cv2.putText(illustrated, f"Source PTS {pts}  {metadata['status']}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
        path = output / f"source_mask_pts{pts}.jpg"
        _require(cv2.imwrite(str(path), illustrated[:, :600],
                             [cv2.IMWRITE_JPEG_QUALITY, 91]),
                 f"Could not save source-only render-mask QA: {path}")
        rows.append({**metadata, "qa_image_path": str(path.resolve()),
                     "qa_image_sha256": frozen_source.digest(path)})
    _require({row["pts"] for row in rows} == set(samples),
             "Missing original source frame in render-mask audit")
    report = dict(schema="rinkspace.reviewed-foreground-render-source-qa.v1",
                  provenance=masker.provenance, native_frames_audited=len(rows),
                  transition_frames_audited=len(TRANSITION_PTS), frames=rows,
                  limitation="This restores only the observed foreground; it does not validate grid alignment")
    report_path = output / "source_mask_qa.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_QA)
    args = parser.parse_args()
    report = audit_original_pixels(args.output, args.manifest)
    print(json.dumps(dict(qa_dir=str(args.output.resolve()),
                          frames=report["native_frames_audited"],
                          transitions=report["transition_frames_audited"],
                          total_source_pixels=sum(row["foreground_pixels"]
                                                  for row in report["frames"])),
                     ensure_ascii=False))
