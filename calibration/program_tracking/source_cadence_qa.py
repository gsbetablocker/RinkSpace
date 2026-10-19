"""Source-image cadence diagnostics collected during the programme render.

These are observations about the encoded source frames, not independent
camera-localization evidence or proof of an earlier capture frame rate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from .projection import safe_project
except ImportError:
    from projection import safe_project


ROI_XYXY = (560, 260, 1650, 940)  # Avoid the checked top-left and lower-right broadcast graphics.
GRAY_SIZE = (320, 180)
PHASE_COUNT = 6
SOURCE_PTS_STEP = 256
REFERENCE_COMPARISON = Path(
    "D:/CodexData/RinkSpace_program_tracking_tmp/source_cadence_comparison/"
    "source_cadence_comparison.json"
)


def _shot_index(pts: int, shots: list[dict]) -> int:
    for index, shot in enumerate(shots):
        if int(shot["start_pts"]) <= pts < int(shot["end_pts"]):
            return index
    raise ValueError(f"Source PTS {pts} is outside reviewed shots")


def _intervals(rows: list[dict], predicate) -> list[dict]:
    intervals, current = [], []
    for row in rows:
        if predicate(row) and (not current or (
                row["previous_pts"] == current[-1]["current_pts"]
                and row["shot_index"] == current[-1]["shot_index"])):
            current.append(row)
        else:
            if len(current) >= 6:
                intervals.append({"shot_index": current[0]["shot_index"],
                                  "first_previous_pts": current[0]["previous_pts"],
                                  "last_current_pts": current[-1]["current_pts"],
                                  "pair_count": len(current)})
            current = [row] if predicate(row) else []
    if len(current) >= 6:
        intervals.append({"shot_index": current[0]["shot_index"],
                          "first_previous_pts": current[0]["previous_pts"],
                          "last_current_pts": current[-1]["current_pts"],
                          "pair_count": len(current)})
    return intervals


class SourceCadenceQA:
    """Compare consecutive original images and fixed scene projections by PTS."""

    def __init__(self, scene_vertices: np.ndarray, shots: list[dict], *,
                 source: str, source_sha256: str, model_sha256: str,
                 comparison_path: Path = REFERENCE_COMPARISON):
        vertices = np.asarray(scene_vertices, np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
            raise ValueError("Cadence QA requires fixed 3D scene vertices")
        if not np.isfinite(vertices).all() or not shots:
            raise ValueError("Cadence QA scene and shots must be finite/nonempty")
        sample = np.unique(np.rint(np.linspace(0, len(vertices)-1,
                                              min(192, len(vertices)))).astype(int))
        self.vertices = np.ascontiguousarray(vertices[sample])
        self.shots = shots
        self.source = str(source)
        self.source_sha256 = source_sha256
        self.model_sha256 = model_sha256
        self.comparison_path = Path(comparison_path)
        self.rows: list[dict] = []
        self.previous: tuple[int, int, np.ndarray, np.ndarray, np.ndarray] | None = None

    def _projection(self, camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        uv, depth, valid = safe_project(np.asarray(camera, np.float64), self.vertices)
        valid = (valid & np.isfinite(uv).all(axis=1) & np.isfinite(depth)
                 & (depth > .05) & (uv[:, 0] >= 0) & (uv[:, 0] < 1920)
                 & (uv[:, 1] >= 0) & (uv[:, 1] < 1080))
        return uv, valid

    def observe(self, pts: int, original_bgr: np.ndarray, camera: np.ndarray) -> None:
        pts = int(pts)
        if original_bgr.shape[:2] != (1080, 1920):
            raise ValueError("Cadence QA requires the original 1920x1080 frame")
        shot = _shot_index(pts, self.shots)
        x0, y0, x1, y1 = ROI_XYXY
        gray = cv2.resize(cv2.cvtColor(original_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY),
                          GRAY_SIZE, interpolation=cv2.INTER_AREA)
        uv, visible = self._projection(camera)
        if self.previous is not None:
            prior_pts, prior_shot, prior_gray, prior_uv, prior_visible = self.previous
            if pts-prior_pts != SOURCE_PTS_STEP:
                raise ValueError("Cadence QA source PTS chain is not consecutive")
            difference = np.abs(gray.astype(np.int16)-prior_gray.astype(np.int16))
            same_shot = prior_shot == shot
            shared = prior_visible & visible if same_shot else np.zeros(len(visible), bool)
            distance = np.linalg.norm(uv[shared]-prior_uv[shared], axis=1)
            self.rows.append({
                "previous_pts": prior_pts, "current_pts": pts,
                "shot_index": shot, "previous_shot_index": prior_shot,
                "same_shot": same_shot, "cut_boundary": not same_shot,
                "phase_mod6_diagnostic": (pts//SOURCE_PTS_STEP) % PHASE_COUNT,
                "roi_gray_mae": round(float(difference.mean()), 5),
                "roi_gray_same_fraction": round(float(np.mean(gray == prior_gray)), 6),
                "visible_common_scene_vertices": int(len(distance)),
                "grid_vertex_shift_p50_px": round(float(np.median(distance)), 4) if len(distance) else None,
                "grid_vertex_shift_p95_px": round(float(np.percentile(distance, 95)), 4) if len(distance) else None,
            })
        self.previous = (pts, shot, gray, uv, visible)

    def report(self, *, expected_pair_count: int | None = None) -> dict:
        if expected_pair_count is not None and len(self.rows) != expected_pair_count:
            raise ValueError("Cadence QA does not cover every adjacent source PTS pair")
        rows = [row.copy() for row in self.rows]
        for index, row in enumerate(rows):
            row["local_other_phase_median_mae"] = None
            if not row["same_shot"]:
                row["cadence_label"] = "camera_cut_not_comparable"
                continue
            candidates = [rows[i]["roi_gray_mae"]
                          for i in range(max(0, index-12), min(len(rows), index+13))
                          if i != index and rows[i]["same_shot"]
                          and rows[i]["shot_index"] == row["shot_index"]
                          and rows[i]["phase_mod6_diagnostic"] != row["phase_mod6_diagnostic"]]
            if len(candidates) < 4:
                row["cadence_label"] = "insufficient_same_shot_context"
                continue
            baseline = float(np.median(candidates))
            row["local_other_phase_median_mae"] = round(baseline, 5)
            mae = row["roi_gray_mae"]
            if mae <= .3 and baseline >= max(1., 4.*mae) and baseline-mae >= .7:
                row["cadence_label"] = "near_repeat_candidate"
            elif baseline < 1. and mae <= .3:
                row["cadence_label"] = "low_motion_ambiguous"
            else:
                row["cadence_label"] = "not_discriminative_near_repeat"
        labels = {label: sum(row["cadence_label"] == label for row in rows)
                  for label in sorted({row["cadence_label"] for row in rows})}
        phases = []
        for phase in range(PHASE_COUNT):
            selected = [row for row in rows if row["same_shot"]
                        and row["phase_mod6_diagnostic"] == phase]
            phases.append({"phase": phase, "same_shot_pairs": len(selected),
                           "median_roi_gray_mae": (round(float(np.median([
                               row["roi_gray_mae"] for row in selected])), 5)
                               if selected else None),
                           "near_repeat_candidates": sum(
                               row["cadence_label"] == "near_repeat_candidate" for row in selected)})
        reference = None
        if self.comparison_path.is_file():
            reference = {"path": str(self.comparison_path.resolve()),
                         "sha256": hashlib.sha256(self.comparison_path.read_bytes()).hexdigest()}
        return {
            "schema": "rinkspace.source-cadence-render-qa.v1",
            "source": self.source, "source_sha256": self.source_sha256,
            "model_sha256": self.model_sha256,
            "source_time_base": "1/15360", "source_pts_step": SOURCE_PTS_STEP,
            "reference_sparse_comparison": reference,
            "method": {
                "original_image_roi_xyxy": list(ROI_XYXY),
                "roi_excludes_checked_broadcast_graphics": True,
                "gray_resize_wh": list(GRAY_SIZE),
                "same_fraction": "exact_equal_gray_values_after_area_resize",
                "grid_shift": "fixed_scene_vertices_visible_inside_both_1920x1080_frames",
                "scene_vertices_sampled": len(self.vertices),
                "phase": "(current_source_pts//256)%6; diagnostic only",
                "candidate_rule": "same shot; gray MAE <= 0.3; >=4 other-phase pairs within +/-12; "
                                  "their median >= max(1.0,4*MAE) and median-MAE >= 0.7",
                "limitation": "Heuristic source-image cadence QA; not independent localization "
                              "acceptance, exact RGB equality, or proof of original capture rate.",
            },
            "summary": {
                "adjacent_pair_count": len(rows),
                "within_shot_pair_count": sum(row["same_shot"] for row in rows),
                "cut_pair_count": sum(row["cut_boundary"] for row in rows),
                "label_counts": labels, "by_phase": phases,
                "low_motion_uncertain_intervals": _intervals(
                    rows, lambda row: row["cadence_label"] == "low_motion_ambiguous"),
            },
            "pairs": rows,
        }

    def save(self, path: Path, *, expected_pair_count: int, overwrite: bool = False) -> dict:
        report = self.report(expected_pair_count=expected_pair_count)
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Cadence QA appeared during rendering: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(path.name + ".writing")
        if pending.exists():
            raise FileExistsError(f"Cadence QA intermediate already exists: {pending}")
        try:
            pending.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            pending.replace(path)
        finally:
            if pending.exists():
                pending.unlink()
        return {"path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "summary": report["summary"]}
