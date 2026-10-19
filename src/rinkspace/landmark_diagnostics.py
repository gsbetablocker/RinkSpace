"""Image-only patch observations for protected, separately reviewed landmarks.

These measurements never feed the registration solver.  Template correlation
is still a candidate observation, not human subpixel ground truth.  In
particular, six corners of one logo do not establish six spatially distributed
physical landmarks or whole-rink accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry import project
from .io import dump_json


def correlate_patch(reference: np.ndarray, current: np.ndarray, xy: list[float],
                    patch_radius: int = 12, search_radius: int = 60) -> dict[str, Any]:
    """Direct-reference NCC with a quadratic peak refinement, without using H."""
    if reference.ndim != 2 or current.shape != reference.shape:
        raise ValueError("Patch correlation requires same-shape grayscale images")
    x, y = (int(round(v)) for v in xy)
    height, width = reference.shape
    if min(x, y) < patch_radius or x+patch_radius >= width or y+patch_radius >= height:
        return {"status": "unavailable", "reason": "reference_patch_outside_image"}
    patch = reference[y-patch_radius:y+patch_radius+1, x-patch_radius:x+patch_radius+1]
    if np.std(patch) < .3:
        return {"status": "unavailable", "reason": "reference_patch_textureless"}
    x0, x1 = max(0, x-search_radius-patch_radius), min(width, x+search_radius+patch_radius+1)
    y0, y1 = max(0, y-search_radius-patch_radius), min(height, y+search_radius+patch_radius+1)
    score = cv2.matchTemplate(current[y0:y1, x0:x1], patch, cv2.TM_CCOEFF_NORMED)
    _, maximum, _, where = cv2.minMaxLoc(score)
    px, py = where
    fine = np.array([float(px), float(py)])
    for axis in (0, 1):
        values = score[py, px-1:px+2] if axis == 0 and 0 < px < score.shape[1]-1 else \
            score[py-1:py+2, px] if axis == 1 and 0 < py < score.shape[0]-1 else np.array([])
        if len(values) == 3:
            denominator = float(values[0] - 2*values[1] + values[2])
            if denominator < -1e-8:
                fine[axis] += np.clip(.5*float(values[0]-values[2])/denominator, -.5, .5)
    alternative = score.copy()
    cv2.circle(alternative, (px, py), 8, -1., -1)
    _, second, _, _ = cv2.minMaxLoc(alternative)
    return {"status": "candidate", "image_xy": (fine + [x0+patch_radius, y0+patch_radius]).tolist(),
            "ncc_score": float(maximum), "competing_peak_gap": float(maximum-second),
            "peak_at_search_boundary": px in (0, score.shape[1]-1) or py in (0, score.shape[0]-1),
            "reference_patch_radius_px": patch_radius, "search_radius_px": search_radius,
            "measurement_method": "direct_reference_patch_NCC_quadratic_peak",
            "human_subpixel_groundtruth": False}


def protected_patch_diagnostic(sample_dir: str | Path, registration_dir: str | Path,
                               reference_sample_index: int, landmarks: list[dict[str, Any]],
                               patch_radius: int = 12, search_radius: int = 60) -> dict[str, Any]:
    sample_dir, output = Path(sample_dir), Path(registration_dir)
    registration = json.loads((output / "registration.json").read_text(encoding="utf-8"))
    rows = registration["frames"]
    reference_row = next(row for row in rows if row["sample_index"] == reference_sample_index)
    reference_image = cv2.imread(str(sample_dir / reference_row["image"]))
    reference = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    protected_ref = cv2.imread(str(output / reference_row["protected_mask_path"]), 0)
    for point in landmarks:
        x, y = (int(round(v)) for v in point["reference_xy"])
        if not np.all(protected_ref[max(0,y-patch_radius):y+patch_radius+1, max(0,x-patch_radius):x+patch_radius+1] > 0):
            raise ValueError("A diagnostic patch was not fully protected from registration")
    (output / "errors").mkdir(exist_ok=True)
    diagnostic_rows = []
    errors = []
    for row in rows:
        image = cv2.imread(str(sample_dir / row["image"]))
        current = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        protected = cv2.imread(str(output / row["protected_mask_path"]), 0)
        observations = []
        overlay = image.copy()
        for number, point in enumerate(landmarks):
            observation = {**point, **correlate_patch(reference, current, point["reference_xy"], patch_radius, search_radius)}
            if observation["status"] == "candidate":
                xy = observation["image_xy"]
                ix, iy = (int(round(v)) for v in xy)
                observation["protected_from_fitting"] = bool(np.all(protected[max(0,iy-patch_radius):iy+patch_radius+1, max(0,ix-patch_radius):ix+patch_radius+1] > 0))
                cv2.circle(overlay, (ix, iy), 5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(overlay, str(number+1), (ix+6, iy-6), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 255), 1, cv2.LINE_AA)
                if row["status"] == "candidate" and row.get("reference_sample_index") == reference_sample_index:
                    prediction = project(np.linalg.inv(np.asarray(row["H_reference_from_current"])), np.array([point["reference_xy"]]))[0]
                    error = float(np.linalg.norm(prediction-xy))
                    observation.update({"prediction_from_external_fit_xy": prediction.tolist(), "error_sample_px": error,
                                        "error_1920_equivalent_px": error*1920/current.shape[1]})
                    cv2.drawMarker(overlay, tuple(np.rint(prediction).astype(int)), (255, 220, 0), cv2.MARKER_CROSS, 10, 1)
                    cv2.line(overlay, (ix, iy), tuple(np.rint(prediction).astype(int)), (0, 220, 255), 1)
                    if row["sample_index"] != reference_sample_index and observation["protected_from_fitting"]:
                        errors.append(error*1920/current.shape[1])
            observations.append(observation)
        title = f"{row['source_time_s']:.3f}s | red: protected patch NCC | cyan: external-fit prediction"
        cv2.putText(overlay, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .49, (0, 255, 255), 1, cv2.LINE_AA)
        error_path = f"errors/{int(row['sample_index']):06d}.jpg"
        cv2.imwrite(str(output / error_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
        row_errors = [obs["error_1920_equivalent_px"] for obs in observations
                      if "error_1920_equivalent_px" in obs and obs.get("protected_from_fitting")]
        diagnostic_rows.append({"sample_index": row["sample_index"], "source_frame_index": row.get("source_frame_index"),
                                "pts": row["pts"], "source_time_s": row["source_time_s"], "observations": observations,
                                "error_image": error_path,
                                "median_1920_equivalent_px": float(np.median(row_errors)) if row_errors else None,
                                "p95_1920_equivalent_px": float(np.percentile(row_errors, 95)) if row_errors else None})
    summary = {"candidate_observation_count": len(errors), "reference_identity_errors_excluded": True,
               "physical_group_count": len({point["physical_group_id"] for point in landmarks}),
               "point_count": len(landmarks), "median_1920_equivalent_px": float(np.median(errors)) if errors else None,
               "p95_1920_equivalent_px": float(np.percentile(errors, 95)) if errors else None,
               "spatial_coverage_accepted": False, "formal_independent_acceptance": False}
    result = {"schema_version": "1.0", "reference_sample_index": reference_sample_index,
              "source_sha256": registration["source_sha256"], "landmarks": landmarks,
              "observation_status": "protected_patch_candidates_require_visual_review",
              "solver_uses_these_observations": False, "human_subpixel_groundtruth": False,
              "limitation": "Single protected physical-group patch correlation: not six distributed landmarks and not whole-shot validation.",
              "summary": summary, "frames": diagnostic_rows}
    dump_json(output / "independent_patch_candidates.json", result)
    return result
