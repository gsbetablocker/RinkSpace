"""Append independent-source Ad2 letters without changing prior contours.

Only the historical original RGB at PTS56218880 and its existing source-edge
rectification initialize the materials. Current programme points/cameras are
never read. The small native print remains an uncertain template initializer.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src", Path(__file__).parent):
    if str(p) not in sys.path:
        sys.path.append(str(p))
from calibration.program_tracking.build_ad2_source_material_template import extract_glyphs, transform, plain, sha
from calibration.program_tracking.framewise_model import feature_world

MODEL = ROOT / "events/ankara_2026/models/framewise_8a28_20261013"
DIAGNOSTICS = Path("D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/frame_by_frame_review_20261013/model_binding")
COMPONENT_KEYS = ("ISU_I", "ISU_S", "U", "hyphen", "SKATING_S", "K", "SKATING_A", "SKATING_T", "SKATING_I", "SKATING_N", "SKATING_G", "dot", "COM_C", "COM_O", "COM_M")


def contour_digest(contours):
    return hashlib.sha256(json.dumps(contours, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def append(model_dir=MODEL, diagnostics=DIAGNOSTICS):
    model_dir, diagnostics = Path(model_dir), Path(diagnostics)
    template_path, registry_path, model_path = (model_dir / n for n in ("ad2_material_template.json", "feature_registry.json", "model.json"))
    template = json.loads(template_path.read_text(encoding="utf-8-sig"))
    original = deepcopy(template["contours"])
    original_digest = contour_digest(original)
    source = template["source"]
    source_path = Path(source["source_frame_path"])
    if sha(source_path) != source["source_frame_sha256"]:
        raise ValueError("Independent original source image SHA changed")
    rgb = cv2.cvtColor(cv2.imread(str(source_path)), cv2.COLOR_BGR2RGB)
    if hashlib.sha256(rgb.tobytes()).hexdigest() != source["decoded_native_rgb_sha256"]:
        raise ValueError("Independent original source RGB changed")
    quad = np.asarray(template["source_corners_from_measured_edges_TL_TR_BR_BL"], float)
    requested = [(i, key) for i, key in enumerate(COMPONENT_KEYS) if key not in original]
    if not requested:
        if "COM_O_inner" not in original:
            raise ValueError("Existing full text is missing its O inner contour; do not rewrite original outer shapes")
        print(json.dumps({"status": "already_initialized", "existing_contours_unchanged": True,
                          "material_template_sha256": sha(template_path)}, ensure_ascii=False))
        return
    glyphs, rectified, mask, matrix, display_matrix, component_count = extract_glyphs(
        rgb, quad, glyph_ids=tuple(key for _, key in requested), component_indices=tuple(i for i, _ in requested))
    if component_count != len(COMPONENT_KEYS):
        raise ValueError("Independent original print component count/letter ordering changed")
    if not np.array_equal(matrix, np.asarray(template["source_to_material_homography"], float)):
        raise ValueError("Existing source-to-material chart changed")
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    externals = [(c, h) for c, h in zip(contours, hierarchy[0]) if h[3] < 0 and cv2.contourArea(c) > 60.]
    externals.sort(key=lambda row: cv2.boundingRect(row[0])[0])
    o = next((g for g in glyphs if g["id"] == "COM_O"), template["contours"].get("COM_O"))
    if o is None:
        raise ValueError("Missing independent O external contour")
    child = int(externals[13][1][2])
    holes = []
    while child >= 0:
        native = transform(contours[child][:, 0], np.linalg.inv(display_matrix))
        native = cv2.approxPolyDP(native.astype(np.float32), .35, True)[:, 0].astype(float)
        if not np.array_equal(native[0], native[-1]):
            native = np.vstack((native, native[0]))
        inner = deepcopy(o)
        inner.update(id="COM_O_inner", role="white_print_internal_hole_boundary", parent_contour_key="COM_O",
                     source_native_uv=native, material_uv=transform(native, matrix),
                     original_native_glyph_bbox_xyxy=[*native.min(0), *native.max(0)],
                     source_native_simplification_tolerance_px=.35,
                     internal_hole_boundary_separate_from_external_ink=True,
                     inner_hole_threshold_sensitivity_not_independently_calibrated=True,
                     threshold_sensitivity=None)
        inner.pop("rectified_component_bbox_xywh", None)
        holes.append(inner)
        child = int(hierarchy[0][child][0])
    if len(holes) != 1:
        raise ValueError("Independent original O must contain exactly one separate hole")
    o["separate_inner_contour_keys"] = ["COM_O_inner"]
    if "COM_O_inner" not in original:
        glyphs.extend(holes)
    for glyph in glyphs:
        value = plain(glyph)
        value["source_component_index_0based"] = COMPONENT_KEYS.index(glyph.get("parent_contour_key", glyph["id"]))
        value["current_programme_frame_200_points_used"] = False
        value["initial_material_chart_not_fitted_truth"] = True
        template["contours"][glyph["id"]] = value
    if any(template["contours"][key] != value for key, value in original.items()):
        raise ValueError("Appending new letters modified an existing contour")
    preview = cv2.cvtColor(rectified, cv2.COLOR_RGB2BGR)
    for key, glyph in template["contours"].items():
        xy = transform(glyph["source_native_uv"], display_matrix)
        color = (0, 255, 255) if key.endswith("_inner") else (20, 20, 255)
        cv2.polylines(preview, [np.rint(xy).astype(np.int32)], True, color, 1, cv2.LINE_AA)
    diagnostics.mkdir(parents=True, exist_ok=True)
    preview_path = diagnostics / "ad2_rectified_material_template_full_text.png"
    if not cv2.imwrite(str(preview_path), preview):
        raise OSError(preview_path)
    template["diagnostic_images"]["rectified_complete_text_template"] = str(preview_path)
    template["new_feature_append_complete_ISU_SKATING_COM"] = {
        "source_pts": source["source_pts"], "independent_source_pixels_only": True,
        "new_contour_keys": [g["id"] for g in glyphs], "source_component_order": list(COMPONENT_KEYS),
        "original_contour_count": len(original), "existing_contours_preserved_exactly": True,
        "preserved_existing_contours_sha256": original_digest,
        "current_programme_frame_points_or_camera_used": False,
        "initialization_accuracy_accepted": False,
        "native_source_panel_width_height_approx_px": [190, 33],
        "low_native_glyph_resolution_and_empirical_2px_support_remain": True,
        "rectified_1536x256_preview_adds_no_native_information": True,
        "O_inner_support_not_independently_calibrated": True,
        "larger_current_programme_source_requires_future_material_verification": True,
    }
    template_path.write_text(json.dumps(plain(template), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    registry = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    for glyph in glyphs:
        descriptor = deepcopy(registry["targets"]["Ad2.print.K"])
        descriptor.update(contour_key=glyph["id"], contour_role=glyph["role"], geometry_status="independent_low_resolution_source_template_initialization_not_fitted_truth")
        registry["targets"]["Ad2.print." + glyph["id"]] = descriptor
    registry.setdefault("display_aliases", {}).update(B4="R4", B3="R3")
    for descriptor in registry["targets"].values():
        if descriptor["kind"] == "ice_curve" and descriptor["curve_id"].split(":", 1)[0] in ("R4", "R3"):
            descriptor["display_feature_alias"] = {"R4": "B4", "R3": "B3"}[descriptor["curve_id"].split(":", 1)[0]]
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Read the latest manifest immediately before its update. Parent-owned
    # acceptance/count fields are retained; only two reference hashes change.
    model = json.loads(model_path.read_text(encoding="utf-8-sig"))
    original_model = deepcopy(model)
    model["material_templates"]["sha256"] = sha(template_path)
    model["feature_registry"]["sha256"] = sha(registry_path)
    model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    scene_path = Path(model["scene"]["path"])
    if sha(scene_path) != model["scene"]["sha256"]:
        raise ValueError("Unchanged seed scene reference mismatch")
    scene = json.loads(scene_path.read_text(encoding="utf-8-sig"))
    for key in [g["id"] for g in glyphs]:
        if not np.isfinite(feature_world(scene, registry["targets"]["Ad2.print." + key], template)).all():
            raise ValueError("Nonfinite new material entry: " + key)
    restored = deepcopy(model)
    restored["material_templates"]["sha256"] = original_model["material_templates"]["sha256"]
    restored["feature_registry"]["sha256"] = original_model["feature_registry"]["sha256"]
    if restored != original_model or contour_digest({k: template["contours"][k] for k in original}) != original_digest:
        raise ValueError("Unrelated model fields or original materials changed")
    print(json.dumps({"new_contours": {g["id"]: len(g["material_uv"]) for g in glyphs},
                      "all_previous_contours_preserved": True, "material_template_sha256": sha(template_path),
                      "feature_registry_sha256": sha(registry_path), "accepted_source_frame_count_preserved": model.get("accepted_source_frame_count"),
                      "scene_unchanged": True, "display_aliases": registry["display_aliases"], "preview": str(preview_path)}, ensure_ascii=False))


if __name__ == "__main__":
    append()
