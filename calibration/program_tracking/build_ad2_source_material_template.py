"""Build a provisional Ad2 material chart from an independent original exposure.

The chart contains source-pixel/normalized 2-D curves only.  It does not read the
current frame's accepted points, a camera, or any fitted world coordinates.
Native source blur and the four-edge rectification assumption remain explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rinkspace.vision_source_shapes import read_source_material_transition_profiles

DATA = Path("D:/CodexData/RinkSpace_Data/events/ankara_2026")
DEFAULT_REFERENCE = DATA / "observations/vision_training/source_ads_user_inventory_56218880.json"
DEFAULT_OUTPUT = ROOT / "events/ankara_2026/models/framewise_8a28_20261013/ad2_material_template.json"
DEFAULT_DIAGNOSTICS = DATA / "programmes/okada_mei_sp/results/frame_by_frame_review_20261013/model_binding"

# Visual search locations read from the independent native photograph.  These
# are neither old navigation-circle digitizations nor output measurements.
SEARCH_CORNERS_TL_TR_BR_BL = np.array([[1534., 349.], [1724., 390.],
                                      [1721., 421.], [1533., 380.]])
GLYPH_IDS = ("ISU_S", "U", "hyphen", "SKATING_S", "K")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def line_intersection(a, b):
    # Each line is [vx, vy, x0, y0].
    av, bv = a[:2], b[:2]
    t, _ = np.linalg.solve(np.column_stack((av, -bv)), b[2:] - a[2:])
    return a[2:] + t * av


def measure_edges(rgb, binding):
    corners = SEARCH_CORNERS_TL_TR_BR_BL
    # Keep short-side profiles away from corner joins.  Long-side profiles do
    # not enter lettering or spectators within their source-pixel brackets.
    definitions = {
        "top": (corners[0], corners[1], np.linspace(.07, .93, 12)),
        "right": (corners[1], corners[2], np.linspace(.20, .80, 5)),
        "bottom": (corners[2], corners[3], np.linspace(.07, .93, 12)),
        "left": (corners[3], corners[0], np.linspace(.20, .80, 5)),
    }
    edges, lines = {}, {}
    for name, (a, b, fractions) in definitions.items():
        tangent = (b-a) / np.linalg.norm(b-a)
        normal = np.array([-tangent[1], tangent[0]])
        queries = a + fractions[:, None] * (b-a)
        reading = read_source_material_transition_profiles(rgb, binding=binding,
            search_points_uv=queries, normals=np.tile(normal, (len(queries), 1)),
            search_intervals_px=np.tile([-6., 6.], (len(queries), 1)),
            transition="dark_to_light", expected_RGB_sha256=binding["decoded_native_rgb_sha256"],
            tangent_half_width_px=1, smoothing_sigma_px=.7,
            engineering_support_margin_px=1., keep_profiles=True,
            material_signal_rgb_weights=[.5, -1., .5])
        measured = reading["actual_source_uv"]
        if len(measured) < 3:
            raise ValueError(f"{name}: fewer than three independently measured source transitions")
        line = cv2.fitLine(measured.astype(np.float32), cv2.DIST_HUBER, 0, .001, .001).ravel().astype(float)
        perpendicular = np.array([-line[1], line[0]])
        distances = (measured-line[2:]) @ perpendicular
        lines[name] = line
        empirical_radius = max(float(np.max(np.abs(r["observed_normal_interval_px"])))
            for r in reading["readings"] if r["disposition"] == "measured_material_transition")
        edges[name] = dict(actual_native_uv=measured, profile_reading=reading,
            rectification_line_fit=line,
            source_line_fit_maximum_perpendicular_px=float(np.max(np.abs(distances))),
            empirical_blur_plus_line_residual_half_width_px=empirical_radius+float(np.max(np.abs(distances))),
            source_interval_is_calibrated_covariance=False,
            line_fit_is_rectification_initializer_not_a_replacement_for_measured_edge=True)
    quad = np.array([line_intersection(lines["left"], lines["top"]),
                     line_intersection(lines["top"], lines["right"]),
                     line_intersection(lines["right"], lines["bottom"]),
                     line_intersection(lines["bottom"], lines["left"])])
    corner_intervals = []
    for first, second in (("left", "top"), ("top", "right"),
                          ("right", "bottom"), ("bottom", "left")):
        candidates = []
        for sign_a in (-1, 1):
            for sign_b in (-1, 1):
                a, b = lines[first].copy(), lines[second].copy()
                a[2:] += sign_a * edges[first]["empirical_blur_plus_line_residual_half_width_px"] * np.array([-a[1], a[0]])
                b[2:] += sign_b * edges[second]["empirical_blur_plus_line_residual_half_width_px"] * np.array([-b[1], b[0]])
                candidates.append(line_intersection(a, b))
        candidates = np.asarray(candidates)
        corner_intervals.append(dict(native_uv_min=candidates.min(0), native_uv_max=candidates.max(0),
            interpretation="empirical source blur with fixed fitted edge directions; not confidence bounds"))
    return edges, quad, corner_intervals


def transform(points, matrix):
    return cv2.perspectiveTransform(np.asarray(points, np.float64).reshape(1, -1, 2), matrix)[0]


def extract_glyphs(rgb, quad, glyph_ids=GLYPH_IDS, component_indices=range(1, 6)):
    # Template image chart: u left to right; v top to bottom.  The existing
    # geometry uses bottom-to-top v, so lifting passes 1-template_v.
    material = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], np.float32)
    source_to_material = cv2.getPerspectiveTransform(quad.astype(np.float32), material)
    width, height = 1536, 256
    source_to_display = cv2.getPerspectiveTransform(quad.astype(np.float32),
        np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], np.float32))
    rectified = cv2.warpPerspective(rgb, source_to_display, (width, height), flags=cv2.INTER_LINEAR)
    # Actual original white ink has high G against magenta/purple material.
    # The threshold is an initializer; its sensitivity is recorded below.
    green = rectified[..., 1]
    band = np.zeros(green.shape, np.uint8)
    band[int(.15*height):int(.85*height), int(.10*width):int(.91*width)] = 1
    threshold = 153
    mask = ((green >= threshold) & (band != 0)).astype(np.uint8)*255
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    external = [(c, h) for c, h in zip(contours, hierarchy[0])
                if h[3] < 0 and cv2.contourArea(c) > 60.]
    external.sort(key=lambda item: cv2.boundingRect(item[0])[0])
    # Native source component ordering is independently visible ISU-SKATING.COM.
    # The requested first five shapes follow I, then S,U,-,S,K.
    if len(external) <= max(component_indices):
        raise ValueError("Independent source does not resolve the first six white ink components")
    glyphs = []
    inverse_display = np.linalg.inv(source_to_display)
    threshold_alternatives = {}
    for alternative in (threshold-10, threshold+10):
        alternative_mask = ((green >= alternative) & (band != 0)).astype(np.uint8)*255
        alt_contours, alt_hierarchy = cv2.findContours(alternative_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        threshold_alternatives[alternative] = [c for c, h in zip(alt_contours, alt_hierarchy[0])
            if h[3] < 0 and cv2.contourArea(c) > 60.]
    for glyph_id, component_index in zip(glyph_ids, component_indices):
        contour = external[component_index][0]
        full_source_curve = transform(contour[:, 0], inverse_display)
        source_curve = transform(contour[:, 0], inverse_display)
        # Simplify in native source pixels, so magnification cannot invent
        # precision or increase the solver point budget.
        source_curve = cv2.approxPolyDP(source_curve.astype(np.float32), .45, True)[:, 0].astype(float)
        if not np.array_equal(source_curve[0], source_curve[-1]):
            source_curve = np.vstack((source_curve, source_curve[0]))
        material_curve = transform(source_curve, source_to_material)
        x, y, w, h = cv2.boundingRect(contour)
        threshold_sensitivity = []
        for alternative, candidates in threshold_alternatives.items():
            def bbox_overlap(candidate):
                ax, ay, aw, ah = cv2.boundingRect(candidate)
                return max(0, min(x+w, ax+aw)-max(x, ax)) * max(0, min(y+h, ay+ah)-max(y, ay))
            other = max(candidates, key=bbox_overlap)
            other_source = transform(other[:, 0], inverse_display)
            distances = np.linalg.norm(full_source_curve[:, None]-other_source[None], axis=2)
            hausdorff = max(float(distances.min(0).max()), float(distances.min(1).max()))
            threshold_sensitivity.append(dict(green_threshold=alternative,
                source_native_symmetric_Hausdorff_px=hausdorff))
        source_radius = max(2., max(r["source_native_symmetric_Hausdorff_px"] for r in threshold_sensitivity))
        # Empirical 2 px source support includes unresolved blur.  Transform a
        # square support at each vertex to show the resulting material scale.
        support_corners = source_curve[:, None] + np.array([[-source_radius, -source_radius],
            [-source_radius, source_radius], [source_radius, -source_radius], [source_radius, source_radius]])
        material_support = transform(support_corners.reshape(-1, 2), source_to_material).reshape(-1, 4, 2)
        material_radius = np.max(np.abs(material_support-material_curve[:, None]), axis=(0, 1))
        glyphs.append(dict(id=glyph_id, role="white_print_external_ink_contour", closed=True,
            source_native_uv=source_curve, material_uv=material_curve,
            rectified_component_bbox_xywh=[x, y, w, h],
            original_native_glyph_bbox_xyxy=[*source_curve.min(axis=0), *source_curve.max(axis=0)],
            material_coordinate_domain="Ad2_external_purple_panel_unit_rectangle",
            source_native_boundary_uncertainty_px=source_radius,
            empirical_material_uv_support_half_width=material_radius,
            threshold_sensitivity=threshold_sensitivity,
            source_blur_and_threshold_require_future_verification=True,
            current_frame_points_used=False,
            source_native_simplification_tolerance_px=.45))
    return glyphs, rectified, mask, source_to_material, source_to_display, len(external)


def append_skating_a(output=DEFAULT_OUTPUT, diagnostics=DEFAULT_DIAGNOSTICS):
    """Append the newly visible A without rebuilding any existing contour."""
    from copy import deepcopy
    output, diagnostics = Path(output), Path(diagnostics)
    template = json.loads(output.read_text(encoding="utf-8"))
    original_contours = deepcopy(template["contours"])
    source = template["source"]
    source_path = Path(source["source_frame_path"])
    if sha(source_path) != source["source_frame_sha256"]:
        raise ValueError("Independent source file changed")
    rgb = cv2.cvtColor(cv2.imread(str(source_path)), cv2.COLOR_BGR2RGB)
    if hashlib.sha256(rgb.tobytes()).hexdigest() != source["decoded_native_rgb_sha256"]:
        raise ValueError("Independent source RGB changed")
    quad = np.asarray(template["source_corners_from_measured_edges_TL_TR_BR_BL"], float)
    glyphs, rectified, mask, matrix, display_matrix, _ = extract_glyphs(
        rgb, quad, glyph_ids=("SKATING_A",), component_indices=(6,))
    glyph = glyphs[0]
    contour_rows, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    external = [(c, h) for c, h in zip(contour_rows, hierarchy[0])
                if h[3] < 0 and cv2.contourArea(c) > 60.]
    external.sort(key=lambda item: cv2.boundingRect(item[0])[0])
    child = int(external[6][1][2])
    holes = []
    while child >= 0:
        source_curve = transform(contour_rows[child][:, 0], np.linalg.inv(display_matrix))
        source_curve = cv2.approxPolyDP(source_curve.astype(np.float32), .35, True)[:, 0].astype(float)
        if not np.array_equal(source_curve[0], source_curve[-1]):
            source_curve = np.vstack((source_curve, source_curve[0]))
        material_curve = transform(source_curve, matrix)
        key = "SKATING_A_inner" if not holes else f"SKATING_A_inner{len(holes)+1}"
        hole = deepcopy(glyph)
        hole.update(id=key, role="white_print_internal_hole_boundary",
            parent_contour_key="SKATING_A", source_native_uv=source_curve,
            material_uv=material_curve, source_native_simplification_tolerance_px=.35,
            internal_hole_boundary_separate_from_external_ink=True,
            original_native_glyph_bbox_xyxy=[*source_curve.min(0), *source_curve.max(0)])
        hole.pop("rectified_component_bbox_xywh", None)
        hole["threshold_sensitivity"] = None
        hole["inner_hole_threshold_sensitivity_not_independently_calibrated"] = True
        holes.append(hole)
        child = int(hierarchy[0][child][0])
    if len(holes) != 1:
        raise ValueError(f"Independent source A must expose exactly one separate upper hole, got {len(holes)}")
    glyph["separate_inner_contour_keys"] = [h["id"] for h in holes]
    for contour in [glyph, *holes]:
        key = contour["id"]
        value = plain(contour)
        if key in template["contours"] and template["contours"][key] != value:
            raise ValueError("Existing additional A contour differs; preserve historical version")
        template["contours"][key] = value
    if any(template["contours"][key] != value for key, value in original_contours.items()):
        raise ValueError("Appending a new material feature changed an existing contour")
    preview = cv2.cvtColor(rectified, cv2.COLOR_RGB2BGR)
    for contour in template["contours"].values():
        points = transform(contour["source_native_uv"], display_matrix)
        color = (0, 255, 255) if contour["id"].endswith("inner") else (20, 20, 255)
        cv2.polylines(preview, [np.rint(points).astype(np.int32)], True, color, 2, cv2.LINE_AA)
    diagnostic = diagnostics / "ad2_rectified_material_template_with_A.png"
    if not cv2.imwrite(str(diagnostic), preview):
        raise OSError(diagnostic)
    template["diagnostic_images"]["rectified_template_with_A"] = str(diagnostic.resolve())
    template["new_feature_append_SKATING_A"] = dict(
        source_pts=source["source_pts"], independent_source_pixels_only=True,
        external_contour_key="SKATING_A", internal_hole_contour_keys=[h["id"] for h in holes],
        existing_contours_preserved=True,
        preserved_existing_contours_sha256=hashlib.sha256(json.dumps(original_contours,
            sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
        current_programme_frame_points_used=False, source_native_precision_is_initial_only=True)
    output.write_text(json.dumps(plain(template), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    registry_path = output.with_name("feature_registry.json")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for contour in [glyph, *holes]:
        key = contour["id"]
        target = deepcopy(registry["targets"]["Ad2.print.K"])
        target.update(contour_key=key, contour_role=contour["role"])
        registry["targets"][f"Ad2.print.{key}"] = target
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    model_path = output.with_name("model.json")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model["material_templates"]["sha256"] = sha(output)
    model["feature_registry"]["sha256"] = sha(registry_path)
    model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(stage="appended_A", contours={c["id"]:
        dict(points=len(c["material_uv"]), material_uv_min=np.asarray(c["material_uv"]).min(0).tolist(),
             material_uv_max=np.asarray(c["material_uv"]).max(0).tolist()) for c in [glyph,*holes]},
        existing_contours_preserved=True, output=str(output.resolve())), ensure_ascii=False))
    return template


def build(reference_path=DEFAULT_REFERENCE, output=DEFAULT_OUTPUT, diagnostics=DEFAULT_DIAGNOSTICS):
    reference_path, output, diagnostics = map(Path, (reference_path, output, diagnostics))
    reference = json.loads(reference_path.read_text(encoding="utf-8-sig"))
    binding = reference["source"]
    source_path = Path(binding["source_frame_path"])
    if sha(source_path) != binding["source_frame_sha256"]:
        raise ValueError("Independent native source image file hash changed")
    bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if hashlib.sha256(rgb.tobytes()).hexdigest() != binding["decoded_native_rgb_sha256"]:
        raise ValueError("Independent native source RGB hash changed")
    ad = next(a for a in reference["advertisements"] if a["advertisement_number"] == 2)
    if not ad["physical_identity_human_confirmed"]:
        raise ValueError("Independent source physical Ad2 identity is unconfirmed")
    edges, quad, corner_intervals = measure_edges(rgb, binding)
    glyphs, rectified, mask, matrix, display_matrix, component_count = extract_glyphs(rgb, quad)
    diagnostics.mkdir(parents=True, exist_ok=True)
    profile_path = diagnostics / "ad2_independent_source_material_profiles.json"
    profile_path.write_text(json.dumps(plain(dict(source=binding, edges=edges)),
        ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    edge_summaries = {}
    for name, edge in edges.items():
        summary = {k: v for k, v in edge.items() if k != "profile_reading"}
        summary["source_transition_support"] = [dict(
            candidate_native_uv=r["actual_source_transition_candidate_uv"],
            observed_normal_interval_px=r["observed_normal_interval_px"],
            disposition=r["disposition"], source_flags=r["source_flags"])
            for r in edge["profile_reading"]["readings"]]
        edge_summaries[name] = summary
    canvas = bgr.copy()
    colors = [(0, 220, 255), (100, 220, 30), (30, 90, 255), (255, 150, 0)]
    for color, edge in zip(colors, edges.values()):
        for point in edge["actual_native_uv"]:
            cv2.circle(canvas, tuple(np.rint(point).astype(int)), 2, color, -1, cv2.LINE_AA)
    for glyph in glyphs:
        cv2.polylines(canvas, [np.rint(glyph["source_native_uv"]).astype(np.int32)], True,
                      (20, 20, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(diagnostics / "ad2_independent_source_edges_and_glyphs.png"),
                cv2.resize(canvas[335:440, 1515:1745], (1380, 630), interpolation=cv2.INTER_NEAREST))
    annotated = cv2.cvtColor(rectified, cv2.COLOR_RGB2BGR)
    for glyph in glyphs:
        points = transform(glyph["source_native_uv"], display_matrix)
        cv2.polylines(annotated, [np.rint(points).astype(np.int32)], True, (20, 20, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, glyph["id"], tuple(np.rint(points.min(axis=0)-[0, 8]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(diagnostics / "ad2_rectified_material_template.png"), annotated)
    cv2.imwrite(str(diagnostics / "ad2_rectified_material_source.png"), cv2.cvtColor(rectified, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(diagnostics / "ad2_rectified_initial_white_mask.png"), mask)
    payload = dict(schema="rinkspace.independent-source-ad2-material-template.v1",
        event_id="ankara_2026", advertisement_number=2,
        status="initial_independent_source_template_requires_further_verification",
        accuracy_accepted=False, independently_verified_physical_dimensions=False,
        source_reference_packet=dict(path=str(reference_path.resolve()), sha256=sha(reference_path)),
        source=binding, physical_identity_authority=ad["physical_identity_authority"],
        source_search_corners_native_uv=SEARCH_CORNERS_TL_TR_BR_BL,
        source_search_corners_are_measured_boundaries=False,
        remeasured_source_edges=edge_summaries,
        complete_source_profile_diagnostic=dict(path=str(profile_path.resolve()), sha256=sha(profile_path)),
        source_corners_from_measured_edges_TL_TR_BR_BL=quad,
        source_corner_empirical_support_TL_TR_BR_BL=corner_intervals,
        source_to_material_homography=matrix,
        material_domain=dict(u="photographed left=0, right=1", v="top=0, bottom=1",
            boundary="actual purple/white panel interface", units="dimensionless",
            lifting_api="PairedLocationsSharedSizesGeometry.material(2,u,1-v,current_world_q)",
            independently_measured_world_XYZ=False,
            four_source_edge_line_rectification_is_initializer_assumption=True,
            lens_distortion_and_possible_board_curvature_not_independently_rectified=True),
        contours={glyph["id"]:glyph for glyph in glyphs},
        extraction=dict(white_green_threshold_uint8=153, native_simplification_tolerance_px=.45,
            rectified_preview_width_height=[1536, 256], upsampling_adds_source_resolution=False,
            detected_external_ink_components=component_count,
            observed_original_print_text="ISU-SKATING.COM",
            component_order_first_six=["I", "S", "U", "-", "S", "K"]),
        limitations=["Reference letters are only approximately 15–18 native pixels high; a 2 px edge uncertainty is material.",
            "Four edge intersections and a homography initialize a 2-D material chart; they are not surveyed dimensions.",
            "Glyph threshold, native blur, lens distortion and possible course curvature require future independent verification.",
            "No current-frame accepted point, fitted camera or reverse-projected world coordinate enters this template."],
        diagnostic_images={key: str((diagnostics/name).resolve()) for key, name in (
            ("native_source", "ad2_independent_source_edges_and_glyphs.png"),
            ("rectified_source", "ad2_rectified_material_source.png"),
            ("rectified_template", "ad2_rectified_material_template.png"),
            ("initial_mask", "ad2_rectified_initial_white_mask.png"))})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plain(payload), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output.resolve()), corners=quad.tolist(),
        glyph_point_counts={g["id"]:len(g["material_uv"]) for g in glyphs},
        edge_measured_counts={k:len(v["actual_native_uv"]) for k,v in edges.items()}), ensure_ascii=False))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--append-skating-a", action="store_true",
        help="Append A/its separate hole to the existing template, preserving prior contours")
    args = parser.parse_args()
    if args.append_skating_a:
        append_skating_a(args.output, args.diagnostics)
    else:
        build(args.reference, args.output, args.diagnostics)
