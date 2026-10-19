"""Bind the user's orange/blue source strokes to the existing Lane factors."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import shutil

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(r"D:/CodexData/RinkSpace_Data/events/ankara_2026")
ATTACHMENT = Path(r"C:/Users/REDACTED/AppData/Local/Temp/codex-clipboard-private-attachment-00026.png")
SOURCE = DATA / "calibration/runs/named66_823_9e_two_source_issues_20261005/raw_native_230501120.png"
BASE = DATA / "observations/vision_training/continuous_review/cut6_clear_source_constraints"
PARENT = BASE / "complete_source_clear_anchor_append_20261007/ice_parent_plus_actual_clear_anchors.json"
EVIDENCE = DATA / "evidence/user_annotations_20261007"
FACT = ROOT / "events/ankara_2026/model_constraints/lane2_user_confirmed_20261007.json"
OUTPUT = PARENT.with_name("ice_clear_anchors_user_lane2_lane3_20261007.json")
STATEMENT = "橙色线是lane2上长边，蓝色线是lane3下长边"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    cv2.setNumThreads(1)
    if OUTPUT.exists():
        raise FileExistsError("Preserve already created source evidence")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    saved_attachment = EVIDENCE / ATTACHMENT.name
    if saved_attachment.exists() and sha(saved_attachment) != sha(ATTACHMENT):
        raise ValueError("Existing attachment identity differs")
    if not saved_attachment.exists():
        shutil.copyfile(ATTACHMENT, saved_attachment)
    rgb = np.asarray(Image.open(SOURCE).convert("RGB"))
    marked = np.asarray(Image.open(saved_attachment).convert("RGB"))
    if rgb.shape != (1080, 1920, 3) or marked.shape != (1128, 2000, 3):
        raise ValueError("Unexpected native or attachment dimensions")
    # Actual unchanged image texture verifies 25/24 scale. Bottom three
    # attachment rows are outside the resized native image and never source UV.
    sift = cv2.SIFT_create(nfeatures=1500)
    masks = []
    for image, limit in [(rgb, 810), (marked, 840)]:
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[:limit] = 255
        masks.append(mask)
    kp, desc = sift.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), masks[0])
    kq, deq = sift.detectAndCompute(cv2.cvtColor(marked, cv2.COLOR_RGB2GRAY), masks[1])
    pairs = [a for a, b in cv2.BFMatcher().knnMatch(desc, deq, k=2) if a.distance < .75 * b.distance]
    p = np.float32([kp[a.queryIdx].pt for a in pairs])
    q = np.float32([kq[a.trainIdx].pt for a in pairs])
    affine, inliers = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC,
        ransacReprojThreshold=.7, maxIters=10000)
    residual = np.linalg.norm(p @ affine[:, :2].T + affine[:, 2] - q, axis=1)[inliers[:, 0].astype(bool)]
    if len(residual) < 100 or np.max(np.abs(affine[:, :2] - np.eye(2) * 25 / 24)) > .0002:
        raise ValueError("Actual source registration not established")
    registration = dict(native_wh=[1920, 1080], attachment_wh=[2000, 1128],
        image_content_wh=[2000, 1125], inverse_uniform_scale=24 / 25,
        measured_source_to_attachment_affine=affine.tolist(), matched_pairs=len(pairs),
        actual_texture_inliers=len(residual), median_error_attachment_px=float(np.median(residual)),
        p95_error_attachment_px=float(np.quantile(residual, .95)),
        source_pixel_geometry_is_not_stretched_to_attachment_height=True)
    colors = dict(lane2=(233, 113, 50), lane3=(21, 96, 130))
    edges = dict(lane2=("upper_long", 0), lane3=("lower_long", 2))
    rows, lines = [], []
    for feature, color in colors.items():
        distance = np.max(np.abs(marked.astype(np.int16) - np.asarray(color, np.int16)), axis=2)
        stroke = (distance <= 6)
        stroke[:820] = False
        stroke[1125:] = False
        y, x = np.nonzero(stroke)
        if len(x) < 1000:
            raise ValueError("Expected continuous colored user stroke")
        columns = np.unique(x)
        center_y = np.array([np.median(y[x == c]) for c in columns])
        slope, intercept = np.polyfit(columns, center_y, 1)
        normal = np.array([-slope, 1.]) / np.sqrt(1 + slope * slope)
        uv, supports = [], []
        # Samples represent drawn source axes, not physical paint endpoints.
        for native_x in [450, 600, 1100, 1300, 1450, 1550]:
            c = int(round(native_x * 25 / 24))
            positions = y[x == c]
            if len(positions) == 0:
                raise ValueError("Chosen user source stroke has a gap")
            center = float(np.median(positions))
            actual_x = c * 24 / 25
            uv.append([actual_x, center * 24 / 25])
            supports.append([(float(positions.min()) - .5 - center) * 24 / 25 / np.sqrt(1 + slope*slope),
                             (float(positions.max()) + .5 - center) * 24 / 25 / np.sqrt(1 + slope*slope)])
        edge, segment = edges[feature]
        line = dict(feature_id=feature, edge=edge, segment_index=segment,
            user_display_RGB=list(color), user_native_axis_samples=uv,
            annotation_stroke_normal_interval_px=supports,
            annotation_stroke_extent_native_xyxy=[float(x.min())*.96, float(y.min())*.96,
                                                float(x.max())*.96, float(y.max())*.96],
            painted_endpoint_observation=False, source_occluded_sections_not_inferred=True)
        lines.append(line)
        rows.append(dict(id=f"user_native230501120_{feature}_{edge}_20261007",
            pts=230501120, actual_pts=230501120, feature_id=feature,
            curve=dict(kind="current_ice_segment", feature_id=feature, segment_index=segment),
            geometry_role="nominal_painted_stroke_center_axis", split="fit", uv=uv,
            sigma_px=[4.] * len(uv),
            sigma_semantics="4 native px declared engineering weight for direct user source axis; not calibrated measurement variance",
            normal=[normal.tolist()] * len(uv),
            annotation_stroke_normal_interval_px=supports,
            source_optical_normal_support_authorized=False,
            annotation_stroke_is_not_physical_paint_or_blur_width=True,
            source_raw_frame=str(SOURCE), source_file_sha256=sha(SOURCE),
            source_native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            source_user_annotation=dict(path=str(saved_attachment), sha256=sha(saved_attachment)),
            sourceUV_origin="direct_user_source_drawing_inverse_verified_uniform_display_scale",
            model_projection_used_to_generate_or_select_uv=False,
            source_RGB_actual_measurement=False, user_confirmed_source_axis=True,
            physical_identity_authority=dict(kind="direct_user_message_and_colored_source_stroke",
                user_statement=STATEMENT, record=str(FACT)), endpoint_observation=False))
    trace_path = EVIDENCE / "native230501120_lane2_lane3_user_source_strokes.json"
    write(trace_path, dict(schema="rinkspace.registered-user-source-strokes.v1",
        source_pts=230501120, source_time_base="1/15360", source_time="04:10:06.583333",
        source_video_sha256="f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891",
        original_attachment=dict(path=str(saved_attachment), sha256=sha(saved_attachment)),
        native_source=dict(path=str(SOURCE), sha256=sha(SOURCE)), registration=registration,
        user_statement=STATEMENT, lines=lines,
        raw_source_pixels_and_historical_packets_changed=False,
        segment_name_evidence="src/rinkspace/calibration_companion/structural_constraints.py lane_edge_order upper_long/right_short/lower_long/left_open; current producer preserves 0/2 long sides"))
    prior_fact = json.loads(FACT.read_text(encoding="utf-8"))
    fact = dict(schema="rinkspace.user-confirmed-visible-feature-identity.v2",
        event_id="ankara_2026", programme_id="okada_mei_sp", confirmed_on="2026-10-07",
        authority="direct_user_message_in_current_chat", user_statement=STATEMENT,
        status="active_explicit_colored_two_line_confirmation", active_identity_confirmation=True,
        source_native_pts=230501120, source_time_base="1/15360", source_time="04:10:06.583333",
        source_video_sha256="f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891",
        reference_image=dict(path=str(saved_attachment), sha256=sha(saved_attachment)),
        registered_source_strokes=dict(path=str(trace_path), sha256=sha(trace_path)),
        registration=registration, confirmed_visible_lines=lines,
        fitting_policy="Keep every original source observation active. Append explicit user axes; old 501120 three diagnostic points are not separately appended to FIT.",
        physical_paint_endpoints_or_world_dimensions_confirmed=False,
        source_annotation_stroke_width_is_not_physical_paint_width=True,
        historical_records=[prior_fact])
    write(FACT, fact)
    packet = json.loads(PARENT.read_text(encoding="utf-8"))
    oldrows = deepcopy(packet["rows"])
    packet["rows"].extend(rows)
    if packet["rows"][:len(oldrows)] != oldrows:
        raise ValueError("Original source changed")
    packet["status"] = "ORIGINAL_ACTIVE_SOURCE_PLUS_REGISTERED_DIRECT_USER_LANE_AXES"
    packet["direct_user_lane_axis_append"] = dict(parent_path=str(PARENT), parent_sha256=sha(PARENT),
        source_fact=str(FACT), source_fact_sha256=sha(FACT), added_ids=[r["id"] for r in rows],
        FIT_scalar_rows_added=sum(2*len(r["uv"]) for r in rows),
        all_original_UV_sigma_roles_split_order_exact=True, original_lane_rows_not_retired=True,
        existing501120_RGB_three_point_validation_not_reclassified_as_fit=True,
        human_stroke_support_is_annotation_extent_not_optical_interval=True)
    packet["observation_count"] = len(packet["rows"])
    packet["fit_observation_count"] = sum(r["split"] == "fit" for r in packet["rows"])
    packet["holdout_observation_count"] = sum(r["split"] == "holdout" for r in packet["rows"])
    packet["current_fit_scalar_count"] = sum(2*len(r["uv"]) for r in packet["rows"] if r["split"] == "fit")
    packet["current_primary_held_scalar_count"] = sum(2*len(r["uv"]) for r in packet["rows"] if r["split"] == "holdout")
    write(OUTPUT, packet)
    print(json.dumps(dict(packet=str(OUTPUT), sha256=sha(OUTPUT), FACT=str(FACT), evidence=str(trace_path),
        added_FIT_scalars=24, registration=registration, user_source_lines=lines), ensure_ascii=False))


if __name__ == "__main__":
    main()
