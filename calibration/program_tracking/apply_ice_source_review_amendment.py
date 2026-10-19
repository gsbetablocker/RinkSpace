"""Apply the user's first frame-200 source review using native evidence.

Only the specified ice packet is changed. Shared current-source line estimates
are distinguished from individual native profile readouts and masked inference.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from calibration.program_tracking.extract_single_frame_ice_source import (
    ROOT, read_json, sha, ellipse_distance,
)
from rinkspace.vision_programme_ice_review import _batched_actual_profiles, _actual_profile_candidates
from rinkspace.vision_source_shapes import _MaterialProfileFrame

DATASET = ROOT / "events/ankara_2026/programmes/okada_mei_sp/source_datasets/frame_by_frame_review_20261013/frame_000200_50fps"
MEDIA = Path("D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/frame_by_frame_review_20261013/frame_000200_50fps")


class NativeReadouts:
    def __init__(self, source_binding, rgb, exclusion):
        self.frame = _MaterialProfileFrame(rgb, dict(source_sha256=source_binding["source_video_sha256"],
            source_pts=source_binding['native_pts'], time_base=source_binding['time_base'], native_hw=[1080, 1920], shot_id=1))
        self.exclusion = exclusion
        self.archive = {}

    def read(self, name, query_uv, normal, operator, transition=None, radius=48.):
        queries = np.asarray(query_uv, float)
        normals = np.asarray(normal, float)
        intervals = np.tile([-radius, radius], (len(queries), 1))
        for axis, maximum in enumerate((1919, 1079)):
            move = np.abs(normals[:, axis]) > 1e-12
            lo = np.divide(4 - queries[:, axis], normals[:, axis], out=np.full(len(queries), -np.inf), where=move)
            hi = np.divide(maximum - 4 - queries[:, axis], normals[:, axis], out=np.full(len(queries), np.inf), where=move)
            intervals[:, 0] = np.maximum(intervals[:, 0], np.minimum(lo, hi))
            intervals[:, 1] = np.minimum(intervals[:, 1], np.maximum(lo, hi))
        valid_query = intervals[:, 1] - intervals[:, 0] >= 20.
        queries, normals, intervals = queries[valid_query], normals[valid_query], intervals[valid_query]
        if not len(queries):
            return []
        profiles = _batched_actual_profiles(self.frame, queries, normals, intervals, np.full(len(queries), "dark"))
        for key, value in dict(query_uv=queries, query_normal=normals, query_interval=intervals,
            profile_offsets=profiles[0], profile_values=profiles[1], profile_smoothed=profiles[2], profile_valid=profiles[3]).items():
            self.archive[name + "__" + key] = value
        rows = []
        for qi in range(len(queries)):
            offsets, profile, smooth, valid = [array[qi] for array in profiles]
            values = _actual_profile_candidates(queries[qi], normals[qi], offsets[valid], profile[valid], smooth[valid],
                operator=operator, signal="dark", transition=transition, with_quantiles=True)
            local = []
            for uv, sides, bad, quantiles in values:
                offsets_q = (quantiles - queries[qi]) @ normals[qi]
                sigma = float(np.max(np.abs(offsets_q[:, 2] - offsets_q[:, 0])) / 1.349)
                contrast = [abs(float(np.interp(o[2], offsets[valid], smooth[valid])
                    - np.interp(o[0], offsets[valid], smooth[valid]))) for o in offsets_q]
                shift = float((uv - queries[qi]) @ normals[qi])
                # Only actual material transition support is tested. A distant
                # foreground response in the broad search cannot emit an UV.
                support_uv = np.vstack([uv[None], sides, quantiles.reshape(-1, 2)])
                ix = np.rint(support_uv).astype(int)
                in_frame = (ix >= [0, 0]).all() and (ix <= [1919, 1079]).all()
                excluded = not in_frame or bool(self.exclusion[ix[:, 1], ix[:, 0]].any())
                if bad or excluded or min(contrast) < 3. or sigma > 20. or abs(shift) > 32.:
                    continue
                if operator == "paint_center" and np.linalg.norm(sides[1] - sides[0]) < 1.5:
                    continue
                evidence = dict(query_set=name, query_index=qi, query_uv=queries[qi].tolist(),
                    profile_normal=normals[qi].tolist(), native_profile_shift_px=shift,
                    signed_interface_iqr_contrast=contrast, transition_sigma_px=max(1., sigma),
                    actual_source_sides_uv=sides.tolist(), actual_source_side_quantiles_uv=quantiles.tolist(),
                    actual_source_support_outside_person_and_broadcast_masks=True,
                    measurement_origin="current_native_signed_RGB_material_profiles")
                local.append((abs(shift), dict(uv=np.round(uv, 4).tolist(), sigma_px=max(1., sigma), source_evidence=evidence)))
            local.sort(key=lambda x:x[0])
            if local:
                rows.append(local[0][1])
        return rows


def ellipse_summary(feature):
    rows = [(uv, sigma) for s in feature["segments"] for uv, sigma in zip(s["uv"], s["sigma_px"])]
    points = np.asarray([r[0] for r in rows], np.float32)
    sigma = np.asarray([r[1] for r in rows])
    ellipse = cv2.fitEllipse(points)
    distance = ellipse_distance(points, ellipse)
    feature["geometric_interpretation"] = dict(kind="projected_physical_circle_ellipse",
        inference_only=True, inferred_uv_not_emitted_as_source_points=True,
        basis="one physical circle; all native observed segments jointly interpreted by one ellipse",
        center_uv=list(ellipse[0]), axes_px=list(ellipse[1]), angle_degrees=ellipse[2],
        maximum_source_readout_to_ellipse_distance_px=float(distance.max()),
        maximum_residual_divided_by_readout_sigma=float((distance / sigma).max()),
        all_readouts_within_maximum_of_2point5_px_or_2point5_sigma=bool((distance <= np.maximum(2.5, 2.5 * sigma)).all()),
        hidden_spans_are_inference_not_source_observations=True)
    return ellipse


def fit_source_line(rows):
    uv = np.asarray([r["uv"] for r in rows])
    sigma = np.asarray([r["sigma_px"] for r in rows])
    design = np.c_[uv[:, 0], np.ones(len(uv))]
    active = np.ones(len(uv), bool)
    for _ in range(4):
        normal = design[active].T @ ((1 / sigma[active] ** 2)[:, None] * design[active])
        covariance = np.linalg.inv(normal)
        coefficient = covariance @ (design[active].T @ (uv[active, 1] / sigma[active] ** 2))
        residual = uv[:, 1] - design @ coefficient
        next_active = np.abs(residual) <= np.maximum(3., 2.5 * sigma)
        if np.array_equal(active, next_active):
            break
        active = next_active
    reduced_chi2 = float(np.sum((residual[active] / sigma[active]) ** 2) / max(1, active.sum() - 2))
    covariance *= max(1., reduced_chi2)
    return coefficient, covariance, active, residual


def runs(indices):
    if not len(indices):
        return []
    parts = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return [(int(p[0]), int(p[-1])) for p in parts if len(p)]


def amend_line(feature, reader, exclusion, person, broadcast, inferred, groups):
    existing_rows = [dict(uv=uv, sigma_px=sigma, source_evidence=deepcopy(evidence),
        readout_origin="preserved_native_profile_readout_before_user_correction_01")
        for s in feature["segments"] for uv, sigma, evidence in zip(s["uv"], s["sigma_px"], s["source_evidence"])]
    seed, _, _, _ = fit_source_line(existing_rows)
    xs = np.arange(15., 1910., 80.)
    query = np.c_[xs, np.polyval(seed, xs)]
    in_frame = (query[:, 1] >= 20.) & (query[:, 1] <= 1059.)
    query = query[in_frame]
    normal = np.tile(np.array([-seed[0], 1.]) / np.sqrt(1. + seed[0] ** 2), (len(query), 1))
    native_rows = reader.read(feature["id"].replace(":", "_") + "_source_line", query, normal, "paint_center", radius=32.)
    native_rows = [r for r in native_rows if abs(r["uv"][1] - np.polyval(seed, r["uv"][0])) <= max(4., 2.5 * r["sigma_px"])]
    rows = deepcopy(existing_rows)
    for row in native_rows:
        if not any(np.linalg.norm(np.asarray(row["uv"]) - r["uv"]) < 8. for r in rows):
            rows.append(row)
    coefficient, covariance, active, residual = fit_source_line(rows)
    group_id = "official50_307200_" + feature["id"].replace(".", "_").replace(":", "_") + "_source_line"
    original_line_count = feature["point_count"]
    feature["individual_native_readout_segments_before_shared_line_rebuild"] = deepcopy(feature["segments"])
    feature["source_evidence_group_id"] = group_id
    feature["source_line_fit"] = dict(coefficient_y_equals_ax_plus_b=coefficient.tolist(),
        covariance_ab=covariance.tolist(), native_source_pts=307200,
        current_frame_only=True, old_model_projection_used_as_output_points=False,
        direct_readouts=[dict(r, used_in_line_fit=bool(ok), source_fit_residual_px=float(res))
            for r, ok, res in zip(rows, active, residual)],
        actual_native_readout_count=int(active.sum()),
        maximum_active_residual_divided_by_readout_sigma=float(max(abs(residual[active]) /
            np.array([r["sigma_px"] for r, ok in zip(rows, active) if ok]))))
    xs = np.arange(1920)
    ys = np.polyval(coefficient, xs)
    inside = (ys >= 0) & (ys <= 1079)
    iy = np.clip(np.rint(ys).astype(int), 0, 1079)
    hidden = person[iy, xs] | broadcast[iy, xs]
    visible = inside & ~hidden
    feature["segments"] = []
    for first, last in runs(np.flatnonzero(visible)):
        positions = np.linspace(first, last, max(1, int(np.ceil((last - first) / 220.))) + 1)
        uv = np.c_[positions, np.polyval(coefficient, positions)]
        sigmas = [max(1., float(np.sqrt(np.array([x, 1.]) @ covariance @ np.array([x, 1.])))) for x in positions]
        feature["segments"].append(dict(uv=np.round(uv, 4).tolist(), closed=False, vertex_count=len(uv),
            sigma_px=sigmas, measurement_origin="same_frame_joint_straight_edge_fit",
            representative_points_share_line_estimate=True, source_evidence_group_id=group_id,
            pixels_measured_in_this_frame=False, source_evidence_measured_in_this_frame=True,
            individual_native_pixel_readouts=False, inferred_from_current_source_readouts=True,
            hidden_span_filled=False, user_confirmed=False,
            source_evidence=[dict(kind="shared_current_source_line_representative", source_evidence_group_id=group_id,
                individual_native_pixel_readout=False, coefficient_y_equals_ax_plus_b=coefficient.tolist()) for _ in uv]))
    for first, last in runs(np.flatnonzero(inside & hidden)):
        reason = "人物遮挡的同一物理直线延续，仅几何推断，不是原图观测。"
        if broadcast[iy[first:last + 1], xs[first:last + 1]].any():
            reason = "原图0:01计时字幕框遮挡的同一物理直线延续，仅几何推断，不是原图观测。"
        inferred.append(dict(target_id=feature["target_id"],
            uv=[[float(first), float(np.polyval(coefficient, first))], [float(last), float(np.polyval(coefficient, last))]],
            inference_only=True, reason_zh=reason, source_evidence_group_id=group_id))
    feature["point_count"] = sum(s["vertex_count"] for s in feature["segments"])
    feature["note_zh"] = "用户确认该条冰面标线连续。以本帧原图材料读数共同拟合一条直线，在人物和字幕遮挡外按约220像素均匀设置同源代表点；局部剖面无响应不作为物理断线。代表点共享同一估计，单独原图读数与遮挡推断已分开。"
    feature["geometric_interpretation"] = dict(kind="straight_ice_paint_centerline", inference_only=True,
        current_source_line_y_equals_ax_plus_b=coefficient.tolist(), current_source_line_covariance_ab=covariance.tolist(),
        full_visible_native_frame_extent=True, source_line_representatives_are_not_independent_pixel_readouts=True)
    groups.append(dict(group_id=group_id, target_id=feature["target_id"], native_source_pts=307200,
        correlation_reason="all representative vertices share the same current-source straight-line estimate",
        fit_parameter_order=["slope_a", "intercept_b"], covariance_ab=covariance.tolist(),
        representative_point_count=feature["point_count"], native_individual_readout_count=int(active.sum())))
    return feature["point_count"] - original_line_count


def main():
    packet_path = DATASET / "ice_features.json"
    packet = read_json(packet_path)
    if (packet.get("user_correction_01", {}).get("new_ice_point_count") == packet["point_count"]
            and packet['source'].get('broadcast_timer_exclusion',{}).get('edge_halo_px')==4):
        print(json.dumps(dict(status="already_applied", point_count=packet["point_count"], sha256=sha(packet_path))))
        return
    if packet.get('user_correction_01'):
        snapshot=MEDIA/'user_correction_01_before_timer_halo_ice97.json'
        if not snapshot.exists():snapshot.write_bytes(packet_path.read_bytes())
        packet=read_json(MEDIA/'user_correction_01_before_ice_features_78points.json')
    before = deepcopy(packet)
    for file, output in [(packet_path, "user_correction_01_before_ice_features_78points.json"),
        (DATASET / "frame_000200_points.json", "user_correction_01_before_frame_000200_points_233points.json"),
        (DATASET / "ice_source_point_exclusions.json", "user_correction_01_before_ice_source_point_exclusions.json")]:
        if file.exists() and not (MEDIA / output).exists():
            shutil.copyfile(file, MEDIA / output)
    authority_path = DATASET / "user_correction_01.json"
    authority = dict(path=str(authority_path), sha256=sha(authority_path),
        annotation_is_identity_and_location_evidence_not_native_uv_measurements=True)
    source_binding = read_json(MEDIA / "source_binding.json")
    bgr = cv2.imread(str(MEDIA / "source.png")); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    assert hashlib.sha256(rgb.tobytes()).hexdigest() == source_binding["decoded_RGB_sha256"]
    person = cv2.imread(str(MEDIA / "person_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    broadcast = np.zeros_like(person)
    white = (bgr[920:1020, 1650:1900].min(axis=2) > 216).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(white)
    component = max(stats[1:], key=lambda s:s[4])
    bx, by, bw, bh = map(int, component[:4] + [1650, 920, 0, 0])
    broadcast[by:by + bh, bx:bx + bw] = True
    broadcast=cv2.dilate(broadcast.astype('uint8'),np.ones((9,9),np.uint8))>0
    reader = NativeReadouts(source_binding, rgb, person | broadcast)
    r4 = next(f for f in packet["features"] if f["target_id"] == "ice.R4:outer")
    red = np.array([[808.2313, 592.5643], [820.8047, 593.1386], [849.9234, 601.294]])
    remaining, red_removed = [], []
    for segment in r4["segments"]:
        ids = []
        for i, uv in enumerate(segment["uv"]):
            if (np.linalg.norm(red - uv, axis=1) < .001).any():
                red_removed.append(uv)
            else:
                ids.append(i)
        if ids:
            revised = deepcopy(segment)
            for key in ("uv", "sigma_px", "source_evidence"):
                revised[key] = [segment[key][i] for i in ids]
            revised["vertex_count"] = len(ids); revised["closed"] = False
            remaining.append(revised)
    assert len(red_removed) == 3
    r4["segments"] = remaining; r4["point_count"] -= 3
    r4["note_zh"] += " 用户红圈内三个偏点已删除；其余原图读数保持。"
    ellipse_summary(r4)
    old82 = read_json(MEDIA / "ice_features_before_reflection_review_82points.json")
    full_r3 = next(f for f in old82["features"] if f["target_id"] == "ice.R3:outer")
    r3 = next(f for f in packet["features"] if f["target_id"] == "ice.R3:outer")
    restored = [s for s in full_r3["segments"] if all(uv[0] < 1020 for uv in s["uv"])]
    old_left_rows = [dict(uv=uv, sigma_px=sigma, source_evidence=deepcopy(evidence),
        physical_identity_confirmed_by_user=True) for s in restored
        for uv, sigma, evidence in zip(s["uv"], s["sigma_px"], s["source_evidence"])]
    all_old = np.array([uv for s in full_r3["segments"] for uv in s["uv"]], np.float32)
    ellipse = cv2.fitEllipse(all_old)
    center, axes, angle = ellipse; rx, ry = max(axes) / 2., min(axes) / 2.
    theta_rotation = np.deg2rad(angle - 90.)
    rotation = np.array([[np.cos(theta_rotation), -np.sin(theta_rotation)], [np.sin(theta_rotation), np.cos(theta_rotation)]])
    query, normal = [], []
    for theta in [-.6, -.5, -.4, -.28, -.18, 0., .18, .28, .4, .5, .6]:
        local = np.array([-np.cos(theta) * rx, np.sin(theta) * ry])
        query.append(local @ rotation.T + center)
        n = np.array([local[0] / rx ** 2, local[1] / ry ** 2]) @ rotation.T
        normal.append(n / np.linalg.norm(n))
    fresh_left = reader.read("B3_outer_left_user_confirmed", query, normal, "material_interface", "dark_to_light")
    left_rows = deepcopy(old_left_rows)
    for row in fresh_left:
        if ellipse_distance(np.array(row["uv"])[None], ellipse)[0] > max(4., 2.5 * row["sigma_px"]):
            continue
        if not any(np.linalg.norm(np.array(row["uv"]) - existing["uv"]) < 10. for existing in left_rows):
            left_rows.append(row)
    def angle_for(row):
        local = (np.array(row["uv"]) - center) @ rotation
        return float(np.arctan2(local[1] / ry, -local[0] / rx))
    left_rows.sort(key=angle_for)
    r3["segments"].append(dict(uv=[r["uv"] for r in left_rows], sigma_px=[r["sigma_px"] for r in left_rows],
        source_evidence=[r["source_evidence"] for r in left_rows], closed=False, vertex_count=len(left_rows),
        measurement_origin="current_native_signed_RGB_material_profiles", pixels_measured_in_this_frame=True,
        source_evidence_measured_in_this_frame=True, hidden_span_filled=False, user_confirmed=False,
        physical_identity_confirmation=authority,
        note_zh="裙左段由用户明确确认为B3最外圈；恢复原生读数并重读整段外圈材料界面，不把截图蓝色涂画当测量坐标。"))
    r3["point_count"] = sum(s["vertex_count"] for s in r3["segments"])
    r3["note_zh"] = "用户明确确认裙左弧是B3最外圈，撤销前轮倒影归属拒绝。整段原图材料界面已重读，所有可见段按同一个物理圆的投影椭圆共同解释；人物遮挡处不补直接读数。"
    r3.pop("source_visible_material_exclusions", None)
    r3["physical_identity_confirmation"] = authority
    r3["current_source_curve_branch_guide"].pop("initial_search_hypotheses_include_now_withheld_candidates", None)
    ellipse_summary(r3)
    packet["inferred_segments"] = []; packet["source_correlated_groups"] = []
    line_changes = {}
    for target in ("ice.lane3:edge2", "ice.lane2:edge0"):
        feature = next(f for f in packet["features"] if f["target_id"] == target)
        line_changes[target] = amend_line(feature, reader, person | broadcast, person, broadcast,
            packet["inferred_segments"], packet["source_correlated_groups"])
    packet["point_count"] = sum(f["point_count"] for f in packet["features"])
    packet["unresolved_source_regions"] = [r for r in packet["unresolved_source_regions"] if r["target_id"] != "ice.R3:outer"]
    packet["source_visible_point_exclusions"] = [dict(target_id="ice.R4:outer", source_uv=red_removed,
        note_zh="用户红圈指定的三个R4外圈偏点已撤下，其余保持。")]
    packet["visual_review"]["independent_stationary_material_review_exclusions"] = []
    packet["visual_review"]["previous_B3_left_reflection_rejection_superseded_by_user_confirmation"] = authority
    packet["source"]["broadcast_timer_exclusion"] = dict(actual_native_rectangle_xyxy_inclusive=[bx, by, bx + bw - 1, by + bh - 1],
        edge_halo_px=4,
        method="current-source white timer component enclosing rectangle, holes include text foreground", applied_to_source_readouts_and_line_visibility=True)
    for feature in packet["features"]:
        report = next(r for r in packet["curve_readout_summary"] if r["target_id"] == feature["target_id"])
        report["final_points"] = feature["point_count"]
        if feature["target_id"] in {"ice.R4:outer", "ice.R3:outer", "ice.lane3:edge2", "ice.lane2:edge0"}:
            report["note_zh"] = feature["note_zh"]
            if feature["target_id"] == "ice.R3:outer":
                report["rejections"] = [r for r in report["rejections"] if r.get("reason") != "stationary_material_identity_unconfirmed_possible_body_reflection"]
                report["previous_reflection_identity_rejection_superseded"] = True
    evidence_path = MEDIA / "user_correction_01_native_readout.npz"
    np.savez_compressed(evidence_path, **reader.archive)
    packet.setdefault("source_evidence_additional_archives", []).append(dict(path=str(evidence_path), sha256=sha(evidence_path),
        source_pts=307200, original_native_RGB_sha256=source_binding["decoded_RGB_sha256"]))
    packet["user_correction_01"] = dict(authority=authority, old_ice_point_count=before["point_count"],
        red_R4_removed=red_removed, B3_left_restored_source_readouts=4, B3_left_final_native_readouts=len(left_rows),
        line_point_count_changes=line_changes, new_ice_point_count=packet["point_count"],
        inferred_points_excluded_from_point_count=True, other_features_unchanged=True)
    for old, new in zip(before["features"], packet["features"]):
        if old["target_id"] not in {"ice.R4:outer", "ice.R3:outer", "ice.lane3:edge2", "ice.lane2:edge0"}:
            assert old == new
    for feature in packet["features"]:
        for segment in feature["segments"]:
            for point in segment["uv"]:
                x, y = np.rint(point).astype(int)
                assert 0 <= x <= 1919 and 0 <= y <= 1079 and not person[y, x] and not broadcast[y, x]
    payload = json.dumps(packet, ensure_ascii=False, indent=2) + "\n"
    assert "\ufffd" not in payload
    packet_path.write_text(payload, encoding="utf-8")
    overlay = bgr.copy()
    for feature in packet["features"]:
        color = (222, 85, 230) if "R4:" in feature["id"] else (25, 210, 250) if "R3:" in feature["id"] else (70, 235, 70)
        for segment in feature["segments"]:
            points = np.rint(np.array(segment["uv"]) * 256).astype(np.int32)
            cv2.polylines(overlay, [points], segment["closed"], color, 1, cv2.LINE_AA, shift=8)
            for point in points:
                cv2.circle(overlay, tuple(map(int, point)), 3 * 256, (15, 15, 15), -1, cv2.LINE_AA, shift=8)
                cv2.circle(overlay, tuple(map(int, point)), 2 * 256, color, -1, cv2.LINE_AA, shift=8)
    cv2.imwrite(str(MEDIA / "ice_points_review.png"), overlay)
    print(json.dumps(dict(point_count=packet["point_count"], sha256=sha(packet_path),
        B3_left_native_readouts=len(left_rows), line_point_count_changes=line_changes,
        timer_native_rect=[bx, by, bx + bw - 1, by + bh - 1],
        groups=[g["group_id"] for g in packet["source_correlated_groups"]],
        inferred_segment_count=len(packet["inferred_segments"])), ensure_ascii=False))


if __name__ == "__main__":
    main()
