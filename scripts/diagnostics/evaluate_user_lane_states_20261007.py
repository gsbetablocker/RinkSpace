"""Compare immutable cameras/materials against unchanged user/native axes."""
from pathlib import Path
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["RINKSPACE_PROJECTION_BACKEND"] = "numpy"

import argparse
import hashlib
import json
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from calibration.solver_runtime import install
# This read-only diagnostic consumes authenticated frozen material exports.
# A concurrent edit to an unrelated geometry producer must not obstruct their
# comparison. RuntimeSourceFinder still SHA-checks every actually imported
# legacy module; no edited geometry factory is imported or evaluated here.
install(verify=False)
from rinkspace.vision_camera_initialization import frozen_scene_curve, unit_source_normals
from z190_source_camera_diagnostic import absolute_value_jac15

DATA = Path(r"D:/CodexData/RinkSpace_Data/events/ankara_2026")
RUNS = DATA / "calibration/runs/solver_acceleration_20261003"
SOURCE_DIR = DATA / "observations/vision_training/continuous_review/cut6_clear_source_constraints"
PACKET = SOURCE_DIR / "complete_source_clear_anchor_append_20261007/ice_clear_anchors_user_lane2_lane3_20261007.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accepted(directory, label=None):
    directory = Path(directory)
    identity = read(directory / "state_identity.json")
    scene = directory / "scene_UNACCEPTED.json"
    if sha(scene) != identity["scene_sha256"] or sha(identity["camera_path"]) != identity["camera_sha256"]:
        raise ValueError("Immutable camera or material identity changed")
    with np.load(directory / "complete_state_UNACCEPTED.npz", allow_pickle=False) as saved:
        if hashlib.sha256(saved["q"].tobytes()).hexdigest() != identity["q_sha256"]:
            raise ValueError("Complete q does not match its frozen camera/material")
    return dict(label=label or identity["q_sha256"][:12], q_sha256=identity["q_sha256"],
        scene_path=str(scene), scene_sha256=sha(scene), camera_path=identity["camera_path"],
        camera_sha256=identity["camera_sha256"], state_directory=str(directory))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-dir", action="append")
    parser.add_argument("--run-dir", default=str(RUNS / "okada_lane23_Ad7_fast_complete_20261007"))
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frozen = read(DATA / "programmes/okada_mei_sp/results/shared_radius_05db_pelvis_priority_full_programme_20261005/FROZEN_INPUTS_UNACCEPTED.json")
    if sha(frozen["scene"]["path"]) != frozen["scene"]["sha256"] or sha(frozen["camera"]["path"]) != frozen["camera"]["sha256"]:
        raise ValueError("Old complete-video inputs changed")
    states = [dict(label="old_video4075_camera917c204", q_sha256=frozen["q_sha256"],
        scene_path=frozen["scene"]["path"], scene_sha256=frozen["scene"]["sha256"],
        camera_path=frozen["camera"]["path"], camera_sha256=frozen["camera"]["sha256"]),
        accepted(RUNS / "source_corrected_clear_actual_complete_joint_20261007/accepted_states/0eb5c6d4d2b0a24ab201f65361ea0ed2de409b4bdfdddb1fc68c6b331ae54e63", "clear0eb5_camera46a4f243")]
    directories = ([Path(p) for p in args.state_dir] if args.state_dir else
        sorted((Path(args.run_dir) / "accepted_states").iterdir(), key=lambda p: p.stat().st_mtime))
    states.extend(accepted(p) for p in directories if (p / "state_identity.json").exists())
    packet = read(PACKET)
    user_rows = [r for r in packet["rows"] if r.get("user_confirmed_source_axis") is True]
    if len(user_rows) != 2:
        raise ValueError("Expected exactly two registered direct-user axes")
    old_rows = [r for r in packet["rows"] if r["id"] in
        {"cut6_source_230506752_lane2_upper_long_holdout", "cut6_source_230507776_lane2_upper_long_fit"}]
    ad_packet_path = SOURCE_DIR / "late_230501120_qualified_Ad7_external_source_packet.json"
    ad_rows = read(ad_packet_path)["rows"]
    observations = user_rows + old_rows + ad_rows
    measurements = []
    nearest_by_state = {}
    for state in states:
        scene = read(state["scene_path"])
        with np.load(state["camera_path"], allow_pickle=False) as arrays:
            pts = arrays["pts"].copy()
            cameras = arrays["camera25"].copy()
            aliases = arrays["exposure_representative_pts"].copy()
        nearest_by_state[state["label"]] = {}
        for source in observations:
            if sha(source["source_raw_frame"]) != source["source_file_sha256"]:
                raise ValueError("Actual source image changed")
            uv = np.asarray(source["uv"], float)
            if "normal" in source:
                normals, roundoff = unit_source_normals(source["normal"], len(uv))
                normal_origin = "original_registered_user_stroke_normal"
            else:
                tangent = uv[-1] - uv[0]
                normal = np.array([-tangent[1], tangent[0]])
                normal /= np.linalg.norm(normal)
                normals = np.repeat(normal[None], len(uv), axis=0)
                roundoff = 0.
                normal_origin = "normal_to_original_two_RGB_axis_samples; not a new pixel measurement"
            index = np.flatnonzero(pts == int(source["pts"]))
            if len(index) != 1 or int(aliases[index[0]]) != int(source["actual_pts"]):
                raise ValueError("Exact native source exposure binding failed")
            camera = cameras[index[0]]
            row = dict(source, uv=uv, world=frozen_scene_curve(scene, source["curve"]),
                sigma=np.ones(len(uv)), normal=normals)
            # Same 1025 current-material samples and full native Brown source
            # evaluator as NativeIceCurveFactor, with sigma removed for pixels.
            raw = absolute_value_jac15(camera, [row], False).reshape(-1, 2)
            signed = np.einsum("ni,ni->n", raw, normals)
            diagnostic_intervals = source.get("annotation_stroke_normal_interval_px")
            if diagnostic_intervals is None:
                diagnostic_intervals = source.get("normal_interval_px")
            if diagnostic_intervals is None:
                diagnostic_intervals = [[(s["source_q10_q50_q90_axis"][0] - s["source_q10_q50_q90_axis"][1]) * normals[i, 1],
                    (s["source_q10_q50_q90_axis"][2] - s["source_q10_q50_q90_axis"][1]) * normals[i, 1]]
                    for i, s in enumerate(source["source_search_support"])]
            bounds = np.asarray(diagnostic_intervals, float)
            outside = signed - np.clip(signed, bounds[:, 0], bounds[:, 1])
            measurement = dict(state=state["label"], q_sha256=state["q_sha256"], id=source["id"],
                pts=source["pts"], actual_pts=source["actual_pts"], curve=source["curve"],
                source_uv=uv.tolist(), unit_source_normal=normals.tolist(), source_normal_origin=normal_origin,
                source_normal_roundoff=roundoff, signed_model_minus_source_normal_px=signed.tolist(),
                absolute_normal_mean_px=float(np.abs(signed).mean()),
                absolute_normal_median_px=float(np.median(np.abs(signed))),
                absolute_normal_maximum_px=float(np.abs(signed).max()),
                raw_vector_mean_px=float(np.linalg.norm(raw, axis=1).mean()),
                raw_model_minus_source_UV=raw.tolist(),
                outside_recorded_support_normal_mean_px=float(np.abs(outside).mean()),
                outside_recorded_support_normal_maximum_px=float(np.abs(outside).max()),
                support_role="annotation stroke extent, not optical accuracy interval" if source.get("user_confirmed_source_axis")
                    else "observed RGB q10-q90 blur support; not exact physical boundary",
                user_confirmed_source_identity=bool(source.get("user_confirmed_source_axis")),
                inherited_later_identity_context="existing actual RGB band-navigation chain, not another direct user confirmation"
                    if source in old_rows else None,
                physical_pixel_or_whole_shot_accuracy_accepted=False)
            measurements.append(measurement)
            if source in user_rows:
                nearest_by_state[state["label"]][source["id"]] = uv + raw
    ranking = []
    for state in states:
        selected = [m for m in measurements if m["state"] == state["label"] and m["user_confirmed_source_identity"]]
        ranking.append(dict(state=state["label"], user_both_lines_average_absolute_normal_px=float(np.mean([m["absolute_normal_mean_px"] for m in selected])),
            worst_user_point_absolute_normal_px=max(m["absolute_normal_maximum_px"] for m in selected),
            per_line=[dict(id=m["id"], mean=m["absolute_normal_mean_px"], maximum=m["absolute_normal_maximum_px"]) for m in selected]))
    ranking.sort(key=lambda r: r["user_both_lines_average_absolute_normal_px"])
    # One native-scale source crop per baseline/latest stacked without resizing.
    # Only observed user axes and closest model POINTS appear; this is a residual
    # diagnostic, not a complete grid rendering or whole-shot acceptance review.
    raw = cv2.imread(user_rows[0]["source_raw_frame"])
    labels = [states[0]["label"], states[1]["label"], states[-1]["label"]]
    panels = []
    for label in labels:
        crop = raw[800:1080].copy()
        for source in user_rows:
            color = (50, 113, 233) if source["feature_id"] == "lane2" else (130, 96, 21)
            uv = np.asarray(source["uv"])
            projected = nearest_by_state[label][source["id"]]
            for a, b in zip(uv, projected):
                a = tuple(np.rint(a - [0, 800]).astype(int))
                b = tuple(np.rint(b - [0, 800]).astype(int))
                cv2.circle(crop, a, 3, color, -1, cv2.LINE_AA)
                cv2.drawMarker(crop, b, (210, 30, 220), cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
                cv2.line(crop, a, b, (210, 30, 220), 1, cv2.LINE_AA)
        header = np.full((42, 1920, 3), 245, np.uint8)
        values = next(r for r in ranking if r["state"] == label)
        cv2.putText(header, f"{label} | SOURCE dots: user axes; CROSS: model nearest point | mean {values['user_both_lines_average_absolute_normal_px']:.2f} px; max {values['worst_user_point_absolute_normal_px']:.2f} px", (10, 26), 0, .6, (20, 20, 20), 1)
        panels.append(np.vstack((header, crop)))
    image_path = out / "native230501120_user_axes_model_residual_points.png"
    cv2.imwrite(str(image_path), np.vstack(panels))
    report = dict(schema="rinkspace.immutable-user-lane-source-state-comparison.v1",
        source_packet=dict(path=str(PACKET), sha256=sha(PACKET)), states=states, measurements=measurements,
        same_frame_external_Ad7_source_packet=dict(path=str(ad_packet_path), sha256=sha(ad_packet_path)),
        best_by_registered_user_axes=ranking[0], ranking=ranking,
        native_scale_residual_diagnostic=str(image_path), native_crop_xyxy=[0, 800, 1920, 1080],
        scene_geometry_read_from_authenticated_solver_export=True,
        runtime_source_validation="Every imported legacy module SHA-checked by RuntimeSourceFinder; global unrelated dependency closure not checked during concurrent geometry edit",
        source_operator="existing frozen_scene_curve and absolute_value_jac15; same current lane source material sampling",
        source_points_models_and_input_packets_modified=False,
        input_fitted_user_axes_are_not_independent_holdout_accuracy=True,
        whole_shot_or_complete_video_accuracy_accepted=False)
    report_path = out / "user_lane_immutable_state_normal_comparison.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(report=str(report_path), native_comparison=str(image_path), ranking=ranking), ensure_ascii=False))


if __name__ == "__main__":
    main()
