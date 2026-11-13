"""Replay the jointly fitted state, audit it, and render the native-source comparison."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .body_skate_fit import Assembly, Problem, quantiles, revised_binding
from .inputs import PROGRAMME, PROJECT, atomic_json, atomic_npz, distribute, emit, read, digest
from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference, curve_from_packet
from calibration.program_tracking.body_tracking.continuous_physics import ContinuousMassGeometry
from calibration.program_tracking.body_tracking.programme_com import export_com_diagnostics
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
from calibration.program_tracking.body_tracking.sequence_skate_state import shoe_curve_from_packet
from calibration.program_tracking.body_tracking.skate_delivery import export_candidate
from calibration.program_tracking.body_tracking.skate_physics import fk_state


VARIANT = "whole_skate_joint_1112"
VIDEO = "body_skates_before_after.mp4"
FOOT_COLORS = ((73, 242, 112), (244, 179, 51))
BODY_COLOR = (228, 185, 245)
SIZE = (2400, 1080)
PROOF_SECONDS = (29.68, 36.46, 66.86, 80.60, 89.32, 90.12, 105.98, 118.62, 151.02, 151.16, 186.84)


def verify_checkpoint_evidence(problem, checkpoint, packet):
    if "visual_sha256" in packet:
        if str(packet["visual_sha256"]) != problem.visual_sha256:
            raise ValueError("Whole-skate evidence bytes differ from the checkpoint")
        return
    if Path(str(packet["visual_path"])).resolve() == (problem.visual / "whole_follow_evidence.npz").resolve():
        return
    record_path = PROGRAMME / "skate_video_1112/record.json"
    if record_path.is_file():
        record = read(record_path)
        registered = {item["sha256"] for item in record["controls"].values()}
        if digest(checkpoint) in registered and problem.visual_sha256 == record["visual"]["sha256"]:
            return
    raise ValueError("Moved checkpoint and evidence are not a registered byte-identical pair")


def saved_ref(problem, checkpoint, *, apply_socket=True):
    with np.load(checkpoint, allow_pickle=False) as p:
        np.testing.assert_array_equal(p["source_pts"], problem.i.pts)
        np.testing.assert_array_equal(p["source_time_base"], problem.i.tb)
        np.testing.assert_array_equal(p["camera_sha256"], problem.i.camera_sha)
        verify_checkpoint_evidence(problem, checkpoint, p)
        ref = absolute_reference(problem.ref, curve_from_packet(p))
        ref.relative_skate_curve = shoe_curve_from_packet(p)
        ref.binding = revised_binding(problem.binding_seed, p["shared"])
        ref.binding.reference = ref
        if "wear_reference_ankle_controls" in p:
            from .foot_attachment import ankle_curve_from_packet, apply_wear_reference
            ref.binding.respect_open_collar = bool(p.get("wear_respect_open_collar", False))
            apply_wear_reference(ref, ankle_curve_from_packet(p))
        if apply_socket and "socket_shift_boot_units" in p:
            from .foot_socket import apply_socket_reference
            apply_socket_reference(ref, p["socket_shift_boot_units"])
        return ref


def old_ref(problem):
    ref = absolute_reference(problem.ref, problem.body_seed)
    ref.relative_skate_curve = problem.shoe_seed
    ref.binding = revised_binding(problem.binding_seed, problem.shared_seed)
    ref.binding.reference = ref
    return ref


def observations_px(problem, coco):
    index = np.arange(len(coco))[:, None]
    ids = problem.obs["model_joint_ids"]
    use = problem.obs["use"] & np.isfinite(problem.obs["uv"]).all(axis=2)
    error = np.linalg.norm(coco[index, ids] - problem.obs["uv"], axis=-1)
    return quantiles(error[use])


def phase_speeds(i, packet):
    time = np.diff(i.times)
    same = i.shots[:-1] == i.shots[1:]
    r = packet["skate_R_world_mid"]
    angle = Rotation.from_matrix((r[:-1].swapaxes(-1, -2) @ r[1:]).reshape(-1, 3, 3)).magnitude()
    omega = angle.reshape(-1, 2) / time[:, None]
    speed = np.linalg.norm(np.diff(packet["skate_origin_world_mid"], axis=0), axis=-1) / time[:, None]
    accel = np.linalg.norm(packet["com_acceleration_world_mps2"], axis=-1)
    return dict(skate_linear_mps=quantiles(speed[same]), skate_angular_radps=quantiles(omega[same]),
                model_com_acceleration_mps2=quantiles(accel))


def dimensional_record(binding):
    return {name: value if name.endswith("_fraction") else value * binding.ratio
            for name, value in asdict(binding.packet()["parameters"]).items()} | {
                "upper_height_scale": float(np.exp(binding.upper_height_log))
            }


def open_collar_clearance(problem, ref):
    indices = np.arange(0, len(problem.i.pts), 5)
    floor, cavity = [], []
    binding = ref.binding
    previous = getattr(binding, "respect_open_collar", False)
    binding.respect_open_collar = True
    try:
        for a in range(0, len(indices), 256):
            times = ref.times[indices[a:a + 256]]
            s = fk_state(ref, ref.at(times), np.zeros((len(times), 69)))
            m = binding.foot_metrics(s["native"], s["frames"], skate_relative_R=s["skate_relative_R"])
            for v in m["violations"]:
                floor.extend(v[..., 0].max(axis=1))
                cavity.extend(v.max(axis=(1, 2)))
    finally:
        binding.respect_open_collar = previous
    return dict(source_stride=5, maximum_wall_mm=quantiles(np.asarray(cavity) * binding.ratio * 1000),
                sole_wall_mm=quantiles(np.asarray(floor) * binding.ratio * 1000))


def archived_edge_check(problem, ref, rows=None):
    if rows is None:
        rows = [row for row in problem.i.checks if row["semantic"] == "runner_lower_edge" and row["foot"] in (0, 1)]
    if not rows:
        return {}, {}
    frames = np.unique([problem.i.index[row["pts"]] for row in rows])
    edges = ref.binding.packet()["edges"]
    state = fk_state(ref, ref.at(problem.times[frames].ravel()), np.zeros((5 * len(frames), 69)))
    xyz = state["O"][:, :, None, None] + np.einsum("nfij,ekj->nfeki", state["R"], edges)
    uv = moving_projection(problem.camera, xyz.reshape(len(frames), 5, -1, 3), frames)["uv"]
    uv = uv.reshape(len(frames), 2, *edges.shape[:-1], 2)
    lookup = dict(zip(frames, range(len(frames))))
    scores = {}
    for index, row in enumerate(rows):
        f = problem.i.index[row["pts"]]
        curve = distribute(np.array(row["polyline_uv"], float), 32)
        # Either real edge may be the visible side. Use the closer of two
        # edge curves for the whole observation; the support posts and sole
        # never enter this distance.
        distance = [cKDTree(edge).query(curve)[0].mean() for edge in uv[lookup[f], row["foot"]]]
        key = f"{index}:{row.get('id') or row.get('raw_group_id') or str(row['pts'])}"
        scores[key] = float(min(distance))
    return quantiles(list(scores.values())), scores


def navigation(problem, output, report=None):
    base = read(problem.visual / "navigation.json")
    i = problem.i
    events = []
    times = list(PROOF_SECONDS)
    if report:
        times += [row["source_seconds"] for row in report["worst_held_out"][:8]]
    for sec in sorted(set(times)):
        middle = int(np.argmin(abs(i.times - sec)))
        first = max(0, middle - 16)
        last = min(len(i.pts) - 1, middle + 26)
        events.append(dict(id=f"joint_fit_{i.pts[middle]}", start_pts=int(i.pts[first]),
                           end_pts=int(i.pts[last]), label=f"Review {i.times[middle]:.2f}s"))
    base["events"] = events
    atomic_json(output / "navigation.json", base)


def export(problem, output, recovery, baseline=None):
    t0 = time.perf_counter()
    checkpoint = output / "controls.npz"
    result = output / "fitted" / "result.npz"
    ref = saved_ref(problem, checkpoint)
    old = old_ref(problem) if baseline is None else saved_ref(problem, baseline / "controls.npz")
    i = problem.i
    ref.export_source_path = i.parent_path
    shared = dimensional_record(ref.binding)
    before_geometry = dimensional_record(old.binding)
    fit_receipt = read(output / "fit_validation.json")
    if baseline is not None:
        expected = fit_receipt.get("fit_parent_sha256")
        if expected is not None and expected != digest(baseline / "controls.npz"):
            raise ValueError("Review baseline does not match the fitting parent")
    parameter_sha = digest(checkpoint)
    if result.exists():
        with np.load(result, allow_pickle=False) as file:
            packet = dict(file)
        if json.loads(str(packet["parameter_provenance_json"]))["controls_sha256"] != parameter_sha:
            raise FileExistsError("An export from different joint controls is present")
        emit("resume_native_report", result=str(result))
    else:
        emit("replay_full_native_state", frames=len(i.pts), result=str(result))
        packet = export_candidate(i.prepared, output, ref, problem.camera, problem.info,
                                  problem.doc.get("supports", []), write=False)
        times = ref.times
        zero = np.zeros((len(times), 69))
        samples = ref.at(times)
        state = fk_state(ref, samples, zero)
        mass = ContinuousMassGeometry(ref)
        C, R = mass.from_pose(state)
        packet.update(export_com_diagnostics(mass, times, C, R))
        com_projected = packet["com_world_mid"].copy()
        com_projected[:, 2] = 0.
        packet.update(native_body_local_R_mid=state["local_R"],
                  native_body_FK_world_R_mid=state["Q"][:, None] @ state["frames"],
                  P_velocity_world_mps=ref.root_derivative(times, 1) * ref.binding.ratio,
                  com_projection_world_mid=com_projected, accuracy_accepted=np.array(False),
                  recovery_mode=np.array("new_joint_fit"),
                  footwear_integration=np.array(
                      "native continuous body and ankle-attached skate assemblies; "
                      "foot articulation preserves the fitted skate poses" if "hidden_foot_fit" in fit_receipt
                      else "native continuous body plus both ankle-attached skate assemblies; shared adjustable geometry"),
                  parameter_provenance_json=np.array(json.dumps({
                      "fitted_metric_parameters": shared, "starting_metric_parameters": before_geometry,
                      "controls": str(checkpoint), "controls_sha256": parameter_sha,
                      "source_skate_evidence": str(problem.visual / "whole_follow_evidence.npz"),
                      "source_skate_evidence_sha256": digest(problem.visual / "whole_follow_evidence.npz"),
                      "shared_two_skates": True, "measured_athlete_dimensions": False,
                      "hidden_foot_fit": fit_receipt.get("hidden_foot_fit"),
                      "socket_alignment": fit_receipt.get("socket_alignment"),
                  })),
                  solver_config_json=np.array(json.dumps(dict(source_nodes=5, body_silhouette_stride=3,
                                                              shared_shape_registration=True, native_camera=True))),
                  active_blocks_json=np.array(json.dumps(dict(root_xyz=True, root_rotation=True,
                                                               hip_knee_ankle=True, relative_shoe_pose=True,
                                                               adjustable_shared_geometry=True))),
                  stopping_reason_json=np.array(json.dumps(f"finished {fit_receipt['accepted']} accepted joint iterations")),
                  costs_json=np.array(json.dumps(fit_receipt)))
    np.testing.assert_array_equal(packet["pts"], i.pts)
    np.testing.assert_array_equal(packet["source_pts"], i.pts)
    np.testing.assert_allclose(packet["native_body_local_R_mid"], packet["local_R_mid"], atol=2.e-12, rtol=0)
    if json.loads(str(packet["camera_binding_json"]))["state_sha256"] != i.camera_sha:
        raise ValueError("Export camera state changed")
    for key in ("skate_R_world_mid", "skate_origin_world_mid", "com_world_mid", "com_projection_world_mid"):
        if not np.isfinite(packet[key]).all():
            raise ValueError(key)
    if not result.exists():
        result.parent.mkdir(exist_ok=True)
        atomic_npz(result, **packet)

    def scoring(p):
        return dict(body_landmark_px=observations_px(problem, p["COCO_uv"]),
                    cavity_max_mm=quantiles(1000 * p["foot_cavity_max_violation_m_mid"]),
                    ice_below_allowance_mm=quantiles(1000 * np.maximum(-p["skate_clearance_m_mid"], 0)),
                    feet_below_allowance=int(np.count_nonzero(p["skate_clearance_m_mid"] < 0)),
                    source_support_abs_gap_mm=quantiles(1000 * np.abs(p["skate_support_gap_m_mid"])),
                    motion=phase_speeds(i, p))
    baseline_result = i.parent_path if baseline is None else baseline / "fitted" / "result.npz"
    with np.load(baseline_result, allow_pickle=False) as saved:
        a, b = scoring(saved), scoring(packet)
        a["dimensions"] = json.loads(str(saved["skate_dimensions_json"]))
        b["dimensions"] = json.loads(str(packet["skate_dimensions_json"]))
        length = np.linalg.norm(packet["P_world_mid"] - saved["P_world_mid"], axis=-1)
        view = saved["P_world_mid"] - problem.camera.midpoint["centre"] * problem.info["metres_per_scene_unit"]
        depth = (view * (packet["P_world_mid"] - saved["P_world_mid"])).sum(-1) / np.linalg.norm(view, axis=-1)
        angle = packet["skate_R_world_mid"].swapaxes(-1, -2) @ saved["skate_R_world_mid"]
        delta_deg = Rotation.from_matrix(angle.reshape(-1, 3, 3)).magnitude() * 180 / np.pi
    a["archived_cutting_edge_px"], a_edge = archived_edge_check(problem, old)
    b["archived_cutting_edge_px"], b_edge = archived_edge_check(problem, ref)
    if fit_receipt.get("hidden_foot_fit") is not None:
        a["open_collar_clearance"] = open_collar_clearance(problem, old)
        b["open_collar_clearance"] = open_collar_clearance(problem, ref)
    with np.load(output / "held_out_silhouette.npz", allow_pickle=False) as f:
        before, after = f["before_px"][..., 2], f["after_px"][..., 2]
    at = np.argwhere(np.isfinite(after))
    order = np.argsort(after[tuple(at.T)])[::-1]
    worst = [dict(pts=int(i.pts[j]), source_seconds=float(i.times[j]), foot=int(k),
                  before_px=float(before[j, k]), after_px=float(after[j, k]))
             for j, k in at[order[:16]]]
    validation = dict(before=a, after=b, held_out_outline=fit_receipt, worst_held_out=worst,
                      root_change_m=quantiles(length), root_depth_change_m=quantiles(np.abs(depth)),
                      skate_orientation_change_deg=quantiles(delta_deg),
                      geometry=dict(before=before_geometry, after=shared),
                      source_pts=[int(i.pts[0]), int(i.pts[-1])], native_frames=len(i.pts),
                      native_camera_sha256=i.camera_sha, source_video_sha256=i.media_sha,
                      result_sha256=digest(result), controls_sha256=parameter_sha,
                      edge_label_origin="historical source-crop check traces; foot hypotheses; excluded as silhouette factors",
                      edge_errors_by_id={k: [a_edge[k], b_edge[k]] for k in a_edge},
                      absolute_3d_accuracy_accepted=False,
                      fit_scope="continuous root, hip, knee, ankle, relative skate rotation and shared geometry",
                      elapsed_seconds=time.perf_counter() - t0)
    atomic_json(output / "audit.json", validation)
    navigation(problem, output, validation)

    config = read((recovery if baseline is None else baseline) / "viewer3d.json")
    old_variant = config["variants"][0]
    variant_id = VARIANT
    label = "整件鞋刀与人体联合重拟合"
    if "local_review" in fit_receipt:
        variant_id = "whole_skate_joint_reviewed_1112"
    if "hidden_foot_fit" in fit_receipt:
        variant_id = "whole_skate_mount_1112"
        label = "鞋口装配与鞋内足部修正版"
    if "socket_alignment" in fit_receipt:
        variant_id = "whole_skate_socket_1112"
        label = "鞋内踝点与双腿联合修正版"
    variant = dict(old_variant, id=variant_id, candidate_id=variant_id,
                   label=label, body_reference_label=old_variant["label"],
                   body_refinement_intervals_seconds=[[float(i.times[0]), float(i.times[-1])]],
                   result=str(result), result_sha256=validation["result_sha256"],
                   parameter=str(checkpoint), parameter_sha256=parameter_sha,
                   source_projection_origin="full native five-node joint refit replay")
    variants = [variant, *(v for v in config["variants"] if v["id"] != variant_id)]
    config.update(variants=variants, default_variant=variant_id, session_id=variant_id,
                  latest_review_variant=variant_id, comparison_variants=[variant_id, old_variant["id"]],
                  source_navigation=str(output / "navigation.json"),
                  body_window_scope="九个镜头的连续人体和鞋刀联合重拟合",
                  body_scope_label="新拟合参数覆盖全节目",
                  footwear_scope_label="双鞋刀使用同一组全片拟合尺寸，各自固定安装在对应足部",
                  review_status="原片轮廓与人体观测重拟合。质心为同一新版人体与鞋刀质量模型的估计值。")
    config["initial_visible"].update(com=True, com_projection=True, trajectory=True, pelvis=False,
                                     pelvis_trajectory=False)
    atomic_json(output / "viewer3d.json", config)
    launcher = PROJECT / "launchers" / "START_3D_RESULTS.ps1"
    cmd = f'@echo off\npowershell.exe -NoProfile -ExecutionPolicy Bypass -File "{launcher}" -Config "%~dp0viewer3d.json" %*\n'
    (output / "START_REFITTED_3D.cmd").write_text(cmd, encoding="ascii")
    emit("native_state_and_viewer_ready", frames=len(i.pts), result=str(result),
         report=str(output / "audit.json"), seconds=time.perf_counter() - t0)


def display_mesh(surface):
    # Vertex clustering in model space only, with articulation groups to
    # preserve the hands, separate legs, and the actual skinned body model.
    xyz = surface.shaped * surface.body.scale
    cell = np.floor(xyz / .028).astype(int)
    major = surface.weights.argmax(axis=1)
    major[major >= 25] = np.where(major[major >= 25] < 40, 20, 21)
    cluster, inverse = np.unique(np.c_[cell, major], axis=0, return_inverse=True)
    centroid = np.zeros((len(cluster), 3))
    np.add.at(centroid, inverse, xyz)
    centroid /= np.bincount(inverse)[:, None]
    order = np.argsort(np.linalg.norm(xyz - centroid[inverse], axis=1))
    _, selected = np.unique(inverse[order], return_index=True)
    ids = order[selected]
    faces = inverse[surface.faces]
    keep = ((faces[:, 0] != faces[:, 1]) & (faces[:, 0] != faces[:, 2]) & (faces[:, 1] != faces[:, 2]))
    faces = np.unique(np.sort(faces[keep], axis=1), axis=0)
    return ids, faces


def projection_batch(problem, ref, frames, assembly, vertex_ids):
    query = problem.times[frames].ravel()
    state = fk_state(ref, ref.at(query), np.zeros((len(query), 69)))
    surface = ref.binding.surface
    kwargs = {} if "hand_local_R" not in state else dict(hand_local_R=state["hand_local_R"])
    vertex = surface.sample_vertices(state["native"], state["frames"], vertex_ids,
                                     double_precision=True, **kwargs)
    body = state["P"][:, None] + np.einsum("nij,nkj->nki", state["Q"], vertex)
    shoes = [assembly.points(state, f, transforms=(state["R"], state["O"])) for f in range(2)]
    world = np.concatenate((body, *shoes), axis=1)
    uv = moving_projection(problem.camera, world.reshape(len(frames), 5, -1, 3), frames)["uv"]
    return uv[:, :len(vertex_ids)], uv[:, len(vertex_ids):].reshape(len(frames), 2, -1, 2)


def tint(frame, points, faces, color, opacity):
    lo = np.maximum(0, np.floor(points.min(0) - 3)).astype(int)
    hi = np.minimum(frame.shape[1::-1], np.ceil(points.max(0) + 4)).astype(int)
    w, h = hi - lo
    if w <= 1 or h <= 1:
        return
    vertices = np.rint(np.clip(points - lo, -8000, 8000)).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    for tri in vertices[faces]:
        cv2.fillConvexPoly(mask, tri, 255)
    view = frame[lo[1]:hi[1], lo[0]:hi[0]]
    layer = np.empty_like(view)
    layer[:] = color
    cv2.copyTo(cv2.addWeighted(view, 1. - opacity, layer, opacity, 0), mask, view)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(view, contours, -1, color, 1, cv2.LINE_AA)


def put(frame, label, x, y, size=.60, color=(240, 240, 240)):
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def bounds(points, margin, shape):
    finite = np.concatenate(points, axis=0)
    finite = finite[np.isfinite(finite).all(axis=1)]
    if not len(finite):
        return 0, 0, shape[1], shape[0]
    lo = np.floor(finite.min(axis=0) - margin).astype(int)
    hi = np.ceil(finite.max(axis=0) + margin).astype(int)
    lo = np.clip(lo, 0, np.asarray(shape[1::-1]) - 2)
    hi = np.clip(hi, lo + 2, shape[1::-1])
    return (*lo, *hi)


def panel_crop(panel, frame, box):
    left, top, right, bottom = box
    chip = frame[top:bottom, left:right]
    h, w = panel.shape[:2]
    scale = min(h / chip.shape[0], w / chip.shape[1], 5.)
    target = cv2.resize(chip, (max(1, round(chip.shape[1] * scale)), max(1, round(chip.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    yy, xx = (h - target.shape[0]) // 2, (w - target.shape[1]) // 2
    panel[yy:yy + target.shape[0], xx:xx + target.shape[1]] = target


def comparison(problem, assembly, face, uv, shoe, j, i, source, captured, titles):
    old_body, new_body = uv[0][j], uv[1][j]
    old_skate, new_skate = shoe[0][j], shoe[1][j]
    # One crop per row, used verbatim for source and both models. The raw
    # frame remains complete in the upper row.
    body_box = bounds([old_body, new_body, *old_skate, *new_skate], 25, source.shape)
    foot_boxes = [bounds([old_skate[f], new_skate[f], problem.region[i, f]], 14, source.shape)
                  for f in range(2)]
    images = [source]
    for index in range(2):
        img = source.copy()
        tint(img, uv[index][j], face, BODY_COLOR, .21)
        for f, color in enumerate(FOOT_COLORS):
            tint(img, shoe[index][j, f], assembly[index].faces, color, .35)
        images.append(img)
    canvas = np.full((SIZE[1], SIZE[0], 3), 19, np.uint8)
    for col, (img, title) in enumerate(zip(images, titles)):
        x = col * 800
        put(canvas, title, x + 14, 28, .69)
        canvas[42:492, x:x + 800] = cv2.resize(img, (800, 450), interpolation=cv2.INTER_LINEAR)
        panel_crop(canvas[502:1004, x:x + 434], img, body_box)
        for f, box in enumerate(foot_boxes):
            y = 510 + f * 246
            panel_crop(canvas[y + 24:y + 242, x + 438:x + 795], img, box)
            put(canvas, "L skate" if f == 0 else "R skate", x + 450, y + 17, .56, FOOT_COLORS[f])
    sec = problem.i.times[i]
    put(canvas, f"{problem.i.shots[i].upper()}   Source {sec:07.3f} s   PTS {problem.i.pts[i]}", 18, 1051, .68)
    put(canvas, "Same source image and camera in all three views", 830, 1051, .60)
    if i in captured:
        cv2.imwrite(str(captured[i]), canvas)
    return canvas


def render_video(problem, output, *, proof_only=False, baseline=None, references=None,
                 proof_seconds=PROOF_SECONDS, source_window=None, parameter=None, refit_label=None):
    # Existing NVENC writer and exact PTS validation consume the generator.
    # The earlier isolated experiments use short sibling imports.
    sys.path.insert(0, str(Path(__file__).parent))
    from render_review import render
    sys.path.pop(0)
    cv2.setNumThreads(2)
    if references is None:
        old = old_ref(problem) if baseline is None else saved_ref(problem, baseline / "controls.npz")
        refs = [old, saved_ref(problem, output / "controls.npz")]
        titles = ("SOURCE", "BEFORE  |  Restored joint fit", "AFTER  |  Whole skate joint refit") if baseline is None else (
            "SOURCE", "BEFORE  |  Previous joint fit", "AFTER  |  Revised skate attachment")
    else:
        refs = references
        titles = ("SOURCE", "BEFORE  |  Previous joint fit", "AFTER  |  " + (refit_label or "ExMR + native body/skate fit"))
    assemblies = [Assembly(r.binding) for r in refs]
    ids, face = display_mesh(refs[0].binding.surface)
    emit("native_reprojection_start", body_render_vertices=len(ids), body_render_triangles=len(face),
         skate_render_vertices=[len(a.vertices) for a in assemblies],
         body_full_vertices=len(refs[0].binding.surface.shaped), output=str(output / VIDEO))
    wanted = {int(np.argmin(abs(problem.i.times - s))) for s in proof_seconds}
    proof = output / "frames"
    proof.mkdir(exist_ok=True)
    captured = {i: proof / f"pts_{problem.i.pts[i]}.jpg" for i in wanted}
    if proof_only:
        for i in sorted(wanted):
            frames = np.array([i])
            uv, shoe = zip(*(projection_batch(problem, r, frames, a, ids) for r, a in zip(refs, assemblies)))
            source = next(problem.i.frames(i, i + 1))[1]
            comparison(problem, assemblies, face, uv, shoe, 0, i, source, captured, titles)
        return

    first,last=0,len(problem.i.pts)
    review_inputs=problem.i
    if source_window is not None:
        a,b=problem.i.tb
        first,last=np.searchsorted(problem.i.pts,np.rint(np.asarray(source_window)*b/a).astype(np.int64))
        if not 0 <= first < last <= len(problem.i.pts):
            raise ValueError("Empty or invalid native source review window")
        review_inputs=copy(problem.i)
        for key in ("pts","times","shots"):
            setattr(review_inputs,key,getattr(problem.i,key)[first:last])

    def generate():
        read_frames = iter(problem.i.frames(first, last))
        batches = iter(np.array_split(np.arange(first,last), int(np.ceil((last-first) / 20))))
        with ThreadPoolExecutor(max_workers=2) as pool:
            def submit(frames):
                return [pool.submit(projection_batch, problem, r, frames, a, ids)
                        for r, a in zip(refs, assemblies)]

            frames = next(batches)
            work = submit(frames)
            while frames is not None:
                uv, shoe = zip(*(job.result() for job in work))
                following = next(batches, None)
                work = submit(following) if following is not None else []
                for j, i in enumerate(frames):
                    pt, src = next(read_frames)
                    if int(pt) != int(problem.i.pts[i]):
                        raise ValueError("Decoder returned a different source PTS")
                    yield comparison(problem, assemblies, face, uv, shoe, j, i, src, captured, titles)
                frames = following

    if source_window is None and not (output / "navigation.json").is_file():
        navigation(problem, output)
    render(review_inputs, output, VIDEO, generate(), SIZE)
    atomic_json(output / "video_receipt.json", dict(path=str(output / VIDEO), sha256=digest(output / VIDEO),
                exposures=len(review_inputs.pts), source_time_base=list(problem.i.tb),
                source_pts=[int(review_inputs.pts[0]), int(review_inputs.pts[-1])],
                native_library=problem.info["native_library"], camera_sha256=problem.i.camera_sha,
                controls_sha256=digest(parameter or (output / "controls.npz")), five_actual_exposure_nodes=True,
                render_body_vertices=len(ids), render_body_triangles=len(face),
                rendering="same full body model with render-only mesh reduction; dense mesh in 3D player"))


def main():
    parser = ArgumentParser()
    parser.add_argument("mode", choices=("export", "video", "proof"))
    parser.add_argument("visual", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("recovery", type=Path)
    parser.add_argument("--baseline", type=Path)
    a = parser.parse_args()
    problem = Problem(a.visual, a.output)
    if a.mode == "export":
        export(problem, a.output, a.recovery, a.baseline)
    else:
        render_video(problem, a.output, proof_only=a.mode == "proof", baseline=a.baseline)


if __name__ == "__main__":
    main()
