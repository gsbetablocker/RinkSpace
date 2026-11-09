"""Report the E3 comparison when the accepted camera update is exactly zero."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from calibration.program_tracking.body_tracking.programme_contract import required_clock
from rinkspace.path_policy import guard_project_write
from rinkspace.storage import AssetStore
from scripts.install_structure_candidates import (
    BASELINE_CAMERA,
    PROGRAMME,
    PROJECT,
    camera_identity,
    camera_projection_differences,
    check_camera_binding,
    file_ref,
    read,
    write_once,
)


SURFACE = "surface_intersection"
BODY_ARRAYS = ("P_world_mid", "Q_world_mid", "local_R_mid", "skate_R_world_mid")
SHOE = {
    "old_gear_pixels", "dynamic_foot_compatibility", "neutral_foot_compatibility",
    "finite_relative_skate_wear_prior", "full_source_skate_structure",
    "source_pixel_edge_profile", "complete_runner_extent",
    "complete_visible_structure_extent", "unassigned_source_skate_sets",
}
TIME_FLIGHT = {
    "source_motion", "actual_so3_angular_acceleration", "actual_root_translation_acceleration",
    "actual_local_acceleration", "relative_skate_angular_velocity",
    "actual_skate_so3_angular_acceleration", "temporal_contact_mode", "finite_ice",
    "continuous_support", "episode_support", "native_source_support_contact",
    "step_runner_lateral", "step_runner_travel", "flight_integrated_com",
    "flight_specific_momentum", "rotation_direction_path", "rotation_path_impulse",
    "source_facing_cone",
}
STRUCTURE = {"self_collision", SURFACE, "anatomical_joint_domain", "source_supported_grasp",
             "grip", "signed_joint_relation", "cavity_AL"}


def require(condition, detail):
    if not condition:
        raise ValueError(detail + "。需要重新核实输入或真实共同评估，无法生成退化 E3 对照。")


def close_cost(a, b):
    return math.isclose(a, b, rel_tol=2e-12, abs_tol=1e-8)


def verify_camera(store, report_path, structure, exposure_count, *, selected_path=None):
    report = read(report_path)
    require(report["schema"] == "rinkspace.endpoint-camera-block.v1"
            and report["body_sha256"] == structure["sha256"]
            and report["body_exposures"] == exposure_count,
            "相机报告未绑定指定的全节目结构参数")
    before = camera_identity(store, BASELINE_CAMERA)
    round_camera = camera_identity(store, report_path.with_name("camera.json"))
    after = camera_identity(store, selected_path) if selected_path is not None else round_camera
    old, new = (read(item["record"]["path"]) for item in (before, after))
    if after != round_camera:
        selection = new["selection_report"]
        require(new["schema"] == "rinkspace.conditional_endpoint_camera_selection.v1"
                and new["body_parameter_sha256"] == structure["sha256"]
                and new["parent_camera_sha256"] == before["state"]["sha256"]
                and new["source_exposures"] == exposure_count and new["control_change_max"] == 0
                and Path(selection["path"]).resolve() == report_path.resolve()
                and selection["sha256"] == file_ref(report_path)["sha256"]
                and after["state"]["sha256"] == round_camera["state"]["sha256"],
                "联合阶段实际使用的独立相机记录未绑定端点报告与相同 compact")
    require(after["record"] != before["record"]
            and Path(new["parent_camera_record"]).resolve() == Path(before["record"]["path"])
            and old["point_input_sha256"] == new["point_input_sha256"] == report["static_observations_sha256"]
            and old["scene_sha256"] == new["scene_sha256"] == report["static_scene_sha256"]
            and old["native_library"]["sha256"] == new["native_library"]["sha256"] == report["native_library_sha256"],
            "相机记录的父记录或原生静态场景、观测、运行库身份不一致")
    differences = camera_projection_differences(before["state"]["path"], after["state"]["path"])
    require(not differences and report["control_change_max"] == 0,
            "实际相机数值或报告控制量有变化 " + ", ".join(differences))
    with np.load(before["state"]["path"], allow_pickle=False) as a, np.load(
        after["state"]["path"], allow_pickle=False
    ) as b:
        deps = "joint_numeric_dependencies_json"
        require(json.loads(a[deps].item()) == json.loads(b[deps].item()),
                "相机的完整数值依赖不同")
        embedded = json.loads(b["endpoint_joint_candidate_json"].item())
    require(embedded["body_sha256"] == structure["sha256"] and embedded["windows"] == report["windows"],
            "相机 compact 内嵌证据与端点报告不一致")
    return before, after, report


def static_evidence(report):
    windows = []
    for window in report["windows"]:
        shot, side = window["shot"], window["side"]
        before, after = window["before"], window["after"]
        groups = window["groups"]
        require(before["fit"] == after["fit"] and before["heldout"] == after["heldout"]
                and all(row["before"] == row["after"] for row in groups.values())
                and all(value == 0 for value in window["delta_pan_tilt_logf"]),
                f"{shot} {side} 选中相机的静态证据有变化")
        validation = {
            group: dict(
                selected=groups[group]["before"],
                attempted=attempt,
                attempted_rms_delta_px=attempt["rms"] - groups[group]["before"]["rms"],
            )
            for group, attempt in window["attempted_validation"].items()
        }
        singular = window["static_singular_values"]
        directions = len(window["delta_pan_tilt_logf"])
        rank = window["static_rank"]
        checks = window["native_direction_checks"]
        require(directions == len(singular) == 9 and checks and all(math.isfinite(s) and s >= 0 for s in singular)
                and rank == sum(s > max(singular) * 1e-8 for s in singular),
                f"{shot} {side} 静态条件信息秩与实际奇异值不一致")
        windows.append(dict(
            shot=shot, side=side, seconds=window["seconds"], stop=window["stop"],
            selected_delta_pan_tilt_logf=window["delta_pan_tilt_logf"],
            selected_static_fit=before["fit"], selected_protected=before["heldout"],
            selected_groups=groups, protected_entities=window["protected_entities"],
            validation_entities=window["validation_entities"],
            attempted_validation_pass=window["validation_pass"],
            attempted_z=window["attempted_z"], attempted_validation=validation,
            conditional_static_schur=dict(
                scaled_coordinates="z, value/slope/curvature directions, per component 0.002 rad / 0.002 rad / 0.01 log f",
                static_rank=rank, relative_rank_threshold=1e-8, static_fit_singular_values=singular,
                static_fit_information_eigenvalues=[s * s for s in singular],
                min_eigenvalue_lower_bound=min(singular) ** 2 if rank == directions else 0.0,
                native_direction_checks=checks, tested_direction_valid=window["gradient_valid"],
            ),
        ))
    bounds = sorted(item["seconds"] for item in windows)
    require(bounds and all(a < b for a, b in bounds)
            and all(left[1] < right[0] for left, right in zip(bounds, bounds[1:])),
            "无法相加空窗、零长度窗或重叠端点窗的静态拟合成本")
    return dict(
        windows=windows,
        static_fit_cost_constant=math.fsum(item["selected_static_fit"]["cost"] for item in windows),
        static_fit_rows=sum(item["selected_static_fit"]["rows"] for item in windows),
        all_selected_groups_unchanged=True,
        incremental_heldout_scope=report["heldout_scope"],
        attempted_inner_history_only=report["history"],
        attempted_inner_history_cost_scope="相机小块内部目标包含局部 COCO 及步长项，不进入完整目标对照。",
        conditional_information_scope=(
            "固定场景、世界规范、曲线站位剖面和参数缩放时，静态拟合 AᵀA 是消去人体变量后的"
            "相机 Schur 信息的半正定下界。所示奇异值和有限差分只覆盖各小块，不验证场景或三维精度。"
        ),
    )


def binding_at(packet, identity, shots, origin):
    binding = check_camera_binding(packet, identity["state"]["sha256"], shots, origin)
    for item in [binding, *binding["shots"].values()]:
        require(Path(item["record"]).resolve() == Path(identity["record"]["path"])
                and Path(item["state_path"]).resolve() == Path(identity["state"]["path"]),
                f"{origin} 未绑定指定相机记录与 compact")
    return binding


def verify_body(store, stage0, stage1, paths, camera0, camera1, clock, transition_path, transition):
    pts, shots, tb = clock
    summary = read(stage1 / "run_summary.json")
    require(summary["engineering_ready"] and summary["trajectory_candidate"]
            and summary["source_frames"] == len(pts) and not summary["root_profile_fixed"]
            and not summary["diagnostic_profile"] and summary["additional_accepted_steps"] > 0,
            "联合阶段缺少已完成的全节目人体续算")
    warm_path = stage1 / "warm_start.json"
    warm = read(warm_path)
    require(warm["parameter_sha256"] == paths[0]["sha256"], "联合续算的父参数 SHA 不符")
    require(transition["explicit_revised_objective"] and transition["same_global_body_and_scale"],
            "联合续算未声明相机切换后保持同一全局人体和尺度")

    p0, p1 = (item["path"] for item in paths)
    raw0, raw1 = (stage / "fitted/result.npz" for stage in (stage0, stage1))
    weights_path = stage1 / "objective_block_weights.npz"
    weights_ref = file_ref(weights_path)
    with np.load(p0, allow_pickle=False) as a, np.load(p1, allow_pickle=False) as b, np.load(
        raw0, allow_pickle=False
    ) as r0, np.load(raw1, allow_pickle=False) as r1:
        for result in (r0, r1):
            require(all(np.array_equal(result[key], expected) for key, expected in (
                ("source_pts", pts), ("source_shot_ids", shots), ("time_base", tb))),
                "人体结果的实际 native PTS、镜头范围或 time_base 不一致")
        require(a["state_kind"].item() == b["state_kind"].item() == "absolute_motion_v1"
                and all(a[key].item() == b[key].item() for key in (
                    "prepared_input_sha256", "prepared_observations_sha256"))
                and np.array_equal(a["x"][:17], b["x"][:17])
                and all(np.array_equal(r0[key], r1[key]) for key in ("shared_betas", "shared_body_scale")),
                "续算改变了原始人体输入、共享人体或尺度")
        require(b["objective_block_weights_sha256"].item() == weights_ref["sha256"],
                "目标权重文件与联合参数的 SHA 不一致")
        binding0 = binding_at(a, camera0, shots, p0)
        binding1 = binding_at(b, camera1, shots, p1)
        require(binding0 == binding_at(r0, camera0, shots, raw0) == transition["parent"]
                and binding1 == binding_at(r1, camera1, shots, raw1) == transition["candidate"],
                "联合续算切换记录、保存参数和导出的相机上下文不一致")
        runtime = file_ref(store.source(binding1["native_library"]))
        require(runtime["sha256"] == binding0["native_library_sha256"] == binding1["native_library_sha256"]
                and all(binding0[key] == binding1[key] for key in (
                    "source_time_base", "metres_per_scene_unit", "solver_metres_per_scene_unit", "length_export_ratio")),
                "人体投影运行库、源时钟或投影尺度不同")
        changes = {}
        for name in BODY_ARRAYS:
            first, last = r0[name], r1[name]
            require(first.shape == last.shape and len(first) == len(pts)
                    and np.isfinite(first).all() and np.isfinite(last).all(), "无效人体状态 " + name)
            changes[name] = float(np.max(np.abs(last - first)))
    require(any(changes.values()), "已保留的全节目人体输出没有数值更新")
    with np.load(weights_path, allow_pickle=False) as weights:
        surface_weight = weights[SURFACE].item()
    require(surface_weight == 16.0, "实际 surface_intersection 权重与刷新凭据的固定 16 倍不同")
    return dict(
        warm_start=file_ref(warm_path), camera_transition=file_ref(transition_path),
        joint_run_summary=file_ref(stage1 / "run_summary.json"),
        structure_result=file_ref(raw0), joint_result=file_ref(raw1),
        projection_runtime=runtime, weights=weights_ref, surface_weight=surface_weight,
        additional_accepted_steps=summary["additional_accepted_steps"],
        retained_step=summary["retained_step"], original_primary_guards_pass=summary["original_primary_guards_pass"],
        source_clock=dict(exposures=len(pts), first_pts=int(pts[0]), last_pts=int(pts[-1]), time_base=list(tb)),
        max_abs_body_array_change=changes,
    )


def body_costs(stage, selection, refresh, surface_weight):
    phase_metrics = {}
    reference_cost = {
        "before": selection["run_policy"]["baseline"]["joint_cost"],
        "after": selection["selection"]["retained_metrics"]["joint_cost"],
    }
    for phase, filename in (("before", "initial_metrics.json"), ("after", "final_metrics.json")):
        metrics_path = stage / filename
        metrics = read(metrics_path)
        parts = metrics["weighted_parts"]
        require(all(math.isfinite(v) and v >= 0 for v in parts.values()), f"{phase} 含无效成本分项")
        original_sum = math.fsum(parts.values())
        require(close_cost(original_sum, reference_cost[phase]),
                f"{phase} 原分项和 {original_sum:.12g} 与独立全成本 {reference_cost[phase]:.12g} 不符")
        common_total = refresh[f"full_objective_{phase}_on_new_association"]
        surface_cost = common_total - math.fsum(value for key, value in parts.items() if key != SURFACE)
        require(math.isfinite(surface_cost) and surface_cost >= -1e-8, f"{phase} 刷新表面分项不合法")
        groups = dict(shoe_and_blade={}, time_flight_and_support={}, structure={}, other={})
        for name, cost in (parts | {SURFACE: max(0., surface_cost)}).items():
            term = name.rsplit("/", 1)[-1]
            if term in SHOE or term.startswith("additional_unassigned_source_"):
                group = "shoe_and_blade"
            elif term in TIME_FLIGHT:
                group = "time_flight_and_support"
            elif term in STRUCTURE:
                group = "structure"
            else:
                group = "other"
            groups[group][name] = cost
        phase_metrics[phase] = dict(
            original_metrics=file_ref(metrics_path), original_weighted_parts_sum=original_sum,
            independently_recorded_original_full_cost=reference_cost[phase],
            original_weighted_surface_cost=parts[SURFACE],
            full_body_cost_on_common_refreshed_association=common_total,
            surface_cost_on_common_refreshed_association=dict(
                weighted=max(0., surface_cost), unweighted=max(0., surface_cost) / surface_weight,
                geometry_metrics=refresh["surface_" + phase],
            ),
            reliable_torso_body_pixel_subgroup=metrics["groups"]["torso"],
            reliable_torso_source_joints="COCO shoulders and hips",
            body_pixel_groups=metrics["groups"],
            body_pixel_subgroups_already_in_full_cost=True,
            weighted_cost_groups={key: dict(cost=math.fsum(values.values()), weighted_parts=values)
                                  for key, values in groups.items()},
            original_primary_guards=metrics.get("primary_guards"),
        )
    return phase_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("structure-stage", "joint-stage", "camera-report", "surface-refresh", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    store = AssetStore(PROJECT)
    output = guard_project_write(args.output)
    require(store.configured and output.is_relative_to(store.data) and not output.exists(),
            "报告必须写入已登记 Data 根目录中的新文件")
    stages = [stage.resolve(strict=True) for stage in (args.structure_stage, args.joint_stage)]
    require(stages[0] != stages[1] and all(stage.is_relative_to(store.data) for stage in stages),
            "结构阶段与联合阶段必须是不同的 Data 目录")
    clock = required_clock(read(PROGRAMME / "rebuild_scope.json"))
    require(len(clock[0]) == 8329, "无法核实当前 8329 曝光时钟")
    params = [file_ref(stage / "solve/candidate.npz") for stage in stages]
    transition_path = stages[1] / "camera_block_transition.json"
    require(transition_path.is_file(), "联合续算缺少显式相机切换凭据")
    transition = read(transition_path)
    camera0, camera1, camera_report = verify_camera(
        store, args.camera_report, params[0], len(clock[0]), selected_path=transition["candidate"]["record"],
    )
    static = static_evidence(camera_report)
    refresh = read(args.surface_refresh)
    require(refresh["schema"] == "rinkspace.surface-association-refresh.v1"
            and refresh["same_refreshed_association"] and refresh["removed_witnesses_not_credited_as_cost_descent"]
            and refresh["before_parameter_sha256"] == params[0]["sha256"]
            and refresh["after_parameter_sha256"] == params[1]["sha256"],
            "共同刷新凭据与结构、联合参数或同一表面关联不符")
    body = verify_body(store, *stages, params, camera0, camera1, clock, transition_path, transition)
    selection_path = stages[1] / "solve/candidate.json"
    costs = body_costs(stages[1], read(selection_path), refresh, body["surface_weight"])
    static_cost = static["static_fit_cost_constant"]

    def trial(param, camera, phase):
        full = costs[phase]["full_body_cost_on_common_refreshed_association"]
        return dict(
            body_parameter=param, camera=camera, body_cost_details=phase,
            full_body_cost=full, static_endpoint_fit_cost=static_cost,
            augmented_common_objective=full + static_cost,
        )

    baseline = trial(params[0], camera0, "before")
    camera_only = trial(params[0], camera1, "before")
    body_only = trial(params[1], camera0, "after")
    joint = trial(params[1], camera1, "after")
    report = dict(
        schema="rinkspace.structure-joint-degenerate-trials.v1", accuracy_accepted=False,
        comparison_kind="相机小块未获留出背景支持，选中控制冻结。人体进行一次全节目续算。",
        numerical_equivalences=[["baseline", "camera_only"], ["body_only", "joint"]],
        independent_optimization_runs=1, full_simultaneous_camera_body_solve_verified=False,
        augmented_objective_definition="同一刷新关联的全节目人体成本 + 三个不相交端点窗的静态拟合成本。保护组及增量留出组只列证据。",
        surface_refresh=file_ref(args.surface_refresh), surface_association=refresh["association"],
        refreshed_witnesses_sha256=refresh["witnesses_sha256"], refreshed_witness_rows=refresh["rows"],
        nonincrease_on_refreshed_association=refresh["nonincrease_on_new_association"],
        camera_report=file_ref(args.camera_report), original_joint_cost_record=file_ref(selection_path),
        body_evidence=body, body_costs=costs, static_evidence=static,
        trials=dict(baseline=baseline, camera_only=camera_only, body_only=body_only, joint=joint),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_once(output, report)
    print(json.dumps(dict(output=str(output), before=baseline["full_body_cost"],
                          after=joint["full_body_cost"], static_fit=static_cost), ensure_ascii=True))


if __name__ == "__main__":
    main()
