"""Probe the independent skates against the original shared body and ankle binding."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from evidence import segments
from inputs import PROJECT, PROGRAMME, ROOT, Inputs, atomic_json, atomic_npz, emit, geometry, read
from model import SkateModel
from projection import Native, inverse
from bundle import measure_checks

HIP, KNEE, ANKLE = (1, 2), (4, 5), (7, 8)


def reach(joints, target):
    hip, knee, ankle = joints[:, HIP], joints[:, KNEE], joints[:, ANKLE]
    upper = np.linalg.norm(knee - hip, axis=-1)
    lower = np.linalg.norm(ankle - knee, axis=-1)
    required = np.linalg.norm(target - hip, axis=-1)
    return np.maximum.reduce((required - upper - lower, np.abs(upper - lower) - required,
                              np.zeros_like(required)))


def contact_values(p, r, shape):
    # Actual left/right rail surfaces. The minimum can move along the rocker.
    rails = shape["edges"].reshape(2, -1, 3)
    return (p[..., None, 2] + np.einsum("nfi,fki->nfk", r[:, :, 2, :], rails)).min(axis=-1)


def structural_probe(inputs, new_r, old_r, selection, parent):
    """Evaluate real skin and existing dynamic foot cavity on saved body FK."""
    from calibration.program_tracking.body_tracking.trajectory_backend import load_reference

    ref, _, info, _, _ = load_reference(PROJECT, inputs.prepared, inputs.document_path)
    binding = ref.binding
    with np.load(inputs.parent["parameter"]["path"], allow_pickle=False) as file:
        np.testing.assert_array_equal(file["x"][:17].shape, binding.x.shape)
        binding.x = file["x"][:17].copy()
    ratio = float(info["length_export_ratio"])
    selection = np.asarray(selection)
    p = parent["P_world_mid"][selection]
    q = parent["Q_world_mid"][selection]
    local = ((parent["native22_world_mid"][selection] - p[:, None]) @ q) / ratio
    frames = inverse(q)[:, None] @ parent["native_body_FK_world_R_mid"][selection]
    old_rel = parent["skate_relative_R_mid"][selection]
    change = inverse(old_r[selection]) @ new_r[selection]
    new_rel = old_rel @ change
    baseline, orig = binding.transforms(p / ratio, q, local, frames, skate_relative_R=old_rel)
    np.testing.assert_allclose(baseline, old_r[selection], atol=2e-9, rtol=0)
    np.testing.assert_allclose(orig * ratio, parent["skate_origin_world_mid"][selection], atol=2e-9, rtol=0)
    changed_r, changed_p = binding.transforms(p / ratio, q, local, frames, skate_relative_R=new_rel)
    np.testing.assert_allclose(changed_r, new_r[selection], atol=2e-9, rtol=0)
    hand = parent["hand_local_R_mid"][selection]
    output = []
    for rr in (old_rel, new_rel):
        values = []
        for a in range(0, len(selection), 24):
            sl = slice(a, a + 24)
            metrics = binding.foot_metrics(local[sl], frames[sl], skate_relative_R=rr[sl], hand_local_R=hand[sl])
            values.append(np.stack([np.max(m, axis=(1, 2)) for m in metrics["violations"]], axis=1) * ratio)
        output.append(np.concatenate(values))
    return np.stack(output, axis=0), changed_p * ratio


def describe_actions(inputs, visual, required_gap, height, parent):
    result = []
    for event in inputs.action:
        start, stop = event["start_pts"], event["end_pts_inclusive"]
        use = (inputs.pts >= start) & (inputs.pts <= stop)
        if not use.any():
            continue
        kind = event["kind"]
        row = dict(id=event["episode_id"], kind=kind, start_pts=start, end_pts=stop,
                   max_ankle_reach_excess_m=float(required_gap[use].max()))
        if kind == "rotation_source_path":
            angular = visual["world_angular_velocity"][use[1:] & use[:-1], :, 2]
            row["integrated_world_vertical_turns_diagnostic"] = (
                angular.sum(axis=0) * np.diff(inputs.times).mean() / (2*np.pi)).tolist()
        if kind == "support_candidate":
            support = [f for f, name in enumerate(("left", "right"))
                       if event["per_foot_state"].get(name) == "supported"]
            if support:
                f = support[0]
                row["support_foot"] = f
                row["rail_height_p05_p95_m"] = np.percentile(height[use, f], [5, 95]).tolist()
                row["along_lateral_p50_mps"] = [
                    float(np.median(visual[k][use, f])) for k in ("blade_along_velocity", "blade_lateral_velocity")]
        if kind == "flight_visible_candidate":
            com_accel = parent["com_acceleration_world_mps2"][use]
            row["same_body_COM_gravity_error_p50_mps2"] = float(np.nanmedian(
                np.linalg.norm(com_accel + [0., 0., 9.81], axis=-1)))
            row["visual_rails_min_height_m"] = float(height[use].min())

        result.append(row)
    relation = read(inputs.prepared / "grasp_relations.json")["grip_interval"]
    spec = read(inputs.prepared / "palm_grip_geometry.json")
    foot = int(spec["grip_foot"])
    if foot != relation["foot"] or spec["grip_hand"] != relation["hand"]:
        raise ValueError("Grasp relation and saved palm geometry differ")
    held = (inputs.pts >= relation["start_pts"]) & (inputs.pts <= relation["end_pts"])
    local = np.asarray(spec["segment_local_solver"]) * float(spec["length_export_ratio"])
    ends = (visual["origin_world_m"][held, foot, None]
            + np.einsum("nij,kj->nki", visual["rotation_world"][held, foot], local))
    delta = ends[:, 1] - ends[:, 0]
    palm = parent["grasp_palm_world_mid"][held]
    along = np.clip(np.sum((palm - ends[:, 0]) * delta, axis=-1) / np.sum(delta*delta, axis=-1), 0., 1.)
    gap = np.linalg.norm(palm - ends[:, 0] - along[:, None]*delta, axis=-1)
    result.append(dict(id="registered_s05_palm_to_segment", kind="held_source_interval",
                       start_pts=relation["start_pts"], end_pts=relation["end_pts"], foot=foot,
                       parent_p50_m=float(np.median(parent["palm_to_segment_m_mid"][held])),
                       visual_to_same_palm_p50_m=float(np.median(gap))))
    return result


def viewer_config(inputs, output):
    config = (inputs.recovery_path.with_name("viewer3d.json") if inputs.recovery_path else
              PROGRAMME / "viewer3d_programme_skate_chain_v1.json")
    base = read(config)
    # Display the saved worn variants and navigate the new source failures.
    # The independent shoe state does not replace the attached variant.
    base["session_id"] = "skate_video_v1_recovery_1112" if inputs.recovery_path else "skate_video_v1"
    base["source_navigation"] = str(output / "navigation.json")
    base["review_status"] = "鞋刀视频证据试验失败导航。三维窗口中为保存的穿戴对照；原片、旧版与独立候选三联画另见 PotPlayer。"
    base["initial_source_pts"] = int(inputs.pts[0])
    base["default_variant"] = "programme_skate_chain_v1"
    entry = output / "viewer3d.json"
    atomic_json(entry, base)
    runner = PROJECT / "launchers/START_3D_RESULTS.ps1"
    (output / "START_3D_SKATE_REVIEW.cmd").write_text(
        f'@echo off\npowershell.exe -NoProfile -ExecutionPolicy Bypass -File "{runner}" -Config "%~dp0viewer3d.json" %*\n',
        encoding="utf-8")
    from rinkspace.viewer3d.data import ResultCatalog
    catalog = ResultCatalog(project=PROJECT, data_root=ROOT, config=entry)
    np.testing.assert_array_equal(catalog.pts, inputs.pts)
    emit("viewer_navigation", path=str(entry), events=len(catalog.navigation_points("events")),
         cuts=len(catalog.navigation_points()))


def main(output):
    inputs = Inputs()
    shape, old_p, old_r = geometry(inputs)
    model = SkateModel(shape)
    native = Native(inputs)
    with np.load(output / "visual_shoes.npz", allow_pickle=False) as file:
        visual = {key: file[key] for key in file.files}
    np.testing.assert_array_equal(visual["source_pts"], inputs.pts)
    p, r = visual["origin_world_m"], visual["rotation_world"]
    needed = ("native22_world_mid", "P_world_mid", "Q_world_mid", "native_body_FK_world_R_mid",
              "skate_relative_R_mid", "skate_origin_world_mid", "hand_local_R_mid",
              "grasp_palm_world_mid", "com_acceleration_world_mps2", "palm_to_segment_m_mid")
    with np.load(inputs.parent_path, allow_pickle=False) as file:
        parent = {key: file[key] for key in needed}
    joints = parent["native22_world_mid"]
    axle = joints[:, ANKLE]
    offset = np.einsum("nfji,nfj->nfi", old_r, axle - old_p)
    fixed = np.median(offset, axis=0)
    np.testing.assert_allclose(offset, np.broadcast_to(fixed, offset.shape), atol=5e-8, rtol=0)
    new_ankle = p + np.einsum("nfij,fj->nfi", r, fixed)
    fixed_p = axle - np.einsum("nfij,fj->nfi", r, fixed)
    gap = np.linalg.norm(new_ankle - axle, axis=-1)
    excess = reach(joints, new_ankle)
    height = contact_values(p, r, shape)
    same_body_height = contact_values(fixed_p, r, shape)
    image_check = read(output / "image_checks.json")
    eligible = (visual["visual_factor_counts"] >= 10) & ~inputs.withheld[:, None]
    # Probe only existing source-supported frame groups. No physical objective
    # is optimized to compensate for missing or wrong source instances.
    sampled = np.array(sorted({i for (i, f), rows in inputs.fit.items() if eligible[i, f]
                              and any(r.get("evidence_status") == "observed" for r in rows)}), dtype=int)
    cavity, structural_failure = None, None
    if len(sampled):
        try:
            cavity, check_p = structural_probe(inputs, r, old_r, sampled, parent)
            np.testing.assert_allclose(check_p, fixed_p[sampled], atol=3e-8, rtol=0)
        except (ValueError, OSError, AssertionError) as error:
            structural_failure = f"{type(error).__name__}: {error}"
    bad = eligible & (excess > .02)
    rows = measure_checks(inputs, model, native, fixed_p, r, old_p, old_r)
    atomic_npz(output / "fusion_check.npz", source_pts=inputs.pts, time_base=inputs.tb,
               source_video_sha256=np.array(inputs.media_sha), camera_sha256=np.array(inputs.camera_sha),
               template_sha256=shape["template_sha256"], fixed_parent_ankle_world_m=axle,
               ankle_from_visual_world_m=new_ankle, ankle_pivot_gap_m=gap, fixed_hip_reach_excess_m=excess,
               visual_rail_lowest_z_m=height, same_body_pivot_rail_lowest_z_m=same_body_height,
               pivot_trial_origin_m=fixed_p, pivot_trial_rotation_world=r,
               source_qualified_for_probe=eligible, checked_cavity_pts=inputs.pts[sampled],
               cavity_parent_and_pivot_trial_m=cavity if cavity is not None else np.empty((2, 0, 2)),
               status=np.array("diagnostic_only_visual_and_body_incompatible_or_unverified"))
    report = dict(reach_impossible_foot_exposures=int(bad.sum()), probe_foot_exposures=int(eligible.sum()),
                  worst_excess_m=float(excess[eligible].max()) if eligible.any() else None,
                  gap_p50_p95_m=np.percentile(gap[eligible], [50, 95]).tolist() if eligible.any() else [],
                  cavity_p95_m=np.percentile(cavity, 95, axis=(1, 2)).tolist() if cavity is not None else None,
                  cavity_source_exposures=len(sampled) if cavity is not None else 0, cavity_failure=structural_failure,
                  role="source_gated attachment feasibility and same-body pivot trial",
                  body_joint_or_collision_optimised=False, candidate_accuracy_accepted=False,
                  image_checks=image_check["summary"], fixed_pivot_projection_checks=rows,
                  actions=describe_actions(inputs, visual, excess, height, parent))
    atomic_json(output / "fusion_check.json", report)
    nav = read(output / "navigation.json")
    nav["events"] = [event for event in nav["events"] if not event["id"].startswith("reach-")]
    for a, b in segments(bad.any(axis=1)):
        if b - a + 1 < 6:
            continue
        offset_index, foot = np.unravel_index(np.argmax(excess[a:b+1]), excess[a:b+1].shape)
        nav["events"].append(dict(id=f"reach-{a}", label=f"{'左' if foot == 0 else '右'}踝超出当前髋膝链可达范围",
                                  start_pts=int(inputs.pts[a]), end_pts=int(inputs.pts[b]),
                                  worst_pts=int(inputs.pts[a + offset_index]), exposure_count=int(b-a+1)))
    nav["events"].sort(key=lambda r: (r["start_pts"], r["end_pts"]))
    atomic_json(output / "navigation.json", nav)
    viewer_config(inputs, output)
    emit("fusion_feasibility", **{k: v for k, v in report.items()
                                  if k not in ("fixed_pivot_projection_checks", "actions", "image_checks")})


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
