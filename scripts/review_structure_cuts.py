"""Review original cut exposures and saved camera/body trajectories. No fitting."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from calibration.program_tracking.body_tracking.centroidal_physics_diagnostics import second_difference_operator
from calibration.program_tracking.body_tracking.skating_action_factors import (
    root_so3_diagnostics,
    validate_source_timeline,
)
from rinkspace.core.native import NativeCore
from rinkspace.domain.motion_chart import cubic_samples
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.path_policy import guard_project_write
from rinkspace.review.render import source_frames
from rinkspace.session.camera_diagnostics import inspect_shot_motion
from rinkspace.storage import AssetStore


PROJECT = Path(__file__).resolve().parents[1]
PROGRAMME = PROJECT / "events/ankara_2026/programmes/okada_mei_sp"
WINDOW_SECONDS = (0.12, 0.24, 0.40)
BODY_FIELDS = {"P_world_mid": (3,), "G_world_mid": (3,), "Q_world_mid": (3, 3),
               "skate_R_world_mid": (2, 3, 3)}
POSITION_FIELDS = {"P": "P_world_mid", "G": "G_world_mid", "COM": "com_world_mid"}
TILE_WIDTH, TILE_HEIGHT, SHEET_GAP, SHEET_TITLE, TILE_LABEL = 960, 540, 16, 60, 34
SHEET_SIZE = (2 * TILE_WIDTH + 3 * SHEET_GAP, SHEET_TITLE + 5 * (TILE_HEIGHT + TILE_LABEL + SHEET_GAP))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_camera(path, shots=None):
    with np.load(path, allow_pickle=False) as packet:
        deps = json.loads(packet["joint_numeric_dependencies_json"].item())
        timeline = ShotMotionTimeline(deps["shots"], deps["control_dt"], deps["first_control_time"])
        controls, rigs = packet["controls"], packet["rig"]
    if (controls.shape != (timeline.count, 3) or
            rigs.shape != (len(deps["cameras"]), 43) or
            not np.isfinite(controls).all() or not np.isfinite(rigs).all()):
        raise ValueError(f"Invalid camera layout in {path}")
    if shots is not None:
        keys = ("id", "first_pts", "last_pts", "time_base_num", "time_base_den")
        if [tuple(s[k] for k in keys) for s in timeline.shots] != [
            tuple(s[k] for k in keys) for s in shots
        ]:
            raise ValueError(f"Candidate source shot grid differs in {path}")
    return timeline, controls, rigs


def load_result(path, pts, time_base, shot_ids):
    with np.load(path, allow_pickle=False) as packet:
        if (not np.array_equal(packet["source_pts"], pts) or
                not np.array_equal(packet["time_base"], time_base) or
                not np.array_equal(packet["source_shot_ids"], shot_ids)):
            raise ValueError(f"Result source shot grid differs in {path}")
        fields = dict(BODY_FIELDS)
        if "com_world_mid" in packet:
            fields["com_world_mid"] = (3,)
        result = {key: packet[key] for key in fields}
        scale = float(packet["metres_per_scene_unit"])
    for key, shape in fields.items():
        if result[key].shape != (len(pts), *shape) or not np.isfinite(result[key]).all():
            raise ValueError(f"Invalid {key} in {path}")
    return result, scale


def source_sheet(path, source, pts, left, right, tick, output):
    width, height, gap, title, label = TILE_WIDTH, TILE_HEIGHT, SHEET_GAP, SHEET_TITLE, TILE_LABEL
    canvas = Image.new("RGB", SHEET_SIZE, "#111a24")
    draw = ImageDraw.Draw(canvas)
    headline, caption = ImageFont.load_default(22), ImageFont.load_default(17)
    cut_seconds = int(pts[5]) * tick
    draw.text((gap, 12), f"{left.upper()} to {right.upper()}    CUT {cut_seconds:.2f}s    |    ORIGINAL EXPOSURES",
              font=headline, fill="white")
    for index, (frame_pts, bgr) in enumerate(source_frames(path, source, pts)):
        side, row = divmod(index, 5)
        x, y = gap + side * (width + gap), title + row * (height + label + gap)
        frame = Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))
        canvas.paste(frame.resize((width, height), Image.Resampling.LANCZOS), (x, y))
        shot = (left, right)[side].upper()
        draw.text((x + 2, y + height + 7),
                  f"{shot}    {frame_pts * tick:.2f}s    PTS {frame_pts}",
                  font=caption, fill=("#a6d4ff", "#ffd7ae")[side])
    canvas.save(output)


def reuse_source_sheets(path, source, time_base, shots, shot_pts):
    review = read_json(path)
    cuts = review["cuts"]
    if (review["schema"] != "rinkspace.structure_cut_review.v1" or
            review["original_source_sha256"] != source["sha256"] or
            review["source_size"] != [source["width"], source["height"]] or
            review["time_base"] != list(time_base) or
            review["programme_exposures"] != sum(map(len, shot_pts)) or
            review["reviewed_exposures"] != 80 or len(cuts) != 8 or len(shots) != 9):
        raise ValueError(f"Source review identity or coverage differs in {path}")
    sheets = []
    for i, cut in enumerate(cuts):
        left, right = shot_pts[i][-5:].tolist(), shot_pts[i + 1][:5].tolist()
        if ((cut["left"], cut["right"]) != (shots[i]["id"], shots[i + 1]["id"]) or
                cut["cut_pts"] != right[0] or cut["left_pts"] != left or cut["right_pts"] != right or
                cut["native_pts_step"] != source["native_pts_step"]):
            raise ValueError(f"Cut {i + 1} source grid differs in {path}")
        sheet = Path(cut["source_sheet"])
        if not sheet.is_absolute():
            raise ValueError(f"Cut {i + 1} has no absolute source sheet path in {path}")
        sheet = sheet.resolve(strict=True)
        if sheet in sheets:
            raise ValueError(f"Source sheet is reused for two different cuts in {path}")
        with Image.open(sheet) as image:
            if image.format != "PNG" or image.size != SHEET_SIZE:
                raise ValueError(f"Unexpected source sheet format or dimensions in {sheet}")
            image.verify()
        sheets.append(sheet)
    return sheets


def reuse_background(path, source_review, sheets, source, time_base, shots, shot_pts, camera_hashes):
    report = read_json(path)
    cuts = report["cuts"]
    if (report["schema"] != "rinkspace.fixed_station_cut_background.v1" or
            report["source_sha256"] != source["sha256"] or
            report["source_size"] != [source["width"], source["height"]] or
            report["source_time_base"] != list(time_base) or
            report["source_cut_review"]["sha256"] != sha256(source_review) or
            len(cuts) != len(sheets)):
        raise ValueError(f"Background review source identity or coverage differs in {path}")

    # Match saved projections by camera identity even when a candidate uses another case name.
    cases_by_hash = {value["sha256"]: name for name, value in report["cameras"].items()}
    evidence = []
    for i, (cut, sheet) in enumerate(zip(cuts, sheets, strict=True)):
        left, right = shot_pts[i][-5:].tolist(), shot_pts[i + 1][:5].tolist()
        if ((cut["left"], cut["right"]) != (shots[i]["id"], shots[i + 1]["id"]) or
                cut["cut_pts"] != right[0] or cut["left_pts"] != left or cut["right_pts"] != right or
                cut["base_contact_sheet_sha256"] != sha256(sheet)):
            raise ValueError(f"Cut {i + 1} background evidence differs in {path}")
        by_camera = {}
        for name, digest in camera_hashes.items():
            case = cases_by_hash.get(digest)
            if case is None:
                by_camera[name] = dict(available=False, reason="No saved background for this camera SHA256")
                continue
            details = cut["cameras"][case]
            figure = Path(details["figure"])
            if not figure.is_absolute():
                raise ValueError(f"Cut {i + 1} has no absolute background figure path in {path}")
            summary = {key: details[key] for key in ("rows", "valid", "invalid", "rms_px", "p95_px")}
            by_camera[name] = dict(available=True, evidence_case=case,
                                   figure=str(figure.resolve(strict=True)), **summary)
        evidence.append(by_camera)
    return evidence


def camera_window(core, camera, shot, pts, tick):
    timeline, controls, rigs = camera
    time = timeline.exposure_time(shot, pts)
    rig = rigs[shot["camera_index"]]
    report, exposures, _ = inspect_shot_motion(
        core, rig, controls, timeline.first, timeline.dt, time, pts * tick,
        source_mapper=lambda values: timeline.source_seconds_at(shot, values),
    )
    bounds = report["native_exposure_interval"]
    dense = np.sort(np.unique(np.r_[np.linspace(*bounds, 101), time]))
    indices, weights = cubic_samples(timeline.count, timeline.first, timeline.dt, dense)
    if indices.min() < shot["control_begin"] or indices.max() >= shot["control_stop"]:
        raise ValueError(f"Readout exceeds independent controls for {shot['id']}")
    state = core.evaluate_motion_kinematics(controls, timeline.first, timeline.dt, dense)
    sampled = np.einsum("ni,nij->nj", weights, controls[indices])
    if np.any(state["status"]) or not np.allclose(sampled, state["position"], rtol=0, atol=1e-10):
        raise ValueError(f"Invalid camera derivative window for {shot['id']}")
    to_units = np.array([180 / np.pi, 180 / np.pi, 1.0])
    return dict(
        shot=shot["id"], camera_index=shot["camera_index"], readout_seconds=float(rig[37]),
        source_pts=pts.tolist(), sample_seconds=timeline.source_seconds_at(shot, dense).tolist(),
        position_rate_acceleration_jerk=(state["values"] * to_units).tolist(),
        exposure_samples=(exposures["kinematics"] * to_units).tolist(),
        analytic_motion=report,
    )


def position_window(positions, full_pts, window, cut_pts, side_bounds, tick):
    secs = (full_pts[window] - cut_pts) * tick
    xyz = positions[window]
    dt = np.diff(secs)
    velocity = np.diff(xyz, axis=0) / dt[:, None]
    left_v, bridge_v, right_v = velocity[3:6]
    bridge_residual = (bridge_v - 0.5 * (left_v + right_v)) * dt[4]
    # Use raw three-exposure differences. Keep the two crossing stencils separate.
    acceleration = second_difference_operator(secs) @ (xyz - xyz[0])
    within_shot = [0, 1, 2, 5, 6, 7]
    crossing = acceleration[3:5]
    stencil_pts = np.lib.stride_tricks.sliding_window_view(full_pts[window], 3)
    # Match the offline audit: fit each side relative to its nearest
    # exposure and compare the quadratics at the midpoint of the cut interval.
    source_seconds = full_pts / (1.0 / tick)
    left_time, right_time = source_seconds[window[4:6]]
    midpoint = 0.5 * (left_time + right_time)
    quadratic = {}
    for duration in WINDOW_SECONDS:
        predicted = []
        for (lower, upper), (first, last) in zip(
            side_bounds, ((left_time - duration, left_time), (right_time, right_time + duration)),
            strict=True,
        ):
            selected = ((full_pts >= lower) & (full_pts < upper) &
                        (source_seconds >= first) & (source_seconds <= last))
            local_time = source_seconds[selected] - midpoint
            coeffs = np.polynomial.polynomial.polyfit(local_time, positions[selected], 2)
            predicted.append(dict(exposures=len(local_time), source_pts=full_pts[selected].tolist(),
                                  position_m=coeffs[0].tolist(), velocity_mps=coeffs[1].tolist(),
                                  acceleration_mps2=(2 * coeffs[2]).tolist()))
        delta_p = np.subtract(predicted[1]["position_m"], predicted[0]["position_m"])
        delta_v = np.subtract(predicted[1]["velocity_mps"], predicted[0]["velocity_mps"])
        delta_a = np.subtract(predicted[1]["acceleration_mps2"], predicted[0]["acceleration_mps2"])
        quadratic[str(duration)] = dict(
            left=predicted[0], right=predicted[1], delta_position_m=delta_p.tolist(),
            delta_velocity_mps=delta_v.tolist(), delta_acceleration_mps2=delta_a.tolist(),
            extrapolated_position_difference_m=float(np.linalg.norm(delta_p)),
            extrapolated_velocity_difference_mps=float(np.linalg.norm(delta_v)),
            extrapolated_acceleration_difference_mps2=float(np.linalg.norm(delta_a)),
        )
    return dict(
        available=True, positions_m=xyz.tolist(), interval_velocity_mps=velocity.tolist(),
        interval_speed_mps=np.linalg.norm(velocity, axis=-1).tolist(),
        acceleration_source_pts=stencil_pts[within_shot, 1].tolist(),
        acceleration_mps2=acceleration[within_shot].tolist(),
        acceleration_magnitude_mps2=np.linalg.norm(acceleration[within_shot], axis=-1).tolist(),
        cross_cut_acceleration_stencil_pts=stencil_pts[3:5].tolist(),
        cross_cut_acceleration_mps2=crossing.tolist(),
        cross_cut_acceleration_magnitude_mps2=np.linalg.norm(crossing, axis=-1).tolist(),
        cut_adjacent_acceleration_change_mps2=float(np.linalg.norm(acceleration[5] - acceleration[2])),
        cut_position_delta_m=(xyz[5] - xyz[4]).tolist(),
        cut_position_step_m=float(np.linalg.norm(xyz[5] - xyz[4])),
        cross_cut_secant_mps=bridge_v.tolist(),
        cut_adjacent_velocity_change_mps=float(np.linalg.norm(right_v - left_v)),
        cut_local_kinematic_innovation_m=float(np.linalg.norm(bridge_residual)),
        one_sided_quadratic=quadratic, one_sided_evaluation_seconds=float(midpoint),
    )


def body_window(result, full_pts, cut_pts, window, side_bounds, time_base, source_id):
    clock = validate_source_timeline(full_pts[window], time_base, source_id=source_id)
    tick = time_base[0] / time_base[1]
    dt = np.diff(clock.elapsed_s)
    rotations = {"root": result["Q_world_mid"][window],
                 "left skate": result["skate_R_world_mid"][window, 0],
                 "right skate": result["skate_R_world_mid"][window, 1]}
    rotation_metrics = {}
    for name, values in rotations.items():
        steps = root_so3_diagnostics(clock, values)
        angles = np.rad2deg(steps.principal_increment_rad)
        rotation_metrics[name] = dict(
            interval_step_deg=angles.tolist(), interval_speed_degps=(angles / dt).tolist(),
            near_pi_ambiguous=steps.near_pi_ambiguous.tolist(),
            cut_step_deg=float(angles[4]), cut_speed_degps=float(angles[4] / dt[4]),
        )
    positions = {
        name: position_window(result[key], full_pts, window, cut_pts, side_bounds, tick)
        if key in result else dict(available=False, reason=f"No saved {key} in this result")
        for name, key in POSITION_FIELDS.items()
    }
    return dict(
        interval_source_pts=steps.velocity_interval_pts.tolist(), interval_dt_seconds=dt.tolist(),
        cross_cut_interval_index=4, **positions, rotations=rotation_metrics,
    )


def plot_window(cut, output):
    figure, axes = plt.subplots(8, 3, figsize=(14, 17), layout="constrained")
    for order in (4, 5, 7):
        for channel in (1, 2):
            axes[order, channel].sharey(axes[order, 0])
    cut_seconds = cut["source_seconds"]
    figure.suptitle(
        f"{cut['left'].upper()} to {cut['right'].upper()} at {cut_seconds:.2f}s   |   "
        "independent shot cameras and saved conditional world trajectories",
        fontsize=15,
    )
    for index, (_, sides) in enumerate(cut["cameras"].items()):
        color = f"C{index % 10}"
        for side in ("left", "right"):
            record = sides[side]
            x = np.asarray(record["sample_seconds"]) - cut_seconds
            y = np.asarray(record["position_rate_acceleration_jerk"])
            style = "-" if side == "left" else "--"
            for order in range(4):
                for channel in range(3):
                    axes[order, channel].plot(x, y[:, order, channel], style, color=color)
    for order, derivative in enumerate(("value", "speed", "acceleration", "jerk")):
        for channel, name in enumerate(("Pan", "Tilt", "Log focal length")):
            ax = axes[order, channel]
            ax.set_title(f"{name} {derivative}", loc="left")
            unit = "deg" if channel < 2 else "log px"
            suffix = ("", "/s", "/s²", "/s³")[order]
            ax.set_ylabel(unit + suffix)
            ax.axvline(0, color="#a1a1aa", lw=0.9)
            ax.grid(alpha=0.22)
            if order == 3:
                ax.set_xlabel("Seconds from first exposure of right shot")
    handles = [Line2D([0], [0], color=f"C{i % 10}", label=name)
               for i, name in enumerate(cut["cameras"])]
    handles += [Line2D([0], [0], color="black", ls=ls, label=side)
                for ls, side in (("-", cut["left"]), ("--", cut["right"]))]
    axes[0, 0].legend(handles=handles, fontsize=8, ncol=2)
    tick = cut["source_tick_seconds"]
    com_missing = []
    for index, (name, body) in enumerate(cut["results"].items()):
        color = f"C{index % 10}"
        x = (np.mean(body["interval_source_pts"], axis=1) - cut["cut_pts"]) * tick
        bridge = body["cross_cut_interval_index"]
        for channel, field in enumerate(POSITION_FIELDS):
            row = body[field]
            if not row["available"]:
                com_missing.append(name)
                continue
            speed = row["interval_speed_mps"]
            axes[4, channel].plot(x[:bridge], speed[:bridge], "o-", color=color, label=name, ms=3)
            axes[4, channel].plot(x[bridge + 1:], speed[bridge + 1:], "o--", color=color, ms=3)
            axes[4, channel].plot(x[bridge], speed[bridge], "x", color=color)
            ax = axes[5, channel]
            at = (np.asarray(row["acceleration_source_pts"]) - cut["cut_pts"]) * tick
            magnitude = row["acceleration_magnitude_mps2"]
            ax.plot(at[:3], magnitude[:3], "o-", color=color, ms=3)
            ax.plot(at[3:], magnitude[3:], "o--", color=color, ms=3)
            crossing_pts = np.asarray(row["cross_cut_acceleration_stencil_pts"])[:, 1]
            ax.plot((crossing_pts - cut["cut_pts"]) * tick,
                    row["cross_cut_acceleration_magnitude_mps2"], "x", color=color)
            gaps = [row["one_sided_quadratic"][str(s)]["extrapolated_position_difference_m"]
                    for s in WINDOW_SECONDS]
            axes[7, channel].plot(WINDOW_SECONDS, gaps, "o-", color=color, ms=3)
        for channel, record in enumerate(body["rotations"].values()):
            axes[6, channel].plot(x, record["interval_step_deg"], "o-", color=color, ms=3, label=name)
    for channel, label in enumerate(("P (pelvis)", "G (ground projection)", "Model COM")):
        for row, metric, unit in ((4, "speed", "m/s"), (5, "acceleration magnitude", "m/s²"),
                                  (7, "one-sided position gap", "m")):
            ax = axes[row, channel]
            ax.set_title(f"{label} {metric}", loc="left")
            ax.set_ylabel(unit)
    if com_missing:
        axes[4, 2].text(.03, .97, "Unavailable: " + ", ".join(com_missing),
                        va="top", transform=axes[4, 2].transAxes, fontsize=8)
    for channel, name in enumerate(("Root rotation step", "Left skate rotation step", "Right skate rotation step")):
        ax = axes[6, channel]
        ax.set_title(name, loc="left")
        ax.set_ylabel("deg")
    for row, label in enumerate(("Interval midpoint (s from cut)", "Exposure (s from cut)",
                                 "Interval midpoint (s from cut)", "One-sided fit window (s)"), start=4):
        for ax in axes[row]:
            ax.grid(alpha=0.22)
            if row < 7:
                ax.axvspan(-tick * cut["native_pts_step"], 0, color="#f3c690", alpha=0.27)
            ax.set_xlabel(label)
    handles, _ = axes[4, 0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="x", ls="", color="black", label="Cross-cut secant"))
    axes[4, 0].legend(handles=handles, fontsize=8)
    axes[5, 0].legend(handles=[Line2D([0], [0], marker="x", ls="", color="black",
                                      label="Cross-cut stencil")], fontsize=8)
    figure.savefig(output, dpi=120)
    plt.close(figure)


def named_path(value):
    name, sep, path = value.partition("=")
    if not sep or not name.strip() or not path.strip() or name == "baseline":
        raise argparse.ArgumentTypeError("Use a unique NAME=PATH; baseline is reserved")
    return name, Path(path).resolve(strict=True)


def add_cases(baseline, extra):
    cases = {"baseline": baseline}
    for name, path in extra:
        if name in cases:
            raise ValueError(f"Duplicate case name {name}")
        cases[name] = path
    return cases


def main():
    store = AssetStore(PROJECT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=store.data /
                        "runs/body_s01_20261029/structure_camera_1108/cut_review")
    parser.add_argument("--result", type=named_path, action="append", default=[], metavar="NAME=NPZ")
    parser.add_argument("--baseline", type=Path, metavar="RESULT_NPZ",
                        help="Override the frozen baseline body result, keeping the frozen camera")
    for case in ("structure", "joint"):
        parser.add_argument(f"--{case}", type=Path, metavar="RESULT_NPZ",
                            help=f"Add a saved body result under the {case} case name")
    parser.add_argument("--camera", type=named_path, action="append", default=[], metavar="NAME=NPZ")
    parser.add_argument("--library", type=Path, help="Explicit ABI-compatible native diagnostic DLL")
    parser.add_argument("--source-review", type=Path, metavar="CUT_REVIEW_JSON",
                        help="Verify and reference existing original-exposure sheets without decoding new frames")
    parser.add_argument("--background-review", type=Path, metavar="CUT_BACKGROUND_JSON",
                        help="Reference existing static-background evidence for the supplied source review")
    args = parser.parse_args()
    if args.background_review is not None and args.source_review is None:
        parser.error("--background-review requires --source-review")

    out = guard_project_write(args.out)
    if not out.is_relative_to(store.data.resolve()):
        raise ValueError("Cut review output belongs in the registered Data root")
    if out.exists():
        raise FileExistsError(f"Preserve existing review output: {out}")
    selection = read_json(PROGRAMME / "primary_source_selection.json")["primary_source"]
    video = store.source(selection["path"]).resolve(strict=True)
    if sha256(video) != selection["sha256"]:
        raise ValueError("Source video SHA256 differs from the registered original")
    time_base = Fraction(selection["time_base"])
    time_base_pair = (time_base.numerator, time_base.denominator)
    tick, step = float(time_base), selection["native_pts_step"]
    source = dict(selection, time_base_num=time_base.numerator, time_base_den=time_base.denominator)

    camera_record = read_json(PROGRAMME / "fit_v11/camera_solver_refined_1102.json")
    result_record = read_json(PROGRAMME / "programme_action_finish_1107.json")
    library = args.library or Path(camera_record["native_library"]["path"])
    core = NativeCore(library)
    camera_paths = add_cases(store.source(camera_record["state"]["compact_path"]), args.camera)
    baseline_result = (args.baseline.resolve(strict=True) if args.baseline is not None
                       else store.source(result_record["result"]))
    extra_results = args.result + [(name, path.resolve(strict=True))
                                   for name, path in (("structure", args.structure), ("joint", args.joint))
                                   if path is not None]
    result_paths = add_cases(baseline_result, extra_results)
    camera_hashes = {name: sha256(path) for name, path in camera_paths.items()}
    result_hashes = {name: sha256(path) for name, path in result_paths.items()}
    library_hash = sha256(core.library_path)
    if (camera_hashes["baseline"] != camera_record["state"]["compact_sha256"] or
            (args.baseline is None and result_hashes["baseline"] != result_record["result_sha256"]) or
            (args.library is None and library_hash != camera_record["native_library"]["sha256"])):
        raise ValueError("Baseline input differs from its frozen record")
    cameras = {"baseline": load_camera(camera_paths["baseline"])}
    shots = cameras["baseline"][0].shots
    shot_pts = [np.arange(s["first_pts"], s["last_pts"] + 1, step, dtype=np.int64) for s in shots]
    full_pts = np.concatenate(shot_pts)
    ids = np.repeat([s["id"] for s in shots], [len(p) for p in shot_pts])
    if (np.any(np.diff(full_pts) != step) or len(shots) != 9 or
            len(full_pts) != camera_record["designation"]["source_exposures"] or
            any((s["time_base_num"], s["time_base_den"]) != time_base_pair for s in shots)):
        raise ValueError("Baseline camera does not cover the registered nine-shot source grid")
    source_review = args.source_review.resolve(strict=True) if args.source_review is not None else None
    source_sheets = reuse_source_sheets(source_review, source, time_base_pair, shots, shot_pts) if source_review else None
    cameras.update({name: load_camera(path, shots) for name, path in camera_paths.items() if name != "baseline"})
    results = {name: load_result(path, full_pts, time_base_pair, ids) for name, path in result_paths.items()}
    background_review = args.background_review.resolve(strict=True) if args.background_review is not None else None
    background = reuse_background(
        background_review, source_review, source_sheets, source, time_base_pair, shots, shot_pts, camera_hashes
    ) if background_review is not None else None

    out.mkdir(parents=True, exist_ok=False)
    cuts = []
    for index, (left_pts, right_pts) in enumerate(zip(shot_pts[:-1], shot_pts[1:], strict=True)):
        pts = np.r_[left_pts[-5:], right_pts[:5]]
        cut_pts = int(right_pts[0])
        window = np.searchsorted(full_pts, pts)
        sides = ((int(left_pts[0]), cut_pts), (cut_pts, int(right_pts[-1]) + step))
        left, right = shots[index]["id"], shots[index + 1]["id"]
        cut_seconds = cut_pts * tick
        stem = out / f"cut_{index + 1:02}_{cut_seconds:.2f}"
        source_path = source_sheets[index] if source_sheets is not None else stem.with_name(stem.name + "_source.png")
        motion_path = stem.with_name(stem.name + "_motion.png")
        cut = dict(
            left=left, right=right, source_seconds=cut_seconds, cut_pts=cut_pts,
            left_pts=pts[:5].tolist(), right_pts=pts[5:].tolist(),
            source_tick_seconds=tick, native_pts_step=step, source_sheet=str(source_path),
            motion_figure=str(motion_path), cameras={}, results={},
        )
        for name, camera in cameras.items():
            camera_shots = camera[0].shots
            cut["cameras"][name] = dict(
                left=camera_window(core, camera, camera_shots[index], pts[:5], tick),
                right=camera_window(core, camera, camera_shots[index + 1], pts[5:], tick),
                physical_camera_change=camera_shots[index]["camera_index"] != camera_shots[index + 1]["camera_index"],
            )
        cut["results"] = {
            name: body_window(result, full_pts, cut_pts, window, sides, time_base_pair, selection["id"])
            for name, (result, _) in results.items()
        }
        if source_sheets is None:
            source_sheet(video, source, pts, left, right, tick, source_path)
        if background is not None:
            cut["static_background"] = background[index]
        plot_window(cut, motion_path)
        cuts.append(cut)
        print(f"{left} {right} {cut_seconds:.2f}s", flush=True)
    report = dict(
        schema="rinkspace.structure_cut_review.v1", original_source_path=str(video),
        original_source_sha256=selection["sha256"], source_size=[source["width"], source["height"]],
        time_base=list(time_base_pair), reviewed_exposures=80, programme_exposures=len(full_pts),
        cut_camera_continuity_enforced=False, one_sided_body_fit_windows_seconds=list(WINDOW_SECONDS),
        camera_kinematic_channels=["pan_deg", "tilt_deg", "log_f_px"],
        camera_kinematic_derivative_orders=[0, 1, 2, 3],
        native_library={"path": str(core.library_path), "sha256": library_hash},
        cameras={name: dict(path=str(path), sha256=camera_hashes[name]) for name, path in camera_paths.items()},
        results={name: dict(path=str(path), sha256=result_hashes[name], metres_per_scene_unit=results[name][1],
                            model_com_available="com_world_mid" in results[name][0])
                 for name, path in result_paths.items()},
        notes=[
            "The first frame of the right shot labels the cut. Each camera curve includes only its own shot readout.",
            "P is the saved pelvis and G its ground projection. COM is the saved body and equipment mass model, when present.",
            "Positions are already in conditional world metres. Derivatives use actual source PTS.",
            "The centre position secant crosses the cut and is shown separately from within-shot velocities.",
            "Acceleration uses raw three-exposure differences. The two body stencils crossing the cut are reported and plotted separately.",
            "The quadratic fits use each side separately and evaluate at the cut interval midpoint. Their differences are model diagnostics.",
            "Rotation steps are principal SO(3) angles, and the centre interval spans the cut.",
            "The saved-state diagnostics and original image sheets do not provide independent 3D accuracy.",
        ], cuts=cuts,
    )
    if source_review is not None:
        report["reused_source_review"] = dict(path=str(source_review), sha256=sha256(source_review))
    if background_review is not None:
        report["reused_static_background_review"] = dict(path=str(background_review), sha256=sha256(background_review))
        report["notes"].append("Saved static residuals refer to fixed stations and are associated by camera SHA256.")
    target = out / "cut_review.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(str(target), flush=True)


if __name__ == "__main__":
    main()
