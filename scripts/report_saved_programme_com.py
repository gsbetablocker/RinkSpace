"""Compare archived model COM paths on a registered native programme clock."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from calibration.program_tracking.body_tracking.body_clock import time_base
from calibration.program_tracking.body_tracking.programme_com import acceleration_operator
from calibration.program_tracking.body_tracking.programme_contract import required_clock
from calibration.program_tracking.body_tracking.rebuild_prepare import digest


LSP_SECONDS = (156.2, 165.3)


def receipt(path):
    return {"path": str(path.resolve()), "sha256": digest(path)}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def candidate_arg(value):
    name, sep, path = value.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError("Use LABEL=RESULT.npz")
    return name, Path(path)


def kinematics(positions, pts, tb, step):
    """Differentiate each consecutive native-clock run with no gap bridging."""
    t = (pts-pts[0]) * tb[0] / tb[1]
    ranges = np.r_[0, np.flatnonzero(np.diff(pts) != step)+1, len(pts)]
    velocity = np.full_like(positions, np.nan)
    acceleration = np.full_like(positions, np.nan)
    for begin, end in zip(ranges[:-1], ranges[1:], strict=True):
        if end - begin >= 2:
            order = 2 if end - begin >= 3 else 1
            velocity[begin:end] = np.gradient(positions[begin:end], t[begin:end], axis=0, edge_order=order)
        if end - begin >= 3:
            acceleration[begin+1:end-1] = acceleration_operator(t[begin:end]) @ positions[begin:end]
    return velocity, acceleration


def load_candidate(path, expected_pts, tb, step):
    with np.load(path, allow_pickle=False) as result:
        present_clocks = [name for name in ("source_pts", "pts") if name in result]
        if not present_clocks or time_base(result) != tb:
            raise ValueError(f"Missing or different native source clock: {path}")
        for name in present_clocks:
            clock = result[name]
            if clock.dtype.kind not in "iu" or not np.array_equal(clock, expected_pts):
                raise ValueError(f"Expected the complete registered native exposures in {name}: {path}")
        positions = np.asarray(result["com_world_mid"], dtype=float)
        if positions.shape != (len(expected_pts), 3) or not np.isfinite(positions).all():
            raise ValueError(f"Invalid saved model COM positions: {path}")
        velocity, acc = kinematics(positions, expected_pts, tb, step)
        info = {"input": receipt(path), "position_key": "com_world_mid"}
        for name in ("com_mass_model", "com_derivative_method", "metres_per_scene_unit", "body_length_ratio"):
            if name in result:
                info[name] = result[name].item()
        info.setdefault("com_mass_model", "Older result has saved COM positions without detailed mass-model metadata.")
        for name, values in (("com_velocity_world_mps", velocity), ("com_acceleration_world_mps2", acc)):
            if name in result:
                actual = result[name]
                if actual.shape != values.shape:
                    raise ValueError(f"Saved derivative shape differs from COM positions: {path}")
                valid = np.isfinite(values).all(axis=1)
                if not np.isfinite(actual[valid]).all():
                    raise ValueError(f"Saved derivative is nonfinite on usable rows: {path}")
                info[name + "_max_difference"] = float(np.max(np.abs(actual[valid]-values[valid])))
    return dict(position=positions, velocity=velocity, acceleration=acc, identity=info)


def distribution(values):
    values = np.asarray(values)
    good = values[np.isfinite(values)]
    if not len(good):
        return {"samples": 0}
    return dict(samples=len(good), **dict(zip(("min", "p50", "p95", "max"),
                                             np.percentile(good, [0, 50, 95, 100]).tolist(), strict=True)))


def peak_indices(magnitude, mask, count):
    ids = np.flatnonzero(mask & np.isfinite(magnitude))
    runs = np.split(ids, np.flatnonzero(np.diff(ids) != 1)+1)
    peaks = []
    for run in runs:
        if len(run):
            v = magnitude[run]
            # Earliest representative for equal values on a peak plateau.
            peaks.extend(run[(v > np.r_[-np.inf, v[:-1]]) & (v >= np.r_[v[1:], -np.inf])])
    return sorted(peaks, key=lambda i: (-magnitude[i], i))[:count]


def peak_row(i, values, pts, seconds, shots, cuts, episodes, opening):
    cut = int(cuts[np.argmin(np.abs(pts[i]-pts[cuts]))]) if len(cuts) else None
    membership = [dict(id=e["episode_id"], kind=e["kind"]) for e in episodes
                  if e["start_pts"] <= pts[i] <= e["end_pts_inclusive"]]
    distance = float(seconds[i]-seconds[cut]) if cut is not None else None
    phase = None
    if opening['start_pts'] <= pts[i] <= opening['hand_release_end_pts']:
        phase = 'opening_release_window'
        if pts[i] <= opening['quiet_end_pts']:
            phase = 'opening_quiet'
        elif pts[i] <= opening['hand_release_start_pts']:
            phase = 'opening_pre_release'
    return dict(
        source_pts=int(pts[i]), source_seconds=float(seconds[i]), shot=str(shots[i]),
        acceleration_magnitude_mps2=float(np.linalg.norm(values[i])), acceleration_xyz_mps2=values[i].tolist(),
        episodes=membership, programme_opening_phase=phase, nearest_cut_first_new_shot_pts=None if cut is None else int(pts[cut]),
        nearest_cut_seconds=None if cut is None else float(seconds[cut]),
        signed_seconds_from_nearest_cut=distance,
        abs_seconds_from_nearest_cut=None if distance is None else abs(distance),
        acceleration_stencil_crosses_cut=bool(0 < i < len(shots)-1 and shots[i-1] != shots[i+1]),
    )


def draw(path, candidates, seconds, cuts, mask, title, bands):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    colors = ("#51617d", "#007a79", "#ac5941", "#73589e")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), layout="constrained")
    xy, z, speed, acc = axes.ravel()
    ids = np.flatnonzero(mask)
    start, end = seconds[ids[[0, -1]]]
    for n, (name, data) in enumerate(candidates.items()):
        color = colors[n % len(colors)]
        width = 1.25 if n else 1.
        values = data["position"][ids]
        xy.plot(values[:, 0], values[:, 1], color=color, lw=width, label=name)
        xy.scatter(values[[0, -1], 0], values[[0, -1], 1], color=color, s=12)
        z.plot(seconds[ids], values[:, 2], color=color, lw=width)
        speed.plot(seconds[ids], np.linalg.norm(data["velocity"][ids], axis=1), color=color, lw=width)
        acc.plot(seconds[ids], np.linalg.norm(data["acceleration"][ids], axis=1), color=color, lw=width)
        visible_cuts = cuts[mask[cuts]]
        positions = data["position"][visible_cuts]
        xy.scatter(positions[:, 0], positions[:, 1], s=14, marker="x", color=color, zorder=3)
    xy_title = "COM path in the saved world frame"
    if np.any(mask[cuts]):
        xy_title += " (× camera cut)"
    xy.set(xlabel="World X (m)", ylabel="World Y (m)", title=xy_title)
    xy.set_aspect("equal", adjustable="datalim")
    xy.legend(loc="best", frameon=False)
    for ax, ylabel in ((z, "World Z (m)"), (speed, "Speed (m/s)"), (acc, "Acceleration (m/s²)")):
        ax.set(xlim=(start, end), xlabel="Source time (s)", ylabel=ylabel)
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        for first, last in bands:
            if first < end and last > start:
                ax.axvspan(first, last, facecolor="#b4c7ca", alpha=.21, lw=0, zorder=-1)
        for boundary in cuts[mask[cuts]]:
            ax.axvline(seconds[boundary], color="#8c7460", alpha=.55, lw=.8, ls="--")
    if np.nanmax([np.nanmax(np.linalg.norm(d["acceleration"][ids], axis=1)) for d in candidates.values()]) > 120.:
        acc.set_yscale("symlog", linthresh=10.)
        acc.set_ylabel("Acceleration (m/s², symlog above 10)")
    acc.set_ylim(bottom=0.)
    for ax in axes.ravel():
        ax.grid(alpha=.18, lw=.65)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title + "  |  saved model COM proxy", fontweight="bold")
    fig.savefig(path, dpi=155, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=candidate_arg, action="append", required=True, metavar="LABEL=RESULT.npz")
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--source-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--peaks", type=int, default=8)
    args = parser.parse_args()
    if args.peaks < 1 or len(dict(args.candidate)) != len(args.candidate):
        parser.error("Use distinct candidate labels and a positive peak count")
    scope = read_json(args.scope)
    evidence = read_json(args.source_evidence)
    pts, shots, tb = required_clock(scope)
    step = int(scope["source_clock"]["native_pts_step"])
    seconds = pts*tb[0]/tb[1]
    cuts = np.flatnonzero(shots[1:] != shots[:-1])+1
    episodes = evidence["source_episode_review"]["episodes"]
    opening = evidence["programme_opening"]
    start, quiet_end, opening_end = (int(opening[key]) for key in ("start_pts", "quiet_end_pts", "hand_release_end_pts"))
    windows = {
        "whole": (seconds[0], seconds[-1]),
        "opening": (start*tb[0]/tb[1], opening_end*tb[0]/tb[1]),
        "LSp": LSP_SECONDS,
    }
    masks = {name: (seconds >= left-1e-9) & (seconds <= right+1e-9)
             for name, (left, right) in windows.items()}
    masks["opening_quiet"] = (pts >= start) & (pts <= quiet_end)
    candidates = {name: load_candidate(path, pts, tb, step) for name, path in args.candidate}
    args.output.mkdir(parents=True, exist_ok=False)

    results, csv_rows = {}, []
    for label, data in candidates.items():
        speed = np.linalg.norm(data["velocity"], axis=1)
        acceleration = np.linalg.norm(data["acceleration"], axis=1)
        results[label] = dict(data["identity"], windows={})
        for window, mask in masks.items():
            peaks = [peak_row(i, data["acceleration"], pts, seconds, shots, cuts, episodes, opening)
                     for i in peak_indices(acceleration, mask, args.peaks)]
            selection = data["position"][mask]
            results[label]["windows"][window] = dict(
                source_exposures=int(mask.sum()), source_pts_first_last=pts[mask][[0, -1]].tolist(),
                xyz_range_world_m=np.stack((selection.min(axis=0), selection.max(axis=0))).tolist(),
                speed_mps=distribution(speed[mask]), acceleration_mps2=distribution(acceleration[mask]), peaks=peaks,
            )
            for rank, row in enumerate(peaks, 1):
                csv_rows.append(dict(candidate=label, window=window, rank=rank, source_pts=row["source_pts"],
                    source_seconds=row["source_seconds"], shot=row["shot"],
                    acceleration_mps2=row["acceleration_magnitude_mps2"],
                    ax_mps2=row["acceleration_xyz_mps2"][0], ay_mps2=row["acceleration_xyz_mps2"][1],
                    az_mps2=row["acceleration_xyz_mps2"][2], episodes=";".join(e["id"] for e in row["episodes"]),
                    opening_phase=row["programme_opening_phase"],
                    nearest_cut_pts=row["nearest_cut_first_new_shot_pts"],
                    signed_cut_distance_s=row["signed_seconds_from_nearest_cut"],
                    abs_cut_distance_s=row["abs_seconds_from_nearest_cut"],
                    stencil_crosses_cut=row["acceleration_stencil_crosses_cut"]))

    plots = {
        "whole": ("whole.png", "Whole programme · opening and LSp shaded", [windows["opening"], windows["LSp"]]),
        "opening": ("opening.png", "Opening · quiet window shaded", [(windows["opening"][0], quiet_end*tb[0]/tb[1])]),
        "LSp": ("lsp.png", "LSp · 162.60–162.80 s shaded", [(162.60, 162.80)]),
    }
    for key, (file, title, bands) in plots.items():
        draw(args.output/file, candidates, seconds, cuts, masks[key], title, bands)
    with (args.output/"peaks.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    report = dict(
        schema="rinkspace.saved-programme-com-comparison.v1", source_frames=len(pts), source_pts_step=step,
        source_pts_first_last=pts[[0, -1]].tolist(), source_time_base=list(tb),
        source_clock_gap_count=int(np.count_nonzero(np.diff(pts) != step)),
        scope=receipt(args.scope), source_evidence=receipt(args.source_evidence),
        cuts=[dict(from_shot=str(shots[i-1]), to_shot=str(shots[i]), first_new_shot_pts=int(pts[i]),
                   source_seconds=float(seconds[i])) for i in cuts],
        windows_source_seconds={**windows, "opening_quiet": (start*tb[0]/tb[1], quiet_end*tb[0]/tb[1])},
        definitions=dict(
            COM="Saved mass-model proxy in the common world coordinate frame. No pelvis substitution, per-shot alignment, new fitting, or measured human centre-of-mass claim.",
            speed="Magnitude of the native-clock gradient of the stored COM XYZ. Segment endpoints use one-sided derivatives.",
            acceleration="Magnitude of the original three-sample source-clock acceleration operator. Each continuous native run excludes its two endpoints.",
            camera_cuts="Cuts annotate camera changes. With consecutive source PTS the native derivative is retained across the cut, and a straddling stencil is explicitly flagged.",
            peaks="Native local maxima by magnitude within each interval. Earliest row represents a plateau. Overlapping source-ledger episodes are all listed.",
            peak_scope="Native exposure centres only; these peaks do not bound the continuous curve between exposures.",
        ),
        candidates=results, plots={key: spec[0] for key, spec in plots.items()}, peak_table="peaks.csv",
    )
    (args.output/"summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(args.output/"summary.json")


if __name__ == "__main__":
    main()
