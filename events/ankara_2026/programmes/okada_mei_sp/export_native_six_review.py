"""Freeze a current six-shot problem and export its complete native-PTS review.

This entry does not optimize or update programme_state.json. Existing output
directories and optional compact parameter files are always preserved.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

from rinkspace.path_policy import guard_project_write


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DATA = Path("D:/CodexData/RinkSpace_Data")
TIME_BASE = Fraction(1, 12800)
PTS_STEP = 256


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _name(value):
    # These names also appear as quoted relative filenames in ffmpeg's list.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError("Labels and shot IDs need letters, digits, '_' or '-'")
    return value


def _source_selections(manifest, points, scope=None, *, legacy_six=True,
                       allow_sparse_observations=False):
    """Match each current shot row range to every native source exposure."""
    pts = np.asarray(points["pts"])
    if pts.ndim != 1 or pts.dtype.kind not in "iu":
        raise ValueError("Native observation PTS must be one-dimensional integers")
    shots = manifest["shots"]
    ids = [s["id"] for s in shots]
    if not shots or len(set(ids)) != len(ids):
        raise ValueError("Programme export requires distinct selected shots")
    if legacy_six and len(shots) != 6:
        raise ValueError("This programme export requires six distinct shots")
    if scope is not None:
        registered = {s["id"]: s["processing"] for s in scope["shots"]}
        required = scope["required_shot_ids"]
        if any(shot_id not in required or shot_id not in registered for shot_id in ids):
            raise ValueError("Selected shot is outside rebuild_scope.json")
        if ids != [shot_id for shot_id in required if shot_id in ids]:
            raise ValueError("Selected shots differ from registered source order")
        for shot in shots:
            expected = registered[shot["id"]]
            for key in ("first_pts", "last_pts", "stop_pts_exclusive"):
                if int(shot[key]) != int(expected[key]):
                    raise ValueError("Selected processing range differs from scope: " + shot["id"])
            frame_count = (int(expected["last_pts"]) - int(expected["first_pts"])) // PTS_STEP + 1
            if frame_count != int(expected["native_frame_count"]):
                raise ValueError("Registered native frame count differs: " + shot["id"])
    selections = []
    row_stop = 0
    for shot in shots:
        _name(shot["id"])
        first, last = int(shot["first_pts"]), int(shot["last_pts"])
        if last < first or (last - first) % PTS_STEP:
            raise ValueError("Invalid native PTS range: " + shot["id"])
        if int(shot["stop_pts_exclusive"]) != last + PTS_STEP:
            raise ValueError("Native PTS stop differs: " + shot["id"])
        selection = np.arange(first, last + PTS_STEP, PTS_STEP, dtype=np.int64)
        begin, stop = int(shot["row_begin"]), int(shot["row_stop"])
        if (begin != row_stop or stop < begin or stop > len(pts) or
                (stop == begin and not allow_sparse_observations)):
            raise ValueError("Invalid current observation rows: " + shot["id"])
        observed = np.unique(pts[begin:stop])
        if not np.isin(observed, selection).all():
            raise ValueError("Observation PTS outside registered native shot: " + shot["id"])
        if not allow_sparse_observations and not np.array_equal(observed, selection):
            raise ValueError("Incomplete current exposure coverage: " + shot["id"])
        if selections and selection[0] != selections[-1][-1] + PTS_STEP:
            raise ValueError("The current concat renderer requires contiguous shot PTS")
        selections.append(selection)
        row_stop = stop
    if row_stop != len(points["pts"]):
        raise ValueError("Observations exist outside the selected shots")
    return selections


def _observation_coverage(manifest, points, selections):
    coverage = []
    for shot, selection in zip(manifest["shots"], selections):
        observed = np.unique(points["pts"][shot["row_begin"]:shot["row_stop"]])
        empty = selection[~np.isin(selection, observed)]
        coverage.append(dict(shot=shot["id"], frames=len(selection),
            frames_with_source_observations=len(selection)-len(empty),
            frames_without_source_observations=len(empty), empty_source_pts=empty.tolist()))
    return coverage


def _manifest_for_library(manifest, native_library=None):
    """Select only an explicitly named library; numeric bindings stay untouched."""
    selected = dict(manifest)
    if native_library is not None:
        path = Path(native_library).resolve(strict=True)
        selected.update(native_library=str(path), native_library_sha256=_digest(path))
    return selected


def _verify_native_library_binding(manifest, state):
    """A prepared library must never silently replace the state's solver build."""
    expected = str(state.get("native_library_provenance_sha256", ""))
    if not expected:
        raise ValueError("State lacks native_library_provenance_sha256")
    if expected != manifest["native_library_sha256"]:
        raise ValueError("State native-library provenance differs from selected renderer DLL; "
                         "pass --native-library with the state's actual bound DLL")


def _review_names(label, *, legacy_six):
    """Keep historical six-shot products stable; generic exports carry no count."""
    _name(label)
    return ("concat_six.txt", label + "_six_complete.mp4") if legacy_six else (
        "concat_selected.txt", label + "_complete.mp4")


def export(run_root, state_path, output, label, compact_parameters=None, *, legacy_six=True,
           allow_sparse_observations=False, native_library=None):
    import prepare_native_multicamera as prepared
    from rinkspace.domain.parameters import SceneParameterModel

    root = Path(run_root).resolve(strict=True)
    state_path = Path(state_path).resolve(strict=True)
    out = guard_project_write(output)
    label = _name(label)
    if not out.is_relative_to(DATA.resolve()):
        raise ValueError("Large review outputs must stay inside RinkSpace_Data")
    if out.exists():
        raise FileExistsError("Preserve existing review output: " + str(out))
    compact = None
    if compact_parameters is not None:
        compact = guard_project_write(compact_parameters)
        if compact.suffix.lower() != ".npz":
            raise ValueError("Compact parameter output must end in .npz")
        if compact.exists():
            raise FileExistsError("Preserve existing compact parameters: " + str(compact))

    prepared_manifest = _read(root / "joint_manifest.json")
    manifest = _manifest_for_library(prepared_manifest, native_library)
    mapping = _read(root / "mapping.json")
    scope = _read(HERE / "rebuild_scope.json")
    clock = scope["source_clock"]
    if (Fraction(clock["time_base_num"], clock["time_base_den"]) != TIME_BASE or
            clock["native_pts_step"] != PTS_STEP or
            Fraction(clock["native_fps_num"], clock["native_fps_den"]) != 50):
        raise ValueError("Current renderer requires the registered 50 fps, TB 1/12800 source")
    concat_name, video_name = _review_names(label, legacy_six=legacy_six)
    frozen = out / "frozen_problem"
    frozen_state = out / "frozen_state.npz"
    for filename in ("render_snapshot.json", "render_console.log", concat_name,
                     "render_progress.jsonl", label + "_videos.json",
                     video_name):
        guard_project_write(out / filename)
    for shot in manifest["shots"]:
        shot_id = _name(shot["id"])
        for filename in (f"{label}_{shot_id}.mp4", f"{label}_{shot_id}_review.json",
                         f"{label}_{shot_id}_{shot['last_pts']}.png"):
            guard_project_write(out / filename)
    for filename in ("joint_manifest.json", "mapping.json", "point_observations.npz",
                     "prepared_joint_manifest.json"):
        guard_project_write(frozen / filename)
    guard_project_write(frozen_state)

    # The renderer consumes these copies, rather than re-reading the parent.
    out.mkdir(parents=True, exist_ok=False)
    frozen.mkdir()
    shutil.copyfile(root / "joint_manifest.json", frozen / "prepared_joint_manifest.json")
    shutil.copyfile(root / "mapping.json", frozen / "mapping.json")
    if (_read(frozen / "prepared_joint_manifest.json") != prepared_manifest or
            _read(frozen / "mapping.json") != mapping):
        raise ValueError("Current manifest or mapping changed while freezing")
    if native_library is None:
        shutil.copyfile(frozen / "prepared_joint_manifest.json", frozen / "joint_manifest.json")
    else:
        _write(frozen / "joint_manifest.json", manifest)
    try:
        os.link(root / "point_observations.npz", frozen / "point_observations.npz")
    except PermissionError:
        shutil.copyfile(root / "point_observations.npz", frozen / "point_observations.npz")
    # Checkpoints can be replaced or rewritten later: the state must be a copy.
    shutil.copyfile(state_path, frozen_state)
    points = prepared.npz(frozen / "point_observations.npz")
    state = prepared.npz(frozen_state)
    _verify_native_library_binding(manifest, state)
    for path, expected in ((frozen / "point_observations.npz", manifest["point_input_sha256"]),
                           (Path(manifest["scene_path"]), manifest["scene_file_sha256"]),
                           (Path(manifest["native_library"]), manifest["native_library_sha256"])):
        if _digest(path) != expected:
            raise ValueError("Current render dependency changed: " + str(path))
    for key in ("point_input_sha256", "input_sha256"):
        if key in mapping and mapping[key] != manifest["point_input_sha256"]:
            raise ValueError("Mapping point identity differs: " + key)
    model = SceneParameterModel(_read(manifest["scene_path"]), Path(manifest["scene_path"]),
                                optimize_scene=True)
    if state["rig"].ndim == 2:
        prepared.validate_snapshot(state, model.initial_state.layout, mapping, manifest, points)
    else:
        prepared.old.validate_joint_snapshot(state, model.initial_state.layout, mapping, manifest, points)
    selections = _source_selections(manifest, points, None if legacy_six else scope,
                                    legacy_six=legacy_six,
                                    allow_sparse_observations=allow_sparse_observations)
    expected_pts = np.concatenate(selections)
    observation_coverage = _observation_coverage(manifest, points, selections)
    record = dict(
        schema="rinkspace.native_six_review.v1" if legacy_six else "rinkspace.native_review.v1",
        status="frozen_ready_to_render", label=label,
        selected_shot_ids=[s["id"] for s in manifest["shots"]],
        full_programme_scope=([s["id"] for s in manifest["shots"]] == scope["required_shot_ids"]),
        scope_sha256=_digest(HERE / "rebuild_scope.json"),
        selected_run_root=str(root), selected_state=str(state_path),
        frozen_run_root=str(frozen), numeric_state=str(frozen_state),
        numeric_state_sha256=_digest(frozen_state),
        allow_sparse_observations=bool(allow_sparse_observations),
        observation_coverage=observation_coverage,
        frames_with_source_observations=sum(c['frames_with_source_observations'] for c in observation_coverage),
        frames_without_source_observations=sum(c['frames_without_source_observations'] for c in observation_coverage),
        prepared_manifest_sha256=_digest(frozen / "prepared_joint_manifest.json"),
        prepared_native_library=prepared_manifest["native_library"],
        prepared_native_library_sha256=prepared_manifest["native_library_sha256"],
        selected_native_library=manifest["native_library"],
        native_library_override_explicit=native_library is not None,
        manifest_sha256=_digest(frozen / "joint_manifest.json"),
        mapping_sha256=_digest(frozen / "mapping.json"),
        input_sha256=manifest["point_input_sha256"],
        scene_file_sha256=manifest["scene_file_sha256"],
        native_library_sha256=manifest["native_library_sha256"],
        source_time_base=dict(numerator=TIME_BASE.numerator, denominator=TIME_BASE.denominator),
        source_first_pts=int(expected_pts[0]), source_last_pts=int(expected_pts[-1]),
        source_first_seconds=float(int(expected_pts[0]) * TIME_BASE),
        source_last_seconds=float(int(expected_pts[-1]) * TIME_BASE),
        source_exposures=len(expected_pts), user_visual_acceptance=False,
        shots=[dict(shot=s["id"], first_pts=int(p[0]), last_pts=int(p[-1]), frames=len(p))
               for s, p in zip(manifest["shots"], selections)])
    if compact is not None:
        compact.parent.mkdir(parents=True, exist_ok=True)
        with compact.open("xb") as stream:
            np.savez(stream, **{k: v for k, v in state.items() if k not in ("stations", "source_rows")})
        record.update(compact_final_parameters=str(compact), compact_sha256=_digest(compact))
    report_path = out / "render_snapshot.json"
    _write(report_path, record)
    # Release the full observation/state arrays before the renderer loads them.
    del points, state, model

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = os.pathsep.join((str(REPO / "src"), str(HERE)))
    command = [sys.executable, "-B", str(REPO / "scripts/run_native_entry.py"),
               "--library", manifest["native_library"], str(HERE / "render_native_joint_shots.py"),
               "--run-root", str(frozen), "--snapshot", str(frozen_state),
               "--output", str(out), "--label", label]
    if allow_sparse_observations:
        command.append("--allow-sparse-observations")
    record.update(status="rendering", render_command=command)
    _write(report_path, record)
    try:
        with (out / "render_console.log").open("x", encoding="utf-8") as log:
            subprocess.run(command, cwd=REPO, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        reports = _read(out / (label + "_videos.json"))
        if [r["shot"] for r in reports] != [s["id"] for s in manifest["shots"]]:
            raise ValueError("Rendered shot order differs from the frozen manifest")
        for report, selection, coverage in zip(reports, selections, observation_coverage):
            if (report["frames"] != len(selection) or
                    report["verification"].get("decoded_frames") != len(selection) or
                    report["verification"].get("native_PTS_equal") is not True or
                    report["snapshot_sha256"] != record["numeric_state_sha256"] or
                    report["input_sha256"] != record["input_sha256"] or
                    report["native_library_sha256"] != record["native_library_sha256"] or
                    any(report.get(k) != coverage[k] for k in
                        ("frames_with_source_observations", "frames_without_source_observations", "empty_source_pts"))):
                raise ValueError("Rendered shot did not verify its frozen source PTS: " + report["shot"])
        concat = out / concat_name
        concat.write_text("".join(f"file '{label}_{s['id']}.mp4'\n" for s in manifest["shots"]),
                          encoding="utf-8")
        import imageio_ffmpeg
        from rinkspace.review.render import validate_video

        video = out / video_name
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-n",
                        "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy",
                        "-video_track_timescale", "12800", "-movflags", "+faststart", str(video)],
                       cwd=out, check=True)
        verification = validate_video(video, expected_pts, TIME_BASE)
        record.update(status="complete_and_native_pts_verified", combined_video=str(video),
                      combined_sha256=_digest(video), combined_verification=verification,
                      shots=[{k: r[k] for k in ("shot", "video", "frames", "verification")}
                             for r in reports])
        _write(report_path, record)
    except Exception as exc:
        record.update(status="export_failed", error=f"{type(exc).__name__}: {exc}")
        _write(report_path, record)
        raise
    print(json.dumps(dict(status=record["status"], video=str(video),
                          frames=len(expected_pts),
                          source_seconds=[record["source_first_seconds"], record["source_last_seconds"]]),
                     ensure_ascii=False), flush=True)
    return record


def main(*, legacy_six=True, description=None):
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--compact-parameters", type=Path)
    parser.add_argument("--native-library", type=Path,
                        help="Explicit renderer DLL; its SHA must match state provenance")
    if not legacy_six:
        parser.add_argument("--allow-sparse-observations", action="store_true",
                            help="Render every registered native frame, including frames with no source points")
    args = parser.parse_args()
    options = dict(legacy_six=legacy_six)
    if getattr(args, "allow_sparse_observations", False):
        options["allow_sparse_observations"] = True
    if args.native_library is not None:
        options["native_library"] = args.native_library
    export(args.run_root, args.state, args.output, args.label, args.compact_parameters, **options)


if __name__ == "__main__":
    main()
