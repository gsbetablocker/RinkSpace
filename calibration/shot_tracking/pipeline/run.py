"""Run one stage of the current 121-frame, anchor +/-1 second trial.

The stage boundary keeps the original-source observations, scene-guided
contours, camera solve, evidence checks, and visual review rerunnable.
All generated files stay under calibration/shot_tracking/runs/.

Example:
  .venv/Scripts/python.exe -B calibration/shot_tracking/pipeline/run.py \
      --run calibration/shot_tracking/runs/low_end_pm1 --stage contours --stride 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
PIPELINE = Path(__file__).resolve().parent
RUNS = ROOT / "calibration" / "shot_tracking" / "runs"
DEFAULT_RUN = RUNS / "low_end_pm1"
DEFAULT_SCENE = ROOT / "calibration" / "deliverables" / "scene.json"
DEFAULT_SOURCE = ROOT / "data" / "input" / "oxXu1avhrvs.mp4"


def _inside_runs(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(RUNS.resolve()) or resolved == RUNS.resolve():
        raise ValueError(f"Run outputs must stay inside {RUNS}")
    return resolved


def _require(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing prerequisite: " + ", ".join(missing))


def _script(name: str, *args: object) -> None:
    command = [sys.executable, "-B", str(PIPELINE / name), *(str(a) for a in args)]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--stage", required=True,
                        choices=("observations", "contours", "solve", "check", "review"))
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Original broadcast video, used by observations")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--stride", type=int, default=10,
                        help="Original-source contour reading interval; contours stage only")
    parser.add_argument("--readings", type=Path,
                        help="Endpoint source reading JSON (not held-out validation); check stage only")
    parser.add_argument("--max-nfev", type=int, default=120,
                        help="Joint solver evaluation limit; solve stage only")
    parser.add_argument("--resume", action="store_true",
                        help="Resume joint solve from its checkpoint")
    parser.add_argument("--preview-only", action="store_true",
                        help="At review stage, generate only first/anchor/last contact sheets")
    args = parser.parse_args()
    run = _inside_runs(args.run)
    scene = args.scene.resolve()
    if not 1 <= args.stride <= 60:
        parser.error("--stride must be between 1 and 60 frames")
    if args.max_nfev < 1:
        parser.error("--max-nfev must be positive")
    if args.resume and args.stage != "solve":
        parser.error("--resume applies only to --stage solve")
    if args.preview_only and args.stage != "review":
        parser.error("--preview-only applies only to --stage review")
    if args.stage in ("contours", "solve") and scene != DEFAULT_SCENE.resolve():
        parser.error("contours and solve currently use the frozen three-frame scene.json")

    if args.stage == "observations":
        _require(args.source)
        run.mkdir(parents=True, exist_ok=True)
        _script("observations.py", "--source", args.source.resolve(), "--run-dir", run)
    elif args.stage == "contours":
        _require(scene, run / "source.json", run / "observations.npz",
                 run / "pair_tracks.npz", run / "anchor_direct.npz")
        # The source-pixel motion is a search guide, never a physical camera.
        import image_motion
        import contours
        image_motion.run(run)
        contours.run(args.stride, run)
    elif args.stage == "solve":
        _require(scene, run / "observations.npz", run / "pair_tracks.npz",
                 run / "contours.json")
        command: list[object] = ["--run", run, "--max-nfev", args.max_nfev]
        if args.resume:
            command.append("--resume")
        _script("joint.py", *command)
    elif args.stage == "check":
        readings = args.readings.resolve() if args.readings else run / "agent_readings.json"
        _require(scene, run / "poses.json", readings)
        _script("check.py", "--poses", run / "poses.json", "--scene", scene,
                "--readings", readings, "--output", run / "checks.json")
    else:
        _require(scene, run / "poses.json", run / "source.mp4",
                 run / "source.json", run / "observations.npz")
        command = ["--run", run, "--scene", scene]
        if args.preview_only:
            command.append("--preview-only")
        _script("review.py", *command)

    print(json.dumps({"stage": args.stage, "run": str(run)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
