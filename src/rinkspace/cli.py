from __future__ import annotations
import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path
import numpy as np
from . import __version__
from .io import dump_json, load_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rinkspace", description="RinkSpace tested starter utilities; see README for implemented scope.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Report environment and FFmpeg availability")
    p = sub.add_parser("serve", help="Start the local source-frame and rink-plane workbench")
    p.add_argument("--root", default="."); p.add_argument("--port", type=int, default=8765)
    p.add_argument("--open", action="store_true")
    p = sub.add_parser("probe", help="Audit metadata; packets are not decoded frames")
    p.add_argument("video"); p.add_argument("--packets", action="store_true"); p.add_argument("--out", required=True)
    p = sub.add_parser("index", help="Build complete decoded PTS index (may be slow)")
    p.add_argument("video"); p.add_argument("--out", required=True); p.add_argument("--timeout", type=float, default=1800)
    p = sub.add_parser("sample", help="Extract a source-PTS-preserving short sample")
    p.add_argument("video"); p.add_argument("--out", required=True); p.add_argument("--start", type=float, required=True)
    p.add_argument("--duration", type=float, required=True); p.add_argument("--every-n", type=int, default=1)
    p.add_argument("--width", type=int, default=960)
    p = sub.add_parser("register-window", help="Candidate-only LK/SIFT trial, with a supplied ice mask")
    p.add_argument("--samples", required=True); p.add_argument("--mask", required=True); p.add_argument("--out", required=True)
    p.add_argument("--method", choices=["adjacent-lk", "direct-sift"], default="adjacent-lk")
    p = sub.add_parser("fit-map", help="Fit supplied control points and audit separate checks; no auto-acceptance")
    p.add_argument("points_json"); p.add_argument("--out", required=True)
    p = sub.add_parser("kinematics", help="Local-polynomial derivative of pre-mapped contacts")
    p.add_argument("contacts_json"); p.add_argument("--plane", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("demo", help="Run synthetic-only geometry/kinematics/mosaic demo")
    p.add_argument("--out", default="outputs/synthetic_demo")
    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            from .web import serve
            serve(args.root, args.port, args.open)
            return 0
        if args.command == "doctor":
            packages = {}
            for name in ("numpy", "opencv-python-headless", "opencv-python", "Pillow", "pytest", "scipy", "av"):
                try: packages[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError: packages[name] = None
            result = {"rinkspace": __version__, "python": sys.version, "ffmpeg": shutil.which("ffmpeg"),
                      "ffprobe": shutil.which("ffprobe"), "packages": packages,
                      "runtime_scope": "Local CPU source-PTS workbench; heuristic masks/shots and local registration; no model weights or cloud calls"}
        elif args.command == "probe":
            from .media import probe
            result = probe(args.video, args.packets); dump_json(args.out, result)
        elif args.command == "index":
            from .media import index_frames
            result = index_frames(args.video, args.out, args.timeout)
        elif args.command == "sample":
            from .media import sample_window
            r = sample_window(args.video, args.out, args.start, args.duration, args.every_n, args.width)
            result = {"out": args.out, "samples": len(r["samples"]), "source_sha256": r["source_sha256"]}
        elif args.command == "register-window":
            from .seed import register_window
            result = register_window(args.samples, args.mask, args.out, args.method)
        elif args.command == "fit-map":
            from .geometry import fit_homography, heldout_audit
            d = load_json(args.points_json)
            h, inliers = fit_homography(d["train_src"], d["train_dst"], d.get("threshold_dst", 3.0))
            audit = heldout_audit(h, d["train_src"], d["train_ids"], d["check_src"], d["check_dst"], d["check_ids"])
            result = {"matrix": h.tolist(), "matrix_direction": "H_destination_from_source",
                      "geometry_level": "projective", "status": "candidate", "train_inliers": inliers.tolist(), "heldout": audit}
            dump_json(args.out, result)
        elif args.command == "kinematics":
            from .geometry import PlaneMap
            from .motion import kinematics, window_sensitivity
            c = load_json(args.plane)
            plane = PlaneMap(np.asarray(c["matrix"]), c["level"], c["unit"], map_id=c["map_id"])
            rows = load_json(args.contacts_json)
            result = {"kinematics": kinematics(rows, plane), "sensitivity": window_sensitivity(rows, plane)}
            dump_json(args.out, result)
        else:
            from .demo import synthetic_demo
            result = synthetic_demo(args.out)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        print(f"RinkSpace: {exc}", file=sys.stderr)
        return 2
