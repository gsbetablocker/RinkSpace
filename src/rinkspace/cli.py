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
    p = sub.add_parser("model", help="Native-core structure tools; this slice does not run fitting")
    model_sub = p.add_subparsers(dest="model_command", required=True)
    m = model_sub.add_parser("import-d78", help="One-time unfitted structural seed import; original files unchanged")
    m.add_argument("--model", required=True); m.add_argument("--relations", required=True); m.add_argument("--out", required=True)
    m = model_sub.add_parser("preview", help="Render explicitly incomplete structure through the built native core")
    m.add_argument("--scene", required=True); m.add_argument("--library", required=True); m.add_argument("--out", required=True)
    m = model_sub.add_parser("import-state", help="Import complete frozen d78 state; no fit")
    m.add_argument("--model", required=True); m.add_argument("--complete-state"); m.add_argument("--scope", required=True)
    m.add_argument("--library", required=True); m.add_argument("--out", required=True)
    p = sub.add_parser("prepare", help="Compile current immutable observations and scope")
    p.add_argument("--input", required=True); p.add_argument("--scope", required=True)
    p.add_argument("--shots", required=True); p.add_argument("--out", required=True)
    p.add_argument("--library", help="Current native core; required for ice/board fixed junctions")
    p = sub.add_parser("review", help="Complete native-PTS replay using the same compiled forward; no fit")
    p.add_argument("--prepared", required=True); p.add_argument("--state", required=True); p.add_argument("--assets", required=True)
    p.add_argument("--shots", required=True); p.add_argument("--library", required=True); p.add_argument("--out", required=True)
    p.add_argument("--compare")
    m = model_sub.add_parser("preview-curling", help="Native curling/Lane structure with explicit drawing-support semantics; no fit")
    m.add_argument("--scene", required=True); m.add_argument("--library", required=True); m.add_argument("--out", required=True)
    m = model_sub.add_parser("preview-field", help="Native M1-M5 field geometry review; no fit")
    m.add_argument("--scene", required=True); m.add_argument("--library", required=True); m.add_argument("--out", required=True)
    m = model_sub.add_parser("import-ad-assets", help="Import explicitly selected existing advertisement source records; no fit")
    m.add_argument("--input", required=True); m.add_argument("--out", required=True)
    m = model_sub.add_parser("preview-ad-assets", help="Review original advertisement pixels and native-sampled templates; no fit")
    m.add_argument("--assets", required=True); m.add_argument("--library", required=True); m.add_argument("--out", required=True)
    m = model_sub.add_parser("rebuild-glyphs", help="Rebuild user-confirmed printed text from native source pixels; no scene fit")
    m.add_argument("--input", required=True); m.add_argument("--library", required=True); m.add_argument("--out", required=True)
    m = model_sub.add_parser("preview-print-installation", help="Same native print, continuous board and motion camera; no fit")
    m.add_argument("--scene", required=True); m.add_argument("--assets", required=True); m.add_argument("--instance", required=True)
    m.add_argument("--camera", required=True)
    frame_input = m.add_mutually_exclusive_group(required=True)
    frame_input.add_argument("--frame", help="Unverified image preview; not a source-PTS validation")
    frame_input.add_argument("--source-video", help="Decode the registered video at its native PTS")
    m.add_argument("--scope", help="Current source/shot scope, required with --source-video")
    m.add_argument("--pts", required=True, type=int)
    m.add_argument("--library", required=True); m.add_argument("--out", required=True)
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
        if args.command == "model":
            if args.model_command == "import-d78":
                from .compat.d78_geometry import import_d78
                result = import_d78(args.model, args.relations, args.out)
            elif args.model_command == "import-ad-assets":
                from .session.ad_assets import import_assets
                result = import_assets(args.input, args.out)
            elif args.model_command == "preview-ad-assets":
                from .session.ad_review import render
                result = render(args.assets, args.library, args.out)
            elif args.model_command == "rebuild-glyphs":
                from .session.glyph_rebuild import rebuild
                result = rebuild(args.input, args.library, args.out)
            elif args.model_command == "preview-print-installation":
                from .session.print_installation import render
                result = render(args.scene,args.assets,args.instance,args.camera,args.frame,args.pts,args.library,args.out,
                                source_video_path=args.source_video,scope_path=args.scope)
            elif args.model_command == "preview-field":
                from .session.field_preview import render
                result = render(args.scene, args.library, args.out)
            elif args.model_command == "preview-curling":
                from .session.curling_preview import render
                result = render(args.scene, args.library, args.out)
            elif args.model_command == "import-state":
                from .compat.d78_state import import_d78_state
                path = import_d78_state(args.model, args.out, args.scope, args.library, args.complete_state)
                result = {"state_path": str(path), "fit_executed": False, "purpose": "frozen_d78_replay"}
            else:
                from .session.core_preview import preview
                result = preview(args.scene, args.library, args.out)
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        if args.command == "prepare":
            from dataclasses import asdict
            from .session.prepare import prepare
            from .core.native import NativeCore
            core = NativeCore(args.library) if args.library else None
            result = asdict(prepare(args.input, args.scope, args.shots, args.out, core=core))
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        if args.command == "review":
            from .review.render import render
            receipt = render(args.prepared, args.state, args.assets, args.shots, args.library, args.out, args.compare)
            print(json.dumps({"review_path": receipt.review_path, "scope_status": receipt.review["scope_status"],
                              "fit_executed": False, "artifacts": receipt.review["artifacts"]}, ensure_ascii=False, indent=2))
            return 0
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
