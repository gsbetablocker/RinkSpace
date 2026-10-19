from __future__ import annotations
import argparse
import json
from pathlib import Path
from .io_utils import read_json, write_json, read_jsonl


def main():
    bundle=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description="三机位九镜头接入与审查；生产求解由 Codex 按 CODEX_START.md 接入")
    parser.add_argument("--bundle",type=Path,default=bundle,help="Immutable joint package data root")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare",help="准备全部数据、机位图、材料交叉表、源审查和8434帧任务")
    p.add_argument("--out",required=True);p.add_argument("--normalized")
    p.add_argument("--video");p.add_argument("--skip-video-verification",action="store_true")
    p.add_argument("--control-dt",type=float,default=.4)
    p=sub.add_parser("ingest",help="重新读取已展开的原观测包（默认prepare直接用随包全量规范化数据）")
    p.add_argument("--source-package",required=True);p.add_argument("--out",required=True)
    p=sub.add_parser("verify-video");p.add_argument("--video",required=True)
    p.add_argument("--scope",required=True);p.add_argument("--out",required=True)
    p=sub.add_parser("decode-selected");p.add_argument("--video",required=True)
    p.add_argument("--frames",default=None)
    p.add_argument("--video-audit",required=True);p.add_argument("--out",required=True)
    p=sub.add_parser("screen-sources");p.add_argument("--normalized",default=None)
    p.add_argument("--native-frames",required=True);p.add_argument("--out",required=True)
    p=sub.add_parser("check-graph");p.add_argument("--graph",required=True);p.add_argument("--require-ready",action="store_true")
    p=sub.add_parser("inspect-shot1");p.add_argument("--checkpoint",required=True)
    p=sub.add_parser("export-shot1-dependencies")
    p.add_argument("--checkpoint",required=True);p.add_argument("--observations",required=True)
    p.add_argument("--semantic-manifest",required=True);p.add_argument("--engineering-budget",required=True)
    p.add_argument("--source-clock",required=True);p.add_argument("--out",required=True)
    p.add_argument("--asset-root",action="append",default=[])
    p=sub.add_parser("import-shot1");p.add_argument("--checkpoint",required=True)
    p.add_argument("--dependencies",required=True);p.add_argument("--graph",required=True);p.add_argument("--out",required=True)
    p.add_argument("--normalized",default=None)
    p=sub.add_parser("feedback-run",help="Run a prepared native continuation with mandatory checkpoint feedback")
    p.add_argument("--launch",required=True);p.add_argument("--inspect",action="store_true")
    p=sub.add_parser("status");p.add_argument("--run",required=True)
    args=parser.parse_args()
    bundle=args.bundle.resolve()
    if args.command=="decode-selected" and args.frames is None: args.frames=str(bundle/"data/prepared/frames.jsonl.gz")
    if args.command in {"screen-sources","import-shot1"} and args.normalized is None: args.normalized=str(bundle/"data/prepared")
    if args.command=="prepare":
        from .pipeline import prepare
        result=prepare(bundle,args.out,normalized=args.normalized,video=args.video,
                       verify_source=not args.skip_video_verification,control_dt=args.control_dt)
    elif args.command=="ingest":
        from .data_ingest import normalize_dataset
        result=normalize_dataset(args.source_package,args.out,verify_hashes=True)
    elif args.command=="verify-video":
        from .timeline import verify_video
        result=verify_video(args.video,read_json(args.scope));write_json(args.out,result)
    elif args.command=="decode-selected":
        from .timeline import decode_selected
        result=decode_selected(args.video,list(read_jsonl(args.frames)),args.out,read_json(args.video_audit))
    elif args.command=="screen-sources":
        from .source_screen import screen_sources
        result=screen_sources(args.normalized,args.native_frames,args.out)
    elif args.command=="check-graph":
        from .camera_graph import CameraGraph
        result=CameraGraph.from_json(args.graph).validate(require_ready=args.require_ready)
    elif args.command=="inspect-shot1":
        from .checkpoint_bridge import inspect_checkpoint
        result=inspect_checkpoint(args.checkpoint)
    elif args.command=="export-shot1-dependencies":
        from .checkpoint_bridge import export_dependencies
        result=export_dependencies(args.checkpoint,args.observations,args.semantic_manifest,
            args.engineering_budget,args.source_clock,args.out,asset_roots=args.asset_root)
    elif args.command=="import-shot1":
        from .checkpoint_bridge import snapshot_checkpoint
        result=snapshot_checkpoint(args.checkpoint,args.out,dependency_manifest=args.dependencies,graph=args.graph)
        # The latest repair can change material topology/indices. Rebuild the
        # crosswalk against THAT snapshot, not against bundled i20 indices.
        from .materials import build_registry
        from .io_utils import file_sha256
        model=Path(args.out)/"checkpoint/model.json"
        registry=build_registry(list(read_jsonl(Path(args.normalized)/"observations.jsonl.gz")),
                                read_json(model),file_sha256(model))
        write_json(Path(args.out)/"joint_material_registry.json",registry)
        result["new_joint_material_registry"]="joint_material_registry.json"
        result["recompile_required"]=["operator bindings","dependency recipes","numeric W membership",
                                      "spline support","Jacobians","all source residuals"]
        write_json(Path(args.out)/"joint_rebase_receipt.json",result)
    elif args.command=="feedback-run":
        from .feedback_launch import inspect, run
        result=inspect(args.launch) if args.inspect else run(args.launch)
    else:
        from .pipeline import status
        result=status(args.run)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
