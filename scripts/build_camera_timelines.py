"""Build target-ice timelines from rink_scope.json; preserve historical timelines."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rinkspace.camera_timeline import SOURCES,build_source,dump,audit_timelines
from rinkspace.storage import AssetStore

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = AssetStore(ROOT).directory(
    'outputs/camera_classification/rink_timelines',
    'camera_classification/rink_timelines',
)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',action='append',choices=SOURCES)
    p.add_argument('--audit-only',action='store_true',help='Validate source PTS and target/ignored/pending frame allocation')
    p.add_argument('--workers',type=int,default=4,help='Adjustable CPU parallelism, not a task scope limit')
    p.add_argument('--output',default=str(DEFAULT_OUTPUT),
                   help='New timeline directory; its parent must contain rink_scope.json')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]; out=root/args.output
    if args.audit_only:
        result=audit_timelines(root,out,args.source)
        print({k:result[k] for k in ('passed','errors','total_frames','total_seconds','total_segments','total_shot_intervals','total_uncertain_interval_seconds','total_seams_recomputed')},flush=True)
        if not result['passed']:raise SystemExit(1)
        return
    sources=args.source or list(SOURCES)
    rows=[]; failures=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(build_source,root,s,out):s for s in sources}
        for f in as_completed(futures):
            s=futures[f]
            try:rows.append(f.result())
            except Exception as e:
                failures.append({'source_id':s,'error':f'{type(e).__name__}: {e}'})
                print(f'{s} FAILED {type(e).__name__}: {e}',flush=True)
            dump(out/'summary.json',{'schema_version':'rinkspace.camera-timelines-summary.v1','sources':rows,'failures':failures,
                 'expected_sources':sources,'all_requested_sources_processed':len(rows)==len(sources) and not failures})
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
