"""Build/validate/view the eight requested broadcasts, using existing media."""
from rinkspace.storage import resolve_asset as _resolve_asset
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rinkspace.camera_classification import build_catalog, read, validate_catalog, write, refresh_catalog_metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['scope','build','refresh','validate','view'])
    parser.add_argument('--output',type=Path,default=_resolve_asset(ROOT, ROOT/'outputs/camera_classification'))
    parser.add_argument('--workers',type=int,default=6,help='adjustable CPU parallelism, not a task limit')
    args=parser.parse_args()
    if args.command=='scope':
        from rinkspace.camera_scope import migrate_scope
        result=migrate_scope(ROOT,args.output)
        print(result['summary'])
    elif args.command=='build':
        build_catalog(ROOT,args.output,args.workers)
    elif args.command=='refresh':
        result=refresh_catalog_metadata(ROOT,args.output)
        print(result['metadata_refresh'])
        return 0 if result['structural_pass'] else 1
    elif args.command=='validate':
        result=validate_catalog(read(args.output/'catalog.json'))
        if (args.output/'validation.json').exists():
            result={**read(args.output/'validation.json'),**result}
        write(args.output/'validation.json',result)
        print(result['structural_pass'])
        return 0 if result['structural_pass'] else 1
    else:
        import subprocess
        return subprocess.call([sys.executable,str(ROOT/'scripts/serve_camera_classification.py'),'--catalog',str(args.output/'catalog.json')])
    return 0


if __name__=='__main__':
    raise SystemExit(main())
