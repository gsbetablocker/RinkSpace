"""Check the delivered file manifest; does not execute a solver or install anything."""
from pathlib import Path
import hashlib
import json
import sys


def main():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'package_manifest.json').read_text(encoding='utf-8'))
    failures=[]
    for item in manifest['files']:
        path=root/item['path']
        if not path.is_file():failures.append({'path':item['path'],'error':'missing'});continue
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            failures.append({'path':item['path'],'error':'checksum_changed'})
    print(json.dumps({'verified_files':len(manifest['files']),'failures':failures,
                      'physical_fit_completed':False},indent=2))
    return 2 if failures else 0


if __name__=='__main__':raise SystemExit(main())
