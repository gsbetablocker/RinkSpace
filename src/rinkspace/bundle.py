"""Package a self-contained derived report, without raw sources or caches.

The run folder must explicitly list relative files in review_manifest.json:
{"entrypoint":"index.html", "files":["index.html","comparison.mp4", ...]}.
All listed local HTML assets must exist and be included. Do not silently omit
files to meet an explicitly requested size cap; fail and name the largest files instead.
"""
from __future__ import annotations
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
from urllib.parse import unquote, urlsplit
import zipfile

ALLOWED={'.html','.css','.js','.json','.md','.txt','.png','.jpg','.jpeg','.webp','.mp4','.csv','.npy','.log'}
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links.extend(v for k,v in attrs if k in {'src','href','poster'} and v)


def make_bundle(run:Path,destination:Path,limit_mib:float | None=None) -> dict:
    run=run.resolve(strict=True)
    if not destination.parent.exists():destination.parent.mkdir(parents=True)
    if destination.exists():raise ValueError(f'Refusing to overwrite: {destination}')
    if limit_mib is not None and limit_mib<=0:raise ValueError('Size limit must be positive')
    manifest=json.loads((run/'review_manifest.json').read_text(encoding='utf-8'))
    names=set(manifest['files'])|{'review_manifest.json'}
    if manifest['entrypoint'] not in names:raise ValueError('Entry point is not listed')
    paths={}
    for name in names:
        if not isinstance(name,str) or '\\' in name:raise ValueError('Use relative POSIX file names')
        p=Path(name)
        if p.is_absolute() or '..' in p.parts:raise ValueError(f'Unsafe path: {name}')
        q=(run/p).resolve(strict=True)
        if not q.is_relative_to(run):raise ValueError(f'Escaping symlink: {name}')
        if q.suffix.lower() not in ALLOWED or not q.is_file():raise ValueError(f'Unsupported report asset: {name}')
        paths[name]=q
    for name,q in paths.items():
        if q.suffix.lower()!='.html':continue
        h=Links();h.feed(q.read_text(encoding='utf-8'))
        for link in h.links:
            u=urlsplit(link)
            if u.scheme or u.netloc or not u.path:continue
            target=(q.parent/unquote(u.path)).resolve()
            if not target.is_relative_to(run) or str(target.relative_to(run)) .replace('\\','/') not in names:
                raise ValueError(f'Unlisted local HTML dependency: {name} -> {link}')
    hashes={name:hashlib.sha256(q.read_bytes()).hexdigest() for name,q in sorted(paths.items())}
    tmp_path=None
    try:
        with tempfile.NamedTemporaryFile(suffix='.zip',dir=destination.parent,delete=False) as f:tmp_path=Path(f.name)
        with zipfile.ZipFile(tmp_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for name,q in sorted(paths.items()):z.write(q,name)
            z.writestr('BUNDLE_CHECKSUMS.json',json.dumps(hashes,indent=2)+'\n')
        size=tmp_path.stat().st_size
        if limit_mib is not None and size>limit_mib*1024**2:
            largest=sorted(((n,q.stat().st_size) for n,q in paths.items()),key=lambda x:-x[1])[:5]
            raise ValueError(f'Bundle exceeds {limit_mib} MiB ({size} bytes). Largest assets: {largest}. Adjust the explicit cap or preview encoding without omitting evidence.')
        tmp_path.replace(destination);tmp_path=None
        return {'path':str(destination),'size_bytes':size,'file_count':len(paths)+1,'entrypoint':manifest['entrypoint'],'limit_mib':limit_mib}
    finally:
        if tmp_path is not None:tmp_path.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--limit-mib',type=float,default=None,help='Optional archive size limit in MiB')
    a=p.parse_args()
    try:print(json.dumps(make_bundle(a.run,a.out,a.limit_mib),ensure_ascii=False,indent=2))
    except (ValueError,OSError,KeyError) as e:p.exit(2,f'{e}\n')
