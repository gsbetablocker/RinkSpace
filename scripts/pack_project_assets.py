"""Freeze explicit project directories into one byte-exact compact asset bundle.

Only packages originals. Relocation is a separate reviewed action after readers
and the registry have been checked. No solver or media process is launched.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rinkspace.path_policy import check_project_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,action='append',required=True)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--index',type=Path,required=True)
    p.add_argument('--staging',type=Path,required=True)
    a=p.parse_args()
    archive,index,staging=a.archive.resolve(),a.index.resolve(),a.staging.resolve()
    for target in (archive,index):
        if not target.is_relative_to(ROOT) or check_project_path(target,project_root=ROOT):
            raise ValueError('Invalid project output path')
        if target.exists():raise FileExistsError(target)
    if staging.is_relative_to(ROOT) or staging.exists():raise ValueError('Use a new external staging file')
    sources=[]
    for directory in a.directory:
        directory=directory.resolve()
        if not directory.is_relative_to(ROOT) or not directory.is_dir() or directory.is_symlink():
            raise ValueError('Select explicit ordinary project directories')
        for source in sorted(directory.rglob('*')):
            if source.is_symlink():raise ValueError('Links cannot enter the bundle')
            if source.is_file():sources.append(source)
    if len(set(sources))!=len(sources):raise ValueError('Overlapping input directories')
    staging.parent.mkdir(parents=True,exist_ok=True)
    rows=[];seen={}
    with zipfile.ZipFile(staging,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for ordinal,source in enumerate(sources):
            before=source.stat();value=source.read_bytes()
            if (before.st_size,before.st_mtime_ns)!=(source.stat().st_size,source.stat().st_mtime_ns):
                raise ValueError('Input changed during packing')
            digest=hashlib.sha256(value).hexdigest()
            member=seen.get(digest)
            if member is None:
                member='files/'+digest+source.suffix
                z.writestr(member,value);seen[digest]=member
            rows.append(dict(path=source.relative_to(ROOT).as_posix(),member=member,bytes=len(value),sha256=digest))
    if staging.stat().st_size>100*1024*1024:raise ValueError('Split selection to stay within GitHub ordinary-file limit')
    doc=dict(schema='rinkspace.packed-assets.v1',archive=archive.relative_to(ROOT).as_posix(),
             source_bytes_unchanged=True,original_directories=[str(d.resolve().relative_to(ROOT)) for d in a.directory],assets=rows)
    archive.parent.mkdir(parents=True,exist_ok=True);index.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(staging,archive)
    index.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(files=len(rows),raw_bytes=sum(r['bytes'] for r in rows),archive_bytes=archive.stat().st_size,index_bytes=index.stat().st_size),ensure_ascii=False))


if __name__=='__main__':main()
