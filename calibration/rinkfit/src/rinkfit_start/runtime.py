"""Atomic checkpoints, portable paths and live progress for one run."""
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import is_dataclass, asdict
import hashlib
import json
import os
import socket
import tempfile
import uuid
import numpy as np


def now(): return datetime.now(timezone.utc).isoformat(timespec='seconds')


def json_value(value):
    if is_dataclass(value): return json_value(asdict(value))
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,Path): return str(value)
    if isinstance(value,Mapping): return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [json_value(v) for v in value]
    return value


def canonical_json(value):
    return json.dumps(json_value(value),sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))


def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(json_value(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    fd,temp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def atomic_npz(path,arrays):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            np.savez_compressed(stream,**{k:np.array(v,copy=True) for k,v in arrays.items()})
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def validate_output_path(path):
    path=Path(path).expanduser().resolve()
    if any('onedrive' in part.lower() for part in path.parts):
        raise ValueError('Runtime arrays/logs belong under the existing external Data root, not OneDrive')
    if os.name=='nt' and len(str(path).encode('utf-16-le'))//2>=230:
        raise ValueError('Use a shorter run directory to leave room for output filenames')
    return path


class Journal:
    def __init__(self,run_dir): self.run_dir=Path(run_dir)
    def emit(self,event,**fields):
        record={'utc':now(),'event':event,**fields}
        line=canonical_json(record)
        with (self.run_dir/'progress.jsonl').open('a',encoding='utf-8') as out:
            out.write(line+'\n'); out.flush()
        atomic_json(self.run_dir/'status.json',record)
        print(line,flush=True)


@contextmanager
def run_lock(run_dir):
    """Never delete another session's lock or infer that a remote PID is stale."""
    path=Path(run_dir)/'run.lock'; token=uuid.uuid4().hex
    body=canonical_json({'pid':os.getpid(),'host':socket.gethostname(),'token':token,'utc':now()})
    try: fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError:
        raise RuntimeError(f'Run already locked: {path}. Inspect the recorded process before removing a stale lock.')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as out: out.write(body)
        yield
    finally:
        if path.exists():
            current=json.loads(path.read_text(encoding='utf-8'))
            if current.get('token')==token: path.unlink()


def save_complete_snapshot(directory,arrays,metadata):
    """Commit manifest last; incomplete snapshots are never read as complete."""
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
    frozen={k:np.array(v,copy=True) for k,v in arrays.items()}
    atomic_npz(directory/'state.npz',frozen)
    digest=hashlib.sha256((directory/'state.npz').read_bytes()).hexdigest()
    atomic_json(directory/'manifest.json',{'schema':'rinkfit.snapshot.v1','metadata':metadata,
        'state_sha256':digest,'array_shapes':{k:list(v.shape) for k,v in frozen.items()}})
    return digest


def load_complete_snapshot(directory):
    directory=Path(directory); m=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256((directory/'state.npz').read_bytes()).hexdigest()!=m['state_sha256']:
        raise ValueError('Snapshot checksum mismatch')
    with np.load(directory/'state.npz',allow_pickle=False) as z: arrays={k:z[k].copy() for k in z.files}
    return arrays,m
