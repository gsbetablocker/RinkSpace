"""Lossless disk storage for full source state, without removing observations.

Overlay records preserve Python/NumPy values exactly. Numeric graph arrays are
read-only file mappings; labels, source links and temporal proofs remain live.
Every source frame is retained. Files belong to this replay's output directory.
"""
from __future__ import annotations

from collections.abc import MutableMapping
import hashlib
import json
from pathlib import Path
import pickle
import sqlite3
import zlib

import numpy as np


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


class ExactSourceStateSpill(MutableMapping):
    """Archive source corrections on disk and map full numeric graph state."""
    def __init__(self,directory):
        self.directory=Path(directory).resolve()
        self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/'source_corrections.sqlite'
        if self.path.exists():
            raise FileExistsError('Preserve an earlier source state spill')
        self.connection=sqlite3.connect(self.path)
        self.connection.execute('CREATE TABLE corrections (generation INTEGER, pts INTEGER, payload BLOB, sha256 TEXT, PRIMARY KEY(generation,pts))')
        self.connection.execute('CREATE TABLE graph_arrays (pts INTEGER PRIMARY KEY, filename TEXT, metadata TEXT, sha256 TEXT)')
        self.connection.commit()
        self.generation=0
        self.sealed=False

    def __setitem__(self,key,value):
        if self.sealed:
            raise ValueError('Source spill is already sealed')
        if type(key) is not int:
            raise ValueError('Exact source exposure integer required')
        # This private cache reads only values produced by this running reader.
        # The compressed payload has a separate byte identity before unpickling.
        payload=zlib.compress(pickle.dumps(value,protocol=5),1)
        self.connection.execute('INSERT INTO corrections VALUES (?,?,?,?)',
            (self.generation,key,payload,hashlib.sha256(payload).hexdigest()))
        self.connection.commit()

    def __getitem__(self,key):
        record=self.connection.execute('SELECT payload,sha256 FROM corrections WHERE generation=? AND pts=?',
            (self.generation,int(key))).fetchone()
        if record is None:
            raise KeyError(key)
        payload,digest=record
        if hashlib.sha256(payload).hexdigest()!=digest:
            raise ValueError('Lossless source overlay cache bytes changed')
        return pickle.loads(zlib.decompress(payload))

    def __iter__(self):
        # A cursor yields one source key at a time, with no full frame bank.
        return (int(row[0]) for row in self.connection.execute(
            'SELECT pts FROM corrections WHERE generation=? ORDER BY rowid',(self.generation,)))

    def __len__(self):
        return int(self.connection.execute('SELECT count(*) FROM corrections WHERE generation=?',
            (self.generation,)).fetchone()[0])

    def __delitem__(self,key):
        raise ValueError('Source evidence is archived, never deleted through the mapping')

    def clear(self):
        # The caller advances to another shot. Previous source state stays on
        # disk; only the active lookup namespace changes.
        self.generation+=1

    def spill_graph_frame(self,frame):
        if self.sealed:
            raise ValueError('Source spill is already sealed')
        pts=int(frame['pts'])
        filename=f'graph_numeric_{pts}.bin'
        path=self.directory/filename
        fields=('ids','query_ids','uv','sides','normal','width','curve','descriptor','raw_uv','quantiles')
        metadata={};offset=0
        with path.open('xb') as stream:
            for name in fields:
                if name not in frame:
                    continue
                value=np.ascontiguousarray(frame[name])
                if value.dtype.hasobject:
                    raise ValueError('Graph source pixels require explicit numeric array dtype')
                padding=(-offset)%64
                if padding:
                    stream.write(bytes(padding));offset+=padding
                metadata[name]=dict(dtype=value.dtype.str,shape=list(value.shape),offset=offset,nbytes=value.nbytes)
                if value.nbytes:
                    stream.write(memoryview(value).cast('B'))
                offset+=value.nbytes
        digest=_sha(path)
        self.connection.execute('INSERT INTO graph_arrays VALUES (?,?,?,?)',
            (pts,filename,json.dumps(metadata,separators=(',',':')),digest))
        self.connection.commit()
        for name,item in metadata.items():
            if item['nbytes']:
                frame[name]=np.memmap(path,dtype=np.dtype(item['dtype']),mode='r',
                    offset=item['offset'],shape=tuple(item['shape']))
            else:
                frame[name]=np.empty(item['shape'],dtype=np.dtype(item['dtype']))
                frame[name].flags.writeable=False
        return dict(actual_pts=pts,path=str(path),sha256=digest,
            numeric_source_bytes=offset,points_removed=0)

    def seal(self):
        if self.sealed:
            raise ValueError('Preserve the first sealed source state receipt')
        frames=[]
        for pts,filename,metadata,digest in self.connection.execute('SELECT * FROM graph_arrays ORDER BY pts'):
            path=self.directory/filename
            if _sha(path)!=digest:
                raise ValueError('Mapped source graph bytes changed during replay')
            frames.append(dict(actual_pts=int(pts),path=str(path),sha256=digest,
                fields=json.loads(metadata)))
        for payload,digest in self.connection.execute('SELECT payload,sha256 FROM corrections'):
            if hashlib.sha256(payload).hexdigest()!=digest:
                raise ValueError('Source overlay cache bytes changed during replay')
        count=self.connection.execute('SELECT count(*) FROM corrections').fetchone()[0]
        self.connection.commit();self.connection.close();self.sealed=True
        receipt=self.directory/'source_state_spill_receipt.json'
        result=dict(schema='rinkspace.lossless-source-state-spill.v1',
            correction_database=dict(path=str(self.path),sha256=_sha(self.path)),
            correction_exposures=int(count),graph_numeric_frames=frames,
            source_numeric_arrays_and_dtypes_preserved=True,source_points_removed=0,
            temporal_graph_links_or_labels_removed=False,source_RGB_library_created=False,
            solver_started=False)
        with receipt.open('x',encoding='utf8') as stream:
            json.dump(result,stream,ensure_ascii=False,separators=(',',':'))
        return dict(path=str(receipt),sha256=_sha(receipt))
